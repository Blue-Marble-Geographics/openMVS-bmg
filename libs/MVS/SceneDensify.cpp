/*
* SceneDensify.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

//#include <windows.h>
#include "Common.h"
#include "Scene.h"
#include "SceneDensify.h"
#include "PatchMatchCUDA.h"
// MRF: view selection
#include "../Math/TRWS/MRFEnergy.h"

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif

// S T R U C T S ///////////////////////////////////////////////////

// Dense3D data.events
enum EVENT_TYPE {
	EVT_FAIL = 0,
	EVT_CLOSE,

	EVT_PROCESSIMAGE,

	EVT_ESTIMATEDEPTHMAP,
	EVT_OPTIMIZEDEPTHMAP,
	EVT_SAVEDEPTHMAP,

	EVT_FILTERDEPTHMAP,
	EVT_ADJUSTDEPTHMAP,
};

class EVTFail : public Event
{
public:
	EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
	IIndex idxImage;
	EVTProcessImage(IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTEstimateDepthMap(IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTOptimizeDepthMap(IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTSaveDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTSaveDepthMap(IIndex _idxImage) : Event(EVT_SAVEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTFilterDepthMap(IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};
class EVTAdjustDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTAdjustDepthMap(IIndex _idxImage) : Event(EVT_ADJUSTDEPTHMAP), idxImage(_idxImage) {}
};
/*----------------------------------------------------------------*/


// convert the ZNCC score to a weight used to average the fused points
inline float Conf2Weight(float conf, Depth depth) {
	return 1.f/(FastMaxS(1.f-conf,0.03f)*depth*depth);
}
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////


DepthMapsData::DepthMapsData(Scene& _scene)
	:
	scene(_scene),
	arrDepthData(_scene.images.GetSize())
{
} // constructor

DepthMapsData::~DepthMapsData()
{
} // destructor
/*----------------------------------------------------------------*/

std::atomic<double> ttotal = 0.;

// globally choose the best target view for each image,
// trying in the same time the selected image pairs to cover the whole scene;
// the map of selected neighbors for each image is returned in neighborsMap.
// For each view a list of neighbor views ordered by number of shared sparse points and overlapped image area is given.
// Next a graph is formed such that the vertices are the views and two vertices are connected by an edge if the two views have each other as neighbors.
// For each vertex, a list of possible labels is created using the list of neighbor views and scored accordingly (the score is normalized by the average score).
// For each existing edge, the score is defined such that pairing the same two views for any two vertices is discouraged (a constant high penalty is applied for such edges).
// This primal-dual defined problem, even if NP hard, can be solved by a Belief Propagation like algorithm, obtaining in general a solution close enough to optimality.
bool DepthMapsData::SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap)
{
	// find all pair of images valid for dense reconstruction
	typedef std::unordered_map<uint64_t,float> PairAreaMap;
	PairAreaMap edges;
	double totScore(0);
	unsigned numScores(0);
	FOREACH(i, images) {
		const IIndex idx(images[i]);
		ASSERT(imagesMap[idx] != NO_ID);
		const ViewScoreArr& neighbors(arrDepthData[idx].neighbors);
		ASSERT(neighbors.GetSize() <= OPTDENSE::nMaxViews);
		// register edges
		FOREACHPTR(pNeighbor, neighbors) {
			const IIndex idx2(pNeighbor->ID);
			ASSERT(imagesMap[idx2] != NO_ID);
			edges[MakePairIdx(idx,idx2)] = pNeighbor->area;
			totScore += pNeighbor->score;
			++numScores;
		}
	}
	if (edges.empty())
		return false;
	const float avgScore((float)(totScore/(double)numScores));

	// run global optimization
	const float fPairwiseMul = OPTDENSE::fPairwiseMul; // default 0.3
	const float fEmptyUnaryMult = 6.f;
	const float fEmptyPairwise = 8.f*OPTDENSE::fPairwiseMul;
	const float fSamePairwise = 24.f*OPTDENSE::fPairwiseMul;
	const IIndex _num_labels = OPTDENSE::nMaxViews+1; // N neighbors and an empty state
	const IIndex _num_nodes = images.GetSize();
	typedef MRFEnergy<TypeGeneral> MRFEnergyType;
	CAutoPtr<MRFEnergyType> energy(new MRFEnergyType(TypeGeneral::GlobalSize()));
	CAutoPtrArr<MRFEnergyType::NodeId> nodes(new MRFEnergyType::NodeId[_num_nodes]);
	typedef SEACAVE::cList<TypeGeneral::REAL, TypeGeneral::REAL, 0> EnergyCostArr;
	// unary costs: inverse proportional to the image pair score
	EnergyCostArr arrUnary(_num_labels);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		FOREACH(k, neighbors)
			arrUnary[k] = avgScore/neighbors[k].score; // use average score to normalize the values (not to depend so much on the number of features in the scene)
		arrUnary[neighbors.GetSize()] = fEmptyUnaryMult*(neighbors.IsEmpty()?avgScore*0.01f:arrUnary[neighbors.GetSize()-1]);
		nodes[n] = energy->AddNode(TypeGeneral::LocalSize(neighbors.GetSize()+1), TypeGeneral::NodeData(arrUnary.Begin()));
	}
	// pairwise costs: as ratios between the area to be covered and the area actually covered
	EnergyCostArr arrPairwise(_num_labels*_num_labels);
	for (PairAreaMap::const_reference edge: edges) {
		const PairIdx pair(edge.first);
		const float area(edge.second);
		const ViewScoreArr& neighborsI(arrDepthData[pair.i].neighbors);
		const ViewScoreArr& neighborsJ(arrDepthData[pair.j].neighbors);
		arrPairwise.Empty();
		FOREACHPTR(pNj, neighborsJ) {
			const IIndex i(pNj->ID);
			const float areaJ(area/pNj->area);
			FOREACHPTR(pNi, neighborsI) {
				const IIndex j(pNi->ID);
				const float areaI(area/pNi->area);
				arrPairwise.Insert(pair.i == i && pair.j == j ? fSamePairwise : fPairwiseMul*(areaI+areaJ));
			}
			arrPairwise.Insert(fEmptyPairwise+fPairwiseMul*areaJ);
		}
		for (const ViewScore& Ni: neighborsI) {
			const float areaI(area/Ni.area);
			arrPairwise.Insert(fPairwiseMul*areaI+fEmptyPairwise);
		}
		arrPairwise.Insert(fEmptyPairwise*2);
		const IIndex nodeI(imagesMap[pair.i]);
		const IIndex nodeJ(imagesMap[pair.j]);
		energy->AddEdge(nodes[nodeI], nodes[nodeJ], TypeGeneral::EdgeData(TypeGeneral::GENERAL, arrPairwise.Begin()));
	}

	// minimize energy
	MRFEnergyType::Options options;
	options.m_eps = OPTDENSE::fOptimizerEps;
	options.m_iterMax = OPTDENSE::nOptimizerMaxIters;
	#ifndef _RELEASE
	options.m_printIter = 1;
	options.m_printMinIter = 1;
	#endif
	#if 1
	TypeGeneral::REAL energyVal, lowerBound;
	energy->Minimize_TRW_S(options, lowerBound, energyVal);
	#else
	TypeGeneral::REAL energyVal;
	energy->Minimize_BP(options, energyVal);
	#endif

	// extract optimized depth map
	neighborsMap.Resize(_num_nodes);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		IIndex& idxNeighbor = neighborsMap[n];
		const IIndex label((IIndex)energy->GetSolution(nodes[n]));
		ASSERT(label <= neighbors.GetSize());
		if (label == neighbors.GetSize()) {
			idxNeighbor = NO_ID; // empty
		} else {
			idxNeighbor = label;
			DEBUG_ULTIMATE("\treference image %3u paired with target image %3u (idx %2u)", images[n], neighbors[label].ID, label);
		}
	}

	// remove all images with no valid neighbors
	RFOREACH(i, neighborsMap) {
		if (neighborsMap[i] == NO_ID) {
			// remove image with no neighbors
			for (IIndex& imageMap: imagesMap)
				if (imageMap != NO_ID && imageMap > i)
					--imageMap;
			imagesMap[images[i]] = NO_ID;
			images.RemoveAtMove(i);
			neighborsMap.RemoveAtMove(i);
		}
	}
	return !images.IsEmpty();
} // SelectViews
/*----------------------------------------------------------------*/

// compute visibility for the reference image (the first image in "images")
// and select the best views for reconstructing the depth-map;
// extract also all 3D points seen by the reference image
bool DepthMapsData::SelectViews(DepthData& depthData)
{
	// find and sort valid neighbor views
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(depthData.neighbors.IsEmpty());
	if (scene.images[idxImage].neighbors.empty() &&
		!scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews, OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2, FD2R(OPTDENSE::fOptimAngle), OPTDENSE::nPointInsideROI))
		return false;
	depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

	// remove invalid neighbor views
	const float fMinArea(OPTDENSE::fMinArea);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(OPTDENSE::fMinAngle));
	const float fMaxAngle(FD2R(OPTDENSE::fMaxAngle));
	if (!Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, OPTDENSE::nMaxViews)) {
		DEBUG_EXTRA("error: reference image %3u has no good images in view", idxImage);
		return false;
	}
	return true;
} // SelectViews
/*----------------------------------------------------------------*/

#ifdef DPC_IMAGE_CACHE
static std::vector<std::pair<IDX, std::unique_ptr<Image32F>>> greyImages;
#endif

// select target image for the reference image (the first image in "images"),
// initialize images data, and initialize depth-map and normal-map;
// if idxNeighbor is not NO_ID, only the reference image and the given neighbor are initialized;
// if numNeighbors is not 0, only the first numNeighbors neighbors are initialized;
// otherwise all are initialized;
// if loadImages, the image data is also setup
// if loadDepthMaps is 1, the depth-maps are loaded from disk,
// if 0, the reference depth-map is initialized from sparse point cloud,
// and if -1, the depth-maps are not initialized
// returns false if there are no good neighbors to estimate the depth-map
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors, bool loadImages, int loadDepthMaps)
{
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(!depthData.neighbors.IsEmpty());

	// set this image the first image in the array
	depthData.images.Empty();
	depthData.images.Reserve(depthData.neighbors.GetSize()+1);
	depthData.images.AddEmpty();

	if (idxNeighbor != NO_ID) {
		// set target image as the given neighbor
		const ViewScore& neighbor = depthData.neighbors[idxNeighbor];
		DepthData::ViewData& viewTrg = depthData.images.AddEmpty();
		viewTrg.pImageData = &scene.images[neighbor.ID];
		viewTrg.scale = neighbor.scale;
		viewTrg.camera = viewTrg.pImageData->camera;
		if (loadImages) {
#ifdef DPC_IMAGE_CACHE
			Image32F* found = nullptr;
			for (auto& i : greyImages) {
				if (i.first == neighbor.ID) {
					found = i.second.get();
					break;
				}
			}
			if (!found) {
				greyImages.emplace_back(neighbor.ID, std::make_unique<Image32F>(viewTrg.image));
				viewTrg.pImageData->image.toGray(*greyImages.back().second, cv::COLOR_BGR2GRAY, true);
				found = greyImages.back().second.get();
			}

			viewTrg.image = *found;
#else
			viewTrg.pImageData->image.toGray(viewTrg.image, cv::COLOR_BGR2GRAY, true);
#endif
			if (DepthData::ViewData::ScaleImage(viewTrg.image, viewTrg.image, viewTrg.scale))
				viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, viewTrg.image.size());
		} else {
			if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
				viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, Image8U::computeResize(viewTrg.pImageData->image.size(), viewTrg.scale));
		}
		DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.ID);
	} else {
		// initialize all neighbor views too (global reconstruction is used)
		const float fMinScore(MAXF(depthData.neighbors.First().score*OPTDENSE::fViewMinScoreRatio, OPTDENSE::fViewMinScore));
		FOREACH(idx, depthData.neighbors) {
			const ViewScore& neighbor = depthData.neighbors[idx];
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				(neighbor.score < fMinScore))
				break;
			DepthData::ViewData& viewTrg = depthData.images.AddEmpty();
			viewTrg.pImageData = &scene.images[neighbor.ID];
			viewTrg.scale = neighbor.scale;
			viewTrg.camera = viewTrg.pImageData->camera;
			if (loadImages) {
#ifdef DPC_IMAGE_CACHE
				Image32F* found = nullptr;
				for (auto& i : greyImages) {
					if (i.first == neighbor.ID) {
						found = i.second.get();
						break;
					}
				}
				if (!found) {
					greyImages.emplace_back(neighbor.ID, std::make_unique<Image32F>(viewTrg.image));
	 				viewTrg.pImageData->image.toGray(*greyImages.back().second, cv::COLOR_BGR2GRAY, true);
					found = greyImages.back().second.get();
				}

				viewTrg.image = *found;
#else
				viewTrg.pImageData->image.toGray(viewTrg.image, cv::COLOR_BGR2GRAY, true);
#endif
				if (DepthData::ViewData::ScaleImage(viewTrg.image, viewTrg.image, viewTrg.scale))
					viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, viewTrg.image.size());
			} else {
				if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
					viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, Image8U::computeResize(viewTrg.pImageData->image.size(), viewTrg.scale));
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// print selected views
		if (g_nVerbosityLevel > 2) {
			String msg;
			for (IIndex i=1; i<depthData.images.GetSize(); ++i)
				msg += String::FormatString(" %3u(%.2fscl)", depthData.images[i].GetID(), depthData.images[i].scale);
			VERBOSE("Reference image %3u paired with %u views:%s (%u shared points)", idxImage, depthData.images.GetSize()-1, msg.c_str(), depthData.points.GetSize());
		} else
		DEBUG_EXTRA("Reference image %3u paired with %u views", idxImage, depthData.images.GetSize()-1);
		#endif
	}
	if (depthData.images.GetSize() < 2) {
		depthData.images.Release();
		return false;
	}

	// initialize reference image as well
	DepthData::ViewData& viewRef = depthData.images.First();
	viewRef.scale = 1;
	viewRef.pImageData = &scene.images[idxImage];
	viewRef.camera = viewRef.pImageData->camera;
	if (loadImages) {
#ifdef DPC_IMAGE_CACHE
		Image32F* found = nullptr;
		for (auto& i : greyImages) {
			if (i.first == idxImage) {
				found = i.second.get();
				break;
			}
		}
		if (!found) {
			greyImages.emplace_back(idxImage, std::make_unique<Image32F>(viewRef.image));
			viewRef.pImageData->image.toGray(*greyImages.back().second, cv::COLOR_BGR2GRAY, true);
			found = greyImages.back().second.get();
		}

		viewRef.image = *found;
#else
		viewRef.pImageData->image.toGray(viewRef.image, cv::COLOR_BGR2GRAY, true);
#endif
	}
	// initialize views
	depthData.images[0].Init(viewRef.camera); // JPB WIP BUG
	for (IIndex i=1; i<depthData.images.size(); ++i) {
		DepthData::ViewData& view = depthData.images[i];
		if (loadDepthMaps > 0) {
			// load known depth-map
			String imageFileName;
			IIndexArr IDs;
			cv::Size imageSize;
			Depth dMin, dMax;
			NormalMap normalMap;
			ConfidenceMap confMap;
			ViewsMap viewsMap;
			ImportDepthDataRaw(ComposeDepthFilePath(view.GetID(), "dmap"),
				imageFileName, IDs, imageSize, view.cameraDepthMap.K, view.cameraDepthMap.R, view.cameraDepthMap.C,
				dMin, dMax, view.depthMap, normalMap, confMap, viewsMap, 1);
		}
		view.Init(viewRef.camera);
	}

	if (loadDepthMaps > 0) {
		// load known depth-map and normal-map
		String imageFileName;
		IIndexArr IDs;
		cv::Size imageSize;
		Camera camera;
		ConfidenceMap confMap;
		ViewsMap viewsMap;
		if (!ImportDepthDataRaw(ComposeDepthFilePath(viewRef.GetID(), "dmap"),
				imageFileName, IDs, imageSize, camera.K, camera.R, camera.C, depthData.dMin, depthData.dMax,
				depthData.depthMap, depthData.normalMap, confMap, viewsMap, 3))
			return false;
		ASSERT(viewRef.image.size() == depthData.depthMap.size());
	} else if (loadDepthMaps == 0) {
		// initialize depth and normal maps
		if (OPTDENSE::nMinViewsTrustPoint < 2 || depthData.points.empty()) {
			// compute depth range and initialize known depths, else random
			const Image8U::Size size(viewRef.image.size());
			depthData.depthMap.create(size); depthData.depthMap.memset(0);
			depthData.normalMap.create(size);
			if (depthData.points.empty()) {
				// all values will be initialized randomly
				depthData.dMin = 1e-1f;
				depthData.dMax = 1e+2f;
			} else {
				// initialize with the sparse point-cloud
				const int nPixelArea(2); // half windows size around a pixel to be initialize with the known depth
				depthData.dMin = FLT_MAX;
				depthData.dMax = 0;
				FOREACHPTR(pPoint, depthData.points) {
					const PointCloud::Point& X = scene.pointcloud.Point(*pPoint);
					const Point3 camX(viewRef.camera.TransformPointW2C(Cast<REAL>(X)));
					const ImageRef x(ROUND2INT(viewRef.camera.TransformPointC2I(camX)));
					const float d((float)camX.z);
					const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0));
					const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));
					for (int y=sx.y; y<=ex.y; ++y) {
						for (int x=sx.x; x<=ex.x; ++x) {
							depthData.depthMap(y,x) = d;
							depthData.normalMap(y,x) = Normal::ZERO;
						}
					}
					if (depthData.dMin > d)
						depthData.dMin = d;
					if (depthData.dMax < d)
						depthData.dMax = d;
				}
				depthData.dMin *= 0.9f;
				depthData.dMax *= 1.1f;
			}
		} else {
			ASSERT(!depthData.points.empty());
			// compute rough estimates using the sparse point-cloud
			InitDepthMap(depthData);
		}
	}

	return true;
} // InitViews
/*----------------------------------------------------------------*/

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
bool DepthMapsData::InitDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	// JPB WIP BUG This is called the very first pass. JPB WIP OPT

	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.GetView());
	TriangulatePoints2DepthMap(image, scene.pointcloud, depthData.points, depthData.depthMap, depthData.normalMap, depthData.dMin, depthData.dMax, OPTDENSE::bAddCorners, OPTDENSE::bInitSparse);
	depthData.dMin *= 0.9f;
	depthData.dMax *= 1.1f;

	#if TD_VERBOSE != TD_VERBOSE_OFF
	// save rough depth map as image
	if (g_nVerbosityLevel > 4) {
		ExportDepthMap(ComposeDepthFilePath(image.GetID(), "init.png"), depthData.depthMap);
		ExportNormalMap(ComposeDepthFilePath(image.GetID(), "init.normal.png"), depthData.normalMap);
		ExportPointCloud(ComposeDepthFilePath(image.GetID(), "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
	}
	#endif

	DEBUG_ULTIMATE("Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", image.GetID(), depthData.points.size(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
	return true;
} // InitDepthMap
/*----------------------------------------------------------------*/

#ifdef DPC_EXTENDED_OMP_THREADING2
// initialize the confidence map (NCC score map) with the score of the current estimates
void* STCALL DepthMapsData::ScoreDepthMapTmp(cList<DepthEstimator>& estimators)
{
	const DepthEstimator& anEstimator = estimators.First();
	const __int64 idxCount = anEstimator.coords.GetSize();

#ifndef _RELEASE
	thread_local SEACAVE::Random rnd(SEACAVE::Random::default_seed());
#else
	//thread_local SEACAVE::Random rnd;
#endif

	if (anEstimator.sh.mLowResDepthMapEmpty) {
#pragma omp parallel for num_threads((int) estimators.GetSize())
		for (__int64 i = 0; i < idxCount; ++i) {
			DepthEstimator& pe = estimators[omp_get_thread_num()];

			const ImageRef& x = pe.coords[i];
			constexpr bool HasLowResDepthMap = false;
			if (!pe.PreparePixelPatch(x) || !pe.FillPixelPatch<HasLowResDepthMap>()) {
				pe.depthMap0.pix(x) = 0;
				pe.normalMap0.pix(x) = Normal::ZERO;
				pe.confMap0.pix(x) = 2.f;
				continue;
			}
			Depth& depth = pe.depthMap0.pix(x);
			Normal& normal = pe.normalMap0.pix(x);
			const Normal viewDir(_AsArray(pe.vX0, 0), _AsArray(pe.vX0, 1), _AsArray(pe.vX0, 2)); // Cast<float>(static_cast<const Point3&>(pe.X0)));
			if (!ISINSIDE(depth, pe.dMin, pe.dMax)) {
				// init with random values
				depth = pe.RandomDepth(pe.rnd, pe.dMinSqr, pe.dMaxSqr);
				normal = pe.RandomNormal(pe.rnd, viewDir);
			}
			else if (normal.dot(viewDir) >= 0) {
				// replace invalid normal with random values
				normal = pe.RandomNormal(pe.rnd, viewDir);
			}
			ASSERT(ISEQUAL(norm(normal), 1.f));
			// Any ScorePixel needs sh.mVScoreFactor set accurately.
			// This was done on in the estimator constructor.
			pe.confMap0.pix(x) = pe.ScorePixel(depth, normal);
		}
	}
	else {
#pragma omp parallel for num_threads((int) estimators.GetSize())
		for (__int64 i = 0; i < idxCount; ++i) {
			DepthEstimator& pe = estimators[omp_get_thread_num()];

			const ImageRef& x = pe.coords[i];
			constexpr bool HasLowResDepthMap = true;
			if (!pe.PreparePixelPatch(x) || !pe.FillPixelPatch<HasLowResDepthMap>()) {
				pe.depthMap0.pix(x) = 0;
				pe.normalMap0.pix(x) = Normal::ZERO;
				pe.confMap0.pix(x) = 2.f;
				continue;
			}
			Depth& depth = pe.depthMap0.pix(x);
			Normal& normal = pe.normalMap0.pix(x);
			const Normal viewDir(_AsArray(pe.vX0, 0), _AsArray(pe.vX0, 1), _AsArray(pe.vX0, 2)); // Cast<float>(static_cast<const Point3&>(pe.X0)));
			if (!ISINSIDE(depth, pe.dMin, pe.dMax)) {
				// init with random values
				depth = pe.RandomDepth(pe.rnd, pe.dMinSqr, pe.dMaxSqr);
				normal = pe.RandomNormal(pe.rnd, viewDir);
			}
			else if (normal.dot(viewDir) >= 0) {
				// replace invalid normal with random values
				normal = pe.RandomNormal(pe.rnd, viewDir);
			}
			ASSERT(ISEQUAL(norm(normal), 1.f));
			// Any ScorePixel needs sh.mVScoreFactor set accurately.
			// This was done on in the estimator constructor.
			pe.confMap0.pix(x) = pe.ScorePixel(depth, normal);
		}
	}

	return NULL;
}
#else
void* STCALL DepthMapsData::ScoreDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		if (estimator.sh.mLowResDepthMapEmpty) {
			constexpr bool HasLowResDepthMap = false;
			if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch<HasLowResDepthMap>()) {
				estimator.depthMap0(x) = 0;
				estimator.normalMap0(x) = Normal::ZERO;
				estimator.confMap0(x) = 2.f;
				continue;
			}
		} else {
			constexpr bool HasLowResDepthMap = true;
			if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch<HasLowResDepthMap>()) {
				estimator.depthMap0(x) = 0;
				estimator.normalMap0(x) = Normal::ZERO;
				estimator.confMap0(x) = 2.f;
				continue;
			}
		}
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		const Normal viewDir(_AsArray(estimator.vX0, 0), _AsArray(estimator.vX0, 1), _AsArray(estimator.vX0, 2)); // Cast<float>(static_cast<const Point3&>(estimator.X0)));
		if (!ISINSIDE(depth, estimator.dMin, estimator.dMax)) {
			// init with random values
			depth = estimator.RandomDepth(estimator.rnd, estimator.dMinSqr, estimator.dMaxSqr);
			normal = estimator.RandomNormal(estimator.rnd, viewDir);
		} else if (normal.dot(viewDir) >= 0) {
			// replace invalid normal with random values
			normal = estimator.RandomNormal(estimator.rnd, viewDir);
		}
		ASSERT(ISEQUAL(norm(normal), 1.f));
		estimator.confMap0(x) = estimator.ScorePixel(depth, normal);
	}
	return NULL;
}
#endif // DPC_EXTENDED_OMP_THREADING2

#ifdef DPC_EXTENDED_OMP_THREADING
// run propagation and random refinement cycles
void* STCALL DepthMapsData::EstimateDepthMapTmp(cList<DepthEstimator>& estimators)
{
	// ProcessPixel can read neighbor information without synchronization
	// which appears to alter the output when the threading changes.
	// Thus we preserve the original thread handling (non-chunked).

	const DepthEstimator& anEstimator = estimators.First();
	const __int64 cnt = anEstimator.coords.GetSize();
	// No combination of pinning here works more favorably.
#pragma omp parallel for num_threads(estimators.GetSize()) schedule(guided)
	for (__int64 i = 0; i < cnt; ++i) {
		DepthEstimator& pe = estimators[omp_get_thread_num()];
		pe.ProcessPixel(i);
	}
	return NULL;
}
#else
void* STCALL DepthMapsData::EstimateDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
		estimator.ProcessPixel(idx);
	return NULL;
}
#endif

// remove all estimates with too big score and invert confidence map
#ifdef DPC_EXTENDED_OMP_THREADING
void* STCALL DepthMapsData::EndDepthMapTmp(cList<DepthEstimator>& estimators)
{
	const DepthEstimator& anEstimator = estimators.First();
	const __int64 cnt = anEstimator.coords.GetSize();
#pragma omp parallel for num_threads(estimators.GetSize()) schedule(guided)
	for (__int64 i = 0; i < cnt; ++i) {
		DepthEstimator& pe = estimators[omp_get_thread_num()];

		const ImageRef& x = pe.coords[i];
		//ASSERT(estimator.depthMap0(x) >= 0);
		Depth& depth = pe.depthMap0(x);
		float& conf = pe.confMap0(x);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		if (depth <= 0 || conf >= OPTDENSE::fNCCThresholdKeep) {
			conf = 0;
			depth = 0;
			pe.normalMap0(x) = Normal::ZERO;
		}
		else {
			// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
			conf = conf>=1.f ? 0.f : 1.f-conf;
		}
	}

	return NULL;
}
#else
#ifdef DPC_EXTENDED_OMP_THREADING2
void* STCALL DepthMapsData::EndDepthMapTmp(cList<DepthEstimator>& estimators)
{
	const DepthEstimator& anEstimator = estimators.First();
	const __int64 cnt = anEstimator.coords.GetSize();
#pragma omp parallel for num_threads((int) estimators.GetSize()) schedule(guided)
	for (__int64 i = 0; i < cnt; ++i) {
		const ImageRef& x = anEstimator.coords[i];
		//ASSERT(estimator.depthMap0(x) >= 0);
		Depth& depth = anEstimator.depthMap0.pix(x);
		float& conf = anEstimator.confMap0.pix(x);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		if (depth <= 0 || conf >= OPTDENSE::fNCCThresholdKeep) {
			conf = 0;
			depth = 0;
			anEstimator.normalMap0.pix(x) = Normal::ZERO;
		}
		else {
			// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
			conf = conf>=1.f ? 0.f : 1.f-conf;
		}
	}

	return NULL;
}
#else
void* STCALL DepthMapsData::EndDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	MAYBEUNUSED const float fOptimAngle(FD2R(OPTDENSE::fOptimAngle));
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		ASSERT(estimator.depthMap0(x) >= 0);
		Depth& depth = estimator.depthMap0(x);
		float& conf = estimator.confMap0(x);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		if (depth <= 0 || conf >= OPTDENSE::fNCCThresholdKeep) {
			conf = 0;
			depth = 0;
			estimator.normalMap0(x) = Normal::ZERO;
		} else {
			#if 1
			// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
			conf = conf>=1.f ? 0.f : 1.f-conf;
			#else
			#if 1
			FOREACH(i, estimator.images)
				estimator.scores[i] = ComputeAngle<REAL,float>(estimator.image0.camera.TransformPointI2W(Point3(x,depth)).ptr(), estimator.image0.camera.C.ptr(), estimator.images[i].view.camera.C.ptr());
			#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
			const float fCosAngle(estimator.scores.GetNth(estimator.idxScore));
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
			const float fCosAngle(estimator.scores.mean());
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MIN
			const float fCosAngle(estimator.scores.minCoeff());
			#else
			const float fCosAngle(estimator.idxScore ?
				std::accumulate(estimator.scores.begin(), &estimator.scores.PartialSort(estimator.idxScore), 0.f) / estimator.idxScore :
				*std::min_element(estimator.scores.cbegin(), estimator.scores.cend()));
			#endif
			const float wAngle(MINF(POW(ACOS(fCosAngle)/fOptimAngle,1.5f),1.f));
			#else
			const float wAngle(1.f);
			#endif
			#if 1
			conf = wAngle/MAXF(conf,1e-2f);
			#else
			conf = wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#endif
			#endif
		}
	}
	return NULL;
}
#endif
#endif

DepthData DepthMapsData::ScaleDepthData(const DepthData& inputDeptData, float scale) {
	ASSERT(scale <= 1);
	if (scale == 1)
		return inputDeptData;
	DepthData rescaledDepthData(inputDeptData);
	FOREACH (idxView, rescaledDepthData.images) {
		DepthData::ViewData& viewData = rescaledDepthData.images[idxView];
		ASSERT(viewData.depthMap.empty() || viewData.image.size() == viewData.depthMap.size());
		cv::resize(viewData.image, viewData.image, cv::Size(), scale, scale, cv::INTER_AREA);
		viewData.camera = viewData.pImageData->camera;
		viewData.camera.K = viewData.camera.GetScaledK(viewData.pImageData->GetSize(), viewData.image.size());
		if (!viewData.depthMap.empty()) {
			cv::resize(viewData.depthMap, viewData.depthMap, viewData.image.size(), 0, 0, cv::INTER_AREA);
			viewData.cameraDepthMap = viewData.pImageData->camera;
			viewData.cameraDepthMap.K = viewData.cameraDepthMap.GetScaledK(viewData.pImageData->GetSize(), viewData.image.size());
		}
		viewData.Init(rescaledDepthData.images[0].camera);
	}
	if (!rescaledDepthData.depthMap.empty())
		cv::resize(rescaledDepthData.depthMap, rescaledDepthData.depthMap, cv::Size(), scale, scale, cv::INTER_NEAREST);
	if (!rescaledDepthData.normalMap.empty())
		cv::resize(rescaledDepthData.normalMap, rescaledDepthData.normalMap, cv::Size(), scale, scale, cv::INTER_NEAREST);
	return rescaledDepthData;
}

#ifdef DPC_FASTER_SAMPLING
struct ImageKey_t {
	ImageKey_t(int id, float scale, int width, int height) : mId(id), mScale(scale), mWidth(width), mHeight(height) {}

	int mId;
	float mScale;
	int mWidth;
	int mHeight;

	bool operator==(const ImageKey_t& other) const
	{
		return (mId == other.mId
			&& mScale == other.mScale
			&& mWidth == other.mWidth
			&& mHeight == other.mHeight);
	}
};

template <>
struct std::hash<ImageKey_t>
{
	std::size_t operator()(const ImageKey_t& k) const
	{
		using std::size_t;
		using std::hash;
		using std::string;

		// Compute individual hash values for first,
		// second and third and combine them using XOR
		// and bit shifting:

		return (
			(hash<float>()(k.mScale)
			^ (hash<int>()(k.mWidth) << 1)) >> 1)
			^ (hash<int>()(k.mHeight) << 1)
			^ (hash<int>()(k.mId) << 2);
	}
};
std::mutex sCachedImagesMutex;
std::unordered_map<ImageKey_t, Image32F> sCachedImages;
#endif

// estimate depth-map using propagation and random refinement with NCC score
// as in: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
// The implementations follows closely the paper, although there are some changes/additions.
// Given two views of the same scene, we note as the "reference image" the view for which a depth-map is reconstructed, and the "target image" the other view.
// As a first step, the whole depth-map is approximated by interpolating between the available sparse points.
// Next, the depth-map is passed from top/left to bottom/right corner and the opposite sens for each of the next steps.
// For each pixel, first the current depth estimate is replaced with its neighbor estimates if the NCC score is better.
// Second, the estimate is refined by trying random estimates around the current depth and normal values, keeping the one with the best score.
// The estimation can be stopped at any point, and usually 2-3 iterations are enough for convergence.
// For each pixel, the depth and normal are scored by computing the NCC score between the patch in the reference image and the wrapped patch in the target image, as dictated by the homography matrix defined by the current values to be estimate.
// In order to ensure some smoothness while locally estimating each pixel, a bonus is added to the NCC score if the estimate for this pixel is close to the estimates for the neighbor pixels.
// Optionally, the occluded pixels can be detected by extending the described iterations to the target image and removing the estimates that do not have similar values in both views.
//  - nGeometricIter: current geometric-consistent estimation iteration (-1 - normal patch-match)
bool DepthMapsData::EstimateDepthMap(IIndex idxImage, int nGeometricIter)
{
#ifdef DPC_FLUSH_DENORMALS
	_controlfp_s(NULL, _DN_FLUSH, _MCW_DN);
#endif

	static bool firstTime = true;
	if (firstTime) {
		bool usingGPU = false;
#ifdef _USE_CUDA
		usingGPU = pmCUDA;
#endif
		if (usingGPU) {
			VERBOSE("Running patch-matching on a CUDA-enabled GPU.");
		} else {
			VERBOSE("Running patch-matching on the CPU.");
		}
		firstTime = false;
	}

	#ifdef _USE_CUDA
	if (pmCUDA) {
		pmCUDA->EstimateDepthMap(arrDepthData[idxImage]);
		return true;
	}
	#endif // _USE_CUDA

	TD_TIMER_STARTD();

	const unsigned nMaxThreads(scene.nMaxThreads);
	const unsigned iterBegin(nGeometricIter < 0 ? 0u : OPTDENSE::nEstimationIters+(unsigned)nGeometricIter);
	const unsigned iterEnd(nGeometricIter < 0 ? OPTDENSE::nEstimationIters : iterBegin+1);

	// init threads
	ASSERT(nMaxThreads > 0);
	cList<DepthEstimator> estimators;
	estimators.reserve(nMaxThreads);

#ifndef DPC_EXTENDED_OMP_THREADING
	cList<SEACAVE::Thread> threads;
	if (nMaxThreads > 1)
		threads.resize(nMaxThreads-1); // current thread is also used
	volatile Thread::safe_t idxPixel;
#endif

	// Multi-Resolution : 
	DepthData& fullResDepthData(arrDepthData[idxImage]);
	const unsigned totalScaleNumber(nGeometricIter < 0 ? OPTDENSE::nSubResolutionLevels : 0u);
	DepthMap lowResDepthMap;
	NormalMap lowResNormalMap;
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	#else
	Image64F imageSum0;
	#endif
	DepthMap currentSizeResDepthMap;
	for (unsigned scaleNumber = totalScaleNumber+1; scaleNumber-- > 0; ) {
		// initialize
		float scale = 1.f / POWI(2, scaleNumber);
		DepthData currentDepthData(ScaleDepthData(fullResDepthData, scale));
		DepthData& depthData(scaleNumber==0 ? fullResDepthData : currentDepthData);
		ASSERT(depthData.images.size() > 1);

#ifdef DPC_FASTER_SAMPLING
		const int numImages = (int) depthData.images.size();
#pragma omp parallel for num_threads(nMaxThreads)
		for (int j = 0; j < numImages; ++j) {
			auto& i = depthData.images[j];
			sCachedImagesMutex.lock();
			auto itPair = sCachedImages.try_emplace(ImageKey_t(i.pImageData->ID, scale, i.image.cols, i.image.rows));
			if (itPair.second) { // Insertion took place
				itPair.first->second = Image32F(i.image.rows, i.image.cols*4);
				float* __restrict dst = (float*)itPair.first->second.data;
				for (int y = 0; y < i.image.rows; ++y) {
					const float* row = i.image.ptr<float>(y);
					const float* nextRow = i.image.ptr<float>(y + (y < (i.image.rows-1) ? 1 : 0));
					float s0, s1, s2, s3;
					for (int x = 0; x < i.image.cols-1; ++x, ++row, ++nextRow) {
						s0 = row[0];
						s1 = row[1];
						s2 = nextRow[0];
						s3 = nextRow[1];
						*dst++ = s0; *dst++ = (s1-s0); *dst++ = (s2-s0); *dst++ = (s3-s2-s1+s0);
					}
					s0 = row[0];
					s1 = row[0];
					s2 = nextRow[0];
					s3 = nextRow[0];
					*dst++ = s0; *dst++ = (s1-s0); *dst++ = (s2-s0); *dst++ = (s3-s2-s1+s0);
				}
			}
			sCachedImagesMutex.unlock();

			i.imageBig = itPair.first->second;
		}
#endif

		const DepthData::ViewData& image(depthData.images.front());
		ASSERT(!image.image.empty() && !depthData.images[1].image.empty());
		const Image8U::Size size(image.image.size());
		if (scaleNumber != totalScaleNumber) {
			cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0, OPTDENSE::nIgnoreMaskLabel >= 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
			cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
			depthData.depthMap.copyTo(currentSizeResDepthMap);
		}
		else if (totalScaleNumber > 0) {
			fullResDepthData.depthMap.release();
			fullResDepthData.normalMap.release();
			fullResDepthData.confMap.release();
		}
		depthData.confMap.create(size);

		// init integral images and index to image-ref map for the reference data
		#if DENSE_NCC == DENSE_NCC_WEIGHTED
		#else
		cv::integral(image.image, imageSum0, CV_64F);
		#endif
		if (prevDepthMapSize != size || OPTDENSE::nIgnoreMaskLabel >= 0) {
			BitMatrix mask;
			if (OPTDENSE::nIgnoreMaskLabel >= 0 && DepthEstimator::ImportIgnoreMask(*image.pImageData, depthData.depthMap.size(), (uint8_t)OPTDENSE::nIgnoreMaskLabel, mask))
				depthData.ApplyIgnoreMask(mask);
			DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));
			#if 0
			// show pixels to be processed
			Image8U cmask(size);
			cmask.memset(0);
			for (const DepthEstimator::MapRef& x: coords)
				cmask(x.y, x.x) = 255;
			cmask.Show("cmask");
			#endif
			prevDepthMapSize = size;
		}
		// initialize the reference confidence map (NCC score map) with the score of the current estimates
		{
			// create working threads
#ifndef DPC_EXTENDED_OMP_THREADING
			idxPixel = -1;
#endif
			ASSERT(estimators.empty());
			while (estimators.size() < nMaxThreads) {
				estimators.emplace_back(iterBegin, depthData,
#ifndef DPC_EXTENDED_OMP_THREADING
					idxPixel,
#endif
#if DENSE_NCC == DENSE_NCC_WEIGHTED
					#else
					imageSum0,
					#endif
					coords);
				estimators.Last().lowResDepthMap = currentSizeResDepthMap;
				estimators.Last().sh.mLowResDepthMapEmpty = estimators.Last().lowResDepthMap.empty();
			}
#ifdef DPC_EXTENDED_OMP_THREADING2
			ScoreDepthMapTmp(estimators);
#else
			FOREACH(i, threads)
				threads[i].start(ScoreDepthMapTmp, &estimators[i]);
			ScoreDepthMapTmp(&estimators.back());
			// wait for the working threads to close
			FOREACHPTR(pThread, threads)
				pThread->join();
#endif

			estimators.clear();
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save rough depth map as image
			if (g_nVerbosityLevel > 4 && nGeometricIter < 0) {
				ExportDepthMap(ComposeDepthFilePath(image.GetID(), "rough.png"), depthData.depthMap);
				ExportNormalMap(ComposeDepthFilePath(image.GetID(), "rough.normal.png"), depthData.normalMap);
				ExportPointCloud(ComposeDepthFilePath(image.GetID(), "rough.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
		}

		// run propagation and random refinement cycles on the reference data
		for (unsigned iter=iterBegin; iter<iterEnd; ++iter) {
			// create working threads
#ifndef DPC_EXTENDED_OMP_THREADING
			idxPixel = -1;
#endif
			ASSERT(estimators.empty());
			while (estimators.size() < nMaxThreads) {
				estimators.emplace_back(iter, depthData,
#ifndef DPC_EXTENDED_OMP_THREADING
					idxPixel,
#endif
					#if DENSE_NCC == DENSE_NCC_WEIGHTED
					#else
					imageSum0,
					#endif
					coords);
				estimators.Last().lowResDepthMap = currentSizeResDepthMap;
				estimators.Last().sh.mLowResDepthMapEmpty = estimators.Last().lowResDepthMap.empty();
			}
#ifdef DPC_EXTENDED_OMP_THREADING
			EstimateDepthMapTmp(estimators);
#else
			FOREACH(i, threads)
				threads[i].start(EstimateDepthMapTmp, &estimators[i]);
			EstimateDepthMapTmp(&estimators.back());
			// wait for the working threads to close
			FOREACHPTR(pThread, threads)
				pThread->join();
#endif
			estimators.clear();
			#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
			// save intermediate depth map as image
			if (g_nVerbosityLevel > 4) {
				String path(ComposeDepthFilePath(image.GetID(), "iter")+String::ToString(iter));
				if (nGeometricIter >= 0)
					path += String::FormatString(".geo%d", nGeometricIter);
				ExportDepthMap(path+".png", depthData.depthMap);
				ExportNormalMap(path+".normal.png", depthData.normalMap);
				ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
		}

		// remember sub-resolution estimates for next iteration
		if (scaleNumber > 0) {
			lowResDepthMap = depthData.depthMap;
			lowResNormalMap = depthData.normalMap;
		}
	}

	DepthData& depthData(fullResDepthData);
	// remove all estimates with too big score and invert confidence map
#ifdef DPC_EXTENDED_OMP_THREADING2
	{
		const float fNCCThresholdKeep(OPTDENSE::fNCCThresholdKeep);
		if (nGeometricIter < 0 && OPTDENSE::nEstimationGeometricIters)
			OPTDENSE::fNCCThresholdKeep *= 1.333f;
		// create working threads
		ASSERT(estimators.empty());
		while (estimators.size() < nMaxThreads)
			estimators.emplace_back(0, depthData,
				idxPixel, // Unused
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				#else
				imageSum0,
				#endif
				coords);
		EndDepthMapTmp(estimators);
		estimators.clear();
		OPTDENSE::fNCCThresholdKeep = fNCCThresholdKeep;
	}
#else
	{
		const float fNCCThresholdKeep(OPTDENSE::fNCCThresholdKeep);
		if (nGeometricIter < 0 && OPTDENSE::nEstimationGeometricIters)
			OPTDENSE::fNCCThresholdKeep *= 1.333f;
		// create working threads
#ifndef DPC_EXTENDED_OMP_THREADING
		idxPixel = -1;
#endif
		ASSERT(estimators.empty());
		while (estimators.size() < nMaxThreads)
			estimators.emplace_back(0, depthData,
#ifndef DPC_EXTENDED_OMP_THREADING
				idxPixel,
#endif
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				#else
				imageSum0,
				#endif
				coords);
#ifdef DPC_EXTENDED_OMP_THREADING
		EndDepthMapTmp(estimators);
#else
		FOREACH(i, threads)
			threads[i].start(EndDepthMapTmp, &estimators[i]);
		EndDepthMapTmp(&estimators.back());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
#endif
		estimators.clear();
		OPTDENSE::fNCCThresholdKeep = fNCCThresholdKeep;
	}
#endif

	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", depthData.images.front().GetID(),
		depthData.images.size() > 2 ?
			String::FormatString("estimated using %2u images", depthData.images.size()-1).c_str() :
			String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		depthData.depthMap.cols, depthData.depthMap.rows, TD_TIMER_GET_FMT().c_str());

#ifdef DPC_FLUSH_DENORMALS
	_controlfp_s(NULL, _DN_SAVE, _MCW_DN);
#endif

	return true;
} // EstimateDepthMap
/*----------------------------------------------------------------*/


// filter out small depth segments from the given depth map
bool DepthMapsData::RemoveSmallSegments(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*0.7f);
	unsigned speckle_size = OPTDENSE::nSpeckleSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// allocate memory on heap for dynamic programming arrays
	TImage<bool> done_map(size, false);
	CAutoPtrArr<ImageRef> seg_list(new ImageRef[size.x*size.y]);
	unsigned seg_list_count;
	unsigned seg_list_curr;
	ImageRef neighbor[4];

	// for all pixels do
	for (int u=0; u<size.x; ++u) {
		for (int v=0; v<size.y; ++v) {
			// if the first pixel in this segment has been already processed => skip
			if (done_map(v,u))
				continue;

			// init segment list (add first element
			// and set it to be the next element to check)
			seg_list[0] = ImageRef(u,v);
			seg_list_count = 1;
			seg_list_curr  = 0;

			// add neighboring segments as long as there
			// are none-processed pixels in the seg_list;
			// none-processed means: seg_list_curr<seg_list_count
			while (seg_list_curr < seg_list_count) {
				// get address of current pixel in this segment
				const ImageRef addr_curr(seg_list[seg_list_curr]);
				const Depth& depth_curr = depthMap(addr_curr);

				if (depth_curr>0) {
					// fill list with neighbor positions
					neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
					neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
					neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
					neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

					// for all neighbors do
					for (int i=0; i<4; ++i) {
						// get neighbor pixel address
						const ImageRef& addr_neighbor(neighbor[i]);
						// check if neighbor is inside image
						if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y) {
							// check if neighbor has not been added yet
							bool& done = done_map(addr_neighbor);
							if (!done) {
								// check if the neighbor is valid and similar to the current pixel
								// (belonging to the current segment)
								const Depth& depth_neighbor = depthMap(addr_neighbor);
								if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
									// add neighbor coordinates to segment list
									seg_list[seg_list_count++] = addr_neighbor;
									// set neighbor pixel in done_map to "done"
									// (otherwise a pixel may be added 2 times to the list, as
									//  neighbor of one pixel and as neighbor of another pixel)
									done = true;
								}
							}
						}
					}
				}

				// set current pixel in seg_list to "done"
				++seg_list_curr;

				// set current pixel in done_map to "done"
				done_map(addr_curr) = true;
			} // end: while (seg_list_curr < seg_list_count)

			// if segment NOT large enough => invalidate pixels
			if (seg_list_count < speckle_size) {
				// for all pixels in current segment invalidate pixels
				for (unsigned i=0; i<seg_list_count; ++i) {
					depthMap(seg_list[i]) = 0;
					if (!normalMap.empty()) normalMap(seg_list[i]) = Normal::ZERO;
					if (!confMap.empty()) confMap(seg_list[i]) = 0;
				}
			}
		}
	}

	return true;
} // RemoveSmallSegments
/*----------------------------------------------------------------*/

// try to fill small gaps in the depth map
bool DepthMapsData::GapInterpolation(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*2.5f);
	unsigned nIpolGapSize = OPTDENSE::nIpolGapSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// 1. Row-wise:
	// for each row do
	for (int v=0; v<size.y; ++v) {
		// init counter
		unsigned count = 0;

		// for each element of the row do
		for (int u=0; u<size.x; ++u) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)u > count) {
				// first value index for interpolation
				int u_curr(u-count);
				const int u_first(u_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v,u_first);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v,u_curr) = avg;
					} while (++u_curr<u);						
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v,u_curr) = (d+=diff);
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v,u_first), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v,u_curr) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v,u_curr));
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}

	// 2. Column-wise:
	// for each column do
	for (int u=0; u<size.x; ++u) {

		// init counter
		unsigned count = 0;

		// for each element of the column do
		for (int v=0; v<size.y; ++v) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)v > count) {
				// first value index for interpolation
				int v_curr(v-count);
				const int v_first(v_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v_first,u);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v_curr,u) = avg;
					} while (++v_curr<v);						
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v_curr,u) = (d+=diff);
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v_first,u), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v_curr,u) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v_curr,u));
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}
	return true;
} // GapInterpolation
/*----------------------------------------------------------------*/

// filter depth-map, one pixel at a time, using confidence based fusion or neighbor pixels
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool bAdjust)
{
	TD_TIMER_STARTD();

	// count valid neighbor depth-maps
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
	const IIndex N = idxNeighbors.GetSize();
	ASSERT(OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
	const IIndex nMinViews(MINF(OPTDENSE::nMinViewsFilter,scene.nCalibratedImages-1));
	const IIndex nMinViewsAdjust(MINF(OPTDENSE::nMinViewsFilterAdjust,scene.nCalibratedImages-1));
	if (N < nMinViews || N < nMinViewsAdjust) {
		DEBUG("error: depth map %3u can not be filtered", depthDataRef.GetView().GetID());
		return false;
	}

	// project all neighbor depth-maps to this image
	const DepthData::ViewData& imageRef = depthDataRef.images.First();
	const Image8U::Size sizeRef(depthDataRef.depthMap.size());
	const Camera& cameraRef = imageRef.camera;
	DepthMapArr depthMaps(N);
	ConfidenceMapArr confMaps(N);

	FOREACH(n, depthMaps) {
		DepthMap& depthMap = depthMaps[n];
		depthMap.create(sizeRef);
		depthMap.memset(0);
		ConfidenceMap& confMap = confMaps[n];
		if (bAdjust) {
			confMap.create(sizeRef);
			confMap.memset(0);
		}
		const IIndex idxView = depthDataRef.neighbors[idxNeighbors[(IIndex)n]].ID;
		const DepthData& depthData = arrDepthData[idxView];
		const Camera& camera = depthData.images.First().camera;
		const Image8U::Size size(depthData.depthMap.size());
		for (int i=0; i<size.height; ++i) {
			const Depth* const __restrict pDepth = &depthData.depthMap(i, 0);
			for (int j=0; j<size.width; ++j) {
				const Depth depth(pDepth[j]);
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const ImageRef x(j,i);
				const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));
				const Point3 camX(cameraRef.TransformPointW2C(X));
				if (camX.z <= 0)
					continue;
				#if 0
				// set depth on the rounded image projection only
				const ImageRef xRef(ROUND2INT(cameraRef.TransformPointC2I(camX)));
				if (!depthMap.isInside(xRef))
					continue;
				Depth& depthRef(depthMap(xRef));
				if (depthRef != 0 && depthRef < camX.z)
					continue;
				depthRef = camX.z;
				if (bAdjust)
					confMap(xRef) = depthData.confMap(x);
				#else
				// set depth on the 4 pixels around the image projection
				const Point2 imgX(cameraRef.TransformPointC2I(camX));
				const ImageRef xRefs[4] = {
					ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
				};

				for (int p=0; p<4; ++p) {
					const ImageRef& xRef = xRefs[p];
					if ((unsigned) xRef.x < size.width && (unsigned) xRef.y < size.height) {
						//if (!depthMap.isInside(xRef))
						//	continue;
					Depth& depthRef(depthMap(xRef));
					if (depthRef != 0 && depthRef < (Depth)camX.z)
						continue;
					depthRef = (Depth)camX.z;
					if (bAdjust)
						confMap(xRef) = depthData.confMap(x);
				}
				}
				#endif
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (g_nVerbosityLevel > 3)
			ExportDepthMap(MAKE_PATH(String::FormatString("depthRender%04u.%04u.png", depthDataRef.GetView().GetID(), idxView)), depthMap);
		#endif
	}

	const float thDepthDiff(OPTDENSE::fDepthDiffThreshold*1.2f);
	DepthMap newDepthMap(sizeRef);
	ConfidenceMap newConfMap(sizeRef);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	size_t nProcessed(0), nDiscarded(0);
	#endif
	if (bAdjust) {
		// average similar depths, and decrease confidence if depths do not agree
		// (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// update best depth and confidence estimate with all estimates
				float posConf(depthDataRef.confMap(xRef)), negConf(0);
				Depth avgDepth(depth*posConf);
				unsigned nPosViews(0), nNegViews(0);
				unsigned n(N);
				do {
					const Depth d(depthMaps[--n](xRef));
					if (d == 0) {
						if (nPosViews + nNegViews + n < nMinViews)
							goto DiscardDepth;
						continue;
					}
					ASSERT(d > 0);
					if (IsDepthSimilar(depth, d, thDepthDiff)) {
						// average similar depths
						const float c(confMaps[n](xRef));
						avgDepth += d*c;
						posConf += c;
						++nPosViews;
					} else {
						// penalize confidence
						if (depth > d) {
							// occlusion
							negConf += confMaps[n](xRef);
						} else {
							// free-space violation
							const DepthData& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].ID];
							const Camera& camera = depthData.images.First().camera;
							const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
							const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
							if (depthData.confMap.isInside(x)) {
								const float c(depthData.confMap(x));
								negConf += (c > 0 ? c : confMaps[n](xRef));
							} else
								negConf += confMaps[n](xRef);
						}
						++nNegViews;
					}
				} while (n);
				ASSERT(nPosViews+nNegViews >= nMinViews);
				// if enough good views and positive confidence...
				if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax)) {
					// consider this pixel an inlier
					newDepthMap(xRef) = avgDepth;
					newConfMap(xRef) = posConf - negConf;
				} else {
					// consider this pixel an outlier
					DiscardDepth:
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					#if TD_VERBOSE != TD_VERBOSE_OFF
					++nDiscarded;
					#endif
				}
			}
		}
	} else {
		// remove depth if it does not agree with enough neighbors
		const float thDepthDiffStrict(OPTDENSE::fDepthDiffThreshold*0.8f);
		const unsigned nMinGoodViewsProc(75), nMinGoodViewsDeltaProc(65);
		const unsigned nDeltas(4);
		const unsigned nMinViewsDelta(nMinViews*(nDeltas-2));
		const ImageRef xDs[nDeltas] = { ImageRef(-1,0), ImageRef(1,0), ImageRef(0,-1), ImageRef(0,1) };
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// check if very similar with the neighbors projected to this pixel
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					unsigned n(N);
					do {
						const Depth d(depthMaps[--n](xRef));
						if (d > 0) {
							// valid view
							++nViews;
							if (IsDepthSimilar(depth, d, thDepthDiffStrict)) {
								// agrees with this neighbor
								++nGoodViews;
							}
						}
					} while (n);
					if (nGoodViews < nMinViews || nGoodViews < nViews*nMinGoodViewsProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// check if similar with the neighbors projected around this pixel
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					for (unsigned d=0; d<nDeltas; ++d) {
						const ImageRef xDRef(xRef+xDs[d]);
						unsigned n(N);
						do {
							const Depth d(depthMaps[--n](xDRef));
							if (d > 0) {
								// valid view
								++nViews;
								if (IsDepthSimilar(depth, d, thDepthDiff)) {
									// agrees with this neighbor
									++nGoodViews;
								}
							}
						} while (n);
					}
					if (nGoodViews < nMinViewsDelta || nGoodViews < nViews*nMinGoodViewsDeltaProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// enough good views, keep it
				newDepthMap(xRef) = depth;
				newConfMap(xRef) = depthDataRef.confMap(xRef);
			}
		}
	}
	if (!SaveDepthMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.dmap"), newDepthMap) ||
		!SaveConfidenceMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.cmap"), newConfMap))
		return false;

	DEBUG("Depth map %3u filtered using %u other images: %u/%u depths discarded (%s)",
		imageRef.GetID(), N, nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	return true;
} // FilterDepthMap
/*----------------------------------------------------------------*/


// fuse all depth-maps by simply projecting them in a 3D point cloud
// in the world coordinate space
void DepthMapsData::MergeDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	// estimate total number of 3D points that will be generated
	size_t nPointsEstimate(0);
	for (const DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			nPointsEstimate += (size_t)depthData.depthMap.size().area();

	// fuse all depth-maps
	size_t nDepthMaps(0), nDepths(0);
	pointcloud.ReservePoints(nPointsEstimate);

	size_t maxIdxsPerPoint = arrDepthData.size();

	// Create a flat array for each point with enough space to store a maximum number
	// of ids.
	pointcloud.ReservePointViewsSizeAndOffset(nPointsEstimate*maxIdxsPerPoint);

	if (bEstimateColor)
		pointcloud.ReserveColors(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.ReserveNormals(nPointsEstimate);

	// Each point "i" will have a set of up to maxIdxsPerPoint point view indexes.

	Util::Progress progress(_T("Merged depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();

	size_t pointViewImageOffset = 0;
	FOREACH(idxImage, arrDepthData) {
		TD_TIMER_STARTD();
		DepthData& depthData = arrDepthData[idxImage];
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		const DepthData::ViewData& image = depthData.GetView();
		const size_t nNumPointsPrev(pointcloud.NumPoints());

		size_t pointViewOffset = 0;
		Point3f normal;
		for (int i=0; i<depthData.depthMap.rows; ++i) {
			for (int j=0; j<depthData.depthMap.cols; ++j, pointViewOffset += maxIdxsPerPoint) {
				// ignore invalid depth
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				// create the corresponding 3D point
				const auto tmp = Cast<float>(image.camera.TransformPointI2W(Point3(Cast<float>(x), depth)));

				pointcloud.AddPoint(tmp);

				pointcloud.AddView(idxImage);

				if (bEstimateColor) {
					const auto& color = image.pImageData->image(x);
					pointcloud.AddColor(color);
				}
				if (bEstimateNormal) {
					depthData.GetNormal(x, normal);
					pointcloud.AddNormal(normal);
				}
				++nDepths;
			}
		}
		depthData.DecRef();
		++nDepthMaps;
		//ASSERT(pointcloud.points.size() == pointcloud.pointViews.size());
		DEBUG_ULTIMATE("Depths map for reference image %3u merged using %u depths maps: %u new points (%s)",
			idxImage, depthData.images.size()-1, (pointcloud.NumPoints())-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(idxImage+1);
	}
	GET_LOGCONSOLE().Play();
	progress.close();

	DEBUG_EXTRA("Depth-maps merged: %u depth-maps, %u depths, %u points (%d%%%%) (%s)",
		nDepthMaps, nDepths, pointcloud.NumPoints(), ROUND2INT(100.f*(pointcloud.NumPoints())/nDepths), TD_TIMER_GET_FMT().c_str());
} // MergeDepthMaps
/*----------------------------------------------------------------*/

// fuse all valid depth-maps in the same 3D point cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
#ifdef DPC_NEW_FUSING
#pragma optimize("", on) // JPB WIP BUG
void DepthMapsData::FuseDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	struct Proj {
		union {
			uint32_t idxPixel;
			struct {
				uint16_t x, y; // image pixel coordinates
			};
		};
		inline Proj() {}
		inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
		inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
		inline ImageRef GetCoord() const { return ImageRef(x,y); }
	};
	typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

	// find best connected images
	IndexScoreArr connections(0, scene.images.GetSize());
	size_t nPointsEstimate(0);
	bool bNormalMap(true);
	std::vector<bool> depthDataEmpty(arrDepthData.size());

	std::vector<TRMatrixBase<float>> imagesCameraRt;
	std::vector<Matrix3x4f> imagesCameraP;
	std::vector<Matrix4x4f> imagesCameraPt;
	imagesCameraRt.reserve(scene.images.size());

	FOREACH(i, scene.images) {
		DepthData& depthData = arrDepthData[i];
		imagesCameraRt.emplace_back(Cast<TRMatrixBase<float>>(scene.images[i].camera.R));
		imagesCameraP.emplace_back(Cast<float>(scene.images[i].camera.P));

		Matrix4x4 tmp = Matrix4x4::IDENTITY;
		for (auto r = 0; r < scene.images[i].camera.P.rows; ++r) { //3
			for (auto c = 0; c < scene.images[i].camera.P.cols; ++c) { //4
				tmp(r, c) = scene.images[i].camera.P(r, c);
			}
		}

		Matrix4x4 tmpt;
		for (auto r = 0; r < 4; ++r) {
			for (auto c = 0; c < 4; ++c) {
				tmpt(r, c) = tmp(c, r);
			}
		}
		tmpt(3, 3) = 0.; // Must be zero
		imagesCameraPt.emplace_back(Cast<float>(tmpt));

		depthDataEmpty[i] = depthData.IsEmpty();
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		IndexScore& connection = connections.AddEmpty();
		connection.idx = i;
		connection.score = (float)scene.images[i].neighbors.GetSize();
		nPointsEstimate += depthData.depthMap.area();
		if (depthData.normalMap.empty())
			bNormalMap = false;
	}
	connections.Sort();

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));
	std::atomic<size_t> nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());
	ProjsArr projsarr(0, nPointsEstimate);
	if (bEstimateNormal && !bNormalMap)
		bEstimateNormal = false;

	pointcloud.ReservePoints(nPointsEstimate);

	size_t maxIdxsPerPoint = std::min(16u, arrDepthData.size());
	// Create a flat array for each point with a generous amount of space
	// to store a large number of views and weights.
	pointcloud.ReservePointViewsSizeAndOffset(nPointsEstimate);
	pointcloud.ReservePointWeightsSizeAndOffset(nPointsEstimate);

	if (bEstimateColor)
		pointcloud.ReserveColors(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.ReserveNormals(nPointsEstimate);

		MEMORYSTATUS memState{};
	::GlobalMemoryStatus(&memState);
	const size_t bytesAvailable = memState.dwAvailPhys;

	// Split a quarter of what's left between points and views.
	const size_t elementSize =
		std::max(
			sizeof(decltype(pointcloud.pointViewsMemory)::value_type),
			sizeof(decltype(pointcloud.pointWeightsMemory)::value_type)
		);

	const size_t nElementsAvailableToUse = (bytesAvailable/elementSize)/4;
	pointcloud.ReservePointViewsMemory(nElementsAvailableToUse/4);
	pointcloud.ReservePointWeightsMemory(nElementsAvailableToUse/4);

	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
	GET_LOGCONSOLE().Pause();

	const int maxThreads = omp_get_max_threads();
	std::vector<PointCloudStreaming> pointCloudsByThread(maxThreads);

	const size_t nPointsEstimatePerThread = nPointsEstimate/maxThreads;
	const size_t nElementsAvailableToUsePerThread = (bytesAvailable/elementSize)/(4 * maxThreads);
	for (auto& pcs : pointCloudsByThread) {
		pcs.ReservePoints(nPointsEstimatePerThread);
		pcs.ReservePointViewsSizeAndOffset(nPointsEstimatePerThread);
		pcs.ReservePointWeightsSizeAndOffset(nPointsEstimatePerThread);
		if (bEstimateColor)
			pcs.ReserveColors(nPointsEstimatePerThread);
		if (bEstimateNormal)
			pcs.ReserveNormals(nPointsEstimatePerThread);
		pcs.ReservePointViewsMemory(nElementsAvailableToUsePerThread/4);
		pcs.ReservePointWeightsMemory(nElementsAvailableToUsePerThread/4);
	}

	for (int ci = 0; ci < connections.size(); ++ci) {
		auto* pConnection = connections.begin() + ci;
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor: depthData.neighbors) {
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.ID];
			if (!depthIdxs.empty())
				continue;
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			ASSERT(!depthDataB.IsEmpty());
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}
		ASSERT(!depthData.IsEmpty());
		const Image8U::Size sizeMap(depthData.depthMap.size());
		const Image& imageData = *depthData.images.First().pImageData;
		ASSERT(&imageData-scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const float confMapSentinel = 1.f;
		size_t confMapInc;

		const Point3 imageDataHBasis = 
				imageData.camera.TransformPointI2W(Point3(Point2f(1,0),1.))
				- imageData.camera.TransformPointI2W(Point3(Point2f(0,0),1.));

		const size_t nNumPointsPrev(pointcloud.NumPoints());
		const size_t numNeighbors = depthData.neighbors.size();

		std::vector<uint32_t> views;
		std::vector<float> weights;
		std::vector<Proj> projs;
				CLISTDEF0(Depth*) invalidDepths(0, 32);


// JPB WIP BUG
#pragma omp parallel for num_threads(maxThreads) private( views, weights, projs, invalidDepths)
		for (int i=0; i<sizeMap.height; ++i) {
			const int thread_idx = omp_get_thread_num();
			auto& pcs = pointCloudsByThread[thread_idx];

			const Depth* __restrict pDM = &depthData.depthMap(i, 0);
			cuint32_t* __restrict pDepthIdxs = &depthIdxs(i, 0);

		const float* __restrict pConfMap;
		bool confMapEmpty = depthData.confMap.empty();
		if (confMapEmpty) {
			pConfMap = &confMapSentinel;
			confMapInc = 0;
		} else {
			confMapInc = 1;
		}

			if (!confMapEmpty) {
				pConfMap = &depthData.confMap(i, 0);
			}

			const Normal* __restrict pNormalMap = &depthData.normalMap(i, 0);
			const Pixel8U* __restrict pImage = &imageData.image(i, 0);

#if 1
			Point3 imagePoint =
				imageData.camera.TransformPointI2W(Point3(Point2f(0,i),1.)) - imageData.camera.C;
#else
			double x0 = (0.-imageData.camera.K(0,2))*1./imageData.camera.K(0,0);
			double y0 = (i-imageData.camera.K(1,2))*1./imageData.camera.K(1,1);
			double z0 = 1.f;
			Point3 point0 = (imageData.camera.R.t() * Point3(x0,y0,z0)); // + imageData.camera.C;

			double x1 = (1.-imageData.camera.K(0,2))*1./imageData.camera.K(0,0);
			double y1 = (i-imageData.camera.K(1,2))*1./imageData.camera.K(1,1);
			double z1 = 1.f;
			Point3 point1 = (imageData.camera.R.t() * Point3(x1,y1,z1)); // + imageData.camera.C;

			Point3 delta = (point1-point0);
#endif
			const double invImageDataCameraK00 = 1./imageData.camera.K(0,0);
			const double invImageDataCameraK11 = 1./imageData.camera.K(1,1);
			const double imageDataCameraK02 = imageData.camera.K(0,2);
			const double imageDataCameraK12 = imageData.camera.K(1,2);
			double pointXNoDepthPreTransform = -imageDataCameraK02*invImageDataCameraK00;
			double pointXNoDepthPreTransformDelta = invImageDataCameraK00;
			double pointYNoDepthPreTransform = (i-imageDataCameraK12)*invImageDataCameraK11;

			struct Estimate_t {
				Estimate_t(const Image* _image, const ImageRef& _ref, float _confidenceB) :
					image(_image),
					ref(_ref),
					confidenceB(_confidenceB)
				{}

				const Image* image;
				ImageRef ref;
				float confidenceB;
			};
			boost::container::small_vector<Estimate_t, 16> estimateRefs;

			int localDepths = 0;
			for (int j=0; j<sizeMap.width; ++j, pConfMap += confMapInc, imagePoint += imageDataHBasis, pointXNoDepthPreTransform += pointXNoDepthPreTransformDelta) {
				//const ImageRef x(j,i);
				const Depth depth(pDM[j]); //depthData.depthMap(x));
				if (depth == 0)
					continue;
				++localDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = pDepthIdxs[j]; //depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point

				idxPoint = (int) pcs.NumPoints();

				Point3f point;

				const ImageRef x(j,i);
				//point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth)); // JPB WIP BUG
#if 0
						return TPoint3<TYPE>(
			TYPE((X.x-K(0,2))*X.z/K(0,0)),
			TYPE((X.y-K(1,2))*X.z/K(1,1)),
			X.z );
#endif
				const double pointXWithDepth = pointXNoDepthPreTransform*depth;
				//const double pointXWithDepth2 = (j-imageData.camera.K(0, 2))*depth/imageData.camera.K(0, 0);
				const double pointYWithDepth = pointYNoDepthPreTransform*depth;
				//const double pointYWithDepth2 = (i-imageData.camera.K(1, 2))*depth/imageData.camera.K(1, 1);
				const double pointZWithDepth = depth;

				point.x =
					imageData.camera.R[0*3+0] * pointXWithDepth
					+	imageData.camera.R[1*3+0] * pointYWithDepth
					+	imageData.camera.R[2*3+0] * pointZWithDepth
					+ imageData.camera.C.x;

				point.y =
					imageData.camera.R[0*3+1] * pointXWithDepth
					+	imageData.camera.R[1*3+1] * pointYWithDepth
					+	imageData.camera.R[2*3+1] * pointZWithDepth
					+ imageData.camera.C.y;

				point.z =
					imageData.camera.R[0*3+2] * pointXWithDepth
					+	imageData.camera.R[1*3+2] * pointYWithDepth
					+	imageData.camera.R[2*3+2] * pointZWithDepth
					+ imageData.camera.C.z;

				//point = (imageData.camera.R.t() * Point3(xx, yy, zz)) + imageData.camera.C;

				// pViews->Insert(idxImage);
				views.clear();
				views.push_back(idxImage);

				weights.clear();
				REAL confidence = Conf2Weight(*pConfMap, depth);
				weights.push_back(confidence);

				projs.clear();
				//REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap.empty() ? 1.f : depthData.confMap(x),depth)));
				//	REAL confidence(pWeights->emplace_back(Conf2Weight(*pConfMap,depth)));

				projs.emplace_back(x);

				//const PointCloud::Normal normal(bNormalMap ? Cast<Normal::Type>(imageData.camera.R.t()*Cast<REAL>(depthData.normalMap(x))) : Normal(0,0,-1));
				PointCloud::Normal normal;
				if (bNormalMap) {
					const Normal& n = pNormalMap[j];
					normal.x =
						imageData.camera.R[0*3+0] * n.x
						+	imageData.camera.R[1*3+0] * n.y
						+	imageData.camera.R[2*3+0] * n.z;

					normal.y =
						imageData.camera.R[0*3+1] * n.x
						+	imageData.camera.R[1*3+1] * n.y
						+	imageData.camera.R[2*3+1] * n.z;

					normal.z =
						imageData.camera.R[0*3+2] * n.x
						+	imageData.camera.R[1*3+2] * n.y
						+	imageData.camera.R[2*3+2] * n.z;
				} else {
					normal ={ 0.f, 0.f, -1.f };
				}
				ASSERT(ISEQUAL(norm(normal), 1.f));
				// check the projection in the neighbor depth-maps
				Point3 X(point*confidence);
				float origConfidence = confidence;
				PointCloud::Normal N(normal*confidence);
				invalidDepths.Empty(); // JPB Make this a boost small_vector sized to sizeMap.width

#if 1
				estimateRefs.clear();

				_Data vPointX = _Set(point.x);
				_Data vPointY = _Set(point.y);
				_Data vPointZ = _Set(point.z);
				for (int i = 0, cnt = depthData.neighbors.size(); i < cnt; ++i) {
					const auto pNeighbor = &depthData.neighbors[i];

					const IIndex idxImageB(pNeighbor->ID);
					const Image& imageDataB = scene.images[idxImageB];
					const auto& imageDataBCameraPt = imagesCameraPt[idxImageB];
					DepthData& depthDataB = arrDepthData[idxImageB];
					ASSERT(!depthDataB.IsEmpty());
					DepthMap& depthMapB = depthDataB.depthMap;

					_Data col0 = _Load(&imageDataBCameraPt[0]);
					_Data col1 = _Load(&imageDataBCameraPt[4]);
					_Data col2 = _Load(&imageDataBCameraPt[8]);
					_Data col3 = _Load(&imageDataBCameraPt[12]); // col3.w is always 0

					_Data col0_point = _Mul(col0, vPointX); // ptx, pty, ptz
					_Data col1_point = _Mul(col1, vPointY);
					_Data col2_point = _Mul(col2, vPointZ);

					_Data xyz_1 = _Add(col0_point, col1_point);
					_Data xyz_2 = _Add(col2_point, col3);
					_Data xyz0 = _Add(xyz_1, xyz_2);
					_Data zzz = _Splat(xyz0, 2);
					_Data zzz_wh10 = _Mul(zzz, _SetN(depthMapB.width()-1, depthMapB.height()-1, 1.f, 0.f));

#if 0
					const auto& imageDataBCameraP = imagesCameraP[idxImageB];
					float ptz =
						imageDataBCameraP[2*4+0] * point.x
						+	imageDataBCameraP[2*4+1] * point.y
						+	imageDataBCameraP[2*4+2] * point.z
						+	imageDataBCameraP[2*4+3];

					if (ptz <= 0)
						continue;

					float ptx =
						imageDataBCameraP[0*4+0] * point.x
						+	imageDataBCameraP[0*4+1] * point.y
						+	imageDataBCameraP[0*4+2] * point.z
						+	imageDataBCameraP[0*4+3];

					float pty =
						imageDataBCameraP[1*4+0] * point.x
						+	imageDataBCameraP[1*4+1] * point.y
						+	imageDataBCameraP[1*4+2] * point.z
						+	imageDataBCameraP[1*4+3];
#endif

					_Data result = _CmpGT(xyz0, zzz_wh10);		// x > z*w, y > z*h, z > z, 0 > 0
					_Data result2 = _CmpLT(xyz0, _SetZero());	// x < 0, y < 0, z < 0, 0 < 0 
					_Data orResult = _Or(result, result2);
					if (!AllZerosI(_CastIF(orResult))) {
						continue;
					}

					//_Data imageDataBCameraRow2 = _Load(imageDataB.camera.P + 2*4);

#if 1
					_Data invZZZ = _Div({ 1.f, 1.f, 1.f, 1.f }, zzz);
					xyz0 = _Mul(xyz0, invZZZ);
					float ptz = _vFirst(zzz);
					_DataI xyz0AsInt = _ConvertIF(xyz0);
					const ImageRef xB(_AsArrayI(xyz0AsInt, 0), _AsArrayI(xyz0AsInt, 1));
#else
				FOREACHPTR(pNeighbor, depthData.neighbors) {
					const IIndex idxImageB(pNeighbor->idx.ID);
					const Image& imageDataB = scene.images[idxImageB];
					const auto& imageDataBCameraP = imagesCameraP[idxImageB];

					float ptz =
						imageDataBCameraP[2*4+0] * point.x
						+	imageDataBCameraP[2*4+1] * point.y
						+	imageDataBCameraP[2*4+2] * point.z
						+	imageDataBCameraP[2*4+3];

					if (ptz <= 0)
						continue;

					float ptx =
						imageDataBCameraP[0*4+0] * point.x
						+	imageDataBCameraP[0*4+1] * point.y
						+	imageDataBCameraP[0*4+2] * point.z
						+	imageDataBCameraP[0*4+3];

					float pty =
						imageDataBCameraP[1*4+0] * point.x
						+	imageDataBCameraP[1*4+1] * point.y
						+	imageDataBCameraP[1*4+2] * point.z
						+	imageDataBCameraP[1*4+3];

					DepthData& depthDataB = arrDepthData[idxImageB];
					ASSERT(!depthDataB.IsEmpty());
					DepthMap& depthMapB = depthDataB.depthMap;
					if (
						(ptx >= (depthMapB.width()-1.f) * ptz)
						| (ptx < 0.f)
						| (pty >= (depthMapB.height()-1.f) * ptz)
						| (pty < 0.f)
						) {
						continue;
					}
					float invZ = 1.f / ptz;
					const ImageRef xB(ROUND2INT(ptx*invZ), ROUND2INT(pty*invZ));
#endif


					Depth& depthB = depthMapB.pix(xB);
					if (depthB == 0)
						continue;

					uint32_t& idxPointB = (arrDepthIdx[idxImageB]).pix(xB);
					if (idxPointB != NO_ID)
						continue;

					if (FastAbsS(ptz-depthB) < OPTDENSE::fDepthDiffThreshold *ptz) {
					//if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// check if normals agree
#if 1
					PointCloud::Normal normalB;
					if (bNormalMap) {
						const Normal& nb = depthDataB.normalMap.pix(xB);
						const TRMatrixBase<float>& imageCameraRt = imagesCameraRt[idxImageB];
						normalB.x =
							imageCameraRt[0*3+0] * nb.x
							+	imageCameraRt[1*3+0] * nb.y
							+	imageCameraRt[2*3+0] * nb.z;

						normalB.y =
							imageCameraRt[0*3+1] * nb.x
							+	imageCameraRt[1*3+1] * nb.y
							+	imageCameraRt[2*3+1] * nb.z;

						normalB.z =
							imageCameraRt[0*3+2] * nb.x
							+	imageCameraRt[1*3+2] * nb.y
							+	imageCameraRt[2*3+2] * nb.z;
					} else {
						normalB ={ 0.f, 0.f, -1.f };
					}
#else
						const PointCloud::Normal normalB(bNormalMap ? Cast<Normal::Type>(imageDataB.camera.R.t()*Cast<REAL>(depthDataB.normalMap.pix(xB))) : Normal(0,0,-1));
#endif
						ASSERT(ISEQUAL(norm(normalB), 1.f));
						if (normal.dot(normalB) > normalError) {
							// add view to the 3D point
							//ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
							const float confidenceB(Conf2Weight(depthDataB.confMap.empty() ? 1.f : depthDataB.confMap.pix(xB),depthB));

							// Uses binary search to determine index to insert to
#if 1
							auto it = views.insert(std::upper_bound(std::begin(views), std::end(views), idxImageB), idxImageB);
							const auto idx = std::distance(std::begin(views), it);
							weights.insert(std::begin(weights) + idx, confidenceB);
#else
							const IIndex idx(pViews->InsertSort(idxImageB));

							// Real insertion sorts at idx
							pWeights->InsertAt(idx, confidenceB);
#endif
							projs.insert(std::begin(projs) + idx, Proj(xB));
							//pPointProjs->InsertAt(idx, Proj(xB));

							idxPointB = idxPoint;

							//X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);

							double cx = ((((double) xB.x)-imageDataB.camera.K(0,2))*depthB/imageDataB.camera.K(0,0));
							double cy = ((((double) xB.y)-imageDataB.camera.K(1,2))*depthB/imageDataB.camera.K(1,1));
							double cz = depthB;

							const auto& rot = imageDataB.camera.R;

							auto offsetX = rot(0, 0)*cx + rot(1, 0)*cy + rot(2, 0)*cz;
							auto offsetY = rot(0, 1)*cx + rot(1, 1)*cy + rot(2, 1)*cz;
							auto offsetZ = rot(0, 2)*cx + rot(1, 2)*cy + rot(2, 2)*cz;

							offsetX += imageDataB.camera.C.x;
							offsetY += imageDataB.camera.C.y;
							offsetZ += imageDataB.camera.C.z;

							offsetX *= confidenceB;
							offsetY *= confidenceB;
							offsetZ *= confidenceB;

							X.x += offsetX;
							X.y += offsetY;
							X.z += offsetZ;

							if (bEstimateColor) {
								estimateRefs.emplace_back(&imageDataB, xB, confidenceB);
							}
							if (bEstimateNormal)
								N += normalB*confidenceB;
							confidence += confidenceB;
							
							continue;
						}
					}

					if (ptz < depthB) {
						// discard depth
						invalidDepths.Insert(&depthB);
					}
				}
#endif

				if (views.size() < nMinViewsFuse) {
					// remove point
					for (int v = 0; v < views.size(); ++v) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(projs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && (arrDepthIdx[idxImageB]).pix(x).idx != NO_ID);
						(arrDepthIdx[idxImageB]).pix(x).idx = NO_ID;
					}
				} else {
					// this point is valid, store it
					const auto& baseColor = pImage[j];
					float r = baseColor.r;
					float g = baseColor.g;
					float b = baseColor.b;

					r *= origConfidence;
					g *= origConfidence;
					b *= origConfidence;

					// Handle the deferred image contributions.
					for (const auto& er : estimateRefs) {
						//C += Cast<float>(imageDataB.image.pix(xB))*confidenceB;

						const Pixel8U& pixel = er.image->image.pix(er.ref);

						float imageR = pixel.r;
						float imageG = pixel.g;
						float imageB = pixel.b;

						imageR *= er.confidenceB;
						imageG *= er.confidenceB;
						imageB *= er.confidenceB;

						r += imageR;
						g += imageG;
						b += imageB;
					}

					const REAL nrm(REAL(1)/confidence);
					point = X*nrm;
					ASSERT(ISFINITE(point));

					pcs.AddPoint(point);
					pcs.AddViews(std::begin(views), std::end(views));
					pcs.AddWeights(std::begin(weights), std::end(weights));

					if (bEstimateColor) {
						r *= nrm;
						g *= nrm;
						b *= nrm;
						pcs.AddColor(Pixel8U(_cvt_ftoi_fast(r), _cvt_ftoi_fast(g), _cvt_ftoi_fast(b)));
						//pointcloud.colors.AddConstruct((C*(float)nrm).cast<uint8_t>());
					}
					if (bEstimateNormal) {
						ProjArr tmp;
						for (const auto& i : projs) {
							tmp.Insert(i);
						}
						projsarr.Insert(tmp);
						const Point3f nn(normalized(N* (float)nrm));
						pcs.AddNormal(nn);
						//AddConstruct(normalized(N* (float)nrm));
					}

					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}

			nDepths += localDepths;
		}

		// JPB WIP ASSERT(pointcloud.points.GetSize() == pointcloud.pointViews.GetSize() && pointcloud.points.GetSize() == pointcloud.pointWeights.GetSize() && pointcloud.points.GetSize() == projs.GetSize());
		DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize()-1, (pointcloud.NumPoints())-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(pConnection-connections.Begin());
	}
	
	// Now combine the per-thread clouds into a single cloud.
	for (auto& pcs : pointCloudsByThread) {
		// Copy the points
		std::copy(std::begin(pcs.pointsXYZ), std::end(pcs.pointsXYZ), std::back_inserter(pointcloud.pointsXYZ));
		// Copy the views.
		std::copy(std::begin(pcs.pointViewsSizes), std::end(pcs.pointViewsSizes), std::back_inserter(pointcloud.pointViewsSizes));
		size_t adder = pointcloud.pointViewsMemory.size();
		for (auto off : pcs.pointViewsOffsets) {
			pointcloud.pointViewsOffsets.push_back(off+adder);
		}
		for (auto v : pcs.pointViewsMemory) {
			pointcloud.pointViewsMemory.push_back(v);
		}
		// Copy the weights.
		std::copy(std::begin(pcs.pointWeightsSizes), std::end(pcs.pointWeightsSizes), std::back_inserter(pointcloud.pointWeightsSizes));
		adder = pointcloud.pointWeightsMemory.size();
		for (auto off : pcs.pointWeightsOffsets) {
			pointcloud.pointWeightsOffsets.push_back(off+adder);
		}
		for (auto v : pcs.pointWeightsMemory) {
			pointcloud.pointWeightsMemory.push_back(v);
		}
		if (bEstimateColor) {
			// Copy the colors.
			std::copy(std::begin(pcs.colorsRGB), std::end(pcs.colorsRGB), std::back_inserter(pointcloud.colorsRGB));
		}
		if (bEstimateNormal) {
			// Copy the normals.
			std::copy(std::begin(pcs.normalsXYZ), std::end(pcs.normalsXYZ), std::back_inserter(pointcloud.normalsXYZ));
		}
	}


	
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	size_t printDepths = nDepths;
	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), printDepths, pointcloud.NumPoints(), ROUND2INT((100.f*(pointcloud.NumPoints()))/nDepths), TD_TIMER_GET_FMT().c_str());

	if (bEstimateNormal && pointcloud.PointStream() && pointcloud.NormalStream()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		const int64_t nPoints(pointcloud.NumPoints());
		pointcloud.ReserveNormals(nPoints);
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			const float* firstWeight = pointcloud.WeightsStream(i);
			const size_t numWeights = pointcloud.WeightsStreamSize(i);
			ASSERT(numWeights);
			IIndex idxView(0);
			float bestWeight = *firstWeight;
			for (IIndex idx=1; idx<numWeights; ++idx) {
				const float& weight = firstWeight[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.ViewsStream(i)[idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			depthData.GetNormal(projsarr[i][idxView].GetCoord(), (Point3f&) (pointcloud.NormalStream()[i*3]));
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.NumPoints(), TD_TIMER_GET_FMT().c_str());
	}

	// release all depth-maps
	for (DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			depthData.DecRef();
} // FuseDepthMaps
// #pragma optimize("", on) // JPB WIP BUG
#else
void DepthMapsData::FuseDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	struct Proj {
		union {
			uint32_t idxPixel;
			struct {
				uint16_t x, y; // image pixel coordinates
			};
		};
		inline Proj() {}
		inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
		inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
		inline ImageRef GetCoord() const { return ImageRef(x,y); }
	};
	typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

	// find best connected images
	IndexScoreArr connections(0, scene.images.GetSize());
	size_t nPointsEstimate(0);
	bool bNormalMap(true);
	FOREACH(i, scene.images) {
		DepthData& depthData = arrDepthData[i];
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		IndexScore& connection = connections.AddEmpty();
		connection.idx = i;
		connection.score = (float)scene.images[i].neighbors.GetSize();
		nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));
		if (depthData.normalMap.empty())
			bNormalMap = false;
	}
	connections.Sort();

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());
	ProjsArr projs(0, nPointsEstimate);
	if (bEstimateNormal && !bNormalMap)
		bEstimateNormal = false;
	// JPB WIP BUG pointcloud.points.reserve(nPointsEstimate);
	// JPB WIP BUG pointcloud.pointViews.reserve(nPointsEstimate);
	// JPB WIP BUG pointcloud.pointWeights.reserve(nPointsEstimate);
	// JPB WIP BUG if (bEstimateColor)
	// JPB WIP BUG 	pointcloud.colors.Reserve(nPointsEstimate);
	// JPB WIP BUG if (bEstimateNormal)
	// JPB WIP BUG 	pointcloud.normals.Reserve(nPointsEstimate);
	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
	GET_LOGCONSOLE().Pause();
	FOREACHPTR(pConnection, connections) {
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor: depthData.neighbors) {
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.ID];
			if (!depthIdxs.empty())
				continue;
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (depthDataB.IsEmpty())
				continue;
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}
		ASSERT(!depthData.IsEmpty());
		const Image8U::Size sizeMap(depthData.depthMap.size());
		const Image& imageData = *depthData.images.First().pImageData;
		ASSERT(&imageData-scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.NumPoints());
		for (int i=0; i<sizeMap.height; ++i) {
			for (int j=0; j<sizeMap.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.NumPoints();
				PointCloud::Point point;
				//pointcloud.pointsXYZ..AddEmpty();
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				PointCloud::ViewArr views;
				//= pointcloud.pointViews.AddEmpty();
				views.Insert(idxImage);
				PointCloud::WeightArr weights;
				//= pointcloud.pointWeights.AddEmpty();
				REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap.empty() ? 1.f : depthData.confMap(x),depth)));
				ProjArr& pointProjs = projs.AddEmpty();
				pointProjs.Insert(Proj(x));
				const PointCloud::Normal normal(bNormalMap ? Cast<Normal::Type>(imageData.camera.R.t()*Cast<REAL>(depthData.normalMap(x))) : Normal(0,0,-1));
				ASSERT(ISEQUAL(norm(normal), 1.f));
				// check the projection in the neighbor depth-maps
				Point3 X(point*confidence);
				Pixel32F C(Cast<float>(imageData.image(x))*confidence);
				PointCloud::Normal N(normal*confidence);
				invalidDepths.Empty();
				FOREACHPTR(pNeighbor, depthData.neighbors) {
					const IIndex idxImageB(pNeighbor->ID);
					DepthData& depthDataB = arrDepthData[idxImageB];
					if (depthDataB.IsEmpty())
						continue;
					const Image& imageDataB = scene.images[idxImageB];
					const Point3f pt(imageDataB.camera.ProjectPointP3(point));
					if (pt.z <= 0)
						continue;
					const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					if (idxPointB != NO_ID)
						continue;
					if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// check if normals agree
						const PointCloud::Normal normalB(bNormalMap ? Cast<Normal::Type>(imageDataB.camera.R.t()*Cast<REAL>(depthDataB.normalMap(xB))) : Normal(0,0,-1));
						ASSERT(ISEQUAL(norm(normalB), 1.f));
						if (normal.dot(normalB) > normalError) {
							// add view to the 3D point
							ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
							const float confidenceB(Conf2Weight(depthDataB.confMap.empty() ? 1.f : depthDataB.confMap(xB),depthB));
							const IIndex idx(views.InsertSort(idxImageB));
							weights.InsertAt(idx, confidenceB);
							pointProjs.InsertAt(idx, Proj(xB));
							idxPointB = idxPoint;
							X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
							if (bEstimateColor)
								C += Cast<float>(imageDataB.image(xB))*confidenceB;
							if (bEstimateNormal)
								N += normalB*confidenceB;
							confidence += confidenceB;
							continue;
						}
					}
					if (pt.z < depthB) {
						// discard depth
						invalidDepths.Insert(&depthB);
					}
				}
				if (views.GetSize() < nMinViewsFuse) {
					// remove point
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					projs.RemoveLast();
					//pointcloud.pointWeights.RemoveLast();
					//pointcloud.pointViews.RemoveLast();
					//pointcloud.points.RemoveLast();
				} else {
					// this point is valid, store it
					const REAL nrm(REAL(1)/confidence);
					point = X*nrm;
					ASSERT(ISFINITE(point));
					pointcloud.pointsXYZ.push_back(point.x);
					pointcloud.pointsXYZ.push_back(point.y);
					pointcloud.pointsXYZ.push_back(point.z);

					size_t index = pointcloud.pointViewsMemory.size();
					pointcloud.pointViewsOffsets.push_back((int)index);

					pointcloud.pointViewsSizes.push_back((int) views.size());
					for (auto& v : views) {
						pointcloud.pointViewsMemory.push_back(v);
					}

					size_t index2 = pointcloud.pointWeightsMemory.size();
					pointcloud.pointWeightsOffsets.push_back((int)index2);

					pointcloud.pointWeightsSizes.push_back((int) weights.size());
					for (auto& w : weights) {
						pointcloud.pointViewsMemory.push_back(w);
					}

					if (bEstimateColor) {
						Pixel8U col = (C * (float)nrm).cast<uint8_t>();
						uint8_t rgb[3];
						rgb[0] = col.r;
						rgb[1] = col.g;
						rgb[2] = col.b;
						pointcloud.AddColor(rgb);
					}
					if (bEstimateNormal) {
						auto normal = normalized(N * (float)nrm);
						pointcloud.AddNormal(normal);
					}
					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
		ASSERT(pointcloud.points.GetSize() == pointcloud.pointViews.GetSize() && pointcloud.points.GetSize() == pointcloud.pointWeights.GetSize() && pointcloud.NumPoints() == projs.GetSize());
		DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize()-1, pointcloud.NumPoints()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(pConnection-connections.Begin());
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), nDepths, pointcloud.NumPoints(), ROUND2INT((100.f*pointcloud.NumPoints())/nDepths), TD_TIMER_GET_FMT().c_str());

	if (bEstimateNormal) {
		throw std::exception("Unsupporeted");
	}
#if 0
	if (bEstimateNormal && !pointcloud.points.IsEmpty() && pointcloud.normals.IsEmpty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		pointcloud.normals.Resize(pointcloud.points.GetSize());
		const int64_t nPoints((int64_t)pointcloud.points.GetSize());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.IsEmpty());
			IIndex idxView(0);
			float bestWeight = weights.First();
			for (IIndex idx=1; idx<weights.GetSize(); ++idx) {
				const PointCloud::Weight& weight = weights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.points.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
#endif
	// release all depth-maps
	for (DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			depthData.DecRef();
} // FuseDepthMaps

#endif

/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

DenseDepthMapData::DenseDepthMapData(Scene& _scene, int _nFusionMode)
	: scene(_scene), depthMaps(_scene), idxImage(0), sem(1), nEstimationGeometricIter(-1), nFusionMode(_nFusionMode)
{
	if (nFusionMode < 0) {
		STEREO::SemiGlobalMatcher::CreateThreads(scene.nMaxThreads);
		if (nFusionMode == -1)
			OPTDENSE::nOptimize = 0;
	}
}
DenseDepthMapData::~DenseDepthMapData()
{
	if (nFusionMode < 0)
		STEREO::SemiGlobalMatcher::DestroyThreads();
}

void DenseDepthMapData::SignalCompleteDepthmapFilter()
{
	ASSERT(idxImage > 0);
	if (Thread::safeDec(idxImage) == 0)
		sem.Signal((unsigned)images.GetSize()*2);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);

bool Scene::DenseReconstruction(int nFusionMode, bool bCrop2ROI, float fBorderROI)
{
	DenseDepthMapData data(*this, nFusionMode);

	// estimate depth-maps
	if (!ComputeDepthMaps(data))
		return false;
	if (ABS(nFusionMode) == 1)
		return true;

	// fuse all depth-maps
	pointcloud.Release();
	if (OPTDENSE::nMinViewsFuse < 2) {
		// merge depth-maps
		data.depthMaps.MergeDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
	} else {
		// fuse depth-maps
		data.depthMaps.FuseDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
	}
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print number of points with 3+ views
		size_t nPoints1m(0), nPoints2(0), nPoints3p(0);
		for (size_t i = 0, cnt = pointcloud.NumPoints(); i < cnt; ++i) {
			switch (pointcloud.ViewsStreamSize(i))
			{
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			default:
				++nPoints3p;
			}
		}
		VERBOSE("Dense point-cloud composed of:\n\t%u points with 1- views\n\t%u points with 2 views\n\t%u points with 3+ views", nPoints1m, nPoints2, nPoints3p);
	}
	#endif

	if (!pointcloud.IsEmpty()) {
		if (bCrop2ROI && IsBounded()) {
#if 1 // JPB WIP BUG
			throw std::runtime_error("Unsupported");
#else
			TD_TIMER_START();
			const size_t numPoints = pointcloud.GetSize();
			const OBB3f ROI(fBorderROI == 0 ? obb : (fBorderROI > 0 ? OBB3f(obb).EnlargePercent(fBorderROI) : OBB3f(obb).Enlarge(-fBorderROI)));
			pointcloud.RemovePointsOutside(ROI);
			VERBOSE("Point-cloud trimmed to ROI: %u points removed (%s)",
				numPoints-pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
#endif
		}
		if (!pointcloud.ColorStream() && OPTDENSE::nEstimateColors == 1)
			EstimatePointColors(images, pointcloud);
		if (!pointcloud.NormalStream() && OPTDENSE::nEstimateNormals == 1)
			EstimatePointNormals(images, pointcloud);
	}

	if (OPTDENSE::bRemoveDmaps) {
		// delete all depth-map files
		FOREACH(i, images) {
			const DepthData& depthData = data.depthMaps.arrDepthData[i];
			if (!depthData.IsValid())
				continue;
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
		}
	}
	return true;
} // DenseReconstruction
/*----------------------------------------------------------------*/

// do first half of dense reconstruction: depth map computation
// results are saved to "data"
bool Scene::ComputeDepthMaps(DenseDepthMapData& data)
{
	// JPB WIP BUG Called first pass too!!!
	// compute point-cloud from the existing mesh
	if (!mesh.IsEmpty() && !ImagesHaveNeighbors()) {
		SampleMeshWithVisibility();
		mesh.Release();
	}
	
	// compute point-cloud from the existing mesh
	if (IsEmpty() && !ImagesHaveNeighbors()) {
		VERBOSE("warning: empty point-cloud, rough neighbor views selection based on image pairs baseline");
		EstimateNeighborViewsPointCloud();
	}

	{
	// maps global view indices to our list of views to be processed
	IIndexArr imagesMap;

	// prepare images for dense reconstruction (load if needed)
	{
		TD_TIMER_START();
		data.images.Reserve(images.GetSize());
		imagesMap.Resize(images.GetSize());
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for shared(data, bAbort)
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {
		#endif
			// skip invalid, uncalibrated or discarded images
			Image& imageData = images[idxImage];
			if (!imageData.IsValid()) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				imagesMap[idxImage] = NO_ID;
				continue;
			}
			// map image index
			#ifdef DENSE_USE_OPENMP
			#pragma omp critical
			#endif
			{
				imagesMap[idxImage] = data.images.GetSize();
				data.images.Insert(idxImage);
			}
			// reload image at the appropriate resolution
			unsigned nResolutionLevel(OPTDENSE::nResolutionLevel);
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(nResolutionLevel, OPTDENSE::nMinResolution, OPTDENSE::nMaxResolution));
			if (!imageData.ReloadImage(nMaxResolution)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return false;
				#endif
			}
			imageData.UpdateCamera(platforms);
			// print image camera
			DEBUG_ULTIMATE("K%d = \n%s", idxImage, cvMat2String(imageData.camera.K).c_str());
			DEBUG_LEVEL(3, "R%d = \n%s", idxImage, cvMat2String(imageData.camera.R).c_str());
			DEBUG_LEVEL(3, "C%d = \n%s", idxImage, cvMat2String(imageData.camera.C).c_str());
		}
		#ifdef DENSE_USE_OPENMP
		if (bAbort || data.images.IsEmpty()) {
		#else
		if (data.images.IsEmpty()) {
		#endif
			VERBOSE("error: preparing images for dense reconstruction failed (errors loading images)");
			return false;
		}
		VERBOSE("Preparing images for dense reconstruction completed: %d images (%s)", images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// select images to be used for dense reconstruction
	{
		TD_TIMER_START();
		// for each image, find all useful neighbor views
		IIndexArr invalidIDs;
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for shared(data, invalidIDs)
		for (int_t ID=0; ID<(int_t)data.images.GetSize(); ++ID) {
			const IIndex idx((IIndex)ID);
		#else
		FOREACH(idx, data.images) {
		#endif
			const IIndex idxImage(data.images[idx]);
			ASSERT(imagesMap[idxImage] != NO_ID);
			DepthData& depthData(data.depthMaps.arrDepthData[idxImage]);
			if (!data.depthMaps.SelectViews(depthData)) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				invalidIDs.InsertSort(idx);
			}
		}
		RFOREACH(i, invalidIDs) {
			const IIndex idx(invalidIDs[i]);
			imagesMap[data.images.Last()] = idx;
			imagesMap[data.images[idx]] = NO_ID;
			data.images.RemoveAt(idx);
		}
		// globally select a target view for each reference image
		if (OPTDENSE::nNumViews == 1 && !data.depthMaps.SelectViews(data.images, imagesMap, data.neighborsMap)) {
			VERBOSE("error: no valid images to be dense reconstructed");
			return false;
		}
		ASSERT(!data.images.IsEmpty());
		VERBOSE("Selecting images for dense reconstruction completed: %d images (%s)", data.images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
	}

	#ifdef _USE_CUDA
	// initialize CUDA
	if (SEACAVE::CUDA::desiredDeviceID >= -1 && data.nFusionMode >= 0) {
		data.depthMaps.pmCUDA = new CUDA::PatchMatch(SEACAVE::CUDA::desiredDeviceID);
		if (SEACAVE::CUDA::devices.IsEmpty())
			data.depthMaps.pmCUDA.Release();
		else
			data.depthMaps.pmCUDA->Init(false);
	}
	#endif // _USE_CUDA

	// initialize the queue of images to be processed
	const int nOptimize(OPTDENSE::nOptimize);
	if (OPTDENSE::nEstimationGeometricIters && data.nFusionMode >= 0)
		OPTDENSE::nOptimize = 0;
	data.idxImage = 0;
	ASSERT(data.events.IsEmpty());
	data.events.AddEvent(new EVTProcessImage(0));
	// start working threads
	data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());
	GET_LOGCONSOLE().Pause();
	if (nMaxThreads > 1) {
		// multi-thread execution
		cList<SEACAVE::Thread> threads(2);
		FOREACHPTR(pThread, threads)
			pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
		FOREACHPTR(pThread, threads)
			pThread->join();
	} else {
		// single-thread execution
		DenseReconstructionEstimate((void*)&data);
	}
	GET_LOGCONSOLE().Play();
	if (!data.events.IsEmpty())
		return false;
	data.progress.Release();

	if (data.nFusionMode >= 0) {
		#ifdef _USE_CUDA
		// initialize CUDA
		if (data.depthMaps.pmCUDA && OPTDENSE::nEstimationGeometricIters) {
			data.depthMaps.pmCUDA->Release();
			data.depthMaps.pmCUDA->Init(true);
		}
		#endif // _USE_CUDA
		while (++data.nEstimationGeometricIter < (int)OPTDENSE::nEstimationGeometricIters) {
			// initialize the queue of images to be geometric processed
			if (data.nEstimationGeometricIter+1 == (int)OPTDENSE::nEstimationGeometricIters)
				OPTDENSE::nOptimize = nOptimize;
			data.idxImage = 0;
			ASSERT(data.events.IsEmpty());
			data.events.AddEvent(new EVTProcessImage(0));
			// start working threads
			data.progress = new Util::Progress("Geometric-consistent estimated depth-maps", data.images.GetSize());
			GET_LOGCONSOLE().Pause();
			if (nMaxThreads > 1) {
				// multi-thread execution
				cList<SEACAVE::Thread> threads(2);
				FOREACHPTR(pThread, threads)
					pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
				FOREACHPTR(pThread, threads)
					pThread->join();
			} else {
				// single-thread execution
				DenseReconstructionEstimate((void*)&data);
			}
			GET_LOGCONSOLE().Play();
			if (!data.events.IsEmpty())
				return false;
			data.progress.Release();
			// replace raw depth-maps with the geometric-consistent ones
			for (IIndex idx: data.images) {
				const DepthData& depthData(data.depthMaps.arrDepthData[idx]);
				if (!depthData.IsValid())
					continue;
				const String rawName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
				File::deleteFile(rawName);
				File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName);
			}
		}
		data.nEstimationGeometricIter = -1;
	}

	if ((OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) != 0) {
		// initialize the queue of depth-maps to be filtered
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) {
			// multi-thread execution
			cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
	}
	return true;
} // ComputeDepthMaps
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// initialize the dense reconstruction with the sparse point cloud
void Scene::DenseReconstructionEstimate(void* pData)
{
	// JPB WIP OPT All of these are running single threaded... including the saving of maps
	// It may call into new routines, say estimatedepthmap, which uses multithreading, but this part is likely very slow.

	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	while (true) {
		CAutoPtr<Event> evt(data.events.GetEvent());
		switch (evt->GetID()) {
		case EVT_PROCESSIMAGE: {
			const EVTProcessImage& evtImage = *((EVTProcessImage*)(Event*)evt);
			if (evtImage.idxImage >= data.images.size()) {
				if (nMaxThreads > 1) {
					// close working threads
					data.events.AddEvent(new EVTClose);
				}
				return;
			}
			// select views to reconstruct the depth-map for this image
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			const bool depthmapComputed(data.nFusionMode < 0 || (data.nFusionMode >= 0 && data.nEstimationGeometricIter < 0 && File::access(ComposeDepthFilePath(data.scene.images[idx].ID, "dmap"))));
			// initialize images pair: reference image and the best neighbor view
			ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
			if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews, !depthmapComputed, depthmapComputed ? -1 : (data.nEstimationGeometricIter >= 0 ? 1 : 0))) {
				// process next image
				data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
				break;
			}
			// try to load already compute depth-map for this image
			if (depthmapComputed && data.nFusionMode >= 0) {
				if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
					if (!depthData.Load(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"))) {
						VERBOSE("error: invalid depth-map '%s'", ComposeDepthFilePath(depthData.GetView().GetID(), "dmap").c_str());
						exit(EXIT_FAILURE);
					}
					// optimize depth-map
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				}
				// process next image
				data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			} else {
				// estimate depth-map
				data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));
			}
			break; }

		case EVT_ESTIMATEDEPTHMAP: {
			const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)(Event*)evt);
			// request next image initialization to be performed while computing this depth-map
			data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			// extract depth map
			data.sem.Wait();
			if (data.nFusionMode >= 0) {
				// extract depth-map using Patch-Match algorithm
				data.depthMaps.EstimateDepthMap(data.images[evtImage.idxImage], data.nEstimationGeometricIter);
			} else {
				// extract disparity-maps using SGM algorithm
				if (data.nFusionMode == -1) {
					data.sgm.Match(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews);
				} else {
					// fuse existing disparity-maps
					const IIndex idx(data.images[evtImage.idxImage]);
					DepthData& depthData(data.depthMaps.arrDepthData[idx]);
					data.sgm.Fuse(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews, 2, depthData.depthMap, depthData.confMap);
					if (OPTDENSE::nEstimateNormals == 2)
						EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
					depthData.dMin = ZEROTOLERANCE<float>(); depthData.dMax = FLT_MAX;
				}
			}
			data.sem.Signal();
			if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
				// optimize depth-map
				data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
			} else {
				// save depth-map
				data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			}
			break; }

		case EVT_OPTIMIZEDEPTHMAP: {
			const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 3)
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "raw.png"), depthData.depthMap);
			#endif
			// apply filters
			if (OPTDENSE::nOptimize & (OPTDENSE::REMOVE_SPECKLES)) {
				TD_TIMER_START();
				if (data.depthMaps.RemoveSmallSegments(depthData)) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			if (OPTDENSE::nOptimize & (OPTDENSE::FILL_GAPS)) {
				TD_TIMER_START();
				if (data.depthMaps.GapInterpolation(depthData)) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			// save depth-map
			data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			break; }

		case EVT_SAVEDEPTHMAP: {
			const EVTSaveDepthMap& evtImage = *((EVTSaveDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "png"), depthData.depthMap);
				ExportConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.png"), depthData.confMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
				if (g_nVerbosityLevel > 4) {
					ExportNormalMap(ComposeDepthFilePath(depthData.GetView().GetID(), "normal.png"), depthData.normalMap);
					depthData.confMap.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.pfm"));
				}
			}
			#endif
			// save compute depth-map for this image
			// JPB WIP Saving this asynchronously doesn't help.
			if (!depthData.depthMap.empty())
				depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), data.nEstimationGeometricIter < 0 ? "dmap" : "geo.dmap"));
			depthData.ReleaseImages();
			depthData.Release();
			data.progress->operator++();
			break; }

		case EVT_CLOSE: {
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionEstimate
/*----------------------------------------------------------------*/

void* DenseReconstructionFilterTmp(void* arg) {
	DenseDepthMapData& dataThreads = *((DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionFilter(arg);
	return NULL;
}

// filter estimated depth-maps
void Scene::DenseReconstructionFilter(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	CAutoPtr<Event> evt;
	while ((evt=data.events.GetEvent(0)) != NULL) {
		switch (evt->GetID()) {
		case EVT_FILTERDEPTHMAP: {
			const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			if (!depthData.IsValid()) {
				data.SignalCompleteDepthmapFilter();
				break;
			}
			// make sure all depth-maps are loaded
			depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			const unsigned numMaxNeighbors(8);
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			FOREACH(n, depthData.neighbors) {
				const IIndex idxView = depthData.neighbors[n].ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				if (!depthDataPair.IsValid())
					continue;
				if (depthDataPair.IncRef(ComposeDepthFilePath(depthDataPair.GetView().GetID(), "dmap")) == 0) {
					// signal error and terminate
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.Insert(n);
				if (idxNeighbors.GetSize() == numMaxNeighbors)
					break;
			}
			// filter the depth-map for this image
			if (data.depthMaps.FilterDepthMap(depthData, idxNeighbors, OPTDENSE::bFilterAdjust)) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
			// unload referenced depth-maps
			FOREACHPTR(pIdxNeighbor, idxNeighbors) {
				const IIndex idxView = depthData.neighbors[*pIdxNeighbor].ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				depthDataPair.DecRef();
			}
			depthData.DecRef();
			data.SignalCompleteDepthmapFilter();
			break; }

		case EVT_ADJUSTDEPTHMAP: {
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();
			// load filtered maps
			if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0 ||
				!LoadDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.dmap"), depthData.depthMap) ||
				!LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.cmap"), depthData.confMap))
			{
				// signal error and terminate
				data.events.AddEventFirst(new EVTFail);
				return;
			}
			ASSERT(depthData.GetRef() == 1);
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.dmap").c_str());
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.cmap").c_str());
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.png"), depthData.depthMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
			// save filtered depth-map for this image
			depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			depthData.DecRef();
			data.progress->operator++();
			break; }

		case EVT_FAIL: {
			data.events.AddEventFirst(new EVTFail);
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionFilter
/*----------------------------------------------------------------*/

// filter point-cloud based on camera-point visibility intersections
void Scene::PointCloudFilter(int thRemove)
{
#if 1 // JPB WIP BUG
	throw std::runtime_error("Unsupported");
#else
	TD_TIMER_STARTD();

	typedef TOctree<PointCloud::PointArr,PointCloud::Point::Type,3,uint32_t> Octree;
	struct Collector {
		typedef Octree::IDX_TYPE IDX;
		typedef PointCloud::Point::Type Real;
		typedef TCone<Real,3> Cone;
		typedef TSphere<Real,3> Sphere;
		typedef TConeIntersect<Real,3> ConeIntersect;

		Cone cone;
		const ConeIntersect coneIntersect;
		const PointCloud& pointcloud;
		IntArr& visibility;
		PointCloud::Index idxPoint;
		Real distance;
		int weight;
		#ifdef DENSE_USE_OPENMP
		uint8_t pcs[sizeof(CriticalSection)];
		#endif

		Collector(const Cone::RAY& ray, Real angle, const PointCloud& _pointcloud, IntArr& _visibility)
			: cone(ray, angle), coneIntersect(cone), pointcloud(_pointcloud), visibility(_visibility)
		#ifdef DENSE_USE_OPENMP
		{ new(pcs) CriticalSection; }
		~Collector() { reinterpret_cast<CriticalSection*>(pcs)->~CriticalSection(); }
		inline CriticalSection& GetCS() { return *reinterpret_cast<CriticalSection*>(pcs); }
		#else
		{}
		#endif
		inline void Init(PointCloud::Index _idxPoint, const PointCloud::Point& X, int _weight) {
			const Real thMaxDepth(1.02f);
			idxPoint =_idxPoint;
			const PointCloud::Point::EVec D((PointCloud::Point::EVec&)X-cone.ray.m_pOrig);
			distance = D.norm();
			cone.ray.m_vDir = D/distance;
			cone.maxHeight = MaxDepthDifference(distance, thMaxDepth);
			weight = _weight;
		}
		inline bool Intersects(const Octree::POINT_TYPE& center, Octree::Type radius) const {
			return coneIntersect(Sphere(center, radius*Real(SQRT_3)));
		}
		inline void operator () (const IDX* idices, IDX size) {
			const Real thSimilar(0.01f);
			Real dist;
			FOREACHRAWPTR(pIdx, idices, size) {
				const PointCloud::Index idx(*pIdx);
				if (coneIntersect.Classify(pointcloud.points[idx], dist) == VISIBLE && !IsDepthSimilar(distance, dist, thSimilar)) {
					if (dist > distance)
						visibility[idx] += pointcloud.pointViews[idx].size();
					else
						visibility[idx] -= weight;
				}
			}
		}
	};
	typedef CLISTDEF2(Collector) Collectors;

	// create octree to speed-up search
	Octree octree(pointcloud.points, [](Octree::IDX_TYPE size, Octree::Type /*radius*/) {
		return size > 128;
	});
	IntArr visibility(pointcloud.GetSize()); visibility.Memset(0);
	Collectors collectors; collectors.reserve(images.size());
	FOREACH(idxView, images) {
		const Image& image = images[idxView];
		const Ray3f ray(Cast<float>(image.camera.C), Cast<float>(image.camera.Direction()));
		const float angle(float(image.ComputeFOV(0)/image.width));
		collectors.emplace_back(ray, angle, pointcloud, visibility);
	}

	// run all camera-point visibility intersections
	Util::Progress progress(_T("Point visibility checks"), pointcloud.GetSize());
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel for //schedule(dynamic)
	for (int64_t i=0; i<(int64_t)pointcloud.GetSize(); ++i) {
		const PointCloud::Index idxPoint((PointCloud::Index)i);
	#else
	FOREACH(idxPoint, pointcloud.points) {
	#endif
		const PointCloud::Point& X = pointcloud.points[idxPoint];
		const PointCloud::ViewArr& views = pointcloud.pointViews[idxPoint];
		for (PointCloud::View idxView: views) {
			Collector& collector = collectors[idxView];
			#ifdef DENSE_USE_OPENMP
			Lock l(collector.GetCS());
			#endif
			collector.Init(idxPoint, X, (int)views.size());
			octree.Collect(collector, collector);
		}
		++progress;
	}
	progress.close();

	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print visibility stats
		UnsignedArr counts(0, 64);
		for (int views: visibility) {
			if (views > 0)
				continue;
			while (counts.size() <= IDX(-views))
				counts.push_back(0);
			++counts[-views];
		}
		String msg;
		msg.reserve(64*counts.size());
		FOREACH(c, counts)
			if (counts[c])
				msg += String::FormatString("\n\t% 3u - % 9u", c, counts[c]);
		VERBOSE("Visibility lengths (%u points):%s", pointcloud.GetSize(), msg.c_str());
		// save outlier points
		PointCloud pc;

		RFOREACH(idxPoint, pointcloud.points) {
			if (visibility[idxPoint] <= thRemove) {
				pc.points.push_back(pointcloud.points[idxPoint]);
				pc.colors.push_back(pointcloud.colors[idxPoint]);
			}
		}
		pc.Save(MAKE_PATH("scene_dense_outliers.ply"));
	}
	#endif

	// filter points
	const size_t numInitPoints(pointcloud.GetSize());
	RFOREACH(idxPoint, pointcloud.points) {
		if (visibility[idxPoint] <= thRemove)
			pointcloud.RemovePoint(idxPoint);
	}

	DEBUG_EXTRA("Point-cloud filtered: %u/%u points (%d%%%%) (%s)", pointcloud.points.size(), numInitPoints, ROUND2INT((100.f*pointcloud.points.GetSize())/numInitPoints), TD_TIMER_GET_FMT().c_str());
#endif
	} // PointCloudFilter
/*----------------------------------------------------------------*/
