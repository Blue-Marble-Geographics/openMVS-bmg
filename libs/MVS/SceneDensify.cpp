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

#include <list>
#include <unordered_map>

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif

#undef ESTIMATE_NORMALS // Not used

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
			const String dmapPath(ComposeDepthFilePath(view.GetID(), "dmap"));
			if (File::access(dmapPath)) {
				String imageFileName;
				IIndexArr IDs;
				cv::Size imageSize;
				Depth dMin, dMax;
				NormalMap normalMap;
				ConfidenceMap confMap;
				ViewsMap viewsMap;
				ImportDepthDataRaw(dmapPath,
					imageFileName, IDs, imageSize, view.cameraDepthMap.K, view.cameraDepthMap.R, view.cameraDepthMap.C,
					dMin, dMax, view.depthMap, normalMap, confMap, viewsMap, 1);
			}
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
#if 1
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
#else
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool bAdjust)
{
	TD_TIMER_STARTD();

  // This shouldn't be parallelized since as its caller is.
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
#endif
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

#if 1 // Variant based on latest original work
//#pragma optimize("", off) // JPB WIP BUG Debugging
// Tracks accesses of some hash-able type T and records the least recently accessed.
template<typename T>
class ListFIFO {
public:
	// add or move an key to the front
	void Put(const T& key) {
		const auto it = map.find(key);
		if (it != map.end()) {
			// if key exists, remove it from its current position
			order.erase(it->second);
		}
		// add the key to the front
		order.push_front(key);
		map[key] = order.begin();
	}

	// remove and return the least used key (from the back)
	T Pop() {
		ASSERT(!IsEmpty());
		const T leastUsed = order.back();
		order.pop_back();
		map.erase(leastUsed);
		return leastUsed;
	}

	// return the least used key (from the back)
	const T& Back() {
		ASSERT(!IsEmpty());
		return order.back();
	}

	// check if the list is empty
	bool IsEmpty() const {
		return order.empty();
	}

	// get the size of the list
	size_t Size() const {
		return order.size();
	}

	// return true if the key is in the list
	bool Contains(const T& key) const {
		return map.find(key) != map.end();
	}

	// return the keys currently in cache
	const std::list<T>& GetCachedValues() const {
		return order;
	}

	// print the current order of elements
	void PrintOrder() const {
		std::cout << "Current order: ";
		for (const auto& element : order) {
			std::cout << element << " ";
		}
		std::cout << std::endl;
	}

private:
	std::list<T> order;
	std::unordered_map<T, typename std::list<T>::iterator> map;
};

struct DMapCache {
	DMapCache(DepthDataArr& _arrDepthData, unsigned _loadFlags, size_t _max_memory_bytes)
		:
		loadFlags(_loadFlags), arrDepthData(_arrDepthData),
		maxMemory(_max_memory_bytes), disabledMaxMemory(0), usedMemory(0),
		skipMemoryCheckIdxImage(NO_ID), numImageRead(0)
	{
	}

	bool IsEmpty() const { ASSERT((usedMemory == 0) == fifo.IsEmpty()); return fifo.IsEmpty(); }

	void SetMaxMemory(size_t max_memory_bytes) {
		maxMemory = max_memory_bytes;
		ASSERT(skipMemoryCheckIdxImage == NO_ID);
		Eject();
	}


	// enable/disable memory usage
	void DisableMemoryCheck() { disabledMaxMemory = maxMemory; maxMemory = 0; }
	void EnableMemoryCheck() { if (disabledMaxMemory) { maxMemory = disabledMaxMemory; disabledMaxMemory = 0; Eject(); } }

	// skip memory check if this image index is to be ejected
	void SkipMemoryCheckIdxImage(IIndex idxImage = NO_ID) { skipMemoryCheckIdxImage = idxImage; }

	bool UseImage(IIndex idxImage) const {
		ASSERT(idxImage < arrDepthData.size());
		std::lock_guard<std::mutex> guard(mutex);
		ASSERT(arrDepthData[idxImage].IsValid());
		if (!arrDepthData[idxImage].IsEmpty()) {
			fifo.Put(idxImage);
			return false;
		}
		mutex.unlock();
		const String fileName(ComposeDepthFilePath(arrDepthData[idxImage].GetView().GetID(), "dmap"));
		while (!std::filesystem::is_regular_file(static_cast<const std::string&>(fileName)))
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		arrDepthData[idxImage].Load(fileName, loadFlags);
		ASSERT(!arrDepthData[idxImage].IsEmpty());
		mutex.lock();
		++numImageRead;
		usedMemory += arrDepthData[idxImage].GetMemorySize();
		fifo.Put(idxImage);
		Eject();
		return true;
	}

	IIndexArr GetCachedImageIndices(bool ordered) const {
		std::lock_guard<std::mutex> guard(mutex);
		IIndexArr cachedImageIndices;
		FOREACH(idxImage, arrDepthData)
			if (!arrDepthData[idxImage].IsEmpty())
				cachedImageIndices.push_back(idxImage);
		if (ordered)
			cachedImageIndices.Sort();
		return cachedImageIndices;
	}

	bool IsImageCached(IIndex idxImage) const {
		return fifo.Contains(idxImage);
	}

	// get the number of times images were read from disk
	uint32_t GetNumImageReads() const { return numImageRead; }

	void ClearCache() {
		std::lock_guard<std::mutex> guard(mutex);
		skipMemoryCheckIdxImage = NO_ID;
		while (!IsEmpty())
			EjectOldest();
	}

	size_t GetUsedMemory() const { return usedMemory; }

	size_t ComputeUsedMemory() const {
		std::lock_guard<std::mutex> guard(mutex);
		size_t computedUsedMemory = 0;
		for (const auto& depthData : arrDepthData)
			if (!depthData.IsEmpty())
				computedUsedMemory += depthData.GetMemorySize();
		ASSERT(computedUsedMemory == usedMemory);
		return computedUsedMemory;
	}

	bool Eject() const {
		if (maxMemory == 0)
			return true;
		while (usedMemory > maxMemory) {
			if (!EjectOldest())
				return false;
		}
		return true;
	}

#if 1 // JPB WIP BUG Debugging  this is definitely better
	bool EjectOldest() const {
		ASSERT(!fifo.IsEmpty());
		if (fifo.Back() == skipMemoryCheckIdxImage)
			return false;
		const IIndex idxImage = fifo.Pop();
		// Persist depth invalidations (zeroed pixels) back to disk before releasing
		DepthData& depthData = arrDepthData[idxImage];
		if (!depthData.depthMap.empty())
			depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
		usedMemory -= depthData.GetMemorySize();
		depthData.Release();
		return true;
	}
#else
	bool EjectOldest() const {
		ASSERT(!fifo.IsEmpty());
		if (fifo.Back() == skipMemoryCheckIdxImage)
			return false;
		const IIndex idxImage = fifo.Pop();
		usedMemory -= arrDepthData[idxImage].GetMemorySize();
		// release the depth-data; no need to save the depth-data to disk as it is already saved
		arrDepthData[idxImage].Release();
		return true;
	}
#endif

	unsigned loadFlags;
	DepthDataArr& arrDepthData;

	// maximum and used memory (in bytes)
	size_t maxMemory, disabledMaxMemory;
	mutable size_t usedMemory;

	// index of the image to skip memory check
	IIndex skipMemoryCheckIdxImage;

	// guard access to variables that are dynamically loaded from disk
	mutable std::mutex mutex;

	// track which images are last accessed
	mutable ListFIFO<IIndex> fifo;

	// number of times images were read from disk (debug only)
	mutable uint32_t numImageRead;
};

// compute available memory to be used for depth-data caching
//  - numDMapsReserveFusion: maximum number of depth-maps for which to reserve memory for fusion
size_t GetAvailableMemory(const DepthDataArr& arrDepthData, const BoolArr& fusedDMaps, IIndex numDMapsReserveFusion, size_t currentCacheMemory = 0)
{
	size_t resolution(0);
	IIndex numDMaps(0);
	FOREACH(idxImage, arrDepthData) {
		const DepthData& depthData = arrDepthData[idxImage];
		if (!depthData.IsValid())
			continue;
		if (fusedDMaps[idxImage])
			continue;
		// Use depthData.size instead of depthData.depthMap.area()
		// because the depth map may not be loaded in memory at this point
		resolution += depthData.size.area();
		if (++numDMaps >= numDMapsReserveFusion)
			break;
	}
	if (numDMaps == 0)
		return 0;
	const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
	const size_t neededPointCloudMemory(static_cast<size_t>(resolution * (1/*depth*/+1/*color*/+3/*normal*/+1/*confidence*/) * 4/*bytes*/ * 0.35/*unique pixels per depth-map*/));
	const size_t freeMemory(currentCacheMemory + memInfo.freePhysical);
	const size_t safetyMemory(std::max(static_cast<size_t>(memInfo.totalPhysical * 0.08), size_t(1*1024*1024*1024ull)/*1GB*/));	const size_t neededMemory(neededPointCloudMemory + safetyMemory);
	const size_t minDMapsMemory(resolution / numDMaps * 8/*min dmaps in memory*/ * (1/*depth*/ + 3/*normal*/ + 1/*confidence*/) * 4/*bytes*/);
	if (freeMemory < neededMemory) {
		DEBUG("warning: not enough memory to cache depth-maps (%luMB needed, %luMB available)", neededMemory/1024/1024, freeMemory/1024/1024);
		return MINF(currentCacheMemory, minDMapsMemory);
	}
	return freeMemory - neededMemory;
}

// finds the best depth-map to fuse next that maximizes the number of neighbors already in cache
std::tuple<unsigned, unsigned, unsigned> FetchBestNextDMapIndex(const DepthDataArr& arrDepthData, const DMapCache& cacheDMaps, const BoolArr& fusedDMaps) {
	const IIndexArr cachedImages = cacheDMaps.GetCachedImageIndices(true);
	IIndex bestImageIdx = NO_ID;
	unsigned bestImageScore = 0, bestImageSize = std::numeric_limits<unsigned>::max();
	FOREACH(idxImage, arrDepthData) {
		const DepthData& depthData = arrDepthData[idxImage];
		if (!depthData.IsValid())
			continue;
		if (fusedDMaps[idxImage])
			continue;
		ASSERT(!depthData.neighbors.empty());
		IIndexArr cachedNeighbors;
		if (!cachedImages.empty()) {
			IIndexArr neighbors(0, depthData.neighbors.size());
			for (ViewScore& neighbor: depthData.neighbors)
				neighbors.push_back(neighbor.ID);
			neighbors.Sort();
			std::set_intersection(neighbors.begin(), neighbors.end(),
				cachedImages.begin(), cachedImages.end(),
				std::back_inserter(cachedNeighbors));
		}
		if (bestImageScore < cachedNeighbors.size() ||
			(bestImageScore == cachedNeighbors.size() && bestImageSize > depthData.neighbors.size())) {
			bestImageScore = cachedNeighbors.size();
			bestImageSize = depthData.neighbors.size();
			bestImageIdx = idxImage;
		}
	}
	return std::make_tuple(bestImageIdx, bestImageScore, static_cast<unsigned>(cachedImages.size()));
} // FetchBestNextDMapIndex

// fuse all valid depth-maps in the same 3D point-cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
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
		inline ImageRef GetCoord() const { return ImageRef(x, y); }
	};
	typedef SEACAVE::cList<Proj, const Proj&, 0, 4, uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr, const ProjArr&, 1, 65536> ProjsArr;

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, arrDepthData.size()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));
	const IIndex numDMapsReserveFusion(10);
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(arrDepthData.size());
	const size_t nPointsEstimate(arrDepthData.size() * arrDepthData.First().depthMap.area());
	ProjsArr projs(0, nPointsEstimate);
	pointcloud.ReservePoints(nPointsEstimate);
	pointcloud.ReservePointViewsSizeAndOffset(nPointsEstimate);
	pointcloud.ReservePointWeightsSizeAndOffset(nPointsEstimate);
	unsigned depthDataLoadFlags(HeaderDepthDataRaw::HAS_DEPTH | HeaderDepthDataRaw::HAS_CONF);
	if (bEstimateColor)
		pointcloud.ReserveColors(nPointsEstimate);
#if 0 // JPB WIP BUG
	if (bEstimateNormal) {
		pointcloud.normals.reserve(nPointsEstimate);
		depthDataLoadFlags |= HeaderDepthDataRaw::HAS_NORMAL;
	}
#endif

	MEMORYSTATUS memState{};
	::GlobalMemoryStatus(&memState);
	const size_t bytesAvailable = memState.dwAvailPhys;

	// Split a quarter of what's left between points and views.
	const size_t elementSize =
		std::max(
			sizeof(decltype(pointcloud.pointViewsMemory)::value_type),
			sizeof(decltype(pointcloud.pointWeightsMemory)::value_type)
		);

	const size_t nElementsAvailableToUse = (bytesAvailable / elementSize) / 4;
	pointcloud.ReservePointViewsMemory(nElementsAvailableToUse / 4);
	pointcloud.ReservePointWeightsMemory(nElementsAvailableToUse / 4);

	Util::Progress progress(_T("Fused depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	BoolArr fusedDMaps(arrDepthData.size());
	fusedDMaps.Memset(0);
	DMapCache cacheDMaps(arrDepthData, depthDataLoadFlags, GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion));
	unsigned totalNumImageNeighborsInCache = 0, totalNumImagesInCache = 0;
	IIndex numDMapsFused = 0;

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
	}

	for (; numDMapsFused < arrDepthData.size(); ++numDMapsFused) {
		TD_TIMER_STARTD();
		// find the best depth-map to fuse next as the one with the most neighbors already in cache
		const auto [idxImage, numImageNeighborsInCache, numImagesInCache] = FetchBestNextDMapIndex(arrDepthData, cacheDMaps, fusedDMaps);
		if (idxImage == NO_ID)
			break; // no more depth-maps to fuse (only invalid depth-maps left)
		totalNumImageNeighborsInCache += numImageNeighborsInCache;
		totalNumImagesInCache += numImagesInCache;
		// fuse depth-map
		cacheDMaps.UseImage(idxImage);
		cacheDMaps.SkipMemoryCheckIdxImage(idxImage);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		ASSERT(!depthData.IsEmpty());
#if 0 // JPB WIP BUG
		if (bEstimateNormal && depthData.normalMap.empty())
			EstimateNormalMaps();
#endif
		constexpr IIndex nMaxViewsFuse = 32; // JPB WIP OPTDENSE::nMaxViewsFuse not imported
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
		IIndex numNeighbors(0);
#ifdef DENSE_USE_OPENMP
#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t)depthData.neighbors.size(); ++i) {
			const ViewScore& neighbor = depthData.neighbors[(IIndex)i];
#else
		for (const ViewScore& neighbor : depthData.neighbors) {
#endif
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			if (++numNeighbors >= nMaxViewsFuse)
#ifdef DENSE_USE_OPENMP
				continue;
#else
				break;
#endif
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.ID];
			if (!depthIdxs.empty())
				continue;
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}

		ASSERT(!depthData.IsEmpty());
		const Image& imageData = *depthData.images.front().pImageData;
		ASSERT(&imageData - scene.images.data() == idxImage);
		ASSERT(depthData.depthMap.size() == depthData.size && imageData.GetSize() == depthData.size);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(depthData.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}

		const size_t nNumPointsPrev(pointcloud.NumPoints());
		const Image8U::Size sizeMap(depthData.depthMap.size());

		// =====================================================================
		// FUSE_DIAGNOSTICS: per-image visual dumps so missing regions can be
		// traced to the correct pipeline stage.
		//   - <id>.fuse.input.png    = depth map fed into fuse (what PatchMatch
		//                              + FilterDepthMap produced)
		//   - <id>.fuse.survived.png = pixels that emitted a point
		//   - <id>.fuse.culled.png   = pixels dropped by the nMinViewsFuse gate
		//   - <id>.fuse.preclaimed.png = pixels already owned by a neighbor's
		//                                emitted point (no work done)
		// Set to 0 to disable.
		// =====================================================================
		#define FUSE_DIAGNOSTICS 0
#if FUSE_DIAGNOSTICS
		ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "fuse.input.png"), depthData.depthMap);
		DepthMap dmgSurvived;   dmgSurvived.create(sizeMap);   dmgSurvived.memset(0);
		DepthMap dmgCulled;     dmgCulled.create(sizeMap);     dmgCulled.memset(0);
		DepthMap dmgPreclaimed; dmgPreclaimed.create(sizeMap); dmgPreclaimed.memset(0);
		// Color-coded fate image (BGR):
		//   BLACK = no depth at this pixel (nothing to do)
		//   GREEN = survived -> emitted a point
		//   BLUE  = preclaimed by a neighbor that fused earlier (point exists, just not ours)
		//   RED   = culled by nMinViewsFuse consensus gate (had depth, not enough agreement)
		Image8U3 dmgFate(sizeMap.height, sizeMap.width);
		dmgFate.memset(0);
		// Per-image counters explaining WHY neighbors failed to agree on consensus.
		// Each counter is incremented once per (ref-pixel, neighbor) probe pair.
		uint64_t cNeighborsTotal       = 0; // every (ref-pix, neighbor) probe pair
		uint64_t cNeighborOOB          = 0; // neighbor pixel outside image bounds
		uint64_t cNeighborZeroDepth    = 0; // neighbor pixel has no depth
		uint64_t cNeighborPreclaimed   = 0; // neighbor pixel already owned by another ref
		uint64_t cNeighborDepthMismatch= 0; // |ptz - depthB| >= threshold
		uint64_t cNeighborNormalFail   = 0; // depth OK but normal gate failed
		uint64_t cNeighborAccepted     = 0; // full hit
		// Per-pixel outcome counters.
		uint64_t cPixRefDepths      = 0;
		uint64_t cPixRefPreclaimed  = 0;
		uint64_t cPixRefSurvived    = 0;
		uint64_t cPixRefCulled      = 0;
		// Histogram of nHits value at min-views cull time (only pixels that failed the gate)
		uint64_t cHitsHistCulled[16] = {0};
#endif

		const float confMapSentinel = 1.f;
		size_t confMapInc;

#ifdef ESTIMATE_NORMALS
		boost::container::small_vector<Proj, 16> projs;
#endif

		struct NeighborCache {
			IIndex idxImageB;
			_Data col0, col1, col2, col3;
			_Data zzz_wh10_bounds; // _SetN(w-1, h-1, 1.f, 0.f)
			DepthMap* depthMapB;
			DepthIndex* depthIdxB;
			const ConfidenceMap* __restrict confMapB;
			const NormalMap* __restrict normalMapB;
			const Image* __restrict imageDataB;
			TRMatrixBase<float> const* __restrict cameraRt;
			bool confMapBEmpty;
			// Pre-computed inverse-K constants for Phase 2 back-projection
			double invK00, invK11, K02, K12;
			double Cx, Cy, Cz;
			// R transposed columns for back-projection: rot(row,col) = R[col*3+row] in row-major
			double R00, R10, R20, R01, R11, R21, R02, R12, R22;
			bool     normalMapBEmpty; // BUGFIX: per-neighbor normal-map availability
		};
		boost::container::small_vector<NeighborCache, 16> neighborCache;
		neighborCache.reserve(depthData.neighbors.size());
		for (const auto& neighbor : depthData.neighbors) {
			DepthData& ddb = arrDepthData[neighbor.ID];
			// Match reference: skip invalid / empty neighbors but DO NOT cap at
			// nMaxViewsFuse. The reference's nMaxViewsFuse cap only bounds the
			// depthIdxs initialization loop — its neighbor probe iterates every
			// valid, non-empty neighbor. Capping here drops genuine matches and
			// starves nMinViewsFuse.
			if (!ddb.IsValid() || ddb.IsEmpty())
				continue;
			if (neighborCache.size() >= nMaxViewsFuse)
				break;
			NeighborCache nc;
			nc.idxImageB = neighbor.ID;
			const auto& pt = imagesCameraPt[nc.idxImageB];
			nc.col0 = _Load(&pt[0]);
			nc.col1 = _Load(&pt[4]);
			nc.col2 = _Load(&pt[8]);
			nc.col3 = _Load(&pt[12]);
			nc.depthMapB = &ddb.depthMap;
			nc.zzz_wh10_bounds = _SetN(
				(float)(ddb.depthMap.width() - 1),
				(float)(ddb.depthMap.height() - 1), 1.f, 0.f);
			nc.depthIdxB = &arrDepthIdx[nc.idxImageB];
			nc.confMapB = &ddb.confMap;
			nc.confMapBEmpty = ddb.confMap.empty();
			nc.normalMapB = &ddb.normalMap;
			nc.normalMapBEmpty = ddb.normalMap.empty(); // BUGFIX
			nc.imageDataB = &scene.images[nc.idxImageB];
			nc.cameraRt = &imagesCameraRt[nc.idxImageB];
			Camera const* __restrict camera = &scene.images[nc.idxImageB].camera;
			// Pre-compute inverse-K and camera constants for Phase 2
			nc.invK00 = 1.0 / camera->K(0, 0);
			nc.invK11 = 1.0 / camera->K(1, 1);
			nc.K02 = camera->K(0, 2);
			nc.K12 = camera->K(1, 2);
			nc.Cx = camera->C.x;
			nc.Cy = camera->C.y;
			nc.Cz = camera->C.z;
			const auto& rot = camera->R;
			nc.R00 = rot(0, 0); nc.R10 = rot(1, 0); nc.R20 = rot(2, 0);
			nc.R01 = rot(0, 1); nc.R11 = rot(1, 1); nc.R21 = rot(2, 1);
			nc.R02 = rot(0, 2); nc.R12 = rot(1, 2); nc.R22 = rot(2, 2);
			neighborCache.push_back(nc);
		}

		const float fDepthDiffThreshold = OPTDENSE::fDepthDiffThreshold;
		// Match reference exactly: fusion uses the raw threshold (no multiplier).
		const float fDepthDiffThresholdFuse = fDepthDiffThreshold;
		const _Data vTwo = _Set(2.f);
#if FUSE_DIAGNOSTICS
		// Sanity-check the effective neighbor cache size.  If this is small for
		// problem images, the depth-map streaming cache evicted neighbors faster
		// than they could be loaded — the real cause of fuse starvation.
		const size_t numAvailableNeighbors = depthData.neighbors.size();
		const size_t numPopulatedNeighbors = neighborCache.size();
		size_t numNeighborsValid = 0, numNeighborsEmpty = 0, numNeighborsInvalid = 0;
		for (const auto& neighbor : depthData.neighbors) {
			DepthData& ddb = arrDepthData[neighbor.ID];
			if (!ddb.IsValid())      ++numNeighborsInvalid;
			else if (ddb.IsEmpty())  ++numNeighborsEmpty;
			else                     ++numNeighborsValid;
		}
		VERBOSE("FUSE DIAG img=%3u  cacheSize=%zu  neighbors total=%zu valid=%zu empty=%zu invalid=%zu  nMinViewsFuse=%u",
			depthData.GetView().GetID(),
			numPopulatedNeighbors, numAvailableNeighbors,
			numNeighborsValid, numNeighborsEmpty, numNeighborsInvalid,
			(unsigned)nMinViewsFuse);
#endif

		// --- Lightweight candidate structure for two-phase neighbor check ---
		struct NeighborHit {
			unsigned ncIdx;    // index into neighborCache
			ImageRef xB;
			float    ptz;
			Depth    depthB;   // cached so Phase 2 doesn't re-read
			Depth* pDepthB;  // pointer for invalidation / deferred commit
			uint32_t* pIdxPointB; // pointer for deferred commit
		};
		const unsigned maxNeighbors = (unsigned)neighborCache.size();
		NeighborHit* hitsStorage = (NeighborHit*)_alloca(maxNeighbors * sizeof(NeighborHit));
		NeighborHit* invalidHitsStorage = (NeighborHit*)_alloca(maxNeighbors * sizeof(NeighborHit));
		unsigned nHits = 0, nInvalidHits = 0;

		const unsigned maxViews = (unsigned)neighborCache.size() + 1; // +1 for reference view
		uint32_t* __restrict viewsStorage = (uint32_t*)_alloca(maxViews * sizeof(uint32_t));
		float* __restrict weightsStorage = (float*)_alloca(maxViews * sizeof(float));
		// Defer idxPointB assignments until we know the point survives the fuse check.
		// Collect pointers to idxPointB slots so we can commit them only for accepted points.
		uint32_t** __restrict deferredStorage = (uint32_t**)_alloca(neighborCache.size() * sizeof(uint32_t*));
		unsigned nViews = 0, nDeferred = 0;

		bool bNormalMap = !depthData.normalMap.empty();

		// Hoist reference camera R and C to locals (read once per image, not per pixel)
		const float refR00 = (float)imageData.camera.R[0*3+0];
		const float refR10 = (float)imageData.camera.R[1*3+0];
		const float refR20 = (float)imageData.camera.R[2*3+0];
		const float refR01 = (float)imageData.camera.R[0*3+1];
		const float refR11 = (float)imageData.camera.R[1*3+1];
		const float refR21 = (float)imageData.camera.R[2*3+1];
		const float refR02 = (float)imageData.camera.R[0*3+2];
		const float refR12 = (float)imageData.camera.R[1*3+2];
		const float refR22 = (float)imageData.camera.R[2*3+2];
		const float refCx = (float)imageData.camera.C.x;
		const float refCy = (float)imageData.camera.C.y;
		const float refCz = (float)imageData.camera.C.z;

		for (int i = 0; i < sizeMap.height; ++i) {
			const Depth* __restrict pDM = &depthData.depthMap(i, 0);
			uint32_t* __restrict pDepthIdxs = (uint32_t*)&depthIdxs(i, 0);

			const float* __restrict pConfMap;
			bool confMapEmpty = depthData.confMap.empty();
			if (confMapEmpty) {
				pConfMap = &confMapSentinel;
				confMapInc = 0;
			}
			else {
				confMapInc = 1;
			}

			if (!confMapEmpty) {
				pConfMap = &depthData.confMap(i, 0);
			}

			const Normal* __restrict pNormalMap = &depthData.normalMap(i, 0);
			const Pixel8U* __restrict pImage = &imageData.image(i, 0);

			const double invImageDataCameraK00 = 1. / imageData.camera.K(0, 0);
			const double invImageDataCameraK11 = 1. / imageData.camera.K(1, 1);
			const double imageDataCameraK02 = imageData.camera.K(0, 2);
			const double imageDataCameraK12 = imageData.camera.K(1, 2);
			double pointXNoDepthPreTransform = -imageDataCameraK02 * invImageDataCameraK00;
			double pointXNoDepthPreTransformDelta = invImageDataCameraK00;
			double pointYNoDepthPreTransform = (i - imageDataCameraK12) * invImageDataCameraK11;

			for (int j = 0; j < sizeMap.width; ++j, pConfMap += confMapInc, pointXNoDepthPreTransform += pointXNoDepthPreTransformDelta) {
				const Depth depth(pDM[j]);
				if (depth == 0)
					continue;

				++nDepths;
#if FUSE_DIAGNOSTICS
				++cPixRefDepths;
#endif
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = pDepthIdxs[j];
				if (idxPoint != NO_ID) {
#if FUSE_DIAGNOSTICS
					dmgPreclaimed(i, j) = depth;
					++cPixRefPreclaimed;
					dmgFate(i, j) = Pixel8U(255, 0, 0); // BLUE in BGR
#endif
					continue;
				}

				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.NumPoints();

				const double pointXWithDepth = pointXNoDepthPreTransform * depth;
				const double pointYWithDepth = pointYNoDepthPreTransform * depth;
				const double pointZWithDepth = depth;

				Point3f point;
				point.x =
					refR00 * pointXWithDepth
					+ refR10 * pointYWithDepth
					+ refR20 * pointZWithDepth
					+ refCx;
				point.y =
					refR01 * pointXWithDepth
					+ refR11 * pointYWithDepth
					+ refR21 * pointZWithDepth
					+ refCy;
				point.z =
					refR02 * pointXWithDepth
					+ refR12 * pointYWithDepth
					+ refR22 * pointZWithDepth
					+ refCz;

				// ============================================================
				// PHASE 1: Cheap projection + depth/normal gate only.
				//          No confidence, no back-projection, no color.
				// ============================================================
				nHits = 0;
				nInvalidHits = 0;

				PointCloud::Normal normal;
				if (bNormalMap) {
					const Normal& n = pNormalMap[j];
					normal.x =
						refR00 * n.x
						+ refR10 * n.y
						+ refR20 * n.z;
					normal.y =
						refR01 * n.x
						+ refR11 * n.y
						+ refR21 * n.z;
					normal.z =
						refR02 * n.x
						+ refR12 * n.y
						+ refR22 * n.z;
				}
				else {
					normal = { 0.f, 0.f, -1.f };
				}

				_Data vPointX = _Set(point.x);
				_Data vPointY = _Set(point.y);
				_Data vPointZ = _Set(point.z);

				const unsigned ncCount = (unsigned)neighborCache.size();
				// +1 accounts for the reference view
				const unsigned minHitsNeeded = (nMinViewsFuse > 1) ? (nMinViewsFuse - 1) : 0;
				for (unsigned ncI = 0; ncI < ncCount; ++ncI) {
					// Abort if it is impossible to reach nMinViewsFuse even if all
					// remaining neighbors hit. Invalidation won't run in that case either.
					if (nHits + (ncCount - ncI) < minHitsNeeded)
						break;

					const auto& nc = neighborCache[ncI];
					DepthMap& depthMapB = *nc.depthMapB;

					_Data col0_point = _Mul(nc.col0, vPointX);
					_Data col1_point = _Mul(nc.col1, vPointY);
					_Data col2_point = _Mul(nc.col2, vPointZ);

					_Data xyz_1 = _Add(col0_point, col1_point);
					_Data xyz_2 = _Add(col2_point, nc.col3);
					_Data xyz0 = _Add(xyz_1, xyz_2);
					_Data zzz = _Splat(xyz0, 2);
					_Data zzz_wh10 = _Mul(zzz, nc.zzz_wh10_bounds);

					_Data result = _CmpGT(xyz0, zzz_wh10);
					_Data result2 = _CmpLT(xyz0, _SetZero());
					_Data orResult = _Or(result, result2);
#if FUSE_DIAGNOSTICS
					++cNeighborsTotal;
#endif
					if (!AllZerosI(_CastIF(orResult))) {
#if FUSE_DIAGNOSTICS
						++cNeighborOOB;
#endif
						continue;
					}

					// Compute neighbor pixel coordinate with reference-matching rounding.
					// The reference uses ROUND2INT(pt.x/pt.z) (round-to-nearest). The
					// previous SIMD _ConvertIF has platform-dependent rounding semantics
					// and could truncate, picking an off-by-one neighbor pixel that
					// samples a wildly different surface on slanted regions (roofs,
					// facades) and makes per-view fuse disagree on otherwise-flat data.
					const float ptz = _vFirst(zzz);
					const float invZ = 1.0f / ptz;
					alignas(16) float xyz0Arr[4];
					_mm_store_ps(xyz0Arr, xyz0);
					const ImageRef xB(ROUND2INT(xyz0Arr[0] * invZ), ROUND2INT(xyz0Arr[1] * invZ));

					Depth& depthB = depthMapB.pix(xB);
					if (depthB == 0) {
#if FUSE_DIAGNOSTICS
						++cNeighborZeroDepth;
#endif
						continue;
					}

					uint32_t& idxPointB = nc.depthIdxB->pix(xB);
					if (idxPointB != NO_ID) {
#if FUSE_DIAGNOSTICS
						++cNeighborPreclaimed;
#endif
						continue;
					}

					if (FastAbsS(ptz - depthB) < fDepthDiffThresholdFuse * ptz) {
						// Depth is similar � but only do the cheap normal gate here
						PointCloud::Normal normalB;
						// BUGFIX: only consult neighbor normal map if it actually exists.
						// Previously bNormalMap (ref-side flag) gated a read into the neighbor's
						// (possibly empty) normalMap, producing garbage normals and culling hits.
						const bool bCompareNormals = bNormalMap && !nc.normalMapBEmpty;
						if (bCompareNormals) {
							const Normal& nb = nc.normalMapB->pix(xB);
							const TRMatrixBase<float>& imageCameraRt = *nc.cameraRt;
							normalB.x =
								imageCameraRt[0 * 3 + 0] * nb.x
								+ imageCameraRt[1 * 3 + 0] * nb.y
								+ imageCameraRt[2 * 3 + 0] * nb.z;
							normalB.y =
								imageCameraRt[0 * 3 + 1] * nb.x
								+ imageCameraRt[1 * 3 + 1] * nb.y
								+ imageCameraRt[2 * 3 + 1] * nb.z;
							normalB.z =
								imageCameraRt[0 * 3 + 2] * nb.x
								+ imageCameraRt[1 * 3 + 2] * nb.y
								+ imageCameraRt[2 * 3 + 2] * nb.z;
						}
						else {
							normalB = { 0.f, 0.f, -1.f };
						}

						// BUGFIX: if either side has no normal, skip the gate (accept hit).
						const float dotNB = bCompareNormals
							? (normal.x * normalB.x + normal.y * normalB.y + normal.z * normalB.z)
							: 1.f;
						if (dotNB > normalError) {
							NeighborHit& h = hitsStorage[nHits++];
							h.ncIdx = ncI;
							h.xB = xB;
							h.ptz = ptz;
							h.depthB = depthB;
							h.pDepthB = &depthB;
							h.pIdxPointB = &idxPointB;
#if FUSE_DIAGNOSTICS
							++cNeighborAccepted;
#endif
							continue;
						}
#if FUSE_DIAGNOSTICS
						++cNeighborNormalFail;
#endif
					}
					else {
#if FUSE_DIAGNOSTICS
						++cNeighborDepthMismatch;
#endif
					}

					// Depth not similar or normal check failed � candidate for invalidation
					if (ptz < depthB) {
						NeighborHit& h = invalidHitsStorage[nInvalidHits++];
						h.ncIdx = ncI;
						h.xB = xB;
						h.ptz = ptz;
						h.depthB = depthB;
						h.pDepthB = &depthB;
						h.pIdxPointB = &idxPointB;
					}
				} // END Phase 1 neighbor loop

				// DIAGNOSTIC: set inner #if to 1 to bypass the min-views consensus gate.
				// A point is emitted for every valid reference depth, even with 0 hits.
				// If sparse regions fill in with this active, fuse is the culprit.
				// If they stay sparse, the depth maps themselves lack those pixels
				// (PatchMatch / filter did not produce them).
				#if 0
				// bypassed
				#else
				// +1 for the reference view itself
				if (nHits + 1 < nMinViewsFuse) {
					idxPoint = NO_ID;
#if FUSE_DIAGNOSTICS
					dmgCulled(i, j) = depth;
					++cPixRefCulled;
					cHitsHistCulled[nHits < 16 ? nHits : 15]++;
					dmgFate(i, j) = Pixel8U(0, 0, 255); // RED in BGR
#endif
					continue;
				}
				#endif

				// ============================================================
				// PHASE 2: Only reached when we know the point will survive.
				//          Now do confidence, back-projection, color.
				// ============================================================
				nViews = 0;
				nDeferred = 0;

				viewsStorage[nViews] = idxImage;
				REAL confidence = Conf2Weight(*pConfMap, depth);
				weightsStorage[nViews] = confidence;
				++nViews;
				float origConfidence = confidence;

				Point3 X(point * confidence);
				PointCloud::Normal N(normal * confidence);

				float convergenceR = 0, convergenceG = 0, convergenceB = 0;

				for (unsigned hi = 0; hi < nHits; ++hi) {
					const NeighborHit& h = hitsStorage[hi];
					const auto& nc = neighborCache[h.ncIdx];

					const float confidenceB = nc.confMapBEmpty
						? (1.f / (0.03f * h.depthB * h.depthB))
						: Conf2Weight((*nc.confMapB)(h.xB), h.depthB);
					viewsStorage[nViews] = nc.idxImageB;
					weightsStorage[nViews] = confidenceB;
					++nViews;
#ifdef ESTIMATE_NORMALS
					projs.push_back(Proj(h.xB));
#endif
					deferredStorage[nDeferred++] = h.pIdxPointB;

					double cx = (((double)h.xB.x) - nc.K02) * h.depthB * nc.invK00;
					double cy = (((double)h.xB.y) - nc.K12) * h.depthB * nc.invK11;
					double cz = h.depthB;

					auto offsetX = nc.R00 * cx + nc.R10 * cy + nc.R20 * cz;
					auto offsetY = nc.R01 * cx + nc.R11 * cy + nc.R21 * cz;
					auto offsetZ = nc.R02 * cx + nc.R12 * cy + nc.R22 * cz;

					offsetX += nc.Cx;
					offsetY += nc.Cy;
					offsetZ += nc.Cz;

					offsetX *= confidenceB;
					offsetY *= confidenceB;
					offsetZ *= confidenceB;

					X.x += offsetX;
					X.y += offsetY;
					X.z += offsetZ;

					if (bEstimateColor) {
						const Pixel8U& pixel = nc.imageDataB->image.pix(h.xB);
						convergenceR += pixel.r * confidenceB;
						convergenceG += pixel.g * confidenceB;
						convergenceB += pixel.b * confidenceB;
					}

#ifdef ESTIMATE_NORMALS
					if (bEstimateNormal) {
						PointCloud::Normal normalB;
						if (bNormalMap) {
							const Normal& nb = nc.normalMapB->pix(h.xB);
							const TRMatrixBase<float>& imageCameraRt = *nc.cameraRt;
							normalB.x = imageCameraRt[0 * 3 + 0] * nb.x + imageCameraRt[1 * 3 + 0] * nb.y + imageCameraRt[2 * 3 + 0] * nb.z;
							normalB.y = imageCameraRt[0 * 3 + 1] * nb.x + imageCameraRt[1 * 3 + 1] * nb.y + imageCameraRt[2 * 3 + 1] * nb.z;
							normalB.z = imageCameraRt[0 * 3 + 2] * nb.x + imageCameraRt[1 * 3 + 2] * nb.y + imageCameraRt[2 * 3 + 2] * nb.z;
						}
						else {
							normalB = { 0.f, 0.f, -1.f };
						}
						N += normalB * confidenceB;
					}
#endif
					confidence += confidenceB;
				} // END Phase 2

				// Commit deferred idxPointB assignments
				for (unsigned di = 0; di < nDeferred; ++di)
					*deferredStorage[di] = idxPoint;

				// Sort views+weights in lockstep by view ID
				for (unsigned k = 1; k < nViews; ++k) {
					uint32_t tmpV = viewsStorage[k];
					float    tmpW = weightsStorage[k];
					unsigned hole = k;
					while (hole > 0 && viewsStorage[hole - 1] > tmpV) {
						viewsStorage[hole] = viewsStorage[hole - 1];
						weightsStorage[hole] = weightsStorage[hole - 1];
						--hole;
					}
					viewsStorage[hole] = tmpV;
					weightsStorage[hole] = tmpW;
				}

				const REAL nrm(REAL(1) / confidence);
				point = X * nrm;
				ASSERT(ISFINITE(point));

				pointcloud.AddPoint(point);
				pointcloud.AddViews(viewsStorage, viewsStorage + nViews);
				pointcloud.AddWeights(weightsStorage, weightsStorage + nViews);
#if FUSE_DIAGNOSTICS
				dmgSurvived(i, j) = depth;
				++cPixRefSurvived;
				dmgFate(i, j) = Pixel8U(0, 255, 0); // GREEN in BGR
#endif

				if (bEstimateColor) {
					const auto& baseColor = pImage[j];
					float r = baseColor.r * origConfidence + convergenceR;
					float g = baseColor.g * origConfidence + convergenceG;
					float b = baseColor.b * origConfidence + convergenceB;
					r *= nrm;
					g *= nrm;
					b *= nrm;
					pointcloud.AddColor(Pixel8U(_cvt_ftoi_fast(r), _cvt_ftoi_fast(g), _cvt_ftoi_fast(b)));
				}

				// Match original: invalidate neighbor depths that violated free-space
				// for the surviving point. Unconditional (original did not gate on
				// extra consensus or world-space distance), but still safe because
				// hits reach invalidHits only when ptz < depthB.
				for (unsigned hi = 0; hi < nInvalidHits; ++hi) {
					const NeighborHit& h = invalidHitsStorage[hi];
					*h.pDepthB = 0;
				}
			}
		}

#if FUSE_DIAGNOSTICS
		ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "fuse.survived.png"),   dmgSurvived);
		ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "fuse.culled.png"),     dmgCulled);
		ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "fuse.preclaimed.png"), dmgPreclaimed);
		dmgFate.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "fuse.fate.png"));
		VERBOSE("FUSE DIAG img=%3u  refDepths=%llu  survived=%llu (%.1f%%)  preclaimed=%llu (%.1f%%)  culled=%llu (%.1f%%)",
			depthData.GetView().GetID(),
			(unsigned long long)cPixRefDepths,
			(unsigned long long)cPixRefSurvived,   cPixRefDepths ? 100.0 * cPixRefSurvived   / cPixRefDepths : 0.0,
			(unsigned long long)cPixRefPreclaimed, cPixRefDepths ? 100.0 * cPixRefPreclaimed / cPixRefDepths : 0.0,
			(unsigned long long)cPixRefCulled,     cPixRefDepths ? 100.0 * cPixRefCulled     / cPixRefDepths : 0.0);
		VERBOSE("FUSE DIAG img=%3u  nbrProbes=%llu  OOB=%llu  zeroDepth=%llu  preclaimed=%llu  depthMismatch=%llu  normalFail=%llu  accepted=%llu",
			depthData.GetView().GetID(),
			(unsigned long long)cNeighborsTotal,
			(unsigned long long)cNeighborOOB,
			(unsigned long long)cNeighborZeroDepth,
			(unsigned long long)cNeighborPreclaimed,
			(unsigned long long)cNeighborDepthMismatch,
			(unsigned long long)cNeighborNormalFail,
			(unsigned long long)cNeighborAccepted);
		VERBOSE("FUSE DIAG img=%3u  culled-nHits-hist: 0=%llu 1=%llu 2=%llu 3=%llu 4=%llu 5=%llu 6+=%llu",
			depthData.GetView().GetID(),
			(unsigned long long)cHitsHistCulled[0],
			(unsigned long long)cHitsHistCulled[1],
			(unsigned long long)cHitsHistCulled[2],
			(unsigned long long)cHitsHistCulled[3],
			(unsigned long long)cHitsHistCulled[4],
			(unsigned long long)cHitsHistCulled[5],
			(unsigned long long)(cHitsHistCulled[6]+cHitsHistCulled[7]+cHitsHistCulled[8]+cHitsHistCulled[9]+cHitsHistCulled[10]+cHitsHistCulled[11]+cHitsHistCulled[12]+cHitsHistCulled[13]+cHitsHistCulled[14]+cHitsHistCulled[15]));
#endif

		fusedDMaps[idxImage] = true;
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size() && pointcloud.points.size() == pointcloud.pointWeights.size() && pointcloud.points.size() == projs.size());
		DEBUG_ULTIMATE("Depth-map for reference image %3u fused using %u depth-maps: %u new points, %u/%u cached images (%s)",
			idxImage, depthData.images.size() - 1, pointcloud.NumPoints() - nNumPointsPrev, numImageNeighborsInCache, numImagesInCache, TD_TIMER_GET_FMT().c_str());
		progress.display(numDMapsFused);
		// ensure enough memory is available for the next depth-maps chunk
		cacheDMaps.SkipMemoryCheckIdxImage();
		if (numDMapsFused % numDMapsReserveFusion == 0)
			cacheDMaps.SetMaxMemory(GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion, cacheDMaps.GetUsedMemory()));
	}

	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();
	cacheDMaps.ClearCache();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%), %.2f hits in %.2f cached (%s)",
		numDMapsFused, nDepths, pointcloud.NumPoints(), ROUND2INT((100.f * pointcloud.NumPoints()) / nDepths),
		static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused,
		static_cast<double>(totalNumImagesInCache) / numDMapsFused, TD_TIMER_GET_FMT().c_str());

#if 0 // JPB WIP BUG
	if (bEstimateNormal && !pointcloud.points.empty() && pointcloud.normals.empty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		pointcloud.normals.resize(pointcloud.points.size());
		const int64_t nPoints((int64_t)pointcloud.points.size());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.empty());
			IIndex idxView(0);
			float bestWeight = weights.front();
			for (IIndex idx=1; idx<weights.size(); ++idx) {
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
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
#endif
} // FuseDepthMaps

//#pragma optimize("", on) // JPB WIP BUG Debugging


// ===================================================================
// DenseFuseDepthMaps: Merrell-style recursive fusion adapted to the
// PointCloudStreaming output. Algorithm mirrors the reference
// DenseFuseDepthMaps (component-wise median, recursive neighbor graph
// traversal, reprojection-error gate, confidence gate). Plumbing
// (DMapCache, FetchBestNextDMapIndex, memory reservations) mirrors the
// fast FuseDepthMaps above.
//
// Tunables for nMaxFuseDepth / nMaxPointsFuse / nMinPixelsFuse /
// fDepthReprojectionErrorThreshold / nMaxViewsFuse are not in this
// branch's OPTDENSE; reasonable defaults are hardcoded here matching
// the reference defaults. Promote to OPTDENSE later if needed.
// ===================================================================
void DepthMapsData::DenseFuseDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool _bEstimateNormal)
{
	TD_TIMER_STARTD();

	typedef SEACAVE::BitMatrix UseMask;
	typedef CLISTDEFIDX(UseMask, IIndex) UseMaskArr;

	// =====================================================================
	// DENSE_FUSE_RELAXED: set to 1 to relax the four most-aggressive cull
	// gates (deeper recursion, looser reprojection tolerance, accept
	// singleton clusters, drop confidence gate). Use when dense-fuse is
	// eating into legitimate surfaces.
	//   0 = strict (reference-like)
	//   1 = relaxed (closer to fast-fuse coverage with median benefits)
	// =====================================================================
	#define DENSE_FUSE_RELAXED 1

	// dense-fuse tunables (would be OPTDENSE in upstream)
	#if DENSE_FUSE_RELAXED
	constexpr unsigned kMaxFuseDepth     = 6;
	constexpr unsigned kMaxPointsFuse    = 24;
	constexpr unsigned kMinPixelsFuse    = 1;
	constexpr float    kReprojErrSq      = 9.0f; // 3px tolerance squared
	constexpr IIndex   kMaxViewsFuse     = 32;
	#else
	constexpr unsigned kMaxFuseDepth     = 4;
	constexpr unsigned kMaxPointsFuse    = 24;
	constexpr unsigned kMinPixelsFuse    = 2;
	constexpr float    kReprojErrSq      = 4.0f; // 2px tolerance squared
	constexpr IIndex   kMaxViewsFuse     = 32;
	#endif

	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, arrDepthData.size()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));
	#if DENSE_FUSE_RELAXED
	const float minConfidence(0.f);
	#else
	const float minConfidence(1.f - OPTDENSE::fNCCThresholdKeep);
	#endif
	const IIndex numDMapsReserveFusion(10);
	const bool bEstimateNormal(true); // always estimate normals: needed for the fuse gate
	size_t nDepths(0);

	UseMaskArr arrUseMask(arrDepthData.size());
	const size_t nPointsEstimate(arrDepthData.size() * 9000);
	pointcloud.ReservePoints(nPointsEstimate);
	pointcloud.ReservePointViewsSizeAndOffset(nPointsEstimate);
	pointcloud.ReservePointWeightsSizeAndOffset(nPointsEstimate);
	unsigned depthDataLoadFlags(HeaderDepthDataRaw::HAS_DEPTH | HeaderDepthDataRaw::HAS_CONF);
	if (bEstimateColor)
		pointcloud.ReserveColors(nPointsEstimate);
	if (bEstimateNormal) {
		pointcloud.ReserveNormals(nPointsEstimate);
		depthDataLoadFlags |= HeaderDepthDataRaw::HAS_NORMAL;
	}

	// Reserve flat views/weights memory the same way FuseDepthMaps does.
	{
		MEMORYSTATUS memState{};
		::GlobalMemoryStatus(&memState);
		const size_t bytesAvailable = memState.dwAvailPhys;
		const size_t elementSize =
			std::max(
				sizeof(decltype(pointcloud.pointViewsMemory)::value_type),
				sizeof(decltype(pointcloud.pointWeightsMemory)::value_type));
		const size_t nElementsAvailableToUse = (bytesAvailable / elementSize) / 4;
		pointcloud.ReservePointViewsMemory(nElementsAvailableToUse / 4);
		pointcloud.ReservePointWeightsMemory(nElementsAvailableToUse / 4);
	}

	Util::Progress progress(_T("Dense fused depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	BoolArr fusedDMaps(arrDepthData.size());
	fusedDMaps.Memset(0);
	DMapCache cacheDMaps(arrDepthData, depthDataLoadFlags,
		GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion));
	unsigned totalNumImageNeighborsInCache = 0, totalNumImagesInCache = 0;
	BoolArr neighbors(arrDepthData.size());

	// =====================================================================
	// DENSE_FUSE_HYBRID: when a cluster has fewer than kHybridThresholdViews
	// supporting views, emit each accumulated pixel as its own point
	// (merge-style). Where coverage is good, emit one fused median point
	// (fuse-style). This gives reconstruct enough density in sparse-coverage
	// regions (corners, edges) while keeping the cloud small elsewhere.
	//   0 = pure fuse (median-only output)
	//   1 = hybrid (sparse clusters expand to per-pixel points)
	// =====================================================================
	#define DENSE_FUSE_HYBRID 1
	constexpr unsigned kHybridThresholdViews = 3;

	// =====================================================================
	// DENSE_FUSE_OPT_TUNING: port the hand-tuned per-image camera/pointer
	// cache from FuseDepthMaps into the recursive lambda. Algorithm is
	// unchanged; this only replaces repeated camera-method dispatch and
	// arrDepthData[i] indirections with cached pointers + inlined matrix
	// multiplies. Same precision (double for projection math, float for
	// normals/colors), same operation order. A/B-able for verification.
	//   0 = original code paths
	//   1 = cached camera/pointer fast paths
	// =====================================================================
	#define DENSE_FUSE_OPT_TUNING 1

	#if DENSE_FUSE_OPT_TUNING
	struct DFImageCache {
		bool ready = false;
		// Stable cross-iteration pointers (live for the lifetime of the
		// scene/depth-data arrays; DMapCache may toggle .empty() on the
		// underlying maps, so we re-check empty() dynamically.)
		const DepthMap*      pDepthMap  = nullptr;
		const ConfidenceMap* pConfMap   = nullptr;
		const NormalMap*     pNormalMap = nullptr;
		const Image*         pImageData = nullptr;
		SEACAVE::BitMatrix*  pUseMask   = nullptr; // &arrUseMask[id]
		const ViewScoreArr*  pNeighbors = nullptr; // &pImageData->neighbors
		int width  = 0;                    // depthMap dimensions (fixed per image)
		int height = 0;
		// Camera-derived constants (double, matches reference precision)
		double P[12];                      // 3x4 projection, row-major
		double invK00, invK11;             // intrinsic inverses
		double K02, K12;                   // principal point
		double R00, R10, R20;              // R^T row 0  (= R col 0)
		double R01, R11, R21;              // R^T row 1
		double R02, R12, R22;              // R^T row 2
		double Cx, Cy, Cz;                 // camera center
	};
	std::vector<DFImageCache> imgCache(arrDepthData.size());
	// Fill camera-derived + stable pointer fields. Called once per image at
	// the top of its outer iteration (and lazily as a fallback). Width/height
	// and useMask must be patched later if not yet known.
	const auto InitImageCache = [&](IIndex id) -> DFImageCache& {
		DFImageCache& c = imgCache[id];
		if (c.ready) return c;
		const DepthData& dd = arrDepthData[id];
		const DepthData::ViewData& vd = dd.GetView();
		c.pDepthMap  = &dd.depthMap;
		c.pConfMap   = &dd.confMap;
		c.pNormalMap = &dd.normalMap;
		c.pImageData = vd.pImageData;
		c.pUseMask   = &arrUseMask[id];
		c.pNeighbors = &vd.pImageData->neighbors;
		// Image dimensions are fixed per image; pull from camera if depthMap
		// not yet allocated.
		if (!dd.depthMap.empty()) {
			c.width  = dd.depthMap.width();
			c.height = dd.depthMap.height();
		} else {
			c.width  = (int)vd.pImageData->width;
			c.height = (int)vd.pImageData->height;
		}
		const Camera& cam = vd.camera;
		const REAL* const Pv = cam.P.val;
		for (int k = 0; k < 12; ++k) c.P[k] = (double)Pv[k];
		c.invK00 = 1.0 / (double)cam.K(0,0);
		c.invK11 = 1.0 / (double)cam.K(1,1);
		c.K02 = (double)cam.K(0,2);
		c.K12 = (double)cam.K(1,2);
		// R is the world->camera rotation; R^T = R transposed.
		// (R^T * v)_i = sum_j R(j,i) * v[j].  Cache R(j,i) flat.
		c.R00 = (double)cam.R(0,0); c.R10 = (double)cam.R(1,0); c.R20 = (double)cam.R(2,0);
		c.R01 = (double)cam.R(0,1); c.R11 = (double)cam.R(1,1); c.R21 = (double)cam.R(2,1);
		c.R02 = (double)cam.R(0,2); c.R12 = (double)cam.R(1,2); c.R22 = (double)cam.R(2,2);
		c.Cx = (double)cam.C.x; c.Cy = (double)cam.C.y; c.Cz = (double)cam.C.z;
		c.ready = true;
		return c;
	};
	#endif

	// Per-cluster accumulators. Reused across pixels (cleared after each emit).
	Point3f refPoint(0.f, 0.f, 0.f);
	Point3f refNormal(0.f, 0.f, -1.f);
	CLISTDEF0IDX(float, unsigned) fusedPoints[3];
	std::vector<uint32_t> fusedViews;
	std::vector<float>    fusedWeights;
	Point3d fusedNormal;
	Pixel32F fusedColor;
	// Per-pixel parallel arrays (only used by hybrid emit). Same length as
	// fusedPoints[]; index k = the k-th accepted pixel in the cluster.
	#if DENSE_FUSE_HYBRID
	std::vector<uint32_t> pxView;     // view ID for pixel k
	std::vector<float>    pxWeight;   // Conf2Weight for pixel k
	std::vector<Pixel8U>  pxColor;    // raw pixel color (only if bEstimateColor)
	std::vector<Point3f>  pxNormal;   // world-space normal for pixel k (if normals)
	#endif

	// ---- Per-image diagnostics (set to 0 once stable) -------------------
	#define DENSE_FUSE_DIAG 0
	#if DENSE_FUSE_DIAG
	uint64_t cSeedCalls=0, cSeedOOB=0, cSeedDmEmpty=0, cSeedZeroDepth=0,
	         cSeedAlreadyUsed=0, cSeedLowConf=0, cSeedAccepted=0;
	uint64_t cRecCalls=0, cRecOOB=0, cRecDmEmpty=0, cRecZeroDepth=0,
	         cRecAlreadyUsed=0, cRecLowConf=0, cRecBehindCam=0,
	         cRecDepthMismatch=0, cRecReprojFail=0, cRecNormalFail=0,
	         cRecAccepted=0;
	uint64_t cClustersTried=0, cClustersEmitted=0, cClustersTooSmall=0,
	         cClustersTooFewViews=0;
	#endif
	// ---------------------------------------------------------------------

	const auto FusePoint = [&](IIndex ID, const ImageRef& x, unsigned fuseDepth) -> void {
		const auto lambda = [&](IIndex curID, const ImageRef& curX, unsigned curDepth, const auto& Self) -> void {
			#if DENSE_FUSE_DIAG
			if (curDepth == 0) ++cSeedCalls; else ++cRecCalls;
			#endif
			#if DENSE_FUSE_OPT_TUNING
			const DFImageCache& pic = imgCache[curID];
			const DepthMap&     curDepthMap  = *pic.pDepthMap;
			const ConfidenceMap& curConfMap  = *pic.pConfMap;
			const NormalMap&    curNormalMap = *pic.pNormalMap;
			const Image&        curImageData = *pic.pImageData;
			// Depth-map may be empty if DMapCache evicted it; bail safely.
			if (curDepthMap.empty()) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedDmEmpty; else ++cRecDmEmpty;
				#endif
				return;
			}
			// In-bounds check via cached dims (replaces isInside + cv::Size ctor).
			if ((unsigned)curX.x >= (unsigned)pic.width ||
			    (unsigned)curX.y >= (unsigned)pic.height) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedOOB; else ++cRecOOB;
				#endif
				return;
			}
			const Depth depth = curDepthMap(curX);
			if (depth <= Depth(0)) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedZeroDepth; else ++cRecZeroDepth;
				#endif
				return;
			}
			UseMask& useMask = *pic.pUseMask;
			// useMask is always allocated for IDs in the active recursion
			// graph (outer loop creates it before the pixel sweep), so the
			// .empty() check from the original is unnecessary here.
			if (useMask(curX)) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedAlreadyUsed; else ++cRecAlreadyUsed;
				#endif
				return;
			}
			const float conf(curConfMap.empty() ? 1.f : curConfMap(curX));
			if (conf < minConfidence) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedLowConf; else ++cRecLowConf;
				#endif
				return;
			}
			// If a normal map is available, compute world-space normal; else use ref's.
			const bool bHaveNormal = !curNormalMap.empty();
			Point3f normal;
			if (curDepth > 0) {
				// Inlined ProjectPointP3 in double precision (matches reference math).
				const double rx = (double)refPoint.x;
				const double ry = (double)refPoint.y;
				const double rz = (double)refPoint.z;
				const double ptx_d = pic.P[0]*rx + pic.P[1]*ry + pic.P[2 ]*rz + pic.P[3 ];
				const double pty_d = pic.P[4]*rx + pic.P[5]*ry + pic.P[6 ]*rz + pic.P[7 ];
				const double ptz_d = pic.P[8]*rx + pic.P[9]*ry + pic.P[10]*rz + pic.P[11];
				// Match original: ProjectPointP3 returns Point3d then is narrowed
				// to Point3f, so the z<=0 gate runs on the *float* truncated value.
				const Point3f pt((float)ptx_d, (float)pty_d, (float)ptz_d);
				if (pt.z <= Depth(0)) {
					#if DENSE_FUSE_DIAG
					++cRecBehindCam;
					#endif
					return;
				}
				if (!IsDepthSimilar(depth, pt.z, OPTDENSE::fDepthDiffThreshold)) {
					#if DENSE_FUSE_DIAG
					++cRecDepthMismatch;
					#endif
					return;
				}
				const Point2f diff(pt.x / pt.z - float(curX.x), pt.y / pt.z - float(curX.y));
				if (normSq(diff) > kReprojErrSq) {
					#if DENSE_FUSE_DIAG
					++cRecReprojFail;
					#endif
					return;
				}
				if (bHaveNormal) {
					// Inlined: normal_world = R^T * normalMap(curX)
					const Normal& nLocal = curNormalMap(curX);
					const double nx = (double)nLocal.x;
					const double ny = (double)nLocal.y;
					const double nz = (double)nLocal.z;
					normal.x = (float)(pic.R00*nx + pic.R10*ny + pic.R20*nz);
					normal.y = (float)(pic.R01*nx + pic.R11*ny + pic.R21*nz);
					normal.z = (float)(pic.R02*nx + pic.R12*ny + pic.R22*nz);
					// Only enforce normal gate if we also had a ref normal.
					if (refNormal.z != -1.f || refNormal.x != 0.f || refNormal.y != 0.f) {
						if (refNormal.dot(normal) < normalError) {
							#if DENSE_FUSE_DIAG
							++cRecNormalFail;
							#endif
							return;
						}
					}
				} else {
					normal = refNormal;
				}
				#if DENSE_FUSE_DIAG
				++cRecAccepted;
				#endif
			} else {
				if (bHaveNormal) {
					const Normal& nLocal = curNormalMap(curX);
					const double nx = (double)nLocal.x;
					const double ny = (double)nLocal.y;
					const double nz = (double)nLocal.z;
					normal.x = (float)(pic.R00*nx + pic.R10*ny + pic.R20*nz);
					normal.y = (float)(pic.R01*nx + pic.R11*ny + pic.R21*nz);
					normal.z = (float)(pic.R02*nx + pic.R12*ny + pic.R22*nz);
				} else {
					normal = Point3f(0.f, 0.f, -1.f);
				}
				#if DENSE_FUSE_DIAG
				++cSeedAccepted;
				#endif
			}
			useMask.set(curX);
			// Inlined TransformPointI2W in double precision.
			const double cx_d = ((double)curX.x - pic.K02) * pic.invK00 * (double)depth;
			const double cy_d = ((double)curX.y - pic.K12) * pic.invK11 * (double)depth;
			const double cz_d = (double)depth;
			const Point3f X(
				(float)(pic.R00*cx_d + pic.R10*cy_d + pic.R20*cz_d + pic.Cx),
				(float)(pic.R01*cx_d + pic.R11*cy_d + pic.R21*cz_d + pic.Cy),
				(float)(pic.R02*cx_d + pic.R12*cy_d + pic.R22*cz_d + pic.Cz));
			#else
			const DepthData& depthDataCur = arrDepthData[curID];
			// Depth-map may be empty if DMapCache evicted it; bail safely.
			if (depthDataCur.depthMap.empty()) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedDmEmpty; else ++cRecDmEmpty;
				#endif
				return;
			}
			if (!Image8U::isInside(curX, depthDataCur.depthMap.size())) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedOOB; else ++cRecOOB;
				#endif
				return;
			}
			const Depth depth = depthDataCur.depthMap(curX);
			if (depth <= Depth(0)) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedZeroDepth; else ++cRecZeroDepth;
				#endif
				return;
			}
			UseMask& useMask = arrUseMask[curID];
			if (useMask.empty() || useMask(curX)) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedAlreadyUsed; else ++cRecAlreadyUsed;
				#endif
				return;
			}
			const float conf(depthDataCur.confMap.empty() ? 1.f : depthDataCur.confMap(curX));
			if (conf < minConfidence) {
				#if DENSE_FUSE_DIAG
				if (curDepth == 0) ++cSeedLowConf; else ++cRecLowConf;
				#endif
				return;
			}
			const DepthData::ViewData& image = depthDataCur.GetView();
			// If a normal map is available, compute world-space normal; else use ref's.
			const bool bHaveNormal = !depthDataCur.normalMap.empty();
			Point3f normal;
			if (curDepth > 0) {
				// Project the seed back into this view; check depth + reprojection.
				const Point3f pt(image.camera.ProjectPointP3(Cast<REAL>(refPoint)));
				if (pt.z <= Depth(0)) {
					#if DENSE_FUSE_DIAG
					++cRecBehindCam;
					#endif
					return;
				}
				if (!IsDepthSimilar(depth, pt.z, OPTDENSE::fDepthDiffThreshold)) {
					#if DENSE_FUSE_DIAG
					++cRecDepthMismatch;
					#endif
					return;
				}
				const Point2f diff(pt.x / pt.z - float(curX.x), pt.y / pt.z - float(curX.y));
				if (normSq(diff) > kReprojErrSq) {
					#if DENSE_FUSE_DIAG
					++cRecReprojFail;
					#endif
					return;
				}
				if (bHaveNormal) {
					normal = Cast<float>(image.camera.R.t() * Cast<REAL>(depthDataCur.normalMap(curX)));
					// Only enforce normal gate if we also had a ref normal.
					if (refNormal.z != -1.f || refNormal.x != 0.f || refNormal.y != 0.f) {
						if (refNormal.dot(normal) < normalError) {
							#if DENSE_FUSE_DIAG
							++cRecNormalFail;
							#endif
							return;
						}
					}
				} else {
					normal = refNormal;
				}
				#if DENSE_FUSE_DIAG
				++cRecAccepted;
				#endif
			} else {
				if (bHaveNormal)
					normal = Cast<float>(image.camera.R.t() * Cast<REAL>(depthDataCur.normalMap(curX)));
				else
					normal = Point3f(0.f, 0.f, -1.f);
				#if DENSE_FUSE_DIAG
				++cSeedAccepted;
				#endif
			}
			useMask.set(curX);
			const Point3f X(Cast<float>(image.camera.TransformPointI2W(Point3(REAL(curX.x), REAL(curX.y), REAL(depth)))));
			#endif

			// Accumulate into the fused-point cluster.
			fusedPoints[0].push_back(X.x);
			fusedPoints[1].push_back(X.y);
			fusedPoints[2].push_back(X.z);
			const float weight(Conf2Weight(conf, depth));
			// Insert-sorted-unique into fusedViews; accumulate into fusedWeights at same index.
			{
				const uint32_t vID = (uint32_t)curID;
				auto it = std::lower_bound(fusedViews.begin(), fusedViews.end(), vID);
				const size_t idx = (size_t)(it - fusedViews.begin());
				if (it != fusedViews.end() && *it == vID) {
					fusedWeights[idx] += weight;
				} else {
					fusedViews.insert(it, vID);
					fusedWeights.insert(fusedWeights.begin() + idx, weight);
				}
			}
			if (bEstimateNormal)
				fusedNormal += Cast<double>(normal);
			if (bEstimateColor)
			#if DENSE_FUSE_OPT_TUNING
				fusedColor += Cast<float>(curImageData.image(curX));
			#else
				fusedColor += Cast<float>(image.pImageData->image(curX));
			#endif
			#if DENSE_FUSE_HYBRID
			pxView.push_back((uint32_t)curID);
			pxWeight.push_back(weight);
			if (bEstimateColor)
			#if DENSE_FUSE_OPT_TUNING
				pxColor.push_back(curImageData.image(curX));
			#else
				pxColor.push_back(image.pImageData->image(curX));
			#endif
			if (bEstimateNormal)
				pxNormal.push_back(normal);
			#endif

			if (curDepth == 0) {
				refPoint = X;
				refNormal = normal;
			}

			if (++curDepth >= kMaxFuseDepth || fusedPoints[0].size() >= kMaxPointsFuse)
				return;

			// Recurse into the neighbor graph.
			#if DENSE_FUSE_OPT_TUNING
			for (const ViewScore& neighbor : *pic.pNeighbors) {
				const IIndex nextID(neighbor.ID);
				if (nextID == curID)
					continue;
				if (!neighbors[nextID])
					continue;
				// Inlined ProjectPointP (next camera) in double precision.
				const DFImageCache& npic = imgCache[nextID];
				const double Xx = (double)X.x;
				const double Xy = (double)X.y;
				const double Xz = (double)X.z;
				const double nptx = npic.P[0]*Xx + npic.P[1]*Xy + npic.P[2 ]*Xz + npic.P[3 ];
				const double npty = npic.P[4]*Xx + npic.P[5]*Xy + npic.P[6 ]*Xz + npic.P[7 ];
				const double nptz = npic.P[8]*Xx + npic.P[9]*Xy + npic.P[10]*Xz + npic.P[11];
				// Match original ProjectPointP: invert-z then multiply (not divide).
				const double invNptz = 1.0 / nptz;
				const ImageRef nextx(ROUND2INT(Point2(nptx*invNptz, npty*invNptz)));
				Self(nextID, nextx, curDepth, Self);
			}
			#else
			for (const ViewScore& neighbor : image.pImageData->neighbors) {
				const IIndex nextID(neighbor.ID);
				if (nextID == curID)
					continue;
				if (!neighbors[nextID])
					continue;
				const DepthData& nextDepthData = arrDepthData[nextID];
				const ImageRef nextx(ROUND2INT(nextDepthData.GetCamera().ProjectPointP(Cast<REAL>(X))));
				Self(nextID, nextx, curDepth, Self);
			}
			#endif
		};
		lambda(ID, x, fuseDepth, lambda);
	};

	IIndex numDMapsFused = 0;
	while (true) {
		TD_TIMER_STARTD();
		const auto [idxImage, numImageNeighborsInCache, numImagesInCache] =
			FetchBestNextDMapIndex(arrDepthData, cacheDMaps, fusedDMaps);
		if (idxImage == NO_ID)
			break;
		totalNumImageNeighborsInCache += numImageNeighborsInCache;
		totalNumImagesInCache += numImagesInCache;
		++numDMapsFused;

		cacheDMaps.UseImage(idxImage);
		cacheDMaps.SkipMemoryCheckIdxImage(idxImage);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		ASSERT(!depthData.IsEmpty());

		if (bEstimateNormal && depthData.normalMap.empty())
#if 1
			throw std::runtime_error("Unsupported");
#else
			EstimateNormalMaps();
#endif

		// Mark the active recursion graph: ref + valid+non-empty neighbors (capped).
		neighbors.Memset(0);
		neighbors[idxImage] = true;
		IIndex numNeighbors(0);
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
		for (const ViewScore& neighbor : depthData.neighbors) {
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			neighbors[neighbor.ID] = true;
			UseMask& useMaskB = arrUseMask[neighbor.ID];
			if (useMaskB.empty()) {
				useMaskB.create(depthDataB.depthMap.size());
				useMaskB.memset(0);
			}
			if (++numNeighbors >= kMaxViewsFuse)
				break;
		}

		const Image& imageData = *depthData.images.front().pImageData;
		ASSERT(&imageData - scene.images.data() == idxImage);
		// Use depthMap.size() (the live map) instead of depthData.size
		// (which is unreliable in the streaming pipeline).
		const Image8U::Size sizeMap(depthData.depthMap.size());
		UseMask& useMaskRef = arrUseMask[idxImage];
		if (useMaskRef.empty()) {
			useMaskRef.create(sizeMap);
			useMaskRef.memset(0);
		}

		#if DENSE_FUSE_OPT_TUNING
		// Pre-populate caches for ref + all active recursion neighbors. All
		// depth-maps are loaded at this point (UseImage above), so width/
		// height pull from the live depthMap dims (correct even for scaled
		// depth-map pipelines).
		InitImageCache(idxImage);
		for (const ViewScore& neighbor : depthData.neighbors) {
			if (neighbors[neighbor.ID])
				InitImageCache(neighbor.ID);
		}
		#endif

		const size_t nNumPointsPrev(pointcloud.NumPoints());

		for (int i = 0; i < sizeMap.height; ++i) {
			for (int j = 0; j < sizeMap.width; ++j) {
				FusePoint(idxImage, ImageRef(j, i), 0);

				#if DENSE_FUSE_DIAG
				if (!fusedViews.empty()) ++cClustersTried;
				#endif

				if (fusedPoints[0].size() >= kMinPixelsFuse && fusedViews.size() >= nMinViewsFuse) {
					#if DENSE_FUSE_HYBRID
					if (fusedViews.size() < kHybridThresholdViews) {
						// Sparse-coverage cluster: emit each pixel as its own
						// point (merge-style) so reconstruct sees enough density
						// to keep the surface in the graph cut.
						const size_t nPx = fusedPoints[0].size();
						for (size_t k = 0; k < nPx; ++k) {
							pointcloud.AddPoint(Point3f(
								fusedPoints[0][(unsigned)k],
								fusedPoints[1][(unsigned)k],
								fusedPoints[2][(unsigned)k]));
							pointcloud.AddView(pxView[k]);
							pointcloud.AddWeight(pxWeight[k]);
							if (bEstimateNormal)
								pointcloud.AddNormal(pxNormal[k]);
							if (bEstimateColor)
								pointcloud.AddColor(pxColor[k]);
						}
						#if DENSE_FUSE_DIAG
						++cClustersEmitted;
						#endif
					} else
					#endif
					{
					// Median (component-wise) is robust to one bad depth in the cluster.
					Point3f p(
						fusedPoints[0].GetMedian(),
						fusedPoints[1].GetMedian(),
						fusedPoints[2].GetMedian());
					pointcloud.AddPoint(p);
					pointcloud.AddViews(fusedViews.data(), fusedViews.data() + fusedViews.size());
					pointcloud.AddWeights(fusedWeights.data(), fusedWeights.data() + fusedWeights.size());
					if (bEstimateNormal) {
						const Point3d nrm(normalized(fusedNormal));
						pointcloud.AddNormal(Point3f((float)nrm.x, (float)nrm.y, (float)nrm.z));
					}
					if (bEstimateColor) {
						const float invN = 1.f / static_cast<float>(fusedPoints[0].size());
						pointcloud.AddColor(Pixel8U(
							_cvt_ftoi_fast(fusedColor.r * invN),
							_cvt_ftoi_fast(fusedColor.g * invN),
							_cvt_ftoi_fast(fusedColor.b * invN)));
					}
					#if DENSE_FUSE_DIAG
					++cClustersEmitted;
					#endif
					}
				}
				#if DENSE_FUSE_DIAG
				else if (!fusedViews.empty()) {
					if (fusedPoints[0].size() < kMinPixelsFuse) ++cClustersTooSmall;
					else                                        ++cClustersTooFewViews;
				}
				#endif

				if (!fusedViews.empty()) {
					nDepths += fusedViews.size();
					fusedPoints[0].clear();
					fusedPoints[1].clear();
					fusedPoints[2].clear();
					fusedViews.clear();
					fusedWeights.clear();
					fusedNormal = Point3d::ZERO;
					fusedColor = Pixel32F::BLACK;
					#if DENSE_FUSE_HYBRID
					pxView.clear();
					pxWeight.clear();
					pxColor.clear();
					pxNormal.clear();
					#endif
				}
			}
		}

		#if DENSE_FUSE_DIAG
		VERBOSE("DENSE-FUSE DIAG img=%3u  seedCalls=%llu accepted=%llu  rejects: oob=%llu dmEmpty=%llu zeroDepth=%llu used=%llu lowConf=%llu",
			depthData.GetView().GetID(),
			(unsigned long long)cSeedCalls, (unsigned long long)cSeedAccepted,
			(unsigned long long)cSeedOOB, (unsigned long long)cSeedDmEmpty,
			(unsigned long long)cSeedZeroDepth, (unsigned long long)cSeedAlreadyUsed,
			(unsigned long long)cSeedLowConf);
		VERBOSE("DENSE-FUSE DIAG img=%3u  recCalls=%llu accepted=%llu  rejects: oob=%llu dmEmpty=%llu zeroDepth=%llu used=%llu lowConf=%llu behindCam=%llu depthMis=%llu reproj=%llu normal=%llu",
			depthData.GetView().GetID(),
			(unsigned long long)cRecCalls, (unsigned long long)cRecAccepted,
			(unsigned long long)cRecOOB, (unsigned long long)cRecDmEmpty,
			(unsigned long long)cRecZeroDepth, (unsigned long long)cRecAlreadyUsed,
			(unsigned long long)cRecLowConf, (unsigned long long)cRecBehindCam,
			(unsigned long long)cRecDepthMismatch, (unsigned long long)cRecReprojFail,
			(unsigned long long)cRecNormalFail);
		VERBOSE("DENSE-FUSE DIAG img=%3u  clusters tried=%llu emitted=%llu tooSmall=%llu tooFewViews=%llu",
			depthData.GetView().GetID(),
			(unsigned long long)cClustersTried, (unsigned long long)cClustersEmitted,
			(unsigned long long)cClustersTooSmall, (unsigned long long)cClustersTooFewViews);
		// reset per-image counters
		cSeedCalls=cSeedOOB=cSeedDmEmpty=cSeedZeroDepth=cSeedAlreadyUsed=cSeedLowConf=cSeedAccepted=0;
		cRecCalls=cRecOOB=cRecDmEmpty=cRecZeroDepth=cRecAlreadyUsed=cRecLowConf=cRecBehindCam=0;
		cRecDepthMismatch=cRecReprojFail=cRecNormalFail=cRecAccepted=0;
		cClustersTried=cClustersEmitted=cClustersTooSmall=cClustersTooFewViews=0;
		#endif

		fusedDMaps[idxImage] = true;
		DEBUG_ULTIMATE("Depth-map for reference image %3u dense-fused using %u depth-maps: %u new points, %u/%u cached images (%s)",
			idxImage, depthData.images.size() - 1, pointcloud.NumPoints() - nNumPointsPrev,
			numImageNeighborsInCache, numImagesInCache, TD_TIMER_GET_FMT().c_str());
		progress.display(numDMapsFused);
		cacheDMaps.SkipMemoryCheckIdxImage();
		if (numDMapsFused % numDMapsReserveFusion == 0)
			cacheDMaps.SetMaxMemory(GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion, cacheDMaps.GetUsedMemory()));
	}

	GET_LOGCONSOLE().Play();
	progress.close();
	arrUseMask.Release();
	cacheDMaps.ClearCache();

	DEBUG_EXTRA("Depth-maps dense fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%), %.2f hits in %.2f cached (%s)",
		numDMapsFused, nDepths, pointcloud.NumPoints(),
		nDepths ? ROUND2INT((100.f * pointcloud.NumPoints()) / nDepths) : 0,
		numDMapsFused ? static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused : 0.0,
		numDMapsFused ? static_cast<double>(totalNumImagesInCache) / numDMapsFused : 0.0,
		TD_TIMER_GET_FMT().c_str());
} // DenseFuseDepthMaps


#else
#if 0
// fuse all valid depth-maps in the same 3D point-cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
void DepthMapsData::DenseFuseDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool _bEstimateNormal)
{
	TD_TIMER_STARTD();

	typedef SEACAVE::BitMatrix UseMask;
	typedef CLISTDEFIDX(UseMask,IIndex) UseMaskArr;

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, arrDepthData.size()));
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));
	const float minConfidence(1.f - OPTDENSE::fNCCThresholdKeep);
	const float maxReprojErrorSq(SQUARE(OPTDENSE::fDepthReprojectionErrorThreshold));
	const IIndex numDMapsReserveFusion(10);
	const bool bEstimateNormal(true); // always estimate normals as they are needed for the fusion
	size_t nDepths(0);
	UseMaskArr arrUseMask(arrDepthData.size());
	const size_t nPointsEstimate(arrDepthData.size() * 9000); //TODO: better estimate number of points
	pointcloud.points.reserve(nPointsEstimate);
	pointcloud.pointViews.reserve(nPointsEstimate);
	pointcloud.pointWeights.reserve(nPointsEstimate);
	unsigned depthDataLoadFlags(HeaderDepthDataRaw::HAS_DEPTH | HeaderDepthDataRaw::HAS_CONF);
	if (bEstimateColor)
		pointcloud.colors.reserve(nPointsEstimate);
	if (bEstimateNormal) {
		pointcloud.normals.reserve(nPointsEstimate);
		depthDataLoadFlags |= HeaderDepthDataRaw::HAS_NORMAL;
	}
	Util::Progress progress(_T("Dense fused depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	BoolArr fusedDMaps(arrDepthData.size());
	fusedDMaps.Memset(0);
	DMapCache cacheDMaps(arrDepthData, depthDataLoadFlags, GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion));
	unsigned totalNumImageNeighborsInCache = 0, totalNumImagesInCache = 0;
	BoolArr neighbors(arrDepthData.size());
	PointCloud::Point refPoint;
	PointCloud::Normal refNormal;
	CLISTDEF0IDX(float, unsigned) fusedPoints[3];
	PointCloud::ViewArr fusedViews;
	FloatArr fusedWeights;
	Point3d fusedNormal;
	Pixel32F fusedColor;
	const auto FusePoint = [&](IIndex ID, const ImageRef& x, unsigned fuseDepth) -> void {
		const auto lambda = [&](IIndex ID, const ImageRef& x, unsigned fuseDepth, const auto& FusePointImpl) -> void {
			const DepthData& depthData = arrDepthData[ID];
			if (!Image8U::isInside(x, depthData.size))
				return;
			// ignore pixel if not estimated
			ASSERT(depthData.depthMap.size() == depthData.size);
			const Depth depth = depthData.depthMap(x);
			if (depth <= Depth(0))
				return;
			ASSERT(ISINSIDE(depth, depthData.dMin * 0.95f, depthData.dMax * 1.05f));
			// ignore pixel if already fused
			UseMask& useMask = arrUseMask[ID];
			if (useMask(x))
				return;
			// ignore pixel if not confident
			const float conf(depthData.confMap.empty() ? 1.f : depthData.confMap(x));
			if (conf < minConfidence)
				return;
			const DepthData::ViewData& image = depthData.GetView();
			// if the fusion depth is greater than zero, the initial reference pixel
			// has already been added and we need to check for consistency
			PointCloud::Normal normal;
			if (fuseDepth > 0) {
				// project reference point into current view
				const Point3f pt(image.camera.ProjectPointP3(refPoint));
				// check if depth agrees with current depth
				ASSERT(pt.z > Depth(0) || !IsDepthSimilar(depth, pt.z, OPTDENSE::fDepthDiffThreshold));
				if (!IsDepthSimilar(depth, pt.z, OPTDENSE::fDepthDiffThreshold))
					return;
				// check reprojection error of the reference point in the current view
				const Point2f diff(pt.x / pt.z - float(x.x), pt.y / pt.z - float(x.y));
				if (normSq(diff) > maxReprojErrorSq)
					return;
				// check if normals agree
				normal = image.camera.R.t() * Cast<REAL>(depthData.normalMap(x));
				ASSERT(ISEQUAL(norm(normal), 1.f, 1e-2f), "Norm = ", norm(normal));
				if (refNormal.dot(normal) < normalError)
					return;
			} else {
				normal = image.camera.R.t() * Cast<REAL>(depthData.normalMap(x));
				ASSERT(ISEQUAL(norm(normal), 1.f, 1e-2f), "Norm = ", norm(normal));
			}
			// set the current pixel as visited
			useMask.set(x);
			// compute 3D location of the current depth
			const PointCloud::Point X(image.camera.TransformPointI2W(Point3(REAL(x.x), REAL(x.y), REAL(depth))));
			// accumulate statistics for fused point
			{
				fusedPoints[0].push_back(X(0));
				fusedPoints[1].push_back(X(1));
				fusedPoints[2].push_back(X(2));
				const float weight(Conf2Weight(conf, depth));
				const auto it(fusedViews.InsertSortUnique(ID));
				if (it.second)
					fusedWeights[it.first] += weight;
				else
					fusedWeights.InsertAt(it.first, weight);
				if (bEstimateNormal)
					fusedNormal += Cast<double>(normal);
				if (bEstimateColor)
					fusedColor += Cast<float>(image.pImageData->image(x));
			}
			// remember the first pixel as the reference.
			if (fuseDepth == 0) {
				refPoint = X;
				refNormal = normal;
			}
			// do not traverse the graph infinitely in one branch and
			// limit the maximum number of pixels fused in one point
			// to avoid stack overflow
			if (++fuseDepth >= OPTDENSE::nMaxFuseDepth || fusedPoints[0].size() >= OPTDENSE::nMaxPointsFuse)
				return;
			// traverse the neighbors graph by projecting the point into other views
			for (const ViewScore& neighbor : image.pImageData->neighbors) {
				const IIndex nextID(neighbor.ID);
				ASSERT(nextID != ID);
				if (!neighbors[nextID])
					continue;
				const DepthData& nextDepthData = arrDepthData[nextID];
				const ImageRef nextx(ROUND2INT(nextDepthData.GetCamera().ProjectPointP(X)));
				FusePointImpl(nextID, nextx, fuseDepth, FusePointImpl);
			}
		};
		lambda(ID, x, fuseDepth, lambda);
	};
	// loop over each depth-map
	IIndex numDMapsFused = 0;
	while (true) {
		TD_TIMER_STARTD();
		// find the best depth-map to fuse next as the one with the most neighbors already in cache
		const auto [idxImage, numImageNeighborsInCache, numImagesInCache] = FetchBestNextDMapIndex(arrDepthData, cacheDMaps, fusedDMaps);
		if (idxImage == NO_ID)
			break; // no more depth-maps to fuse (only invalid depth-maps left)
		totalNumImageNeighborsInCache += numImageNeighborsInCache;
		totalNumImagesInCache += numImagesInCache;
		++numDMapsFused;
		// fuse depth-map
		cacheDMaps.UseImage(idxImage);
		cacheDMaps.SkipMemoryCheckIdxImage(idxImage);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		ASSERT(!depthData.IsEmpty());
		if (bEstimateNormal && depthData.normalMap.empty())
			EstimateNormalMaps();
		// make sure all neighbors are cached
		neighbors.Memset(0);
		neighbors[idxImage] = true;
		IIndex numNeighbors(0);
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t)depthData.neighbors.size(); ++i) {
#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const ViewScore& neighbor = depthData.neighbors[(IIndex)i];
#else
		for (const ViewScore& neighbor : depthData.neighbors) {
#endif
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			neighbors[neighbor.ID] = true;
			UseMask& useMask = arrUseMask[neighbor.ID];
			if (!useMask.empty())
				continue;
			useMask.create(depthDataB.depthMap.size());
			useMask.memset(0);
			if (++numNeighbors >= OPTDENSE::nMaxViewsFuse) {
#ifdef DENSE_USE_OPENMP
				bAbort = true;
#pragma omp flush (bAbort)
#else
				break;
#endif
			}
		}
		ASSERT(!depthData.IsEmpty());
		MAYBEUNUSED const Image& imageData = *depthData.images.front().pImageData;
		ASSERT(&imageData - scene.images.data() == idxImage);
		ASSERT(depthData.depthMap.size() == depthData.size && imageData.GetSize() == depthData.size);
		UseMask& useMask = arrUseMask[idxImage];
		if (useMask.empty()) {
			useMask.create(depthData.size);
			useMask.memset(0);
		}
		// try to fuse each depth estimate
		const size_t nNumPointsPrev(pointcloud.points.size());
		for (int i = 0; i < depthData.size.height; ++i) {
			for (int j = 0; j < depthData.size.width; ++j) {
				FusePoint(idxImage, ImageRef(j, i), 0);
				if (fusedPoints[0].size() >= OPTDENSE::nMinPixelsFuse && fusedViews.size() >= nMinViewsFuse) {
					// create the corresponding 3D point
					pointcloud.points.emplace_back(
						fusedPoints[0].GetMedian(),
						fusedPoints[1].GetMedian(),
						fusedPoints[2].GetMedian()
					);
					ASSERT(fusedViews.size() == fusedWeights.size());
					PointCloud::WeightArr& weights = pointcloud.pointWeights.AddEmpty();
					for (float weight : fusedWeights)
						weights.push_back(weight);
					pointcloud.pointViews.emplace_back(fusedViews);
					if (bEstimateNormal)
						pointcloud.normals.emplace_back(normalized(fusedNormal));
					if (bEstimateColor)
						pointcloud.colors.emplace_back((fusedColor / static_cast<float>(fusedPoints[0].size())).cast<uint8_t>());
				}
				if (!fusedViews.empty()) {
					nDepths += fusedViews.size();
					fusedPoints[0].clear();
					fusedPoints[1].clear();
					fusedPoints[2].clear();
					fusedViews.clear();
					fusedWeights.clear();
					fusedNormal = Point3d::ZERO;
					fusedColor = Pixel32F::BLACK;
				}
			}
		}
		fusedDMaps[idxImage] = true;
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size() && pointcloud.points.size() == pointcloud.pointWeights.size());
		DEBUG_ULTIMATE("Depth-map for reference image %3u fused using %u depth-maps: %u new points, %u/%u cached images (%s)",
			idxImage, depthData.images.size() - 1, pointcloud.points.size() - nNumPointsPrev, numImageNeighborsInCache, numImagesInCache, TD_TIMER_GET_FMT().c_str());
		progress.display(numDMapsFused);
		// ensure enough memory is available for the next depth-maps chunk
		cacheDMaps.SkipMemoryCheckIdxImage();
		if (numDMapsFused % numDMapsReserveFusion == 0)
			cacheDMaps.SetMaxMemory(GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion, cacheDMaps.GetUsedMemory()));
		}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrUseMask.Release();
	cacheDMaps.ClearCache();
	if (!_bEstimateNormal)
		pointcloud.normals.Release();

	DEBUG_EXTRA("Depth-maps dense fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%), %.2f hits in %.2f cached (%s)",
		numDMapsFused, nDepths, pointcloud.points.size(), ROUND2INT((100.f * pointcloud.points.size()) / nDepths),
		static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused,
		static_cast<double>(totalNumImagesInCache) / numDMapsFused, TD_TIMER_GET_FMT().c_str());
	} // DenseFuseDepthMaps

// fuse all valid depth-maps in the same 3D point cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
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
		inline ImageRef GetCoord() const { return ImageRef(x, y); }
	};

#ifdef ESTIMATE_NORMALS
  throw std::runtime_error("Normal estimation is not supported in this version.");
	typedef SEACAVE::cList<Proj, const Proj&, 0, 4, uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr, const ProjArr&, 1, 65536> ProjsArr;
#endif

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
	size_t nDepths = 0;
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());

#ifdef ESTIMATE_NORMALS
	ProjsArr projsarr(0, nPointsEstimate);
	if (bEstimateNormal && !bNormalMap)
		bEstimateNormal = false;
#endif

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

	const size_t nElementsAvailableToUse = (bytesAvailable / elementSize) / 4;
	pointcloud.ReservePointViewsMemory(nElementsAvailableToUse / 4);
	pointcloud.ReservePointWeightsMemory(nElementsAvailableToUse / 4);

	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
	GET_LOGCONSOLE().Pause();

	for (int ci = 0; ci < connections.size(); ++ci) {
		auto* pConnection = connections.begin() + ci;
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor : depthData.neighbors) {
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
		ASSERT(&imageData - scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const float confMapSentinel = 1.f;
		size_t confMapInc;

		const Point3 imageDataHBasis =
			imageData.camera.TransformPointI2W(Point3(Point2f(1, 0), 1.))
			- imageData.camera.TransformPointI2W(Point3(Point2f(0, 0), 1.));

		const size_t numNeighbors = depthData.neighbors.size();
		const size_t nNumPointsPrev(pointcloud.NumPoints());

#ifdef ESTIMATE_NORMALS
		boost::container::small_vector<Proj, 16> projs;
#endif

		struct NeighborCache {
			IIndex idxImageB;
			_Data col0, col1, col2, col3;
			_Data zzz_wh10_bounds; // _SetN(w-1, h-1, 1.f, 0.f)
			DepthMap* depthMapB;
			DepthIndex* depthIdxB;
			const ConfidenceMap* __restrict confMapB;
			const NormalMap* __restrict normalMapB;
			const Image* __restrict imageDataB;
			TRMatrixBase<float> const* __restrict cameraRt;
			Camera const* camera;
			bool confMapBEmpty;
			// Pre-computed inverse-K constants for Phase 2 back-projection
			double invK00, invK11, K02, K12;
			double Cx, Cy, Cz;
			// R transposed columns for back-projection: rot(row,col) = R[col*3+row] in row-major
			double R00, R10, R20, R01, R11, R21, R02, R12, R22;
		};
		boost::container::small_vector<NeighborCache, 16> neighborCache;
		neighborCache.reserve(depthData.neighbors.size());
		for (const auto& neighbor : depthData.neighbors) {
			NeighborCache nc;
			nc.idxImageB = neighbor.ID;
			const auto& pt = imagesCameraPt[nc.idxImageB];
			nc.col0 = _Load(&pt[0]);
			nc.col1 = _Load(&pt[4]);
			nc.col2 = _Load(&pt[8]);
			nc.col3 = _Load(&pt[12]);
			DepthData& ddb = arrDepthData[nc.idxImageB];
			nc.depthMapB = &ddb.depthMap;
			nc.zzz_wh10_bounds = _SetN(
				(float)(ddb.depthMap.width() - 1),
				(float)(ddb.depthMap.height() - 1), 1.f, 0.f);
			nc.depthIdxB = &arrDepthIdx[nc.idxImageB];
			nc.confMapB = &ddb.confMap;
			nc.confMapBEmpty = ddb.confMap.empty();
			nc.normalMapB = &ddb.normalMap;
			nc.imageDataB = &scene.images[nc.idxImageB];
			nc.cameraRt = &imagesCameraRt[nc.idxImageB];
			nc.camera = &scene.images[nc.idxImageB].camera;
			// Pre-compute inverse-K and camera constants for Phase 2
			nc.invK00 = 1.0 / nc.camera->K(0, 0);
			nc.invK11 = 1.0 / nc.camera->K(1, 1);
			nc.K02 = nc.camera->K(0, 2);
			nc.K12 = nc.camera->K(1, 2);
			nc.Cx = nc.camera->C.x;
			nc.Cy = nc.camera->C.y;
			nc.Cz = nc.camera->C.z;
			const auto& rot = nc.camera->R;
			nc.R00 = rot(0, 0); nc.R10 = rot(1, 0); nc.R20 = rot(2, 0);
			nc.R01 = rot(0, 1); nc.R11 = rot(1, 1); nc.R21 = rot(2, 1);
			nc.R02 = rot(0, 2); nc.R12 = rot(1, 2); nc.R22 = rot(2, 2);
			neighborCache.push_back(nc);
		}

		const float fDepthDiffThreshold = OPTDENSE::fDepthDiffThreshold;
		const _Data vTwo = _Set(2.f);

			// --- Lightweight candidate structure for two-phase neighbor check ---
		struct NeighborHit {
			unsigned ncIdx;    // index into neighborCache
			ImageRef xB;
			float    ptz;
			Depth    depthB;   // cached so Phase 2 doesn't re-read
			Depth*   pDepthB;  // pointer for invalidation / deferred commit
			uint32_t* pIdxPointB; // pointer for deferred commit
		};
		const unsigned maxNeighbors = (unsigned)neighborCache.size();
		NeighborHit* hitsStorage = (NeighborHit*)_alloca(maxNeighbors * sizeof(NeighborHit));
		NeighborHit* invalidHitsStorage = (NeighborHit*)_alloca(maxNeighbors * sizeof(NeighborHit));
		unsigned nHits = 0, nInvalidHits = 0;

		const unsigned maxViews = (unsigned)neighborCache.size() + 1; // +1 for reference view
		uint32_t* __restrict viewsStorage = (uint32_t*)_alloca(maxViews * sizeof(uint32_t));
		float* __restrict weightsStorage = (float*)_alloca(maxViews * sizeof(float));
		// Defer idxPointB assignments until we know the point survives the fuse check.
		// Collect pointers to idxPointB slots so we can commit them only for accepted points.
		uint32_t** __restrict deferredStorage = (uint32_t**)_alloca(neighborCache.size() * sizeof(uint32_t*));
		unsigned nViews = 0, nDeferred = 0;

		for (int i = 0; i < sizeMap.height; ++i) {
			const Depth* __restrict pDM = &depthData.depthMap(i, 0);
			uint32_t* __restrict pDepthIdxs = (uint32_t*)&depthIdxs(i, 0);

			const float* __restrict pConfMap;
			bool confMapEmpty = depthData.confMap.empty();
			if (confMapEmpty) {
				pConfMap = &confMapSentinel;
				confMapInc = 0;
			}
			else {
				confMapInc = 1;
			}

			if (!confMapEmpty) {
				pConfMap = &depthData.confMap(i, 0);
			}

			const Normal* __restrict pNormalMap = &depthData.normalMap(i, 0);
			const Pixel8U* __restrict pImage = &imageData.image(i, 0);

			Point3 imagePoint =
				imageData.camera.TransformPointI2W(Point3(Point2f(0, i), 1.)) - imageData.camera.C;

			const double invImageDataCameraK00 = 1. / imageData.camera.K(0, 0);
			const double invImageDataCameraK11 = 1. / imageData.camera.K(1, 1);
			const double imageDataCameraK02 = imageData.camera.K(0, 2);
			const double imageDataCameraK12 = imageData.camera.K(1, 2);
			double pointXNoDepthPreTransform = -imageDataCameraK02 * invImageDataCameraK00;
			double pointXNoDepthPreTransformDelta = invImageDataCameraK00;
			double pointYNoDepthPreTransform = (i - imageDataCameraK12) * invImageDataCameraK11;

			for (int j = 0; j < sizeMap.width; ++j, pConfMap += confMapInc, imagePoint += imageDataHBasis, pointXNoDepthPreTransform += pointXNoDepthPreTransformDelta) {
				const Depth depth(pDM[j]);
				if (depth == 0)
					continue;

				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = pDepthIdxs[j];
				if (idxPoint != NO_ID)
					continue;

				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.NumPoints();

				const double pointXWithDepth = pointXNoDepthPreTransform * depth;
				const double pointYWithDepth = pointYNoDepthPreTransform * depth;
				const double pointZWithDepth = depth;

				Point3f point;
				point.x =
					imageData.camera.R[0 * 3 + 0] * pointXWithDepth
					+ imageData.camera.R[1 * 3 + 0] * pointYWithDepth
					+ imageData.camera.R[2 * 3 + 0] * pointZWithDepth
					+ imageData.camera.C.x;
				point.y =
					imageData.camera.R[0 * 3 + 1] * pointXWithDepth
					+ imageData.camera.R[1 * 3 + 1] * pointYWithDepth
					+ imageData.camera.R[2 * 3 + 1] * pointZWithDepth
					+ imageData.camera.C.y;
				point.z =
					imageData.camera.R[0 * 3 + 2] * pointXWithDepth
					+ imageData.camera.R[1 * 3 + 2] * pointYWithDepth
					+ imageData.camera.R[2 * 3 + 2] * pointZWithDepth
					+ imageData.camera.C.z;

				// ============================================================
				// PHASE 1: Cheap projection + depth/normal gate only.
				//          No confidence, no back-projection, no color.
				// ============================================================
				nHits = 0;
				nInvalidHits = 0;

				PointCloud::Normal normal;
				if (bNormalMap) {
					const Normal& n = pNormalMap[j];
					normal.x =
						imageData.camera.R[0 * 3 + 0] * n.x
						+ imageData.camera.R[1 * 3 + 0] * n.y
						+ imageData.camera.R[2 * 3 + 0] * n.z;
					normal.y =
						imageData.camera.R[0 * 3 + 1] * n.x
						+ imageData.camera.R[1 * 3 + 1] * n.y
						+ imageData.camera.R[2 * 3 + 1] * n.z;
					normal.z =
						imageData.camera.R[0 * 3 + 2] * n.x
						+ imageData.camera.R[1 * 3 + 2] * n.y
						+ imageData.camera.R[2 * 3 + 2] * n.z;
				}
				else {
					normal = { 0.f, 0.f, -1.f };
				}

				_Data vPointX = _Set(point.x);
				_Data vPointY = _Set(point.y);
				_Data vPointZ = _Set(point.z);

				const unsigned ncCount = (unsigned)neighborCache.size();
				for (unsigned ncI = 0; ncI < ncCount; ++ncI) {
					const auto& nc = neighborCache[ncI];
					DepthMap& depthMapB = *nc.depthMapB;

					_Data col0_point = _Mul(nc.col0, vPointX);
					_Data col1_point = _Mul(nc.col1, vPointY);
					_Data col2_point = _Mul(nc.col2, vPointZ);

					_Data xyz_1 = _Add(col0_point, col1_point);
					_Data xyz_2 = _Add(col2_point, nc.col3);
					_Data xyz0 = _Add(xyz_1, xyz_2);
					_Data zzz = _Splat(xyz0, 2);
					_Data zzz_wh10 = _Mul(zzz, nc.zzz_wh10_bounds);

					_Data result = _CmpGT(xyz0, zzz_wh10);
					_Data result2 = _CmpLT(xyz0, _SetZero());
					_Data orResult = _Or(result, result2);
					if (!AllZerosI(_CastIF(orResult)))
						continue;

					// Fast reciprocal with one Newton-Raphson refinement (~4 cycles vs ~14 for _Div)
					_Data rcp = _mm_rcp_ps(zzz);
					_Data invZZZ = _Mul(rcp, _Sub(vTwo, _Mul(zzz, rcp)));
					xyz0 = _Mul(xyz0, invZZZ);
					float ptz = _vFirst(zzz);
					_DataI xyz0AsInt = _ConvertIF(xyz0);
					const ImageRef xB(_AsArrayI(xyz0AsInt, 0), _AsArrayI(xyz0AsInt, 1));

					Depth& depthB = depthMapB.pix(xB);
					if (depthB == 0)
						continue;

					uint32_t& idxPointB = nc.depthIdxB->pix(xB);
					if (idxPointB != NO_ID)
						continue;

					if (FastAbsS(ptz - depthB) < fDepthDiffThreshold * ptz) {
						// Depth is similar � but only do the cheap normal gate here
						PointCloud::Normal normalB;
						if (bNormalMap) {
							const Normal& nb = nc.normalMapB->pix(xB);
							const TRMatrixBase<float>& imageCameraRt = *nc.cameraRt;
							normalB.x =
								imageCameraRt[0 * 3 + 0] * nb.x
								+ imageCameraRt[1 * 3 + 0] * nb.y
								+ imageCameraRt[2 * 3 + 0] * nb.z;
							normalB.y =
								imageCameraRt[0 * 3 + 1] * nb.x
								+ imageCameraRt[1 * 3 + 1] * nb.y
								+ imageCameraRt[2 * 3 + 1] * nb.z;
							normalB.z =
								imageCameraRt[0 * 3 + 2] * nb.x
								+ imageCameraRt[1 * 3 + 2] * nb.y
								+ imageCameraRt[2 * 3 + 2] * nb.z;
						} else {
							normalB = { 0.f, 0.f, -1.f };
						}

						const float dotNB = normal.x * normalB.x + normal.y * normalB.y + normal.z * normalB.z;
						if (dotNB > normalError) {
							NeighborHit& h = hitsStorage[nHits++];
							h.ncIdx = ncI;
							h.xB = xB;
							h.ptz = ptz;
							h.depthB = depthB;
							h.pDepthB = &depthB;
							h.pIdxPointB = &idxPointB;
							continue;
						}
					}

					// Depth not similar or normal check failed � candidate for invalidation
					if (ptz < depthB) {
						NeighborHit& h = invalidHitsStorage[nInvalidHits++];
						h.ncIdx = ncI;
						h.xB = xB;
						h.ptz = ptz;
						h.depthB = depthB;
						h.pDepthB = &depthB;
						h.pIdxPointB = &idxPointB;
					}
				} // END Phase 1 neighbor loop

				// +1 for the reference view itself
				if (nHits + 1 < nMinViewsFuse) {
					idxPoint = NO_ID;
					continue;
				}

				// ============================================================
				// PHASE 2: Only reached when we know the point will survive.
				//          Now do confidence, back-projection, color.
				// ============================================================
				nViews = 0;
				nDeferred = 0;

				viewsStorage[nViews] = idxImage;
				REAL confidence = Conf2Weight(*pConfMap, depth);
				weightsStorage[nViews] = confidence;
				++nViews;
				float origConfidence = confidence;

				Point3 X(point * confidence);
				PointCloud::Normal N(normal * confidence);

				float convergenceR = 0, convergenceG = 0, convergenceB = 0;

				for (unsigned hi = 0; hi < nHits; ++hi) {
					const NeighborHit& h = hitsStorage[hi];
					const auto& nc = neighborCache[h.ncIdx];

					const float confidenceB = nc.confMapBEmpty
						? (1.f / (0.03f * h.depthB * h.depthB))
						: Conf2Weight((*nc.confMapB)(h.xB), h.depthB);
					viewsStorage[nViews] = nc.idxImageB;
					weightsStorage[nViews] = confidenceB;
					++nViews;
#ifdef ESTIMATE_NORMALS
					projs.push_back(Proj(h.xB));
#endif
					deferredStorage[nDeferred++] = h.pIdxPointB;

					double cx = (((double)h.xB.x) - nc.K02) * h.depthB * nc.invK00;
					double cy = (((double)h.xB.y) - nc.K12) * h.depthB * nc.invK11;
					double cz = h.depthB;

					auto offsetX = nc.R00 * cx + nc.R10 * cy + nc.R20 * cz;
					auto offsetY = nc.R01 * cx + nc.R11 * cy + nc.R21 * cz;
					auto offsetZ = nc.R02 * cx + nc.R12 * cy + nc.R22 * cz;

					offsetX += nc.Cx;
					offsetY += nc.Cy;
					offsetZ += nc.Cz;

					offsetX *= confidenceB;
					offsetY *= confidenceB;
					offsetZ *= confidenceB;

					X.x += offsetX;
					X.y += offsetY;
					X.z += offsetZ;

					if (bEstimateColor) {
						const Pixel8U& pixel = nc.imageDataB->image.pix(h.xB);
						convergenceR += pixel.r * confidenceB;
						convergenceG += pixel.g * confidenceB;
						convergenceB += pixel.b * confidenceB;
					}

#ifdef ESTIMATE_NORMALS
					if (bEstimateNormal) {
						PointCloud::Normal normalB;
						if (bNormalMap) {
							const Normal& nb = nc.normalMapB->pix(h.xB);
							const TRMatrixBase<float>& imageCameraRt = *nc.cameraRt;
							normalB.x = imageCameraRt[0*3+0]*nb.x + imageCameraRt[1*3+0]*nb.y + imageCameraRt[2*3+0]*nb.z;
							normalB.y = imageCameraRt[0*3+1]*nb.x + imageCameraRt[1*3+1]*nb.y + imageCameraRt[2*3+1]*nb.z;
							normalB.z = imageCameraRt[0*3+2]*nb.x + imageCameraRt[1*3+2]*nb.y + imageCameraRt[2*3+2]*nb.z;
						} else {
							normalB = { 0.f, 0.f, -1.f };
						}
						N += normalB * confidenceB;
					}
#endif
					confidence += confidenceB;
				} // END Phase 2

				// Commit deferred idxPointB assignments
				for (unsigned di = 0; di < nDeferred; ++di)
					*deferredStorage[di] = idxPoint;

				// Sort views+weights in lockstep by view ID
				for (unsigned k = 1; k < nViews; ++k) {
					uint32_t tmpV = viewsStorage[k];
					float    tmpW = weightsStorage[k];
					unsigned hole = k;
					while (hole > 0 && viewsStorage[hole - 1] > tmpV) {
						viewsStorage[hole]   = viewsStorage[hole - 1];
						weightsStorage[hole] = weightsStorage[hole - 1];
						--hole;
					}
					viewsStorage[hole]   = tmpV;
					weightsStorage[hole] = tmpW;
				}

				const REAL nrm(REAL(1) / confidence);
				point = X * nrm;
				ASSERT(ISFINITE(point));

				pointcloud.AddPoint(point);
				pointcloud.AddViews(viewsStorage, viewsStorage + nViews);
				pointcloud.AddWeights(weightsStorage, weightsStorage + nViews);

				if (bEstimateColor) {
					const auto& baseColor = pImage[j];
					float r = baseColor.r * origConfidence + convergenceR;
					float g = baseColor.g * origConfidence + convergenceG;
					float b = baseColor.b * origConfidence + convergenceB;
					r *= nrm;
					g *= nrm;
					b *= nrm;
					pointcloud.AddColor(Pixel8U(_cvt_ftoi_fast(r), _cvt_ftoi_fast(g), _cvt_ftoi_fast(b)));
				}

				// invalidate all neighbor depths that do not agree with it
				for (unsigned hi = 0; hi < nInvalidHits; ++hi)
					*invalidHitsStorage[hi].pDepthB = 0;
			}
		}

		DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize() - 1, (pointcloud.NumPoints()) - nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(pConnection - connections.Begin());
	}

	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), nDepths, pointcloud.NumPoints(), ROUND2INT((100.f * (pointcloud.NumPoints())) / nDepths), TD_TIMER_GET_FMT().c_str());

#if 0
	if (bEstimateNormal && pointcloud.NumPoints() > 0 && !pointcloud.NormalStream()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		const int64_t nPoints((int64_t)pointcloud.NumPoints());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			const float* pWeights = pointcloud.WeightsStream(i);
			const size_t numWeights = pointcloud.WeightsStreamSize(i);
			ASSERT(numWeights > 0);
			IIndex idxView(0);
			float bestWeight = pWeights[0];
			for (IIndex idx=1; idx<numWeights; ++idx) {
				const float weight = pWeights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.ViewsStream(i)[idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			Point3f n;
			depthData.GetNormal(projs[i][idxView].GetCoord(), n);
			pointcloud.AddNormal(n);
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.NumPoints(), TD_TIMER_GET_FMT().c_str());
	}
#endif

	// release all depth-maps
	for (DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			depthData.DecRef();
} // FuseDepthMaps
/*----------------------------------------------------------------*/
#endif
#endif

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
	} else if (OPTDENSE::bDenseFuse) {
		// recursive dense fuse (median-based; more outlier-resistant)
		data.depthMaps.DenseFuseDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
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

			// Adjusting is required.  The filtered depth map is stored back to dmap and this is used to
			// filter and fuse later.
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
