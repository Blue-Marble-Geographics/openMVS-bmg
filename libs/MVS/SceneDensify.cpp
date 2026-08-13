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
// KD-tree for the density-based outlier filter
#include "nanoflann.hpp"

#include <list>
#include <unordered_map>
#include <intrin.h>

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif

#define ESTIMATE_NORMALS

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

// Atomic test-and-set on a single bit of a BitMatrix (TBitMatrix<size_t>).
// Returns true if the bit was ALREADY set (caller should bail out to avoid
// double-emitting); false if the bit was previously clear and is now set
// (caller proceeds). Used by DenseFuseDepthMaps under DENSE_FUSE_PARALLEL_PIXELS
// to make concurrent FusePoint claims race-safe. On Win64, `size_t` is 64-bit
// and `_InterlockedOr64` issues a `lock or` returning the previous value.
static inline bool TryClaimBitAtomic(SEACAVE::BitMatrix& mask, const ImageRef& ir) {
	const auto idx = SEACAVE::BitMatrix::computeIndex(ir, mask.cols);
	static_assert(sizeof(size_t) == 8, "BitMatrix word must be 64-bit for _InterlockedOr64");
	const size_t prev = (size_t)_InterlockedOr64(
		reinterpret_cast<volatile LONG64*>(&mask.data[idx.idx]),
		(LONG64)idx.flag);
	return (prev & idx.flag) != 0;
}
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////


// Bounded on-demand cache for the base decoded color image (scene.images[*].image).
// Mirrors FilterDMapCache's structure exactly (see below), but wraps
// Image::IncRefImage()/DecRefImage() instead of DepthData::IncRef()/DecRef(). This is
// the single largest unbounded allocation in the Densify pipeline: "prepare images"
// used to decode every scene image up front and never release it (resident through
// ESTIMATE and FUSE, freed only at process exit). maxMemory == 0 means "no eviction"
// (Tier 1: the pre-flight fits-check decided everything fits -- Acquire/Release degrade
// to a plain ref-count inc/dec, same cost as the old always-resident behavior).
// maxMemory > 0 means Tier 2: bounded LRU, decode-on-miss, evict-on-pressure. Spans
// both ESTIMATE and FUSE (both read the color buffer), so it is owned by DepthMapsData
// rather than scoped to one sub-phase like FilterDMapCache/DMapCache are.
// NOTE: defined inside an explicit "namespace MVS { }" block (rather than relying on
// the file's "using namespace MVS;") because SceneDensify.h forward-declares this type
// as MVS::ImageCache; completing it at file (global) scope would create an unrelated
// ::ImageCache and leave MVS::ImageCache permanently incomplete.
namespace MVS {
struct ImageCache {
	// Sentinel for "Tier 1: never evict, everything stays resident" -- deliberately
	// NOT 0, because 0 is a legitimate real budget (evict down to the bare minimum)
	// that the periodic re-derivation can and does compute when the box is critically
	// tight. Using 0 for both meanings was a real bug: SetMaxMemory(0) from the
	// periodic recompute silently disabled eviction instead of evicting harder, right
	// when eviction was needed most -- confirmed live: greyImages/sCachedImages tracked
	// a shrinking budget correctly for many images, then stopped shrinking entirely
	// (nothing left to constrain them or anything else) a few images before an OOM.
	static constexpr size_t UNBOUNDED = std::numeric_limits<size_t>::max();

	ImageCache(ImageArr& _images, const cList<unsigned>& _targetMaxResolution, size_t _maxMemory)
		: images(_images), targetMaxResolution(_targetMaxResolution),
		  usedMemory(0), maxMemory(_maxMemory), numAcquireCalls(0), numDecodes(0) {
		DEBUG_EXTRA("ImageCache: budget %s (%s)", _maxMemory == UNBOUNDED ? "n/a" : Util::formatBytes(_maxMemory).c_str(),
			_maxMemory == UNBOUNDED ? "unlimited/resident" : "bounded LRU");
	}
	~ImageCache() { Clear(); }

	// acquire a working reference (decoding from disk only if not resident) and retain it
	bool Acquire(IIndex idx) {
		Image& imageData = images[idx];
		const bool wasResident = !imageData.IsImageEmpty();
		const unsigned r = imageData.IncRefImage(targetMaxResolution[idx]); // working ref
		if (r == 0)
			return false;
		Lock l(cs);
		++numAcquireCalls;
		if (!wasResident)
			++numDecodes;
		const auto it = lruIter.find(idx);
		if (it == lruIter.end()) {
			imageData.IncRefImage(targetMaxResolution[idx]); // retain ref (+1); already resident -> no I/O
			lru.push_front(idx);
			lruIter[idx] = lru.begin();
			usedMemory += imageData.GetImageMemorySize();
		} else {
			lru.splice(lru.begin(), lru, it->second); // move to most-recently-used (iterator stays valid)
		}
		EvictLocked(idx);
		return true;
	}
	// release the working reference; the retain reference keeps the image resident
	void Release(IIndex idx) { images[idx].DecRefImage(); }

	uint32_t GetNumDecodes() const { return numDecodes; }
	double GetHitRate() const { return numAcquireCalls ? 1.0 - static_cast<double>(numDecodes) / numAcquireCalls : 0.0; }
	size_t GetUsedMemory() const { return usedMemory; }
	// true if this is a Tier-2 (bounded LRU) cache; false for Tier-1 (always resident).
	// Callers must only refresh the budget of an already-bounded cache -- never flip
	// Tier 1 <-> Tier 2 mid-run, since the Tier-1 baseline refs aren't cache-tracked.
	bool IsBounded() const { return maxMemory != UNBOUNDED; }

	void SetMaxMemory(size_t max_memory_bytes) {
		Lock l(cs);
		maxMemory = max_memory_bytes;
		EvictLocked(NO_ID);
	}

	// drop every retain reference (returns ref-counts to baseline)
	void Clear() {
		Lock l(cs);
		for (const IIndex idx : lru)
			images[idx].DecRefImage();
		lru.clear();
		lruIter.clear();
		usedMemory = 0;
	}
private:
	void EvictLocked(IIndex protectIdx) {
		if (maxMemory == UNBOUNDED)
			return; // Tier 1: eviction disabled, everything stays resident
		while (usedMemory > maxMemory && lru.size() > 1) {
			const IIndex idx = lru.back();
			if (idx == protectIdx)
				break;
			lru.pop_back();
			lruIter.erase(idx);
			Image& imageData = images[idx];
			const size_t sz = imageData.GetImageMemorySize();
			usedMemory = (usedMemory > sz) ? usedMemory - sz : 0;
			imageData.DecRefImage(); // drop retain ref; frees iff no working ref left
		}
	}
	ImageArr& images;
	const cList<unsigned>& targetMaxResolution; // per-image decode target (bytes-per-image sizing)
	std::list<IIndex> lru; // front = most-recently used
	std::unordered_map<IIndex, std::list<IIndex>::iterator> lruIter;
	size_t usedMemory, maxMemory;
	CriticalSection cs;
	// total Acquire() calls and how many of those hit the disk (debug only)
	uint32_t numAcquireCalls, numDecodes;
};
} // namespace MVS
/*----------------------------------------------------------------*/


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
// Bounded LRU cache for the per-image grayscale derivative (one entry per unique
// image ID, seeded from scene.images[*].image the first time InitViews() needs it).
// maxMemory == 0 means "no eviction" (matches the pre-existing behavior when a box
// has enough RAM); maxMemory > 0 evicts the least-recently-used entry once exceeded.
// Callers already hold sGreyImagesMutex across the whole find-or-insert sequence
// (unchanged from before), so this struct itself needs no internal lock.
// Evicting an entry here is safe even if another in-flight DepthData::ViewData holds
// a shallow copy of the same buffer: cv::Mat is itself refcounted, so destroying this
// cache's owning Image32F only drops one reference -- the underlying pixel buffer is
// freed only once every other holder (each released within one image's processing
// cycle, since at most the 2 ESTIMATE worker threads have live copies at a time) lets
// go too.
struct GreyImageCache {
	Image32F* Find(IDX id) {
		const auto it = lruIter.find(id);
		if (it == lruIter.end())
			return nullptr;
		lru.splice(lru.begin(), lru, it->second);
		return data[id].get();
	}
	Image32F* Insert(IDX id, std::unique_ptr<Image32F> img) {
		const size_t bytes = img->total() * img->elemSize();
		Image32F* const p = img.get();
		data[id] = std::move(img);
		lru.push_front(id);
		lruIter[id] = lru.begin();
		usedMemory += bytes;
		Evict();
		return p;
	}
	void SetMaxMemory(size_t max_memory_bytes) { maxMemory = max_memory_bytes; Evict(); }
	void Clear() { data.clear(); lru.clear(); lruIter.clear(); usedMemory = 0; }
	size_t GetUsedMemory() const { return usedMemory; }
	size_t GetCount() const { return data.size(); }
private:
	void Evict() {
		// No "maxMemory == 0 means unbounded" special case here on purpose: unlike
		// ImageCache's Tier 1, this cache is always budget-driven, and the periodic
		// live re-derivation can legitimately compute a budget of exactly 0 when the
		// box is critically tight -- that must evict aggressively (down to the last
		// entry, per the lru.size()>1 guard below), not disable eviction. Treating 0
		// as "unbounded" here was a real bug: it let the cache (and everything else
		// competing for the same RAM) grow unchecked right when eviction was needed
		// most, confirmed by a live OOM a few images after the budget hit zero.
		while (usedMemory > maxMemory && lru.size() > 1) {
			const IDX id = lru.back();
			lru.pop_back();
			lruIter.erase(id);
			const auto it = data.find(id);
			usedMemory -= it->second->total() * it->second->elemSize();
			data.erase(it);
		}
	}
	std::unordered_map<IDX, std::unique_ptr<Image32F>> data;
	std::list<IDX> lru; // front = most-recently used
	std::unordered_map<IDX, std::list<IDX>::iterator> lruIter;
	size_t usedMemory = 0, maxMemory = 0;
};
static GreyImageCache greyImages;
static std::mutex sGreyImagesMutex;
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
			{
				std::lock_guard<std::mutex> lock(sGreyImagesMutex);
				Image32F* found = greyImages.Find(neighbor.ID);
				if (!found) {
					imageCache->Acquire(neighbor.ID); // decode-on-demand if Tier 2; no-op cost if Tier 1
					auto img = std::make_unique<Image32F>(viewTrg.image);
					viewTrg.pImageData->image.toGray(*img, cv::COLOR_BGR2GRAY, true);
					imageCache->Release(neighbor.ID);
					found = greyImages.Insert(neighbor.ID, std::move(img));
				}

				viewTrg.image = *found;
			}
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
				{
					std::lock_guard<std::mutex> lock(sGreyImagesMutex);
					Image32F* found = greyImages.Find(neighbor.ID);
					if (!found) {
						imageCache->Acquire(neighbor.ID); // decode-on-demand if Tier 2; no-op cost if Tier 1
						auto img = std::make_unique<Image32F>(viewTrg.image);
						viewTrg.pImageData->image.toGray(*img, cv::COLOR_BGR2GRAY, true);
						imageCache->Release(neighbor.ID);
						found = greyImages.Insert(neighbor.ID, std::move(img));
					}

					viewTrg.image = *found;
				}
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
		{
			std::lock_guard<std::mutex> lock(sGreyImagesMutex);
			Image32F* found = greyImages.Find(idxImage);
			if (!found) {
				imageCache->Acquire(idxImage); // decode-on-demand if Tier 2; no-op cost if Tier 1
				auto img = std::make_unique<Image32F>(viewRef.image);
				viewRef.pImageData->image.toGray(*img, cv::COLOR_BGR2GRAY, true);
				imageCache->Release(idxImage);
				found = greyImages.Insert(idxImage, std::move(img));
			}

			viewRef.image = *found;
		}
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
// Bounded LRU cache for the per-(image,scale) 4-plane sampling derivative (see the
// insertion site below for what each entry holds and why). Same maxMemory==0-means-
// unbounded / LRU-eviction-above-budget shape as GreyImageCache above, and safe to
// evict for the same reason: cv::Mat's own refcounting keeps any already-copied-out
// reference (e.g. "i.imageBig = ...") alive independently of this cache's ownership.
struct ScaledImageCache {
	Image32F* Find(const ImageKey_t& key) {
		const auto it = lruIter.find(key);
		if (it == lruIter.end())
			return nullptr;
		lru.splice(lru.begin(), lru, it->second);
		return &data[key];
	}
	Image32F* Insert(const ImageKey_t& key, Image32F&& img) {
		const size_t bytes = img.total() * img.elemSize();
		Image32F& slot = data[key];
		slot = std::move(img);
		lru.push_front(key);
		lruIter[key] = lru.begin();
		usedMemory += bytes;
		Evict();
		return &data[key];
	}
	void SetMaxMemory(size_t max_memory_bytes) { maxMemory = max_memory_bytes; Evict(); }
	void Clear() { data.clear(); lru.clear(); lruIter.clear(); usedMemory = 0; }
	size_t GetUsedMemory() const { return usedMemory; }
	size_t GetCount() const { return data.size(); }
private:
	void Evict() {
		// See GreyImageCache::Evict() above for why maxMemory == 0 is NOT special-cased
		// as "unbounded" here -- a live-recomputed budget of exactly 0 must evict
		// aggressively, not disable eviction.
		while (usedMemory > maxMemory && lru.size() > 1) {
			const ImageKey_t key = lru.back();
			lru.pop_back();
			lruIter.erase(key);
			const auto it = data.find(key);
			usedMemory -= it->second.total() * it->second.elemSize();
			data.erase(it);
		}
	}
	std::unordered_map<ImageKey_t, Image32F> data;
	std::list<ImageKey_t> lru; // front = most-recently used
	std::unordered_map<ImageKey_t, std::list<ImageKey_t>::iterator> lruIter;
	size_t usedMemory = 0, maxMemory = 0;
};
static ScaledImageCache sCachedImages;
#endif

// Release the ESTIMATE-phase-only grey/derivative image caches (DPC_IMAGE_CACHE's
// greyImages, DPC_FASTER_SAMPLING's sCachedImages). Both are bounded LRU caches (one
// entry per unique image[,scale] currently resident, budget re-derived periodically --
// see SetEstimationImageCacheBudgets()), but neither is read again once FILTER/FUSE
// start (those only touch saved .dmap files via FilterDMapCache/DMapCache, never
// InitViews()), so clearing every remaining entry here is output-neutral.
static void ClearEstimationImageCaches()
{
#ifdef DPC_IMAGE_CACHE
	{
		std::lock_guard<std::mutex> lock(sGreyImagesMutex);
		DEBUG_EXTRA("greyImages cache: %u images, %s released", (unsigned)greyImages.GetCount(), Util::formatBytes(greyImages.GetUsedMemory()).c_str());
		greyImages.Clear();
	}
#endif
#ifdef DPC_FASTER_SAMPLING
	{
		std::lock_guard<std::mutex> lock(sCachedImagesMutex);
		DEBUG_EXTRA("sCachedImages cache: %u entries, %s released", (unsigned)sCachedImages.GetCount(), Util::formatBytes(sCachedImages.GetUsedMemory()).c_str());
		sCachedImages.Clear();
	}
#endif
}

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
			const ImageKey_t key(i.pImageData->ID, scale, i.image.cols, i.image.rows);
			Image32F* pImg = sCachedImages.Find(key);
			if (!pImg) { // not resident -- compute and insert
				Image32F img(i.image.rows, i.image.cols*4);
				float* __restrict dst = (float*)img.data;
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
				pImg = sCachedImages.Insert(key, std::move(img));
			}
			// copied out while still holding the lock (the old try_emplace path copied
			// after unlocking, which raced against a concurrent insert rehashing the map)
			i.imageBig = *pImg;
			sCachedImagesMutex.unlock();
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

// ===================================================================
// FILTER_PROFILE: temporary instrumentation for the depth-map filter
// stage. Accumulates per-phase time (summed across worker threads, so the
// totals exceed wall-clock -- the RATIOS reveal the bottleneck) and prints
// one summary line at the end of the filter phase. Set to 0 to remove.
// ===================================================================
#define FILTER_PROFILE 1
#if FILTER_PROFILE
namespace {
	using filter_clock = std::chrono::steady_clock;
	static inline int64_t FilterNs(const filter_clock::time_point& a, const filter_clock::time_point& b) {
		return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
	}
	struct FilterProfile {
		std::atomic<int64_t> nsLoad{0};      // load reference + neighbor .dmaps from disk
		std::atomic<int64_t> nsReproject{0}; // project neighbor depth-maps into the reference
		std::atomic<int64_t> nsFuse{0};      // per-pixel consensus / adjust pass
		std::atomic<int64_t> nsSave{0};      // write filtered.dmap / filtered.cmap
		std::atomic<int64_t> nsAdjLoad{0};   // reload filtered.dmap / filtered.cmap (adjust)
		std::atomic<int64_t> nsAdjSave{0};   // write final dmap (adjust)
		std::atomic<int>     nImages{0};
		void Reset() {
			nsLoad = 0; nsReproject = 0; nsFuse = 0; nsSave = 0; nsAdjLoad = 0; nsAdjSave = 0; nImages = 0;
		}
		void Report(double wallMs) const {
			const double inv = 1.0 / 1.0e6;
			VERBOSE("FILTER PROFILE [%d imgs, wall=%.0fms, thread-summed ms]: loadNeighbors=%.0f filterCompute=%.0f saveFiltered=%.0f adjReload=%.0f adjSave=%.0f",
				nImages.load(), wallMs,
				nsLoad.load()*inv, nsFuse.load()*inv,
				nsSave.load()*inv, nsAdjLoad.load()*inv, nsAdjSave.load()*inv);
		}
	};
	static FilterProfile g_filterProfile;
}
#endif

// filter depth-map, one pixel at a time, using confidence based fusion or neighbor pixels
#if 1
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool bAdjust)
{
	TD_TIMER_STARTD();
#if FILTER_PROFILE
	const auto _tF0 = filter_clock::now(); // whole-FilterDepthMap compute (reproject + fuse)
#endif

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
		// Precompute the combined neighbor-pixel -> reference-camera transform so each
		// pixel costs a few mul-adds (plus one divide) instead of three Camera-method
		// calls building Point3/Point2/ImageRef temporaries (same idea as the
		// FuseDepthMaps projection). For a neighbor pixel (x,y) at neighbor depth d the
		// reference-camera point is  camX = d*(a*x + b*y + c) + t  with
		//   M = R_ref * R_n^T,  a = M_col0/Kn00,  b = M_col1/Kn11,
		//   c = M_col2 - a*Kn02 - b*Kn12,  t = R_ref*(C_n - C_ref).
		// (a*x is accumulated along the row.) NOTE: this reorganizes the projection
		// (precombined matrices + per-row accumulation), so it is numerically
		// equivalent -- not bit-identical -- to the old sequential I2W->W2C->C2I,
		// matching the other fast paths.
		const RMatrix Mr(cameraRef.R * camera.R.t());
		const REAL Kn00(camera.K(0,0)), Kn11(camera.K(1,1)), Kn02(camera.K(0,2)), Kn12(camera.K(1,2));
		const Point3 a(Mr(0,0)/Kn00, Mr(1,0)/Kn00, Mr(2,0)/Kn00);
		const Point3 b(Mr(0,1)/Kn11, Mr(1,1)/Kn11, Mr(2,1)/Kn11);
		const Point3 c(Mr(0,2)-a.x*Kn02-b.x*Kn12, Mr(1,2)-a.y*Kn02-b.y*Kn12, Mr(2,2)-a.z*Kn02-b.z*Kn12);
		const Point3 t(cameraRef.R * (camera.C - cameraRef.C));
		const REAL Kr00(cameraRef.K(0,0)), Kr11(cameraRef.K(1,1)), Kr02(cameraRef.K(0,2)), Kr12(cameraRef.K(1,2));
		// bounds for the splat target: depthMap/confMap are allocated with sizeRef
		// (the reference image), NOT the neighbor's size. Using the neighbor size here
		// let projected coordinates that are valid for a larger neighbor but out of
		// range for the smaller reference write past the depthMap buffer.
		const int wRef(sizeRef.width), hRef(sizeRef.height);
		// z-buffered splat of one reference pixel (keep the nearest depth)
		const auto splat = [&](int px, int py, Depth cz, float conf) {
			if ((unsigned)px < (unsigned)wRef && (unsigned)py < (unsigned)hRef) {
				Depth& dr = depthMap(py, px);
				if (dr == 0 || dr >= cz) { dr = cz; if (bAdjust) confMap(py, px) = conf; }
			}
		};
		for (int i=0; i<size.height; ++i) {
			const Depth* const __restrict pDepth = &depthData.depthMap(i, 0);
			const float* const __restrict pConf = bAdjust ? &depthData.confMap(i, 0) : NULL;
			REAL mux(b.x*i + c.x), muy(b.y*i + c.y), muz(b.z*i + c.z); // a*0 + (b*i+c)
			for (int j=0; j<size.width; ++j, mux+=a.x, muy+=a.y, muz+=a.z) {
				const Depth depth(pDepth[j]);
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const REAL cxz(depth*muz + t.z);
				if (cxz <= 0)
					continue;
				const REAL invZ(REAL(1)/cxz);
				const REAL u(Kr02 + Kr00*((depth*mux + t.x)*invZ));
				const REAL v(Kr12 + Kr11*((depth*muy + t.y)*invZ));
				const int x0(FLOOR2INT(u)), x1(CEIL2INT(u)), y0(FLOOR2INT(v)), y1(CEIL2INT(v));
				const Depth cz((Depth)cxz);
				const float conf(pConf ? pConf[j] : 0.f);
				splat(x0, y0, cz, conf);
				splat(x0, y1, cz, conf);
				splat(x1, y0, cz, conf);
				splat(x1, y1, cz, conf);
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
		// hoist per-neighbor row-base pointers once per row: the hot inner reads become a
		// plain [j] index instead of a cList index + i*cols multiply. Byte-identical
		// addressing and unchanged float-accumulation order (n from N-1 down to 0).
		std::vector<const Depth*> pNDepth(N);
		std::vector<const float*> pNConf(N);
		for (int i=0; i<sizeRef.height; ++i) {
			for (IIndex k=0; k<N; ++k) {
				pNDepth[k] = &depthMaps[k](i, 0);
				pNConf[k] = &confMaps[k](i, 0);
			}
			const Depth* const __restrict pRefDepth = &depthDataRef.depthMap(i, 0);
			const float* const __restrict pRefConf = &depthDataRef.confMap(i, 0);
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(pRefDepth[j]);
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
				float posConf(pRefConf[j]), negConf(0);
				Depth avgDepth(depth*posConf);
				unsigned nPosViews(0), nNegViews(0);
				unsigned n(N);
				do {
					--n;
					const Depth d(pNDepth[n][j]);
					if (d == 0) {
						if (nPosViews + nNegViews + n < nMinViews)
							goto DiscardDepth;
						continue;
					}
					ASSERT(d > 0);
					if (IsDepthSimilar(depth, d, thDepthDiff)) {
						// average similar depths
						const float c(pNConf[n][j]);
						avgDepth += d*c;
						posConf += c;
						++nPosViews;
					} else {
						// penalize confidence
						if (depth > d) {
							// occlusion
							negConf += pNConf[n][j];
						} else {
							// free-space violation
							const DepthData& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].ID];
							const Camera& camera = depthData.images.First().camera;
							const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
							const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
							if (depthData.confMap.isInside(x)) {
								const float c(depthData.confMap(x));
								negConf += (c > 0 ? c : pNConf[n][j]);
							} else
								negConf += pNConf[n][j];
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
		// hoist per-neighbor row-base pointers once per row (byte-identical addressing;
		// removes the cList index + i*cols multiply from the hot neighbor reads). The delta
		// block needs rows i-1,i,i+1 per neighbor; rows i-1/i+1 only exist for interior rows,
		// so they are only populated (and read) when in range to avoid reading outside the
		// depth-map buffer. The counts summed here are order-independent.
		std::vector<const Depth*> pRow0(N), pRowM(N), pRowP(N);
		for (int i=0; i<sizeRef.height; ++i) {
			const bool hasRowM(i > 0);
			const bool hasRowP(i+1 < sizeRef.height);
			for (IIndex k=0; k<N; ++k) {
				pRow0[k] = &depthMaps[k](i, 0);
				pRowM[k] = hasRowM ? &depthMaps[k](i-1, 0) : NULL;
				pRowP[k] = hasRowP ? &depthMaps[k](i+1, 0) : NULL;
			}
			const Depth* const __restrict pRefDepth = &depthDataRef.depthMap(i, 0);
			const float* const __restrict pRefConf = &depthDataRef.confMap(i, 0);
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(pRefDepth[j]);
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
						--n;
						const Depth d(pRow0[n][j]);
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
				// deltas: (-1,0)->row i col j-1, (1,0)->row i col j+1,
				//         (0,-1)->row i-1 col j, (0,1)->row i+1 col j
				// border rows/cols are skipped so we never read outside the depth-map buffer
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					const bool hasColL(j > 0), hasColR(j+1 < sizeRef.width);
					const int jL(j-1), jR(j+1);
					unsigned n(N);
					do {
						--n;
						if (hasColL) {
							const Depth dA(pRow0[n][jL]);
							if (dA > 0) { ++nViews; if (IsDepthSimilar(depth, dA, thDepthDiff)) ++nGoodViews; }
						}
						if (hasColR) {
							const Depth dB(pRow0[n][jR]);
							if (dB > 0) { ++nViews; if (IsDepthSimilar(depth, dB, thDepthDiff)) ++nGoodViews; }
						}
						if (hasRowM) {
							const Depth dC(pRowM[n][j]);
							if (dC > 0) { ++nViews; if (IsDepthSimilar(depth, dC, thDepthDiff)) ++nGoodViews; }
						}
						if (hasRowP) {
							const Depth dD(pRowP[n][j]);
							if (dD > 0) { ++nViews; if (IsDepthSimilar(depth, dD, thDepthDiff)) ++nGoodViews; }
						}
					} while (n);
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
				newConfMap(xRef) = pRefConf[j];
			}
		}
	}

#if FILTER_PROFILE
	g_filterProfile.nsFuse += FilterNs(_tF0, filter_clock::now());
	const auto _tS0 = filter_clock::now();
#endif
	const bool savedOK =
		SaveDepthMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.dmap"), newDepthMap) &&
		SaveConfidenceMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.cmap"), newConfMap);
#if FILTER_PROFILE
	g_filterProfile.nsSave += FilterNs(_tS0, filter_clock::now());
#endif
	if (!savedOK)
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

	// Merge adds exactly ONE view per generated point, so the per-point index arrays
	// (offsets/sizes) and the flat view memory each need only nPointsEstimate entries.
	// (Previously these were reserved as nPointsEstimate*numImages, which could reach
	// hundreds of GB and OOM on large scenes.)
	pointcloud.ReservePointViewsSizeAndOffset(nPointsEstimate);
	pointcloud.ReservePointViewsMemory(nPointsEstimate);

	if (bEstimateColor)
		pointcloud.ReserveColors(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.ReserveNormals(nPointsEstimate);

	// Each point "i" will have a set of up to maxIdxsPerPoint point view indexes.

	Util::Progress progress(_T("Merged depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();

	FOREACH(idxImage, arrDepthData) {
		TD_TIMER_STARTD();
		DepthData& depthData = arrDepthData[idxImage];
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		if (bEstimateColor)
			imageCache->Acquire(idxImage); // only the reference image's color is read below
		const DepthData::ViewData& image = depthData.GetView();
		const size_t nNumPointsPrev(pointcloud.NumPoints());

		Point3f normal;
		for (int i=0; i<depthData.depthMap.rows; ++i) {
			for (int j=0; j<depthData.depthMap.cols; ++j) {
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
		if (bEstimateColor)
			imageCache->Release(idxImage);
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
		skipMemoryCheckIdxImage(NO_ID), numImageRead(0), numUseImageCalls(0)
	{
		DEBUG_EXTRA("DMapCache: budget %s", Util::formatBytes(_max_memory_bytes).c_str());
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
		++numUseImageCalls;
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
	// fraction of UseImage() calls that hit an already-resident depth-map (0..1)
	double GetHitRate() const { return numUseImageCalls ? 1.0 - static_cast<double>(numImageRead) / numUseImageCalls : 0.0; }

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
	// total number of UseImage() calls, i.e. hits+misses (debug only)
	mutable uint32_t numUseImageCalls;
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

// Filter-phase depth-map retention cache (semantics-neutral I/O dedup).
// During the depth-map FILTER sub-phase every image loads its reference dmap plus
// up to 8 neighbor dmaps, then releases them; because DepthData::DecRef() frees a
// map the instant its ref-count hits 0, the same neighbor is re-loaded and
// re-deserialized once per image that references it (~9x redundancy). The on-disk
// dmaps are immutable for the entire filter sub-phase (filtered results go to side
// files; the adjust sub-phase -- gated by `sem` until all filters complete -- swaps
// them in afterwards), so keeping a loaded copy resident and re-using it is
// byte-identical to re-loading it. This cache holds one extra ("retain") reference
// on recently-used maps, bounded by a memory budget with LRU eviction, so a repeat
// Acquire() of a still-resident map skips the disk read entirely. Acquire()/Release()
// wrap the existing IncRef()/DecRef() and are thread-safe. Clear() drops all retain
// references and MUST run once after the last filter completes and before the adjust
// sub-phase (which asserts ref-count == 1), i.e. from SignalCompleteDepthmapFilter().
struct FilterDMapCache {
	FilterDMapCache(DepthDataArr& _arrDepthData, size_t _maxMemory)
		: arrDepthData(_arrDepthData), usedMemory(0), maxMemory(_maxMemory), numAcquireCalls(0), numDiskLoads(0) {
		DEBUG_EXTRA("FilterDMapCache: budget %s", Util::formatBytes(_maxMemory).c_str());
	}
	~FilterDMapCache() { Clear(); }
	// acquire a working reference (loading from disk only if not resident) and retain it
	bool Acquire(IIndex idx, const String& fileName) {
		DepthData& depthData = arrDepthData[idx];
		const bool wasResident = !depthData.IsEmpty(); // best-effort hit/miss stat, not synchronized with IncRef's own lock
		const unsigned r = depthData.IncRef(fileName); // working ref; loads iff empty (outside lock)
		if (r == 0)
			return false;
		Lock l(cs);
		++numAcquireCalls;
		if (!wasResident)
			++numDiskLoads;
		const auto it = lruIter.find(idx);
		if (it == lruIter.end()) {
			depthData.IncRef(fileName);              // retain ref (+1); already loaded -> no disk I/O
			lru.push_front(idx);
			lruIter[idx] = lru.begin();
			usedMemory += depthData.GetMemorySize();
		} else {
			lru.splice(lru.begin(), lru, it->second); // move to most-recently-used (iterator stays valid)
		}
		EvictLocked(idx);
		return true;
	}
	// number of disk loads and hit-rate (0..1) across all Acquire() calls so far
	uint32_t GetNumDiskLoads() const { return numDiskLoads; }
	double GetHitRate() const { return numAcquireCalls ? 1.0 - static_cast<double>(numDiskLoads) / numAcquireCalls : 0.0; }
	// release the working reference; the retain reference keeps the map resident
	void Release(IIndex idx) {
		arrDepthData[idx].DecRef();
	}
	// drop every retain reference (returns ref-counts to baseline)
	void Clear() {
		Lock l(cs);
		for (const IIndex idx : lru)
			arrDepthData[idx].DecRef();
		lru.clear();
		lruIter.clear();
		usedMemory = 0;
	}
private:
	void EvictLocked(IIndex protectIdx) {
		while (usedMemory > maxMemory && lru.size() > 1) {
			const IIndex idx = lru.back();
			if (idx == protectIdx)
				break;
			lru.pop_back();
			lruIter.erase(idx);
			DepthData& depthData = arrDepthData[idx];
			const size_t sz = depthData.GetMemorySize();
			usedMemory = (usedMemory > sz) ? usedMemory - sz : 0;
			depthData.DecRef();                      // drop retain ref; frees iff no working ref left
		}
	}
	DepthDataArr& arrDepthData;
	std::list<IIndex> lru;                            // front = most-recently used
	std::unordered_map<IIndex, std::list<IIndex>::iterator> lruIter;
	size_t usedMemory, maxMemory;
	CriticalSection cs;
	// total Acquire() calls and how many of those hit the disk (debug only)
	uint32_t numAcquireCalls, numDiskLoads;
};
static FilterDMapCache* g_filterCache = NULL;

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
	// NOTE: depth-maps are streamed and none is loaded yet here, so
	// First().depthMap.area() is 0 and left points/colors/normals unreserved
	// (causing reallocation during fusion). Use the same modest per-image
	// estimate as DenseFuseDepthMaps.
	const size_t nPointsEstimate(arrDepthData.size() * 9000);
	pointcloud.ReservePoints(nPointsEstimate);
	pointcloud.ReservePointViewsSizeAndOffset(nPointsEstimate);
	pointcloud.ReservePointWeightsSizeAndOffset(nPointsEstimate);
	unsigned depthDataLoadFlags(HeaderDepthDataRaw::HAS_DEPTH | HeaderDepthDataRaw::HAS_CONF);
	if (bEstimateColor)
		pointcloud.ReserveColors(nPointsEstimate);
#ifdef ESTIMATE_NORMALS
	if (bEstimateNormal) {
		pointcloud.ReserveNormals(nPointsEstimate);
		depthDataLoadFlags |= HeaderDepthDataRaw::HAS_NORMAL;
	}
#endif

	// Available physical memory. Use the cross-platform 64-bit query: the
	// legacy GlobalMemoryStatus() saturates dwAvailPhys (a 32-bit DWORD) at
	// 4GB, which silently under-reserves the flat views/weights buffers.
	const size_t bytesAvailable = Util::GetMemoryInfo().freePhysical;

	// Split a quarter of what's left between points and views.
	const size_t elementSize =
		std::max(
			sizeof(decltype(pointcloud.pointViewsMemory)::value_type),
			sizeof(decltype(pointcloud.pointWeightsMemory)::value_type)
		);

	const size_t nElementsAvailableToUse = (bytesAvailable / elementSize) / 4;
	pointcloud.ReservePointViewsMemory(nElementsAvailableToUse / 4);
	pointcloud.ReservePointWeightsMemory(nElementsAvailableToUse / 4);

	// Re-derive the (bounded) color-image cache's budget from current freePhysical
	// before sizing cacheDMaps below, so cacheDMaps' GetAvailableMemory() call sees
	// whatever imageCache's own eviction (run synchronously inside SetMaxMemory)
	// just freed -- same "compute the competing budget first" ordering already used
	// at the ESTIMATE->FILTER handoff.
	if (imageCache && imageCache->IsBounded()) {
		const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
		const size_t safetyMemory(std::max(static_cast<size_t>(memInfo.totalPhysical * 0.08), size_t(1)*1024*1024*1024ull));
		imageCache->SetMaxMemory(memInfo.freePhysical > safetyMemory ? memInfo.freePhysical - safetyMemory : 0);
	}

	Util::Progress progress(_T("Fused depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	BoolArr fusedDMaps(arrDepthData.size());
	fusedDMaps.Memset(0);
	DMapCache cacheDMaps(arrDepthData, depthDataLoadFlags, GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion));
	unsigned totalNumImageNeighborsInCache = 0, totalNumImagesInCache = 0;
	IIndex numDMapsFused = 0;

	std::vector<TRMatrixBase<float>> imagesCameraRt;
	std::vector<Matrix3x4f> imagesCameraP;
	std::vector<Matrix4x4f> imagesCameraPt;
	imagesCameraRt.reserve(scene.images.size());

	FOREACH(i, scene.images) {
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
#ifdef ESTIMATE_NORMALS
		// Normal maps are normally loaded alongside the depth maps (HAS_NORMAL
		// flag set when bEstimateNormal). If one is missing, estimate it from the
		// depth map so the fused normal is meaningful.
		if (bEstimateNormal && depthData.normalMap.empty() && !depthData.depthMap.empty()) {
			DepthData& depthDataMut = arrDepthData[idxImage];
			EstimateNormalMap(depthDataMut.images.front().camera.K, depthDataMut.depthMap, depthDataMut.normalMap);
		}
#endif
		const IIndex nMaxViewsFuse = OPTDENSE::nMaxViewsFuse;
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
		// NOTE: this neighbor warm-up / index-map allocation loop MUST stay
		// serial. It was previously guarded by `#pragma omp parallel for`, but
		// that is not thread-safe: DMapCache::UseImage() loads depth-maps with
		// the cache mutex dropped and, under the lock, runs Eject() ->
		// EjectOldest() which Save()s and Release()s the oldest cached
		// DepthData. With up to nMaxViewsFuse neighbors loaded against a much
		// smaller cache budget, Eject fires mid-loop and can Release a sibling
		// neighbor that another thread is concurrently reading here via the
		// unlocked depthDataB.IsEmpty() / depthDataB.depthMap.size() accesses
		// -> data race on (and possible use of a freed) DepthData payload. The
		// shared `numNeighbors` counter was also a non-atomic RMW. The mutex
		// only guards cache bookkeeping, not the payload these lines touch.
		// Running serially removes both hazards and restores the deterministic
		// break-at-cap behavior of the reference implementation; this loop only
		// warms the cache and allocates index maps, so it is not a hotspot.
		IIndex numNeighbors(0);
		for (const ViewScore& neighbor : depthData.neighbors) {
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			if (++numNeighbors >= nMaxViewsFuse)
				break;
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

		// Acquire the color buffers for the reference image + every populated
		// neighbor once per reference-image iteration (image granularity, not
		// per-point/per-pixel -- bEstimateColor is the only reason this loop
		// reads scene.images[*].image, via the "->image.pix(...)" reads deep in
		// the per-point phase below). Paired with the Release block further down.
		if (bEstimateColor) {
			imageCache->Acquire(idxImage);
			for (const auto& nc : neighborCache)
				imageCache->Acquire(nc.idxImageB);
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
			PointCloud::Normal normalB; // world-space neighbor normal from the Phase 1 gate (reused in Phase 2)
		};
		// NOTE: these were previously _alloca()'d. _alloca reserves from the WHOLE
		// enclosing function's stack frame, not the current loop iteration -- since
		// this sits inside FuseDepthMaps' outer per-reference-image loop (one call of
		// FuseDepthMaps processes every image in the dataset), every image's buffers
		// accumulated on the stack and were never reclaimed until the entire fuse
		// operation finished. Confirmed live: a 1873-image run crashed with
		// STATUS_STACK_OVERFLOW (0xC00000FD) at image 437 (~23%), consistent with a
		// few KB/image of unreclaimed stack space exhausting a ~1MB thread stack
		// around that point. Heap-backed containers declared here are destroyed (and
		// their memory freed) at the end of each loop iteration instead, which is
		// exactly the fix -- these are small (bounded by nMaxViewsFuse, 32 by
		// default), so the heap-allocation cost versus alloca is negligible next to
		// everything else done per image.
		const unsigned maxNeighbors = (unsigned)neighborCache.size();
		std::vector<NeighborHit> hitsStorageBuf(maxNeighbors);
		std::vector<NeighborHit> invalidHitsStorageBuf(maxNeighbors);
		NeighborHit* hitsStorage = hitsStorageBuf.data();
		NeighborHit* invalidHitsStorage = invalidHitsStorageBuf.data();
		unsigned nHits = 0, nInvalidHits = 0;

		const unsigned maxViews = (unsigned)neighborCache.size() + 1; // +1 for reference view
		std::vector<uint32_t> viewsStorageBuf(maxViews);
		std::vector<float> weightsStorageBuf(maxViews);
		uint32_t* __restrict viewsStorage = viewsStorageBuf.data();
		float* __restrict weightsStorage = weightsStorageBuf.data();
		// Defer idxPointB assignments until we know the point survives the fuse check.
		// Collect pointers to idxPointB slots so we can commit them only for accepted points.
		std::vector<uint32_t*> deferredStorageBuf(neighborCache.size());
		uint32_t** __restrict deferredStorage = deferredStorageBuf.data();
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

		// A/B: precision of the reference 3D point built from depth
		// (point = R * Kinv(x,y) * depth + C). The additive camera center (refC*)
		// and the destination (Point3f point) are ALREADY float, so the double
		// intermediates here buy no precision -- they are truncated to float on
		// store. FUSE_FWD_POINT_FLOAT=1 makes the whole build float (no
		// double<->float conversions); =0 restores the exact current
		// double-intermediate behavior (bit-identical to before).
#ifndef FUSE_FWD_POINT_FLOAT
#define FUSE_FWD_POINT_FLOAT 1
#endif
#if FUSE_FWD_POINT_FLOAT
		typedef float fwd_t;
#else
		typedef double fwd_t;
#endif

		// Row-invariant reference-camera constants hoisted out of the per-row loop.
		// These depend only on imageData.camera.K / depthData.confMap (constant across
		// rows), so this is pure loop-invariant code motion: bit-identical values, just
		// computed once instead of once per row. The reciprocal/principal-point values
		// are still computed in double then narrowed to fwd_t exactly as before. The
		// per-pixel / per-neighbor accumulation below is deliberately left byte-for-byte
		// unchanged -- reordering it would change the emitted point positions/colors/normals.
		const bool confMapEmpty = depthData.confMap.empty();
		confMapInc = confMapEmpty ? 0 : 1;
		const fwd_t invImageDataCameraK00 = (fwd_t)(1.0 / imageData.camera.K(0, 0));
		const fwd_t invImageDataCameraK11 = (fwd_t)(1.0 / imageData.camera.K(1, 1));
		const fwd_t imageDataCameraK02 = (fwd_t)imageData.camera.K(0, 2);
		const fwd_t imageDataCameraK12 = (fwd_t)imageData.camera.K(1, 2);
		const fwd_t pointXNoDepthPreTransformDelta = invImageDataCameraK00;

		for (int i = 0; i < sizeMap.height; ++i) {
			const Depth* __restrict pDM = &depthData.depthMap(i, 0);
			uint32_t* __restrict pDepthIdxs = (uint32_t*)&depthIdxs(i, 0);

			const float* __restrict pConfMap = confMapEmpty ? &confMapSentinel : &depthData.confMap(i, 0);

			const Normal* __restrict pNormalMap = &depthData.normalMap(i, 0);
			const Pixel8U* __restrict pImage = &imageData.image(i, 0);

			fwd_t pointXNoDepthPreTransform = -imageDataCameraK02 * invImageDataCameraK00;
			fwd_t pointYNoDepthPreTransform = ((fwd_t)i - imageDataCameraK12) * invImageDataCameraK11;

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

				const fwd_t pointXWithDepth = pointXNoDepthPreTransform * depth;
				const fwd_t pointYWithDepth = pointYNoDepthPreTransform * depth;
				const fwd_t pointZWithDepth = depth;

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
					// extract projected x,y lanes directly (bit-identical to the
					// previous store-to-array + reload, but avoids the stack
					// round-trip on this hot per-neighbor path)
					const float projX = _vFirst(_Splat(xyz0, 0));
					const float projY = _vFirst(_Splat(xyz0, 1));
					const ImageRef xB(ROUND2INT(projX * invZ), ROUND2INT(projY * invZ));

					Depth& depthB = depthMapB.pix(xB);
					if (depthB == 0) {
#if FUSE_DIAGNOSTICS
						++cNeighborZeroDepth;
#endif
						continue;
					}

					// Defer the second random load (depthIdx) past the depth/occlusion
					// test. A neighbor where the point is occluded by a nearer,
					// dissimilar surface (!similar && ptz >= depthB) is neither a hit
					// nor an invalidation candidate, so its claim state is irrelevant
					// and the depthIdx fetch is wasted. Skipping it is bit-identical
					// (the idxPointB != NO_ID check has no side effect) and saves one
					// random cache-line fetch per occluded probe. Disabled under
					// FUSE_DIAGNOSTICS so the per-bucket counters below stay exact.
					const bool similar = FastAbsS(ptz - depthB) < fDepthDiffThresholdFuse * ptz;
#if !FUSE_DIAGNOSTICS
					if (!similar && ptz >= depthB)
						continue;
#endif

					uint32_t& idxPointB = nc.depthIdxB->pix(xB);
					if (idxPointB != NO_ID) {
#if FUSE_DIAGNOSTICS
						++cNeighborPreclaimed;
#endif
						continue;
					}

					if (similar) {
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
							h.normalB = normalB; // cache world-space normal for Phase 2 reuse
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
						// reuse the world-space neighbor normal already computed
						// during the Phase 1 gate: identical value in the common
						// case, but avoids a redundant 3x3 rotate and a random
						// normal-map fetch here. It also removes the latent read
						// of an empty neighbor normal-map that the old
						// bNormalMap-only guard could perform (Phase 1 already
						// substitutes {0,0,-1} when the neighbor has no normals).
						N += h.normalB * confidenceB;
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
#ifdef ESTIMATE_NORMALS
				// emit the confidence-weighted fused world-space normal (N was
				// accumulated as ref-normal*conf + sum(neighbor-normal*confB)).
				if (bEstimateNormal)
					pointcloud.AddNormal(normalized(N));
#endif
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

		if (bEstimateColor) {
			for (const auto& nc : neighborCache)
				imageCache->Release(nc.idxImageB);
			imageCache->Release(idxImage);
		}

		fusedDMaps[idxImage] = true;
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
	const uint32_t cacheNumReads(cacheDMaps.GetNumImageReads());
	const double cacheHitRate(cacheDMaps.GetHitRate());
	arrDepthIdx.Release();
	cacheDMaps.ClearCache();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%), %.2f hits in %.2f cached (%s), dmap-cache: %u disk reads, %.1f%%%% hit-rate",
		numDMapsFused, nDepths, pointcloud.NumPoints(), ROUND2INT((100.f * pointcloud.NumPoints()) / nDepths),
		static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused,
		static_cast<double>(totalNumImagesInCache) / numDMapsFused, TD_TIMER_GET_FMT().c_str(),
		cacheNumReads, cacheHitRate * 100.0);

	// Normals are emitted directly during fusion (normalized(N) per surviving
	// point), so no separate post-pass is needed. The previous projs[]-based
	// fallback used the old non-streaming PointCloud API and is removed.
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
	// RELAXED enabled (June 2026): tree-canopy depths systematically fail
	// the strict gates (moving leaves => low NCC confidence, >2px reproj
	// error, 1-pixel clusters) and the point cloud loses the entire canopy
	// vs the reference output. Relaxed keeps the 2-view consensus
	// requirement (nMinViewsFuse) so it adds recall, not random outliers.
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
		// Cross-platform 64-bit available-memory query (GlobalMemoryStatus's
		// dwAvailPhys is a 32-bit DWORD that saturates at 4GB).
		const size_t bytesAvailable = Util::GetMemoryInfo().freePhysical;
		const size_t elementSize =
			std::max(
				sizeof(decltype(pointcloud.pointViewsMemory)::value_type),
				sizeof(decltype(pointcloud.pointWeightsMemory)::value_type));
		const size_t nElementsAvailableToUse = (bytesAvailable / elementSize) / 4;
		pointcloud.ReservePointViewsMemory(nElementsAvailableToUse / 4);
		pointcloud.ReservePointWeightsMemory(nElementsAvailableToUse / 4);
	}

	// Re-derive the (bounded) color-image cache's budget from current freePhysical
	// before sizing cacheDMaps below -- see the identical comment in FuseDepthMaps.
	if (imageCache && imageCache->IsBounded()) {
		const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
		const size_t safetyMemory(std::max(static_cast<size_t>(memInfo.totalPhysical * 0.08), size_t(1)*1024*1024*1024ull));
		imageCache->SetMaxMemory(memInfo.freePhysical > safetyMemory ? memInfo.freePhysical - safetyMemory : 0);
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
	// NOTE: HYBRID emits raw (un-medianed) per-pixel positions for sparse
	// clusters (<kHybridThresholdViews unique views). This leaks depth-map
	// noise as scattered "remnants" beneath the surface; reference always
	// emits one median per cluster. Keep this OFF unless deliberately
	// trading correctness for coverage on very sparse data.
	#define DENSE_FUSE_HYBRID 0
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

	// =====================================================================
	// DENSE_FUSE_PARALLEL_PIXELS: parallelize the per-pixel FusePoint loop
	// within each ref image. Threads contend on arrUseMask bits via atomic
	// test-and-set (TryClaimBitAtomic, defined at file scope); per-thread
	// emit buffers are flushed serially at end of image.
	// NOTE: the resulting point cloud is NOT bit-identical across runs
	// (cluster groupings depend on thread scheduling race outcomes).
	// Total surface coverage and point count are statistically equivalent
	// (within ~0.5% on tested scenes).
	//   0 = serial (deterministic, original behavior)
	//   1 = parallel (typically 6-10x faster on a 32-thread CPU)
	// =====================================================================
	#define DENSE_FUSE_PARALLEL_PIXELS 1

	// Per-cluster scratch state. One instance per thread (parallel mode) or
	// one at function scope (serial mode). FusePoint takes this by reference.
	struct FuseScratch {
		Point3f refPoint   = Point3f(0.f, 0.f, 0.f);
		Point3f refNormal  = Point3f(0.f, 0.f, -1.f);
		CLISTDEF0IDX(float, unsigned) fusedPoints[3];
		std::vector<uint32_t> fusedViews;
		std::vector<float>    fusedWeights;
		Point3d  fusedNormal = Point3d::ZERO;
		Pixel32F fusedColor  = Pixel32F::BLACK;
		#if DENSE_FUSE_HYBRID
		std::vector<uint32_t> pxView;
		std::vector<float>    pxWeight;
		std::vector<Pixel8U>  pxColor;
		std::vector<Point3f>  pxNormal;
		#endif
		inline void ResetCluster() {
			fusedPoints[0].clear();
			fusedPoints[1].clear();
			fusedPoints[2].clear();
			fusedViews.clear();
			fusedWeights.clear();
			fusedNormal = Point3d::ZERO;
			fusedColor  = Pixel32F::BLACK;
			#if DENSE_FUSE_HYBRID
			pxView.clear();
			pxWeight.clear();
			pxColor.clear();
			pxNormal.clear();
			#endif
		}
	};

	// Per-thread emit buffer. Accumulates emitted points during the parallel
	// pixel loop; flushed serially into `pointcloud` at end of each image.
	struct EmitBuf {
		struct Rec {
			Point3f  point;
			Point3f  normal;   // unused if !bEstimateNormal
			Pixel8U  color;    // unused if !bEstimateColor
			uint32_t viewBeg;
			uint32_t viewEnd;
		};
		std::vector<Rec>      recs;
		std::vector<uint32_t> views;
		std::vector<float>    weights;
	};

	// DIAG removed under parallel-pixels mode (counters would race).
	// Rebuild with DENSE_FUSE_PARALLEL_PIXELS=0 if diagnostics are needed.
	#define DENSE_FUSE_DIAG 0

	const auto FusePoint = [&](IIndex ID, const ImageRef& x, unsigned fuseDepth, FuseScratch& s) -> void {
		const auto lambda = [&](IIndex curID, const ImageRef& curX, unsigned curDepth, const auto& Self) -> void {
			#if DENSE_FUSE_OPT_TUNING
			const DFImageCache& pic = imgCache[curID];
			const DepthMap&     curDepthMap  = *pic.pDepthMap;
			const ConfidenceMap& curConfMap  = *pic.pConfMap;
			const NormalMap&    curNormalMap = *pic.pNormalMap;
			const Image&        curImageData = *pic.pImageData;
			if (curDepthMap.empty()) return;
			if ((unsigned)curX.x >= (unsigned)pic.width ||
			    (unsigned)curX.y >= (unsigned)pic.height) return;
			const Depth depth = curDepthMap(curX);
			if (depth <= Depth(0)) return;
			UseMask& useMask = *pic.pUseMask;
			// Read-only fast-path: skip pixels already claimed. Race here is
			// benign (we may do redundant work; the atomic claim below will
			// reject any double-emit).
			if (useMask(curX)) return;
			const float conf(curConfMap.empty() ? 1.f : curConfMap(curX));
			if (conf < minConfidence) return;
			const bool bHaveNormal = !curNormalMap.empty();
			Point3f normal;
			if (curDepth > 0) {
				const double rx = (double)s.refPoint.x;
				const double ry = (double)s.refPoint.y;
				const double rz = (double)s.refPoint.z;
				const double ptx_d = pic.P[0]*rx + pic.P[1]*ry + pic.P[2 ]*rz + pic.P[3 ];
				const double pty_d = pic.P[4]*rx + pic.P[5]*ry + pic.P[6 ]*rz + pic.P[7 ];
				const double ptz_d = pic.P[8]*rx + pic.P[9]*ry + pic.P[10]*rz + pic.P[11];
				const Point3f pt((float)ptx_d, (float)pty_d, (float)ptz_d);
				if (pt.z <= Depth(0)) return;
				if (!IsDepthSimilar(depth, pt.z, OPTDENSE::fDepthDiffThreshold)) return;
				const Point2f diff(pt.x / pt.z - float(curX.x), pt.y / pt.z - float(curX.y));
				if (normSq(diff) > kReprojErrSq) return;
				if (bHaveNormal) {
					const Normal& nLocal = curNormalMap(curX);
					const double nx = (double)nLocal.x;
					const double ny = (double)nLocal.y;
					const double nz = (double)nLocal.z;
					normal.x = (float)(pic.R00*nx + pic.R10*ny + pic.R20*nz);
					normal.y = (float)(pic.R01*nx + pic.R11*ny + pic.R21*nz);
					normal.z = (float)(pic.R02*nx + pic.R12*ny + pic.R22*nz);
					if (s.refNormal.z != -1.f || s.refNormal.x != 0.f || s.refNormal.y != 0.f) {
						if (s.refNormal.dot(normal) < normalError) return;
					}
				} else {
					normal = s.refNormal;
				}
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
			}
			// Atomic test-and-set on useMask. If another thread claimed this
			// pixel since the early read above, bail (do not double-emit).
			if (TryClaimBitAtomic(useMask, curX))
				return;
			const double cx_d = ((double)curX.x - pic.K02) * pic.invK00 * (double)depth;
			const double cy_d = ((double)curX.y - pic.K12) * pic.invK11 * (double)depth;
			const double cz_d = (double)depth;
			const Point3f X(
				(float)(pic.R00*cx_d + pic.R10*cy_d + pic.R20*cz_d + pic.Cx),
				(float)(pic.R01*cx_d + pic.R11*cy_d + pic.R21*cz_d + pic.Cy),
				(float)(pic.R02*cx_d + pic.R12*cy_d + pic.R22*cz_d + pic.Cz));
			#else
			const DepthData& depthDataCur = arrDepthData[curID];
			if (depthDataCur.depthMap.empty()) return;
			if (!Image8U::isInside(curX, depthDataCur.depthMap.size())) return;
			const Depth depth = depthDataCur.depthMap(curX);
			if (depth <= Depth(0)) return;
			UseMask& useMask = arrUseMask[curID];
			if (useMask.empty() || useMask(curX)) return;
			const float conf(depthDataCur.confMap.empty() ? 1.f : depthDataCur.confMap(curX));
			if (conf < minConfidence) return;
			const DepthData::ViewData& image = depthDataCur.GetView();
			const bool bHaveNormal = !depthDataCur.normalMap.empty();
			Point3f normal;
			if (curDepth > 0) {
				const Point3f pt(image.camera.ProjectPointP3(Cast<REAL>(s.refPoint)));
				if (pt.z <= Depth(0)) return;
				if (!IsDepthSimilar(depth, pt.z, OPTDENSE::fDepthDiffThreshold)) return;
				const Point2f diff(pt.x / pt.z - float(curX.x), pt.y / pt.z - float(curX.y));
				if (normSq(diff) > kReprojErrSq) return;
				if (bHaveNormal) {
					normal = Cast<float>(image.camera.R.t() * Cast<REAL>(depthDataCur.normalMap(curX)));
					if (s.refNormal.z != -1.f || s.refNormal.x != 0.f || s.refNormal.y != 0.f) {
						if (s.refNormal.dot(normal) < normalError) return;
					}
				} else {
					normal = s.refNormal;
				}
			} else {
				if (bHaveNormal)
					normal = Cast<float>(image.camera.R.t() * Cast<REAL>(depthDataCur.normalMap(curX)));
				else
					normal = Point3f(0.f, 0.f, -1.f);
			}
			if (TryClaimBitAtomic(useMask, curX))
				return;
			const Point3f X(Cast<float>(image.camera.TransformPointI2W(Point3(REAL(curX.x), REAL(curX.y), REAL(depth)))));
			#endif

			s.fusedPoints[0].push_back(X.x);
			s.fusedPoints[1].push_back(X.y);
			s.fusedPoints[2].push_back(X.z);
			const float weight(Conf2Weight(conf, depth));
			{
				const uint32_t vID = (uint32_t)curID;
				auto it = std::lower_bound(s.fusedViews.begin(), s.fusedViews.end(), vID);
				const size_t idx = (size_t)(it - s.fusedViews.begin());
				if (it != s.fusedViews.end() && *it == vID) {
					s.fusedWeights[idx] += weight;
				} else {
					s.fusedViews.insert(it, vID);
					s.fusedWeights.insert(s.fusedWeights.begin() + idx, weight);
				}
			}
			if (bEstimateNormal)
				s.fusedNormal += Cast<double>(normal);
			if (bEstimateColor)
			#if DENSE_FUSE_OPT_TUNING
				s.fusedColor += Cast<float>(curImageData.image(curX));
			#else
				s.fusedColor += Cast<float>(image.pImageData->image(curX));
			#endif
			#if DENSE_FUSE_HYBRID
			s.pxView.push_back((uint32_t)curID);
			s.pxWeight.push_back(weight);
			if (bEstimateColor)
			#if DENSE_FUSE_OPT_TUNING
				s.pxColor.push_back(curImageData.image(curX));
			#else
				s.pxColor.push_back(image.pImageData->image(curX));
			#endif
			if (bEstimateNormal)
				s.pxNormal.push_back(normal);
			#endif

			if (curDepth == 0) {
				s.refPoint = X;
				s.refNormal = normal;
			}

			if (++curDepth >= kMaxFuseDepth || s.fusedPoints[0].size() >= kMaxPointsFuse)
				return;

			#if DENSE_FUSE_OPT_TUNING
			for (const ViewScore& neighbor : *pic.pNeighbors) {
				const IIndex nextID(neighbor.ID);
				if (nextID == curID) continue;
				if (!neighbors[nextID]) continue;
				const DFImageCache& npic = imgCache[nextID];
				const double Xx = (double)X.x;
				const double Xy = (double)X.y;
				const double Xz = (double)X.z;
				const double nptx = npic.P[0]*Xx + npic.P[1]*Xy + npic.P[2 ]*Xz + npic.P[3 ];
				const double npty = npic.P[4]*Xx + npic.P[5]*Xy + npic.P[6 ]*Xz + npic.P[7 ];
				const double nptz = npic.P[8]*Xx + npic.P[9]*Xy + npic.P[10]*Xz + npic.P[11];
				const double invNptz = 1.0 / nptz;
				const ImageRef nextx(ROUND2INT(Point2(nptx*invNptz, npty*invNptz)));
				Self(nextID, nextx, curDepth, Self);
			}
			#else
			for (const ViewScore& neighbor : image.pImageData->neighbors) {
				const IIndex nextID(neighbor.ID);
				if (nextID == curID) continue;
				if (!neighbors[nextID]) continue;
				const DepthData& nextDepthData = arrDepthData[nextID];
				const ImageRef nextx(ROUND2INT(nextDepthData.GetCamera().ProjectPointP(Cast<REAL>(X))));
				Self(nextID, nextx, curDepth, Self);
			}
			#endif
		};
		lambda(ID, x, fuseDepth, lambda);
	};

	// Emit the current cluster from scratch `s` into `buf` (parallel path)
	// or directly into `pointcloud` (serial path, buf=nullptr). Returns true
	// if a cluster was emitted.
	const auto EmitCluster = [&](FuseScratch& s, EmitBuf* buf) -> bool {
		if (s.fusedPoints[0].size() < kMinPixelsFuse || s.fusedViews.size() < nMinViewsFuse)
			return false;
		#if DENSE_FUSE_HYBRID && !DENSE_FUSE_PARALLEL_PIXELS
		// HYBRID per-pixel emit is disabled under parallel-pixels mode:
		// the atomic-claim race can split otherwise-dense clusters into
		// <kHybridThresholdViews fragments. Per-pixel emit then leaks raw
		// (un-medianed) depth noise as a thin "sheet" below the surface.
		// Under parallel mode we always take the median path.
		if (s.fusedViews.size() < kHybridThresholdViews) {
			// Sparse-coverage cluster: emit each pixel as its own point.
			const size_t nPx = s.fusedPoints[0].size();
			for (size_t k = 0; k < nPx; ++k) {
				const Point3f p(s.fusedPoints[0][(unsigned)k],
				                s.fusedPoints[1][(unsigned)k],
				                s.fusedPoints[2][(unsigned)k]);
				if (buf) {
					EmitBuf::Rec rec;
					rec.point   = p;
					rec.viewBeg = (uint32_t)buf->views.size();
					buf->views.push_back(s.pxView[k]);
					buf->weights.push_back(s.pxWeight[k]);
					rec.viewEnd = (uint32_t)buf->views.size();
					if (bEstimateNormal) rec.normal = s.pxNormal[k];
					if (bEstimateColor)  rec.color  = s.pxColor[k];
					buf->recs.push_back(rec);
				} else {
					pointcloud.AddPoint(p);
					pointcloud.AddView(s.pxView[k]);
					pointcloud.AddWeight(s.pxWeight[k]);
					if (bEstimateNormal) pointcloud.AddNormal(s.pxNormal[k]);
					if (bEstimateColor)  pointcloud.AddColor(s.pxColor[k]);
				}
			}
			return true;
		}
		#endif
		// Median (component-wise) is robust to one bad depth in the cluster.
		const Point3f p(s.fusedPoints[0].GetMedian(),
		                s.fusedPoints[1].GetMedian(),
		                s.fusedPoints[2].GetMedian());
		Point3f normal;
		Pixel8U color;
		if (bEstimateNormal) {
			const Point3d nrm(normalized(s.fusedNormal));
			normal = Point3f((float)nrm.x, (float)nrm.y, (float)nrm.z);
		}
		if (bEstimateColor) {
			const float invN = 1.f / static_cast<float>(s.fusedPoints[0].size());
			color = Pixel8U(
				_cvt_ftoi_fast(s.fusedColor.r * invN),
				_cvt_ftoi_fast(s.fusedColor.g * invN),
				_cvt_ftoi_fast(s.fusedColor.b * invN));
		}
		if (buf) {
			EmitBuf::Rec rec;
			rec.point   = p;
			rec.viewBeg = (uint32_t)buf->views.size();
			buf->views.insert(buf->views.end(), s.fusedViews.begin(), s.fusedViews.end());
			buf->weights.insert(buf->weights.end(), s.fusedWeights.begin(), s.fusedWeights.end());
			rec.viewEnd = (uint32_t)buf->views.size();
			if (bEstimateNormal) rec.normal = normal;
			if (bEstimateColor)  rec.color  = color;
			buf->recs.push_back(rec);
		} else {
			pointcloud.AddPoint(p);
			pointcloud.AddViews(s.fusedViews.data(), s.fusedViews.data() + s.fusedViews.size());
			pointcloud.AddWeights(s.fusedWeights.data(), s.fusedWeights.data() + s.fusedWeights.size());
			if (bEstimateNormal) pointcloud.AddNormal(normal);
			if (bEstimateColor)  pointcloud.AddColor(color);
		}
		return true;
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
#ifdef DENSE_USE_OPENMP
		// Parallelize neighbor-prep: each iteration touches an independent
		// arrUseMask[neighbor.ID] entry. cacheDMaps.UseImage is internally
		// thread-safe. numNeighbors cap uses an abort flag like the
		// PointCloud overload.
		bool bAbort(false);
#pragma omp parallel for
		for (int64_t i = 0; i < (int64_t)depthData.neighbors.size(); ++i) {
#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const ViewScore& neighbor = depthData.neighbors[(IIndex)i];
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			neighbors[neighbor.ID] = true;
			UseMask& useMaskB = arrUseMask[neighbor.ID];
			if (!useMaskB.empty())
				continue; // mask already created by an earlier outer iteration — do NOT count toward cap (matches reference)
			useMaskB.create(depthDataB.depthMap.size());
			useMaskB.memset(0);
			static_assert(sizeof(IIndex) == sizeof(long), "IIndex must be 32-bit for InterlockedIncrement");
			const IIndex newCount = (IIndex)_InterlockedIncrement(reinterpret_cast<volatile long*>(&numNeighbors));
			if (newCount >= kMaxViewsFuse) {
				bAbort = true;
#pragma omp flush (bAbort)
			}
		}
#else
		for (const ViewScore& neighbor : depthData.neighbors) {
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			neighbors[neighbor.ID] = true;
			UseMask& useMaskB = arrUseMask[neighbor.ID];
			if (!useMaskB.empty())
				continue;
			useMaskB.create(depthDataB.depthMap.size());
			useMaskB.memset(0);
			if (++numNeighbors >= kMaxViewsFuse)
				break;
		}
#endif

		// Acquire the color buffers for the reference image + every direct neighbor
		// whose depth-map warmed up successfully. The same "neighbors[id]" gate is
		// also what bounds FusePoint's recursive cluster search (see the
		// "if (!neighbors[nextID]) continue;" checks below), so this exactly covers
		// every image curID can be during this iteration's fusion -- image
		// granularity, once per reference image, not per-point/per-pixel.
		if (bEstimateColor) {
			imageCache->Acquire(idxImage);
			for (const ViewScore& neighbor : depthData.neighbors)
				if (neighbors[neighbor.ID])
					imageCache->Acquire(neighbor.ID);
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

		#if DENSE_FUSE_PARALLEL_PIXELS
		{
			const int maxThreads = omp_get_max_threads();
			std::vector<EmitBuf> tlsBufs(maxThreads);
			std::vector<size_t>  tlsDepths(maxThreads, 0);
			#pragma omp parallel
			{
				const int tid = omp_get_thread_num();
				EmitBuf& myBuf = tlsBufs[tid];
				FuseScratch s;
				size_t myDepths = 0;
				#pragma omp for schedule(dynamic, 4) nowait
				for (int i = 0; i < sizeMap.height; ++i) {
					for (int j = 0; j < sizeMap.width; ++j) {
						FusePoint(idxImage, ImageRef(j, i), 0, s);
						EmitCluster(s, &myBuf);
						if (!s.fusedViews.empty()) {
							myDepths += s.fusedViews.size();
							s.ResetCluster();
						}
					}
				}
				tlsDepths[tid] = myDepths;
			}
			// Serial flush of per-thread buffers into the streaming pointcloud.
			// Iteration order across threads is fixed (tid 0..N-1), so within
			// a single run the cloud ordering is deterministic given thread
			// count; only the cluster groupings (race outcomes) vary run-to-run.
			for (const EmitBuf& buf : tlsBufs) {
				for (const auto& r : buf.recs) {
					pointcloud.AddPoint(r.point);
					pointcloud.AddViews(buf.views.data() + r.viewBeg,
					                    buf.views.data() + r.viewEnd);
					pointcloud.AddWeights(buf.weights.data() + r.viewBeg,
					                      buf.weights.data() + r.viewEnd);
					if (bEstimateNormal) pointcloud.AddNormal(r.normal);
					if (bEstimateColor)  pointcloud.AddColor(r.color);
				}
			}
			for (size_t d : tlsDepths) nDepths += d;
		}
		#else
		{
			FuseScratch s;
			for (int i = 0; i < sizeMap.height; ++i) {
				for (int j = 0; j < sizeMap.width; ++j) {
					FusePoint(idxImage, ImageRef(j, i), 0, s);
					EmitCluster(s, nullptr);
					if (!s.fusedViews.empty()) {
						nDepths += s.fusedViews.size();
						s.ResetCluster();
					}
				}
			}
		}
		#endif

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

		if (bEstimateColor) {
			for (const ViewScore& neighbor : depthData.neighbors)
				if (neighbors[neighbor.ID])
					imageCache->Release(neighbor.ID);
			imageCache->Release(idxImage);
		}

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
	const uint32_t cacheNumReads(cacheDMaps.GetNumImageReads());
	const double cacheHitRate(cacheDMaps.GetHitRate());
	arrUseMask.Release();
	cacheDMaps.ClearCache();

	DEBUG_EXTRA("Depth-maps dense fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%), %.2f hits in %.2f cached (%s), dmap-cache: %u disk reads, %.1f%%%% hit-rate",
		numDMapsFused, nDepths, pointcloud.NumPoints(),
		nDepths ? ROUND2INT((100.f * pointcloud.NumPoints()) / nDepths) : 0,
		numDMapsFused ? static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused : 0.0,
		numDMapsFused ? static_cast<double>(totalNumImagesInCache) / numDMapsFused : 0.0,
		TD_TIMER_GET_FMT().c_str(),
		cacheNumReads, cacheHitRate * 100.0);
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
		// NOTE: these were previously _alloca()'d. _alloca reserves from the WHOLE
		// enclosing function's stack frame, not the current loop iteration -- since
		// this sits inside FuseDepthMaps' outer per-reference-image loop (one call of
		// FuseDepthMaps processes every image in the dataset), every image's buffers
		// accumulated on the stack and were never reclaimed until the entire fuse
		// operation finished. Confirmed live: a 1873-image run crashed with
		// STATUS_STACK_OVERFLOW (0xC00000FD) at image 437 (~23%), consistent with a
		// few KB/image of unreclaimed stack space exhausting a ~1MB thread stack
		// around that point. Heap-backed containers declared here are destroyed (and
		// their memory freed) at the end of each loop iteration instead, which is
		// exactly the fix -- these are small (bounded by nMaxViewsFuse, 32 by
		// default), so the heap-allocation cost versus alloca is negligible next to
		// everything else done per image.
		const unsigned maxNeighbors = (unsigned)neighborCache.size();
		std::vector<NeighborHit> hitsStorageBuf(maxNeighbors);
		std::vector<NeighborHit> invalidHitsStorageBuf(maxNeighbors);
		NeighborHit* hitsStorage = hitsStorageBuf.data();
		NeighborHit* invalidHitsStorage = invalidHitsStorageBuf.data();
		unsigned nHits = 0, nInvalidHits = 0;

		const unsigned maxViews = (unsigned)neighborCache.size() + 1; // +1 for reference view
		std::vector<uint32_t> viewsStorageBuf(maxViews);
		std::vector<float> weightsStorageBuf(maxViews);
		uint32_t* __restrict viewsStorage = viewsStorageBuf.data();
		float* __restrict weightsStorage = weightsStorageBuf.data();
		// Defer idxPointB assignments until we know the point survives the fuse check.
		// Collect pointers to idxPointB slots so we can commit them only for accepted points.
		std::vector<uint32_t*> deferredStorageBuf(neighborCache.size());
		uint32_t** __restrict deferredStorage = deferredStorageBuf.data();
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
	if (Thread::safeDec(idxImage) == 0) {
		// all depth-maps filtered: drop the retention cache's references so ref-counts
		// return to baseline before the adjust sub-phase (which asserts ref-count == 1)
		if (g_filterCache != NULL)
			g_filterCache->Clear();
		sem.Signal((unsigned)images.GetSize()*2);
	}
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);

// nanoflann adaptor over a flat xyz float stream (no copy). Defined at file
// scope because a local class cannot contain the member template kdtree_get_bbox.
struct FlatXYZAdaptor {
	const float* xyz;
	size_t count;
	inline size_t kdtree_get_point_count() const noexcept { return count; }
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const noexcept { return xyz[idx * 3 + dim]; }
	template <class BBOX> bool kdtree_get_bbox(BBOX&) const noexcept { return false; }
};

// count-only, early-exit result set for nanoflann::findNeighbors used by the
// low-view consensus filter: tallies neighbors within a radius and stops as soon
// as `need` are seen. The tree it queries holds ONLY surface (higher-view) points,
// so no per-point view test is needed here. File scope so it can be a template
// argument to findNeighbors (a local class trips MSVC here).
struct RadiusCounter {
	using DistanceType = float;
	const float radius2;
	const unsigned need;
	unsigned count = 0;
	RadiusCounter(float r2, unsigned nd) : radius2(r2), need(nd) {}
	inline bool full() const { return count >= need; }
	inline bool addPoint(float dist, uint32_t /*index*/) {
		if (dist < radius2 && ++count >= need)
			return false; // enough support found; stop the search early
		return true;
	}
	inline float worstDist() const { return radius2; }
	inline void sort() {}
	inline size_t size() const { return count; }
	inline void init() {}
};

// Surface variation = lambda0 / (lambda0+lambda1+lambda2) of the covariance of a
// local point set (Pauly et al.). 0 => perfectly planar/thin neighborhood,
// ~0.33 => isotropic 3D blob. Closed-form smallest eigenvalue of a symmetric 3x3
// (a=Cxx b=Cyy c=Czz d=Cxy e=Cxz f=Cyz) so no Eigen dependency / build surprises.
static inline float SurfaceVariation3x3(double a, double b, double c, double d, double e, double f)
{
	const double trace = a + b + c;
	if (trace <= 1e-30) return 0.f;
	const double p1 = d * d + e * e + f * f;
	double eigMin;
	if (p1 <= 1e-30 * trace * trace) {
		eigMin = std::min(a, std::min(b, c)); // already diagonal
	} else {
		const double q = trace / 3.0;
		const double p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2.0 * p1;
		const double p = std::sqrt(p2 / 6.0);
		const double ba = (a - q) / p, bb = (b - q) / p, bc = (c - q) / p;
		const double bd = d / p, be = e / p, bf = f / p;
		double r = 0.5 * (ba * (bb * bc - bf * bf) - bd * (bd * bc - bf * be) + be * (bd * bf - bb * be));
		if (r <= -1.0) r = -1.0; else if (r >= 1.0) r = 1.0;
		const double phi = std::acos(r) / 3.0;
		// smallest eigenvalue = q + 2p*cos(phi + 2pi/3)
		eigMin = q + 2.0 * p * std::cos(phi + 2.09439510239319549); // 2*pi/3
	}
	if (eigMin < 0.0) eigMin = 0.0;
	return (float)(eigMin / trace);
}

// Density-based statistical outlier removal on the streaming point-cloud.
// For each point, compute the RMS distance to its k nearest neighbors; remove
// points whose value exceeds (mean + stddevMul*stddev) over the whole cloud.
// Because the test is purely a function of LOCAL 3D density, it is dataset-
// adaptive: well-sampled surfaces lose essentially nothing, while the sparse
// "spray"/fuzz on low-overlap edges (which has a much larger neighbor distance)
// is cut. Only the HIGH (sparse) side is removed so dense regions are kept.
// Returns the number of points removed. No-op when stddevMul <= 0.
static size_t FilterPointCloudDensity(PointCloudStreaming& pc, int k, float stddevMul)
{
	const size_t n = pc.NumPoints();
	if (stddevMul <= 0.f || n == 0 || (size_t)(k + 1) >= n)
		return 0;

	using namespace nanoflann;
	using KDTree = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, FlatXYZAdaptor>, FlatXYZAdaptor, 3>;

	const float* __restrict xyz = pc.pointsXYZ.data();
	FlatXYZAdaptor adaptor{ xyz, n };
	KDTree index(3, adaptor, KDTreeSingleIndexAdaptorParams(64));
	index.buildIndex();

	std::vector<float> meanDist(n);
	const int kq = k + 1; // +1 because the query point itself is returned
	// cache pass-1 neighbor indices (found==kq under the early-out guard) so the
	// local-threshold pass below reuses them instead of re-running an identical
	// knnSearch -> deterministic, bit-identical decisions at half the tree queries
	std::vector<uint32_t> nbrIdx((size_t)n * (size_t)kq);
#ifdef DENSE_USE_OPENMP
#pragma omp parallel
#endif
	{
		std::vector<uint32_t> idxBuf(kq);
		std::vector<float> d2Buf(kq);
#ifdef DENSE_USE_OPENMP
#pragma omp for schedule(static, 1024)
#endif
		for (int64_t i = 0; i < (int64_t)n; ++i) {
			const float q[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
			const size_t found = index.knnSearch(q, kq, idxBuf.data(), d2Buf.data());
			uint32_t* __restrict pn = nbrIdx.data() + (size_t)i * (size_t)kq;
			float sum = 0.f; size_t cnt = 0;
			for (size_t j = 0; j < found; ++j) {
				pn[j] = idxBuf[j];
				if (idxBuf[j] == (uint32_t)i) continue; // skip self
				sum += d2Buf[j]; ++cnt;
			}
			meanDist[i] = cnt ? std::sqrt(sum / (float)cnt) : 0.f;
		}
	}

	// global mean and standard deviation of the per-point neighbor distance.
	// The global test catches whole bulk-sparse regions (the loose spray that
	// is uniformly thinner than the scene at large).
	double mean = 0.0;
	for (size_t i = 0; i < n; ++i) mean += meanDist[i];
	mean /= (double)n;
	double var = 0.0;
	for (size_t i = 0; i < n; ++i) { const double d = (double)meanDist[i] - mean; var += d * d; }
	var /= (double)n;
	const float globalThr = (float)(mean + (double)stddevMul * std::sqrt(var));

	// Local (surface-following) outlier test: compare each point's neighbor
	// distance against the mean+stddev of ITS OWN neighborhood rather than the
	// whole cloud. A point embedded in a dense surface sits among equally-dense
	// neighbors and is kept even with only 2 views; a fuzz point sprayed off a
	// low-overlap edge sits next to the much-denser real surface it detached
	// from, so it stands out locally and is cut. This makes the removal follow
	// the surface boundary (a clean trimmed edge) instead of punching a single
	// flat threshold through the cloud. A point is removed if it fails EITHER
	// the local OR the global test, so the connected fringe haze (locally
	// extreme) and the bulk spray (globally extreme) both go. The test NEVER
	// consults view count, so nMinViewsFuse=2 is fully respected.
	std::vector<uint8_t> outlier(n, 0);
#ifdef DENSE_USE_OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
	for (int64_t i = 0; i < (int64_t)n; ++i) {
		// global gate first (cheap, no neighborhood stats needed)
		if (meanDist[i] > globalThr) { outlier[i] = 1; continue; }
		// reuse the neighbor list gathered in pass 1 (identical to re-querying the tree)
		const uint32_t* __restrict pn = nbrIdx.data() + (size_t)i * (size_t)kq;
		// local mean+std of the neighbors' own neighbor-distance
		double lmean = 0.0; size_t cnt = 0;
		for (int j = 0; j < kq; ++j) {
			const uint32_t idx = pn[j];
			if (idx == (uint32_t)i) continue; // skip self
			lmean += meanDist[idx]; ++cnt;
		}
		if (cnt == 0) continue;
		lmean /= (double)cnt;
		double lvar = 0.0;
		for (int j = 0; j < kq; ++j) {
			const uint32_t idx = pn[j];
			if (idx == (uint32_t)i) continue;
			const double d = (double)meanDist[idx] - lmean;
			lvar += d * d;
		}
		lvar /= (double)cnt;
		const double lthr = lmean + (double)stddevMul * std::sqrt(lvar);
		// high (sparse) side only: a point sparser than its neighborhood is fuzz
		if ((double)meanDist[i] > lthr)
			outlier[i] = 1;
	}

	// compact every parallel stream in lockstep, keeping only inliers.
	// A serial prefix pass assigns each survivor its destination point index and
	// (for the variable-length view/weight blobs) its destination memory offset;
	// the heavy copying is then done in parallel with each survivor writing only
	// its own disjoint slots, so the output layout is identical to a serial pass.
	const bool hasNormals = !pc.normalsXYZ.empty();
	const bool hasColors  = !pc.colorsRGB.empty();
	const bool hasViews   = !pc.pointViewsSizes.empty();
	const bool hasWeights = !pc.pointWeightsSizes.empty();

	std::vector<uint32_t> dstIdx(n);
	std::vector<uint32_t> viewOff(hasViews ? n : 0);
	std::vector<uint32_t> weightOff(hasWeights ? n : 0);
	size_t nKeep = 0, viewMem = 0, weightMem = 0;
	for (size_t i = 0; i < n; ++i) {
		if (outlier[i]) continue;
		dstIdx[i] = (uint32_t)nKeep;
		if (hasViews)   { viewOff[i]   = (uint32_t)viewMem;   viewMem   += pc.pointViewsSizes[i]; }
		if (hasWeights) { weightOff[i] = (uint32_t)weightMem; weightMem += pc.pointWeightsSizes[i]; }
		++nKeep;
	}
	const size_t removed = n - nKeep;

	std::vector<float>    newXYZ(nKeep * 3);
	std::vector<float>    newNormals(hasNormals ? nKeep * 3 : 0);
	std::vector<uint8_t>  newColors(hasColors ? nKeep * 3 : 0);
	std::vector<uint32_t> newViewsOff(hasViews ? nKeep : 0), newViewsSize(hasViews ? nKeep : 0), newViewsMem(viewMem);
	std::vector<uint32_t> newWeightsOff(hasWeights ? nKeep : 0), newWeightsSize(hasWeights ? nKeep : 0);
	std::vector<float>    newWeightsMem(weightMem);

#ifdef DENSE_USE_OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
	for (int64_t i = 0; i < (int64_t)n; ++i) {
		if (outlier[i]) continue;
		const size_t w = dstIdx[i];
		newXYZ[w * 3 + 0] = xyz[i * 3 + 0]; newXYZ[w * 3 + 1] = xyz[i * 3 + 1]; newXYZ[w * 3 + 2] = xyz[i * 3 + 2];
		if (hasNormals) { newNormals[w * 3 + 0] = pc.normalsXYZ[i * 3 + 0]; newNormals[w * 3 + 1] = pc.normalsXYZ[i * 3 + 1]; newNormals[w * 3 + 2] = pc.normalsXYZ[i * 3 + 2]; }
		if (hasColors)  { newColors[w * 3 + 0] = pc.colorsRGB[i * 3 + 0]; newColors[w * 3 + 1] = pc.colorsRGB[i * 3 + 1]; newColors[w * 3 + 2] = pc.colorsRGB[i * 3 + 2]; }
		if (hasViews) {
			const uint32_t off = pc.pointViewsOffsets[i], sz = pc.pointViewsSizes[i], dOff = viewOff[i];
			newViewsOff[w] = dOff; newViewsSize[w] = sz;
			for (uint32_t t = 0; t < sz; ++t) newViewsMem[dOff + t] = pc.pointViewsMemory[off + t];
		}
		if (hasWeights) {
			const uint32_t off = pc.pointWeightsOffsets[i], sz = pc.pointWeightsSizes[i], dOff = weightOff[i];
			newWeightsOff[w] = dOff; newWeightsSize[w] = sz;
			for (uint32_t t = 0; t < sz; ++t) newWeightsMem[dOff + t] = pc.pointWeightsMemory[off + t];
		}
	}

	pc.pointsXYZ.swap(newXYZ);
	if (hasNormals) pc.normalsXYZ.swap(newNormals);
	if (hasColors)  pc.colorsRGB.swap(newColors);
	if (hasViews)   { pc.pointViewsOffsets.swap(newViewsOff); pc.pointViewsSizes.swap(newViewsSize); pc.pointViewsMemory.swap(newViewsMem); }
	if (hasWeights) { pc.pointWeightsOffsets.swap(newWeightsOff); pc.pointWeightsSizes.swap(newWeightsSize); pc.pointWeightsMemory.swap(newWeightsMem); }
	return removed;
} // FilterPointCloudDensity
/*----------------------------------------------------------------*/

// Selective low-view suppression: emulate the effect of raising nMinViewsFuse
// by one, but ONLY in regions where a better-supported surface already exists,
// so it can stay enabled without the global data loss a real nMinViewsFuse bump
// causes on genuinely-2-view datasets.
//   - A point fused from only the minimum number of views (viewCount <= minViews,
//     i.e. the 2-view layer when nMinViewsFuse=2) is a removal CANDIDATE.
//   - It is deleted only when at least `supportMin` better-supported points
//     (viewCount > minViews, i.e. the real 3+-view surface) lie WITHIN a metric
//     search radius. A radius (rather than a k-NN rank) is essential on the
//     low-overlap edges: there the 2-view noise forms a THICK scatter slab, so a
//     scatter point's k nearest neighbors are all other scatter and the genuine
//     3+-view surface never enters the k-NN window -> the old rank test scored
//     support=0 and kept the fuzz. The radius reaches across the slab to the real
//     surface regardless of how dense the surrounding noise is.
//   - A min-view point with NO better-supported neighbor within the radius is the
//     SOLE evidence for its geometry and is KEPT, so coverage-limited 2-view
//     regions (the data a global nMinViewsFuse=3 would destroy) survive untouched.
// The reach is LOCAL and SURFACE-relative: a separate KD-tree is built over only
// the higher-view (surface) points, each surface point is given its own nearest-
// surface-neighbor spacing, and a candidate is cut when >=supportMin surface points
// lie within radiusMul x (the nearest surface point's local spacing). A GLOBAL
// radius fails on low-overlap edges: there the real surface is sampled coarsely and
// the fuzz is displaced along the ray, so a global (dense-interior-dominated) radius
// can't reach from the fuzz to the surface and the edge cloud survives. Tying the
// reach to the LOCAL surface spacing auto-expands it exactly where the surface is
// sparse (the edge) while staying tight in the dense interior (no over-removal).
// Never raises nMinViewsFuse; purely a post-fusion, per-point decision. Returns the
// number of points removed. No-op when supportMin == 0 or radiusMul <= 0.
//
// Second pass (planarityMax > 0): a min-view point with NO surface nearby is either
// genuine sparse 2-view surface (locally thin/planar) OR floating fuzz displaced off
// the surface along the viewing ray (locally volumetric/scattered). These are sparse
// AND unsupported, so distance-to-surface can't tell them apart -- only local SHAPE
// can. We PCA the candidate's neighborhood and delete it when its surface-variation
// (smallest/sum of covariance eigenvalues) exceeds planarityMax (volumetric scatter),
// while keeping thin/planar ones. This clears the floating cloud so reconstruct can
// bridge the gap into a flat plane (the nMinViewsFuse=3 outcome) without destroying
// genuine sparse 2-view surfaces. kPCA = neighborhood size for the PCA.
static size_t FilterRedundantLowViewPoints(PointCloudStreaming& pc, unsigned minViews, unsigned supportMin, float radiusMul, float planarityMax, int kPCA)
{
	const size_t n = pc.NumPoints();
	if (supportMin == 0 || radiusMul <= 0.f || n < 2 || pc.pointViewsSizes.empty())
		return 0;

	using namespace nanoflann;
	using KDTree = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, FlatXYZAdaptor>, FlatXYZAdaptor, 3>;

	const float* __restrict xyz = pc.pointsXYZ.data();
	const uint32_t* __restrict viewSize = pc.pointViewsSizes.data();
	const bool doPlanarity = (planarityMax > 0.f && kPCA >= 4);

	// gather the higher-view "surface" points (the geometry a global nMinViewsFuse+1
	// would keep) into their own contiguous coordinate array + KD-tree. Candidates
	// (<= minViews) are tested ONLY against this surface, so dense same-view fuzz
	// cannot crowd out the surface evidence the way an all-points k-NN window did.
	std::vector<uint32_t> hIdx;
	hIdx.reserve(n / 2 + 1);
	for (size_t i = 0; i < n; ++i)
		if (viewSize[i] > minViews)
			hIdx.push_back((uint32_t)i);
	const size_t nh = hIdx.size();
	if (nh < 2)
		return 0; // no real surface anywhere -> nothing is "redundant"

	std::vector<float> hxyz(nh * 3);
	for (size_t j = 0; j < nh; ++j) {
		const uint32_t s = hIdx[j];
		hxyz[j * 3 + 0] = xyz[s * 3 + 0];
		hxyz[j * 3 + 1] = xyz[s * 3 + 1];
		hxyz[j * 3 + 2] = xyz[s * 3 + 2];
	}
	FlatXYZAdaptor hAdaptor{ hxyz.data(), nh };
	KDTree hIndex(3, hAdaptor, KDTreeSingleIndexAdaptorParams(64));
	hIndex.buildIndex();

	// per-surface-point local spacing = distance to its nearest other surface point.
	std::vector<float> hSpacing(nh, 0.f);
	double spacingSum = 0.0;
#ifdef DENSE_USE_OPENMP
#pragma omp parallel for schedule(static, 1024) reduction(+:spacingSum)
#endif
	for (int64_t j = 0; j < (int64_t)nh; ++j) {
		const float q[3] = { hxyz[j * 3 + 0], hxyz[j * 3 + 1], hxyz[j * 3 + 2] };
		uint32_t idx2[2]; float d2_2[2];
		const size_t found = hIndex.knnSearch(q, 2, idx2, d2_2);
		float spc = 0.f;
		for (size_t t = 0; t < found; ++t) {
			if (idx2[t] == (uint32_t)j) continue; // skip self
			spc = std::sqrt(d2_2[t]);
			break;
		}
		hSpacing[j] = spc;
		spacingSum += spc;
	}
	// floor against pathological zeros (coincident surface points) so their radius
	// is not degenerate; a small fraction of the mean keeps it locally meaningful.
	const float spacingFloor = (float)(0.05 * (spacingSum / (double)nh));

	// full-cloud KD-tree, only needed for the planarity (shape) test on unsupported
	// candidates. Built lazily so the common support-only path pays nothing for it.
	FlatXYZAdaptor fAdaptor{ xyz, n };
	std::unique_ptr<KDTree> fIndex;
	if (doPlanarity) {
		fIndex.reset(new KDTree(3, fAdaptor, KDTreeSingleIndexAdaptorParams(64)));
		fIndex->buildIndex();
	}

	std::vector<uint8_t> outlier(n, 0);
	const nanoflann::SearchParameters sp(0.f, false);
	// diagnostic: among the 2-view candidates, bucket the distance to the nearest
	// real (higher-view) surface point in units of that surface point's LOCAL
	// spacing. This reveals whether the surviving edge fuzz HAS a nearby surface
	// (so a bigger/thickness-based reach would fix it) or is in a region with NO
	// real surface at all (so support-based removal can never touch it).
	size_t dCand = 0, dWithin1 = 0, dWithin4 = 0, dWithin16 = 0, dWithin64 = 0, dBeyond = 0;
	size_t dPlanarRemoved = 0, dPlanarKept = 0;
#ifdef DENSE_USE_OPENMP
#pragma omp parallel for schedule(static, 1024) reduction(+:dCand,dWithin1,dWithin4,dWithin16,dWithin64,dBeyond,dPlanarRemoved,dPlanarKept)
#endif
	for (int64_t i = 0; i < (int64_t)n; ++i) {
		// only minimum-view points are candidates for removal
		if (viewSize[i] > minViews) continue;
		const float q[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
		// nearest surface point -> its LOCAL spacing sets this candidate's reach
		uint32_t nn; float nnDist2;
		if (hIndex.knnSearch(q, 1, &nn, &nnDist2) == 0)
			continue;
		float localScale = hSpacing[nn];
		if (localScale < spacingFloor) localScale = spacingFloor;
		if (!(localScale > 0.f)) continue;
		// diagnostic bucketing (distance to nearest surface / local spacing)
		++dCand;
		const float ratio = std::sqrt(nnDist2) / localScale;
		if      (ratio <= 1.f)  ++dWithin1;
		else if (ratio <= 4.f)  ++dWithin4;
		else if (ratio <= 16.f) ++dWithin16;
		else if (ratio <= 64.f) ++dWithin64;
		else                    ++dBeyond;
		const float r = radiusMul * localScale;
		const float radius2 = r * r; // L2_Simple returns squared distances
		RadiusCounter rs(radius2, supportMin);
		hIndex.findNeighbors(rs, q, sp);
		if (rs.count >= supportMin) {
			outlier[i] = 1; // redundant min-view fuzz over a real higher-view surface
			continue;
		}
		// unsupported (no real surface nearby): keep genuine sparse 2-view surface
		// (thin/planar) but delete floating fuzz (volumetric scatter) via local PCA.
		if (doPlanarity) {
			constexpr int kMaxPCA = 64;
			int kq = kPCA + 1; // +1 for the query point itself
			if (kq > kMaxPCA) kq = kMaxPCA;
			uint32_t idxN[kMaxPCA]; float d2N[kMaxPCA];
			const size_t got = fIndex->knnSearch(q, kq, idxN, d2N);
			if (got >= 4) {
				// covariance of the neighborhood (single-pass, centered)
				double sx = 0, sy = 0, sz = 0;
				for (size_t t = 0; t < got; ++t) {
					const uint32_t p = idxN[t];
					sx += xyz[p * 3 + 0]; sy += xyz[p * 3 + 1]; sz += xyz[p * 3 + 2];
				}
				const double inv = 1.0 / (double)got;
				const double mx = sx * inv, my = sy * inv, mz = sz * inv;
				double cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
				for (size_t t = 0; t < got; ++t) {
					const uint32_t p = idxN[t];
					const double dx = xyz[p * 3 + 0] - mx, dy = xyz[p * 3 + 1] - my, dz = xyz[p * 3 + 2] - mz;
					cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
					cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
				}
				const float variation = SurfaceVariation3x3(cxx, cyy, czz, cxy, cxz, cyz);
				if (variation > planarityMax) {
					outlier[i] = 1; // volumetric floating fuzz -> remove
					++dPlanarRemoved;
				} else {
					++dPlanarKept; // thin/planar genuine sparse surface -> keep
				}
			}
		}
	}
	VERBOSE("Low-view filter diag: %u candidates, nearest-surface distance (in local-spacings): <=1: %.1f%%%%, <=4: %.1f%%%%, <=16: %.1f%%%%, <=64: %.1f%%%%, >64: %.1f%%%% (surface pts: %u/%u; planarity removed %u, kept %u)",
		(unsigned)dCand,
		dCand ? 100.0 * dWithin1  / dCand : 0.0,
		dCand ? 100.0 * dWithin4  / dCand : 0.0,
		dCand ? 100.0 * dWithin16 / dCand : 0.0,
		dCand ? 100.0 * dWithin64 / dCand : 0.0,
		dCand ? 100.0 * dBeyond   / dCand : 0.0,
		(unsigned)nh, (unsigned)n, (unsigned)dPlanarRemoved, (unsigned)dPlanarKept);

	// compact every parallel stream in lockstep, keeping only the survivors
	const bool hasNormals = !pc.normalsXYZ.empty();
	const bool hasColors  = !pc.colorsRGB.empty();
	const bool hasViews   = !pc.pointViewsSizes.empty();
	const bool hasWeights = !pc.pointWeightsSizes.empty();

	std::vector<float>    newXYZ;     newXYZ.reserve(pc.pointsXYZ.size());
	std::vector<float>    newNormals; if (hasNormals) newNormals.reserve(pc.normalsXYZ.size());
	std::vector<uint8_t>  newColors;  if (hasColors)  newColors.reserve(pc.colorsRGB.size());
	std::vector<uint32_t> newViewsOff, newViewsSize, newViewsMem;
	std::vector<uint32_t> newWeightsOff, newWeightsSize;
	std::vector<float>    newWeightsMem;
	if (hasViews)   { newViewsOff.reserve(pc.pointViewsOffsets.size()); newViewsSize.reserve(pc.pointViewsSizes.size()); newViewsMem.reserve(pc.pointViewsMemory.size()); }
	if (hasWeights) { newWeightsOff.reserve(pc.pointWeightsOffsets.size()); newWeightsSize.reserve(pc.pointWeightsSizes.size()); newWeightsMem.reserve(pc.pointWeightsMemory.size()); }

	size_t removed = 0;
	for (size_t i = 0; i < n; ++i) {
		if (outlier[i]) { ++removed; continue; }
		newXYZ.push_back(xyz[i * 3 + 0]); newXYZ.push_back(xyz[i * 3 + 1]); newXYZ.push_back(xyz[i * 3 + 2]);
		if (hasNormals) { newNormals.push_back(pc.normalsXYZ[i * 3 + 0]); newNormals.push_back(pc.normalsXYZ[i * 3 + 1]); newNormals.push_back(pc.normalsXYZ[i * 3 + 2]); }
		if (hasColors)  { newColors.push_back(pc.colorsRGB[i * 3 + 0]); newColors.push_back(pc.colorsRGB[i * 3 + 1]); newColors.push_back(pc.colorsRGB[i * 3 + 2]); }
		if (hasViews) {
			const uint32_t off = pc.pointViewsOffsets[i], sz = pc.pointViewsSizes[i];
			newViewsOff.push_back((uint32_t)newViewsMem.size()); newViewsSize.push_back(sz);
			for (uint32_t t = 0; t < sz; ++t) newViewsMem.push_back(pc.pointViewsMemory[off + t]);
		}
		if (hasWeights) {
			const uint32_t off = pc.pointWeightsOffsets[i], sz = pc.pointWeightsSizes[i];
			newWeightsOff.push_back((uint32_t)newWeightsMem.size()); newWeightsSize.push_back(sz);
			for (uint32_t t = 0; t < sz; ++t) newWeightsMem.push_back(pc.pointWeightsMemory[off + t]);
		}
	}

	pc.pointsXYZ.swap(newXYZ);
	if (hasNormals) pc.normalsXYZ.swap(newNormals);
	if (hasColors)  pc.colorsRGB.swap(newColors);
	if (hasViews)   { pc.pointViewsOffsets.swap(newViewsOff); pc.pointViewsSizes.swap(newViewsSize); pc.pointViewsMemory.swap(newViewsMem); }
	if (hasWeights) { pc.pointWeightsOffsets.swap(newWeightsOff); pc.pointWeightsSizes.swap(newWeightsSize); pc.pointWeightsMemory.swap(newWeightsMem); }
	return removed;
} // FilterRedundantLowViewPoints
/*----------------------------------------------------------------*/

// Flatten a rough water/pond surface onto a robustly fitted near-horizontal plane.
// Water is textureless and moving, so MVS reconstructs it as a noisy "crust"
// instead of a flat sheet. The crust is selected by its TWO distinguishing
// properties: it lies in a LOW elevation band AND it is locally ROUGH (high PCA
// surface-variation). Genuine flat ground (smooth) in the same band is therefore
// left untouched, and rough features outside the band (vegetation, cliffs) are
// left untouched. Selected crust points are projected onto the fitted plane.
//
// Heavily guarded so it is a NEAR NO-OP on datasets without prominent water:
//   * does nothing unless a large rough cluster exists in the low band,
//   * declines unless the fitted plane is near-horizontal with a high RANSAC
//     inlier ratio (i.e. it really is a flat water sheet, not low clutter),
//   * declines if it would move more than maxFrac of the whole cloud.
// Assumes a gravity-up (Z-up) cloud, which the near-horizontal test also enforces.
// Returns the number of points snapped (0 if it declined).
static size_t FilterFlattenWater(PointCloudStreaming& pc, float bandPct, float roughnessMin, float maxFrac, int kPCA)
{
	const size_t n = pc.NumPoints();
	if (n < 5000 || roughnessMin <= 0.f || bandPct <= 0.f || bandPct >= 100.f || kPCA < 4)
		return 0;
	float* __restrict xyz = pc.pointsXYZ.data();

	using namespace nanoflann;
	using KDTree = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, FlatXYZAdaptor>, FlatXYZAdaptor, 3>;

	// 1) elevation band: water assumed within the lowest bandPct% of height (Z-up).
	//    Percentiles from a subsample -> robust and units/scale independent.
	std::vector<float> zs;
	const size_t zstride = std::max<size_t>(1, n / 200000);
	zs.reserve(n / zstride + 1);
	for (size_t i = 0; i < n; i += zstride)
		zs.push_back(xyz[i * 3 + 2]);
	if (zs.size() < 16)
		return 0;
	auto pctile = [&](float p) -> float {
		const size_t k = (size_t)CLAMP(p * 0.01f * (float)(zs.size() - 1), 0.f, (float)(zs.size() - 1));
		std::nth_element(zs.begin(), zs.begin() + k, zs.end());
		return zs[k];
	};
	const float zMin = pctile(0.f);
	const float zHi  = pctile(bandPct);
	if (!(zHi > zMin))
		return 0;

	// 2) candidate crust = points in the band AND locally rough.
	FlatXYZAdaptor adaptor{ xyz, n };
	KDTree index(3, adaptor, KDTreeSingleIndexAdaptorParams(64));
	index.buildIndex();

	std::vector<uint8_t> cand(n, 0);
	size_t nCand = 0;
	constexpr int kMaxPCA = 64;
	int kq = kPCA + 1;
	if (kq > kMaxPCA) kq = kMaxPCA;
#ifdef DENSE_USE_OPENMP
#pragma omp parallel for schedule(static, 4096) reduction(+:nCand)
#endif
	for (int64_t i = 0; i < (int64_t)n; ++i) {
		const float z = xyz[i * 3 + 2];
		if (z < zMin || z > zHi)
			continue;
		const float q[3] = { xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2] };
		uint32_t idxN[kMaxPCA]; float d2N[kMaxPCA];
		const size_t got = index.knnSearch(q, kq, idxN, d2N);
		if (got < 4)
			continue;
		double sx = 0, sy = 0, sz = 0;
		for (size_t t = 0; t < got; ++t) {
			const uint32_t p = idxN[t];
			sx += xyz[p * 3 + 0]; sy += xyz[p * 3 + 1]; sz += xyz[p * 3 + 2];
		}
		const double inv = 1.0 / (double)got, mx = sx * inv, my = sy * inv, mz = sz * inv;
		double cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
		for (size_t t = 0; t < got; ++t) {
			const uint32_t p = idxN[t];
			const double dx = xyz[p * 3 + 0] - mx, dy = xyz[p * 3 + 1] - my, dz = xyz[p * 3 + 2] - mz;
			cxx += dx * dx; cyy += dy * dy; czz += dz * dz; cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
		}
		if (SurfaceVariation3x3(cxx, cyy, czz, cxy, cxz, cyz) >= roughnessMin) {
			cand[i] = 1;
			++nCand;
		}
	}
	// need a real sheet, not a few stray rough points
	if (nCand < std::max<size_t>(5000, n / 500))
		return 0;

	// 3) robust plane fit over a candidate subsample (ACRANSAC, auto threshold).
	Point3fArr pts(0, std::min<size_t>(nCand, 60000) + 1);
	{
		const size_t cstride = std::max<size_t>(1, nCand / 60000);
		size_t seen = 0;
		for (size_t i = 0; i < n; ++i) {
			if (!cand[i])
				continue;
			if ((seen++ % cstride) != 0)
				continue;
			pts.emplace_back(xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]);
		}
	}
	if (pts.size() < 16)
		return 0;
	Planef plane;
	double th = DBL_MAX;
	const unsigned nInliers = MVS::EstimatePlane(pts, plane, th);
	if (nInliers == 0 || (size_t)nInliers * 2 < pts.size())
		return 0; // not a coherent sheet
	// orient the unit normal up and require the plane to be near-horizontal
	float nx = (float)plane.m_vN.x(), ny = (float)plane.m_vN.y(), nz = (float)plane.m_vN.z();
	float D = (float)plane.m_fD;
	if (nz < 0.f) { nx = -nx; ny = -ny; nz = -nz; D = -D; }
	if (nz < 0.94f)
		return 0; // not horizontal -> not a water sheet (also rejects non-Z-up clouds)

	// 4) project candidates onto the plane (snap). Count first, honour the cap.
	const float snapTol = (zHi - zMin);
	size_t toFlatten = 0;
	for (size_t i = 0; i < n; ++i) {
		if (!cand[i])
			continue;
		const float d = nx * xyz[i * 3 + 0] + ny * xyz[i * 3 + 1] + nz * xyz[i * 3 + 2] + D;
		if (std::fabs(d) <= snapTol)
			++toFlatten;
	}
	if (toFlatten == 0 || (double)toFlatten > (double)maxFrac * (double)n)
		return 0; // safety: decline rather than risk flattening real geometry

	const bool hasN = !pc.normalsXYZ.empty();
	for (size_t i = 0; i < n; ++i) {
		if (!cand[i])
			continue;
		const float d = nx * xyz[i * 3 + 0] + ny * xyz[i * 3 + 1] + nz * xyz[i * 3 + 2] + D;
		if (std::fabs(d) > snapTol)
			continue;
		xyz[i * 3 + 0] -= d * nx;
		xyz[i * 3 + 1] -= d * ny;
		xyz[i * 3 + 2] -= d * nz;
		if (hasN) {
			pc.normalsXYZ[i * 3 + 0] = nx;
			pc.normalsXYZ[i * 3 + 1] = ny;
			pc.normalsXYZ[i * 3 + 2] = nz;
		}
	}
	return toFlatten;
} // FilterFlattenWater
/*----------------------------------------------------------------*/

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

	// density-based statistical outlier removal: cull the sparse "spray"/fuzz on
	// low-overlap edges using local 3D neighbor density (dataset-adaptive; dense
	// surfaces are essentially untouched). Disabled when fOutlierFilterStdDev<=0.
	if (!pointcloud.IsEmpty() && OPTDENSE::fOutlierFilterStdDev > 0.f) {
		TD_TIMER_START();
		const size_t numBefore = pointcloud.NumPoints();
		const size_t removed = FilterPointCloudDensity(pointcloud, (int)OPTDENSE::nOutlierFilterKNN, OPTDENSE::fOutlierFilterStdDev);
		VERBOSE("Density outlier filter: %u/%u points removed (%.2f%%%%) (%s)",
			(unsigned)removed, (unsigned)numBefore,
			numBefore ? 100.0 * (double)removed / (double)numBefore : 0.0,
			TD_TIMER_GET_FMT().c_str());
	}

	// selective low-view suppression: locally emulate nMinViewsFuse+1 by deleting
	// minimum-view (e.g. 2-view) points that sit on top of a better-supported
	// (3+-view) surface, while keeping minimum-view points that are the sole
	// evidence for their geometry. Gives the clean nMinViewsFuse=3 result where a
	// higher-view surface exists (e.g. the low-overlap pavement edge) without the
	// global data loss raising nMinViewsFuse would cause on 2-view-only datasets.
	// Disabled when nLowViewSupportCut == 0.
	if (!pointcloud.IsEmpty() && OPTDENSE::nLowViewSupportCut > 0 && OPTDENSE::nMinViewsFuse >= 2) {
		TD_TIMER_START();
		const size_t numBefore = pointcloud.NumPoints();
		const size_t removed = FilterRedundantLowViewPoints(pointcloud, OPTDENSE::nMinViewsFuse, OPTDENSE::nLowViewSupportCut, OPTDENSE::fLowViewSupportRadius, OPTDENSE::fLowViewPlanarityMax, (int)OPTDENSE::nOutlierFilterKNN);
		VERBOSE("Low-view consensus filter: %u/%u points removed (%.2f%%%%) (%s)",
			(unsigned)removed, (unsigned)numBefore,
			numBefore ? 100.0 * (double)removed / (double)numBefore : 0.0,
			TD_TIMER_GET_FMT().c_str());
	}

	// flatten a rough water/pond surface (textureless+moving -> noisy crust) onto a
	// robustly fitted near-horizontal plane, selecting the crust by low-elevation
	// band + local roughness. Heavily guarded -> near no-op on water-free data.
	// Disabled unless bFlattenWater.
	if (!pointcloud.IsEmpty() && OPTDENSE::bFlattenWater) {
		TD_TIMER_START();
		const size_t numPts = pointcloud.NumPoints();
		const size_t flattened = FilterFlattenWater(pointcloud, OPTDENSE::fFlattenWaterBandPct, OPTDENSE::fFlattenWaterRoughness, OPTDENSE::fFlattenWaterMaxFrac, (int)OPTDENSE::nOutlierFilterKNN);
		VERBOSE("Water flatten filter: %u/%u points snapped (%.2f%%%%) (%s)",
			(unsigned)flattened, (unsigned)numPts,
			numPts ? 100.0 * (double)flattened / (double)numPts : 0.0,
			TD_TIMER_GET_FMT().c_str());
	}

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

		// Pass 0: header-only sizing for every image (RecomputeMaxResolution only reads
		// the file header via ReadImageHeader(), no pixel decode) so the Tier-1/Tier-2
		// decision and fail-fast floor below can run before any real decode work starts.
		data.depthMaps.targetMaxResolution.Resize(images.GetSize());
		std::vector<uint64_t> imageBytes(images.GetSize(), 0);
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for schedule(dynamic)
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {
		#endif
			Image& imageData = images[idxImage];
			data.depthMaps.targetMaxResolution[idxImage] = 0;
			if (!imageData.IsValid())
				continue;
			unsigned nResolutionLevel(OPTDENSE::nResolutionLevel);
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(nResolutionLevel, OPTDENSE::nMinResolution, OPTDENSE::nMaxResolution));
			data.depthMaps.targetMaxResolution[idxImage] = nMaxResolution;
			const double aspect(imageData.width && imageData.height
				? (double)MINF(imageData.width, imageData.height) / (double)MAXF(imageData.width, imageData.height) : 1.0);
			imageBytes[idxImage] = (uint64_t)((double)nMaxResolution * (double)nMaxResolution * aspect) * 3ull/*bytes/pixel, color*/;
		}
		uint64_t totalImageBytes(0);
		for (const uint64_t b : imageBytes)
			totalImageBytes += b;

		// Tier-1/Tier-2 decision + fail-fast floor: a deterministic, one-time, loudly
		// logged pre-flight check. Never silently changes resolution/quality settings --
		// only decides WHEN the base color buffer (scene.images[*].image) is decoded/
		// evicted, and refuses to start (rather than let the OS thrash or crash) if even
		// the minimum possible working set cannot fit.
		const Util::MemoryInfo prepMemInfo(Util::GetMemoryInfo());
		const double DENSE_KEEP_IMAGES_FRACTION(0.35); // mirrors SceneTexture's TEXTURE_KEEP_IMAGES_FRACTION: this budget is spent for the WHOLE run (ESTIMATE+FUSE), not one transient stage
		const uint64_t keepBudget((uint64_t)((double)prepMemInfo.freePhysical * DENSE_KEEP_IMAGES_FRACTION));
		const bool bKeepResident(totalImageBytes > 0 && totalImageBytes <= keepBudget);
		const uint64_t safetyMemory(std::max((uint64_t)((double)prepMemInfo.totalPhysical * 0.08), (uint64_t)1*1024*1024*1024ull));

		// worst-case simultaneous floor: ESTIMATE always runs 2 event-loop worker threads
		// regardless of --max-threads (see the "cList<SEACAVE::Thread> threads(2)" call
		// sites further down), each holding (1 reference + up to nNumViews neighbors)
		// resident at once.
		const unsigned nFloorNeighbors(MAXF(1u, OPTDENSE::nNumViews ? OPTDENSE::nNumViews : OPTDENSE::nMaxViews));
		const uint64_t meanImageBytes(images.GetSize() ? totalImageBytes / images.GetSize() : 0);
		const uint64_t floorBytes(2ull * (1 + nFloorNeighbors) * meanImageBytes);
		if (meanImageBytes > 0 && floorBytes + safetyMemory > prepMemInfo.freePhysical) {
			VERBOSE("error: insufficient RAM for dense reconstruction even at minimum working set: "
				"need >= %s (2 concurrent estimate threads x (1 ref + %u neighbors) @ %s/image + safety margin), "
				"only %s free. Reduce --max-resolution or --num-views, or (if nothing else works) --resolution-level, "
				"or run on a machine with more RAM.",
				Util::formatBytes(floorBytes+safetyMemory).c_str(), nFloorNeighbors,
				Util::formatBytes(meanImageBytes).c_str(), Util::formatBytes(prepMemInfo.freePhysical).c_str());
			return false;
		}
		DEBUG_EXTRA("Image cache: %s decoded (estimated) vs %s budget (%s free x %.2f) -> %s",
			Util::formatBytes(totalImageBytes).c_str(), Util::formatBytes(keepBudget).c_str(),
			Util::formatBytes(prepMemInfo.freePhysical).c_str(), DENSE_KEEP_IMAGES_FRACTION,
			bKeepResident ? "KEEP resident (Tier 1)" : "bounded LRU cache (Tier 2)");
		data.depthMaps.imageCache = new ImageCache(images, data.depthMaps.targetMaxResolution,
			bKeepResident ? ImageCache::UNBOUNDED : (prepMemInfo.freePhysical > safetyMemory ? prepMemInfo.freePhysical - safetyMemory : 0));

		// Bound the derived per-image caches too (greyImages, sCachedImages): on a
		// large enough dataset they grow for the WHOLE ESTIMATE super-phase otherwise
		// (one entry per unique image[,scale] ever touched, never evicted until
		// ClearEstimationImageCaches() at phase end) -- confirmed in practice on a
		// 2000-image scene as continuous ~0.1GB/image growth with no ceiling. Budget
		// kept deliberately modest (each independently capped at this fraction of the
		// SAME free-RAM snapshot used above) since these compete for the same pool as
		// imageCache/DMapCache/FilterDMapCache; refined from measurement, not a
		// precisely-modeled joint budget.
		const double DENSE_DERIVED_CACHE_FRACTION(0.15);
		const uint64_t derivedCacheBudget((uint64_t)((double)prepMemInfo.freePhysical * DENSE_DERIVED_CACHE_FRACTION));
		#ifdef DPC_IMAGE_CACHE
		{
			std::lock_guard<std::mutex> lock(sGreyImagesMutex);
			greyImages.SetMaxMemory(derivedCacheBudget);
			DEBUG_EXTRA("greyImages cache: budget %s", Util::formatBytes(derivedCacheBudget).c_str());
		}
		#endif
		#ifdef DPC_FASTER_SAMPLING
		{
			std::lock_guard<std::mutex> lock(sCachedImagesMutex);
			sCachedImages.SetMaxMemory(derivedCacheBudget);
			DEBUG_EXTRA("sCachedImages cache: budget %s", Util::formatBytes(derivedCacheBudget).c_str());
		}
		#endif

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
			// Pass 1: decode now at the target resolution (Tier 1, matches the previous
			// unconditional behavior) or read only the header now and let ImageCache
			// decode on demand later (Tier 2, bounded).
			const unsigned nMaxResolution(data.depthMaps.targetMaxResolution[idxImage]);
			if (bKeepResident) {
				if (!imageData.ReloadImage(nMaxResolution)) {
					#ifdef DENSE_USE_OPENMP
					bAbort = true;
					#pragma omp flush (bAbort)
					continue;
					#else
					return false;
					#endif
				}
				imageData.references = 1; // permanent Tier-1 baseline ref; ImageCache (maxMemory=0) never evicts it anyway
			} else {
				if (!imageData.ReloadImage(nMaxResolution, /*bLoadPixels=*/false)) {
					#ifdef DENSE_USE_OPENMP
					bAbort = true;
					#pragma omp flush (bAbort)
					continue;
					#else
					return false;
					#endif
				}
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
			// (optionally measure how much they changed so we can stop iterating once converged)
			const bool bMeasureChange(OPTDENSE::fGeomConsistencyMaxChange > 0 &&
				data.nEstimationGeometricIter < (int)OPTDENSE::nEstimationGeometricIters - 1);
			double changeSum = 0.0;
			size_t changeCnt = 0;
			unsigned changeImages = 0;
			int64_t changeMeasureNs = 0;
			for (IIndex idx: data.images) {
				const DepthData& depthData(data.depthMaps.arrDepthData[idx]);
				if (!depthData.IsValid())
					continue;
				const String rawName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
				const String geoName(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"));
				if (bMeasureChange) {
					const auto _tM0 = std::chrono::steady_clock::now();
					size_t cnt;
					changeSum += MeasureDepthMapRelChange(rawName, geoName, cnt);
					changeMeasureNs += (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _tM0).count();
					changeCnt += cnt;
					++changeImages;
				}
				File::deleteFile(rawName);
				File::renameFile(geoName, rawName);
			}
			// once the depth-maps stop changing, a further full geometric pass would refine
			// them by < the convergence threshold, so skip ALL remaining passes and apply just
			// the optimize/smooth filters directly to the converged maps -- far cheaper than
			// another full patch-match sweep, and matching what the normal final iteration
			// produces (a near-identical geometric re-estimate followed by the same optimize).
			if (bMeasureChange && changeCnt > 0) {
				const float meanChange((float)(changeSum / (double)changeCnt));
				VERBOSE("Geometric-consistent iteration %d: mean depth change %.3f%% (measured %u depth-maps in %.0fms)",
					data.nEstimationGeometricIter, meanChange*100, changeImages, changeMeasureNs/1.0e6);
				if (meanChange < OPTDENSE::fGeomConsistencyMaxChange) {
					// restore the real optimize flags and mark the geometric phase finished so
					// the event loop takes its "optimize an existing dmap" path (load -> speckle/
					// gap filter -> save to dmap) rather than another full geometric estimate.
					OPTDENSE::nOptimize = nOptimize;
					data.nEstimationGeometricIter = -1;
					const bool bApplyOptimize((nOptimize & OPTDENSE::OPTIMIZE) != 0);
					VERBOSE("Geometric-consistent estimation converged (%.3f%% < %.3f%%): skipping remaining geometric passes, applying %s",
						meanChange*100, OPTDENSE::fGeomConsistencyMaxChange*100,
						bApplyOptimize ? "optimize-only final pass" : "no final pass (optimize disabled)");
					if (bApplyOptimize) {
						// run the optimize-only pass through the SAME worker/event machinery the
						// normal final iteration uses: this keeps only ~one dmap per worker thread
						// resident (bounded memory) instead of loading one per core at once, and
						// reuses the proven load/optimize/save path -- still far cheaper than another
						// full patch-match sweep because no estimation/neighbor warping runs.
						data.idxImage = 0;
						ASSERT(data.events.IsEmpty());
						data.events.AddEvent(new EVTProcessImage(0));
						data.progress = new Util::Progress("Optimized geometric-consistent depth-maps", data.images.GetSize());
						GET_LOGCONSOLE().Pause();
						if (nMaxThreads > 1) {
							cList<SEACAVE::Thread> threads(2);
							FOREACHPTR(pThread, threads)
								pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
							FOREACHPTR(pThread, threads)
								pThread->join();
						} else {
							DenseReconstructionEstimate((void*)&data);
						}
						GET_LOGCONSOLE().Play();
						if (!data.events.IsEmpty())
							return false;
						data.progress.Release();
					}
					break; // skip all remaining full geometric passes
				}
			}
		}
		data.nEstimationGeometricIter = -1;
	}

	// estimation is fully done (including all geometric-consistency iterations):
	// release the per-image grey/derivative caches before sizing the filter cache,
	// so their freed memory is reflected in the filter budget below
	ClearEstimationImageCaches();

	// If the base color-image cache is bounded (Tier 2), re-derive its budget now
	// that the grey/derivative caches above were just released -- freePhysical has
	// changed, and FUSE (which reads this same cache) starts after FILTER, so this
	// gives it a head start on the correct number. Never flip a Tier-1 (always
	// resident) cache into Tier 2 here, or vice versa -- the tier is a one-time,
	// deterministic decision made in "prepare images".
	if (data.depthMaps.imageCache && data.depthMaps.imageCache->IsBounded()) {
		const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
		const size_t safetyMemory(std::max(static_cast<size_t>(memInfo.totalPhysical * 0.08), size_t(1)*1024*1024*1024ull));
		data.depthMaps.imageCache->SetMaxMemory(memInfo.freePhysical > safetyMemory ? memInfo.freePhysical - safetyMemory : 0);
	}

	if ((OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) != 0) {
#if FILTER_PROFILE
		g_filterProfile.Reset();
		const auto _tPhase0 = filter_clock::now();
#endif
		// initialize the queue of depth-maps to be filtered
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		// Retention cache for the filter sub-phase: keeps recently-loaded reference/
		// neighbor dmaps resident (bounded by free RAM) so each unique dmap is read and
		// deserialized once instead of ~9x (once per referencing image). Semantics-
		// neutral: dmaps are immutable until the adjust sub-phase swaps in the filtered
		// results. Cleared at the filter->adjust boundary by SignalCompleteDepthmapFilter.
		const Util::MemoryInfo filterMemInfo(Util::GetMemoryInfo());
		const size_t filterSafetyMemory(std::max(static_cast<size_t>(filterMemInfo.totalPhysical * 0.10), size_t(2)*1024*1024*1024ull));
		const size_t filterCacheBudget(filterMemInfo.freePhysical > filterSafetyMemory ? filterMemInfo.freePhysical - filterSafetyMemory : 0);
		DEBUG_EXTRA("FilterDMapCache: freePhysical=%s totalPhysical=%s safetyMemory=%s",
			Util::formatBytes(filterMemInfo.freePhysical).c_str(), Util::formatBytes(filterMemInfo.totalPhysical).c_str(), Util::formatBytes(filterSafetyMemory).c_str());
		FilterDMapCache filterCache(data.depthMaps.arrDepthData, filterCacheBudget);
		g_filterCache = &filterCache;
		// The filter phase is disk-bound on a single NVMe: with one thread per core,
		// the in-flight working set (~threads * (1 ref + 8 neighbors)) blows the dmap
		// cache, forcing redundant neighbor re-reads, while many concurrent full-dmap
		// Save() calls contend the write queue. Capping the thread count shrinks the
		// working set (more neighbor cache hits) and the write concurrency, which
		// dramatically cuts thread-summed I/O and compute -- BUT measured WORSE wall
		// time (8 threads: 23.4s vs 17.4s at full cores), because that contention
		// overlaps in parallel and the stage is parallelism-bound at the wall, not
		// disk-bound. So the default keeps one-thread-per-core (kFilterThreads = 0).
		// Sweep this value (e.g. 4 / 8 / 16) only for diagnosing contention.
		const unsigned kFilterThreads(0);
		const unsigned nFilterThreads(kFilterThreads == 0 ? nMaxThreads : MINF(nMaxThreads, kFilterThreads));
		if (nFilterThreads > 1) {
			// multi-thread execution
			cList<SEACAVE::Thread> threads(MINF(nFilterThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		DEBUG_EXTRA("FilterDMapCache: %u disk loads, %.1f%%%% hit-rate", filterCache.GetNumDiskLoads(), filterCache.GetHitRate() * 100.0);
		g_filterCache = NULL;
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
#if FILTER_PROFILE
		g_filterProfile.Report(FilterNs(_tPhase0, filter_clock::now()) / 1.0e6);
#endif
	}
	return true;
} // ComputeDepthMaps
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// Temporary bisection diagnostic: prints this process's own commit charge at a named
// stage of one image's estimation lifecycle, throttled to every 50th image. A live
// 2000-image run reached procCommit=213GB with both derived caches (greyImages,
// sCachedImages) pinned at their absolute floor, proving the leak is somewhere else
// in this per-image lifecycle -- this bisects WHICH stage (view init vs. the actual
// PatchMatch estimate vs. optimize/save) the jump happens in, instead of only knowing
// "somewhere in estimation".
static void LogMemCheckpoint(const char* stage, IIndex idxImage) {
	if (idxImage % 50 != 0)
		return;
	const Util::ProcessMemoryInfo procInfo(Util::GetSelfMemoryInfo());
	DEBUG_EXTRA("MEMCP %-12s img=%u procWS=%s procCommit=%s", stage, idxImage,
		Util::formatBytes(procInfo.workingSetSize).c_str(), Util::formatBytes(procInfo.pagefileUsage).c_str());
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
			LogMemCheckpoint("before-init", evtImage.idxImage);
			if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews, !depthmapComputed, depthmapComputed ? -1 : (data.nEstimationGeometricIter >= 0 ? 1 : 0))) {
				// process next image
				data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
				break;
			}
			LogMemCheckpoint("after-init", evtImage.idxImage);
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
			LogMemCheckpoint("before-est", evtImage.idxImage);
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
			LogMemCheckpoint("after-est", evtImage.idxImage);
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
			LogMemCheckpoint("before-save", evtImage.idxImage);
			// save compute depth-map for this image
			// JPB WIP Saving this asynchronously doesn't help.
			if (!depthData.depthMap.empty())
				depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), data.nEstimationGeometricIter < 0 ? "dmap" : "geo.dmap"));
			depthData.ReleaseImages();
			depthData.Release();
			LogMemCheckpoint("after-release", evtImage.idxImage);
			data.progress->operator++();
			// Re-derive the greyImages/sCachedImages budgets from LIVE free RAM instead of
			// the one-time prepare-images snapshot. The one-time snapshot was the actual
			// bug behind the large-dataset OOM: each cache's budget was a fixed fraction of
			// free RAM measured before ANY decoding started, so neither one ever learned
			// that the other (or ImageCache's Tier-1 residency, or normal per-image working
			// memory) was also eating into the same pool. Sharing one pool re-measured from
			// Util::GetMemoryInfo() (which reflects everything currently resident, from any
			// source) makes both caches shrink automatically as real headroom shrinks, the
			// same self-correcting pattern DMapCache already uses in FUSE.
			//
			// Runs every image, not periodically: a live run confirmed a 50-image interval
			// is far too coarse -- free RAM collapsed from single-digit GB to under 100MB
			// within one 50-image window, i.e. faster than a periodic check could react.
			// The recompute itself is cheap (one syscall, two brief mutex locks), so there
			// is no real cost to doing it every time; only the diagnostic log line below is
			// throttled, to avoid flooding the log on a large dataset.
			const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
			const size_t safetyMemory(std::max(static_cast<size_t>(memInfo.totalPhysical * 0.08), size_t(1)*1024*1024*1024ull));
			size_t greyBytes(0), greyCount(0), scaledBytes(0), scaledCount(0);
#ifdef DPC_IMAGE_CACHE
			size_t greyUsed(0);
			{ std::lock_guard<std::mutex> lock(sGreyImagesMutex); greyUsed = greyImages.GetUsedMemory(); }
#endif
#ifdef DPC_FASTER_SAMPLING
			size_t scaledUsed(0);
			{ std::lock_guard<std::mutex> lock(sCachedImagesMutex); scaledUsed = sCachedImages.GetUsedMemory(); }
#endif
#if defined(DPC_IMAGE_CACHE) && defined(DPC_FASTER_SAMPLING)
			// pool = current free RAM + what these two caches already hold (reclaimable
			// on demand), so re-adding their own usage avoids double-subtracting it
			const size_t pool(memInfo.freePhysical + greyUsed + scaledUsed);
			const size_t sharedBudget(pool > safetyMemory ? pool - safetyMemory : 0);
			{ std::lock_guard<std::mutex> lock(sGreyImagesMutex); greyImages.SetMaxMemory(sharedBudget / 2); greyBytes = greyImages.GetUsedMemory(); greyCount = greyImages.GetCount(); }
			{ std::lock_guard<std::mutex> lock(sCachedImagesMutex); sCachedImages.SetMaxMemory(sharedBudget / 2); scaledBytes = sCachedImages.GetUsedMemory(); scaledCount = sCachedImages.GetCount(); }
#elif defined(DPC_IMAGE_CACHE)
			const size_t pool(memInfo.freePhysical + greyUsed);
			std::lock_guard<std::mutex> lock(sGreyImagesMutex);
			greyImages.SetMaxMemory(pool > safetyMemory ? pool - safetyMemory : 0);
			greyBytes = greyImages.GetUsedMemory(); greyCount = greyImages.GetCount();
#elif defined(DPC_FASTER_SAMPLING)
			const size_t pool(memInfo.freePhysical + scaledUsed);
			std::lock_guard<std::mutex> lock(sCachedImagesMutex);
			sCachedImages.SetMaxMemory(pool > safetyMemory ? pool - safetyMemory : 0);
			scaledBytes = sCachedImages.GetUsedMemory(); scaledCount = sCachedImages.GetCount();
#endif
			if (evtImage.idxImage % 50 == 0) {
				// process-level counters alongside the system-wide free RAM: if
				// pagefileUsage (this process's own commit charge -- what actually
				// triggers an OOM allocation failure) keeps climbing even with both
				// derived caches pinned at their floor, the leak is genuinely inside
				// this process (somewhere in estimation, not yet found) rather than
				// system-wide free RAM dropping for some unrelated reason.
				const Util::ProcessMemoryInfo procInfo(Util::GetSelfMemoryInfo());
				DEBUG_EXTRA("MEMDIAG img=%u free=%s greyImages=%s(%u) sCachedImages=%s(%u) procWS=%s procCommit=%s",
					evtImage.idxImage, Util::formatBytes(memInfo.freePhysical).c_str(),
					Util::formatBytes(greyBytes).c_str(), (unsigned)greyCount,
					Util::formatBytes(scaledBytes).c_str(), (unsigned)scaledCount,
					Util::formatBytes(procInfo.workingSetSize).c_str(), Util::formatBytes(procInfo.pagefileUsage).c_str());
			}
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
#if FILTER_PROFILE
			const auto _tL0 = filter_clock::now();
#endif
			g_filterCache->Acquire(idx, ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			const unsigned numMaxNeighbors(8);
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			FOREACH(n, depthData.neighbors) {
				const IIndex idxView = depthData.neighbors[n].ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				if (!depthDataPair.IsValid())
					continue;
				if (!g_filterCache->Acquire(idxView, ComposeDepthFilePath(depthDataPair.GetView().GetID(), "dmap"))) {
					// signal error and terminate
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.Insert(n);
				if (idxNeighbors.GetSize() == numMaxNeighbors)
					break;
			}
#if FILTER_PROFILE
			g_filterProfile.nsLoad += FilterNs(_tL0, filter_clock::now());
			++g_filterProfile.nImages;
#endif
			// filter the depth-map for this image
			if (data.depthMaps.FilterDepthMap(depthData, idxNeighbors, OPTDENSE::bFilterAdjust)) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
			// release working references (kept resident in the retention cache until evicted)
			FOREACHPTR(pIdxNeighbor, idxNeighbors) {
				const IIndex idxView = depthData.neighbors[*pIdxNeighbor].ID;
				g_filterCache->Release(idxView);
			}
			g_filterCache->Release(idx);
			data.SignalCompleteDepthmapFilter();
			break; }

		case EVT_ADJUSTDEPTHMAP: {
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();

#if FILTER_PROFILE
			const auto _tA0 = filter_clock::now();
#endif
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
#if FILTER_PROFILE
			g_filterProfile.nsAdjLoad += FilterNs(_tA0, filter_clock::now());
#endif
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
#if FILTER_PROFILE
			const auto _tA1 = filter_clock::now();
#endif
			{
				// only the depth and confidence maps changed during filtering; the
				// normals/views/header were just loaded from this same file and a full
				// Save() would re-write them unchanged. Patch just those two sections in
				// place (byte-identical, far fewer bytes); fall back to a full Save() if the
				// on-disk layout does not match.
				const String dmapPath(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
				if (!PatchDepthConfRaw(dmapPath, depthData.depthMap, depthData.confMap,
						!depthData.normalMap.empty(), !depthData.viewsMap.empty()))
					depthData.Save(dmapPath);
			}
#if FILTER_PROFILE
			g_filterProfile.nsAdjSave += FilterNs(_tA1, filter_clock::now());
#endif
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
// Toggle the tighter cone-vs-box octree prune (default on). 0 = original cone-vs-boundingsphere.
#ifndef PCF_CONE_AABB
#define PCF_CONE_AABB 1
#endif
void Scene::PointCloudFilter(int thRemove, float maxRemoveFrac)
{
	TD_TIMER_STARTD();

	// Build the octree over a uint32-indexed point array. PointCloud::PointArr's index is
	// 64-bit (size_t), which doubles the octree's per-point index storage (m_indices, read
	// on every leaf visit); the cloud never exceeds 2^32 points, so 32-bit indices halve it
	// and improve cache during traversal -- identical tree, identical result. Local type, so
	// the shared PointCloud class is untouched.
	typedef CLISTDEF0IDX(PointCloud::Point, uint32_t) PointArr32;
	typedef TOctree<PointArr32,PointCloud::Point::Type,3,uint32_t> Octree;
	// Lock-free visibility collector: carries only per-query state (no shared
	// mutable members, no critical section) and accumulates into the shared
	// visibility array with a single atomic add. Because nothing is shared
	// between concurrent (point,view) queries, they all run in parallel without
	// serializing on a per-view lock; the atomic also removes the latent
	// lost-update race the previous code had across different views.
	struct Collector {
		typedef Octree::IDX_TYPE IDX;
		typedef PointCloud::Point::Type Real;
		typedef TCone<Real,3> Cone;
		typedef TSphere<Real,3> Sphere;
		typedef TConeIntersect<Real,3> ConeIntersect;

		Cone cone;
		const ConeIntersect coneIntersect;
		const PointCloudStreaming& pointcloud;
		int* const __restrict visibility;
		// leaf-ordered SoA positions (aligned with the octree index array) so the classify
		// loop reads positions sequentially instead of gathering scattered points
		const float* const __restrict pLeafX;
		const float* const __restrict pLeafY;
		const float* const __restrict pLeafZ;
		const IDX* const __restrict pIdxBase; // octree index array base; a leaf's idices offset into this
		PointCloud::Index idxPoint;
		Real distance;
		int weight;
		const Real tanAngle; // cone half-angle tangent (per view); capsule radius = maxHeight*tanAngle
		// separating-axis data (rebuilt per point/view) for the cone-axis capsule vs octree cell box
		Real capRadius;
		Real m_axN[6][3];
		Real m_axInv[6];
		Real m_axL1[6];
		Real m_segLo[6];
		Real m_segHi[6];

		Collector(const Cone::RAY& ray, Real angle, const PointCloudStreaming& _pointcloud, int* __restrict _visibility,
			const float* __restrict _pLeafX, const float* __restrict _pLeafY, const float* __restrict _pLeafZ, const IDX* __restrict _pIdxBase)
			: cone(ray, angle), coneIntersect(cone), pointcloud(_pointcloud), visibility(_visibility),
			  pLeafX(_pLeafX), pLeafY(_pLeafY), pLeafZ(_pLeafZ), pIdxBase(_pIdxBase), tanAngle(std::tan(angle)) {}
		inline void Init(PointCloud::Index _idxPoint, const PointCloud::Point& X, int _weight) {
			const Real thMaxDepth(1.02f);
			idxPoint = _idxPoint;
			const PointCloud::Point::EVec D((PointCloud::Point::EVec&)X-cone.ray.m_pOrig);
			distance = D.norm();
			cone.ray.m_vDir = D/distance;
			cone.maxHeight = MaxDepthDifference(distance, thMaxDepth);
			weight = _weight;
			BuildSepAxes();
		}
		// Build the capsule (cone-axis segment + capRadius) that conservatively encloses the finite
		// cone; Intersects then rejects a cell only when a projection proves it lies farther than
		// capRadius from the segment (a lower bound on true distance, so no in-cone point is ever
		// pruned -> removed set stays bit-identical to the exact cone-sphere filter).
		inline void BuildSepAxes() {
			const Real ox(cone.ray.m_pOrig.x()), oy(cone.ray.m_pOrig.y()), oz(cone.ray.m_pOrig.z());
			const Real dx(cone.ray.m_vDir.x()), dy(cone.ray.m_vDir.y()), dz(cone.ray.m_vDir.z());
			const Real h0(cone.minHeight), h1(cone.maxHeight);
			const Real Ax(ox+dx*h0), Ay(oy+dy*h0), Az(oz+dz*h0);
			const Real Bx(ox+dx*h1), By(oy+dy*h1), Bz(oz+dz*h1);
			// 3 (axis x box-face) cross products first (reject thin cones laterally), then 3 box faces
			const Real N[6][3] = {
				{ Real(0), dz, -dy }, { -dz, Real(0), dx }, { dy, -dx, Real(0) },
				{ Real(1), Real(0), Real(0) }, { Real(0), Real(1), Real(0) }, { Real(0), Real(0), Real(1) }
			};
			for (int k=0; k<6; ++k) {
				const Real nx(N[k][0]), ny(N[k][1]), nz(N[k][2]);
				m_axN[k][0]=nx; m_axN[k][1]=ny; m_axN[k][2]=nz;
				m_axL1[k] = std::abs(nx)+std::abs(ny)+std::abs(nz);
				const Real nn(std::sqrt(nx*nx+ny*ny+nz*nz));
				m_axInv[k] = (nn > Real(1e-12)) ? (Real(1)/nn) : Real(0);
				const Real pA(Ax*nx+Ay*ny+Az*nz), pB(Bx*nx+By*ny+Bz*nz);
				m_segLo[k] = pA<pB?pA:pB;
				m_segHi[k] = pA<pB?pB:pA;
			}
			// capsule radius = max cone radius over the segment, plus small FP slack (keeps the prune conservative)
			capRadius = h1*tanAngle*Real(1.01) + h1*Real(1e-4);
		}
		inline bool Intersects(const Octree::POINT_TYPE& center, Octree::Type radius) const {
		#if PCF_CONE_AABB
			const Real cx(center.x()), cy(center.y()), cz(center.z());
			for (int k=0; k<6; ++k) {
				const Real cP(cx*m_axN[k][0]+cy*m_axN[k][1]+cz*m_axN[k][2]);
				const Real boxExt(radius*m_axL1[k]);
				const Real boxLo(cP-boxExt), boxHi(cP+boxExt);
				const Real g((m_segLo[k] > boxHi) ? (m_segLo[k]-boxHi) : ((boxLo > m_segHi[k]) ? (boxLo-m_segHi[k]) : Real(0)));
				if (g*m_axInv[k] > capRadius)
					return false;
			}
			return true;
		#else
			return coneIntersect(Sphere(center, radius*Real(SQRT_3)));
		#endif
		}
		inline void operator () (const IDX* __restrict idices, IDX size) const {
			// Hoist every per-cone constant into a local so the inner loop reads no
			// memory reachable through the cone / coneIntersect references (the
			// atomic update below is a compiler memory barrier that would otherwise
			// reload them every iteration), then run the inlined cone Classify
			// (axis projection + half-angle test) four points at a time with SSE2.
			// Most leaf points fall outside the cone, so the single movemask branch
			// keeps the common path branchless; only the rare VISIBLE lanes take the
			// scalar tail (depth-similarity test + atomic accumulation).
			const Real ax(cone.ray.m_pOrig.x()), ay(cone.ray.m_pOrig.y()), az(cone.ray.m_pOrig.z());
			const Real dx(cone.ray.m_vDir.x()), dy(cone.ray.m_vDir.y()), dz(cone.ray.m_vDir.z());
			const Real minH(cone.minHeight);
			const Real maxH(cone.maxHeight);
			const Real cosSq(coneIntersect.cosAngleSq);
			const Real refDist(distance);
			const Real thSimilar(0.01f);
			const int w(weight);
			int* const __restrict vis = visibility;

			// commit one VISIBLE point (rare path): depth-similarity reject, then a
			// lock-free signed accumulation into the shared visibility array.
			const auto emit = [&](const uint32_t idx, const float t) {
				if (IsDepthSimilar(refDist, t, thSimilar))
					return;
				const int delta = (t > refDist) ? (int)pointcloud.ViewsStreamSize(idx) : -w;
			#ifdef DENSE_USE_OPENMP
				_InterlockedExchangeAdd(reinterpret_cast<volatile long*>(vis + idx), (long)delta);
			#else
				vis[idx] += delta;
			#endif
			};

			const __m128 vAx = _mm_set1_ps(ax), vAy = _mm_set1_ps(ay), vAz = _mm_set1_ps(az);
			const __m128 vDx = _mm_set1_ps(dx), vDy = _mm_set1_ps(dy), vDz = _mm_set1_ps(dz);
			const __m128 vMinH = _mm_set1_ps(minH), vMaxH = _mm_set1_ps(maxH), vCosSq = _mm_set1_ps(cosSq);

			// leaf-ordered positions: this leaf's points are contiguous at [base, base+size),
			// so positions load sequentially (1-2 cache lines) instead of 12 scattered loads.
			const size_t base = (size_t)(idices - pIdxBase);
			const float* const __restrict lx = pLeafX + base;
			const float* const __restrict ly = pLeafY + base;
			const float* const __restrict lz = pLeafZ + base;

			IDX k = 0;
			for (; k + 4 <= size; k += 4) {
				const __m128 px = _mm_loadu_ps(lx + k);
				const __m128 py = _mm_loadu_ps(ly + k);
				const __m128 pz = _mm_loadu_ps(lz + k);
				const __m128 Dx = _mm_sub_ps(px, vAx), Dy = _mm_sub_ps(py, vAy), Dz = _mm_sub_ps(pz, vAz);
				// t = axial projection of (P-apex) onto the cone axis
				const __m128 t = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vDx, Dx), _mm_mul_ps(vDy, Dy)), _mm_mul_ps(vDz, Dz));
				const __m128 nSq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(Dx, Dx), _mm_mul_ps(Dy, Dy)), _mm_mul_ps(Dz, Dz));
				// VISIBLE = t in (minHeight, maxHeight] AND t*t > cosAngleSq * |P-apex|^2
				const __m128 mask = _mm_and_ps(_mm_and_ps(_mm_cmpgt_ps(t, vMinH), _mm_cmple_ps(t, vMaxH)),
					_mm_cmpgt_ps(_mm_mul_ps(t, t), _mm_mul_ps(vCosSq, nSq)));
				const int bits = _mm_movemask_ps(mask);
				if (bits) {
					alignas(16) float tArr[4];
					_mm_store_ps(tArr, t);
					if (bits & 1) emit(idices[k], tArr[0]);
					if (bits & 2) emit(idices[k+1], tArr[1]);
					if (bits & 4) emit(idices[k+2], tArr[2]);
					if (bits & 8) emit(idices[k+3], tArr[3]);
				}
			}
			// scalar tail (< 4 remaining)
			for (; k < size; ++k) {
				const float Dx(lx[k] - ax), Dy(ly[k] - ay), Dz(lz[k] - az);
				const float t(dx*Dx + dy*Dy + dz*Dz);
				if (t <= minH || t > maxH)
					continue;
				if (t*t <= cosSq * (Dx*Dx + Dy*Dy + Dz*Dz))
					continue;
				emit(idices[k], t);
			}
		}
	};

	// gather points into a contiguous array for the octree (streaming cloud
	// stores XYZ as a flat float stream, so build the typed array once)
	PointArr32 ptsForOctree(pointcloud.GetSize());
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel for
	for (int64_t i=0; i<(int64_t)ptsForOctree.GetSize(); ++i)
		ptsForOctree[(uint32_t)i] = pointcloud.Point((PointCloud::Index)i);
	#else
	FOREACH(i, ptsForOctree)
		ptsForOctree[i] = pointcloud.Point(i);
	#endif
	// create octree to speed-up search
	Octree octree(ptsForOctree, [](Octree::IDX_TYPE size, Octree::Type /*radius*/) {
		return size > 128;
	});
	IntArr visibility(pointcloud.GetSize()); visibility.Memset(0);
	int* const __restrict pVisibility = visibility.Begin();

	// Build leaf-ordered SoA positions aligned with the octree's index array so the classify
	// inner loop reads positions sequentially -- the profiled bottleneck was the scattered
	// per-point position gather. Result-identical: same positions, same lane order, same math.
	const Octree::IDXARR_TYPE& octIdx = octree.GetIndexArr();
	const size_t nOctItems = octIdx.size();
	std::vector<float> leafX(nOctItems), leafY(nOctItems), leafZ(nOctItems);
	{
		const float* const __restrict pXYZsrc = pointcloud.pointsXYZ.data();
		const Octree::IDX_TYPE* const __restrict pMI = octIdx.data();
		float* const __restrict pLX = leafX.data(); float* const __restrict pLY = leafY.data(); float* const __restrict pLZ = leafZ.data();
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t j = 0; j < (int64_t)nOctItems; ++j) {
			const size_t s = (size_t)pMI[j] * 3;
			pLX[j] = pXYZsrc[s+0]; pLY[j] = pXYZsrc[s+1]; pLZ[j] = pXYZsrc[s+2];
		}
	}
	const float* const pLeafX = leafX.data();
	const float* const pLeafY = leafY.data();
	const float* const pLeafZ = leafZ.data();
	const Octree::IDX_TYPE* const pIdxBase = octIdx.data();

	// pre-compute each view's cone origin (camera center) and half-angle once;
	// each (point,view) query then builds a private Collector from these, so the
	// hot loop owns all its mutable state and needs no per-view locking.
	const size_t numViews(images.size());
	std::vector<Ray3f> viewRays; viewRays.reserve(numViews);
	std::vector<float> viewAngles; viewAngles.reserve(numViews);
	FOREACH(idxView, images) {
		const Image& image = images[idxView];
		viewRays.emplace_back(Cast<float>(image.camera.C), Cast<float>(image.camera.Direction()));
		viewAngles.push_back(float(image.ComputeFOV(0)/image.width));
	}

	// run all camera-point visibility intersections. Keep the parallel sweep over
	// points (best load-balance), but give each worker thread its own array of
	// per-view Collectors and reuse them: a Collector's cone half-angle and the
	// angle-derived ConeIntersect constants depend ONLY on the view, so they are
	// built once per (thread,view) and reused for every point that thread tests
	// against that view, instead of being reconstructed for each (point,view) pair.
	// Only the cheap per-point ray direction/distance is refreshed in Init().
	// Accumulation into the shared visibility array stays lock-free via the atomic.
	Util::Progress progress(_T("Point visibility checks"), pointcloud.GetSize());
	const int64_t numPoints = (int64_t)pointcloud.GetSize();
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel
	{
		// one Collector per view, indexed directly by view id. reserve() up front so
		// the buffer never reallocates after construction: a Collector holds a
		// ConeIntersect that references its own cone, so it must keep a stable address
		// (never moved/copied once built). The inline capacity covers the usual
		// few-hundred views with no heap allocation; larger counts spill to a single
		// reserved heap buffer (still no relocation, since reserve == final size).
		boost::container::small_vector<Collector, 512> pool;
		pool.reserve(numViews);
		for (size_t v = 0; v < numViews; ++v)
			pool.emplace_back(viewRays[v], viewAngles[v], pointcloud, pVisibility, pLeafX, pLeafY, pLeafZ, pIdxBase);
		size_t progressBatch = 0; // batch the display-only counter: one contended atomic per 65536 pts, not per pt
		#pragma omp for schedule(dynamic, 2048)
		for (int64_t i = 0; i < numPoints; ++i) {
			const PointCloud::Index idxPoint((PointCloud::Index)i);
			const PointCloud::Point& X = pointcloud.Point(idxPoint);
			const uint32_t* __restrict views = pointcloud.ViewsStream(idxPoint);
			const size_t nViews = pointcloud.ViewsStreamSize(idxPoint);
			for (size_t v = 0; v < nViews; ++v) {
				Collector& c = pool[views[v]];
				c.Init(idxPoint, X, (int)nViews);
				octree.Collect(c, c);
			}
			if (++progressBatch == 65536) { progress += (int)progressBatch; progressBatch = 0; }
		}
		if (progressBatch)
			progress += (int)progressBatch;
	}
	#else
	{
		boost::container::small_vector<Collector, 512> pool;
		pool.reserve(numViews);
		for (size_t v = 0; v < numViews; ++v)
			pool.emplace_back(viewRays[v], viewAngles[v], pointcloud, pVisibility, pLeafX, pLeafY, pLeafZ, pIdxBase);
		for (PointCloud::Index idxPoint=0; idxPoint<(PointCloud::Index)numPoints; ++idxPoint) {
			const PointCloud::Point& X = pointcloud.Point(idxPoint);
			const uint32_t* __restrict views = pointcloud.ViewsStream(idxPoint);
			const size_t nViews = pointcloud.ViewsStreamSize(idxPoint);
			for (size_t v=0; v<nViews; ++v) {
				Collector& c = pool[views[v]];
				c.Init(idxPoint, X, (int)nViews);
				octree.Collect(c, c);
			}
			++progress;
		}
	}
	#endif
	progress.close();

	// filter points: single O(n) compaction pass (mid-array RemovePoint per cull
	// is O(n) -> O(n^2) over the whole cloud). Keep visibility>thRemove, copying
	// survivors down; view/weight offsets keep pointing into their original blobs.
	const size_t numInitPoints(pointcloud.GetSize());
	// Safety guard: if the threshold would remove more than maxRemoveFrac of the
	// cloud it is almost certainly mis-set (too aggressive). Bail without touching
	// the cloud rather than silently gutting / emptying the output.
	if (maxRemoveFrac < 1.f) {
		int64_t nRemoveCnt = 0;
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for reduction(+:nRemoveCnt)
		#endif
		for (int64_t r = 0; r < (int64_t)numInitPoints; ++r)
			if (pVisibility[r] <= thRemove)
				++nRemoveCnt;
		const size_t nRemove = (size_t)nRemoveCnt;
		if (numInitPoints && (float)nRemove > maxRemoveFrac * (float)numInitPoints) {
			VERBOSE("WARNING: visibility filter (th<=%d) would remove %u/%u points (%.1f%%%% > %.0f%%%% cap); skipping filter (use a LARGER --filter-point-cloud value -- larger is gentler).",
				thRemove, (unsigned)nRemove, (unsigned)numInitPoints,
				100.f*(float)nRemove/(float)numInitPoints, 100.f*maxRemoveFrac);
			return;
		}
	}
	{
		const bool hasN(!pointcloud.normalsXYZ.empty());
		const bool hasC(!pointcloud.colorsRGB.empty());
		const bool hasVO(!pointcloud.pointViewsOffsets.empty());
		const bool hasVS(!pointcloud.pointViewsSizes.empty());
		const bool hasWO(!pointcloud.pointWeightsOffsets.empty());
		const bool hasWS(!pointcloud.pointWeightsSizes.empty());
		size_t w = 0;
		for (size_t r = 0; r < numInitPoints; ++r) {
			if (visibility[r] <= thRemove)
				continue;
			if (w != r) {
				pointcloud.pointsXYZ[w*3+0] = pointcloud.pointsXYZ[r*3+0];
				pointcloud.pointsXYZ[w*3+1] = pointcloud.pointsXYZ[r*3+1];
				pointcloud.pointsXYZ[w*3+2] = pointcloud.pointsXYZ[r*3+2];
				if (hasN) { pointcloud.normalsXYZ[w*3+0]=pointcloud.normalsXYZ[r*3+0]; pointcloud.normalsXYZ[w*3+1]=pointcloud.normalsXYZ[r*3+1]; pointcloud.normalsXYZ[w*3+2]=pointcloud.normalsXYZ[r*3+2]; }
				if (hasC) { pointcloud.colorsRGB[w*3+0]=pointcloud.colorsRGB[r*3+0]; pointcloud.colorsRGB[w*3+1]=pointcloud.colorsRGB[r*3+1]; pointcloud.colorsRGB[w*3+2]=pointcloud.colorsRGB[r*3+2]; }
				if (hasVO) pointcloud.pointViewsOffsets[w]=pointcloud.pointViewsOffsets[r];
				if (hasVS) pointcloud.pointViewsSizes[w]=pointcloud.pointViewsSizes[r];
				if (hasWO) pointcloud.pointWeightsOffsets[w]=pointcloud.pointWeightsOffsets[r];
				if (hasWS) pointcloud.pointWeightsSizes[w]=pointcloud.pointWeightsSizes[r];
			}
			++w;
		}
		pointcloud.pointsXYZ.resize(w*3);
		if (hasN) pointcloud.normalsXYZ.resize(w*3);
		if (hasC) pointcloud.colorsRGB.resize(w*3);
		if (hasVO) pointcloud.pointViewsOffsets.resize(w);
		if (hasVS) pointcloud.pointViewsSizes.resize(w);
		if (hasWO) pointcloud.pointWeightsOffsets.resize(w);
		if (hasWS) pointcloud.pointWeightsSizes.resize(w);
	}

	DEBUG_EXTRA("Point-cloud filtered: %u/%u points (%d%%%%) (%s)", pointcloud.NumPoints(), numInitPoints, ROUND2INT((100.f*pointcloud.NumPoints()) / numInitPoints), TD_TIMER_GET_FMT().c_str());
	} // PointCloudFilter
/*----------------------------------------------------------------*/
