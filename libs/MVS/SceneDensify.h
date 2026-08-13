/*
* SceneDensify.h
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

#ifndef _MVS_SCENEDENSIFY_H_
#define _MVS_SCENEDENSIFY_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "SemiGlobalMatcher.h"


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {
	
// Forward declarations
class MVS_API Scene;
struct ImageCache; // bounded on-demand cache for the base decoded color images (see SceneDensify.cpp)
#ifdef _USE_CUDA
namespace CUDA {
class PatchMatch;
} // namespace CUDA
#endif // _USE_CUDA

// structure used to compute all depth-maps
class MVS_API DepthMapsData
{
public:
	DepthMapsData(Scene& _scene);
	~DepthMapsData();

	bool SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap);
	bool SelectViews(DepthData& depthData);
	bool InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors, bool loadImages, int loadDepthMaps);
	bool InitDepthMap(DepthData& depthData);
	bool EstimateDepthMap(IIndex idxImage, int nGeometricIter);

	bool RemoveSmallSegments(DepthData& depthData);
	bool GapInterpolation(DepthData& depthData);

	bool FilterDepthMap(DepthData& depthData, const IIndexArr& idxNeighbors, bool bAdjust=true);
	void MergeDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool bEstimateNormal);
	void FuseDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool bEstimateNormal);
	void DenseFuseDepthMaps(PointCloudStreaming& pointcloud, bool bEstimateColor, bool bEstimateNormal);

	static DepthData ScaleDepthData(const DepthData& inputDeptData, float scale);

protected:
#ifdef DPC_EXTENDED_OMP_THREADING
	static void* STCALL ScoreDepthMapTmp(cList<DepthEstimator>& estimators);
	static void* STCALL EstimateDepthMapTmp(cList<DepthEstimator>& estimators);
	static void* STCALL EndDepthMapTmp(cList<DepthEstimator>& estimators);
#else
	static void* STCALL EstimateDepthMapTmp(void*);
#ifdef DPC_EXTENDED_OMP_THREADING2
	static void* STCALL ScoreDepthMapTmp(cList<DepthEstimator>& estimators);
	static void* STCALL EndDepthMapTmp(cList<DepthEstimator>& estimators);
#else
	static void* STCALL ScoreDepthMapTmp(void*);
	static void* STCALL EndDepthMapTmp(void*);
#endif
#endif

public:
	Scene& scene;

	DepthDataArr arrDepthData;

	// bounded on-demand cache for scene.images[*].image (the base decoded color
	// buffers); spans ESTIMATE and FUSE, both of which read from it. targetMaxResolution
	// is the per-image decode target computed once in the "prepare images" step and
	// referenced by imageCache for the lifetime of the run; declared before imageCache
	// so it is already constructed when imageCache's constructor binds a reference to it.
	cList<unsigned> targetMaxResolution;
	CAutoPtr<ImageCache> imageCache;

	// used internally to estimate the depth-maps
	Image8U::Size prevDepthMapSize; // remember the size of the last estimated depth-map
	Image8U::Size prevDepthMapSizeTrg; // ... same for target image
	DepthEstimator::MapRefArr coords; // map pixel index to zigzag matrix coordinates
	DepthEstimator::MapRefArr coordsTrg; // ... same for target image

	#ifdef _USE_CUDA
	// used internally to estimate the depth-maps using CUDA
	CAutoPtr<MVS::CUDA::PatchMatch> pmCUDA;
	#endif // _USE_CUDA
};
/*----------------------------------------------------------------*/

struct MVS_API DenseDepthMapData {
	Scene& scene;
	IIndexArr images;
	IIndexArr neighborsMap;
	DepthMapsData depthMaps;
	volatile Thread::safe_t idxImage;
	SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
	Semaphore sem;
	CAutoPtr<Util::Progress> progress;
	int nEstimationGeometricIter;
	int nFusionMode;
	STEREO::SemiGlobalMatcher sgm;

	DenseDepthMapData(Scene& _scene, int _nFusionMode=0);
	~DenseDepthMapData();

	void SignalCompleteDepthmapFilter();
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif
