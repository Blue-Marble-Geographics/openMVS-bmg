/*
* PatchMatchCUDALegacy.inl
*
* The original CUDA PatchMatch depth-map estimator, restored verbatim from
* commit 235b9712 and kept alongside the reworked one for A/B testing.
* Compiled only when PATCHMATCH_CUDA_LEGACY is 1 -- see PatchMatchCUDASelect.h.
*
* Copyright (c) 2014-2021 SEACAVE
*
* Author(s):
*
*	  cDc <cdc.seacave@gmail.com>
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
*	  You are required to preserve legal notices and author attributions in
*	  that material or in the Appropriate Legal Notices displayed by works
*	  containing it.
*/

#ifndef _MVS_PATCHMATCHCUDALEGACY_INL_
#define _MVS_PATCHMATCHCUDALEGACY_INL_

#include "PatchMatchCUDASelect.h"

#if PATCHMATCH_CUDA_LEGACY


// I N C L U D E S /////////////////////////////////////////////////

#include "CUDA/Camera.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

struct DepthData;

namespace CUDA {

class PatchMatch {
public:
	struct Params {
		int nNumViews = 5;
		int nEstimationIters = 3;
		float fDepthMin = 0.f;
		float fDepthMax = 100.f;
		int nInitTopK = 3;
		bool bGeomConsistency = false;
		bool bLowResProcessed = false;
		float fThresholdKeepCost = 0;
	};

public:
	PatchMatch(int device=0);
	~PatchMatch();

	void Init(bool bGeomConsistency);
	void Release();

	void EstimateDepthMap(DepthData&);

	// No-op in this variant: it uploads every image and depth-map on each call
	// and frees them again, so there is no cross-pass cache to invalidate. It
	// exists only so the geometric-consistency loop in Scene::ComputeDepthMaps
	// can call it unconditionally, whichever variant is compiled.
	void InvalidateCachedDepthMaps() {}

	float4 GetPlaneHypothesis(const int index);
	float GetCost(const int index);

private:
	void ReleaseCUDA();
	void AllocatePatchMatchCUDA(const cv::Mat1f& image);
	void AllocateImageCUDA(size_t i, const cv::Mat1f& image, bool bInitImage, bool bInitDepthMap);
	void RunCUDA(float* ptrCostMap=NULL, uint32_t* ptrViewsMap=NULL);

public:
	Params params;

	std::vector<cv::Mat1f> images;
	std::vector<Camera> cameras;
	std::vector<cudaTextureObject_t> textureImages;
	std::vector<cudaTextureObject_t> textureDepths;
	Point4* depthNormalEstimates;

	Camera *cudaCameras;
	std::vector<cudaArray_t> cudaImageArrays;
	std::vector<cudaArray_t> cudaDepthArrays;
	cudaTextureObject_t* cudaTextureImages;
	cudaTextureObject_t* cudaTextureDepths;
	Point4* cudaDepthNormalEstimates;
	float* cudaLowDepths;
	float* cudaDepthNormalCosts;
	curandState* cudaRandStates;
	uint32_t* cudaSelectedViews;
};
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS

#endif // PATCHMATCH_CUDA_LEGACY

#endif // _MVS_PATCHMATCHCUDALEGACY_INL_
