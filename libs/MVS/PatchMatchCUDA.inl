/*
* PatchMatchCUDA.inl
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

#ifndef _MVS_PATCHMATCHCUDA_INL_
#define _MVS_PATCHMATCHCUDA_INL_

#include "PatchMatchCUDASelect.h"

#if !PATCHMATCH_CUDA_LEGACY


// I N C L U D E S /////////////////////////////////////////////////

#include "CUDA/Camera.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <list>
#include <map>
#include <vector>


// D E F I N E S ///////////////////////////////////////////////////

// Checkerboard block geometry. Declared here rather than in PatchMatchCUDA.cu so
// the host side can report it; the .cu turns these into the kernels'
// __launch_bounds__, which is what keeps the launch legal (see the note there).
#ifndef PMCUDA_BLOCK_W
#define PMCUDA_BLOCK_W 32
#endif
#ifndef PMCUDA_BLOCK_H
#define PMCUDA_BLOCK_H 16
#endif
#ifndef PMCUDA_MIN_BLOCKS
#define PMCUDA_MIN_BLOCKS 0
#endif

// Store the grey images sampled by the ZNCC kernels as fp16 textures instead of
// fp32: half the texture bandwidth AND full-rate bilinear filtering (fp32 textures
// filter at half rate on every NVIDIA architecture this ships on). The images are
// [0,1] intensities derived from 8-bit sources, so the fp16 quantization step
// (~5e-4) sits an order of magnitude BELOW the source quantization (1/255); the
// per-view depth-map textures used by geometric consistency keep fp32 because
// depth has real dynamic range. Flip to 0 to A/B against fp32 textures.
#ifndef PMCUDA_OPT_FP16_TEX
#define PMCUDA_OPT_FP16_TEX 1
#endif


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

			// per image-pair constants of the plane-induced homography decomposition
			//   H(n,d) = A + b * (n^T K0^-1) / (n . X(p,d))
			// where A = K1*R1*R0^T*K0^-1 and b = K1*R1*(C0-C1) depend only on the pair,
			// so they are computed once per pair on the host (in double precision) instead
			// of re-deriving K0^-1 and two full 3x3 products per plane score on the device
			struct PairConstants {
				float A[9]; // row-major 3x3
				float b[3];
			};

		public:
			PatchMatch(int device = 0);
			~PatchMatch();

			void Init(bool bGeomConsistency);
			void Release();

			void EstimateDepthMap(DepthData&);

			// the depth-maps on disk changed (a geometric-consistency pass replaced them):
			// drop every cached depth texture so the next pass re-uploads current content
			void InvalidateCachedDepthMaps();

		private:
			// VRAM texture cache
			// key.a  = (image ID << 1) | isDepthMap
			// key.b  = source WxH (the map this content was resized from; equals dst for
			//          unscaled content) packed with destination WxH -- the source dims
			//          keep entries produced through different resize chains distinct,
			//          so a hit always returns bit-identical content to a cold upload
			struct TexKey {
				uint64_t a, b;
				bool operator<(const TexKey& r) const { return a < r.a || (a == r.a && b < r.b); }
			};
			struct CachedTexture {
				cudaArray_t array = NULL;
				cudaTextureObject_t tex = 0;
				size_t bytes = 0;
				uint32_t generation = 0; // depth-maps only: which pass produced the content
				int width = 0, height = 0;
				unsigned lockCount = 0; // held by the in-flight estimate; never evicted
				std::list<TexKey>::iterator lruIt;
			};

			cudaTextureObject_t AcquireImageTexture(uint32_t ID, int srcWidth, int srcHeight, const cv::Mat1f& image);
			cudaTextureObject_t AcquireDepthTexture(uint32_t ID, const cv::Mat1f& depthMap, const cv::Size& targetSize);
			CachedTexture& AcquireTexture(const TexKey& key, int width, int height, bool bDepth, bool& bNeedUpload);
			void UnlockTextures();
			void EvictTextures();
			void ReleaseTextureCache();
			void RefreshTextureCacheBudget();

			void EnsureCapacity(size_t pixels, size_t numImages);
			void EnsureConvertCapacity(size_t pixels);
			void ReleaseCUDA();

			void RunCUDA(int width, int height);
			template <int MAXV>
			void RunCUDAT(int width, int height);

		public:
			Params params;

		private:
			// host mirrors for the current image set
			std::vector<Camera> cameras;
			std::vector<PairConstants> pairs;
			std::vector<cudaTextureObject_t> texImages;
			std::vector<cudaTextureObject_t> texDepths;
			std::vector<CachedTexture*> lockedTextures;
			cv::Mat1f scratchDepthMap; // host staging for depth-maps resized before upload

			// texture cache state
			std::map<TexKey, CachedTexture> texCache;
			std::list<TexKey> texLRU; // front = most recently used
			size_t texCacheUsed = 0, texCacheBudget = 0;
			uint32_t depthGeneration = 1;

			// per image-set device arrays (grow-only by image count)
			Camera* cudaCameras = NULL;
			PairConstants* cudaPairs = NULL;
			cudaTextureObject_t* cudaTextureImages = NULL;
			cudaTextureObject_t* cudaTextureDepths = NULL;
			size_t capacityImages = 0;

			// reference-image-sized device working set (grow-only by pixel count)
			Point4* cudaPlanes = NULL;      // per-pixel depth+normal estimate
			float* cudaCosts = NULL;        // per-pixel aggregated ZNCC cost
			unsigned* cudaViews = NULL;     // per-pixel selected-views bit-mask
			float* cudaDepthIn = NULL;      // uploaded depth prior (also serves as the low-res prior)
			float* cudaNormalIn = NULL;     // uploaded normal prior (3 floats/px)
			float* cudaDepthOut = NULL;
			float* cudaNormalOut = NULL;    // 3 floats/px
			float* cudaConfOut = NULL;
			uint8_t* cudaViewsOut = NULL;   // 4 bytes/px, ViewsID layout
			size_t capacityPixels = 0;

			// view-image-sized conversion scratch (grow-only)
			float* cudaConvScratch = NULL;  // fp32 image staging before fp16 conversion
			void* cudaHalfScratch = NULL;   // fp16 converted image (2 bytes/px)
			size_t capacityConvertPixels = 0;

			// pinned host staging for the per-image upload/download
			float* pinnedIn = NULL;         // 16 B/px: depth + normal
			float* pinnedOut = NULL;        // 24 B/px: depth + normal + conf + views
			size_t capacityPinnedPixels = 0;

			cudaStream_t stream = NULL;
		};
		/*----------------------------------------------------------------*/

		// small device-kernel wrappers implemented in PatchMatchCUDA.cu, callable from
		// the host-compiled PatchMatchCUDA.cpp
		void PMConvertFloatToHalf(const float* d_src, void* d_dstHalf, int count, cudaStream_t stream);
		void PMPackPlanes(const float* d_depthIn, const float* d_normalIn, Point4* d_planes, int count, cudaStream_t stream);
		void PMUnpackResults(const Point4* d_planes, const float* d_costs, const unsigned* d_views,
			float* d_depthOut, float* d_normalOut, float* d_confOut/*may be NULL*/, uint8_t* d_viewsOut/*may be NULL*/,
			int count, int maxPixelViews, cudaStream_t stream);

		// Checkerboard block height actually launched, defined in PatchMatchCUDA.cu.
		// PMCUDA_BLOCK_H normally; halved as needed on a device that will not accept that
		// many threads for the compiled kernel, which is what keeps a low-end GPU running
		// the CUDA path instead of failing the launch with cudaErrorLaunchOutOfResources.
		// Lives here because the .cu sees none of the SEACAVE Common headers and so has no
		// logger of its own -- PatchMatchCUDA.cpp reports it.
		extern int g_pmcudaEffectiveBlockH;
		/*----------------------------------------------------------------*/

	} // namespace CUDA

} // namespace MVS

#endif // !PATCHMATCH_CUDA_LEGACY

#endif // _MVS_PATCHMATCHCUDA_INL_
