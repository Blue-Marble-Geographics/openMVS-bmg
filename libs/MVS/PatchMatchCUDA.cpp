/*
* PatchMatchCUDA.cpp
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

#include "Common.h"
#include "PatchMatchCUDASelect.h"
#include "PatchMatchCUDA.h"
#include "DepthMap.h"

#include <chrono>

#if defined(_USE_CUDA) && !PATCHMATCH_CUDA_LEGACY



// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

namespace CUDA {

// ESTIMATE-phase breakdown timers (see MVS::CUDAEstimateBreakdown). Steady clock,
// read four times per scale -- negligible against the work being measured.
namespace {
	using pmc_clock = std::chrono::steady_clock;
	inline int64_t PmcNs(const pmc_clock::time_point& a, const pmc_clock::time_point& b) {
		return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
	}
}

// copy a cv::Mat's payload into a linear buffer (and back), tolerating
// non-continuous matrices; the estimate maps are created continuous, but the
// copies are cheap enough that correctness should not depend on it
static void CopyMatToBuffer(const cv::Mat& src, void* dst)
{
	ASSERT(!src.empty());
	const size_t rowBytes((size_t)src.cols * src.elemSize());
	if (src.isContinuous()) {
		memcpy(dst, src.data, rowBytes * (size_t)src.rows);
		return;
	}
	uint8_t* d = (uint8_t*)dst;
	for (int r = 0; r < src.rows; ++r, d += rowBytes)
		memcpy(d, src.ptr(r), rowBytes);
}
static void CopyBufferToMat(const void* src, cv::Mat& dst)
{
	ASSERT(!dst.empty());
	const size_t rowBytes((size_t)dst.cols * dst.elemSize());
	if (dst.isContinuous()) {
		memcpy(dst.data, src, rowBytes * (size_t)dst.rows);
		return;
	}
	const uint8_t* s = (const uint8_t*)src;
	for (int r = 0; r < dst.rows; ++r, s += rowBytes)
		memcpy(dst.ptr(r), s, rowBytes);
}


PatchMatch::PatchMatch(int device)
{
	// initialize CUDA device if needed
	if (SEACAVE::CUDA::devices.IsEmpty())
		SEACAVE::CUDA::initDevice(device);
}

PatchMatch::~PatchMatch()
{
	Release();
	if (stream) {
		cudaStreamDestroy(stream);
		stream = NULL;
	}
}

void PatchMatch::Release()
{
	ReleaseTextureCache();
	ReleaseCUDA();
}

void PatchMatch::ReleaseCUDA()
{
	// per image-set arrays
	cudaFree(cudaCameras); cudaCameras = NULL;
	cudaFree(cudaPairs); cudaPairs = NULL;
	cudaFree(cudaTextureImages); cudaTextureImages = NULL;
	cudaFree(cudaTextureDepths); cudaTextureDepths = NULL;
	capacityImages = 0;
	// pixel working set
	cudaFree(cudaPlanes); cudaPlanes = NULL;
	cudaFree(cudaCosts); cudaCosts = NULL;
	cudaFree(cudaViews); cudaViews = NULL;
	cudaFree(cudaDepthIn); cudaDepthIn = NULL;
	cudaFree(cudaNormalIn); cudaNormalIn = NULL;
	cudaFree(cudaDepthOut); cudaDepthOut = NULL;
	cudaFree(cudaNormalOut); cudaNormalOut = NULL;
	cudaFree(cudaConfOut); cudaConfOut = NULL;
	cudaFree(cudaViewsOut); cudaViewsOut = NULL;
	capacityPixels = 0;
	// conversion scratch
	cudaFree(cudaConvScratch); cudaConvScratch = NULL;
	cudaFree(cudaHalfScratch); cudaHalfScratch = NULL;
	capacityConvertPixels = 0;
	// pinned staging
	cudaFreeHost(pinnedIn); pinnedIn = NULL;
	cudaFreeHost(pinnedOut); pinnedOut = NULL;
	capacityPinnedPixels = 0;
}

void PatchMatch::Init(bool bGeomConsistency)
{
	if (bGeomConsistency) {
		params.bGeomConsistency = true;
		params.nEstimationIters = 1;
	} else {
		params.bGeomConsistency = false;
		params.nEstimationIters = OPTDENSE::nEstimationIters;
	}
}

void PatchMatch::InvalidateCachedDepthMaps()
{
	// the depth-maps on disk were just replaced by a geometric-consistency pass:
	// stamp a new generation so cached depth textures re-upload on next use
	++depthGeneration;
}


// ---------------------------------------------------------------------
// VRAM texture cache
//
// Every reference image re-samples the same neighbor images (and, during the
// geometric-consistency passes, their depth-maps): with the default 8 views
// each image used to be re-uploaded ~9 times per pass, every pass. Textures
// are instead kept resident in an LRU cache bounded by the free VRAM measured
// after the working buffers are allocated, so consecutive reference images hit
// the cache for most of their neighbor set and PCIe traffic drops to the
// actual content changes (new images, or depth-maps from a newer pass).
// ---------------------------------------------------------------------

void PatchMatch::ReleaseTextureCache()
{
	UnlockTextures();
	for (auto& it: texCache) {
		CachedTexture& entry = it.second;
		if (entry.tex)
			cudaDestroyTextureObject(entry.tex);
		if (entry.array)
			cudaFreeArray(entry.array);
	}
	if (!texCache.empty())
		DENSIFY_DIAG("PatchMatch CUDA texture cache released: %u entries, %s",
			(unsigned)texCache.size(), Util::formatBytes(texCacheUsed).c_str());
	texCache.clear();
	texLRU.clear();
	texCacheUsed = 0;
}

void PatchMatch::RefreshTextureCacheBudget()
{
	size_t freeBytes(0), totalBytes(0);
	if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
		texCacheBudget = 0;
		return;
	}
	// leave headroom for the display/WDDM and any transient driver allocations:
	// exceeding physical VRAM on Windows does not fail, it silently demotes
	// allocations to system memory and the run crawls, so stay well clear
	size_t reserve(totalBytes * 3 / 20); // 15%
	constexpr size_t minReserve((size_t)768 << 20);
	if (reserve < minReserve)
		reserve = minReserve;
	// the entries already resident are part of the allowance (they are not part
	// of the reported free memory anymore)
	texCacheBudget = (freeBytes > reserve ? freeBytes - reserve : 0) + texCacheUsed;
	#ifdef _MSC_VER
	char envBuf[64];
	size_t envLen(0);
	if (getenv_s(&envLen, envBuf, sizeof(envBuf), "OPENMVS_CUDA_TEXCACHE_MB") == 0 && envLen > 0)
		texCacheBudget = (size_t)_atoi64(envBuf) << 20;
	#else
	if (const char* env = getenv("OPENMVS_CUDA_TEXCACHE_MB"))
		texCacheBudget = (size_t)atoll(env) << 20;
	#endif
	DENSIFY_DIAG("PatchMatch CUDA texture cache budget: %s (free %s of %s)",
		Util::formatBytes(texCacheBudget).c_str(), Util::formatBytes(freeBytes).c_str(), Util::formatBytes(totalBytes).c_str());
	EvictTextures();
}

void PatchMatch::EvictTextures()
{
	// evict least-recently-used unlocked entries until under budget
	auto it = texLRU.end();
	while (texCacheUsed > texCacheBudget && it != texLRU.begin()) {
		--it;
		const auto cit = texCache.find(*it);
		ASSERT(cit != texCache.end());
		CachedTexture& entry = cit->second;
		if (entry.lockCount > 0)
			continue; // in use by the in-flight estimate
		if (entry.tex)
			cudaDestroyTextureObject(entry.tex);
		if (entry.array)
			cudaFreeArray(entry.array);
		texCacheUsed -= entry.bytes;
		texCache.erase(cit);
		it = texLRU.erase(it);
	}
}

PatchMatch::CachedTexture& PatchMatch::AcquireTexture(const TexKey& key, int width, int height, bool bDepth, bool& bNeedUpload)
{
	const auto it = texCache.find(key);
	if (it != texCache.end()) {
		CachedTexture& entry = it->second;
		texLRU.splice(texLRU.begin(), texLRU, entry.lruIt); // touch
		++entry.lockCount;
		lockedTextures.push_back(&entry);
		// a cached depth-map is stale once a newer pass replaced the files
		bNeedUpload = bDepth && entry.generation != depthGeneration;
		return entry;
	}

	CachedTexture entry;
	entry.width = width;
	entry.height = height;
	#if PMCUDA_OPT_FP16_TEX
	const cudaChannelFormatDesc channelDesc = bDepth ?
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat) :
		cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
	entry.bytes = (size_t)width * height * (bDepth ? 4 : 2);
	#else
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	entry.bytes = (size_t)width * height * 4;
	#endif
	if (cudaMallocArray(&entry.array, &channelDesc, width, height) != cudaSuccess) {
		// VRAM pressure: drop the whole unlocked cache and retry once
		cudaGetLastError();
		const size_t oldBudget(texCacheBudget);
		texCacheBudget = 0;
		EvictTextures();
		texCacheBudget = oldBudget;
		CUDA_CHECK(cudaMallocArray(&entry.array, &channelDesc, width, height));
	}

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = entry.array;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	// clamp, not wrap: wrap is only honored with normalized coordinates, so the
	// former cudaAddressModeWrap setting was already behaving as clamp
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode  = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	CUDA_CHECK(cudaCreateTextureObject(&entry.tex, &resDesc, &texDesc, NULL));

	texLRU.push_front(key);
	CachedTexture& stored = texCache[key];
	stored = entry;
	stored.lruIt = texLRU.begin();
	stored.lockCount = 1;
	lockedTextures.push_back(&stored);
	texCacheUsed += stored.bytes;
	bNeedUpload = true;
	EvictTextures();
	return stored;
}

cudaTextureObject_t PatchMatch::AcquireImageTexture(uint32_t ID, int srcWidth, int srcHeight, const cv::Mat1f& image)
{
	const TexKey key{
		((uint64_t)ID << 1) | 0,
		((uint64_t)(uint16_t)srcWidth << 48) | ((uint64_t)(uint16_t)srcHeight << 32) | ((uint64_t)(uint16_t)image.cols << 16) | (uint64_t)(uint16_t)image.rows
	};
	bool bNeedUpload;
	CachedTexture& entry = AcquireTexture(key, image.cols, image.rows, false, bNeedUpload);
	if (bNeedUpload) {
		#if PMCUDA_OPT_FP16_TEX
		// upload fp32, convert to fp16 on-device, blit into the texture array;
		// the sync copy runs on the legacy default stream which orders against
		// the (blocking) worker stream in both directions, so reusing the
		// scratch buffers across consecutive uploads is race-free
		const size_t area((size_t)image.cols * image.rows);
		EnsureConvertCapacity(area);
		CUDA_CHECK(cudaMemcpy2D(cudaConvScratch, (size_t)image.cols * sizeof(float), image.ptr<float>(0), image.step[0], (size_t)image.cols * sizeof(float), image.rows, cudaMemcpyHostToDevice));
		PMConvertFloatToHalf(cudaConvScratch, cudaHalfScratch, (int)area, stream);
		CUDA_CHECK(cudaMemcpy2DToArrayAsync(entry.array, 0, 0, cudaHalfScratch, (size_t)image.cols * 2, (size_t)image.cols * 2, image.rows, cudaMemcpyDeviceToDevice, stream));
		#else
		CUDA_CHECK(cudaMemcpy2DToArray(entry.array, 0, 0, image.ptr<float>(0), image.step[0], (size_t)image.cols * sizeof(float), image.rows, cudaMemcpyHostToDevice));
		#endif
	}
	return entry.tex;
}

cudaTextureObject_t PatchMatch::AcquireDepthTexture(uint32_t ID, const cv::Mat1f& depthMap, const cv::Size& targetSize)
{
	const TexKey key{
		((uint64_t)ID << 1) | 1,
		((uint64_t)(uint16_t)depthMap.cols << 48) | ((uint64_t)(uint16_t)depthMap.rows << 32) | ((uint64_t)(uint16_t)targetSize.width << 16) | (uint64_t)(uint16_t)targetSize.height
	};
	bool bNeedUpload;
	CachedTexture& entry = AcquireTexture(key, targetSize.width, targetSize.height, true, bNeedUpload);
	if (bNeedUpload) {
		const cv::Mat1f* pMap = &depthMap;
		if (depthMap.size() != targetSize) {
			cv::resize(depthMap, scratchDepthMap, targetSize, 0, 0, cv::INTER_LINEAR);
			pMap = &scratchDepthMap;
		}
		CUDA_CHECK(cudaMemcpy2DToArray(entry.array, 0, 0, pMap->ptr<float>(0), pMap->step[0], (size_t)targetSize.width * sizeof(float), targetSize.height, cudaMemcpyHostToDevice));
		entry.generation = depthGeneration;
	}
	return entry.tex;
}

void PatchMatch::UnlockTextures()
{
	for (CachedTexture* entry: lockedTextures) {
		ASSERT(entry->lockCount > 0);
		--entry->lockCount;
	}
	lockedTextures.clear();
}


void PatchMatch::EnsureConvertCapacity(size_t pixels)
{
	if (pixels <= capacityConvertPixels)
		return;
	// a previous image's async convert/blit may still reference the old scratch
	CUDA_CHECK(cudaStreamSynchronize(stream));
	cudaFree(cudaConvScratch); cudaConvScratch = NULL;
	cudaFree(cudaHalfScratch); cudaHalfScratch = NULL;
	CUDA_CHECK(cudaMalloc((void**)&cudaConvScratch, sizeof(float) * pixels));
	CUDA_CHECK(cudaMalloc(&cudaHalfScratch, 2 * pixels));
	capacityConvertPixels = pixels;
}

void PatchMatch::EnsureCapacity(size_t pixels, size_t numImages)
{
	if (!stream)
		CUDA_CHECK(cudaStreamCreate(&stream));
	else
		CUDA_CHECK(cudaStreamSynchronize(stream)); // never free behind queued work
	if (numImages > capacityImages) {
		cudaFree(cudaCameras); cudaCameras = NULL;
		cudaFree(cudaPairs); cudaPairs = NULL;
		cudaFree(cudaTextureImages); cudaTextureImages = NULL;
		cudaFree(cudaTextureDepths); cudaTextureDepths = NULL;
		CUDA_CHECK(cudaMalloc((void**)&cudaCameras, sizeof(Camera) * numImages));
		CUDA_CHECK(cudaMalloc((void**)&cudaPairs, sizeof(PairConstants) * numImages));
		CUDA_CHECK(cudaMalloc((void**)&cudaTextureImages, sizeof(cudaTextureObject_t) * numImages));
		CUDA_CHECK(cudaMalloc((void**)&cudaTextureDepths, sizeof(cudaTextureObject_t) * numImages));
		capacityImages = numImages;
	}
	if (pixels > capacityPixels) {
		cudaFree(cudaPlanes); cudaPlanes = NULL;
		cudaFree(cudaCosts); cudaCosts = NULL;
		cudaFree(cudaViews); cudaViews = NULL;
		cudaFree(cudaDepthIn); cudaDepthIn = NULL;
		cudaFree(cudaNormalIn); cudaNormalIn = NULL;
		cudaFree(cudaDepthOut); cudaDepthOut = NULL;
		cudaFree(cudaNormalOut); cudaNormalOut = NULL;
		cudaFree(cudaConfOut); cudaConfOut = NULL;
		cudaFree(cudaViewsOut); cudaViewsOut = NULL;
		CUDA_CHECK(cudaMalloc((void**)&cudaPlanes, sizeof(Point4) * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaCosts, sizeof(float) * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaViews, sizeof(unsigned) * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaDepthIn, sizeof(float) * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaNormalIn, sizeof(float) * 3 * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaDepthOut, sizeof(float) * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaNormalOut, sizeof(float) * 3 * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaConfOut, sizeof(float) * pixels));
		CUDA_CHECK(cudaMalloc((void**)&cudaViewsOut, 4 * pixels));
		capacityPixels = pixels;
		// the working set just changed: re-derive how much VRAM the texture
		// cache may keep resident
		RefreshTextureCacheBudget();
	}
	if (pixels > capacityPinnedPixels) {
		cudaFreeHost(pinnedIn); pinnedIn = NULL;
		cudaFreeHost(pinnedOut); pinnedOut = NULL;
		// in: depth + normal (16 B/px); out: depth + normal + conf + views (24 B/px)
		CUDA_CHECK(cudaHostAlloc((void**)&pinnedIn, sizeof(float) * 4 * pixels, cudaHostAllocDefault));
		CUDA_CHECK(cudaHostAlloc((void**)&pinnedOut, sizeof(float) * 6 * pixels, cudaHostAllocDefault));
		capacityPinnedPixels = pixels;
	}
}


void PatchMatch::EstimateDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.size() > 1);

	// multi-resolution
	DepthData& fullResDepthData(depthData);
	const unsigned totalScaleNumber(params.bGeomConsistency ? 0u : OPTDENSE::nSubResolutionLevels);
	DepthMap lowResDepthMap;
	NormalMap lowResNormalMap;
	ViewsMap lowResViewsMap;
	const IIndex numImages = depthData.images.size();
	params.nNumViews = (int)numImages-1;
	params.nInitTopK = std::min(params.nInitTopK, params.nNumViews);
	params.fDepthMin = depthData.dMin;
	params.fDepthMax = depthData.dMax;
	const int maxPixelViews(MINF(params.nNumViews, 4));
	for (unsigned scaleNumber = totalScaleNumber+1; scaleNumber-- > 0; ) {
		// initialize
		const auto _tPrep0 = pmc_clock::now();
		int64_t nsTexThisScale = 0;
		const float scale = 1.f / POWI(2, scaleNumber);
		DepthData currentDepthData(DepthMapsData::ScaleDepthData(fullResDepthData, scale));
		DepthData& depthData(scaleNumber==0 ? fullResDepthData : currentDepthData);
		const Image8U::Size size(depthData.images.front().image.size());
		params.bLowResProcessed = false;
		if (scaleNumber != totalScaleNumber) {
			// all resolutions, but the smallest one, if multi-resolution is enabled
			params.bLowResProcessed = true;
			cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0, cv::INTER_LINEAR);
			cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
			cv::resize(lowResViewsMap, depthData.viewsMap, size, 0, 0, cv::INTER_NEAREST);
		} else {
			if (totalScaleNumber > 0) {
				// smallest resolution, when multi-resolution is enabled
				fullResDepthData.depthMap.release();
				fullResDepthData.normalMap.release();
				fullResDepthData.confMap.release();
				fullResDepthData.viewsMap.release();
			}
			// smallest resolution if multi-resolution is enabled; highest otherwise
			if (depthData.viewsMap.empty())
				depthData.viewsMap.create(size);
		}
		if (scaleNumber == 0) {
			// highest resolution
			if (depthData.confMap.empty())
				depthData.confMap.create(size);
		}

		// set keep threshold to:
		params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep;
		if (totalScaleNumber) {
			// multi-resolution enabled
			if (scaleNumber > 0 && scaleNumber != totalScaleNumber) {
				// all sub-resolutions, but the smallest and highest
				params.fThresholdKeepCost = 0.f; // disable filtering
			} else if (scaleNumber == totalScaleNumber || (!params.bGeomConsistency && OPTDENSE::nEstimationGeometricIters)) {
				// smallest sub-resolution OR highest resolution and geometric consistency is not running but enabled
				params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep*1.2f;
			}
		} else {
			// multi-resolution disabled
			if (!params.bGeomConsistency && OPTDENSE::nEstimationGeometricIters) {
				// geometric consistency is not running but enabled
				params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep*1.2f;
			}
		}

		const size_t area((size_t)size.width * size.height);
		EnsureCapacity(area, numImages);
		cameras.resize(numImages);
		pairs.resize(numImages > 1 ? numImages-1 : 0);
		texImages.resize(numImages);
		texDepths.resize(numImages > 1 ? numImages-1 : 0);

		// set cameras, per-pair homography constants and image/depth textures
		const DepthData::ViewData& viewRef = depthData.images[0];
		for (IIndex i = 0; i < numImages; ++i) {
			const DepthData::ViewData& view = depthData.images[i];
			const Image32F& image = view.image;
			cameras[i] = Camera(
				Eigen::Map<const SEACAVE::Matrix3x3::EMat>(view.camera.K.val).cast<float>(),
				Eigen::Map<const SEACAVE::Matrix3x3::EMat>(view.camera.R.val).cast<float>(),
				Eigen::Map<const SEACAVE::Point3::EVec>(view.camera.C.ptr()).cast<float>(),
				image.cols, image.rows);
			if (i > 0) {
				// per-pair constants of the plane-induced homography (see
				// PairConstants); computed in double precision from the original
				// cameras, replacing the per-plane K0^-1 and 3x3 products the
				// kernel used to re-derive for every score
				const Eigen::Map<const SEACAVE::Matrix3x3::EMat> K0(viewRef.camera.K.val), R0(viewRef.camera.R.val);
				const Eigen::Map<const SEACAVE::Matrix3x3::EMat> K1(view.camera.K.val), R1(view.camera.R.val);
				const Eigen::Map<const SEACAVE::Point3::EVec> C0(viewRef.camera.C.ptr()), C1(view.camera.C.ptr());
				const SEACAVE::Matrix3x3::EMat A(K1 * R1 * R0.transpose() * K0.inverse());
				const SEACAVE::Point3::EVec b(K1 * (R1 * (C0 - C1)));
				PairConstants& pair = pairs[i-1];
				for (int r = 0; r < 3; ++r) {
					for (int c = 0; c < 3; ++c)
						pair.A[r*3+c] = (float)A(r,c);
					pair.b[r] = (float)b(r);
				}
			}
			// resolve the image texture through the VRAM cache; the source dims
			// (this view's un-sub-scaled size) keep entries produced through
			// different resize chains distinct
			const Image32F& srcImage = fullResDepthData.images[i].image;
			const auto _tTex0 = pmc_clock::now();
			texImages[i] = AcquireImageTexture(view.GetID(), srcImage.cols, srcImage.rows, image);
			if (params.bGeomConsistency && i > 0) {
				// resolve the previously computed depth-map through the cache
				texDepths[i-1] = view.depthMap.empty() ? 0 :
					AcquireDepthTexture(view.GetID(), view.depthMap, image.size());
			}
			nsTexThisScale += PmcNs(_tTex0, pmc_clock::now());
		}

		// setup per image-set CUDA memory (tiny, synchronous)
		CUDA_CHECK(cudaMemcpy(cudaTextureImages, texImages.data(), sizeof(cudaTextureObject_t) * numImages, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cudaCameras, cameras.data(), sizeof(Camera) * numImages, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cudaPairs, pairs.data(), sizeof(PairConstants) * params.nNumViews, cudaMemcpyHostToDevice));
		if (params.bGeomConsistency) {
			ASSERT(depthData.depthMap.size() == depthData.GetView().image.size());
			CUDA_CHECK(cudaMemcpy(cudaTextureDepths, texDepths.data(), sizeof(cudaTextureObject_t) * params.nNumViews, cudaMemcpyHostToDevice));
		}

		// upload the depth/normal priors through pinned staging and fuse them
		// into the per-pixel plane estimates on-device (replaces the former
		// serial host-side pack loop); the uploaded depth buffer doubles as the
		// low-resolution prior read by the kernels when bLowResProcessed is set,
		// exactly the same content the old code uploaded separately
		ASSERT(!depthData.depthMap.empty() && !depthData.normalMap.empty());
		CopyMatToBuffer(depthData.depthMap, pinnedIn);
		CopyMatToBuffer(depthData.normalMap, pinnedIn + area);
		// host prep ends here; everything to the stream sync below is device traffic
		const auto _tDev0 = pmc_clock::now();
		g_cudaEstimateBreakdown.nsPrep += PmcNs(_tPrep0, _tDev0) - nsTexThisScale;
		g_cudaEstimateBreakdown.nsTexture += nsTexThisScale;
		CUDA_CHECK(cudaMemcpyAsync(cudaDepthIn, pinnedIn, sizeof(float) * area, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(cudaNormalIn, pinnedIn + area, sizeof(float) * 3 * area, cudaMemcpyHostToDevice, stream));
		PMPackPlanes(cudaDepthIn, cudaNormalIn, cudaPlanes, (int)area, stream);

		// run CUDA patch-match
		ASSERT(!depthData.viewsMap.empty());
		RunCUDA(size.width, size.height);

		// split the results back into maps on-device, then download through
		// pinned staging; the confidence conversion and the views bit-mask ->
		// index expansion run in the unpack kernel (final scale only -- the
		// sub-resolution views download of the old code was dead data: the
		// selected views are re-derived from scratch by InitializeScore)
		const bool bFinalScale(scaleNumber == 0);
		const bool bWantConf(bFinalScale && !depthData.confMap.empty());
		float* const pinnedDepth(pinnedOut);
		float* const pinnedNormal(pinnedOut + area);
		float* const pinnedConf(pinnedOut + area * 4);
		uint8_t* const pinnedViews((uint8_t*)(pinnedOut + area * 5));
		PMUnpackResults(cudaPlanes, cudaCosts, cudaViews,
			cudaDepthOut, cudaNormalOut,
			bWantConf ? cudaConfOut : NULL, bFinalScale ? cudaViewsOut : NULL,
			(int)area, maxPixelViews, stream);
		CUDA_CHECK(cudaMemcpyAsync(pinnedDepth, cudaDepthOut, sizeof(float) * area, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(pinnedNormal, cudaNormalOut, sizeof(float) * 3 * area, cudaMemcpyDeviceToHost, stream));
		if (bWantConf)
			CUDA_CHECK(cudaMemcpyAsync(pinnedConf, cudaConfOut, sizeof(float) * area, cudaMemcpyDeviceToHost, stream));
		if (bFinalScale)
			CUDA_CHECK(cudaMemcpyAsync(pinnedViews, cudaViewsOut, 4 * area, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaGetLastError());
		// the uploads/kernels/downloads above are all enqueued asynchronously on one
		// stream, so their GPU time lands here at the sync, not at the call sites
		const auto _tRead0 = pmc_clock::now();
		g_cudaEstimateBreakdown.nsDevice += PmcNs(_tDev0, _tRead0);
		UnlockTextures();

		CopyBufferToMat(pinnedDepth, depthData.depthMap);
		CopyBufferToMat(pinnedNormal, depthData.normalMap);
		if (bWantConf)
			CopyBufferToMat(pinnedConf, depthData.confMap);
		if (bFinalScale)
			CopyBufferToMat(pinnedViews, depthData.viewsMap);

		// remember sub-resolution estimates for next iteration
		if (scaleNumber > 0) {
			lowResDepthMap = depthData.depthMap;
			lowResNormalMap = depthData.normalMap;
			lowResViewsMap = depthData.viewsMap;
		}
		g_cudaEstimateBreakdown.nsReadback += PmcNs(_tRead0, pmc_clock::now());
	}

	// apply ignore mask
	if (OPTDENSE::nIgnoreMaskLabel >= 0) {
		const DepthData::ViewData& view = depthData.GetView();
		BitMatrix mask;
		if (DepthEstimator::ImportIgnoreMask(*view.pImageData, depthData.depthMap.size(), (uint8_t)OPTDENSE::nIgnoreMaskLabel, mask))
			depthData.ApplyIgnoreMask(mask);
	}

	// aggregated by DepthMapsData::EstimateDepthMap's caller-side tally
	DENSIFY_DIAG("Depth-map for image %3u %s: %dx%d (%s)", depthData.images.front().GetID(),
		depthData.images.GetSize() > 2 ?
		String::FormatString("estimated using %2u images", depthData.images.size()-1).c_str() :
		String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		depthData.depthMap.cols, depthData.depthMap.rows, TD_TIMER_GET_FMT().c_str());
}
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS

#else

// Variant not selected: this file preprocesses away completely. The typedef
// only keeps the translation unit non-empty (MSVC C4206 / nvcc).
namespace MVS { namespace CUDA { typedef int PatchMatchCUDANotBuilt; } }

#endif // _USE_CUDA && !PATCHMATCH_CUDA_LEGACY
