/*
* PointCloudFilterCUDA.cu
*
* GPU sweep for Scene::PointCloudFilter's camera-point visibility test
* (exploration, 2026-09).
*
* The CPU sweep casts, for every (point X, view v) pair, a cone from v's center
* towards X with a half-angle of FOV/width (about one pixel) and height 1.02*|X-C|,
* collects every cloud point inside it through an octree, and accumulates a signed
* integer score per collected point. Here the same queries run per view:
*
*   1. project the whole cloud into v and key every point by its pixel
*      (1-pixel bins, with a margin around the image), packed with its index;
*   2. drop the points outside the grid (a large scene puts most of the cloud
*      outside any one view), radix-sort the rest by bin, find each bin's
*      [start,end) and gather the positions in bin order;
*   3. for every point that observes v, test the points binned within a
*      conservative pixel radius of its projection with EXACTLY the CPU's classify
*      and emit arithmetic (explicit round-to-nearest intrinsics, no FMA
*      contraction), accumulating with integer atomics.
*
* The grid is only a prune: the classify test decides, and integer accumulation is
* order-independent, so the scores are identical to the CPU sweep. Queries the grid
* cannot bound (X behind the camera, rays near 90 degrees off-axis, windows leaving
* the grid margin) are handed back for the caller's octree.
*
* Deliberately self-contained: no SEACAVE headers, as PatchMatchCUDA.cu.
*
* Compiled only with PCF_EXPERIMENTAL_SWEEPS 1 (see PointCloudFilterCUDA.h): the
* sweep is not in the pipeline, and it is the project's only CUB user, whose radix
* sort otherwise costs every CUDA build its compile time and binary size.
*/

#include "PointCloudFilterCUDA.h"

#if PCF_GPU_SWEEP

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <chrono>
#include <cmath>

namespace MVS {
namespace CUDA {

namespace {

constexpr int PCF_MARGIN = 64; // grid margin around the image, pixels

struct DevView {
	float fx, fy, cx, cy;
	float R[9];
	float C[3];
	float angle, cosSq;
	int gw, gh, bs;
	unsigned nBins;
};

// camera projection used only for binning and for the query window (both
// conservative, so its rounding is irrelevant to the result)
__device__ __forceinline__ bool ProjectPoint(const DevView& v, float px, float py, float pz, float& u, float& w, float& xn, float& yn)
{
	const float X = px - v.C[0], Y = py - v.C[1], Z = pz - v.C[2];
	const float z = v.R[6]*X + v.R[7]*Y + v.R[8]*Z;
	if (!(z > 0.f))
		return false;
	const float iz = 1.f / z;
	xn = (v.R[0]*X + v.R[1]*Y + v.R[2]*Z) * iz;
	yn = (v.R[3]*X + v.R[4]*Y + v.R[5]*Z) * iz;
	u = v.fx*xn + v.cx;
	w = v.fy*yn + v.cy;
	return true;
}

__device__ __forceinline__ unsigned BinOf(const DevView& v, float u, float w)
{
	const float bu = (u + (float)PCF_MARGIN) / (float)v.bs, bw = (w + (float)PCF_MARGIN) / (float)v.bs;
	if (!(bu >= 0.f && bw >= 0.f && bu < (float)v.gw && bw < (float)v.gh))
		return v.nBins; // invalid: sorts after every real bin
	return (unsigned)(int)bw * (unsigned)v.gw + (unsigned)(int)bu;
}

// (bin << 32 | point index); bin == nBins marks a point outside the grid
__global__ void KKeys(const float* __restrict__ xyz, unsigned n, DevView v, unsigned long long* __restrict__ packed)
{
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	float u, w, xn, yn;
	const unsigned key = ProjectPoint(v, xyz[i*3+0], xyz[i*3+1], xyz[i*3+2], u, w, xn, yn) ? BinOf(v, u, w) : v.nBins;
	packed[i] = ((unsigned long long)key << 32) | i;
}

struct InGrid {
	unsigned nBins;
	__host__ __device__ __forceinline__ bool operator()(const unsigned long long& x) const { return (unsigned)(x >> 32) < nBins; }
};

// bin bounds (start/end are zeroed before, so empty bins read [0,0)), point indices
// and positions in bin order, over the m in-grid points
__global__ void KBounds(const unsigned long long* __restrict__ packed, unsigned m,
	const float* __restrict__ xyz, unsigned* __restrict__ binStart, unsigned* __restrict__ binEnd,
	unsigned* __restrict__ vals, float* __restrict__ binXYZ)
{
	const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= m)
		return;
	const unsigned long long pk = packed[j];
	const unsigned k = (unsigned)(pk >> 32);
	if (j == 0 || (unsigned)(packed[j-1] >> 32) != k)
		binStart[k] = j;
	if (j == m-1 || (unsigned)(packed[j+1] >> 32) != k)
		binEnd[k] = j + 1;
	const unsigned s = (unsigned)pk;
	vals[j] = s;
	binXYZ[j*3+0] = xyz[s*3+0];
	binXYZ[j*3+1] = xyz[s*3+1];
	binXYZ[j*3+2] = xyz[s*3+2];
}

__global__ void KQuery(const float* __restrict__ xyz, const unsigned* __restrict__ viewSizes,
	const unsigned* __restrict__ qPts, unsigned nQ, unsigned viewIdx, DevView v, float slackPx,
	const unsigned* __restrict__ binStart, const unsigned* __restrict__ binEnd,
	const float* __restrict__ binXYZ, const unsigned* __restrict__ vals,
	int* __restrict__ vis, unsigned* __restrict__ fbCount, unsigned* __restrict__ fbPts, unsigned* __restrict__ fbViews, unsigned fbCap)
{
	const unsigned qi = blockIdx.x * blockDim.x + threadIdx.x;
	if (qi >= nQ)
		return;
	const unsigned idx = qPts[qi];
	const float px = xyz[idx*3+0], py = xyz[idx*3+1], pz = xyz[idx*3+2];
	// --- the CPU's Query::InitView, operation for operation ---
	// D = X - apex; distance = D.norm() (Eigen 3.4 unrolls a 3-sum as x + (y + z));
	// d = D / distance; maxH = distance * 1.02
	const float Dx0 = __fsub_rn(px, v.C[0]), Dy0 = __fsub_rn(py, v.C[1]), Dz0 = __fsub_rn(pz, v.C[2]);
	const float distance = __fsqrt_rn(__fadd_rn(__fmul_rn(Dx0, Dx0), __fadd_rn(__fmul_rn(Dy0, Dy0), __fmul_rn(Dz0, Dz0))));
	const float dx = __fdiv_rn(Dx0, distance), dy = __fdiv_rn(Dy0, distance), dz = __fdiv_rn(Dz0, distance);
	const float maxH = __fmul_rn(distance, 1.02f);
	const float cosSq = v.cosSq;
	const int wgt = (int)viewSizes[idx];
	// --- conservative window (not part of the result) ---
	float u0, w0, xn0, yn0;
	bool fb = !ProjectPoint(v, px, py, pz, u0, w0, xn0, yn0);
	int bx0 = 0, by0 = 0, bx1 = -1, by1 = -1;
	if (!fb) {
		const float th1 = atanf(sqrtf(xn0*xn0 + yn0*yn0)) + v.angle;
		if (th1 > 1.3f) {
			fb = true;
		} else {
			const float tn = tanf(th1);
			const float rad = 1.05f * fmaxf(v.fx, v.fy) * v.angle * (1.f + tn*tn) + slackPx;
			const float lo_u = (u0 - rad + (float)PCF_MARGIN) / (float)v.bs, hi_u = (u0 + rad + (float)PCF_MARGIN) / (float)v.bs;
			const float lo_w = (w0 - rad + (float)PCF_MARGIN) / (float)v.bs, hi_w = (w0 + rad + (float)PCF_MARGIN) / (float)v.bs;
			if (!(lo_u >= 0.f && lo_w >= 0.f && hi_u < (float)v.gw && hi_w < (float)v.gh)) {
				fb = true;
			} else {
				bx0 = (int)lo_u; bx1 = (int)hi_u;
				by0 = (int)lo_w; by1 = (int)hi_w;
			}
		}
	}
	if (fb) {
		const unsigned s = atomicAdd(fbCount, 1u);
		if (s < fbCap) { fbPts[s] = idx; fbViews[s] = viewIdx; }
		return;
	}
	// --- exact classify + emit, as Query::operator() ---
	for (int by = by0; by <= by1; ++by) {
		for (int bx = bx0; bx <= bx1; ++bx) {
			const unsigned b = (unsigned)by * (unsigned)v.gw + (unsigned)bx;
			const unsigned e = binEnd[b];
			for (unsigned j = binStart[b]; j < e; ++j) {
				const float Dx = __fsub_rn(binXYZ[j*3+0], v.C[0]);
				const float Dy = __fsub_rn(binXYZ[j*3+1], v.C[1]);
				const float Dz = __fsub_rn(binXYZ[j*3+2], v.C[2]);
				// t = (dx*Dx + dy*Dy) + dz*Dz, as the SSE add(add(mul,mul),mul)
				const float t = __fadd_rn(__fadd_rn(__fmul_rn(dx, Dx), __fmul_rn(dy, Dy)), __fmul_rn(dz, Dz));
				if (!(t > 0.f && t <= maxH))
					continue;
				const float nSq = __fadd_rn(__fadd_rn(__fmul_rn(Dx, Dx), __fmul_rn(Dy, Dy)), __fmul_rn(Dz, Dz));
				if (!(__fmul_rn(t, t) > __fmul_rn(cosSq, nSq)))
					continue;
				// IsDepthSimilar(refDist, t, 0.01): |refDist - t| / refDist < 0.01
				if (__fdiv_rn(fabsf(__fsub_rn(distance, t)), distance) < 0.01f)
					continue;
				const unsigned c = vals[j];
				atomicAdd(vis + c, (t > distance) ? (int)viewSizes[c] : -wgt);
			}
		}
	}
}

struct DevBuf {
	void* p = nullptr;
	~DevBuf() { if (p) cudaFree(p); }
	template <typename T> T* as() const { return (T*)p; }
};

#define PCF_CHECK(x) do { if ((x) != cudaSuccess) { cudaGetLastError(); return false; } } while (0)

} // namespace

bool PointCloudVisibilityCUDA(
	const float* xyz, size_t numPoints, const uint32_t* viewSizes,
	const PCFViewDesc* views, size_t numViews,
	const uint32_t* queryOffsets, const uint32_t* queryPoints,
	int binSize, float slackPx,
	int* vis,
	std::vector<uint32_t>& fallbackPoints, std::vector<uint32_t>& fallbackViews,
	PCFStats& stats)
{
	typedef std::chrono::steady_clock clk;
	const clk::time_point t0 = clk::now();
	fallbackPoints.clear(); fallbackViews.clear();
	if (numPoints == 0 || numViews == 0)
		return true;
	if (numPoints >= 0xFFFFFFFFull)
		return false;
	const unsigned n = (unsigned)numPoints;
	const size_t nQueries = queryOffsets[numViews];
	const int bs = binSize > 0 ? binSize : 1;

	// largest grid over all views
	unsigned maxBins = 0;
	for (size_t v = 0; v < numViews; ++v) {
		const unsigned gw = (unsigned)((views[v].width + 2*PCF_MARGIN + bs - 1) / bs);
		const unsigned gh = (unsigned)((views[v].height + 2*PCF_MARGIN + bs - 1) / bs);
		maxBins = gw*gh > maxBins ? gw*gh : maxBins;
	}
	int endBit = 1;
	while (endBit < 32 && ((1ull << endBit) <= (unsigned long long)maxBins))
		++endBit;

	cudaStream_t stream;
	PCF_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	struct StreamGuard { cudaStream_t s; ~StreamGuard() { cudaStreamDestroy(s); } } sg{stream};

	DevBuf dXYZ, dVS, dQ, dVis, dPackA, dPackB, dVals, dNumSel, dStart, dEnd, dBinXYZ, dTmp, dFb, dFbPts, dFbViews;
	const unsigned fbCap = 1u << 20;
	PCF_CHECK(cudaMalloc(&dXYZ.p, sizeof(float) * 3 * numPoints));
	PCF_CHECK(cudaMalloc(&dVS.p, sizeof(unsigned) * numPoints));
	PCF_CHECK(cudaMalloc(&dQ.p, sizeof(unsigned) * (nQueries ? nQueries : 1)));
	PCF_CHECK(cudaMalloc(&dVis.p, sizeof(int) * numPoints));
	PCF_CHECK(cudaMalloc(&dPackA.p, sizeof(unsigned long long) * numPoints));
	PCF_CHECK(cudaMalloc(&dPackB.p, sizeof(unsigned long long) * numPoints));
	PCF_CHECK(cudaMalloc(&dVals.p, sizeof(unsigned) * numPoints));
	PCF_CHECK(cudaMalloc(&dNumSel.p, sizeof(int)));
	PCF_CHECK(cudaMalloc(&dStart.p, sizeof(unsigned) * maxBins));
	PCF_CHECK(cudaMalloc(&dEnd.p, sizeof(unsigned) * maxBins));
	PCF_CHECK(cudaMalloc(&dBinXYZ.p, sizeof(float) * 3 * numPoints));
	PCF_CHECK(cudaMalloc(&dFb.p, sizeof(unsigned)));
	PCF_CHECK(cudaMalloc(&dFbPts.p, sizeof(unsigned) * fbCap));
	PCF_CHECK(cudaMalloc(&dFbViews.p, sizeof(unsigned) * fbCap));
	size_t tmpBytes = 0;
	{
		size_t sortBytes = 0, selBytes = 0;
		cub::DoubleBuffer<unsigned long long> k(dPackA.as<unsigned long long>(), dPackB.as<unsigned long long>());
		PCF_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sortBytes, k, (int)n, 32, 32 + endBit, stream));
		PCF_CHECK(cub::DeviceSelect::If(nullptr, selBytes, dPackA.as<unsigned long long>(), dPackB.as<unsigned long long>(),
			dNumSel.as<int>(), (int)n, InGrid{0u}, stream));
		tmpBytes = sortBytes > selBytes ? sortBytes : selBytes;
	}
	PCF_CHECK(cudaMalloc(&dTmp.p, tmpBytes));
	stats.bytesDevice = sizeof(float)*3*numPoints*2 + sizeof(unsigned)*numPoints*4 + sizeof(unsigned long long)*numPoints*2 + sizeof(unsigned)*nQueries
		+ sizeof(unsigned)*(size_t)maxBins*2 + tmpBytes + sizeof(unsigned)*2*(size_t)fbCap;

	cudaEvent_t ev[8];
	for (auto& e : ev) PCF_CHECK(cudaEventCreate(&e));
	struct EventGuard { cudaEvent_t* e; ~EventGuard() { for (int i = 0; i < 8; ++i) cudaEventDestroy(e[i]); } } eg{ev};

	cudaEventRecord(ev[0], stream);
	PCF_CHECK(cudaMemcpyAsync(dXYZ.p, xyz, sizeof(float) * 3 * numPoints, cudaMemcpyHostToDevice, stream));
	PCF_CHECK(cudaMemcpyAsync(dVS.p, viewSizes, sizeof(unsigned) * numPoints, cudaMemcpyHostToDevice, stream));
	if (nQueries)
		PCF_CHECK(cudaMemcpyAsync(dQ.p, queryPoints, sizeof(unsigned) * nQueries, cudaMemcpyHostToDevice, stream));
	PCF_CHECK(cudaMemsetAsync(dVis.p, 0, sizeof(int) * numPoints, stream));
	PCF_CHECK(cudaMemsetAsync(dFb.p, 0, sizeof(unsigned), stream));
	cudaEventRecord(ev[1], stream);
	PCF_CHECK(cudaStreamSynchronize(stream));
	float ms = 0;
	cudaEventElapsedTime(&ms, ev[0], ev[1]); stats.msUpload = ms;

	constexpr unsigned TPB = 256;
	for (size_t vi = 0; vi < numViews; ++vi) {
		const unsigned q0 = queryOffsets[vi], q1 = queryOffsets[vi+1];
		if (q0 == q1)
			continue;
		const PCFViewDesc& s = views[vi];
		DevView dv;
		dv.fx = s.fx; dv.fy = s.fy; dv.cx = s.cx; dv.cy = s.cy;
		for (int k = 0; k < 9; ++k) dv.R[k] = s.R[k];
		for (int k = 0; k < 3; ++k) dv.C[k] = s.C[k];
		dv.angle = s.angle; dv.cosSq = s.cosSq; dv.bs = bs;
		dv.gw = (s.width + 2*PCF_MARGIN + bs - 1) / bs;
		dv.gh = (s.height + 2*PCF_MARGIN + bs - 1) / bs;
		dv.nBins = (unsigned)dv.gw * (unsigned)dv.gh;

		cudaEventRecord(ev[2], stream);
		KKeys<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(dXYZ.as<float>(), n, dv, dPackA.as<unsigned long long>());
		cudaEventRecord(ev[3], stream);
		// keep only the in-grid points (A -> B); the count comes back to size the sort
		PCF_CHECK(cub::DeviceSelect::If(dTmp.p, tmpBytes, dPackA.as<unsigned long long>(), dPackB.as<unsigned long long>(),
			dNumSel.as<int>(), (int)n, InGrid{dv.nBins}, stream));
		int m = 0;
		PCF_CHECK(cudaMemcpyAsync(&m, dNumSel.p, sizeof(int), cudaMemcpyDeviceToHost, stream));
		cudaEventRecord(ev[7], stream);
		PCF_CHECK(cudaStreamSynchronize(stream));
		stats.nInView += (size_t)m;
		// sort by bin: bits [32, 32+endBit) of the packed key (B, with A as the alternate)
		cub::DoubleBuffer<unsigned long long> k(dPackB.as<unsigned long long>(), dPackA.as<unsigned long long>());
		if (m > 1)
			PCF_CHECK(cub::DeviceRadixSort::SortKeys(dTmp.p, tmpBytes, k, m, 32, 32 + endBit, stream));
		cudaEventRecord(ev[4], stream);
		PCF_CHECK(cudaMemsetAsync(dStart.p, 0, sizeof(unsigned) * dv.nBins, stream));
		PCF_CHECK(cudaMemsetAsync(dEnd.p, 0, sizeof(unsigned) * dv.nBins, stream));
		if (m > 0)
			KBounds<<<((unsigned)m + TPB - 1) / TPB, TPB, 0, stream>>>(k.Current(), (unsigned)m, dXYZ.as<float>(),
				dStart.as<unsigned>(), dEnd.as<unsigned>(), dVals.as<unsigned>(), dBinXYZ.as<float>());
		cudaEventRecord(ev[5], stream);
		const unsigned nQ = q1 - q0;
		KQuery<<<(nQ + TPB - 1) / TPB, TPB, 0, stream>>>(dXYZ.as<float>(), dVS.as<unsigned>(), dQ.as<unsigned>() + q0, nQ, (unsigned)vi, dv, slackPx,
			dStart.as<unsigned>(), dEnd.as<unsigned>(), dBinXYZ.as<float>(), dVals.as<unsigned>(),
			dVis.as<int>(), dFb.as<unsigned>(), dFbPts.as<unsigned>(), dFbViews.as<unsigned>(), fbCap);
		cudaEventRecord(ev[6], stream);
		PCF_CHECK(cudaGetLastError());
		// per-view phase timing (the sync costs a little; this is an exploration build)
		PCF_CHECK(cudaEventSynchronize(ev[6]));
		cudaEventElapsedTime(&ms, ev[2], ev[3]); stats.msKeys += ms;
		cudaEventElapsedTime(&ms, ev[3], ev[7]); stats.msSelect += ms;
		cudaEventElapsedTime(&ms, ev[7], ev[4]); stats.msSort += ms;
		cudaEventElapsedTime(&ms, ev[4], ev[5]); stats.msBounds += ms;
		cudaEventElapsedTime(&ms, ev[5], ev[6]); stats.msQuery += ms;
	}

	cudaEventRecord(ev[2], stream);
	unsigned nFb = 0;
	PCF_CHECK(cudaMemcpyAsync(&nFb, dFb.p, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	PCF_CHECK(cudaStreamSynchronize(stream));
	if (nFb > fbCap)
		return false; // too many unbounded queries for the fallback list: let the CPU do it all
	// accumulate into the caller's array (it may already hold scores)
	std::vector<int> tmp(numPoints);
	PCF_CHECK(cudaMemcpyAsync(tmp.data(), dVis.p, sizeof(int) * numPoints, cudaMemcpyDeviceToHost, stream));
	fallbackPoints.resize(nFb); fallbackViews.resize(nFb);
	if (nFb) {
		PCF_CHECK(cudaMemcpyAsync(fallbackPoints.data(), dFbPts.p, sizeof(unsigned) * nFb, cudaMemcpyDeviceToHost, stream));
		PCF_CHECK(cudaMemcpyAsync(fallbackViews.data(), dFbViews.p, sizeof(unsigned) * nFb, cudaMemcpyDeviceToHost, stream));
	}
	cudaEventRecord(ev[3], stream);
	PCF_CHECK(cudaStreamSynchronize(stream));
	cudaEventElapsedTime(&ms, ev[2], ev[3]); stats.msDownload = ms;
	for (size_t i = 0; i < numPoints; ++i)
		vis[i] += tmp[i];
	stats.nFallback = nFb;
	stats.msTotal = (double)std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count() * 1e-3;
	return true;
}

} // namespace CUDA
} // namespace MVS

#else

namespace MVS { namespace CUDA { typedef int PointCloudFilterCUDANotBuilt; } }

#endif // PCF_GPU_SWEEP
