/*
* PatchMatchCUDA.cu
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

#include "PatchMatchCUDASelect.h"

#if !PATCHMATCH_CUDA_LEGACY

#include "PatchMatchCUDA.inl"

#include <cuda_fp16.h>

// =====================================================================
// Optional perf toggles (semantics-preserving; flip to 0 to A/B test).
//
// PMCUDA_OPT_FAST_INTRIN  : use single-precision/intrinsic math
//                           (__expf/__sincosf/sqrtf) on hot paths instead
//                           of the implicitly-double exp/sin/cos. Slightly
//                           less precise (a few ULP) but PatchMatch is
//                           iterative+aggregated so drift is irrelevant.
//
// PMCUDA_OPT_RESTRICT     : add __restrict__ to kernel pointer params so
//                           NVCC can use the read-only/LDG cache and
//                           reorder loads. Pure annotation, zero runtime
//                           cost when off.
//
// PMCUDA_TDR_MAX_PIXEL_VIEWS :
//                           cap on (pixels x views) processed by a single
//                           checkerboard kernel launch. Windows' WDDM driver
//                           resets the GPU when one kernel runs longer than
//                           TdrDelay (2s default); the propagation kernels
//                           scale with image area x views, so on low-end
//                           GPUs a full-frame launch can cross that line and
//                           kill the whole run. Each checkerboard phase only
//                           reads OPPOSITE-parity cells written by the
//                           previous phase (every neighbor offset in the
//                           sampling tables has odd |dx|+|dy|), so splitting
//                           one phase into several row-band launches is
//                           bit-identical to a single launch -- banding is a
//                           pure reliability measure, not an approximation.
// =====================================================================
#ifndef PMCUDA_OPT_FAST_INTRIN
#define PMCUDA_OPT_FAST_INTRIN 1
#endif
#ifndef PMCUDA_OPT_RESTRICT
#define PMCUDA_OPT_RESTRICT 1
#endif
#ifndef PMCUDA_TDR_MAX_PIXEL_VIEWS
#define PMCUDA_TDR_MAX_PIXEL_VIEWS (4<<20)
#endif

// ---------------------------------------------------------------------
// Occupancy controls for the checkerboard kernels.
//
// MEASURED, 276-image run, Quadro RTX 5000 (Turing SM 7.5), photometric device=:
//
//   no __launch_bounds__      27820 ms   <- default, and the fastest
//   __launch_bounds__(512)    28168 ms
//   __launch_bounds__(256)    31778 ms
//
// Monotonic, and the opposite of the obvious prediction. ProcessPixel carries a
// lot of per-thread state -- costArray[8][MAXV] (dynamically indexed, hence
// local-memory resident), the RefPatch, plus the sampling/selection arrays -- so
// the expectation was that giving nvcc a bigger register budget would cut spill
// and win. It loses. Turing has 64K registers and room for 1024 threads per SM,
// so __launch_bounds__ caps registers at 65536/maxThreadsPerBlock: 128/thread at
// 512 threads, 255 at 256. Raising that cap let nvcc keep more per thread, which
// dropped resident threads per SM, and this kernel is latency-bound -- it needs
// the occupancy far more than it needs to avoid the spill.
//
// So the knobs stay, defaulted OFF: the stock nvcc heuristic wins. The one
// direction NOT yet tried is the tightening one -- PMCUDA_USE_LAUNCH_BOUNDS=1
// with PMCUDA_MIN_BLOCKS=2 at BLOCK_H=16 pins 2 blocks x 512 threads = full
// Turing occupancy at 64 registers/thread. That is the way the evidence points.
//
// Neither knob changes results. Every pixel in a checkerboard phase reads only
// opposite-parity cells written by the previous phase, and the RNG is seeded from
// the pixel index, so which thread owns a pixel is irrelevant.
// ---------------------------------------------------------------------
#ifndef PMCUDA_BLOCK_W
#define PMCUDA_BLOCK_W 32
#endif
#ifndef PMCUDA_BLOCK_H
#define PMCUDA_BLOCK_H 16
#endif
#ifndef PMCUDA_USE_LAUNCH_BOUNDS
#define PMCUDA_USE_LAUNCH_BOUNDS 0
#endif
#ifndef PMCUDA_MIN_BLOCKS
#define PMCUDA_MIN_BLOCKS 0
#endif
#if !PMCUDA_USE_LAUNCH_BOUNDS
#define PMCUDA_LAUNCH_BOUNDS
#elif PMCUDA_MIN_BLOCKS > 0
#define PMCUDA_LAUNCH_BOUNDS __launch_bounds__(PMCUDA_BLOCK_W*PMCUDA_BLOCK_H, PMCUDA_MIN_BLOCKS)
#else
#define PMCUDA_LAUNCH_BOUNDS __launch_bounds__(PMCUDA_BLOCK_W*PMCUDA_BLOCK_H)
#endif

#if PMCUDA_OPT_FAST_INTRIN
	#define PM_EXPF(x)   __expf(x)
	#define PM_SQRTF(x)  sqrtf(x)
	#define PM_SINF(x)   __sinf(x)
	#define PM_COSF(x)   __cosf(x)
#else
	#define PM_EXPF(x)   exp(x)
	#define PM_SQRTF(x)  sqrt(x)
	#define PM_SINF(x)   sin(x)
	#define PM_COSF(x)   cos(x)
#endif

#if PMCUDA_OPT_RESTRICT
	#define PM_RESTRICT __restrict__
#else
	#define PM_RESTRICT
#endif

// static max supported views
#define MAX_VIEWS 32

// samples used to perform views selection
#define NUM_SAMPLES 32

// patch window radius
#define nSizeHalfWindow 4

// patch stepping
#define nSizeStep 2

// number of samples the strided patch window visits: the loops run
// -nSizeHalfWindow..+nSizeHalfWindow step nSizeStep in each axis
#define nSizeTexels (((nSizeHalfWindow*2)/nSizeStep + 1) * ((nSizeHalfWindow*2)/nSizeStep + 1))


namespace MVS {

namespace CUDA {

#define ImagePixels cudaTextureObject_t

// ---------------------------------------------------------------------
// Stateless per-pixel RNG.
//
// The original code kept one curandState (XORWOW) per pixel in global memory:
// 48 B/pixel of VRAM, an expensive curand_init() sweep per scale, and a global
// load+store around every draw. PatchMatch only needs decorrelated uniforms,
// so each kernel instance derives a small register-resident xorshift32 state
// from (pixel index, launch salt) through an integer finalizer. Determinism is
// preserved (same inputs -> same outputs) and, because the seed depends only
// on the pixel, results are independent of the TDR row-banding. The draw
// SEQUENCE differs from curand's, so outputs differ from the old build the
// same way a different fixed seed would: statistically equivalent.
// ---------------------------------------------------------------------
struct RNGState {
	uint32_t s;
};
__device__ inline RNGState MakeRNG(int idx, uint32_t salt) {
	uint32_t x = (uint32_t)idx * 0x9E3779B9u + salt * 0x85EBCA6Bu + 0x1234567u;
	// splitmix32-style finalizer: adjacent (idx,salt) values map to well-mixed seeds
	x ^= x >> 16; x *= 0x7FEB352Du;
	x ^= x >> 15; x *= 0x846CA68Bu;
	x ^= x >> 16;
	RNGState r; r.s = x | 1u; // xorshift32 state must be non-zero
	return r;
}
__device__ inline uint32_t RandNext(RNGState& r) {
	uint32_t x = r.s;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return r.s = x;
}
// uniform float in (0,1], matching curand_uniform()'s range
__device__ inline float RandUniform(RNGState& r) {
	return 1.f - (__uint_as_float(0x3F800000u | (RandNext(r) >> 9)) - 1.f);
}
#define RandState RNGState

// set/check a bit
__device__ constexpr void SetBit(unsigned& input, unsigned i) {
	input |= (1u << i);
}
__device__ constexpr int IsBitSet(unsigned input, unsigned i) {
	return (input >> i) & 1u;
}

// sort the given values array using bubble sort algorithm
__device__ inline void Sort(const float* values, float* sortedValues, int n) {
	for (int i = 0; i < n; ++i)
		sortedValues[i] = values[i];
	do {
		int newn = 0;
		for (int i = 1; i < n; ++i) {
			if (sortedValues[i-1] > sortedValues[i]) {
				Swap(sortedValues[i-1], sortedValues[i]);
				newn = i;
			}
		}
		n = newn;
	} while(n);
}

// find the index of the minimum value in the given values array
__device__ inline int FindMinIndex(const float* values, const int n) {
	float minValue = values[0];
	int minValueIdx = 0;
	for (int i = 1; i < n; ++i) {
		if (minValue > values[i]) {
			minValue = values[i];
			minValueIdx = i;
		}
	}
	return minValueIdx;
}

// convert Probability Density Function (PDF) to Cumulative Distribution Function (CDF)
__device__ inline void PDF2CDF(float* probs, const int numProbs) {
	float probSum = 0.f;
	for (int i = 0; i < numProbs; ++i)
		probSum += probs[i];
	const float invProbSum = 1.f / probSum;
	float sumProb = 0.f;
	for (int i = 0; i < numProbs-1; ++i) {
		sumProb += probs[i] * invProbSum;
		probs[i] = sumProb;
	}
	probs[numProbs-1] = 1.f;
}
/*----------------------------------------------------------------*/


// generate a random normal
__device__ inline Point3 GenerateRandomNormal(const CUDA::Camera& camera, const Point2i& p, RandState& randState)
{
	float q1, q2, s;
	do {
		q1 = 2.f * RandUniform(randState) - 1.f;
		q2 = 2.f * RandUniform(randState) - 1.f;
		s = q1 * q1 + q2 * q2;
	} while (s >= 1.f);
	const float sq = PM_SQRTF(1.f - s);
	Point3 normal(
		2.f * q1 * sq,
		2.f * q2 * sq,
		1.f - 2.f * s);

	const Point3 viewDirection = camera.model.ViewDirection(p);
	if (normal.dot(viewDirection) > 0.f)
		normal = -normal;
	return normal.normalized();
}

// randomly perturb a normal
__device__ inline Point3 GeneratePerturbedNormal(const CUDA::Camera& camera, const Point2i& p, const Point3& normal, RandState& randState, const float perturbation)
{
	const Point3 viewDirection = camera.model.ViewDirection(p);

	const float a1 = (RandUniform(randState) - 0.5f) * perturbation;
	const float a2 = (RandUniform(randState) - 0.5f) * perturbation;
	const float a3 = (RandUniform(randState) - 0.5f) * perturbation;

	const float sinA1 = PM_SINF(a1);
	const float sinA2 = PM_SINF(a2);
	const float sinA3 = PM_SINF(a3);
	const float cosA1 = PM_COSF(a1);
	const float cosA2 = PM_COSF(a2);
	const float cosA3 = PM_COSF(a3);

	Matrix3 perturb; perturb <<
		cosA2 * cosA3,
		cosA3 * sinA1 * sinA2 - cosA1 * sinA3,
		sinA1 * sinA3 + cosA1 * cosA3 * sinA2,
		cosA2 * sinA3,
		cosA1 * cosA3 + sinA1 * sinA2 * sinA3,
		cosA1 * sinA2 * sinA3 - cosA3 * sinA1,
		-sinA2,
		cosA2 * sinA1,
		cosA1 * cosA2;

	Point3 normalPerturbed = perturb * normal.topLeftCorner<3,1>();
	if (normalPerturbed.dot(viewDirection) >= 0.f)
		return normal;
	return normalPerturbed.normalized();
}

// randomly perturb a depth
__device__ inline float GeneratePerturbedDepth(float depth, RandState& randState, const float perturbation, const PatchMatch::Params& params)
{
	const float depthMinPerturbed = (1.f - perturbation) * depth;
	const float depthMaxPerturbed = (1.f + perturbation) * depth;
	float depthPerturbed;
	do {
		depthPerturbed = RandUniform(randState) * (depthMaxPerturbed - depthMinPerturbed) + depthMinPerturbed;
	// NOTE: '&&' is unsatisfiable (a value cannot be below fDepthMin AND above
	// fDepthMax), so this loop always exits after one draw -- upstream behavior,
	// kept verbatim because "fixing" it to '||' would change the random sequence
	// and therefore the results
	} while (depthPerturbed < params.fDepthMin && depthPerturbed > params.fDepthMax);
	return depthPerturbed;
}

// interpolate given pixel's estimate to the current position
__device__ inline float InterpolatePixel(const CUDA::Camera& camera, const Point2i& p, const Point2i& np, float depth, const Point3& normal, const PatchMatch::Params& params)
{
	float depthNew;
	if (p.x() == np.x()) {
		const float nx1 = (p.y() - camera.model.p.y()) / camera.model.f.y();
		const float denom = normal.z() + nx1 * normal.y();
		if (abs(denom) < FLT_EPSILON)
			return depth;
		const float x1 = (np.y() - camera.model.p.y()) / camera.model.f.y();
		const float nom = depth * (normal.z() + x1 * normal.y());
		depthNew = nom / denom;
	} else if (p.y() == np.y()) {
		const float nx1 = (p.x() - camera.model.p.x()) / camera.model.f.x();
		const float denom = normal.z() + nx1 * normal.x();
		if (abs(denom) < FLT_EPSILON)
			return depth;
		const float x1 = (np.x() - camera.model.p.x()) / camera.model.f.x();
		const float nom = depth * (normal.z() + x1 * normal.x());
		depthNew = nom / denom;
	} else {
		const float planeD = normal.dot(camera.model.TransformPointI2C(np.cast<float>(), depth));
		depthNew = planeD / normal.dot(camera.model.TransformPointI2C(p.cast<float>()));
	}
	return (depthNew >= params.fDepthMin && depthNew <= params.fDepthMax) ? depthNew : depth;
}

// compute normal to the surface given the 4 neighbors
__device__ inline Point3 ComputeDepthGradient(const LinearCameraModel& model, float depth, const Point2i& pos, const Point4& ndepth) {
	constexpr float2 nposg[4] = {{0,-1}, {0,1}, {-1,0}, {1,0}};
	Point2 dg(0,0);
	// add neighbor depths at the gradient locations
	for (int i=0; i<4; ++i)
		dg += Point2(nposg[i].x,nposg[i].y) * (ndepth[i] - depth);
	// compute depth gradient
	const Point2 d = dg*0.5f;
	// compute normal from depth gradient
	return Point3(
		model.f.x()*d.x(),
		model.f.y()*d.y(),
		(model.p.x()-pos.x())*d.x()+(model.p.y()-pos.y())*d.y()-depth).normalized();
}

// compose the homography that maps a reference-image point to the source image
// through the given plane, from the precomputed per-pair constants:
//   H = A + b * (n^T K0^-1) / (n . X(p,d))
// where A = K1*R1*R0^T*K0^-1 and b = K1*R1*(C0-C1) (see PairConstants); this is
// algebraically identical to the former per-call K1*(R1*R0^T + t*n^T)*K0^-1
// (which re-derived K0^-1 and two full 3x3 products for EVERY plane scored)
__device__ inline Matrix3 ComputeHomography(const PatchMatch::PairConstants& pair, const LinearCameraModel& refModel, const Point2& p, const Point4& plane)
{
	const Point3 n = plane.topLeftCorner<3,1>();
	const Point3 X = refModel.TransformPointI2C(p, plane.w());
	const float s = 1.f / n.dot(X);
	// kn = n^T * K0^-1, exploiting K0's upper-triangular structure
	const Point3 kn(
		n.x() / refModel.f.x(),
		n.y() / refModel.f.y(),
		n.z() - (n.x() * refModel.p.x()) / refModel.f.x() - (n.y() * refModel.p.y()) / refModel.f.y());
	const Point3 bs(pair.b[0] * s, pair.b[1] * s, pair.b[2] * s);
	Matrix3 H; H <<
		pair.A[0] + bs.x() * kn.x(), pair.A[1] + bs.x() * kn.y(), pair.A[2] + bs.x() * kn.z(),
		pair.A[3] + bs.y() * kn.x(), pair.A[4] + bs.y() * kn.y(), pair.A[5] + bs.y() * kn.z(),
		pair.A[6] + bs.z() * kn.x(), pair.A[7] + bs.z() * kn.y(), pair.A[8] + bs.z() * kn.z();
	return H;
}

// weight a neighbor texel based on color similarity and distance to the center texel
__device__ inline float ComputeBilateralWeight(int xDist, int yDist, float pix, float centerPix)
{
	constexpr float sigmaSpatial = -1.f / (2.f * (nSizeHalfWindow-1)*(nSizeHalfWindow-1));
	constexpr float sigmaColor = -1.f / (2.f * 25.f/255.f*25.f/255.f);
	const float spatialDistSq = float(xDist * xDist + yDist * yDist);
	const float colorDistSq = Square(pix - centerPix);
	return PM_EXPF(spatialDistSq * sigmaSpatial + colorDistSq * sigmaColor);
}

// ---------------------------------------------------------------------
// Reference-patch invariants.
//
// Every quantity ScorePlane derives from the REFERENCE image depends only on the
// pixel p -- not on the plane hypothesis and not on the target view. ProcessPixel
// evaluates 13 hypotheses (8 checkerboard neighbors + the current plane + up to 4
// refinements) against nNumViews targets, so the reference half of the patch loop
// used to be recomputed 13*nNumViews times per pixel per iteration when one
// computation would do: at the default 8 views that is 104 repetitions of 25
// texture fetches and 25 bilateral exp() calls.
//
// Building it once and passing it in leaves the arithmetic untouched -- the sums
// are accumulated over the same texels in the same order, so they are bit-identical
// to the fused loop -- and removes roughly half of the kernel's texture traffic
// along with nearly all of its exp() calls.
// ---------------------------------------------------------------------
struct RefPatch {
	float weight[nSizeTexels];       // bilateral weight per texel
	float weightRefPix[nSizeTexels]; // weight * reference pixel
	float sumRef;                    // sum of weightRefPix
	float bilateralWeightSum;        // sum of weight
	float varRef;                    // sumRefRef*bilateralWeightSum - sumRef^2
	float factorDeltaDepth;          // exp(varRef * smoothSigmaDepth); only read when lowDepth > 0
};

// Fill the invariants for pixel p. Texel order matches ScorePlane's loop exactly,
// which is what keeps the accumulated sums bit-identical to the original.
__device__ inline void BuildRefPatch(const ImagePixels refImage, const Point2i& p, const float lowDepth, RefPatch& rp)
{
	float sumRef = 0.f, sumRefRef = 0.f, bilateralWeightSum = 0.f;
	const float refCenterPix = tex2D<float>(refImage, p.x() + 0.5f, p.y() + 0.5f);
	int k = 0;
	#pragma unroll
	for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
		#pragma unroll
		for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep, ++k) {
			const Point2i refPt = Point2i(p.x() + j, p.y() + i);
			const float refPix = tex2D<float>(refImage, refPt.x() + 0.5f, refPt.y() + 0.5f);
			const float weight = ComputeBilateralWeight(j, i, refPix, refCenterPix);
			const float weightRefPix = weight * refPix;
			rp.weight[k] = weight;
			rp.weightRefPix[k] = weightRefPix;
			sumRef += weightRefPix;
			sumRefRef += weightRefPix * refPix;
			bilateralWeightSum += weight;
		}
	}
	rp.sumRef = sumRef;
	rp.bilateralWeightSum = bilateralWeightSum;
	rp.varRef = sumRefRef * bilateralWeightSum - sumRef * sumRef;
	// 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
	constexpr float smoothSigmaDepth(-1.f / (1.f * 0.02f));
	rp.factorDeltaDepth = lowDepth > 0 ? PM_EXPF(rp.varRef * smoothSigmaDepth) : 0.f;
}

// compute the geometric consistency weight
__device__ inline float GeometricConsistencyWeight(const ImagePixels depthImage, const CUDA::Camera& refCamera, const CUDA::Camera& trgCamera, const Point4& plane, const Point2i& p)
{
	if (depthImage == NULL)
		return 0.f;
	constexpr float maxDist = 4.f;
	const Point3 forwardPoint = refCamera.TransformPointI2W(p.cast<float>(), plane.w());
	const Point2 trgPt = trgCamera.TransformPointW2I(forwardPoint);
	const float trgDepth = tex2D<float>(depthImage, trgPt.x() + 0.5f, trgPt.y() + 0.5f);
	if (trgDepth == 0.f)
		return maxDist;
	const Point3 trgX = trgCamera.TransformPointI2W(trgPt, trgDepth);
	const Point2 backwardPoint = refCamera.TransformPointW2I(trgX);
	const Point2 diff = p.cast<float>() - backwardPoint;
	const float dist = diff.norm();
	return min(maxDist, PM_SQRTF(dist*(dist+2.f)));
}

// compute photometric score using weighted ZNCC
__device__ float ScorePlane(const CUDA::Camera& refCamera, const ImagePixels trgImage, const CUDA::Camera& trgCamera, const PatchMatch::PairConstants& pair, const Point2i& p, const Point4& plane, const float lowDepth, const RefPatch& rp, const PatchMatch::Params& params)
{
	constexpr float maxCost = 1.2f;

	// Textureless reference patch. The original reached this test only after running
	// the whole sample loop, but varRef depends on neither the plane nor the target
	// view, so the outcome is the same -- and both this and the out-of-bounds test
	// below return maxCost, so hoisting it above them cannot change the result. It
	// now costs nothing: such a pixel issues no target fetch at all.
	if (lowDepth <= 0 && rp.varRef < 1e-8f)
		return maxCost;

	Matrix3 H = ComputeHomography(pair, refCamera.model, p.cast<float>(), plane);
	const Point2 pt = (H * p.cast<float>().homogeneous()).hnormalized();
	if (pt.x() >= trgCamera.size.x() || pt.x() < 0.f || pt.y() >= trgCamera.size.y() || pt.y() < 0.f)
		return maxCost;
	Point3 X = H * Point2(p.x()-nSizeHalfWindow, p.y()-nSizeHalfWindow).homogeneous();
	Point3 baseX(X);
	H *= float(nSizeStep);

	// only the target-dependent sums remain; the reference ones come from rp,
	// accumulated over these same texels in this same order
	float sumTrg = 0.f;
	float sumTrgTrg = 0.f;
	float sumRefTrg = 0.f;
	int k = 0;
	#pragma unroll
	for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
		#pragma unroll
		for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep, ++k) {
			const Point2 trgPt = X.hnormalized();
			const float trgPix = tex2D<float>(trgImage, trgPt.x() + 0.5f, trgPt.y() + 0.5f);
			const float weightTrgPix = rp.weight[k] * trgPix;
			sumTrg += weightTrgPix;
			sumTrgTrg += weightTrgPix * trgPix;
			sumRefTrg += rp.weightRefPix[k] * trgPix;
			X += H.col(0);
		}
		baseX += H.col(1);
		X = baseX;
	}

	const float varTrg = sumTrgTrg * rp.bilateralWeightSum - sumTrg * sumTrg;
	const float varRefTrg = rp.varRef * varTrg;
	if (varRefTrg < 1e-16f)
		return maxCost;
	const float covarTrgRef = sumRefTrg * rp.bilateralWeightSum - rp.sumRef * sumTrg;
	float ncc = 1.f - covarTrgRef / PM_SQRTF(varRefTrg);

	// apply depth prior weight based on patch textureless
	if (lowDepth > 0) {
		const float depth(plane.w());
		const float deltaDepth(MIN((abs(lowDepth-depth) / lowDepth), 0.5f));
		ncc = (1.f-rp.factorDeltaDepth)*ncc + rp.factorDeltaDepth*deltaDepth;
	}
	return max(0.f, min(2.f, ncc));
}

// compute photometric score for all neighbor images
__device__ inline void MultiViewScorePlane(const ImagePixels* images, const ImagePixels* depthImages, const CUDA::Camera* cameras, const PatchMatch::PairConstants* pairs, const Point2i& p, const Point4& plane, const float lowDepth, const RefPatch& rp, float* costVector, const PatchMatch::Params& params)
{
	for (int imgId = 1; imgId <= params.nNumViews; ++imgId)
		costVector[imgId-1] = ScorePlane(cameras[0], images[imgId], cameras[imgId], pairs[imgId-1], p, plane, lowDepth, rp, params);
	if (params.bGeomConsistency)
		for (int imgId = 0; imgId < params.nNumViews; ++imgId)
			costVector[imgId] += 0.1f * GeometricConsistencyWeight(depthImages[imgId], cameras[0], cameras[imgId+1], plane, p);
}
// same as above, but interpolate the plane to current pixel position
__device__ inline float MultiViewScoreNeighborPlane(const ImagePixels* images, const ImagePixels* depthImages, const CUDA::Camera* cameras, const PatchMatch::PairConstants* pairs, const Point2i& p, const Point2i& np, Point4 plane, const float lowDepth, const RefPatch& rp, float* costVector, const PatchMatch::Params& params)
{
	plane.w() = InterpolatePixel(cameras[0], p, np, plane.w(), plane.topLeftCorner<3,1>(), params);
	MultiViewScorePlane(images, depthImages, cameras, pairs, p, plane, lowDepth, rp, costVector, params);
	return plane.w();
}

// aggregate photometric score from all images
__device__ inline float AggregateMultiViewScores(const unsigned* viewWeights, const float* costVector, int numViews)
{
	float cost = 0;
	for (int imgId = 0; imgId < numViews; ++imgId)
		if (viewWeights[imgId])
			cost += viewWeights[imgId] * costVector[imgId];
	return cost / NUM_SAMPLES;
}

// propagate and refine the plane estimate for the current pixel employing the asymmetric approach described in:
// "Multi-View Stereo with Asymmetric Checkerboard Propagation and Multi-Hypothesis Joint View Selection", 2018
// MAXV bounds the per-thread scratch arrays: they are dynamically indexed, hence
// local-memory resident, and sizing them to the true view count (<=8 in the
// default configuration) instead of the static MAX_VIEWS cuts the spill
// footprint ~4x, which is what limits this kernel
template <int MAXV>
__device__ void ProcessPixel(const ImagePixels* images, const ImagePixels* depthImages, const CUDA::Camera* cameras, const PatchMatch::PairConstants* pairs, Point4* planes, const float* lowDepths, float* costs, unsigned* selectedViews, const Point2i& p, const PatchMatch::Params& params, const int iter, const uint32_t rngSalt)
{
	const int width = cameras[0].size.x();
	const int height = cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	RandState randState = MakeRNG(idx, rngSalt);
	float lowDepth = 0;
	if (params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	// Every hypothesis scored below -- 8 checkerboard neighbors, the current plane
	// and up to 4 refinements -- shares this pixel's reference patch, and so does
	// every view within each of them. Building it once here is what collapses
	// 13*nNumViews reconstructions of it into one.
	RefPatch rp;
	BuildRefPatch(images[0], p, lowDepth, rp);

	// adaptive sampling: 0 up-near, 1 down-near, 2 left-near, 3 right-near, 4 up-far, 5 down-far, 6 left-far, 7 right-far
	static constexpr int2 dirs[8][11] = {
		{{ 0,-1},{-1,-2},{ 1,-2},{-2,-3},{ 2,-3},{-3,-4},{ 3,-4}},
		{{ 0, 1},{-1, 2},{ 1, 2},{-2, 3},{ 2, 3},{-3, 4},{ 3, 4}},
		{{-1, 0},{-2,-1},{-2, 1},{-3,-2},{-3, 2},{-4,-3},{-4, 3}},
		{{ 1, 0},{ 2,-1},{ 2, 1},{ 3,-2},{ 3, 2},{ 4,-3},{ 4, 3}},
		{{0,-3},{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-23}},
		{{0, 3},{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 23}},
		{{-3,0},{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-23,0}},
		{{ 3,0},{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 23,0}}
	};
	static constexpr int numDirs[8] = {7, 7, 7, 7, 11, 11, 11, 11};
	const int neighborPositions[4] = {
		idx - width,
		idx + width,
		idx - 1,
		idx + 1,
	};
	bool valid[8] = {false, false, false, false, false, false, false, false};
	int positions[8];
	float neighborDepths[8];
	float costArray[8][MAXV];

	for (int posId=0; posId<8; ++posId) {
		const int2* samples = dirs[posId];
		Point2i bestNx; float bestConf(FLT_MAX);
		for (int dirId=0; dirId<numDirs[posId]; ++dirId) {
			const int2& offset = samples[dirId];
			const Point2i np(p.x()+offset.x, p.y()+offset.y);
			if (!(np.x()>=0 && np.y()>=0 && np.x()<width && np.y()<height))
				continue;
			const int nidx = Point2Idx(np, width);
			const float nconf = costs[nidx];
			if (bestConf > nconf) {
				bestNx = np;
				bestConf = nconf;
			}
		}
		if (bestConf < FLT_MAX) {
			valid[posId] = true;
			positions[posId] = Point2Idx(bestNx, width);
			neighborDepths[posId] = MultiViewScoreNeighborPlane(images, depthImages, cameras, pairs, p, bestNx, planes[positions[posId]], lowDepth, rp, costArray[posId], params);
		}
	}

	// multi-hypothesis view selection
	float viewSelectionPriors[MAXV] = {};
	for (int posId = 0; posId < 4; ++posId) {
		if (valid[posId]) {
			const unsigned selectedView = selectedViews[neighborPositions[posId]];
			for (int j = 0; j < params.nNumViews; ++j)
				viewSelectionPriors[j] += (IsBitSet(selectedView, j) ? 0.9f : 0.1f);
		}
	}
	float samplingProbs[MAXV];
	constexpr float thCostBad = 1.2f;
	const float thCost = 0.8f * PM_EXPF(Square((float)iter) / (-2.f * 4.f*4.f));
	for (int imgId = 0; imgId < params.nNumViews; ++imgId) {
		float sumW = 0;
		unsigned count = 0;
		unsigned countBad = 0;
		for (int posId = 0; posId < 8; posId++) {
			if (valid[posId]) {
				if (costArray[posId][imgId] < thCost) {
					sumW += PM_EXPF(Square(costArray[posId][imgId]) / (-2.f * 0.3f*0.3f));
					++count;
				} else if (costArray[posId][imgId] > thCostBad) {
					++countBad;
				}
			}
		}
		if (count > 2 && countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * sumW / count;
		} else if (countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * PM_EXPF(Square(thCost) / (-2.f * 0.4f*0.4f));
		} else {
			samplingProbs[imgId] = 0.f;
		}
	}
	PDF2CDF(samplingProbs, params.nNumViews);
	unsigned viewWeights[MAXV] = {};
	for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
		const float randProb = RandUniform(randState);
		for (int imgId = 0; imgId < params.nNumViews; ++imgId) {
			if (samplingProbs[imgId] > randProb) {
				++viewWeights[imgId];
				break;
			}
		}
	}

	// propagate best neighbor plane
	Point4& plane = planes[idx];
	float& cost = costs[idx];
	unsigned newSelectedViews = 0;
	for (int imgId = 0; imgId < params.nNumViews; ++imgId)
		if (viewWeights[imgId])
			SetBit(newSelectedViews, imgId);
	float finalCosts[8];
	for (int posId = 0; posId < 8; ++posId)
		finalCosts[posId] = AggregateMultiViewScores(viewWeights, costArray[posId], params.nNumViews);
	const int minCostIdx = FindMinIndex(finalCosts, 8);
	float costVector[MAXV];
	MultiViewScorePlane(images, depthImages, cameras, pairs, p, plane, lowDepth, rp, costVector, params);
	cost = AggregateMultiViewScores(viewWeights, costVector, params.nNumViews);
	if (finalCosts[minCostIdx] < cost && valid[minCostIdx]) {
		plane = planes[positions[minCostIdx]];
		plane.w() = neighborDepths[minCostIdx];
		cost = finalCosts[minCostIdx];
		selectedViews[idx] = newSelectedViews;
	}
	const float depth = plane.w();

	// refine estimate
	constexpr float perturbationDepth = 0.005f;
	constexpr float perturbationNormal = 0.01f * (float)M_PI;
	const float depthPerturbed = GeneratePerturbedDepth(depth, randState, perturbationDepth, params);
	const Point3 perturbedNormal = GeneratePerturbedNormal(cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal);
	const Point3 normalRand = GenerateRandomNormal(cameras[0], p, randState);
	int numValidPlanes = 3;
	Point3 surfaceNormal;
	if (valid[0] && valid[1] && valid[2] && valid[3]) {
		// estimate normal from surrounding surface
		const Point4 ndepths(
			planes[neighborPositions[0]].w(),
			planes[neighborPositions[1]].w(),
			planes[neighborPositions[2]].w(),
			planes[neighborPositions[3]].w()
		);
		surfaceNormal = ComputeDepthGradient(cameras[0].model, depth, p, ndepths);
		numValidPlanes = 4;
	}
	constexpr int numPlanes = 4;
	const float depths[numPlanes] = {depthPerturbed, depth, depth, depth};
	const Point3 normals[numPlanes] = {plane.topLeftCorner<3,1>(), perturbedNormal, normalRand, surfaceNormal};
	for (int i = 0; i < numValidPlanes; ++i) {
		Point4 newPlane;
		newPlane.topLeftCorner<3,1>() = normals[i];
		newPlane.w() = depths[i];
		MultiViewScorePlane(images, depthImages, cameras, pairs, p, newPlane, lowDepth, rp, costVector, params);
		const float costPlane = AggregateMultiViewScores(viewWeights, costVector, params.nNumViews);
		if (cost > costPlane) {
			cost = costPlane;
			plane = newPlane;
		}
	}
}

// compute the score of the current plane estimate
template <int MAXV>
__device__ void InitializePixelScore(const ImagePixels *images, const ImagePixels* depthImages, const CUDA::Camera* cameras, const PatchMatch::PairConstants* pairs, Point4* planes, const float* lowDepths, float* costs, unsigned* selectedViews, const Point2i& p, const PatchMatch::Params params)
{
	const int width = cameras[0].size.x();
	const int height = cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	float lowDepth = 0;
	if (params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	// initialize estimate randomly if not set
	RandState randState = MakeRNG(idx, 0u);
	Point4& plane = planes[idx];
	float depth = plane.w();
	if (depth <= 0.f) {
		// generate random plane
		plane.topLeftCorner<3,1>() = GenerateRandomNormal(cameras[0], p, randState);
		plane.w() = RandUniform(randState) * (params.fDepthMax - params.fDepthMin) + params.fDepthMin;
	} else if (plane.topLeftCorner<3,1>().dot(cameras[0].model.ViewDirection(p)) >= 0.f) {
		// generate random normal
		plane.topLeftCorner<3,1>() = GenerateRandomNormal(cameras[0], p, randState);
	}
	// compute costs
	RefPatch rp;
	BuildRefPatch(images[0], p, lowDepth, rp);
	float costVector[MAXV];
	MultiViewScorePlane(images, depthImages, cameras, pairs, p, plane, lowDepth, rp, costVector, params);
	// select best views
	float costVectorSorted[MAXV];
	Sort(costVector, costVectorSorted, params.nNumViews);
	float cost = 0.f;
	for (int i = 0; i < params.nInitTopK; ++i)
		cost += costVectorSorted[i];
	const float costThreshold = costVectorSorted[params.nInitTopK - 1];
	unsigned& selectedView = selectedViews[idx];
	selectedView = 0;
	for (int imgId = 0; imgId < params.nNumViews; ++imgId)
		if (costVector[imgId] <= costThreshold)
			SetBit(selectedView, imgId);
	costs[idx] = cost / params.nInitTopK;
}
template <int MAXV>
__global__ void PMCUDA_LAUNCH_BOUNDS InitializeScore(const cudaTextureObject_t* PM_RESTRICT textureImages, const cudaTextureObject_t* PM_RESTRICT textureDepths, const CUDA::Camera* PM_RESTRICT cameras, const PatchMatch::PairConstants* PM_RESTRICT pairs, Point4* PM_RESTRICT planes, const float* PM_RESTRICT lowDepths, float* PM_RESTRICT costs, unsigned* PM_RESTRICT selectedViews, const PatchMatch::Params params, const int yBase)
{
	Point2i p = GetThreadIndex2();
	p.y() += yBase;
	InitializePixelScore<MAXV>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, cameras, pairs, planes, lowDepths, costs, selectedViews, p, params);
}

// traverse image in a black/red checkerboard pattern
// yBase is a half-row offset used by the TDR row-banding (see
// PMCUDA_TDR_MAX_PIXEL_VIEWS above); the RNG seed depends only on the pixel,
// so results are independent of how the frame is split into bands
template <int MAXV>
__global__ void PMCUDA_LAUNCH_BOUNDS BlackPixelProcess(const cudaTextureObject_t* PM_RESTRICT textureImages, const cudaTextureObject_t* PM_RESTRICT textureDepths, const CUDA::Camera* PM_RESTRICT cameras, const PatchMatch::PairConstants* PM_RESTRICT pairs, Point4* PM_RESTRICT planes, const float* PM_RESTRICT lowDepths, float* PM_RESTRICT costs, unsigned* PM_RESTRICT selectedViews, const PatchMatch::Params params, const int iter, const int yBase)
{
	Point2i p = GetThreadIndex2();
	p.y() = (p.y() + yBase) * 2 + (threadIdx.x % 2 == 0 ? 0 : 1);
	ProcessPixel<MAXV>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, cameras, pairs, planes, lowDepths, costs, selectedViews, p, params, iter, 1u + 2u * (uint32_t)iter);
}
template <int MAXV>
__global__ void PMCUDA_LAUNCH_BOUNDS RedPixelProcess(const cudaTextureObject_t* PM_RESTRICT textureImages, const cudaTextureObject_t* PM_RESTRICT textureDepths, const CUDA::Camera* PM_RESTRICT cameras, const PatchMatch::PairConstants* PM_RESTRICT pairs, Point4* PM_RESTRICT planes, const float* PM_RESTRICT lowDepths, float* PM_RESTRICT costs, unsigned* PM_RESTRICT selectedViews, const PatchMatch::Params params, const int iter, const int yBase)
{
	Point2i p = GetThreadIndex2();
	p.y() = (p.y() + yBase) * 2 + (threadIdx.x % 2 == 0 ? 1 : 0);
	ProcessPixel<MAXV>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, cameras, pairs, planes, lowDepths, costs, selectedViews, p, params, iter, 2u + 2u * (uint32_t)iter);
}

// filter depth/normals
__global__ void FilterPlanes(Point4* PM_RESTRICT planes, float* PM_RESTRICT costs, unsigned* PM_RESTRICT selectedViews, int width, int height, const PatchMatch::Params params)
{
	const Point2i p = GetThreadIndex2();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	// filter estimates if the score is not good enough
	Point4& plane = planes[idx];
	float conf = costs[idx];
	if (plane.w() <= 0 || conf >= params.fThresholdKeepCost) {
		conf = 0;
		plane = Point4::Zero();
		selectedViews[idx] = 0;
	}
}
/*----------------------------------------------------------------*/


// convert a linear fp32 image to fp16 before it is copied into a texture array
__global__ void KConvertFloatToHalf(const float* PM_RESTRICT src, __half* PM_RESTRICT dst, const int count)
{
	const int i = GetThreadIndex();
	if (i < count)
		dst[i] = __float2half(src[i]);
}

// fuse the uploaded depth and normal maps into the per-pixel plane estimates
// (replaces the serial host-side pack loop)
__global__ void KPackPlanes(const float* PM_RESTRICT depthIn, const float* PM_RESTRICT normalIn, Point4* PM_RESTRICT planes, const int count)
{
	const int i = GetThreadIndex();
	if (i >= count)
		return;
	Point4 dn;
	dn.x() = normalIn[i*3+0];
	dn.y() = normalIn[i*3+1];
	dn.z() = normalIn[i*3+2];
	dn.w() = depthIn[i];
	planes[i] = dn;
}

// split the plane estimates back into depth/normal maps and, on the final
// scale, convert the ZNCC cost to a confidence and the selected-views bit-mask
// to the up-to-4 view indices of the ViewsID layout
// (replaces the serial host-side unpack loop)
__global__ void KUnpackResults(const Point4* PM_RESTRICT planes, const float* PM_RESTRICT costs, const unsigned* PM_RESTRICT views, float* PM_RESTRICT depthOut, float* PM_RESTRICT normalOut, float* PM_RESTRICT confOut, uint8_t* PM_RESTRICT viewsOut, const int count, const int maxPixelViews)
{
	const int i = GetThreadIndex();
	if (i >= count)
		return;
	const Point4 dn = planes[i];
	const float depth = dn.w();
	depthOut[i] = depth;
	normalOut[i*3+0] = dn.x();
	normalOut[i*3+1] = dn.y();
	normalOut[i*3+2] = dn.z();
	if (confOut) {
		// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
		const float c = costs[i];
		confOut[i] = (c >= 1.f ? 0.f : 1.f - c);
	}
	if (viewsOut) {
		// map pixel views from bit-mask to index
		uint8_t v[4] = {255, 255, 255, 255};
		if (depth > 0) {
			const unsigned bits = views[i];
			int j = 0;
			for (int k = 0; k < 32; ++k) {
				if (bits & (1u << k)) {
					v[j] = (uint8_t)k;
					if (++j == maxPixelViews)
						break;
				}
			}
		}
		viewsOut[i*4+0] = v[0];
		viewsOut[i*4+1] = v[1];
		viewsOut[i*4+2] = v[2];
		viewsOut[i*4+3] = v[3];
	}
}
/*----------------------------------------------------------------*/


// host wrappers for the helper kernels (callable from PatchMatchCUDA.cpp)
__host__ void PMConvertFloatToHalf(const float* d_src, void* d_dstHalf, int count, cudaStream_t stream)
{
	constexpr int nThreads = 256;
	KConvertFloatToHalf<<<(count + nThreads - 1) / nThreads, nThreads, 0, stream>>>(d_src, (__half*)d_dstHalf, count);
}

__host__ void PMPackPlanes(const float* d_depthIn, const float* d_normalIn, Point4* d_planes, int count, cudaStream_t stream)
{
	constexpr int nThreads = 256;
	KPackPlanes<<<(count + nThreads - 1) / nThreads, nThreads, 0, stream>>>(d_depthIn, d_normalIn, d_planes, count);
}

__host__ void PMUnpackResults(const Point4* d_planes, const float* d_costs, const unsigned* d_views,
	float* d_depthOut, float* d_normalOut, float* d_confOut, uint8_t* d_viewsOut,
	int count, int maxPixelViews, cudaStream_t stream)
{
	constexpr int nThreads = 256;
	KUnpackResults<<<(count + nThreads - 1) / nThreads, nThreads, 0, stream>>>(d_planes, d_costs, d_views, d_depthOut, d_normalOut, d_confOut, d_viewsOut, count, maxPixelViews);
}
/*----------------------------------------------------------------*/


template <int MAXV>
__host__ void PatchMatch::RunCUDAT(const int width, const int height)
{
	// must match the __launch_bounds__ the kernels were compiled with
	constexpr unsigned BLOCK_W = PMCUDA_BLOCK_W;
	constexpr unsigned BLOCK_H = PMCUDA_BLOCK_H;
	const dim3 blockSize(BLOCK_W, BLOCK_H, 1);
	const unsigned gridW = ((unsigned)width + BLOCK_W - 1) / BLOCK_W;

	// TDR row-banding: bound the work of a single launch by
	// PMCUDA_TDR_MAX_PIXEL_VIEWS (pixels x views); see the note at the top.
	// Band boundaries do not affect results.
	const int64_t pixelViewBudget = (int64_t)PMCUDA_TDR_MAX_PIXEL_VIEWS;
	const int views = params.nNumViews > 1 ? params.nNumViews : 1;

	// full-frame rows per launch for InitializeScore: it evaluates a single
	// hypothesis per pixel (~an order of magnitude lighter than the
	// propagation kernels), so it gets 8x the budget
	int bandRowsInit = (int)((pixelViewBudget * 8) / ((int64_t)width * views));
	if (bandRowsInit < (int)BLOCK_H)
		bandRowsInit = (int)BLOCK_H;
	if (bandRowsInit > height)
		bandRowsInit = height;
	bandRowsInit = ((bandRowsInit + BLOCK_H - 1) / BLOCK_H) * BLOCK_H;
	for (int y = 0; y < height; y += bandRowsInit) {
		const int rows = (height - y < bandRowsInit) ? height - y : bandRowsInit;
		const dim3 grid(gridW, ((unsigned)rows + BLOCK_H - 1) / BLOCK_H, 1);
		InitializeScore<MAXV><<<grid, blockSize, 0, stream>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPairs, cudaPlanes, cudaDepthIn, cudaCosts, cudaViews, params, y);
	}

	// checkerboard rows per launch (each checkerboard row covers 2 image rows);
	// (height+1)/2 instead of the former height/2 so the last image row of an
	// odd-height frame is processed too (it previously never was whenever
	// height/2 happened to be a multiple of the block height)
	const int halfRows = (height + 1) / 2;
	int bandHalfRows = (int)(pixelViewBudget / ((int64_t)width * views * 2));
	if (bandHalfRows < (int)BLOCK_H)
		bandHalfRows = (int)BLOCK_H;
	if (bandHalfRows > halfRows)
		bandHalfRows = halfRows;
	bandHalfRows = ((bandHalfRows + BLOCK_H - 1) / BLOCK_H) * BLOCK_H;

	for (int iter = 0; iter < params.nEstimationIters; ++iter) {
		for (int y = 0; y < halfRows; y += bandHalfRows) {
			const int rows = (halfRows - y < bandHalfRows) ? halfRows - y : bandHalfRows;
			const dim3 grid(gridW, ((unsigned)rows + BLOCK_H - 1) / BLOCK_H, 1);
			BlackPixelProcess<MAXV><<<grid, blockSize, 0, stream>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPairs, cudaPlanes, cudaDepthIn, cudaCosts, cudaViews, params, iter, y);
		}
		for (int y = 0; y < halfRows; y += bandHalfRows) {
			const int rows = (halfRows - y < bandHalfRows) ? halfRows - y : bandHalfRows;
			const dim3 grid(gridW, ((unsigned)rows + BLOCK_H - 1) / BLOCK_H, 1);
			RedPixelProcess<MAXV><<<grid, blockSize, 0, stream>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaPairs, cudaPlanes, cudaDepthIn, cudaCosts, cudaViews, params, iter, y);
		}
	}

	if (params.fThresholdKeepCost > 0) {
		const dim3 grid(gridW, ((unsigned)height + BLOCK_H - 1) / BLOCK_H, 1);
		FilterPlanes<<<grid, blockSize, 0, stream>>>(cudaPlanes, cudaCosts, cudaViews, width, height, params);
	}
}

__host__ void PatchMatch::RunCUDA(const int width, const int height)
{
	// Dispatch on the compile-time bound of the per-thread scratch arrays.
	//
	// MAXV sizes costArray[8][MAXV] -- the largest per-thread array, dynamically
	// indexed and therefore local-memory resident -- plus viewSelectionPriors,
	// samplingProbs, viewWeights and costVector. It therefore drives register
	// pressure and spill directly, which is what limits this kernel. Instantiating
	// the small view counts as well as 8 keeps that footprint proportional to the
	// views actually in use: a run with --number-views 6 (5.9 views/image measured,
	// max 6) was paying 8-view scratch on every thread of every launch.
	//
	// Semantics are unaffected. MAXV only bounds the arrays; every loop still runs
	// to params.nNumViews, and the dispatch guarantees MAXV >= nNumViews.
	if (params.nNumViews <= 4)
		RunCUDAT<4>(width, height);
	else if (params.nNumViews <= 6)
		RunCUDAT<6>(width, height);
	else if (params.nNumViews <= 8)
		RunCUDAT<8>(width, height);
	else
		RunCUDAT<MAX_VIEWS>(width, height);
}
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS

#else

// Variant not selected: this file preprocesses away completely. The typedef
// only keeps the translation unit non-empty (MSVC C4206 / nvcc).
namespace MVS { namespace CUDA { typedef int PatchMatchCUDAKernelsNotBuilt; } }

#endif // !PATCHMATCH_CUDA_LEGACY
