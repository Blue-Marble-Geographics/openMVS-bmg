/*
* SceneRefine.cpp
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

#include "Common.h"
#include "Scene.h"
#include "SceneRefineAVX2.h"
#include <chrono>
#include <functional>
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

using namespace MVS;

namespace {
bool SupportsAVX2()
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
	int cpuInfo[4];
	__cpuid(cpuInfo, 0);
	if (cpuInfo[0] < 7)
		return false;
	__cpuidex(cpuInfo, 1, 0);
	constexpr int avxAndOSXSAVE = (1 << 28) | (1 << 27);
	if ((cpuInfo[2] & avxAndOSXSAVE) != avxAndOSXSAVE || (_xgetbv(0) & 0x6) != 0x6)
		return false;
	__cpuidex(cpuInfo, 7, 0);
	return (cpuInfo[1] & (1 << 5)) != 0;
#elif (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
	return __builtin_cpu_supports("avx2");
#else
	return false;
#endif
}
} // namespace

// D E F I N E S ///////////////////////////////////////////////////

#define RASSERT(cond, fmt, ...)                                           \
    do {                                                                    \
      if (!(cond)) {                                                        \
        DEBUG("ASSERT FAILED: %s\n  File: %s:%d\n  Function: %s\n  " fmt "\n",\
              #cond, __FILE__, __LINE__, __func__, ##__VA_ARGS__);         \
        __debugbreak();                                                     \
      }                                                                     \
    } while (0)

#undef VALIDATE_GRADIENT

constexpr int TILEX = 16;
constexpr int TILEY = 16;

// uncomment to ensure edge size and improve vertex valence
// (should enable more stable flow)
#define MESHOPT_ENSUREEDGESIZE 1 // 0 - at all resolution

// uncomment to use constant z-buffer bias
// (should be enough, as the numerical error does not depend on the depth)
// DISABLED (field A/B, Aug 2026): the upstream one-sided 0.05 const-bias warp
// occlusion test looked WORSE here than the two-sided 1% relative band the
// CUDA kernel uses; leave undefined so the warp keeps the GPU semantics.
//#define MESHOPT_DEPTHCONSTBIAS 0.05f

// uncomment to enable memory pool
// (should reduce the allocation times for frequent used images)
#define MESHOPT_TYPEPOOL

// per-iteration diagnostic logging (avgGrad/maxDisp/activeTiles/photoEnergy/
// smooth-photo). Pure logging: gated off by default because the smooth/photo
// reduction is an extra parallel double-sqrt pass over every vertex each
// iteration. Convergence values are compiled out when both diagnostics and
// early exit are disabled.
#define MESHOPT_ITER_DIAGNOSTICS 0

// Stage-resolved pair-support counters (iter 0 of each scale only): per vertex,
// count pairs whose faces rasterized in A (raster) and whose pixels passed the
// warp mask (mask), alongside photoGradNorm (grad). Localizes WHERE support
// dies for zero-support regions. Dumped as MeshRefine{Raster,Mask}%u.ply at -v 3.
#define MESHOPT_SUPPORT_STAGE_DIAG (1 && MESHOPT_ITER_DIAGNOSTICS)

// [PROFILE] time the ProjectMesh phase vs the photo-consistency pair loop in
// ScoreMesh, to size the cost of the reference-view streaming (Option B), which
// would add ~one extra ProjectMesh per iteration. Set to 0 to remove all overhead.
// Measured ~12-14% on a 12.8M-face scene; not worth it on a 64 GB box (peak 33 GB).
// MEASURED AGAIN (RichmondHistoric, res-level 1, 7.19M faces): RasterizeFaceMap ~=
// 12-14% of the pair loop (~2.77 s of ~22.5 s per finest-scale iter). Keeping
// faceMaps resident (un-streaming) would save ~that at +~7 GB RAM.
// Set to 1 to emit the full per-scale phase breakdown (see MeshProf below);
// set BACK TO 0 for any release/quality build (adds per-call timing overhead).
#define MESHOPT_PROFILE 0

#if MESHOPT_PROFILE
// ---------------------------------------------------------------------------
// [PROFILE] Fine-grained phase profiler (opportunity hunting). Zero overhead
// when MESHOPT_PROFILE==0. Two kinds of buckets:
//   WALL  - measured on the main thread around a serial region or an omp
//           barrier (true wall-clock): scale/iter phases + ScoreMesh subtotals.
//   PAIR  - measured inside the omp-parallel pair loop with per-thread
//           accumulators, reduced at the barrier. THREAD-SUMMED (~nThreads x
//           wall); reported as a share of pair-loop compute so you can see
//           where the hot inner loop actually spends its time.
// A table prints at the end of each scale, plus a grand total in RefineMesh.
// ---------------------------------------------------------------------------
namespace MeshProf {
	enum Phase {
		P_InitImages, P_ListVtxPre, P_Subdivide, P_ListVtxPost,             // scale (wall)
		P_ScoreMesh, P_Momentum, P_MaxDisp,                                 // iter (wall)
		P_ListCameraFaces, P_PairLoopWall, P_Smooth1, P_Smooth2, P_Combine, // ScoreMesh (wall)
		P_StreamPrep, P_StreamFaceAreas, P_StreamBatches, P_StreamResident,  // view-stream (wall)
		P_Rasterize, P_RefVar, P_Warp, P_WarpVar, P_ZNCC, P_PhotoGrad, // pair loop (thread-summed)
		P_COUNT
	};
	static const int kFirstPairPhase = (int)P_Rasterize;
	static const char* const kName[P_COUNT] = {
		"InitImages", "ListVtxPre", "Subdivide", "ListVtxPost",
		"ScoreMesh(total)", "MomentumUpdate", "MaxDispScan",
		"  ListCameraFaces", "  PairLoop(wall)", "  Smooth1", "  Smooth2", "  Combine",
		"  StreamPrepChunked", "  StreamFaceAreas", "  StreamBuildBatches", "  StreamResident",
		"    Rasterize", "    RefVariance", "    Warp", "    WarpVariance", "    ZNCC", "    PhotoGrad"
	};
	struct Acc { double ms[P_COUNT] = {}; unsigned long long calls[P_COUNT] = {}; };
	static Acc gScale; // reset each scale
	static Acc gTotal; // whole run
	static thread_local double tlMs[P_COUNT] = {};
	static thread_local unsigned long long tlCalls[P_COUNT] = {};
	struct Timer {
		std::chrono::steady_clock::time_point t0{ std::chrono::steady_clock::now() };
		double ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
	};
	static inline void Add(Phase p, double ms) { gScale.ms[p] += ms; ++gScale.calls[p]; }
	static inline void AddTL(Phase p, double ms) { tlMs[p] += ms; ++tlCalls[p]; }
	static inline void FlushTL() { // call inside an omp critical section
		for (int i = 0; i < (int)P_COUNT; ++i) if (tlCalls[i]) { gScale.ms[i] += tlMs[i]; gScale.calls[i] += tlCalls[i]; tlMs[i] = 0; tlCalls[i] = 0; }
	}
	static void Report(const char* tag, const Acc& a) {
		double pairSum = 0; for (int i = kFirstPairPhase; i < (int)P_COUNT; ++i) pairSum += a.ms[i];
		DEBUG_EXTRA("[PROFILE] ================ %s ================", tag);
		for (int i = 0; i < (int)P_COUNT; ++i) {
			if (!a.calls[i]) continue;
			if (i >= kFirstPairPhase) {
				DEBUG_EXTRA("[PROFILE] %-20s %10.1f ms  %5.1f%% pair  (%llu calls)",
					kName[i], a.ms[i], pairSum > 0.0 ? 100.0 * a.ms[i] / pairSum : 0.0, a.calls[i]);
			} else {
				DEBUG_EXTRA("[PROFILE] %-20s %10.1f ms  (%llu calls, %.4f ms/call)",
					kName[i], a.ms[i], a.calls[i], a.ms[i] / (double)a.calls[i]);
			}
		}
	}
	static inline void RollUp() { for (int i = 0; i < (int)P_COUNT; ++i) { gTotal.ms[i] += gScale.ms[i]; gTotal.calls[i] += gScale.calls[i]; } }
	static inline void ResetScale() { gScale = Acc(); }
	struct Scope { Phase p; Timer t; explicit Scope(Phase p_) : p(p_) {} ~Scope() { Add(p, t.ms()); } };
} // namespace MeshProf
#define MESHPROF_SCOPE(ph) MeshProf::Scope _mp_scope(MeshProf::ph)
#else
#define MESHPROF_SCOPE(ph) ((void)0)
#endif // MESHOPT_PROFILE

// [PERF] ComputePhotometricGradient hot loop: cache the barycentric setup
// (bsu*/bsv*/invZ/bInvArea) by A-side face. The fixed-size thread-local cache
// persists across tiles within one pair, but is reset per call so mesh updates
// cannot leave stale geometry. Memory is constant (~7 KB per worker).
#ifndef MESHOPT_PG_FACECACHE
#define MESHOPT_PG_FACECACHE 1
#endif

// Interleave gx/gy during the 2x2 bilinear sample so both channels share one
// vertical and horizontal SIMD interpolation pipeline. Same samples and
// per-channel operation order as the split path; fewer vector instructions.
#ifndef MESHOPT_PG_BILERP_MERGE
#define MESHOPT_PG_BILERP_MERGE 1
#endif

// Center the sampled image-B gradient on the projected sample: gradX[x] =
// I(x+1)-I(x) is the derivative at x+0.5, so gathering the plane at xB yields
// the derivative half a pixel away from where the warp/ZNCC energy was
// measured. Gather at (xB-0.5, yB-0.5) instead; this also matches the CUDA
// kernel exactly (its tex2D forward difference == this plane sampled at -0.5).
// 0 = previous half-pixel-shifted gather.
#ifndef MESHOPT_PG_GRAD_CENTERED
#define MESHOPT_PG_GRAD_CENTERED 1
#endif

// Build the per-view gradient planes with upstream's 3x5 Gaussian-smoothed
// CENTERED derivative (CreateDerivativeKernel3x5 + filter2D, as in v2.4.0 and
// this fork before the size-reduction work) instead of the raw 2-tap forward
// difference that replaced it. The raw difference feeds unfiltered image noise
// straight into the photometric flow (bumpy surfaces); the smoothed kernel is
// the reference behaviour. Runs once per view per scale (negligible cost).
// 0 = forward difference (previous behaviour).
#ifndef MESHOPT_GRAD_KERNEL3X5
#define MESHOPT_GRAD_KERNEL3X5 1
#endif
#if MESHOPT_GRAD_KERNEL3X5 && MESHOPT_PG_GRAD_CENTERED
// the 3x5 kernel is already centered on the pixel; the -0.5 gather shift only
// compensates the forward difference's half-pixel offset
#undef MESHOPT_PG_GRAD_CENTERED
#define MESHOPT_PG_GRAD_CENTERED 0
#endif

// Store the per-view gradient planes as float32 instead of int16 quantized.
// The int16 quantization TRUNCATES toward zero: every gradient below 1/8192
// (1.22e-4) becomes exactly 0 and small ones are biased downward, silently
// deleting the photometric signal in dark/low-contrast regions (shadowed
// alley / roof shadow band — the exact geography of the CPU-vs-GPU melt,
// where the surface then falls to smoothing-only and bulges). The GPU
// differentiates fp16 texels in float (no dead zone); upstream samples float
// planes. Costs +4 B/px/view. 0 = int16 quantized (previous behaviour).
#ifndef MESHOPT_GRAD_F32
#define MESHOPT_GRAD_F32 0
#endif

// Skip the per-pixel ray normalization in ComputePhotometricGradient: |ray|
// cancels exactly between the Jacobian ray dots and 1/(N.d), so the gradient
// is computed on the unnormalized ray (grazing test in squared form,
// N.ray >= 0 || (N.ray)^2 < 0.01*|ray|^2 <=> N.d > -0.1) and the projection
// reuses those same ray dots (P*X = (P.ray)*depth + P.C; X never built).
// Removes the rsqrt+Newton serial chain at the loop head. Algebraically
// identical but NOT bit-identical to the normalize path (rounding differs on
// borderline grazing pixels / sample coords). Set to 0 for an exact A/B.
#ifndef MESHOPT_PG_UNNORM_RAY
#define MESHOPT_PG_UNNORM_RAY 1
#endif

// Fuse warped-image variance and A*B covariance into the ZNCC derivative pass.
// Uses width-only scratch instead of three full-image intermediates. Set to 0
// for an immediate performance/quality A/B against the previous split path.
#ifndef MESHOPT_FUSED_ZNCC
#define MESHOPT_FUSED_ZNCC 1
#endif

// Stream fused ZNCC derivatives to photometric scatter in 16-row tile bands.
// This preserves tile accumulation order while avoiding the full-frame
// imageDZNCC plane and its later cold reread. Set to 0 for a direct A/B against
// the previous full-frame fused-ZNCC path.
#ifndef MESHOPT_ZNCC_BANDS
#define MESHOPT_ZNCC_BANDS 1
#endif

// GPU-parity ZNCC derivative: average the per-pixel derivative terms
// {1/sqrt(varA*varB), zncc/varB, meanA*invS - meanB*zncc/varB} over the masked
// 5x5 ZNCC window, exactly like the CUDA ComputeImageDZNCC kernel, instead of
// using only the center-pixel term (the upstream-CPU approximation). The
// windowed derivative is the true gradient of the summed overlapping-window
// ZNCC energy: smoother, better conditioned, and the main CPU/GPU quality gap.
// Deviates from the CUDA PTX only in skipping non-interior neighbor terms
// (where the GPU reads zeroed var planes and produces inf; upstream's .cu port
// skips them too). Costs extra ZNCC-phase time (scalar ring-buffer path, no
// AVX2 kernel yet). Requires MESHOPT_FUSED_ZNCC. 0 = center-only (previous).
#ifndef MESHOPT_DZNCC_WINDOWED
#define MESHOPT_DZNCC_WINDOWED 1
#endif
#if MESHOPT_DZNCC_WINDOWED && !MESHOPT_FUSED_ZNCC
#undef MESHOPT_DZNCC_WINDOWED
#define MESHOPT_DZNCC_WINDOWED 0
#endif
#if MESHOPT_DZNCC_WINDOWED
#define MESHOPT_FUSED_ZNCC_FN ComputeLocalZNCCFusedWindowed
#else
#define MESHOPT_FUSED_ZNCC_FN ComputeLocalZNCCFused
#endif

// Eight-wide fused-ZNCC row kernel. The implementation is isolated in
// SceneRefineAVX2.cpp and entered only after CPUID/XGETBV runtime detection.
// Set to 0 for a direct SSE2 fallback A/B.
#ifndef MESHOPT_AVX2
#define MESHOPT_AVX2 1
#endif

// Resident-depth reuse adds a cold depth-map read for every covered sample and
// is slower on the current CPU path. Keep the generation-stamped scratch
// z-buffer as the default; set to 1 only for controlled profiling.
#ifndef MESHOPT_RASTER_REUSE_DEPTH
#define MESHOPT_RASTER_REUSE_DEPTH 0
#endif

// keep the decoded color image resident per view so the coarse->fine scale
// levels reuse it instead of re-decoding the identical JPEG from disk each
// level (imageSize depends only on nResolutionLevel, so it is constant per
// run). Trades RAM (num images x reloaded WxH x 3 bytes, e.g. ~5.5 GB at
// 367 x 5 MPix) for skipping the redundant decode + disk I/O on every scale
// after the first. Results are bit-identical to the decode path. Set back to
// 0 if a scene is RAM-tight.
#define MESHOPT_CACHE_IMAGES 1 // JPB WIP BUG Should be enabled for max perf

// Robust per-vertex gradient clip for the momentum minimizer: cap each vertex
// gradient magnitude at MESHOPT_GRAD_CLIP_K * (mean per-vertex gradient magnitude)
// before the momentum step. On a flat painted surface the photo term's only
// signal is the paint edges, so a few very-high-contrast vertices otherwise
// dominate the flow and carve (res-0) or shift/jitter (res-1) the surface.
// Clipping those outliers keeps flat painted areas flat WITHOUT globally
// over-smoothing real geometry the way a larger regularity weight does.
// Scene-adaptive (keyed to the current mean); 0 disables (original behaviour).
#define MESHOPT_GRAD_CLIP_K 0.0f

#define MESHOPT_DISABLE_TILE_SKIP 1
#define MESHOPT_DISABLE_EARLY_EXIT 1
#define MESHOPT_CUDA_PARITY_MASK 1

// Convergence-based early exit for the refinement iteration loop. Field logs
// show the photometric term plateaus within a few iterations at each scale
// (same-parity photoEnergy flat to <0.1% from iter ~4, while finest-scale
// iterations cost ~7.5 s); the residual gradient decay afterwards is mostly
// smoothing-term relaxation. photoEnergy alternates with the pair direction,
// so the plateau test compares iteration i against i-2 (same parity). On a
// sustained plateau the loop jumps to the scheduled final both-directions
// iteration (never skipped; step decay fast-forwarded). Costs the tile-energy
// accumulation in release builds (~1% of the pair loop). 0 = fixed iteration
// count (previous behaviour); set 0 for quality-parity A/B runs.
#ifndef MESHOPT_CONVERGED_EXIT
#define MESHOPT_CONVERGED_EXIT 1 // JPB WIP BUG Let's just test this.  Should be restored... looks very good.
#endif
// plateau = same-parity relative improvement below RTOL, HITS times in a row
#define MESHOPT_CONVERGED_EXIT_RTOL 0.001
#define MESHOPT_CONVERGED_EXIT_HITS 2

#if !MESHOPT_DISABLE_TILE_SKIP || !MESHOPT_DISABLE_EARLY_EXIT || MESHOPT_ITER_DIAGNOSTICS || MESHOPT_CONVERGED_EXIT
#define MESHOPT_NEEDS_TILE_ENERGY 1
#else
#define MESHOPT_NEEDS_TILE_ENERGY 0
#endif

// CUDA-parity rasterization coverage: the CPU rasterizer dropped any triangle
// with a projected vertex inside a 10 px frame border, losing A-side coverage
// (mask, depth, ListFaceAreas counts) for every border-crossing face in every
// view; the GPU kernel rasterizes the full frame and only the B-side warp
// sampling enforces the 10 px border (both paths). Rasterize border-crossing
// triangles instead; the wide guard band only rejects near-plane blowups
// whose huge screen coords would break edge-function precision.
// 0 = previous whole-triangle border reject.
#ifndef MESHOPT_RASTER_FULL_FRAME
#define MESHOPT_RASTER_FULL_FRAME 1
#endif

// Rasterize both windings like the GPU ProjectMesh kernel: it divides by the
// SIGNED area and never back-face culls (the z-buffer keeps the front surface
// on a watertight mesh). The CPU-only cull drops grazing/silhouette slivers
// whose projected-area sign flips, starving steep faces (building walls) of
// all A-side coverage and hence all photometric support. 0 = previous cull.
#ifndef MESHOPT_RASTER_NO_BACKFACE_CULL
#define MESHOPT_RASTER_NO_BACKFACE_CULL 1
#endif

// Zero the depth of uncovered pixels after rasterization: the per-view
// depth-map buffer is reused across iterations/scales and only covered pixels
// are rewritten, so uncovered pixels otherwise hold stale (or on a fresh
// allocation, garbage) depths that the pair warp's B-side occlusion test can
// wrongly match within its 1% tolerance. The GPU clears its depth-map on
// every projection. 0 = previous behaviour.
#ifndef MESHOPT_DEPTH_CLEAR_UNCOVERED
#define MESHOPT_DEPTH_CLEAR_UNCOVERED 1
#endif

// TEMPORARY DIAGNOSTIC - how large do projected triangles actually get?
//
// The raster walks barycentrics incrementally (w += w_dx along a row, w_row +=
// w_dy down the column) instead of re-evaluating the edge function per pixel
// like the reference (Types.inl RasterizeTriangleBary). That accumulates float
// rounding error in proportion to the step count, while the per-pixel
// barycentric gradient is ~1/span - so worst-case edge misclassification scales
// as steps*span*ulp(1). At the default max-face-area of 32 px^2 (span ~8 px)
// this is ~4e-6 px and entirely moot.
//
// The gap: SubdivideMesh bounds face size using the pixel count a face WON in
// the faceMap (ListFaceAreas), which cannot see off-screen extent. Under
// MESHOPT_RASTER_FULL_FRAME a face crossing the image border can have a huge
// true span, contribute few on-screen pixels, and so escape subdivision.
// Clipping bounds the step count but NOT the span, so those faces are the only
// ones where the incremental walk could drift visibly.
//
// This counts faces whose UNCLIPPED projected bbox span exceeds the threshold.
// If the count is 0 on real scenes the incremental walk is safe and this whole
// block can be compiled out (set to 0).
// Confirmed [SPAN] ListFaceAreas: 0/1432457216 rasterized faces span >= 256 px (0.0000%), max span 0.0 px, worst-case bary drift 0.00000 px
// Rasterization is constrained in size and float accuracy is sufficient.
#ifndef MESHOPT_RASTER_SPAN_DIAG
#define MESHOPT_RASTER_SPAN_DIAG 0
#endif
// unclipped bbox span, in pixels, at or above which a face is counted as "big"
#ifndef MESHOPT_RASTER_SPAN_DIAG_THRESHOLD
#define MESHOPT_RASTER_SPAN_DIAG_THRESHOLD 256.f
#endif

// Minimum directed-pair support required for a vertex to receive a photometric
// push in the gradient combine; below it the vertex is driven by smoothing only.
// Field A/B (Aug 2026): 2 made the occlusion-region melt WORSE — smoothing-only
// is the melting agent and even a single pair's photometric push anchors the
// surface. Keep 1 (== GPU/upstream parity, photoGradNorm > 0).
#ifndef MESHOPT_MIN_PAIR_SUPPORT
#define MESHOPT_MIN_PAIR_SUPPORT 1
#endif

// A/B refinement-quality gate. Groups the three settings that most limit how
// much the surface can move so they can be flipped together for comparison:
//   0 = current aggressive/fast behaviour (fewer iterations, aggressive tile
//       deactivation + early exit). Byte-identical to the pre-gate code.
//   1 = enhanced quality (more iterations, relaxed tile threshold, later
//       early-exit) to see the headroom the fast settings give up.
// Flip to 1 to A/B; set back to 0 to restore exactly the current behaviour.
#define MESHOPT_REFINE_QUALITY 1

// Option A (compute-neutral higher resolution). Run ONLY the final (finest)
// refinement scale one resolution level finer (~4x pixels) to recover genuine
// high-frequency detail, and offset the cost by cutting that scale's iteration
// count (see iters below). Coarse scales are left untouched. Only engages when
// a finer level is available (nResolutionLevel > 0). 0 = off (byte-identical).
//
// DISABLED (0): on a large scene at --resolution-level 1 (RichmondHistoric,
// 367 imgs / ~7 MPix each) the finest scale's ~2x pixel pyramid added ~+23 GB
// commit (50.9 -> 73.8 GB), crossing 63.7 GB physical RAM -> pagefile thrash
// (page faults 43.8M -> 318.9M) -> finest-scale refine 117 s -> 612 s. The extra
// RAM is (2*HIRES_SCALE)^2 x the per-view image pyramid ON THE FINAL SCALE. Only
// re-enable when base peak + that pyramid growth still fits in RAM; on a
// RAM-tight scene lower HIRES_SCALE (e.g. 0.6 => ~1.44x pyramid) or keep this 0.
#define MESHOPT_REFINE_HIRES_FINAL 0

// Memory/detail dial for Option A. Loading the finer level is ~2x linear
// (~4x pixels and ~4x per-view RAM). This factor downsamples that final-scale
// image back down to bound memory (effective final resolution vs base is
// 2 * this factor, linear):
//   1.0  = full level  (~4x pixels / RAM vs base) -- most detail, biggest spike
//   0.71 = ~1.4x linear (~2x pixels / RAM vs base) -- ~half the spike
//   0.5  = back to base resolution (no benefit)
// Valid range (0.5, 1.0]. Only used when MESHOPT_REFINE_HIRES_FINAL is on.
#define MESHOPT_REFINE_HIRES_SCALE 0.71f

// Keep the output face count near the base-resolution result when Option A is
// on. The higher-res final scale makes faces project to ~(2*HIRES_SCALE)^2 more
// pixels, so ~that many more faces subdivide. This scales the final-scale
// face-area budget by the same ratio, so vertex positions still refine against
// the high-res images but tessellation stays near baseline.
//   1 = compensate (face count ~ base). 0 = allow the extra faces (more
//       capacity for the very finest detail, ~2x faces at HIRES_SCALE 0.71).
#define MESHOPT_REFINE_HIRES_KEEPFACES 1

// uncomment to enable CERES optimization module
// (similar performance with the custom minimizer)
#ifdef _USE_CERES
#define MESHOPT_CERES
#endif

#ifdef MESHOPT_TYPEPOOL
#define DEC_BitMatrix(var)		BitMatrix& var = *BitMatrixPool()
#define DEC_Image(type, var)	TImage<type>& var = *ImagePool<type>()
#define DST_BitMatrix(var)		BitMatrixPool(&(var))
#define DST_Image(var)			ImagePool(&(var))
#else
#define DEC_BitMatrix(var)		BitMatrix var;
#define DEC_Image(type, var)	TImage<type> var;
#define DST_BitMatrix(var)
#define DST_Image(var)
#endif

// choose a scale so gradients (usually within ±2) fit in int16
constexpr float kScale = 8192.0f;
constexpr float kInvScale = 1.0f / kScale;

#if MESHOPT_GRAD_F32
typedef float GradStoreT; // full-precision gradient planes
#else
typedef int16_t GradStoreT; // 1/8192-quantized gradient planes
#endif

// MESHOPT_IMAGE_U16: store each resident per-view grayscale image as uint16
// fixed-point (0..65535) instead of float32, halving the largest resident plane
// (~3.5 GB at res-level 1). The source is an 8-bit JPEG, so the 1/65535 step is
// far below the ZNCC variance floor (1e-4) and the flatness prune (0.005) -> the
// result is quality-neutral; the hot ZNCC/warp loops are bandwidth-bound so the
// extra uint16->float widen is ~perf-neutral. 0 = float32 (byte-identical).
#ifndef MESHOPT_IMAGE_U16
#define MESHOPT_IMAGE_U16 1
#endif
#if MESHOPT_IMAGE_U16
typedef TImage<uint16_t> ImageStore;
constexpr size_t kImageStoreBytes = sizeof(uint16_t);
#else
typedef Image32F ImageStore;
constexpr size_t kImageStoreBytes = sizeof(float);
#endif
constexpr float kImgU16Scale = 1.0f / 65535.0f;

// Per-view gradient planes. These deliberately go through the CRT heap, NOT
// VirtualAlloc: a VirtualAlloc/VirtualFree(MEM_RELEASE) pair unmaps the pages, so
// every reallocation gets demand-zero pages and pays a soft page fault plus a
// kernel page-zero on first touch. At res-1 the planes are ~16 GB per scale, i.e.
// ~4M extra faults and 16 GB of kernel zeroing per scale that heap reuse avoids
// entirely -- measured as roughly a doubling of wall time, in EVERY path including
// non-streamed. (VirtualAlloc was introduced here only to test whether freed-but-
// retained heap explained a budget/peak gap; it did not -- the actual cause was
// PointCloudStreaming::Release using vector::clear(), which frees nothing.)
static inline void* PlaneAlloc(size_t bytes) {
	void* p = _aligned_malloc(bytes, 16);
	if (p == nullptr) {
		// Turn a genuine OOM here into a clean, catchable failure instead of the
		// null-deref crash that would otherwise happen on the very next write into
		// this plane (see the try/catch around RefineMesh's caller).
		VERBOSE("error: out of memory allocating a %.2f MB gradient plane", bytes / (1024.0 * 1024.0));
		throw std::bad_alloc();
	}
	return p;
}
static inline void PlaneFree(void* p) { _aligned_free(p); }

// Current process COMMIT (private bytes), which is the number that matters for
// "will this fit": Windows trims the working set under pressure, so WorkingSetSize
// falls while the process pages harder. PagefileUsage does not lie that way.
#ifdef _MSC_VER
#include <Psapi.h>
#pragma comment(lib, "Psapi.lib")
static uint64_t ProcessCommitBytes()
{
	PROCESS_MEMORY_COUNTERS pmc;
	if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
		return 0;
	return (uint64_t)pmc.PagefileUsage;
}
#else
static uint64_t ProcessCommitBytes() { return 0; }
#endif

// NOTE: the per-view byte constants live AFTER the MESHOPT_* gate block below,
// because they test MESHOPT_FACEMAP_RESIDENT and MESHOPT_VIEW_STREAM_EVICT_MAPS.
// Defining them here would read those macros as undefined (0) and silently size
// the budget wrong.

// ===========================================================================
// LOW-MEMORY (16 GB target) GATES — scaffolding. All default 0 (current
// behaviour byte-identical). See the design + CORRECTED cost notes below; the
// dynamic scheduler for view streaming is intentionally deferred.
//
// Resident wall at res-1: the per-view pyramid is held for ALL views at once
// (image u16 2B + gradX/gradY 4B + depthMap 4B + isValid 1B = 11 B/px). Access
// roles: gradX/gradY are B-ONLY; isValid is A-ONLY; image+depthMap are both.
//
// MESHOPT_GRAD_RECOMPUTE (drop the ~7 GB gradX/gradY planes):
//   CORRECTION to the earlier "independent, ~+10-15%" estimate — after reading
//   the code this does NOT hold as an independent lever:
//     * per-PAIR rebuild of B's gradient plane = ~70x the gradient-build work
//       (two 3x5 filter2D per (A,B) per iter) -> refine would balloon to tens
//       of minutes. Not viable standalone.
//     * per-SAMPLE recompute (3x5 Sobel at the 4 gather neighbours from the
//       resident u16 image) is ~+50-80% total and, crucially, must reproduce
//       filter2D's exact correlation + BORDER_REFLECT_101 edges or it SILENTLY
//       changes the gradient => a quality drift. Needs the VALIDATE_GRADIENT
//       harness / an A/B before it can be trusted.
//   Conclusion: the CHEAP, correct form of this only exists once B's gradient
//   is rebuilt ONCE PER B PER ITER and reused across its pairs — i.e. it is a
//   rider on MESHOPT_VIEW_STREAM, not a standalone gate. Left 0 until then.
#ifndef MESHOPT_GRAD_RECOMPUTE
#define MESHOPT_GRAD_RECOMPUTE 0
#endif

// MESHOPT_DEPTH_HALF (store depthMap as float16, ~-3.5 GB): quality-neutral
//   (warp uses a 1% depth tolerance; half's ~5e-4 rel err is 20x under). BUT
//   `DepthMap` is a global typedef used across the codebase, so this needs a
//   PARALLEL per-view half store + convert-on-read at the hot readers
//   (ImageMeshWarp depthMapA/depthMapB, ComputePhotometricGradient depthRowA).
//   Efficient half<->float wants F16C (_mm_cvtph_ps); confirm the build enables
//   it (Ryzen has it) or the scalar path is slow. Independent but invasive.
#ifndef MESHOPT_DEPTH_HALF
#define MESHOPT_DEPTH_HALF 0
#endif

// MESHOPT_ISVALID_STREAM (regenerate the A-ONLY isValid per reference view,
//   ~-1.6 GB): factor the ProjectMesh pruning pass into an on-demand regen that
//   runs alongside RasterizeFaceMap (already streamed) from the resident
//   depthMap + image. Deterministic -> quality-neutral. Modest win, reuses the
//   existing faceMap-streaming machinery.
#ifndef MESHOPT_ISVALID_STREAM
#define MESHOPT_ISVALID_STREAM 1
#endif

#if MESHOPT_ISVALID_STREAM && MESHOPT_CUDA_PARITY_MASK
#define MESHOPT_DIRECT_VALIDITY 1
#else
#define MESHOPT_DIRECT_VALIDITY 0
#endif

// Keep the rasterized per-view faceMap resident (+4 B/px per view, e.g. ~7 GB
// at 367 x 5 MPix): ProjectMesh already rasterizes the identical map every
// iteration, so the pair loop and ListFaceAreas consume it directly instead of
// re-rasterizing each reference view on demand (RasterizeFaceMap measured at
// ~12-14% of the pair loop). BIT-IDENTICAL by construction: under
// MESHOPT_DIRECT_VALIDITY the raster pass is the sole writer of the map in
// both paths (same face order, same depth tie-breaking). Set to 0 to restore
// streaming when RAM-tight.
#ifndef MESHOPT_FACEMAP_RESIDENT
#define MESHOPT_FACEMAP_RESIDENT 1 // JPB WIP BUG Should be enabled for max performance
#endif
#if MESHOPT_FACEMAP_RESIDENT && !MESHOPT_DIRECT_VALIDITY
// the streamed prune path mutates faceMap after rasterization (and
// ComputeIsValid re-derives coverage from it), so residency is only proven
// identical for the direct-validity configuration; self-disable otherwise
#undef MESHOPT_FACEMAP_RESIDENT
#define MESHOPT_FACEMAP_RESIDENT 0
#endif

// Eight-wide AVX2 row kernel for ImageMeshWarp (the largest scalar phase of
// the pair loop; it runs over the full frame for every directed pair).
// BIT-IDENTICAL: every per-pixel op is lane-independent and mirrored
// mul/add-for-mul/add (no FMA contraction in either TU), 1/zcB is an IEEE
// divide, _cvt_ftoi_fast == cvttps truncation, and the early-out cascade
// becomes progressive lane masks with masked gathers (dead lanes never touch
// memory). Runtime CPUID-gated like the ZNCC kernel; scalar tail handles
// cols%8. Only valid for the direct-validity + u16-image configuration.
// Set to 0 for a direct scalar A/B.
#ifndef MESHOPT_WARP_AVX2
#define MESHOPT_WARP_AVX2 1
#endif
#if MESHOPT_WARP_AVX2 && !(MESHOPT_DIRECT_VALIDITY && MESHOPT_IMAGE_U16)
#undef MESHOPT_WARP_AVX2
#define MESHOPT_WARP_AVX2 0
#endif

// EXACT frustum-cull (candidate face list) reuse across refinement iterations.
// Cameras are static; only the mesh moves. The per-camera candidate lists
// produced by the octree cull only need to be a SUPERSET of the faces that
// actually rasterize (the rasterizer rejects off-frustum faces itself via the
// 10 px border + invZ checks), so culling once with every frustum plane pushed
// outward by `pad` world units stays exact for all following iterations while
// the accumulated per-vertex displacement is <= pad. This replaces the
// centroid+octree rebuild + traversal + PreprocessCameraFaces done EVERY
// iteration with typically ~2 rebuilds per scale; if the displacement budget
// is ever exceeded, it simply reculls (still exact). Candidate order is frozen
// during reuse, so depth-tie rasterization is deterministic across reused
// iterations. 0 = rebuild every iteration (previous behaviour).
#ifndef MESHOPT_VISIBILITY_REUSE
#define MESHOPT_VISIBILITY_REUSE 1
#endif
// pad = this factor x the max per-vertex displacement applied in the last
// iteration; displacements decay (gstep *= 0.98), so 16x covers a whole
// scale's iterations with generous margin. Too small only costs extra reculls.
#define MESHOPT_VISIBILITY_PAD_FACTOR 16.0f

// MESHOPT_VIEW_STREAM (THE 16 GB lever): bounded resident view working-set.
//   Partition views into camera-proximity BATCHES; per batch build the pyramids
//   for the batch views + their neighbours, run all pairs whose reference A is
//   in-batch, LRU-evict. Parallelize WITHIN a batch (not across all views).
//   Resident views drop ~367 -> ~40-80 (pyramid ~19 GB -> ~2-4 GB); build
//   g_cameraData per-batch too. Quality-neutral by construction (same directed
//   pairs, same math, same iters — only ORDER + residency change). Cost = a
//   bounded per-view REBUILD MULTIPLIER (~1-3x, batched by locality). This is
//   also what makes MESHOPT_GRAD_RECOMPUTE affordable (rebuild B's gradient once
//   per B per iter, reused across its pairs). The DYNAMIC scheduler (batching +
//   eviction) is deferred per plan; this gate reserves the design.
#ifndef MESHOPT_VIEW_STREAM
#define MESHOPT_VIEW_STREAM 1
#endif
// Resident working-set budget for the STREAMED per-view planes (image + gradX +
// gradY), in GB. This replaces the old hand-tuned VIEW COUNT, which was wrong on
// two axes: it ignored how much RAM the box actually has (so a 512 GB machine
// batched exactly as hard as a 16 GB one, paying the batching cost for nothing),
// and it assumed every view costs the same, which is false the moment views differ
// in resolution -- and false ACROSS SCALES, where bytes/view grows ~4x per level.
//    0 (DEFAULT) = AUTO: size from real headroom, measured as
//                  (free physical + what streaming could hand back right now).
//                  Adding back the currently-resident streamed bytes matters
//                  because BuildViewBatches runs just after SubdivideMesh's
//                  geometry-prep pass, when free RAM is near its minimum for the
//                  scale; measuring raw freePhysical there would collapse the
//                  budget to nothing and force maximal batching.
//   >0           = fixed budget of that many GB (repeatable A/B).
//   <0           = unlimited: always one batch. This is the parity control -- it
//                  must match MESHOPT_VIEW_STREAM=0 in both output and wall time.
// Whatever the source, the budget is floored at the largest single reference-view
// neighbourhood, since a batch that cannot hold one reference view plus its pair
// neighbours cannot make progress at all.
#ifndef MESHOPT_VIEW_STREAM_BUDGET_GB
#define MESHOPT_VIEW_STREAM_BUDGET_GB 0
#endif
// Share of measured headroom the streamed planes may occupy under AUTO. The rest
// is margin for the pinned footprint this budget does NOT cover (depthMap/faceMap
// are held for every view regardless of batching -- see EvictView), allocator
// slack, and other processes.
#ifndef MESHOPT_VIEW_STREAM_FREE_FRACTION
#define MESHOPT_VIEW_STREAM_FREE_FRACTION 0.70
#endif

// MESHOPT_MEM_SIMULATE_PHYSICAL_GB: size the AUTO budget as if this machine had
// only this many GB of physical RAM. 0 = off (use the real machine). Only affects
// the AUTO path (MESHOPT_VIEW_STREAM_BUDGET_GB == 0).
//
// TESTING AID -- it changes the DECISION, not the PRESSURE. The process still
// commits exactly what it would otherwise commit, so a simulated run answers
// "how many batches would a smaller box pick, and does the output survive that
// batching?" It does NOT answer "would the run fit?". For that, compare the
// reported PeakPagefileUsage against the simulated size yourself: the budget only
// governs the streamed image+gradient planes, which are a minority of the peak
// (depthMap/faceMap are pinned by EvictView, and the mesh is not streamed at all).
#ifndef MESHOPT_MEM_SIMULATE_PHYSICAL_GB
#define MESHOPT_MEM_SIMULATE_PHYSICAL_GB 0 // JPB WIP BUG was 0 works at 32
#endif

// MESHOPT_MEM_TARGET_GB: peak-commit CEILING this process aims to stay under, in
// GB. 0 = off (size from measured free RAM, as before).
//
// This exists because the free-RAM formula structurally cannot hit a target. Once
// a box is oversubscribed its measured free memory clamps to zero and the budget
// collapses to "70% of what I am already holding" -- which always permits roughly
// the current residency, so it can prevent GROWTH but can never drive usage DOWN.
// A ceiling has to be computed from the ceiling:
//
//     budget = target - pinned,   pinned = processCommit - reclaimable
//
// where `reclaimable` is what view streaming can actually release (the per-view
// image + gradient planes and the decode cache) and `pinned` is everything else
// we are holding: depthMap/faceMap, the mesh, per-vertex arrays, allocator slack.
//
// HONEST LIMIT: if pinned alone already exceeds the target, no view-stream budget
// can meet it -- streaming governs a minority of the peak. That case is logged
// explicitly rather than silently producing a tiny budget and thrashing anyway.
//   >0 = fixed ceiling of that many GB (repeatable A/B, or a shared-machine cap).
//    0 = DERIVE the ceiling from this machine: totalPhysical x
//        MESHOPT_MEM_TARGET_FRACTION. This is the default because a hardcoded
//        number is wrong on every box but one: 32 needlessly batches a 64 GB
//        machine (+14% for nothing), while on a 16 GB machine it is unreachable.
//        Derived, the SAME binary runs flat out where there is room and degrades
//        only in SPEED where there is not -- which is the actual contract:
//        quality never varies with the machine, only wall time does.
//   <0 = no ceiling at all; fall back to the free-RAM AUTO budget + speed floor.
#ifndef MESHOPT_MEM_TARGET_GB
#define MESHOPT_MEM_TARGET_GB 0
#endif
// Share of physical RAM the whole process may commit when the ceiling is derived.
// Headroom left over covers the OS, other processes, and the terms this budget
// does not model (per-thread pair-loop scratch, allocator slack, g_cameraData).
#ifndef MESHOPT_MEM_TARGET_FRACTION
#define MESHOPT_MEM_TARGET_FRACTION 0.85
#endif
// Absolute floor on top of the fraction above: on a small box, 15% of total can
// be thinner than the OS + background apps actually need (15% of 16 GB is only
// 2.4 GB). The effective target is min(FRACTION*total, total-RESERVE), so this
// only ever tightens the target on boxes small enough that the fraction alone
// would leave less than RESERVE free (below ~20 GB at the defaults) -- above
// that, FRACTION*total is already smaller and this term never binds, so it costs
// nothing and changes nothing on a well-provisioned machine.
#ifndef MESHOPT_MEM_RESERVE_GB
#define MESHOPT_MEM_RESERVE_GB 3.0
#endif

// MESHOPT_VIEW_STREAM_VERIFY: assert, once per batch, that every view the batch is
// about to read is actually loaded, projected for the CURRENT mesh generation, and
// that both endpoints of every directed pair it will run are resident.
//
// This is the guard for the one failure mode batching can introduce and that no
// output check would localize: silently computing a pair against an evicted or
// stale-epoch view, which does not crash -- it just quietly degrades the mesh.
// Cost is O(views + pairs) ONCE PER BATCH against a pair loop that is O(pixels),
// so it is free in practice. Leave it on until the multi-batch path has real
// mileage; set 0 only if it ever shows up in a profile.
#ifndef MESHOPT_VIEW_STREAM_VERIFY
#define MESHOPT_VIEW_STREAM_VERIFY 1
#endif

// MESHOPT_VIEW_STREAM_EVICT_MAPS: let EvictView release the rasterized depthMap /
// faceMap as well, not just the image + gradient planes.
//
// Those maps are ~8 B/px per view and were pinned for EVERY view no matter how
// small the budget was, so they dominated `pinned` and put a 32 GB target out of
// reach except by shrinking tier 1 into thrash. Evicting them re-enters a view at
// the cost of a ProjectMesh instead of just a ThInitImage, which the epoch stamp
// already handles: EvictView clears viewProjEpoch, so the existing staleness test
// in EnsureViewsResident re-projects it. Output-neutral -- ProjectMesh is a pure
// function of (mesh geometry, camera, g_cameraData), all unchanged by eviction.
// Set 0 for an A/B against the tier-1-only behaviour.
#ifndef MESHOPT_VIEW_STREAM_EVICT_MAPS
#define MESHOPT_VIEW_STREAM_EVICT_MAPS 1
#endif

// MESHOPT_RELEASE_POINTCLOUD: drop the dense point cloud once neighbour selection
// has consumed it (the only place refinement uses it). Worth 3.64 GB of PINNED at
// 66.2M points -- roughly 8x the size of the mesh actually being refined.
//
// Gated because it has one visible consequence: RefineMesh saves the scene, so the
// output .mvs carries no point cloud. The refined MESH is unaffected. Set 0 if a
// downstream stage reads points from the refined scene rather than from the
// densify output.
#ifndef MESHOPT_RELEASE_POINTCLOUD
#define MESHOPT_RELEASE_POINTCLOUD 1
#endif

// ---- per-view byte accounting (must follow every gate above that it tests) ----
// TIER 1 -- per-view state rebuildable WITHOUT re-rasterizing: the gray image and
// the two gradient planes. Re-entry costs a ThInitImage (the decode is skipped when
// the colour cache still holds the view; the two 3x5 filter2D passes are not cached).
constexpr size_t kStreamTier1BytesPerPixel = kImageStoreBytes + 2 * sizeof(GradStoreT);

// TIER 2 -- the rasterized maps: depthMap plus, when it is held resident, faceMap.
// These cannot be rebuilt from the decode cache; re-entry costs a ProjectMesh. They
// were previously PINNED for every view regardless of budget (~16 GB at 437 views x
// 4.65 MPix), which is why a 32 GB target could only be met by squeezing tier 1 to
// 0.82 GB -> 60 batches -> 2.8x wall time for a 2% peak reduction. Making them
// evictable is what lets the budget reach the footprint that actually dominates.
// isValid rides in tier 2 as well: under MESHOPT_DIRECT_VALIDITY the raster pass
// is its sole producer, so it is created and invalidated on exactly the same events
// as depthMap/faceMap (~2 GB at 437 views x 4.65 MPix x 1 B).
#if MESHOPT_VIEW_STREAM_EVICT_MAPS
constexpr size_t kStreamTier2BytesPerPixel =
	sizeof(Depth)
#if MESHOPT_FACEMAP_RESIDENT
	+ sizeof(uint32_t)
#endif
#if MESHOPT_DIRECT_VALIDITY
	+ sizeof(uint8_t)
#endif
	;
#else
constexpr size_t kStreamTier2BytesPerPixel = 0;
#endif

// total evictable bytes per pixel -- what the view budget actually governs
constexpr size_t kStreamBytesPerPixel = kStreamTier1BytesPerPixel + kStreamTier2BytesPerPixel;
// Log free/total physical RAM + wall time at scale/iteration boundaries.
// Purely observational; independent of MESHOPT_VIEW_STREAM so the current
// (non-streamed) path can be baselined too.
#ifndef MESHOPT_MEM_DIAG
#define MESHOPT_MEM_DIAG 0
#endif
// Repeat ThInitImage calls under streaming would otherwise re-decode the JPEG
// from disk every time a view re-enters residency (the dominant per-iteration
// cost at small budgets). Force the existing decoded-color cache on so those
// calls skip straight to it; EnsureViewsResident bounds its size separately.
#if MESHOPT_VIEW_STREAM && !MESHOPT_CACHE_IMAGES
#undef MESHOPT_CACHE_IMAGES
#define MESHOPT_CACHE_IMAGES 1
#endif
// Decode-cache budget, as a MULTIPLE of the streamed-plane budget: the decode
// cache is ~3x cheaper per pixel (3 B/px colour vs ~8-10 B/px for the u16 image +
// gradX + gradY), and every entry it holds is a JPEG decode a re-entering view
// gets to skip, so it is the cheapest thing to spend headroom on.
#ifndef MESHOPT_VIEW_STREAM_DECODE_BUDGET_RATIO
#define MESHOPT_VIEW_STREAM_DECODE_BUDGET_RATIO 1.0
#endif

#if defined(MESHOPT_CERES) || MESHOPT_SUPPORT_STAGE_DIAG
#pragma intrinsic(_InterlockedCompareExchange)
static inline float AtomicAddFloat(float* addr, float val)
{
	static_assert(sizeof(float) == sizeof(LONG), "float and LONG must be same size");

	volatile LONG* intAddr = reinterpret_cast<volatile LONG*>(addr);
	LONG oldInt = *intAddr;
	float oldVal;

	for (;;)
	{
		oldVal = *reinterpret_cast<float*>(&oldInt);
		float newVal = oldVal + val;
		LONG newInt = *reinterpret_cast<LONG*>(&newVal);

		LONG prev = _InterlockedCompareExchange(intAddr, newInt, oldInt);
		if (prev == oldInt)
			break; // success
		oldInt = prev; // retry with updated oldInt
	}

	return oldVal;
}

static inline void AtomicMaxFloat(float* addr, float val)
{
	volatile LONG* intAddr = reinterpret_cast<volatile LONG*>(addr);
	LONG oldInt = *intAddr;
	for (;;) {
		const float oldVal = *reinterpret_cast<float*>(&oldInt);
		if (val <= oldVal)
			return;
		const LONG newInt = *reinterpret_cast<const LONG*>(&val);
		const LONG prev = _InterlockedCompareExchange(intAddr, newInt, oldInt);
		if (prev == oldInt)
			return;
		oldInt = prev;
	}
}
#endif

// S T R U C T S ///////////////////////////////////////////////////

typedef float Real;
typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;

struct CameraRenderData;

static std::vector<CameraRenderData> g_cameraData;
#if MESHOPT_VIEW_STREAM
// Points at ListCameraFaces' candidate lists (per camera, the global face indices
// the octree cull kept). Retained under streaming so g_cameraData can be expanded
// one batch at a time instead of for all 437 views at once.
static CLISTDEF2(Mesh::FaceIdxArr)* g_cameraFaces = nullptr;
#endif

// barycentric (bx,by) packed as two uint16 in one 32-bit word; bz = 1-bx-by.
// Quantized to 1/65535 (negligible as a gradient distribution weight); this
// stores 4 bytes/pixel instead of 12 for a full Point3f barycentric map.
static __forceinline uint32_t PackBary(float bx, float by) {
	bx = bx < 0.f ? 0.f : (bx > 1.f ? 1.f : bx);
	by = by < 0.f ? 0.f : (by > 1.f ? 1.f : by);
	return (uint32_t)(bx * 65535.f + 0.5f) | ((uint32_t)(by * 65535.f + 0.5f) << 16);
}
static __forceinline Point3f UnpackBary(uint32_t p) {
	constexpr float inv = 1.0f / 65535.0f;
	const float bx = (p & 0xFFFFu) * inv;
	const float by = (p >> 16) * inv;
	return Point3f(bx, by, 1.0f - bx - by);
}

class MeshRefine {
public:
	typedef TPoint3<Real> Grad;
	using GradArr = std::vector<Grad>;

	typedef TImage<cuint32_t> FaceMap;
	// barycentric coords packed as two uint16 (bx,by) in one 32-bit word;
	// bz = 1-bx-by. 4 bytes/pixel vs 12 for Point3f (see Pack/UnpackBary).
	typedef TImage<cuint32_t> BaryMap;

	struct VGrad
	{
		VIndex idx;    // vertex index (usually uint32_t or int)
		Grad   g;      // accumulated gradient, 16-byte padded
	};

	// store necessary data about a view
	struct View {
		typedef TPoint2<float> Grad;
		typedef TImage<Grad> ImageGrad;
		ImageStore image; // image pixels (float32 or uint16 fixed-point, see MESHOPT_IMAGE_U16)
#if MESHOPT_CACHE_IMAGES
		Image8U3 imageColorCache; // decoded color reused across scale levels
#endif
		ImageGrad imageGrad; // image pixel gradients
		//TImage<Real> imageMean; // image pixels mean
		//TImage<Real> imageVar; // image pixels variance
		// NOTE: the per-pixel SoA (ray/X/invNd/storedNormal/bary/verticesPerPix/
		// facesNormalPerPix) used to be cached here for every view. That was the
		// dominant memory consumer (~76 B/pixel x every image, resident at once).
		// It is now recomputed on demand in ComputePhotometricGradient from the
		// kept maps (depthMap/faceMap) + camera + CameraRenderData. The barycentric
		// coords are likewise recomputed there from faceMap + rd (no baryMap plane).
		GradStoreT* gradX = nullptr; // planar x-gradient (width*height; type per MESHOPT_GRAD_F32)
		GradStoreT* gradY = nullptr; // planar y-gradient (width*height)
		std::vector<uint8_t, AlignedAllocator<uint8_t, 16>> isValid;
		// MESHOPT_FACEMAP_RESIDENT=1: ProjectMesh persists it here each iteration
		// and the pair loop / ListFaceAreas consume it directly (no re-raster).
		// Otherwise STREAMED: regenerated per reference view on demand
		// (RasterizeFaceMap) in ScoreMesh/ListFaceAreas and released after use,
		// so only ~one faceMap per worker thread is alive at once.
		FaceMap faceMap; // for each pixel, the (local) face that projects there
		DepthMap depthMap; // depth-map
		int width, height;
		int blockWidth, blockStride;
		int allocatedSize = 0;
		std::vector<float> tileEnergyAccum;  // accumulated across threads
		std::vector<uint8_t> tileActive;     // used for skipping
		int tilesX, tilesY;
		//std::vector<uint8_t> marks;
		//uint8_t currentMark;
	};
	typedef CLISTDEF2(View) ViewsArr;

	// used to render a mesh for optimization
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		BaryMap& baryMap;
		FIndex idxFace;
		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap, BaryMap& _baryMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap), baryMap(_baryMap) {
		}
		void Clear() {
			Base::Clear();
			faceMap.fill(NO_ID);
			baryMap.memset(0);
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
			const Depth z(ComputeDepth(t, pbary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = idxFace;
				baryMap(pt) = PackBary(pbary.x, pbary.y);
			}
		}
	};


public:
	MeshRefine(Scene& _scene, unsigned _nReduceMemory, unsigned _nAlternatePair = true, Real _weightRegularity = 1.5f, Real _ratioRigidityElasticity = 0.8f, unsigned _nResolutionLevel = 0, unsigned _nMinResolution = 640, unsigned nMaxViews = 8, unsigned nMaxThreads = 1);
	~MeshRefine();

	bool IsValid() const { return !pairs.IsEmpty(); }

	bool InitImages(Real scale, Real sigma = 0);

	void ListVertexFacesPre();
	void ListVertexFacesPost();
	void ListCameraFaces(bool rebuildOctree = true);
#if MESHOPT_VIEW_STREAM
	void BuildViewBatches();
	void EnsureViewsResident(const std::vector<uint32_t>& neededViews, bool doProject = true);
	void EnsureAllViewsResident(bool doProject = true);
	void EvictView(uint32_t idxImage, bool alsoMaps = true);
	void EvictDecodeCache(uint32_t idxImage);
	void ReleaseAllViewPlanes();
	void ProjectAllViews();
	void PrepareViewGeometryChunked();
#if MESHOPT_VIEW_STREAM_VERIFY
	// takes the vectors rather than the batch: ViewBatch is declared further down
	void VerifyBatchResidency(const std::vector<uint32_t>& allViews,
		const std::vector<uint32_t>& refViews) const;
#endif
	void EnsureCameraData(uint32_t idxImage);
	void ReleaseCameraData(uint32_t idxImage);
	uint64_t StreamBytesPerView(uint32_t idxImage) const;
	uint64_t DecodeBytesPerView(uint32_t idxImage) const;
	uint64_t CameraDataBytesPerView(uint32_t idxImage) const;
	uint64_t BatchBytesPerView(uint32_t idxImage) const;
	uint64_t PairLoopScratchBytes() const;
	void LogPinnedBreakdown(uint64_t commit, uint64_t reclaimable, uint64_t scratch) const;
	uint64_t StreamBytesResident() const;
	uint64_t DecodeCacheBytesResident() const;
	uint64_t ResolveStreamBudget(uint64_t maxNeighbourhoodBytes, uint64_t totalStreamBytes) const;
	uint64_t ResolveTargetBytes() const;
#endif

	// streamMaps: rasterize+release each view's maps inside the loop instead of
	// requiring them all resident up front (subdivision pass only; see definition)
	void ListFaceAreas(Mesh::AreaArr& maxAreas, bool streamMaps = false);
	void SubdivideMesh(uint32_t maxArea, float fDecimate = 1.f, unsigned nCloseHoles = 15, unsigned nEnsureEdgeSize = 1);

	double ScoreMesh(float* gradients, bool rebuildOctree = true);

	// given a vertex position and a projection camera, compute the projected position and its derivative
	template <typename TP, typename TX, typename T, typename TJ>
	static T ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian = NULL);

	static bool IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z);
	void MeshRefine::ProjectMesh(
		View& view,
		const Camera& camera,
		const CameraRenderData& rd);
	// rasterize ONLY the (local) face-index map for a view; used to stream
	// the reference-view faceMap on demand instead of holding it resident.
	void RasterizeFaceMap(View& view, const CameraRenderData& rd);
#if MESHOPT_ISVALID_STREAM && !MESHOPT_DIRECT_VALIDITY
	// regenerate the (A-only) pruned isValid for a reference view on demand,
	// from its resident depthMap + the just-rasterized faceMap; used to keep
	// isValid out of the resident per-view set (MESHOPT_ISVALID_STREAM).
	void ComputeIsValid(View& view, const Camera& camera, const CameraRenderData& rd);
#endif
	static void ImageMeshWarp(
		const View& viewA,
		const DepthMap& depthMapA, const Camera& cameraA,
		const DepthMap& depthMapB, const Camera& cameraB,
		const ImageStore& imageB, TImage<uint16_t>& imageAB, std::vector<uint8_t>& mask);
	static void ComputeLocalVariance(
		const Image32F& image, const  std::vector<uint8_t>& mask,
		TImage<uint16_t>& imageMean, TImage<Real>& imageVar);
	static void ComputeLocalVariance2(
		const TImage<uint16_t>& image,               // now can be CV_16U
		const std::vector<uint8_t>& mask,
		TImage<uint16_t>& imageMean,
		TImage<Real>& imageVar);
	static void ComputeLocalVariance2Unmasked(
		const TImage<uint16_t>& image,
		TImage<uint16_t>& imageMean,
		TImage<Real>& imageVar);
	static float ComputeLocalZNCC(
		const ImageStore& imageA, const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
		const TImage<uint16_t>& imageB, const TImage<uint16_t>& imageMeanB, const TImage<Real>& imageVarB,
		const std::vector<uint8_t>& mask, TImage<Real>& imageDZNCC);
	static float ComputeLocalZNCCFused(
		const ImageStore& imageA, const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
		const TImage<uint16_t>& imageB, const std::vector<uint8_t>& mask,
		TImage<Real>& imageDZNCC,
		const std::function<void(const TImage<Real>&, size_t, size_t, bool)>& consumeBand = {});
	static float ComputeLocalZNCCFusedWindowed(
		const ImageStore& imageA, const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
		const TImage<uint16_t>& imageB, const std::vector<uint8_t>& mask,
		TImage<Real>& imageDZNCC,
		const std::function<void(const TImage<Real>&, size_t, size_t, bool)>& consumeBand = {});
	static void ComputePhotometricGradient(
		const View& viewA,
		const CameraRenderData& rd,
		const Mesh::NormalArr& faceNormals,
		const Camera& cameraA,
		const Camera& cameraB,
		const View& viewB,
		const TImage<Real>& imageDZNCC,
		const  std::vector<uint8_t>& mask,
		GradArr& photoGrad,
		std::vector<uint32_t>& photoGradNorm,
		Real RegularizationScale,
		uint64_t setupEpoch,
		std::vector<float>& tileEnergyLocal,
		size_t rowBegin = 0,
		size_t rowEnd = SIZE_MAX,
		size_t dznccRowOffset = 0,
		bool finalizePair = true);
	static float ComputeSmoothnessGradient1(
		const Mesh::VertexArr& vertices, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
		GradArr& smoothGrad1, VIndex idxStart, VIndex idxEnd);
	static void ComputeSmoothnessGradient2(
		const GradArr& smoothGrad1, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
		GradArr& smoothGrad2, VIndex idxStart, VIndex idxEnd);
	template<typename TYPE>
	static TYPE* TypePool(TYPE* = NULL);
	template<typename TYPE>
	static inline TImage<TYPE>* ImagePool(TImage<TYPE>* pImage = NULL) { return TypePool< TImage<TYPE> >(pImage); }
	static inline BitMatrix* BitMatrixPool(BitMatrix* pMask = NULL) { return TypePool<BitMatrix>(pMask); }

	static void* ThreadWorkerTmp(void*);
	void ThreadWorker();
	void WaitThreadWorkers(size_t nJobs);
	void ThSelectNeighbors(uint32_t idxImage, std::unordered_set<uint64_t>& mapPairs, unsigned nMaxViews);
	void ThInitImage(uint32_t idxImage, Real scale, Real sigma);
	void ThProjectMesh(uint32_t idxImage, const Mesh::FaceIdxArr& cameraFaces, const CameraRenderData& rd);
	void ThProcessPair(uint32_t idxImageA, uint32_t idxImageB, GradArr& localGrad, std::vector<uint32_t>& threadNorm, std::vector<std::vector<float>>& tileEnergyLocal,
		const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA);
	void ThSmoothVertices1(VIndex idxStart, VIndex idxEnd);
	void ThSmoothVertices2(VIndex idxStart, VIndex idxEnd);

public:
	const Real weightRegularity; // a scalar regularity weight to balance between photo-consistency and regularization terms
	Real ratioRigidityElasticity; // a scalar ratio used to compute the regularity gradient as a combination of rigidity and elasticity
	const unsigned nResolutionLevel; // how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // how many times to scale down the images before mesh optimization
	// MESHOPT_REFINE_HIRES_FINAL: per-scale override of nResolutionLevel used by
	// ThInitImage; unsigned(-1) means "use nResolutionLevel" (normal behaviour).
	unsigned resolutionLevelOverride = unsigned(-1);
	const unsigned nReduceMemory; // recompute image mean and variance in order to reduce memory requirements
	unsigned nAlternatePair; // using an image pair alternatively as reference image (0 - both, 1 - alternate, 2 - only left, 3 - only right)
	unsigned iteration; // current refinement iteration
	double photoEnergyLast = 0.0;
	double tProjectMeshMs = 0.0; // [PROFILE] wall time of the ProjectMesh phase in the last ListCameraFaces
	double tRasterizeMs = 0.0; // [PROFILE] summed per-thread time of RasterizeFaceMap in the last pair loop
	uint64_t faceSetupEpoch = 0; // invalidates worker-local face setup caches after geometry updates
#if MESHOPT_VISIBILITY_REUSE
	// exact padded-frustum visibility reuse (see MESHOPT_VISIBILITY_REUSE)
	bool candidateCacheValid = false; // g_cameraData matches current topology/image dims
	float frustumPadWorld = 0.f;      // outward plane pad used at the last recull
	float accumDispSinceCull = 0.f;   // summed per-iteration max vertex displacement
	float lastIterMaxDisp = 0.f;      // sizes the pad for the next recull
	void OnVerticesDisplaced(float maxDisp) { accumDispSinceCull += maxDisp; lastIterMaxDisp = maxDisp; }
#endif

	Scene& scene; // the mesh vertices and faces

	// gradient related
#ifdef MESHOPT_CERES
	float scorePhoto;
	float scoreSmooth;
#endif
	GradArr photoGrad;
	FloatArr photoGradNorm;
#if MESHOPT_SUPPORT_STAGE_DIAG
	FloatArr rasterSupport;
	FloatArr maskSupport;
	// per GLOBAL face, views where it: was a candidate / survived setup to a
	// clamped bbox / had a pixel pass the inside test / won the depth test
	FloatArr faceCandViews, faceBBoxViews, faceInsideViews, faceWonViews;
	// max relative margin (z_lost-z_stored)/z_stored over its losing pixels:
	// ~0 = FP-tie at a shared silhouette edge, >1% = a real closer occluder
	FloatArr faceLoseMargin;
#endif
	FloatArr vertexDepth;
	GradArr smoothGrad1;
	GradArr smoothGrad2;

	// valid after ListCameraFaces()
	Mesh::NormalArr& faceNormals; // normals corresponding to each face

	// valid the entire time, but changes
	Mesh::VertexArr& vertices;
	Mesh::FaceArr& faces;
	Mesh::VertexVerticesArr& vertexVertices; // for each vertex, the list of adjacent vertices
	Mesh::VertexFacesArr& vertexFaces; // for each vertex, the list of faces containing it
	BoolArr& vertexBoundary; // for each vertex, stores if it is at the boundary or not

	// constant the entire time
	ImageArr& images;
	ViewsArr views; // views' data
	PairIdxArr pairs; // image pairs used to refine the mesh
	// per reference view, the neighbor views it is paired with (both pair
	// directions collapsed); built lazily, drives faceMap streaming order.
	std::vector<std::vector<uint32_t>> refViewNeighbors;
#if MESHOPT_VIEW_STREAM
	struct ViewBatch {
		std::vector<uint32_t> refViews; // views acting as reference (A) in this batch
		std::vector<uint32_t> allViews; // refViews + their pair-neighbors (need residency)
	};
	std::vector<ViewBatch> viewBatches;
	std::vector<uint8_t> viewResident; // 1 if a view's image/gradX/gradY are currently loaded
	std::vector<uint8_t> decodeResident; // 1 if a view's decoded-color cache is currently loaded
	std::deque<uint32_t> decodeOrder; // FIFO eviction order for the decode cache (oldest at front)
	Real streamScale = 1, streamSigma = 0; // stashed for on-demand ThInitImage
	// faceSetupEpoch at which each view's depthMap/faceMap were last rasterized.
	// ProjectMesh is a pure function of (mesh geometry, camera, g_cameraData), and
	// all three are refreshed exactly once per ScoreMesh -- ListCameraFaces bumps
	// faceSetupEpoch at its head. So a view listed by a LATER batch as a neighbour,
	// having already been projected by an EARLIER batch of the SAME ScoreMesh call,
	// must not be projected again: the mesh cannot have moved in between (ScoreMesh
	// only accumulates gradients; vertices are updated afterwards in the momentum
	// step). This stamp is what removes the per-batch re-projection multiplier that
	// made streaming cost ~2-3x the ProjectMesh work of the non-streamed path.
	// EvictView releases only image/gradX/gradY and never the maps, so the stamp
	// stays valid across an evict/reload cycle within one epoch.
	std::vector<uint64_t> viewProjEpoch;
	// Two stamps, because the two halves of CameraRenderData have DIFFERENT
	// lifetimes and conflating them made the expensive half run every iteration:
	//   topo  - PreprocessCameraFaces output (local faces, globalFace/globalVert).
	//           Depends on the candidate lists and mesh TOPOLOGY, so it is only
	//           stale when the octree cull is rebuilt. Previously ~2x per scale.
	//   verts - UpdateCameraVertsAndNormals output (camera-space vertex positions).
	//           Depends on vertex POSITIONS, so it is stale every iteration.
	// 0 means "not built" for both.
	std::vector<uint64_t> viewCamTopoEpoch;
	std::vector<uint64_t> viewCamVertsEpoch;
	uint64_t cullEpoch = 0; // bumped only when the frustum cull is rebuilt
	std::vector<uint8_t> viewNeeded; // scratch membership mask for EnsureViewsResident
	// resident image+grad budget driving BuildViewBatches; unbounded until the
	// first BuildViewBatches of a scale resolves it, so the full-residency passes
	// that run before it (SubdivideMesh) are never throttled by a stale budget
	uint64_t streamBudgetBytes = std::numeric_limits<uint64_t>::max();
	// Bytes we FREED whose pages the CRT did not return to the OS. Measured, not
	// guessed: the point-cloud release frees a known number of bytes and we read
	// commit on both sides -- whatever did not come back is retained-free heap.
	// It counts in commit (so `pinned` sees it) but it is never touched again, so
	// it never faults back in and costs no resident RAM. Charging it against the
	// ceiling would shrink the budget by ~3.6 GB for memory that is not really in
	// use, which is the difference between fitting a 16 GB box and not.
	uint64_t retainedFreeBytes = 0;
	bool streamSingleBatch = false; // true when the whole view set fit in one batch
	// true when PrepareViewGeometryChunked determined the whole streamed set fits
	// under the ceiling, so it keeps the planes rather than making the pair loop
	// reload them
	bool streamKeepPlanes = false;
	// Counts pair-loop passes so the batch order can alternate on EVERY pass.
	// Not `iteration`: the optimizer can call ScoreMesh more than once per
	// iteration, and two passes in a row with the same order re-pay the
	// wrap-around transition this exists to remove.
	uint64_t pairPassCounter = 0;
#endif

	// multi-threading
	static SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
	static SEACAVE::cList<SEACAVE::Thread> threads; // worker threads
	static CriticalSection cs; // mutex
	static Semaphore sem; // signal job end

	enum { HalfSize = 2 }; // half window size used to compute ZNCC
};

// call with empty parameter to get an unused image;
// call with an image pointer retrieved earlier to signal that is not needed anymore
template<typename TYPE>
TYPE* MeshRefine::TypePool(TYPE* pObj)
{
	typedef CAutoPtr<TYPE> TypePtr;
	static CriticalSection cs;
	static cList<TypePtr, TYPE*> objects;
	static cList<TYPE*, TYPE*, 0> unused;
	Lock l(cs);
	if (pObj == NULL) {
		if (unused.IsEmpty())
			return objects.AddConstruct(new TYPE);
		return unused.RemoveTail();
	}
	else {
		ASSERT(objects.Find(pObj) != NO_IDX);
		ASSERT(unused.Find(pObj) == NO_IDX);
		unused.Insert(pObj);
		return NULL;
	}
}


enum EVENT_TYPE {
	EVT_JOB = 0,
	EVT_CLOSE,
};

class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};
class EVTSelectNeighbors : public Event
{
public:
	uint32_t idxImage;
	std::unordered_set<uint64_t>& mapPairs;
	unsigned nMaxViews;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSelectNeighbors(idxImage, mapPairs, nMaxViews);
		return true;
	}
	EVTSelectNeighbors(uint32_t _idxImage, std::unordered_set<uint64_t>& _mapPairs, unsigned _nMaxViews) : Event(EVT_JOB), idxImage(_idxImage), mapPairs(_mapPairs), nMaxViews(_nMaxViews) {}
};
class EVTInitImage : public Event
{
public:
	uint32_t idxImage;
	Real scale, sigma;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThInitImage(idxImage, scale, sigma);
		return true;
	}
	EVTInitImage(uint32_t _idxImage, Real _scale, Real _sigma) : Event(EVT_JOB), idxImage(_idxImage), scale(_scale), sigma(_sigma) {}
};
class EVTProjectMesh : public Event
{
public:
	uint32_t idxImage;
	const Mesh::FaceIdxArr& cameraFaces;
	const CameraRenderData& rd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProjectMesh(idxImage, cameraFaces, rd);
		return true;
	}
	EVTProjectMesh(uint32_t _idxImage, const Mesh::FaceIdxArr& _cameraFaces, const CameraRenderData& _rd) : Event(EVT_JOB), idxImage(_idxImage), cameraFaces(_cameraFaces), rd(_rd) {}
};

#if 0
class EVTProcessPair : public Event
{
public:
	uint32_t idxImageA, idxImageB;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProcessPair(idxImageA, idxImageB);
		return true;
	}
	EVTProcessPair(uint32_t _idxImageA, uint32_t _idxImageB) : Event(EVT_JOB), idxImageA(_idxImageA), idxImageB(_idxImageB) {}
};
#endif
class EVTSmoothVertices1 : public Event
{
public:
	VIndex idxStart, idxEnd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSmoothVertices1(idxStart, idxEnd);
		return true;
	}
	EVTSmoothVertices1(VIndex _idxStart, VIndex _idxEnd) : Event(EVT_JOB), idxStart(_idxStart), idxEnd(_idxEnd) {}
};
class EVTSmoothVertices2 : public Event
{
public:
	VIndex idxStart, idxEnd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSmoothVertices2(idxStart, idxEnd);
		return true;
	}
	EVTSmoothVertices2(VIndex _idxStart, VIndex _idxEnd) : Event(EVT_JOB), idxStart(_idxStart), idxEnd(_idxEnd) {}
};

SEACAVE::EventQueue MeshRefine::events;
SEACAVE::cList<SEACAVE::Thread> MeshRefine::threads;
CriticalSection MeshRefine::cs;
Semaphore MeshRefine::sem;

MeshRefine::MeshRefine(Scene& _scene, unsigned _nReduceMemory, unsigned _nAlternatePair, Real _weightRegularity, Real _ratioRigidityElasticity, unsigned _nResolutionLevel, unsigned _nMinResolution, unsigned nMaxViews, unsigned nMaxThreads)
	:
	weightRegularity(_weightRegularity),
	ratioRigidityElasticity(_ratioRigidityElasticity),
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution),
	nReduceMemory(_nReduceMemory),
	nAlternatePair(_nAlternatePair),
	scene(_scene),
	faceNormals(_scene.mesh.faceNormals),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	vertexVertices(_scene.mesh.vertexVertices),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	images(_scene.images)
{
	// start worker threads
	ASSERT(nMaxThreads > 0);
	ASSERT(threads.IsEmpty());
	threads.Resize(nMaxThreads);
	FOREACHPTR(pThread, threads)
		pThread->start(ThreadWorkerTmp, this);
	// keep only best neighbor views for each image
	std::unordered_set<uint64_t> mapPairs;
	mapPairs.reserve(images.GetSize() * nMaxViews);
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTSelectNeighbors(idxImage, mapPairs, nMaxViews));
	WaitThreadWorkers(images.GetSize());
	pairs.Reserve(mapPairs.size());
	for (uint64_t pair : mapPairs)
		pairs.AddConstruct(pair);
#if MESHOPT_VIEW_STREAM && MESHOPT_RELEASE_POINTCLOUD
	// SelectNeighborViews above is the ONLY consumer of the dense point cloud in
	// refinement; from here on it is dead weight held for the whole run. Measured
	// 3.64 GB (1.67 points/normals/colours + 1.97 per-point view lists) at 66.2M
	// points -- against a 16 GB machine's 13.60 GB ceiling that is decisive, and it
	// is 8x the entire mesh being refined (0.47 GB).
	//
	// SIDE EFFECT, deliberate and the reason this is gated: RefineMesh saves the
	// scene, so the output .mvs will carry 0 points instead of 66.2M. The refined
	// MESH is byte-for-byte unaffected -- only the point cloud is dropped from the
	// saved file. Set 0 if anything downstream reads points from the refined scene.
	{
		const size_t numPoints = scene.pointcloud.GetSize();
		if (numPoints > 0) {
			// scene.pointcloud is a PointCloudStreaming: flat CSR arrays, not
			// per-point containers. Size them by CAPACITY, since that is what
			// Release() has to hand back.
			const PointCloudStreaming& pc = scene.pointcloud;
			const uint64_t freedBytes =
				(uint64_t)pc.pointsXYZ.capacity() * sizeof(float) +
				(uint64_t)pc.normalsXYZ.capacity() * sizeof(float) +
				(uint64_t)pc.colorsRGB.capacity() * sizeof(uint8_t) +
				(uint64_t)pc.pointViewsOffsets.capacity() * sizeof(uint32_t) +
				(uint64_t)pc.pointViewsSizes.capacity() * sizeof(uint32_t) +
				(uint64_t)pc.pointViewsMemory.capacity() * sizeof(uint32_t) +
				(uint64_t)pc.pointWeightsOffsets.capacity() * sizeof(uint32_t) +
				(uint64_t)pc.pointWeightsSizes.capacity() * sizeof(uint32_t) +
				(uint64_t)pc.pointWeightsMemory.capacity() * sizeof(float);
			const uint64_t before = ProcessCommitBytes();
			scene.pointcloud.Release();
			// Release() frees the containers, but the CRT does not necessarily hand
			// the pages back to the OS -- and `pinned` is measured from COMMIT, so
			// retained-free memory counts against the ceiling exactly as if it were
			// live. Measured: releasing 3.64 GB of point cloud moved `pinned` by only
			// 0.30 GB, the rest reappearing as UNACCOUNTED. _heapmin asks the CRT to
			// decommit its free blocks; the before/after commit below says whether
			// that is where the ~9 GB of unaccounted `pinned` actually lives.
			_heapmin();
			const uint64_t after = ProcessCommitBytes();
			const uint64_t returned = (before > after) ? (before - after) : 0;
			retainedFreeBytes = (freedBytes > returned) ? (freedBytes - returned) : 0;
			constexpr double GBd = 1024.0 * 1024.0 * 1024.0;
			DEBUG_EXTRA("released the dense point cloud (%zu points, %.2f GB of capacity) after "
				"neighbor selection: unused by refinement. commit %.2f -> %.2f GB (returned "
				"%.2f GB; %.2f GB not returned by the allocator and therefore excluded from "
				"`pinned`, since those pages are never touched again). NOTE the saved scene "
				"will contain no point cloud (the refined mesh is unaffected).",
				numPoints, freedBytes / GBd, before / GBd, after / GBd,
				returned / GBd, retainedFreeBytes / GBd);
		}
	}
#endif
}

MeshRefine::~MeshRefine()
{
	// wait for the working threads to close
	FOREACH(i, threads)
		events.AddEvent(new EVTClose());
	FOREACHPTR(pThread, threads)
		pThread->join();
	FOREACHPTR(pView, views) {
		PlaneFree(pView->gradX);
		PlaneFree(pView->gradY);
		pView->gradX = nullptr;
		pView->gradY = nullptr;
	}
	scene.mesh.ReleaseExtra();
}

// load and initialize all images at the given scale
// and compute the gradient for each input image
// optional: blur them using the given sigma
bool MeshRefine::InitImages(Real scale, Real sigma)
{
	views.Resize(images.GetSize());
#if MESHOPT_VIEW_STREAM
	// new scale -> resolution changed; release stale resident static data
	// instead of eagerly reloading every view (EnsureViewsResident loads
	// lazily, per batch, inside ScoreMesh)
	streamScale = scale;
	streamSigma = sigma;
	// Size the bookkeeping BEFORE the eviction loop: EvictView writes
	// viewProjEpoch under MESHOPT_VIEW_STREAM_EVICT_MAPS, so it must already exist.
	if (viewResident.size() != images.GetSize())
		viewResident.assign(images.GetSize(), 0);
	if (decodeResident.size() != images.GetSize())
		decodeResident.assign(images.GetSize(), 0);
	if (viewProjEpoch.size() != images.GetSize())
		viewProjEpoch.assign(images.GetSize(), 0);
	if (viewCamTopoEpoch.size() != images.GetSize())
		viewCamTopoEpoch.assign(images.GetSize(), 0);
	if (viewCamVertsEpoch.size() != images.GetSize())
		viewCamVertsEpoch.assign(images.GetSize(), 0);
	if (viewNeeded.size() != images.GetSize())
		viewNeeded.assign(images.GetSize(), 0);
	FOREACH(idxImage, views)
		if (viewResident[idxImage])
			EvictView((uint32_t)idxImage);
	// A new scale changes every view's resolution, so both the batch partition and
	// the rasterized maps are stale: bytes/view grows ~4x per level, which makes a
	// partition computed at the previous scale meaningless, and depthMap/faceMap
	// are still sized to the old resolution. Force a rebuild of both.
	viewBatches.clear();
	streamBudgetBytes = std::numeric_limits<uint64_t>::max();
	std::fill(viewProjEpoch.begin(), viewProjEpoch.end(), 0);
#else
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTInitImage(idxImage, scale, sigma));
	WaitThreadWorkers(images.GetSize());
#endif
	iteration = 0;
#if MESHOPT_VISIBILITY_REUSE
	// image dimensions changed -> cull frusta are stale; keep lastIterMaxDisp as
	// a scale-bridging pad estimate (a too-small pad only costs an extra recull)
	candidateCacheValid = false;
#endif
	return true;
}

// extract array of triangles incident to each vertex
// and check each vertex if it is at the boundary or not
void MeshRefine::ListVertexFacesPre()
{
	scene.mesh.EmptyExtra();
	scene.mesh.ListIncidenteFaces();
#if MESHOPT_VISIBILITY_REUSE
	// called after every topology mutation (Clean/Subdivide/EnsureEdgeSize):
	// candidate face lists reference dead face/vertex indices -> force recull
	candidateCacheValid = false;
#endif
}

void MeshRefine::ListVertexFacesPost()
{
	scene.mesh.ListIncidenteVertices();
	scene.mesh.ListBoundaryVertices();
}

struct CamVert {
	float x, y, z, invZ;
};

struct CameraRenderData {
	std::vector<CamVert> verts;       // per-camera compact camera-space vertices
	std::vector<Face> faces;          // compact faces using LOCAL vertex indices
	// normals are NOT stored per camera: they duplicate faceNormals[globalFace] and
	// are looked up on demand in ProjectMesh/ComputePhotometricGradient instead.
	std::vector<uint32_t> globalFace; // localFaceIndex -> global face index
	std::vector<uint32_t> globalVert; // localVertIndex -> global vertex index
};

std::vector<CamVert> camVerts;
std::vector<uint32_t> mapIndex;

void PreprocessCameraFaces(
	const Mesh::FaceIdxArr& cameraFaces,   // global face indices visible to this camera
	const Mesh::FaceArr& faces,            // global mesh faces
	const Mesh::VertexArr& vertices,       // global mesh vertices (world space)
	const Mesh::NormalArr& faceNormals,    // global face normals
	CameraRenderData& out
) {
	const size_t numFaces = cameraFaces.size();

	// ------------------------------------------------------------------
	// STATIC THREAD-LOCAL REMAP TABLE (fastest possible approach)
	// ------------------------------------------------------------------
	thread_local std::vector<uint32_t> remap;     // global vertex -> local index
	thread_local std::vector<uint32_t> remapGen;  // generation markers
	thread_local uint32_t curGen = 1;

	// Ensure remap tables are large enough
	if (remap.size() < vertices.size()) {
		remap.resize(vertices.size());
		remapGen.resize(vertices.size(), 0);
	}

	// Bump generation (wrap & clear if needed)
	curGen++;
	if (curGen == 0) {
		// Hard reset (rare)
		std::fill(remapGen.begin(), remapGen.end(), 0);
		curGen = 1;
	}

	// ------------------------------------------------------------------
	// 1. Allocate outputs + copy the global face index list
	// ------------------------------------------------------------------
	std::vector<uint32_t>& used = out.globalVert;
	used.clear();
	used.reserve(numFaces);          // closed manifold: V ~ F/2; strips approach F

	out.faces.resize(numFaces);
	out.globalFace.resize(numFaces);

	// globalFace is cameraFaces verbatim (arrCameraFaces is released right after
	// this call, so we must own a copy) -> bulk copy, not a per-face store
	const uint32_t* __restrict cf = cameraFaces.data();
	memcpy(out.globalFace.data(), cf, numFaces * sizeof(uint32_t));

	// ------------------------------------------------------------------
	// 2. Build compact vertex list (used[]) + localized faces, TILED
	//
	// The insert pass is branchy; the localize pass is branchless and streams.
	// Keeping them as separate loops preserves the localize loop's memory-level
	// parallelism (the CPU runs far ahead of it, keeping many gathers in flight),
	// while a tile small enough to stay in L1/L2 (2048 * 12B = 24KB of Face data)
	// removes its second trip to DRAM for faces[]. Measured on a 7950X over
	// 200k..8M-face meshes: 12-27% faster than the previous untiled two-pass when
	// run under the omp loop in ListCameraFaces.
	//
	// Do NOT "simplify" this into a single fused loop: that measures SLOWER
	// single-threaded (up to +88% in one config) because the mispredicted insert
	// branches collapse the out-of-order window the gathers depend on.
	//
	// Vertex visit order -- hence used[] and the out.verts layout every consumer
	// of globalVert depends on -- is identical to the untiled version, since
	// tiles are processed in increasing face order.
	// ------------------------------------------------------------------
	Face* __restrict lfp = out.faces.data();
	constexpr size_t TILE = 2048;
	for (size_t base = 0; base < numFaces; base += TILE) {
		const size_t end = std::min(base + TILE, numFaces);

		for (size_t i = base; i < end; ++i) {
			const Face& gf = faces[cf[i]];

			// Unroll manually for speed
			const uint32_t gv0 = gf[0];
			const uint32_t gv1 = gf[1];
			const uint32_t gv2 = gf[2];

			if (remapGen[gv0] != curGen) {
				remapGen[gv0] = curGen;
				remap[gv0] = (uint32_t)used.size();
				used.push_back(gv0);
			}
			if (remapGen[gv1] != curGen) {
				remapGen[gv1] = curGen;
				remap[gv1] = (uint32_t)used.size();
				used.push_back(gv1);
			}
			if (remapGen[gv2] != curGen) {
				remapGen[gv2] = curGen;
				remap[gv2] = (uint32_t)used.size();
				used.push_back(gv2);
			}
		}

		// remap is sized to vertices.size() up front, so the inserts above never
		// reallocate it and this pointer stays valid for the whole tile
		const uint32_t* __restrict rm = remap.data();
		for (size_t i = base; i < end; ++i) {
			const Face& gf = faces[cf[i]];

			// Localize face (remap global indices)
			lfp[i][0] = rm[gf[0]];
			lfp[i][1] = rm[gf[1]];
			lfp[i][2] = rm[gf[2]];
		}
	}

	// ------------------------------------------------------------------
	// 3. Resize camera-space vertices buffer (computed later)
	// ------------------------------------------------------------------
	out.verts.resize(used.size());   // no clear; overwritten later
}

void UpdateCameraVertsAndNormals(
	const Mesh::VertexArr& vertices,
	const Camera& camera,
	const Mesh::NormalArr& faceNormals,
	CameraRenderData& out
) {
	const size_t numVerts = out.globalVert.size();

	// Camera-space projection matrix P (3x4)
	const float M00 = camera.Pf(0, 0), M01 = camera.Pf(0, 1), M02 = camera.Pf(0, 2), M03 = camera.Pf(0, 3);
	const float M10 = camera.Pf(1, 0), M11 = camera.Pf(1, 1), M12 = camera.Pf(1, 2), M13 = camera.Pf(1, 3);
	const float M20 = camera.Pf(2, 0), M21 = camera.Pf(2, 1), M22 = camera.Pf(2, 2), M23 = camera.Pf(2, 3);

	// ---------------------------------------------------------------
	// 1. Recompute per-camera vertices in CAMERA SPACE
	// ---------------------------------------------------------------
	for (size_t i = 0; i < numVerts; ++i) {
		uint32_t gv = out.globalVert[i];
		const Vertex& v = vertices[gv];

		float xc = M00 * v.x + M01 * v.y + M02 * v.z + M03;
		float yc = M10 * v.x + M11 * v.y + M12 * v.z + M13;
		float zc = M20 * v.x + M21 * v.y + M22 * v.z + M23;

		out.verts[i] = { xc, yc, zc, zc > 1e-6f ? 1.f / zc : 0.f };
	}
	// per-camera normals are no longer materialized; ProjectMesh and
	// ComputePhotometricGradient read faceNormals[globalFace] directly.
}

// extract array of faces viewed by each image
void MeshRefine::ListCameraFaces(bool rebuildOctree)
{
	++faceSetupEpoch;
	// JPB WIP BUG Restrict multithreading?

	// extract array of faces viewed by each camera
	typedef CLISTDEF2(Mesh::FaceIdxArr) CameraFacesArr;

	static CameraFacesArr arrCameraFaces;
	auto& cameraData = g_cameraData;
#if MESHOPT_VIEW_STREAM
	g_cameraFaces = &arrCameraFaces;
#endif
	const int64_t numImages = (int64_t)images.GetSize();
	cameraData.resize(numImages);

#if MESHOPT_VISIBILITY_REUSE
	// reuse the padded-frustum candidate face lists while the accumulated mesh
	// displacement stays within the pad they were culled with (exact superset;
	// see MESHOPT_VISIBILITY_REUSE)
	if (rebuildOctree && candidateCacheValid && accumDispSinceCull <= frustumPadWorld)
		rebuildOctree = false;
#endif

	if (rebuildOctree) {
#if MESHOPT_VISIBILITY_REUSE
		// pad sized from the last applied displacement; non-finite (forced recull
		// markers, e.g. FLT_MAX from the Ceres path) degrades to the unpadded cull
		const float padRaw = MESHOPT_VISIBILITY_PAD_FACTOR * lastIterMaxDisp;
		const float frustumPad = ISFINITE(padRaw) ? padRaw : 0.f;
#endif
		arrCameraFaces.resize(images.GetSize());
		for (auto& cameraFaces : arrCameraFaces) // Never drop capacity in the inner vectors.
			cameraFaces.clear();

		// Build octree
		{
			Mesh::Octree octree;
			Mesh::FacesInserter::CreateOctree(octree, scene.mesh);
			// Frustum-cull each camera against the octree in parallel.
			// octree.Traverse() is const (read-only) and every camera writes
			// its own arrCameraFaces[ID] via a private inserter, so the
			// cameras are fully independent. Result is identical to the
			// previous serial FOREACH (per-camera face lists are unordered
			// sets consumed independently downstream).
#pragma omp parallel for schedule(dynamic)
			for (int64_t ID = 0; ID < numImages; ++ID) {
				const Image& imageData = images[ID];
				if (!imageData.IsValid())
					continue;
				typedef TFrustum<float, 5> Frustum;
				Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
#if MESHOPT_VISIBILITY_REUSE
				// expand the frustum outward by the displacement budget (plane
				// normals are unit-length and point outside the volume)
				if (frustumPad > 0.f)
					for (int p = 0; p < 5; ++p)
						frustum.m_planes[p].m_fD -= frustumPad;
#endif
				Mesh::FacesInserter inserter(arrCameraFaces[ID]);
				octree.Traverse(frustum, inserter);
			}
		}

#if MESHOPT_VIEW_STREAM
		// STREAMING: do NOT expand every camera here. g_cameraData is ~16 B per
		// camera-face plus per-camera vertices -- 4.68 GB across 437 views at the
		// finest scale -- yet the pair loop only ever reads it for the CURRENT
		// batch. Building it for all views made `pinned` 12.23 GB at 32 threads,
		// which drove the budget below the neighbourhood floor and produced 175
		// batches instead of 17. Expanded per batch in EnsureCameraData instead.
		// The candidate lists (4 B per camera-face, ~0.75 GB) are KEPT, since they
		// are the input to that rebuild -- still a large net win over 4.68 GB.
		++cullEpoch; // candidate lists changed -> the expanded topology is stale
#else
		// Build per-camera faces.
#pragma omp parallel for
		for (int64_t ID = 0; ID < numImages; ++ID) {
			if (!images[ID].IsValid()) continue;

			PreprocessCameraFaces(
				arrCameraFaces[ID],
				faces,
				scene.mesh.vertices,
				scene.mesh.faceNormals,
				cameraData[ID]
			);
		}

		// arrCameraFaces has been consumed into cameraData (globalFace); it is dead
		// weight afterward (ThProjectMesh ignores its cameraFaces argument). Release
		// the per-camera face-index lists now so they are not held resident through
		// the memory-peak pair loop. The outer array stays sized so indexing in the
		// (ignored) EVTProjectMesh dispatch remains valid.
		for (auto& cf : arrCameraFaces)
			cf.Release();
#endif

#if MESHOPT_VISIBILITY_REUSE
		candidateCacheValid = true;
		frustumPadWorld = frustumPad;
		accumDispSinceCull = 0.f;
#endif
	}

	// Compute global face normals
	scene.mesh.ComputeNormalFaces();

#if MESHOPT_VIEW_STREAM
	// Deferred per batch (EnsureCameraData). Nothing to invalidate here: the
	// camera-space vertex stamp is compared against faceSetupEpoch, which this
	// call already bumped, so positions are implicitly stale while the expensive
	// topology half survives until the cull is actually rebuilt.
#else
#pragma omp parallel for
	for (int64_t ID = 0; ID < numImages; ++ID) {
		if (!images[ID].IsValid()) continue;

		// Recompute ONLY the camera-space vertices + normals
		// Faces and remapping remain intact
		UpdateCameraVertsAndNormals(
			scene.mesh.vertices,
			images[ID].camera,
			scene.mesh.faceNormals,
			cameraData[ID]
		);
	}
#endif

	// project mesh to each camera plane; under MESHOPT_VIEW_STREAM this is
	// deferred per-batch (EnsureViewsResident) instead of eagerly for all views
#if !MESHOPT_VIEW_STREAM
	ASSERT(events.IsEmpty());
#if MESHOPT_PROFILE
	const auto _tPM0 = std::chrono::steady_clock::now();
#endif
	FOREACH(idxImage, images)
		events.AddEvent(new EVTProjectMesh(idxImage, arrCameraFaces[idxImage], cameraData[idxImage]));
	WaitThreadWorkers(images.GetSize());
#if MESHOPT_PROFILE
	tProjectMeshMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tPM0).count();
#endif
#endif // !MESHOPT_VIEW_STREAM
}

#if MESHOPT_VIEW_STREAM
// streamed bytes for one view at the CURRENT scale (see kStreamBytesPerPixel)
uint64_t MeshRefine::StreamBytesPerView(uint32_t idxImage) const
{
	const View& view = views[idxImage];
	return (uint64_t)view.width * (uint64_t)view.height * kStreamBytesPerPixel;
}

// Decoded-colour bytes for one view. The cache holds the image at the PRE-scale
// decode resolution, not at view.width x view.height, so it is derived by undoing
// the scale factor ThInitImage applied; when the entry is resident the measured
// size is used instead.
uint64_t MeshRefine::DecodeBytesPerView(uint32_t idxImage) const
{
#if MESHOPT_CACHE_IMAGES
	const View& view = views[idxImage];
	const Image8U3& c = view.imageColorCache;
	if (!c.empty())
		return (uint64_t)c.width() * (uint64_t)c.height() * 3;
	const double s = (streamScale > 0 ? (double)streamScale : 1.0);
	return (uint64_t)(((double)view.width / s) * ((double)view.height / s) * 3.0);
#else
	return 0;
#endif
}

// TRUE per-view cost of admitting a view to a batch: the evictable planes PLUS its
// decode-cache entry. The decode cache was falling through the model entirely --
// subtracted from `pinned` as "reclaimable", yet unable to shrink below the current
// batch's own views, which eviction deliberately protects. Measured: 183 views x
// 4.76 MPix x 3 B = 2.6 GB against 2.55 GB observed. Charging it here puts it
// inside the budget BuildViewBatches actually partitions against.
// Bytes the expanded CameraRenderData will occupy for one view. Estimated from the
// RETAINED candidate list, because at budget time nothing is expanded yet: per
// camera-face 12 B (local face) + 4 B (globalFace), plus ~one vertex per two faces
// at 16 B (CamVert) + 4 B (globalVert) => ~26 B per camera-face. Without this the
// budget is computed against g_cameraData == 0 and then reality adds ~0.9 GB per
// batch on top -- measured 15.06 GB peak against a 13.60 GB target.
uint64_t MeshRefine::CameraDataBytesPerView(uint32_t idxImage) const
{
	if (g_cameraFaces == nullptr || idxImage >= g_cameraFaces->size())
		return 0;
	const uint64_t camFaces = (uint64_t)(*g_cameraFaces)[idxImage].GetSize();
	return camFaces * (sizeof(Face) + sizeof(uint32_t))
		+ (camFaces / 2) * (sizeof(CamVert) + sizeof(uint32_t));
}

uint64_t MeshRefine::BatchBytesPerView(uint32_t idxImage) const
{
	return StreamBytesPerView(idxImage) + DecodeBytesPerView(idxImage)
		+ CameraDataBytesPerView(idxImage);
}

// Per-thread pair-loop scratch: every worker holds vertex-sized gradient
// accumulators and full-frame planes, and both scale with the mesh and image rather
// than with the batch, so no view budget can bound them. Never charged anywhere,
// which is why the model read ~2 GB optimistic during the pair loop. An estimate by
// construction -- a RESERVE, not an allocation we control -- so it is built only
// from the buffers that are unambiguously full-frame and thread_local (refMeanA,
// refVarA); the ZNCC/warp scratch is banded under MESHOPT_ZNCC_BANDS and is left
// out rather than over-reserved, since over-reserving costs extra batches.
// Itemize `pinned`. Until now pinned was an opaque measured quantity
// (commit - reclaimable), which was fine while the STREAMABLE state dominated the
// peak. It no longer does: at a 16 GB target, pinned alone (16.90 GB) exceeded the
// whole 13.60 GB ceiling, so what is inside it decides whether that machine is
// reachable at full quality or needs a resolution drop. Everything here is measured
// from the live containers; the `unaccounted` remainder is the number that matters
// -- it is where any remaining dead weight is hiding.
void MeshRefine::LogPinnedBreakdown(uint64_t commit, uint64_t reclaimable, uint64_t scratch) const
{
	constexpr double GBd = 1024.0 * 1024.0 * 1024.0;
	// --- mesh ---
	const Mesh& m = scene.mesh;
	uint64_t bMesh =
		(uint64_t)m.vertices.GetSize() * sizeof(Mesh::Vertex) +
		(uint64_t)m.faces.GetSize() * sizeof(Mesh::Face) +
		(uint64_t)m.faceNormals.GetSize() * sizeof(Mesh::Normal) +
		(uint64_t)m.vertexBoundary.GetSize() * sizeof(bool);
	// adjacency lists are arrays-of-arrays: charge the inner allocations too
	uint64_t bAdj = 0;
	FOREACH(i, m.vertexVertices)
		bAdj += (uint64_t)m.vertexVertices[i].GetSize() * sizeof(Mesh::VIndex) + sizeof(Mesh::VertexIdxArr);
	FOREACH(i, m.vertexFaces)
		bAdj += (uint64_t)m.vertexFaces[i].GetSize() * sizeof(Mesh::FIndex) + sizeof(Mesh::FaceIdxArr);
	// --- per-vertex refinement state ---
	const uint64_t bRefine =
		(uint64_t)photoGrad.size() * sizeof(Grad) +
		(uint64_t)photoGradNorm.GetSize() * sizeof(float) +
		(uint64_t)vertexDepth.GetSize() * sizeof(Depth);
	// --- per-camera render data ---
	// Also report the vertex/face ratio and the capacity slack. A triangle mesh
	// gives V/F ~ 0.5; anything near 3.0 means PreprocessCameraFaces is not
	// deduplicating shared vertices and every face is contributing three.
	uint64_t bCam = 0, camVerts = 0, camFaces = 0, camSlack = 0;
	for (const CameraRenderData& rd : g_cameraData) {
		bCam += (uint64_t)rd.verts.size() * sizeof(CamVert)
			+ (uint64_t)rd.faces.size() * sizeof(Face)
			+ (uint64_t)rd.globalFace.size() * sizeof(uint32_t)
			+ (uint64_t)rd.globalVert.size() * sizeof(uint32_t);
		camVerts += rd.verts.size();
		camFaces += rd.faces.size();
		// reserved-but-unused bytes: globalVert is reserve()d at numFaces while a
		// manifold only needs ~numFaces/2
		camSlack += (uint64_t)(rd.globalVert.capacity() - rd.globalVert.size()) * sizeof(uint32_t)
			+ (uint64_t)(rd.verts.capacity() - rd.verts.size()) * sizeof(CamVert)
			+ (uint64_t)(rd.faces.capacity() - rd.faces.size()) * sizeof(Face);
	}
	// --- the DENSE POINT CLOUD: used once, by SelectNeighborViews in the
	// constructor, then held for the whole run. Prime suspect for dead weight.
	const PointCloud& pc = scene.pointcloud;
	uint64_t bPC =
		(uint64_t)pc.points.GetSize() * sizeof(PointCloud::Point) +
		(uint64_t)pc.normals.GetSize() * sizeof(PointCloud::Normal) +
		(uint64_t)pc.colors.GetSize() * sizeof(PointCloud::Color);
	uint64_t bPCViews = 0;
	FOREACH(i, pc.pointViews)
		bPCViews += (uint64_t)pc.pointViews[i].GetSize() * sizeof(PointCloud::View) + sizeof(PointCloud::ViewArr);
	// --- decoded source images still held by Image objects (separate from our
	// own decode cache, which is already counted in `reclaimable`) ---
	uint64_t bSrcImg = 0;
	FOREACH(i, images)
		bSrcImg += (uint64_t)images[i].image.width() * (uint64_t)images[i].image.height() * images[i].image.elemSize();

	const uint64_t accounted = bMesh + bAdj + bRefine + bCam + bPC + bPCViews + bSrcImg + scratch;
	const uint64_t pinnedMeasured = (commit > reclaimable) ? (commit - reclaimable) : 0;
	const uint64_t pinnedTotal = pinnedMeasured + scratch;
	const int64_t unaccounted = (int64_t)pinnedTotal - (int64_t)accounted;
	DEBUG_EXTRA("[PINNED] %.2f GB total = mesh %.2f + adjacency %.2f + per-vertex %.2f + "
		"g_cameraData %.2f + pointcloud %.2f (+views %.2f) + srcImages %.2f + scratch %.2f "
		"-> UNACCOUNTED %.2f GB",
		pinnedTotal / GBd, bMesh / GBd, bAdj / GBd, bRefine / GBd, bCam / GBd,
		bPC / GBd, bPCViews / GBd, bSrcImg / GBd, scratch / GBd, unaccounted / GBd);
	DEBUG_EXTRA("[PINNED] g_cameraData detail: %llu verts / %llu faces (V/F %.2f -- expect ~0.5 "
		"for a manifold; ~3.0 means shared vertices are NOT being deduplicated), "
		"reserved-but-unused %.2f GB",
		(unsigned long long)camVerts, (unsigned long long)camFaces,
		camFaces ? (double)camVerts / (double)camFaces : 0.0, camSlack / GBd);
}

uint64_t MeshRefine::PairLoopScratchBytes() const
{
	const uint64_t nThreads = (uint64_t)MAXF(size_t(1), threads.GetSize());
	const uint64_t numVerts = (uint64_t)vertices.GetSize();
	uint64_t maxPx = 0;
	FOREACH(i, views) {
		const uint64_t px = (uint64_t)views[i].width * (uint64_t)views[i].height;
		if (px > maxPx)
			maxPx = px;
	}
	constexpr uint64_t kPerVertex = sizeof(Grad) + sizeof(uint32_t); // localGrad + localNorm
	constexpr uint64_t kPerPixel = sizeof(uint16_t) + sizeof(Real);  // refMeanA + refVarA
	uint64_t bytes = nThreads * (numVerts * kPerVertex + maxPx * kPerPixel);
#if MESHOPT_NEEDS_TILE_ENERGY
	// tileEnergyLocal: static thread_local, sized for EVERY view (not just the
	// current batch), one float per 16x16 tile. At 4.65 MPix that is 18k tiles x
	// 437 views x 4 B = ~32 MB per thread -- 1.02 GB at 32 threads, and it was
	// charged to nothing, which is most of the thread scaling the model missed.
	// Plus the per-view (not per-thread) accumulator and active flags.
	const uint64_t tilesPerView = (maxPx + (TILEX * TILEY) - 1) / (TILEX * TILEY);
	const uint64_t numViews = (uint64_t)views.GetSize();
	// Measured current usage, but it lags: at the FIRST budget of a scale the
	// buffers still hold the previous scale's (smaller) tile count, so scale it to
	// this scale's tiles. Floored at one pass' worth (each thread references at
	// least numViews/nThreads reference views) so a cold counter cannot under-charge
	// to zero.
	bytes += nThreads * numViews * tilesPerView * sizeof(float);
	bytes += numViews * tilesPerView * (sizeof(float) + sizeof(uint8_t));
#endif
	return bytes;
}

// Streamed bytes currently held resident. Measured from the actual buffers rather
// than from viewResident, because the two tiers are now independently resident:
// the subdivision pass deliberately holds maps with no tier-1 planes behind them.
uint64_t MeshRefine::StreamBytesResident() const
{
	uint64_t bytes = 0;
	FOREACH(idxImage, views) {
		const View& view = views[idxImage];
		const uint64_t px = (uint64_t)view.width * (uint64_t)view.height;
		if (!view.image.empty())
			bytes += px * kStreamTier1BytesPerPixel;
#if MESHOPT_VIEW_STREAM_EVICT_MAPS
		if (!view.depthMap.empty())
			bytes += px * kStreamTier2BytesPerPixel;
#endif
	}
	return bytes;
}

// bytes actually held by the decoded-colour cache (measured, not modelled: the
// cache holds images at the pre-scale decode resolution, not at view.width x
// view.height, so a per-pixel model of the scaled view would under-count it)
uint64_t MeshRefine::DecodeCacheBytesResident() const
{
	uint64_t bytes = 0;
#if MESHOPT_CACHE_IMAGES
	FOREACH(idxImage, views) {
		if (!decodeResident[idxImage])
			continue;
		const Image8U3& c = views[idxImage].imageColorCache;
		bytes += (uint64_t)c.width() * (uint64_t)c.height() * 3;
	}
#endif
	return bytes;
}

// Derives the ceiling from this machine's total physical RAM alone (see
// MESHOPT_MEM_TARGET_FRACTION / MESHOPT_MEM_RESERVE_GB). Shared by
// MeshRefine::ResolveTargetBytes() below and by Scene::ResolveRefineMeshSafeSettings
// (the pre-flight sizing check, called before a MeshRefine even exists) so there is
// exactly one formula for "how much RAM RefineMesh may use", not two that can drift.
static uint64_t ResolveTargetBytesFromTotal(uint64_t totalPhys)
{
	constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
	if (totalPhys == 0)
		return std::numeric_limits<uint64_t>::max();
	const uint64_t fractionTarget = (uint64_t)((double)totalPhys * MESHOPT_MEM_TARGET_FRACTION);
	const uint64_t reserve = (uint64_t)(MESHOPT_MEM_RESERVE_GB * (double)GB);
	const uint64_t reserveFloorTarget = (totalPhys > reserve) ? (totalPhys - reserve) : 0;
	return MINF(fractionTarget, reserveFloorTarget);
}

// Just the ceiling, without the budget arithmetic or logging: fixed if configured,
// otherwise derived from this machine (see MESHOPT_MEM_TARGET_GB).
uint64_t MeshRefine::ResolveTargetBytes() const
{
	constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
	if (MESHOPT_MEM_TARGET_GB > 0)
		return (uint64_t)(MESHOPT_MEM_TARGET_GB * (double)GB);
	if (MESHOPT_MEM_TARGET_GB < 0)
		return std::numeric_limits<uint64_t>::max();
	const Util::MemoryInfo mi(Util::GetMemoryInfo());
	uint64_t totalPhys = (uint64_t)mi.totalPhysical;
	if (MESHOPT_MEM_SIMULATE_PHYSICAL_GB > 0)
		totalPhys = (uint64_t)(MESHOPT_MEM_SIMULATE_PHYSICAL_GB * (double)GB);
	return ResolveTargetBytesFromTotal(totalPhys);
}

// Resolve the streamed-plane budget for the CURRENT scale. Called from
// BuildViewBatches, i.e. once per scale, not per iteration.
uint64_t MeshRefine::ResolveStreamBudget(uint64_t maxNeighbourhoodBytes, uint64_t totalStreamBytes) const
{
	constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
	uint64_t budget;
	String src;
	if (MESHOPT_VIEW_STREAM_BUDGET_GB < 0) {
		// parity control: never batch
		budget = std::numeric_limits<uint64_t>::max();
		src = _T("unlimited (parity control)");
	} else if (MESHOPT_VIEW_STREAM_BUDGET_GB > 0) {
		budget = (uint64_t)(MESHOPT_VIEW_STREAM_BUDGET_GB * (double)GB);
		src = String::FormatString(_T("fixed %.1f GB"), budget / (double)GB);
	} else if (MESHOPT_MEM_TARGET_GB >= 0) {
		// CEILING mode: budget = target - everything we hold that streaming cannot
		// release. Measured, not modelled, so it stays honest as the mesh grows.
		uint64_t target;
		if (MESHOPT_MEM_TARGET_GB > 0) {
			target = (uint64_t)(MESHOPT_MEM_TARGET_GB * (double)GB);
		} else {
			// derive from this machine (see MESHOPT_MEM_TARGET_GB == 0)
			const Util::MemoryInfo miT(Util::GetMemoryInfo());
			uint64_t totalPhys = (uint64_t)miT.totalPhysical;
			if (MESHOPT_MEM_SIMULATE_PHYSICAL_GB > 0)
				totalPhys = (uint64_t)(MESHOPT_MEM_SIMULATE_PHYSICAL_GB * (double)GB);
			if (totalPhys == 0) {
				// cannot size a ceiling without knowing the machine: hold everything
				// rather than invent a number and batch for no reason
				DEBUG_EXTRA("view-stream budget: unlimited (physical memory query failed)");
				return std::numeric_limits<uint64_t>::max();
			}
			target = ResolveTargetBytesFromTotal(totalPhys);
		}
		const uint64_t commit = ProcessCommitBytes();
		const uint64_t reclaimable = StreamBytesResident() + DecodeCacheBytesResident();
		if (commit == 0) {
			budget = std::numeric_limits<uint64_t>::max();
			src = _T("unlimited (process commit query failed)");
		} else {
			// `commit` is sampled BEFORE the pair loop allocates its per-thread
			// scratch, so measured pinned systematically under-reports what the run
			// will actually hold. Add the reserve explicitly rather than discovering
			// it as a 2 GB overshoot every time.
			const uint64_t scratch = PairLoopScratchBytes();
			uint64_t pinnedMeasured = (commit > reclaimable) ? (commit - reclaimable) : 0;
			// discount memory we freed that the CRT never returned: it inflates
			// commit but is never resident (see retainedFreeBytes)
			pinnedMeasured = (pinnedMeasured > retainedFreeBytes) ? (pinnedMeasured - retainedFreeBytes) : 0;
			const uint64_t pinned = pinnedMeasured + scratch;
#if MESHOPT_MEM_DIAG
			LogPinnedBreakdown(commit, reclaimable, scratch);
#endif
			if (pinned >= target) {
				// Streaming governs a minority of the peak; say so loudly rather
				// than pretending a tiny budget will meet the target.
				budget = 0; // floored to one neighbourhood below
				src = String::FormatString(
					_T("target %.2f GB UNREACHABLE by streaming alone: pinned %.2f GB (commit %.2f - reclaimable %.2f) already exceeds it"),
					target / (double)GB, pinned / (double)GB, commit / (double)GB, reclaimable / (double)GB);
			} else {
				budget = target - pinned;
				src = String::FormatString(
					_T("ceiling %.2f GB (target %.2f - pinned %.2f [measured %.2f + scratch %.2f]; commit %.2f, reclaimable %.2f)"),
					budget / (double)GB, target / (double)GB, pinned / (double)GB,
					pinnedMeasured / (double)GB, scratch / (double)GB,
					commit / (double)GB, reclaimable / (double)GB);
			}
		}
	} else {
		const Util::MemoryInfo mi(Util::GetMemoryInfo());
		uint64_t totalPhys = (uint64_t)mi.totalPhysical;
		uint64_t freePhys = (uint64_t)mi.freePhysical;
		String simNote;
		if (MESHOPT_MEM_SIMULATE_PHYSICAL_GB > 0 && totalPhys > 0) {
			// Model a smaller box: everything the system holds right now would
			// still have to be held there, so the simulated free memory is
			// whatever is left of the smaller box after that same usage. Clamps
			// at 0 rather than going negative -- and 0 free is meaningful here,
			// because `reclaimable` below is still real headroom we can free.
			const uint64_t simTotal = (uint64_t)(MESHOPT_MEM_SIMULATE_PHYSICAL_GB * (double)GB);
			const uint64_t used = totalPhys - freePhys;
			freePhys = (simTotal > used) ? (simTotal - used) : 0;
			totalPhys = simTotal;
			simNote = String::FormatString(_T(" [SIMULATED %.2f GB box; real free was %.1f GB]"),
				simTotal / (double)GB, mi.freePhysical / (double)GB);
		}
		if (totalPhys == 0) {
			// OS query failed: fall back to holding everything rather than
			// guessing a small number and batching hard for no reason
			budget = std::numeric_limits<uint64_t>::max();
			src = _T("unlimited (memory query failed)");
		} else {
			// True headroom = what is free now PLUS what streaming could hand
			// back. The second term is essential: this runs right after
			// SubdivideMesh's geometry-prep + face-area pass, when freePhysical is
			// near its minimum, so raw free RAM would under-report headroom.
			const uint64_t reclaimable = StreamBytesResident() + DecodeCacheBytesResident();
			const uint64_t headroom = freePhys + reclaimable;
			budget = (uint64_t)((double)headroom * MESHOPT_VIEW_STREAM_FREE_FRACTION);
			if (mi.freeVirtual > 0) {
				const uint64_t virtualBudget = (uint64_t)(((double)mi.freeVirtual + reclaimable) * MESHOPT_VIEW_STREAM_FREE_FRACTION);
				if (budget > virtualBudget)
					budget = virtualBudget;
			}
			src = String::FormatString(_T("auto %.1f GB (%.1f free + %.1f reclaimable, x%.2f)%s"),
				budget / (double)GB, freePhys / (double)GB,
				reclaimable / (double)GB, (double)MESHOPT_VIEW_STREAM_FREE_FRACTION,
				simNote.c_str());
		}
	}
	// SPEED FLOOR -- "if I have the RAM, be as fast as before".
	// The 1->2 batch step is a CLIFF, not a slope: measured +37% on scale 3 the
	// moment a second batch appears, and barely more beyond that. So whenever the
	// entire streamed set genuinely fits in measured headroom, take one batch even
	// if the fractional budget said otherwise. Running at ~85% of real headroom
	// instead of 70% is a RISK the in-flight governor can walk back; splitting is a
	// CERTAIN cost. Asymmetric consequences deserve an asymmetric threshold.
	// Deliberately NOT applied when an explicit ceiling is set and cannot fit --
	// a hard target must still be honoured.
	if (budget < totalStreamBytes) {
		const Util::MemoryInfo mi(Util::GetMemoryInfo());
		const uint64_t reclaimable = StreamBytesResident() + DecodeCacheBytesResident();
		const uint64_t headroom = (uint64_t)mi.freePhysical + reclaimable;
		const bool ceilingMode = (MESHOPT_MEM_TARGET_GB >= 0);
		if (!ceilingMode && mi.totalPhysical > 0 && totalStreamBytes <= (uint64_t)(headroom * 0.85)) {
			budget = totalStreamBytes;
			src += String::FormatString(
				_T(" -> raised to %.2f GB: whole set fits in %.2f GB headroom, taking ONE batch (1->2 costs ~37%%)"),
				budget / (double)GB, headroom / (double)GB);
		}
	}
	// A batch must be able to hold at least one reference view plus every view it
	// is paired with, or the BFS below cannot make progress at all.
	if (budget < maxNeighbourhoodBytes) {
		budget = maxNeighbourhoodBytes;
		src += String::FormatString(_T(" -> floored to %.2f GB (largest reference neighbourhood)"), budget / (double)GB);
	}
	DEBUG_EXTRA("view-stream budget: %s", src.c_str());
	return budget;
}

// Partition views into bounded-residency batches from the pair graph
// (refViewNeighbors). Greedy BFS: a batch closes once its resident-view set
// (reference views + their pair-neighbors) would exceed the BYTE budget.
// Rebuilt once per scale: the pair graph is fixed for the whole refinement, but
// bytes/view is not -- it grows ~4x per resolution level, so a partition computed
// at the coarsest scale is meaningless by the finest one.
void MeshRefine::BuildViewBatches()
{
	MESHPROF_SCOPE(P_StreamBatches);
	viewBatches.clear();
	const size_t numViews = images.GetSize();
	if (viewResident.size() != numViews)
		viewResident.assign(numViews, 0);
	if (decodeResident.size() != numViews)
		decodeResident.assign(numViews, 0);
	if (viewProjEpoch.size() != numViews)
		viewProjEpoch.assign(numViews, 0);
	if (viewCamTopoEpoch.size() != numViews)
		viewCamTopoEpoch.assign(numViews, 0);
	if (viewCamVertsEpoch.size() != numViews)
		viewCamVertsEpoch.assign(numViews, 0);
	if (viewNeeded.size() != numViews)
		viewNeeded.assign(numViews, 0);

	// largest single reference-view neighbourhood (floors the budget), and the whole
	// streamed set (lets the budget snap to a single batch when it genuinely fits)
	uint64_t maxNeighbourhoodBytes = 0, totalStreamBytes = 0;
	for (size_t v = 0; v < numViews; ++v) {
		if (refViewNeighbors[v].empty())
			continue;
		totalStreamBytes += BatchBytesPerView((uint32_t)v);
		uint64_t bytes = BatchBytesPerView((uint32_t)v);
		for (uint32_t nb : refViewNeighbors[v])
			bytes += BatchBytesPerView(nb);
		if (bytes > maxNeighbourhoodBytes)
			maxNeighbourhoodBytes = bytes;
	}
	streamBudgetBytes = ResolveStreamBudget(maxNeighbourhoodBytes, totalStreamBytes);

	std::vector<uint8_t> assigned(numViews, 0);
	for (size_t start = 0; start < numViews; ++start) {
		if (assigned[start] || refViewNeighbors[start].empty())
			continue;
		ViewBatch batch;
		std::unordered_set<uint32_t> resident;
		uint64_t residentBytes = 0;
		std::vector<uint32_t> queue{ (uint32_t)start };
		size_t qi = 0;
		while (true) {
			if (qi >= queue.size()) {
				// The BFS queue can drain while the batch still has plenty of budget:
				// views assigned to earlier batches break the pair graph into pieces,
				// so the rest of the scene is unreachable from this seed. Closing the
				// batch here produced a 130/148/14/75 partition (refViews) whose last
				// two batches held 27 and 101 resident views against a ~177-view
				// ceiling -- they fit together comfortably, and every extra batch costs
				// a full evict/reload/re-project pass on EVERY iteration. So when the
				// queue drains, re-seed from the lowest-indexed unassigned reference
				// view that still fits and keep filling. Lowest-indexed, not nearest,
				// so the partition stays deterministic.
				bool seeded = false;
				for (size_t v2 = 0; v2 < numViews; ++v2) {
					if (assigned[v2] || refViewNeighbors[v2].empty())
						continue;
					uint64_t seedBytes = resident.count((uint32_t)v2) ? 0 : BatchBytesPerView((uint32_t)v2);
					for (uint32_t nb : refViewNeighbors[v2])
						if (nb != (uint32_t)v2 && !resident.count(nb))
							seedBytes += BatchBytesPerView(nb);
					if (!batch.refViews.empty() && residentBytes + seedBytes > streamBudgetBytes)
						continue;
					queue.push_back((uint32_t)v2);
					seeded = true;
					break;
				}
				if (!seeded)
					break;
			}
			const uint32_t v = queue[qi++];
			if (assigned[v])
				continue;
			// cost of admitting v = v itself plus any neighbour not already resident
			uint64_t addBytes = resident.count(v) ? 0 : BatchBytesPerView(v);
			for (uint32_t nb : refViewNeighbors[v])
				if (nb != v && !resident.count(nb))
					addBytes += BatchBytesPerView(nb);
			// Skip a view that would blow the budget, but KEEP SCANNING rather than
			// closing the batch. Breaking here abandoned the rest of the queue, and
			// those views then formed their own tiny batches -- a 22-batch partition
			// had seven batches of <=23 views (one of 3), each paying a full
			// evict/reload/re-project transition for a fraction of a batch's work.
			// Later queue entries are often nearly free to admit because their
			// neighbours are already resident, so continuing fills the batch instead.
			// The first reference view is always admitted (the budget floor
			// guarantees one neighbourhood fits).
			if (!batch.refViews.empty() && residentBytes + addBytes > streamBudgetBytes)
				continue;
			assigned[v] = 1;
			batch.refViews.push_back(v);
			resident.insert(v);
			for (uint32_t nb : refViewNeighbors[v]) {
				resident.insert(nb);
				if (!assigned[nb])
					queue.push_back(nb);
			}
			residentBytes += addBytes;
		}
		if (!batch.refViews.empty()) {
			batch.allViews.assign(resident.begin(), resident.end());
			// Deterministic order. allViews: the BFS/hash-set order is not stable
			// across runs. refViews: sorting makes the SINGLE-BATCH case walk the
			// reference views in exactly the ascending order the non-streamed loop
			// uses, so the parity control differs from it in nothing but where
			// ProjectMesh is dispatched from. Ordering within a batch does not
			// affect residency (allViews drives that), so this costs nothing.
			std::sort(batch.allViews.begin(), batch.allViews.end());
			std::sort(batch.refViews.begin(), batch.refViews.end());
			viewBatches.push_back(std::move(batch));
		}
	}
	streamSingleBatch = (viewBatches.size() <= 1);
	if (!streamSingleBatch) {
		// Imbalance is expensive twice over: a small batch still pays a full
		// evict/reload transition, and the largest batch sets the per-thread scratch.
		std::string sizes;
		for (const ViewBatch& b : viewBatches) {
			char buf[64];
			sprintf(buf, "%s%zu/%zu", sizes.empty() ? "" : ", ", b.refViews.size(), b.allViews.size());
			sizes += buf;
		}
		DEBUG_EXTRA("view-stream batch balance (refViews/resident): %s", sizes.c_str());
	}
	// The parity claim lives or dies here: one batch means EnsureViewsResident is
	// called exactly once per ScoreMesh over every view, which is precisely what
	// the non-streamed path does in ListCameraFaces. Log it so a "streaming is
	// slower" measurement can always be attributed to batching or ruled out.
	uint64_t totalBytes = 0;
	for (size_t v = 0; v < numViews; ++v)
		if (!refViewNeighbors[v].empty())
			totalBytes += BatchBytesPerView((uint32_t)v);
	constexpr double GBd = 1024.0 * 1024.0 * 1024.0;
	DEBUG_EXTRA("view-stream: %zu batch(es) over %zu views, %.2f GB streamed planes total%s",
		viewBatches.size(), numViews, totalBytes / GBd,
		streamSingleBatch ? " [SINGLE BATCH -> non-streamed parity]" : "");
	// A budget that is only a small multiple of ONE reference neighbourhood cannot
	// batch usefully: every batch transition evicts nearly everything, so each
	// iteration rebuilds ~every view's gradient planes. Measured at 60 batches on a
	// 437-view scene: 2.8x total wall time for a 2% peak reduction. The peak in that
	// case was already set BEFORE the pair loop by SubdivideMesh's full-residency
	// pass, which no view budget can influence -- so shrinking the budget further
	// buys nothing and costs everything. Say so rather than silently thrashing.
	if (!streamSingleBatch && viewBatches.size() > 8) {
		DEBUG_EXTRA("WARNING: view-stream batching is deep -- %zu batches (budget %.2f GB vs "
			"%.2f GB for the largest single reference neighbourhood). Every batch transition "
			"rebuilds the planes it evicted, so each iteration re-does roughly a full pass of "
			"ThInitImage + ProjectMesh. The budget does now bound the pair loop, so this is a "
			"real speed-for-memory trade rather than waste -- but if it is deeper than intended, "
			"raise MESHOPT_MEM_TARGET_GB or cut the PINNED footprint (mesh, g_cameraData, "
			"per-thread scratch), which buys batches far more cheaply than shrinking the budget.",
			viewBatches.size(), streamBudgetBytes / GBd, maxNeighbourhoodBytes / GBd);
	}
}

#if MESHOPT_VIEW_STREAM
// Expand one view's CameraRenderData from its retained candidate list, if it is not
// already current for this mesh generation. This is the per-batch replacement for
// the all-views loop that used to run inside ListCameraFaces.
void MeshRefine::EnsureCameraData(uint32_t idxImage)
{
	ASSERT(g_cameraFaces != nullptr);
	// expensive half: only when the cull/topology changed (~twice per scale)
	if (viewCamTopoEpoch[idxImage] != cullEpoch) {
		PreprocessCameraFaces(
			(*g_cameraFaces)[idxImage],
			faces,
			scene.mesh.vertices,
			scene.mesh.faceNormals,
			g_cameraData[idxImage]);
		viewCamTopoEpoch[idxImage] = cullEpoch;
		viewCamVertsEpoch[idxImage] = 0; // positions must follow a fresh expansion
	}
	// cheap half: vertices moved, so this is stale every iteration
	if (viewCamVertsEpoch[idxImage] != faceSetupEpoch) {
		UpdateCameraVertsAndNormals(
			scene.mesh.vertices,
			images[idxImage].camera,
			scene.mesh.faceNormals,
			g_cameraData[idxImage]);
		viewCamVertsEpoch[idxImage] = faceSetupEpoch;
	}
}

// drop one view's expanded CameraRenderData (the candidate list it was built from
// is kept, so it can be rebuilt cheaply when the view re-enters a batch)
void MeshRefine::ReleaseCameraData(uint32_t idxImage)
{
	CameraRenderData& rd = g_cameraData[idxImage];
	std::vector<CamVert>().swap(rd.verts);
	std::vector<Face>().swap(rd.faces);
	std::vector<uint32_t>().swap(rd.globalFace);
	std::vector<uint32_t>().swap(rd.globalVert);
	viewCamTopoEpoch[idxImage] = 0;
	viewCamVertsEpoch[idxImage] = 0;
}
#endif

// release a view's evictable state.
//   TIER 1 (always): decoded image + gradient planes -- rebuilt by ThInitImage.
//   TIER 2 (MESHOPT_VIEW_STREAM_EVICT_MAPS): the rasterized depthMap/faceMap.
// Clearing viewProjEpoch is the correctness linchpin for tier 2: the stamp asserts
// "this view's maps are valid for mesh generation N", and once the maps are gone
// that claim must not survive. Zeroing it routes the view back through the normal
// staleness test in EnsureViewsResident, which re-projects it on re-entry -- no
// separate code path, and VerifyBatchResidency fails loudly if it is ever missed.
void MeshRefine::EvictView(uint32_t idxImage, bool alsoMaps)
{
	View& view = views[idxImage];
	view.image.release();
	PlaneFree(view.gradX); view.gradX = nullptr;
	PlaneFree(view.gradY); view.gradY = nullptr;
	viewResident[idxImage] = 0;
#if MESHOPT_VIEW_STREAM_EVICT_MAPS
	if (alsoMaps) {
		view.depthMap.release();
		view.faceMap.release();
#if MESHOPT_DIRECT_VALIDITY
		// swap-with-empty, not clear(): clear() keeps the capacity, which would
		// leave the ~2 GB allocated and make this eviction tier a no-op
		{ std::vector<uint8_t, AlignedAllocator<uint8_t, 16>>().swap(view.isValid); }
#endif
		viewProjEpoch[idxImage] = 0;
		ReleaseCameraData(idxImage);
	}
#endif
}

// Drop the TIER-1 planes for every view while keeping the rasterized maps (and
// therefore their epoch stamps) intact. Used by the subdivision face-area pass:
// ListFaceAreas reads faceMap + g_cameraData and never touches a pixel, so the
// ~12 GB of image/gradient data loaded purely to rescale the cameras does not
// need to coexist with the maps that pass actually consumes.
// Rescale every view's camera and dimensions for the current scale WITHOUT ever
// holding more than a chunk of pixel data at once.
//
// The subdivision pass wants only ThInitImage's SIDE EFFECTS: UpdateCamera rescales
// the intrinsics for this scale, and view.width/height get set. It never reads a
// pixel. Loading all 437 views up front cost 437 x 4.65 MPix x 6 B = 12.19 GB of
// peak commit for data released moments later -- and after the maps were streamed
// out of this pass, that load WAS the run's peak (29.73 GB, vs 27.13 GB for the
// pair loop it is supposed to be bounded by). Chunking bounds it to
// chunk x 6 B/px (~0.9 GB at 32 views).
//
// Identical results by construction: the same ThInitImage calls, in the same order,
// with the same arguments -- only the number resident simultaneously changes. In
// particular this does NOT try to recompute the post-resize dimensions analytically,
// which would have to reproduce cv::resize's rounding exactly to stay correct.
void MeshRefine::PrepareViewGeometryChunked()
{
	MESHPROF_SCOPE(P_StreamPrep);
	std::vector<uint32_t> all;
	all.reserve(images.GetSize());
	FOREACH(idxImage, images)
		if (images[idxImage].IsValid())
			all.push_back((uint32_t)idxImage);
	const size_t chunk = MAXF(size_t(1), threads.GetSize());
	for (size_t i = 0; i < all.size(); i += chunk) {
		const size_t end = MINF(i + chunk, all.size());
		ASSERT(events.IsEmpty());
		for (size_t k = i; k < end; ++k)
			events.AddEvent(new EVTInitImage(all[k], streamScale, streamSigma));
		WaitThreadWorkers(end - i);
		// ThInitImage populates imageColorCache, so record those entries: this path
		// bypasses EnsureViewsResident, and without the bookkeeping the cache is
		// invisible to DecodeCacheBytesResident. That understated `reclaimable` by
		// its full size at the first budget of a scale (observed: reclaimable 0.00
		// while the very next probe reported decode 6.09 GB), inflating `pinned` by
		// the same amount and shrinking the budget -- conservative, but wrong, and
		// on a tighter box it would force batching that is not needed.
		for (size_t k = i; k < end; ++k) {
			const uint32_t v = all[k];
			if (!decodeResident[v]) {
				decodeResident[v] = 1;
				decodeOrder.push_back(v);
			}
		}
		// Decide ONCE, after the first chunk has given us real view dimensions,
		// whether the whole streamed set will fit under the ceiling. If it will,
		// the pair loop is about to reload every one of these views anyway, so
		// releasing here just buys a second full ThInitImage pass per scale --
		// measured as a large part of a 55s -> 76s regression on a scene that was
		// single-batch at every scale. If it will not fit, release as before: the
		// subdivision pass must not hold planes it does not read.
		if (i == 0) {
			const uint64_t perView = BatchBytesPerView(all[0]);
			const uint64_t estTotal = perView * (uint64_t)all.size();
			const uint64_t target = ResolveTargetBytes();
			const uint64_t commit = ProcessCommitBytes();
			streamKeepPlanes = (target == std::numeric_limits<uint64_t>::max())
				|| (commit + estTotal < target);
		}
		if (!streamKeepPlanes) {
			// drop the pixels; keep only camera + dimensions. alsoMaps=false because
			// no maps exist yet at this point and none should be invalidated.
			for (size_t k = i; k < end; ++k)
				EvictView(all[k], false);
		}
	}
}

void MeshRefine::ReleaseAllViewPlanes()
{
	FOREACH(idxImage, views)
		if (viewResident[idxImage])
			EvictView((uint32_t)idxImage, false);
}

// Rasterize every valid view whose maps are stale for the current mesh generation,
// WITHOUT loading tier-1 planes. Distinct from EnsureAllViewsResident(true), which
// would treat the just-released planes as missing and reload all of them.
void MeshRefine::ProjectAllViews()
{
	static const Mesh::FaceIdxArr kEmptyFaces;
	std::vector<uint32_t> toProject;
	FOREACH(idxImage, images)
		if (images[idxImage].IsValid() && viewProjEpoch[idxImage] != faceSetupEpoch)
			toProject.push_back((uint32_t)idxImage);
	if (toProject.empty())
		return;
	ASSERT(events.IsEmpty());
	for (uint32_t v : toProject)
		events.AddEvent(new EVTProjectMesh(v, kEmptyFaces, g_cameraData[v]));
	WaitThreadWorkers(toProject.size());
	for (uint32_t v : toProject)
		viewProjEpoch[v] = faceSetupEpoch;
}

// release the cheaper decoded-color cache entry (see EnsureViewsResident)
void MeshRefine::EvictDecodeCache(uint32_t idxImage)
{
	views[idxImage].imageColorCache.release();
	decodeResident[idxImage] = 0;
}

// load any missing views' static data + refresh their per-iteration depthMap/
// faceMap, evicting views this batch doesn't need. Bit-identical to the
// resident path: same ThInitImage/ThProjectMesh, only order + residency differ.
void MeshRefine::EnsureViewsResident(const std::vector<uint32_t>& neededViews, bool doProject)
{
	MESHPROF_SCOPE(P_StreamResident);
	// Mark the needed set once (O(n)) instead of scanning neededViews for every
	// resident view: at 367 views that inner search ran ~134k times per call.
	for (uint32_t v : neededViews)
		viewNeeded[v] = 1;
	// Evict on ACTUAL BUFFER PRESENCE, not on the viewResident flag. The two are
	// not the same thing: ReleaseAllViewPlanes drops the tier-1 planes and clears
	// viewResident while deliberately KEEPING the rasterized maps, so a view can
	// hold ~8 B/px of depthMap/faceMap with viewResident == 0. Gating on the flag
	// made every such view invisible to this scan, and the maps of all 437 views
	// (~16 GB) then survived the entire first pass through the batches -- which is
	// exactly where the peak was measured (24.25 GB resident on the first pass vs
	// 14.92 GB on every later one, commit peaking at 45.26 GB).
	FOREACH(idxImage, views) {
		if (viewNeeded[idxImage])
			continue;
		const View& view = views[idxImage];
		const bool holdsAnything =
			viewResident[idxImage] || !view.image.empty() || view.gradX || view.gradY
			|| !view.depthMap.empty() || !view.faceMap.empty();
		if (holdsAnything)
			EvictView((uint32_t)idxImage);
	}
	std::vector<uint32_t> toLoad;
	for (uint32_t v : neededViews)
		if (!viewResident[v])
			toLoad.push_back(v);
	// Bound the decode cache (it deliberately survives EvictView above) so it
	// cannot just grow to full residency. Budgeted in BYTES against the measured
	// cache size, since entries are held at the pre-scale decode resolution and
	// so are not all the same size. FIFO: an entry's value is "some future batch
	// re-enters this view", which age predicts as well as recency here.
	// The decode cache is a pure SPEED cache; the per-view planes are REQUIRED data.
	// So it gets whatever this batch's planes leave over, scaled by the ratio knob:
	// unbounded on a roomy box (the budget dwarfs one batch, so nothing is ever
	// evicted and the fast path is untouched), squeezed to nothing exactly when the
	// target binds. The previous `ratio * streamBudget` had it backwards -- at a
	// 17 GB budget the 6.2 GB cache never hit its bound, so it stayed fully resident
	// while the planes it was competing with got squeezed into extra batches.
	uint64_t neededPlaneBytes = 0;
	for (uint32_t v : neededViews)
		neededPlaneBytes += BatchBytesPerView(v);
	// SINGLE BATCH -> never bound the cache. No view ever re-enters residency, so
	// the cache cannot pay for itself by avoiding a reload, and evicting it only
	// buys a re-decode of every image at the next scale. This case is not
	// hypothetical: the speed floor raises the budget to exactly totalStreamBytes,
	// so `budget - neededPlaneBytes` is exactly 0 and the leftovers rule would
	// otherwise evict the entire cache on precisely the fast path it must not touch.
	const uint64_t decodeBudget =
		(streamSingleBatch || streamBudgetBytes == std::numeric_limits<uint64_t>::max())
		? std::numeric_limits<uint64_t>::max()
		: (streamBudgetBytes > neededPlaneBytes
			? (uint64_t)((streamBudgetBytes - neededPlaneBytes) * MESHOPT_VIEW_STREAM_DECODE_BUDGET_RATIO)
			: 0);
	// Admit first, then enforce the budget UNCONDITIONALLY. The enforcement used to
	// live inside this admission loop, behind `if (decodeResident[v]) continue;` --
	// so once every view had been cached once, nothing was ever new, the `continue`
	// fired every time, and the eviction pass never ran at all. The cache sat at
	// 6.09 GB for an entire run even at a computed budget of ~0.01 GB, i.e. 6 GB the
	// budget counted as reclaimable and never reclaimed.
	for (uint32_t v : toLoad) {
		if (!decodeResident[v]) {
			decodeResident[v] = 1;
			decodeOrder.push_back(v);
		}
	}
	uint64_t decodeBytes = DecodeCacheBytesResident();
	// `guard` bounds this to a single pass over the queue: entries needed by THIS
	// call are rotated to the back rather than evicted, and without the bound a
	// queue consisting entirely of needed views would spin forever.
	size_t guard = decodeOrder.size();
	while (decodeOrder.size() > 1 && decodeBytes > decodeBudget && guard-- > 0) {
		const uint32_t victim = decodeOrder.front();
		decodeOrder.pop_front();
		if (viewNeeded[victim]) {
			decodeOrder.push_back(victim); // still in use; try the next oldest
			continue;
		}
		const Image8U3& c = views[victim].imageColorCache;
		const uint64_t victimBytes = (uint64_t)c.width() * (uint64_t)c.height() * 3;
		decodeBytes = (victimBytes < decodeBytes) ? decodeBytes - victimBytes : 0;
		EvictDecodeCache(victim);
	}
	ASSERT(events.IsEmpty());
	for (uint32_t v : toLoad)
		events.AddEvent(new EVTInitImage(v, streamScale, streamSigma));
	WaitThreadWorkers(toLoad.size());
	for (uint32_t v : toLoad)
		viewResident[v] = 1;
	// depthMap/faceMap depend on the current mesh geometry, which changes exactly
	// once per ScoreMesh (ListCameraFaces bumps faceSetupEpoch at its head). Only
	// project views whose maps are stale for THIS epoch: a view that an earlier
	// batch already projected in this same call is still valid, because the mesh
	// cannot move mid-ScoreMesh. Without this, every view was re-projected once
	// per batch that listed it as a neighbour -- the ~2-3x ProjectMesh multiplier
	// that made streaming lose to the non-streamed path even with RAM to spare.
	//
	// doProject=false loads WITHOUT projecting. ThInitImage is what rescales the
	// camera intrinsics for this scale (UpdateCamera), so on the first residency
	// pass of a scale g_cameraData is still built from the PREVIOUS scale's
	// cameras -- projecting against it rasterizes the mesh at the wrong scale into
	// a correctly-sized buffer. The caller loads first (doProject=false), rebuilds
	// g_cameraData via ListCameraFaces, then projects.
	// NOTE: CameraRenderData is expanded inside ThProjectMesh, NOT here. This
	// function runs under `#pragma omp single`, so doing it here serialised a
	// per-view camera-space vertex pass onto one thread while 31 waited -- the
	// non-streamed path runs the same work under `#pragma omp parallel for`.
	// ThProjectMesh already runs on the worker pool, one view per task, and each
	// view touches only its own g_cameraData entry, so it parallelises for free.
	static const Mesh::FaceIdxArr kEmptyFaces;
	std::vector<uint32_t> toProject;
	if (doProject)
		for (uint32_t v : neededViews)
			if (viewProjEpoch[v] != faceSetupEpoch)
				toProject.push_back(v);
	if (!toProject.empty()) {
		ASSERT(events.IsEmpty());
		for (uint32_t v : toProject)
			events.AddEvent(new EVTProjectMesh(v, kEmptyFaces, g_cameraData[v]));
		WaitThreadWorkers(toProject.size());
		for (uint32_t v : toProject)
			viewProjEpoch[v] = faceSetupEpoch;
	}
	// reset the scratch mask for the next call
	for (uint32_t v : neededViews)
		viewNeeded[v] = 0;
}

#if MESHOPT_VIEW_STREAM_VERIFY
// Fail loudly if a batch is about to read data that is not there. Batching's one
// dangerous failure mode is silent: reading an evicted or stale-epoch view does
// not crash, it just quietly corrupts the gradient for those pairs.
void MeshRefine::VerifyBatchResidency(const std::vector<uint32_t>& allViews,
	const std::vector<uint32_t>& refViews) const
{
	size_t notLoaded = 0, staleProj = 0, missingEndpoint = 0, missingMap = 0;
	for (uint32_t v : allViews) {
		const View& view = views[v];
		if (!viewResident[v] || view.image.empty() || !view.gradX || !view.gradY)
			++notLoaded;
		// depthMap/faceMap must match the CURRENT mesh generation
		if (viewProjEpoch[v] != faceSetupEpoch)
			++staleProj;
		// under tier-2 eviction the maps really can be gone, so check they are
		// actually back -- a released map reads as all-zero, not as a crash
		if (view.depthMap.empty())
			++missingMap;
#if MESHOPT_FACEMAP_RESIDENT
		if (view.faceMap.empty())
			++missingMap;
#endif
	}
	// every directed pair this batch will run reads BOTH endpoints: A supplies
	// faceMap/depthMap/image, B supplies image + gradX/gradY + depthMap
	for (uint32_t a : refViews)
		for (uint32_t b : refViewNeighbors[a])
			if (!viewResident[b] || views[b].image.empty() || !views[b].gradX)
				++missingEndpoint;
	if (notLoaded || staleProj || missingEndpoint || missingMap)
		ABORT("view-stream residency violation (batch: %zu refs, %zu views): "
			"%zu not loaded, %zu stale projections, %zu missing maps, %zu pair endpoints missing",
			refViews.size(), allViews.size(), notLoaded, staleProj, missingMap, missingEndpoint);
}
#endif

// spike to full residency for the one-off subdivision face-area pass, since it
// needs every view's image size; the next ScoreMesh call re-bounds it
void MeshRefine::EnsureAllViewsResident(bool doProject)
{
	std::vector<uint32_t> all;
	all.reserve(images.GetSize());
	FOREACH(idxImage, images)
		if (images[idxImage].IsValid())
			all.push_back((uint32_t)idxImage);
	EnsureViewsResident(all, doProject);
}
#endif // MESHOPT_VIEW_STREAM

__forceinline void AtomicInc16(uint16_t& x) noexcept
{
	_InterlockedIncrement16(reinterpret_cast<volatile SHORT*>(&x));
}

__forceinline void AtomicMax16(uint16_t& dest, uint16_t value) noexcept
{
	volatile SHORT* ptr = reinterpret_cast<volatile SHORT*>(&dest);
	SHORT old = *ptr;
	while ((USHORT)old < (USHORT)value) {   // unsigned compare — matters if areas can exceed 32767
		SHORT prev = _InterlockedCompareExchange16(ptr, (SHORT)value, old);
		if (prev == old) break;
		old = prev;
	}
}

// compute for each face the projection area as the maximum area in both images of a pair
// (make sure ListCameraFaces() was called before)
#if MESHOPT_RASTER_SPAN_DIAG
// defined further down, next to the raster loops it instruments
static void SpanDiagReport(const char* tag);
#endif

void MeshRefine::ListFaceAreas(Mesh::AreaArr& maxAreas, bool streamMaps)
{
	MESHPROF_SCOPE(P_StreamFaceAreas);
#if 1
	ASSERT(maxAreas.IsEmpty());

	typedef cList<Mesh::AreaArr> ImageAreaArr;
	ImageAreaArr viewAreas(images.GetSize());

#pragma omp parallel for schedule(dynamic, 1)
	for (int idxImage = 0; idxImage < (int)images.GetSize(); ++idxImage) {
		const Image& imageData = images[idxImage];
		if (!imageData.IsValid())
			continue;

		Mesh::AreaArr& areas = viewAreas[idxImage];
		areas.Resize(faces.GetSize());          // global-face sized
		areas.Memset(0);

#if MESHOPT_VIEW_STREAM
		// streamMaps: rasterize THIS view's maps here and drop them immediately, so
		// the subdivision face-area pass holds ~nThreads sets instead of all 437.
		// This pass reads faceMap one view at a time and never needs two at once,
		// yet it was the single largest resident state in the run: 437 x 9 B/px
		// (~16 GB at 4.65 MPix) purely so a sequential counting loop could read
		// them. Streamed, that becomes nThreads x 9 B/px (~1.3 GB). Independent of
		// MESHOPT_FACEMAP_RESIDENT, which governs the PAIR LOOP, where residency is
		// worth ~12-14% and must stay.
		if (streamMaps) {
			EnsureCameraData((uint32_t)idxImage);
			ProjectMesh(views[idxImage], imageData.camera, g_cameraData[idxImage]);
		}
#endif
#if !MESHOPT_FACEMAP_RESIDENT
		// faceMap is streamed (not resident): ProjectMesh above rasterizes into a
		// THREAD-LOCAL scratch and discards it, so view.faceMap is still empty here.
		// Regenerate it onto the view, and do it AFTER EnsureCameraData: under
		// VIEW_STREAM the CameraRenderData is expanded lazily, and rasterizing
		// against an unexpanded entry yields an all-NO_ID map with no error --
		// measured as `[SUBDIV] 0/2720158 faces over maxArea=32` at every scale,
		// i.e. subdivision silently stopped and the mesh came out at input size.
#if MESHOPT_VIEW_STREAM
		EnsureCameraData((uint32_t)idxImage);
#endif
		RasterizeFaceMap(views[idxImage], g_cameraData[idxImage]);
#endif
		const FaceMap& faceMap = views[idxImage].faceMap;
		const CameraRenderData& rd = g_cameraData[idxImage];
		const uint32_t numLocal = (uint32_t)rd.globalFace.size();

		for (int j = 0; j < faceMap.rows; ++j) {
			const FIndex* facePtr = faceMap.ptr<FIndex>(j);
			for (int i = 0; i < faceMap.cols; ++i) {
				const FIndex idxLocal = facePtr[i];
				if (idxLocal == NO_ID)
					continue;
				ASSERT(idxLocal < numLocal);
				const FIndex idxGlobal = rd.globalFace[idxLocal];   // <-- translate
				++areas[idxGlobal];
			}
		}
#if !MESHOPT_FACEMAP_RESIDENT
		views[idxImage].faceMap.release();
#endif
#if MESHOPT_VIEW_STREAM
		if (streamMaps) {
			// release everything ProjectMesh just produced. The stamp must go with
			// them: it asserts "these maps are valid for the current generation",
			// and they no longer exist. The next ScoreMesh re-projects whatever its
			// batches need -- which it would do anyway, since SubdivideMesh changes
			// the geometry immediately after this pass.
			View& view = views[idxImage];
			view.faceMap.release();
			view.depthMap.release();
			{ std::vector<uint8_t, AlignedAllocator<uint8_t, 16>>().swap(view.isValid); }
			viewProjEpoch[idxImage] = 0;
			// Release the expanded CameraRenderData only when memory is actually
			// tight. Its expensive half (PreprocessCameraFaces) is keyed to
			// cullEpoch, which the subdivision about to happen does NOT change --
			// so dropping it here just forces a second full expansion in the pair
			// loop, once per scale. The maps above must still go: subdivision does
			// invalidate those.
			if (!streamKeepPlanes)
				ReleaseCameraData((uint32_t)idxImage);
		}
#endif
	}

	maxAreas.Resize(faces.GetSize());
	maxAreas.Memset(0);

#pragma omp parallel for
	for (int p = 0; p < (int)pairs.size(); ++p) {
		const auto& pair = pairs[p];
		const Mesh::AreaArr& areasA = viewAreas[pair.i];
		const Mesh::AreaArr& areasB = viewAreas[pair.j];
		ASSERT(areasA.GetSize() == areasB.GetSize());
		const size_t n = areasA.size();
		for (size_t f = 0; f < n; ++f) {
			const uint16_t minArea = MINF(areasA[f], areasB[f]);
			AtomicMax16(maxAreas[f], minArea);
		}
	}
#else
	// original
	ASSERT(maxAreas.IsEmpty());
	// for each image, compute the projection area of visible faces
	typedef cList<Mesh::AreaArr> ImageAreaArr;
	ImageAreaArr viewAreas(images.GetSize());
	FOREACH(idxImage, images) {
		const Image& imageData = images[idxImage];
		if (!imageData.IsValid())
			continue;
		Mesh::AreaArr& areas = viewAreas[idxImage];
		areas.Resize(faces.GetSize());
		areas.Memset(0);
		const FaceMap& faceMap = views[idxImage].faceMap;
		// compute area covered by all vertices (incident faces) viewed by this image
		for (int j = 0; j < faceMap.rows; ++j) {
			for (int i = 0; i < faceMap.cols; ++i) {
				const FIndex idxFace(faceMap(j, i));
#if MESHOPT_RASTER_NO_BACKFACE_CULL
				// back-face pixels carry depth but NO_ID (GPU parity)
				ASSERT(idxFace == NO_ID || views[idxImage].depthMap(j, i) > 0);
#else
				ASSERT((idxFace == NO_ID && views[idxImage].depthMap(j, i) == 0) || (idxFace != NO_ID && views[idxImage].depthMap(j, i) > 0));
#endif
				if (idxFace == NO_ID)
					continue;
				++areas[idxFace];
			}
		}
	}

	maxAreas.Resize(faces.GetSize());
	maxAreas.Memset(0);
	FOREACHPTR(pPair, pairs) {
		const Mesh::AreaArr& areasA = viewAreas[pPair->i];
		const Mesh::AreaArr& areasB = viewAreas[pPair->j];
		ASSERT(areasA.GetSize() == areasB.GetSize());
		FOREACH(f, areasA) {
			const uint16_t minArea(MINF(areasA[f], areasB[f]));
			uint16_t& maxArea = maxAreas[f];
			if (maxArea < minArea)
				maxArea = minArea;
		}
	}
#endif

#if MESHOPT_RASTER_SPAN_DIAG
	// cumulative over every raster call so far; runs once per SubdivideMesh
	SpanDiagReport("ListFaceAreas");
#endif
}

// decimate or subdivide mesh such that for each face there is no image pair in which
// its projection area is bigger than the given number of pixels in both images
void MeshRefine::SubdivideMesh(uint32_t maxArea, float fDecimate, unsigned nCloseHoles, unsigned nEnsureEdgeSize)
{
	Mesh::AreaArr maxAreas;

	// first decimate if necessary
	const bool bNoDecimation(fDecimate >= 1.f);
	const bool bNoSimplification(maxArea == 0);
	if (!bNoDecimation) {
		if (fDecimate > 0.f) {
			// decimate to the desired resolution
			scene.mesh.Clean(fDecimate, 0.f, false, nCloseHoles, 0u, 0.f, false);
			scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);

#ifdef MESHOPT_ENSUREEDGESIZE
			// make sure there are no edges too small or too long
			if (nEnsureEdgeSize > 0 && bNoSimplification) {
				scene.mesh.EnsureEdgeSize();
				scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);
			}
#endif

			// re-map vertex and camera faces
			ListVertexFacesPre();
		}
		else {
			// extract array of faces viewed by each camera
			ListCameraFaces();
#if MESHOPT_VIEW_STREAM
#if MESHOPT_MEM_DIAG
			DEBUG_EXTRA("[MEM] before chunked camera/dimension rescale"); Util::LogMemoryInfo();
#endif
			// Rescale cameras/dimensions for this scale, in chunks so the pixel
			// data never all coexists. Must happen BEFORE the second ListCameraFaces:
			// g_cameraData above is still built from the PREVIOUS scale's cameras,
			// and rasterizing against it would place the mesh at the old scale in a
			// new-resolution buffer -> ListFaceAreas under-counts by ~(old/new)^2 ->
			// subdivision under-triggers, compounding every scale.
			PrepareViewGeometryChunked();
#if MESHOPT_MEM_DIAG
			DEBUG_EXTRA("[MEM] after chunked camera/dimension rescale"); Util::LogMemoryInfo();
#endif
			// camera intrinsics only get rescaled to this resolution inside
			// EnsureAllViewsResident's ThInitImage; redo the camera-space vertex
			// projection now that they're correct (candidate lists are reused,
			// cheap, since nothing moved between the two calls)
			ListCameraFaces();
			// The tier-1 planes were loaded ONLY to rescale the cameras above;
			// ListFaceAreas below reads faceMap + g_cameraData and never a pixel.
			// Drop them before rasterizing so the maps do not have to coexist with
			// ~12 GB of image/gradient data this pass does not read.
#endif

			// estimate the faces' area that have big projection areas in both images of a pair
			// (streamMaps under VIEW_STREAM: each view is rasterized and released
			// inside the loop, so this pass never holds all 437 map sets at once;
			// the tier-1 planes were already released chunk-by-chunk above)
#if MESHOPT_VIEW_STREAM
			ListFaceAreas(maxAreas, true);
#else
			ListFaceAreas(maxAreas);
#endif
			ASSERT(!maxAreas.IsEmpty());
#if MESHOPT_MEM_DIAG
			{
				size_t nonZero = 0; uint64_t sumArea = 0;
				FOREACH(fi, maxAreas) { if (maxAreas[fi] > 0) { ++nonZero; sumArea += maxAreas[fi]; } }
				DEBUG_EXTRA("[MEM] maxAreas (auto-decimate estimate): %zu/%zu faces nonzero, sum=%llu", nonZero, maxAreas.GetSize(), sumArea);
			}
#endif

			const float fMaxArea((float)(maxArea > 0 ? maxArea : 64));
			const float fMedianArea(6.f * (float)Mesh::AreaArr(maxAreas).GetMedian());
			// [QUALITY] This comparison selects between two different code paths, so a
			// hair's difference in maxAreas flips it and changes the output DISCRETELY.
			// That is the shape of the observed face-count bimodality (~4000 faces apart
			// in two tight clusters rather than a spread). Log both sides and the
			// decision so a cross-configuration output difference can be attributed
			// here -- or ruled out -- instead of inferred.
			DEBUG_EXTRA("[SUBDIV] auto-decimate test: fMedianArea=%.6f vs fMaxArea=%.6f -> %s",
				fMedianArea, fMaxArea, (fMedianArea < fMaxArea) ? "DECIMATE" : "keep");
			if (fMedianArea < fMaxArea) {
				maxAreas.Empty();

				// decimate to the auto detected resolution
				scene.mesh.Clean(MAXF(0.1f, fMedianArea / fMaxArea), 0.f, false, nCloseHoles, 0u, 0.f, false);
				scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);

#ifdef MESHOPT_ENSUREEDGESIZE
				// make sure there are no edges too small or too long
				if (nEnsureEdgeSize > 0 && bNoSimplification) {
					scene.mesh.EnsureEdgeSize();
					scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);
				}
#endif

				// re-map vertex and camera faces
				ListVertexFacesPre();
			}
		}
	}
	if (bNoSimplification)
		return;

	if (maxAreas.IsEmpty()) {
		// extract array of faces viewed by each camera
		ListCameraFaces();
#if MESHOPT_VIEW_STREAM
#if MESHOPT_MEM_DIAG
		DEBUG_EXTRA("[MEM] before chunked camera/dimension rescale"); Util::LogMemoryInfo();
#endif
		// chunked camera/dimension rescale -- see the matching call above
		PrepareViewGeometryChunked();
#if MESHOPT_MEM_DIAG
		DEBUG_EXTRA("[MEM] after chunked camera/dimension rescale"); Util::LogMemoryInfo();
		{
			// What this pass must guarantee is that every view has DIMENSIONS and a
			// rescaled camera -- not pixels. The planes are deliberately released
			// chunk-by-chunk, so "image empty" is the expected state here and is
			// reported separately rather than counted as a fault; a nonzero
			// `no dimensions` is the only thing that indicates a real problem.
			size_t nNoDims = 0, nReady = 0, nPixelsHeld = 0;
			uint64_t totalCamFaces = 0;
			FOREACH(idxImage, images) {
				if (!images[idxImage].IsValid()) continue;
				const View& view = views[idxImage];
				if (view.width == 0 || view.height == 0) ++nNoDims;
				else ++nReady;
				if (!view.image.empty()) ++nPixelsHeld;
				totalCamFaces += g_cameraData[idxImage].faces.size();
			}
			DEBUG_EXTRA("[MEM] pre-ListFaceAreas: %zu ready (dims+camera), %zu NO DIMENSIONS (of %zu); "
				"%zu still holding pixels (expected 0 -- released per chunk), totalCamFaces=%llu",
				nReady, nNoDims, (size_t)images.GetSize(), nPixelsHeld, totalCamFaces);
		}
#endif
		// camera intrinsics only get rescaled to this resolution inside
		// EnsureAllViewsResident's ThInitImage; redo the camera-space vertex
		// projection now that they're correct (candidate lists are reused,
		// cheap, since nothing moved between the two calls)
		ListCameraFaces();
		// Tier-1 planes were loaded only to rescale the cameras; ListFaceAreas
		// below reads faceMap + g_cameraData and never a pixel -- drop them before
		// rasterizing (see the matching call in the branch above).
#endif

		// estimate the faces' area that have big projection areas in both images of a pair
		// (streamMaps under VIEW_STREAM -- see the matching call in the branch above)
#if MESHOPT_VIEW_STREAM
		ListFaceAreas(maxAreas, true);
#else
		ListFaceAreas(maxAreas);
#endif
#if MESHOPT_MEM_DIAG
		{
			size_t nonZero = 0; uint64_t sumArea = 0;
			FOREACH(fi, maxAreas) { if (maxAreas[fi] > 0) { ++nonZero; sumArea += maxAreas[fi]; } }
			DEBUG_EXTRA("[MEM] maxAreas (pre-subdivide): %zu/%zu faces nonzero, sum=%llu", nonZero, maxAreas.GetSize(), sumArea);
		}
#endif
	}

	// subdivide mesh faces if its projection area is bigger than the given number of pixels
	const size_t numVertsOld(vertices.GetSize());
	const size_t numFacesOld(faces.GetSize());
	// [QUALITY] The count of faces over the threshold is what DIRECTLY sets the output
	// face count, so if two configurations (different memory, different batching)
	// disagree on the final mesh, this line says whether they disagreed about the
	// INPUT to subdivision or only about how it was carried out. Also report how many
	// sit within 1 unit of the threshold: a large boundary population means the face
	// count is inherently sensitive to last-bit differences and a small delta between
	// runs is expected rather than a defect.
	if (!maxAreas.IsEmpty()) {
		size_t nOver = 0, nBoundary = 0;
		FOREACH(fi, maxAreas) {
			if (maxAreas[fi] > maxArea) ++nOver;
			if (maxAreas[fi] == maxArea || maxAreas[fi] + 1 == maxArea) ++nBoundary;
		}
		DEBUG_EXTRA("[SUBDIV] %zu/%zu faces over maxArea=%u (%zu within 1 of the threshold)",
			nOver, maxAreas.GetSize(), maxArea, nBoundary);
	}
	scene.mesh.Subdivide(maxAreas, maxArea);

#ifdef MESHOPT_ENSUREEDGESIZE
	// make sure there are no edges too small or too long
#if MESHOPT_ENSUREEDGESIZE==1
	if ((nEnsureEdgeSize == 1 && !bNoDecimation) || nEnsureEdgeSize > 1)
#endif
	{
		scene.mesh.EnsureEdgeSize();
		scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);
	}
#endif

	// re-map vertex and camera faces
	ListVertexFacesPre();

	DEBUG_EXTRA("Mesh subdivided: %u/%u -> %u/%u vertices/faces", numVertsOld, numFacesOld, vertices.GetSize(), faces.GetSize());

#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 3)
		scene.mesh.Save(MAKE_PATH("MeshSubdivided.ply"));
#endif
}

// score mesh using photo-consistency
// and compute vertices gradient using analytical method
double MeshRefine::ScoreMesh(float* gradients, bool rebuildOctree)
{
#if MESHOPT_SUPPORT_STAGE_DIAG
	// zero BEFORE ListCameraFaces: that call dispatches the ProjectMesh round
	// whose per-face raster stages we count on iteration 0
	if (iteration == 0) {
		faceCandViews.Resize(faces.GetSize());
		faceCandViews.Memset(0);
		faceBBoxViews.Resize(faces.GetSize());
		faceBBoxViews.Memset(0);
		faceInsideViews.Resize(faces.GetSize());
		faceInsideViews.Memset(0);
		faceWonViews.Resize(faces.GetSize());
		faceWonViews.Memset(0);
		faceLoseMargin.Resize(faces.GetSize());
		faceLoseMargin.Memset(0);
	}
#endif
	// extract array of faces viewed by each camera
#if MESHOPT_PROFILE
	MeshProf::Timer _tLCF;
#endif
	ListCameraFaces(rebuildOctree);
#if MESHOPT_PROFILE
	MeshProf::Add(MeshProf::P_ListCameraFaces, _tLCF.ms());
#endif

	int64_t numImages = (int64_t)images.size();
#pragma omp parallel for
	for (int64_t ID = 0; ID < numImages; ++ID)
	{
		if (!images[ID].IsValid()) continue;

		View& view = views[ID];

		view.tilesX = (view.width + TILEX - 1) / TILEX;
		view.tilesY = (view.height + TILEY - 1) / TILEY;
		size_t numTiles = view.tilesX * view.tilesY;

		// Reset accumulators (to be filled during ScoreMesh)
#if MESHOPT_NEEDS_TILE_ENERGY
		view.tileEnergyAccum.assign(numTiles, 0.f);

		// Only reset tileActive when relevant
		if (iteration == 0 || rebuildOctree) {
			if (view.tileActive.size() != numTiles) {
				view.tileActive.assign(numTiles, 1);
			}
			else {
				std::fill(view.tileActive.begin(), view.tileActive.end(), 1);
			}
		}
#endif
	}

	// JPB WIP BUG Nneded twice?
	//scene.mesh.ComputeNormalFaces();

	// for each pair of images, compute a photo-consistency score
	// between the reference image and the pixels of the second image
	// projected in the reference image through the mesh surface
#ifdef MESHOPT_CERES
	scorePhoto = 0;
#endif
	photoGrad.assign(vertices.GetSize(), Grad(0, 0, 0));
	photoGradNorm.Resize(vertices.GetSize());
	photoGradNorm.Memset(0);
#if MESHOPT_SUPPORT_STAGE_DIAG
	if (iteration == 0) {
		rasterSupport.Resize(vertices.GetSize());
		rasterSupport.Memset(0);
		maskSupport.Resize(vertices.GetSize());
		maskSupport.Memset(0);
	}
#endif
	if (!vertexDepth.IsEmpty()) {
		ASSERT(vertexDepth.GetSize() == vertices.GetSize());
		vertexDepth.MemsetValue(FLT_MAX);
	}

	// Build (once) the per-reference-view neighbor lists so the pair loop can be
	// grouped by reference view: this lets us regenerate the streamed faceMap for
	// view A once, process every pair in which A is the reference, then release
	// it. pairs is constant for the whole refinement.
	if (refViewNeighbors.empty() && !pairs.IsEmpty()) {
		refViewNeighbors.assign(images.GetSize(), {});
		FOREACHPTR(pPair, pairs) {
			refViewNeighbors[pPair->i].push_back(pPair->j);
			refViewNeighbors[pPair->j].push_back(pPair->i);
		}
	}
#if MESHOPT_VIEW_STREAM
	if (viewBatches.empty())
		BuildViewBatches();
#endif

	// JPB WIP BUG Fix this since it is doing pairs of pairs:

#if 1
	// No benefit to trying to reduce i->j and j->i pairs separately.
	// No benefit to grouping i->j i->j2 i->j3... 
#if MESHOPT_PROFILE
	const auto _tPair0 = std::chrono::steady_clock::now();
	tRasterizeMs = 0.0;
#endif
	++pairPassCounter;
	int numPairThreads = 1;
#ifdef _OPENMP
	numPairThreads = omp_get_max_threads();
#endif
	std::vector<GradArr*> threadGradBuffers(numPairThreads, nullptr);
	std::vector<std::vector<uint32_t>*> threadNormBuffers(numPairThreads, nullptr);
#if MESHOPT_NEEDS_TILE_ENERGY
	std::vector<std::vector<std::vector<float>>*> threadTileBuffers(numPairThreads, nullptr);
#endif
#pragma omp parallel
	{
#ifdef _OPENMP
#pragma omp single
		{ numPairThreads = omp_get_num_threads(); }
#endif
#if MESHOPT_PROFILE
		double _tRasterLocalMs = 0.0;
#endif
		static thread_local std::vector<std::vector<float>> tileEnergyLocal;
		// Allocate per-thread tileEnergyLocal
		// One-time resize per thread
		if (tileEnergyLocal.size() != views.size()) {
			tileEnergyLocal.resize(views.size());
		}

		// Ensure each view has a correctly sized, ZEROED tile buffer.
		// This is deliberately eager for ALL views rather than lazy-per-reference-
		// view: lazy allocation makes a thread's buffer state depend on the dynamic
		// schedule, and that showed up as run-to-run non-determinism far larger than
		// float-ordering noise (scale-3 iter-1 gradient 104.798 vs a 104.992+-0.002
		// cluster, from an IDENTICAL binary). The ~1 GB it would save at 32 threads
		// is not worth making the output schedule-dependent.
#if MESHOPT_NEEDS_TILE_ENERGY
		for (size_t v = 0; v < views.size(); v++) {
			size_t tcount = views[v].tilesX * views[v].tilesY;
			if (tileEnergyLocal[v].size() != tcount)
				tileEnergyLocal[v].assign(tcount, 0.f);
			else
				std::fill(tileEnergyLocal[v].begin(), tileEnergyLocal[v].end(), 0.f);
		}
#endif

		static thread_local GradArr localGrad;
		static thread_local std::vector<uint32_t> localNorm;

		// This will be maintained as zero on thread exit.
		const int numVerts = (int)photoGrad.size();
		if ((int)localGrad.size() < numVerts) {
			localGrad.resize(numVerts, Grad(0, 0, 0));
			localNorm.resize(numVerts, 0);
		}
		int pairThread = 0;
#ifdef _OPENMP
		pairThread = omp_get_thread_num();
#endif
		threadGradBuffers[pairThread] = &localGrad;
		threadNormBuffers[pairThread] = &localNorm;
#if MESHOPT_NEEDS_TILE_ENERGY
		threadTileBuffers[pairThread] = &tileEnergyLocal;
#endif

		// Reference-view A mean/variance, computed ONCE per reference view (with
		// a full mask) and reused for every neighbor b, instead of recomputing it
		// inside each ThProcessPair. Thread-local and safe: the omp-for below is
		// over reference views, so each thread owns a whole reference view at once.
		static thread_local TImage<uint16_t> refMeanA;
		static thread_local TImage<Real> refVarA;

		// Each thread owns a reference view while its enabled directed pairs run.
#if MESHOPT_VIEW_STREAM
		size_t _batchIdx = 0;
		// Serpentine (boustrophedon) batch order: walk 1,2,3 then 3,2,1 then 1,2,3.
		// The batch resident when a pass ends is the first one the NEXT pass needs,
		// so the wrap-around transition -- which otherwise evicts the last batch and
		// reloads the first, in full, on every pass -- costs nothing. Measured at
		// scale 3: 89.8 s of batch loading in a 217.6 s pair loop over 18 loads
		// (6 passes x 3 batches); this removes about one load per pass. Same
		// batches, same pairs, same math -- residency order only, though it does
		// re-order the per-thread gradient summation exactly as schedule(dynamic)
		// already does.
		const bool _reverseBatches = (pairPassCounter & 1ull) != 0ull;
		for (size_t _bi = 0; _bi < viewBatches.size(); ++_bi) {
			auto& viewBatch = viewBatches[_reverseBatches ? viewBatches.size() - 1 - _bi : _bi];
#pragma omp single
		{
#if MESHOPT_MEM_DIAG
			// Commit measured on BOTH sides of the residency swap. This is what
			// localizes the gap between the modelled ceiling and the real peak: if
			// commit climbs at each transition and never returns, the freed buffers
			// are being retained rather than released (a heap/allocator problem, and
			// PlaneAlloc's VirtualAlloc backing should fix it). If commit is flat
			// across transitions and simply higher than the model, the excess is LIVE
			// memory the budget never accounted for -- per-thread pair-loop scratch
			// and friends -- and no allocator change will help.
			const uint64_t _cBefore = ProcessCommitBytes();
#endif
			EnsureViewsResident(viewBatch.allViews);
#if MESHOPT_VIEW_STREAM_VERIFY
			VerifyBatchResidency(viewBatch.allViews, viewBatch.refViews);
#endif
#if MESHOPT_MEM_DIAG
			{
				constexpr double GBd = 1024.0 * 1024.0 * 1024.0;
				const uint64_t _cAfter = ProcessCommitBytes();
				DEBUG_EXTRA("[MEM] batch %zu/%zu (%zu views): commit %.2f -> %.2f GB, resident planes %.2f GB, decode %.2f GB",
					_batchIdx + 1, viewBatches.size(), viewBatch.allViews.size(),
					_cBefore / GBd, _cAfter / GBd,
					StreamBytesResident() / GBd, DecodeCacheBytesResident() / GBd);
			}
#endif
		}
#pragma omp for schedule(dynamic)
		for (int ai = 0; ai < (int)viewBatch.refViews.size(); ++ai) {
			const int a = (int)viewBatch.refViews[ai];
			const std::vector<uint32_t>& nbrs = refViewNeighbors[a];
#else
#pragma omp for schedule(dynamic)
		for (int a = 0; a < (int)refViewNeighbors.size(); ++a) {
			const std::vector<uint32_t>& nbrs = refViewNeighbors[a];
#endif
			if (nbrs.empty())
				continue;
			const auto pairEnabled = [this, a](uint32_t b) {
				const bool forward = (uint32_t)a < b;
				switch (nAlternatePair) {
				case 1: return forward == ((iteration & 1u) == 0);
				case 2: return forward;
				case 3: return !forward;
				default: return true;
				}
			};
			bool hasEnabledPair = false;
			for (uint32_t b : nbrs) {
				if (pairEnabled(b)) {
					hasEnabledPair = true;
					break;
				}
			}
			if (!hasEnabledPair)
				continue;
#if !MESHOPT_FACEMAP_RESIDENT
			// regenerate the reference-view faceMap on demand
#if MESHOPT_PROFILE
			const auto _tR0 = std::chrono::steady_clock::now();
#endif
			RasterizeFaceMap(views[a], g_cameraData[a]);
#if MESHOPT_PROFILE
			{
				const double _rms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tR0).count();
				_tRasterLocalMs += _rms;
				MeshProf::AddTL(MeshProf::P_Rasterize, _rms);
			}
#endif
#endif // !MESHOPT_FACEMAP_RESIDENT
#if MESHOPT_ISVALID_STREAM && !MESHOPT_DIRECT_VALIDITY
			// regenerate A's pruned isValid from the just-rasterized faceMap +
			// resident depthMap (isValid is not kept resident under this gate).
			ComputeIsValid(views[a], images[a].camera, g_cameraData[a]);
#endif
			// compute reference-view A mean/variance ONCE (full mask); A's stats
			// depend only on viewA.image (the per-pair mask merely gates which
			// pixels ZNCC later reads), so reusing them for every b is bit-identical.
			const ImageStore& imageA = views[a].image;
#if MESHOPT_PROFILE
			MeshProf::Timer _tRV;
#endif
#if MESHOPT_IMAGE_U16
			ComputeLocalVariance2Unmasked(imageA, refMeanA, refVarA);
#else
			ComputeLocalVariance(imageA, std::vector<uint8_t>((size_t)imageA.cols * imageA.rows, 0xFF), refMeanA, refVarA);
#endif
#if MESHOPT_PROFILE
			MeshProf::AddTL(MeshProf::P_RefVar, _tRV.ms());
#endif
			for (uint32_t b : nbrs) {
				if (pairEnabled(b))
					ThProcessPair((uint32_t)a, b, localGrad, localNorm, tileEnergyLocal, refMeanA, refVarA);
			}
#if !MESHOPT_FACEMAP_RESIDENT
			// free the streamed faceMap; keeps only ~nThreads maps alive at once
			views[a].faceMap.release();
#endif
#if MESHOPT_ISVALID_STREAM && !MESHOPT_DIRECT_VALIDITY
			// free the per-view isValid we regenerated for view A above
			{ std::vector<uint8_t, AlignedAllocator<uint8_t, 16>>().swap(views[a].isValid); }
#endif
		}
#if MESHOPT_VIEW_STREAM
			++_batchIdx;
		} // end for viewBatch : viewBatches
#endif
#if MESHOPT_PROFILE
#pragma omp critical
		{ tRasterizeMs += _tRasterLocalMs; MeshProf::FlushTL(); }
#endif

		// Barrier implicit here at end of 'omp for'
#pragma omp for schedule(static)
		for (int v = 0; v < numVerts; ++v) {
			Grad grad(0, 0, 0);
			uint32_t norm = 0;
			for (int t = 0; t < numPairThreads; ++t) {
				GradArr& threadGrad = *threadGradBuffers[t];
				std::vector<uint32_t>& threadNorm = *threadNormBuffers[t];
				grad += threadGrad[v];
				norm += threadNorm[v];
				threadGrad[v] = Grad(0, 0, 0);
				threadNorm[v] = 0;
			}
			photoGrad[v] = grad;
			photoGradNorm[v] = (float)norm;
		}

		// -----------------------------
		// REDUCE TILE ENERGIES
		// -----------------------------
#if MESHOPT_NEEDS_TILE_ENERGY
#pragma omp for schedule(static)
		for (int vi = 0; vi < (int)views.size(); ++vi) {
			View& vw = views[vi];
			const size_t T = (size_t)vw.tilesX * (size_t)vw.tilesY;
			float* __restrict acc = vw.tileEnergyAccum.data();
			std::fill(acc, acc + T, 0.f);
			// Accumulate thread-major rather than tile-major: each thread's buffer is
			// walked once, sequentially, and a thread that never referenced this view
			// has no buffer to walk at all.
			for (int thread = 0; thread < numPairThreads; ++thread) {
				std::vector<float>& threadEnergy = (*threadTileBuffers[thread])[vi];
				for (size_t t = 0; t < T; ++t) {
					acc[t] += threadEnergy[t];
					threadEnergy[t] = 0.f;
				}
			}
		}
#endif
	} // end omp parallel

#if MESHOPT_PROFILE
	{
		const double tPairMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tPair0).count();
		MeshProf::Add(MeshProf::P_PairLoopWall, tPairMs);
		const double pmPct = tPairMs > 0.0 ? 100.0 * tProjectMeshMs / tPairMs : 0.0;
		const int nThr = (int)threads.GetSize();
		const double tRasterWallMs = nThr > 0 ? tRasterizeMs / nThr : tRasterizeMs; // sum of per-thread time -> approx wall
		const double rasterPct = tPairMs > 0.0 ? 100.0 * tRasterWallMs / tPairMs : 0.0;
		DEBUG_EXTRA("[PROFILE] iter %u: ProjectMesh=%.1f ms (%.1f%% of pair loop)  PairLoop=%.1f ms  RasterizeFaceMap~=%.1f ms wall (%.1f%% of pair loop = potential un-streaming saving)  (rebuildOctree=%d)",
			iteration, tProjectMeshMs, pmPct, tPairMs, tRasterWallMs, rasterPct, (int)rebuildOctree);
	}
#endif

	// --------------------------------------------------
	// DIAGNOSTICS: accumulate photometric energy
	// (must be done BEFORE clearing tileEnergyAccum)
	// photoEnergyLast is only read by the iteration diagnostics and the
	// converged-exit plateau test; skip the O(tiles) reduction otherwise.
	// --------------------------------------------------
#if MESHOPT_ITER_DIAGNOSTICS || MESHOPT_CONVERGED_EXIT
	double photoEnergyIter = 0.0;
	for (size_t vid = 0; vid < views.size(); ++vid) {
		const View& view = views[vid];
		for (float e : view.tileEnergyAccum)
			photoEnergyIter += e;
	}

	photoEnergyLast = photoEnergyIter;
#endif

	// -----------------------------------------
	// Update tileActive for the NEXT iteration
	// -----------------------------------------
#if MESHOPT_NEEDS_TILE_ENERGY
// Start conservative, get aggressive as we converge
#if MESHOPT_REFINE_QUALITY
	float tileThreshold = (iteration < 3) ? 1e-4f : 2e-4f;
#else
	float tileThreshold = (iteration < 3) ? 5e-4f : 1e-3f;
#endif

	for (size_t vid = 0; vid < views.size(); ++vid) {
		View& view = views[vid];
		size_t tcount = view.tilesX * view.tilesY;

		for (size_t t = 0; t < tcount; ++t) {
			float E = view.tileEnergyAccum[t];

			view.tileActive[t] = (E > tileThreshold ? 1 : 0);

			// Clear AFTER energy was captured
			view.tileEnergyAccum[t] = 0.f;
		}
	}
#endif
#else

	ASSERT(events.IsEmpty());
	FOREACHPTR(pPair, pairs) {
		ASSERT(pPair->i < pPair->j);
		switch (nAlternatePair) {
		case 1:
			events.AddEvent(iteration % 2 ? new EVTProcessPair(pPair->j, pPair->i) : new EVTProcessPair(pPair->i, pPair->j));
			break;
		case 2:
			events.AddEvent(new EVTProcessPair(pPair->i, pPair->j));
			break;
		case 3:
			events.AddEvent(new EVTProcessPair(pPair->j, pPair->i));
			break;
		default:
			for (int ip = 0; ip < 2; ++ip)
				events.AddEvent(ip ? new EVTProcessPair(pPair->j, pPair->i) : new EVTProcessPair(pPair->i, pPair->j));
		}
	}
	WaitThreadWorkers(nAlternatePair ? pairs.GetSize() : pairs.GetSize() * 2);
#endif

	// loop through all vertices and compute the smoothing score
#ifdef MESHOPT_CERES
	scoreSmooth = 0;
#endif
	const VIndex idxStep((vertices.GetSize() + (VIndex)threads.GetSize() - 1) / (VIndex)threads.GetSize());
#if MESHOPT_PROFILE
	MeshProf::Timer _tS1;
#endif
	smoothGrad1.resize(vertices.GetSize());
	{
		ASSERT(events.IsEmpty());
		VIndex idx(0);
		while (idx < vertices.GetSize()) {
			const VIndex idxNext(MINF(idx + idxStep, vertices.GetSize()));
			events.AddEvent(new EVTSmoothVertices1(idx, idxNext));
			idx = idxNext;
		}
		WaitThreadWorkers(threads.GetSize());
	}
#if MESHOPT_PROFILE
	MeshProf::Add(MeshProf::P_Smooth1, _tS1.ms());
	MeshProf::Timer _tS2;
#endif
	// loop through all vertices and compute the smoothing gradient
	smoothGrad2.resize(vertices.GetSize());
	{
		ASSERT(events.IsEmpty());
		VIndex idx(0);
		while (idx < vertices.GetSize()) {
			const VIndex idxNext(MINF(idx + idxStep, vertices.GetSize()));
			events.AddEvent(new EVTSmoothVertices2(idx, idxNext));
			idx = idxNext;
		}
		WaitThreadWorkers(threads.GetSize());
	}
#if MESHOPT_PROFILE
	MeshProf::Add(MeshProf::P_Smooth2, _tS2.ms());
	MeshProf::Timer _tCombine;
#endif

	// set the final gradient as the combination of photometric and smoothness gradients
	const int numVertsCombine = (int)vertices.GetSize();
	if (ratioRigidityElasticity >= 1.f) {
#pragma omp parallel for schedule(static)
		for (int v = 0; v < numVertsCombine; ++v)
			((Point3f*)gradients)[v] = photoGradNorm[v] >= (float)MESHOPT_MIN_PAIR_SUPPORT ?
			Cast<float>(photoGrad[v] / photoGradNorm[v] + smoothGrad2[v] * weightRegularity) :
			Cast<float>(smoothGrad2[v] * weightRegularity);
	} else {
		// compute smoothing gradient as a combination of level 1 and 2 of the Laplacian operator;
		// (see page 105 of "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004)
		const Real rigidity((Real(1) - ratioRigidityElasticity) * weightRegularity);
		const Real elasticity(ratioRigidityElasticity * weightRegularity);
#pragma omp parallel for schedule(static)
		for (int v = 0; v < numVertsCombine; ++v)
			((Point3f*)gradients)[v] = photoGradNorm[v] >= (float)MESHOPT_MIN_PAIR_SUPPORT ?
			Cast<float>(photoGrad[v] / photoGradNorm[v] + smoothGrad2[v] * elasticity - smoothGrad1[v] * rigidity) :
			Cast<float>(smoothGrad2[v] * elasticity - smoothGrad1[v] * rigidity);
	}
#if MESHOPT_PROFILE
	MeshProf::Add(MeshProf::P_Combine, _tCombine.ms());
#endif
#ifdef MESHOPT_CERES
	return (nAlternatePair ? 0.2f : 0.1f) * scorePhoto + 0.01f * scoreSmooth;
#else
	return 0.0;
#endif
}

// given a vertex position and a projection camera, compute the projected position and its derivative
// returns the depth
template <typename TP, typename TX, typename T, typename TJ>
T MeshRefine::ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian)
{
	const TX& x1(X[0]);
	const TX& x2(X[1]);
	const TX& x3(X[2]);

	const TP& p1_1(P[0]);
	const TP& p1_2(P[1]);
	const TP& p1_3(P[2]);
	const TP& p1_4(P[3]);
	const TP& p2_1(P[4]);
	const TP& p2_2(P[5]);
	const TP& p2_3(P[6]);
	const TP& p2_4(P[7]);
	const TP& p3_1(P[8]);
	const TP& p3_2(P[9]);
	const TP& p3_3(P[10]);
	const TP& p3_4(P[11]);

	const TP t5(p3_4 + p3_1 * x1 + p3_2 * x2 + p3_3 * x3);
	const TP t6(1.0 / t5);
	const TP t10(p1_4 + p1_1 * x1 + p1_2 * x2 + p1_3 * x3);
	const TP t11(t10 * t6);
	const TP t15(p2_4 + p2_1 * x1 + p2_2 * x2 + p2_3 * x3);
	const TP t16(t15 * t6);
	x[0] = T(t11);
	x[1] = T(t16);
	if (jacobian) {
		jacobian[0] = TJ((p1_1 - p3_1 * t11) * t6);
		jacobian[1] = TJ((p1_2 - p3_2 * t11) * t6);
		jacobian[2] = TJ((p1_3 - p3_3 * t11) * t6);
		jacobian[3] = TJ((p2_1 - p3_1 * t16) * t6);
		jacobian[4] = TJ((p2_2 - p3_2 * t16) * t6);
		jacobian[5] = TJ((p2_3 - p3_3 * t16) * t6);
	}
	return T(t5);
}

// check if any of the depths surrounding the given coordinate is similar to the given value
bool MeshRefine::IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z)
{
	const ImageRef tl(FLOOR2INT(pt));
	for (int x = 0; x < 2; ++x) {
		for (int y = 0; y < 2; ++y) {
			const ImageRef ir(tl.x + x, tl.y + y);
			if (!depthMap.isInsideWithBorder<int, 3>(ir))
				continue;
			const Depth& depth = depthMap(ir);
#ifndef MESHOPT_DEPTHCONSTBIAS
			if (depth <= 0 || ABS(depth - z) > z * 0.01f /*!IsDepthSimilar(depth, z, 0.01f)*/)
#else
			if (depth <= 0 || depth + MESHOPT_DEPTHCONSTBIAS < z)
#endif
				continue;
			return true;
		}
	}
	return false;
}

#undef VALIDATE_RASTERIZER
#undef VALIDATE_COUNT

#if MESHOPT_RASTER_SPAN_DIAG
// see MESHOPT_RASTER_SPAN_DIAG at the top of this file; counters are cumulative
// across every view, iteration and scale until reported
static volatile LONG64 g_spanDiagFaces = 0;   // faces reaching the raster loop
static volatile LONG64 g_spanDiagBig = 0;     // ... of those, span >= threshold
static volatile LONG   g_spanDiagMaxSpan = 0; // float bits: max unclipped span
static volatile LONG   g_spanDiagMaxDrift = 0;// float bits: worst-case drift, px

static inline void SpanDiagAtomicMaxF(volatile LONG* addr, float val)
{
	LONG oldInt = *addr;
	for (;;) {
		const float oldVal = *reinterpret_cast<const float*>(&oldInt);
		if (!(val > oldVal))
			return;
		const LONG newInt = *reinterpret_cast<const LONG*>(&val);
		const LONG prev = _InterlockedCompareExchange(addr, newInt, oldInt);
		if (prev == oldInt)
			return;
		oldInt = prev;
	}
}

// boxMin*/boxMax*: UNCLIPPED float bbox. min*i/max*Inc: clipped INCLUSIVE ints.
static inline void SpanDiagAccum(
	float boxMinX, float boxMinY, float boxMaxX, float boxMaxY,
	int minXi, int minYi, int maxXInc, int maxYInc)
{
	// batched so the common path costs a thread-local increment, not a contended
	// atomic; the per-thread tail (< 4096) is never flushed, so the denominator
	// undercounts slightly. It is only used for the percentage - the big-span
	// count and the max values below are exact.
	static thread_local LONG64 facesLocal = 0;
	if (++facesLocal >= 4096) {
		_InterlockedExchangeAdd64(&g_spanDiagFaces, facesLocal);
		facesLocal = 0;
	}

	const float spanX = boxMaxX - boxMinX;
	const float spanY = boxMaxY - boxMinY;
	const float span = MAXF(spanX, spanY);
	if (span < MESHOPT_RASTER_SPAN_DIAG_THRESHOLD)
		return;

	_InterlockedIncrement64(&g_spanDiagBig);
	SpanDiagAtomicMaxF(&g_spanDiagMaxSpan, span);

	// Worst case, i.e. every rounding biased the same way:
	//   |dw| ~ steps * ulp(1),  d(bary)/d(pixel) ~ 1/span  ->  steps*span*ulp
	// Real drift is usually far lower (errors partially cancel, ~sqrt(steps)),
	// so a SMALL number here is conclusive; a large one only means "go measure".
	constexpr float kUlp = 5.9604645e-8f; // 2^-24
	const int stepsX = maxXInc - minXi + 1;
	const int stepsY = maxYInc - minYi + 1;
	float drift = 0.f;
	if (stepsX > 0 && spanX > 0.f)
		drift += (float)stepsX * spanX * kUlp;
	if (stepsY > 0 && spanY > 0.f)
		drift += (float)stepsY * spanY * kUlp;
	SpanDiagAtomicMaxF(&g_spanDiagMaxDrift, drift);
}

static void SpanDiagReport(const char* tag)
{
	const LONG64 n = g_spanDiagFaces;
	const LONG64 nBig = g_spanDiagBig;
	const LONG maxSpanBits = g_spanDiagMaxSpan;
	const LONG maxDriftBits = g_spanDiagMaxDrift;
	VERBOSE("[SPAN] %s: %lld/%lld rasterized faces span >= %.0f px (%.4f%%), "
		"max span %.1f px, worst-case bary drift %.5f px",
		tag, nBig, n, (double)(MESHOPT_RASTER_SPAN_DIAG_THRESHOLD),
		n ? 100.0 * (double)nBig / (double)n : 0.0,
		*reinterpret_cast<const float*>(&maxSpanBits),
		*reinterpret_cast<const float*>(&maxDriftBits));
}
#endif

template <typename TYPE>
float EdgeFunction2(const TPoint2<TYPE>& x0,
	const TPoint2<TYPE>& x1,
	const TPoint2<TYPE>& x2) {
	// explicitly compute in float precision
	float dx1 = static_cast<float>(x1.x - x0.x);
	float dy1 = static_cast<float>(x1.y - x0.y);
	float dx2 = static_cast<float>(x2.x - x0.x);
	float dy2 = static_cast<float>(x2.y - x0.y);

	// perform 2D cross product in float
	return dx1 * dy2 - dy1 * dx2;
}

#undef INVARIANT1
#undef INVARIANT2

// project mesh to the given camera plane
void MeshRefine::ProjectMesh(
	View& view,
	const Camera& camera,
	const CameraRenderData& rd)
{
	DepthMap& depthMap = view.depthMap;
#if MESHOPT_FACEMAP_RESIDENT
	// keep the rasterized faceMap resident on the view: the pair loop and
	// ListFaceAreas consume it directly instead of re-rasterizing the reference
	// view every iteration via RasterizeFaceMap (bit-identical output).
	FaceMap& faceMap = view.faceMap;
#else
	// faceMap is NOT persisted per view (streamed to save ~4 B/pixel across all
	// views). Rasterize+prune into a thread-local scratch, then discard; the
	// reference-view faceMap is regenerated on demand via RasterizeFaceMap.
	static thread_local FaceMap faceMapScratch;
	FaceMap& faceMap = faceMapScratch;
#endif
	// dimensions come from view.width/height, NOT view.image.size(): under tier-1
	// eviction the pixel buffer may legitimately be gone while the view still has
	// to be rasterized (the subdivision face-area pass needs faceMap, never pixels).
	// ThInitImage sets both from the same resized image, so these are identical.
	const cv::Size size(view.width, view.height);
	const size_t numPixels = size.width * size.height;

	static thread_local std::vector<uint16_t> depthGen;
	static thread_local uint16_t depthCurGen = 1;
		// Resize if needed
	if (depthGen.size() != numPixels) {
		depthGen.assign(numPixels, 0);
		depthCurGen = 1;
	}
		// Bump generation
	depthCurGen++;
	if (depthCurGen == 0) {
		std::fill(depthGen.begin(), depthGen.end(), 0);
		depthCurGen = 1;
	}

	// init view data
	// Maps must be completely filled or have a generator.
	depthMap.create(size);
	faceMap.create(size);
#if MESHOPT_FACEMAP_RESIDENT
	// consumers gate validity on NO_ID (the raster only writes covered pixels);
	// same fill RasterizeFaceMap performed on the streamed path
	faceMap.fill(NO_ID);
#endif

#ifdef VALIDATE_RASTERIZER
	depthMap.memset(0);
	faceMap.fill(NO_ID);
	baryMap.memset(0);
#endif

#if !MESHOPT_DIRECT_VALIDITY
	view.isValid.assign(numPixels, 0);
#endif

	//depthMap.memset(0);
	//faceMap.memset(NO_ID); JPB WIP BUG Wrong format should be fill()
	//baryMap.memset(0);

	struct Triangle {
		Point2f pti[3];
	} t;

	const int width = size.width;
	const int height = size.height;

#if MESHOPT_RASTER_FULL_FRAME
	const float minX = -4.f * (float)width;
	const float minY = -4.f * (float)height;
	const float maxX = 5.f * (float)width;
	const float maxY = 5.f * (float)height;
#else
	const float minX = 10.f;
	const float minY = 10.f;
	const float maxX = (float)(width - 10);
	const float maxY = (float)(height - 10);
#endif

	const float widthMinus1 = (float)(width - 1);
	const float heightMinus1 = (float)(height - 1);

	for (size_t fi = 0, cnt = rd.faces.size(); fi < cnt; ++fi) {
		const Face& face = rd.faces[fi];
#if MESHOPT_SUPPORT_STAGE_DIAG
		const bool diagCount = (iteration == 0 && faceCandViews.GetSize() == faces.GetSize());
		if (diagCount)
			AtomicAddFloat(&faceCandViews[rd.globalFace[fi]], 1.f);
#endif
		// ==== Camera-space vertices (pre-transformed) ====
		const CamVert& c0 = rd.verts[face[0]];
		const CamVert& c1 = rd.verts[face[1]];
		const CamVert& c2 = rd.verts[face[2]];
		if (c0.invZ == 0.f || c1.invZ == 0.f || c2.invZ == 0.f)
			continue;
		{
			// ==== Perspective divide ====
			float u0 = c0.x * c0.invZ;
			float v0i = c0.y * c0.invZ;

			float u1 = c1.x * c1.invZ;
			float v1i = c1.y * c1.invZ;

			float u2 = c2.x * c2.invZ;
			float v2i = c2.y * c2.invZ;

			// ==== Scalar bounds check (simd not needed anymore) ====
			if (u0 < minX || u0 > maxX ||
				u1 < minX || u1 > maxX ||
				u2 < minX || u2 > maxX ||
				v0i < minY || v0i > maxY ||
				v1i < minY || v1i > maxY ||
				v2i < minY || v2i > maxY)
				continue;

			// ==== Store results (same as your SSE path) ====
			t.pti[0] = { u0, v0i };
			t.pti[1] = { u1, v1i };
			t.pti[2] = { u2, v2i };
		}
		// draw triangle
		const auto& v1 = t.pti[0];
		const auto& v2 = t.pti[1];
		const auto& v3 = t.pti[2];

		// ignore back oriented triangles (negative area)
		// flip winding to match OpenMVS screen-space convention
		const float area = EdgeFunction2(v1, v2, v3);
#if MESHOPT_RASTER_NO_BACKFACE_CULL
		// GPU parity: back faces still occlude (depth write) but get NO_ID in the
		// faceMap (PTX stores -1 + zero bary when the winding predicate fails)
		const bool frontFace = area < 0.f;
		if (area == 0.f) {
			continue;
		}
#else
		if (area >= 0.f) {
			continue;
		}
#endif
		// compute bounding-box fully containing the triangle
		float boxMinX = v1.x;
		float boxMinY = v1.y;
		float boxMaxX = v1.x;
		float boxMaxY = v1.y;

		if (v2.x < boxMinX) boxMinX = v2.x;
		if (v3.x < boxMinX) boxMinX = v3.x;
		if (v2.y < boxMinY) boxMinY = v2.y;
		if (v3.y < boxMinY) boxMinY = v3.y;

		if (v2.x > boxMaxX) boxMaxX = v2.x;
		if (v3.x > boxMaxX) boxMaxX = v3.x;
		if (v2.y > boxMaxY) boxMaxY = v2.y;
		if (v3.y > boxMaxY) boxMaxY = v3.y;

		// ---- Quick reject: fully outside screen ----
		// This is the minimal correct check
		if (boxMaxX < 0.0f || boxMinX > widthMinus1 ||
			boxMaxY < 0.0f || boxMinY > heightMinus1)
			continue;

		// ---- Convert to integer bounding-box & clamp ----
		int minXi = _cvt_ftoi_fast(boxMinX);
		int minYi = _cvt_ftoi_fast(boxMinY);
		int maxXi = _cvt_ftoi_fast(boxMaxX + 1);   // faster than ceil
		int maxYi = _cvt_ftoi_fast(boxMaxY + 1);

		if (minXi < 0) minXi = 0;
		if (minYi < 0) minYi = 0;
		if (maxXi > width)  maxXi = width;
		if (maxYi > height) maxYi = height;

		ImageRef boxMinI(minXi, minYi);
		ImageRef boxMaxI(maxXi - 1, maxYi - 1);   // convert from half-open to inclusive

#if MESHOPT_SUPPORT_STAGE_DIAG
		if (diagCount && boxMinI.x <= boxMaxI.x && boxMinI.y <= boxMaxI.y)
			AtomicAddFloat(&faceBBoxViews[rd.globalFace[fi]], 1.f);
		int diagInside = 0, diagWon = 0;
		float diagLoseMax = 0.f;
#endif

#ifdef INVARIANT2
		constexpr int border = 0;
		if (boxMinI.x < border)
			boxMinI.x = border;
		if (boxMinI.y < border)
			boxMinI.y = border;
		if (boxMaxI.x >= (size.width - border))
			boxMaxI.x = (size.width - (border + 1));
		if (boxMaxI.y >= (size.height - border))
			boxMaxI.y = (size.height - (border + 1));
#endif

#if MESHOPT_RASTER_SPAN_DIAG
		SpanDiagAccum(boxMinX, boxMinY, boxMaxX, boxMaxY,
			boxMinI.x, boxMinI.y, boxMaxI.x, boxMaxI.y);
#endif

		const float invArea = 1.f / area;

		// edge deltas
		const float w0_dx = (v2.y - v3.y) * invArea;
		const float w0_dy = (v3.x - v2.x) * invArea;
		const float w1_dx = (v3.y - v1.y) * invArea;
		const float w1_dy = (v1.x - v3.x) * invArea;
		const float w2_dx = (v1.y - v2.y) * invArea;
		const float w2_dy = (v2.x - v1.x) * invArea;

		// Original work doesn't use pixel centers - premultiply by invArea
		const float px0 = (float)boxMinI.x;
		const float py0 = (float)boxMinI.y;

		// initial barycentrics for first pixel center
		float w0_row = EdgeFunction2(v2, v3, { px0, py0 }) * invArea;
		float w1_row = EdgeFunction2(v3, v1, { px0, py0 }) * invArea;
		float w2_row = EdgeFunction2(v1, v2, { px0, py0 }) * invArea;

		// vertex depths
		const float z0 = c0.z;
		const float z1 = c1.z;
		const float z2 = c2.z;

		// reciprocal depths
		const float iz0 = c0.invZ;
		const float iz1 = c1.invZ;
		const float iz2 = c2.invZ;

		Depth* __restrict depthPtr = depthMap.ptr<float>(0);
		uint16_t* __restrict depthGenPtr = depthGen.data();
		cuint32_t* __restrict facePtr = faceMap.ptr<cuint32_t>(0);
#if !MESHOPT_DIRECT_VALIDITY
		uint8_t* __restrict validPtr = view.isValid.data();
#endif

		for (size_t y = boxMinI.y; y <= boxMaxI.y; ++y) {
			size_t base = size_t(y) * width;
			uint16_t* __restrict depthGenRow = depthGenPtr + base;
			Depth* __restrict depthRow = depthPtr + base;
			cuint32_t* __restrict faceRow = facePtr + base;
#if !MESHOPT_DIRECT_VALIDITY
			uint8_t* __restrict validRow = validPtr + base;
#endif

			float w0 = w0_row;
			float w1 = w1_row;
			float w2 = w2_row;

			// -------------------------------------------------------------------
			// Phase 1: advance until entering triangle (cheap rejects only)
			// -------------------------------------------------------------------
			int x = minXi;
			while (x <= boxMaxI.x) {
				if (w0 >= 0.f && w1 >= 0.f && w0 + w1 <= 1.f) {
					break; // found first inside pixel
				}

				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;

				++x;
			}

			if (x > boxMaxI.x)
				goto next_row; // entire row outside, skip

#if MESHOPT_SUPPORT_STAGE_DIAG
			++diagInside;
#endif

			// -------------------------------------------------------------------
			// Phase 2: inside span - no inside tests in main loop
			// -------------------------------------------------------------------
			for (; x <= boxMaxI.x; ++x) {
				// perspective correct barycentrics
				float denom = w0 * iz0 + w1 * iz1 + w2 * iz2;
				float invDen = 1.f / denom;

				float bx = (w0 * iz0) * invDen;
				float by = (w1 * iz1) * invDen;
				float bz = 1.f - bx - by;

				float z = bx * z0 + by * z1 + bz * z2;

				float old = (depthGenRow[x] == depthCurGen)
					? depthRow[x]
					: std::numeric_limits<float>::infinity();

				if (old > z) {
					depthGenRow[x] = depthCurGen;
					depthRow[x] = z;
#if MESHOPT_RASTER_NO_BACKFACE_CULL
					faceRow[x] = frontFace ? (cuint32_t)fi : (cuint32_t)NO_ID;
#else
					faceRow[x] = (cuint32_t)fi;
#endif
#if MESHOPT_SUPPORT_STAGE_DIAG
					++diagWon;
#endif
#if !MESHOPT_DIRECT_VALIDITY
#if MESHOPT_RASTER_NO_BACKFACE_CULL
					validRow[x] = frontFace ? 1 : 0;
#else
					validRow[x] = 1;
#endif
#endif
				}
#if MESHOPT_SUPPORT_STAGE_DIAG
				else if (diagCount) {
					const float rel = (z - old) / old;
					if (rel > diagLoseMax)
						diagLoseMax = rel;
				}
#endif

				// advance
				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;

				// exit span
				if (w0 < 0.f || w1 < 0.f || (w0 + w1) > 1.f)
					break;
			}

		next_row:
			w0_row += w0_dy;
			w1_row += w1_dy;
			w2_row += w2_dy;
		}
#if MESHOPT_SUPPORT_STAGE_DIAG
		if (diagCount) {
			if (diagInside > 0)
				AtomicAddFloat(&faceInsideViews[rd.globalFace[fi]], 1.f);
			if (diagWon > 0)
				AtomicAddFloat(&faceWonViews[rd.globalFace[fi]], 1.f);
			if (diagLoseMax > 0.f)
				AtomicMaxFloat(&faceLoseMargin[rd.globalFace[fi]], diagLoseMax);
		}
#endif
	}

#if MESHOPT_DEPTH_CLEAR_UNCOVERED
	// coverage authority is the generation stamp, valid in both faceMap configs
	{
		const uint16_t* __restrict genPtr = depthGen.data();
		Depth* __restrict clearPtr = depthMap.ptr<float>(0);
#if MESHOPT_RASTER_NO_BACKFACE_CULL
		// GPU CrossCheckProjection parity: pixels won by a back face (depth
		// written, faceMap NO_ID) are fully invalidated, depth included;
		// covered-this-gen pixels always hold fresh face ids in both configs
		const cuint32_t* __restrict crossPtr = faceMap.ptr<cuint32_t>(0);
		for (size_t px = 0; px < numPixels; ++px)
			if (genPtr[px] != depthCurGen || crossPtr[px] == (cuint32_t)NO_ID)
				clearPtr[px] = 0;
#else
		for (size_t px = 0; px < numPixels; ++px)
			if (genPtr[px] != depthCurGen)
				clearPtr[px] = 0;
#endif
	}
#endif

#ifdef VALIDATE_RASTERIZER

	DepthMap depthMap2;
	FaceMap faceMap2;
	BaryMap baryMap2;

	depthMap2.create(size);
	faceMap2.create(size);
	baryMap2.create(size);

	depthMap2.memset(0);
	faceMap2.memset((uint8_t)NO_ID);
	baryMap2.memset(0);

	// project all triangles on this image and keep the closest ones
	RasterMesh rasterer(vertices, camera, depthMap2, faceMap2, baryMap2);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	for (auto idxFace : cameraFaces) {
		const Face& facet = faces[idxFace];
		rasterer.idxFace = idxFace;
		rasterer.Project(facet, triangleRasterizer);
	}

	const auto size2 = depthMap2.size();

	size_t diffDepthCount = 0;
	size_t diffFaceCount = 0;
	size_t diffBaryCount = 0;
	double maxDepthDiff = 0.0;
	double maxBaryDiff = 0.0;

	for (int y = 0; y < size2.height; ++y) {
		for (int x = 0; x < size2.width; ++x) {
			const Depth dr = depthMap2(y, x);
			const Depth do_ = depthMap(y, x);

			// Depth difference
			if (dr != 0 || do_ != 0) {
				const double diff = fabs((double)dr - (double)do_);
				if (diff > 1e-4 && !(isnan(dr) && isnan(do_))) {
					++diffDepthCount;
					if (diff > maxDepthDiff) maxDepthDiff = diff;
				}
			}

			// Face difference
			const int fr = faceMap2(y, x);
			const int fo = faceMap(y, x);
			if (fr != fo)
				++diffFaceCount;

			// Barycentric difference
			if (fr == fo) {
				const Point3f& br = baryMap2(y, x);
				const Point3f& bo = baryMap(y, x);
				const double db0 = fabs((double)br.x - (double)bo.x);
				const double db1 = fabs((double)br.y - (double)bo.y);
				const double db2 = fabs((double)br.z - (double)bo.z);
				const double bd = std::max(db0, std::max(db1, db2));
				if (bd > 1e-3) {
					++diffBaryCount;
					if (bd > maxBaryDiff) maxBaryDiff = bd;
				}
			}
		}
	}

	static std::mutex coutMutex;
	{
		std::lock_guard<std::mutex> lock(coutMutex);

		const int total = size2.width * size2.height;
		VERBOSE("Validation results:\n");
		VERBOSE("  Depth differences: %zu / %d (%.6f%%), max delta = %.6g\n",
			diffDepthCount, total, 100.0 * diffDepthCount / total, maxDepthDiff);
		VERBOSE("  Face  differences: %zu / %d (%.6f%%)\n",
			diffFaceCount, total, 100.0 * diffFaceCount / total);
		VERBOSE("  Bary  differences: %zu / %d (%.6f%%), max delta = %.6g\n",
			diffBaryCount, total, 100.0 * diffBaryCount / total, maxBaryDiff);
	}
#endif

	// NOTE: the per-view SoA (ray/X/invNd/normal/bary/verticesPerPix) is no
	// longer stored; it is recomputed on demand in ComputePhotometricGradient
	// from depthMap/faceMap/baryMap + camera + CameraRenderData. This removes
	// the dominant resident memory (~76 B/pixel x every image).
	const size_t rows = size.height;
	const size_t cols = size.width;
	const size_t count = rows * cols;

	view.width = cols;
	view.height = rows;

#ifdef VALIDATE_COUNT
	int validCnt = 0;
#endif

#if !MESHOPT_ISVALID_STREAM
	const Point3f cameraC = Cast<float>(camera.C);
	const float r00 = camera.R(0, 0), r01 = camera.R(0, 1), r02 = camera.R(0, 2);
	const float r10 = camera.R(1, 0), r11 = camera.R(1, 1), r12 = camera.R(1, 2);
	const float r20 = camera.R(2, 0), r21 = camera.R(2, 1), r22 = camera.R(2, 2);
	const float fx = camera.K(0, 0);
	const float fy = camera.K(1, 1);
	const float cx = camera.K(0, 2);
	const float cy = camera.K(1, 2);
	const float invFx = 1.0f / fx;
	const float invFy = 1.0f / fy;

#if 1
	//-------------------------------------------------------------------------
		// Tiled Iteration
		// We iterate through valid Tiles, then iterate pixels inside them.
		//-------------------------------------------------------------------------
	for (size_t ty = 0; ty < rows; ty += TILEY) {
		for (size_t tx = 0; tx < cols; tx += TILEX) {

			// Inner Loop: 0..31
			for (size_t ly = 0; ly < TILEY; ++ly) {
				const size_t r = ty + ly;

				// Bounds check: If this tile hangs off the bottom of the image
				if (r >= rows) { break; }

				// Pointers for INPUT arrays (Linear)
				cuint32_t* __restrict faceRow = faceMap.ptr<cuint32_t>(r);
				float* __restrict depthRow = depthMap.ptr<float>(r);
				uint8_t* __restrict isValidRow = &view.isValid[r * cols];

				const float dy = (static_cast<float>(r) - cy) * invFy;  // (v - cy)/fy

				for (size_t lx = 0; lx < TILEX; ++lx) {
					const size_t c = tx + lx;

					// Bounds check: If this tile hangs off the right of the image
					if (c >= cols) { break; }

					if (!isValidRow[c]) {
						faceRow[c] = NO_ID;
						continue;
					}

					// view.depthMap(r,c) guaranteed > 0
					const float depth = depthRow[c];

					// ---------------------------------------------------------------------
					// Unified Pruning Block (High-Impact Early Rejects)
					// ---------------------------------------------------------------------

					// 0. Basic depth check
					if (depth <= 0.0f) {
						isValidRow[c] = 0;
						continue;
					}

					const FIndex f = faceRow[c];  // Needed for normal check anyway

          // JPB WIP BUG Experiment with depth discontinuity culling
					// 1. Depth discontinuity pruning (A-side)
					// Adaptive threshold: smaller for close, larger for far
					const float depthThresh = 0.05f * depth + 0.01f;  // 5% + 1cm baseline
					const float normalThresh = 0.15f;  // ~81' angle change

					bool isEdge = false;

					// Combined horizontal + vertical depth check (branchless)
					if (c > 0 && c < cols - 1) {
						float dzdx = MAXF(fabs(depth - depthRow[c - 1]),
							fabs(depth - depthRow[c + 1]));

						if (r > 0 && r < rows - 1) {
							float dzdy = MAXF(fabs(depth - depthMap.ptr<float>(r - 1)[c]),
								fabs(depth - depthMap.ptr<float>(r + 1)[c]));

							// Single comparison for both axes
							if (MAXF(dzdx, dzdy) > depthThresh) {
								isEdge = true;
							}
						}
						else if (dzdx > depthThresh) {
							isEdge = true;
						}
					}

					// Normal check (only if depth check passed)
					if (!isEdge && c > 0 && c < cols - 1) {
						const FIndex fLeft = faceRow[c - 1];
						const FIndex fRight = faceRow[c + 1];

						if (isValidRow[c - 1] && isValidRow[c + 1]) {
							const FIndex fLeft = faceRow[c - 1];   // Safe: we know it's valid
							const FIndex fRight = faceRow[c + 1];  // Safe: we know it's valid

							// normals looked up on demand from the global faceNormals
							const Grad& N = faceNormals[rd.globalFace[f]];
							const float dotLeft = N.dot(faceNormals[rd.globalFace[fLeft]]);
							const float dotRight = N.dot(faceNormals[rd.globalFace[fRight]]);

							// Branchless: compute minimum dot product
							const float minDot = (dotLeft < dotRight ? dotLeft : dotRight);

							if (minDot < (1.0f - normalThresh)) {
								isEdge = true;
							}
						}
					}
					if (isEdge) {
						isValidRow[c] = 0;
						continue;
					}

					// Make vertical symmetric with horizontal
					if (r > 0 && r < rows - 1) {
						float dzdy_above = fabs(depth - depthMap.ptr<float>(r - 1)[c]);
						float dzdy_below = fabs(depth - depthMap.ptr<float>(r + 1)[c]);
						if (dzdy_above > depthThresh && dzdy_below > depthThresh) {
							isValidRow[c] = 0;
							continue;
						}
					}

					// 2. Texture flatness prune (A-image)
					// Remove low-information pixels before expensive steps.
					// Check 4-connected neighborhood
#if MESHOPT_IMAGE_U16
					const float center = view.image(r, c) * kImgU16Scale;
					const float left = (c > 0) ? view.image(r, c - 1) * kImgU16Scale : center;
					const float right = (c < cols - 1) ? view.image(r, c + 1) * kImgU16Scale : center;
					const float up = (r > 0) ? view.image(r - 1, c) * kImgU16Scale : center;
					const float down = (r < rows - 1) ? view.image(r + 1, c) * kImgU16Scale : center;
#else
					const float center = view.image(r, c);
					const float left = (c > 0) ? view.image(r, c - 1) : center;
					const float right = (c < cols - 1) ? view.image(r, c + 1) : center;
					const float up = (r > 0) ? view.image(r - 1, c) : center;
					const float down = (r < rows - 1) ? view.image(r + 1, c) : center;
#endif

					const float maxGrad = MAXF(
						MAXF(fabs(center - left), fabs(center - right)),
						MAXF(fabs(center - up), fabs(center - down))
					);

					if (maxGrad < 0.005f) {  // More permissive (0.5% instead of 1%)
						isValidRow[c] = 0;
						continue;
					}

					// At this point we know depth is OK, neighbors are OK, texture exists.
					// Proceed to compute rayW, X, dA, Nd.
					// ---------------------------------------------------------------------

					const float dx = (static_cast<float>(c) - cx) * invFx;
					const Point3f rayW(
						r00 * dx + r10 * dy + r20,
						r01 * dx + r11 * dy + r21,
						r02 * dx + r12 * dy + r22
					);

					const float lenSq = rayW.x * rayW.x + rayW.y * rayW.y + rayW.z * rayW.z;
					const float invLen = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lenSq)));
					const Point3f dA = { rayW.x * invLen, rayW.y * invLen, rayW.z * invLen };
					const Grad& N = faceNormals[rd.globalFace[f]];
					const float Nd = N.dot(dA);

          			// JPB WIP BUG Experiment with less strict Nd cull (was 0.1)
					// ---------------------------------------------------------------------
					// Surface Angle & Stability Pruning
					// Nd = N · dA has already been computed here.
					// ---------------------------------------------------------------------

					// 3. Grazing angle reject (relaxed threshold)
					constexpr float minNd = -0.95f;  // ~18' from tangent (safe limit)
					if (Nd > minNd) {  // Includes grazing angles + shallow angles
						isValidRow[c] = 0;
						continue;
					}

					// geometry (dA/X/invNd/vertices/bary) is recomputed on demand
					// in ComputePhotometricGradient; nothing to store here.

#ifdef VALIDATE_COUNT
					++validCnt;
#endif
				} // end lx
			} // end ly
		} // end tx
	} // end ty
#else
	for (size_t r = 0; r < rows; ++r) {
		cuint32_t* __restrict faceRow = faceMap.ptr<cuint32_t>(r);
		float* __restrict depthRow = depthMap.ptr<float>(r);
		uint8_t* __restrict isValidRow = &view.isValid[r * cols];
		Point3f* __restrict baryRow = baryMap.ptr<Point3f>(r);
		const float dy = (static_cast<float>(r) - cy) * invFy;  // (v - cy)/fy

		for (size_t c = 0; c < cols; ++c) {
			if (!isValidRow[c]) {
				faceRow[c] = NO_ID;
				continue;
			}

			const size_t idx = r * cols + c;

			// view.depthMap(r,c) guaranteed > 0
			const float depth = depthRow[c];

			// Unnormalized direction in camera coords (z=1), rotated to world
			// RayPoint = R^T * TransformPointI2C([ (u-cx)/fx, (v-cy)/fy, 1 ])

			//const Point3f rayW = camera.RayPoint(Point2(c, r));
			// Reconstruct 3D point in world: X = C + (rayW * depth)
			// (equivalently: X = R^T * ([x',y',1] * depth) + C)
			const float dx = (static_cast<float>(c) - cx) * invFx;  // (u - cx)/fx
			const Point3f rayW(
				r00 * dx + r10 * dy + r20,
				r01 * dx + r11 * dy + r21,
				r02 * dx + r12 * dy + r22
			);

			const Point3f X = rayW * depth + cameraC;

			// Face index for this pixel
			const FIndex f = faceRow[c];
			// view.faceMap(r,c) guaranteed not NO_ID

			// Normalized ray ONLY for Nd
			const float lenSq = rayW.x * rayW.x + rayW.y * rayW.y + rayW.z * rayW.z;
			const float invLen = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lenSq)));
			const Point3f dA = { rayW.x * invLen, rayW.y * invLen, rayW.z * invLen };
			const Grad& N = faceNormals[f];
			const float Nd = N.dot(dA);
			if (Nd > -0.1f || (Nd == 0.f)) {
				isValidRow[c] = 0;
				continue;
			}

			// Store SoA geometry
			// normalized ray for Nd / later Jacobian use
			view.ray[idx] = dA;

			view.X[idx] = X;// full 3D point (world)
			view.storedNormal[idx] = N;
			//view.Nd[idx] = Nd;

			float invNd = 1.0f / Nd;

			if ((invNd < -10.f) || (invNd >= 0.f)) {
				isValidRow[c] = 0;
				continue;
			}

			view.invNd[idx] = invNd;

			const Face& face = faces[f];
			FIndex faceIndexes[3] = { face[0], face[1], face[2] };

			view.verticesPerPix[idx * 3 + 0] = faceIndexes[0];
			view.verticesPerPix[idx * 3 + 1] = faceIndexes[1];
			view.verticesPerPix[idx * 3 + 2] = faceIndexes[2];
			view.facesNormalPerPix[idx] = N;

			// Barycentrics if needed later
			view.bary[idx] = baryRow[c];

			// JPB WIP BUG Not needed isValidRow[c] = 1;
#ifdef VALIDATE_COUNT
			++validCnt;
#endif
		}
	}
#endif
#else
	// In the non-parity streamed fallback, isValid is regenerated per reference
	// view on demand. The direct-validity path never allocates it.
#if !MESHOPT_DIRECT_VALIDITY
	{ std::vector<uint8_t, AlignedAllocator<uint8_t, 16>>().swap(view.isValid); }
#endif
#endif // !MESHOPT_ISVALID_STREAM

#ifdef VALIDATE_COUNT
	{
		static std::mutex coutMutex;
		VERBOSE("View: %d valid pixels out of %d (%.2f%%)\n",
			validCnt, count, 100.f * validCnt / count);
	}
#endif
}

#if MESHOPT_ISVALID_STREAM && !MESHOPT_DIRECT_VALIDITY
// Regenerate the A-only isValid pruning for a reference view on demand, from its
// resident depthMap + the just-rasterized (streamed) view.faceMap + image +
// faceNormals. This reproduces the exact pruning ProjectMesh runs inline (same
// inputs, same tiled order): coverage is seeded from faceMap != NO_ID, which is
// exactly the =1 seed the ProjectMesh raster pass writes on covered pixels, so
// the resulting isValid is bit-identical. Used when MESHOPT_ISVALID_STREAM keeps
// isValid out of the resident per-view set (regenerated per reference view A).
void MeshRefine::ComputeIsValid(View& view, const Camera& camera, const CameraRenderData& rd)
{
	// dimensions come from view.width/height, NOT view.image.size(): under tier-1
	// eviction the pixel buffer may legitimately be gone while the view still has
	// to be rasterized (the subdivision face-area pass needs faceMap, never pixels).
	// ThInitImage sets both from the same resized image, so these are identical.
	const cv::Size size(view.width, view.height);
	const size_t rows = size.height;
	const size_t cols = size.width;
	FaceMap& faceMap = view.faceMap;
	DepthMap& depthMap = view.depthMap;

	// Seed coverage from the streamed faceMap (covered <=> not NO_ID), matching
	// the =1 the ProjectMesh raster pass writes on covered pixels.
	view.isValid.assign((size_t)rows * cols, 0);
	for (size_t r = 0; r < rows; ++r) {
		cuint32_t* __restrict faceRow = faceMap.ptr<cuint32_t>(r);
		const float* __restrict depthRow = depthMap.ptr<float>(r);
		uint8_t* __restrict isValidRow = &view.isValid[r * cols];
		for (size_t c = 0; c < cols; ++c)
			isValidRow[c] = (faceRow[c] != NO_ID && depthRow[c] > 0.f) ? (uint8_t)1 : (uint8_t)0;
	}

#if !MESHOPT_CUDA_PARITY_MASK
	const Point3f cameraC = Cast<float>(camera.C);
	const float r00 = camera.R(0, 0), r01 = camera.R(0, 1), r02 = camera.R(0, 2);
	const float r10 = camera.R(1, 0), r11 = camera.R(1, 1), r12 = camera.R(1, 2);
	const float r20 = camera.R(2, 0), r21 = camera.R(2, 1), r22 = camera.R(2, 2);
	const float fx = camera.K(0, 0);
	const float fy = camera.K(1, 1);
	const float cx = camera.K(0, 2);
	const float cy = camera.K(1, 2);
	const float invFx = 1.0f / fx;
	const float invFy = 1.0f / fy;

	for (size_t ty = 0; ty < rows; ty += TILEY) {
		for (size_t tx = 0; tx < cols; tx += TILEX) {

			for (size_t ly = 0; ly < TILEY; ++ly) {
				const size_t r = ty + ly;

				if (r >= rows) { break; }

				cuint32_t* __restrict faceRow = faceMap.ptr<cuint32_t>(r);
				float* __restrict depthRow = depthMap.ptr<float>(r);
				uint8_t* __restrict isValidRow = &view.isValid[r * cols];

				const float dy = (static_cast<float>(r) - cy) * invFy;  // (v - cy)/fy

				for (size_t lx = 0; lx < TILEX; ++lx) {
					const size_t c = tx + lx;

					if (c >= cols) { break; }

					if (!isValidRow[c]) {
						faceRow[c] = NO_ID;
						continue;
					}

					const float depth = depthRow[c];

					if (depth <= 0.0f) {
						isValidRow[c] = 0;
						continue;
					}

					const FIndex f = faceRow[c];  // Needed for normal check anyway

					const float depthThresh = 0.05f * depth + 0.01f;  // 5% + 1cm baseline
					const float normalThresh = 0.15f;  // ~81' angle change

					bool isEdge = false;

					if (c > 0 && c < cols - 1) {
						float dzdx = MAXF(fabs(depth - depthRow[c - 1]),
							fabs(depth - depthRow[c + 1]));

						if (r > 0 && r < rows - 1) {
							float dzdy = MAXF(fabs(depth - depthMap.ptr<float>(r - 1)[c]),
								fabs(depth - depthMap.ptr<float>(r + 1)[c]));

							if (MAXF(dzdx, dzdy) > depthThresh) {
								isEdge = true;
							}
						}
						else if (dzdx > depthThresh) {
							isEdge = true;
						}
					}

					if (!isEdge && c > 0 && c < cols - 1) {
						const FIndex fLeft = faceRow[c - 1];
						const FIndex fRight = faceRow[c + 1];

						if (isValidRow[c - 1] && isValidRow[c + 1]) {
							const FIndex fLeft = faceRow[c - 1];   // Safe: we know it's valid
							const FIndex fRight = faceRow[c + 1];  // Safe: we know it's valid

							const Grad& N = faceNormals[rd.globalFace[f]];
							const float dotLeft = N.dot(faceNormals[rd.globalFace[fLeft]]);
							const float dotRight = N.dot(faceNormals[rd.globalFace[fRight]]);

							const float minDot = (dotLeft < dotRight ? dotLeft : dotRight);

							if (minDot < (1.0f - normalThresh)) {
								isEdge = true;
							}
						}
					}
					if (isEdge) {
						isValidRow[c] = 0;
						continue;
					}

					if (r > 0 && r < rows - 1) {
						float dzdy_above = fabs(depth - depthMap.ptr<float>(r - 1)[c]);
						float dzdy_below = fabs(depth - depthMap.ptr<float>(r + 1)[c]);
						if (dzdy_above > depthThresh && dzdy_below > depthThresh) {
							isValidRow[c] = 0;
							continue;
						}
					}

#if MESHOPT_IMAGE_U16
					const float center = view.image(r, c) * kImgU16Scale;
					const float left = (c > 0) ? view.image(r, c - 1) * kImgU16Scale : center;
					const float right = (c < cols - 1) ? view.image(r, c + 1) * kImgU16Scale : center;
					const float up = (r > 0) ? view.image(r - 1, c) * kImgU16Scale : center;
					const float down = (r < rows - 1) ? view.image(r + 1, c) * kImgU16Scale : center;
#else
					const float center = view.image(r, c);
					const float left = (c > 0) ? view.image(r, c - 1) : center;
					const float right = (c < cols - 1) ? view.image(r, c + 1) : center;
					const float up = (r > 0) ? view.image(r - 1, c) : center;
					const float down = (r < rows - 1) ? view.image(r + 1, c) : center;
#endif

					const float maxGrad = MAXF(
						MAXF(fabs(center - left), fabs(center - right)),
						MAXF(fabs(center - up), fabs(center - down))
					);

					if (maxGrad < 0.005f) {  // More permissive (0.5% instead of 1%)
						isValidRow[c] = 0;
						continue;
					}

					const float dx = (static_cast<float>(c) - cx) * invFx;
					const Point3f rayW(
						r00 * dx + r10 * dy + r20,
						r01 * dx + r11 * dy + r21,
						r02 * dx + r12 * dy + r22
					);

					const float lenSq = rayW.x * rayW.x + rayW.y * rayW.y + rayW.z * rayW.z;
					const float invLen = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lenSq)));
					const Point3f dA = { rayW.x * invLen, rayW.y * invLen, rayW.z * invLen };
					const Grad& N = faceNormals[rd.globalFace[f]];
					const float Nd = N.dot(dA);

					constexpr float minNd = -0.95f;  // ~18' from tangent (safe limit)
					if (Nd > minNd) {  // Includes grazing angles + shallow angles
						isValidRow[c] = 0;
						continue;
					}
				} // end lx
			} // end ly
		} // end tx
	} // end ty
#endif
}
#endif // MESHOPT_ISVALID_STREAM && !MESHOPT_DIRECT_VALIDITY

// Rasterize ONLY the (local) face-index map for a view, reproducing exactly the
// faceMap that ProjectMesh's raster pass writes. Used to stream the reference-
// view faceMap on demand instead of keeping it resident for all views. The
// per-view isValid/depthMap (already resident) remain the pruning authority;
// faceMap is only read at valid pixels, so a plain re-rasterization is lossless.
// Uncovered pixels are set to NO_ID (needed by ListFaceAreas). The resident
// depthMap is read-only and identifies the same first minimum-depth face.
void MeshRefine::RasterizeFaceMap(View& view, const CameraRenderData& rd)
{
	// dimensions come from view.width/height, NOT view.image.size(): under tier-1
	// eviction the pixel buffer may legitimately be gone while the view still has
	// to be rasterized (the subdivision face-area pass needs faceMap, never pixels).
	// ThInitImage sets both from the same resized image, so these are identical.
	const cv::Size size(view.width, view.height);
	const size_t numPixels = (size_t)size.width * size.height;

#if !MESHOPT_RASTER_REUSE_DEPTH
	static thread_local std::vector<float> depthScratch;
	static thread_local std::vector<uint16_t> depthGen;
	static thread_local uint16_t depthCurGen = 1;
	if (depthGen.size() != numPixels) {
		depthGen.assign(numPixels, 0);
		depthScratch.assign(numPixels, 0.f);
		depthCurGen = 1;
	}
	depthCurGen++;
	if (depthCurGen == 0) {
		std::fill(depthGen.begin(), depthGen.end(), 0);
		depthCurGen = 1;
	}
#endif

	FaceMap& faceMap = view.faceMap;
	faceMap.create(size);
	faceMap.fill(NO_ID);

	struct Triangle {
		Point2f pti[3];
	} t;

	const int width = size.width;
	const int height = size.height;

#if MESHOPT_RASTER_FULL_FRAME
	const float minX = -4.f * (float)width;
	const float minY = -4.f * (float)height;
	const float maxX = 5.f * (float)width;
	const float maxY = 5.f * (float)height;
#else
	const float minX = 10.f;
	const float minY = 10.f;
	const float maxX = (float)(width - 10);
	const float maxY = (float)(height - 10);
#endif

	const float widthMinus1 = (float)(width - 1);
	const float heightMinus1 = (float)(height - 1);

	for (size_t fi = 0, cnt = rd.faces.size(); fi < cnt; ++fi) {
		const Face& face = rd.faces[fi];
		const CamVert& c0 = rd.verts[face[0]];
		const CamVert& c1 = rd.verts[face[1]];
		const CamVert& c2 = rd.verts[face[2]];
		if (c0.invZ == 0.f || c1.invZ == 0.f || c2.invZ == 0.f)
			continue;
		{
			float u0 = c0.x * c0.invZ;
			float v0i = c0.y * c0.invZ;
			float u1 = c1.x * c1.invZ;
			float v1i = c1.y * c1.invZ;
			float u2 = c2.x * c2.invZ;
			float v2i = c2.y * c2.invZ;
			if (u0 < minX || u0 > maxX ||
				u1 < minX || u1 > maxX ||
				u2 < minX || u2 > maxX ||
				v0i < minY || v0i > maxY ||
				v1i < minY || v1i > maxY ||
				v2i < minY || v2i > maxY)
				continue;
			t.pti[0] = { u0, v0i };
			t.pti[1] = { u1, v1i };
			t.pti[2] = { u2, v2i };
		}
		const auto& v1 = t.pti[0];
		const auto& v2 = t.pti[1];
		const auto& v3 = t.pti[2];

		const float area = EdgeFunction2(v1, v2, v3);
#if MESHOPT_RASTER_NO_BACKFACE_CULL
		// GPU parity: back faces occlude but get NO_ID (see ProjectMesh)
		const bool frontFace = area < 0.f;
		if (area == 0.f)
			continue;
#else
		if (area >= 0.f)
			continue;
#endif

		float boxMinX = v1.x, boxMinY = v1.y, boxMaxX = v1.x, boxMaxY = v1.y;
		if (v2.x < boxMinX) boxMinX = v2.x;
		if (v3.x < boxMinX) boxMinX = v3.x;
		if (v2.y < boxMinY) boxMinY = v2.y;
		if (v3.y < boxMinY) boxMinY = v3.y;
		if (v2.x > boxMaxX) boxMaxX = v2.x;
		if (v3.x > boxMaxX) boxMaxX = v3.x;
		if (v2.y > boxMaxY) boxMaxY = v2.y;
		if (v3.y > boxMaxY) boxMaxY = v3.y;

		if (boxMaxX < 0.0f || boxMinX > widthMinus1 ||
			boxMaxY < 0.0f || boxMinY > heightMinus1)
			continue;

		int minXi = _cvt_ftoi_fast(boxMinX);
		int minYi = _cvt_ftoi_fast(boxMinY);
		int maxXi = _cvt_ftoi_fast(boxMaxX + 1);
		int maxYi = _cvt_ftoi_fast(boxMaxY + 1);

		if (minXi < 0) minXi = 0;
		if (minYi < 0) minYi = 0;
		if (maxXi > width)  maxXi = width;
		if (maxYi > height) maxYi = height;

		ImageRef boxMinI(minXi, minYi);
		ImageRef boxMaxI(maxXi - 1, maxYi - 1);

#if MESHOPT_RASTER_SPAN_DIAG
		SpanDiagAccum(boxMinX, boxMinY, boxMaxX, boxMaxY,
			boxMinI.x, boxMinI.y, boxMaxI.x, boxMaxI.y);
#endif

		const float invArea = 1.f / area;

		const float w0_dx = (v2.y - v3.y) * invArea;
		const float w0_dy = (v3.x - v2.x) * invArea;
		const float w1_dx = (v3.y - v1.y) * invArea;
		const float w1_dy = (v1.x - v3.x) * invArea;
		const float w2_dx = (v1.y - v2.y) * invArea;
		const float w2_dy = (v2.x - v1.x) * invArea;

		const float px0 = (float)boxMinI.x;
		const float py0 = (float)boxMinI.y;

		float w0_row = EdgeFunction2(v2, v3, { px0, py0 }) * invArea;
		float w1_row = EdgeFunction2(v3, v1, { px0, py0 }) * invArea;
		float w2_row = EdgeFunction2(v1, v2, { px0, py0 }) * invArea;

		const float z0 = c0.z;
		const float z1 = c1.z;
		const float z2 = c2.z;

		const float iz0 = c0.invZ;
		const float iz1 = c1.invZ;
		const float iz2 = c2.invZ;

#if MESHOPT_RASTER_REUSE_DEPTH
		const volatile float* depthPtr = view.depthMap.ptr<float>(0);
#else
		float* __restrict depthPtr = depthScratch.data();
		uint16_t* __restrict depthGenPtr = depthGen.data();
#endif
		cuint32_t* __restrict facePtr = faceMap.ptr<cuint32_t>(0);

		for (size_t y = boxMinI.y; y <= boxMaxI.y; ++y) {
			size_t base = size_t(y) * width;
#if MESHOPT_RASTER_REUSE_DEPTH
			const volatile float* depthRow = depthPtr + base;
#else
			uint16_t* __restrict depthGenRow = depthGenPtr + base;
			float* __restrict depthRow = depthPtr + base;
#endif
			cuint32_t* __restrict faceRow = facePtr + base;

			float w0 = w0_row;
			float w1 = w1_row;
			float w2 = w2_row;

			int x = minXi;
			while (x <= boxMaxI.x) {
				if (w0 >= 0.f && w1 >= 0.f && w0 + w1 <= 1.f) {
					break;
				}
				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;
				++x;
			}

			if (x > boxMaxI.x)
				goto next_row;

			for (; x <= boxMaxI.x; ++x) {
				float denom = w0 * iz0 + w1 * iz1 + w2 * iz2;
				float invDen = 1.f / denom;

				float bx = (w0 * iz0) * invDen;
				float by = (w1 * iz1) * invDen;
				float bz = 1.f - bx - by;

				float z = bx * z0 + by * z1 + bz * z2;

#if MESHOPT_RASTER_REUSE_DEPTH
				if (faceRow[x] == NO_ID && depthRow[x] == z) {
#if MESHOPT_RASTER_NO_BACKFACE_CULL
					if (frontFace)
						faceRow[x] = (cuint32_t)fi;
#else
					faceRow[x] = (cuint32_t)fi;
#endif
				}
#else
				float old = (depthGenRow[x] == depthCurGen)
					? depthRow[x]
					: std::numeric_limits<float>::infinity();

				if (old > z) {
					depthGenRow[x] = depthCurGen;
					depthRow[x] = z;
#if MESHOPT_RASTER_NO_BACKFACE_CULL
					faceRow[x] = frontFace ? (cuint32_t)fi : (cuint32_t)NO_ID;
#else
					faceRow[x] = (cuint32_t)fi;
#endif
				}
#endif

				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;

				if (w0 < 0.f || w1 < 0.f || (w0 + w1) > 1.f)
					break;
			}

		next_row:
			w0_row += w0_dy;
			w1_row += w1_dy;
			w2_row += w2_dy;
		}
	}
}

#if 1
#undef MESHREFINE_WARP_VALIDATE
void MeshRefine::ImageMeshWarp(
	const View& viewA,
	const DepthMap& depthMapA, const Camera& cameraA,
	const DepthMap& depthMapB, const Camera& cameraB,
	const ImageStore& imageB,
	TImage<uint16_t>& imageAB,
	std::vector<uint8_t>& mask)
{
	ASSERT(!imageA.empty());

	const size_t rows = depthMapA.rows;
	const size_t cols = depthMapA.cols;

	// Camera A intrinsics
	const float fxA = (float) cameraA.K(0, 0);
	const float fyA = (float) cameraA.K(1, 1);
	const float cxA = (float) cameraA.K(0, 2);
	const float cyA = (float) cameraA.K(1, 2);
	const float rfxA = 1.f / fxA;
	const float rfyA = 1.f / fyA;

	// Camera A extrinsics (R^t and C)
	const float rA00 = (float) cameraA.R(0, 0), rA01 = (float) cameraA.R(0, 1), rA02 = (float) cameraA.R(0, 2);
	const float rA10 = (float) cameraA.R(1, 0), rA11 = (float) cameraA.R(1, 1), rA12 = (float) cameraA.R(1, 2);
	const float rA20 = (float) cameraA.R(2, 0), rA21 = (float) cameraA.R(2, 1), rA22 = (float) cameraA.R(2, 2);

	const float cAx = (float) cameraA.C.x;
	const float cAy = (float) cameraA.C.y;
	const float cAz = (float) cameraA.C.z;

	// Camera B intrinsics
	const float fxB = cameraB.K(0, 0);
	const float fyB = cameraB.K(1, 1);
	const float cxB = cameraB.K(0, 2);
	const float cyB = cameraB.K(1, 2);

	// Camera B extrinsics
	const float rB00 = (float) cameraB.R(0, 0), rB01 = (float) cameraB.R(0, 1), rB02 = (float) cameraB.R(0, 2);
	const float rB10 = (float) cameraB.R(1, 0), rB11 = (float) cameraB.R(1, 1), rB12 = (float) cameraB.R(1, 2);
	const float rB20 = (float) cameraB.R(2, 0), rB21 = (float) cameraB.R(2, 1), rB22 = (float) cameraB.R(2, 2);

	const float cBx = cameraB.C.x;
	const float cBy = cameraB.C.y;
	const float cBz = cameraB.C.z;

	// Compute M = R_B * R_A
	const float m00 = rB00 * rA00 + rB01 * rA01 + rB02 * rA02;
	const float m01 = rB00 * rA10 + rB01 * rA11 + rB02 * rA12;
	const float m02 = rB00 * rA20 + rB01 * rA21 + rB02 * rA22;

	const float m10 = rB10 * rA00 + rB11 * rA01 + rB12 * rA02;
	const float m11 = rB10 * rA10 + rB11 * rA11 + rB12 * rA12;
	const float m12 = rB10 * rA20 + rB11 * rA21 + rB12 * rA22;

	const float m20 = rB20 * rA00 + rB21 * rA01 + rB22 * rA02;
	const float m21 = rB20 * rA10 + rB21 * rA11 + rB22 * rA12;
	const float m22 = rB20 * rA20 + rB21 * rA21 + rB22 * rA22;

	// Compute world-offset transformed by R_B
	// T = R_B * (C_A - C_B)
	const float tcx = cAx - cBx;
	const float tcy = cAy - cBy;
	const float tcz = cAz - cBz;

	const float t0 = rB00 * tcx + rB01 * tcy + rB02 * tcz;
	const float t1 = rB10 * tcx + rB11 * tcy + rB12 * tcz;
	const float t2 = rB20 * tcx + rB21 * tcy + rB22 * tcz;

	boost::container::small_vector<float, 4096> xnRow(cols);
	float x = (-cxA) * rfxA;
	for (size_t i = 0; i < cols; ++i) {
		xnRow[i] = x;
		x += rfxA;
	}

	if (depthMapB.size() != imageB.size()) {
		ERROR("ImageMeshWarp: depth map B and image B have different sizes");
		return;
  }
	const size_t colsB = depthMapB.cols;
	const size_t rowsB = depthMapB.rows;

	const float* __restrict depthMapBPtr = (float*)depthMapB.data;
#if MESHOPT_IMAGE_U16
	const uint16_t* __restrict imageBPtr = (const uint16_t*)imageB.data;
#else
	const float* __restrict imageBPtr = (float*)imageB.data;
#endif

#if MESHOPT_WARP_AVX2
	static const bool useWarpAVX2 = SupportsAVX2();
#endif

	for (size_t j = 0; j < rows; ++j) {
		const float yn = ((float)j - cyA) * rfyA;

		const float biasX = m01 * yn + m02;
		const float biasY = m11 * yn + m12;
		const float biasZ = m21 * yn + m22;

		const float* __restrict depthRowA = depthMapA.ptr<const float>(j);
#if MESHOPT_DIRECT_VALIDITY
		const cuint32_t* __restrict faceRowA = viewA.faceMap.ptr<cuint32_t>(j);
#else
		const uint8_t* __restrict validRow = &viewA.isValid[j * cols];
#endif
		uint16_t* __restrict outRow = imageAB.ptr<uint16_t>(j);
		uint8_t* __restrict maskRow = &mask[j * cols];
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict imageRowA = viewA.image.ptr<uint16_t>(j);
#else
		const float* __restrict imageRowA = viewA.image.ptr<float>(j);
#endif

		size_t i = 0;
#if MESHOPT_WARP_AVX2
		if (useWarpAVX2)
			i = (size_t)SceneRefineWarpRowAVX2(
				xnRow.data(), depthRowA,
				reinterpret_cast<const uint32_t*>(faceRowA), imageRowA,
				depthMapBPtr, imageBPtr,
				(int)cols, (int)colsB, (int)rowsB,
				biasX, biasY, biasZ, m00, m10, m20, t0, t1, t2,
				fxB, fyB, cxB, cyB, (uint32_t)NO_ID, outRow, maskRow);
#endif
		for (; i < cols; ++i) {
			const float xn = xnRow[i];
#if MESHOPT_IMAGE_U16
			const uint16_t fallback = imageRowA[i];
#else
			const uint16_t fallback = (uint16_t)(MAXF(0.f, MINF(1.f, imageRowA[i])) * 65535.f + 0.5f);
#endif

#if MESHOPT_DIRECT_VALIDITY
			if (faceRowA[i] == NO_ID) {
#else
			if (!validRow[i]) {
#endif
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

			const float z = depthRowA[i];
			if (z <= 0.0f) {
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

			// Compute Z first
			const float termZ = m20 * xn + biasZ;
			const float zcB = z * termZ + t2;

			if (zcB <= 0.0f) {
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

			const float invZ = 1.0f / zcB;

			// Compute X and Y
			const float termX = m00 * xn + biasX;
			const float termY = m10 * xn + biasY;

			const float xcB = z * termX + t0;
			const float ycB = z * termY + t1;

			const float fxInv = fxB * invZ;
			const float fyInv = fyB * invZ;

			const float u = cxB + fxInv * xcB;
			const float v = cyB + fyInv * ycB;
			if (!(u > 10.f && u < (float)(colsB - 10) &&
				  v > 10.f && v < (float)(rowsB - 10))) {
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

			// Compute integer pixel index (fast trunc)
			const int x0 = _cvt_ftoi_fast(u);
			const int y0 = _cvt_ftoi_fast(v);

			// Unified OOB test
			if ((unsigned)x0 >= (unsigned)(colsB - 1) ||
				(unsigned)y0 >= (unsigned)(rowsB - 1)) {
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

			const size_t idx = (size_t)y0 * colsB;

			// Depth rows (depthMapB)
			const float* __restrict dm0 = depthMapBPtr + idx;
			const float* __restrict dm1 = dm0 + colsB;

			// Load 4 depths
			const float d00 = dm0[x0];
			const float d10 = dm0[x0 + 1];
			const float d01 = dm1[x0];
			const float d11 = dm1[x0 + 1];

			// Early reject: all 4 depths <= 0?
			if (d00 <= 0.f && d10 <= 0.f && d01 <= 0.f && d11 <= 0.f) {
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

#ifdef MESHOPT_DEPTHCONSTBIAS
			// reference IsDepthSimilar ACTIVE branch (the macro is DEFINED, here and
			// upstream): one-sided constant z-buffer bias — reject only when every
			// valid neighbor OCCLUDES the point (stored depth in front of zcB by
			// more than the bias); points in FRONT of B's stored surface stay
			// visible (silhouette gaps). The 1% band below is upstream's dead
			// #ifndef branch (it is, however, what the CUDA kernel uses).
			const bool depthSimilar =
				(d00 > 0.f && d00 + MESHOPT_DEPTHCONSTBIAS >= zcB) ||
				(d10 > 0.f && d10 + MESHOPT_DEPTHCONSTBIAS >= zcB) ||
				(d01 > 0.f && d01 + MESHOPT_DEPTHCONSTBIAS >= zcB) ||
				(d11 > 0.f && d11 + MESHOPT_DEPTHCONSTBIAS >= zcB);
#else
			// Precompute depth tolerance
			const float thr = zcB * 0.01f;
			const float zMin = zcB - thr;
			const float zMax = zcB + thr;

			const bool depthSimilar =
				(d00 > 0.f && d00 >= zMin && d00 <= zMax) ||
				(d10 > 0.f && d10 >= zMin && d10 <= zMax) ||
				(d01 > 0.f && d01 >= zMin && d01 <= zMax) ||
				(d11 > 0.f && d11 >= zMin && d11 <= zMax);
#endif
			if (!depthSimilar) {
				outRow[i] = fallback;
				maskRow[i] = 0;
				continue;
			}

			// Fractional bilinear weights
			const float fx = u - (float)x0;
			const float fy = v - (float)y0;

			const float fx1 = 1.f - fx;
			const float fy1 = 1.f - fy;

			// Image rows (imageB)
#if MESHOPT_IMAGE_U16
			const uint16_t* i0 = imageBPtr + idx;
			const uint16_t* i1 = i0 + colsB;

			// Load pixel intensities (uint16 fixed-point -> [0,1])
			const float a = i0[x0]     * kImgU16Scale;
			const float b = i0[x0 + 1] * kImgU16Scale;
			const float c = i1[x0]     * kImgU16Scale;
			const float d = i1[x0 + 1] * kImgU16Scale;
#else
			const float* i0 = imageBPtr + idx;
			const float* i1 = i0 + colsB;

			// Load pixel intensities
			const float a = i0[x0];
			const float b = i0[x0 + 1];
			const float c = i1[x0];
			const float d = i1[x0 + 1];
#endif

			// Fused bilinear interpolation
			const float top = a * fx1 + b * fx;
			const float bot = c * fx1 + d * fx;

			float val = top * fy1 + bot * fy;

			// Clamp final value (like your previous code)
			if (val < 0.f) val = 0.f;
			if (val > 1.f) val = 1.f;

			outRow[i] = (uint16_t)(val * 65535.f + 0.5f);
			maskRow[i] = 1;

#ifdef MESHREFINE_WARP_VALIDATE
			{
				const Point3 XwRef = cameraA.TransformPointI2W(Point3(i, j, z));
				const Point3f XcBRef = cameraB.TransformPointW2C(XwRef);
				const Point2f uvRef = cameraB.TransformPointC2I(XcBRef);

				float valRef = imageB.sample<
					Sampler::Linear<float>,
					Sampler::Linear<float>::Type
				>(Sampler::Linear<float>(), uvRef);

				if (fabsf(uvRef.x - u) > 0.01f ||
					fabsf(uvRef.y - v) > 0.01f ||
					fabsf(valRef - val) > 0.02f) {
					DEBUG("Warp mismatch at pixel %zu,%zu\n", j, i);
				}
			}
#endif
		}
	}
}
#else
// project image from view B to view A through the mesh;
// the projected image is stored in imageA
// (imageAB is assumed to be initialize to the right size)
void MeshRefine::ImageMeshWarp(
	const View& viewA,
	const DepthMap& depthMapA, const Camera& cameraA,
	const DepthMap& depthMapB, const Camera& cameraB,
	const Image32F& imageB, TImage<uint16_t>& imageAB, std::vector<uint8_t>& mask)
{
	ASSERT(!imageA.empty());
	typedef Sampler::Linear<float> Sampler;
	const Sampler sampler;
  const size_t cols = depthMapA.cols;
  const size_t rows = depthMapA.rows;
	for (size_t j = 0; j < rows; ++j) {
    uint8_t* __restrict maskRow = &mask[j * depthMapA.cols];
		const uint8_t* __restrict isValidRow = &viewA.isValid[j * cols];
		uint16_t* __restrict outRow = imageAB.ptr<uint16_t>(j);

		for (size_t i = 0; i < depthMapA.cols; ++i) {
			if (isValidRow[i]) {
				const Depth& depthA = depthMapA(j, i);
				const Point3 X(cameraA.TransformPointI2W(Point3(i, j, depthA)));
				const Point3f ptC(cameraB.TransformPointW2C(X));
				const Point2f pt(cameraB.TransformPointC2I(ptC));
				if (!IsDepthSimilar(depthMapB, pt, ptC.z)) {
					outRow[i] = 0;
					maskRow[i] = 0;
					continue;
				}
				float v = imageB.sample<Sampler, Sampler::Type>(sampler, pt);
				if (v < 0.f) v = 0.f;
				if (v > 1.f) v = 1.f;
				outRow[i] = (uint16_t)(v * 65535.0f + 0.5f);
				maskRow[i] = 1;
			}
			else {
				outRow[i] = 0;
        maskRow[i] = 0;
			}
		}
	}
}
#endif

// compute local variance for each image pixel
void MeshRefine::ComputeLocalVariance(
	const Image32F& image,
	const std::vector<uint8_t>& mask,
	TImage<uint16_t>& imageMean,   // fixed-point 0..65535
	TImage<Real>& imageVar)
{
	ASSERT(image.size() == mask.size());
	imageMean.create(image.size());
	imageVar.create(image.size());

	const int rows = image.rows;
	const int cols = image.cols;

	const int hs = HalfSize;
	const int window = 2 * hs + 1;
	const int n = window * window;
	const double invN = 1.0 / double(n);

	if (rows == 0 || cols == 0)
		return;

	// Safe interior bounds
	const int rowStart = std::max(0, hs);
	const int rowEnd = std::min(rows, rows - hs);
	const int colStart = std::max(0, hs);
	const int colEnd = std::min(cols, cols - hs);

	if (rowStart >= rowEnd || colStart >= colEnd)
		return;  // window doesn't fit inside image

	// Thread-local buffers (double: float rolling sums drift to ~the 1e-4 var
	// floor over thousands of slides; see ComputeLocalVariance2Unmasked)
	static thread_local std::vector<double> colSum;
	static thread_local std::vector<double> colSumSq;

	colSum.assign(cols, 0.0);
	colSumSq.assign(cols, 0.0);

	// Seed vertical window safely:
	// sum rows [0 .. min(rows-1, 2*hs)]
	const int vStart = 0;
	const int vEnd = std::min(rows - 1, 2 * hs);

	// __restrict pointers are scoped to each accumulator loop (below) so the
	// per-row updates vectorize (colSum/colSumSq are Real, same type as the
	// image rows). Tight scoping keeps the horizontal-window and rebuild loops,
	// which touch the same buffers via colSum[]/colSumSq[], out of restrict scope.
	{
		double* __restrict cs = colSum.data();
		double* __restrict css = colSumSq.data();
		for (int rr = vStart; rr <= vEnd; ++rr) {
			const float* __restrict src = image.ptr<float>(rr);
			for (int c = 0; c < cols; ++c) {
				const double v = double(src[c]);
				cs[c] += v;
				css[c] += v * v;
			}
		}
	}

	// Main scanning loop
	for (int r = rowStart; r < rowEnd; ++r) {
		// Compute actual vertical window bounds at this row
		const int vTop = std::max(0, r - hs);
		const int vBot = std::min(rows - 1, r + hs);

		// If this differs from initial window, rebuild vertical window
		if (vTop > vStart || vBot < vEnd) {

			std::fill(colSum.begin(), colSum.end(), 0.0);
			std::fill(colSumSq.begin(), colSumSq.end(), 0.0);

			for (int rr = vTop; rr <= vBot; ++rr) {
				const float* src = image.ptr<float>(rr);
				for (int c = 0; c < cols; ++c) {
					const double v = double(src[c]);
					colSum[c] += v;
					colSumSq[c] += v * v;
				}
			}
		}

		// Build initial horizontal window at (r, colStart)
		double winSum = 0.0;
		double winSumSq = 0.0;

		const int hLeft = std::max(0, colStart - hs);
		const int hRight = std::min(cols - 1, colStart + hs);

		for (int cc = hLeft; cc <= hRight; ++cc) {
			winSum += colSum[cc];
			winSumSq += colSumSq[cc];
		}

		uint16_t* __restrict meanRow = imageMean.ptr<uint16_t>(r);
		Real* __restrict varRow = imageVar.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		// Slide horizontally across interior
		for (int c = colStart; c < colEnd; ++c) {
			if (maskRow[c]) {
				const double meanD = winSum * invN;
				Real var = Real(winSumSq * invN - meanD * meanD);
				Real mean = Real(meanD);

				if (var < Real(0.0001)) var = Real(0.0001);
				if (mean < Real(0)) mean = Real(0);
				if (mean > Real(1)) mean = Real(1);

				meanRow[c] = uint16_t(mean * 65535.0f + 0.5f);
				varRow[c] = var;
			}

			// Slide window one pixel right, safely
			const int addC = c + hs + 1;
			const int remC = c - hs;

			if (addC < cols) winSum += colSum[addC];
			if (remC >= 0)   winSum -= colSum[remC];

			if (addC < cols) winSumSq += colSumSq[addC];
			if (remC >= 0)   winSumSq -= colSumSq[remC];
		}

		// Slide vertical window
		const int newAddR = r + hs + 1;
		const int newRemR = r - hs;

		if (newAddR < rows && newRemR >= 0) {
			const float* __restrict addRow = image.ptr<float>(newAddR);
			const float* __restrict remRow = image.ptr<float>(newRemR);
			double* __restrict cs = colSum.data();
			double* __restrict css = colSumSq.data();

			for (int c = 0; c < cols; ++c) {
				const double a = double(addRow[c]);
				const double d = double(remRow[c]);
				cs[c] += (a - d);
				css[c] += (a * a - d * d);
			}
		}
	}
}

void MeshRefine::ComputeLocalVariance2(
	const TImage<uint16_t>& image,
	const std::vector<uint8_t>& mask,
	TImage<uint16_t>& imageMean,
	TImage<Real>& imageVar)
{
	ASSERT(image.size() == mask.size());
	imageMean.create(image.size());
	imageVar.create(image.size());

	const int rows = image.rows;
	const int cols = image.cols;

	const int hs = HalfSize;
	const int window = 2 * hs + 1;
	const int n = window * window;

	const Real invN = Real(1) / Real(n);
	const Real scale = Real(1.0 / 65535.0);

	if (rows == 0 || cols == 0)
		return;

	// Clamp valid region to ensure safety even on tiny images
	const int rowStart = std::max(0, hs);
	const int rowEnd = std::min(rows, rows - hs);
	const int colStart = std::max(0, hs);
	const int colEnd = std::min(cols, cols - hs);

	if (rowStart >= rowEnd || colStart >= colEnd)
		return; // image too small for variance window

	// Thread-local column accumulators
	static thread_local std::vector<Real> colSum, colSumSq;
	colSum.assign(cols, Real(0));
	colSumSq.assign(cols, Real(0));

	// Seed vertical window safely:
	// sums rows[max(0, r-hs) .. min(rows-1, r+hs)]
	const int vertStart = 0;
	const int vertEnd = std::min(rows - 1, 2 * hs);

	// __restrict scoped per loop (see ComputeLocalVariance).
	{
		Real* __restrict cs = colSum.data();
		Real* __restrict css = colSumSq.data();
		for (int rr = vertStart; rr <= vertEnd; ++rr) {
			const uint16_t* __restrict src = image.ptr<uint16_t>(rr);
			for (int c = 0; c < cols; ++c) {
				Real v = Real(src[c]) * scale;
				cs[c] += v;
				css[c] += v * v;
			}
		}
	}

	// Main scanning loop
	for (int r = rowStart; r < rowEnd; ++r) {

		// Compute vertical window bounds
		const int vTop = std::max(0, r - hs);
		const int vBot = std::min(rows - 1, r + hs);

		// Rebuild vertical window when needed
		// (For normal OpenMVS sizes this path is never taken,
		//  but it is required to be safe on tiny images.)
		if (vTop > vertStart || vBot < vertEnd) {
			// Rebuild the vertical window from scratch safely
			std::fill(colSum.begin(), colSum.end(), Real(0));
			std::fill(colSumSq.begin(), colSumSq.end(), Real(0));

			for (int rr = vTop; rr <= vBot; ++rr) {
				const uint16_t* src = image.ptr<uint16_t>(rr);
				for (int c = 0; c < cols; ++c) {
					Real v = Real(src[c]) * scale;
					colSum[c] += v;
					colSumSq[c] += v * v;
				}
			}
		}

		// Build initial horizontal window at (r, colStart)
		Real winSum = Real(0);
		Real winSumSq = Real(0);

		int hLeft = std::max(0, colStart - hs);
		int hRight = std::min(cols - 1, colStart + hs);

		for (int cc = hLeft; cc <= hRight; ++cc) {
			winSum += colSum[cc];
			winSumSq += colSumSq[cc];
		}

		uint16_t* __restrict meanRow = imageMean.ptr<uint16_t>(r);
		Real* __restrict varRow = imageVar.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		for (int c = colStart; c < colEnd; ++c) {

			if (maskRow[c]) {
				Real mean = winSum * invN;
				Real var = winSumSq * invN - mean * mean;

				if (var < Real(0.0001)) var = Real(0.0001);
				mean = std::max<Real>(0, std::min<Real>(1, mean));

				meanRow[c] = uint16_t(mean * 65535.0f + 0.5f);
				varRow[c] = var;
			}

			// Slide window horizontally one pixel, safely
			int addC = c + hs + 1;
			int remC = c - hs;

			if (addC < cols)
				winSum += colSum[addC];
			if (remC >= 0)
				winSum -= colSum[remC];

			if (addC < cols)
				winSumSq += colSumSq[addC];
			if (remC >= 0)
				winSumSq -= colSumSq[remC];
		}

		// Slide vertical window
		int newAddR = r + hs + 1;
		int newRemR = r - hs;

		if (newAddR < rows && newRemR >= 0) {
			const uint16_t* __restrict addRow = image.ptr<uint16_t>(newAddR);
			const uint16_t* __restrict remRow = image.ptr<uint16_t>(newRemR);
			Real* __restrict cs = colSum.data();
			Real* __restrict css = colSumSq.data();

			for (int c = 0; c < cols; ++c) {
				Real a = Real(addRow[c]) * scale;
				Real d = Real(remRow[c]) * scale;
				cs[c] += (a - d);
				css[c] += (a * a - d * d);
			}
		}
	}
}

void MeshRefine::ComputeLocalVariance2Unmasked(
	const TImage<uint16_t>& image,
	TImage<uint16_t>& imageMean,
	TImage<Real>& imageVar)
{
	imageMean.create(image.size());
	imageVar.create(image.size());

	const int rows = image.rows;
	const int cols = image.cols;
	const int hs = HalfSize;
	const int rowStart = hs;
	const int rowEnd = rows - hs;
	const int colStart = hs;
	const int colEnd = cols - hs;
	if (rowStart >= rowEnd || colStart >= colEnd)
		return;

	const Real invN = Real(1) / Real((2 * hs + 1) * (2 * hs + 1));
	const Real scale = Real(1.0 / 65535.0);
	// double accumulators: float rolling sums drift over thousands of row/col
	// slides, the same order as the 1e-4 variance floor; the GPU recomputes
	// each window exactly and upstream used CV_64F integral images
	static thread_local std::vector<double> colSum, colSumSq;
	colSum.assign(cols, 0.0);
	colSumSq.assign(cols, 0.0);

	for (int rr = 0; rr <= 2 * hs; ++rr) {
		const uint16_t* __restrict src = image.ptr<uint16_t>(rr);
		double* __restrict sums = colSum.data();
		double* __restrict sumsSq = colSumSq.data();
		for (int c = 0; c < cols; ++c) {
			const double value = double(src[c]) * scale;
			sums[c] += value;
			sumsSq[c] += value * value;
		}
	}

	for (int r = rowStart; r < rowEnd; ++r) {
		double winSum = 0;
		double winSumSq = 0;
		for (int cc = 0; cc <= 2 * hs; ++cc) {
			winSum += colSum[cc];
			winSumSq += colSumSq[cc];
		}

		uint16_t* __restrict meanRow = imageMean.ptr<uint16_t>(r);
		Real* __restrict varRow = imageVar.ptr<Real>(r);
		for (int c = colStart; c < colEnd; ++c) {
			const double meanD = winSum * invN;
			Real var = Real(winSumSq * invN - meanD * meanD);
			if (var < Real(0.0001)) var = Real(0.0001);
			Real mean = Real(meanD);
			mean = std::max<Real>(0, std::min<Real>(1, mean));
			meanRow[c] = uint16_t(mean * 65535.0f + 0.5f);
			varRow[c] = var;

			const int addC = c + hs + 1;
			const int remC = c - hs;
			if (addC < cols) {
				winSum += colSum[addC];
				winSumSq += colSumSq[addC];
			}
			if (remC >= 0) {
				winSum -= colSum[remC];
				winSumSq -= colSumSq[remC];
			}
		}

		const int addR = r + hs + 1;
		const int remR = r - hs;
		if (addR < rows && remR >= 0) {
			const uint16_t* __restrict addRow = image.ptr<uint16_t>(addR);
			const uint16_t* __restrict remRow = image.ptr<uint16_t>(remR);
			double* __restrict sums = colSum.data();
			double* __restrict sumsSq = colSumSq.data();
			for (int c = 0; c < cols; ++c) {
				const double add = double(addRow[c]) * scale;
				const double rem = double(remRow[c]) * scale;
				sums[c] += add - rem;
				sumsSq[c] += add * add - rem * rem;
			}
		}
	}
}

// compute local ZNCC and its gradient for each image pixel
float MeshRefine::ComputeLocalZNCC(
	const ImageStore& imageA,
	const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
	const TImage<uint16_t>& imageB, const TImage<uint16_t>& imageMeanB,
	const TImage<Real>& imageVarB,
	const std::vector<uint8_t>& mask,
	TImage<Real>& imageDZNCC)
{
	ASSERT(imageA.size() == imageB.size());
	ASSERT(imageA.size() == mask.size());

	const int rows = imageA.rows;
	const int cols = imageA.cols;

	const int hs = HalfSize;
	const int rowStart = hs;
	const int rowEnd = rows - hs;
	const int colStart = hs;
	const int colEnd = cols - hs;

	const int n = (2 * hs + 1) * (2 * hs + 1);
	const float invN = 1.0f / float(n);
	const float scale16 = 1.0f / 65535.0f;

	// output
	imageDZNCC.create(rows, cols);
	// no memset needed; we only write valid pixels

	// integral buffer (thread-local to avoid alloc)
	static thread_local cv::Mat integralAB;
	integralAB.create(rows + 1, cols + 1, CV_32F);
	if (integralAB.isContinuous()) {
    memset(integralAB.ptr<float>(0), 0, (rows + 1) * (cols + 1) * sizeof(float));
	} else {
		integralAB.setTo(0);
	}

	// ----------------------------------------------------------------
	// Build integral of A * B
	// ----------------------------------------------------------------
	for (int r = 0; r < rows; ++r) {
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict aPtr = imageA.ptr<uint16_t>(r);
#else
		const float* __restrict aPtr = imageA.ptr<float>(r);
#endif
		const uint16_t* __restrict bPtr = imageB.ptr<uint16_t>(r);

		float* __restrict dst = integralAB.ptr<float>(r + 1);
		const float* __restrict prev = integralAB.ptr<float>(r);

		float acc = 0.f;
		for (int c = 0; c < cols; ++c) {
#if MESHOPT_IMAGE_U16
			acc += (float(aPtr[c]) * scale16) * (float(bPtr[c]) * scale16);
#else
			acc += aPtr[c] * (float(bPtr[c]) * scale16);
#endif
			dst[c + 1] = prev[c + 1] + acc;
		}
	}

	// ----------------------------------------------------------------
	// Main loop: compute ZNCC & gradient in one pass
	// ----------------------------------------------------------------
#ifdef MESHOPT_CERES
	float score = 0.0f;
#endif

#if 1 //SSE2?
	for (int r = rowStart; r < rowEnd; ++r) {
		const uint8_t* __restrict maskRow = &mask[r * cols];
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict aRow = imageA.ptr<uint16_t>(r);
#else
		const float* __restrict aRow = imageA.ptr<float>(r);
#endif
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(r);

		const float* __restrict up = integralAB.ptr<float>(r - hs);
		const float* __restrict dn = integralAB.ptr<float>(r + hs + 1);

		Real* __restrict gradRow = imageDZNCC.ptr<Real>(r);

		int c = colStart;
		while (c < colEnd) {
			// skip invalids
			while (c < colEnd && !maskRow[c]) ++c;
			if (c >= colEnd) break;

			// valid run [runStart, runEnd)
			const int runStart = c;
			while (c < colEnd && maskRow[c]) ++c;
			const int runEnd = c;

			// process 4 at a time
			int i = runStart;
			const int vecEnd = runStart + ((runEnd - runStart) & ~3);

			for (; i < vecEnd; i += 4) {
				// All per-lane inputs are CONTIGUOUS in k (cc = i..i+3), so load them
				// directly with SSE2 instead of the old scalar per-k gather into temp
				// arrays. Bit-identical: same values, same op association as the scalar.
				const __m128 scale16v = _mm_set1_ps(scale16);
				const __m128i zero16 = _mm_setzero_si128();

				// sumAB[k] = dn[i+k+hs+1] - dn[i+k-hs] - up[i+k+hs+1] + up[i+k-hs]
				// (each term contiguous in k); preserve the exact scalar association.
				const __m128 dn_x1 = _mm_loadu_ps(dn + i + hs + 1);
				const __m128 dn_x0 = _mm_loadu_ps(dn + i - hs);
				const __m128 up_x1 = _mm_loadu_ps(up + i + hs + 1);
				const __m128 up_x0 = _mm_loadu_ps(up + i - hs);
				const __m128 sumAB = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(dn_x1, dn_x0), up_x1), up_x0);

				// widen 4 contiguous uint16 -> float, then * scale16 (== scalar path)
				const uint16_t* __restrict pMeanA = imageMeanA.ptr<uint16_t>(r) + i;
				const uint16_t* __restrict pMeanB = imageMeanB.ptr<uint16_t>(r) + i;
				__m128 meanA = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)pMeanA), zero16)), scale16v);
				__m128 meanB = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)pMeanB), zero16)), scale16v);
#if MESHOPT_IMAGE_U16
				__m128 aVal = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)(aRow + i)), zero16)), scale16v);
#else
				__m128 aVal = _mm_loadu_ps(aRow + i);
#endif
				__m128 bVal = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)(bRow + i)), zero16)), scale16v);
				__m128 varA = _mm_loadu_ps(imageVarA.ptr<Real>(r) + i);
				__m128 varB = _mm_loadu_ps(imageVarB.ptr<Real>(r) + i);

				__m128 cov = _mm_mul_ps(sumAB, _mm_set1_ps(invN));

				// invS = rsqrt(varA*varB), clamp
				__m128 prod = _mm_mul_ps(varA, varB);
				__m128 clamp = _mm_max_ps(prod, _mm_set1_ps(1e-12f));
				__m128 invS = _mm_rsqrt_ps(clamp);
				// refine once: invS *= (1.5 - 0.5*x*invS^2)
				__m128 invSsq = _mm_mul_ps(invS, invS);
				__m128 refine = _mm_sub_ps(_mm_set1_ps(1.5f), _mm_mul_ps(_mm_set1_ps(0.5f), _mm_mul_ps(clamp, invSsq)));
				invS = _mm_mul_ps(invS, refine);

				// zncc = (cov - meanA*meanB) * invS
				__m128 zncc = _mm_mul_ps(_mm_sub_ps(cov, _mm_mul_ps(meanA, meanB)), invS);

				// ZNCCinvVB = zncc / varB (protect small varB)
				__m128 invVB = _mm_div_ps(_mm_set1_ps(1.0f), _mm_max_ps(varB, _mm_set1_ps(1e-12f)));
				__m128 ZNCCinvVB = _mm_mul_ps(zncc, invVB);

				// dZNCC = aVal*invS - bVal*ZNCCinvVB + meanB*ZNCCinvVB - meanA*invS
				__m128 dZNCC = _mm_sub_ps(
					_mm_add_ps(_mm_mul_ps(aVal, invS), _mm_mul_ps(meanB, ZNCCinvVB)),
					_mm_add_ps(_mm_mul_ps(bVal, ZNCCinvVB), _mm_mul_ps(meanA, invS))
				);

				// reliability = min(varA,varB)/(min+0.0015)
				__m128 minV = _mm_min_ps(varA, varB);
				__m128 reliability = _mm_div_ps(minV, _mm_add_ps(minV, _mm_set1_ps(0.0015f)));

				__m128 grad = _mm_mul_ps(_mm_set1_ps(-1.0f), _mm_mul_ps(reliability, dZNCC));
#ifdef MESHOPT_CERES
				__m128 term = _mm_mul_ps(reliability, _mm_sub_ps(_mm_set1_ps(1.0f), zncc));
#endif

				// Store and accumulate
				float gradOut[4];
				_mm_storeu_ps(gradOut, grad);
#ifdef MESHOPT_CERES
				float termOut[4];
				_mm_storeu_ps(termOut, term);
#endif

				gradRow[i + 0] = Real(gradOut[0]);
				gradRow[i + 1] = Real(gradOut[1]);
				gradRow[i + 2] = Real(gradOut[2]);
				gradRow[i + 3] = Real(gradOut[3]);

#ifdef MESHOPT_CERES
				score += termOut[0] + termOut[1] + termOut[2] + termOut[3];
#endif
			}

			// tail
			for (; i < runEnd; ++i) {
				const int x0 = i - hs;
				const int x1 = i + hs + 1;

				const float sumAB = dn[x1] - dn[x0] - up[x1] + up[x0];
				const Real cov = Real(sumAB * invN);

				const Real meanA = Real(imageMeanA(r, i)) * scale16;
				const Real meanB = Real(imageMeanB(r, i)) * scale16;

				const Real varA = imageVarA(r, i);
				const Real varB = imageVarB(r, i);

				float x = float(varA * varB);
				if (x < 1e-12f) x = 1e-12f;
				float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
				invS = invS * (1.5f - 0.5f * x * invS * invS);

				const Real zncc = (cov - meanA * meanB) * invS;

#if MESHOPT_IMAGE_U16
				const Real aVal = Real(aRow[i]) * scale16;
#else
				const Real aVal = Real(aRow[i]);
#endif
				const Real bVal = Real(bRow[i]) * scale16;

				const Real invVB = (varB > Real(1e-12) ? Real(1) / varB : Real(1e12));
				const Real ZNCCinvVB = zncc * invVB;

				const Real dZNCC =
					aVal * invS
					- bVal * ZNCCinvVB
					+ meanB * ZNCCinvVB
					- meanA * invS;

				const Real minV = (varA < varB ? varA : varB);
				const Real reliability = minV / (minV + Real(0.0015));

				gradRow[i] = -reliability * dZNCC;
#ifdef MESHOPT_CERES
				score += float(reliability * (Real(1) - zncc));
#endif
			}
		} // run
	} // rows

#else
	for (int r = rowStart; r < rowEnd; ++r) {
		const uint8_t* __restrict maskRow = &mask[r * cols];
		const float* __restrict aRow = imageA.ptr<float>(r);
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(r);

		const float* __restrict up = integralAB.ptr<float>(r - hs);
		const float* __restrict dn = integralAB.ptr<float>(r + hs + 1);

		Real* __restrict gradRow = imageDZNCC.ptr<Real>(r);

		for (int c = colStart; c < colEnd; ++c) {
			if (!maskRow[c])
				continue;

			const int x0 = c - hs;
			const int x1 = c + hs + 1;

			// covariance over window
			const float sumAB = dn[x1] - dn[x0] - up[x1] + up[x0];
			const Real cov = Real(sumAB * invN);

			// unpack means
			const Real meanA = Real(imageMeanA(r, c)) * scale16;
			const Real meanB = Real(imageMeanB(r, c)) * scale16;

			// variances
			const Real varA = imageVarA(r, c);
			const Real varB = imageVarB(r, c);

			// inverse sqrt(varA*varB)
			float x = float(varA * varB);
			if (x < 1e-12f) x = 1e-12f;

			float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
			invS = invS * (1.5f - 0.5f * x * invS * invS);

			// ZNCC
			const Real zncc = (cov - meanA * meanB) * invS;

			// ZNCC gradient
			const Real aVal = Real(aRow[c]);
			const Real bVal = Real(bRow[c]) * scale16;

			const Real ZNCCinvVB = zncc / varB;

			const Real dZNCC =
				aVal * invS
				- bVal * ZNCCinvVB
				+ meanB * ZNCCinvVB
				- meanA * invS;

			// reliability (OpenMVS)
			const Real minV = (varA < varB ? varA : varB);
			const Real reliability = minV / (minV + Real(0.0015));

			gradRow[c] = -reliability * dZNCC;
#ifdef MESHOPT_CERES
			score += float(reliability * (Real(1) - zncc));
#endif
		}
	}
#endif

#ifdef MESHOPT_CERES
	return score;
#else
	return 0.0f;
#endif
}

// Compute warped-image mean/variance, A*B covariance and the ZNCC derivative
// in one rolling-window pass. Scratch is O(image width), replacing the full
// imageMeanAB/imageVarAB/integralAB buffers used by the split path.
float MeshRefine::ComputeLocalZNCCFused(
	const ImageStore& imageA,
	const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
	const TImage<uint16_t>& imageB,
	const std::vector<uint8_t>& mask,
	TImage<Real>& imageDZNCC,
	const std::function<void(const TImage<Real>&, size_t, size_t, bool)>& consumeBand)
{
	ASSERT(imageA.size() == imageB.size());
	ASSERT(imageA.size() == mask.size());

	const int rows = imageA.rows;
	const int cols = imageA.cols;
	const int hs = HalfSize;
	const int rowStart = hs;
	const int rowEnd = rows - hs;
	const int colStart = hs;
	const int colEnd = cols - hs;
	const int n = (2 * hs + 1) * (2 * hs + 1);
	const float invN = 1.0f / float(n);
	const float scale16 = 1.0f / 65535.0f;

	const bool banded = (bool)consumeBand;
	imageDZNCC.create(banded ? TILEY : rows, cols);
	if (rowStart >= rowEnd || colStart >= colEnd)
		return 0.0f;
	static const bool useAVX2 = [] {
		const bool enabled = MESHOPT_AVX2 && SupportsAVX2();
		LOG(_T("Mesh refinement ZNCC SIMD: %s"), enabled ? _T("AVX2") : _T("SSE2"));
		return enabled;
	}();

	static thread_local std::vector<float> colB, colB2, colAB;
	colB.assign(cols, 0.f);
	colB2.assign(cols, 0.f);
	colAB.assign(cols, 0.f);

	for (int rr = 0; rr <= 2 * hs; ++rr) {
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict aRow = imageA.ptr<uint16_t>(rr);
#else
		const float* __restrict aRow = imageA.ptr<float>(rr);
#endif
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(rr);
		for (int c = 0; c < cols; ++c) {
			const float b = float(bRow[c]) * scale16;
#if MESHOPT_IMAGE_U16
			const float a = float(aRow[c]) * scale16;
#else
			const float a = aRow[c];
#endif
			colB[c] += b;
			colB2[c] += b * b;
			colAB[c] += a * b;
		}
	}

#ifdef MESHOPT_CERES
	float score = 0.f;
#endif
	const __m128 invNv = _mm_set1_ps(invN);
	const __m128 scale16v = _mm_set1_ps(scale16);
	const __m128 scale65535v = _mm_set1_ps(65535.f);
	const __m128 halfv = _mm_set1_ps(0.5f);
	const __m128 minVarv = _mm_set1_ps(0.0001f);
	const __m128 epsv = _mm_set1_ps(1e-12f);
	const __m128 reliabilityFloorv = _mm_set1_ps(0.0015f);
	const __m128i zero16 = _mm_setzero_si128();

	for (int r = rowStart; r < rowEnd; ++r) {
		const uint8_t* __restrict maskRow = &mask[(size_t)r * cols];
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict aRow = imageA.ptr<uint16_t>(r);
#else
		const float* __restrict aRow = imageA.ptr<float>(r);
#endif
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(r);
		const uint16_t* __restrict meanARow = imageMeanA.ptr<uint16_t>(r);
		const float* __restrict varARow = imageVarA.ptr<float>(r);
		float* __restrict gradRow = imageDZNCC.ptr<float>(banded ? r % TILEY : r);

		int i = colStart;
#if MESHOPT_IMAGE_U16
#ifdef MESHOPT_CERES
		float* avx2Score = useAVX2 ? &score : nullptr;
#else
		float* avx2Score = nullptr;
#endif
		if (useAVX2) {
			i = SceneRefineZNCCRowAVX2(
				maskRow, aRow, bRow, meanARow, varARow,
				colB.data(), colB2.data(), colAB.data(), gradRow,
					colStart, colEnd, invN, avx2Score);
		}
#endif
		const int vecEnd = colStart + ((colEnd - colStart) & ~3);
		for (; i < vecEnd; i += 4) {
			uint32_t mask4;
			memcpy(&mask4, maskRow + i, sizeof(mask4));
			if (mask4 == 0)
				continue;

				const __m128 sumB = _mm_add_ps(
					_mm_add_ps(_mm_loadu_ps(colB.data() + i - 2), _mm_loadu_ps(colB.data() + i - 1)),
					_mm_add_ps(_mm_add_ps(_mm_loadu_ps(colB.data() + i), _mm_loadu_ps(colB.data() + i + 1)), _mm_loadu_ps(colB.data() + i + 2)));
				const __m128 sumB2 = _mm_add_ps(
					_mm_add_ps(_mm_loadu_ps(colB2.data() + i - 2), _mm_loadu_ps(colB2.data() + i - 1)),
					_mm_add_ps(_mm_add_ps(_mm_loadu_ps(colB2.data() + i), _mm_loadu_ps(colB2.data() + i + 1)), _mm_loadu_ps(colB2.data() + i + 2)));
				const __m128 sumAB = _mm_add_ps(
					_mm_add_ps(_mm_loadu_ps(colAB.data() + i - 2), _mm_loadu_ps(colAB.data() + i - 1)),
					_mm_add_ps(_mm_add_ps(_mm_loadu_ps(colAB.data() + i), _mm_loadu_ps(colAB.data() + i + 1)), _mm_loadu_ps(colAB.data() + i + 2)));
				const __m128 meanBRaw = _mm_mul_ps(sumB, invNv);
				const __m128 meanBScaled = _mm_add_ps(_mm_mul_ps(meanBRaw, scale65535v), halfv);
				const __m128i meanBInt = _mm_cvttps_epi32(meanBScaled);
				const __m128 meanB = _mm_mul_ps(_mm_cvtepi32_ps(meanBInt), scale16v);
				const __m128 varB = _mm_max_ps(
					_mm_sub_ps(_mm_mul_ps(sumB2, invNv), _mm_mul_ps(meanBRaw, meanBRaw)),
					minVarv);
				const __m128 meanA = _mm_mul_ps(
					_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)(meanARow + i)), zero16)),
					scale16v);
				const __m128 varA = _mm_loadu_ps(varARow + i);
				const __m128 cov = _mm_mul_ps(sumAB, invNv);

				const __m128 prod = _mm_max_ps(_mm_mul_ps(varA, varB), epsv);
				__m128 invS = _mm_rsqrt_ps(prod);
				invS = _mm_mul_ps(invS, _mm_sub_ps(
					_mm_set1_ps(1.5f),
					_mm_mul_ps(_mm_set1_ps(0.5f), _mm_mul_ps(prod, _mm_mul_ps(invS, invS)))));
				const __m128 zncc = _mm_mul_ps(_mm_sub_ps(cov, _mm_mul_ps(meanA, meanB)), invS);
				const __m128 invVB = _mm_div_ps(_mm_set1_ps(1.f), _mm_max_ps(varB, epsv));
				const __m128 znccInvVB = _mm_mul_ps(zncc, invVB);
#if MESHOPT_IMAGE_U16
				const __m128 aVal = _mm_mul_ps(
					_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)(aRow + i)), zero16)),
					scale16v);
#else
				const __m128 aVal = _mm_loadu_ps(aRow + i);
#endif
				const __m128 bVal = _mm_mul_ps(
					_mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)(bRow + i)), zero16)),
					scale16v);
				const __m128 dZNCC = _mm_sub_ps(
					_mm_add_ps(_mm_mul_ps(aVal, invS), _mm_mul_ps(meanB, znccInvVB)),
					_mm_add_ps(_mm_mul_ps(bVal, znccInvVB), _mm_mul_ps(meanA, invS)));
				const __m128 minV = _mm_min_ps(varA, varB);
				const __m128 reliability = _mm_div_ps(minV, _mm_add_ps(minV, reliabilityFloorv));
				_mm_storeu_ps(gradRow + i, _mm_mul_ps(_mm_set1_ps(-1.f), _mm_mul_ps(reliability, dZNCC)));
#ifdef MESHOPT_CERES
				const __m128i maskBytes = _mm_cvtsi32_si128((int)mask4);
				const __m128 laneMask = _mm_castsi128_ps(_mm_cmpeq_epi32(
					_mm_cvtepu8_epi32(maskBytes), _mm_set1_epi32(1)));
				alignas(16) float terms[4];
				_mm_store_ps(terms, _mm_and_ps(
					laneMask,
					_mm_mul_ps(reliability, _mm_sub_ps(_mm_set1_ps(1.f), zncc))));
				score += terms[0] + terms[1] + terms[2] + terms[3];
#endif
		}

		for (; i < colEnd; ++i) {
			if (!maskRow[i])
				continue;
				float sumB = 0.f, sumB2 = 0.f, sumAB = 0.f;
				for (int offset = -hs; offset <= hs; ++offset) {
					sumB += colB[i + offset];
					sumB2 += colB2[i + offset];
					sumAB += colAB[i + offset];
				}
				const float meanBRaw = sumB * invN;
				const float meanB = float((uint16_t)(meanBRaw * 65535.f + 0.5f)) * scale16;
				const float varB = MAXF(sumB2 * invN - meanBRaw * meanBRaw, 0.0001f);
				const float meanA = float(meanARow[i]) * scale16;
				const float varA = varARow[i];
				float product = MAXF(varA * varB, 1e-12f);
				float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(product)));
				invS *= 1.5f - 0.5f * product * invS * invS;
				const float zncc = (sumAB * invN - meanA * meanB) * invS;
#if MESHOPT_IMAGE_U16
				const float aVal = float(aRow[i]) * scale16;
#else
				const float aVal = aRow[i];
#endif
				const float bVal = float(bRow[i]) * scale16;
				const float znccInvVB = zncc / varB;
				const float dZNCC = aVal * invS - bVal * znccInvVB + meanB * znccInvVB - meanA * invS;
				const float minV = MINF(varA, varB);
				const float reliability = minV / (minV + 0.0015f);
				gradRow[i] = -reliability * dZNCC;
#ifdef MESHOPT_CERES
				score += reliability * (1.f - zncc);
#endif
		}

		const int addR = r + hs + 1;
		const int remR = r - hs;
		if (addR < rows && remR >= 0) {
#if MESHOPT_IMAGE_U16
			const uint16_t* __restrict addA = imageA.ptr<uint16_t>(addR);
			const uint16_t* __restrict remA = imageA.ptr<uint16_t>(remR);
#else
			const float* __restrict addA = imageA.ptr<float>(addR);
			const float* __restrict remA = imageA.ptr<float>(remR);
#endif
			const uint16_t* __restrict addB = imageB.ptr<uint16_t>(addR);
			const uint16_t* __restrict remB = imageB.ptr<uint16_t>(remR);
			for (int x = 0; x < cols; ++x) {
				const float bAdd = float(addB[x]) * scale16;
				const float bRem = float(remB[x]) * scale16;
#if MESHOPT_IMAGE_U16
				const float aAdd = float(addA[x]) * scale16;
				const float aRem = float(remA[x]) * scale16;
#else
				const float aAdd = addA[x];
				const float aRem = remA[x];
#endif
				colB[x] += bAdd - bRem;
				colB2[x] += bAdd * bAdd - bRem * bRem;
				colAB[x] += aAdd * bAdd - aRem * bRem;
			}
		}

		if (banded && (((r + 1) % TILEY) == 0 || r + 1 == rowEnd)) {
			const size_t bandBegin = (size_t)(r / TILEY) * TILEY;
			consumeBand(imageDZNCC, bandBegin, (size_t)r + 1, r + 1 == rowEnd);
		}
	}

#ifdef MESHOPT_CERES
	return score;
#else
	return 0.f;
#endif
}

#if MESHOPT_DZNCC_WINDOWED
// GPU-parity ZNCC derivative (see MESHOPT_DZNCC_WINDOWED): identical rolling
// term math to ComputeLocalZNCCFused, but the emitted derivative averages the
// per-pixel terms over the masked interior 5x5 window, matching the CUDA
// ComputeImageDZNCC kernel. Term rows live in small ring buffers and output
// rows are emitted HalfSize rows behind the term pass, so the banded
// streaming flow (consumeBand) is preserved unchanged.
float MeshRefine::ComputeLocalZNCCFusedWindowed(
	const ImageStore& imageA,
	const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
	const TImage<uint16_t>& imageB,
	const std::vector<uint8_t>& mask,
	TImage<Real>& imageDZNCC,
	const std::function<void(const TImage<Real>&, size_t, size_t, bool)>& consumeBand)
{
	ASSERT(imageA.size() == imageB.size());
	ASSERT(imageA.size() == mask.size());

	const int rows = imageA.rows;
	const int cols = imageA.cols;
	const int hs = HalfSize;
	const int rowStart = hs;
	const int rowEnd = rows - hs;
	const int colStart = hs;
	const int colEnd = cols - hs;
	const int n = (2 * hs + 1) * (2 * hs + 1);
	const float invN = 1.0f / float(n);
	const float scale16 = 1.0f / 65535.0f;

	const bool banded = (bool)consumeBand;
	imageDZNCC.create(banded ? TILEY : rows, cols);
	if (rowStart >= rowEnd || colStart >= colEnd)
		return 0.0f;

	// double accumulators (see ComputeLocalVariance2Unmasked): the rolling sums
	// otherwise drift across thousands of row slides vs the GPU's exact windows
	static thread_local std::vector<double> colB, colB2, colAB;
	colB.assign(cols, 0.0);
	colB2.assign(cols, 0.0);
	colAB.assign(cols, 0.0);

	for (int rr = 0; rr <= 2 * hs; ++rr) {
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict aRow = imageA.ptr<uint16_t>(rr);
#else
		const float* __restrict aRow = imageA.ptr<float>(rr);
#endif
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(rr);
		for (int c = 0; c < cols; ++c) {
			const float b = float(bRow[c]) * scale16;
#if MESHOPT_IMAGE_U16
			const float a = float(aRow[c]) * scale16;
#else
			const float a = aRow[c];
#endif
			colB[c] += b;
			colB2[c] += (double)b * b;
			colAB[c] += (double)a * b;
		}
	}

	// per-row derivative-term ring (zero-filled at invalid pixels so the box
	// sums stay branch-free); RING rows cover the emission delay exactly
	constexpr int RING = 2 * HalfSize + 1;
	const size_t plane = (size_t)cols;
	static thread_local std::vector<float> ringInvS, ringZovb, ringMean, ringVarB, ringValid;
	ringInvS.assign(RING * plane, 0.f);
	ringZovb.assign(RING * plane, 0.f);
	ringMean.assign(RING * plane, 0.f);
	ringVarB.assign(RING * plane, 0.f);
	ringValid.assign(RING * plane, 0.f);

#ifdef MESHOPT_CERES
	float score = 0.f;
#endif

	static thread_local std::vector<float> colInv, colZ, colM, colN;
	colInv.resize(cols);
	colZ.resize(cols);
	colM.resize(cols);
	colN.resize(cols);

	const auto emitRow = [&](int rOut) {
		std::fill(colInv.begin(), colInv.end(), 0.f);
		std::fill(colZ.begin(), colZ.end(), 0.f);
		std::fill(colM.begin(), colM.end(), 0.f);
		std::fill(colN.begin(), colN.end(), 0.f);
		const int qr0 = MAXF(rOut - hs, rowStart);
		const int qr1 = MINF(rOut + hs, rowEnd - 1);
		for (int qr = qr0; qr <= qr1; ++qr) {
			const size_t s = (size_t)(qr % RING) * plane;
			const float* __restrict ri = ringInvS.data() + s;
			const float* __restrict rz = ringZovb.data() + s;
			const float* __restrict rm = ringMean.data() + s;
			const float* __restrict rv = ringValid.data() + s;
			float* __restrict ci = colInv.data();
			float* __restrict cz = colZ.data();
			float* __restrict cm = colM.data();
			float* __restrict cn = colN.data();
			for (int c = 0; c < cols; ++c) {
				ci[c] += ri[c];
				cz[c] += rz[c];
				cm[c] += rm[c];
				cn[c] += rv[c];
			}
		}
		const uint8_t* __restrict maskRow = &mask[(size_t)rOut * cols];
#if MESHOPT_IMAGE_U16
		const uint16_t* __restrict aRow = imageA.ptr<uint16_t>(rOut);
#else
		const float* __restrict aRow = imageA.ptr<float>(rOut);
#endif
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(rOut);
		const float* __restrict varARow = imageVarA.ptr<float>(rOut);
		const float* __restrict varBRow = ringVarB.data() + (size_t)(rOut % RING) * plane;
		float* __restrict gradRow = imageDZNCC.ptr<float>(banded ? rOut % TILEY : rOut);
		for (int i = colStart; i < colEnd; ++i) {
			if (!maskRow[i])
				continue;
			float sInv = 0.f, sZ = 0.f, sM = 0.f, sN = 0.f;
			for (int o = -hs; o <= hs; ++o) {
				sInv += colInv[i + o];
				sZ += colZ[i + o];
				sM += colM[i + o];
				sN += colN[i + o];
			}
			// the center term is always present -> sN >= 1
			const float invCnt = 1.f / sN;
#if MESHOPT_IMAGE_U16
			const float aVal = float(aRow[i]) * scale16;
#else
			const float aVal = aRow[i];
#endif
			const float bVal = float(bRow[i]) * scale16;
			const float minV = MINF(varARow[i], varBRow[i]);
			const float reliability = minV / (minV + 0.0015f);
			// == -reliability * avg_q(dZNCC_q); reduces to the center-only value
			// when the window holds a single valid term
			gradRow[i] = reliability * (bVal * sZ - aVal * sInv + sM) * invCnt;
		}
		if (banded && (((rOut + 1) % TILEY) == 0 || rOut + 1 == rowEnd)) {
			const size_t bandBegin = (size_t)(rOut / TILEY) * TILEY;
			consumeBand(imageDZNCC, bandBegin, (size_t)rOut + 1, rOut + 1 == rowEnd);
		}
	};

	int nextEmit = rowStart;
	for (int r = rowStart; r < rowEnd; ++r) {
		const size_t s = (size_t)(r % RING) * plane;
		float* __restrict rowInvS = ringInvS.data() + s;
		float* __restrict rowZovb = ringZovb.data() + s;
		float* __restrict rowMean = ringMean.data() + s;
		float* __restrict rowVarB = ringVarB.data() + s;
		float* __restrict rowValid = ringValid.data() + s;
		memset(rowInvS, 0, plane * sizeof(float));
		memset(rowZovb, 0, plane * sizeof(float));
		memset(rowMean, 0, plane * sizeof(float));
		memset(rowVarB, 0, plane * sizeof(float));
		memset(rowValid, 0, plane * sizeof(float));

		const uint8_t* __restrict maskRow = &mask[(size_t)r * cols];
		const uint16_t* __restrict meanARow = imageMeanA.ptr<uint16_t>(r);
		const float* __restrict varARow = imageVarA.ptr<float>(r);
		for (int i = colStart; i < colEnd; ++i) {
			if (!maskRow[i])
				continue;
			double sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
			for (int o = -hs; o <= hs; ++o) {
				sumB += colB[i + o];
				sumB2 += colB2[i + o];
				sumAB += colAB[i + o];
			}
			const double meanBRawD = sumB * invN;
			const float meanBRaw = (float)meanBRawD;
			const float meanB = float((uint16_t)(meanBRaw * 65535.f + 0.5f)) * scale16;
			const float varB = MAXF((float)(sumB2 * invN - meanBRawD * meanBRawD), 0.0001f);
			const float meanA = float(meanARow[i]) * scale16;
			const float varA = varARow[i];
			float product = MAXF(varA * varB, 1e-12f);
			float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(product)));
			invS *= 1.5f - 0.5f * product * invS * invS;
			const float zncc = (float)(sumAB * invN - (double)meanA * meanB) * invS;
			const float zovb = zncc / varB;
			rowInvS[i] = invS;
			rowZovb[i] = zovb;
			rowMean[i] = meanA * invS - meanB * zovb;
			rowVarB[i] = varB;
			rowValid[i] = 1.f;
#ifdef MESHOPT_CERES
			const float minV = MINF(varA, varB);
			score += (minV / (minV + 0.0015f)) * (1.f - zncc);
#endif
		}

		const int addR = r + hs + 1;
		const int remR = r - hs;
		if (addR < rows && remR >= 0) {
#if MESHOPT_IMAGE_U16
			const uint16_t* __restrict addA = imageA.ptr<uint16_t>(addR);
			const uint16_t* __restrict remA = imageA.ptr<uint16_t>(remR);
#else
			const float* __restrict addA = imageA.ptr<float>(addR);
			const float* __restrict remA = imageA.ptr<float>(remR);
#endif
			const uint16_t* __restrict addB = imageB.ptr<uint16_t>(addR);
			const uint16_t* __restrict remB = imageB.ptr<uint16_t>(remR);
			for (int x = 0; x < cols; ++x) {
				const float bAdd = float(addB[x]) * scale16;
				const float bRem = float(remB[x]) * scale16;
#if MESHOPT_IMAGE_U16
				const float aAdd = float(addA[x]) * scale16;
				const float aRem = float(remA[x]) * scale16;
#else
				const float aAdd = addA[x];
				const float aRem = remA[x];
#endif
				colB[x] += (double)bAdd - bRem;
				colB2[x] += (double)bAdd * bAdd - (double)bRem * bRem;
				colAB[x] += (double)aAdd * bAdd - (double)aRem * bRem;
			}
		}

		// emission trails the term pass so rOut's full window is available
		while (nextEmit + hs <= r)
			emitRow(nextEmit++);
	}
	while (nextEmit < rowEnd)
		emitRow(nextEmit++);

#ifdef MESHOPT_CERES
	return score;
#else
	return 0.f;
#endif
}
#endif // MESHOPT_DZNCC_WINDOWED

#if 1

#define FMA(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)

void MeshRefine::ComputePhotometricGradient(
	const View& viewA,
	const CameraRenderData& rd,
	const Mesh::NormalArr& faceNormals,
	const Camera& cameraA,
	const Camera& cameraB,
	const View& viewB,
	const TImage<Real>& imageDZNCC,
	const std::vector<uint8_t>& mask,
	GradArr& threadGrad,
	std::vector<uint32_t>& threadNorm,
	Real RegularizationScale,
	uint64_t setupEpoch,
	std::vector<float>& tileEnergyLocal,
	size_t rowBegin,
	size_t rowEnd,
	size_t dznccRowOffset,
	bool finalizePair)
{
	ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
	//ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());

 	size_t cols = viewA.image.cols;
	const size_t RowsEnd = MINF(rowEnd, (size_t)viewA.image.rows - HalfSize);
	const size_t ColsEnd = cols - HalfSize;

	static thread_local std::vector<uint8_t> touchedVertices;
	touchedVertices.resize(threadGrad.size(), 0);
	static thread_local std::vector<uint32_t> pairTouched;
	if (rowBegin == 0)
		pairTouched.clear();

	const size_t stride = viewA.width;

	// camera A parameters + kept maps, for on-demand recompute of per-pixel
	// geometry that used to be cached in the per-view SoA.
	const Point3f cameraC = Cast<float>(cameraA.C);
	const float rA00 = cameraA.R(0, 0), rA01 = cameraA.R(0, 1), rA02 = cameraA.R(0, 2);
	const float rA10 = cameraA.R(1, 0), rA11 = cameraA.R(1, 1), rA12 = cameraA.R(1, 2);
	const float rA20 = cameraA.R(2, 0), rA21 = cameraA.R(2, 1), rA22 = cameraA.R(2, 2);
	const float cxA = (float)cameraA.K(0, 2), cyA = (float)cameraA.K(1, 2);
	const float invFxA = 1.0f / (float)cameraA.K(0, 0), invFyA = 1.0f / (float)cameraA.K(1, 1);
	const DepthMap& depthMapA = viewA.depthMap;
	const FaceMap& faceMapA = viewA.faceMap;

	const float* P = &cameraB.Pf[0];
	const float p0 = P[0], p1 = P[1], p2 = P[2], p3 = P[3];
	const float p4 = P[4], p5 = P[5], p6 = P[6], p7 = P[7];
	const float p8 = P[8], p9 = P[9], p10 = P[10], p11 = P[11];
#if MESHOPT_PG_UNNORM_RAY
	// P*C + last column folded once: numX = (P0.ray)*depthA + projCX etc.
	const float projCX = p0 * cameraC.x + p1 * cameraC.y + p2 * cameraC.z + p3;
	const float projCY = p4 * cameraC.x + p5 * cameraC.y + p6 * cameraC.z + p7;
	const float projCW = p8 * cameraC.x + p9 * cameraC.y + p10 * cameraC.z + p11;
#endif

#if 0
	for (int ty = 0; ty < RowsEnd; ty += TILE) {
		const int yEnd = std::min(ty + TILE, RowsEnd);
		for (int tx = 0; tx < ColsEnd; tx += TILE) {
			const int xEnd = std::min(tx + TILE, ColsEnd);

			//------------------------------------------------------------------
			//  Tile inner loop
			//------------------------------------------------------------------
			for (int r = ty; r < yEnd; ++r) {
				const int base = r * stride;
				int c = tx;

				for (; c < xEnd; ++c) {
					if (!mask(r, c)) continue;
					// All image pixels are legal here; we only need to check the view marks.
					const int idx = base + c;
					const FIndex idxFace(viewA.faceMap(r, c));
					ASSERT(idxFace != NO_ID);
					const Grad N(normals[idxFace]);
					const Point3f& dA = viewA.ray[idx];
					const Point3f& storedN = viewA.storedNormal[idx];
					float Nd = viewA.Nd[idx];
					float invNd = viewA.invNd[idx];
#ifdef DEBUGPG
					if ((invNd < -10.f) || (invNd >= 0.f)) {
						__debugbreak();
						continue;
					}
#endif
					const Point3f& X = viewA.X[idx];
					// project point in second image and
					// projection Jacobian matrix in the second image of the 3D point on the surface
					MAYBEUNUSED const float depthB(ProjectVertex(cameraB.P.val, X.ptr(), xB.ptr(), xJac.val));
					ASSERT(depthB > 0);
					// compute gradient in image B
					const TMatrix<Real, 1, 2> gB(viewB.imageGrad.sample<Sampler, View::Grad>(sampler, xB));
					// compute gradient scale
					const Real dZNCC(imageDZNCC(r, c));
					const Real sg((gB * (xJac * (const TMatrix<Real, 3, 1>&)dA))(0) * dZNCC * RegularizationScale * invNd);
					// add gradient to the three vertices
					const Face& face(faces[idxFace]);
					const Point3f& b(viewA.baryMap(r, c));
					for (int v = 0; v < 3; ++v) {
						const Grad g(N * (sg * (Real)b[v]));
						const VIndex idxVert(face[v]);
#ifdef DEBUGPG
						if (idxVert >= photoGrad.size()) {
							__debugbreak();
						}
#endif

						photoGrad[idxVert] += g;
#ifdef DEBUGPG
						if (!isfinite(g.x) || !isfinite(g.y) || !isfinite(g.z)) {
							__debugbreak();
						}
#endif
						++photoGradNorm[idxVert];
					}
				}
			}
		}
	}

#else

	const __m128 invScale = _mm_set1_ps(kInvScale);
	const int maxX = static_cast<int>(viewB.width) - 2;
  	const int maxY = static_cast<int>(viewB.height) - 2;
  	const int wB = static_cast<int>(viewB.width); // planar gradient row stride
	const GradStoreT* __restrict gradXB = viewB.gradX;
	const GradStoreT* __restrict gradYB = viewB.gradY;

#if MESHOPT_PG_FACECACHE
	constexpr uint32_t FACE_SETUP_CACHE_SIZE = 128;
	struct FaceSetup {
		FIndex key;
		uint32_t g0, g1, g2;   // global vertex indices, pre-resolved
		float nx, ny, nz;      // face normal, pre-resolved
		float su0, sv0, su1, sv1, su2, sv2;
		float invZ0, invZ1, invZ2, invArea;
	};
	static thread_local FaceSetup faceSetupCache[FACE_SETUP_CACHE_SIZE];
	static thread_local const CameraRenderData* faceSetupOwner = nullptr;
	static thread_local uint64_t faceSetupOwnerEpoch = 0;
	if (faceSetupOwner != &rd || faceSetupOwnerEpoch != setupEpoch) {
		for (FaceSetup& setup : faceSetupCache)
			setup.key = (FIndex)NO_ID;
		faceSetupOwner = &rd;
		faceSetupOwnerEpoch = setupEpoch;
	}
#endif

	for (size_t ty = rowBegin; ty < RowsEnd; ty += TILEY) {
		const size_t yEnd = std::min(ty + TILEY, RowsEnd);

		for (size_t tx = 0; tx < ColsEnd; tx += TILEX) {
#if MESHOPT_NEEDS_TILE_ENERGY
			const size_t gridRow = ty / TILEY;
			const size_t gridCol = tx / TILEX;
			const size_t tileIndex = (gridRow * viewA.tilesX) + gridCol;
#endif

      		// JPB WIP BUG Tile energy check
#if !MESHOPT_DISABLE_TILE_SKIP
			if (!viewA.tileActive[tileIndex])
				continue;
#endif

#if MESHOPT_NEEDS_TILE_ENERGY
			float tileEnergy = 0.f;
#endif

			const size_t xEnd = std::min(tx + TILEX, ColsEnd);

			constexpr int LOCAL_CAP = TILEX * TILEY * 4;  // enough for most 32x32 tiles
			struct TGrad { float v[3]; };
			alignas(64) TGrad tileGrad[LOCAL_CAP];

			// tileVertices must be 0xFFFFFFFF initially, but we save from re-initializing it every time
			// by having the flushing of the tile reset this.
			alignas(64)static thread_local uint32_t tileVertices[LOCAL_CAP];
			static thread_local bool initialized = false;
			if (!initialized) {
				for (int i = 0; i < LOCAL_CAP; ++i)
					tileVertices[i] = 0xFFFFFFFF;
				initialized = true;
			}

			alignas(64) uint32_t usedIdx[LOCAL_CAP];
			size_t usedCount = 0;

			//------------------------------------------------------------------
			//  Tile inner loop
			//------------------------------------------------------------------
			for (size_t r = ty; r < yEnd; ++r) {
				if (r < (size_t)HalfSize)
					continue;
				const uint8_t* __restrict maskRow = &mask[r * cols];
				const float* __restrict pdZNCC = imageDZNCC.ptr<float>(r - dznccRowOffset);
				const cuint32_t* __restrict faceRowA = faceMapA.ptr<cuint32_t>(r);
				const float* __restrict depthRowA = depthMapA.ptr<float>(r);
				// signed cast: MSVC's u64->float conversion is a branchy sequence
				const float rowF = (float)(int)r;
				const float dyA = (rowF - cyA) * invFyA;
				const float rayRowMulX = rA10 * dyA;
				const float rayRowMulY = rA11 * dyA;
				const float rayRowMulZ = rA12 * dyA;

				const size_t cStart = MAXF(tx, (size_t)HalfSize);
				for (size_t c = cStart; c < xEnd; ++c) {
					if (!maskRow[c]) {
						// coalesce unmasked runs: skip 8 zero mask bytes at a time
						while (c + 8 <= xEnd) {
							uint64_t m8;
							memcpy(&m8, maskRow + c, sizeof(m8));
							if (m8 != 0)
								break;
							c += 8;
						}
						continue;
					}

					// recompute per-pixel geometry (previously cached in per-view SoA)
					const FIndex fA = faceRowA[c];
					const float depthA = depthRowA[c];
					const float dxA = ((float)(int)c - cxA) * invFxA;
					const Point3f rayWA(
						rA00 * dxA + rayRowMulX + rA20,
						rA01 * dxA + rayRowMulY + rA21,
						rA02 * dxA + rayRowMulZ + rA22);
#if MESHOPT_PG_FACECACHE
					// fetch/fill the face setup up front: on a hit (consecutive pixels
					// usually share a face) this replaces the dependent
					// globalFace->faceNormals and 3x globalVert gathers with one struct
					FaceSetup& setup = faceSetupCache[
						((uint32_t)fA * 0x9E3779B1u) & (FACE_SETUP_CACHE_SIZE - 1)];
					if (setup.key != fA) {
						setup.key = fA;
						const Face& faceA = rd.faces[fA];
						const CamVert& cv0 = rd.verts[faceA[0]];
						const CamVert& cv1 = rd.verts[faceA[1]];
						const CamVert& cv2 = rd.verts[faceA[2]];
						setup.g0 = rd.globalVert[faceA[0]];
						setup.g1 = rd.globalVert[faceA[1]];
						setup.g2 = rd.globalVert[faceA[2]];
						const Grad& N = faceNormals[rd.globalFace[fA]];
						setup.nx = N.x; setup.ny = N.y; setup.nz = N.z;
						setup.invZ0 = cv0.invZ; setup.invZ1 = cv1.invZ; setup.invZ2 = cv2.invZ;
						setup.su0 = cv0.x * setup.invZ0; setup.sv0 = cv0.y * setup.invZ0;
						setup.su1 = cv1.x * setup.invZ1; setup.sv1 = cv1.y * setup.invZ1;
						setup.su2 = cv2.x * setup.invZ2; setup.sv2 = cv2.y * setup.invZ2;
						setup.invArea = 1.f / (
							(setup.su1 - setup.su0) * (setup.sv2 - setup.sv0) -
							(setup.sv1 - setup.sv0) * (setup.su2 - setup.su0));
					}
					const float Nx = setup.nx, Ny = setup.ny, Nz = setup.nz;
#else
					const Grad& Nrec = faceNormals[rd.globalFace[fA]];
					const float Nx = Nrec.x, Ny = Nrec.y, Nz = Nrec.z;
#endif
#if MESHOPT_PG_UNNORM_RAY
					// |ray| cancels between the Jacobian dots and 1/(N.d): grazing test
					// in squared form (== N.d > -0.1), gradient on the unnormalized ray
					const float lenSqA = rayWA.x * rayWA.x + rayWA.y * rayWA.y + rayWA.z * rayWA.z;
					const float NdU = Nx * rayWA.x + Ny * rayWA.y + Nz * rayWA.z;
					if (NdU >= 0.f || NdU * NdU < 0.01f * lenSqA)
						continue;
					const float invNd = 1.0f / NdU;
					const float dx = rayWA.x, dy = rayWA.y, dz = rayWA.z;

					// projection reuses the Jacobian ray dots: P*(ray*depth + C)
					// = (P.ray)*depth + P.C, with P.C folded per call (projC*)
					const float t0 = p0 * dx + p1 * dy + p2 * dz;
					const float t1 = p4 * dx + p5 * dy + p6 * dz;
					const float tW = p8 * dx + p9 * dy + p10 * dz;
					const float numX = t0 * depthA + projCX;
					const float numY = t1 * depthA + projCY;
					const float denW = tW * depthA + projCW;
#else
					const float lenSqA = rayWA.x * rayWA.x + rayWA.y * rayWA.y + rayWA.z * rayWA.z;
					float invLenA = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lenSqA)));
					invLenA *= 1.5f - 0.5f * lenSqA * invLenA * invLenA;
					const Point3f dA(rayWA.x * invLenA, rayWA.y * invLenA, rayWA.z * invLenA);
					const float Nd = Nx * dA.x + Ny * dA.y + Nz * dA.z;
					if (Nd > -0.1f)
						continue;
					const float invNd = 1.0f / Nd;
					const float dx = dA.x, dy = dA.y, dz = dA.z;
					const Point3f X(rayWA * depthA + cameraC);

					const float X0 = X.x, X1 = X.y, X2 = X.z;

					// Compute projection numerator and denominator
					const float numX = p0 * X0 + p1 * X1 + p2 * X2 + p3;
					const float numY = p4 * X0 + p5 * X1 + p6 * X2 + p7;
					const float denW = p8 * X0 + p9 * X1 + p10 * X2 + p11;
#endif

					// Depth is strictly > 0 for valid pixels,
					// Thus denW cannot be 0 unless the point is exactly at the camera center, 
					// which never happens in our geometry.
					const float invW = 1.0f / denW;

					// Final projected pixel coords in B
					const float xB = numX * invW;
					const float yB = numY * invW;

					// ------------------------------------------------------------
					//  Bilinear sample from pre-packed gradient blocks
					// ------------------------------------------------------------
#if MESHOPT_PG_GRAD_CENTERED
					// masked pixels have xB,yB > 10 (warp border), so gsx/gsy stay positive
					const float gsx = xB - 0.5f;
					const float gsy = yB - 0.5f;
#else
					const float gsx = xB;
					const float gsy = yB;
#endif
					int xi = _cvt_ftoi_fast(gsx);
					int yi = _cvt_ftoi_fast(gsy);

					if ((unsigned)xi > (unsigned)maxX || (unsigned)yi > (unsigned)maxY)
						continue;

					const float fx = gsx - xi;
					const float fy = gsy - yi;
					const size_t off = static_cast<size_t>(yi) * wB + xi;
#if MESHOPT_GRAD_F32
					// scalar 2x2 lerp on the float planes (SIMD gather paths are int16-only)
					const float* __restrict gx0 = gradXB + off;
					const float* __restrict gy0 = gradYB + off;
					const float gxTop = gx0[0] + fx * (gx0[1] - gx0[0]);
					const float gxBot = gx0[wB] + fx * (gx0[wB + 1] - gx0[wB]);
					const float gBx = gxTop + fy * (gxBot - gxTop);
					const float gyTop = gy0[0] + fx * (gy0[1] - gy0[0]);
					const float gyBot = gy0[wB] + fx * (gy0[wB + 1] - gy0[wB]);
					const float gBy = gyTop + fy * (gyBot - gyTop);
#else
					const int16_t* __restrict gx0 = gradXB + off;
					const int16_t* __restrict gy0 = gradYB + off;

					const __m128 fxv = _mm_set1_ps(fx);
					const __m128 fyv = _mm_set1_ps(fy);
#if MESHOPT_PG_BILERP_MERGE
					// Pack {gx00,gx01,gy00,gy01} and the matching bottom row,
					// then interpolate both channels in the same SIMD lanes.
					const __m128i top16 = _mm_unpacklo_epi32(
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gx0)),
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gy0)));
					const __m128i bot16 = _mm_unpacklo_epi32(
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gx0 + wB)),
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gy0 + wB)));

					__m128 topv = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(top16)), invScale);
					__m128 botv = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi16_epi32(bot16)), invScale);
					const __m128 lerpV = _mm_add_ps(topv, _mm_mul_ps(fyv, _mm_sub_ps(botv, topv)));
					const __m128 shiftV = _mm_shuffle_ps(lerpV, lerpV, _MM_SHUFFLE(2, 3, 0, 1));
					const __m128 resV = _mm_add_ps(lerpV, _mm_mul_ps(fxv, _mm_sub_ps(shiftV, lerpV)));

					const float gBx = _mm_cvtss_f32(resV);
					const float gBy = _mm_cvtss_f32(_mm_movehl_ps(resV, resV));
#else
					// gather 2x2 bilinear block from two adjacent planar rows
					// each row holds gx00,gx01 (resp. gy) contiguously; combine with yi+1
					const __m128i gx16 = _mm_unpacklo_epi32(
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gx0)),       // gx00,gx01
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gx0 + wB))); // gx10,gx11
					const __m128i gy16 = _mm_unpacklo_epi32(
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gy0)),       // gy00,gy01
						_mm_cvtsi32_si128(*reinterpret_cast<const int*>(gy0 + wB))); // gy10,gy11

					__m128 gxv = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gx16));
					__m128 gyv = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gy16));

					// rescale back to original float range
					gxv = _mm_mul_ps(gxv, invScale);
					gyv = _mm_mul_ps(gyv, invScale);

					// unpack rows: top (00,01), bottom (10,11)
					const __m128 gx_top = _mm_movelh_ps(gxv, gxv); // 00,01,00,01
					const __m128 gx_bot = _mm_movehl_ps(gxv, gxv); // 10,11,10,11
					const __m128 gy_top = _mm_movelh_ps(gyv, gyv);
					const __m128 gy_bot = _mm_movehl_ps(gyv, gyv);

					// vertical interpolation
					gxv = _mm_add_ps(gx_top, _mm_mul_ps(fyv, _mm_sub_ps(gx_bot, gx_top)));
					gyv = _mm_add_ps(gy_top, _mm_mul_ps(fyv, _mm_sub_ps(gy_bot, gy_top)));

					// horizontal interpolation
					const __m128 gx_shift = _mm_shuffle_ps(gxv, gxv, _MM_SHUFFLE(3, 3, 1, 1));
					const __m128 gy_shift = _mm_shuffle_ps(gyv, gyv, _MM_SHUFFLE(3, 3, 1, 1));

					const __m128 gx_res = _mm_add_ps(gxv, _mm_mul_ps(fxv, _mm_sub_ps(gx_shift, gxv)));
					const __m128 gy_res = _mm_add_ps(gyv, _mm_mul_ps(fxv, _mm_sub_ps(gy_shift, gyv)));

					// gBx, gBy now contain the scalar gradients from the bilinear sample
					const float gBx = _mm_cvtss_f32(gx_res);
					const float gBy = _mm_cvtss_f32(gy_res);
#endif
#endif // MESHOPT_GRAD_F32

					// ------------------------------------------------------------
					//  Jacobian partials
					// ------------------------------------------------------------
#if !MESHOPT_PG_UNNORM_RAY
					// Precompute linear dot products
					const float t0 = p0 * dx + p1 * dy + p2 * dz;     // numerator x part
					const float t1 = p4 * dx + p5 * dy + p6 * dz;     // numerator y part
					const float tW = p8 * dx + p9 * dy + p10 * dz;    // denominator part
#endif

					// Final Jacobian dot-products
					const float dot0 = t0 - xB * tW;
					const float dot1 = t1 - yB * tW;

					// ------------------------------------------------------------
					//  Gradient scale
					// ------------------------------------------------------------
					const float dZNCC = pdZNCC[c];

					// JPB WIP BUG
					// This eliminates pixels where the photometric patch correlation is weak, 
					// meaning the pixel is not contributing meaningful refinement signal.
					// Instead of checking dZNCC gradient, check actual ZNCC score
					// (requires passing ZNCC values from ComputeLocalZNCC)
					//if (dZNCC < 0.1f) continue;
					const float sg = (gBx * dot0 + gBy * dot1) * invW * invNd * RegularizationScale * dZNCC;
#if MESHOPT_NEEDS_TILE_ENERGY
					tileEnergy += sg * sg;
#endif

					// ------------------------------------------------------------
					//  Accumulate per-vertex gradients
					// ------------------------------------------------------------
#if MESHOPT_PG_FACECACHE
					// setup was fetched/filled at the top of the pixel
					const float bpx = (float)(int)c, bpy = rowF;
					const float bw0 = ((setup.su2 - setup.su1) * (bpy - setup.sv1) - (setup.sv2 - setup.sv1) * (bpx - setup.su1)) * setup.invArea;
					const float bw1 = ((setup.su0 - setup.su2) * (bpy - setup.sv2) - (setup.sv0 - setup.sv2) * (bpx - setup.su2)) * setup.invArea;
					const float bw2 = 1.f - bw0 - bw1;
					const float bDen = 1.f / (bw0 * setup.invZ0 + bw1 * setup.invZ1 + bw2 * setup.invZ2);
					const float bx = (bw0 * setup.invZ0) * bDen;
					const float by = (bw1 * setup.invZ1) * bDen;
					const float bz = 1.f - bx - by;
#else
					const Face& faceA = rd.faces[fA];
					const FIndex v1 = faceA[0];
					const FIndex v2 = faceA[1];
					const FIndex v3 = faceA[2];

					// recompute perspective-correct barycentric of pixel (c,r) within
					// face fA directly from its camera-space triangle (same math the
					// rasterizer used) instead of storing a baryMap plane.
					const CamVert& bcv0 = rd.verts[v1];
					const CamVert& bcv1 = rd.verts[v2];
					const CamVert& bcv2 = rd.verts[v3];
					const float bsu0 = bcv0.x * bcv0.invZ, bsv0 = bcv0.y * bcv0.invZ;
					const float bsu1 = bcv1.x * bcv1.invZ, bsv1 = bcv1.y * bcv1.invZ;
					const float bsu2 = bcv2.x * bcv2.invZ, bsv2 = bcv2.y * bcv2.invZ;
					const float bInvArea = 1.f / ((bsu1 - bsu0) * (bsv2 - bsv0) - (bsv1 - bsv0) * (bsu2 - bsu0));
					const float bpx = (float)(int)c, bpy = rowF;
					const float bw0 = ((bsu2 - bsu1) * (bpy - bsv1) - (bsv2 - bsv1) * (bpx - bsu1)) * bInvArea;
					const float bw1 = ((bsu0 - bsu2) * (bpy - bsv2) - (bsv0 - bsv2) * (bpx - bsu2)) * bInvArea;
					const float bw2 = 1.f - bw0 - bw1;
					const float bDen = 1.f / (bw0 * bcv0.invZ + bw1 * bcv1.invZ + bw2 * bcv2.invZ);
					const float bx = (bw0 * bcv0.invZ) * bDen;
					const float by = (bw1 * bcv1.invZ) * bDen;
					const float bz = 1.f - bx - by;
#endif

					const Grad Ng(Nx * sg, Ny * sg, Nz * sg);      // scale once

					auto accum = [&](uint32_t vi, const Grad& g)
					{
						uint32_t slot = (vi * 0x9E3779B1u) & (LOCAL_CAP - 1);
						uint32_t step = 1;

						uint32_t* __restrict tv = tileVertices;
						TGrad* __restrict tg = tileGrad;

						for (;;) {
							uint32_t v = tv[slot];
							if (v == vi) {
								auto& t = tg[slot];
								t.v[0] += g.x;
								t.v[1] += g.y;
								t.v[2] += g.z;
								return;
							}

							if (v == 0xFFFFFFFF) {
								tv[slot] = vi;
								auto& t = tg[slot];
								t.v[0] = g.x;
								t.v[1] = g.y;
								t.v[2] = g.z;
								usedIdx[usedCount++] = slot;
								return;
							}

							slot = (slot + step) & (LOCAL_CAP - 1);
							step += 1;
						}
					};

#if MESHOPT_PG_FACECACHE
					accum(setup.g0, Ng * bx);
					accum(setup.g1, Ng * by);
					accum(setup.g2, Ng * bz);
#else
					accum(rd.globalVert[v1], Ng * bx);
					accum(rd.globalVert[v2], Ng * by);
					accum(rd.globalVert[v3], Ng * bz);
#endif
				}
			} // end rows in tile

			// First pass: accumulate gradient only
			for (size_t i = 0; i < usedCount; ++i) {
				uint32_t slot = usedIdx[i];
				uint32_t vi = tileVertices[slot];
				const auto& g = tileGrad[slot];

				Grad& tg = threadGrad[vi];
				tg.x += g.v[0];
				tg.y += g.v[1];
				tg.z += g.v[2];

				if (!touchedVertices[vi]) {
					touchedVertices[vi] = 1;
					pairTouched.push_back(vi);
				}

				tileVertices[slot] = 0xFFFFFFFF;
			}
			usedCount = 0;
#if MESHOPT_NEEDS_TILE_ENERGY
			tileEnergyLocal[tileIndex] += tileEnergy;
#endif
		} // end tx
	} // end ty
#endif

	if (finalizePair) {
		for (uint32_t vi : pairTouched) {
			++threadNorm[vi];
			touchedVertices[vi] = 0;
		}
	}
}
#else
//original
// compute the photometric gradient for all vertices seen by an image pair
void MeshRefine::ComputePhotometricGradient(
	const Mesh::FaceArr& faces, const Mesh::NormalArr& normals,
	const DepthMap& depthMapA, const FaceMap& faceMapA, const BaryMap& baryMapA, const Camera& cameraA,
	const Camera& cameraB, const View& viewB,
	const TImage<Real>& imageDZNCC, const BitMatrix& mask, GradArr& photoGrad, UnsignedArr& photoGradNorm, Real RegularizationScale)
{
	ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
	ASSERT(depthMapA.size() == mask.size() && faceMapA.size() == mask.size() && baryMapA.size() == mask.size() && imageDZNCC.size() == mask.size() && !mask.empty());
	ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());
	const int RowsEnd(mask.rows - HalfSize);
	const int ColsEnd(mask.cols - HalfSize);
	typedef Sampler::Linear<View::Grad::Type> Sampler;
	const Sampler sampler;
	TMatrix<Real, 2, 3> xJac;
	Point2f xB;
	photoGrad.Memset(0);
	photoGradNorm.Memset(0);
	for (int r = HalfSize; r < RowsEnd; ++r) {
		for (int c = HalfSize; c < ColsEnd; ++c) {
			if (!mask(r, c))
				continue;
			const FIndex idxFace(faceMapA(r, c));
			ASSERT(idxFace != NO_ID);
			const Grad N(normals[idxFace]);
			const Point3 rayA(cameraA.RayPoint(Point2(c, r)));
			const Grad dA(normalized(rayA));
			const Real Nd(N.dot(dA));
#if 1
			if (Nd > -0.1)
				continue;
#endif
			const Depth depthA(depthMapA(r, c));
			ASSERT(depthA > 0);
			const Point3 X(rayA * REAL(depthA) + cameraA.C);
			// project point in second image and
			// projection Jacobian matrix in the second image of the 3D point on the surface
			MAYBEUNUSED const float depthB(ProjectVertex(cameraB.P.val, X.ptr(), xB.ptr(), xJac.val));
			ASSERT(depthB > 0);
			// compute gradient in image B
			const TMatrix<Real, 1, 2> gB(viewB.imageGrad.sample<Sampler, View::Grad>(sampler, xB));
			// compute gradient scale
			const Real dZNCC(imageDZNCC(r, c));
			const Real sg((gB * (xJac * (const TMatrix<Real, 3, 1>&)dA))(0) * dZNCC * RegularizationScale / Nd);
			// add gradient to the three vertices
			const Face& face(faces[idxFace]);
			const Point3f& b(baryMapA(r, c));
			for (int v = 0; v < 3; ++v) {
				const Grad g(N * (sg * (Real)b[v]));
				const VIndex idxVert(face[v]);
				photoGrad[idxVert] += g;
				++photoGradNorm[idxVert];
			}
		}
	}
}
#endif

// computes the discrete analog of the Laplacian using
// the umbrella-operator on the first triangle ring at each point
float MeshRefine::ComputeSmoothnessGradient1(
	const Mesh::VertexArr& vertices, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
	GradArr& smoothGrad1, VIndex idxStart, VIndex idxEnd)
{
	ASSERT(!vertices.IsEmpty() && vertices.GetSize() == vertexVertices.GetSize() && vertices.GetSize() == smoothGrad1.GetSize());
#ifdef MESHOPT_CERES
	float score(0);
#endif
	for (VIndex idxV = idxStart; idxV < idxEnd; ++idxV) {
		Grad& grad = smoothGrad1[idxV];
		grad = Grad::ZERO;
#if 1
		if (vertexBoundary[idxV])
			continue;
#endif
		const Mesh::VertexIdxArr& verts = vertexVertices[idxV];
		if (verts.IsEmpty())
			continue;
		FOREACH(v, verts)
			grad += Cast<Real>(vertices[verts[v]]);
		grad = grad / (Real)verts.GetSize() - Cast<Real>(vertices[idxV]);
#ifdef MESHOPT_CERES
		const float regularityScore((float)norm(grad));
		ASSERT(ISFINITE(regularityScore));
		score += regularityScore;
#endif
	}
#ifdef MESHOPT_CERES
	return score;
#else
	return 0.0f;
#endif
}
// same as above, but used to compute level 2;
// normalized as in "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004
void MeshRefine::ComputeSmoothnessGradient2(
	const GradArr& smoothGrad1, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
	GradArr& smoothGrad2, VIndex idxStart, VIndex idxEnd)
{
	ASSERT(!smoothGrad1.IsEmpty() && smoothGrad1.GetSize() == vertexVertices.GetSize() && smoothGrad1.GetSize() == smoothGrad2.GetSize());
	for (VIndex idxV = idxStart; idxV < idxEnd; ++idxV) {
		Grad& grad = smoothGrad2[idxV];
		grad = Grad::ZERO;
#if 1
		if (vertexBoundary[idxV])
			continue;
#endif
		const Mesh::VertexIdxArr& verts = vertexVertices[idxV];
		if (verts.IsEmpty())
			continue;
		Real w(0);
		FOREACH(v, verts) {
			const VIndex idxVert(verts[v]);
			grad += smoothGrad1[idxVert];
			const VIndex numVert(vertexVertices[idxVert].GetSize());
			if (numVert > 0)
				w += Real(1) / (Real)numVert;
		}
		const Real numVert((Real)verts.GetSize());
		const Real nrm(Real(1) / (Real(1) + w / numVert));
		grad = grad * (nrm / numVert) - smoothGrad1[idxV] * nrm;
	}
}


void* MeshRefine::ThreadWorkerTmp(void* arg) {
	MeshRefine& refine = *((MeshRefine*)arg);
	refine.ThreadWorker();
	return NULL;
}
void MeshRefine::ThreadWorker()
{
	while (true) {
		CAutoPtr<Event> evt(events.GetEvent());
		switch (evt->GetID()) {
		case EVT_JOB:
			evt->Run(this);
			break;
		case EVT_CLOSE:
			return;
		default:
			ASSERT("Should not happen!" == NULL);
		}
		sem.Signal();
	}
}
void MeshRefine::WaitThreadWorkers(size_t nJobs)
{
	while (nJobs-- > 0)
		sem.Wait();
	ASSERT(events.IsEmpty());
}
void MeshRefine::ThSelectNeighbors(uint32_t idxImage, std::unordered_set<uint64_t>& mapPairs, unsigned nMaxViews)
{
	// keep only best neighbor views
	const float fMinArea(0.1f);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(2.5f)), fMaxAngle(FD2R(45.f));
	Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	if (imageData.neighbors.IsEmpty()) {
		IndexArr points;
		scene.SelectNeighborViews(idxImage, points);
	}
	ViewScoreArr neighbors(imageData.neighbors);
	Scene::FilterNeighborViews(neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews);
	Lock l(cs);
	FOREACHPTR(pNeighbor, neighbors) {
		ASSERT(images[pNeighbor->idx.ID].IsValid());
		mapPairs.insert(MakePairIdx((uint32_t)idxImage, pNeighbor->ID));
	}
}

void MeshRefine::ThInitImage(uint32_t idxImage, Real scale, Real sigma)
{
	Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	// load and init image
	unsigned level(resolutionLevelOverride != unsigned(-1) ? resolutionLevelOverride : nResolutionLevel);
	const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
	View& view = views[idxImage];
#if MESHOPT_IMAGE_U16
	// float working buffer; quantized into the uint16 view.image store below.
	Image32F img;
#else
	Image32F& img = view.image;
#endif
#if MESHOPT_CACHE_IMAGES
	// Reuse the decoded color image across the coarse->fine scale levels:
	// imageSize depends only on nResolutionLevel (constant per run), so the
	// decoded pixels are identical every level. On the first level we decode
	// and stash the color buffer in the view; later levels skip the disk
	// decode entirely. toGray + the per-level blur/resize below still run on a
	// fresh gray image, so results are bit-identical to the non-cached path.
	if (view.imageColorCache.empty() || MAXF(view.imageColorCache.width(), view.imageColorCache.height()) != (int)imageSize) {
		if ((imageData.image.empty() || MAXF(imageData.width, imageData.height) != imageSize) && !imageData.ReloadImage(imageSize))
			ABORT("can not load image");
		cv::swap(imageData.image, view.imageColorCache);
	}
	imageData.width = view.imageColorCache.width();
	imageData.height = view.imageColorCache.height();
	view.imageColorCache.toGray(img, cv::COLOR_BGR2GRAY, true);
#else
	if ((imageData.image.empty() || MAXF(imageData.width, imageData.height) != imageSize) && !imageData.ReloadImage(imageSize))
		ABORT("can not load image");
	imageData.image.toGray(img, cv::COLOR_BGR2GRAY, true);
	imageData.image.release();
#endif
	if (sigma > 0)
		cv::GaussianBlur(img, img, cv::Size(), sigma);
	if (scale < 1.0) {
		cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
		imageData.width = img.width(); imageData.height = img.height();
	}
	imageData.UpdateCamera(scene.platforms);
	if (!nReduceMemory) {
		throw; // Unsupported
#if 0
		// compute image mean and variance
		ComputeLocalVariance(img, std::vector<uint8_t>(img.cols * img.rows, 0xFF), view.imageMean, view.imageVar);
#endif
	}
	// compute image gradient
	typedef View::Grad::Type GradType;
	static thread_local TImage<GradType> grad[2];
#if MESHOPT_GRAD_KERNEL3X5
	{
		const TMatrix<GradType, 3, 5> kernel(CreateDerivativeKernel3x5());
		cv::filter2D(img, grad[0], cv::DataType<GradType>::type, kernel);
		cv::filter2D(img, grad[1], cv::DataType<GradType>::type, kernel.t());
	}
#else
	grad[0].create(img.rows, img.cols);
	grad[1].create(img.rows, img.cols);

	for (int y = 0; y < img.rows; ++y) {
		const GradType* __restrict src = img.ptr<GradType>(y);
		const GradType* __restrict next = img.ptr<GradType>(MINF(y + 1, img.rows - 1));
		GradType* __restrict gradX = grad[0].ptr<GradType>(y);
		GradType* __restrict gradY = grad[1].ptr<GradType>(y);
		for (int x = 0; x < img.cols; ++x) {
			gradX[x] = img.ptr<GradType>(y)[MINF(x + 1, img.cols - 1)] - src[x];
			gradY[x] = next[x] - src[x];
		}
	}
#endif
#ifdef VALIDATE_GRADIENT
	cv::merge(grad, 2, view.imageGrad);
#endif

	// ------------------------------------------------------------------
	// Build packed 2x2 gradient blocks (int16 quantized)
	// ------------------------------------------------------------------
	const int width = view.width = grad[0].cols;
	const int height = view.height = grad[0].rows;
	// Store gradients as two planar int16 planes (gx, gy) at full resolution.
	// The 2x2 bilinear block consumed by ComputePhotometricGradient is gathered
	// on the fly from two adjacent rows. This is 4x smaller than pre-expanding
	// the redundant 2x2 blocks (16 -> 4 bytes/pixel) and more cache-friendly.
	const size_t numPx = static_cast<size_t>(height) * width;

	PlaneFree(view.gradX); // Will leak on exit.
	PlaneFree(view.gradY);
	view.gradX = (GradStoreT*)PlaneAlloc(numPx * sizeof(GradStoreT));
	view.gradY = (GradStoreT*)PlaneAlloc(numPx * sizeof(GradStoreT));

#if MESHOPT_GRAD_F32
	// float planes: copy the filter output rows verbatim (no quantization)
	for (int y = 0; y < height; ++y) {
		memcpy(view.gradX + (size_t)y * width, grad[0].ptr<GradType>(y), sizeof(float) * width);
		memcpy(view.gradY + (size_t)y * width, grad[1].ptr<GradType>(y), sizeof(float) * width);
	}
#else
	auto quant = [](float f) -> int16_t {
		float scaled = f * kScale;
		if (scaled > 32767.f)  scaled = 32767.f;
		if (scaled < -32767.f) scaled = -32767.f;
		return static_cast<int16_t>(scaled);
	};

	// SSE2 quantize: (f*kScale) clamped to +/-32767, truncated to int16. The
	// clamp is done in float (min/max), cvtt truncates toward zero (== the
	// static_cast above) and packs saturates (never triggered post-clamp), so
	// the 8-wide path is bit-identical to the scalar quant. Processes 8 grads
	// per store; a scalar tail handles widths not divisible by 8.
	const __m128 vScale = _mm_set1_ps(kScale);
	const __m128 vHi = _mm_set1_ps(32767.f);
	const __m128 vLo = _mm_set1_ps(-32767.f);
	auto quant8 = [&](const GradType* __restrict src, int16_t* __restrict dst, int x) {
		__m128 a = _mm_mul_ps(_mm_loadu_ps(src + x), vScale);
		__m128 b = _mm_mul_ps(_mm_loadu_ps(src + x + 4), vScale);
		a = _mm_min_ps(_mm_max_ps(a, vLo), vHi);
		b = _mm_min_ps(_mm_max_ps(b, vLo), vHi);
		const __m128i packed = _mm_packs_epi32(_mm_cvttps_epi32(a), _mm_cvttps_epi32(b));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), packed);
	};

	for (int y = 0; y < height; ++y) {
		const GradType* __restrict gxRow = grad[0].ptr<GradType>(y);
		const GradType* __restrict gyRow = grad[1].ptr<GradType>(y);
		int16_t* __restrict gxOut = view.gradX + static_cast<size_t>(y) * width;
		int16_t* __restrict gyOut = view.gradY + static_cast<size_t>(y) * width;
		int x = 0;
		for (; x + 8 <= width; x += 8) {
			quant8(gxRow, gxOut, x);
			quant8(gyRow, gyOut, x);
		}
		for (; x < width; ++x) {
			gxOut[x] = quant(gxRow[x]);
			gyOut[x] = quant(gyRow[x]);
		}
	}
#endif // MESHOPT_GRAD_F32

#if MESHOPT_IMAGE_U16
	// Quantize the float grayscale [0,1] into the resident uint16 store (0..65535).
	// Halves view.image from 4 to 2 B/px; the 8-bit JPEG source makes this
	// near-lossless and below the ZNCC variance floor / flatness-prune thresholds.
	view.image.create(img.rows, img.cols);
	for (int y = 0; y < img.rows; ++y) {
		const float* __restrict s = img.ptr<float>(y);
		uint16_t* __restrict d = view.image.ptr<uint16_t>(y);
		for (int x = 0; x < img.cols; ++x) {
			float v = s[x];
			if (v < 0.f) v = 0.f; else if (v > 1.f) v = 1.f;
			d[x] = (uint16_t)(v * 65535.f + 0.5f);
		}
	}
#endif

	// The Sobel float scratch (grad[0]/grad[1]) is static thread_local, so it
	// otherwise stays resident for the ENTIRE refinement (2*4 B/px per thread,
	// ~1.2 GB across 32 threads at res-level 1) even though it is only needed
	// here in InitImage to build the int16 gradX/gradY planes. Release it now;
	// it is recreated on the next scale's InitImage (once per scale = negligible).
	grad[0].release();
	grad[1].release();
}

void MeshRefine::ThProjectMesh(uint32_t idxImage, const Mesh::FaceIdxArr& cameraFaces, const CameraRenderData& rd)
{
	const Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
#if MESHOPT_VIEW_STREAM
	// expand/refresh this view's CameraRenderData here so it happens on the worker
	// pool in parallel rather than serially in EnsureViewsResident's omp single
	EnsureCameraData(idxImage);
#endif
	// project mesh to the given camera plane
	View& view = views[idxImage];
	ProjectMesh(view, imageData.camera, rd);

	view.tilesX = (view.width + TILEX - 1) / TILEX;
	view.tilesY = (view.height + TILEY - 1) / TILEY;
}
void MeshRefine::ThProcessPair(uint32_t idxImageA, uint32_t idxImageB, GradArr& threadGrad, std::vector<uint32_t>& threadNorm, std::vector<std::vector<float>>& tileEnergyLocal,
	const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA)
{
	// fetch view A data
	const Image& imageDataA = images[idxImageA];
	ASSERT(imageDataA.IsValid());
	const View& viewA = views[idxImageA];
	const FaceMap& faceMapA = viewA.faceMap;
	const DepthMap& depthMapA = viewA.depthMap;
	const ImageStore& imageA = viewA.image;
	const Camera& cameraA = imageDataA.camera;
	// fetch view B data
	const Image& imageDataB = images[idxImageB];
	ASSERT(imageDataB.IsValid());
	const View& viewB = views[idxImageB];
	const DepthMap& depthMapB = viewB.depthMap;
	const ImageStore& imageB = viewB.image;
	const Camera& cameraB = imageDataB.camera;
	// warp imageB to imageA using the mesh

	size_t numPixels = imageA.cols * imageA.rows;
	static thread_local std::vector<uint8_t> mask;
	mask.resize(numPixels);

	static thread_local TImage<uint16_t> imageAB;
	imageAB.create(imageA.rows, imageA.cols);
#if MESHOPT_PROFILE
	MeshProf::Timer _tWarp;
#endif
	ImageMeshWarp(viewA, depthMapA, cameraA, depthMapB, cameraB, imageB, imageAB, mask);
#if MESHOPT_PROFILE
	MeshProf::AddTL(MeshProf::P_Warp, _tWarp.ms());
#endif
#if MESHOPT_SUPPORT_STAGE_DIAG
	// iter-0-only: per-pair dedup via generation stamps, atomic merge into the
	// global stage counters (diagnostic; cost confined to the first iteration)
	if (iteration == 0) {
		static thread_local std::vector<uint32_t> rasterStamp, maskStamp;
		static thread_local uint32_t stampGen = 0;
		rasterStamp.resize(vertices.GetSize(), 0);
		maskStamp.resize(vertices.GetSize(), 0);
		// stamps only ever hold past generations, so growth/shrink across scales is safe
		if (++stampGen == 0) {
			std::fill(rasterStamp.begin(), rasterStamp.end(), 0);
			std::fill(maskStamp.begin(), maskStamp.end(), 0);
			stampGen = 1;
		}
		for (int r = 0; r < faceMapA.rows; ++r) {
			const FIndex* faceRow = faceMapA.ptr<FIndex>(r);
			const uint8_t* maskRow = &mask[(size_t)r * faceMapA.cols];
			const CameraRenderData& rdA = g_cameraData[idxImageA];
			for (int c = 0; c < faceMapA.cols; ++c) {
				const FIndex idxFace = faceRow[c];
				if (idxFace == NO_ID)
					continue;
				// faceMap ids are camera-LOCAL; resolve via the render data remap
				const Face& face = rdA.faces[idxFace];
				const bool m = maskRow[c] != 0;
				for (int v = 0; v < 3; ++v) {
					const VIndex idxVert = rdA.globalVert[face[v]];
					if (rasterStamp[idxVert] != stampGen) {
						rasterStamp[idxVert] = stampGen;
						AtomicAddFloat(&rasterSupport[idxVert], 1.f);
					}
					if (m && maskStamp[idxVert] != stampGen) {
						maskStamp[idxVert] = stampGen;
						AtomicAddFloat(&maskSupport[idxVert], 1.f);
					}
				}
			}
		}
	}
#endif

	// compute ZNCC and its gradient
	// imageMeanA/imageVarA are precomputed once per reference view A by the
	// caller (ScoreMesh) and passed in, instead of being recomputed here for
	// every neighbor b. A's mean/variance depend only on viewA.image, so this
	// is bit-identical to the old per-pair recompute.
	static thread_local TImage<Real> imageDZNCC;
	const Real RegularizationScale((Real)((REAL)(imageDataA.avgDepth * imageDataB.avgDepth) / (cameraA.GetFocalLength() * cameraB.GetFocalLength())));
#if MESHOPT_PROFILE
	MeshProf::Timer _tZNCC;
#endif
#if MESHOPT_FUSED_ZNCC
#if MESHOPT_ZNCC_BANDS
	const auto consumeZNCCBand = [&](const TImage<Real>& band, size_t rowBegin, size_t rowEnd, bool finalizePair) {
		ComputePhotometricGradient(
			viewA, g_cameraData[idxImageA], faceNormals, cameraA, cameraB, viewB,
			band, mask, threadGrad, threadNorm, RegularizationScale, faceSetupEpoch,
			tileEnergyLocal[idxImageA], rowBegin, rowEnd, rowBegin, finalizePair);
	};
#ifdef MESHOPT_CERES
	const float score(MESHOPT_FUSED_ZNCC_FN(imageA, imageMeanA, imageVarA, imageAB, mask, imageDZNCC, consumeZNCCBand));
#else
	MESHOPT_FUSED_ZNCC_FN(imageA, imageMeanA, imageVarA, imageAB, mask, imageDZNCC, consumeZNCCBand);
#endif
#else
#ifdef MESHOPT_CERES
	const float score(MESHOPT_FUSED_ZNCC_FN(imageA, imageMeanA, imageVarA, imageAB, mask, imageDZNCC));
#else
	MESHOPT_FUSED_ZNCC_FN(imageA, imageMeanA, imageVarA, imageAB, mask, imageDZNCC);
#endif
#endif
#else
	static thread_local TImage<uint16_t> imageMeanAB;
	static thread_local TImage<Real> imageVarAB;
	ComputeLocalVariance2(imageAB, mask, imageMeanAB, imageVarAB);
#ifdef MESHOPT_CERES
	const float score(ComputeLocalZNCC(imageA, imageMeanA, imageVarA, imageAB, imageMeanAB, imageVarAB, mask, imageDZNCC));
#else
	ComputeLocalZNCC(imageA, imageMeanA, imageVarA, imageAB, imageMeanAB, imageVarAB, mask, imageDZNCC);
#endif
#endif
#if MESHOPT_PROFILE
	MeshProf::AddTL(MeshProf::P_ZNCC, _tZNCC.ms());
#endif

#if !MESHOPT_FUSED_ZNCC || !MESHOPT_ZNCC_BANDS
	// compute field gradient
	//DEC_BitMatrix(localNorm);
	//jpb wip bug set this here
	//localNorm.memset(0);
#if MESHOPT_PROFILE
	MeshProf::Timer _tPG;
#endif
	ComputePhotometricGradient(viewA, g_cameraData[idxImageA], faceNormals, cameraA, cameraB, viewB, imageDZNCC, mask, threadGrad, threadNorm, RegularizationScale, faceSetupEpoch, tileEnergyLocal[idxImageA]);
#if MESHOPT_PROFILE
	MeshProf::AddTL(MeshProf::P_PhotoGrad, _tPG.ms());
#endif
#endif

#if 0
	// JPB WIP BUG Lock l(cs);
	if (vertexDepth.IsEmpty()) {
		FOREACH(i, photoGrad) {
			if (_photoGradNorm[i] > 0) {
				AtomicAddFloat(&photoGrad[i][0], _photoGrad[i][0]);
				AtomicAddFloat(&photoGrad[i][1], _photoGrad[i][1]);
				AtomicAddFloat(&photoGrad[i][2], _photoGrad[i][2]);
				AtomicAddFloat(&photoGradNorm[i], 1.f);
			}
		}
	}
	else {
		Lock l(cs);
		const float depth(MINF(imageDataA.avgDepth, imageDataB.avgDepth));
		FOREACH(i, photoGrad) {
			if (_photoGradNorm[i] > 0) {
				photoGrad[i] += _photoGrad[i];
				photoGradNorm[i] += 1.f;
				if (vertexDepth[i] > depth)
					vertexDepth[i] = depth;
			}
		}
	}
#endif

#ifdef MESHOPT_CERES
	AtomicAddFloat(&scorePhoto, (float)RegularizationScale * score);
#endif
}
void MeshRefine::ThSmoothVertices1(VIndex idxStart, VIndex idxEnd)
{
#ifdef MESHOPT_CERES
	const float score(ComputeSmoothnessGradient1(vertices, vertexVertices, vertexBoundary, smoothGrad1, idxStart, idxEnd));
	Lock l(cs);
	scoreSmooth += score;
#else
	ComputeSmoothnessGradient1(vertices, vertexVertices, vertexBoundary, smoothGrad1, idxStart, idxEnd);
#endif
}
void MeshRefine::ThSmoothVertices2(VIndex idxStart, VIndex idxEnd)
{
	ComputeSmoothnessGradient2(smoothGrad1, vertexVertices, vertexBoundary, smoothGrad2, idxStart, idxEnd);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

#ifdef MESHOPT_CERES

#pragma push_macro("LOG")
#undef LOG
#pragma push_macro("CHECK")
#undef CHECK
#pragma push_macro("ERROR")
#undef ERROR
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#pragma pop_macro("ERROR")
#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")

namespace ceres {
	class MeshProblem : public FirstOrderFunction, public IterationCallback
	{
	public:
		MeshProblem(MeshRefine& _refine) : refine(_refine), params(refine.vertices.GetSize() * 3) {
			// init params
			FOREACH(i, refine.vertices)
				* ((Point3d*)params.Begin() + i) = refine.vertices[i];
		}
		virtual ~MeshProblem() {}

		void ApplyParams() const {
			FOREACH(i, refine.vertices)
				refine.vertices[i] = *((Point3d*)params.Begin() + i);
		}
		void ApplyParams(const double* parameters) const {
			memcpy(params.Begin(), parameters, sizeof(double) * params.GetSize());
			ApplyParams();
		}

		bool Evaluate(const double* const parameters, double* cost, double* gradient) const {
			// update surface parameters
			ApplyParams(parameters);
#if MESHOPT_VISIBILITY_REUSE
			// displacement unknown on this path -> force an exact recull
			refine.OnVerticesDisplaced(FLT_MAX);
#endif
			// evaluate residuals and gradients
			Point3dArr gradients;
			if (!gradient) {
				gradients.Resize(refine.vertices.GetSize());
				gradient = (double*)gradients.Begin();
			}
			*cost = refine.ScoreMesh(gradient);
			return true;
		}

		CallbackReturnType operator()(const IterationSummary& summary) {
			refine.iteration = summary.iteration;
			return ceres::SOLVER_CONTINUE;
		}

		int NumParameters() const { return (int)params.GetSize(); }
		const double* GetParameters() const { return params.Begin(); }
		double* GetParameters() { return params.Begin(); }

	protected:
		MeshRefine& refine;
		DoubleArr params;
	};
} // namespace ceres

#endif // MESHOPT_CERES

#if MESHOPT_ITER_DIAGNOSTICS
// dump mesh with per-vertex pair-support colors: red=0, orange=1, yellow=2, gray>=3
static void SavePairSupportPLY(const String& fileName, const Mesh& mesh, const FloatArr& support)
{
	if (mesh.vertices.GetSize() != support.GetSize())
		return;
	FILE* f = fopen(fileName, "wb");
	if (!f)
		return;
	fprintf(f, "ply\nformat binary_little_endian 1.0\n"
		"element vertex %u\n"
		"property float x\nproperty float y\nproperty float z\n"
		"property uchar red\nproperty uchar green\nproperty uchar blue\n"
		"element face %u\n"
		"property list uchar int vertex_indices\n"
		"end_header\n", mesh.vertices.GetSize(), mesh.faces.GetSize());
	FOREACH(v, mesh.vertices) {
		const Mesh::Vertex& vert = mesh.vertices[v];
		const float n = support[v];
		uint8_t rgb[3];
		if (n <= 0.f)      { rgb[0]=255; rgb[1]=0;   rgb[2]=0;   }
		else if (n <= 1.f) { rgb[0]=255; rgb[1]=128; rgb[2]=0;   }
		else if (n <= 2.f) { rgb[0]=255; rgb[1]=255; rgb[2]=0;   }
		else               { rgb[0]=190; rgb[1]=190; rgb[2]=190; }
		fwrite(&vert, sizeof(float), 3, f);
		fwrite(rgb, 1, 3, f);
	}
	FOREACH(fi, mesh.faces) {
		const uint8_t cnt = 3;
		fwrite(&cnt, 1, 1, f);
		fwrite(&mesh.faces[fi], sizeof(uint32_t), 3, f);
	}
	fclose(f);
}
#endif // MESHOPT_ITER_DIAGNOSTICS


#ifdef _USE_CUDA
// Everything from here to the matching #endif exists only to decide between the CPU
// and the CUDA refinement path, so it is compiled only where both paths exist.

// Host CPU base clock in MHz, or 0 when this platform cannot report it (macOS,
// and any Windows box whose registry does not carry ~MHz). Deliberately the BASE
// clock, not the current one: the current clock depends on load and on whatever
// the governor is doing right now, and the caller needs an answer that is the
// same on every run of the same machine.
static unsigned GetHostBaseClockMHz()
{
#ifdef _MSC_VER
	HKEY key;
	if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, _T("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
		return 0;
	DWORD mhz(0), type(0), size((DWORD)sizeof(mhz));
	const bool bValid(RegQueryValueEx(key, _T("~MHz"), NULL, &type, (LPBYTE)&mhz, &size) == ERROR_SUCCESS && type == REG_DWORD);
	RegCloseKey(key);
	return bValid ? (unsigned)mhz : 0u;
#elif defined(__linux__)
	// cpufreq reports the maximum clock in kHz, which is stable; /proc/cpuinfo's
	// "cpu MHz" is only whatever the core happened to be running at, so it is the
	// fallback rather than the first choice
	if (FILE* f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r")) {
		unsigned long kHz(0);
		const int n(fscanf(f, "%lu", &kHz));
		fclose(f);
		if (n == 1 && kHz > 0)
			return (unsigned)(kHz/1000);
	}
	if (FILE* f = fopen("/proc/cpuinfo", "r")) {
		char line[256];
		double best(0);
		while (fgets(line, sizeof(line), f) != NULL) {
			double mhz;
			if (sscanf(line, "cpu MHz : %lf", &mhz) == 1 && mhz > best)
				best = mhz;
		}
		fclose(f);
		if (best > 0)
			return (unsigned)best;
	}
	return 0;
#else
	return 0;
#endif
}

// CPU-vs-GPU path selection for mesh refinement; see Scene::PreferCPUMeshRefinement.
//
// Minimum total physical RAM [GB] for the CPU path to be preferred. Below this the
// CPU path spends its time batching and evicting images instead of computing (see
// ResolveRefineMeshSafeSettings below for the floor it cannot batch its way out of),
// which is one of the two regimes where the GPU path wins outright. Not scaled by
// the GPU: a memory-starved host loses to any GPU that can hold the scene.
#ifndef MESHOPT_CPU_PATH_MIN_RAM_GB
#define MESHOPT_CPU_PATH_MIN_RAM_GB 48.0
#endif
// Absolute floor on usable threads (after --max-threads), NOT a comparison against
// the GPU: below a handful of cores the CPU path is dominated by its own per-batch
// overhead whatever the clock says.
#ifndef MESHOPT_CPU_PATH_MIN_THREADS
#define MESHOPT_CPU_PATH_MIN_THREADS 8
#endif
// CPU throughput proxy needed to beat a REFERENCE-CLASS GPU, in GHz-threads (usable
// threads x base clock). Derived from the measurement this whole check exists to
// encode: on Richmond Historic, RefineMesh took 44.3 s on a fast desktop CPU
// (13900K class, ~32 threads x 3.0 GHz base = 96 GHz-threads) against 82.6 s on an
// RTX 3060 12 GB -- the CPU was 1.86x faster, so the crossover against that GPU
// sits at 96 / 1.86 = ~52 GHz-threads. Scaled by the actual device's strength below.
#ifndef MESHOPT_CPU_PATH_MIN_GHZ_THREADS
#define MESHOPT_CPU_PATH_MIN_GHZ_THREADS 52.0
#endif
// The reference GPU those 82.6 s were measured on: an RTX 3060 12 GB, i.e.
// 28 SMs x 128 cores x 1.777 GHz = ~6370 GHz-cores, 192-bit at 15 Gbps = 360 GB/s.
// A device stronger than this raises the bar the CPU has to clear, and vice versa.
#ifndef MESHOPT_GPU_REF_GHZ_CORES
#define MESHOPT_GPU_REF_GHZ_CORES 6370.0
#endif
#ifndef MESHOPT_GPU_REF_BANDWIDTH_GBS
#define MESHOPT_GPU_REF_BANDWIDTH_GBS 360.0
#endif

// Answers "should mesh refinement stay on the CPU even though a CUDA device is
// available?" -- see the declaration in Scene.h for the measurement this encodes.
//
// The comparison is GPU-RELATIVE. The 44.3 s / 82.6 s data point above fixes one
// point on the curve (a fast desktop CPU against a 3060); anything else is placed
// by scoring the device actually installed and moving the CPU bar by the same
// factor. That device score is the geometric mean of two ratios against the
// reference, FP32 throughput and memory bandwidth, because neither alone tracks
// this kernel: pure FLOPS would credit an Ada card with a ~6x speedup its memory
// system cannot feed, and pure bandwidth would ignore the arithmetic entirely.
//
// Keyed on STATIC attributes only: total physical RAM, usable thread count, base
// clock, AVX2, and the device's own advertised capability. Never on live free RAM,
// current clock or a timing micro-benchmark. Which path runs is output-affecting
// (the CPU and CUDA paths do not produce bit-identical meshes), and an
// output-affecting decision that moved with whatever else happens to be running
// would make the identical scene on the identical machine irreproducible -- the
// same rule ResolveRefineMeshSafeSettings below follows.
//
// The metric is coarse: GHz-threads scores a 13900K at 96 and a 7950X at 144
// although the two refine at similar speed, so the derived bar is only good to
// within ~1.5x. That is tolerable precisely BECAUSE it is a crossover: a host
// landing near the bar performs about the same either way, by definition, so a
// wrong call there costs little. It is the far-from-the-bar cases -- a fast desktop
// against a mid-range card, a laptop against the same card -- that this has to get
// right, and those it gets right by a wide margin. Every number is logged, and all
// five thresholds above are overridable, so a machine that contradicts the model
// can be pinned with --cuda-policy instead of argued with.
//
// Any input this platform cannot report (clock, RAM, device capability) simply stops
// gating: an unreadable value is not evidence that the host is slow.
bool Scene::PreferCPUMeshRefinement(unsigned nResolutionLevel, unsigned nMinResolution) const
{
	// Every branch below logs the reason it decided, because the caller cannot know
	// it and must not paraphrase it: "the GPU lost on speed", "the GPU cannot hold
	// the scene" and "there is no GPU path on this driver" are three different
	// answers that all return true here.
	//
	// No usable GPU path at all on a CUDA-12+ driver (the refine kernels are on the
	// legacy driver API -- see CUDA::HasLegacyDriverAPI), so there is nothing to
	// weigh. Answering early keeps the caller from snapshotting the mesh and
	// entering RefineMeshCUDA only to be refused there.
	if (!SEACAVE::CUDA::HasLegacyDriverAPI()) {
		LOG(_T("Mesh refinement device check: this driver does not export the legacy CUDA entry points the GPU ")
			_T("refinement kernels are built on (removed in CUDA 12.0); there is no GPU path here -- refining on the CPU"));
		return true;
	}

	// Hard gate, ahead of any speed comparison: a device that cannot hold the finest
	// scale is not a candidate however fast it is. Without this the run would reach
	// the last scale, refuse itself in ResolveResidency(), and hand a discarded GPU
	// run's worth of time back to the CPU path (see EstimateRefineMeshCUDAVRAM).
	uint64_t needBytes(0), budgetBytes(0);
	if (EstimateRefineMeshCUDAVRAM(nResolutionLevel, nMinResolution, needBytes, budgetBytes) && needBytes > budgetBytes) {
		LOG(_T("Mesh refinement device check: the finest scale needs about %s of device memory but this GPU ")
			_T("offers only %s; refining on the CPU"),
			Util::formatBytes(needBytes).c_str(), Util::formatBytes(budgetBytes).c_str());
		return true;
	}
	constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
	const uint64_t totalPhys = (uint64_t)Util::GetMemoryInfo().totalPhysical;
	const unsigned threads = nMaxThreads > 0 ? nMaxThreads : Thread::hardwareConcurrency();
	const unsigned mhz = GetHostBaseClockMHz();
	// the CPU path's inner loops (the ZNCC score and the image warp) have AVX2
	// kernels; without them it falls back to SSE2 and loses the comparison
	const bool bAVX2 = SupportsAVX2();
	const double ghzThreads = mhz > 0 ? threads * (mhz / 1000.0) : 0.0;

	// how much stronger (or weaker) the installed device is than the RTX 3060 the
	// crossover was measured against
	double gpuFactor = 1.0;
	String gpuDesc(_T("unknown device, assuming reference class"));
	SEACAVE::CUDA::DeviceCapability cap;
	if (SEACAVE::CUDA::GetDeviceCapability(SEACAVE::CUDA::desiredDeviceID, cap) && cap.ComputeScore() > 0) {
		const double compute = cap.ComputeScore();
		const double bandwidth = cap.BandwidthGBs();
		const double computeRatio = compute / MESHOPT_GPU_REF_GHZ_CORES;
		if (bandwidth > 0)
			gpuFactor = SQRT(computeRatio * (bandwidth / MESHOPT_GPU_REF_BANDWIDTH_GBS));
		else
			gpuFactor = computeRatio; // no bandwidth to blend in; compute alone it is
		gpuDesc = String::FormatString(_T("%s: %.0f GHz-cores, %.0f GB/s, %s VRAM -> %.2fx the reference RTX 3060"),
			cap.name, compute, bandwidth, Util::formatBytes(cap.totalMem).c_str(), gpuFactor);
	}
	const double reqGhzThreads = MESHOPT_CPU_PATH_MIN_GHZ_THREADS * gpuFactor;

	const bool bEnoughRAM = totalPhys == 0 || (double)totalPhys / GB >= MESHOPT_CPU_PATH_MIN_RAM_GB;
	const bool bEnoughThreads = threads >= (unsigned)MESHOPT_CPU_PATH_MIN_THREADS;
	const bool bFastEnough = mhz == 0 || ghzThreads >= reqGhzThreads;
	const bool bPreferCPU = bEnoughRAM && bEnoughThreads && bAVX2 && bFastEnough;

	LOG(_T("Mesh refinement device check: GPU %s"), gpuDesc.c_str());
	LOG(_T("Mesh refinement host check: %.1f GB RAM, %u usable threads, %s base clock, %s, score %s -> %s path ")
		_T("(CPU path needs >= %.0f GB, >= %u threads, AVX2, and >= %.0f GHz-threads against this device)"),
		totalPhys / (double)GB, threads,
		mhz > 0 ? String::FormatString(_T("%.2f GHz"), mhz / 1000.0).c_str() : _T("unknown"),
		bAVX2 ? _T("AVX2") : _T("no AVX2"),
		mhz > 0 ? String::FormatString(_T("%.0f GHz-threads"), ghzThreads).c_str() : _T("unknown (clock unavailable, not gated on it)"),
		bPreferCPU ? _T("CPU") : _T("GPU"),
		(double)MESHOPT_CPU_PATH_MIN_RAM_GB, (unsigned)MESHOPT_CPU_PATH_MIN_THREADS, reqGhzThreads);
	return bPreferCPU;
}
#endif // _USE_CUDA


// Pre-flight, deterministic memory-safety check for RefineMesh, called once before
// any image is decoded or MeshRefine is constructed.
//
// The batching/streaming machinery inside MeshRefine (BuildViewBatches et al.) can
// only ever trade SPEED for memory: it shrinks the resident working set by doing
// more passes, but the result is unchanged (see kStreamBytesPerPixel and friends).
// There is exactly one thing it cannot do anything about: the "neighbourhood
// floor" -- one reference view plus its nMaxViews neighbours, at the configured
// resolution, is the smallest unit that can ever be resident at once. If THAT
// alone does not fit under this machine's target, no amount of batching helps and
// the run either overshoots into pagefile thrash or hits an out-of-memory abort.
//
// This function is the deterministic last resort for that specific case: it
// estimates the floor from scene metadata alone (image count/resolution and mesh
// face count -- nothing here decodes a pixel), and if the floor does not fit even
// in the best case, reduces nMaxViews (cheaper in quality than losing image
// detail) and only then, as a further last resort, raises nResolutionLevel by the
// minimum amount needed. Both are applied automatically (there is no user to ask
// in an unattended pipeline) but never silently: any change is logged.
//
// Deliberately keyed on this machine's TOTAL physical RAM, never on live free RAM.
// Free RAM fluctuates with whatever else happens to be running at launch, and an
// output-affecting decision that depended on that would make the result non-
// reproducible for the identical scene on the identical machine -- exactly what
// the neutral/affecting split elsewhere in this file exists to avoid. Live free
// RAM remains fine for the SPEED-only decisions inside MeshRefine's own budget.
//
// No-op, and no measurable cost, on any machine where the scene already fits at
// the requested settings -- true for the overwhelming majority of runs on an
// adequately sized box, which is exactly the case this function must not slow down.
void Scene::ResolveRefineMeshSafeSettings(unsigned& nResolutionLevel, unsigned nMinResolution, unsigned& nMaxViews) const
{
	if (images.IsEmpty() || nMaxViews == 0)
		return;
	constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
	const Util::MemoryInfo mi(Util::GetMemoryInfo());
	const uint64_t totalPhys = (uint64_t)mi.totalPhysical;
	if (totalPhys == 0)
		return; // can't size anything without knowing the machine; leave settings untouched
	const uint64_t target = ResolveTargetBytesFromTotal(totalPhys);

	// The largest image in the scene drives the worst-case neighbourhood (a batch
	// pairing two big images costs the most), at full (level-0) resolution.
	uint64_t maxPixels0 = 0;
	for (IIndex i = 0; i < images.GetSize(); ++i)
		maxPixels0 = std::max(maxPixels0, (uint64_t)images[i].width * (uint64_t)images[i].height);
	if (maxPixels0 == 0)
		return; // width/height not populated (e.g. images not yet matched); nothing to estimate from

	// Rough pinned baseline: the mesh (vertices/faces/normals/faceFaces/
	// vertexVertices/vertexFaces all populated costs roughly 100-150 B per face,
	// see the Mesh memory audit) plus a fixed process floor. Deliberately
	// conservative -- this is a pre-flight GATE meant to catch the clearly-unsafe
	// case, not the runtime budget, which measures actual commit for real.
	const uint64_t pinnedEstimate = (uint64_t)mesh.faces.GetSize() * 150ull + GB;

	const unsigned kMinViewsFloor = 3; // below this, neighbour coverage degrades badly
	const unsigned kMaxLevelBump = 4;  // refuse to chase a pathological scene forever

	const unsigned requestedViews = nMaxViews;
	const unsigned requestedLevel = nResolutionLevel;
	unsigned level = requestedLevel;
	unsigned views = requestedViews;
	for (unsigned bump = 0; ; ++bump, ++level) {
		// Pixels at this level: each level halves width and height (~4x fewer pixels).
		uint64_t px = maxPixels0;
		for (unsigned l = 0; l < level; ++l)
			px = (px + 3) / 4;
		views = requestedViews;
		for (;;) {
			const uint64_t floorBytes = (uint64_t)(1 + views) * kStreamBytesPerPixel * px;
			const uint64_t ceiling = (pinnedEstimate < target) ? (target - pinnedEstimate) : 0;
			if (floorBytes <= ceiling || views <= kMinViewsFloor)
				break;
			--views;
		}
		const uint64_t floorBytes = (uint64_t)(1 + views) * kStreamBytesPerPixel * px;
		const uint64_t ceiling = (pinnedEstimate < target) ? (target - pinnedEstimate) : 0;
		if (floorBytes <= ceiling) {
			if (level != requestedLevel || views != requestedViews) {
				LOG(_T("Pre-flight memory check: this machine (%.1f GB RAM) can not fit the requested ")
					_T("settings (max-views %u, resolution-level %u) for a scene this size; automatically ")
					_T("using max-views %u, resolution-level %u instead (last-resort, quality-affecting ")
					_T("fallback -- every memory-neutral option was assumed exhausted first)"),
					totalPhys / (double)GB, requestedViews, requestedLevel, views, level);
				nMaxViews = views;
				nResolutionLevel = level;
			}
			return;
		}
		if (bump >= kMaxLevelBump)
			break;
	}
	// Even the minimum settings tried did not clear the estimate: apply the finest
	// bump / fewest views we tried anyway and let the runtime batching/eviction
	// system and the PlaneAlloc safety net take it from here -- this scene is
	// genuinely at or past what this machine can hold at full quality.
	LOG(_T("Pre-flight memory check: this scene may exceed what this machine (%.1f GB RAM) can hold even at ")
		_T("the most conservative settings tried (max-views %u, resolution-level %u); proceeding anyway -- ")
		_T("expect heavy batching, or an out-of-memory abort if this estimate is still short"),
		totalPhys / (double)GB, views, level);
	nMaxViews = views;
	nResolutionLevel = level;
}

// optimize mesh using photo-consistency
// fThPlanarVertex - threshold used to remove vertices on planar patches (percentage of the minimum depth, 0 - disable)
bool Scene::RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews,
	float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea,
	unsigned nScales, float fScaleStep,
	unsigned nReduceMemory, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fThPlanarVertex, float fGradientStep)
{
	// TESTING NOP: define MESHOPT_SKIP_REFINE=1 (compile-time) to skip refinement
	// entirely and leave the mesh untouched (no decimation/subdivision/gradient
	// steps). The caller still saves its output file, so you get the un-refined
	// mesh in the same output path. Set to 0 for normal refinement.
#ifndef MESHOPT_SKIP_REFINE
#define MESHOPT_SKIP_REFINE 0  // <-- set to 1 to skip refinement (NOP) for testing
#endif
#if MESHOPT_SKIP_REFINE
	DEBUG_EXTRA("Mesh refinement skipped (MESHOPT_SKIP_REFINE=1)");
	return true;
#endif

	cv::setNumThreads(0);// disable OpenCV internal threading to eliminate oversubscription with our threading

	if (pointcloud.IsEmpty() && !ImagesHaveNeighbors())
		SampleMeshWithVisibility();

	MeshRefine refine(*this, nReduceMemory, nAlternatePair, fRegularityWeight, fRatioRigidityElasticity, nResolutionLevel, nMinResolution, nMaxViews, nMaxThreads);
	if (!refine.IsValid())
		return false;
#if MESHOPT_MEM_DIAG
	const auto memDiagT0 = std::chrono::steady_clock::now();
#endif

	// run the mesh optimization on multiple scales (coarse to fine)
	for (unsigned nScale = 0; nScale < nScales; ++nScale) {
#if MESHOPT_MEM_DIAG
		{
			const Util::MemoryInfo mi = Util::GetMemoryInfo();
			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - memDiagT0).count();
			DEBUG_EXTRA("[MEM] scale %u/%u start: freePhysical=%zuMB / totalPhysical=%zuMB  elapsed=%.1fs",
				nScale + 1, nScales, mi.freePhysical / 1024 / 1024, mi.totalPhysical / 1024 / 1024, elapsed);
			// process-level breakdown (PeakWorkingSetSize/PagefileUsage), unlike the system-wide line above
			Util::LogMemoryInfo();
		}
#endif
#if MESHOPT_PROFILE
		MeshProf::ResetScale();
#endif
		// init images
		const Real scale(POWI(fScaleStep, nScales - nScale - 1));
		const Real step(POWI(2.f, nScales - nScale));
		DEBUG_ULTIMATE("Refine mesh at: %.2f image scale", scale);
		Real initScale = scale;
#if MESHOPT_REFINE_HIRES_FINAL
		// Option A: only the finest scale loads one resolution level finer.
		refine.resolutionLevelOverride =
			(nScale + 1 == nScales && nResolutionLevel > 0) ? nResolutionLevel - 1 : unsigned(-1);
		// bound the ~4x memory spike by downsampling the finer image back toward
		// base (see MESHOPT_REFINE_HIRES_SCALE). Only when the override engaged.
		if (refine.resolutionLevelOverride != unsigned(-1))
			initScale = scale * Real(MESHOPT_REFINE_HIRES_SCALE);
#endif
#if MESHOPT_PROFILE
		MeshProf::Timer _tII;
#endif
		if (!refine.InitImages(initScale, Real(0.12) * step + Real(0.2)))
			return false;
#if MESHOPT_PROFILE
		MeshProf::Add(MeshProf::P_InitImages, _tII.ms());
#endif

		// extract array of triangles incident to each vertex
#if MESHOPT_PROFILE
		MeshProf::Timer _tLVPre;
#endif
		refine.ListVertexFacesPre();
#if MESHOPT_PROFILE
		MeshProf::Add(MeshProf::P_ListVtxPre, _tLVPre.ms());
#endif

		// automatic mesh subdivision
		unsigned effMaxFaceArea = nMaxFaceArea;
#if MESHOPT_REFINE_HIRES_FINAL && MESHOPT_REFINE_HIRES_KEEPFACES
		// Option A raises the final-scale resolution, which makes faces project to
		// more pixels and roughly (2*HIRES_SCALE)^2 x more of them subdivide. Scale
		// the face-area budget by that ratio on the final scale to keep the output
		// face count near the base-resolution result.
		if (nScale + 1 == nScales && nResolutionLevel > 0 && nMaxFaceArea > 0) {
			const float areaRatio = (2.0f * MESHOPT_REFINE_HIRES_SCALE) * (2.0f * MESHOPT_REFINE_HIRES_SCALE);
			effMaxFaceArea = (unsigned)(nMaxFaceArea * areaRatio + 0.5f);
		}
#endif
#if MESHOPT_PROFILE
		MeshProf::Timer _tSub;
#endif
		refine.SubdivideMesh(effMaxFaceArea, nScale == 0 ? fDecimateMesh : 1.f, nCloseHoles, nEnsureEdgeSize);
#if MESHOPT_PROFILE
		MeshProf::Add(MeshProf::P_Subdivide, _tSub.ms());
		MeshProf::Timer _tLVPost;
#endif

		// extract array of triangle normals
		refine.ListVertexFacesPost();
#if MESHOPT_PROFILE
		MeshProf::Add(MeshProf::P_ListVtxPost, _tLVPost.ms());
#endif

#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefine%u.ply", nScales - nScale - 1)));
#endif

		// minimize
#ifdef MESHOPT_CERES
		if (fGradientStep == 0) {
			// DefineProblem
			refine.ratioRigidityElasticity = 1.f;
			ceres::MeshProblem* problemData(new ceres::MeshProblem(refine));
			ceres::GradientProblem problem(problemData);
			// SetMinimizerOptions
			ceres::GradientProblemSolver::Options options;
			if (VERBOSITY_LEVEL > 1) {
				options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
				options.minimizer_progress_to_stdout = true;
			}
			else {
				options.logging_type = ceres::LoggingType::SILENT;
				options.minimizer_progress_to_stdout = false;
			}
			options.function_tolerance = 1e-3;
			options.gradient_tolerance = 1e-7;
			options.max_num_line_search_step_size_iterations = 10;
			options.callbacks.push_back(problemData);
			ceres::GradientProblemSolver::Summary summary;
			// SolveProblem
			ceres::Solve(options, problem, problemData->GetParameters(), &summary);
			DEBUG_ULTIMATE(summary.FullReport().c_str());
			switch (summary.termination_type) {
			case ceres::TerminationType::NO_CONVERGENCE:
				DEBUG_EXTRA("CERES: maximum number of iterations reached!");
			case ceres::TerminationType::CONVERGENCE:
			case ceres::TerminationType::USER_SUCCESS:
				break;
			default:
				VERBOSE("CERES surface refine error: %s!", summary.message.c_str());
				return false;
			}
			ASSERT(summary.IsSolutionUsable());
			problemData->ApplyParams();
		}
		else
#endif // MESHOPT_CERES
		{
#if 1
			uint32_t nVerts = refine.vertices.GetSize();
			int iters;
			double gstep = 1.5;
#if MESHOPT_REFINE_QUALITY
			if (nScale == 0) iters = 12;     // coarse
			else if (nScale == 1) iters = 10;// mid
			else iters = 8;                  // fine
#else
			if (nScale == 0) iters = 8;      // coarse
			else if (nScale == 1) iters = 6; // mid
			else iters = 4;                  // fine
#endif
			if (fGradientStep > 1.f) {
				const int baseIters = FLOOR2INT(fGradientStep);
				iters = MAXF(baseIters / (int)(nScale + 1), 8);
				gstep = (fGradientStep - (float)baseIters) * 10.0;
			}
#if MESHOPT_REFINE_HIRES_FINAL
			// Option A: the final scale runs at ~4x the pixels, so cut its
			// iterations to keep the pass roughly compute-neutral.
			if (nScale + 1 == nScales)
				iters = 3;
#endif
			double decay = 0.98;
#if MESHOPT_CONVERGED_EXIT
			double cvPrevEnergy[2] = { 0.0, 0.0 }; // photoEnergy history per pair-direction parity
			int cvHits = 0;
#endif

			Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> gradients;
			gradients.resize(nVerts, 3);

			Util::Progress progress(_T("Processed iterations"), iters);
			GET_LOGCONSOLE().Pause();

			for (int iter = 0; iter < iters; ++iter)
			{
#if MESHOPT_MEM_DIAG
				{
					const Util::MemoryInfo mi = Util::GetMemoryInfo();
					const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - memDiagT0).count();
					DEBUG_EXTRA("[MEM] scale %u/%u iter %d/%d: freePhysical=%zuMB / totalPhysical=%zuMB  elapsed=%.1fs",
						nScale + 1, nScales, iter + 1, iters, mi.freePhysical / 1024 / 1024, mi.totalPhysical / 1024 / 1024, elapsed);
				}
#endif
				refine.iteration = iter;
				refine.nAlternatePair = (iter + 1 < iters ? nAlternatePair : 0);
				refine.ratioRigidityElasticity =
					(iter <= iters * 7 / 10 ? fRatioRigidityElasticity : 1.f);

				bool rebuildOctree = true;

#if MESHOPT_PROFILE
						MeshProf::Timer _tSM;
#endif
						double cost = refine.ScoreMesh(gradients.data(), rebuildOctree);
#if MESHOPT_PROFILE
						MeshProf::Add(MeshProf::P_ScoreMesh, _tSM.ms());
#endif
#if MESHOPT_ITER_DIAGNOSTICS && TD_VERBOSE != TD_VERBOSE_OFF
				// start-of-scale support: geometry not yet displaced at this scale
				if (iter == 0 && VERBOSITY_LEVEL > 2) {
					SavePairSupportPLY(MAKE_PATH(String::FormatString("MeshRefineSupport%u.ply", nScales - nScale - 1)), mesh, refine.photoGradNorm);
#if MESHOPT_SUPPORT_STAGE_DIAG
					SavePairSupportPLY(MAKE_PATH(String::FormatString("MeshRefineRaster%u.ply", nScales - nScale - 1)), mesh, refine.rasterSupport);
					SavePairSupportPLY(MAKE_PATH(String::FormatString("MeshRefineMask%u.ply", nScales - nScale - 1)), mesh, refine.maskSupport);
					{
						// vertex = MAX over incident faces: which raster stage zeroes a region
						const FloatArr* stages[4] = { &refine.faceCandViews, &refine.faceBBoxViews, &refine.faceInsideViews, &refine.faceWonViews };
						const char* stageNames[4] = { "MeshRefineFaceCand%u.ply", "MeshRefineFaceBBox%u.ply", "MeshRefineFaceInside%u.ply", "MeshRefineFaceWon%u.ply" };
						FloatArr vtxAgg;
						vtxAgg.Resize(mesh.vertices.GetSize());
						for (int k = 0; k < 4; ++k) {
							if (stages[k]->GetSize() != mesh.faces.GetSize())
								continue;
							vtxAgg.Memset(0);
							FOREACH(fIdx, mesh.faces) {
								const float n = (*stages[k])[fIdx];
								const Mesh::Face& fc = mesh.faces[fIdx];
								for (int v = 0; v < 3; ++v)
									if (vtxAgg[fc[v]] < n)
										vtxAgg[fc[v]] = n;
							}
							SavePairSupportPLY(MAKE_PATH(String::FormatString(stageNames[k], nScales - nScale - 1)), mesh, vtxAgg);
						}
						// losers among inside-tested faces: red = beaten by a >1% closer
						// surface (real occluder), orange = <1%, yellow = FP-tie only,
						// gray = won somewhere (or never had an inside pixel)
						if (refine.faceLoseMargin.GetSize() == mesh.faces.GetSize() &&
							refine.faceInsideViews.GetSize() == mesh.faces.GetSize()) {
							FOREACH(vi, vtxAgg)
								vtxAgg[vi] = 3.f;
							FOREACH(fIdx, mesh.faces) {
								if (refine.faceInsideViews[fIdx] <= 0.f)
									continue;
								float cls = 3.f;
								if (refine.faceWonViews[fIdx] <= 0.f) {
									const float m = refine.faceLoseMargin[fIdx];
									cls = m >= 0.01f ? 0.f : m > 1e-5f ? 1.f : 2.f;
								}
								const Mesh::Face& fc = mesh.faces[fIdx];
								for (int v = 0; v < 3; ++v)
									if (vtxAgg[fc[v]] > cls)
										vtxAgg[fc[v]] = cls;
							}
							SavePairSupportPLY(MAKE_PATH(String::FormatString("MeshRefineFaceLose%u.ply", nScales - nScale - 1)), mesh, vtxAgg);
						}
					}
#endif
				}
#endif
				// convergence trace comparable across gate configs and with the CUDA
				// path's per-iteration log (cost is 0 unless built with Ceres)
				DEBUG_EXTRA("\t%2d. f: %.5f (%.4e)\tg: %.5f (%.4e)\ts: %.3f", iter + 1,
					cost, cost / (double)nVerts, gradients.norm(), gradients.norm() / (double)nVerts, gstep);
				// Gradient update
				// -------------------------------------------------------------
				const float step = float(gstep);
				float gradClipThresh = 0.f;
				if (MESHOPT_GRAD_CLIP_K > 0.f) {
					double activeGradSum = 0.0;
					int activeGradCount = 0;
#pragma omp parallel for schedule(static) reduction(+:activeGradSum,activeGradCount)
					for (int v = 0; v < (int)nVerts; ++v) {
						const float gx = gradients(v, 0);
						const float gy = gradients(v, 1);
						const float gz = gradients(v, 2);
						const double mag = FastSqrtD(double(gx * gx + gy * gy + gz * gz));
						if (mag > 0.0 && ISFINITE(mag)) {
							activeGradSum += mag;
							++activeGradCount;
						}
					}
					if (activeGradCount > 0)
						gradClipThresh = MESHOPT_GRAD_CLIP_K * float(activeGradSum / activeGradCount);
				}
#if MESHOPT_PROFILE
				MeshProf::Timer _tMom;
#endif
#if MESHOPT_VISIBILITY_REUSE
				float iterMaxDispSq = 0.f;
#endif
#pragma omp parallel
				{
#if MESHOPT_VISIBILITY_REUSE
					float localMaxDispSq = 0.f;
#endif
#pragma omp for schedule(static) nowait
					for (int v = 0; v < (int)nVerts; v++) {
						float gx = gradients(v, 0);
						float gy = gradients(v, 1);
						float gz = gradients(v, 2);
						if (!ISFINITE(gx) || !ISFINITE(gy) || !ISFINITE(gz))
							continue;
						if (gradClipThresh > 0.f) {
							const float mag = float(FastSqrtD(double(gx * gx + gy * gy + gz * gz)));
							if (mag > gradClipThresh) {
								const float scale = gradClipThresh / mag;
								gx *= scale;
								gy *= scale;
								gz *= scale;
							}
						}

						const float dx = gx * step;
						const float dy = gy * step;
						const float dz = gz * step;
#if MESHOPT_VISIBILITY_REUSE
						// track the applied displacement so the padded-frustum cull
						// can prove candidate reuse stays exact
						const float dispSq = dx * dx + dy * dy + dz * dz;
						if (dispSq > localMaxDispSq)
							localMaxDispSq = dispSq;
#endif
						Vertex& vert = refine.vertices[v];
						vert.x -= dx;
						vert.y -= dy;
						vert.z -= dz;
					}
#if MESHOPT_VISIBILITY_REUSE
#pragma omp critical
					{
						if (localMaxDispSq > iterMaxDispSq)
							iterMaxDispSq = localMaxDispSq;
					}
#endif
				}
#if MESHOPT_VISIBILITY_REUSE
				refine.OnVerticesDisplaced(sqrtf(iterMaxDispSq));
#endif
#if MESHOPT_PROFILE
				MeshProf::Add(MeshProf::P_Momentum, _tMom.ms());
#endif


#if !MESHOPT_DISABLE_EARLY_EXIT || MESHOPT_ITER_DIAGNOSTICS
				double avgGrad = gradients.norm() / double(nVerts);

#if MESHOPT_ITER_DIAGNOSTICS
				DEBUG_EXTRA("Iter %d avgGrad = %.6e", iter, avgGrad);
#endif

				// =============================================================
				// DIAGNOSTICS BLOCK INSERTED HERE
				// =============================================================

				// 1. Max per-vertex displacement
// 1. Max per-vertex displacement (manual reduction)
#if MESHOPT_PROFILE
				MeshProf::Timer _tMD;
#endif
				double maxDisp = 0.0;
				{
					int numThreads = 1;
#ifdef _OPENMP
					numThreads = omp_get_max_threads();
#endif
					std::vector<double> localMax(numThreads, 0.0);

#pragma omp parallel
					{
						int tid = 0;
#ifdef _OPENMP
						tid = omp_get_thread_num();
#endif
						double lm = 0.0;

#pragma omp for nowait
						for (int v = 0; v < (int)nVerts; v++) {
							float dx = gradients(v, 0) * step;
							float dy = gradients(v, 1) * step;
							float dz = gradients(v, 2) * step;

							double disp = FastSqrtD(double(dx * dx + dy * dy + dz * dz));
							if (disp > lm)
								lm = disp;
						}

						localMax[tid] = lm;
					}

					for (size_t i = 0; i < localMax.size(); i++)
						if (localMax[i] > maxDisp)
							maxDisp = localMax[i];
				}
#if MESHOPT_PROFILE
				MeshProf::Add(MeshProf::P_MaxDisp, _tMD.ms());
#endif
#if MESHOPT_ITER_DIAGNOSTICS
				// outlier profile: is the stalled maxDisp a few low-support vertices?
				if (maxDisp > 0.0) {
					VIndex argMax = 0; double bestD = -1.0;
					int nHalf = 0, nTenth = 0;
					for (int v = 0; v < (int)nVerts; ++v) {
						const float dx = gradients(v, 0) * step;
						const float dy = gradients(v, 1) * step;
						const float dz = gradients(v, 2) * step;
						const double disp = FastSqrtD(double(dx * dx + dy * dy + dz * dz));
						if (disp > bestD) { bestD = disp; argMax = (VIndex)v; }
						if (disp > 0.5 * maxDisp) ++nHalf;
						if (disp > 0.1 * maxDisp) ++nTenth;
					}
					DEBUG_EXTRA("Iter %d outliers: maxDisp vertex %u (pairs=%g, boundary=%d)  disp>50%%max: %d  disp>10%%max: %d",
						iter, argMax, refine.photoGradNorm[argMax], (int)refine.vertexBoundary[argMax], nHalf, nTenth);
				}
				{
					// pair-support histogram: low support everywhere = broken coverage/mask
					size_t h[7] = {}; uint64_t sumPairs = 0;
					for (int v = 0; v < (int)nVerts; ++v) {
						const float n = refine.photoGradNorm[v];
						sumPairs += (uint64_t)n;
						const int b = n <= 0.f ? 0 : n <= 1.f ? 1 : n <= 2.f ? 2 : n <= 4.f ? 3 : n <= 8.f ? 4 : n <= 16.f ? 5 : 6;
						++h[b];
					}
					DEBUG_EXTRA("Iter %d pair support: 0:%zu 1:%zu 2:%zu 3-4:%zu 5-8:%zu 9-16:%zu >16:%zu  avg=%.2f",
						iter, h[0], h[1], h[2], h[3], h[4], h[5], h[6], nVerts ? (double)sumPairs / nVerts : 0.0);
				}
#endif
#endif // !MESHOPT_DISABLE_EARLY_EXIT || MESHOPT_ITER_DIAGNOSTICS

				// 2. Active tiles
#if MESHOPT_NEEDS_TILE_ENERGY
				int totalTiles = 0;
				int activeTiles = 0;

				for (size_t vi = 0; vi < refine.views.size(); vi++) {
					const MeshRefine::View& vw = refine.views[vi];
					int tiles = (int)vw.tileActive.size();
					totalTiles += tiles;

					for (int t = 0; t < tiles; t++)
						activeTiles += (vw.tileActive[t] ? 1 : 0);
				}

				double activeRatio = (totalTiles > 0)
					? double(activeTiles) / double(totalTiles)
					: 0.0;
#endif

#if MESHOPT_ITER_DIAGNOSTICS
				// 3. Photometric energy sum
				double photoEnergy = refine.photoEnergyLast;

				// 4. Smooth/photo gradient ratio (manual reduction)
				double photoGradSum = 0.0;
				double smoothGradSum = 0.0;

				{
					int numThreads = 1;
#ifdef _OPENMP
					numThreads = omp_get_max_threads();
#endif

					std::vector<double> localPhoto(numThreads, 0.0);
					std::vector<double> localSmooth(numThreads, 0.0);

#pragma omp parallel
					{
						int tid = 0;
#ifdef _OPENMP
						tid = omp_get_thread_num();
#endif
						double lp = 0.0;
						double ls = 0.0;

#pragma omp for nowait
						for (int v = 0; v < (int)nVerts; v++) {
							float gx = gradients(v, 0);
							float gy = gradients(v, 1);
							float gz = gradients(v, 2);
							lp += sqrt(double(gx * gx + gy * gy + gz * gz));

							const auto& sg = refine.smoothGrad2[v];
							ls += FastSqrtD(double(sg.x * sg.x + sg.y * sg.y + sg.z * sg.z));
						}

						localPhoto[tid] = lp;
						localSmooth[tid] = ls;
					}

					for (size_t i = 0; i < localPhoto.size(); i++)
						photoGradSum += localPhoto[i];

					for (size_t i = 0; i < localSmooth.size(); i++)
						smoothGradSum += localSmooth[i];
				}

				double smoothPhotoRatio =
					(photoGradSum > 0.0 ? smoothGradSum / photoGradSum : 0.0);

				// 5. Print diagnostics
				DEBUG_EXTRA(
					"Iter %d summary: avgGrad=%.6e  maxDisp=%.6e  activeTiles=%d/%d (%.2f%%)  photoEnergy=%.6e  smooth/photo=%.3f",
					iter,
					avgGrad,
					maxDisp,
					activeTiles,
					totalTiles,
					activeRatio * 100.0,
					photoEnergy,
					smoothPhotoRatio
				);
#endif // MESHOPT_ITER_DIAGNOSTICS

				// =============================================================
				// END DIAGNOSTICS BLOCK
				// =============================================================

				// Optional safety exit (rare)
#if !MESHOPT_DISABLE_EARLY_EXIT
				if (iter > 0 && avgGrad < 1e-9) {
					DEBUG_EXTRA("Early exit: avgGrad extremely small (%.6e)", avgGrad);
					break;
				}

#if MESHOPT_REFINE_QUALITY
				if (
					iter > 4 &&
					activeRatio < 1e-4 &&
					maxDisp < 0.1 * scale
					) {
					break;
				}
#else
				if (
					iter > 2 &&
					activeRatio < 5e-4 &&
					maxDisp < 0.3 * scale
					) {
					break;
				}
#endif
#endif

#if MESHOPT_CONVERGED_EXIT
				// same-parity photo-energy plateau => the photometric term is done;
				// jump so the next iteration is the scheduled final both-directions one
				if (iter + 2 < iters) {
					const int par = iter & 1;
					const double prev = cvPrevEnergy[par];
					cvPrevEnergy[par] = refine.photoEnergyLast;
					if (prev > 0.0 && prev - refine.photoEnergyLast < prev * MESHOPT_CONVERGED_EXIT_RTOL) {
						if (++cvHits >= MESHOPT_CONVERGED_EXIT_HITS) {
							DEBUG_EXTRA("Photo-consistency converged at iteration %d/%d; skipping to the final iteration", iter + 1, iters);
							for (int k = iter; k < iters - 2; ++k)
								gstep *= decay; // fast-forward the scheduled step decay
							iter = iters - 2;
						}
					} else
						cvHits = 0;
				}
#endif
				gstep *= decay;
				progress.display(iter);
			}

			GET_LOGCONSOLE().Play();
			progress.close();
#else
			// loop a constant number of iterations and apply the gradient
			int iters(75);
			double gstep(0.4);
			if (fGradientStep > 1) {
				iters = FLOOR2INT(fGradientStep);
				gstep = (fGradientStep - (float)iters) * 10;
			}
			iters = MAXF(iters / (int)(nScale + 1), 8);
			const int iterStop(iters * 7 / 10);
			const int iterStart(fThPlanarVertex > 0 ? iters * 4 / 10 : INT_MAX);
			Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> gradients(refine.vertices.GetSize(), 3);
			Util::Progress progress(_T("Processed iterations"), iters);
			GET_LOGCONSOLE().Pause();

			constexpr int numGradentApplicationsBeforeOctreeRebuild = 10;
			int numGradientApplications = numGradentApplicationsBeforeOctreeRebuild;
			for (int iter = 0; iter < iters; ++iter) {
				refine.iteration = (unsigned)iter;
				refine.nAlternatePair = (iter + 1 < iters ? nAlternatePair : 0);
				refine.ratioRigidityElasticity = (iter <= iterStop ? fRatioRigidityElasticity : 1.f);
				const bool bAdaptMesh(iter >= iterStart && (iter - iterStart) % 3 == 0 && iters - iter > 5);
				// evaluate residuals and gradients
				if (bAdaptMesh)
					refine.vertexDepth.Resize(refine.vertices.GetSize());

				// Octree rebuilding is very expensive so we limit its frequency.
				// It is technically not accurate to do so, but in practice the vertices move slowly enough for this to be acceptable.
				bool rebuildOctree = numGradientApplications >= numGradentApplicationsBeforeOctreeRebuild;
				const double cost = refine.ScoreMesh(gradients.data(), rebuildOctree);
				if (rebuildOctree) {
					numGradientApplications = 0;
				}

				double gv(0);
				VIndex numVertsRemoved(0);
				if (bAdaptMesh) {
					// apply gradients and
					// remove planar vertices (small gradient and almost on the center of their surrounding patch)
					ASSERT(refine.vertexDepth.GetSize() == refine.vertices.GetSize());
					Mesh::VertexIdxArr vertexRemove;
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						vert -= Cast<Vertex::Type>(grad * gstep);
						const double gn(norm(grad));
						gv += gn;
						const float depth(refine.vertexDepth[v]);
						if (depth < FLT_MAX) {
							const float th(depth * fThPlanarVertex);
							if (!refine.vertexBoundary[v] && (float)gn < th && norm(refine.smoothGrad1[v]) < th)
								vertexRemove.Insert(v);
						}
					}
					if (!vertexRemove.IsEmpty()) {
						numVertsRemoved = vertexRemove.GetSize();
						mesh.Decimate(vertexRemove);
						refine.ListVertexFacesPost();
					}
					refine.vertexDepth.Empty();
					numGradientApplications = numGradentApplicationsBeforeOctreeRebuild;
				}
				else {
					// apply gradients
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						vert -= Cast<Vertex::Type>(grad * gstep);
						gv += norm(grad);
					}
					++numGradientApplications;
				}
#if MESHOPT_VISIBILITY_REUSE
				// legacy branch does not track applied displacement -> force an
				// exact recull on the next ScoreMesh
				refine.OnVerticesDisplaced(FLT_MAX);
#endif
				DEBUG_EXTRA("\t%2d. f: %.5f (%.4e)\tg: %.5f (%.4e - %.4e)\ts: %.3f\tv: %5u", iter + 1, cost, cost / refine.vertices.GetSize(), gradients.norm(), gradients.norm() / refine.vertices.GetSize(), gv / refine.vertices.GetSize(), gstep, numVertsRemoved);
				gstep *= 0.98;
				progress.display(iter);
			}
			GET_LOGCONSOLE().Play();
			progress.close();
#endif
		}


	cv::setNumThreads(-1);// Restore OpenCV internal threading

#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2) {
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefined%u.ply", nScales - nScale - 1)));
#if MESHOPT_ITER_DIAGNOSTICS
			// support colors from the final (both-directions) ScoreMesh of this scale
			SavePairSupportPLY(MAKE_PATH(String::FormatString("MeshRefinedSupport%u.ply", nScales - nScale - 1)), mesh, refine.photoGradNorm);
#endif
		}
#endif
#if MESHOPT_PROFILE
		MeshProf::Report(String::FormatString("SCALE %u/%u (image scale %.3f)", nScale + 1, nScales, (double)scale).c_str(), MeshProf::gScale);
		MeshProf::RollUp();
#endif
	}

	// Mesh was mutated and may no longer be valid.

	// Release the per-camera render data (file-scope global) so its memory is
	// reclaimed before the Scene save and the next pipeline stage instead of
	// leaking until the next RefineMesh call.
	g_cameraData.clear();
	g_cameraData.shrink_to_fit();

#if MESHOPT_PROFILE
	MeshProf::Report("GRAND TOTAL (all scales)", MeshProf::gTotal);
#endif
	return true;
} // RefineMesh
/*----------------------------------------------------------------*/
