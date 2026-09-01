/*
* Common.h
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

#ifndef _MVS_COMMON_H_
#define _MVS_COMMON_H_


// I N C L U D E S /////////////////////////////////////////////////

#if defined(MVS_EXPORTS) && !defined(Common_EXPORTS)
#define Common_EXPORTS
#endif

#include "../Common/Common.h"
#include "../IO/Common.h"
#include "../Math/Common.h"

#include <cstdlib> // std::getenv, for the DENSIFY_DIAG gate below

#ifndef MVS_API
#define MVS_API GENERAL_API
#endif
#ifndef MVS_TPL
#define MVS_TPL GENERAL_TPL
#endif


// D E F I N E S ///////////////////////////////////////////////////
// 3 is noticeably better than 2, 4 not so.
#define DPC_NUM_ITERS (3) // Was 4 in _USE_CUDA pathway, 3 not.
#define DPC_IMAGE_CACHE
#undef DPC_EXTENDED_OMP_THREADING // Changes threading order and drastically affects the result.
#define DPC_EXTENDED_OMP_THREADING2
#define DPC_NEW_FUSING

// Use a faster, but less accurate, exp function in score factor calculation?
#define DPC_FASTER_SCORE_FACTOR

// Reduce the precision in Dir2Normal and Normal2Dir to improve performance?
#define DPC_FASTER_RANDOM_ITER_CALC

#define DPC_FASTER_SAMPLING

// Disabled due to precision problems in sampling.
#undef DPC_FASTER_SAMPLING_USE_INV_Z

// Reduce the detail calculation accuracy to improve performance?
#define DPC_FASTER_SCORE_PIXEL_DETAIL
#define DPC_FASTER_SCORE_PIXEL_DETAIL2

// Use a parallel version of pca_estimate_normals (requires TBB)?
// JPB WIP OPT Restore when the support for this can be added to the build.
#define DPC_FASTER_NORMAL_ESTIMATION

// DPC_FASTER_SAMPLING related:

// Replace the per-sample projective divides in the bilinear sampler
// (GatherSampleInfo/IsScorable3) with _mm_rcp_ps refined by one Newton-Raphson
// step. Unlike DPC_FASTER_SAMPLING_USE_INV_Z above (which interpolated 1/z
// across the patch and accumulated error), this refines every sample's
// reciprocal to ~1-2 ulp of a true divide: the coordinate error stays under
// 0.01px, which the 1-texel slack on both sides of IsScorable3's corner test
// absorbs, so no out-of-bounds fetch is possible.
// MEASURED SLOWER (part of 8:56 -> 9:34 with both new toggles on), so off.
// The likely mechanism: with the divides already latency-hidden at the loop
// end ("what worked" note 9), divps runs on the otherwise-idle divider port
// and is close to free, while rcp+NR is 4x the uops on the mul/add ports the
// sampling FMAs saturate -- it adds contention where there was none. Matches
// the old raw-rcp experiment showing benefit (1 op) where this (4 ops) loses.
// Kept for A/B on other microarchitectures; safe to enable (precision-wise)
// on any of them.
#undef DPC_FAST_RCP_DIV

// Skip the INTER_AREA resize of a NEIGHBOR view's grey image in ScaleDepthData
// when the (image, scale, source dims) entry is already resident in
// sCachedImages: under DPC_FASTER_SAMPLING a neighbor's scaled grey is only
// ever read to build the 4-plane sampling derivative, and each neighbor serves
// ~nNumViews reference images per pass, so nearly all of those resizes
// recompute bytes the cache already holds. The cache stores the grey alongside
// the derivative and the view adopts it (shallow, bit-identical), so every
// downstream reader sees exactly what the skipped resize would have produced.
// The reference view always resizes for real.
#define DPC_SKIP_CACHED_NEIGHBOR_RESIZE

// Tried and REMOVED (measured no gain): a precomputed per-8x8-block bit-mask of
// the views each pixel could ever be scorable in (the projection over all
// depths is a ray, testable once per block), consulted before IsScorable3.
// Redundant: IsScorable3's branchless SIMD corner test already rejects in ~25
// ops, and with ~90% of calls scorable the ceiling of making rejection free is
// ~1% of scoring time. Any future early-out work here must target the ~90%
// that PASS, not the 10% that fail.

// Skip ALL per-view scoring for a hypothesis when the reference patch is so
// flat that the prior blend decides the score anyway. Pixels with a low-res
// prior deliberately bypass the fDescriptorMinMagnitudeThreshold cull (that is
// the prior-guided textureless fill), so glassy water/saturated sky run the
// full 13-hypotheses x nNumViews gather pipeline -- yet with
// finalScore = (1-f)*ncc + f*deltaDepth and f = exp(-normSq0/0.02), a patch at
// f >= 0.999 (normSq0 <= ~2e-5, intensity stddev under ~0.5% grey) produces
// blended scores equal to f*deltaDepth to within (1-f)*thRobust < 0.003 no
// matter what the gathers return. Returning that value directly is a bounded
// approximation (well under any decision threshold; note it also skips the
// all-views-unscorable -> thRobust case for such pixels near image borders).
#define DPC_FLAT_PATCH_PRIOR_SHORTCUT

// Per-pixel view short-listing, the CPU analogue of the GPU path's per-pixel
// view selection: the final score is the min-mean of the TWO best per-view
// scores, so most views' gathers feed an aggregation that discards them. The
// FIRST hypothesis a pixel scores each iteration still evaluates every view;
// the DPC_VIEW_SHORTLIST_SIZE best of its final blended scores then become the
// only views scored for the pixel's remaining hypotheses that iteration
// (neighbor propagation and perturbation refinement). Exemptions that keep the
// mask at all-views: the completely-random recovery loop (conf >= thConfRand
// explores depths where a DIFFERENT view set may match, e.g. occlusions), and
// first hypotheses where fewer than DPC_VIEW_SHORTLIST_SIZE+1 views scored.
// Semantic change: later hypotheses lose the pruned views as candidates for
// their best-two (~(H-1)/H * (N-K)/N of view scoring saved).
// MEASURED SLOWER (1:45 vs 1:39) and disabled. The mechanism: min-mean-2 over
// a subset is >= min-mean-2 over all views, so shortlisted hypotheses score
// systematically worse -- and the refinement control flow is SCORE-ADAPTIVE
// (thConfSmall/thConfBig/thConfRand gates, the RefineIters re-loop): inflated
// scores push more pixels into the conf>=thConfRand random-recovery branch
// (nRandomIters hypotheses at full view cost, exempt from the shortlist by
// design) and reduce adoptions per iteration. The GPU's per-pixel view
// selection is safe only because its control flow is static. Do not re-enable
// without also rethinking the conf gates; possibly still a net win at
// --number-views 12 where the savings fraction is largest.
#undef DPC_VIEW_SHORTLIST
#define DPC_VIEW_SHORTLIST_SIZE 4

// D I A G N O S T I C   G A T E S ////////////////////////////////////
//
// The four module gates below -- DENSIFY_DIAG, MESH_DIAG, TEXTURE_DIAG and
// REFINE_DIAG -- are COMPILE-TIME switches, off by default.
//
// They began as runtime switches (OPENMVS_*_DIAG=1 in the environment, or -v 4)
// and that was wrong on both counts:
//   * -v is shared with the per-image PNG/PLY/conf dumps, so asking for a
//     counter meant paying for gigabytes of debug imagery -- and -v 3, the level
//     an operator actually reaches for, sits one BELOW these gates and opened
//     none of them. That is exactly how a -v 3 run came back with no [MESH-*]
//     trace in it at all while still writing a 5.9M-vertex diagnostic PLY.
//   * an environment variable is invisible in the log it changes: a run either
//     had OPENMVS_MESH_DIAG set or it did not, and nothing in the output says
//     which, so an absent line is ambiguous between "not measured" and "gate
//     was closed".
// Neither trigger can remove the strings, the arithmetic or the extra passes
// from a shipped binary either, so the algorithm-disclosure argument that
// motivates REFINE_DIAG (below) was only ever half-honoured.
//
// As compile-time constants the guarded blocks fold away in an optimised build:
// no cost, no strings, nothing to disclose. Turning one ON makes its lines
// UNCONDITIONAL -- no verbosity level to reach, no environment variable to set.
//
// Set them either from the build system
//     -DOPENMVS_DIAG=1            (all four)
//     -DOPENMVS_MESH_DIAG=1       (one module)
// or by flipping the default here. OPENMVS_DIAG is the master default for any
// module that is not set explicitly, so a per-module define always wins.
//
// The macros keep the exact shape they had -- `if (<gate>) VERBOSE(...)` -- so
// the arguments stay referenced by the compiler and a closed gate does not
// produce a wall of unused-variable warnings in the blocks that feed it.
#ifndef OPENMVS_DIAG
#define OPENMVS_DIAG 1
#endif
#ifndef OPENMVS_DENSIFY_DIAG
#define OPENMVS_DENSIFY_DIAG OPENMVS_DIAG
#endif
#ifndef OPENMVS_MESH_DIAG
#define OPENMVS_MESH_DIAG OPENMVS_DIAG
#endif
#ifndef OPENMVS_TEXTURE_DIAG
#define OPENMVS_TEXTURE_DIAG OPENMVS_DIAG
#endif
#ifndef OPENMVS_REFINE_DIAG
#define OPENMVS_REFINE_DIAG OPENMVS_DIAG
#endif

// MVS_DUMP_FILES: intermediate DEBUG FILES -- poisson_removed_components.ply,
// mesh_frag.txt and anything else written purely to be opened later.
//
// Deliberately NOT on the *_DIAG gates above. Those control LOG TEXT, which is
// cheap, has no side effects, and which a diagnostic build should carry freely.
// Writing files is none of those things: poisson_removed_components.ply alone is
// a 5.9M-vertex PLY emitted on every run. Compiling in readable logs must not
// also turn on littering the output directory -- that is a second, unrelated
// decision, and bundling them meant you could not have one without the other.
//
// Verbosity is the right control here: per-run, set on the command line, no
// rebuild, and it is already how every other intermediate dump in this tree is
// controlled.
//
// > 3, i.e. -v 4 -- and note this is deliberately ONE LEVEL ABOVE the upstream
// dump sites, which use > 2 (-v 3): the per-image depth/conf/ply (Scene.cpp,
// SceneDensify.cpp), MeshRefine*.ply and MeshRefineSupport*.ply (SceneRefine,
// SceneRefineCUDA), ExportCamerasMLP (DensifyPointCloud). That is an intentional
// inconsistency, not an oversight. -v 3 is a level reached for casually while
// chasing a log line -- it was reached for twice in one session here -- and these
// two dumps are far heavier than a per-image PNG: poisson_removed_components.ply
// carries the FULL 5.9M-vertex array to describe 515k removed faces. Costing a
// gigabyte because someone wanted a counter is the failure this is avoiding.
//
// The upstream -v 3 sites are left alone: they are a long-standing convention and
// rewriting ten of them to chase symmetry is a bigger change than it is worth. Be
// aware that -v 3 still writes those.
#if TD_VERBOSE == TD_VERBOSE_OFF
#define MVS_DUMP_FILES()	false
#else
#define MVS_DUMP_FILES()	(VERBOSITY_LEVEL > 3)
#endif


// DENSIFY_DIAG: visibility gate for everything the densify module logs beyond
// its top-level result, for two reasons that land on the same switch.
//
// 1. Development instrumentation -- the memory probes (MEMCP/MEMDIAG), the cache
//    budgets and hit-rates (ImageCache/greyImages/sCachedImages/DMapCache/
//    FilterDMapCache), the filter-phase profile, the per-filter distance
//    histograms, and the once-per-image pairing/estimate/filter lines. Bisection
//    aids, not results, and they bury the handful of lines that matter.
// 2. Algorithm disclosure -- the stage names and their internal metrics (that
//    patch-match runs and on what device, views/image, geometric-consistency
//    convergence and its threshold, dmap-cache hit-rates, which point filters
//    ran and at what standard-deviation cutoffs). A default run should say what
//    it produced, not how.
//
// So the default log keeps only the phase results and every error/warning; the
// rest is a rebuild away -- build with -DOPENMVS_DENSIFY_DIAG=1 (or the master
// -DOPENMVS_DIAG=1) and they come back unconditionally, at any verbosity.
// Only VISIBILITY changes: every message keeps its text, its arguments, and the
// conditions under which it is computed, so a build with the gate open logs
// exactly what the ungated code logged.
#if TD_VERBOSE == TD_VERBOSE_OFF || !OPENMVS_DENSIFY_DIAG
#define DENSIFY_DIAG_ENABLED()	false
#else
#define DENSIFY_DIAG_ENABLED()	true
#endif
#if TD_VERBOSE == TD_VERBOSE_OFF
#define DENSIFY_DIAG(...)
#else
#define DENSIFY_DIAG(...)	{ if (DENSIFY_DIAG_ENABLED()) VERBOSE(__VA_ARGS__); }
#endif


// MESH_DIAG: the same visibility gate for the mesh module -- ReconstructMesh.cpp,
// SceneReconstruct.cpp's Poisson path and Mesh::Clean. It covers the development
// instrumentation those carry: the [MESH-POLICY]/[MESH-EXTENT]/[MESH-CALIB]/
// [MESH-SLOPE]/[MESH-REFINE]/[MESH-ATLAS]/[MESH-CAP] policy trace, the
// [MESH-OCCUPANCY]/[MESH-OVERHANG]/[MESH-CULL]/[MESH-FRAG] distributions, the
// vertex-density percentile ladders, and Clean's per-phase "DIAG ..." counters.
// Between them they emit ~90 lines per run at the DEFAULT verbosity, which buries
// the handful that actually describe the reconstruction.
//
// Built with -DOPENMVS_MESH_DIAG=1 (or the master -DOPENMVS_DIAG=1); independent
// of the other three, so a build can carry one module's trace and not the rest.
//
// This gate is deliberately used for the COMPUTATION as well as the printing: the
// percentile ladders sort 200k-sample arrays, [MESH-EXTENT] runs a Jacobi PCA and
// a 2048-bin band search, and [MESH-OVERHANG] builds a second chamfer transform
// over the occupancy grid -- all of it feeding nothing but a log line. Use
// MESH_DIAG_ENABLED() to skip those blocks whole.
//
// What is NOT gated: anything that decides geometry, any error or warning, the
// per-stage result lines, and mesh_plan.txt (the orchestrator parses it). Files
// written purely to be looked at later -- mesh_frag.txt -- ride the gate.
#if TD_VERBOSE == TD_VERBOSE_OFF || !OPENMVS_MESH_DIAG
#define MESH_DIAG_ENABLED()	false
#else
#define MESH_DIAG_ENABLED()	true
#endif
#if TD_VERBOSE == TD_VERBOSE_OFF
#define MESH_DIAG(...)
#else
#define MESH_DIAG(...)		{ if (MESH_DIAG_ENABLED()) VERBOSE(__VA_ARGS__); }
#endif


// TEXTURE_DIAG: the same visibility gate for the texturing module -- TextureMesh.cpp
// and SceneTexture.cpp. It covers the development instrumentation those carry: the
// [MEM]/[MEM-DECOMP] footprint probes, the [PROFILE] stage and per-view sub-stage
// timings, the memory/thread budget traces (image residency, seam-leveling workers),
// the [CROP-CAP]/[CROP-DECOMP]/[ATLAS-FIT]/[ATLAS-GUTTER] atlas sizing trace, the
// [TEX-SHEET]/[SEAM-SKIP]/[SEAM-OUTSET]/[DATACOLOR-*] boundary and fill counters, and
// the [LBP-DIAG] unobserved floor. Between them they emit ~100 lines per run at the
// DEFAULT verbosity, which buries the handful that actually describe the texture.
//
// Built with -DOPENMVS_TEXTURE_DIAG=1 (or the master -DOPENMVS_DIAG=1); independent
// of the other three, so a build can carry one module's trace and not the rest.
//
// Like MESH_DIAG this gate is used for the COMPUTATION as well as the printing: the
// footprint probes walk every face's observation list, the component reports copy and
// sort the whole size distribution, and the border-loop report re-scans every vertex --
// all of it feeding nothing but a log line. Use TEXTURE_DIAG_ENABLED() to skip those
// blocks whole; anything whose result also DECIDES something stays outside the gate.
//
// What is NOT gated: anything that decides geometry or the atlas, any error or warning,
// the per-stage result lines, the atlas-overflow/adaptive-downscale notices (they
// explain a resolution loss in the deliverable), and the counts of faces the texturing
// pass removes from or moves on the output mesh.
#if TD_VERBOSE == TD_VERBOSE_OFF || !OPENMVS_TEXTURE_DIAG
#define TEXTURE_DIAG_ENABLED()	false
#else
#define TEXTURE_DIAG_ENABLED()	true
#endif
#if TD_VERBOSE == TD_VERBOSE_OFF
#define TEXTURE_DIAG(...)
#else
#define TEXTURE_DIAG(...)		{ if (TEXTURE_DIAG_ENABLED()) VERBOSE(__VA_ARGS__); }
#endif


// REFINE_DIAG: the same visibility gate for the mesh-refinement module --
// RefineMesh.cpp, SceneRefine.cpp (CPU path) and SceneRefineCUDA.cpp (GPU path).
//
// This one is drawn on a stricter line than the three above, because the refiner's
// log was not merely noisy -- it NARRATED THE ALGORITHM. Left open it recited the
// coarse-to-fine scale schedule and its per-iteration cost/gradient trace, the
// view-streaming budget and batch partition, the GPU residency ladder and its VRAM
// arithmetic, the auto-decimate and subdivision thresholds, and the host-vs-device
// scoring model with all five of its tuning constants. So the rule here is not
// "is this noisy?" but "does this describe HOW the refiner works?" -- and anything
// that does rides the gate, including single lines and including warnings.
//
// What stays visible is what the operator can act on or must know, phrased as an
// OUTCOME rather than a mechanism:
//   * which device ran, and the switch that would change it;
//   * settings this run reduced on their behalf, and that quality moved as a result
//     (both are command-line options, so naming them exposes nothing internal);
//   * that a run will be slow, or that memory is short, WITHOUT the partition,
//     budgets and internal knobs behind that judgement;
//   * anything that changes what lands on disk -- notably that the saved scene will
//     not carry the dense point cloud;
//   * errors and failures, named by what failed rather than by which internal
//     structure was being built at the time.
// Every one of those has its mechanism re-attached under the gate, on the line
// immediately after it, so a diagnostic run reads exactly as the old log did.
//
// Built with -DOPENMVS_REFINE_DIAG=1 (or the master -DOPENMVS_DIAG=1); independent
// of the other three, so a build can carry one module's trace and not the rest.
//
// Used for the COMPUTATION as well as the printing wherever the numbers exist only
// to be printed: the convergence table costs two O(vertices) Eigen norm reductions
// per iteration, the [SUBDIV] line a full pass over the per-face area array, the
// batch-balance line a string built per batch, and MESHOPT_CUDA_PROFILE_SYNC a
// cuCtxSynchronize per reference group. Use REFINE_DIAG_ENABLED() to skip those
// blocks whole.
#if TD_VERBOSE == TD_VERBOSE_OFF || !OPENMVS_REFINE_DIAG
#define REFINE_DIAG_ENABLED()	false
#else
#define REFINE_DIAG_ENABLED()	true
#endif
#if TD_VERBOSE == TD_VERBOSE_OFF
#define REFINE_DIAG(...)
#else
#define REFINE_DIAG(...)		{ if (REFINE_DIAG_ENABLED()) VERBOSE(__VA_ARGS__); }
#endif


// P R O T O T Y P E S /////////////////////////////////////////////

using namespace SEACAVE;

#define _USE_OPENCV
#define _DISABLE_NO_ID
#include "Interface.h"

namespace MVS {

// Initialize / close the library; should be called at the beginning and end of the program
void Initialize(LPCTSTR appname, unsigned nMaxThreads=0, int nProcessPriority=0);
void Finalize();
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_COMMON_H_
