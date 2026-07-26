/*
* SceneTexture.cpp
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
#include "RectsBinPack.h"
#include "robin_map.h" // assumes robin_map.h is in include path

#include <boost/graph/filtered_graph.hpp>
// connected components
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using namespace MVS; 

// [MEM] TEMPORARY per-stage peak-memory probe (Windows). PeakWorkingSetSize is
// monotonic, so the FIRST stage boundary where it jumps owns the peak. Used to
// target the 32 GB fit; remove once the dominant allocation is identified.
#ifdef _WIN32
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
static void LogPeakMem(const char* stage) {
	PROCESS_MEMORY_COUNTERS pmc = {};
	pmc.cb = sizeof(pmc);
	if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
		DEBUG_EXTRA("[MEM] %-26s peakWS=%.2f GB  curWS=%.2f GB", stage,
			pmc.PeakWorkingSetSize / (1024.0*1024.0*1024.0),
			pmc.WorkingSetSize / (1024.0*1024.0*1024.0));
}
#else
static void LogPeakMem(const char*) {}
#endif

// Return freed heap pages to the OS. cv::Mat pixel buffers go through the CRT
// heap (malloc); on Windows a released large allocation is put back on the heap
// free list but NOT decommitted, so a transient that decodes hundreds of full-res
// source images (~20 GB) keeps inflating the working set of every later stage even
// after release(). _heapmin() (HeapCompact) decommits those free pages so the
// working set actually drops. Cheap; call it once after a big transient is freed.
#ifdef _WIN32
#include <malloc.h>
static inline void TrimHeap() { _heapmin(); }
#else
#include <malloc.h>
static inline void TrimHeap() {
#ifdef __GLIBC__
	malloc_trim(0);
#endif
}
#endif

// ============================================================================
// [PROFILE] Lightweight, easily-removable stage timing for this module.
// Everything is gated behind TEXTURE_PROFILE. Set it to 0 (or delete this block
// plus the TEX_PROFILE_* call sites) to strip every timer at compile time with
// ZERO residual overhead -- when disabled the macros expand to ((void)0).
// Output goes through DEBUG_EXTRA with a [PROFILE] tag so it sits next to the
// existing [MEM] probes.
//   - TEX_PROFILE_SCOPE("name")   : times the enclosing { } scope via RAII.
//   - TEX_PROFILE_BEGIN(tok)      : start a manual timer named by token `tok`.
//   - TEX_PROFILE_END(tok,"name") : stop that timer and log elapsed ms.
// NOTE: place SCOPE/BEGIN/END only on the MAIN thread (at stage granularity),
// never inside an omp parallel-for body, or the log will be spammed per-item.
// ============================================================================
#ifndef TEXTURE_PROFILE
#define TEXTURE_PROFILE 1
#endif

#if TEXTURE_PROFILE
#include <chrono>
namespace { namespace texprof {
	using Clock = std::chrono::steady_clock;
	static inline void Log(const char* name, Clock::time_point t0) {
		const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		DEBUG_EXTRA("[PROFILE] %-44s %10.2f ms", name, ms);
	}
	struct ScopeTimer {
		const char* name; Clock::time_point t0;
		inline ScopeTimer(const char* n) : name(n), t0(Clock::now()) {}
		inline ~ScopeTimer() { Log(name, t0); }
	};
} }
#define TEX_PROFILE_CONCAT_(a, b) a##b
#define TEX_PROFILE_CONCAT(a, b) TEX_PROFILE_CONCAT_(a, b)
#define TEX_PROFILE_SCOPE(name) texprof::ScopeTimer TEX_PROFILE_CONCAT(texprofScope_, __LINE__)(name)
#define TEX_PROFILE_BEGIN(tok) const texprof::Clock::time_point tok(texprof::Clock::now())
#define TEX_PROFILE_END(tok, name) texprof::Log((name), (tok))
#else
#define TEX_PROFILE_SCOPE(name) ((void)0)
#define TEX_PROFILE_BEGIN(tok) ((void)0)
#define TEX_PROFILE_END(tok, name) ((void)0)
#endif

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define TEXOPT_USE_OPENMP
#endif

#define FASTER_OUTLIER_DETECTION
#undef COUNT_ITERATIONS
#undef STATS

// TEXTURE_DATACOLOR_UNOBSERVED: when 1, faces with NO genuine camera view
// (e.g. Poisson-extrapolated / hole-filled surface) are NOT smeared with a
// neighbor's projected texture (the label-propagation pass is skipped for them);
// instead each is filled with a flat color sampled from the dense point cloud
// (nearest populated voxel), baked as a solid texel appended to the atlas -- the
// competitor's "vertex-colored" margin on poorly-defined areas. Default 0 ->
// texturing behaves EXACTLY as before (zero impact when off).
#ifndef TEXTURE_DATACOLOR_UNOBSERVED
#define TEXTURE_DATACOLOR_UNOBSERVED 1
#endif

// Iterations of Laplacian color-smoothing over face adjacency applied to the
// data-colored faces, so the flat per-face fill reads as a smooth color field
// instead of a noisy triangle patchwork. 0 = off (raw per-face nearest color).
// The Bayer dither in the bake (build 8) de-correlates the per-atlas-cell 8-bit
// rounding, so heavy flattening NO LONGER reintroduces the "crazing" seam network
// that capped this at 40 before. 240 strongly averages out the streaky gradients
// produced when an unobserved region is ringed by disparate boundary colours
// (e.g. interior holes: dark inside vs bright outside) -> a smooth, artifact-free
// fill, which is what reads best on these synthesized regions. Lower (~40) keeps
// more spatial variation but shows those seed streaks; higher is essentially a
// per-component flat colour.
#ifndef TEXTURE_DATACOLOR_SMOOTH_ITERS
#define TEXTURE_DATACOLOR_SMOOTH_ITERS 240
#endif

// TEXTURE_DATACOLOR_SEAM_SMOOTH_ITERS: bounded seam-smoothing passes applied to the
// component-tile fill AFTER the nearest-colour inpaint has completed. The old single
// TEXTURE_DATACOLOR_SMOOTH_ITERS knob did BOTH the fill and the smoothing in one loop,
// so a large value (240) was needed to fill big holes -- but that same large value then
// kept averaging every already-filled vertex, collapsing each connected component to a
// single flat colour (the "muddy smear"). The component path now fills first (nearest
// observed surface colour, so a fill spanning e.g. grass->road keeps grass near grass and
// road near road) and ONLY THEN applies this many light averaging passes to hide the
// propagation-front seams -- with the boundary seed vertices PINNED so the fill stays
// locked to the real texture it abuts (no drift = no visible seam). ~40 removes the fronts
// without flattening the large-scale spatial variation; raise for a softer/flatter fill,
// lower to keep more variation (and slightly sharper internal fronts). Only used by the
// component-tile bake; the legacy per-face bake still uses TEXTURE_DATACOLOR_SMOOTH_ITERS.
#ifndef TEXTURE_DATACOLOR_SEAM_SMOOTH_ITERS
#define TEXTURE_DATACOLOR_SEAM_SMOOTH_ITERS 40
#endif

// TEXTURE_DATACOLOR_TILE_NORMAL_COS: the component-tile bake projects each connected
// no-view region onto ONE plane (its average face normal). If a connected fill wraps
// around a multi-faceted object (e.g. the top AND sides of a box), faces steeply angled
// to that single plane foreshorten to near-degenerate tile triangles -> they get too few
// texels and their shared edges render as a faint CRACK NETWORK tracing the tessellation.
// To keep every tile near-planar (injective, foldover-free projection) the component BFS
// only merges two edge-adjacent faces into the SAME tile when their normals agree to at
// least this cosine; sharper creases start a new tile. The per-vertex fill colour is
// computed over the FULL mesh adjacency BEFORE tiling, so it stays continuous across a
// split (both tiles share the crease vertices' colours) -> the split boundary is a
// near-invisible seam at a genuine geometry crease, not a mid-surface crack. 0.5 ~= split
// at creases sharper than 60deg (keeps gentle curves as one tile). Raise toward 1 to split
// more aggressively (flatter tiles, more/smaller tiles); lower to merge more (risking
// foldover). 1.0 splits at every non-coplanar edge; <=0 disables the split (old behaviour).
#ifndef TEXTURE_DATACOLOR_TILE_NORMAL_COS
#define TEXTURE_DATACOLOR_TILE_NORMAL_COS 0.5f
#endif

// TEXTURE_DATACOLOR_DETAIL_GAIN: amount of REAL high-frequency texture detail transplanted
// onto the smooth fill. A correct, smooth, seamless fill still reads as an obvious blurry
// blob against detailed surroundings (worst on vegetation/foliage, gravel, etc.). To fix
// that, the bake picks the largest OBSERVED face bordering each fill tile, crops a square of
// its real atlas texture, high-passes it (crop minus a blur = zero-mean grain), and adds that
// detail -- mirror-tiled -- on top of the smooth base colour. The base colour (correct and
// seamless) is preserved; only local contrast/grain is injected, so the fill looks like the
// surrounding surface instead of a flat patch. 1.0 = full real detail amplitude; lower for a
// subtler effect; 0 disables (smooth base only, previous behaviour).
// SET TO 0 for the same reason as TEXTURE_DATACOLOR_MIRROR_GAIN: the injected high-frequency
// grain is the very thing that made the fill read as textured next to the smooth-converged
// observed band, drawing the polygonal seam line. Smooth fill = continuous frequency at the
// boundary = no traceable edge (the chosen "prefer smooth" behaviour).
#ifndef TEXTURE_DATACOLOR_DETAIL_GAIN
#define TEXTURE_DATACOLOR_DETAIL_GAIN 0.0f
#endif

// TEXTURE_DATACOLOR_MIRROR_GAIN: strength of REFLECTION PADDING at the fill boundary. The
// smooth fill reads as a flat blob next to the sharp real texture no matter how well the
// colour matches, because it lacks the neighbour's high-frequency structure. To fix that,
// each fill face that borders an OBSERVED (real-texture) face mirrors that neighbour's
// actual texture across their shared edge into the fill: a fill texel at barycentric depth
// d from the shared edge samples the observed neighbour at the same along-edge position and
// depth d on ITS side (a reflection). At the edge (d=0) the sample IS the neighbour's colour
// at the shared edge -> seamless; moving inward it pulls real detail from the neighbour,
// faded by (1-d) so it vanishes at the fill face's far vertex. Each boundary face mirrors
// its OWN neighbour, so different stretches of the boundary get locally-correct detail (no
// single-donor tiling look). This only affects the ~1-face-deep fill boundary band -- the
// seam -- where the crisp/flat mismatch is most visible. 1.0 = full mirrored detail at the
// edge; lower for a subtler effect; 0 disables (smooth base + grain only, previous look).
// SET TO 0: the mirrored real detail made the fill GRAINY right at the boundary while the
// seam-feather converges the OBSERVED side to the pure smooth vr/vg/vb field (no grain).
// That frequency reversal -- grainy-fill next to smooth-observed at the SAME colour -- is
// exactly the crisp polygonal LINE the eye traced along the triangle edges. Zeroing it makes
// the fill the same smooth field the feather ends on, so value AND frequency are continuous
// across the seam (no line). Re-enable (0.3-0.5) only if the fill must carry real grain AND
// the observed feather is changed to preserve detail too.
#ifndef TEXTURE_DATACOLOR_MIRROR_GAIN
#define TEXTURE_DATACOLOR_MIRROR_GAIN 0.0f
#endif

// TEXTURE_DATACOLOR_MIN_VIEW_COS: faces whose ONLY/best camera observation is at
// a grazing angle (cos(faceNormal, camDir) below this) are the Poisson-extrapolated
// mesh RIM past good camera coverage. The labeling forces a view onto them anyway
// ("assign best available camera"), so they end up as per-face projected patches
// that render as gray facets (a distant/oblique view's pixels stretched across the
// rim). When TEXTURE_DATACOLOR_UNOBSERVED is on, such faces are instead demoted to
// NO_ID so the surface-propagated data-color fill takes them from their well-seen
// neighbours -- exactly like the rest of the unobserved fill. cos is a true cosine
// (1 = frontal/nadir, 0 = edge-on). 0 disables the demotion (pre-existing behaviour).
// 0.20 ~= demote observations more grazing than ~78deg; raise to demote more of the
// rim, lower to keep more grazing-but-real texture.
#ifndef TEXTURE_DATACOLOR_MIN_VIEW_COS
#define TEXTURE_DATACOLOR_MIN_VIEW_COS 0.20f
#endif

// TEXTURE_DATACOLOR_BAKE: master switch for the actual data-color FILL/bake step in
// GenerateTexture. Independent of TEXTURE_DATACOLOR_UNOBSERVED on purpose: with this
// 0 and TEXTURE_DATACOLOR_UNOBSERVED still 1, the legacy label-forcing stays gated
// off (no gray-facet smear comes back) AND the per-face quadratic-patch bake is
// skipped, so no-view faces simply stay NO_ID / colEmpty (blank). Use this to review
// the camera-textured result WITHOUT the synthesized-fill triangle-banding anomaly.
// Set back to 1 to restore the data-color fill. Set to 0 to go back to the blank/grey
// review state.
// TEST 3 RESULT: bake OFF did NOT change the trailer veins -> that surface is
// OBSERVED texture, not data-color fill. Bake restored to 1.
#ifndef TEXTURE_DATACOLOR_BAKE
#define TEXTURE_DATACOLOR_BAKE 1
#endif

// TEXTURE_DATACOLOR_COMPONENT_BAKE: selects HOW the data-color fill is baked into the
// atlas (only matters when TEXTURE_DATACOLOR_BAKE is on).
//  1 = COMPONENT-TILE bake: each connected component of no-view faces is packed into
//      ONE shared atlas tile and the smooth colour field is rasterised across the whole
//      tile at once. Adjacent faces share texels along shared edges -> there are no
//      per-face cell boundaries, so the per-cell 8-bit quantization "crazing" that the
//      per-face bake shows on large near-uniform regions cannot occur. The fill is a
//      smooth field, so the planar per-component projection tolerates foldover of
//      non-planar (tree-blob) components (overlapping faces paint near-equal colours).
//  0 = legacy PER-FACE cell bake (one tiny M x M quadratic-patch cell per face; smooth
//      but crazes on big regions).
#ifndef TEXTURE_DATACOLOR_COMPONENT_BAKE
#define TEXTURE_DATACOLOR_COMPONENT_BAKE 1
#endif

// TEXTURE_DATACOLOR_FEATHER_RINGS: width (in face-rings) of the seam-feather band that
// softens the boundary between the real observed texture and the synthesized fill (and the
// mesh silhouette/rim). Within this band the observed texture is progressively BLURRED
// (smeared) toward the boundary so the sharp detail dissolves into the blurry fill/halo and
// the transition becomes invisible -- rather than a sharp region abutting a blurry one. The
// blur radius ramps from TEXTURE_DATACOLOR_FEATHER_BLUR_PX at the boundary (fully smeared,
// matching the fill) down to 0 at the far edge of the band. Wider = the smear extends deeper
// into the real texture (more of it lost, but a longer/gentler dissolve). 0 disables the
// feather (hard boundary, previous behaviour); 4-8 is a reasonable band.
// RAISED to 14: the fill is heavily blurred, so a wide observed band gives the real
// texture room to dissolve gradually into it; combined with the smoothstep ramp in the
// blend, the point where real detail returns is imperceptible (no visible inner edge).
#ifndef TEXTURE_DATACOLOR_FEATHER_RINGS
#define TEXTURE_DATACOLOR_FEATHER_RINGS 14
#endif

// TEXTURE_DATACOLOR_FEATHER_STRENGTH: scales the maximum smear (at the boundary). 1.0 = the
// boundary observed texels are blurred with the full TEXTURE_DATACOLOR_FEATHER_BLUR_PX radius
// (maximally smeared -> best hides the seam, most detail lost right at the edge); lower keeps
// a little more detail at the seam. Only used when TEXTURE_DATACOLOR_FEATHER_RINGS > 0.
#ifndef TEXTURE_DATACOLOR_FEATHER_STRENGTH
#define TEXTURE_DATACOLOR_FEATHER_STRENGTH 1.0f
#endif

// TEXTURE_DATACOLOR_FEATHER_BLUR_PX: the maximum blur radius (in atlas pixels) applied to the
// observed texture right at the fill/rim boundary. Each band texel is replaced by a box
// average of the original observed atlas over a radius that ramps from this value at the
// boundary to 0 at the band's inner edge. Larger = a heavier smear that more completely
// dissolves the seam (match it to how blurry the fill looks); smaller = a subtler smear.
// Cost grows ~radius^2 per band texel, so keep it modest (6-12).
// RAISED to 18 to match the heavily-blurred fill: at 10 the observed side stayed too crisp
// right up to the boundary, so the sharp/blurry mismatch still read as a clear separation.
// If this ever becomes a bake-time bottleneck, switch the smear to a single pre-blurred
// atlas + blend-by-ramp (O(atlas) instead of O(band*radius^2)); costs one atlas-sized temp.
#ifndef TEXTURE_DATACOLOR_FEATHER_BLUR_PX
#define TEXTURE_DATACOLOR_FEATHER_BLUR_PX 18
#endif

// TEXTURE_DATACOLOR_DEBUG_TINT: diagnostic only. When 1, every synthesized data-color
// FILL texel is painted solid bright MAGENTA (255,0,255) instead of its computed colour,
// so in the viewer it is instantly obvious which faces are synthetic fill vs real observed
// texture. Use this to tell whether a blurry region abutting the real texture is the fill
// on the SAME surface (it turns magenta) or a separate/background surface seen past the
// mesh edge (it stays as-is while the magenta fill is elsewhere) -- i.e. whether the hard
// transition is a texture-atlas problem or a geometry/occlusion silhouette. 0 = normal
// (off). REMOVE / set back to 0 after diagnosing.
// DIAGNOSIS DONE: the fill = the Poisson-extrapolated OUTER RIM/skirt of the mesh (a thin
// ragged border in plan view; droops down so it fills the frame in oblique shots). It is on
// the same surface, so it is texturable. Tint restored to 0.
// (July 2026: briefly re-enabled to diagnose the jagged line, but the "still there" reports were
// from running the WRONG exe -- diagnosis invalid. Tint back to 0 so the ACTUAL current build
// (smoothstep ramp + mirror/detail grain zeroed) can be retested cleanly.)
#ifndef TEXTURE_DATACOLOR_DEBUG_TINT
#define TEXTURE_DATACOLOR_DEBUG_TINT 0
#endif

// TEXTURE_DATACOLOR_DIAG / TEXTURE_DATACOLOR_DIAG_TINT: investigation instrumentation for the
// synthetic-vs-real SEAM. DIAG (1) logs a [DATACOLOR-DIAG] line with: #no-view faces, #component
// tiles, #feather band faces, how many band texels the feather WROTE vs SKIPPED (skipEmpty = the
// band texel sat on colEmpty in the atlas so the blend was skipped -> an UNBLENDED seam there;
// skipAlpha = the ramp had reached 0), and the boundary VALUE-STEP stats: meanStep/maxStep =
// |observed - fill target| at the ring-0 boundary (how big a jump the feather must cross), and
// meanResid/maxResid = |blended result - fill target| there. meanResid should be ~0 if the feather
// fully flattens the observed side to the fill colour at the seam; if it is LARGE the observed side
// never actually reaches the fill colour at the boundary -> that residual step IS the visible line.
// DIAG_TINT (1) additionally FALSE-COLOURS the atlas so the seam location is unambiguous: synthetic
// FILL tiles -> MAGENTA, feather BAND -> GREEN (bright at the boundary, dimming inward), plain
// observed texture -> unchanged. In the render, note whether the jagged seam lies on the
// MAGENTA<->GREEN edge (fill/band seam), the GREEN<->normal edge (band inner edge), or entirely
// within normal texture (a camera-to-camera OBSERVED seam, unrelated to the fill). Set BOTH back to
// 0 for a normal render. Cheap: the stats loop only runs over the feather band.
#ifndef TEXTURE_DATACOLOR_DIAG
#define TEXTURE_DATACOLOR_DIAG 1
#endif
#ifndef TEXTURE_DATACOLOR_DIAG_TINT
#define TEXTURE_DATACOLOR_DIAG_TINT 0
#endif

// TEXTURE_DATACOLOR_FILL_RES / TEXTURE_DATACOLOR_FILL_MAX_TILE: resolution (longest-side px)
// each connected fill component's tile is baked at, and the hard cap on a tile's dimension.
// The fill reads as a BLURRY BLOB largely because it was baked at only 96 px then magnified
// over the (drooping) skirt, so the mirrored real texture + injected grain are smeared to
// mush. Raising this bakes the fill sharper -> the reflection-padded real detail near the
// boundary and the grain actually resolve. Cost: more appended atlas rows (~(res/96)^2 for
// the fill area, which is only the rim), so more memory/atlas size. Lower if the atlas gets
// too big; the old values were 96 / 256.
#ifndef TEXTURE_DATACOLOR_FILL_RES
#define TEXTURE_DATACOLOR_FILL_RES 192
#endif
#ifndef TEXTURE_DATACOLOR_FILL_MAX_TILE
#define TEXTURE_DATACOLOR_FILL_MAX_TILE 384
#endif

// TEXTURE_FINAL_RESIZE: gates the final atlas downscale (the INTER_AREA resize
// that caps the atlas to nMaxTextureSize). Set to 0 to skip the cap and keep the
// full-resolution atlas -- tests whether the residual dark patch-boundary seam
// lines are caused by the downscale/mip bleed. 1 = normal.
#ifndef TEXTURE_FINAL_RESIZE
#define TEXTURE_FINAL_RESIZE 1
#endif

// TEXTURE_ENABLE_SHARPEN: gates the final unsharp-mask sharpening pass. Unsharp
// masking puts dark halos on every contrast edge, so a faint patch-boundary
// exposure step can be amplified into a crisp dark line. Set to 0 to skip
// sharpening entirely (regardless of fSharpnessWeight).
// TEST 1 (ACTIVE): sharpen DISABLED to check if the unsharp-mask halo is the
// source of the residual dark patch-boundary seam lines. Restore to 1 after.
#ifndef TEXTURE_ENABLE_SHARPEN
#define TEXTURE_ENABLE_SHARPEN 0
#endif

// TEXTURE_ENABLE_GLOBAL_SEAM / TEXTURE_ENABLE_LOCAL_SEAM: compile-time gates that
// AND with the runtime bGlobalSeamLeveling / bLocalSeamLeveling flags, so a pass
// can be disabled without changing the command line. GlobalSeamLeveling equalizes
// whole-patch EXPOSURE between patches (low-frequency); LocalSeamLeveling DRAWS a
// mean-colour seam line and Poisson-blends a ~3px ribbon (high-frequency). On a
// large near-uniform bright wall the local drawn seam line reads as a faint vein
// tracing patch boundaries if the two patches differ in exposure.
// TEST 2 (ACTIVE): local seam leveling DISABLED to check if the wall veins are the
// local drawn seams. Restore to 1 after.
// CANDIDATE FINAL: seam leveling RE-ENABLED (needed scene-wide). The trailer veins
// were amplified by sharpen (now off) + global seam over an underlying geometry
// patch-boundary ghost; gSmoothnessWeight was raised (1000->2000) to cut those
// boundaries so there is less to ghost/mis-level. If the trailer vein returns
// strongly, global seam is the amplifier -> set TEXTURE_ENABLE_GLOBAL_SEAM 0.
#ifndef TEXTURE_ENABLE_GLOBAL_SEAM
#define TEXTURE_ENABLE_GLOBAL_SEAM 1
#endif
#ifndef TEXTURE_ENABLE_LOCAL_SEAM
#define TEXTURE_ENABLE_LOCAL_SEAM 1
#endif

// TEXTURE_SEAM_MEM_BUDGET_GB: memory-budget cap on the LocalSeamLeveling worker count.
// Each worker holds, sized to the patch it is currently processing, 2x Image32F3
// (24 B/px) + an Image8U mask + the PoissonBlendingNoBias thread_local scratch
// (~77 B/px LIVE). The MEASURED peak footprint is ~4x that (~256 B/px): per-patch
// cv::Mat::create()/vector resize() churn transiently holds old+new during realloc,
// and the pseam:: vectors keep their largest-patch capacity across patches. Sized to
// the LARGEST patch x every core, this is the +20-30 GB TRANSIENT that sets the whole
// texturing peak on res-0 (large multi-MP patches x many threads). The cap limits
// workers so that T * (maxPatchArea * ~256 B) stays under this many GB. Semantics-
// neutral: patches are seam-corrected independently, so fewer workers changes only
// parallelism, never the output pixels. 0 disables the cap (use every core).
#ifndef TEXTURE_SEAM_MEM_BUDGET_GB
#define TEXTURE_SEAM_MEM_BUDGET_GB 8
#endif

// TEXTURE_CROP_IMAGES: after view-selection, each source image only contributes the
// bounding box of the faces assigned to it (~0.6 MP of an ~18.6 MP original), yet the
// whole full-res image sits in RAM for the entire run (~20 GB across 367 images on a
// high-res aerial set). When 1: (a) ListCameraFaces frees each image right after its
// view (so that stage holds only ~nThreads images, not all of them); (b) GenerateTexture
// projects patch rects from the stored image DIMENSIONS (camera is intact, pixels gone);
// (c) right after projection each image is reloaded and CROPPED to the union of its patch
// rects, and those rects are rebased to the crop origin. Total resident image memory then
// drops ~30x (20 GB -> ~0.7 GB) with ZERO resolution loss -- crops are 1:1 from the full
// source. Coordinate-transparent: every downstream stage samples via rect.tl(), and
// rebasing rect by -union.tl() makes proj = faceTexcoords + rect.tl() land in crop-local
// coords, so seam leveling / merge / assemble need no changes. Cost: one extra image
// decode pass (~25 s on 367 imgs). 0 = keep all full-res images resident (previous behaviour).
#ifndef TEXTURE_CROP_IMAGES
#define TEXTURE_CROP_IMAGES 1
#endif

// uncomment to use SparseLU for solving the linear systems
// (should be faster, but not working on old Eigen)
#if !defined(EIGEN_DEFAULT_TO_ROW_MAJOR) || EIGEN_WORLD_VERSION>3 || (EIGEN_WORLD_VERSION==3 && EIGEN_MAJOR_VERSION>2)
#define TEXOPT_SOLVER_SPARSELU
#endif

// method used to try to detect outlier face views
// (should enable more consistent textures, but it is not working)
#define TEXOPT_FACEOUTLIER_NA 0
#define TEXOPT_FACEOUTLIER_MEDIAN 1
#define TEXOPT_FACEOUTLIER_GAUSS_DAMPING 2
#define TEXOPT_FACEOUTLIER_GAUSS_CLAMPING 3
#define TEXOPT_FACEOUTLIER TEXOPT_FACEOUTLIER_GAUSS_CLAMPING

// method used to find optimal view per face
#define TEXOPT_INFERENCE_LBP 1
#define TEXOPT_INFERENCE_TRWS 2
#define TEXOPT_INFERENCE TEXOPT_INFERENCE_LBP
#define INCREASE_PATCHES

static const Scene* gSceneForSmoothness = nullptr;

// inference algorithm
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
// LBP message-passing sweep cap. Measured on a 2.75M-face scene: the MRF energy
// (dominated by the Potts seam term) is HIGHLY sensitive to this -- energy@25
// was 33% higher than energy@50, i.e. materially more patch fragmentation. The
// tail sweeps are NOT cosmetic here; they consolidate labels / remove seams
// (even 50 is not fully converged). Keep at 50 for quality; lower ONLY after
// A/B-ing the rendered texture and confirming no added fragmentation on THIS
// dataset.
#ifndef LBP_MAX_ITERS
#define LBP_MAX_ITERS 50u
#endif
#include "../Math/LBP.h"
namespace MVS {
typedef LBPInference::NodeID NodeID;
// Potts model as smoothness function
LBPInference::EnergyType STCALL SmoothnessPotts(LBPInference::NodeID, LBPInference::NodeID, LBPInference::LabelID l1, LBPInference::LabelID l2) {
	return l1 == l2 && l1 != 0 && l2 != 0 ? LBPInference::EnergyType(0) : LBPInference::EnergyType(LBPInference::MaxEnergy);
}

static LBPInference::EnergyType gSmoothnessWeight = 2000;

#ifdef INCREASE_PATCHES
static LBPInference::EnergyType SmoothnessPottsStrong(
	LBPInference::NodeID n1,
	LBPInference::NodeID n2,
	LBPInference::LabelID l1,
	LBPInference::LabelID l2)
{
	if (l1 == l2) {
		return 0;
	}

	const Normal& N1 = gSceneForSmoothness->mesh.faceNormals[n1];
	const Normal& N2 = gSceneForSmoothness->mesh.faceNormals[n2];

	float cosAngle = N1.dot(N2);
	if (cosAngle < 0.0f) {
		cosAngle = 0.0f;
	}

	// CRACK-NETWORK FIX: the previous version applied a 0.8 cosine GATE and then
	// rescaled [0.8,1.0] -> [0,1], so any pair of faces whose smoothed normals
	// differed by more than ~37deg got only the tiny switchPenalty (~2), and even
	// gently-curved / normal-noisy neighbours (cos 0.85-0.95, extremely common on
	// real MVS road/ground meshes) got just ~250-750. With the rank-based data
	// cost that was frequently cheaper than switching cameras, so large flat
	// surfaces SHATTERED into hundreds of tiny patches whose seams render as the
	// crack network. We now keep a STRONG constant floor (switching a label is
	// always expensive) PLUS a coplanarity bonus, so the graph cut still prefers
	// to place seams at genuine geometry creases but the surface no longer
	// fragments. Tuning: lower gSmoothnessWeight to permit more/smaller patches,
	// raise it for even fewer.
	const float baseSwitchPenalty = (float)gSmoothnessWeight;            // always-on strong floor
	const float coplanarBonus     = (float)gSmoothnessWeight * cosAngle; // extra "glue" on flat surfaces

	const float w = baseSwitchPenalty + coplanarBonus;

	// If EnergyType is integer, rounding is better than truncation
	return (LBPInference::EnergyType)(w + 0.5f);
}
#else
static LBPInference::EnergyType SmoothnessPottsStrong(
	LBPInference::NodeID n1,
	LBPInference::NodeID n2,
	LBPInference::LabelID l1,
	LBPInference::LabelID l2)
{
	if (l1 == l2 && l1 != 0 && l2 != 0)
		return 0;

	const Normal& N1 = gSceneForSmoothness->mesh.faceNormals[n1];
	const Normal& N2 = gSceneForSmoothness->mesh.faceNormals[n2];

	float cosAngle = N1.dot(N2);
	cosAngle = std::max(0.0f, cosAngle);

	float w = gSmoothnessWeight * cosAngle;

	return (LBPInference::EnergyType)w;
}
#endif

}
#endif
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
#include "../Math/TRWS/MRFEnergy.h"
namespace MVS {
// TRWS MRF energy using Potts model
typedef unsigned NodeID;
typedef unsigned LabelID;
typedef TypePotts::REAL EnergyType;
static const EnergyType MaxEnergy(1);
struct TRWSInference {
	typedef MRFEnergy<TypePotts> MRFEnergyType;
	typedef MRFEnergy<TypePotts>::Options MRFOptions;

	CAutoPtr<MRFEnergyType> mrf;
	CAutoPtrArr<MRFEnergyType::NodeId> nodes;

	inline TRWSInference() {}
	void Init(NodeID nNodes, LabelID nLabels) {
		mrf = new MRFEnergyType(TypePotts::GlobalSize(nLabels));
		nodes = new MRFEnergyType::NodeId[nNodes];
	}
	inline bool IsEmpty() const {
		return mrf == NULL;
	}
	inline void AddNode(NodeID n, const EnergyType* D) {
		nodes[n] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
	}
	inline void AddEdge(NodeID n1, NodeID n2) {
		mrf->AddEdge(nodes[n1], nodes[n2], TypePotts::EdgeData(MaxEnergy));
	}
	EnergyType Optimize() {
		MRFOptions options;
		options.m_eps = 0.005;
		options.m_iterMax = 1000;
		#if 1
		EnergyType lowerBound, energy;
		mrf->Minimize_TRW_S(options, lowerBound, energy);
		#else
		EnergyType energy;
		mrf->Minimize_BP(options, energy);
		#endif
		return energy;
	}
	inline LabelID GetLabel(NodeID n) const {
		return mrf->GetSolution(nodes[n]);
	}
};
}
#endif

#if defined(_MSC_VER)
#define DEBUG_BREAK() __debugbreak()
#else
#define DEBUG_BREAK() __builtin_trap()
#endif

#define HARD_ASSERT(cond) \
  do { \
    if (!(cond)) { \
      (void)__FILE__; \
      (void)__LINE__; \
      DEBUG_BREAK(); \
    } \
  } while (0)


// S T R U C T S ///////////////////////////////////////////////////

typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;
typedef Mesh::TexCoord TexCoord;

typedef int MatIdx;
typedef Eigen::Triplet<float,MatIdx> MatEntry;
typedef Eigen::SparseMatrix<float, Eigen::ColMajor, MatIdx> SparseMat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor, MatIdx> SparseMatRM;

enum Mask {
	empty = 0,
	border = 128,
	interior = 255
};

struct MeshTexture {
	// used to render the surface to a view camera
	typedef TImage<cuint32_t> FaceMap;
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		FIndex idxFace;
		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap) {}
		void Clear() {
			Base::Clear();
			faceMap.fill(NO_ID);
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
			const Depth z(ComputeDepth(t, pbary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = idxFace;
			}
		}
	};

	// used to represent a pixel color
	typedef Point3f Color;
	typedef CLISTDEF0(Color) Colors;

	// used to store info about a face (view, quality)
	struct FaceData {
		IIndex idxView;// the view seeing this face
		float quality; // how well the face is seen by this view
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		Color color; // additionally store mean color (used to remove outliers)
		#endif
	};
	typedef cList<FaceData,const FaceData&,0,8,uint32_t> FaceDataArr; // store information about one face seen from several views
	typedef cList<FaceDataArr,const FaceDataArr&,2,1024,FIndex> FaceDataViewArr; // store data for all the faces of the mesh

	typedef cList<Mesh::FaceIdxArr, const Mesh::FaceIdxArr&,2,1024, FIndex> VirtualFaceIdxsArr; // store face indices for each virtual face

	// used to assign a view to a face
	typedef uint32_t Label;
	typedef cList<Label,Label,0,1024,FIndex> LabelArr;

	// represents a texture patch
	struct TexturePatch {
		Label label; // view index
		Mesh::FaceIdxArr faces; // indices of the faces contained by the patch
		RectsBinPack::Rect rect; // the bounding box in the view containing the patch
		std::vector<MatIdx> vertexRows;   // same size as faces.size() * 3
#if TEXTURE_CROP_IMAGES
		Image8U3 image; // per-patch private crop of the source image (rect-local, rect.tl()==0)
#endif
	};
	typedef cList<TexturePatch,const TexturePatch&,1,1024,FIndex> TexturePatchArr;

	// used to optimize texture patches
	struct SeamVertex
	{
		struct Patch
		{
			struct Edge
			{
				uint32_t idxSeamVertex;
				FIndex idxFace;

				inline Edge() {}

				inline Edge(uint32_t _idxSeamVertex)
					: idxSeamVertex(_idxSeamVertex),
					idxFace(NO_ID) {
				}

				inline Edge(uint32_t _idxSeamVertex, FIndex _idxFace)
					: idxSeamVertex(_idxSeamVertex),
					idxFace(_idxFace) {
				}

				inline bool operator==(uint32_t _idxSeamVertex) const {
					return idxSeamVertex == _idxSeamVertex;
				}
			};

			typedef cList<Edge, const Edge&, 0, 4, uint32_t> Edges;

			uint32_t idxPatch;
			Point2f proj;
			Edges edges;

			inline Patch() {}
			inline Patch(uint32_t _idxPatch) : idxPatch(_idxPatch) {}

			inline bool operator==(uint32_t _idxPatch) const {
				return idxPatch == _idxPatch;
			}

			struct PatchEdgeLookup
			{
				boost::container::small_vector<uint32_t, 8> seamVertexIds;
				boost::container::small_vector<uint16_t, 8> indices;

				inline void clear()
				{
					seamVertexIds.clear();
					indices.clear();
				}

				inline void reserve(uint32_t n)
				{
					seamVertexIds.reserve(n);
					indices.reserve(n);
				}

				inline void Insert(uint32_t id, uint16_t idx)
				{
					// no overwrite logic needed in your usage,
					// but keep identical semantics
					for (uint32_t i = 0; i < seamVertexIds.size(); ++i) {
						if (seamVertexIds[i] == id) {
							indices[i] = idx;
							return;
						}
					}
					seamVertexIds.push_back(id);
					indices.push_back(idx);
				}

				inline const uint16_t* Find(uint32_t id) const
				{
					for (uint32_t i = 0; i < seamVertexIds.size(); ++i) {
						if (seamVertexIds[i] == id)
							return &indices[i];
					}
					return nullptr;
				}
			};

			PatchEdgeLookup edgeLookup;
		};

		struct PatchContainer {
			std::vector<uint32_t> patchIds;
			std::vector<uint16_t> indices;

			inline void clear() {
				patchIds.clear();
				indices.clear();
			}

			inline void reserve(uint32_t cnt) {
				patchIds.reserve(cnt);
				indices.reserve(cnt);
			}

			inline void Insert(uint32_t patch, uint16_t idx) {
				for (uint32_t i = 0; i < patchIds.size(); ++i) {
					if (patchIds[i] == patch) {
						indices[i] = idx;
						return;
					}
				}
				patchIds.push_back(patch);
				indices.push_back(idx);
			}

			inline uint16_t At(uint32_t patch) const {
				const uint16_t* p = Find(patch);
				ASSERT(p != nullptr);
				return *p;
			}

			inline const uint16_t* Find(uint32_t patch) const {
				for (uint32_t i = 0; i < patchIds.size(); ++i)
					if (patchIds[i] == patch)
						return &indices[i];
				return nullptr;
			}
		};

		VIndex idxVertex;
		std::vector<Patch> patches;
		PatchContainer patchIndexLookup;

		SeamVertex() = default;
		inline SeamVertex(uint32_t _idxVertex) : idxVertex(_idxVertex) {}

		inline bool operator==(uint32_t _idxVertex) const {
			return idxVertex == _idxVertex;
		}

		inline void SortByPatchIndex(std::vector<uint32_t>& indices) const
		{
			const size_t n = patches.size();

			indices.resize(n);

			for (size_t i = 0; i < n; ++i)
				indices[i] = (uint32_t)i;

			std::sort(indices.begin(), indices.end(),
				[&](uint32_t a, uint32_t b)
				{
					return patches[a].idxPatch < patches[b].idxPatch;
				});
		}

		Patch& GetPatch(uint32_t idxPatch) {
			for (Patch& p : patches) {
				if (p.idxPatch == idxPatch) {
					return p;
				}
			}
			patches.emplace_back(idxPatch);
			return patches.back();
		}

		const Patch* FindPatch(uint32_t idxPatch) const {
			for (const Patch& p : patches) {
				if (p.idxPatch == idxPatch) {
					return &p;
				}
			}
			return nullptr;
		}

		Patch* FindPatch(uint32_t idxPatch) {
			for (Patch& p : patches) {
				if (p.idxPatch == idxPatch) {
					return &p;
				}
			}
			return nullptr;
		}
	};

	typedef cList<SeamVertex,const SeamVertex&,1,256,uint32_t> SeamVertices;

	// used to sample seam edges
	typedef TAccumulator<Color> AccumColor;
	typedef Sampler::Linear<float> Sampler;
	struct SampleImage {
		AccumColor accumColor;
		const Image8U3& image;
		const Sampler sampler;

		inline SampleImage(const Image8U3& _image) : image(_image), sampler() {}
		// sample the edge with linear weights
		void AddEdge(const TexCoord& p0, const TexCoord& p1) {
			const TexCoord p01(p1 - p0);
			const float length(norm(p01));
			ASSERT(length > 0.f);
			const int nSamples(ROUND2INT(MAXF(length, 1.f) * 2.f)-1);
			AccumColor edgeAccumColor;
			for (int s=0; s<nSamples; ++s) {
				const float len(static_cast<float>(s) / nSamples);
				const TexCoord samplePos(p0 + p01 * len);
				const Color color(image.sample<Sampler,Color>(sampler, samplePos));
				edgeAccumColor.Add(RGB2YCBCR(color), 1.f-len);
			}
			accumColor.Add(edgeAccumColor.Normalized(), length);
		}
		// returns accumulated color
		Color GetColor() const {
			return accumColor.Normalized();
		}
	};

	// used to interpolate adjustments color over the whole texture patch
	typedef TImage<Color> ColorMap;


public:
	MeshTexture(Scene& _scene, unsigned _nResolutionLevel=0, unsigned _nMinResolution=640);
	~MeshTexture();

	void ListVertexFaces();

	bool ListCameraFaces(FaceDataViewArr&, float fOutlierThreshold, const IIndexArr& views);

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	bool FaceOutlierDetection(FaceDataArr& faceDatas, float fOutlierThreshold) const;
	#endif
	
	void CreateVirtualFaces(const FaceDataViewArr& facesDatas, FaceDataViewArr& virtualFacesDatas, VirtualFaceIdxsArr& virtualFaces, unsigned minCommonCameras=2, float thMaxNormalDeviation=25.f) const;
	IIndexArr SelectBestView(const FaceDataArr& faceDatas, FIndex fid, unsigned minCommonCameras, float ratioAngleToQuality) const;

	bool FaceViewSelection(LabelArr& labels, unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness, const IIndexArr& views);
	
	void CreateSeamVertices();
	void GlobalSeamLeveling();
	void LocalSeamLeveling();
	void GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize);

	// Source pixels for a patch: its private rect-local crop when TEXTURE_CROP_IMAGES is on
	// (built in GenerateTexture right after projection), else the shared full source image.
	// Centralizes the switch for every seam-leveling / assemble read.
	inline const Image8U3& PatchSrcImage(uint32_t idxPatch) const {
#if TEXTURE_CROP_IMAGES
		return texturePatches[idxPatch].image;
#else
		return images[texturePatches[idxPatch].label].image;
#endif
	}

	template <typename PIXEL>
	static inline PIXEL RGB2YCBCR(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		return PIXEL(
			v[0] * T(0.299) + v[1] * T(0.587) + v[2] * T(0.114),
			v[0] * T(-0.168736) + v[1] * T(-0.331264) + v[2] * T(0.5) + T(128),
			v[0] * T(0.5) + v[1] * T(-0.418688) + v[2] * T(-0.081312) + T(128)
		);
	}
	template <typename PIXEL>
	static inline PIXEL YCBCR2RGB(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		const T v1(v[1] - T(128));
		const T v2(v[2] - T(128));
		return PIXEL(
			v[0]/* * T(1) + v1 * T(0)*/ + v2 * T(1.402),
			v[0]/* * T(1)*/ + v1 * T(-0.34414) + v2 * T(-0.71414),
			v[0]/* * T(1)*/ + v1 * T(1.772)/* + v2 * T(0)*/
		);
	}


protected:
	static void ProcessMask3(Image8U& mask);
	static void PoissonBlendingNoBias(const Image32F3& src, Image32F3& dst, const Image8U& mask);


public:
	const unsigned nResolutionLevel; // how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // how many times to scale down the images before mesh optimization

	// store found texture patches
	TexturePatchArr texturePatches;

	// used to compute the seam leveling
	PairIdxArr seamEdges; // the (face-face) edges connecting different texture patches
	Mesh::FaceIdxArr components; // for each face, stores the texture patch index to which belongs
	//IndexArr mapIdxPatch; // remap texture patch indices after invalid patches removal
	SeamVertices seamVertices; // array of vertices on the border between two or more patches

	// valid the entire time
	Mesh::VertexFacesArr& vertexFaces; // for each vertex, the list of faces containing it
	BoolArr& vertexBoundary; // for each vertex, stores if it is at the boundary or not
	Mesh::FaceFacesArr& faceFaces; // for each face, the list of adjacent faces, NO_ID for border edges (optional)
	Mesh::TexCoordArr& faceTexcoords; // for each face, the texture-coordinates of the vertices
	Image8U3& textureDiffuse; // texture containing the diffuse color

	// constant the entire time
	Mesh::VertexArr& vertices;
	Mesh::FaceArr& faces;
	ImageArr& images;

	Scene& scene; // the mesh vertices and faces
};

MeshTexture::MeshTexture(Scene& _scene, unsigned _nResolutionLevel, unsigned _nMinResolution)
	:
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	faceFaces(_scene.mesh.faceFaces),
	faceTexcoords(_scene.mesh.faceTexcoords),
	textureDiffuse(_scene.mesh.textureDiffuse),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	images(_scene.images),
	scene(_scene)
{
}
MeshTexture::~MeshTexture()
{
	vertexFaces.Release();
	vertexBoundary.Release();
	faceFaces.Release();
}

__forceinline float FastLog1pAccurate(float x) {
	float y = x + 1.0f;

	// Extract exponent
	union {
		float f;
		uint32_t i;
	} u = { y };

	int exp = int((u.i >> 23) & 255) - 127;

	// Normalize mantissa to [1,2)
	u.i = (u.i & 0x7FFFFF) | 0x3F800000;
	float m = u.f - 1.0f;

	// Minimax polynomial for log(1+m) on [0,1]
	const float c1 = 0.999996f;
	const float c2 = -0.499874f;
	const float c3 = 0.331799f;
	const float c4 = -0.240733f;
	const float c5 = 0.167654f;

	float m2 = m * m;
	float m3 = m2 * m;
	float m4 = m3 * m;
	float m5 = m4 * m;

	float logMantissa =
		c1 * m + c2 * m2 + c3 * m3 + c4 * m4 + c5 * m5;

	const float ln2 = 0.69314718056f;

	return exp * ln2 + logMantissa;
}

// extract array of triangles incident to each vertex
// and check each vertex if it is at the boundary or not
void MeshTexture::ListVertexFaces()
{
	scene.mesh.EmptyExtra();
	scene.mesh.ListIncidenteFaces();
	scene.mesh.ListBoundaryVertices();
	scene.mesh.ListIncidenteFaceFaces();
}

struct CamVert {
	float x, y, z, invZ;
};

struct CameraRenderData {
	std::vector<CamVert> verts;       // per-camera compact camera-space vertices
	std::vector<Face> faces;          // compact faces using LOCAL vertex indices
	std::vector<uint32_t> globalFace; // localFaceIndex -> global face index
	std::vector<uint32_t> globalVert; // localVertIndex -> global vertex index
};

void PreprocessCameraFaces(
	const Mesh::FaceIdxArr& cameraFaces,   // global face indices visible to this camera
	const Mesh::FaceArr& faces,            // global mesh faces
	const Mesh::VertexArr& vertices,       // global mesh vertices (world space)
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
	// 1. Build compact vertex list (used[])
	// ------------------------------------------------------------------
	std::vector<uint32_t>& used = out.globalVert;
	used.clear();

	if (used.capacity() < numFaces * 2)
		used.reserve(numFaces * 2);

	for (uint32_t fIdx : cameraFaces) {
		const Face& gf = faces[fIdx];

		// Unroll manually for speed
		uint32_t gv0 = gf[0];
		uint32_t gv1 = gf[1];
		uint32_t gv2 = gf[2];

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

	// ------------------------------------------------------------------
	// 2. Resize camera-space vertices buffer (computed later)
	// ------------------------------------------------------------------
	out.verts.resize(used.size());   // no clear; overwritten later

	// ------------------------------------------------------------------
	// 3. Build per-camera local face list + global face index + normals
	// ------------------------------------------------------------------
	out.faces.resize(numFaces);
	out.globalFace.resize(numFaces);

	size_t idx = 0;
	for (uint32_t fIdx : cameraFaces) {
		const Face& gf = faces[fIdx];

		// Localize face (remap global indices)
		Face& lf = out.faces[idx];
		lf[0] = remap[gf[0]];
		lf[1] = remap[gf[1]];
		lf[2] = remap[gf[2]];

		out.globalFace[idx] = fIdx;

		idx++;
	}
}

void UpdateCameraVertsAndNormals(
	const Mesh::VertexArr& vertices,     // global world vertices
	const Camera& camera,
	CameraRenderData& out
) {
	const size_t numVerts = out.globalVert.size();
	const size_t numFaces = out.faces.size();

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

		if (zc < 1e-6f) zc = 1e-6f;

		out.verts[i] = { xc, yc, zc, 1.f / zc };
	}
}

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

inline int CeilPos(float y) {
	int iy = (int)y;
	return iy + (y > float(iy));
}

// extract array of faces viewed by each image
// Cap on candidate views (observations) kept per face in ListCameraFaces. On
// high-overlap scenes a face can be seen by 60+ views; the MRF view-selection
// only ever picks the best (winner is ~top-1 by quality; the smoothness term
// rarely reaches past the top few), so keeping the long low-quality tail only
// bloats facesDatas and the MRF graph. Capping to the top-N by quality bounds
// that memory ~linearly with the texture essentially unchanged.
// 0 = unlimited (original behaviour).
// NOTE: measured NO benefit on RichmondHistoric (facesDatas is small; the peak is
// the resident image/mesh base + seam-leveling transient, not per-face-per-view),
// and it is output-affecting, so it defaults OFF. Left as a knob for scenes that
// genuinely have a huge views/face tail.
#ifndef TEXTURE_MAX_VIEWS_PER_FACE
#define TEXTURE_MAX_VIEWS_PER_FACE 0
#endif

template<typename TYPE1, typename TYPE2 = TYPE1>
inline TYPE2 ComputeAngle2(const TYPE1* V1, const TYPE1* V2) {
	return CLAMP(TYPE2((V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]) / FastSqrtS((V1[0] * V1[0] + V1[1] * V1[1] + V1[2] * V1[2]) * (V2[0] * V2[0] + V2[1] * V2[1] + V2[2] * V2[2]))), TYPE2(-1), TYPE2(1));
} // ComputeAngle

bool MeshTexture::ListCameraFaces(FaceDataViewArr& facesDatas, float fOutlierThreshold, const IIndexArr& _views)
{
	TEX_PROFILE_SCOPE("ListCameraFaces (total)");
	// create faces octree
	TEX_PROFILE_BEGIN(_tLcfOctree);
	Mesh::Octree octree;
	Mesh::FacesInserter::CreateOctree(octree, scene.mesh);
	TEX_PROFILE_END(_tLcfOctree, "ListCameraFaces: build octree");

	// extract array of faces viewed by each image
	IIndexArr views(_views);
	if (views.empty()) {
		views.resize(images.size());
		std::iota(views.begin(), views.end(), IIndex(0));
	}
	facesDatas.Resize(faces.size());
	Util::Progress progress(_T("Initialized views"), views.size());
	typedef float real;
	TImage<real> imageGradMag;
	TImage<real>::EMat mGrad[2];
	FaceMap faceMap;
	DepthMap depthMap;

	struct FaceAccum {
		float quality;
		uint32_t area;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		// Integer per-channel color sums (c0,c1,c2), averaged to a float mean once per
		// face at finalize. Accumulating raw uint8 channels avoids a per-pixel int->float
		// convert (cvtdq2ps x3) and turns each channel into a single add-to-memory.
		// Bound: 255 * area must fit uint32 -> safe up to ~16.7M rasterized px per face.
		uint32_t color[3];
#endif
	};

	struct FaceOut {
		FIndex idxFace;
		FaceData data; // includes idxView, quality, and optional color
	};
	std::vector<std::vector<FaceOut>> perViewOut;
	perViewOut.resize(views.size());

	// Since we are controlling the threading per-view, don't let cv thread
	// when performing its work.
	const int prevCvThreads = cv::getNumThreads();
	cv::setNumThreads(0);

	TEX_PROFILE_BEGIN(_tLcfCenters);
	static std::vector<Point3f> gFaceCenter;
	gFaceCenter.resize(faces.size());

#pragma omp parallel for schedule(static)
	for (int64_t f = 0; f < (int64_t)faces.size(); ++f) {
		const Face& fc = faces[(size_t)f];
		const Vertex& a = vertices[fc[0]];
		const Vertex& b = vertices[fc[1]];
		const Vertex& c = vertices[fc[2]];
		gFaceCenter[(size_t)f] = Point3f(
			(a.x + b.x + c.x) * (1.0f / 3.0f),
			(a.y + b.y + c.y) * (1.0f / 3.0f),
			(a.z + b.z + c.z) * (1.0f / 3.0f)
		);
	}
	TEX_PROFILE_END(_tLcfCenters, "ListCameraFaces: face centers");

	TEX_PROFILE_BEGIN(_tLcfPerView);
#ifdef TEXOPT_USE_OPENMP
	bool bAbort(false);
#pragma omp parallel for private(imageGradMag, mGrad, faceMap, depthMap)
	for (int_t idx = 0; idx < (int_t)views.size(); ++idx) {
#pragma omp flush (bAbort)
		if (bAbort) {
			++progress;
			continue;
		}
		const IIndex idxView(views[(IIndex)idx]);
#else
	for (IIndex idxView : views) {
#endif
		Image& imageData = images[idxView];
		if (!imageData.IsValid()) {
			++progress;
			continue;
		}
		// load image
		unsigned level(nResolutionLevel);
		const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
		if ((imageData.image.empty() || MAXF(imageData.width, imageData.height) != imageSize) && !imageData.ReloadImage(imageSize)) {
#ifdef TEXOPT_USE_OPENMP
			bAbort = true;
#pragma omp flush (bAbort)
			continue;
#else
			return false;
#endif
		}
		imageData.UpdateCamera(scene.platforms);
		// compute gradient magnitude
		imageData.image.toGray(imageGradMag, cv::COLOR_BGR2GRAY, true);
		cv::Mat grad[2];
		mGrad[0].resize(imageGradMag.rows, imageGradMag.cols);
		grad[0] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[0].data());
		mGrad[1].resize(imageGradMag.rows, imageGradMag.cols);
		grad[1] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[1].data());
#if 1
		cv::Sobel(imageGradMag, grad[0], cv::DataType<real>::type, 1, 0, 3, 1.0 / 8.0);
		cv::Sobel(imageGradMag, grad[1], cv::DataType<real>::type, 0, 1, 3, 1.0 / 8.0);
#elif 1
		const TMatrix<real, 3, 5> kernel(CreateDerivativeKernel3x5());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
#else
		const TMatrix<real, 5, 7> kernel(CreateDerivativeKernel5x7());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
#endif
		(TImage<real>::EMatMap)imageGradMag = (mGrad[0].cwiseAbs2() + mGrad[1].cwiseAbs2()).cwiseSqrt();
		// Free the two full-res Sobel planes now (per-thread, 8 B/px each): they are
		// not used again this iteration, and releasing them BEFORE the concurrent
		// rasterization (faceMap/depthMap) below lowers the multi-thread memory peak.
		// Re-grown by resize() on the next image (1 alloc/image, negligible).
		{ TImage<real>::EMat empty0, empty1; mGrad[0].swap(empty0); mGrad[1].swap(empty1); }
		// apply some blur on the gradient to lower noise/glossiness effects onto face-quality score
		cv::GaussianBlur(imageGradMag, imageGradMag, cv::Size(15, 15), 0, 0, cv::BORDER_DEFAULT);
		// select faces inside view frustum
		Mesh::FaceIdxArr cameraFaces;
		Mesh::FacesInserter inserter(cameraFaces);
		typedef TFrustum<float, 5> Frustum;
		const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
		octree.Traverse(frustum, inserter);
		// project all triangles in this view and keep the closest ones
		faceMap.create(imageData.height, imageData.width);
		depthMap.create(imageData.height, imageData.width);
		RasterMesh rasterer(vertices, imageData.camera, depthMap, faceMap);
		rasterer.Clear();
		// Rasterize storing a LOCAL face index (position in cameraFaces) into faceMap
		// instead of the global face id. This lets the per-pixel accumulation below
		// index a dense, cameraFaces-sized array directly instead of hashing every
		// rasterized pixel into a robin_map (millions of hash ops per view otherwise).
		const uint32_t numCameraFaces((uint32_t)cameraFaces.size());
		for (uint32_t li = 0; li < numCameraFaces; ++li) {
			const Face& facet = faces[cameraFaces[li]];
			rasterer.idxFace = (FIndex)li;
			rasterer.Project(facet);
		}

		// accumulate per-face quality/area/color over the rasterized pixels; the dense
		// buffer is indexed by the local face index written into faceMap above.
		std::vector<FaceAccum> acc(numCameraFaces); // Value initializes acc as well.

		for (int j = 0; j < faceMap.rows; ++j) {
			const FIndex* __restrict fm = faceMap.ptr<FIndex>(j);
			const real* __restrict gm = imageGradMag.ptr<real>(j);
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
			const auto* __restrict im = imageData.image.ptr<decltype(imageData.image)::value_type>(j);
#endif

			for (int i = 0; i < faceMap.cols; ++i) {
				const uint32_t li = (uint32_t)fm[i];
				if (li == (uint32_t)NO_ID) {
					continue;
				}
				FaceAccum& a = acc[li];
				a.quality += (float)gm[i];
				a.area++;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				const auto& px = im[i];
				a.color[0] += px[0];
				a.color[1] += px[1];
				a.color[2] += px[2];
#endif
			}
		}

		// ---- angle adjustment per face ----
		// Build output list for this view:
		std::vector<FaceOut> out;
		out.reserve(numCameraFaces);
		for (uint32_t li = 0; li < numCameraFaces; ++li) {
			FaceAccum& a = acc[li];
			if (a.area == 0)
				continue; // face had no rasterized pixel in this view
			const FIndex idxFace = cameraFaces[li];

			const Face& f = faces[idxFace];
			const auto& faceCenter = gFaceCenter[idxFace];
			const Point3f camDir(Cast<Mesh::Type>(imageData.camera.C) - faceCenter);
			const Normal& faceNormal = scene.mesh.faceNormals[idxFace];
			const float rawCosFaceCam(ComputeAngle2(camDir.ptr(), faceNormal.ptr()));
			// skip observations where the camera looks at the back of the face
			// (rawCos <= 0); keeping them causes wrong-side texturing on edges.
			if (rawCosFaceCam <= 0.f) {
				continue;
			}
			a.quality *= SQUARE(rawCosFaceCam);

			FaceOut fo;
			fo.idxFace = idxFace;
			fo.data.idxView = idxView;
			fo.data.quality = a.quality;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
			const float inv = 1.f / (float)a.area;
			const Color c((float)a.color[0] * inv, (float)a.color[1] * inv, (float)a.color[2] * inv);
			fo.data.color = RGB2YCBCR(c);
#endif
			out.push_back(fo);
		}

		perViewOut[(size_t)idx].swap(out);

#if TEXTURE_CROP_IMAGES
		// Crop-images mode: this view's pixels are fully consumed. Free them now so
		// ListCameraFaces holds only ~nThreads images resident instead of all of them.
		// GenerateTexture reloads each image and crops it to its used region. The
		// width/height/camera members stay intact (release() only frees the pixel Mat),
		// so projection can still compute patch rects from the stored dimensions.
		imageData.image.release();
#endif

		++progress;
	}

#ifdef TEXOPT_USE_OPENMP
	if (bAbort)
		return false;
#endif
	TEX_PROFILE_END(_tLcfPerView, "ListCameraFaces: per-view rasterize+quality");

	// [MEM-DECOMP] TEMPORARY: decompose the ListCameraFaces resident footprint so we
	// can target the ~20 GB base that stage-level probes leave unexplained. Measures
	// the ACTUAL bytes of the two big suspects at their peak (perViewOut is fully
	// built here, just before the merge frees it). Remove once the hog is identified.
	{
		size_t pvBytes = 0, pvObs = 0;
		for (const auto& v : perViewOut) { pvBytes += v.capacity() * sizeof(FaceOut); pvObs += v.size(); }
		size_t imgBytes = 0; unsigned imgResident = 0;
		for (const Image& im : images) if (!im.image.empty()) { imgBytes += (size_t)im.image.total() * im.image.elemSize(); ++imgResident; }
		const double G = 1.0 / (1024.0*1024.0*1024.0);
		DEBUG_EXTRA("[MEM-DECOMP] perViewOut=%.2f GB (%zu obs, %zuB/obs) | images=%.2f GB (%u resident, %zuB/px*3) | mesh.faces=%u verts=%u",
			pvBytes*G, pvObs, sizeof(FaceOut), imgBytes*G, imgResident, (size_t)3, faces.GetSize(), vertices.GetSize());
		LogPeakMem("LCF: after per-view (perViewOut full)");
	}

	TEX_PROFILE_BEGIN(_tLcfMerge);
	// Pass 1: count how many FaceOut entries per face
	std::vector<uint32_t> addCount(facesDatas.GetSize(), 0);

	for (size_t idx = 0; idx < perViewOut.size(); ++idx) {
		const auto& out = perViewOut[idx];
		for (const FaceOut& fo : out) {
			++addCount[(size_t)fo.idxFace];
		}
	}

	// Number of candidate views kept per face (see TEXTURE_MAX_VIEWS_PER_FACE).
	const uint32_t capN = TEXTURE_MAX_VIEWS_PER_FACE;

	// Pass 2: reserve capacity in each FaceDataArr once (bounded by the cap so we
	// never allocate room for the long low-quality observation tail).
	for (size_t f = 0; f < (size_t)facesDatas.GetSize(); ++f) {
		uint32_t nAdd = addCount[f];
		if (nAdd == 0) {
			continue;
		}
		if (capN != 0 && nAdd > capN)
			nAdd = capN;
		FaceDataArr& arr = facesDatas[(FIndex)f];
		arr.Reserve(arr.GetSize() + nAdd);
	}

	// Pass 3: merge into facesDatas, CAPPING each face to its top-N observations
	// by quality, and freeing each per-view buffer AS IT IS CONSUMED. Without the
	// cap a face on a high-overlap scene keeps 60+ observations; the MRF only uses
	// the best, so the tail is pure memory. Without the per-view free, perViewOut
	// and facesDatas would both stay fully resident at the peak.
	for (size_t idx = 0; idx < perViewOut.size(); ++idx) {
		auto& out = perViewOut[idx];
		for (const FaceOut& fo : out) {
			FaceDataArr& arr = facesDatas[fo.idxFace];
			if (capN == 0 || arr.GetSize() < capN) {
				FaceData& fd = arr.AddEmpty();
				fd = fo.data;
			} else {
				// at capacity: keep the top-N by quality -- replace the lowest-quality
				// kept observation if this one is better (N is small so the scan is cheap)
				uint32_t worst = 0;
				float worstQ = arr[0].quality;
				for (uint32_t k = 1; k < arr.GetSize(); ++k) {
					if (arr[k].quality < worstQ) { worstQ = arr[k].quality; worst = k; }
				}
				if (fo.data.quality > worstQ)
					arr[worst] = fo.data;
			}
		}
		std::vector<FaceOut>().swap(out); // release this view's buffer immediately
	}
	std::vector<std::vector<FaceOut>>().swap(perViewOut); // drop the outer array too
	TEX_PROFILE_END(_tLcfMerge, "ListCameraFaces: merge into facesDatas");

	// [MEM-DECOMP] TEMPORARY: size facesDatas now that perViewOut is freed. Includes
	// both the live observations and the cList capacity slack (grow-by-doubling can
	// hold ~2x the live bytes). Remove once the hog is identified.
	{
		size_t fdData = 0, fdCap = 0, fdObs = 0;
		for (FIndex f = 0; f < facesDatas.GetSize(); ++f) {
			const FaceDataArr& a = facesDatas[f];
			fdObs += a.GetSize();
			fdData += (size_t)a.GetSize() * sizeof(FaceData);
			fdCap += (size_t)a.GetCapacity() * sizeof(FaceData);
		}
		const double G = 1.0 / (1024.0*1024.0*1024.0);
		DEBUG_EXTRA("[MEM-DECOMP] facesDatas live=%.2f GB cap=%.2f GB (%zu obs, %zuB/obs, %u faces headers=%.2f GB)",
			fdData*G, fdCap*G, fdObs, sizeof(FaceData), facesDatas.GetSize(),
			((size_t)facesDatas.GetSize()*sizeof(FaceDataArr))*G);
		LogPeakMem("LCF: after merge (facesDatas full)");
	}

	progress.close();

	// Restore cv's ability to thread.
	cv::setNumThreads(prevCvThreads);

#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		TEX_PROFILE_BEGIN(_tLcfOutlier);
		// try to detect outlier views for each face
		// (views for which the face is occluded by a dynamic object in the scene, ex. pedestrians)
		for (FaceDataArr& faceDatas : facesDatas)
			FaceOutlierDetection(faceDatas, fOutlierThreshold);
		TEX_PROFILE_END(_tLcfOutlier, "ListCameraFaces: outlier detection");
	}
#endif
	return true;
	}

// order the camera view scores with highest score first and return the list of first <minCommonCameras> cameras
// ratioAngleToQuality represents the ratio in witch we combine normal angle to quality for a face to obtain the selection score
//  - a ratio of 1 means only angle is considered
//  - a ratio of 0.5 means angle and quality are equally important
//  - a ratio of 0 means only camera quality is considered when sorting
IIndexArr MeshTexture::SelectBestView(const FaceDataArr& faceDatas, FIndex fid, unsigned minCommonCameras, float ratioAngleToQuality) const
{
	ASSERT(!faceDatas.empty());

	// Order views by descending absolute quality
	IIndexArr order(faceDatas.size());
	std::iota(order.begin(), order.end(), 0);

	order.Sort([&faceDatas](IIndex a, IIndex b) {
		return faceDatas[a].quality > faceDatas[b].quality;
		});

	unsigned n = MIN(minCommonCameras, (unsigned)faceDatas.size());
	if (n > 1) n = std::max(1u, n - 1);

	IIndexArr cameras(n);
	for (unsigned i = 0; i < n; ++i)
		cameras[i] = faceDatas[order[i]].idxView;

	return cameras;
}

static bool IsFaceVisible(const MeshTexture::FaceDataArr& faceDatas, const IIndexArr& cameraList) {
	size_t camFoundCounter(0);
	for (const MeshTexture::FaceData& faceData : faceDatas) {
		const IIndex cfCam = faceData.idxView;
		for (IIndex camId : cameraList) {
			if (cfCam == camId) {
				if (++camFoundCounter == cameraList.size())
					return true;	
				break;
			}
		}
	}
	return camFoundCounter == cameraList.size();
}

// build virtual faces with:
// - similar normal
// - high percentage of common images that see them
void MeshTexture::CreateVirtualFaces(const FaceDataViewArr& facesDatas, FaceDataViewArr& virtualFacesDatas, VirtualFaceIdxsArr& virtualFaces, unsigned minCommonCameras, float thMaxNormalDeviation) const
{
	const float ratioAngleToQuality(0.67f);
	const float cosMaxNormalDeviation(COS(FD2R(thMaxNormalDeviation)));
	Mesh::FaceIdxArr remainingFaces(faces.size());
	std::iota(remainingFaces.begin(), remainingFaces.end(), 0);
	std::vector<bool> selectedFaces(faces.size(), false);
	cQueue<FIndex, FIndex, 0> currentVirtualFaceQueue;
	std::unordered_set<FIndex> queuedFaces;
	do {
		const FIndex startPos = RAND() % remainingFaces.size();
		const FIndex virtualFaceCenterFaceID = remainingFaces[startPos];
		ASSERT(currentVirtualFaceQueue.IsEmpty());
		const Normal& normalCenter = scene.mesh.faceNormals[virtualFaceCenterFaceID];
		const FaceDataArr& centerFaceDatas = facesDatas[virtualFaceCenterFaceID];
		// select the common cameras
		Mesh::FaceIdxArr virtualFace;
		FaceDataArr virtualFaceDatas;
		if (centerFaceDatas.empty()) {
			virtualFace.emplace_back(virtualFaceCenterFaceID);
			selectedFaces[virtualFaceCenterFaceID] = true;
			const auto posToErase = remainingFaces.FindFirst(virtualFaceCenterFaceID);
			ASSERT(posToErase != Mesh::FaceIdxArr::NO_INDEX);
			remainingFaces.RemoveAtMove(posToErase);
		} else {
			const IIndexArr selectedCams = SelectBestView(centerFaceDatas, virtualFaceCenterFaceID, minCommonCameras, ratioAngleToQuality);
			currentVirtualFaceQueue.AddTail(virtualFaceCenterFaceID);
			queuedFaces.clear();
			do {
				const FIndex currentFaceId = currentVirtualFaceQueue.GetHead();
				currentVirtualFaceQueue.PopHead();
				// check for condition to add in current virtual face
				// normal angle smaller than thMaxNormalDeviation degrees
				const Normal& faceNormal = scene.mesh.faceNormals[currentFaceId];
				const float cosFaceToCenter(ComputeAngleN(normalCenter.ptr(), faceNormal.ptr()));
				if (cosFaceToCenter < cosMaxNormalDeviation)
					continue;
				// check if current face is seen by all cameras in selectedCams
				ASSERT(!selectedCams.empty());
				if (!IsFaceVisible(facesDatas[currentFaceId], selectedCams))
					continue;
				// remove it from remaining faces and add it to the virtual face
				{
					const auto posToErase = remainingFaces.FindFirst(currentFaceId);
					ASSERT(posToErase != Mesh::FaceIdxArr::NO_INDEX);
					remainingFaces.RemoveAtMove(posToErase);
					selectedFaces[currentFaceId] = true;
					virtualFace.push_back(currentFaceId);
				}
				// add all new neighbors to the queue
				const Mesh::FaceFaces& ffaces = faceFaces[currentFaceId];
				for (int i = 0; i < 3; ++i) {
					const FIndex fIdx = ffaces[i];
					if (fIdx == NO_ID)
						continue;
					if (!selectedFaces[fIdx] && queuedFaces.find(fIdx) == queuedFaces.end()) {
						currentVirtualFaceQueue.AddTail(fIdx);
						queuedFaces.emplace(fIdx);
					}
				}
			} while (!currentVirtualFaceQueue.IsEmpty());
			// compute virtual face quality and create virtual face
			for (IIndex idxView: selectedCams) {
				FaceData& virtualFaceData = virtualFaceDatas.AddEmpty();
				virtualFaceData.quality = 0;
				virtualFaceData.idxView = idxView;
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				virtualFaceData.color = Point3f::ZERO;
				#endif
				unsigned processedFaces(0);
				for (FIndex fid : virtualFace) {
					const FaceDataArr& faceDatas = facesDatas[fid];
					for (FaceData& faceData: faceDatas) {
						if (faceData.idxView == idxView) {
							virtualFaceData.quality += faceData.quality;
							#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
							virtualFaceData.color += faceData.color;
							#endif
							++processedFaces;
							break;
						}
					}
				}
				ASSERT(processedFaces > 0);
				virtualFaceData.quality /= processedFaces;
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				virtualFaceData.color /= processedFaces;
				#endif
			}
			ASSERT(!virtualFaceDatas.empty());
		}
		virtualFacesDatas.emplace_back(std::move(virtualFaceDatas));
		virtualFaces.emplace_back(std::move(virtualFace));
	} while (!remainingFaces.empty());
}

#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_MEDIAN

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// consider as outlier if the absolute difference to the median is outside this threshold
	if (thOutlier <= 0)
		thOutlier = 0.15f*255.f;

	// init colors array
	if (faceDatas.GetSize() <= 3)
		return false;
	FloatArr channels[3];
	for (int c=0; c<3; ++c)
		channels[c].Resize(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c)
			channels[c][i] = color[c];
	}

	// find median
	for (int c=0; c<3; ++c)
		channels[c].Sort();
	const unsigned idxMedian(faceDatas.GetSize() >> 1);
	Color median;
	for (int c=0; c<3; ++c)
		median[c] = channels[c][idxMedian];

	// abort if there are not at least 3 inliers
	int nInliers(0);
	BoolArr inliers(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c) {
			if (ABS(median[c]-color[c]) > thOutlier) {
				inliers[i] = false;
				goto CONTINUE_LOOP;
			}
		}
		inliers[i] = true;
		++nInliers;
		CONTINUE_LOOP:;
	}
	if (nInliers == faceDatas.GetSize())
		return true;
	if (nInliers < 3)
		return false;

	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	return true;
}

#elif TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA

// A multi-variate normal distribution which is NOT normalized such that the integral is 1
// - centered is the vector for which the function is to be evaluated with the mean subtracted [Nx1]
// - X is the vector for which the function is to be evaluated [Nx1]
// - mu is the mean around which the distribution is centered [Nx1]
// - covarianceInv is the inverse of the covariance matrix [NxN]
// return exp(-1/2 * (X-mu)^T * covariance_inv * (X-mu))
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& centered, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return EXP(T(-0.5) * T(centered.adjoint() * covarianceInv * centered));
}
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& X, const Eigen::Matrix<T,N,1>& mu, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return MultiGaussUnnormalized<T,N>(X - mu, covarianceInv);
}

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	if (thOutlier <= 0) {
		thOutlier = 6e-2f;
	}

	const double minCovariance = 1e-3;
	const unsigned maxIterations = 10;
	const unsigned minInliers = 4;

	const uint32_t nAll = (uint32_t)faceDatas.GetSize();
	if (nAll <= minInliers) {
		return false;
	}

	if (thOutlier <= 0 || thOutlier >= 1.0f) {
		thOutlier = 0.06f;
	}
	const double thMahalanobisSq = -2.0 * std::log((double)thOutlier);

	// Preconvert colors once (SoA for cache + no Eigen temporaries).
	std::vector<double> c0(nAll), c1(nAll), c2(nAll);
	for (uint32_t i = 0; i < nAll; ++i) {
		const Color::EVec v = (const Color::EVec)faceDatas[i].color;
		c0[i] = (double)v[0];
		c1[i] = (double)v[1];
		c2[i] = (double)v[2];
	}

	// Inlier mask and active index list.
	std::vector<uint8_t> inlierMask(nAll, 1);
	std::vector<uint32_t> inlierIdx(nAll);
	for (uint32_t i = 0; i < nAll; ++i) {
		inlierIdx[i] = i;
	}
	uint32_t nIn = nAll;

	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	Eigen::Matrix3d covarianceInv;

	for (unsigned iter = 0; iter < maxIterations; ++iter) {
		// Mean over current inliers.
		double m0 = 0.0, m1 = 0.0, m2 = 0.0;
		for (uint32_t k = 0; k < nIn; ++k) {
			const uint32_t i = inlierIdx[k];
			m0 += c0[i];
			m1 += c1[i];
			m2 += c2[i];
		}
		const double invN = 1.0 / (double)nIn;
		m0 *= invN;
		m1 *= invN;
		m2 *= invN;
		mean[0] = m0;
		mean[1] = m1;
		mean[2] = m2;

		// Covariance (symmetric, sample covariance with 1/(n-1)).
		double c00 = 0.0, c01 = 0.0, c02 = 0.0;
		double c11 = 0.0, c12 = 0.0;
		double c22 = 0.0;

		for (uint32_t k = 0; k < nIn; ++k) {
			const uint32_t i = inlierIdx[k];
			const double dx0 = c0[i] - m0;
			const double dx1 = c1[i] - m1;
			const double dx2 = c2[i] - m2;

			c00 += dx0 * dx0;
			c01 += dx0 * dx1;
			c02 += dx0 * dx2;
			c11 += dx1 * dx1;
			c12 += dx1 * dx2;
			c22 += dx2 * dx2;
		}

		const double inv = 1.0 / std::max(1.0, (double)(nIn - 1));
		covariance(0, 0) = c00 * inv;
		covariance(0, 1) = c01 * inv;
		covariance(0, 2) = c02 * inv;
		covariance(1, 0) = c01 * inv;
		covariance(1, 1) = c11 * inv;
		covariance(1, 2) = c12 * inv;
		covariance(2, 0) = c02 * inv;
		covariance(2, 1) = c12 * inv;
		covariance(2, 2) = c22 * inv;

		if (covariance.array().abs().maxCoeff() < minCovariance) {
			// Same behavior: remove non-inliers and return true.
			// Do stable compaction once (preserves original order).
			uint32_t w = 0;
			for (uint32_t i = 0; i < nAll; ++i) {
				if (inlierMask[i]) {
					if (w != i) {
						faceDatas[w] = faceDatas[i];
					}
					++w;
				}
			}
			while (faceDatas.GetSize() > w) {
				faceDatas.RemoveLast();
			}
			return true;
		}

		Eigen::LLT<Eigen::Matrix3d> llt(covariance);
		if (llt.info() != Eigen::Success) {
			return false;
		}
		covarianceInv = llt.solve(Eigen::Matrix3d::Identity());

		// Classify all points (same rule as your code).
		// Build next inlier list without touching faceDatas yet.
		uint32_t newNIn = 0;
		bool changed = false;

		const double i00 = covarianceInv(0, 0), i01 = covarianceInv(0, 1), i02 = covarianceInv(0, 2);
		const double i10 = covarianceInv(1, 0), i11 = covarianceInv(1, 1), i12 = covarianceInv(1, 2);
		const double i20 = covarianceInv(2, 0), i21 = covarianceInv(2, 1), i22 = covarianceInv(2, 2);

		for (uint32_t i = 0; i < nAll; ++i) {
			const double dx0 = c0[i] - m0;
			const double dx1 = c1[i] - m1;
			const double dx2 = c2[i] - m2;

			const double t0 = i00 * dx0 + i01 * dx1 + i02 * dx2;
			const double t1 = i10 * dx0 + i11 * dx1 + i12 * dx2;
			const double t2 = i20 * dx0 + i21 * dx1 + i22 * dx2;

			const double dist2 = dx0 * t0 + dx1 * t1 + dx2 * t2;

			const uint8_t isIn = (dist2 < thMahalanobisSq) ? 1 : 0;
			if (isIn) {
				inlierIdx[newNIn++] = i;
			}

			if (inlierMask[i] != isIn) {
				inlierMask[i] = isIn;
				changed = true;
			}
		}

		nIn = newNIn;
		if (nIn == nAll) {
			return true;
		}
		if (nIn < minInliers) {
			return false;
		}
		if (!changed) {
			break;
		}
	}

#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_DAMPING
	{
		const float factorOutlierRemoval = 0.2f;
		covarianceInv *= (double)factorOutlierRemoval;

		for (uint32_t i = 0; i < nAll; ++i) {
			if (!inlierMask[i]) {
				continue;
			}
			Eigen::Vector3d color;
			color[0] = c0[i];
			color[1] = c1[i];
			color[2] = c2[i];

			const double gaussValue = MultiGaussUnnormalized<double, 3>(color, mean, covarianceInv);
			faceDatas[i].quality *= (float)gaussValue;
		}
	}
#endif

#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_CLAMPING
	{
		// Stable in-place compaction (preserve order).
		uint32_t w = 0;
		for (uint32_t i = 0; i < nAll; ++i) {
			if (inlierMask[i]) {
				if (w != i) {
					faceDatas[w] = faceDatas[i];
				}
				++w;
			}
		}
		while (faceDatas.GetSize() > w) {
			faceDatas.RemoveLast();
		}
	}
#endif

	return true;
}
#endif

static void
CollapseSmallLabelIslands(
	const Mesh::FaceFacesArr& faceFaces,
	const MeshTexture::FaceDataViewArr& facesDatas,
	MeshTexture::LabelArr& labels,
	uint32_t maxIslandFaces,
	uint32_t maxPasses)
{
	return;
	const uint32_t faceCount = (uint32_t)labels.size();
	if (faceCount == 0) {
		return;
	}

	auto FaceHasLabelCandidate = [&](uint32_t f, MeshTexture::Label lbl) -> bool {
		const MeshTexture::FaceDataArr& arr = facesDatas[(FIndex)f];
		for (const MeshTexture::FaceData& fd : arr) {
			if ((MeshTexture::Label)fd.idxView == lbl) {
				return true;
			}
		}
		return false;
		};

	// Union-find helper
	std::vector<uint32_t> parent(faceCount);
	std::vector<uint8_t> rank(faceCount);

	auto FindRoot = [&](uint32_t x) -> uint32_t {
		while (parent[x] != x) {
			parent[x] = parent[parent[x]];
			x = parent[x];
		}
		return x;
		};

	auto Union = [&](uint32_t a, uint32_t b) {
		a = FindRoot(a);
		b = FindRoot(b);
		if (a == b) {
			return;
		}
		const uint8_t ra = rank[a];
		const uint8_t rb = rank[b];
		if (ra < rb) {
			parent[a] = b;
		}
		else if (ra > rb) {
			parent[b] = a;
		}
		else {
			parent[b] = a;
			rank[a] = (uint8_t)(ra + 1);
		}
		};

	for (uint32_t pass = 0; pass < maxPasses; ++pass) {
		// Build components of same-label adjacency
		for (uint32_t i = 0; i < faceCount; ++i) {
			parent[i] = i;
			rank[i] = 0;
		}

		for (uint32_t f = 0; f < faceCount; ++f) {
			const MeshTexture::Label lf = labels[(FIndex)f];
			if (lf == NO_ID) {
				continue;
			}

			const Mesh::FaceFaces& adj = faceFaces[(FIndex)f];
			for (int k = 0; k < 3; ++k) {
				const FIndex fn = adj[k];
				if (fn == NO_ID) {
					continue;
				}
				const uint32_t g = (uint32_t)fn;
				if (labels[(FIndex)g] != lf) {
					continue;
				}
				Union(f, g);
			}
		}

		// Component size + representative label
		std::vector<uint32_t> compSize(faceCount, 0);
		std::vector<MeshTexture::Label> compLabel(faceCount, NO_ID);

		for (uint32_t f = 0; f < faceCount; ++f) {
			const MeshTexture::Label lf = labels[(FIndex)f];
			if (lf == NO_ID) {
				continue;
			}
			const uint32_t r = FindRoot(f);
			++compSize[r];
			if (compLabel[r] == NO_ID) {
				compLabel[r] = lf;
			}
		}

		// For each small component, pick a neighbor label by boundary vote
		// Use a small fixed top list per component to avoid hash maps.
		struct Top2 {
			MeshTexture::Label lbl[2];
			uint32_t cnt[2];
		};

		std::vector<Top2> top(faceCount);
		for (uint32_t r = 0; r < faceCount; ++r) {
			top[r].lbl[0] = NO_ID; top[r].cnt[0] = 0;
			top[r].lbl[1] = NO_ID; top[r].cnt[1] = 0;
		}

		auto AddVote = [&](Top2& t, MeshTexture::Label lbl) {
			if (lbl == NO_ID) {
				return;
			}
			if (t.lbl[0] == lbl) {
				++t.cnt[0];
				return;
			}
			if (t.lbl[1] == lbl) {
				++t.cnt[1];
				return;
			}
			if (t.lbl[0] == NO_ID) {
				t.lbl[0] = lbl;
				t.cnt[0] = 1;
				return;
			}
			if (t.lbl[1] == NO_ID) {
				t.lbl[1] = lbl;
				t.cnt[1] = 1;
				return;
			}
			// replace weaker
			if (t.cnt[0] < t.cnt[1]) {
				t.lbl[0] = lbl;
				t.cnt[0] = 1;
			}
			else {
				t.lbl[1] = lbl;
				t.cnt[1] = 1;
			}
			};

		for (uint32_t f = 0; f < faceCount; ++f) {
			const MeshTexture::Label lf = labels[(FIndex)f];
			if (lf == NO_ID) {
				continue;
			}

			const uint32_t rf = FindRoot(f);
			if (compSize[rf] == 0 || compSize[rf] > maxIslandFaces) {
				continue;
			}

			const Mesh::FaceFaces& adj = faceFaces[(FIndex)f];
			for (int k = 0; k < 3; ++k) {
				const FIndex fn = adj[k];
				if (fn == NO_ID) {
					continue;
				}
				const uint32_t g = (uint32_t)fn;
				const MeshTexture::Label lg = labels[(FIndex)g];
				if (lg == NO_ID || lg == lf) {
					continue;
				}
				AddVote(top[rf], lg);
			}
		}

		// Relabel faces in small components if the winning neighbor label is a valid candidate
		uint32_t changed = 0;

		for (uint32_t f = 0; f < faceCount; ++f) {
			const MeshTexture::Label lf = labels[(FIndex)f];
			if (lf == NO_ID) {
				continue;
			}

			const uint32_t rf = FindRoot(f);
			if (compSize[rf] == 0 || compSize[rf] > maxIslandFaces) {
				continue;
			}

			Top2& t = top[rf];
			MeshTexture::Label best = t.lbl[0];
			uint32_t bestCnt = t.cnt[0];
			if (t.cnt[1] > bestCnt) {
				best = t.lbl[1];
				bestCnt = t.cnt[1];
			}

			if (best == NO_ID) {
				continue;
			}

			// Critical safety: only relabel if that camera is actually a candidate for this face
			if (!FaceHasLabelCandidate(f, best)) {
				continue;
			}

			if (labels[(FIndex)f] != best) {
				labels[(FIndex)f] = best;
				++changed;
			}
		}

		if (changed == 0) {
			break;
		}
	}
}

bool MeshTexture::FaceViewSelection(LabelArr& labels, unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness, const IIndexArr& views)
{
	TEX_PROFILE_SCOPE("FaceViewSelection (total)");
	// extract array of triangles incident to each vertex
	TEX_PROFILE_BEGIN(_tFvsListVerts);
	ListVertexFaces();
	TEX_PROFILE_END(_tFvsListVerts, "FaceViewSelection: ListVertexFaces");

	// create texture patches
	{
		// compute face normals and smoothen them
		TEX_PROFILE_BEGIN(_tFvsSmoothN);
		scene.mesh.SmoothNormalFaces();
		TEX_PROFILE_END(_tFvsSmoothN, "FaceViewSelection: SmoothNormalFaces");

		// list all views for each face
		FaceDataViewArr facesDatas;
		if (!ListCameraFaces(facesDatas, fOutlierThreshold, views))
			return false;
		LogPeakMem("FVS: after ListCameraFaces");

#ifdef STATS
		double avgChoices = 0.0;
		double avgSpread = 0.0;
		int counted = 0;

		for (FIndex f = 0; f < facesDatas.size(); ++f) {
			const FaceDataArr& arr = facesDatas[f];
			if (arr.size() < 2)
				continue;

			float minQ = FLT_MAX;
			float maxQ = -FLT_MAX;

			for (const FaceData& fd : arr) {
				minQ = std::min(minQ, fd.quality);
				maxQ = std::max(maxQ, fd.quality);
			}

			avgChoices += arr.size();
			avgSpread += (maxQ - minQ);
			counted++;
		}

		if (counted > 0) {
			printf("Avg cameras per face: %.2f\n", avgChoices / counted);
			printf("Avg quality spread per face: %.6f\n", avgSpread / counted);
		}

#endif

#if 0 // Unused?
		std::vector<double> camSum(images.size(), 0.0);
		std::vector<uint64_t> camCnt(images.size(), 0);

		for (size_t f = 0; f < facesDatas.size(); ++f) {
			for (const FaceData& fd : facesDatas[f]) {
				camSum[fd.idxView] += fd.color[0]; // Y channel
				camCnt[fd.idxView] += 1;
			}
		}

		std::vector<float> camGain(images.size(), 1.0f);

		double globalMean = 0.0;
		int valid = 0;

		for (size_t c = 0; c < images.size(); ++c) {
			if (camCnt[c] > 0) {
				camSum[c] /= camCnt[c];
				globalMean += camSum[c];
				valid++;
			}
		}

		globalMean /= std::max(1, valid);

		for (size_t c = 0; c < images.size(); ++c) {
			if (camCnt[c] > 0) {
				float g = (float)(globalMean / camSum[c]);
				if (g < 0.7f) g = 0.7f;
				if (g > 1.3f) g = 1.3f;
				camGain[c] = g;
			}
		}
#endif

		labels.clear();

		size_t maxIdxView = 0;
		bool haveAny = false;
		for (const FaceDataArr& fdArr : facesDatas) {
			for (const FaceData& fd : fdArr) {
				haveAny = true;
				if ((size_t)fd.idxView > maxIdxView)
					maxIdxView = (size_t)fd.idxView;
			}
		}
		const size_t numViews = haveAny ? (maxIdxView + 1) : 0; // views are [0..maxIdxView]
		const size_t numLabels = numViews + 1;                   // add label 0

		// construct and use virtual faces for patch creation instead of actual mesh faces;
		// the virtual faces are composed of coplanar triangles sharing same views
		const bool bUseVirtualFaces(minCommonCameras > 0);

#if 0
		// JPB WIP Unused
		if (bUseVirtualFaces) {
			typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
			typedef boost::graph_traits<Graph>::edge_iterator EdgeIter;
			typedef boost::graph_traits<Graph>::out_edge_iterator EdgeOutIter;
			Graph graph;

			throw std::runtime_error("Unsupported");
			// create faces graph

			// 1) create FaceToVirtualFaceMap
			FaceDataViewArr virtualFacesDatas;
			VirtualFaceIdxsArr virtualFaces; // stores each virtual face as an array of mesh face ID
			CreateVirtualFaces(facesDatas, virtualFacesDatas, virtualFaces, minCommonCameras);
			Mesh::FaceIdxArr mapFaceToVirtualFace(faces.size()); // for each mesh face ID, store the virtual face ID witch contains it
			size_t controlCounter(0);
			FOREACH(idxVF, virtualFaces) {
				const Mesh::FaceIdxArr& vf = virtualFaces[idxVF];
				for (FIndex idxFace : vf) {
					mapFaceToVirtualFace[idxFace] = idxVF;
					++controlCounter;
				}
			}
			ASSERT(controlCounter == faces.size());
			// 2) create function to find virtual faces neighbors
			VirtualFaceIdxsArr virtualFaceNeighbors; { // for each virtual face, the list of virtual faces with at least one vertex in common
				virtualFaceNeighbors.resize(virtualFaces.size());
				FOREACH(idxVF, virtualFaces) {
					const Mesh::FaceIdxArr& vf = virtualFaces[idxVF];
					Mesh::FaceIdxArr& vfNeighbors = virtualFaceNeighbors[idxVF];
					for (FIndex idxFace : vf) {
						const Mesh::FaceFaces& adjFaces = faceFaces[idxFace];
						for (int i = 0; i < 3; ++i) {
							const FIndex fAdj(adjFaces[i]);
							if (fAdj == NO_ID)
								continue;
							if (mapFaceToVirtualFace[fAdj] == idxVF)
								continue;
							if (fAdj != idxFace && vfNeighbors.Find(mapFaceToVirtualFace[fAdj]) == Mesh::FaceIdxArr::NO_INDEX) {
								vfNeighbors.emplace_back(mapFaceToVirtualFace[fAdj]);
							}
						}
					}
				}
			}
			// 3) use virtual faces to build the graph
			// 4) assign images to virtual faces
			// 5) spread image ID to each mesh face from virtual face
			FOREACH(idxFace, virtualFaces) {
				MAYBEUNUSED const Mesh::FIndex idx((Mesh::FIndex)boost::add_vertex(graph));
				ASSERT(idx == idxFace);
			}
			FOREACH(idxVirtualFace, virtualFaces) {
				const Mesh::FaceIdxArr& afaces = virtualFaceNeighbors[idxVirtualFace];
				for (FIndex idxVirtualFaceAdj : afaces) {
					if (idxVirtualFace >= idxVirtualFaceAdj)
						continue;
					const bool bInvisibleFace(virtualFacesDatas[idxVirtualFace].empty());
					const bool bInvisibleFaceAdj(virtualFacesDatas[idxVirtualFaceAdj].empty());
					if (bInvisibleFace || bInvisibleFaceAdj)
						continue;
					boost::add_edge(idxVirtualFace, idxVirtualFaceAdj, graph);
				}
			}

			ASSERT((Mesh::FIndex)boost::num_vertices(graph) == virtualFaces.size());
			// assign the best view to each face
			labels.resize(faces.size());
			components.resize(faces.size());
			{
				// normalize quality values
				float maxQuality(0);
				for (const FaceDataArr& faceDatas : virtualFacesDatas) {
					for (const FaceData& faceData : faceDatas)
						if (maxQuality < faceData.quality)
							maxQuality = faceData.quality;
				}
				Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);
				for (const FaceDataArr& faceDatas : virtualFacesDatas) {
					for (const FaceData& faceData : faceDatas)
						hist.Add(faceData.quality);
				}
				const float normQuality(hist.GetApproximatePermille(0.95f));

#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
				// initialize inference structures
				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness * LBPInference::MaxEnergy);
				LBPInference inference;
				{
					inference.SetNumNodes(virtualFaces.size());
					inference.SetSmoothCost(SmoothnessPotts);
					EdgeOutIter ei, eie;
					FOREACH(f, virtualFaces) {
						for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
							ASSERT(f == (FIndex)ei->m_source);
							const FIndex fAdj((FIndex)ei->m_target);
							ASSERT(components.empty() || components[f] == components[fAdj]);
							if (f < fAdj) // add edges only once
								inference.SetNeighbors(f, fAdj);
						}
						// set costs for label 0 (undefined)
						inference.SetDataCost((Label)0, f, MaxEnergy);
					}
				}

				// set data costs for all labels (except label 0 - undefined)
				FOREACH(f, virtualFacesDatas) {
					const FaceDataArr& faceDatas = virtualFacesDatas[f];
					for (const FaceData& faceData : faceDatas) {
						const Label label((Label)faceData.idxView + 1);
						const float normalizedQuality(faceData.quality >= normQuality ? 1.f : faceData.quality / normQuality);
						const float dataCost((1.f - normalizedQuality) * MaxEnergy);
						inference.SetDataCost(label, f, dataCost);
					}
				}

				// assign the optimal view (label) to each face
				// (label 0 is reserved as undefined)
				inference.Optimize();

				// extract resulting labeling
				LabelArr virtualLabels(virtualFaces.size());
				virtualLabels.Memset(0xFF);
				FOREACH(l, virtualLabels) {
					const Label label(inference.GetLabel(l));
					ASSERT(label < images.GetSize() + 1);
					if (label > 0)
						virtualLabels[l] = label - 1;
				}
				FOREACH(l, labels) {
					labels[l] = virtualLabels[mapFaceToVirtualFace[l]];
				}
#endif
			}

			graph.clear();
		}
#endif

		uint32_t numFaces = faces.size();
		// ------------------------------------------------------------
		// Compute geometric connected components WITHOUT Boost
		// ------------------------------------------------------------
		TEX_PROFILE_BEGIN(_tFvsCompGeom);
		std::vector<int> compGeom(numFaces, -1);

		int nGeomComponents = 0;

		std::vector<FIndex> stack;
		stack.reserve(256);

		for (FIndex f = 0; f < numFaces; ++f) {
			if (compGeom[f] != -1)
				continue;

			// start new component
			compGeom[f] = nGeomComponents;
			stack.clear();
			stack.push_back(f);

			while (!stack.empty()) {
				FIndex cur = stack.back();
				stack.pop_back();

				const Mesh::FaceFaces& adj = faceFaces[cur];

				for (int k = 0; k < 3; ++k) {
					FIndex fn = adj[k];
					if (fn == NO_ID)
						continue;

					if (compGeom[fn] == -1) {
						compGeom[fn] = nGeomComponents;
						stack.push_back(fn);
					}
				}
			}

			++nGeomComponents;
		}
		TEX_PROFILE_END(_tFvsCompGeom, "FaceViewSelection: geometric components");

		ASSERT((Mesh::FIndex)boost::num_vertices(graph) == numFaces);

		// start patch creation starting directly from individual faces
		if (!bUseVirtualFaces) {
			// assign the best view to each face
			labels.resize(numFaces); {
				// normalize quality values
				float maxQuality(0);
				for (const FaceDataArr& faceDatas : facesDatas) {
					for (const FaceData& faceData : faceDatas)
						if (maxQuality < faceData.quality)
							maxQuality = faceData.quality;
				}
				Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);
				for (const FaceDataArr& faceDatas : facesDatas) {
					for (const FaceData& faceData : faceDatas)
						hist.Add(faceData.quality);
				}
				const float normQuality(hist.GetApproximatePermille(0.95f));

#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
				// initialize inference structures
				const LBPInference::EnergyType MaxEnergy(
					fRatioDataSmoothness * LBPInference::MaxEnergy);
				TEX_PROFILE_BEGIN(_tFvsLbpBuild);
				LBPInference inference;
				{
					inference.SetNumNodes(numFaces);

					gSceneForSmoothness = &scene;
					inference.SetSmoothCost(SmoothnessPottsStrong);
					// SmoothnessPottsStrong is a generalized Potts model (0 for l1==l2,
					// a per-edge normal-based constant otherwise) -> enable the O(L1+L2)
					// message fast path (bit-identical result, avoids the L1*L2 dot-product
					// evaluations that dominate LBP Optimize).
					inference.SetPottsSmoothness(true);

					// ---- 1. Count undirected edges once ----
					size_t edgeCount = 0;

					for (FIndex f = 0; f < (FIndex)numFaces; ++f)
					{
						const Mesh::FaceFaces& adj = faceFaces[f];

						for (int k = 0; k < 3; ++k)
						{
							FIndex fn = adj[k];
							if (fn != NO_ID && f < fn)
								++edgeCount;
						}
					}

					// ---- 2. Reserve storage once ----
					inference.edges.reserve(edgeCount * 2);

					// ---- 3. Build graph ----
					for (FIndex f = 0; f < (FIndex)numFaces; ++f)
					{
						const Mesh::FaceFaces& adj = faceFaces[f];

						for (int k = 0; k < 3; ++k)
						{
							FIndex fn = adj[k];
							if (fn != NO_ID && f < fn)
								inference.SetNeighbors(f, fn);
						}
					}
				}
#ifdef STATS
				printf("Graph edges: %zu\n", (size_t)boost::num_edges(graph));
#endif

				// set data costs for all labels (except label 0 - undefined)
				// Must be single threaded
				std::vector<int> order;
				order.reserve(64);

				const auto undefinedCost =
					LBPInference::MaxEnergy * 3.0f;

#pragma omp for schedule(dynamic, 128)
				for (int64_t f = 0; f < (int64_t)numFaces; ++f) {
					const FaceDataArr& faceDatas = facesDatas[f];

					const int nFD = (int)faceDatas.size();

					LBPInference::Node& node = inference.nodes[f];

					node.labels.clear();
					node.dataCosts.clear();

					//node.labels.reserve(nFD + 1);
					//node.dataCosts.reserve(nFD + 1);

					// ---- undefined label first ----
					node.labels.push_back(0);
					node.dataCosts.push_back(undefinedCost);

					if (nFD == 0)
						continue;

					order.resize(nFD);

					for (int i = 0; i < nFD; ++i)
						order[i] = i;

					// insertion sort (faster for small nFD)
					for (int i = 1; i < nFD; ++i) {
						int key = order[i];
						float qkey = faceDatas[key].quality;

						int j = i - 1;
						while (j >= 0 &&
							faceDatas[order[j]].quality < qkey)
						{
							order[j + 1] = order[j];
							--j;
						}
						order[j + 1] = key;
					}

					const float invDen =
						(nFD > 1) ? (1.0f / float(nFD - 1)) : 0.0f;

					const float scale =
						invDen * MaxEnergy;

					for (int rank = 0; rank < nFD; ++rank) {
						const FaceData& fd =
							faceDatas[order[rank]];

						Label lbl =
							(Label)fd.idxView + 1;

						float dataCost =
							rank * scale;

						node.labels.push_back(lbl);
						node.dataCosts.push_back(dataCost);
					}
				}
				TEX_PROFILE_END(_tFvsLbpBuild, "FaceViewSelection: LBP build graph+datacost");

				TEX_PROFILE_BEGIN(_tFvsLbpOpt);
				inference.Optimize();
				TEX_PROFILE_END(_tFvsLbpOpt, "FaceViewSelection: LBP Optimize");

#ifdef STATS
				int countLabel0 = 0;
				int countNonZero = 0;

				for (size_t i = 0; i < (size_t)numFaces; ++i) {
					Label lbl = inference.finalLabels[i];
					if (lbl == 0) countLabel0++;
					else countNonZero++;
				}

				printf("LBP result: label0=%d  nonzero=%d\n", countLabel0, countNonZero);
#endif

				const auto& labelVec = inference.finalLabels;

				for (int64_t l = 0; l < (int64_t)numFaces; ++l) {
					Label lbl = labelVec[l];
					labels[l] = (lbl > 0) ? (lbl - 1) : NO_ID;
				}

#ifdef STATS
				double neighborDisagree = 0.0;
				int pairs = 0;

				for (FIndex f = 0; f < numFaces; ++f) {
					const auto& adj = faceFaces[f];
					for (int k = 0; k < 3; ++k) {
						FIndex fn = adj[k];
						if (fn == NO_ID || f >= fn)
							continue;

						pairs++;
						if (labels[f] != labels[fn])
							neighborDisagree++;
					}
				}

				printf("Neighbor disagreement ratio: %.4f\n",
					neighborDisagree / pairs);

				int countNoID = 0;
				for (size_t i = 0; i < labels.size(); ++i)
					if (labels[i] == NO_ID)
						countNoID++;

				printf("labels[]: NO_ID=%d  valid=%d\n", countNoID, (int)labels.size() - countNoID);
#endif
#endif
			}
		}

		// Geometry which has NO_ID faces will be colored incorrectly.
		// Here we work to remove it before it even becomes a polygon.
		// ----------------------------------------------------------------------
		// High-quality confidence-weighted boundary refinement
		// ----------------------------------------------------------------------
		// GATED OFF when TEXTURE_DATACOLOR_UNOBSERVED is on: this whole legacy
		// label-forcing stage (8-pass boundary propagation + "assign best available
		// camera" Pass 1 + Pass 2 smear) exists to leave NO face NO_ID, which fills
		// the unobserved/grazing rim with stretched oblique projections (the gray
		// facets). With data-color on, the MRF labels stand as-is and every face left
		// NO_ID is handed to the surface-propagated data-color fill instead.
#if !TEXTURE_DATACOLOR_UNOBSERVED
		{
			const float normalThreshold = 0.3f;   // cosine threshold
			const int maxPasses = 8;              // controlled expansion depth

			// --------------------------------------------------
			// Compute per-face confidence for labeled faces
			// --------------------------------------------------
			std::vector<float> faceConfidence(numFaces, 0.0f);

			for (FIndex f = 0; f < numFaces; ++f) {
				if (labels[f] == NO_ID)
					continue;

				const FaceDataArr& fDatas = facesDatas[f];

				for (const FaceData& fd : fDatas) {
					if (fd.idxView == labels[f]) {
						faceConfidence[f] = fd.quality;
						break;
					}
				}
			}

			// Normalize confidence
			float maxQ = 0.0f;
			for (float q : faceConfidence)
				if (q > maxQ) maxQ = q;

			if (maxQ > 0.0f) {
				for (float& q : faceConfidence)
					q /= maxQ;
			}

			// --------------------------------------------------
			// Controlled propagation
			// --------------------------------------------------
			for (int pass = 0; pass < maxPasses; ++pass) {
				bool changed = false;
				LabelArr newLabels = labels;
				std::vector<float> newConfidence = faceConfidence;

				for (FIndex f = 0; f < numFaces; ++f) {
					if (labels[f] != NO_ID)
						continue;

					const Mesh::FaceFaces& adj = faceFaces[f];

					Label bestLabel = NO_ID;
					float bestScore = -1.0f;

					for (int i = 0; i < 3; ++i) {
						FIndex n = adj[i];
						if (n == NO_ID)
							continue;

						if (labels[n] == NO_ID)
							continue;

						// Normal consistency check
						float cosAngle =
							scene.mesh.faceNormals[f].dot(scene.mesh.faceNormals[n]);

						if (cosAngle < normalThreshold)
							continue;

						const FaceDataArr& fDatas = facesDatas[f];

						for (const FaceData& fd : fDatas) {

							if (fd.idxView != labels[n])
								continue;

							// Confidence-weighted score
							float score = fd.quality * faceConfidence[n];

							if (score > bestScore) {
								bestScore = score;
								bestLabel = labels[n];
							}
						}
					}

					if (bestLabel != NO_ID) {
						newLabels[f] = bestLabel;
						newConfidence[f] = bestScore;   // inherit weakened confidence
						changed = true;
					}
				}

				labels.swap(newLabels);
				faceConfidence.swap(newConfidence);

				if (!changed)
					break;
			}

			// --------------------------------------------------
			// Final safety: assign best available camera if still NO_ID
			// (prevents holes entirely)
			// --------------------------------------------------
			// Pass 1: assign from own camera data if available
			for (FIndex f = 0; f < numFaces; ++f) {
				if (labels[f] != NO_ID)
					continue;

				const FaceDataArr& fDatas = facesDatas[f];
				if (fDatas.empty())
					continue;

				Label bestLabel = NO_ID;
				float bestQuality = -1.0f;

				for (const FaceData& fd : fDatas) {
					if (fd.quality > bestQuality) {
						bestQuality = fd.quality;
						bestLabel = fd.idxView;
					}
				}

				if (bestLabel != NO_ID) {
					labels[f] = bestLabel;
					faceConfidence[f] = bestQuality / maxQ;
				}
			}

			// Pass 2: for faces with NO camera data at all (invisible faces,
			// e.g. from hole-filling), propagate the label from any adjacent
			// labeled face. This eliminates the white specks in the texture
			// caused by faces that point into uninitialized atlas regions.
			{
				bool changed = true;
				int propagationPasses = 0;
				const int maxPropagationPasses = 20;

				while (changed && propagationPasses < maxPropagationPasses) {
					changed = false;
					++propagationPasses;

					for (FIndex f = 0; f < numFaces; ++f) {
						if (labels[f] != NO_ID)
							continue;

						const Mesh::FaceFaces& adj = faceFaces[f];
						Label bestLabel = NO_ID;
						float bestConf = -1.0f;

						for (int k = 0; k < 3; ++k) {
							FIndex fn = adj[k];
							if (fn == NO_ID)
								continue;
							if (labels[fn] == NO_ID)
								continue;

							if (faceConfidence[fn] > bestConf) {
								bestConf = faceConfidence[fn];
								bestLabel = labels[fn];
							}
						}

						if (bestLabel != NO_ID) {
							labels[f] = bestLabel;
							faceConfidence[f] = bestConf * 0.5f; // decay
							changed = true;
						}
					}
				}

				if (propagationPasses > 1) {
					DEBUG("Label propagation to invisible faces: %d passes", propagationPasses);
				}
			}
		}
#endif // !TEXTURE_DATACOLOR_UNOBSERVED (legacy label-forcing stage; data-color owns NO_ID faces instead)

#if 0 // JPB WIP BUG Until i can get this to work. def INCREASE_PATCHES
		// create texture patches (connected components of same-label adjacency)
		{
			seamEdges.clear();

			const uint32_t faceCount = (uint32_t)numFaces;

			// Build seam edges (optional)
			for (FIndex f = 0; f < (FIndex)faceCount; ++f) {
				const Mesh::FaceFaces& adj = faceFaces[f];

				for (int k = 0; k < 3; ++k) {
					const FIndex fn = adj[k];
					if (fn == NO_ID) {
						continue;
					}
					if (f >= fn) {
						continue;
					}
					if (labels[f] == NO_ID || labels[fn] == NO_ID || labels[f] != labels[fn]) {
						seamEdges.emplace_back(f, fn);
					}
				}
			}

			// Union-find on faces
			std::vector<uint32_t> parent(faceCount);
			std::vector<uint8_t> rank(faceCount, 0);

			auto ResetUF = [&]() {
				for (uint32_t i = 0; i < faceCount; ++i) {
					parent[i] = i;
					rank[i] = 0;
				}
				};

			auto FindRoot = [&](uint32_t x) -> uint32_t {
				while (parent[x] != x) {
					parent[x] = parent[parent[x]];
					x = parent[x];
				}
				return x;
				};

			auto Union = [&](uint32_t a, uint32_t b) {
				a = FindRoot(a);
				b = FindRoot(b);
				if (a == b) {
					return;
				}
				const uint8_t ra = rank[a];
				const uint8_t rb = rank[b];
				if (ra < rb) {
					parent[a] = b;
				}
				else if (ra > rb) {
					parent[b] = a;
				}
				else {
					parent[b] = a;
					rank[a] = (uint8_t)(ra + 1);
				}
				};

			auto BuildUFForCurrentLabels = [&]() {
				ResetUF();

				for (uint32_t f = 0; f < faceCount; ++f) {
					if (labels[(FIndex)f] == NO_ID) {
						continue;
					}

					const Mesh::FaceFaces& adj = faceFaces[(FIndex)f];

					for (int k = 0; k < 3; ++k) {
						const FIndex fn = adj[k];
						if (fn == NO_ID) {
							continue;
						}

						const uint32_t g = (uint32_t)fn;
						if (g == f) {
							continue;
						}

						if (labels[(FIndex)g] == NO_ID) {
							continue;
						}

						if (labels[(FIndex)f] != labels[(FIndex)g]) {
							continue;
						}

						if (compGeom[(FIndex)f] != compGeom[(FIndex)g]) {
							continue;
						}

						if (labels[f] == labels[g]) {
							float dot = scene.mesh.faceNormals[f].dot(scene.mesh.faceNormals[g]);
							// Be more relaxed here (e.g., 0.9 instead of 0.98)
							// This allows the road to "curve" or have "bumps" without shattering into patches
							if (dot > 0.9f) {
								Union(f, g);
								continue;
							}
						}
					}
				}
				};

			// Build UF for initial labels
			{
				const uint32_t targetPatches = 3000;
				const uint32_t minFaces = 256;
				const uint32_t maxIters = 6;
				const uint32_t faceCount = (uint32_t)numFaces;

				for (uint32_t iter = 0; iter < maxIters; ++iter) {
					BuildUFForCurrentLabels();

					// 1. Storage for this iteration
					std::vector<uint32_t> root(faceCount);
					std::vector<uint32_t> compSize(faceCount, 0);
					std::vector<uint32_t> rootToList(faceCount, 0xFFFFFFFFu);
					uint32_t listCount = 0;

					for (uint32_t f = 0; f < faceCount; ++f) {
						if (labels[(FIndex)f] == NO_ID) {
							root[f] = 0xFFFFFFFFu;
							continue;
						}
						uint32_t r = FindRoot(f);
						root[f] = r;
						if (compSize[r] == 0) {
							rootToList[r] = listCount++;
						}
						compSize[r]++;
					}

					if (listCount <= targetPatches) break;

					// 2. Build Offset Array (Standard CSR)
					std::vector<uint32_t> offsets(listCount + 1, 0);
					for (uint32_t r = 0; r < faceCount; ++r) {
						uint32_t id = rootToList[r];
						if (id != 0xFFFFFFFFu) {
							offsets[id + 1] = compSize[r];
						}
					}
					for (uint32_t i = 1; i <= listCount; ++i) {
						offsets[i] += offsets[i - 1];
					}

					// 3. Populate Faces By Component
					std::vector<uint32_t> facesByComp(offsets.back());
					std::vector<uint32_t> cursor = offsets;
					for (uint32_t f = 0; f < faceCount; ++f) {
						uint32_t r = root[f];
						if (r != 0xFFFFFFFFu) {
							facesByComp[cursor[rootToList[r]]++] = f;
						}
					}

					uint32_t changedComps = 0;
					struct LabelChange { uint32_t rootId; Label newLabel; };
					std::vector<LabelChange> pendingChanges;

					// 4. Voting Logic
					std::vector<float> scores((size_t)numLabels);
					std::vector<uint8_t> used((size_t)numLabels);

					for (uint32_t r = 0; r < faceCount; ++r) {
						uint32_t id = rootToList[r];
						if (id == 0xFFFFFFFFu || compSize[r] >= minFaces) continue;

						const uint32_t beg = offsets[id];
						const uint32_t end = offsets[id + 1];

						std::fill(scores.begin(), scores.end(), 0.0f);
						std::fill(used.begin(), used.end(), 0);

						// Track if a label is actually "reachable" (meets the UF criteria)
						std::vector<uint8_t> reachable((size_t)numLabels, 0);

						for (uint32_t it = beg; it < end; ++it) {
							uint32_t f = facesByComp[it];
							const Mesh::FaceFaces& adj = faceFaces[(FIndex)f];

							for (int k = 0; k < 3; ++k) {
								FIndex fn = adj[k];
								if (fn == NO_ID) continue;
								uint32_t g = (uint32_t)fn;
								if (root[g] == r) continue; // Same component

								// CRITICAL: If the UF wouldn't merge these based on Geometry/Normals,
								// we MUST NOT let this neighbor influence the label.
								if (compGeom[(FIndex)f] != compGeom[(FIndex)g]) continue;

								float dot = scene.mesh.faceNormals[f].dot(scene.mesh.faceNormals[g]);
								if (dot <= 0.9f) continue; // Must match BuildUFForCurrentLabels threshold

								Label lg = labels[fn];
								if (lg == NO_ID) continue;

								// If we're here, this neighbor is a valid merge candidate
								float weight = (dot > 0.98f) ? 15.0f : 1.0f;
								if (compSize[root[g]] < minFaces) weight *= 2.0f;

								scores[(size_t)lg] += weight;
								used[(size_t)lg] = 1;
								reachable[(size_t)lg] = 1;
							}
						}

						int bestLabel = -1;
						float bestScore = -1.0f;
						for (int l = 0; l < (int)numLabels; ++l) {
							if (!used[l] || !reachable[l]) continue;

							float s = scores[l];
							// Only apply stickiness if the current label is actually a valid neighbor
							if ((Label)l == labels[(FIndex)facesByComp[beg]]) s *= 1.2f;

							if (s > bestScore) {
								bestScore = s;
								bestLabel = l;
							}
						}

						if (bestLabel >= 0 && (Label)bestLabel != labels[(FIndex)facesByComp[beg]]) {
							pendingChanges.push_back({ r, (Label)bestLabel });
							changedComps++;
						}
					}
					for (const auto& change : pendingChanges) {
						uint32_t id = rootToList[change.rootId];
						for (uint32_t it = offsets[id]; it < offsets[id + 1]; ++it) {
							labels[(FIndex)facesByComp[it]] = change.newLabel;
						}
					}
					if (changedComps == 0) break;
				}

				CollapseSmallLabelIslands(faceFaces, facesDatas, labels, 512, 3);
				BuildUFForCurrentLabels();
			}

			// Assign compact component ids
			std::vector<uint32_t> rootToComp(faceCount, 0xFFFFFFFFu);
			uint32_t nextComp = 0;

			components.resize(faceCount);

			for (uint32_t f = 0; f < faceCount; ++f) {
				if (labels[(FIndex)f] == NO_ID) {
					components[(FIndex)f] = -1;
					continue;
				}

				const uint32_t r = FindRoot(f);
				uint32_t& cid = rootToComp[r];
				if (cid == 0xFFFFFFFFu) {
					cid = nextComp++;
				}
				components[(FIndex)f] = (int)cid;
			}

			// Build patches from components
			texturePatches.clear();
			texturePatches.resize(nextComp);

			for (uint32_t p = 0; p < nextComp; ++p) {
				texturePatches[p].label = NO_ID;
				texturePatches[p].faces.clear();
			}

			for (uint32_t f = 0; f < faceCount; ++f) {
				const int cid = components[(FIndex)f];
				if (cid < 0) {
					continue;
				}

				TexturePatch& tp = texturePatches[(size_t)cid];
				const Label lbl = labels[(FIndex)f];

				if (tp.label == NO_ID) {
					tp.label = lbl;
				}
				else {
					ASSERT(tp.label == lbl);
				}

				tp.faces.Insert((FIndex)f);
			}
		}
	}
#else
		// create texture patches
		{
			TEX_PROFILE_SCOPE("FaceViewSelection: create patches");
			seamEdges.clear();

			// Build seam edges directly from face adjacency
			FOREACH(f, faces)	{
				const Mesh::FaceFaces& adj = faceFaces[f];

				for (int k = 0; k < 3; ++k) {
					FIndex fn = adj[k];
					if (fn == NO_ID)
						continue;

					if (f >= fn)
						continue; // avoid duplicates

					if (labels[f] == NO_ID ||
						labels[fn] == NO_ID ||
						labels[f] != labels[fn])
					{
						seamEdges.emplace_back(f, fn);
					}
				}
			}

			// Create texture patches as CONNECTED COMPONENTS of same-label face
			// adjacency (flood fill over faceFaces where both faces share a label).
			// The previous grouping (compGeom*numLabels + label) merged ALL same-label
			// faces of a geometric component into ONE patch even when they were SCATTERED
			// across the mesh -> the patch's image bounding box became the WHOLE source
			// frame (MEASURED: 367 patches, each ~19 Mpx = a full image, 7076 Mpx total).
			// That blew the per-patch crops to ~20 GB AND forced the packer to tile 367
			// full images into an ~84000px atlas that was then downscaled ~5.7x to fit the
			// GPU cap -- silently crushing res-0 detail. Connected components give each
			// contiguous region its OWN tight bbox: crops shrink to the real coverage, the
			// atlas packs near 1:1 (little/no downscale -> SHARPER), and NO new seams are
			// introduced (the split pieces are non-adjacent, so they never shared a seam
			// edge -- seam leveling sees the identical set of label boundaries as before).
			components.resize(numFaces);
			for (FIndex f = 0; f < numFaces; ++f)
				components[f] = (FIndex)-1;

			int nextComp = 0;
			{
				std::vector<FIndex> stack;
				stack.reserve(1024);
				for (FIndex f0 = 0; f0 < numFaces; ++f0) {
					if (labels[f0] == NO_ID || (int)components[f0] != -1)
						continue;
					const int cid = nextComp++;
					const Label lbl = labels[f0];
					components[f0] = (FIndex)cid;
					stack.clear();
					stack.push_back(f0);
					while (!stack.empty()) {
						const FIndex cur = stack.back();
						stack.pop_back();
						const Mesh::FaceFaces& adj = faceFaces[cur];
						for (int k = 0; k < 3; ++k) {
							const FIndex fn = adj[k];
							if (fn == NO_ID || (int)components[fn] != -1 || labels[fn] != lbl)
								continue;
							components[fn] = (FIndex)cid;
							stack.push_back(fn);
						}
					}
				}
			}

			texturePatches.clear();
			texturePatches.resize(nextComp);
			for (int p = 0; p < nextComp; ++p)
				texturePatches[(uint32_t)p].label = NO_ID;

			for (FIndex f = 0; f < numFaces; ++f) {
				const int cid = (int)components[f];
				if (cid < 0)
					continue;
				TexturePatch& tp = texturePatches[(uint32_t)cid];
				if (tp.label == NO_ID)
					tp.label = labels[f];
				tp.faces.Insert(f);
			}
		}
	}
#endif

	return true;
}

// create seam vertices and edges
void MeshTexture::CreateSeamVertices()
{
	// each vertex will contain the list of patches it separates,
	// except the patch containing invisible faces;
	// each patch contains the list of edges belonging to that texture patch, starting from that vertex
	// (usually there are pairs of edges in each patch, representing the two edges starting from that vertex separating two valid patches)
	seamVertices.clear();

	std::vector<uint32_t> faceToPatch(faces.GetSize(), UINT32_MAX);

	for (uint32_t p = 0; p < texturePatches.GetSize(); ++p) {
		if (texturePatches[p].label == NO_ID)
			continue;

		for (const FIndex f : texturePatches[p].faces)
			faceToPatch[f] = p;
	}

	VIndex vs[2];
	uint32_t vs0[2], vs1[2];
	std::unordered_map<VIndex, uint32_t> mapVertexSeam;
	const unsigned patchCount = texturePatches.GetSize();
	for (const PairIdx& edge : seamEdges) {
		// store edge for the later seam optimization
		ASSERT(edge.i < edge.j);
		const uint32_t idxPatch0 = faceToPatch[edge.i];
		const uint32_t idxPatch1 = faceToPatch[edge.j];

		if (idxPatch0 == UINT32_MAX || idxPatch1 == UINT32_MAX)
			continue;

		if (idxPatch0 == idxPatch1)
			continue;

		seamVertices.ReserveExtra(2);
		scene.mesh.GetEdgeVertices(edge.i, edge.j, vs0, vs1);
		ASSERT(faces[edge.i][vs0[0]] == faces[edge.j][vs1[0]]);
		ASSERT(faces[edge.i][vs0[1]] == faces[edge.j][vs1[1]]);
		vs[0] = faces[edge.i][vs0[0]];
		vs[1] = faces[edge.i][vs0[1]];

		const auto itSeamVertex0(mapVertexSeam.emplace(std::make_pair(vs[0], seamVertices.GetSize())));
		if (itSeamVertex0.second)
			seamVertices.emplace_back(vs[0]);
		SeamVertex& seamVertex0 = seamVertices[itSeamVertex0.first->second];

		const auto itSeamVertex1(mapVertexSeam.emplace(std::make_pair(vs[1], seamVertices.GetSize())));
		if (itSeamVertex1.second)
			seamVertices.emplace_back(vs[1]);
		SeamVertex& seamVertex1 = seamVertices[itSeamVertex1.first->second];

		{
			const TexCoord offset0(texturePatches[idxPatch0].rect.tl());

			SeamVertex::Patch& patch00 =
				seamVertex0.GetPatch(idxPatch0);

			SeamVertex::Patch& patch10 =
				seamVertex1.GetPatch(idxPatch0);

			const uint32_t seamIdx1 =
				itSeamVertex1.first->second;

			const uint32_t seamIdx0 =
				itSeamVertex0.first->second;

			ASSERT(patch00.edges.Find(seamIdx1) == NO_ID);
			patch00.edges.emplace_back(
				seamIdx1,
				edge.i
			);
			patch00.proj =
				faceTexcoords[edge.i * 3 + vs0[0]] + offset0;

			ASSERT(patch10.edges.Find(seamIdx0) == NO_ID);
			patch10.edges.emplace_back(
				seamIdx0,
				edge.i
			);
			patch10.proj =
				faceTexcoords[edge.i * 3 + vs0[1]] + offset0;
		}

		{
			const TexCoord offset1(texturePatches[idxPatch1].rect.tl());

			SeamVertex::Patch& patch01 =
				seamVertex0.GetPatch(idxPatch1);

			SeamVertex::Patch& patch11 =
				seamVertex1.GetPatch(idxPatch1);

			const uint32_t seamIdx1 =
				itSeamVertex1.first->second;

			const uint32_t seamIdx0 =
				itSeamVertex0.first->second;

			ASSERT(patch01.edges.Find(seamIdx1) == NO_ID);
			patch01.edges.emplace_back(
				seamIdx1,
				edge.j
			);
			patch01.proj =
				faceTexcoords[edge.j * 3 + vs1[0]] + offset1;

			ASSERT(patch11.edges.Find(seamIdx0) == NO_ID);
			patch11.edges.emplace_back(
				seamIdx0,
				edge.j
			);
			patch11.proj =
				faceTexcoords[edge.j * 3 + vs1[1]] + offset1;
		}
	}
	seamEdges.Release();
}


static DWORD_PTR PinThreadToCoreAndSave(int coreIndex) {
	DWORD_PTR newMask = (DWORD_PTR)1 << coreIndex;
	return SetThreadAffinityMask(GetCurrentThread(), newMask);
}

static void RestoreThreadAffinity(DWORD_PTR oldMask) {
	SetThreadAffinityMask(GetCurrentThread(), oldMask);
}

constexpr float invTable[9] = {
		0.0f,  // unused (n=0)
		1.0f,
		0.5f,
		1.0f / 3.0f,
		0.25f,
		0.2f,
		1.0f / 6.0f,
		1.0f / 7.0f,
		0.125f
};

struct VertexPatchRow {
	std::vector<uint32_t> patches;
	std::vector<MatIdx> rows;
};

template <typename PIXEL>
static inline PIXEL YCBCR_DELTA2RGB(const PIXEL& d) {
	typedef typename PIXEL::Type T;
	const T dCb(d[1]);
	const T dCr(d[2]);
	return PIXEL(
		d[0] + dCr * T(1.402),
		d[0] + dCb * T(-0.34414) + dCr * T(-0.71414),
		d[0] + dCb * T(1.772)
	);
}

// Per-thread GlobalSeamLeveling scratch. Declared at namespace scope (as
// thread_local) rather than as function-local `static thread_local` so it can
// be released after seam leveling (see ReleaseSeamScratch); function-local
// statics are unreachable from outside and would stay pinned for the whole run.
namespace { namespace gseam {
	thread_local std::vector<MatIdx> faceRowIndices;
	thread_local TImage<Point3f> imageAdj; // == MeshTexture::ColorMap (Color=Point3f); Color is a private member typedef, not visible here
	thread_local std::vector<uint8_t> mask;
} }

void MeshTexture::GlobalSeamLeveling()
{
	TEX_PROFILE_SCOPE("GlobalSeamLeveling (total)");
	ASSERT(!seamVertices.empty());
	const unsigned numPatches(texturePatches.size());

	// ------------------------------------------------------------
	// Build vertex -> sorted-unique patch adjacency via a flat CSR layout.
	//
	// The old approach allocated T * numVertices tiny std::vectors (millions
	// of empty vector headers) and grew each per-(thread,vertex) bucket with
	// push_back, then merged by scanning all T buckets for every vertex and
	// dedup'd each with a full std::sort + std::unique. Here we instead size
	// an exact flat buffer from an incidence count, scatter patch ids once
	// with no reallocation, then dedup each (tiny) per-vertex run in place
	// with a single-pass sorted insert. The dominant interior-vertex case
	// (all incidences share one patch id) collapses in O(n) with zero shifts.
	// ------------------------------------------------------------

	const size_t numVertices = vertices.size();

	// Pass 1: count face incidences per vertex (valid patches only).
	std::vector<size_t> offsets(numVertices + 1, 0);
#pragma omp parallel for schedule(dynamic, 64)
	for (int64_t p = 0; p < (int64_t)texturePatches.size(); ++p) {
		if (texturePatches[p].label == NO_ID)
			continue;
		for (const FIndex f : texturePatches[p].faces) {
			const Face& face = faces[f];
			for (int k = 0; k < 3; ++k) {
#pragma omp atomic
				++offsets[face[k] + 1];
			}
		}
	}

	// Exclusive prefix sum -> CSR row offsets [offsets[v], offsets[v+1]).
	for (size_t v = 0; v < numVertices; ++v)
		offsets[v + 1] += offsets[v];
	const size_t totalIncidences = offsets[numVertices];

	// Pass 2: scatter patch ids into the flat buffer. `cursor` starts at each
	// vertex's base offset and is bumped per write (no reallocs). Uses a C++
	// std::atomic fetch_add rather than `#pragma omp atomic capture` so it
	// builds under MSVC's default OpenMP 2.0 (which lacks the capture clause).
	std::vector<uint32_t> flatPatches(totalIncidences);
	{
		std::vector<std::atomic<size_t>> cursor(numVertices);
		for (size_t v = 0; v < numVertices; ++v)
			cursor[v].store(offsets[v], std::memory_order_relaxed);
#pragma omp parallel for schedule(dynamic, 64)
		for (int64_t p = 0; p < (int64_t)texturePatches.size(); ++p) {
			if (texturePatches[p].label == NO_ID)
				continue;
			for (const FIndex f : texturePatches[p].faces) {
				const Face& face = faces[f];
				for (int k = 0; k < 3; ++k) {
					const size_t pos = cursor[face[k]].fetch_add(1, std::memory_order_relaxed);
					flatPatches[pos] = (uint32_t)p;
				}
			}
		}
	}

	// Dedup each per-vertex run in place (sorted insert) and record the unique
	// count. Runs are tiny: interior vertices touch exactly one patch, seam
	// vertices a handful, so this is cheaper than sort+unique and keeps the
	// ascending order the downstream two-pointer merges rely on.
	std::vector<VertexPatchRow> vertpatch2rows(numVertices);
	std::vector<MatIdx> uniqCount(numVertices);
#pragma omp parallel for schedule(static)
	for (int64_t v = 0; v < (int64_t)numVertices; ++v) {
		uint32_t* const first = flatPatches.data() + offsets[v];
		uint32_t* const last = flatPatches.data() + offsets[v + 1];
		uint32_t* out = first; // [first,out) stays sorted & unique
		for (uint32_t* it = first; it != last; ++it) {
			const uint32_t val = *it;
			uint32_t* pos = out;
			while (pos != first && *(pos - 1) > val)
				--pos;
			if (pos != first && *(pos - 1) == val)
				continue; // duplicate (only possible at pos-1, the largest <= val)
			for (uint32_t* q = out; q != pos; --q)
				*q = *(q - 1);
			*pos = val;
			++out;
		}
		const size_t n = (size_t)(out - first);
		uniqCount[v] = (MatIdx)n;
		vertpatch2rows[v].patches.assign(first, first + n);
	}

	// Assign a contiguous row index to each (vertex,patch) pair.
	MatIdx rowsX = 0;
	std::vector<MatIdx> base(numVertices);
	for (size_t v = 0; v < numVertices; ++v) {
		base[v] = rowsX;
		rowsX += uniqCount[v];
	}

#pragma omp parallel for schedule(static)
	for (int64_t v = 0; v < (int64_t)numVertices; ++v) {
		auto& dst = vertpatch2rows[v];
		const MatIdx b = base[v];
		const size_t n = dst.patches.size();
		dst.rows.resize(n);
		for (size_t i = 0; i < n; ++i)
			dst.rows[i] = b + (MatIdx)i;
	}

	// fill Tikhonov's Gamma matrix (regularization constraints)
	const float lambda = 0.1f;
	const float w = lambda * lambda;

	CLISTDEF0(MatEntry) AtATriplets; // sized exactly in the merge below

	Eigen::MatrixXf Atb = Eigen::MatrixXf::Zero(rowsX, 3);

	const int numThreads = omp_get_max_threads();

	std::vector<CLISTDEF0(MatEntry)> threadTriplets(numThreads);
	std::vector<std::vector<std::pair<MatIdx, Color>>> threadAtb(numThreads);

#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		auto& localTriplets = threadTriplets[tid];
		auto& localAtb = threadAtb[tid];

		localTriplets.Reserve(8192);
		localAtb.reserve(4096);

		boost::container::small_vector<VIndex, 32> adjVerts;
		std::vector<uint32_t> indices;
		Colors vertexColors;
		std::vector<MatIdx> cols;

#pragma omp for schedule(static)
		for (int64_t v = 0; v < (int64_t)vertices.size(); ++v) {
			// -------- GAMMA (regularization) --------
			adjVerts.clear();
			scene.mesh.GetAdjVertices(v, adjVerts);

			const auto& rowMapV = vertpatch2rows[v];

			for (const VIndex vAdj : adjVerts) {
				if (v >= vAdj)
					continue;

				const auto& rowMapAdj = vertpatch2rows[vAdj];

				// two pointer merge
				size_t i = 0, j = 0;

				while (i < rowMapV.patches.size() &&
					j < rowMapAdj.patches.size()) {
					uint32_t p0 = rowMapV.patches[i];
					uint32_t p1 = rowMapAdj.patches[j];

					if (p0 == p1) {
						MatIdx col0 = rowMapV.rows[i];
						MatIdx col1 = rowMapAdj.rows[j];

						localTriplets.Insert(MatEntry(col0, col0, w));
						localTriplets.Insert(MatEntry(col1, col1, w));

						if (col0 >= col1)
							localTriplets.Insert(MatEntry(col0, col1, -w));
						else
							localTriplets.Insert(MatEntry(col1, col0, -w));

						++i; ++j;
					}
					else if (p0 < p1) ++i;
					else ++j;
				}
			}
		}

		// -------- SEAM CONSTRAINTS --------
#pragma omp for schedule(static)
		for (int s = 0; s < (int)seamVertices.size(); ++s) {
			const SeamVertex& seamVertex = seamVertices[s];

			if (seamVertex.patches.size() < 2)
				continue;

			// ---- sort patch indices ----
			seamVertex.SortByPatchIndex(indices);

			const size_t n = indices.size();

			vertexColors.resize(n);
			cols.resize(n);

			const auto& rowMap =
				vertpatch2rows[seamVertex.idxVertex];

			// ---- map patches -> matrix columns (safe two-pointer) ----
			size_t rp = 0;
			const size_t rowCount = rowMap.patches.size();

			for (size_t i = 0; i < n; ++i) {
				uint32_t patchId =
					seamVertex.patches[indices[i]].idxPatch;

				while (rp < rowCount &&
					rowMap.patches[rp] < patchId)
					++rp;

				ASSERT(rp < rowCount);
				cols[i] = rowMap.rows[rp];
			}

			// ---- sample colors per patch ----
			for (size_t i = 0; i < n; ++i) {
				const SeamVertex::Patch& patch0 =
					seamVertex.patches[indices[i]];

				SampleImage sampler(
					PatchSrcImage(patch0.idxPatch)
				);

				for (const auto& edge : patch0.edges) {
					const SeamVertex& sv1 =
						seamVertices[edge.idxSeamVertex];

					uint32_t idxPatch1 = UINT32_MAX;

					for (uint32_t k = 0; k < sv1.patches.size(); ++k) {
						if (sv1.patches[k].idxPatch == patch0.idxPatch) {
							idxPatch1 = k;
							break;
						}
					}
					if (idxPatch1 == NO_ID)
						continue; // safety guard

					const SeamVertex::Patch& patch1 =
						sv1.patches[idxPatch1];

					sampler.AddEdge(patch0.proj, patch1.proj);
				}

				vertexColors[i] = sampler.GetColor();
			}

			// ---- pair constraints ----
			for (size_t i = 0; i + 1 < n; ++i) {
				MatIdx col0 = cols[i];

				for (size_t j = i + 1; j < n; ++j) {
					MatIdx col1 = cols[j];

					Color delta =
						vertexColors[j] - vertexColors[i];

					// diagonal terms
					localTriplets.Insert(MatEntry(col0, col0, 1.f));
					localTriplets.Insert(MatEntry(col1, col1, 1.f));

					// off-diagonal (lower triangle only)
					if (col0 >= col1)
						localTriplets.Insert(MatEntry(col0, col1, -1.f));
					else
						localTriplets.Insert(MatEntry(col1, col0, -1.f));

					// RHS
					localAtb.emplace_back(col0, delta);
					localAtb.emplace_back(
						col1,
						Color(-delta[0], -delta[1], -delta[2])
					);
				}
			}
		}
	}

	// -------- MERGE --------
	// Concatenate the per-thread triplet lists into one contiguous array.
	// The lists are disjoint and MatEntry is trivially copyable, so we size
	// the destination exactly once (thread blocks + the Tikhonov diagonal)
	// and let threads memcpy their block into place in parallel. The old
	// code copied every triplet one-by-one through Insert() on a single
	// thread, reallocating as it grew.
	const float eps = 1e-6f;

	std::vector<IDX> triOffset(numThreads + 1, 0);
	for (int t = 0; t < numThreads; ++t)
		triOffset[t + 1] = triOffset[t] + threadTriplets[t].GetSize();
	const IDX totalTri = triOffset[numThreads];

	AtATriplets.Resize(totalTri + (IDX)rowsX);

#pragma omp parallel for schedule(dynamic, 1)
	for (int t = 0; t < numThreads; ++t) {
		const IDX n = threadTriplets[t].GetSize();
		if (n)
			std::memcpy(AtATriplets.Begin() + triOffset[t],
				threadTriplets[t].Begin(),
				(size_t)n * sizeof(MatEntry));
	}

	// Tikhonov diagonal epsilon occupies the disjoint tail region.
#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < (int64_t)rowsX; ++i)
		AtATriplets[totalTri + (IDX)i] = MatEntry((MatIdx)i, (MatIdx)i, eps);

	// Atb accumulation stays serial: seam rows collide across threads and
	// the volume (seam constraints only) is tiny next to the triplets.
	for (int t = 0; t < numThreads; ++t)
		for (auto& acc : threadAtb[t]) {
			Atb(acc.first, 0) += acc.second[0];
			Atb(acc.first, 1) += acc.second[1];
			Atb(acc.first, 2) += acc.second[2];
		}

	SparseMatRM Lhs(rowsX, rowsX);
	Lhs.reserve(AtATriplets.GetSize());

	Lhs.setFromTriplets(
		AtATriplets.Begin(),
		AtATriplets.End(),
		std::plus<float>()
	);
	// Force CSR compression (important for RowMajor CG)
	Lhs.makeCompressed();

	Eigen::setNbThreads(1); // Eigen now single-threaded

	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
		colorAdjustments(rowsX, 3);

	Eigen::ConjugateGradient<
		SparseMatRM,
		Eigen::Lower,
		Eigen::DiagonalPreconditioner<float>
	> solver;

	// Do not adjust these.  Lowering this
	// in the spirit of better performance is likely to
	// reduce quality significantly.
	solver.setMaxIterations(1000);
	solver.setTolerance(1e-4f);
	solver.compute(Lhs);
	ASSERT(solver.info() == Eigen::Success);

#pragma omp parallel for num_threads(3) schedule(static)
	for (int c = 0; c < 3; ++c) {
		DWORD_PTR oldMask = PinThreadToCoreAndSave(c * 2);

		Eigen::Ref<const Eigen::VectorXf> rhs(Atb.col(c));
		Eigen::VectorXf x = solver.solve(rhs);
		ASSERT(solver.info() == Eigen::Success);

		x.array() -= x.mean();

		colorAdjustments.col(c) = x;

		RestoreThreadAffinity(oldMask);
	}

	Eigen::setNbThreads(0); // restore

	// adjust texture patches using the correction colors
	// Build direct vertex->row lookup for this patch
#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(dynamic) // Much faster as dynamic
	for (int i = 0; i < (int)numPatches; ++i) {
#else
	for (unsigned i = 0; i < numPatches; ++i) {
#endif
		const uint32_t idxPatch = (uint32_t)i;
		TexturePatch& texturePatch = texturePatches[idxPatch];

		// ---- build row indices once for this patch ----
		auto& faceRowIndices = gseam::faceRowIndices; // thread_local scratch (released post-seam)

		const size_t needed = texturePatch.faces.size() * 3;
		if (faceRowIndices.size() < needed)
			faceRowIndices.resize(needed);

		for (size_t fIdx = 0; fIdx < texturePatch.faces.size(); ++fIdx) {
			const FIndex idxFace = texturePatch.faces[fIdx];
			const Face& face = faces[idxFace];

			MatIdx* rowIdx = &faceRowIndices[fIdx * 3];

			for (int k = 0; k < 3; ++k) {
				const VIndex v = face[k];
				const auto& rowMap = vertpatch2rows[v];

				// small linear scan (patch count per vertex is tiny)
				const auto& patches = rowMap.patches;
				const size_t n = patches.size();

				for (size_t j = 0; j < n; ++j)
					if (patches[j] == idxPatch) {
						rowIdx[k] = rowMap.rows[j];
						break;
					}
			}
		}

		auto& imageAdj = gseam::imageAdj; // thread_local scratch (released post-seam)
		if (imageAdj.size() != texturePatch.rect.size())
			imageAdj.create(texturePatch.rect.size());

		// Faster than custom memset if contiguous
		std::memset(imageAdj.data, 0,
			imageAdj.width() * imageAdj.height() * sizeof(Color));

		auto& mask = gseam::mask; // thread_local scratch (released post-seam)
		int w = texturePatch.rect.width;
		const int h = texturePatch.rect.height;
		const int n = w * h;

		if ((int)mask.size() < n)
			mask.resize(n);

		std::memset(mask.data(), 0, n);		// interpolate color adjustments over the whole patch

		struct RasterPatch {
			const TexCoord* tri;
			Color colors[3];
			ColorMap& image;
			inline RasterPatch(ColorMap& _image) : image(_image) {}
			inline cv::Size Size() const { return image.size(); }
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				ASSERT(image.isInside(pt));
				image(pt) = colors[0] * bary.x + colors[1] * bary.y + colors[2] * bary.z;
			}
		} data(imageAdj);

		for (size_t fIdx = 0; fIdx < texturePatch.faces.size(); ++fIdx) {
			const FIndex idxFace = texturePatch.faces[fIdx];
			const Face& face = faces[idxFace];

			data.tri = faceTexcoords.data() + idxFace * 3;

			MatIdx* rowIdx = &faceRowIndices[fIdx * 3];

			for (int k = 0; k < 3; ++k)
				data.colors[k] = colorAdjustments.row(rowIdx[k]);

			ColorMap::RasterizeTriangleBaryMasked(
				data.tri[0], data.tri[1], data.tri[2], data, mask.data());
		}

		// dilate with one pixel width, in order to make sure patch border smooths out a little
		// Fused and optimized for DilateMean<1> case
		const Image8U3& imgImage = PatchSrcImage(idxPatch);

		cv::Rect roi = texturePatch.rect;

		const int imgW = imgImage.cols;
		const int imgH = imgImage.rows;

		// Clamp left/top by shrinking
		if (roi.x < 0) {
			roi.width += roi.x;
			roi.x = 0;
		}
		if (roi.y < 0) {
			roi.height += roi.y;
			roi.y = 0;
		}

		// Clamp right/bottom
		if (roi.x + roi.width > imgW) {
			roi.width = imgW - roi.x;
		}
		if (roi.y + roi.height > imgH) {
			roi.height = imgH - roi.y;
		}

		// If it collapsed, skip
		if (roi.width <= 0 || roi.height <= 0) {
			continue;
		}

		// Use roi (NOT texturePatch.rect)
		cv::Mat image(imgImage(roi));
		uint8_t* maskData = mask.data();

		w = image.cols;

		for (int r = 1; r < image.rows - 1; ++r) {
			const Color* __restrict prev = (Color*)imageAdj.ptr(r - 1);
			const Color* __restrict curr = (Color*)imageAdj.ptr(r);
			const Color* __restrict next = (Color*)imageAdj.ptr(r + 1);

			uint8_t* __restrict maskPrev = mask.data() + (r - 1) * w;
			uint8_t* __restrict maskCurr = mask.data() + r * w;
			uint8_t* __restrict maskNext = mask.data() + (r + 1) * w;

			Pixel8U* __restrict out = image.ptr<Pixel8U>(r);

			for (int c = 1; c < w - 1; ++c) {
				Color a;

				if (maskCurr[c]) {
					// pixel already has correction
					a = curr[c];
				}
				else {
					Color sum(0);
					int n = 0;

					if (maskPrev[c - 1]) { sum += prev[c - 1]; ++n; }
					if (maskPrev[c]) { sum += prev[c];     ++n; }
					if (maskPrev[c + 1]) { sum += prev[c + 1]; ++n; }

					if (maskCurr[c - 1]) { sum += curr[c - 1]; ++n; }
					if (maskCurr[c + 1]) { sum += curr[c + 1]; ++n; }

					if (maskNext[c - 1]) { sum += next[c - 1]; ++n; }
					if (maskNext[c]) { sum += next[c];     ++n; }
					if (maskNext[c + 1]) { sum += next[c + 1]; ++n; }

					if (!n)
						continue;

					a = sum * invTable[n];
				}

				Pixel8U& v = out[c];

				const Color deltaRGB = YCBCR_DELTA2RGB(a);
				Color acol = Color(v) + deltaRGB;

				__m128 rgbf = _mm_set_ps(0.0f, acol[2], acol[1], acol[0]);
				__m128i rgbi = _mm_cvtps_epi32(rgbf);

				const __m128i zero = _mm_setzero_si128();
				const __m128i max255 = _mm_set1_epi32(255);

				rgbi = _mm_max_epi32(rgbi, zero);
				rgbi = _mm_min_epi32(rgbi, max255);

				__m128i pack16 = _mm_packus_epi32(rgbi, rgbi);
				__m128i pack8 = _mm_packus_epi16(pack16, pack16);

				uint32_t rgb8 = (uint32_t)_mm_cvtsi128_si32(pack8);
				v[0] = (uint8_t)(rgb8 & 0xFF);
				v[1] = (uint8_t)((rgb8 >> 8) & 0xFF);
				v[2] = (uint8_t)((rgb8 >> 16) & 0xFF);
			}
		}
	}
}

// set to one in order to dilate also on the diagonal of the border
// (normally not needed)
#define DILATE_EXTRA 0

// Per-thread ProcessMask3 scratch, reused across patches so LocalSeamLeveling
// does not re-allocate the patch-sized visited/frontier buffers 168x per run.
// Namespace-scope thread_local (not function-local static) so ReleaseSeamScratch
// can free it after seam leveling. Pure scratch -> semantics-neutral.
namespace { namespace pmask {
	thread_local std::vector<int> frontier;
	thread_local std::vector<int> nextFrontier;
	thread_local std::vector<uint8_t> visited;
} }

void MeshTexture::ProcessMask3(Image8U& mask)
{
	typedef Image8U::Type Type;

	const int width = mask.width();
	const int height = mask.height();
	const int stride = width;

	Type* data = (Type*)mask.data;

	auto Idx = [&](int x, int y) {
		return y * stride + x;
		};

	// ------------------------------------------------------------
	// 1) DILATE border -> interior (4-neighborhood)
	// ------------------------------------------------------------
	for (int y = 1; y < height - 1; ++y) {
		Type* row = data + y * stride;
		for (int x = 1; x < width - 1; ++x) {
			if (row[x] != border)
				continue;

			Type& up = data[(y - 1) * stride + x];
			Type& down = data[(y + 1) * stride + x];
			Type& left = row[x - 1];
			Type& right = row[x + 1];

			if (up != border) up = interior;
			if (down != border) down = interior;
			if (left != border) left = interior;
			if (right != border) right = interior;
		}
	}

	// ------------------------------------------------------------
	// 2) ERODE interior -> empty (edge consistency)
	// ------------------------------------------------------------
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Type& v = data[y * stride + x];
			if (v != interior)
				continue;

			auto Sample = [&](int xx, int yy) -> Type {
				if ((unsigned)xx >= (unsigned)width ||
					(unsigned)yy >= (unsigned)height)
					return empty;
				return data[yy * stride + xx];
				};

			// horizontal
			if ((Sample(x - 1, y) == border && Sample(x + 1, y) == empty) ||
				(Sample(x + 1, y) == border && Sample(x - 1, y) == empty)) {
				v = empty;
				continue;
			}

			// vertical
			if ((Sample(x, y - 1) == border && Sample(x, y + 1) == empty) ||
				(Sample(x, y + 1) == border && Sample(x, y - 1) == empty)) {
				v = empty;
				continue;
			}

			// diagonals
			if ((Sample(x - 1, y - 1) == border && Sample(x + 1, y + 1) == empty) ||
				(Sample(x + 1, y + 1) == border && Sample(x - 1, y - 1) == empty) ||
				(Sample(x - 1, y + 1) == border && Sample(x + 1, y - 1) == empty) ||
				(Sample(x + 1, y - 1) == border && Sample(x - 1, y + 1) == empty)) {
				v = empty;
				continue;
			}
		}
	}

	// ------------------------------------------------------------
	// 3) Mark interior pixels touching empty as border
	// ------------------------------------------------------------
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			Type& v = data[y * stride + x];
			if (v != interior)
				continue;

			if (data[(y - 1) * stride + x] == empty ||
				data[(y + 1) * stride + x] == empty ||
				data[y * stride + x - 1] == empty ||
				data[y * stride + x + 1] == empty) {
				v = border;
			}
		}
	}

	const int nPix = width * height;

	// reused thread_local scratch (released post-seam via ReleaseSeamScratch)
	std::vector<int>& frontier = pmask::frontier;
	std::vector<int>& nextFrontier = pmask::nextFrontier;
	std::vector<uint8_t>& visited = pmask::visited;
	frontier.clear();
	nextFrontier.clear();
	frontier.reserve(width * 4);        // heuristic
	nextFrontier.reserve(width * 4);

	visited.assign(nPix, 0);

	// ------------------------------------------------------------
	// 1) Initial frontier (single full scan)
	// ------------------------------------------------------------
	for (int y = 0; y < height; ++y) {
		const int row = y * stride;

		for (int x = 0; x < width; ++x) {
			const int idx = row + x;

			if (data[idx] == empty)
				continue;

			bool touchesEmpty = false;

			const int y0 = (y > 0) ? y - 1 : y;
			const int y1 = (y + 1 < height) ? y + 1 : y;
			const int x0 = (x > 0) ? x - 1 : x;
			const int x1 = (x + 1 < width) ? x + 1 : x;

			for (int yy = y0; yy <= y1 && !touchesEmpty; ++yy) {
				const int r2 = yy * stride;
				for (int xx = x0; xx <= x1; ++xx) {
					if (data[r2 + xx] == empty) {
						touchesEmpty = true;
						break;
					}
				}
			}

			if (touchesEmpty) {
				frontier.push_back(idx);
				visited[idx] = 1;
			}
		}
	}

	// ------------------------------------------------------------
	// 2) 3 layers of erosion using compact frontier
	// ------------------------------------------------------------
	for (int pass = 0; pass < 3 && !frontier.empty(); ++pass) {

		// Remove current frontier
		for (int idx : frontier)
			data[idx] = empty;

		nextFrontier.clear();

		// Expand from frontier only
		for (int idx : frontier) {

			const int y = idx / stride;
			const int x = idx - y * stride;

			const int y0 = (y > 0) ? y - 1 : y;
			const int y1 = (y + 1 < height) ? y + 1 : y;
			const int x0 = (x > 0) ? x - 1 : x;
			const int x1 = (x + 1 < width) ? x + 1 : x;

			for (int yy = y0; yy <= y1; ++yy) {
				const int r2 = yy * stride;

				for (int xx = x0; xx <= x1; ++xx) {
					const int nidx = r2 + xx;

					if (data[nidx] == empty)
						continue;
					if (visited[nidx])
						continue;

					visited[nidx] = 1;
					nextFrontier.push_back(nidx);
				}
			}
		}

		frontier.swap(nextFrontier);
	}

	// ------------------------------------------------------------
	// 3) Mark remaining frontier as border
	// ------------------------------------------------------------
	for (int idx : frontier)
		data[idx] = border;
}

inline MeshTexture::Color ColorLaplacian(const Image32F3& img, int i) {
	const int width(img.width());
	return img(i-width) + img(i-1) + img(i+1) + img(i+width) - img(i)*4.f;
}

#if 1
struct Float3 {
	float x, y, z;
};

static inline float Dot3_SSE(const Float3& a, const Float3& b)
{
	__m128 va = _mm_set_ps(0.f, a.z, a.y, a.x);
	__m128 vb = _mm_set_ps(0.f, b.z, b.y, b.x);
	__m128 m = _mm_mul_ps(va, vb);

	__m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 1, 0, 3));
	__m128 sums = _mm_add_ps(m, shuf);
	shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
	sums = _mm_add_ps(sums, shuf);

	return _mm_cvtss_f32(sums);
}

#include <emmintrin.h>
#include <vector>

static inline float Dot3Sse2(const Float3& a, const Float3& b)
{
	__m128 va = _mm_set_ps(0.0f, a.z, a.y, a.x);
	__m128 vb = _mm_set_ps(0.0f, b.z, b.y, b.x);
	__m128 m = _mm_mul_ps(va, vb);

	__m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 1, 0, 3));
	__m128 sums = _mm_add_ps(m, shuf);
	shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
	sums = _mm_add_ps(sums, shuf);

	return _mm_cvtss_f32(sums);
}

inline int Idx(int x, int y, int w)
{
	return y * w + x;
}

struct PoissonStencil {
	MatIdx up;
	MatIdx left;
	MatIdx right;
	MatIdx down;
};

#define MAX_ABS3_SSE2(rR, rG, rB, out) do {            \
  __m128 v = _mm_set_ps(0.0f, (rB), (rG), (rR));       \
  __m128 sign = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)); \
  v = _mm_and_ps(v, sign);                             \
                                                        \
  __m128 t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1)); \
  v = _mm_max_ps(v, t);                                \
                                                        \
  t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,0,3,2));      \
  v = _mm_max_ps(v, t);                                \
                                                        \
  (out) = _mm_cvtss_f32(v);                             \
} while (0)

#ifdef COUNT_ITERATIONS
std::atomic<int> calls = 0;
std::atomic<int> iterations = 0;
#endif

static void SolvePoissonSOR_Compact(
	const PoissonStencil* __restrict stencil,
	const float* __restrict bR,
	const float* __restrict bG,
	const float* __restrict bB,
	float* __restrict xR,
	float* __restrict xG,
	float* __restrict xB,
	std::vector<MatIdx>& redInterior,
	std::vector<MatIdx>& blackInterior,
	float omega = 1.85f,
	float tol = 1e-6f,
	int maxIters = 400)
{
	float maxResidual = 0.0f;

#ifdef COUNT_ITERATIONS
	++calls;
	int lIters = 0;
#endif

	float initialResidual = -1.0f;

	for (int iter = 0; iter < maxIters; ++iter) {
#ifdef COUNT_ITERATIONS
		++lIters;
#endif
		// Red pass
		for (int k = 0, cnt = (int)redInterior.size(); k < cnt; ++k) {
			int i = redInterior[k];

			const PoissonStencil& s = stencil[i];

			const int u = s.up;
			const int l = s.left;
			const int r = s.right;
			const int d = s.down;

			const float sumR = xR[u] + xR[l] + xR[r] + xR[d] - bR[i];
			const float sumG = xG[u] + xG[l] + xG[r] + xG[d] - bG[i];
			const float sumB = xB[u] + xB[l] + xB[r] + xB[d] - bB[i];

			const float newR = sumR * 0.25f;
			const float newG = sumG * 0.25f;
			const float newB = sumB * 0.25f;

			xR[i] += omega * (newR - xR[i]);
			xG[i] += omega * (newG - xG[i]);
			xB[i] += omega * (newB - xB[i]);
		}

		// Black pass
		for (int k = 0, cnt = (int)blackInterior.size(); k < cnt; ++k) {
			int i = blackInterior[k];

			const PoissonStencil& s = stencil[i];

			const int u = s.up;
			const int l = s.left;
			const int r = s.right;
			const int d = s.down;

			const float sumR = xR[u] + xR[l] + xR[r] + xR[d] - bR[i];
			const float sumG = xG[u] + xG[l] + xG[r] + xG[d] - bG[i];
			const float sumB = xB[u] + xB[l] + xB[r] + xB[d] - bB[i];

			const float newR = sumR * 0.25f;
			const float newG = sumG * 0.25f;
			const float newB = sumB * 0.25f;

			xR[i] += omega * (newR - xR[i]);
			xG[i] += omega * (newG - xG[i]);
			xB[i] += omega * (newB - xB[i]);
		}

#if 0 // Removing residual check.  Hardcoding at a fixed # of iterations.
		// Residual check
		if ((iter & 7) == 0) {
			maxResidual = 0.0f;

			for (int k = 0, cnt = (int)redInterior.size(); k < cnt; ++k) {
				int i = redInterior[k];

				const PoissonStencil& s = stencil[i];

				int u = s.up;
				int l = s.left;
				int r = s.right;
				int d = s.down;

				float rR =
					-4.0f * xR[i] +
					xR[u] + xR[l] + xR[r] + xR[d] -
					bR[i];

				float rG =
					-4.0f * xG[i] +
					xG[u] + xG[l] + xG[r] + xG[d] -
					bG[i];

				float rB =
					-4.0f * xB[i] +
					xB[u] + xB[l] + xB[r] + xB[d] -
					bB[i];

				float a;
				MAX_ABS3_SSE2(rR, rG, rB, a);

				if (a > maxResidual)
					maxResidual = a;
		}

			for (int k = 0, cnt = (int)blackInterior.size(); k < cnt; ++k) {
				int i = blackInterior[k];

				const PoissonStencil& s = stencil[i];

				int u = s.up;
				int l = s.left;
				int r = s.right;
				int d = s.down;

				float rR =
					-4.0f * xR[i] +
					xR[u] + xR[l] + xR[r] + xR[d] -
					bR[i];

				float rG =
					-4.0f * xG[i] +
					xG[u] + xG[l] + xG[r] + xG[d] -
					bG[i];

				float rB =
					-4.0f * xB[i] +
					xB[u] + xB[l] + xB[r] + xB[d] -
					bB[i];

				float a;
				MAX_ABS3_SSE2(rR, rG, rB, a);

				if (a > maxResidual)
					maxResidual = a;
			}

			if (initialResidual < 0.0f) {
				initialResidual = maxResidual;
			}

			// relative + absolute stopping
			if (maxResidual < tol || maxResidual < initialResidual * 1e-3f) {
			break;
			}
		}
#endif
	}

#ifdef COUNT_ITERATIONS
	iterations += lIters;
#endif
}

// Per-thread Poisson (LocalSeamLeveling) scratch; namespace-scope thread_local
// so ReleaseSeamScratch can free it after seam leveling instead of it staying
// pinned (sized to the largest patch x every worker) for the whole run.
namespace { namespace pseam {
	thread_local TImage<MatIdx> indices;
	thread_local std::vector<PoissonStencil> stencil;
	thread_local std::vector<float> xR, xG, xB, bR, bG, bB;
	thread_local std::vector<MatIdx> redInterior, blackInterior;
} }

// Free every per-thread seam-leveling scratch buffer. Must be called once per
// worker (from an omp parallel region) so each thread releases its OWN copies.
// Pure scratch -> semantics-neutral; buffers are re-grown on next use.
static void ReleaseSeamScratch()
{
	std::vector<MatIdx>().swap(gseam::faceRowIndices);
	gseam::imageAdj.release();
	std::vector<uint8_t>().swap(gseam::mask);
	pseam::indices.release();
	std::vector<PoissonStencil>().swap(pseam::stencil);
	std::vector<float>().swap(pseam::xR); std::vector<float>().swap(pseam::xG); std::vector<float>().swap(pseam::xB);
	std::vector<float>().swap(pseam::bR); std::vector<float>().swap(pseam::bG); std::vector<float>().swap(pseam::bB);
	std::vector<MatIdx>().swap(pseam::redInterior);
	std::vector<MatIdx>().swap(pseam::blackInterior);
	std::vector<int>().swap(pmask::frontier);
	std::vector<int>().swap(pmask::nextFrontier);
	std::vector<uint8_t>().swap(pmask::visited);
}

void MeshTexture::PoissonBlendingNoBias(
	const Image32F3& src,
	Image32F3& dst,
	const Image8U& mask)
{
	ASSERT(src.width() == mask.width() && src.width() == dst.width());
	ASSERT(src.height() == mask.height() && src.height() == dst.height());
	ASSERT(src.channels() == 3 && dst.channels() == 3 && mask.channels() == 1);
	ASSERT(src.type() == CV_32FC3 && dst.type() == CV_32FC3 && mask.type() == CV_8U);

	const int width = dst.width();
	const int height = dst.height();
	const int n = width * height;

	// Compact indexing (now using tiles)
	auto& indices = pseam::indices; // thread_local scratch (released post-seam)

	if (indices.size() != dst.size()) {
		indices.create(dst.size());   // or whatever alloc method TImage uses
	}

	indices.memset(0xff);

	MatIdx nnz = 0;

	// Pass 1: interior pixels first (solver hot set)
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const int i = y * width + x;
			if (mask(i) == interior) {
				indices(i) = nnz++;
			}
		}
	}

	// Pass 2: border pixels (cold, mostly fixed)
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const int i = y * width + x;
			if (mask(i) == border)
				indices(i) = nnz++;
		}
	}

	auto& stencil = pseam::stencil; // thread_local scratch (released post-seam)
	auto& xR = pseam::xR; auto& xG = pseam::xG; auto& xB = pseam::xB;
	auto& bR = pseam::bR; auto& bG = pseam::bG; auto& bB = pseam::bB;
	auto& redInterior = pseam::redInterior;
	auto& blackInterior = pseam::blackInterior;

	stencil.resize(nnz);
	xR.resize(nnz);
	xG.resize(nnz);
	xB.resize(nnz);
	bR.resize(nnz);
	bG.resize(nnz);
	bB.resize(nnz);

	redInterior.clear();
	blackInterior.clear();
	redInterior.reserve(nnz);
	blackInterior.reserve(nnz);

	float* __restrict xRp = xR.data();
	float* __restrict xGp = xG.data();
	float* __restrict xBp = xB.data();
	float* __restrict bRp = bR.data();
	float* __restrict bGp = bG.data();
	float* __restrict bBp = bB.data();

	// Build compact system
	for (int y = 0; y < height; ++y) {
		const float* __restrict srcPrev = (y > 0) ? src.ptr<float>(y - 1) : nullptr;
		const float* __restrict srcCur = src.ptr<float>(y);
		const float* __restrict srcNext = (y + 1 < height) ? src.ptr<float>(y + 1) : nullptr;
		const int rowBase = y * width;

		for (int x = 0; x < width; ++x) {
			const int i = rowBase + x;
			const MatIdx idx = indices(i);
			if (idx == (MatIdx)-1)
				continue;

			PoissonStencil& s = stencil[idx];

			if (mask(i) == border) {
				s.up = s.left = s.right = s.down = idx;

				const Color& c0 = (const Color&)dst(i);
				xRp[idx] = c0.x;
				xGp[idx] = c0.y;
				xBp[idx] = c0.z;

				continue;
			}

			// ---- Neighbor indices: idxOrSelf inline, no lambdas ----
			// up
			if (y > 0) {
				const MatIdx t = indices(i - width);
				s.up = (t != (MatIdx)-1) ? t : idx;
			}
			else {
				s.up = idx;
			}

			// down
			if (y + 1 < height) {
				const MatIdx t = indices(i + width);
				s.down = (t != (MatIdx)-1) ? t : idx;
			}
			else {
				s.down = idx;
			}

			// left
			if (x > 0) {
				const MatIdx t = indices(i - 1);
				s.left = (t != (MatIdx)-1) ? t : idx;
			}
			else {
				s.left = idx;
			}

			// right
			if (x + 1 < width) {
				const MatIdx t = indices(i + 1);
				s.right = (t != (MatIdx)-1) ? t : idx;
			}
			else {
				s.right = idx;
			}

			// ---- src Laplacian: use the three row pointers, no src.ptr in inner loop ----
			const int c = 3 * x;

			const float cR = srcCur[c + 0];
			const float cG = srcCur[c + 1];
			const float cB = srcCur[c + 2];

			const float uR = (srcPrev) ? srcPrev[c + 0] : cR;
			const float uG = (srcPrev) ? srcPrev[c + 1] : cG;
			const float uB = (srcPrev) ? srcPrev[c + 2] : cB;

			const float dR = (srcNext) ? srcNext[c + 0] : cR;
			const float dG = (srcNext) ? srcNext[c + 1] : cG;
			const float dB = (srcNext) ? srcNext[c + 2] : cB;

			const float lR = (x > 0) ? srcCur[c - 3] : cR;
			const float lG = (x > 0) ? srcCur[c - 2] : cG;
			const float lB = (x > 0) ? srcCur[c - 1] : cB;

			const float rR = (x + 1 < width) ? srcCur[c + 3] : cR;
			const float rG = (x + 1 < width) ? srcCur[c + 4] : cG;
			const float rB = (x + 1 < width) ? srcCur[c + 5] : cB;

			const float lSr = uR + lR + rR + dR - 4.0f * cR;
			const float lSg = uG + lG + rG + dG - 4.0f * cG;
			const float lSb = uB + lB + rB + dB - 4.0f * cB;

			bRp[idx] = lSr;
			bGp[idx] = lSg;
			bBp[idx] = lSb;

			// if useSrcOnly, keep original behavior if needed (either set from src or from dst(i))
			const Color& c0 = (const Color&)dst(i);
			xRp[idx] = c0.x;
			xGp[idx] = c0.y;
			xBp[idx] = c0.z;

			if (((x + y) & 1) == 0)
				redInterior.push_back(idx);
			else
				blackInterior.push_back(idx);
		}
	}

	if (redInterior.size() + blackInterior.size() < 100)
		return;

	// Can't sort this. std::sort(redInterior.begin(), redInterior.end());
	// Can't sort this. std::sort(blackInterior.begin(), blackInterior.end());

	// FAST SOLVE (this replaces Eigen)
	SolvePoissonSOR_Compact(
		stencil.data(),
		bRp,
		bGp,
		bBp,
		xRp,
		xGp,
		xBp,
		redInterior,
		blackInterior,
		1.7f,    // omega
		0.f, // drop tolereance1e-6f,    // tolerance
		25       // max iters (400 tested: no effect on the residual boundary
		         // lines, so LocalSeamLeveling convergence is NOT the cause;
		         // kept low for speed)
	);

	// Scatter back
	for (int y = 0; y < height; ++y) {
		float* __restrict row = dst.ptr<float>(y);
		const MatIdx* __restrict idxRow = indices.ptr<MatIdx>(y);

		for (int x = 0; x < width; ++x) {
			MatIdx idx = idxRow[x];
			if (idx == (MatIdx)-1)
				continue;

			float* __restrict d = row + 3 * x;

			d[0] = xRp[idx];
			d[1] = xGp[idx];
			d[2] = xBp[idx];
		}
	}
}

#else
void MeshTexture::PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias)
{
	ASSERT(src.width() == mask.width() && src.width() == dst.width());
	ASSERT(src.height() == mask.height() && src.height() == dst.height());
	ASSERT(src.channels() == 3 && dst.channels() == 3 && mask.channels() == 1);
	ASSERT(src.type() == CV_32FC3 && dst.type() == CV_32FC3 && mask.type() == CV_8U);

	#ifndef _RELEASE
	// check the mask border has no pixels marked as interior
	for (int x=0; x<mask.cols; ++x)
		ASSERT(mask(0,x) != interior && mask(mask.rows-1,x) != interior);
	for (int y=0; y<mask.rows; ++y)
		ASSERT(mask(y,0) != interior && mask(y,mask.cols-1) != interior);
	#endif

	const int n(dst.area());
	const int width(dst.width());

	TImage<MatIdx> indices(dst.size());
	indices.memset(0xff);
	MatIdx nnz(0);
	for (int i = 0; i < n; ++i)
		if (mask(i) != empty)
			indices(i) = nnz++;
	if (nnz <= 0)
		return;

	Colors coeffB(nnz);
	CLISTDEF0(MatEntry) coeffA(0, nnz);
	for (int i = 0; i < n; ++i) {
		switch (mask(i)) {
		case border: {
			const MatIdx idx(indices(i));
			ASSERT(idx != -1);
			coeffA.emplace_back(idx, idx, 1.f);
			coeffB[idx] = (const Color&)dst(i);
		} break;
		case interior: {
			const MatIdx idxUp(indices(i - width));
			const MatIdx idxLeft(indices(i - 1));
			const MatIdx idxCenter(indices(i));
			const MatIdx idxRight(indices(i + 1));
			const MatIdx idxDown(indices(i + width));
			// all indices should be either border conditions or part of the optimization
			ASSERT(idxUp != -1 && idxLeft != -1 && idxCenter != -1 && idxRight != -1 && idxDown != -1);
			coeffA.emplace_back(idxCenter, idxUp, 1.f);
			coeffA.emplace_back(idxCenter, idxLeft, 1.f);
			coeffA.emplace_back(idxCenter, idxCenter,-4.f);
			coeffA.emplace_back(idxCenter, idxRight, 1.f);
			coeffA.emplace_back(idxCenter, idxDown, 1.f);
			// set target coefficient
			coeffB[idxCenter] = (bias == 1.f ?
								 ColorLaplacian(src,i) :
								 ColorLaplacian(src,i)*bias + ColorLaplacian(dst,i)*(1.f-bias));
		} break;
		}
	}

	SparseMat A(nnz, nnz);
	A.setFromTriplets(coeffA.Begin(), coeffA.End());
	coeffA.Release();

	#ifdef TEXOPT_SOLVER_SPARSELU
	// use SparseLU factorization
	// (faster, but not working if EIGEN_DEFAULT_TO_ROW_MAJOR is defined, bug inside Eigen)
	const Eigen::SparseLU< SparseMat, Eigen::COLAMDOrdering<MatIdx> > solver(A);
	#else
	// use BiCGSTAB solver
	const Eigen::BiCGSTAB< SparseMat, Eigen::IncompleteLUT<float> > solver(A);
	#endif
	ASSERT(solver.info() == Eigen::Success);
	for (int channel=0; channel<3; ++channel) {
		const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.front().ptr()+channel, nnz);
		const Eigen::VectorXf x(solver.solve(b));
		ASSERT(solver.info() == Eigen::Success);
		for (int i = 0; i < n; ++i) {
			const MatIdx index(indices(i));
			if (index != -1)
				dst(i)[channel] = x[index];
		}
	}
}
#endif

#if 1
void MeshTexture::LocalSeamLeveling()
{
	TEX_PROFILE_SCOPE("LocalSeamLeveling (total)");
	ASSERT(!seamVertices.empty());
	const uint32_t numPatches = (uint32_t)texturePatches.size();

	// ----------------------------------------------------------------------
	// Optional precomputation: build fast O(1) lookup for each seam vertex
	// ----------------------------------------------------------------------
	for (SeamVertex& v : seamVertices) {

		// Patch lookup
		v.patchIndexLookup.clear();
		uint32_t cnt = v.patches.size();
		v.patchIndexLookup.reserve(cnt);

		for (uint32_t j = 0; j < cnt; ++j)
			v.patchIndexLookup.Insert(
				v.patches[j].idxPatch,
				(uint16_t)j
			);

		// Edge lookup per patch
		for (SeamVertex::Patch& p : v.patches) {

			p.edgeLookup.clear();
			p.edgeLookup.reserve((uint32_t)p.edges.size());

			for (uint32_t k = 0; k < p.edges.size(); ++k)
				p.edgeLookup.Insert(
					p.edges[k].idxSeamVertex,
					(uint16_t)k
				);
		}
	}

	std::vector<std::vector<uint32_t>> patchToSeamVertices(texturePatches.size());

	for (uint32_t v = 0; v < seamVertices.size(); ++v) {
		const auto& vertex = seamVertices[v];
		for (const auto& patch : vertex.patches)
			patchToSeamVertices[patch.idxPatch].push_back(v);
	}

	struct RasterPatch
	{
		Image32F3& image;
		Image8U& mask;
		const Image32F3& image0;
		const Image8U3& image1;

		TexCoord p0;
		TexCoord p0Dir;
		TexCoord p1;
		TexCoord p1Dir;

		float invLen2;

		Sampler sampler;

		inline RasterPatch(
			Image32F3& _image,
			Image8U& _mask,
			const Image32F3& _image0,
			const Image8U3& _image1,
			const TexCoord& _p0,
			const TexCoord& _p0Adj,
			const TexCoord& _p1,
			const TexCoord& _p1Adj)
			: image(_image),
			mask(_mask),
			image0(_image0),
			image1(_image1),
			p0(_p0),
			p0Dir(_p0Adj - _p0),
			p1(_p1),
			p1Dir(_p1Adj - _p1),
			sampler()
		{
			float len2 = p0Dir.x * p0Dir.x + p0Dir.y * p0Dir.y;
			invLen2 = (len2 > 1e-12f) ? (1.0f / len2) : 0.0f;
		}

		inline void operator()(const ImageRef& pt)
		{
			if (invLen2 == 0.0f)
				return;

			const int outW = image.width();
			const int outH = image.height();

			const int px = (int)pt.x;
			const int py = (int)pt.y;

			if ((unsigned)px >= (unsigned)outW || (unsigned)py >= (unsigned)outH) {
				return;
			}

			float dx = float(pt.x) - p0.x;
			float dy = float(pt.y) - p0.y;

			float l = (dx * p0Dir.x + dy * p0Dir.y) * invLen2;

			float sx0 = p0.x + p0Dir.x * l;
			float sy0 = p0.y + p0Dir.y * l;

			float sx1 = p1.x + p1Dir.x * l;
			float sy1 = p1.y + p1Dir.y * l;

			const Color c0 = image0.sample<Sampler, Color>(sampler, TexCoord(sx0, sy0));
			const Color c1 = image1.sample<Sampler, Color>(sampler, TexCoord(sx1, sy1));

			static constexpr float inv255 = 1.0f / 255.0f;

			float r = (c0[0] + c1[0] * inv255) * 0.5f;
			float g = (c0[1] + c1[1] * inv255) * 0.5f;
			float b = (c0[2] + c1[2] * inv255) * 0.5f;

			image(pt) = Color(r, g, b);
			mask(pt) = border;
		}
	};

	const int prevCvThreads = cv::getNumThreads();
	cv::setNumThreads(0);

	// Better to fully thread
  // Notice, we must use dynamic scheduling here as patch size can vary a lot and cause load imbalance.
	int T = omp_get_max_threads();

	// MEMORY-BUDGET THREAD CAP (see TEXTURE_SEAM_MEM_BUDGET_GB): the per-thread patch
	// buffers below (2x Image32F3 + mask + Poisson scratch, ~80 B/px) sized to the
	// LARGEST patch x T workers are the transient that sets the texturing peak on res-0.
	// Cap T so that footprint stays under the budget. Fewer workers -> identical output,
	// just less parallelism. 0 disables the cap.
	if (TEXTURE_SEAM_MEM_BUDGET_GB > 0) {
		uint64_t maxPatchArea = 0;
		for (uint32_t p = 0; p < numPatches; ++p) {
			const TexturePatch& tp = texturePatches[p];
			if (tp.label == NO_ID || tp.rect.width <= 0 || tp.rect.height <= 0)
				continue;
			const uint64_t area = (uint64_t)tp.rect.width * (uint64_t)tp.rect.height;
			if (area > maxPatchArea)
				maxPatchArea = area;
		}
		if (maxPatchArea > 0) {
			// Effective peak footprint per worker. The LIVE buffers are only ~77 B/px
			// (2x Image32F3 = 24, mask = 1, Poisson stencil = 16, 6 float x/b = 24,
			// red/black interior ~8, indices = 4). BUT the measured peak is ~4x that:
			// the per-patch cv::Mat::create()/vector resize() churn transiently holds
			// OLD+NEW during realloc, and the pseam:: vectors keep their LARGEST-patch
			// capacity across patches. A naive 80 B/px model never fired (cap computed
			// > core count) and the peak blew ~27 GB past an 8 GB budget on a real
			// 227-image res-0 scene, so use the empirically-observed ~256 B/px.
			const uint64_t perThreadBytes = maxPatchArea * 256ull;
			const uint64_t budgetBytes = (uint64_t)TEXTURE_SEAM_MEM_BUDGET_GB * 1024ull * 1024ull * 1024ull;
			int cap = (int)(budgetBytes / perThreadBytes);
			if (cap < 1)
				cap = 1;
			// Always log the decision (even when NOT capping) so the memory budget is
			// observable in the profile trace.
			DEBUG_EXTRA("LocalSeamLeveling: threads %d -> %d (largest patch %llu px, ~%.2f GB/thread est, %d GB budget)",
				T, (cap < T ? cap : T), (unsigned long long)maxPatchArea,
				perThreadBytes / (1024.0*1024.0*1024.0), (int)TEXTURE_SEAM_MEM_BUDGET_GB);
			if (cap < T)
				T = cap;
		}
	}

	std::vector<Image32F3> tlsImage((size_t)T);
	std::vector<Image32F3> tlsImageOrg((size_t)T);
	std::vector<Image8U> tlsMask((size_t)T);

#pragma omp parallel for schedule(dynamic, 4) num_threads(T)
	for (int i = 0; i < (int)numPatches; ++i) {
		const int tid = omp_get_thread_num();

		Image32F3& image = tlsImage[(size_t)tid];
		Image32F3& imageOrg = tlsImageOrg[(size_t)tid];
		Image8U& mask = tlsMask[(size_t)tid];

		const uint32_t idxPatch = (uint32_t)i;
		const TexturePatch& texturePatch = texturePatches[idxPatch];

		// Skip invalid patches defensively
		if (texturePatch.label == NO_ID) {
			continue;
		}
		if (texturePatch.rect.width <= 0 || texturePatch.rect.height <= 0) {
			continue;
		}

		const Image8U3& image0 = PatchSrcImage(idxPatch);
		const cv::Size sz = texturePatch.rect.size();

		if (image.size() != sz) {
			image.create(sz);
		}
		if (imageOrg.size() != sz) {
			imageOrg.create(sz);
		}
		if (mask.size() != sz) {
			mask.create(sz);
		}

		mask.setTo(0);

		image0(texturePatch.rect).convertTo(image, CV_32FC3, 1.0f / 255.0f);
		image.copyTo(imageOrg);

		struct RasterMesh {
			Image8U& image;
			inline void operator()(const ImageRef& pt) const {
				if (image.isInside(pt))
					image(pt) = interior;
			}
		} raster{ mask };

		for (const FIndex idxFace : texturePatch.faces) {
			const TexCoord* tri = faceTexcoords.data() + idxFace * 3;
			ColorMap::RasterizeTriangle(tri[0], tri[1], tri[2], raster);
		}

		const Sampler sampler;
		const TexCoord offset(texturePatch.rect.tl());

		// ------------------------------------------------------------------
		// Main loop over seam vertices
		// ------------------------------------------------------------------
		for (uint32_t vIdx : patchToSeamVertices[idxPatch]) {
			const SeamVertex& seamVertex0 = seamVertices[vIdx];
			if (seamVertex0.patches.size() < 2)
				continue;
			const uint32_t idxVertPatch0 = seamVertex0.patchIndexLookup.At(idxPatch);

			const SeamVertex::Patch& patch0 = seamVertex0.patches[idxVertPatch0];
			const TexCoord p0(patch0.proj - offset);

			// Each edge of this vertex
			for (const SeamVertex::Patch::Edge& edge0 : patch0.edges) {
				const SeamVertex& seamVertex1 = seamVertices[edge0.idxSeamVertex];

				const uint16_t* itAdj =
					seamVertex1.patchIndexLookup.Find(idxPatch);

				if (itAdj == nullptr)
					continue;

				const uint32_t idxVertPatch0Adj = *itAdj;

				const SeamVertex::Patch& patch0Adj = seamVertex1.patches[idxVertPatch0Adj];
				const TexCoord p0Adj(patch0Adj.proj - offset);

				// find the other patch sharing the same edge
				uint32_t numPatchesAtVertex = (uint32_t)seamVertex0.patches.size();

				if (numPatchesAtVertex == 2) {
					// Fast manifold case: only two patches share this vertex
					uint32_t idxVertPatch1 = (idxVertPatch0 == 0) ? 1 : 0;

					const SeamVertex::Patch& patch1 =
						seamVertex0.patches[idxVertPatch1];

					// edge must exist in manifold case
					const uint16_t* itEdge =
						patch1.edgeLookup.Find(edge0.idxSeamVertex);

					if (itEdge != nullptr) {
						const TexCoord& p1 = patch1.proj;

						const uint16_t* itAdj1 =
							seamVertex1.patchIndexLookup.Find(patch1.idxPatch);

						if (itAdj1 != nullptr) {

							const SeamVertex::Patch& patch1Adj =
								seamVertex1.patches[*itAdj1];

							const TexCoord& p1Adj = patch1Adj.proj;

							const Image8U3& image1 =
								PatchSrcImage(patch1.idxPatch);

							RasterPatch data(
								image, mask,
								imageOrg, image1,
								p0, p0Adj,
								p1, p1Adj
							);

							Image32F3::DrawLine(p0, p0Adj, data);
						}
					}
				}
				else {
					// Rare non-manifold case — fallback to original logic
					for (uint32_t idxVertPatch1 = 0;
						idxVertPatch1 < numPatchesAtVertex;
						++idxVertPatch1) {
						if (idxVertPatch1 == idxVertPatch0)
							continue;

						const SeamVertex::Patch& patch1 =
							seamVertex0.patches[idxVertPatch1];

						const uint16_t* itEdge =
							patch1.edgeLookup.Find(edge0.idxSeamVertex);

						if (itEdge == nullptr)
							continue;

						const TexCoord& p1 = patch1.proj;

						const uint16_t* itAdj1 =
							seamVertex1.patchIndexLookup.Find(patch1.idxPatch);

						if (itAdj1 == nullptr)
							continue;

						const SeamVertex::Patch& patch1Adj =
							seamVertex1.patches[*itAdj1];

						const TexCoord& p1Adj = patch1Adj.proj;

						const Image8U3& image1 =
							PatchSrcImage(patch1.idxPatch);

						RasterPatch data(
							image, mask,
							imageOrg, image1,
							p0, p0Adj,
							p1, p1Adj
						);

						Image32F3::DrawLine(p0, p0Adj, data);
						break; // identical behavior
					}
				}
			}

			// render vertex color
			AccumColor accumColor;
			for (const SeamVertex::Patch& patch : seamVertex0.patches) {
				const Image8U3& img = PatchSrcImage(patch.idxPatch);
				accumColor.Add(img.sample<Sampler, Color>(sampler, patch.proj) / 255.f, 1.f);
			}

			const ImageRef pt(ROUND2INT(patch0.proj - offset));

			if (image.isInside(pt)) {
				image(pt) = accumColor.Normalized();
				mask(pt) = border;
			}
		}

		ProcessMask3(mask); // Hardcoded at 3
		PoissonBlendingNoBias(imageOrg, image, mask);

		// apply color correction to patch image
		cv::Mat imagePatch(image0(texturePatch.rect));
		for (int r = 0; r < image.rows; ++r) {
			Pixel8U* row = imagePatch.ptr<Pixel8U>(r);
			// Cache the mask/image row bases once per row: rows are contiguous, so
			// maskRow[c]/imgRow[c] avoid cv::Mat_::operator()'s per-pixel step*r recompute.
			const uint8_t* const maskRow = mask.ptr<uint8_t>(r);
			const Color* const imgRow = image.ptr<Color>(r);
			for (int c = 0; c < image.cols; ++c) {
				if (maskRow[c] == empty)
					continue;
				const Color& a = imgRow[c];
				Pixel8U& v = row[c];
				// scale once
				__m128 af = _mm_set_ps(0.0f, a[2] * 255.f, a[1] * 255.f, a[0] * 255.f);

				// round all 3 channels at once
				__m128i ai = _mm_cvtps_epi32(af);

				// clamp to [0,255]
				const __m128i zero = _mm_setzero_si128();
				const __m128i max255 = _mm_set1_epi32(255);

				ai = _mm_max_epi32(ai, zero);
				ai = _mm_min_epi32(ai, max255);

				// pack to bytes
				__m128i pack16 = _mm_packus_epi32(ai, ai);
				__m128i pack8 = _mm_packus_epi16(pack16, pack16);

				// store RGB
				uint32_t rgb8 = (uint32_t)_mm_cvtsi128_si32(pack8);
				v[0] = (uint8_t)(rgb8 & 0xFF);
				v[1] = (uint8_t)((rgb8 >> 8) & 0xFF);
				v[2] = (uint8_t)((rgb8 >> 16) & 0xFF);
			}
		}
	}

	cv::setNumThreads(prevCvThreads); // Restore OpenCV threading

#ifdef COUNT_ITERATIONS
	double tt = iterations;
	int cc = calls;

	DEBUG("%d tries avg iterations %g", cc, tt / cc);
#endif
}
#else
void MeshTexture::LocalSeamLeveling()
{
	ASSERT(!seamVertices.empty());
	const unsigned numPatches(texturePatches.size()-1);

	// adjust texture patches locally, so that the border continues smoothly inside the patch
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);
		const TexturePatch& texturePatch = texturePatches[idxPatch];
		// extract image
		const Image8U3& image0(images[texturePatch.label].image);
		Image32F3 image, imageOrg;
		image0(texturePatch.rect).convertTo(image, CV_32FC3, 1.0/255.0);
		image.copyTo(imageOrg);
		// render patch coverage
		Image8U mask(image.size()); {
			mask.memset(0);
			struct RasterMesh {
				Image8U& image;
				inline void operator()(const ImageRef& pt) {
					ASSERT(image.isInside(pt));
					image(pt) = interior;
				}
			} data{mask};
			for (const FIndex idxFace: texturePatch.faces) {
				const TexCoord* tri = faceTexcoords.data()+idxFace*3;
				ColorMap::RasterizeTriangle(tri[0], tri[1], tri[2], data);
			}
		}
		// render the patch border meeting neighbor patches
		const Sampler sampler;
		const TexCoord offset(texturePatch.rect.tl());
		for (const SeamVertex& seamVertex0: seamVertices) {
			if (seamVertex0.patches.size() < 2)
				continue;
			const uint32_t idxVertPatch0(seamVertex0.patches.Find(idxPatch));
			if (idxVertPatch0 == SeamVertex::Patches::NO_INDEX)
				continue;
			const SeamVertex::Patch& patch0 = seamVertex0.patches[idxVertPatch0];
			const TexCoord p0(patch0.proj-offset);
			// for each edge of this vertex belonging to this patch...
			for (const SeamVertex::Patch::Edge& edge0: patch0.edges) {
				// select the same edge leaving from the adjacent vertex
				const SeamVertex& seamVertex1 = seamVertices[edge0.idxSeamVertex];
				const uint32_t idxVertPatch0Adj(seamVertex1.patches.Find(idxPatch));
				ASSERT(idxVertPatch0Adj != SeamVertex::Patches::NO_INDEX);
				const SeamVertex::Patch& patch0Adj = seamVertex1.patches[idxVertPatch0Adj];
				const TexCoord p0Adj(patch0Adj.proj-offset);
				// find the other patch sharing the same edge (edge with same adjacent vertex)
				FOREACH(idxVertPatch1, seamVertex0.patches) {
					if (idxVertPatch1 == idxVertPatch0)
						continue;
					const SeamVertex::Patch& patch1 = seamVertex0.patches[idxVertPatch1];
					const uint32_t idxEdge1(patch1.edges.Find(edge0.idxSeamVertex));
					if (idxEdge1 == SeamVertex::Patch::Edges::NO_INDEX)
						continue;
					const TexCoord& p1(patch1.proj);
					// select the same edge belonging to the second patch leaving from the adjacent vertex
					const uint32_t idxVertPatch1Adj(seamVertex1.patches.Find(patch1.idxPatch));
					ASSERT(idxVertPatch1Adj != SeamVertex::Patches::NO_INDEX);
					const SeamVertex::Patch& patch1Adj = seamVertex1.patches[idxVertPatch1Adj];
					const TexCoord& p1Adj(patch1Adj.proj);
					// this is an edge separating two (valid) patches;
					// draw it on this patch as the mean color of the two patches
					const Image8U3& image1(images[texturePatches[patch1.idxPatch].label].image);
					struct RasterPatch {
						Image32F3& image;
						Image8U& mask;
						const Image32F3& image0;
						const Image8U3& image1;
						const TexCoord p0, p0Dir;
						const TexCoord p1, p1Dir;
						const float length;
						const Sampler sampler;
						inline RasterPatch(Image32F3& _image, Image8U& _mask, const Image32F3& _image0, const Image8U3& _image1,
							const TexCoord& _p0, const TexCoord& _p0Adj, const TexCoord& _p1, const TexCoord& _p1Adj)
							: image(_image), mask(_mask), image0(_image0), image1(_image1),
							p0(_p0), p0Dir(_p0Adj-_p0), p1(_p1), p1Dir(_p1Adj-_p1), length((float)norm(p0Dir)), sampler() {}
						inline void operator()(const ImageRef& pt) {
							const float l((float)norm(TexCoord(pt)-p0)/length);
							// compute mean color
							const TexCoord samplePos0(p0 + p0Dir * l);
							const Color color0(image0.sample<Sampler,Color>(sampler, samplePos0));
							const TexCoord samplePos1(p1 + p1Dir * l);
							const Color color1(image1.sample<Sampler,Color>(sampler, samplePos1)/255.f);
							image(pt) = Color((color0 + color1) * 0.5f);
							// set mask edge also
							mask(pt) = border;
						}
					} data(image, mask, imageOrg, image1, p0, p0Adj, p1, p1Adj);
					Image32F3::DrawLine(p0, p0Adj, data);
					// skip remaining patches,
					// as a manifold edge is shared by maximum two face (one in each patch), which we found already
					break;
				}
			}
			// render the vertex at the patch border meeting neighbor patches
			AccumColor accumColor;
			// for each patch...
			for (const SeamVertex::Patch& patch: seamVertex0.patches) {
				// add its view to the vertex mean color
				const Image8U3& img(images[texturePatches[patch.idxPatch].label].image);
				accumColor.Add(img.sample<Sampler,Color>(sampler, patch.proj)/255.f, 1.f);
			}
			const ImageRef pt(ROUND2INT(patch0.proj-offset));
			image(pt) = accumColor.Normalized();
			mask(pt) = border;
		}
		// make sure the border is continuous and
		// keep only the exterior tripe of the given size
		ProcessMask(mask, 20);
		// compute texture patch blending
		PoissonBlending(imageOrg, image, mask);
		// apply color correction to the patch image
		cv::Mat imagePatch(image0(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				if (mask(r,c) == empty)
					continue;
				const Color& a = image(r,c);
				Pixel8U& v = imagePatch.at<Pixel8U>(r,c);
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(a[p]*255.f), 0, 255);
			}
		}
	}
}
#endif

void resizeHugeImage(const cv::Mat& src, cv::Mat& dst, int nMaxTextureSize)
{
	const int srcW = src.cols;
	const int srcH = src.rows;

	// scale factor
	const double scale = static_cast<double>(nMaxTextureSize) /
		static_cast<double>(std::max(srcW, srcH));
	const int dstW = static_cast<int>(std::ceil(srcW * scale));
	const int dstH = static_cast<int>(std::ceil(srcH * scale));

	dst.create(dstH, dstW, src.type());
	dst.setTo(0);

	// process row strips in tiles
	const int tileH = 2048; // you can tune this (balance speed vs memory)
	cv::Mat strip, stripResized;

	for (int y = 0; y < srcH; y += tileH)
	{
		int h = std::min(tileH, srcH - y);

		// take a strip from source
		cv::Rect roi(0, y, srcW, h);
		strip = src(roi);

		// resize this strip
		double scaleY = scale; // same scale in both directions
		cv::resize(strip, stripResized, cv::Size(), scale, scaleY, cv::INTER_AREA);

		// compute destination y offset
		int yDst = static_cast<int>(y * scale);
		int hDst = stripResized.rows;
		if (yDst + hDst > dst.rows)
			hDst = dst.rows - yDst;

		// copy into output
		stripResized(cv::Rect(0, 0, dst.cols, hDst)).copyTo(dst(cv::Rect(0, yDst, dst.cols, hDst)));
	}
}

struct PackedRect {
	cv::Rect rect;
	int index;
};

static inline int RoundUpToMultiple(int x, int m) {
  if (m <= 0) return x;
  return ((x + m - 1) / m) * m;
}

static bool PackShelfFast(
	int atlasW,
	int atlasH,
	const RectsBinPack::RectArr& inRects,
	RectsBinPack::RectArr& outRects)
{
	struct Item {
		int w, h;
		int index;
	};

	const int n = (int)inRects.size();
	std::vector<Item> items;
	items.resize((size_t)n);

	for (int i = 0; i < n; ++i) {
		items[(size_t)i].w = inRects[(size_t)i].width;
		items[(size_t)i].h = inRects[(size_t)i].height;
		items[(size_t)i].index = i;
	}

	std::sort(items.begin(), items.end(),
		[](const Item& a, const Item& b) {
			if (a.h != b.h) return a.h > b.h;
			if (a.w != b.w) return a.w > b.w;
			return a.index < b.index;
		});

	outRects.resize((size_t)n);

	int shelfX = 0;
	int shelfY = 0;
	int shelfH = 0;

	for (int k = 0; k < n; ++k) {
		int w = items[(size_t)k].w;
		int h = items[(size_t)k].h;

		// too large ever
		if ((w > atlasW || h > atlasH) && (h > atlasW || w > atlasH)) {
			return false;
		}

		// If it doesn't fit on this shelf, start a new shelf
		if (shelfX + w > atlasW) {
			shelfY += shelfH;
			shelfX = 0;
			shelfH = 0;
		}

		// If still doesn't fit horizontally, try rotating (fresh shelf or not)
		if (shelfX + w > atlasW) {
			// must be because w > atlasW; try rotate
			int tw = h;
			int th = w;
			w = tw;
			h = th;
			if (w > atlasW) {
				return false;
			}
		}

		// Optional: if it fits rotated but not unrotated (common), rotate
		if (shelfX + w <= atlasW) {
			// ok as-is
		}
		else {
			int tw = h;
			int th = w;
			if (shelfX + tw <= atlasW) {
				w = tw;
				h = th;
			}
			else {
				// new shelf and retry could be done, but keep it simple
				return false;
			}
		}

		if (shelfY + h > atlasH) {
			return false;
		}

		RectsBinPack::Rect placed;
		placed.x = shelfX;
		placed.y = shelfY;
		placed.width = w;
		placed.height = h;
		outRects[(size_t)items[(size_t)k].index] = placed;

		shelfX += w;
		if (h > shelfH) shelfH = h;
	}

	return true;
}

static void ComputeUsedBounds(
  const RectsBinPack::RectArr& placed,
  int& usedW,
  int& usedH)
{
  usedW = 0;
  usedH = 0;
  for (const auto& r : placed) {
    usedW = std::max(usedW, r.x + r.width);
    usedH = std::max(usedH, r.y + r.height);
  }
}

static bool PackShelfReasonablySquare(
  const RectsBinPack::RectArr& rects,
  int multiple,
  int maxDim,
  RectsBinPack::RectArr& outPlaced,
  int& outAtlasW,
  int& outAtlasH)
{
  uint64_t area = 0;
  int maxW = 0;
  int maxH = 0;

  for (const auto& r : rects) {
    area += (uint64_t)r.width * (uint64_t)r.height;
    maxW = std::max(maxW, r.width);
    maxH = std::max(maxH, r.height);
  }

  if (rects.empty()) {
    outPlaced.clear();
    outAtlasW = 0;
    outAtlasH = 0;
    return true;
  }

  // Target square width near sqrt(area)
  int baseW = (int)std::sqrt((double)area);
  baseW = std::max(baseW, maxW);
  baseW = RoundUpToMultiple(baseW, multiple);
  baseW = std::min(baseW, maxDim);

  // Try widths around baseW (geometric sweep)
  // This keeps it fast for 80k (few dozen pack attempts).
  const int kTries = 18;
  int bestScore = INT_MAX;
  int bestW = 0;
  int bestH = 0;
  RectsBinPack::RectArr bestPlaced;

  RectsBinPack::RectArr placed;

  for (int t = -kTries / 2; t <= kTries / 2; ++t) {
    double scale = std::pow(2.0, (double)t / 3.0); // steps of ~1.26x
    int candW = (int)(baseW * scale);

    candW = std::max(candW, maxW);
    candW = RoundUpToMultiple(candW, multiple);
    candW = std::min(candW, maxDim);

    // Skip duplicate widths
    if (candW == bestW) {
      continue;
    }

    // Allow full height up to maxDim; packing determines usedH
    if (!PackShelfFast(candW, maxDim, rects, placed)) {
      continue;
    }

    int usedW = 0;
    int usedH = 0;
    ComputeUsedBounds(placed, usedW, usedH);

    if (usedH > maxDim || usedW > maxDim) {
      continue;
    }

    // Score: prefer square and small max dimension.
    // aspectPenalty is 0 when square, grows as it stretches.
    int maxSide = std::max(usedW, usedH);
    int minSide = std::max(1, std::min(usedW, usedH));
    int aspectPenalty = (maxSide * 1000) / minSide; // 1000 == perfect square

    // Weight aspect heavily, but also prefer smaller atlases.
    int score = aspectPenalty * 100 + maxSide;

    if (score < bestScore) {
      bestScore = score;
      bestW = usedW;
      bestH = usedH;
      bestPlaced = placed;
    }
  }

  if (bestScore == INT_MAX) {
    return false; // cannot fit within maxDim; you need multi-atlas paging
  }

  outPlaced.swap(bestPlaced);
  outAtlasW = bestW;
  outAtlasH = bestH;
  return true;
}

// TEXTURE_OUTWARD_DILATE_PX: after the atlas is assembled, bleed real texel
// colors outward into the empty (colEmpty) background by this many pixels via a
// frontier flood (each ring = mean of known neighbors). Fills atlas gutters /
// seams and grows the textured margin outward so the edge fades into plausible
// color instead of a hard empty band -- the "dilate the texture to fill the
// outside" effect, benefiting both the 3D mesh and the ortho. 0 = disabled.
#ifndef TEXTURE_OUTWARD_DILATE_PX
#define TEXTURE_OUTWARD_DILATE_PX 24
#endif

// TEXTURE_ATLAS_FULL_FLOOD: when 1, the outward dilation does not stop after
// TEXTURE_OUTWARD_DILATE_PX rings -- it keeps flooding until EVERY colEmpty texel in
// the atlas has been filled with its nearest known colour (the background gap between
// patches is fully padded). This is the standard "seamless atlas gutter" and it is the
// real fix for the thin DARK RIM that traces a patch/fill silhouette: with a bounded
// 24px margin the observed patch edge still borders dark colEmpty a little further out,
// and GPU MIPMAP MINIFICATION (when the surface is viewed at distance) averages across
// that margin into the dark background, painting a dark line along the edge. Leaving no
// colEmpty anywhere near a used texel means no mip level can ever pull in the dark
// background, so the rim disappears at every viewing distance. The flood is a single
// frontier pass over the atlas (each texel filled once) so it is O(atlas area) -- the
// extra rings past 24 only touch the still-empty background, which is never sampled at
// base resolution, so this changes nothing at full res and only sanitises the mip chain.
// 0 = keep the bounded TEXTURE_OUTWARD_DILATE_PX margin (previous behaviour).
#ifndef TEXTURE_ATLAS_FULL_FLOOD
#define TEXTURE_ATLAS_FULL_FLOOD 1
#endif

// When --max-texture-size is NEGATIVE, the final atlas is capped to the GPU's
// GL_MAX_TEXTURE_SIZE instead of a fixed value: the viewer is always OpenGL, so
// that is the largest atlas it can upload without doing its own crude rescale.
// (0 keeps its original meaning: UNBOUNDED / no cap.) Windows-only: spin up a
// throwaway hidden-window WGL context once, query GL_MAX_TEXTURE_SIZE, tear it
// down, and cache the result. On any failure (headless / no driver) fall back to
// 16384, the minimum every modern GL implementation guarantees.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#endif

static int GetOpenGLMaxTextureSize()
{
	static int cached = 0;
	if (cached != 0)
		return cached;
	int result = 16384; // GL guaranteed minimum; safe fallback
#ifdef _WIN32
	WNDCLASSA wc = {};
	wc.lpfnWndProc = DefWindowProcA;
	wc.hInstance = GetModuleHandleA(nullptr);
	wc.lpszClassName = "OpenMVS_GLProbe";
	RegisterClassA(&wc);
	if (HWND hWnd = CreateWindowA(wc.lpszClassName, "", WS_OVERLAPPEDWINDOW, 0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr)) {
		if (HDC hDC = GetDC(hWnd)) {
			PIXELFORMATDESCRIPTOR pfd = {};
			pfd.nSize = sizeof(pfd);
			pfd.nVersion = 1;
			pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
			pfd.iPixelType = PFD_TYPE_RGBA;
			pfd.cColorBits = 32;
			pfd.cDepthBits = 24;
			pfd.iLayerType = PFD_MAIN_PLANE;
			const int pf = ChoosePixelFormat(hDC, &pfd);
			if (pf && SetPixelFormat(hDC, pf, &pfd)) {
				if (HGLRC hRC = wglCreateContext(hDC)) {
					if (wglMakeCurrent(hDC, hRC)) {
						GLint maxTex = 0;
						glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
						if (maxTex >= 1024)
							result = (int)maxTex;
						wglMakeCurrent(nullptr, nullptr);
					}
					wglDeleteContext(hRC);
				}
			}
			ReleaseDC(hWnd, hDC);
		}
		DestroyWindow(hWnd);
	}
	UnregisterClassA(wc.lpszClassName, wc.hInstance);
#endif
	cached = result;
	return cached;
}

void MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize)
{
	// --max-texture-size < 0 => cap the final atlas to the GPU's native
	// GL_MAX_TEXTURE_SIZE (the viewer is OpenGL). == 0 keeps its original meaning:
	// UNBOUNDED (no cap). Resolved once here so BOTH the packer's starting
	// dimension and the final "enforce max size" step use it.
	if (nMaxTextureSize < 0) {
		nMaxTextureSize = GetOpenGLMaxTextureSize();
		DEBUG_EXTRA("Texture max size < 0 -> GL_MAX_TEXTURE_SIZE = %d", nMaxTextureSize);
	}
	// project patches in the corresponding view and compute texture-coordinates and bounding-box
	TEX_PROFILE_BEGIN(_tGtProject);
	const int border(2);
	faceTexcoords.resize(faces.size() * 3);
#ifdef TEXOPT_USE_OPENMP
	const unsigned numPatches(texturePatches.size()); // no dummy anymore
#pragma omp parallel for schedule(static, 1)
	for (int_t idx = 0; idx < (int_t)numPatches; ++idx) {
		TexturePatch& texturePatch = texturePatches[(uint32_t)idx];
#else
	for (TexturePatch* pTexturePatch = texturePatches.Begin(), *pTexturePatchEnd = texturePatches.End() - 1; pTexturePatch < pTexturePatchEnd; ++pTexturePatch) {
		TexturePatch& texturePatch = *pTexturePatch;
#endif
		const Image& imageData = images[texturePatch.label];
#if TEXTURE_CROP_IMAGES
		// Pixels were released after ListCameraFaces; use the stored dimensions (camera
		// is intact). They equal the res-0 loaded size, so these projections + rects match
		// the image that will be reloaded and cropped immediately after this loop.
		const int imgCols = (int)imageData.width;
		const int imgRows = (int)imageData.height;
#else
		const int imgCols = imageData.image.cols;
		const int imgRows = imageData.image.rows;
#endif
		AABB2f aabb(true);
		for (const FIndex idxFace : texturePatch.faces) {
			const Face& face = faces[idxFace];
			TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
			for (int i = 0; i < 3; ++i) {
				texcoords[i] = imageData.camera.ProjectPointP(vertices[face[i]]);
				ASSERT(texcoords[i].x >= (float)border && texcoords[i].x < (float)(imgCols - border) &&
					texcoords[i].y >= (float)border && texcoords[i].y < (float)(imgRows - border));
				aabb.InsertFull(texcoords[i]);
			}
		}
		// compute relative texture coordinates
		int x0 = FLOOR2INT(aabb.ptMin[0]) - border;
		int y0 = FLOOR2INT(aabb.ptMin[1]) - border;
		int x1 = CEIL2INT(aabb.ptMax[0]) + border;
		int y1 = CEIL2INT(aabb.ptMax[1]) + border;

		// Clamp BEFORE building rect
		x0 = std::max(0, x0);
		y0 = std::max(0, y0);
		x1 = std::min(imgCols, x1);
		y1 = std::min(imgRows, y1);

		texturePatch.rect = cv::Rect(
			x0,
			y0,
			std::max(0, x1 - x0),
			std::max(0, y1 - y0)
		);

		// Safety: skip empty patches (rare but possible)
		if (texturePatch.rect.width <= 0 ||
			texturePatch.rect.height <= 0)
		{
			continue;
		}

		const TexCoord offset(texturePatch.rect.tl());

		for (const FIndex idxFace : texturePatch.faces) {
			TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
			for (int v = 0; v < 3; ++v)
				texcoords[v] -= offset;
		}
	}
	TEX_PROFILE_END(_tGtProject, "GenerateTexture: project patches");

	LogPeakMem("GenTex: after projection");

#if TEXTURE_CROP_IMAGES
	// PER-PATCH EXTRACTION: give every texture patch its OWN pixel buffer, cropped from
	// the source image to just THAT patch's rect (full resolution), then rebase the rect
	// to its own origin (0,0). A union-per-label crop failed because on aerial data a
	// single image's assigned faces scatter across MANY components, so the union bbox is
	// ~the whole image (no saving). Per-patch keeps only the sum of the actual patch rects
	// (~= the atlas area), dropping resident image memory ~20 GB -> ~1-2 GB at FULL res.
	// Coordinate-transparent: faceTexcoords stay 0-based to the old rect and every
	// downstream stage reads PatchSrcImage(idx) at faceTexcoords + rect.tl(); with rect
	// rebased to (0,0) that is exactly the patch-local coordinate. The overlapping-patch
	// MERGE is DISABLED below when this is on (it would reassign a contained patch's faces
	// to another patch's buffer, which no longer shares source pixels).
	{
		TEX_PROFILE_SCOPE("GenerateTexture: extract per-patch images");
		const uint32_t nImages = (uint32_t)images.size();
		std::vector<std::vector<uint32_t>> patchesByLabel(nImages);
		for (uint32_t p = 0; p < texturePatches.GetSize(); ++p) {
			const TexturePatch& tp = texturePatches[p];
			if (tp.label == NO_ID || tp.rect.width <= 0 || tp.rect.height <= 0)
				continue;
			patchesByLabel[tp.label].push_back(p);
		}
		std::vector<uint32_t> usedLabels;
		usedLabels.reserve(nImages);
		for (uint32_t lbl = 0; lbl < nImages; ++lbl)
			if (!patchesByLabel[lbl].empty())
				usedLabels.push_back(lbl);

		// Decode into a BOUNDED set of REUSED per-thread buffers instead of a fresh
		// imageData.image per image. Image::ReadImage() calls image.create(H,W), which
		// REUSES an existing same-size buffer, so at res-0 (all source images identical
		// size) each worker decodes every one of its images into the SAME scratch Mat.
		// The OpenCV allocator therefore never grows past ~nThreads full images (~1.8 GB)
		// instead of retaining all 367 freed decodes (~20 GB). That retention is why the
		// peak was ~46 GB: release() returns each decode to OpenCV's OWN CRT heap free
		// list, and the app-side _heapmin() cannot decommit a different module's heap, so
		// the freed 20 GB stayed committed under every later stage. The tiny crops
		// (~0.6 GB total) are cloned out and kept; the scratch is freed right after.
		const int T = std::max(1, omp_get_max_threads());
		std::vector<Image8U3> reloadScratch((size_t)T);
#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(T)
#endif
		for (int li = 0; li < (int)usedLabels.size(); ++li) {
			const uint32_t label = usedLabels[(size_t)li];
			Image& imageData = images[label];
			Image8U3& scratch = reloadScratch[(size_t)omp_get_thread_num()];

			// decode the full source image into the reused scratch (create() reuses the
			// buffer when the size matches -> no per-image allocation churn)
			unsigned level(nResolutionLevel);
			const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
			if (Image::ReadImage(imageData.name, scratch) == NULL)
				continue; // decode failed; patches keep empty images -> skipped downstream
			// match the resolution level ListCameraFaces used (res-0 => imageSize == full
			// original, so no resize and the scratch buffer is reused verbatim)
			if ((unsigned)MAXF(scratch.cols, scratch.rows) > imageSize) {
				const double s = (double)imageSize / (double)MAXF(scratch.cols, scratch.rows);
				cv::resize(scratch, scratch, cv::Size(), s, s, cv::INTER_AREA);
			}
			imageData.UpdateCamera(scene.platforms);
			if (scratch.empty())
				continue;

			const int srcW = scratch.cols, srcH = scratch.rows;
			for (const uint32_t p : patchesByLabel[label]) {
				TexturePatch& tp = texturePatches[p];
				cv::Rect r = tp.rect;
				// clamp to the reloaded image (projection already clamped; defensive)
				if (r.x < 0) { r.width += r.x; r.x = 0; }
				if (r.y < 0) { r.height += r.y; r.y = 0; }
				if (r.x + r.width > srcW) r.width = srcW - r.x;
				if (r.y + r.height > srcH) r.height = srcH - r.y;
				if (r.width <= 0 || r.height <= 0) { tp.rect = cv::Rect(0, 0, 0, 0); continue; }
				tp.image = scratch(r).clone();               // private full-res crop
				tp.rect = cv::Rect(0, 0, r.width, r.height); // rebase to its own origin
			}
		}
		// free the (few) reused decode buffers now that all crops are cloned out
		reloadScratch.clear();
		reloadScratch.shrink_to_fit();
		// [CROP-DECOMP] TEMP decisive measurement: is the +20 GB after this stage the
		// CROPS themselves, or retained/committed image-decode memory that our buffer
		// management can't reach? Sum the actual bytes held in every patch crop. If this
		// prints ~0.6 GB while curWS jumped ~20 GB, the crops are NOT the hog (it is
		// OpenCV/jpeg decode retention). If it prints ~20 GB, the patch rects really are
		// huge and the packing-area estimate was wrong. Remove once answered.
		{
			size_t cropBytes = 0, cropPx = 0; unsigned nCrop = 0, nBig = 0; size_t maxPx = 0;
			for (uint32_t p = 0; p < texturePatches.GetSize(); ++p) {
				const Image8U3& im = texturePatches[p].image;
				if (im.empty()) continue;
				const size_t px = (size_t)im.total();
				cropBytes += px * im.elemSize(); cropPx += px; ++nCrop;
				if (px > maxPx) maxPx = px;
				if (px > 4000000) ++nBig; // patches bigger than ~2000x2000
			}
			DEBUG_EXTRA("[CROP-DECOMP] crops=%.2f GB (%u patches, %.1f Mpx total, largest %.1f Mpx, %u patches >4Mpx)",
				cropBytes / (1024.0*1024.0*1024.0), nCrop, cropPx / 1e6, maxPx / 1e6, nBig);
		}
		LogPeakMem("GenTex: after per-patch extract");
	}
#endif // TEXTURE_CROP_IMAGES

	// perform seam leveling
	// (is introducing a green tint).
	if (texturePatches.GetSize() > 2 && (bGlobalSeamLeveling || bLocalSeamLeveling)) {
		// create seam vertices and edges
		TEX_PROFILE_BEGIN(_tGtSeamVerts);
		CreateSeamVertices();
		TEX_PROFILE_END(_tGtSeamVerts, "GenerateTexture: CreateSeamVertices");
		LogPeakMem("GenTex: after CreateSeamVertices");

		// perform global seam leveling
		if (bGlobalSeamLeveling && TEXTURE_ENABLE_GLOBAL_SEAM) {
			TD_TIMER_STARTD();
			GlobalSeamLeveling();
			DEBUG_ULTIMATE("\tglobal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}
		LogPeakMem("GenTex: after GlobalSeamLeveling");

		// perform local seam leveling
		if (bLocalSeamLeveling && TEXTURE_ENABLE_LOCAL_SEAM) {
			TD_TIMER_STARTD();
			LocalSeamLeveling();
			DEBUG_ULTIMATE("\tlocal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}
		LogPeakMem("GenTex: after LocalSeamLeveling");

		// Release the per-thread seam-leveling scratch now. It is thread_local
		// (lives for the whole run in the OMP worker pool) and sized to the
		// largest patch, so otherwise it stays resident straight through the
		// data-color bake + save. Each worker frees its OWN copies via this
		// parallel region (semantics-neutral: pure scratch, re-grown on next use).
#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel
		{ ReleaseSeamScratch(); }
#else
		ReleaseSeamScratch();
#endif
	}
	LogPeakMem("GenTex: after seam leveling (pre-merge)");

	// merge texture patches with overlapping rectangles
	TEX_PROFILE_BEGIN(_tGtMergePatches);
#if !TEXTURE_CROP_IMAGES
	// (disabled under per-patch extraction: each patch owns a PRIVATE rebased crop, so a
	// contained patch cannot be re-pointed into another patch's buffer)
	for (unsigned i = 0; i < texturePatches.size(); ++i) {
		TexturePatch& texturePatchBig = texturePatches[i];

		for (unsigned j = i + 1; j < texturePatches.size(); /* no ++j here */) {
			TexturePatch& texturePatchSmall = texturePatches[j];

			if (texturePatchBig.label != texturePatchSmall.label ||
				!RectsBinPack::IsContainedIn(texturePatchSmall.rect, texturePatchBig.rect)) {
				++j;
				continue;
			}

			const TexCoord offset(texturePatchSmall.rect.tl() - texturePatchBig.rect.tl());

			for (const FIndex idxFace : texturePatchSmall.faces) {
				TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
				for (int v = 0; v < 3; ++v) {
					texcoords[v] += offset;
				}
			}

			texturePatchBig.faces.JoinRemove(texturePatchSmall.faces);
			texturePatches.RemoveAtMove(j); // j stays the same (new element moved into j)
		}
	}
#endif // !TEXTURE_CROP_IMAGES
	TEX_PROFILE_END(_tGtMergePatches, "GenerateTexture: merge overlapping patches");

	// create texture
	{
		// Prepare rects
		RectsBinPack::RectArr rects(texturePatches.GetSize());
		for (size_t i = 0; i < (size_t)texturePatches.GetSize(); ++i) {
			rects[i] = texturePatches[(uint32_t)i].rect;
		}

		RectsBinPack::RectArr placed;
		int atlasW = 0;
		int atlasH = 0;

		int maxDim = std::max(64, nMaxTextureSize); // starting point

		// Optional: hard safety cap so you don't allocate something insane
		const int hardCap = 262144; // pick something you can tolerate

		TEX_PROFILE_BEGIN(_tGtPack);
		bool ok = false;

		while (!ok) {
			ok = PackShelfReasonablySquare(
				rects,
				(int)nTextureSizeMultiple,
				maxDim,
				placed,
				atlasW,
				atlasH);

			if (ok) {
				break;
			}

			// Grow and retry
			if (maxDim >= hardCap) {
				// At this point you either need paging or you accept very large atlases.
				// Since you said "don't fail", last resort: just keep going (or set higher cap).
				maxDim = maxDim * 2;
			}
			else {
				maxDim = std::min(hardCap, maxDim * 2);
			}
		}

		rects.swap(placed);
		TEX_PROFILE_END(_tGtPack, "GenerateTexture: rect packing");

		LogPeakMem("GenTex: after seam+pack");
		TEX_PROFILE_BEGIN(_tGtAssemble);
		textureDiffuse.create(atlasH, atlasW);
		{
			const cv::Scalar emptyScalar(colEmpty.b, colEmpty.g, colEmpty.r);
#ifdef TEXOPT_USE_OPENMP
			// Parallelize the whole-atlas background fill across row bands
			// (cv::Mat::setTo is single-threaded); identical result.
#pragma omp parallel
			{
				const int nth = omp_get_num_threads();
				const int tid = omp_get_thread_num();
				const int band = (atlasH + nth - 1) / nth;
				const int r0 = tid * band;
				const int r1 = std::min(atlasH, r0 + band);
				if (r1 > r0)
					textureDiffuse.rowRange(r0, r1).setTo(emptyScalar);
			}
#else
			textureDiffuse.setTo(emptyScalar);
#endif
		}

		// Assemble the atlas GROUPED BY SOURCE IMAGE (label), releasing each image's
		// pixel buffer the moment its last patch has been copied. On a high-res scene
		// the source images are ~15 GB; during the (thread-capped, minutes-long)
		// LocalSeamLeveling the OS pages most of them out, then a naive parallel-over-
		// patches assemble faults ALL of them back in at once -> a ~15 GB spike that
		// sets the whole GenerateTexture peak (observed 45.7 GB on a 367-image res-0
		// scene, PageFaultCount ~114M = thrashing). Processing one label at a time and
		// releasing its image keeps only ~nThreads images resident during assemble
		// instead of all of them. Each label is owned by exactly one thread, so the
		// per-image release is race-free. Output is bit-identical (same copy + same
		// per-face UV offsets); only the traversal order and image lifetime change.
		std::vector<std::vector<int_t>> patchesByLabel(images.size());
		for (int_t i = 0; i < (int_t)texturePatches.size(); ++i)
			patchesByLabel[texturePatches[i].label].push_back(i);
		std::vector<uint32_t> usedLabels;
		usedLabels.reserve(images.size());
		for (uint32_t lbl = 0; lbl < (uint32_t)images.size(); ++lbl)
			if (!patchesByLabel[lbl].empty())
				usedLabels.push_back(lbl);

#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
		for (int_t li = 0; li < (int_t)usedLabels.size(); ++li)
		{
			const uint32_t label = usedLabels[(size_t)li];

			for (const int_t i : patchesByLabel[label])
			{
				TexturePatch& texturePatch = texturePatches[i];
				const RectsBinPack::Rect& rect = rects[i];

				int x = 0;
				int y = 1;

				// Source pixels: per-patch crop (TEXTURE_CROP_IMAGES) or shared source image.
				const Image8U3& srcImg = PatchSrcImage((uint32_t)i);
				const int imgW = srcImg.cols;
				const int imgH = srcImg.rows;

				// Clamp ROI to image bounds BEFORE making cv::Mat(roi)
				cv::Rect roi = texturePatch.rect;

				// Clamp left/top by shrinking width/height
				if (roi.x < 0) {
					roi.width += roi.x;
					roi.x = 0;
				}
				if (roi.y < 0) {
					roi.height += roi.y;
					roi.y = 0;
				}

				// Clamp right/bottom
				if (roi.x + roi.width > imgW) {
					roi.width = imgW - roi.x;
				}
				if (roi.y + roi.height > imgH) {
					roi.height = imgH - roi.y;
				}

				// Skip if invalid
				if (roi.width <= 0 || roi.height <= 0) {
#if TEXTURE_CROP_IMAGES
					texturePatch.image.release();
#endif
					continue;
				}

				const cv::Mat patch(srcImg(roi));

				if (rect.width != roi.width) {
					x = 1;
					y = 0;
					// Transpose directly into the atlas ROI: one pass, no temp Mat
					// (avoids patch.t()'s allocation + a second copyTo).
					cv::transpose(patch, textureDiffuse(rect));
				}
				else {
					patch.copyTo(textureDiffuse(rect));
				}

				// Update per-face UV offsets
				const TexCoord offset(rect.tl());
				for (const FIndex idxFace : texturePatch.faces) {
					TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
					for (int v = 0; v < 3; ++v) {
						TexCoord& texcoord = texcoords[v];

						float u = texcoord[x] + offset.x;
						float v2 = texcoord[y] + offset.y;

						texcoord = TexCoord(u, v2);
					}
				}

#if TEXTURE_CROP_IMAGES
				// per-patch buffer consumed -> free it immediately
				texturePatch.image.release();
#endif
			}

#if !TEXTURE_CROP_IMAGES
			// This image's pixels are fully consumed -> free them now so the atlas
			// assemble never holds more than ~nThreads source images resident.
			images[label].image.release();
#endif
		}
		TEX_PROFILE_END(_tGtAssemble, "GenerateTexture: assemble atlas");

		// Release any remaining source-image pixel buffers (labels with no patches,
		// or invalid/unreferenced images). The referenced ones were freed inline in
		// the label loop above. Every remaining step in GenerateTexture (outward
		// dilate, data-color bake, resize/sharpen) operates SOLELY on textureDiffuse
		// and never samples images[].image again. Metadata/cameras stay intact;
		// ReloadImage would restore pixels on demand if a later stage needed them.
		{
			TEX_PROFILE_SCOPE("GenerateTexture: release source images");
			for (Image& imageData : images)
				imageData.image.release();
			TrimHeap(); // decommit the freed crop/image pages before the atlas post-processing
			LogPeakMem("GenTex: after releasing source images");
		}

#if 0
		for (int y = 1; y < textureDiffuse.rows - 1; ++y) {
			for (int x = 1; x < textureDiffuse.cols - 1; ++x) {

				cv::Vec3b& p = textureDiffuse.at<cv::Vec3b>(y, x);

				if (p[0] == 255 && p[1] == 0 && p[2] == 255) {

					cv::Vec3i sum(0, 0, 0);
					int n = 0;

					for (int dy = -1; dy <= 1; ++dy)
						for (int dx = -1; dx <= 1; ++dx) {
							cv::Vec3b nb = textureDiffuse.at<cv::Vec3b>(y + dy, x + dx);
							if (!(nb[0] == 255 && nb[1] == 0 && nb[2] == 255)) {
								sum += cv::Vec3i(nb);
								++n;
							}
						}

					if (n > 0) {
						p[0] = sum[0] / n;
						p[1] = sum[1] / n;
						p[2] = sum[2] / n;
					}
				}
			}
		}
#endif

#if TEXTURE_OUTWARD_DILATE_PX > 0
		// ------------------------------------------------------------
		// Texture dilation: bleed real texel colors outward into the empty
		// (colEmpty) background by TEXTURE_OUTWARD_DILATE_PX pixels via a
		// frontier flood (each ring = mean of known neighbors). Fills atlas
		// gutters/seams and grows the textured margin outward so the edge fades
		// into plausible color instead of a hard empty band. Full-res (pre-resize).
		// ------------------------------------------------------------
		{
			TEX_PROFILE_SCOPE("GenerateTexture: outward dilate");
			const int H = textureDiffuse.rows, W = textureDiffuse.cols;
			const uint8_t eb = (uint8_t)colEmpty.b, eg = (uint8_t)colEmpty.g, er = (uint8_t)colEmpty.r;
			cv::Mat_<uint8_t> known(H, W);
			// Cache raw base pointers + row strides once: the flood loops do random
			// (yy,xx) neighbor access millions of times; direct base[y*stride+x]
			// indexing avoids cv::Mat_::operator()/at<>()'s per-call step-member
			// read and type dispatch.
			uint8_t* const kbase = known.ptr<uint8_t>(0);
			const size_t kstride = known.step.p[0];
			cv::Vec3b* const tbase = textureDiffuse.ptr<cv::Vec3b>(0);
			const size_t tstride = textureDiffuse.step.p[0] / sizeof(cv::Vec3b);
#pragma omp parallel for schedule(static)
			for (int y = 0; y < H; ++y) {
				const cv::Vec3b* row = tbase + (size_t)y * tstride;
				uint8_t* k = kbase + (size_t)y * kstride;
				for (int x = 0; x < W; ++x)
					k[x] = (row[x][0] == eb && row[x][1] == eg && row[x][2] == er) ? 0 : 1;
			}
			std::vector<cv::Point> frontier, next;
			// Initial frontier build (parallel; Jacobi wave -> frontier order does not
			// affect the result, so per-thread collection + concat is neutral).
			{
				const int nThreads = omp_get_max_threads();
				std::vector<std::vector<cv::Point>> tls((size_t)nThreads);
#pragma omp parallel
				{
					std::vector<cv::Point>& loc = tls[(size_t)omp_get_thread_num()];
#pragma omp for schedule(static)
					for (int y = 0; y < H; ++y) {
						const uint8_t* krow = kbase + (size_t)y * kstride;
						for (int x = 0; x < W; ++x) {
							if (krow[x]) continue;
							bool adj = false;
							for (int dy = -1; dy <= 1 && !adj; ++dy) {
								const int yy = y + dy;
								if ((unsigned)yy >= (unsigned)H) continue;
								const uint8_t* knrow = kbase + (size_t)yy * kstride;
								for (int dx = -1; dx <= 1; ++dx) {
									const int xx = x + dx;
									if ((unsigned)xx < (unsigned)W && knrow[xx] == 1) { adj = true; break; }
								}
							}
							if (adj) loc.emplace_back(x, y);
						}
					}
				}
				for (std::vector<cv::Point>& v : tls) frontier.insert(frontier.end(), v.begin(), v.end());
			}
			// Ring cap: bounded margin, OR a full flood of every colEmpty texel (mip-safe,
			// kills the dark rim). (H + W) is >= the max Manhattan distance from any empty
			// texel to a known one, so it always completes the flood; the loop still exits
			// early via `!frontier.empty()` once nothing is left to fill.
			const int dilateRings = TEXTURE_ATLAS_FULL_FLOOD ? (H + W) : TEXTURE_OUTWARD_DILATE_PX;
			for (int it = 0; it < dilateRings && !frontier.empty(); ++it) {
				const int nFront = (int)frontier.size();
				std::vector<cv::Vec3b> fillCol(nFront);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < nFront; ++i) {
					const int x = frontier[i].x, y = frontier[i].y;
					cv::Vec3i sum(0, 0, 0); int n = 0;
					for (int dy = -1; dy <= 1; ++dy) {
						const int yy = y + dy;
						if ((unsigned)yy >= (unsigned)H) continue;
						const uint8_t* knrow = kbase + (size_t)yy * kstride;
						const cv::Vec3b* tnrow = tbase + (size_t)yy * tstride;
						for (int dx = -1; dx <= 1; ++dx) {
							const int xx = x + dx;
							if ((unsigned)xx < (unsigned)W && knrow[xx] == 1) {
								sum += cv::Vec3i(tnrow[xx]); ++n;
							}
						}
					}
					fillCol[i] = n ? cv::Vec3b((uchar)(sum[0] / n), (uchar)(sum[1] / n), (uchar)(sum[2] / n)) : cv::Vec3b(eb, eg, er);
				}
#pragma omp parallel for schedule(static)
				for (int i = 0; i < nFront; ++i) {
					const int x = frontier[i].x, y = frontier[i].y;
					uint8_t& kxy = kbase[(size_t)y * kstride + x];
					if (kxy == 1) continue;
					tbase[(size_t)y * tstride + x] = fillCol[i];
					kxy = 1;
				}
				next.clear();
				next.reserve((size_t)nFront * 2);
				for (int i = 0; i < nFront; ++i) {
					const int x = frontier[i].x, y = frontier[i].y;
					for (int dy = -1; dy <= 1; ++dy) {
						const int yy = y + dy;
						if ((unsigned)yy >= (unsigned)H) continue;
						uint8_t* knrow = kbase + (size_t)yy * kstride;
						for (int dx = -1; dx <= 1; ++dx) {
							const int xx = x + dx;
							if ((unsigned)xx < (unsigned)W && knrow[xx] == 0) {
								knrow[xx] = 2; // tentatively queued (dedup within wave)
								next.emplace_back(xx, yy);
							}
						}
					}
				}
				for (const cv::Point& p : next) {
					uint8_t& kp = kbase[(size_t)p.y * kstride + p.x];
					if (kp == 2) kp = 0;
				}
				frontier.swap(next);
			}
		}
#endif // TEXTURE_OUTWARD_DILATE_PX

#if TEXTURE_DATACOLOR_UNOBSERVED && TEXTURE_DATACOLOR_BAKE
		// ------------------------------------------------------------
		// Data-color faces with NO genuine camera view (smear-propagation was
		// skipped for them). Take the color from the OBSERVED faces adjacent to
		// them on the same surface (their real texture color) and propagate it
		// inward along the mesh, baking a per-vertex gradient texel into the atlas.
		// This gives a locally-correct color instead of smeared neighbor texture.
		// ------------------------------------------------------------
		{
			{
				TEX_PROFILE_SCOPE("GenerateTexture: data-color bake");
				std::vector<uint8_t> covered(faces.size(), 0);
				for (const TexturePatch& tp : texturePatches)
					for (const FIndex fc : tp.faces)
						covered[fc] = 1;
				std::vector<FIndex> noViewFaces;
				for (FIndex f = 0; f < (FIndex)faces.size(); ++f)
					if (!covered[f])
						noViewFaces.push_back(f);
				if (!noViewFaces.empty()) {
					// PER-VERTEX color so shared vertices get ONE color -> the per-face
					// gradient baked below is C0-continuous across shared edges (no facet
					// seams). A flat per-face color is ALWAYS faceted (constant per triangle,
					// jumps at every edge) no matter how much the field is smoothed.
					const int cols = textureDiffuse.cols;
					const int oldRows = textureDiffuse.rows;
					const int N = (int)noViewFaces.size();
					const size_t NV = vertices.size();
					std::vector<uint8_t> vUsed(NV, 0);
					for (int i = 0; i < N; ++i) {
						const Face& face = faces[noViewFaces[i]];
						vUsed[face[0]] = vUsed[face[1]] = vUsed[face[2]] = 1;
					}
					// compact list of the unobserved-region vertices: the sampling and
					// diffusion loops below touch only these, not the whole mesh (the old
					// code rescanned all NV vertices 2x per smoothing iteration).
					std::vector<uint32_t> usedList;
					usedList.reserve((size_t)N * 2);
					for (size_t v = 0; v < NV; ++v) if (vUsed[v]) usedList.push_back((uint32_t)v);
					const int NU = (int)usedList.size();
					const uint32_t* __restrict pUsed = usedList.data();
					std::vector<float> vr(NV, 0.f), vg(NV, 0.f), vb(NV, 0.f);
					std::vector<uint8_t> vValid(NV, 0);
					// vSeed marks the boundary vertices seeded directly from real observed
					// texture; these are PINNED during smoothing so the fill boundary never
					// drifts away from the texture it meets (kills the hard fill/real seam).
					std::vector<uint8_t> vSeed(NV, 0);
					// Seed the unobserved-region BOUNDARY vertices with the REAL color of
					// the OBSERVED faces adjacent to them (sampled from the atlas at the
					// observed face's texels). faceTexcoords for covered faces are absolute
					// atlas pixel coords at this point. The diffusion below then spreads this
					// colour inward ALONG THE MESH SURFACE, so each unobserved face takes the
					// colour of the nearest observed surface it is connected to -- not a 3D
					// point-cloud average that blends unrelated surfaces.
					{
						const int texW = textureDiffuse.cols, texH = textureDiffuse.rows;
						const uchar eB = colEmpty.b, eG = colEmpty.g, eR = colEmpty.r;
						// Cache base pointer + row stride once: the seed sampler does scattered
						// (sy,sx) reads; base[sy*stride+sx] avoids cv::Mat::at<>()'s per-call
						// step read + type dispatch.
						const cv::Vec3b* const texBase = textureDiffuse.ptr<cv::Vec3b>(0);
						const size_t texStride = textureDiffuse.step.p[0] / sizeof(cv::Vec3b);
						std::vector<float> ssr(NV, 0.f), ssg(NV, 0.f), ssb(NV, 0.f);
						std::vector<uint32_t> scnt(NV, 0);
						// Barycentric interpolation factors from a boundary vertex toward the
						// observed triangle's centroid. Sampling SEVERAL texels a little way
						// INSIDE the triangle (never the exact corner, which frequently lands on
						// a patch gutter / seam-levelling border) and averaging them makes the
						// seed robust to single-pixel JPEG/compression noise. The old code took
						// one texel at the raw corner -- the main source of the "wrong colour"
						// fill, because that single pixel was often a gutter or outlier texel.
						static const float kSeedT[4] = { 0.18f, 0.34f, 0.50f, 0.66f };
						for (const TexturePatch& tp : texturePatches) {
							for (const FIndex fc : tp.faces) {
								const Face& face = faces[fc];
								if (!(vUsed[face[0]] || vUsed[face[1]] || vUsed[face[2]]))
									continue; // face doesn't touch the fill region
								const TexCoord* tc = faceTexcoords.data() + (size_t)fc * 3;
								const float cx = (tc[0].x + tc[1].x + tc[2].x) * (1.f / 3.f);
								const float cy = (tc[0].y + tc[1].y + tc[2].y) * (1.f / 3.f);
								for (int k = 0; k < 3; ++k) {
									const uint32_t v = face[k];
									if (!vUsed[v]) continue; // only boundary verts feed the fill
									float ar = 0.f, ag = 0.f, ab = 0.f; int an = 0;
									for (int s = 0; s < 4; ++s) {
										const float t = kSeedT[s];
										const int sx = (int)(tc[k].x + (cx - tc[k].x) * t + 0.5f);
										const int sy = (int)(tc[k].y + (cy - tc[k].y) * t + 0.5f);
										if ((unsigned)sx >= (unsigned)texW || (unsigned)sy >= (unsigned)texH) continue;
										const cv::Vec3b& bgr = texBase[(size_t)sy * texStride + sx];
										if (bgr[0] == eB && bgr[1] == eG && bgr[2] == eR) continue; // skip empty/gutter texels
										ar += (float)bgr[2]; ag += (float)bgr[1]; ab += (float)bgr[0]; ++an;
									}
									if (an > 0) {
										const float inv = 1.f / (float)an;
										ssr[v] += ar * inv; ssg[v] += ag * inv; ssb[v] += ab * inv;
										++scnt[v];
									}
								}
							}
						}
						for (int u = 0; u < NU; ++u) {
							const uint32_t v = pUsed[u];
							if (scnt[v] > 0) {
								const float inv = 1.f / (float)scnt[v];
								vr[v] = ssr[v] * inv; vg[v] = ssg[v] * inv; vb[v] = ssb[v] * inv;
								vValid[v] = 1; vSeed[v] = 1;
							}
						}
					}
					// global average of the valid colors (fallback so unfilled vertices
					// NEVER use the empty color, which would speckle)
					double gsr = 0, gsg = 0, gsb = 0; size_t gcnt = 0;
					for (int u = 0; u < NU; ++u) {
						const uint32_t v = pUsed[u];
						if (vValid[v]) { gsr += vr[v]; gsg += vg[v]; gsb += vb[v]; ++gcnt; }
					}
					const float gAvgR = gcnt ? (float)(gsr / gcnt) : (float)colEmpty.r;
					const float gAvgG = gcnt ? (float)(gsg / gcnt) : (float)colEmpty.g;
					const float gAvgB = gcnt ? (float)(gsb / gcnt) : (float)colEmpty.b;
					// Build neighbour incidence (CSR) over the no-view faces ONCE, so each
					// diffusion iteration is a conflict-free parallel Jacobi gather instead
					// of a serial per-face scatter (+= into shared vertices). Each face
					// contributes its two opposite vertices to every corner -- duplicates
					// kept, matching the old per-face accumulation's weighting.
					std::vector<uint32_t> off(NV + 1, 0);
					for (int i = 0; i < N; ++i) {
						const Face& face = faces[noViewFaces[i]];
						off[face[0] + 1] += 2; off[face[1] + 1] += 2; off[face[2] + 1] += 2;
					}
					for (size_t v = 0; v < NV; ++v) off[v + 1] += off[v];
					std::vector<uint32_t> nbr(off[NV]);
					{
						std::vector<uint32_t> cur(off.begin(), off.end());
						for (int i = 0; i < N; ++i) {
							const Face& face = faces[noViewFaces[i]];
							const uint32_t a0 = face[0], a1 = face[1], a2 = face[2];
							nbr[cur[a0]++] = a1; nbr[cur[a0]++] = a2;
							nbr[cur[a1]++] = a0; nbr[cur[a1]++] = a2;
							nbr[cur[a2]++] = a0; nbr[cur[a2]++] = a1;
						}
					}
					// TWO-PHASE INPAINT (replaces the old single 240-iteration Jacobi blur,
					// which did filling and smoothing in one loop and therefore averaged every
					// component down to one flat colour = the "muddy smear").
					//
					// PHASE 1 -- FILL: propagate the boundary seed colour inward using the
					// NEAREST valid neighbours only. Seeds stay fixed, so each interior vertex
					// converges to the colour of the observed surface it is closest to along the
					// mesh. A fill that spans two surfaces (e.g. grass->road) keeps grass near
					// grass and road near road instead of blending into mud. Runs until the front
					// stops advancing (fully filled, or an isolated remainder -> global average).
					{
						const int fillMax = 8192; // safety cap; loop early-exits when the front stalls
						const uint32_t* __restrict pOff = off.data();
						const uint32_t* __restrict pNbr = nbr.data();
						std::vector<float> nr(vr), ng(vg), nb(vb);
						std::vector<uint8_t> nValid(vValid);
						for (int iter = 0; iter < fillMax; ++iter) {
							int filled = 0;
							const float* __restrict pvr = vr.data();
							const float* __restrict pvg = vg.data();
							const float* __restrict pvb = vb.data();
							const uint8_t* __restrict pVal = vValid.data();
							float* __restrict pnr = nr.data();
							float* __restrict png = ng.data();
							float* __restrict pnb = nb.data();
							uint8_t* __restrict pnVal = nValid.data();
#ifdef TEXOPT_USE_OPENMP
							#pragma omp parallel for schedule(static) reduction(+:filled)
#endif
							for (int u = 0; u < NU; ++u) {
								const uint32_t v = pUsed[u];
								if (pVal[v]) { pnr[v] = pvr[v]; png[v] = pvg[v]; pnb[v] = pvb[v]; pnVal[v] = 1; continue; }
								float sr = 0.f, sg = 0.f, sb = 0.f; uint32_t c = 0;
								const uint32_t e = pOff[v + 1];
								for (uint32_t k = pOff[v]; k < e; ++k) {
									const uint32_t w = pNbr[k];
									if (pVal[w]) { sr += pvr[w]; sg += pvg[w]; sb += pvb[w]; ++c; }
								}
								if (c > 0) { const float inv = 1.f / c; pnr[v] = sr * inv; png[v] = sg * inv; pnb[v] = sb * inv; pnVal[v] = 1; ++filled; }
								else { pnr[v] = pvr[v]; png[v] = pvg[v]; pnb[v] = pvb[v]; pnVal[v] = 0; }
							}
							vr.swap(nr); vg.swap(ng); vb.swap(nb); vValid.swap(nValid);
							if (filled == 0) break;
						}
					}
					// PHASE 2 -- light seam smoothing: a BOUNDED number of averaging passes over
					// the filled interior, with the boundary seed vertices PINNED. This blurs the
					// propagation fronts left by phase 1 without collapsing the large-scale spatial
					// variation (that is why it no longer needs the 240-iteration global blur), and
					// pinning the seeds keeps the fill locked to the real texture at its border ->
					// no visible fill/real seam.
					{
						const int iters = std::max(0, (int)TEXTURE_DATACOLOR_SEAM_SMOOTH_ITERS);
						const uint32_t* __restrict pOff = off.data();
						const uint32_t* __restrict pNbr = nbr.data();
						const uint8_t* __restrict pSeed = vSeed.data();
						std::vector<float> nr(vr), ng(vg), nb(vb);
						for (int iter = 0; iter < iters; ++iter) {
							const float* __restrict pvr = vr.data();
							const float* __restrict pvg = vg.data();
							const float* __restrict pvb = vb.data();
							const uint8_t* __restrict pVal = vValid.data();
							float* __restrict pnr = nr.data();
							float* __restrict png = ng.data();
							float* __restrict pnb = nb.data();
#ifdef TEXOPT_USE_OPENMP
							#pragma omp parallel for schedule(static)
#endif
							for (int u = 0; u < NU; ++u) {
								const uint32_t v = pUsed[u];
								if (pSeed[v] || !pVal[v]) { pnr[v] = pvr[v]; png[v] = pvg[v]; pnb[v] = pvb[v]; continue; }
								float sr = pvr[v], sg = pvg[v], sb = pvb[v]; uint32_t c = 1;
								const uint32_t e = pOff[v + 1];
								for (uint32_t k = pOff[v]; k < e; ++k) {
									const uint32_t w = pNbr[k];
									if (pVal[w]) { sr += pvr[w]; sg += pvg[w]; sb += pvb[w]; ++c; }
								}
								const float inv = 1.f / c; pnr[v] = sr * inv; png[v] = sg * inv; pnb[v] = sb * inv;
							}
							vr.swap(nr); vg.swap(ng); vb.swap(nb);
						}
					}
					// any vertex STILL unfilled (isolated from all valid color) -> global average
					for (int u = 0; u < NU; ++u) {
						const uint32_t v = pUsed[u];
						if (!vValid[v]) { vr[v] = gAvgR; vg[v] = gAvgG; vb[v] = gAvgB; }
					}
#if TEXTURE_DATACOLOR_COMPONENT_BAKE
					// COMPONENT-TILE bake: pack each connected component of no-view faces
					// into ONE shared atlas tile and rasterise the smooth per-vertex colour
					// field across the whole tile in a single pass. Faces sharing an edge
					// share texels (same tile coordinate space) -> no per-face cell boundary,
					// so the per-cell 8-bit quantization crazing of the per-face bake cannot
					// appear. Because the colour is a smooth diffused field, the planar
					// per-component projection tolerates the foldover of non-planar blobs.
					const int gutter = 2;
					const float targetMaxPx = (float)TEXTURE_DATACOLOR_FILL_RES;
					const int maxTilePx = TEXTURE_DATACOLOR_FILL_MAX_TILE;
					const int minTileDim = 2 + 2 * gutter;
					// 1) connected components of the no-view faces (edge adjacency), split at
					//    sharp creases so each tile stays near-planar (avoids the single-plane
					//    projection foldover that turns steep faces into degenerate tile triangles
					//    -> faint crack network). Colour is already continuous across the split.
					const float tileNormalCos = (float)TEXTURE_DATACOLOR_TILE_NORMAL_COS;
					std::vector<uint8_t> isNoView(faces.size(), 0);
					for (int i = 0; i < N; ++i) isNoView[noViewFaces[i]] = 1;
					std::vector<int> faceComp(faces.size(), -1);
					std::vector<std::vector<FIndex>> compFaces;
					{
						std::vector<FIndex> bfs;
						for (int i = 0; i < N; ++i) {
							const FIndex f0 = noViewFaces[i];
							if (faceComp[f0] != -1) continue;
							const int cid = (int)compFaces.size();
							compFaces.emplace_back();
							faceComp[f0] = cid; bfs.clear(); bfs.push_back(f0);
							while (!bfs.empty()) {
								const FIndex f = bfs.back(); bfs.pop_back();
								compFaces[cid].push_back(f);
								const Normal& nf = scene.mesh.faceNormals[f];
								const Mesh::FaceFaces& adj = faceFaces[f];
								for (int k = 0; k < 3; ++k) {
									const FIndex fn = adj[k];
									if (fn == NO_ID || !isNoView[fn] || faceComp[fn] != -1) continue;
									// keep the tile near-planar: don't cross a sharp crease
									if (tileNormalCos > 0.f) {
										const Normal& nn = scene.mesh.faceNormals[fn];
										if (nf.dot(nn) < tileNormalCos) continue;
									}
									faceComp[fn] = cid; bfs.push_back(fn);
								}
							}
						}
					}
					const int nComp = (int)compFaces.size();
					// 2) per-component planar basis (avg face normal), projection bounds, tile size
					struct Tile { Point3f c, t, b; float umin, vmin, scale; int x, y, w, h; };
					std::vector<Tile> tiles(nComp);
					for (int cc = 0; cc < nComp; ++cc) {
						const std::vector<FIndex>& cfs = compFaces[cc];
						Point3f nrm(0, 0, 0), cen(0, 0, 0); int vcnt = 0;
						for (const FIndex f : cfs) {
							const Normal& fnm = scene.mesh.faceNormals[f];
							nrm.x += fnm.x; nrm.y += fnm.y; nrm.z += fnm.z;
							const Face& face = faces[f];
							for (int k = 0; k < 3; ++k) { const Vertex& P = vertices[face[k]]; cen.x += P.x; cen.y += P.y; cen.z += P.z; ++vcnt; }
						}
						const float invc = vcnt ? 1.f / (float)vcnt : 1.f;
						cen.x *= invc; cen.y *= invc; cen.z *= invc;
						float nl = std::sqrt(nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z);
						if (nl < 1e-12f) { nrm = Point3f(0, 0, 1); nl = 1.f; }
						nrm.x /= nl; nrm.y /= nl; nrm.z /= nl;
						const Point3f a = (fabsf(nrm.x) < 0.9f) ? Point3f(1, 0, 0) : Point3f(0, 1, 0);
						Point3f t(a.y * nrm.z - a.z * nrm.y, a.z * nrm.x - a.x * nrm.z, a.x * nrm.y - a.y * nrm.x);
						float tl = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z); if (tl < 1e-12f) tl = 1.f;
						t.x /= tl; t.y /= tl; t.z /= tl;
						const Point3f bb(nrm.y * t.z - nrm.z * t.y, nrm.z * t.x - nrm.x * t.z, nrm.x * t.y - nrm.y * t.x);
						float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
						for (const FIndex f : cfs) {
							const Face& face = faces[f];
							for (int k = 0; k < 3; ++k) {
								const Vertex& P = vertices[face[k]];
								const float du = (P.x - cen.x) * t.x + (P.y - cen.y) * t.y + (P.z - cen.z) * t.z;
								const float dv = (P.x - cen.x) * bb.x + (P.y - cen.y) * bb.y + (P.z - cen.z) * bb.z;
								umin = std::min(umin, du); umax = std::max(umax, du);
								vmin = std::min(vmin, dv); vmax = std::max(vmax, dv);
							}
						}
						const float eu = std::max(0.f, umax - umin), ev = std::max(0.f, vmax - vmin);
						const float ext = std::max(eu, ev);
						const float scale = (ext > 1e-9f) ? (targetMaxPx / ext) : 1.f;
						int w = std::min(std::max((int)std::ceil(eu * scale) + 2 * gutter, minTileDim), maxTilePx);
						int h = std::min(std::max((int)std::ceil(ev * scale) + 2 * gutter, minTileDim), maxTilePx);
						if (w > cols) w = cols;
						tiles[cc] = Tile{ cen, t, bb, umin, vmin, scale, 0, 0, w, h };
					}
					// 3) shelf-pack the tiles into appended atlas rows
					int shelfX = 0, shelfY = 0, shelfH = 0;
					for (int cc = 0; cc < nComp; ++cc) {
						Tile& T = tiles[cc];
						if (shelfX + T.w > cols) { shelfY += shelfH; shelfX = 0; shelfH = 0; }
						T.x = shelfX; T.y = oldRows + shelfY;
						shelfX += T.w; shelfH = std::max(shelfH, T.h);
					}
					const int extraRows = shelfY + shelfH;
					cv::Mat newTex(oldRows + extraRows, cols, CV_8UC3, cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));
					textureDiffuse.copyTo(newTex(cv::Rect(0, 0, cols, oldRows)));
					// Cache newTex base pointer + row stride once (shared read-only across the
					// component-parallel loop; each thread writes a disjoint tile). Replaces the
					// per-texel newTex.at<cv::Vec3b>() in the bake + flood hot loops.
					cv::Vec3b* const nbase = newTex.ptr<cv::Vec3b>(0);
					const size_t nstride = newTex.step.p[0] / sizeof(cv::Vec3b);
					static const float kBayer4c[16] = { 0.f,8.f,2.f,10.f, 12.f,4.f,14.f,6.f, 3.f,11.f,1.f,9.f, 15.f,7.f,13.f,5.f };
					const float detailGain = (float)TEXTURE_DATACOLOR_DETAIL_GAIN;
					const int DETAIL_D = 64; // donor detail patch size (px)
					// Reflection-padding (mirror) source = the ORIGINAL observed atlas, read-only
					// (writes go to newTex, so sampling textureDiffuse here is thread-safe).
					const float mirrorGain = (float)TEXTURE_DATACOLOR_MIRROR_GAIN;
					const cv::Vec3b* const tdBase = textureDiffuse.ptr<cv::Vec3b>(0);
					const size_t tdStride = textureDiffuse.step.p[0] / sizeof(cv::Vec3b);
					const int tdW = textureDiffuse.cols, tdH = textureDiffuse.rows;
					const uchar meB = (uchar)colEmpty.b, meG = (uchar)colEmpty.g, meR = (uchar)colEmpty.r;
					// 4) rasterise each component's faces into its tile (Gouraud over the shared
					//    field), then a small gutter dilation so GPU bilinear at the tile border
					//    samples in-colour texels, not the colEmpty background. Tiles are disjoint
					//    atlas regions and each face is in exactly one component -> parallel safe.
#ifdef TEXOPT_USE_OPENMP
					#pragma omp parallel for schedule(dynamic, 8)
#endif
					for (int cc = 0; cc < nComp; ++cc) {
						const Tile& T = tiles[cc];
						const int tx = T.x, ty = T.y, tw = T.w, th = T.h;
						std::vector<uint8_t> painted((size_t)tw * th, 0);
						// DETAIL DONOR: real high-frequency texture from the observed surface that
						// borders this fill tile, so the fill carries foliage/gravel grain instead
						// of reading as a flat blob. Pick the largest-area OBSERVED neighbour face,
						// crop a square of real atlas texture around it, and high-pass it (subtract a
						// blur) -> zero-mean grain in BGR. Added mirror-tiled onto the smooth base
						// during the texel bake below. textureDiffuse here is still the ORIGINAL
						// observed atlas (read-only) and covered faces keep absolute-atlas texcoords
						// -> thread-safe.
						cv::Mat detail; // DETAIL_D x DETAIL_D CV_32FC3 high-pass, empty => none
						if (detailGain > 0.f) {
							const int dW = textureDiffuse.cols, dH = textureDiffuse.rows;
							FIndex donorFace = NO_ID; float donorArea = 0.f;
							for (const FIndex f : compFaces[cc]) {
								const Mesh::FaceFaces& adj = faceFaces[f];
								for (int k = 0; k < 3; ++k) {
									const FIndex fn = adj[k];
									if (fn == NO_ID || !covered[fn]) continue;
									const TexCoord* tc = faceTexcoords.data() + (size_t)fn * 3;
									const float aw = std::max(std::max(tc[0].x, tc[1].x), tc[2].x) - std::min(std::min(tc[0].x, tc[1].x), tc[2].x);
									const float ah = std::max(std::max(tc[0].y, tc[1].y), tc[2].y) - std::min(std::min(tc[0].y, tc[1].y), tc[2].y);
									const float area = aw * ah;
									if (area > donorArea) { donorArea = area; donorFace = fn; }
								}
							}
							if (donorFace != NO_ID && dW >= DETAIL_D && dH >= DETAIL_D) {
								const TexCoord* tc = faceTexcoords.data() + (size_t)donorFace * 3;
								int x0 = (int)((tc[0].x + tc[1].x + tc[2].x) / 3.f) - DETAIL_D / 2;
								int y0 = (int)((tc[0].y + tc[1].y + tc[2].y) / 3.f) - DETAIL_D / 2;
								x0 = std::max(0, std::min(x0, dW - DETAIL_D));
								y0 = std::max(0, std::min(y0, dH - DETAIL_D));
								cv::Mat crop, low;
								textureDiffuse(cv::Rect(x0, y0, DETAIL_D, DETAIL_D)).convertTo(crop, CV_32FC3);
								cv::blur(crop, low, cv::Size(9, 9));
								detail = crop - low; // zero-mean high-pass grain (BGR)
							}
						}
						const cv::Vec3f* const dbase = detail.empty() ? nullptr : detail.ptr<cv::Vec3f>(0);
						const size_t dstride = detail.empty() ? 0 : detail.step.p[0] / sizeof(cv::Vec3f);
						for (const FIndex f : compFaces[cc]) {
							const Face& face = faces[f];
							float ax[3], ay[3], cr[3], cg[3], cb[3];
							for (int k = 0; k < 3; ++k) {
								const Vertex& P = vertices[face[k]];
								const float du = (P.x - T.c.x) * T.t.x + (P.y - T.c.y) * T.t.y + (P.z - T.c.z) * T.t.z;
								const float dv = (P.x - T.c.x) * T.b.x + (P.y - T.c.y) * T.b.y + (P.z - T.c.z) * T.b.z;
								ax[k] = tx + gutter + (du - T.umin) * T.scale;
								ay[k] = ty + gutter + (dv - T.vmin) * T.scale;
								cr[k] = vr[face[k]]; cg[k] = vg[face[k]]; cb[k] = vb[face[k]];
							}
							TexCoord* texcoords = faceTexcoords.data() + (size_t)f * 3;
							texcoords[0] = TexCoord(ax[0], ay[0]);
							texcoords[1] = TexCoord(ax[1], ay[1]);
							texcoords[2] = TexCoord(ax[2], ay[2]);
							// MIRROR SOURCE: if this fill face borders an OBSERVED face, set up
							// reflection padding across their shared edge. mS0,mS1 = this face's
							// vertex slots on the shared edge; mFo = its far (opposite) slot;
							// mOTsa/mOTsb/mOToc = the observed neighbour's atlas texcoords for
							// (sharedA, sharedB, neighbourOpposite). A fill texel with barycentric
							// depth d = w[mFo] from the shared edge samples the neighbour at bary
							// (sharedA,sharedB,oppO) = ((1-d)*along, (1-d)*(1-along), d) -> the
							// reflected point; blended by mirrorGain*(1-d) below.
							int mS0 = -1, mS1 = -1, mFo = -1;
							TexCoord mOTsa, mOTsb, mOToc;
							if (mirrorGain > 0.f) {
								const Mesh::FaceFaces& adjF = faceFaces[f];
								for (int k = 0; k < 3; ++k) {
									const FIndex fo = adjF[k];
									if (fo == NO_ID || !covered[fo]) continue;
									const Face& of = faces[fo];
									int sSlot[2], ns = 0, oppSlot = -1;
									for (int a = 0; a < 3; ++a) {
										bool sh = (face[a] == of[0] || face[a] == of[1] || face[a] == of[2]);
										if (sh) { if (ns < 2) sSlot[ns] = a; ++ns; } else oppSlot = a;
									}
									if (ns != 2 || oppSlot < 0) continue;
									const uint32_t vsa = face[sSlot[0]], vsb = face[sSlot[1]];
									int oa = -1, ob = -1, ocSlot = -1;
									for (int b = 0; b < 3; ++b) {
										if (of[b] == vsa) oa = b; else if (of[b] == vsb) ob = b; else ocSlot = b;
									}
									if (oa < 0 || ob < 0 || ocSlot < 0) continue;
									const TexCoord* otc = faceTexcoords.data() + (size_t)fo * 3;
									mS0 = sSlot[0]; mS1 = sSlot[1]; mFo = oppSlot;
									mOTsa = otc[oa]; mOTsb = otc[ob]; mOToc = otc[ocSlot];
									break;
								}
							}
							const float denom = (ay[1] - ay[2]) * (ax[0] - ax[2]) + (ax[2] - ax[1]) * (ay[0] - ay[2]);
							const float invDen = (fabsf(denom) > 1e-6f) ? 1.f / denom : 0.f;
							int minx = (int)std::floor(std::min(ax[0], std::min(ax[1], ax[2])));
							int maxx = (int)std::ceil (std::max(ax[0], std::max(ax[1], ax[2])));
							int miny = (int)std::floor(std::min(ay[0], std::min(ay[1], ay[2])));
							int maxy = (int)std::ceil (std::max(ay[0], std::max(ay[1], ay[2])));
							minx = std::max(minx, tx); maxx = std::min(maxx, tx + tw - 1);
							miny = std::max(miny, ty); maxy = std::min(maxy, ty + th - 1);
							for (int py = miny; py <= maxy; ++py)
								for (int px = minx; px <= maxx; ++px) {
									const float fx = px + 0.5f, fy = py + 0.5f;
									float w0 = ((ay[1] - ay[2]) * (fx - ax[2]) + (ax[2] - ax[1]) * (fy - ay[2])) * invDen;
									float w1 = ((ay[2] - ay[0]) * (fx - ax[2]) + (ax[0] - ax[2]) * (fy - ay[2])) * invDen;
									float w2 = 1.f - w0 - w1;
									if (w0 < -0.01f || w1 < -0.01f || w2 < -0.01f) continue;
									if (w0 < 0) w0 = 0; if (w1 < 0) w1 = 0; if (w2 < 0) w2 = 0;
									const float s = w0 + w1 + w2; if (s <= 1e-6f) continue;
									const float iw = 1.f / s; w0 *= iw; w1 *= iw; w2 *= iw;
									float rr = cr[0]*w0 + cr[1]*w1 + cr[2]*w2;
									float gg = cg[0]*w0 + cg[1]*w1 + cg[2]*w2;
									float bv = cb[0]*w0 + cb[1]*w1 + cb[2]*w2;
									// REFLECTION PADDING: mirror the observed neighbour's real texture
									// across the shared edge into this fill texel, strong at the edge
									// (seamless) and fading inward -> the fill carries real detail.
									if (mS0 >= 0) {
										const float wv[3] = { w0, w1, w2 };
										const float d = wv[mFo];              // depth from shared edge
										const float ab = wv[mS0] + wv[mS1];    // along-edge weight
										if (ab > 1e-6f) {
											const float along = wv[mS0] / ab;
											const float bsa = (1.f - d) * along;
											const float bsb = (1.f - d) * (1.f - along);
											const float mmx = bsa * mOTsa.x + bsb * mOTsb.x + d * mOToc.x;
											const float mmy = bsa * mOTsa.y + bsb * mOTsb.y + d * mOToc.y;
											const int ix = (int)(mmx + 0.5f), iy = (int)(mmy + 0.5f);
											if ((unsigned)ix < (unsigned)tdW && (unsigned)iy < (unsigned)tdH) {
												const cv::Vec3b& mc = tdBase[(size_t)iy * tdStride + ix];
												if (!(mc[0] == meB && mc[1] == meG && mc[2] == meR)) {
													const float wm = mirrorGain * (1.f - d);
													bv += wm * ((float)mc[0] - bv);
													gg += wm * ((float)mc[1] - gg);
													rr += wm * ((float)mc[2] - rr);
												}
											}
										}
									}
									// inject real texture grain (mirror-tiled donor high-pass)
									if (!detail.empty()) {
										const int per = 2 * DETAIL_D;
										int mx = (px - tx) % per; if (mx < 0) mx += per; if (mx >= DETAIL_D) mx = per - 1 - mx;
										int my = (py - ty) % per; if (my < 0) my += per; if (my >= DETAIL_D) my = per - 1 - my;
										const cv::Vec3f& dv = dbase[(size_t)my * dstride + mx];
										bv += detailGain * dv[0]; gg += detailGain * dv[1]; rr += detailGain * dv[2];
									}
									const float dth = (kBayer4c[(py & 3) * 4 + (px & 3)] + 0.5f) * (1.f / 16.f) - 0.5f;
									int ir = (int)(rr + 0.5f + dth); ir = ir < 0 ? 0 : (ir > 255 ? 255 : ir);
									int ig = (int)(gg + 0.5f + dth); ig = ig < 0 ? 0 : (ig > 255 ? 255 : ig);
									int ib = (int)(bv + 0.5f + dth); ib = ib < 0 ? 0 : (ib > 255 ? 255 : ib);
#if TEXTURE_DATACOLOR_DEBUG_TINT || TEXTURE_DATACOLOR_DIAG_TINT
									// DEBUG: mark all synthetic fill bright magenta (BGR 255,0,255)
									ib = 255; ig = 0; ir = 255;
#endif
									nbase[(size_t)py * nstride + px] = cv::Vec3b((uchar)ib, (uchar)ig, (uchar)ir);
									painted[(size_t)(py - ty) * tw + (px - tx)] = 1;
								}
						}
						// FULL-TILE FLOOD: dilate the painted fill colour outward until the ENTIRE
						// tile background is filled (not just a 2px gutter). The tile background is
						// initialised to the dark colEmpty; if any of it survives near the fill's
						// outer boundary, GPU bilinear filtering -- and especially mipmap
						// minification -- blends the fill with that dark background and paints a
						// dark rim tracing the fill edge (the observed patches avoid this via the
						// 24px TEXTURE_OUTWARD_DILATE_PX; the fill tiles only had 2px). Flooding the
						// whole tile with nearest-colour removes the dark background entirely, so the
						// fill boundary blends cleanly into the surrounding real texture. Tiles are
						// small (<=256px) so a full flood is cheap. (gutter still insets the triangle
						// placement; this just guarantees no colEmpty is ever sampled.)
						// FRONTIER (BFS wavefront) flood: same multi-source Jacobi rings as a
						// full-tile-rescan-per-pass, but O(tile area) total instead of
						// O(area * perimeter) -- only cells on the advancing front are visited,
						// not the whole tile every pass. Each cell is filled with the average of
						// its already-painted neighbours at the moment it is reached, and all
						// cells of a ring are committed together (reads use the pre-ring painted
						// state), so the rings and averages -- hence the output -- are identical
						// to the rescan version.
						std::vector<int> frontier, nextf;
						std::vector<uint8_t> queued((size_t)tw * th, 0);
						// seed: unpainted cells adjacent to a painted cell (one scan, O(area))
						for (int ly = 0; ly < th; ++ly)
							for (int lx = 0; lx < tw; ++lx) {
								if (painted[(size_t)ly * tw + lx]) continue;
								bool adj = false;
								for (int dy = -1; dy <= 1 && !adj; ++dy) {
									const int ny = ly + dy; if ((unsigned)ny >= (unsigned)th) continue;
									for (int dx = -1; dx <= 1; ++dx) {
										const int nx = lx + dx; if ((unsigned)nx >= (unsigned)tw) continue;
										if (painted[(size_t)ny * tw + nx]) { adj = true; break; }
									}
								}
								if (adj) { const int id = (int)((size_t)ly * tw + lx); frontier.push_back(id); queued[id] = 1; }
							}
						std::vector<std::pair<int, cv::Vec3b>> adds;
						while (!frontier.empty()) {
							adds.clear();
							for (const int id : frontier) {
								const int ly = id / tw, lx = id % tw;
								cv::Vec3i sum(0, 0, 0); int cntn = 0;
								for (int dy = -1; dy <= 1; ++dy) {
									const int ny = ly + dy; if ((unsigned)ny >= (unsigned)th) continue;
									for (int dx = -1; dx <= 1; ++dx) {
										const int nx = lx + dx; if ((unsigned)nx >= (unsigned)tw) continue;
										if (!painted[(size_t)ny * tw + nx]) continue;
										sum += cv::Vec3i(nbase[(size_t)(ty + ny) * nstride + (tx + nx)]); ++cntn;
									}
								}
								if (cntn) adds.emplace_back(id, cv::Vec3b((uchar)(sum[0] / cntn), (uchar)(sum[1] / cntn), (uchar)(sum[2] / cntn)));
							}
							if (adds.empty()) break;
							// commit this ring (all reads above used the pre-ring painted state)
							for (const auto& ad : adds) {
								const int ly = ad.first / tw, lx = ad.first % tw;
								nbase[(size_t)(ty + ly) * nstride + (tx + lx)] = ad.second;
								painted[ad.first] = 1;
							}
							// build next front from the newly painted cells' unpainted neighbours
							nextf.clear();
							for (const auto& ad : adds) {
								const int ly = ad.first / tw, lx = ad.first % tw;
								for (int dy = -1; dy <= 1; ++dy) {
									const int ny = ly + dy; if ((unsigned)ny >= (unsigned)th) continue;
									for (int dx = -1; dx <= 1; ++dx) {
										const int nx = lx + dx; if ((unsigned)nx >= (unsigned)tw) continue;
										const int nid = (int)((size_t)ny * tw + nx);
										if (!painted[nid] && !queued[nid]) { queued[nid] = 1; nextf.push_back(nid); }
									}
								}
							}
							frontier.swap(nextf);
						}
					}
#if TEXTURE_DATACOLOR_FEATHER_RINGS > 0
					// ------------------------------------------------------------
					// SEAM BLEND: make the boundary of the real observed texture disappear --
					// both where it meets the synthesized fill AND where the mesh simply ends
					// (the silhouette/rim). Over a band of OBSERVED faces (up to K face-rings
					// from the boundary) the observed texture is cross-faded toward the FILL's
					// OWN pinned colour field, ramping from FEATHER_STRENGTH at the boundary to
					// 0 at the band's inner edge. Because both sides use the SAME colour along
					// the shared mesh edge, there is no continuity break (no jagged seam line);
					// and the fade to a smooth colour also matches the fill's blurriness so the
					// sharp-vs-blurry mismatch is gone too. Per-vertex ring (shared at a vertex)
					// keeps the ramp continuous across shared edges.
					{
						const int K = (int)TEXTURE_DATACOLOR_FEATHER_RINGS;
						const float maxAlpha = (float)TEXTURE_DATACOLOR_FEATHER_STRENGTH;
						// 1) face BFS ring index. Ring 0 = every COVERED (real-texture) face on
						//    the BOUNDARY of the textured region -- i.e. adjacent to a no-view
						//    fill face OR to an OPEN mesh edge (faceFaces == NO_ID, the mesh
						//    silhouette/rim). Seeding the rim as well lets the feather soften the
						//    crisp silhouette where the mesh simply ENDS (the fill/flood halo
						//    beyond it is blurry, so a hard crisp edge stands out), not only the
						//    observed<->fill seam. One O(faces) scan.
						std::vector<int> fRing(faces.size(), -1);
						std::vector<FIndex> cur, nxt, bandFaces;
						for (FIndex f = 0; f < (FIndex)faces.size(); ++f) {
							if (!covered[f]) continue;
							const Mesh::FaceFaces& adj = faceFaces[f];
							bool boundary = false;
							for (int k = 0; k < 3; ++k) {
								const FIndex fn = adj[k];
								if (fn == NO_ID || !covered[fn]) { boundary = true; break; }
							}
							if (boundary) { fRing[f] = 0; cur.push_back(f); bandFaces.push_back(f); }
						}
						for (int r = 1; r < K && !cur.empty(); ++r) {
							nxt.clear();
							for (const FIndex f : cur) {
								const Mesh::FaceFaces& adj = faceFaces[f];
								for (int k = 0; k < 3; ++k) {
									const FIndex fn = adj[k];
									if (fn == NO_ID || !covered[fn] || fRing[fn] != -1) continue;
									fRing[fn] = r; nxt.push_back(fn); bandFaces.push_back(fn);
								}
							}
							cur.swap(nxt);
						}
						if (!bandFaces.empty()) {
							const size_t NVv = vertices.size();
							// 2) per-vertex ring (min over incident band faces) -> the alpha ramp.
							std::vector<int> vRing(NVv, K);
							for (const FIndex f : bandFaces) {
								const Face& face = faces[f];
								for (int k = 0; k < 3; ++k) { const uint32_t v = face[k]; if (fRing[f] < vRing[v]) vRing[v] = fRing[f]; }
							}
							// 3) CONTINUITY BLEND (replaces the box-blur smear, which left a jagged
							//    seam: it blurred each side toward its OWN neighbourhood, so the
							//    observed and fill values still differed at the shared mesh edge).
							//    Instead, blend the observed band toward the FILL's OWN pinned colour
							//    field (vr/vg/vb): extend that field outward across the band, then at
							//    each band texel mix observed->fill-colour by the ramped alpha (1 at
							//    the boundary, 0 at the band's inner edge). At the shared edge the
							//    observed target IS vr/vg/vb -- exactly what the fill face uses there
							//    -- so both sides evaluate the SAME colour along the edge => no seam.
							//    Moving inward the alpha fades so the real detail returns; the flatten
							//    toward the (smooth) fill colour also matches the fill's blurriness.
							std::vector<float> tr(NVv, 0.f), tg(NVv, 0.f), tb(NVv, 0.f);
							std::vector<uint8_t> tv(NVv, 0);
							for (const FIndex f : bandFaces) {
								const Face& face = faces[f];
								for (int k = 0; k < 3; ++k) {
									const uint32_t v = face[k];
									if (!tv[v] && vValid[v]) { tr[v] = vr[v]; tg[v] = vg[v]; tb[v] = vb[v]; tv[v] = 1; }
								}
							}
							// propagate the fill colour outward to every band vertex (K passes over
							// the band faces; each pass fills an invalid vertex from its face's valid
							// verts) so the whole band has a fill-consistent target colour.
							for (int pass = 0; pass < K; ++pass) {
								int filled = 0;
								for (const FIndex f : bandFaces) {
									const Face& face = faces[f];
									float sr = 0.f, sg = 0.f, sb = 0.f; int cn = 0;
									for (int k = 0; k < 3; ++k) { const uint32_t v = face[k]; if (tv[v]) { sr += tr[v]; sg += tg[v]; sb += tb[v]; ++cn; } }
									if (cn == 0 || cn == 3) continue;
									const float inv = 1.f / (float)cn;
									for (int k = 0; k < 3; ++k) { const uint32_t v = face[k]; if (!tv[v]) { tr[v] = sr * inv; tg[v] = sg * inv; tb[v] = sb * inv; tv[v] = 2; ++filled; } }
								}
								for (const FIndex f : bandFaces) { const Face& face = faces[f]; for (int k = 0; k < 3; ++k) if (tv[face[k]] == 2) tv[face[k]] = 1; }
								if (filled == 0) break;
							}
							// 4) rasterise each band face, blend observed -> fill colour by the ramp.
							//    SOURCE observed = tdBase (original, unmodified) to avoid feedback;
							//    DEST = newTex. Serial: band faces in a patch can share edge texels.
#if TEXTURE_DATACOLOR_DIAG
							long long dgWritten = 0, dgSkipEmpty = 0, dgSkipAlpha = 0, dgBoundN = 0;
							double dgSumStep = 0, dgMaxStep = 0, dgSumResid = 0, dgMaxResid = 0;
#endif
							const float invK = (K > 1) ? 1.f / (float)(K - 1) : 0.f;
							for (const FIndex f : bandFaces) {
								const Face& face = faces[f];
								const TexCoord* tc = faceTexcoords.data() + (size_t)f * 3;
								float ax[3], ay[3], al[3], cr[3], cg[3], cb[3];
								bool ok = true;
								for (int k = 0; k < 3; ++k) {
									const uint32_t v = face[k];
									ax[k] = tc[k].x; ay[k] = tc[k].y;
									if (!tv[v]) { ok = false; break; }
									cr[k] = tr[v]; cg[k] = tg[v]; cb[k] = tb[v];
									// SMOOTHSTEP ramp: t=1 at the boundary (full continuity with the
									// fill) -> t=0 at the band's inner edge. The S-curve has ~zero
									// slope at BOTH ends, so the seam stays locked AND the real detail
									// returns with no visible front (a linear ramp leaves a faint kink).
									float t = (K > 1) ? (float)(K - 1 - vRing[v]) * invK : 1.f;
									t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
									al[k] = maxAlpha * t * t * (3.f - 2.f * t);
								}
								if (!ok) continue;
								const float denom = (ay[1] - ay[2]) * (ax[0] - ax[2]) + (ax[2] - ax[1]) * (ay[0] - ay[2]);
								const float invDen = (fabsf(denom) > 1e-6f) ? 1.f / denom : 0.f;
								int minx = (int)std::floor(std::min(ax[0], std::min(ax[1], ax[2])));
								int maxx = (int)std::ceil (std::max(ax[0], std::max(ax[1], ax[2])));
								int miny = (int)std::floor(std::min(ay[0], std::min(ay[1], ay[2])));
								int maxy = (int)std::ceil (std::max(ay[0], std::max(ay[1], ay[2])));
								minx = std::max(minx, 0); maxx = std::min(maxx, tdW - 1);
								miny = std::max(miny, 0); maxy = std::min(maxy, tdH - 1);
								for (int py = miny; py <= maxy; ++py)
									for (int px = minx; px <= maxx; ++px) {
										const float fx = px + 0.5f, fy = py + 0.5f;
										float w0 = ((ay[1] - ay[2]) * (fx - ax[2]) + (ax[2] - ax[1]) * (fy - ay[2])) * invDen;
										float w1 = ((ay[2] - ay[0]) * (fx - ax[2]) + (ax[0] - ax[2]) * (fy - ay[2])) * invDen;
										float w2 = 1.f - w0 - w1;
										if (w0 < -0.01f || w1 < -0.01f || w2 < -0.01f) continue;
										if (w0 < 0) w0 = 0; if (w1 < 0) w1 = 0; if (w2 < 0) w2 = 0;
										const float s = w0 + w1 + w2; if (s <= 1e-6f) continue;
										const float iw = 1.f / s; w0 *= iw; w1 *= iw; w2 *= iw;
										const float a = al[0] * w0 + al[1] * w1 + al[2] * w2;
										if (a <= 0.f) {
#if TEXTURE_DATACOLOR_DIAG
											++dgSkipAlpha;
#endif
											continue;
										}
										cv::Vec3b& p3 = nbase[(size_t)py * nstride + px];
										if (p3[0] == meB && p3[1] == meG && p3[2] == meR) {
#if TEXTURE_DATACOLOR_DIAG
											++dgSkipEmpty;
#endif
											continue;
										}
										const cv::Vec3b& obs = tdBase[(size_t)py * tdStride + px];
										const float tR = cr[0] * w0 + cr[1] * w1 + cr[2] * w2;
										const float tG = cg[0] * w0 + cg[1] * w1 + cg[2] * w2;
										const float tB = cb[0] * w0 + cb[1] * w1 + cb[2] * w2;
										int ib = (int)((float)obs[0] + a * (tB - (float)obs[0]) + 0.5f); ib = ib < 0 ? 0 : (ib > 255 ? 255 : ib);
										int ig = (int)((float)obs[1] + a * (tG - (float)obs[1]) + 0.5f); ig = ig < 0 ? 0 : (ig > 255 ? 255 : ig);
										int ir = (int)((float)obs[2] + a * (tR - (float)obs[2]) + 0.5f); ir = ir < 0 ? 0 : (ir > 255 ? 255 : ir);
#if TEXTURE_DATACOLOR_DIAG
										{
											const double tl = ((double)tR + tG + tB) * (1.0 / 3.0);
											const double ol = ((double)obs[0] + obs[1] + obs[2]) * (1.0 / 3.0);
											const double rl = ((double)ib + ig + ir) * (1.0 / 3.0);
											++dgWritten;
											if (a > 0.9f) { // near the ring-0 boundary
												++dgBoundN;
												const double ds = std::fabs(ol - tl); dgSumStep += ds; if (ds > dgMaxStep) dgMaxStep = ds;
												const double rs = std::fabs(rl - tl); dgSumResid += rs; if (rs > dgMaxResid) dgMaxResid = rs;
											}
										}
#endif
#if TEXTURE_DATACOLOR_DIAG_TINT
										// region false-colour: feather BAND = GREEN (bright at the boundary,
										// dimming inward) so fill/band/observed are visually distinct.
										{ int gv = (int)(60.f + 195.f * a); if (gv > 255) gv = 255; p3 = cv::Vec3b(0, (uchar)gv, 0); continue; }
#endif
										p3 = cv::Vec3b((uchar)ib, (uchar)ig, (uchar)ir);
									}
							}
#if TEXTURE_DATACOLOR_DIAG
							DEBUG("[DATACOLOR-DIAG] noView=%d nComp=%d band=%d | written=%lld skipEmpty=%lld skipAlpha=%lld | boundary=%lld meanStep=%.2f maxStep=%.2f meanResid=%.2f maxResid=%.2f",
								N, nComp, (int)bandFaces.size(),
								dgWritten, dgSkipEmpty, dgSkipAlpha, dgBoundN,
								dgBoundN ? dgSumStep / (double)dgBoundN : 0.0, dgMaxStep,
								dgBoundN ? dgSumResid / (double)dgBoundN : 0.0, dgMaxResid);
#endif
						}
					}
#endif // TEXTURE_DATACOLOR_FEATHER_RINGS
					textureDiffuse = newTex;
					DEBUG("Data-colored %d unobserved faces in %d component tiles (%d atlas rows, nearest-fill + %d seam-smooth iters)", N, nComp, extraRows, (int)TEXTURE_DATACOLOR_SEAM_SMOOTH_ITERS);
#else
					// PN-triangle (quadratic Bezier) edge control colours -- this is what
					// removes the residual triangle BANDING. A linear barycentric blend of
					// the 3 vertex colours is only C0: its value matches across a shared
					// edge but its SLOPE jumps, so on the large unobserved-region triangles
					// the eye reads those C1 discontinuities as facets (Mach banding).
					// Promote every face to a quadratic colour patch with 3 extra edge
					// control colours. Each edge control colour is computed SYMMETRICALLY
					// from the two faces that share the edge (the 2 edge-endpoint colours +
					// the 2 opposite-vertex colours), so both faces produce the IDENTICAL
					// control colour for the shared edge -> the patch is continuous in value
					// AND cross-edge slope = no facet seams. ecX[i*3+e] is the control for
					// edge e of face i: e0=(v0,v1), e1=(v1,v2), e2=(v2,v0).
					std::vector<float> ecR((size_t)N * 3), ecG((size_t)N * 3), ecB((size_t)N * 3);
					{
						struct EdgeOpp { int count; int o0; int o1; }; // value-inits to {0,0,0}
						std::unordered_map<uint64_t, EdgeOpp> edgeOpp;
						edgeOpp.reserve((size_t)N * 3);
						auto ekey = [](uint32_t a, uint32_t b) -> uint64_t {
							if (a > b) { const uint32_t t = a; a = b; b = t; }
							return ((uint64_t)a << 32) | (uint64_t)b;
						};
						// Pass 1: collect the (up to 2) opposite vertices per edge.
						for (int i = 0; i < N; ++i) {
							const Face& face = faces[noViewFaces[i]];
							const uint32_t v0 = face[0], v1 = face[1], v2 = face[2];
							const uint32_t ea[3] = { v0, v1, v2 };
							const uint32_t eb[3] = { v1, v2, v0 };
							const uint32_t opp[3] = { v2, v0, v1 };
							for (int e = 0; e < 3; ++e) {
								EdgeOpp& s = edgeOpp[ekey(ea[e], eb[e])];
								if (s.count == 0) s.o0 = (int)opp[e];
								else if (s.count == 1) s.o1 = (int)opp[e];
								++s.count;
							}
						}
						// Pass 2: per-face edge control colour = mean of the surrounding
						// vertices (2 endpoints + up to 2 opposite). Shared edges resolve to
						// the same vertex set from either side -> identical control colour.
						for (int i = 0; i < N; ++i) {
							const Face& face = faces[noViewFaces[i]];
							const uint32_t v0 = face[0], v1 = face[1], v2 = face[2];
							const uint32_t ea[3] = { v0, v1, v2 };
							const uint32_t eb[3] = { v1, v2, v0 };
							for (int e = 0; e < 3; ++e) {
								const uint32_t a = ea[e], b = eb[e];
								float sr = vr[a] + vr[b], sg = vg[a] + vg[b], sb = vb[a] + vb[b];
								int cnt = 2;
								const auto it = edgeOpp.find(ekey(a, b));
								if (it != edgeOpp.end()) {
									const EdgeOpp& s = it->second;
									const int no = s.count < 2 ? s.count : 2;
									if (no >= 1) { const int o = s.o0; sr += vr[o]; sg += vg[o]; sb += vb[o]; ++cnt; }
									if (no >= 2) { const int o = s.o1; sr += vr[o]; sg += vg[o]; sb += vb[o]; ++cnt; }
								}
								const float inv = 1.f / (float)cnt;
								const size_t idx = (size_t)i * 3 + e;
								ecR[idx] = sr * inv; ecG[idx] = sg * inv; ecB[idx] = sb * inv;
							}
						}
					}
					// bake one small quadratic-patch cell per face; UVs -> the 3 corners
					// M is the cell size; G is a GUTTER (px) kept on every side. The
					// triangle's 3 corners are inset by G so the rendered triangle edges
					// sit >=G px inside the cell. The bake fills the FULL cell (clamped
					// barycentric outside the triangle = edge colour), so when the GPU
					// bilinearly samples a texel right on a triangle edge, its filter
					// kernel reaches only into same-colour gutter texels -- never into the
					// neighbouring atlas cell or the dark colEmpty background. Without the
					// gutter (G~0.5) that bleed shows up as dark seams along every edge.
					const int M = 12; // cell size (px)
					const float G = 2.0f; // gutter (px) on each side
					const int cellsPerRow = std::max(1, cols / M);
					const int cellRows = (N + cellsPerRow - 1) / cellsPerRow;
					const int extraRows = cellRows * M;
					cv::Mat newTex(oldRows + extraRows, cols, CV_8UC3, cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));
					textureDiffuse.copyTo(newTex(cv::Rect(0, 0, cols, oldRows)));
					// 4x4 ordered (Bayer) dither, indexed by ABSOLUTE atlas pixel (px,py).
					// Each face bakes its own atlas cell and rounds to 8-bit independently;
					// on a smooth field the per-cell rounding mismatch along a shared edge is
					// COHERENT and traces every triangle -> faint polygon outlines (the same
					// root cause as the over-flatten "crazing"). Adding a sub-LSB dither keyed
					// to the global pixel position de-correlates that rounding across the edge,
					// so the residual mismatch becomes incoherent noise instead of a clean
					// outline -> the tessellation stops reading. Normalised to [-0.5, 0.5).
					static const float kBayer4[16] = {
						 0.f,  8.f,  2.f, 10.f,
						12.f,  4.f, 14.f,  6.f,
						 3.f, 11.f,  1.f,  9.f,
						15.f,  7.f, 13.f,  5.f
					};
					// each face writes its OWN M x M atlas cell + its own 3 texcoords ->
					// disjoint outputs, safe to bake in parallel.
#ifdef TEXOPT_USE_OPENMP
					#pragma omp parallel for schedule(static)
#endif
					for (int i = 0; i < N; ++i) {
						const FIndex f = noViewFaces[i];
						const Face& face = faces[f];
						const int cx = (i % cellsPerRow) * M;
						const int cy = oldRows + (i / cellsPerRow) * M;
						const float x0 = cx + G, y0 = cy + G;
						const float x1 = cx + M - G, y1 = cy + G;
						const float x2 = cx + G, y2 = cy + M - G;
						const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
						const float invDen = (fabsf(denom) > 1e-6f) ? 1.f / denom : 0.f;
						const float R[3] = { vr[face[0]], vr[face[1]], vr[face[2]] };
						const float G[3] = { vg[face[0]], vg[face[1]], vg[face[2]] };
						const float B[3] = { vb[face[0]], vb[face[1]], vb[face[2]] };
						// quadratic edge controls (e0=v0v1, e1=v1v2, e2=v2v0)
						const size_t ei = (size_t)i * 3;
						const float ER[3] = { ecR[ei + 0], ecR[ei + 1], ecR[ei + 2] };
						const float EG[3] = { ecG[ei + 0], ecG[ei + 1], ecG[ei + 2] };
						const float EB[3] = { ecB[ei + 0], ecB[ei + 1], ecB[ei + 2] };
						for (int py = cy; py < cy + M; ++py)
							for (int px = cx; px < cx + M; ++px) {
								const float fx = px + 0.5f, fy = py + 0.5f;
								float w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * invDen;
								float w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * invDen;
								float w2 = 1.f - w0 - w1;
								if (w0 < 0) w0 = 0; if (w1 < 0) w1 = 0; if (w2 < 0) w2 = 0;
								const float sum = w0 + w1 + w2; if (sum <= 1e-6f) continue;
								const float iw = 1.f / sum;
								w0 *= iw; w1 *= iw; w2 *= iw;
								// quadratic Bezier triangle of the surface-propagated colour:
								//   C = c0 w0^2 + c1 w1^2 + c2 w2^2
								//     + 2 e01 w0 w1 + 2 e12 w1 w2 + 2 e20 w2 w0
								// Continuous in value AND cross-edge slope (edge controls are
								// shared between adjacent faces) -> no Gouraud facet banding.
								const float b00 = w0 * w0, b11 = w1 * w1, b22 = w2 * w2;
								const float b01 = 2.f * w0 * w1, b12 = 2.f * w1 * w2, b20 = 2.f * w2 * w0;
								const float rr = R[0]*b00 + R[1]*b11 + R[2]*b22 + ER[0]*b01 + ER[1]*b12 + ER[2]*b20;
								const float gg = G[0]*b00 + G[1]*b11 + G[2]*b22 + EG[0]*b01 + EG[1]*b12 + EG[2]*b20;
								const float bb = B[0]*b00 + B[1]*b11 + B[2]*b22 + EB[0]*b01 + EB[1]*b12 + EB[2]*b20;
								// sub-LSB ordered dither keyed to absolute atlas position so the
								// per-cell 8-bit rounding mismatch along shared edges is incoherent
								// (breaks the faint polygon outlines) rather than a clean seam.
								const float dth = (kBayer4[(py & 3) * 4 + (px & 3)] + 0.5f) * (1.f / 16.f) - 0.5f;
								int ir = (int)(rr + 0.5f + dth); ir = ir < 0 ? 0 : (ir > 255 ? 255 : ir);
								int ig = (int)(gg + 0.5f + dth); ig = ig < 0 ? 0 : (ig > 255 ? 255 : ig);
								int ib = (int)(bb + 0.5f + dth); ib = ib < 0 ? 0 : (ib > 255 ? 255 : ib);
								newTex.at<cv::Vec3b>(py, px) = cv::Vec3b((uchar)ib, (uchar)ig, (uchar)ir);
							}
						TexCoord* texcoords = faceTexcoords.data() + f * 3;
						texcoords[0] = TexCoord(x0, y0);
						texcoords[1] = TexCoord(x1, y1);
						texcoords[2] = TexCoord(x2, y2);
					}
					textureDiffuse = newTex;
					DEBUG("Data-colored %d unobserved faces (surface-propagated quadratic patch, %d atlas rows, %d smooth iters)", N, extraRows, (int)TEXTURE_DATACOLOR_SMOOTH_ITERS);
#endif // TEXTURE_DATACOLOR_COMPONENT_BAKE (per-face cell fallback)
				}
			}
		}
#endif // TEXTURE_DATACOLOR_UNOBSERVED && TEXTURE_DATACOLOR_BAKE

		LogPeakMem("GenTex: after data-color bake");
		TEX_PROFILE_BEGIN(_tGtFinalize);
		// 4) Enforce max size BEFORE allocating huge texture
		// (nMaxTextureSize == 0 means UNBOUNDED -> skip the cap entirely; < 0 was
		// already resolved to GL_MAX_TEXTURE_SIZE at the top of GenerateTexture.)
#if TEXTURE_FINAL_RESIZE
		if (nMaxTextureSize > 0 && std::max(textureDiffuse.cols, textureDiffuse.rows) > nMaxTextureSize) {
			double scale = double(nMaxTextureSize) /
				double(std::max(textureDiffuse.cols, textureDiffuse.rows));

			cv::resize(textureDiffuse, textureDiffuse,
				cv::Size(),
				scale, scale,
				cv::INTER_AREA);

			for (TexCoord& uv : faceTexcoords) {
				uv.x *= float(scale);
				uv.y *= float(scale);
			}
		}
#endif // TEXTURE_FINAL_RESIZE

#if TEXTURE_ENABLE_SHARPEN
		// Optional sharpening (now safe, atlas is final size)
		if (fSharpnessWeight > 0.0f) {
			cv::Mat small;
			cv::resize(textureDiffuse, small,
				cv::Size(),
				0.25, 0.25,
				cv::INTER_AREA);

			cv::Mat blurredSmall;
			cv::GaussianBlur(small, blurredSmall, cv::Size(), 1.5);

			cv::addWeighted(
				small,
				1.0 + fSharpnessWeight,
				blurredSmall,
				-fSharpnessWeight,
				0.0,
				small
			);

			cv::resize(
				small,
				textureDiffuse,
				textureDiffuse.size(),
				0, 0,
				cv::INTER_LINEAR
			);
		}
#endif // TEXTURE_ENABLE_SHARPEN
		TEX_PROFILE_END(_tGtFinalize, "GenerateTexture: resize+sharpen");
	}
}

// texture mesh
//  - minCommonCameras: generate texture patches using virtual faces composed of coplanar triangles sharing at least this number of views (0 - disabled, 3 - good value)
//  - fSharpnessWeight: sharpness weight to be applied on the texture (0 - disabled, 0.5 - good value)
bool Scene::TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness,
	bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight,
	int nMaxTextureSize, const IIndexArr& views)
{
	TEX_PROFILE_SCOPE("TextureMesh (grand total)");
	MeshTexture texture(*this, nResolutionLevel, nMinResolution);
	LogPeakMem("TextureMesh: start");

	MeshTexture::LabelArr labels;

	// assign the best view to each face
	{
		TD_TIMER_STARTD();
		if (!texture.FaceViewSelection(labels, minCommonCameras, fOutlierThreshold, fRatioDataSmoothness, views))
			return false;
		DEBUG_EXTRA("Assigning the best view to each face completed: %u faces (%s)", mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
	LogPeakMem("after FaceViewSelection");

	// generate the texture image and atlas
	{
		TD_TIMER_STARTD();
		texture.GenerateTexture(bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty, fSharpnessWeight, nMaxTextureSize);
		DEBUG_EXTRA("Generating texture atlas and image completed: %u patches, %u image size (%s)", texture.texturePatches.GetSize(), mesh.textureDiffuse.width(), TD_TIMER_GET_FMT().c_str());
	}
	LogPeakMem("after GenerateTexture");

	return true;
} // TextureMesh
/*----------------------------------------------------------------*/
