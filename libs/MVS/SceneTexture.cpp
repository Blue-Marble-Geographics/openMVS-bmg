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

// [MEM] per-stage peak-memory probe (Windows). PeakWorkingSetSize is monotonic, so
// the FIRST stage boundary where it jumps owns the peak. Used to target the 32 GB
// fit. Rides TEXTURE_DIAG: the OS query itself is skipped when the gate is closed,
// so a normal run pays nothing for the probe points left in place.
#ifdef _WIN32
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
static void LogPeakMem(const char* stage) {
	if (!TEXTURE_DIAG_ENABLED())
		return;
	PROCESS_MEMORY_COUNTERS pmc = {};
	pmc.cb = sizeof(pmc);
	if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
		TEXTURE_DIAG("[MEM] %-26s peakWS=%.2f GB  curWS=%.2f GB", stage,
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
// Output goes through TEXTURE_DIAG with a [PROFILE] tag so it sits next to the
// [MEM] probes and stays silent unless that gate is open (the clock reads are
// per-stage/per-view, i.e. free, so they are left compiled in).
//   - TEX_PROFILE_SCOPE("name")   : times the enclosing { } scope via RAII.
//   - TEX_PROFILE_BEGIN(tok)      : start a manual timer named by token `tok`.
//   - TEX_PROFILE_END(tok,"name") : stop that timer and log elapsed ms.
// NOTE: place SCOPE/BEGIN/END only on the MAIN thread (at stage granularity),
// never inside an omp parallel-for body, or the log will be spammed per-item.
// ============================================================================
#ifndef TEXTURE_PROFILE
#define TEXTURE_PROFILE 0
#endif

#if TEXTURE_PROFILE
#include <chrono>
#include <atomic>
namespace { namespace texprof {
	using Clock = std::chrono::steady_clock;
	static inline void Log(const char* name, Clock::time_point t0) {
		if (!TEXTURE_DIAG_ENABLED())
			return;
		const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
		TEXTURE_DIAG("[PROFILE] %-44s %10.2f ms", name, ms);
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

// TEXTURE_DELETE_BOUNDARY_UNOBSERVED: delete the unobserved regions that touch the
// mesh BORDER (the Poisson skirt over water at the survey edge) instead of
// data-colouring them, while keeping every ENCLOSED unobserved region (textureless
// roofs, interior water, small tears) which is what the data-colour fill is for.
//
// Camera visibility is the only signal that separates these two. VERIFIED on a
// 1940-view aerial river survey: the skirt sits on real (spurious) dense-matching
// points, so SurfaceTrimmer sees ordinary density (--poisson-trim cannot reach it
// without eating real sparse terrain) and the distance-to-cloud cull measures
// ordinary proximity (it removed ZERO faces). But no camera validates the skirt --
// the geometry is in the wrong place, so the depth test fails in every view, which
// is precisely why it lands in the no-view set rather than in a texture patch.
//
// Also recovers atlas budget: those faces no longer get data-colour rows appended
// (measured 2559 rows, ~15% of a 13599-px atlas, for 212,357 unobserved faces).
#ifndef TEXTURE_DELETE_BOUNDARY_UNOBSERVED
#define TEXTURE_DELETE_BOUNDARY_UNOBSERVED 0   // OFF deliberately, and this is an ARCHITECTURAL
                                               // change: this pass existed to delete the Poisson
                                               // skirt that ReconstructMesh was not removing. Now
                                               // POISSON_TRIM_HARD_OUTSIDE removes it upstream, so
                                               // re-carving here only replaces a smooth
                                               // footprint-following boundary with the ragged
                                               // per-face camera-visibility contour. Turning it off
                                               // also disables the rim peel, contour majority
                                               // filter and orphan filter, which live in its #if.
                                               // (was 1)
#endif
// How much synthetic edging to keep, as a MULTIPLE OF THE MEDIAN EDGE LENGTH measured
// inward from the last observed surface.
//
// Expressed in world units (via the median edge) rather than in face hops: hop count
// produces a band that is wide where triangles are large and narrow where they are
// small, with an edge that zigzags along the tessellation -- the "jaggedy" look. A
// metric band has uniform physical width regardless of triangulation. Scaling by the
// median edge keeps it scene-scale invariant.
//
// 3.0 is deliberately modest: a hard cut at the data limit reads worse than a small
// dressed margin, but the first version (8 face hops, ~6-10 m) was far too much.
// 0 = cut flush.
//
// REVERTED 1.0 -> 3.0. Lowering it was a mistake and made the reported artifact WORSE, for a
// reason worth recording: this is not a cosmetic band, it is a GEODESIC DILATION of the cut.
// keep = enclosed OR dist <= marginDist, where dist is distance-along-the-surface from the
// observed frontier -- and level sets of a distance field get SMOOTHER the further out they
// sit, because small features in the frontier are absorbed. So a WIDE margin hides a ragged
// visibility contour and a NARROW one exposes it. At 1.0 the cut hugged the raw contour and the
// fringe of fingers became more visible, not less.
//
// The fingers themselves are not this band. They are the shape of the camera-visibility cut
// over WATER, where the depth test succeeds patchily (glint, grazing incidence, a moving
// surface). PROVEN by elimination on the SchnellTests corridor: ReconstructMesh hands over a
// NEAR-CLOSED envelope --
//     DIAG decimate input: 1384601 live verts, 1264 border verts (0.1%), 1270 border edges
//     DIAG component sizes (top of 1): 2768196
// a 1,270-edge loop (~885 units of perimeter) on an 883-unit site, i.e. almost no boundary at
// all to be ragged -- and RefineMesh only subdivides (601,182 -> 621,750 verts, no deletion).
// The open, ragged silhouette is therefore created HERE, by the boundary-sheet delete
// (41,895 faces), not upstream. That is why every ReconstructMesh knob tried against it
// (SurfaceTrimmer trim, alpha-tighten, rim erode) failed to move it.
//
// WHY IT GOT WORSE WITHOUT THIS VALUE CHANGING: the width is `N * median edge`, so it is the
// fourth threshold in this pipeline expressed relative to MESH RESOLUTION rather than in world
// units -- alongside the Poisson trim density, TEXTURE_DATACOLOR_FEATHER_RINGS (face rings)
// and MESH_ALPHA_TIGHTEN_K_X100 (K * median edge). All four silently change meaning whenever
// the depth policy moves, which is why each needed hand-repair after the depth fix rather than
// simply working. The durable fix for all of them is to express the quantity in world units --
// here, a multiple of the point cloud's median NN spacing or the Poisson cell, both already
// computed upstream -- so this value stops depending on how finely the mesh happens to be
// tessellated. Until that is done, re-read the "within X-unit edge margin" figure in
// [TEX-SHEET] after any depth or decimation change.
//
// Also note this band interacts with ReconstructMesh's band refine, which subdivides the edge
// band two levels AFTER alpha-tighten -- so whatever fringe survives arrives here at ~4x the
// interior face density, which is what makes it read as wisps of polygons rather than a clean
// edge. Narrowing the margin treats the symptom; the ordering there is the other half.
// NOW 0 (flush cut), because the PREMISE CHANGED -- this is not a reversal of the revert above,
// it is a response to a different upstream mesh.
//
// The revert to 3.0 was right while ReconstructMesh handed over a near-closed envelope whose
// only boundary was the one THIS cut carved out of camera visibility: that contour is ragged
// (visibility over water is patchy), and the margin's geodesic dilation was genuinely smoothing
// it. Once POISSON_ADAPTIVE_TRIM was enabled, the geometry arrives with a clean, deliberate
// boundary derived from density + data footprint -- so there is nothing ragged left to dress,
// and the dressing became the only artifact.
//
// MEASURED, same scene, before vs after enabling the adaptive trim:
//     faces deleted here      44530 -> 33230   (11,300 FEWER removed)
//     margin faces kept        8778 -> 11675
//     ring-0 seeded by rim     2053 ->  6809   (3.3x -- the envelope is now open)
//     feather band            26604 -> 47387
// i.e. the trim opened the envelope, and this margin then retained 11,675 unobserved,
// data-coloured faces hanging off the newly-clean rim. Those are the "spikey under parts": the
// texture stage putting back a softened copy of what ReconstructMesh had just removed.
//
// 0 cuts flush to the observed surface, which is the correct behaviour once the geometric
// boundary is itself trustworthy. Raise it again ONLY if the upstream trim is disabled, since
// then the ragged-visibility problem returns and the dilation is worth its cost.
#ifndef TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES
#define TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES 0.0
#endif
// Majority-filter passes over face adjacency applied to the cut boundary. Removes the
// single-face spikes and notches a per-triangle threshold always leaves behind; a face
// in the interior of either region has all three neighbours agreeing and never moves,
// so this smooths the contour without shifting the band. 0 = raw threshold.
// RAISED 3 -> 8. This is the pass that owns the "choppy edge", and it became the load-bearing
// one when TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES went to 0: the margin's geodesic dilation used to
// smooth the contour implicitly (level sets of a distance field get smoother the further out they
// sit), and cutting flush removed that, exposing the raw per-face camera-visibility boundary.
// Visibility is decided per triangle, so that boundary is choppy by construction.
//
// A majority filter is the right instrument and 3 passes is simply too few: each pass can only
// remove a feature about ONE face wide, so a 6-face notch needs ~6 passes. It also cleans single
// and double-face spikes, which is the same artifact reported as "slivers" on this contour --
// so one change addresses both symptoms.
//
// Safe by construction, independent of the count: only a face whose neighbours mostly DISAGREE
// can flip, so a face in the interior of either region never moves and the band as a whole does
// not shift. It also cannot reach observed faces or interior fill.
//
// TUNE FROM THE NEW LOG LINE, not by eye: "contour majority filter: N passes, flips per pass:
// ..." should DECAY toward ~0. Converged (e.g. 900 400 120 40 10 3 1 0) means the contour is
// clean and extra passes are free but pointless. Still large on the last pass means it is eating
// into the boundary rather than removing spikes -- lower it.
#ifndef TEXTURE_UNOBSERVED_EDGE_SMOOTH_PASSES
#define TEXTURE_UNOBSERVED_EDGE_SMOOTH_PASSES 8
#endif
// Rim-peel passes over the boundary this cut creates. A face joined to the mesh by a
// single edge has TWO border edges -- a dangling sliver, which is what the leftover
// "spikes" along the cut are. Mesh::Clean's alpha-tighten removes exactly these but ran
// in ReconstructMesh, before this boundary existed, so nothing downstream tidies it.
//
// This is the one step that peels OBSERVED faces too: a two-border-edge triangle is
// degenerate boundary geometry whether or not a camera saw it. A normal rim face has
// ONE border edge and is never touched, so the real survey edge cannot be eroded --
// each pass only removes what is already dangling. 0 = disable.
#ifndef TEXTURE_UNOBSERVED_RIM_PEEL_PASSES
#define TEXTURE_UNOBSERVED_RIM_PEEL_PASSES 6
#endif
// Orphaned-component removal, applied to the boundary this stage's cuts create. A
// component of the SURVIVING surface is kept only if its face count is at least this
// fraction of the largest component's; smaller ones are floating islands and are
// deleted with the rest. Thousandths of a percent, same rule and same units as
// MESH_KEEP_COMPONENT_PCT_X1000 in Mesh.cpp, so the two stages behave identically.
// Higher removes more (and risks dropping a genuinely isolated real structure);
// 0 disables the filter.
//
// WHY THIS IS HERE AND NOT ONLY IN Mesh::Clean: every cut above is a PER-FACE
// decision (no camera, outside the margin, dangling sliver) and none of them consider
// connectivity, so when the deleted set was the only thing joining a patch of surface
// to the body, that patch is left floating. Mesh::Clean re-runs its component filter
// after each pass that can sever a neck (rim-erode, alpha-tighten); this stage severs
// necks harder than either and had no equivalent, so its islands reached the product.
//
// MEASURED, 115-view corridor survey: ReconstructMesh emitted ONE connected component
// (1,322,827 faces -- "DIAG component sizes (top of 1)") with no islands visible in
// the mesh, yet islands appeared after texturing. Marginal blobs were connected
// THROUGH the low-confidence boundary sheet, so deleting that sheet -- the very thing
// that gives this stage its clean edging -- cut them loose.
// RAISED from 100 (0.1%) to 2000 (2%) after the first field test: the filter cut the
// component count from 18 to 4, but the three survivors were each well above the 576-face
// floor that 0.1% produced on a 576,320-face body, and they were still visible as
// detached blobs in the render.
//
// A 2% floor (~11,500 faces there) is defensible on this class of data specifically: a
// single-site aerial survey is ONE contiguous surface, so anything this stage's cuts leave
// disconnected is junk essentially by definition and there is no legitimately separate
// structure for the threshold to destroy. That argument does NOT hold for a scene that is
// genuinely several separate objects -- lower it back toward 100 if that is ever the input.
// RAISED AGAIN, 2000 -> 4000, and this time the size distribution picked the value
// instead of a guess. MEASURED on the same scene with the distribution line above:
//
//   533056 | 16628 11586 9430 5605 | 1331 983 714 636 490 349 254 ...
//
// There is a clear SLAB tier of four components (16.6k down to 5.6k) and then a ~4x drop
// to the speck tier. At 2% the floor landed at 10,661 -- mid-slab -- so the two largest
// slabs survived and were still visible as islands. 4% puts the floor at ~21.3k, above
// the whole slab tier and inside the real gap, so the cut is not arbitrary.
//
// Prefer reading the distribution to raising this blind: if a future scene shows no gap
// (a smooth ramp from the body down), then no floor is the right tool and the fix belongs
// upstream, exactly as the distance-cull attempt in SceneReconstruct.cpp concluded.
// LOWERED 4000 -> 1000, provisionally. 4000 (a ~21,000-face floor) was chosen from a
// component distribution measured while TEXTURE_DATACOLOR_MIN_VIEW_COS was 0.35 and
// fragmenting the surface; that condition is gone, so the distribution it was fitted to no
// longer describes the input. Re-read the "component sizes (top of N)" line above and pick
// a value that lands in a real GAP before raising it again.
//
// Erring low on purpose: an island that survives is visible and fixable, whereas a
// legitimate structure deleted by an over-large floor is a silent loss of deliverable.
#ifndef TEXTURE_ORPHAN_COMPONENT_PCT_X1000
#define TEXTURE_ORPHAN_COMPONENT_PCT_X1000 1000
#endif

// STOP TUNING THE PERCENTAGE -- when the INPUT mesh was a single connected component,
// the percentage is the wrong instrument entirely and this flag replaces it.
//
// The argument is structural, not statistical. Every cut in this stage is a per-face
// decision on a mesh that arrived as ONE surface, so any component this stage ends up
// with beyond the largest was manufactured HERE by severing a neck. There is no
// legitimately-separate structure for a floor to protect, because there was no separate
// structure in the input. Keeping only the largest component is therefore exact, and it
// needs no scene-specific number.
//
// MEASURED, 117-view OKState corridor (2026-08-18 06:48 run), which is what forced this:
//   ReconstructMesh -> RefineMesh emitted 681,390 faces in ONE component (verified by
//   flooding scene_0_dense_mesh_refine.ply directly: "CONNECTED COMPONENTS: 1").
//   This stage's cuts then produced 24:
//     [TEX-SHEET] component sizes (top of 24): 555377 21224 6014 3325 2361 1652 ...
//   The 1% floor (5,553 faces) dropped 21 of them and KEPT 21,224 and 6,014, which are
//   the two islands visible in the render -- the 21,224 one centred at X=+230.7, the
//   right-hand blob. It is 3.822% of the body, so every floor tried so far (0.1%, 1%,
//   2%) passes it, and 4% would clear it by only 4.5% margin on this scene alone.
//   Below the body the distribution is a smooth ramp (26x gap to 21k, then 3.5x, 1.8x,
//   1.4x...), which is exactly the "no clear gap -> a floor is not the right tool"
//   condition this file already warns about.
//
// The percentage path is kept for the case the premise does not hold: if the input was
// genuinely several objects, the largest-only rule would delete the smaller ones, so we
// fall back to the floor. The input component count is measured, not assumed.
//
// Set to 0 to disable and use TEXTURE_ORPHAN_COMPONENT_PCT_X1000 unconditionally.
#ifndef TEXTURE_ORPHAN_KEEP_LARGEST_IF_INPUT_SINGLE
#define TEXTURE_ORPHAN_KEEP_LARGEST_IF_INPUT_SINGLE 1
#endif
// Minimum border-loop bounding-box diagonal, as a fraction of the whole-mesh diagonal,
// for that loop to count as the OUTER silhouette rather than an interior hole.
//
// Border edges form closed loops: one huge one around the survey outline and potentially
// thousands of tiny ones around canopy gaps. Only components reaching an OUTER loop are
// candidates for boundary-sheet deletion -- treating every hole rim as "the boundary"
// deleted unobserved patches beside canopy gaps that used to data-colour correctly, and
// punched visible holes through vegetation. Same shape of rule as
// MESH_HOLE_MAX_DIAG_FRAC_X1000 in Mesh.cpp, which separates real edge from interior hole
// the same way.
#ifndef TEXTURE_OUTER_LOOP_MIN_DIAG_FRAC
#define TEXTURE_OUTER_LOOP_MIN_DIAG_FRAC 0.10
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
//
// SET TO 0: at 0.20 this was culling real roof/wall faces, not just the rim. DIAG_TINT
// showed the white edges are the feather band (GREEN) wrapped around genuine no-view
// fill cores (MAGENTA) sitting mid-roof and mid-wall -- i.e. those faces reached the
// data-colour path because this gate threw away EVERY one of their observations. Two
// reasons it lands on real surface: the threshold is compared against the per-face
// normal from scene.mesh.faceNormals, whose noise on a Poisson mesh is comparable to
// the gap between 0.20 and a genuinely grazing view (so neighbouring faces flip across
// it -> blobby, not banded); and geometry under eaves / on dormer cheeks legitimately
// has no better-than-78deg view in any camera.
//
// Note this gate fights TEXTURE_RASTER_NO_BACKFACE_CULL, added one commit earlier to
// cure the SAME symptom ("the magenta wall") by letting the depth buffer decide
// visibility instead of the normal sign. That comment promised the paired rawCos reject
// would become "a floored-quality last resort"; the code below is still a hard
// `continue`, so the demotion came back through a different door. 0 is documented as
// bit-identical to the pre-wiring behaviour (see the ternary at the use site), which
// makes this a clean revert rather than a new tuning value.
#ifndef TEXTURE_DATACOLOR_MIN_VIEW_COS
#define TEXTURE_DATACOLOR_MIN_VIEW_COS 0.f
#endif

// TEXTURE_DATACOLOR_SEED_MIN_COS: quality gate on which OBSERVED boundary faces are
// allowed to SEED the synthesized fill colour. The fill diffuses outward from the
// real-texture faces bordering a no-view region; on a ROUGH mesh that region shatters
// into many small islands, each seeded from whatever grazing sliver happens to border
// it, and grazing faces sample noisy/shadowed real texture -> every island converges to
// a slightly different flat colour = the blotchy CPU fill. Only faces whose OWN view
// (their patch label camera) sees them more frontally than this cosine are kept as seed
// donors; grazing donors are skipped, so islands seed from clean frontal texture (or, if
// an island has no frontal boundary, fall back to the uniform global average instead of a
// noisy patchwork). cos is a true cosine (1 = frontal, 0 = edge-on). 0.35 ~= reject
// donors more grazing than ~70deg; raise to demand more frontal seeds; 0 disables the
// gate (seed from every observed boundary face, previous behaviour).
// REVERTED to 0 (field test, Aug 2026): the gate made the fill more UNIFORM but the
// fuller (ungated) seeding colours the region more accurately; the real remaining
// artifact is highlight over-brightening, handled by TEXTURE_DATACOLOR_SEED_HIGHLIGHT_CLAMP.
#ifndef TEXTURE_DATACOLOR_SEED_MIN_COS
#define TEXTURE_DATACOLOR_SEED_MIN_COS 0.0f
#endif

// TEXTURE_DATACOLOR_SEED_HIGHLIGHT_CLAMP: caps each fill seed colour's luminance to
// its OWN connected fill component's median_seed_luma * this factor. A specular highlight
// caught on a boundary donor face seeds an abnormally BRIGHT colour that the two-phase
// inpaint then spreads inward, over-brightening the fill and smearing it wider than the
// GPU (whose smoother mesh boundary often misses the highlight). Clamping the seed luma to
// a robust per-region ceiling (RGB scaled uniformly so hue is preserved) stops highlights
// from dominating without dimming normal-brightness seeds -- and, because the ceiling is
// per connected component, a genuinely-bright fill region is measured against its own
// neighbourhood, not dragged down by a dark region's median elsewhere. 1.6 ~= allow up to
// 60% above the component median before clamping; lower to clamp harder, raise to allow
// brighter seeds; 0 disables (no clamp, previous behaviour).
#ifndef TEXTURE_DATACOLOR_SEED_HIGHLIGHT_CLAMP
#define TEXTURE_DATACOLOR_SEED_HIGHLIGHT_CLAMP 1.6f
#endif

// TEXTURE_RASTER_NO_BACKFACE_CULL: match the GPU ProjectMesh (which does NOT
// back-face cull). The stock texture rasterizer (RasterizeTriangleBary CULL=true)
// drops any triangle whose projected winding is negative; a near-vertical/grazing
// wall projects to ~0 signed area, so tiny per-face normal noise flips it negative
// and it is culled in EVERY view -> demoted to NO_ID / synthesized fill (the
// magenta wall). With this on, RasterMesh rasterizes both windings and the depth
// buffer decides visibility; the paired rawCos hard-reject in ListCameraFaces is
// relaxed to a floored-quality last resort so these depth-visible faces keep their
// real (grazing) texture instead of the fill. 0 = stock cull behaviour.
// NEGATIVE RESULT (2026-08-19) -- do not set this to 0 again without new evidence.
// Tested at 0 to reproduce the visibility pipeline of 4669f06c (2026-07-26), the last
// revision reported to render this scene sharp. The result was MUCH WORSE: markedly more
// pale bleeding across the walls and eaves, and the porch roof smeared badly. So the
// grazing-wall loss this switch's comment describes is the dominant effect on this data --
// with the stock cull ON, walls whose projected winding flips are culled in EVERY view,
// land in the no-view set, and get fill + feather. Keeping the cull off and letting the
// depth buffer decide visibility is load-bearing here, not a refinement.
//
// The competing theory it was meant to test -- that back-facing Poisson flaps under the
// eaves write depth and occlude real surface -- is NOT supported: if it dominated, turning
// the cull on would have improved those areas. It did the reverse.
//
// Corollary worth keeping: the 7/26 "was sharp" anchor cannot be reproduced by reverting
// THIS file's visibility switches, which points the search at the mesh instead (a1d7d613
// reworked the mesh resolution/memory policy, and a different mesh gives a different
// no-view set no matter what texturing does).
#ifndef TEXTURE_RASTER_NO_BACKFACE_CULL
#define TEXTURE_RASTER_NO_BACKFACE_CULL 1
#endif

// TEXTURE_FAST_RASTER: rasterize each view's faceMap/depthMap with the same
// inlined, incremental-barycentric walk MeshRefine::ProjectMesh uses, instead of
// TRasterMesh::Project + TImage::RasterizeTriangleBary. Output semantics are
// unchanged (see RasterizeCameraFacesFast); what goes away is the redundant
// double-precision per-face vertex transform, the per-pixel edge-function
// re-evaluation over the whole bounding box, and the per-pixel PARSER callback.
// 0 = previous TRasterMesh path (kept for A/B).
// PARTIAL NEGATIVE RESULT (2026-08-19): tested at 0 (stock TRasterMesh path, as on 7/26)
// TOGETHER WITH TEXTURE_RASTER_NO_BACKFACE_CULL 0, and the combination was much worse. That
// test does NOT exonerate this rasterizer on its own -- the two were changed together, and
// the cull switch alone is sufficient to explain the regression (see its NEGATIVE RESULT
// note). This one remains UNVERIFIED against the stock path: it is uncommitted work that
// decides which face wins each pixel, hence the no-view set, and its header's claim that
// output semantics are unchanged has never been checked on real data.
//
// To test it cleanly, set THIS to 0 while leaving NO_BACKFACE_CULL at 1 -- note that pairing
// needs the stock RasterMesh::Project override, which is already conditioned on the cull
// switch, so the combination is meaningful. Restored to 1 meanwhile so the working
// configuration is the fast path.
#ifndef TEXTURE_FAST_RASTER
#define TEXTURE_FAST_RASTER 1
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
//
// LOWERED 14 -> 4. RINGS is the knob that controls how FAR the fill colour reaches into
// real texture; STRENGTH controls whether a SEAM appears at the boundary. Those are not
// interchangeable, and the reported defect is reach, not a seam:
//   * At alpha 1.0 the ring-0 texel is 100% fill colour, which is the entire reason the
//     shared mesh edge has no discontinuity -- both sides evaluate the same value there
//     (see the CONTINUITY BLEND note at the blend site). Lowering STRENGTH would buy a
//     dimmer halo by re-introducing exactly the step this pass exists to remove.
//   * RINGS costs nothing at the boundary. Cutting it leaves ring 0 untouched (still
//     alpha 1, still seamless) and simply ends the ramp sooner, so the pale wash stops
//     ~4 faces in instead of ~14.
// 14 was chosen to dissolve into a fill baked at DENSITY_SCALE 0.35 -- i.e. deliberately
// blurrier than its surroundings -- so a wide band was needed to match frequency. That
// reasoning still holds; it just does not justify 14 rings once the fill regions
// themselves are small. The file's own guidance says 4-8 is the reasonable range.
//
// If a crisp inner edge ever becomes visible where real detail returns, raise this (6-8)
// rather than touching STRENGTH.
//
// KNOWN UNIT PROBLEM, not yet fixed: this is a BFS depth over FACE ADJACENCY, so its
// physical width is K triangles, whatever those happen to measure. The geometry chain runs
// at --resolution-level 2 while texturing runs at 0, so the mesh is coarse relative to the
// atlas and each ring is wide -- wider still wherever the local triangles are large, which
// is why the band does not read as a uniform-width border.
//
// This is the same mistake TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES already corrected for the
// boundary margin cut: "hop count produces a band that is wide where triangles are large
// and narrow where they are small ... A metric band has uniform physical width regardless
// of triangulation." The real fix is to ramp alpha on the per-vertex GEODESIC DISTANCE from
// the boundary (scaled by the median edge, as the margin cut does) instead of on vRing.
// Deliberately NOT done yet: shrinking the band only hides a feather that should not be
// firing on that surface at all -- see the covered[f] note at the boundary-sheet cut.
//
// BACK TO 4. Briefly restored to 14 for the 7/26 bracketing test; that test came back worse
// for an unrelated reason (see the NEGATIVE RESULT at TEXTURE_RASTER_NO_BACKFACE_CULL), so 14
// is not vindicated by it. 4 remains the best value measured on this data: at 14 the band
// visibly washed the walls out along the eaves, at 4 it was "much better" with no crisp inner
// edge reported.
//
// Still masking rather than fixing, and worth remembering as such: the band only appears where
// a face borders the no-view set, so every ring of it is a symptom. The fix is to stop those
// eave/roof faces being unobserved in the first place.
#ifndef TEXTURE_DATACOLOR_FEATHER_RINGS
#define TEXTURE_DATACOLOR_FEATHER_RINGS 4
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
//
// DEAD MACRO -- nothing reads it. The box-blur smear this described was replaced by the
// CONTINUITY BLEND (blend observed toward the fill's own pinned vr/vg/vb field), which
// has no blur radius: the smoothness comes from the fill field being smooth, not from
// averaging the observed atlas. Its value has no effect at any setting; the live levers
// are RINGS (how far the band reaches) and STRENGTH (alpha at the boundary). Left in
// place only so the two paragraphs above still document what the old smear did.
#ifndef TEXTURE_DATACOLOR_FEATHER_BLUR_PX
#define TEXTURE_DATACOLOR_FEATHER_BLUR_PX 18
#endif

// TEXTURE_FEATHER_CUT_SILHOUETTE: whether the silhouette created by THIS stage's own cuts
// (boundary-sheet margin cut, rim peel, orphan-component filter) is treated as a feather
// boundary, by clearing covered[f] on every face queued for deletion.
//
// 1 = clear it (the behaviour added alongside the coverage gutter). The feather seeds ring 0
//     wherever a neighbour is NO_ID or not covered, so clearing the flag makes each surviving
//     neighbour of a deleted face a ring-0 boundary face at alpha 1 -- i.e. its texture is
//     replaced outright by the fill colour field and smeared K rings inward.
// 0 = leave it set. The cut leaves a crisp edge, and the surviving faces keep their real
//     observed texture.
//
// DEFAULTS TO 0, and the reason is that the justification for 1 does not survive this
// configuration. Three reasons were given for clearing the flag; two are dead code here:
//   * the fill tiles' DETAIL donor search reads covered -- but TEXTURE_DATACOLOR_DETAIL_GAIN
//     is 0.0f, so no detail is transplanted at all;
//   * the MIRROR donor search reads covered -- but TEXTURE_DATACOLOR_MIRROR_GAIN is 0.0f,
//     likewise inert.
// So in this build the ONLY live effect is the third one: feathering the cut silhouette. And
// that is a net loss on this data -- these cuts run along the mesh rim (eaves, roof edges,
// the survey boundary), which is real, well-observed, sharp texture. Flattening a K-ring band
// of it to a diffused fill colour to soften an edge that the renderer clips anyway trades
// detail people can see for a seam they cannot. A crisp edge at a deliberate cut is the
// correct output, not a defect.
//
// Set to 1 only if the detail/mirror gains are re-enabled, or if a hard rim edge is ever
// judged worse than losing the band -- and note the band width is in FACE RINGS, so how much
// texture it costs depends on the mesh resolution (see TEXTURE_DATACOLOR_FEATHER_RINGS).
#ifndef TEXTURE_FEATHER_CUT_SILHOUETTE
#define TEXTURE_FEATHER_CUT_SILHOUETTE 0
#endif

// TEXTURE_FEATHER_SKIP_HIDDEN_FILL: don't let HIDDEN geometry seed the seam feather.
//
// THE PROBLEM THIS FIXES, measured on the Niwot house (496,114 faces):
//   [DATACOLOR-DIAG] noView=29942 nComp=479 band=193381
//   [SEAM-OUTSET]    feather band 193381 faces -> 61762467 texels written
//   Data-colored 29942 unobserved faces in 479 component tiles (351 atlas rows = 0.022)
// The synthesized fill occupies 2.2% of the atlas, yet the feather it triggers rewrote
// 61.8M of 266.9M texels -- 23% of the atlas -- and pulled 193,381 faces, 39% OF THE WHOLE
// MESH, into the band. The multiplier is the COMPONENT COUNT: 29,942 no-view faces are
// scattered across 479 separate blobs averaging 62 faces each, and every blob grows its own
// K-ring band, dragging ~404 real faces in apiece. At that scale the feather stops being a
// boundary treatment and becomes a mesh-wide wash over good texture.
//
// Lowering TEXTURE_DATACOLOR_FEATHER_RINGS cannot fix it. Ring 0 is alpha 1.0 at EVERY K
// (t = (K-1-vRing)/(K-1) is 1 when vRing is 0), by design, because that is what makes the
// shared edge seamless -- so the faces immediately around all 479 blobs are replaced
// outright no matter how short the ramp is.
//
// The real distinction is WHY a face has no view, which [TEX-VIEWCLASS] measures:
//   back-winding-cull = 17670  -- never projected front-wound in any view
//   occluded/subpix   =  9845  -- projected front-wound, never won a pixel
// The first group is inverted/interior geometry: flaps sitting behind the real surface,
// which lose the depth test in every view and are therefore INVISIBLE in the render.
// Softening the transition into something nobody can see buys nothing and costs a full
// alpha-1 ring plus K-1 rings of ramp on the real texture wrapped around it.
//
// So with this on, a no-view neighbour only seeds ring 0 if everFrontWound says a camera
// could have seen it head-on. Genuine textureless surface, interior holes and tears still
// feather exactly as before; hidden flaps get a crisp edge nobody will ever look at.
//
// NOTE this changes only what FEATHERS. The faces are still kept and still filled -- see
// the deletion option discussed at TEXTURE_DELETE_BOUNDARY_UNOBSERVED, which would remove
// them from the mesh outright and is the stronger (but non-reversible) form of this fix.
// An OPEN mesh edge (faceFaces == NO_ID) always seeds, regardless of this switch: that is
// the true silhouette, not a fill region.
// 0 = previous behaviour (every no-view neighbour seeds the band).
#ifndef TEXTURE_FEATHER_SKIP_HIDDEN_FILL
#define TEXTURE_FEATHER_SKIP_HIDDEN_FILL 1
#endif

// TEXTURE_DELETE_HIDDEN_UNOBSERVED: delete unobserved faces that are hidden GEOMETRY, instead of
// data-colouring them as if they were textureless surface.
//
// The keep/delete rule below classifies unobserved regions by TOPOLOGY: ENCLOSED (ringed by
// observed surface) is assumed to be a textureless roof, interior water or a small tear and is
// kept and filled; only regions reaching the outer border loop are candidates for deletion. That
// assumption breaks on a Poisson mesh, because Poisson returns a near-CLOSED envelope --
// MEASURED on SchnellTests, `DIAG decimate input: ... 1562 border verts (0.1%)` even after the
// adaptive trim. The envelope therefore has a whole inward-facing UNDERSIDE, most of it not
// reachable from the one qualifying outer loop, so it classifies as "enclosed" and gets dressed
// in synthesized colour. That is what shows up as spikes hanging off the silhouette in the
// textured render while the untextured mesh looks clean -- the geometry was always there; only
// texturing made it opaque.
//
// The distinguishing signal already exists: everFrontWound (built in ListCameraFaces for
// TEXTURE_FEATHER_SKIP_HIDDEN_FILL).
//   everFrontWound = 1 -> some camera could see this face HEAD-ON. If it is still unobserved it
//                         was occluded or too poorly seen to win a pixel: genuine surface, fill it.
//   everFrontWound = 0 -> never front-wound in ANY view, i.e. inward-facing. Invisible in every
//                         render. Delete it; there is nothing for a fill colour to represent.
// A flat roof or a lake inside the survey faces the cameras and is front-wound, so neither is
// affected. MEASURED share of the unobserved set that is back-wound only:
// `[TEX-VIEWCLASS] ... back-winding-cull=36844` of 72,416 unobserved, i.e. about half.
//
// NOTE this is a stronger action than TEXTURE_FEATHER_SKIP_HIDDEN_FILL, which used the same
// signal only to stop such faces SEEDING the feather. Here they leave the mesh. That is the
// intent -- they are invisible -- but it is a geometry deletion, so it is gated separately and
// logged ("hidden-geometry filter: N of M ... dropped instead of filled"). 0 restores the old
// keep-everything-enclosed behaviour.
// NEGATIVE RESULT (2026-08-19) -- do not re-enable without a DIFFERENT signal. Winding is not a
// sound test for "inward-facing" on this data, and the reason is already documented one macro
// group away, at TEXTURE_RASTER_NO_BACKFACE_CULL: "a near-vertical/grazing wall projects to ~0
// signed area, so tiny per-face normal noise flips it negative and it is culled in EVERY view".
// Quarry faces, steep slopes and vegetation are exactly that -- genuinely visible surface that
// happens to be back-wound everywhere. Deleting it re-creates the "magenta wall" artifact that
// enabling NO_BACKFACE_CULL was introduced to cure.
//
// MEASURED, in two steps, which is what makes the diagnosis solid rather than a guess:
//   v1 (everFrontWound only): dropped 63,662 of 71,223 unobserved -- exactly back-winding 42,935
//      PLUS off-frustum 20,725, i.e. it also deleted faces outside every frustum that were never
//      TESTED. Scattered holes through real terrain.
//   v2 (+ everProjected, so "never tested" fails safe to keep): dropped 42,937 -- precisely the
//      back-winding bucket, the conflation fixed. keep-enclosed stayed at 5,123 (from 22,927),
//      so the ~17,800 removed really were back-wound-only -- AND THE HOLES REMAINED. That is the
//      proof: back-wound-only is not the same as invisible.
//
// The underside problem this was aimed at is therefore still open, but it needs a signal that
// does not rely on projected winding. Depth-buffer occlusion is too small a bucket here
// (occluded/subpix was only 7,563 of 71,223), and a geometric "below the observed surface at the
// same XY" test is the MESH_DOWN_CULL family, which is recorded as removing building walls.
// 0 = keep the topological enclosed/boundary rule alone.
#ifndef TEXTURE_DELETE_HIDDEN_UNOBSERVED
#define TEXTURE_DELETE_HIDDEN_UNOBSERVED 0
#endif

// TEXTURE_FILL_MIN_COMPONENT_FACES: minimum face count for an ENCLOSED unobserved region to be
// kept and data-coloured. Smaller ones are chopped.
//
// The keep rule classifies unobserved regions purely by TOPOLOGY: anything not reachable from a
// qualifying outer border loop counts as "enclosed" and is assumed to be a textureless roof,
// interior water or a small tear worth filling. That is right for a lake and wrong for a tongue,
// and topology cannot tell them apart -- but SIZE can. A lake is thousands of faces; the synthetic
// clutter along the edge is specks and slivers.
//
// This is the "just chop them off" rule. It is deliberately NOT a visibility test: the previous
// attempt at one (TEXTURE_DELETE_HIDDEN_UNOBSERVED, using projected winding) deleted near-vertical
// REAL surface, because a grazing wall projects to ~0 signed area and flips winding on noise -- see
// its NEGATIVE RESULT note. Region size carries no such ambiguity.
//
// 256 is PROVISIONAL. The companion log line prints the full component-size distribution, so pick
// the value from a real GAP between the big regions and the speck tier rather than trusting this
// default -- the same discipline the orphan filter's own comment insists on ("Prefer reading the
// distribution to raising this blind"). If the tongues turn out to be large connected regions,
// size will not separate them either and this is the wrong instrument.
// 0 disables (keep every enclosed region, previous behaviour).
// NEGATIVE RESULT (2026-08-19) -- tested at 256 and DISABLED. It did not reduce the edge clutter
// and it punched holes: DELETE went 45,158 -> 64,562 faces, close to the 69,096 that produced
// scattered holes when TEXTURE_DELETE_HIDDEN_UNOBSERVED was briefly on.
//
// The companion distribution line is why, and it was decisive: 1279 components sized
//     10391 3990 2444 2268 2263 2228 2217 1951 1596 1315 1306 1286 ...
// a SMOOTH RAMP with no break. Any threshold on that is arbitrary -- it sweeps the tail while
// leaving every thousand-face region, and at that scale a water body and a tongue are the same
// size. This is exactly the condition the orphan filter's comment warns about: 'if a future scene
// shows no gap (a smooth ramp from the body down), then no floor is the right tool and the fix
// belongs upstream.'
//
// Keep the code and the distribution log -- reading that line is how you tell in one run whether
// a scene HAS a gap -- but do not re-enable without seeing one. Size is the third downstream
// criterion to fail here, after projected winding (deleted near-vertical real walls) and topology
// (calls tongues 'enclosed'). The separating signal is density at the data perimeter, upstream:
// see POISSON_TRIM_EDGE_MULT_X100 in SceneReconstruct.cpp.
#ifndef TEXTURE_FILL_MIN_COMPONENT_FACES
#define TEXTURE_FILL_MIN_COMPONENT_FACES 0
#endif

// TEXTURE_BOUNDARY_SMOOTH_*: Taubin-smooth the FINAL silhouette, after the boundary-sheet delete.
//
// WHY A NEW PASS IS NEEDED AT ALL. The visible outline of the deliverable is the silhouette of
// the OBSERVED region, decided per-triangle by camera visibility when the patches were built, and
// nothing reshapes it:
//   * the contour majority filter below only rewrites keepF for UNOBSERVED faces -- its own
//     comment says observed faces are never touched -- so it dresses the fringe AGAINST that
//     outline rather than straightening it. It also cannot straighten in principle: a face needs
//     MOST of its neighbours to disagree before it flips, so a one-face STAIRCASE is a stable
//     fixed point. Converged output measured as `1842 64 15 3 1 0` and the edge stayed choppy.
//   * Mesh::Clean's alpha-tighten, rim-erode and its own Phase-10 Taubin pass all run back in
//     ReconstructMesh, long before this cut exists -- the comment on the majority filter says so
//     explicitly. They polish a boundary that no longer bounds the visible surface.
//   * raising TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES does smooth it (a geodesic dilation's level
//     sets get smoother with distance) but pays in synthesized edging, which is the artifact that
//     margin was set to 0 to remove.
//
// So this is the same Taubin curve smoother as Mesh.cpp Phase 10, re-implemented against
// MVS::Mesh and run at the only point where the final boundary exists. For each boundary vertex
// with exactly TWO border-curve neighbours, move toward their midpoint by +lambda, then away by
// -mu; the alternating pair cancels the curve-shortening of plain Laplacian smoothing, so the
// outline gets smoother WITHOUT receding and no coverage is lost. Pinch/junction vertices (>2
// border neighbours) and every interior vertex are never moved, and no face is added or removed,
// so topology is untouched.
//
// COST, stated plainly: this moves vertices AFTER texturing. Texcoords are per-face and stay
// attached, so nothing is remapped, but the texture stretches very slightly right at the edge.
// At sub-edge-length movement that is invisible, and the atlas gutter already covers sampling
// past the triangles. Genuine sharp corners in the survey outline will also round off along with
// the noise -- ITERS is the knob for that.
#ifndef TEXTURE_BOUNDARY_SMOOTH_ENABLED
#define TEXTURE_BOUNDARY_SMOOTH_ENABLED 0   // OFF: it polished the visibility contour this stage
                                            // used to carve. With that delete disabled there is no
                                            // new boundary here to smooth. (was 1)
#endif
// lambda+mu pairs. More = smoother outline and more rounding of real corners.
#ifndef TEXTURE_BOUNDARY_SMOOTH_ITERS
#define TEXTURE_BOUNDARY_SMOOTH_ITERS 10   // 20 measured: mean 0.0858 -> 0.1084 (+26%) for 2x work, no visible gain -- Taubin is at its fixed point, so extra pairs only round real corners
#endif
// Taubin shrink step, x100. Matches Mesh.cpp Phase 10.
#ifndef TEXTURE_BOUNDARY_SMOOTH_LAMBDA_X100
#define TEXTURE_BOUNDARY_SMOOTH_LAMBDA_X100 50
#endif
// Taubin inflate step MAGNITUDE, x100; must exceed lambda or the curve still shrinks.
#ifndef TEXTURE_BOUNDARY_SMOOTH_MU_X100
#define TEXTURE_BOUNDARY_SMOOTH_MU_X100 53
#endif

// TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED: don't offer label 0 (UNDEFINED) to a face that has
// at least one candidate view. Stops the MRF from RECRUITING textureable faces into the
// synthesized fill.
//
// THE DEFECT. The two SmoothnessPottsStrong variants disagree on one clause:
//     #ifdef INCREASE_PATCHES   if (l1 == l2)                       return 0;
//     #else                     if (l1 == l2 && l1 != 0 && l2 != 0) return 0;
// The stock form deliberately EXCLUDES label 0 so two undefined neighbours pay the pair
// penalty like anyone else. The INCREASE_PATCHES form (the active one) lost that guard, so
// an undefined<->undefined edge is FREE. Undefined regions therefore become energetically
// self-reinforcing: once a blob exists, a bordering face that has a real view finds it
// cheaper to join the blob (0 smoothness) than to keep its label and pay the switch penalty
// at the blob edge.
//
// MEASURED on the Niwot house -- and note the file's own [LBP-DIAG] counter was added to
// catch precisely this ("undefined well above [the floor] means textureable faces are being
// pulled into undefined patches by the smoothness term"):
//     faces=496114 unobserved=24720   <- hard floor: faces with NO candidate view at all
//     iter=1  undefined=26998
//     iter=10 undefined=29737
//     iter=50 undefined=29984         <- monotonic climb, never recovers
// 29,984 - 24,720 = 5,264 faces (17.6% of the no-view set) had usable views and were
// discarded anyway. Each then costs twice: a data-colour fill tile, AND a seam-feather band
// smeared over the real texture around it.
//
// WHY THE FIX IS HERE AND NOT IN THE SMOOTHNESS TERM. Restoring the missing `l1 != 0` guard
// looks like the obvious one-line fix and is a silent no-op: SetPottsSmoothness(true) makes
// LBP.h sample the pairwise term ONCE per edge as fncSmoothCost(n1, n2, 0, 1) and then
// hardcode "0 when l1 == l2", so that branch is never evaluated during the solve. Turning the
// flag off is not an option either -- "0 == 0 costs W but A == A costs 0" is not a Potts
// model, and the general path re-evaluates fncSmoothCost L1*L2 times per directed edge per
// sweep (~5.8e9 normal dot-products on this scene). See the note above SmoothnessPottsStrong.
//
// Removing the label instead is exact and needs no tuning: a face with a candidate view
// simply cannot be assigned undefined, so `undefined` can only equal the genuine floor. It
// touches no energy value, so the Potts structure and its fast path are untouched.
//
// TRADE-OFF, stated plainly: a face whose only view is poor now gets that view's (possibly
// stretched, grazing) texture instead of synthesized fill. That is the same call already made
// by setting TEXTURE_DATACOLOR_MIN_VIEW_COS to 0, and it is the direction this scene wants --
// the areas in question were sharp in earlier builds. It does remove the MRF's ability to
// reject a lone outlier view; FaceOutlierDetection upstream is the mechanism for that, not
// label 0, which can only replace the face with blurred fill.
//
// WATCH ON THE NEXT RUN: `undefined=` should sit at the floor (24720) and stop climbing across
// sweeps, noView should fall by ~5k, and the feather band with it. gSmoothnessWeight (2000)
// was tuned WITH the defect present, so patch structure will shift -- check the patch count
// and finalEnergy, and if large flat areas fragment, gSmoothnessWeight is the knob, not this.
// 0 = previous behaviour (every face offered label 0).
#ifndef TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED
#define TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED 1
#endif

// TEXTURE_DATACOLOR_BAKE_OUTSET_PX: how far OUTSIDE each triangle, in atlas pixels, the
// fill bake and the seam feather keep writing.
//
// THE HAIRLINE THIS FIXES, and why the atlas gutter did not. Both bakes rasterise with
// a texel-CENTRE-inside test (the old `w < -0.01` barycentric epsilon, which on an ~11 px
// triangle is about a tenth of a pixel -- effectively "centre inside"). A texel that
// STRADDLES the triangle edge, centre just outside, is therefore never written. Inside a
// patch that is harmless: the neighbouring face owns that texel and writes it. But along
// the fill boundary the neighbour is a fill face living in a completely different part of
// the atlas, so nothing writes it -- it keeps its raw, un-feathered value. The renderer
// samples exactly there when it draws the shared mesh edge, and bilinear pulls roughly
// half its weight from that stale texel: a one-texel hairline tracing the entire
// synthetic/observed boundary.
//
// The coverage gutter (TEXTURE_ATLAS_COVERAGE_GUTTER) cannot reach it, by construction:
// coverage marks that texel as COVERED -- the triangle really does overlap it -- so it is
// deliberately preserved rather than cleared and reflooded. The two passes are solving
// different halves of the same problem. The gutter fixes texels no triangle touches; this
// fixes texels a triangle touches but the centre test skipped.
//
// So both bakes now write out to this many pixels past their edges, extrapolating by
// clamping the barycentrics to the edge (value AND alpha), which is exactly the colour the
// edge itself carries -> continuous, no step. The outset is converted per-triangle into a
// per-barycentric tolerance (outset*|opposite edge|/|2*area|), because a flat barycentric
// epsilon outsets a large triangle by many pixels and a small one by none.
//
// 1.5 covers a straddling texel plus its bilinear partner. Raise if a hairline persists at
// heavy minification (the mip chain reaches further than one texel); lower toward 0 to get
// the old centre-inside behaviour back. Values above ~2 start writing past the patch rect's
// own `border` margin and into whatever was packed next to it.
#ifndef TEXTURE_DATACOLOR_BAKE_OUTSET_PX
#define TEXTURE_DATACOLOR_BAKE_OUTSET_PX 1.5f
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
// ENABLED (numeric only -- DIAG does not alter a single output texel, unlike DIAG_TINT).
// The residual it reports is the one number that says whether the feather actually lands
// on the fill colour at the seam; every seam fix so far has been evaluated by eye. Set
// back to 0 once the seam is settled if the extra log line is unwanted.
// RE-ENABLED (numeric only; DIAG_TINT stays 0 so output texels are unchanged). With
// MIN_VIEW_COS at 0 the fill regions shrank but did not vanish, and the question left is
// WHY the survivors have no view -- which is precisely what the [TEX-VIEWCLASS] line and
// the TexViewClass.ply this switch produces answer, per face:
//   observed / angle-rejected / occluded-or-subpixel / back-winding-culled / off-frustum
// Guessing between those five is what the last three rounds did. Set back to 0 once the
// residual is understood; it costs one extra double-precision reprojection of every
// candidate face per view, plus the PLY write.
//
// BACK TO 0 (the "set back to 0" above, now that the trace has served its purpose). This
// is the one piece of texturing instrumentation that is NOT merely a log line, so it does
// not ride TEXTURE_DIAG: it adds a reprojection of every candidate face in every view,
// three per-face byte arrays, per-texel luminance stats inside the feather bake, and it
// writes TexViewClass.ply -- a debug-only artifact -- on EVERY run. All of that is in hot
// loops where a runtime branch is the wrong tool, so it stays a compile-time switch: set
// it to 1 (here or via -DTEXTURE_DATACOLOR_DIAG=1) to get the [TEX-VIEWCLASS] /
// [DATACOLOR-DIAG] lines and the PLY back. No output texel changes either way (that is
// DIAG_TINT, still 0).
#ifndef TEXTURE_DATACOLOR_DIAG
#define TEXTURE_DATACOLOR_DIAG 0
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
// TEXTURE_DATACOLOR_DENSITY_SCALE: fill tile density as a fraction of the REAL texture
// density it abuts. 1.0 matches the surrounding texture; lower is coarser and cheaper.
//
// These fills are water that does not reconstruct -- a smooth field from nearest-fill
// plus seam smoothing, carrying no detail at any resolution. Matching real-texture
// density is far more than they need, and the cost is not marginal: MEASURED, 22,027 fill
// tiles took 6,194 atlas rows, 27.9% of the atlas, on a scene whose real texture was
// already 3.4x under-resolved. Worse, appending those rows pushed the atlas from 16,007
// to 22,201 rows against a 16,384 cap, triggering a 0.738x downscale of EVERYTHING.
//
// 0.35 keeps the fill visibly smooth against its neighbours while returning most of that
// budget to terrain. Raise toward 1.0 if fill/real seams read as blocky.
#ifndef TEXTURE_DATACOLOR_DENSITY_SCALE
#define TEXTURE_DATACOLOR_DENSITY_SCALE 0.35
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
// (~77 B/px LIVE, of which the pseam:: vectors retain their largest-patch capacity).
// On large multi-MP patches this is the transient that can set the whole texturing
// peak on res-0. The cap limits workers so that the sum of the N LARGEST patch areas
// x ~96 B/px stays under the budget -- see bytesPerPixel at the cap site for why the
// bound is sum-of-top-N rather than N x largest, and why the constant is 96. Semantics-
// neutral: patches are seam-corrected independently, so fewer workers changes only
// parallelism, never the output pixels.
//   0 (DEFAULT) = AUTO: measure the free physical RAM at the moment the pass starts
//                 and spend TEXTURE_SEAM_MEM_FREE_FRACTION of it. A fixed budget has
//                 to be sized for the smallest machine, which throttles a big one to
//                 a handful of workers for no reason; auto scales the worker count up
//                 to every core on a box that has the headroom, and still clamps down
//                 on a box (or a run) that does not.
//  >0            = fixed budget of that many GB (previous behaviour, for repeatability).
//  <0            = cap disabled entirely (use every core, whatever the footprint).
#ifndef TEXTURE_SEAM_MEM_BUDGET_GB
#define TEXTURE_SEAM_MEM_BUDGET_GB 0
#endif

// AUTO-budget tuning (only used when TEXTURE_SEAM_MEM_BUDGET_GB == 0):
// TEXTURE_SEAM_MEM_FREE_FRACTION: share of the CURRENTLY-free physical RAM handed to
//   the seam pass. Free RAM is measured after the images/patches are already resident,
//   so it is real headroom; the remaining fraction is the safety margin for the OS,
//   for other stages' allocator slack, and for the fact that the ~96 B/px model is an
//   estimate. 0.75 leaves a quarter of the headroom untouched.
// TEXTURE_SEAM_MEM_BUDGET_MIN_GB: floor, so a transient dip in free RAM cannot collapse
//   the budget to nothing (the worker count clamps to >=1 regardless).
// TEXTURE_SEAM_MEM_BUDGET_MAX_GB: optional ceiling, 0 = unlimited. Set it only to keep
//   texturing from claiming a whole shared machine.
// TEXTURE_SEAM_MEM_BUDGET_FALLBACK_GB: used when the OS query fails (returns zeros).
#ifndef TEXTURE_SEAM_MEM_FREE_FRACTION
#define TEXTURE_SEAM_MEM_FREE_FRACTION 0.75
#endif
#ifndef TEXTURE_SEAM_MEM_BUDGET_MIN_GB
#define TEXTURE_SEAM_MEM_BUDGET_MIN_GB 1
#endif
#ifndef TEXTURE_SEAM_MEM_BUDGET_MAX_GB
#define TEXTURE_SEAM_MEM_BUDGET_MAX_GB 0
#endif
#ifndef TEXTURE_SEAM_MEM_BUDGET_FALLBACK_GB
#define TEXTURE_SEAM_MEM_BUDGET_FALLBACK_GB 8
#endif

// TEXTURE_SEAM_MAX_WORKERS: PARALLEL-EFFICIENCY cap, applied on top of the memory cap.
// LocalSeamLeveling is DRAM-bandwidth-bound (Poisson blending streams several float
// buffers per patch, and the concurrent working set is far larger than L3 at any useful
// thread count), so it stops scaling well before the logical core count and then goes
// BACKWARDS. Measured on a 16-core/32-thread 7950X, same 9910-patch scene:
//     3 workers -> 5198 ms (15.6 core-s)
//    14 workers -> 2987 ms (41.8 core-s)
//    32 workers -> 4196 ms (134 core-s)
// i.e. 32 workers burn ~8.6x the CPU of 3 for the same output and finish SLOWER than 14.
// This is not a memory-capacity problem -- there was 45 GB free and the process peak did
// not even move -- so no memory budget can express it; it needs its own cap. Physical
// cores is the principled default: the SMT siblings add contention but no extra load/store
// bandwidth, which is the resource this pass is actually short of.
//   0 (DEFAULT) = auto: cap at the physical core count (no cap if detection fails).
//  >0           = explicit worker cap, for tuning a specific machine.
#ifndef TEXTURE_SEAM_MAX_WORKERS
#define TEXTURE_SEAM_MAX_WORKERS 0
#endif

// NEGATIVE RESULT -- do not re-try without new evidence: the same physical-core cap was
// applied to the ListCameraFaces per-view loop, on the theory that its ~20 B/px of private
// full-res scratch (mGrad x2, imageGradMag, faceMap, depthMap -- ~335 MB/worker at 16.7 MP,
// ~10.7 GB across 32 workers) would make it DRAM-bound the same way. It did not: on a
// 192-image res-0 scene, 32 -> 16 workers moved that stage 11165 -> 11475 ms, i.e. no gain
// (slightly worse, inside run-to-run noise), for a 1.8 GB lower process peak. The two
// passes are NOT the same regime. The per-view loop allocates its scratch ONCE per worker
// and reuses it across every view (all source images are the same size), so it streams a
// FIXED working set. LocalSeamLeveling resizes its buffers per patch across 15k patches of
// wildly varying size, and it is that realloc churn -- not streaming volume alone -- that
// makes it contend. Reverted; only the seam pass is capped.
// (Since TEXTURE_FUSED_GRADIENT, the two mGrad planes are gone and that private scratch is
// ~12 B/px -- imageGradMag, faceMap, depthMap -- plus ~46 B per candidate face for the
// raster binning; see the accounting at the per-view declarations. The loop IS bandwidth-
// bound, as later profiling showed, but capping workers is still not the lever: cutting
// bytes moved per view is.)

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

// TEXTURE_CROP_OVERSAMPLE: cap crop extraction at this multiple (LINEAR) of the texel
// density the atlas can actually hold. 0 disables the cap (extract at source density).
//
// Crops are extracted at source resolution and AdaptiveFitPatches then discards most of
// it. MEASURED on a 942,700 unit^2 site: 2,792.9 Mpx of crops -- ~8.4 GB -- extracted to
// fill a 247.4 Mpx atlas, so 91% is decoded, rectified and thrown away at full peak cost.
//
// 2.0 is chosen so the change is VISUALLY NEUTRAL, not just cheaper. Extract-then-filter
// is supersampling, and INTER_AREA over a 2x oversampled source is indistinguishable from
// the same filter over an 11x one -- the benefit saturates around 2x. Do NOT drop this to
// 1.0: at small downscale ratios (a 1.77x scene measured elsewhere) the second resample
// genuinely antialiases better than extracting straight at target density.
//
// The cap is derived from capacity/totalWorldArea, which UNDERSTATES the density the fit
// finally settles on (patches already under the ceiling are left alone, so the bisection
// lands higher). Measured actual-to-estimate ratios: 1.00 and 1.32; a 2x linear cap is 4x
// in area, clear of both.
#ifndef TEXTURE_CROP_OVERSAMPLE
#define TEXTURE_CROP_OVERSAMPLE 2.0
#endif

// TEXTURE_KEEP_IMAGES_RESIDENT: skip the second decode pass when RAM allows.
// TEXTURE_CROP_IMAGES frees each source image the moment its view is rasterized, so the
// per-patch extract stage has to RE-READ AND RE-DECODE every used image from disk purely
// to cut its crops -- every source image is decoded TWICE per run (~0.5 s on a 109-image
// 2.9 MP set, ~25 s on a 367-image 18.6 MP set). That trade only makes sense when the
// images do not fit: 109 x 2.9 MP x 3 B is 0.94 GB, i.e. 2% of a 64 GB box's free RAM.
// When 1 (default), the decoded-image total is estimated up front and compared against
// the free physical RAM; if it fits inside TEXTURE_KEEP_IMAGES_FRACTION of it, the images
// are KEPT resident through ListCameraFaces and the extract stage crops straight out of
// them -- one decode pass instead of two, no behaviour change (the crops are cloned from
// identical pixels either way). If it does not fit, the release-and-reload path runs
// exactly as before. Note the estimate is a headroom check made ONCE, before the images
// are loaded, so it deliberately measures against a fraction well under 1.
// 0 = always release and reload (previous behaviour).
#ifndef TEXTURE_KEEP_IMAGES_RESIDENT
#define TEXTURE_KEEP_IMAGES_RESIDENT 1
#endif
// Share of free physical RAM the resident source images may occupy. Kept low because
// this budget is spent for the WHOLE run (the images stay live through view selection,
// patch extraction and seam leveling), unlike the seam-leveling transient.
#ifndef TEXTURE_KEEP_IMAGES_FRACTION
#define TEXTURE_KEEP_IMAGES_FRACTION 0.35
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

// TEXTURE_LBP_PENALIZE_UNDEFINED_AGREEMENT: stop label 0 (UNDEFINED) from being a free
// match against itself in the INCREASE_PATCHES smoothness term.
//
// THE DEFECT. The two SmoothnessPottsStrong variants below disagree on one clause:
//     #ifdef INCREASE_PATCHES   if (l1 == l2)                      return 0;
//     #else                     if (l1 == l2 && l1 != 0 && l2 != 0) return 0;
// The stock (non-INCREASE_PATCHES) form deliberately EXCLUDES label 0, so two undefined
// neighbours pay the pair penalty like any other. The INCREASE_PATCHES form lost that
// guard, so an undefined<->undefined edge costs NOTHING -- which makes undefined regions
// energetically self-reinforcing: once a blob exists, a bordering face with a real view
// available finds it cheaper to join the blob (0 smoothness) than to keep its label and
// pay the switch penalty at the blob edge. Faces that HAVE usable views get recruited into
// the fill.
//
// MEASURED on the Niwot house, and the file's own [LBP-DIAG] counter was added to catch
// exactly this ("undefined well above [the floor] means textureable faces are being pulled
// into undefined patches by the smoothness term"):
//     faces=496114 unobserved=24720   <- hard floor: faces with NO candidate view at all
//     iter=1  undefined=26998
//     iter=10 undefined=29737
//     iter=50 undefined=29984         <- monotonic climb, never recovers
// 29,984 - 24,720 = 5,264 faces (17.6% of the no-view set) had usable views and were
// discarded anyway. Each one then gets a data-colour fill tile AND a seam-feather band on
// the real texture around it, so the cost is paid twice over.
//
// WHY THE FIX IS NOT HERE -- do not "restore the missing l1 != 0 guard" in this function.
// It cannot work, and it fails SILENTLY, which is worse than not trying.
//
// FaceViewSelection calls inference.SetPottsSmoothness(true). Under that flag LBP.h treats
// the pairwise term as a strict Potts model: it samples the per-edge constant EXACTLY ONCE,
// as fncSmoothCost(n1, n2, 0, 1) (a DISTINCT pair) in PrepareTopology, and its message fast
// path then hardcodes "0 when l1 == l2, edgeWeight otherwise". So the l1 == l2 branch of
// this function is never evaluated during the solve, and any change to it is a no-op.
//
// Nor can the flag simply be turned off. "0 == 0 costs W but A == A costs 0" is not a Potts
// model, so the fast path structurally cannot express it, and the general O(L1*L2) path
// evaluates fncSmoothCost L1*L2 times per directed edge per sweep. On this scene that is
// edges=1,485,746 x avgLabels 8.84^2 x 50 sweeps ~= 5.8e9 normal dot-products -- the exact
// hot spot the fast path was introduced to remove.
//
// The fix lives in the DATA cost instead, where it is exact and needs no tuning: a face that
// HAS a candidate view is simply not offered label 0. See
// TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED at the node-setup loop in FaceViewSelection.
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

// Number of PHYSICAL cores (not SMT siblings), or 0 if it cannot be determined.
// Used to cap the bandwidth-bound passes -- see TEXTURE_SEAM_MAX_WORKERS.
// Queried once on first use.
static int GetPhysicalCoreCount()
{
	static const int nCores = []() -> int {
		#if defined(_MSC_VER)
		DWORD len = 0;
		::GetLogicalProcessorInformation(NULL, &len);
		if (::GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
			return 0;
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
			(len + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) - 1) / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
		if (!::GetLogicalProcessorInformation(buf.data(), &len))
			return 0;
		int n = 0;
		for (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION& e : buf)
			if (e.Relationship == RelationProcessorCore)
				++n;
		return n;
		#else
		// No portable physical-core query; callers treat 0 as "unknown" and skip the cap.
		return 0;
		#endif
	}();
	return nCores;
}

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
#if TEXTURE_RASTER_NO_BACKFACE_CULL
		// GPU-parity: rasterize both windings (no back-face cull); the depth buffer
		// decides visibility, so grazing walls whose winding flips still get pixels
		void Project(const Mesh::Face& facet) {
			typename Base::Triangle triangle;
			typename Base::TriangleRasterizer tr(triangle, *this);
			for (int v = 0; v < 3; ++v)
				if (!this->ProjectVertex(vertices[facet[v]], v, triangle))
					return;
			Image8U3::RasterizeTriangleBary<float, typename Base::TriangleRasterizer, false>(
				triangle.pti[0], triangle.pti[1], triangle.pti[2], tr);
		}
#endif
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
	// Returns false if the atlas cannot be built at all (patches don't fit within
	// nMaxTextureSize even after AdaptiveFitPatches) -- see TEXTURE_ATLAS_ADAPTIVE_FIT.
	bool GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize);
	// Downscale only the patches whose native resolution is denser than the mesh
	// geometry they cover actually needs, until total patch area fits budgetAreaPixels;
	// also unconditionally clamps any single patch whose own width or height alone
	// exceeds maxPatchDim (area-based trimming alone cannot rescue that case -- see the
	// implementation). Patches already within both limits are left untouched. Returns
	// the number of patches rescaled. No-op (returns 0) when TEXTURE_CROP_IMAGES is off,
	// since patches don't own a private, independently resizable pixel buffer.
	unsigned AdaptiveFitPatches(uint64_t budgetAreaPixels, int maxPatchDim);

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

	// Boundary-sheet faces selected for removal (see TEXTURE_DELETE_BOUNDARY_UNOBSERVED).
	// Collected during the data-colour pass but applied only AFTER texturing finishes:
	// texturePatches, components and faceTexcoords are all face-indexed, so deleting
	// mid-pipeline would invalidate them.
	std::vector<FIndex> unobservedToDelete;

	// Per face: did it EVER project with FRONT winding in ANY view? Filled by
	// ListCameraFaces, read by the data-colour stage (see
	// TEXTURE_FEATHER_SKIP_HIDDEN_FILL). Separates the two very different reasons a face
	// can end up with no view:
	//   1 = real surface a camera could have seen head-on, but it lost the depth test
	//       (occluded) or was too small / too poorly seen to win a pixel. Feathering the
	//       transition into it is meaningful -- it abuts texture the viewer sees.
	//   0 = never front-wound anywhere, i.e. inverted or interior geometry (a flap behind
	//       the real surface). Invisible in every render, so nothing about its edge needs
	//       softening. On Niwot this is 17,670 of 29,942 no-view faces.
	// Empty if ListCameraFaces has not run, which every consumer treats as "no info".
	std::vector<uint8_t> everFrontWound;

	// Per face: did it EVER project in-bounds and in front of a camera in ANY view, EITHER
	// winding? Needed because everFrontWound == 0 is ambiguous on its own and conflates two
	// completely different situations:
	//   everProjected=1, everFrontWound=0 -> it WAS tested, and every time it faced away.
	//                                        Inward-facing geometry. Safe to delete.
	//   everProjected=0                   -> NEVER tested (outside every frustum). No evidence
	//                                        either way; this is real surface at the edge of
	//                                        coverage and must be KEPT.
	// MEASURED why this matters: treating the two alike dropped 63,662 of 71,223 unobserved
	// faces (89%) on SchnellTests -- exactly back-winding-cull 42,935 + off-frustum 20,725 --
	// and the off-frustum share punched scattered holes through real terrain.
	std::vector<uint8_t> everProjected;

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

	// TEXTURE_KEEP_IMAGES_RESIDENT: decided once in ListCameraFaces from the free physical
	// RAM. When true the source images are NOT released after their view is rasterized, and
	// the per-patch extract stage crops from the resident pixels instead of re-decoding.
	bool bKeepImagesResident;

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
	bKeepImagesResident(false), // decided in ListCameraFaces once the budget is known
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

		// invZ == 0 is the "not in front of the camera" sentinel the rasterizer
		// tests. Do NOT clamp zc positive: that would project a vertex behind the
		// near plane to a finite in-front coordinate instead of rejecting it.
		out.verts[i] = { xc, yc, zc, zc > 1e-6f ? 1.f / zc : 0.f };
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

// TEXTURE_FUSED_GRADIENT: build the per-view gradient-magnitude plane in one pass
// over a rolling 3-row window instead of toGray -> Sobel x -> Sobel y -> combine.
//
// The staged version is 34.4% of ListCameraFaces' per-view loop, and almost all of
// that is DRAM traffic on intermediates nothing outside the stage ever reads. At
// 20 MP: toGray writes an 80 MB float plane, each Sobel reads it and writes another
// 80 MB, and the Eigen combine reads both back to write 80 MB -- ~740 MB per view to
// produce a plane that is 80. The two Sobel planes are also freshly allocated and
// released per view (160 MB of first-touch page faults each time round).
//
// Fused, the same plane costs "read the BGR image once, write the result once"
// (~140 MB), with the gray row, the horizontally-filtered rows, and the 3-row window
// all inside ~140 KB of L2. 0 = previous staged path (kept for A/B).
#ifndef TEXTURE_FUSED_GRADIENT
#define TEXTURE_FUSED_GRADIENT 1
#endif

#if TEXTURE_FUSED_GRADIENT
// Gradient magnitude of the grayscale image, |Sobel3x3|, fused into one pass.
//
// Numerically this is the staged chain it replaces, term for term:
//  * gray = (0.114*B + 0.587*G + 0.299*R) / 255, in float, associated left to right
//    -- exactly TImage::toGray(COLOR_BGR2GRAY, bNormalize=true, bSRGB=false);
//  * Sobel is applied separably, the way cv::Sobel does it (getDerivKernels(1,0,3)
//    is [-1,0,1] across and [1,2,1] down; dy swaps them), so each row keeps a
//    horizontal derivative hD and a horizontal smooth hS, and the vertical pass
//    reads the 3-row window: gx = (hD[y-1] + 2*hD[y] + hD[y+1])/8,
//    gy = (hS[y+1] - hS[y-1])/8. OpenCV correlates rather than convolves, hence
//    these signs -- irrelevant here anyway, only the magnitude is kept;
//  * borders are BORDER_REFLECT_101 (cv::Sobel's BORDER_DEFAULT): row -1 mirrors to
//    row 1, column -1 to column 1, and likewise at the far edge.
// Only the float rounding order differs from OpenCV's separable pass, in the last
// ulp of a value that is then averaged over a face's ~150 pixels to rank views.
static void ComputeGradientMagnitudeFused(
	const Image8U3& image, TImage<float>& gradMag, std::vector<float>& ring)
{
	const int W = image.cols, H = image.rows;
	gradMag.create(H, W);
	if (W < 2 || H < 2) {
		gradMag.memset(0); // no 3x3 neighbourhood exists; the staged path would too
		return;
	}

	// one gray row + 3 rows each of hD and hS: 7 x W floats, ~140 KB at 5000 px
	ring.resize((size_t)W * 7);
	float* const gray = ring.data(); // no __restrict: it is captured by the lambda below
	float* const hD[3] = { ring.data() + (size_t)W * 1, ring.data() + (size_t)W * 2, ring.data() + (size_t)W * 3 };
	float* const hS[3] = { ring.data() + (size_t)W * 4, ring.data() + (size_t)W * 5, ring.data() + (size_t)W * 6 };

	constexpr float cb = 0.114f, cg = 0.587f, cr = 0.299f;
	constexpr float inv255 = 1.f / 255.f;
	constexpr float scale = 1.f / 8.f;

	// gray-convert source row y (reflect_101 if outside) and horizontally filter it
	// into ring slot `slot`
	const auto LoadRow = [&](int y, int slot) {
		if (y < 0) y = -y;                     // -1 -> 1
		else if (y >= H) y = 2 * (H - 1) - y;  // H -> H-2
		const uint8_t* __restrict const s = image.ptr<uint8_t>(y);
		for (int x = 0; x < W; ++x)
			gray[x] = (cb * s[x * 3 + 0] + cg * s[x * 3 + 1] + cr * s[x * 3 + 2]) * inv255;
		float* __restrict const d = hD[slot];
		float* __restrict const t = hS[slot];
		// column reflect_101: gray[-1] -> gray[1], gray[W] -> gray[W-2]
		d[0] = 0.f; // gray[1] - gray[1]
		t[0] = gray[1] + 2.f * gray[0] + gray[1];
		for (int x = 1; x < W - 1; ++x) {
			d[x] = gray[x + 1] - gray[x - 1];
			t[x] = gray[x - 1] + 2.f * gray[x] + gray[x + 1];
		}
		d[W - 1] = 0.f;
		t[W - 1] = gray[W - 2] + 2.f * gray[W - 1] + gray[W - 2];
	};

	// row r lives in slot (r+1)%3, so the window for row y is slots y, y+1, y+2 (mod 3)
	LoadRow(-1, 0);
	LoadRow(0, 1);
	for (int y = 0; y < H; ++y) {
		LoadRow(y + 1, (y + 2) % 3); // overwrites row y-2, which has aged out
		const float* __restrict const dm = hD[(y + 0) % 3]; // row y-1
		const float* __restrict const dc = hD[(y + 1) % 3]; // row y
		const float* __restrict const dp = hD[(y + 2) % 3]; // row y+1
		const float* __restrict const tm = hS[(y + 0) % 3];
		const float* __restrict const tp = hS[(y + 2) % 3];
		float* __restrict const out = gradMag.ptr<float>(y);
		for (int x = 0; x < W; ++x) {
			const float gx = (dm[x] + 2.f * dc[x] + dp[x]) * scale;
			const float gy = (tp[x] - tm[x]) * scale;
			out[x] = FastSqrtS(gx * gx + gy * gy);
		}
	}
}
#endif

#if TEXTURE_FAST_RASTER
// Rasterize one view's faces into faceMap (LOCAL cameraFaces index of the nearest
// face covering each pixel) + depthMap (that face's depth). Replaces
// TRasterMesh::Project + TImage::RasterizeTriangleBary in ListCameraFaces.
//
// PROFILED, not guessed. On a 367-view / 20 MP scene the straight face-order walk
// (this function's first version, and the stock path before it) cost 44.7 ns per
// frame pixel, while the sequential accumulation scan right after it -- same buffers,
// same threads, MORE bytes touched per pixel -- cost 4.9. A 9x gap on identical
// hardware is not arithmetic, it is misses: faceMap+depthMap are 8 B x 20 MP = 160 MB
// per view, a face covers ~150 px spread over ~12 rows, and consecutive faces land
// anywhere in the frame. Every triangle row is a cold line in each of two buffers,
// and 40k pages per buffer thrashes the L2 TLB on top. Cutting ALU work off that
// (which the first version did: unique-vertex float transforms instead of ~6x
// redundant double ones, incremental barycentrics instead of three edge functions
// per bbox pixel, no per-pixel PARSER callback) barely moved the stage, because the
// arithmetic was already hiding behind the stalls.
//
// So the faces are BINNED INTO SCREEN TILES and rasterized tile by tile. A tile's
// slice of both buffers is TEXTURE_RASTER_TILE^2 x 8 B (128 KB at 128 px), which
// lives in L2 for the whole time faces are drawn into it. Two consequences:
//  * every depth read-modify-write after the first hits cache, not DRAM;
//  * the tile's CLEAR moves inside the loop, so the buffers are written once and
//    never read back. That drops the raster's DRAM traffic from ~3 passes over
//    160 MB (memset, then read depth, then write depth+face) to ~1.
//
// Output is BIT-IDENTICAL to the untiled walk: pixels belong to exactly one tile,
// and a tile's bin is filled in ascending face order, so each pixel still sees its
// covering faces in the same relative order and the `d > z` first-wins tie goes the
// same way. Set TEXTURE_RASTER_TILE to 0 to collapse this to a single frame-sized
// tile (i.e. the untiled walk) for A/B.
//
// Costs kept from the first version, all still worth having once the misses are gone:
//  * each UNIQUE vertex is transformed once, in float, through the pre-composed P
//    matrix (camera.Pf) -- ProjectVertex ran TransformPointW2C + TransformPointC2I
//    in REAL (double) for all three vertices of EVERY face, so a vertex shared by
//    ~6 faces was transformed ~6 times per view;
//  * barycentrics are evaluated once per row segment and stepped across it, instead
//    of three edge functions at every bbox pixel including the ~half that miss;
//  * PerspectiveCorrectBarycentricCoordinates + ComputeDepth (six products and a
//    divide, behind a callback) collapse to z = 1 / (w0/z0 + w1/z1 + w2/z2).
//
// Behaviour preserved from the stock path: whole-triangle reject unless all three
// vertices are in front of the camera AND project inside the image with a 3 px
// border (the test is written positively, so NaN coordinates fail it, as they did
// through isInsideWithBorder); both windings rasterize under
// TEXTURE_RASTER_NO_BACKFACE_CULL, positive-area winding only otherwise (note
// EdgeFunction2 is the negation of Common's EdgeFunction, hence the flipped
// comparison); nearest depth wins, ties keep the earlier face.
// One deliberate difference: exactly-zero-area triangles are dropped. The stock path
// divided by that zero, and the resulting NaN barycentrics passed its negativity
// rejects, letting a degenerate face scribble over its bounding box.

// Screen-tile edge in pixels. Both buffer slices of one tile must fit L2 alongside
// whatever the co-resident SMT sibling is doing: 128 px -> 128 KB, comfortable
// everywhere. 0 = one frame-sized tile (untiled walk, for A/B).
#ifndef TEXTURE_RASTER_TILE
#define TEXTURE_RASTER_TILE 128
#endif

// Per-thread scratch for the tiled rasterizer. Held by the caller (OpenMP-private)
// so the buffers grow once per thread rather than per view. ~21 B per candidate
// face: 16 for the clipped bbox, ~4.4 for the CSR bin payload at ~1.1 tiles/face.
struct RasterScratch {
	std::vector<int32_t> boxes;       // 4/face: minX,minY,maxX,maxY (minX > maxX = culled)
	std::vector<uint32_t> tileStart;  // CSR offsets, numTiles+1
	std::vector<uint32_t> tileCursor; // per-tile write cursor while scattering
	std::vector<uint32_t> tileFaces;  // CSR payload: local face indices, ascending per tile
};

// One face's projected triangle. Recomputed in the tile pass rather than cached: it
// is 3 loads + 6 multiplies against a verts array that stays hot, versus ~90 B per
// face of extra streamed state.
struct TriProj {
	Point2f v[3];
	const CamVert* c[3];
	float area;
};

// Project one face and apply the stock culls. False = this face draws nothing.
static inline bool ProjectCameraFace(const CameraRenderData& rd, size_t fi,
	float minB, float maxBX, float maxBY, TriProj& t)
{
	const Face& face = rd.faces[fi];
	t.c[0] = &rd.verts[face[0]];
	t.c[1] = &rd.verts[face[1]];
	t.c[2] = &rd.verts[face[2]];
	for (int i = 0; i < 3; ++i) {
		const CamVert& c = *t.c[i];
		// in front of the camera? (invZ == 0 is the behind/at-camera sentinel
		// UpdateCameraVertsAndNormals sets)
		if (c.invZ == 0.f)
			return false;
		t.v[i] = Point2f(c.x * c.invZ, c.y * c.invZ);
		// inside the image with the 3 px border? (isInsideWithBorder<float,3>)
		if (!(t.v[i].x >= minB && t.v[i].x <= maxBX && t.v[i].y >= minB && t.v[i].y <= maxBY))
			return false;
	}
	t.area = EdgeFunction2(t.v[0], t.v[1], t.v[2]);
#if TEXTURE_RASTER_NO_BACKFACE_CULL
	return t.area != 0.f; // degenerate only: invArea would be infinite
#else
	return t.area < 0.f;  // back-oriented (EdgeFunction2 == -EdgeFunction)
#endif
}

static void RasterizeCameraFacesFast(
	const CameraRenderData& rd,
	TImage<cuint32_t>& faceMap,
	DepthMap& depthMap,
	RasterScratch& scratch)
{
	const int width = depthMap.cols;
	const int height = depthMap.rows;
	static_assert(NO_ID == (uint32_t)0xFFFFFFFF, "faceMap clear assumes NO_ID is all-bits-1");
	if (width <= 0 || height <= 0)
		return;

	// isInsideWithBorder<float,3>: pt >= 3 and pt <= size-4, inclusive both ends
	constexpr int border = 3;
	const float minB = (float)border;
	const float maxBX = (float)(width - (border + 1));
	const float maxBY = (float)(height - (border + 1));

	Depth* __restrict const depthPtr = depthMap.ptr<Depth>(0);
	cuint32_t* __restrict const facePtr = faceMap.ptr<cuint32_t>(0);

#if TEXTURE_RASTER_TILE > 0
	// constexpr so the tile-index divisions below fold to shifts: pass 1 and pass 2
	// each do four of them per face, and idiv is ~20-40 cycles
	constexpr int tileSize = TEXTURE_RASTER_TILE;
#else
	const int tileSize = MAXF(width, height); // one frame-sized tile (untiled A/B)
#endif
	const int tilesX = (width + tileSize - 1) / tileSize;
	const int tilesY = (height + tileSize - 1) / tileSize;
	const size_t numTiles = (size_t)tilesX * (size_t)tilesY;
	const size_t numFaces = rd.faces.size();

	// an image narrower than the border, or no candidates: the maps must still come
	// back cleared, so pass 3 runs regardless and only the binning is skipped
	const bool anyFaces = (maxBX >= minB && maxBY >= minB && numFaces > 0);

	scratch.tileStart.assign(numTiles + 1, 0);
	if (anyFaces) {
		// ---- pass 1: project + cull once, keep the bbox, count per-tile hits ------
		scratch.boxes.resize(numFaces * 4);
		int32_t* __restrict const boxes = scratch.boxes.data();
		// deliberately NOT __restrict: the prefix sum below walks the same storage
		// through scratch.tileStart while this pointer is still in scope
		uint32_t* const counts = scratch.tileStart.data();
		TriProj t;
		for (size_t fi = 0; fi < numFaces; ++fi) {
			int32_t* __restrict const box = boxes + fi * 4;
			if (!ProjectCameraFace(rd, fi, minB, maxBX, maxBY, t)) {
				box[0] = 1; box[2] = 0; // minX > maxX marks it culled
				continue;
			}
			float boxMinX = t.v[0].x, boxMaxX = t.v[0].x;
			float boxMinY = t.v[0].y, boxMaxY = t.v[0].y;
			for (int i = 1; i < 3; ++i) {
				if (t.v[i].x < boxMinX) boxMinX = t.v[i].x;
				if (t.v[i].x > boxMaxX) boxMaxX = t.v[i].x;
				if (t.v[i].y < boxMinY) boxMinY = t.v[i].y;
				if (t.v[i].y > boxMaxY) boxMaxY = t.v[i].y;
			}
			// every vertex is inside the border, so the box needs no clipping;
			// coordinates are >= border > 0, so truncation is floor
			box[0] = (int32_t)boxMinX;
			box[1] = (int32_t)boxMinY;
			box[2] = (int32_t)CeilPos(boxMaxX); // inclusive, as RasterizeTriangleBary
			box[3] = (int32_t)CeilPos(boxMaxY);
			const int tx0 = box[0] / tileSize, tx1 = box[2] / tileSize;
			const int ty0 = box[1] / tileSize, ty1 = box[3] / tileSize;
			for (int ty = ty0; ty <= ty1; ++ty)
				for (int tx = tx0; tx <= tx1; ++tx)
					++counts[(size_t)ty * tilesX + tx + 1];
		}

		// ---- pass 2: prefix-sum the counts, scatter face indices into the bins ----
		for (size_t tl = 1; tl <= numTiles; ++tl)
			scratch.tileStart[tl] += scratch.tileStart[tl - 1];
		scratch.tileFaces.resize(scratch.tileStart[numTiles]);
		scratch.tileCursor.assign(scratch.tileStart.begin(), scratch.tileStart.end() - 1);
		uint32_t* __restrict const cursor = scratch.tileCursor.data();
		uint32_t* __restrict const bins = scratch.tileFaces.data();
		for (size_t fi = 0; fi < numFaces; ++fi) {
			const int32_t* __restrict const box = boxes + fi * 4;
			if (box[0] > box[2])
				continue;
			const int tx0 = box[0] / tileSize, tx1 = box[2] / tileSize;
			const int ty0 = box[1] / tileSize, ty1 = box[3] / tileSize;
			for (int ty = ty0; ty <= ty1; ++ty)
				for (int tx = tx0; tx <= tx1; ++tx)
					bins[cursor[(size_t)ty * tilesX + tx]++] = (uint32_t)fi;
		}
	}

	// ---- pass 3: clear and rasterize one tile at a time, both hot in L2 -----------
	const uint32_t* __restrict const bins = (anyFaces && !scratch.tileFaces.empty()) ? scratch.tileFaces.data() : nullptr;
	const int32_t* __restrict const boxes = anyFaces ? scratch.boxes.data() : nullptr;
	TriProj t;
	for (int tyi = 0; tyi < tilesY; ++tyi) {
		const int ty0 = tyi * tileSize;
		const int ty1 = MINF(ty0 + tileSize, height) - 1;
		for (int txi = 0; txi < tilesX; ++txi) {
			const int tx0 = txi * tileSize;
			const int tx1 = MINF(tx0 + tileSize, width) - 1;
			const size_t tileW = (size_t)(tx1 - tx0 + 1);

			// Clear this tile's slice HERE rather than memsetting the whole frame up
			// front: the lines stay in L2 for the faces drawn into them right below,
			// so the depth read-modify-write never goes back to DRAM.
			for (int y = ty0; y <= ty1; ++y) {
				const size_t off = (size_t)y * (size_t)width + (size_t)tx0;
				::memset(depthPtr + off, 0, tileW * sizeof(Depth));
				::memset(facePtr + off, 0xFF, tileW * sizeof(cuint32_t));
			}

			if (bins == nullptr)
				continue;
			const size_t tile = (size_t)tyi * tilesX + txi;
			const uint32_t binBeg = scratch.tileStart[tile], binEnd = scratch.tileStart[tile + 1];
			for (uint32_t b = binBeg; b < binEnd; ++b) {
				const uint32_t fi = bins[b];
				// it passed the culls in pass 1, so this cannot fail
				(void)ProjectCameraFace(rd, fi, minB, maxBX, maxBY, t);
				const Point2f& v0 = t.v[0];
				const Point2f& v1 = t.v[1];
				const Point2f& v2 = t.v[2];
				const float invArea = 1.f / t.area;

				const int32_t* __restrict const box = boxes + (size_t)fi * 4;
				const int colMin = MAXF((int)box[0], tx0), colMax = MINF((int)box[2], tx1);
				const int rowMin = MAXF((int)box[1], ty0), rowMax = MINF((int)box[3], ty1);

				// per-pixel barycentric gradient along a row (w_i weights vertex i)
				const float w0_dx = (v1.y - v2.y) * invArea;
				const float w1_dx = (v2.y - v0.y) * invArea;
				const float w2_dx = (v0.y - v1.y) * invArea;

				// only the reciprocals are needed: the perspective-correct depth is
				// 1/sum(w_i/z_i), so the z_i themselves cancel out
				const float iz0 = t.c[0]->invZ, iz1 = t.c[1]->invZ, iz2 = t.c[2]->invZ;

				for (int y = rowMin; y <= rowMax; ++y) {
					const size_t base = (size_t)y * (size_t)width;
					Depth* __restrict const depthRow = depthPtr + base;
					cuint32_t* __restrict const faceRow = facePtr + base;

					// Each row segment's starting barycentrics are evaluated exactly,
					// not carried by w_row += w_dy: three edge functions per ROW is
					// noise next to that row's pixels, and it keeps the incremental
					// error bounded by the segment width instead of by width*height.
					// Unlike the refine rasterizer, whose faces are subdivided to
					// ~32 px^2, a texturing face (a floor, a facade) can project across
					// thousands of rows. The stock path sampled at integer pixel
					// coordinates (Cast<T>(pt)) rather than pixel centers, so match it.
					const Point2f pRow((float)colMin, (float)y);
					float w0 = EdgeFunction2(v1, v2, pRow) * invArea;
					float w1 = EdgeFunction2(v2, v0, pRow) * invArea;
					float w2 = EdgeFunction2(v0, v1, pRow) * invArea;

					// skip the leading run of pixels outside the triangle (adds only)
					int x = colMin;
					while (x <= colMax && !(w0 >= 0.f && w1 >= 0.f && w0 + w1 <= 1.f)) {
						w0 += w0_dx;
						w1 += w1_dx;
						w2 += w2_dx;
						++x;
					}
					// the covered span is contiguous, so the segment ends when it does
					for (; x <= colMax; ++x) {
						// perspective-correct depth: PerspectiveCorrectBarycentric-
						// Coordinates then ComputeDepth is exactly 1/sum(w_i/z_i)
						const float z = 1.f / (w0 * iz0 + w1 * iz1 + w2 * iz2);
						const Depth d = depthRow[x];
						if (d == 0 || d > z) {
							depthRow[x] = z;
							faceRow[x] = (cuint32_t)fi;
						}
						w0 += w0_dx;
						w1 += w1_dx;
						w2 += w2_dx;
						if (w0 < 0.f || w1 < 0.f || w0 + w1 > 1.f)
							break;
					}
				}
			}
		}
	}
}
#endif

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

#if TEXTURE_DATACOLOR_DIAG
// dump mesh colored by per-face view class (lower = worse; vertex takes the worst
// incident face):
//  0 off-frustum/behind camera (blue)   1 back-winding cull only (red, FIXABLE)
//  2 front-winding but never won a pixel = occluded/sub-pixel (yellow)
//  3 rasterized but rawCos angle-rejected (orange)   4 observed (gray)
static void SaveFaceViewClassPLY(const String& fileName, const Mesh& mesh, const std::vector<uint8_t>& faceClass)
{
	if (faceClass.size() != mesh.faces.GetSize())
		return;
	std::vector<uint8_t> vClass(mesh.vertices.GetSize(), 5);
	FOREACH(f, mesh.faces) {
		const uint8_t c = faceClass[f];
		const Mesh::Face& fc = mesh.faces[f];
		for (int v = 0; v < 3; ++v)
			if (vClass[fc[v]] > c)
				vClass[fc[v]] = c;
	}
	FILE* fp = fopen(fileName, "wb");
	if (!fp)
		return;
	fprintf(fp, "ply\nformat binary_little_endian 1.0\n"
		"element vertex %u\n"
		"property float x\nproperty float y\nproperty float z\n"
		"property uchar red\nproperty uchar green\nproperty uchar blue\n"
		"element face %u\n"
		"property list uchar int vertex_indices\n"
		"end_header\n", mesh.vertices.GetSize(), mesh.faces.GetSize());
	FOREACH(v, mesh.vertices) {
		const Mesh::Vertex& vert = mesh.vertices[v];
		uint8_t rgb[3];
		switch (vClass[v]) {
		case 0:  rgb[0] = 0;   rgb[1] = 0;   rgb[2] = 255; break; // off-frustum/behind
		case 1:  rgb[0] = 255; rgb[1] = 0;   rgb[2] = 0;   break; // back-winding cull (fixable)
		case 2:  rgb[0] = 255; rgb[1] = 255; rgb[2] = 0;   break; // front-winding, occluded/sub-pixel
		case 3:  rgb[0] = 255; rgb[1] = 128; rgb[2] = 0;   break; // rasterized, angle-rejected
		case 4:  rgb[0] = 190; rgb[1] = 190; rgb[2] = 190; break; // observed
		default: rgb[0] = 90;  rgb[1] = 90;  rgb[2] = 90;  break; // untouched
		}
		fwrite(&vert, sizeof(float), 3, fp);
		fwrite(rgb, 1, 3, fp);
	}
	FOREACH(f, mesh.faces) {
		const uint8_t cnt = 3;
		fwrite(&cnt, 1, 1, fp);
		fwrite(&mesh.faces[f], sizeof(uint32_t), 3, fp);
	}
	fclose(fp);
}
#endif

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
#if TEXTURE_FEATHER_SKIP_HIDDEN_FILL
	// see the member declarations; written per view below, read by the data-colour stage
	everFrontWound.assign(faces.size(), 0);
	everProjected.assign(faces.size(), 0);
#endif
	Util::Progress progress(_T("Initialized views"), views.size());
	typedef float real;
	TImage<real> imageGradMag;
	TImage<real>::EMat mGrad[2];
	FaceMap faceMap;
	DepthMap depthMap;
#if TEXTURE_FAST_RASTER
	// Per-thread scratch for the fast rasterizer: the compacted (unique) camera-space
	// vertices + local faces of the view being rasterized. Declared here so the
	// OpenMP private() clause below gives each thread one that is reused across all
	// the views it handles (buffers grow once) and is destroyed at the end of the
	// region. Costs ~24 B per candidate face per THREAD (16 B/unique vertex for the
	// projected verts, 16 B/face for the local face + global-face index), on top of
	// the 4 B/face cameraFaces already holds; that is the price of not re-projecting
	// every shared vertex ~6 times in double precision.
	CameraRenderData camRender;
	// binning + bbox scratch for the tiled raster; ~21 B per candidate face per thread
	RasterScratch rasterScratch;
#endif
#if TEXTURE_FUSED_GRADIENT
	// rolling window for the fused gradient pass: 7 image rows of float, per thread
	std::vector<float> gradRing;
#endif

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

#if TEXTURE_DATACOLOR_DIAG
	// per-face flag: rasterized (front-facing by winding) in >=1 view, BEFORE the
	// rawCos angle cull, so we can tell why NO_ID faces were demoted
	std::vector<uint8_t> everRasterized(faces.size(), 0);
	// projected in front & on-image in >=1 view, split by winding sign, to further
	// classify never-rasterized faces (winding cull vs occlusion vs off-frustum)
	std::vector<uint8_t> projFront(faces.size(), 0), projBack(faces.size(), 0);
#endif

	// KEEP-IMAGES-RESIDENT DECISION (see TEXTURE_KEEP_IMAGES_RESIDENT): if every decoded
	// source image fits comfortably in the free physical RAM, hold them through to the
	// per-patch extract stage so that stage does not have to decode the whole set a
	// SECOND time just to cut crops.
	// The size MUST come from RecomputeMaxResolution (which reads the image FILE header),
	// not from the scene's stored width/height: those two disagree whenever the .mvs was
	// written by a stage that ran at a reduced resolution. On a real 168-image scene the
	// stored dims said 2.86 MP/image (1.41 GB total) while the files were 2x larger per
	// side and actually decoded to 11.2 MP/image (5.63 GB) -- a 4x under-estimate, which
	// on a bigger set would keep images resident that do not fit. Only the max dimension
	// comes back, so pair it with the stored ASPECT ratio, which scaling preserves.
#if TEXTURE_CROP_IMAGES && TEXTURE_KEEP_IMAGES_RESIDENT
	{
		constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
		// header reads hit the disk, so do them in parallel (this is the same query the
		// per-view loop below makes for each image anyway)
		std::vector<uint64_t> viewBytes(views.size(), 0);
#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for schedule(dynamic)
#endif
		for (int vi = 0; vi < (int)views.size(); ++vi) {
			const Image& imageData = images[views[(IIndex)vi]];
			if (!imageData.IsValid() || imageData.width == 0 || imageData.height == 0)
				continue;
			unsigned level(nResolutionLevel);
			// max dimension the loader will actually produce, derived from the FILE header
			const uint64_t imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
			// aspect is scale-invariant, so the stored dims still give the short side
			const double aspect((double)MINF(imageData.width, imageData.height) / (double)MAXF(imageData.width, imageData.height));
			viewBytes[(size_t)vi] = (uint64_t)(imageSize * imageSize * aspect) * 3ull;
		}
		uint64_t imagesBytes = 0;
		for (uint64_t b : viewBytes)
			imagesBytes += b;
		const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
		// a failed query (zeros) leaves the budget at 0 -> fall back to release-and-reload
		const uint64_t keepBudget = (uint64_t)((double)memInfo.freePhysical * TEXTURE_KEEP_IMAGES_FRACTION);
		bKeepImagesResident = (imagesBytes > 0 && imagesBytes <= keepBudget);
		TEXTURE_DIAG("ListCameraFaces: source images %.2f GB decoded, budget %.2f GB (%.1f GB free x %.2f) -> %s",
			imagesBytes / (double)GB, keepBudget / (double)GB, memInfo.freePhysical / (double)GB,
			(double)TEXTURE_KEEP_IMAGES_FRACTION,
			bKeepImagesResident ? "KEEP resident (single decode pass)" : "release + reload (two decode passes)");
	}
#endif

	// Since we are controlling the threading per-view, don't let cv thread
	// when performing its work.
	const int prevCvThreads = cv::getNumThreads();
	cv::setNumThreads(0);

	TEX_PROFILE_BEGIN(_tLcfCenters);
	static std::vector<Point3f> gFaceCenter;
	gFaceCenter.resize(faces.size());

	constexpr float oneThird = 1.0f / 3.0f;
#pragma omp parallel for schedule(static)
	for (int64_t f = 0; f < (int64_t)faces.size(); ++f) {
		const Face& fc = faces[(size_t)f];
		const Vertex& a = vertices[fc[0]];
		const Vertex& b = vertices[fc[1]];
		const Vertex& c = vertices[fc[2]];
		gFaceCenter[(size_t)f] = Point3f(
			(a.x + b.x + c.x) * oneThird,
			(a.y + b.y + c.y) * oneThird,
			(a.z + b.z + c.z) * oneThird
		);
	}
	TEX_PROFILE_END(_tLcfCenters, "ListCameraFaces: face centers");

	TEX_PROFILE_BEGIN(_tLcfPerView);
#if TEXTURE_PROFILE
	// The single stage timer around this loop cannot say whether the per-view time is
	// image filtering, the raster, or the full-frame accumulation scan -- and without
	// that split, optimizing the raster is guesswork. These counters accumulate CPU
	// nanoseconds per sub-stage ACROSS all worker threads (so the sum is ~nThreads x
	// the wall time of the loop) and are reported once, after it. ~8 clock reads per
	// view, i.e. free at this granularity.
	std::atomic<uint64_t> _pfDecode(0), _pfGrad(0), _pfBlur(0), _pfCull(0),
		_pfPrep(0), _pfRaster(0), _pfAccum(0), _pfOut(0);
	std::atomic<uint64_t> _pfRasterPix(0), _pfRasterFaces(0);
	#define LCF_TICK(v) const texprof::Clock::time_point v(texprof::Clock::now())
	#define LCF_TOCK(v, acc) (acc).fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(texprof::Clock::now() - (v)).count(), std::memory_order_relaxed)
#else
	#define LCF_TICK(v) ((void)0)
	#define LCF_TOCK(v, acc) ((void)0)
#endif
#ifdef TEXOPT_USE_OPENMP
	bool bAbort(false);
#if TEXTURE_FAST_RASTER
#pragma omp parallel for private(imageGradMag, mGrad, faceMap, depthMap, camRender, rasterScratch, gradRing)
#else
#pragma omp parallel for private(imageGradMag, mGrad, faceMap, depthMap)
#endif
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
		LCF_TICK(_t0);
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
		LCF_TOCK(_t0, _pfDecode);
		imageData.UpdateCamera(scene.platforms);
		// compute gradient magnitude
		LCF_TICK(_t1);
#if TEXTURE_FUSED_GRADIENT
		ComputeGradientMagnitudeFused(imageData.image, imageGradMag, gradRing);
		// mGrad stays empty on this path: the two full-res Sobel planes it held are
		// exactly the intermediates the fused pass keeps in L2 instead
#else
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
#endif
		LCF_TOCK(_t1, _pfGrad);
		// apply some blur on the gradient to lower noise/glossiness effects onto face-quality score
		LCF_TICK(_t2);
		cv::GaussianBlur(imageGradMag, imageGradMag, cv::Size(15, 15), 0, 0, cv::BORDER_DEFAULT);
		LCF_TOCK(_t2, _pfBlur);
		// select faces inside view frustum
		LCF_TICK(_t3);
		Mesh::FaceIdxArr cameraFaces;
		Mesh::FacesInserter inserter(cameraFaces);
		typedef TFrustum<float, 5> Frustum;
		const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
		octree.Traverse(frustum, inserter);
		LCF_TOCK(_t3, _pfCull);
		// project all triangles in this view and keep the closest ones
		faceMap.create(imageData.height, imageData.width);
		depthMap.create(imageData.height, imageData.width);
		// Rasterize storing a LOCAL face index (position in cameraFaces) into faceMap
		// instead of the global face id. This lets the per-pixel accumulation below
		// index a dense, cameraFaces-sized array directly instead of hashing every
		// rasterized pixel into a robin_map (millions of hash ops per view otherwise).
		const uint32_t numCameraFaces((uint32_t)cameraFaces.size());
#if TEXTURE_FAST_RASTER
		// PreprocessCameraFaces walks cameraFaces in order, so a face's index in
		// camRender.faces IS the local index the accumulation below indexes by.
		LCF_TICK(_t4);
		PreprocessCameraFaces(cameraFaces, faces, vertices, camRender);
		UpdateCameraVertsAndNormals(vertices, imageData.camera, camRender);
		LCF_TOCK(_t4, _pfPrep);
		LCF_TICK(_t5);
		RasterizeCameraFacesFast(camRender, faceMap, depthMap, rasterScratch);
		LCF_TOCK(_t5, _pfRaster);
#else
		LCF_TICK(_t5);
		RasterMesh rasterer(vertices, imageData.camera, depthMap, faceMap);
		rasterer.Clear();
		for (uint32_t li = 0; li < numCameraFaces; ++li) {
			const Face& facet = faces[cameraFaces[li]];
			rasterer.idxFace = (FIndex)li;
			rasterer.Project(facet);
		}
		LCF_TOCK(_t5, _pfRaster);
#endif
#if TEXTURE_PROFILE
		_pfRasterFaces.fetch_add(numCameraFaces, std::memory_order_relaxed);
		_pfRasterPix.fetch_add((uint64_t)faceMap.rows * (uint64_t)faceMap.cols, std::memory_order_relaxed);
#endif
#if TEXTURE_FEATHER_SKIP_HIDDEN_FILL
		// FRONT-WOUND RECORD -- always on, and deliberately cheap. See the everFrontWound
		// member and TEXTURE_FEATHER_SKIP_HIDDEN_FILL. Benign race: every writer stores 1.
		{
			uint8_t* const __restrict pFW = everFrontWound.data();
			uint8_t* const __restrict pEP = everProjected.data();
#if TEXTURE_FAST_RASTER
			// Reuses the camera-space verts UpdateCameraVertsAndNormals just produced for
			// this view, so the whole record costs one edge function per candidate face and
			// no reprojection at all. Bounds are the rasterizer's own
			// (isInsideWithBorder<float,3>), so a face counts here exactly when the raster
			// would have considered it.
			const float fwMinB = 3.f;
			const float fwMaxBX = (float)((int)imageData.width - 4);
			const float fwMaxBY = (float)((int)imageData.height - 4);
			const size_t nLocalFaces = camRender.faces.size();
			TriProj fwT;
			for (size_t li = 0; li < nLocalFaces; ++li) {
				if (!ProjectCameraFace(camRender, li, fwMinB, fwMaxBX, fwMaxBY, fwT))
					continue;
				// It projected in-bounds and in front, whatever its winding: this face HAS been
				// tested, which is what distinguishes it from one outside every frustum.
				pEP[camRender.globalFace[li]] = 1;
				// EdgeFunction2 is the negation of Common's EdgeFunction, so FRONT winding
				// (EdgeFunction > 0) is area < 0 here -- the same sign test
				// ProjectCameraFace applies when back-face culling is enabled.
				if (fwT.area < 0.f)
					pFW[camRender.globalFace[li]] = 1;
			}
#else
			// stock path: no pre-transformed verts to reuse, so project as the classifier
			// below does (this branch is only for A/B, where the extra cost is irrelevant)
			for (uint32_t li = 0; li < numCameraFaces; ++li) {
				const FIndex idxFace = cameraFaces[li];
				const Face& facet = faces[idxFace];
				Point2f pf[3];
				bool ok = true;
				for (int v = 0; v < 3; ++v) {
					const Point3 Xc(imageData.camera.TransformPointW2C(Cast<REAL>(vertices[facet[v]])));
					if (Xc.z <= 0) { ok = false; break; }
					pf[v] = imageData.camera.TransformPointC2I(Xc);
					if (!depthMap.isInsideWithBorder<float, 3>(pf[v])) { ok = false; break; }
				}
				if (!ok)
					continue;
				pEP[idxFace] = 1; // tested, whatever the winding
				if (EdgeFunction(pf[0], pf[1], pf[2]) > 0.f)
					pFW[idxFace] = 1;
			}
#endif
		}
#endif
#if TEXTURE_DATACOLOR_DIAG
		// classify projection/winding of each candidate face (same math as the
		// rasterizer's ProjectVertex + EdgeFunction cull) to sub-bucket NO_ID faces
		for (uint32_t li = 0; li < numCameraFaces; ++li) {
			const FIndex idxFace = cameraFaces[li];
			const Face& facet = faces[idxFace];
			Point2f pf[3];
			bool ok = true;
			for (int v = 0; v < 3; ++v) {
				const Point3 Xc(imageData.camera.TransformPointW2C(Cast<REAL>(vertices[facet[v]])));
				if (Xc.z <= 0) { ok = false; break; }
				pf[v] = imageData.camera.TransformPointC2I(Xc);
				if (!depthMap.isInsideWithBorder<float, 3>(pf[v])) { ok = false; break; }
			}
			if (!ok)
				continue;
			if (EdgeFunction(pf[0], pf[1], pf[2]) > 0.f)
				projFront[idxFace] = 1;
			else
				projBack[idxFace] = 1;
		}
#endif

		// accumulate per-face quality/area/color over the rasterized pixels; the dense
		// buffer is indexed by the local face index written into faceMap above.
		LCF_TICK(_t6);
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

		LCF_TOCK(_t6, _pfAccum);
		// ---- angle adjustment per face ----
		// Build output list for this view:
		LCF_TICK(_t7);
		std::vector<FaceOut> out;
		out.reserve(numCameraFaces);
		for (uint32_t li = 0; li < numCameraFaces; ++li) {
			FaceAccum& a = acc[li];
			if (a.area == 0)
				continue; // face had no rasterized pixel in this view
			const FIndex idxFace = cameraFaces[li];
#if TEXTURE_DATACOLOR_DIAG
			// rasterized front-facing here (won the depth test); benign racing set
			everRasterized[idxFace] = 1;
#endif

			const Face& f = faces[idxFace];
			const auto& faceCenter = gFaceCenter[idxFace];
			const Point3f camDir(Cast<Mesh::Type>(imageData.camera.C) - faceCenter);
			const Normal& faceNormal = scene.mesh.faceNormals[idxFace];
			const float rawCosFaceCam(ComputeAngle2(camDir.ptr(), faceNormal.ptr()));
#if TEXTURE_RASTER_NO_BACKFACE_CULL
			// visibility is the depth buffer (this face won the pixel), not the
			// smoothed-normal sign; keep grazing/back-normal faces as a floored-
			// quality last resort instead of demoting them to synthesized fill
			//
			// ...EXCEPT past TEXTURE_DATACOLOR_MIN_VIEW_COS, which this wires (it was a
			// dead macro -- see its comment). Rejecting the OBSERVATION rather than the
			// face is what makes it safe: a face that has any acceptable view keeps it,
			// and only one whose every view is this grazing ends up with no candidates
			// and falls through to NO_ID, where the data-colour fill and the
			// boundary-sheet delete can finally act on it.
			//
			// The ternary keeps a 0 threshold bit-identical to the previous behaviour
			// (only an exactly-zero cosine dropped, back-normal observations retained),
			// so the macro is a clean A/B rather than a rewrite of the default path.
			const float minViewCos = TEXTURE_DATACOLOR_MIN_VIEW_COS;
			if (minViewCos > 0.f ? (rawCosFaceCam < minViewCos)
			                     : (rawCosFaceCam == 0.f))
				continue;
			a.quality *= SQUARE(MAXF(rawCosFaceCam, 0.05f));
#else
			// skip observations where the camera looks at the back of the face
			// (rawCos <= 0); keeping them causes wrong-side texturing on edges.
			if (rawCosFaceCam <= 0.f) {
				continue;
			}
			a.quality *= SQUARE(rawCosFaceCam);
#endif

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
		LCF_TOCK(_t7, _pfOut);

#if TEXTURE_CROP_IMAGES
		// Crop-images mode: this view's pixels are fully consumed. Free them now so
		// ListCameraFaces holds only ~nThreads images resident instead of all of them.
		// GenerateTexture reloads each image and crops it to its used region. The
		// width/height/camera members stay intact (release() only frees the pixel Mat),
		// so projection can still compute patch rects from the stored dimensions.
		// Unless bKeepImagesResident: the whole set fits in RAM, so keeping the pixels
		// lets the extract stage crop from them instead of decoding everything again.
		if (!bKeepImagesResident)
			imageData.image.release();
#endif

		++progress;
	}

	progress.close();

#ifdef TEXOPT_USE_OPENMP
	if (bAbort)
		return false;
#endif
	TEX_PROFILE_END(_tLcfPerView, "ListCameraFaces: per-view rasterize+quality");
#if TEXTURE_PROFILE
	if (TEXTURE_DIAG_ENABLED()) {
		// Summed over threads, so read these as SHARES of the loop, not wall time.
		// "raster" is what TEXTURE_FAST_RASTER touches; if it is a small slice of the
		// total, a faster rasterizer cannot move this stage no matter how fast it is.
		const double ms = 1e-6;
		const double dec = _pfDecode.load() * ms, gra = _pfGrad.load() * ms, blu = _pfBlur.load() * ms,
			cul = _pfCull.load() * ms, pre = _pfPrep.load() * ms, ras = _pfRaster.load() * ms,
			acc = _pfAccum.load() * ms, out = _pfOut.load() * ms;
		const double tot = dec + gra + blu + cul + pre + ras + acc + out;
		const double inv = tot > 0 ? 100.0 / tot : 0.0;
		TEXTURE_DIAG("[PROFILE] LCF per-view CPU-ms summed over threads (total %.0f):", tot);
		TEXTURE_DIAG("[PROFILE]   decode/reload %9.0f (%4.1f%%) | toGray+Sobel %9.0f (%4.1f%%) | blur15x15 %9.0f (%4.1f%%)",
			dec, dec * inv, gra, gra * inv, blu, blu * inv);
		TEXTURE_DIAG("[PROFILE]   frustum cull  %9.0f (%4.1f%%) | raster prep  %9.0f (%4.1f%%) | RASTER     %9.0f (%4.1f%%)",
			cul, cul * inv, pre, pre * inv, ras, ras * inv);
		TEXTURE_DIAG("[PROFILE]   accum scan    %9.0f (%4.1f%%) | face-out     %9.0f (%4.1f%%)",
			acc, acc * inv, out, out * inv);
		const uint64_t nf = _pfRasterFaces.load(), np = _pfRasterPix.load();
		TEXTURE_DIAG("[PROFILE]   raster: %llu candidate faces, %llu frame px over %u views -> %.1f ns/face, %.2f ns/frame-px",
			(unsigned long long)nf, (unsigned long long)np, (unsigned)views.size(),
			nf ? _pfRaster.load() / (double)nf : 0.0, np ? _pfRaster.load() / (double)np : 0.0);
	}
#endif
#undef LCF_TICK
#undef LCF_TOCK

	// [MEM-DECOMP] decompose the ListCameraFaces resident footprint so we can target
	// the ~20 GB base that stage-level probes leave unexplained. Measures the ACTUAL
	// bytes of the two big suspects at their peak (perViewOut is fully built here,
	// just before the merge frees it). Both scans exist only for the log line, so the
	// whole block rides TEXTURE_DIAG.
	if (TEXTURE_DIAG_ENABLED()) {
		size_t pvBytes = 0, pvObs = 0;
		for (const auto& v : perViewOut) { pvBytes += v.capacity() * sizeof(FaceOut); pvObs += v.size(); }
		size_t imgBytes = 0; unsigned imgResident = 0;
		for (const Image& im : images) if (!im.image.empty()) { imgBytes += (size_t)im.image.total() * im.image.elemSize(); ++imgResident; }
		const double G = 1.0 / (1024.0*1024.0*1024.0);
		TEXTURE_DIAG("[MEM-DECOMP] perViewOut=%.2f GB (%zu obs, %zuB/obs) | images=%.2f GB (%u resident, %zuB/px*3) | mesh.faces=%u verts=%u",
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

	// [MEM-DECOMP] size facesDatas now that perViewOut is freed. Includes both the live
	// observations and the cList capacity slack (grow-by-doubling can hold ~2x the live
	// bytes). The per-face scan is log-only, so it rides TEXTURE_DIAG.
	if (TEXTURE_DIAG_ENABLED()) {
		size_t fdData = 0, fdCap = 0, fdObs = 0;
		for (FIndex f = 0; f < facesDatas.GetSize(); ++f) {
			const FaceDataArr& a = facesDatas[f];
			fdObs += a.GetSize();
			fdData += (size_t)a.GetSize() * sizeof(FaceData);
			fdCap += (size_t)a.GetCapacity() * sizeof(FaceData);
		}
		const double G = 1.0 / (1024.0*1024.0*1024.0);
		TEXTURE_DIAG("[MEM-DECOMP] facesDatas live=%.2f GB cap=%.2f GB (%zu obs, %zuB/obs, %u faces headers=%.2f GB)",
			fdData*G, fdCap*G, fdObs, sizeof(FaceData), facesDatas.GetSize(),
			((size_t)facesDatas.GetSize()*sizeof(FaceDataArr))*G);
		LogPeakMem("LCF: after merge (facesDatas full)");
	}

	// Restore cv's ability to thread.
	cv::setNumThreads(prevCvThreads);

#if TEXTURE_DATACOLOR_DIAG
	// classify every face to distinguish WHY it is NO_ID (magenta fill): observed /
	// rawCos-rejected / occluded / back-winding-culled / off-frustum
	{
		std::vector<uint8_t> faceClass(faces.size(), 0);
		size_t nObs = 0, nAngle = 0, nOccl = 0, nBackWind = 0, nOff = 0;
		for (size_t f = 0; f < faces.size(); ++f) {
			if (facesDatas[(FIndex)f].GetSize() > 0)      { faceClass[f] = 4; ++nObs; }
			else if (everRasterized[f])                   { faceClass[f] = 3; ++nAngle; }
			else if (projFront[f])                        { faceClass[f] = 2; ++nOccl; }
			else if (projBack[f])                         { faceClass[f] = 1; ++nBackWind; }
			else                                          { faceClass[f] = 0; ++nOff; }
		}
		DEBUG_EXTRA("[TEX-VIEWCLASS] observed=%zu angle-rejected=%zu occluded/subpix=%zu back-winding-cull=%zu off-frustum=%zu (of %u faces)",
			nObs, nAngle, nOccl, nBackWind, nOff, faces.GetSize());
		SaveFaceViewClassPLY(MAKE_PATH("TexViewClass.ply"), scene.mesh, faceClass);
	}
#endif

#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		TEX_PROFILE_BEGIN(_tLcfOutlier);
		// try to detect outlier views for each face
		// (views for which the face is occluded by a dynamic object in the scene, ex. pedestrians)
		// Each face owns its own view list and the detector is const and reads no member
		// state, so this is embarrassingly parallel and stays bit-exact whatever the thread
		// count: there is no reduction and no shared accumulator, only per-face writes.
		// dynamic, not static: per-face cost tracks the view count, which is spatially
		// correlated (faces mid-capture are seen by far more cameras than those at the
		// edges), so a static split would leave threads idle. 1024 keeps the scheduling
		// overhead negligible against a per-face body this small.
		const int_t nOutlierFaces = (int_t)facesDatas.GetSize();
		#pragma omp parallel for schedule(dynamic, 1024)
		for (int_t f = 0; f < nOutlierFaces; ++f)
			FaceOutlierDetection(facesDatas[(FIndex)f], fOutlierThreshold);
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

	// Scratch buffers. nAll is the number of views seeing this one face, so it is small
	// (typically well under 32). The caller runs this over every mesh face from an omp
	// parallel-for, so sizing these on the heap would put every worker on the process
	// heap five times per face, millions of times over - allocator contention, not the
	// arithmetic, would then set the ceiling. Keep the working set on the stack for the
	// common case and touch the heap only for the rare heavily-observed face. The stack
	// path also leaves nothing in this body that can throw, which matters because an
	// exception escaping an omp region is undefined behavior.
	constexpr uint32_t maxStackViews = 64;
	double stackC0[maxStackViews], stackC1[maxStackViews], stackC2[maxStackViews];
	uint8_t stackMask[maxStackViews];
	uint32_t stackIdx[maxStackViews];
	std::vector<double> heapC0, heapC1, heapC2;
	std::vector<uint8_t> heapMask;
	std::vector<uint32_t> heapIdx;
	double *c0, *c1, *c2;
	uint8_t* inlierMask;
	uint32_t* inlierIdx;
	if (nAll <= maxStackViews) {
		c0 = stackC0; c1 = stackC1; c2 = stackC2;
		inlierMask = stackMask;
		inlierIdx = stackIdx;
	} else {
		heapC0.resize(nAll); heapC1.resize(nAll); heapC2.resize(nAll);
		heapMask.resize(nAll);
		heapIdx.resize(nAll);
		c0 = heapC0.data(); c1 = heapC1.data(); c2 = heapC2.data();
		inlierMask = heapMask.data();
		inlierIdx = heapIdx.data();
	}

	// Preconvert colors once (SoA for cache + no Eigen temporaries), seeding the inlier
	// mask and the active index list with every view in the same pass.
	for (uint32_t i = 0; i < nAll; ++i) {
		const Color::EVec v = (const Color::EVec)faceDatas[i].color;
		c0[i] = (double)v[0];
		c1[i] = (double)v[1];
		c2[i] = (double)v[2];
		inlierMask[i] = 1;
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

					// Label counts up front for the CSR data-cost store (see
					// LBPInference::ReserveDataCosts): label 0 plus one per candidate
					// view, matching exactly what the two loops below then write.
					{
						std::vector<uint32_t> labelCounts(virtualFaces.size(), 1u);
						FOREACH(f, virtualFacesDatas)
							labelCounts[f] += (uint32_t)virtualFacesDatas[f].size();
						inference.ReserveDataCosts(labelCounts.data());
					}

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

				// Faces with no candidate view at all: they only ever get label 0, so
				// they are the hard floor for the "undefined=" figure the LBP reports
				// per sweep. undefined ~= this count means the solve is discarding
				// nothing; undefined well above it means textureable faces are being
				// pulled into undefined patches by the smoothness term (label 0 is a
				// free match against itself under INCREASE_PATCHES) and end up NO_ID.
				// Plain counter, not a reduction: this "omp for" is orphaned (no
				// enclosing parallel region), so it runs serially -- same reason the
				// shared "order" scratch above is safe.
				size_t numUnobservedFaces = 0;

				// ---- label-count pre-pass, for the CSR data-cost store ----
				// LBP.h keeps labels/dataCosts in flat arrays rather than two
				// small_vectors inlined into every Node, which needs the exact per-node
				// count up front. It is already known here: one label for an unobserved
				// face, otherwise one per candidate view (plus the undefined label when
				// TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED is off).
				{
					std::vector<uint32_t> labelCounts(numFaces);
					#pragma omp parallel for schedule(static)
					for (int64_t f = 0; f < (int64_t)numFaces; ++f) {
						const uint32_t nFD = (uint32_t)facesDatas[f].size();
#if TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED
						labelCounts[f] = nFD ? nFD : 1u;
#else
						labelCounts[f] = nFD + 1u;
#endif
					}
					inference.ReserveDataCosts(labelCounts.data());
				}

#pragma omp for schedule(dynamic, 128)
				for (int64_t f = 0; f < (int64_t)numFaces; ++f) {
					const FaceDataArr& faceDatas = facesDatas[f];

					const int nFD = (int)faceDatas.size();

					// Node itself is now just (label, dataCost); the candidate labels and
					// their costs are written straight into the reserved CSR slots.
					LBPInference::LabelID* __restrict lblOut = inference.Labels((LBPInference::NodeID)f);
					LBPInference::EnergyType* __restrict costOut = inference.DataCosts((LBPInference::NodeID)f);
					uint32_t nWritten = 0;

#if TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED
					// ---- undefined label: ONLY for a face with no candidate view ----
					// Offering it to a face that HAS a view lets the free undefined<->undefined
					// smoothness edge recruit that face into a fill blob (5,264 faces measured).
					// See TEXTURE_LBP_NO_UNDEFINED_WHEN_VIEWED, including why this cannot be
					// fixed in SmoothnessPottsStrong. nFD >= 1 below, so the node still has at
					// least one label, as LBP requires.
					if (nFD == 0) {
						lblOut[nWritten] = 0;
						costOut[nWritten] = undefinedCost;
						++numUnobservedFaces;
						continue;
					}
#else
					// ---- undefined label first ----
					lblOut[nWritten] = 0;
					costOut[nWritten] = undefinedCost;
					++nWritten;

					if (nFD == 0) {
						++numUnobservedFaces;
						continue;
					}
#endif

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

						lblOut[nWritten] = lbl;
						costOut[nWritten] = dataCost;
						++nWritten;
					}
					ASSERT(nWritten == inference.NumLabels((LBPInference::NodeID)f));
				}
				TEX_PROFILE_END(_tFvsLbpBuild, "FaceViewSelection: LBP build graph+datacost");

				// Floor for the per-sweep "undefined=" count printed by the LBP below.
				TEXTURE_DIAG("[LBP-DIAG] faces=%u unobserved=%zu (%.2f%%) -- faces with no candidate view; 'undefined' cannot go below this",
					(unsigned)numFaces, numUnobservedFaces,
					numFaces ? 100.0 * (double)numUnobservedFaces / (double)numFaces : 0.0);

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
					TEXTURE_DIAG("Label propagation to invisible faces: %d passes", propagationPasses);
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

	// PARALLEL-EFFICIENCY CAP (see TEXTURE_SEAM_MAX_WORKERS). Applied BEFORE the memory
	// cap because it also shrinks the memory model's top-N window: fewer workers can only
	// hold fewer patches at once. This pass is bandwidth-bound and measurably regresses
	// past the physical core count (32 workers were slower than 14 on a 7950X), which is
	// a scaling limit no memory budget can express.
	{
		const int nScalingCap = (TEXTURE_SEAM_MAX_WORKERS > 0 ? (int)TEXTURE_SEAM_MAX_WORKERS : GetPhysicalCoreCount());
		if (nScalingCap > 0 && nScalingCap < T) {
			TEXTURE_DIAG("LocalSeamLeveling: threads %d -> %d (bandwidth-bound pass capped at %s)",
				T, nScalingCap, (TEXTURE_SEAM_MAX_WORKERS > 0 ? "TEXTURE_SEAM_MAX_WORKERS" : "physical cores"));
			T = nScalingCap;
		}
	}

	// MEMORY-BUDGET THREAD CAP (see TEXTURE_SEAM_MEM_BUDGET_GB): the per-thread patch
	// buffers below (2x Image32F3 + mask + Poisson scratch) can, across T workers, be the
	// transient that sets the texturing peak on res-0. Cap T so the footprint of the T
	// largest patches stays under the budget. Fewer workers -> identical output, just less
	// parallelism. The budget is measured at runtime by default, so a machine with headroom
	// is not throttled by a fixed number sized for the smallest box.
	{
		constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
		uint64_t budgetBytes;   // 0 == no cap
		String strBudgetSrc;
		if (TEXTURE_SEAM_MEM_BUDGET_GB < 0) {
			// cap disabled by build option
			budgetBytes = 0;
		} else if (TEXTURE_SEAM_MEM_BUDGET_GB > 0) {
			// fixed build-time budget
			budgetBytes = (uint64_t)TEXTURE_SEAM_MEM_BUDGET_GB * GB;
			strBudgetSrc = String::FormatString("fixed %.1f GB", budgetBytes / (double)GB);
		} else {
			// AUTO: size the budget from the free physical RAM as it is RIGHT NOW, i.e.
			// after the source images and the texture patches are already resident, so
			// what we measure is genuine headroom for this pass.
			const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
			if (memInfo.totalPhysical == 0) {
				// OS query failed -> conservative fixed fallback rather than "unlimited"
				budgetBytes = (uint64_t)TEXTURE_SEAM_MEM_BUDGET_FALLBACK_GB * GB;
				strBudgetSrc = String::FormatString("fallback %.1f GB (memory query failed)", budgetBytes / (double)GB);
			} else {
				budgetBytes = (uint64_t)((double)memInfo.freePhysical * TEXTURE_SEAM_MEM_FREE_FRACTION);
				// on a constrained address space the virtual headroom can bind first
				if (memInfo.freeVirtual > 0) {
					const uint64_t virtualBytes = (uint64_t)((double)memInfo.freeVirtual * TEXTURE_SEAM_MEM_FREE_FRACTION);
					if (budgetBytes > virtualBytes)
						budgetBytes = virtualBytes;
				}
				if (budgetBytes < (uint64_t)TEXTURE_SEAM_MEM_BUDGET_MIN_GB * GB)
					budgetBytes = (uint64_t)TEXTURE_SEAM_MEM_BUDGET_MIN_GB * GB;
				if (TEXTURE_SEAM_MEM_BUDGET_MAX_GB > 0 && budgetBytes > (uint64_t)TEXTURE_SEAM_MEM_BUDGET_MAX_GB * GB)
					budgetBytes = (uint64_t)TEXTURE_SEAM_MEM_BUDGET_MAX_GB * GB;
				strBudgetSrc = String::FormatString("auto %.1f GB (%.1f of %.1f GB physical free x %.2f)",
					budgetBytes / (double)GB, memInfo.freePhysical / (double)GB,
					memInfo.totalPhysical / (double)GB, (double)TEXTURE_SEAM_MEM_FREE_FRACTION);
			}
		}
		if (budgetBytes > 0) {
			// FOOTPRINT MODEL: SUM OF THE T LARGEST PATCHES, not T x the largest one.
			// Patch areas are extremely skewed -- a real 109-image res-0 set had 9910
			// patches totalling 276.5 Mpx (27.9 kpx MEAN) with only 4 patches over 4 Mpx
			// and a single 10.2 Mpx outlier. Charging every worker the size of that
			// outlier assumes all T workers sit on it simultaneously, which cannot happen
			// when there are only 4 big patches: the old T x max model predicted 34 GB for
			// 14 workers while the process high-water mark did not move at all (peakWS
			// 7.58 GB before AND after the pass; curWS delta ~0.8 GB) -- over-estimating
			// by well over an order of magnitude and needlessly capping 32 cores to 14.
			// The sum of the T largest areas is still a rigorous worst case: the workers
			// hold at most T patches at once, so the costliest possible set is the T
			// biggest. It is also what bounds the RETAINED pseam:: scratch, whose worst
			// case is likewise "each of the T largest landed on a different thread". And
			// it is never looser than the old bound, since sum(top T) <= T * max always.
			std::vector<uint64_t> areas;
			areas.reserve(numPatches);
			for (uint32_t p = 0; p < numPatches; ++p) {
				const TexturePatch& tp = texturePatches[p];
				if (tp.label == NO_ID || tp.rect.width <= 0 || tp.rect.height <= 0)
					continue;
				areas.push_back((uint64_t)tp.rect.width * (uint64_t)tp.rect.height);
			}
			if (!areas.empty()) {
				// only the T largest can ever be resident at once
				const size_t nTop = MINF((size_t)MAXF(T, 1), areas.size());
				std::partial_sort(areas.begin(), areas.begin()+nTop, areas.end(),
					[](uint64_t a, uint64_t b) { return a > b; });
				// Bytes per patch pixel, sized to the WORST CASE: every one of the top-N
				// patches resident on its own worker at once. Live buffers are ~77 B/px
				// (2x Image32F3 = 24, mask = 1, Poisson stencil = 16, 6 float x/b = 24,
				// red/black interior ~8, indices = 4); of that, the pseam:: vectors (~52)
				// keep their largest-patch capacity across patches while the cv::Mats
				// (~25) are resized per patch. 96 B/px covers the full 77 plus ~25% for
				// the realloc churn that transiently holds OLD+NEW during a resize.
				//
				// This was 256 B/px, which was calibrated against the OLD T x max area
				// model and was really absorbing THAT model's error. With the area term
				// fixed the two conservatisms compounded and the estimate ran ~15x over
				// on every scene measured (working-set delta across LocalSeamLeveling,
				// read before ReleaseSeamScratch):
				//   109 img, top-14 patches   ->  est 11.5 GB, actual 0.79 GB
				//   168 img, top-16 patches   ->  est 20.8 GB, actual 1.18 GB
				//   192 img, top-16 = 111.8 Mpx -> est 28.6 GB, actual 2.04 GB (~18 B/px)
				// That last one sat at 90% of a 31.9 GB budget, i.e. one slightly larger
				// scene (or a busier machine) away from throttling workers back down and
				// undoing the 2 -> 16 worker win that cut this stage from 27.8 s to 10.2 s.
				// 96 B/px still bounds the rigorous worst case; it just stops double-
				// counting. Raise it if a real scene is ever measured above ~77 B/px.
				constexpr uint64_t bytesPerPixel = 96ull;
				// largest worker count whose top-N area sum stays inside the budget (>=1)
				int cap = 0;
				uint64_t sumArea = 0;
				for (size_t i = 0; i < nTop; ++i) {
					const uint64_t nextSum = sumArea + areas[i];
					if (nextSum * bytesPerPixel > budgetBytes)
						break;
					sumArea = nextSum;
					++cap;
				}
				if (cap < 1)
					cap = 1;
				const int finalT = (cap < T ? cap : T);
				// footprint estimate for the workers actually used (cap may exceed T)
				uint64_t finalArea = 0;
				for (size_t i = 0; i < (size_t)finalT && i < nTop; ++i)
					finalArea += areas[i];
				// Always log the decision (even when NOT capping) so the memory budget is
				// observable in the profile trace.
				TEXTURE_DIAG("LocalSeamLeveling: threads %d -> %d (%u patches, largest %llu px, ~%.2f GB est for %d workers, budget %s)",
					T, finalT, (unsigned)areas.size(), (unsigned long long)areas[0],
					finalArea * bytesPerPixel / (double)GB, finalT, strBudgetSrc.c_str());
				T = finalT;
			}
		} else {
			TEXTURE_DIAG("LocalSeamLeveling: threads %d (memory budget cap disabled)", T);
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

// TEXTURE_ATLAS_COVERAGE_GUTTER: rebuild the atlas gutter from FACE COVERAGE instead
// of from "is this texel still colEmpty".
//
// THE CRACK THIS FIXES. Each patch is copied into the atlas as a whole RECTANGLE of
// raw source-image pixels -- `patch.copyTo(textureDiffuse(rect))` -- not as a masked
// set of triangles. The rect is the face bounding box grown by `border`(2), so every
// texel of it lands in the atlas including the ones NO face covers: whatever the camera
// happened to see just past the mesh edge (water, sky, background, an unrelated
// surface). Patches are then shelf-packed with ZERO spacing (`shelfX += w`), so the
// atlas ends up wall-to-wall raw camera pixels with no colEmpty between patches at all.
//
// Consequences, all of them visible exactly on the synthetic/observed boundary:
//   * The seam feather paints only INSIDE the band triangles, so the raw pixels sitting
//     one texel outside them are never touched. GPU bilinear at a triangle edge -- and
//     the INTER_AREA final resize, which averages a whole 1/scale^2 neighbourhood --
//     mix those foreign pixels into the seam. That is a hairline of unrelated colour
//     tracing the entire boundary, and no amount of feather width or blur radius can
//     remove it because the feather never writes there.
//   * The old colEmpty-based flood could not repair it either: with the rects wall to
//     wall there is no colEmpty left to flood, so the flood only ever touched the
//     unused remainder of the atlas.
//
// With this on, a coverage mask is rasterised from the FINAL per-face texcoords (real
// patches and synthesized fill tiles alike), every texel no face covers is reset to
// colEmpty, and the existing frontier flood then regrows the gutter from the nearest
// covered texel. Every texel a filter kernel can reach outside a triangle is now that
// triangle's own final colour, so there is nothing foreign left to bleed in.
// 0 = previous behaviour (flood only what was already empty).
#ifndef TEXTURE_ATLAS_COVERAGE_GUTTER
#define TEXTURE_ATLAS_COVERAGE_GUTTER 1
#endif
// Conservative outset, in atlas pixels, applied when rasterising the coverage mask.
// A texel whose centre is up to this far OUTSIDE a triangle still counts as covered, so
// rounding at the triangle edge can never clear a texel the renderer samples at base
// resolution. Keep it under 1: larger values preserve more of the raw rect and give the
// bleed its hairline back.
#ifndef TEXTURE_ATLAS_COVERAGE_OUTSET_PX
#define TEXTURE_ATLAS_COVERAGE_OUTSET_PX 0.75f
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

// NOT static: SceneReconstruct.cpp declares this extern so ResolveAtlasMaxDim() can
// size its atlas face budget from the same hardware limit this file will pack into.
// Keeping it file-local let the two stages disagree by 4x in face capacity on a
// 32768-capable host.
//
// This is now the HOST LIMIT ONLY, not the pack decision. GenerateTexture resolves the
// atlas dimension through ResolveAtlasMaxDimEx (pin > this probe > fallback), because
// calling this directly there ignored OPENMVS_ATLAS_MAX_DIM and let an env pin move the
// geometry stages' face cap without moving the atlas. Remaining direct callers should
// want the host maximum specifically -- e.g. [ATLAS-FINAL], which reports whether the
// finished atlas is sampleable on THIS machine, a genuine host question and one whose
// answer is legitimately "no" when the deliverable targets a better GPU.
int GetOpenGLMaxTextureSize()
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

// TEXTURE_ATLAS_ADAPTIVE_FIT: when the patches (at native resolution) do not fit
// within nMaxTextureSize, shrink only the outlier patches -- those captured at a
// source resolution denser than their mesh footprint needs -- instead of growing the
// atlas past a real GPU/memory ceiling. This replaces an escape valve that used to
// double the atlas dimension past a "hard" cap toward a multi-hundred-GB allocation
// on any scene whose total patch area exceeded one nMaxTextureSize square; that
// valve is gone, so a scene which cannot be adaptively fit now fails loudly instead.
// 0 = restore the old "just keep growing" behaviour (kept for A/B only; do not ship
// with this off, it defeats --max-texture-size as a real ceiling).
#ifndef TEXTURE_ATLAS_ADAPTIVE_FIT
#define TEXTURE_ATLAS_ADAPTIVE_FIT 1
#endif
// Fraction of nMaxTextureSize^2 the adaptive-fit target aims for on its first
// attempt, leaving slack for the shelf packer's imperfect packing efficiency
// (rects rarely tile a square with zero waste). Tightened automatically on retry.
//
// This fraction IS the deliverable's texture resolution. AdaptiveFitPatches caps every
// patch to a uniform texel density chosen so the total lands exactly on this budget --
// MEASURED on a 1940-view aerial site, realized area came in at 100% of budget, i.e.
// the margin binds directly and the leftover 15% of the atlas is never used. Texel size
// scales as 1/sqrt(margin), so 0.85 -> 0.97 is ~6% finer texture for free.
//
// Start HIGH and step down gently rather than low and collapse. Each attempt shrinks
// monotonically from the previous attempt's state, so arriving at 0.82 through four
// tries is byte-identical to going straight there -- a failed high attempt costs only
// repack time, never quality. The old ladder (0.85 x 0.85 per retry -> 0.72, 0.61)
// could only ever undershoot, and never tested whether the packer had room to spare.
//
// LOWERED 0.97 -> 0.93 after MEASURING the packer instead of guessing at it. On a
// 16384 px host the ladder settled at 0.8245 and produced:
//     patches placed  0.8245 x 268.4 Mpx = 221.3 Mpx
//     atlas built     14877 x 15609      = 232.2 Mpx   -> 95.3% fill
// So shelf packing achieves ~95% here, not the 60-85% the step comment below
// assumed -- and that makes 0.97 STRUCTURALLY UNREACHABLE, since 0.97 / 0.953 > 1:
// the first rung can never pass, on any scene that packs this well. It was not a
// high starting guess, it was a wasted attempt, and the coarse step then jumped
// clean over the feasible band to 0.8245.
//
// 0.93 sits just under the measured ceiling, so a well-packing scene lands on the
// FIRST attempt and keeps 0.93/0.8245 = 1.128x the texel area, i.e. ~6% finer
// linear detail. A scene that packs worse simply steps down (see below).
//
// Caveat, stated plainly: 95.3% is one scene. If other scenes pack materially
// worse, 0.93 costs an extra repack pass or two and lands wherever they can. It
// cannot cost quality -- attempts shrink monotonically, so reaching 0.82 through
// three tries is byte-identical to going straight there.
//
// VERIFY on the next run with the ungated "[ATLAS] built WxH px, ceiling N px"
// line: W*H should rise toward 268 Mpx, and the "keeps 0.24x the pixels it asked
// for" figure should improve by roughly the same 1.13x.
#ifndef TEXTURE_ATLAS_FIT_MARGIN
#define TEXTURE_ATLAS_FIT_MARGIN 0.93
#endif
// Multiplicative step between adaptive-fit attempts.
//
// RAISED 0.95 -> 0.85 (and attempts 4 -> 8). The margin is not a safety fudge, it is a
// PACKING-EFFICIENCY estimate: 0.97 assumes a near-perfect pack, whereas shelf packing tens
// of thousands of heterogeneous rects realistically achieves 60-85% occupancy. The old
// ladder (0.97 / 0.92 / 0.88 / 0.83) never reached that range, so on a scene whose total
// patch AREA fit while the PACK did not, no attempt ever trimmed anything and the stage
// failed. 0.85 over 8 attempts walks 0.97 down to ~0.31, which covers any real packer.
// Quality is unaffected on scenes that pack at the first try; a failed high attempt costs
// only repack time.
//
// RESTORED 0.85 -> 0.95. The "60-85% occupancy" premise above was an estimate that has
// now been measured, and it was wrong: this packer achieved 95.3% (see the arithmetic on
// TEXTURE_ATLAS_FIT_MARGIN). The reasoning that motivated 0.85 -- reach the real
// occupancy range so SOMETHING trims -- still holds, but the real range is near 0.95, and
// a 0.85 step overshoots straight through it: from a 0.93 start the second rung is 0.883
// and the third 0.839, so a scene whose true limit is 0.91 gives up 8% of its texel area
// to step granularity alone.
//
// 0.95 over 8 attempts walks 0.93 down to 0.65, which still clears any plausible packer
// while landing within ~5% of the true limit instead of ~15%. The extra rungs cost repack
// time only on scenes that actually need them, and only until one passes.
#ifndef TEXTURE_ATLAS_FIT_MARGIN_STEP
#define TEXTURE_ATLAS_FIT_MARGIN_STEP 0.95
#endif
// Number of shrink-and-repack attempts. Each costs one pack pass, which is cheap next to
// failing the whole stage after 28 s of texturing.
//
// RAISED 8 -> 12 to hold the ladder's FLOOR while the step got finer. What matters here
// is not the attempt count but where the last rung lands, because exhausting the ladder
// is what failed the stage outright:
//     0.97 x 0.85^7  = 0.31   (old start, old step, 8 attempts)
//     0.93 x 0.95^7  = 0.65   (new start, new step, 8 attempts)  <-- would have regressed
//     0.93 x 0.95^11 = 0.53   (new start, new step, 12 attempts)
// 0.53 stays below the 60% worst case the step comment cites, so the finer search buys
// precision near the top of the range without giving up reach at the bottom. The added
// rungs only ever execute on a scene that needs them, and only until one passes -- a
// scene that packs on the first attempt sees no change at all.
#ifndef TEXTURE_ATLAS_FIT_ATTEMPTS
#define TEXTURE_ATLAS_FIT_ATTEMPTS 12
#endif
// When --max-texture-size is 0 (deliberately unbounded, no GPU ceiling to respect),
// the atlas still has to be built in memory, so derive a RAM-safe packing dimension
// from this fraction of total physical RAM instead -- same shape as
// MESHOPT_MEM_TARGET_FRACTION in SceneRefine.cpp.
#ifndef TEXTURE_ATLAS_MEM_TARGET_FRACTION
#define TEXTURE_ATLAS_MEM_TARGET_FRACTION 0.85
#endif
// Used only if the OS memory query fails (returns zeros).
#ifndef TEXTURE_ATLAS_MEM_TARGET_FALLBACK_GB
#define TEXTURE_ATLAS_MEM_TARGET_FALLBACK_GB 8
#endif
// Predictive memory pre-flight on the atlas, for the case the fraction above does NOT
// cover: an EXPLICIT --max-texture-size (or OPENMVS_ATLAS_MAX_DIM pin).
//
// The RAM-safe derivation above runs only when the size is <= 0. Pin 32768 and nothing
// checks anything: the atlas is 32768^2 * 3 = 3.2 GB in one Image8U3, on top of a
// 368-view working set already resident, and the first sign of trouble is an allocation
// failure part-way through a multi-hour pipeline.
//
// WHY IT DEGRADES RATHER THAN FAILS. GlobalMapper retries TextureMesh at a lower
// resolution level when it sees low memory, but it detects that by OBSERVING
// mMinAvailMem < 1 GB during the run (ImageToCloudConverter.cpp:8164) -- it does not read
// the log. A clean early failure would leave memory untouched, so no retry would fire and
// the pipeline would simply die. Reducing the dimension and continuing also matches what
// the <= 0 path already does, so the two cases now behave alike.
//
// The reduction is reported at normal verbosity and names the RAM that WOULD have been
// needed, because on a machine building deliverables for better GPUs a silent downgrade
// to 16k is the exact failure this whole atlas-dimension effort exists to prevent -- the
// operator has to know to build on a bigger box.
//
// 0 disables the check (restores the previous unguarded behaviour for explicit sizes).
#ifndef TEXTURE_ATLAS_MEM_PREFLIGHT
#define TEXTURE_ATLAS_MEM_PREFLIGHT 1
#endif
// Bytes of peak TRANSIENT residency per atlas texel, used to size the pre-flight.
//
// Derived, not guessed, from where the atlas path actually peaks:
//
//   3.0  textureDiffuse itself (Image8U3), allocated at textureDiffuse.create() right
//        after the pack -- note this is BEFORE any source image is released, so the
//        freePhysical the pre-flight reads already accounts for the images correctly.
//   3.1  newTex during the data-color bake, which builds a SECOND full atlas
//        (oldRows + extraRows, measured 1.02x) and only then does
//        `textureDiffuse = newTex`. Both are resident across that copy -- this is the
//        real peak of the whole atlas path, and it is invisible in the [MEM] lines
//        because they sample after the assignment. Conditional on unobserved faces
//        existing, which is the norm (100,917 of 3.73M on a 32768 run).
//   ~0.4 headroom for the gutter clear-and-regrow, which peaks LOWER than the bake
//        (one atlas + a per-texel mask + a frontier queue ~ 4.2 B/texel) and so is
//        already covered by the figure above.
//
// MEASURED on a 32768 run (1071.3 Mpx): atlas-phase curWS settled at 3.87 GB = 3.6
// B/texel, but that is sampled AFTER the bake released the old buffer -- it is the
// steady state, not the peak. Sizing to 3.6 or 4.0 would approve an atlas that then OOMs
// mid-bake on a machine without slack. That run never showed it because its process peak
// (11.25 GB) had already been set by the source images during seam levelling, so the
// double-atlas moment fitted underneath.
//
// Erring high costs atlas dimension; erring low costs an OOM in the last minute of a
// multi-hour pipeline. Kept deliberately loose for that reason.
#ifndef TEXTURE_ATLAS_MEM_BYTES_PER_TEXEL
#define TEXTURE_ATLAS_MEM_BYTES_PER_TEXEL 6.5
#endif

unsigned MeshTexture::AdaptiveFitPatches(uint64_t budgetAreaPixels, int maxPatchDim)
{
#if !TEXTURE_CROP_IMAGES
	// patches share the full source image buffer -- there is no private, independently
	// resizable crop to shrink, so this is a no-op; the caller's retry loop will exhaust
	// itself and fail loudly, which is honest given there is really nothing to trim here.
	return 0;
#else
	const unsigned numPatches = texturePatches.GetSize();

	// Resize one patch's crop to (newW,newH) and rescale its texcoords by the REALIZED
	// integer ratio (not the theoretical target), so rounding to whole pixels can never
	// leave a texcoord outside the actually-resized buffer. Shared by both steps below.
	const auto ResizePatch = [this](TexturePatch& tp, int newW, int newH) {
		newW = std::max(1, newW);
		newH = std::max(1, newH);
		if (newW >= tp.rect.width && newH >= tp.rect.height)
			return;
		if (!tp.image.empty()) {
			cv::Mat resized;
			cv::resize(tp.image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
			tp.image = resized;
		}
		const float sx = (float)newW / (float)tp.rect.width;
		const float sy = (float)newH / (float)tp.rect.height;
		for (const FIndex idxFace : tp.faces) {
			TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
			for (int v = 0; v < 3; ++v) {
				texcoords[v].x *= sx;
				texcoords[v].y *= sy;
				// A trimmed patch's texcoords must land inside its new, smaller rect --
				// a mismatch here means a face would sample the wrong pixels (silently
				// wrong texture), not just lower detail, so this is checked, not assumed.
				ASSERT(texcoords[v].x >= -0.5f && texcoords[v].x <= (float)newW + 0.5f &&
					texcoords[v].y >= -0.5f && texcoords[v].y <= (float)newH + 0.5f);
			}
		}
		tp.rect = cv::Rect(0, 0, newW, newH);
	};

	unsigned numScaled = 0;

	// STEP 1: unconditionally clamp any SINGLE patch whose own width or height alone
	// exceeds maxPatchDim. No area-budget/density logic below can rescue this case:
	// PackShelfFast rejects the WHOLE pack the instant one rect cannot fit the atlas in
	// EITHER orientation (see its unplaceable check), regardless of how small every other
	// patch is -- a pathologically elongated patch (low density/area, but one dimension
	// wider than the atlas, e.g. a long thin strip) would otherwise sail through the
	// area-based water-filling in step 2 untouched and still fail to pack.
	if (maxPatchDim > 0) {
		for (unsigned p = 0; p < numPatches; ++p) {
			TexturePatch& tp = texturePatches[p];
			if (tp.rect.width <= maxPatchDim && tp.rect.height <= maxPatchDim)
				continue;
			const double s = (double)maxPatchDim / (double)std::max(tp.rect.width, tp.rect.height);
			ResizePatch(tp, (int)std::lround(tp.rect.width * s), (int)std::lround(tp.rect.height * s));
			++numScaled;
		}
	}

	// STEP 2: density-based water-filling over total area, reading the (possibly
	// step-1-clamped) rects.
	std::vector<double> areaPx(numPatches, 0.0);
	std::vector<double> areaWorld(numPatches, 0.0);

	// Per-patch mesh-space (world) area, summed over each patch's OWN face list in a
	// fixed index order. Patches are independent (disjoint face lists, distinct output
	// slots), so parallelizing ACROSS patches is safe; the accumulation WITHIN a patch
	// stays sequential and order-fixed, so the result does not depend on the OpenMP
	// schedule -- required, since this feeds a decision (which patches get trimmed and
	// by how much) that must be identical for identical input, not just "close enough".
#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
#endif
	for (int_t p = 0; p < (int_t)numPatches; ++p) {
		const TexturePatch& tp = texturePatches[(uint32_t)p];
		if (tp.rect.width <= 0 || tp.rect.height <= 0)
			continue;
		areaPx[(size_t)p] = (double)tp.rect.width * (double)tp.rect.height;
		double world = 0.0;
		for (const FIndex idxFace : tp.faces) {
			const Face& face = faces[idxFace];
			world += ComputeTriangleArea(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
		}
		areaWorld[(size_t)p] = world;
	}

	double totalAreaPx = 0.0, maxDensity = 0.0;
	for (unsigned p = 0; p < numPatches; ++p) {
		totalAreaPx += areaPx[p];
		if (areaWorld[p] > 1e-12) {
			const double d = areaPx[p] / areaWorld[p];
			if (d > maxDensity)
				maxDensity = d;
		}
	}
	if (totalAreaPx <= (double)budgetAreaPixels || maxDensity <= 0.0)
		// Area fits the budget -- but the CALLER only gets here after a pack FAILURE, so
		// this does not mean the pack will now succeed: packing efficiency, not total area,
		// is what failed. The caller must respond by lowering the budget (margin) and
		// calling again, NOT by giving up on a 0 return.
		return numScaled; // (step 1 may still have clamped an outlier dimension)

	// Binary search the density ceiling D: total(D) = sum(min(areaPx[p], D*areaWorld[p]))
	// is monotonically non-decreasing in D, so bisection converges directly to the
	// largest D that keeps total area under budget. Capping every patch denser than D
	// down to exactly D (and leaving everything else untouched) is the minimal set of
	// patches touched for a given area reduction -- well-matched patches never lose
	// resolution, only genuine outliers do.
	double lo = 0.0, hi = maxDensity;
	for (int iter = 0; iter < 40; ++iter) {
		const double mid = 0.5 * (lo + hi);
		double total = 0.0;
		for (unsigned p = 0; p < numPatches; ++p) {
			const double cap = mid * areaWorld[p];
			total += (areaWorld[p] > 1e-12 && cap < areaPx[p]) ? cap : areaPx[p];
		}
		if (total > (double)budgetAreaPixels)
			hi = mid;
		else
			lo = mid;
	}
	const double D = lo;

	for (unsigned p = 0; p < numPatches; ++p) {
		if (areaWorld[p] <= 1e-12)
			continue;
		const double targetArea = D * areaWorld[p];
		if (targetArea >= areaPx[p])
			continue; // at or below the ceiling already -- untouched
		TexturePatch& tp = texturePatches[p];
		const double scale = std::sqrt(targetArea / areaPx[p]);
		ResizePatch(tp, (int)std::lround(tp.rect.width * scale), (int)std::lround(tp.rect.height * scale));
		++numScaled;
	}

	// REALIZED area, not the theoretical target: ResizePatch rounds both dimensions to
	// whole pixels and refuses to grow, so the outcome can land under the budget. How
	// far under is the question that matters -- every pixel between the realized total
	// and the budget is atlas capacity the deliverable never gets, and unlike the
	// budget it cannot be recovered later (ResizePatch destroys the source pixels).
	//
	// D is a texel density in world units, so 1/sqrt(D) is the world size of one texel:
	// the achievable texture GSD for this mesh at this atlas size. Compare it against
	// the imagery GSD to see how much resolution the atlas ceiling is costing.
	// Log-only (the fit itself is already applied above), so it rides TEXTURE_DIAG --
	// the fact that a downscale HAPPENED is reported ungated by the caller.
	if (TEXTURE_DIAG_ENABLED()) {
		double realizedPx = 0.0, totalWorld = 0.0;
		for (unsigned p = 0; p < numPatches; ++p) {
			realizedPx += (double)texturePatches[p].rect.width * (double)texturePatches[p].rect.height;
			totalWorld += areaWorld[p];
		}
		// No percent sign anywhere in this format. A literal percent is not portable
		// across this codebase's log macros: plain %% rendered as garbage here (its %
		// merged with the following " of" into an octal conversion that ate the next
		// argument), while %%%% rendered as a literal "%%". A ratio says the same thing
		// and cannot be misparsed.
		TEXTURE_DIAG("[ATLAS-FIT] budget=%.1f Mpx | patch area %.1f -> %.1f Mpx (x%.3f of budget)"
			" | density ceiling D=%.1f px/unit^2 -> texel %.4g world units | surface %.4g unit^2",
			(double)budgetAreaPixels * 1e-6, totalAreaPx * 1e-6, realizedPx * 1e-6,
			budgetAreaPixels ? realizedPx / (double)budgetAreaPixels : 0.0,
			D, (D > 0.0) ? 1.0 / std::sqrt(D) : 0.0, totalWorld);
	}
	return numScaled;
#endif // TEXTURE_CROP_IMAGES
}

// The atlas-dimension policy (env pin > host GL probe > built-in fallback) lives in
// SceneReconstruct.cpp, which sizes the DECIMATION face cap from it. Declared locally
// rather than in a header, matching what ReconstructMesh.cpp and RefineMesh.cpp do for
// this same pair -- and note both that definition and this declaration are at GLOBAL
// scope (SceneReconstruct.cpp does `using namespace MVS`, it is not inside the
// namespace), which is what makes them link.
extern int ResolveAtlasMaxDimEx(int atlasMaxDim, int* pHostLimit, int* pEnvPin);

bool MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize)
{
	// --max-texture-size < 0 => cap the final atlas to the atlas dimension the GEOMETRY
	// stages already sized their face budget against. == 0 keeps its original meaning:
	// UNBOUNDED (no cap). Resolved once here so BOTH the packer's starting dimension
	// and the final "enforce max size" step use it.
	//
	// Routed through ResolveAtlasMaxDimEx rather than calling GetOpenGLMaxTextureSize
	// directly, which is what this did before. That bypassed OPENMVS_ATLAS_MAX_DIM:
	// ReconstructMesh's decimation cap and RefineMesh's audit BOTH honour the env pin,
	// so pinning 32768 on a 16384-limited host sized the mesh for a 16.3M-face atlas
	// while this function packed into 16384 -- the mesh arrived 4x over-dense for the
	// atlas actually built, with nothing saying so. One resolver for all three stages is
	// the only way the pin can mean one thing. (The reverse hazard -- the two stages
	// disagreeing by 4x in face capacity -- is the one already called out at the
	// GetOpenGLMaxTextureSize definition above and at ResolveAtlasMaxDimEx.)
	int atlasHostLimit = 0, atlasEnvPin = 0;
	if (nMaxTextureSize < 0) {
		nMaxTextureSize = ResolveAtlasMaxDimEx(0, &atlasHostLimit, &atlasEnvPin);
		TEXTURE_DIAG("Texture max size < 0 -> atlas %d px (%s; host GL_MAX_TEXTURE_SIZE %d)",
			nMaxTextureSize,
			(atlasEnvPin > 0)    ? "pinned by OPENMVS_ATLAS_MAX_DIM" :
			(atlasHostLimit > 0) ? "host GPU limit" :
			                       "fallback, no GPU reachable",
			atlasHostLimit);
	} else {
		// An explicit (or explicitly-unbounded) --max-texture-size is an operator
		// decision and still wins -- same precedence ResolveAtlasMaxDimEx documents.
		// Probe anyway, purely to CHECK it against the pin below. The probe caches, so
		// asking here costs nothing.
		ResolveAtlasMaxDimEx(0, &atlasHostLimit, &atlasEnvPin);
	}
	// Any disagreement means the mesh was sized for a DIFFERENT atlas than the one about
	// to be packed, by (ratio)^2 in face capacity. Reported at normal verbosity, not on
	// TEXTURE_DIAG: this is a silently wrong deliverable, not a mechanism.
	if (atlasEnvPin > 0 && nMaxTextureSize > 0 && nMaxTextureSize != atlasEnvPin)
		VERBOSE("[ATLAS] warning: packing into %d px but OPENMVS_ATLAS_MAX_DIM pins %d px"
			" -- the mesh face cap was sized for the PINNED atlas, so this mesh carries"
			" %.2fx the faces this atlas can texture; pass --max-texture-size %d to agree",
			nMaxTextureSize, atlasEnvPin,
			((double)atlasEnvPin * atlasEnvPin) / ((double)nMaxTextureSize * nMaxTextureSize),
			atlasEnvPin);
	else if (atlasEnvPin > 0 && nMaxTextureSize == 0)
		VERBOSE("[ATLAS] warning: --max-texture-size 0 (unbounded) but"
			" OPENMVS_ATLAS_MAX_DIM pins %d px -- the mesh face cap was sized for the"
			" pinned atlas while this atlas is bounded only by RAM; the two will not agree",
			atlasEnvPin);
	// The pin exceeding the host is the more dangerous direction and must not pass
	// silently: the atlas is built, then cannot be sampled by an OpenGL viewer.
	if (atlasHostLimit > 0 && nMaxTextureSize > atlasHostLimit)
		VERBOSE("[ATLAS] warning: atlas %d px is LARGER THAN THIS HOST CAN SAMPLE"
			" (GL_MAX_TEXTURE_SIZE %d px) -- the result will not upload to an OpenGL"
			" viewer without a rescale", nMaxTextureSize, atlasHostLimit);
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
		// CROP EXTRACTION DENSITY CAP.
		//
		// Crops are extracted at SOURCE resolution and AdaptiveFitPatches then throws most
		// of it away to fit the atlas. MEASURED on a 942,700 unit^2 site: 2,792.9 Mpx of
		// crops (~8.4 GB) extracted to fill a 247.4 Mpx atlas -- 91% decoded, rectified and
		// discarded, and the peak memory is paid in full.
		//
		// Extracting at a bounded multiple of the density the atlas can actually hold is
		// VISUALLY NEUTRAL, not merely cheaper: the two-stage path is supersampling, and
		// INTER_AREA from a 2x oversampled source is indistinguishable from the same filter
		// over an 11x oversampled one. The benefit saturates at ~2x. (It is NOT neutral at
		// small ratios -- at the 1.77x downscale of a different scene, extract-then-filter
		// genuinely antialiases better than a single pass, which is why this is a CAP at 2x
		// rather than extraction straight at target density.)
		//
		// Safety margin: the cap is set from capacity/totalWorldArea, which UNDERSTATES the
		// density AdaptiveFitPatches finally settles on (patches already below the ceiling
		// are left untouched, so the bisection lands higher). Measured ratios of actual to
		// this estimate: 1.00 and 1.32. The 2x linear cap is 4x in area -- comfortably clear
		// of both, so this can only ever remove detail the atlas could never have shown.
		std::vector<float> patchWorldArea;
		float cropMaxDensity = FLT_MAX;   // px per world unit, LINEAR
#if TEXTURE_CROP_IMAGES
		if (TEXTURE_CROP_OVERSAMPLE > 0.0) {
			patchWorldArea.assign(texturePatches.GetSize(), 0.f);
			double totalWorld = 0.0;
			for (uint32_t p = 0; p < texturePatches.GetSize(); ++p) {
				double a = 0.0;
				for (const FIndex f : texturePatches[p].faces) {
					const Face& fc = faces[f];
					a += ComputeTriangleArea(vertices[fc[0]], vertices[fc[1]], vertices[fc[2]]);
				}
				patchWorldArea[p] = (float)a;
				totalWorld += a;
			}
			const double dim = (nMaxTextureSize > 0) ? (double)nMaxTextureSize : 16384.0;
			const double capacity = dim * dim * TEXTURE_ATLAS_FIT_MARGIN;
			if (totalWorld > 1e-9)
				cropMaxDensity = (float)(TEXTURE_CROP_OVERSAMPLE * std::sqrt(capacity / totalWorld));
			TEXTURE_DIAG("[CROP-CAP] atlas capacity %.1f Mpx over %.4g unit^2 -> atlas density"
				" %.2f px/unit; extracting at up to %.1fx that = %.2f px/unit",
				capacity * 1e-6, totalWorld, std::sqrt(capacity / std::max(1e-9, totalWorld)),
				(double)TEXTURE_CROP_OVERSAMPLE, cropMaxDensity);
		}
#endif

		const int T = std::max(1, omp_get_max_threads());
		std::vector<Image8U3> reloadScratch((size_t)T);
		// RELOAD FAILURE IS FATAL, NOT SKIPPABLE.
		//
		// This used to `continue` on a failed decode, leaving that view's patches with
		// empty crops. Downstream silently skips an empty patch, so the atlas SHIPS with
		// black holes where real texture belongs, and every later stage reports success.
		// That is the worst possible failure mode: a wrong deliverable that looks like a
		// finished one.
		//
		// And it is not an exotic case. This branch only runs BECAUSE the image set did
		// not fit the RAM budget, so a decode allocation is exactly what fails first on a
		// box that is short of memory -- the silent path is the likely one. Retry (a
		// transient allocation failure usually clears once this thread drops its own
		// scratch, the largest block it holds), then abort with the file named.
		unsigned nReloadFailed(0);
		String strFirstFailed;
#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(T)
#endif
		for (int li = 0; li < (int)usedLabels.size(); ++li) {
			const uint32_t label = usedLabels[(size_t)li];
			Image& imageData = images[label];
			Image8U3& scratch = reloadScratch[(size_t)omp_get_thread_num()];

			// When the whole image set fit in RAM (bKeepImagesResident) ListCameraFaces
			// never released the pixels, so crop straight out of them -- this is the entire
			// point of that decision: it removes a second full decode pass over every image
			// (~25 s on a 367-image 18.6 MP set). Same pixels, same crops, just not re-read.
			const bool bResident = bKeepImagesResident && !imageData.image.empty();
			if (!bResident) {
				// decode the full source image into the reused scratch (create() reuses the
				// buffer when the size matches -> no per-image allocation churn)
				unsigned level(nResolutionLevel);
				const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
				bool bDecoded(Image::ReadImage(imageData.name, scratch) != NULL);
				for (int nRetry = 0; !bDecoded && nRetry < 2; ++nRetry) {
					scratch.release(); // hand the allocator back our biggest block first
					bDecoded = (Image::ReadImage(imageData.name, scratch) != NULL);
				}
				if (!bDecoded) {
					#pragma omp critical
					{
						if (nReloadFailed++ == 0)
							strFirstFailed = imageData.name;
					}
					continue; // recorded; the post-loop check turns this into a hard failure
				}
				// match the resolution level ListCameraFaces used (res-0 => imageSize == full
				// original, so no resize and the scratch buffer is reused verbatim)
				if ((unsigned)MAXF(scratch.cols, scratch.rows) > imageSize) {
					const double s = (double)imageSize / (double)MAXF(scratch.cols, scratch.rows);
					cv::resize(scratch, scratch, cv::Size(), s, s, cv::INTER_AREA);
				}
				imageData.UpdateCamera(scene.platforms);
			}
			// resident pixels were already loaded at this resolution level and had their
			// camera updated by ListCameraFaces, so no reload/resize/UpdateCamera needed
			const Image8U3& srcImage = bResident ? imageData.image : scratch;
			if (srcImage.empty()) {
				// Same silent-hole hazard as a failed decode, so account for it the same way.
				#pragma omp critical
				{
					if (nReloadFailed++ == 0)
						strFirstFailed = imageData.name;
				}
				continue;
			}

			const int srcW = srcImage.cols, srcH = srcImage.rows;
			for (const uint32_t p : patchesByLabel[label]) {
				TexturePatch& tp = texturePatches[p];
				cv::Rect r = tp.rect;
				// clamp to the reloaded image (projection already clamped; defensive)
				if (r.x < 0) { r.width += r.x; r.x = 0; }
				if (r.y < 0) { r.height += r.y; r.y = 0; }
				if (r.x + r.width > srcW) r.width = srcW - r.x;
				if (r.y + r.height > srcH) r.height = srcH - r.y;
				if (r.width <= 0 || r.height <= 0) { tp.rect = cv::Rect(0, 0, 0, 0); continue; }
				// Density cap (see [CROP-CAP] above): resize STRAIGHT OUT of the source
				// rather than cloning full-res and shrinking after, so the full-resolution
				// crop is never allocated and the saving shows up in peak memory.
				float cropScale = 1.f;
				if (cropMaxDensity < FLT_MAX && p < patchWorldArea.size() && patchWorldArea[p] > 1e-9f) {
					const float d = FastSqrtS((float)r.width * (float)r.height / patchWorldArea[p]);
					if (d > cropMaxDensity)
						cropScale = cropMaxDensity / d;
				}
				const int newW = std::max(1, (int)std::lround(r.width  * cropScale));
				const int newH = std::max(1, (int)std::lround(r.height * cropScale));
				if (newW < r.width || newH < r.height) {
					cv::resize(srcImage(r), tp.image, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
					// Rescale by the REALIZED integer ratio, matching ResizePatch: rounding to
					// whole pixels must never leave a texcoord outside the resized buffer.
					const float sx = (float)newW / (float)r.width;
					const float sy = (float)newH / (float)r.height;
					for (const FIndex idxFace : tp.faces) {
						TexCoord* tc = faceTexcoords.data() + idxFace * 3;
						for (int v = 0; v < 3; ++v) { tc[v].x *= sx; tc[v].y *= sy; }
					}
				} else {
					tp.image = srcImage(r).clone();         // private full-res crop
				}
				tp.rect = cv::Rect(0, 0, newW, newH);        // rebase to its own origin
			}
			// Crops are cloned, and with TEXTURE_CROP_IMAGES every downstream read goes
			// through PatchSrcImage() -> the patch's own buffer, so the full image is dead
			// weight from here on. Release it now (rather than at the later bulk release)
			// so seam leveling measures the freed RAM as available headroom.
			if (bResident)
				imageData.image.release();
		}
		// free the (few) reused decode buffers now that all crops are cloned out
		reloadScratch.clear();
		reloadScratch.shrink_to_fit();
		if (nReloadFailed > 0) {
			VERBOSE("error: GenerateTexture: FATAL -- %u of %u source images could not be re-decoded"
				" for patch extraction (first: %s). Texturing would ship an atlas with black holes,"
				" so it is aborted instead. This pass re-reads every image because the set did not"
				" fit the memory budget; free RAM or lower --resolution-level and retry.",
				nReloadFailed, (unsigned)usedLabels.size(), strFirstFailed.c_str());
			return false;
		}
		// [CROP-DECOMP] decisive measurement: is the +20 GB after this stage the CROPS
		// themselves, or retained/committed image-decode memory that our buffer
		// management can't reach? Sum the actual bytes held in every patch crop. If this
		// prints ~0.6 GB while curWS jumped ~20 GB, the crops are NOT the hog (it is
		// OpenCV/jpeg decode retention). If it prints ~20 GB, the patch rects really are
		// huge and the packing-area estimate was wrong. Log-only scan -> TEXTURE_DIAG.
		if (TEXTURE_DIAG_ENABLED()) {
			size_t cropBytes = 0, cropPx = 0; unsigned nCrop = 0, nBig = 0; size_t maxPx = 0;
			for (uint32_t p = 0; p < texturePatches.GetSize(); ++p) {
				const Image8U3& im = texturePatches[p].image;
				if (im.empty()) continue;
				const size_t px = (size_t)im.total();
				cropBytes += px * im.elemSize(); cropPx += px; ++nCrop;
				if (px > maxPx) maxPx = px;
				if (px > 4000000) ++nBig; // patches bigger than ~2000x2000
			}
			TEXTURE_DIAG("[CROP-DECOMP] crops=%.2f GB (%u patches, %.1f Mpx total, largest %.1f Mpx, %u patches >4Mpx)",
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

		// `nMaxTextureSize` is a real ceiling (GPU/viewer texture-size limit), not a
		// starting guess -- packing must never exceed it. When --max-texture-size is 0
		// (deliberately unbounded) there is no GPU ceiling to respect, so derive a
		// RAM-safe one instead: an "unbounded" atlas still has to fit in memory to be
		// built at all. See TEXTURE_ATLAS_MEM_TARGET_FRACTION.
		int hardMaxDim = nMaxTextureSize;
		if (hardMaxDim <= 0) {
			constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
			const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
			uint64_t targetBytes = (uint64_t)TEXTURE_ATLAS_MEM_TARGET_FALLBACK_GB * GB;
			if (memInfo.totalPhysical > 0)
				targetBytes = (uint64_t)((double)memInfo.totalPhysical * TEXTURE_ATLAS_MEM_TARGET_FRACTION);
			hardMaxDim = (int)std::sqrt((double)targetBytes / 3.0/*Image8U3*/);
			VERBOSE("[ATLAS] no texture-size limit requested -> capped at %d px so the atlas"
				" fits in memory (%.1f GB target)", hardMaxDim, targetBytes / (double)GB);
		}
		hardMaxDim = std::max(64, hardMaxDim);
		// Everything the patches wanted, before any packing decision. Kept so the
		// downscale below can be stated as a LOSS against this rather than as a bare
		// count of shrunk patches -- how much resolution was given up is the question,
		// and the patch count does not answer it.
		uint64_t wantedAreaPixels = 0;
		for (size_t i = 0; i < (size_t)texturePatches.GetSize(); ++i) {
			const auto& r = texturePatches[(uint32_t)i].rect;
			wantedAreaPixels += (uint64_t)std::max(0, r.width) * (uint64_t)std::max(0, r.height);
		}

#if TEXTURE_ATLAS_MEM_PREFLIGHT
		// Will the atlas this is about to pack actually fit in RAM? See
		// TEXTURE_ATLAS_MEM_PREFLIGHT for why this degrades instead of failing.
		//
		// Predicted from wantedAreaPixels, not from hardMaxDim^2: the patches are already
		// measured at this point, so the honest estimate is what the pack will REACH,
		// capped by the ceiling. Sizing on hardMaxDim^2 alone would refuse a 32768 pin on
		// every small scene that would never have grown near it.
		{
			constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
			const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
			const uint64_t ceilTexels = (uint64_t)hardMaxDim * (uint64_t)hardMaxDim;
			// The pack lands near wanted/margin (measured 249.7 of 268.4 Mpx = 0.93 on a
			// 16384 run, which is TEXTURE_ATLAS_FIT_MARGIN); never above the ceiling.
			const uint64_t predTexels = std::min(ceilTexels,
				(uint64_t)((double)wantedAreaPixels / TEXTURE_ATLAS_FIT_MARGIN));
			const double bytesPerTexel = (double)TEXTURE_ATLAS_MEM_BYTES_PER_TEXEL;
			const uint64_t predBytes = (uint64_t)((double)predTexels * bytesPerTexel);

			// Same safety convention the densify stage uses throughout: hold back
			// max(8% of total, 1 GB). freePhysical rather than totalPhysical because the
			// view working set is ALREADY resident at this point -- that is the whole
			// reason to check here rather than at entry -- so counting it again would
			// double-charge it.
			uint64_t budgetBytes;
			if (memInfo.totalPhysical > 0) {
				const uint64_t safety = std::max(
					(uint64_t)((double)memInfo.totalPhysical * 0.08), 1ull * GB);
				budgetBytes = (memInfo.freePhysical > safety) ? memInfo.freePhysical - safety : 0;
			} else {
				// OS query failed: fall back to the same figure the unbounded path uses.
				budgetBytes = (uint64_t)TEXTURE_ATLAS_MEM_TARGET_FALLBACK_GB * GB;
			}

			TEXTURE_DIAG("[ATLAS-MEM] atlas prediction %.1f Mpx x %.1f B/texel = %.2f GB"
				" | free %.2f GB, budget %.2f GB | ceiling %d px",
				predTexels * 1e-6, bytesPerTexel, predBytes / (double)GB,
				memInfo.freePhysical / (double)GB, budgetBytes / (double)GB, hardMaxDim);

			if (predBytes > budgetBytes && budgetBytes > 0) {
				// Largest dimension that fits, snapped DOWN to nTextureSizeMultiple so the
				// packer's own alignment is preserved, and floored at 64 (the same floor
				// applied to hardMaxDim above) so a hopeless budget still produces
				// something rather than a zero-sized atlas.
				int fitDim = (int)std::sqrt((double)budgetBytes / bytesPerTexel);
				const int mult = (int)std::max(1u, nTextureSizeMultiple);
				fitDim = (fitDim / mult) * mult;
				fitDim = std::max(64, std::min(fitDim, hardMaxDim));
				// Needed RAM for the atlas the operator ASKED for, so the message says what
				// to change about the machine rather than only what was taken away.
				const uint64_t neededBytes = (uint64_t)((double)predTexels * bytesPerTexel)
					+ ((memInfo.totalPhysical > 0)
						? std::max((uint64_t)((double)memInfo.totalPhysical * 0.08), 1ull * GB)
						: 1ull * GB);
				VERBOSE("[ATLAS-MEM] NOT ENOUGH MEMORY for the requested atlas:"
					" %d px needs about %.2f GB (%.1f Mpx x %.1f B/texel) but only %.2f GB"
					" is safely available -- capping the atlas at %d px, which keeps %.2fx"
					" the texture pixels (%.2fx linear detail)."
					" About %.1f GB of free RAM would be needed to build %d px here.",
					hardMaxDim, predBytes / (double)GB, predTexels * 1e-6, bytesPerTexel,
					budgetBytes / (double)GB, fitDim,
					((double)fitDim * fitDim) / (double)predTexels,
					std::sqrt(((double)fitDim * fitDim) / (double)predTexels),
					neededBytes / (double)GB, hardMaxDim);
				hardMaxDim = fitDim;
			}
		}
#endif

		TEX_PROFILE_BEGIN(_tGtPack);
		bool ok = PackShelfReasonablySquare(rects, (int)nTextureSizeMultiple, hardMaxDim, placed, atlasW, atlasH);

		// Set only if patch resolution was actually surrendered, so the report below
		// distinguishes "the atlas came out small because the scene is small" from
		// "the atlas came out small because it was not allowed to be bigger".
		bool bAtlasReduced = false;
		if (!ok && TEXTURE_ATLAS_ADAPTIVE_FIT) {
			// Patches don't fit within hardMaxDim at native resolution. Rather than
			// growing the atlas past a real GPU/memory ceiling, shrink only the patches
			// denser than the mesh geometry actually needs (AdaptiveFitPatches), then
			// retry packing at the SAME hardMaxDim -- the atlas dimension never moves.
			double margin = TEXTURE_ATLAS_FIT_MARGIN;
			for (int attempt = 0; attempt < TEXTURE_ATLAS_FIT_ATTEMPTS && !ok;
				++attempt, margin *= TEXTURE_ATLAS_FIT_MARGIN_STEP) {
				const uint64_t budgetAreaPixels = (uint64_t)((double)hardMaxDim * (double)hardMaxDim * margin);
				const unsigned numFit = AdaptiveFitPatches(budgetAreaPixels, hardMaxDim);
				if (numFit > 0)
					bAtlasReduced = true;
				// Per-attempt detail only. Resolution IS being surrendered here and that
				// must reach the default log, but the loop can run several times and
				// numFit can legitimately be 0 (area fits, pack still failed), so the
				// single authoritative statement is the [ATLAS] summary after the pack
				// succeeds -- it reports the total loss rather than one step of it.
				TEXTURE_DIAG("GenerateTexture: atlas overflow at %d px -- adaptively downscaled %u/%u patches to fit (attempt %d, margin %.2f)",
					hardMaxDim, numFit, texturePatches.GetSize(), attempt + 1, margin);
				for (size_t i = 0; i < (size_t)texturePatches.GetSize(); ++i)
					rects[i] = texturePatches[(uint32_t)i].rect;
				ok = PackShelfReasonablySquare(rects, (int)nTextureSizeMultiple, hardMaxDim, placed, atlasW, atlasH);
				// numFit == 0 is NOT a reason to stop. Its usual cause is that the total
				// patch AREA already fits the budget while the PACK still fails -- see the
				// early-out in AdaptiveFitPatches. Area fitting does not imply packability:
				// FIELD-OBSERVED at 30,321 patches, where the area sat inside 0.97 of the
				// atlas yet shelf packing could not place them, and breaking here reported
				// FATAL after downscaling nothing. The remedy is a SMALLER budget, which is
				// precisely what the next iteration supplies.
#if !TEXTURE_CROP_IMAGES
				// The one genuine "nothing to trim" case: patches share the source image
				// buffer, so there is no private crop to shrink and no budget will change
				// that. Retrying would only repeat the pack.
				break;
#endif
			}
		}

		if (!ok) {
			// Cannot fit even after adaptive downscaling: fail loudly rather than repeat
			// the old behaviour of silently doubling the atlas dimension past a "hard"
			// cap toward a multi-hundred-GB allocation.
			VERBOSE("error: cannot fit the texture within %d px even after reducing patch"
				" resolution; raise --max-texture-size or reduce mesh/image resolution", hardMaxDim);
			return false;
		}

		// THE ATLAS THAT WILL ACTUALLY BE BUILT. Nothing reported this before -- the
		// dimension was decided, used and never named, so a run that quietly produced a
		// quarter of the texture resolution it asked for was indistinguishable from one
		// that did not.
		//
		// The loss is measured on the PATCHES, not on the atlas: atlasW*atlasH includes
		// shelf-packing waste, so comparing the atlas area against the requested area
		// can read above 1.0 on a run that genuinely downscaled. Summing the patch rects
		// after the fit is the honest measure -- AdaptiveFitPatches shrinks those rects
		// in place, so the difference against wantedAreaPixels is exactly the resolution
		// given up. The equivalent square dimension is reported alongside because that
		// is the number that maps onto --max-texture-size; a pixel-area figure does not.
		{
			uint64_t finalAreaPixels = 0;
			for (size_t i = 0; i < (size_t)texturePatches.GetSize(); ++i) {
				const auto& r = texturePatches[(uint32_t)i].rect;
				finalAreaPixels += (uint64_t)std::max(0, r.width) * (uint64_t)std::max(0, r.height);
			}
			if (bAtlasReduced && wantedAreaPixels > 0) {
				const double keptFrac = (double)finalAreaPixels / (double)wantedAreaPixels;
				const int wantedDim = (int)std::ceil(std::sqrt((double)wantedAreaPixels));
				TEXTURE_DIAG("[ATLAS] built %dx%d px, ceiling %d px -- SMALLER THAN WANTED:"
					" full resolution needed about %d px, so the texture keeps %.2fx the"
					" pixels it asked for (about %.2fx linear detail)",
					atlasW, atlasH, hardMaxDim, wantedDim, keptFrac, std::sqrt(keptFrac));
			} else {
				TEXTURE_DIAG("[ATLAS] built %dx%d px, ceiling %d px -- full requested resolution",
					atlasW, atlasH, hardMaxDim);
			}
		}

		// [ATLAS-PATCHES] Where the allocated patch area actually goes, bucketed by patch
		// size. Added to answer ONE question that [ATLAS-GUTTER] raises but cannot settle.
		//
		// ATLAS-GUTTER measures the end state: 51.6% of a 16384^2 atlas carried no triangle
		// on a 7.2M-face run -- 138.3M texels cleared and regrown by the flood. Packing is
		// not the cause (patch rects 249.6 Mpx into a 262.0 Mpx atlas is only 4.7% waste),
		// so the loss is INSIDE the rects. Two very different causes produce that same
		// number and they have opposite fixes:
		//
		//   * MANY TINY PATCHES -- every patch pays a fixed border, so overhead scales with
		//     PERIMETER while payload scales with AREA. A patch near the floor is mostly
		//     border. Fix: fewer, larger charts (the view-labelling smoothness term).
		//   * FEW LARGE PATCHES WITH SLACK BOUNDING BOXES -- an irregular chart inside an
		//     axis-aligned rect leaves the corners empty regardless of size. Fix: chart
		//     shape or rotation, an entirely different change.
		//
		// The discriminator is whether the small buckets hold a share of AREA out of all
		// proportion to their share of FACES. They carry the same information either way,
		// so area-per-face is the payload; a bucket holding 30% of the atlas for 5% of the
		// faces is pure overhead, and that is the signature of fragmentation.
		//
		// Deliberately NOT reporting a computed gutter overhead here: the border width on
		// this path is not a single named constant (the 2 px `gutter` near the tile bake is
		// a different code path), so any per-patch overhead figure would rest on an assumed
		// value. Counts, areas and faces are all measured directly.
		if (TEXTURE_DIAG_ENABLED()) {
			// Bucket by the LONGER side: a 6x400 strip is a fragmentation artifact, not a
			// large patch, and bucketing on area alone would file it with the healthy ones.
			const int kEdges[] = { 8, 16, 32, 64, 128, 256, 512, INT_MAX };
			const size_t nB = sizeof(kEdges) / sizeof(kEdges[0]);
			std::vector<uint64_t> bCount(nB, 0), bArea(nB, 0), bFaces(nB, 0);
			uint64_t totArea = 0, totFaces = 0;
			int minSide = INT_MAX, maxSide = 0;
			for (size_t i = 0; i < (size_t)texturePatches.GetSize(); ++i) {
				const auto& p = texturePatches[(uint32_t)i];
				const int w = std::max(0, p.rect.width), h = std::max(0, p.rect.height);
				const int side = std::max(w, h);
				const uint64_t a = (uint64_t)w * (uint64_t)h;
				const uint64_t nf = (uint64_t)p.faces.size();
				size_t b = 0;
				while (b + 1 < nB && side >= kEdges[b]) ++b;
				++bCount[b]; bArea[b] += a; bFaces[b] += nf;
				totArea += a; totFaces += nf;
				if (side < minSide) minSide = side;
				if (side > maxSide) maxSide = side;
			}
			TEXTURE_DIAG("[ATLAS-PATCHES] %u patches | %llu faces | %.1f Mpx allocated |"
				" side min=%d max=%d -- bucket columns are: patches, %% of allocated area,"
				" %% of faces, texels/face",
				texturePatches.GetSize(), (unsigned long long)totFaces,
				(double)totArea * 1e-6, minSide == INT_MAX ? 0 : minSide, maxSide);
			const char* kNames[] = { "<8", "8-15", "16-31", "32-63", "64-127",
				"128-255", "256-511", ">=512" };
			for (size_t b = 0; b < nB; ++b) {
				if (!bCount[b])
					continue;
				TEXTURE_DIAG("[ATLAS-PATCHES]   side %-8s %8llu patches | area %5.1f%% |"
					" faces %5.1f%% | %7.1f texels/face",
					kNames[b], (unsigned long long)bCount[b],
					totArea  ? 100.0 * (double)bArea[b]  / (double)totArea  : 0.0,
					totFaces ? 100.0 * (double)bFaces[b] / (double)totFaces : 0.0,
					bFaces[b] ? (double)bArea[b] / (double)bFaces[b] : 0.0);
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

// Superseded by the coverage gutter, which runs AFTER the data-color bake and
// rebuilds the whole gutter from the FINAL per-face colours. Running this pass here
// as well is pure duplicated work: it can only pad what is already colEmpty, and
// everything it writes is recomputed later anyway. See TEXTURE_ATLAS_COVERAGE_GUTTER
// for why the ordering matters (the feather writes after this point, so a gutter
// grown here is stale by the time it is sampled).
#if TEXTURE_OUTWARD_DILATE_PX > 0 && !TEXTURE_ATLAS_COVERAGE_GUTTER
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

				// ------------------------------------------------------------
				// INTERIOR FILL vs BOUNDARY SHEET.
				//
				// Unobserved regions come in two topologically distinct kinds, and they
				// want opposite treatment:
				//
				//   ENCLOSED  -- ringed entirely by observed surface (textureless roofs,
				//                water inside the survey, small tears). Data-colouring
				//                these is the whole point of the feature: it closes a hole
				//                with locally-correct colour. KEEP.
				//
				//   BOUNDARY  -- touches the mesh border and extends outward. This is the
				//                Poisson skirt over water at the survey edge: geometry in
				//                the WRONG PLACE, which is exactly why no camera validates
				//                it (the depth test fails in every view) and why it ends up
				//                here rather than in a patch. Data-colouring it dresses a
				//                fabricated sheet up as finished surface. DELETE -- except
				//                for a deliberate margin, because a little synthetic edging
				//                reads better than a hard cut at the data limit.
				//
				// Density-based trimming cannot make this distinction: the sheet sits ON
				// real (spurious) points, so SurfaceTrimmer sees ordinary density and the
				// distance cull measures ordinary proximity -- both verified on this data.
				// Camera visibility is the only signal that separates them, and it only
				// exists here.
				std::vector<FIndex>& deleteFaces = unobservedToDelete;
#if TEXTURE_DELETE_BOUNDARY_UNOBSERVED
				if (!noViewFaces.empty() && faceFaces.size() == faces.size()) {
					const FIndex nF = (FIndex)faces.size();
					const auto centroid = [&](FIndex f) {
						const Face& fc = faces[f];
						return (vertices[fc[0]] + vertices[fc[1]] + vertices[fc[2]]) / 3.f;
					};
					// Explicit 3D length: norm() is only exercised on 2D TexCoord in this
					// translation unit, so do not rely on a 3D overload being visible.
					const auto len3 = [](const Point3f& d) {
						return std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
					};
					// METRIC geodesic distance from the observed surface, not a hop count.
					//
					// Hop count makes a JAGGED band: its iso-contours follow triangle
					// adjacency, and triangle size varies across the mesh, so the kept band
					// comes out wide where triangles are large, narrow where they are small,
					// and its edge zigzags along the tessellation. Distance in world units
					// gives a band of uniform physical width whose contour does not care how
					// the surface happens to be triangulated.
					constexpr float kUnreached = FLT_MAX;
					std::vector<float> dist(nF, kUnreached);
					// Median edge length, so the margin can be expressed as a multiple of the
					// mesh's own resolution and behave the same on any scene scale.
					float medianEdge = 0.f;
					{
						std::vector<float> el;
						el.reserve(std::min<size_t>(faces.size(), 20000));
						const size_t step = std::max<size_t>(1, faces.size() / 20000);
						for (size_t f = 0; f < faces.size(); f += step) {
							const Face& fc = faces[(FIndex)f];
							el.push_back(len3(vertices[fc[1]] - vertices[fc[0]]));
						}
						if (!el.empty()) {
							std::nth_element(el.begin(), el.begin() + el.size()/2, el.end());
							medianEdge = el[el.size()/2];
						}
					}
					const float marginDist = (medianEdge > 0.f)
						? (float)TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES * medianEdge : 0.f;
					// Shared by the distance relaxation and the border flood below.
					std::vector<FIndex> cur, next;
					{
						// Label-correcting relaxation rather than a priority queue: <queue>
						// is not included here, and the band is only a few median edges wide
						// so this converges in a handful of rounds. Bounded by marginDist, so
						// the work scales with the band, not with the whole no-view set.
						for (const FIndex f : noViewFaces) {
							const Mesh::FaceFaces& ff = faceFaces[f];
							for (int e = 0; e < 3; ++e)
								if (ff[e] != NO_ID && covered[ff[e]]) { dist[f] = 0.f; cur.push_back(f); break; }
						}
						for (int round = 0; round < 64 && !cur.empty(); ++round) {
							next.clear();
							for (const FIndex f : cur) {
								const float df = dist[f];
								if (df > marginDist)
									continue;
								const Point3f cf = centroid(f);
								const Mesh::FaceFaces& ff = faceFaces[f];
								for (int e = 0; e < 3; ++e) {
									const FIndex g = ff[e];
									if (g == NO_ID || covered[g])
										continue;
									const float nd = df + len3(centroid(g) - cf);
									if (nd < dist[g]) { dist[g] = nd; next.push_back(g); }
								}
							}
							cur.swap(next);
						}
					}
					// A component counts as BOUNDARY-touching only if it reaches the OUTER
					// silhouette -- not merely any border edge.
					//
					// REGRESSION THIS FIXES: seeding from every NO_ID neighbour treated the rim
					// of every INTERIOR hole as "the boundary". Tree canopy is full of small
					// pre-existing holes, so unobserved patches beside them were classified as
					// boundary sheet and deleted instead of filled -- punching visible holes
					// through vegetation that previously data-coloured correctly.
					//
					// Border edges form closed loops: one huge loop around the survey outline
					// (plus any large interior void) and thousands of tiny ones around canopy
					// gaps. Union-find the border vertices into loops, measure each loop's
					// bounding box, and seed only from loops that are a meaningful fraction of
					// the whole mesh. Same shape of rule as the hole-closing gate, which also
					// separates "real edge" from "interior hole" by size against the mesh
					// diagonal.
					std::vector<uint8_t> touchesBorder(nF, 0);
					{
						// Plain arrays and an iterative union-find: no AABB3f / unordered_map /
						// std::function / 3D norm(), none of which are known-available here.
						const size_t nV = vertices.size();
						std::vector<uint32_t> uf(nV);
						for (size_t v = 0; v < nV; ++v) uf[v] = (uint32_t)v;
						const auto find = [&uf](uint32_t a) {
							while (uf[a] != a) { uf[a] = uf[uf[a]]; a = uf[a]; }
							return a;
						};
						const auto unite = [&](uint32_t a, uint32_t b) {
							a = find(a); b = find(b); if (a != b) uf[b] = a;
						};
						for (FIndex f = 0; f < nF; ++f) {
							const Mesh::FaceFaces& ff = faceFaces[f];
							const Face& fc = faces[f];
							for (int e = 0; e < 3; ++e)
								if (ff[e] == NO_ID)
									unite(fc[e], fc[(e + 1) % 3]);
						}
						// Per-loop bbox, indexed by union-find root.
						std::vector<float> bxLo(nV, FLT_MAX), byLo(nV, FLT_MAX), bzLo(nV, FLT_MAX);
						std::vector<float> bxHi(nV, -FLT_MAX), byHi(nV, -FLT_MAX), bzHi(nV, -FLT_MAX);
						std::vector<uint8_t> isLoopRoot(nV, 0);
						const auto addToLoop = [&](uint32_t root, const Point3f& p) {
							isLoopRoot[root] = 1;
							if (p.x < bxLo[root]) bxLo[root] = p.x;
							if (p.y < byLo[root]) byLo[root] = p.y;
							if (p.z < bzLo[root]) bzLo[root] = p.z;
							if (p.x > bxHi[root]) bxHi[root] = p.x;
							if (p.y > byHi[root]) byHi[root] = p.y;
							if (p.z > bzHi[root]) bzHi[root] = p.z;
						};
						for (FIndex f = 0; f < nF; ++f) {
							const Mesh::FaceFaces& ff = faceFaces[f];
							const Face& fc = faces[f];
							for (int e = 0; e < 3; ++e) {
								if (ff[e] != NO_ID)
									continue;
								const uint32_t r = find(fc[e]);
								addToLoop(r, vertices[fc[e]]);
								addToLoop(r, vertices[fc[(e + 1) % 3]]);
							}
						}
						const auto loopDiag = [&](uint32_t root) {
							if (!isLoopRoot[root]) return 0.f;
							return len3(Point3f(bxHi[root]-bxLo[root],
							                    byHi[root]-byLo[root],
							                    bzHi[root]-bzLo[root]));
						};
						float mnx = FLT_MAX, mny = FLT_MAX, mnz = FLT_MAX;
						float mxx = -FLT_MAX, mxy = -FLT_MAX, mxz = -FLT_MAX;
						for (size_t v = 0; v < nV; ++v) {
							const Point3f& p = vertices[(Mesh::VIndex)v];
							if (p.x < mnx) mnx = p.x;  if (p.x > mxx) mxx = p.x;
							if (p.y < mny) mny = p.y;  if (p.y > mxy) mxy = p.y;
							if (p.z < mnz) mnz = p.z;  if (p.z > mxz) mxz = p.z;
						}
						const float meshDiag = len3(Point3f(mxx-mnx, mxy-mny, mxz-mnz));
						const float minLoopDiag =
							(float)TEXTURE_OUTER_LOOP_MIN_DIAG_FRAC * meshDiag;
						// Loop census: reported only, the cut below tests each loop directly.
						size_t nLoops = 0, nOuterLoops = 0;
						if (TEXTURE_DIAG_ENABLED()) {
							for (size_t v = 0; v < nV; ++v) {
								if (!isLoopRoot[v]) continue;
								++nLoops;
								if (loopDiag((uint32_t)v) >= minLoopDiag) ++nOuterLoops;
							}
						}
						cur.clear();
						for (const FIndex f : noViewFaces) {
							const Mesh::FaceFaces& ff = faceFaces[f];
							const Face& fc = faces[f];
							for (int e = 0; e < 3; ++e) {
								if (ff[e] != NO_ID)
									continue;
								if (loopDiag(find(fc[e])) >= minLoopDiag) {
									touchesBorder[f] = 1; cur.push_back(f); break;
								}
							}
						}
						TEXTURE_DIAG("[TEX-SHEET] border loops: %zu total, %zu qualify as outer"
							" (bbox diag >= %.3g = %.3g of mesh diag %.4g)",
							nLoops, nOuterLoops, minLoopDiag,
							(double)TEXTURE_OUTER_LOOP_MIN_DIAG_FRAC, meshDiag);
					}
					while (!cur.empty()) {
						next.clear();
						for (const FIndex f : cur) {
							const Mesh::FaceFaces& ff = faceFaces[f];
							for (int e = 0; e < 3; ++e) {
								const FIndex g = ff[e];
								if (g != NO_ID && !covered[g] && !touchesBorder[g]) {
									touchesBorder[g] = 1; next.push_back(g);
								}
							}
						}
						cur.swap(next);
					}
					// CONNECTED COMPONENTS of the unobserved set. The keep rule below judges an
					// ENCLOSED region by TOPOLOGY alone, which cannot tell a lake from a tongue; its
					// SIZE can. See TEXTURE_FILL_MIN_COMPONENT_FACES. Same flood/label/report idiom as
					// the orphan filter above, so the distribution can be read before choosing a value.
					std::vector<int32_t> fComp(nF, -1);
					std::vector<uint32_t> compSizeNV;
					{
						std::vector<uint8_t> isNV(nF, 0);
						for (const FIndex f : noViewFaces) isNV[f] = 1;
						std::vector<FIndex> stk;
						for (const FIndex f0 : noViewFaces) {
							if (fComp[f0] != -1) continue;
							const int32_t c = (int32_t)compSizeNV.size();
							compSizeNV.push_back(0);
							fComp[f0] = c; stk.push_back(f0);
							while (!stk.empty()) {
								const FIndex f = stk.back(); stk.pop_back();
								++compSizeNV[c];
								const Mesh::FaceFaces& ffc = faceFaces[f];
								for (int e = 0; e < 3; ++e) {
									const FIndex g = ffc[e];
									if (g != NO_ID && isNV[g] && fComp[g] == -1) { fComp[g] = c; stk.push_back(g); }
								}
							}
						}
						// Distribution report only -- the keep rule below reads compSizeNV directly.
						if (TEXTURE_DIAG_ENABLED()) {
							std::vector<uint32_t> srt(compSizeNV);
							std::sort(srt.begin(), srt.end(), [](uint32_t a, uint32_t b) { return a > b; });
							std::string szStr;
							for (size_t i = 0; i < srt.size() && i < 12; ++i) { szStr += std::to_string(srt[i]); szStr += ' '; }
							TEXTURE_DIAG("[TEX-SHEET] unobserved components: %zu total, sizes (top of %zu): %s"
								"| ENCLOSED kept only at >= %d faces -- read this distribution and pick a"
								" value in a real GAP between the big regions (lakes, textureless roofs) and"
								" the speck/tongue tier",
								compSizeNV.size(), srt.size(), szStr.c_str(),
								(int)TEXTURE_FILL_MIN_COMPONENT_FACES);
						}
					}
					// Per-face keep/drop, before smoothing.
					std::vector<uint8_t> keepF(nF, 0);
					size_t nHiddenDropped = 0, nSmallDropped = 0;
#if TEXTURE_DELETE_HIDDEN_UNOBSERVED
					// See TEXTURE_DELETE_HIDDEN_UNOBSERVED. Reuses the everFrontWound record built
					// in ListCameraFaces; empty means it did not run, so fall through to the
					// previous keep-everything-enclosed behaviour rather than deleting blindly.
					const uint8_t* const pFWkeep =
						(everFrontWound.size() == faces.size()) ? everFrontWound.data() : nullptr;
					const uint8_t* const pEPkeep =
						(everProjected.size() == faces.size()) ? everProjected.data() : nullptr;
#endif
					for (const FIndex f : noViewFaces) {
#if TEXTURE_DELETE_HIDDEN_UNOBSERVED
						// HIDDEN GEOMETRY, not textureless surface. A face that never projected
						// FRONT-WOUND in any view is part of the envelope's inward-facing side --
						// invisible in every render, and on a near-closed Poisson mesh that is the
						// UNDERSIDE. Dressing it with synthesized colour is what makes it show up
						// as spikes hanging off the silhouette. Genuine textureless surface (a flat
						// roof, water inside the survey) faces the cameras and IS front-wound, so
						// it still reaches the fill below.
						//
						// everProjected is REQUIRED here. Without it the test also catches faces
						// outside every frustum, which were never tested at all -- and deleting
						// those punched scattered holes through real terrain (63,662 of 71,223
						// dropped, i.e. back-winding 42,935 PLUS off-frustum 20,725). "Never
						// tested" must fail safe to KEEP.
						if (pFWkeep && pEPkeep && pEPkeep[f] && !pFWkeep[f]) { ++nHiddenDropped; continue; }
#endif
						// Enclosed -> always keep (interior fill). Boundary component -> keep
						// the metric margin nearest real surface. dist stays kUnreached for a
						// component that never reaches an observed face at all: a detached
						// island with no real texture to blend from, always dropped.
						const bool bEnclosed = !touchesBorder[f];
#if TEXTURE_FILL_MIN_COMPONENT_FACES > 0
						// A SMALL enclosed region is not a lake or a textureless roof -- it is a speck or
						// a tongue of synthetic surface, which is what reads as clutter at the edge. Chop
						// it rather than dress it. Large enclosed regions (the water) are untouched.
						if (bEnclosed && fComp[f] >= 0 &&
							compSizeNV[(size_t)fComp[f]] < (uint32_t)TEXTURE_FILL_MIN_COMPONENT_FACES) {
							++nSmallDropped;
							continue;
						}
#endif
						if (bEnclosed || (dist[f] != kUnreached && dist[f] <= marginDist))
							keepF[f] = 1;
					}
					TEXTURE_DIAG("[TEX-SHEET] hidden-geometry filter: %zu of %zu unobserved faces never"
						" projected front-wound in any view -> dropped instead of filled",
						nHiddenDropped, noViewFaces.size());
					TEXTURE_DIAG("[TEX-SHEET] small-component filter: %zu unobserved faces dropped as"
						" enclosed components under %d faces", nSmallDropped,
						(int)TEXTURE_FILL_MIN_COMPONENT_FACES);

					// BOUNDARY REGULARIZATION -- majority filter over face adjacency.
					//
					// Even with a metric band the cut is a threshold on a field sampled per
					// triangle, so it still leaves single-face spikes sticking out and
					// single-face notches bitten in: the "jaggedy" edge. Nothing downstream
					// fixes it -- Mesh::Clean's alpha-tighten and Taubin boundary smoothing
					// run back in ReconstructMesh, long before this cut exists.
					//
					// A face whose neighbours mostly disagree with it is exactly such a spike
					// or notch, so flip it. Two or three passes remove them without moving the
					// band as a whole (a face in the interior of either region has all three
					// neighbours agreeing and never flips). Applied ONLY to unobserved faces
					// in boundary components -- observed faces and interior fill are never
					// touched.
					// Per-pass flip count, logged below. Without it there is no way to tell a pass
					// count that has CONVERGED (flips falling to ~0, more passes would be free but
					// pointless) from one that is still cutting into the contour every pass -- and
					// that is exactly the question when raising this knob.
					// SEQUENTIAL (Gauss-Seidel) UPDATE, not synchronous (Jacobi). This was a real
					// bug, and the flip counter is what exposed it: with a Jacobi update the pass
					// read the OLD keepF and wrote a separate `upd`, so two adjacent faces could
					// each see the other disagreeing, both flip together, and both flip back on the
					// next pass -- forever. MEASURED before the fix:
					//     flips per pass: 3072 2585 2555 2549 2547 2547 2546 2546
					// i.e. ~2,546 faces oscillating with period 2 and no convergence at any pass
					// count. The contour was never being smoothed, it was thrashing, and the result
					// depended on whether the pass count happened to be even or odd -- which is why
					// both 3 and 8 passes left it choppy.
					//
					// Writing in place makes each face see its neighbours' ALREADY-UPDATED values,
					// which breaks the two-face symmetry that sustains the cycle: every flip then
					// strictly reduces local disagreement, so the filter descends to a fixed point.
					// Order is the fixed noViewFaces order, so the result stays deterministic.
					std::string smoothFlips;
					const bool bFlipTrace = TEXTURE_DIAG_ENABLED(); // the trace, not the filter
					for (int pass = 0; pass < TEXTURE_UNOBSERVED_EDGE_SMOOTH_PASSES; ++pass) {
						size_t nFlips = 0;
						for (const FIndex f : noViewFaces) {
							if (!touchesBorder[f])
								continue;   // interior fill: not ours to reshape
							const Mesh::FaceFaces& ff = faceFaces[f];
							int agree = 0, disagree = 0;
							for (int e = 0; e < 3; ++e) {
								const FIndex g = ff[e];
								if (g == NO_ID)
									continue;
								// An observed neighbour counts as "keep": the band should hug
								// real surface, never pull away from it.
								const uint8_t gk = covered[g] ? 1 : keepF[g];
								if (gk == keepF[f]) ++agree; else ++disagree;
							}
							if (disagree > agree) {
								keepF[f] = keepF[f] ? 0 : 1; // in place: see the note above
								++nFlips;
							}
						}
						if (bFlipTrace) {
							smoothFlips += (pass ? " " : "");
							smoothFlips += std::to_string(nFlips);
						}
						if (nFlips == 0)
							break; // fixed point reached; further passes cannot change anything
					}
					TEXTURE_DIAG("[TEX-SHEET] contour majority filter: %d passes, flips per pass: %s"
						" -- falling to ~0 means the contour has converged and more passes are"
						" pointless; still large on the last pass means it is eating into the"
						" boundary rather than removing spikes",
						(int)TEXTURE_UNOBSERVED_EDGE_SMOOTH_PASSES, smoothFlips.c_str());

					// RIM PEEL on the boundary this cut just created.
					//
					// A face joined to the mesh by a single edge has TWO border edges: a
					// sliver hanging off the rim, which is what the leftover "spikes" along
					// the cut are. Mesh::Clean's alpha-tighten removes exactly these, but it
					// ran in ReconstructMesh -- long before this boundary existed -- so the
					// new rim is raw and nothing downstream will tidy it.
					//
					// Unlike everything above this peels OBSERVED faces too: a two-border-edge
					// triangle is degenerate boundary geometry regardless of whether a camera
					// happened to see it, and leaving textured spikes behind was the visible
					// defect. A normal rim face has ONE border edge and is never touched, so
					// this cannot erode the real survey edge -- it only removes what is
					// already dangling.
					size_t nPeeled = 0, nOrphanFaces = 0, nOrphanComps = 0;
					{
						// survives = will still be in the mesh: observed faces, plus the
						// unobserved ones the band decided to keep.
						//
						// This scope is UNCONDITIONAL (the peel loop below is already a no-op
						// when its pass count is 0) because the orphan-component filter that
						// follows needs `alive` regardless of whether the peel is enabled --
						// the margin cut alone severs necks even with the peel off.
						std::vector<uint8_t> alive(nF, 0);
						for (FIndex f = 0; f < nF; ++f)
							alive[f] = (covered[f] || keepF[f]) ? 1 : 0;
						for (int pass = 0; pass < TEXTURE_UNOBSERVED_RIM_PEEL_PASSES; ++pass) {
							std::vector<FIndex> doomed;
							for (FIndex f = 0; f < nF; ++f) {
								if (!alive[f])
									continue;
								const Mesh::FaceFaces& ff = faceFaces[f];
								int border = 0;
								bool anyNew = false;
								for (int e = 0; e < 3; ++e) {
									const FIndex g = ff[e];
									if (g == NO_ID) {
										++border;            // pre-existing hole rim
									} else if (!alive[g]) {
										++border; anyNew = true;  // border WE just created
									}
								}
								// Require at least one NEWLY created border edge.
								//
								// REGRESSION THIS FIXES: without it, a face sitting between two
								// pre-existing holes already has two border edges, so it peels,
								// which exposes more such faces -- six passes of that erodes tree
								// canopy (dense with small gaps) into visible holes. Confining
								// the peel to borders this cut produced keeps it doing its job on
								// the new rim while leaving existing topology untouched.
								if (border >= 2 && anyNew)
									doomed.push_back(f);
							}
							if (doomed.empty())
								break;
							for (const FIndex f : doomed) { alive[f] = 0; ++nPeeled; }
						}

						// ORPHANED-COMPONENT REMOVAL -- the islands these cuts create.
						//
						// Runs LAST, after the margin cut, the majority filter and the peel, so it
						// sees the final surviving surface: any of the three can be the pass that
						// severs the last neck holding a blob on. See
						// TEXTURE_ORPHAN_COMPONENT_PCT_X1000 for the measurement behind it.
						//
						// CAVEAT, inherited from the relative-to-largest rule: a genuinely separate
						// small structure is indistinguishable from debris by size alone and will be
						// dropped with it. That is why the component and face counts are logged
						// rather than silently applied -- a large number means the threshold is
						// wrong for this scene, not that the mesh was dirty.
#if TEXTURE_ORPHAN_COMPONENT_PCT_X1000 > 0
						{
							// Flood the surviving faces over the same adjacency the cuts used.
							// Explicit stack, not recursion: a component here can be the whole
							// mesh (hundreds of thousands of faces deep).
							std::vector<int32_t> comp(nF, -1);
							std::vector<uint32_t> compSize;
							std::vector<FIndex> compStack;

							// How many components did we START with? Same flood, over the
							// ORIGINAL adjacency, ignoring `alive` -- this is the input mesh
							// as ReconstructMesh/RefineMesh handed it over. One component here
							// means every extra component below was manufactured by this
							// stage's cuts, which licenses the largest-only rule.
							size_t nInputComps = 0;
							{
								std::vector<int32_t> icomp(nF, -1);
								for (FIndex s = 0; s < nF; ++s) {
									if (icomp[s] != -1)
										continue;
									const int32_t c = (int32_t)nInputComps++;
									icomp[s] = c;
									compStack.push_back(s);
									while (!compStack.empty()) {
										const FIndex f = compStack.back();
										compStack.pop_back();
										const Mesh::FaceFaces& ff = faceFaces[f];
										for (int e = 0; e < 3; ++e) {
											const FIndex g = ff[e];
											if (g != NO_ID && icomp[g] == -1) {
												icomp[g] = c;
												compStack.push_back(g);
											}
										}
									}
								}
							}

							for (FIndex s = 0; s < nF; ++s) {
								if (!alive[s] || comp[s] != -1)
									continue;
								const int32_t c = (int32_t)compSize.size();
								compSize.push_back(0);
								comp[s] = c;
								compStack.push_back(s);
								while (!compStack.empty()) {
									const FIndex f = compStack.back();
									compStack.pop_back();
									++compSize[c];
									const Mesh::FaceFaces& ff = faceFaces[f];
									for (int e = 0; e < 3; ++e) {
										const FIndex g = ff[e];
										if (g != NO_ID && alive[g] && comp[g] == -1) {
											comp[g] = c;
											compStack.push_back(g);
										}
									}
								}
							}
							uint32_t largest = 0;
							for (const uint32_t sz : compSize)
								if (sz > largest)
									largest = sz;
							// Size distribution, descending. This is the line that says whether the
							// threshold is even the right tool: if the runners-up after the body are
							// LARGE, the survivors are real slabs of surface and no size floor
							// removes them safely; if they are all tiny, the cut is only shedding
							// specks and the floor is doing its job. Same idiom as
							// "DIAG component sizes" in Mesh.cpp.
							if (TEXTURE_DIAG_ENABLED()) {
								std::vector<uint32_t> sorted(compSize);
								std::sort(sorted.begin(), sorted.end(),
									[](uint32_t a, uint32_t b) { return a > b; });
								std::string dist;
								for (size_t i = 0; i < sorted.size() && i < 12; ++i) {
									dist += std::to_string(sorted[i]);
									dist += ' ';
								}
								TEXTURE_DIAG("[TEX-SHEET] component sizes (top of %zu): %s",
									compSize.size(), dist.c_str());
							}
							if (compSize.size() > 1 && largest > 0) {
								// Input was one surface -> every other component is our own
								// doing; keep only the largest. minKeep == largest drops
								// everything strictly smaller (a size tie survives, which is
								// the safe direction).
								const bool bKeepLargestOnly =
									TEXTURE_ORPHAN_KEEP_LARGEST_IF_INPUT_SINGLE && nInputComps == 1;
								const uint32_t minKeep = bKeepLargestOnly ? largest
									: std::max<uint32_t>(1, (uint32_t)(
										(double)largest
										* (double)TEXTURE_ORPHAN_COMPONENT_PCT_X1000 / 100000.0));
								for (const uint32_t sz : compSize)
									if (sz < minKeep)
										++nOrphanComps;
								for (FIndex f = 0; f < nF; ++f) {
									if (alive[f] && compSize[comp[f]] < minKeep) {
										alive[f] = 0;
										++nOrphanFaces;
									}
								}
								if (bKeepLargestOnly) {
									TEXTURE_DIAG("[TEX-SHEET] orphan filter: input was 1 component,"
										" this stage's cuts made %zu -> keeping only the largest"
										" (%u faces), dropped %zu components (%zu faces)",
										compSize.size(), largest, nOrphanComps, nOrphanFaces);
								} 
								else {
									TEXTURE_DIAG("[TEX-SHEET] orphan filter: input had %zu components,"
										" %zu survive the cut, largest %u faces -> dropped %zu"
										" components (%zu faces) under %u faces (%.3g%% of largest)",
										nInputComps, compSize.size(), largest, nOrphanComps,
										nOrphanFaces, minKeep,
										(double)TEXTURE_ORPHAN_COMPONENT_PCT_X1000 / 1000.0);
								}
							} else {
								TEXTURE_DIAG("[TEX-SHEET] orphan filter: surface is a single component"
									" (%u faces) -- nothing to drop", largest);
							}
						}
#endif

						// Fold the peel and the orphan filter back in. Only OBSERVED faces are
						// queued here: a dropped UNOBSERVED face just has its keepF cleared and
						// the loop below picks it up, so queueing it here too would double-count
						// it.
						for (FIndex f = 0; f < nF; ++f) {
							if (alive[f])
								continue;
							if (covered[f]) {
								deleteFaces.push_back(f);
#if TEXTURE_FEATHER_CUT_SILHOUETTE
								// Stop counting it as real texture: it is leaving the mesh.
								// The seam feather seeds its ring 0 on `covered` (a face is a
								// boundary face when a neighbour is NO_ID or not covered), so
								// leaving the flag set meant the silhouette THESE cuts create
								// was never seen as a boundary and never feathered -- a hard
								// crisp edge along the whole new rim. The fill tiles' detail
								// and mirror donor search reads `covered` too, and would
								// otherwise transplant texture from a face about to vanish.
								covered[f] = 0;
#endif
								// Left SET by default: every surviving neighbour of this face
								// would otherwise become a ring-0 feather boundary at alpha 1,
								// flattening real rim texture (eaves, roof edges) to the fill
								// colour field. See TEXTURE_FEATHER_CUT_SILHOUETTE for why the
								// two donor-search reasons above are inert in this build.
								// NOTE the face is still queued for deletion either way -- this
								// controls only whether the cut FEATHERS, never what is cut.
							}
							keepF[f] = 0;
						}
					}

					std::vector<FIndex> keep;
					keep.reserve(noViewFaces.size());
					size_t nEnclosed = 0, nMargin = 0;
					for (const FIndex f : noViewFaces) {
						if (!keepF[f]) { deleteFaces.push_back(f); continue; }
						keep.push_back(f);
						if (touchesBorder[f]) ++nMargin; else ++nEnclosed;
					}
					TEXTURE_DIAG("[TEX-SHEET] unobserved %zu faces -> keep %zu enclosed (interior fill)"
						" + %zu within %.3g-unit edge margin (%.1f x median edge %.3g,"
						" %d smoothing passes) | rim peel removed %zu 2-border-edge slivers"
						" | orphan filter removed %zu faces in %zu islands"
						" | DELETE %zu faces total",
						noViewFaces.size(), nEnclosed, nMargin, marginDist,
						(double)TEXTURE_UNOBSERVED_EDGE_MARGIN_EDGES, medianEdge,
						(int)TEXTURE_UNOBSERVED_EDGE_SMOOTH_PASSES, nPeeled,
						nOrphanFaces, nOrphanComps, deleteFaces.size());
					noViewFaces.swap(keep);
				}
#endif
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
						const float seedMinCos = (float)TEXTURE_DATACOLOR_SEED_MIN_COS;
						for (const TexturePatch& tp : texturePatches) {
							for (const FIndex fc : tp.faces) {
								const Face& face = faces[fc];
								if (!(vUsed[face[0]] || vUsed[face[1]] || vUsed[face[2]]))
									continue; // face doesn't touch the fill region
								if (seedMinCos > 0.f) {
									// reject grazing seed donors: their noisy/shadowed texture fragments the fill
									const Vertex& sA = vertices[face[0]]; const Vertex& sB = vertices[face[1]]; const Vertex& sC = vertices[face[2]];
									const Point3f sCen((sA.x + sB.x + sC.x) * (1.f / 3.f), (sA.y + sB.y + sC.y) * (1.f / 3.f), (sA.z + sB.z + sC.z) * (1.f / 3.f));
									const Point3f sCamDir(Cast<Mesh::Type>(images[tp.label].camera.C) - sCen);
									if (ComputeAngle2(sCamDir.ptr(), scene.mesh.faceNormals[fc].ptr()) < seedMinCos)
										continue;
								}
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
					// HIGHLIGHT CLAMP (PER-COMPONENT): cap each seed's luma to a robust ceiling
					// (median x factor) computed over its OWN connected fill region, so a specular
					// highlight caught on that region's boundary can't over-brighten and spread a
					// too-bright colour through it, while a genuinely-bright region elsewhere is not
					// dragged down by a dark region's median (RGB scaled -> hue preserved).
					{
						const float seedHiClamp = (float)TEXTURE_DATACOLOR_SEED_HIGHLIGHT_CLAMP;
						if (seedHiClamp > 0.f) {
							// label connected components of the no-view faces (edge adjacency) and
							// tag each used (boundary/interior) vertex with its component id
							std::vector<uint8_t> isNoViewHi(faces.size(), 0);
							for (int i = 0; i < N; ++i) isNoViewHi[noViewFaces[i]] = 1;
							std::vector<int> fCompHi(faces.size(), -1);
							std::vector<int> vCompHi(NV, -1);
							int nCompHi = 0;
							{
								std::vector<FIndex> bfs;
								for (int i = 0; i < N; ++i) {
									const FIndex f0 = noViewFaces[i];
									if (fCompHi[f0] != -1) continue;
									const int cid = nCompHi++;
									fCompHi[f0] = cid; bfs.clear(); bfs.push_back(f0);
									while (!bfs.empty()) {
										const FIndex f = bfs.back(); bfs.pop_back();
										const Face& fa = faces[f];
										for (int k = 0; k < 3; ++k) { const uint32_t v = fa[k]; if (vCompHi[v] < 0) vCompHi[v] = cid; }
										const Mesh::FaceFaces& adj = faceFaces[f];
										for (int k = 0; k < 3; ++k) { const FIndex fn = adj[k]; if (fn == NO_ID || !isNoViewHi[fn] || fCompHi[fn] != -1) continue; fCompHi[fn] = cid; bfs.push_back(fn); }
									}
								}
							}
							// per-component median luma -> ceiling
							std::vector<std::vector<float>> lumByComp((size_t)nCompHi);
							for (int u = 0; u < NU; ++u) { const uint32_t v = pUsed[u]; if (vValid[v] && vCompHi[v] >= 0) lumByComp[(size_t)vCompHi[v]].push_back(0.299f * vr[v] + 0.587f * vg[v] + 0.114f * vb[v]); }
							std::vector<float> ceilByComp((size_t)nCompHi, -1.f);
							for (int c = 0; c < nCompHi; ++c) { auto& L = lumByComp[(size_t)c]; if (L.empty()) continue; std::nth_element(L.begin(), L.begin() + L.size() / 2, L.end()); ceilByComp[(size_t)c] = L[L.size() / 2] * seedHiClamp; }
							// clamp each seed to its component's ceiling
							for (int u = 0; u < NU; ++u) {
								const uint32_t v = pUsed[u];
								if (!vValid[v] || vCompHi[v] < 0) continue;
								const float cl = ceilByComp[(size_t)vCompHi[v]];
								if (cl < 0.f) continue;
								const float L = 0.299f * vr[v] + 0.587f * vg[v] + 0.114f * vb[v];
								if (L > cl && L > 1e-3f) { const float s = cl / L; vr[v] *= s; vg[v] *= s; vb[v] *= s; }
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

					// TEXEL DENSITY OF THE REAL TEXTURE, in px per world unit.
					//
					// targetMaxPx alone gives EVERY component a tile of the same pixel size
					// regardless of how much world it covers, so the atlas cost scales with
					// the NUMBER of components rather than their area. MEASURED on a 437-view
					// scene: 1252 fill components took 2559 atlas rows -- 17.8% of the atlas
					// for 3.3% of the faces, i.e. 613 px per filled face against 95 px per
					// real textured face. Most of that goes to small slivers being handed a
					// full-size tile.
					//
					// These fills are water that does not reconstruct, so they carry no detail
					// at any resolution; baking them denser than the texture they sit next to
					// buys nothing. Matching that density is the correct target -- sharper is
					// waste, blurrier shows a discontinuity at the seam.
					//
					// Used as a CAP only (min with the existing budget), so no tile ever grows
					// and the change cannot introduce a quality regression -- large components
					// still stop at targetMaxPx, small ones stop at real-texture density.
					float texelDensity = FLT_MAX;
					{
						double pxArea = 0.0, worldArea = 0.0;
						for (const TexturePatch& tp : texturePatches) {
							pxArea += (double)tp.rect.width * (double)tp.rect.height;
							for (const FIndex f : tp.faces) {
								const Face& fc = faces[f];
								worldArea += ComputeTriangleArea(
									vertices[fc[0]], vertices[fc[1]], vertices[fc[2]]);
							}
						}
						if (worldArea > 1e-9 && pxArea > 0.0)
							texelDensity = (float)std::sqrt(pxArea / worldArea);
						// These fills are water that does not reconstruct: a smooth field
						// produced by nearest-fill plus seam smoothing, carrying no detail at
						// ANY resolution. Matching real-texture density is still far more than
						// they need, and the atlas cost is not marginal -- MEASURED, 22,027
						// fill tiles took 6,194 rows, 27.9% of the atlas, on a scene whose real
						// texture is already 3.4x under-resolved. Every row spent here is taken
						// from terrain that could have used it.
						texelDensity *= (float)TEXTURE_DATACOLOR_DENSITY_SCALE;
					}
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
					// eu/ev (the component's projected extent) are kept on the tile so its
					// pixel size can always be RE-DERIVED from its scale -- see tileDim.
					struct Tile { Point3f c, t, b; float umin, vmin, scale, eu, ev; int x, y, w, h; };
					std::vector<Tile> tiles(nComp);
					// Tile pixel size for a given scale. Used both when a tile is first sized
					// and when the atlas clamp shrinks it, so the two can never disagree.
					//
					// THE BUG THIS REPLACES: the clamp used to scale the stored dimension
					// directly, `T.w = lround(T.w * shrink)`. But T.w already contained the
					// 2*gutter inset, so that shrank the GUTTER along with the content while
					// the triangles were still placed at +gutter. Required width is
					// (T.w - 2*gutter)*shrink + 2*gutter, strictly MORE than T.w*shrink, so
					// the triangles ran past the tile edge. The rasteriser clamps at
					// tx+tw-1 and never painted those texels, but the texcoords still pointed
					// at them -- i.e. the outer strip of the fill patch sampled the NEXT
					// tile, a different component with a different colour, as a band along
					// the fill boundary. Only fired when [DATACOLOR-CLAMP] was logged.
					const auto tileDim = [&](float ext, float scale) {
						return std::min(std::max((int)std::ceil(ext * scale) + 2 * gutter, minTileDim), maxTilePx);
					};
					for (int cc = 0; cc < nComp; ++cc) {
						const std::vector<FIndex>& cfs = compFaces[cc];
						Point3f nrm(0, 0, 0), cen(0, 0, 0); int vcnt = 0;
						// A SINGLE non-finite face normal must not decide this tile's basis.
						// The component sum is what makes that possible: one NaN normal (a
						// zero-area face put through an unguarded normalize -- see
						// SafeNormalizeFaceNormal) poisons nrm for the WHOLE component, and
						// because every `x < eps` test below is FALSE for NaN, none of the
						// degenerate-case fallbacks fire. The basis then goes NaN, so does
						// every (du,dv), and every face in the component is written a NaN
						// texcoord -- MEASURED: 2 components, 26,353 faces, 79,059 vertices,
						// rendering as two canopy-sized BLACK blobs. Skip the bad normals;
						// the rest of the component still gives a good average.
						for (const FIndex f : cfs) {
							const Normal& fnm = scene.mesh.faceNormals[f];
							if (ISFINITE(fnm.x) && ISFINITE(fnm.y) && ISFINITE(fnm.z)) {
								nrm.x += fnm.x; nrm.y += fnm.y; nrm.z += fnm.z;
							}
							const Face& face = faces[f];
							for (int k = 0; k < 3; ++k) { const Vertex& P = vertices[face[k]]; cen.x += P.x; cen.y += P.y; cen.z += P.z; ++vcnt; }
						}
						const float invc = vcnt ? 1.f / (float)vcnt : 1.f;
						cen.x *= invc; cen.y *= invc; cen.z *= invc;
						// Belt and braces: if the centroid itself is non-finite (a NaN vertex
						// in the mesh) there is no usable frame at all, so fall back rather
						// than propagate it into the texcoords.
						if (!(ISFINITE(cen.x) && ISFINITE(cen.y) && ISFINITE(cen.z)))
							cen = Point3f(0, 0, 0);
						float nl = std::sqrt(nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z);
						// `!(nl > eps)`, not `nl < eps`: the negated form is TRUE for NaN, so
						// the fallback fires on the case it exists to catch.
						if (!(nl > 1e-12f)) { nrm = Point3f(0, 0, 1); nl = 1.f; }
						nrm.x /= nl; nrm.y /= nl; nrm.z /= nl;
						const Point3f a = (fabsf(nrm.x) < 0.9f) ? Point3f(1, 0, 0) : Point3f(0, 1, 0);
						Point3f t(a.y * nrm.z - a.z * nrm.y, a.z * nrm.x - a.x * nrm.z, a.x * nrm.y - a.y * nrm.x);
						float tl = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z); if (!(tl > 1e-12f)) { t = Point3f(1, 0, 0); tl = 1.f; }
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
						// px per world unit: the fixed per-component budget, capped at the
						// density of the real texture this fill abuts (see texelDensity).
						float scale = (ext > 1e-9f) ? (targetMaxPx / ext) : 1.f;
						if (scale > texelDensity)
							scale = texelDensity;
						int w = tileDim(eu, scale);
						int h = tileDim(ev, scale);
						if (w > cols) w = cols;
						tiles[cc] = Tile{ cen, t, bb, umin, vmin, scale, eu, ev, 0, 0, w, h };
					}
					// 3) shelf-pack the tiles into appended atlas rows
					int shelfX = 0, shelfY = 0, shelfH = 0;
					for (int cc = 0; cc < nComp; ++cc) {
						Tile& T = tiles[cc];
						if (shelfX + T.w > cols) { shelfY += shelfH; shelfX = 0; shelfH = 0; }
						T.x = shelfX; T.y = oldRows + shelfY;
						shelfX += T.w; shelfH = std::max(shelfH, T.h);
					}
					int extraRows = shelfY + shelfH;

					// HARD CLAMP: the fill must not push the atlas past nMaxTextureSize.
					//
					// Fill rows are APPENDED after the patch pack, and the final resize scales
					// the WHOLE atlas -- real texture included -- if the total exceeds the
					// ceiling. MEASURED: 16,007 packed rows + 6,194 fill rows = 22,201 against
					// a 16,384 cap, i.e. a silent 0.738x linear hit on ALL terrain to make room
					// for water fill. That is strictly the wrong trade: the fill is a smooth
					// field with no detail, terrain is the deliverable.
					//
					// So absorb the overflow in the FILL instead. Shrink the tiles and re-pack
					// until they fit the rows that remain. Worst case the fill degrades to
					// near-flat per-component colour, which for water is what it already looks
					// like -- and the global downscale never fires.
					if (nMaxTextureSize > 0 && oldRows + extraRows > nMaxTextureSize) {
						const int avail = std::max(0, nMaxTextureSize - oldRows);
						const int wanted = extraRows;
						for (int attempt = 0; attempt < 8 && oldRows + extraRows > nMaxTextureSize; ++attempt) {
							if (avail <= 0)
								break;
							const float shrink = std::sqrt(
								(float)avail / (float)std::max(1, extraRows)) * 0.95f;
							for (int cc = 0; cc < nComp; ++cc) {
								Tile& T = tiles[cc];
								T.scale *= shrink;
								// Re-derive from the extent, so the gutter keeps its full
								// 2*gutter px at every shrink step (see tileDim).
								T.w = tileDim(T.eu, T.scale);
								T.h = tileDim(T.ev, T.scale);
								if (T.w > cols) T.w = cols;
							}
							shelfX = 0; shelfY = 0; shelfH = 0;
							for (int cc = 0; cc < nComp; ++cc) {
								Tile& T = tiles[cc];
								if (shelfX + T.w > cols) { shelfY += shelfH; shelfX = 0; shelfH = 0; }
								T.x = shelfX; T.y = oldRows + shelfY;
								shelfX += T.w; shelfH = std::max(shelfH, T.h);
							}
							extraRows = shelfY + shelfH;
						}
						TEXTURE_DIAG("[DATACOLOR-CLAMP] fill wanted %d rows on top of %d packed"
							" (cap %d) -> shrunk to %d rows. Global atlas downscale avoided;"
							" the loss is taken by the flat fill, not by real texture.",
							wanted, oldRows, nMaxTextureSize, extraRows);
					}
					cv::Mat newTex(oldRows + extraRows, cols, CV_8UC3, cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));
					textureDiffuse.copyTo(newTex(cv::Rect(0, 0, cols, oldRows)));
					// Cache newTex base pointer + row stride once (shared read-only across the
					// component-parallel loop; each thread writes a disjoint tile). Replaces the
					// per-texel newTex.at<cv::Vec3b>() in the bake + flood hot loops.
					cv::Vec3b* const nbase = newTex.ptr<cv::Vec3b>(0);
					const size_t nstride = newTex.step.p[0] / sizeof(cv::Vec3b);
					static const float kBayer4c[16] = { 0.f,8.f,2.f,10.f, 12.f,4.f,14.f,6.f, 3.f,11.f,1.f,9.f, 15.f,7.f,13.f,5.f };
					const float detailGain = (float)TEXTURE_DATACOLOR_DETAIL_GAIN;
					// Shared by the fill-tile bake and the seam feather below: both must
					// write past their triangle edges or the straddling texels at the
					// synthetic/observed seam stay stale. See TEXTURE_DATACOLOR_BAKE_OUTSET_PX.
					const float bakeOutset = (float)TEXTURE_DATACOLOR_BAKE_OUTSET_PX;
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
					// chunk 1, not 8: tile cost is O(tile area) and tiles run from minTileDim
					// (6 px) to maxTilePx (384 px), a ~4000x spread, so an 8-wide chunk can hand
					// one thread eight large tiles while others idle. 1300-odd iterations make
					// the finer scheduling free.
#ifdef TEXOPT_USE_OPENMP
					#pragma omp parallel for schedule(dynamic, 1)
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
							// Write OUTSET px past every edge (see TEXTURE_DATACOLOR_BAKE_OUTSET_PX):
							// texels straddling the triangle edge must carry the fill colour too, or
							// the renderer bilinearly mixes a stale texel in at the shared mesh edge.
							// The tile clamp below still confines the write to this tile.
							const float ad = fabsf(denom);
							const float ot0 = (ad > 1e-6f) ? bakeOutset * std::sqrt((ax[1]-ax[2])*(ax[1]-ax[2]) + (ay[1]-ay[2])*(ay[1]-ay[2])) / ad : 0.01f;
							const float ot1 = (ad > 1e-6f) ? bakeOutset * std::sqrt((ax[2]-ax[0])*(ax[2]-ax[0]) + (ay[2]-ay[0])*(ay[2]-ay[0])) / ad : 0.01f;
							const float ot2 = (ad > 1e-6f) ? bakeOutset * std::sqrt((ax[0]-ax[1])*(ax[0]-ax[1]) + (ay[0]-ay[1])*(ay[0]-ay[1])) / ad : 0.01f;
							const float obb = bakeOutset + 1.f;
							int minx = (int)std::floor(std::min(ax[0], std::min(ax[1], ax[2])) - obb);
							int maxx = (int)std::ceil (std::max(ax[0], std::max(ax[1], ax[2])) + obb);
							int miny = (int)std::floor(std::min(ay[0], std::min(ay[1], ay[2])) - obb);
							int maxy = (int)std::ceil (std::max(ay[0], std::max(ay[1], ay[2])) + obb);
							minx = std::max(minx, tx); maxx = std::min(maxx, tx + tw - 1);
							miny = std::max(miny, ty); maxy = std::min(maxy, ty + th - 1);
							for (int py = miny; py <= maxy; ++py)
								for (int px = minx; px <= maxx; ++px) {
									const float fx = px + 0.5f, fy = py + 0.5f;
									float w0 = ((ay[1] - ay[2]) * (fx - ax[2]) + (ax[2] - ax[1]) * (fy - ay[2])) * invDen;
									float w1 = ((ay[2] - ay[0]) * (fx - ax[2]) + (ax[0] - ax[2]) * (fy - ay[2])) * invDen;
									float w2 = 1.f - w0 - w1;
									if (w0 < -ot0 || w1 < -ot1 || w2 < -ot2) continue;
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
						size_t nHiddenLinksSkipped = 0, nSeedRim = 0, nSeedFill = 0;
#if TEXTURE_FEATHER_SKIP_HIDDEN_FILL
						// A no-view neighbour earns a feather only if a camera could actually
						// have seen it head-on. One that never projected front-wound anywhere
						// is an inverted/interior flap behind the surface -- invisible in every
						// render -- so softening the transition into it buys nothing while
						// costing an alpha-1 ring plus K-1 rings of ramp on real texture.
						// See TEXTURE_FEATHER_SKIP_HIDDEN_FILL for the measurements.
						// An empty vector means ListCameraFaces did not fill it in: fall back
						// to the old "every no-view neighbour seeds" behaviour rather than
						// silently feathering nothing.
						const uint8_t* const pFW =
							(everFrontWound.size() == faces.size()) ? everFrontWound.data() : nullptr;
						// same "never tested != inward-facing" distinction as the keep/delete rule:
						// a face outside every frustum is genuine unobserved surface and SHOULD
						// seed a feather band, unlike an inward-facing flap.
						const uint8_t* const pEP =
							(everProjected.size() == faces.size()) ? everProjected.data() : nullptr;
#endif
						for (FIndex f = 0; f < (FIndex)faces.size(); ++f) {
							if (!covered[f]) continue;
							const Mesh::FaceFaces& adj = faceFaces[f];
							bool boundary = false;
							for (int k = 0; k < 3; ++k) {
								const FIndex fn = adj[k];
								// An OPEN mesh edge is the true silhouette, not a fill region,
								// and always seeds -- this switch is only about fill.
								if (fn == NO_ID) { boundary = true; ++nSeedRim; break; }
								if (covered[fn]) continue;
#if TEXTURE_FEATHER_SKIP_HIDDEN_FILL
								if (pFW && pEP && pEP[fn] && !pFW[fn]) { ++nHiddenLinksSkipped; continue; }
#endif
								boundary = true;
								++nSeedFill;
								break;
							}
							if (boundary) { fRing[f] = 0; cur.push_back(f); bandFaces.push_back(f); }
						}
						TEXTURE_DIAG("[SEAM-SKIP] feather ring 0 = %zu faces (%zu seeded by an open mesh"
							" edge, %zu by a visible no-view neighbour); %zu neighbour links"
							" skipped as hidden geometry (never front-wound in any view)",
							cur.size(), nSeedRim, nSeedFill, nHiddenLinksSkipped);
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
							//    DEST = newTex.
							//
							// THREADED BY PATCH. This was the last serial sweep in the bake and the
							// most expensive one -- K rings of observed faces around every fill
							// region, each rasterised over its full atlas footprint.
							//
							// Two band faces can collide on an atlas texel only if they belong to
							// the SAME patch: patches occupy disjoint packed rects, each face's
							// texcoords lie inside its own rect, and the outset (<= 2 px) stays
							// within that rect's own `border` margin -- so a write can never cross
							// into a neighbouring patch. Bucketing the band by patch and staying
							// serial WITHIN a bucket therefore preserves the exact write order that
							// produced the previous output: bit-identical, no lock, no race.
							const float invK = (K > 1) ? 1.f / (float)(K - 1) : 0.f;
							std::vector<uint32_t> facePatch(faces.size(), (uint32_t)NO_ID);
							for (size_t p = 0; p < texturePatches.size(); ++p)
								for (const FIndex fc : texturePatches[(uint32_t)p].faces)
									facePatch[fc] = (uint32_t)p;
							std::vector<std::vector<FIndex>> patchBand(texturePatches.size());
							for (const FIndex f : bandFaces) {
								const uint32_t p = facePatch[f];
								if (p != (uint32_t)NO_ID)
									patchBand[p].push_back(f);
							}
							const int nPB = (int)patchBand.size();
							const int nThreadsF = omp_get_max_threads();
							// How much of the band write is the outset rim -- i.e. how many texels
							// the old centre-inside test was leaving stale. A near-zero count here
							// would mean the seam hairline is NOT the straddling-texel effect and
							// the search moves elsewhere.
							std::vector<long long> tlsBandW((size_t)nThreadsF, 0), tlsOutsetW((size_t)nThreadsF, 0);
#if TEXTURE_DATACOLOR_DIAG
							struct DgAcc { long long written, skipEmpty, skipAlpha, boundN; double sumStep, maxStep, sumResid, maxResid; };
							std::vector<DgAcc> tlsDg((size_t)nThreadsF, DgAcc{ 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0 });
#endif
							// dynamic, 1: bucket sizes span orders of magnitude (one patch may hold
							// the whole band around a large fill region, most hold a handful).
#pragma omp parallel for schedule(dynamic, 1)
							for (int pb = 0; pb < nPB; ++pb) {
							const int tid = omp_get_thread_num();
							long long nBandWritten = 0, nOutsetWritten = 0;
#if TEXTURE_DATACOLOR_DIAG
							long long dgWritten = 0, dgSkipEmpty = 0, dgSkipAlpha = 0, dgBoundN = 0;
							double dgSumStep = 0, dgMaxStep = 0, dgSumResid = 0, dgMaxResid = 0;
#endif
							for (const FIndex f : patchBand[(size_t)pb]) {
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
								// OUTSET (see TEXTURE_DATACOLOR_BAKE_OUTSET_PX). This is the one
								// that matters: a ring-0 band face's outer edge IS the seam, its
								// neighbour across that edge is a fill face in a different part of
								// the atlas, so a texel straddling the edge is written by nobody
								// and keeps raw observed texture. That texel is exactly what
								// bilinear samples when the renderer draws the shared mesh edge.
								// Clamping the barycentrics below extends the edge value AND the
								// edge alpha outward, which at ring 0 is alpha 1 -> the fill's own
								// colour, so the extrapolated rim matches the fill face across the
								// edge instead of stepping away from it.
								const float ad = fabsf(denom);
								const float ot0 = (ad > 1e-6f) ? bakeOutset * FastSqrtS((ax[1]-ax[2])*(ax[1]-ax[2]) + (ay[1]-ay[2])*(ay[1]-ay[2])) / ad : 0.01f;
								const float ot1 = (ad > 1e-6f) ? bakeOutset * FastSqrtS((ax[2]-ax[0])*(ax[2]-ax[0]) + (ay[2]-ay[0])*(ay[2]-ay[0])) / ad : 0.01f;
								const float ot2 = (ad > 1e-6f) ? bakeOutset * FastSqrtS((ax[0]-ax[1])*(ax[0]-ax[1]) + (ay[0]-ay[1])*(ay[0]-ay[1])) / ad : 0.01f;
								const float obb = bakeOutset + 1.f;
								int minx = (int)std::floor(std::min(ax[0], std::min(ax[1], ax[2])) - obb);
								int maxx = (int)std::ceil (std::max(ax[0], std::max(ax[1], ax[2])) + obb);
								int miny = (int)std::floor(std::min(ay[0], std::min(ay[1], ay[2])) - obb);
								int maxy = (int)std::ceil (std::max(ay[0], std::max(ay[1], ay[2])) + obb);
								minx = std::max(minx, 0); maxx = std::min(maxx, tdW - 1);
								miny = std::max(miny, 0); maxy = std::min(maxy, tdH - 1);
								for (int py = miny; py <= maxy; ++py)
									for (int px = minx; px <= maxx; ++px) {
										const float fx = px + 0.5f, fy = py + 0.5f;
										float w0 = ((ay[1] - ay[2]) * (fx - ax[2]) + (ax[2] - ax[1]) * (fy - ay[2])) * invDen;
										float w1 = ((ay[2] - ay[0]) * (fx - ax[2]) + (ax[0] - ax[2]) * (fy - ay[2])) * invDen;
										float w2 = 1.f - w0 - w1;
										if (w0 < -ot0 || w1 < -ot1 || w2 < -ot2) continue;
										const bool bOutside = (w0 < 0.f || w1 < 0.f || w2 < 0.f);
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
										++nBandWritten;
										if (bOutside) ++nOutsetWritten;
										const cv::Vec3b& obs = tdBase[(size_t)py * tdStride + px];
										const float tR = cr[0] * w0 + cr[1] * w1 + cr[2] * w2;
										const float tG = cg[0] * w0 + cg[1] * w1 + cg[2] * w2;
										const float tB = cb[0] * w0 + cb[1] * w1 + cb[2] * w2;
										int ib = (int)((float)obs[0] + a * (tB - (float)obs[0]) + 0.5f); ib = ib < 0 ? 0 : (ib > 255 ? 255 : ib);
										int ig = (int)((float)obs[1] + a * (tG - (float)obs[1]) + 0.5f); ig = ig < 0 ? 0 : (ig > 255 ? 255 : ig);
										int ir = (int)((float)obs[2] + a * (tR - (float)obs[2]) + 0.5f); ir = ir < 0 ? 0 : (ir > 255 ? 255 : ir);
#if TEXTURE_DATACOLOR_DIAG
										{
											constexpr double oneThird = 1.0 / 3.0;
											const double tl = ((double)tR + tG + tB) * oneThird;
											const double ol = ((double)obs[0] + obs[1] + obs[2]) * oneThird;
											const double rl = ((double)ib + ig + ir) * oneThird;
											++dgWritten;
											if (a > 0.9f) { // near the ring-0 boundary
												++dgBoundN;
												const double ds = FastAbsD(ol - tl); dgSumStep += ds; if (ds > dgMaxStep) dgMaxStep = ds;
												const double rs = FastAbsD(rl - tl); dgSumResid += rs; if (rs > dgMaxResid) dgMaxResid = rs;
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
							tlsBandW[(size_t)tid] += nBandWritten;
							tlsOutsetW[(size_t)tid] += nOutsetWritten;
#if TEXTURE_DATACOLOR_DIAG
							{
								DgAcc& dg = tlsDg[(size_t)tid];
								dg.written += dgWritten; dg.skipEmpty += dgSkipEmpty;
								dg.skipAlpha += dgSkipAlpha; dg.boundN += dgBoundN;
								dg.sumStep += dgSumStep; if (dgMaxStep > dg.maxStep) dg.maxStep = dgMaxStep;
								dg.sumResid += dgSumResid; if (dgMaxResid > dg.maxResid) dg.maxResid = dgMaxResid;
							}
#endif
							} // patch bucket
							long long nBandWritten = 0, nOutsetWritten = 0;
							for (int t = 0; t < nThreadsF; ++t) {
								nBandWritten += tlsBandW[(size_t)t];
								nOutsetWritten += tlsOutsetW[(size_t)t];
							}
#if TEXTURE_DATACOLOR_DIAG
							long long dgWritten = 0, dgSkipEmpty = 0, dgSkipAlpha = 0, dgBoundN = 0;
							double dgSumStep = 0, dgMaxStep = 0, dgSumResid = 0, dgMaxResid = 0;
							for (const DgAcc& d : tlsDg) {
								dgWritten += d.written; dgSkipEmpty += d.skipEmpty;
								dgSkipAlpha += d.skipAlpha; dgBoundN += d.boundN;
								dgSumStep += d.sumStep; if (d.maxStep > dgMaxStep) dgMaxStep = d.maxStep;
								dgSumResid += d.sumResid; if (d.maxResid > dgMaxResid) dgMaxResid = d.maxResid;
							}
#endif
							TEXTURE_DIAG("[SEAM-OUTSET] feather band %zu faces -> %lld texels written,"
								" %lld of them (%.1f%%) in the %.2f px outset rim that the old"
								" centre-inside test left stale",
								bandFaces.size(), nBandWritten, nOutsetWritten,
								nBandWritten ? 100.0 * (double)nOutsetWritten / (double)nBandWritten : 0.0,
								(double)bakeOutset);
#if TEXTURE_DATACOLOR_DIAG
							TEXTURE_DIAG("[DATACOLOR-DIAG] noView=%d nComp=%d band=%d | written=%lld skipEmpty=%lld skipAlpha=%lld | boundary=%lld meanStep=%.2f maxStep=%.2f meanResid=%.2f maxResid=%.2f",
								N, nComp, (int)bandFaces.size(),
								dgWritten, dgSkipEmpty, dgSkipAlpha, dgBoundN,
								dgBoundN ? dgSumStep / (double)dgBoundN : 0.0, dgMaxStep,
								dgBoundN ? dgSumResid / (double)dgBoundN : 0.0, dgMaxResid);
#endif
						}
					}
#endif // TEXTURE_DATACOLOR_FEATHER_RINGS
					textureDiffuse = newTex;
					TEXTURE_DIAG("Data-colored %d unobserved faces in %d component tiles (%d atlas rows"
						" = %.3f of atlas, capped at real-texture density %.1f px/unit,"
						" nearest-fill + %d seam-smooth iters)",
						N, nComp, extraRows,
						(oldRows + extraRows) > 0 ? (double)extraRows / (double)(oldRows + extraRows) : 0.0,
						texelDensity, (int)TEXTURE_DATACOLOR_SEAM_SMOOTH_ITERS);
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
					TEXTURE_DIAG("Data-colored %d unobserved faces (surface-propagated quadratic patch, %d atlas rows, %d smooth iters)", N, extraRows, (int)TEXTURE_DATACOLOR_SMOOTH_ITERS);
#endif // TEXTURE_DATACOLOR_COMPONENT_BAKE (per-face cell fallback)
				}
			}
		}
#endif // TEXTURE_DATACOLOR_UNOBSERVED && TEXTURE_DATACOLOR_BAKE

#if TEXTURE_OUTWARD_DILATE_PX > 0 && TEXTURE_ATLAS_COVERAGE_GUTTER
		// ------------------------------------------------------------
		// COVERAGE GUTTER + outward dilate. Runs HERE, last, because it must see the
		// FINAL colour of every texel a face owns: seam levelling, the feather band and
		// the synthesized fill tiles have all been written by this point. Growing a
		// gutter any earlier makes it stale, and a stale gutter is sampled at every
		// triangle edge by bilinear filtering and by the INTER_AREA final resize.
		// See TEXTURE_ATLAS_COVERAGE_GUTTER for the full argument.
		// ------------------------------------------------------------
		{
			TEX_PROFILE_SCOPE("GenerateTexture: coverage gutter + outward dilate");
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

			// 1) COVERAGE: rasterise the final texcoords of every face that survives.
			//    Both kinds count -- real texture patches and synthesized fill tiles --
			//    because the gutter has to be continuous across the boundary between
			//    them, which is precisely where the crack shows.
			known.setTo(0);
			const FIndex nFaces = (FIndex)faces.size();
			std::vector<uint8_t> dropF(nFaces, 0);
			for (const FIndex f : unobservedToDelete)
				if (f < nFaces)
					dropF[f] = 1;
			// Must be at least the bake outset, or this pass would clear the very rim the
			// fill bake and the seam feather just extrapolated past their triangle edges
			// (TEXTURE_DATACOLOR_BAKE_OUTSET_PX) and reflood it from the raw neighbours --
			// putting the hairline straight back.
			const float outset = std::max((float)TEXTURE_ATLAS_COVERAGE_OUTSET_PX,
										  (float)TEXTURE_DATACOLOR_BAKE_OUTSET_PX);
			const TexCoord* const __restrict pUV = faceTexcoords.data();
			const uint8_t* const __restrict pDrop = dropF.data();
			// dynamic, not static: cost per face is its atlas AREA, which spans orders of
			// magnitude here (a 5 px fill-tile triangle next to a large foreground patch
			// face), so a static split leaves threads idle. 1024 keeps the scheduling
			// overhead negligible against ~1.8M iterations while still balancing.
#pragma omp parallel for schedule(dynamic, 1024)
			for (int_t fi = 0; fi < (int_t)nFaces; ++fi) {
				if (pDrop[(size_t)fi])
					continue;
				const TexCoord* const __restrict tc = pUV + (size_t)fi * 3;
				const float x0 = tc[0].x, y0 = tc[0].y;
				const float x1 = tc[1].x, y1 = tc[1].y;
				const float x2 = tc[2].x, y2 = tc[2].y;
				// A face that never received atlas coordinates (no patch and no fill
				// tile) still carries the zero triple; rasterising it would nail a bogus
				// blob of coverage at the atlas origin.
				if (x0 == 0.f && y0 == 0.f && x1 == 0.f && y1 == 0.f && x2 == 0.f && y2 == 0.f)
					continue;
				// Coverage is additive -- it can only PRESERVE texels, never clear them --
				// so a false positive is harmless, but a NaN would make the bbox
				// arithmetic below meaningless.
				if (!(std::isfinite(x0) && std::isfinite(y0) && std::isfinite(x1) &&
					  std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2)))
					continue;
				int minx = (int)std::floor(std::min(x0, std::min(x1, x2)) - outset - 1.f);
				int maxx = (int)std::ceil (std::max(x0, std::max(x1, x2)) + outset + 1.f);
				int miny = (int)std::floor(std::min(y0, std::min(y1, y2)) - outset - 1.f);
				int maxy = (int)std::ceil (std::max(y0, std::max(y1, y2)) + outset + 1.f);
				minx = std::max(minx, 0); maxx = std::min(maxx, W - 1);
				miny = std::max(miny, 0); maxy = std::min(maxy, H - 1);
				if (minx > maxx || miny > maxy)
					continue;
				const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
				if (fabsf(denom) < 1e-6f) {
					// Degenerate in the atlas (a sliver, or a fill-tile foldover): mark
					// the whole bbox rather than nothing, so a thin face keeps its texels
					// instead of being cleared out from under itself.
					for (int py = miny; py <= maxy; ++py) {
						uint8_t* krow = kbase + (size_t)py * kstride;
						for (int px = minx; px <= maxx; ++px)
							krow[px] = 1;
					}
					continue;
				}
				const float invDen = 1.f / denom;
				// Per-barycentric tolerance worth exactly `outset` PIXELS. w_k falls off
				// at 1/altitude_k across the triangle and altitude_k = |denom| / |edge
				// opposite k|, so the pixel outset converts to outset*|edge_k|/|denom|.
				// Using a flat bary epsilon instead would outset large triangles by many
				// pixels and small ones by none.
				const float ad = fabsf(denom);
				const float L0 = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
				const float L1 = std::sqrt((x2 - x0) * (x2 - x0) + (y2 - y0) * (y2 - y0));
				const float L2 = std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
				const float t0 = outset * L0 / ad, t1 = outset * L1 / ad, t2 = outset * L2 / ad;
				for (int py = miny; py <= maxy; ++py) {
					uint8_t* krow = kbase + (size_t)py * kstride;
					const float fy = py + 0.5f;
					for (int px = minx; px <= maxx; ++px) {
						const float fx = px + 0.5f;
						const float w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * invDen;
						const float w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * invDen;
						const float w2 = 1.f - w0 - w1;
						if (w0 < -t0 || w1 < -t1 || w2 < -t2)
							continue;
						krow[px] = 1; // overlapping faces all store the same 1 -- benign
					}
				}
			}

			// 2) Everything no face covers becomes gutter, whatever it currently holds.
			//    That includes the raw source-image pixels each patch rect brought in
			//    around its triangles -- the pixels the feather never touches and the
			//    old colEmpty-based flood could never reach, because they were never
			//    empty. This is the step that actually removes the seam hairline.
			//
			//    FUSED with the flood's initial frontier build. Both are full sweeps of a
			//    quarter-billion-texel atlas driven by the SAME `known` mask, and `known`
			//    is complete before either runs -- so the clear cannot change what the
			//    frontier test sees, and one pass does the work of two. The neighbour test
			//    is the expensive half (9 reads per empty texel, ~1.2G reads here), so
			//    folding it into the pass that is already streaming those rows saves a full
			//    traversal of the atlas plus its cache misses.
			const int nThreads = omp_get_max_threads();
			const cv::Vec3b vEmpty(eb, eg, er);
			std::vector<std::vector<cv::Point>> tlsFront((size_t)nThreads);
			std::vector<long long> tlsCleared((size_t)nThreads, 0);
#pragma omp parallel
			{
				const int tid = omp_get_thread_num();
				std::vector<cv::Point>& loc = tlsFront[(size_t)tid];
				long long nloc = 0;
#pragma omp for schedule(static)
				for (int y = 0; y < H; ++y) {
					cv::Vec3b* const __restrict row = tbase + (size_t)y * tstride;
					const uint8_t* const __restrict krow = kbase + (size_t)y * kstride;
					const uint8_t* const kup = (y > 0) ? kbase + (size_t)(y - 1) * kstride : nullptr;
					const uint8_t* const kdn = (y + 1 < H) ? kbase + (size_t)(y + 1) * kstride : nullptr;
					for (int x = 0; x < W; ++x) {
						if (krow[x])
							continue;
						row[x] = vEmpty;
						++nloc;
						// 8-neighbourhood, rows hoisted out of the inner test
						const int xlo = x > 0 ? x - 1 : 0, xhi = x + 1 < W ? x + 1 : W - 1;
						bool adj = false;
						for (int xx = xlo; xx <= xhi && !adj; ++xx) {
							if (kup && kup[xx] == 1) adj = true;
							else if (kdn && kdn[xx] == 1) adj = true;
							else if (xx != x && krow[xx] == 1) adj = true;
						}
						if (adj) loc.emplace_back(x, y);
					}
				}
				tlsCleared[(size_t)tid] = nloc;
			}
			long long nCleared = 0;
			for (const long long n : tlsCleared) nCleared += n;
			std::vector<cv::Point> frontier, next;
			{
				size_t nf = 0;
				for (const std::vector<cv::Point>& v : tlsFront) nf += v.size();
				frontier.reserve(nf);
				for (std::vector<cv::Point>& v : tlsFront) {
					frontier.insert(frontier.end(), v.begin(), v.end());
					std::vector<cv::Point>().swap(v); // release as we go: nf can be millions
				}
			}
			TEXTURE_DIAG("[ATLAS-GUTTER] %u faces (%zu dropped) -> cleared %lld of %lld texels"
				" (%.1f%%) and regrew them from the final per-face colours (seed frontier %zu)",
				nFaces, unobservedToDelete.size(), nCleared, (long long)H * (long long)W,
				100.0 * (double)nCleared / (double)std::max<long long>(1, (long long)H * (long long)W),
				frontier.size());

			// 3) REGROW -- the same frontier flood as before, but every seed is now a
			//    FINAL texel (seam-levelled, feathered, or synthesized fill), so no
			//    foreign colour is left anywhere a filter kernel or a mip level can
			//    reach from inside a triangle.
			// Ring cap: bounded margin, OR a full flood of every colEmpty texel (mip-safe,
			// kills the dark rim). (H + W) is >= the max Manhattan distance from any empty
			// texel to a known one, so it always completes the flood; the loop still exits
			// early via `!frontier.empty()` once nothing is left to fill.
			const int dilateRings = TEXTURE_ATLAS_FULL_FLOOD ? (H + W) : TEXTURE_OUTWARD_DILATE_PX;
			// Hoisted out of the ring loop: with the coverage gutter the early rings carry
			// millions of texels, and reallocating + value-initialising that buffer every
			// ring was pure overhead. resize() keeps the capacity once the largest ring has
			// been seen, and the per-thread wave buffers likewise reuse their storage.
			std::vector<cv::Vec3b> fillCol;
			std::vector<std::vector<cv::Point>> tlsNext((size_t)nThreads);
			for (int it = 0; it < dilateRings && !frontier.empty(); ++it) {
				const int nFront = (int)frontier.size();
				fillCol.resize((size_t)nFront);
				const cv::Point* const __restrict pFront = frontier.data();
				cv::Vec3b* const __restrict pFill = fillCol.data();
#pragma omp parallel for schedule(static)
				for (int i = 0; i < nFront; ++i) {
					const int x = pFront[i].x, y = pFront[i].y;
					// Plain ints rather than cv::Vec3i: the operator+= / conversion pair kept
					// this inner loop out of the optimiser's reach for no benefit at 3 lanes.
					// x-bounds hoisted out of the neighbour loop for the same reason.
					int sb = 0, sg = 0, sr = 0, n = 0;
					const int xlo = x > 0 ? x - 1 : 0, xhi = x + 1 < W ? x + 1 : W - 1;
					for (int dy = -1; dy <= 1; ++dy) {
						const int yy = y + dy;
						if ((unsigned)yy >= (unsigned)H) continue;
						const uint8_t* const __restrict knrow = kbase + (size_t)yy * kstride;
						const cv::Vec3b* const __restrict tnrow = tbase + (size_t)yy * tstride;
						for (int xx = xlo; xx <= xhi; ++xx) {
							if (knrow[xx] != 1) continue;
							const cv::Vec3b& c = tnrow[xx];
							sb += c[0]; sg += c[1]; sr += c[2]; ++n;
						}
					}
					pFill[i] = n ? cv::Vec3b((uchar)(sb / n), (uchar)(sg / n), (uchar)(sr / n)) : vEmpty;
				}
#pragma omp parallel for schedule(static)
				for (int i = 0; i < nFront; ++i) {
					const int x = pFront[i].x, y = pFront[i].y;
					uint8_t& kxy = kbase[(size_t)y * kstride + x];
					if (kxy == 1) continue;
					tbase[(size_t)y * tstride + x] = pFill[i];
					kxy = 1;
				}
				// Build the next wave in parallel. This was the last serial loop in the
				// flood, and with the coverage gutter it walks 9 neighbours for every one of
				// ~132M filled texels -- so once the other three loops are threaded it is by
				// far the dominant cost of the pass.
				//
				// The `= 2` dedup marker races benignly: two threads may both observe 0 and
				// both push the same texel. A duplicate costs one recomputation of an
				// identical colour, the commit loop's `kxy == 1` guard drops the second
				// write, and the reset below is idempotent. The flood is a Jacobi wave, so
				// wave ORDER does not affect the result either.
#pragma omp parallel
				{
					std::vector<cv::Point>& loc = tlsNext[(size_t)omp_get_thread_num()];
					loc.clear();
#pragma omp for schedule(static)
					for (int i = 0; i < nFront; ++i) {
						const int x = pFront[i].x, y = pFront[i].y;
						const int xlo = x > 0 ? x - 1 : 0, xhi = x + 1 < W ? x + 1 : W - 1;
						for (int dy = -1; dy <= 1; ++dy) {
							const int yy = y + dy;
							if ((unsigned)yy >= (unsigned)H) continue;
							uint8_t* const knrow = kbase + (size_t)yy * kstride;
							for (int xx = xlo; xx <= xhi; ++xx) {
								if (knrow[xx] != 0) continue;
								knrow[xx] = 2; // tentatively queued (dedup within wave)
								loc.emplace_back(xx, yy);
							}
						}
					}
				}
				next.clear();
				{
					size_t nn = 0;
					for (const std::vector<cv::Point>& v : tlsNext) nn += v.size();
					next.reserve(nn);
					for (const std::vector<cv::Point>& v : tlsNext)
						next.insert(next.end(), v.begin(), v.end());
				}
				const int nNext = (int)next.size();
				cv::Point* const __restrict pNext = next.data();
#pragma omp parallel for schedule(static)
				for (int i = 0; i < nNext; ++i) {
					uint8_t& kp = kbase[(size_t)pNext[i].y * kstride + pNext[i].x];
					if (kp == 2) kp = 0;
				}
				frontier.swap(next);
			}
		}
		// The flood's wave buffers are the one thing here that is sized by the DATA rather
		// than by the atlas: `frontier`/`next` hold a cv::Point (8 B) per texel on the
		// advancing front, and with the coverage gutter the first front is most of the
		// cleared set rather than the handful of texels the old colEmpty flood started
		// from. The seed count is logged above and this samples the peak, so if it turns
		// out to be a real spike the fix is a 4-byte linear index instead of a cv::Point
		// (halves it) -- measured first rather than assumed, since the front is the
		// BOUNDARY of the cleared region, not the region itself, and its size depends
		// entirely on how the uncovered texels are distributed.
		LogPeakMem("GenTex: after coverage gutter");
#endif // TEXTURE_OUTWARD_DILATE_PX && TEXTURE_ATLAS_COVERAGE_GUTTER

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
	return true;
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
		TEXTURE_DIAG("Assigning the best view to each face completed: %u faces (%s)", mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
	LogPeakMem("after FaceViewSelection");

	// generate the texture image and atlas
	{
		TD_TIMER_STARTD();
		if (!texture.GenerateTexture(bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty, fSharpnessWeight, nMaxTextureSize))
			return false;
		DEBUG_EXTRA("Generating texture atlas and image completed: %u patches, %u image size (%s)", texture.texturePatches.GetSize(), mesh.textureDiffuse.width(), TD_TIMER_GET_FMT().c_str());
		if (TEXTURE_DIAG_ENABLED()) {
			// [ATLAS-FINAL] What the atlas actually came out as. On the TEXTURE_DIAG build
			// switch, paired with RefineMesh's [REFINE-FACES] on REFINE_DIAG: the two are
			// the texel-budget audit trail and are only meaningful read together, so a
			// -DOPENMVS_DIAG=1 build carries both. ReconstructMesh's [ATLAS] budget line
			// and the "SMALLER THAN WANTED" notice are on the same gates as of this
			// change, so the whole atlas trail appears and disappears together; the one
			// [ATLAS] line still ungated is the cap that actually DECIMATES the mesh,
			// which changes the deliverable rather than describing the budget.
			//
			// The line above reports textureDiffuse.width() ONLY. The atlas is not square --
			// measured 14876 x 16054 on one run -- so reading that single number as the atlas
			// size understates the area by ~8% and makes the fill look far worse than it is.
			// That misreading produced a wrong "18% of the atlas is wasted" conclusion that
			// had to be walked back; the real figure came from dividing the ATLAS-GUTTER texel
			// count by the width. Report both dimensions and the fill directly.
			//
			// texels/face is the quantity ReconstructMesh's atlas cap is trying to hold at 64
			// (D^2*0.97/64). It is computed here on the POST-refine mesh, which is the one that
			// actually gets textured -- see [REFINE-FACES].
			const double atlasW = (double)mesh.textureDiffuse.width();
			const double atlasH = (double)mesh.textureDiffuse.height();
			const double atlasPx = atlasW * atlasH;
			const int hostDim = GetOpenGLMaxTextureSize();
			const double hostPx = (double)hostDim * (double)hostDim;
			const unsigned nF = mesh.faces.GetSize();
			// The host comparison is stated in words rather than as a bare ratio. It used to
			// print "%.3f of the %d px host maximum", which on a deliberately-pinned atlas
			// LARGER than the build machine's GL limit reads as "3.991 of the 16384 px host
			// maximum" -- indistinguishable from a scaling bug, and this is a normal, wanted
			// state when the deliverable targets better hardware than the box building it.
			// Say which side of the limit the atlas landed on and leave the ratio out.
			const bool bOverHost = (hostDim > 0 && (atlasW > hostDim || atlasH > hostDim));
			String hostNote;
			if (hostDim <= 0)
				hostNote = _T("no GL limit probed");
			else if (bOverHost)
				hostNote = String::FormatString(
					"EXCEEDS this build host's %d px GL limit -- fine if the consumer's GPU"
					" is larger, not sampleable here", hostDim);
			else
				hostNote = String::FormatString("within this host's %d px GL limit (%.1f Mpx)",
					hostDim, hostPx * 1e-6);
			TEXTURE_DIAG("[ATLAS-FINAL] %.0f x %.0f = %.1f Mpx | %s"
				" | %u faces -> %.0f texels/face | %u patches",
				atlasW, atlasH, atlasPx * 1e-6, hostNote.c_str(),
				nF, nF ? atlasPx / (double)nF : 0.0,
				texture.texturePatches.GetSize());
		}

		// Apply the boundary-sheet removal decided during the data-colour pass. Done
		// HERE, after texturing, because texturePatches / components / faceTexcoords are
		// all face-indexed and deleting mid-pipeline would invalidate them. The atlas is
		// already final; these faces simply stop referencing it (and never consumed
		// data-colour rows in the first place, which is where the atlas saving comes from).
		if (!texture.unobservedToDelete.empty()) {
			const Mesh::FIndex nF = mesh.faces.GetSize();
			std::vector<uint8_t> drop(nF, 0);
			for (const Mesh::FIndex f : texture.unobservedToDelete)
				if (f < nF) drop[f] = 1;
			const bool hasUV = (mesh.faceTexcoords.GetSize() == nF * 3);
			Mesh::FaceArr keptFaces;      keptFaces.Reserve(nF);
			Mesh::TexCoordArr keptUV;     if (hasUV) keptUV.Reserve(nF * 3);
			for (Mesh::FIndex f = 0; f < nF; ++f) {
				if (drop[f])
					continue;
				keptFaces.Insert(mesh.faces[f]);
				if (hasUV)
					for (int k = 0; k < 3; ++k)
						keptUV.Insert(mesh.faceTexcoords[f * 3 + k]);
			}
			const Mesh::FIndex removed = nF - keptFaces.GetSize();
			mesh.faces.Swap(keptFaces);
			if (hasUV)
				mesh.faceTexcoords.Swap(keptUV);
			// Vertices left unreferenced are harmless (no consumer indexes by vertex
			// count here, and nothing runs Clean after texturing), so they are not
			// compacted -- doing so would mean remapping every face index for no gain.
			DEBUG("[TEX-SHEET] removed %u boundary-sheet faces after texturing"
				" -> %u faces remain", (unsigned)removed, (unsigned)mesh.faces.GetSize());
		}

#if TEXTURE_BOUNDARY_SMOOTH_ENABLED
		// FINAL SILHOUETTE POLISH -- see TEXTURE_BOUNDARY_SMOOTH_*. Runs here because this is the
		// only point at which the boundary the viewer actually sees exists: patch-building fixed
		// the observed region per-triangle, and the delete above finished carving it.
		{
			TD_TIMER_STARTD();
			// Adjacency MUST be rebuilt -- the face array was just rewritten above, so any
			// previously-computed faceFaces refers to the old indices.
			//
			// Do NOT reach for Mesh::EmptyExtra() here, however tempting the symmetry with
			// ListVertexFaces is: it clears faceTexcoords AND releases textureDiffuse, i.e. it
			// would silently throw away the entire texture this stage just produced. Only the two
			// adjacency arrays are invalid, so only those are rebuilt. ListIncidenteFaceFaces
			// asserts vertexFaces.size() == vertices.size(), hence the order.
			mesh.vertexFaces.Empty();
			mesh.faceFaces.Empty();
			mesh.ListIncidenteFaces();
			mesh.ListIncidenteFaceFaces();
			const Mesh::FIndex nF = mesh.faces.GetSize();
			const size_t NV = mesh.vertices.GetSize();
			if (nF > 0 && NV > 0 && mesh.faceFaces.GetSize() == nF) {
				// Border-curve neighbours per vertex. Edge e of face f joins fc[e] and
				// fc[(e+1)%3] -- the same convention the border-loop union-find above uses --
				// and an edge is on the border exactly when faceFaces[f][e] == NO_ID.
				// cnt == 2 is a clean curve vertex; cnt > 2 is a pinch/junction and is skipped.
				std::vector<uint32_t> nb0(NV, NO_ID), nb1(NV, NO_ID);
				std::vector<uint8_t> cnt(NV, 0);
				for (Mesh::FIndex f = 0; f < nF; ++f) {
					const Mesh::FaceFaces& ff = mesh.faceFaces[f];
					const Mesh::Face& fc = mesh.faces[f];
					for (int e = 0; e < 3; ++e) {
						if (ff[e] != NO_ID)
							continue;
						const uint32_t ab[2] = { fc[e], fc[(e + 1) % 3] };
						for (int s = 0; s < 2; ++s) {
							const uint32_t v = ab[s], w = ab[1 - s];
							if (v >= NV || w >= NV)
								continue;
							if (cnt[v] == 0) { nb0[v] = w; cnt[v] = 1; }
							else if (cnt[v] == 1) { if (nb0[v] != w) { nb1[v] = w; cnt[v] = 2; } }
							else if (cnt[v] == 2) { if (nb0[v] != w && nb1[v] != w) cnt[v] = 3; }
						}
					}
				}
				size_t nCurve = 0, nJunction = 0;
				for (size_t v = 0; v < NV; ++v) {
					if (cnt[v] == 2) ++nCurve;
					else if (cnt[v] > 2) ++nJunction;
				}
				if (nCurve > 0) {
					const float lambda = float(TEXTURE_BOUNDARY_SMOOTH_LAMBDA_X100) / 100.f;
					const float mu = -float(TEXTURE_BOUNDARY_SMOOTH_MU_X100) / 100.f;
					// SIMULTANEOUS (Jacobi) update, deliberately -- unlike the contour majority
					// filter, which needed sequential updates to stop oscillating. A curve
					// smoother must read one consistent state or vertices chase their
					// already-moved neighbours and the whole outline creeps along itself.
					std::vector<Mesh::Vertex> cur(NV), nxt(NV);
					for (size_t v = 0; v < NV; ++v)
						cur[v] = mesh.vertices[(Mesh::VIndex)v];
					nxt = cur;
					for (int it = 0; it < TEXTURE_BOUNDARY_SMOOTH_ITERS * 2; ++it) {
						const float step = (it & 1) ? mu : lambda;
						for (size_t v = 0; v < NV; ++v) {
							if (cnt[v] != 2)
								continue;
							const Mesh::Vertex& p = cur[v];
							const Mesh::Vertex& a = cur[nb0[v]];
							const Mesh::Vertex& b = cur[nb1[v]];
							nxt[v] = Mesh::Vertex(
								p.x + step * ((a.x + b.x) * 0.5f - p.x),
								p.y + step * ((a.y + b.y) * 0.5f - p.y),
								p.z + step * ((a.z + b.z) * 0.5f - p.z));
						}
						cur.swap(nxt);
					}
					// Commit ONLY the curve vertices: everything else is bit-identical anyway,
					// and this makes it impossible for the pass to disturb interior geometry.
					double moved = 0.0, movedMax = 0.0;
					for (size_t v = 0; v < NV; ++v) {
						if (cnt[v] != 2)
							continue;
						const Mesh::Vertex& o = mesh.vertices[(Mesh::VIndex)v];
						const Mesh::Vertex& n = cur[v];
						const double d = std::sqrt((double)SQUARE(n.x - o.x) +
							(double)SQUARE(n.y - o.y) + (double)SQUARE(n.z - o.z));
						moved += d;
						if (d > movedMax) movedMax = d;
						mesh.vertices[(Mesh::VIndex)v] = n;
					}
					DEBUG("[TEX-SMOOTH] silhouette Taubin: %zu curve vertices moved"
						" (mean %.4g, max %.4g world units), %zu junction vertices skipped,"
						" %d lambda/mu pairs (%.2f/%.2f) (%s)",
						nCurve, moved / (double)nCurve, movedMax, nJunction,
						(int)TEXTURE_BOUNDARY_SMOOTH_ITERS, lambda, mu,
						TD_TIMER_GET_FMT().c_str());
				} else {
					// nothing changed on the mesh -> trace, not a result
					TEXTURE_DIAG("[TEX-SMOOTH] silhouette Taubin: no clean border-curve vertices"
						" (%zu junctions) -- nothing to smooth", nJunction);
				}
			}
		}
#endif // TEXTURE_BOUNDARY_SMOOTH_ENABLED
	}
	LogPeakMem("after GenerateTexture");

	return true;
} // TextureMesh
/*----------------------------------------------------------------*/
