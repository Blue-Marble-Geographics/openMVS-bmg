/*
* SceneReconstruct.cpp
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

#define FIX_MANIFOLD // No manifold repair at all
#define MANIFOLD_FIXUP
#undef PRE_OPENMVS21
#undef EARLY_OUT_WEIGHTING
#define PARALLEL_GRAPH_CUT_EXTRACTION

// --- Optional point-cloud pre-filter (applied AFTER StatisticalOutlierRemoval,
// BEFORE Delaunay insertion). Defaults to OFF.
//
// Confidence-weighted filter: uses the per-point pointWeights data
//    (per-view confidence). For each point we take the MAX weight across
//    its views (the strongest single-view evidence). Weight distributions
//    are typically right-skewed (lognormal-ish), so a symmetric mean-kσ
//    threshold goes negative and drops nothing. Instead we drop the bottom
//    KPCT percent by quantile -- robust to distribution shape.
//    KPCT=10 means drop bottom 10% (lowest-confidence). Set to 0 to disable.
#ifndef RECONSTRUCT_CONFIDENCE_FILTER
#define RECONSTRUCT_CONFIDENCE_FILTER 0
#endif

// =====================================================================
// SCRREC_OPT_PREFETCH: latency-hiding optimizations (semantics-preserving).
//   A) Software prefetch in the deferred-apply scatter loop
//      (per-vertex weighting). Warms `cell_info_t.f[]` lines ~8 atomics
//      ahead so the CAS-loop pipelines instead of stalling on DRAM.
//   B) Software prefetch in BuildGraphNodesAndEdges Phase 3. Warms
//      neighbor `infoCells[]` and `qualAngles[]` cachelines for the
//      next iteration before the random gather happens.
//   C) _mm_pause() in AtomicAddFloat's CAS retry loop. Yields the µop
//      port under contention; helps the long-tail of histogram cells
//      that get hit by many threads at once.
// All three are pure latency hiding -- no math changes, no data structure
// changes, identical results. Set to 0 to A/B test.
// =====================================================================
#ifndef SCRREC_OPT_PREFETCH
#define SCRREC_OPT_PREFETCH 1
#endif
#ifndef RECONSTRUCT_CONFIDENCE_FILTER_KPCT
#define RECONSTRUCT_CONFIDENCE_FILTER_KPCT 3
#endif

// RECONSTRUCT_EARLY_CONFIDENCE_FILTER: Run the confidence filter BEFORE
// spatial_sort / SOR / permuteScatter2 instead of after.  Same drop logic
// (bottom KPCT% by max-per-point view weight), same KPCT setting.
// When this is 1, the late confidence filter is automatically suppressed
// to prevent double-dropping.  Effect: every later O(N) and O(N log N)
// stage sees a smaller N (kd-tree build, KNN search, sort, DT insertion,
// per-point cell walks).  Constraints:
//   - Skipped if pointWeights are not available (no-op, falls through).
//   - Skipped in ROI mode -- ProcessPoints<true> compacts indices[] to
//     post-filter indices, breaking original-cloud-index lookups into
//     pointWeightsOffsets/Sizes.  (The late filter has the same constraint
//     implicitly; it just isn't usually enabled together with ROI.)
// NOTE: Found to drop quality-bearing boundary/feature points on aerial
// scenes (low-confidence !=  noise here -- it correlates with low view
// count which marks boundaries and thin features). Reverted to 0.
// Set to 0 to revert to the original ordering.
#ifndef RECONSTRUCT_EARLY_CONFIDENCE_FILTER
#define RECONSTRUCT_EARLY_CONFIDENCE_FILTER 0
#endif

// RECONSTRUCT_VOXEL_PREFILTER: Density-based pre-decimation. For each input
// point compute a voxel-grid cell index; collapse all points in the same
// cell to a single representative (highest-confidence wins; first-arrival
// if no pointWeights). Effect: removes redundant near-duplicates from
// over-sampled flat regions WITHOUT removing isolated boundary/feature
// points (sparse cells keep their single point unconditionally). Unlike
// the confidence filter this is data-density driven, so it should preserve
// silhouette / thin-feature detail.
//
// AUTO-TUNED VOXEL SIZE: Voxel size is derived from the data's *densest*
// region rather than hardcoded. We probe a coarse 64^3 occupancy grid
// over the active bbox, find the peak bin, and estimate dense-region
// inter-point spacing assuming a 2D surface within that bin:
//      bin_side = max_span / 64
//      dense_spacing ~= bin_side / sqrt(K_peak)
//      voxel = dense_spacing * VOXEL_SAFETY
// VOXEL_SAFETY < 1.0 = strictly more conservative than the dense
// spacing (only true duplicates merge). VOXEL_SAFETY = 1.0 = match
// dense spacing (some mild merging where over-sampled). VOXEL_SAFETY > 1.0
// = aggressive (collapses real geometry; not recommended).
//
// Resulting kGrid is clamped to [256, 65535] so the voxel can never grow
// larger than max_span/256 (a hard upper bound that prevents catastrophic
// merge on degenerate / sparse data) nor smaller than max_span/65535
// (16-bit-per-axis grid limit).
//
// Cost: one bbox pass + one density probe (~50ms) + one voxel-key build +
// tbb::parallel_sort on (key, idx).
// Side benefit: post-sort indices[] are in approximately z-order, which
// often warms up the subsequent CGAL spatial_sort.
//
// Constraints:
//   - Skipped in ROI mode (same indices[] semantics issue as above).
//   - Skipped if numVertices == 0.
//   - Skipped if dense-region probe yields uninformative result (K_peak
//     < some threshold) -- means the cloud is uniformly sparse, no
//     redundancy to remove.
// Set to 0 to disable.
#ifndef RECONSTRUCT_VOXEL_PREFILTER
#define RECONSTRUCT_VOXEL_PREFILTER 1
#endif

// RECONSTRUCT_VOXEL_SAFETY: Multiplier applied to estimated dense-region
// inter-point spacing to derive voxel size.
//   < 1.0  : voxel smaller than dense spacing -> only exact duplicates merge
//   1.0    : voxel matches dense spacing -> light merging in dense regions,
//            none in sparse regions (RECOMMENDED starting point)
//   > 1.0  : voxel larger than dense spacing -> real geometry can be lost
// Stored as int * 0.01 so it's preprocessor-friendly. 100 = 1.00.
#ifndef RECONSTRUCT_VOXEL_SAFETY_X100
#define RECONSTRUCT_VOXEL_SAFETY_X100 100
#endif

// DIRECT_VIEW_EXPANSION: Instead of storing point indices in per-vertex
// allViews[] during insertion (random small_vector pushes), store vertex
// index per point (sequential writes), then expand views in a single
// cache-friendly sequential pass after insertion is complete.
// Set to 1 to enable the new path, 0 for the original path.
#ifndef DIRECT_VIEW_EXPANSION
#define DIRECT_VIEW_EXPANSION 1
#endif

// INSERTION_BFS_MAX_CELLS: Cap on the per-restart BFS cell-visit count
// during nearest-vertex search inside the distInsert>0 insertion path.
// 0 = unbounded (semantically exact: BFS terminates only when proven
//     no closer vertex exists).
// >0 = approximate: if BFS hits this cap, accept current best as nearest.
//     Worst case: pick a vertex slightly further than true nearest →
//     downstream "insert vs skip" check errs toward inserting (NOT
//     toward dropping legitimate points). Mesh quality preserved;
//     DT may grow slightly. Recommended: 32-64. Lower = faster + larger DT.
#ifndef INSERTION_BFS_MAX_CELLS
#define INSERTION_BFS_MAX_CELLS 64
#endif

// RECONSTRUCT_FAST_DISTINSERT: Skip the BFS-based 3D-nearest refinement
// during the distInsert>0 insertion path and decide insert/reject by
// running the per-view projection check directly against the 4 vertices
// of the cell containing the candidate point.
//
// Semantic relationship to the BFS path:
//   Original: find the truly-nearest existing vertex (3D Euclidean) via
//     BFS over cell-neighbors, then run the projection check vs that one
//     vertex. Reject iff that vertex projects within distInsert px in
//     all of the candidate's views.
//   Fast:    a candidate is rejected iff ANY of the 4 cell-vertices
//     containing the candidate projects within distInsert px in all of
//     the candidate's views. No BFS, no vertexMarks writes, no
//     tds_data().marker updates.
//
// Why this is close to (and arguably more faithful than) the original:
// the original uses 3D-nearest as a *heuristic* for "the vertex most
// likely to be close in projection". The fast path checks the 4
// containing-cell vertices directly against the actual projection
// criterion the user asked for ("no existing vertex within distInsert
// pixels"). It can occasionally accept a point the original rejected
// (if the true 3D-nearest was *outside* the containing cell and was the
// only projection-close vertex), but in dense regions where rejections
// concentrate, the containing cell almost always already holds at least
// one rejector.
//
// Cost saved: the BFS at ~lines 3990-4170 plus all vertexMarks /
// tds_data().marker bookkeeping plus the redundant projection check
// pass. On scenes with high rejection rates (e.g. --min-point-distance
// 2.5 dropping ~32% of inputs) this can save several seconds off DT
// insertion.
//
// Set to 0 to use the original BFS-refined path.
#ifndef RECONSTRUCT_FAST_DISTINSERT
#define RECONSTRUCT_FAST_DISTINSERT 1
#endif

// RECONSTRUCT_PARALLEL_PASS5: Parallelize the Morton-cell Pass 5 compaction
// that splits enumerated cells into finiteCells[] + finiteCellMeta[] (parallel
// scatter via two-pass count + prefix-sum) and hullFacets[] (sequential tail
// pass; infinite cells are <0.001% of total — ~300 of 53M on typical scenes).
//
// Original Pass 5 was a single serial loop reading isFinite[] and writing
// to finiteCells / finiteCellMeta / hullFacets. On 53M cells with random
// origMeta[perm[k]] reads it's bandwidth-bound at ~1-2s. The parallel scatter
// distributes the bandwidth across cores; tested wins ~0.5-1.5s on 16+ core
// systems.
//
// Correctness: the output finiteCells[] / finiteCellMeta[] order is preserved
// bit-for-bit relative to the serial version (each thread receives a
// contiguous chunk via schedule(static) and writes to a pre-computed offset,
// so the global output ordering matches the input scan order).
//
// Set to 0 to use the original serial Pass 5.
#ifndef RECONSTRUCT_PARALLEL_PASS5
#define RECONSTRUCT_PARALLEL_PASS5 1
#endif

// RECONSTRUCT_HILBERT_SORT: Use CGAL::hilbert_sort directly for pre-DT vertex
// ordering instead of CGAL::spatial_sort.  spatial_sort is hilbert_sort wrapped
// in a randomized splitter (random_shuffle on the first sqrt(N) elements,
// then a recursive median split) intended to harden against adversarial
// inputs.  For real-world point clouds (already roughly clustered by voxel
// pre-filter / SOR / etc.) the randomization isn't needed and just costs a
// pass.  Drop saves ~0.3-0.8s on 8M+ vertex scenes; DT insertion behavior
// is unchanged because the Hilbert curve gives the same locality benefit.
//
// Set to 0 to revert to CGAL::spatial_sort.
//
// NOTE: empirically this REGRESSES total time on scenes using
// --min-point-distance > 0 with RECONSTRUCT_FAST_DISTINSERT=1.  The
// random-prefix shuffle inside spatial_sort isn't just adversarial
// hardening: it seeds the DT with spatially scattered points first, so
// locate() lands in cells whose 4 vertices are representative
// neighbors, which makes the cell-corner distInsert rejection
// effective.  Pure Hilbert order walks the curve corner-to-corner,
// early insertions cluster, locate() lands in coarse neighborhoods,
// fewer points get rejected, vertex count grows ~2%, and the extra
// vertices cost downstream more than the sort saves.  Default OFF.
#ifndef RECONSTRUCT_HILBERT_SORT
#define RECONSTRUCT_HILBERT_SORT 0
#endif

// RECONSTRUCT_RADIX_MORTON: Replace std::sort(par_unseq, perm, ...) on the
// 63-bit Morton keys (cell-enum phase, ~53M cells) with a parallel 8-bit LSD
// radix sort.
//
// Comparison-sort on 53M elements costs ~N*log2(N) = 26*N compares, each of
// which does two indirect random reads from keys[]; total ~2.8B random reads.
// 8-pass LSD radix does 16*N indirect reads but in a single sweep per pass,
// so the prefetcher and DRAM page-locality help vastly more.
//
// Implementation: per-thread 256-bin histogram, column-major exclusive
// prefix-sum (per-thread per-bucket write cursors), parallel scatter with
// matching schedule(static) so each thread visits the same indices in the
// scatter pass as in the count pass.  Stable ordering across equal keys.
//
// Extra memory: one uint32 ping-pong buffer of size numAll (~212MB on 53M
// cells) + a tiny nT*256*size_t histogram (~32KB).  Freed before Pass 4.
//
// Set to 0 to revert to std::sort.
#ifndef RECONSTRUCT_RADIX_MORTON
#define RECONSTRUCT_RADIX_MORTON 1
#endif

// RECONSTRUCT_SKIP_SOR: Skip Statistical Outlier Removal in ReconstructMesh.
// When 1, all points pass (mask filled with 1); saves ~1.3s on 18M-point
// clouds + the 72MB verticesf kd-tree build. Set to 0 (default) to keep SOR.
#ifndef RECONSTRUCT_SKIP_SOR
#define RECONSTRUCT_SKIP_SOR 0
#endif

// POISSON_BOUNDARY_SMOOTH: Number of Laplacian relaxation iterations applied
// to boundary (open-edge) vertices of the Poisson mesh after trimming.
// Smooths the octree-aligned staircase left by SurfaceTrimmer. Interior
// vertices are never touched. 0 = disable. 3-5 = subtle, 8-12 = aggressive.
#ifndef POISSON_BOUNDARY_SMOOTH
#define POISSON_BOUNDARY_SMOOTH 48 // Was 96, 24 creates more pseudo edge.
#endif

// POISSON_DISTANCE_CULL: Distance-to-cloud face cull applied to the Poisson
// surface AFTER reconstruction + density trim + NaN sanitize, BEFORE returning
// from ReconstructMeshPoisson. Orthogonal to SurfaceTrimmer's density trim:
// trim removes low-confidence surface, this removes surface that floats too far
// from the actual input samples (Poisson's invented membranes / skirts /
// balloons over cars, water, sky, etc.). Poisson only invents geometry AWAY
// from real points, so "is this triangle near real data?" deletes exactly the
// crazy stuff and leaves everything built ON the cloud untouched.
//
// Mechanism: build a nanoflann kd-tree on the same finite input cloud written
// to the Poisson PLY, query the nearest input point for every mesh vertex, and
// cull a face when ALL THREE of its vertices are farther than
//     POISSON_CULL_FACTOR_X100/100 * medianSpacing
// from the cloud (medianSpacing = median nearest-neighbour spacing, the same
// statistic EstimatePoissonDepth computes). "All three" is the conservative
// policy: it keeps boundary faces that straddle the data edge (one or two far
// vertices) and only deletes faces fully detached from the data. Vertices and
// faces are then compacted exactly like the NaN-removal block.
//
// WAS DEFAULT OFF because a single GLOBAL threshold tied to the median
// (densest-region) NN spacing over-culls real surface: point density varies
// enormously across a scene (obliquely-viewed walls, distant ground, canopy
// fringe), so legitimate surface in sparse-but-real regions sits many
// median-spacings from the nearest sample and is indistinguishable from invented
// skirts by a global distance test. Measured: ~13.5% of faces removed at 3.5x,
// most of them good geometry.
//
// The threshold is now LOCAL (see the implementation): each vertex is compared
// against the point spacing measured where it actually sits, estimated from the
// K-th nearest neighbour distance, so the test is scale-invariant per region.
// That is the "local-density-adaptive replacement" the old note asked for, and it
// targets the axis SurfaceTrimmer cannot: trim removes LOW-DENSITY surface, which
// also removes real-but-sparse regions and is what makes datasets less complete.
// This removes surface that is FAR FROM ANY DATA regardless of its density.
//
// Backed by POISSON_CULL_MAX_FRACTION: if the cull wants more than that share of
// the mesh, it is abandoned wholesale rather than gutting the surface, so the
// worst case is a no-op.
//
// ENABLED (0 -> 1). The recorded negative result for this pass -- "removed ZERO faces,
// d1/s_local max 2.82 vs 3.50, no gap" -- was measured on the OKState corridor, where the
// spurious lobes SAT ON REAL (if wrong) dense-matching points. A distance-to-cloud test can
// obviously find nothing when there is data underneath, so that result says nothing about a
// scene where the lobes have no point support at all.
//
// MEASURED AND REVERTED (2026-08-19). The reasoning below was wrong, and the new [MESH-CULL]
// distribution line shows exactly why -- the failure is STRUCTURAL, not scene-specific:
//     d1/s_local over 1384602 vertices: p50=0.97 p90=2.40 p99=2.76 p99.9=2.81 max=2.83
//     (cull fires above 3.50) -> removed no faces
// on a scene where the dense cloud was inspected and has NO POINTS under the lobes at all. The
// LOCAL normalization defeats the test: s_local is the spacing measured at the NEAREST cloud
// point, so inside a void d1 and s_local grow TOGETHER and the ratio stays bounded. The pass
// therefore cannot separate "far from data in a sparse region" from "close to data in a dense
// region" -- the one distinction it exists to make. Nor can the factor be lowered into range:
// with p50=0.97 and p90=2.40 there is no gap between bulk and tail, so anything under 2.83
// starts cutting ordinary surface indiscriminately. The diagnostic says it outright: "a usable
// cull needs a GAP between the bulk and the tail."
//
// Fixing it would mean referencing the distance to a GLOBAL or footprint-derived scale rather
// than to the local spacing at the nearest point -- i.e. the occupancy test, not this one.
//
// Superseded on this data by simply setting --poisson-trim from the density distribution, which
// the new percentile line makes possible: the trim was BINDING (min 5.47 vs threshold 5.50), not
// inert, and the lobes survive only because their density sits above 5.5 while real terrain
// occupies a tight 11.7-13.4 band. Raising the trim into the 8.8 (p5) region removes the tail.
//
// (The original note below is retained for the record; the OKState zero-removal result it
// discusses is now known to be the same structural failure, not a data-support difference.)
// Every other criterion tried on this scene failed for a structural reason:
//   * SurfaceTrimmer density trim  -- removes LOW-density surface; extrapolated lobes can
//     carry ordinary density, and a 1/3/7 sweep moved only 0.9-2.5%
//   * aRatio 0.01 -> 0.001         -- merge knob only, ~2.8x on a 2.5% base, no visible change
//   * alpha-tighten, rim erode     -- both touch BORDER faces only, and the mesh arrives
//     near-closed (1264 border verts, 0.1%), so there is no rim for them to work on
//   * --number-views-fuse, low-view filters -- nothing to filter; there are no points there
//
// SAFETY: the cull is conservative twice over. A face goes only when ALL THREE vertices are
// beyond the threshold (so boundary faces straddling the data edge are kept), the threshold is
// LOCAL (compared against the point spacing measured where each vertex sits, so sparse-but-real
// regions are judged on their own scale rather than against the dense interior), and
// POISSON_CULL_MAX_FRACTION abandons the whole pass if it wants more than 6% of the mesh. Worst
// case is therefore a logged no-op, not a gutted surface.
//
// WATCH: the cull's own log line reports the removed count and fraction. If it self-abandons at
// the 6% guard, that is the signal to check whether it is reaching real geometry before simply
// raising POISSON_CULL_MAX_FRACTION.
#ifndef POISSON_DISTANCE_CULL
#define POISSON_DISTANCE_CULL 0
#endif
// Abandon the distance cull entirely if it would remove more than this fraction of
// faces -- see the SAFETY GUARD note at the call site. Incompleteness is worse than
// surplus geometry, so this fails toward keeping everything.
#ifndef POISSON_CULL_MAX_FRACTION
#define POISSON_CULL_MAX_FRACTION 0.06
#endif

// POISSON_CULL_FACTOR_X100: distance threshold as a multiple of the median NN
// spacing, stored x100 (preprocessor-friendly). 350 = 3.5x. Higher = more
// conservative (only clearly-invented surface removed); lower = more aggressive
// (can start eating real boundary). Safe starting range 300-400.
#ifndef POISSON_CULL_FACTOR_X100
#define POISSON_CULL_FACTOR_X100 350
#endif

// POISSON_ADAPTIVE_TRIM: spatially-varying replacement for SurfaceTrimmer's
// single global density threshold. When enabled, the standard SurfaceTrimmer
// call is skipped and a custom trim is applied to the (untrimmed) Poisson mesh
// using each vertex's screened-Poisson density value together with its position
// in the input cloud's XY footprint:
//   - DEEP INTERIOR of the footprint -> threshold = trimThreshold (e.g. 5.5) so
//     low-density interior fill (water) is retained.
//   - NEAR / BEYOND the footprint PERIMETER -> threshold ramps up to
//     trimThreshold * (POISSON_TRIM_EDGE_MULT_X100/100) so Poisson's balloon /
//     extrapolation past the edge of the data is removed.
// Footprint = coarse XY occupancy grid (cell = POISSON_TRIM_CELL_FACTOR_X100/100
// * median NN spacing), dilated by POISSON_TRIM_CLOSE_CELLS to bridge shoreline
// gaps, then hole-filled via an exterior flood-fill so interior water bodies
// count as interior regardless of size. A 2-pass chamfer distance transform
// gives distance-to-perimeter; the per-vertex edge factor
// e = clamp(1 - dist/POISSON_TRIM_RAMP_CELLS, 0, 1) maps 0 (deep interior) -> 1
// (perimeter). A face is culled when its average vertex density is below the
// average local threshold of its 3 vertices. trimThreshold is still the
// interior baseline. Set to 0 to disable (use the stock SurfaceTrimmer).
//
// ENABLED (0 -> 1), and the data now justifies it precisely. A single GLOBAL trim provably
// cannot satisfy this scene. Measured on SchnellTests at depth 11, post-trim density is
//     min 5.47 | p1 7.43 | p5 8.8 | p25 11.7 | p50 12.4 | p95 13 | max 13.4
// so real terrain occupies a tight 11.7-13.4 band with a long thin extrapolation tail running
// from 5.47 up to ~8.8. Empirically, on the same mesh:
//     --poisson-trim 5.5 -> keeps the water, keeps the ragged extrapolation fringe
//     --poisson-trim 9   -> removes the fringe, ALSO removes the water, loses completeness
// Those are the two ends of one knob, and the deliverable needs both behaviours in different
// PLACES. That is exactly what this pass provides: interior = trimThreshold, ramping to
// trimThreshold * POISSON_TRIM_EDGE_MULT_X100/100 at the perimeter. With the interior back at
// 5.5 the edge lands on 5.5 * 1.60 = 8.8 -- the measured p5 that removed the fringe -- applied
// only where the survey actually ran out of data.
//
// SET --poisson-trim BACK TO 5.5. Under this pass it is the INTERIOR threshold, not a global one.
//
// This REPLACES the SurfaceTrimmer call, so --poisson-island-ratio / --removeIslands no longer
// apply; isolated junk is handled by the component filters in Mesh::Clean and the texture-stage
// orphan filter, both active and logged. POISSON_TRIM_HARD_OUTSIDE stays 0 -- see its note; the
// ramped threshold does the useful work and, being a density test, cannot sever the surface.
#ifndef POISSON_ADAPTIVE_TRIM
#define POISSON_ADAPTIVE_TRIM 1   // (was briefly 0 for an A/B; the footprint path lives here)
//                                 NOTE disabling this takes the
                                  // FOOTPRINT machinery with it: the whole block, including
                                  // POISSON_TRIM_HARD_OUTSIDE and the escalating guard, is
                                  // inside #if POISSON_ADAPTIVE_TRIM. What runs instead is the
                                  // stock SurfaceTrimmer at a GLOBAL --poisson-trim.
#endif
// NOTE: releasing the dense point cloud before the Poisson solve is a RUNTIME
// option (`--release-pointcloud`, the releasePointCloud argument below), not a
// compile-time gate -- see ReconstructMeshPoisson.
// Edge threshold as a multiple of the interior (passed) trimThreshold, x100.
// 160 = edge threshold is 1.60x the interior threshold (e.g. 5.5 -> 8.8).
// RAISED 160 -> 180 (edge threshold 5.5 * 1.80 = 9.9), chosen from the measured density
// distribution rather than guessed:
//     min 5.47 | p1 7.31 | p5 8.75 | p25 11.6 | p50 12.4 | p95 13 | max 13.4
// Real terrain is the tight 11.6-13.4 band; the extrapolation tail runs 5.47 to ~8.8. At 160 the
// edge threshold sat on p5 (8.8) and the adaptive trim removed only 0.4% of faces, leaving the
// tongues. A GLOBAL --poisson-trim 9 was separately measured to remove the fringe properly -- it
// just also ate the water, because it applied everywhere. 1.80 puts the PERIMETER threshold at
// 9.9, past that known-good global value, while the deep interior stays at 5.5 and keeps the
// water. That is the whole point of having a ramp.
//
// This is the right place for the fix. Every DOWNSTREAM criterion has now failed to separate
// fabricated surface from real, for a documented reason each time: projected winding deletes
// near-vertical real walls, topology calls tongues 'enclosed', and component size shows no gap
// (1279 components ramping 10391/3990/2444/2268/... with no break). Density near the data
// perimeter is the one signal that does separate them.
//
// Headroom before this eats real ground: p25 is 11.6, so 9.9 still sits below the bulk. If the
// edge starts losing genuine shoreline step back toward 170; if tongues persist, 190-200 is still
// under p25 but getting close, and past that the interior/edge split stops protecting you.
#ifndef POISSON_TRIM_EDGE_MULT_X100
// FLATTENED TO 100 (edge == interior, i.e. no ramp). The elevated perimeter threshold is now
// REDUNDANT AND HARMFUL:
//   * redundant -- it existed to remove perimeter extrapolation via DENSITY, before the footprint
//     cut existed. POISSON_TRIM_HARD_OUTSIDE now does that on OCCUPANCY, which actually
//     discriminates fabricated from real surface where density cannot.
//   * harmful -- density near a survey perimeter is genuinely low for REAL surface too, so the ramp
//     cuts real ground first. Measured twice: RAMP_CELLS 16 doubled removal but lost shoreline while
//     tongues survived, and at interior 5.63 the perimeter threshold of 5.63 x 1.80 = 10.13 punched
//     a hole through a water body sitting near the survey edge -- water fill cannot survive 10.
//
// At 100 the density test is uniform at the percentile-derived interior value, which is exactly the
// behaviour the hand-tuned 5.5-5.75 / 6.5 values were calibrated against (a single GLOBAL trim). So
// density handles interior low-confidence surface, the footprint cut handles the boundary, and
// neither is asked to do the other's job.
//
// Raise it again ONLY if POISSON_TRIM_HARD_OUTSIDE is disabled, since then density is once more the
// only boundary mechanism and the ramp is the least-bad way to bias it outward.
#define POISSON_TRIM_EDGE_MULT_X100 100
#endif
// POISSON_TRIM_INTERIOR_MULT_X100: the same dial for the OTHER end of the ramp. The threshold
// runs trimInterior (deep interior) -> trimEdge (perimeter); EDGE_MULT scales the perimeter
// end and this scales the interior end. Until now the interior was pinned to trimBase with no
// dial at all, so the only way to relax it was to move the percentile, which moves both.
//
// WHY IT IS WANTED. Enclosed low-return regions -- water above all -- are held up by nothing
// but Poisson's interpolation across the gap, so their density sits at the very bottom of the
// distribution. The exterior flood already recognises them as interior (that is what
// POISSON_TRIM_CLOSE_CELLS seals the shore channel for), so they take the INTERIOR threshold,
// and lowering it is the one lever that spares them without touching the boundary.
//
// FIELD CASE: SchnellTests lost the middle of a water body after the operator's dense-cloud
// outlier filter was improved. The filter took 6239 cells from having-any-returns to none and
// cut the sparse tail 19.5% -> 11.8%; those spurious returns were the only thing holding that
// water above the trim. Its ladder: min=3 p0.1=5 p0.2=5.76 (= trimBase) p1=7.31 p50=12.4, so
// the water bridge sits somewhere in 3-5.76 and a 0.60 multiplier puts the interior threshold
// at 3.46 -- under p0.1, above min.
//
// SAFE DIRECTION. This can only ever KEEP more, never cut more, so it cannot re-open
// RichmondHistoric's bottom or worsen any over-cut scene. Nor does it touch Marco's or Redy's
// edge slop, which is a boundary problem the footprint cut owns.
//
// The cost is that genuine low-density interior surface also survives -- noise shells and
// spikes over well-surveyed ground. Disconnected ones still go to the small-component filter;
// connected ones will not. If spikes appear over roofs, raise this back toward 100.
//
// TUNE IT FROM THE LOGGED LADDER, not by guessing: the "vertex density percentiles" line gives
// min and the low tail for that run. Put the interior threshold below the density of whatever
// you want to keep and above the noise floor.
// POISSON_TRIM_INTERIOR_CELLS: how far inside the footprint the interior relaxation reaches
// FULL strength, in grid cells. It must be well clear of POISSON_TRIM_RAMP_CELLS (6) or the
// relaxation bleeds into the perimeter band and loosens the edge -- see the two-ramp note in
// the per-vertex threshold loop for the measurement that forced this to exist.
//
// This is the "how far in before water is protected" dial. At 24 cells on SchnellTests
// (cell 1.213) full relaxation starts 29 units inside the boundary, and the threshold slides
// from trimBase at 7.3 units to trimInterior at 29. A lake in the middle of a scene is far
// past that; a lake sitting ON the survey edge is not, and will still be trimmed -- that case
// is called out in the POISSON_TRIM_EDGE_MULT_X100 notes and this does not solve it.
// Raise it to protect the edge more and water less; lower it for the reverse.
#ifndef POISSON_TRIM_INTERIOR_CELLS
#define POISSON_TRIM_INTERIOR_CELLS 24
#endif
// REVERTED TO 100 (flat) 2026-08-22. At 60 it restored SchnellTests' water but the whole
// mechanism was moving 0.03% of the mesh -- density trim 3408 -> 658 faces at 60 with one
// ramp, 1111 with two, against 3,051,493 total, and final face counts across all three runs
// spanning 0.31%. Meanwhile the operator's dense-cloud filter change had moved 6239 grid
// cells from having returns to having none (sparse tail 19.5% -> 11.8%). The leverage is
// three orders of magnitude apart: this is not the place to compensate for a cloud change,
// and doing it globally makes nine scenes pay for one scene's input.
//
// The machinery is kept and is inert at 100 (trimInterior == trimBase, identical to the
// original flat behaviour). It is the correct lever if enclosed low-return regions ever need
// protecting for their own sake rather than as compensation -- set 60 with
// POISSON_TRIM_INTERIOR_CELLS controlling how far in it reaches.
#ifndef POISSON_TRIM_INTERIOR_MULT_X100
#define POISSON_TRIM_INTERIOR_MULT_X100 100
#endif
// Ramp width (in grid cells) over which the threshold blends interior->edge.
// RAISED 6 -> 16 (7.4 -> 19.7 world units at a 1.232 cell). This is the knob that decides how far
// INWARD from the data perimeter the elevated edge threshold reaches, and it was the real
// bottleneck -- not the edge multiplier.
//
// MEASURED: raising POISSON_TRIM_EDGE_MULT_X100 160 -> 180 (edge 8.8 -> 9.9) moved removal only
// 11,597 -> 12,394 faces, +7%, still 0.4% of the mesh. Yet ~12-15% of VERTICES sit below 9.9
// (it falls between p5=8.75 and p25=11.6). That gap is the proof: the low-density material is
// almost all INSIDE the footprint, where e = 1 - dist/ramp has already decayed the threshold back
// toward the 5.50 interior value. Only the thin band actually outside the footprint ever saw 9.9.
//
// The tongues extend tens of units, while the elevated band was 7.4 units deep starting from a
// perimeter that CLOSE_CELLS dilation had already pushed 9.9 units outward -- so their bases sat
// at ~5.50 and survived every edge-threshold increase. 16 cells covers ~20 units inward, which is
// the scale of the artifact.
//
// TRADE, and it is the same one as always: this elevates the threshold on anything within ~20
// units of the perimeter, INCLUDING near-shore water. Deep interior (a lake centre) is unaffected
// and still sits at 5.50. If shoreline starts disappearing, step back to 10-12; the deep-interior
// protection is what distinguishes this from the global --poisson-trim 9 that ate the water.
#ifndef POISSON_TRIM_RAMP_CELLS
#define POISSON_TRIM_RAMP_CELLS 6
#endif
//
// NEGATIVE RESULT (2026-08-19): tested at 16 (19.7 units) and REVERTED to 6. It doubled removal
// (12,394 -> 23,506 faces, vs only +7% from raising the edge multiplier), so the mechanism was
// real -- the band's DEPTH was the constraint, not its height. But the outcome was a bad trade:
// shoreline was lost in areas with little synthetic geometry, while many tongues survived.
//
// WHY, and it generalises: distance-from-perimeter is ORTHOGONAL to whether geometry is
// fabricated. It raises the threshold uniformly along the whole boundary, and the real survey edge
// is genuinely low-density there (fewer views at the coverage limit), so it is cut first --
// while a tongue carrying ordinary density survives. Density x distance cannot separate them.
//
// That is the fourth downstream criterion to fail on this artifact, after projected winding
// (deletes near-vertical real walls), topology (calls tongues 'enclosed') and component size (no
// gap in the distribution). The reason is structural: at the survey boundary, real surface and
// extrapolated surface share every property these tests measure. The only true discriminator is
// whether there are POINTS underneath -- confirmed absent under the tongues on this scene -- which
// means the occupancy/footprint test, not another threshold.
// Footprint grid cell size as a multiple of median NN spacing, x100. 300 = 3x.
// RAISED 300 -> 2000 (20x median spacing), the value this footprint machinery was actually
// FITTED with -- the live file had drifted to 300, which is a different lineage. It matters a
// lot: the occupancy test needs >= POISSON_TRIM_MIN_PTS_PER_CELL points per cell, so 3x-spacing
// cells (0.18 units on SchnellTests) almost never reach 20 points and the footprint fragments
// into noise -- which makes nearly everything read as "near the perimeter" and applies the EDGE
// threshold everywhere, i.e. exactly the global-trim behaviour this pass exists to avoid.
// 20x gives 1.23-unit cells there, ~50 points/cell average against a 20 minimum.
// VERIFY on new data via the "X of Y cells occupied (fraction Z)" field in the trim log: near
// 1.0 means the min-count is too low and the footprint is swallowing the fringe.
#ifndef POISSON_TRIM_CELL_FACTOR_X100
#define POISSON_TRIM_CELL_FACTOR_X100 2000
#endif
// Minimum points in a grid cell for it to count as inside the surveyed footprint. This is the
// "is there ENOUGH data here" test -- the one criterion that separates fabricated surface from
// real, since both can carry ordinary density and both can face any direction. Fitted with
// POISSON_TRIM_CELL_FACTOR_X100 2000; the two must be tuned together (halving the cell area
// quarters the expected count).
// LOWERED 20 -> 8 to close the residual interior holes, and the fringe cannot come back with it.
// The two artifacts have DIFFERENT point counts, which is what makes this separable:
//   * the extrapolated fringe has ZERO points (verified by inspecting the dense cloud), so ANY
//     threshold >= 1 excludes it from the footprint -- lowering this cannot resurrect it;
//   * the holes are in rough/vegetated terrain that has SOME coverage but under 20 points per
//     cell, so it fell outside the footprint, was treated as perimeter, and got the 8.80 edge
//     threshold instead of the 5.50 interior one.
// Scale check: at median spacing 0.0616 a fully-sampled 1.232-unit cell holds (1.232/0.0616)^2
// ~= 400 points, so 20 was demanding 5% of nominal density and 8 asks for 2% -- still decisively
// "there is data here", just no longer excluding sparsely-imaged real ground.
//
// MEASURED at 20: trim removed 28691 of 2839730 faces (1.0%), occupancy 27898 of 89075 cells
// (0.313). Occupancy well below 1.0 is the headroom that makes this safe -- per the tuning note
// above, only a fraction NEAR 1.0 means the min-count is too low and the footprint has started
// swallowing the fringe. Re-check that field after this change.
//
// NOTE this lever only reaches holes with NON-ZERO coverage. A hole over a small POND has no
// points at all, so no min-count helps; those depend on the exterior flood-fill failing to reach
// them, i.e. on POISSON_TRIM_CLOSE_CELLS dilation sealing the channel. If holes persist over
// water specifically, raise CLOSE_CELLS instead.
#ifndef POISSON_TRIM_MIN_PTS_PER_CELL
// TESTED AT 12 AND REVERTED TO 8 (2026-08-19): not better. That is informative -- raising it drops
// the THINLY-SUPPORTED cells first, so if the fringe did not retreat, the fringe cells DO have
// adequate point support at this grid scale. Which is unsurprising at 20x median spacing (~1.2 unit
// cells): a cell containing a tongue almost certainly also contains the real terrain points the
// tongue grew out of, so the footprint cannot resolve one from the other. To separate them the GRID
// would have to be finer (POISSON_TRIM_CELL_FACTOR_X100), with MIN_PTS lowered in proportion --
// halving the cell area quarters the expected count -- at the cost of a noisier footprint.
//
// Original rationale for 8, which still stands: this is the SELECTIVE way to shrink the footprint: a
// cell joins only with >= N points, so raising it drops the THINLY-SUPPORTED cells first, which is
// exactly what the fringe is -- unlike POISSON_TRIM_MARGIN_CELLS, which pulls the whole boundary in
// uniformly and takes well-covered shoreline with it.
//
// Chosen as a middle step, not a return to the original: 20 was measured to punch holes in
// thinly-covered real ground (rough/vegetated terrain under 20 points per cell), and 8 fixed them.
// Headroom for this comes from the occupancy fraction, which was running 0.41-0.45 -- well clear of
// the 1.0 that would mean the footprint has started swallowing the fringe.
//
// WATCH, in the trim log line: "X of Y cells occupied (fraction Z)" should FALL from ~0.42 (the
// footprint really shrinking) and "outside-footprint cut M" should RISE from ~7-12k (the extra
// fringe being cut). If Z falls but M barely moves, the fringe is not in the cells this removed and
// neither this nor MARGIN_CELLS is the right instrument. The failure mode is holes reappearing in
// thinly-covered INTERIOR areas -- if that happens, back off toward 10. If 12 is clean and the edge
// still needs work, 16 is the next notch.
// LOWERED 8 -> 4 (2026-08-20). The notes above are written for RAISING this to shrink the
// footprint; this is the same lever run in reverse, to GROW it.
//
// DIAGNOSED on RichmondHistoric: a band along the bottom of the scene was missing from the
// output at both depth 10 and depth 11, and from the raw solve too. --poisson-trim 0
// restored it, and POISSON_TRIM_HARD_OUTSIDE 0 restored it with the density trim still on,
// which isolates the FOOTPRINT HARD CUT as the cause. The band is real but very thinly
// covered -- the cell is 20x median spacing, so a cell at median density holds ~400 points
// and >=8 is already only ~2% of nominal, yet those cells still fail it.
//
// MARGIN_CELLS cannot fix this: the margin grows the footprint outward from its edge, but
// these cells are not near the edge, they are not IN the footprint at all. Raising the
// margin from 0.44% to 0.73% of extent changed nothing visible, which is what confirmed it.
//
// WATCH the trim line: occupancy should RISE from 0.413 and "outside-footprint cut" should
// FALL from 24153. The guardrail is occupancy approaching 1.0, which means the footprint has
// begun swallowing the extrapolated fringe this test exists to catch -- the notes above put
// the healthy band at 0.41-0.45, so treat anything past ~0.6 as suspect and check the
// under-cut scenes (Marco, Redy) before keeping it.
#define POISSON_TRIM_MIN_PTS_PER_CELL 4
#endif
// Hard cut: remove a face when ALL THREE vertices sit in cells OUTSIDE the footprint,
// regardless of density.
//
// STAYS 0. It is CONNECTIVITY-BLIND and severs large scenes: measured on a 1228-unit corridor it
// removed only 1.5% of faces but produced 4,844 small components and left the bottom half of the
// model hanging off the top by a single sliver (setting it to 0 restored it, confirming the
// cause). The exterior flood-fill protects ENCLOSED gaps, but a low-coverage CHANNEL reaching in
// from the side of a corridor floods as exterior and is cut, isthmus included. SchnellTests is
// the same class (883 units, 1:14.4 aspect), so this must not be turned on here.
//
// Re-enable only with the escalating-dilation + connected-component guard described at
// POISSON_TRIM_CLOSE_CELLS, so a scene dilates as far as it needs instead of paying one global
// value everywhere. The RAMPED THRESHOLD below (interior vs edge) does the useful work without
// this, and cannot sever anything because it is still a density test.
// ENABLED (0 -> 1), now that the escalating-dilation + connected-component guard the note above
// asks for actually exists. The cut itself is unchanged and still connectivity-blind; what is new
// is that its damage is MEASURED before it is committed. Each attempt builds the footprint at a
// dilation, counts the components of the surface that would survive, and accepts the cut only if
// that stays inside a budget relative to the uncut baseline; otherwise the dilation doubles and it
// tries again. If no dilation passes, the cut is skipped for that scene and logged. Worst case is
// the previous behaviour (density ramp only), never a severed model.
//
// WHY THIS INSTRUMENT, having exhausted four others on this artifact. Downstream, fabricated and
// real surface are indistinguishable: projected winding deletes near-vertical real walls, topology
// calls tongues "enclosed", component size shows no gap in the distribution, and density x
// distance-from-perimeter cuts the genuinely low-density real survey edge first. The one property
// that DOES differ is whether there is data underneath -- and this is the OCCUPANCY form of that
// test, which unlike POISSON_DISTANCE_CULL is not defeated by local normalisation.
//
// It also produces the right SHAPE of boundary: the outline follows the data footprint at grid
// resolution instead of Poisson's ragged isosurface, which the original note calls "a
// competitor-style smooth outline rather than a feathery fringe". POISSON_TRIM_CLOSE_CELLS then
// becomes the "how much invented coverage do we accept" knob -- raise it for MORE synthetic surface
// with a smoother outline, which appears to be what the competitor does on this dataset.
#ifndef POISSON_TRIM_HARD_OUTSIDE
#define POISSON_TRIM_HARD_OUTSIDE 1
#endif
// Guard on the hard cut. ATTEMPTS doubles the dilation each time; the component budget is
// baseline * RATIO/100 + SLACK. The corridor failure that originally disabled this cut showed 4,844
// components against ~880 baseline, so 1.5x + 32 rejects it decisively while tolerating the handful
// of specks a healthy cut sheds (measured 895 vs 880 on a scene where it worked).
#ifndef POISSON_TRIM_GUARD_ATTEMPTS
#define POISSON_TRIM_GUARD_ATTEMPTS 5
#endif
#ifndef POISSON_TRIM_GUARD_COMP_RATIO_X100
#define POISSON_TRIM_GUARD_COMP_RATIO_X100 150
#endif
#ifndef POISSON_TRIM_GUARD_COMP_SLACK
#define POISSON_TRIM_GUARD_COMP_SLACK 32
#endif
// Morphological dilation radius (cells) to bridge thin shoreline gaps before
// the interior-hole flood fill. 0 disables.
// RAISED 2 -> 8, the fitted value. With POISSON_TRIM_HARD_OUTSIDE off its role here is to keep
// the WATER on the interior side of the ramp: water has no points, so its cells are unoccupied,
// and it only receives the low interior threshold if the exterior flood-fill cannot REACH it.
// Dilating the occupied region closes narrow shore channels, so a lake connected to the survey
// edge by a gap narrower than ~2x the dilation becomes enclosed and is kept. 8 cells is ~9.9
// units at a 1.23-unit cell.
//
// TENSION: dilation grows the footprint, so extrapolation within ~9.9 units of the real data
// edge is also treated as interior and survives. That is the direct trade against fringe
// removal -- LOWER this if the fringe persists, RAISE it if water or shoreline is lost.
#ifndef POISSON_TRIM_CLOSE_CELLS
// LOWERED 8 -> 4 (9.86 -> 4.93 units) to PULL THE BOUNDARY IN. With the hard cut active this is the
// knob that positions the outline: the footprint reaches this far past the real data, so the cut
// lands there. 8 gave the smoothness the user wants but too much synthetic margin.
//
// The guard makes this safe to try: it counted components 370 -> 369 at dilation 8 against a 587
// budget, i.e. no severing at all with room to spare, and if 4 does sever it will DOUBLE back to 8
// automatically. What the guard does NOT check is water loss -- a lake joined to the exterior by a
// channel wider than the dilation floods as exterior and is cut outright, with the component count
// still looking healthy. So check the lake visually at each step, not just the log.
#define POISSON_TRIM_CLOSE_CELLS 8
#endif
// POISSON_TRIM_MARGIN_CELLS: how many cells of footprint to keep BEYOND the data, i.e. how much
// synthetic margin the hard cut leaves. Now independent of the seal radius above -- see the
// morphological-closing note in buildExt. This is the "pull the boundary in/out" dial, and the
// guard escalates it (not the seal radius) when a cut would sever the surface.
// 2 cells ~= 2.5 units at a 1.232 cell.
#ifndef POISSON_TRIM_MARGIN_CELLS
// LOWERED 5 -> 3 to cull more edge. This is the UNIFORM lever: it pulls the whole boundary in by
// ~40% regardless of how well covered any part of it is, which is the opposite selectivity to
// POISSON_TRIM_MIN_PTS_PER_CELL (tested at 12, no better -- see its note). Since the footprint
// cannot resolve a tongue from the data it grew out of at this grid scale, moving the contour
// bodily is what is left.
//
// Physical effect varies per scene because this is in GRID CELLS and the cell tracks point density:
// at 3 cells it is ~3.7 units on SchnellTests (cell 1.232), ~3.4 on the depth-10 D scene (1.12),
// ~1.25 on E (0.4177) and ~1.55 on F (0.516). That 3x spread is the open cells-vs-world-units
// question; if the tight-cell scenes now cut too much while the coarse ones still leave fringe,
// that is the evidence for converting this knob to world units.
//
// The guard escalates this upward if a tighter margin severs the surface, so the floor is enforced
// automatically -- watch for "attempt 2" in the hard-cut line.
#define POISSON_TRIM_MARGIN_CELLS 3
#endif
// POISSON_TRIM_MARGIN_FLOOR_FRAC: lower bound on the hard-cut margin, as a fraction of
// the XY footprint extent. The margin above is in GRID CELLS and the grid cell tracks
// POINT DENSITY, which has nothing to do with how far the Poisson balloon reaches -- so
// on a dense scene the physical margin collapses and the cut eats real perimeter.
//
// MEASURED over nine scenes, labelled by eye. Sorted by margin/extent (extent = the
// LONGER grid axis), the three the operator called over-cut are the three tightest in
// the corpus and nothing else is below 0.59%:
//   MechanicFalls 0.287% OVER | RichmondHistoric 0.438% OVER | RichmondWater 0.459% OVER
//   SchnellTests 0.592% ok | Redy 0.673% under | Randy 0.750% ok | GEOTAG 0.833% ok
//   Marco 1.250% under | Niwot 1.299% ok
// Absolute margin does NOT separate them (Redy under-cuts at 1.403 units while
// RichmondHistoric over-cuts at 1.252), and neither does grid cell alone. The ratio does.
//
// 0.006 sits above the over-cut cluster and below the first healthy scene. A BIGGER
// margin means a BIGGER footprint, so the cut removes LESS -- the direction these three
// need. Because the margin is an INTEGER cell count the floor overshoots slightly:
//   MechanicFalls    3 -> 7 cells  1.944 -> 4.535 units  (0.287% -> 0.669%)
//   RichmondHistoric 3 -> 5 cells  1.252 -> 2.087 units  (0.438% -> 0.731%)
//   RichmondWater    3 -> 4 cells  1.510 -> 2.013 units  (0.459% -> 0.613%)
//   SchnellTests     3 -> 4 cells  3.716 -> 4.956 units  (0.592% -> 0.789%)  <-- side effect
//   Randy / GEOTAG / Niwot / Redy / Marco: unchanged (floor lands at or below 3 cells)
// SchnellTests is the one healthy scene that moves, and only because its 3.042 rounds up.
// It lands between Randy (0.750%) and GEOTAG (0.833%), both healthy, so it should be
// safe -- but it is the scene to check first for a new under-cut.
//
// This is a FLOOR only. It cannot make the margin smaller, so it cannot make the
// under-cut scenes (Marco, Redy) worse -- those are a different mechanism, still open;
// see [MESH-OVERHANG].
#ifndef POISSON_TRIM_MARGIN_FLOOR_FRAC
#define POISSON_TRIM_MARGIN_FLOOR_FRAC 0.006f
#endif
// POISSON_TRIM_MARGIN_CAP_UNITS: the matching CEILING, in world units. The floor above stops
// a DENSE scene (small cell) getting a margin too tight to clear the balloon; this stops a
// SPARSE scene (large cell) getting one absurdly wide from the same cell count. Together they
// are the "convert this knob to world units" the MARGIN_CELLS note asks for.
//
// Only Marco is affected at 7.0 (3 -> 2 cells, 10.46 -> 6.97 units); the other eight scenes
// have cells small enough that the cap sits above their count. Set 0 to disable the cap.
#ifndef POISSON_TRIM_MARGIN_CAP_UNITS
#define POISSON_TRIM_MARGIN_CAP_UNITS 7.0f
#endif
// POISSON_COMP_GAP_RESCUE / _OVERLAP: exempt a below-threshold component from deletion
// when the kept mesh does not already cover the ground it sits on. See the GAP RESCUE
// block in the small-component filter for the reasoning and the observation behind it.
//
// _OVERLAP is the fraction of a component's XY cells that must ALREADY be covered by the
// kept mesh for it to count as redundant. 0.5 = "more than half of this thing is
// duplicating surface we already have" -> delete. Lower it to rescue less (stricter,
// only components almost entirely in open ground); raise it to rescue more.
//
// UNTESTED as of writing. The failure mode to watch for is the opposite complaint --
// floaters and shells surviving where they used to be cleaned -- which would show up as
// the under-cut scenes (Marco, Redy) getting worse, or as spikes above roofs. The
// "gap rescue kept N of M" line reports exactly what it saved on every run.
// POISSON_REACH_CUT: remove surface sitting further than a WORLD-UNIT distance from the
// nearest surveyed cell, independent of the footprint. See the REACH CUT block in the
// adaptive trim for the corpus calibration.
//
// This exists because the footprint's margin is in GRID CELLS and the grid cell tracks
// point density, so a sparsely-sampled scene gets a huge physical margin -- Marco's cell
// is 3.486 units against 0.17-1.6 elsewhere, giving its 3-cell margin a 10.46-unit reach
// where most scenes get 0.5-2.1. Extrapolation reach is a physical distance and has nothing
// to do with sample spacing, so the ceiling on it must be physical too.
//
// max(MIN_UNITS, EXTENT_FRAC * extent): the floor leaves small scenes alone (their overhang
// maxima are 7-42 units, all at or under the floor), the fraction keeps the tolerance
// proportionate on large ones. At 30 / 0.05 this lands almost entirely on Marco and removes
// nothing at all from Redy or Randy.
//
// Deliberately NOT gated on the footprint guard (bHardCutOK): a face 40+ units from any data
// is not part of a surface worth preserving even when the footprint cut has been suppressed.
// DEFAULTED OFF -- it fired ZERO faces on the scene it was written for.
// MEASURED on Marco: armed at 41.46 units, 995 of 19291 mesh cells beyond it, 0 faces cut.
// The footprint cut runs first in the same loop and had already taken all of them: anything
// 41 units out is far outside a footprint that reaches only 10.46 units past the data, so
// this criterion is redundant by construction at any threshold above the margin.
//
// The [MESH-OVERHANG] percentiles that motivated it are misleading for this purpose because
// they conflate the outer balloon with ENCLOSED interior gaps -- Marco has 23% of mesh cells
// outside surveyed data on a scene with 7.82% unobserved faces (courtyards, water, shadow),
// and those are kept by design. Its p50 of 13.94 units is mostly interior, not reach.
// Marco's real slop turned out to be INSIDE the footprint's own margin; see
// POISSON_TRIM_MARGIN_CAP_UNITS.
//
// Kept because the machinery is sound and would become useful once [MESH-OVERHANG] is split
// by the exterior flood (`ext`, already computed) so it measures outer reach alone. Until
// then it is dead weight -- do not enable it without that split.
#ifndef POISSON_REACH_CUT
#define POISSON_REACH_CUT 0
#endif
#ifndef POISSON_REACH_CUT_MIN_UNITS
#define POISSON_REACH_CUT_MIN_UNITS 30.0
#endif
#ifndef POISSON_REACH_CUT_EXTENT_FRAC
#define POISSON_REACH_CUT_EXTENT_FRAC 0.05
#endif
#ifndef POISSON_COMP_GAP_RESCUE
#define POISSON_COMP_GAP_RESCUE 1
#endif
#ifndef POISSON_COMP_GAP_OVERLAP
#define POISSON_COMP_GAP_OVERLAP 0.5
#endif
// POISSON_TRIM_PERCENTILE_X100: take the INTERIOR trim threshold as this percentile of the scene's
// own density distribution instead of from --poisson-trim. Units are hundredths of a percent, so
// 50 = the 0.50th percentile. Depth- and dataset-invariant, which a raw density value is not.
//
// CALIBRATED AT p0.20 against three scenes' own low-tail ladders (2026-08-19). The claim of this
// knob is that ONE percentile reproduces thresholds that previously had to be hand-tuned per scene:
//
// CAUTION when re-tuning: only A and B were tuned by EYE ("looks best at..."). Every other raw
// value in the log is just whatever the pipeline happened to pass, and those defaults have moved
// over time (one scene ran 6.50 while newer builds pass 5.50). A default is not a target -- treating
// one as evidence nearly caused a recalibration of this knob away from two genuinely tuned scenes.
//
//   scene              depth  hand-tuned    p0.1   p0.2   p0.35  p0.5   p50
//   A (fills water)     11    5.5 - 5.75    4.95   5.63*  6.19   6.61   12.4
//   B                   12    ~6.5, or a    5.33   6.28*  7.1    7.51    9.8
//                             little lower
//   C                   --    (see note)    4.0    4.44   4.98   5.13    9.89
//   D                   10    --            --     5.60   --     --      11.3
//   E                   10    --            4.67   5.26   5.87   6.09    11.8
//
// UNRESOLVED, and previously mis-attributed here: scene E's ugly water was observed to DROP on the
// p0.20 build, and this comment claimed that as evidence the percentile orders scenes correctly.
// That claim does not survive its own arithmetic -- E resolves to 5.26 against the 5.50 it had been
// running, and a LOWER threshold can only remove LESS. So the density test cannot be what dropped
// that water.
//
// The likely cause is POISSON_TRIM_HARD_OUTSIDE, which removes surface with no data underneath
// REGARDLESS of density. Water has no points, so whether it survives is decided by footprint
// TOPOLOGY, not by any threshold: enclosed (the exterior flood cannot reach it) -> kept, as on scene
// A; reachable through a shore channel -> outside the footprint -> cut, which would explain E.
//
// TO SETTLE IT read the adaptive-trim line: "removed N ... | outside-footprint cut M". If M accounts
// for the water, occupancy did it and the percentile is irrelevant to that outcome. Do not restate
// the opposite-outcomes claim without that number.
//
// Four of the five land in 5.26-6.28 at p0.20 while their p50 ranges 9.8-12.4, so this point is a
// stable feature of the distribution even when the bulk shifts ~25%. Note also that the two scenes
// with a LOW bulk (B and C, p50 9.8/9.89) are the ones needing the highest/lowest absolute values --
// further evidence the absolute scale is not comparable across scenes and the percentile is.
//
// p0.20 lands mid-window on A and just under target on B -- both hand-tuned values from a single
// setting. Note how far the ABSOLUTE value moves for the same percentile (5.63 / 6.28 / 4.44) and
// how differently the distributions sit (p50 12.4 / 9.8 / 9.89). That spread is precisely why a raw
// --poisson-trim could not be carried across depths or datasets, and why 35/45/50 all looked alike:
// they over-trim relative to these targets, in a range where the footprint cut already owns the
// boundary.
//
// SCENE C DOES NOT WANT A HIGHER PERCENTILE. Its complaint was a floating blob in a corner, and at
// p0.20 it gets 4.44 -- less trimming than the 5.0 it had been running. That is the right call
// anyway: a blob is a DISCONNECTED COMPONENT and belongs to MESH_KEEP_COMPONENT_PCT_X1000, not to a
// density threshold. Raising the percentile far enough to dissolve it would badly over-trim A and B,
// since C's whole distribution sits ~2.5 lower. Read "post-tighten component filter: removed N faces
// in M blobs (threshold T faces)" and raise that filter instead.
//
// (Earlier note, superseded: 50 was calibrated against the single field-tuned value then available --
// on
// distribution was min 3 / p1 7.31 / p5 8.75, and 5.75 was reported as looking good -- which sits
// between min and p1, i.e. around the half-percent mark. The edge threshold is still this x
// POISSON_TRIM_EDGE_MULT_X100.
//
// RAISE it to trim more everywhere (p1 = 7.31 here, p5 = 8.75); note the file's own warning that a
// threshold AT p1 "cuts the lowest 1% everywhere and speckles the middle with holes", so this wants
// to sit BELOW p1. 0 disables and restores the raw --poisson-trim value.
#ifndef POISSON_TRIM_PERCENTILE_X100
#define POISSON_TRIM_PERCENTILE_X100 20   // p0.20 -- calibrated, see table above
#endif

// Easier to configure this here.
#pragma comment(linker, "/STACK:0x400000,0x400000")

#include "Common.h"
#include "Scene.h"
// Delaunay: mesh reconstruction
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polyhedron_3.h>
#include <tbb/parallel_sort.h>
#include "robin_map.h" // assumes robin_map.h is in include path
#include <vcg/complex/algorithms/clean.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/hilbert_policy_tags.h>   // <-- defines Hilbert_sort_median_policy
#include <CGAL/Hilbert_sort_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/spatial_sort.h>

// Tier 2 Poisson is integrated in-process via the isolated PoissonReconLib
// static library (Kazhdan PoissonRecon + SurfaceTrimmer), so NO CGAL Poisson
// headers are pulled in — CGAL's deprecated Surface_mesher fails to instantiate
// under MSVC.
#include <fstream>
#include <sstream>
#include <iterator>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <cstdlib>

#define NANOFANN_USE_OMP 1   // optional, but good hint
#include "nanoflann.hpp"
// In-process Kazhdan PoissonRecon + SurfaceTrimmer (replaces the external-exe
// shell-out). Plain (template-free) interface; the templated machinery lives in
// the isolated PoissonReconLib static library.
#include "../../PoissonRecon/Src/PoissonReconLib.h"

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

template <typename T>
struct NoInitAllocator
{
  using value_type = T;
  NoInitAllocator() = default;

  template <class U>
  constexpr NoInitAllocator(const NoInitAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }

  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p);
  }
};

template<class T, class _Alloc = std::allocator<T>>
struct alignas(64) PaddedVector
{
	std::vector<T, _Alloc> mData;
	char mPadding[64-sizeof(mData)];

	template<typename... Args>
	void emplace_back(Args&&... args)
	{
		mData.emplace_back(std::forward<Args>(args)...);
	}

	void resize(size_t n) { mData.resize(n); }
	void reserve(size_t n) { mData.reserve(n); }
	void push_back(const T& val) { mData.push_back(val); }
	void push_back(T&& val) { mData.push_back(std::move(val)); }

	auto begin() { return mData.begin(); }
	auto end()   { return mData.end(); }
	auto begin() const { return mData.begin(); }
	auto end()   const { return mData.end(); }

	T& operator[](size_t i) { return mData[i]; }
	const T& operator[](size_t i) const { return mData[i]; }

	size_t size() const { return mData.size(); }
	T* data() { return mData.data(); }
	const T* data() const { return mData.data(); }
};

template<typename T, size_t InlineCap>
struct SmallQueue {
	size_t _size;
	T inlineData[InlineCap];
	std::vector<T> overflow; // only used if needed

	SmallQueue() : _size(0) {}

	__forceinline void clear() {
		_size = 0;
		overflow.clear();
	}

	__forceinline void push(const T& v) {
		if (_size < InlineCap) {
			inlineData[_size++] = v;
		}
		else {
			overflow.push_back(v);
		}
	}

	inline size_t size() const {
		return _size + overflow.size();
	}

	__forceinline T& operator[](size_t i) {
    return const_cast<T&>(static_cast<const SmallQueue&>(*this)[i]);
	}

	__forceinline const T& operator[](size_t i) const {
		if (i < InlineCap)
			return inlineData[i];
		return overflow[i - InlineCap];
	}
};

#include <intrin.h>   // For __rdtscp, __rdtsc
#include <cstdint>    // For uint64_t
#include <windows.h>  // For SetThreadAffinityMask, Sleep

// Optional: Pin to a single core for consistency
DWORD_PTR SetAffinityToCPU0()
{
  HANDLE thread = GetCurrentThread();
  return SetThreadAffinityMask(thread, 1); // Use only CPU 0
}

void RestoreAffinity(DWORD_PTR originalMask)
{
  HANDLE thread = GetCurrentThread();
  SetThreadAffinityMask(thread, originalMask);
}

// Safe, serialized RDTSC start
inline uint64_t rdtscStart()
{
  int dummy;
  _mm_lfence(); // Serialize
  return __rdtscp(reinterpret_cast<unsigned int*>(&dummy));
}

// Safe, serialized RDTSC end
inline uint64_t rdtscEnd()
{
  unsigned int dummy;
  uint64_t tsc = __rdtscp(&dummy);
  _mm_lfence(); // Serialize
  return tsc;
}

double estimateCpuHz()
{
  //SetAffinityToCPU0(); // Optional, but improves accuracy

  uint64_t start = rdtscStart();
  Sleep(100); // 100 ms
  uint64_t end = rdtscEnd();

  uint64_t delta = end - start;
  return static_cast<double>(delta) * 10.0; // since 100 ms = 0.1 s
}

// Convert delta to seconds
inline double rdtscToSeconds(uint64_t delta, double cpuHz)
{
  return static_cast<double>(delta) / cpuHz;
}

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

#undef VALIDATE

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DELAUNAY_USE_OPENMP
#endif

// uncomment to enable reconstruction algorithm of weakly supported surfaces
#define DELAUNAY_WEAKSURF

// uncomment to use IBFS algorithm for max-flow
// (faster, but not clear license policy)
#define DELAUNAY_MAXFLOW_IBFS


// S T R U C T S ///////////////////////////////////////////////////

#ifdef DELAUNAY_MAXFLOW_IBFS
#include "../Math/IBFS/IBFS.h"
template <typename NType, typename VType>
class MaxFlow
{
public:
	// Type-Definitions
	typedef NType node_type;
	typedef VType value_type;
	typedef IBFS::IBFSGraph graph_type;

public:
	MaxFlow(size_t numNodes)
	{
		graph.initSize((int)numNodes, (int)numNodes*2);
	}

	inline void AddNode(node_type n, value_type source, value_type sink)
	{
		ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
		graph.addNode((int)n, source, sink);
	}

	inline void AddEdge(node_type n1, node_type n2, value_type capacity, value_type reverseCapacity)
	{
		ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
		graph.addEdge((int)n1, (int)n2, capacity, reverseCapacity);
	}

	value_type ComputeMaxFlow()
	{
		graph.initGraph();
		return graph.computeMaxFlow();
	}

	inline bool IsNodeOnSrcSide(node_type n) const
	{
		return graph.isNodeOnSrcSide((int)n);
	}

	void FinalizeGraph()
	{
    graph.finalizeGraph();
	}

	// Variant for callers that wrote arcs[] directly and set arcCount.
	// Skips the atomic counter copy; only runs arc sort.
	void FinalizeGraphPrebuilt()
	{
		graph.finalizeGraphPrebuilt();
	}

	graph_type graph;
};
#else
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <vcg/complex/algorithms/clean.h>
template <typename NType, typename VType>
class MaxFlow
{
public:
	// Type-Definitions
	typedef NType node_type;
	typedef VType value_type;
	typedef boost::vecS out_edge_list_t;
	typedef boost::vecS vertex_list_t;
	typedef boost::adjacency_list_traits<out_edge_list_t, vertex_list_t, boost::directedS> graph_traits;
	typedef typename graph_traits::edge_descriptor edge_descriptor;
	typedef typename graph_traits::vertex_descriptor vertex_descriptor;
	typedef typename graph_traits::vertices_size_type vertex_size_type;
	struct Edge {
		value_type capacity;
		value_type residual;
		edge_descriptor reverse;
	};
	typedef boost::adjacency_list<out_edge_list_t, vertex_list_t, boost::directedS, size_t, Edge> graph_type;
	typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;
	typedef typename boost::graph_traits<graph_type>::out_edge_iterator out_edge_iterator;

public:
	MaxFlow(size_t numNodes) : graph(numNodes+2), S(node_type(numNodes)), T(node_type(numNodes+1)) {}

	void AddNode(node_type n, value_type source, value_type sink) {
		ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
		if (source > 0) {
			edge_descriptor e(boost::add_edge(S, n, graph).first);
			edge_descriptor er(boost::add_edge(n, S, graph).first);
			graph[e].capacity = source;
			graph[e].reverse = er;
			graph[er].reverse = e;
		}
		if (sink > 0) {
			edge_descriptor e(boost::add_edge(n, T, graph).first);
			edge_descriptor er(boost::add_edge(T, n, graph).first);
			graph[e].capacity = sink;
			graph[e].reverse = er;
			graph[er].reverse = e;
		}
	}

	void AddEdge(node_type n1, node_type n2, value_type capacity, value_type reverseCapacity) {
		ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
		edge_descriptor e(boost::add_edge(n1, n2, graph).first);
		edge_descriptor er(boost::add_edge(n2, n1, graph).first);
		graph[e].capacity = capacity;
		graph[er].capacity = reverseCapacity;
		graph[e].reverse = er;
		graph[er].reverse = e;
	}

	value_type ComputeMaxFlow() {
		vertex_size_type n_verts(boost::num_vertices(graph));
		color.resize(n_verts);
		std::vector<edge_descriptor> pred(n_verts);
		std::vector<vertex_size_type> dist(n_verts);
		return boost::boykov_kolmogorov_max_flow(graph,
			boost::get(&Edge::capacity, graph),
			boost::get(&Edge::residual, graph),
			boost::get(&Edge::reverse, graph),
			&pred[0],
			&color[0],
			&dist[0],
			boost::get(boost::vertex_index, graph),
			S, T
		);
	}

	inline bool IsNodeOnSrcSide(node_type n) const {
		return (color[n] != boost::white_color);
	}

protected:
	graph_type graph;
	std::vector<boost::default_color_type> color;
	const node_type S;
	const node_type T;
};
#endif
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////

// construct the mesh out of the dense point cloud using Delaunay tetrahedralization & graph-cut method
// see "Exploiting Visibility Information in Surface Reconstruction to Preserve Weakly Supported Surfaces", Jancosek and Pajdla, 2015
namespace DELAUNAY {
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::Point_3 point_t;
typedef kernel_t::Vector_3 vector_t;
typedef kernel_t::Direction_3 direction_t;
typedef kernel_t::Segment_3 segment_t;
typedef kernel_t::Plane_3 plane_t;
typedef kernel_t::Triangle_3 triangle_t;
typedef kernel_t::Ray_3 ray_t;

typedef uint32_t vert_size_t;
typedef uint32_t cell_size_t;

typedef float edge_cap_t;

typedef TFrustum<REAL, 4> Frustum;

__forceinline double fast_sqdist2(double x0, double y0, double z0, double x1, double y1, double z1)
{
	const double dx = (x1-x0);
	const double dy = (y1-y0);
	const double dz = (z1-z0);

	const double dx2 = dx*dx;
	const double dy2 = dy*dy;
	const double dz2 = dz*dz;

	return dx2+dy2+dz2;
}

struct vert_info_t {
	vert_info_t() :
		 idx( g_idx++ )
	{}
	uint32_t idx;
	static uint32_t g_idx;
};

uint32_t vert_info_t::g_idx = 0;

typedef edge_cap_t Type;
struct view_t {
	PointCloud::View idxView; // view index
	//float weight;
	inline view_t() {}
	inline view_t(PointCloud::View _idxView) : idxView(_idxView) {}
	inline bool operator <(const view_t& v) const { return idxView < v.idxView; }
	inline operator PointCloud::View() const { return idxView; }
};

// Use 2 cache lines per entry to start.
#if !DIRECT_VIEW_EXPANSION
typedef boost::container::small_vector<uint32_t, 26> view_vec_t;
view_vec_t* allViews; // faces' weight from the cell outwards

__forceinline void InsertViews(size_t vertexId, const PointCloudStreaming& pc, uint32_t idxPoint)
{
	// Note we silently enforce indices no larger than a uint32_t
	allViews[vertexId].push_back(idxPoint);
}
#else
// New path: store vertex index per point during insertion.
// Expansion into per-vertex view lists happens as a separate pass.
uint32_t* pointToVertex; // sized numVertices, maps sorted-point-index -> DT vertex idx
// Per-vertex expanded view data (built after insertion)
struct ExpandedViewCount {
	uint32_t id;
	uint16_t count;
};
ExpandedViewCount* vcData;   // flat array of all view counts
uint32_t* vcOffsets;          // vcData offset per vertex idx
uint16_t* vcSizes;            // number of ViewCounts per vertex idx
#endif

#ifdef VALIDATE
struct vert_info_t2 {
	typedef edge_cap_t Type;
	struct view_t2 {
		PointCloud::View idxView; // view index
		Type weight; // point's weight
		inline view_t2() {}
		inline view_t2(PointCloud::View _idxView, Type _weight) : idxView(_idxView), weight(_weight) {}
		inline bool operator <(const view_t& v) const { return idxView < v.idxView; }
		inline operator PointCloud::View() const { return idxView; }
	};
	typedef SEACAVE::cList<view_t2,const view_t&,0,4,uint32_t> view_vec_t2;
	view_vec_t2 views; // faces' weight from the cell outwards
	inline vert_info_t2() {}
	void InsertViews(const PointCloudStreaming& pc, PointCloud::Index idxPoint) {
		const uint32_t* _views = pc.ViewsStream(idxPoint);
		const uint32_t cnt = pc.ViewsStreamSize(idxPoint);
		ASSERT(!_views.IsEmpty());
		const float* pweights(pc.WeightsStream(idxPoint));
		for (uint32_t i = 0; i < cnt; ++i) {
			const PointCloud::View viewID(_views[i]);
			const PointCloud::Weight weight(pweights ? pweights[i] : PointCloud::Weight(1));
			// insert viewID in increasing order
			views.Insert(view_t2(viewID, weight));
#if 0
			const uint32_t idx(views.FindFirstEqlGreater(viewID));
			if (idx < views.GetSize() && views[idx] == viewID) {
				// the new view is already in the array
				ASSERT(views.FindFirst(viewID) == idx);
				// update point's weight
				views[idx].weight += weight;
			} else {
				// the new view is not in the array,
				// insert it
				views.InsertAt(idx, view_t2(viewID, weight));
				ASSERT(views.IsSorted());
			}
#endif
		}
	}
};
#endif

struct cell_info_t {
	typedef edge_cap_t Type;
	Type f[4]; // faces' weight from the cell outwards
	Type s; // cell's weight towards s-source
	Type t; // cell's weight towards t-sink
	inline const Type* ptr() const { return f; }
	inline Type* ptr() { return f; }
};

typedef CGAL::Triangulation_vertex_base_with_info_3<vert_info_t, kernel_t> vertex_base_t;
typedef CGAL::Triangulation_cell_base_with_info_3<cell_size_t, kernel_t> cell_base_t;
typedef CGAL::Triangulation_data_structure_3<vertex_base_t, cell_base_t> triangulation_data_structure_t;
typedef CGAL::Delaunay_triangulation_3<kernel_t, triangulation_data_structure_t, CGAL::Compact_location> delaunay_t;
typedef delaunay_t::Vertex_handle vertex_handle_t;
typedef delaunay_t::Cell_handle cell_handle_t;
typedef delaunay_t::Facet facet_t;
typedef delaunay_t::Edge edge_t;

#ifdef VALIDATE
typedef CGAL::Triangulation_vertex_base_with_info_3<vert_info_t2, kernel_t> vertex_base_t2;
typedef CGAL::Triangulation_cell_base_with_info_3<cell_size_t, kernel_t> cell_base_t2;
typedef CGAL::Triangulation_data_structure_3<vertex_base_t2, cell_base_t2> triangulation_data_structure_t2;
typedef CGAL::Delaunay_triangulation_3<kernel_t, triangulation_data_structure_t2, CGAL::Compact_location> delaunay_t2;
typedef delaunay_t2::Vertex_handle vertex_handle_t2;
typedef delaunay_t2::Cell_handle cell_handle_t2;
typedef delaunay_t2::Facet facet_t2;
typedef delaunay_t2::Edge edge_t2;
#endif

struct camera_cell_t {
	cell_handle_t cell; // cell containing the camera
	std::vector<facet_t> facets; // all facets on the convex-hull in view of the camera (ordered by importance)
};

struct adjacent_vertex_back_inserter_t {
	const delaunay_t& delaunay;
	const point_t& p;
	vertex_handle_t& v;
	inline adjacent_vertex_back_inserter_t(const delaunay_t& _delaunay, const point_t& _p, vertex_handle_t& _v) : delaunay(_delaunay), p(_p), v(_v) {}
	inline adjacent_vertex_back_inserter_t& operator*() { return *this; }
	inline adjacent_vertex_back_inserter_t& operator++(int) { return *this; }
	inline void operator=(const vertex_handle_t& w) {
		ASSERT(!delaunay.is_infinite(v));
		if (!delaunay.is_infinite(w) && delaunay.geom_traits().compare_distance_3_object()(p, w->point(), v->point()) == CGAL::SMALLER)
			v = w;
	}
};


#ifdef VALIDATE

struct adjacent_vertex_back_inserter_t2 {
	const delaunay_t2& delaunay;
	const point_t& p;
	vertex_handle_t2& v;
	inline adjacent_vertex_back_inserter_t2(const delaunay_t2& _delaunay, const point_t& _p, vertex_handle_t2& _v) : delaunay(_delaunay), p(_p), v(_v) {}
	inline adjacent_vertex_back_inserter_t2& operator*() { return *this; }
	inline adjacent_vertex_back_inserter_t2& operator++(int) { return *this; }
	inline void operator=(const vertex_handle_t2& w) {
		ASSERT(!delaunay.is_infinite(v));
		if (!delaunay.is_infinite(w) && delaunay.geom_traits().compare_distance_3_object()(p, w->point(), v->point()) == CGAL::SMALLER)
			v = w;
	}
};
#endif


typedef TPoint3<kernel_t::RT> DPoint3;
template <typename TYPE>
inline TPoint3<TYPE> CGAL2MVS(const point_t& p) {
	return TPoint3<TYPE>((TYPE)p.x(), (TYPE)p.y(), (TYPE)p.z());
}
template <typename TYPE>
inline point_t MVS2CGAL(const TPoint3<TYPE>& p) {
	return point_t((kernel_t::RT)p.x, (kernel_t::RT)p.y, (kernel_t::RT)p.z);
}

#if 0
// Given a facet, compute the plane containing it
__forceinline CGAL::Plane_3<kernel_t> getFacetPlane(const facet_t& facet)
{
	const point_t& v0(facet.first->vertex((facet.second+1)%4)->point());
	const point_t& v1(facet.first->vertex((facet.second+2)%4)->point());
	const point_t& v2(facet.first->vertex((facet.second+3)%4)->point());

	return CGAL::Plane_3<kernel_t>(v0, v1, v2);
}
#endif

// Check if a point (p) is coplanar with a triangle (a, b, c);
// return orientation type
#if _PLATFORM_X86 && defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target ("no-fma")
#endif
static inline int orientation(const point_t& a, const point_t& b, const point_t& c, const point_t& p)
{
	#if 0
	return CGAL::orientation(a, b, c, p);
	#else
	// inexact_orientation
	const double px = a.x(), py = a.y(), pz = a.z();

	const double pqx = b.x() - px;
	const double pqy = b.y() - py;
	const double pqz = b.z() - pz;

	const double prx = c.x() - px;
	const double pry = c.y() - py;
	const double prz = c.z() - pz;

	const double psx = p.x() - px;
	const double psy = p.y() - py;
	const double psz = p.z() - pz;

	const double det =
		(pqx * pry - prx * pqy) * psz
		- (pqx * psy - psx * pqy) * prz
		+ (prx * psy - psx * pry) * pqz;

	const double max0 = std::max({ fabs(pqx), fabs(pqy), fabs(pqz) });
	const double max1 = std::max({ fabs(prx), fabs(pry), fabs(prz) });
	const double max2 = std::max({ fabs(psx), fabs(psy), fabs(psz) });

	const double eps =
		5.1107127829973299e-15 * max0 * max1 * max2;

	if (det > eps) return CGAL::POSITIVE;
	if (det < -eps) return CGAL::NEGATIVE;

	return CGAL::orientation(a, b, c, p); // exact fallback
	#endif
}
#if _PLATFORM_X86 && defined(__GNUC__)
#pragma GCC pop_options
#endif

// Check if a point (p) is inside a frustum
// given the four corners (a, b, c, d) and the origin (o) of the frustum
inline bool checkPointInside(const point_t& a, const point_t& b, const point_t& c, const point_t& d, const point_t& o, const point_t& p)
{
	return (
		orientation(o, a, b, p) == CGAL::POSITIVE &&
		orientation(o, b, c, p) == CGAL::POSITIVE &&
		orientation(o, c, d, p) == CGAL::POSITIVE &&
		orientation(o, d, a, p) == CGAL::POSITIVE
	);
}

// Given a cell and a camera inside it, if the cell is infinite,
// find all facets on the convex-hull and inside the camera frustum,
// else return all four cell's facets
template <int FacetOrientation>
void fetchCellFacets(const delaunay_t& Tr, const Frustum& viewFrustum, const std::vector<facet_t>& hullFacets, const cell_handle_t& cell, const Image& imageData, std::vector<facet_t>& facets)
{
	if (!Tr.is_infinite(cell)) {
		// store all 4 facets of the cell
		for (int i=0; i<4; ++i) {
			facets.emplace_back(cell, i);
			ASSERT(!Tr.is_infinite(facets.back()));
		}
		return;
	}
	// find all facets on the convex-hull in camera's view
	// create the 4 frustum planes
	ASSERT(facets.empty());
	// loop over all cells
	const point_t ptOrigin(MVS2CGAL(imageData.camera.C));
	for (const facet_t& face: hullFacets) {
		// add face if visible
		const triangle_t verts(Tr.triangle(face));
		if (orientation(verts[0], verts[1], verts[2], ptOrigin) != FacetOrientation)
			continue;
		AABB3 ab(CGAL2MVS<REAL>(verts[0]));
		for (int i=1; i<3; ++i)
			ab.Insert(CGAL2MVS<REAL>(verts[i]));
		if (viewFrustum.Classify(ab) == CULLED)
			continue;
		facets.push_back(face);
	}
}

struct IntersectHelper
{
	double segDiff[3];
	double segDiffNeg[3];
	double v1v0Diff[3]; // 1-0
	double v2v0Diff[3]; // 2-0
	double qv0Diff[3]; // 3-0
	double pv0Diff[3]; // 4-0
	double pqDiff[3]; // 4-3
	double v1qDiff[3]; // 1-3
	double v2qDiff[3]; // 2-3
	double v1pDiff[3]; // 1-4
	double v2pDiff[3]; // 2-4
};

static inline int fasterOrientation(const double* __restrict qDiff, const double* __restrict aDiff, const double* __restrict bDiff)
{
	// inexact_orientation
	constexpr double eps(1e-12);

	const double t1 = qDiff[0] * aDiff[1] - aDiff[0] * qDiff[1];
	const double t2 = qDiff[0] * bDiff[1] - bDiff[0] * qDiff[1];
	const double t3 = aDiff[0] * bDiff[1] - bDiff[0] * aDiff[1];

	const double det = 
		(t1 * bDiff[2]) 
		- (t2 * aDiff[2]) 
		+ (t3 * qDiff[2]);

	return (det > eps) ? CGAL::POSITIVE : (det < -eps ? CGAL::NEGATIVE : CGAL::COPLANAR);
}

// information about an intersection between a segment and a facet
struct intersection_t {
	enum Type {FACET, EDGE, VERTEX};
	cell_handle_t ncell; // cell neighbor to the last intersected facet
	vertex_handle_t v1; // vertex for vertex intersection, 1st edge vertex for edge intersection
	vertex_handle_t v2; // 2nd edge vertex for edge intersection
	facet_t facet; // intersected facet
	Type type; // type of intersection (inside facet, on edge, or vertex)
	float dist; // distance from starting point (camera) to this facet
	bool bigger; // are we advancing away or towards the starting point?
	const Ray3 ray; // the ray from starting point into the direction of the end point (point -> camera/end-point)
	inline intersection_t() {}
	inline intersection_t(const Point3& pt, const Point3& dir) : dist(-FLT_MAX), bigger(true), ray(pt, dir) {}
};

// Check if a segment (p, q) is coplanar with edges of a triangle (a, b, c):
//  coplanar [in,out] : pointer to the 3 int array of indices of the edges coplanar with pq
// return number of entries in coplanar
inline int checkEdges(const point_t& a, const point_t& b, const point_t& c, const point_t& p, const point_t& q, int coplanar[3])
{
	int nCoplanar(0);
	double qDiff[] { q.x()-p.x(), q.y()-p.y(), q.z()-p.z() };
	double aDiff[] { a.x()-p.x(), a.y()-p.y(), a.z()-p.z() };
	double bDiff[] { b.x()-p.x(), b.y()-p.y(), b.z()-p.z() };

	// pq ab
	switch (fasterOrientation(qDiff, aDiff, bDiff)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 0;
	}

	double cDiff[] { c.x()-p.x(), c.y()-p.y(), c.z()-p.z() };
	switch (fasterOrientation(qDiff, bDiff, cDiff)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 1;
	}
	switch (fasterOrientation(qDiff, cDiff, aDiff)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 2;
	}
	return nCoplanar;
}

#if 1

__forceinline int CheckEdges2FastP(
	const IntersectHelper& ih,
	int* __restrict coplanar
)
{
	// return checkEdges2FastP(pv0Diff, b, c, p, segDiff, coplanar);

	constexpr double eps = 1e-12;
	int nCoplanar = 0;

	// Load qDiff once
	const double qx = ih.segDiff[0], qy = ih.segDiff[1], qz = ih.segDiff[2]; // correct

	// Precompute diffs once
	const double ax = -ih.pv0Diff[0], ay = -ih.pv0Diff[1], az = -ih.pv0Diff[2]; // correct

	// b-p is v1-p
	const double bx = ih.v1pDiff[0], by = ih.v1pDiff[1], bz = ih.v1pDiff[2]; // correct
	// c-p is v2-p
	const double cx = ih.v2pDiff[0], cy = ih.v2pDiff[1], cz = ih.v2pDiff[2];// correct

	// Precompute shared 2D cross terms with q
	const double qxa_y = qx * ay - ax * qy;
	const double qxb_y = qx * by - bx * qy;
	const double qxc_y = qx * cy - cx * qy;

	// ---- Edge 1: pq, ab ----
	{
		const double t3 = ax * by - bx * ay;
		const double det = qxa_y * bz - qxb_y * az + t3 * qz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 0;
	}

	// ---- Edge 2: pq, bc ----
	{
		const double t3 = bx * cy - cx * by;
		const double det = qxb_y * cz - qxc_y * bz + t3 * qz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 1;
	}

	// ---- Edge 3: pq, ca ----
	{
		const double t3 = cx * ay - ax * cy;
		const double det = qxc_y * az - qxa_y * cz + t3 * qz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 2;
	}

	return nCoplanar;
}

__forceinline int CheckEdges2FastQ(
	const IntersectHelper& ih,
	int* __restrict coplanar)
{
	constexpr double eps = 1e-12;
	int nCoplanar = 0;

	// return checkEdges2FastQ(qv0Diff, b, c, q, segDiffN, coplanar);


	// Load qDiff once
	const double qx = ih.segDiffNeg[0], qy = ih.segDiffNeg[1], qz = ih.segDiffNeg[2]; // correct

	// Precompute diffs once
	const double ax = -ih.qv0Diff[0], ay = -ih.qv0Diff[1], az = -ih.qv0Diff[2]; // correct

	// b-q is v1-q
	const double bx = ih.v1qDiff[0], by = ih.v1qDiff[1], bz = ih.v1qDiff[2]; // correct
	// c-q is v2-q
	const double cx = ih.v2qDiff[0], cy = ih.v2qDiff[1], cz = ih.v2qDiff[2]; // correct

	// Precompute shared 2D cross terms with q
	const double qxa_y = qx * ay - ax * qy;
	const double qxb_y = qx * by - bx * qy;
	const double qxc_y = qx * cy - cx * qy;

	// ---- Edge 1: pq, ab ----
	{
		const double t3 = ax * by - bx * ay;
		const double det = qxa_y * bz - qxb_y * az + t3 * qz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 0;
	}

	// ---- Edge 2: pq, bc ----
	{
		const double t3 = bx * cy - cx * by;
		const double det = qxb_y * cz - qxc_y * bz + t3 * qz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 1;
	}

	// ---- Edge 3: pq, ca ----
	{
		const double t3 = cx * ay - ax * cy;
		const double det = qxc_y * az - qxa_y * cz + t3 * qz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 2;
	}

	return nCoplanar;
}

#else

__forceinline int checkEdges2_fast(
  const double* __restrict negPa,
  const point_t& b,
  const point_t& c,
  const point_t& p,
  const double* __restrict qDiff,
  int* __restrict coplanar)
{
  constexpr double eps = 1e-12;
  int nCoplanar = 0;

  // Load qDiff once
  const double qx = qDiff[0], qy = qDiff[1], qz = qDiff[2];

  // Precompute diffs once
  const double ax = -negPa[0], ay = -negPa[1], az = -negPa[2];
  const double bx = b.x() - p.x(), by = b.y() - p.y(), bz = b.z() - p.z();
  const double cx = c.x() - p.x(), cy = c.y() - p.y(), cz = c.z() - p.z();

  // Precompute shared 2D cross terms with q
  const double qxa_y = qx * ay - ax * qy;
  const double qxb_y = qx * by - bx * qy;
  const double qxc_y = qx * cy - cx * qy;

  // ---- Edge 1: pq, ab ----
  {
    const double t3 = ax * by - bx * ay;
    const double det = qxa_y * bz - qxb_y * az + t3 * qz;
    if (det > eps) return -1;
    if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 0;
  }

  // ---- Edge 2: pq, bc ----
  {
    const double t3 = bx * cy - cx * by;
    const double det = qxb_y * cz - qxc_y * bz + t3 * qz;
    if (det > eps) return -1;
    if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 1;
  }

  // ---- Edge 3: pq, ca ----
  {
    const double t3 = cx * ay - ax * cy;
    const double det = qxc_y * az - qxa_y * cz + t3 * qz;
    if (det > eps) return -1;
    if (det >= -eps && det <= eps) coplanar[nCoplanar++] = 2;
  }

  return nCoplanar;
}
#endif

#if 0 // original
static inline int orientationv(const point_t& a, const point_t& b, const point_t& c, const point_t& p)
{
	// inexact_orientation
	const double& px = a.x(); const double& py = a.y(); const double& pz = a.z();
	const double pqx(b.x()-px); const double prx(c.x()-px); const double psx(p.x()-px);
	const double pqy(b.y()-py); const double pry(c.y()-py); const double psy(p.y()-py);
	#if 1
	const double det((pqx*pry-prx*pqy)*(p.z()-pz) - (pqx*psy-psx*pqy)*(c.z()-pz) + (prx*psy-psx*pry)*(b.z()-pz));
	const double eps(1e-12);
	#else // very slow due to ABS()
	const double pqz(b.z()-pz); const double prz(c.z()-pz); const double psz(p.z()-pz);
	const double det(CGAL::determinant(
		pqx, pqy, pqz,
		prx, pry, prz,
		psx, psy, psz));
	const double max0(MAXF3(ABS(pqx), ABS(pqy), ABS(pqz)));
	const double max1(MAXF3(ABS(prx), ABS(pry), ABS(prz)));
	const double eps(5.1107127829973299e-15 * MAXF(max0, max1));
	#endif
	if (det >  eps) return CGAL::POSITIVE;
	if (det < -eps) return CGAL::NEGATIVE;
	return CGAL::COPLANAR;
}
inline int checkEdgesv(const point_t& a, const point_t& b, const point_t& c, const point_t& p, const point_t& q, int coplanar[3])
{
	int nCoplanar(0);
	switch (orientationv(p,q,a,b)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 0;
	}
	switch (orientationv(p,q,b,c)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 1;
	}
	switch (orientationv(p,q,c,a)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 2;
	}
	return nCoplanar;
}

inline Plane getFacetPlanev(const facet_t& facet)
{
	const point_t& v0(facet.first->vertex((facet.second+1)%4)->point());
	const point_t& v1(facet.first->vertex((facet.second+2)%4)->point());
	const point_t& v2(facet.first->vertex((facet.second+3)%4)->point());
	return Plane(CGAL2MVS<REAL>(v0), CGAL2MVS<REAL>(v1), CGAL2MVS<REAL>(v2));
}
int intersectv(const triangle_t& t, const segment_t& s, int coplanar[3])
{
	const point_t& a = t.vertex(0);
	const point_t& b = t.vertex(1);
	const point_t& c = t.vertex(2);
	const point_t& p = s.source();
	const point_t& q = s.target();

	switch (orientationv(a,b,c,p)) {
	case CGAL::POSITIVE:
		switch (orientationv(a,b,c,q)) {
		case CGAL::POSITIVE:
			// the segment lies in the positive open halfspaces defined by the
			// triangle's supporting plane
			return -1;
		case CGAL::COPLANAR:
			// q belongs to the triangle's supporting plane
			// p sees the triangle in counterclockwise order
			return checkEdgesv(a,b,c,p,q,coplanar);
		case CGAL::NEGATIVE:
			// p sees the triangle in counterclockwise order
			return checkEdgesv(a,b,c,p,q,coplanar);
		default:
			break;
		}
	case CGAL::NEGATIVE:
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// q sees the triangle in counterclockwise order
			return checkEdgesv(a,b,c,q,p,coplanar);
		case CGAL::COPLANAR:
			// q belongs to the triangle's supporting plane
			// p sees the triangle in clockwise order
			return checkEdgesv(a,b,c,q,p,coplanar);
		case CGAL::NEGATIVE:
			// the segment lies in the negative open halfspaces defined by the
			// triangle's supporting plane
			return -1;
		default:
			break;
		}
	case CGAL::COPLANAR: // p belongs to the triangle's supporting plane
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// q sees the triangle in counterclockwise order
			return checkEdgesv(a,b,c,q,p,coplanar);
		case CGAL::COPLANAR:
			// the segment is coplanar with the triangle's supporting plane
			// as we know that it is inside the tetrahedron it intersects the face
			//coplanar[0] = coplanar[1] = coplanar[2] = 3;
			return 3;
		case CGAL::NEGATIVE:
			// q sees the triangle in clockwise order
			return checkEdgesv(a,b,c,p,q,coplanar);
		default:
			break;
		}
	}
	ASSERT("should not happen" == NULL);
	return -1;
}
bool intersectv(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	ASSERT(!in_facets.empty());
	static const int facet_vertex_order[] = {2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0};
	int coplanar[3];
	const REAL prevDist(inter.dist);
	for (const facet_t& in_facet: in_facets) {
		ASSERT(!Tr.is_infinite(in_facet));
		const int nb_coplanar(intersectv(Tr.triangle(in_facet), seg, coplanar));
		if (nb_coplanar >= 0) {
			// skip this cell if the intersection is not in the desired direction
			const REAL interDist(inter.ray.IntersectsDist(getFacetPlanev(in_facet)));
			if ((interDist > prevDist) != inter.bigger)
				continue;
			// vertices of facet i: j = 4 * i, vertices = facet_vertex_order[j,j+1,j+2] negative orientation
			inter.facet = in_facet;
			inter.dist = interDist;
			switch (nb_coplanar) {
			case 0: {
				// face intersection
				inter.type = intersection_t::FACET;
				// now find next facets to be checked as
				// the three faces in the neighbor cell different than the origin face
				out_facets.clear();
				const cell_handle_t nc(inter.facet.first->neighbor(inter.facet.second));
				ASSERT(!Tr.is_infinite(nc));
				for (int i=0; i<4; ++i)
					if (nc->neighbor(i) != inter.facet.first)
						out_facets.push_back(facet_t(nc, i));
				return true; }
			case 1: {
				// coplanar with 1 edge = intersect edge
				const int j(4 * inter.facet.second);
				const int i1(j + coplanar[0]);
				inter.type = intersection_t::EDGE;
				inter.v1 = inter.facet.first->vertex(facet_vertex_order[i1+0]);
				inter.v2 = inter.facet.first->vertex(facet_vertex_order[i1+1]);
				// now find next facets to be checked as
				// the two faces in this cell opposing this edge
				out_facets.clear();
				const edge_t out_edge(inter.facet.first, facet_vertex_order[i1+0], facet_vertex_order[i1+1]);
				const typename delaunay_t::Cell_circulator efc(Tr.incident_cells(out_edge));
				typename delaunay_t::Cell_circulator ifc(efc);
				do {
					const cell_handle_t c(ifc);
					if (c == inter.facet.first) continue;
					const facet_t f1(c, c->index(inter.v1));
					if (!Tr.is_infinite(f1))
						out_facets.push_back(f1);
					const facet_t f2(c, c->index(inter.v2));
					if (!Tr.is_infinite(f2))
						out_facets.push_back(f2);
				} while (++ifc != efc);
				return true; }
			case 2: {
				// coplanar with 2 edges = hit a vertex
				// find vertex index
				const int j(4 * inter.facet.second);
				const int i1(j + coplanar[0]);
				const int i2(j + coplanar[1]);
				int i;
				if (facet_vertex_order[i1] == facet_vertex_order[i2] || facet_vertex_order[i1] == facet_vertex_order[i2+1]) {
					i = facet_vertex_order[i1];
				} else
				if (facet_vertex_order[i1+1] == facet_vertex_order[i2] || facet_vertex_order[i1+1] == facet_vertex_order[i2+1]) {
					i = facet_vertex_order[i1+1];
				} else {
					ASSERT("2 edges intersections without common vertex" == NULL);
				}
				inter.type = intersection_t::VERTEX;
				inter.v1 = inter.facet.first->vertex(i);
				ASSERT(!Tr.is_infinite(inter.v1));
				if (inter.v1->point() == seg.target()) {
					// target reached
					out_facets.clear();
					return false;
				}
				// now find next facets to be checked as
				// the faces in the cells around opposing this common vertex
				out_facets.clear();
				struct cell_back_inserter_t {
					const delaunay_t& Tr;
					const vertex_handle_t v;
					const cell_handle_t current_cell;
					std::vector<facet_t>& out_facets;
					inline cell_back_inserter_t(const delaunay_t& _Tr, const intersection_t& inter, std::vector<facet_t>& _out_facets)
						: Tr(_Tr), v(inter.v1), current_cell(inter.facet.first), out_facets(_out_facets) {}
					inline cell_back_inserter_t& operator*() { return *this; }
					inline cell_back_inserter_t& operator++(int) { return *this; }
					inline void operator=(cell_handle_t c) {
						if (c == current_cell)
							return;
						const facet_t f(c, c->index(v));
						if (Tr.is_infinite(f))
							return;
						out_facets.push_back(f);
					}
				};
				Tr.finite_incident_cells(inter.v1, cell_back_inserter_t(Tr, inter, out_facets));
				return true; }
			}
			// coplanar with 3 edges = tangent = impossible?
			break;
		}
	}
	// Bad end: no intersection found and we are not at the end of the segment (very rarely, but it happens)!
	out_facets.clear();
	return false;
}
#else

static constexpr uint32_t facetIdx[4][3] = {
  {2,1,3}, // i=0 v2,v1,v3
  {2,3,0}, // i=1 v1,v2,v3
  {0,3,1}, // i=2 v2,v1,v3
  {0,1,2}  // i=3 v1,v2,v3
};

// Check intersection between a facet (f) and a segment (s)
// (derived from CGAL::do_intersect in CGAL/Triangle_3_Segment_3_do_intersect.h)
//  coplanar [out] : pointer to the 3 int array of indices of the edges coplanar with (s)
// return -1 if there is no intersection or
// the number of edges coplanar with the segment (0 = intersection inside the triangle)
#if 0
#if 1 // try again
__forceinline int intersect(
	const IntersectHelper& ih,
	int* __restrict coplanar
)
{
	// edges: (b - a), (c - a)
	const double bax = ih.v1v0Diff[0], bay = ih.v1v0Diff[1], baz = ih.v1v0Diff[2];
	const double cax = ih.v2v0Diff[0], cay = ih.v2v0Diff[1], caz = ih.v2v0Diff[2];

	// unnormalized normal n = (b - a) x (c - a)
	const double nx = bay * caz - baz * cay;
	const double ny = baz * cax - bax * caz;
	const double nz = bax * cay - bay * cax;

	// endpoint offsets from 'a' (compute once; cheap)
	const double pax = ih.pv0Diff[0], pay = ih.pv0Diff[1], paz = ih.pv0Diff[2];
	const double qax = ih.qv0Diff[0], qay = ih.qv0Diff[1], qaz = ih.qv0Diff[2];

	// signed distances to plane (no 'd' needed): dp = n�(p-a), dq = n�(q-a)
	const double dp = nx * pax + ny * pay + nz * paz;
	const double dq = nx * qax + ny * qay + nz * qaz;

	// classify with tight epsilon (branchless-ish)
	constexpr double eps = 1e-12;
	const int sp = (dp > eps) - (dp < -eps);
	const int sq = (dq > eps) - (dq < -eps);

	// both strictly same side -> no intersection with plane
	if ((sp > 0 && sq > 0) || (sp < 0 && sq < 0))
		return -1;

	// entirely coplanar with plane
	if (sp == 0 && sq == 0)
		return 3;

	// choose which endpoint to use for edge tests:
	// Use P-side when (sp >= 0 && sq <= 0); otherwise use Q-side.
	if (sp >= 0 && sq <= 0) {
		return CheckEdges2FastP(ih, coplanar);
	}
	else {
		return CheckEdges2FastQ(ih, coplanar);
	}
}
#else
int intersect(const vertex_handle_t vs[3], const segment_t& s, const double* __restrict segDiff /* target - source */, const double* __restrict segDiffN /* source - target */, int* __restrict coplanar)
{
	const point_t& a = vs[0]->point(); // t.vertex(0);
	const point_t& b = vs[1]->point(); // t.vertex(1);
	const point_t& c = vs[2]->point(); // t.vertex(2);
	const point_t& p = s.source();
	const point_t& q = s.target();

	const double bDiff[] { b.x()-a.x(), b.y()-a.y(), b.z()-a.z() };
	const double cDiff[] { c.x()-a.x(), c.y()-a.y(), c.z()-a.z() };
	const double pDiff[] { p.x()-a.x(), p.y()-a.y(), p.z()-a.z() };
	const double qDiff[] { q.x()-a.x(), q.y()-a.y(), q.z()-a.z() };

	switch (fasterOrientation(bDiff, cDiff, pDiff)) { //orientation(a,b,c,p)) {
		case CGAL::POSITIVE:
			switch (fasterOrientation(bDiff, cDiff, qDiff)) { //orientation(a,b,c,q)) {
				case CGAL::POSITIVE:
					// the segment lies in the positive open halfspaces defined by the
					// triangle's supporting plane
					return -1;
				case CGAL::COPLANAR:
					// q belongs to the triangle's supporting plane
					// p sees the triangle in counterclockwise order
					//return checkEdges(a,b,c,p,q,coplanar);
					return checkEdges2_fast(pDiff,b,c,p,segDiff,coplanar);
				case CGAL::NEGATIVE:
					// p sees the triangle in counterclockwise order
					//return checkEdges(a,b,c,p,q,coplanar);
					return checkEdges2_fast(pDiff,b,c,p,segDiff,coplanar);
				default:
					break;
				}
		case CGAL::NEGATIVE:
			switch (fasterOrientation(bDiff, cDiff, qDiff)) { //orientation(a,b,c,q)) {
				case CGAL::POSITIVE:
					// q sees the triangle in counterclockwise order
					//return checkEdges(a,b,c,q,p,coplanar);
					return checkEdges2_fast(qDiff,b,c,q,segDiffN,coplanar);
				case CGAL::COPLANAR:
					// q belongs to the triangle's supporting plane
					// p sees the triangle in clockwise order
					//return checkEdges(a,b,c,q,p,coplanar);
					return checkEdges2_fast(qDiff,b,c,q,segDiffN,coplanar);
				case CGAL::NEGATIVE:
					// the segment lies in the negative open halfspaces defined by the
					// triangle's supporting plane
					return -1;
				default:
					break;
				}
		case CGAL::COPLANAR: // p belongs to the triangle's supporting plane
			switch (fasterOrientation(bDiff, cDiff, qDiff)) { //orientation(a,b,c,q)) {
				case CGAL::POSITIVE:
					// q sees the triangle in counterclockwise order
					//return checkEdges(a,b,c,q,p,coplanar);
					return checkEdges2_fast(qDiff,b,c,q,segDiffN,coplanar);
				case CGAL::COPLANAR:
					// the segment is coplanar with the triangle's supporting plane
					// as we know that it is inside the tetrahedron it intersects the face
					//coplanar[0] = coplanar[1] = coplanar[2] = 3;
					return 3;
				case CGAL::NEGATIVE:
					// q sees the triangle in clockwise order
					//return checkEdges(a,b,c,p,q,coplanar);
					return checkEdges2_fast(pDiff,b,c,p,segDiff,coplanar);
				default:
					break;
				}
	}
	ASSERT("should not happen" == NULL);
	return -1;
}
#endif
#endif

#if 0
__forceinline bool IntersectsPrecheck(
	const SEACAVE::Ray3& ray,
	const double v0x,
	const double v0y,
	const double v0z,
	double prevDist,
	bool bigger,
	double& voOut,
	double& vdOut)
{
	const double v10x = ih.v1v0Diff[0];
	const double v10y = ih.v1v0Diff[1];
	const double v10z = ih.v1v0Diff[2];

	const double v20x = ih.v2v0Diff[0];
	const double v20y = ih.v2v0Diff[1];
	const double v20z = ih.v2v0Diff[2];

	const double nx = v10y * v20z - v10z * v20y;
	const double ny = v10z * v20x - v10x * v20z;
	const double nz = v10x * v20y - v10y * v20x;

	const double d = -(nx * v0x + ny * v0y + nz * v0z);

	const double dx = ray.m_vDir.x();
	const double dy = ray.m_vDir.y();
	const double dz = ray.m_vDir.z();
	const double ox = ray.m_pOrig.x();
	const double oy = ray.m_pOrig.y();
	const double oz = ray.m_pOrig.z();

	const double vd = nx * dx + ny * dy + nz * dz;
	constexpr double eps = 1e-12;
	if (std::abs(vd) < eps)
		return false; // parallel or nearly so

	const double vo = -(nx * ox + ny * oy + nz * oz + d);

	// Compare without divide:
	// interDist > prevDist  iff  (vd >= 0 ? vo - prevDist*vd > 0 : vo - prevDist*vd < 0)
	const double delta = vo - prevDist * vd;
	const bool isGreater = (vd >= 0.0) ? (delta > 0.0) : (delta < 0.0);
	if (isGreater != bigger)
		return false;

	voOut = vo;
	vdOut = vd;
	return true;
}
#endif

// Return false if nearly parallel or compare fails; otherwise return vo, vd
__forceinline bool IntersectsPrecheckFast(
	const SEACAVE::Ray3& ray,
	double v0x, double v0y, double v0z,
	double nx, double ny, double nz,
	double prevDist, bool bigger,
	double& voOut, double& vdOut)
{
	const double dx = ray.m_vDir.x(), dy = ray.m_vDir.y(), dz = ray.m_vDir.z();
	const double ox = ray.m_pOrig.x(), oy = ray.m_pOrig.y(), oz = ray.m_pOrig.z();

	const double vd = nx * dx + ny * dy + nz * dz;
	constexpr double eps = 1e-12;
	if (FastAbsD(vd) < eps) return false;

	const double d = -(nx * v0x + ny * v0y + nz * v0z);
	const double vo = -(nx * ox + ny * oy + nz * oz + d);

	const double delta = vo - prevDist * vd;

	// branchless sign-consistent comparison
	if (((delta * vd) > 0.0) != bigger) return false;

	voOut = vo; vdOut = vd;
	return true;
}

// Shared core for edge checks.
// Select endpoint set with template bool UseP: true => P-side, false => Q-side.
#if 1 // JPB WIP BUG Alternative version which drops the "1" case and improves performance
template<bool UseP>
__forceinline int CheckEdges2FastCore(
	double sx, double sy, double sz,
	double ax, double ay, double az,   // a - endpoint (v0 - p or v0 - q)
	double bx, double by, double bz,   // v1 - endpoint
	double cx, double cy, double cz,   // v2 - endpoint
	int* __restrict coplanar)
{
	constexpr double eps = 1e-12;
	int nCop = 0;

	const double sxa_y = sx * ay - ax * sy;
	const double sxb_y = sx * by - bx * sy;
	const double sxc_y = sx * cy - cx * sy;

	{ // (s, a, b)
		const double t3 = ax * by - bx * ay;
		const double det = sxa_y * bz - sxb_y * az + t3 * sz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCop++] = 0;
	}
	{ // (s, b, c)
		const double t3 = bx * cy - cx * by;
		const double det = sxb_y * cz - sxc_y * bz + t3 * sz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCop++] = 1;
	}
	{ // (s, c, a)
		const double t3 = cx * ay - ax * cy;
		const double det = sxc_y * az - sxa_y * cz + t3 * sz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCop++] = 2;
	}
	return nCop;
}

// Plane-side classification; pick which endpoint (P vs Q) to use for edges.
// Returns -1 (no hit), 3 (coplanar), or 0/1/2 count from checkEdges2FastCore.
__forceinline int IntersectPlaneAndEdges(
	// plane normal (v1-v0) x (v2-v0)
	double nx, double ny, double nz,
	// p-v0, q-v0
	double pax, double pay, double paz,
	double qax, double qay, double qaz,
	double px, double py, double pz,
	double v1x, double v1y, double v1z,
	double v2x, double v2y, double v2z,
	double qx, double qy, double qz,
	// segment vectors
	double sPx, double sPy, double sPz,  // q - p
	double sQx, double sQy, double sQz,  // p - q
	int* RESTRICT coplanar) {

	//	const point_t& a = t.vertex(0);
	//	const point_t& b = t.vertex(1);
	//	const point_t& c = t.vertex(2);
	//	const point_t& p = s.source();
	//	const point_t& q = s.target();

	const double dp = nx * pax + ny * pay + nz * paz;
	const double dq = nx * qax + ny * qay + nz * qaz;

	constexpr double eps = 1e-12;

	const bool pPos = dp > eps;
	const bool pNeg = dp < -eps;
	const bool qPos = dq > eps;
	const bool qNeg = dq < -eps;

	if ((pPos && qPos) || (pNeg && qNeg)) return -1;
	if (!(pPos | pNeg | qPos | qNeg)) return 3;

	if (!pNeg && !qPos) {
		// P-side: a = v0-p = -pax, b = v1-p, c = v2-p
		const double bxP = v1x - px, byP = v1y - py, bzP = v1z - pz;
		const double cxP = v2x - px, cyP = v2y - py, czP = v2z - pz;
		return CheckEdges2FastCore<true>(sPx, sPy, sPz, -pax, -pay, -paz, bxP, byP, bzP, cxP, cyP, czP, coplanar);
	}
	else {
		// Q-side: a = v0-q = -qax, b = v1-q, c = v2-q
		const double bxQ = v1x - qx, byQ = v1y - qy, bzQ = v1z - qz;
		const double cxQ = v2x - qx, cyQ = v2y - qy, czQ = v2z - qz;
		return CheckEdges2FastCore<false>(sQx, sQy, sQz, -qax, -qay, -qaz, bxQ, byQ, bzQ, cxQ, cyQ, czQ, coplanar);
	}
}

constexpr int facet_vertex_order[] = { 2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0 };

bool intersect(const delaunay_t& Tr,
	const segment_t& seg,
	const std::vector<facet_t>& in_facets,
	std::vector<facet_t>& out_facets,
	intersection_t& inter,
	const uint32_t*       __restrict cellNbrID,
	const cell_handle_t*  __restrict allCellsArr) {

	ASSERT(!in_facets.empty());

	int coplanar[3];
	const double prevDist = inter.dist;

	// segment endpoints and direction
	const point_t& p = seg.source();
	const point_t& q = seg.target();
	const double px = p.x(), py = p.y(), pz = p.z();
	const double qx = q.x(), qy = q.y(), qz = q.z();
	const double sPx = qx - px, sPy = qy - py, sPz = qz - pz;
	const double sQx = -sPx, sQy = -sPy, sQz = -sPz;

	for (const facet_t& inFacet : in_facets) {
		ASSERT(!Tr.is_infinite(inFacet));

		const uint32_t* __restrict m = facetIdx[inFacet.second]; // inFacet.second <= 3 guaranteed

		const point_t& v0 = inFacet.first->vertex(m[0])->point();
		const point_t& v1 = inFacet.first->vertex(m[1])->point();
		const point_t& v2 = inFacet.first->vertex(m[2])->point();

		const double v0x = v0.x(), v0y = v0.y(), v0z = v0.z();
		const double v1x = v1.x(), v1y = v1.y(), v1z = v1.z();
		const double v2x = v2.x(), v2y = v2.y(), v2z = v2.z();

		// plane normal
		double nx, ny, nz;

		const double bax = v1x - v0x, bay = v1y - v0y, baz = v1z - v0z;
		const double cax = v2x - v0x, cay = v2y - v0y, caz = v2z - v0z;

		nx = bay * caz - baz * cay;
		ny = baz * cax - bax * caz;
		nz = bax * cay - bay * cax;

		// p-v0, q-v0
		const double pax = px - v0x, pay = py - v0y, paz = pz - v0z;
		const double qax = qx - v0x, qay = qy - v0y, qaz = qz - v0z;

		const int nbCoplanar = IntersectPlaneAndEdges(
			nx, ny, nz,
			pax, pay, paz,
			qax, qay, qaz,
			px, py, pz,
			v1x, v1y, v1z,
			v2x, v2y, v2z,
			qx, qy, qz,
			sPx, sPy, sPz,
			sQx, sQy, sQz,
			coplanar);

		if (nbCoplanar < 0)
			continue;

		double vo, vd;
		if (!IntersectsPrecheckFast(
			inter.ray,
			v0x, v0y, v0z,
			nx, ny, nz,
			prevDist,
			inter.bigger,
			vo, vd))
			continue;

		inter.facet = inFacet;
		const cell_handle_t back = inter.facet.first;
		// Cache the back cell's ID once. The cell record was just touched by
		// the vertex/point loads above, so this read is free (same line).
		const cell_size_t backID = back->info();

		// ------------------------------------------------------------
		// FAST FACET PATH (dominant)
		// ------------------------------------------------------------
		if (nbCoplanar == 0) {
			inter.type = intersection_t::FACET;

			out_facets.clear();

			// Flat-array hop: backID -> ncID via cellNbrID, then handle via allCellsArr.
			// Replaces back->neighbor(facet.second) random cell-pool deref.
			const cell_size_t   ncID = cellNbrID[(size_t)backID * 4 + inter.facet.second];
			const cell_handle_t nc   = allCellsArr[ncID];
			ASSERT(!Tr.is_infinite(nc));

			// Back-check via 4 sequential u32 reads (one 16 B chunk) instead of
			// 4 random nc->neighbor(i) derefs.
			const uint32_t* __restrict ncNbrs = cellNbrID + (size_t)ncID * 4;
			for (int i = 0; i < 4; ++i) {
				if (ncNbrs[i] == backID)
					continue;
				out_facets.emplace_back(nc, i);
			}

			inter.dist = (float)(vo / vd);
			return true;
		}

		// ------------------------------------------------------------
		// nbCoplanar == 1  (edge intersection)
		// ------------------------------------------------------------
		if (nbCoplanar == 1) {
			const int j = 4 * inter.facet.second;
			const int i1 = j + coplanar[0];

			inter.type = intersection_t::EDGE;
			inter.v1 = inter.facet.first->vertex(facet_vertex_order[i1 + 0]);
			inter.v2 = inter.facet.first->vertex(facet_vertex_order[i1 + 1]);

			out_facets.clear();
			const edge_t out_edge(inter.facet.first,
				facet_vertex_order[i1 + 0],
				facet_vertex_order[i1 + 1]);

			typename delaunay_t::Cell_circulator efc(Tr.incident_cells(out_edge));
			typename delaunay_t::Cell_circulator ifc = efc;
			do {
				const cell_handle_t c(ifc);
				if (c == inter.facet.first) continue;

				const facet_t f1(c, c->index(inter.v1));
				if (!Tr.is_infinite(f1)) out_facets.push_back(f1);

				const facet_t f2(c, c->index(inter.v2));
				if (!Tr.is_infinite(f2)) out_facets.push_back(f2);
			} while (++ifc != efc);

			inter.dist = (float)(vo / vd);
			return true;
		}

		// ------------------------------------------------------------
		// nbCoplanar == 2  (candidate vertex hit)
		// ------------------------------------------------------------
		// NEW: true vertex gate
		constexpr double vertexEps = 1e-12;
		const bool isVertexHit =
			(vo >= -vertexEps && vo <= vertexEps) ||
			((vd - vo) >= -vertexEps && (vd - vo) <= vertexEps);

		if (!isVertexHit) {
			// Treat edge-like as FACET (cheap, conservative)
			inter.type = intersection_t::FACET;

			out_facets.clear();

			// Flat-array hop, same as the dominant FACET path.
			const cell_size_t   ncID = cellNbrID[(size_t)backID * 4 + inter.facet.second];
			const cell_handle_t nc   = allCellsArr[ncID];
			ASSERT(!Tr.is_infinite(nc));

			const uint32_t* __restrict ncNbrs = cellNbrID + (size_t)ncID * 4;
			const uint32_t n0 = ncNbrs[0];
			const uint32_t n1 = ncNbrs[1];
			const uint32_t n2 = ncNbrs[2];
			const uint32_t n3 = ncNbrs[3];

			if (n0 != backID) out_facets.emplace_back(nc, 0);
			if (n1 != backID) out_facets.emplace_back(nc, 1);
			if (n2 != backID) out_facets.emplace_back(nc, 2);
			if (n3 != backID) out_facets.emplace_back(nc, 3);

			inter.dist = (float)(vo / vd);
			return true;
		}

		// ------------------------------------------------------------
		// TRUE VERTEX HIT (rare)
		// ------------------------------------------------------------
		inter.type = intersection_t::VERTEX;

		const int j = 4 * inter.facet.second;
		const int i1 = j + coplanar[0];
		const int i2 = j + coplanar[1];

		int vi;
		if (facet_vertex_order[i1] == facet_vertex_order[i2] ||
			facet_vertex_order[i1] == facet_vertex_order[i2 + 1]) {
			vi = facet_vertex_order[i1];
		}
		else {
			vi = facet_vertex_order[i1 + 1];
		}

		inter.v1 = inter.facet.first->vertex(vi);
		ASSERT(!Tr.is_infinite(inter.v1));

		out_facets.clear();
		if (inter.v1->point() == seg.target()) {
			return false;
		}

		struct cell_back_inserter_t {
			const delaunay_t& Tr;
			const vertex_handle_t v;
			const cell_handle_t current_cell;
			std::vector<facet_t>& out;
			inline cell_back_inserter_t(
				const delaunay_t& _Tr,
				const intersection_t& inter,
				std::vector<facet_t>& _out)
				: Tr(_Tr), v(inter.v1), current_cell(inter.facet.first), out(_out) {
			}
			inline cell_back_inserter_t& operator*() { return *this; }
			inline cell_back_inserter_t& operator++(int) { return *this; }
			inline void operator=(cell_handle_t c) {
				if (c == current_cell) return;
				const facet_t f(c, c->index(v));
				if (!Tr.is_infinite(f))
					out.push_back(f);
			}
		};

		Tr.finite_incident_cells(
			inter.v1,
			cell_back_inserter_t(Tr, inter, out_facets));

		inter.dist = (float)(vo / vd);
		return true;
	}

	out_facets.clear();
	return false;
}

#else
template<bool UseP>
__forceinline int CheckEdges2FastCore(
	double sx, double sy, double sz,   // segment (q - p) if UseP, else (p - q)
	double ax, double ay, double az,   // -(endpoint - v0)
	double bx, double by, double bz,   // v1 - endpoint
	double cx, double cy, double cz,   // v2 - endpoint
	int* __restrict coplanar) {
	constexpr double eps = 1e-12;
	int nCop = 0;

	// (-ay*sx) - (-ax*sy)
	// a = -aysx + aysy
	// ay*sy - ay*sx
	const double sxa_y = ax * sy - sx * ay; // sx * ay - ax * sy;
	const double sxb_y = sx * by - bx * sy;
	const double sxc_y = cx * sy - sx * cy; // sx * cy - cx * sy;


	{ // (s, ab)
		const double t3 = bx * ay - ax * by; // ax * by - bx * ay;
		const double det = sxa_y * bz + sxb_y * az + t3 * sz; // sxa_y * bz - sxb_y * az + t3 * sz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCop++] = 0;
	}
	{ // (s, bc)
		const double t3 = bx * cy - cx * by;
		const double det = sxb_y * cz + sxc_y * bz + t3 * sz; // sxb_y * cz - sxc_y * bz + t3 * sz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCop++] = 1;
	}
	{ // (s, ca)
		const double t3 = ax * cy - cx * ay; // cx* ay - ax * cy;
		const double det = sxc_y * az - sxa_y * cz + t3 * sz; // sxc_y * az - sxa_y * cz + t3 * sz;
		if (det > eps) return -1;
		if (det >= -eps && det <= eps) coplanar[nCop++] = 2;
	}
	return nCop;
}

// Plane-side classification; pick which endpoint (P vs Q) to use for edges.
// Returns -1 (no hit), 3 (coplanar), or 0/1/2 count from checkEdges2FastCore.
__forceinline int IntersectPlaneAndEdges(
	// plane normal (v1-v0) x (v2-v0)
	double nx, double ny, double nz,
	// p-v0, q-v0
	double pax, double pay, double paz,
	double qax, double qay, double qaz,
	double px, double py, double pz,
	double v1x, double v1y, double v1z,
	double v2x, double v2y, double v2z,
	double qx, double qy, double qz,
	// segment vectors
	double sPx, double sPy, double sPz,  // q - p
	double sQx, double sQy, double sQz,  // p - q
	int* RESTRICT coplanar) {

//	const point_t& a = t.vertex(0);
//	const point_t& b = t.vertex(1);
//	const point_t& c = t.vertex(2);
//	const point_t& p = s.source();
//	const point_t& q = s.target();

	const double dp = nx * pax + ny * pay + nz * paz;
	const double dq = nx * qax + ny * qay + nz * qaz;

	constexpr double eps = 1e-12;
	const int sp = (dp > eps) - (dp < -eps);
	const int sq = (dq > eps) - (dq < -eps);

	if ((sp > 0 && sq > 0) || (sp < 0 && sq < 0)) return -1;
	if (sp == 0 && sq == 0) return 3;

	if (sp >= 0 && sq <= 0) {
		// P-side endpoint = p: a = -(p-v0), b = v1-p, c = v2-p
		const double bxP = v1x - px, byP = v1y - py, bzP = v1z - pz;
		const double cxP = v2x - px, cyP = v2y - py, czP = v2z - pz;
		return CheckEdges2FastCore<true>(sPx, sPy, sPz, pax, pay, paz, bxP, byP, bzP, cxP, cyP, czP, coplanar);
	} else {
		// Q-side endpoint = q: a = -(q-v0), b = v1-q, c = v2-q
		const double bxQ = v1x - qx, byQ = v1y - qy, bzQ = v1z - qz;
		const double cxQ = v2x - qx, cyQ = v2y - qy, czQ = v2z - qz;
		return CheckEdges2FastCore<false>(sQx, sQy, sQz, qax, qay, qaz, bxQ, byQ, bzQ, cxQ, cyQ, czQ, coplanar);
	}
}

constexpr int facet_vertex_order[] = { 2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0 };

// Find which facet is intersected by the segment (seg) and return next facets to check:
//  in_facets [in] : vector of facets to check
//  out_facets [out] : vector of facets to check at next step (can be in_facets)
//  out_inter [out] : kind of intersection
// return false if no intersection found and the end of the segment was not reached
bool intersect(const delaunay_t& Tr,
	const segment_t& seg,
	const std::vector<facet_t>& in_facets,
	std::vector<facet_t>& out_facets,
	intersection_t& inter) {
	ASSERT(!in_facets.empty());

	int coplanar[3];
	const double prevDist = inter.dist;

	// segment endpoints and direction (hoisted once)
	const point_t& p = seg.source();
	const point_t& q = seg.target();
	const double px = p.x(), py = p.y(), pz = p.z();
	const double qx = q.x(), qy = q.y(), qz = q.z();
	const double sPx = qx - px, sPy = qy - py, sPz = qz - pz; // q - p
	const double sQx = -sPx, sQy = -sPy, sQz = -sPz;    // p - q

	for (const facet_t& inFacet : in_facets) {
		ASSERT(!Tr.is_infinite(inFacet));

		// facet vertices (a=v0, b=v1, c=v2)
		const uint32_t* __restrict m = facetIdx[inFacet.second & 3]; // {i0,i1,i2}

		const point_t& v0 = inFacet.first->vertex(m[0])->point();
		const point_t& v1 = inFacet.first->vertex(m[1])->point();
		const point_t& v2 = inFacet.first->vertex(m[2])->point();

		const double v0x = v0.x(), v0y = v0.y(), v0z = v0.z();
		const double v1x = v1.x(), v1y = v1.y(), v1z = v1.z();
		const double v2x = v2.x(), v2y = v2.y(), v2z = v2.z();

		// edges from v0
		const double bax = v1x - v0x, bay = v1y - v0y, baz = v1z - v0z;
		const double cax = v2x - v0x, cay = v2y - v0y, caz = v2z - v0z;

		// plane normal n = (v1 - v0) x (v2 - v0)
		const double nx = bay * caz - baz * cay;
		const double ny = baz * cax - bax * caz;
		const double nz = bax * cay - bay * cax;

		// p - v0, q - v0
		const double pax = px - v0x, pay = py - v0y, paz = pz - v0z;
		const double qax = qx - v0x, qay = qy - v0y, qaz = qz - v0z;

		// plane classify + edge checks (unchanged logic through your helper)
		const int nbCoplanar = IntersectPlaneAndEdges(
			nx, ny, nz,
			pax, pay, paz,
			qax, qay, qaz,
			px, py, pz,
			v1x, v1y, v1z,
			v2x, v2y, v2z,
			qx, qy, qz,
			sPx, sPy, sPz, sQx, sQy, sQz,
			coplanar);

		if (nbCoplanar >= 0) {
			// skip this cell if the intersection is not in the desired direction
			double vo, vd;
			if (!IntersectsPrecheckFast(inter.ray, v0x, v0y, v0z, nx, ny, nz,
				prevDist, inter.bigger, vo, vd)) {
				continue;
			}

			// distance along ray (one divide only on pass)
			const double interDist = vo / vd;

			// record the hit facet + distance exactly like before
			inter.facet = inFacet;
			inter.dist = interDist;

			switch (nbCoplanar) {
			case 0: {
				// face intersection
				inter.type = intersection_t::FACET;

				// next facets: the three faces in the neighbor cell different than the origin face
				out_facets.clear();
				const cell_handle_t nc(inter.facet.first->neighbor(inter.facet.second));
				ASSERT(!Tr.is_infinite(nc));
				for (int i = 0; i < 4; ++i) {
					if (nc->neighbor(i) != inter.facet.first)
						out_facets.emplace_back(nc, i);
				}
				return true;
			}

			case 1: {
				// coplanar with 1 edge = intersect edge
				const int j = 4 * inter.facet.second;
				const int i1 = j + coplanar[0];

				inter.type = intersection_t::EDGE;
				inter.v1 = inter.facet.first->vertex(facet_vertex_order[i1 + 0]);
				inter.v2 = inter.facet.first->vertex(facet_vertex_order[i1 + 1]);

				// next facets: the two faces in cells opposing this edge
				out_facets.clear();
				const edge_t out_edge(inter.facet.first,
					facet_vertex_order[i1 + 0],
					facet_vertex_order[i1 + 1]);

				typename delaunay_t::Cell_circulator efc(Tr.incident_cells(out_edge));
				typename delaunay_t::Cell_circulator ifc = efc;
				do {
					const cell_handle_t c(ifc);
					if (c == inter.facet.first) continue;

					const facet_t f1(c, c->index(inter.v1));
					if (!Tr.is_infinite(f1)) out_facets.push_back(f1);

					const facet_t f2(c, c->index(inter.v2));
					if (!Tr.is_infinite(f2)) out_facets.push_back(f2);
				} while (++ifc != efc);

				return true;
			}

			case 2: {
				// coplanar with 2 edges = hit a vertex
				const int j = 4 * inter.facet.second;
				const int i1 = j + coplanar[0];
				const int i2 = j + coplanar[1];

				int vi;
				if (facet_vertex_order[i1] == facet_vertex_order[i2] ||
					facet_vertex_order[i1] == facet_vertex_order[i2 + 1]) {
					vi = facet_vertex_order[i1];
				}
				else
					if (facet_vertex_order[i1 + 1] == facet_vertex_order[i2] ||
						facet_vertex_order[i1 + 1] == facet_vertex_order[i2 + 1]) {
						vi = facet_vertex_order[i1 + 1];
					}
					else {
						ASSERT("2 edges intersections without common vertex" == NULL);
						break;
					}

				inter.type = intersection_t::VERTEX;
				inter.v1 = inter.facet.first->vertex(vi);
				ASSERT(!Tr.is_infinite(inter.v1));

				if (inter.v1->point() == seg.target()) {
					// target reached
					out_facets.clear();
					return false;
				}

				// next facets: faces in cells around opposing this common vertex
				out_facets.clear();
				struct cell_back_inserter_t {
					const delaunay_t& Tr;
					const vertex_handle_t v;
					const cell_handle_t current_cell;
					std::vector<facet_t>& out;
					inline cell_back_inserter_t(const delaunay_t& _Tr,
						const intersection_t& inter,
						std::vector<facet_t>& _out)
						: Tr(_Tr), v(inter.v1), current_cell(inter.facet.first), out(_out) {
					}
					inline cell_back_inserter_t& operator*() { return *this; }
					inline cell_back_inserter_t& operator++(int) { return *this; }
					inline void operator=(cell_handle_t c) {
						if (c == current_cell) return;
						const facet_t f(c, c->index(v));
						if (Tr.is_infinite(f)) return;
						out.push_back(f);
					}
				};
				Tr.finite_incident_cells(inter.v1, cell_back_inserter_t(Tr, inter, out_facets));
				return true;
			}
			}

			// coplanar with 3 edges = tangent = impossible?
			// fall through and keep checking other facets
		}
	}

	// no intersection found and not at end of segment (rare)
	out_facets.clear();
	return false;
}
#endif

#endif

#if 0 // JPB Freespace support removed.
// same as above, but simplified only to find face intersection (otherwise terminate);
// terminate if cell containing the segment endpoint is found or if an infinite cell is encountered
bool intersectFace(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	int coplanar[3];
	for (std::vector<facet_t>::const_iterator it=in_facets.cbegin(); it!=in_facets.cend(); ++it) {
		ASSERT(!Tr.is_infinite(*it));
		if (intersect(Tr.triangle(*it), seg, coplanar) == 0) {
			// face intersection
			inter.facet = *it;
			inter.type = intersection_t::FACET;
			// now find next facets to be checked as
			// the three faces in the neighbor cell different than the origin face
			out_facets.clear();
			inter.ncell = inter.facet.first->neighbor(inter.facet.second);
			if (Tr.is_infinite(inter.ncell))
				return false;
			for (int i=0; i<4; ++i)
				if (inter.ncell->neighbor(i) != inter.facet.first)
					out_facets.push_back(facet_t(inter.ncell, i));
			return true;
		}
	}
	out_facets.clear();
	return false;
}
// same as above, but starts from a known vertex and incident cell
inline bool intersectFace(const delaunay_t& Tr, const segment_t& seg, const vertex_handle_t& v, const cell_handle_t& cell, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	if (cell == cell_handle_t())
		return false;
	if (Tr.is_infinite(cell)) {
		inter.ncell = inter.facet.first = cell;
		return true;
	}
	std::vector<facet_t>& in_facets = out_facets;
	ASSERT(in_facets.empty());
	in_facets.push_back(facet_t(cell, cell->index(v)));
	return intersectFace(Tr, seg, in_facets, out_facets, inter);
}

// Given a cell, compute the free-space support for it
edge_cap_t freeSpaceSupport(const delaunay_t& Tr, const std::vector<cell_info_t>& infoCells, const cell_handle_t& cell)
{
	// sum up all 4 incoming weights
	// (corresponding to the 4 facets of the neighbor cells)
	edge_cap_t wf(0);
	for (int i=0; i<4; ++i) {
		const facet_t& mfacet(Tr.mirror_facet(facet_t(cell, i)));
		wf += infoCells[mfacet.first->info()].f[mfacet.second];
	}
	return wf;
}
#endif

// Fetch the triangle formed by the facet vertices,
// making sure the facet orientation is kept (as in CGAL::Triangulation_3::triangle())
// return the vertex handles of the triangle
struct triangle_vhandles_t {
	vertex_handle_t verts[3];
	triangle_vhandles_t() {}
	triangle_vhandles_t(vertex_handle_t _v0, vertex_handle_t _v1, vertex_handle_t _v2)
		#ifdef _SUPPORT_CPP11
		: verts{_v0,_v1,_v2} {}
		#else
		{ verts[0] = _v0; verts[1] = _v1; verts[2] = _v2; }
		#endif
};
inline triangle_vhandles_t getTriangle(cell_handle_t cell, int i)
{
	ASSERT(i >= 0 && i <= 3);
	if ((i&1) == 0)
		return triangle_vhandles_t(
			cell->vertex((i+2)&3),
			cell->vertex((i+1)&3),
			cell->vertex((i+3)&3) );
	return triangle_vhandles_t(
		cell->vertex((i+1)&3),
		cell->vertex((i+2)&3),
		cell->vertex((i+3)&3) );
}

// Compute the angle between the plane containing the given facet and the cell's circumscribed sphere
// return cosines of the angle
#if 1 // Faster

// Helper: map getTriangle(cell,k) verts back to cell indices 0..3.
// idx[0], idx[1], idx[2] correspond to Pa, Pb, Pc (exact order from getTriangle).
__forceinline void facetOrderIndices(const cell_handle_t& cell, int k, int idx[3]) {
	const auto tri = getTriangle(cell, k);
	for (int j = 0; j < 3; ++j) {
		auto vj = tri.verts[j];
		int found = -1;
		// There are only 4, do a tiny linear scan (fast, branchless enough for MSVC).
		if (vj == cell->vertex(0)) found = 0;
		else if (vj == cell->vertex(1)) found = 1;
		else if (vj == cell->vertex(2)) found = 2;
		else /* vj == cell->vertex(3) */ found = 3;
		idx[j] = found;
	}
}
// Compute plane-sphere "angle cosine" for all 4 facets of a tetra at once.
// out[0..3] correspond to facet indices 0..3 (vertex opposite the facet).
// Uses SSE if available; otherwise falls back to scalar.
// Assumes: facet index k triangle is "all vertices except k".
// 4-at-once, sqrt-based, no temp arrays, minimal loads/stores.
// Build with /O2 (and /fp:precise or /fp:fast as you prefer).
// Strict match to scalar: uses getTriangle(cell, k) ordering,
// computes result = dot / sqrt(|N|^2 * |C|^2), clamps to [-1,1],
// and returns 0.5f for degenerates (same as your scalar).
// 2-space indent, braces same line, camelCase variables.
__forceinline void computeOneMinusPlaneSphereAngle4(
	const delaunay_t& Tr, const cell_handle_t& cell,
	_Data vQual, float* __restrict out)
{
	if (Tr.is_infinite(cell)) {
		_mm_store_ps(out, _mm_setzero_ps());
		return;
	}

	// Load once.
	const auto& pp0 = cell->vertex(0)->point();
	const auto& pp1 = cell->vertex(1)->point();
	const auto& pp2 = cell->vertex(2)->point();
	const auto& pp3 = cell->vertex(3)->point();

#if CGAL_VERSION_NR < 1041101000
	const auto cc = cell->circumcenter(Tr.geom_traits());
#else
	const auto cc = Tr.geom_traits().construct_circumcenter_3_object()(pp0, pp1, pp2, pp3);
#endif

	// float SoA in registers (no stack arrays).
	const __m128 vx = _mm_setr_ps((float)pp0.x(), (float)pp1.x(),
		(float)pp2.x(), (float)pp3.x());
	const __m128 vy = _mm_setr_ps((float)pp0.y(), (float)pp1.y(),
		(float)pp2.y(), (float)pp3.y());
	const __m128 vz = _mm_setr_ps((float)pp0.z(), (float)pp1.z(),
		(float)pp2.z(), (float)pp3.z());

	// Pa,Pb,Pc per lane via shuffles (no memory):
	//   k=0:[2,1,3]  k=1:[2,3,0]  k=2:[0,3,1]  k=3:[0,1,2]
	const __m128 ax = _mm_shuffle_ps(vx, vx, _MM_SHUFFLE(0, 0, 2, 2));
	const __m128 ay = _mm_shuffle_ps(vy, vy, _MM_SHUFFLE(0, 0, 2, 2));
	const __m128 az = _mm_shuffle_ps(vz, vz, _MM_SHUFFLE(0, 0, 2, 2));
	const __m128 bx = _mm_shuffle_ps(vx, vx, _MM_SHUFFLE(1, 3, 3, 1));
	const __m128 by = _mm_shuffle_ps(vy, vy, _MM_SHUFFLE(1, 3, 3, 1));
	const __m128 bz = _mm_shuffle_ps(vz, vz, _MM_SHUFFLE(1, 3, 3, 1));
	const __m128 cx = _mm_shuffle_ps(vx, vx, _MM_SHUFFLE(2, 1, 0, 3));
	const __m128 cy = _mm_shuffle_ps(vy, vy, _MM_SHUFFLE(2, 1, 0, 3));
	const __m128 cz = _mm_shuffle_ps(vz, vz, _MM_SHUFFLE(2, 1, 0, 3));

	const __m128 Ax = _mm_sub_ps(bx, ax), Ay = _mm_sub_ps(by, ay), Az = _mm_sub_ps(bz, az);
	const __m128 Bx = _mm_sub_ps(cx, ax), By = _mm_sub_ps(cy, ay), Bz = _mm_sub_ps(cz, az);

	const __m128 Nx = _mm_sub_ps(_mm_mul_ps(Ay, Bz), _mm_mul_ps(Az, By));
	const __m128 Ny = _mm_sub_ps(_mm_mul_ps(Az, Bx), _mm_mul_ps(Ax, Bz));
	const __m128 Nz = _mm_sub_ps(_mm_mul_ps(Ax, By), _mm_mul_ps(Ay, Bx));

	const __m128 Cx = _mm_sub_ps(_mm_set1_ps((float)cc.x()), ax);
	const __m128 Cy = _mm_sub_ps(_mm_set1_ps((float)cc.y()), ay);
	const __m128 Cz = _mm_sub_ps(_mm_set1_ps((float)cc.z()), az);

	const __m128 fnLenSq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(Nx, Nx), _mm_mul_ps(Ny, Ny)), _mm_mul_ps(Nz, Nz));
	const __m128 ctLenSq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(Cx, Cx), _mm_mul_ps(Cy, Cy)), _mm_mul_ps(Cz, Cz));
	const __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(Nx, Cx), _mm_mul_ps(Ny, Cy)), _mm_mul_ps(Nz, Cz));

	const __m128 denom = _mm_mul_ps(fnLenSq, ctLenSq);
	// rsqrt + 1 NR step
	__m128 r = _mm_rsqrt_ps(denom);
	const __m128 half = _mm_set1_ps(0.5f), three = _mm_set1_ps(3.0f);
	r = _mm_mul_ps(_mm_mul_ps(half, r),
		_mm_sub_ps(three, _mm_mul_ps(denom, _mm_mul_ps(r, r))));

	__m128 res = _mm_mul_ps(dot, r);
	const __m128 one = _mm_set1_ps(1.0f);
	res = _mm_min_ps(_mm_max_ps(res, _mm_set1_ps(-1.0f)), one);

	// Degenerate fallback (cold).
	const __m128 mBad = _mm_cmple_ps(denom, _mm_setzero_ps());
	if (_mm_movemask_ps(mBad)) {
		res = _mm_or_ps(_mm_and_ps(mBad, _mm_set1_ps(0.5f)),
			_mm_andnot_ps(mBad, res));
	}

	_mm_store_ps(out, _mm_mul_ps(vQual, _mm_sub_ps(one, res)));
}

#if 0
inline float computePlaneSphereAngle(const delaunay_t& Tr, const facet_t& facet)
{
  if (Tr.is_infinite(facet.first))
    return 1.f;

  // Get triangle vertices
  const triangle_vhandles_t tri = getTriangle(facet.first, facet.second);
  const auto& p0 = tri.verts[0]->point();
  const auto& p1 = tri.verts[1]->point();
  const auto& p2 = tri.verts[2]->point();

  // Convert to Point3f
  const float x0 = (float) p0.x(), y0 = (float) p0.y(), z0 = (float) p0.z();
  const float x1 = (float) p1.x(), y1 = (float) p1.y(), z1 = (float) p1.z();
  const float x2 = (float) p2.x(), y2 = (float) p2.y(), z2 = (float) p2.z();

  // Compute edges
  const float ax = x1 - x0, ay = y1 - y0, az = z1 - z0;
  const float bx = x2 - x0, by = y2 - y0, bz = z2 - z0;

  // Compute normal
  const float nx = ay * bz - az * by;
  const float ny = az * bx - ax * bz;
  const float nz = ax * by - ay * bx;

  const float fnLenSq = nx*nx + ny*ny + nz*nz;
  if (fnLenSq == 0.f)
    return 0.5f;

  // Circumcenter
#if CGAL_VERSION_NR < 1041101000
  const auto cc_pt = facet.first->circumcenter(Tr.geom_traits());
#else
  const auto cc_pt = Tr.geom_traits().construct_circumcenter_3_object()(
    facet.first->vertex(0)->point(),
    facet.first->vertex(1)->point(),
    facet.first->vertex(2)->point(),
    facet.first->vertex(3)->point());
#endif

  const float cx = (float) cc_pt.x() - x0;
  const float cy = (float) cc_pt.y() - y0;
  const float cz = (float) cc_pt.z() - z0;
  const float ctLenSq = cx*cx + cy*cy + cz*cz;
  if (ctLenSq == 0.f)
    return 0.5f;

  // Dot product
  const float dot = nx * cx + ny * cy + nz * cz;

  float denom = fnLenSq * ctLenSq;
  if (denom <= 0.f)
    return 0.5f;

  const float invSqrt = 1.0f / std::sqrt(denom);
  float result = dot * invSqrt;

  // clamp to [-1, 1]
  return result < -1.f ? -1.f : (result > 1.f ? 1.f : result);
}
#endif
#else
float computePlaneSphereAngle(const delaunay_t& Tr, const facet_t& facet)
{
	// compute facet normal
	if (Tr.is_infinite(facet.first))
		return 1.f;
	const triangle_vhandles_t tri(getTriangle(facet.first, facet.second));
	const Point3f v0(CGAL2MVS<float>(tri.verts[0]->point()));
	const Point3f v1(CGAL2MVS<float>(tri.verts[1]->point()));
	const Point3f v2(CGAL2MVS<float>(tri.verts[2]->point()));
	const Point3f fn((v1-v0).cross(v2-v0));
		const float fnLenSq(normSq(fn));
		if (fnLenSq == 0.f)
			return 0.5f;

	// compute the co-tangent to the circumscribed sphere in one of the vertices
	#if CGAL_VERSION_NR < 1041101000
	const Point3f cc(CGAL2MVS<float>(facet.first->circumcenter(Tr.geom_traits())));
	#else
	struct Tools {
		static point_t circumcenter(const delaunay_t& Tr, const facet_t& facet) {
			return Tr.geom_traits().construct_circumcenter_3_object()(
				facet.first->vertex(0)->point(),
				facet.first->vertex(1)->point(),
				facet.first->vertex(2)->point(),
				facet.first->vertex(3)->point()
			);
		}
	};
	const Point3f cc(CGAL2MVS<float>(Tools::circumcenter(Tr, facet)));
	#endif
	const Point3f ct(cc-v0);
	const float ctLenSq(normSq(ct));
	if (ctLenSq == 0.f)
		return 0.5f;

	// compute the angle between the two vectors
	return CLAMP((fn.dot(ct))/SQRT(fnLenSq*ctLenSq), -1.f, 1.f);
}
#endif

// Compute the angle between the plane containing the given facet and the cell's circumscribed sphere
// return cosines of the angle
float computePlaneSphereAngle(const delaunay_t& Tr, const facet_t& facet)
{
	// compute facet normal
	if (Tr.is_infinite(facet.first))
		return 1.f;
	const triangle_vhandles_t tri(getTriangle(facet.first, facet.second));
	const Point3f v0(CGAL2MVS<float>(tri.verts[0]->point()));
	const Point3f v1(CGAL2MVS<float>(tri.verts[1]->point()));
	const Point3f v2(CGAL2MVS<float>(tri.verts[2]->point()));
	const Point3f fn((v1-v0).cross(v2-v0));
		const float fnLenSq(normSq(fn));
		if (fnLenSq == 0.f)
			return 0.5f;

	// compute the co-tangent to the circumscribed sphere in one of the vertices
	#if CGAL_VERSION_NR < 1041101000
	const Point3f cc(CGAL2MVS<float>(facet.first->circumcenter(Tr.geom_traits())));
	#else
	struct Tools {
		static point_t circumcenter(const delaunay_t& Tr, const facet_t& facet) {
			return Tr.geom_traits().construct_circumcenter_3_object()(
				facet.first->vertex(0)->point(),
				facet.first->vertex(1)->point(),
				facet.first->vertex(2)->point(),
				facet.first->vertex(3)->point()
			);
		}
	};
	const Point3f cc(CGAL2MVS<float>(Tools::circumcenter(Tr, facet)));
	#endif
	const Point3f ct(cc-v0);
	const float ctLenSq(normSq(ct));
	if (ctLenSq == 0.f)
		return 0.5f;

	// compute the angle between the two vectors
	return CLAMP((fn.dot(ct))/SQRT(fnLenSq*ctLenSq), -1.f, 1.f);
}

static void BuildGraphNodesAndEdges(
	MaxFlow<cell_size_t, edge_cap_t>& graph,
	cell_handle_t* __restrict allCellHandles,
	const delaunay_t& delaunay,
	const cell_info_t* __restrict infoCells,
	const uint32_t*   __restrict cellNbrID,
	const uint8_t*    __restrict cellNbrSlot,
	size_t totalCells,
	float kQual,
	float maxCap,
	double cpuHz = 0.0)
{
	const bool diag = (cpuHz > 0.0);
	const auto tStart = rdtscEnd();

	// Phase 1: nodes � parallel reduction for flow, direct excess write
	{
		edge_cap_t totalFlow = 0;
#pragma omp parallel for schedule(static) reduction(+:totalFlow)
		for (ptrdiff_t ciID = 0; ciID < (ptrdiff_t)totalCells; ++ciID) {
			const cell_info_t& ciInfo = infoCells[ciID];
			const edge_cap_t s = ciInfo.s;
			const edge_cap_t t = MINF(ciInfo.t, maxCap);
			totalFlow += (s < t ? s : t);
			graph.graph.nodes[ciID].excess = s - t;
		}
		graph.graph.flow = totalFlow;
	}
	const auto tP1 = rdtscEnd();

	// Phase 2: compute quality angles only. Per-cell neighbor IDs +
	// reverse-slot indices (cellNbrID, cellNbrSlot) are now built once at
	// cell-enumeration time and passed in — no cell-pool derefs needed here.
	// qualAngles still local: only this function uses it.
	float* __restrict qualAngles = (float*)_aligned_malloc(sizeof(float) * totalCells * 4, 64);

	{
		const _Data vQual = _mm_set1_ps(kQual);

		// P2: angle math only — vertex pool derefs (4× ci->vertex(k)->point()
		// per cell + circumcenter). No cell-pool derefs.
#pragma omp parallel for schedule(static)
		for (ptrdiff_t idx = 0; idx < (ptrdiff_t)totalCells; ++idx) {
			float*              __restrict dstQ = qualAngles + idx * 4;
			const cell_handle_t            ci   = allCellHandles[idx];

			if (delaunay.is_infinite(ci)) {
				dstQ[0] = dstQ[1] = dstQ[2] = dstQ[3] = 0.0f;
			} else {
				computeOneMinusPlaneSphereAngle4(delaunay, ci, vQual, dstQ);
			}
		}

		if (diag) {
			std::cout << "     [P2 angle math   ] "
			          << rdtscToSeconds(rdtscEnd() - tP1, cpuHz) << "\n";
		}
	}
	const auto tP2 = rdtscEnd();

	// Phase 3: direct arc build. Each cell writes its own 4 outgoing arcs
	// to nodes[idx].arcs[0..3] using neighbor index `i` as the slot. No
	// atomics, no edges[] materialization, no cross-thread aliasing.
	//
	// Slot determinism: arc slot on cell ci = neighbor index i. The reverse
	// slot on cj is cellNbrSlot[ci][i] = j (cj->index(ci)). Each peer pair
	// independently agrees on its slots without coordination.
	//
	// Cap symmetry: q = MINF(qi[i], qj[j]) and q' = MINF(qj[j], qi[i]) — same.
	// Both peers compute identical q. ci writes rCap = ciInfo.f[i]+q;
	// cj writes rCap = cjInfo.f[j]+q. Identical to legacy AddEdge output.
	//
	// Manual load scheduling: gather all 4 neighbor IDs and slot indices
	// first, then issue all 8 random loads back-to-back so the LSU can
	// dispatch them in parallel up to its LFB count. Bounded MSHR/LFB
	// occupancy per iteration; no inter-iteration speculation, no risk of
	// TLB thrash.
	{
		auto* __restrict ibNodes = graph.graph.nodes;

#pragma omp parallel for schedule(static)
		for (ptrdiff_t idx = 0; idx < (ptrdiff_t)totalCells; ++idx) {
			// Stage 1: linear loads — own cell's caches.
			const float*       __restrict ciQ    = qualAngles + idx * 4;
			const uint32_t*    __restrict nbrIds = cellNbrID  + idx * 4;
			const cell_info_t&            ciInfo = infoCells[idx];
			const uint8_t                 slotPk = cellNbrSlot[idx];

#if SCRREC_OPT_PREFETCH
			// (B) Warm next iteration's neighbor lines (~6 random gathers).
			// nbrIds for idx+1 is itself a sequential read, so it's free.
			if (idx + 1 < (ptrdiff_t)totalCells) {
				const uint32_t* __restrict nNbr = cellNbrID + (idx + 1) * 4;
				const uint32_t n0 = nNbr[0], n1 = nNbr[1], n2 = nNbr[2], n3 = nNbr[3];
				_mm_prefetch((const char*)&infoCells[n0],            _MM_HINT_T0);
				_mm_prefetch((const char*)&infoCells[n1],            _MM_HINT_T0);
				_mm_prefetch((const char*)&infoCells[n2],            _MM_HINT_T0);
				_mm_prefetch((const char*)&infoCells[n3],            _MM_HINT_T0);
				_mm_prefetch((const char*)(qualAngles + (size_t)n0 * 4), _MM_HINT_T0);
				_mm_prefetch((const char*)(qualAngles + (size_t)n2 * 4), _MM_HINT_T0);
			}
#endif

			// Stage 2: resolve all 4 neighbor IDs and slot indices up front.
			const uint32_t cj0 = nbrIds[0], cj1 = nbrIds[1];
			const uint32_t cj2 = nbrIds[2], cj3 = nbrIds[3];
			const int j0 = (slotPk     ) & 3;
			const int j1 = (slotPk >> 2) & 3;
			const int j2 = (slotPk >> 4) & 3;
			const int j3 = (slotPk >> 6) & 3;

			// Stage 3: issue all 8 random loads back-to-back.
			const float qN0 = qualAngles[(size_t)cj0 * 4 + j0];
			const float qN1 = qualAngles[(size_t)cj1 * 4 + j1];
			const float qN2 = qualAngles[(size_t)cj2 * 4 + j2];
			const float qN3 = qualAngles[(size_t)cj3 * 4 + j3];
			const float fN0 = infoCells[cj0].f[j0];
			const float fN1 = infoCells[cj1].f[j1];
			const float fN2 = infoCells[cj2].f[j2];
			const float fN3 = infoCells[cj3].f[j3];

			// Stage 4: own-cell f[i] (linear) + arithmetic.
			const float fi0 = ciInfo.f[0], fi1 = ciInfo.f[1];
			const float fi2 = ciInfo.f[2], fi3 = ciInfo.f[3];
			const float qi0 = ciQ[0], qi1 = ciQ[1], qi2 = ciQ[2], qi3 = ciQ[3];

			const float q0 = MINF(qi0, qN0);
			const float q1 = MINF(qi1, qN1);
			const float q2 = MINF(qi2, qN2);
			const float q3 = MINF(qi3, qN3);

			const float fwd0 = fi0 + q0, rev0 = fN0 + q0;
			const float fwd1 = fi1 + q1, rev1 = fN1 + q1;
			const float fwd2 = fi2 + q2, rev2 = fN2 + q2;
			const float fwd3 = fi3 + q3, rev3 = fN3 + q3;

			// Stage 5: write the 4 arcs (linear, same cacheline as node).
			auto& u = ibNodes[idx];
			auto& a0 = u.arcs[0];
			a0.initFields((uint32_t)cj0, (uint8_t)j0);
			a0.rCap = fwd0;
			auto& a1 = u.arcs[1];
			a1.initFields((uint32_t)cj1, (uint8_t)j1);
			a1.rCap = fwd1;
			auto& a2 = u.arcs[2];
			a2.initFields((uint32_t)cj2, (uint8_t)j2);
			a2.rCap = fwd2;
			auto& a3 = u.arcs[3];
			a3.initFields((uint32_t)cj3, (uint8_t)j3);
			a3.rCap = fwd3;
			u.residBits = (uint8_t)(
				((rev0 > 0.0f) << 0) |
				((rev1 > 0.0f) << 1) |
				((rev2 > 0.0f) << 2) |
				((rev3 > 0.0f) << 3));
			u.arcCount = 4;
		}
	}
	_aligned_free(qualAngles);
	const auto tP3 = rdtscEnd();

	if (diag) {
		std::cout << "     [P1 excess     ] " << rdtscToSeconds(tP1 - tStart, cpuHz) << "\n";
		std::cout << "     [P2 quals+nbrs ] " << rdtscToSeconds(tP2 - tP1, cpuHz) << "\n";
		std::cout << "     [P3 arc build  ] " << rdtscToSeconds(tP3 - tP2, cpuHz) << "\n";
	}
}

#ifdef PARALLEL_GRAPH_CUT_EXTRACTION
// Extract the surface mesh from the graph-cut result.
// Parallel pass 1: find boundary facets across all cells.
// Sequential pass 2: deduplicate vertices and build mesh arrays.
static void ExtractGraphCutSurface(
	const delaunay_t& delaunay,
	cell_handle_t* __restrict allCellHandles,
	const MaxFlow<cell_size_t, edge_cap_t>& graph,
	size_t totalCells,
	const Point3f* __restrict idToPoint,
	Mesh& mesh,
	double cpuHz = 0.0)
{
	// Per-face record collected in parallel
	struct RawFace {
		uint32_t vidx[3]; // vert_info_t::idx for each corner
		bool     flip;
	};

	const auto tStart = rdtscEnd();

	// Precompute src-side flag for every cell ID. graph.IsNodeOnSrcSide(id)
	// indirects into the IBFS node array; doing it twice per cell from
	// inside the inner loop misses that array randomly. A single linear
	// pass populates a flat byte array, then Pass 1 reads it sequentially.
	// Memory: ~286 MB for 286M cells; freed before Pass 2.
	uint8_t* __restrict srcSide = (uint8_t*)_aligned_malloc(totalCells, 64);
#pragma omp parallel for schedule(static)
	for (ptrdiff_t i = 0; i < (ptrdiff_t)totalCells; ++i) {
		srcSide[i] = graph.IsNodeOnSrcSide((cell_size_t)i) ? 1u : 0u;
	}

	const auto tSrcSide = rdtscEnd();

	const int nThreads = omp_get_max_threads();
	std::vector<std::vector<RawFace>> threadFaces(nThreads);

	// Pass 1: parallel � identify all boundary facets
#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		auto& localFaces = threadFaces[tid];
		localFaces.reserve(totalCells / (4 * nThreads));

#pragma omp for schedule(static)
		for (ptrdiff_t idx = 0; idx < (ptrdiff_t)totalCells; ++idx) {
			const cell_handle_t ci = allCellHandles[idx];
			const cell_size_t ciID = ci->info();
			const uint8_t ciType = srcSide[ciID];

			for (int i = 0; i < 4; ++i) {
				if (delaunay.is_infinite(ci, i)) continue;
				const cell_handle_t cj = ci->neighbor(i);
				const cell_size_t cjID = cj->info();
				if (ciID < cjID) continue;

				if (ciType == srcSide[cjID]) continue;

				const triangle_vhandles_t tri(getTriangle(ci, i));
				RawFace rf;
				rf.vidx[0] = tri.verts[0]->info().idx;
				rf.vidx[1] = tri.verts[1]->info().idx;
				rf.vidx[2] = tri.verts[2]->info().idx;
				rf.flip = !ciType;
				localFaces.push_back(rf);
			}
		}
	}

	const auto tPass1 = rdtscEnd();

	_aligned_free(srcSide);

	// Count total faces for reservation
	size_t totalFaceCount = 0;
	for (const auto& tf : threadFaces)
		totalFaceCount += tf.size();

	// Pass 2: sequential vertex dedup via flat remap array (vert_info_t::idx is
	// dense in [0, g_idx)). Replaces robin_map<uint32_t,VIndex> -- O(1) lookup,
	// no hashing, no allocations per insert. Sentinel = ~0u for "unseen".
	const uint32_t numIDs = vert_info_t::g_idx;
	constexpr Mesh::VIndex INVALID_VIDX = (Mesh::VIndex)~0u;
	std::vector<Mesh::VIndex> vertRemap(numIDs, INVALID_VIDX);

	const size_t nEstimatedNumVerts = delaunay.number_of_vertices();
	mesh.vertices.Reserve((Mesh::VIndex)nEstimatedNumVerts);
	mesh.faces.Reserve((Mesh::FIndex)totalFaceCount);

	for (const auto& localFaces : threadFaces) {
		for (const RawFace& rf : localFaces) {
			Mesh::Face& face = mesh.faces.AddEmpty();
			for (int v = 0; v < 3; ++v) {
				const uint32_t vi = rf.vidx[v];
				Mesh::VIndex& slot = vertRemap[vi];
				if (slot == INVALID_VIDX) {
					slot = (Mesh::VIndex)mesh.vertices.GetSize();
					const Point3f& pt = idToPoint[vi];
					mesh.vertices.Insert(Mesh::Vertex(pt.x, pt.y, pt.z));
				}
				face[v] = slot;
			}
			if (rf.flip)
				std::swap(face[0], face[2]);
		}
	}

	if (cpuHz > 0.0) {
		const auto tEnd = rdtscEnd();
		std::cout << "     [End srcSide   ] " << rdtscToSeconds(tSrcSide - tStart, cpuHz) << "\n";
		std::cout << "     [End pass1     ] " << rdtscToSeconds(tPass1   - tSrcSide, cpuHz) << "\n";
		std::cout << "     [End pass2     ] " << rdtscToSeconds(tEnd     - tPass1, cpuHz) << "\n";
	}
}
#endif

} // namespace DELAUNAY

#pragma intrinsic(_InterlockedCompareExchange)

static inline float AtomicAddFloat(float* addr, float val)
{
	static_assert(sizeof(float) == sizeof(LONG), "float and LONG must be same size");

	LONG* intAddr = reinterpret_cast<LONG*>(addr);
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
#if SCRREC_OPT_PREFETCH
		_mm_pause(); // (C) yield µop port under contention; collapses retry storms
#endif
		oldInt = prev; // retry with updated oldInt
	}

	return oldVal;
}

static inline float
fasterpow2(float p)
{
	float clipp = (p < -126) ? -126.0f : p;
	union { uint32_t i; float f; } v = { static_cast<uint32_t>((1 << 23) * (clipp + 126.94269504f)) };
	return v.f;
}

static inline float JPBEXP(float p)
{
	return fasterpow2(1.442695040f * p);
}

template <bool UseROI>
size_t ProcessPoints(
	const float* __restrict pPointStream,
	size_t numVertices,
	DELAUNAY::point_t* __restrict origVertices,
	std::ptrdiff_t* __restrict indices,
	const SEACAVE::OBB3f& obb
)
{
	if constexpr (UseROI) {
		size_t validCount = 0;

		for (size_t i = 0, j = 0; i < numVertices; ++i, j += 3) {
			const float x = pPointStream[j];
			const float y = pPointStream[j + 1];
			const float z = pPointStream[j + 2];

			if constexpr (UseROI) {
				const PointCloud::Point X(x, y, z);
				if (!obb.Intersects(X)) continue;
			}

			origVertices[validCount] = DELAUNAY::point_t(x, y, z);
#ifndef VALIDATE
			indices[validCount] = validCount;
#endif
			++validCount;
		}

		return validCount;
	}
	else {
		const int64_t cnt = numVertices;
		// Faster without warmingg.
		// NOTE: More than twice as fast when parallelized.
#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < cnt; ++i)
		{
			const int64_t j = i * 3;
			const float x = pPointStream[j];
			const float y = pPointStream[j + 1];
			const float z = pPointStream[j + 2];

			origVertices[i] = DELAUNAY::point_t(x, y, z);
#ifndef VALIDATE
			indices[i] = (int64_t)i;
#endif
		}
	}

	return numVertices;
}

#include <algorithm>
#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>
#if defined(_MSC_VER) && _MSVC_LANG >= 201703L
#include <execution>
#endif

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_sort.h>

#define PERMUTE_SCATTER_WARMUP

template<class T0, class T1, class T2, class T3>
void permuteScatter2(
	const ptrdiff_t* __restrict orderNewToOld,
	size_t cnt,
	const T0* __restrict src0,
	T0* __restrict dst0,
	T1* __restrict dst0f,
	const std::vector<T2>& src1, T2* __restrict dst1,
	const std::vector<T3>& src2, T3* __restrict dst2)
{
	if (cnt == 0) return;

	// ---- Optional warm-up for large buffers ----
	const size_t totalBytes =
		sizeof(T0) * cnt + sizeof(T1) * cnt + sizeof(T2) * cnt + sizeof(T3) * cnt;

#ifdef PERMUTE_SCATTER_WARMUP
	if (totalBytes > (1ull << 24)) { // ~16 MB threshold
		tbb::parallel_for(tbb::blocked_range<size_t>(0, cnt, 1 << 16),
			[&](auto const& r)
			{
				for (size_t i = r.begin(); i < r.end(); i += 64 / sizeof(T0))
					dst0[i] = T0();
				for (size_t i = r.begin(); i < r.end(); i += 64 / sizeof(T1))
					dst0f[i] = T1();
				for (size_t i = r.begin(); i < r.end(); i += 64 / sizeof(T2))
					dst1[i] = T2();
				for (size_t i = r.begin(); i < r.end(); i += 64 / sizeof(T3))
					dst2[i] = T3();
			});
	}
#endif

	// ---- Main scatter ----
	tbb::parallel_for(tbb::blocked_range<size_t>(0, cnt, 1 << 16),
		[&](auto const& r)
		{
			size_t i = r.begin();
			const size_t end = r.end();

			for (; i + 3 < end; i += 4) {
				const size_t o0 = orderNewToOld[i + 0];
				const size_t o1 = orderNewToOld[i + 1];
				const size_t o2 = orderNewToOld[i + 2];
				const size_t o3 = orderNewToOld[i + 3];

				const T0& p0 = src0[o0];
				const T0& p1 = src0[o1];
				const T0& p2 = src0[o2];
				const T0& p3 = src0[o3];

				dst0[i + 0] = p0;
				dst0[i + 1] = p1;
				dst0[i + 2] = p2;
				dst0[i + 3] = p3;

				dst0f[i + 0] = T1(p0[0], p0[1], p0[2]);
				dst0f[i + 1] = T1(p1[0], p1[1], p1[2]);
				dst0f[i + 2] = T1(p2[0], p2[1], p2[2]);
				dst0f[i + 3] = T1(p3[0], p3[1], p3[2]);

				dst1[i + 0] = src1[o0];
				dst1[i + 1] = src1[o1];
				dst1[i + 2] = src1[o2];
				dst1[i + 3] = src1[o3];

				dst2[i + 0] = src2[o0];
				dst2[i + 1] = src2[o1];
				dst2[i + 2] = src2[o2];
				dst2[i + 3] = src2[o3];
			}

			for (; i < end; ++i) {
				const size_t oldId = orderNewToOld[i];
				const T0& pt = src0[oldId];

				dst0[i] = pt;
				dst0f[i] = T1(pt[0], pt[1], pt[2]);
				dst1[i] = src1[oldId];
				dst2[i] = src2[oldId];
			}
		});
}


// Euclidean squared distance
inline double dist2(const DELAUNAY::point_t& a, const DELAUNAY::point_t& b)
{
	double dx = a.x() - b.x();
	double dy = a.y() - b.y();
	double dz = a.z() - b.z();
	return dx * dx + dy * dy + dz * dz;
}

struct PointCloudAdapter {
	const Point3f* pts;
	size_t n;

	inline size_t kdtree_get_point_count() const noexcept { return n; }

	inline float kdtree_get_pt(const size_t idx, const size_t dim) const noexcept {
		if (dim == 0) return pts[idx].x;
		if (dim == 1) return pts[idx].y;
		return pts[idx].z;
	}

	template<class BBOX>
	bool kdtree_get_bbox(BBOX&) const noexcept { return false; }
};

static void knnMeanDistSq_nanoflann_fast(
	const Point3f* __restrict pts, size_t n, int k, float* __restrict out)
{
	using namespace nanoflann;
	using KDTree = KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<float, PointCloudAdapter>,
		PointCloudAdapter, 3 > ;

	if (n == 0) return;
	// No need to warm up output data.

	PointCloudAdapter cloud{ pts, n };
	KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(64));
	index.buildIndex();

	struct alignas(64) ThreadBuffers {
		uint32_t idx[64];
		float    d2[64];
	};
	const int T = omp_get_max_threads();
	std::vector<ThreadBuffers> tbuf(T);

#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		ThreadBuffers& buf = tbuf[tid];
		auto* idx = buf.idx;
		auto* d2 = buf.d2;

#pragma omp for schedule(static,1024)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i)
		{
			const size_t found = index.knnSearch((float*)&pts[i], k + 1, idx, d2);
			if (found <= 1) { out[i] = 0.0f; continue; }

			// Sum squared distances (no sqrt in loop)
			const size_t end = found & ~7ULL;
			__m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
			size_t j = 1;
			for (; j < end; j += 8)
			{
				__m128 v0 = _mm_loadu_ps(&d2[j]);
				__m128 v1 = _mm_loadu_ps(&d2[j + 4]);
				acc0 = _mm_add_ps(acc0, v0);
				acc1 = _mm_add_ps(acc1, v1);
			}
			acc0 = _mm_add_ps(acc0, acc1);
			__m128 tmp = _mm_movehl_ps(acc0, acc0);
			acc0 = _mm_add_ps(acc0, tmp);
			tmp = _mm_shuffle_ps(acc0, acc0, 1);
			acc0 = _mm_add_ss(acc0, tmp);
			float sum = _mm_cvtss_f32(acc0);

			for (; j < found; ++j)
				sum += d2[j];

			// ONE sqrt per point: sqrt of mean squared distance = RMS distance
			out[i] = FastSqrtS(sum / float(found - 1));
		}
	}
}

void StatisticalOutlierRemoval(
	const Point3f* __restrict pts, size_t n,
	unsigned char* __restrict mask,
	int k = 16, float stddevMul = 2.0f,
	float interiorMul = 0.5f) // lower side factor
{
	TD_TIMER_START();
	if (n == 0) return;
	// allocate as float to cut memory traffic in half
	float* meanDist = static_cast<float*>(_aligned_malloc(sizeof(float) * n, 64));
	knnMeanDistSq_nanoflann_fast(pts, n, k, meanDist);
	// --- Welford�s one-pass algorithm for mean and variance -----------------
	float mean = 0.0f, m2 = 0.0f;
	size_t count = 0;
#pragma omp parallel
	{
		float local_mean = 0.0f, local_m2 = 0.0f;
		size_t local_n = 0;
#pragma omp for nowait
		for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i) {
			++local_n;
			float delta = meanDist[i] - local_mean;
			local_mean += delta / local_n;
			local_m2 += delta * (meanDist[i] - local_mean);
		}
#pragma omp critical
		{
			float delta = local_mean - mean;
			size_t new_n = count + local_n;
			mean += delta * (float)local_n / (float)new_n;
			m2 += local_m2 + delta * delta * (float)count * (float)local_n / (float)new_n;
			count = new_n;
		}
	}
	const float var = m2 / (float)n;
	const float stdev = FastSqrtS(var);
	// High outliers (sparse points)
	const float upperThr = mean + stddevMul * stdev;
	// Low outliers (interior / over-dense points)
	const float lowerThr = mean - interiorMul * stdev;
	// Clamp lowerThr to positive (just in case)
	const float lowerBound = std::max(lowerThr, 1e-6f);
	// --- Apply both filters in one pass ------------------------------------
	int removed = 0;
#pragma omp parallel for reduction(+:removed) schedule(static)
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i) {
		const float d = meanDist[i];
		// keep if within thresholds
		const bool keep = (d >= lowerBound && d <= upperThr);
		mask[i] = keep ? 1 : 0;
		removed += !keep;
	}
	_aligned_free(meanDist);
	MESH_DIAG("%d filtered (k=%d, stddevMul=%.2f, interiorMul=%.2f, mean=%.4f, stdev=%.4f)",
		removed, k, stddevMul, interiorMul, mean, stdev);
}

template <typename Container>
inline void insertion_sort_small(Container& c)
{
	auto& a = c;                 // alias for brevity
	const size_t n = a.size();
	for (size_t i = 1; i < n; ++i)
	{
		auto key = a[i];
		size_t j = i;
		// move larger elements up one slot
		while (j > 0 && a[j - 1] > key)
		{
			a[j] = a[j - 1];
			--j;
		}
		a[j] = key;
	}
}

static __forceinline uint64_t hashEdge(uint32_t a, uint32_t b)
{
	uint64_t x = (uint64_t(a) << 32) | uint64_t(b);
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdULL;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ULL;
	x ^= x >> 33;
	return x;
}

static __forceinline bool is_canonical_edge(const DELAUNAY::cell_handle_t c, int i, int j)
{
	const int k = 6 - i - j;           // opposite vertex index
	const DELAUNAY::cell_handle_t n = c->neighbor(k);
	return c->info() < n->info();
}

// Load a binary-little-endian triangle-mesh PLY (as written by PoissonRecon /
// SurfaceTrimmer) directly into an MVS::Mesh. Unlike Mesh::LoadPLY this is robust
// to EXTRA per-vertex properties — PoissonRecon emits a 'value' (density) float
// that the fixed-layout Mesh::LoadPLY mis-reads (misaligning the binary stream).
// We keep x/y/z and the triangle indices and skip everything else.
static bool LoadPoissonMeshPLY(const String& path, Mesh& mesh)
{
	std::ifstream f(path.c_str(), std::ios::binary);
	if (!f.is_open())
		return false;
	const auto GetLine = [&](std::string& s) -> bool {
		if (!std::getline(f, s)) return false;
		if (!s.empty() && s.back() == '\r') s.pop_back();
		return true;
	};
	const auto TypeSize = [](const std::string& t) -> int {
		if (t=="char"||t=="uchar"||t=="int8"||t=="uint8") return 1;
		if (t=="short"||t=="ushort"||t=="int16"||t=="uint16") return 2;
		if (t=="int"||t=="uint"||t=="int32"||t=="uint32"||t=="float"||t=="float32") return 4;
		if (t=="double"||t=="float64"||t=="int64"||t=="uint64") return 8;
		return 4;
	};
	std::string line;
	if (!GetLine(line) || line.compare(0, 3, "ply") != 0)
		return false;
	bool binary = false, little = true;
	size_t numVerts = 0, numFaces = 0;
	enum { NONE, VERT, FACE } cur = NONE;
	struct Prop { int size; int axis; }; // axis: 0=x,1=y,2=z,-1=other
	std::vector<Prop> vprops;
	int faceCountSize = 4, faceIdxSize = 4;
	while (GetLine(line)) {
		std::istringstream ss(line);
		std::string tok; ss >> tok;
		if (tok == "format") {
			std::string fmt; ss >> fmt;
			binary = (fmt.find("binary") != std::string::npos);
			little = (fmt != "binary_big_endian");
		} else if (tok == "element") {
			std::string name; size_t cnt = 0; ss >> name >> cnt;
			if (name == "vertex") { cur = VERT; numVerts = cnt; }
			else if (name == "face") { cur = FACE; numFaces = cnt; }
			else cur = NONE;
		} else if (tok == "property") {
			if (cur == VERT) {
				std::string type, name; ss >> type >> name;
				Prop p; p.size = TypeSize(type);
				p.axis = (name=="x") ? 0 : (name=="y") ? 1 : (name=="z") ? 2 : -1;
				vprops.push_back(p);
			} else if (cur == FACE) {
				std::string kind; ss >> kind;
				if (kind == "list") {
					std::string ct, it, nm; ss >> ct >> it >> nm;
					faceCountSize = TypeSize(ct);
					faceIdxSize = TypeSize(it);
				}
			}
		} else if (tok == "end_header") {
			break;
		}
	}
	if (!binary || !little || numVerts == 0 || numFaces == 0)
		return false;
	int stride = 0, ox = -1, oy = -1, oz = -1;
	for (const Prop& p : vprops) {
		if (p.axis == 0) ox = stride; else if (p.axis == 1) oy = stride; else if (p.axis == 2) oz = stride;
		stride += p.size;
	}
	if (ox < 0 || oy < 0 || oz < 0)
		return false;
	mesh.vertices.Resize((Mesh::VIndex)numVerts);
	{
		// Bulk-read the entire fixed-stride vertex block in one I/O, then extract
		// x/y/z in parallel by direct index. Direct indexing preserves file order
		// (face indices reference vertices positionally), unlike append/Insert.
		std::vector<char> vblock((size_t)stride * numVerts);
		f.read(vblock.data(), (std::streamsize)vblock.size());
		if (!f) return false;
		const char* __restrict src = vblock.data();
		Mesh::Vertex* __restrict dst = mesh.vertices.GetData();
		const int sox = ox, soy = oy, soz = oz, sstride = stride;
#ifdef _USE_OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (ptrdiff_t i = 0; i < (ptrdiff_t)numVerts; ++i) {
			const char* __restrict r = src + (size_t)i*sstride;
			float x, y, z;
			memcpy(&x, r+sox, 4); memcpy(&y, r+soy, 4); memcpy(&z, r+soz, 4);
			dst[i] = Mesh::Vertex(x, y, z);
		}
	}
	// Bulk-read the remaining (face) section in one I/O, then parse from memory.
	// Faces are variable-stride lists so the parse stays serial, but it walks an
	// in-RAM buffer instead of issuing a stream read per face/index.
	{
		std::vector<char> fblock((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		const char* __restrict fp = fblock.data();
		const char* const fend = fp + fblock.size();
		mesh.faces.Reserve((Mesh::FIndex)numFaces);
		for (size_t i = 0; i < numFaces; ++i) {
			if (fp + faceCountSize > fend) break;
			uint32_t n = 0;
			memcpy(&n, fp, faceCountSize); fp += faceCountSize;
			if (n != 3) {
				fp += (size_t)faceIdxSize * n;
				if (fp > fend) break;
				continue;
			}
			if (fp + (size_t)faceIdxSize * 3 > fend) break;
			uint32_t idx[3] = { 0, 0, 0 };
			memcpy(&idx[0], fp, faceIdxSize); fp += faceIdxSize;
			memcpy(&idx[1], fp, faceIdxSize); fp += faceIdxSize;
			memcpy(&idx[2], fp, faceIdxSize); fp += faceIdxSize;
			mesh.faces.Insert(Mesh::Face(idx[0], idx[1], idx[2]));
		}
	}
	return !mesh.faces.IsEmpty();
} // LoadPoissonMeshPLY
/*----------------------------------------------------------------*/

// Voxel edge for the Poisson INPUT subsample, expressed as a right-shift of the
// finest octree cell: voxel = cell / 2^SHIFT. Powers of two are used so the voxel
// grid lines up with the octree (and with a Morton ordering built over the same
// box), which keeps same-voxel points contiguous.
//   1 = cell/2  (~4 pts per finest cell)  most aggressive
//   2 = cell/4  (~16 pts per cell)        DEFAULT -- ample for samplesPerNode 1.5
//   3 = cell/8  (~64 pts per cell)        conservative
#ifndef POISSON_INPUT_SUBSAMPLE_SHIFT
#define POISSON_INPUT_SUBSAMPLE_SHIFT 2
#endif
// MEASURED AND REJECTED -- default OFF. Set to 1 only to re-test.
//
// The idea was that PoissonRecon merges its input into octree samples, so points
// beyond a few per finest cell only cost tree-construction memory. The measurement
// on a 263M-point / depth-12 aerial scene says otherwise:
//
//                        full cloud      subsampled (voxel = cell/4)
//   input points         262,967,556     190,393,190   (1.4x)
//   octree nodes          50,436,585      50,408,257   (-0.06%)
//   tree-read current       7,077 MB       11,432 MB
//   tree build time            17.3 s          13.0 s
//
// 7077 + 4357 (the subsample array) = 11434, i.e. the tree used IDENTICAL memory
// both times and the whole delta was the copy. At a fixed depth PoissonRecon's
// footprint is bounded by the OCTREE STRUCTURE, not by input point count, so
// thinning the input buys wall time (-25%) and costs memory (+4.4 GB). Bad trade.
//
// Secondary finding: the contiguous-run filter only reached 1.4x where a 6.5 cm
// voxel over 2.36 cm spacing predicts ~7.6x, so same-voxel points are NOT
// contiguous at that scale -- the cloud's Morton granularity is coarser than the
// voxel. A hash-set dedup would recover the full ratio, but per the above it would
// not help memory anyway, so it was not written.
//
// The real peak on this stage is the dense cloud itself: ~75 B/point at load, of
// which ~51 B/point (colors/views/weights) Poisson never reads. Attack that
// instead -- it needs a scene-load flag, not a filter here.
#ifndef POISSON_INPUT_SUBSAMPLE
#define POISSON_INPUT_SUBSAMPLE 0
#endif

// Run a coarse reconstruction first, purely to measure the face-density
// coefficient k, then re-decide the depth with the measured value. This is what
// makes the depth fully automatic: k is terrain roughness and cannot be derived
// from the cloud (see the probe block in ReconstructMeshPoisson).
//
// COST: one extra solve at target-2 depth. The per-point phases (tree read, kernel
// density, normal field) dominate and are depth-independent, so expect roughly
// +25-40% on this stage. Set to 0 to skip it and fall back to the assumed k, which
// is deliberately conservative (a high k picks a shallower depth).
#ifndef POISSON_CALIBRATION_PROBE
#define POISSON_CALIBRATION_PROBE 1
#endif

// sqrt(RefineMesh --max-face-area). A face is subdivided when it projects larger
// than max-face-area (default 32 px), so the finest geometry refinement can
// introduce is sqrt(32) ~ 5.66 working pixels across. Used to decide whether
// RefineMesh can add detail to a given mesh at all.
// Decay of ln(k) per octree depth level, used to extrapolate the calibration probe
// forward to the target depth. 0 disables the correction (probe value used raw,
// which over-predicts k and costs a depth level). See the probe block for the four
// measurements this is fitted to.
#ifndef POISSON_K_DEPTH_DECAY
#define POISSON_K_DEPTH_DECAY 0.10
#endif
// Run a SECOND, one-level-shallower probe to measure k's actual decay slope and report
// it ([MESH-SLOPE]). Never changes a decision.
//
// DEFAULT 0 -- the experiment ran and the answer was no. MEASURED on the 1940-view
// aerial set: k(8)=1.828, k(9)=1.376 gives lambda=0.284/level, which predicts k(11)=0.779
// against a true 1.079 -- off by -27.8%, where the flat POISSON_K_DEPTH_DECAY constant
// managed +4.4%. Six times worse AND in the unsafe direction (under-predicting k allows a
// deeper solve than the budget intended).
//
// Root cause: lambda is not a constant to be measured, it decays with depth as k
// approaches its asymptote -- 0.284, 0.200, 0.105, 0.081 at depths 8->13. What governs
// the answer is lambda NEAR THE TARGET depth, and a cheap shallow probe cannot see it.
// The 0.10 constant works on this site precisely because it encodes the near-target value
// (10->12 measures 0.105). Probing near the target would be accurate but costs more than
// the solve.
//
// Also measured: the second probe is NOT cheap. Both probes took 38 s against ~21 s for
// one, so the shallower probe cost ~17 s, not the ~5 s a 1/4-work estimate predicts --
// PoissonRecon's fixed overhead (reading 263M points, building the base octree) dominates
// and does not scale with depth.
//
// Set to 1 to re-collect on a new dataset; the diagnostic is still correct and honest.
#ifndef POISSON_K_SLOPE_PROBE
#define POISSON_K_SLOPE_PROBE 0
#endif

// Relative error in the probe's predicted k that warrants a warning. A depth level is
// 4x in faces and therefore 4x in k, so errors well under 100% cannot move the chosen
// depth except right at a boundary. Observed with the decay correction applied: 0.6%,
// 2.5%, ~1%. 25% leaves ample margin while still catching a genuinely broken probe.
#ifndef POISSON_PROBE_ERROR_WARN_PCT
#define POISSON_PROBE_ERROR_WARN_PCT 25.0
#endif

// How much finer than the texture-imposed output cap the Poisson solve may go.
// Adaptive decimation needs SOME surplus to redistribute detail with, but heavy ratios
// erode the features the fine solve captured (measured: 2.8x clean, 6x rounds roof
// ridges and kerbs, 10.3x visibly damaging). 3 sits inside the safe range and, because
// faces scale 4x per depth level, effectively means "at most one level finer".
#ifndef POISSON_TEXTURE_OVERSOLVE
#define POISSON_TEXTURE_OVERSOLVE 3.0
#endif

// Subdivision headroom to leave RefineMesh, as a multiple of the reliability floor.
//
// This is the ceiling that matters most, and the one that was missing.
//
// A face must project to at least sqrt(--max-face-area) ~ 5.66 pixels for TextureMesh's
// per-face view selection to be well determined; below that, adjacent faces pick
// different views and the result SPECKLES (observed at 3.7 px/face). RefineMesh
// subdivides down TOWARD that floor, adding geometry only where the imagery justifies
// it -- so the Poisson mesh should start a factor of 2 above it, leaving exactly one
// level of adaptive subdivision.
//
// Building finer than this in Poisson is doubly wrong: the extra geometry cannot be
// textured reliably, AND it consumes the headroom RefineMesh needs, disabling the very
// stage designed to add detail. MEASURED: depth 12 on a 142 m scene gave 3.82 cm cells
// (3.4 px/face), 35M faces, 27 minutes, speckled texture and RefineMesh skipped; depth
// 10 gives 15.3 cm cells (13.7 px/face), 1.7M faces, 4 minutes, and RefineMesh runs.
//
// 2.0 reproduces the previously hand-tuned depth on both a 142 m and a 974 m site.
#ifndef POISSON_REFINE_HEADROOM
#define POISSON_REFINE_HEADROOM 2.0
#endif
// Headroom used when RefineMesh WILL run: two levels instead of one, so the coarse
// Poisson mesh is deliberately left for RefineMesh to subdivide adaptively. Only
// applies when refinement is both affordable and has signal -- otherwise nothing would
// add the detail back and POISSON_REFINE_HEADROOM (one level) is used instead.
//
// CALIBRATED against two scenes whose hand-tuned depth was known good:
//   142 m / 109 views,  GSD 1.116 cm -> depth 10 requires headroom in (1.21, 2.42]
//   276 m / 368 views,  GSD 1.191 cm -> depth 10 requires headroom in (2.20, 4.41]
// The overlap is (2.20, 2.42], so 2.3. Above it the 142 m scene drops to depth 9
// (coarser than known-good); below it the 276 m scene rises to depth 11, which
// measured 5x the faces, pushed atlas padding from ~1.3x to ~1.75x, cost 5 extra
// minutes, and produced a visually indistinguishable render.
#ifndef POISSON_REFINE_HEADROOM_ADAPTIVE
#define POISSON_REFINE_HEADROOM_ADAPTIVE 2.3
#endif

// POISSON_DEPTH_PIXELS_TOL: rounding tolerance, in log2 units, on the PIXEL ceiling only.
//
// WHY A SCALAR HEADROOM CANNOT BE CALIBRATED. depthPixels is
//     log2(scaleFactor * extent / (headroom * SUBDIV_FACTOR * gsd))
// floored to an integer. The fractional part therefore MOVES WITH log2(extent), so no single
// headroom holds it on the same side of an integer across scenes of different size. Measured
// on the three scenes we have numbers for, all of which want depth 10:
//     132.2 m, gsd 1.114 cm  ->  9.970  @H=2.3 -> 9  WRONG    | @H=2.0 -> 10.172 -> 10 ok
//     142   m, gsd 1.116 cm  -> 10.070  @H=2.3 -> 10 ok       | @H=2.0 -> 10.272 -> 10 ok
//     276   m, gsd 1.191 cm  -> 10.935  @H=2.3 -> 10 ok       | @H=2.0 -> 11.137 -> 11 WRONG
// H=2.3 fails the first, H=2.0 fails the third. The two-scene calibration that produced 2.3
// (see above) landed it at the very top of the 142 m scene's window, and a 132 m scene of the
// same camera class falls just outside -- 0.030 in log2, i.e. a cell 2.1% coarser than depth
// 10 needs. floor() then drops a whole level: cell 0.2841 instead of 0.1420, and ~4x fewer
// faces (measured: 802k raw at depth 9 where depth 10 gives ~3.2M).
//
// THE ASYMMETRY THAT JUSTIFIES A TOLERANCE. This ceiling is a SOFT quality target assembled
// from two estimates (gsd and extent) whose own error this file puts at 0.6-2%. Overshooting
// it by a few percent costs a few percent of per-face pixels; undershooting by one level
// costs 100% of the cell size and 4x the geometry. So the two directions are not
// symmetric and should not be treated as if a hard floor were the safe choice.
//
// 0.05 (3.5% in cell size, inside the acknowledged estimate error) puts all three scenes on
// depth 10:  9.970 -> 10.020,  10.070 -> 10.120,  10.935 -> 10.985. Margin to the next
// boundary is ~0.02 on both sides, so re-check this if a fourth scene lands near a boundary
// -- the [MESH-POLICY] line now prints the unfloored value so that is a log read, not a
// rebuild.
//
// DELIBERATELY NOT APPLIED to depthBudget or depthTexture. Those are HARD ceilings (memory
// envelope, atlas capacity) where floor is correct and the existing comment says so: "faces
// scale 4x per level, so rounding up can overshoot the budget by nearly 2x". Only the soft
// pixel target gets the tolerance. 0 restores the strict floor.
#ifndef POISSON_DEPTH_PIXELS_TOL
#define POISSON_DEPTH_PIXELS_TOL 0.05
#endif

#ifndef POISSON_REFINE_SUBDIV_FACTOR
#define POISSON_REFINE_SUBDIV_FACTOR 5.66
#endif
// Resident bytes per source pixel per view inside RefineMesh's view streamer.
// MEASURED: a 1940-view / 15.19 MPixel run at resolution-level 0 reached 120 GB
// commit with 1239 views resident -- ~97 MB/view, i.e. ~6.4 B/pixel.
//
// That 6.4 is TIER 1 ONLY. RefineMesh's own kStreamBytesPerPixel (SceneRefine.cpp)
// is 15 B/px: tier 1 is the u16 gray plane + two int16 gradient planes = 6 B/px, and
// tier 2 is depthMap 4 + faceMap 4 + isValid 1 = 9 B/px. Tier 2 is independently
// evictable and was mostly NOT resident during the measurement, which is why the
// commit figure lands on tier 1 almost exactly (6.38 measured vs 6.00).
//
// So use 6.4 here -- this constant answers "what will a resident view typically
// cost", which is the right question for choosing a resolution level. Do NOT
// propagate it into SceneRefine's own budget: that one must stay at the worst case
// (15), because under-charging there risks an OOM abort rather than a slow run.
// An earlier revision of this comment claimed the streamer assumed 33.5 MB/view
// (2.2 B/px) and was "2.9x low"; no such constant exists, and the sign was
// backwards -- at 15 vs a typical 6 the streamer OVER-charges and over-batches.
#ifndef POISSON_REFINE_BYTES_PER_PIXEL
#define POISSON_REFINE_BYTES_PER_PIXEL 6.4
#endif
// Below this many input points, skip the subsample entirely -- small clouds are
// not the problem and every point matters to the normal field.
#ifndef POISSON_INPUT_SUBSAMPLE_MIN_POINTS
#define POISSON_INPUT_SUBSAMPLE_MIN_POINTS 2000000
#endif
// Never hand PoissonRecon fewer than this many points, whatever the voxel says.
#ifndef POISSON_INPUT_SUBSAMPLE_FLOOR
#define POISSON_INPUT_SUBSAMPLE_FLOOR 100000
#endif

// Finest PoissonRecon cell at a given depth: one bounding-box pass, no kd-tree.
// Used when the caller pinned the depth, so the input subsample below can still
// be sized from the mesh resolution rather than a constant.
static float PoissonCellSizeAtDepth(const float* ptsRaw, size_t numPoints,
	float scaleFactor, int depth)
{
	if (numPoints == 0 || depth <= 0 || depth > 30)
		return 0.f;
	const Point3f* __restrict pts = reinterpret_cast<const Point3f*>(ptsRaw);
	float minx = FLT_MAX, miny = FLT_MAX, minz = FLT_MAX;
	float maxx = -FLT_MAX, maxy = -FLT_MAX, maxz = -FLT_MAX;
	for (size_t i = 0; i < numPoints; ++i) {
		const Point3f& p = pts[i];
		if (p.x < minx) minx = p.x;  if (p.x > maxx) maxx = p.x;
		if (p.y < miny) miny = p.y;  if (p.y > maxy) maxy = p.y;
		if (p.z < minz) minz = p.z;  if (p.z > maxz) maxz = p.z;
	}
	const float ext = std::max(std::max(maxx - minx, maxy - miny), maxz - minz);
	return (ext > 0.f) ? scaleFactor * ext / (float)(1u << depth) : 0.f;
} // PoissonCellSizeAtDepth
/*----------------------------------------------------------------*/

// Voxel-subsample the oriented points handed to PoissonRecon.
//
// WHY: PoissonRecon merges the input into octree samples, so points beyond a few
// per finest cell buy nothing but tree-construction memory. A 263M-point / depth-12
// case produced 7.9M samples -- 97% of the input was merged away, after paying
// ~23 GB of transient peak to load it. The mesh is unchanged; only the bill isn't.
//
// HOW: `voxel` is derived from the octree cell, so this scales with the dataset
// instead of hardcoding a rate -- a small object scan gets a small voxel and keeps
// nearly everything, an aerial site gets a large one. Emission is one forward pass
// that keeps the FIRST point whenever the quantised cell changes from the previous
// kept one. That relies on same-voxel points being CONTIGUOUS, which holds for a
// spatially sorted (e.g. Morton-ordered) cloud.
//
// GENERALITY: if the cloud is NOT spatially sorted, consecutive points land in
// different voxels and almost everything is kept -- the function degrades to "no
// reduction", never to a wrong or biased subsample. The caller checks the achieved
// reduction and falls back to the full cloud, so an unsorted input costs one cheap
// pass and nothing else. Deterministic in all cases (first point wins, input order).
//
// First-point rather than centroid-averaging is deliberate: averaging would pull
// samples off the measured surface and could bridge thin structures whose two sides
// share a voxel (walls, railings) -- a quality regression that would only show up
// on non-aerial data. PoissonRecon does its own averaging downstream anyway.
//
// Returns the number of points written into outPts/outNrm.
static size_t VoxelSubsampleOrientedPoints(
	const float* __restrict ptsRaw, const float* __restrict nrmRaw, size_t numPoints,
	float voxel, std::vector<float>& outPts, std::vector<float>& outNrm)
{
	if (numPoints == 0 || !(voxel > 0.f))
		return 0;
	// Quantise RELATIVE to the bounding-box minimum, not to the absolute origin.
	// A cloud carried in projected coordinates (UTM easting ~5e5) has only ~1
	// fractional digit left in float32, so absolute quantisation at a centimetre
	// voxel would be pure noise and this filter would silently keep everything or
	// collapse unrelated points. Offsetting first keeps the arithmetic in the
	// scene's own extent, where float32 has millimetre headroom. One extra pass.
	float ox = FLT_MAX, oy = FLT_MAX, oz = FLT_MAX;
	for (size_t i = 0; i < numPoints; ++i) {
		const float* __restrict p = ptsRaw + i * 3;
		if (p[0] < ox) ox = p[0];
		if (p[1] < oy) oy = p[1];
		if (p[2] < oz) oz = p[2];
	}
	const float inv = 1.f / voxel;
	// Count first, then fill exactly. A guessed reserve that undershoots forces a
	// reallocation holding old+new simultaneously -- 1-2 GB of transient waste on a
	// 263M-point cloud, which is precisely what this function exists to avoid. The
	// extra pass is a sequential scan of memory already being streamed.
	const auto CellOf = [&](size_t i, int64_t& cx, int64_t& cy, int64_t& cz) {
		const float* __restrict p = ptsRaw + i * 3;
		cx = (int64_t)std::floor((p[0] - ox) * inv);
		cy = (int64_t)std::floor((p[1] - oy) * inv);
		cz = (int64_t)std::floor((p[2] - oz) * inv);
	};
	size_t kept = 0;
	{
		int64_t lx = INT64_MIN, ly = INT64_MIN, lz = INT64_MIN, cx, cy, cz;
		for (size_t i = 0; i < numPoints; ++i) {
			CellOf(i, cx, cy, cz);
			if (cx == lx && cy == ly && cz == lz)
				continue;
			lx = cx; ly = cy; lz = cz;
			++kept;
		}
	}
	if (kept == 0)
		return 0;

	outPts.clear(); outNrm.clear();
	outPts.resize(kept * 3);
	outNrm.resize(kept * 3);
	float* __restrict oP = outPts.data();
	float* __restrict oN = outNrm.data();
	{
		int64_t lx = INT64_MIN, ly = INT64_MIN, lz = INT64_MIN, cx, cy, cz;
		size_t w = 0;
		for (size_t i = 0; i < numPoints; ++i) {
			CellOf(i, cx, cy, cz);
			if (cx == lx && cy == ly && cz == lz)
				continue;
			lx = cx; ly = cy; lz = cz;
			const float* __restrict p = ptsRaw + i * 3;
			const float* __restrict n = nrmRaw + i * 3;
			oP[w*3+0]=p[0]; oP[w*3+1]=p[1]; oP[w*3+2]=p[2];
			oN[w*3+0]=n[0]; oN[w*3+1]=n[1]; oN[w*3+2]=n[2];
			++w;
		}
		ASSERT(w == kept);
	}
	return kept;
} // VoxelSubsampleOrientedPoints
/*----------------------------------------------------------------*/

// Median full-resolution ground sample distance, in cloud units: for a stride-
// sampled subset of points, the distance to the nearest camera centre divided by
// that camera's focal length in pixels. Deliberately needs neither per-point
// visibility nor Image::avgDepth (neither is reliably populated on a scene loaded
// straight from disk), so it works on any scene that has poses. Returns 0 when it
// cannot be measured, which callers must treat as "unknown" rather than "zero".
static float EstimateSceneGSD(const ImageArr& images, const float* ptsRaw, size_t numPoints)
{
	if (numPoints == 0 || images.IsEmpty())
		return 0.f;

	// Collect valid camera centres and 1/f^2 once. Keeping the reciprocal squared
	// lets the sample loop stay entirely in squared units -- see below.
	std::vector<Point3f> centers;
	std::vector<float> invFocals2;
	centers.reserve((size_t)images.GetSize());
	invFocals2.reserve((size_t)images.GetSize());
	for (size_t i = 0; i < (size_t)images.GetSize(); ++i) {
		const Image& img = images[(IIndex)i];
		if (!img.IsValid())
			continue;
		const float f = (float)img.camera.K(0, 0);
		if (!(f > 0.f))
			continue;
		centers.emplace_back((float)img.camera.C.x, (float)img.camera.C.y, (float)img.camera.C.z);
		invFocals2.push_back(1.f / (f*f));
	}
	if (centers.empty())
		return 0.f;

	const Point3f* __restrict pts = reinterpret_cast<const Point3f*>(ptsRaw);
	const size_t kSamples = std::min<size_t>(numPoints, 20000);
	const size_t stride = std::max<size_t>(1, numPoints / kSamples);
	const size_t nq = (numPoints + stride - 1) / stride;
	// Squared GSD per sample, (d/f)^2. sqrt() is monotone on non-negatives, so the
	// median of the squares sits at the same element as the median of the values --
	// one sqrt at the end instead of one per sample.
	std::vector<float> gsd2(nq, -1.f);
	float* __restrict pG = gsd2.data();
	const size_t nc = centers.size();
	const Point3f* __restrict pC = centers.data();
	const float* __restrict pIF2 = invFocals2.data();
#ifdef _USE_OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (ptrdiff_t q = 0; q < (ptrdiff_t)nq; ++q) {
		const Point3f& p = pts[(size_t)q * stride];
		float best = FLT_MAX; size_t bestI = 0;
		for (size_t c = 0; c < nc; ++c) {
			const float dx = p.x - pC[c].x, dy = p.y - pC[c].y, dz = p.z - pC[c].z;
			const float d2 = dx*dx + dy*dy + dz*dz;
			if (d2 < best) { best = d2; bestI = c; }
		}
		if (best > 0.f && best < FLT_MAX)
			pG[q] = best * pIF2[bestI];
	}
	std::vector<float> valid;
	valid.reserve(nq);
	for (size_t q = 0; q < nq; ++q)
		if (pG[q] > 0.f) valid.push_back(pG[q]);
	if (valid.empty())
		return 0.f;
	std::nth_element(valid.begin(), valid.begin() + valid.size() / 2, valid.end());
	return std::sqrt(valid[valid.size() / 2]);
} // EstimateSceneGSD
/*----------------------------------------------------------------*/

// Mesh resolution policy: every downstream stage's detail setting, derived from
// the cloud instead of hardcoded, so the same numbers work on a 20 m object scan
// and a 1 km aerial site.
//
// The governing identity. PoissonRecon divides a cube of side scale*maxExtent
// into 2^depth cells per axis, so cell = scale*ext / 2^depth, and the output face
// count is
//     faces ~= k * (ext / cell)^2                        (k = face-density coeff)
// Substituting cell and solving for depth eliminates ext entirely:
//     depth = 0.5 * log2( scale^2 * F / k )               (F = face budget)
// So DEPTH IS A FUNCTION OF THE FACE BUDGET, NOT OF SCENE SIZE. Extent is handled
// by tile count instead. This is what makes one policy work at every scale: a
// small scene reaches a fine cell at the same depth a large scene reaches a
// coarse one, and both stay inside the same memory envelope.
//
// Two independent ceilings apply, and we take the lower:
//   * budget-limited: the depth above, from F (memory / how big a mesh you want)
//   * data-limited:   round(log2(scale*ext / s)), s = median NN spacing. Past this
//                     the octree cells hold no new samples and Poisson just
//                     amplifies noise -- a heavier, bumpier mesh for no detail.
//
// NOTE the semantic change from the previous version: depth used to come from the
// data-limited term ALONE, which on dense aerial clouds asks for depth 15+ (a
// multi-billion-face mesh) and was only survivable because maxDepth capped it.
// The cap is now a backstop rather than the actual mechanism.
//
// The defaults (F = 40M faces, k = 3.0) yield depth 12, matching the value this
// tree used before, so dropping this in changes nothing until you pass a budget.
struct MeshPolicy {
	int    depth        = 0;     // PoissonRecon octree depth
	float  cellSingle   = 0.f;   // finest cell reachable in ONE untiled mesh
	float  cell         = 0.f;   // cell actually delivered (== cellSingle if tiles==1)
	int    tilesPerAxis = 1;     // >1 : target needs tiled meshing to reach `cell`
	int    refineLevel  = -1;    // RefineMesh resolution-level; -1 = unknown (no GSD)
	float  extent       = 0.f;   // cloud bounding-box max axis extent
	float  spacing      = 0.f;   // median nearest-neighbour spacing (0 = unmeasured)
	float  k            = 0.f;   // face-density coefficient used
	double facesBudget  = 0.0;   // face budget used
	// The four ceilings, as chosen (depth == min of them, clamped). Exposed so the
	// caller can tell whether k could possibly have changed the answer: budget and
	// texture are functions of k, data and pixels are not.
	int    depthBudget  = 0;
	int    depthData    = 0;
	int    depthTexture = 0;
	int    depthPixels  = 0;
	// depthPixels BEFORE flooring. The pixel ceiling is the binding one on most scenes,
	// and its fractional part decides a 2x cell / 4x face outcome, so the raw value is the
	// number to read when a scene comes out coarser than expected: a value like 9.97 means
	// the scene missed the next level by 3% of a cell, not by a level's worth of quality.
	// See POISSON_DEPTH_PIXELS_TOL.
	float  depthPixelsRaw = 0.f;
	// Octree cube scale handed to PoissonRecon (its --scale). 1.1 unless the cube was
	// padded to unlock a deeper depth -- see POISSON_CUBE_PAD_MAX. cellSingle/cell are
	// computed from THIS, not from the 1.1 default, so a padded run reports the cell it
	// will actually get.
	float  scale        = 1.1f;
	// Pad factor actually applied (scale / 1.1). 1.0 = no padding. Logged so a run whose
	// depth came from padding is distinguishable from one that cleared the ceiling itself.
	float  cubePad      = 1.f;
};

// targetCell   : desired output cell size; <=0 means "best single mesh"
// facesBudget  : max faces per mesh; <=0 falls back to OPENMVS_POISSON_FACE_BUDGET_M
//                then the built-in default. Derive it from available RAM at the
//                orchestrator (~40M faces per 64 GB, linear), which is also where
//                the value is needed to size RefineMesh.
// faceDensityK : terrain roughness coefficient; <=0 falls back to
//                OPENMVS_POISSON_FACE_K then the built-in default.
//                Flat ground runs ~2, mixed terrain ~3, dense vegetation/urban
//                5-8. Calibrate per dataset by reconstructing once at depth 9 and
//                solving k = faces * scale^2 / 4^9.
// gsdFull      : full-resolution ground sample distance, same units as the cloud.
//                Only used to derive refineLevel; <=0 leaves it -1 (unknown).
// Effective face budget / roughness, resolved once from (in priority order) an
// explicit argument, an environment override, then the built-in default. The env
// path exists because both are per-DATASET properties: baking them into macros
// would mean a rebuild per site, which defeats the point of an adaptive policy.
//   OPENMVS_POISSON_FACE_BUDGET_M : max faces per mesh, in millions
//   OPENMVS_POISSON_FACE_K        : measured face-density coefficient
// Bytes of solve-time footprint per output face, measured on the 974 m / depth-12
// run: during the linear solve the process held ~13.3 GB while producing 14.6M raw
// faces, of which ~6.0 GB was the retained xyz+normals cloud -- so the octree and
// solver accounted for ~7.3 GB, i.e. ~500 B/face. Used to turn available RAM into
// a face budget. Log-and-compare on every run via [MESH-CALIB] so this can be
// re-derived if it drifts.
// MEASURED on the 974 m depth-12 run: "Finalized tree" held 10,022 MB, of which
// 6,018 MB was the retained xyz+normals cloud -> 4.0 GB of octree for 14,502,317
// raw faces = 276 B/face. Rounded up slightly for headroom. (An earlier value of
// 500 was inferred from total process memory and was ~1.8x too conservative, which
// cost a whole depth level.) Re-derive from the [MESH-CALIB] projected-vs-actual
// line if it drifts on other terrain.
#ifndef POISSON_BYTES_PER_FACE
#define POISSON_BYTES_PER_FACE 290.0
#endif
// Bytes per point freed by releasePointCloud before the solve -- the colours,
// per-point view lists and weights that Poisson never reads. From the
// release-pointcloud measurement: ~51 B/point (2.6 GB of a 5.67 GB peak at 50.6M
// points). This memory is NOT available when the budget is computed (the cloud is
// already loaded, so the OS has it excluded from availPhys) but IS available during
// the solve, so it must be added back or the budget under-shoots badly.
#ifndef POISSON_CLOUD_RECLAIMABLE_BYTES_PER_POINT
#define POISSON_CLOUD_RECLAIMABLE_BYTES_PER_POINT 51.0
#endif
// Total resident bytes per point of the FULLY loaded dense cloud: the 24 B of
// xyz+normals we keep plus the ~51 B of colours/views/weights released before the
// solve. This is what sets the process peak before the octree exists (measured:
// 262,967,556 points -> 22.23 GB observed peak, ~85 B/point including allocator
// slack), and it is used as a floor on the face budget so the chosen depth does not
// vary with unrelated memory pressure on the machine.
#ifndef POISSON_CLOUD_LOADED_BYTES_PER_POINT
#define POISSON_CLOUD_LOADED_BYTES_PER_POINT 85.0
#endif
// Fraction of AVAILABLE physical memory the mesh stage may plan to occupy. The
// remainder absorbs allocator slack, the trim/cull kd-tree, and the OS.
#ifndef POISSON_MEMORY_FRACTION
#define POISSON_MEMORY_FRACTION 0.70
#endif

static double PoissonFaceBudget(double override_, size_t numPoints)
{
	static const double env = []() -> double {
		const char* v = std::getenv("OPENMVS_POISSON_FACE_BUDGET_M");
		return v ? std::atof(v) * 1.0e6 : 0.0;
	}();
	if (override_ > 0.0) return override_;
	if (env > 0.0)       return env;

	// Derive from what the machine actually has, so the same build produces a
	// coarser mesh on a 16 GB laptop and a finer one on a 64 GB workstation with no
	// operator input. Falls back to a fixed budget if the query is unavailable.
	double availBytes = 0.0;
#ifdef _WIN32
	MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
	if (GlobalMemoryStatusEx(&ms))
		availBytes = (double)ms.ullAvailPhys;
#endif
	if (!(availBytes > 0.0))
		return 40.0e6;
	// The cloud is resident when this runs, so availPhys already excludes all of it.
	// Add back the part releasePointCloud frees before the solve, and subtract only
	// the xyz+normals that stay resident throughout it.
	const double reclaimable = (double)numPoints * POISSON_CLOUD_RECLAIMABLE_BYTES_PER_POINT;
	const double retained    = (double)numPoints * 6.0 * sizeof(float);
	double usable = (availBytes + reclaimable) * POISSON_MEMORY_FRACTION - retained;

	// Floor: loading the dense cloud has ALREADY set the process peak (~75 B/point
	// with all its streams). An octree that stays under that peak therefore costs
	// nothing extra -- measured, a depth-13 solve reached 20.57 GB inside a
	// cloud-driven peak of 22.23 GB, so it was free. Without this floor the chosen
	// depth swings with whatever else happens to be running on the machine, which
	// makes the output resolution non-reproducible for the same dataset.
	const double alreadyPaid = (double)numPoints * POISSON_CLOUD_LOADED_BYTES_PER_POINT - retained;
	if (alreadyPaid > usable)
		usable = alreadyPaid;

	const double faces  = usable / POISSON_BYTES_PER_FACE;
	// Clamp: below the floor the mesh is useless, above the ceiling the downstream
	// RefineMesh/TextureMesh stages become the real constraint anyway.
	return std::min(std::max(faces, 2.0e6), 400.0e6);
}
static float PoissonFaceDensityK(float override_)
{
	static const float env = []() -> float {
		const char* v = std::getenv("OPENMVS_POISSON_FACE_K");
		return v ? (float)std::atof(v) : 0.f;
	}();
	if (override_ > 0.f) return override_;
	if (env > 0.f)       return env;
	// Conservative: a HIGH k selects a SHALLOWER depth and fewer faces, so an
	// uncalibrated run under-delivers resolution rather than exhausting memory.
	// Measured values run far lower than this on aerial terrain (~1.05 on a
	// 974 m site); every run now reports its own k -- see the [MESH-CALIB] line.
	return 3.0f;
}
/*----------------------------------------------------------------*/

// Finest RefineMesh resolution level whose view planes fit the memory we may use.
//
// Depends only on view pixel count and free RAM -- NOT on the mesh -- which is exactly
// what lets the depth policy consult it BEFORE any mesh exists, and so decide how much
// subdivision headroom to leave RefineMesh.
// freedBeforeStage: bytes this process still holds that will be gone before the stage
// in question runs. RefineMesh and TextureMesh are SEPARATE PROCESSES launched after
// this one exits, so the entire dense cloud counts -- querying availPhys with 22 GB of
// cloud resident understates their budget by exactly that, which measured as a whole
// depth level lost on a 1940-view scene (texture ceiling 11 instead of 12).
static int RefineFinestAffordableLevel(const ImageArr& images, size_t freedBeforeStage,
	size_t& outViews, double& outTotalPixels, double& outAllowBytes)
{
	outViews = 0;
	outTotalPixels = 0.0;
	for (size_t i = 0; i < (size_t)images.GetSize(); ++i) {
		const Image& img = images[(IIndex)i];
		if (!img.IsValid())
			continue;
		outTotalPixels += (double)img.width * (double)img.height;
		++outViews;
	}
	double availBytes = 0.0;
#ifdef _WIN32
	MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
	if (GlobalMemoryStatusEx(&ms))
		availBytes = (double)ms.ullAvailPhys;
#endif
	outAllowBytes = ((availBytes > 0.0 ? availBytes : 32.0e9) + (double)freedBeforeStage)
		* POISSON_MEMORY_FRACTION;
	if (outViews == 0 || outTotalPixels <= 0.0)
		return 0;
	int lvl = 0;
	while (lvl < 6 &&
		outTotalPixels * POISSON_REFINE_BYTES_PER_PIXEL / std::pow(4.0, (double)lvl) > outAllowBytes)
		++lvl;
	return lvl;
}
/*----------------------------------------------------------------*/

// Face budget imposed by TEXTURING, which is the tightest stage in the chain.
//
// TextureMesh holds ~132 observations per face on a 1940-view scene (measured:
// 975,795,624 obs over 7,388,176 faces) across two structures at 44 B/obs --
// perViewOut 24 B plus facesDatas 20 B -- so ~5.8 KB per FACE, roughly 21x the
// Poisson solve's ~272 B/face. Observations per face scale with VIEW COUNT, not with
// face count (a face is seen by however many cameras cover it, and subdividing it
// does not change that), so the coefficient is per-view.
//
// NOT a function of atlas size: atlas demand is surfaceArea/GSD^2 and is invariant to
// face count -- halving the faces doubles each face's texel footprint. The atlas caps
// texture RESOLUTION, not geometry. Only per-patch seam padding gives face count any
// atlas cost at all, and that is a ~1.3-2x effect, not an ordering constraint.
//
// NON-STATIC: both the depth policy below (to avoid solving far finer than the output
// can carry) and ReconstructMesh.cpp (to set its decimation target) need this number,
// and they must not disagree.
// freedBeforeStage: see RefineFinestAffordableLevel. TextureMesh is a separate process
// launched after this one exits, so anything this process still holds is available to
// it and must be added back before sizing its budget.
double ComputeTextureFaceBudget(size_t nViews, size_t freedBeforeStage)
{
	constexpr double kObsPerFacePerView = 132.0 / 1940.0; // measured
	constexpr double kBytesPerObs       = 44.0;           // 24 B + 20 B
	constexpr double kMemFraction       = 0.75;           // OS + atlas headroom
	if (nViews == 0)
		return 0.0;
	double availBytes = 0.0;
#ifdef _WIN32
	MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
	if (GlobalMemoryStatusEx(&ms))
		availBytes = (double)ms.ullAvailPhys;
#endif
	if (!(availBytes > 0.0))
		return 0.0; // unknown: caller falls back to no cap
	const double obsPerFace = kObsPerFacePerView * (double)nViews;
	double budget = ((availBytes + (double)freedBeforeStage) * kMemFraction)
		/ (obsPerFace * kBytesPerObs);
	if (budget < 500000.0)    budget = 500000.0;
	if (budget > 200000000.0) budget = 200000000.0;
	return budget;
}
/*----------------------------------------------------------------*/

// Face budget imposed by the TEXTURE ATLAS, independent of RAM.
//
// A face that occupies only a handful of atlas texels carries no texture detail -- it is
// a flat-shaded triangle with a geometry cost. So the atlas bounds how many faces are
// worth building:
//
//     faces <= atlasTexels / texelsPerFace
//
// SURFACE AREA CANCELS. Texel size is sqrt(surfaceArea / atlasTexels), and the texels a
// face covers is faceArea / texel^2 = faceArea * atlasTexels / surfaceArea -- so for a
// mesh of N roughly-equal faces the texels per face is just atlasTexels / N. The bound is
// therefore a pure function of the hardware, valid at any scene scale, which is what lets
// it generalize: detect a 32k atlas and every stage upstream is allowed to be finer.
//
// This is the constraint the earlier "NOT a function of atlas size" note on
// ComputeTextureFaceBudget missed. That note is correct that atlas CONTENT demand is
// surfaceArea/GSD^2 and invariant to face count -- but total demand is content plus
// per-patch padding, and padding is not small: MEASURED on a 276 m / 368-view scene,
// 5x the faces cost 1.75x the atlas demand (demand ~ faces^0.35). Halving the faces on an
// atlas-limited scene buys roughly 10% finer texture.
//
// MEASURED at 16384^2 on the 1940-view aerial site: 4.1M faces allowed against a 2.6M
// mesh, so it does not bind there -- correct, since that scene's texel budget is spent on
// area, not on face count. It binds on scenes that are small in extent but heavily
// subdivided, which is exactly where over-solving is wasted work.
#ifndef POISSON_ATLAS_MAX_DIM
#define POISSON_ATLAS_MAX_DIM 16384   // GL_MAX_TEXTURE_SIZE on current mainstream GPUs
#endif
// POISSON_CUBE_PAD_MAX: largest factor by which the octree cube may be grown past
// scale*extent in order to unlock one more octree level. See the CUBE PADDING block
// in ComputeMeshPolicy() for the argument; this is the cost dial.
//
// Padding does NOT make a depth increase cheaper -- a level is still ~4x the faces,
// ~3.5x the solve nodes, one finer RefineMesh level (4x the view planes) and ~4x the
// faces through texturing. It only removes the case where a scene pays for depth D-1
// while its own quality floor permitted D.
//
// 1.07 is deliberately tight: it covers the near-boundary scenes, which are the ones
// losing a whole factor of 2 for a few percent of cube, and declines the ones that
// would need a 40-50% bigger cube for a 1.35x cell (measured: BellisPark 1.47,
// MechanicFalls 1.48). Those are ordinary depth increases wearing a disguise -- if
// they are wanted, raise the face budget, do not smuggle them in here.
//
// Per-scene pad needed to reach the next level, measured over six scenes:
//   Marco 1.007 | OKState 1.04 | RichmondWater 1.064 | Niwot(127) 1.064
//   MechanicFalls 1.48 | BellisPark 1.47
// Set to 1.0 to disable padding entirely and restore the pre-change behaviour.
// RAISED 1.07 -> 1.10 (2026-08-22). FIELD CASE that proves the mechanism and set the value:
// Niwot at HIGHEST, after the dense cloud was cleaned of spurious points beneath the surface.
// Cleaning it shrank the bbox 134.1 -> 122.8 (-8.4%), which dropped the raw pixel ceiling
// 9.957 -> 9.881 and crossed an integer boundary: depth 10 -> 9, cell 0.1409 -> 0.2638,
// refine level 1 -> 2, final mesh 2,163,593 -> 475,873 faces. IMPROVING THE INPUT MADE THE
// OUTPUT 4.5x COARSER, and left HIGHEST producing a worse mesh than HIGH on the same scene.
//
// The pad it needed was 1.086 and the cap was 1.07 -- short by 1.5%. 1.10 admits it.
//
// Checked against the whole corpus: at 1.10 the ONLY additional scene that pads is this one.
// Every other scene either already padded at 1.07 (RichmondWater 1.045), was already carried
// by the tolerance (Marco 12.011, RichmondHistoric 10.980, Niwot-at-HIGH 9.957), is capped by
// its face budget (MechanicFalls, OKState), or needs far more than 1.10 (Redy 1.289,
// GEOTAG 1.399, BellisPark 1.46, Randy 1.813, SchnellTests 1.214).
//
// Do not read 1.10 as "padding is cheap". A level is still ~4x the faces, ~4x the refine
// view-planes and ~4x the textured faces; the cap is a cost dial and raising it widens the
// band of scenes that pay. It is justified here because the alternative is a tier inversion.
// RAISED 1.10 -> 1.15 (2026-08-24). 1.10 was set from Niwot-at-HIGHEST needing 1.086, and that
// margin was called out as uncomfortably tight at the time. It then failed on the very next
// bbox change: Niwot at HIGH needed 1.1018 and was declined by 0.16%, dropping depth 10 -> 9 and
// the final mesh from 2,163,593 faces to 359,807.
//
// 1.15 is NOT fitted to that number -- it is the middle of an empty band. Pad required to reach
// the next level, across the corpus:
//   Niwot HIGHEST 1.086 | Niwot HIGH 1.1018 | <-- gap --> | SchnellTests 1.214 | Redy 1.289
//   GEOTAG 1.399 | BellisPark 1.46 | Randy 1.813 ... (MechanicFalls, OKState budget-capped;
//   Marco, RichmondHistoric already carried by POISSON_DEPTH_PIXELS_TOL)
// Nothing sits between 1.102 and 1.214, so 1.15 buys ~4.5% of bbox drift tolerance without
// admitting a single additional scene. Anything above ~1.21 starts paying a full octree level
// on SchnellTests.
//
// ROOT CAUSE, not fixed here: the depth decision keys off the RAW bbox, which is outlier-driven
// and therefore moves whenever the densify filters change -- 134.1 -> 122.3 on this scene while
// the robust 0.2-99.8 extent held at 103.7 -> 104.2. Padding absorbs the drift; it does not
// remove it. The structural fix is to choose the depth from the robust extent and size the cube
// from the raw one, so cloud-filter work stops moving mesh resolution.
#ifndef POISSON_CUBE_PAD_MAX
#define POISSON_CUBE_PAD_MAX 1.15
#endif
// Usable fraction of the atlas: must match TEXTURE_ATLAS_FIT_MARGIN in SceneTexture.cpp,
// which is the fraction AdaptiveFitPatches actually fills (measured: realized area lands
// on 100% of that budget).
//
// KNOWN 4% GENEROUS -- deliberately left at 0.97. Do not "fix" without re-measuring the
// decimation cap it feeds.
//
// The real usable fraction is 0.93, now measured twice at two different atlas sizes on the
// realized packing rather than inferred from which ladder attempt succeeded:
//   16384: patches occupied 249.6 of 268.4 Mpx = 0.930
//   32768: patches occupied 998.6 of 1071.3 Mpx = 0.930
// It matches TEXTURE_ATLAS_FIT_MARGIN (0.93) exactly, which is the point -- shelf packing
// genuinely needs ~7% slack, so that margin is ACCURATE rather than wasteful.
//
// SUPERSEDES an earlier note here claiming the true value was 0.82 and the cap 18% too
// generous. That came from reading which fit-ladder attempt succeeded (0.97 -> overflow ->
// 0.82) back when the ladder stepped by 0.85; the ladder now starts at 0.93 with a 0.95
// step and lands on its first attempt, so the old inference no longer describes anything.
//
// Left at 0.97 on purpose: correcting to 0.93 tightens the face cap ~4% (4.07M -> 3.90M at
// 16384), which costs geometry on a validated path to make a constant honest, with no
// visible gain. Treat the atlas face budget as ~4% optimistic and size accordingly.
#ifndef POISSON_ATLAS_USABLE_FRACTION
#define POISSON_ATLAS_USABLE_FRACTION 0.97
#endif
// Minimum atlas texels per face, i.e. an 8x8 texel patch. Below this a face cannot show
// texture detail and the geometry is decoration the render cannot display.
#ifndef POISSON_ATLAS_TEXELS_PER_FACE
#define POISSON_ATLAS_TEXELS_PER_FACE 64.0
#endif
// Share of the atlas face budget withheld from THIS stage so RefineMesh has room to
// subdivide inside it. The pre-refine decimation cap becomes budget / this.
//
// WHY THIS EXISTS. Two gates now enforce the same atlas ceiling: the decimation cap here
// (pre-refine) and ClampSubdivideAreaToAtlasBudget in SceneRefine.cpp (during refine).
// At 1.0 they are in series against the SAME number, so this stage spends the entire
// budget and refine inherits nothing. MEASURED on RichmondHistoric at 16384 px: the cap
// left 11,414 faces of headroom, 0.28% of the budget, and the finest refine scale then
// got 258 splits against 672,252 it wanted -- 0.038%. Refine was not declining to work,
// it was starved.
//
// At 1.7 (the measured subdivision factor; the range across scenes is 1.48-1.88) this
// stage emits ~budget/1.7 and refine grows it back toward the full budget. Same final
// face count, but the faces come from refine's PROJECTED-AREA test instead of Poisson
// depth -- see the note at the refineHeadroom decision below, which measured that at
// equal face counts the adaptive route wins.
//
// TWO REASONS IT DEFAULTS TO 1.0, i.e. off:
//  - It is a PREDICTION. Everything else about this cap is deliberately estimate-free
//    (see the texCap note below on why the atlas bound is not turned into a depth). A
//    wrong factor here costs subdivision headroom in one direction and geometry in the
//    other; it can never breach the ceiling, because refine's own clamp is exact.
//  - This stage cannot know whether RefineMesh will actually RUN. If it does not, the
//    withheld geometry is simply lost. The Global Mapper chain always runs it, so 1.7 is
//    safe there, but that is a property of the caller, not of this file.
// Applied to the ATLAS component only, never to the memory one -- refine's clamp enforces
// the atlas ceiling and knows nothing about RAM, so under-capping the memory budget here
// would hand the growth to a ceiling no later stage checks.
#ifndef POISSON_ATLAS_REFINE_RESERVE
#define POISSON_ATLAS_REFINE_RESERVE 1.0
#endif
// Defined in SceneTexture.cpp. Spins up a throwaway hidden-window WGL context once,
// queries GL_MAX_TEXTURE_SIZE, caches it, and falls back to 16384 when no driver is
// reachable. Shared rather than duplicated so this stage sizes its face cap from the
// SAME number the texturing stage will actually pack into. If the two disagree the face
// capacity is wrong by (dim ratio)^2 -- 4x on a 32768-capable host, which silently
// decimates the mesh for an atlas that is not the one being built.
//
// Sharing the PROBE is necessary but was not sufficient: TextureMesh used to resolve
// --max-texture-size < 0 by calling this probe directly, which honoured the host limit
// but silently ignored OPENMVS_ATLAS_MAX_DIM -- so an env pin moved this stage's face
// cap without moving the atlas it was being sized for, reintroducing the same (dim
// ratio)^2 error the shared probe exists to prevent. MeshTexture::GenerateTexture now
// calls ResolveAtlasMaxDimEx below instead, so all three stages share the whole
// POLICY (pin > probe > fallback), not just the probe, and it warns when an explicit
// --max-texture-size disagrees with the pin.
extern int GetOpenGLMaxTextureSize();

// Atlas dimension the texturing stage will pack into, in precedence order:
//   1. explicit argument     -- caller already knows the value
//   2. OPENMVS_ATLAS_MAX_DIM -- operator override
//   3. host GL_MAX_TEXTURE_SIZE
//   4. POISSON_ATLAS_MAX_DIM -- only if the probe itself returns nothing usable
//
// The env override deliberately outranks the probe. GL_MAX_TEXTURE_SIZE is 16384 on
// AMD/Intel and 32768 on NVIDIA, so on a mixed fleet an unpinned probe makes the same
// job produce different texture resolution on different machines. Pinning one
// dimension here and the matching --max-texture-size on TextureMesh is the only way
// to make a quality tier mean the same thing everywhere.
// The dimension alone is not enough to report the DECISION. "Atlas 8192" reads as a
// setting; "atlas 8192, host supports 16384" reads as a loss, and only the second tells
// an operator that the mesh was decimated 4x harder than this host could have carried.
// So the resolution is done once here and the inputs are handed back, deliberately as
// plain ints so ReconstructMesh.cpp can declare it extern without a shared header (see
// the note on ComputeTextureFaceBudget at its call site).
//   pHostLimit : GL_MAX_TEXTURE_SIZE as probed, or 0 if no driver was reachable
//   pEnvPin    : OPENMVS_ATLAS_MAX_DIM if set, else 0
// Either may be null.
int ResolveAtlasMaxDimEx(int atlasMaxDim, int* pHostLimit, int* pEnvPin)
{
	static const int envDim = []() -> int {
		const char* v = std::getenv("OPENMVS_ATLAS_MAX_DIM");
		const int d = v ? std::atoi(v) : 0;
		return d > 0 ? d : 0;
	}();
	// Probed even when an explicit argument or the env pin decides the answer: the
	// point of the report is the gap between what we take and what the host offers,
	// and the probe caches after the first call so this costs nothing to ask twice.
	const int glDim = GetOpenGLMaxTextureSize();
	if (pHostLimit) *pHostLimit = glDim;
	if (pEnvPin)    *pEnvPin    = envDim;
	if (atlasMaxDim > 0)
		return atlasMaxDim;
	if (envDim > 0)
		return envDim;
	return (glDim > 0) ? glDim : (int)POISSON_ATLAS_MAX_DIM;
}

int ResolveAtlasMaxDim(int atlasMaxDim)
{
	return ResolveAtlasMaxDimEx(atlasMaxDim, nullptr, nullptr);
}

// The TRUE atlas ceiling: how many faces the atlas can texture at the target texel
// density. This is the hard limit, so it is what the stage that sees the FINAL mesh must
// enforce -- ClampSubdivideAreaToAtlasBudget in SceneRefine.cpp, and RefineMesh's
// [REFINE-FACES] audit. Do not apply the refine reserve here; that would move the
// ceiling itself rather than reserving room beneath it.
double ComputeAtlasFaceBudget(int atlasMaxDim)
{
	const double dim = (double)ResolveAtlasMaxDim(atlasMaxDim);
	if (!(dim > 0.0))
		return 0.0; // unknown: caller falls back to no cap
	const double texels = dim * dim * POISSON_ATLAS_USABLE_FRACTION;
	return texels / POISSON_ATLAS_TEXELS_PER_FACE;
}

// The same ceiling minus the share reserved for RefineMesh's subdivision -- what the
// PRE-REFINE decimation cap in ReconstructMesh.cpp should use. Identical to
// ComputeAtlasFaceBudget when POISSON_ATLAS_REFINE_RESERVE is 1.0 (the default), so the
// two are interchangeable until an operator opts into the split.
double ComputeAtlasFaceBudgetPreRefine(int atlasMaxDim)
{
	const double budget = ComputeAtlasFaceBudget(atlasMaxDim);
	const double reserve = (double)POISSON_ATLAS_REFINE_RESERVE;
	if (!(budget > 0.0) || !(reserve > 1.0))
		return budget; // unknown, or no reserve requested
	return budget / reserve;
}

// Fraction of an atlas that carries actual triangle coverage, i.e. what is left after
// per-patch bounding-box slack. Needed because the atlas holds the patches' BOUNDING
// BOXES, not the surface itself, so sizing on covered area alone under-asks by ~2x.
//
// CALIBRATED against TextureMesh's own measured appetite, by inverting
// fill = surface / (gsd^2 * wantedDim^2) on finished runs:
//
//     Randy           1.57M faces, sub-ceiling    -> 0.442   gsd 0.0069, surface 6824
//     RichmondWater  11.59M faces, capped 32768   -> 0.557
//     RichmondWater  11.13M faces, uncapped       -> 0.584
//     Niwot           1.60M faces, capped 16384   -> 0.637
//
// The earlier 0.52 came from the ATLAS-GUTTER coverage figure (138.9 of 268.4 Mpx
// cleared), which measures a related but DIFFERENT quantity -- texels cleared and regrown,
// not the packing efficiency the sizing needs -- and it ran predictions 6-12% high.
//
// TREAT THIS AS A ROUGH SIZING FIGURE, NOT A CALIBRATED CONSTANT. The measured spread is
// 0.442-0.637, a 1.44x range, and it is NOT monotonic in tessellation (Randy is the finest
// gsd and the lowest fill; the tidy "coarser mesh packs better" story from the first three
// points does not survive the fourth). Predictions run roughly +-20% in dim, in BOTH
// directions -- 12% high on RichmondWater, 12% low on Randy.
//
// Why a loose figure is still acceptable: this value never sets the atlas. TextureMesh
// resolves its own ceiling and sizes from actual patch rects, so it lands on the right
// dimension regardless (Randy predicted 15794, TextureMesh built 16384). All this feeds is
// the pre-refine face BUDGET, which across five validated runs has never once bound --
// GM's --decimate (0.25 / 0.4 / 0.98) always cut first. The exposure is a low prediction
// meeting a mesh near the budget, which would over-decimate by up to ~25%. Do not add a
// fudge factor to compensate; measure more scenes first.
#ifndef POISSON_ATLAS_PATCH_FILL
#define POISSON_ATLAS_PATCH_FILL 0.59
#endif

// THE ATLAS DIMENSION FOR THIS SCENE.
//
// The atlas size is a property of the SCENE, not of the mesh and not of the machine.
// Measured: TextureMesh's own "full resolution needed about N px" held at
// 30132 / 30046 / 30028 / 29989 across face counts of 7.22M / 4.54M / 4.07M / 3.64M --
// a 0.5% spread against a 2x change in tessellation. Patch size follows the SOURCE
// PIXELS covering the surface, so the appetite is
//
//     texels = surfaceArea / (gsd^2 * POISSON_ATLAS_PATCH_FILL)
//
// which every stage can evaluate from data it already holds. That is what lets
// ReconstructMesh, RefineMesh and TextureMesh agree on one dimension in a SINGLE RUN
// with nothing passed between them -- no environment variable, no file, no prior run.
// They agree because they measure the same invariant, not because they were told.
//
// VALIDATED against a real run: surface 1.6e5 units^2 at gsd 0.0155424 predicts 35689 px
// where TextureMesh measured 34887 px from the actual patch rects -- 2.3% high.
//
// The CEILING is the host's GL_MAX_TEXTURE_SIZE, deliberately: this deliverable is
// displayed on the machine that builds it, so an atlas it cannot sample natively is not
// a higher-quality result, it is a broken one. That does mean the same scene yields a
// larger atlas on a 32768-capable host -- an intended product behaviour, not the silent
// stage-to-stage disagreement ResolveAtlasMaxDimEx warns about.
//
// Returns the ceiling unchanged when the scene cannot be measured (no poses, empty mesh),
// which reproduces the previous behaviour exactly. Out-params are for the log line that
// has to explain the choice; any may be null.
int ComputeSceneAtlasDim(const ImageArr& images, const Mesh& mesh,
	int* pCeiling, double* pWantDim, double* pSurfaceArea, double* pGsd)
{
	const int ceiling = ResolveAtlasMaxDim(0);
	if (pCeiling) *pCeiling = ceiling;
	if (pWantDim) *pWantDim = 0.0;
	if (pSurfaceArea) *pSurfaceArea = 0.0;
	if (pGsd) *pGsd = 0.0;
	if (mesh.faces.IsEmpty() || mesh.vertices.IsEmpty() || ceiling <= 0)
		return ceiling;

	// Same estimator the depth policy uses; it needs only poses, so it works on a mesh
	// whose vertices stand in for the cloud.
	const float gsd = EstimateSceneGSD(images,
		reinterpret_cast<const float*>(mesh.vertices.Begin()), mesh.vertices.GetSize());
	if (!(gsd > 0.f))
		return ceiling; // no usable poses: cannot size, keep the ceiling

	double surfaceArea = 0.0;
	const Mesh::VertexArr& V = mesh.vertices;
#ifdef _USE_OPENMP
	#pragma omp parallel for reduction(+:surfaceArea) schedule(static)
#endif
	for (ptrdiff_t i = 0; i < (ptrdiff_t)mesh.faces.GetSize(); ++i) {
		const Mesh::Face& f = mesh.faces[(Mesh::FIndex)i];
		const Mesh::Vertex& a = V[f[0]];
		const Mesh::Vertex& b = V[f[1]];
		const Mesh::Vertex& c = V[f[2]];
		const double ux = (double)b.x-a.x, uy = (double)b.y-a.y, uz = (double)b.z-a.z;
		const double vx = (double)c.x-a.x, vy = (double)c.y-a.y, vz = (double)c.z-a.z;
		const double cx = uy*vz - uz*vy, cy = uz*vx - ux*vz, cz = ux*vy - uy*vx;
		surfaceArea += 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
	}
	if (pSurfaceArea) *pSurfaceArea = surfaceArea;
	if (pGsd) *pGsd = (double)gsd;
	if (!(surfaceArea > 0.0))
		return ceiling;

	const double wantTexels = surfaceArea / ((double)gsd * gsd * POISSON_ATLAS_PATCH_FILL);
	const double wantDim = std::sqrt(wantTexels);
	if (pWantDim) *pWantDim = wantDim;

	// Never below a floor that can hold anything at all, never above what the host can
	// sample. A scene wanting less than the ceiling gets a SMALLER atlas -- correct, and
	// its face budget scales down with it so texels/face stays on target.
	const int dim = (int)std::lround(wantDim);
	return std::max(64, std::min(dim, ceiling));
} // ComputeSceneAtlasDim
/*----------------------------------------------------------------*/

static MeshPolicy ComputeMeshPolicy(const float* ptsRaw, size_t numPoints,
	float scaleFactor = 1.1f, float targetCell = 0.f,
	double facesBudget = 0.0, float faceDensityK = 0.f, float gsdFull = 0.f,
	int minDepth = 8, int maxDepth = 14,
	float knownExtent = 0.f, float knownSpacing = 0.f,
	double textureFaceCap = 0.0,
	double refineHeadroom = POISSON_REFINE_HEADROOM)
{
	// RefineMesh subdivides any face projecting larger than --max-face-area (32 px
	// by default), so its working GSD must be ~1/sqrt(32) of the cell for
	// refinement to add detail rather than idle. 8 is sqrt(32) rounded up.
	constexpr float kRefinePixelsPerCell = 8.f;

	MeshPolicy pol;
	pol.k           = PoissonFaceDensityK(faceDensityK);
	pol.facesBudget = PoissonFaceBudget(facesBudget, numPoints);

	// Budget-limited depth. Independent of the cloud, so it is also the answer for
	// every degenerate-input path below.
	//
	// FLOOR, not round: faces scale 4x per level, so rounding up can overshoot the
	// budget by nearly 2x (12.73 -> 13 turned a 40M budget into 58M on the 974 m
	// site). The budget is a ceiling, so truncate and let the operator raise the
	// budget if they want the extra level.
	const double sc2 = (double)scaleFactor * (double)scaleFactor;
	const int depthBudget = (int)std::floor(
		0.5 * std::log2(sc2 * pol.facesBudget / (double)pol.k));

	// Third ceiling: the OUTPUT face count, which texturing bounds. Solving far finer
	// than the deliverable can carry is pure waste -- and worse than waste, because the
	// surplus has to be decimated away and heavy decimation erodes exactly the features
	// the fine solve captured. MEASURED: 53.5M raw faces decimated 10.3x to 5.2M took
	// 9m38s and visibly rounded edges, where 14.5M decimated 2.8x to the same 5.2M took
	// 2m40s and did not. So allow only a modest oversolve above the cap -- enough for
	// adaptive decimation to redistribute detail, not enough to erode it.
	const int depthTexture = (textureFaceCap > 0.0)
		? (int)std::floor(0.5 * std::log2(
			sc2 * textureFaceCap * POISSON_TEXTURE_OVERSOLVE / (double)pol.k))
		: maxDepth;

	// Fourth ceiling, and in practice the binding one: keep the cell large enough that
	// faces project to enough pixels for reliable per-face view selection, with one
	// level of subdivision headroom for RefineMesh. See POISSON_REFINE_HEADROOM.
	// Needs the extent, so it is resolved inside finish() below where that is known.
	const double minCellForTexture = (gsdFull > 0.f)
		? refineHeadroom * POISSON_REFINE_SUBDIV_FACTOR * (double)gsdFull
		: 0.0;

	// Degenerate inputs: no spacing to measure, so no data-limited ceiling. Fall
	// back to the budget alone rather than the old midpoint guess, which ignored
	// both the data and the memory envelope.
	const auto finish = [&](int depthData) {
		int chosen = (depthData > 0) ? std::min(depthBudget, depthData) : depthBudget;
		chosen = std::min(chosen, depthTexture);
		// Baseline cube: whatever the caller asked for. Cube padding below may raise it.
		pol.scale   = scaleFactor;
		pol.cubePad = 1.f;
		// Deepest depth whose cell is still >= minCellForTexture.
		int depthPixels = maxDepth;
		double depthPixelsRaw = 0.0; // unfloored, logged so a near-boundary scene is visible
		if (minCellForTexture > 0.0 && pol.extent > 0.f) {
			depthPixelsRaw =
				std::log2((double)scaleFactor * (double)pol.extent / minCellForTexture);
			// Tolerance on this ceiling ONLY -- it is a soft quality target, not a hard
			// resource limit, and its fractional part moves with log2(extent) so a scalar
			// headroom cannot keep it off an integer boundary. See POISSON_DEPTH_PIXELS_TOL.
			depthPixels = (int)std::floor(depthPixelsRaw + POISSON_DEPTH_PIXELS_TOL);
			chosen = std::min(chosen, depthPixels);
		}
		pol.depthPixelsRaw = (float)depthPixelsRaw;

		// CUBE PADDING. The octree cube is scale*extent, and `scale` is a free
		// parameter -- PoissonRecon only needs it big enough to contain the points.
		// The achievable cell is therefore quantised in factors of 2 ONLY because the
		// cube is held at 1.1x: cell = scale*ext / 2^depth.
		//
		// So when the PIXEL ceiling alone is what blocks a deeper depth, growing the
		// cube is strictly better than accepting the coarser level. MEASURED on two
		// runs of the same 109-image scene: extent 134.1 put depth 10 only 0.6% under
		// the floor (tolerance absorbed it, cell 0.1441), while extent 127 put it 6.4%
		// under (tolerance could not, so it fell back to depth 9 and cell 0.2728 --
		// 1.88x coarser than that scene's OWN floor permitted). A 6.4% bigger cube
		// would have given the second run depth 10 with the cell legally ON the floor.
		//
		// This is preferable to simply widening POISSON_DEPTH_PIXELS_TOL: the tolerance
		// buys the deeper depth by accepting a cell BELOW the reliability floor, while
		// padding buys the same depth with the cell at or above it. No violation.
		//
		// Bounded by POISSON_CUBE_PAD_MAX, and only ever up to the depth the OTHER
		// three ceilings already allow -- padding cannot buy past a resource limit,
		// only past the quantisation artefact. Scenes whose budget/data/texture ceiling
		// is already at depthPixels (measured: MechanicFalls and OKState, both
		// budget=10 pixels=10) are unaffected by design.
		if (minCellForTexture > 0.0 && pol.extent > 0.f && POISSON_CUBE_PAD_MAX > 1.0) {
			// Deepest depth the non-quantised ceilings permit.
			int otherCeil = depthBudget;
			if (depthData > 0) otherCeil = std::min(otherCeil, depthData);
			otherCeil = std::min(otherCeil, depthTexture);
			otherCeil = std::min(otherCeil, maxDepth);
			for (int d = chosen + 1; d <= otherCeil; ++d) {
				// Cube side needed for depth d to land the cell exactly on the floor.
				const double needScale =
					minCellForTexture * (double)(1u << d) / (double)pol.extent;
				const double pad = needScale / (double)scaleFactor;
				if (!(pad > 0.0) || pad > POISSON_CUBE_PAD_MAX)
					break;              // next level costs more cube than we allow
				pol.scale   = (float)needScale;
				pol.cubePad = (float)pad;
				chosen      = d;
			}
		}

		pol.depth        = std::clamp(chosen, minDepth, maxDepth);
		pol.depthBudget  = depthBudget;
		// Normalized to maxDepth when unmeasurable (degenerate cloud, no GSD) so that
		// callers can take a plain min of the four without re-testing for "not set".
		pol.depthData    = (depthData > 0) ? depthData : maxDepth;
		pol.depthTexture = depthTexture;
		pol.depthPixels  = depthPixels;
		// From pol.scale, NOT scaleFactor: on a padded run these differ and the cell
		// that matters downstream is the one the padded cube will actually produce.
		pol.cellSingle = (pol.extent > 0.f)
			? pol.scale * pol.extent / (float)(1u << pol.depth) : 0.f;
		// Never target finer than the cloud actually resolves.
		float desired = (targetCell > 0.f) ? targetCell : pol.cellSingle;
		if (pol.spacing > 0.f && desired < pol.spacing)
			desired = pol.spacing;
		pol.tilesPerAxis = (desired > 0.f && pol.cellSingle > desired)
			? (int)std::ceil(pol.cellSingle / desired) : 1;
		if (pol.tilesPerAxis < 1) pol.tilesPerAxis = 1;
		pol.cell = (pol.extent > 0.f)
			? pol.scale * (pol.extent / (float)pol.tilesPerAxis) / (float)(1u << pol.depth)
			: 0.f;
		if (gsdFull > 0.f && pol.cell > 0.f)
			pol.refineLevel = std::clamp((int)std::lround(
				std::log2(pol.cell / (kRefinePixelsPerCell * gsdFull))), 0, 4);

		MESH_DIAG("[MESH-POLICY] ext=%.4g spacing=%.4g k=%.2f budget=%.1fM faces",
			pol.extent, pol.spacing, pol.k, pol.facesBudget * 1e-6);
		MESH_DIAG("[MESH-POLICY] depth: budget=%d data=%d texture=%d pixels=%d (raw %.3f"
			" +tol %.2f) -> %d%s"
			" | cell_single=%.4g target=%.4g tiles=%dx%d cell=%.4g | refineLevel=%d",
			depthBudget, depthData, depthTexture, depthPixels,
			depthPixelsRaw, (double)POISSON_DEPTH_PIXELS_TOL, pol.depth,
			(pol.depth != chosen) ? " (CLAMPED)" : "",
			pol.cellSingle, desired, pol.tilesPerAxis, pol.tilesPerAxis, pol.cell,
			pol.refineLevel);
		// Padding changes the meaning of the depth line above -- `pixels` reports the
		// ceiling at the UNPADDED cube, so without this the chosen depth reads as if it
		// had violated its own ceiling. This is also the first line to check if a padded
		// run regresses: several cleanup thresholds are multiples of the median edge and
		// weaken as the mesh gets finer.
		if (pol.cubePad > 1.0001f) {
			const double cellWas =
				(double)scaleFactor * (double)pol.extent / (double)(1u << depthPixels);
			MESH_DIAG("[MESH-POLICY] CUBE PAD x%.4g: cube %.4g -> %.4g to reach depth %d"
				" instead of %d. Cell %.4g instead of %.4g (%.2fx finer), sitting ON the"
				" %.4g floor rather than under it. Costs a full level: ~4x faces,"
				" ~4x refine view-planes, ~4x textured faces."
				" Set POISSON_CUBE_PAD_MAX 1.0 to disable.",
				(double)pol.cubePad,
				(double)scaleFactor * (double)pol.extent,
				(double)pol.scale * (double)pol.extent,
				pol.depth, depthPixels,
				(double)pol.cellSingle, cellWas,
				(pol.cellSingle > 0.f) ? cellWas / (double)pol.cellSingle : 0.0,
				minCellForTexture);
		}
		// The pixel ceiling binds on most scenes and its FRACTIONAL part decides a 2x cell /
		// 4x face outcome, so say so out loud when a scene is sitting near a boundary --
		// that is the difference between "this scene wants depth N" and "this scene missed
		// depth N+1 by 3% of a cell". Threshold is twice the tolerance, i.e. the band where
		// re-reading POISSON_DEPTH_PIXELS_TOL is a reasonable response.
		if (depthPixelsRaw > 0.0 && depthPixels == chosen) {
			const double frac = depthPixelsRaw - std::floor(depthPixelsRaw);
			if (frac > 1.0 - 2.0 * (double)POISSON_DEPTH_PIXELS_TOL)
				MESH_DIAG("[MESH-POLICY] NOTE: pixel ceiling is %.3f -- within %.1f%% of a cell"
					" of depth %d. The tolerance (%.2f) decided this; a scene this close to a"
					" boundary is where the headroom calibration is least reliable.",
					depthPixelsRaw, 100.0 * (std::pow(2.0, 1.0 - frac) - 1.0),
					(int)std::floor(depthPixelsRaw) + 1, (double)POISSON_DEPTH_PIXELS_TOL);
		}
		if (pol.tilesPerAxis > 1)
			MESH_DIAG("[MESH-POLICY] cell %.4g needs %dx%d tiled meshing; an untiled run"
				" delivers %.4g instead", desired, pol.tilesPerAxis, pol.tilesPerAxis,
				pol.cellSingle);
		return pol;
	};

	// Caller already measured the cloud (e.g. re-deciding after a calibration probe)
	// -- skip the bbox pass and the kd-tree, which is the expensive half.
	if (knownExtent > 0.f && knownSpacing > 0.f) {
		pol.extent = knownExtent;
		pol.spacing = knownSpacing;
		return finish((int)std::lround(std::log2((scaleFactor * knownExtent) / knownSpacing)));
	}

	if (numPoints < 100)
		return finish(0);
	const Point3f* __restrict pts = reinterpret_cast<const Point3f*>(ptsRaw);

	// 1) axis-aligned bounding box -> largest axis extent. Serial single pass.
	// Input is pre-filtered (all finite), so no ISFINITE check needed.
	float minx = FLT_MAX, miny = FLT_MAX, minz = FLT_MAX;
	float maxx = -FLT_MAX, maxy = -FLT_MAX, maxz = -FLT_MAX;
	for (size_t i = 0; i < numPoints; ++i) {
		const Point3f& p = pts[i];
		if (p.x < minx) minx = p.x;  if (p.x > maxx) maxx = p.x;
		if (p.y < miny) miny = p.y;  if (p.y > maxy) maxy = p.y;
		if (p.z < minz) minz = p.z;  if (p.z > maxz) maxz = p.z;
	}
	const float ext = std::max(std::max(maxx - minx, maxy - miny), maxz - minz);
	pol.extent = ext;
	if (!(ext > 0.f))
		return finish(0);

	// ROBUST EXTENT CHECK -- the raw min/max above is set by the single most distant
	// point on each axis, so a handful of far outliers (dense-matching flyers over
	// water, sky blobs) inflate it. That is not cosmetic: PoissonRecon sizes its octree
	// cube from the SAME bounding box, so
	//     cell = scale * extent / 2^depth
	// means every cell is coarser by exactly the inflation factor, at every depth. The
	// mesh is blurred and the solve wastes resolution on empty space, and the balloon
	// has further to travel before the trimmer stops it.
	//
	// Compare against a percentile extent measured on a stride-sampled subset. Report
	// only -- the fix is to REMOVE the outliers so Poisson's own bbox shrinks too;
	// shrinking this number alone would make the policy disagree with the solver and
	// produce cells of a size nobody asked for.
	//
	// REPORT-ONLY, so the whole block rides MESH_DIAG -- nothing below it reads a
	// single value computed here. That matters beyond the log lines: this copies up
	// to three 200k-float arrays, nth_elements them a dozen times, runs a cyclic
	// Jacobi PCA and a 2048-bin sliding-window band search, all to print three lines.
	if (MESH_DIAG_ENABLED()) {
		const size_t kExtSamples = std::min<size_t>(numPoints, 200000);
		const size_t extStride = std::max<size_t>(1, numPoints / kExtSamples);
		std::vector<float> ax, ay, az;
		ax.reserve(numPoints / extStride + 1);
		ay.reserve(numPoints / extStride + 1);
		az.reserve(numPoints / extStride + 1);
		for (size_t i = 0; i < numPoints; i += extStride) {
			ax.push_back(pts[i].x); ay.push_back(pts[i].y); az.push_back(pts[i].z);
		}
		const auto span = [](std::vector<float>& v, double lo, double hi) -> float {
			if (v.size() < 10) return 0.f;
			const size_t iLo = (size_t)(lo * (double)(v.size() - 1));
			const size_t iHi = (size_t)(hi * (double)(v.size() - 1));
			std::nth_element(v.begin(), v.begin() + iLo, v.end());
			const float a = v[iLo];
			std::nth_element(v.begin(), v.begin() + iHi, v.end());
			return v[iHi] - a;
		};
		const float rx = span(ax, 0.002, 0.998);
		const float ry = span(ay, 0.002, 0.998);
		const float rz = span(az, 0.002, 0.998);
		const float robust = std::max(std::max(rx, ry), rz);
		if (robust > 0.f) {
			const double inflation = (double)ext / (double)robust;
			MESH_DIAG("[MESH-EXTENT] raw bbox %.4g vs 0.2-99.8 percentile %.4g"
				" -> inflation x%.2f | per-axis raw %.4g/%.4g/%.4g robust %.4g/%.4g/%.4g%s",
				ext, robust, inflation,
				maxx - minx, maxy - miny, maxz - minz, rx, ry, rz,
				(inflation > 1.15)
					? "  <-- OUTLIERS ARE INFLATING THE OCTREE CUBE; every cell is this"
					  " much coarser than the data warrants"
					: "");

			// The octree is a CUBE sized by the LARGEST axis, so if that axis is the
			// vertical one on an aerial survey, cell size is being set by something that
			// is probably not terrain. MEASURED on a 227-view site: raw z 1251 against
			// x 829 / y 582, i.e. the cube -- and therefore every cell -- is ~1.5x larger
			// than the horizontal footprint warrants, with most of the octree volume
			// empty air.
			//
			// A percentile cut is not enough to decide what to do: the same 0.2/99.8 trim
			// left z at 1070, so this is not a handful of flyers. Print the full ladder so
			// the SHAPE is visible -- a tight core with long thin tails (outliers, cuttable
			// at the gap) reads completely differently from a broad spread (real relief, or
			// a systematically bad reconstruction, neither of which a cut would fix).
			const auto pctOf = [](std::vector<float>& v, double p) -> float {
				if (v.empty()) return 0.f;
				const size_t i = (size_t)(p * (double)(v.size() - 1));
				std::nth_element(v.begin(), v.begin() + i, v.end());
				return v[i];
			};
			const int domAxis = (rz >= rx && rz >= ry) ? 2 : ((rx >= ry) ? 0 : 1);
			std::vector<float>& dom = (domAxis == 2) ? az : ((domAxis == 0) ? ax : ay);

			// IS THE CLOUD A SLAB, AND IN WHICH ORIENTATION?
			//
			// The axis-aligned extents cannot answer this. Terrain is a thin sheet, so its
			// point cloud must be slab-shaped in SOME frame -- but if that frame is rotated
			// relative to the coordinate axes, the sheet's thickness projects onto all
			// three axes and every one of them reads large. MEASURED on this scene:
			// 829/582/1251 with a smooth z distribution (IQR 383) is exactly that pattern,
			// and it is indistinguishable from "the cloud is genuinely volumetric, i.e.
			// mostly garbage" without looking at the principal axes.
			//
			// Principal extents settle it in one pass:
			//   thin 3rd extent  -> a slab lying oblique to the axes. The cloud is fine,
			//                       the bbox is honest, and there is nothing to reclaim --
			//                       note an oblique slab actually gives a SMALLER cube than
			//                       an aligned one (a length-L object along (1,1,1) spans
			//                       only L/sqrt(3) per axis), so re-orienting would HURT.
			//   all three large  -> the cloud really is volumetric and the problem is
			//                       upstream in densification, not in meshing.
			{
				const size_t n = ax.size();
				if (n > 100) {
					double mx = 0, my = 0, mz = 0;
					for (size_t i = 0; i < n; ++i) { mx += ax[i]; my += ay[i]; mz += az[i]; }
					mx /= (double)n; my /= (double)n; mz /= (double)n;
					double cxx=0, cxy=0, cxz=0, cyy=0, cyz=0, czz=0;
					for (size_t i = 0; i < n; ++i) {
						const double dx = ax[i]-mx, dy = ay[i]-my, dz = az[i]-mz;
						cxx+=dx*dx; cxy+=dx*dy; cxz+=dx*dz; cyy+=dy*dy; cyz+=dy*dz; czz+=dz*dz;
					}
					// Self-contained cyclic Jacobi for a symmetric 3x3 -- deliberately no
					// Eigen here: this file does not include <Eigen/Eigenvalues> and pulling
					// a header in for one diagnostic is not worth a build break.
					double A[3][3] = { {cxx/(double)n, cxy/(double)n, cxz/(double)n},
					                   {cxy/(double)n, cyy/(double)n, cyz/(double)n},
					                   {cxz/(double)n, cyz/(double)n, czz/(double)n} };
					double V[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };
					for (int sweep = 0; sweep < 24; ++sweep) {
						double off = 0.0;
						for (int p = 0; p < 3; ++p) for (int q = p+1; q < 3; ++q) off += A[p][q]*A[p][q];
						if (off < 1e-24) break;
						for (int p = 0; p < 3; ++p) for (int q = p+1; q < 3; ++q) {
							if (std::fabs(A[p][q]) < 1e-30) continue;
							const double theta = (A[q][q]-A[p][p]) / (2.0*A[p][q]);
							const double t = (theta >= 0.0 ? 1.0 : -1.0)
								/ (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
							const double c = 1.0/std::sqrt(t*t+1.0), s = t*c;
							for (int k = 0; k < 3; ++k) {
								const double akp = A[k][p], akq = A[k][q];
								A[k][p] = c*akp - s*akq;  A[k][q] = s*akp + c*akq;
							}
							for (int k = 0; k < 3; ++k) {
								const double apk = A[p][k], aqk = A[q][k];
								A[p][k] = c*apk - s*aqk;  A[q][k] = s*apk + c*aqk;
								const double vkp = V[k][p], vkq = V[k][q];
								V[k][p] = c*vkp - s*vkq;  V[k][q] = s*vkp + c*vkq;
							}
						}
					}
					// Order eigenvector columns by eigenvalue (A[i][i]) descending.
					int ord[3] = { 0, 1, 2 };
					for (int i = 0; i < 3; ++i) for (int j = i+1; j < 3; ++j)
						if (A[ord[j]][ord[j]] > A[ord[i]][ord[i]]) std::swap(ord[i], ord[j]);
					double pmin[3] = { 1e300, 1e300, 1e300 }, pmax[3] = { -1e300, -1e300, -1e300 };
					for (size_t i = 0; i < n; ++i) {
						const double dx = ax[i]-mx, dy = ay[i]-my, dz = az[i]-mz;
						for (int a = 0; a < 3; ++a) {
							const int c = ord[a];
							const double t = dx*V[0][c] + dy*V[1][c] + dz*V[2][c];
							if (t < pmin[a]) pmin[a] = t;
							if (t > pmax[a]) pmax[a] = t;
						}
					}
					// a=0 is the largest-variance direction, a=2 the smallest (the normal).
					const double e2 = pmax[0]-pmin[0], e1 = pmax[1]-pmin[1], e0 = pmax[2]-pmin[2];
					const int cn = ord[2];
					// ROD vs SLAB vs VOLUMETRIC, and the distinction is NOT cosmetic: it
					// decides whether re-orienting the cloud would shrink the octree cube
					// or grow it.
					//   ROD  (middle extent << longest): behaves like a line segment. A
					//        length-L rod along (1,1,1) spans only L/sqrt(3) per axis, so an
					//        OBLIQUE orientation gives the SMALLER cube. Aligning it makes
					//        the cube the full rod length -- worse. Nothing to reclaim.
					//   SLAB (middle extent ~ longest, thin third): rotation GROWS the
					//        axis-aligned box, so ALIGNING minimizes the cube. Worth doing.
					//   VOLUMETRIC: not a surface in any orientation -- the problem is
					//        upstream in densification, not in the mesh policy.
					// Testing thinnest-vs-longest alone cannot tell a rod from a slab, and
					// an earlier revision of this line reported a 1490/284/177 corridor
					// (clearly a rod at 5.2:1 middle-to-longest) as a SLAB.
					const bool isRod  = (e1 * 3.0 < e2);
					const bool isSlab = !isRod && (e0 * 5.0 < e1);
					MESH_DIAG("[MESH-EXTENT] principal extents %.4g / %.4g / %.4g"
						" (thinnest axis dir %.3f,%.3f,%.3f)"
						" | mid:long 1:%.1f thin:long 1:%.1f -- %s",
						e2, e1, e0, V[0][cn], V[1][cn], V[2][cn],
						(e1 > 1e-9) ? e2/e1 : 0.0, (e0 > 1e-9) ? e2/e0 : 0.0,
						isRod
							? "ROD/CORRIDOR: elongated, so the OBLIQUE orientation already"
							  " gives the smallest cube -- aligning it would make the cube the"
							  " full corridor length. Nothing to reclaim by re-orienting;"
							  " tiling ALONG the corridor is what would buy finer cells"
						: isSlab
							? "SLAB: a sheet oblique to the axes. Rotation GROWS an"
							  " axis-aligned box for a slab, so aligning the cloud WOULD"
							  " shrink the cube -- worth measuring"
							: "VOLUMETRIC: not a surface in any orientation. The problem is"
							  " upstream in densification, not in the mesh policy");
				}
			}

			// DENSEST CONTIGUOUS BAND. This is both the measurement and the basis of the
			// fix, so it is computed rather than eyeballed off the ladder.
			//
			// On a gravity-aligned aerial survey the terrain is a THIN slab in z, while
			// bad depth estimates smear points along camera rays over hundreds of metres.
			// That smear is CONTINUOUS, so neither a percentile (it survived 0.2/99.8 with
			// z still at 1070) nor a gap test (there is no void to find) separates it. What
			// does separate it is density: the terrain band holds almost every point in a
			// small fraction of the range, and the smear holds almost none over the rest.
			//
			// Slide a window over a histogram and report the NARROWEST band containing
			// kBandFrac of the points. Self-limiting by construction -- if the data really
			// does span the full range, the band is the full range and there is nothing to
			// cut, so this cannot damage a scene that does not have the problem.
			{
				constexpr int   kBins     = 2048;
				constexpr double kBandFrac = 0.995;
				const float lo = pctOf(dom, 0.0), hi = pctOf(dom, 1.0);
				if (hi > lo && dom.size() > 1000) {
					std::vector<uint32_t> hist(kBins, 0);
					const double invW = (double)kBins / ((double)hi - (double)lo);
					for (float v : dom) {
						int b = (int)(((double)v - (double)lo) * invW);
						if (b < 0) b = 0; else if (b >= kBins) b = kBins - 1;
						++hist[(size_t)b];
					}
					const uint64_t need = (uint64_t)(kBandFrac * (double)dom.size());
					int bestLo = 0, bestHi = kBins - 1;
					uint64_t acc = 0;
					int l = 0;
					for (int r = 0; r < kBins; ++r) {
						acc += hist[(size_t)r];
						while (acc - hist[(size_t)l] >= need) { acc -= hist[(size_t)l]; ++l; }
						if (acc >= need && (r - l) < (bestHi - bestLo)) { bestLo = l; bestHi = r; }
					}
					const double binW = ((double)hi - (double)lo) / (double)kBins;
					const double bandLo = (double)lo + bestLo * binW;
					const double bandHi = (double)lo + (bestHi + 1) * binW;
					const double bandSpan = bandHi - bandLo;
					const double rawSpan = (double)hi - (double)lo;
					MESH_DIAG("[MESH-EXTENT] densest band holding %.3f of points on axis %c:"
						" [%.6g .. %.6g] span %.4g of raw %.4g -> cube could shrink x%.2f"
						" (cell and every downstream size improve by the same factor)",
						kBandFrac, (domAxis == 0) ? 'x' : ((domAxis == 1) ? 'y' : 'z'),
						bandLo, bandHi, bandSpan, rawSpan,
						(bandSpan > 0.0) ? rawSpan / bandSpan : 1.0);
				}
			}
			MESH_DIAG("[MESH-EXTENT] dominant axis %c ladder:"
				" p0=%.4g p1=%.4g p5=%.4g p25=%.4g p50=%.4g p75=%.4g p95=%.4g p99=%.4g p100=%.4g"
				" | core p5-p95 spans %.4g of the raw %.4g",
				(domAxis == 0) ? 'x' : ((domAxis == 1) ? 'y' : 'z'),
				pctOf(dom, 0.0), pctOf(dom, 0.01), pctOf(dom, 0.05), pctOf(dom, 0.25),
				pctOf(dom, 0.50), pctOf(dom, 0.75), pctOf(dom, 0.95), pctOf(dom, 0.99),
				pctOf(dom, 1.0),
				pctOf(dom, 0.95) - pctOf(dom, 0.05),
				(domAxis == 0) ? (maxx - minx) : ((domAxis == 1) ? (maxy - miny) : (maxz - minz)));
		}
	}

	// 2) median nearest-neighbour spacing on a deterministic stride-sampled subset,
	// each queried against the FULL cloud (so the spacing is the true local one).
	// The kd-tree build dominates; the per-query 1-NN lookups are independent.
	using namespace nanoflann;
	using KDTree = KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<float, PointCloudAdapter>, PointCloudAdapter, 3>;
	PointCloudAdapter cloud{ pts, numPoints };
	KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(64));
	index.buildIndex();

	const size_t kSamples = std::min<size_t>(numPoints, 50000);
	const size_t stride = std::max<size_t>(1, numPoints / kSamples);
	const size_t nq = (numPoints + stride - 1) / stride;
	std::vector<float> spacing(nq, -1.f);
	float* __restrict pSp = spacing.data();
#ifdef _USE_OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (ptrdiff_t q = 0; q < (ptrdiff_t)nq; ++q) {
		const size_t i = (size_t)q * stride;
		uint32_t idx[2]; float d2[2];
		const size_t found = index.knnSearch((const float*)&pts[i], 2, idx, d2);
		if (found >= 2 && d2[1] > 0.f)
			pSp[q] = FastSqrtS(d2[1]); // d2[0] is the query point itself
	}
	std::vector<float> valid;
	valid.reserve(nq);
	for (size_t q = 0; q < nq; ++q)
		if (pSp[q] > 0.f) valid.push_back(pSp[q]);
	if (valid.empty())
		return finish(0);
	std::nth_element(valid.begin(), valid.begin() + valid.size() / 2, valid.end());
	const float s = valid[valid.size() / 2];
	if (!(s > 0.f))
		return finish(0);
	pol.spacing = s;

	// Data-limited ceiling: the depth whose finest cell matches the measured point
	// spacing. finish() takes min(this, budget-limited).
	const int depthData = (int)std::lround(std::log2((scaleFactor * ext) / s));
	return finish(depthData);
} // ComputeMeshPolicy
/*----------------------------------------------------------------*/

// Back-compatible thin wrapper: the depth alone, with default budget/roughness.
// Prefer ComputeMeshPolicy() at call sites that can also consume cell/tiles/
// refineLevel -- those are what let RefineMesh and TextureMesh agree with the mesh
// instead of each picking its own resolution.
static int EstimatePoissonDepth(const float* ptsRaw, size_t numPoints,
	float scaleFactor = 1.1f, int minDepth = 8, int maxDepth = 14)
{
	return ComputeMeshPolicy(ptsRaw, numPoints, scaleFactor,
		0.f, 0.0, 0.f, 0.f, minDepth, maxDepth).depth;
}
/*----------------------------------------------------------------*/

// Tier 2 Poisson surface reconstruction via Kazhdan's PoissonRecon +
// SurfaceTrimmer (external tools, shell-out). This avoids CGAL's deprecated
// Surface_mesher (which fails to instantiate under MSVC) and adds the density
// TRIMMING the CGAL convenience function lacks — so on an open scene it yields
// the trimmed, complete, competitor-style surface instead of a closed balloon.
//
// PoissonRecon.exe and SurfaceTrimmer.exe must sit next to the running
// executable, or in the folder named by the OPENMVS_POISSONRECON_DIR environment
// variable. Build them once from the bundled repo at <tree>/PoissonRecon
// (AdaptiveSolvers.sln, MIT-licensed).
//   depth          : octree depth (detail). <=0 => auto-select from the cloud's
//                    own point density (EstimatePoissonDepth); ~11 for aerial.
//   trimThreshold  : SurfaceTrimmer density threshold (>0 trims the low-confidence
//                    extrapolated balloon; 0 disables trimming). ~7 typical.
//   samplesPerNode : PoissonRecon smoothing (higher = smoother). default 1.5.
//   pointWeight    : screened-Poisson interpolation weight. default 2.
//   islandRatio    : SurfaceTrimmer --aRatio (+ --removeIslands); after trimming,
//                    delete isolated components whose area is below this fraction
//                    of the whole mesh (removes small floating blobs). 0 (default)
//                    = OFF, i.e. the tool's stock behavior (native aRatio 0.001).
//   releasePointCloud : drop the dense cloud once the solve has what it needs.
//                    The resident cloud is 43 B/point fixed (xyz 12 + normals 12
//                    + color 3 + view off/size 8 + weight off/size 8) plus
//                    8 B/point per average view -- ~75 B/point at 4 views -- but
//                    the solve only ever reads xyz and normals. Keeping just
//                    those (24 B/point, moved out, not copied) frees ~51 B/point:
//                    measured 2.6 GB of a 5.67 GB peak on a 50.6M-point cloud.
//                    SIDE EFFECT, and the reason this is optional: the caller's
//                    scene.Save() will write 0 points instead of the full cloud.
//                    The MESH is unaffected. RefineMesh reads the cloud only via
//                    SelectNeighborViews, which is skipped when Image::neighbors
//                    is already populated (it is serialized, and ReconstructMesh
//                    fills it before calling this); TextureMesh never reads it.
//                    Pass false if anything downstream reads points back out of
//                    the reconstructed scene file.
bool Scene::ReconstructMeshPoisson(int depth, float trimThreshold, float samplesPerNode, float pointWeight, float islandRatio, bool releasePointCloud)
{
	TD_TIMER_STARTD();
	ASSERT(!pointcloud.IsEmpty());
	mesh.Release();
	VERBOSE("Mesh reconstruction (OpenMVS-bmg build %d)", OPENMVS_BMG_BUILD);

	// Poisson requires oriented normals; estimate them if the cloud has none.
	if (!pointcloud.NormalStream()) {
		VERBOSE("Estimating point normals...");
		// Larger neighborhood (32 vs default 16) yields smoother, more stable
		// PCA normals on flat surfaces, reducing Poisson waviness (e.g. bumpy road).
		EstimatePointNormals(images, pointcloud, 32);
		if (!pointcloud.NormalStream()) {
			VERBOSE("error: mesh reconstruction requires point normals; estimation failed");
			return false;
		}
	}

	// Gather the finite oriented points for Poisson FIRST, so that
	// EstimatePoissonDepth (when depth<=0) operates on clean data without
	// per-point ISFINITE branches or NaN-polluted kd-tree queries.
	const float* poissonPts;
	const float* poissonNrm;
	size_t poissonCount;
	std::vector<float> pts, nrm; // only populated if filtering is needed
	{
		const size_t numPoints = pointcloud.NumPoints();
		const float* __restrict pPts = pointcloud.PointStream();
		const float* __restrict pNrm = pointcloud.NormalStream();

		// Fast check: are ALL points finite? (parallel, early-exit per thread impossible
		// with OMP reduction, but the branch-free check is cheap on 18M points ~2ms)
		ptrdiff_t numBad = 0;
#ifdef _USE_OPENMP
		#pragma omp parallel for schedule(static) reduction(+:numBad)
#endif
		for (ptrdiff_t i = 0; i < (ptrdiff_t)numPoints; ++i) {
			const float* __restrict p = pPts + i*3;
			const float* __restrict n = pNrm + i*3;
			if (!(ISFINITE(p[0]) && ISFINITE(p[1]) && ISFINITE(p[2]) &&
			      ISFINITE(n[0]) && ISFINITE(n[1]) && ISFINITE(n[2])))
				++numBad;
		}

		if (numBad == 0) {
			// All finite — pass streams directly, zero copy.
			poissonPts = pPts;
			poissonNrm = pNrm;
			poissonCount = numPoints;
		} else {
			// Some non-finite — filter into contiguous arrays.
			VERBOSE("Skipping %u/%u points with non-finite position/normal",
				(unsigned)numBad, (unsigned)numPoints);
			std::vector<uint32_t> validIdx;
			validIdx.reserve(numPoints - numBad);
			for (size_t i = 0; i < numPoints; ++i) {
				const float* __restrict p = pPts + i*3;
				const float* __restrict n = pNrm + i*3;
				if (ISFINITE(p[0]) && ISFINITE(p[1]) && ISFINITE(p[2]) &&
				    ISFINITE(n[0]) && ISFINITE(n[1]) && ISFINITE(n[2]))
					validIdx.push_back((uint32_t)i);
			}
			const size_t numValid = validIdx.size();
			if (numValid == 0) {
				VERBOSE("error: no finite oriented points for mesh reconstruction");
				return false;
			}
			pts.resize(numValid * 3);
			nrm.resize(numValid * 3);
			float* __restrict pP = pts.data();
			float* __restrict pN = nrm.data();
			const uint32_t* __restrict pIdx = validIdx.data();
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (ptrdiff_t j = 0; j < (ptrdiff_t)numValid; ++j) {
				const size_t i = (size_t)pIdx[j];
				const float* __restrict p = pPts + i*3;
				const float* __restrict n = pNrm + i*3;
				pP[(size_t)j*3+0]=p[0]; pP[(size_t)j*3+1]=p[1]; pP[(size_t)j*3+2]=p[2];
				pN[(size_t)j*3+0]=n[0]; pN[(size_t)j*3+1]=n[1]; pN[(size_t)j*3+2]=n[2];
			}
			poissonPts = pts.data();
			poissonNrm = nrm.data();
			poissonCount = numValid;
		}
	}

	// auto-select the mesh resolution policy when the caller passes depth <= 0 --
	// let the data, not a fixed guess, set the detail. Uses the already-filtered
	// poissonPts (all finite, no NaN in kd-tree).
	//
	// Only `depth` is consumed here. The policy also derives the cell size, whether
	// the requested cell needs tiled meshing, and the RefineMesh resolution-level
	// that MATCHES this mesh -- refining a 26 cm surface against full-resolution
	// imagery streams tens of GB to adjust geometry that cannot represent it. Those
	// are logged for the orchestrator to pick up until the later stages read them
	// directly; see the [MESH-POLICY] lines.
	// meshCell is also needed by the input subsample below, so it is resolved on
	// both paths -- via the policy when auto-selecting, or from a cheap bounding-box
	// pass (no kd-tree) when the caller pinned the depth.
	float meshCell = 0.f;
	// Octree cube scale handed to PoissonRecon. 1.1 (its CLI default) unless the policy
	// padded the cube to unlock a level -- see POISSON_CUBE_PAD_MAX. Must reach every
	// Reconstruct() call, the probe included, or the probe measures a different cube than
	// the solve and its k comes out wrong.
	float poissonScale = 1.1f;
	// Probe's predicted k, kept so the post-solve check can score the PROBE rather than
	// re-deriving a depth under a single ceiling that may not have been the binding one.
	double probeK = 0.0;
	if (depth <= 0) {
		const float gsd = EstimateSceneGSD(images, poissonPts, poissonCount);
		if (!(gsd > 0.f))
			MESH_DIAG("[MESH-POLICY] warning: could not measure GSD (no valid poses?);"
				" refineLevel will be reported as unknown");
		// Views and RefineMesh affordability, both needed BEFORE the depth is chosen.
		// The whole loaded cloud is released before RefineMesh/TextureMesh run (they are
		// separate processes), so add it back when sizing their budgets.
		const size_t cloudBytesFreed =
			(size_t)((double)poissonCount * POISSON_CLOUD_LOADED_BYTES_PER_POINT);
		size_t nViewsValid = 0;
		double totalViewPixels = 0.0, refineAllowBytes = 0.0;
		const int lvlAfford = RefineFinestAffordableLevel(
			images, cloudBytesFreed, nViewsValid, totalViewPixels, refineAllowBytes);
		// TextureMesh is bounded by two independent resources, and either can be the
		// tighter one: RAM (scales with VIEW COUNT -- observations per face) and the
		// atlas (scales with nothing -- a pure hardware ceiling). A 109-view scene on a
		// big box is atlas-bound; a 1940-view scene is RAM-bound. Take the min so the
		// same build is correct on both without knowing which it is.
		const double texCapMem   = ComputeTextureFaceBudget(nViewsValid, cloudBytesFreed);
		// PreRefine, matching what the decimation cap in ReconstructMesh.cpp will actually
		// apply -- otherwise this line's "-bound at decimation" verdict names a figure the
		// decimation does not use, which is exactly the sort of two-numbers-for-one-thing
		// confusion the shared resolver was introduced to end.
		const double texCapAtlas = ComputeAtlasFaceBudgetPreRefine(0);
		// REPORT-ONLY here, deliberately. The atlas cap is a face count, and turning a
		// face count into a DEPTH goes through k -- which is the least reliable number in
		// this file. MEASURED: applying it to depthTexture drops the 1251 m / 227-view
		// scene from depth 12 to 11 (cell 0.336 -> 0.672 m), purely because its k is
		// estimated at 2.118 against a true 0.474. The cap itself is right (that scene's
		// final mesh is 2.66M against a 4.07M atlas budget, comfortably inside); the k it
		// would be divided by is not.
		//
		// So the atlas bound is enforced where the face count is known EXACTLY and no k
		// is involved -- the decimation cap in ReconstructMesh.cpp, which sees the solved
		// mesh. Same ceiling, applied where it cannot be corrupted by an estimate.
		// Revisit binding it here once k prediction is trustworthy.
		const double texCap = texCapMem;
		MESH_DIAG("[MESH-ATLAS] texture face cap: memory %.1fM (%u views) vs atlas %.1fM"
			" (%d px, %.0f texels/face) -> %s-bound at decimation"
			" | depth ceiling uses memory only (k too noisy to divide by)",
			texCapMem * 1e-6, (unsigned)nViewsValid, texCapAtlas * 1e-6,
			ResolveAtlasMaxDim(0), (double)POISSON_ATLAS_TEXELS_PER_FACE,
			(texCapAtlas < texCapMem) ? "ATLAS" : "RAM");

		// How much subdivision headroom to leave RefineMesh.
		//
		// The two stages can both produce detail, but not equally well: RefineMesh
		// splits only faces whose PROJECTED area exceeds the threshold, so it adds
		// geometry where the imagery supports it, while Poisson at a finer depth adds it
		// everywhere -- including where there is nothing but noise. At equal face counts
		// adaptive wins.
		//
		// MEASURED on a 276 m / 368-view scene: depth 10 + heavy RefineMesh subdivision
		// and depth 11 + almost none produced visually indistinguishable results, but the
		// depth-11 route cost 5x the faces, pushed atlas padding from ~1.3x to ~1.75x
		// (158 -> 277 Mpx against ~223 Mpx capacity, i.e. from fitting to downscaling),
		// and took 5 minutes longer. So hand the work to RefineMesh whenever it will run.
		//
		// The test is non-circular because affordability does not depend on the mesh. At
		// headroom H the cell is ~H x the reliability floor, so the signal reaches level
		// log2(H); with H = 4 that is level 2. If the finest affordable level is within
		// that, RefineMesh will have both signal and memory at the coarse depth.
		const double refineHeadroom = (lvlAfford <= 2)
			? POISSON_REFINE_HEADROOM_ADAPTIVE   // RefineMesh will run: build coarse
			: POISSON_REFINE_HEADROOM;           // it cannot: Poisson must do the work

		MeshPolicy pol = ComputeMeshPolicy(poissonPts, poissonCount,
			1.1f,   // scale: must match PoissonReconLib's octree cube scale
			0.f,    // targetCell: 0 = best cell reachable in one untiled mesh
			0.0,    // facesBudget: 0 = auto from available RAM
			0.f,    // faceDensityK: 0 = auto (probe below), else default
			gsd, 8, 14, 0.f, 0.f,
			texCap, refineHeadroom);

		// CALIBRATION PROBE. The face-density coefficient k (terrain roughness) is a
		// per-dataset property and CANNOT be derived from the cloud: median nearest-
		// neighbour spacing under-predicts true surface area by a dataset-dependent
		// factor, because multi-view fusion leaves near-coincident points (measured
		// 3.4x low on a 974 m aerial site). So measure it: reconstruct once at a
		// coarse depth, read k off the face count, then re-decide.
		//
		//     faces = k * (2^depth / scale)^2   =>   k = faces * scale^2 / 4^depth
		//
		// Probe depth is target-2 rather than something fixed, to keep the
		// extrapolation short. MEASURED direction of the drift: k comes out HIGHER at
		// coarse depth (1.575 at depth 9 vs a true 1.046 at depth 12 on the 974 m
		// site), because a coarse octree's face count is inflated by bounding-box and
		// boundary structure relative to (2^depth/scale)^2. That over-prediction
		// selects a SHALLOWER depth, so the error is in the safe direction -- but it
		// does cost resolution, which is why the offset is kept small. The re-decide
		// reuses the already-measured extent/spacing, so it costs no second kd-tree.
#if POISSON_CALIBRATION_PROBE
		// Below this the probe costs more than the information is worth, and a
		// shallow scene's k is dominated by the bounding box rather than terrain.
		constexpr int kProbeFloor = 9;

		// SECOND gate: can a better k change the answer at all?
		//
		// depth = min(budget, data, texture, pixels), and only budget and texture are
		// functions of k. The default k is deliberately HIGH (see PoissonFaceDensityK),
		// so measuring it almost always LOWERS k, which RAISES those two ceilings. If
		// they are already at or above the k-independent pair, raising them further
		// cannot move the minimum -- the probe is pure cost.
		//
		// Strictly-less, so a TIE skips: a tie means the k-independent ceiling already
		// caps the depth, and raising a co-binding ceiling changes nothing.
		//
		// MEASURED, 3 for 3 -- the probe changed no depth and cost 15-60 s each time:
		//   aerial 1940 views  budget=12 data=15 texture=12 pixels=12  (12 vs 12, tie)
		//   1251 m 227 views   budget=12 data=13 texture=12 pixels=12  (12 vs 12, tie)
		//   ...and on the second the probe was wrong by +347%, printing a MISPREDICTED
		//   warning about a number that could not have mattered.
		//
		// Residual risk: if the true k is HIGHER than the default, these ceilings drop
		// instead of rise and could become binding unnoticed. Bounded and small -- the
		// measured range across every dataset is 0.47 to 3.7 against a 3.0 default, so
		// the worst case is 23% more faces than budgeted at the chosen depth, well
		// inside POISSON_MEMORY_FRACTION. Dropping a whole depth level would take a 4x
		// error in k, which has never been observed.
		const int kDepDepth   = std::min(pol.depthBudget, pol.depthTexture);
		const int kIndepDepth = std::min(pol.depthData,   pol.depthPixels);
		const bool kCanBind   = kDepDepth < kIndepDepth;
		if (!kCanBind)
			MESH_DIAG("[MESH-CALIB] probe skipped: depth is set by the %s ceiling (%d),"
				" which does not depend on k (budget=%d texture=%d)",
				(pol.depthPixels <= pol.depthData) ? "pixel" : "data",
				kIndepDepth, pol.depthBudget, pol.depthTexture);

		if (kCanBind && pol.depth > kProbeFloor) {
			const int probeDepth = std::max(kProbeFloor, pol.depth - 2);
			PoissonReconLib::Mesh probe;
			PoissonReconLib::ReconParams rp;
			rp.depth          = probeDepth;
			// Same cube as the real solve, or k is measured against a different octree
			// than the one it will be used to predict.
			rp.scale          = pol.scale;
			rp.samplesPerNode = samplesPerNode;
			rp.pointWeight    = pointWeight;
			rp.density        = false;   // not needed; skips a pass
			rp.verbose        = false;
			MESH_DIAG("[MESH-CALIB] probing at depth %d to measure k...", probeDepth);
			if (PoissonReconLib::Reconstruct(poissonPts, poissonNrm, poissonCount, rp, probe)
				&& probe.TriangleCount() > 0)
			{
				// Must be the cube the probe actually ran with, not the 1.1 default: k is
				// defined by faces ~= k*(scale*ext/cell)^2, so a padded cube shifts it.
				const double sc2   = (double)pol.scale * (double)pol.scale;
				const double kRaw  = (double)probe.TriangleCount() * sc2
					/ std::pow(4.0, (double)probeDepth);
				double kMeas = kRaw;

				// SLOPE PROBE -- measurement only, does NOT change kMeas.
				//
				// POISSON_K_DEPTH_DECAY is one constant calibrated on one site, and the
				// required value spans 7x across datasets: 0.10/level on the dense 974 m
				// aerial set (263M points), 0.75/level on the sparse 1251 m Marco Ulises
				// set (3.09M points), where the probe overshot by +347%. The obvious fix
				// -- probe twice and use the measured slope -- is NOT safe as written,
				// which is why this only reports:
				//
				//   lambda is not constant WITHIN a dataset either; it falls with depth.
				//   From the four measurements above: 9->10 gives 0.200, 10->12 gives
				//   0.105, 12->13 gives 0.081. Two SHALLOW probes therefore measure the
				//   STEEPEST slope and over-decay when extrapolated forward. On the aerial
				//   set that predicts k=1.055 against a true ~1.161 (-9.2%), versus the
				//   current constant's +11.1% -- same magnitude, but flipped to the UNSAFE
				//   direction, since under-predicting k allows a DEEPER solve than the
				//   budget intended.
				//
				// So measure lambda across datasets first and decide with data. The second
				// probe is one level shallower, hence ~1/4 the cost of the first (~+25% on
				// probe time, measured ~5 s on a 21 s aerial probe).
				//
				// What to look for: if lambda_measured tracks the value that would have
				// predicted the true k (reported by the [MESH-CALIB] solve line), the slope
				// is usable. If it consistently over-decays on dense scenes, the mechanism
				// is depth-dependent and needs a convex model (3 probes) or a
				// points-per-cell predictor -- lambda correlates with octree occupancy:
				// ~1214 points/cell on the aerial (lambda 0.12) vs ~3.6 on Marco (0.85).
#if POISSON_K_SLOPE_PROBE
				if (probeDepth - 1 >= 6) {
					PoissonReconLib::Mesh probe2;
					PoissonReconLib::ReconParams rp2 = rp;
					rp2.depth = probeDepth - 1;
					if (PoissonReconLib::Reconstruct(poissonPts, poissonNrm, poissonCount, rp2, probe2)
						&& probe2.TriangleCount() > 0)
					{
						const double kRaw2 = (double)probe2.TriangleCount() * sc2
							/ std::pow(4.0, (double)rp2.depth);
						if (kRaw2 > 0.0 && kRaw > 0.0) {
							const double lambda = std::log(kRaw2 / kRaw); // per level, d-1 -> d
							const int off = pol.depth - probeDepth;
							MESH_DIAG("[MESH-SLOPE] k(%d)=%.3f k(%d)=%.3f -> lambda=%.3f/level"
								" (assumed %.3f) | at depth %d predicts %.3f vs assumed-constant %.3f"
								" | %.0f points/cell at probe depth | MEASUREMENT ONLY",
								rp2.depth, kRaw2, probeDepth, kRaw, lambda,
								(double)POISSON_K_DEPTH_DECAY, pol.depth,
								kRaw * std::exp(-lambda * (double)off),
								kRaw * std::exp(-POISSON_K_DEPTH_DECAY * (double)off),
								(double)poissonCount * sc2 / std::pow(4.0, (double)probeDepth));
						}
					} else {
						MESH_DIAG("[MESH-SLOPE] second probe failed; no slope measured");
					}
				}
#endif
				// Correct for k's drift with depth. MEASURED on the 974 m site:
				//   depth  9 -> 1.575
				//   depth 10 -> 1.289
				//   depth 12 -> 1.046
				//   depth 13 -> 0.965
				// i.e. ln(k) falls by ~0.1 per level over the useful range, because a
				// coarse octree's face count carries proportionally more bounding-box
				// and boundary structure. Extrapolating the probe forward turns a 23-33%
				// over-prediction into ~1%: 1.289 * exp(-0.2) = 1.056 vs a true 1.046.
				// Without this the probe reliably costs a whole depth level.
				const int probeOffset = pol.depth - probeDepth;
				if (POISSON_K_DEPTH_DECAY > 0.0 && probeOffset > 0)
					kMeas *= std::exp(-POISSON_K_DEPTH_DECAY * (double)probeOffset);
				if (kMeas > 0.0) {
					probeK = kMeas;
					const MeshPolicy pol2 = ComputeMeshPolicy(poissonPts, poissonCount,
						1.1f, 0.f, 0.0, (float)kMeas, gsd, 8, 14,
						pol.extent, pol.spacing,    // reuse measurements
						texCap, refineHeadroom);
					MESH_DIAG("[MESH-CALIB] probe faces=%u -> k=%.3f (assumed %.3f);"
						" depth %d -> %d, cell %.4g -> %.4g",
						(unsigned)probe.TriangleCount(), kMeas, pol.k,
						pol.depth, pol2.depth, pol.cell, pol2.cell);
					pol = pol2;
				}
			} else {
				MESH_DIAG("[MESH-CALIB] probe failed; keeping the assumed k");
			}
		}
#endif
		// Is RefineMesh worth running on the mesh we are about to build?
		//
		// It only adds GEOMETRY where --max-face-area triggers subdivision, i.e.
		// where its subdivision floor falls below the mesh cell:
		//     floor(level) = sqrt(maxFaceArea) * GSD * 2^level
		// Below that threshold it merely nudges vertex positions -- a small gain for
		// a large bill. Meanwhile its cost is driven by VIEW COUNT, not face count:
		// every iteration streams nViews * imagePixels * bytes/px / 4^level.
		//
		// So the decision is whether the deepest level that still subdivides is also
		// affordable. NOT whether the depth is some particular number -- depth is
		// scene-relative, and the same depth means millimetre cells on an object scan
		// and decimetre cells on an aerial site.
		bool refineRun = false;
		if (gsd > 0.f && pol.cell > 0.f && !images.IsEmpty()) {
			const double totalPixels = totalViewPixels;
			const size_t nValid = nViewsValid;
			if (nValid > 0 && totalPixels > 0.0) {
				// Coarsest level whose working GSD is still FINER than the mesh cell,
				// i.e. the level beyond which there is no sub-face image detail left to
				// refine against.
				//
				// NOTE this was originally the level at which --max-face-area triggers
				// SUBDIVISION, which was too strict: RefineMesh's main job is
				// photoconsistency-driven vertex refinement -- flattening noise on
				// planar surfaces and sharpening real edges -- and that works whether or
				// not any face gets split. Keying on subdivision skipped the stage on a
				// 109-view scene where it cost ~10 GB and would have fixed exactly the
				// lumpy-road / melted-edge artefacts it was meant to prevent.
				// The floor is sqrt(--max-face-area) pixels per face, not 1: vertex
				// refinement needs texture INSIDE a face to align against, and it is
				// refining against the very images that produced the depth maps Poisson
				// already fitted. MEASURED, both datasets sit below it -- 3.7 px/face on
				// the 109-view residential scene, 2.3 px at the affordable level on the
				// 1940-view aerial one -- so both correctly SKIP. An earlier revision
				// used a 1-pixel floor and turned those into runs that could not have
				// helped. Coarse mesh + fine imagery (tens of px/face) is the regime this
				// stage exists for.
				const int lvlSignal = (int)std::floor(
					std::log2((double)pol.cell / (POISSON_REFINE_SUBDIV_FACTOR * (double)gsd)));
				// Affordability was already resolved above (it does not depend on the
				// mesh) and fed into the headroom choice.
				const double allow = refineAllowBytes;
				refineRun = (lvlAfford <= lvlSignal);
				// Run at the level that MATCHES the mesh, not the finest affordable one.
				// "Finest affordable" cost 4 minutes on a 368-view scene for no visible
				// change: level 1 instead of the cell-matched level 2 quadruples the view
				// data for detail the mesh cannot express. Raise to lvlAfford only when
				// memory forces it coarser.
				if (refineRun) {
					const int lvlMatched = (int)std::lround(
						std::log2((double)pol.cell / (8.0 * (double)gsd)));
					pol.refineLevel = std::clamp(std::max(lvlMatched, lvlAfford), 0, 4);
				}
				const double bytesAtAfford =
					totalPixels * POISSON_REFINE_BYTES_PER_PIXEL / std::pow(4.0, (double)lvlAfford);
				// MPix/view is printed because it is NOT redundant with the scene-load
				// banner: that banner reports whatever resolution the dense scene stored,
				// while RefineMesh reloads from the ORIGINAL files. A 16x gap between the
				// two (densify --resolution-level 2, i.e. Image::scale 1/4) is invisible
				// otherwise, and it moves lvlAfford by two whole levels.
				MESH_DIAG("[MESH-REFINE] cell=%.4g gsd=%.4g views=%u @ %.2f MPix headroom=%.1f"
					" | signal down to level %d | finest affordable level %d"
					" (%.1f GB of %.1f GB allowed) => %s at level %d",
					pol.cell, gsd, (unsigned)nValid,
					totalPixels / (1.0e6 * (double)nValid), refineHeadroom, lvlSignal,
					lvlAfford, bytesAtAfford * 1e-9, allow * 1e-9,
					refineRun ? "RUN RefineMesh" : "SKIP RefineMesh (affordable level has no signal)",
					pol.refineLevel);
			}
		}

		// MACHINE-READABLE PLAN -- the hand-off across the process boundary. This tool
		// decides the detail settings from the data; the stages after it (RefineMesh,
		// TextureMesh) need those decisions, and they run as separate processes.
		//
		// Written to a FILE rather than left in the log: the log's name, location and
		// verbosity all depend on how we were invoked, so parsing it is fragile. The
		// working folder is passed explicitly by the caller (--working-folder), so
		// MAKE_PATH gives a path both sides can compute without agreeing on anything
		// else. Keep the key names and ordering stable.
		//
		// Single file per working folder: with scene clustering (several bins meshed
		// into one folder) the last bin's plan wins. The values are dataset-level
		// properties, so that is acceptable -- but it is why the consumer should read
		// it once after the reconstruct loop rather than per bin.
		{
			char planLine[512];
			snprintf(planLine, sizeof(planLine),
				"[MESH-PLAN] depth=%d cell=%.6g tiles=%d gsd=%.6g refine=%s"
				" refine_level=%d texture_level=0\n",
				pol.depth, (double)pol.cell, pol.tilesPerAxis, (double)gsd,
				refineRun ? "RUN" : "SKIP", pol.refineLevel);
			// The FILE is the contract with the orchestrator and is written either way.
			// The log copy is not: it spells out the octree depth, the cell size and the
			// downstream stage plan, which is exactly the internal detail the default log
			// should not carry. Read mesh_plan.txt, or open the gate.
			MESH_DIAG("%s", planLine);
			const String planPath(MAKE_PATH("mesh_plan.txt"));
			std::ofstream planOut(planPath.c_str(), std::ios::out | std::ios::trunc);
			if (planOut)
				planOut << planLine;
			else
				VERBOSE("warning: could not write the mesh plan to %s", planPath.c_str());
		}

		depth        = pol.depth;
		meshCell     = pol.cell;
		poissonScale = pol.scale;
	} else {
		meshCell = PoissonCellSizeAtDepth(poissonPts, poissonCount, poissonScale, depth);
	}

	// Compact to xyz+normals only, then release the full cloud before the solve
	// (see releasePointCloud in the header comment above). Done AFTER
	// EstimatePoissonDepth, which reads poissonPts. pts/nrm outlive the solve,
	// so poissonPts/poissonNrm stay valid for the adaptive-trim path below.
	if (releasePointCloud) {
		if (pts.empty()) {
			// Zero-copy path: poissonPts/poissonNrm still alias the cloud's
			// streams. MOVE the two geometry arrays out of the cloud -- an O(1)
			// vector steal, not a 24 B/point copy -- so this costs no time and
			// adds no transient peak. The moved-from vectors are left empty,
			// which Release() below handles fine.
			pts = std::move(pointcloud.pointsXYZ);
			nrm = std::move(pointcloud.normalsXYZ);
			poissonPts = pts.data();
			poissonNrm = nrm.data();
		}
		// else: the non-finite filter already built pts/nrm as filtered copies
		// that do not alias the cloud -- nothing to move.
		pointcloud.Release();
		MESH_DIAG("Poisson: released the dense point cloud (%u points kept as %u MB of xyz+normals)",
			(unsigned)poissonCount, (unsigned)((poissonCount * 6 * sizeof(float)) >> 20));
	}

	// Voxel-subsample the SOLVE INPUT only. The full poissonPts/poissonNrm stay
	// intact for the adaptive-trim and cull paths below, whose accuracy depends on
	// querying every real sample -- this trims only what PoissonRecon would merge
	// away itself. Voxel is derived from meshCell, so it scales with the dataset.
	std::vector<float> subPts, subNrm;
	const float* reconPts   = poissonPts;
	const float* reconNrm   = poissonNrm;
	size_t       reconCount = poissonCount;
	if (POISSON_INPUT_SUBSAMPLE &&
		meshCell > 0.f && poissonCount >= (size_t)POISSON_INPUT_SUBSAMPLE_MIN_POINTS) {
		const float voxel = meshCell / (float)(1u << POISSON_INPUT_SUBSAMPLE_SHIFT);
		const size_t kept = VoxelSubsampleOrientedPoints(
			poissonPts, poissonNrm, poissonCount, voxel, subPts, subNrm);
		// Require a real reduction AND a sane floor. A cloud that is not spatially
		// sorted keeps almost everything (see the function comment); in that case
		// discard the copy and solve on the full cloud rather than pay for both.
		if (kept >= (size_t)POISSON_INPUT_SUBSAMPLE_FLOOR &&
			kept <= poissonCount - poissonCount / 4)
		{
			reconPts = subPts.data(); reconNrm = subNrm.data(); reconCount = kept;
			MESH_DIAG("Poisson: solve input subsampled at voxel %.4g (cell/%u):"
				" %u -> %u points (%.1fx), %u MB",
				voxel, 1u << POISSON_INPUT_SUBSAMPLE_SHIFT,
				(unsigned)poissonCount, (unsigned)kept,
				(double)poissonCount / (double)std::max<size_t>(kept, 1),
				(unsigned)((kept * 6 * sizeof(float)) >> 20));
		} else {
			subPts.clear(); subPts.shrink_to_fit();
			subNrm.clear(); subNrm.shrink_to_fit();
			MESH_DIAG("Poisson: solve-input subsample rejected (kept %u of %u);"
				" cloud may not be spatially sorted -- using the full cloud",
				(unsigned)kept, (unsigned)poissonCount);
		}
	}

	// 1) In-process screened-Poisson reconstruction (replaces PoissonRecon.exe),
	// with per-vertex density so the trimmer can threshold by it.
	PoissonReconLib::Mesh pmesh;
	{
		PoissonReconLib::ReconParams rp;
		rp.depth          = depth;
		rp.scale          = poissonScale;
		rp.samplesPerNode = samplesPerNode;
		rp.pointWeight    = pointWeight;
		rp.density        = true;
		// The solver's own banner and per-level trace are the most detailed statement of
		// method anything in this stage emits, so they follow the same gate as our own
		// instrumentation rather than being unconditionally on.
		rp.verbose        = MESH_DIAG_ENABLED();
		VERBOSE("Reconstructing surface...");
		MESH_DIAG("Poisson: running PoissonRecon (depth=%d)...", depth);
		if (!PoissonReconLib::Reconstruct(reconPts, reconNrm, reconCount, rp, pmesh) || pmesh.TriangleCount() == 0) {
			VERBOSE("error: mesh reconstruction failed");
			return false;
		}
	}
	// The solve is done; the subsample is dead weight during trim/cull.
	subPts.clear(); subPts.shrink_to_fit();
	subNrm.clear(); subNrm.shrink_to_fit();

	// Self-calibration. k is a per-dataset property (terrain roughness) and cannot
	// be derived from the cloud -- median NN spacing under-predicts true surface
	// area by a dataset-dependent factor, because multi-view fusion leaves
	// near-coincident points. But it falls straight out of the RAW (pre-trim) face
	// count of the solve we just ran:
	//     faces = k * (2^depth / scale)^2   =>   k = faces * scale^2 / 4^depth
	// Report it, and the depth it would have selected, so the next run on this site
	// is calibrated instead of guessed. Pass it back via OPENMVS_POISSON_FACE_K.
	{
		// Cube actually solved, not the 1.1 default -- see the probe's sc2 above.
		const double sc2  = (double)poissonScale * (double)poissonScale;
		const double kMeas = (double)pmesh.TriangleCount() * sc2 / std::pow(4.0, (double)depth);
		if (kMeas > 0.0) {
			// Score the PROBE against the truth, not two depths derived under different
			// ceilings. The earlier version recomputed a depth from the budget ceiling
			// alone and flagged any disagreement -- but the budget is frequently NOT the
			// binding ceiling (pixels and texture usually bind first), so it reported
			// "PROBE MISPREDICTED" on runs where the probe was accurate to 2.5%. A
			// diagnostic that cries wolf is worse than none: the depth-10 clamp survived
			// for months behind exactly that kind of noise.
			//
			// A whole depth level is 4x in faces, hence 4x in k, so only a large relative
			// error can move the chosen depth. Flag on that, not on incidental drift.
			if (probeK > 0.0) {
				const double errPct = (probeK - kMeas) / kMeas * 100.0;
				MESH_DIAG("[MESH-CALIB] solve: raw faces=%u at depth=%d -> true k=%.3f;"
					" probe predicted %.3f (%+.1f%%)%s",
					(unsigned)pmesh.TriangleCount(), depth, kMeas, probeK, errPct,
					(std::fabs(errPct) > POISSON_PROBE_ERROR_WARN_PCT)
						? " -- PROBE MISPREDICTED" : "");
			} else {
				MESH_DIAG("[MESH-CALIB] solve: raw faces=%u at depth=%d -> true k=%.3f"
					" (no probe ran; set OPENMVS_POISSON_FACE_K=%.3f to use it)",
					(unsigned)pmesh.TriangleCount(), depth, kMeas, kMeas);
			}
		}
	}

	// 2) Optional in-process density trimming (replaces SurfaceTrimmer.exe;
	// removes the open-boundary balloon). Matches the previous tool invocation:
	// the trim threshold is always applied; aRatio defaults to the trimmer's
	// native 0.001 (which drives the trim-boundary component merge even without
	// island removal), and only when islandRatio>0 do we raise aRatio AND enable
	// removeIslands (which actually deletes the isolated small components).
#if !POISSON_ADAPTIVE_TRIM
	if (trimThreshold > 0.f) {
		PoissonReconLib::TrimParams tp;
		tp.trim          = trimThreshold;
		tp.aRatio        = (islandRatio > 0.f) ? islandRatio : 0.001f;
		tp.removeIslands = (islandRatio > 0.f);
		tp.verbose       = true;
		MESH_DIAG("Poisson: running SurfaceTrimmer (--trim %g%s)...", (double)trimThreshold,
			(islandRatio > 0.f) ? String::FormatString(" --aRatio %g --removeIslands", (double)islandRatio).c_str() : "");
		PoissonReconLib::Mesh tmesh;
		if (PoissonReconLib::Trim(pmesh, tp, tmesh) && tmesh.TriangleCount() > 0)
			pmesh = std::move(tmesh);
		else
			VERBOSE("warning: density trimming produced no output; using the untrimmed mesh");
	}
#endif // !POISSON_ADAPTIVE_TRIM

	// Copy the in-memory result into the MVS mesh (positions + triangles; the
	// per-vertex density is not needed by downstream stages).
	mesh.Release();
	{
		const size_t nv = pmesh.VertexCount(), nt = pmesh.TriangleCount();
		const float* __restrict pv = pmesh.vertices.data();
		const uint32_t* __restrict ptri = pmesh.triangles.data();

		// Vertices: source is stride-4 (x,y,z,density), dest is stride-3 (x,y,z).
		// Resize + parallel 12-byte memcpy per vertex (skips density float).
		mesh.vertices.Resize((Mesh::VIndex)nv);
		Mesh::Vertex* __restrict dst = mesh.vertices.GetData();
#ifdef _USE_OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (ptrdiff_t i = 0; i < (ptrdiff_t)nv; ++i)
			memcpy(&dst[i], pv + i * 4, 3 * sizeof(float));

		// DENSITY DISTRIBUTION -- logged on EVERY path, including Tier 2 / SurfaceTrimmer.
		//
		// The Poisson per-vertex density that --poisson-trim thresholds against is on an
		// ARBITRARY scale that shifts with octree depth and with the dataset, so a trim value
		// fitted on one scene or one depth is silently either a no-op or destructive. That
		// failure is invisible in the log, and it has already cost real debugging time on this
		// pipeline: a --poisson-trim 1 that sat BELOW THE ENTIRE DISTRIBUTION looked like a weak
		// trim for hours when it was structurally inert (13 faces of 103,291).
		//
		// Until now the percentile line existed only inside POISSON_ADAPTIVE_TRIM, which is
		// compiled out -- so the shipped configuration is exactly the one that cannot see this.
		// The data is present either way (pmesh is stride-4 x,y,z,density on every path), so
		// this is pure instrumentation: no threshold, no geometry, nothing else reads it.
		//
		// HOW TO READ IT ON THE TIER-2 PATH: SurfaceTrimmer has ALREADY run, so these are the
		// SURVIVING vertices. That makes `min` the diagnostic:
		//   min >~ trim  -> the trim was BINDING (it removed everything below the threshold)
		//   min <<  trim -> the threshold never applied; the surviving surface sits below it,
		//                   so the trim is inert on this scene at this depth and the number
		//                   needs re-picking from these percentiles, not nudging.
		// To cut roughly the lowest X% of surface, aim the threshold near pX.
		//
		// Pure instrumentation -- no threshold, no geometry, nothing else reads it --
		// so the 200k-sample gather and sort go behind the gate along with the line.
		if (nv > 0 && MESH_DIAG_ENABLED()) {
			std::vector<float> ds;
			const size_t dstride = std::max<size_t>(1, (size_t)nv / 200000);
			ds.reserve((size_t)nv / dstride + 1);
			for (size_t i = 0; i < (size_t)nv; i += dstride)
				ds.push_back(pv[i * 4 + 3]);
			if (!ds.empty()) {
				std::sort(ds.begin(), ds.end());
				const auto pct = [&](double q) {
					const size_t k = (size_t)(q * (double)(ds.size() - 1) + 0.5);
					return (double)ds[k];
				};
				// The label matters: this block sits BEFORE the adaptive trim but AFTER the
				// SurfaceTrimmer call, so which one it is depends on the build. Getting this
				// wrong sends you hunting a "min=3" that is simply the untrimmed minimum.
#if POISSON_ADAPTIVE_TRIM
				static const char* const kTrimStage = "PRE-trim, adaptive trim runs next";
#else
				static const char* const kTrimStage = "POST-trim";
#endif
				// LOW-TAIL LADDER as well as the coarse percentiles. The useful trim values live
				// BELOW p1 -- a field-tuned 5.75 on a depth-11 scene sat between min 3 and
				// p1 7.31 -- and p1 is far too coarse to calibrate
				// POISSON_TRIM_PERCENTILE_X100 against. These are the numbers to read when
				// matching a known-good raw threshold to a portable percentile.
				MESH_DIAG("Poisson: vertex density percentiles (n=%zu of %zu sampled, %s):"
					" min=%.3g | LOW TAIL p0.1=%.3g p0.2=%.3g p0.35=%.3g p0.5=%.3g p0.75=%.3g"
					" p1=%.3g p2=%.3g | p5=%.3g p25=%.3g p50=%.3g p75=%.3g p95=%.3g p99=%.3g"
					" max=%.3g | --poisson-trim %.2f -- this scale shifts with depth and"
					" dataset, so set the trim from THIS line; min far below the trim means"
					" the trim is inert here",
					ds.size(), (size_t)nv, kTrimStage, (double)ds.front(),
					pct(0.001), pct(0.002), pct(0.0035), pct(0.005), pct(0.0075),
					pct(0.01), pct(0.02),
					pct(0.05), pct(0.25), pct(0.50), pct(0.75),
					pct(0.95), pct(0.99), (double)ds.back(), (double)trimThreshold);
			}
		}

#if !POISSON_ADAPTIVE_TRIM
		// The Poisson vertex array (16 B/vertex: x,y,z,density) is dead once the
		// positions are copied out, so release it BEFORE allocating the face
		// array rather than holding both. swap-with-empty, not clear(): clear()
		// keeps the capacity allocated and frees nothing.
		// NOTE: deliberately not done under POISSON_ADAPTIVE_TRIM -- that path
		// reads pmesh.vertices (for per-vertex density) after this block, and
		// its guard tests VertexCount(), so freeing here would silently skip
		// the trim rather than fail loudly.
		// pv dangles after this point; it is not used again.
		std::vector<float>().swap(pmesh.vertices);
#endif

		// Faces: source is packed uint32_t triples, same layout as Mesh::Face.
		static_assert(sizeof(Mesh::Face) == 3 * sizeof(Mesh::VIndex), "Face layout mismatch");
		mesh.faces.Resize((Mesh::FIndex)nt);
		memcpy(mesh.faces.GetData(), ptri, nt * sizeof(Mesh::Face));
	}

#if POISSON_ADAPTIVE_TRIM
	// ADAPTIVE FOOTPRINT TRIM (replaces SurfaceTrimmer; see macro comment).
	// Runs HERE, while mesh vertex order still matches pmesh 1:1 (before the
	// NaN-sanitize compaction below), so per-vertex density can be read
	// directly from pmesh (stride-4: x,y,z,density).
	if (trimThreshold > 0.f && pmesh.VertexCount() > 0 && poissonCount >= 100 &&
	    (size_t)mesh.vertices.GetSize() == pmesh.VertexCount() && !mesh.faces.IsEmpty()) {
		TD_TIMER_STARTD();
		const Mesh::VIndex numV = mesh.vertices.GetSize();
		const float* __restrict pv = pmesh.vertices.data(); // stride-4: x,y,z,density
		const Point3f* __restrict cloud = reinterpret_cast<const Point3f*>(poissonPts);

		// --- median NN spacing of the input cloud (stride-sampled) ---
		float medianSpacing = 0.f;
		{
			using namespace nanoflann;
			using KDTree = KDTreeSingleIndexAdaptor<
				L2_Simple_Adaptor<float, PointCloudAdapter>, PointCloudAdapter, 3>;
			PointCloudAdapter pc{ cloud, poissonCount };
			KDTree kidx(3, pc, KDTreeSingleIndexAdaptorParams(64));
			kidx.buildIndex();
			const size_t kS = std::min<size_t>(poissonCount, 50000);
			const size_t st = std::max<size_t>(1, poissonCount / kS);
			const size_t nq = (poissonCount + st - 1) / st;
			std::vector<float> sp; sp.reserve(nq);
			for (size_t q = 0; q < nq; ++q) {
				const size_t i = q * st;
				uint32_t id2[2]; float d2[2];
				if (kidx.knnSearch((const float*)&cloud[i], 2, id2, d2) >= 2 && d2[1] > 0.f)
					sp.push_back(std::sqrt(d2[1]));
			}
			if (!sp.empty()) {
				std::nth_element(sp.begin(), sp.begin() + sp.size() / 2, sp.end());
				medianSpacing = sp[sp.size() / 2];
			}
		}

		if (medianSpacing > 0.f) {
			// --- XY footprint occupancy grid over the input cloud ---
			float minx = FLT_MAX, miny = FLT_MAX, maxx = -FLT_MAX, maxy = -FLT_MAX;
			for (size_t i = 0; i < poissonCount; ++i) {
				const Point3f& p = cloud[i];
				if (p.x < minx) minx = p.x;  if (p.x > maxx) maxx = p.x;
				if (p.y < miny) miny = p.y;  if (p.y > maxy) maxy = p.y;
			}
			const float cellSz = (POISSON_TRIM_CELL_FACTOR_X100 / 100.f) * medianSpacing;
			const float invCell = 1.f / cellSz;
			// Hard-cut margin, floored against the SCENE EXTENT so a dense scene (small
			// grid cell) cannot end up with a margin too tight to clear the balloon.
			// See POISSON_TRIM_MARGIN_FLOOR_FRAC for the nine-scene calibration.
			const float footExtent = std::max(maxx - minx, maxy - miny);
			const int marginCellsFloor =
				(POISSON_TRIM_MARGIN_FLOOR_FRAC > 0.f && cellSz > 0.f && footExtent > 0.f)
				? (int)std::ceil((double)POISSON_TRIM_MARGIN_FLOOR_FRAC
					* (double)footExtent / (double)cellSz)
				: 0;
			// ... and a CEILING in world units, for the opposite failure. The cell is 20x
			// median point spacing, so a sparsely-sampled scene gets a huge PHYSICAL margin
			// from the same cell count -- and extrapolation reach is a physical distance,
			// unrelated to how densely the cloud happens to be sampled.
			//
			// MEASURED margin over the corpus: Marco 10.46 units, then SchnellTests 4.96,
			// GEOTAG 4.81, MechanicFalls 4.54, everything else 0.51-2.09. Marco is 2.1x the
			// next scene and is the one reported as slop. A 7-unit cap takes it 3 -> 2 cells
			// (10.46 -> 6.97) and leaves all eight other scenes untouched, since their cells
			// are small enough that the cap lands above their count.
			//
			// 7 rather than 5 because Marco's own FLOOR is 0.006*829.2 = 4.98 units, which a
			// 5-unit cap would collide with. If slop persists the next lever is
			// POISSON_TRIM_CLOSE_CELLS (the seal: 8 cells = 27.9 units on Marco), not this.
			// Safe by construction: the escalating guard doubles the margin back up if a
			// tighter one would sever the surface.
			const int marginBase = std::max((int)POISSON_TRIM_MARGIN_CELLS, marginCellsFloor);
			const int marginCellsCap =
				(POISSON_TRIM_MARGIN_CAP_UNITS > 0.f && cellSz > 0.f)
				? std::max(1, (int)std::floor((double)POISSON_TRIM_MARGIN_CAP_UNITS / (double)cellSz))
				: INT_MAX;
			const int marginCells = std::min(marginBase, marginCellsCap);
			if (marginCells < marginBase)
				MESH_DIAG("Poisson: hard-cut margin capped %d -> %d cells (%.4g -> %.4g units)"
					" -- a %.4g cell puts the margin past the %.4g-unit ceiling; cells track"
					" point density, not extrapolation reach",
					marginBase, marginCells,
					(double)((float)marginBase * cellSz),
					(double)((float)marginCells * cellSz),
					(double)cellSz, (double)POISSON_TRIM_MARGIN_CAP_UNITS);
			if (marginCells > (int)POISSON_TRIM_MARGIN_CELLS)
				MESH_DIAG("Poisson: hard-cut margin floored %d -> %d cells (%.4g -> %.4g units)"
					" -- %d cells is %.3f%% of the %.4g extent, below the %.3f%% floor;"
					" the grid cell tracks point density, not balloon reach",
					(int)POISSON_TRIM_MARGIN_CELLS, marginCells,
					(double)((float)POISSON_TRIM_MARGIN_CELLS * cellSz),
					(double)((float)marginCells * cellSz),
					(int)POISSON_TRIM_MARGIN_CELLS,
					100.0 * (double)((float)POISSON_TRIM_MARGIN_CELLS * cellSz) / (double)footExtent,
					(double)footExtent, 100.0 * (double)POISSON_TRIM_MARGIN_FLOOR_FRAC);
			int gw = (int)((maxx - minx) * invCell) + 3;
			int gh = (int)((maxy - miny) * invCell) + 3;
			gw = std::min(std::max(gw, 16), 4096);
			gh = std::min(std::max(gh, 16), 4096);
			const float ox = minx - cellSz; // 1-cell empty border
			const float oy = miny - cellSz;
			auto cellOf = [&](float x, float y, int& cx, int& cy) {
				cx = (int)((x - ox) * invCell); cy = (int)((y - oy) * invCell);
				if (cx < 0) cx = 0; else if (cx >= gw) cx = gw - 1;
				if (cy < 0) cy = 0; else if (cy >= gh) cy = gh - 1;
			};
			const size_t nCells = (size_t)gw * gh;
			// Occupancy requires a MINIMUM POINT COUNT per cell, not merely one point.
			//
			// This is the difference between a footprint meaning "anywhere there is data"
			// and one meaning "the well-surveyed area". Binary occupancy marks a cell
			// occupied for a single stray sample, so the footprint swallows exactly the
			// sparse extrapolated fringe this pass exists to cut, and the computed
			// perimeter ends up tracing the OUTSIDE of the lobes. MEASURED on the
			// 115-view OKState corridor: the lobes sit on real (spurious) dense-matching
			// points -- which is also why POISSON_DISTANCE_CULL found nothing and why
			// SurfaceTrimmer's density trim cannot reach them. Any criterion of the form
			// "is there data here" answers yes. A criterion of the form "is there ENOUGH
			// data here" is what separates surveyed ground from extrapolation.
			std::vector<uint32_t> cnt(nCells, 0);
			for (size_t i = 0; i < poissonCount; ++i) {
				int cx, cy; cellOf(cloud[i].x, cloud[i].y, cx, cy);
				++cnt[(size_t)cy * gw + cx];
			}
			std::vector<uint8_t> occ(nCells, 0);
			size_t nOcc = 0;
			for (size_t k = 0; k < nCells; ++k)
				if (cnt[k] >= (uint32_t)POISSON_TRIM_MIN_PTS_PER_CELL) { occ[k] = 1; ++nOcc; }
			// [MESH-OCCUPANCY] What the threshold is actually choosing between.
			//
			// MEASURED on RichmondHistoric: halving the threshold 8 -> 4 moved occupancy only
			// 0.413 -> 0.425 (4401 of 388512 cells) and the outside-footprint cut 24153 ->
			// 18979. A 2x relaxation buying 2.7% more footprint means the per-cell counts are
			// BIMODAL -- cells hold either plenty of points or almost none -- so no value of
			// this threshold recovers a band whose cells are near-empty.
			//
			// The ladder says which regime a scene is in without another build-and-run cycle:
			// if >=1 sits far above >=4, there is thin-but-real data the threshold is
			// excluding and lowering it helps. If >=1 is close to >=4, the missing ground has
			// no points at all, the surface over it is pure extrapolation, and the hard cut is
			// working as designed -- at which point the question is whether that cut is wanted,
			// not what its threshold should be.
			//
			// The ladder is read off `cnt`, which is dropped immediately below, so the
			// six-way tally has to happen here or not at all -- hence the gate around
			// the counting pass and not merely around the line it prints.
			if (MESH_DIAG_ENABLED()) {
				size_t c1 = 0, c2 = 0, c4 = 0, c8 = 0, c16 = 0, c64 = 0;
				for (size_t k = 0; k < nCells; ++k) {
					const uint32_t n = cnt[k];
					if (n >= 1)  ++c1;   if (n >= 2)  ++c2;
					if (n >= 4)  ++c4;   if (n >= 8)  ++c8;
					if (n >= 16) ++c16;  if (n >= 64) ++c64;
				}
				const double inv = nCells ? 1.0 / (double)nCells : 0.0;
				MESH_DIAG("[MESH-OCCUPANCY] cells=%zu | >=1 %zu (%.4f) >=2 %zu (%.4f) >=4 %zu (%.4f)"
					" >=8 %zu (%.4f) >=16 %zu (%.4f) >=64 %zu (%.4f) | threshold=%d cell=%.4g"
					" (%.0fx median spacing %.4g, ~%.0f pts at median density)",
					nCells, c1, c1 * inv, c2, c2 * inv, c4, c4 * inv, c8, c8 * inv,
					c16, c16 * inv, c64, c64 * inv,
					(int)POISSON_TRIM_MIN_PTS_PER_CELL, (double)cellSz,
					(double)POISSON_TRIM_CELL_FACTOR_X100 / 100.0, (double)medianSpacing,
					std::pow((double)POISSON_TRIM_CELL_FACTOR_X100 / 100.0, 2.0));
			}
			std::vector<uint32_t>().swap(cnt);
			// ---------------------------------------------------------------
			// FOOTPRINT: occupancy -> dilate -> flood the EXTERIOR so enclosed low-coverage
			// regions (water) count as inside. Built inside an ESCALATING-DILATION loop guarded
			// by a connectivity check, which is what makes the hard outside-footprint cut safe
			// to use at all -- see POISSON_TRIM_HARD_OUTSIDE.
			// ---------------------------------------------------------------
			std::vector<uint8_t> ext(nCells, 0);
			int drUsed = POISSON_TRIM_CLOSE_CELLS;
			bool bHardCutOK = false;
			{
				const auto buildExt = [&](int margin, std::vector<uint8_t>& extOut) {
					// MORPHOLOGICAL CLOSING, not plain dilation. POISSON_TRIM_CLOSE_CELLS is the
					// SEAL radius: it exists so the exterior flood cannot reach a lake through a
					// narrow shore channel. Dilating and never eroding back ALSO pushes the outer
					// boundary that far past the data -- a separate concern, and one number cannot
					// serve both. Sealing wants a large radius; a tight boundary wants a small one,
					// which is exactly why holes and excess margin traded against each other.
					// So: dilate by the seal radius, flood the exterior (classification is then
					// fixed, lake included), then erode the footprint back to leave only `margin`
					// cells beyond the data. The lake survives because it sits in the footprint
					// INTERIOR, far from the rind the erosion removes.
					const int dr = POISSON_TRIM_CLOSE_CELLS;
					std::vector<uint8_t> occD(occ);
					if (dr > 0) {
						for (int cy = 0; cy < gh; ++cy) for (int cx = 0; cx < gw; ++cx) {
							if (!occ[(size_t)cy * gw + cx]) continue;
							for (int dy = -dr; dy <= dr; ++dy) { int ny = cy + dy; if (ny < 0 || ny >= gh) continue;
								for (int dx = -dr; dx <= dr; ++dx) { int nx = cx + dx; if (nx < 0 || nx >= gw) continue;
									occD[(size_t)ny * gw + nx] = 1; } }
						}
					}
					extOut.assign(nCells, 0);
					std::vector<int> stk;
					const auto pushIf = [&](int cx, int cy) {
						const size_t k = (size_t)cy * gw + cx;
						if (!occD[k] && !extOut[k]) { extOut[k] = 1; stk.push_back((int)k); }
					};
					for (int cx = 0; cx < gw; ++cx) { pushIf(cx, 0); pushIf(cx, gh - 1); }
					for (int cy = 0; cy < gh; ++cy) { pushIf(0, cy); pushIf(gw - 1, cy); }
					while (!stk.empty()) {
						const int k = stk.back(); stk.pop_back();
						const int cx = k % gw, cy = k / gw;
						if (cx > 0) pushIf(cx - 1, cy);  if (cx < gw - 1) pushIf(cx + 1, cy);
						if (cy > 0) pushIf(cx, cy - 1);  if (cy < gh - 1) pushIf(cx, cy + 1);
					}
					// Erode the footprint back == dilate the EXTERIOR inward by (seal - margin).
					// Growing `ext` cannot reopen the sealed channel: interior/exterior was already
					// decided by the flood above, so this only thickens the exterior region.
					const int er = dr - margin;
					if (er > 0) {
						std::vector<uint8_t> extE(extOut);
						for (int cy = 0; cy < gh; ++cy) for (int cx = 0; cx < gw; ++cx) {
							if (!extOut[(size_t)cy * gw + cx]) continue;
							for (int dy = -er; dy <= er; ++dy) { int ny = cy + dy; if (ny < 0 || ny >= gh) continue;
								for (int dx = -er; dx <= er; ++dx) { int nx = cx + dx; if (nx < 0 || nx >= gw) continue;
									extE[(size_t)ny * gw + nx] = 1; } }
						}
						extOut.swap(extE);
					}
				};
				buildExt(marginCells, ext);
#if POISSON_TRIM_HARD_OUTSIDE
				std::vector<uint8_t> vo(numV, 0);
				const auto vOutFrom = [&](const std::vector<uint8_t>& e) {
					for (Mesh::VIndex v = 0; v < numV; ++v) {
						int cx, cy; cellOf(mesh.vertices[v].x, mesh.vertices[v].y, cx, cy);
						vo[v] = e[(size_t)cy * gw + cx] ? 1 : 0;
					}
				};
				// Component count of the surface that SURVIVES the cut, by union-find over the
				// vertices of kept faces. Face adjacency does not exist yet at this point in the
				// pipeline, and this is exactly the quantity the documented corridor failure
				// showed up in: 4,844 components against ~880 on a scene where the cut was fine.
				std::vector<uint32_t> uf(numV);
				std::vector<uint8_t> ufUsed(numV);
				const auto compCount = [&](bool applyCut, size_t& nCut) -> size_t {
					for (Mesh::VIndex v = 0; v < numV; ++v) { uf[v] = v; ufUsed[v] = 0; }
					nCut = 0;
					const auto find = [&uf](uint32_t a) {
						while (uf[a] != a) { uf[a] = uf[uf[a]]; a = uf[a]; }
						return a;
					};
					FOREACH(f, mesh.faces) {
						const Mesh::Face& fc = mesh.faces[f];
						if (applyCut && vo[fc[0]] && vo[fc[1]] && vo[fc[2]]) { ++nCut; continue; }
						const uint32_t r0 = find(fc[0]);
						for (int k = 1; k < 3; ++k) { const uint32_t r = find(fc[k]); if (r != r0) uf[r] = r0; }
						ufUsed[fc[0]] = ufUsed[fc[1]] = ufUsed[fc[2]] = 1;
					}
					size_t n = 0;
					for (Mesh::VIndex v = 0; v < numV; ++v) if (ufUsed[v] && find(v) == v) ++n;
					return n;
				};
				size_t dummy = 0;
				const size_t compBase = compCount(false, dummy);
				const size_t compBudget = (size_t)((double)compBase *
					(double)POISSON_TRIM_GUARD_COMP_RATIO_X100 / 100.0)
					+ (size_t)POISSON_TRIM_GUARD_COMP_SLACK;
				// ESCALATE THE MARGIN, not the seal radius. A bigger margin means a bigger
				// footprint, so the cut removes LESS -- the direction that makes a failing guard
				// safer. Escalating the seal radius would no longer help, since boundary position
				// is now decoupled from it. It also tops out correctly: once margin reaches the
				// seal radius the erosion is zero and this becomes exactly the old dilate-only
				// behaviour, which is the known-safe configuration.
				int dr = marginCells;
				for (int attempt = 0; attempt < POISSON_TRIM_GUARD_ATTEMPTS; ++attempt) {
					if (attempt) { dr = (dr < 1 ? 1 : dr * 2); buildExt(dr, ext); }
					vOutFrom(ext);
					size_t nCut = 0;
					const size_t compAfter = compCount(true, nCut);
					MESH_DIAG("Poisson: footprint hard-cut attempt %d: margin %d cells (%.4g units, seal %d)"
						" -> would cut %zu faces, components %zu -> %zu (budget %zu)",
						attempt + 1, dr, (double)((float)dr * cellSz), (int)POISSON_TRIM_CLOSE_CELLS, nCut,
						compBase, compAfter, compBudget);
					if (nCut == 0 || compAfter <= compBudget) {
						drUsed = dr; bHardCutOK = true; break;
					}
				}
				if (!bHardCutOK) {
					// FAIL SAFE -- incompleteness is worse than surplus geometry, the same policy
					// as POISSON_CULL_MAX_FRACTION. Rebuild at the base dilation so the ramped
					// DENSITY threshold still runs, and leave the hard cut off for this scene.
					buildExt(POISSON_TRIM_CLOSE_CELLS, ext);
					drUsed = POISSON_TRIM_CLOSE_CELLS;
					MESH_DIAG("Poisson: footprint hard-cut ABANDONED after %d attempts -- every"
						" dilation severed the surface past the %zu-component budget. Falling back"
						" to the density ramp alone.", (int)POISSON_TRIM_GUARD_ATTEMPTS, compBudget);
				}
#endif
			}
			// 2-pass chamfer distance-to-perimeter over footprint (exterior = 0).
			constexpr float BIG = 1e9f, d1 = 1.f, d2c = 1.41421356f;
			std::vector<float> dist(nCells, BIG);
			for (size_t k = 0; k < nCells; ++k) if (ext[k]) dist[k] = 0.f;
			for (int cy = 0; cy < gh; ++cy) for (int cx = 0; cx < gw; ++cx) {
				const size_t k = (size_t)cy * gw + cx; float m = dist[k];
				if (cx > 0)                 m = std::min(m, dist[k - 1] + d1);
				if (cy > 0)                 m = std::min(m, dist[k - gw] + d1);
				if (cx > 0 && cy > 0)       m = std::min(m, dist[k - gw - 1] + d2c);
				if (cx < gw - 1 && cy > 0)  m = std::min(m, dist[k - gw + 1] + d2c);
				dist[k] = m;
			}
			for (int cy = gh - 1; cy >= 0; --cy) for (int cx = gw - 1; cx >= 0; --cx) {
				const size_t k = (size_t)cy * gw + cx; float m = dist[k];
				if (cx < gw - 1)                 m = std::min(m, dist[k + 1] + d1);
				if (cy < gh - 1)                 m = std::min(m, dist[k + gw] + d1);
				if (cx < gw - 1 && cy < gh - 1)  m = std::min(m, dist[k + gw + 1] + d2c);
				if (cx > 0 && cy < gh - 1)       m = std::min(m, dist[k + gw - 1] + d2c);
				dist[k] = m;
			}

			// REACH CUT. Per-vertex "this sits further past the surveyed data than any real
			// surface should", filled by the overhang block below and consumed by the
			// outside-footprint cut. Empty/all-zero when disabled.
			//
			// WHY A SECOND CRITERION. The footprint cut asks a binary in/out question and its
			// margin is in GRID CELLS, which track point density -- so on a sparse scene the
			// cell is huge and the footprint legitimately reaches a long way past the data.
			// MEASURED: Marco's cell is 3.486 units against a 0.17-1.6 median elsewhere, so
			// its 3-cell margin is 10.46 units where most scenes get 0.5-2.1. That is the slop.
			//
			// Distance from the data is the one measurement that isolates Marco across the
			// nine-scene corpus. Overhang of mesh-covered cells, in WORLD UNITS:
			//   Marco p50 13.9  p90 76.0  p99 214    <-- 3.4x the next scene at p90
			//   SchnellTests p90 22.4 | GEOTAG 21.6 | MechanicFalls 11.5 | RichmondWater 8.2
			//   Niwot 7.3 | RichmondHistoric 6.3 | Redy 5.5 | Randy 2.2
			// In CELLS the same scenes are 11.8-21.8 and do not separate at all -- so this
			// threshold must be in world units. Extrapolation reach is a physical distance;
			// it has nothing to do with how densely the cloud happens to be sampled.
			std::vector<uint8_t> vFar;
			size_t nFarCells = 0;
			float reachCutUnits = 0.f;
			// [MESH-OVERHANG] INSTRUMENTATION for the UNDER-cut failure (Marco, Redy: too
			// much surface past the real edge). No behaviour attached.
			//
			// The margin floor above addresses over-cutting. Under-cutting is the opposite
			// mechanism and nothing measured so far explains it -- margin/extent puts Redy
			// (0.67%) between healthy scenes at 0.59% and 0.79%, so it is not a threshold in
			// the other direction. What is missing is a measure of how far the SOLVE reaches
			// past the DATA, independent of whatever the footprint then does about it.
			//
			// So: chamfer-DT from the well-surveyed cells (occ), then report the distribution
			// of that distance over cells the mesh actually covers. A scene that extrapolates
			// a long way shows a fat tail here regardless of its grid cell or its margin.
			// Measured PRE-cut, so it describes the balloon the trim has to deal with, not
			// what survived.
			//
			// The chamfer transform below has ONE non-diagnostic consumer, the reach cut,
			// so the block runs whenever that is compiled in and otherwise only when the
			// diagnostics are asked for. With POISSON_REACH_CUT off (the shipped setting)
			// this is a second full two-pass DT over the occupancy grid plus a sort of
			// every mesh-covered cell, feeding nothing but one line.
			if (POISSON_REACH_CUT || MESH_DIAG_ENABLED()) {
				const bool bDiag = MESH_DIAG_ENABLED();
				std::vector<float> dOcc(nCells, BIG);
				for (size_t k = 0; k < nCells; ++k) if (occ[k]) dOcc[k] = 0.f;
				for (int cy = 0; cy < gh; ++cy) for (int cx = 0; cx < gw; ++cx) {
					const size_t k = (size_t)cy * gw + cx; float m = dOcc[k];
					if (cx > 0)                 m = std::min(m, dOcc[k - 1] + d1);
					if (cy > 0)                 m = std::min(m, dOcc[k - gw] + d1);
					if (cx > 0 && cy > 0)       m = std::min(m, dOcc[k - gw - 1] + d2c);
					if (cx < gw - 1 && cy > 0)  m = std::min(m, dOcc[k - gw + 1] + d2c);
					dOcc[k] = m;
				}
				for (int cy = gh - 1; cy >= 0; --cy) for (int cx = gw - 1; cx >= 0; --cx) {
					const size_t k = (size_t)cy * gw + cx; float m = dOcc[k];
					if (cx < gw - 1)                 m = std::min(m, dOcc[k + 1] + d1);
					if (cy < gh - 1)                 m = std::min(m, dOcc[k + gw] + d1);
					if (cx < gw - 1 && cy < gh - 1)  m = std::min(m, dOcc[k + gw + 1] + d2c);
					if (cx > 0 && cy < gh - 1)       m = std::min(m, dOcc[k + gw - 1] + d2c);
					dOcc[k] = m;
				}
				std::vector<uint8_t> mcov(nCells, 0);
				for (Mesh::VIndex v = 0; v < numV; ++v) {
					int cx, cy; cellOf(mesh.vertices[v].x, mesh.vertices[v].y, cx, cy);
					mcov[(size_t)cy * gw + cx] = 1;
				}
				// nMeshCells is also reported by the reach-cut line below, so it is counted
				// either way; `oh` is the overhang distribution and exists only to be printed.
				std::vector<float> oh;
				size_t nMeshCells = 0;
				for (size_t k = 0; k < nCells; ++k) {
					if (!mcov[k]) continue;
					++nMeshCells;
					if (bDiag && !occ[k]) oh.push_back(dOcc[k]);
				}
				if (bDiag) {
					std::sort(oh.begin(), oh.end());
					const auto pctl = [&](double p) -> double {
						if (oh.empty()) return 0.0;
						size_t i = (size_t)(p * (double)(oh.size() - 1) + 0.5);
						if (i >= oh.size()) i = oh.size() - 1;
						return (double)oh[i];
					};
					MESH_DIAG("[MESH-OVERHANG] dataCells=%zu meshCells=%zu (x%.3f) outside=%zu (%.4f)"
						" | overhang cells p50=%.2f p90=%.2f p99=%.2f max=%.2f"
						" | units p50=%.4g p90=%.4g p99=%.4g max=%.4g | cell=%.4g extent=%.4g",
						nOcc, nMeshCells,
						nOcc ? (double)nMeshCells / (double)nOcc : 0.0,
						oh.size(), nMeshCells ? (double)oh.size() / (double)nMeshCells : 0.0,
						pctl(0.50), pctl(0.90), pctl(0.99), oh.empty() ? 0.0 : (double)oh.back(),
						pctl(0.50) * (double)cellSz, pctl(0.90) * (double)cellSz,
						pctl(0.99) * (double)cellSz,
						(oh.empty() ? 0.0 : (double)oh.back()) * (double)cellSz,
						(double)cellSz, (double)footExtent);
				}

#if POISSON_REACH_CUT
				// Threshold in world units: an absolute floor so small scenes are untouched,
				// raised on large ones so the tolerance stays proportionate.
				//
				// CALIBRATION against the corpus, T = max(30, 0.05*extent):
				//   Marco     T=41.5  p90=76.0  -> cuts >10% of its outside cells  <-- target
				//   SchnellT  T=31.2  p90=22.4  -> a few % of outside cells
				//   GEOTAG    T=30.0  p90=21.6  -> a few %
				//   MechFalls T=33.8  p90=11.5  -> ~3%
				//   RichWater T=30.0  p99=32.9  -> ~1%
				//   RichHist  T=30.0  p99=29.2  -> ~1%
				//   Niwot     T=30.0  max=42.3  -> a handful of cells
				//   Redy      T=30.0  max=28.8  -> NOTHING
				//   Randy     T=30.0  max=7.3   -> nothing
				// So it lands almost entirely on Marco, which is what was asked for. It does
				// NOT help Redy -- Redy's overhang is lower than several healthy scenes, so its
				// slop is not the mesh reaching past the cloud, it is the CLOUD reaching too
				// far. That needs fixing upstream in densify, not here.
				//
				// Requiring all three vertices beyond T (at the cut site) keeps any face that
				// straddles the threshold, so the boundary lands outside real surface.
				reachCutUnits = std::max((float)POISSON_REACH_CUT_MIN_UNITS,
					(float)POISSON_REACH_CUT_EXTENT_FRAC * footExtent);
				if (reachCutUnits > 0.f && cellSz > 0.f) {
					const float dLimit = reachCutUnits / cellSz;   // back into cell units
					vFar.assign(numV, 0);
					for (Mesh::VIndex v = 0; v < numV; ++v) {
						int cx, cy; cellOf(mesh.vertices[v].x, mesh.vertices[v].y, cx, cy);
						if (dOcc[(size_t)cy * gw + cx] > dLimit) vFar[v] = 1;
					}
					for (size_t k = 0; k < nCells; ++k)
						if (mcov[k] && dOcc[k] > dLimit) ++nFarCells;
					MESH_DIAG("Poisson: reach cut armed at %.4g units (%.1f cells) = max(%.4g,"
						" %.3f x extent %.4g) -- %zu of %zu mesh cells are beyond it",
						(double)reachCutUnits, (double)dLimit,
						(double)POISSON_REACH_CUT_MIN_UNITS,
						(double)POISSON_REACH_CUT_EXTENT_FRAC, (double)footExtent,
						nFarCells, nMeshCells);
				}
#endif
			}

			const float ramp = (float)POISSON_TRIM_RAMP_CELLS;
			// SELF-CALIBRATING INTERIOR THRESHOLD. Poisson density is on an arbitrary scale that
			// shifts with octree DEPTH and with the dataset, so a raw --poisson-trim can never be
			// portable: MEASURED, the same interior/edge pair removed 4.2% at depth 10, 0.02% at
			// depth 11 and 0.8% at depth 12 on one cloud, and in the field one site was well tuned
			// at 5.75 on depth 11 while another needed a visibly higher value purely because it
			// solved at depth 12. That is a UNITS problem, not a per-dataset tuning problem -- the
			// same class as the four resolution-relative thresholds elsewhere in this pipeline.
			//
			// So the threshold is taken as a PERCENTILE of THIS scene's own density distribution,
			// which is depth- and dataset-invariant by construction. --poisson-trim then only has
			// to be > 0 to enable the pass (the block is gated on it); its magnitude stops
			// mattering. Set POISSON_TRIM_PERCENTILE_X100 to 0 to go back to the raw value.
			float trimBase = trimThreshold;
#if POISSON_TRIM_PERCENTILE_X100 > 0
			{
				std::vector<float> dsq;
				const size_t qstride = std::max<size_t>(1, (size_t)numV / 200000);
				dsq.reserve((size_t)numV / qstride + 1);
				for (Mesh::VIndex v = 0; v < numV; v += (Mesh::VIndex)qstride)
					dsq.push_back(pv[(size_t)v * 4 + 3]);
				if (!dsq.empty()) {
					std::sort(dsq.begin(), dsq.end());
					const double q = (double)POISSON_TRIM_PERCENTILE_X100 / 10000.0;
					const size_t kq = (size_t)(q * (double)(dsq.size() - 1) + 0.5);
					const float qv = dsq[kq < dsq.size() ? kq : dsq.size() - 1];
					MESH_DIAG("Poisson: trim from distribution: p%.2f = %.3g (raw --poisson-trim was"
						" %.2f) -- percentile is depth- and dataset-invariant, the raw value is not",
						q * 100.0, (double)qv, (double)trimThreshold);
					trimBase = qv;
				}
			}
#endif
			const float trimInterior = trimBase * (POISSON_TRIM_INTERIOR_MULT_X100 / 100.f);
			const float trimEdge = trimBase * (POISSON_TRIM_EDGE_MULT_X100 / 100.f);

			// per-vertex local threshold (e=0 deep interior -> e=1 at perimeter)
			const Mesh::Vertex* __restrict pVtx = mesh.vertices.GetData();
			std::vector<float> vTrim(numV);
			std::vector<uint8_t> vOut(numV, 0); // 1 = this vertex's XY cell is OUTSIDE the footprint
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v) {
				int cx, cy; cellOf(pVtx[v].x, pVtx[v].y, cx, cy);
				const float dcell = dist[(size_t)cy * gw + cx];
				// TWO independent ramps, because they answer different questions and the
				// distances are an order of magnitude apart.
				//
				// 1. EDGE ramp, over POISSON_TRIM_RAMP_CELLS (6): trimBase -> trimEdge as you
				//    approach the perimeter. Unchanged.
				// 2. INTERIOR ramp, over POISSON_TRIM_INTERIOR_CELLS (24): trimBase ->
				//    trimInterior as you move away from it. Starts only where the edge ramp
				//    has finished.
				//
				// Sharing one ramp for both was wrong and measurably so. With a 6-cell ramp,
				// trimEdge applies only at dist == 0 and everything within 7.3 units of the
				// boundary interpolates toward the interior value -- so lowering the interior
				// end to protect mid-scene water dragged the whole perimeter band down with it.
				// MEASURED on SchnellTests: the density trim fell 3408 -> 658 faces and the
				// edge went visibly sloppy, when the intent was to change the interior alone.
				//
				// Anchoring both ramps at trimBase keeps the perimeter exactly as it was before
				// the interior dial existed: at dist 0 the threshold is trimEdge, at dist ramp
				// it is trimBase, and only past that does it start relaxing.
				float e = 1.f - dcell / ramp; if (e < 0.f) e = 0.f; else if (e > 1.f) e = 1.f;
				float vt = trimBase + e * (trimEdge - trimBase);
				const float interiorSpan = (float)POISSON_TRIM_INTERIOR_CELLS - ramp;
				if (interiorSpan > 0.f) {
					float ei = (dcell - ramp) / interiorSpan;
					if (ei < 0.f) ei = 0.f; else if (ei > 1.f) ei = 1.f;
					vt += ei * (trimInterior - trimBase);
				}
				vTrim[v] = vt;
				vOut[v] = (dcell <= 0.f) ? 1 : 0; // exterior cells were seeded to 0
			}

			// DENSITY DISTRIBUTION -- so the interior/edge pair can be set FROM THE DATA
			// instead of guessed. MEASURED: at depth 11 interior=1/edge=4 removed 816 of
			// 3,672,117 faces (fraction 0.000, i.e. inert) while at depth 10 interior=2/edge=8
			// removed 4.2% and at depth 12 the same pair removed 0.8%. The Poisson density scale
			// is NOT portable across octree depths, so a threshold carried over from another
			// depth is silently a no-op -- which is invisible without this line.
			//
			// Read it as: to cut roughly the top X% of low-density surface, set the EDGE
			// threshold near pX. The interior should sit near p1-p5 so the well-surveyed middle
			// keeps its coverage.
			//
			// Instrumentation only (the threshold itself comes from the percentile block
			// above), so the gather and sort go behind the gate with the line.
			if (MESH_DIAG_ENABLED()) {
				std::vector<float> ds;
				const size_t dstride = std::max<size_t>(1, (size_t)numV / 200000);
				ds.reserve((size_t)numV / dstride + 1);
				for (Mesh::VIndex v = 0; v < numV; v += (Mesh::VIndex)dstride)
					ds.push_back(pv[(size_t)v * 4 + 3]);
				if (!ds.empty()) {
					std::sort(ds.begin(), ds.end());
					const auto pct = [&](double q) {
						const size_t k = (size_t)(q * (double)(ds.size() - 1) + 0.5);
						return (double)ds[k];
					};
					MESH_DIAG("Poisson: vertex density percentiles (n=%zu of %u sampled):"
						" p1=%.3g p5=%.3g p25=%.3g p50=%.3g p75=%.3g p95=%.3g p99=%.3g max=%.3g"
						" | current interior=%.2f edge=%.2f -- set these from THIS distribution,"
						" they do not transfer across depths",
						ds.size(), (unsigned)numV,
						pct(0.01), pct(0.05), pct(0.25), pct(0.50), pct(0.75),
						pct(0.95), pct(0.99), (double)ds.back(),
						trimInterior, trimEdge);
				}
			}

			// cull faces: keep iff avg density >= avg local threshold
			const Mesh::FIndex numF = mesh.faces.GetSize();
			std::vector<uint8_t> keepV(numV, 0);
			Mesh::FaceArr newFaces; newFaces.Reserve(numF);
			size_t culled = 0, culledOutside = 0, culledReach = 0;
			FOREACH(f, mesh.faces) {
				const Mesh::Face& face = mesh.faces[f];
				const float dAvg = (pv[(size_t)face[0] * 4 + 3] + pv[(size_t)face[1] * 4 + 3] +
				                    pv[(size_t)face[2] * 4 + 3]) * (1.f / 3.f);
				const float tAvg = (vTrim[face[0]] + vTrim[face[1]] + vTrim[face[2]]) * (1.f / 3.f);
				// HARD FOOTPRINT CUT: a face whose ALL THREE vertices sit in cells OUTSIDE the
				// surveyed footprint is extrapolation into space the survey never covered, and it
				// goes regardless of density.
				//
				// This is the region decision the threshold machinery was standing in for and
				// failing to make. Poisson density is not comparable across octree depths -- the
				// same interior/edge pair removed 4.2% at depth 10, 0.8% at 12 and 0.02% at 11 --
				// so a value fitted at one depth is silently inert at another. The FOOTPRINT has
				// no such problem: it is a point-count occupancy grid (measured at 26% of the bbox
				// at >=20 pts/cell on this data) and it is depth-independent.
				//
				// Conservative by construction: the footprint is already dilated by
				// POISSON_TRIM_CLOSE_CELLS (2 cells ~ 3 units here) so it reaches BEYOND the data,
				// and requiring all three vertices outside keeps every face that straddles the
				// boundary -- so the cut lands just outside real surface rather than into it.
				// Set POISSON_TRIM_HARD_OUTSIDE to 0 to disable.
				// bHardCutOK: the escalating-dilation guard above accepted a dilation at which this
				// cut does NOT sever the surface. If none passed, the cut is skipped entirely and
				// only the density ramp runs -- see POISSON_TRIM_HARD_OUTSIDE.
				if (POISSON_TRIM_HARD_OUTSIDE && bHardCutOK &&
					vOut[face[0]] && vOut[face[1]] && vOut[face[2]]) {
					++culledOutside; ++culled; continue;
				}
				// Reach cut: independent of the footprint, and NOT gated on bHardCutOK.
				// The footprint cut is suppressed when its guard finds it would sever the
				// surface, but a face sitting 40+ units from any surveyed cell is not part
				// of a surface worth preserving -- that is the case the footprint's
				// cells-based margin structurally cannot see. See the REACH CUT note above.
				if (!vFar.empty() &&
					vFar[face[0]] && vFar[face[1]] && vFar[face[2]]) {
					++culledReach; ++culled; continue;
				}
				if (dAvg < tAvg) { ++culled; continue; }
				keepV[face[0]] = keepV[face[1]] = keepV[face[2]] = 1;
				newFaces.Insert(face);
			}
			if (culled > 0) {
				std::vector<Mesh::VIndex> remap(numV);
				Mesh::VIndex vW = 0;
				for (Mesh::VIndex v = 0; v < numV; ++v) remap[v] = keepV[v] ? vW++ : NO_ID;
				Mesh::VertexArr nvarr; nvarr.Resize(vW);
				Mesh::Vertex* __restrict pDst = nvarr.GetData();
				const Mesh::Vertex* __restrict pSrc = mesh.vertices.GetData();
				for (Mesh::VIndex v = 0; v < numV; ++v) if (keepV[v]) pDst[remap[v]] = pSrc[v];
				const Mesh::FIndex nfk = newFaces.GetSize();
				Mesh::Face* __restrict pFK = newFaces.GetData();
#ifdef _USE_OPENMP
				#pragma omp parallel for schedule(static)
#endif
				for (ptrdiff_t f = 0; f < (ptrdiff_t)nfk; ++f) {
					pFK[f][0] = remap[pFK[f][0]]; pFK[f][1] = remap[pFK[f][1]]; pFK[f][2] = remap[pFK[f][2]];
				}
				mesh.vertices.Swap(nvarr);
				mesh.faces.Swap(newFaces);
			}
			// The trim CHANGES the mesh, so the headline count stays at normal verbosity;
			// the parameter dump that explains how it was arrived at does not.
			MESH_DIAG("Poisson: adaptive footprint trim removed %u of %u faces (fraction %.3f) [%s]",
				(unsigned)culled, (unsigned)numF,
				numF ? (double)culled / (double)numF : 0.0, TD_TIMER_GET_FMT().c_str());
			MESH_DIAG("Poisson: adaptive footprint trim removed %u of %u faces (fraction %.3f)"
				" | interior=%.2f (x%.2f of base %.2f, full at %d cells) edge=%.2f"
				" ramp=%d cells (%.4g units)"
				" | grid %dx%d cell=%.4g, %zu of %zu cells occupied (fraction %.3f)"
				" at >=%d pts/cell | outside-footprint cut %zu faces"
				" | reach cut %zu faces (>%.4g units from data) [%s]",
				(unsigned)culled, (unsigned)numF,
				numF ? (double)culled / (double)numF : 0.0,
				trimInterior, (double)POISSON_TRIM_INTERIOR_MULT_X100 / 100.0, trimBase,
				(int)POISSON_TRIM_INTERIOR_CELLS, trimEdge, POISSON_TRIM_RAMP_CELLS,
				(double)(POISSON_TRIM_RAMP_CELLS * cellSz),
				gw, gh, cellSz, nOcc, nCells,
				nCells ? (double)nOcc / (double)nCells : 0.0,
				(int)POISSON_TRIM_MIN_PTS_PER_CELL, culledOutside,
				culledReach, (double)reachCutUnits, TD_TIMER_GET_FMT().c_str());
		}
	}
#endif // POISSON_ADAPTIVE_TRIM

	// Sanitize the reconstructed mesh: PoissonRecon/SurfaceTrimmer can still
	// emit non-finite (NaN/Inf) vertices for degenerate octree nodes. Drop
	// them and any incident face, then compact the vertex indices, so that
	// downstream stages (e.g. RefineMesh) never consume NaN geometry.
	{
		const Mesh::VIndex numV = mesh.vertices.GetSize();
		uint8_t* __restrict pKeep = (uint8_t*)_aligned_malloc(numV, 64);
		ptrdiff_t numBad = 0;
		const Mesh::Vertex* __restrict pV = mesh.vertices.GetData();
#ifdef _USE_OPENMP
		#pragma omp parallel for schedule(static) reduction(+:numBad)
#endif
		for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v) {
			const Mesh::Vertex& X = pV[v];
			const bool finite = ISFINITE(X.x) && ISFINITE(X.y) && ISFINITE(X.z);
			pKeep[v] = finite ? 1 : 0;
			if (!finite) ++numBad;
		}
		if (numBad > 0) {
			// Build vertex remap via prefix-sum of pKeep (serial scan, ~numV iterations).
			std::vector<Mesh::VIndex> remap(numV);
			Mesh::VIndex writePos = 0;
			for (Mesh::VIndex v = 0; v < numV; ++v) {
				if (pKeep[v]) { remap[v] = writePos++; } else { remap[v] = NO_ID; }
			}
			// Scatter surviving vertices (parallel).
			Mesh::VertexArr newVerts;
			newVerts.Resize(writePos);
			Mesh::Vertex* __restrict pDst = newVerts.GetData();
			const Mesh::Vertex* __restrict pSrc = mesh.vertices.GetData();
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v)
				if (pKeep[v]) pDst[remap[v]] = pSrc[v];
			// Compact faces: count survivors, Resize, then parallel scatter with remap.
			const Mesh::FIndex numF = mesh.faces.GetSize();
			uint8_t* __restrict fKeep = (uint8_t*)_aligned_malloc(numF, 64);
			const Mesh::Face* __restrict pFaces = mesh.faces.GetData();
			ptrdiff_t numFBad = 0;
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static) reduction(+:numFBad)
#endif
			for (ptrdiff_t f = 0; f < (ptrdiff_t)numF; ++f) {
				const Mesh::Face& face = pFaces[f];
				const bool keep = pKeep[face[0]] && pKeep[face[1]] && pKeep[face[2]];
				fKeep[f] = keep ? 1 : 0;
				if (!keep) ++numFBad;
			}
			Mesh::FaceArr newFaces;
			newFaces.Resize((Mesh::FIndex)(numF - numFBad));
			Mesh::Face* __restrict pFDst = newFaces.GetData();
			Mesh::FIndex fWrite = 0;
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				if (fKeep[f]) {
					const Mesh::Face& face = pFaces[f];
					pFDst[fWrite++] = Mesh::Face(remap[face[0]], remap[face[1]], remap[face[2]]);
				}
			}
			_aligned_free(fKeep);
			mesh.vertices.Swap(newVerts);
			mesh.faces.Swap(newFaces);
			VERBOSE("Removed %u non-finite vertices (and incident faces)", (unsigned)numBad);
		}
		_aligned_free(pKeep);
	}

#if POISSON_DISTANCE_CULL
	// Distance-to-cloud cull: delete Poisson faces whose vertices ALL sit
	// farther than (POISSON_CULL_FACTOR_X100/100) * medianSpacing from the
	// nearest actual input point. Orthogonal to SurfaceTrimmer's density trim;
	// removes invented membranes / skirts / balloons that float away from the
	// real samples. See the macro comment near the top of this file.
	// Reads poissonPts/poissonCount, NOT `pointcloud`: those are the filtered
	// finite samples actually handed to the solver, and they stay valid when
	// releasePointCloud has already dropped the cloud. Reading the cloud here
	// would make this block silently no-op under --release-pointcloud (the
	// guard would see 0 points) -- and a lever that skips work looks exactly
	// like a lever that saves time.
	if (!mesh.faces.IsEmpty() && !mesh.vertices.IsEmpty() && poissonCount >= 100) {
		TD_TIMER_STARTD();
		const size_t numCloud = poissonCount;
		const Point3f* __restrict cloudPts = reinterpret_cast<const Point3f*>(poissonPts);

		using namespace nanoflann;
		using KDTree = KDTreeSingleIndexAdaptor<
			L2_Simple_Adaptor<float, PointCloudAdapter>, PointCloudAdapter, 3>;
		PointCloudAdapter cloud{ cloudPts, numCloud };
		KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(64));
		index.buildIndex();

		// median nearest-neighbour spacing over a deterministic stride-sampled
		// subset of the cloud (same statistic EstimatePoissonDepth uses).
		const size_t kSamples = std::min<size_t>(numCloud, 50000);
		const size_t stride = std::max<size_t>(1, numCloud / kSamples);
		const size_t nq = (numCloud + stride - 1) / stride;
		std::vector<float> spc(nq, -1.f);
		float* __restrict pSp = spc.data();
#ifdef _USE_OPENMP
		#pragma omp parallel for schedule(static)
#endif
		for (ptrdiff_t q = 0; q < (ptrdiff_t)nq; ++q) {
			const size_t i = (size_t)q * stride;
			uint32_t idx[2]; float d2[2];
			const size_t found = index.knnSearch((const float*)&cloudPts[i], 2, idx, d2);
			if (found >= 2 && d2[1] > 0.f)
				pSp[q] = std::sqrt(d2[1]); // d2[0] is the query point itself
		}
		std::vector<float> validSp;
		validSp.reserve(nq);
		for (size_t q = 0; q < nq; ++q)
			if (pSp[q] > 0.f) validSp.push_back(pSp[q]);

		if (!validSp.empty()) {
			std::nth_element(validSp.begin(), validSp.begin() + validSp.size() / 2, validSp.end());
			const float medianSpacing = validSp[validSp.size() / 2];
			const float factor = float(POISSON_CULL_FACTOR_X100) / 100.0f;
			const float globalThr = factor * medianSpacing;

			// LOCALLY ADAPTIVE threshold -- the "local-density-adaptive replacement"
			// the macro comment above demands before this cull can be switched on.
			//
			// The global version compares every vertex against a threshold derived from
			// the median (i.e. DENSEST-region) NN spacing. Point density varies by orders
			// of magnitude across a scene -- obliquely-viewed walls, distant ground,
			// canopy fringe -- so real surface in a sparse region sits many global-medians
			// from its nearest sample and is culled along with the invented skirts.
			// MEASURED: ~13.5% of faces removed at 3.5x, most of them good geometry. That
			// is the "trimming makes datasets less complete" failure, and it is a property
			// of the THRESHOLD, not of the criterion.
			//
			// Fix: measure the spacing where the vertex actually is. For points spread on
			// a 2D surface the distance to the K-th nearest neighbour grows as
			// s_local * sqrt(K), so s_local ~= d_K / sqrt(K) is a local density estimate
			// that costs one wider kd-tree query and no extra passes.
			//
			//   dense, well-sampled surface : d_1 ~ 0                -> keep
			//   sparse but REAL surface     : d_1 ~ s_local (large)  -> keep
			//   invented skirt / balloon    : d_1 >> s_local         -> cull
			//
			// Scale-invariant per region, which is exactly what a global threshold cannot
			// be. The global threshold is retained as a FLOOR so that a pathologically
			// tight local cluster cannot produce a threshold near zero and cull surface
			// sitting a few millimetres off its own samples.
			constexpr uint32_t kLocalK = 8;
			// Never let the adaptive threshold fall below this fraction of the global one.
			const float thrFloor = 0.5f * globalThr;

			const Mesh::VIndex numV = mesh.vertices.GetSize();
			const bool bDiag = MESH_DIAG_ENABLED();
			std::vector<uint8_t> farV(numV);
			// d1/s_local, for the diagnostic below and nothing else -- 4 B/vertex that
			// only exists to be sorted once, so it is not allocated unless it is wanted.
			std::vector<float> ratioV(bDiag ? (size_t)numV : (size_t)0, 0.f);
			const Mesh::Vertex* __restrict pV = mesh.vertices.GetData();
			uint8_t* __restrict pFar = farV.data();
			float* __restrict pRatio = ratioV.empty() ? nullptr : ratioV.data();
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v) {
				const Mesh::Vertex& X = pV[v];
				const float qp[3] = { X.x, X.y, X.z };
				uint32_t nIdx[kLocalK]; float nD2[kLocalK];
				const size_t found = index.knnSearch(qp, kLocalK, nIdx, nD2);
				if (found < 1) { pFar[v] = 0; continue; }   // no data at all: keep
				const float d1 = std::sqrt(nD2[0]);
				// d_K over however many were actually returned.
				const float dK = std::sqrt(nD2[found - 1]);
				const float sLocal = dK / std::sqrt((float)found);
				float thr = factor * sLocal;
				if (thr < thrFloor) thr = thrFloor;
				pFar[v] = (d1 > thr) ? 1 : 0;
				if (pRatio) pRatio[v] = (sLocal > 0.f) ? (d1 / sLocal) : 0.f;
			}

			// Is there anything separable at ANY threshold? The cull firing zero times
			// says nothing sits past 3.5x local spacing, but not whether a lower cutoff
			// would find a real population of detached surface or would just start eating
			// into the body of the distribution.
			//
			// A distance cull can only work if d1/s_local is BIMODAL -- a bulk near zero
			// (vertices sitting on their samples) plus a separated tail (invented surface).
			// If the high percentiles sit just above the bulk with no gap, there is no
			// detached geometry to find and NO threshold works: every setting trades real
			// surface for invented surface at roughly one-for-one. That is the difference
			// between "tune the factor" and "abandon this approach", and it cannot be read
			// off a single pass/fail count.
			if (bDiag) {
				std::vector<float> r(pRatio, pRatio + numV);
				const auto pct = [&r](double p) -> float {
					if (r.empty()) return 0.f;
					size_t i = (size_t)(p * (double)(r.size() - 1));
					std::nth_element(r.begin(), r.begin() + i, r.end());
					return r[i];
				};
				const float p50 = pct(0.50), p90 = pct(0.90), p99 = pct(0.99);
				const float p999 = pct(0.999), pMax = pct(1.0);
				MESH_DIAG("[MESH-CULL] d1/s_local distribution over %u vertices:"
					" p50=%.2f p90=%.2f p99=%.2f p99.9=%.2f max=%.2f (cull fires above %.2f)"
					" -- a usable cull needs a GAP between the bulk and the tail",
					(unsigned)numV, p50, p90, p99, p999, pMax, factor);
			}

			// Cull faces with ALL THREE vertices far; mark surviving vertices.
			std::vector<uint8_t> keepV(numV, 0);
			Mesh::FaceArr newFaces;
			newFaces.Reserve(mesh.faces.GetSize());
			size_t culled = 0;
			FOREACH(f, mesh.faces) {
				const Mesh::Face& face = mesh.faces[f];
				if (pFar[face[0]] && pFar[face[1]] && pFar[face[2]]) {
					++culled;
					continue;
				}
				keepV[face[0]] = keepV[face[1]] = keepV[face[2]] = 1;
				newFaces.Insert(face);
			}

			// SAFETY GUARD. This cull is the one stage that can make a dataset LESS
			// COMPLETE, and incompleteness is far worse than a little surplus geometry --
			// a hole is a missing deliverable, extra skirt is a cosmetic defect. So bound
			// the damage: if the threshold wants to remove an implausible share of the
			// mesh, the threshold is wrong for this scene, not the mesh. Abandon the cull
			// entirely and say so, rather than shipping a gutted surface.
			//
			// Calibrated against the failure this replaces: the GLOBAL threshold removed
			// ~13.5% on an aerial scene, most of it good geometry. A correct adaptive
			// threshold should be targeting the ~8% of faces no camera observes, and in
			// practice a subset of those. Anything past this bound means the local density
			// estimate is not working on this data.
			const double culledFrac = mesh.faces.IsEmpty()
				? 0.0 : (double)culled / (double)mesh.faces.GetSize();
			bool cullAbandoned = false;
			if (culledFrac > POISSON_CULL_MAX_FRACTION) {
				cullAbandoned = true;
				// No percent signs in this format -- they do not survive the log macro
				// (see the [ATLAS-FIT] note in SceneTexture.cpp). Fractions instead.
				MESH_DIAG("[MESH-CULL] distance cull wanted %u of %u faces"
					" (fraction %.3f > cap %.3f) -- ABANDONED, keeping the full mesh."
					" The local density estimate is not discriminating on this scene;"
					" raise POISSON_CULL_FACTOR_X100 (now %.2fx) or set POISSON_DISTANCE_CULL 0.",
					(unsigned)culled, (unsigned)mesh.faces.GetSize(),
					culledFrac, (double)POISSON_CULL_MAX_FRACTION, factor);
				culled = 0;
			}

			if (culled > 0) {
				// Compact vertices: prefix-sum remap + parallel scatter.
				std::vector<Mesh::VIndex> remap(numV);
				Mesh::VIndex vWrite = 0;
				for (Mesh::VIndex v = 0; v < numV; ++v) {
					if (keepV[v]) { remap[v] = vWrite++; } else { remap[v] = NO_ID; }
				}
				Mesh::VertexArr newVerts;
				newVerts.Resize(vWrite);
				Mesh::Vertex* __restrict pVDst = newVerts.GetData();
				const Mesh::Vertex* __restrict pVSrc = mesh.vertices.GetData();
#ifdef _USE_OPENMP
				#pragma omp parallel for schedule(static)
#endif
				for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v)
					if (keepV[v]) pVDst[remap[v]] = pVSrc[v];
				// Remap face indices (parallel — pure per-element transform).
				const Mesh::FIndex nfk = newFaces.GetSize();
				Mesh::Face* __restrict pFK = newFaces.GetData();
#ifdef _USE_OPENMP
				#pragma omp parallel for schedule(static)
#endif
				for (ptrdiff_t f = 0; f < (ptrdiff_t)nfk; ++f) {
					Mesh::Face& face = pFK[f];
					face[0] = remap[face[0]];
					face[1] = remap[face[1]];
					face[2] = remap[face[2]];
				}
				const size_t keptFaces = newFaces.GetSize();
				mesh.vertices.Swap(newVerts);
				mesh.faces.Swap(newFaces);
				MESH_DIAG("[MESH-CULL] distance cull removed %u/%u faces"
					" (%.2fx LOCAL spacing, floor %.4g = 0.5 x %.2fx global median %.4g) [%s]",
					(unsigned)culled, (unsigned)(culled + keptFaces),
					factor, 0.5f * globalThr, factor, medianSpacing,
					TD_TIMER_GET_FMT().c_str());
			} else if (!cullAbandoned) {
				MESH_DIAG("[MESH-CULL] distance cull removed no faces"
					" (%.2fx LOCAL spacing, global median %.4g)", factor, medianSpacing);
			}
		}
	}
#endif

#if 0 // SKIRT CULL DISABLED — eroded boundary unacceptably across multiple approaches
	// (face-normal angle, escalating threshold, Z-descent). Left as dead code for
	// reference; the boundary smooth + connected-component cleanup handle the visual
	// quality without removing real surface geometry.
		if (mesh.faces.GetSize() > 0) {
			const Mesh::VIndex numV = mesh.vertices.GetSize();
			const Mesh::FIndex numF = mesh.faces.GetSize();
			const Mesh::Face* __restrict pF = mesh.faces.GetData();
			const Mesh::Vertex* __restrict pV = mesh.vertices.GetData();

			// Find boundary vertices (same logic as the smooth below).
			struct EdgeHash {
				size_t operator()(const std::pair<Mesh::VIndex,Mesh::VIndex>& e) const {
					return std::hash<uint64_t>()(((uint64_t)e.first << 32) | e.second);
				}
			};
			std::unordered_map<std::pair<Mesh::VIndex,Mesh::VIndex>, uint8_t, EdgeHash> edgeCnt;
			edgeCnt.reserve(numF * 3);
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				const Mesh::Face& face = pF[f];
				for (int e = 0; e < 3; ++e) {
					Mesh::VIndex a = face[e], b = face[(e+1)%3];
					if (a > b) std::swap(a, b);
					++edgeCnt[{a, b}];
				}
			}
			std::vector<uint8_t> isBnd(numV, 0);
			for (const auto& kv : edgeCnt) {
				if (kv.second == 1) {
					isBnd[kv.first.first] = 1;
					isBnd[kv.first.second] = 1;
				}
			}

			// Cull boundary faces that form the Poisson "skirt" (curtain geometry
			// hanging below the trim edge). Uses Z-DESCENT criterion: a boundary
			// face is skirt if its centroid Z is BELOW the average centroid Z of
			// its adjacent non-boundary faces. This directly detects "hanging
			// below the surface" regardless of face angle, without eating real
			// sloped terrain (whose boundary faces sit at the same Z level as
			// neighbors). Iterates: removing skirt exposes a new boundary which
			// may also descend, flood-deleting the entire drape.
			constexpr int kMaxSkirtPasses = 30;
			size_t totalSkirtCulled = 0;
			for (int pass = 0; pass < kMaxSkirtPasses; ++pass) {
				const Mesh::VIndex numVCur = mesh.vertices.GetSize();
				const Mesh::FIndex numFCur = mesh.faces.GetSize();
				const Mesh::Face* __restrict pFCur = mesh.faces.GetData();
				const Mesh::Vertex* __restrict pVCur = mesh.vertices.GetData();

				// Build edge counts and face adjacency
				std::unordered_map<std::pair<Mesh::VIndex,Mesh::VIndex>, uint8_t, EdgeHash> edgeCntCur;
				edgeCntCur.reserve(numFCur * 3);
				// Also build edge->face map for face adjacency
				std::unordered_map<std::pair<Mesh::VIndex,Mesh::VIndex>, std::pair<Mesh::FIndex,Mesh::FIndex>, EdgeHash> edgeFaces;
				edgeFaces.reserve(numFCur * 3);
				for (Mesh::FIndex f = 0; f < numFCur; ++f) {
					const Mesh::Face& face = pFCur[f];
					for (int e = 0; e < 3; ++e) {
						Mesh::VIndex a = face[e], b = face[(e+1)%3];
						if (a > b) std::swap(a, b);
						++edgeCntCur[{a, b}];
						auto it = edgeFaces.find({a, b});
						if (it == edgeFaces.end())
							edgeFaces[{a, b}] = {f, (Mesh::FIndex)~0u};
						else
							it->second.second = f;
					}
				}
				// Identify boundary vertices
				std::vector<uint8_t> isBndCur(numVCur, 0);
				for (const auto& kv : edgeCntCur) {
					if (kv.second == 1) {
						isBndCur[kv.first.first] = 1;
						isBndCur[kv.first.second] = 1;
					}
				}
				// Identify boundary faces (touch at least one boundary vertex)
				std::vector<uint8_t> isBndFace(numFCur, 0);
				for (Mesh::FIndex f = 0; f < numFCur; ++f) {
					const Mesh::Face& face = pFCur[f];
					if (isBndCur[face[0]] || isBndCur[face[1]] || isBndCur[face[2]])
						isBndFace[f] = 1;
				}
				// Compute face centroids Z
				std::vector<float> faceCentZ(numFCur);
				for (Mesh::FIndex f = 0; f < numFCur; ++f) {
					const Mesh::Face& face = pFCur[f];
					faceCentZ[f] = (pVCur[face[0]].z + pVCur[face[1]].z + pVCur[face[2]].z) * (1.f/3.f);
				}
				// For each boundary face, compare its centroid Z to the average
				// centroid Z of adjacent NON-boundary faces. If below → skirt.
				Mesh::FaceArr newFacesCur;
				newFacesCur.Reserve(numFCur);
				size_t passculled = 0;
				for (Mesh::FIndex f = 0; f < numFCur; ++f) {
					const Mesh::Face& face = pFCur[f];
					if (!isBndFace[f]) {
						newFacesCur.Insert(face);
						continue;
					}
					// Find adjacent non-boundary faces via shared edges
					float sumNbrZ = 0.f;
					int nbrCount = 0;
					for (int e = 0; e < 3; ++e) {
						Mesh::VIndex a = face[e], b = face[(e+1)%3];
						if (a > b) std::swap(a, b);
						auto it = edgeFaces.find({a, b});
						if (it == edgeFaces.end()) continue;
						Mesh::FIndex nb = (it->second.first == f) ? it->second.second : it->second.first;
						if (nb == (Mesh::FIndex)~0u) continue;
						if (!isBndFace[nb]) {
							sumNbrZ += faceCentZ[nb];
							++nbrCount;
						}
					}
					if (nbrCount > 0) {
						const float avgNbrZ = sumNbrZ / (float)nbrCount;
						// Face is skirt if its centroid is below neighbor average
						// (any amount — even tiny descent = start of fold-over)
						if (faceCentZ[f] < avgNbrZ - 1e-6f) {
							++passculled;
							continue;
						}
					}
					newFacesCur.Insert(face);
				}
				if (passculled == 0) break; // converged
				totalSkirtCulled += passculled;
				mesh.faces.Swap(newFacesCur);
			}
			if (totalSkirtCulled > 0) {
				// Final vertex compaction
				const Mesh::VIndex numVFinal = mesh.vertices.GetSize();
				const Mesh::FIndex numFFinal = mesh.faces.GetSize();
				std::vector<uint8_t> keepVF(numVFinal, 0);
				const Mesh::Face* __restrict pFF = mesh.faces.GetData();
				for (Mesh::FIndex f = 0; f < numFFinal; ++f) {
					keepVF[pFF[f][0]] = keepVF[pFF[f][1]] = keepVF[pFF[f][2]] = 1;
				}
				std::vector<Mesh::VIndex> remapF(numVFinal);
				Mesh::VIndex vWriteF = 0;
				for (Mesh::VIndex v = 0; v < numVFinal; ++v)
					remapF[v] = keepVF[v] ? vWriteF++ : NO_ID;
				if (vWriteF < numVFinal) {
					Mesh::VertexArr newVertsF;
					newVertsF.Resize(vWriteF);
					Mesh::Vertex* __restrict pVDstF = newVertsF.GetData();
					const Mesh::Vertex* __restrict pVSrcF = mesh.vertices.GetData();
					for (Mesh::VIndex v = 0; v < numVFinal; ++v)
						if (keepVF[v]) pVDstF[remapF[v]] = pVSrcF[v];
					Mesh::Face* __restrict pFMF = mesh.faces.GetData();
					for (Mesh::FIndex f = 0; f < numFFinal; ++f) {
						pFMF[f][0] = remapF[pFMF[f][0]];
						pFMF[f][1] = remapF[pFMF[f][1]];
						pFMF[f][2] = remapF[pFMF[f][2]];
					}
					mesh.vertices.Swap(newVertsF);
				}
				MESH_DIAG("Poisson: skirt cull removed %u boundary faces (Z-descent)", (unsigned)totalSkirtCulled);
			}
		}
	}
#endif // SKIRT CULL DISABLED

	// Connected-component cleanup: remove small isolated face patches ("blobs")
	// that survive the skirt cull. These are orphaned groups of faces disconnected
	// from the main surface. Keep only components larger than 0.5% of total faces.
	{
		if (mesh.faces.GetSize() > 100) {
			const Mesh::FIndex numF = mesh.faces.GetSize();
			const Mesh::VIndex numV = mesh.vertices.GetSize();
			const Mesh::Face* __restrict pF = mesh.faces.GetData();

			// Build face adjacency via shared edges
			struct EdgeHash {
				size_t operator()(const std::pair<Mesh::VIndex,Mesh::VIndex>& e) const {
					return std::hash<uint64_t>()(((uint64_t)e.first << 32) | e.second);
				}
			};
			std::unordered_map<std::pair<Mesh::VIndex,Mesh::VIndex>, Mesh::FIndex, EdgeHash> edgeToFace;
			edgeToFace.reserve(numF * 3);
			std::vector<std::vector<Mesh::FIndex>> faceAdj(numF);
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				const Mesh::Face& face = pF[f];
				for (int e = 0; e < 3; ++e) {
					Mesh::VIndex a = face[e], b = face[(e+1)%3];
					if (a > b) std::swap(a, b);
					auto it = edgeToFace.find({a, b});
					if (it != edgeToFace.end()) {
						faceAdj[f].push_back(it->second);
						faceAdj[it->second].push_back(f);
					} else {
						edgeToFace[{a, b}] = f;
					}
				}
			}

			// BFS to find connected components
			std::vector<Mesh::FIndex> compId(numF, (Mesh::FIndex)~0u);
			std::vector<Mesh::FIndex> compSize;
			std::vector<Mesh::FIndex> queue;
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				if (compId[f] != (Mesh::FIndex)~0u) continue;
				const Mesh::FIndex cid = (Mesh::FIndex)compSize.size();
				Mesh::FIndex cnt = 0;
				queue.clear();
				queue.push_back(f);
				compId[f] = cid;
				while (!queue.empty()) {
					const Mesh::FIndex cur = queue.back(); queue.pop_back();
					++cnt;
					for (Mesh::FIndex nb : faceAdj[cur]) {
						if (compId[nb] == (Mesh::FIndex)~0u) {
							compId[nb] = cid;
							queue.push_back(nb);
						}
					}
				}
				compSize.push_back(cnt);
			}

			// Find the largest component; remove anything < 0.5% of total
			//
			// KNOWN UNITS PROBLEM -- this threshold is not scene-invariant. It is a fraction
			// of the FACE COUNT, and face count depends on both scene size and octree cell,
			// so the physical area it deletes swings ~58x across the corpus:
			//   Randy 51 m2 | Niwot 148 | Redy 224 | RichmondHistoric 762 | RichmondWater 1019
			//   MechanicFalls 1210 | SchnellTests 1839 | Marco 1850 | GEOTAG 2973
			// (area = (numF/200) * cell^2 / 2). A scene solved at a finer depth deletes
			// BIGGER physical pieces, which is backwards -- the finer solve is exactly the
			// one that fragments real surface into detachable pieces.
			//
			// NOT yet converted to an area threshold, because area alone does not separate
			// the labelled scenes either: SchnellTests (1839 m2) and GEOTAG (2973 m2) delete
			// more than RichmondHistoric (762 m2) and are both fine. Whether a deleted piece
			// matters depends on whether it is real terrain, not on how big it is -- hence
			// the -v 3 export below, which answers that by inspection instead of by proxy.
			const Mesh::FIndex minCompSize = std::max<Mesh::FIndex>(numF / 200, 10);
			// [MESH-FRAG] INSTRUMENTATION, no behaviour attached yet.
			//
			// Whether a deeper octree is worth taking is NOT decided by cell size alone:
			// MEASURED, Niwot at depth 10 solved into 115 components on 3.0M faces and
			// looked right, while RichmondHistoric at depth 11 solved into 1510 on 13.1M
			// and lost its whole fringe -- 4x the faces but 13x the components, and the
			// filter below then deleted 494k faces of real perimeter terrain. Cell/GSD
			// cannot see that: RichmondHistoric had the LARGER cell relative to its median
			// spacing (8.0x vs 5.1x) and fragmented anyway, because the failure is in the
			// sparse fringe and the median is blind to it.
			//
			// So the quantity to gate on is connectivity, normalised by mesh size:
			// components per million faces. Niwot 38/M, RichmondHistoric 115/M. Logged on
			// every run so a threshold can be set from a real corpus rather than fitted to
			// two scenes -- grep [MESH-FRAG] across runs and pair it with whether the
			// output looked right.
			//
			// Also reported: the largest component's share, which separates "one sheet
			// plus confetti" (healthy: share ~1.0) from "the surface came apart"
			// (share well below 1). A scene can have many components and still be fine if
			// they are all tiny.
			Mesh::FIndex largestComp = 0;
			for (size_t c = 0; c < compSize.size(); ++c)
				if (compSize[c] > largestComp) largestComp = compSize[c];
			// GAP RESCUE. Size is the wrong question for this filter.
			//
			// OBSERVED on RichmondHistoric at depth 11: the 495k faces this removes are
			// scattered right across the scene, not concentrated where the output looks
			// wrong -- yet the regions that ARE missing (centre-bottom, upper-right) have
			// their surface in there too. Both at once means the removed set is a MIXTURE:
			// mostly redundant shells sitting directly over surface the kept mesh already
			// has (Poisson emits these near the main sheet at fine depths, and deleting
			// them is correct), plus a minority that are the ONLY surface in their spot,
			// where deleting them opens a hole.
			//
			// No size threshold can tell those apart -- they are the same size. What
			// separates them is whether the kept mesh already covers that ground. So:
			// rasterise the KEPT components into an XY grid, then rescue any small
			// component whose footprint is mostly NOT already covered.
			//
			// Z is deliberately ignored: a shell floating above a roof projects onto the
			// same cells as the roof and is correctly judged redundant, which is the whole
			// point. Set POISSON_COMP_GAP_RESCUE 0 to restore the size-only behaviour.
			std::vector<uint8_t> compRescue(compSize.size(), 0);
			size_t nRescued = 0, nRescuedFaces = 0;
#if POISSON_COMP_GAP_RESCUE
			if (numF > 0) {
				float bx0 = FLT_MAX, by0 = FLT_MAX, bx1 = -FLT_MAX, by1 = -FLT_MAX;
				for (Mesh::VIndex v = 0; v < numV; ++v) {
					const Mesh::Vertex& p = mesh.vertices[v];
					if (p.x < bx0) bx0 = p.x;  if (p.x > bx1) bx1 = p.x;
					if (p.y < by0) by0 = p.y;  if (p.y > by1) by1 = p.y;
				}
				const float span = std::max(bx1 - bx0, by1 - by0);
				if (span > 0.f) {
					constexpr int GDIM = 1024;
					const float gcell = span / (float)GDIM;
					const float ginv = 1.f / gcell;
					const int ggw = std::min(GDIM + 2, (int)((bx1 - bx0) * ginv) + 2);
					const int ggh = std::min(GDIM + 2, (int)((by1 - by0) * ginv) + 2);
					const size_t gN = (size_t)ggw * ggh;
					const auto gcellOf = [&](const Mesh::Face& fc, int& cx, int& cy) {
						const Mesh::Vertex& a = mesh.vertices[fc[0]];
						const Mesh::Vertex& b = mesh.vertices[fc[1]];
						const Mesh::Vertex& c = mesh.vertices[fc[2]];
						cx = (int)(((a.x + b.x + c.x) / 3.f - bx0) * ginv);
						cy = (int)(((a.y + b.y + c.y) / 3.f - by0) * ginv);
						if (cx < 0) cx = 0; else if (cx >= ggw) cx = ggw - 1;
						if (cy < 0) cy = 0; else if (cy >= ggh) cy = ggh - 1;
					};
					// Ground the KEPT components cover.
					std::vector<uint8_t> covered(gN, 0);
					for (Mesh::FIndex f = 0; f < numF; ++f) {
						if (compSize[compId[f]] < minCompSize) continue;
						int cx, cy; gcellOf(pF[f], cx, cy);
						covered[(size_t)cy * ggw + cx] = 1;
					}
					// Per small component: cells touched, and how many were already covered.
					std::vector<uint32_t> cTot(compSize.size(), 0), cHit(compSize.size(), 0);
					std::vector<uint32_t> seen(gN, 0xFFFFFFFFu);
					for (Mesh::FIndex f = 0; f < numF; ++f) {
						const uint32_t cid = compId[f];
						if (compSize[cid] >= minCompSize) continue;
						int cx, cy; gcellOf(pF[f], cx, cy);
						const size_t k = (size_t)cy * ggw + cx;
						if (seen[k] == cid) continue;   // count each cell once per component
						seen[k] = cid;
						++cTot[cid];
						if (covered[k]) ++cHit[cid];
					}
					for (size_t c = 0; c < compSize.size(); ++c) {
						if (compSize[c] >= minCompSize || cTot[c] == 0) continue;
						const double overlap = (double)cHit[c] / (double)cTot[c];
						if (overlap < POISSON_COMP_GAP_OVERLAP) {
							compRescue[c] = 1;
							++nRescued;
							nRescuedFaces += compSize[c];
						}
					}
				}
			}
#endif
			size_t blobsRemoved = 0;
			Mesh::FaceArr newFaces;
			newFaces.Reserve(numF);
			// What the filter deletes, kept so it can be written out and LOOKED AT rather
			// than inferred from counts. See the export below.
			Mesh::FaceArr cutFaces;
			// -v 4 (MVS_DUMP_FILES), NOT the MESH_DIAG build switch: a build compiled for
			// readable logs must not also start writing a 5.9M-vertex PLY every run. The
			// two previous triggers were both wrong for the same reason in different
			// ways -- -v 3 is a level reached for routinely, and OPENMVS_DUMP_REMOVED,
			// once exported, stays exported -- so this wrote on every run unnoticed.
			// See MVS_DUMP_FILES in Common.h for why file dumps sit on verbosity while
			// log text sits on a compile define.
			const bool bWantCut = MVS_DUMP_FILES();
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				const uint32_t cid = compId[f];
				if (compSize[cid] < minCompSize && !compRescue[cid]) {
					++blobsRemoved;
					if (bWantCut)
						cutFaces.Insert(pF[f]);
					continue;
				}
				newFaces.Insert(pF[f]);
			}
			if (nRescued > 0)
				MESH_DIAG("Poisson: gap rescue kept %zu of %zu small components (%zu faces,"
					" %.4f of mesh) whose ground the kept mesh does not already cover"
					" (overlap < %.2f)", nRescued, compSize.size(), nRescuedFaces,
					numF ? (double)nRescuedFaces / (double)numF : 0.0,
					(double)POISSON_COMP_GAP_OVERLAP);
			// EXPORT THE DELETED MATERIAL (-v 3). Four rounds of metrics have not settled
			// whether this filter removes real terrain or floating noise, and every scalar
			// tried so far (components/M faces, overhang percentiles, absolute area
			// threshold) fails to separate the labelled scenes. Looking at the geometry
			// answers it directly: load this next to the final mesh and the deleted parts
			// are either the missing perimeter or they are not.
			//
			// Written BEFORE the vertex compaction below, so face indices still address the
			// pre-filter vertex array.
			// Swap rather than copy: the vertex array stays exactly where it is and only
			// the face list is exchanged, so there is no aliasing between two Mesh objects
			// and no second copy of a multi-million-vertex array. The written PLY carries
			// the full vertex set with only the removed faces indexing into it -- the
			// unreferenced vertices are harmless in any viewer.
			if (bWantCut && cutFaces.GetSize() > 0) {
				const String cutPath(MAKE_PATH("poisson_removed_components.ply"));
				mesh.faces.Swap(cutFaces);          // mesh.faces := removed
				const bool bSaved = mesh.Save(cutPath);
				mesh.faces.Swap(cutFaces);          // restore; cutFaces := removed again
				if (bSaved)
					VERBOSE("Poisson: wrote the %u removed faces to %s (-v 4 diagnostic)",
						(unsigned)cutFaces.GetSize(), cutPath.c_str());
			}
			if (blobsRemoved > 0) {
				mesh.faces.Swap(newFaces);
				// Compact vertices
				const Mesh::FIndex numFK = mesh.faces.GetSize();
				std::vector<uint8_t> keepV(numV, 0);
				const Mesh::Face* __restrict pFK = mesh.faces.GetData();
				for (Mesh::FIndex f = 0; f < numFK; ++f) {
					keepV[pFK[f][0]] = keepV[pFK[f][1]] = keepV[pFK[f][2]] = 1;
				}
				std::vector<Mesh::VIndex> remap(numV);
				Mesh::VIndex vW = 0;
				for (Mesh::VIndex v = 0; v < numV; ++v)
					remap[v] = keepV[v] ? vW++ : NO_ID;
				if (vW < numV) {
					Mesh::VertexArr nv; nv.Resize(vW);
					Mesh::Vertex* __restrict pDst = nv.GetData();
					const Mesh::Vertex* __restrict pSrc = mesh.vertices.GetData();
					for (Mesh::VIndex v = 0; v < numV; ++v)
						if (keepV[v]) pDst[remap[v]] = pSrc[v];
					Mesh::Face* __restrict pFM = mesh.faces.GetData();
					for (Mesh::FIndex f = 0; f < numFK; ++f) {
						pFM[f][0] = remap[pFM[f][0]];
						pFM[f][1] = remap[pFM[f][1]];
						pFM[f][2] = remap[pFM[f][2]];
					}
					mesh.vertices.Swap(nv);
				}
				// NOTE the second %u is the TOTAL component count, not the number removed
				// -- the old wording ("in %u small components") read as the latter and was
				// misleading when the two differ by orders of magnitude.
				MESH_DIAG("Poisson: removed %u faces, %u components total (threshold %u faces)",
					(unsigned)blobsRemoved, (unsigned)compSize.size(), (unsigned)minCompSize);
			}
			// Emitted whenever the diagnostics are asked for, including when nothing was
			// removed -- a run that fragmented but stayed above the threshold is exactly
			// as interesting for calibration as one that did not.
			//
			// mesh_frag.txt no longer goes with it. Unlike mesh_plan.txt nothing consumes
			// that file -- it exists so a corpus of runs can be collected by hand -- and
			// the LINE is what makes it collectable in the first place. Printing text and
			// dropping a file in the output directory are separate decisions, so the line
			// rides MESH_DIAG and the file rides -v 4 (MVS_DUMP_FILES), same split as
			// poisson_removed_components.ply above.
			if (MESH_DIAG_ENABLED()) {
				char fragLine[512];
				snprintf(fragLine, sizeof(fragLine),
					"[MESH-FRAG] depth=%d faces=%u components=%u perM=%.1f"
					" largest=%u largestFrac=%.4f removed=%u removedFrac=%.4f threshold=%u",
					depth, (unsigned)numF, (unsigned)compSize.size(),
					numF ? (1.0e6 * (double)compSize.size() / (double)numF) : 0.0,
					(unsigned)largestComp,
					numF ? (double)largestComp / (double)numF : 0.0,
					(unsigned)blobsRemoved,
					numF ? (double)blobsRemoved / (double)numF : 0.0,
					(unsigned)minCompSize);
				VERBOSE("%s", fragLine);
				// Durable copy at a STABLE path. The app logs carry a fresh timestamp suffix
				// every run and live in a temp folder, so collecting this across a corpus by
				// hand means globbing for the newest file each time. Its own file rather than
				// a second line in mesh_plan.txt, which the orchestrator parses.
				if (MVS_DUMP_FILES()) {
					const String fragPath(MAKE_PATH("mesh_frag.txt"));
					std::ofstream fragOut(fragPath.c_str(), std::ios::out | std::ios::trunc);
					if (fragOut)
						fragOut << fragLine << "\n";
				}
			}
		}
	}

	// Boundary-edge Laplacian smooth: the SurfaceTrimmer cuts along octree cells,
	// leaving a staircase boundary. Iteratively relax boundary vertices AND their
	// 1-ring interior neighbors toward the average of their neighbors. Boundary
	// vertices get lambda=0.7 (strong); 1-ring interior band gets lambda=0.3
	// (gentle transition). This smooths both the boundary polyline AND the adjacent
	// surface so the staircase disappears visually, not just geometrically.
	{
		const int kBoundarySmooth = POISSON_BOUNDARY_SMOOTH; // iterations; 0 = disable
		if (kBoundarySmooth > 0 && mesh.faces.GetSize() > 0) {
			const Mesh::VIndex numV = mesh.vertices.GetSize();
			const Mesh::FIndex numF = mesh.faces.GetSize();
			// Count edge uses: boundary edge = used by exactly 1 face.
			struct EdgeHash {
				size_t operator()(const std::pair<Mesh::VIndex,Mesh::VIndex>& e) const {
					return std::hash<uint64_t>()(((uint64_t)e.first << 32) | e.second);
				}
			};
			std::unordered_map<std::pair<Mesh::VIndex,Mesh::VIndex>, uint8_t, EdgeHash> edgeCount;
			edgeCount.reserve(numF * 3);
			const Mesh::Face* __restrict pF = mesh.faces.GetData();
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				const Mesh::Face& face = pF[f];
				for (int e = 0; e < 3; ++e) {
					Mesh::VIndex a = face[e], b = face[(e+1)%3];
					if (a > b) std::swap(a, b);
					++edgeCount[{a, b}];
				}
			}
			// Build full mesh adjacency (all edges, not just boundary).
			std::vector<std::vector<Mesh::VIndex>> allNbrs(numV);
			for (const auto& kv : edgeCount) {
				allNbrs[kv.first.first].push_back(kv.first.second);
				allNbrs[kv.first.second].push_back(kv.first.first);
			}
			// Classify vertices: 0=interior(untouched), 1=boundary, 2=band(1-ring of boundary), 3=band2(2-ring).
			std::vector<uint8_t> vClass(numV, 0);
			for (const auto& kv : edgeCount) {
				if (kv.second == 1) { // boundary edge
					vClass[kv.first.first] = 1;
					vClass[kv.first.second] = 1;
				}
			}
			// Mark 1-ring interior band (connected to a boundary vertex but not itself boundary).
			for (Mesh::VIndex v = 0; v < numV; ++v) {
				if (vClass[v] != 1) continue;
				for (Mesh::VIndex n : allNbrs[v]) {
					if (vClass[n] == 0) vClass[n] = 2;
				}
			}
			// Mark 2-ring band (connected to a 1-ring band vertex but not already classified).
			for (Mesh::VIndex v = 0; v < numV; ++v) {
				if (vClass[v] != 2) continue;
				for (Mesh::VIndex n : allNbrs[v]) {
					if (vClass[n] == 0) vClass[n] = 3;
				}
			}
			// Iterative TANGENTIAL Laplacian relaxation with per-class lambda.
			// The tangent constraint projects out the vertex-normal component of
			// the displacement so vertices slide along the surface (rounding the
			// boundary from above) but never droop off it into Poisson's "skirt".
			const float lambdaBnd = 0.8f;   // boundary vertices: strong
			const float lambdaBand = 0.4f;  // 1-ring band: moderate
			const float lambdaBand2 = 0.15f; // 2-ring band: gentle feather

			// Precompute per-vertex normals (area-weighted face normals) for
			// all affected vertices. Only needs to be approximate — it just
			// prevents the smooth from pulling in the off-surface direction.
			std::vector<Point3f> vNormals(numV, Point3f(0.f, 0.f, 0.f));
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				const Mesh::Face& face = pF[f];
				// Only bother with faces touching affected vertices
				if (!vClass[face[0]] && !vClass[face[1]] && !vClass[face[2]]) continue;
				const Mesh::Vertex& a = mesh.vertices[face[0]];
				const Mesh::Vertex& b = mesh.vertices[face[1]];
				const Mesh::Vertex& c = mesh.vertices[face[2]];
				// Cross product (area-weighted normal)
				const float e1x = b.x-a.x, e1y = b.y-a.y, e1z = b.z-a.z;
				const float e2x = c.x-a.x, e2y = c.y-a.y, e2z = c.z-a.z;
				const float nx = e1y*e2z - e1z*e2y;
				const float ny = e1z*e2x - e1x*e2z;
				const float nz = e1x*e2y - e1y*e2x;
				for (int v = 0; v < 3; ++v) {
					if (vClass[face[v]]) {
						vNormals[face[v]].x += nx;
						vNormals[face[v]].y += ny;
						vNormals[face[v]].z += nz;
					}
				}
			}
			// Normalize
			for (Mesh::VIndex v = 0; v < numV; ++v) {
				if (!vClass[v]) continue;
				Point3f& n = vNormals[v];
				const float len = FastSqrtS(n.x*n.x + n.y*n.y + n.z*n.z);
				if (len > 1e-8f) { n.x /= len; n.y /= len; n.z /= len; }
				else { n.x = 0.f; n.y = 0.f; n.z = 1.f; } // fallback: up
			}

			Mesh::Vertex* __restrict pV = mesh.vertices.GetData();
			std::vector<Mesh::Vertex> tmp(numV);
			for (int iter = 0; iter < kBoundarySmooth; ++iter) {
				for (Mesh::VIndex v = 0; v < numV; ++v) {
					const uint8_t cls = vClass[v];
					if (cls == 0) { tmp[v] = pV[v]; continue; }
					const auto& nbrs = allNbrs[v];
					if (nbrs.empty()) { tmp[v] = pV[v]; continue; }
					float sx = 0.f, sy = 0.f, sz = 0.f;
					for (Mesh::VIndex n : nbrs) {
						sx += pV[n].x; sy += pV[n].y; sz += pV[n].z;
					}
					const float inv = 1.f / (float)nbrs.size();
					const float lam = (cls == 1) ? lambdaBnd : (cls == 2) ? lambdaBand : lambdaBand2;
					// Compute full displacement
					float dx = (sx * inv - pV[v].x) * lam;
					float dy = (sy * inv - pV[v].y) * lam;
					float dz = (sz * inv - pV[v].z) * lam;
					// Project out the normal component (tangential-only)
					const Point3f& nrm = vNormals[v];
					const float dot = dx*nrm.x + dy*nrm.y + dz*nrm.z;
					dx -= dot * nrm.x;
					dy -= dot * nrm.y;
					dz -= dot * nrm.z;
					tmp[v].x = pV[v].x + dx;
					tmp[v].y = pV[v].y + dy;
					tmp[v].z = pV[v].z + dz;
				}
				// Write back only affected vertices.
				for (Mesh::VIndex v = 0; v < numV; ++v)
					if (vClass[v]) pV[v] = tmp[v];
			}
			// The per-class tally is only ever printed, so it is counted only when asked for.
			if (MESH_DIAG_ENABLED()) {
				unsigned nBnd = 0, nBand = 0, nBand2 = 0;
				for (Mesh::VIndex v = 0; v < numV; ++v) {
					if (vClass[v] == 1) ++nBnd;
					else if (vClass[v] == 2) ++nBand;
					else if (vClass[v] == 3) ++nBand2;
				}
				MESH_DIAG("Poisson: boundary smooth (%d iter, lambda=%.2f/%.2f/%.2f, %u boundary + %u band1 + %u band2 vertices)",
					kBoundarySmooth, lambdaBnd, lambdaBand, lambdaBand2, nBnd, nBand, nBand2);
			}
		}
	}

	DEBUG_EXTRA("Surface reconstructed: %u vertices, %u faces (%s)",
		mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	MESH_DIAG("Poisson (Tier 2: PoissonRecon%s) reconstructed: %u vertices, %u faces (%s)",
		trimThreshold > 0.f ? "+SurfaceTrimmer" : "", mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	return !mesh.faces.IsEmpty();
} // ReconstructMeshPoisson
/*----------------------------------------------------------------*/

// First, iteratively create a Delaunay triangulation of the existing point-cloud by inserting point by point,
// iif the point to be inserted is not closer than distInsert pixels in at least one of its views to
// the projection of any of already inserted points.
// Next, the score is computed for all the edges of the directed graph composed of points as vertices.
// Finally, graph-cut algorithm is used to split the tetrahedrons in inside and outside,
// and the surface is such extracted.
bool Scene::ReconstructMesh(float distInsert, bool bUseFreeSpaceSupport, bool bUseOnlyROI, unsigned nItersFixNonManifold,
	float kSigma, float kQual, float kb,
	float kf, float kRel, float kAbs, float kOutl,
	float kInf
)
{
	double cpuHz = estimateCpuHz();

	using namespace DELAUNAY;
	ASSERT(!pointcloud.IsEmpty());
	mesh.Release();

	size_t numPtsCloud = pointcloud.GetSize();
	ptrdiff_t* indices = (ptrdiff_t*)_aligned_malloc(sizeof(ptrdiff_t) * numPtsCloud, 64);
	uint32_t* offsets = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numPtsCloud, 64);
	uint32_t* sizes = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numPtsCloud, 64);
#ifdef VALIDATE
	delaunay_t2 delaunay2;
	{

		// create the Delaunay triangulation
		std::vector<cell_info_t> infoCells;
		std::vector<camera_cell_t> camCells;
		std::vector<facet_t> hullFacets;
	{
	TD_TIMER_STARTD();

			std::vector<point_t> vertices(pointcloud.GetSize());
			// fetch points
			if (bUseOnlyROI && !IsBounded())
				bUseOnlyROI = false;
			for (int i = 0; i < pointcloud.GetSize(); ++i) {
				Point3f pp = pointcloud.Point(i);

				const PointCloud::Point X(pp.x, pp.y, pp.z);
				if (bUseOnlyROI && !obb.Intersects(X))
					continue;
				vertices[i] = point_t(X.x, X.y, X.z);
				indices[i] = i;
					}				
			// sort vertices
			typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t2::Geom_traits, point_t*> Search_traits;
			CGAL::spatial_sort(indices.begin(), indices.end(), Search_traits(&vertices[0], delaunay2.geom_traits()));
			// insert vertices
			Util::Progress progress(_T("Points inserted"), indices.size());
			const float distInsertSq(SQUARE(distInsert));
			vertex_handle_t2 hint;
			delaunay_t2::Locate_type lt;
			int li, lj;
			std::for_each(indices.cbegin(), indices.cend(), [&](size_t idx)
				{
					const point_t& p = vertices[idx];
					const PointCloud::Point& point = pointcloud.Point(idx);;
					const uint32_t* views = pointcloud.ViewsStream(idx);
					ASSERT(!views.IsEmpty());
					if (hint == vertex_handle_t2()) {
						// this is the first point,
						// insert it
						hint = delaunay2.insert(p);
						ASSERT(hint != vertex_handle_t2());
					} else {
						if (distInsert <= 0) {
							// insert all points
							hint = delaunay2.insert(p, hint);
							ASSERT(hint != vertex_handle_t2());
						} else {
							// locate cell containing this point
							const cell_handle_t2 c(delaunay2.locate(p, lt, li, lj, hint->cell()));
							if (lt == delaunay_t::VERTEX) {
								// duplicate point, nothing to insert,
								// just update its visibility info
								hint = c->vertex(li);
								ASSERT(hint != delaunay.infinite_vertex());
							} else {
								// locate the nearest vertex
								vertex_handle_t2 nearest;
								if (delaunay2.dimension() < 3) {
									// use a brute-force algorithm if dimension < 3
									delaunay_t2::Finite_vertices_iterator vit = delaunay2.finite_vertices_begin();
									nearest = vit;
									++vit;
									adjacent_vertex_back_inserter_t2 inserter(delaunay2, p, nearest);
									for (delaunay_t2::Finite_vertices_iterator end = delaunay2.finite_vertices_end(); vit != end; ++vit)
										inserter = vit;
								} else {
									// - start with the closest vertex from the located cell
									// - repeatedly take the nearest of its incident vertices if any
									// - if not, we're done
									ASSERT(c != cell_handle_t2());
									nearest = delaunay2.nearest_vertex_in_cell(p, c);
									while (true) {
										const vertex_handle_t2 v(nearest);
										delaunay2.adjacent_vertices(nearest, adjacent_vertex_back_inserter_t2(delaunay2, p, nearest));
										if (v == nearest)
											break;
				}
			}
								ASSERT(nearest == delaunay2.nearest_vertex(p, hint->cell()));
								hint = nearest;
								// check if point is far enough to all existing points
								for (int j = 0; j < pointcloud.ViewsStreamSize(idx); ++j) {
									const Image& imageData = images[views[j]];
									const Point3f pn(imageData.camera.ProjectPointP3(point));
									const Point3f pe(imageData.camera.ProjectPointP3(CGAL2MVS<float>(nearest->point())));
									if (!IsDepthSimilar(pn.z, pe.z) || normSq(Point2f(pn)-Point2f(pe)) > distInsertSq) {
										// point far enough to an existing point,
										// insert as a new point
										hint = delaunay2.insert(p, lt, c, li, lj);
										ASSERT(hint != vertex_handle_t());
										break;
		}
			}
		}
	}
	}
					// update point visibility info
					hint->info().InsertViews(pointcloud, idx);
					++progress;
				});
			progress.close();
}
	std::cerr << "1st has " << delaunay2.number_of_vertices() << "\n";

}
#endif

	std::vector<TMatrix<float,3,4>> viewCameras(images.size());
	std::vector<Frustum> viewFrustums(images.size());
	FOREACH(i, images) {
		Image& imageData = images[i];
		if (!imageData.IsValid())
			continue;
		for (int j = 0; j < imageData.camera.P.elems; ++j) {
			viewCameras[i][j] = (float)imageData.camera.P[j];
		}
		viewFrustums[i] = Frustum(imageData.camera.P, imageData.width, imageData.height, 0, 1);
	}

	// create the Delaunay triangulation
	const size_t numPointCloudVertices = pointcloud.NumPoints();
	if (numPointCloudVertices >= std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("Unsupported");
	}

	delaunay_t delaunay;

	cell_info_t* __restrict infoCells;
	std::vector<camera_cell_t> camCells;
	std::vector<facet_t> hullFacets;
	Point3f* idToPoint;
	// Per-cell topology caches built once at cell enumeration time (after
	// info() and allCells[] are fully populated). Survive through ray-walk
	// weighting and graph-cut build; freed after extraction. Today only
	// BuildGraphNodesAndEdges' Phase 3 reads these; step 2 wires them into
	// intersect()'s FACET fast path so ray walks become flat-array lookups.
	uint32_t* __restrict cellNbrID = nullptr;   // 4 IDs per cell (totalCells*4 entries)
	uint8_t*  __restrict cellNbrSlot = nullptr; // packed 4x2-bit reverse slot per cell

	size_t numVertices;
#ifdef FACET_DIAGNOSTICS
	size_t numFiniteFacets;
	size_t numFacets;
#endif
	size_t numDelaunayVertices;
	//std::unique_ptr<float[]> distsSq;
	float approxMedian;
	point_t* vertices;
	const float distInsertSq(SQUARE(distInsert));
	delaunay_t::Locate_type lt;
	int li, lj;
	size_t totalCells;
	uint8_t* mask;
	cell_handle_t* __restrict allCells;

	{
		TD_TIMER_STARTD();

		point_t* origVertices = (point_t*)_aligned_malloc(sizeof(point_t) * numPointCloudVertices, 64);
		{
			TD_TIMER_STARTD();

			// fetch points
			if (bUseOnlyROI && !IsBounded())
				bUseOnlyROI = false;

			// 21ms
			numVertices =
				bUseOnlyROI ?
				ProcessPoints<true>(
					pointcloud.PointStream(),
					numPointCloudVertices,
					origVertices,
					indices,
					obb
				) :
				ProcessPoints<false>(
					pointcloud.PointStream(),
					numPointCloudVertices,
					origVertices,
					indices,
					obb
				);

#if RECONSTRUCT_VOXEL_PREFILTER
			// VOXEL PRE-FILTER: collapse near-duplicate points (same voxel
			// cell) to a single representative.  This removes redundancy from
			// over-sampled flat regions while leaving isolated boundary /
			// thin-feature points untouched (a sparse cell with one point
			// keeps its single point unconditionally -- no quality loss).
			//
			// Mechanism: bbox -> voxel size = maxSpan / kGrid.  For each
			// point, compute a packed 63-bit voxel key (21 bits per axis).
			// Build (key, pointIdx, conf) tuples; parallel-sort by key;
			// linear-scan to keep the highest-confidence representative per
			// run of equal keys.  conf falls back to 0 (first-arrival wins)
			// if pointWeights is empty.
			//
			// Side benefit: the surviving indices[] come out roughly in
			// z-order, which tends to warm up the subsequent CGAL
			// spatial_sort (a clustered seed input helps it).
			//
			// Skipped in ROI mode (indices[] semantics: ProcessPoints<true>
			// has already compacted indices[i] to non-original-cloud values,
			// breaking pointWeightsOffsets[i] lookup).
			if (!bUseOnlyROI && numVertices > 0) {
				TD_TIMER_STARTD();

				// Pass 1: parallel bbox over the active point set (origVertices[0..numVertices)).
				float bbMinX =  std::numeric_limits<float>::max();
				float bbMinY =  std::numeric_limits<float>::max();
				float bbMinZ =  std::numeric_limits<float>::max();
				float bbMaxX = -std::numeric_limits<float>::max();
				float bbMaxY = -std::numeric_limits<float>::max();
				float bbMaxZ = -std::numeric_limits<float>::max();
#pragma omp parallel
				{
					float lMinX =  std::numeric_limits<float>::max();
					float lMinY =  std::numeric_limits<float>::max();
					float lMinZ =  std::numeric_limits<float>::max();
					float lMaxX = -std::numeric_limits<float>::max();
					float lMaxY = -std::numeric_limits<float>::max();
					float lMaxZ = -std::numeric_limits<float>::max();
#pragma omp for nowait schedule(static)
					for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
						const auto& p = origVertices[i];
						const float x = (float)p.x();
						const float y = (float)p.y();
						const float z = (float)p.z();
						if (x < lMinX) lMinX = x;
						if (y < lMinY) lMinY = y;
						if (z < lMinZ) lMinZ = z;
						if (x > lMaxX) lMaxX = x;
						if (y > lMaxY) lMaxY = y;
						if (z > lMaxZ) lMaxZ = z;
					}
#pragma omp critical
					{
						if (lMinX < bbMinX) bbMinX = lMinX;
						if (lMinY < bbMinY) bbMinY = lMinY;
						if (lMinZ < bbMinZ) bbMinZ = lMinZ;
						if (lMaxX > bbMaxX) bbMaxX = lMaxX;
						if (lMaxY > bbMaxY) bbMaxY = lMaxY;
						if (lMaxZ > bbMaxZ) bbMaxZ = lMaxZ;
					}
				}

				const float spanX = bbMaxX - bbMinX;
				const float spanY = bbMaxY - bbMinY;
				const float spanZ = bbMaxZ - bbMinZ;
				const float maxSpan = std::max(std::max(spanX, spanY), spanZ);

				if (maxSpan > 0.0f) {
					// ---- Pass 2: density probe (auto-tune voxel size) ----
					// Build a coarse 64^3 occupancy histogram; find the peak
					// bin K_peak. Estimate dense-region inter-point spacing
					// assuming a 2D surface populates that bin:
					//   bin_side  = maxSpan / 64
					//   spacing  ~= bin_side / sqrt(K_peak)
					//   voxel     = spacing * VOXEL_SAFETY (config'd)
					// Cap the resulting kGrid to a safe range.
					constexpr int kProbe = 64;
					constexpr int kProbeBins = kProbe * kProbe * kProbe;
					const float probeSide = maxSpan / float(kProbe);
					const float invProbe = 1.0f / probeSide;

					const int nT = omp_get_max_threads();
					std::vector<std::vector<uint32_t>> tProbe(nT,
						std::vector<uint32_t>(kProbeBins, 0));
#pragma omp parallel
					{
						const int tid = omp_get_thread_num();
						auto& lh = tProbe[tid];
#pragma omp for schedule(static)
						for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
							const auto& p = origVertices[i];
							int bx = (int)(((float)p.x() - bbMinX) * invProbe);
							int by = (int)(((float)p.y() - bbMinY) * invProbe);
							int bz = (int)(((float)p.z() - bbMinZ) * invProbe);
							if (bx < 0) bx = 0; else if (bx >= kProbe) bx = kProbe - 1;
							if (by < 0) by = 0; else if (by >= kProbe) by = kProbe - 1;
							if (bz < 0) bz = 0; else if (bz >= kProbe) bz = kProbe - 1;
							++lh[(bz * kProbe + by) * kProbe + bx];
						}
					}
					// Reduce: per-bin sum + max-occupancy.
					uint32_t kPeak = 0;
					for (int b = 0; b < kProbeBins; ++b) {
						uint32_t s = 0;
						for (int t = 0; t < nT; ++t) s += tProbe[t][b];
						if (s > kPeak) kPeak = s;
					}

					// Default: fall back to a safe grid if probe is uninformative.
					int kGrid = 8192;
					float voxel = maxSpan / float(kGrid);

					if (kPeak >= 16) {
						// Surface-density assumption: ~K_peak points spread
						// over a (probeSide x probeSide) face of the bin.
						const float estDense = probeSide / std::sqrt((float)kPeak);
						const float safety = float(RECONSTRUCT_VOXEL_SAFETY_X100) / 100.0f;
						const float wantVoxel = estDense * safety;
						// kGrid must be in [256, 65535] so voxel stays in
						// [maxSpan/65535, maxSpan/256]. The lower bound (65535)
						// is the 16-bit-per-axis grid limit; the upper bound
						// (256) prevents catastrophic merging on degenerate data.
						float wantGrid = maxSpan / wantVoxel;
						if (wantGrid < 256.0f)   wantGrid = 256.0f;
						if (wantGrid > 65535.0f) wantGrid = 65535.0f;
						kGrid = (int)wantGrid;
						voxel = maxSpan / float(kGrid);
					}
					const float invVoxel = 1.0f / voxel;
					const bool haveWeights = !pointcloud.pointWeightsMemory.empty();

					// Build (voxelKey, pointIdx, conf) tuples.
					// 16 bytes/entry; for 18M pts -> ~290MB, freed before insertion.
					struct VoxelEntry {
						uint64_t key;       // 21 bits per axis, LE-packed
						uint32_t pointIdx;  // original cloud index
						float    conf;      // tiebreaker; higher wins
					};
					VoxelEntry* __restrict entries = (VoxelEntry*)_aligned_malloc(
						sizeof(VoxelEntry) * numVertices, 64);

#pragma omp parallel for schedule(static)
					for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
						const auto& p = origVertices[i];
						const float x = (float)p.x();
						const float y = (float)p.y();
						const float z = (float)p.z();
						uint32_t vx = (uint32_t)((x - bbMinX) * invVoxel);
						uint32_t vy = (uint32_t)((y - bbMinY) * invVoxel);
						uint32_t vz = (uint32_t)((z - bbMinZ) * invVoxel);
						// Clamp the boundary point that lands on the upper edge.
						if (vx >= (uint32_t)kGrid) vx = (uint32_t)kGrid - 1;
						if (vy >= (uint32_t)kGrid) vy = (uint32_t)kGrid - 1;
						if (vz >= (uint32_t)kGrid) vz = (uint32_t)kGrid - 1;
						const uint64_t key =
							((uint64_t)vx)         |
							((uint64_t)vy << 21)   |
							((uint64_t)vz << 42);

						float c = 0.0f;
						if (haveWeights) {
							const uint32_t off = pointcloud.pointWeightsOffsets[i];
							const uint32_t cnt = pointcloud.pointWeightsSizes[i];
							for (uint32_t k = 0; k < cnt; ++k) {
								const float w = pointcloud.pointWeightsMemory[off + k];
								if (w > c) c = w;
							}
						}

						entries[i].key      = key;
						entries[i].pointIdx = (uint32_t)i;
						entries[i].conf     = c;
					}

					// Parallel sort by voxel key.  TBB's parallel_sort scales
					// well on 18M elements (~0.4-0.7s on 32-thread host).
					tbb::parallel_sort(entries, entries + numVertices,
						[](const VoxelEntry& a, const VoxelEntry& b) {
							return a.key < b.key;
						});

					// Linear scan: within each run of equal keys, keep the
					// highest-conf entry. Sequential by design (run-length
					// detection is cheaper than parallel reduction here, and
					// we're streaming a sorted array -- L1/L2 friendly).
					size_t outIdx = 0;
					size_t i = 0;
					const size_t N = (size_t)numVertices;
					while (i < N) {
						const uint64_t k = entries[i].key;
						size_t bestEntry = i;
						float  bestConf  = entries[i].conf;
						size_t j = i + 1;
						while (j < N && entries[j].key == k) {
							if (entries[j].conf > bestConf) {
								bestConf  = entries[j].conf;
								bestEntry = j;
							}
							++j;
						}
						indices[outIdx++] = (ptrdiff_t)entries[bestEntry].pointIdx;
						i = j;
					}

					const size_t before  = (size_t)numVertices;
					const size_t dropped = before - outIdx;
					numVertices = outIdx;

					MESH_DIAG("Voxel pre-filter (auto: kPeak=%u, grid=%d, voxel=%.4g, bbox=[%.1f x %.1f x %.1f]): %zu/%zu dropped (%.1f%%) [%s]",
						kPeak, kGrid, voxel, spanX, spanY, spanZ,
						dropped, before,
						100.0 * (double)dropped / (double)before,
						TD_TIMER_GET_FMT().c_str());

					_aligned_free(entries);
				} else {
					MESH_DIAG("Voxel pre-filter: degenerate bbox -- skipped");
				}
			} else {
				MESH_DIAG("Voxel pre-filter: skipped (%s)",
					bUseOnlyROI ? "ROI mode" : "empty");
			}
#endif

#if RECONSTRUCT_EARLY_CONFIDENCE_FILTER
			// EARLY confidence filter: drop the bottom-KPCT% lowest-confidence
			// points NOW so that spatial_sort / SOR / DT insertion all see a
			// smaller N.  Operates by compacting indices[] in-place; leaves
			// origVertices[] at full size (it's freed right after permute and
			// the bytes are unused anyway).  Spatial_sort consults
			// origVertices[indices[i]] -- still valid.  permuteScatter2 looks
			// up pointcloud arrays via indices[i] -- still valid (indices[]
			// holds original cloud indices throughout).
			//
			// Skipped in ROI mode (ProcessPoints<true> compacts indices to
			// non-original-cloud indices, breaking pointWeights lookup).
			// Skipped if pointWeights empty (nothing to score with).
			if (!bUseOnlyROI && !pointcloud.pointWeightsMemory.empty()) {
				TD_TIMER_STARTD();
				constexpr unsigned kPct = RECONSTRUCT_CONFIDENCE_FILTER_KPCT;
				static_assert(kPct > 0 && kPct < 100,
					"RECONSTRUCT_CONFIDENCE_FILTER_KPCT must be in (0, 100)");

				float* __restrict ptConf = (float*)_aligned_malloc(
					sizeof(float) * numVertices, 64);

				// Pass 1: per-point max-weight + global min/max.
				// In non-ROI mode after ProcessPoints, indices[i] == i, so we
				// can index pointWeightsOffsets/Sizes directly by i.
				float gMin =  std::numeric_limits<float>::max();
				float gMax = -std::numeric_limits<float>::max();
#pragma omp parallel
				{
					float lMin =  std::numeric_limits<float>::max();
					float lMax = -std::numeric_limits<float>::max();
#pragma omp for nowait schedule(static)
					for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
						const uint32_t off = pointcloud.pointWeightsOffsets[i];
						const uint32_t cnt = pointcloud.pointWeightsSizes[i];
						float best = 0.0f;
						for (uint32_t k = 0; k < cnt; ++k) {
							const float w = pointcloud.pointWeightsMemory[off + k];
							if (w > best) best = w;
						}
						ptConf[i] = best;
						if (best < lMin) lMin = best;
						if (best > lMax) lMax = best;
					}
#pragma omp critical
					{
						if (lMin < gMin) gMin = lMin;
						if (lMax > gMax) gMax = lMax;
					}
				}

				const float relSpread = (gMax > 0.0f)
					? (gMax - gMin) / gMax
					: 0.0f;
				constexpr float kMinRelSpread = 0.10f; // 10%

				if (gMax > gMin && relSpread >= kMinRelSpread) {
					// Pass 2: 4096-bin histogram.
					constexpr int kBins = 4096;
					const float invSpan = float(kBins) / (gMax - gMin);

					const int nT = omp_get_max_threads();
					std::vector<std::vector<size_t>> tHist(nT,
						std::vector<size_t>(kBins, 0));

#pragma omp parallel
					{
						const int tid = omp_get_thread_num();
						auto& lh = tHist[tid];
#pragma omp for schedule(static)
						for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
							int b = (int)((ptConf[i] - gMin) * invSpan);
							if (b < 0) b = 0;
							else if (b >= kBins) b = kBins - 1;
							++lh[b];
						}
					}
					std::vector<size_t> hist(kBins, 0);
					for (int t = 0; t < nT; ++t)
						for (int b = 0; b < kBins; ++b)
							hist[b] += tHist[t][b];

					const size_t target = ((size_t)numVertices * kPct + 99) / 100;
					size_t cum = 0;
					int cutBin = 0;
					for (; cutBin < kBins; ++cutBin) {
						cum += hist[cutBin];
						if (cum >= target) break;
					}
					const float thr = gMin + (float)(cutBin + 1) / invSpan;

					// Pass 3: in-place compaction of indices[].  Sequential
					// because order matters (we want surviving indices to
					// retain their relative order so spatial_sort sees a
					// reasonable initial layout).  origVertices left intact.
					size_t outIdx = 0;
					for (size_t i = 0; i < (size_t)numVertices; ++i) {
						if (ptConf[i] >= thr) {
							indices[outIdx++] = (ptrdiff_t)i;
						}
					}
					const size_t before = numVertices;
					const size_t dropped = before - outIdx;
					numVertices = outIdx;

					MESH_DIAG("Early confidence filter (bottom %u%%, range=[%.4g,%.4g], thr=%.4g): %zu/%zu dropped (%.1f%%) [%s]",
						kPct, gMin, gMax, thr, dropped, before,
						100.0 * (double)dropped / (double)before,
						TD_TIMER_GET_FMT().c_str());
				} else {
					MESH_DIAG("Early confidence filter: range=[%.4g,%.4g] relSpread=%.3f -- skipped (uniform or below %.0f%% threshold)",
						gMin, gMax, relSpread, kMinRelSpread * 100.0f);
				}

				_aligned_free(ptConf);
			} else {
				MESH_DIAG("Early confidence filter: skipped (%s)",
					bUseOnlyROI ? "ROI mode" : "no pointWeights");
			}
#endif

#ifndef VALIDATE
			// sort vertices (1.17s)
			typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t::Geom_traits, point_t*> Search_traits;
			// Sequential_tag: nth_element is non-stable, so parallel splits
			// (tbb::parallel_invoke) produce non-deterministic output ordering
			// among equal-coordinate points. This causes 7s+ insertion variance.
			// Sequential sort is ~1.2s — negligible vs 30s insertion.
#if RECONSTRUCT_HILBERT_SORT
			// hilbert_sort: skip the random-shuffle splitter inside spatial_sort.
			// Same Hilbert-curve locality for DT insertion, ~0.3-0.8s faster on
			// 8M+ vertex scenes. The randomization in spatial_sort hardens against
			// adversarial inputs; not needed here.
			CGAL::hilbert_sort<CGAL::Sequential_tag>(
				indices, indices + numVertices,
				Search_traits(&origVertices[0], delaunay.geom_traits())
			);
#else
			CGAL::spatial_sort<CGAL::Sequential_tag>(
				indices, indices + numVertices,
				Search_traits(&origVertices[0], delaunay.geom_traits())
			);
#endif
#endif

			// origVertices[i] refers to the original data.
			// indices[i] maps the sorted data to the original data.
			// Rewrite the vertex data in index sorted form:

			vertices = (point_t*)_aligned_malloc(sizeof(point_t) * numVertices, 64);
			Point3f* verticesf = (Point3f*)_aligned_malloc(sizeof(Point3f) * numVertices, 64);

			// 69ms
			permuteScatter2(
				indices,
				numVertices,
				origVertices, vertices, verticesf,
				pointcloud.pointViewsOffsets, offsets,
				pointcloud.pointViewsSizes, sizes
			);
			_aligned_free(origVertices);
			origVertices = 0;
			// NOTE: 'indices' (sorted->original mapping) is freed AFTER the
			// optional confidence filter below, which needs to look up
			// pointWeights in original index space.

			// The points of the cloud are now kept in vertices.  Ancillary data, for view information,
			// is also maintained.
			// Go through the cloud and eliminate outliers, but do so in-place (without disturbing
			// the ancillary data.
			// Mask[i] is a boolean indicating 0 = outlier.
			mask = (uint8_t*)_aligned_malloc(sizeof(uint8_t) * numVertices, 64);
			// 8/16 about the same, but better than 32, 64 much worse
			// 16 removes more outliers.
			// 1.39s/1.286s
			// EDGING A/B: keep more sparse boundary/canopy-fringe points. The
			// tight variant below (16, 2) cuts the upper band at 2-sigma, which
			// removes the sparse points that ARE the silhouette/canopy fringe
			// before the Delaunay ever sees them (measured ~34% boundary-point
			// loss). Set this guard to 0 to restore the tight cut.
#if 1
			// - stddevMul=3.0: wider upper band keeps sparse boundary points.
			// - interiorMul=FLT_MAX: disable the lower-bound (over-dense) filter
			//   (lowerThr = mean - FLT_MAX*stdev -> clamped to 1e-6f, keeps everything)
			StatisticalOutlierRemoval(verticesf, numVertices, mask, 16, 3.0f, FLT_MAX);
#else
			StatisticalOutlierRemoval(verticesf, numVertices, mask, 16, 2);
#endif

			_aligned_free(verticesf);
			verticesf = 0;

#if RECONSTRUCT_CONFIDENCE_FILTER && !RECONSTRUCT_EARLY_CONFIDENCE_FILTER
			// (3) Confidence-weighted filter. Per-point scalar confidence is
			// the MAX over its per-view weights (strongest single-view
			// evidence). Threshold = quantile at KPCT% (drops the bottom
			// KPCT% by confidence). Quantile is robust to right-skewed weight
			// distributions where mean - k*stdev would go negative.
			// pointWeights are NOT permuted by permuteScatter2 -- look up the
			// original index via 'indices' (still alive at this point).
			if (pointcloud.pointWeightsMemory.empty()) {
				MESH_DIAG("Confidence filter: pointWeights empty -- skipped");
			} else {
				constexpr unsigned kPct = RECONSTRUCT_CONFIDENCE_FILTER_KPCT;
				static_assert(kPct > 0 && kPct < 100,
					"RECONSTRUCT_CONFIDENCE_FILTER_KPCT must be in (0, 100)");

				float* __restrict ptConf = (float*)_aligned_malloc(
					sizeof(float) * numVertices, 64);

				// Pass 1: per-point max-weight + global min/max for histogram.
				float gMin =  std::numeric_limits<float>::max();
				float gMax = -std::numeric_limits<float>::max();
#pragma omp parallel
				{
					float lMin =  std::numeric_limits<float>::max();
					float lMax = -std::numeric_limits<float>::max();
#pragma omp for nowait schedule(static)
					for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
						const size_t o = (size_t)indices[i];
						const uint32_t off = pointcloud.pointWeightsOffsets[o];
						const uint32_t cnt = pointcloud.pointWeightsSizes[o];
						float best = 0.0f;
						for (uint32_t k = 0; k < cnt; ++k) {
							const float w = pointcloud.pointWeightsMemory[off + k];
							if (w > best) best = w;
						}
						ptConf[i] = best;
						if (mask[i]) {
							if (best < lMin) lMin = best;
							if (best > lMax) lMax = best;
						}
					}
#pragma omp critical
					{
						if (lMin < gMin) gMin = lMin;
						if (lMax > gMax) gMax = lMax;
					}
				}

				// Relative-spread guard: if the weight distribution is nearly
				// flat, the bottom-KPCT% cut becomes a near-random spatial
				// drop and can collapse thin/flat scenes.  Skip when the
				// span is below a small fraction of the upper end.
				const float relSpread = (gMax > 0.0f)
					? (gMax - gMin) / gMax
					: 0.0f;
				constexpr float kMinRelSpread = 0.10f; // 10%

				if (gMax > gMin && relSpread >= kMinRelSpread) {
					// Pass 2: 4096-bin histogram of kept points.
					constexpr int kBins = 4096;
					std::vector<size_t> hist(kBins, 0);
					const float invSpan = float(kBins) / (gMax - gMin);

					const int nT = omp_get_max_threads();
					std::vector<std::vector<size_t>> tHist(nT,
						std::vector<size_t>(kBins, 0));

#pragma omp parallel
					{
						const int tid = omp_get_thread_num();
						auto& lh = tHist[tid];
#pragma omp for schedule(static)
						for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
							if (!mask[i]) continue;
							int b = (int)((ptConf[i] - gMin) * invSpan);
							if (b < 0) b = 0;
							else if (b >= kBins) b = kBins - 1;
							++lh[b];
						}
					}
					for (int t = 0; t < nT; ++t)
						for (int b = 0; b < kBins; ++b)
							hist[b] += tHist[t][b];

					size_t totalKept = 0;
					for (int b = 0; b < kBins; ++b) totalKept += hist[b];

					// Find smallest bin index where cumulative >= kPct% of totalKept.
					const size_t target = (totalKept * kPct + 99) / 100;
					size_t cum = 0;
					int cutBin = 0;
					for (; cutBin < kBins; ++cutBin) {
						cum += hist[cutBin];
						if (cum >= target) break;
					}
					// Threshold = upper edge of cutBin -- drop strictly below.
					const float thr = gMin + (float)(cutBin + 1) / invSpan;

					size_t dropped = 0;
#pragma omp parallel for reduction(+:dropped) schedule(static)
					for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i) {
						if (mask[i] && ptConf[i] < thr) {
							mask[i] = 0;
							++dropped;
						}
					}
					MESH_DIAG("Confidence filter (bottom %u%%, range=[%.4g,%.4g], thr=%.4g): %zu points dropped (%.1f%% of %zu)",
						kPct, gMin, gMax, thr, dropped,
						100.0 * (double)dropped / (double)totalKept,
						totalKept);
				} else {
					MESH_DIAG("Confidence filter: range=[%.4g,%.4g] relSpread=%.3f -- skipped (uniform or below %.0f%% threshold)",
						gMin, gMax, relSpread, kMinRelSpread * 100.0f);
				}

				_aligned_free(ptConf);
			}
#endif

			_aligned_free(indices);
			indices = 0;

			// insert vertices
			// 6x vertices is a generous worst case, but uses too much memory.
			// Potentially allow some dynamic allocation to better keep
			// memory within reasonable limits.
			// 178ms
			delaunay.tds().cells().reserve(numVertices * 4); // May reserve dynamically
			delaunay.tds().vertices().reserve(numVertices); // Should be sufficient to prevent reallocations.

			// Can't avoid initialization.
#if !DIRECT_VIEW_EXPANSION
			allViews = (view_vec_t*)_aligned_malloc(sizeof(view_vec_t) * numVertices, 64);

			//42ms
#pragma omp parallel for schedule(static)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i)
				new (&allViews[i]) view_vec_t();
#else
			pointToVertex = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numVertices, 64);
			// UINT32_MAX sentinel = point was masked out or not inserted.
			// Parallel first-touch: this is a 4*N-byte buffer (e.g. ~68MB at
			// 17M points). A single-threaded memset is bandwidth-bound on one
			// memory channel; parallelizing across cores engages more channels
			// and also performs NUMA-friendly first-touch so subsequent writes
			// during insertion hit local pages. Static schedule keeps each
			// thread's pages contiguous.
			{
				const ptrdiff_t cnt = (ptrdiff_t)numVertices;
				#pragma omp parallel for schedule(static)
				for (ptrdiff_t i = 0; i < cnt; ++i)
					pointToVertex[i] = UINT32_MAX;
			}
#endif

			MESH_DIAG("Total prep time is: %s", TD_TIMER_GET_FMT().c_str());
		}
		Util::Progress progress(_T("Points inserted"), numVertices);

		// Here we keep track of versioning for testing.
		// Both delaunay.info() and vcg::tri::Info() only compile
		// if we are using custom versions of these libraries.
		// The version returned can be used to track revisions
		// and can be manually adjusted.c
		// delaunay.info() is parallel can be used to make sure
		// we are compiling and using the work with TBB.
#if 1
		MESH_DIAG("------------------------------------------");
		MESH_DIAG("ReconstructMesh optimization version 1.1.23");
		const auto [isParallel, CGALversion] = CGAL::info();
		MESH_DIAG("Parallel: %s", isParallel ? "true" : "false");
		MESH_DIAG("CGAL version: = %d", CGALversion);
		constexpr int vcgVersion = vcg::tri::Info();
		MESH_DIAG("VCG version: = %d", vcgVersion);
		MESH_DIAG("------------------------------------------");
#endif
		// Fixed storage is slightly faster, but difficult to maintain.
		constexpr size_t kMaxCells = 16384;
		SmallQueue<cell_handle_t, kMaxCells> cellQueue;

		vertex_handle_t hint;
		const vertex_handle_t infV = delaunay.infinite_vertex();  // cheap pointer compare

		// InsertViews first parameter must be the dt's index --verified by validation code.
		if (distInsert <= 0) {
			for (uint32_t i = 0; i < numVertices; ++i) {
				if (!mask[i]) {
					goto advance;
				}
				const point_t& p = vertices[i]; // These are the sorted vertices.
				// insert all points
				hint = delaunay.insert(p, hint);
				ASSERT(anchor != vertex_handle_t());
				// update point visibility info
#if !DIRECT_VIEW_EXPANSION
				InsertViews(hint->info().idx, pointcloud, i);
#else
				pointToVertex[i] = hint->info().idx;
#endif
			advance:
				if (!(i & 16383)) {
					progress += 16384;
				}
			}
		}
		else {
#if !RECONSTRUCT_FAST_DISTINSERT
			std::vector<uint32_t> vertexMarks(numVertices); // Must be uint32_t
			uint32_t marker = 0;
#endif

			uint32_t i = 0;
			for (bool done = false; !done; ++i) {
				if (mask[i]) {
					hint = delaunay.insert(vertices[i]);
#if !DIRECT_VIEW_EXPANSION
					InsertViews(hint->info().idx, pointcloud, i);
#else
					pointToVertex[i] = hint->info().idx;
#endif
					done = true;
				}
			}

			for (; i < numVertices; ++i) {
				if (!mask[i]) {
					goto advance2;
				}

				const point_t& p = vertices[i];

				uint32_t offset = offsets[i];
				uint32_t numViews = sizes[i];
				const PointCloud::View* __restrict views;

				// Although not strictly needed (the previous anchor->cell() offers a hint),
				// The dt result is significantly smaller if we do refine the hint.
				// Locate starting from last known good cell
				cell_handle_t c = delaunay.locate(p, lt, li, lj, hint->cell());
				vertex_handle_t nearest;
				if (delaunay.dimension() < 3) {
					// use a brute-force algorithm if dimension < 3
					delaunay_t::Finite_vertices_iterator vit = delaunay.finite_vertices_begin();
					nearest = vit;
					++vit;
					adjacent_vertex_back_inserter_t inserter(delaunay, vertices[i], nearest);
					for (delaunay_t::Finite_vertices_iterator end = delaunay.finite_vertices_end(); vit != end; ++vit)
						inserter = vit;

					views = pointcloud.pointViewsMemory.data() + offset;
				}
				else {
#if !RECONSTRUCT_FAST_DISTINSERT
					// Optimized BFS-style neighbor search
					nearest = delaunay.nearest_vertex_in_cell3(p, c); // Was cell3 JPB WIP BUG

					views = pointcloud.pointViewsMemory.data() + offset;
					_mm_prefetch((const char*)views, _MM_HINT_T0);

					const double qx = p.x(), qy = p.y(), qz = p.z();
					const point_t& nearestPt = nearest->point();
					double bestSq = fast_sqdist2(qx, qy, qz, nearestPt.x(), nearestPt.y(), nearestPt.z());

					// MSVC: hoist a restrict-qualified base pointer for vertexMarks so the
					// optimizer can keep bestSq/qx/qy/qz in xmm registers across the inner
					// loop and avoid reloading the std::vector base each iteration. The
					// only writes inside the BFS go to *this* array and to tds_data().marker
					// (a different allocation), so __restrict is sound.
					uint32_t* __restrict pVertexMarks = vertexMarks.data();

					// The key difference from the original code is that the original determines
					// all adjacent cells and then looks at them.
					// Here, we identify the adjacent cells as needed.
					while (true) {
						++marker;
						if (marker == 0) {
							std::fill(vertexMarks.begin(), vertexMarks.end(), 0);

							// wrapped � reset all cell markers
							// NOTE: you do NOT need to reset conflict_state
							for (auto ci = delaunay.all_cells_begin(); ci != delaunay.all_cells_end(); ++ci) {
								ci->tds_data().marker = 0;
							}
							marker = 1;
						}

						vertex_handle_t best = nearest;

						cellQueue.clear();
						cell_handle_t start = nearest->cell();
						cellQueue.push(start);
						start->tds_data().marker = marker;

						size_t queueIndex = 0;
#if 1
						// inside your loop:
#if INSERTION_BFS_MAX_CELLS > 0
						size_t bfsVisited = 0;
#endif
						while (queueIndex < cellQueue.size()) {
#if INSERTION_BFS_MAX_CELLS > 0
							if (bfsVisited >= INSERTION_BFS_MAX_CELLS) {
								// Cap hit: accept current `best` as approximate nearest.
								// Falling out here behaves identically to natural exhaustion
								// of the queue with no improvement found — outer loop will
								// see best == nearest (if no improvement during this pass)
								// and break. If improvement DID happen, `goto refine_restart`
								// already fired and we wouldn't reach here.
								break;
							}
							++bfsVisited;
#endif
							const cell_handle_t c = cellQueue[queueIndex++];

							// fetch vertices once; reuse in both phases
							const vertex_handle_t v0 = c->vertex(0);
							const vertex_handle_t v1 = c->vertex(1);
							const vertex_handle_t v2 = c->vertex(2);
							const vertex_handle_t v3 = c->vertex(3);

							// Issue 4 parallel random loads of the Vertex objects (point + info
							// share a cache line in Vertex_with_info_3). Without this, the 4
							// TRY_VERTEX_ZFIRST calls below serialize on the data-dependency
							// chain  vh -> vh->info().idx -> vertexMarks[idx]. With the prefetch
							// the lines are en route while the first TRY runs, so subsequent
							// iterations see L1/L2 hits. Pure latency-hiding; bandwidth cheap.
							_mm_prefetch((const char*)&v0->point(), _MM_HINT_T0);
							_mm_prefetch((const char*)&v1->point(), _MM_HINT_T0);
							_mm_prefetch((const char*)&v2->point(), _MM_HINT_T0);
							_mm_prefetch((const char*)&v3->point(), _MM_HINT_T0);

#define TRY_VERTEX_ZFIRST(vh, rejectLabel) do {                    \
  if ((vh) != nearest && (vh) != infV) {                           \
    uint32_t* m = &pVertexMarks[(vh)->info().idx];                 \
    if (*m != marker) {                                            \
      *m = marker;                                                 \
      const point_t& pt = (vh)->point();                           \
      const double dz = pt.z() - qz;                                \
      double d2 = dz * dz;                                         \
      if (d2 >= bestSq) goto rejectLabel;                           \
      const double dx = pt.x() - qx;                                \
      d2 += dx * dx;                                               \
      if (d2 >= bestSq) goto rejectLabel;                           \
      const double dy = pt.y() - qy;                                \
      d2 += dy * dy;                                               \
      if (d2 < bestSq) { bestSq = d2; best = (vh); goto refine_restart; } \
    }                                                              \
  }                                                                \
} while (0)

							{
								TRY_VERTEX_ZFIRST(v0, reject0);
							reject0:;

								TRY_VERTEX_ZFIRST(v1, reject1);
							reject1:;

								TRY_VERTEX_ZFIRST(v2, reject2);
							reject2:;

								TRY_VERTEX_ZFIRST(v3, reject3);
							reject3:;
							}

							// === Only expand neighbors if no refinement happened ===
							if (best == nearest) {
								// reuse v0..v3 we already loaded to avoid re-calling vertex(i)
								// enqueue exactly like your code: same cells, same order
								if (v0 != nearest) {
									cell_handle_t next = c->neighbor(0);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push(next); }
								}
								if (v1 != nearest) {
									cell_handle_t next = c->neighbor(1);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push(next); }
								}
								if (v2 != nearest) {
									cell_handle_t next = c->neighbor(2);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push(next); }
								}
								if (v3 != nearest) {
									cell_handle_t next = c->neighbor(3);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push(next); }
								}
							}
							else {
								// refinement occurred, stop immediately
								break;
							}
						}
#else
						while (queueIndex < cellQueue.size()) {
							const cell_handle_t c = cellQueue[queueIndex++];

							// Inline TRY_VERTEX for each vertex
							const vertex_handle_t v0 = c->vertex(0);
							const vertex_handle_t v1 = c->vertex(1);
							const vertex_handle_t v2 = c->vertex(2);
							const vertex_handle_t v3 = c->vertex(3);

							// Prefetching 0 makes no sense, but prefetching 1-3 also may not be useful
							// since we may prematurely exit.

#define TRY_VERTEX(vh) do { \
							if ((vh) != nearest && !delaunay.is_infinite(vh)) { \
								uint32_t& mark = vertexMarks[(vh)->info().idx]; \
								if (mark != marker) { \
									mark = marker; \
									const point_t& pt = (vh)->point(); \
									const double d2 = fast_sqdist2(qx, qy, qz, pt.x(), pt.y(), pt.z()); \
									if (d2 < bestSq) { \
										bestSq = d2; \
										best = vh; \
										goto refine_restart; \
									} \
								} \
							} \
						} while (0)

							TRY_VERTEX(v0);
							TRY_VERTEX(v1);
							TRY_VERTEX(v2);
							TRY_VERTEX(v3);

							// === Only expand neighbors if no refinement happened ===
							if (best == nearest) {
								for (int i = 0; i < 4; ++i) {
									if (c->vertex(i) == nearest) continue;

									cell_handle_t next = c->neighbor(i);
									auto& nm = next->tds_data().marker;
									if (nm == marker) continue;

									nm = marker;

									cellQueue.push_back(next);  // no pop; head advances
								}
							}
							else {
								// refinement occurred, stop immediately
								break;
							}
						}
#endif

					refine_restart:
						if (best == nearest)
							break;

						nearest = best;
						pVertexMarks[nearest->info().idx] = marker;
						const point_t& nearestPtNew = nearest->point();
						bestSq = fast_sqdist2(qx, qy, qz, nearestPtNew.x(), nearestPtNew.y(), nearestPtNew.z());
					}
#else
					// === Variant B fast path: skip BFS, pick rejector from cell's 4 vertices ===
					//
					// The downstream projection check below accepts the candidate iff its
					// `nearest` projects > distInsert px in at least one of the candidate's
					// views. We choose `nearest` here so that this downstream check produces
					// the desired accept/reject decision without a BFS:
					//   - If any of the 4 cell vertices is "too close in ALL views" (a rejector),
					//     set nearest = that vertex. Downstream check will then say "not far in
					//     any view" → shouldInsert=false → no insert. Correct rejection.
					//   - Otherwise no cell vertex rejects; set nearest = nearest_vertex_in_cell3
					//     (the closest cell vertex in 3D). Downstream check will say "far in
					//     some view" by construction → shouldInsert=true → insert. Correct
					//     acceptance.
					//
					// This loses the original semantics ONLY when a true rejector exists
					// *outside* the containing cell. In dense regions (where rejections cluster)
					// the containing cell almost always already holds at least one rejector;
					// in sparse regions there's no rejector anywhere anyway.
					//
					// FIX 2: extend the rejector search to include the unique vertex of each
					// of the 4 face-neighbor cells (the vertex of `nc` opposite the face shared
					// with `c`). That vertex is by construction NOT one of the 4 vertices of
					// `c`, so we collect up to 8 distinct candidate rejectors total. This
					// covers virtually all rejectors that the BFS path would have found, at
					// bounded O(1) cost per insertion (no marker bookkeeping, no restarts).
					nearest = delaunay.nearest_vertex_in_cell3(p, c);
					views = pointcloud.pointViewsMemory.data() + offset;
					_mm_prefetch((const char*)views, _MM_HINT_T0);

					{
						const float pxFv = (float)p.x();
						const float pyFv = (float)p.y();
						const float pzFv = (float)p.z();

						// Collect up to 8 candidate rejector vertices: 4 from the containing
						// cell + the unique vertex of each face-neighbor cell. The neighbor's
						// unique vertex is `nc->vertex(nc->index(c))` — the vertex of nc
						// opposite the face shared with c, which by definition is not in c.
						vertex_handle_t candidates[8];
						int nCands = 0;
						for (int k = 0; k < 4; ++k) {
							const vertex_handle_t v = c->vertex(k);
							if (v != infV) candidates[nCands++] = v;
						}
						for (int k = 0; k < 4; ++k) {
							const cell_handle_t nc = c->neighbor(k);
							if (delaunay.is_infinite(nc)) continue;
							const vertex_handle_t vu = nc->vertex(nc->index(c));
							if (vu != infV) candidates[nCands++] = vu;
						}

						for (int vi = 0; vi < nCands; ++vi) {
							const vertex_handle_t vh = candidates[vi];
							const point_t& np = vh->point();
							const float nxF = (float)np.x();
							const float nyF = (float)np.y();
							const float nzF = (float)np.z();

							// Same per-view math as the downstream projection check.
							// "farInSomeView" mirrors the downstream `shouldInsert` for THIS
							// specific candidate-vs-vh pair.
							bool farInSomeView = false;
							for (size_t j = 0; j < numViews; ++j) {
								if (j + 1 < numViews)
									_mm_prefetch((const char*)&viewCameras[views[j + 1]][0], _MM_HINT_T0);
								const float* __restrict camera = &viewCameras[views[j]][0];

								const float pez = camera[8] * pxFv + camera[9] * pyFv + camera[10] * pzFv + camera[11];
								if (pez <= 0.f) continue;
								const float pnz = camera[8] * nxF + camera[9] * nyF + camera[10] * nzF + camera[11];
								if (pnz <= 0.f) continue;

								// FIX 1: the depth-mismatch short-circuit that used to live here
								// (|pnz-pez| >= depthThresholdFv*pez ⇒ farInSomeView=true) was NOT
								// part of the downstream projection check, so the fast path could
								// declare vh "far" while downstream would treat the same vh as a
								// rejector — causing FAST_DISTINSERT to accept points the BFS path
								// would reject. Removed so this per-vertex test now exactly mirrors
								// the downstream `shouldInsert` criterion (D-only, with the sound
								// axis bounds B/C below as early-exits since |dx|>bound ⟹ D true).

								const float zprod = pez * pnz;
								const float bound = distInsert * zprod;

								const float pex = camera[0] * pxFv + camera[1] * pyFv + camera[2] * pzFv + camera[3];
								const float pnx = camera[0] * nxF + camera[1] * nyF + camera[2] * nzF + camera[3];
								const float dx = pex * pnz - pnx * pez;
								if (FastAbsS(dx) > bound) { farInSomeView = true; break; }

								const float pey = camera[4] * pxFv + camera[5] * pyFv + camera[6] * pzFv + camera[7];
								const float pny = camera[4] * nxF + camera[5] * nyF + camera[6] * nzF + camera[7];
								const float dy = pey * pnz - pny * pez;
								if (FastAbsS(dy) > bound) { farInSomeView = true; break; }

								if (dx * dx + dy * dy > distInsertSq * zprod * zprod) {
									farInSomeView = true; break;
								}
							}

							if (!farInSomeView) {
								// vh is close to p in all observed views → rejector found.
								// Set nearest to it so downstream projection check yields
								// shouldInsert=false.
								nearest = vh;
								break;
							}
						}
					}
#endif
				}
				hint = nearest;

				//const auto& hintPt2 = hint->point();
#if !defined(_RELEASE) && !RECONSTRUCT_FAST_DISTINSERT
				// IMPORTANT: this verification runs a FULL CGAL nearest-vertex walk
				// per point — it dominates insertion cost when present. Keep it gated
				// so it only runs in debug/validation builds, not in Release.
				// Also skipped in the fast distInsert path because `hint` there is
				// intentionally a cell-corner vertex (rejector or nearest_vertex_in_cell3),
				// not the true 3D-nearest.
				ASSERT(hint == delaunay.nearest_vertex(p, hint->cell()));
#endif

				// Projection visibility check
				const float pxF = (float)p.x(), pyF = (float)p.y(), pzF = (float)p.z();
				const float nxF = (float)hint->point().x(), nyF = (float)hint->point().y(), nzF = (float)hint->point().z();

				constexpr float depthThreshold = 0.01f;

#if 0 // JPB WIP BUG doesn't improve quality.
				bool shouldInsert = false;
				for (size_t j = 0; j < numViews; ++j) {
					const float* __restrict camera = &viewCameras[views[j]][0];

					const float pez = camera[8] * pxF + camera[9] * pyF + camera[10] * pzF + camera[11];
					if (pez <= 0.f) continue;

					const float pnz = camera[8] * nxF + camera[9] * nyF + camera[10] * nzF + camera[11];
					if (pnz <= 0.f) continue;

					const float zprod = pez * pnz;

					// ---- X axis only ----
					const float pex = camera[0] * pxF + camera[1] * pyF + camera[2] * pzF + camera[3];
					const float pnx = camera[0] * nxF + camera[1] * nyF + camera[2] * nzF + camera[3];

					const float dx = pex * pnz - pnx * pez;

					// ---- Y axis only ----
					const float pey = camera[4] * pxF + camera[5] * pyF + camera[6] * pzF + camera[7];
					const float pny = camera[4] * nxF + camera[5] * nyF + camera[6] * nzF + camera[7];

					const float dy = pey * pnz - pny * pez;

					if (dx * dx + dy * dy > distInsertSq * zprod * zprod) {
						shouldInsert = true;
						break;
					}
				}
#else
				bool shouldInsert = false;
				for (size_t j = 0; j < numViews; ++j) {
					// Prefetch next view's camera matrix (random gather by view ID,
					// ~1 cache line per camera). Cheap latency hide on small loops.
					if (j + 1 < numViews)
						_mm_prefetch((const char*)&viewCameras[views[j + 1]][0], _MM_HINT_T0);
					const float* __restrict camera = &viewCameras[views[j]][0];

					const float pez = camera[8] * pxF + camera[9] * pyF + camera[10] * pzF + camera[11];
					if (pez <= 0.f) continue;

					const float pnz = camera[8] * nxF + camera[9] * nyF + camera[10] * nzF + camera[11];
					if (pnz <= 0.f) continue;

					if (FastAbsS(pnz - pez) >= depthThreshold * pez) {
						shouldInsert = true;
						break;
					}

					const float zprod = pez * pnz;
					const float bound = distInsert * zprod;

					// ---- X axis only ----
					const float pex = camera[0] * pxF + camera[1] * pyF + camera[2] * pzF + camera[3];
					const float pnx = camera[0] * nxF + camera[1] * nyF + camera[2] * nzF + camera[3];

					const float dx = pex * pnz - pnx * pez;
					if (FastAbsS(dx) > bound) {
						shouldInsert = true;
						break;
					}

					// ---- Y axis only (only if X passed) ----
					const float pey = camera[4] * pxF + camera[5] * pyF + camera[6] * pzF + camera[7];
					const float pny = camera[4] * nxF + camera[5] * nyF + camera[6] * nzF + camera[7];

					const float dy = pey * pnz - pny * pez;
					if (FastAbsS(dy) > bound) {
						shouldInsert = true;
						break;
					}

					// ---- Exact check only if both axes passed ----
					if (dx * dx + dy * dy > distInsertSq * zprod * zprod) {
						shouldInsert = true;
						break;
					}
				}
#endif

				if (shouldInsert) {
					hint = delaunay.insert(p, lt, c, li, lj);
					ASSERT(anchor != vertex_handle_t());
				}

				// Visibility information not needed for the dt, but used in the next step.
				// idx is the index of the spatially sorted point.
#if !DIRECT_VIEW_EXPANSION
				InsertViews(hint->info().idx, pointcloud, i);
#else
				pointToVertex[i] = hint->info().idx;
#endif
			advance2:
				if (!(i & 4095)) progress += 4096;
			}
		}

		progress.process();
		progress.close();

		_aligned_free(mask);
		mask = 0;
		_aligned_free(vertices);
		vertices = 0;

#if DIRECT_VIEW_EXPANSION
		// --- Expansion pass: convert pointToVertex[] into flat per-vertex ViewCount arrays ---
		// This replaces the per-vertex allViews random pushes with a single sequential scan.
		// Phase 3 is parallelized: each vertex is independent, each thread gets its own countByImageID.
		{
			TD_TIMER_STARTD();
			const uint32_t numVtxIDs = vert_info_t::g_idx; // total vertex IDs assigned

			// Phase 1: Count how many points map to each vertex AND accumulate
			// per-vertex view-bound (sum of sizes of its points = upper bound on unique views).
			uint32_t* __restrict vtxPointCount = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numVtxIDs, 64);
			uint32_t* __restrict vtxViewBound = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numVtxIDs, 64);
			memset(vtxPointCount, 0, sizeof(uint32_t) * numVtxIDs);
			memset(vtxViewBound, 0, sizeof(uint32_t) * numVtxIDs);
			{
				const uint32_t* __restrict pTV = pointToVertex;
				const uint32_t* __restrict sz = sizes;
				for (uint32_t i = 0; i < numVertices; ++i) {
					const uint32_t v = pTV[i];
					if (v != UINT32_MAX) {
						++vtxPointCount[v];
						vtxViewBound[v] += sz[i];
					}
				}
			}

			// Phase 2: Build CSR for point-to-vertex gather + compute per-vertex
			// upper-bound offsets into vcData (so each vertex has a non-overlapping write region).
			uint32_t* __restrict vtxPointOffsets = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * (numVtxIDs + 1), 64);
			vcOffsets = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numVtxIDs, 64);
			vcSizes = (uint16_t*)_aligned_malloc(sizeof(uint16_t) * numVtxIDs, 64);

			vtxPointOffsets[0] = 0;
			vcOffsets[0] = 0;
			for (uint32_t v = 0; v < numVtxIDs; ++v) {
				vtxPointOffsets[v + 1] = vtxPointOffsets[v] + vtxPointCount[v];
				if (v + 1 < numVtxIDs)
					vcOffsets[v + 1] = vcOffsets[v] + vtxViewBound[v];
			}
			const uint64_t totalViewsBound = (uint64_t)vcOffsets[numVtxIDs - 1] + vtxViewBound[numVtxIDs - 1];

			const uint32_t totalMapped = vtxPointOffsets[numVtxIDs];
			uint32_t* __restrict vtxPointList = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * (totalMapped + 1), 64);

			// Scatter points into per-vertex buckets (reuse vtxPointCount as write cursors)
			memset(vtxPointCount, 0, sizeof(uint32_t) * numVtxIDs);
			{
				const uint32_t* __restrict pTV = pointToVertex;
				for (uint32_t i = 0; i < numVertices; ++i) {
					const uint32_t v = pTV[i];
					if (v != UINT32_MAX) {
						vtxPointList[vtxPointOffsets[v] + vtxPointCount[v]] = i;
						++vtxPointCount[v];
					}
				}
			}

			_aligned_free(vtxPointCount);
			_aligned_free(vtxViewBound);

			// Phase 3: Expand views per vertex with deduplication — PARALLEL.
			// Each vertex writes to vcData[vcOffsets[v]..vcOffsets[v]+vcSizes[v]).
			// Per-thread countByImageID for O(1) dedup.
			// Static scheduling: vertices are roughly uniform work, and static gives
			// each thread a contiguous block → sequential reads through vtxPointOffsets,
			// vtxPointList, and vcData regions. Dynamic would randomize access patterns.
			vcData = (ExpandedViewCount*)_aligned_malloc(sizeof(ExpandedViewCount) * (totalViewsBound + 1), 64);
			const uint32_t numImages = (uint32_t)images.size();

			// Capture restrict-qualified pointers for the parallel region.
			const uint32_t* __restrict pOffsets = offsets;
			const uint32_t* __restrict pSizes = sizes;
			const uint32_t* __restrict pViewsMem = pointcloud.pointViewsMemory.data();
			const uint32_t* __restrict pVtxPtList = vtxPointList;
			const uint32_t* __restrict pVtxPtOffsets = vtxPointOffsets;
			ExpandedViewCount* __restrict pVcData = vcData;
			uint32_t* __restrict pVcOffsets = vcOffsets;
			uint16_t* __restrict pVcSizes = vcSizes;

#pragma omp parallel
			{
				uint16_t* __restrict myCountByImageID = (uint16_t*)_aligned_malloc(sizeof(uint16_t) * numImages, 64);
				memset(myCountByImageID, 0, sizeof(uint16_t) * numImages);

#pragma omp for schedule(static)
				for (int64_t v = 0; v < (int64_t)numVtxIDs; ++v) {
					const uint32_t pBegin = pVtxPtOffsets[v];
					const uint32_t pEnd = pVtxPtOffsets[v + 1];
					if (pBegin == pEnd) {
						pVcSizes[v] = 0;
						continue;
					}

					ExpandedViewCount* __restrict dst = &pVcData[pVcOffsets[v]];
					uint16_t numViews = 0;

					for (uint32_t pi = pBegin; pi < pEnd; ++pi) {
						const uint32_t ptIdx = pVtxPtList[pi];
						// Prefetch next point's view data while processing current
						if (pi + 1 < pEnd) {
							const uint32_t nextPt = pVtxPtList[pi + 1];
							_mm_prefetch((const char*)&pViewsMem[pOffsets[nextPt]], _MM_HINT_T0);
						}
						const uint32_t* __restrict src = &pViewsMem[pOffsets[ptIdx]];
						const uint32_t cnt = pSizes[ptIdx];
						for (uint32_t k = 0; k < cnt; ++k) {
							const uint32_t id = src[k];
							uint16_t& slot = myCountByImageID[id];
							if (slot != 0) {
								++dst[slot - 1].count;
							} else {
								dst[numViews] = { id, 1 };
								slot = static_cast<uint16_t>(numViews + 1);
								++numViews;
							}
						}
					}

					// Reset touched slots
					for (uint16_t j = 0; j < numViews; ++j)
						myCountByImageID[dst[j].id] = 0;

					pVcSizes[v] = numViews;
				}

				_aligned_free(myCountByImageID);
			} // omp parallel

			// --- Compaction pass: pack vcData tightly so the weighting loop streams sequentially ---
			// The parallel phase wrote each vertex's data at upper-bound offsets (with gaps).
			// This linear sweep moves entries to contiguous positions. Cost: ~memcpy of actual data.
			{
				uint32_t writePos = 0;
				for (uint32_t v = 0; v < numVtxIDs; ++v) {
					const uint16_t sz = pVcSizes[v];
					if (sz == 0) {
						pVcOffsets[v] = writePos;
						continue;
					}
					const uint32_t oldOff = pVcOffsets[v];
					if (oldOff != writePos)
						// Safe: writePos <= oldOff always (accumulated actual sizes
						// <= accumulated upper-bound offsets), so dst is at a lower
						// or equal address than src.  memcpy avoids memmove's
						// per-call overlap-direction check.
						memcpy(&pVcData[writePos], &pVcData[oldOff], sizeof(ExpandedViewCount) * sz);
					pVcOffsets[v] = writePos;
					writePos += sz;
				}
				// writePos is now the actual total — much smaller than totalViewsBound
			}

			_aligned_free(vtxPointList);
			_aligned_free(vtxPointOffsets);
			_aligned_free(pointToVertex);
			pointToVertex = 0;

			// Free offsets/sizes now — they're no longer needed
			_aligned_free(offsets);
			offsets = 0;
			_aligned_free(sizes);
			sizes = 0;

			MESH_DIAG("View expansion pass completed: %u vertices, %llu view bound (%s)",
				numVtxIDs, (unsigned long long)totalViewsBound, TD_TIMER_GET_FMT().c_str());
		}
#endif

		// JPB WIP BUG decltype(cellQueue)().swap(cellQueue);
		decltype(viewCameras)().swap(viewCameras);

		numDelaunayVertices = delaunay.number_of_vertices(); // Number of finite vertices, has one more.
		std::cerr << "Verts : " << numDelaunayVertices << "\n";
		const size_t numNodes(delaunay.number_of_cells());
		const size_t numCells = numNodes; // cheaper than all_cells.size() if available
		// AFTER: Enumerate cells and build hull facets without storing all iterators.
		// Also build a flat arrays for all cells and a separate one for finite cells for the median pass.
		cell_size_t ciID(0);

		size_t maxCells = delaunay.number_of_cells();
		cell_handle_t* __restrict finiteCells = (cell_handle_t*)_aligned_malloc(sizeof(cell_handle_t) * maxCells, 64);
		allCells = (cell_handle_t*)_aligned_malloc(sizeof(cell_handle_t) * maxCells, 64);

		// Per-cell side cache populated during the cell enumeration that is
		// already required to assign IDs (Morton or baseline). Both passes
		// already cold-deref every cell + its 4 vertex_t records to compute
		// the centroid / isFinite — capturing idx[] and the pointer-ordering
		// edge mask here costs almost nothing because the cachelines are hot.
		// The median pass downstream then runs as pure SoA streaming with zero
		// CGAL pointer derefs (eliminates ~5-7 s on 286M-cell scenes).
		// Mask uses vertex_handle_t pointer compare to match the original
		// median semantics bit-for-bit.
		struct CellMeta {
			uint32_t idx[4]; // 16 B  vertex idx of vertex(0..3)
			uint8_t  mask;   //  1 B  edges where vh_lo < vh_hi (pointer compare)
			uint8_t  _pad[3];
		};
		static_assert(sizeof(CellMeta) == 20, "unexpected CellMeta layout");
		CellMeta* __restrict finiteCellMeta = (CellMeta*)_aligned_malloc(sizeof(CellMeta) * maxCells, 64);

		size_t numFiniteCells = 0;
		size_t numInfiniteCells = 0;

		// Toggle: spatial reordering of cell IDs by Morton key on cell centroids.
		// Set to 0 to restore CGAL creation-order assignment for benchmarking.
		#ifndef RECONSTRUCT_OPT_MORTON_CELLS
		#define RECONSTRUCT_OPT_MORTON_CELLS 1
		#endif

#if RECONSTRUCT_OPT_MORTON_CELLS
		// Spatial reordering of cell IDs by Morton (Z-order) key on cell centroids.
		// CGAL iterates cells in creation/refinement order, which is only weakly
		// spatial. IBFS does graph-local BFS walks; placing graph-adjacent cells
		// at adjacent IDs ⇒ memory-adjacent ⇒ massively better L1/L2/TLB hit
		// rates during BuildGraphNodesAndEdges and especially during max-flow.
		// Pure permutation; no edge weights, no source/sink caps, no cut result
		// changes — byte-identical mesh out, just faster.
		{
			cell_handle_t* __restrict tmpCells = (cell_handle_t*)_aligned_malloc(sizeof(cell_handle_t) * maxCells, 64);
			float* __restrict cx = (float*)_aligned_malloc(sizeof(float) * maxCells, 64);
			float* __restrict cy = (float*)_aligned_malloc(sizeof(float) * maxCells, 64);
			float* __restrict cz = (float*)_aligned_malloc(sizeof(float) * maxCells, 64);
			// Per-cell metadata in original (tmpCells) order. Filled in Pass 2
			// alongside the centroid loads — same cold cell+vertex_t cachelines.
			// Permuted into finiteCellMeta during Pass 5 compaction. Freed at
			// end of the Morton block. Memory: ~5.7 GB for 286M cells.
			CellMeta* __restrict origMeta = (CellMeta*)_aligned_malloc(sizeof(CellMeta) * maxCells, 64);

			// Pass 1 (serial, fast): collect cell handles via CGAL iterator.
			// CGAL's All_cells_iterator is a forward iterator over an internal
			// linked list, so this can't be parallelized without a copy step.
			size_t kk = 0;
			for (delaunay_t::All_cells_iterator ci = delaunay.all_cells_begin(), eci = delaunay.all_cells_end(); ci != eci; ++ci, ++kk) {
				tmpCells[kk] = ci;
			}
			const size_t numAll = kk;

			// Pass 2 (parallel): centroid + per-thread bbox reduction +
			// per-cell metadata cache (vertex idx[4] + pointer-order edge mask).
			float bbMinX =  std::numeric_limits<float>::max();
			float bbMinY =  std::numeric_limits<float>::max();
			float bbMinZ =  std::numeric_limits<float>::max();
			float bbMaxX = -std::numeric_limits<float>::max();
			float bbMaxY = -std::numeric_limits<float>::max();
			float bbMaxZ = -std::numeric_limits<float>::max();
#pragma omp parallel
			{
				float lMinX =  std::numeric_limits<float>::max();
				float lMinY =  std::numeric_limits<float>::max();
				float lMinZ =  std::numeric_limits<float>::max();
				float lMaxX = -std::numeric_limits<float>::max();
				float lMaxY = -std::numeric_limits<float>::max();
				float lMaxZ = -std::numeric_limits<float>::max();
#pragma omp for schedule(static) nowait
				for (ptrdiff_t i = 0; i < (ptrdiff_t)numAll; ++i) {
					const cell_handle_t ci = tmpCells[i];
					const auto vh0 = ci->vertex(0);
					const auto vh1 = ci->vertex(1);
					const auto vh2 = ci->vertex(2);
					const auto vh3 = ci->vertex(3);
					const bool fin = (vh0 != infV) & (vh1 != infV) & (vh2 != infV) & (vh3 != infV);

					float sx = 0.f, sy = 0.f, sz = 0.f;
					if (fin) {
						// Single point load each — used for centroid AND idx[].
						const auto& p0 = vh0->point();
						const auto& p1 = vh1->point();
						const auto& p2 = vh2->point();
						const auto& p3 = vh3->point();
						sx = float(p0.x()) + float(p1.x()) + float(p2.x()) + float(p3.x());
						sy = float(p0.y()) + float(p1.y()) + float(p2.y()) + float(p3.y());
						sz = float(p0.z()) + float(p1.z()) + float(p2.z()) + float(p3.z());

						// Pointer-order edge mask — bit-identical to legacy median.
						uint8_t m = 0;
						m |= (vh0 < vh1) ? 0x01 : 0;
						m |= (vh0 < vh2) ? 0x02 : 0;
						m |= (vh0 < vh3) ? 0x04 : 0;
						m |= (vh1 < vh2) ? 0x08 : 0;
						m |= (vh1 < vh3) ? 0x10 : 0;
						m |= (vh2 < vh3) ? 0x20 : 0;

						CellMeta& cm = origMeta[i];
						cm.idx[0] = vh0->info().idx;
						cm.idx[1] = vh1->info().idx;
						cm.idx[2] = vh2->info().idx;
						cm.idx[3] = vh3->info().idx;
						cm.mask = m;
						const float inv = 0.25f;
						sx *= inv; sy *= inv; sz *= inv;
					} else {
						// Infinite cell: centroid over finite vertices only,
						// matching legacy behavior. Meta unused downstream.
						int n = 0;
						if (vh0 != infV) { const auto& p = vh0->point(); sx += float(p.x()); sy += float(p.y()); sz += float(p.z()); ++n; }
						if (vh1 != infV) { const auto& p = vh1->point(); sx += float(p.x()); sy += float(p.y()); sz += float(p.z()); ++n; }
						if (vh2 != infV) { const auto& p = vh2->point(); sx += float(p.x()); sy += float(p.y()); sz += float(p.z()); ++n; }
						if (vh3 != infV) { const auto& p = vh3->point(); sx += float(p.x()); sy += float(p.y()); sz += float(p.z()); ++n; }
						const float inv = 1.0f / float(n);
						sx *= inv; sy *= inv; sz *= inv;
						origMeta[i].mask = 0; // sentinel; not consumed for infinite cells
					}
					cx[i] = sx; cy[i] = sy; cz[i] = sz;
					if (sx < lMinX) lMinX = sx; if (sy < lMinY) lMinY = sy; if (sz < lMinZ) lMinZ = sz;
					if (sx > lMaxX) lMaxX = sx; if (sy > lMaxY) lMaxY = sy; if (sz > lMaxZ) lMaxZ = sz;
				}
#pragma omp critical
				{
					if (lMinX < bbMinX) bbMinX = lMinX; if (lMinY < bbMinY) bbMinY = lMinY; if (lMinZ < bbMinZ) bbMinZ = lMinZ;
					if (lMaxX > bbMaxX) bbMaxX = lMaxX; if (lMaxY > bbMaxY) bbMaxY = lMaxY; if (lMaxZ > bbMaxZ) bbMaxZ = lMaxZ;
				}
			}

			const float spanX = std::max(bbMaxX - bbMinX, 1e-6f);
			const float spanY = std::max(bbMaxY - bbMinY, 1e-6f);
			const float spanZ = std::max(bbMaxZ - bbMinZ, 1e-6f);
			const float scaleX = float((1u << 21) - 1) / spanX;
			const float scaleY = float((1u << 21) - 1) / spanY;
			const float scaleZ = float((1u << 21) - 1) / spanZ;

			// 21-bit-per-axis Morton encoder (3*21=63 bits → fits uint64_t).
			auto splitBy3 = [](uint32_t a) -> uint64_t {
				uint64_t v = a & 0x1FFFFFu;
				v = (v | (v << 32)) & 0x1F00000000FFFFull;
				v = (v | (v << 16)) & 0x1F0000FF0000FFull;
				v = (v | (v <<  8)) & 0x100F00F00F00F00Full;
				v = (v | (v <<  4)) & 0x10C30C30C30C30C3ull;
				v = (v | (v <<  2)) & 0x1249249249249249ull;
				return v;
			};

			uint64_t* __restrict keys = (uint64_t*)_aligned_malloc(sizeof(uint64_t) * numAll, 64);
			uint32_t* __restrict perm = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numAll, 64);

			// Pass 3 (parallel): Morton key + permutation init.
#pragma omp parallel for schedule(static)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)numAll; ++i) {
				const uint32_t qx = (uint32_t)((cx[i] - bbMinX) * scaleX);
				const uint32_t qy = (uint32_t)((cy[i] - bbMinY) * scaleY);
				const uint32_t qz = (uint32_t)((cz[i] - bbMinZ) * scaleZ);
				keys[i] = splitBy3(qx) | (splitBy3(qy) << 1) | (splitBy3(qz) << 2);
				perm[i] = (uint32_t)i;
			}

#if RECONSTRUCT_RADIX_MORTON
			// Parallel 8-bit LSD radix sort on perm[], keyed indirectly by keys[].
			// 8 passes × (count + prefix-sum + scatter). Stable across equal keys.
			{
				constexpr int kRBits = 8;
				constexpr int kRBkts = 1 << kRBits;
				constexpr uint64_t kRMask = kRBkts - 1;
				const int nTR = omp_get_max_threads();
				uint32_t* __restrict permTmp = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * numAll, 64);
				size_t* __restrict tHist = (size_t*)_aligned_malloc(sizeof(size_t) * (size_t)nTR * kRBkts, 64);
				uint32_t* __restrict src = perm;
				uint32_t* __restrict dst = permTmp;
				for (int shift = 0; shift < 64; shift += kRBits) {
					// Phase 1: per-thread histograms.  schedule(static) gives each
					// thread a deterministic contiguous chunk reused by Phase 3.
					std::memset(tHist, 0, sizeof(size_t) * (size_t)nTR * kRBkts);
					#pragma omp parallel
					{
						const int tid = omp_get_thread_num();
						size_t* __restrict h = tHist + (size_t)tid * kRBkts;
						#pragma omp for schedule(static) nowait
						for (ptrdiff_t i = 0; i < (ptrdiff_t)numAll; ++i) {
							const uint32_t b = (uint32_t)((keys[src[i]] >> shift) & kRMask);
							++h[b];
						}
					}
					// Phase 2: column-major exclusive prefix sum.  Converts each
					// tHist[tid][b] into the starting write offset for thread tid's
					// bucket b.  Order: bucket-major, thread-minor — stable.
					size_t running = 0;
					for (int b = 0; b < kRBkts; ++b) {
						for (int t = 0; t < nTR; ++t) {
							size_t* slot = tHist + (size_t)t * kRBkts + b;
							const size_t c = *slot;
							*slot = running;
							running += c;
						}
					}
					// Phase 3: scatter src -> dst using each thread's write cursors.
					#pragma omp parallel
					{
						const int tid = omp_get_thread_num();
						size_t* __restrict h = tHist + (size_t)tid * kRBkts;
						#pragma omp for schedule(static) nowait
						for (ptrdiff_t i = 0; i < (ptrdiff_t)numAll; ++i) {
							const uint32_t v = src[i];
							const uint32_t b = (uint32_t)((keys[v] >> shift) & kRMask);
							dst[h[b]++] = v;
						}
					}
					std::swap(src, dst);
				}
				// After an even number of passes (8), src == perm again.  Guard
				// the copy anyway for safety against future pass-count changes.
				if (src != perm)
					std::memcpy(perm, src, sizeof(uint32_t) * numAll);
				_aligned_free(permTmp);
				_aligned_free(tHist);
			}
#else
			// Parallel sort if available (MSVC <execution> / libstdc++ par).
#if defined(_MSC_VER) && _MSVC_LANG >= 201703L
			std::sort(std::execution::par_unseq, perm, perm + numAll,
				[keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
#else
			std::sort(perm, perm + numAll,
				[keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
#endif
#endif

			// Pass 4 (parallel): assign info() / allCells; cache isFinite.
			// Drops the 4-vertex deref for isFinite — origMeta already has it
			// implicitly (mask==0 only for infinite cells; we use a separate
			// isFinite[] flag to keep Pass 5 a clean linear scan).
			// ci->info() and allCells[ciID] are per-cell distinct memory locations,
			// so parallel writes are race-free.
			uint8_t* __restrict isFinite = (uint8_t*)_aligned_malloc(numAll, 64);
#pragma omp parallel for schedule(static)
			for (ptrdiff_t k = 0; k < (ptrdiff_t)numAll; ++k) {
				const uint32_t origIdx = perm[k];
				const cell_handle_t ci = tmpCells[origIdx];
				const cell_size_t id = (cell_size_t)k;
				ci->info() = id;
				allCells[id] = ci;
				// Vertex-handle compare to infV — these pointers live in the
				// cell record we just touched for the info() write, so the
				// loads are essentially free (same cacheline).
				const bool fin = (ci->vertex(0) != infV) & (ci->vertex(1) != infV)
					& (ci->vertex(2) != infV) & (ci->vertex(3) != infV);
				isFinite[k] = fin ? 1u : 0u;
			}

#if RECONSTRUCT_PARALLEL_PASS5
			// Pass 5 (parallel compaction): two-pass count + prefix-sum + scatter.
			// Each thread is assigned a contiguous chunk of [0, numAll) via
			// schedule(static) (matching chunking across the count and scatter
			// passes), writes its finite cells into the global output starting
			// at a pre-computed offset. Output order is bit-identical to the
			// serial version. hullFacets gathered in a tiny sequential tail pass.
			{
				int numThreads = 1;
				size_t* blockCount = nullptr;
				size_t* blockOffset = nullptr;

#pragma omp parallel
				{
#pragma omp single
					{
						numThreads = omp_get_num_threads();
						blockCount = (size_t*)_aligned_malloc(sizeof(size_t) * numThreads, 64);
						blockOffset = (size_t*)_aligned_malloc(sizeof(size_t) * (numThreads + 1), 64);
						for (int t = 0; t < numThreads; ++t) blockCount[t] = 0;
					}
					// implicit barrier after single

					const int tid = omp_get_thread_num();

					// Pass 5a: count finite cells per chunk.
					size_t cnt = 0;
#pragma omp for schedule(static) nowait
					for (ptrdiff_t k = 0; k < (ptrdiff_t)numAll; ++k) {
						if (isFinite[k]) ++cnt;
					}
					blockCount[tid] = cnt;

#pragma omp barrier

					// Pass 5b: exclusive prefix-sum (serial, tiny — ~32 entries).
#pragma omp single
					{
						blockOffset[0] = 0;
						for (int t = 0; t < numThreads; ++t)
							blockOffset[t + 1] = blockOffset[t] + blockCount[t];
					}
					// implicit barrier after single

					// Pass 5c: parallel scatter into pre-computed offsets.
					// Each thread re-scans its own chunk (schedule(static) gives
					// identical iteration ranges as the count pass) and writes
					// sequentially starting at blockOffset[tid].
					size_t out = blockOffset[tid];
#pragma omp for schedule(static) nowait
					for (ptrdiff_t k = 0; k < (ptrdiff_t)numAll; ++k) {
						if (isFinite[k]) {
							finiteCellMeta[out] = origMeta[perm[k]];
							finiteCells[out] = allCells[k];
							++out;
						}
					}
				} // omp parallel

				numFiniteCells = blockOffset[numThreads];

				// Pass 5d (serial tail): hullFacets for infinite cells. There
				// are typically only a few hundred of these out of tens of
				// millions, so the cost of the linear scan is negligible and
				// keeping it serial preserves hullFacets ordering exactly.
				for (size_t k = 0; k < numAll; ++k) {
					if (!isFinite[k]) {
						const cell_handle_t ci = allCells[k];
						++numInfiniteCells;
						hullFacets.emplace_back(ci, ci->index(infV));
					}
				}

				_aligned_free(blockOffset);
				_aligned_free(blockCount);
			}
#else
			// Pass 5 (serial compaction): finiteCells + finiteCellMeta + hullFacets.
			// Reads origMeta[perm[k]] when finite — random access pattern but
			// only ~5 GB total over the finite cells, bandwidth-friendly.
			for (size_t k = 0; k < numAll; ++k) {
				const cell_handle_t ci = allCells[k];
				if (isFinite[k]) {
					finiteCellMeta[numFiniteCells] = origMeta[perm[k]];
					finiteCells[numFiniteCells++] = ci;
				} else {
					++numInfiniteCells;
					hullFacets.emplace_back(ci, ci->index(infV));
				}
			}
#endif
			ciID = (cell_size_t)numAll;

			_aligned_free(isFinite);
			_aligned_free(perm);
			_aligned_free(keys);
			_aligned_free(cz);
			_aligned_free(cy);
			_aligned_free(cx);
			_aligned_free(origMeta);
			_aligned_free(tmpCells);
		}
#else
		// Original creation-order assignment (baseline for A/B test).
		for (delaunay_t::All_cells_iterator ci = delaunay.all_cells_begin(), eci = delaunay.all_cells_end(); ci != eci; ++ci, ++ciID) {
			ci->info() = ciID;
			allCells[ciID] = ci;

			const auto vh0 = ci->vertex(0);
			const auto vh1 = ci->vertex(1);
			const auto vh2 = ci->vertex(2);
			const auto vh3 = ci->vertex(3);
			if (vh0 != infV && vh1 != infV && vh2 != infV && vh3 != infV) {
				CellMeta& cm = finiteCellMeta[numFiniteCells];
				cm.idx[0] = vh0->info().idx;
				cm.idx[1] = vh1->info().idx;
				cm.idx[2] = vh2->info().idx;
				cm.idx[3] = vh3->info().idx;
				uint8_t m = 0;
				m |= (vh0 < vh1) ? 0x01 : 0;
				m |= (vh0 < vh2) ? 0x02 : 0;
				m |= (vh0 < vh3) ? 0x04 : 0;
				m |= (vh1 < vh2) ? 0x08 : 0;
				m |= (vh1 < vh3) ? 0x10 : 0;
				m |= (vh2 < vh3) ? 0x20 : 0;
				cm.mask = m;
				finiteCells[numFiniteCells++] = ci;
			} else {
				++numInfiniteCells;
				hullFacets.emplace_back(ci, ci->index(delaunay.infinite_vertex()));
			}
		}
#endif
		totalCells = ciID;

		// Build per-cell neighbor ID + reverse-slot caches once, here, where
		// allCells[] and ci->info() are fully populated. Identical to the
		// code that previously lived in BuildGraphNodesAndEdges' P2b — just
		// relocated so the data is available during ray-walk weighting too.
		// Cell pool is touched once (cold) per cell here; later phases read
		// only the flat arrays.
		cellNbrID   = (uint32_t*)_aligned_malloc(sizeof(uint32_t) * totalCells * 4, 64);
		cellNbrSlot = (uint8_t*) _aligned_malloc(sizeof(uint8_t)  * totalCells,     64);
#pragma omp parallel for schedule(static)
		for (ptrdiff_t idx = 0; idx < (ptrdiff_t)totalCells; ++idx) {
			uint32_t* __restrict dstNb = cellNbrID + idx * 4;
			const cell_handle_t  ci    = allCells[idx];
			uint8_t slotPack = 0;
			for (int i = 0; i < 4; ++i) {
				const cell_handle_t cj   = ci->neighbor(i);
				const cell_size_t   cjID = cj->info();
				dstNb[i] = (uint32_t)cjID;
				slotPack |= (uint8_t)((cj->index(ci) & 3u) << (i * 2));
			}
			cellNbrSlot[idx] = slotPack;
		}

		auto t0 = rdtscEnd();

#if 1 // New idea for median calculation
		const auto maxIndex = vert_info_t::g_idx;
		idToPoint = (Point3f*)_aligned_malloc(sizeof(Point3f) * (maxIndex + 1), 64);
		for (auto vit = delaunay.finite_vertices_begin(), end = delaunay.finite_vertices_end(); vit != end; ++vit) {
			const auto& p = vit->point();
			const uint32_t id = vit->info().idx;
			idToPoint[id].x = float(p.x());
			idToPoint[id].y = float(p.y());
			idToPoint[id].z = float(p.z());
		}

		// Per-cell metadata is already in finiteCellMeta — populated during
		// the cell-enumeration pass that had to cold-deref every cell anyway.
		// Median work is now pure SoA streaming: one popcount pass for offsets,
		// one distance pass over finiteCellMeta + idToPoint. No CGAL pointer
		// derefs at all.
		const int nMedianThreads = omp_get_max_threads();
		std::vector<size_t> threadEdgeCounts(nMedianThreads, 0);

		// Pass 1: count edges per thread (popcount over cached masks only).
#pragma omp parallel
		{
			const int tid = omp_get_thread_num();
			size_t localCount = 0;

#pragma omp for schedule(static)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)numFiniteCells; ++i) {
				localCount += __popcnt(finiteCellMeta[i].mask);
			}
			threadEdgeCounts[tid] = localCount;
		}

		// Prefix-sum to get per-thread write offsets.
		std::vector<size_t> threadOffsets(nMedianThreads + 1, 0);
		for (int t = 0; t < nMedianThreads; ++t)
			threadOffsets[t + 1] = threadOffsets[t] + threadEdgeCounts[t];
		const size_t totalEdges = threadOffsets[nMedianThreads];

		float* __restrict dists = (float*)_aligned_malloc(sizeof(float) * totalEdges, 64);

		// Pass 2: read finiteCellMeta[i] + idToPoint[]. No cell-handle or vertex_t
		// loads. Same write order per cell as before -> identical dists[].
#pragma omp parallel
		{
			const int tid = omp_get_thread_num();
			float* __restrict dst = dists + threadOffsets[tid];

#pragma omp for schedule(static)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)numFiniteCells; ++i) {
				const CellMeta& cm = finiteCellMeta[i];
				const Point3f& p0 = idToPoint[cm.idx[0]];
				const Point3f& p1 = idToPoint[cm.idx[1]];
				const Point3f& p2 = idToPoint[cm.idx[2]];
				const Point3f& p3 = idToPoint[cm.idx[3]];
				const uint8_t  m = cm.mask;

#define MEDIAN_EDGE_DIST_IF(bit, pa, pb) do {                          \
				if (m & (bit)) {                                   \
					const float dx = pa.x - pb.x;                  \
					const float dy = pa.y - pb.y;                  \
					const float dz = pa.z - pb.z;                  \
					*dst++ = dx*dx + dy*dy + dz*dz;                \
				}                                                  \
			} while(0)

				MEDIAN_EDGE_DIST_IF(0x01, p0, p1);
				MEDIAN_EDGE_DIST_IF(0x02, p0, p2);
				MEDIAN_EDGE_DIST_IF(0x04, p0, p3);
				MEDIAN_EDGE_DIST_IF(0x08, p1, p2);
				MEDIAN_EDGE_DIST_IF(0x10, p1, p3);
				MEDIAN_EDGE_DIST_IF(0x20, p2, p3);
#undef MEDIAN_EDGE_DIST_IF
			}
		}

		_aligned_free(finiteCellMeta);
		_aligned_free(finiteCells);
		finiteCells = 0;

		std::nth_element(dists, dists + totalEdges / 2, dists + totalEdges);
		approxMedian = dists[totalEdges / 2];

		_aligned_free(dists);
		dists = 0;
#else
		const int numThreads = omp_get_max_threads();
		std::vector<std::vector<float>> threadDists(numThreads);

#ifdef VALIDATE
		int oldThreadCount = omp_get_max_threads();
		omp_set_num_threads(1);
#endif

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			auto& local = threadDists[tid];
			local.reserve(6 * finiteCells.size() / nMaxThreads);

#pragma omp for schedule(static)
			for (ptrdiff_t i = 0; i < (ptrdiff_t)finiteCells.size(); ++i) {
				const cell_handle_t ci = finiteCells[i];

				const auto v0 = ci->vertex(0);
				const auto v1 = ci->vertex(1);
				const auto v2 = ci->vertex(2);
				const auto v3 = ci->vertex(3);

				const point_t& __restrict p0 = v0->point();
				const point_t& __restrict p1 = v1->point();
				const point_t& __restrict p2 = v2->point();
				const point_t& __restrict p3 = v3->point();

				if (v0 < v1) { float dx = p0.x() - p1.x(), dy = p0.y() - p1.y(), dz = p0.z() - p1.z(); local.push_back(dx * dx + dy * dy + dz * dz); }
				if (v0 < v2) { float dx = p0.x() - p2.x(), dy = p0.y() - p2.y(), dz = p0.z() - p2.z(); local.push_back(dx * dx + dy * dy + dz * dz); }
				if (v0 < v3) { float dx = p0.x() - p3.x(), dy = p0.y() - p3.y(), dz = p0.z() - p3.z(); local.push_back(dx * dx + dy * dy + dz * dz); }
				if (v1 < v2) { float dx = p1.x() - p2.x(), dy = p1.y() - p2.y(), dz = p1.z() - p2.z(); local.push_back(dx * dx + dy * dy + dz * dz); }
				if (v1 < v3) { float dx = p1.x() - p3.x(), dy = p1.y() - p3.y(), dz = p1.z() - p3.z(); local.push_back(dx * dx + dy * dy + dz * dz); }
				if (v2 < v3) { float dx = p2.x() - p3.x(), dy = p2.y() - p3.y(), dz = p2.z() - p3.z(); local.push_back(dx * dx + dy * dy + dz * dz); }
			}
		}

		size_t totalSize = 0;
		for (const auto& vec : threadDists)
			totalSize += vec.size();

		distsSq.reset(new float[totalSize]);
		float* out = distsSq.get();

		for (auto& vec : threadDists) {
			std::memcpy(out, vec.data(), vec.size() * sizeof(float));
			out += vec.size();
		}

#ifdef VALIDATE
		omp_set_num_threads(oldThreadCount);
#endif

		// Compute median approximately.
		// For odd length data this is exact.  Even length is potentially very wrong, but we are
		// going with the idea that this is a large piece of irregular data where a little
		// error is tolerable.  Here we technically want the average of the two middle elements,
		// but we are just using the first of these elements.
		std::nth_element(distsSq.get(), distsSq.get() + totalSize / 2, distsSq.get() + totalSize);
		approxMedian = distsSq[totalSize / 2];
#endif

		auto t1 = rdtscEnd();

		MESH_DIAG("Median time %g", rdtscToSeconds(t1 - t0, cpuHz));

		// Prefer memset as cell_info_t will value initialize multiple fields.
		infoCells = (cell_info_t*)VirtualAlloc(
			nullptr,
			sizeof(cell_info_t) * totalCells,
			MEM_RESERVE | MEM_COMMIT,
			PAGE_READWRITE
		);

		// find all cells containing a camera
		camCells.resize(images.GetSize());
		FOREACH(i, images)
		{
			const Image& imageData = images[i];
			if (!imageData.IsValid())
				continue;
			const Camera& camera = imageData.camera;
			camera_cell_t& camCell = camCells[i];
			camCell.cell = delaunay.locate(MVS2CGAL(camera.C));
			ASSERT(camCell.cell != cell_handle_t());
			fetchCellFacets<CGAL::POSITIVE>(delaunay, viewFrustums[i], hullFacets, camCell.cell, imageData, camCell.facets);
			// link all cells contained by the camera to the source
			for (const facet_t& f : camCell.facets)
				infoCells[f.first->info()].s = kInf;
		}

#ifdef FACET_DIAGNOSTICS // Just used in diagnostics
		numFiniteFacets = 0;
		numFacets = 0;
		for (auto fi = delaunay.facets_begin(), ffi = delaunay.facets_end(); fi != ffi; ++fi) {
			if (!delaunay.is_infinite(*fi)) {
				++numFiniteFacets;
			}
			++numFacets;
		}
#endif

#ifdef FACET_DIAGNOSTICS
		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, %u (+%u) faces (%s)",
			numVertices, delaunay.number_of_vertices(), numFiniteCells, infiniteCells, numFiniteFacets, numFacets - numFiniteFacets, TD_TIMER_GET_FMT().c_str());
#else
		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, faces not calculated (%s)",
			numVertices, delaunay.number_of_vertices(), numFiniteCells, numInfiniteCells, TD_TIMER_GET_FMT().c_str());
#endif
	}

	const float sigma = SQRT(approxMedian)*kSigma;

	// for every camera-point ray intersect it with the tetrahedrons and
	// add alpha_vis(point) to cell's directed edge in the graph
	{
		TD_TIMER_STARTD();
	
#ifdef VALIDATE
		// 37.39s 213334085207
		// estimate the size of the smallest reconstructible object
		DWORD64 t0 = __rdtsc();

		FloatArr distsSq(0, delaunay.number_of_edges());
		for (delaunay_t::Finite_edges_iterator ei=delaunay.finite_edges_begin(), eei=delaunay.finite_edges_end(); ei!=eei; ++ei) {
			const cell_handle_t& c(ei->first);
			distsSq.Insert(normSq(CGAL2MVS<float>(c->vertex(ei->second)->point()) - CGAL2MVS<float>(c->vertex(ei->third)->point())));
		}
		DWORD64 t1 = __rdtsc();
		MESH_DIAG("Median time %llu\n", t1-t0);

		std::nth_element(distsSq.begin(), distsSq.begin() + distsSq.size()/2, distsSq.end());
		const float sigma(SQRT(distsSq[distsSq.size()/2] ) * kSigma); // .GetMedian())* kSigma);
		//const float sigma(SQRT(distsSq.GetMedian())*kSigma);
		MESH_DIAG("Sigma is %f", sigma);

		// Notice we negate inv2SigmaSq here to aid the vector calculations below.
		const float inv2SigmaSq(-0.5f/(sigma*sigma));
		// distsSq may consume a lot of memory.  Delete it now.
		distsSq.Release();

#else
		MESH_DIAG("Sigma is %f", sigma);
		// Notice we negate inv2SigmaSq here to aid the vector calculations below.
		float inv2SigmaSq(-0.5f/(sigma*sigma));
		// distsSq may consume a lot of memory.  Delete it now.
		//distsSq.release();
#endif

		// compute the weights for each edge
		Util::Progress progress(_T("Points weighted"), numDelaunayVertices);

	//	std::atomic<uint64_t> rays = 0;
	//	std::atomic<uint64_t> steps = 0;

#if 0 // original work
		{
			inv2SigmaSq = -inv2SigmaSq; // original logic needs original signma
			std::vector<facet_t> facets;

			TD_TIMER_STARTD();
			Util::Progress progress(_T("Points weighted"), delaunay.number_of_vertices());
			delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());
			const int64_t nVerts(delaunay.number_of_vertices() + 1);
#pragma omp parallel for private(facets)
			for (int64_t i = 0; i < nVerts; ++i) {
				delaunay_t::Vertex_iterator vi;
#pragma omp critical
				vi = vertexIter++;
				vert_info_t& vert(vi->info());
				auto& viewInstance = allViews[vert.idx];
				if (viewInstance.empty())//IsEmpty())
					continue;
				const point_t& p(vi->point());
				const Point3 pt(CGAL2MVS<REAL>(p));

				std::vector<uint32_t> viewIdxs;

				for (auto& i : viewInstance) {
					auto* __restrict src = pointcloud.ViewsStream(i);
					auto cnt = pointcloud.ViewsStreamSize(i);
					std::copy(src, src + cnt, std::back_inserter(viewIdxs));
				}

				std::sort(
					std::begin(viewIdxs),
					std::end(viewIdxs),
					[](const auto lhs, const auto rhs)
					{
						return lhs < rhs;
					}
				);

				auto it = std::begin(viewIdxs);
				const auto end = std::end(viewIdxs);
				while (it != end) {
					// Advance past duplicates
					auto first = it;
					auto current = *it;
					while (it != end && *it == current) {
						++it;
					}
					const uint32_t imageID(current);
					const edge_cap_t alpha_vis(std::distance(first, it));
					const Image& imageData = images[imageID];
					ASSERT(imageData.IsValid());
					const Camera& camera = imageData.camera;
					const camera_cell_t& camCell = camCells[imageID];
					// compute the ray used to find point intersection
					const Point3 vecCamPoint(pt - camera.C);
					const REAL invLenCamPoint(REAL(1) / norm(vecCamPoint));
					intersection_t inter(pt, Point3(vecCamPoint * invLenCamPoint));
					// find faces intersected by the camera-point segment
					const segment_t segCamPoint(MVS2CGAL(camera.C), p);
					if (!intersectv(delaunay, segCamPoint, camCell.facets, facets, inter))
						continue;
					do {
						// assign score, weighted by the distance from the point to the intersection
						const edge_cap_t w(alpha_vis * (1.f - EXP(-SQUARE((float)inter.dist) * inv2SigmaSq)));
						edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
						#pragma omp atomic
						f += w;
					} while (intersectv(delaunay, segCamPoint, facets, facets, inter));
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);

					// cell2Cam only used for free sppace
					// find faces intersected by the endpoint-point segment
					inter.dist = FLT_MAX; inter.bigger = false;
					const Point3 endPoint(pt + vecCamPoint * (invLenCamPoint * sigma));
					const segment_t segEndPoint(MVS2CGAL(endPoint), p);
					const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
					ASSERT(endCell != cell_handle_t());
					fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
					edge_cap_t& t(infoCells[endCell->info()].t);
					#pragma omp atomic
					t += alpha_vis;
					while (intersectv(delaunay, segEndPoint, facets, facets, inter)) {
						// assign score, weighted by the distance from the point to the intersection
						const facet_t& mf(delaunay.mirror_facet(inter.facet));
						const edge_cap_t w(alpha_vis * (1.f - EXP(-SQUARE((float)inter.dist) * inv2SigmaSq)));
						edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
						#pragma omp atomic
						f += w;
					}
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
					// cell2end only used for freespace
				}
				++progress;
			}
			progress.close();
			DEBUG_ULTIMATE("\tweighting completed in %s", TD_TIMER_GET_FMT().c_str());
		}
#else
		constexpr int kViewBatch = 32;   // 4 or 8 are usually best (slightly faster with large batch).
		constexpr int kMaxSteps = 512;
		constexpr int kMaxStepsPerBatch = kViewBatch * kMaxSteps;

		struct ViewCount {
			uint32_t id;
			uint8_t  count;
		};

		struct ThreadData
		{
			PaddedVector<facet_t> mFacets;
			edge_cap_t* mPts[kMaxStepsPerBatch];
			edge_cap_t  mVis[kMaxStepsPerBatch];
			edge_cap_t  mDist[kMaxStepsPerBatch];
			PaddedVector<ViewCount>     mViewCounts;
			uint16_t* mCountByImageID; // direct-indexed lookup, sized to images.size()
		};

		std::vector<ThreadData> perThreadData;

		delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());
		const int64_t nVerts(delaunay.number_of_vertices());

		delaunay_t::Vertex_handle* vertexHandles = (delaunay_t::Vertex_handle*) _aligned_malloc(sizeof(delaunay_t::Vertex_handle) * nVerts, 64);
		{
			delaunay_t::Vertex_iterator it = delaunay.vertices_begin();
			for (int64_t i = 0; i < nVerts; ++i, ++it) {
				vertexHandles[i] = it;
			}
		}

#ifdef VALIDATE
		delaunay_t2::Vertex_iterator vertexIter2(delaunay2.vertices_begin());
		const int64_t nVerts2(delaunay2.number_of_vertices());

		std::vector<delaunay_t2::Vertex_handle> vertexHandles2(nVerts2);
		{
			delaunay_t2::Vertex_iterator it = delaunay2.vertices_begin();
			for (int64_t i = 0; i < nVerts2; ++i, ++it) {
				vertexHandles2[i] = it;
			}
		}
#endif

		float kSigmaCut = sigma * 3.0f;    // or 2.5f if you want to test
		float kMin = sigma * 1e-4f;

#ifdef VALIDATE
#pragma omp parallel num_threads(1)
#else
#pragma omp parallel
#endif
		{
			// First one sets up everything.
#pragma omp single
			{
				int numThreads = omp_get_num_threads();
				perThreadData.resize(numThreads);
			}

			const int id = omp_get_thread_num();
			ThreadData& td = perThreadData[id];

			auto& facets = td.mFacets.mData;
			facets.reserve(512);

			auto& viewCounts = td.mViewCounts.mData;
			viewCounts.resize(images.size());

			td.mCountByImageID = (uint16_t*)_aligned_malloc(sizeof(uint16_t) * images.size(), 64);
			memset(td.mCountByImageID, 0, sizeof(uint16_t)* images.size());

#pragma omp for schedule(static, 1024) // 1024 better than alternatives on 7950X
			for (int64_t i = 0; i < nVerts; ++i) {
#if 1
				auto vi = vertexHandles[i];
#else
				delaunay_t::Vertex_iterator vi;
#pragma omp critical
				vi = vertexIter++;
#endif
				vert_info_t& vert(vi->info());
#if !DIRECT_VIEW_EXPANSION
				auto& viewInstance = allViews[vert.idx];
				if (viewInstance.empty())//IsEmpty())
					continue;
				const point_t& p(vi->point());
				const Point3f pt(p.x(), p.y(), p.z());

				// To accelerate the vert.views creation, we just store
				// them as fast as possible.
				// Here, because there may be duplicates we count them
				// and assign a weight to the point which equals
				// the number of observations from each view.
				uint32_t numViews = 0;
				// viewCounts is sized to images.size() and reused per-vertex.
				// Use a parallel flat array for O(1) duplicate detection.
				// countByImageID[id] holds the index+1 into viewCounts (0 = absent).
				uint16_t* __restrict countByImageID = td.mCountByImageID;

				for (uint32_t v : viewInstance) {
					const uint32_t* src = &pointcloud.pointViewsMemory[offsets[v]];
					const uint32_t  cnt = sizes[v];

					for (uint32_t k = 0; k < cnt; ++k) {
						const uint32_t id = src[k];
						uint16_t& slot = countByImageID[id];
						if (slot != 0) {
							// already seen � increment count
							++viewCounts[slot - 1].count;
						}
						else {
							// new view
							viewCounts[numViews] = { id, 1 };
							slot = static_cast<uint16_t>(numViews + 1);
							++numViews;
						}
					}
				}

				// Reset only the slots we touched (cheaper than memset over all images)
				for (uint32_t j = 0; j < numViews; ++j)
					countByImageID[viewCounts[j].id] = 0;
#else
				const uint16_t vcCount = vcSizes[vert.idx];
				if (vcCount == 0)
					continue;
				const point_t& p(vi->point());
				const Point3f pt(p.x(), p.y(), p.z());

				const uint32_t numViews = vcCount;
				const ExpandedViewCount* __restrict vcEntry = &vcData[vcOffsets[vert.idx]];
#endif

#ifdef VALIDATE
				auto vi2 = vertexHandles2[i];
				vert_info_t2& vert2(vi2->info());
				//vi->info().idx (is the original index).

				std::cerr << "new: \n   ";
				for (auto& i : viewIdxs) {
					std::cerr << i << " ";
				}
				std::cerr << "\n";

				std::vector<uint32_t> vv;
				for (auto& i : vert2.views) {
					vv.push_back(i);
				}

				std::sort(
					std::begin(vv),
					std::end(vv),
					[](const auto lhs, const auto rhs)
					{
						return lhs < rhs;
					}
				);


				std::cerr << "old: \n   ";
				for (auto& i : vv) {
					std::cerr << i << " ";
				}
				std::cerr << "\n";

				std::cerr << "\n";
#endif

				for (uint32_t vBase = 0; vBase < numViews; vBase += kViewBatch) {
					const uint32_t vEnd = std::min(vBase + kViewBatch, numViews);

					// reset accumulation for THIS BATCH
					edge_cap_t** __restrict pPts = td.mPts;
					edge_cap_t* __restrict pVis = td.mVis;
					edge_cap_t* __restrict pDist = td.mDist;
					int totalSteps = 0;

					// ===============================
					// process a small batch of views
					// ===============================
					for (uint32_t viBatch = vBase; viBatch < vEnd; ++viBatch) {
#if !DIRECT_VIEW_EXPANSION
						const uint32_t imageID = viewCounts[viBatch].id;
						const edge_cap_t alpha_vis = edge_cap_t(viewCounts[viBatch].count);
#else
						const uint32_t imageID = vcEntry[viBatch].id;
						const edge_cap_t alpha_vis = edge_cap_t(vcEntry[viBatch].count);
#endif

						const Image& imageData = images[imageID];
						ASSERT(imageData.IsValid());
						const Camera& camera = imageData.camera;
						const camera_cell_t& camCell = camCells[imageID];

						// compute the ray used to find point intersection
						// Points are ahead of us.  No form of culling appears to help here.
						const Point3f camC = Cast<float>(camera.C);
						const Point3f vecCamPointF = pt - camC;
						const float  invLenCamPointF = 1.0f / FastSqrtS(
							vecCamPointF.x * vecCamPointF.x +
							vecCamPointF.y * vecCamPointF.y +
							vecCamPointF.z * vecCamPointF.z);

						// normalized ray direction (float)
						const Point3f dirF = vecCamPointF * invLenCamPointF;
						// find faces intersected by the camera-point segment
						const segment_t segCamPoint(MVS2CGAL(camera.C), p);
						intersection_t inter(
							pt,
							Point3(
								REAL(dirF.x),
								REAL(dirF.y),
								REAL(dirF.z)
							)
						);

						int steps = 0;

						// Prematurely exiting the intersection loop can leave facets non-empty.
						facets.resize(0);
						if (!intersect(delaunay, segCamPoint, camCell.facets, facets, inter, cellNbrID, allCells))
							continue;

						float lastDist = inter.dist;

						for (;;) {
							// store CURRENT intersection
							edge_cap_t& f = infoCells[inter.facet.first->info()].f[inter.facet.second];
							pPts[totalSteps] = &f;
							pVis[totalSteps] = alpha_vis;
							pDist[totalSteps] = (edge_cap_t)inter.dist;

							++totalSteps;
							if (++steps >= kMaxSteps)
								break;

#ifdef EARLY_OUT_WEIGHTING
							if (inter.dist > kSigmaCut)
								break;
#endif

							// advance to NEXT intersection
							if (!intersect(delaunay, segCamPoint, facets, facets, inter, cellNbrID, allCells))
								break;

#ifdef EARLY_OUT_WEIGHTING
							if (inter.dist - lastDist < kMin)
								break;
#endif

							lastDist = inter.dist;
						}

						ASSERT(inter.type == intersection_t::VERTEX && inter.v1 == vi);
						// find faces intersected by the endpoint-point segment
						inter.dist = FLT_MAX; inter.bigger = false;
						const Point3f endPoint(pt + vecCamPointF * (invLenCamPointF * sigma));
						const segment_t segEndPoint(MVS2CGAL(endPoint), p);
						const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
						ASSERT(endCell != cell_handle_t());
						fetchCellFacets<CGAL::NEGATIVE>(delaunay, viewFrustums[imageID], hullFacets, endCell, imageData, facets);
						edge_cap_t& t(infoCells[endCell->info()].t);
						AtomicAddFloat(&t, alpha_vis);

						steps = 0;
						if (intersect(delaunay, segEndPoint, facets, facets, inter, cellNbrID, allCells)) {
							lastDist = inter.dist;
							for (;;) {
								cell_handle_t  c = inter.facet.first;
								const int      i = inter.facet.second;
								// Replace c->neighbor(i) random deref + delaunay.mirror_index(c, i)
								// (which itself does nc->index(c)) with two flat-array reads.
								// c->info() reads a field on the cell record that intersect() just
								// touched, so it's a warm-cache load.
								const cell_size_t cID  = c->info();
								const cell_size_t ncID = cellNbrID[(size_t)cID * 4 + i];
								const int         mi   = (cellNbrSlot[cID] >> (i * 2)) & 3;

								// assign score, weighted by the distance from the point to the intersection
								// inline mirror_facet
								edge_cap_t* fp = &infoCells[ncID].f[mi];
								pPts[totalSteps] = fp;
								pVis[totalSteps] = alpha_vis;
								pDist[totalSteps] = (edge_cap_t)inter.dist;
								++totalSteps;

								if (++steps >= kMaxSteps)
									break;

#ifdef EARLY_OUT_WEIGHTING
								if (inter.dist > kSigmaCut)
									break;
#endif

								if (!intersect(delaunay, segEndPoint, facets, facets, inter, cellNbrID, allCells))
									break;

#ifdef EARLY_OUT_WEIGHTING
								if (inter.dist - lastDist < kMin)
									break;
#endif

								lastDist = inter.dist;
							}
						}

						ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
					}

					// Here we apply the deferred intersection results to the edges all at once.
					// Not faster on 8s
					const _Data vInv2SigmaSq = _Set(inv2SigmaSq);
					const _Data vOne = { 1.f, 1.f, 1.f, 1.f };

					const size_t numFours = totalSteps / 4;
					const size_t numRemaining = totalSteps & 3;

					for (size_t i = 0; i < numFours; ++i) {
						const size_t base = i << 2;

#if SCRREC_OPT_PREFETCH
						// (A) Warm cachelines ~2 quads ahead (8 atomics of latency).
						// pPts is linear so reading the future pointers is free;
						// the targets they point at are random per cell.
						if (i + 2 < numFours) {
							const size_t pf = (i + 2) << 2;
							_mm_prefetch((const char*)pPts[pf + 0], _MM_HINT_T0);
							_mm_prefetch((const char*)pPts[pf + 1], _MM_HINT_T0);
							_mm_prefetch((const char*)pPts[pf + 2], _MM_HINT_T0);
							_mm_prefetch((const char*)pPts[pf + 3], _MM_HINT_T0);
						}
#endif

						// load pointers
						float* p0 = pPts[base + 0];
						float* p1 = pPts[base + 1];
						float* p2 = pPts[base + 2];
						float* p3 = pPts[base + 3];

						// SIMD math
						const _Data vDists = _LoadA(pDist + base);
						const _Data vAlphaVis = _LoadA(pVis + base);
						const _Data vDistsSq = _Mul(vDists, vDists);
						const _Data vDistsSqFactor = _Mul(vDistsSq, vInv2SigmaSq);
						_Data vExp = BetterFastExpSse(vDistsSqFactor); // JPB WIP BUG Why does it fail? FastExpNoClampNegativeRcp(vDistsSqFactor);
						const _Data vOneMinusExpAndFactor = _Sub(vOne, vExp);
						const _Data vResult = _Mul(vOneMinusExpAndFactor, vAlphaVis);
						alignas(16) float vRes[4];
						_mm_store_ps(vRes, vResult);

#if 1
						// common case: all distinct
						// Slightly faster if we check
						if (p0 != p1 && p2 != p3 &&
							p0 != p2 && p1 != p3)
						{
#endif
							AtomicAddFloat(p0, vRes[0]);
							AtomicAddFloat(p1, vRes[1]);
							AtomicAddFloat(p2, vRes[2]);
							AtomicAddFloat(p3, vRes[3]);
#if 1
						}
						else
						{
							// rare slow path
							if (p0 == p1) {
								AtomicAddFloat(p0, vRes[0] + vRes[1]);
								AtomicAddFloat(p2, vRes[2]);
								AtomicAddFloat(p3, vRes[3]);
					}
							else if (p2 == p3) {
								AtomicAddFloat(p0, vRes[0]);
								AtomicAddFloat(p1, vRes[1]);
								AtomicAddFloat(p2, vRes[2] + vRes[3]);
							}
							else {
								// extremely rare: cross-pair duplicates
								float* ptr[4] = { p0, p1, p2, p3 };

								for (int i = 0; i < 4; ++i) {
									if (!ptr[i]) continue;
									float sum = vRes[i];
									for (int j = i + 1; j < 4; ++j) {
										if (ptr[j] == ptr[i]) {
											sum += vRes[j];
											ptr[j] = nullptr;
										}
									}
									AtomicAddFloat(ptr[i], sum);
								}
							}
						}
#endif
					}

					if (numRemaining) {
						const size_t remBase = numFours << 2;

						alignas(16) float tmpDist[4];
						alignas(16) float tmpVis[4];

						// Fill only the valid lanes
						for (size_t i = 0; i < numRemaining; ++i) {
							tmpDist[i] = pDist[remBase + i];
							tmpVis[i] = pVis[remBase + i];
						}

						// SIMD math (always 4-wide)
						const _Data vD = _LoadA(tmpDist);
						const _Data vAlphaVis = _LoadA(tmpVis);
						const _Data vSq = _Mul(vD, vD);
						const _Data vF = _Mul(vSq, vInv2SigmaSq);
						_Data vExp = BetterFastExpSse(vF); // JPB WIP BUG Is this wrong? FastExpNoClampNegativeRcp(vF);
						const _Data vRes = _Mul(_Sub(vOne, vExp), vAlphaVis);

						// Scalar atomics ONLY for valid lanes
						for (size_t i = 0; i < numRemaining; ++i) {
							AtomicAddFloat(pPts[remBase + i], _AsArray(vRes, int(i)));
						}
					}
				}

#ifdef VALIDATE
				it = std::begin(viewIdxs);
				while (it != end) {
					// Advance past duplicates
					auto first = it;
					auto current = *it;
					while (it != end && *it == current) {
						++it;
					}
					const uint32_t imageID(current);
					const edge_cap_t alpha_vis(std::distance(first, it));
					const Image& imageData = images[imageID];
					ASSERT(imageData.IsValid());
					const Camera& camera = imageData.camera;
					const camera_cell_t& camCell = camCells[imageID];
					// compute the ray used to find point intersection
					const Point3 vecCamPoint(pt - camera.C);
					const REAL invLenCamPoint(REAL(1) / norm(vecCamPoint));
					// find faces intersected by the camera-point segment
					const segment_t segCamPoint(MVS2CGAL(camera.C), p);

					std::cout << "oldi (first):   ";
					intersection_t inter(pt, Point3(vecCamPoint * invLenCamPoint));
					if (!intersectv(delaunay, segCamPoint, camCell.facets, facets, inter))
						continue;
					do {
						// assign score, weighted by the distance from the point to the intersection
						const edge_cap_t w(alpha_vis * (1.f - EXP(-SQUARE((float)inter.dist) * inv2SigmaSq)));
						std::cout << w << " ";
						//edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
						//#ifdef DELAUNAY_USE_OPENMP
						//#pragma omp atomic
						//#endif
						//f += w;
					} while (intersectv(delaunay, segCamPoint, facets, facets, inter));
					std::cout << "\n\noldi (second):   ";

					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
					// find faces intersected by the endpoint-point segment
					inter.dist = FLT_MAX; inter.bigger = false;
					const Point3 endPoint2(pt + vecCamPoint * (invLenCamPoint * sigma));
					const segment_t segEndPoint2(MVS2CGAL(endPoint2), p);
					const cell_handle_t endCell(delaunay.locate(segEndPoint2.source(), vi->cell()));
					ASSERT(endCell != cell_handle_t());
					fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
					edge_cap_t& t(infoCells[endCell->info()].t);
					//#ifdef DELAUNAY_USE_OPENMP
					//#pragma omp atomic
					//#endif
					//t += alpha_vis;
					while (intersectv(delaunay, segEndPoint2, facets, facets, inter)) {
						// assign score, weighted by the distance from the point to the intersection
						const facet_t& mf(delaunay.mirror_facet(inter.facet));
						const edge_cap_t w(alpha_vis * (1.f - EXP(-SQUARE((float)inter.dist) * inv2SigmaSq)));
						std::cout << w << " ";
						//edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
						//#ifdef DELAUNAY_USE_OPENMP
						//#pragma omp atomic
						//#endif
						//f += w;
					}
					std::cout << "\n";
				}
#endif

#if 0 // inside now
				// Here we apply the deferred intersection results to the edges all at once.
				const _Data vInv2SigmaSq = _Set(inv2SigmaSq);

#if 1  // on fours
				const size_t numFours = steps / 4;
				const size_t numRemaining = steps & 3;
				float** __restrict pp = pts.data();
				const float* __restrict v = vis.data();
				const float* __restrict d = dist.data();

				const _Data vOne = { 1.f, 1.f, 1.f, 1.f };

				for (size_t i = 0; i < numFours; ++i, d += 4, v += 4, pp += 4) {
					// load pointers
					float* p0 = pp[0];
					float* p1 = pp[1];
					float* p2 = pp[2];
					float* p3 = pp[3];

					// SIMD math
					const _Data vDists = _Load(d);
					const _Data vAlphaVis = _Load(v);
					const _Data vDistsSq = _Mul(vDists, vDists);
					const _Data vDistsSqFactor = _Mul(vDistsSq, vInv2SigmaSq);
					_Data vExp = BetterFastExpSse(vDistsSqFactor);
					const _Data vOneMinusExpAndFactor = _Sub(vOne, vExp);
					const _Data vResult = _Mul(vOneMinusExpAndFactor, vAlphaVis);

					float v0 = _AsArray(vResult, 0);
					float v1 = _AsArray(vResult, 1);
					float v2 = _AsArray(vResult, 2);
					float v3 = _AsArray(vResult, 3);

					// common case: all distinct
					// Slightly faster if we check
					if (p0 != p1 && p2 != p3 &&
						p0 != p2 && p1 != p3)
					{
						AtomicAddFloat(p0, v0);
						AtomicAddFloat(p1, v1);
						AtomicAddFloat(p2, v2);
						AtomicAddFloat(p3, v3);
					}
					else
					{
						// rare slow path
						if (p0 == p1) {
							AtomicAddFloat(p0, v0 + v1);
							AtomicAddFloat(p2, v2);
							AtomicAddFloat(p3, v3);
						}
						else if (p2 == p3) {
							AtomicAddFloat(p0, v0);
							AtomicAddFloat(p1, v1);
							AtomicAddFloat(p2, v2 + v3);
						}
						else {
							// extremely rare: cross-pair duplicates
							float* ptr[4] = { p0, p1, p2, p3 };
							float  val[4] = { v0, v1, v2, v3 };

							for (int i = 0; i < 4; ++i) {
								if (!ptr[i]) continue;
								float sum = val[i];
								for (int j = i + 1; j < 4; ++j) {
									if (ptr[j] == ptr[i]) {
										sum += val[j];
										ptr[j] = nullptr;
									}
								}
								AtomicAddFloat(ptr[i], sum);
							}
						}
					}
				}

				for (size_t i = 0; i < numRemaining; ++i, ++d, ++v, ++pp) {
					float tmp = (*v * (1.f - JPBEXP(SQUARE((float)*d) * inv2SigmaSq)));
					AtomicAddFloat(*pp, tmp);
				}
#else
				const size_t cnt = pPts - pts.data();
				const size_t numEights = cnt / 8;
				const size_t numRemaining = cnt & 7;
				float** __restrict pp = pts.data();
				const float* __restrict v = vis.data();
				const float* __restrict d = dist.data();

				const _Data vOne = { 1.f, 1.f, 1.f, 1.f };

#ifdef VALIDATE
				std::cout << "newi :   ";
#endif
				for (size_t i = 0; i < numEights; ++i, d += 8, v += 8, pp += 8) {
					// Manually unfolded to reduce load/store delays.
					float* p0 = pp[0]; // No __restrict
					float* p1 = pp[1];
					float* p2 = pp[2];
					float* p3 = pp[3];
					float* p4 = pp[4];
					float* p5 = pp[5];
					float* p6 = pp[6];
					float* p7 = pp[7];

					const _Data vDists = _Load(d);
					const _Data vDists2 = _Load(d + 4);
					const _Data vAlphaVis = _Load(v);
					const _Data vAlphaVis2 = _Load(v + 4);
					const _Data vDistsSq = _Mul(vDists, vDists);
					const _Data vDistsSq2 = _Mul(vDists2, vDists2);
					const _Data vDistsSqFactor = _Mul(vDistsSq, vInv2SigmaSq);
					const _Data vDistsSqFactor2 = _Mul(vDistsSq2, vInv2SigmaSq);
					_Data vExp;
					_Data vExp2;
					// JPB WIP OPT Fix the min/max in this for better performance.
					BetterFastExpSsePair(vExp, vExp2, vDistsSqFactor, vDistsSqFactor2);
					const _Data vOneMinusExpAndFactor = _Sub(vOne, vExp);
					const _Data vOneMinusExpAndFactor2 = _Sub(vOne, vExp2);
					const _Data vResult = _Mul(vOneMinusExpAndFactor, vAlphaVis);
					const _Data vResult2 = _Mul(vOneMinusExpAndFactor2, vAlphaVis2);

					const float v0 = _AsArray(vResult, 0);
					const float v1 = _AsArray(vResult, 1);
					const float v2 = _AsArray(vResult, 2);
					const float v3 = _AsArray(vResult, 3);
					const float v4 = _AsArray(vResult2, 0);
					const float v5 = _AsArray(vResult2, 1);
					const float v6 = _AsArray(vResult2, 2);
					const float v7 = _AsArray(vResult2, 3);
#ifdef VALIDATE
					std::cout << v0 << " " << v1 << " " << v2 << " " << v3 << " " << v4 << " " << v5 << " " << v6 << " " << v7 << " ";
#endif
					// Separate merge is much slower.
					AtomicAddFloat(p0, v0);
					AtomicAddFloat(p1, v1);
					AtomicAddFloat(p2, v2);
					AtomicAddFloat(p3, v3);
					AtomicAddFloat(p4, v4);
					AtomicAddFloat(p5, v5);
					AtomicAddFloat(p6, v6);
					AtomicAddFloat(p7, v7);
				}

				for (size_t i = 0; i < numRemaining; ++i, ++d, ++v, ++pp) {
					float tmp = (*v * (1.f - JPBEXP(SQUARE((float)*d) * (inv2SigmaSq))));
#ifdef VALIDATE
					std::cout << tmp << " ";
#endif
					AtomicAddFloat(*pp, tmp);
				}
#endif
#endif

#ifdef VALIDATE
				std::cout << "\n";
#endif

				if (!(i & 1023)) { progress += 1024; }
			}
		} // parallel
#endif

		progress.process();
		progress.close();

		decltype(hullFacets)().swap(hullFacets);
		decltype(viewFrustums)().swap(viewFrustums);

		_aligned_free(vertexHandles);
		vertexHandles = 0;

		// Free per-thread lookup arrays
		for (auto& td : perThreadData)
			_aligned_free(td.mCountByImageID);

#if !DIRECT_VIEW_EXPANSION
		// Confirmed faster to parallel destroy.
#pragma omp parallel for schedule(static, 256)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)numVertices; ++i)
			allViews[i].~view_vec_t();
		_aligned_free(allViews);
		allViews = 0;
		
		_aligned_free(offsets);
		offsets = 0;
		_aligned_free(sizes);
		sizes = 0;
#else
		_aligned_free(vcData);
		vcData = 0;
		_aligned_free(vcOffsets);
		vcOffsets = 0;
		_aligned_free(vcSizes);
		vcSizes = 0;
#endif
		decltype(camCells)().swap(camCells);

	 // DEBUG_EXTRA("Rays %g, steps %g, avg steps per ray %f", (double) rays.load(), (double) steps.load(), (double)steps.load() / (float)rays.load());

#ifdef FACET_DIAGNOSTICS
		DEBUG_EXTRA("Delaunay tetrahedras weighting completed: %u cells, %u faces (%s)", delaunay.number_of_cells(), numFacets, TD_TIMER_GET_FMT().c_str());
		#else
		DEBUG_EXTRA("Delaunay tetrahedras weighting completed: %u cells, unknown faces (%s)", delaunay.number_of_cells(), TD_TIMER_GET_FMT().c_str());
		#endif
	}


		// run graph-cut and extract the mesh
	{
		TD_TIMER_STARTD();

#if 1 // new conservative work
		auto t0 = rdtscStart();

		MaxFlow<cell_size_t, edge_cap_t> graph(totalCells);
		constexpr edge_cap_t maxCap(3.402823466e+34f);

		BuildGraphNodesAndEdges(graph, allCells, delaunay, infoCells, cellNbrID, cellNbrSlot, totalCells, kQual, maxCap, cpuHz);
		auto tFG0 = rdtscEnd();
		graph.FinalizeGraphPrebuilt();
		auto tFG1 = rdtscEnd();
		std::cout << "     [FinalizeGraph ] " << rdtscToSeconds(tFG1 - tFG0, cpuHz) << "\n";

		auto t1 = rdtscStart();
		std::cout << "   Startup: " << rdtscToSeconds(t1 - t0, cpuHz) << "\n";

		// ---------------------------------------------------------------
		// Picard-Queyranne fixability diagnostic.
		// For each node i with excess = s_i - t_i:
		//   sumFwd = sum of arc.rCap for outgoing arcs    (cap i->j)
		//   sumRev = sum of nodes[j].arcs[revIdx].rCap     (cap j->i)
		// Pass-1 fixable counts (no propagation):
		//   excess >  sumFwd  -> S-side
		//  -excess >  sumRev  -> T-side
		// Reports the % of cells that are *immediately* fixable. The
		// iterative propagation pass will only add to this -- this is
		// a strict lower bound on the reduced-graph savings.
		// Side effect: zero (read-only).
		{
			auto tDiag0 = rdtscStart();
			auto* __restrict ibNodes = graph.graph.nodes;
			const ptrdiff_t N = (ptrdiff_t)totalCells;

			size_t fixS = 0, fixT = 0;
			size_t hasArcs = 0;
			double sumFwdAvg = 0.0, sumRevAvg = 0.0;
			double sumExcAvg = 0.0;

			// Arc-count histogram (cnt in [0..4]).
			size_t cntHist0 = 0, cntHist1 = 0, cntHist2 = 0, cntHist3 = 0, cntHist4 = 0;

#pragma omp parallel for schedule(static) \
				reduction(+:fixS,fixT,hasArcs,sumFwdAvg,sumRevAvg,sumExcAvg, \
				           cntHist0,cntHist1,cntHist2,cntHist3,cntHist4)
			for (ptrdiff_t i = 0; i < N; ++i) {
				const auto& u = ibNodes[i];
				const int cnt = u.arcCount;
				edge_cap_t sumFwd = 0.f;
				edge_cap_t sumRev = 0.f;
				for (int a = 0; a < cnt; ++a) {
					const auto& arc = u.arcs[a];
					sumFwd += arc.rCap;
					sumRev += ibNodes[arc.headIdx()].arcs[arc.revIdx()].rCap;
				}
				const edge_cap_t exc = u.excess;
				if (cnt > 0) ++hasArcs;
				sumFwdAvg += (double)sumFwd;
				sumRevAvg += (double)sumRev;
				sumExcAvg += (double)std::abs(exc);
				if      (exc       > sumFwd) ++fixS;
				else if ((-exc)    > sumRev) ++fixT;

				switch (cnt) {
					case 0: ++cntHist0; break;
					case 1: ++cntHist1; break;
					case 2: ++cntHist2; break;
					case 3: ++cntHist3; break;
					case 4: ++cntHist4; break;
					default: break;
				}
			}

			auto tDiag1 = rdtscEnd();
			const double dT = rdtscToSeconds(tDiag1 - tDiag0, cpuHz);
			const double pctS = 100.0 * (double)fixS / (double)N;
			const double pctT = 100.0 * (double)fixT / (double)N;
			const double pctTot = pctS + pctT;
			std::cout << "   [PQ-Fixable Pass1] "
			          << fixS << " S (" << pctS << "%) + "
			          << fixT << " T (" << pctT << "%) = "
			          << (fixS+fixT) << " / " << N
			          << " (" << pctTot << "%)  in " << dT << "s\n";
			if (hasArcs > 0) {
				std::cout << "   [PQ-Fixable means] |excess|=" << (sumExcAvg / (double)hasArcs)
				          << "  sumFwd=" << (sumFwdAvg / (double)hasArcs)
				          << "  sumRev=" << (sumRevAvg / (double)hasArcs) << "\n";
			}
			{
				const double dN = (double)N;
				std::cout << "   [ArcCount Hist   ] "
				          << "0:" << cntHist0 << " (" << (100.0*cntHist0/dN) << "%) "
				          << "1:" << cntHist1 << " (" << (100.0*cntHist1/dN) << "%) "
				          << "2:" << cntHist2 << " (" << (100.0*cntHist2/dN) << "%) "
				          << "3:" << cntHist3 << " (" << (100.0*cntHist3/dN) << "%) "
				          << "4:" << cntHist4 << " (" << (100.0*cntHist4/dN) << "%)\n";
			}
			// Decision guide:
			//   pctTot > 70% -> implement full propagation+reduce, expect 5-15x
			//   30-70%       -> moderate gain expected (~2-3x)
			//   < 30%        -> not worth it on this dataset
		}

		// find graph-cut solution
		const float maxflow(graph.ComputeMaxFlow());

		auto t2 = rdtscEnd();
		std::cout << "   Graph-cut itself: " << rdtscToSeconds(t2 - t1, cpuHz) << "\n";

#if IBSTATS
		{
			IBFS::IBFSStats st = graph.graph.getStats();
			std::cout << "     [IBSTATS augs       ] " << st.getAugs() << "\n";
			std::cout << "     [IBSTATS growthS    ] " << st.getGrowthS() << "\n";
			std::cout << "     [IBSTATS growthT    ] " << st.getGrowthT() << "\n";
			std::cout << "     [IBSTATS growthArcs ] " << st.getGrowthArcs() << "\n";
			std::cout << "     [IBSTATS pushes     ] " << st.getPushes() << "\n";
			std::cout << "     [IBSTATS orphans    ] " << st.getOrphans() << "\n";
			std::cout << "     [IBSTATS orphanArcs1] " << st.getOrphanArcs1() << "\n";
			std::cout << "     [IBSTATS orphanArcs2] " << st.getOrphanArcs2() << "\n";
			std::cout << "     [IBSTATS orphanArcs3] " << st.getOrphanArcs3() << "\n";
			std::cout << "     [IBSTATS augLenMin  ] " << st.getAugLenMin() << "\n";
			std::cout << "     [IBSTATS augLenMax  ] " << st.getAugLenMax() << "\n";
		}
#endif

#ifdef PARALLEL_GRAPH_CUT_EXTRACTION
		// extract surface formed by the facets between inside/outside cells
		ExtractGraphCutSurface(delaunay, allCells, graph, totalCells, idToPoint, mesh, cpuHz);
#else
		const size_t nEstimatedNumVerts(delaunay.number_of_vertices());
		std::unordered_map<void*,Mesh::VIndex> mapVertices;
		#if defined(_MSC_VER) && (_MSC_VER > 1600)
		mapVertices.reserve(nEstimatedNumVerts);
		#endif
		mesh.vertices.Reserve((Mesh::VIndex)nEstimatedNumVerts);
		mesh.faces.Reserve((Mesh::FIndex)nEstimatedNumVerts*2);
		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), ce=delaunay.all_cells_end(); ci!=ce; ++ci) {
			const cell_size_t ciID(ci->info());
			for (int i=0; i<4; ++i) {
				if (delaunay.is_infinite(ci, i)) continue;
				const cell_handle_t cj(ci->neighbor(i));
				const cell_size_t cjID(cj->info());
				if (ciID < cjID) continue;
				const bool ciType(graph.IsNodeOnSrcSide(ciID));
				if (ciType == graph.IsNodeOnSrcSide(cjID)) continue;
				Mesh::Face& face = mesh.faces.AddEmpty();
				const triangle_vhandles_t tri(getTriangle(ci, i));
				for (int v=0; v<3; ++v) {
					const vertex_handle_t vh(tri.verts[v]);
					ASSERT(vh->point() == delaunay.triangle(ci,i)[v]);
					const auto pairItID(mapVertices.insert(std::make_pair(vh.for_compact_container(), (Mesh::VIndex)mesh.vertices.GetSize())));
					if (pairItID.second)
						mesh.vertices.Insert(CGAL2MVS<Mesh::Vertex::Type>(vh->point()));
					ASSERT(pairItID.first->second < mesh.vertices.GetSize());
					face[v] = pairItID.first->second;
				}
				// correct face orientation
				if (!ciType)
					std::swap(face[0], face[2]);
			}
		}
#endif

		_aligned_free(allCells);
		allCells = 0;
		_aligned_free(cellNbrSlot);
		cellNbrSlot = nullptr;
		_aligned_free(cellNbrID);
		cellNbrID = nullptr;

		auto t3 = rdtscEnd();
		std::cout << "   End: " << rdtscToSeconds(t3 - t2, cpuHz) << "\n";
#else

		// create graph
		MaxFlow<cell_size_t,edge_cap_t> graph(delaunay.number_of_cells());

		// set weights
		constexpr edge_cap_t maxCap(3.402823466e+34f/*FLT_MAX*0.0001f*/);
		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), ce=delaunay.all_cells_end(); ci!=ce; ++ci) {
			const cell_size_t ciID(ci->info());
			const cell_info_t& ciInfo(infoCells[ciID]);
			graph.AddNode(ciID, ciInfo.s, MINF(ciInfo.t, maxCap));
			for (int i=0; i<4; ++i) {
				const cell_handle_t cj(ci->neighbor(i));
				const cell_size_t cjID(cj->info());
				if (cjID < ciID) continue;
				const cell_info_t& cjInfo(infoCells[cjID]);
				const int j(cj->index(ci));
				const edge_cap_t q((1.f - MINF(computePlaneSphereAngle(delaunay, facet_t(ci,i)), computePlaneSphereAngle(delaunay, facet_t(cj,j))))*kQual);
				graph.AddEdge(ciID, cjID, ciInfo.f[i]+q, cjInfo.f[j]+q);
			}
		}

		graph.FinalizeGraph();


		// find graph-cut solution
		const float maxflow(graph.ComputeMaxFlow());
		// extract surface formed by the facets between inside/outside cells
		const size_t nEstimatedNumVerts(delaunay.number_of_vertices());
		std::unordered_map<void*,Mesh::VIndex> mapVertices;
		#if defined(_MSC_VER) && (_MSC_VER > 1600)
		mapVertices.reserve(nEstimatedNumVerts);
		#endif
		mesh.vertices.Reserve((Mesh::VIndex)nEstimatedNumVerts);
		mesh.faces.Reserve((Mesh::FIndex)nEstimatedNumVerts*2);
		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), ce=delaunay.all_cells_end(); ci!=ce; ++ci) {
			const cell_size_t ciID(ci->info());
			for (int i=0; i<4; ++i) {
				if (delaunay.is_infinite(ci, i)) continue;
				const cell_handle_t cj(ci->neighbor(i));
				const cell_size_t cjID(cj->info());
				if (ciID < cjID) continue;
				const bool ciType(graph.IsNodeOnSrcSide(ciID));
				if (ciType == graph.IsNodeOnSrcSide(cjID)) continue;
				Mesh::Face& face = mesh.faces.AddEmpty();
				const triangle_vhandles_t tri(getTriangle(ci, i));
				for (int v=0; v<3; ++v) {
					const vertex_handle_t vh(tri.verts[v]);
					ASSERT(vh->point() == delaunay.triangle(ci,i)[v]);
					const auto pairItID(mapVertices.insert(std::make_pair(vh.for_compact_container(), (Mesh::VIndex)mesh.vertices.GetSize())));
					if (pairItID.second)
						mesh.vertices.Insert(CGAL2MVS<Mesh::Vertex::Type>(vh->point()));
					ASSERT(pairItID.first->second < mesh.vertices.GetSize());
					face[v] = pairItID.first->second;
				}
				// correct face orientation
				if (!ciType)
					std::swap(face[0], face[2]);
			}
		}
#endif
		delaunay.clear();
		DEBUG_EXTRA("Delaunay tetrahedras graph-cut completed (%g flow): %u vertices, %u faces (%s)", maxflow, mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

#ifdef FIX_MANIFOLD
	auto tfs = rdtscStart();

#ifdef MANIFOLD_FIXUP
#ifdef PRE_OPENMVS21
	// fix non-manifold vertices and edges
	for (unsigned i = 0; i < nItersFixNonManifold; ++i)
		if (!mesh.FixNonManifold())
			break;
#else
	// fix non-manifold vertices and edges
	mesh.FixNonManifold();
#endif
#endif

	auto tfe = rdtscEnd();
	MESH_DIAG("Manifold time %g", rdtscToSeconds(tfe - tfs, cpuHz));

#endif

	return true;
}
/*----------------------------------------------------------------*/
