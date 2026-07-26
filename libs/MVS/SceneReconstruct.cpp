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
// DEFAULT OFF: a single global threshold tied to the median (densest-region)
// NN spacing over-culls real surface. Point density varies enormously across a
// scene (obliquely-viewed walls, distant ground, canopy fringe), so legitimate
// surface in sparse-but-real regions sits many median-spacings from the nearest
// sample and is indistinguishable from invented skirts/balloons by a global
// distance test. Measured on an aerial scene: ~13.5% of faces removed at 3.5x,
// most of them good geometry. Density trimming (SurfaceTrimmer --trim) already
// handles the invented-balloon case far more reliably. Leave this at 0 unless
// you have a local-density-adaptive replacement for the threshold.
// Set to 1 to re-enable (and tune POISSON_CULL_FACTOR_X100, expect 8-10x+).
#ifndef POISSON_DISTANCE_CULL
#define POISSON_DISTANCE_CULL 0
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
#ifndef POISSON_ADAPTIVE_TRIM
#define POISSON_ADAPTIVE_TRIM 0
#endif
// Edge threshold as a multiple of the interior (passed) trimThreshold, x100.
// 160 = edge threshold is 1.60x the interior threshold (e.g. 5.5 -> 8.8).
#ifndef POISSON_TRIM_EDGE_MULT_X100
#define POISSON_TRIM_EDGE_MULT_X100 160
#endif
// Ramp width (in grid cells) over which the threshold blends interior->edge.
#ifndef POISSON_TRIM_RAMP_CELLS
#define POISSON_TRIM_RAMP_CELLS 6
#endif
// Footprint grid cell size as a multiple of median NN spacing, x100. 300 = 3x.
#ifndef POISSON_TRIM_CELL_FACTOR_X100
#define POISSON_TRIM_CELL_FACTOR_X100 300
#endif
// Morphological dilation radius (cells) to bridge thin shoreline gaps before
// the interior-hole flood fill. 0 disables.
#ifndef POISSON_TRIM_CLOSE_CELLS
#define POISSON_TRIM_CLOSE_CELLS 2
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
	DEBUG_EXTRA("%d filtered (k=%d, stddevMul=%.2f, interiorMul=%.2f, mean=%.4f, stdev=%.4f)",
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

// Estimate a sensible Poisson octree depth from the cloud itself. The useful
// depth is the one whose finest voxel matches the cloud's actual point spacing:
// PoissonRecon builds its octree over a cube of side scale*maxExtent divided into
// 2^depth cells per axis, so the finest cell width = scale*maxExtent / 2^depth.
// Setting that equal to the median nearest-neighbour spacing s gives
//     depth = round( log2( scale * maxExtent / s ) ).
// Beyond this the octree cells contain no new samples and Poisson only amplifies
// noise (a heavier, bumpier mesh for no real detail), so the result is clamped to
// [minDepth, maxDepth] to also bound memory. minDepth/maxDepth keep absurd inputs
// (sparse or huge scenes) from picking a runaway depth.
static int EstimatePoissonDepth(const float* ptsRaw, size_t numPoints,
	float scaleFactor = 1.1f, int minDepth = 8, int maxDepth = 10)
{
	if (numPoints < 100)
		return (minDepth + maxDepth) / 2;
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
	if (!(ext > 0.f))
		return (minDepth + maxDepth) / 2;

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
			pSp[q] = std::sqrt(d2[1]); // d2[0] is the query point itself
	}
	std::vector<float> valid;
	valid.reserve(nq);
	for (size_t q = 0; q < nq; ++q)
		if (pSp[q] > 0.f) valid.push_back(pSp[q]);
	if (valid.empty())
		return (minDepth + maxDepth) / 2;
	std::nth_element(valid.begin(), valid.begin() + valid.size() / 2, valid.end());
	const float s = valid[valid.size() / 2];
	if (!(s > 0.f))
		return (minDepth + maxDepth) / 2;

	// 3) depth whose finest voxel matches the median spacing, clamped.
	int depth = (int)std::lround(std::log2((scaleFactor * ext) / s));
	if (depth < minDepth) depth = minDepth;
	if (depth > maxDepth) depth = maxDepth;
	VERBOSE("Poisson: cloud extent %.4g, median point spacing %.4g -> auto depth %d",
		ext, s, depth);
	return depth;
} // EstimatePoissonDepth
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
bool Scene::ReconstructMeshPoisson(int depth, float trimThreshold, float samplesPerNode, float pointWeight, float islandRatio)
{
	TD_TIMER_STARTD();
	ASSERT(!pointcloud.IsEmpty());
	mesh.Release();
	VERBOSE("Poisson reconstruction (OpenMVS-bmg build %d)", OPENMVS_BMG_BUILD);

	// Poisson requires oriented normals; estimate them if the cloud has none.
	if (!pointcloud.NormalStream()) {
		VERBOSE("Poisson: cloud has no normals, estimating them...");
		// Larger neighborhood (32 vs default 16) yields smoother, more stable
		// PCA normals on flat surfaces, reducing Poisson waviness (e.g. bumpy road).
		EstimatePointNormals(images, pointcloud, 32);
		if (!pointcloud.NormalStream()) {
			VERBOSE("error: Poisson reconstruction requires normals; estimation failed");
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
			VERBOSE("Poisson: skipping %u/%u points with non-finite position/normal",
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
				VERBOSE("error: no finite oriented points for Poisson reconstruction");
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

	// auto-select the octree depth from the cloud's actual point spacing when the
	// caller passes depth <= 0 -- let the data, not a fixed guess, set the detail.
	// Uses the already-filtered poissonPts (all finite, no NaN in kd-tree).
	if (depth <= 0)
		depth = EstimatePoissonDepth(poissonPts, poissonCount);

	// 1) In-process screened-Poisson reconstruction (replaces PoissonRecon.exe),
	// with per-vertex density so the trimmer can threshold by it.
	PoissonReconLib::Mesh pmesh;
	{
		PoissonReconLib::ReconParams rp;
		rp.depth          = depth;
		rp.samplesPerNode = samplesPerNode;
		rp.pointWeight    = pointWeight;
		rp.density        = true;
		rp.verbose        = true;
		VERBOSE("Poisson: running PoissonRecon (depth=%d)...", depth);
		if (!PoissonReconLib::Reconstruct(poissonPts, poissonNrm, poissonCount, rp, pmesh) || pmesh.TriangleCount() == 0) {
			VERBOSE("error: Poisson reconstruction failed");
			return false;
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
		VERBOSE("Poisson: running SurfaceTrimmer (--trim %g%s)...", (double)trimThreshold,
			(islandRatio > 0.f) ? String::FormatString(" --aRatio %g --removeIslands", (double)islandRatio).c_str() : "");
		PoissonReconLib::Mesh tmesh;
		if (PoissonReconLib::Trim(pmesh, tp, tmesh) && tmesh.TriangleCount() > 0)
			pmesh = std::move(tmesh);
		else
			VERBOSE("warning: SurfaceTrimmer produced no output; using untrimmed Poisson mesh");
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
			std::vector<uint8_t> occ(nCells, 0);
			for (size_t i = 0; i < poissonCount; ++i) {
				int cx, cy; cellOf(cloud[i].x, cloud[i].y, cx, cy);
				occ[(size_t)cy * gw + cx] = 1;
			}
			// dilate to bridge thin shoreline gaps
			std::vector<uint8_t> occD = occ;
			const int dr = POISSON_TRIM_CLOSE_CELLS;
			if (dr > 0) {
				for (int cy = 0; cy < gh; ++cy) for (int cx = 0; cx < gw; ++cx) {
					if (!occ[(size_t)cy * gw + cx]) continue;
					for (int dy = -dr; dy <= dr; ++dy) { int ny = cy + dy; if (ny < 0 || ny >= gh) continue;
						for (int dx = -dr; dx <= dr; ++dx) { int nx = cx + dx; if (nx < 0 || nx >= gw) continue;
							occD[(size_t)ny * gw + nx] = 1; } }
				}
			}
			// exterior = flood fill of empty cells from the border; footprint
			// (interior, incl. enclosed water holes) = NOT exterior.
			std::vector<uint8_t> ext(nCells, 0);
			std::vector<int> stk;
			auto pushIf = [&](int cx, int cy) {
				const size_t k = (size_t)cy * gw + cx;
				if (!occD[k] && !ext[k]) { ext[k] = 1; stk.push_back((int)k); }
			};
			for (int cx = 0; cx < gw; ++cx) { pushIf(cx, 0); pushIf(cx, gh - 1); }
			for (int cy = 0; cy < gh; ++cy) { pushIf(0, cy); pushIf(gw - 1, cy); }
			while (!stk.empty()) {
				const int k = stk.back(); stk.pop_back();
				const int cx = k % gw, cy = k / gw;
				if (cx > 0) pushIf(cx - 1, cy);  if (cx < gw - 1) pushIf(cx + 1, cy);
				if (cy > 0) pushIf(cx, cy - 1);  if (cy < gh - 1) pushIf(cx, cy + 1);
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

			const float ramp = (float)POISSON_TRIM_RAMP_CELLS;
			const float trimInterior = trimThreshold;
			const float trimEdge = trimThreshold * (POISSON_TRIM_EDGE_MULT_X100 / 100.f);

			// per-vertex local threshold (e=0 deep interior -> e=1 at perimeter)
			const Mesh::Vertex* __restrict pVtx = mesh.vertices.GetData();
			std::vector<float> vTrim(numV);
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v) {
				int cx, cy; cellOf(pVtx[v].x, pVtx[v].y, cx, cy);
				const float dcell = dist[(size_t)cy * gw + cx];
				float e = 1.f - dcell / ramp; if (e < 0.f) e = 0.f; else if (e > 1.f) e = 1.f;
				vTrim[v] = trimInterior + e * (trimEdge - trimInterior);
			}

			// cull faces: keep iff avg density >= avg local threshold
			const Mesh::FIndex numF = mesh.faces.GetSize();
			std::vector<uint8_t> keepV(numV, 0);
			Mesh::FaceArr newFaces; newFaces.Reserve(numF);
			size_t culled = 0;
			FOREACH(f, mesh.faces) {
				const Mesh::Face& face = mesh.faces[f];
				const float dAvg = (pv[(size_t)face[0] * 4 + 3] + pv[(size_t)face[1] * 4 + 3] +
				                    pv[(size_t)face[2] * 4 + 3]) * (1.f / 3.f);
				const float tAvg = (vTrim[face[0]] + vTrim[face[1]] + vTrim[face[2]]) * (1.f / 3.f);
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
			VERBOSE("Poisson: adaptive footprint trim removed %u faces (interior=%.2f edge=%.2f, ramp=%d, grid %dx%d cell=%.4g) [%s]",
				(unsigned)culled, trimInterior, trimEdge, POISSON_TRIM_RAMP_CELLS, gw, gh, cellSz, TD_TIMER_GET_FMT().c_str());
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
			VERBOSE("Poisson: removed %u non-finite vertices (and incident faces)", (unsigned)numBad);
		}
		_aligned_free(pKeep);
	}

#if POISSON_DISTANCE_CULL
	// Distance-to-cloud cull: delete Poisson faces whose vertices ALL sit
	// farther than (POISSON_CULL_FACTOR_X100/100) * medianSpacing from the
	// nearest actual input point. Orthogonal to SurfaceTrimmer's density trim;
	// removes invented membranes / skirts / balloons that float away from the
	// real samples. See the macro comment near the top of this file.
	if (!mesh.faces.IsEmpty() && !mesh.vertices.IsEmpty() && pointcloud.NumPoints() >= 100) {
		TD_TIMER_STARTD();
		const size_t numCloud = pointcloud.NumPoints();
		const Point3f* __restrict cloudPts = reinterpret_cast<const Point3f*>(pointcloud.PointStream());

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
			const float thr = factor * medianSpacing;
			const float thrSq = thr * thr;

			// Per-vertex: nearest-input-point distance > threshold?
			const Mesh::VIndex numV = mesh.vertices.GetSize();
			std::vector<uint8_t> farV(numV);
			const Mesh::Vertex* __restrict pV = mesh.vertices.GetData();
			uint8_t* __restrict pFar = farV.data();
#ifdef _USE_OPENMP
			#pragma omp parallel for schedule(static)
#endif
			for (ptrdiff_t v = 0; v < (ptrdiff_t)numV; ++v) {
				const Mesh::Vertex& X = pV[v];
				const float qp[3] = { X.x, X.y, X.z };
				uint32_t nIdx; float nD2;
				const size_t found = index.knnSearch(qp, 1, &nIdx, &nD2);
				pFar[v] = (found >= 1 && nD2 > thrSq) ? 1 : 0;
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
				VERBOSE("Poisson: distance cull removed %u/%u faces (thr=%.4g = %.2fx median spacing %.4g) [%s]",
					(unsigned)culled, (unsigned)(culled + keptFaces),
					thr, factor, medianSpacing, TD_TIMER_GET_FMT().c_str());
			} else {
				VERBOSE("Poisson: distance cull removed no faces (thr=%.4g = %.2fx median spacing %.4g)",
					thr, factor, medianSpacing);
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
				VERBOSE("Poisson: skirt cull removed %u boundary faces (Z-descent)", (unsigned)totalSkirtCulled);
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
			const Mesh::FIndex minCompSize = std::max<Mesh::FIndex>(numF / 200, 10);
			size_t blobsRemoved = 0;
			Mesh::FaceArr newFaces;
			newFaces.Reserve(numF);
			for (Mesh::FIndex f = 0; f < numF; ++f) {
				if (compSize[compId[f]] < minCompSize) {
					++blobsRemoved;
					continue;
				}
				newFaces.Insert(pF[f]);
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
				VERBOSE("Poisson: removed %u faces in %u small components (threshold %u faces)",
					(unsigned)blobsRemoved, (unsigned)compSize.size(), (unsigned)minCompSize);
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
				const float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
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
			unsigned nBnd = 0, nBand = 0, nBand2 = 0;
			for (Mesh::VIndex v = 0; v < numV; ++v) {
				if (vClass[v] == 1) ++nBnd;
				else if (vClass[v] == 2) ++nBand;
				else if (vClass[v] == 3) ++nBand2;
			}
			VERBOSE("Poisson: boundary smooth (%d iter, lambda=%.2f/%.2f/%.2f, %u boundary + %u band1 + %u band2 vertices)",
				kBoundarySmooth, lambdaBnd, lambdaBand, lambdaBand2, nBnd, nBand, nBand2);
		}
	}

	DEBUG_EXTRA("Poisson (Tier 2: PoissonRecon%s) reconstructed: %u vertices, %u faces (%s)",
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

					DEBUG_EXTRA("Voxel pre-filter (auto: kPeak=%u, grid=%d, voxel=%.4g, bbox=[%.1f x %.1f x %.1f]): %zu/%zu dropped (%.1f%%) [%s]",
						kPeak, kGrid, voxel, spanX, spanY, spanZ,
						dropped, before,
						100.0 * (double)dropped / (double)before,
						TD_TIMER_GET_FMT().c_str());

					_aligned_free(entries);
				} else {
					DEBUG_EXTRA("Voxel pre-filter: degenerate bbox -- skipped");
				}
			} else {
				DEBUG_EXTRA("Voxel pre-filter: skipped (%s)",
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

					DEBUG_EXTRA("Early confidence filter (bottom %u%%, range=[%.4g,%.4g], thr=%.4g): %zu/%zu dropped (%.1f%%) [%s]",
						kPct, gMin, gMax, thr, dropped, before,
						100.0 * (double)dropped / (double)before,
						TD_TIMER_GET_FMT().c_str());
				} else {
					DEBUG_EXTRA("Early confidence filter: range=[%.4g,%.4g] relSpread=%.3f -- skipped (uniform or below %.0f%% threshold)",
						gMin, gMax, relSpread, kMinRelSpread * 100.0f);
				}

				_aligned_free(ptConf);
			} else {
				DEBUG_EXTRA("Early confidence filter: skipped (%s)",
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
				DEBUG_EXTRA("Confidence filter: pointWeights empty -- skipped");
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
					DEBUG_EXTRA("Confidence filter (bottom %u%%, range=[%.4g,%.4g], thr=%.4g): %zu points dropped (%.1f%% of %zu)",
						kPct, gMin, gMax, thr, dropped,
						100.0 * (double)dropped / (double)totalKept,
						totalKept);
				} else {
					DEBUG_EXTRA("Confidence filter: range=[%.4g,%.4g] relSpread=%.3f -- skipped (uniform or below %.0f%% threshold)",
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

			DEBUG_EXTRA("Total prep time is: %s", TD_TIMER_GET_FMT().c_str());
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
		DEBUG("------------------------------------------");
		DEBUG("ReconstructMesh optimization version 1.1.22");
		const auto [isParallel, CGALversion] = CGAL::info();
		DEBUG("Parallel: %s", isParallel ? "true" : "false");
		DEBUG("CGAL version: = %d", CGALversion);
		constexpr int vcgVersion = vcg::tri::Info();
		DEBUG("VCG version: = %d", vcgVersion);
		DEBUG("------------------------------------------");
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

			DEBUG_EXTRA("View expansion pass completed: %u vertices, %llu view bound (%s)",
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

		DEBUG("Median time %g", rdtscToSeconds(t1 - t0, cpuHz));

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
		DEBUG("Median time %llu\n", t1-t0);

		std::nth_element(distsSq.begin(), distsSq.begin() + distsSq.size()/2, distsSq.end());
		const float sigma(SQRT(distsSq[distsSq.size()/2] ) * kSigma); // .GetMedian())* kSigma);
		//const float sigma(SQRT(distsSq.GetMedian())*kSigma);
		DEBUG_EXTRA("Sigma is %f", sigma);

		// Notice we negate inv2SigmaSq here to aid the vector calculations below.
		const float inv2SigmaSq(-0.5f/(sigma*sigma));
		// distsSq may consume a lot of memory.  Delete it now.
		distsSq.Release();

#else
		DEBUG_EXTRA("Sigma is %f", sigma);
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
	DEBUG_EXTRA("Manifold time %g", rdtscToSeconds(tfe - tfs, cpuHz));

#endif

	return true;
}
/*----------------------------------------------------------------*/
