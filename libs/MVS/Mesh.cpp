/*
* Mesh.cpp
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
#include "Mesh.h"
// fix non-manifold vertices
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include "robin_map.h"
#include "robin_set.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244 4267 4305)
#ifdef _SUPPORT_CPP17
namespace std {
template <typename ArgumentType, typename ResultType>
struct unary_function {
};
} // namespace std
#endif // _SUPPORT_CPP17
#endif // _MSC_VER
// VCG: mesh reconstruction post-processing
#define _SILENCE_STDEXT_HASH_DEPRECATION_WARNINGS
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/stat.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/smooth.h>
#include <vcg/complex/algorithms/hole.h>
#include <vcg/complex/algorithms/polygon_support.h>
#include <vcg/complex/algorithms/isotropic_remeshing.h>
#include <vcg/complex/algorithms/refine.h>
// VCG: mesh simplification
#include <vcg/complex/algorithms/update/position.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/selection.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>
#undef Split
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
// GLTF: mesh import/export
#define JSON_NOEXCEPTION
#define TINYGLTF_NOEXCEPTION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include "../IO/json.hpp"
#include "../IO/tiny_gltf.h"

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

#undef VALIDATE_MESH

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define MESH_USE_OPENMP
#endif

// select fast ray-face intersection search method
#define USE_MESH_BF 0 // brute-force
#define USE_MESH_OCTREE 1 // octree (misses some triangles)
#define USE_MESH_BVH 2 // BVH (misses some triangles)
#define USE_MESH_INT USE_MESH_BVH

// MESH_TOOTH_PEEL_ENABLED: enable the boundary-tooth peel in Mesh::Clean
// (Phase 3.4), which iteratively removes "loose polygon slop" (single
// triangles hanging off the silhouette by one edge, and 1-wide whisker
// chains) from the graph-cut footprint edge. Disabled by default; set to 1
// to enable.
#ifndef MESH_TOOTH_PEEL_ENABLED
#define MESH_TOOTH_PEEL_ENABLED 0
#endif

// MESH_SAIL_PEEL_ENABLED: near-vertical boundary "sail/curtain" peel
// (Mesh::Clean Phase 3.45, right after the tooth peel, before decimation).
// Poisson (and graph-cut) reconstruction drapes a near-VERTICAL skirt of
// triangles off the surface silhouette where density falls away -- the
// "appendages that hang vertically". Raising the Poisson --trim threshold
// removes them but also erodes genuine low-density boundary (loss of
// COMPLETENESS). This pass removes ONLY the drapes: iterative rings that delete
// a face that (a) touches the OPEN boundary, (b) is near-vertical -- its unit
// normal's |z| < MESH_SAIL_PEEL_MAX_NZ_X100/100 (normal nearly horizontal), and
// (c) is elongated -- longest edge > (MESH_SAIL_PEEL_LEN_MULT_X100/100) x median
// edge. Because it keys on VERTICALITY, flat low-density boundary faces (normal
// ~vertical) are KEPT so completeness is preserved; interior building facades
// are safe because they never touch the open boundary. The length gate is an
// extra completeness guard -- set MESH_SAIL_PEEL_LEN_MULT_X100 to 0 to disable
// it and peel on steepness alone. Assumes +Z is up (aerial/top-down scenes).
//   LOWER MAX_NZ = only steeper drapes peeled (gentler); HIGHER = more aggressive.
//   MESH_SAIL_PEEL_ENABLED 0 disables the whole pass (pure A/B off switch).
#ifndef MESH_SAIL_PEEL_ENABLED
#define MESH_SAIL_PEEL_ENABLED 0
#endif
#ifndef MESH_SAIL_PEEL_MAX_NZ_X100
#define MESH_SAIL_PEEL_MAX_NZ_X100 45     // 0.45 -> peel boundary faces steeper than ~63deg
#endif
#ifndef MESH_SAIL_PEEL_LEN_MULT_X100
#define MESH_SAIL_PEEL_LEN_MULT_X100 1200 // 12.0 x median edge (0 = steepness-only, no length gate)
#endif
#ifndef MESH_SAIL_PEEL_MAX_ITERS
#define MESH_SAIL_PEEL_MAX_ITERS 20       // hard cap on erosion rings (self-terminating)
#endif

// Global down-facing cull (Mesh::Clean Phase 3.46, right after the sail peel).
// The sail peel only reaches faces on an OPEN boundary; a watertight Poisson
// envelope (top sheet + down-facing bottom sheet meeting at a manifold
// silhouette fold) has no such boundary, so its underside survives
// (sailRemoved==0). This pass deletes faces by ORIENTATION alone, regardless of
// boundary: any face whose unit normal.z < MESH_DOWN_CULL_MAX_NZ_X100/100 is
// removed. Keep the threshold NEGATIVE so only clearly-downward faces (the
// underside billow) are culled and the top surface + near-vertical facades are
// preserved. Assumes +Z up and coherent outward orientation (set by the
// spurious-removal pass). CAVEAT: also removes legitimate down-facing surfaces
// (eave/overhang undersides) -- acceptable for aerial 2.5D. 0 disables.
#ifndef MESH_DOWN_CULL_ENABLED
#define MESH_DOWN_CULL_ENABLED 0
#endif
#ifndef MESH_DOWN_CULL_MAX_NZ_X100
#define MESH_DOWN_CULL_MAX_NZ_X100 (-10)  // cull faces with unit normal.z < -0.10
#endif

// Underside DEFLATE (Mesh::Clean Phase 3.48). Instead of DELETING the down-facing
// Poisson underside lobes that hang below the terrain (leaving holes), PUSH them
// UP to the surface -- terrain is a height field, so any vertex sitting below
// OTHER surface at its XY is an artifact that should collapse onto the top. Build
// a top-down max-Z grid; any vertex more than MARGIN*medianEdge below the local
// grid top is lifted to (localTop - MARGIN*medianEdge), then the lifted band is
// Laplacian-smoothed in Z to blend the seam. Connectivity is preserved (no
// deletion -> no holes). A vertex with nothing above it (valley, open dip) is the
// local top and is NEVER moved, so genuine low terrain is untouched. ASSUMES +Z up.
//   CELL_MULT = grid cell in median edges (coarser = a downward bulge more reliably
//               shares a cell with the higher surrounding terrain that lifts it).
//   MARGIN_MULT = how far below local top a vertex may stay (median edges).
//   SMOOTH_ITERS = Z-only Laplacian passes over lifted+ring verts to blend.
//   0 disables the pass (A/B off switch).
#ifndef MESH_DEFLATE_ENABLED
#define MESH_DEFLATE_ENABLED 0
#endif
#ifndef MESH_DEFLATE_CELL_MULT_X100
#define MESH_DEFLATE_CELL_MULT_X100 400   // grid cell = 4.00 x median edge
#endif
#ifndef MESH_DEFLATE_MARGIN_MULT_X100
#define MESH_DEFLATE_MARGIN_MULT_X100 200 // lift verts hanging > 2.00 x median edge below local top
#endif
#ifndef MESH_DEFLATE_SMOOTH_ITERS
#define MESH_DEFLATE_SMOOTH_ITERS 5       // Z-Laplacian blend passes over the lifted band
#endif

// Detached-debris removal (Mesh::Clean, "small connected components"),
// RELATIVE-TO-LARGEST. A connected component is kept only if its face count is
// at least this fraction of the LARGEST component's face count; everything
// smaller (floating junk islands) is deleted. The main body is always the
// largest component by a huge margin, so this auto-scales to any scene with no
// hand-tuned absolute size -- robust across datasets. Stored as thousandths of
// a percent: 2000 = 2.000% of the largest. Higher removes more (and risks
// dropping a genuinely-isolated real structure); lower keeps more. Only touches
// DISCONNECTED components; attached peninsulas are never removed.
//
// MEASURED, 115-view corridor scene:
//     DIAG component sizes (top of 2): 1307464 17485
// Only two components exist and the gap between them is 75x. The island is
// 1.337% of the body, so a 0.1% floor (1,307 faces) and a 1% floor (13,074
// faces) BOTH pass it -- it survived Clean and was visible as a floating blob in
// the render. 2% is 26,149 faces, which sits inside that gap and is the first
// floor that actually drops it. Removing it HERE also means it never reaches
// TextureMesh, so the texture-stage orphan filter does not have to catch it after
// decimation has scaled every count.
//
// This is the correct lever for floating blobs -- NOT --poisson-island-ratio,
// which is a SurfaceTrimmer MERGE knob (raising it merges across the trim contour
// and can mask trimming entirely; keep it at 0 / 0.001).
//
// CAVEAT: this drops a genuinely separate structure smaller than 2% of the body.
// Defensible for single-site aerial surveys, which are one contiguous surface;
// lower it if the input is ever several distinct objects. Read the component-sizes
// DIAG line before changing it -- if there is no clear gap, no floor is the right
// tool.
#ifndef MESH_KEEP_COMPONENT_PCT_X1000
#define MESH_KEEP_COMPONENT_PCT_X1000 2000
#endif

// Hole-closing geometry gate (Mesh::Clean Phase 4 / Phase 8).
// Edge count alone cannot tell a genuine INTERIOR hole from the outer
// silhouette / a concave bay (both can be large), so filling by an edge-count
// cap fans sheets across the footprint. The fan-preventer is GEOMETRIC: a
// boundary loop is filled only if its 3D bounding-box diagonal is at most
// MESH_HOLE_MAX_DIAG_FRAC of the whole-mesh bbox diagonal -- interior holes are
// compact (small loop bbox) and pass; the footprint perimeter and the bays
// that are part of it span a large fraction of the mesh and are left open.
// The edge cap itself stays user-controlled via --close-holes (nCloseHoles):
// raise it to close bigger interior holes. MESH_HOLE_MAX_EDGES is only a hard
// runtime safety ceiling (per-hole ear-cutting is O(n^2)). The frac is stored
// as int*1000 to stay preprocessor-friendly (100 = 0.10).
#ifndef MESH_HOLE_MAX_DIAG_FRAC_X1000
#define MESH_HOLE_MAX_DIAG_FRAC_X1000 100  // 0.10 of mesh diagonal
#endif
#ifndef MESH_HOLE_MAX_EDGES
#define MESH_HOLE_MAX_EDGES 2000           // hard safety ceiling on loop edges
#endif

// SECOND geometric gate, in units of the mesh's own resolution rather than the
// scene's extent. The frac gate above is scene-RELATIVE, so on a large site it
// stops bounding anything physical: MEASURED on a 1251 m scene it worked out to a
// 157 m allowance, and the edge cap (92 edges x 1.28 m median edge, ~53 m across
// for a compact loop) was the only thing actually limiting fills. 2,194 holes were
// sealed, and holes that wide are not surface the solve was ever uncertain about --
// no camera observed them, so the fill is invented geometry that then textures
// black. That is the one defect visible in a finished render.
//
// A hole is safe to fill when the surrounding surface still constrains it, and
// "surrounding" is measured in the mesh's OWN sampling, not in metres: at 12 median
// edges the fan spans about a dozen triangles and the fill stays local to geometry
// that was actually observed. Beyond that the interpolation is unconstrained.
//
// Deliberately expressed in median edges, not in Poisson cells: Clean() runs after
// decimation (and inside RefineMesh's subdivide loop), so the cell size is neither
// available here nor still true of the mesh -- but the median edge is measured from
// the mesh in hand and tracks whatever it has become.
//
// Set to 0 to disable and revert to the frac gate alone.
#ifndef MESH_HOLE_MAX_SPAN_EDGES
#define MESH_HOLE_MAX_SPAN_EDGES 12
#endif

// Density-aware edge cap for --close-holes. A boundary loop of a FIXED physical
// size has ~perimeter/median-edge edges, so the same edge count closes a
// physically SMALLER hole as the mesh gets finer (e.g. lowering
// --min-point-distance shrinks the median edge ~linearly). To keep --close-holes
// meaning a CONSTANT physical hole size regardless of density, the effective
// edge cap is scaled by the mesh fineness, measured scene-invariantly as
// (mesh-bbox-diagonal / median-edge) = "edges across the mesh diagonal".
// MESH_HOLE_REF_EDGES_ACROSS_DIAG is the fineness at which --close-holes is
// calibrated: at that fineness the scale factor is 1.0 (backward-compatible);
// a 2.5x-finer mesh gets a 2.5x-larger cap. Both terms scale with scene units,
// so the ratio is dimensionless and works at any absolute scale. Set to 0 to
// disable scaling (revert to the raw --close-holes edge cap).
#ifndef MESH_HOLE_REF_EDGES_ACROSS_DIAG
#define MESH_HOLE_REF_EDGES_ACROSS_DIAG 4000
#endif

// Fallback fill for the SMALL holes that SelfIntersectionEar refuses to fill
// ("ear-rejected": real 3D tears between structures where a flat ear would
// intersect nearby relief). TrivialEar closes them regardless of intersection,
// making the mesh watertight -- but a non-planar hole may get a small flat
// cap/fold. Gated to <= this many boundary edges AND <= 2x the
// MESH_HOLE_MAX_DIAG_FRAC gate, so the outer footprint boundary and any wide
// bay can never be trivially bridged (no large fans). Set to 0 to disable and
// leave these holes open (zero fan risk, but visible gaps remain).
#ifndef MESH_HOLE_FALLBACK_MAX_EDGES
#define MESH_HOLE_FALLBACK_MAX_EDGES 80
#endif

// Alpha-shape perimeter tightening (Mesh::Clean Phase 9, AFTER hole-closing).
// "Rolls a disk of radius alpha around the silhouette" the SAFE (erosion-only)
// way: iteratively delete rim faces whose circumradius > alpha -- that is the
// alpha-shape criterion (a triangle too big/thin for a radius-alpha disk to
// certify is not part of the alpha-solid). This peels saw-tooth slivers and
// spikes off the ragged edge while genuine bays (lined with normal faces)
// survive. It ONLY deletes, never bridges, so it can never create a fan.
//   alpha = (MESH_ALPHA_TIGHTEN_K_X100/100) * median edge length (auto-scales).
//   LOWER K = more aggressive (nibbles more edge); HIGHER K = gentler.
//   Iterations are capped (can never cascade into the decimated interior).
// A/B: set MESH_ALPHA_TIGHTEN_ENABLED to 0 to skip the whole Phase 9 block.
#ifndef MESH_ALPHA_TIGHTEN_ENABLED
#define MESH_ALPHA_TIGHTEN_ENABLED 1
#endif
#ifndef MESH_ALPHA_TIGHTEN_K_X100
#define MESH_ALPHA_TIGHTEN_K_X100 500   // was 300 3.00 * median edge
#endif
#ifndef MESH_ALPHA_TIGHTEN_ITERS
#define MESH_ALPHA_TIGHTEN_ITERS 6      // hard cap on erosion rings (safety)
#endif

// Adaptive (error-bounded) decimation COMPILE-TIME DEFAULT for the strength k (= X100/100).
// Normally this is driven at runtime by ReconstructMesh's --decimate-error; this macro is
// only the fallback used when that CLI value is 0. When k>0, Phase 3.5 stops at a geometric
// -error tolerance tau = scale*(k*medianEdge)^2 instead of a fixed face fraction: flat
// regions collapse maximally, detail is preserved, and the final face count emerges from
// the geometry. --decimate then acts as the KEEP FLOOR (never decimate below that
// fraction). 0 = DISABLED (legacy fixed-ratio behavior, default). Raise k for lighter
// meshes, lower for more detail.
#ifndef MESH_DECIMATE_ERROR_K_X100
#define MESH_DECIMATE_ERROR_K_X100 0    // 0 = off; e.g. 100 = k=1.0
#endif

// AGGRESSIVE perimeter straightening (Mesh::Clean Phase 8.5, AFTER hole-closing,
// BEFORE alpha-tighten). Unlike the tooth-peel (only faces with >=2 open edges)
// and alpha-tighten (only over-large circumradius slivers), this UNIFORMLY peels
// MESH_RIM_ERODE_RINGS full rings of border faces off the whole silhouette. Any
// fringe narrower than ~2R triangles (whisker tendrils, thin peninsulas, isthmus
// necks bridging floating flecks) is completely consumed, and the jagged outline
// recedes to a smoother R-rings-in contour. Because erosion can sever the thin
// necks that connect floating junk to the body, the small-connected-component
// filter is re-run afterwards to drop anything newly disconnected.
//   HIGHER R = more aggressive straightening (loses more genuine edge detail).
//   R = 0 disables the whole pass (pure A/B off switch).
// Only touches BORDER faces, so it never reopens a sealed interior hole.
#ifndef MESH_RIM_ERODE_RINGS
#define MESH_RIM_ERODE_RINGS 0
#endif

// Boundary-only TAUBIN smoothing (Mesh::Clean Phase 10, LAST geometric pass).
// Smooths the ragged silhouette polyline IN PLACE without receding it: only
// open-boundary vertices that lie on a clean 2-neighbor border edge are moved;
// interior geometry is never touched. Taubin's alternating shrink(lambda)/
// inflate(mu) steps cancel the curve-shortening that plain Laplacian smoothing
// causes, so the outline gets smoother WITHOUT losing coverage (unlike erosion).
// Pinch/junction boundary vertices (>2 border neighbors) are skipped (safe).
//   ITERS = number of lambda+mu Taubin pairs; LAMBDA/MU stored as *100.
//   Set MESH_BOUNDARY_SMOOTH_ENABLED 0 to skip the whole pass (A/B off).
#ifndef MESH_BOUNDARY_SMOOTH_ENABLED
#define MESH_BOUNDARY_SMOOTH_ENABLED 1
#endif
#ifndef MESH_BOUNDARY_SMOOTH_ITERS
#define MESH_BOUNDARY_SMOOTH_ITERS 10 // Was 40     // lambda+mu pairs (scale up with band density)
#endif
#ifndef MESH_BOUNDARY_SMOOTH_RINGS
#define MESH_BOUNDARY_SMOOTH_RINGS 1 // Was 8      // band thickness: rings smoothed inward from the edge
#endif
#ifndef MESH_BOUNDARY_SMOOTH_LAMBDA_X100
#define MESH_BOUNDARY_SMOOTH_LAMBDA_X100 50   // 0.50 shrink step
#endif
#ifndef MESH_BOUNDARY_SMOOTH_MU_X100
#define MESH_BOUNDARY_SMOOTH_MU_X100 53       // 0.53 inflate step (magnitude)
#endif

// Boundary-band REFINEMENT (Mesh::Clean Phase 9.5, AFTER alpha-tighten, BEFORE
// the boundary smooth). Decimation preserves the silhouette at full RECONSTRUCTION
// density but never denser, so smoothing the edge can only do so much; this
// midpoint-subdivides the faces within MESH_BAND_REFINE_RINGS rings of the open
// boundary, adding NEW vertices around the edges (and on the silhouette line
// itself) that the subsequent Taubin smooth then rounds far more finely. Only the
// edge band is refined -- the decimated interior stays light, so texture/file cost
// stays bounded. LEVELS=2 quadruples band density again (use sparingly).
//   RINGS = band thickness refined; LEVELS = number of 1->4 midpoint splits.
//   Set MESH_BAND_REFINE_ENABLED 0 to skip the whole pass (A/B off).
#ifndef MESH_BAND_REFINE_ENABLED
#define MESH_BAND_REFINE_ENABLED 1
#endif
#ifndef MESH_BAND_REFINE_RINGS
#define MESH_BAND_REFINE_RINGS 4
#endif
#ifndef MESH_BAND_REFINE_LEVELS
#define MESH_BAND_REFINE_LEVELS 2
#endif

// Edge DILATION / outward apron (Mesh::Clean Phase 9.7, AFTER band refine, BEFORE
// the boundary smooth). ENLARGES coverage by extruding the open silhouette
// OUTWARD: each ring offsets every boundary vertex along its (direction-smoothed)
// in-plane outward normal by MESH_EDGE_DILATE_STEP_X100% of the median edge, at
// the rim's own height, and stitches a triangle strip to the old edge. Repeats
// for MESH_EDGE_DILATE_RINGS rings. The new apron faces are unobserved -> they
// take approximate/flat color at texturing (the competitor's vertex-colored
// outer margin). DIRSMOOTH passes blur the offset direction along the boundary to
// reduce self-intersection at concave notches.
//   CAVEAT: large margins on a ragged non-convex silhouette WILL self-intersect
//   at canopy notches -- keep RINGS/STEP modest, or do dilation in the raster
//   (orthophoto) domain for big, fold-free extensions.
//   RINGS = 0 disables the whole pass (A/B off).
#ifndef MESH_EDGE_DILATE_RINGS
#define MESH_EDGE_DILATE_RINGS 0
#endif
#ifndef MESH_EDGE_DILATE_STEP_X100
#define MESH_EDGE_DILATE_STEP_X100 150   // 1.50 x median edge per ring
#endif
#ifndef MESH_EDGE_DILATE_DIRSMOOTH
#define MESH_EDGE_DILATE_DIRSMOOTH 4     // outward-direction blur passes per ring
#endif

#if USE_MESH_INT == USE_MESH_BVH
#include <unsupported/Eigen/BVH>
#endif

static constexpr size_t BLOCK_SIZE = 8 * 1024 * 1024;  // objects per block
std::vector<void*> g_qBlocks;
size_t g_qOffset = BLOCK_SIZE;

// S T R U C T S ///////////////////////////////////////////////////

// free all memory
void Mesh::Release()
{
	vertices.Release();
	faces.Release();
	ReleaseExtra();
} // Release
void Mesh::ReleaseExtra()
{
	vertexNormals.Release();
	vertexVertices.Release();
	vertexFaces.Release();
	vertexBoundary.Release();
	faceNormals.Release();
	faceFaces.Release();
	faceTexcoords.Release();
	textureDiffuse.release();
} // ReleaseExtra
void Mesh::EmptyExtra()
{
	vertexNormals.Empty();
	vertexVertices.Empty();
	vertexFaces.Empty();
	vertexBoundary.Empty();
	faceNormals.Empty();
	faceFaces.Empty();
	faceTexcoords.Empty();
	textureDiffuse.release();
} // EmptyExtra
Mesh& Mesh::Swap(Mesh& rhs)
{
	vertices.Swap(rhs.vertices);
	faces.Swap(rhs.faces);
	vertexNormals.Swap(rhs.vertexNormals);
	vertexVertices.Swap(rhs.vertexVertices);
	vertexFaces.Swap(rhs.vertexFaces);
	vertexBoundary.Swap(rhs.vertexBoundary);
	faceNormals.Swap(rhs.faceNormals);
	faceFaces.Swap(rhs.faceFaces);
	faceTexcoords.Swap(rhs.faceTexcoords);
	std::swap(textureDiffuse, rhs.textureDiffuse);
	return *this;
} // Swap
// combine this mesh with the given mesh, without removing duplicate vertices
Mesh& Mesh::Join(const Mesh& mesh)
{
	ASSERT(!HasTexture() && !mesh.HasTexture());
	if (mesh.IsEmpty())
		return *this;
	vertexVertices.Release();
	vertexFaces.Release();
	vertexBoundary.Release();
	faceFaces.Release();
	if (IsEmpty()) {
		*this = mesh;
		return *this;
	}
	const VIndex offsetV(vertices.size());
	vertices.Join(mesh.vertices);
	vertexNormals.Join(mesh.vertexNormals);
	faces.ReserveExtra(mesh.faces.size());
	for (const Face& face: mesh.faces)
		faces.emplace_back(face.x+offsetV, face.y+offsetV, face.z+offsetV);
	faceNormals.Join(mesh.faceNormals);
	return *this;
}
/*----------------------------------------------------------------*/


bool Mesh::IsWatertight()
{
	if (vertexBoundary.empty()) {
		if (vertexFaces.empty())
			ListIncidenteFaces();
		ListBoundaryVertices();
	}
	for (const bool b : vertexBoundary)
		if (b)
			return false;
	return true;
}

// compute the axis-aligned bounding-box of the mesh
Mesh::Box Mesh::GetAABB() const
{
	Box box(true);
	for (const Vertex& X: vertices)
		box.InsertFull(X);
	return box;
}
// same, but only for vertices inside the given AABB
Mesh::Box Mesh::GetAABB(const Box& bound) const
{
	Box box(true);
	for (const Vertex& X: vertices)
		if (bound.Intersects(X))
			box.InsertFull(X);
	return box;
}

// compute the center of the point-cloud as the median
Mesh::Vertex Mesh::GetCenter() const
{
	const VIndex step(5);
	const VIndex numPoints(vertices.size()/step);
	if (numPoints == 0)
		return Vertex::INF;
	typedef CLISTDEF0IDX(Vertex::Type,VIndex) Scalars;
	Scalars x(numPoints), y(numPoints), z(numPoints);
	for (VIndex i=0; i<numPoints; ++i) {
		const Vertex& X = vertices[i*step];
		x[i] = X.x;
		y[i] = X.y;
		z[i] = X.z;
	}
	return Vertex(x.GetMedian(), y.GetMedian(), z.GetMedian());
}
/*----------------------------------------------------------------*/


// extract array of vertices incident to each vertex
void Mesh::ListIncidenteVertices()
{
#if 1
	// vertices.size() = nVerts, faces.size() = nFaces
	vertexVertices.clear();
	vertexVertices.resize(vertices.size());

	// ---------- 1.  count degree ----------
	std::vector<uint32_t> degree(vertices.size(), 0);
#pragma omp parallel for schedule(static)
	for (int f = 0; f < (int)faces.size(); ++f) {
		const Face& face = faces[f];
		// each undirected edge contributes to both ends
		_InterlockedIncrement(reinterpret_cast<long*>(&degree[face[0]]));
		_InterlockedIncrement(reinterpret_cast<long*>(&degree[face[1]]));
		_InterlockedIncrement(reinterpret_cast<long*>(&degree[face[1]]));
		_InterlockedIncrement(reinterpret_cast<long*>(&degree[face[2]]));
		_InterlockedIncrement(reinterpret_cast<long*>(&degree[face[2]]));
		_InterlockedIncrement(reinterpret_cast<long*>(&degree[face[0]]));
	}

	// ---------- 2.  prefix-sum offsets ----------
	std::vector<uint64_t> offset(vertices.size() + 1, 0);
	for (size_t v = 0; v < vertices.size(); ++v)
		offset[v + 1] = offset[v] + degree[v];
	const uint64_t nEdgesDir = offset.back();

	std::vector<VIndex> adj(nEdgesDir);    // contiguous neighbor storage
	std::fill(degree.begin(), degree.end(), 0);  // reuse as write cursor

	// ---------- 3.  fill adjacency ----------
#pragma omp parallel for schedule(static)
	for (int f = 0; f < (int)faces.size(); ++f) {
		const Face& face = faces[f];
		for (int e = 0; e < 3; ++e) {
			VIndex a = face[e];
			VIndex b = face[(e + 1) % 3];
			uint64_t posA = offset[a] + _InterlockedIncrement(reinterpret_cast<long*>(&degree[a])) - 1;
			uint64_t posB = offset[b] + _InterlockedIncrement(reinterpret_cast<long*>(&degree[b])) - 1;
			adj[posA] = b;
			adj[posB] = a;
		}
	}

	// ---------- 4.  deduplicate neighbors per vertex ----------
#pragma omp parallel for schedule(static)
	for (int v = 0; v < (int)vertices.size(); ++v) {
		const uint64_t start = offset[v];
		const uint64_t end = offset[v + 1];
		auto first = adj.begin() + start;
		auto last = adj.begin() + end;
		std::sort(first, last);              // very small local sorts
		last = std::unique(first, last);

		const size_t uniqueCount = (last - first);
		vertexVertices[v].clear();
		vertexVertices[v].reserve(uniqueCount);
		for (auto it = first; it != last; ++it) {
			vertexVertices[v].push_back(*it);
		}
	}
#else
	vertexVertices.Empty();
	vertexVertices.Resize(vertices.GetSize());
	FOREACH(i, faces) {
		const Face& face = faces[i];
		for (int v=0; v<3; ++v) {
			VertexIdxArr& verts(vertexVertices[face[v]]);
			for (int i=1; i<3; ++i) {
				const VIndex idxVert(face[(v+i)%3]);
				if (verts.Find(idxVert) == VertexIdxArr::NO_INDEX)
					verts.Insert(idxVert);
			}
		}
	}
#endif
}

// extract the (ordered) array of triangles incident to each vertex
void Mesh::ListIncidenteFaces()
{
	vertexFaces.clear();
	vertexFaces.resize(vertices.size());

	// Pre-pass: remove degenerate faces
	size_t write = 0;
	for (size_t i = 0; i < faces.size(); ++i) {
		const Face& face = faces[i];
		if (face[0] != face[1] &&
			face[1] != face[2] &&
			face[2] != face[0]) {
			if (write != i)
				faces[write] = face;
			++write;
		}
	}
	faces.resize(write);

	// Build adjacency (indices are now monotonically increasing and already sorted)
	for (size_t i = 0; i < faces.size(); ++i) {
		const Face& face = faces[i];
		for (int v = 0; v < 3; ++v)
			vertexFaces[face[v]].push_back((FIndex)i);
	}
}

// extract array face adjacencies for each face in the mesh (3 * number of faces);
// each triple describes the adjacent face triangles for a given face
// in the following edge order: v1v2, v2v3, v3v1;
// NO_ID indicates there is no adjacent face on that edge
#if 1
void Mesh::ListIncidenteFaceFaces()
{
	ASSERT(vertexFaces.size() == vertices.size());

	faceFaces.resize(faces.size());

	// Parallel: writes to faceFaces[f] are unique per face (no aliasing across
	// threads); reads of vertexFaces[*] are const. Static schedule keeps cache
	// behavior contiguous per-thread.
	const int numFaces = (int)faces.size();
	#pragma omp parallel for schedule(static)
	for (int f = 0; f < numFaces; ++f)
	{
		const Face& face = faces[f];
		FaceFaces& out = faceFaces[f];

		for (int e = 0; e < 3; ++e)
		{
			const FaceIdxArr& A = vertexFaces[face[e]];
			const FaceIdxArr& B = vertexFaces[face[(e + 1) % 3]];

			FIndex adj = NO_ID;

			size_t i = 0;
			size_t j = 0;
			const size_t na = A.size();
			const size_t nb = B.size();

			// Full two-pointer intersection scan (like std::set_intersection)
			while (i < na && j < nb)
			{
				const FIndex fa = A[i];
				const FIndex fb = B[j];

				if (fa == fb)
				{
					if (fa != (FIndex)f)
					{
						adj = fa;
						break;          // manifold case: first valid match
					}
					++i;
					++j;
				}
				else if (fa < fb)
				{
					++i;
				}
				else
				{
					++j;
				}
			}

			out[e] = adj;
		}
	}
}
#else
void Mesh::ListIncidenteFaceFaces()
{
	ASSERT(vertexFaces.size() == vertices.size());
	struct inserter_data_t {
		const FIndex idxF;
		FaceFaces& faces;
		int idx;
		inline inserter_data_t(FIndex _idxF, FaceFaces& _faces) : idxF(_idxF), faces(_faces), idx(0) {}
		inline void operator=(FIndex f) { faces[idx++] = f; }
	};
	struct face_back_inserter_t {
		inserter_data_t* data;
		inline face_back_inserter_t(inserter_data_t& _data) : data(&_data) {}
		inline face_back_inserter_t& operator*() { return *this; }
		inline face_back_inserter_t& operator++() { return *this; }
		inline void operator=(FIndex f) { if (f != data->idxF) *data = f; }
	};
	faceFaces.resize(faces.size());
	FOREACH(f, faces) {
		const Face& face = faces[f];
		const FaceIdxArr* const pFaces[] = {&vertexFaces[face[0]], &vertexFaces[face[1]], &vertexFaces[face[2]]};
		inserter_data_t inserterData(f, faceFaces[f]);
		face_back_inserter_t faceBackInserter(inserterData);
		for (int v=0; v<3; ++v) {
			const FaceIdxArr& facesI = *pFaces[v];
			const FaceIdxArr& facesJ = *pFaces[(v+1)%3];
			std::set_intersection(
				facesI.begin(), facesI.end(),
				facesJ.begin(), facesJ.end(),
				faceBackInserter);
			if (inserterData.idx == v)
				inserterData = NO_ID;
		}
	}
}
#endif

// check each vertex if it is at the boundary or not
// (make sure you called ListIncidenteFaces() before)
void Mesh::ListBoundaryVertices()
{
#if 1
	vertexBoundary.clear();
	vertexBoundary.resize(vertices.size());
	vertexBoundary.Memset(0);

#pragma omp parallel
	{
		std::vector<VIndex> neigh;
		std::vector<uint8_t> count;

		neigh.reserve(64);
		count.reserve(64);

#pragma omp for schedule(static)
		for (int v = 0; v < (int)vertices.size(); ++v) {
			const auto& vf = vertexFaces[v];
			neigh.clear();
			count.clear();

			for (VIndex fIdx : vf) {
				const Face& f = faces[fIdx];
				for (int k = 0; k < 3; ++k) {
					VIndex n = f[k];
					if (n == v) continue;

					int i;
					for (i = 0; i < (int)neigh.size(); ++i)
						if (neigh[i] == n) break;
					if (i == (int)neigh.size()) {
						neigh.push_back(n);
						count.push_back(1);
					}
					else {
						++count[i];
					}
				}
			}

			for (int i = 0; i < (int)neigh.size(); ++i) {
				if (count[i] == 1) {
					vertexBoundary[v] = true;
					break;
				}
			}
		}
	}
#else
	vertexBoundary.clear();
	vertexBoundary.resize(vertices.size());
	vertexBoundary.Memset(0);
	VertCountMap mapVerts; mapVerts.reserve(12*2);
	FOREACH(idxV, vertices) {
		const FaceIdxArr& vf = vertexFaces[idxV];
		// count how many times vertices in the first triangle ring are seen;
		// usually they are seen two times each as the vertex in not at the boundary
		// so there are two triangles (on the ring) containing same vertex
		ASSERT(mapVerts.empty());
		FOREACHPTR(pFaceIdx, vf) {
			const Face& face = faces[*pFaceIdx];
			for (int i=0; i<3; ++i) {
				const VIndex idx(face[i]);
				if (idx != idxV)
					++mapVerts[idx].count;
			}
		}
		for (const auto& vc: mapVerts) {
			ASSERT(vc.second.count == 1 || vc.second.count == 2);
			if (vc.second.count != 2) {
				vertexBoundary[idxV] = true;
				break;
			}
		}
		mapVerts.clear();
	}
#endif
}


// compute normal for all faces
// Unit face normal that is SAFE on a degenerate face.
//
// normalized() is cv::normalize(), i.e. v * (1/norm(v)), divided unguarded -- so a
// zero-area face gives 0/0 = NaN. MEASURED: a 3,152,587-face refined mesh carried 24
// faces with an exactly zero-length cross product.
//
// A NaN normal is not a local defect. Any consumer that AVERAGES normals over a region
// spreads it across that whole region, and the usual `if (len < eps)` fallbacks are
// FALSE for NaN, so they never fire. SceneTexture's data-colour fill sums face normals
// per connected component to build the tile's projection basis: two of those 24 faces
// landed in large no-view components and turned 79,059 vertices' texcoords into NaN,
// i.e. two tree-canopy-sized BLACK blobs in the atlas.
//
// A zero normal instead of NaN is the containable failure: it sums benignly, and every
// existing length guard in the codebase already handles it.
static inline Mesh::Normal SafeNormalizeFaceNormal(const Mesh::Normal& n)
{
	const float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
	// `> 0` (not `< eps` negated) so NaN, which compares false against everything,
	// takes the fallback branch rather than sailing through it.
	return (len > 0.f) ? Mesh::Normal(n.x/len, n.y/len, n.z/len) : Mesh::Normal(0,0,0);
}

void Mesh::ComputeNormalFaces()
{
	faceNormals.Resize(faces.GetSize());
	#ifndef _USE_CUDA
		#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)faces.size(); ++i)
				faceNormals[i] = SafeNormalizeFaceNormal(FaceNormal(faces[i]));
	#else
	if (kernelComputeFaceNormal.IsValid()) {
		reportCudaError(kernelComputeFaceNormal((int)faces.size(),
			vertices,
			faces,
			CUDA::KernelRT::OutputParam(faceNormals.GetDataSize()),
			faces.GetSize()
		));
		reportCudaError(kernelComputeFaceNormal.GetResult(0,
			faceNormals
		));
		kernelComputeFaceNormal.Reset();
		// The device kernel normalizes unguarded too, so scrub the same degenerate case
		// out of its result -- otherwise the CUDA and CPU paths disagree on exactly the
		// input that causes the damage (see SafeNormalizeFaceNormal).
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < (int)faceNormals.size(); ++i) {
			const Normal& n = faceNormals[i];
			if (!(std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z) > 0.f))
				faceNormals[i] = Normal(0,0,0);
		}
	} else {
		FOREACH(idxFace, faces)
			faceNormals[idxFace] = SafeNormalizeFaceNormal(FaceNormal(faces[idxFace]));
	}
	#endif
}

// compute normal for all vertices
#if 1
// computes the vertex normal as the area weighted face normals average
void Mesh::ComputeNormalVertices()
{
	vertexNormals.resize(vertices.size());
	vertexNormals.Memset(0);
	for (const Face& face: faces) {
		const Vertex& v0 = vertices[face[0]];
		const Vertex& v1 = vertices[face[1]];
		const Vertex& v2 = vertices[face[2]];
		const Normal t((v1 - v0).cross(v2 - v0));
		vertexNormals[face[0]] += t;
		vertexNormals[face[1]] += t;
		vertexNormals[face[2]] += t;
	}
	for (Normal& vertexNormal: vertexNormals)
		normalize(vertexNormal);
}
#else
// computes the vertex normal as an angle weighted average
// (the vertex first ring of faces and the face normals are used)
//
// The normal of a vertex v computed as a weighted sum f the incident face normals.
// The weight is simply the angle of the involved wedge. Described in:
// G. Thurmer, C. A. Wuthrich "Computing vertex normals from polygonal facets", Journal of Graphics Tools, 1998
void Mesh::ComputeNormalVertices()
{
	ASSERT(!faceNormals.IsEmpty());
	vertexNormals.Resize(vertices.GetSize());
	vertexNormals.Memset(0);
	FOREACH(idxFace, faces) {
		const Face& face = faces[idxFace];
		const Normal& t = faceNormals[idxFace];
		const Vertex& v0 = vertices[face[0]];
		const Vertex& v1 = vertices[face[1]];
		const Vertex& v2 = vertices[face[2]];
		const Normal e0(normalized(v1-v0));
		const Normal e1(normalized(v2-v1));
		const Normal e2(normalized(v0-v2));
		vertexNormals[face[0]] += t*ACOS(-ComputeAngleN(e0.ptr(), e2.ptr()));
		vertexNormals[face[1]] += t*ACOS(-ComputeAngleN(e0.ptr(), e1.ptr()));
		vertexNormals[face[2]] += t*ACOS(-ComputeAngleN(e1.ptr(), e2.ptr()));
	}
	for (Normal& vertexNormal: vertexNormals)
		normalize(vertexNormal);
}
#endif

// Smoothen the normals for each face
//  - fMaxGradient: maximum angle (in degrees) difference between neighbor normals that is
//    allowed to take into consideration; higher angles are ignored
//  - fOriginalWeight: weight (0..1] to use for current normal value when averaging with neighbor normals
//  - nIterations: number of times to repeat the smoothening process
#if 1
void Mesh::SmoothNormalFaces(float fMaxGradient,
	float fOriginalWeight,
	unsigned nIterations)
{
	if (faceNormals.size() != faces.size())
		ComputeNormalFaces();
	if (vertexFaces.size() != vertices.size())
		ListIncidenteFaces();
	if (faceFaces.size() != faces.size())
		ListIncidenteFaceFaces();

	const float cosMaxGradient = COS(FD2R(fMaxGradient));
	const float w0 = fOriginalWeight;
	const float w1 = 1.0f - w0;

	NormalArr newFaceNormals(faceNormals.size());

	for (unsigned rep = 0; rep < nIterations; ++rep) {

		FOREACH(idxFace, faces) {
			const Normal& orig = faceNormals[idxFace];

			Normal sum = Normal::ZERO;
			int count = 0;

			// at most 3 neighbors
			for (int i = 0; i < 3; ++i) {
				const FIndex fIdx = faceFaces[idxFace][i];
				if (fIdx == NO_ID)
					continue;

				const Normal& nb = faceNormals[fIdx];

				// dot product instead of ComputeAngleN
				const float dot =
					orig.x * nb.x + orig.y * nb.y + orig.z * nb.z;

				if (dot >= cosMaxGradient) {
					sum += nb;
					++count;
				}
			}

			Normal blended;

			if (count > 0) {
				// normalize neighbor sum once
				const float invLen =
					1.0f / sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);

				const Normal avg(sum.x * invLen,
					sum.y * invLen,
					sum.z * invLen);

				blended.x = orig.x * w0 + avg.x * w1;
				blended.y = orig.y * w0 + avg.y * w1;
				blended.z = orig.z * w0 + avg.z * w1;
			}
			else {
				// no valid neighbors
				blended = orig;
			}

			// final normalize (still required)
			const float invLen =
				1.0f / sqrt(blended.x * blended.x +
					blended.y * blended.y +
					blended.z * blended.z);

			newFaceNormals[idxFace].x = blended.x * invLen;
			newFaceNormals[idxFace].y = blended.y * invLen;
			newFaceNormals[idxFace].z = blended.z * invLen;
		}

		newFaceNormals.Swap(faceNormals);
	}
}
#else
void Mesh::SmoothNormalFaces(float fMaxGradient, float fOriginalWeight, unsigned nIterations) {
	if (faceNormals.size() != faces.size())
		ComputeNormalFaces();
	if (vertexFaces.size() != vertices.size())
		ListIncidenteFaces();
	if (faceFaces.size() != faces.size())
		ListIncidenteFaceFaces();
	const float cosMaxGradient = COS(FD2R(fMaxGradient));
	for (unsigned rep = 0; rep < nIterations; ++rep) {
		NormalArr newFaceNormals(faceNormals.size());
		FOREACH(idxFace, faces) {
			const Normal& originalNormal = faceNormals[idxFace];
			Normal sumNeighborNormals = Normal::ZERO;
			for (int i = 0; i < 3; ++i) {
				const FIndex fIdx = faceFaces[idxFace][i];
				if (fIdx == NO_ID)
					continue;
				const Normal& neighborNormal = faceNormals[fIdx];
				if (ComputeAngleN(originalNormal.ptr(), neighborNormal.ptr()) >= cosMaxGradient)
					sumNeighborNormals += neighborNormal;
			}
			const Normal avgNeighborsNormal = normalized(sumNeighborNormals);
			const Normal newFaceNormal = normalized(originalNormal * fOriginalWeight + avgNeighborsNormal * (1.f - fOriginalWeight));
			newFaceNormals[idxFace] = newFaceNormal;
		}
		newFaceNormals.Swap(faceNormals);
	}
}
#endif
/*----------------------------------------------------------------*/


void Mesh::GetEdgeFaces(VIndex v0, VIndex v1, FaceIdxArr& afaces) const
{
	const FaceIdxArr& faces0 = vertexFaces[v0];
	const FaceIdxArr& faces1 = vertexFaces[v1];
	std::unordered_set<FIndex> setFaces1(faces1.Begin(), faces1.End());
	FOREACH(i, faces0) {
		if (setFaces1.find(faces0[i]) != setFaces1.end())
			afaces.Insert(faces0[i]);
	}
}

void Mesh::GetFaceFaces(FIndex f, FaceIdxArr& afaces) const
{
	const Face& face = faces[f];
	const FaceIdxArr& faces0 = vertexFaces[face[0]];
	const FaceIdxArr& faces1 = vertexFaces[face[1]];
	const FaceIdxArr& faces2 = vertexFaces[face[2]];
	std::unordered_set<FIndex> setFaces(faces1.Begin(), faces1.End());
	FOREACHPTR(pIdxFace, faces0) {
		if (f != *pIdxFace && setFaces.find(*pIdxFace) != setFaces.end())
			afaces.InsertSortUnique(*pIdxFace);
	}
	FOREACHPTR(pIdxFace, faces2) {
		if (f != *pIdxFace && setFaces.find(*pIdxFace) != setFaces.end())
			afaces.InsertSortUnique(*pIdxFace);
	}
	setFaces.clear();
	setFaces.insert(faces2.Begin(), faces2.End());
	FOREACHPTR(pIdxFace, faces0) {
		if (f != *pIdxFace && setFaces.find(*pIdxFace) != setFaces.end())
			afaces.InsertSortUnique(*pIdxFace);
	}
}

void Mesh::GetEdgeVertices(FIndex f0, FIndex f1, uint32_t* vs0, uint32_t* vs1) const
{
	const Face& face0 = faces[f0];
	const Face& face1 = faces[f1];
	int i(0);
	for (int v=0; v<3; ++v) {
		if ((vs1[i] = FindVertex(face1, face0[v])) != NO_ID) {
			vs0[i] = v;
			if (++i == 2)
				return;
		}
	}
}

void Mesh::GetAdjVertices(VIndex v, boost::container::small_vector<VIndex, 32>& indices) const
{
#if 1
	ASSERT(vertexFaces.GetSize() == vertices.GetSize());
	const FaceIdxArr& idxFaces = vertexFaces[v];

	// Each thread gets its own visited array and stamp
	thread_local static std::vector<uint32_t> visited;
	thread_local static uint32_t stamp = 1;

	// Make sure visited array is large enough
	if (visited.size() < vertices.size()) {
		visited.resize(vertices.size(), 0);
	}

	// Handle stamp wraparound
	if (++stamp == 0) {
		std::fill(visited.begin(), visited.end(), 0);
		stamp = 1;
	}

	indices.clear();

	for (FIndex fi : idxFaces) {
		const Face& face = faces[fi];
		for (int i = 0; i < 3; ++i) {
			VIndex vAdj = face[i];
			if (vAdj == v) continue;

			if (visited[vAdj] != stamp) {
				visited[vAdj] = stamp;
				indices.push_back(vAdj);
			}
		}
	}
#else
	ASSERT(vertexFaces.size() == vertices.size());
	const FaceIdxArr& idxFaces = vertexFaces[v];
	std::unordered_set<VIndex> setIndices;
	for (FIndex idxFace : idxFaces) {
		const Face& face = faces[idxFace];
		for (int i = 0; i < 3; ++i) {
			const VIndex vAdj(face[i]);
			if (vAdj != v && setIndices.insert(vAdj).second)
				indices.emplace_back(vAdj);
		}
	}
#endif
}

void Mesh::GetAdjVertexFaces(VIndex idxVCenter, VIndex idxVAdj, FaceIdxArr& indices) const
{
	ASSERT(vertexFaces.GetSize() == vertices.GetSize());
	const FaceIdxArr& idxFaces = vertexFaces[idxVCenter];
	FOREACHPTR(pIdxFace, idxFaces) {
		const Face& face = faces[*pIdxFace];
		ASSERT(FindVertex(face, idxVCenter) != NO_ID);
		if (FindVertex(face, idxVAdj) != NO_ID)
			indices.Insert(*pIdxFace);
	}
}
/*----------------------------------------------------------------*/
#ifdef OPENMVS_21
// find the adjacent face for the given face edge;
// return NO_ID if no adjacent faces exist OR
// more than one adjacent face exist OR
// the edge have opposite orientations in each face
Mesh::FIndex Mesh::GetEdgeAdjacentFace(FIndex idxFace, VIndex v0, VIndex v1) const {
	ASSERT(vertexFaces.size() == vertices.size());

	const Face& baseFace = faces[idxFace];
	const bool baseOrientation = GetEdgeOrientation(baseFace, v0, v1);

	FIndex adjFaceIdx = NO_ID;

	// loop over faces incident to v0
	for (FIndex iF : vertexFaces[v0]) {
		if (iF == idxFace) continue; // skip self

		const Face& f = faces[iF];

		// search for v1 in this face
		for (int j = 0; j < 3; ++j) {
			if (f[j] == v1) {
				// found edge (v0,v1) in candidate face
				if (adjFaceIdx != NO_ID)
					return NO_ID; // more than one adjacent -> non-manifold

				const bool adjOrientation = GetEdgeOrientation(f, v0, v1);
				if (adjOrientation == baseOrientation)
					return NO_ID; // same orientation -> non-manifold

				adjFaceIdx = iF;
				break; // no need to check rest of vertices
			}
		}
	}

	return adjFaceIdx;
}
#if 0
unsigned Mesh::FixNonManifold(float magDisplacementDuplicateVertices,
	VertexIdxArr* duplicatedVertices)
{
	ASSERT(!vertices.empty() && !faces.empty());
	if (vertexFaces.size() != vertices.size())
		ListIncidenteFaces();

	const size_t nVerts = vertices.size();
	const size_t nFaces = faces.size();

	vertices.reserve(vertices.size() * 2);
	vertexFaces.reserve(vertices.size() * 2);

	struct ComponentRange {
		uint32_t begin;
		uint32_t count;
	};

	struct VertexWork {
		int numComponents = 0;
		std::vector<FIndex> facesFlat;
		std::vector<ComponentRange> components;
	};

	std::vector<VertexWork> work(nVerts);
	for (size_t i = 0; i < nVerts; ++i) {
		size_t deg = vertexFaces[i].size();
		work[i].facesFlat.reserve(deg);
		work[i].components.reserve(deg);
	}

	// -------- Phase 1: Parallel discovery --------
#pragma omp parallel
	{
		std::vector<int> faceTag(nFaces, -1);
		int myGen = 0;

		boost::container::small_vector<FIndex, 64> queue;

#pragma omp for schedule(static)
		for (ptrdiff_t idxVert = 0; idxVert < (ptrdiff_t)nVerts; ++idxVert) {
			++myGen;

			VertexWork& out = work[idxVert];
			out.facesFlat.clear();
			out.components.clear();
			out.numComponents = 0;

			const FaceIdxArr& vertFaces = vertexFaces[idxVert];
			if (vertFaces.empty())
				continue;

			int component = 0;

			for (size_t it = 0; it < vertFaces.size(); ++it) {
				FIndex seed = vertFaces[it];
				if (faceTag[seed] == myGen)
					continue;

				queue.clear();
				queue.push_back(seed);
				faceTag[seed] = myGen;

				ComponentRange range;
				range.begin = (uint32_t)out.facesFlat.size();

				// ---- BFS (IDENTICAL traversal) ----
				while (!queue.empty()) {
					FIndex curF = queue.back();
					queue.pop_back();
					out.facesFlat.push_back(curF);

					const Face& face = faces[curF];
					for (int i = 0; i < 3; ++i) {
						VIndex vAdj = face[i];
						if (vAdj == (VIndex)idxVert)
							continue;

						FIndex fAdj = GetEdgeAdjacentFace(curF, idxVert, vAdj);
						if (fAdj != NO_ID && faceTag[fAdj] != myGen) {
							faceTag[fAdj] = myGen;
							queue.push_back(fAdj);
						}
					}
				}

				range.count =
					(uint32_t)out.facesFlat.size() - range.begin;
				out.components.push_back(range);
				++component;
			}

			out.numComponents = component;
		}
	}

	// -------- Phase 2: Apply changes (UNCHANGED) --------
	unsigned numIssues = 0;

	for (size_t idxVert = 0; idxVert < nVerts; ++idxVert) {
		VertexWork& vw = work[idxVert];
		if (vw.numComponents <= 1)
			continue;

		for (int c = 1; c < vw.numComponents; ++c) {
			const ComponentRange& r = vw.components[c];

			const VIndex idxVertNew = vertices.size();
			vertices.emplace_back(vertices[idxVert]);

			if (duplicatedVertices)
				duplicatedVertices->emplace_back(idxVert);

			FaceIdxArr& vertFacesNew = vertexFaces.emplace_back();
			vertFacesNew.reserve(r.count);

			const uint32_t end = r.begin + r.count;
			for (uint32_t i = r.begin; i < end; ++i) {
				FIndex fidx = vw.facesFlat[i];
				Face& f = faces[fidx];
				for (int k = 0; k < 3; ++k) {
					if (f[k] == idxVert) {
						f[k] = idxVertNew;
						vertFacesNew.push_back(fidx);
						break;
					}
				}
			}

			++numIssues;
		}

		// Optional displacement (UNCHANGED)
		if (magDisplacementDuplicateVertices > 0) {
			boost::container::small_vector<VIndex, 32> adjVerts;
			boost::container::small_vector<VIndex, 3> verts(vw.numComponents);

			verts[0] = idxVert;
			for (int c = 1; c < vw.numComponents; ++c)
				verts[c] = vertices.size() - (vw.numComponents - c);

			for (VIndex vIdx : verts) {
				adjVerts.clear();
				GetAdjVertices(vIdx, adjVerts);
				TAccumulator<Vertex> accum;
				for (VIndex iV : adjVerts)
					accum.Add(vertices[iV], 1.f);
				const Vertex bv(accum.Normalized());
				Vertex& v = vertices[vIdx];
				v += (bv - v) * magDisplacementDuplicateVertices;
			}
		}
	}

	if (numIssues > 0) {
		vertexFaces.Release();
		DEBUG_ULTIMATE("Removed %u non-manifold issues", numIssues);
	}

	return numIssues;
}
#else

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

unsigned Mesh::FixNonManifold(float magDisplacementDuplicateVertices, VertexIdxArr* duplicatedVertices)
{
	ASSERT(!vertices.empty() && !faces.empty());
	if (vertexFaces.size() != vertices.size())
		ListIncidenteFaces();

	// iterate over all vertices and separates the components
	// incident to the same vertex by duplicating the vertex
	unsigned numNonManifoldIssues(0);
	CLISTDEF0IDX(int, FIndex) components(faces.size());
	FaceIdxArr queueFaces;
	boost::container::small_vector<VIndex, 32> adjVerts;
	FOREACH(idxVert, vertices) {
		// reset component indices to which each face connected to this vertex
		const FaceIdxArr& vertFaces = vertexFaces[idxVert];
		for (FIndex iF: vertFaces)
			components[iF] = -1;
		// find the components connected to this vertex
		queueFaces.clear();
		queueFaces.reserve(vertFaces.size());
		FIndex idxFaceNext(0);
		int component(0);
		for ( ; ; ++component) {
			// find one face not yet belonging to a component
			while (idxFaceNext < vertFaces.size()) {
				const FIndex iF(vertFaces[idxFaceNext++]);
				if (components[iF] == -1) {
					// add component as seed to the list
					queueFaces.push_back(iF);
					// mark the current face with a new component
					components[iF] = component;
					// process component
					goto ProcessComponent;
		}
				}
			// no more components found
			break;
		ProcessComponent:
			// grow seed face component until no more connected faces found
			do {
				const FIndex idxFaceCurrent(queueFaces.back());
				queueFaces.pop_back();
				const Face& face = faces[idxFaceCurrent];
				// go over all vertices of the current face
				for (int i = 0; i < 3; ++i) {
					const VIndex idxVertAdj(face[i]);
					if (idxVertAdj == idxVert)
						continue;
					// if there is exactly one face adjacent to this edge
					// tag it with the current component and add it to the queue
					const FIndex idxFaceAdj(GetEdgeAdjacentFace(idxFaceCurrent, idxVert, idxVertAdj));
					if (idxFaceAdj != NO_ID && components[idxFaceAdj] == -1) {
						components[idxFaceAdj] = component;
						queueFaces.push_back(idxFaceAdj);
			}
			}
			} while (!queueFaces.empty());
		}
		// if there is only one component, continue with the next vertex
		if (component <= 1)
			continue;
		// separate the vertex components
		for (int c = 1; c < component; ++c) {
			// duplicate the point to achieve the separation
			const VIndex idxVertNew = vertices.size();
			const Vertex v = vertices[idxVert];
			vertices.emplace_back(v);
			if (duplicatedVertices)
				duplicatedVertices->emplace_back(idxVert);
			// update the face indices of the current component
			vertexFaces.reserve(vertexFaces.size() + 1);
			FaceIdxArr& vertFacesNew = vertexFaces.emplace_back();
			// Re-fetch after potential reallocation
			FaceIdxArr& vertFaces = vertexFaces[idxVert];
			RFOREACH(ivf, vertFaces) {
				const FIndex idxFace = vertFaces[ivf];
				if (components[idxFace] != c)
					continue;
				// link face to the new vertex and remove it from the original vertex
				Face& face = faces[idxFace];
				for (int i = 0; i < 3; ++i) {
					if (face[i] == idxVert) {
						face[i] = idxVertNew;
						vertFacesNew.push_back(idxFace);
						break;
					}
				}
				vertFaces.RemoveAtMove(ivf);
			}
			// reverse to restore original InsertAt(0,...) order
			std::reverse(vertFacesNew.begin(), vertFacesNew.end());
			++numNonManifoldIssues;
		}
		// adjust vertex positions
		if (magDisplacementDuplicateVertices > 0) {
			// list changed vertices
			boost::container::small_vector<VIndex, 3> verts(component);
			verts[0] = idxVert;
			for (int c = 1; c < component; ++c)
				verts[c] = vertices.size()-(component-c);
			// adjust the position of the vertices in the direction
			// to the center of the first ring of faces
			FOREACH(i, verts) {
				const VIndex idxVert(verts[i]);
			adjVerts.clear();
				GetAdjVertices(idxVert, adjVerts);
				TAccumulator<Vertex> accum;
				for (VIndex iV: adjVerts)
					accum.Add(vertices[iV], 1.f);
				const Vertex bv(accum.Normalized());
				Vertex& v(vertices[idxVert]);
				const Vertex dir(bv-v);
				v += dir * magDisplacementDuplicateVertices;
			}
		}
	}

#if 0 // Validate non-manifold
	for (FIndex f = 0; f < faces.size(); ++f)
	{
		const Face& face = faces[f];

		for (int i = 0; i < 3; ++i)
		{
			HARD_ASSERT(face[i] < vertices.size());
		}

		// detect degenerate triangle
		HARD_ASSERT(face[0] != face[1]);
		HARD_ASSERT(face[1] != face[2]);
		HARD_ASSERT(face[2] != face[0]);
	}

	ListIncidenteFaceFaces(); // JPB WIP BUG
	ValidateVertexFacesSorted();
	ValidateFaceFaces();
#endif

	if (numNonManifoldIssues > 0) {
		vertexFaces.Release();
		DEBUG_ULTIMATE("Removed %u non-manifold issues", numNonManifoldIssues);
	}

	return numNonManifoldIssues;
}
#endif
#endif // 2.1 version
/*----------------------------------------------------------------*/

namespace CLEAN {
	// define mesh type
	class Vertex; class Edge; class Face;
	struct UsedTypes : public vcg::UsedTypes<
		vcg::Use<Vertex>::AsVertexType,
		vcg::Use<Edge>  ::AsEdgeType,
		vcg::Use<Face>  ::AsFaceType   > {
	};

	class Vertex : public vcg::Vertex<UsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::VFAdj, vcg::vertex::Mark, vcg::vertex::BitFlags> {};
	class Face : public vcg::Face<  UsedTypes, vcg::face::VertexRef, vcg::face::Normal3f, vcg::face::FFAdj, vcg::face::VFAdj, vcg::face::Mark, vcg::face::BitFlags> {};
	class Edge : public vcg::Edge<  UsedTypes, vcg::edge::VertexRef, vcg::edge::Mark, vcg::edge::BitFlags> {};

	class Mesh : public vcg::tri::TriMesh< std::vector<Vertex>, std::vector<Face>, std::vector<Edge> > {
	public:
		bool hasDeletedFaces = false;
	};

	// decimation helper classes
	typedef	vcg::SimpleTempData< Mesh::VertContainer, vcg::math::Quadric<double> > QuadricTemp;

	class QHelper
	{
	public:
		QHelper() {}
		static void Init() {}
		static vcg::math::Quadric<double>& Qd(const Vertex& v) { return TD()[v]; }
		static vcg::math::Quadric<double>& Qd(const Vertex* v) { return TD()[*v]; }
		static Vertex::ScalarType W(Vertex* /*v*/) { return 1.0; }
		static Vertex::ScalarType W(Vertex& /*v*/) { return 1.0; }
		static void Merge(Vertex& /*v_dest*/, Vertex const& /*v_del*/) {}
		static QuadricTemp*& TDp() { static QuadricTemp* td; return td; }
		static QuadricTemp& TD() { return *TDp(); }
	};

	typedef vcg::tri::BasicVertexPair<Vertex> VertexPair;

	class TriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<Mesh, VertexPair, TriEdgeCollapse, QHelper>
	{
	public:
		typedef vcg::tri::TriEdgeCollapseQuadric<Mesh, VertexPair, TriEdgeCollapse, QHelper> TECQ;
		inline TriEdgeCollapse(const VertexPair& p, int i) :TECQ(p, i) {}

		// Custom allocator
		static __forceinline void* operator new(std::size_t size)
		{
			// Operator new always needs to give you a real pointer to the memory where the object is stored.
			if (g_qOffset == BLOCK_SIZE) {
				g_qBlocks.push_back(::operator new(BLOCK_SIZE * size));
				// The very first block is marker for "nullptr" and is unused.
				g_qOffset = (1 == g_qBlocks.size()) ? 1 : 0;
			}


			//const uint64_t value = ((g_qBlocks.size() - 1) << (8 + 8 + 7)) + (g_qOffset++);
			//return reinterpret_cast<void*>(static_cast<uintptr_t>(value));
			return (uint8_t*)g_qBlocks.back() + size * g_qOffset++;
		}

		static void Release()
		{
			for (void* block : g_qBlocks)
			{
				::operator delete(block, std::nothrow);
			}
			g_qBlocks.clear();
			g_qOffset = 0;
		}
	};
};

#if 0 // JPB WIP
#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <algorithm>

class ScratchPad {
public:
	explicit ScratchPad(size_t lanes = 1, size_t initialBytes = 0) {
		resizeLanes(lanes, initialBytes);
	}

	void resizeLanes(size_t lanes, size_t initialBytes = 0) {
		lanes_.resize(lanes);
		for (size_t i = 0; i < lanes_.size(); ++i) {
			if (initialBytes > 0 && lanes_[i].buffer.capacity() < initialBytes) {
				lanes_[i].buffer.reserve(initialBytes);
			}
			lanes_[i].cursor = 0;
		}
	}

	void clearAll(bool shrink = false) {
		for (auto& ln : lanes_) {
			ln.buffer.clear();
			if (shrink) {
				ln.buffer.shrink_to_fit();
			}
			ln.cursor = 0;
		}
	}

	void resetAll() {
		for (auto& ln : lanes_) {
			ln.cursor = 0;
		}
	}

	void reset(size_t lane) {
		lanes_.at(lane).cursor = 0;
	}

	void reserve(size_t lane, size_t bytes) {
		lanes_.at(lane).buffer.reserve(bytes);
	}

	size_t size(size_t lane) const {
		return lanes_.at(lane).cursor;
	}

	size_t capacity(size_t lane) const {
		return lanes_.at(lane).buffer.capacity();
	}

	size_t remaining(size_t lane) const {
		const auto& ln = lanes_.at(lane);
		return ln.buffer.capacity() >= ln.cursor ? (ln.buffer.capacity() - ln.cursor) : 0;
	}

	void* allocateRaw(size_t lane, size_t bytes, size_t alignment = alignof(std::max_align_t)) {
		Lane& ln = lanes_.at(lane);
		size_t aligned = alignUp(ln.cursor, alignment);
		ensure(lane, aligned + bytes, alignment);
		ln.cursor = aligned;
		void* ptr = ln.ptr() + ln.cursor;
		ln.cursor += bytes;
		return ptr;
	}

	template<typename T>
	T* allocate(size_t lane, size_t count = 1) {
		return static_cast<T*>(allocateRaw(lane, sizeof(T) * count, alignof(T)));
	}

	template<typename T>
	T* allocateCleared(size_t lane, size_t count = 1) {
		T* p = allocate<T>(lane, count);
		std::byte* b = reinterpret_cast<std::byte*>(p);
		std::fill(b, b + sizeof(T) * count, std::byte{ 0 });
		return p;
	}

	template<typename T>
	T* emplaceDefault(size_t lane, size_t count = 1) {
		static_assert(std::is_default_constructible<T>::value, "T must be default constructible");
		T* p = allocate<T>(lane, count);
		for (size_t i = 0; i < count; ++i) {
			new (p + i) T();
		}
		return p;
	}

	template<typename T>
	T* castAt(size_t lane, size_t byteOffset) {
		Lane& ln = lanes_.at(lane);
		return reinterpret_cast<T*>(ln.ptr() + byteOffset);
	}

	// Access the underlying vector-of-vectors if needed.
	const std::vector<unsigned char>& laneBuffer(size_t lane) const {
		return lanes_.at(lane).buffer;
	}

private:
	struct Lane {
		std::vector<unsigned char> buffer;
		size_t cursor = 0;

		unsigned char* ptr() {
			if (buffer.empty()) return nullptr;
			return buffer.data();
		}
	};

	static size_t alignUp(size_t x, size_t a) {
		size_t mask = a - 1;
		return (x + mask) & ~mask;
	}

	void ensure(size_t lane, size_t neededCursor, size_t alignment) {
		Lane& ln = lanes_.at(lane);
		size_t needBytes = neededCursor;
		if (ln.buffer.capacity() < needBytes) {
			size_t newCap = std::max(needBytes, std::max<size_t>(64, ln.buffer.capacity() * 2));
			// Keep alignment-friendly capacity growth (round up to alignment).
			newCap = alignUp(newCap, alignment);
			std::vector<unsigned char> tmp;
			tmp.reserve(newCap);
			if (!ln.buffer.empty()) {
				tmp.insert(tmp.end(), ln.buffer.begin(), ln.buffer.begin() + ln.cursor);
			}
			else if (ln.cursor > 0) {
				tmp.resize(ln.cursor);
			}
			ln.buffer.swap(tmp);
			if (ln.buffer.size() < ln.cursor) {
				ln.buffer.resize(ln.cursor);
			}
		}
		if (ln.buffer.size() < needBytes) {
			ln.buffer.resize(needBytes);
		}
	}

	std::vector<Lane> lanes_;
};

#endif

struct CleanStats
{
	int removedFaces = 0;
	int removedVerts = 0;
	int removedSpikes = 0;
	int removedComponents = 0;
	int removedLongEdgeFaces = 0;
	int closedHoles = 0;
	int removedNonManifoldFaces = 0;
};

static void KillEdges(CLEAN::Mesh& mesh) {
	for (auto& e : mesh.edge)
		vcg::tri::Allocator<CLEAN::Mesh>::DeleteEdge(mesh, e);
	vcg::tri::Allocator<CLEAN::Mesh>::CompactEdgeVector(mesh);
}

// Helper: restore consistency after deletions
inline void RestoreConsistency(CLEAN::Mesh& mesh)
{
	vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
	KillEdges(mesh);
	vcg::tri::Allocator<CLEAN::Mesh>::CompactEveryVector(mesh);
	vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
	vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
}

// Count faces with area smaller than epsilon
static int CountZeroAreaFaces(const CLEAN::Mesh& mesh, CLEAN::Mesh::ScalarType eps = 1e-12)
{
	int n = 0;
	const auto* vBegin = mesh.vert.empty() ? nullptr : &*mesh.vert.begin();
	const auto* vEnd = vBegin ? vBegin + mesh.vert.size() : nullptr;

	for (size_t i = 0; i < mesh.face.size(); ++i) {
		const auto& f = mesh.face[i];
		if (f.IsD()) continue;

		auto* v0 = f.V(0);
		auto* v1 = f.V(1);
		auto* v2 = f.V(2);
		if (!v0 || !v1 || !v2) continue;
		if (v0 < vBegin || v0 >= vEnd) continue;
		if (v1 < vBegin || v1 >= vEnd) continue;
		if (v2 < vBegin || v2 >= vEnd) continue;
		if (v0->IsD() || v1->IsD() || v2->IsD()) continue;

		const auto& p0 = v0->cP();
		const auto& p1 = v1->cP();
		const auto& p2 = v2->cP();
		CLEAN::Mesh::ScalarType area2 = ((p1 - p0) ^ (p2 - p0)).Norm();

		if (area2 <= eps) ++n;
	}
	return n;
}

#ifdef VALIDATE_MESH
template<class MeshType>
bool ValidateMesh(const MeshType& m, const char* stage, bool checkEdges = false)
{
	bool ok = true;

	const auto* vBegin = m.vert.empty() ? nullptr : &*m.vert.begin();
	const auto* vEnd = vBegin ? vBegin + m.vert.size() : nullptr;

	// Faces
	for (size_t i = 0; i < m.face.size(); ++i) {
		const auto& f = m.face[i];
		if (f.IsD()) continue;
		for (int j = 0; j < f.VN(); ++j) {
			auto* v = f.V(j);
			if (!v) {
				DEBUG("[%s] Face %zu has null vertex pointer\n", stage, i);
				ok = false; continue;
			}
			if (v < vBegin || v >= vEnd) {
				DEBUG("[%s] Face %zu points outside vertex array (%p)\n",
					stage, i, (void*)v);
				ok = false;
			}
			else if (v->IsD()) {
				DEBUG("[%s] Face %zu points to deleted vertex at idx %zd\n",
					stage, i, v - vBegin);
				ok = false;
			}
		}
	}

	if (checkEdges) {
		// Edges
		const auto* eBegin = m.edge.empty() ? nullptr : &*m.edge.begin();
		const auto* eEnd = eBegin ? eBegin + m.edge.size() : nullptr;

		for (size_t i = 0; i < m.edge.size(); ++i) {
			const auto& e = m.edge[i];
			if (e.IsD()) continue;
			for (int j = 0; j < 2; ++j) {
				auto* v = e.V(j);
				if (!v) {
					DEBUG("[%s] Edge %zu has null vertex pointer\n", stage, i);
					ok = false; continue;
				}
				if (v < vBegin || v >= vEnd) {
					DEBUG("[%s] Edge %zu points outside vertex array (%p)\n",
						stage, i, (void*)v);
					ok = false;
				}
				else if (v->IsD()) {
					DEBUG("[%s] Edge %zu points to deleted vertex at idx %zd\n",
						stage, i, v - vBegin);
					ok = false;
				}
			}
		}
	}

	return ok;
}
#else
template<class MeshType>
bool ValidateMesh(const MeshType&, const char*, bool checkEdges = true) { return true; }
#endif

inline void LightCleanup(CLEAN::Mesh& mesh, CleanStats& clean)
{
	int removedUnref = vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
	clean.removedVerts += removedUnref;

	KillEdges(mesh);

	vcg::tri::Allocator<CLEAN::Mesh>::CompactVertexVector(mesh);
	vcg::tri::Allocator<CLEAN::Mesh>::CompactFaceVector(mesh);
	vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
	vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
}

static inline void FastClean(CLEAN::Mesh& m, CleanStats& clean)
{
	using Tri = vcg::tri::Clean<CLEAN::Mesh>;
	using Topo = vcg::tri::UpdateTopology<CLEAN::Mesh>;
	using Alloc = vcg::tri::Allocator<CLEAN::Mesh>;

	// 1. Ensure adjacency is valid before manifold checks
	Topo::FaceFace(m);
	Topo::VertexFace(m);

	int removedNMf = Tri::RemoveNonManifoldFace(m);
	int removedUnref = Tri::RemoveUnreferencedVertex(m);
	clean.removedVerts += removedUnref;

	// 4. Canonicalize once
	if (removedNMf > 0 || removedUnref > 0) {
		KillEdges(m);
		Alloc::CompactEveryVector(m); // Will destroy adjacency information

		// 5. Final topology rebuild (post-compaction)
		Topo::FaceFace(m);
		Topo::VertexFace(m);
	}
}

float ComputeMedianEdgeLength(const CLEAN::Mesh& mesh)
{
	std::vector<float> edgeLens2;
	edgeLens2.reserve(mesh.fn * 3);

	for (const auto& f : mesh.face) {
		if (f.IsD()) continue;

		for (int i = 0; i < 3; ++i) {
			const auto* v0 = f.V(i);
			const auto* v1 = f.V((i + 1) % 3);
			const float d2 = (v1->cP() - v0->cP()).SquaredNorm();
			if (d2 > 0) // skip degenerate edges
				edgeLens2.push_back(d2);
		}
	}

	if (edgeLens2.empty())
		return 0.0f;

	const size_t mid = edgeLens2.size() / 2;
	std::nth_element(edgeLens2.begin(),
		edgeLens2.begin() + mid,
		edgeLens2.end());

	return std::sqrt(edgeLens2[mid]);
}

static void RemoveSpikes(CLEAN::Mesh& mesh, CleanStats& stats)
{
	int nTotalSpikes = 0;

	if (mesh.fn == 0 || mesh.vn == 0) {
		DEBUG_ULTIMATE("Removed %d spikes", nTotalSpikes);
		return;
	}

	const int numVerts = (int)mesh.vert.size();
	std::vector<int> valence(numVerts, 0);
	std::vector<uint8_t> alive(numVerts, 0);

	for (int i = 0; i < numVerts; ++i)
		alive[i] = mesh.vert[i].IsD() ? 0u : 1u;

	auto vpIndex = [&](CLEAN::Mesh::VertexPointer vp) -> int {
		return int(vp - &mesh.vert[0]);
		};

	// Pass 1: count valence
	size_t numFaceRefs = 0;
	for (auto& f : mesh.face) {
		if (f.IsD()) continue;
		auto* v0 = f.V(0);
		auto* v1 = f.V(1);
		auto* v2 = f.V(2);
		const int i0 = vpIndex(v0);
		const int i1 = vpIndex(v1);
		const int i2 = vpIndex(v2);

		if (!(alive[i0] & alive[i1] & alive[i2])) continue;

		valence[i0]++; valence[i1]++; valence[i2]++;
		numFaceRefs += 3;
	}

	// CSR offsets
	std::vector<uint32_t> incOffsets(numVerts + 1);
	uint64_t run = 0;
	for (int i = 0; i < numVerts; ++i) {
		incOffsets[i] = (uint32_t)run;
		run += (uint32_t)valence[i];
	}
	incOffsets[numVerts] = (uint32_t)run;

	// Pass 2: fill flat incident-face array
	std::vector<CLEAN::Mesh::FacePointer> incFacesFlat(numFaceRefs);
	std::vector<uint32_t> cursor = incOffsets;
	for (auto& f : mesh.face) {
		if (f.IsD()) continue;
		auto* v0 = f.V(0);
		auto* v1 = f.V(1);
		auto* v2 = f.V(2);
		const int i0 = vpIndex(v0);
		const int i1 = vpIndex(v1);
		const int i2 = vpIndex(v2);
		if (!(alive[i0] & alive[i1] & alive[i2])) continue;
		incFacesFlat[cursor[i0]++] = &f;
		incFacesFlat[cursor[i1]++] = &f;
		incFacesFlat[cursor[i2]++] = &f;
	}

	auto facesBegin = [&](int vi) { return incFacesFlat.data() + incOffsets[vi]; };
	auto facesEnd = [&](int vi) { return incFacesFlat.data() + incOffsets[vi + 1]; };

	// Seed queue with spikes
	std::vector<int> q;
	q.reserve(numVerts);
	for (int i = 0; i < numVerts; ++i)
		if (alive[i] && valence[i] == 1)
			q.push_back(i);

	// Spike pruning (with cascading)
	size_t head = 0;
	while (head < q.size()) {
		const int vi = q[head++];
		if ((unsigned)vi >= (unsigned)numVerts || !alive[vi] || valence[vi] != 1)
			continue;

		// Remove all incident faces and update neighbor valences
		for (auto p = facesBegin(vi), e = facesEnd(vi); p != e; ++p) {
			CLEAN::Mesh::FacePointer f = *p;
			if (!f || f->IsD())
				continue;

			// Decrement valence of the other vertices in this face
			for (int k = 0; k < 3; ++k) {
				const int vj = vpIndex(f->V(k));
				if (vj != vi && alive[vj]) {
					const int newVal = --valence[vj];
					if (newVal == 1) {
						q.push_back(vj); // newly created spike
					}
				}
			}

			vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *f);
		}

		// Delete the spike vertex itself
		vcg::tri::Allocator<CLEAN::Mesh>::DeleteVertex(mesh, mesh.vert[vi]);
		alive[vi] = 0;
		valence[vi] = 0;
		++nTotalSpikes;
	}

	int removedUnref = vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
	stats.removedVerts += removedUnref;

	KillEdges(mesh);
	vcg::tri::Allocator<CLEAN::Mesh>::CompactVertexVector(mesh);
	vcg::tri::Allocator<CLEAN::Mesh>::CompactFaceVector(mesh);
	vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
	vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);

  stats.removedSpikes = nTotalSpikes;
}

void Mesh::Clean(
	float fDecimate, float fSpurious, bool bRemoveSpikes,
	unsigned nCloseHoles, unsigned nSmooth, float fEdgeLength, bool bLastClean,
	float fDecimateError)
{
	if (vertices.IsEmpty() || faces.IsEmpty())
		return;

	CleanStats stats;

	TD_TIMER_STARTD();

	CLEAN::Mesh mesh;
	{
		// Headroom for the phases that ADD elements. Reserving 2x unconditionally
		// doubled the VCG footprint (48 B/vertex + 112 B/face = ~136 B/face on a
		// triangle mesh) even on the common path where nothing grows -- on a
		// 30M-face mesh that is ~4 GB of capacity that is never touched.
		// Growth comes from remeshing (fEdgeLength>0, which also duplicates the
		// whole mesh via MeshCopy), from hole closing, and from the band refine
		// (MESH_BAND_REFINE_ENABLED -> vcg::tri::RefineMidpoint, which subdivides);
		// decimation only shrinks. Anything that does exceed the reserve still
		// grows CORRECTLY through vcg::tri::Allocator, which fixes up the
		// VF/FF/VertexRef pointers on reallocation -- it just costs one
		// realloc+copy, which is why the growing paths keep real headroom.
		// The 1.5 is not calibrated: log the peak VN()/FN() against these
		// reserves before tightening it further.
		const float fGrowth(fEdgeLength > 0.f ? 2.f :
			((nCloseHoles > 0 || MESH_BAND_REFINE_ENABLED) ? 1.5f : 1.05f));
		mesh.vert.reserve((size_t)(vertices.size() * fGrowth));

		CLEAN::Mesh::VertexIterator vi = vcg::tri::Allocator<CLEAN::Mesh>::AddVertices(mesh, vertices.GetSize());
		FOREACHPTR(pVert, vertices) {
			const Vertex& p(*pVert);
			CLEAN::Vertex::CoordType& P((*vi).P());
			P[0] = p.x;
			P[1] = p.y;
			P[2] = p.z;
			++vi;
		}
		vertices.Release();
		vi = mesh.vert.begin();
		std::vector<CLEAN::Mesh::VertexPointer> indices(mesh.vert.size());
		for (CLEAN::Mesh::VertexPointer& idx : indices) {
			idx = &*vi;
			++vi;
		}

		mesh.face.reserve((size_t)(faces.size() * fGrowth));

		CLEAN::Mesh::FaceIterator fi = vcg::tri::Allocator<CLEAN::Mesh>::AddFaces(mesh, faces.GetSize());
		FOREACHPTR(pFace, faces) {
			const Face& f(*pFace);
			ASSERT((*fi).VN() == 3);
			ASSERT(f[0] < (uint32_t)mesh.vn);
			(*fi).V(0) = indices[f[0]];
			ASSERT(f[1] < (uint32_t)mesh.vn);
			(*fi).V(1) = indices[f[1]];
			ASSERT(f[2] < (uint32_t)mesh.vn);
			(*fi).V(2) = indices[f[2]];
			++fi;
		}
		faces.Release();
	}

	constexpr CLEAN::Mesh::ScalarType eps = 1e-12;

	// Compact-only helper: removes unreferenced verts and (if any were removed)
	// kills dangling edges and shrinks the vert/face containers. Does NOT
	// rebuild FF/VF topology. Use this when the next code block will rebuild
	// topology itself, so we don't pay for two rebuilds back-to-back.
	auto Compact = [&]() {
		int removedUnref = vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
		stats.removedVerts += removedUnref;
		if (removedUnref > 0) {
			KillEdges(mesh);
			vcg::tri::Allocator<CLEAN::Mesh>::CompactVertexVector(mesh);
			vcg::tri::Allocator<CLEAN::Mesh>::CompactFaceVector(mesh);
		}
	};

	auto CompactAndRefresh = [&]() {
		Compact();
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
	};

	auto LightRefresh = [&]() {
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
	};

	ValidateMesh(mesh, "start", true);

	// NOTE: Decimation is intentionally deferred until AFTER spurious-face
	// and spike removal (see below). Running quadric edge-collapse on the
	// raw graph-cut surface wastes collapses fixing artifacts (long-edge
	// faces, inverted sheets, spikes) instead of preserving real features
	// like roof edges, eaves, and curbs.

	// Phase 2: topology repair.
	//
	// Only PASS 1 (long-edge removal) is spurious-specific and keyed on
	// fSpurious. The three passes after it -- orientation coherence, flipped-sheet
	// removal, and small-connected-component removal -- are general topology
	// repair that every reconstruction path needs, so they run unconditionally.
	//
	// WHY THIS SPLIT EXISTS: ReconstructMesh passes fSpurious=0 under --poisson on
	// purpose (long-edge removal eats the Poisson edge extrapolation, and the
	// surface is already coherently oriented in principle). But all four passes
	// used to sit inside one `if (fSpurious > 0)`, so that single decision also
	// switched off:
	//   - OrientCoherentlyMesh / FlipNormalOutside, which the down-cull documents
	//     as its precondition ("coherent outward orientation, set by the
	//     spurious-removal pass") -- so that precondition was never met;
	//   - the flipped-normal sheet removal;
	//   - the small-component filter, which is the ONLY island removal that runs
	//     BEFORE decimation and hole-closing. Deferred to Phase 9.1, islands first
	//     consume decimation budget and then get SEALED WATERTIGHT by hole-closing,
	//     which is why Poisson meshes showed solid floating blobs rather than
	//     scraps -- and why no `DIAG component sizes` line ever appeared on that
	//     path to diagnose them with.
	// None of that has anything to do with long edges; it was incidental nesting.
	{
		if (fSpurious > 0) {
			vcg::tri::UpdateTopology<CLEAN::Mesh>::AllocateEdge(mesh);

			FloatArr edgeLens(0, mesh.EN());
			for (auto& e : mesh.edge) {
				const auto& P0 = e.V(0)->cP();
				const auto& P1 = e.V(1)->cP();
				edgeLens.Insert((P1 - P0).SquaredNorm());
			}

			// JPB: use 98th percentile (was 95th) so the threshold is anchored to
			// truly extreme edges -- bbox-bridging sheets sit far in the tail,
			// while merely-long boundary triangles fall safely below.
			const float longEdge =
				sqrtf(edgeLens.GetNth(edgeLens.size() * 98 / 100)) * fSpurious;

			// Pass 1: Remove faces with long edges (original spurious removal)
			// Build FaceFace topology so we can identify boundary faces.
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateSelection<CLEAN::Mesh>::Clear(mesh);
			vcg::tri::UpdateSelection<CLEAN::Mesh>::FaceOutOfRangeEdge(mesh, 0, longEdge);

			int removed = 0;
			for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && fi->IsS()) {
					vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fi);
					++removed;
				}
			}
			stats.removedLongEdgeFaces += removed;

			// TIER 1: Pass 2 (below) immediately rebuilds FaceFace topology, so we
			// only need to compact here -- the FF/VF rebuild would be wasted.
			Compact();

			DEBUG("DIAG after long-edge removal: %d vn, %d fn (removed %d)",
				mesh.vn, mesh.fn, removed);
		}

		// Pass 2: Fix normal orientation globally.
		// The mesh from reconstruction should be predominantly correctly oriented.
		// OrientCoherentlyMesh propagates orientation from each component's seed face.
		// Then FlipNormalOutside uses a voting scheme to ensure outward orientation.
		// After that, any remaining faces that are incoherently oriented (the backwards
		// sheets that couldn't be made coherent) are detected and removed.
		{
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);

			// First, try to make all faces in each connected component coherent.
			bool isOriented = false, isOrientable = false;
			vcg::tri::Clean<CLEAN::Mesh>::OrientCoherentlyMesh(mesh, isOriented, isOrientable);

			if (!isOriented) {
				// Some faces had to be flipped to achieve coherent orientation.
				// Now ensure the dominant orientation faces outward.
				vcg::tri::UpdateNormal<CLEAN::Mesh>::PerVertexAngleWeighted(mesh);
				vcg::tri::UpdateNormal<CLEAN::Mesh>::NormalizePerVertex(mesh);
				vcg::tri::Clean<CLEAN::Mesh>::FlipNormalOutside(mesh);

				// TIER 1: Pass 3 below immediately rebuilds FaceFace, so we only
				// need to compact here -- the FF/VF rebuild would be wasted.
				Compact();
				DEBUG("DIAG after orientation fix: %d vn, %d fn (was%s oriented, %sorientable)",
					mesh.vn, mesh.fn,
					isOriented ? "" : " not",
					isOrientable ? "" : "not ");
			}

			// NOTE: We intentionally do NOT call RemoveNonManifoldFace here even
			// if !isOrientable. RemoveNonManifoldFace sorts candidate faces by
			// area and removes the smaller ones at each non-manifold edge. On thin
			// reconstructed surfaces the "real" triangles can be smaller than the
			// spurious backwards-sheet triangles, causing the wrong faces to be
			// deleted and creating the small holes visible in the wireframe.
			//
			// Instead, we rely on the small connected component removal below to
			// clean up any disconnected backwards patches that survived orientation
			// fixing. This is both safer (no holes) and more effective (removes
			// entire spurious patches rather than individual faces).
			if (!isOrientable) {
				DEBUG("DIAG mesh is non-orientable; deferring cleanup to component removal");
			}
		}

#if 1
		// Pass 3: Remove faces whose normals disagree with neighbors (catches
		// any remaining inverted faces that survived orientation fixing).
		{
			vcg::tri::UpdateNormal<CLEAN::Mesh>::PerFaceNormalized(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);

			// Single-pass collection: mark faces for removal without iteration.
			// This prevents cascading where removing one face exposes neighbors
			// to subsequent removal, punching growing holes in the surface.
			std::vector<CLEAN::Mesh::FacePointer> toDelete;
			for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (fi->IsD()) continue;

				const auto& fNormal = fi->N();
				const float fNormSq = fNormal.SquaredNorm();
				if (fNormSq < 1e-12f) {
					toDelete.push_back(&*fi);
					continue;
				}

				int nAgree = 0, nDisagree = 0;
				for (int e = 0; e < 3; ++e) {
					auto* adj = fi->FFp(e);
					if (adj == &*fi || adj->IsD())
						continue;
					const float dot = fNormal * adj->N();
					if (dot > 0) ++nAgree;
					else ++nDisagree;
				}

				// Remove only if ALL neighbors disagree AND we have all 3
				// valid neighbors. This ensures we only catch faces that are
				// fully surrounded by opposing normals (interior of an
				// inverted sheet), never faces on boundaries or creases.
				if (nDisagree == 3 && nAgree == 0) {
					toDelete.push_back(&*fi);
				}
			}

			for (auto* fp : toDelete)
				vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fp);

			if (!toDelete.empty()) {
				stats.removedLongEdgeFaces += (int)toDelete.size();
				DEBUG("DIAG removed %d remaining flipped-normal faces", (int)toDelete.size());
			}
		}
#else
		// JPB WIP BUG This work definitely gets rid of backward normal sheets.
		// Pass 3: Remove faces whose normals disagree with neighbors (catches
		// any remaining inverted faces that survived orientation fixing).
		{
			vcg::tri::UpdateNormal<CLEAN::Mesh>::PerFaceNormalized(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);

			int removedFlipped = 0;
			// Iterate multiple times since removing one face may expose new disagreements
			for (int iter = 0; iter < 3; ++iter) {
				int iterRemoved = 0;
				for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
					if (fi->IsD()) continue;

					const auto& fNormal = fi->N();
					const float fNormSq = fNormal.SquaredNorm();
					if (fNormSq < 1e-12f) {
						vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fi);
						++iterRemoved;
						continue;
					}

					int nAgree = 0, nDisagree = 0;
					for (int e = 0; e < 3; ++e) {
						auto* adj = fi->FFp(e);
						if (adj == &*fi || adj->IsD())
							continue;
						const float dot = fNormal * adj->N();
						if (dot > 0) ++nAgree;
						else ++nDisagree;
					}

					// Only remove if ALL neighbors disagree AND we have at least
					// 2 valid neighbors. A face with only 1 neighbor on a boundary
					// or sharp crease should never be removed — that creates holes.
					if (nDisagree >= 2 && nAgree == 0) {
						vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fi);
						++iterRemoved;
					}
				}
				removedFlipped += iterRemoved;
				if (iterRemoved == 0) break;

				// Rebuild topology for next iteration
				vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
				vcg::tri::UpdateNormal<CLEAN::Mesh>::PerFaceNormalized(mesh);
			}

			if (removedFlipped > 0) {
				stats.removedLongEdgeFaces += removedFlipped;
				DEBUG("DIAG removed %d remaining flipped-normal faces", removedFlipped);
			}
		}
#endif

		// Small connected component removal (RELATIVE-TO-LARGEST face count).
		// Keep components with >= MESH_KEEP_COMPONENT_PCT_X1000/1000 percent of the
		// largest component's face count; delete the rest. The main body is always
		// the largest, so this drops floating junk on any scene without an absolute
		// size. Only disconnected components are affected.
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		{
			std::vector<std::pair<int, CLEAN::Mesh::FacePointer>> CCV;
			vcg::tri::Clean<CLEAN::Mesh>::ConnectedComponents(mesh, CCV);
			int largest = 0;
			for (auto& cc : CCV) largest = std::max(largest, cc.first);

			// DIAG: component face-count distribution (top 12). If the 2nd-
			// largest is itself big, leftover junk is a LARGE component -> raise
			// the %. If everything after the body is tiny, the remaining junk is
			// CONNECTED to the body (a thin thread) and no size filter can drop
			// it -- that needs a different fix.
			{
				std::vector<int> sz; sz.reserve(CCV.size());
				for (auto& cc : CCV) sz.push_back(cc.first);
				std::sort(sz.begin(), sz.end(), std::greater<int>());
				char buf[256]; int off = 0;
				for (size_t i = 0; i < sz.size() && i < 12 && off < 230; ++i)
					off += snprintf(buf + off, sizeof(buf) - off, "%d ", sz[i]);
				DEBUG("DIAG component sizes (top of %zu): %s", CCV.size(), buf);
			}

			const int fnBefore = mesh.fn;
			if (largest > 0 && CCV.size() > 1) {
				const double frac = double(MESH_KEEP_COMPONENT_PCT_X1000) / 100000.0; // (pct/1000)/100
				const int sizeThreshold = std::max(1, (int)(frac * (double)largest));
				vcg::tri::Clean<CLEAN::Mesh>::RemoveSmallConnectedComponentsSize(mesh, sizeThreshold);
				stats.removedComponents += (fnBefore - mesh.fn);
				if (fnBefore != mesh.fn)
					DEBUG("Removed %d faces in small components (kept >= %.3g%% of largest=%d faces -> threshold %d faces, of %zu components)",
						fnBefore - mesh.fn, float(MESH_KEEP_COMPONENT_PCT_X1000) / 1000.f, largest, sizeThreshold, CCV.size());
			}
		}

		CompactAndRefresh();

		DEBUG("DIAG after topology repair: %d vn, %d fn (long-edge pass %s)",
			mesh.vn, mesh.fn,
			fSpurious > 0 ? "ran" : "skipped, fSpurious=0");
		ValidateMesh(mesh, "after topology repair", false);
	}

	// =============================================================
	// Phase 3: Spike removal
	// =============================================================
	if (bRemoveSpikes) {
		LightRefresh();
		RemoveSpikes(mesh, stats);

		DEBUG("DIAG after spikes: %d vn, %d fn", mesh.vn, mesh.fn);
		ValidateMesh(mesh, "after spike removal", true);
	}

	// =============================================================
	// Phase 3.4: Boundary-tooth peel -- removes "loose polygon slop"
	// hanging off the silhouette.
	//
	// The graph-cut footprint edge leaves a sawtooth fringe: single
	// triangles attached to the body by ONE edge (their other two edges
	// open = "teeth"), and 1-triangle-wide whisker chains. Top-down they
	// read as ragged, detached-looking polygon slop along the edges.
	//
	// We iteratively delete any face with >= 2 OPEN edges. A tooth (2 open
	// edges) contributes nothing to the footprint -- removing it replaces
	// two jagged outline edges with one straighter one. Whisker chains
	// erode tip-inward as each removal exposes the next. CRITICAL: a face
	// with only 1 open edge is NEVER touched, so the pass stops at the
	// first solid (2D) ring and cannot recede the real silhouette or punch
	// holes. Few iterations, tiny face counts, self-terminating.
	// Compile-time gated by MESH_TOOTH_PEEL_ENABLED (default 0 = off).
#if MESH_TOOTH_PEEL_ENABLED
	{
		constexpr int  TOOTH_MAX_ITERS    = 4;
		{
			int toothRemoved = 0, toothIters = 0;
			for (; toothIters < TOOTH_MAX_ITERS; ++toothIters) {
				vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
				std::vector<CLEAN::Mesh::FacePointer> toDelete;
				for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
					if (fi->IsD()) continue;
					int nOpen = 0;
					for (int e = 0; e < 3; ++e) {
						auto* adj = fi->FFp(e);
						if (adj == &*fi || adj->IsD()) ++nOpen;
					}
					if (nOpen >= 2)
						toDelete.push_back(&*fi);
				}
				if (toDelete.empty()) break;
				for (auto* fp : toDelete)
					vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fp);
				toothRemoved += (int)toDelete.size();
			}
			if (toothRemoved > 0) {
				stats.removedLongEdgeFaces += toothRemoved;
				CompactAndRefresh();
				DEBUG("DIAG boundary-tooth peel: removed %d sawtooth/whisker faces in %d rings",
					toothRemoved, toothIters);
			}
		}
	}
#endif // MESH_TOOTH_PEEL_ENABLED

	// JPB WIP P BUG Sail peel not needed for Poisson
	// =============================================================
	// Phase 3.45: Boundary sail/curtain/underside peel (orientation-based)
	//
	// Removes the Poisson/graph-cut skirt hanging off the silhouette AND the
	// down-facing back sheet Poisson bills UNDER a top-only reconstruction
	// (reverse/cyan normals). Iterative rings: delete a face that touches the
	// OPEN boundary AND is NOT clearly upward-facing (unit normal.z < minUpZ)
	// AND (if the length gate is on) is elongated (longest edge > lenThresh).
	// The flat top surface (nz ~ +1) is KEPT so completeness is preserved;
	// vertical drapes (nz ~ 0) and the down-facing underside (nz < 0) are peeled.
	// Removing the outer ring exposes the next, so multi-triangle drapes/undersides
	// erode inward; a clearly-upward ring stops the pass. Assumes +Z is up.
	// Compile-time gated by MESH_SAIL_PEEL_ENABLED (see macro block).
#if MESH_SAIL_PEEL_ENABLED
	{
		const float medianEdge = ComputeMedianEdgeLength(mesh);
		const float lenThresh = (float(MESH_SAIL_PEEL_LEN_MULT_X100) / 100.f) * medianEdge;
		const float minUpZ = float(MESH_SAIL_PEEL_MAX_NZ_X100) / 100.f;
		int sailRemoved = 0, sailRings = 0;
		if (medianEdge > 0.f) {
			for (; sailRings < MESH_SAIL_PEEL_MAX_ITERS; ++sailRings) {
				vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
				std::vector<CLEAN::Mesh::FacePointer> toDelete;
				for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
					if (fi->IsD()) continue;
					// (a) must touch the open boundary
					bool onBoundary = false;
					for (int e = 0; e < 3; ++e) {
						auto* adj = fi->FFp(e);
						if (adj == &*fi || adj->IsD()) { onBoundary = true; break; }
					}
					if (!onBoundary) continue;
					const auto& p0 = fi->V(0)->cP();
					const auto& p1 = fi->V(1)->cP();
					const auto& p2 = fi->V(2)->cP();
					// (c) length gate (optional: lenThresh <= 0 disables it)
					if (lenThresh > 0.f) {
						const float e0 = (p1 - p0).Norm();
						const float e1 = (p2 - p1).Norm();
						const float e2 = (p0 - p2).Norm();
						const float lMax = std::max(e0, std::max(e1, e2));
						if (lMax <= lenThresh) continue;
					}
					// (b) orientation: peel UNLESS the face is clearly UPWARD-facing.
					// uz = unit normal.z (Z-up). Top surface uz~+1 (KEPT); vertical
					// drape uz~0 and ballooned Poisson underside uz<0 (both PEELED).
					const CLEAN::Mesh::CoordType nrm = (p1 - p0) ^ (p2 - p0);
					const float nl = nrm.Norm();
					if (nl <= 1e-12f) { toDelete.push_back(&*fi); continue; }
					if (nrm[2] / nl < minUpZ)
						toDelete.push_back(&*fi);
				}
				if (toDelete.empty()) break;
				for (auto* fp : toDelete)
					vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fp);
				sailRemoved += (int)toDelete.size();
			}
		}
		if (sailRemoved > 0) {
			stats.removedLongEdgeFaces += sailRemoved;
			CompactAndRefresh();
			DEBUG("DIAG boundary-sail peel: removed %d non-upward drape/underside faces in %d rings (keep nz>=%.2f, len-gate %.2fx median %.3g)",
				sailRemoved, sailRings, minUpZ, float(MESH_SAIL_PEEL_LEN_MULT_X100) / 100.f, medianEdge);
		}
		ValidateMesh(mesh, "after sail peel", true);
	}
#endif // MESH_SAIL_PEEL_ENABLED

	// =============================================================
	// Phase 3.46: Global down-facing cull (watertight-underside removal)
	// See MESH_DOWN_CULL_ENABLED macro. Boundary-independent orientation cull.
	// =============================================================
#if MESH_DOWN_CULL_ENABLED
	{
		const float maxNZ = float(MESH_DOWN_CULL_MAX_NZ_X100) / 100.f;
		std::vector<CLEAN::Mesh::FacePointer> toDelete;
		for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
			if (fi->IsD()) continue;
			const auto& p0 = fi->V(0)->cP();
			const auto& p1 = fi->V(1)->cP();
			const auto& p2 = fi->V(2)->cP();
			const CLEAN::Mesh::CoordType nrm = (p1 - p0) ^ (p2 - p0);
			const float nl = nrm.Norm();
			if (nl <= 1e-12f) { toDelete.push_back(&*fi); continue; }
			if (nrm[2] / nl < maxNZ)
				toDelete.push_back(&*fi);
		}
		if (!toDelete.empty()) {
			for (auto* fp : toDelete)
				vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fp);
			stats.removedLongEdgeFaces += (int)toDelete.size();
			CompactAndRefresh();
			DEBUG("DIAG down-cull: removed %d down-facing faces (nz < %.2f) -> %d fn",
				(int)toDelete.size(), maxNZ, mesh.fn);
		}
		ValidateMesh(mesh, "after down-cull", true);
	}
#endif // MESH_DOWN_CULL_ENABLED

	// =============================================================
	// Phase 3.48: Underside deflate (push down-hanging lobes up to the surface)
	// See MESH_DEFLATE_ENABLED macro. Height-field clamp + seam smooth; no deletes.
	// =============================================================
#if MESH_DEFLATE_ENABLED
	{
		const float medianEdge = ComputeMedianEdgeLength(mesh);
		if (medianEdge > 0.f && mesh.vn > 0) {
			vcg::tri::UpdateBounding<CLEAN::Mesh>::Box(mesh);
			const auto& bbmin = mesh.bbox.min;
			const auto& bbmax = mesh.bbox.max;
			const float spanX = std::max(1e-6f, bbmax[0] - bbmin[0]);
			const float spanY = std::max(1e-6f, bbmax[1] - bbmin[1]);
			float cell = (float(MESH_DEFLATE_CELL_MULT_X100) / 100.f) * medianEdge;
			if (cell <= 0.f) cell = std::max(spanX, spanY);
			const int MAXDIM = 4096;
			int gw = (int)(spanX / cell) + 2;
			int gh = (int)(spanY / cell) + 2;
			if (gw > MAXDIM || gh > MAXDIM) {
				cell = std::max(spanX / (MAXDIM - 2), spanY / (MAXDIM - 2));
				gw = (int)(spanX / cell) + 2;
				gh = (int)(spanY / cell) + 2;
			}
			const float invCell = 1.f / cell;
			const float margin = (float(MESH_DEFLATE_MARGIN_MULT_X100) / 100.f) * medianEdge;
			auto cellIndex = [&](float x, float y) -> size_t {
				int cx = (int)((x - bbmin[0]) * invCell); if (cx < 0) cx = 0; if (cx >= gw) cx = gw - 1;
				int cy = (int)((y - bbmin[1]) * invCell); if (cy < 0) cy = 0; if (cy >= gh) cy = gh - 1;
				return (size_t)cy * (size_t)gw + (size_t)cx;
			};
			std::vector<float> topZ((size_t)gw * (size_t)gh, -FLT_MAX);
			for (auto& v : mesh.vert) {
				if (v.IsD()) continue;
				const auto& p = v.cP();
				float& t = topZ[cellIndex(p[0], p[1])];
				if (p[2] > t) t = p[2];
			}
			// Lift any vertex hanging > margin below the local top up to (top - margin).
			const size_t NV = mesh.vert.size();
			std::vector<uint8_t> lifted(NV, 0);
			int nLifted = 0;
			for (size_t i = 0; i < NV; ++i) {
				auto& v = mesh.vert[i];
				if (v.IsD()) continue;
				const auto& p = v.cP();
				const float t = topZ[cellIndex(p[0], p[1])];
				if (t > -FLT_MAX && (t - p[2]) > margin) {
					v.P()[2] = t - margin;
					lifted[i] = 1;
					++nLifted;
				}
			}
			// Z-only Laplacian smoothing over lifted verts + their 1-ring to blend the seam.
			if (nLifted > 0 && MESH_DEFLATE_SMOOTH_ITERS > 0) {
				vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
				// build the smoothing set: lifted verts and their immediate neighbors
				std::vector<uint8_t> inBand = lifted;
				for (auto& f : mesh.face) {
					if (f.IsD()) continue;
					const int a = (int)vcg::tri::Index(mesh, f.V(0));
					const int b = (int)vcg::tri::Index(mesh, f.V(1));
					const int c = (int)vcg::tri::Index(mesh, f.V(2));
					if (lifted[a] || lifted[b] || lifted[c]) { inBand[a] = inBand[b] = inBand[c] = 1; }
				}
				std::vector<float> zAccum(NV), zCnt(NV);
				for (int it = 0; it < MESH_DEFLATE_SMOOTH_ITERS; ++it) {
					std::fill(zAccum.begin(), zAccum.end(), 0.f);
					std::fill(zCnt.begin(), zCnt.end(), 0.f);
					for (auto& f : mesh.face) {
						if (f.IsD()) continue;
						const int idx[3] = {
							(int)vcg::tri::Index(mesh, f.V(0)),
							(int)vcg::tri::Index(mesh, f.V(1)),
							(int)vcg::tri::Index(mesh, f.V(2)) };
						for (int e = 0; e < 3; ++e) {
							const int u = idx[e], w = idx[(e + 1) % 3];
							zAccum[u] += mesh.vert[w].cP()[2]; zCnt[u] += 1.f;
							zAccum[w] += mesh.vert[u].cP()[2]; zCnt[w] += 1.f;
						}
					}
					for (size_t i = 0; i < NV; ++i)
						if (inBand[i] && zCnt[i] > 0.f && !mesh.vert[i].IsD())
							mesh.vert[i].P()[2] = 0.5f * mesh.vert[i].cP()[2] + 0.5f * (zAccum[i] / zCnt[i]);
				}
			}
			if (nLifted > 0) {
				CompactAndRefresh();
				DEBUG("DIAG underside-deflate: lifted %d hanging verts to local top (grid %dx%d cell %.3g, margin %.3g) -> %d fn",
					nLifted, gw, gh, cell, margin, mesh.fn);
			}
			ValidateMesh(mesh, "after underside deflate", true);
		}
	}
#endif // MESH_DEFLATE_ENABLED

	// =============================================================
	// Phase 3.5: Decimation (deferred from Phase 1)
	//
	// Run AFTER spurious + spike removal so the quadric edge-collapse
	// operates on a clean mesh.  This preserves sharp features (roof
	// edges, building outlines) that would otherwise be lost while the
	// collapser "fixes" artifacts.
	// =============================================================
	if ((fDecimate > 0 && fDecimate < 1.0f) || fDecimateError > 0.f) {
		const int removedArea = vcg::tri::Clean<CLEAN::Mesh>::RemoveFaceOutOfRangeArea(mesh, eps);
		const int removedDup = vcg::tri::Clean<CLEAN::Mesh>::RemoveDuplicateFace(mesh);

		if (removedArea > 0 || removedDup > 0)
			LightRefresh();

		// VCG quadric edge-collapse REQUIRES a clean, compacted mesh, and its
		// Init() internally calls FaceBorderFromVF() to find the boundary that
		// PreserveBoundary must lock.
		//
		// ROOT CAUSE (fixed 2026 in vcglib update/flag.h): the fork's optimized
		// "seenGen" branch of FaceBorderFromVF was broken -- it set BORDERFLAG on
		// the first time each neighbor was seen and never cleared it via parity,
		// so EVERY edge (incl. interior 2-face edges) was flagged border. That
		// made this diagnostic read "100% border", PreserveBoundary ClearW-locked
		// every vertex, the collapse heap came up empty, and decimation silently
		// did nothing. (Separately, PreserveBoundary=false on the resulting state
		// crashed with 0xC0000005.) With the vcglib parity bug fixed, the border
		// set is now the true silhouette/holes and PreserveBoundary=true works.
		//
		// We still Compact() here (the spike/tooth-peel passes DeleteFace()'d and
		// only LightRefresh()'d, leaving stale storage) and we ONLY ever run the
		// crash-safe PreserveBoundary=true mode.
		Compact();

		// Clamp the target so a degenerate fDecimate can never request 0 faces. In adaptive
		// mode fDecimate (if in (0,1)) is the KEEP FLOOR; if decimation is otherwise disabled
		// (fDecimate >= 1) only a minimal safety floor applies and the error metric drives it.
		// Effective strength: runtime --decimate-error (fDecimateError) overrides the
		// MESH_DECIMATE_ERROR_K_X100 compile-time default; 0 = legacy fixed-ratio decimation.
		const int OriginalFaceNum(mesh.face.size());
		const double kEff = (fDecimateError > 0.f) ? (double)fDecimateError : (MESH_DECIMATE_ERROR_K_X100 / 100.0);
		int targetFaces = (fDecimate > 0 && fDecimate < 1.f)
			? std::max<int>(4, ROUND2INT(fDecimate * mesh.fn))
			: 4;
		float medianEdgeLen = 0.f;
		double errorDev = 0.0;
		if (kEff > 0) {
			// Adaptive (error-bounded) decimation: stop when the next quadric collapse's
			// geometric error exceeds tau (from the scene scale, set per-pass below), so flat
			// regions collapse maximally and detail is preserved.
			medianEdgeLen = ComputeMedianEdgeLength(mesh);
			errorDev = (double)medianEdgeLen * kEff;
			DEBUG("Adaptive decimation ON: medianEdge=%.4g, k=%.2f, allowed-deviation~%.4g world units, keep-floor=%d faces",
				medianEdgeLen, kEff, errorDev, targetFaces);
		}
		DEBUG("Original faces: %d, target faces: %d", OriginalFaceNum, targetFaces);

		const auto oldNested = omp_get_nested();
		const auto oldDynamic = omp_get_dynamic();

		// One boundary-preserving quadric-collapse pass.
		// NOTE: DoOptimization(size_t)'s argument is only a reserve() hint in
		// this fork -- it does NOT bound collapses; a single call runs until
		// the heap empties or the SetTargetSimplices() goal is reached.
		// PreserveBoundary=true is the ONLY safe mode (PreserveBoundary=false
		// crashes on any residual non-manifold edge). Returns faces collapsed.
		auto runDecimate = [&](bool logDiag) -> int {
			vcg::tri::TriEdgeCollapseQuadricParameter pp;
			pp.OptimalPlacement = true;
			pp.PreserveBoundary = true;
			pp.PreserveTopology = false;
			if (kEff > 0) {
				// deterministic scene-normalized quadric so the error threshold below is in
				// the collapse-priority's own units (Init sets g_ScaleFactor = 1e8/diag^6)
				pp.ScaleIndependent = true;
			}

			vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
			vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromVF(mesh);

			// Diagnostic (read-only): how much of the mesh is border? Every
			// border vertex is locked by PreserveBoundary, so if this is near
			// 100% the heap comes up empty and decimation is a silent no-op.
			if (logDiag) {
				const auto* baseV = &mesh.vert[0];
				std::vector<char> vb(mesh.vert.size(), 0);
				size_t borderEdges = 0;
				for (auto& f : mesh.face) {
					if (f.IsD()) continue;
					for (int j = 0; j < 3; ++j) {
						if (f.IsB(j)) {
							++borderEdges;
							vb[(size_t)(f.V(j)  - baseV)] = 1;
							vb[(size_t)(f.V1(j) - baseV)] = 1;
						}
					}
				}
				size_t borderVerts = 0, liveVerts = 0;
				for (size_t i = 0; i < mesh.vert.size(); ++i) {
					if (mesh.vert[i].IsD()) continue;
					++liveVerts;
					if (vb[i]) ++borderVerts;
				}
				DEBUG("DIAG decimate input: %u live verts, %u border verts (%.1f%%), %u border edges, fn=%d, target=%d, OptimalPlacement=1",
					(unsigned)liveVerts, (unsigned)borderVerts,
					liveVerts ? 100.0 * (double)borderVerts / (double)liveVerts : 0.0,
					(unsigned)borderEdges, mesh.fn, targetFaces);
			}

			vcg::math::Quadric<double> QZero; QZero.SetZero();
			CLEAN::QuadricTemp TD(mesh.vert, QZero);
			CLEAN::QHelper::TDp() = &TD;

			vcg::LocalOptimization<CLEAN::Mesh> deci(mesh, &pp);
			deci.Init<CLEAN::TriEdgeCollapse>();
			deci.SetTargetSimplices(targetFaces);
			if (kEff > 0) {
				// Express the deviation tolerance in the collapse-priority's units. Init just
				// refreshed mesh.bbox and set g_ScaleFactor = 1e8/diag^6 (ScaleIndependent), so
				// tau = g_ScaleFactor * deviation^2. SetTargetSimplices above stays active as
				// the keep floor -- decimation stops at whichever goal triggers first.
				const double diag = mesh.bbox.Diag();
				if (diag > 0 && errorDev > 0) {
					const double scale = 1e8 * std::pow(1.0 / diag, 6.0);
					deci.SetTargetMetric((CLEAN::Mesh::ScalarType)(scale * errorDev * errorDev));
				}
			}
			deci.SetTimeBudget(1.f);

			const int faceBefore = mesh.fn;
			Util::Progress progress(_T("Decimating"), faceBefore - targetFaces);
			while (mesh.fn > targetFaces && deci.DoOptimization(mesh.vert.size()))
				progress.display(faceBefore - mesh.fn);
			deci.Finalize<CLEAN::TriEdgeCollapse>();
			progress.close();
			if (logDiag && kEff > 0) {
				const bool hitFloor = (mesh.fn <= targetFaces);
				const bool hitMetric = (deci.currMetric > deci.targetMetric);
				DEBUG("DIAG decimate stop: %s | final-error=%.4g tau=%.4g (ratio=%.2f)",
					hitFloor ? "FLOOR" : hitMetric ? "METRIC" : "HEAP-EMPTY",
					(double)deci.currMetric, (double)deci.targetMetric,
					deci.targetMetric > 0 ? (double)deci.currMetric / (double)deci.targetMetric : 0.0);
			}
			return faceBefore - mesh.fn;
		};

		int totalCollapsed = 0;
		if (mesh.fn > targetFaces)
			totalCollapsed += runDecimate(true);
		DEBUG("Decimation pass 1 (PreserveBoundary=true): %d collapsed, fn=%d", totalCollapsed, mesh.fn);

		// Fallback: if pass 1 collapsed nothing, the mesh is genuinely
		// non-2-manifold even after compaction (every edge a true border).
		// Repair manifoldness so the collapser has a real interior, then retry
		// -- still PreserveBoundary=true (never the crashing boundary-free mode).
		if (totalCollapsed == 0 && mesh.fn > targetFaces) {
			using Tri = vcg::tri::Clean<CLEAN::Mesh>;
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			const int nmf = Tri::RemoveNonManifoldFace(mesh);
			const int nmv = Tri::SplitNonManifoldVertex(mesh, 0);
			const int unref = Tri::RemoveUnreferencedVertex(mesh);
			Compact();
			DEBUG("Decimation manifold-repair: removed %d non-manifold faces, split %d non-manifold verts, removed %d unref verts -> fn=%d",
				nmf, nmv, unref, mesh.fn);

			if (mesh.fn > targetFaces) {
				const int again = runDecimate(true);
				totalCollapsed += again;
				DEBUG("Decimation pass 2 (after manifold repair): %d collapsed, fn=%d", again, mesh.fn);
			}
		}

		omp_set_nested(oldNested);
		omp_set_dynamic(oldDynamic);

		// TIER 1: hole-closing phase below rebuilds both FF and VF as its first
		// action, so the topology rebuild here would be wasted.
		Compact();

		DEBUG("DIAG after decimation: %d vn, %d fn (%d total collapsed)", mesh.vn, mesh.fn, totalCollapsed);
		// Safety net: decimation must never empty a non-empty mesh. If it does
		// (degenerate quadrics on pathological input), warn loudly — the
		// guards above should prevent it, but this catches any residual case
		// before the empty mesh propagates to the output file.
		if (mesh.fn == 0 && OriginalFaceNum > 0)
			DEBUG("warning: decimation emptied the mesh (was %d faces) — check input topology/flags", OriginalFaceNum);
		ValidateMesh(mesh, "after decimation", true);
	}

	// =============================================================
	// Phase 4: Hole closing
	//
	// Fill INTERIOR holes only. Edge count cannot distinguish a real interior
	// hole from the outer silhouette / a concave bay, so we gate on geometric
	// extent: a loop is filled only if its bbox diagonal is within
	// MESH_HOLE_MAX_DIAG_FRAC of the whole-mesh diagonal (compact = interior),
	// up to MESH_HOLE_MAX_EDGES edges. The big outer/bay loops are left open so
	// ear-cutting never fans a sheet across them. SelfIntersectionEar remains
	// the per-ear backstop.
	// =============================================================
	if (nCloseHoles > 0) {
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
		vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
		vcg::tri::UpdateNormal<CLEAN::Mesh>::PerFaceNormalized(mesh);
		vcg::tri::UpdateNormal<CLEAN::Mesh>::PerVertexAngleWeighted(mesh);
		vcg::tri::UpdateBounding<CLEAN::Mesh>::Box(mesh);

		const float meshDiag = mesh.bbox.Diag();
		// Measured once and used by BOTH gates below (the edge cap and the
		// resolution-relative span cap), so it is hoisted out of the edge-cap block.
		const float medianEdge = ComputeMedianEdgeLength(mesh);
		// --close-holes (nCloseHoles) is the edge cap, made DENSITY-AWARE so a
		// given value closes the same PHYSICAL hole size at any mesh fineness
		// (see MESH_HOLE_REF_EDGES_ACROSS_DIAG). Finer mesh -> more edges per
		// hole -> proportionally larger cap. Clamped to the runtime ceiling.
		int holeEdgeCap;
		{
			int scaledCloseHoles = (int)nCloseHoles;
			if (MESH_HOLE_REF_EDGES_ACROSS_DIAG > 0) {
				if (medianEdge > 0.f && meshDiag > 0.f) {
					const float edgesAcrossDiag = meshDiag / medianEdge;
					const float densityScale = edgesAcrossDiag / float(MESH_HOLE_REF_EDGES_ACROSS_DIAG);
					scaledCloseHoles = std::max(1, ROUND2INT(nCloseHoles * densityScale));
					DEBUG("DIAG holes density-scale: median-edge=%.4g, edges-across-diag=%.0f, scale=%.2f, --close-holes %u -> %d",
						medianEdge, edgesAcrossDiag, densityScale, nCloseHoles, scaledCloseHoles);
				}
			}
			holeEdgeCap = std::min<int>(scaledCloseHoles, MESH_HOLE_MAX_EDGES);
		}
		// Take the more restrictive of the scene-relative and resolution-relative
		// gates (see MESH_HOLE_MAX_SPAN_EDGES). On small/close-range scenes the frac
		// gate is already the tighter of the two and nothing changes; on large sites
		// the span gate is what stops the fill inventing tens of metres of surface.
		const float fracHoleDiag = (float(MESH_HOLE_MAX_DIAG_FRAC_X1000) / 1000.f) * meshDiag;
		const float spanHoleDiag = (MESH_HOLE_MAX_SPAN_EDGES > 0 && medianEdge > 0.f)
			? float(MESH_HOLE_MAX_SPAN_EDGES) * medianEdge : fracHoleDiag;
		const float maxHoleDiag = std::min(fracHoleDiag, spanHoleDiag);

#if 0 // JPB WIP BUG Diag
		// DIAG: classify every boundary loop so we can see WHY interior holes
		// remain open after filling -- gate-skipped (too wide / too many edges)
		// vs ear-rejected (passes the gate but SelfIntersectionEar can't fill
		// flat near nearby geometry). Compare `pass gate` here to the `Closed N`
		// count below: if Closed << pass-gate, ear-rejection dominates; if many
		// are skip-wide, raise MESH_HOLE_MAX_DIAG_FRAC_X1000; if skip-edges,
		// raise --close-holes.
		{
			std::vector<vcg::tri::Hole<CLEAN::Mesh>::Info> loops;
			vcg::tri::Hole<CLEAN::Mesh>::GetInfo(mesh, false, loops);
			size_t nPass = 0, nWide = 0, nMany = 0;
			float maxDiag = 0.f; int maxEdges = 0;
			int wideBucket[5] = { 0,0,0,0,0 }; // diag as % of mesh: (g-20],(20-30],(30-50],(50-100],>100
			for (auto& L : loops) {
				if (L.size < 3) continue;
				const float d = L.bb.Diag();
				if (d > maxDiag) maxDiag = d;
				if (L.size > maxEdges) maxEdges = L.size;
				const bool wide = d > maxHoleDiag;
				const bool many = L.size >= holeEdgeCap;
				if (wide) {
					++nWide;
					const float pct = 100.f * d / meshDiag;
					const int b = pct <= 20.f ? 0 : pct <= 30.f ? 1 : pct <= 50.f ? 2 : pct <= 100.f ? 3 : 4;
					++wideBucket[b];
				}
				if (many) ++nMany;
				if (!wide && !many) ++nPass;
			}
			DEBUG("DIAG holes pre-fill: %zu loops | %zu pass gate | %zu skip-wide | %zu skip-edges(>=%d) | largest diag=%.3g (%.0f%% mesh) edges=%d",
				loops.size(), nPass, nWide, nMany, holeEdgeCap, maxDiag, meshDiag > 0 ? 100.f * maxDiag / meshDiag : 0.f, maxEdges);
			DEBUG("DIAG wide-hole diag buckets (%% of mesh diag): (gate-20]=%d (20-30]=%d (30-50]=%d (50-100]=%d (>100]=%d",
				wideBucket[0], wideBucket[1], wideBucket[2], wideBucket[3], wideBucket[4]);
		}
#endif

		const int closed = vcg::tri::Hole<CLEAN::Mesh>::EarCuttingIntersectionFill<
			vcg::tri::SelfIntersectionEar<CLEAN::Mesh>>(mesh, holeEdgeCap, false, nullptr, maxHoleDiag);

		if (closed > 0) {
			DEBUG("Closed %d interior holes (<= %.3g world units [%s: frac %.3g vs span %.3g = %d x median edge %.4g], up to %d edges)",
				closed, maxHoleDiag,
				(spanHoleDiag < fracHoleDiag) ? "span-gated" : "frac-gated",
				fracHoleDiag, spanHoleDiag, MESH_HOLE_MAX_SPAN_EDGES, medianEdge, holeEdgeCap);
			CompactAndRefresh();
		}

		stats.closedHoles += closed;
		ValidateMesh(mesh, "following hole closing", true);
	}

	// =============================================================
	// Phase 5: Smoothing
	// =============================================================
	if (nSmooth > 0) {
		vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
		vcg::tri::Smooth<CLEAN::Mesh>::VertexCoordLaplacian(mesh, nSmooth, false, false);
		if (vcg::tri::Clean<CLEAN::Mesh>::RemoveFaceOutOfRangeArea(mesh, eps))
			CompactAndRefresh();

		ValidateMesh(mesh, "after smoothing", true);
	}

	// =============================================================
	// Phase 6: Remeshing
	// =============================================================
	if (fEdgeLength > 0) {
		CLEAN::Mesh original;
		vcg::tri::Append<CLEAN::Mesh, CLEAN::Mesh>::MeshCopy(original, mesh);

		vcg::tri::IsotropicRemeshing<CLEAN::Mesh>::Params params;
		params.SetTargetLen(fEdgeLength);
		params.iter = 3;
		params.cleanFlag = true;

		try {
			vcg::tri::IsotropicRemeshing<CLEAN::Mesh>::Do(mesh, original, params);
		}
		catch (vcg::MissingPreconditionException& e) {
			VERBOSE("Remesh error: %s", e.what());
		}

		vcg::tri::Clean<CLEAN::Mesh>::RemoveDuplicateFace(mesh);
		vcg::tri::Clean<CLEAN::Mesh>::RemoveFaceOutOfRangeArea(mesh, eps);
		// TIER 1: next phase (FastClean / Phase 8) rebuilds its own topology, so
		// only compact here.
		Compact();
	}

	// =============================================================
	// Phase 7: Final cleanup
	// =============================================================
	if (bLastClean) {
		FastClean(mesh, stats);
		ValidateMesh(mesh, "after final clean", true);
	}

	// =============================================================
	// Phase 8: Final conservative seal
	//
	// First fix any remaining non-manifold edges/vertices so that
	// hole boundaries form clean loops that VCG can detect and fill.
	// Then fill with intersection-aware ears for larger holes, and
	// trivial ears for the tiny 3-6 edge gaps that remain.
	// =============================================================
	if (nCloseHoles > 0) {
		// Fix non-manifold topology left by earlier face deletions so
		// that border half-edge loops are valid for hole detection.
		{
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
			const int nmfRemoved = vcg::tri::Clean<CLEAN::Mesh>::RemoveNonManifoldFace(mesh);
			const int nmvSplit = vcg::tri::Clean<CLEAN::Mesh>::SplitNonManifoldVertex(mesh, 0);
			if (nmfRemoved > 0 || nmvSplit > 0) {
				DEBUG("DIAG pre-seal fix: removed %d NM faces, split %d NM vertices",
					nmfRemoved, nmvSplit);
				vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
				KillEdges(mesh);
				vcg::tri::Allocator<CLEAN::Mesh>::CompactEveryVector(mesh);
			}
		}

		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
		vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
		vcg::tri::UpdateNormal<CLEAN::Mesh>::PerFaceNormalized(mesh);
		vcg::tri::UpdateNormal<CLEAN::Mesh>::PerVertexAngleWeighted(mesh);
		vcg::tri::UpdateBounding<CLEAN::Mesh>::Box(mesh);

		const float sealMeshDiag = mesh.bbox.Diag();
		const float sealMedianEdge = ComputeMedianEdgeLength(mesh);
		// Density-aware edge cap, matching Phase 4 (see MESH_HOLE_REF_EDGES_ACROSS_DIAG).
		int sealHoleEdgeCap;
		{
			int scaledCloseHoles = (int)nCloseHoles;
			if (MESH_HOLE_REF_EDGES_ACROSS_DIAG > 0) {
				if (sealMedianEdge > 0.f && sealMeshDiag > 0.f) {
					const float densityScale = (sealMeshDiag / sealMedianEdge) / float(MESH_HOLE_REF_EDGES_ACROSS_DIAG);
					scaledCloseHoles = std::max(1, ROUND2INT(nCloseHoles * densityScale));
				}
			}
			sealHoleEdgeCap = std::min<int>(scaledCloseHoles, MESH_HOLE_MAX_EDGES);
		}
		// Same two-gate rule as Phase 4. This also tightens the fallback pass below,
		// which is defined as 2x this value -- it was reaching 315 m on a 1251 m scene.
		const float sealFracHoleDiag = (float(MESH_HOLE_MAX_DIAG_FRAC_X1000) / 1000.f) * sealMeshDiag;
		const float sealSpanHoleDiag = (MESH_HOLE_MAX_SPAN_EDGES > 0 && sealMedianEdge > 0.f)
			? float(MESH_HOLE_MAX_SPAN_EDGES) * sealMedianEdge : sealFracHoleDiag;
		const float sealMaxHoleDiag = std::min(sealFracHoleDiag, sealSpanHoleDiag);

		// First pass: geometry-gated interior-hole fill (same rule as Phase 4),
		// so the final seal closes remaining compact holes without fanning the
		// silhouette/bays.
		int closed = vcg::tri::Hole<CLEAN::Mesh>::EarCuttingIntersectionFill<
			vcg::tri::SelfIntersectionEar<CLEAN::Mesh>>(mesh, sealHoleEdgeCap, false, nullptr, sealMaxHoleDiag);

		// Second pass: fallback fill for the small holes SelfIntersectionEar
		// rejected (ear-rejected 3D tears). TrivialEar fills regardless of
		// intersection. Gated by edge count AND a diag ceiling (2x the main gate)
		// so the outer boundary / wide bays are never bridged. Per-hole loop with
		// all loop face-pointers registered for AddFaces realloc-safety.
		if (MESH_HOLE_FALLBACK_MAX_EDGES > 0) {
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateTopology<CLEAN::Mesh>::VertexFace(mesh);
			vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);

			const float fallbackMaxDiag = 2.0f * sealMaxHoleDiag;
			std::vector<vcg::tri::Hole<CLEAN::Mesh>::Info> loops;
			vcg::tri::Hole<CLEAN::Mesh>::GetInfo(mesh, false, loops);
			std::vector<CLEAN::Mesh::FacePointer*> upd;
			upd.reserve(loops.size());
			for (auto& L : loops) upd.push_back(&L.p.f);
			int closedTiny = 0, skippedStale = 0;
			for (auto& L : loops) {
				if (L.size < 3 || L.size > MESH_HOLE_FALLBACK_MAX_EDGES) continue;
				if (L.bb.Diag() > fallbackMaxDiag) continue; // never bridge a wide loop

				// STALE-POS GUARD -- without this the loop can HANG.
				//
				// GetInfo() collects one Pos per border loop UP FRONT, and nothing
				// revalidates them as we fill. Sealing one hole removes border status from
				// the vertices it closes, so a later loop's Pos can end up on an edge that
				// is no longer a border -- or on a vertex with no border edge left at all
				// when two loops shared vertices. Pos::NextB() is then unbounded:
				//
				//     assert(f->FFp(z)==f);   // compiled out in release
				//     do NextE(); while(!IsBorder());
				//
				// it walks the vertex fan forever looking for a border that no longer
				// exists. (vcglib/vcg/simplex/face/pos.h)
				//
				// Latent until the mesh got fine enough for loops to be adjacent: this pass
				// used to close 1-3 holes, and at a 0.13 cell it sees ~3700 loops, where
				// loops sharing a vertex are close to certain.
				//
				// Re-testing the Pos is cheap and is the standard vcg idiom. The two
				// conditions are exactly NextB's own preconditions: the edge must be a
				// border, and the Pos vertex must belong to that edge.
				if (L.p.f == nullptr || L.p.f->IsD() || !L.p.IsBorder()) {
					++skippedStale;
					continue;
				}
				if (!(L.p.f->V(L.p.z) == L.p.v ||
				      L.p.f->V(L.p.f->Next(L.p.z)) == L.p.v)) {
					++skippedStale;
					continue;
				}

				vcg::tri::Hole<CLEAN::Mesh>::FillHoleEar<vcg::tri::TrivialEar<CLEAN::Mesh>>(mesh, L.p, upd);
				++closedTiny;
			}
			if (skippedStale > 0)
				DEBUG("DIAG fallback seal: skipped %d loops whose Pos was invalidated by an"
					" earlier fill (of %zu loops)", skippedStale, loops.size());
			closed += closedTiny;
			if (closedTiny > 0)
				DEBUG("DIAG fallback seal: trivially closed %d small holes (<=%d edges, <= %.3g-unit diag)",
					closedTiny, MESH_HOLE_FALLBACK_MAX_EDGES, fallbackMaxDiag);
		}

		if (closed > 0) {
			DEBUG("Final seal: closed %d interior holes (<= %.3g world units [%s], up to %d edges)",
				closed, sealMaxHoleDiag,
				(sealSpanHoleDiag < sealFracHoleDiag) ? "span-gated" : "frac-gated",
				sealHoleEdgeCap);
			// TIER 1: nothing after this consumes FF/VF -- the export loop just
			// iterates mesh.vert / mesh.face and skips IsD() entries.
			Compact();
		}

#if 0 // JPB WIP BUG Diag
		// DIAG: how many OPEN boundary loops actually remain after all sealing?
		// The "Closed N" counters above count ATTEMPTS (holes that passed the
		// gate), not full seals -- SelfIntersectionEar leaves a hole partially
		// open when a flat ear would intersect nearby 3D geometry. This is the
		// ground truth: if ~0 loops remain, the mesh is watertight and any white
		// interior areas are filled-but-untextured patches (texturing issue). If
		// many remain (and few are wider than the gate), they are ear-rejected
		// holes that hug relief -- raising the size gate will NOT close them.
		{
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
			std::vector<vcg::tri::Hole<CLEAN::Mesh>::Info> rem;
			vcg::tri::Hole<CLEAN::Mesh>::GetInfo(mesh, false, rem);
			size_t openLoops = 0, widerThanGate = 0;
			float maxD = 0.f; int maxE = 0;
			for (auto& L : rem) {
				if (L.size < 3) continue;
				++openLoops;
				const float d = L.bb.Diag();
				if (d > maxD) maxD = d;
				if (L.size > maxE) maxE = L.size;
				if (d > sealMaxHoleDiag) ++widerThanGate;
			}
			DEBUG("DIAG holes post-seal: %zu open loops remain (%zu wider than %.3g-unit gate), largest diag=%.3g (%.0f%% mesh) edges=%d",
				openLoops, widerThanGate, sealMaxHoleDiag, maxD,
				sealMeshDiag > 0 ? 100.f * maxD / sealMeshDiag : 0.f, maxE);
			// Enumerate every remaining open loop so we can correlate it with the
			// visible holes and tell WHY it survived: a loop whose diag > the gate
			// was skipped (raise the gate); a loop with diag <= gate that is still
			// open was ATTEMPTED and ear-rejected (SelfIntersectionEar refused
			// every flat fill -- gate change won't help). Center locates it.
			{
				int shown = 0;
				for (auto& L : rem) {
					if (L.size < 3) continue;
					if (++shown > 40) { DEBUG("   ... (%zu more)", rem.size() - 40); break; }
					const float d = L.bb.Diag();
					const auto c = L.bb.Center();
					DEBUG("   open loop: diag=%.3g (%.0f%% mesh) edges=%d %s center=(%.1f,%.1f,%.1f)",
						d, sealMeshDiag > 0 ? 100.f * d / sealMeshDiag : 0.f, L.size,
						d > sealMaxHoleDiag ? "SKIPPED-wide" : "ear-rejected", c[0], c[1], c[2]);
				}
			}
		}
#endif

		ValidateMesh(mesh, "after final seal", true);
	}

#if MESH_RIM_ERODE_RINGS > 0
	// =============================================================
	// Phase 8.5: Aggressive uniform rim erosion (perimeter straightening)
	//
	// Peel MESH_RIM_ERODE_RINGS complete rings of border faces off the whole
	// silhouette. Each ring: mark border faces (FaceBorderFromFF) and delete
	// every face that touches the boundary, then refresh topology so the next
	// inward ring becomes the new border. Fringe narrower than ~2R triangles
	// (whiskers, thin peninsulas, isthmus necks holding floating flecks) is
	// fully consumed and the jagged outline recedes to a smoother contour;
	// the solid body simply loses R rings (negligible vs the whole mesh).
	//
	// Because severing a neck can disconnect floating junk from the body, the
	// small-connected-component filter is re-run afterwards to drop anything
	// newly isolated. Only border faces are ever deleted, so a sealed interior
	// hole (no border edges) is never reopened.
	// =============================================================
	{
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		int rimRemoved = 0, rimRings = 0;
		for (; rimRings < MESH_RIM_ERODE_RINGS; ++rimRings) {
			vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
			std::vector<CLEAN::Mesh::FacePointer> toDelete;
			for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (fi->IsD()) continue;
				if (fi->IsB(0) || fi->IsB(1) || fi->IsB(2))
					toDelete.push_back(&*fi);
			}
			if (toDelete.empty()) break;
			for (auto* fp : toDelete)
				vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, *fp);
			rimRemoved += (int)toDelete.size();
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh); // expose next ring
		}
		if (rimRemoved > 0) {
			vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
			Compact();
			DEBUG("DIAG rim-erode: peeled %d border faces in %d rings -> %d fn",
				rimRemoved, rimRings, mesh.fn);

			// Re-run small-component filter: erosion may have severed thin necks
			// that connected floating junk to the body.
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			std::vector<std::pair<int, CLEAN::Mesh::FacePointer>> CCV;
			vcg::tri::Clean<CLEAN::Mesh>::ConnectedComponents(mesh, CCV);
			int largest = 0;
			for (auto& cc : CCV) largest = std::max(largest, cc.first);
			const int fnBefore = mesh.fn;
			if (largest > 0 && CCV.size() > 1) {
				const double frac = double(MESH_KEEP_COMPONENT_PCT_X1000) / 100000.0;
				const int sizeThreshold = std::max(1, (int)(frac * (double)largest));
				vcg::tri::Clean<CLEAN::Mesh>::RemoveSmallConnectedComponentsSize(mesh, sizeThreshold);
				stats.removedComponents += (fnBefore - mesh.fn);
				if (fnBefore != mesh.fn) {
					Compact();
					DEBUG("DIAG rim-erode: dropped %d newly-disconnected faces (kept >= %.3g%% of largest=%d, %zu components)",
						fnBefore - mesh.fn, float(MESH_KEEP_COMPONENT_PCT_X1000) / 1000.f, largest, CCV.size());
				}
			}
		}
		ValidateMesh(mesh, "after rim erosion", true);
	}
#endif // MESH_RIM_ERODE_RINGS > 0

#if MESH_ALPHA_TIGHTEN_ENABLED
	// =============================================================
	// Phase 9: Alpha-shape perimeter tightening (erosion-only)
	//
	// Roll a disk of radius alpha around the border: iteratively delete rim
	// faces whose circumradius > alpha (the alpha-shape criterion), peeling the
	// saw-tooth sliver/spike fringe off the silhouette while leaving genuine
	// bays (lined with normal-size faces) intact. Deletes only -> no fans.
	// alpha auto-scales from the median edge length; iterations capped so it
	// can never cascade into the coarse decimated interior. A/B via the
	// MESH_ALPHA_TIGHTEN_ENABLED macro.
	// =============================================================
	{
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);

		// alpha = K * median edge length (sampled over all live faces).
		std::vector<float> elen;
		elen.reserve(60000);
		for (auto& f : mesh.face) {
			if (f.IsD()) continue;
			const auto& p0 = f.V(0)->cP();
			const auto& p1 = f.V(1)->cP();
			elen.push_back((p1 - p0).Norm());
			if (elen.size() >= 60000) break;
		}
		float medianEdge = 0.f;
		if (!elen.empty()) {
			std::nth_element(elen.begin(), elen.begin() + elen.size() / 2, elen.end());
			medianEdge = elen[elen.size() / 2];
		}
		const float alpha = (float(MESH_ALPHA_TIGHTEN_K_X100) / 100.f) * medianEdge;

		int totalPeeled = 0;
		if (alpha > 0.f) {
			for (int iter = 0; iter < MESH_ALPHA_TIGHTEN_ITERS; ++iter) {
				vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
				int peeled = 0;
				for (auto& f : mesh.face) {
					if (f.IsD()) continue;
					if (!(f.IsB(0) || f.IsB(1) || f.IsB(2))) continue; // rim faces only
					const auto& p0 = f.V(0)->cP();
					const auto& p1 = f.V(1)->cP();
					const auto& p2 = f.V(2)->cP();
					const float a = (p1 - p0).Norm();
					const float b = (p2 - p1).Norm();
					const float c = (p0 - p2).Norm();
					const float cross2 = ((p1 - p0) ^ (p2 - p0)).Norm(); // = 2*Area
					// circumradius R = abc/(4A) = abc/(2*cross2); R > alpha  <=>
					// abc > alpha*2*cross2. Degenerate sliver (cross2 ~ 0) -> peel.
					if (cross2 <= 1e-9f || (a * b * c) > alpha * 2.f * cross2) {
						vcg::tri::Allocator<CLEAN::Mesh>::DeleteFace(mesh, f);
						++peeled;
					}
				}
				if (peeled == 0) break;
				totalPeeled += peeled;
				vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh); // refresh border
			}
		}

		if (totalPeeled > 0) {
			vcg::tri::Clean<CLEAN::Mesh>::RemoveUnreferencedVertex(mesh);
			Compact();
			DEBUG("Alpha-tighten: peeled %d rim faces (alpha=%.3g = %.2f x median edge %.3g, <=%d iters) -> %d fn",
				totalPeeled, alpha, float(MESH_ALPHA_TIGHTEN_K_X100) / 100.f, medianEdge,
				MESH_ALPHA_TIGHTEN_ITERS, mesh.fn);
		} else {
			DEBUG("Alpha-tighten: nothing to peel (alpha=%.3g, median edge %.3g)", alpha, medianEdge);
		}
		ValidateMesh(mesh, "after alpha tighten", true);
	}
#endif

	// =============================================================
	// Phase 9.1: Post-tighten small-component removal
	//
	// Alpha-tighten and tooth-peel can sever thin bridges that previously
	// connected small Poisson fragments to the main body. Re-run the
	// relative-to-largest component filter to drop any newly-orphaned blobs.
	// =============================================================
	{
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		std::vector<std::pair<int, CLEAN::Mesh::FacePointer>> CCV;
		vcg::tri::Clean<CLEAN::Mesh>::ConnectedComponents(mesh, CCV);
		int largest = 0;
		for (auto& cc : CCV) largest = std::max(largest, cc.first);
		if (largest > 0 && CCV.size() > 1) {
			const double frac = double(MESH_KEEP_COMPONENT_PCT_X1000) / 100000.0;
			const int sizeThreshold = std::max(1, (int)(frac * (double)largest));
			const int fnBefore = mesh.fn;
			vcg::tri::Clean<CLEAN::Mesh>::RemoveSmallConnectedComponentsSize(mesh, sizeThreshold);
			if (fnBefore != mesh.fn) {
				CompactAndRefresh();
				DEBUG("DIAG post-tighten component filter: removed %d faces in %zu orphaned blobs (threshold %d faces)",
					fnBefore - mesh.fn, CCV.size() - 1, sizeThreshold);
			}
		}
	}

#if MESH_BAND_REFINE_ENABLED
	// =============================================================
	// Phase 9.5: Boundary-band refinement (more polygons around the edges)
	//
	// Midpoint-subdivide the faces within MESH_BAND_REFINE_RINGS rings of the
	// open boundary. PreserveBoundary decimation keeps the silhouette only at
	// reconstruction density; this adds NEW vertices in the edge band (and on
	// the silhouette line itself) so the following Taubin smooth has finer
	// control. Only the band is refined -> interior stays decimated. vcg
	// RefineMidpoint handles T-junctions at the band's inner edge (manifold).
	// =============================================================
	{
		// Edge predicate: split an edge if either endpoint is a band vertex.
		struct BandEdgePred {
			const uint8_t* band;
			const CLEAN::Mesh::VertexType* base;
			size_t nv;
			bool operator()(vcg::face::Pos<CLEAN::Mesh::FaceType> ep) const {
				const size_t a = (size_t)(ep.f->V(ep.z) - base);
				const size_t b = (size_t)(ep.f->V1(ep.z) - base);
				return (a < nv && band[a]) || (b < nv && band[b]);
			}
		};

		const int RINGS = MESH_BAND_REFINE_RINGS < 1 ? 1 : MESH_BAND_REFINE_RINGS;
		const int RING_UNSET = 0x3fffffff;
		int totalAddedV = 0, totalAddedF = 0;
		for (int level = 0; level < MESH_BAND_REFINE_LEVELS; ++level) {
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
			const size_t NV = mesh.vert.size();
			if (NV == 0) break;

			std::vector<int> ringDist(NV, RING_UNSET);
			for (auto& f : mesh.face) {
				if (f.IsD()) continue;
				for (int e = 0; e < 3; ++e)
					if (f.IsB(e)) {
						ringDist[(int)vcg::tri::Index(mesh, f.V(e))] = 0;
						ringDist[(int)vcg::tri::Index(mesh, f.V((e + 1) % 3))] = 0;
					}
			}
			for (int k = 0; k < RINGS; ++k) {
				bool any = false;
				for (auto& f : mesh.face) {
					if (f.IsD()) continue;
					int rmin = RING_UNSET;
					for (int i = 0; i < 3; ++i)
						rmin = std::min(rmin, ringDist[(int)vcg::tri::Index(mesh, f.V(i))]);
					if (rmin == RING_UNSET || rmin + 1 > RINGS) continue;
					for (int i = 0; i < 3; ++i) {
						const int vi = (int)vcg::tri::Index(mesh, f.V(i));
						if (ringDist[vi] > rmin + 1) { ringDist[vi] = rmin + 1; any = true; }
					}
				}
				if (!any) break;
			}
			std::vector<uint8_t> band(NV, 0);
			for (size_t v = 0; v < NV; ++v)
				if (ringDist[v] <= RINGS) band[v] = 1;

			BandEdgePred pred{ band.data(), &mesh.vert[0], NV };
			const int vBefore = (int)mesh.vert.size(), fBefore = (int)mesh.face.size();
			vcg::tri::RefineMidpoint<CLEAN::Mesh, BandEdgePred>(mesh, pred, false);
			vcg::tri::Allocator<CLEAN::Mesh>::CompactEveryVector(mesh);
			totalAddedV += (int)mesh.vert.size() - vBefore;
			totalAddedF += (int)mesh.face.size() - fBefore;
		}
		if (totalAddedF > 0) {
			CompactAndRefresh();
			DEBUG("Band refine: subdivided edge band (%d rings, %d levels) -> +%d verts, +%d faces (now %d fn)",
				RINGS, MESH_BAND_REFINE_LEVELS, totalAddedV, totalAddedF, mesh.fn);
		}
		ValidateMesh(mesh, "after band refine", true);
	}
#endif // MESH_BAND_REFINE_ENABLED

#if MESH_EDGE_DILATE_RINGS > 0
	// =============================================================
	// Phase 9.7: Edge dilation / outward apron (enlarge coverage)
	//
	// Extrude the open silhouette OUTWARD to approximate more area than was
	// reconstructed. Per ring: for every open-boundary vertex compute its
	// in-plane outward normal (perpendicular to incident border edges, pointing
	// away from the interior), blur that direction along the boundary to limit
	// folding, offset a new vertex outward at the rim's height, and stitch a
	// triangle strip to the old edge. Apron faces are unobserved -> approximate
	// color at texturing. Modest margins only (concave notches self-intersect
	// for large extensions -> raster-domain dilation is the fold-free route).
	// =============================================================
	{
		std::vector<float> el; el.reserve(60000);
		for (auto& f : mesh.face) {
			if (f.IsD()) continue;
			el.push_back((f.V(1)->cP() - f.V(0)->cP()).Norm());
			if (el.size() >= 60000) break;
		}
		float medE = 0.f;
		if (!el.empty()) {
			std::nth_element(el.begin(), el.begin() + el.size() / 2, el.end());
			medE = el[el.size() / 2];
		}
		const float step = (float(MESH_EDGE_DILATE_STEP_X100) / 100.f) * medE;

		int totalAddedV = 0, totalAddedF = 0;
		if (step > 0.f) for (int ring = 0; ring < MESH_EDGE_DILATE_RINGS; ++ring) {
			vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
			vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);
			const size_t NV = mesh.vert.size();
			if (NV == 0) break;

			std::vector<CLEAN::Mesh::CoordType> outDir(NV, CLEAN::Mesh::CoordType(0, 0, 0));
			std::vector<int> nbr0(NV, -1), nbr1(NV, -1);
			std::vector<uint8_t> nbrCnt(NV, 0), isB(NV, 0);
			std::vector<std::pair<int, int>> bedges;
			for (auto& f : mesh.face) {
				if (f.IsD()) continue;
				for (int e = 0; e < 3; ++e) {
					if (!f.IsB(e)) continue;
					const int a = (int)vcg::tri::Index(mesh, f.V(e));
					const int b = (int)vcg::tri::Index(mesh, f.V((e + 1) % 3));
					const int c = (int)vcg::tri::Index(mesh, f.V((e + 2) % 3));
					bedges.emplace_back(a, b);
					CLEAN::Mesh::CoordType edge = mesh.vert[b].cP() - mesh.vert[a].cP(); edge[2] = 0;
					CLEAN::Mesh::CoordType perp(edge[1], -edge[0], 0);
					const float pn = perp.Norm();
					if (pn <= 1e-12f) continue;
					perp /= pn;
					CLEAN::Mesh::CoordType mid = (mesh.vert[a].cP() + mesh.vert[b].cP()) * 0.5f;
					CLEAN::Mesh::CoordType toC = mesh.vert[c].cP() - mid; toC[2] = 0;
					if (perp * toC > 0) perp = -perp; // point AWAY from interior
					outDir[a] += perp; outDir[b] += perp; isB[a] = isB[b] = 1;
					auto add = [&](int u, int w) {
						if (nbrCnt[u] == 0) { nbr0[u] = w; nbrCnt[u] = 1; }
						else if (nbrCnt[u] == 1) { if (nbr0[u] != w) { nbr1[u] = w; nbrCnt[u] = 2; } }
					};
					add(a, b); add(b, a);
				}
			}
			// blur outward direction along the boundary to reduce concave folding
			for (int s = 0; s < MESH_EDGE_DILATE_DIRSMOOTH; ++s) {
				std::vector<CLEAN::Mesh::CoordType> tmp = outDir;
				for (size_t v = 0; v < NV; ++v)
					if (nbrCnt[v] == 2)
						tmp[v] = outDir[v] + outDir[nbr0[v]] + outDir[nbr1[v]];
				outDir.swap(tmp);
			}
			for (size_t v = 0; v < NV; ++v) {
				if (!isB[v]) continue;
				const float n = outDir[v].Norm();
				if (n > 1e-12f) outDir[v] /= n; else isB[v] = 0;
			}

			std::vector<int> bvList; bvList.reserve(NV);
			std::vector<int> newIdx(NV, -1);
			for (size_t v = 0; v < NV; ++v)
				if (isB[v]) { newIdx[v] = (int)bvList.size(); bvList.push_back((int)v); }
			if (bvList.empty()) break;

			std::vector<CLEAN::Mesh::CoordType> newPos(bvList.size());
			for (size_t k = 0; k < bvList.size(); ++k) {
				const int v = bvList[k];
				CLEAN::Mesh::CoordType p = mesh.vert[v].cP() + outDir[v] * step;
				p[2] = mesh.vert[v].cP()[2]; // keep rim height (flat extrapolation)
				newPos[k] = p;
			}
			const size_t oldNV = NV;
			auto vit = vcg::tri::Allocator<CLEAN::Mesh>::AddVertices(mesh, (int)bvList.size());
			for (size_t k = 0; k < bvList.size(); ++k, ++vit)
				vit->P() = newPos[k];

			// drop edges whose endpoints did not both get an apron vertex
			std::vector<std::pair<int, int>> validEdges;
			validEdges.reserve(bedges.size());
			for (auto& be : bedges)
				if (newIdx[be.first] >= 0 && newIdx[be.second] >= 0)
					validEdges.push_back(be);

			auto fit = vcg::tri::Allocator<CLEAN::Mesh>::AddFaces(mesh, (int)validEdges.size() * 2);
			auto setTri = [&](CLEAN::Mesh::FaceIterator& it, int i0, int i1, int i2) {
				it->V(0) = &mesh.vert[i0]; it->V(1) = &mesh.vert[i1]; it->V(2) = &mesh.vert[i2];
				const CLEAN::Mesh::CoordType nrm =
					(mesh.vert[i1].cP() - mesh.vert[i0].cP()) ^ (mesh.vert[i2].cP() - mesh.vert[i0].cP());
				if (nrm[2] < 0) std::swap(it->V(1), it->V(2)); // force +z (top-facing)
				++it;
			};
			for (auto& be : validEdges) {
				const int a = be.first, b = be.second;
				const int ap = (int)oldNV + newIdx[a];
				const int bp = (int)oldNV + newIdx[b];
				setTri(fit, a, b, bp);
				setTri(fit, a, bp, ap);
			}
			totalAddedV += (int)bvList.size();
			totalAddedF += (int)validEdges.size() * 2;
		}
		if (totalAddedF > 0) {
			CompactAndRefresh();
			DEBUG("Edge dilate: extended boundary outward (%d rings, step %.3g) -> +%d verts, +%d faces (now %d fn)",
				MESH_EDGE_DILATE_RINGS, step, totalAddedV, totalAddedF, mesh.fn);
		}
		ValidateMesh(mesh, "after edge dilate", true);
	}
#endif // MESH_EDGE_DILATE_RINGS > 0

#if MESH_BOUNDARY_SMOOTH_ENABLED
	// =============================================================
	// Phase 10: Boundary-only Taubin smoothing (silhouette polishing)
	//
	// Smooths the ragged open-boundary polyline in place. For each boundary
	// vertex with exactly two border neighbors, target = midpoint of those two
	// neighbors; move toward it by +lambda (shrink) then -mu (inflate) per
	// Taubin pair so the curve smooths WITHOUT net recession. Interior vertices
	// and pinch/junction boundary vertices (>2 border neighbors) are never
	// moved. No topology change -> no Compact needed. A/B via the macro.
	// =============================================================
	{
		vcg::tri::UpdateTopology<CLEAN::Mesh>::FaceFace(mesh);
		vcg::tri::UpdateFlags<CLEAN::Mesh>::FaceBorderFromFF(mesh);

		const size_t NV = mesh.vert.size();
		if (NV > 0) {
			const int RINGS = MESH_BOUNDARY_SMOOTH_RINGS < 1 ? 1 : MESH_BOUNDARY_SMOOTH_RINGS;
			const int RING_UNSET = 0x3fffffff;
			std::vector<int> nbr0(NV, -1), nbr1(NV, -1);
			std::vector<uint8_t> nbrCnt(NV, 0);
			std::vector<int> ringDist(NV, RING_UNSET);

			// ring 0 = open-boundary vertices; also record their two border-curve
			// neighbors (used to keep the silhouette line itself clean).
			for (auto& f : mesh.face) {
				if (f.IsD()) continue;
				for (int e = 0; e < 3; ++e) {
					if (!f.IsB(e)) continue;
					const int a = (int)vcg::tri::Index(mesh, f.V(e));
					const int b = (int)vcg::tri::Index(mesh, f.V((e + 1) % 3));
					ringDist[a] = 0; ringDist[b] = 0;
					auto add = [&](int u, int w) {
						if (nbrCnt[u] == 0) { nbr0[u] = w; nbrCnt[u] = 1; }
						else if (nbrCnt[u] == 1) { if (nbr0[u] != w) { nbr1[u] = w; nbrCnt[u] = 2; } }
						else if (nbrCnt[u] == 2) { if (nbr0[u] != w && nbr1[u] != w) nbrCnt[u] = 3; }
					};
					add(a, b);
					add(b, a);
				}
			}

			// BFS ring distance inward (face propagation, capped at RINGS).
			for (int k = 0; k < RINGS; ++k) {
				bool any = false;
				for (auto& f : mesh.face) {
					if (f.IsD()) continue;
					int rmin = RING_UNSET;
					for (int i = 0; i < 3; ++i)
						rmin = std::min(rmin, ringDist[(int)vcg::tri::Index(mesh, f.V(i))]);
					if (rmin == RING_UNSET || rmin + 1 > RINGS) continue;
					for (int i = 0; i < 3; ++i) {
						const int vi = (int)vcg::tri::Index(mesh, f.V(i));
						if (ringDist[vi] > rmin + 1) { ringDist[vi] = rmin + 1; any = true; }
					}
				}
				if (!any) break;
			}

			// 1-ring adjacency for the inner band (rings 1..RINGS) = umbrella smoothing.
			std::vector<std::vector<int>> adj(NV);
			for (auto& f : mesh.face) {
				if (f.IsD()) continue;
				const int v[3] = {
					(int)vcg::tri::Index(mesh, f.V(0)),
					(int)vcg::tri::Index(mesh, f.V(1)),
					(int)vcg::tri::Index(mesh, f.V(2)) };
				for (int i = 0; i < 3; ++i) {
					const int u = v[i];
					if (ringDist[u] >= 1 && ringDist[u] <= RINGS) {
						adj[u].push_back(v[(i + 1) % 3]);
						adj[u].push_back(v[(i + 2) % 3]);
					}
				}
			}
			for (size_t v = 0; v < NV; ++v)
				if (!adj[v].empty()) {
					std::sort(adj[v].begin(), adj[v].end());
					adj[v].erase(std::unique(adj[v].begin(), adj[v].end()), adj[v].end());
				}

			// movable = boundary-curve vertices (ring 0, exactly 2 border nbrs) OR
			// inner-band vertices (ring 1..RINGS with neighbors). Beyond RINGS fixed.
			auto isCurve = [&](size_t v) { return ringDist[v] == 0 && nbrCnt[v] == 2; };
			auto isBand  = [&](size_t v) { return ringDist[v] >= 1 && ringDist[v] <= RINGS && !adj[v].empty(); };

			const float lambda = float(MESH_BOUNDARY_SMOOTH_LAMBDA_X100) / 100.f;
			const float mu = -float(MESH_BOUNDARY_SMOOTH_MU_X100) / 100.f;
			std::vector<CLEAN::Mesh::CoordType> buf(NV);
			int nCurve = 0, nBand = 0;
			for (size_t v = 0; v < NV; ++v) {
				if (mesh.vert[v].IsD()) continue;
				if (isCurve(v)) ++nCurve;
				else if (isBand(v)) ++nBand;
			}

			for (int it = 0; it < MESH_BOUNDARY_SMOOTH_ITERS * 2; ++it) {
				const float step = (it & 1) ? mu : lambda; // Taubin alternation
				for (size_t v = 0; v < NV; ++v) {
					if (mesh.vert[v].IsD()) continue;
					if (isCurve(v)) {
						const auto& p = mesh.vert[v].cP();
						const auto& pa = mesh.vert[nbr0[v]].cP();
						const auto& pb = mesh.vert[nbr1[v]].cP();
						const CLEAN::Mesh::CoordType avg = (pa + pb) * 0.5f;
						buf[v] = p + (avg - p) * step;
					} else if (isBand(v)) {
						const auto& p = mesh.vert[v].cP();
						CLEAN::Mesh::CoordType sum(0, 0, 0);
						for (int w : adj[v]) sum += mesh.vert[w].cP();
						const CLEAN::Mesh::CoordType avg = sum / (float)adj[v].size();
						buf[v] = p + (avg - p) * step;
					}
				}
				for (size_t v = 0; v < NV; ++v) {
					if (mesh.vert[v].IsD()) continue;
					if (isCurve(v) || isBand(v)) mesh.vert[v].P() = buf[v];
				}
			}
			if (nCurve + nBand > 0)
				DEBUG("Boundary smooth: Taubin-smoothed %d edge + %d band vertices (%d rings, %d iters, lambda=%.2f mu=%.2f)",
					nCurve, nBand, RINGS, MESH_BOUNDARY_SMOOTH_ITERS, lambda, mu);
		}
		ValidateMesh(mesh, "after boundary smooth", true);
	}
#endif // MESH_BOUNDARY_SMOOTH_ENABLED

	// =============================================================
	// Export
	// =============================================================
	ASSERT(vertices.IsEmpty() && faces.IsEmpty());
	vertices.Reserve(mesh.VN());
	vcg::SimpleTempData<CLEAN::Mesh::VertContainer, VIndex> indices(mesh.vert);

	VIndex idx = 0;
	for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
		if (vi->IsD()) continue;
		Vertex& v = vertices.AddEmpty();
		const auto& P = vi->cP();
		v.x = P[0];
		v.y = P[1];
		v.z = P[2];
		indices[vi] = idx++;
	}

	faces.Reserve(mesh.FN());
	for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
		if (fi->IsD()) continue;
		Face& f = faces.AddEmpty();
		f[0] = indices[fi->cV(0)];
		f[1] = indices[fi->cV(1)];
		f[2] = indices[fi->cV(2)];
	}

	DEBUG("Final cleaned mesh: %u vertices, %u faces (%s)",
		vertices.GetSize(), faces.GetSize(), TD_TIMER_GET_FMT().c_str());
}
/*----------------------------------------------------------------*/


// project vertices and compute bounding-box;
// account for differences in pixel center convention: while OpenMVS uses the same convention as OpenCV and DirectX 9 where the center
// of a pixel is defined at integer coordinates, i.e. the center is at (0, 0) and the top left corner is at (-0.5, -0.5),
// DirectX 10+, OpenGL, and Vulkan convention is the center of a pixel is defined at half coordinates, i.e. the center is at (0.5, 0.5)
// and the top left corner is at (0, 0)
static const Mesh::TexCoord halfPixel(0.5f, 0.5f);

// translate, normalize and flip Y axis of the texture coordinates
void Mesh::FaceTexcoordsNormalize(TexCoordArr& newFaceTexcoords, bool flipY) const
{
	ASSERT(!faceTexcoords.empty() && !textureDiffuse.empty());
	const TexCoord invNorm(1.f/(float)textureDiffuse.cols, 1.f/(float)textureDiffuse.rows);
	newFaceTexcoords.resize(faceTexcoords.size());
	if (flipY) {
		FOREACH(i, faceTexcoords) {
			const TexCoord& texcoord = faceTexcoords[i];
			newFaceTexcoords[i] = TexCoord(
				(texcoord.x+halfPixel.x)*invNorm.x,
				1.f-(texcoord.y+halfPixel.y)*invNorm.y
			);
		}
	} else {
		FOREACH(i, faceTexcoords)
			newFaceTexcoords[i] = (faceTexcoords[i]+halfPixel)*invNorm;
	}
} // FaceTexcoordsNormalize

// flip Y axis, unnormalize and translate back texture coordinates
void Mesh::FaceTexcoordsUnnormalize(TexCoordArr& newFaceTexcoords, bool flipY) const
{
	ASSERT(!faceTexcoords.empty() && !textureDiffuse.empty());
	const TexCoord scale((float)textureDiffuse.cols, (float)textureDiffuse.rows);
	newFaceTexcoords.resize(faceTexcoords.size());
		if (flipY) {
			FOREACH(i, faceTexcoords) {
				const TexCoord& texcoord = faceTexcoords[i];
			newFaceTexcoords[i] = TexCoord(
				texcoord.x*scale.x-halfPixel.x,
				(1.f-texcoord.y)*scale.y-halfPixel.y
			);
		}
	} else {
		FOREACH(i, faceTexcoords)
			newFaceTexcoords[i] = faceTexcoords[i]*scale - halfPixel;
	}
} // FaceTexcoordsUnnormalize
/*----------------------------------------------------------------*/


// define a PLY file format composed only of vertices and triangles
namespace BasicPLY {
	// list of property information for a vertex
	static const PLY::PlyProperty vert_props[] = {
		{"x", PLY::Float32, PLY::Float32, offsetof(Mesh::Vertex,x), 0, 0, 0, 0},
		{"y", PLY::Float32, PLY::Float32, offsetof(Mesh::Vertex,y), 0, 0, 0, 0},
		{"z", PLY::Float32, PLY::Float32, offsetof(Mesh::Vertex,z), 0, 0, 0, 0}
	};
	struct VertexNormal {
		Mesh::Vertex v;
		Mesh::Normal n;
	};
	static const PLY::PlyProperty vert_normal_props[] = {
		{ "x", PLY::Float32, PLY::Float32, offsetof(VertexNormal,v.x), 0, 0, 0, 0},
		{ "y", PLY::Float32, PLY::Float32, offsetof(VertexNormal,v.y), 0, 0, 0, 0},
		{ "z", PLY::Float32, PLY::Float32, offsetof(VertexNormal,v.z), 0, 0, 0, 0},
		{"nx", PLY::Float32, PLY::Float32, offsetof(VertexNormal,n.x), 0, 0, 0, 0},
		{"ny", PLY::Float32, PLY::Float32, offsetof(VertexNormal,n.y), 0, 0, 0, 0},
		{"nz", PLY::Float32, PLY::Float32, offsetof(VertexNormal,n.z), 0, 0, 0, 0}
	};
	// list of property information for a face
	struct Face {
		uint8_t num;
		Mesh::Face* pFace;
	};
	static const PLY::PlyProperty face_props[] = {
		{"vertex_indices", PLY::Uint32, PLY::Uint32, offsetof(Face,pFace), 1, PLY::Uint8, PLY::Uint8, offsetof(Face,num)}
	};
	struct TexCoord {
		uint8_t num;
		Mesh::TexCoord* pTex;
	};
	struct FaceTex {
		Face face;
		TexCoord tex;
	};
	static const PLY::PlyProperty face_tex_props[] = {
		{"vertex_indices", PLY::Uint32, PLY::Uint32, offsetof(FaceTex,face.pFace), 1, PLY::Uint8, PLY::Uint8, offsetof(FaceTex,face.num)},
		{"texcoord", PLY::Float32, PLY::Float32, offsetof(FaceTex,tex.pTex), 1, PLY::Uint8, PLY::Uint8, offsetof(FaceTex,tex.num)}
	};
	// vertex carrying a per-vertex texture coordinate (written as standard
	// "s"/"t" properties so any importer reads UVs through its safe
	// per-vertex path instead of scattering per-face texcoords)
	struct VertexTex {
		Mesh::Vertex v;
		Mesh::TexCoord t;
	};
	static const PLY::PlyProperty vert_tex_props[] = {
		{"x", PLY::Float32, PLY::Float32, offsetof(VertexTex,v.x), 0, 0, 0, 0},
		{"y", PLY::Float32, PLY::Float32, offsetof(VertexTex,v.y), 0, 0, 0, 0},
		{"z", PLY::Float32, PLY::Float32, offsetof(VertexTex,v.z), 0, 0, 0, 0},
		{"s", PLY::Float32, PLY::Float32, offsetof(VertexTex,t.x), 0, 0, 0, 0},
		{"t", PLY::Float32, PLY::Float32, offsetof(VertexTex,t.y), 0, 0, 0, 0}
	};
	struct VertexNormalTex {
		Mesh::Vertex v;
		Mesh::Normal n;
		Mesh::TexCoord t;
	};
	static const PLY::PlyProperty vert_normal_tex_props[] = {
		{ "x", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,v.x), 0, 0, 0, 0},
		{ "y", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,v.y), 0, 0, 0, 0},
		{ "z", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,v.z), 0, 0, 0, 0},
		{"nx", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,n.x), 0, 0, 0, 0},
		{"ny", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,n.y), 0, 0, 0, 0},
		{"nz", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,n.z), 0, 0, 0, 0},
		{"s", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,t.x), 0, 0, 0, 0},
		{"t", PLY::Float32, PLY::Float32, offsetof(VertexNormalTex,t.y), 0, 0, 0, 0}
	};
	// list of the kinds of elements in the PLY
	static const char* elem_names[] = {
		"vertex",
		"face"
	};
} // namespace BasicPLY

// import the mesh from the given file
bool Mesh::Load(const String& fileName)
{
	TD_TIMER_STARTD();
	const String ext(Util::getFileExt(fileName).ToLower());
	bool ret;
	if (ext == _T(".obj"))
		ret = LoadOBJ(fileName);
	else
	if (ext == _T(".gltf") || ext == _T(".glb"))
		ret = LoadGLTF(fileName, ext == _T(".glb"));
	else
		ret = LoadPLY(fileName);
	if (!ret)
		return false;
	DEBUG_EXTRA("Mesh loaded: %u vertices, %u faces (%s)", vertices.GetSize(), faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	return true;
}
// import the mesh as a PLY file
bool Mesh::LoadPLY(const String& fileName)
{
	ASSERT(!fileName.IsEmpty());
	Release();

	// open PLY file and read header
	PLY ply;
	if (!ply.read(fileName)) {
		DEBUG_EXTRA("error: invalid PLY file");
		return false;
	}
	for (int i = 0; i < (int)ply.elems.size(); ++i) {
		int elem_count;
		LPCSTR elem_name = ply.setup_element_read(i, &elem_count);
		if (PLY::equal_strings(BasicPLY::elem_names[0], elem_name)) {
			vertices.Resize(elem_count);
		} else
		if (PLY::equal_strings(BasicPLY::elem_names[1], elem_name)) {
			faces.Resize(elem_count);
		}
	}
	if (vertices.IsEmpty() || faces.IsEmpty()) {
		DEBUG_EXTRA("error: invalid mesh file");
		return false;
	}

	// read PLY body
	for (int i = 0; i < (int)ply.elems.size(); i++) {
		int elem_count;
		LPCSTR elem_name = ply.setup_element_read(i, &elem_count);
		if (PLY::equal_strings(BasicPLY::elem_names[0], elem_name)) {
			ASSERT(vertices.size() == (VIndex)elem_count);
			ply.setup_property(BasicPLY::vert_props[0]);
			ply.setup_property(BasicPLY::vert_props[1]);
			ply.setup_property(BasicPLY::vert_props[2]);
			FOREACHPTR(pVert, vertices)
				ply.get_element(pVert);
		} else
		if (PLY::equal_strings(BasicPLY::elem_names[1], elem_name)) {
			ASSERT(faces.size() == (FIndex)elem_count);
			if (ply.find_property(ply.elems[i], BasicPLY::face_tex_props[1].name.c_str()) == -1) {
				// load vertex indices
				BasicPLY::Face face;
				ply.setup_property(BasicPLY::face_props[0]);
				FOREACHPTR(pFace, faces) {
					ply.get_element(&face);
					if (face.num != 3) {
						DEBUG_EXTRA("error: unsupported mesh file (face not triangle)");
						return false;
					}
					memcpy(pFace, face.pFace, sizeof(VIndex)*3);
					delete[] face.pFace;
				}
			} else {
				// load vertex indices and texture coordinates
				faceTexcoords.resize((FIndex)elem_count*3);
				BasicPLY::FaceTex face;
				ply.setup_property(BasicPLY::face_tex_props[0]);
				ply.setup_property(BasicPLY::face_tex_props[1]);
				FOREACH(f, faces) {
					ply.get_element(&face);
					if (face.face.num != 3) {
						DEBUG_EXTRA("error: unsupported mesh file (face not triangle)");
						return false;
					}
					memcpy(faces.data()+f, face.face.pFace, sizeof(VIndex)*3);
					delete[] face.face.pFace;
					if (face.tex.num != 6) {
						DEBUG_EXTRA("error: unsupported mesh file (texture coordinates not per face vertex)");
						return false;
					}
					memcpy(faceTexcoords.data()+f*3, face.tex.pTex, sizeof(TexCoord)*3);
					delete[] face.tex.pTex;
				}
				// load the texture
				for (const std::string& comment: ply.get_comments()) {
					if (_tcsncmp(comment.c_str(), _T("TextureFile "), 12) == 0) {
						const String textureFileName(comment.substr(12));
						textureDiffuse.Load(Util::getFilePath(fileName)+textureFileName);
						break;
					}
				}
				// flip Y axis, unnormalize and translate back texture coordinates
				TexCoordArr unnormFaceTexcoords;
				FaceTexcoordsUnnormalize(unnormFaceTexcoords, true);
				faceTexcoords.Swap(unnormFaceTexcoords);
			}
		} else {
			ply.get_other_element();
		}
	}
	return true;
}
// import the mesh as a OBJ file
bool Mesh::LoadOBJ(const String& fileName)
{
	ASSERT(!fileName.IsEmpty());
	Release();

	// open and parse OBJ file
	ObjModel model;
	if (!model.Load(fileName)) {
		DEBUG_EXTRA("error: invalid OBJ file");
		return false;
	}
	if (model.get_vertices().empty() || model.get_groups().size() != 1 || model.get_groups()[0].faces.empty()) {
		DEBUG_EXTRA("error: invalid mesh file");
		return false;
	}

	// store vertices
	ASSERT(sizeof(ObjModel::Vertex) == sizeof(Vertex));
	ASSERT(model.get_vertices().size() < std::numeric_limits<VIndex>::max());
	vertices.CopyOf(&model.get_vertices()[0], (VIndex)model.get_vertices().size());

	// store vertex normals
	ASSERT(sizeof(ObjModel::Normal) == sizeof(Normal));
	ASSERT(model.get_vertices().size() < std::numeric_limits<VIndex>::max());
	if (!model.get_normals().empty()) {
		ASSERT(model.get_normals().size() == model.get_vertices().size());
		vertexNormals.CopyOf(&model.get_normals()[0], (VIndex)model.get_normals().size());
	}

	// store faces
	const ObjModel::Group& group = model.get_groups()[0];
	ASSERT(group.faces.size() < std::numeric_limits<FIndex>::max());
	faces.Reserve((FIndex)group.faces.size());
	for (const ObjModel::Face& f: group.faces) {
		ASSERT(f.vertices[0] != NO_ID);
		faces.emplace_back(f.vertices[0], f.vertices[1], f.vertices[2]);
		if (f.texcoords[0] != NO_ID) {
			for (int i=0; i<3; ++i)
				faceTexcoords.emplace_back(model.get_texcoords()[f.texcoords[i]]);
		}
		if (f.normals[0] != NO_ID) {
			Normal& n = faceNormals.emplace_back(Normal::ZERO);
			for (int i=0; i<3; ++i)
				n += normalized(model.get_normals()[f.normals[i]]);
			normalize(n);
		}
	}

	// store texture
	ObjModel::MaterialLib::Material* pMaterial(model.GetMaterial(group.material_name));
	if (pMaterial && pMaterial->LoadDiffuseMap())
		cv::swap(textureDiffuse, pMaterial->diffuse_map);
	
	// flip Y axis, unnormalize and translate back texture coordinates
	if (!faceTexcoords.empty()) {
		TexCoordArr unnormFaceTexcoords;
		FaceTexcoordsUnnormalize(unnormFaceTexcoords, true);
		faceTexcoords.Swap(unnormFaceTexcoords);
	}
	return true;
}
// import the mesh as a GLTF file
bool Mesh::LoadGLTF(const String& fileName, bool bBinary)
{
	ASSERT(!fileName.IsEmpty());
	Release();

	// load model
	tinygltf::Model gltfModel; {
		tinygltf::TinyGLTF loader;
		std::string err, warn;
		if (bBinary ?
			!loader.LoadBinaryFromFile(&gltfModel, &err, &warn, fileName) :
			!loader.LoadASCIIFromFile(&gltfModel, &err, &warn, fileName))
			return false;
		if (!err.empty()) {
			VERBOSE("error: %s", err.c_str());
			return false;
		}
		if (!warn.empty())
			DEBUG("warning: %s", warn.c_str());
	}
	
	// parse model
	for (const tinygltf::Mesh& gltfMesh : gltfModel.meshes) {
		for (const tinygltf::Primitive& gltfPrimitive : gltfMesh.primitives) {
			if (gltfPrimitive.mode != TINYGLTF_MODE_TRIANGLES)
				continue;
			Mesh mesh;
			// read vertices
			{
				const tinygltf::Accessor& gltfAccessor = gltfModel.accessors[gltfPrimitive.attributes.at("POSITION")];
				if (gltfAccessor.type != TINYGLTF_TYPE_VEC3)
					continue;
				const tinygltf::BufferView& gltfBufferView = gltfModel.bufferViews[gltfAccessor.bufferView];
				const tinygltf::Buffer& buffer = gltfModel.buffers[gltfBufferView.buffer];
				const uint8_t* pData = buffer.data.data() + gltfBufferView.byteOffset + gltfAccessor.byteOffset;
				mesh.vertices.resize((VIndex)gltfAccessor.count);
				if (gltfAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
					ASSERT(gltfBufferView.byteLength == sizeof(Vertex) * gltfAccessor.count);
					memcpy(mesh.vertices.data(), pData, gltfBufferView.byteLength);
				}
				else if (gltfAccessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE) {
					for (VIndex i = 0; i < gltfAccessor.count; ++i)
						mesh.vertices[i] = ((const Point3d*)pData)[i];
				}
				else {
					VERBOSE("error: unsupported vertices (component type)");
					continue;
				}
			}
			// read faces
			{
				const tinygltf::Accessor& gltfAccessor = gltfModel.accessors[gltfPrimitive.indices];
				if (gltfAccessor.type != TINYGLTF_TYPE_SCALAR)
					continue;
				const tinygltf::BufferView& gltfBufferView = gltfModel.bufferViews[gltfAccessor.bufferView];
				const tinygltf::Buffer& buffer = gltfModel.buffers[gltfBufferView.buffer];
				const uint8_t* pData = buffer.data.data() + gltfBufferView.byteOffset + gltfAccessor.byteOffset;
				mesh.faces.resize((FIndex)(gltfAccessor.count/3));
				if (gltfAccessor.componentType == TINYGLTF_COMPONENT_TYPE_INT ||
					gltfAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
					ASSERT(gltfBufferView.byteLength == sizeof(uint32_t) * gltfAccessor.count);
					memcpy(mesh.faces.data(), pData, gltfBufferView.byteLength);
				}
				else {
					VERBOSE("error: unsupported faces (component type)");
					continue;
				}
			}
			Join(mesh);
		}
	}
	return true;
} // Load
/*----------------------------------------------------------------*/


void Mesh::ValidateFaceFaces() const
{
	HARD_ASSERT(!faceFaces.empty());
	const size_t n = faces.size();

	for (FIndex f = 0; f < n; ++f)
	{
		for (int e = 0; e < 3; ++e)
		{
			const FIndex g = faceFaces[f][e];
			if (g == NO_ID)
				continue;

			HARD_ASSERT(g < n);

			bool foundBack = false;
			for (int k = 0; k < 3; ++k)
			{
				if (faceFaces[g][k] == f)
				{
					foundBack = true;
					break;
				}
			}

			if (!foundBack)
			{
				std::cout << "Asymmetry: face " << f
					<< " thinks neighbor is " << g
					<< " but reverse not found\n";
				DEBUG_BREAK();
			}
		}
	}
}

void Mesh::ValidateEdgeConsistency() const
{
	HARD_ASSERT(!faces.empty());
	HARD_ASSERT(!faceFaces.empty());
	const size_t n = faces.size();

	for (FIndex f = 0; f < n; ++f)
	{
		const Face& F = faces[f];

		for (int e = 0; e < 3; ++e)
		{
			const FIndex g = faceFaces[f][e];
			if (g == NO_ID)
				continue;

			const Face& G = faces[g];

			int shared = 0;
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					if (F[i] == G[j])
						++shared;

			if (shared != 2)
			{
				std::cout << "Invalid adjacency:\n";
				std::cout << "Face " << f << " : "
					<< F[0] << "," << F[1] << "," << F[2] << "\n";
				std::cout << "Face " << g << " : "
					<< G[0] << "," << G[1] << "," << G[2] << "\n";
				std::cout << "Shared = " << shared << "\n";
				DEBUG_BREAK();
			}
		}
	}
}

void Mesh::ValidateVertexFacesSorted() const
{
	HARD_ASSERT(!vertexFaces.empty());
	for (size_t v = 0; v < vertexFaces.size(); ++v)
	{
		const auto& vf = vertexFaces[v];
		for (size_t i = 1; i < vf.size(); ++i)
		{
			if (vf[i - 1] > vf[i])
			{
				std::cout << "vertexFaces not sorted at vertex " << v << "\n";
				DEBUG_BREAK();
			}
		}
	}
}

// export the mesh to the given file
bool Mesh::Save(const String& fileName, const cList<String>& comments, bool bBinary) const
{
	// Meshes are not guaranteed to be in a consistent state.  This is the responsibility
	// of the caller.

	TD_TIMER_STARTD();
	const String ext(Util::getFileExt(fileName).ToLower());
	bool ret;
	if (ext == _T(".obj"))
		ret = SaveOBJ(fileName);
	else
	if (ext == _T(".gltf") || ext == _T(".glb"))
		ret = SaveGLTF(fileName, ext == _T(".glb"));
	else
	if (ext == _T(".gmmesh"))
		ret = SaveGMMesh(fileName);
	else
		ret = SavePLY(ext != _T(".ply") ? String(fileName+_T(".ply")) : fileName, comments, bBinary);
	if (!ret)
		return false;
	DEBUG_EXTRA("Mesh saved: %u vertices, %u faces (%s)", vertices.GetSize(), faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	return true;
}
// export the mesh as a PLY file
bool Mesh::SavePLY(const String& fileName, const cList<String>& comments, bool bBinary) const
{
	ASSERT(!fileName.empty());
	Util::ensureFolder(fileName);

	// create PLY object
	const size_t bufferSize(vertices.size()*(4*3/*pos*/+2/*eol*/) + faces.size()*(1*1/*len*/+4*3/*idx*/+2/*eol*/) + 2048/*extra size*/);
	PLY ply;
	if (!ply.write(fileName, 2, BasicPLY::elem_names, bBinary?PLY::BINARY_LE:PLY::ASCII, bufferSize)) {
		DEBUG_EXTRA("error: can not create the mesh file");
		return false;
	}

	// export comments
	FOREACHPTR(pStr, comments)
		ply.append_comment(pStr->c_str());

	// export texture file name as comment if needed
	String textureFileName;
	if (!faceTexcoords.empty() && !textureDiffuse.empty()) {
		// Use JPEG (fast TurboJPEG encode) rather than PNG (zlib level 6),
		// which is catastrophically slow on multi-gigapixel texture atlases.
		textureFileName = Util::getFileFullName(fileName)+_T(".jpg");
		ply.append_comment((_T("TextureFile ")+Util::getFileNameExt(textureFileName)).c_str());
	}

	if (faceTexcoords.empty()) {
		// no texture coordinates: export plain vertices then faces
		if (vertexNormals.empty()) {
			// describe what properties go into the vertex elements
			ply.describe_property(BasicPLY::elem_names[0], 3, BasicPLY::vert_props);

			// export the array of vertices
			FOREACHPTR(pVert, vertices)
				ply.put_element(pVert);
		} else {
			ASSERT(vertices.size() == vertexNormals.size());

			// describe what properties go into the vertex elements
			ply.describe_property(BasicPLY::elem_names[0], 6, BasicPLY::vert_normal_props);

			// export the array of vertices
			BasicPLY::VertexNormal vn;
			FOREACH(i, vertices) {
				vn.v = vertices[i];
				vn.n = vertexNormals[i];
				ply.put_element(&vn);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// describe what properties go into the face elements
		ply.describe_property(BasicPLY::elem_names[1], 1, BasicPLY::face_props);

		// export the array of faces
		BasicPLY::Face face = {3};
		FOREACHPTR(pFace, faces) {
			face.pFace = pFace;
			ply.put_element(&face);
		}
	} else {
		ASSERT(faceTexcoords.size() == faces.size()*3);

		// OpenMVS stores texture coordinates per face-corner (3 per triangle).
		// Exporting them as the PLY face "texcoord" list forces importers to
		// scatter per-wedge UVs into a per-vertex array, which loses UVs at
		// chart seams (last-writer-wins) and, after primitive-type splitting,
		// can desync from the vertex count and crash the consumer. Instead we
		// split shared vertices at UV seams (so each unique vertex/UV pair is
		// its own vertex) and write standard per-vertex "s"/"t" properties,
		// matching what the OBJ exporter effectively produces.

		// translate, normalize and flip Y axis of the texture coordinates
		TexCoordArr normFaceTexcoords;
		FaceTexcoordsNormalize(normFaceTexcoords, true);

		// build vertex-split arrays and re-indexed faces
		const bool bNormals = !vertexNormals.empty();
		ASSERT(!bNormals || vertices.size() == vertexNormals.size());
		std::vector< std::vector<std::pair<TexCoord,VIndex>> > vertUVMap(vertices.size());
		std::vector<Vertex> splitVerts;
		std::vector<Normal> splitNorms;
		std::vector<TexCoord> splitUVs;
		std::vector<Face> splitFaces(faces.size());
		splitVerts.reserve(vertices.size());
		splitUVs.reserve(vertices.size());
		if (bNormals)
			splitNorms.reserve(vertices.size());
		FOREACH(f, faces) {
			for (int i = 0; i < 3; ++i) {
				const VIndex ov = faces[f][i];
				const TexCoord& uv = normFaceTexcoords[f*3+i];
				VIndex ni = NO_ID;
				for (const std::pair<TexCoord,VIndex>& it : vertUVMap[ov]) {
					if (it.first.x == uv.x && it.first.y == uv.y) {
						ni = it.second;
						break;
					}
				}
				if (ni == NO_ID) {
					ni = (VIndex)splitVerts.size();
					splitVerts.push_back(vertices[ov]);
					splitUVs.push_back(uv);
					if (bNormals)
						splitNorms.push_back(vertexNormals[ov]);
					vertUVMap[ov].emplace_back(uv, ni);
				}
				splitFaces[f][i] = ni;
			}
		}

		// export the split vertices with per-vertex texture coordinates
		if (!bNormals) {
			ply.describe_property(BasicPLY::elem_names[0], 5, BasicPLY::vert_tex_props);
			BasicPLY::VertexTex vt;
			FOREACH(i, splitVerts) {
				vt.v = splitVerts[i];
				vt.t = splitUVs[i];
				ply.put_element(&vt);
			}
		} else {
			ply.describe_property(BasicPLY::elem_names[0], 8, BasicPLY::vert_normal_tex_props);
			BasicPLY::VertexNormalTex vnt;
			FOREACH(i, splitVerts) {
				vnt.v = splitVerts[i];
				vnt.n = splitNorms[i];
				vnt.t = splitUVs[i];
				ply.put_element(&vnt);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// export the array of faces (vertex indices only)
		ply.describe_property(BasicPLY::elem_names[1], 1, BasicPLY::face_props);
		BasicPLY::Face face = {3};
		FOREACH(f, splitFaces) {
			face.pFace = splitFaces.data()+f;
			ply.put_element(&face);
		}

		// export the texture
		if (!textureDiffuse.empty())
			textureDiffuse.Save(textureFileName);
	}
	if (ply.get_current_element_count() == 0)
		return false;

	// write to file
	return ply.header_complete();
}
// write the diffuse texture as a Global Mapper native multi-strip JPEG
// container (.gmtex). A single multi-gigapixel atlas JPEG can only be
// decoded single-threaded by libjpeg-turbo, which dominates Global
// Mapper's mesh-load time. Splitting the atlas into N independent
// horizontal-strip JPEGs lets the reader decode one strip per core.
//
// Container layout (all little-endian):
//   char   magic[4]   = "GMTX"
//   uint32 version    = 1
//   uint32 width
//   uint32 height
//   uint32 numStrips
//   uint32 flags      (reserved, 0)
//   { uint32 stripHeight; uint32 jpegLen; } [numStrips]   // top-to-bottom
//   <numStrips JPEG blobs concatenated, in strip order>
static bool SaveTextureStripsGMTex(const Image8U3& image, const String& fileName)
{
	const int W = image.cols;
	const int H = image.rows;
	if (W <= 0 || H <= 0)
		return false;

	// Aim for ~512 rows per strip, capped at 64 strips, so a tall atlas
	// exposes enough independent JPEGs to saturate the reader's cores
	// while keeping per-strip JPEG-header overhead negligible.
	int numStrips = (H + 511) / 512;
	if (numStrips < 1)   numStrips = 1;
	if (numStrips > 64)  numStrips = 64;

	// Even split, remainder spread over the first strips.
	std::vector<int> stripHeight(numStrips), topRow(numStrips);
	const int baseH = H / numStrips;
	const int remH  = H % numStrips;
	int y = 0;
	for (int i = 0; i < numStrips; ++i) {
		stripHeight[i] = baseH + (i < remH ? 1 : 0);
		topRow[i] = y;
		y += stripHeight[i];
	}
	ASSERT(y == H);

	// Encode each horizontal strip to an in-memory JPEG in parallel.
	std::vector<std::vector<uint8_t>> blobs(numStrips);
	std::vector<int> ok(numStrips, 0);
	const std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 90 };
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < numStrips; ++i) {
		const cv::Mat strip = image.rowRange(topRow[i], topRow[i] + stripHeight[i]);
		try {
			ok[i] = cv::imencode(_T(".jpg"), strip, blobs[i], params) ? 1 : 0;
		} catch (...) {
			ok[i] = 0;
		}
	}
	for (int i = 0; i < numStrips; ++i) {
		if (!ok[i]) {
			DEBUG_EXTRA("error: failed to JPEG-encode .gmtex strip %d/%d", i, numStrips);
			return false;
		}
	}

	// Write the container.
	File f(fileName, File::WRITE, File::CREATE | File::TRUNCATE);
	if (!f.isOpen()) {
		DEBUG_EXTRA("error: can not create the .gmtex file");
		return false;
	}
	const uint32_t version  = 1;
	const uint32_t width    = (uint32_t)W;
	const uint32_t height   = (uint32_t)H;
	const uint32_t nStrips  = (uint32_t)numStrips;
	const uint32_t flags    = 0;
	f.write("GMTX", 4);
	f.write(&version, sizeof(version));
	f.write(&width,   sizeof(width));
	f.write(&height,  sizeof(height));
	f.write(&nStrips, sizeof(nStrips));
	f.write(&flags,   sizeof(flags));
	for (int i = 0; i < numStrips; ++i) {
		const uint32_t sh  = (uint32_t)stripHeight[i];
		const uint32_t len = (uint32_t)blobs[i].size();
		f.write(&sh,  sizeof(sh));
		f.write(&len, sizeof(len));
	}
	for (int i = 0; i < numStrips; ++i)
		f.write(blobs[i].data(), blobs[i].size());
	return true;
}
// export the mesh as a Global Mapper native binary mesh (.gmmesh)
//
// Layout (all little-endian; Windows x86/x64 native byte order):
//   char    magic[4]            = "GMM1"
//   uint32  version             = 1
//   uint32  flags               bit0=normals, bit1=texCoords, bit2=vertexColors
//   uint32  textureNameLen      UTF-8 byte length, no NUL
//   uint64  numVertices
//   uint64  numFaces
//   char    textureName[textureNameLen]   UTF-8, relative to mesh file
//   float   vertices[3 * numVertices]      x,y,z
//   float   normals[3 * numVertices]       (only if bit0)
//   float   texCoords[2 * numVertices]     u,v  (only if bit1)
//   float   vertexColors[4 * numVertices]  r,g,b,a 0..1 (only if bit2; not emitted here)
//   uint32  faces[3 * numFaces]            vertex indices
//
// Mirrors the SavePLY texture path: OpenMVS stores UVs per face-corner, so we
// split shared vertices at UV seams to produce a clean per-vertex UV array that
// the Global Mapper reader consumes directly (no Assimp parse).
bool Mesh::SaveGMMesh(const String& fileName) const
{
	ASSERT(!fileName.empty());
	Util::ensureFolder(fileName);

	// these packed layouts are what the GM reader assumes (bulk fwrite)
	static_assert(sizeof(Vertex) == 3*sizeof(float), "Vertex must be 3 packed floats");
	static_assert(sizeof(Normal) == 3*sizeof(float), "Normal must be 3 packed floats");
	static_assert(sizeof(TexCoord) == 2*sizeof(float), "TexCoord must be 2 packed floats");
	static_assert(sizeof(Face) == 3*sizeof(uint32_t), "Face must be 3 packed uint32");

	const bool bTexture = !faceTexcoords.empty() && !textureDiffuse.empty();
	const bool bNormals = !vertexNormals.empty();

	// vertex/face arrays actually written (split at UV seams when textured)
	const Vertex* pVerts;
	const Normal* pNorms;
	const TexCoord* pUVs;
	const Face* pFaces;
	uint64_t numVertices, numFaces;

	std::vector<Vertex> splitVerts;
	std::vector<Normal> splitNorms;
	std::vector<TexCoord> splitUVs;
	std::vector<Face> splitFaces;

	String textureFileName;
	if (bTexture) {
		// translate, normalize and flip Y axis of the texture coordinates
		// (same convention the PLY exporter / Assimp path produced)
		TexCoordArr normFaceTexcoords;
		FaceTexcoordsNormalize(normFaceTexcoords, true);
		ASSERT(normFaceTexcoords.size() == faces.size()*3);
		ASSERT(!bNormals || vertices.size() == vertexNormals.size());

		std::vector< std::vector<std::pair<TexCoord,VIndex>> > vertUVMap(vertices.size());
		splitVerts.reserve(vertices.size());
		splitUVs.reserve(vertices.size());
		if (bNormals)
			splitNorms.reserve(vertices.size());
		splitFaces.resize(faces.size());
		FOREACH(f, faces) {
			for (int i = 0; i < 3; ++i) {
				const VIndex ov = faces[f][i];
				const TexCoord& uv = normFaceTexcoords[f*3+i];
				VIndex ni = NO_ID;
				for (const std::pair<TexCoord,VIndex>& it : vertUVMap[ov]) {
					if (it.first.x == uv.x && it.first.y == uv.y) {
						ni = it.second;
						break;
					}
				}
				if (ni == NO_ID) {
					ni = (VIndex)splitVerts.size();
					splitVerts.push_back(vertices[ov]);
					splitUVs.push_back(uv);
					if (bNormals)
						splitNorms.push_back(vertexNormals[ov]);
					vertUVMap[ov].emplace_back(uv, ni);
				}
				splitFaces[f][i] = ni;
			}
		}
		pVerts = splitVerts.data();
		pNorms = bNormals ? splitNorms.data() : NULL;
		pUVs   = splitUVs.data();
		pFaces = splitFaces.data();
		numVertices = splitVerts.size();
		numFaces    = splitFaces.size();

		// reference (and, if needed, write) the diffuse texture alongside the
		// mesh, matching the PLY exporter's JPEG naming so a co-located .ply
		// save can share the same image instead of re-encoding it.
		textureFileName = Util::getFileFullName(fileName)+_T(".jpg");
		if (!File::isFile(textureFileName))
			textureDiffuse.Save(textureFileName);

		// Also emit a multi-strip container (<base>.gmtex) beside the .jpg.
		// Global Mapper prefers it and decodes the strips in parallel; the
		// .jpg remains for the .ply fallback and as a graceful degrade path.
		SaveTextureStripsGMTex(textureDiffuse, Util::getFileFullName(fileName)+_T(".gmtex"));
	} else {
		pVerts = vertices.data();
		pNorms = bNormals ? vertexNormals.data() : NULL;
		pUVs   = NULL;
		pFaces = faces.data();
		numVertices = vertices.size();
		numFaces    = faces.size();
	}

	if (numVertices == 0 || numFaces == 0) {
		DEBUG_EXTRA("error: refusing to write empty .gmmesh file");
		return false;
	}

	// open the output file (buffered, truncating)
	File f(fileName, File::WRITE, File::CREATE | File::TRUNCATE);
	if (!f.isOpen()) {
		DEBUG_EXTRA("error: can not create the mesh file");
		return false;
	}

	// header
	const uint32_t version = 1;
	uint32_t flags = 0;
	if (pNorms) flags |= 0x1;
	if (pUVs)   flags |= 0x2;
	// vertex colors (bit2) are not exported by the texturing pipeline

	// store texture name relative to the mesh file (name + extension only)
	const String texName(textureFileName.empty() ? String() : Util::getFileNameExt(textureFileName));
	const uint32_t textureNameLen = (uint32_t)texName.size();

	f.write("GMM1", 4);
	f.write(&version, sizeof(version));
	f.write(&flags, sizeof(flags));
	f.write(&textureNameLen, sizeof(textureNameLen));
	f.write(&numVertices, sizeof(numVertices));
	f.write(&numFaces, sizeof(numFaces));
	if (textureNameLen)
		f.write(texName.c_str(), textureNameLen);

	// vertex positions
	f.write(pVerts, (size_t)numVertices*sizeof(Vertex));
	// optional per-vertex normals
	if (pNorms)
		f.write(pNorms, (size_t)numVertices*sizeof(Normal));
	// optional per-vertex texture coordinates
	if (pUVs)
		f.write(pUVs, (size_t)numVertices*sizeof(TexCoord));
	// face indices
	f.write(pFaces, (size_t)numFaces*sizeof(Face));

	return true;
}
// export the mesh as a OBJ file
bool Mesh::SaveOBJ(const String& fileName) const
{
	ASSERT(!fileName.empty());
	Util::ensureFolder(fileName);

	// create the OBJ model
	ObjModel model;

	// store vertices
	ASSERT(sizeof(ObjModel::Vertex) == sizeof(Vertex));
	model.get_vertices().insert(model.get_vertices().begin(), vertices.begin(), vertices.end());

	// store vertex normals
	ASSERT(sizeof(ObjModel::Normal) == sizeof(Normal));
	ASSERT(model.get_vertices().size() < std::numeric_limits<VIndex>::max());
	if (!vertexNormals.empty()) {
		ASSERT(vertexNormals.size() == vertices.size());
		model.get_normals().insert(model.get_normals().begin(), vertexNormals.begin(), vertexNormals.end());
	}

	// store face texture coordinates
	ASSERT(sizeof(ObjModel::TexCoord) == sizeof(TexCoord));
	if (!faceTexcoords.empty()) {
		// translate, normalize and flip Y axis of the texture coordinates
		TexCoordArr normFaceTexcoords;
		FaceTexcoordsNormalize(normFaceTexcoords, true);
		ASSERT(normFaceTexcoords.size() == faces.size()*3);
		model.get_texcoords().insert(model.get_texcoords().begin(), normFaceTexcoords.begin(), normFaceTexcoords.end());
	}

	// store faces
	ObjModel::Group& group = model.AddGroup(_T("material_0"));
	group.faces.reserve(faces.size());
	FOREACH(idxFace, faces) {
		const Face& face = faces[idxFace];
		ObjModel::Face f;
		memset(&f, 0xFF, sizeof(ObjModel::Face));
		for (int i=0; i<3; ++i) {
			f.vertices[i] = face[i];
			if (!faceTexcoords.empty())
				f.texcoords[i] = idxFace*3+i;
			if (!vertexNormals.empty())
				f.normals[i] = face[i];
		}
		group.faces.push_back(f);
	}

	// store texture
	ObjModel::MaterialLib::Material* pMaterial(model.GetMaterial(group.material_name));
	ASSERT(pMaterial != NULL);
	pMaterial->diffuse_map = textureDiffuse;

	return model.Save(fileName);
}
// export the mesh as a GLTF file
template <typename T>
void ExtendBufferGLTF(const T* src, size_t size, tinygltf::Buffer& dst, size_t& byte_offset, size_t& byte_length) {
	byte_offset = dst.data.size();
	byte_length = sizeof(T) * size;
	byte_length = ((byte_length + 3) / 4) * 4;
	dst.data.resize(byte_offset + byte_length);
	memcpy(&dst.data[byte_offset], &src[0], byte_length);
}
bool Mesh::SaveGLTF(const String& fileName, bool bBinary) const
{
	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);

	// store a copy of the mesh if it has texture, in order to convert
	// the texture coordinates from per face to per vertex
	Mesh meshCompressed;
	if (HasTexture())
		ConvertTexturePerVertex(meshCompressed);
	const Mesh& mesh(HasTexture() ? meshCompressed : *this);

	// create GLTF model
	tinygltf::Model gltfModel;
	tinygltf::Scene gltfScene;
	tinygltf::Mesh gltfMesh;
	tinygltf::Primitive gltfPrimitive;
	tinygltf::Buffer gltfBuffer;
	gltfScene.name = "scene";
	gltfMesh.name = "mesh";

	// setup vertices
	{
		STATIC_ASSERT(3 * sizeof(Vertex::Type) == sizeof(Vertex)); // VertexArr should be continuous
		const Box box(GetAABB());
		gltfPrimitive.attributes["POSITION"] = (int)gltfModel.accessors.size();
		tinygltf::Accessor vertexPositionAccessor;
		vertexPositionAccessor.name = "vertexPositionAccessor";
		vertexPositionAccessor.bufferView = (int)gltfModel.bufferViews.size();
		vertexPositionAccessor.type = TINYGLTF_TYPE_VEC3;
		vertexPositionAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
		vertexPositionAccessor.count = mesh.vertices.size();
		vertexPositionAccessor.minValues = {box.ptMin.x(), box.ptMin.y(), box.ptMin.z()};
		vertexPositionAccessor.maxValues = {box.ptMax.x(), box.ptMax.y(), box.ptMax.z()};
		gltfModel.accessors.emplace_back(std::move(vertexPositionAccessor));
		// setup vertices buffer
		tinygltf::BufferView vertexPositionBufferView;
		vertexPositionBufferView.name = "vertexPositionBufferView";
		vertexPositionBufferView.buffer = (int)gltfModel.buffers.size();
		ExtendBufferGLTF(mesh.vertices.data(), mesh.vertices.size(), gltfBuffer,
			vertexPositionBufferView.byteOffset, vertexPositionBufferView.byteLength);
		gltfModel.bufferViews.emplace_back(std::move(vertexPositionBufferView));
	}

	// setup faces
	{
		STATIC_ASSERT(3 * sizeof(Face::Type) == sizeof(Face)); // FaceArr should be continuous
		gltfPrimitive.indices = (int)gltfModel.accessors.size();
		tinygltf::Accessor triangleAccessor;
		triangleAccessor.name = "triangleAccessor";
		triangleAccessor.bufferView = (int)gltfModel.bufferViews.size();
		triangleAccessor.type = TINYGLTF_TYPE_SCALAR;
		triangleAccessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
		triangleAccessor.count = mesh.faces.size() * 3;
		gltfModel.accessors.emplace_back(std::move(triangleAccessor));
		// setup triangles buffer
		tinygltf::BufferView triangleBufferView;
		triangleBufferView.name = "triangleBufferView";
		triangleBufferView.buffer = (int)gltfModel.buffers.size();
		ExtendBufferGLTF(mesh.faces.data(), mesh.faces.size(), gltfBuffer,
			triangleBufferView.byteOffset, triangleBufferView.byteLength);
		gltfModel.bufferViews.emplace_back(std::move(triangleBufferView));
		gltfPrimitive.mode = TINYGLTF_MODE_TRIANGLES;
	}

	// setup material
	gltfPrimitive.material = (int)gltfModel.materials.size();
	tinygltf::Material gltfMaterial;
	gltfMaterial.name = "material";
	gltfMaterial.doubleSided = true;
	if (mesh.HasTexture()) {
		// setup texture
		gltfMaterial.emissiveFactor = std::vector<double>{0,0,0};
		gltfMaterial.pbrMetallicRoughness.baseColorTexture.index = (int)gltfModel.textures.size();
		gltfMaterial.pbrMetallicRoughness.baseColorTexture.texCoord = 0;
		gltfMaterial.pbrMetallicRoughness.baseColorFactor = std::vector<double>{1,1,1,1};
		gltfMaterial.pbrMetallicRoughness.metallicFactor = 0;
		gltfMaterial.pbrMetallicRoughness.roughnessFactor = 1;
		gltfMaterial.extensions = {{"KHR_materials_unlit", {}}};
		gltfModel.extensionsUsed = {"KHR_materials_unlit"};
		// setup texture coordinates accessor
		gltfPrimitive.attributes["TEXCOORD_0"] = (int)gltfModel.accessors.size();
		tinygltf::Accessor vertexTexcoordAccessor;
		vertexTexcoordAccessor.name = "vertexTexcoordAccessor";
		vertexTexcoordAccessor.bufferView = (int)gltfModel.bufferViews.size();
		vertexTexcoordAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
		vertexTexcoordAccessor.count = mesh.faceTexcoords.size();
		vertexTexcoordAccessor.type = TINYGLTF_TYPE_VEC2;
		gltfModel.accessors.emplace_back(std::move(vertexTexcoordAccessor));
		// setup texture coordinates
		STATIC_ASSERT(2 * sizeof(TexCoord::Type) == sizeof(TexCoord)); // TexCoordArr should be continuous
		ASSERT(mesh.vertices.size() == mesh.faceTexcoords.size());
		tinygltf::BufferView vertexTexcoordBufferView;
		vertexTexcoordBufferView.name = "vertexTexcoordBufferView";
		vertexTexcoordBufferView.buffer = (int)gltfModel.buffers.size();
		TexCoordArr normFaceTexcoords;
		mesh.FaceTexcoordsNormalize(normFaceTexcoords, false);
		ExtendBufferGLTF(normFaceTexcoords.data(), normFaceTexcoords.size(), gltfBuffer,
			vertexTexcoordBufferView.byteOffset, vertexTexcoordBufferView.byteLength);
		gltfModel.bufferViews.emplace_back(std::move(vertexTexcoordBufferView));
		// setup texture
		tinygltf::Texture texture;
		texture.name = "texture";
		texture.source = (int)gltfModel.images.size();
		texture.sampler = (int)gltfModel.samplers.size();
		gltfModel.textures.emplace_back(std::move(texture));
		// setup texture image
		tinygltf::Image image;
		image.name = Util::getFileFullName(fileName);
		image.width = mesh.textureDiffuse.cols;
		image.height = mesh.textureDiffuse.rows;
		image.component = 3;
		image.bits = 8;
		image.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
		image.mimeType = "image/png";
		image.image.resize(mesh.textureDiffuse.size().area() * 3);
		mesh.textureDiffuse.copyTo(cv::Mat(mesh.textureDiffuse.size(), CV_8UC3, image.image.data()));
		gltfModel.images.emplace_back(std::move(image));
		// setup texture sampler
		tinygltf::Sampler sampler;
		sampler.name = "sampler";
		sampler.minFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
		sampler.magFilter = TINYGLTF_TEXTURE_FILTER_LINEAR;
		sampler.wrapS = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE;
		sampler.wrapT = TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE;
		gltfModel.samplers.emplace_back(std::move(sampler));
	}
	gltfModel.materials.emplace_back(std::move(gltfMaterial));
	gltfModel.buffers.emplace_back(std::move(gltfBuffer));
	gltfMesh.primitives.emplace_back(std::move(gltfPrimitive));

	// setup scene node
	gltfScene.nodes.emplace_back((int)gltfModel.nodes.size());
	tinygltf::Node node;
	node.name = "node";
	node.mesh = (int)gltfModel.meshes.size();
	gltfModel.nodes.emplace_back(std::move(node));
	gltfModel.meshes.emplace_back(std::move(gltfMesh));
	gltfModel.scenes.emplace_back(std::move(gltfScene));
	gltfModel.asset.generator = "OpenMVS";
	gltfModel.asset.version = "2.0";
	gltfModel.defaultScene = 0;

	// setup GLTF
	struct Tools {
		static bool WriteImageData(const std::string *basepath, const std::string *filename,
			tinygltf::Image *image, bool embedImages, void *) {
			ASSERT(!embedImages);
			image->uri = Util::isFullPath(filename->c_str()) ?
				Util::getRelativePath(*basepath, *filename) : String(*filename);
			String basePath(*basepath);
			return cv::imwrite(
				Util::ensureFolderSlash(basePath) + image->uri,
				cv::Mat(image->height, image->width, CV_8UC3, image->image.data()));
		}
	};
	tinygltf::TinyGLTF gltf;
	gltf.SetImageWriter(Tools::WriteImageData, NULL);
	const bool bEmbedImages(false), bEmbedBuffers(true), bPrettyPrint(true);
	return gltf.WriteGltfSceneToFile(&gltfModel, fileName, bEmbedImages, bEmbedBuffers, bPrettyPrint, bBinary);
} // Save
/*----------------------------------------------------------------*/

bool Mesh::Save(const FacesChunkArr& chunks, const String& fileName, const cList<String>& comments, bool bBinary) const
{
	if (chunks.size() < 2)
		return Save(fileName, comments, bBinary);
	FOREACH(i, chunks) {
		const Mesh mesh(SubMesh(chunks[i].faces));
		if (!mesh.Save(Util::insertBeforeFileExt(fileName, String::FormatString("_chunk%02u", i)), comments, bBinary))
			return false;
	}
	return true;
}

bool Mesh::Save(const VertexArr& vertices, const String& fileName, bool bBinary)
{
	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);

	// create PLY object
	const size_t bufferSize(vertices.GetSize()*(4*3/*pos*/+2/*eol*/) + 2048/*extra size*/);
	PLY ply;
	if (!ply.write(fileName, 1, BasicPLY::elem_names, bBinary?PLY::BINARY_LE:PLY::ASCII, bufferSize)) {
		DEBUG_EXTRA("error: can not create the mesh file");
		return false;
	}

	// describe what properties go into the vertex elements
	ply.describe_property(BasicPLY::elem_names[0], 3, BasicPLY::vert_props);

	// export the array of vertices
	FOREACHPTR(pVert, vertices)
		ply.put_element(pVert);
	if (ply.get_current_element_count() == 0)
		return false;

	// write to file
	return ply.header_complete();
}
/*----------------------------------------------------------------*/



// Ensure edge size and improve vertex valence;
// inspired by TransforMesh library of Andrei Zaharescu (cooperz@gmail.com)
// Code: https://scm.gforge.inria.fr/anonscm/svn/mvviewer
// Paper: http://perception.inrialpes.fr/Publications/2007/ZBH07/

#include <CGAL/Simple_cartesian.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <CGAL/Inverse_index.h>

#define CURVATURE_TH 0 // 0.1
#define ROBUST_NORMALS 0 // 4
#define ENSURE_MIN_AREA 0 // 2
#define SPLIT_BORDER_EDGES 1
#define STRONG_EDGE_COLLAPSE_CHECK 0

namespace CLN {
typedef CGAL::Simple_cartesian<double>                 Kernel;
typedef Kernel::Point_3                                Point;
typedef Kernel::Vector_3                               Vector;
typedef Kernel::Triangle_3                             Triangle;
typedef Kernel::Plane_3	                               Plane;

inline double v_norm(const Vector& A) {
	return SQRT(A*A); // operator * is overloaded as dot product
}
inline Vector v_normalized(const Vector& A) {
	const double nrmSq(A*A);
	return (nrmSq==0 ? A : A / SQRT(nrmSq));
}
inline double v_angle(const Vector& A, const Vector& B) {
	return acos(MAXF(-1.0, MINF(1.0, v_normalized(A)*v_normalized(B))));
}
inline double p_angle(const Point& A, const Point& B, const Point& C) {
	return v_angle(A-B, C-B);
}
#define edge_size(h) v_norm((h)->vertex()->point() - (h)->next()->next()->vertex()->point())

template <class Refs, class T, class P, class Normal>
class MeshVertex : public CGAL::HalfedgeDS_vertex_base<Refs, T, P>
{
public:
	typedef CGAL::HalfedgeDS_vertex_base<Refs, T, P> Base;

	enum FALGS {
		FLG_EULER = (1 << 0), // Euler operations
		FLG_BORDER = (1 << 1), // border edge
	};
	Flags flags;

	Normal normal;
	Normal laplacian;
	Normal laplacian_deriv;
	#if CURVATURE_TH>0
	float mean_curvature;
	#endif

	MeshVertex() {}
	MeshVertex(const P& pt) : CGAL::HalfedgeDS_vertex_base<Refs, T, P>(pt) {}

	void setBorder() { flags.set(FLG_BORDER); }
	void unsetBorder() { flags.unset(FLG_BORDER); }
	bool isBorder() const { return flags.isSet(FLG_BORDER); }

	void move(const Vector& offset) {
		if (isBorder()) return;
		this->point() = this->point() + offset;
	}
};

template <class Refs, class T, class Normal>
class MeshFacet : public CGAL::HalfedgeDS_face_base<Refs, T>
{
public:
	typedef CGAL::HalfedgeDS_face_base<Refs, T>  Base;
	typedef typename Refs::Vertex_handle         Vertex_handle;
	typedef typename Refs::Vertex_const_handle   Vertex_const_handle;
	typedef typename Refs::Halfedge_handle       Halfedge_handle;
	typedef typename Refs::Halfedge_const_handle Halfedge_const_handle;
	typedef typename Refs::Face_handle           Face_handle;
	typedef typename Refs::Face_const_handle     Face_const_handle;

	char removal_status; // for self intersection removal: U -  unvisited; 'P' - partially valid; V - valid
						 // for connected components: U - unvisited; V - visited

	MeshFacet() : removal_status('U') {}

	inline bool isTrinagle() const {
		return this->halfedge()->vertex() == this->halfedge()->next()->next()->next()->vertex();
	}

	inline Triangle triangle() const {
		ASSERT(isTrinagle());
		return Triangle(get_point(0), get_point(1), get_point(2));
	}

	inline Point center() const {
		ASSERT(isTrinagle());
		return baricentric(0.333f, 0.333f);
	}

	inline Point baricentric(float u1, float u2) const {
		ASSERT(isTrinagle());
		Point p[3];
		Point result;
		p[0] = get_point(0);
		p[1] = get_point(1);
		p[2] = get_point(2);
		return Point(p[0].x()*u1 + p[1].x()*u2 + p[2].x()*(1.f - u1 -u2),
					 p[0].y()*u1 + p[1].y()*u2 + p[2].y()*(1.f - u1 -u2),
					 p[0].z()*u1 + p[1].z()*u2 + p[2].z()*(1.f - u1 -u2));

	}

	inline Halfedge_const_handle get_edge(int index) const {
		ASSERT(isTrinagle());
		switch (index) {
		case 0: return this->halfedge();
		case 1: return this->halfedge()->next();
		case 2: return this->halfedge()->next()->next();
		}
		ASSERT("invalid index" == NULL);
		return Halfedge_const_handle();
	}
	inline Halfedge_handle get_edge(int index) {
		ASSERT(isTrinagle());
		switch (index) {
		case 0: return this->halfedge();
		case 1: return this->halfedge()->next();
		case 2: return this->halfedge()->next()->next();
		}
		ASSERT("invalid index" == NULL);
		return Halfedge_handle();
	}
	inline Vertex_const_handle get_vertex(int index) const {
		return get_edge(index)->vertex();
	}
	inline Vertex_handle get_vertex(int index) {
		return get_edge(index)->vertex();
	}
	inline Point get_point(int index) const {
		return get_edge(index)->vertex()->point();
	}

	inline double edgeStatistics(int mode=1) const { // 0 - min; 1-avg; 2-max
		ASSERT(isTrinagle());
		const double e1(v_norm(get_point(0)-get_point(1)));
		const double e2(v_norm(get_point(0)-get_point(2)));
		const double e3(v_norm(get_point(1)-get_point(2)));
		switch (mode) {
		case 0: return MINF3(e1, e2, e3);
		case 1: return (e1+e2+e3) / 3;
		case 2: return MAXF3(e1, e2, e3);
		}
		ASSERT("invalid mode" == NULL);
		return 0;
	}

	inline double edgeMin(Halfedge_handle& h) {
		ASSERT(isTrinagle());
		const double e1(v_norm(get_point(0)-get_point(2)));
		const double e2(v_norm(get_point(1)-get_point(0)));
		const double e3(v_norm(get_point(2)-get_point(1)));
		if (e1 < e2) {
			if (e1 < e3) {
				h = get_edge(0);
				return e1;
			} else {
				h = get_edge(2);
				return e3;
			}
		} else {
			if (e2 < e3) {
				h = get_edge(1);
				return e2;
			} else {
				h = get_edge(2);
				return e3;
			}
		}
	}

	inline Normal normal() const {
		return CGAL::cross_product(get_point(1)-get_point(0), get_point(2)-get_point(0));
	}

	inline double area() const {
		return v_norm(normal())/2;
	}
};

class MeshItems : public CGAL::Polyhedron_items_3
{
public:
	template <class Refs, class Traits>
	struct Vertex_wrapper {
		typedef typename Traits::Point_3  Point;
		typedef typename Traits::Vector_3 Normal;
		typedef MeshVertex<Refs, CGAL::Tag_true, Point, Normal> Vertex;
	};
	template <class Refs, class Traits>
	struct Face_wrapper {
		typedef typename Traits::Vector_3 Normal;
		typedef MeshFacet<Refs, CGAL::Tag_true, Normal> Face;
	};
};

typedef CGAL::Polyhedron_3<Kernel, MeshItems>          Polyhedron;

typedef Polyhedron::Vertex                             Vertex;
typedef Polyhedron::Facet                              Facet;
typedef Polyhedron::Halfedge                           Halfedge;

typedef Polyhedron::Vertex_iterator                    Vertex_iterator;
typedef Polyhedron::Vertex_const_iterator              Vertex_const_iterator;
typedef Polyhedron::Facet_iterator                     Facet_iterator;
typedef Polyhedron::Facet_const_iterator               Facet_const_iterator;

typedef Polyhedron::Point_iterator                     Point_iterator;
typedef Polyhedron::Point_const_iterator               Point_const_iterator;
typedef Polyhedron::Edge_iterator                      Edge_iterator;
typedef Polyhedron::Edge_const_iterator                Edge_const_iterator;
typedef Polyhedron::Halfedge_iterator                  Halfedge_iterator;
typedef Polyhedron::Halfedge_const_iterator            Halfedge_const_iterator;
typedef Polyhedron::Halfedge_around_facet_circulator   HF_circulator;
typedef Polyhedron::Halfedge_around_vertex_circulator  HV_circulator;

struct Stats {
	double min, avg, stdDev, max;
};
static void ComputeStatsArea(const Polyhedron& p, Stats& stats)
{
	MeanStd<double> mean;
	stats.min = FLT_MAX;
	stats.max = 0;
	for (Facet_const_iterator fi = p.facets_begin(); fi!=p.facets_end(); fi++) {
		const double tmpArea(fi->area());
		if (stats.min > tmpArea)
			stats.min = tmpArea;
		if (stats.max < tmpArea)
			stats.max = tmpArea;
		mean.Update(tmpArea);
	}
	stats.avg = mean.GetMean();
	stats.stdDev = mean.GetStdDev();
}
static void ComputeStatsEdge(const Polyhedron& p, Stats& stats)
{
	MeanStd<double> mean;
	stats.min = FLT_MAX;
	stats.max = 0;
	for (Edge_const_iterator ei = p.edges_begin(); ei!=p.edges_end(); ei++) {
		const double tmpEdge(v_norm(ei->vertex()->point() - ei->prev()->vertex()->point()));
		if (stats.min > tmpEdge)
			stats.min = tmpEdge;
		if (stats.max < tmpEdge)
			stats.max = tmpEdge;
		mean.Update(tmpEdge);
	}
	stats.avg = mean.GetMean();
	stats.stdDev = mean.GetStdDev();
}
static void ComputeStatsLaplacian(const Polyhedron& p, Stats& stats)
{
	MeanStd<double> mean;
	stats.min = FLT_MAX;
	stats.max = 0;
	for (Vertex_const_iterator vi = p.vertices_begin(); vi !=p.vertices_end(); vi++) {
		double tmpNorm(v_norm(vi->laplacian));
		if (vi->laplacian*vi->normal<0.f)
			tmpNorm = -tmpNorm;
		if (stats.min > tmpNorm)
			stats.min = tmpNorm;
		if (stats.max < tmpNorm)
			stats.max = tmpNorm;
		mean.Update(tmpNorm);
	}
	stats.avg = mean.GetMean();
	stats.stdDev = mean.GetStdDev();
}

inline double OppositeAngle(Vertex::Halfedge_handle h) {
	ASSERT(h->facet()->is_triangle());
	return v_angle(h->vertex()->point() - h->next()->vertex()->point(), h->prev()->vertex()->point() - h->next()->vertex()->point());
}

inline bool CanCollapseCenterVertex(Vertex::Vertex_handle v) {
	if (!v->is_trivalent()) return false;
	if (v->isBorder()) return false;
	return (v->halfedge()->prev()->opposite()->facet() != v->halfedge()->opposite()->next()->opposite()->facet());
}

#define REPLACE_POINT(V,P1,P2,PMIDDLE) (((V==P1)||(V==P2)) ? (PMIDDLE) : (V))
static bool CanCollapseEdge(Vertex::Halfedge_handle v0v1)
{
	if (v0v1->is_border_edge())
		return false;
	Vertex::Halfedge_handle v1v0 = v0v1->opposite();
	if (v0v1->next()->opposite()->facet() == v1v0->prev()->opposite()->facet())
		return false;
	Vertex::Vertex_handle v0 = v0v1->vertex();
	if (v0->isBorder())
		return false;
	Vertex::Vertex_handle v1 = v1v0->vertex();
	if (v1->isBorder())
		return false;

	Vertex::Vertex_handle vl, vr;
	Vertex::Halfedge_handle h1, h2;
	if (!v0v1->is_border()) {
		vl = v0v1->next()->vertex();
		h1 = v0v1->next();
		h2 = v0v1->next()->next();
		if (h1->is_border() || h2->is_border())
			return false;
	}
	if (!v1v0->is_border()) {
		vr = v1v0->next()->vertex();
		h1 = v1v0->next();
		h2 = v1v0->next()->next();
		if (h1->is_border() || h2->is_border())
			return false;
	}
	// if vl and vr are equal or both invalid -> fail
	if (vl == vr)
		return false;

	HV_circulator c, d;

	// test intersection of the one-rings of v0 and v1
	c = v0->vertex_begin(); d = c;
	CGAL_For_all(c, d)
		c->opposite()->vertex()->flags.unset(Vertex::FLG_EULER);

	c = v1->vertex_begin(); d = c;
	CGAL_For_all(c, d)
		c->opposite()->vertex()->flags.set(Vertex::FLG_EULER);

	c = v0->vertex_begin(); d = c;
	CGAL_For_all(c, d) {
		Vertex::Vertex_handle vTmp =c->opposite()->vertex();
		if (vTmp->flags.isSet(Vertex::FLG_EULER) && (vTmp!=vl) && (vTmp!=vr))
			return false;
	}

	// test weather when performing the edge collapse we change the signed area of any triangle
	#if STRONG_EDGE_COLLAPSE_CHECK==1
	Point p0 = v0->point();
	Point p1 = v1->point();
	Point p_middle = p0 + (p1 - p0) / 2;
	Point t1, t2, t3;
	Vector a1, a2;
	for (int x=0; x<2; x++) {
		if (x==0) {
			c = v0->vertex_begin(); d = c;
		} else {
			c = v1->vertex_begin(); d = c;
		}
		CGAL_For_all(c, d) {
			t1 = c->vertex()->point();
			t2 = c->next()->vertex()->point();
			t3 = c->next()->next()->vertex()->point();
			a1 = CGAL::cross_product(t2-t1, t3-t2);
			t1 = REPLACE_POINT(t1, p0, p1, p_middle);
			t2 = REPLACE_POINT(t2, p0, p1, p_middle);
			t3 = REPLACE_POINT(t3, p0, p1, p_middle);
			a2 = CGAL::cross_product(t2-t1, t3-t2);

			if ((v_norm(a2) != 0) && (v_angle(a1, a2) > PI/2))
				return false;
		}
	}
	#endif
	return true;
}
static void CollapseEdge(Polyhedron& p, Vertex::Halfedge_handle h)
{
	Vertex::Halfedge_handle h1 = h->next();
	Vertex::Halfedge_handle h2 = h->opposite()->prev();
	Point p1 = h->vertex()->point();
	Point p2 = h->opposite()->vertex()->point();
	Point p3 = p1 + (p2-p1) /2;
	size_t degree_p1 = h->vertex()->vertex_degree();
	size_t degree_p2 = h->opposite()->vertex()->vertex_degree();

	#if 0
	if (h->vertex()->isBorder())
		p3=p1;
	else if (h->opposite()->vertex()->isBorder())
		p3=p2;
	else
	#endif
	if (degree_p1 > degree_p2)
		p3=p1;
	else
		p3=p2;

	h->vertex()->point() = p3;

	p.join_facet(h1->opposite());
	p.join_facet(h2->opposite());
	p.join_vertex(h);
}

static bool CanFlipEdge(Vertex::Halfedge_handle h)
{
	if (h->is_border_edge()) return false;
	const Vertex::Halfedge_handle null_h;
	if ((h->next() == null_h) || (h->prev() == null_h) || (h->opposite() == null_h) || (h->opposite()->next() == null_h)) return false;
	Vertex::Vertex_handle v0 = h->next()->vertex();
	Vertex::Vertex_handle v1 = h->opposite()->next()->vertex();

	v0->flags.unset(Vertex::FLG_EULER);

	HV_circulator c = v1->vertex_begin();
	HV_circulator d = c;
	CGAL_For_all(c, d)
		c->opposite()->vertex()->flags.set(Vertex::FLG_EULER);

	if (v0->flags.isSet(Vertex::FLG_EULER)) return false;

	// check if it increases the quality overall
	double a1 = OppositeAngle(h);
	double a2 = OppositeAngle(h->next());
	double a3 = OppositeAngle(h->next()->next());
	double b1 = OppositeAngle(h->opposite());
	double b2 = OppositeAngle(h->opposite()->next());
	double b3 = OppositeAngle(h->opposite()->next()->next());

	if ((a1*a1 + b1*b1) / (a2*a2 + a3*a3 + b2*b2 + b3*b3) < 1.01) return false;

	Vector v_perp_1 = CGAL::cross_product(h->vertex()->point() - h->next()->vertex()->point(), h->vertex()->point() - h->prev()->vertex()->point());
	//Vector v_perp_2 = CGAL::cross_product(h->opposite()->vertex()->point()-h->opposite()->next()->vertex()->point(),h->opposite()->vertex()->point()-h->opposite()->prev()->vertex()->point());
	Vector v_perp_2 = CGAL::cross_product(h->opposite()->next()->vertex()->point() - h->opposite()->vertex()->point(), h->vertex()->point()-h->prev()->vertex()->point());
	if (v_angle(v_perp_1, v_perp_2) > D2R(20)) return false;

	return (h->next()->opposite()->facet() != h->opposite()->prev()->opposite()->facet()) &&
		(h->prev()->opposite()->facet() != h->opposite()->next()->opposite()->facet()) &&
		(CGAL::circulator_size(h->opposite()->vertex_begin()) >= 3) &&
		(CGAL::circulator_size(h->vertex_begin()) >= 3);
}
inline void FlipEdge(Polyhedron& p, Vertex::Halfedge_handle h) {
	p.flip_edge(h);
}

#if SPLIT_BORDER_EDGES>0
inline bool CanSplitEdge(Vertex::Halfedge_handle& h) {
	if (h->vertex()->point() == h->opposite()->vertex()->point())
		return false;
	if (h->face() == Vertex::Face_handle())
		h = h->opposite();
	return true;
}
#else
inline bool CanSplitEdge(Vertex::Halfedge_handle h) {
	if (h->is_border_edge()) return false;
	if (h->facet() == h->opposite()->facet()) return false;
	ASSERT(h->facet()->is_triangle() && h->opposite()->facet()->is_triangle());
	return (h->vertex()->point() != h->opposite()->vertex()->point());
}
#endif
// 1 - middle ; 2-projection of the 3rd vertex
static void SplitEdge(Polyhedron& p, Vertex::Halfedge_handle h, int mode=1)
{
	ASSERT(mode == 1 || mode == 2);
	Point p1 = h->vertex()->point();
	Point p2 = h->opposite()->vertex()->point();
	Point p3 = h->next()->vertex()->point();

	Point p_midddle;
	if (mode==1) { // middle
		const double ratio(0.5);
		p_midddle = p1 + (p2-p1) * ratio;
	} else { // projection of the 3rd vertex
		const double ratio(v_norm(p3-p2) * cos(OppositeAngle(h->next())) / v_norm(p1-p2));
		p_midddle = p2 + (p1-p2) * ratio;
	}

	Vertex::Halfedge_handle hnew = p.split_edge(h);
	hnew->vertex()->point() = p_midddle;

	p.split_facet(hnew, h->next());
	#if SPLIT_BORDER_EDGES>0
	if (h->opposite()->face() != Vertex::Face_handle())
	#endif
	p.split_facet(h->opposite(), hnew->opposite()->next());
}


// mode : 0 - min, 1 - avg, 2- max
static float ComputeVertexStatistics(Vertex& v, int mode)
{
	ASSERT((mode>=0) && (mode <=2));
	if (v.vertex_degree()==0) return 0;
	HV_circulator h = v.vertex_begin();
	float edge_stats((float)edge_size(h));
	int no_h(1);
	do {
		switch (mode) {
		case 0: edge_stats = MINF((float)edge_size(h), edge_stats); break;
		case 1: edge_stats += (float)edge_size(h), ++no_h; break;
		case 2: edge_stats = MAXF((float)edge_size(h), edge_stats); break;
		}
	} while (++h != v.vertex_begin());
	if (mode==1) edge_stats = (edge_stats/no_h);
	return edge_stats;
}

static int ImproveVertexValence(Polyhedron& p, int valence_mode=2)
{
	int total_no_ops(0);
	switch (valence_mode) {
	case 1: {
		//erase all the center triangles!
		for (Vertex_iterator vi=p.vertices_begin(); vi!=p.vertices_end(); ++vi) {
			Vertex::Vertex_handle old_vi = vi;
			if (CanCollapseCenterVertex(old_vi))
				p.erase_center_vertex(old_vi->halfedge());
		}
		for (Vertex_iterator vi=p.vertices_begin(); vi!=p.vertices_end(); ) {
			Vertex::Vertex_handle old_vi = vi;
			vi++;
			size_t degree = old_vi->vertex_degree();
			//std::cout << "degree" << degree << std::endl;
			//if (CanCollapseCenterVertex(old_vi))
			//	p.erase_center_vertex(old_vi->halfedge());
			//else
			if (degree==4) {
				double edge_stats = ComputeVertexStatistics(*old_vi, 0); //min
				//std::cout << edge_stats << std::endl;
				HV_circulator c, d;
				c = old_vi->vertex_begin();
				float current_edge_stats;
				current_edge_stats = (float)edge_size(c);

				while (edge_stats!=current_edge_stats) {
					//std::cout << "current edge stats:" << current_edge_stats << std::endl;
					c++;
					current_edge_stats = (float)edge_size(c);
				}
				d = c;
				//bool collapsed(false);
				CGAL_For_all(c, d) {
					if (CanCollapseEdge(c->opposite())) {
						//if ((c->opposite()->vertex()==vi) && (vi!=p.vertices_end())) vi++;
						CollapseEdge(p, c->opposite());
						//collapsed = true;
						total_no_ops++;
						break;
					}
				}
				//if (!collapsed) std::cout << "could not collapse edge!" << std::endl;
			}
		}
	} break;

	case 2: {
		int iters(0), no_ops;
		do {
			iters++;
			no_ops = 0;
			for (Edge_iterator ei=p.edges_begin(); ei!=p.edges_end(); ++ei) {
				if (ei->is_border_edge())
					continue;
				int d1_1 = (int)ei->vertex()->vertex_degree();
				int d1_2 = (int)ei->opposite()->vertex()->vertex_degree();
				int d2_1 = (int)ei->next()->vertex()->vertex_degree();
				int d2_2 = (int)ei->opposite()->next()->vertex()->vertex_degree();
				Vertex::Halfedge_handle h = ei;
				if (((d1_1+d1_2) - (d2_1+d2_2) > 2) && CanFlipEdge(h)) {
					FlipEdge(p, h);
					no_ops++;
				}
			}
			//FixDegeneracy(p, 0.2,150);
			total_no_ops += no_ops;
		} while ((no_ops>0) && (iters<1));
	} break;

	case 3: {
		int iters(0), no_ops;
		do {
			iters++;
			no_ops = 0;
			for (Edge_iterator ei=p.edges_begin(); ei!=p.edges_end(); ++ei) {
				if (ei->is_border_edge())
					continue;
				Point p1=ei->vertex()->point();
				Point p2=ei->next()->vertex()->point();
				Point p3=ei->prev()->vertex()->point();
				Point p4=ei->opposite()->next()->vertex()->point();

				float cost1((float)MINF(MINF(MINF(MINF(MINF(p_angle(p3, p1, p2), p_angle(p1, p3, p2)), p_angle(p4, p1, p3)), p_angle(p4, p3, p1)), p_angle(p1, p4, p2)), p_angle(p1, p2, p3)));
				float cost2((float)MINF(MINF(MINF(MINF(MINF(p_angle(p1, p2, p4), p_angle(p1, p4, p2)), p_angle(p3, p4, p2)), p_angle(p4, p2, p3)), p_angle(p4, p1, p2)), p_angle(p4, p3, p2)));

				Vertex::Halfedge_handle h = ei;
				if ((cost2 > cost1) && CanFlipEdge(h)) {
					FlipEdge(p, h);
					no_ops++;
				}
			}
			//FixDegeneracy(p, 0.2,150);
			total_no_ops += no_ops;
		} while ((no_ops>0) && (iters<1));
	} break;
	}
	return total_no_ops;
}


static void UpdateMeshData(Polyhedron& p);

// Description: 
//  It iterates through all the mesh vertices and it tries to fix degenerate triangles.
//  There are conditions that check for large and small angles.
// Parameters:
//  - degenerateAngleDeg 
//     - for large angles: if an angle is bigger than degenerateAngleDeg.
//     - a good values to use is typically 170
//  - collapseRatio 
//     - for small angles: given the corresponding edges (a,b,c) in all permutations, if (a/b < collapseRatio) & (a/c < collapseRatio)
//     - a good value to use is 0.1 
static int FixDegeneracy(Polyhedron& p, double collapseRatio, double degenerateAngleDeg)
{
	DEBUG_LEVEL(3, "Fix degeneracy: %g collapse-ratio, %g degenerate-angle", collapseRatio, degenerateAngleDeg);
	double edge[3];
	int no_ops(0), counter(0);
	Vertex::Halfedge_handle edge_h[3];
	for (Facet_iterator fi = p.facets_begin(); fi!=p.facets_end(); ) {
		counter++;
		if (fi->triangle().is_degenerate()) {
			const double avg_edge(fi->edgeStatistics(1));
			DEBUG_LEVEL(3, "Degenerate angle: %g (avg edge %g)", v_angle(fi->get_point(0)-fi->get_point(1), fi->get_point(0)-fi->get_point(2)), avg_edge);
			const double delta(avg_edge*0.01);
			fi->halfedge()->vertex()->point() = fi->halfedge()->vertex()->point() + Vector(0, 0, delta);
			fi->halfedge()->next()->vertex()->point() = fi->halfedge()->next()->vertex()->point() + Vector(0, delta, 0);
			fi->halfedge()->next()->next()->vertex()->point() = fi->halfedge()->next()->next()->vertex()->point() + Vector(delta, 0, 0);
		}

		// collect facet statistics
		HF_circulator hf = fi->facet_begin();
		if (hf == NULL) continue;
		int i(0);
		do {
			ASSERT(ISFINITE(v_norm(hf->vertex()->point() - CGAL::ORIGIN)));
			ASSERT(i < 3); // a triangular mesh
			edge_h[i] = hf;
			edge[i++] = v_norm(hf->vertex()->point() - hf->prev()->vertex()->point());
		} while (++hf !=fi->facet_begin());

		fi++;
		//we should have the 3 sizes of the edges by now
		for (i=0; i<3; i++) {
			if ((edge[i]/edge[(i+1)%3] < collapseRatio) && (edge[i]/edge[(i+2)%3] < collapseRatio)) {
				if (CanCollapseEdge(edge_h[i])) {
					while ((fi!=p.facets_end()) && (fi==edge_h[i]->opposite()->facet())) fi++;
					CollapseEdge(p, edge_h[i]);
					no_ops++;
					break;
				}
				#if 0
				if (CanCollapseEdge(edge_h[i]->opposite())) {
					while ((fi!=p.facets_end()) && (fi==edge_h[i]->facet())) fi++;
					CollapseEdge(p, edge_h[i]->opposite());
					no_ops++;
					break;
				}
				#endif
			} else {
				const double tmpAngle(R2D(OppositeAngle(edge_h[i])));
				if (tmpAngle > degenerateAngleDeg && CanFlipEdge(edge_h[i])) {
					FlipEdge(p, edge_h[i]);
					no_ops++;
					break;
				}
				#if 0
				if (tmpAngle > degenerateAngleDeg && CanSplitEdge(edge_h[i])) {
					SplitEdge(p, edge_h[i], 2);
					if (CanCollapseEdge(edge_h[i]->prev())) {
						while ((fi!=p.facets_end()) && (fi==edge_h[i]->prev()->opposite()->facet())) fi++;
						CollapseEdge(p, edge_h[i]->prev());
						no_ops++;
					}
					no_ops++;
					break;
				}
				#endif
			}
		}
	}
	if (no_ops)
		UpdateMeshData(p);
	return no_ops;
}
static void FixAllDegeneracy(Polyhedron& p, double collapseRatio, double degenerateAngleDeg)
{
	int runs(0);
	do {
		if (FixDegeneracy(p, collapseRatio, degenerateAngleDeg) == 0)
			break;
		ImproveVertexValence(p);
	} while (runs++ < 3);
}


class MeshConnectedComponent {
public:
	Vertex::Facet_handle start_facet;
	int size;
	float area;
	float edge_min, edge_avg, edge_max;
	bool is_open;
	MeshConnectedComponent() {
		start_facet = NULL;
		size=0;
		edge_min=FLT_MAX;
		edge_max=0;
		edge_avg=0;
		area=0.f;
		is_open=false;
	}
	void setParams(Vertex::Facet_handle start_facet_, int size_) {
		start_facet = start_facet_;
		size = size_;
	}
	void updateStats(float edge_size) {
		edge_max=MAXF(edge_max, edge_size);
		edge_min=MINF(edge_min, edge_size);
		edge_avg+=edge_size;
	}
};
static void ComputeConnectedComponents(Polyhedron& p, std::vector<MeshConnectedComponent>& connected_components)
{
	connected_components.clear();
	std::queue<Vertex::Facet_handle> facet_queue;
	MeshConnectedComponent lastComponent;

	// reset the facet status
	for (Facet_iterator i = p.facets_begin(); i != p.facets_end(); ++i)
		i->removal_status = 'U';

	std::cout << "Connected components of: ";
	// traverse the mesh via facets
	for (Facet_iterator i = p.facets_begin(); i != p.facets_end(); ++i) {
		if (i->removal_status=='U') { // start a new component
			lastComponent.setParams(i, 0);
			i->removal_status='V'; // it is now visited
			facet_queue.push(i);
			while (!facet_queue.empty()) { // fill the current component
				Vertex::Facet_handle f = facet_queue.front(); facet_queue.pop();
				lastComponent.size++;
				HF_circulator h = f->facet_begin();
				do {
					if (h->is_border_edge()) continue;
					lastComponent.is_open=true;
					float edge_size = (float)v_norm(h->vertex()->point() - h->prev()->vertex()->point());
					lastComponent.updateStats(edge_size);
					lastComponent.area += (float)f->area();
					Vertex::Facet_handle opposite_f = h->opposite()->facet();
					if ((opposite_f!=Vertex::Facet_handle()) && (opposite_f->removal_status=='U')) {
						opposite_f->removal_status='V'; // it is now visited
						facet_queue.push(opposite_f);
					}

				} while (++h != f->facet_begin());
			} // done traversing the current component
			lastComponent.edge_avg/=lastComponent.size*3;
			connected_components.push_back(lastComponent);
			std::cout << lastComponent.size << " faces ";
		} // found a new component
	} // done traversing the mesh
	std::cout << "(" << connected_components.size() << " components)" << std::endl;
}
static void RemoveConnectedComponents(Polyhedron& p, int size_threshold, float edge_threshold)
{
	std::vector<MeshConnectedComponent> connected_components;
	ComputeConnectedComponents(p, connected_components);
	for (std::vector<MeshConnectedComponent>::iterator vi=connected_components.begin(); vi!=connected_components.end(); ) {
		if ((vi->size<=size_threshold) || (vi->edge_max<=edge_threshold) || ((vi->area<edge_threshold*edge_threshold*PI*2) && (vi->is_open==false))) {
			p.erase_connected_component(vi->start_facet->facet_begin());
			vi = connected_components.erase(vi);
		} else
			vi++;
	}
}

// mode : 0 - tangential; 1 - across the normal
inline Vector ComputeVectorComponent(Vector n, Vector v, int mode)
{
	ASSERT((mode>=0) && (mode<2));
	Vector across_normal(n*(n*v));
	if (mode==1)
		return across_normal;
	else
		return v - across_normal;
}

static void Smooth(Polyhedron& p, double delta, int mode=0)
{
	// 0 - both components;
	// 1 - tangential;
	// 2 - normal;
	// 3 - second order;
	// 4 - combined;
	// 5 - tangential only if bigger than normal;
	// 6 - both components - laplacian_avg;
	ASSERT((mode>=0) && (mode<=6));
	Stats laplacian;
	if (mode==6)
		ComputeStatsLaplacian(p, laplacian);
	for (Vertex_iterator vi=p.vertices_begin(); vi!=p.vertices_end(); vi++) {
		Vector displacement;
		switch (mode) {
		case 0: displacement = vi->laplacian*delta/*vert->getMeanCurvatureFlow().Norm()*/; break;
		case 1: displacement = ComputeVectorComponent(vi->normal, vi->laplacian, 0) *delta; break;
		case 2: displacement = ComputeVectorComponent(vi->normal, vi->laplacian, 1) *delta; break;
		case 3: displacement = vi->laplacian_deriv*delta; break;
		case 4: displacement = vi->laplacian*delta - vi->laplacian_deriv*delta; break;
		case 5: {
			Vector d_tan(ComputeVectorComponent(vi->normal, vi->laplacian, 0)*delta);
			Vector d_norm(ComputeVectorComponent(vi->normal, vi->laplacian, 1)*delta);
			displacement = (v_norm(d_tan) > 2*v_norm(d_norm) ? d_tan : Vector(0, 0, 0));
		} break;
		case 6: displacement = vi->laplacian *delta - vi->normal*laplacian.avg*delta; break;
		}
		vi->move(displacement);
	}
	UpdateMeshData(p);
}


// Description: 
// - The goal of this method is to ensure that all the edges of the mesh are within the interval [epsilonMin,epsilonMax].
//   In order to do so, edge collapses and edge split operations are performed.
// - The method also attempts to fix degeneracies by invoking FixDegeneracy(collapseRatio,degenerate_angle_deg) and performs some local smoothing, based on the operating mode.
// Parameters:
// - [epsilonMin, epsilonMax] - the desired edge interval (negative if to be used as multiplier of the initial mean edge length)
// - collapseRatio, degenerate_angle_deg - parameters used to invoke FixDegeneracy (see function for more details)
// - mode : 0 - fixDegeneracy=No  smoothing=Yes;
//          1 - fixDegeneracy=Yes smoothing=Yes; (default)
//         10 - fixDegeneracy=Yes smoothing=No;
// - max_iter (default=30) - maximum number of iterations to be performed; since there is no guarantee that one operations (such as a collapse, for example)
//   will not in turn generate new degeneracies, operations are being performed on the mesh in an iterative fashion. 
static void EnsureEdgeSize(Polyhedron& p, double epsilonMin, double epsilonMax, double collapseRatio, double degenerate_angle_deg, int mode, int max_iters, int comp_size_threshold)
{
	if (mode>0)
		FixDegeneracy(p, collapseRatio, degenerate_angle_deg);

	#if ENSURE_MIN_AREA>0
	Stats area;
	ComputeStatsArea(p, area);
	const float thArea((float)area.avg/ENSURE_MIN_AREA);
	#endif

	Stats edge;
	ComputeStatsEdge(p, edge);
	if (epsilonMin < 0)
		epsilonMin = edge.avg * (-epsilonMin);
	if (epsilonMax < 0)
		epsilonMax = MAXF(edge.avg * (-epsilonMax), epsilonMin * 2);
	DEBUG_LEVEL(3, "Ensuring edge size in [%g, %g] with edges currently in [%g, %g]", epsilonMin, epsilonMax, edge.min, edge.max);

	typedef TIndexScore<Vertex::Halfedge_handle, float> EdgeScore;
	typedef CLISTDEF0(EdgeScore) EdgeScoreArr;
	EdgeScoreArr bigEdges(0, 1024);
	int iters(0), total_no_ops(0), no_ops(1);
	while ((edge.min<epsilonMin || edge.max>epsilonMax) && (no_ops>0) && (++iters<max_iters)) {
		no_ops = 0;
		if (iters > 1)
			ComputeStatsEdge(p, edge);

		// process big edges
		ASSERT(bigEdges.IsEmpty());
		for (Halfedge_iterator h = p.edges_begin(); h != p.edges_end(); ++h, ++h) {
			ASSERT(++Halfedge_iterator(h) == h->opposite());
			const double edgeSize(edge_size(h));
			if (edgeSize > epsilonMax)
				bigEdges.AddConstruct(h, (float)edgeSize);
		}
		DEBUG_LEVEL(3, "Big edges: %u", bigEdges.GetSize());
		bigEdges.Sort(); // process big edges first
		for (EdgeScoreArr::IDX i=0; i<bigEdges.GetSize(); ++i) {
			Vertex::Halfedge_handle h(bigEdges[i].idx);
			if (!CanSplitEdge(h))
				continue;
			#if CURVATURE_TH>0
			const float avg_mean_curv(MAXF(ABS(h->vertex()->mean_curvature), ABS(h->prev()->vertex()->mean_curvature)));
			if (avg_mean_curv < CURVATURE_TH)
				continue;
			#endif
			#if ENSURE_MIN_AREA>0
			if (h->facet()->area() < thArea)
				continue;
			#endif
			SplitEdge(p, h);
			no_ops++;
		}
		bigEdges.Empty();

		// process small edges
		Vertex::Halfedge_handle h;
		Facet_iterator f(p.facets_begin());
		while (f != p.facets_end()) {
			const double minEdge(f->edgeMin(h));
			f++;
			if (minEdge < epsilonMin && CanCollapseEdge(h)) {
				while ((f!=p.facets_end()) && ((f==h->opposite()->facet()) || (f==h->facet()))) f++;
				CollapseEdge(p, h);
				no_ops++;
			}
		}

		if (mode <= 10)
			ImproveVertexValence(p);
		if (mode == 10)
			FixDegeneracy(p, collapseRatio, degenerate_angle_deg);

		UpdateMeshData(p);
		if (mode < 10)
			Smooth(p, 0.1, 1);

		total_no_ops += no_ops;
	}
	if (mode > 0) {
		FixAllDegeneracy(p, collapseRatio, degenerate_angle_deg);
		if (mode <10)
			Smooth(p, 0.1, 1);
	}

	if (comp_size_threshold > 0)
		RemoveConnectedComponents(p, comp_size_threshold, (float)edge.min*2);

	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2) {
		ComputeStatsEdge(p, edge);
		VERBOSE("Edge size in [%g, %g] (requested in [%g, %g]): %d ops, %d iters", edge.min, edge.max, epsilonMin, epsilonMax, total_no_ops, iters);
	}
	#endif	
}

static void ComputeVertexNormals(Polyhedron& p)
{
	for (Vertex_iterator vi = p.vertices_begin(); vi!=p.vertices_end(); vi++)
		vi->normal = CGAL::NULL_VECTOR;
	for (Facet_iterator fi = p.facets_begin(); fi!=p.facets_end(); fi++) {
		const Vector t(fi->normal());
		Vector& n0(fi->get_vertex(0)->normal); n0 = n0 + t;
		Vector& n1(fi->get_vertex(1)->normal); n1 = n1 + t;
		Vector& n2(fi->get_vertex(2)->normal); n2 = n2 + t;
	}
	for (Vertex_iterator vi = p.vertices_begin(); vi!=p.vertices_end(); vi++) {
		Vector& normal(vi->normal);
		const double nrm(v_norm(normal));
		#if ROBUST_NORMALS==0
		if (nrm != 0)
			normal = normal / nrm;
		#else
		// only temporarily set here, it will be set in normal when computing the robust measure
		vi->laplacian_deriv = (nrm != 0 ? (normal / nrm) : CGAL::NULL_VECTOR);
		#endif
	}
}
#if ROBUST_NORMALS>0
static std::vector< std::pair<Vertex*, int> > GetRingNeighbourhood(Vertex& v, int ring_size, bool include_original) {
	std::map<Vertex*, int> neigh_map;
	std::map<Vertex*, int>::iterator iter;

	std::queue<Vertex*> elems;
	std::vector< std::pair<Vertex*, int> > result;

	// add base level	
	elems.push(&v);
	neigh_map[&v]=0;

	if (ring_size < 0) return result;

	while (elems.size() > 0) {
		Vertex* el = elems.front(); elems.pop();
		if ((el != &v) || include_original)
			result.push_back(std::pair<Vertex*, int>(el, neigh_map[el]));
		if (neigh_map[el]==ring_size) continue;
		//circulate one ring neighborhood
		HV_circulator c = el->vertex_begin();
		HV_circulator d = c;
		CGAL_For_all(c, d) {
			Vertex* next_el = &(*(c->opposite()->vertex()));
			iter=neigh_map.find(next_el);
			if (iter == neigh_map.end()) { // if the vertex has not been already taken
				elems.push(next_el);
				neigh_map[next_el]=neigh_map[el]+1;
			}
		}
	}

	return result;
}
static void ComputeVertexRobustNormal(Vertex& v, int maxRings)
{
	std::vector< std::pair<Vertex*, int> > neighs(GetRingNeighbourhood(v, maxRings, true));
	Vector& normal(v.normal);
	normal = CGAL::NULL_VECTOR;
	for (std::vector< std::pair<Vertex*, int> >::const_iterator n_it=neighs.cbegin(); n_it!=neighs.cend(); ++n_it) {
		Vertex* neigh_v = n_it->first;
		//const float w = n_it->second / maxRings;
		normal = normal + neigh_v->laplacian_deriv;//*w;
	}
	const double nrm(v_norm(normal));
	if (nrm != 0)
		normal = normal / nrm;
}
#endif

static void ComputeVertexLaplacian(Vertex& v)
{
	// formula taken from "Mesh Smoothing via Mean and Median Filtering Applied to Face Normals"
	HV_circulator vi = v.vertex_begin();
	ASSERT(vi != NULL);
	Vector result_laplacian(0, 0, 0);
	#ifndef LAPLACIAN_ROBUST
	size_t order = 0;
	do {
		++order;
		if (vi->is_border_edge()) {
			v.laplacian = Vector(0, 0, 0);
			return;
		}
		result_laplacian = result_laplacian + (vi->prev()->vertex()->point() - CGAL::ORIGIN);
	} while (++vi != v.vertex_begin());
	result_laplacian = result_laplacian/(double)order - (v.point() - CGAL::ORIGIN);
	#else
	float w_total(0);
	Vector e, e_next, e_prev;
	do {
		e_next = vi->next()->vertex()->point() - vi->vertex()->point();
		e = vi->prev()->vertex()->point() - vi->vertex()->point();
		e_prev = vi->opposite()->next()->vertex()->point() - vi->vertex()->point();

		float theta_1((float)v_angle(e, e_next));
		float theta_2((float)v_angle(e, e_prev));
		float w((tan(theta_1/2)+tan(theta_2/2))/v_norm(e));

		w_total += w;
		result_laplacian = result_laplacian + w*e;
	} while (++vi != v.vertex_begin());
	result_laplacian = result_laplacian / (double)w_total;
	#endif
	v.laplacian = result_laplacian;
}
static void ComputeVertexLaplacianDeriv(Vertex& v)
{
	HV_circulator vi = v.vertex_begin();
	ASSERT(vi != NULL);
	v.laplacian_deriv = Vector(0, 0, 0);
	size_t order(0);
	do {
		++order;
		v.laplacian_deriv = v.laplacian_deriv + (vi->prev()->vertex()->laplacian-v.laplacian);
	} while (++vi != v.vertex_begin());
	v.laplacian_deriv = v.laplacian_deriv / (double)order;
}

#if CURVATURE_TH>0
static void ComputeVertexCurvature(Vertex& v)
{
	float edge_avg(ComputeVertexStatistics(v, 2));
	float mean_curv(ABS((float)v_norm(v.laplacian) / edge_avg));
	if (v.laplacian*v.normal<0)
		mean_curv = -mean_curv;
	if (!ISFINITE(mean_curv))
		mean_curv = 0;
	v.mean_curvature = mean_curv;
}
#endif

static void UpdateMeshData(Polyhedron& p)
{
	p.normalize_border();

	// compute vertex normal
	ComputeVertexNormals(p);
	#if ROBUST_NORMALS>0
	// compute robust vertex normal		
	for (Vertex_iterator vi=p.vertices_begin(); vi!=p.vertices_end(); vi++)
		ComputeVertexRobustNormal(*vi, ROBUST_NORMALS);
	#endif

	// compute Laplacians
	for (Vertex_iterator vi=p.vertices_begin(); vi!=p.vertices_end(); vi++) {
		vi->unsetBorder();
		ComputeVertexLaplacian(*vi);
	}

	// compute curvature and Laplacian derivative
	for (Vertex_iterator vi=p.vertices_begin(); vi!=p.vertices_end(); vi++) {
		#if CURVATURE_TH>0
		ComputeVertexCurvature(*vi);
		#endif
		ComputeVertexLaplacianDeriv(*vi);
	}

	// set border edges	
	for (Halfedge_iterator hi=p.border_halfedges_begin(); hi!=p.halfedges_end(); hi++)
		hi->vertex()->setBorder();
}

// a modifier creating a triangle with the incremental builder
template <class HDS, class K>
class TMeshBuilder : public CGAL::Modifier_base<HDS>
{
public:
	const Mesh::VertexArr& vertices;
	const Mesh::FaceArr& faces;
	bool bProblems;

	TMeshBuilder(const Mesh::VertexArr& _vertices, const Mesh::FaceArr& _faces) : vertices(_vertices), faces(_faces), bProblems(false) {}

	void operator() (HDS& hds) {
		typedef typename HDS::Vertex::Point Point;
		CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, false);
		B.begin_surface(vertices.GetSize(), faces.GetSize());
		// add the vertices		
		FOREACH(i, vertices) {
			const Mesh::Vertex& v = vertices[i];
			B.add_vertex(Point(v.x, v.y, v.z));
		}
		// add the facets
		#if TD_VERBOSE != TD_VERBOSE_OFF
		String msgFaces;
		#endif
		FOREACH(i, faces) {
			const Mesh::Face& f = faces[i];
			if (!B.test_facet(f.ptr(), f.ptr()+3)) {
				bProblems = true;
				#if TD_VERBOSE != TD_VERBOSE_OFF
				if (VERBOSITY_LEVEL > 1)
					msgFaces += String::FormatString(" %u", i);
				#endif
				continue;
			}
			B.add_facet(f.ptr(), f.ptr()+3);
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (bProblems)
			DEBUG_EXTRA("warning: ignoring the following facet(s) violating the manifold constraint:%s", msgFaces.c_str());
		#endif
		if (B.check_unconnected_vertices()) {
			DEBUG_EXTRA("warning: remove unconnected vertices");
			B.remove_unconnected_vertices();
		}
		B.end_surface();
	}
};
typedef TMeshBuilder<Polyhedron::HalfedgeDS, Kernel> MeshBuilder;
static bool ImportMesh(Polyhedron& p, const Mesh::VertexArr& vertices, const Mesh::FaceArr& faces) {
	MeshBuilder builder(vertices, faces);
	p.delegate(builder);
	UpdateMeshData(p);
	DEBUG_ULTIMATE("Mesh imported: %u vertices, %u facets (%u border edges)", p.size_of_vertices(), p.size_of_facets(), p.size_of_border_edges());
	return true;
}
static bool ExportMesh(const Polyhedron& p, Mesh::VertexArr& vertices, Mesh::FaceArr& faces) {
	if (p.size_of_vertices() >= std::numeric_limits<Mesh::VIndex>::max())
		return false;
	if (p.size_of_facets() >= std::numeric_limits<Mesh::FIndex>::max())
		return false;
	unsigned nCount;
	// extract vertices
	nCount = 0;
	vertices.Resize((Mesh::VIndex)p.size_of_vertices());
	for (Polyhedron::Vertex_const_iterator it=p.vertices_begin(), ite=p.vertices_end(); it!=ite; ++it) {
		Mesh::Vertex& v = vertices[nCount++];
		v.x = (float)CGAL::to_double(it->point().x());
		v.y = (float)CGAL::to_double(it->point().y());
		v.z = (float)CGAL::to_double(it->point().z());
	}
	// extract the faces
	nCount = 0;
	faces.Resize((Mesh::FIndex)p.size_of_facets());
	CGAL::Inverse_index<Polyhedron::Vertex_const_iterator> index(p.vertices_begin(), p.vertices_end());
	for (Polyhedron::Face_const_iterator it=p.facets_begin(), ite=p.facets_end(); it!=ite; ++it) {
		ASSERT(it->is_triangle());
		Polyhedron::Halfedge_around_facet_const_circulator hc = it->facet_begin();
		ASSERT(CGAL::circulator_size(hc) == 3);
		Mesh::Face& facet = faces[nCount++];
		#if 0
		Polyhedron::Halfedge_around_facet_const_circulator hc_end = hc;
		unsigned i(0);
		do {
			facet[i++] = (Mesh::FIndex)index[Polyhedron::Vertex_const_iterator(hc->vertex())];
		} while (++hc != hc_end);
		#else
		for (int i=0; i<3; ++i, ++hc)
			facet[i] = (Mesh::FIndex)index[Polyhedron::Vertex_const_iterator(hc->vertex())];
		#endif
	}
	DEBUG_ULTIMATE("Mesh exported: %u vertices, %u facets (%u border edges)", p.size_of_vertices(), p.size_of_facets(), p.size_of_border_edges());
	return true;
}
} // namespace CLN

void Mesh::EnsureEdgeSize(float epsilonMin, float epsilonMax, float collapseRatio, float degenerate_angle_deg, int mode, int max_iters)
{
	CLN::Polyhedron p;
	CLN::ImportMesh(p, vertices, faces);
	Release();
	CLN::EnsureEdgeSize(p, epsilonMin, epsilonMax, collapseRatio, degenerate_angle_deg, mode, max_iters, 0);
	CLN::ExportMesh(p, vertices, faces);
}
/*----------------------------------------------------------------*/

// subdivide mesh faces if its projection area
// is bigger than the given number of pixels
void Mesh::Subdivide(const AreaArr& maxAreas, uint32_t maxArea)
{
	ASSERT(vertexFaces.GetSize() == vertices.GetSize());

	// each face that needs to split, remember for each edge the new vertex index
	// (each new vertex index corresponds to the edge opposed to the existing vertex index)
	struct SplitFace {
		VIndex idxVert[3];
		bool bSplit;
		enum {NO_VERT = (VIndex)-1};
		inline SplitFace() : bSplit(false) { memset(idxVert, 0xFF, sizeof(VIndex)*3); }
		static VIndex FindSharedEdge(const Face& f, const Face& a) {
			for (int i=0; i<2; ++i) {
				const VIndex v(f[i]);
				if (v != a[0] && v != a[1] && v != a[2])
					return i;
			}
			ASSERT(f[2] != a[0] && f[2] != a[1] && f[2] != a[2]);
			return 2;
		}
	};
	typedef std::unordered_map<FIndex,SplitFace> FacetSplitMap;

	// used to find adjacent face
	typedef Mesh::FacetCountMap FacetCountMap;

	// for each image, compute the projection area of visible faces
	FacetSplitMap mapSplits; mapSplits.reserve(faces.GetSize());
	FacetCountMap mapFaces; mapFaces.reserve(12*3);
	vertices.Reserve(vertices.GetSize()*2);
	faces.Reserve(faces.GetSize()*3);
	const uint32_t maxAreaTh(2*maxArea);
	FOREACH(f, maxAreas) {
		const AreaArr::Type area(maxAreas[f]);
		if (area <= maxAreaTh)
			continue;
		// split face in four triangles
		// by adding a new vertex at the middle of each edge
		faces.ReserveExtra(4);
		Face& newface = faces.AddEmpty(); // defined by the three new vertices
		const Face& face = faces[(FIndex)f];
		SplitFace& split = mapSplits[(FIndex)f];
		for (int i=0; i<3; ++i) {
			// if the current edge was already split, used the existing vertex
			if (split.idxVert[i] != SplitFace::NO_VERT) {
				newface[i] = split.idxVert[i];
				continue;
			}
			// create a new vertex at the middle of the current edge
			// (current edge is the opposite edge to the current vertex index)
			split.idxVert[i] = newface[i] = vertices.GetSize();
			vertices.AddConstruct((vertices[face[(i+1)%3]]+vertices[face[(i+2)%3]])*0.5f);
		}
		// create the last three faces, defined by one old and two new vertices
		for (int i=0; i<3; ++i) {
			Face& nf = faces.AddEmpty();
			nf[0] = face[i];
			nf[1] = newface[(i+2)%3];
			nf[2] = newface[(i+1)%3];
		}
		split.bSplit = true;
		// find all three adjacent faces and inform them of the split
		ASSERT(mapFaces.empty());
		for (int i=0; i<3; ++i) {
			const Mesh::FaceIdxArr& vf = vertexFaces[face[i]];
			FOREACHPTR(pFace, vf)
				++mapFaces[*pFace].count;
		}
		for (const auto& fc: mapFaces) {
			ASSERT(fc.second.count <= 2 || (fc.second.count == 3 && fc.first == f));
			if (fc.second.count != 2)
				continue;
			if (fc.first < f && maxAreas[fc.first] > maxAreaTh) {
				// already fully split, nothing to do
				ASSERT(mapSplits[fc.first].idxVert[SplitFace::FindSharedEdge(faces[fc.first], face)] == newface[SplitFace::FindSharedEdge(face, faces[fc.first])]);
				continue;
			}
			const VIndex idxVertex(newface[SplitFace::FindSharedEdge(face, faces[fc.first])]);
			VIndex& idxSplit = mapSplits[fc.first].idxVert[SplitFace::FindSharedEdge(faces[fc.first], face)];
			ASSERT(idxSplit == SplitFace::NO_VERT || idxSplit == idxVertex);
			idxSplit = idxVertex;
		}
		mapFaces.clear();
	}

	// add all faces partially split
	int indices[3];
	for (const auto& s: mapSplits) {
		const SplitFace& split = s.second;
		if (split.bSplit)
			continue;
		int count(0);
		for (int i=0; i<3; ++i) {
			if (split.idxVert[i] != SplitFace::NO_VERT)
				indices[count++] = i;
		}
		ASSERT(count > 0);
		faces.ReserveExtra(4);
		const Face& face = faces[s.first];
		switch (count) {
		case 1: {
			// one edge is split; create two triangles
			const int i(indices[0]);
			Face& nf0 = faces.AddEmpty();
			nf0[0] = split.idxVert[i];
			nf0[1] = face[(i+2)%3];
			nf0[2] = face[i];
			Face& nf1 = faces.AddEmpty();
			nf1[0] = split.idxVert[i];
			nf1[1] = face[i];
			nf1[2] = face[(i+1)%3];
			break; }
		case 2: {
			// two edges are split; create three triangles
			const int i0(indices[0]);
			const int i1(indices[1]);
			Face& nf0 = faces.AddEmpty();
			Face& nf1 = faces.AddEmpty();
			Face& nf2 = faces.AddEmpty();
			if (i0==0) {
				if (i1==1) {
					nf0[0] = split.idxVert[1];
					nf0[1] = split.idxVert[0];
					nf0[2] = face[2];
					nf1[0] = face[0];
					nf1[1] = face[1];
					nf1[2] = split.idxVert[0];
					nf2[0] = face[0];
					nf2[1] = split.idxVert[0];
					nf2[2] = split.idxVert[1];
				} else {
					nf0[0] = split.idxVert[2];
					nf0[1] = face[1];
					nf0[2] = split.idxVert[0];
					nf1[0] = face[0];
					nf1[1] = split.idxVert[2];
					nf1[2] = face[2];
					nf2[0] = split.idxVert[2];
					nf2[1] = split.idxVert[0];
					nf2[2] = face[2];
				}
			} else {
				ASSERT(i0==1 && i1==2);
				nf0[0] = face[0];
				nf0[1] = split.idxVert[2];
				nf0[2] = split.idxVert[1];
				nf1[0] = split.idxVert[1];
				nf1[1] = face[1];
				nf1[2] = face[2];
				nf2[0] = split.idxVert[2];
				nf2[1] = face[1];
				nf2[2] = split.idxVert[1];
			}
			break; }
		case 3: {
			// all three edges are split; create four triangles
			// create the new triangle in the middle
			Face& newface = faces.AddEmpty();
			newface[0] = split.idxVert[0];
			newface[1] = split.idxVert[1];
			newface[2] = split.idxVert[2];
			// create the last three faces, defined by one old and two new vertices
			for (int i=0; i<3; ++i) {
				Face& nf = faces.AddEmpty();
				nf[0] = face[i];
				nf[1] = newface[(i+2)%3];
				nf[2] = newface[(i+1)%3];
			}
			break; }
		}
	}

	// remove all faces that split
	ASSERT(faces.GetSize()-(faces.GetCapacity()/3)/*initial size*/ > mapSplits.size());
	for (const auto& s: mapSplits)
		faces.RemoveAt(s.first);
}
/*----------------------------------------------------------------*/

// decimate mesh by removing the given list of vertices
//#define DECIMATE_JOINHOLES // not finished
void Mesh::Decimate(VertexIdxArr& verticesRemove)
{
	ASSERT(vertices.GetSize() == vertexFaces.GetSize());
	FaceIdxArr facesRemove(0, verticesRemove.GetSize()*8);
	#ifdef DECIMATE_JOINHOLES
	cList<VertexIdxArr> holes;
	#endif
	FOREACHPTR(pIdxV, verticesRemove) {
		const VIndex idxV(*pIdxV);
		ASSERT(idxV < vertices.GetSize());
		// create the list of consecutive vertices around selected vertex
		VertexIdxArr verts;
		{
			FaceIdxArr& vf(vertexFaces[idxV]);
			if (vf.IsEmpty())
				continue;
			const FIndex n(vf.GetSize());
			facesRemove.Join(vf);
			ASSERT(verts.IsEmpty());
			{
				// add vertices of the first face
				const Face& f = faces[vf.First()];
				const uint32_t i(FindVertex(f, idxV));
				verts.Insert(f[SmallMod3(i+1)]);
				verts.Insert(f[SmallMod3(i+2)]);
				vf.RemoveAt(0);
			}
			while (verts.GetSize() < n) {
				// find the face that contains our vertex and the last added vertex
				const VIndex idxVL(verts.Last());
				FOREACH(idxF, vf) {
					const Face& f = faces[vf[idxF]];
					ASSERT(FindVertex(f, idxV) != NO_ID);
					const uint32_t i(FindVertex(f, idxVL));
					if (i == NO_ID)
						continue;
					// add the missing vertex at the end
					ASSERT(f[(i+2)%3] == idxV);
					const FIndex idxVN(f[SmallMod3(i+1)]);
					ASSERT(verts.First() != idxVN);
					verts.Insert(idxVN);
					vf.RemoveAt(idxF);
					goto NEXT_FACE_FORWARD;
				}
			#ifndef DECIMATE_JOINHOLES
				vf.Release();
				goto NEXT_VERTEX;
				NEXT_FACE_FORWARD:;
			}
			vf.Release();
			#else
				break;
				NEXT_FACE_FORWARD:;
			}
			while (!vf.IsEmpty()) {
				// find the face that contains our vertex and the first added vertex
				const VIndex idxVF(verts.First());
				FOREACH(idxF, vf) {
					const Face& f = faces[vf[idxF]];
					ASSERT(FindVertex(f, idxV) != NO_ID);
					const uint32_t i(FindVertex(f, idxVF));
					if (i == NO_ID)
						continue;
					// add the missing vertex at the beginning
					ASSERT(f[(i+1)%3] == idxV);
					const FIndex idxVP(f[SmallMod3(i+2)]);
					ASSERT(verts.Last() != idxVP || vf.GetSize() == 1);
					if (verts.Last() != idxVP)
						verts.InsertAt(0, idxVP);
					vf.RemoveAt(idxF);
					goto NEXT_FACE_BACKWARD;
				}
				vf.Release();
				goto NEXT_VERTEX;
				NEXT_FACE_BACKWARD:;
			}
			#endif
		}
		// remove the deleted faces from each vertex face list
		FOREACHPTR(pV, verts) {
			FaceIdxArr& vf(vertexFaces[*pV]);
			RFOREACH(i, vf) {
				const Face& f = faces[vf[i]];
				if (FindVertex(f, idxV) != NO_ID)
					vf.RemoveAt(i);
			}
		}
		#ifdef DECIMATE_JOINHOLES
		// find the hole that contains the vertex to be deleted
		FOREACHPTR(pHole, holes) {
			const VIndex idxVH(pHole->Find(idxV));
			if (idxVH == VertexIdxArr::NO_INDEX)
				continue;
			// extend the hole with the new loop vertices
			VertexIdxArr& hole(*pHole);
			hole.RemoveAtMove(idxVH);
			const VIndex idxS((idxVH+hole.GetSize()-1)%hole.GetSize());
			const VIndex idxL(verts.Find(hole[idxS]));
			ASSERT(idxL != VertexIdxArr::NO_INDEX);
			ASSERT(verts[(idxL+verts.GetSize()-1)%verts.GetSize()] == hole[(idxS+1)%hole.GetSize()]);
			const VIndex n(verts.GetSize()-2);
			for (VIndex v=1; v<=n; ++v)
				hole.InsertAt(idxS+v, verts[(idxL+v)%verts.GetSize()]);
			goto NEXT_VERTEX;
		}
		// or create a new hole
		if (verts.GetSize() < 3)
			continue;
		verts.Swap(holes.AddEmpty());
		#else
		// close the holes defined by the complete loop of consecutive vertices
		// (the loop can be opened, cause some of the vertices can be on the border)
		if (verts.GetSize() > 2)
			CloseHoleQuality(verts);
		#endif
		NEXT_VERTEX:;
	}
	#ifndef _RELEASE
	// check all removed vertices are completely disconnected from the mesh
	FOREACHPTR(pIdxV, verticesRemove)
		ASSERT(vertexFaces[*pIdxV].IsEmpty());
	#endif

	// remove deleted faces
	RemoveFaces(facesRemove, true);

	// remove deleted vertices
	RemoveVertices(verticesRemove);

	#ifdef DECIMATE_JOINHOLES
	// close the holes defined by the complete loop of consecutive vertices
	// (the loop can be opened, cause some of the vertices can be on the border)
	FOREACHPTR(pHole, holes) {
		ASSERT(pHole->GetSize() > 2);
		CloseHoleQuality(*pHole);
	}
	#endif

	#ifndef _RELEASE
	// check all faces see valid vertices
	FOREACH(idxF, faces) {
		const Face& face = faces[idxF];
		for (int v=0; v<3; ++v)
			ASSERT(face[v] < vertices.GetSize());
	}
	#endif
}
/*----------------------------------------------------------------*/

// given a hole defined by a complete loop of consecutive vertices,
// split it recursively in two halves till the splits becomes a face
void Mesh::CloseHole(VertexIdxArr& split0)
{
	ASSERT(split0.GetSize() >= 3);
	if (split0.GetSize() == 3) {
		const FIndex idxF(faces.GetSize());
		faces.AddConstruct(split0[0], split0[1], split0[2]);
		for (int v=0; v<3; ++v) {
			#ifndef _RELEASE
			FaceIdxArr indices;
			GetAdjVertexFaces(split0[v], split0[(v+1)%3], indices);
			ASSERT(indices.GetSize() < 2);
			indices.Empty();
			GetAdjVertexFaces(split0[v], split0[(v+2)%3], indices);
			ASSERT(indices.GetSize() < 2);
			#endif
			vertexFaces[split0[v]].Insert(idxF);
		}
		return;
	}
	const VIndex i(split0.GetSize() >> 1);
	const VIndex j(split0.GetSize()-i);
	VertexIdxArr split1(0, j+1);
	split1.Join(split0.Begin()+i, j);
	split1.Insert(split0.First());
	split0.RemoveLast(j-1);
	CloseHole(split0);
	CloseHole(split1);
}

// given a hole defined by a complete loop of consecutive vertices,
// fills it using an heap to choose the best candidate face to be added
void Mesh::CloseHoleQuality(VertexIdxArr& verts)
{
	struct CandidateFace
	{
		Face face;
		float angle;
		float dihedral;
		float aspectRatio;

		CandidateFace() {}
		// the vertices of the given face must be in the order they appear on the border of the hole
		// (the middle face vertex must be between the first and third on the border)
		CandidateFace(VIndex v0, VIndex v1, VIndex v2, const Mesh& mesh) : face(v0,v1,v2) {
			const Normal n(mesh.FaceNormal(face));
			// compute the angle between the two existing edges of the face
			// (the angle computation takes into account the case of reversed face)
			angle = ACOS(ComputeAngle(mesh.vertices[face[1]].ptr(), mesh.vertices[face[0]].ptr(), mesh.vertices[face[2]].ptr()));
			if (n.dot(mesh.VertexNormal(face[1])) < 0)
				angle = float(2*M_PI) - angle;
			// compute quality as a composition of dihedral angle and area/sum(edge^2);
			// the dihedral angle uses the normal of the edge faces
			// which are possible not to exist if the edges are on the border
			FaceIdxArr indices;
			mesh.GetAdjVertexFaces(face[2], face[0], indices);
			if (indices.GetSize() > 1) {
				aspectRatio = -1;
				return;
			}
			indices.Empty();
			mesh.GetAdjVertexFaces(face[0], face[1], indices);
			if (indices.GetSize() > 1) {
				aspectRatio = -1;
				return;
			}
			const FIndex i0(indices.GetSize());
			mesh.GetAdjVertexFaces(face[1], face[2], indices);
			if (indices.GetSize()-i0 > 1) {
				aspectRatio = -1;
				return;
			}
			if (indices.IsEmpty())
				dihedral = FD2R(33.f);
			else {
				const Normal n0(mesh.FaceNormal(mesh.faces[indices[0]]));
				if (indices.GetSize() == 1)
					dihedral = ACOS(ComputeAngle(n.ptr(), n0.ptr()));
				else {
					const Normal n1(mesh.FaceNormal(mesh.faces[indices[1]]));
					dihedral = MAXF(ACOS(ComputeAngle(n.ptr(), n0.ptr())), ACOS(ComputeAngle(n.ptr(), n1.ptr())));
				}
			}
			aspectRatio = ComputeTriangleQuality(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]);
		}

		inline operator const Face&() const { return face; }
		inline bool IsConcave() const { return angle > (float)M_PI; }
		inline float GetQuality() const { return aspectRatio - 0.3f/*diedral weight*/*(dihedral/(float)M_PI); }

		// In the heap, by default, we retrieve the LARGEST value,
		// so if we need the ear with minimal dihedral angle, we must reverse the sign of the comparison.
		// The concave elements must be all in the end of the heap, sorted accordingly,
		// So if only one of the two ear is Concave that one is always the minimum one.
		inline bool operator < (const CandidateFace& c) const {
			if ( IsConcave() && !c.IsConcave()) return true;
			if (!IsConcave() &&  c.IsConcave()) return false;
			return GetQuality() < c.GetQuality();
		}
	};

	// create the initial list of new possible face along the edge of the hole
	ASSERT(verts.GetSize() > 2);
	cList<CandidateFace> candidateFaces(0, verts.GetSize());
	FOREACH(v, verts) {
		if (candidateFaces.AddConstruct(verts[v], verts[(v+1)%verts.GetSize()], verts[(v+2)%verts.GetSize()], *this).aspectRatio < 0)
			candidateFaces.RemoveLast();
	}
	candidateFaces.Sort();

	// add new faces until there are only two vertices left
	while(true) {
		// add the best candidate face
		ASSERT(!candidateFaces.IsEmpty());
		const Face& candidateFace = candidateFaces.Last();
		ASSERT(verts.Find(candidateFace[0]) != VertexIdxArr::NO_INDEX);
		ASSERT(verts.Find(candidateFace[1]) != VertexIdxArr::NO_INDEX);
		ASSERT(verts.Find(candidateFace[2]) != VertexIdxArr::NO_INDEX);
		const FIndex idxF(faces.GetSize());
		faces.Insert(candidateFace);
		for (int v=0; v<3; ++v) {
			#ifndef _RELEASE
			FaceIdxArr indices;
			GetAdjVertexFaces(candidateFace[v], candidateFace[(v+1)%3], indices);
			ASSERT(indices.GetSize() < 2);
			indices.Empty();
			GetAdjVertexFaces(candidateFace[v], candidateFace[(v+2)%3], indices);
			ASSERT(indices.GetSize() < 2);
			#endif
			vertexFaces[candidateFace[v]].Insert(idxF);
		}
		if (verts.GetSize() <= 3)
			break;
		const VIndex idxV(verts.Find(candidateFace[1]));
		// remove all candidate face containing this vertex
		{
		candidateFaces.RemoveLast();
		const VIndex idxVert(verts[idxV]);
		int n(0);
		RFOREACH(c, candidateFaces)
			if (FindVertex(candidateFaces[c].face, idxVert) != NO_ID) {
				candidateFaces.RemoveAtMove(c);
				if (++n == 2)
					break;
			}
		}
		// insert the two new candidate faces
		const VIndex idxB(idxV+verts.GetSize());
		const VIndex idxVB2(verts[(idxB-2)%verts.GetSize()]);
		const VIndex idxVB1(verts[(idxB-1)%verts.GetSize()]);
		const VIndex idxVF1(verts[(idxV+1)%verts.GetSize()]);
		const VIndex idxVF2(verts[(idxV+2)%verts.GetSize()]);
		{
			const CandidateFace newCandidateFace(idxVB2, idxVB1, idxVF1, *this);
			if (newCandidateFace.aspectRatio >= 0)
				candidateFaces.InsertSort(newCandidateFace);
		}
		{
			const CandidateFace newCandidateFace(idxVB1, idxVF1, idxVF2, *this);
			if (newCandidateFace.aspectRatio >= 0)
				candidateFaces.InsertSort(newCandidateFace);
		}
		verts.RemoveAtMove(idxV);
	}
}
/*----------------------------------------------------------------*/

// crop mesh such that none of its faces is touching or outside the given bounding-box
void Mesh::RemoveFacesOutside(const OBB3f& obb) {
	ASSERT(obb.IsValid());
	VertexIdxArr vertexRemove;
	FOREACH(i, vertices)
		if (!obb.Intersects(vertices[i]))
			vertexRemove.emplace_back(i);
	if (!vertexRemove.empty()) {
		if (vertices.size() != vertexFaces.size())
			ListIncidenteFaces();
		RemoveVertices(vertexRemove, true);
	}
}

// remove the given list of faces
void Mesh::RemoveFaces(FaceIdxArr& facesRemove, bool bUpdateLists)
{
	facesRemove.Sort();
	FIndex idxLast(FaceIdxArr::NO_INDEX);
	if (!bUpdateLists || vertexFaces.empty()) {
		RFOREACHPTR(pIdxF, facesRemove) {
			const FIndex idxF(*pIdxF);
			if (idxLast == idxF)
				continue;
			faces.RemoveAt(idxF);
			if (!faceTexcoords.empty())
				faceTexcoords.RemoveAt(idxF * 3, 3);
			idxLast = idxF;
		}
	} else {
		ASSERT(vertices.size() == vertexFaces.size());
		RFOREACHPTR(pIdxF, facesRemove) {
			const FIndex idxF(*pIdxF);
			if (idxLast == idxF)
				continue;
			{
				// remove face from vertex face list
				const Face& face = faces[idxF];
				for (int v=0; v<3; ++v) {
					const VIndex idxV(face[v]);
					FaceIdxArr& vf(vertexFaces[idxV]);
					const FIndex idx(vf.Find(idxF));
					if (idx != FaceIdxArr::NO_INDEX)
						vf.RemoveAt(idx);
				}
			}
			const FIndex idxFM(faces.size()-1);
			if (idxF < idxFM) {
				// update all vertices of the moved face
				const Face& face = faces[idxFM];
				for (int v=0; v<3; ++v) {
					const VIndex idxV(face[v]);
					FaceIdxArr& vf(vertexFaces[idxV]);
					const FIndex idx(vf.Find(idxFM));
					if (idx != FaceIdxArr::NO_INDEX)
						vf[idx] = idxF;
				}
			}
			faces.RemoveAt(idxF);
			if (!faceTexcoords.empty())
				faceTexcoords.RemoveAt(idxF * 3, 3);
			idxLast = idxF;
		}
	}
	vertexVertices.Release();
}

// remove the given list of vertices, together with all faces containing them
void Mesh::RemoveVertices(VertexIdxArr& vertexRemove, bool bUpdateLists)
{
	ASSERT(vertices.size() == vertexFaces.size());
	vertexRemove.Sort();
	VIndex idxLast(VertexIdxArr::NO_INDEX);
	if (!bUpdateLists) {
		RFOREACHPTR(pIdxV, vertexRemove) {
			const VIndex idxV(*pIdxV);
			if (idxLast == idxV)
				continue;
			const VIndex idxVM(vertices.size()-1);
			if (idxV < idxVM) {
				// update all faces of the moved vertex
				const FaceIdxArr& vf(vertexFaces[idxVM]);
				FOREACHPTR(pIdxF, vf)
					GetVertex(faces[*pIdxF], idxVM) = idxV;
			}
			vertexFaces.RemoveAt(idxV);
			vertices.RemoveAt(idxV);
			idxLast = idxV;
		}
		return;
	}
	FaceIdxArr facesRemove;
	RFOREACHPTR(pIdxV, vertexRemove) {
		const VIndex idxV(*pIdxV);
		if (idxLast == idxV)
			continue;
		const VIndex idxVM(vertices.size()-1);
		if (idxV < idxVM) {
			// update all faces of the moved vertex
			const FaceIdxArr& vf(vertexFaces[idxVM]);
			FOREACHPTR(pIdxF, vf)
				GetVertex(faces[*pIdxF], idxVM) = idxV;
		}
		if (!vertexFaces.IsEmpty()) {
			facesRemove.Join(vertexFaces[idxV]);
			vertexFaces.RemoveAt(idxV);
		}
		if (!vertexVertices.IsEmpty())
			vertexVertices.RemoveAt(idxV);
		vertices.RemoveAt(idxV);
		idxLast = idxV;
	}
	if (!facesRemove.empty())
		RemoveFaces(facesRemove);
}

// remove all vertices that are not assigned to any face
// (require vertexFaces)
Mesh::VIndex Mesh::RemoveUnreferencedVertices(bool bUpdateLists)
{
	ASSERT(vertices.size() == vertexFaces.size());
	VertexIdxArr vertexRemove;
	FOREACH(idxV, vertexFaces) {
		if (vertexFaces[idxV].empty())
			vertexRemove.push_back(idxV);
	}
	if (vertexRemove.empty())
		return 0;
	RemoveVertices(vertexRemove, bUpdateLists);
	return vertexRemove.size();
}

// convert textured mesh to store texture coordinates per vertex instead of per face
void Mesh::ConvertTexturePerVertex(Mesh& mesh) const
{
	ASSERT(HasTexture());
	mesh.vertices = vertices;
	mesh.faces.resize(faces.size());
	mesh.faceTexcoords.reserve(vertices.size()*3/2);
	mesh.faceTexcoords.resize(vertices.size());
	VertexIdxArr mapVertices(vertices.size(), vertices.size()*3/2);
	mapVertices.Memset(0xff);
	FOREACH(idxF, faces) {
		// face vertices inside a patch are simply copied;
		// face vertices on the patch boundary are duplicated,
		// with the same position, but different texture coordinates
		const Face& face = faces[idxF];
		Face& newface = mesh.faces[idxF];
		for (int i=0; i<3; ++i) {
			const TexCoord& tc = faceTexcoords[idxF*3+i];
			VIndex idxV(face[i]);
			while (true) {
				VIndex& idxVT = mapVertices[idxV];
				if (idxVT == NO_ID) {
					// vertex seen for the first time, so just copy it
					mesh.faceTexcoords[newface[i] = idxVT = idxV] = tc;
					break;
				}
				// vertex already seen in an other face, check the texture coordinates
				if (mesh.faceTexcoords[idxV] == tc) {
					// texture coordinates equal, patch interior vertex, link to it
					newface[i] = idxV;
					break;
				}
				if (idxVT == idxV) {
					// duplicate vertex, copy position, but update its texture coordinates
					mapVertices.emplace_back(newface[i] = idxVT = mesh.vertices.size());
					mesh.vertices.emplace_back(vertices[face[i]]);
					mesh.faceTexcoords.emplace_back(tc);
					break;
				}
				// continue with the next linked vertex which share the position,
				// but use different texture coordinates
				idxV = idxVT;
			}
		}
	}
	mesh.textureDiffuse = textureDiffuse;
} // ConvertTexturePerVertex
/*----------------------------------------------------------------*/


// estimate the ground-plane as the plane agreeing with most vertices
//  - sampleMesh: uniformly samples points on the mesh (0 - disabled, <0 - number of points, >0 - sample density per square unit)
//  - planeThreshold: threshold used to estimate the ground plane (0 - auto)
Planef Mesh::EstimateGroundPlane(const ImageArr& images, float sampleMesh, float planeThreshold, const String& fileExportPlane) const
{
	ASSERT(!IsEmpty());
	PointCloud pointcloud;
	if (sampleMesh != 0) {
		// create the point cloud by sampling the mesh
		if (sampleMesh > 0)
			SamplePoints(sampleMesh, 0, pointcloud);
		else
			SamplePoints(ROUND2INT<unsigned>(-sampleMesh), pointcloud);
	} else {
		// create the point cloud containing all vertices
		for (const Vertex& X: vertices)
			pointcloud.points.emplace_back(X);
	}
	return pointcloud.EstimateGroundPlane(images, planeThreshold, fileExportPlane);
}
/*----------------------------------------------------------------*/


// computes the centroid of the given mesh face
Mesh::Vertex Mesh::ComputeCentroid(FIndex idxFace) const
{
	const Face& face = faces[idxFace];
	return (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) * (Type(1)/Type(3));
}

// computes the area of the given mesh face
Mesh::Type Mesh::ComputeArea(FIndex idxFace) const
{
	const Face& face = faces[idxFace];
	return ComputeTriangleArea(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
}

// computes the area of the mesh surface as the sum of the signed areas of its faces
REAL Mesh::ComputeArea() const
{
	REAL area(0);
	for (const Face& face: faces)
		area += ComputeTriangleArea(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
	return area;
}

// computes the signed volume of the domain bounded by the mesh surface
// (note: valid only for closed and orientable manifolds)
REAL Mesh::ComputeVolume() const
{
	REAL volume(0);
	for (const Face& face: faces)
		volume += ComputeTriangleVolume(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
	return volume;
}
/*----------------------------------------------------------------*/


// project mesh to the given camera plane
void Mesh::SamplePoints(unsigned numberOfPoints, PointCloud& pointcloud) const
{
	// total mesh surface
	const REAL area(ComputeArea());
	if (area < ZEROTOLERANCE<float>()) {
		pointcloud.Release();
		return;
	}
	const REAL samplingDensity(numberOfPoints / area);
	return SamplePoints(samplingDensity, numberOfPoints, pointcloud);
}
void Mesh::SamplePoints(REAL samplingDensity, PointCloud& pointcloud) const
{
	// compute the total area to deduce the number of points
	const REAL area(ComputeArea());
	const unsigned theoreticNumberOfPoints((unsigned)CEIL2INT(area * samplingDensity));
	return SamplePoints(samplingDensity, theoreticNumberOfPoints, pointcloud);
}
void Mesh::SamplePoints(REAL samplingDensity, unsigned mumPointsTheoretic, PointCloud& pointcloud) const
{
	ASSERT(!IsEmpty());
	pointcloud.Release();
	if (mumPointsTheoretic > 0) {
		pointcloud.points.reserve(mumPointsTheoretic);
		if (HasTexture())
			pointcloud.colors.reserve(mumPointsTheoretic);
	}

	// for each triangle
	std::mt19937 rnd((std::random_device())());
	std::uniform_real_distribution<REAL> dist(0,1);
	FOREACH(idxFace, faces) {
		const Face& face = faces[idxFace];

		// vertices (OAB)
		const Vertex& O = vertices[face[0]];
		const Vertex& A = vertices[face[1]];
		const Vertex& B = vertices[face[2]];

		// edges (OA and OB)
		const Vertex u(A - O);
		const Vertex v(B - O);

		// compute triangle area
		const REAL area(norm(u.cross(v)) * REAL(0.5));

		// deduce the number of points to generate on this face
		const REAL fPointsToAdd(area*samplingDensity);
		unsigned pointsToAdd(static_cast<unsigned>(fPointsToAdd));

		// take care of the remaining fractional part;
		// add a point with the same probability as its (relative) area
		const REAL fracPart(fPointsToAdd - static_cast<REAL>(pointsToAdd));
		if (dist(rnd) <= fracPart)
			pointsToAdd++;

		for (unsigned i = 0; i < pointsToAdd; ++i) {
			// generate random points as in:
			// "Generating random points in triangles", Greg Turk;
			// in A. S. Glassner, editor, Graphics Gems, pages 24-28. Academic Press, 1990
			REAL x(dist(rnd));
			REAL y(dist(rnd));

			// test if the generated point lies on the right side of (AB)
			if (x + y > REAL(1)) {
				x = REAL(1) - x;
				y = REAL(1) - y;
			}

			// compute position
			pointcloud.points.emplace_back(O + static_cast<Vertex::Type>(x)*u + static_cast<Vertex::Type>(y)*v);

			if (HasTexture()) {
				// compute color
				const FIndex idxTexCoord(idxFace*3);
				const TexCoord& TO = faceTexcoords[idxTexCoord+0];
				const TexCoord& TA = faceTexcoords[idxTexCoord+1];
				const TexCoord& TB = faceTexcoords[idxTexCoord+2];
				const TexCoord xt(TO + static_cast<TexCoord::Type>(x)*(TA - TO) + static_cast<TexCoord::Type>(y)*(TB - TO));
				pointcloud.colors.emplace_back(textureDiffuse.sampleSafe(xt));
			}
		}
	}
}
/*----------------------------------------------------------------*/

// project mesh to the given camera plane
void Mesh::Project(const Camera& camera, DepthMap& depthMap) const
{
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		RasterMesh(const VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap)
			: Base(_vertices, _camera, _depthMap) {
		}
	};
	RasterMesh rasterer(vertices, camera, depthMap);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	for (const Face& facet : faces)
		rasterer.Project(facet, triangleRasterizer);
}

void Mesh::Project(const Camera& camera, DepthMap& depthMap, Image8U3& image) const
{
	ASSERT(!faceTexcoords.empty() && !textureDiffuse.empty());
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		const Mesh& mesh;
		Image8U3& image;
		FIndex idxFaceTex;
		TexCoord xt;
		RasterMesh(const Mesh& _mesh, const Camera& _camera, DepthMap& _depthMap, Image8U3& _image)
			: Base(_mesh.vertices, _camera, _depthMap), mesh(_mesh), image(_image) {
		}
		inline void Clear() {
			Base::Clear();
			image.memset(0);
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
			const Depth z(ComputeDepth(t, pbary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				xt = mesh.faceTexcoords[idxFaceTex + 0] * pbary[0];
				xt += mesh.faceTexcoords[idxFaceTex + 1] * pbary[1];
				xt += mesh.faceTexcoords[idxFaceTex + 2] * pbary[2];
				image(pt) = mesh.textureDiffuse.sampleSafe(xt);
			}
		}
	};
	if (image.size() != depthMap.size())
		image.create(depthMap.size());
	RasterMesh rasterer(*this, camera, depthMap, image);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	FOREACH(idxFace, faces) {
		const Face& facet = faces[idxFace];
		rasterer.idxFaceTex = idxFace * 3;
		rasterer.Project(facet, triangleRasterizer);
	}
}
// project mesh to the given camera plane, computing also the normal-map (in camera space)
void Mesh::Project(const Camera& camera, DepthMap& depthMap, NormalMap& normalMap) const
{
	ASSERT(vertexNormals.size() == vertices.size());
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		const Mesh& mesh;
		NormalMap& normalMap;
		const Face::Type* idxVerts;
		const Matrix3x3f R;
		RasterMesh(const Mesh& _mesh, const Camera& _camera, DepthMap& _depthMap, NormalMap& _normalMap)
			: Base(_mesh.vertices, _camera, _depthMap), mesh(_mesh), normalMap(_normalMap), R(camera.R) {
		}
		inline void Clear() {
			Base::Clear();
			normalMap.memset(0);
		}
		inline void Project(const Face& facet, TriangleRasterizer& tr) {
			idxVerts = facet.ptr();
			Base::Project(facet, tr);
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
			const Depth z(ComputeDepth(t, pbary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == Depth(0) || depth > z) {
				depth = z;
				normalMap(pt) = R * normalized(
					mesh.vertexNormals[idxVerts[0]] * pbary[0] +
					mesh.vertexNormals[idxVerts[1]] * pbary[1] +
					mesh.vertexNormals[idxVerts[2]] * pbary[2]
				);
			}
		}
	};
	if (normalMap.size() != depthMap.size())
		normalMap.create(depthMap.size());
	RasterMesh rasterer(*this, camera, depthMap, normalMap);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	// render the entire mesh
	for (const Face& facet : faces)
		rasterer.Project(facet, triangleRasterizer);
}
// project mesh to the given camera plane using orthographic projection
void Mesh::ProjectOrtho(const Camera& camera, DepthMap& depthMap) const
{
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		RasterMesh(const VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap)
			: Base(_vertices, _camera, _depthMap) {
		}
		inline bool ProjectVertex(const Mesh::Vertex& pt, int v, Triangle& t) {
			return (t.ptc[v] = camera.TransformPointW2C(Cast<REAL>(pt))).z > 0 &&
				depthMap.isInsideWithBorder<float, 3>(t.pti[v] = camera.TransformPointOrthoC2I(t.ptc[v]));
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Depth z(ComputeDepth(t, bary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z)
				depth = z;
		}
	};
	RasterMesh rasterer(vertices, camera, depthMap);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	for (const Face& facet : faces)
		rasterer.Project(facet, triangleRasterizer);
}
void Mesh::ProjectOrtho(const Camera& camera, DepthMap& depthMap, Image8U3& image) const
{
	ASSERT(!faceTexcoords.empty() && !textureDiffuse.empty());
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		const Mesh& mesh;
		Image8U3& image;
		FIndex idxFaceTex;
		TexCoord xt;
		RasterMesh(const Mesh& _mesh, const Camera& _camera, DepthMap& _depthMap, Image8U3& _image)
			: Base(_mesh.vertices, _camera, _depthMap), mesh(_mesh), image(_image) {
		}
		inline void Clear() {
			Base::Clear();
			image.memset(0);
		}
		inline bool ProjectVertex(const Mesh::Vertex& pt, int v, Triangle& t) {
			return (t.ptc[v] = camera.TransformPointW2C(Cast<REAL>(pt))).z > 0 &&
				depthMap.isInsideWithBorder<float, 3>(t.pti[v] = camera.TransformPointOrthoC2I(t.ptc[v]));
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Depth z(ComputeDepth(t, bary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				xt = mesh.faceTexcoords[idxFaceTex + 0] * bary[0];
				xt += mesh.faceTexcoords[idxFaceTex + 1] * bary[1];
				xt += mesh.faceTexcoords[idxFaceTex + 2] * bary[2];
				image(pt) = mesh.textureDiffuse.sampleSafe(xt);
			}
		}
	};
	if (image.size() != depthMap.size())
		image.create(depthMap.size());
	RasterMesh rasterer(*this, camera, depthMap, image);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	FOREACH(idxFace, faces) {
		const Face& facet = faces[idxFace];
		rasterer.idxFaceTex = idxFace * 3;
		rasterer.Project(facet, triangleRasterizer);
	}
}
// assuming the mesh is properly oriented, ortho-project it to a camera looking from top to down
void Mesh::ProjectOrthoTopDown(unsigned resolution, Image8U3& image, Image8U& mask, Point3& center) const
{
	ASSERT(!IsEmpty() && !textureDiffuse.empty());
	// initialize camera
	const AABB3f box(vertices.Begin(), vertices.GetSize());
	const Point3 size(Vertex(box.GetSize())*1.01f/*border*/);
	center = Vertex(box.GetCenter());
	Camera camera;
	camera.R.SetFromDirUp(Vec3(Point3(0,0,-1)), Vec3(Point3(0,1,0)));
	camera.C = center;
	camera.C.z += size.z;
	camera.K = KMatrix::IDENTITY;
	if (size.x > size.y) {
		image.create(CEIL2INT(size.y*(resolution-1)/size.x), (int)resolution);
		camera.K(0,0) = camera.K(1,1) = (resolution-1)/size.x;
	} else {
		image.create((int)resolution, CEIL2INT(size.x*(resolution-1)/size.y));
		camera.K(0,0) = camera.K(1,1) = (resolution-1)/size.y;
	}
	camera.K(0,2) = (REAL)(image.width()-1)/2;
	camera.K(1,2) = (REAL)(image.height()-1)/2;
	// project mesh
	DepthMap depthMap(image.size());
	ProjectOrtho(camera, depthMap, image);
	// create mask for the valid image pixels
	if (mask.size() != depthMap.size())
		mask.create(depthMap.size());
	for (int r=0; r<mask.rows; ++r)
		for (int c=0; c<mask.cols; ++c)
			mask(r,c) = depthMap(r,c) > 0 ? 255 : 0;
	// compute 3D coordinates for the image center
	const ImageRef xCenter(image.width()/2, image.height()/2);
	const Depth depthCenter(depthMap(xCenter));
	center = camera.TransformPointI2W(Point3(xCenter.x, xCenter.y, depthCenter > 0 ? depthCenter : camera.C.z-center.z));
}
/*----------------------------------------------------------------*/

// split mesh into sub-meshes such that each has maxArea
bool Mesh::Split(FacesChunkArr& chunks, float maxArea)
{
	TD_TIMER_STARTD();
	Octree octree;
	FacesInserter::CreateOctree(octree, *this);
	FloatArr areas(faces.size());
	FOREACH(i, faces)
		areas[i] = ComputeArea(i);
	struct AreaInserter {
		const FloatArr& areas;
		float area;
		inline void operator() (const Octree::IDX_TYPE* indices, Octree::SIZE_TYPE size) {
			FOREACHRAWPTR(pIdx, indices, size)
				area += areas[*pIdx];
		}
		inline float PopArea() {
			const float a(area);
			area = 0;
			return a;
		}
	} areaEstimator{areas, 0.f};
	struct ChunkInserter {
		const Octree& octree;
		FacesChunkArr& chunks;
		void operator() (const Octree::CELL_TYPE& parentCell, Octree::Type parentRadius, const UnsignedArr& children) {
			ASSERT(!children.empty());
			FaceChunk& chunk = chunks.AddEmpty();
			struct Inserter {
				FaceIdxArr& faces;
				inline void operator() (const Octree::IDX_TYPE* indices, Octree::SIZE_TYPE size) {
					faces.Join(indices, size);
				}
			} inserter{chunk.faces};
			if (children.size() == 1) {
				octree.CollectCells(parentCell.GetChild(children.front()), inserter);
				chunk.box = parentCell.GetChildAabb(children.front(), parentRadius);
			} else {
				chunk.box.Reset();
				for (unsigned c: children) {
					octree.CollectCells(parentCell.GetChild(c), inserter);
					chunk.box.Insert(parentCell.GetChildAabb(c, parentRadius));
				}
			}
			if (chunk.faces.empty())
				chunks.RemoveLast();
		}
	} chunkInserter{octree, chunks};
	octree.SplitVolume(maxArea, areaEstimator, chunkInserter);
	if (chunks.size() < 2)
		return false;
	DEBUG_EXTRA("Mesh split (%g max-area): %u chunks (%s)", maxArea, chunks.size(), TD_TIMER_GET_FMT().c_str());
	return true;
} // Split
/*----------------------------------------------------------------*/

// extract the sub-mesh corresponding to the given chunk of faces
Mesh Mesh::SubMesh(const FaceIdxArr& chunk) const
{
	ASSERT(!chunk.empty());
	Mesh mesh;
	mesh.vertices = vertices;
	mesh.faces.reserve(chunk.size());
	for (FIndex idxFace: chunk)
		mesh.faces.emplace_back(faces[idxFace]);
	mesh.ListIncidenteFaces();
	mesh.RemoveUnreferencedVertices();
	// fix non-manifold vertices and edges
	mesh.FixNonManifold();
	return mesh;
} // SubMesh
/*----------------------------------------------------------------*/



// transfer the texture of this mesh to the new mesh;
// the two meshes should be aligned and the new mesh to have UV-coordinates
#if USE_MESH_INT == USE_MESH_BVH
struct FaceBox {
	Eigen::AlignedBox3f box;
	Mesh::FIndex idxFace;
};
inline Eigen::AlignedBox3f bounding_box(const FaceBox& faceBox) {
	return faceBox.box;
}
#endif
bool Mesh::TransferTexture(Mesh& mesh, unsigned textureSize)
{
	ASSERT(HasTexture() && mesh.HasTexture());
	if (vertexFaces.size() != vertices.size())
		ListIncidenteFaces();
	if (mesh.vertexNormals.size() != mesh.vertices.size())
		mesh.ComputeNormalVertices();
	if (mesh.textureDiffuse.empty())
		mesh.textureDiffuse.create(textureSize, textureSize);
	#if USE_MESH_INT == USE_MESH_BVH
	std::vector<FaceBox> boxes;
	boxes.reserve(faces.size());
	FOREACH(idxFace, faces)
		boxes.emplace_back([this](FIndex idxFace) {
			const Face& face = faces[idxFace];
			Eigen::AlignedBox3f box;
			box.extend<Eigen::Vector3f>(vertices[face[0]]);
			box.extend<Eigen::Vector3f>(vertices[face[1]]);
			box.extend<Eigen::Vector3f>(vertices[face[2]]);
			return FaceBox{box, idxFace};
		} (idxFace));
	typedef Eigen::KdBVH<Type,3,FaceBox> BVH;
	BVH tree(boxes.begin(), boxes.end());
	#endif
	struct IntersectRayMesh {
		const Mesh& mesh;
		const Ray3f& ray;
		IndexDist pick;
		IntersectRayMesh(const Mesh& _mesh, const Ray3f& _ray)
			: mesh(_mesh), ray(_ray) {
			#if USE_MESH_INT == USE_MESH_BF
			FOREACH(idxFace, mesh.faces)
				IntersectsRayFace(idxFace);
			#endif
		}
		inline void IntersectsRayFace(FIndex idxFace) {
			const Face& face = mesh.faces[idxFace];
			Type dist;
			if (ray.Intersects<true>(Triangle3f(mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]), &dist)) {
				ASSERT(dist >= 0);
				if (pick.dist > dist) {
					pick.dist = dist;
					pick.idx = idxFace;
				}
			}
		}
		#if USE_MESH_INT == USE_MESH_BVH
		inline bool intersectVolume(const BVH::Volume &volume) {
			return ray.Intersects(AABB3f(volume.min(), volume.max()));
		}
		inline bool intersectObject(const BVH::Object &object) {
			IntersectsRayFace(object.idxFace);
			return false;
		}
		#endif
	};
	#if USE_MESH_INT == USE_MESH_BF || USE_MESH_INT == USE_MESH_BVH
	const float diagonal(GetAABB().GetSize().norm());
	#elif USE_MESH_INT == USE_MESH_OCTREE
	const Octree octree(vertices, [](Octree::IDX_TYPE size, Octree::Type /*radius*/) {
		return size > 8;
	});
	const float diagonal(octree.GetAabb().GetSize().norm());
	struct OctreeIntersectRayMesh : IntersectRayMesh {
		OctreeIntersectRayMesh(const Octree& octree, const Mesh& _mesh, const Ray3f& _ray)
			: IntersectRayMesh(_mesh, _ray) {
			octree.Collect(*this, *this);
		}
		inline bool Intersects(const Octree::POINT_TYPE& center, Octree::Type radius) const {
			return ray.Intersects(AABB3f(center, radius));
		}
		void operator () (const Octree::IDX_TYPE* idices, Octree::IDX_TYPE size) {
			// store all contained faces only once
			std::unordered_set<FIndex> set;
			FOREACHRAWPTR(pIdx, idices, size) {
				const VIndex idxVertex((VIndex)*pIdx);
				const FaceIdxArr& faces = mesh.vertexFaces[idxVertex];
				set.insert(faces.begin(), faces.end());
			}
			// test face intersection and keep the closest
			for (FIndex idxFace : set)
				IntersectsRayFace(idxFace);
		}
	};
	#endif
	#ifdef MESH_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int_t i=0; i<(int_t)mesh.faces.size(); ++i) {
		const FIndex idxFace((FIndex)i);
	#else
	FOREACH(idxFace, mesh.faces) {
	#endif
		struct RasterTraiangle {
			#if USE_MESH_INT == USE_MESH_OCTREE
			const Octree& octree;
			#elif USE_MESH_INT == USE_MESH_BVH
			BVH& tree;
			#endif
			const Mesh& meshRef;
			Mesh& meshTrg;
			const Face& face;
			float diagonal;
			inline cv::Size Size() const { return meshTrg.textureDiffuse.size(); }
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				ASSERT(meshTrg.textureDiffuse.isInside(pt));
				const Vertex X(meshTrg.vertices[face[0]]*bary.x + meshTrg.vertices[face[1]]*bary.y + meshTrg.vertices[face[2]]*bary.z);
				const Normal N(normalized(meshTrg.vertexNormals[face[0]]*bary.x + meshTrg.vertexNormals[face[1]]*bary.y + meshTrg.vertexNormals[face[2]]*bary.z));
				const Ray3f ray(Vertex(X+N*diagonal), Normal(-N));
				#if USE_MESH_INT == USE_MESH_BF
				const IntersectRayMesh intRay(meshRef, ray);
				#elif USE_MESH_INT == USE_MESH_BVH
				IntersectRayMesh intRay(meshRef, ray);
				Eigen::BVIntersect(tree, intRay);
				#else
				const OctreeIntersectRayMesh intRay(octree, meshRef, ray);
				#endif
				if (intRay.pick.IsValid()) {
					const FIndex refIdxFace((FIndex)intRay.pick.idx);
					const Face& refFace = meshRef.faces[refIdxFace];
					const Vertex refX(ray.GetPoint((Type)intRay.pick.dist));
					const Vertex baryRef(CorrectBarycentricCoordinates(BarycentricCoordinatesUV(meshRef.vertices[refFace[0]], meshRef.vertices[refFace[1]], meshRef.vertices[refFace[2]], refX)));
					const TexCoord* tri = meshRef.faceTexcoords.data()+refIdxFace*3;
					const TexCoord x(tri[0]*baryRef.x + tri[1]*baryRef.y + tri[2]*baryRef.z);
					const Pixel8U color(meshRef.textureDiffuse.sample(x));
					meshTrg.textureDiffuse(pt) = color;
				}
			}
		#if USE_MESH_INT == USE_MESH_BF
		} data{*this, mesh, mesh.faces[idxFace], diagonal};
		#elif USE_MESH_INT == USE_MESH_BVH
		} data{tree, *this, mesh, mesh.faces[idxFace], diagonal};
		#else
		} data{octree, *this, mesh, mesh.faces[idxFace], diagonal};
		#endif
		// render triangle and for each pixel interpolate the color
		// from the triangle corners using barycentric coordinates
		const TexCoord* tri = mesh.faceTexcoords.data()+idxFace*3;
		Image8U::RasterizeTriangleBary(tri[0], tri[1], tri[2], data);
	}
	return true;
} // TransferTexture
/*----------------------------------------------------------------*/


#ifdef _USE_CUDA
CUDA::KernelRT Mesh::kernelComputeFaceNormal;

bool Mesh::InitKernels(int device)
{
	// kernelComputeFaceNormal is a KernelRT, which is built on the pre-CUDA-12
	// launch API; bail out cleanly on a driver that no longer exports it so
	// ComputeNormalFaces() takes its CPU branch instead of faulting on a
	// delay-load stub. See CUDA::HasLegacyDriverAPI().
	if (!CUDA::HasLegacyDriverAPI())
		return false;

	// initialize CUDA device if needed
	if (CUDA::devices.IsEmpty() && CUDA::initDevice(device) != CUDA_SUCCESS)
		return false;

	// initialize CUDA kernels
	if (!kernelComputeFaceNormal.IsValid()) {
		// kernel used to compute face normal, given the array of face vertices and vertex positions
		STATIC_ASSERT(sizeof(Vertex) == sizeof(float)*3);
		STATIC_ASSERT(sizeof(Face) == sizeof(VIndex)*3 && sizeof(VIndex) == sizeof(uint32_t));
		STATIC_ASSERT(sizeof(Normal) == sizeof(float)*3);
		#define FUNC "ComputeFaceNormal"
		LPCSTR const szKernel =
			".version 3.2\n"
			".target sm_20\n"
			".address_size 64\n"
			"\n"
			".visible .entry " FUNC "(\n"
			"	.param .u64 .ptr param_1, // array vertices (float*3 * numVertices)\n"
			"	.param .u64 .ptr param_2, // array faces (uint32_t*3 * numFaces)\n"
			"	.param .u64 .ptr param_3, // array normals (float*3 * numFaces) [out]\n"
			"	.param .u32 param_4 // numFaces = numNormals (uint32_t)\n"
			")\n"
			"{\n"
			"	.reg .f32 %f<32>;\n"
			"	.reg .pred %p<2>;\n"
			"	.reg .u32 %r<17>;\n"
			"	.reg .u64 %rl<18>;\n"
			"\n"
			"	ld.param.u64 %rl4, [param_1];\n"
			"	ld.param.u64 %rl5, [param_2];\n"
			"	ld.param.u64 %rl6, [param_3];\n"
			"	ld.param.u32 %r2,  [param_4];\n"
			"	cvta.to.global.u64 %rl1, %rl6;\n"
			"	cvta.to.global.u64 %rl2, %rl4;\n"
			"	cvta.to.global.u64 %rl3, %rl5;\n"
			"	mov.u32 %r3, %ntid.x;\n"
			"	mov.u32 %r4, %ctaid.x;\n"
			"	mov.u32 %r5, %tid.x;\n"
			"	mad.lo.u32 %r1, %r3, %r4, %r5;\n"
			"	setp.ge.u32 %p1, %r1, %r2;\n"
			"	@%p1 bra BB00_1;\n"
			"\n"
			"	mul.lo.u32 %r6, %r1, 3;\n"
			"	mul.wide.u32 %rl7, %r6, 4;\n"
			"	add.u64 %rl8, %rl3, %rl7;\n"
			"	ld.global.u32 %r8, [%rl8];\n"
			"	mul.lo.u32 %r10, %r8, 3;\n"
			"	mul.wide.u32 %rl11, %r10, 4;\n"
			"	add.u64 %rl12, %rl2, %rl11;\n"
			"	ld.global.u32 %r11, [%rl8+4];\n"
			"	mul.lo.u32 %r13, %r11, 3;\n"
			"	mul.wide.u32 %rl13, %r13, 4;\n"
			"	add.u64 %rl14, %rl2, %rl13;\n"
			"	ld.global.u32 %r14, [%rl8+8];\n"
			"	mul.lo.u32 %r16, %r14, 3;\n"
			"	mul.wide.u32 %rl15, %r16, 4;\n"
			"	add.u64 %rl16, %rl2, %rl15;\n"
			"\n"
			"	ld.global.f32 %f1, [%rl14];\n"
			"	ld.global.f32 %f2, [%rl12];\n"
			"	sub.f32 %f3, %f1, %f2;\n"
			"	ld.global.f32 %f4, [%rl14+4];\n"
			"	ld.global.f32 %f5, [%rl12+4];\n"
			"	sub.f32 %f6, %f4, %f5;\n"
			"	ld.global.f32 %f7, [%rl14+8];\n"
			"	ld.global.f32 %f8, [%rl12+8];\n"
			"	sub.f32 %f9, %f7, %f8;\n"
			"	ld.global.f32 %f10, [%rl16];\n"
			"	sub.f32 %f11, %f10, %f2;\n"
			"	ld.global.f32 %f12, [%rl16+4];\n"
			"	sub.f32 %f13, %f12, %f5;\n"
			"	ld.global.f32 %f14, [%rl16+8];\n"
			"	sub.f32 %f15, %f14, %f8;\n"
			"\n"
			"	mul.f32 %f16, %f6, %f15;\n"
			"	neg.f32 %f17, %f9;\n"
			"	fma.rn.f32 %f18, %f17, %f13, %f16;\n"
			"	mul.f32 %f19, %f9, %f11;\n"
			"	neg.f32 %f20, %f3;\n"
			"	fma.rn.f32 %f21, %f20, %f15, %f19;\n"
			"	mul.f32 %f22, %f3, %f13;\n"
			"	neg.f32 %f23, %f6;\n"
			"\n"
			"	fma.rn.f32 %f24, %f23, %f11, %f22;\n"
			"	mul.f32 %f25, %f21, %f21;\n"
			"	fma.rn.f32 %f26, %f18, %f18, %f25;\n"
			"	fma.rn.f32 %f27, %f24, %f24, %f26;\n"
			"\n"
			"	sqrt.rn.f32 %f28, %f27;\n"
			"	div.rn.f32 %f29, %f18, %f28;\n"
			"	div.rn.f32 %f30, %f21, %f28;\n"
			"	div.rn.f32 %f31, %f24, %f28;\n"
			"\n"
			"	add.u64 %rl17, %rl1, %rl7;\n"
			"	st.global.f32 [%rl17], %f29;\n"
			"	st.global.f32 [%rl17+4], %f30;\n"
			"	st.global.f32 [%rl17+8], %f31;\n"
			"\n"
			"	BB00_1:\n"
			"	ret;\n"
			"}\n";
		if (kernelComputeFaceNormal.Reset(szKernel, FUNC) != CUDA_SUCCESS)
			return false;
		ASSERT(kernelComputeFaceNormal.IsValid());
		#undef FUNC
	}

	return true;
}
/*----------------------------------------------------------------*/
#endif
