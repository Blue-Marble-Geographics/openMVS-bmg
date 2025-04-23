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

// Easier to configure this here.
#pragma comment(linker, "/STACK:0x400000,0x400000")

#include "Common.h"
#include "Scene.h"
// Delaunay: mesh reconstruction
#include <CGAL/circulator.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polyhedron_3.h>
#include <boost/container/small_vector.hpp>
#include <algorithm>
#include <execution>
#include <chrono>
#include <functional>
#include "P2PUtils.h"
#include <condition_variable>
#include <thread>
#include <concurrent_queue.h>
#include <CGAL/Triangulation_3.h>         // base class for all 3D triangulations
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>  // a common kernel
#include <CGAL/point_generators_3.h>      // if you're generating test points
#include <CGAL/squared_distance_3.h>      // for squared distance functions
#include <boost/container/flat_set.hpp>

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/stat.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/smooth.h>
#include <vcg/complex/algorithms/hole.h>
#include <vcg/complex/algorithms/polygon_support.h>
#include <vcg/complex/algorithms/isotropic_remeshing.h>
// VCG: mesh simplification
#include <vcg/complex/algorithms/update/position.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/selection.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

using namespace MVS;
using namespace concurrency;
using namespace std::chrono_literals;

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DELAUNAY_USE_OPENMP
#endif

// uncomment to enable reconstruction algorithm of weakly supported surfaces
#define DELAUNAY_WEAKSURF

// uncomment to use IBFS algorithm for max-flow
// (faster, but not clear license policy)
#define DELAUNAY_MAXFLOW_IBFS

#define FASTER_WEIGHTING
#define FASTER_ADJ_VERTICES
#define FASTER_ESTIMATES // Verified the same (Sigma value) 30% faster
#undef USE_PARALLEL_GRAPHCUT_FIXUP

// S T R U C T S ///////////////////////////////////////////////////

#ifdef DELAUNAY_MAXFLOW_IBFS
#include "../Math/IBFS/IBFS.h"
template <typename NType, typename VType>
class MaxFlow
{
public:
	// Type-Definitions
	typedef NType node_type;
	typedef IBFS::IBFSGraph graph_type;
	// Removed value_type.  Externally, always works in double.
	// Internally, in IBFS, it works as EdgeCap

public:
	MaxFlow(size_t numNodes) {
		graph.initSize((int)numNodes, (int)numNodes*2);
	}

	inline float AddNode(const void* n, double source, double sink) {
		ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
		return graph.addNode(n, source, sink);
	}

	inline void AddEdge(const void* nhf, const void* nht, double capacity, double reverseCapacity) {
		ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
		graph.addEdge(nhf, nht, capacity, reverseCapacity);
	}

	inline void InitFlow(double flow) {
		graph.initFlow(flow);
	}

	double ComputeMaxFlow() {
		graph.initGraph();
		return graph.computeMaxFlow();
	}

	inline bool IsNodeOnSrcSide(const void* node) const {
		return graph.isNodeOnSrcSide(node);
	}

	inline const void* NodeHandle(node_type n) const {
		return graph.nodeHandle((int)n);
	}


	float getflow() const // JPB WIP BUG
	{
		return graph.flow;
	}

protected:
	graph_type graph;
};

#else
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
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

__forceinline double __vectorcall fast_sqdist(_DataD ax, _DataD ay, _DataD az, const point_t& b) {
	_DataD bx = _SetD(b.x());
	_DataD dx = _SubD(ax, bx);
	dx = _MulD(dx, dx);

	_DataD by = _SetD(b.y());
	_DataD dy = _SubD(ay, by);
	dy = _MulD(dy, dy);

	_DataD bz = _SetD(b.z());
	_DataD dz = _SubD(az, bz);
	dz = _MulD(dz, dz);

	_DataD sum = _AddD(_AddD(dx, dy), dz);
	return _vFirstD(sum);
}

#ifdef DELAUNAY_WEAKSURF
struct view_info_t;
#endif

struct vert_info_t {
	typedef edge_cap_t Type;
	struct view_t {
		PointCloud::View idxView; // view index
		float weight;
		inline view_t() {}
		inline view_t(PointCloud::View _idxView, Type _weight) : idxView(_idxView), weight(_weight) {}
		inline bool operator <(const view_t& v) const { return idxView < v.idxView; }
		inline operator PointCloud::View() const { return idxView; }
	};
	typedef SEACAVE::cList<view_t,const view_t&,0,4,uint32_t> view_vec_t;
	view_vec_t views; // faces' weight from the cell outwards
	void InsertViews(const PointCloudStreaming& pc, PointCloud::Index idxPoint) {
		const uint32_t* _views = pc.ViewsStream(idxPoint);
		const uint32_t cnt = (uint32_t) pc.ViewsStreamSize(idxPoint);
		for (uint32_t i = 0; i < cnt; ++i) {
			const PointCloud::View viewID(_views[i]);
			// insert viewID in increasing order
			const uint32_t idx(views.FindFirstEqlGreater(viewID));
			if (idx < views.GetSize() && views[idx] == viewID) {
				// the new view is already in the array
				ASSERT(views.FindFirst(viewID) == idx);
				// update point's weight
				views[idx].weight++;
			} else {
				// the new view is not in the array,
				// insert it
				views.InsertAt(idx, view_t(viewID, 1.f));
				ASSERT(views.IsSorted());
			}
		}
	}
};

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
typedef CGAL::Delaunay_triangulation_3<kernel_t, triangulation_data_structure_t, CGAL::Fast_location> delaunay_t;
typedef delaunay_t::Vertex_handle vertex_handle_t;
typedef delaunay_t::Cell_handle cell_handle_t;
typedef delaunay_t::Facet facet_t;
typedef delaunay_t::Edge edge_t;

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

typedef TPoint3<kernel_t::RT> DPoint3;
template <typename TYPE>
__forceinline TPoint3<TYPE> CGAL2MVS(const point_t& p) {
	return TPoint3<TYPE>((TYPE)p.x(), (TYPE)p.y(), (TYPE)p.z());
}
template <typename TYPE>
__forceinline point_t MVS2CGAL(const TPoint3<TYPE>& p) {
	return point_t((kernel_t::RT)p.x, (kernel_t::RT)p.y, (kernel_t::RT)p.z);
}

// Given a facet, compute the plane containing it
__forceinline Plane getFacetPlane(const facet_t& facet)
{
	const point_t& v0(facet.first->vertex((facet.second+1)%4)->point());
	const point_t& v1(facet.first->vertex((facet.second+2)%4)->point());
	const point_t& v2(facet.first->vertex((facet.second+3)%4)->point());
	return Plane(CGAL2MVS<REAL>(v0), CGAL2MVS<REAL>(v1), CGAL2MVS<REAL>(v2));
}

// Check if a point (p) is coplanar with a triangle (a, b, c);
// return orientation type
#if _PLATFORM_X86 && defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target ("no-fma")
#endif

static inline int fasterOrientation(const double* __restrict qDiff, const double* __restrict aDiff, const double* __restrict bDiff)
{
	// inexact_orientation
	const double pqx(qDiff[0]); const double prx(aDiff[0]); const double psx(bDiff[0]);
	const double pqy(qDiff[1]); const double pry(aDiff[1]); const double psy(bDiff[1]);
	const double det((pqx*pry-prx*pqy)*(bDiff[2]) - (pqx*psy-psx*pqy)*(aDiff[2]) + (prx*psy-psx*pry)*(qDiff[2]));
	constexpr double eps(1e-12);
	if (det >  eps) return CGAL::POSITIVE;
	if (det < -eps) return CGAL::NEGATIVE;
	return CGAL::COPLANAR;
}

static inline int orientation(const point_t& a, const point_t& b, const point_t& c, const point_t& p)
{
	#if 0
	return CGAL::orientation(a, b, c, p);
	#else
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
void fetchCellFacets(const delaunay_t& Tr, const std::vector<facet_t>& hullFacets, const cell_handle_t& cell, const Image& imageData, std::vector<facet_t>& facets)
{
	if (!Tr.is_infinite(cell)) {
		// store all 4 facets of the cell
		for (int i=0; i<4; ++i) {
			const facet_t f(cell, i);
			ASSERT(!Tr.is_infinite(f));
			facets.push_back(f);
		}
		return;
	}
	// find all facets on the convex-hull in camera's view
	// create the 4 frustum planes
	ASSERT(facets.empty());
	typedef TFrustum<REAL,4> Frustum;
	Frustum frustum(imageData.camera.P, imageData.width, imageData.height, 0, 1);
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
		if (frustum.Classify(ab) == CULLED)
			continue;
		facets.push_back(face);
	}
}


// information about an intersection between a segment and a facet
struct intersection_t {
	enum Type {FACET, EDGE, VERTEX};
	cell_handle_t ncell; // cell neighbor to the last intersected facet
	vertex_handle_t v1; // vertex for vertex intersection, 1st edge vertex for edge intersection
	vertex_handle_t v2; // 2nd edge vertex for edge intersection
	facet_t facet; // intersected facet
	Type type; // type of intersection (inside facet, on edge, or vertex)
	REAL dist; // distance from starting point (camera) to this facet
	bool bigger; // are we advancing away or towards the starting point?
	const Ray3 ray; // the ray from starting point into the direction of the end point (point -> camera/end-point)
	inline intersection_t() {}
	inline intersection_t(const Point3& pt, const Point3& dir) : dist(-FLT_MAX), bigger(true), ray(pt, dir) {}
};

#ifndef FASTER_WEIGHTING
// Check if a segment (p, q) is coplanar with edges of a triangle (a, b, c):
//  coplanar [in,out] : pointer to the 3 int array of indices of the edges coplanar with pq
// return number of entries in coplanar
inline int checkEdges(const point_t& a, const point_t& b, const point_t& c, const point_t& p, const point_t& q, int coplanar[3])
{
	int nCoplanar(0);
	switch (orientation(p,q,a,b)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 0;
	}
	switch (orientation(p,q,b,c)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 1;
	}
	switch (orientation(p,q,c,a)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 2;
	}
	return nCoplanar;
}

// Check intersection between a facet (f) and a segment (s)
// (derived from CGAL::do_intersect in CGAL/Triangle_3_Segment_3_do_intersect.h)
//  coplanar [out] : pointer to the 3 int array of indices of the edges coplanar with (s)
// return -1 if there is no intersection or
// the number of edges coplanar with the segment (0 = intersection inside the triangle)
int intersect(const triangle_t& t, const segment_t& s, int coplanar[3])
{
	const point_t& a = t.vertex(0);
	const point_t& b = t.vertex(1);
	const point_t& c = t.vertex(2);
	const point_t& p = s.source();
	const point_t& q = s.target();

	switch (orientation(a,b,c,p)) {
	case CGAL::POSITIVE:
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// the segment lies in the positive open halfspaces defined by the
			// triangle's supporting plane
			return -1;
		case CGAL::COPLANAR:
			// q belongs to the triangle's supporting plane
			// p sees the triangle in counterclockwise order
			return checkEdges(a,b,c,p,q,coplanar);
		case CGAL::NEGATIVE:
			// p sees the triangle in counterclockwise order
			return checkEdges(a,b,c,p,q,coplanar);
		default:
			break;
		}
	case CGAL::NEGATIVE:
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// q sees the triangle in counterclockwise order
			return checkEdges(a,b,c,q,p,coplanar);
		case CGAL::COPLANAR:
			// q belongs to the triangle's supporting plane
			// p sees the triangle in clockwise order
			return checkEdges(a,b,c,q,p,coplanar);
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
			return checkEdges(a,b,c,q,p,coplanar);
		case CGAL::COPLANAR:
			// the segment is coplanar with the triangle's supporting plane
			// as we know that it is inside the tetrahedron it intersects the face
			//coplanar[0] = coplanar[1] = coplanar[2] = 3;
			return 3;
		case CGAL::NEGATIVE:
			// q sees the triangle in clockwise order
			return checkEdges(a,b,c,p,q,coplanar);
		default:
			break;
		}
	}
	ASSERT("should not happen" == NULL);
	return -1;
}

// Find which facet is intersected by the segment (seg) and return next facets to check:
//  in_facets [in] : vector of facets to check
//  out_facets [out] : vector of facets to check at next step (can be in_facets)
//  out_inter [out] : kind of intersection
// return false if no intersection found and the end of the segment was not reached
bool intersect(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	ASSERT(!in_facets.empty());
	static const int facet_vertex_order[] = {2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0};
	int coplanar[3];
	const REAL prevDist(inter.dist);
	for (const facet_t& in_facet: in_facets) {
		ASSERT(!Tr.is_infinite(in_facet));
		const int nb_coplanar(intersect(Tr.triangle(in_facet), seg, coplanar));
		if (nb_coplanar >= 0) {
			// skip this cell if the intersection is not in the desired direction
			const REAL interDist(inter.ray.IntersectsDist(getFacetPlane(in_facet)));
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

inline int checkEdges2(const double* __restrict negPa, const point_t& b, const point_t& c, const point_t& p, const double* __restrict qDiff, int* __restrict coplanar)
{
	int nCoplanar(0);
	const double aDiff[] { -negPa[0], -negPa[1], -negPa[2]};
	const double bDiff[] { b.x()-p.x(), b.y()-p.y(), b.z()-p.z() };

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

// Check intersection between a facet (f) and a segment (s)
// (derived from CGAL::do_intersect in CGAL/Triangle_3_Segment_3_do_intersect.h)
//  coplanar [out] : pointer to the 3 int array of indices of the edges coplanar with (s)
// return -1 if there is no intersection or
// the number of edges coplanar with the segment (0 = intersection inside the triangle)
int intersect(const triangle_t& t, const segment_t& s, const double* __restrict segDiff /* target - source */, const double* __restrict segDiffN /* source - target */, int* __restrict coplanar)
{
	const point_t& a = t.vertex(0);
	const point_t& b = t.vertex(1);
	const point_t& c = t.vertex(2);
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
					return checkEdges2(pDiff,b,c,p,segDiff,coplanar);
				case CGAL::NEGATIVE:
					// p sees the triangle in counterclockwise order
					//return checkEdges(a,b,c,p,q,coplanar);
					return checkEdges2(pDiff,b,c,p,segDiff,coplanar);
				default:
					break;
				}
		case CGAL::NEGATIVE:
			switch (fasterOrientation(bDiff, cDiff, qDiff)) { //orientation(a,b,c,q)) {
				case CGAL::POSITIVE:
					// q sees the triangle in counterclockwise order
					//return checkEdges(a,b,c,q,p,coplanar);
					return checkEdges2(qDiff,b,c,q,segDiffN,coplanar);
				case CGAL::COPLANAR:
					// q belongs to the triangle's supporting plane
					// p sees the triangle in clockwise order
					//return checkEdges(a,b,c,q,p,coplanar);
					return checkEdges2(qDiff,b,c,q,segDiffN,coplanar);
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
					return checkEdges2(qDiff,b,c,q,segDiffN,coplanar);
				case CGAL::COPLANAR:
					// the segment is coplanar with the triangle's supporting plane
					// as we know that it is inside the tetrahedron it intersects the face
					//coplanar[0] = coplanar[1] = coplanar[2] = 3;
					return 3;
				case CGAL::NEGATIVE:
					// q sees the triangle in clockwise order
					//return checkEdges(a,b,c,p,q,coplanar);
					return checkEdges2(pDiff,b,c,p,segDiff,coplanar);
				default:
					break;
				}
	}
	ASSERT("should not happen" == NULL);
	return -1;
}

// Find which facet is intersected by the segment (seg) and return next facets to check:
//  in_facets [in] : vector of facets to check
//  out_facets [out] : vector of facets to check at next step (can be in_facets)
//  out_inter [out] : kind of intersection
// return false if no intersection found and the end of the segment was not reached
bool intersect(const delaunay_t& Tr, const segment_t& seg, const double* __restrict segDiff, const double* __restrict segDiff2, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	ASSERT(!in_facets.empty());
	static const int facet_vertex_order[] = {2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0};
	int coplanar[3];
	const REAL prevDist(inter.dist);
	for (const facet_t& in_facet: in_facets) {
		ASSERT(!Tr.is_infinite(in_facet));
		const int nb_coplanar(intersect(Tr.triangle(in_facet), seg, segDiff, segDiff2, coplanar));
		if (nb_coplanar >= 0) {
			// skip this cell if the intersection is not in the desired direction
			const REAL interDist(inter.ray.IntersectsDist(getFacetPlane(in_facet)));
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
						out_facets.emplace_back(nc, i);
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
					inline cell_back_inserter_t(const delaunay_t& _Tr, const intersection_t& inter,std::vector<facet_t>& _out_facets)
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
#endif // FASTER_WEIGHTING

#if 0 // WIP
// same as above, but simplified only to find face intersection (otherwise terminate);
// terminate if cell containing the segment endpoint is found or if an infinite cell is encountered
bool intersectFace(const delaunay_t& Tr, const segment_t& seg, std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	int coplanar[3];
	for (auto it=in_facets.cbegin(); it!=in_facets.cend(); ++it) {
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
	auto& in_facets = out_facets;
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
	return fn.dot(ct)/SQRT(fnLenSq*ctLenSq);
}

#undef CUT_TIMINGS // To profile each step.
void graphcut(std::vector<delaunay_t::All_cells_iterator>& cellIterators, delaunay_t& delaunay, std::vector<cell_info_t>& infoCells, Mesh& mesh, float kQual)
{
#ifndef CUT_TIMINGS
	TD_TIMER_STARTD();
#endif

	MaxFlow<cell_size_t,float> graph(delaunay.number_of_cells());
	const __int64 idxCount = cellIterators.size();
	double flow = 0;
	{
#ifdef CUT_TIMINGS
		TD_TIMER_STARTD();
#endif
		// create graph
		// set weights
		constexpr float maxCap(FLT_MAX*0.0001f);
		// JPB WIP OPT Revisit for parallel
		for (__int64 i = 0; i < idxCount; ++i) {
			auto ci = cellIterators[i];
			const cell_size_t ciID(ci->info());
			const void* nhi = graph.NodeHandle(ciID);
			const cell_info_t& ciInfo(infoCells[ciID]);
			graph.AddNode(nhi, ciInfo.s, MINF(ciInfo.t, maxCap));
			for (int j=0; j<4; ++j) {
				const cell_handle_t cj(ci->neighbor(j));
				const cell_size_t cjID(cj->info());
				if (cjID < ciID) continue;
				const void* nhj = graph.NodeHandle(cjID);
				const cell_info_t& cjInfo(infoCells[cjID]);
				const int k(cj->index(ci));
				const edge_cap_t q((1.f - CLAMP(MINF(computePlaneSphereAngle(delaunay, facet_t(ci,j)), computePlaneSphereAngle(delaunay, facet_t(cj,k))),  -1.f, 1.f))*kQual);
				graph.AddEdge(nhi, nhj, ciInfo.f[j]+q, cjInfo.f[k]+q);
			}
		}
#ifdef CUT_TIMINGS
		DEBUG_EXTRA("%s", TD_TIMER_GET_FMT().c_str());
#endif
	}

	double maxFlow;

	{
#ifdef CUT_TIMINGS
		TD_TIMER_STARTD();
#endif

		infoCells.clear();
		// find graph-cut solution
		graph.InitFlow(flow);
		maxFlow = graph.ComputeMaxFlow();
#ifdef CUT_TIMINGS
		DEBUG_EXTRA("%s", TD_TIMER_GET_FMT().c_str());
#endif
	}

#ifdef CUT_TIMINGS
	{
	TD_TIMER_STARTD();
#endif

	// JPB WIP OPT Recheck this.
	// Although the parallel version is much faster, it appears to emit unique
	// vertices that are not cache friendly in later stages.
	// The modest performance slowdown using the serial version
	// dwarfs the slowdown that occurs later.
#ifdef USE_PARALLEL_GRAPHCUT_FIXUP
	const size_t nEstimatedNumVerts(delaunay.number_of_vertices());
		tbb::concurrent_hash_map<void*,Mesh::VIndex> mapVertices(nEstimatedNumVerts);
		mesh.vertices.Resize((Mesh::VIndex)nEstimatedNumVerts);
		mesh.faces.Resize(idxCount*4); // Upper bound
		std::atomic<int> numPoints = 0;
		std::atomic<int> numFaces = 0;
#pragma omp parallel for schedule(static, 4096)
		for (__int64 i = 0; i < idxCount; ++i) {
			auto ci = cellIterators[i];
			const cell_size_t ciID(ci->info());
			const void* nodeHandleI = graph.NodeHandle(ciID);
			const bool ciType(graph.IsNodeOnSrcSide(nodeHandleI));
			for (int j=0; j<4; ++j) {
				if (delaunay.is_infinite(ci, j)) continue;
				const cell_handle_t cj(ci->neighbor(j));
				const cell_size_t cjID(cj->info());
				if (ciID < cjID) continue;
				const void* nodeHandleJ = graph.NodeHandle(cjID);
				if (ciType == graph.IsNodeOnSrcSide(nodeHandleJ)) continue;
				const int myIndex = numFaces++;

				Mesh::Face& face = mesh.faces[myIndex]; //mesh.faces.AddEmpty();
				const triangle_vhandles_t tri(getTriangle(ci, j));
				for (int v=0; v<3; ++v) {
					const vertex_handle_t vh(tri.verts[v]);
					ASSERT(vh->point() == delaunay.triangle(ci,j)[v]);
				  tbb::concurrent_hash_map<void*, Mesh::VIndex>::accessor accessor;
					if (mapVertices.insert(accessor, vh.for_compact_container())) {
						const int myVertexIdx = numPoints++;
						// Element was inserted.
						accessor->second = (Mesh::VIndex)myVertexIdx;
						mesh.vertices[myVertexIdx] = CGAL2MVS<Mesh::Vertex::Type>(vh->point());
					}				
					//ASSERT(pairItID.first->second < mesh.vertices.GetSize());
					face[v] = accessor->second;
				}
				// correct face orientation
				if (!ciType)
					std::swap(face[0], face[2]);
			}
		}
		mesh.vertices.ResizeExact(numPoints.load(std::memory_order_relaxed));
		mesh.faces.ResizeExact(numFaces.load(std::memory_order_relaxed));
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
		const void* nodeHandleI = graph.NodeHandle(ciID);
		for (int i=0; i<4; ++i) {
			if (delaunay.is_infinite(ci, i)) continue;
			const cell_handle_t cj(ci->neighbor(i));
			const cell_size_t cjID(cj->info());
			if (ciID < cjID) continue;
			const void* nodeHandleJ = graph.NodeHandle(cjID);
			const bool ciType(graph.IsNodeOnSrcSide(nodeHandleI));
			if (ciType == graph.IsNodeOnSrcSide(nodeHandleJ)) continue;
			Mesh::Face& face = mesh.faces.AddEmpty();
			const triangle_vhandles_t tri(getTriangle(ci, i));
			for (int v=0; v<3; ++v) {
				const vertex_handle_t vh(tri.verts[v]);
				ASSERT(vh->point() == delaunay.triangle(ci,i)[v]);
				const auto pairItID(mapVertices.emplace(vh.for_compact_container(), (Mesh::VIndex)mesh.vertices.GetSize()));
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
#ifdef CUT_TIMINGS
	}
#endif

	DEBUG_EXTRA("Delaunay tetrahedras graph-cut completed (%g flow): %u vertices, %u faces (%s)", maxFlow, mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
}


// Helper to convert edge index to vertex pairs (standard tetrahedron)
constexpr int edge_vertex_table[6][2] = {
    {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
};

__forceinline int edge_vertex(int edge, int i) {
    return edge_vertex_table[edge][i];
}

using edge_id_t = uint64_t;

// Create a unique edge ID from two vertex handles (order-insensitive)
static inline edge_id_t make_edge_id(vertex_handle_t a, vertex_handle_t b) {
	auto id1 = reinterpret_cast<std::uintptr_t>(&*a);
	auto id2 = reinterpret_cast<std::uintptr_t>(&*b);
	return (id1 < id2)
		? (static_cast<edge_id_t>(id1) << 32) | id2
		: (static_cast<edge_id_t>(id2) << 32) | id1;
}

template<class T>
struct PaddedVector
{
		std::vector<T> mData;
		char mPadding[64-sizeof(mData)];
};

size_t CalculateEstimates(std::unique_ptr<float[]>& distsSq, const std::vector<cell_handle_t>& allFiniteCells, const delaunay_t& Tr) {
	int64_t numFiniteCells = Tr.number_of_cells();

	struct edge_entry_t {
		edge_entry_t(edge_id_t _id, float _d2) : id{ _id }, d2{ _d2 } {}
		edge_id_t id;
		float d2;
	};

	std::vector<PaddedVector<edge_entry_t>> localBuffers;

	int numThreads;

	// Iterate over the finite cells in parallel and calculate the distance squared between each cell's finite edges.
	// There may be duplicates which are resolved later.
#pragma omp parallel
	{
		int id = omp_get_thread_num();

		// Let the first thread resize.
		#pragma omp single
		{
			numThreads = omp_get_num_threads();
			localBuffers.resize(numThreads);
		}

		auto& out = localBuffers[id].mData;
		out.reserve(6*(1 + numFiniteCells / numThreads));

#pragma omp for schedule(static, 1024)
		for (int64_t i = 0; i < numFiniteCells; ++i) {
			const cell_handle_t& c = allFiniteCells[i];

			for (int e = 0; e < 6; ++e) {
				int vi = edge_vertex(e, 0);
				int vj = edge_vertex(e, 1);
				vertex_handle_t v1 = c->vertex(vi);
				vertex_handle_t v2 = c->vertex(vj);

				if (Tr.is_infinite(v1) || Tr.is_infinite(v2))
					continue;

				// Although this prevents duplicates -in- the buffer group, it does not prevent
				// separate buffer groups from having duplicates.
				if (v1 > v2) continue;

				edge_id_t id = make_edge_id(v1, v2);
				auto p1 = CGAL2MVS<float>(v1->point());
				auto p2 = CGAL2MVS<float>(v2->point());
				float d2 = normSq(p1 - p2);

				out.emplace_back(id, d2);
			}
		}
	}

	// Copy these separate buffers to a linear array in parallel.
	// Here we determine the start index in the array for each thread.
	std::vector<size_t> offsets(numThreads + 1, 0);
	for (int i = 0; i < numThreads; ++i)
		offsets[i + 1] = offsets[i] + localBuffers[i].mData.size();

	// total is the true length of this linear array.
	const size_t total = offsets[numThreads];

	// Reserve space for the copy.  Avoid the penalty of initializing each item.
	auto* allEdges = static_cast<edge_entry_t*>(std::malloc(sizeof(edge_entry_t) * total));

	// Copy each thread buffer to the big linear array.
#pragma omp parallel for
	for (int i = 0; i < numThreads; ++i)
		std::memcpy(allEdges + offsets[i], localBuffers[i].mData.data(), localBuffers[i].mData.size() * sizeof(edge_entry_t));

	// Sort the result in parallel (it may have duplicates).
	std::sort(std::execution::par, allEdges, allEdges + total,
		[](const edge_entry_t& a, const edge_entry_t& b) { return a.id < b.id; });

	// Visit the sorted result and copy the unique id distances to the output.
	distsSq.reset(new float[total]);

	float* __restrict dst = distsSq.get();

	if (total > 0) {
		edge_id_t last = allEdges[0].id;
		*dst++ = allEdges[0].d2;

		for (size_t j = 1; j < total; ++j) {
			if (allEdges[j].id != last) {
				last = allEdges[j].id;
				*dst++ = allEdges[j].d2;
			}
		}
	}

	std::free(allEdges);

	return dst - distsSq.get();
}

} // namespace DELAUNAY

static inline float
fasterpow2( float p )
{
	float clipp = ( p < -126 ) ? -126.0f : p;
	union { uint32_t i; float f; } v = { static_cast<uint32_t>( ( 1 << 23 ) * ( clipp + 126.94269504f ) ) };
	return v.f;
}

static inline float JPBEXP( float p )
{
	return fasterpow2( 1.442695040f * p );
}

// Not complete in itself, but used to store space for fixed storage
// that can promote to dynamic storage as needed.
template<class T, int N>
struct HybridArray
{
  HybridArray() :
   mStart(mData),
   mCapacity(N)
  {}

  void resize(int curSize, int newCapacity)
  {
   auto newBlock = std::unique_ptr<T[]>(new T[newCapacity]);
   if (!mDynamicData) {
      ::memcpy(newBlock.get(), mData, sizeof(T)*curSize);
   } else {
      ::memcpy(newBlock.get(), mDynamicData.get(), sizeof(T)*curSize);
   }
   mDynamicData = std::move(newBlock);
   mStart = mDynamicData.get();
   mCapacity = newCapacity;
  }
  
  void reset()
  {
   mStart = mData;
   mCapacity = N;
   mDynamicData.reset();
  }
  
  T* mStart;
  T mData[N];
  std::unique_ptr<T[]> mDynamicData;
  int mCapacity;
};

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
	using namespace DELAUNAY;
	ASSERT(!pointcloud.IsEmpty());
	mesh.Release();

	// create the Delaunay triangulation
	delaunay_t delaunay;
	std::vector<cell_info_t> infoCells;
	std::vector<camera_cell_t> camCells;
	std::vector<facet_t> hullFacets;
	std::vector<delaunay_t::All_cells_iterator> cellIterators;
	
	const size_t numVertices = pointcloud.NumPoints();
	size_t numFiniteFacets;
	size_t numFacets;
	size_t numDelaunayVertices;
	size_t numEstimates;
	std::unique_ptr<float[]> distsSq;

	{
		TD_TIMER_STARTD();


		std::vector<point_t> vertices;
		vertices.reserve(numVertices);
		point_t* __restrict dstVertices = vertices.data();

		std::vector<std::ptrdiff_t> indices;
		indices.reserve(numVertices);
		ptrdiff_t* __restrict dstIndices = indices.data();

		// fetch points
		if (bUseOnlyROI && !IsBounded())
			bUseOnlyROI = false;
		const float* __restrict pPointStream = pointcloud.PointStream();
		for (size_t i = 0, j = 0; i < numVertices; ++i, j += 3) {
			const PointCloud::Point X(pPointStream[j], pPointStream[j+1], pPointStream[j+2]);
			if (bUseOnlyROI && !obb.Intersects(X))
				continue;
			dstVertices[i] = point_t(X.x, X.y, X.z);
			dstIndices[i] = i;
		}

		vertices.resize(numVertices);
		indices.resize(numVertices);

		DEBUG_EXTRA("time %s", TD_TIMER_GET_FMT().c_str());

		// sort vertices
		typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t::Geom_traits, point_t*> Search_traits;
		// JPB The runtime here is not consistent.
		CGAL::spatial_sort<CGAL::Parallel_tag>(indices.begin(), indices.end(), Search_traits(&vertices[0], delaunay.geom_traits()));

		// insert vertices
		Util::Progress progress(_T("Points inserted"), indices.size());
		const float distInsertSq(SQUARE(distInsert));
		vertex_handle_t hint;
		delaunay_t::Locate_type lt;
		int li, lj;

		delaunay.tds().cells().reserve(numVertices); // JPB WIP BUG Enough?
		delaunay.tds().vertices().reserve(numVertices);
		
		// Indices always >= 1
		auto it = indices.cbegin();
		const size_t idx = *it++;
		const point_t& p = vertices[idx];

		// Here we keep track of versioning for testing.
		// Both delaunay.info() and vcg::tri::Info() only compile
		// if we are using custom versions of these libraries.
		// The version returned can be used to track revisions
		// and can be manually adjusted.c
		// delaunay.info() is parallel can be used to make sure
		// we are compiling and using the work with TBB.
		DEBUG("----------------------------------------");
		DEBUG("ReconstructMesh optimization version 1.0");
		const auto [isParallel, CGALversion] = delaunay.info();
		DEBUG("Parallel: %s", isParallel ? "true" : "false");
		DEBUG("CGAL version: = %d", CGALversion);
		const int vcgVersion = vcg::tri::Info();
		DEBUG("VCG version: = %d", vcgVersion);
		DEBUG("----------------------------------------");

		hint = delaunay.insert(p);
		ASSERT(hint != vertex_handle_t());
		hint->info().InsertViews(pointcloud, idx);

		// JPB WIP BUG Expand as needed.
		alignas(64) cell_handle_t cell_stack[8192];
		alignas(64) cell_handle_t incident_cells[16384];

		size_t cnt = 0;
		if (distInsert <= 0) {
			std::for_each(it, indices.cend(), [&](size_t idx) {
				const point_t& p = vertices[idx];
				// insert all points
				hint = delaunay.insert(p, hint);
				ASSERT(hint != vertex_handle_t());
				// update point visibility info
				hint->info().InsertViews(pointcloud, idx);
				++cnt;
				if (!(cnt & 31)) {
					progress += 32;
				}
			});
		} else {
			auto insertPointImpl = [&](size_t idx) {
				const point_t& p = vertices[idx];
				// locate cell containing this point
				// Other variants are not nearly as good.
				const cell_handle_t c(delaunay.locate(p, lt, li, lj, hint->cell()));
				if (lt == delaunay_t::VERTEX) {
					// duplicate point, nothing to insert,
					// just update its visibility info
					hint = c->vertex(li);
					ASSERT(hint != delaunay.infinite_vertex());
				} else {
					// locate the nearest vertex
					vertex_handle_t nearest;
					if (delaunay.dimension() < 3) {
						// use a brute-force algorithm if dimension < 3
						delaunay_t::Finite_vertices_iterator vit = delaunay.finite_vertices_begin();
						nearest = vit;
						++vit;
						adjacent_vertex_back_inserter_t inserter(delaunay, p, nearest);
						for (delaunay_t::Finite_vertices_iterator end = delaunay.finite_vertices_end(); vit != end; ++vit)
							inserter = vit;
					} else {
#ifdef FASTER_ADJ_VERTICES
						// - start with the closest vertex from the located cell
						// - repeatedly take the nearest of its incident vertices if any
						// - if not, we're done
						// This is essentially a rewrite of adjacent_vertices for performance given we
						// know a little about what we are working with.
						nearest = delaunay.nearest_vertex_in_cell(p, c);

						static uint16_t marker = 0;

						const _DataD ax = _SetD(p.x());
						const _DataD ay = _SetD(p.y());
						const _DataD az = _SetD(p.z());
						
						double best_sq = fast_sqdist(ax, ay, az, nearest->point());

						bool changed;
						int restart_count = 0;
						constexpr int restart_limit = 5;

						do {
							++marker;
							if (marker == 0) marker = 1;

							changed = false;

							cell_handle_t* __restrict pCells = cell_stack;

							// Reseed from current nearest
							cell_handle_t seed = nearest->cell();
							*pCells++ = seed;
							seed->tds_data().marker = marker;
							nearest->visited_for_vertex_extractor = marker;

							// Evaluate the seed cell's vertices
							for (int vi = 0; vi < 4; ++vi) {
								vertex_handle_t w = seed->vertex(vi);
								if (w == nearest || delaunay.is_infinite(w)) continue;
								if (w->visited_for_vertex_extractor == marker) continue;

								w->visited_for_vertex_extractor = marker;
								double distSq = fast_sqdist(ax, ay, az, w->point());
								if (distSq < best_sq) {
									nearest = w;
									best_sq = distSq;
									changed = true;
									break;  // restart immediately
								}
							}
							if (changed) continue;

							while (pCells != cell_stack) {
								auto c = *--pCells;

								for (int ni = 0; ni < 4; ++ni) {
									if (c->vertex(ni) == nearest) continue;

									cell_handle_t next = c->neighbor(ni);
									if (next->tds_data().marker == marker) continue;

									next->tds_data().marker = marker;
									*pCells++ = next;

									for (int vi = 0; vi < 4; ++vi) {
										vertex_handle_t w = next->vertex(vi);
										if (w == nearest || delaunay.is_infinite(w)) continue;
										if (w->visited_for_vertex_extractor == marker) continue;

										w->visited_for_vertex_extractor = marker;
										const double distSq = fast_sqdist(ax, ay, az, w->point());
										if (distSq < best_sq) {
											nearest = w;
											best_sq = distSq;
											changed = true;
											break;  // restart cleanly
										}
									}
									if (changed) break;  // stop neighbor loop
								}
								if (changed) break;  // stop traversal
							}

							++restart_count;
							if (restart_count >= restart_limit) break;

						} while (changed);
#else
						ASSERT(c != cell_handle_t());
						nearest = delaunay.nearest_vertex_in_cell(p, c);
						while (true) {
							const vertex_handle_t v(nearest);
							delaunay.adjacent_vertices<true>(nearest, adjacent_vertex_back_inserter_t(delaunay, p, nearest));
							if (v == nearest)
								break;
						}
#endif // FASTER_ADJ_VERTICES
					}
					ASSERT(nearest == delaunay.nearest_vertex(p, hint->cell()));
					hint = nearest;
					// check if point is far enough to all existing points
					const Point3 point(pPointStream[idx*3], pPointStream[idx*3+1], pPointStream[idx*3+2]);
					const PointCloud::View* __restrict views = pointcloud.ViewsStream(idx);
					const size_t numViews = pointcloud.ViewsStreamSize(idx);
					ASSERT(numViews);
					const Point3 np(nearest->point().x(), nearest->point().y(), nearest->point().z()); // CGAL2MVS<float>(nearest->point());
					for (size_t j = 0; j < numViews; ++j) {
						const Image& imageData = images[views[j]];
						const Point3 pn(imageData.camera.ProjectPointP3(point));
						const Point3 pe(imageData.camera.ProjectPointP3(np));
						// ABS(d0-d1)/d0 < 0.01
						// ABS(pn.z-pe.z)/d0 < 0.01
						// ABS(pn.z-pe.z) < 0.01*d0
						if (!IsDepthSimilar(pn.z, pe.z) || normSq(Point2f(pn)-Point2f(pe)) > distInsertSq) {

						//if (!(ABS(pn.z-pe.z) < pn.z*0.01f) || normSq(Point2f(pn)-Point2f(pe)) > distInsertSq) {
							// point far enough to an existing point,
							// insert as a new point
							hint = delaunay.insert(p, lt, c, li, lj);
							ASSERT(hint != vertex_handle_t());
							break;
						}
					}
				}
			};
			std::for_each(it, indices.cend(), [&](size_t idx) {
				insertPointImpl(idx);
				// update point visibility info
				hint->info().InsertViews(pointcloud, idx);
				++cnt;
				if (!(cnt & 15)) {
					progress += 16;
				}
			});
		}

		progress.close();

		numFiniteFacets = 0;
		numFacets = 0;
		for (auto fi=delaunay.facets_begin(), ffi=delaunay.facets_end(); fi!=ffi; ++fi) {
			if (!delaunay.is_infinite(*fi) ) {
				++numFiniteFacets;
			}
			++numFacets;
		}

		// Create stores for the intermediate results.
		numDelaunayVertices = delaunay.number_of_vertices(); // Number of finite vertices, has one more.

#ifdef FASTER_ESTIMATES
		// loop over all cells and store the finite facet of the infinite cells
		const size_t numNodes(delaunay.number_of_cells());
		cellIterators.reserve(numNodes);
		
		cell_size_t ciID(0);

		size_t infiniteCells = 0;
		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), eci=delaunay.all_cells_end(); ci!=eci; ++ci, ++ciID) {
			cellIterators.push_back(ci);
	
			ci->info() = ciID;
			// skip the finite cells
			if (!delaunay.is_infinite(ci))
				continue;
			++infiniteCells;
			// find the finite face
			for (int f=0; f<4; ++f) {
				const facet_t facet(ci, f);
				if (!delaunay.is_infinite(facet)) {
					// store face
					hullFacets.push_back(facet);
					break;
				}
			}
		}

		// Remember, even if we give EstimatesWorker a signature with a reference (say delaunay_t&) it will
		// copy construct it.  We have to explicitly make the ref.

		// Passing distsSq by reference is not the right thing to do here, but it allows us
		// to stay compatible with alternative pathway.
		numEstimates = CalculateEstimates(distsSq, cellIterators, std::ref(delaunay));

		infoCells.resize(numNodes);
		memset(&infoCells[0], 0, sizeof(cell_info_t)*numNodes);
#else
		// distsSq eventually hold the result.  It is guaranteed larger than the number of finite edges.
		distsSq.reset(new float[numDelaunayVertices+numFiniteFacets]); // Upper bound.

		// original
		// Prep the thread that works forwards...
		std::thread worker([&](
			float* __restrict dst,
			delaunay_t::Finite_edges_iterator it,
			delaunay_t::Finite_edges_iterator ite,
			size_t cnt,
			size_t* numEstimates
		) {
			float* start = dst;
			while (it != ite) {
				auto edgeIt = it++;
				const cell_handle_t& c(edgeIt->first);
				*dst++ = normSq(CGAL2MVS<float>(c->vertex(edgeIt->second)->point()) - CGAL2MVS<float>(c->vertex(edgeIt->third)->point()));
			}
			*numEstimates = dst - start;
		},
		distsSq.get(),
		delaunay.finite_edges_begin(),
		delaunay.finite_edges_end(), // Used when cnt == 0
		0, // all
		&numEstimates
		);

		// Now perform various work while the partial estimates are being calculated... 

		// init cells weights and
		// loop over all cells and store the finite facet of the infinite cells
		const size_t numNodes(delaunay.number_of_cells());
		cellIterators.reserve(numNodes);
		
		cell_size_t ciID(0);

		size_t infiniteCells = 0;
		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), eci=delaunay.all_cells_end(); ci!=eci; ++ci, ++ciID) {
			cellIterators.push_back(ci);
	
			ci->info() = ciID;
			// skip the finite cells
			if (!delaunay.is_infinite(ci))
				continue;
			++infiniteCells;
			// find the finite face
			for (int f=0; f<4; ++f) {
				const facet_t facet(ci, f);
				if (!delaunay.is_infinite(facet)) {
					// store face
					hullFacets.push_back(facet);
					break;
				}
			}
		}

		infoCells.resize(numNodes);
		memset(&infoCells[0], 0, sizeof(cell_info_t)*numNodes);
#endif

		// find all cells containing a camera
		camCells.resize(images.GetSize());
		FOREACH(i, images) {
			const Image& imageData = images[i];
			if (!imageData.IsValid())
				continue;
			const Camera& camera = imageData.camera;
			camera_cell_t& camCell = camCells[i];
			camCell.cell = delaunay.locate(MVS2CGAL(camera.C));
			ASSERT(camCell.cell != cell_handle_t());
			fetchCellFacets<CGAL::POSITIVE>(delaunay, hullFacets, camCell.cell, imageData, camCell.facets);
			// link all cells contained by the camera to the source
			for (const facet_t& f: camCell.facets)
				infoCells[f.first->info()].s = kInf;
		}

		size_t numFiniteCells = cellIterators.size() - infiniteCells;

#ifndef FASTER_ESTIMATES
		// Wait for estimation process to finish.
		worker.join();
#endif

		// And perform the median calculation:
		// Values are positive.  Perform the work in the integer unit and in parallel.
		std::nth_element(std::execution::par, (uint32_t*) distsSq.get(), (uint32_t*) distsSq.get() + ( numEstimates / 2 ), (uint32_t*) distsSq.get() + numEstimates);
		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, %u (+%u) faces (%s)",
			indices.size(), delaunay.number_of_vertices(), numFiniteCells, infiniteCells, numFiniteFacets,  numFacets-numFiniteFacets, TD_TIMER_GET_FMT().c_str());
	}

	// for every camera-point ray intersect it with the tetrahedrons and
	// add alpha_vis(point) to cell's directed edge in the graph
	{
		TD_TIMER_STARTD();
	
		const float sigma(SQRT(distsSq[numEstimates/2])* kSigma);
		DEBUG_EXTRA("Sigma is %f", sigma);

#ifndef FASTER_WEIGHTING
		const float inv2SigmaSq(0.5f/(sigma*sigma));
		distsSq.release();

		std::vector<facet_t> facets;

		// compute the weights for each edge
		{
		TD_TIMER_STARTD();
		Util::Progress progress(_T("Points weighted"), delaunay.number_of_vertices());
		#ifdef DELAUNAY_USE_OPENMP
		delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());
		const int64_t nVerts(delaunay.number_of_vertices()+1);
		#pragma omp parallel for private(facets)
		for (int64_t i=0; i<nVerts; ++i) {
			delaunay_t::Vertex_iterator vi;
			#pragma omp critical
			vi = vertexIter++;
		#else
		for (delaunay_t::Vertex_iterator vi=delaunay.vertices_begin(), vie=delaunay.vertices_end(); vi!=vie; ++vi) {
		#endif
			vert_info_t& vert(vi->info());
			if (vert.views.IsEmpty())
				continue;
			const point_t& p(vi->point());
			const Point3 pt(CGAL2MVS<REAL>(p));
			FOREACH(v, vert.views) {
				const typename vert_info_t::view_t view(vert.views[v]);
				const uint32_t imageID(view.idxView);
				const edge_cap_t alpha_vis(view.weight);
				const Image& imageData = images[imageID];
				ASSERT(imageData.IsValid());
				const Camera& camera = imageData.camera;
				const camera_cell_t& camCell = camCells[imageID];
				// compute the ray used to find point intersection
				const Point3 vecCamPoint(pt-camera.C);
				const REAL invLenCamPoint(REAL(1)/norm(vecCamPoint));
				intersection_t inter(pt, Point3(vecCamPoint*invLenCamPoint));
				// find faces intersected by the camera-point segment
				const segment_t segCamPoint(MVS2CGAL(camera.C), p);
				if (!intersect(delaunay, segCamPoint, camCell.facets, facets, inter))
					continue;
				do {
					// assign score, weighted by the distance from the point to the intersection
					const edge_cap_t w(alpha_vis*(1.f-EXP(-SQUARE((float)inter.dist)*inv2SigmaSq)));
					edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
					#ifdef DELAUNAY_USE_OPENMP
					#pragma omp atomic
					#endif
					f += w;
				} while (intersect(delaunay, segCamPoint, facets, facets, inter));
				ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
				// find faces intersected by the endpoint-point segment
				inter.dist = FLT_MAX; inter.bigger = false;
				const Point3 endPoint(pt+vecCamPoint*(invLenCamPoint*sigma));
				const segment_t segEndPoint(MVS2CGAL(endPoint), p);
				const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
				ASSERT(endCell != cell_handle_t());
				fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
				edge_cap_t& t(infoCells[endCell->info()].t);
				#ifdef DELAUNAY_USE_OPENMP
				#pragma omp atomic
				#endif
				t += alpha_vis;
				while (intersect(delaunay, segEndPoint, facets, facets, inter)) {
					// assign score, weighted by the distance from the point to the intersection
					const facet_t& mf(delaunay.mirror_facet(inter.facet));
					const edge_cap_t w(alpha_vis*(1.f-EXP(-SQUARE((float)inter.dist)*inv2SigmaSq)));
					edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
					#ifdef DELAUNAY_USE_OPENMP
					#pragma omp atomic
					#endif
					f += w;
				}
				ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
			}
			++progress;
		}
		progress.close();
		DEBUG_ULTIMATE("\tweighting completed in %s", TD_TIMER_GET_FMT().c_str());
		}
		camCells.clear();

#else
		// distsSq may consume a lot of memory.  Delete it now.
		distsSq.release();
		const float inv2SigmaSq(-0.5f/(sigma*sigma));

		// compute the weights for each edge
		Util::Progress progress(_T("Points weighted"), numDelaunayVertices);
		// Each thread stores calculated pts/vis/dist information in an array
		// until it reaches about 70% of its size.  Then it switches to 
		// dynamic allocation.
		// Note, there is no good way to manage this for performance so HybridArray
		// just holds the data and we make the changes as necessary.
		struct ThreadData
		{
			std::vector<facet_t> mData;

			HybridArray<edge_cap_t*, 4096> mPts;
			HybridArray<edge_cap_t, 4096> mVis;
			HybridArray<edge_cap_t, 4096> mDist;
		};

		std::vector<ThreadData> perThreadData;

		delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());
		const int64_t nVerts(delaunay.number_of_vertices());
#pragma omp parallel
		{
			// First one sets up everything.
			#pragma omp single
			{
				int numThreads = omp_get_num_threads();
				perThreadData.resize(numThreads);
			}

			const int id = omp_get_thread_num();
			ThreadData& td = perThreadData[id];

			std::vector<facet_t>& facets = td.mData;
			td.mData.reserve(128); // JPB WIP OPT more?
			auto& pts = td.mPts;
			auto& vis = td.mVis;
			auto& dist = td.mDist;

#pragma omp for schedule(static, 1024)
			for (int64_t i=0; i<nVerts; ++i) {
				delaunay_t::Vertex_iterator vi;
#pragma omp critical
				vi = vertexIter++;
				vert_info_t& vert(vi->info());
				if (vert.views.IsEmpty())
					continue;
				const point_t& p(vi->point());
				const Point3 pt(CGAL2MVS<REAL>(p));

				size_t cnt = 0;
				pts.reset();
				vis.reset();
				dist.reset();

				edge_cap_t** __restrict pPts = pts.mData;
				edge_cap_t* __restrict pVis = vis.mData;
				edge_cap_t* __restrict pDist = dist.mData;

				FOREACH(v, vert.views)
				{
					const typename vert_info_t::view_t view(vert.views[v]);
					const uint32_t imageID(view.idxView);
					const edge_cap_t alpha_vis(view.weight);
					const Image& imageData = images[imageID];
					ASSERT(imageData.IsValid());
					const Camera& camera = imageData.camera;
					const camera_cell_t& camCell = camCells[imageID];
					// compute the ray used to find point intersection
					const Point3 vecCamPoint(pt-camera.C);
					const REAL invLenCamPoint(REAL(1)/norm(vecCamPoint));
					intersection_t inter(pt, Point3(vecCamPoint*invLenCamPoint));
					// find faces intersected by the camera-point segment
					const segment_t segCamPoint(MVS2CGAL(camera.C), p);

					const point_t& source = segCamPoint.source();
					const point_t& target = segCamPoint.target();
					double segDiff[] = { target.x() - source.x(), target.y() - source.y(), target.z() - source.z() };
					double segDiffN[] = { -segDiff[0], -segDiff[1], -segDiff[2] };

					if (!intersect(delaunay, segCamPoint, segDiff, segDiffN, camCell.facets, facets, inter))
						continue;
					do {
						// assign score, weighted by the distance from the point to the intersection
						edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
						pPts[cnt] = &f;
						pVis[cnt] = alpha_vis;
						pDist[cnt] = (edge_cap_t)inter.dist;
						++cnt;
					}
					while (intersect(delaunay, segCamPoint, segDiff, segDiffN, facets, facets, inter));
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
					// find faces intersected by the endpoint-point segment
					inter.dist = FLT_MAX; inter.bigger = false;
					const Point3 endPoint(pt+vecCamPoint*(invLenCamPoint*sigma));
					const segment_t segEndPoint(MVS2CGAL(endPoint), p);
					const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
					ASSERT(endCell != cell_handle_t());
					fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
					edge_cap_t& t(infoCells[endCell->info()].t);
#ifdef DELAUNAY_USE_OPENMP
#pragma omp atomic
#endif
					t += alpha_vis;

					const point_t& source2 = segEndPoint.source();
					const point_t& target2 = segEndPoint.target();
					double segDiff2[] = { target2.x() - source2.x(), target2.y() - source2.y(), target2.z() - source2.z() };
					double segDiff2N[] = { -segDiff2[0], -segDiff2[1], -segDiff2[2] };

					while (intersect(delaunay, segEndPoint, segDiff2, segDiff2N, facets, facets, inter)) {
						// assign score, weighted by the distance from the point to the intersection
						const facet_t& mf(delaunay.mirror_facet(inter.facet));
						edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
						// The hybrid array makes these very fast stores.
						pPts[cnt] = &f;
						pVis[cnt] = alpha_vis;
						pDist[cnt] = (edge_cap_t)inter.dist;
						++cnt;
					}
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);

					// When we complete the ray, see if we need to switch to or
					// expand dynamic storage.
					if (cnt >= (pts.mCapacity * 7)/10) {
						pts.resize((int)cnt, pts.mCapacity*2);
						vis.resize((int)cnt, pts.mCapacity*2);
						dist.resize((int)cnt, pts.mCapacity*2);

						pPts = pts.mDynamicData.get();
						pVis = vis.mDynamicData.get();
						pDist = dist.mDynamicData.get();
					}
				}

				// Here we apply the deferred intersection results to the edges all at once.
				// We are essentially trading large numbers of atomic accesses for
				// a single critical section.
				// This allows us to unfold the vectorize most of the instructions as well.
				const _Data vInv2SigmaSq = _Set(inv2SigmaSq);

				const size_t numEights = cnt/8;
				const size_t numRemaining = cnt & 7;
				float** __restrict pp = pPts;
				float* __restrict v = pVis;
				float* __restrict d = pDist;

				const _Data vOne = { 1.f, 1.f, 1.f, 1.f };

#pragma omp critical
			{
				for (size_t i = 0; i < numEights; ++i, d += 8, v += 8, pp += 8) {
					// Manually unfolded to reduce load/store delays.
					float* __restrict p0 = pp[0];
					float* __restrict p1 = pp[1];
					float* __restrict p2 = pp[2];
					float* __restrict p3 = pp[3];
					float* __restrict p4 = pp[4];
					float* __restrict p5 = pp[5];
					float* __restrict p6 = pp[6];
					float* __restrict p7 = pp[7];

					float p0Before = *p0;
					float p1Before = *p1;
					float p2Before = *p2;
					float p3Before = *p3;
					float p4Before = *p4;
					float p5Before = *p5;
					float p6Before = *p6;
					float p7Before = *p7;

					const _Data vDists = _Load(d);
					const _Data vDists2 = _Load(d+4);
					const _Data vAlphaVis = _Load(v);
					const _Data vAlphaVis2 = _Load(v+4);
					const _Data vDistsSq = _Mul(vDists, vDists);
					const _Data vDistsSq2 = _Mul(vDists2, vDists2);
					const _Data vNegDistsSqFactor = _Mul(vDistsSq, vInv2SigmaSq);
					const _Data vNegDistsSqFactor2 = _Mul(vDistsSq2, vInv2SigmaSq);
					_Data vExp;
					_Data vExp2;
					FastExpAlwaysNegativePair(vExp, vExp2, vNegDistsSqFactor, vNegDistsSqFactor2);
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

					p0Before += v0;
					p1Before += v1;
					p2Before += v2;
					p3Before += v3;
					p4Before += v4;
					p5Before += v5;
					p6Before += v6;
					p7Before += v7;

					*p0 = p0Before;
					*p1 = p1Before;
					*p2 = p2Before;
					*p3 = p3Before;
					*p4 = p4Before;
					*p5 = p5Before;
					*p6 = p6Before;
					*p7 = p7Before;
				}

				for (size_t i = 0; i < numRemaining; ++i, ++d, ++v, ++pp) {
					**pp += (*v*(1.f- JPBEXP(SQUARE((float)*d)*inv2SigmaSq)));
				}
			} // Critical

				if (!(i&31)) { progress += 32; }
			}
		} // parallel

		progress.close();

		camCells.clear();

#if 0 // WIP
		#ifdef DELAUNAY_WEAKSURF
		// enforce t-edges for each point-camera pair with free-space support weights
		if (bUseFreeSpaceSupport) {
		TD_TIMER_STARTD();
		#ifdef DELAUNAY_USE_OPENMP
		delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());
		const int64_t nVerts(delaunay.number_of_vertices()+1);
		#pragma omp parallel for private(facets)
		for (int64_t i=0; i<nVerts; ++i) {
			delaunay_t::Vertex_iterator vi;
			#pragma omp critical
			vi = vertexIter++;
		#else
		for (delaunay_t::Vertex_iterator vi=delaunay.vertices_begin(), vie=delaunay.vertices_end(); vi!=vie; ++vi) {
		#endif
			const vert_info_t& vert(vi->info());
			if (vert.views.empty())
				continue;
			const point_t& p(vi->point());
			const Point3f pt(CGAL2MVS<float>(p));
			for (auto& v : vert.views) {
				const uint32_t imageID(vert.views[v.idxView]);
				const Image& imageData = images[imageID];
				ASSERT(imageData.IsValid());
				const Camera& camera = imageData.camera;
				// compute the ray used to find point intersection
				const Point3f vecCamPoint(pt-Cast<float>(camera.C));
				const float invLenCamPoint(1.f/norm(vecCamPoint));
				// find faces intersected by the point-camera segment and keep the max free-space support score
				const Point3f bgnPoint(pt-vecCamPoint*(invLenCamPoint*sigma*kf));
				const segment_t segPointBgn(p, MVS2CGAL(bgnPoint));
				intersection_t inter;
				if (!intersectFace(delaunay, segPointBgn, vi, vert.viewsInfo[v].cell2Cam, facets, inter))
					continue;
				edge_cap_t beta(0);
				do {
					const edge_cap_t fs(freeSpaceSupport(delaunay, infoCells, inter.facet.first));
					if (beta < fs)
						beta = fs;
				} while (intersectFace(delaunay, segPointBgn, facets, facets, inter));
				// find faces intersected by the point-endpoint segment
				const Point3f endPoint(pt+vecCamPoint*(invLenCamPoint*sigma*kb));
				const segment_t segPointEnd(p, MVS2CGAL(endPoint));
				if (!intersectFace(delaunay, segPointEnd, vi, vert.viewsInfo[v].cell2End, facets, inter))
					continue;
				edge_cap_t gammaMin(FLT_MAX), gammaMax(0);
				do {
					const edge_cap_t fs(freeSpaceSupport(delaunay, infoCells, inter.facet.first));
					if (gammaMin > fs)
						gammaMin = fs;
					if (gammaMax < fs)
						gammaMax = fs;
				} while (intersectFace(delaunay, segPointEnd, facets, facets, inter));
				const edge_cap_t gamma((gammaMin+gammaMax)*0.5f);
				// if the point can be considered an interface point,
				// enforce the t-edge weight of the end cell
				const edge_cap_t epsAbs(beta-gamma);
				const edge_cap_t epsRel(gamma/beta);
				if (epsRel < kRel && epsAbs > kAbs && gamma < kOutl) {
					edge_cap_t& t(infoCells[inter.ncell->info()].t);
					#ifdef DELAUNAY_USE_OPENMP
					#pragma omp atomic
					#endif
					t *= epsAbs;
				}
			}
		}
		DEBUG_ULTIMATE("\tt-edge reinforcement completed in %s", TD_TIMER_GET_FMT().c_str());
		}
		#endif
#endif

#endif // ORIGINAL_WEIGHTING

		DEBUG_EXTRA("Delaunay tetrahedras weighting completed: %u cells, %u faces (%s)", delaunay.number_of_cells(), numFacets, TD_TIMER_GET_FMT().c_str());
	}

	// run graph-cut and extract the mesh
	graphcut(cellIterators, delaunay, infoCells, mesh, kQual);

	// fix non-manifold vertices and edges
	mesh.FixNonManifold();
	return true;
}
/*----------------------------------------------------------------*/
