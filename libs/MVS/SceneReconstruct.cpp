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

#define MANIFOLD_FIXUP
#undef PRE_OPENMVS21
#define APPROXIMATE_GRAPHCUT

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
typedef boost::container::small_vector<uint32_t, 26> view_vec_t;
std::vector<view_vec_t> allViews; // faces' weight from the cell outwards

void InsertViews(size_t vertexId, const PointCloudStreaming& pc, PointCloud::Index idxPoint)
{
	// Note we silently enforce indices no larger than a uint32_t
	allViews[vertexId].push_back(idxPoint);
}

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

// Given a facet, compute the plane containing it
__forceinline CGAL::Plane_3<kernel_t> getFacetPlane(const facet_t& facet)
{
	const point_t& v0(facet.first->vertex((facet.second+1)%4)->point());
	const point_t& v1(facet.first->vertex((facet.second+2)%4)->point());
	const point_t& v2(facet.first->vertex((facet.second+3)%4)->point());

	return CGAL::Plane_3<kernel_t>(v0, v1, v2);
}

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
	REAL dist; // distance from starting point (camera) to this facet
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
#if 1 // try agin
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

	// signed distances to plane (no 'd' needed): dp = n·(p-a), dq = n·(q-a)
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
	double& voOut, double& vdOut) {
	const double dx = ray.m_vDir.x(), dy = ray.m_vDir.y(), dz = ray.m_vDir.z();
	const double ox = ray.m_pOrig.x(), oy = ray.m_pOrig.y(), oz = ray.m_pOrig.z();

	const double vd = nx * dx + ny * dy + nz * dz;
	constexpr double eps = 1e-12;
	if (FastAbsS(vd) < eps) return false;

	const double d = -(nx * v0x + ny * v0y + nz * v0z);
	const double vo = -(nx * ox + ny * oy + nz * oz + d);

	const double delta = vo - prevDist * vd;
	const bool isGreater = (vd >= 0.0) ? (delta > 0.0) : (delta < 0.0);
	if (isGreater != bigger) return false;

	voOut = vo; vdOut = vd;
	return true;
}


// Shared core for edge checks.
// Select endpoint set with template bool UseP: true => P-side, false => Q-side.
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
		return CheckEdges2FastCore<false>(sQx, sQy, sQz, qaz, qay, qaz, bxQ, byQ, bzQ, cxQ, cyQ, czQ, coplanar);
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
__forceinline void computePlaneSphereAngle4(const delaunay_t& Tr,
	const cell_handle_t& cell,
	float out[4]) {
	if (Tr.is_infinite(cell)) {
		out[0] = out[1] = out[2] = out[3] = 1.0f;
		return;
	}

	// Load the four cell vertices once (SoA source)
	const auto& p0 = cell->vertex(0)->point();
	const auto& p1 = cell->vertex(1)->point();
	const auto& p2 = cell->vertex(2)->point();
	const auto& p3 = cell->vertex(3)->point();

	const float px[4] = { (float)p0.x(), (float)p1.x(), (float)p2.x(), (float)p3.x() };
	const float py[4] = { (float)p0.y(), (float)p1.y(), (float)p2.y(), (float)p3.y() };
	const float pz[4] = { (float)p0.z(), (float)p1.z(), (float)p2.z(), (float)p3.z() };

#if CGAL_VERSION_NR < 1041101000
	const auto cc = cell->circumcenter(Tr.geom_traits());
#else
	const auto cc = Tr.geom_traits().construct_circumcenter_3_object()(
		cell->vertex(0)->point(),
		cell->vertex(1)->point(),
		cell->vertex(2)->point(),
		cell->vertex(3)->point());
#endif
	const float ccx = (float)cc.x();
	const float ccy = (float)cc.y();
	const float ccz = (float)cc.z();

#if defined(__SSE2__)
	// Gather Pa/Pb/Pc from your exact facet triangles (matches scalar order).
	// facet k: tri = getTriangle(cell, k); Pa=tri[0], Pb=tri[1], Pc=tri[2]

  // Per-facet lane index mapping that matches getTriangle(cell,k)
	int i0[3], i1[3], i2[3], i3[3];
	facetOrderIndices(cell, 0, i0);
	facetOrderIndices(cell, 1, i1);
	facetOrderIndices(cell, 2, i2);
	facetOrderIndices(cell, 3, i3);

	__m128 ax = _mm_setr_ps(px[i0[0]], px[i1[0]], px[i2[0]], px[i3[0]]);
	__m128 ay = _mm_setr_ps(py[i0[0]], py[i1[0]], py[i2[0]], py[i3[0]]);
	__m128 az = _mm_setr_ps(pz[i0[0]], pz[i1[0]], pz[i2[0]], pz[i3[0]]);

	__m128 bx = _mm_setr_ps(px[i0[1]], px[i1[1]], px[i2[1]], px[i3[1]]);
	__m128 by = _mm_setr_ps(py[i0[1]], py[i1[1]], py[i2[1]], py[i3[1]]);
	__m128 bz = _mm_setr_ps(pz[i0[1]], pz[i1[1]], pz[i2[1]], pz[i3[1]]);

	__m128 cx = _mm_setr_ps(px[i0[2]], px[i1[2]], px[i2[2]], px[i3[2]]);
	__m128 cy = _mm_setr_ps(py[i0[2]], py[i1[2]], py[i2[2]], py[i3[2]]);
	__m128 cz = _mm_setr_ps(pz[i0[2]], pz[i1[2]], pz[i2[2]], pz[i3[2]]);

	// A = Pb - Pa; B = Pc - Pa
	__m128 Ax = _mm_sub_ps(bx, ax);
	__m128 Ay = _mm_sub_ps(by, ay);
	__m128 Az = _mm_sub_ps(bz, az);

	__m128 Bx = _mm_sub_ps(cx, ax);
	__m128 By = _mm_sub_ps(cy, ay);
	__m128 Bz = _mm_sub_ps(cz, az);

	// N = A x B
	__m128 Nx = _mm_sub_ps(_mm_mul_ps(Ay, Bz), _mm_mul_ps(Az, By));
	__m128 Ny = _mm_sub_ps(_mm_mul_ps(Az, Bx), _mm_mul_ps(Ax, Bz));
	__m128 Nz = _mm_sub_ps(_mm_mul_ps(Ax, By), _mm_mul_ps(Ay, Bx));

	// |N|^2
	__m128 Nx2 = _mm_mul_ps(Nx, Nx);
	__m128 Ny2 = _mm_mul_ps(Ny, Ny);
	__m128 Nz2 = _mm_mul_ps(Nz, Nz);
	__m128 n12 = _mm_add_ps(Nx2, Ny2);
	__m128 fnLenSq = _mm_add_ps(n12, Nz2);

	// C = CC - Pa
	__m128 CCx = _mm_set1_ps(ccx);
	__m128 CCy = _mm_set1_ps(ccy);
	__m128 CCz = _mm_set1_ps(ccz);
	__m128 Cx = _mm_sub_ps(CCx, ax);
	__m128 Cy = _mm_sub_ps(CCy, ay);
	__m128 Cz = _mm_sub_ps(CCz, az);

	// |C|^2
	__m128 Cx2 = _mm_mul_ps(Cx, Cx);
	__m128 Cy2 = _mm_mul_ps(Cy, Cy);
	__m128 Cz2 = _mm_mul_ps(Cz, Cz);
	__m128 c12 = _mm_add_ps(Cx2, Cy2);
	__m128 ctLenSq = _mm_add_ps(c12, Cz2);

	// dot(N, C)
	__m128 d0 = _mm_mul_ps(Nx, Cx);
	__m128 d1 = _mm_mul_ps(Ny, Cy);
	__m128 d2 = _mm_mul_ps(Nz, Cz);
	__m128 d01 = _mm_add_ps(d0, d1);
	__m128 dot = _mm_add_ps(d01, d2);

	// denom and real sqrt
	__m128 denom = _mm_mul_ps(fnLenSq, ctLenSq);
	__m128 sqrtDen = _mm_sqrt_ps(denom);
	__m128 res = _mm_div_ps(dot, sqrtDen);

	// clamp [-1,1]
	__m128 one = _mm_set1_ps(1.0f);
	__m128 negOne = _mm_set1_ps(-1.0f);
	res = _mm_min_ps(_mm_max_ps(res, negOne), one);

	// degenerates -> 0.5f
	__m128 zero = _mm_set1_ps(0.0f);
	__m128 halfVal = _mm_set1_ps(0.5f);
	__m128 mBad = _mm_or_ps(_mm_or_ps(_mm_cmple_ps(fnLenSq, zero),
		_mm_cmple_ps(ctLenSq, zero)),
		_mm_cmple_ps(denom, zero));
	__m128 outv = _mm_or_ps(_mm_and_ps(mBad, halfVal),
		_mm_andnot_ps(mBad, res));

	_mm_storeu_ps(out, outv);

#else
	// Scalar fallback identical to your math (kept for portability)
	for (int k = 0; k < 4; ++k) {
		const auto tri = getTriangle(cell, k);
		const auto& pa = tri.verts[0]->point();
		const auto& pb = tri.verts[1]->point();
		const auto& pc = tri.verts[2]->point();
		const float x0 = (float)pa.x(), y0 = (float)pa.y(), z0 = (float)pa.z();
		const float x1 = (float)pb.x(), y1 = (float)pb.y(), z1 = (float)pb.z();
		const float x2 = (float)pc.x(), y2 = (float)pc.y(), z2 = (float)pc.z();
		const float ax = x1 - x0, ay = y1 - y0, az = z1 - z0;
		const float bx = x2 - x0, by = y2 - y0, bz = z2 - z0;
		const float nx = ay * bz - az * by;
		const float ny = az * bx - ax * bz;
		const float nz = ax * by - ay * bx;
		const float fn = nx * nx + ny * ny + nz * nz;
		if (fn == 0.0f) { out[k] = 0.5f; continue; }
		const float cx = ccx - x0, cy = ccy - y0, cz = ccz - z0;
		const float ct = cx * cx + cy * cy + cz * cz;
		const float d = fn * ct;
		if (d <= 0.0f) { out[k] = 0.5f; continue; }
		float r = (nx * cx + ny * cy + nz * cz) / std::sqrt(d);
		out[k] = r < -1.0f ? -1.0f : (r > 1.0f ? 1.0f : r);
	}
#endif
}




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

  const float cx = cc_pt.x() - x0;
  const float cy = cc_pt.y() - y0;
  const float cz = cc_pt.z() - z0;
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

} // namespace DELAUNAY

#pragma intrinsic(_InterlockedCompareExchange)
__forceinline float AtomicAddFloat(float* __restrict addr, float val)
{
  static_assert(sizeof(float) == sizeof(LONG), "Expected float and LONG to be same size");

	volatile LONG* __restrict intAddr = reinterpret_cast<volatile LONG*>(addr);
	union {
		float f;
		LONG i;
	} oldVal, newVal;

	do {
		oldVal.i = *intAddr;
		newVal.f = oldVal.f + val;
		newVal.i = *reinterpret_cast<LONG*>(&newVal.f); // or just reuse newVal.i = *(LONG*)&newVal.f;
	} while (_InterlockedCompareExchange(intAddr, newVal.i, oldVal.i) != oldVal.i);

	return newVal.f;
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
		const int64_t cnt = (int64_t)numVertices;
		// NOTE: More than twice as fast natively parallelized.
#pragma omp parallel for
		for (int64_t i = 0; i < cnt; ++i) {
			const int64_t j = i * 3;
			const float x = pPointStream[j];
			const float y = pPointStream[j + 1];
			const float z = pPointStream[j + 2];

			origVertices[i] = DELAUNAY::point_t(x, y, z);
#ifndef VALIDATE
			indices[i] = i;
#endif
		}

		return numVertices;
	}
}

float Quantize(float cap, float maxCap)
{
#if 1
#if 1
	// Branchless clamp to [0, maxCap]
	float clamped = FastClampS(cap, 0.0f, maxCap);

	// Exact "round half up" for non-negative values:
	// MSVC lowers this to add + cvttss2si (truncate) -> very fast.
	int scaled = static_cast<int>(clamped * 4.0f + 0.5f);
	return 0.25f * static_cast<float>(scaled);
#else
	if (!std::isfinite(cap)) return maxCap;
	if (cap <= 0.0f)
	{
		return 0.f;
	}
	else if (cap >= maxCap)
	{
		return maxCap;
	}

	int scaled = static_cast<int>(cap * 2.0f + 0.5f);
	return 0.5f * scaled;
#endif
#else
#ifdef APPROXIMATE_GRAPHCUT
#if 1
  int scaled = static_cast<int>(cap * 2.0f + 0.5f);
  return 0.5f * scaled;
#else
  // Step is 0.2, so multiply by 5 and round to nearest int
  int scaled = static_cast<int>(cap * 5.0f + 0.5f);
  return 0.2f * scaled;
#endif
#else
  if (!std::isfinite(cap)) return maxCap;
  return cap <= 0.0f ? 0.0f : (cap >= maxCap ? maxCap : cap);
#endif
#endif
}

#include <algorithm>
#include <cstdint>
#include <vector>
#include <limits>
#include <cmath>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

template<class T0, class T1, class T2>
void permuteScatter2(
	const std::vector<ptrdiff_t>& orderNewToOld,
	const std::vector<T0>& src0, T0* __restrict dst0,
	const std::vector<T1>& src1, std::vector<T1>& dst1,
	const std::vector<T2>& src2, std::vector<T2>& dst2
)
{
	const size_t n = orderNewToOld.size();
	dst1.resize(n);
	dst2.resize(n);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 1 << 16), [&](auto const& r) {
		for (size_t i = r.begin(); i != r.end(); ++i) {
			const size_t oldId = static_cast<size_t>(orderNewToOld[i]); // indices[new]=old
			dst0[i] = src0[oldId];
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

typedef CGAL::Simple_cartesian<double>              Kernel;
typedef Kernel::Point_3                             CGALPoint;
typedef CGAL::Search_traits_3<Kernel>               Traits;
typedef CGAL::Kd_tree<Traits>                       Tree;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>  KSearch;
typedef KSearch::Tree                              KdTree;

static void knnMeanDist(const DELAUNAY::point_t* pts, size_t n, int k,
	std::vector<double>& out)
{
	out.resize(n);
	std::vector<CGALPoint> cloud;
	cloud.reserve(n);
	for (size_t i = 0; i < n; ++i)
		cloud.emplace_back(pts[i].x(), pts[i].y(), pts[i].z());

	Tree tree(cloud.begin(), cloud.end());

#pragma omp parallel for schedule(static)
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i) {
		KSearch search(tree, cloud[i], k + 1); // include self
		double sum = 0.0;
		int count = 0;
		for (auto it = search.begin(); it != search.end(); ++it) {
			if (count == 0) { count++; continue; } // skip self
			sum += FastSqrtD(it->second);
			count++;
		}
		out[(size_t)i] = (count > 1 ? sum / (count - 1) : 0.0);
	}
}

// Main routine: fills mask[0..n-1] with 1 = keep, 0 = drop
void StatisticalOutlierRemoval(const DELAUNAY::point_t* pts, size_t n,
	unsigned char* mask,
	int k = 16, double stddevMul = 1.5)
{
	TD_TIMER_STARTD();

	int removed = 0;
	if (n != 0) {
		std::vector<double> meanDist;
		knnMeanDist(pts, n, k, meanDist);

		// Compute global mean and stdev
		double mean = 0.0;
		for (double v : meanDist) mean += v;
		mean /= n;

		double var = 0.0;
		for (double v : meanDist) {
			double d = v - mean;
			var += d * d;
		}
		var /= n;
		double stdev = std::sqrt(var);

		double threshold = mean + stddevMul * stdev;

		// Write mask
		for (size_t i = 0; i < n; ++i) {
			mask[i] = (meanDist[i] <= threshold ? 1 : 0);
			if (!mask[i]) {
				++removed;
			}
		}
	}

	DEBUG_EXTRA(
		"%d point outliers removed in %s",
		removed,
		TD_TIMER_GET_FMT().c_str()
	);
}

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

	std::vector<std::ptrdiff_t> indices(pointcloud.GetSize());
	std::vector<uint32_t> offsets(pointcloud.GetSize());
	std::vector<uint32_t> sizes(pointcloud.GetSize());
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

	FOREACH(i, images) {
		Image& imageData = images[i];
		if (!imageData.IsValid())
			continue;
		for (int j = 0; j < imageData.camera.P.elems; ++j) {
			viewCameras[i][j] = (float)imageData.camera.P[j];
		}
	}

	// create the Delaunay triangulation
	const size_t numPointCloudVertices = pointcloud.NumPoints();
	if (numPointCloudVertices >= std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("Unsupported");
	}

	delaunay_t delaunay;

	std::vector<cell_info_t> infoCells;
	std::vector<camera_cell_t> camCells;
	std::vector<facet_t> hullFacets;
	std::vector<delaunay_t::All_cells_iterator> cellIterators;
	std::vector<vertex_handle_t> idToVertex; // indexed by uint32_t id
	
	size_t numVertices;
#ifdef FACET_DIAGNOSTICS
	size_t numFiniteFacets;
	size_t numFacets;
#endif
	size_t numDelaunayVertices;
	std::unique_ptr<float[]> distsSq;
	float approxMedian;
	std::unique_ptr<point_t[]> vertices;
	const float distInsertSq(SQUARE(distInsert));
	delaunay_t::Locate_type lt;
	int li, lj;
	size_t totalCells;
	std::vector<unsigned char> mask;

	{
		TD_TIMER_STARTD();

		std::vector<point_t> origVertices;
		origVertices.resize(numPointCloudVertices);
		{
			TD_TIMER_STARTD();

			// fetch points
			if (bUseOnlyROI && !IsBounded())
				bUseOnlyROI = false;

			numVertices =
				bUseOnlyROI ?
				ProcessPoints<true>(
					pointcloud.PointStream(),
					numPointCloudVertices,
					origVertices.data(),
					indices.data(),
					obb
				) :
				ProcessPoints<false>(
					pointcloud.PointStream(),
					numPointCloudVertices,
					origVertices.data(),
					indices.data(),
					obb
				);

			indices.resize(numVertices);
			origVertices.resize(numVertices);
			offsets.resize(numVertices);
			sizes.resize(numVertices);
#ifndef VALIDATE
			// sort vertices
			typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t::Geom_traits, point_t*> Search_traits;
			// JPB The runtime here is not consistent.
			// Defaults on this work really well.
			CGAL::spatial_sort<CGAL::Parallel_tag>(
				indices.begin(), indices.end(),
				Search_traits(&origVertices[0], delaunay.geom_traits())
			);
#endif

			// origVertices[i] refers to the original data.
			// indices[i] maps the sorted data to the original data.
			// Rewrite the vertex data in index sorted form:
			vertices.reset(new point_t[numVertices]);
			permuteScatter2(
				indices,
				origVertices, vertices.get(),
				pointcloud.pointViewsOffsets, offsets,
				pointcloud.pointViewsSizes, sizes
			);
			decltype(origVertices)().swap(origVertices);
			decltype(indices)().swap(indices);

			// The points of the cloud are now kept in vertices.  Ancillary data, for view information,
			// is also maintained.
			// Go through the cloud and eliminate outliers, but do so in-place (without disturbing
			// the ancillary data.
			mask.resize(numVertices);
			StatisticalOutlierRemoval(vertices.get(), numVertices, mask.data(), 32, 2);

			// insert vertices
			// 6x vertices is a generous worst case, but uses too much memory.
			// Potentally allow some dynamic allocation to better keep
			// memory within reasonable limits.
			delaunay.tds().cells().reserve(numVertices*4); // May reserve dynamically
			delaunay.tds().vertices().reserve(numVertices); // Should be sufficient to prevent reallocations.s
			allViews.resize(numVertices);

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
		DEBUG("ReconstructMesh optimization version 1.1.10");
		const auto [isParallel, CGALversion] = CGAL::info();
		DEBUG("Parallel: %s", isParallel ? "true" : "false");
		DEBUG("CGAL version: = %d", CGALversion);
		constexpr int vcgVersion = vcg::tri::Info();
		DEBUG("VCG version: = %d", vcgVersion);
		DEBUG("------------------------------------------");
#endif
		// Fixed storage is slightly faster, but difficult to maintain.
		constexpr size_t kMaxCells = 16384;
		std::vector<cell_handle_t> cellQueue;
		cellQueue.reserve(kMaxCells);

		vertex_handle_t hint;

		// InsertViews first parameter must be the dt's index --verified by validation code.
		if (distInsert <= 0) {
			for (size_t i = 0; i < numVertices; ++i) {
				const point_t& p = vertices[i]; // These are the sorted vertices.
				if (!mask[i]) {
					continue;
				}
				// insert all points
				hint = delaunay.insert(p, hint);
				ASSERT(anchor != vertex_handle_t());
				// update point visibility info
				InsertViews(hint->info().idx, pointcloud, i);
				if (!(i & 255)) {
					progress += 256;
				}
			}
		} else {
			std::vector<uint32_t> vertexMarks(numVertices);
			uint32_t marker = 0;
			const vertex_handle_t infV = delaunay.infinite_vertex();  // cheap pointer compare

			int i = 0;
			for (bool done = false; !done; ++i) {
				if (mask[i]) {
					hint = delaunay.insert(vertices[i]);
					InsertViews(hint->info().idx, pointcloud, 0);
					done = true;
				}
			}

			for (; i < numVertices; ++i) {
				if (!mask[i]) {
					continue;
				}

				const point_t& p = vertices[i];
				const double px = p.x();
				const double py = p.y();
				const double pz = p.z();

				const uint32_t* __restrict pointViewsOffset = &offsets[i];
				const uint32_t* __restrict pointViewSizes = &sizes[i];

				size_t offset;
				size_t numViews;
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

					offset = *pointViewsOffset;
					numViews = *pointViewSizes;
					views = pointcloud.pointViewsMemory.data() + offset;
				} else {
					offset = *pointViewsOffset;
					numViews = *pointViewSizes;

					// Optimized BFS-style neighbor search
					nearest = delaunay.nearest_vertex_in_cell3(p, c); // Was cell3 JPB WIP BUG

					views = pointcloud.pointViewsMemory.data() + offset;
					_mm_prefetch((const char*)views, _MM_HINT_T0);

					const double qx = p.x(), qy = p.y(), qz = p.z();
					const point_t& nearestPt = nearest->point();
					double bestSq = fast_sqdist2(qx, qy, qz, nearestPt.x(), nearestPt.y(), nearestPt.z());

					vertexMarks[nearest->info().idx] = marker;

					// The key difference from the original code is that the original determines
					// all adjacent cells and then looks at them.
					// Here, we identify the adjacent cells as needed.
					while (true) {
						++marker;
						// 2^32 iteration limit.

						vertex_handle_t best = nearest;

						cellQueue.clear();
						cell_handle_t start = nearest->cell();
						cellQueue.push_back(start);
						start->tds_data().marker = marker;

						size_t queueIndex = 0;
#if 1
						// inside your loop:
						while (queueIndex < cellQueue.size()) {
							const cell_handle_t c = cellQueue[queueIndex++];

							// fetch vertices once; reuse in both phases
							const vertex_handle_t v0 = c->vertex(0);
							const vertex_handle_t v1 = c->vertex(1);
							const vertex_handle_t v2 = c->vertex(2);
							const vertex_handle_t v3 = c->vertex(3);

							{
								// v0
								if (v0 != nearest && v0 != infV) {
									uint32_t* m = &vertexMarks[v0->info().idx];
									if (*m != marker) {
										*m = marker;
										const point_t& pt = v0->point();
										const double dx = pt.x() - qx, dy = pt.y() - qy, dz = pt.z() - qz;
										const double d2 = dx * dx + dy * dy + dz * dz;
										if (d2 < bestSq) { bestSq = d2; best = v0; goto refine_restart; }
									}
								}

								// v1
								if (v1 != nearest && v1 != infV) {
									uint32_t* m = &vertexMarks[v1->info().idx];
									if (*m != marker) {
										*m = marker;
										const point_t& pt = v1->point();
										const double dx = pt.x() - qx, dy = pt.y() - qy, dz = pt.z() - qz;
										const double d2 = dx * dx + dy * dy + dz * dz;
										if (d2 < bestSq) { bestSq = d2; best = v1; goto refine_restart; }
									}
								}

								// v2
								if (v2 != nearest && v2 != infV) {
									uint32_t* m = &vertexMarks[v2->info().idx];
									if (*m != marker) {
										*m = marker;
										const point_t& pt = v2->point();
										const double dx = pt.x() - qx, dy = pt.y() - qy, dz = pt.z() - qz;
										const double d2 = dx * dx + dy * dy + dz * dz;
										if (d2 < bestSq) { bestSq = d2; best = v2; goto refine_restart; }
									}
								}

								// v3
								if (v3 != nearest && v3 != infV) {
									uint32_t* m = &vertexMarks[v3->info().idx];
									if (*m != marker) {
										*m = marker;
										const point_t& pt = v3->point();
										const double dx = pt.x() - qx, dy = pt.y() - qy, dz = pt.z() - qz;
										const double d2 = dx * dx + dy * dy + dz * dz;
										if (d2 < bestSq) { bestSq = d2; best = v3; goto refine_restart; }
									}
								}
							}

							// === Only expand neighbors if no refinement happened ===
							if (best == nearest) {
								// reuse v0..v3 we already loaded to avoid re-calling vertex(i)
								// enqueue exactly like your code: same cells, same order
								if (v0 != nearest) {
									cell_handle_t next = c->neighbor(0);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push_back(next); }
								}
								if (v1 != nearest) {
									cell_handle_t next = c->neighbor(1);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push_back(next); }
								}
								if (v2 != nearest) {
									cell_handle_t next = c->neighbor(2);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push_back(next); }
								}
								if (v3 != nearest) {
									cell_handle_t next = c->neighbor(3);
									auto& nm = next->tds_data().marker;
									if (nm != marker) { nm = marker; cellQueue.push_back(next); }
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
							} else {
								// refinement occurred, stop immediately
								break;
							}
						}
#endif

refine_restart:
						if (best == nearest)
							break;

						nearest = best;
						vertexMarks[nearest->info().idx] = marker;
						const point_t& nearestPtNew = nearest->point();
						bestSq = fast_sqdist2(qx, qy, qz, nearestPtNew.x(), nearestPtNew.y(), nearestPtNew.z());
					}
				}
				hint = nearest;


				//const auto& hintPt2 = hint->point();
				ASSERT(hint == delaunay.nearest_vertex(p, hint->cell()));

				// Projection visibility check
				const float pxF = (float)px, pyF = (float)py, pzF = (float)pz;
				const float nxF = (float)hint->point().x(), nyF = (float)hint->point().y(), nzF = (float)hint->point().z();

				constexpr float depthThreshold = 0.01f;

				bool shouldInsert = false;
				for (size_t j = 0; j < numViews; ++j) {
					const float * __restrict camera = &viewCameras[views[j]][0];

					const float pez = camera[8]*pxF + camera[9]*pyF + camera[10]*pzF + camera[11];
					if (pez <= 0.f) continue;

					const float pnz = camera[8]*nxF + camera[9]*nyF + camera[10]*nzF + camera[11];
					if (pnz <= 0.f) continue;

					if (fabsf(pnz - pez) >= depthThreshold * pez) {
						shouldInsert = true;
						break;
					}

					const float invPez = 1.f / pez;
					const float invPnz = 1.f / pnz;

					const float pex = (camera[0]*pxF + camera[1]*pyF + camera[2]*pzF + camera[3]) * invPez;
					const float pnx = (camera[0]*nxF + camera[1]*nyF + camera[2]*nzF + camera[3]) * invPnz;

					const float pey = (camera[4]*pxF + camera[5]*pyF + camera[6]*pzF + camera[7]) * invPez;
					const float pny = (camera[4]*nxF + camera[5]*nyF + camera[6]*nzF + camera[7]) * invPnz;

					const float dx = pex - pnx;
					const float dy = pey - pny;
					if (dx*dx + dy*dy > distInsertSq) {
						shouldInsert = true;
						break;
					}
				}

				if (shouldInsert) {
					hint = delaunay.insert(p, lt, c, li, lj);
					ASSERT(anchor != vertex_handle_t());
				}

				// Visibility information not needed for the dt, but used in the next step.
				// idx is the index of the spatially sorted point.
				InsertViews(hint->info().idx, pointcloud, i);
advance:
				if (!(i & 255)) progress += 256;
			}
		}

		progress.process();
		progress.close();

		decltype(cellQueue)().swap(cellQueue);
		decltype(viewCameras)().swap(viewCameras);

		numDelaunayVertices = delaunay.number_of_vertices(); // Number of finite vertices, has one more.
		std::cerr << "verts : " << numDelaunayVertices << "\n";
		const size_t numNodes(delaunay.number_of_cells());
		const size_t numCells = numNodes; // cheaper than all_cells.size() if available
		cellIterators.reserve(numCells);
		cell_size_t ciID(0);
		
		DWORD64 t0 = __rdtsc();

		// Exact edges 4.36 s
		std::vector<cell_handle_t> finiteCells;
		finiteCells.reserve(delaunay.number_of_cells());
		size_t infiniteCells = 0;

		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), eci=delaunay.all_cells_end(); ci!=eci; ++ci, ++ciID) {
			cellIterators.push_back(ci);
			ci->info() = ciID;
	
			// skip the finite cells
			if (!delaunay.is_infinite(ci)) {
				finiteCells.push_back(ci);
				continue;
			}
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

		totalCells = ciID;

#if 1 // New idea for median calculation
		// 25510599386 4.47s:
		const auto maxIndex = vert_info_t::g_idx;
		idToVertex.resize(maxIndex + 1); // or allocate based on vertex count
		for (auto vit = delaunay.finite_vertices_begin(), end = delaunay.finite_vertices_end(); vit != end; ++vit) {
			idToVertex[vit->info().idx] = vit;
		}

		const size_t totalEstimate = 6ull * finiteCells.size();
		std::vector<uint64_t> edges(totalEstimate);

#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int numThreads = omp_get_num_threads();

		// Assign cell range to this thread
		size_t cellStart = finiteCells.size() * tid / numThreads;
		size_t cellEnd = finiteCells.size() * (tid + 1) / numThreads;

		// Each cell produces 6 edges and compute edge range
		size_t edgeStart = 6 * cellStart;
		uint64_t* __restrict p = edges.data() + edgeStart;

		for (size_t i = cellStart; i < cellEnd; ++i) {
			const cell_handle_t ci = finiteCells[i];

			uint32_t ids[4] = {
			  ci->vertex(0)->info().idx,
			  ci->vertex(1)->info().idx,
			  ci->vertex(2)->info().idx,
			  ci->vertex(3)->info().idx
			};

#define STORE(u, v) do { \
			if ((u) > (v)) std::swap((u), (v)); \
				*p++ = ((uint64_t)(u) << 32) | (v); \
			} while (0)

			STORE(ids[0], ids[1]);
			STORE(ids[0], ids[2]);
			STORE(ids[0], ids[3]);
			STORE(ids[1], ids[2]);
			STORE(ids[1], ids[3]);
			STORE(ids[2], ids[3]);
		}
	}

		decltype(finiteCells)().swap(finiteCells);

		// Sort and deduplicate
		tbb::parallel_sort(edges.begin(), edges.end());
		edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

		std::vector<float> dists(edges.size());

#pragma omp parallel for schedule(static)
		for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(edges.size()); ++i) {
			uint64_t code = edges[i];
			uint32_t id0 = code >> 32;
			uint32_t id1 = code & 0xFFFFFFFF;

			const point_t& p0 = idToVertex[id0]->point();
			const point_t& p1 = idToVertex[id1]->point();

			const double x0 = p0.x(), y0 = p0.y(), z0 = p0.z();
			const double x1 = p1.x(), y1 = p1.y(), z1 = p1.z();

			const float dx = (float)(x0 - x1);
			const float dy = (float)(y0 - y1);
			const float dz = (float)(z0 - z1);
			dists[i] = dx * dx + dy * dy + dz * dz;
		}

		const size_t numDistances = dists.size();
		std::nth_element(dists.begin(), dists.begin() + numDistances/2, dists.end());
		approxMedian = dists[numDistances/2];
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
	
				if (v0 < v1) { float dx = p0.x() - p1.x(), dy = p0.y() - p1.y(), dz = p0.z() - p1.z(); local.push_back(dx*dx + dy*dy + dz*dz); }
				if (v0 < v2) { float dx = p0.x() - p2.x(), dy = p0.y() - p2.y(), dz = p0.z() - p2.z(); local.push_back(dx*dx + dy*dy + dz*dz); }
				if (v0 < v3) { float dx = p0.x() - p3.x(), dy = p0.y() - p3.y(), dz = p0.z() - p3.z(); local.push_back(dx*dx + dy*dy + dz*dz); }
				if (v1 < v2) { float dx = p1.x() - p2.x(), dy = p1.y() - p2.y(), dz = p1.z() - p2.z(); local.push_back(dx*dx + dy*dy + dz*dz); }
				if (v1 < v3) { float dx = p1.x() - p3.x(), dy = p1.y() - p3.y(), dz = p1.z() - p3.z(); local.push_back(dx*dx + dy*dy + dz*dz); }
				if (v2 < v3) { float dx = p2.x() - p3.x(), dy = p2.y() - p3.y(), dz = p2.z() - p3.z(); local.push_back(dx*dx + dy*dy + dz*dz); }
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
		std::nth_element(distsSq.get(), distsSq.get() + totalSize/2, distsSq.get() + totalSize);
		approxMedian = distsSq[totalSize/2];
#endif

		decltype(dists)().swap(dists);

		auto t1 = rdtscEnd();

		DEBUG("Median time %g", rdtscToSeconds(t1 - t0, cpuHz));

		infoCells.resize(totalCells);
		memset(&infoCells[0], 0, sizeof(cell_info_t)*totalCells);

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
			fetchCellFacets<CGAL::POSITIVE>(delaunay, hullFacets, camCell.cell, imageData, camCell.facets);
			// link all cells contained by the camera to the source
			for (const facet_t& f: camCell.facets)
				infoCells[f.first->info()].s = kInf;
		}

		const size_t numFiniteCells = cellIterators.size() - infiniteCells;

#ifdef FACET_DIAGNOSTICS // Just used in diagnostics
		numFiniteFacets = 0;
		numFacets = 0;
		for (auto fi=delaunay.facets_begin(), ffi=delaunay.facets_end(); fi!=ffi; ++fi) {
			if (!delaunay.is_infinite(*fi)) {
				++numFiniteFacets;
			}
			++numFacets;
		}
#endif

#ifdef FACET_DIAGNOSTICS
		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, %u (+%u) faces (%s)",
			numVertices, delaunay.number_of_vertices(), numFiniteCells, infiniteCells, numFiniteFacets,  numFacets-numFiniteFacets, TD_TIMER_GET_FMT().c_str());
#else
		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, faces not calculated (%s)",
			numVertices, delaunay.number_of_vertices(), numFiniteCells, infiniteCells, TD_TIMER_GET_FMT().c_str());
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
		distsSq.release();
#endif

		// compute the weights for each edge
		Util::Progress progress(_T("Points weighted"), numDelaunayVertices);

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
		struct ThreadData
		{
			PaddedVector<facet_t> mFacets;
			PaddedVector<edge_cap_t*> mPts;
			PaddedVector<edge_cap_t> mVis;
			PaddedVector<edge_cap_t> mDist;
			PaddedVector<uint32_t> mViewIdxs;
		};

		std::vector<ThreadData> perThreadData;

		delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());
		const int64_t nVerts(delaunay.number_of_vertices());

		std::vector<delaunay_t::Vertex_handle> vertexHandles(nVerts);
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
			auto& pts = td.mPts.mData;
			auto& vis = td.mVis.mData;
			auto& dist = td.mDist.mData;
			auto& viewIdxs = td.mViewIdxs.mData;

			facets.reserve(128); // JPB WIP OPT more?
			pts.reserve(8192);
			vis.reserve(8192);
			dist.reserve(8192);
			viewIdxs.reserve(256);

#pragma omp for schedule(static, 1024) // 1024 better than alternatives on 7950X
			for (int64_t i=0; i<nVerts; ++i) {
#if 1
				auto vi = vertexHandles[i];
#else
				delaunay_t::Vertex_iterator vi;
#pragma omp critical
				vi = vertexIter++;
#endif
				vert_info_t& vert(vi->info());
				auto& viewInstance = allViews[vert.idx];
				if (viewInstance.empty())//IsEmpty())
					continue;
				const point_t& p(vi->point());
				const Point3 pt(CGAL2MVS<REAL>(p));

				pts.clear();
				vis.clear();
				dist.clear();
				viewIdxs.clear();

				// To accelerate the vert.views creation, we just store
				// them as fast as possible.
				// Here, because there may be duplicates we sort them
				// and assign a (constant) weight to the point which equals
				// the number of views.
				for (auto& i : viewInstance) {
					auto* src = &pointcloud.pointViewsMemory[offsets[i]];
					auto cnt = sizes[i];
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
					// find faces intersected by the camera-point segment
					const segment_t segCamPoint(MVS2CGAL(camera.C), p);
					intersection_t inter(pt, Point3(vecCamPoint * invLenCamPoint));

					if (!intersect(delaunay, segCamPoint, camCell.facets, facets, inter))
						continue;
					do {
						// assign score, weighted by the distance from the point to the intersection
						edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
						pts.push_back(&f);
						vis.push_back(alpha_vis);
						dist.push_back((edge_cap_t)inter.dist);
					} while (intersect(delaunay, segCamPoint, facets, facets, inter));
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
					// find faces intersected by the endpoint-point segment
					inter.dist = FLT_MAX; inter.bigger = false;
					const Point3 endPoint(pt + vecCamPoint * (invLenCamPoint * sigma));
					const segment_t segEndPoint(MVS2CGAL(endPoint), p);
					const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell()));
					ASSERT(endCell != cell_handle_t());
					fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
					edge_cap_t& t(infoCells[endCell->info()].t);
					AtomicAddFloat(&t, alpha_vis);
					while (intersect(delaunay, segEndPoint, facets, facets, inter)) {
						// assign score, weighted by the distance from the point to the intersection
						// inline mirror_facet
						cell_handle_t  c = inter.facet.first;
						const int      i = inter.facet.second;
						cell_handle_t  nc = c->neighbor(i);
						const int      mi = delaunay.mirror_index(c, i);

						edge_cap_t* fp = &infoCells[nc->info()].f[mi];

						pts.push_back(fp);
						vis.push_back(alpha_vis);
						dist.push_back((edge_cap_t)inter.dist);
					}
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
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
					const Point3 vecCamPoint(pt-camera.C);
					const REAL invLenCamPoint(REAL(1)/norm(vecCamPoint));
					// find faces intersected by the camera-point segment
					const segment_t segCamPoint(MVS2CGAL(camera.C), p);

					std::cout << "oldi (first):   ";
					intersection_t inter(pt, Point3(vecCamPoint*invLenCamPoint));
					if (!intersectv(delaunay, segCamPoint, camCell.facets, facets, inter))
						continue;
					do {
						// assign score, weighted by the distance from the point to the intersection
						const edge_cap_t w(alpha_vis*(1.f-EXP(-SQUARE((float)inter.dist)*inv2SigmaSq)));
						std::cout << w << " ";
						//edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
						//#ifdef DELAUNAY_USE_OPENMP
						//#pragma omp atomic
						//#endif
						//f += w;
					}
					while (intersectv(delaunay, segCamPoint, facets, facets, inter));
					std::cout << "\n\noldi (second):   ";

					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
					// find faces intersected by the endpoint-point segment
					inter.dist = FLT_MAX; inter.bigger = false;
					const Point3 endPoint2(pt+vecCamPoint*(invLenCamPoint*sigma));
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
						const edge_cap_t w(alpha_vis*(1.f-EXP(-SQUARE((float)inter.dist)*inv2SigmaSq)));
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

				// Here we apply the deferred intersection results to the edges all at once.
				const _Data vInv2SigmaSq = _Set(inv2SigmaSq);

				const size_t numEights = pts.size()/8;
				const size_t numRemaining = pts.size() & 7;
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
					const _Data vDists2 = _Load(d+4);
					const _Data vAlphaVis = _Load(v);
					const _Data vAlphaVis2 = _Load(v+4);
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
					float tmp = (*v*(1.f- JPBEXP(SQUARE((float)*d)*(inv2SigmaSq))));
#ifdef VALIDATE
					std::cout << tmp << " ";
#endif
					**pp += tmp;
				}
#ifdef VALIDATE
				std::cout << "\n";
#endif

				if (!(i&255)) { progress += 256; }
			}
		} // parallel
#endif

		progress.process();
		progress.close();

		decltype(allViews)().swap(allViews);
		decltype(offsets)().swap(offsets);
		decltype(sizes)().swap(sizes);
		decltype(camCells)().swap(camCells);

#ifdef FACET_DIAGNOSTICS
		DEBUG_EXTRA("Delaunay tetrahedras weighting completed: %u cells, %u faces (%s)", delaunay.number_of_cells(), numFacets, TD_TIMER_GET_FMT().c_str());
		#else
		DEBUG_EXTRA("Delaunay tetrahedras weighting completed: %u cells, unknown faces (%s)", delaunay.number_of_cells(), TD_TIMER_GET_FMT().c_str());
		#endif
	}

	// JPB WIP BUG Parallel graphcut neds to change compuatePlaneSphareAngle 

	// run graph-cut and extract the mesh
	{
		TD_TIMER_STARTD();
		//DWORD_PTR originalMask = SetAffinityToCPU0();

		auto t0 = rdtscStart();

		// create graph
		constexpr edge_cap_t maxCap(3.402823466e+34f/*FLT_MAX*0.0001f*/);

#if 1 // parallel graph-cut set up brings 29s to about 9s
		MaxFlow<cell_size_t,edge_cap_t> graph(cellIterators.size());

		struct FacetAngleIndex {
			// Flat angle array, 4 per cell
			PaddedVector<float> angle;             // size = 4 * totalCells
			PaddedVector<int32_t> cellIDToIdx;     // size = totalCells
		};

		FacetAngleIndex facetData;
		facetData.angle.resize(totalCells * 4); // 1.6 Gb
		facetData.cellIDToIdx.resize(totalCells); // 400Mb

		// --- Compute facet angles in parallel ---
		#pragma omp parallel for schedule(static)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)totalCells; ++i) {
			const auto ci = cellIterators[i];
			const cell_size_t cellID = ci->info();

			// Safe single-thread write if cellIDs are unique
			facetData.cellIDToIdx[cellID] = (int32_t)i;
#if 1
			float a4[4];
			computePlaneSphereAngle4(delaunay, ci, a4);
			facetData.angle[i * 4 + 0] = a4[0];
			facetData.angle[i * 4 + 1] = a4[1];
			facetData.angle[i * 4 + 2] = a4[2];
			facetData.angle[i * 4 + 3] = a4[3];
#else
			facetData.angle[i * 4 + 0] = computePlaneSphereAngle(delaunay, facet_t(ci, 0));
			facetData.angle[i * 4 + 1] = computePlaneSphereAngle(delaunay, facet_t(ci, 1));
			facetData.angle[i * 4 + 2] = computePlaneSphereAngle(delaunay, facet_t(ci, 2));
			facetData.angle[i * 4 + 3] = computePlaneSphereAngle(delaunay, facet_t(ci, 3));
#endif
		}

		// --- Graph construction pass ---
		struct EdgeDesc
		{
			EdgeDesc() {}
			EdgeDesc(int f, int t, edge_cap_t c, edge_cap_t r):
				from(f),
				to(t),
				cap(c),
				revCap(r)
			{}
			
			int from, to;
			edge_cap_t cap, revCap;
		};

		struct NodeDesc
		{
			NodeDesc() {}
			NodeDesc(int id_, edge_cap_t source_, edge_cap_t sink_) :
				id(id_),
				source(source_),
				sink(sink_)
			{}

			int id;
			edge_cap_t source, sink;
		};

		struct alignas(64)  ThreadLocalBuffer
		{
			std::vector<NodeDesc> nodes;
			std::vector<EdgeDesc> edges;
			edge_cap_t flow = 0;
			char pad[64 - sizeof(flow)]; // force `flow` onto its own cache line
		};

#if 0

		node is big 144 bytes memset takes a long time 8 seconds or so

			need this par:

		template < class T, class Allocator, class Increment_policy, class TimeStamper >
		void Compact_container<T, Allocator, Increment_policy, TimeStamper>::clear()
		{
			for (typename All_items::iterator it = all_items.begin(), itend = all_items.end();
				it != itend; ++it) {
				pointer p = it->first;
				size_type s = it->second;
				for (pointer pp = p + 1; pp != p + s - 1; ++pp) {
					if (type(pp) == USED)
					{
						std::allocator_traits<allocator_type>::destroy(alloc, pp);
						set_type(pp, nullptr, FREE);
					}
				}
				alloc.deallocate(p, s);
			}
			init();
		}



		eneed flow parallel(atomic)

			save scene as binary
			--archive - type  1 add to GM
#endif

		const int threadCount = omp_get_max_threads();
		std::vector<ThreadLocalBuffer> threadBuffers(threadCount);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			ThreadLocalBuffer& buf = threadBuffers[tid];

			const size_t est = (totalCells + threadCount - 1) / threadCount;
			constexpr size_t pad = 2048; // schedule
			buf.nodes.resize(est + pad); // 1.2Gb total
			buf.edges.resize((est + pad) * 4); // Worst case. 6.4Gb total
			auto* __restrict dstNodes = buf.nodes.data();
			auto* __restrict dstEdges = buf.edges.data();

			// Manual static partition
			const ptrdiff_t chunk = (totalCells + threadCount - 1) / threadCount;
			const ptrdiff_t start = tid * chunk;
			const ptrdiff_t end = std::min<ptrdiff_t>(start + chunk, totalCells);

			for (ptrdiff_t idx = start; idx < end; ++idx) {
				const auto ci = cellIterators[idx];
				const int ciID = ci->info();
				const auto& ciInfo = infoCells[ciID];

				// Compute terminal capacities
				edge_cap_t s = ciInfo.s;
				edge_cap_t t = FastMinS(ciInfo.t, maxCap);

				edge_cap_t f = graph.graph.nodes[ciID].excess;
				if (f > 0) s += f;
				else t -= f;

				edge_cap_t push = FastMinS(s, t);
				buf.flow += push;

				dstNodes->id = ciID;
				dstNodes->source = s;
				dstNodes->sink = t;
				++dstNodes;

				for (int i = 0; i < 4; ++i) {
					const auto cj = ci->neighbor(i);
					const int cjID = cj->info();

					if (cjID < ciID) continue;

					const int j = cj->index(ci);
					const auto& cjInfo = infoCells[cjID];

					const float angleCi = facetData.angle[idx * 4 + i];
					const int cjIdx = facetData.cellIDToIdx[cjID];
					const float angleCj = facetData.angle[cjIdx * 4 + j];

					float minAngle = FastMinS(angleCi, angleCj);
					dstEdges->from = ciID;
					dstEdges->to = cjID;
					uint32_t bits;
					memcpy(&bits, &minAngle, sizeof(float));

					// Fast finite check for float (avoids std::isfinite overhead)
					if ((bits & 0x7f800000u) == 0x7f800000u) {
						dstEdges->cap = dstEdges->revCap = maxCap; // NaN or Inf -> clamp to max
					}
					else {
						const edge_cap_t q = (1.f - minAngle) * kQual;
						const edge_cap_t iCap = (ciInfo.f[i] >= maxCap) ? maxCap : Quantize(ciInfo.f[i] + q, maxCap);
						const edge_cap_t jCap = (cjInfo.f[j] >= maxCap) ? maxCap : Quantize(cjInfo.f[j] + q, maxCap);
						dstEdges->cap = iCap;
						dstEdges->revCap = jCap;
					}
					++dstEdges;
				}
			}

			buf.edges.resize(dstEdges - buf.edges.data());
		}

		for (const auto& buf : threadBuffers)
			graph.graph.flow += buf.flow;

		for (const auto& buf : threadBuffers)
			for (const auto& n : buf.nodes)
				graph.AddNode(n.id, n.source, n.sink);

#if 1
#pragma omp parallel for schedule(static)
		for (int i = 0; i < (int)threadBuffers.size(); ++i) {
			const auto& buf = threadBuffers[i];
			for (const auto& e : buf.edges) {
				graph.AddEdge(e.from, e.to, e.cap, e.revCap); // now thread-safe
			}
		}
#else
		for (const auto& buf : threadBuffers)
			for (const auto& e : buf.edges)
				graph.AddEdge(e.from, e.to, e.cap, e.revCap);
#endif

#else
		MaxFlow<cell_size_t,edge_cap_t> graph(delaunay.number_of_cells());
		// set weights
		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), ce=delaunay.all_cells_end(); ci!=ce; ++ci) {
			const cell_size_t ciID(ci->info());
			const cell_info_t& ciInfo(infoCells[ciID]);
			graph.AddNode(ciID, ciInfo.s, MINF(ciInfo.t, maxCap));
		for (int i = 0; i < 4; ++i) {
				const cell_handle_t cj(ci->neighbor(i));
				const cell_size_t cjID(cj->info());
			if (cjID < ciID) continue;
				const cell_info_t& cjInfo(infoCells[cjID]);
				const int j(cj->index(ci));
				const edge_cap_t q((1.f - MINF(computePlaneSphereAngle(delaunay, facet_t(ci,i)), computePlaneSphereAngle(delaunay, facet_t(cj,j))))*kQual);
				graph.AddEdge(ciID, cjID, ciInfo.f[i]+q, cjInfo.f[j]+q);
		}
		}
		#endif

		std::vector<cell_info_t>().swap(infoCells); // release memory
		auto t1 = rdtscEnd();
	  //RestoreAffinity(originalMask); // Restore original affinity

		std::cout << "   Startup: " << rdtscToSeconds(t1 - t0, cpuHz) << "\n";

		// find graph-cut solution
		const float maxflow(graph.ComputeMaxFlow());

		//originalMask = SetAffinityToCPU0();
		auto t2 = rdtscStart();

		std::cout << "   Graph-cut itself: " << rdtscToSeconds(t2 - t1, cpuHz) << "\n";

#if 1 // parallel surface extraction.  Originally 12s-13s, now 2-3
		struct LocalMeshData {
			std::vector<Mesh::Face> localFaces;
		  std::vector<uint32_t> localVertexIDs; // vertex idxs
			tsl::robin_map<uint32_t, Mesh::VIndex> localIndexMap;
		};
		const int nThreads = omp_get_max_threads();

		std::vector<LocalMeshData> localData(nThreads);

		const size_t cellsPerThread = (totalCells + nThreads - 1) / nThreads;
		for (int t = 0; t < nThreads; ++t) {
			localData[t].localFaces.reserve(cellsPerThread * 4);
			localData[t].localVertexIDs.reserve(cellsPerThread * 12);
			// Reserve on the map is much more expensive than vector reserve.
		}

		#pragma omp parallel
		{
			// Remove all barriers.
			const int tid = omp_get_thread_num();
			auto& local = localData[tid];
			int nt = omp_get_num_threads();

			ptrdiff_t chunk = (totalCells + nt - 1) / nt;
			ptrdiff_t start = tid * chunk;
			ptrdiff_t end = std::min(start + chunk, (ptrdiff_t) totalCells);

			for (ptrdiff_t idx = start; idx < end; ++idx) {
				auto ci = cellIterators[idx];
				const cell_size_t ciID = ci->info();

				Mesh::Face face;
				for (int f = 0; f < 4; ++f) {
					if (delaunay.is_infinite(ci, f)) continue;
					const cell_handle_t cj = ci->neighbor(f);
					const cell_size_t cjID = cj->info();
					if (ciID < cjID) continue;

					const bool ciType = graph.IsNodeOnSrcSide(ciID);
					if (ciType == graph.IsNodeOnSrcSide(cjID)) continue;

					const triangle_vhandles_t tri = getTriangle(ci, f);

					for (int v = 0; v < 3; ++v) {
						const vertex_handle_t vh = tri.verts[v];
						const uint32_t vertexID = vh->info().idx;

						auto [it, inserted] = local.localIndexMap.try_emplace(vertexID, (Mesh::VIndex)local.localVertexIDs.size());
						if (inserted)
							local.localVertexIDs.push_back(vertexID);

						face[v] = it->second; // local index
					}

					if (!ciType)
						std::swap(face[0], face[2]);

					local.localFaces.emplace_back(face[0], face[1], face[2]);
				}
			}
		}

		std::vector<delaunay_t::All_cells_iterator>().swap(cellIterators); // free memory

		constexpr Mesh::VIndex kInvalid = ~0;
		const size_t numVertices = delaunay.number_of_vertices();
		std::vector<Mesh::VIndex> idxToGlobalIndex(numVertices, kInvalid);

		mesh.vertices.Reserve((Mesh::VIndex)numVertices);
		mesh.faces.Reserve((Mesh::FIndex)numVertices * 4); // Worst-case

		for (const auto& local : localData) {
			for (uint32_t idx : local.localVertexIDs) {
				if (idxToGlobalIndex[idx] == kInvalid) {
					idxToGlobalIndex[idx] = mesh.vertices.GetSize();
					vertex_handle_t vh = idToVertex[idx];
					mesh.vertices.Insert(CGAL2MVS<Mesh::Vertex::Type>(vh->point()));
				}
			}
		}

		// Remap local faces to global indices and insert
		for (const auto& local : localData) {
			for (const auto& face : local.localFaces) {
				Mesh::Face& f = mesh.faces.AddEmpty(); // creates a reference directly in-place
				for (int i = 0; i < 3; ++i)
					f[i] = idxToGlobalIndex[local.localVertexIDs[face[i]]];
			}
		}
#else
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
		// JPB WIP BUG Not needed memory released on exit. delaunay.clear();

		auto t3 = rdtscEnd();
	  //RestoreAffinity(originalMask); // Restore original affinity

		std::cout << "   End: " << rdtscToSeconds(t3 - t2, cpuHz) << "\n";

		DEBUG_EXTRA("Delaunay tetrahedras graph-cut completed (%g flow): %u vertices, %u faces (%s)", maxflow, mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

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

	return true;
}
/*----------------------------------------------------------------*/
