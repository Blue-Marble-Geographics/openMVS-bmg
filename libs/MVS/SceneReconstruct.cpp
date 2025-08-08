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

#undef NO_MANIFOLD_FIXUP

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

typedef boost::container::small_vector<PointCloud::Index, 10> view_vec_t;
std::vector<view_vec_t> allViews; // faces' weight from the cell outwards

void InsertViews(size_t vertexId, const PointCloudStreaming& pc, PointCloud::Index idxPoint)
{
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

	const double cDiff[] { c.x()-p.x(), c.y()-p.y(), c.z()-p.z() };
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

#ifdef VALIDATE
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
#endif

// Check intersection between a facet (f) and a segment (s)
// (derived from CGAL::do_intersect in CGAL/Triangle_3_Segment_3_do_intersect.h)
//  coplanar [out] : pointer to the 3 int array of indices of the edges coplanar with (s)
// return -1 if there is no intersection or
// the number of edges coplanar with the segment (0 = intersection inside the triangle)
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

inline double IntersectsDist(const SEACAVE::Ray3& ray, const CGAL::Plane_3<kernel_t>& plane)
{
	const auto& normal = plane.orthogonal_vector();
	const double nx = normal.x();
	const double ny = normal.y();
	const double nz = normal.z();

	const double Vd = nx * ray.m_vDir.x() + ny * ray.m_vDir.y() + nz * ray.m_vDir.z();
	const double Vo = -(nx * ray.m_pOrig.x() + ny * ray.m_pOrig.y() + nz * ray.m_pOrig.z() + plane.d());

	constexpr double eps = 1e-12;
	const double safeVd = (std::abs(Vd) < eps) ? std::copysign(eps, Vd) : Vd;

	return Vo / safeVd;
} // IntersectsDist(PLANE)

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
	vertex_handle_t vs[3];
	for (const facet_t& in_facet: in_facets) {
		ASSERT(!Tr.is_infinite(in_facet));
		Tr.triangle_vertices(vs, in_facet.first, in_facet.second);
		const int nb_coplanar(intersect(vs, seg, segDiff, segDiff2, coplanar));
		if (nb_coplanar >= 0) {
			// skip this cell if the intersection is not in the desired direction
			//const REAL interDist(inter.ray.IntersectsDist(getFacetPlane(in_facet)));
			const REAL interDist(IntersectsDist(inter.ray, getFacetPlane(in_facet)));
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
  const float x0 = p0.x(), y0 = p0.y(), z0 = p0.z();
  const float x1 = p1.x(), y1 = p1.y(), z1 = p1.z();
  const float x2 = p2.x(), y2 = p2.y(), z2 = p2.z();

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
__forceinline float AtomicAddFloat(float* __restrict addr, float val) {
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

template <bool UseROI>
size_t ProcessPoints(
	const float* __restrict pPointStream,
	size_t numVertices,
	DELAUNAY::point_t* __restrict origVertices,
	ptrdiff_t* __restrict indices,
  const SEACAVE::OBB3f& obb
)
    {
  size_t validCount = 0;

  for (size_t i = 0, j = 0; i < numVertices; ++i, j += 3) {
    const float x = pPointStream[j];
    const float y = pPointStream[j+1];
    const float z = pPointStream[j+2];

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

float Quantize(float cap)
{
#if 1
  int scaled = static_cast<int>(cap * 2.0f + 0.5f);
  return 0.5f * scaled;
#else
  // Step is 0.2, so multiply by 5 and round to nearest int
  int scaled = static_cast<int>(cap * 5.0f + 0.5f);
  return 0.2f * scaled;
#endif
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

	FOREACH(i, images)
{
		Image& imageData = images[i];
		if (!imageData.IsValid())
					continue;
		for (int j = 0; j < imageData.camera.P.elems; ++j) {
			viewCameras[i][j] = (float) imageData.camera.P[j];
}
}
	using namespace DELAUNAY;
	ASSERT(!pointcloud.IsEmpty());
	mesh.Release();

	// create the Delaunay triangulation
	const size_t numPointCloudVertices = pointcloud.NumPoints();
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

	{
		TD_TIMER_STARTD();

		std::vector<point_t> origVertices;
		origVertices.resize(numPointCloudVertices);
		//std::vector<std::ptrdiff_t> indices(numPointCloudVertices);
		//indices.resize(numPointCloudVertices);
		double eps2 = 1e-10;

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


#ifndef VALIDATE
		// sort vertices
		typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t::Geom_traits, point_t*> Search_traits;
		// JPB The runtime here is not consistent.
			CGAL::spatial_sort<CGAL::Parallel_tag>(indices.begin(), indices.begin() + numVertices, Search_traits(&origVertices[0], delaunay.geom_traits()));
#endif

			// origVertices[i] refers to the original data.
			// indices[i] maps the sorted data to the original data.
			// Rewrite the vertex data in index sorted form:
			vertices.reset(new point_t[numVertices]);
			for (size_t i = 0; i < numVertices; ++i) {
				vertices[i] = origVertices[indices[i]];
			}

		// insert vertices
			// JPB WIP BUG delaunay.tds().cells().reserve(numVertices*6); // May reserve dynamically
			// JPB WIP BUG delaunay.tds().vertices().reserve(numVertices);
			allViews.resize(numVertices);

			// jpb wip bug this is very interesting it was reserve before. but allowed the filteredPt[i] write.
		
			DEBUG_EXTRA("Total prep time is: %s", TD_TIMER_GET_FMT().c_str());
		}
		Util::Progress progress(_T("Points inserted"), indices.size());

		// Here we keep track of versioning for testing.
		// Both delaunay.info() and vcg::tri::Info() only compile
		// if we are using custom versions of these libraries.
		// The version returned can be used to track revisions
		// and can be manually adjusted.c
		// delaunay.info() is parallel can be used to make sure
		// we are compiling and using the work with TBB.
#if 1
		DEBUG("------------------------------------------");
		DEBUG("ReconstructMesh optimization version 1.1.3");
		const auto [isParallel, CGALversion] = CGAL::info();
		DEBUG("Parallel: %s", isParallel ? "true" : "false");
		DEBUG("CGAL version: = %d", CGALversion);
		const int vcgVersion = vcg::tri::Info();
		DEBUG("VCG version: = %d", vcgVersion);
		DEBUG("------------------------------------------");
#endif
		// Fixed storage is slightly faster, but difficult to maintain.
		constexpr size_t kMaxCells = 16384;
		std::vector<cell_handle_t> cellQueue;
		cellQueue.reserve(kMaxCells);

		auto itIndices = std::cbegin(indices);
		vertex_handle_t hint;

		// InsertViews first parameter must be the dt's index --verified by validation code.
		if (distInsert <= 0) {
			for (size_t i = 0; i < numVertices; ++i, ++itIndices) {
				const auto idx = *itIndices; // This sorted vertex' original position.
				const point_t& p = vertices[i]; // These are the sorted vertices.
				// insert all points
				hint = delaunay.insert(p, hint);
				ASSERT(anchor != vertex_handle_t());
				// update point visibility info
				InsertViews(hint->info().idx, pointcloud, idx);
				if (!(i & 255)) {
					progress += 256;
				}
			}
		} else {
			std::vector<uint32_t> vertexMarks(numVertices);
			uint32_t marker = 0;

			ptrdiff_t idx = *itIndices++;
			hint = delaunay.insert(vertices[0]);
			InsertViews(hint->info().idx, pointcloud, idx);

			for (int i = 1; i < numVertices; ++i, ++itIndices) {
				const point_t& p = vertices[i];
				const double px = p.x();
				const double py = p.y();
				const double pz = p.z();
				const ptrdiff_t idx = *itIndices;

				const uint32_t* __restrict pointViewsOffset = pointcloud.pointViewsOffsets.data() + idx;
				_mm_prefetch((const char*)pointViewsOffset, _MM_HINT_T1);

				const uint32_t* __restrict pointViewSizes = pointcloud.pointViewsSizes.data() + idx;
				_mm_prefetch((const char*)pointViewSizes, _MM_HINT_T1);

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
									auto& nextMarker = next->tds_data().marker;
									if (nextMarker == marker) continue;

									for (int j = 0; j < 4; ++j) {
										if (next->vertex(j) == nearest) {
											nextMarker = marker;
											cellQueue.push_back(next);
											break;
										}
									}
								}
							} else {
								// refinement occurred, stop immediately
								break;
							}
						}

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
				InsertViews(hint->info().idx, pointcloud, idx);
advance:
				if (!(i & 255)) progress += 256;
			}
		}

		progress.process();
		progress.close();

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
	// 25510599386 4,47s:
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
  size_t cellEnd   = finiteCells.size() * (tid + 1) / numThreads;

  // Each cell produces 6 edges and compute edge range
  size_t edgeStart = 6 * cellStart;
  uint64_t* p = edges.data() + edgeStart;

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
    dists[i] = dx*dx + dy*dy + dz*dz;
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

		auto t1 = rdtscEnd();

		DEBUG("Median time %g\n", rdtscToSeconds(t1 - t0, cpuHz));

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
			indices.size(), delaunay.number_of_vertices(), numFiniteCells, infiniteCells, numFiniteFacets,  numFacets-numFiniteFacets, TD_TIMER_GET_FMT().c_str());
#else
		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, faces not calculated (%s)",
			indices.size(), delaunay.number_of_vertices(), numFiniteCells, infiniteCells, TD_TIMER_GET_FMT().c_str());
#endif
	}

	const float sigma = (SQRT(approxMedian)*kSigma);

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
		const float sigma(SQRT(approxMedian)* kSigma);
		DEBUG_EXTRA("Sigma is %f", sigma);
		// Notice we negate inv2SigmaSq here to aid the vector calculations below.
		const float inv2SigmaSq(-0.5f/(sigma*sigma));
		// distsSq may consume a lot of memory.  Delete it now.
		distsSq.release();
#endif

		// compute the weights for each edge
		Util::Progress progress(_T("Points weighted"), numDelaunayVertices);

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
					const Point3 vecCamPoint(pt-camera.C);
					const REAL invLenCamPoint(REAL(1)/norm(vecCamPoint));
					// find faces intersected by the camera-point segment
					const segment_t segCamPoint(MVS2CGAL(camera.C), p);
					{
						intersection_t inter(pt, Point3(vecCamPoint*invLenCamPoint));

					const point_t& source = segCamPoint.source();
					const point_t& target = segCamPoint.target();
						const double segDiff[] = { target.x() - source.x(), target.y() - source.y(), target.z() - source.z() };
						const double segDiffN[] = { -segDiff[0], -segDiff[1], -segDiff[2] };

					if (!intersect(delaunay, segCamPoint, segDiff, segDiffN, camCell.facets, facets, inter))
						continue;
					do {
						// assign score, weighted by the distance from the point to the intersection
						edge_cap_t& f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
						pts.push_back(&f);
						vis.push_back(alpha_vis);
						dist.push_back((edge_cap_t)inter.dist);
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
						const double segDiff2[] = { target2.x() - source2.x(), target2.y() - source2.y(), target2.z() - source2.z() };
						const double segDiff2N[] = { -segDiff2[0], -segDiff2[1], -segDiff2[2] };

					while (intersect(delaunay, segEndPoint, segDiff2, segDiff2N, facets, facets, inter)) {
						// assign score, weighted by the distance from the point to the intersection
						const facet_t& mf(delaunay.mirror_facet(inter.facet));
						edge_cap_t& f(infoCells[mf.first->info()].f[mf.second]);
						pts.push_back(&f);
						vis.push_back(alpha_vis);
						dist.push_back((edge_cap_t)inter.dist);
					}
					ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
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

		progress.process();
		progress.close();

		camCells.clear();

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
		facetData.angle.resize(totalCells * 4);
		facetData.cellIDToIdx.resize(totalCells);

		// --- Compute facet angles in parallel ---
		#pragma omp parallel for schedule(static)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)totalCells; ++i) {
			const auto ci = cellIterators[i];
			const cell_size_t cellID = ci->info();

			// Safe single-thread write if cellIDs are unique
			facetData.cellIDToIdx[cellID] = (int32_t)i;

			facetData.angle[i * 4 + 0] = computePlaneSphereAngle(delaunay, facet_t(ci, 0));
			facetData.angle[i * 4 + 1] = computePlaneSphereAngle(delaunay, facet_t(ci, 1));
			facetData.angle[i * 4 + 2] = computePlaneSphereAngle(delaunay, facet_t(ci, 2));
			facetData.angle[i * 4 + 3] = computePlaneSphereAngle(delaunay, facet_t(ci, 3));
		}

		// --- Graph construction pass ---
		struct EdgeDesc
		{
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

		int threadCount = omp_get_max_threads();
		std::vector<ThreadLocalBuffer> threadBuffers(threadCount);

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			ThreadLocalBuffer& buf = threadBuffers[tid];

			size_t est = (totalCells + threadCount - 1) / threadCount;
			buf.nodes.reserve(est);
			buf.edges.reserve(est * 4); // Worst case.

#pragma omp for schedule(static)
			for (ptrdiff_t idx = 0; idx < (ptrdiff_t)totalCells; ++idx) {
				const auto ci = cellIterators[idx];
				const int ciID = ci->info();
				const auto& ciInfo = infoCells[ciID];

				// Compute terminal capacities
				edge_cap_t s = ciInfo.s;
				edge_cap_t t = MINF(ciInfo.t, maxCap);

				edge_cap_t f = graph.graph.nodes[ciID].excess;
				if (f > 0) s += f;
				else t -= f;

				edge_cap_t push = MINF(s, t);
				buf.flow += push;

				buf.nodes.emplace_back(ciID, s, t);

				for (int i = 0; i < 4; ++i) {
					const auto cj = ci->neighbor(i);
					const int cjID = cj->info();

					if (cjID < ciID) continue;

					const int j = cj->index(ci);
					const auto& cjInfo = infoCells[cjID];

					const float angleCi = facetData.angle[idx * 4 + i];
					const int cjIdx = facetData.cellIDToIdx[cjID];
					const float angleCj = facetData.angle[cjIdx * 4 + j];

					edge_cap_t q = (1.f - MINF(angleCi, angleCj)) * kQual;

					edge_cap_t iCap = (ciInfo.f[i] >= kInf) ? kInf : Quantize(ciInfo.f[i] + q);
					edge_cap_t jCap = (cjInfo.f[j] >= kInf) ? kInf : Quantize(cjInfo.f[j] + q);
					buf.edges.emplace_back(ciID, cjID, iCap, jCap);
				}
			}
		}

		for (const auto& buf : threadBuffers)
			graph.graph.flow += buf.flow;

		for (const auto& buf : threadBuffers)
			for (const auto& n : buf.nodes)
				graph.AddNode(n.id, n.source, n.sink);

		for (const auto& buf : threadBuffers)
			for (const auto& e : buf.edges)
				graph.AddEdge(e.from, e.to, e.cap, e.revCap);
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

		infoCells.clear();
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

		std::vector<LocalMeshData> localData(omp_get_max_threads());

		#pragma omp parallel for schedule(static)
		for (ptrdiff_t idx = 0; idx < (ptrdiff_t)totalCells; ++idx) {
			auto ci = cellIterators[idx];
			const cell_size_t ciID = ci->info();
			auto& local = localData[omp_get_thread_num()];

			for (int f = 0; f < 4; ++f) {
				if (delaunay.is_infinite(ci, f)) continue;
				const cell_handle_t cj = ci->neighbor(f);
				const cell_size_t cjID = cj->info();
				if (ciID < cjID) continue;

				const bool ciType = graph.IsNodeOnSrcSide(ciID);
				if (ciType == graph.IsNodeOnSrcSide(cjID)) continue;

				const triangle_vhandles_t tri = getTriangle(ci, f);
				Mesh::Face face;

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

				local.localFaces.push_back(std::move(face));
			}
	}

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
		delaunay.clear();

		auto t3 = rdtscEnd();
	  //RestoreAffinity(originalMask); // Restore original affinity

		std::cout << "   End: " << rdtscToSeconds(t3 - t2, cpuHz) << "\n";

		DEBUG_EXTRA("Delaunay tetrahedras graph-cut completed (%g flow): %u vertices, %u faces (%s)", maxflow, mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

#ifndef NO_MANIFOLD_FIXUP
	// fix non-manifold vertices and edges
	mesh.FixNonManifold();
#endif

	return true;
}
/*----------------------------------------------------------------*/
