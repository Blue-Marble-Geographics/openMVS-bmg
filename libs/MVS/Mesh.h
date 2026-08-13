/*
* Mesh.h
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

#ifndef _MVS_MESH_H_
#define _MVS_MESH_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Platform.h"
#include "PointCloud.h"
#include <boost/container/small_vector.hpp>
#include <type_traits>

// D E F I N E S ///////////////////////////////////////////////////

#define OPENMVS_21 // Migrated from OpenMVS 2.1

// S T R U C T S ///////////////////////////////////////////////////

#pragma optimize("", on) // JPB WIP BUG

namespace MVS {

// a mesh represented by a list vertices and triangles (faces)
class MVS_API Mesh
{
public:
	typedef float Type;

	typedef TPoint3<Type> Vertex;
	typedef uint32_t VIndex;
	typedef TPoint3<VIndex> Face;
	typedef uint32_t FIndex;

	typedef SEACAVE::cList<Vertex,const Vertex&,0,8192,VIndex> VertexArr;
	typedef SEACAVE::cList<Face,const Face&,0,8192,FIndex> FaceArr;

	typedef SEACAVE::cList<VIndex,VIndex,0,16,VIndex> VertexIdxArr;
	typedef SEACAVE::cList<FIndex,FIndex,0,16,FIndex> FaceIdxArr;
	typedef SEACAVE::cList<VertexIdxArr,const VertexIdxArr&,4,8192,VIndex> VertexVerticesArr;
	typedef SEACAVE::cList<FaceIdxArr,const FaceIdxArr&,4,8192,VIndex> VertexFacesArr;

	typedef TPoint3<Type> Normal;
	typedef SEACAVE::cList<Normal,const Normal&,0,8192,FIndex> NormalArr;

	typedef TPoint2<Type> TexCoord;
	typedef cList<TexCoord,const TexCoord&,0,8192,FIndex> TexCoordArr;

	typedef TPoint3<FIndex> FaceFaces;
	typedef cList<FaceFaces,const FaceFaces&,0,8192,FIndex> FaceFacesArr;

	// used to find adjacent face
	struct FaceCount {
		int count;
		inline FaceCount() : count(0) {}
	};
	typedef std::unordered_map<FIndex,FaceCount> FacetCountMap;
	typedef FaceCount VertCount;
	typedef std::unordered_map<VIndex,VertCount> VertCountMap;

	typedef AABB3f Box;

	// used to render a mesh
	typedef TOctree<VertexArr,Vertex::Type,3> Octree;
	struct FacesInserter {
		FaceIdxArr& cameraFaces;
		FacesInserter(FaceIdxArr& _cameraFaces)
			: cameraFaces(_cameraFaces) {}
		inline void operator() (const Octree::IDX_TYPE* indices, Octree::SIZE_TYPE size) {
			cameraFaces.Join(indices, size);
		}
		static void CreateOctree(Octree& octree, const Mesh& mesh) {
			CreateOctree(octree, mesh.faces, mesh.vertices);
		}
		// same, but from explicit arrays (allows building from a vertex snapshot on a worker thread)
		static void CreateOctree(Octree& octree, const FaceArr& faces, const VertexArr& vertices) {
			VertexArr centroids(faces.size());
			#ifdef _USE_OPENMP
			#pragma omp parallel for
			for (int_t idx=0; idx<(int_t)faces.size(); ++idx) {
				const Face& face = faces[(FIndex)idx];
				centroids[(FIndex)idx] = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) * (Type(1)/Type(3));
			}
			#else
			FOREACH(idx, faces) {
				const Face& face = faces[idx];
				centroids[idx] = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) * (Type(1)/Type(3));
			}
			#endif
			// Absolute minimum cell radius, used as a recursion floor.
			// NOTE: do NOT use octree.GetRadius() here - m_radius is only set
			// inside Insert(), so before insertion it is uninitialized and yields
			// a garbage floor that corrupts the octree bounds.
			// Without a floor, >32 coincident/degenerate centroids keep splitting
			// until the cell radius underflows to 0 and _Insert recurses forever
			// (each level does new CELL_TYPE[8]).
			// The floor MUST be derived from the centroids that are actually
			// inserted (not the mesh vertices): Insert() sizes the cell radius
			// from the centroid bounding box, so the radius the split predicate
			// receives is on that scale. Using the (always larger) vertex box
			// can make minRadius exceed the octree's own child radius and collapse
			// the whole tree to a few leaves. A non-positive/non-finite extent
			// (e.g. a diverged NaN/Inf vertex) disables the floor instead.
			Box box(true);
			FOREACH(idx, centroids)
				box.InsertFull(centroids[idx]);
			const Octree::Type extent(box.GetSize().maxCoeff());
			const Octree::Type minRadius(ISFINITE(extent) && extent > Octree::Type(0) ? extent / Octree::Type(1u << 20) : Octree::Type(0));
			octree.Insert(centroids, [minRadius](Octree::IDX_TYPE size, Octree::Type radius) {
				return size > 32 && radius > minRadius;
			});
			#if 0 && !defined(_RELEASE)
			Octree::DEBUGINFO_TYPE info;
			octree.GetDebugInfo(&info);
			Octree::LogDebugInfo(info);
			#endif
			octree.ResetItems();
		}
	};

	struct FaceChunk {
		FaceIdxArr faces;
		Box box;
	};
	typedef cList<FaceChunk,const FaceChunk&,2,16,uint32_t> FacesChunkArr;

public:
	VertexArr vertices;
	FaceArr faces;

	NormalArr vertexNormals; // for each vertex, the normal to the surface in that point (optional)
	VertexVerticesArr vertexVertices; // for each vertex, the list of adjacent vertices (optional)
	VertexFacesArr vertexFaces; // for each vertex, the list of faces containing it (optional)
	BoolArr vertexBoundary; // for each vertex, stores if it is at the boundary or not (optional)

	NormalArr faceNormals; // for each face, the normal to it (optional)
	FaceFacesArr faceFaces; // for each face, the list of adjacent faces, NO_ID for border edges (optional)
	TexCoordArr faceTexcoords; // for each face, the texture-coordinates corresponding to the contained vertices (optional)

	Image8U3 textureDiffuse; // texture containing the diffuse color (optional)

	#ifdef _USE_CUDA
	static SEACAVE::CUDA::KernelRT kernelComputeFaceNormal;
	#endif

public:
	#ifdef _USE_CUDA
	inline Mesh() {
		InitKernels(SEACAVE::CUDA::desiredDeviceID);
	}
	#endif

	void Release();
	void ReleaseExtra();
	void EmptyExtra();
	Mesh& Swap(Mesh&);
	Mesh& Join(const Mesh&);
	bool IsEmpty() const { return vertices.empty(); }
	bool IsWatertight();
	inline bool HasTexture() const { ASSERT(faceTexcoords.empty() == textureDiffuse.empty()); return !faceTexcoords.empty(); }

	Box GetAABB() const;
	Box GetAABB(const Box& bound) const;
	Vertex GetCenter() const;

	void ListIncidenteVertices();
	void ListIncidenteFaces();
	void ListIncidenteFaceFaces();
	void ListBoundaryVertices();
	void ComputeNormalFaces();
	void ComputeNormalVertices();

	void SmoothNormalFaces(float fMaxGradient=25.f, float fOriginalWeight=0.5f, unsigned nIterations=3);

	void GetEdgeFaces(VIndex, VIndex, FaceIdxArr&) const;
	void GetFaceFaces(FIndex, FaceIdxArr&) const;
	void GetEdgeVertices(FIndex, FIndex, uint32_t vs0[2], uint32_t vs1[2]) const;
	void GetAdjVertices(VIndex, boost::container::small_vector<VIndex, 32>&) const;
	void GetAdjVertexFaces(VIndex, VIndex, FaceIdxArr&) const;

#ifdef OPENMVS_21
	__forceinline int SmallMod3(int n) const noexcept { return (0x0924 >> (n << 1)) & 3; }
#if 1
	// get the edge orientation in the given face:
	// return false for backward, true for forward
	__forceinline bool GetEdgeOrientation(const Face& face, VIndex iV0, VIndex iV1) const
	{
		for (int i = 0; i < 3; ++i) {
			if (face[i] == iV0) {
				return face[SmallMod3(i + 1)] == iV1;
			}
		}
		return false; // invalid edge
	}
#else
	bool GetEdgeOrientation(FIndex idxFace, VIndex iV0, VIndex iV1) const;
#endif
	FIndex GetEdgeAdjacentFace(FIndex idxFace, VIndex iV0, VIndex iV1) const;
	unsigned FixNonManifold(float magDisplacementDuplicateVertices=0.01f, VertexIdxArr* duplicatedVertices=NULL);
#else
	bool FixNonManifold();
#endif
	void Clean(float fDecimate=0.7f, float fSpurious=10.f, bool bRemoveSpikes=true, unsigned nCloseHoles=30, unsigned nSmoothMesh=2, float fEdgeLength=0, bool bLastClean=true, float fDecimateError=0);

	void EnsureEdgeSize(float minEdge=-0.5f, float maxEdge=-4.f, float collapseRatio=0.2, float degenerate_angle_deg=150, int mode=1, int max_iters=50);

	typedef cList<uint16_t,uint16_t,0,16,FIndex> AreaArr;
	void Subdivide(const AreaArr& maxAreas, uint32_t maxArea);
	void Decimate(VertexIdxArr& verticesRemove);
	void CloseHole(VertexIdxArr& vertsLoop);
	void CloseHoleQuality(VertexIdxArr& vertsLoop);
	void RemoveFacesOutside(const OBB3f&);
	void RemoveFaces(FaceIdxArr& facesRemove, bool bUpdateLists=false);
	void RemoveVertices(VertexIdxArr& vertexRemove, bool bUpdateLists=false);
	VIndex RemoveUnreferencedVertices(bool bUpdateLists=false);
	void ConvertTexturePerVertex(Mesh&) const;

	void FaceTexcoordsNormalize(TexCoordArr& newFaceTexcoords, bool flipY=true) const;
	void FaceTexcoordsUnnormalize(TexCoordArr& newFaceTexcoords, bool flipY=true) const;

	inline Normal FaceNormal(const Face& f) const {
		return ComputeTriangleNormal(vertices[f[0]], vertices[f[1]], vertices[f[2]]);
	}
	inline Normal VertexNormal(VIndex idxV) const {
		ASSERT(vertices.GetSize() == vertexFaces.GetSize());
		const FaceIdxArr& vf = vertexFaces[idxV];
		ASSERT(!vf.IsEmpty());
		Normal n(Normal::ZERO);
		FOREACHPTR(pIdxF, vf)
			n += normalized(FaceNormal(faces[*pIdxF]));
		return n;
	}

	Planef EstimateGroundPlane(const ImageArr& images, float sampleMesh=0, float planeThreshold=0, const String& fileExportPlane="") const;

	Vertex ComputeCentroid(FIndex) const;
	Type ComputeArea(FIndex) const;
	REAL ComputeArea() const;
	REAL ComputeVolume() const;

	void SamplePoints(unsigned numberOfPoints, PointCloud&) const;
	void SamplePoints(REAL samplingDensity, PointCloud&) const;
	void SamplePoints(REAL samplingDensity, unsigned mumPointsTheoretic, PointCloud&) const;

	void Project(const Camera& camera, DepthMap& depthMap) const;
	void Project(const Camera& camera, DepthMap& depthMap, Image8U3& image) const;
	void Project(const Camera& camera, DepthMap& depthMap, NormalMap& normalMap) const;
	void ProjectOrtho(const Camera& camera, DepthMap& depthMap) const;
	void ProjectOrtho(const Camera& camera, DepthMap& depthMap, Image8U3& image) const;
	void ProjectOrthoTopDown(unsigned resolution, Image8U3& image, Image8U& mask, Point3& center) const;

	bool Split(FacesChunkArr&, float maxArea);
	Mesh SubMesh(const FaceIdxArr& faces) const;

	bool TransferTexture(Mesh& mesh, unsigned textureSize=1024);

	// file IO

	void ValidateFaceFaces() const;
	void ValidateEdgeConsistency() const;
	void ValidateVertexFacesSorted() const;

	bool Load(const String& fileName);
	bool Save(const String& fileName, const cList<String>& comments=cList<String>(), bool bBinary=true) const;
	bool Save(const FacesChunkArr&, const String& fileName, const cList<String>& comments=cList<String>(), bool bBinary=true) const;
	static bool Save(const VertexArr& vertices, const String& fileName, bool bBinary=true);

	static inline uint32_t FindVertex(const Face& f, VIndex v) { for (uint32_t i=0; i<3; ++i) if (f[i] == v) return i; return NO_ID; }
	static inline VIndex GetVertex(const Face& f, VIndex v) { const uint32_t idx(FindVertex(f, v)); ASSERT(idx != NO_ID); return f[idx]; }
	static inline VIndex& GetVertex(Face& f, VIndex v) { const uint32_t idx(FindVertex(f, v)); ASSERT(idx != NO_ID); return f[idx]; }

protected:
	bool LoadPLY(const String& fileName);
	bool LoadOBJ(const String& fileName);
	bool LoadGLTF(const String& fileName, bool bBinary=true);

	bool SavePLY(const String& fileName, const cList<String>& comments=cList<String>(), bool bBinary=true) const;
	bool SaveOBJ(const String& fileName) const;
	bool SaveGLTF(const String& fileName, bool bBinary=true) const;
	// Global Mapper native binary mesh (.gmmesh) — per-vertex UVs, fast read
	bool SaveGMMesh(const String& fileName) const;

	#ifdef _USE_CUDA
	static bool InitKernels(int device=-1);
	#endif

	#ifdef _USE_BOOST
	// implement BOOST serialization
	friend class boost::serialization::access;
#if 1

	template <typename T>
	struct is_cList : std::false_type {};

	template <typename T, typename C, int a, int b, typename I>
	struct is_cList<SEACAVE::cList<T, C, a, b, I>> : std::true_type {};

	// Portable accessor: works everywhere
	template <typename T>
	struct is_cList_v_helper {
		static const bool value = is_cList<T>::value;
	};

	// Use like a function-style macro for simplicity
#define is_cList_v(T) (is_cList_v_helper<T>::value)

	// ================================================================
	// serializeCList: handles cList<T> and nested cList<cList<T>>
	// ================================================================
	template <class Archive, typename T>
	static void serializeCList(Archive& ar, T& clist)
	{
		size_t n = clist.size();
		ar& n;
		if constexpr (Archive::is_loading::value)
			clist.resize(n);
		if (n == 0) return;

		using Elem = typename std::remove_reference<decltype(clist[0])>::type;

		if constexpr (std::is_trivially_copyable<Elem>::value) {
			if constexpr (Archive::is_saving::value)
				ar.save_binary(clist.data(), n * sizeof(Elem));
			else
				ar.load_binary(clist.data(), n * sizeof(Elem));
		}
		else if constexpr (is_cList_v(Elem)) {
			// recursive nested cList<cList<...>> path
			for (size_t i = 0; i < n; ++i)
				serializeCList(ar, clist[i]);
		}
		else {
			// fallback for non-trivial types
			for (size_t i = 0; i < n; ++i)
				ar& clist[i];
		}
	}

	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		serializeCList(ar, vertices);
		serializeCList(ar, faces);
		serializeCList(ar, vertexNormals);
		serializeCList(ar, vertexVertices);
		serializeCList(ar, vertexFaces);
		serializeCList(ar, vertexBoundary);
		serializeCList(ar, faceNormals);
		serializeCList(ar, faceTexcoords);
		ar& textureDiffuse;
	}
#else
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ar & vertices;
		ar & faces;
		ar & vertexNormals;
		ar & vertexVertices;
		ar & vertexFaces;
		ar & vertexBoundary;
		ar & faceNormals;
		ar & faceTexcoords;
		ar & textureDiffuse;
	}
#endif
	#endif
};
/*----------------------------------------------------------------*/
#include <malloc.h>
#include <new>
#include <vector>

template <typename T, size_t Align = 16>
struct AlignedAllocator
{
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = size_t;
	using difference_type = ptrdiff_t;

	AlignedAllocator() noexcept = default;

	template <class U> AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

	pointer allocate(size_type n)
	{
		if (n > static_cast<size_type>(-1) / sizeof(T))
			throw std::bad_array_new_length();

#if defined(_MSC_VER)
		void* p = _aligned_malloc(n * sizeof(T), Align);
		if (!p) throw std::bad_alloc();
#else
		void* p = nullptr;
		if (posix_memalign(&p, Align, n * sizeof(T)))
			throw std::bad_alloc();
#endif
		return static_cast<pointer>(p);
	}

	void deallocate(pointer p, size_type) noexcept
	{
#if defined(_MSC_VER)
		_aligned_free(p);
#else
		free(p);
#endif
	}

	template <class U>
	struct rebind { using other = AlignedAllocator<U, Align>; };
};

template <class T, class U, size_t A>
inline bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }

template <class T, class U, size_t A>
inline bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }


// used to render a 3D triangle
template <typename DERIVED>
struct TRasterMeshBase {
	typedef DERIVED Rasterizer;

	struct Triangle {
		Point3 ptc[3];
		Point2f pti[3];
	};

	const Camera& camera;
	DepthMap& depthMap;

	TRasterMeshBase(const Camera& _camera, DepthMap& _depthMap)
		: camera(_camera), depthMap(_depthMap) {
	}

	inline void Clear() {
		depthMap.memset(0);
	}
	inline cv::Size Size() const {
		return depthMap.size();
	}

	inline bool ProjectVertex(const Point3f& pt, int v, Triangle& t) {
		return (t.ptc[v] = camera.TransformPointW2C(Cast<REAL>(pt))).z > 0 &&
			depthMap.isInsideWithBorder<float, 3>(t.pti[v] = camera.TransformPointC2I(t.ptc[v]));
	}

	inline Point3f PerspectiveCorrectBarycentricCoordinates(const Triangle& t, const Point3f& bary) {
		return SEACAVE::PerspectiveCorrectBarycentricCoordinates(bary, (float)t.ptc[0].z, (float)t.ptc[1].z, (float)t.ptc[2].z);
	}
	inline float ComputeDepth(const Triangle& t, const Point3f& pbary) {
		return pbary[0] * (float)t.ptc[0].z + pbary[1] * (float)t.ptc[1].z + pbary[2] * (float)t.ptc[2].z;
	}
	void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
		const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
		const Depth z(ComputeDepth(t, pbary));
		ASSERT(z > Depth(0));

		Depth& depth = depthMap(pt);
		if (depth == 0 || depth > z) {
			depth = z;
		}
	}

	struct TriangleRasterizer {
		Triangle& triangle;
		Rasterizer& rasterizer;
		TriangleRasterizer(Triangle& t, Rasterizer& r) : triangle(t), rasterizer(r) {}
		inline cv::Size Size() const {
			return rasterizer.Size();
		}
		inline void operator()(const ImageRef& pt, const Point3f& bary) const {
			rasterizer.Raster(pt, triangle, bary);
		}
	};
	inline TriangleRasterizer CreateTriangleRasterizer(Triangle& triangle) {
		return TriangleRasterizer(triangle, *static_cast<DERIVED*>(this));
	}
};

// used to render a mesh
template <typename DERIVED>
struct TRasterMesh : TRasterMeshBase<DERIVED> {
	typedef TRasterMeshBase<DERIVED> Base;
	using typename Base::Triangle;

	using Base::camera;
	using Base::depthMap;

	const Mesh::VertexArr& vertices;

	TRasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap)
		: Base(_camera, _depthMap), vertices(_vertices) {}

	template <typename TriangleRasterizer>
	void Project(const Mesh::Face& facet, TriangleRasterizer& tr) {
		// project face vertices to image plane
		for (int v=0; v<3; ++v) {
			// skip face if not completely inside
			if (!static_cast<DERIVED*>(this)->ProjectVertex(vertices[facet[v]], v, tr.triangle))
				return;
		}
		// draw triangle
		Image8U3::RasterizeTriangleBary(tr.triangle.pti[0], tr.triangle.pti[1], tr.triangle.pti[2], tr);
	}
	void Project(const Mesh::Face& facet) {
		Triangle triangle;
		Project(facet, this->CreateTriangleRasterizer(triangle));
	}
};

bool TestMeshProjectionMT(const Mesh& mesh, const Image& image);
/*----------------------------------------------------------------*/


struct IntersectRayMesh {
	typedef Mesh::Octree Octree;
	typedef typename Octree::IDX_TYPE IDX;

	const Mesh& mesh;
	const Ray3& ray;
	IndexDist pick;

	IntersectRayMesh(const Octree& octree, const Ray3& _ray, const Mesh& _mesh)
		: mesh(_mesh), ray(_ray)
	{
		octree.Collect(*this, *this);
	}

	inline bool Intersects(const typename Octree::POINT_TYPE& center, typename Octree::Type radius) const {
		return ray.Intersects(AABB3f(center, radius));
	}

	void operator() (const IDX* idices, IDX size) {
		// store all intersected faces only once
		typedef std::unordered_set<Mesh::FIndex> FaceSet;
		FaceSet set;
		FOREACHRAWPTR(pIdx, idices, size) {
			const Mesh::VIndex idxVertex((Mesh::VIndex)*pIdx);
			const Mesh::FaceIdxArr& faces = mesh.vertexFaces[idxVertex];
			set.insert(faces.begin(), faces.end());
		}
		// test face intersection and keep the closest
		for (Mesh::FIndex idxFace : set) {
			const Mesh::Face& face = mesh.faces[idxFace];
			REAL dist;
			if (ray.Intersects<true>(Triangle3(Cast<REAL>(mesh.vertices[face[0]]), Cast<REAL>(mesh.vertices[face[1]]), Cast<REAL>(mesh.vertices[face[2]])), &dist)) {
				ASSERT(dist >= 0);
				if (pick.dist > dist) {
					pick.dist = dist;
					pick.idx = idxFace;
				}
			}
		}
	}
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_MESH_H_
