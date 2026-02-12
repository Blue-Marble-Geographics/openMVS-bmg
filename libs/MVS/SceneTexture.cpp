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

#include <boost/graph/filtered_graph.hpp>
// connected components
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using namespace MVS; 

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define TEXOPT_USE_OPENMP
#endif

#define FASTER_OUTLIER_DETECTION
#undef COUNT_ITERATIONS

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

// inference algorithm
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
#include "../Math/LBP.h"
namespace MVS {
typedef LBPInference::NodeID NodeID;
// Potts model as smoothness function
LBPInference::EnergyType STCALL SmoothnessPotts(LBPInference::NodeID, LBPInference::NodeID, LBPInference::LabelID l1, LBPInference::LabelID l2) {
	return l1 == l2 && l1 != 0 && l2 != 0 ? LBPInference::EnergyType(0) : LBPInference::EnergyType(LBPInference::MaxEnergy);
}
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
typedef Eigen::SparseMatrix<float,Eigen::ColMajor,MatIdx> SparseMat;

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
	};
	typedef cList<TexturePatch,const TexturePatch&,1,1024,FIndex> TexturePatchArr;

	// used to optimize texture patches
	struct SeamVertex {
		struct Patch {
			struct Edge {
				uint32_t idxSeamVertex; // the other vertex of this edge
				FIndex idxFace; // the face containing this edge in this patch

				inline Edge() {}
				inline Edge(uint32_t _idxSeamVertex) : idxSeamVertex(_idxSeamVertex) {}
				inline bool operator == (uint32_t _idxSeamVertex) const {
					return (idxSeamVertex == _idxSeamVertex);
				}
			};
			typedef cList<Edge,const Edge&,0,4,uint32_t> Edges;

			uint32_t idxPatch; // the patch containing this vertex
			Point2f proj; // the projection of this vertex in this patch
			Edges edges; // the edges starting from this vertex, contained in this patch (exactly two for manifold meshes)

			inline Patch() {}
			inline Patch(uint32_t _idxPatch) : idxPatch(_idxPatch) {}
			inline bool operator == (uint32_t _idxPatch) const {
				return (idxPatch == _idxPatch);
			}
		};
		typedef cList<Patch,const Patch&,1,4,uint32_t> Patches;
		struct PatchEntry 
		{
			PatchEntry() {}
			PatchEntry(uint32_t _patch, uint16_t _idx) : patch(_patch), idx(_idx) {}
			uint32_t patch;
			uint16_t idx;
		};
		struct PatchContainer
		{
			void clear() { storage.clear(); }
			void reserve(uint32_t cnt) { storage.reserve(cnt); }
			void Insert(uint32_t patch, uint16_t idx) {
				for (auto& i : storage) {
					if (i.patch == patch) {
						i.idx = idx;
						return;
					}
				}
				storage.emplace_back(patch, idx);
			}

			uint16_t At(uint32_t patch) const
			{
				const uint16_t* p = Find(patch);
				HARD_ASSERT(p != nullptr);
				return *p;
			}

			const uint16_t* Find(uint32_t patch) const
			{
				for (const auto& i : storage) {
					if (i.patch == patch) {
						return &i.idx;
					}
				}
				return nullptr;
			}

      std::vector<PatchEntry> storage;
		};

		PatchContainer patchIndexLookup;

		VIndex idxVertex; // the index of this vertex
		Patches patches; // the patches meeting at this vertex (two or more)

		// Make it move friendly
		SeamVertex() = default;

		inline SeamVertex(uint32_t _idxVertex) : idxVertex(_idxVertex) {}
		inline bool operator == (uint32_t _idxVertex) const {
			return (idxVertex == _idxVertex);
		}
		Patch& GetPatch(uint32_t idxPatch) {
			const uint32_t idx(patches.Find(idxPatch));
			if (idx == NO_ID)
				return patches.emplace_back(idxPatch);
			return patches[idx];
		}
		inline void SortByPatchIndex(IndexArr& indices) const {
			indices.Resize(patches.GetSize());
			std::iota(indices.Begin(), indices.End(), 0);
			std::sort(indices.Begin(), indices.End(), [&](IndexArr::Type i0, IndexArr::Type i1) -> bool {
				return patches[i0].idxPatch < patches[i1].idxPatch;
			});
		}
	};
	typedef cList<SeamVertex,const SeamVertex&,1,256,uint32_t> SeamVertices;

	// used to iterate vertex labels
	struct PatchIndex {
		bool bIndex;
		union {
			uint32_t idxPatch;
			uint32_t idxSeamVertex;
		};
	};
	typedef CLISTDEF0(PatchIndex) PatchIndices;
	struct VertexPatchIterator {
		uint32_t idx;
		uint32_t idxPatch;
		const SeamVertex::Patches* pPatches;
		inline VertexPatchIterator(const PatchIndex& patchIndex, const SeamVertices& seamVertices) : idx(NO_ID) {
			if (patchIndex.bIndex) {
				pPatches = &seamVertices[patchIndex.idxSeamVertex].patches;
			} else {
				idxPatch = patchIndex.idxPatch;
				pPatches = NULL;
			}
		}
		inline operator uint32_t () const {
			return idxPatch;
		}
		inline bool Next() {
			if (pPatches == NULL)
				return (idx++ == NO_ID);
			if (++idx >= pPatches->GetSize())
				return false;
			idxPatch = (*pPatches)[idx].idxPatch;
			return true;
		}
	};

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

	bool FaceViewSelection(unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness, const IIndexArr& views);
	
	void CreateSeamVertices();
	void GlobalSeamLeveling();
	void LocalSeamLeveling();
	void GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize);

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
	static void ProcessMask(Image8U& mask, int stripWidth);
	static void PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias=1.f);


public:
	const unsigned nResolutionLevel; // how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // how many times to scale down the images before mesh optimization

	// store found texture patches
	TexturePatchArr texturePatches;

	// used to compute the seam leveling
	PairIdxArr seamEdges; // the (face-face) edges connecting different texture patches
	Mesh::FaceIdxArr components; // for each face, stores the texture patch index to which belongs
	IndexArr mapIdxPatch; // remap texture patch indices after invalid patches removal
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
bool MeshTexture::ListCameraFaces(FaceDataViewArr& facesDatas, float fOutlierThreshold, const IIndexArr& _views)
{
	// create faces octree
	Mesh::Octree octree;
	Mesh::FacesInserter::CreateOctree(octree, scene.mesh);

	// extract array of faces viewed by each image
	IIndexArr views(_views);
	if (views.empty()) {
		views.resize(images.size());
		std::iota(views.begin(), views.end(), IIndex(0));
	}
	facesDatas.Resize(faces.size());
	const size_t cap = std::min<size_t>(views.size(), 8);
#pragma omp for schedule(static)
	for (int_t idx = 0; idx < (int_t)facesDatas.size(); ++idx) {
		facesDatas[idx].Reserve((uint32_t)cap);
	}

	Util::Progress progress(_T("Initialized views"), views.size());
	typedef float real;
	TImage<real> imageGradMag;
	TImage<real>::EMat mGrad[2];

#ifdef TEXOPT_USE_OPENMP
	bool bAbort(false);
	const int numThreads = omp_get_max_threads();

	struct FaceAccum {
		real quality;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		uint32_t c[3];
		uint32_t cnt;
#endif
	};

	struct PixelXY {
		uint16_t x;
		uint16_t y;
	};

	static thread_local std::vector<FaceAccum> gLocalAccum;
	static thread_local std::vector<uint16_t> gLocalGen;
	static thread_local uint16_t gLocalCurGen = 1;
	static thread_local std::vector<FIndex> gLocalTouched;
	static thread_local std::vector<uint16_t> gPixelGen;
	static thread_local uint16_t gPixelCurGen = 1;
	static thread_local std::vector<PixelXY> gTouchedPixels;
	static thread_local FaceMap faceMap;
	static thread_local DepthMap depthMap;

	struct FaceDataAndView {
		FIndex idxFace;   // which face this belongs to
		IIndex idxView;   // which camera/view
		float quality;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		float color[3];
#endif
	};
	static thread_local std::vector<FaceDataAndView> publish;

	// Since we are controlling the threading per-view, don't let cv thread
	// when performing its work.
	cv::setNumThreads(1); // No threading.  0 is not "no threading".

	// It is not faster to restrict the threading here.
#pragma omp parallel private(imageGradMag, mGrad)
{
#pragma omp for schedule(static)
	for (int_t idx = 0; idx < (int_t)views.size(); ++idx) {
#pragma omp flush (bAbort)
		if (bAbort) {
			++progress;
			continue;
		}

		const size_t numFaces = faces.GetSize();
		const size_t expectedVerts = numFaces * 2;

		static thread_local	CameraRenderData crd;

		crd.globalVert.clear();
		crd.faces.clear();
		crd.globalFace.clear();
		crd.verts.clear();

		crd.globalVert.reserve(expectedVerts);
		crd.verts.reserve(expectedVerts);
		crd.faces.reserve(numFaces);
		crd.globalFace.reserve(numFaces);

		if (gLocalAccum.size() != numFaces) {
			gLocalAccum.resize(numFaces);
			gLocalGen.assign(numFaces, 0);
			gLocalCurGen = 1;
		}
		++gLocalCurGen;
		if (gLocalCurGen == 0) {
			std::fill(gLocalGen.begin(), gLocalGen.end(), 0);
			gLocalCurGen = 1;
		}
		gLocalTouched.clear();

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

		// Libjpg is taking the image, decompressing it, and storing in R[0], G[1], B[2] order.
		// This is expensive.
		// The rest of the pipeline is expecting this and changing it isn't trivial.

		// 1) RGB -> gray (8-bit)
		cv::Mat gray8;
		cv::cvtColor(imageData.image, gray8, cv::COLOR_RGB2GRAY); // This is wrong in the original source.

		// 2) Downsample in 8-bit (fast + correct)
		cv::Mat graySmall8;
		cv::resize(
			gray8,
			graySmall8,
			cv::Size(),
			0.5,
			0.5,
			cv::INTER_AREA
		);

		// 3) Convert once to float + normalize
		graySmall8.convertTo(imageGradMag, CV_32F, 1.0f / 255.0f);

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
		(TImage<real>::EMatMap)imageGradMag = (mGrad[0].cwiseAbs2() + mGrad[1].cwiseAbs2()); // Drop the sqrt (won't affect results) .cwiseSqrt();
		// apply some blur on the gradient to lower noise/glossiness effects onto face-quality score
		cv::GaussianBlur(imageGradMag, imageGradMag, cv::Size(15, 15), 0, 0, cv::BORDER_DEFAULT);
		// select faces inside view frustum
		Mesh::FaceIdxArr cameraFaces;
		Mesh::FacesInserter inserter(cameraFaces);
		typedef TFrustum<float, 5> Frustum;
		const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
		octree.Traverse(frustum, inserter);

		PreprocessCameraFaces(
			cameraFaces,
			faces,
			scene.mesh.vertices,
			crd
		);

		UpdateCameraVertsAndNormals(
			scene.mesh.vertices,
			imageData.camera,
			crd
		);

		// project all triangles in this view and keep the closest ones
		const int width = imageData.width;
		const int height = imageData.height;
		const size_t numPixels = size_t(width) * size_t(height);

		// ensure pixel-gen buffer exists
		if (gPixelGen.size() != numPixels) {
			gPixelGen.assign(numPixels, 0);
			gPixelCurGen = 1;
		}

		// advance generation for THIS view
		++gPixelCurGen;
		if (gPixelCurGen == 0) {
			std::fill(gPixelGen.begin(), gPixelGen.end(), 0);
			gPixelCurGen = 1;
		}
		gTouchedPixels.clear();

		// init view data
		// Both maps must be completely filled or have a generator.

		depthMap.create(height, width);
		//std::fill(depthMap.begin(), depthMap.end(), -std::numeric_limits<float>::infinity());
		__stosd((PDWORD)&*depthMap.begin(),
			0xFF800000u,
			numPixels);

		faceMap.create(height, width);
		faceMap.memset(0xFF); // Essentially NO_ID

		struct Triangle {
			Point2f pti[3];
		} t;

		const float minX = 3.f;
		const float minY = 3.f;
		const float maxX = (float)(width - 4);
		const float maxY = (float)(height - 4);

		const float widthMinus1 = (float)(width - 1);
		const float heightMinus1 = (float)(height - 1);

		for (size_t fi = 0, cnt = crd.faces.size(); fi < cnt; ++fi) {
			const Face& face = crd.faces[fi];
			// ==== Camera-space vertices (pre-transformed) ====
			const CamVert& c0 = crd.verts[face[0]];
			const CamVert& c1 = crd.verts[face[1]];
			const CamVert& c2 = crd.verts[face[2]];

			{
				// ==== Perspective divide ====
				float u0 = c0.x * c0.invZ;
				float v0i = c0.y * c0.invZ;

				float u1 = c1.x * c1.invZ;
				float v1i = c1.y * c1.invZ;

				float u2 = c2.x * c2.invZ;
				float v2i = c2.y * c2.invZ;

				// ==== Scalar bounds check (simd not needed anymore) ====
				if (u0 < minX || u0 > maxX ||
					u1 < minX || u1 > maxX ||
					u2 < minX || u2 > maxX ||
					v0i < minY || v0i > maxY ||
					v1i < minY || v1i > maxY ||
					v2i < minY || v2i > maxY)
					continue;

				// ==== Store results (same as your SSE path) ====
				t.pti[0] = { u0, v0i };
				t.pti[1] = { u1, v1i };
				t.pti[2] = { u2, v2i };
			}

			// draw triangle
			const auto& v1 = t.pti[0];
			const auto& v2 = t.pti[1];
			const auto& v3 = t.pti[2];

			// ignore back oriented triangles (negative area)
			// flip winding to match OpenMVS screen-space convention
			const float area = EdgeFunction2(v1, v2, v3);
			if (area >= 0.f) {
				continue;
			}

			// compute bounding-box fully containing the triangle
			float boxMinX = v1.x;
			float boxMinY = v1.y;
			float boxMaxX = v1.x;
			float boxMaxY = v1.y;

			if (v2.x < boxMinX) boxMinX = v2.x;
			if (v3.x < boxMinX) boxMinX = v3.x;
			if (v2.y < boxMinY) boxMinY = v2.y;
			if (v3.y < boxMinY) boxMinY = v3.y;

			if (v2.x > boxMaxX) boxMaxX = v2.x;
			if (v3.x > boxMaxX) boxMaxX = v3.x;
			if (v2.y > boxMaxY) boxMaxY = v2.y;
			if (v3.y > boxMaxY) boxMaxY = v3.y;

			// ---- Quick reject: fully outside screen ----
			// This is the minimal correct check
			if (boxMaxX < 0.0f || boxMinX > widthMinus1 ||
				boxMaxY < 0.0f || boxMinY > heightMinus1)
				continue;

			// -----------------------------------------------
			// Bounding box (float -> int, half-open)
			// -----------------------------------------------
			int minXi = (int)std::floor(boxMinX);
			int minYi = (int)std::floor(boxMinY);
			int maxXi = (int)std::ceil(boxMaxX);
			int maxYi = (int)std::ceil(boxMaxY);

			if (minXi < 0) minXi = 0;
			if (minYi < 0) minYi = 0;
			if (maxXi > width)  maxXi = width;
			if (maxYi > height) maxYi = height;

			const int x0 = minXi;
			const int y0 = minYi;
			const int x1 = maxXi - 1;
			const int y1 = maxYi - 1;

			if (x0 > x1 || y0 > y1)
				continue;

			// -----------------------------------------------
			// Fixed-point setup
			// -----------------------------------------------
			constexpr int FP_BITS = 4;
			constexpr int FP_SCALE = 1 << FP_BITS;

			// vertices in fixed-point
			const int x0i = int(v1.x * FP_SCALE);
			const int y0i = int(v1.y * FP_SCALE);
			const int x1i = int(v2.x * FP_SCALE);
			const int y1i = int(v2.y * FP_SCALE);
			const int x2i = int(v3.x * FP_SCALE);
			const int y2i = int(v3.y * FP_SCALE);

			// -----------------------------------------------
			// Integer edge deltas
			// -----------------------------------------------
			const int e0_dx = y1i - y2i;
			const int e0_dy = x2i - x1i;

			const int e1_dx = y2i - y0i;
			const int e1_dy = x0i - x2i;

			const int e2_dx = y0i - y1i;
			const int e2_dy = x1i - x0i;

			// edge constants (64-bit!)
			const int64_t e0_c = int64_t(x1i) * y2i - int64_t(x2i) * y1i;
			const int64_t e1_c = int64_t(x2i) * y0i - int64_t(x0i) * y2i;
			const int64_t e2_c = int64_t(x0i) * y1i - int64_t(x1i) * y0i;

			// -----------------------------------------------
			// Top-left rule bias (integer)
			// -----------------------------------------------
			auto TopLeftBias = [](int dx, int dy) -> int64_t {
				return (dx > 0 || (dx == 0 && dy < 0)) ? 0 : -1;
				};

			const int64_t bias0 = TopLeftBias(e0_dx, e0_dy);
			const int64_t bias1 = TopLeftBias(e1_dx, e1_dy);
			const int64_t bias2 = TopLeftBias(e2_dx, e2_dy);

			// -----------------------------------------------
			// Pixel-center start (important)
			// -----------------------------------------------
			const int64_t x0p = (int64_t(x0) << FP_BITS) + (FP_SCALE >> 1);
			const int64_t y0p = (int64_t(y0) << FP_BITS) + (FP_SCALE >> 1);

			// row start edge values
			int64_t e0_row = int64_t(e0_dx) * x0p + int64_t(e0_dy) * y0p + e0_c + bias0;
			int64_t e1_row = int64_t(e1_dx) * x0p + int64_t(e1_dy) * y0p + e1_c + bias1;
			int64_t e2_row = int64_t(e2_dx) * x0p + int64_t(e2_dy) * y0p + e2_c + bias2;

			// -----------------------------------------------
			// Fixed-point step amounts
			// -----------------------------------------------
			int64_t e0_stepX = int64_t(e0_dx) * FP_SCALE;
			int64_t e1_stepX = int64_t(e1_dx) * FP_SCALE;
			int64_t e2_stepX = int64_t(e2_dx) * FP_SCALE;

			int64_t e0_stepY = int64_t(e0_dy) * FP_SCALE;
			int64_t e1_stepY = int64_t(e1_dy) * FP_SCALE;
			int64_t e2_stepY = int64_t(e2_dy) * FP_SCALE;

			// -----------------------------------------------
			// Area in same scale (FP_SCALE^2)
			// -----------------------------------------------
			const int64_t ax = int64_t(x1i) - int64_t(x0i);
			const int64_t ay = int64_t(y1i) - int64_t(y0i);
			const int64_t bx = int64_t(x2i) - int64_t(x0i);
			const int64_t by = int64_t(y2i) - int64_t(y0i);

			int64_t areaScaled = ax * by - ay * bx;
			if (areaScaled == 0)
				continue;

			if (areaScaled < 0) {
				areaScaled = -areaScaled;

				e0_row = -e0_row;
				e1_row = -e1_row;
				e2_row = -e2_row;

				e0_stepX = -e0_stepX;
				e1_stepX = -e1_stepX;
				e2_stepX = -e2_stepX;

				e0_stepY = -e0_stepY;
				e1_stepY = -e1_stepY;
				e2_stepY = -e2_stepY;
			}
			const float invAreaScaled = 1.0f / float(areaScaled);

			// -----------------------------------------------
			// Depth plane (consistent with integer edges)
			// -----------------------------------------------
			const float iz0 = c0.invZ;
			const float iz1 = c1.invZ;
			const float iz2 = c2.invZ;

			const float dz_dx =
				(float(e0_stepX) * iz0 +
					float(e1_stepX) * iz1 +
					float(e2_stepX) * iz2) * invAreaScaled;

			const float dz_dy =
				(float(e0_stepY) * iz0 +
					float(e1_stepY) * iz1 +
					float(e2_stepY) * iz2) * invAreaScaled;

			float invZ_row =
				(float(e0_row) * iz0 +
					float(e1_row) * iz1 +
					float(e2_row) * iz2) * invAreaScaled;

			// -----------------------------------------------
			// Raster loop
			// -----------------------------------------------
			Depth* __restrict depthPtr = depthMap.ptr<float>(0);
			cuint32_t* __restrict facePtr = faceMap.ptr<cuint32_t>(0);

			constexpr int64_t EDGE_THRESH = 1 * FP_SCALE; // conservative

			// Bias edges
			e0_row -= EDGE_THRESH;
			e1_row -= EDGE_THRESH;
			e2_row -= EDGE_THRESH;

			FaceAccum* __restrict accum = gLocalAccum.data();
			uint16_t* __restrict faceGen = gLocalGen.data();

			// Unrolled is not faster.
			for (int y = y0; y <= y1; ++y) {
				const size_t base = size_t(y) * width + x0;

				Depth* __restrict depthRow = depthPtr + base;
				cuint32_t* __restrict faceRow = facePtr + base;
				uint16_t* __restrict genRow = gPixelGen.data() + base;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				uint8_t* __restrict  pImageBase = imageData.image.ptr<uint8_t>(y);
#endif

				int64_t e0 = e0_row;
				int64_t e1 = e1_row;
				int64_t e2 = e2_row;
				float   invZ = invZ_row;

				uint32_t pix = (uint32_t)base;
				for (int x = x0; x <= x1; ++x) {
					if ((e0 | e1 | e2) < 0) {
						if (invZ > *depthRow) {
							*depthRow = invZ;
							uint32_t faceId = (cuint32_t)crd.globalFace[fi];

							// Determine which pixels correspond to which faces, and accumulate quality/color for each face.

							FaceAccum& acc = accum[faceId];
							uint16_t& gen = faceGen[faceId];

							const real q = imageGradMag.ptr<real>(y >> 1)[x >> 1];

#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
							const uint8_t* __restrict p = pImageBase + 3 * x;
#endif

							if (gen == gLocalCurGen) {
								acc.quality += q;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
								acc.c[0] += p[0];
								acc.c[1] += p[1];
								acc.c[2] += p[2];
								++acc.cnt;
#endif
							}
							else {
								gen = gLocalCurGen;
								acc.quality = q;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
								acc.c[0] = p[0];
								acc.c[1] = p[1];
								acc.c[2] = p[2];
								acc.cnt = 1;
#endif
								gLocalTouched.push_back(faceId);
							}

						}
					}

					++pix;

					e0 += e0_stepX;
					e1 += e1_stepX;
					e2 += e2_stepX;
					invZ += dz_dx;

					++depthRow;
					++faceRow;
					++genRow;
				}

				e0_row += e0_stepY;
				e1_row += e1_stepY;
				e2_row += e2_stepY;
				invZ_row += dz_dy;
			}
		}

		const int cols = faceMap.cols;
		const int rows = faceMap.rows;

		publish.clear();
		publish.reserve(gLocalTouched.size());

		for (FIndex idxFace : gLocalTouched) {
			const FaceAccum& src = gLocalAccum[idxFace];

			const Face& f = faces[idxFace];

			const Vertex faceCenter =
				(vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) * (1.0f / 3.0f);

			Point3f camDir =
				Cast<Mesh::Type>(imageData.camera.C) - faceCenter;

			const Normal& faceNormal = scene.mesh.faceNormals[idxFace];
	
			float invLen = 1.0f / FastSqrtS(camDir.dot(camDir));
			camDir *= invLen;

			float cosFaceCam = camDir.dot(faceNormal);
			if (cosFaceCam <= 0.0f)
				continue;

			// clamp grazing angles
			cosFaceCam = std::max(cosFaceCam, 0.2f);
			float angleWeight = cosFaceCam * cosFaceCam; // cos^2
			
			FaceDataAndView& fd = publish.emplace_back();
			fd.idxFace = idxFace;
			fd.idxView = idxView;
			fd.quality = src.quality * angleWeight;
			fd.quality = fd.quality / (1.0f + 0.1f * fd.quality);

#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
			const float invArea = 1.0f / float(src.cnt);

			fd.color[0] =
				(0.299f * src.c[0] +
					0.587f * src.c[1] +
					0.114f * src.c[2]) * invArea;

			fd.color[1] =
				(-0.168736f * src.c[0] -
					0.331264f * src.c[1] +
					0.5f * src.c[2]) * invArea + 128.0f;

			fd.color[2] =
				(0.5f * src.c[0] -
					0.418688f * src.c[1] -
					0.081312f * src.c[2]) * invArea + 128.0f;
#endif
		}

		// Notice, we have minimized the work inside the critical.
#ifdef TEXOPT_USE_OPENMP
#pragma omp critical
#endif
		{
			for (const FaceDataAndView& e : publish) {
				FaceDataArr& fd = facesDatas[e.idxFace];
				FaceData& dst = fd.AddEmpty();
				dst.color[0] = e.color[0];
				dst.color[1] = e.color[1];
				dst.color[2] = e.color[2];
				dst.idxView = e.idxView;
				dst.quality = e.quality;
			}
		}

		++progress;
	} // per view
}

	progress.process();
	progress.close();

	// Restore cv's ability to thread.
	cv::setNumThreads(-1);

#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		// try to detect outlier views for each face
		// (views for which the face is occluded by a dynamic object in the scene, ex. pedestrians)

#pragma omp parallel for schedule(static)
		for (int f = 0; f < (int)facesDatas.size(); ++f) {
			FaceOutlierDetection(facesDatas[f], fOutlierThreshold);
		}
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
	// reject all views whose gauss value is below this threshold
	if (thOutlier <= 0)
		thOutlier = 6e-2f;

	const float minCovariance(1e-3f); // if all covariances drop below this the outlier detection aborted

	const unsigned maxIterations(10);
	const unsigned minInliers(4);

	// init colors array
	if (faceDatas.GetSize() <= minInliers)
		return false;
	Eigen::Matrix3Xd colorsAll(3, faceDatas.GetSize());
	BoolArr inliers(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		colorsAll.col(i) = ((const Color::EVec)faceDatas[i].color).cast<double>();
		inliers[i] = true;
	}

	// perform outlier removal; abort if something goes wrong
	// (number of inliers below threshold or can not invert the covariance)
#ifdef FASTER_OUTLIER_DETECTION
	size_t numInliers = faceDatas.GetSize();
	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	Eigen::Matrix3d covarianceInv;

	// Precompute squared Mahalanobis threshold once.
	// exp(-0.5 * d2) > thOutlier  <=>  d2 < -2*log(thOutlier)

	if (thOutlier <= 0 || thOutlier >= 1.0f) {
		thOutlier = 0.06f;
	}
	const double thMahalanobisSq = -2.0 * std::log((double)thOutlier);

	for (unsigned iter = 0; iter < maxIterations; ++iter)
	{
		// === compute mean & covariance for inliers ===
		const Eigen::Block<Eigen::Matrix3Xd, 3, Eigen::Dynamic, !Eigen::Matrix3Xd::IsRowMajor>
			colors(colorsAll.leftCols(numInliers));

		mean = colors.rowwise().mean();
#if 1
		Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

		const Eigen::Index n = colors.cols();
		const double inv = 1.0 / std::max(1.0, double(n - 1));

		for (Eigen::Index i = 0; i < n; ++i) {
			const auto x = colors.col(i);

			const double dx0 = x[0] - mean[0];
			const double dx1 = x[1] - mean[1];
			const double dx2 = x[2] - mean[2];

			cov(0, 0) += dx0 * dx0;
			cov(0, 1) += dx0 * dx1;
			cov(0, 2) += dx0 * dx2;
			cov(1, 1) += dx1 * dx1;
			cov(1, 2) += dx1 * dx2;
			cov(2, 2) += dx2 * dx2;
		}

		cov(1, 0) = cov(0, 1);
		cov(2, 0) = cov(0, 2);
		cov(2, 1) = cov(1, 2);

		covariance = cov * inv;
#else
		const Eigen::Matrix3Xd centered(colors.colwise() - mean);
		covariance = (centered * centered.transpose()) / std::max(1.0, double(colors.cols() - 1));
#endif

		// stop if covariance nearly zero
		if (covariance.array().abs().maxCoeff() < minCovariance)
		{
			RFOREACH(i, faceDatas)
				if (!inliers[i])
					faceDatas.RemoveAt(i);
			return true;
		}

		// === fast stable inverse (LLT for SPD 3×3) ===
		Eigen::LLT<Eigen::Matrix3d> llt(covariance);
		if (llt.info() != Eigen::Success)
			return false;
		covarianceInv = llt.solve(Eigen::Matrix3d::Identity());

		// === classify ===
		size_t newCount = 0;
		bool bChanged = false;

		for (uint32_t i = 0, cnt = (uint32_t)faceDatas.GetSize(); i < cnt; ++i)
		{
			const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
			const double dx0 = color[0] - mean[0];
			const double dx1 = color[1] - mean[1];
			const double dx2 = color[2] - mean[2];

			const double t0 = covarianceInv(0, 0) * dx0 + covarianceInv(0, 1) * dx1 + covarianceInv(0, 2) * dx2;
			const double t1 = covarianceInv(1, 0) * dx0 + covarianceInv(1, 1) * dx1 + covarianceInv(1, 2) * dx2;
			const double t2 = covarianceInv(2, 0) * dx0 + covarianceInv(2, 1) * dx1 + covarianceInv(2, 2) * dx2;

			const double dist2 = dx0 * t0 + dx1 * t1 + dx2 * t2; // Mahalanobis distance squared

			const bool isInlier = (dist2 < thMahalanobisSq);
			bool& inlier = inliers[i];

			if (isInlier)
			{
				colorsAll.col(newCount++) = color;
				if (!inlier) { inlier = true; bChanged = true; }
			}
			else
			{
				if (inlier) { inlier = false; bChanged = true; }
			}
		}

		numInliers = newCount;
		if (numInliers == faceDatas.GetSize())
			return true;
		if (numInliers < minInliers)
			return false;
		if (!bChanged)
			break;
	}
#else
	size_t numInliers(faceDatas.GetSize());
	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	Eigen::Matrix3d covarianceInv;
	for (unsigned iter = 0; iter < maxIterations; ++iter) {
		// compute the mean color and color covariance only for inliers
		const Eigen::Block<Eigen::Matrix3Xd,3,Eigen::Dynamic,!Eigen::Matrix3Xd::IsRowMajor> colors(colorsAll.leftCols(numInliers));
		mean = colors.rowwise().mean();
		const Eigen::Matrix3Xd centered(colors.colwise() - mean);
		covariance = (centered * centered.transpose()) / double(colors.cols() - 1);

		// stop if all covariances gets very small
		if (covariance.array().abs().maxCoeff() < minCovariance) {
			// remove the outliers
			RFOREACH(i, faceDatas)
				if (!inliers[i])
					faceDatas.RemoveAt(i);
			return true;
		}

		// invert the covariance matrix
		// (FullPivLU not the fastest, but gives feedback about numerical stability during inversion)
		const Eigen::FullPivLU<Eigen::Matrix3d> lu(covariance);
		if (!lu.isInvertible())
			return false;
		covarianceInv = lu.inverse();

		// filter inliers
		// (all views with a gauss value above the threshold)
		numInliers = 0;
		bool bChanged(false);
		FOREACH(i, faceDatas) {
			const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
			const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
			bool& inlier = inliers[i];
			if (gaussValue > thOutlier) {
				// set as inlier
				colorsAll.col(numInliers++) = color;
				if (inlier != true) {
					inlier = true;
					bChanged = true;
				}
			} else {
				// set as outlier
				if (inlier != false) {
					inlier = false;
					bChanged = true;
				}
			}
		}
		if (numInliers == faceDatas.GetSize())
			return true;
		if (numInliers < minInliers)
			return false;
		if (!bChanged)
			break;
	}
#endif

	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_DAMPING
	// select the final inliers
	const float factorOutlierRemoval(0.2f);
	covarianceInv *= factorOutlierRemoval;
	RFOREACH(i, faceDatas) {
		const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
		const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
		ASSERT(gaussValue >= 0 && gaussValue <= 1);
		faceDatas[i].quality *= gaussValue;
	}
	#endif
	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_CLAMPING
	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	#endif
	return true;
}
#endif

bool MeshTexture::FaceViewSelection(unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness, const IIndexArr& views)
{
	// extract array of triangles incident to each vertex
	ListVertexFaces();

	// create texture patches
	{
		// compute face normals and smoothen them
		scene.mesh.SmoothNormalFaces();

		// list all views for each face
		FaceDataViewArr facesDatas;
		if (!ListCameraFaces(facesDatas, fOutlierThreshold, views))
			return false;

		LabelArr labels;

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

		// JPB WIP Unused
		if (bUseVirtualFaces) {
			// create faces graph
			typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
			typedef boost::graph_traits<Graph>::edge_iterator EdgeIter;
			typedef boost::graph_traits<Graph>::out_edge_iterator EdgeOutIter;
			Graph graph;

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
				for (FIndex idxVirtualFaceAdj: afaces) {
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
				for (const FaceDataArr& faceDatas: virtualFacesDatas) {
					for (const FaceData& faceData: faceDatas)
						if (maxQuality < faceData.quality)
							maxQuality = faceData.quality;
				}
				Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);
				for (const FaceDataArr& faceDatas: virtualFacesDatas) {
					for (const FaceData& faceData: faceDatas)
						hist.Add(faceData.quality);
				}
				const float normQuality(hist.GetApproximatePermille(0.95f));

				#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
				// initialize inference structures
				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness*LBPInference::MaxEnergy);
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
					for (const FaceData& faceData: faceDatas) {
						const Label label((Label)faceData.idxView+1);
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
						const float dataCost((1.f-normalizedQuality)*MaxEnergy);
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
					ASSERT(label < images.GetSize()+1);
					if (label > 0)
						virtualLabels[l] = label-1;
				}
				FOREACH(l, labels) {
					labels[l] = virtualLabels[mapFaceToVirtualFace[l]];
				}
				#endif
			}

			graph.clear();
		}
	
		// start patch creation starting directly from individual faces
		if (!bUseVirtualFaces) {
			// No longer uses boost graph handling.
			// assign the best view to each face
			labels.resize(faces.size());
			{
#undef USE_HISTOGRAM_APPROXIMATE_PERMILLE
#ifdef USE_HISTOGRAM_APPROXIMATE_PERMILLE
				// normalize quality values
				float maxQuality(0);
				for (const FaceDataArr& faceDatas: facesDatas) {
					for (const FaceData& faceData: faceDatas)
						if (maxQuality < faceData.quality)
							maxQuality = faceData.quality;
				}
				Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);
				for (const FaceDataArr& faceDatas: facesDatas) {
					for (const FaceData& faceData: faceDatas)
						hist.Add(faceData.quality);
				}
				const float normQuality(hist.GetApproximatePermille(0.95f));
#else
				const float normQuality = 0.8f; // e.g. 0.7f or 0.8f
#endif

#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
				// initialize inference structures
				labels.Memset(0xFF);
				int64_t numFaces = (uint32_t)faces.size();
				std::vector<uint8_t> active(numFaces, 1);

				size_t numActive = 0;

				FOREACH(f, faces) {
					const FaceDataArr& fd = facesDatas[f];

					if (fd.IsEmpty()) {
						labels[f] = NO_ID;
						active[f] = 0;
						continue;
					}

					if (fd.GetSize() == 1) {
						labels[f] = fd[0].idxView;
						active[f] = 0;
						continue;
					}

					active[f] = 1;
					++numActive;
				}

				// ---------- degree + edge counting (ACTIVE ONLY) ----------
				std::vector<uint32_t> degree(numFaces, 0);
				uint64_t numDirectedEdges = 0;

				FOREACH(f, faces) {
					if (!active[f])
						continue;

					const Mesh::FaceFaces& afaces = faceFaces[f];
					for (int k = 0; k < 3; ++k) {
						const FIndex fAdj = afaces[k];
						if (fAdj == NO_ID || f >= fAdj)
							continue;
						if (!active[fAdj])
							continue;

						++degree[f];
						++degree[fAdj];
						numDirectedEdges += 2;
					}
				}

				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness*LBPInference::MaxEnergy);

				LBPInference inference;
				{
					inference.SetNumNodes(faces.size());
					inference.SetSmoothCost(SmoothnessPotts);

					inference.edges.reserve(numDirectedEdges);

					FOREACH(f, faces) {
						if (!active[f])
							continue;

						const Mesh::FaceFaces& afaces = faceFaces[f];
						for (int k = 0; k < 3; ++k) {
							const FIndex fAdj = afaces[k];
							if (fAdj == NO_ID || f >= fAdj)
								continue;
							if (!active[fAdj])
								continue;
							inference.SetNeighbors(f, fAdj);
						}
					}

#pragma omp parallel for schedule(static)
					for (int64_t f = 0; f < (int64_t)numFaces; ++f) {
						if (!active[f]) continue;
						// set costs for label 0 (undefined)
						inference.SetDataCost((Label)0, f, MaxEnergy);
					}
				}

				// JPB WIP Why is this released? faceFaces.Release();

				// set data costs for all labels (except label 0 - undefined)
				const float invNormQuality = 1.f / normQuality;

#pragma omp parallel for schedule(static)
				for (int f = 0; f < (int)facesDatas.size(); ++f) {
					if (!active[f])
						continue;

					const FaceDataArr& faceDatas = facesDatas[f];
					if (faceDatas.IsEmpty())
						continue;

					LBPInference::Node& node = inference.nodes[f];
					node.labels.clear();
					node.dataCosts.clear();
					const size_t numLabels = faceDatas.GetSize() + 1;
					//node.labels.reserve(numLabels);
					//node.dataCosts.reserve(numLabels);
					node.dataCost = MaxEnergy;
					node.label = 0;

					for (const FaceData& faceData : faceDatas) {
						const Label label = (Label)faceData.idxView + 1;

						const float normalizedQuality =
							(faceData.quality >= normQuality)
							? 1.f
							: faceData.quality * invNormQuality;

						const LBPInference::EnergyType dataCost =
							(LBPInference::EnergyType)((1.f - normalizedQuality) * MaxEnergy);

						// --- inline SetDataCost, without per-edge growth ---
						node.labels.push_back(label);
						node.dataCosts.push_back(dataCost);

						if (dataCost < node.dataCost) {
							node.label = (Label)(node.labels.size() - 1);
							node.dataCost = dataCost;
						}
					}
				}

#if 0
				// 2) Reserve message buffers per edge (parallel, edge-centric)
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)inference.edges.size(); ++i) {
					auto& e = inference.edges[i];
					if (inference.nodes[e.nodeID2].labels.empty())
						continue;

					size_t maxLabels =
						std::max(inference.nodes[e.nodeID1].labels.size(),
							inference.nodes[e.nodeID2].labels.size());
					e.oldMsgs.reserve(maxLabels);
					e.newMsgs.reserve(maxLabels);
				}

				// 3) Resize message buffers per edge (parallel, edge-centric)
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)inference.edges.size(); ++i) {
					auto& e = inference.edges[i];
					if (inference.nodes[e.nodeID2].labels.empty())
						continue;

					size_t numLabels = inference.nodes[e.nodeID2].labels.size();
					e.oldMsgs.resize(numLabels);
					e.newMsgs.resize(numLabels);
				}
#endif

				// assign the optimal view (label) to each face
				// (label 0 is reserved as undefined)
#if 0 // JPB WIP BUG
				FOREACH(f, faces) {
					if (!active[f])
						continue;

					const auto& node = inference.nodes[f];
					HARD_ASSERT(!node.labels.empty());
					HARD_ASSERT(node.labels.size() == node.dataCosts.size());
					HARD_ASSERT(node.label < node.labels.size());
				}
#endif
				inference.Optimize();

				// extract resulting labeling
				FOREACH(f, faces) {
					if (!active[f])
						continue;

					const Label label = inference.GetLabel(f);
					ASSERT(label < images.size() + 1);
					if (label > 0)
						labels[f] = label - 1;
				}
#endif

				#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
				// find connected components
				const FIndex nComponents(boost::connected_components(graph, components.data()));

				// map face ID from global to component space
				typedef cList<NodeID, NodeID, 0, 128, NodeID> NodeIDs;
				NodeIDs nodeIDs(faces.GetSize());
				NodeIDs sizes(nComponents);
				sizes.Memset(0);
				FOREACH(c, components)
					nodeIDs[c] = sizes[components[c]]++;

				// initialize inference structures
				const LabelID numLabels(images.GetSize()+1);
				CLISTDEFIDX(TRWSInference, FIndex) inferences(nComponents);
				FOREACH(s, sizes) {
					const NodeID numNodes(sizes[s]);
					ASSERT(numNodes > 0);
					if (numNodes <= 1)
						continue;
					TRWSInference& inference = inferences[s];
					inference.Init(numNodes, numLabels);
				}

				// set data costs
				{
					// add nodes
					CLISTDEF0(EnergyType) D(numLabels);
					FOREACH(f, facesDatas) {
						TRWSInference& inference = inferences[components[f]];
						if (inference.IsEmpty())
							continue;
						D.MemsetValue(MaxEnergy);
						const FaceDataArr& faceDatas = facesDatas[f];
						for (const FaceData& faceData: faceDatas) {
							const Label label((Label)faceData.idxView);
							const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
							const EnergyType dataCost(MaxEnergy*(1.f-normalizedQuality));
							D[label] = dataCost;
						}
						const NodeID nodeID(nodeIDs[f]);
						inference.AddNode(nodeID, D.Begin());
					}
					// add edges
					EdgeOutIter ei, eie;
					FOREACH(f, faces) {
						TRWSInference& inference = inferences[components[f]];
						if (inference.IsEmpty())
							continue;
						for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
							ASSERT(f == (FIndex)ei->m_source);
							const FIndex fAdj((FIndex)ei->m_target);
							ASSERT(components[f] == components[fAdj]);
							if (f < fAdj) // add edges only once
								inference.AddEdge(nodeIDs[f], nodeIDs[fAdj]);
						}
					}
				}

				// assign the optimal view (label) to each face
				#ifdef TEXOPT_USE_OPENMP
				#pragma omp parallel for schedule(dynamic)
				for (int i=0; i<(int)inferences.GetSize(); ++i) {
				#else
				FOREACH(i, inferences) {
				#endif
					TRWSInference& inference = inferences[i];
					if (inference.IsEmpty())
						continue;
					inference.Optimize();
				}
				// extract resulting labeling
				labels.Memset(0xFF);
				FOREACH(l, labels) {
					TRWSInference& inference = inferences[components[l]];
					if (inference.IsEmpty())
						continue;
					const Label label(inference.GetLabel(nodeIDs[l]));
					ASSERT(label >= 0 && label < numLabels);
					if (label < images.GetSize())
						labels[l] = label;
				}
				#endif
			}
		}

		// create texture patches
		{
			// create texture patches (no Boost, unified CC + seam edges)
			{
				const FIndex numFaces = faces.GetSize();

				components.Resize(numFaces);
				components.Memset(0xFF); // MemsetValue(NO_ID);

				seamEdges.clear();
				seamEdges.Reserve(numFaces);

				boost::container::small_vector<FIndex, 128> stack;

				FIndex curComponent = 0;

				// ---- DFS components + seam edges in one pass ----
				for (FIndex f = 0; f < numFaces; ++f) {
					if (components[f] != NO_ID)
						continue;

					components[f] = curComponent;
					stack.clear();
					stack.push_back(f);

					while (!stack.empty()) {
						const FIndex u = stack.back();
						stack.pop_back();

						const Mesh::FaceFaces& afaces = faceFaces[u];
						for (int k = 0; k < 3; ++k) {
							const FIndex v = afaces[k];
							if (v == NO_ID)
								continue;

							if (labels[u] != labels[v]) {
								if (u < v)
									seamEdges.emplace_back(u, v);
								continue;
							}

							if (components[v] == NO_ID) {
								components[v] = curComponent;
								stack.push_back(v);
							}
						}
					}

					++curComponent;
				}

				const FIndex nComponents = curComponent;

				// ---- compute component sizes ----
				LabelArr sizes(nComponents);
				sizes.Memset(0);
				FOREACH(i, components)
					++sizes[components[i]];

				// ---- build texture patches ----
				texturePatches.Resize(nComponents + 1);
				texturePatches.Last().label = NO_ID;

				FOREACH(f, faces) {
					const Label label = labels[f];
					const FIndex c = components[f];
					TexturePatch& texturePatch = texturePatches[c];

					ASSERT(texturePatch.label == label || texturePatch.faces.IsEmpty());

					if (label == NO_ID) {
						texturePatch.label = NO_ID;
						texturePatches.Last().faces.Insert(f);
					}
					else {
						if (texturePatch.faces.IsEmpty()) {
							texturePatch.label = label;
							texturePatch.faces.Reserve(sizes[c]);
						}
						texturePatch.faces.Insert(f);
					}
				}

				// ---- compact patches and build mapIdxPatch ----
				mapIdxPatch.Resize(nComponents);
				std::iota(mapIdxPatch.Begin(), mapIdxPatch.End(), 0);

				FIndex write = 0;
				for (FIndex read = 0; read < nComponents; ++read) {
					if (texturePatches[read].label != NO_ID) {
						if (write != read) {
							texturePatches[write] = std::move(texturePatches[read]);
							mapIdxPatch[write] = mapIdxPatch[read];
						}
						++write;
					}
				}

				texturePatches.Resize(write);
				mapIdxPatch.Resize(write);

				const unsigned numPatches = texturePatches.GetSize() - 1;
				uint32_t idxPatch = 0;

				for (IndexArr::IDX i = 0; i < mapIdxPatch.GetSize(); ++i) {
					while (i < mapIdxPatch[i])
						mapIdxPatch.InsertAt(i++, numPatches);
					mapIdxPatch[i] = idxPatch++;
				}

				while (mapIdxPatch.GetSize() <= nComponents)
					mapIdxPatch.Insert(numPatches);
			}
		}
	}
	return true;
}


// create seam vertices and edges
void MeshTexture::CreateSeamVertices()
{
	// each vertex will contain the list of patches it separates,
	// except the patch containing invisible faces;
	// each patch contains the list of edges belonging to that texture patch, starting from that vertex
	// (usually there are pairs of edges in each patch, representing the two edges starting from that vertex separating two valid patches)
	VIndex vs[2];
	uint32_t vs0[2], vs1[2];
	std::unordered_map<VIndex, uint32_t> mapVertexSeam;
	const unsigned numPatches(texturePatches.GetSize()-1);
	for (const PairIdx& edge: seamEdges) {
		// store edge for the later seam optimization
		ASSERT(edge.i < edge.j);
		const uint32_t idxPatch0(mapIdxPatch[components[edge.i]]);
		const uint32_t idxPatch1(mapIdxPatch[components[edge.j]]);
		ASSERT(idxPatch0 != idxPatch1 || idxPatch0 == numPatches);
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

		if (idxPatch0 < numPatches) {
			const TexCoord offset0(texturePatches[idxPatch0].rect.tl());
			SeamVertex::Patch& patch00 = seamVertex0.GetPatch(idxPatch0);
			SeamVertex::Patch& patch10 = seamVertex1.GetPatch(idxPatch0);
			ASSERT(patch00.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch00.edges.emplace_back(itSeamVertex1.first->second).idxFace = edge.i;
			patch00.proj = faceTexcoords[edge.i*3+vs0[0]]+offset0;
			ASSERT(patch10.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch10.edges.emplace_back(itSeamVertex0.first->second).idxFace = edge.i;
			patch10.proj = faceTexcoords[edge.i*3+vs0[1]]+offset0;
		}
		if (idxPatch1 < numPatches) {
			const TexCoord offset1(texturePatches[idxPatch1].rect.tl());
			SeamVertex::Patch& patch01 = seamVertex0.GetPatch(idxPatch1);
			SeamVertex::Patch& patch11 = seamVertex1.GetPatch(idxPatch1);
			ASSERT(patch01.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch01.edges.emplace_back(itSeamVertex1.first->second).idxFace = edge.j;
			patch01.proj = faceTexcoords[edge.j*3+vs1[0]]+offset1;
			ASSERT(patch11.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch11.edges.emplace_back(itSeamVertex0.first->second).idxFace = edge.j;
			patch11.proj = faceTexcoords[edge.j*3+vs1[1]]+offset1;
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

void MeshTexture::GlobalSeamLeveling()
{
	ASSERT(!seamVertices.IsEmpty());
	const unsigned numPatches(texturePatches.GetSize()-1);

	// find the patch ID for each vertex
	PatchIndices patchIndices(vertices.GetSize());
	patchIndices.Memset(0);
	FOREACH(f, faces) {
		const uint32_t idxPatch(mapIdxPatch[components[f]]);
		const Face& face = faces[f];
		for (int v=0; v<3; ++v)
			patchIndices[face[v]].idxPatch = idxPatch;
	}
	FOREACH(i, seamVertices) {
		const SeamVertex& seamVertex = seamVertices[i];
		ASSERT(!seamVertex.patches.IsEmpty());
		PatchIndex& patchIndex = patchIndices[seamVertex.idxVertex];
		patchIndex.bIndex = true;
		patchIndex.idxSeamVertex = i;
	}

	// assign a row index within the solution vector x to each vertex/patch
	ASSERT(vertices.GetSize() < static_cast<VIndex>(std::numeric_limits<MatIdx>::max()));
	MatIdx rowsX(0);
	typedef std::unordered_map<uint32_t,MatIdx> VertexPatch2RowMap;
	cList<VertexPatch2RowMap> vertpatch2rows(vertices.GetSize());
	FOREACH(i, vertices) {
		const PatchIndex& patchIndex = patchIndices[i];
		VertexPatch2RowMap& vertpatch2row = vertpatch2rows[i];
		if (patchIndex.bIndex) {
			// vertex is part of multiple patches
			const SeamVertex& seamVertex = seamVertices[patchIndex.idxSeamVertex];
			ASSERT(seamVertex.idxVertex == i);
			for (const SeamVertex::Patch& patch: seamVertex.patches) {
				ASSERT(patch.idxPatch != numPatches);
				vertpatch2row[patch.idxPatch] = rowsX++;
			}
		} else
		if (patchIndex.idxPatch < numPatches) {
			// vertex is part of only one patch
			vertpatch2row[patchIndex.idxPatch] = rowsX++;
		}
	}

	// fill Tikhonov's Gamma matrix (regularization constraints)
	////

	// ------------------------------------------------------------
	// Parallel assembly of A (Gamma folded + data constraints)
	// ------------------------------------------------------------

	const float lambda = 0.1f;

	const int numThreads = omp_get_max_threads();

	// Thread-local storage
	std::vector<CLISTDEF0(MatEntry)> tlsTriplets(numThreads);
	std::vector<Colors>             tlsCoeffB(numThreads);

	// Reserve heuristics (safe guesses)
	for (int t = 0; t < numThreads; ++t) {
		tlsTriplets[t].reserve(seamVertices.GetSize() * 4 / numThreads + 64);
		tlsCoeffB[t].reserve(seamVertices.GetSize() * 2 / numThreads + 32);
	}

#pragma omp parallel
	{
		const int tid = omp_get_thread_num();

		CLISTDEF0(MatEntry)& localTriplets = tlsTriplets[tid];
		Colors& localCoeffB = tlsCoeffB[tid];

		IndexArr indices;
		Colors vertexColors;
		boost::container::small_vector<VIndex, 32> adjVerts;

		// ----------------------------------------------------------
		// 1) Gamma constraints (same as original, just parallel)
		// ----------------------------------------------------------
#pragma omp for schedule(static)
		for (int vi = 0; vi < (int)vertices.GetSize(); ++vi) {
			const VIndex v = (VIndex)vi;

			adjVerts.clear();
			scene.mesh.GetAdjVertices(v, adjVerts);

			VertexPatchIterator itV(patchIndices[v], seamVertices);
			while (true) {
				if (!itV.Next())
					break;

				const uint32_t idxPatch = itV;
				if (idxPatch == numPatches)
					continue;

				const MatIdx col = vertpatch2rows[v].at(idxPatch);

				for (const VIndex vAdj : adjVerts) {
					if (v >= vAdj)
						continue;

					VertexPatchIterator itVAdj(patchIndices[vAdj], seamVertices);
					while (true) {
						if (!itVAdj.Next())
							break;

						const uint32_t idxPatchAdj = itVAdj;
						if (idxPatchAdj == idxPatch) {
							const MatIdx colAdj = vertpatch2rows[vAdj].at(idxPatchAdj);

							const MatIdx row = (MatIdx)localCoeffB.GetSize();
							localCoeffB.Insert(Color(0.f, 0.f, 0.f));

							localTriplets.emplace_back(row, col, lambda);
							localTriplets.emplace_back(row, colAdj, -lambda);
						}
					}
				}
			}
		}

		// ----------------------------------------------------------
		// 2) Data constraints (same as original, just parallel)
		// ----------------------------------------------------------
#pragma omp for schedule(static)
		for (int si = 0; si < (int)seamVertices.GetSize(); ++si) {
			const SeamVertex& seamVertex = seamVertices[(IDX)si];
			if (seamVertex.patches.GetSize() < 2)
				continue;

			seamVertex.SortByPatchIndex(indices);
			vertexColors.Resize(indices.GetSize());

			FOREACH(i, indices) {
				const SeamVertex::Patch& patch0 = seamVertex.patches[indices[i]];
				SampleImage sampler(images[texturePatches[patch0.idxPatch].label].image);

				for (const SeamVertex::Patch::Edge& edge : patch0.edges) {
					const SeamVertex& sv1 = seamVertices[edge.idxSeamVertex];
					const auto idxPatch1 = sv1.patches.Find(patch0.idxPatch);
					sampler.AddEdge(patch0.proj, sv1.patches[idxPatch1].proj);
				}
				vertexColors[i] = sampler.GetColor();
			}

			const VertexPatch2RowMap& v2r = vertpatch2rows[seamVertex.idxVertex];

			for (IDX i = 0; i < indices.GetSize() - 1; ++i) {
				const uint32_t p0 = seamVertex.patches[indices[i]].idxPatch;
				const MatIdx c0 = v2r.at(p0);
				const Color& col0 = vertexColors[i];

				for (IDX j = i + 1; j < indices.GetSize(); ++j) {
					const uint32_t p1 = seamVertex.patches[indices[j]].idxPatch;
					const MatIdx c1 = v2r.at(p1);
					const Color& col1 = vertexColors[j];

					const MatIdx row = (MatIdx)localCoeffB.GetSize();
					localCoeffB.Insert(col1 - col0);

					localTriplets.emplace_back(row, c0, 1.f);
					localTriplets.emplace_back(row, c1, -1.f);
				}
			}
		}
	}

	// ------------------------------------------------------------
	// Merge thread-local buffers (preserves per-thread row indexing)
	// ------------------------------------------------------------

	Colors coeffB;
	CLISTDEF0(MatEntry) aTriplets;

	std::vector<MatIdx> rowOffsets(numThreads, 0);
	for (int t = 1; t < numThreads; ++t)
		rowOffsets[t] = rowOffsets[t - 1] + tlsCoeffB[t - 1].GetSize();

	for (int t = 0; t < numThreads; ++t) {

		for (const Color& c : tlsCoeffB[t])
			coeffB.Insert(c);

		for (auto& e : tlsTriplets[t]) {
			aTriplets.emplace_back(
				e.row() + rowOffsets[t],
				e.col(),
				e.value()
			);
		}
	}

	// ------------------------------------------------------------
	// 3) Build A from combined triplets
	// ------------------------------------------------------------
	// Eigen par not helping here.  Completely dominated by Solve.

	const MatIdx rowsA = (MatIdx)coeffB.GetSize();

	SparseMat A(rowsA, rowsX);
	A.setFromTriplets(aTriplets.Begin(), aTriplets.End());
	A.makeCompressed();

	// ------------------------------------------------------------
	// 4) Normal equations (FAST)
	// ------------------------------------------------------------

	// Precompute transpose ONCE
	SparseMat AT = A.transpose();
	AT.makeCompressed();

	// A^T A only (no Gamma^T Gamma)
	SparseMat Lhs(rowsX, rowsX);
	Lhs = AT * A;
	Lhs.makeCompressed();

	// Ensure every diagonal entry exists
	for (MatIdx i = 0; i < rowsX; ++i) {
		if (Lhs.coeff(i, i) == 0.f)
			Lhs.coeffRef(i, i) = 1e-6f;
	}
	// Keep only lower triangle (CG fast path)
	Lhs.prune([](const int& r, const int& c, const float&) {
		return c <= r;
		});
	Lhs.makeCompressed();

	// Output
	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
		colorAdjustments(rowsX, 3);

	// JPB WIP BUG Crashes Lhs.diagonal().array() += 1e-6f;

	// ------------------------------------------------------------
	// 5) CG solve (fast, stable, same accuracy)
	// ------------------------------------------------------------

#if 1
	Eigen::setNbThreads(1);

	static Eigen::VectorXf xPrev[3];
	static bool init = false;
	if (!init) {
		for (int i = 0; i < 3; ++i)
			xPrev[i].setZero(rowsX);
		init = true;
	}

#pragma omp parallel for num_threads(3) schedule(static)
	for (int c = 0; c < 3; ++c) {

		DWORD_PTR oldMask = PinThreadToCoreAndSave(c * 2);

		Eigen::ConjugateGradient<
			SparseMat,
			Eigen::Lower,
			Eigen::DiagonalPreconditioner<float>
		> solverLocal;

		solverLocal.setMaxIterations(1000);
		solverLocal.setTolerance(3e-4f);
		solverLocal.compute(Lhs);

		Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned,
			Eigen::Stride<0, 3>> b(coeffB.front().ptr() + c, rowsA);

		Eigen::VectorXf rhs(rowsX);
		rhs.noalias() = AT * b;

		Eigen::VectorXf x = solverLocal.solveWithGuess(rhs, xPrev[c]);
		xPrev[c] = x;

		const float invRowsX = 1.0f / rowsX;
		float mean = x.sum() * invRowsX;
		x.array() -= mean;

		Eigen::Map<Eigen::VectorXf, Eigen::Unaligned,
			Eigen::Stride<0, 3>> out(colorAdjustments.data() + c, rowsX);
		out.noalias() = x;

		RestoreThreadAffinity(oldMask);
	}

	Eigen::setNbThreads(0); // restore
#else
	Eigen::ConjugateGradient<
		SparseMat,
		Eigen::Lower,
		Eigen::DiagonalPreconditioner<float>
	> solver;

	solver.setMaxIterations(1000);
	solver.setTolerance(3e-4f);
	solver.compute(Lhs);
	ASSERT(solver.info() == Eigen::Success);

	Eigen::VectorXf xPrev(rowsX);
	xPrev.setZero();

	for (int c = 0; c < 3; ++c) {
		Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned,
			Eigen::Stride<0, 3>> b(coeffB.front().ptr() + c, rowsA);

		Eigen::VectorXf rhs(rowsX);
		rhs.noalias() = AT * b;

		Eigen::VectorXf x = solver.solveWithGuess(rhs, xPrev);
		ASSERT(solver.info() == Eigen::Success);

		float mean = x.sum() / x.size();
		xPrev = x;
		x.array() -= mean;

		Eigen::Map<Eigen::VectorXf, Eigen::Unaligned,
			Eigen::Stride<0, 3>> out(colorAdjustments.data() + c, rowsX);
		out.noalias() = x;
	}
#endif

	////
	// adjust texture patches using the correction colors
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(static, 1)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);
		TexturePatch& texturePatch = texturePatches[idxPatch];
		ColorMap imageAdj(texturePatch.rect.size());
		imageAdj.memset(0);
		// interpolate color adjustments over the whole patch
		struct RasterPatch {
			const TexCoord* tri;
			Color colors[3];
			ColorMap& image;
			inline RasterPatch(ColorMap& _image) : image(_image) {}
			inline cv::Size Size() const { return image.size(); }
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				ASSERT(image.isInside(pt));
				image(pt) = colors[0]*bary.x + colors[1]*bary.y + colors[2]*bary.z;
			}
		} data(imageAdj);
		
		for (const FIndex idxFace : texturePatch.faces) {
			const Face& face = faces[idxFace];
			data.tri = faceTexcoords.Begin() + idxFace * 3;
#if 1 // JPB WIP BUG
			bool valid = true;
			for (int v = 0; v < 3; ++v) {
				auto& tmp = vertpatch2rows[face[v]];
				auto it = tmp.find(idxPatch);
				if (it == tmp.end()) {
					valid = false;
					break;
				}
				data.colors[v] = colorAdjustments.row(it->second);
			}

			if (!valid) {
				data.colors[0] = data.colors[1] = data.colors[2] = Color::ZERO;
			}
#else
			for (int v = 0; v < 3; ++v)
				data.colors[v] = colorAdjustments.row(vertpatch2rows[face[v]].at(idxPatch));
#endif
			// render triangle and for each pixel interpolate the color adjustment
			// from the triangle corners using barycentric coordinates
			ColorMap::RasterizeTriangleBary(data.tri[0], data.tri[1], data.tri[2], data);
		}

#if 1 // Retry, to remove splotches.  Make dilation 0 and fuse it with application:
		cv::Mat image(images[texturePatch.label].image(texturePatch.rect));

		for (int r = 1; r < image.rows - 1; ++r) {
			const Color* __restrict prev = (Color*)imageAdj.ptr(r - 1);
			const Color* __restrict curr = (Color*)imageAdj.ptr(r);
			const Color* __restrict next = (Color*)imageAdj.ptr(r + 1);

			Pixel8U* __restrict out = image.ptr<Pixel8U>(r);

			for (int c = 1; c < image.cols - 1; ++c) {

				Color a = curr[c];

				// If no correction here, try to dilate from neighbors
				if (a == Color::ZERO) {

					Color sum(0);
					int n = 0;

					const Color v0 = prev[c - 1];
					const Color v1 = prev[c];
					const Color v2 = prev[c + 1];
					const Color v3 = curr[c - 1];
					const Color v4 = curr[c + 1];
					const Color v5 = next[c - 1];
					const Color v6 = next[c];
					const Color v7 = next[c + 1];

					if (v0 != Color::ZERO) { sum += v0; ++n; }
					if (v1 != Color::ZERO) { sum += v1; ++n; }
					if (v2 != Color::ZERO) { sum += v2; ++n; }
					if (v3 != Color::ZERO) { sum += v3; ++n; }
					if (v4 != Color::ZERO) { sum += v4; ++n; }
					if (v5 != Color::ZERO) { sum += v5; ++n; }
					if (v6 != Color::ZERO) { sum += v6; ++n; }
					if (v7 != Color::ZERO) { sum += v7; ++n; }

					if (!n)
						continue;

					a = (n > 1 ? sum / n : sum);
				}

				// Apply correction immediately
				Pixel8U& v = out[c];

				// scalar color space math (unchanged, correct)
				const Color col = RGB2YCBCR(Color(v));
				const Color acol = YCBCR2RGB(Color(col + a));

				// SIMD round + clamp (this is the only optimized part)
				__m128 rgbf = _mm_set_ps(0.0f, acol[2], acol[1], acol[0]);
				__m128i rgbi = _mm_cvtps_epi32(rgbf);

				const __m128i zero = _mm_setzero_si128();
				const __m128i max255 = _mm_set1_epi32(255);

				rgbi = _mm_max_epi32(rgbi, zero);
				rgbi = _mm_min_epi32(rgbi, max255);

				// pack to bytes
				__m128i pack16 = _mm_packus_epi32(rgbi, rgbi);
				__m128i pack8 = _mm_packus_epi16(pack16, pack16);

				uint32_t rgb8 = (uint32_t)_mm_cvtsi128_si32(pack8);
				v[0] = (uint8_t)(rgb8 & 0xFF);
				v[1] = (uint8_t)((rgb8 >> 8) & 0xFF);
				v[2] = (uint8_t)((rgb8 >> 16) & 0xFF);
			}
		}
#else
		// dilate with one pixel width, in order to make sure patch border smooths out a little
		// JPB WIP BUG Try imageAdj.DilateMean<1>(imageAdj, Color::ZERO);
		imageAdj.DilateMean<0>(imageAdj, Color::ZERO);
		// apply color correction to the patch image
		cv::Mat image(images[texturePatch.label].image(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				const Color& a = imageAdj(r,c);
				if (a == Color::ZERO)
					continue;
				Pixel8U& v = image.at<Pixel8U>(r,c);
				const Color col(RGB2YCBCR(Color(v)));
				const Color acol(YCBCR2RGB(Color(col+a)));
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(acol[p]), 0, 255);
			}
		}
#endif
	}
}

// set to one in order to dilate also on the diagonal of the border
// (normally not needed)
#define DILATE_EXTRA 0
#if 1 // too aggressive?
void MeshTexture::ProcessMask(Image8U& mask, int stripWidth)
{
	typedef Image8U::Type Type;

	const int width = mask.width();
	const int height = mask.height();
	const int stride = width;

	Type* data = (Type*) mask.data;

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

	// ------------------------------------------------------------
	// 4) Compute initial border frontier
	// ------------------------------------------------------------
	std::vector<int> frontier;
	frontier.reserve(width * 2 + height * 2);

	std::vector<uint8_t> visited(width * height, 0);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const int idx = Idx(x, y);
			if (data[idx] == empty)
				continue;

			bool touchesEmpty = false;
			for (int dy = -1; dy <= 1 && !touchesEmpty; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int xn = x + dx;
					int yn = y + dy;
					if ((unsigned)xn >= (unsigned)width ||
						(unsigned)yn >= (unsigned)height)
						continue;
					if (data[Idx(xn, yn)] == empty) {
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
	// 5) Iterative strip erosion (frontier-based)
	// ------------------------------------------------------------
	std::vector<int> nextFrontier;
	nextFrontier.reserve(frontier.size());

	for (int s = 0; s < stripWidth; ++s) {
		nextFrontier.clear();

		// Remove current frontier
		for (int idx : frontier)
			data[idx] = empty;

		// Grow new frontier
		for (int idx : frontier) {
			int x = idx % stride;
			int y = idx / stride;

			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int xn = x + dx;
					int yn = y + dy;
					if ((unsigned)xn >= (unsigned)width ||
						(unsigned)yn >= (unsigned)height)
						continue;

					int nidx = Idx(xn, yn);
					if (data[nidx] != empty && !visited[nidx]) {
						visited[nidx] = 1;
						nextFrontier.push_back(nidx);
					}
				}
			}
		}

		frontier.swap(nextFrontier);
		if (frontier.empty())
			break;
	}

	// ------------------------------------------------------------
	// 6) Final cleanup: keep only remaining frontier as border
	// ------------------------------------------------------------
	for (int i = 0; i < width * height; ++i)
		if (data[i] != empty)
			data[i] = empty;

	for (int idx : frontier)
		data[idx] = border;
}
#else
void MeshTexture::ProcessMask(Image8U& mask, int stripWidth)
{
	typedef Image8U::Type Type;

	// dilate and erode around the border,
	// in order to fill all gaps and remove outside pixels
	// (due to imperfect overlay of the raster line border and raster faces)
	#define DILATEDIR(rd,cd) { \
		Type& vi = mask(r+(rd),c+(cd)); \
		if (vi != border) \
			vi = interior; \
	}
	const int HalfSize(1);
	const int RowsEnd(mask.rows-HalfSize);
	const int ColsEnd(mask.cols-HalfSize);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(mask(r,c));
			if (v != border)
				continue;
			#if DILATE_EXTRA
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw,cw);
					if (vi != border)
						vi = interior;
				}
			}
			#else
			DILATEDIR(-1, 0);
			DILATEDIR(1, 0);
			DILATEDIR(0, -1);
			DILATEDIR(0, 1);
			#endif
		}
	}
	#undef DILATEDIR
	#define ERODEDIR(rd,cd) { \
		const int rl(r-(rd)), cl(c-(cd)), rr(r+(rd)), cr(c+(cd)); \
		const Type vl(mask.isInside(ImageRef(cl,rl)) ? mask(rl,cl) : uint8_t(empty)); \
		const Type vr(mask.isInside(ImageRef(cr,rr)) ? mask(rr,cr) : uint8_t(empty)); \
		if ((vl == border && vr == empty) || (vr == border && vl == empty)) { \
			v = empty; \
			continue; \
		} \
	}
	#if DILATE_EXTRA
	for (int i=0; i<2; ++i)
	#endif
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			ERODEDIR(0, 1);
			ERODEDIR(1, 0);
			ERODEDIR(1, 1);
			ERODEDIR(-1, 1);
		}
	}
	#undef ERODEDIR

	// mark all interior pixels with empty neighbors as border
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if (mask(r-1,c) == empty ||
				mask(r,c-1) == empty ||
				mask(r+1,c) == empty ||
				mask(r,c+1) == empty)
				v = border;
		}
	}

	#if 0
	// mark all interior pixels with border neighbors on two sides as border
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if ((orgMask(r+1,c+0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r+1,c+0) == border && orgMask(r-0,c-1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r-0,c-1) == border))
				v = border;
		}
	}
	}
	#endif

	// compute the set of valid pixels at the border of the texture patch
	#define ISEMPTY(mask, x,y) (mask(y,x) == empty)
	const int width(mask.width()), height(mask.height());
	typedef std::unordered_set<ImageRef> PixelSet;
	PixelSet borderPixels;
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (ISEMPTY(mask, x,y))
				continue;
			// valid border pixels need no invalid neighbors
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				borderPixels.insert(ImageRef(x,y));
				continue;
			}
			// check the direct neighborhood of all invalid pixels
			for (int j=-1; j<=1; ++j) {
				for (int i=-1; i<=1; ++i) {
					// if the valid pixel has an invalid neighbor...
					const int xn(x+i), yn(y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						ISEMPTY(mask, xn,yn)) {
						// add the pixel to the set of valid border pixels
						borderPixels.insert(ImageRef(x,y));
						goto CONTINUELOOP;
					}
				}
			}
			CONTINUELOOP:;
		}
	}

	// iteratively erode all border pixels
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	typedef std::vector<ImageRef> PixelVector;
	for (int s=0; s<stripWidth; ++s) {
		PixelVector emptyPixels(borderPixels.begin(), borderPixels.end());
		borderPixels.clear();
		// mark the new empty pixels as empty in the mask
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it)
			orgMask(*it) = empty;
		// find the set of valid pixels at the border of the valid area
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it) {
			for (int j=-1; j<=1; j++) {
				for (int i=-1; i<=1; i++) {
					const int xn(it->x+i), yn(it->y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						!ISEMPTY(orgMask, xn, yn))
						borderPixels.insert(ImageRef(xn,yn));
				}
			}
		}
	}
	#undef ISEMPTY

	// mark all remaining pixels empty in the mask
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (orgMask(y,x) != empty)
				mask(y,x) = empty;
		}
	}
	}

	// mark all border pixels
	std::vector<ImageRef> borderVec(borderPixels.begin(), borderPixels.end());

	for (int i = 0; i < (int)borderVec.size(); ++i) {
		mask(borderVec[i]) = border;
	}

	#if 0
	// dilate border
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(orgMask(r, c));
			if (v != border)
				continue;
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw, cw);
					if (vi == empty)
						vi = border;
				}
			}
		}
	}
	}
	#endif
}
#endif

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

		// Residual check
		if ((iter & 15) == 0) {
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
	}

#ifdef COUNT_ITERATIONS
	iterations += lIters;
#endif
}

void MeshTexture::PoissonBlending(
	const Image32F3& src,
	Image32F3& dst,
	const Image8U& mask,
	float bias)
{
	ASSERT(src.width() == mask.width() && src.width() == dst.width());
	ASSERT(src.height() == mask.height() && src.height() == dst.height());
	ASSERT(src.channels() == 3 && dst.channels() == 3 && mask.channels() == 1);
	ASSERT(src.type() == CV_32FC3 && dst.type() == CV_32FC3 && mask.type() == CV_8U);

#ifndef _RELEASE
	for (int x = 0; x < mask.cols; ++x) {
		ASSERT(mask(0, x) != interior);
		ASSERT(mask(mask.rows - 1, x) != interior);
	}
	for (int y = 0; y < mask.rows; ++y) {
		ASSERT(mask(y, 0) != interior);
		ASSERT(mask(y, mask.cols - 1) != interior);
	}
#endif

	const int width = dst.width();
	const int height = dst.height();
	const int n = width * height;

	// Compact indexing (now using tiles)
	TImage<MatIdx> indices(dst.size());
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

	std::vector<PoissonStencil> stencil(nnz);
	std::vector<float> xR(nnz);
	std::vector<float> xG(nnz);
	std::vector<float> xB(nnz);
	std::vector<float> bR(nnz);
	std::vector<float> bG(nnz);
	std::vector<float> bB(nnz);
	std::vector<MatIdx> redInterior;
	std::vector<MatIdx> blackInterior;

	redInterior.reserve(nnz);
	blackInterior.reserve(nnz);

	// Build compact system
	const bool useSrcOnly = (bias == 1.0f);
	const float invBias = 1.0f - bias;

	for (int y = 0; y < height; ++y) {
		const float* __restrict srcPrev = (y > 0) ? src.ptr<float>(y - 1) : nullptr;
		const float* __restrict srcCur = src.ptr<float>(y);
		const float* __restrict srcNext = (y + 1 < height) ? src.ptr<float>(y + 1) : nullptr;

		const float* __restrict dstPrev = (!useSrcOnly && y > 0) ? dst.ptr<float>(y - 1) : nullptr;
		const float* __restrict dstCur = (!useSrcOnly) ? dst.ptr<float>(y) : nullptr;
		const float* __restrict dstNext = (!useSrcOnly && y + 1 < height) ? dst.ptr<float>(y + 1) : nullptr;

		for (int x = 0; x < width; ++x) {
			const int i = y * width + x;
			if (mask(i) == empty)
				continue;

			const MatIdx idx = indices(i);
			ASSERT(idx != -1);

			if (mask(i) == border) {
				PoissonStencil& s = stencil[idx];
				s.up = s.left = s.right = s.down = idx;

				const Color& c = (const Color&)dst(i);
				xR[idx] = c.x;
				xG[idx] = c.y;
				xB[idx] = c.z;

				continue;
			}

			// interior
			const int c = 3 * x;

			PoissonStencil& s = stencil[idx];
			s.up = indices(i - width);
			s.left = indices(i - 1);
			s.right = indices(i + 1);
			s.down = indices(i + width);

			// src Laplacian
			const float lSr =
				srcPrev[c] + srcCur[c - 3] + srcCur[c + 3] + srcNext[c] - srcCur[c] * 4.0f;
			const float lSg =
				srcPrev[c + 1] + srcCur[c - 2] + srcCur[c + 4] + srcNext[c + 1] - srcCur[c + 1] * 4.0f;
			const float lSb =
				srcPrev[c + 2] + srcCur[c - 1] + srcCur[c + 5] + srcNext[c + 2] - srcCur[c + 2] * 4.0f;

			if (useSrcOnly) {
				bR[idx] = lSr;
				bG[idx] = lSg;
				bB[idx] = lSb;
			}
			else {
				// dst Laplacian
				const float lDr =
					dstPrev[c] + dstCur[c - 3] + dstCur[c + 3] + dstNext[c] - dstCur[c] * 4.0f;
				const float lDg =
					dstPrev[c + 1] + dstCur[c - 2] + dstCur[c + 4] + dstNext[c + 1] - dstCur[c + 1] * 4.0f;
				const float lDb =
					dstPrev[c + 2] + dstCur[c - 1] + dstCur[c + 5] + dstNext[c + 2] - dstCur[c + 2] * 4.0f;

				bR[idx] = lSr * bias + lDr * invBias;
				bG[idx] = lSg * bias + lDg * invBias;
				bB[idx] = lSb * bias + lDb * invBias;
			}

			const Color& c0 = (const Color&)dst(i);
			xR[idx] = c0.x;
			xG[idx] = c0.y;
			xB[idx] = c0.z;

			if (((x + y) & 1) == 0)
				redInterior.push_back(idx);
			else
				blackInterior.push_back(idx);
		}
	}

	// Can't sort this. std::sort(redInterior.begin(), redInterior.end());
	// Can't sort this. std::sort(blackInterior.begin(), blackInterior.end());

	// FAST SOLVE (this replaces Eigen)
	SolvePoissonSOR_Compact(
		stencil.data(),
		bR.data(),
		bG.data(),
		bB.data(),
		xR.data(),
		xG.data(),
		xB.data(),
		redInterior,
		blackInterior,
		1.85f,    // omega
		1e-6f,    // tolerance
		400       // max iters
	);

	// Scatter back
	for (int y = 0; y < height; ++y) {
		float* __restrict row = dst.ptr<float>(y);

		for (int x = 0; x < width; ++x) {
			const int i = y * width + x;
			if (mask(i) == empty)
				continue;

			const MatIdx idx = indices(i);
			float* __restrict d = row + 3 * x;

			d[0] = xR[idx];
			d[1] = xG[idx];
			d[2] = xB[idx];
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
	ASSERT(!seamVertices.empty());
	const uint32_t numPatches = (uint32_t)texturePatches.size() - 1;

	// ----------------------------------------------------------------------
	// Optional precomputation: build fast O(1) lookup for each seam vertex
	// ----------------------------------------------------------------------
	{
		const uint32_t totalPatches = (uint32_t)texturePatches.size();
		for (SeamVertex& v : seamVertices) {
			v.patchIndexLookup.clear();
			uint32_t cnt = v.patches.size();
			v.patchIndexLookup.reserve(cnt);
			for (uint32_t j = 0; j < cnt; ++j) {
				v.patchIndexLookup.Insert(v.patches[j].idxPatch, (uint16_t)j); // JPB WIP BUG Is this always unique?
			}
		}
	}

	std::vector<std::vector<uint32_t>> patchToSeamVertices(texturePatches.size());

	for (uint32_t v = 0; v < seamVertices.size(); ++v) {
		const auto& vertex = seamVertices[v];
		for (const auto& patch : vertex.patches)
			patchToSeamVertices[patch.idxPatch].push_back(v);
	}

#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(static, 1)
	for (int i = 0; i < (int)numPatches; ++i) {
#else
	for (uint32_t i = 0; i < numPatches; ++i) {
#endif
		const uint32_t idxPatch = (uint32_t)i;
		const TexturePatch& texturePatch = texturePatches[idxPatch];
		const Image8U3& image0 = images[texturePatch.label].image;

		// Extract image region
		Image32F3 image, imageOrg;
		image0(texturePatch.rect).convertTo(image, CV_32FC3, 1.0f / 255.0f);
		image.copyTo(imageOrg);

		// Create coverage mask
		Image8U mask(image.size());
		mask.memset(0);

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
				for (uint32_t idxVertPatch1 = 0; idxVertPatch1 < seamVertex0.patches.size(); ++idxVertPatch1) {
					if (idxVertPatch1 == idxVertPatch0)
						continue;

					const SeamVertex::Patch& patch1 = seamVertex0.patches[idxVertPatch1];

					// === INLINE FIND: patch1.edges.Find(edge0.idxSeamVertex)
					const auto& edges = patch1.edges;
					uint32_t idxEdge1 = SeamVertex::Patch::Edges::NO_INDEX;
					for (uint32_t k = 0, n = (uint32_t)edges.size(); k < n; ++k) {
						if (edges[k].idxSeamVertex == edge0.idxSeamVertex) {
							idxEdge1 = k;
							break;
						}
					}
					if (idxEdge1 == SeamVertex::Patch::Edges::NO_INDEX)
						continue;

					const TexCoord& p1(patch1.proj);

					// === INLINE FIND: seamVertex1.patches.Find(patch1.idxPatch)
					const uint16_t* itAdj1 =
						seamVertex1.patchIndexLookup.Find(patch1.idxPatch);

					if (itAdj1 == nullptr)
						continue;

					const uint32_t idxVertPatch1Adj = *itAdj1;

					const SeamVertex::Patch& patch1Adj = seamVertex1.patches[idxVertPatch1Adj];
					const TexCoord& p1Adj(patch1Adj.proj);

					// this is an edge separating two (valid) patches;
					// draw it on this patch as the mean color of the two patches
					const Image8U3& image1 = images[texturePatches[patch1.idxPatch].label].image;
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
							p0(_p0), p0Dir(_p0Adj - _p0), p1(_p1), p1Dir(_p1Adj - _p1), length((float)norm(p0Dir)), sampler() {
						}
						inline void operator()(const ImageRef& pt) {
							const float l((float)norm(TexCoord(pt) - p0) / length);
							// compute mean color
							const TexCoord samplePos0(p0 + p0Dir * l);
							const Color color0(image0.sample<Sampler, Color>(sampler, samplePos0));
							const TexCoord samplePos1(p1 + p1Dir * l);
							const Color color1(image1.sample<Sampler, Color>(sampler, samplePos1) / 255.f);
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

			// render vertex color
			AccumColor accumColor;
			for (const SeamVertex::Patch& patch : seamVertex0.patches) {
				const Image8U3& img = images[texturePatches[patch.idxPatch].label].image;
				accumColor.Add(img.sample<Sampler, Color>(sampler, patch.proj) / 255.f, 1.f);
			}

			const ImageRef pt(ROUND2INT(patch0.proj - offset));
			image(pt) = accumColor.Normalized();
			mask(pt) = border;
		}

		ProcessMask(mask, 20);
		PoissonBlending(imageOrg, image, mask); // , 0.8f /* JPB WIP BUG Experiment with bias */);

		// apply color correction to patch image
		cv::Mat imagePatch(image0(texturePatch.rect));
		for (int r = 0; r < image.rows; ++r) {
			Pixel8U* row = imagePatch.ptr<Pixel8U>(r);
			for (int c = 0; c < image.cols; ++c) {
				if (mask(r, c) == empty)
					continue;
				const Color& a = image(r, c);
				Pixel8U& v = row[c];
#if 1
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
#else
				v[0] = (uint8_t)CLAMP(ROUND2INT(a[0] * 255.f), 0, 255);
				v[1] = (uint8_t)CLAMP(ROUND2INT(a[1] * 255.f), 0, 255);
				v[2] = (uint8_t)CLAMP(ROUND2INT(a[2] * 255.f), 0, 255);
#endif
			}
		}
	}

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

		std::cout << "Processed rows " << y << "–" << (y + h) << std::endl;
	}
}

struct PackedRect {
	cv::Rect rect;
	int index;
};

bool PackShelf(
	int atlasW,
	int atlasH,
	const RectsBinPack::RectArr& rects,
	RectsBinPack::RectArr& outRects) {

	struct Item {
		cv::Rect r;
		int index;
	};

	std::vector<Item> items;
	items.reserve(rects.size());

	for (int i = 0; i < (int)rects.size(); ++i)
		items.push_back({ rects[i], i });

	// sort: tallest first, then widest, stable
	std::sort(items.begin(), items.end(),
		[](const Item& a, const Item& b) {
			if (a.r.height != b.r.height)
				return a.r.height > b.r.height;
			if (a.r.width != b.r.width)
				return a.r.width > b.r.width;
			return a.index < b.index;
		});

	outRects.resize(rects.size());

	int shelfY = 0;
	int shelfH = 0;
	int shelfX = 0;

	for (const Item& it : items) {
		int w = it.r.width;
		int h = it.r.height;

		// too large to ever fit
		if (w > atlasW || h > atlasH)
			return false;

		// new shelf if needed
		if (shelfX + w > atlasW) {
			shelfY += shelfH;
			shelfX = 0;
			shelfH = 0;
		}

		// no vertical space left
		if (shelfY + h > atlasH)
			return false;

		// place
		outRects[it.index] = cv::Rect(shelfX, shelfY, w, h);

		shelfX += w;
		if (h > shelfH)
			shelfH = h;
	}

	return true;
}


void MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize)
{
	// project patches in the corresponding view and compute texture-coordinates and bounding-box
	const int border(2);
	faceTexcoords.resize(faces.size() * 3);
#ifdef TEXOPT_USE_OPENMP
	const unsigned numPatches(texturePatches.size() - 1);
#pragma omp parallel for schedule(static, 1)
	for (int_t idx = 0; idx < (int_t)numPatches; ++idx) {
		TexturePatch& texturePatch = texturePatches[(uint32_t)idx];
#else
	for (TexturePatch* pTexturePatch = texturePatches.Begin(), *pTexturePatchEnd = texturePatches.End() - 1; pTexturePatch < pTexturePatchEnd; ++pTexturePatch) {
		TexturePatch& texturePatch = *pTexturePatch;
#endif
		const Image& imageData = images[texturePatch.label];
		AABB2f aabb(true);
		for (const FIndex idxFace : texturePatch.faces) {
			const Face& face = faces[idxFace];
			TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
			for (int i = 0; i < 3; ++i) {
				texcoords[i] = imageData.camera.ProjectPointP(vertices[face[i]]);
				ASSERT(imageData.image.isInsideWithBorder(texcoords[i], border));
				aabb.InsertFull(texcoords[i]);
			}
		}
		// compute relative texture coordinates
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMin)));
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMax)));
		texturePatch.rect.x = FLOOR2INT(aabb.ptMin[0]) - border;
		texturePatch.rect.y = FLOOR2INT(aabb.ptMin[1]) - border;
		texturePatch.rect.width = CEIL2INT(aabb.ptMax[0] - aabb.ptMin[0]) + border * 2;
		texturePatch.rect.height = CEIL2INT(aabb.ptMax[1] - aabb.ptMin[1]) + border * 2;
		ASSERT(imageData.image.isInside(texturePatch.rect.tl()));
		ASSERT(imageData.image.isInside(texturePatch.rect.br()));
		const TexCoord offset(texturePatch.rect.tl());
		for (const FIndex idxFace : texturePatch.faces) {
			TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
			for (int v = 0; v < 3; ++v)
				texcoords[v] -= offset;
		}
	}
	{
		// init last patch to point to a small uniform color patch
		TexturePatch& texturePatch = texturePatches.Last();
		const int sizePatch(border * 2 + 1);
		texturePatch.rect = cv::Rect(0, 0, sizePatch, sizePatch);
		for (const FIndex idxFace : texturePatch.faces) {
			TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
			for (int i = 0; i < 3; ++i)
				texcoords[i] = TexCoord(0.5f, 0.5f);
		}
	}

	// perform seam leveling
	if (texturePatches.GetSize() > 2 && (bGlobalSeamLeveling || bLocalSeamLeveling)) {
		// create seam vertices and edges
		CreateSeamVertices();

		// perform global seam leveling
		if (bGlobalSeamLeveling) {
			TD_TIMER_STARTD();
			GlobalSeamLeveling();
			DEBUG_ULTIMATE("\tglobal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}

		// perform local seam leveling
		if (bLocalSeamLeveling) {
			TD_TIMER_STARTD();
			LocalSeamLeveling();
			DEBUG_ULTIMATE("\tlocal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}
	}

	// merge texture patches with overlapping rectangles
#if 1
// merge texture patches with overlapping rectangles (optimized)
// merge texture patches with overlapping rectangles (parallel)
	{
		const int patchCount = (int)texturePatches.size();

		// --- group patches by label ---
		std::unordered_map<Label, std::vector<int>> byLabel;
		byLabel.reserve(patchCount);

		for (int i = 0; i < patchCount; ++i) {
			byLabel[texturePatches[i].label].push_back(i);
		}

		// mark patches to remove (shared, but writes are disjoint by label)
		std::vector<uint8_t> removed(patchCount, 0);

		// --- parallel over label groups ---
#pragma omp parallel
		{
			// local temp storage if needed later (not required here)

#pragma omp for schedule(static, 1)
			for (int lblIdx = 0; lblIdx < (int)byLabel.size(); ++lblIdx) {

				// iterate by index because unordered_map is not indexable
				auto it = byLabel.begin();
				std::advance(it, lblIdx);

				std::vector<int>& ids = it->second;
				if (ids.size() < 2)
					continue;

				// sort by descending area (local to this label)
				std::sort(ids.begin(), ids.end(),
					[&](int a, int b) {
						const auto& ra = texturePatches[a].rect;
						const auto& rb = texturePatches[b].rect;
						return (ra.width * ra.height) > (rb.width * rb.height);
					});

				// containment scan
				for (int bi = 0; bi < (int)ids.size(); ++bi) {
					const int i = ids[bi];
					if (removed[i])
						continue;

					TexturePatch& big = texturePatches[i];
					const auto* __restrict Rb = &big.rect;

					for (int si = bi + 1; si < (int)ids.size(); ++si) {
						const int j = ids[si];
						if (removed[j])
							continue;

						TexturePatch& small = texturePatches[j];
						const auto* __restrict Rs = &small.rect;

						// fast rejects
						if (Rs->width > Rb->width || Rs->height > Rb->height)
							continue;

						if (!RectsBinPack::IsContainedIn(*Rs, *Rb))
							continue;

						// --- merge small into big ---
						const TexCoord offset(Rs->tl() - Rb->tl());

						for (const FIndex idxFace : small.faces) {
							TexCoord* __restrict texcoords = faceTexcoords.data() + idxFace * 3;
							texcoords[0] += offset;
							texcoords[1] += offset;
							texcoords[2] += offset;
						}

						big.faces.JoinRemove(small.faces);
						removed[j] = 1;
					}
				}
			}
		}

		// --- compact texturePatches (single serial pass) ---
		{
			int write = 0;
			for (int read = 0; read < patchCount; ++read) {
				if (!removed[read]) {
					if (write != read)
						texturePatches[write] = std::move(texturePatches[read]);
					++write;
				}
			}
			texturePatches.Resize(write);
		}
	}

#else
	for (unsigned i = 0; i < texturePatches.size() - 1; ++i) {
		TexturePatch& texturePatchBig = texturePatches[i];
		for (unsigned j = 1; j < texturePatches.size(); ++j) {
			if (i == j)
				continue;
			TexturePatch& texturePatchSmall = texturePatches[j];
			if (texturePatchBig.label != texturePatchSmall.label)
				continue;
			if (!RectsBinPack::IsContainedIn(texturePatchSmall.rect, texturePatchBig.rect))
				continue;
			// translate texture coordinates
			const TexCoord offset(texturePatchSmall.rect.tl() - texturePatchBig.rect.tl());
			for (const FIndex idxFace : texturePatchSmall.faces) {
				TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
				for (int v = 0; v < 3; ++v)
					texcoords[v] += offset;
			}
			// join faces lists
			texturePatchBig.faces.JoinRemove(texturePatchSmall.faces);
			// remove the small patch
			texturePatches.RemoveAtMove(j--);
		}
	}
#endif

	// create texture
	{
		// arrange texture patches to fit the smallest possible texture image
		RectsBinPack::RectArr rects(texturePatches.GetSize());
		FOREACH(i, texturePatches)
			rects[i] = texturePatches[i].rect;
		int textureSize(RectsBinPack::ComputeTextureSize(rects, nTextureSizeMultiple));
		// increase texture size till all patches fit
		while (true) {
			TD_TIMER_STARTD();
			bool bPacked(false);
			const unsigned typeRectsBinPack(nRectPackingHeuristic / 100);
			const unsigned typeSplit((nRectPackingHeuristic - typeRectsBinPack * 100) / 10);
			const unsigned typeHeuristic(nRectPackingHeuristic % 10);
			switch (typeRectsBinPack) {
			case 0: {
				MaxRectsBinPack pack(textureSize, textureSize);
				bPacked = pack.Insert(rects, (MaxRectsBinPack::FreeRectChoiceHeuristic)typeHeuristic);
				break;
			}
			case 1: {
#if 1
				RectsBinPack::RectArr packedRects;
				bPacked = PackShelf(textureSize, textureSize, rects, packedRects);

				if (bPacked) {
					rects.swap(packedRects);
				}
#else
				SkylineBinPack pack(textureSize, textureSize, typeSplit != 0);
				bPacked = pack.Insert(rects, (SkylineBinPack::LevelChoiceHeuristic)typeHeuristic);
#endif
				break;
			}
			case 2: {
				GuillotineBinPack pack(textureSize, textureSize);
				bPacked = pack.Insert(rects, false, (GuillotineBinPack::FreeRectChoiceHeuristic)typeHeuristic, (GuillotineBinPack::GuillotineSplitHeuristic)typeSplit);
				break;
			}
			default:
				ABORT("error: unknown RectsBinPack type");
			}
			DEBUG_ULTIMATE("\tpacking texture completed: %u patches, %u texture-size (%s)", rects.size(), textureSize, TD_TIMER_GET_FMT().c_str());
			if (bPacked)
				break;
			textureSize *= 2;
		}
		// create texture image
		textureDiffuse.create(textureSize, textureSize);
		textureDiffuse.setTo(cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));
#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(static, 1)
		for (int_t i = 0; i < (int_t)texturePatches.size(); ++i) {
			const uint32_t idxPatch((uint32_t)i);
#else
		FOREACH(idxPatch, texturePatches) {
#endif
			const TexturePatch& texturePatch = texturePatches[idxPatch];
			const RectsBinPack::Rect& rect = rects[idxPatch];
			// copy patch image
			ASSERT((rect.width == texturePatch.rect.width && rect.height == texturePatch.rect.height) ||
				(rect.height == texturePatch.rect.width && rect.width == texturePatch.rect.height));
			int x(0), y(1);
			if (texturePatch.label != NO_ID) {
				const Image& imageData = images[texturePatch.label];
				cv::Mat patch(imageData.image(texturePatch.rect));
				if (rect.width != texturePatch.rect.width) {
					// flip patch and texture-coordinates
					patch = patch.t();
					x = 1; y = 0;
				}
				patch.copyTo(textureDiffuse(rect));
			}
			// compute final texture coordinates
			const TexCoord offset(rect.tl());
			for (const FIndex idxFace : texturePatch.faces) {
				TexCoord* texcoords = faceTexcoords.data() + idxFace * 3;
				for (int v = 0; v < 3; ++v) {
					TexCoord& texcoord = texcoords[v];
					texcoord = TexCoord(
						texcoord[x] + offset.x,
						texcoord[y] + offset.y
					);
				}
			}
		}

		// apply some sharpening
#if 1
		if (fSharpnessWeight > 0.0f) {
			cv::Mat small;
			cv::resize(
				textureDiffuse,
				small,
				cv::Size(),
				0.25, 0.25,
				cv::INTER_AREA
			);

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
#else
		if (fSharpnessWeight > 0) {
			constexpr double sigma = 1.5;
			Image8U3 blurryTextureDiffuse;
			cv::GaussianBlur(textureDiffuse, blurryTextureDiffuse, cv::Size(), sigma);
			cv::addWeighted(textureDiffuse, 1 + fSharpnessWeight, blurryTextureDiffuse, -fSharpnessWeight, 0, textureDiffuse);
		}
#endif

		const int nMaxTextureSize = 60000;
		// only downscale if needed
		if (textureDiffuse.cols > nMaxTextureSize || textureDiffuse.rows > nMaxTextureSize)
		{
			const double scale = static_cast<double>(nMaxTextureSize) /
				static_cast<double>(std::max(textureDiffuse.cols, textureDiffuse.rows));

			cv::Mat original = textureDiffuse.clone();
			resizeHugeImage(original, textureDiffuse, nMaxTextureSize);

			// *** rescale UVs to match the resized atlas ***
			for (TexCoord& uv : faceTexcoords) {
				uv.x *= static_cast<float>(scale);
				uv.y *= static_cast<float>(scale);
			}
		}
	}
}

// texture mesh
//  - minCommonCameras: generate texture patches using virtual faces composed of coplanar triangles sharing at least this number of views (0 - disabled, 3 - good value)
//  - fSharpnessWeight: sharpness weight to be applied on the texture (0 - disabled, 0.5 - good value)
bool Scene::TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness,
	bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight,
	int nMaxTextureSize, const IIndexArr& views)
{
	MeshTexture texture(*this, nResolutionLevel, nMinResolution);

	// assign the best view to each face
	{
		TD_TIMER_STARTD();
		if (!texture.FaceViewSelection(minCommonCameras, fOutlierThreshold, fRatioDataSmoothness, views))
			return false;
		DEBUG_EXTRA("Assigning the best view to each face completed: %u faces (%s)", mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// generate the texture image and atlas
	{
		TD_TIMER_STARTD();
		texture.GenerateTexture(bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty, fSharpnessWeight, nMaxTextureSize);
		DEBUG_EXTRA("Generating texture atlas and image completed: %u patches, %u image size (%s)", texture.texturePatches.GetSize(), mesh.textureDiffuse.width(), TD_TIMER_GET_FMT().c_str());
	}

	return true;
} // TextureMesh
/*----------------------------------------------------------------*/
