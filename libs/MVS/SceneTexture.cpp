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

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define TEXOPT_USE_OPENMP
#endif

#define FASTER_OUTLIER_DETECTION
#undef COUNT_ITERATIONS
#undef STATS

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
#include "../Math/LBP.h"
namespace MVS {
typedef LBPInference::NodeID NodeID;
// Potts model as smoothness function
LBPInference::EnergyType STCALL SmoothnessPotts(LBPInference::NodeID, LBPInference::NodeID, LBPInference::LabelID l1, LBPInference::LabelID l2) {
	return l1 == l2 && l1 != 0 && l2 != 0 ? LBPInference::EnergyType(0) : LBPInference::EnergyType(LBPInference::MaxEnergy);
}

static LBPInference::EnergyType gSmoothnessWeight = 1000;

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

	// Gate: below this, do not encourage sameness at all
	// Raising this reduces merging (more patches); lowering increases merging (fewer patches).
	const float cosGate = 0.8f;

	if (cosAngle < cosGate) {
		cosAngle = 0.0f;
	}
	else {
		cosAngle = (cosAngle - cosGate) / (1.0f - cosGate);
	}

	// Baseline penalty for switching labels (prevents checkerboard fragmentation)
	// Increase -> fewer patches; decrease -> more patches.
	const float switchPenalty = 2.0f;

	// Angle-weighted additional penalty
	const float w = switchPenalty + (float)gSmoothnessWeight * cosAngle;

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
		Color color;
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
	cv::setNumThreads(1);

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
		for (auto idxFace : cameraFaces) {
			const Face& facet = faces[idxFace];
			rasterer.idxFace = idxFace;
			rasterer.Project(facet);
		}
		// compute the projection area of visible faces
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		CLISTDEF0IDX(uint32_t, FIndex) areas(faces.GetSize());
		areas.Memset(0);
#endif

		tsl::robin_map<uint32_t, FaceAccum> acc;
		acc.reserve(cameraFaces.size()); // decent heuristic

		for (int j = 0; j < faceMap.rows; ++j) {
			const FIndex* __restrict fm = faceMap.ptr<FIndex>(j);
			const real* __restrict gm = imageGradMag.ptr<real>(j);
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
			const auto* __restrict im = imageData.image.ptr<decltype(imageData.image)::value_type>(j);
#endif

			for (int i = 0; i < faceMap.cols; ++i) {
				const uint32_t f = (uint32_t)fm[i];
				if (f == (uint32_t)NO_ID) {
					continue;
				}

				auto ins = acc.emplace(f, FaceAccum{});
				FaceAccum& a = ins.first.value();

				if (ins.second) {
					a.quality = (float)gm[i];
					a.area = 1;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					a.color = Color(im[i]);
#endif
				}
				else {
					a.quality += (float)gm[i];
					a.area++;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					a.color += Color(im[i]);
#endif
				}
			}
		}

		// ---- angle adjustment per face (same as your logic) ----
		// Build output list for this view:
		std::vector<FaceOut> out;
		out.reserve(acc.size());
		for (auto it = acc.begin(); it != acc.end(); ++it) {
			const FIndex idxFace = it.key();
			FaceAccum& a = it.value();

			const Face& f = faces[idxFace];
			const auto& faceCenter = gFaceCenter[idxFace];
			const Point3f camDir(Cast<Mesh::Type>(imageData.camera.C) - faceCenter);
			const Normal& faceNormal = scene.mesh.faceNormals[idxFace];
			const float cosFaceCam(MAXF(0.001f, ComputeAngle(camDir.ptr(), faceNormal.ptr())));
			a.quality *= SQUARE(cosFaceCam);

			FaceOut fo;
			fo.idxFace = idxFace;
			fo.data.idxView = idxView;
			fo.data.quality = a.quality;
#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
			Color c = a.color * (1.f / (float)a.area);
			fo.data.color = RGB2YCBCR(Color(c));
#endif
			out.push_back(fo);
		}

		perViewOut[(size_t)idx].swap(out);

		++progress;
	}

#ifdef TEXOPT_USE_OPENMP
	if (bAbort)
		return false;
#endif
	// Pass 1: count how many FaceOut entries per face
	std::vector<uint32_t> addCount(facesDatas.GetSize(), 0);

	for (size_t idx = 0; idx < perViewOut.size(); ++idx) {
		const auto& out = perViewOut[idx];
		for (const FaceOut& fo : out) {
			++addCount[(size_t)fo.idxFace];
		}
	}

	// Pass 2: reserve capacity in each FaceDataArr once
	for (size_t f = 0; f < (size_t)facesDatas.GetSize(); ++f) {
		const uint32_t nAdd = addCount[f];
		if (nAdd == 0) {
			continue;
		}
		FaceDataArr& arr = facesDatas[(FIndex)f];
		// Reserve exactly current size + new adds.
		// Use whatever your container's reserve is called (Reserve / reserve / EnsureCapacity).
		arr.Reserve(arr.GetSize() + nAdd);
	}

	// Pass 3: your original merge (unchanged)
	for (size_t idx = 0; idx < perViewOut.size(); ++idx) {
		const auto& out = perViewOut[idx];
		for (const FaceOut& fo : out) {
			FaceDataArr& arr = facesDatas[fo.idxFace];
			FaceData& fd = arr.AddEmpty();
			fd = fo.data;
		}
	}

	progress.close();

	// Restore cv's ability to thread.
	cv::setNumThreads(prevCvThreads);

#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		// try to detect outlier views for each face
		// (views for which the face is occluded by a dynamic object in the scene, ex. pedestrians)
		for (FaceDataArr& faceDatas : facesDatas)
			FaceOutlierDetection(faceDatas, fOutlierThreshold);
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
				LBPInference inference;
				{
					inference.SetNumNodes(numFaces);

					gSceneForSmoothness = &scene;
					inference.SetSmoothCost(SmoothnessPottsStrong);

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

				inference.Optimize();

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
		{
			const float normalThreshold = 0.6f;   // cosine threshold
			const int maxPasses = 3;              // controlled expansion depth

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
		}

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

			// Speckles caused by patch fragmentation.
			// Large flat areas can contain many small patches which can produce contrast
			// variation.

			components.resize(numFaces);

			for (FIndex f = 0; f < numFaces; ++f) {
				if (labels[f] == NO_ID)	{
					components[f] = -1;
					continue;
				}

				components[f] =
					compGeom[f] * numLabels + labels[f];
			}

			FIndex nComponents =
				nGeomComponents * numLabels;

			texturePatches.clear();
			texturePatches.reserve(nComponents);

			std::unordered_map<int, int> compToPatch;

			for (FIndex f = 0; f < numFaces; ++f)	{
				if (labels[f] == NO_ID)
					continue;

				int compID = components[f];

				auto it = compToPatch.find(compID);

				if (it == compToPatch.end()) {
					TexturePatch patch;
					patch.label = labels[f];

					int patchIdx = (int)texturePatches.size();
					texturePatches.push_back(std::move(patch));
					compToPatch[compID] = patchIdx;

					it = compToPatch.find(compID);
				}

				texturePatches[it->second].faces.Insert(f);
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

void MeshTexture::GlobalSeamLeveling()
{
	ASSERT(!seamVertices.empty());
	const unsigned numPatches(texturePatches.size());

	// ------------------------------------------------------------
	// Build vertex -> patch adjacency without critical section
	// ------------------------------------------------------------

	const int T = omp_get_max_threads();
	const size_t numVertices = vertices.size();

	std::vector<std::vector<uint32_t>> vertexPatches(numVertices);

	// Per-thread buckets: [thread][vertex] -> list of patches
	std::vector<std::vector<std::vector<uint32_t>>> threadBuckets(
		T, std::vector<std::vector<uint32_t>>(numVertices)
	);

	// Parallel fill (no locking)
#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		auto& local = threadBuckets[tid];

#pragma omp for schedule(static)
		for (int64_t p = 0; p < (int64_t)texturePatches.size(); ++p) {
			if (texturePatches[p].label == NO_ID)
				continue;

			for (const FIndex f : texturePatches[p].faces) {
				const Face& face = faces[f];

				local[face[0]].push_back((uint32_t)p);
				local[face[1]].push_back((uint32_t)p);
				local[face[2]].push_back((uint32_t)p);
			}
		}
	}

	for (auto& vp : vertexPatches)
		vp.reserve(4);

	// Merge phase (single-threaded, no locks needed)
	for (size_t v = 0; v < numVertices; ++v) {
		auto& dst = vertexPatches[v];

		for (int t = 0; t < T; ++t) {
			auto& src = threadBuckets[t][v];
			if (!src.empty()) {
				dst.insert(dst.end(), src.begin(), src.end());
			}
		}
	}

#pragma omp parallel for
	for (int v = 0; v < vertexPatches.size(); ++v) {
		auto& vp = vertexPatches[v];
		if (vp.empty()) continue;
		std::sort(vp.begin(), vp.end());
		vp.erase(std::unique(vp.begin(), vp.end()), vp.end());
	}

	// assign a row index within the solution vector x to each vertex/patch
	std::vector<VertexPatchRow> vertpatch2rows(vertices.size());

	std::vector<MatIdx> counts(vertices.size());

#pragma omp parallel for
	for (int v = 0; v < vertices.size(); ++v)
		counts[v] = vertexPatches[v].size();

	MatIdx rowsX = 0;
	for (size_t v = 0; v < counts.size(); ++v) {
		MatIdx c = counts[v];
		counts[v] = rowsX;
		rowsX += c;
	}

#pragma omp parallel for
	for (int v = 0; v < vertices.size(); ++v) {
		auto& dst = vertpatch2rows[v];
		const auto& vp = vertexPatches[v];

		dst.patches = vp;
		dst.rows.resize(vp.size());

		MatIdx base = counts[v];
		for (size_t i = 0; i < vp.size(); ++i)
			dst.rows[i] = base + i;
	}

	// fill Tikhonov's Gamma matrix (regularization constraints)
	const float lambda = 0.1f;
	const float w = lambda * lambda;

	CLISTDEF0(MatEntry) AtATriplets;
	AtATriplets.Reserve(vertices.size() * 8);

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

				const uint32_t label =
					texturePatches[patch0.idxPatch].label;

				SampleImage sampler(
					images[label].image
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

	// -------- SERIAL MERGE --------

	for (int t = 0; t < numThreads; ++t) {
		auto& localTriplets = threadTriplets[t];
		const IDX n = localTriplets.GetSize();

		AtATriplets.Reserve(AtATriplets.GetSize() + n);

		for (IDX i = 0; i < n; ++i)
			AtATriplets.Insert(localTriplets[i]);

		for (auto& acc : threadAtb[t]) {
			Atb(acc.first, 0) += acc.second[0];
			Atb(acc.first, 1) += acc.second[1];
			Atb(acc.first, 2) += acc.second[2];
		}
	}

	const float eps = 1e-6f;

	for (MatIdx i = 0; i < rowsX; ++i)
		AtATriplets.emplace_back(i, i, eps);

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
		static thread_local std::vector<MatIdx> faceRowIndices;

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

		static thread_local ColorMap imageAdj;
		if (imageAdj.size() != texturePatch.rect.size())
			imageAdj.create(texturePatch.rect.size());

		// Faster than custom memset if contiguous
		std::memset(imageAdj.data, 0,
			imageAdj.width() * imageAdj.height() * sizeof(Color));

		static thread_local std::vector<uint8_t> mask;
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
		const Image& img = images[texturePatch.label];

		cv::Rect roi = texturePatch.rect;

		const int imgW = img.image.cols;
		const int imgH = img.image.rows;

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
		cv::Mat image(img.image(roi));
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

	std::vector<int> frontier;
	std::vector<int> nextFrontier;
	frontier.reserve(width * 4);        // heuristic
	nextFrontier.reserve(width * 4);

	std::vector<uint8_t> visited(nPix, 0);

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
	static thread_local TImage<MatIdx> indices;

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

	static thread_local std::vector<PoissonStencil> stencil;
	static thread_local std::vector<float> xR, xG, xB;
	static thread_local std::vector<float> bR, bG, bB;
	static thread_local std::vector<MatIdx> redInterior;
	static thread_local std::vector<MatIdx> blackInterior;

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
		25 /* JPB WIP BUG On 400 */       // max iters
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

	cv::setNumThreads(1); // Disable OpenCV threading

	// Better to fully thread
  // Notice, we must use dynamic scheduling here as patch size can vary a lot and cause load imbalance.
	const int T = omp_get_max_threads();

	std::vector<Image32F3> tlsImage((size_t)T);
	std::vector<Image32F3> tlsImageOrg((size_t)T);
	std::vector<Image8U> tlsMask((size_t)T);

#pragma omp parallel for schedule(dynamic, 4)
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

		const Image8U3& image0 = images[texturePatch.label].image;
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
								images[texturePatches[patch1.idxPatch].label].image;

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
							images[texturePatches[patch1.idxPatch].label].image;

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
				const Image8U3& img = images[texturePatches[patch.idxPatch].label].image;
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
			for (int c = 0; c < image.cols; ++c) {
				if (mask(r, c) == empty)
					continue;
				const Color& a = image(r, c);
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

	cv::setNumThreads(-1); // Restore OpenCV threading

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

void MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int nMaxTextureSize)
{
	// project patches in the corresponding view and compute texture-coordinates and bounding-box
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
		int x0 = FLOOR2INT(aabb.ptMin[0]) - border;
		int y0 = FLOOR2INT(aabb.ptMin[1]) - border;
		int x1 = CEIL2INT(aabb.ptMax[0]) + border;
		int y1 = CEIL2INT(aabb.ptMax[1]) + border;

		// Clamp BEFORE building rect
		x0 = std::max(0, x0);
		y0 = std::max(0, y0);
		x1 = std::min(imageData.image.cols, x1);
		y1 = std::min(imageData.image.rows, y1);

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

	// perform seam leveling
	// (is introducing a green tint).
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

		textureDiffuse.create(atlasH, atlasW);
		textureDiffuse.setTo(cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));

#ifdef TEXOPT_USE_OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
		for (int_t i = 0; i < (int_t)texturePatches.size(); ++i)
		{
			const TexturePatch& texturePatch = texturePatches[i];
			const RectsBinPack::Rect& rect = rects[i];

			int x = 0;
			int y = 1;

			const Image& imageData = images[texturePatch.label];

			// Clamp ROI to image bounds BEFORE making cv::Mat(roi)
			cv::Rect roi = texturePatch.rect;

			const int imgW = imageData.image.cols;
			const int imgH = imageData.image.rows;

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
				continue;
			}

			cv::Mat patch(imageData.image(roi));

			if (rect.width != roi.width) {
				patch = patch.t();
				x = 1;
				y = 0;
			}

			patch.copyTo(textureDiffuse(rect));

#if 0
			// --- DEBUG: color atlas by camera index instead of copying image ---
			cv::Scalar debugColor(
				(texturePatch.label * 53) % 255,
				(texturePatch.label * 97) % 255,
				(texturePatch.label * 193) % 255
			);

			textureDiffuse(rect).setTo(debugColor);
#endif

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

		// 4) Enforce max size BEFORE allocating huge texture
		const int nMaxTextureSize = 60000;

		if (std::max(textureDiffuse.cols, textureDiffuse.rows) > nMaxTextureSize) {
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

	MeshTexture::LabelArr labels;

	// assign the best view to each face
	{
		TD_TIMER_STARTD();
		if (!texture.FaceViewSelection(labels, minCommonCameras, fOutlierThreshold, fRatioDataSmoothness, views))
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
