/*
* SceneRefine.cpp
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

using namespace MVS;

#pragma optimize("", on) // JPB WIP BUG
// D E F I N E S ///////////////////////////////////////////////////

#define RASSERT(cond, fmt, ...)                                           \
    do {                                                                    \
      if (!(cond)) {                                                        \
        DEBUG("ASSERT FAILED: %s\n  File: %s:%d\n  Function: %s\n  " fmt "\n",\
              #cond, __FILE__, __LINE__, __func__, ##__VA_ARGS__);         \
        __debugbreak();                                                     \
      }                                                                     \
    } while (0)

#undef VALIDATE_GRADIENT

constexpr int TILEX = 16;
constexpr int TILEY = 16;

// uncomment to ensure edge size and improve vertex valence
// (should enable more stable flow)
#define MESHOPT_ENSUREEDGESIZE 1 // 0 - at all resolution

// uncomment to use constant z-buffer bias
// (should be enough, as the numerical error does not depend on the depth)
#define MESHOPT_DEPTHCONSTBIAS 0.05f

// uncomment to enable memory pool
// (should reduce the allocation times for frequent used images)
#define MESHOPT_TYPEPOOL

// uncomment to enable CERES optimization module
// (similar performance with the custom minimizer)
#ifdef _USE_CERES
#define MESHOPT_CERES
#endif

#ifdef MESHOPT_TYPEPOOL
#define DEC_BitMatrix(var)		BitMatrix& var = *BitMatrixPool()
#define DEC_Image(type, var)	TImage<type>& var = *ImagePool<type>()
#define DST_BitMatrix(var)		BitMatrixPool(&(var))
#define DST_Image(var)			ImagePool(&(var))
#else
#define DEC_BitMatrix(var)		BitMatrix var;
#define DEC_Image(type, var)	TImage<type> var;
#define DST_BitMatrix(var)
#define DST_Image(var)
#endif

// choose a scale so gradients (usually within ±2) fit in int16
constexpr float kScale = 1024.0f;
constexpr float kInvScale = 1.0f / kScale;

#pragma intrinsic(_InterlockedCompareExchange)
static inline float AtomicAddFloat(float* addr, float val)
{
	static_assert(sizeof(float) == sizeof(LONG), "float and LONG must be same size");

	volatile LONG* intAddr = reinterpret_cast<volatile LONG*>(addr);
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
		oldInt = prev; // retry with updated oldInt
	}

	return oldVal;
}


// S T R U C T S ///////////////////////////////////////////////////

typedef float Real;
typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;

struct CameraRenderData;

class MeshRefine {
public:
	typedef TPoint3<Real> Grad;
	using GradArr = std::vector<Grad>;

	typedef TImage<cuint32_t> FaceMap;
	typedef TImage<Point3f> BaryMap;

	struct VGrad
	{
		VIndex idx;    // vertex index (usually uint32_t or int)
		Grad   g;      // accumulated gradient, 16-byte padded
	};

	// store necessary data about a view
	struct View {
		typedef TPoint2<float> Grad;
		typedef TImage<Grad> ImageGrad;
		Image32F image; // image pixels
		ImageGrad imageGrad; // image pixel gradients
		//TImage<Real> imageMean; // image pixels mean
		//TImage<Real> imageVar; // image pixels variance
		Point3f* ray = nullptr;
		Point3f* X = nullptr;
		//std::vector<float, AlignedAllocator<float, 16>> Nd;
		float* invNd = nullptr;
		Point3f* storedNormal = nullptr;
		Point3f* bary = nullptr;
		int16_t* gradBlockInt16 = nullptr;
		cuint32_t* verticesPerPix = nullptr;
		Normal* facesNormalPerPix = nullptr;
		std::vector<uint8_t, AlignedAllocator<uint8_t, 16>> isValid;
		FaceMap faceMap; // remember for each pixel what face projects there
		DepthMap depthMap; // depth-map
		BaryMap baryMap; // barycentric coordinates
		int width, height;
		int blockWidth, blockStride;
		int allocatedSize = 0;
		std::vector<float> tileEnergyAccum;  // accumulated across threads
		std::vector<uint8_t> tileActive;     // used for skipping
		int tilesX, tilesY;
		//std::vector<uint8_t> marks;
		//uint8_t currentMark;
	};
	typedef CLISTDEF2(View) ViewsArr;

	// used to render a mesh for optimization
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		BaryMap& baryMap;
		FIndex idxFace;
		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap, BaryMap& _baryMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap), baryMap(_baryMap) {
		}
		void Clear() {
			Base::Clear();
			faceMap.fill(NO_ID);
			baryMap.memset(0);
		}
		void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
			const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
			const Depth z(ComputeDepth(t, pbary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = idxFace;
				baryMap(pt) = pbary;
			}
		}
	};


public:
	MeshRefine(Scene& _scene, unsigned _nReduceMemory, unsigned _nAlternatePair = true, Real _weightRegularity = 1.5f, Real _ratioRigidityElasticity = 0.8f, unsigned _nResolutionLevel = 0, unsigned _nMinResolution = 640, unsigned nMaxViews = 8, unsigned nMaxThreads = 1);
	~MeshRefine();

	bool IsValid() const { return !pairs.IsEmpty(); }

	bool InitImages(Real scale, Real sigma = 0);

	void ListVertexFacesPre();
	void ListVertexFacesPost();
	void ListCameraFaces(bool rebuildOctree = true);

	void ListFaceAreas(Mesh::AreaArr& maxAreas);
	void SubdivideMesh(uint32_t maxArea, float fDecimate = 1.f, unsigned nCloseHoles = 15, unsigned nEnsureEdgeSize = 1);

	double ScoreMesh(float* gradients, bool rebuildOctree = true);

	// given a vertex position and a projection camera, compute the projected position and its derivative
	template <typename TP, typename TX, typename T, typename TJ>
	static T ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian = NULL);

	static bool IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z);
	void MeshRefine::ProjectMesh(
		View& view,
		const Camera& camera,
		const CameraRenderData& rd);
	static void ImageMeshWarp(
		const View& viewA,
		const DepthMap& depthMapA, const Camera& cameraA,
		const DepthMap& depthMapB, const Camera& cameraB,
		const Image32F& imageB, TImage<uint16_t>& imageAB, std::vector<uint8_t>& mask);
	static void ComputeLocalVariance(
		const Image32F& image, const  std::vector<uint8_t>& mask,
		TImage<uint16_t>& imageMean, TImage<Real>& imageVar);
	static void ComputeLocalVariance2(
		const TImage<uint16_t>& image,               // now can be CV_16U
		const std::vector<uint8_t>& mask,
		TImage<uint16_t>& imageMean,
		TImage<Real>& imageVar);
	static float ComputeLocalZNCC(
		const Image32F& imageA, const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
		const TImage<uint16_t>& imageB, const TImage<uint16_t>& imageMeanB, const TImage<Real>& imageVarB,
		const std::vector<uint8_t>& mask, TImage<Real>& imageDZNCC);
	static void ComputePhotometricGradient(
		const View& viewA,
		const Camera& cameraA,
		const Camera& cameraB,
		const View& viewB,
		const TImage<Real>& imageDZNCC,
		const  std::vector<uint8_t>& mask,
		GradArr& photoGrad,
		std::vector<uint64_t>& photoGradNorm,
		Real RegularizationScale,
		std::vector<float>& tileEnergyLocal);
	static float ComputeSmoothnessGradient1(
		const Mesh::VertexArr& vertices, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
		GradArr& smoothGrad1, VIndex idxStart, VIndex idxEnd);
	static void ComputeSmoothnessGradient2(
		const GradArr& smoothGrad1, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
		GradArr& smoothGrad2, VIndex idxStart, VIndex idxEnd);
	template<typename TYPE>
	static TYPE* TypePool(TYPE* = NULL);
	template<typename TYPE>
	static inline TImage<TYPE>* ImagePool(TImage<TYPE>* pImage = NULL) { return TypePool< TImage<TYPE> >(pImage); }
	static inline BitMatrix* BitMatrixPool(BitMatrix* pMask = NULL) { return TypePool<BitMatrix>(pMask); }

	static void* ThreadWorkerTmp(void*);
	void ThreadWorker();
	void WaitThreadWorkers(size_t nJobs);
	void ThSelectNeighbors(uint32_t idxImage, std::unordered_set<uint64_t>& mapPairs, unsigned nMaxViews);
	void ThInitImage(uint32_t idxImage, Real scale, Real sigma);
	void ThProjectMesh(uint32_t idxImage, const Mesh::FaceIdxArr& cameraFaces, const CameraRenderData& rd);
	void ThProcessPair(uint32_t idxImageA, uint32_t idxImageB, GradArr& localGrad, std::vector<uint32_t>& threadNorm, std::vector<std::vector<float>>& tileEnergyLocal);
	void ThSmoothVertices1(VIndex idxStart, VIndex idxEnd);
	void ThSmoothVertices2(VIndex idxStart, VIndex idxEnd);

public:
	const Real weightRegularity; // a scalar regularity weight to balance between photo-consistency and regularization terms
	Real ratioRigidityElasticity; // a scalar ratio used to compute the regularity gradient as a combination of rigidity and elasticity
	const unsigned nResolutionLevel; // how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // how many times to scale down the images before mesh optimization
	const unsigned nReduceMemory; // recompute image mean and variance in order to reduce memory requirements
	unsigned nAlternatePair; // using an image pair alternatively as reference image (0 - both, 1 - alternate, 2 - only left, 3 - only right)
	unsigned iteration; // current refinement iteration
	double photoEnergyLast = 0.0;

	Scene& scene; // the mesh vertices and faces

	// gradient related
	float scorePhoto;
	float scoreSmooth;
	GradArr photoGrad;
	FloatArr photoGradNorm;
	FloatArr vertexDepth;
	GradArr smoothGrad1;
	GradArr smoothGrad2;

	// valid after ListCameraFaces()
	Mesh::NormalArr& faceNormals; // normals corresponding to each face

	// valid the entire time, but changes
	Mesh::VertexArr& vertices;
	Mesh::FaceArr& faces;
	Mesh::VertexVerticesArr& vertexVertices; // for each vertex, the list of adjacent vertices
	Mesh::VertexFacesArr& vertexFaces; // for each vertex, the list of faces containing it
	BoolArr& vertexBoundary; // for each vertex, stores if it is at the boundary or not

	// constant the entire time
	ImageArr& images;
	ViewsArr views; // views' data
	PairIdxArr pairs; // image pairs used to refine the mesh

	// multi-threading
	static SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
	static SEACAVE::cList<SEACAVE::Thread> threads; // worker threads
	static CriticalSection cs; // mutex
	static Semaphore sem; // signal job end

	enum { HalfSize = 3 }; // half window size used to compute ZNCC
};

// call with empty parameter to get an unused image;
// call with an image pointer retrieved earlier to signal that is not needed anymore
template<typename TYPE>
TYPE* MeshRefine::TypePool(TYPE* pObj)
{
	typedef CAutoPtr<TYPE> TypePtr;
	static CriticalSection cs;
	static cList<TypePtr, TYPE*> objects;
	static cList<TYPE*, TYPE*, 0> unused;
	Lock l(cs);
	if (pObj == NULL) {
		if (unused.IsEmpty())
			return objects.AddConstruct(new TYPE);
		return unused.RemoveTail();
	}
	else {
		ASSERT(objects.Find(pObj) != NO_IDX);
		ASSERT(unused.Find(pObj) == NO_IDX);
		unused.Insert(pObj);
		return NULL;
	}
}


enum EVENT_TYPE {
	EVT_JOB = 0,
	EVT_CLOSE,
};

class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};
class EVTSelectNeighbors : public Event
{
public:
	uint32_t idxImage;
	std::unordered_set<uint64_t>& mapPairs;
	unsigned nMaxViews;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSelectNeighbors(idxImage, mapPairs, nMaxViews);
		return true;
	}
	EVTSelectNeighbors(uint32_t _idxImage, std::unordered_set<uint64_t>& _mapPairs, unsigned _nMaxViews) : Event(EVT_JOB), idxImage(_idxImage), mapPairs(_mapPairs), nMaxViews(_nMaxViews) {}
};
class EVTInitImage : public Event
{
public:
	uint32_t idxImage;
	Real scale, sigma;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThInitImage(idxImage, scale, sigma);
		return true;
	}
	EVTInitImage(uint32_t _idxImage, Real _scale, Real _sigma) : Event(EVT_JOB), idxImage(_idxImage), scale(_scale), sigma(_sigma) {}
};
class EVTProjectMesh : public Event
{
public:
	uint32_t idxImage;
	const Mesh::FaceIdxArr& cameraFaces;
	const CameraRenderData& rd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProjectMesh(idxImage, cameraFaces, rd);
		return true;
	}
	EVTProjectMesh(uint32_t _idxImage, const Mesh::FaceIdxArr& _cameraFaces, const CameraRenderData& _rd) : Event(EVT_JOB), idxImage(_idxImage), cameraFaces(_cameraFaces), rd(_rd) {}
};
#if 0
class EVTProcessPair : public Event
{
public:
	uint32_t idxImageA, idxImageB;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProcessPair(idxImageA, idxImageB);
		return true;
	}
	EVTProcessPair(uint32_t _idxImageA, uint32_t _idxImageB) : Event(EVT_JOB), idxImageA(_idxImageA), idxImageB(_idxImageB) {}
};
#endif
class EVTSmoothVertices1 : public Event
{
public:
	VIndex idxStart, idxEnd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSmoothVertices1(idxStart, idxEnd);
		return true;
	}
	EVTSmoothVertices1(VIndex _idxStart, VIndex _idxEnd) : Event(EVT_JOB), idxStart(_idxStart), idxEnd(_idxEnd) {}
};
class EVTSmoothVertices2 : public Event
{
public:
	VIndex idxStart, idxEnd;
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThSmoothVertices2(idxStart, idxEnd);
		return true;
	}
	EVTSmoothVertices2(VIndex _idxStart, VIndex _idxEnd) : Event(EVT_JOB), idxStart(_idxStart), idxEnd(_idxEnd) {}
};

SEACAVE::EventQueue MeshRefine::events;
SEACAVE::cList<SEACAVE::Thread> MeshRefine::threads;
CriticalSection MeshRefine::cs;
Semaphore MeshRefine::sem;

MeshRefine::MeshRefine(Scene& _scene, unsigned _nReduceMemory, unsigned _nAlternatePair, Real _weightRegularity, Real _ratioRigidityElasticity, unsigned _nResolutionLevel, unsigned _nMinResolution, unsigned nMaxViews, unsigned nMaxThreads)
	:
	weightRegularity(_weightRegularity),
	ratioRigidityElasticity(_ratioRigidityElasticity),
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution),
	nReduceMemory(_nReduceMemory),
	nAlternatePair(_nAlternatePair),
	scene(_scene),
	faceNormals(_scene.mesh.faceNormals),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	vertexVertices(_scene.mesh.vertexVertices),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	images(_scene.images)
{
	// start worker threads
	ASSERT(nMaxThreads > 0);
	ASSERT(threads.IsEmpty());
	threads.Resize(nMaxThreads);
	FOREACHPTR(pThread, threads)
		pThread->start(ThreadWorkerTmp, this);
	// keep only best neighbor views for each image
	std::unordered_set<uint64_t> mapPairs;
	mapPairs.reserve(images.GetSize() * nMaxViews);
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTSelectNeighbors(idxImage, mapPairs, nMaxViews));
	WaitThreadWorkers(images.GetSize());
	pairs.Reserve(mapPairs.size());
	for (uint64_t pair : mapPairs)
		pairs.AddConstruct(pair);
}
MeshRefine::~MeshRefine()
{
	// wait for the working threads to close
	FOREACH(i, threads)
		events.AddEvent(new EVTClose());
	FOREACHPTR(pThread, threads)
		pThread->join();
	scene.mesh.ReleaseExtra();
}

// load and initialize all images at the given scale
// and compute the gradient for each input image
// optional: blur them using the given sigma
bool MeshRefine::InitImages(Real scale, Real sigma)
{
	views.Resize(images.GetSize());
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTInitImage(idxImage, scale, sigma));
	WaitThreadWorkers(images.GetSize());
	iteration = 0;
	return true;
}

// extract array of triangles incident to each vertex
// and check each vertex if it is at the boundary or not
void MeshRefine::ListVertexFacesPre()
{
	scene.mesh.EmptyExtra();
	scene.mesh.ListIncidenteFaces();
}
void MeshRefine::ListVertexFacesPost()
{
	scene.mesh.ListIncidenteVertices();
	scene.mesh.ListBoundaryVertices();
}

struct CamVert {
	float x, y, z, invZ;
};

struct CameraRenderData {
	std::vector<CamVert> verts;       // per-camera compact camera-space vertices
	std::vector<Face> faces;          // compact faces using LOCAL vertex indices
	std::vector<MeshRefine::Grad> normals; // per-camera normals (aligned with faces)
	std::vector<uint32_t> globalFace; // localFaceIndex -> global face index
	std::vector<uint32_t> globalVert; // localVertIndex -> global vertex index
};

std::vector<CamVert> camVerts;
std::vector<uint32_t> mapIndex;

void PreprocessCameraFaces(
	const Mesh::FaceIdxArr& cameraFaces,   // global face indices visible to this camera
	const Mesh::FaceArr& faces,            // global mesh faces
	const Mesh::VertexArr& vertices,       // global mesh vertices (world space)
	const Mesh::NormalArr& faceNormals,    // global face normals
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
	used.reserve(numFaces * 2);   // realistic estimate

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
	out.normals.resize(numFaces);

	size_t idx = 0;
	for (uint32_t fIdx : cameraFaces) {
		const Face& gf = faces[fIdx];

		// Localize face (remap global indices)
		Face& lf = out.faces[idx];
		lf[0] = remap[gf[0]];
		lf[1] = remap[gf[1]];
		lf[2] = remap[gf[2]];

		out.globalFace[idx] = fIdx;

		// Optional: pre-transform normals into camera space
		// (enable this if you want faster Nd computation downstream)
#if 0
		const auto& N = faceNormals[fIdx];
		out.normals[idx] = TransformNormalToCamera(N, camera);
#else
		out.normals[idx] = faceNormals[fIdx];
#endif

		idx++;
	}
}

void UpdateCameraVertsAndNormals(
	const Mesh::VertexArr& vertices,     // global world vertices
	const Camera& camera,
	const Mesh::NormalArr& faceNormals,  // updated global normals
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

	// ---------------------------------------------------------------
	// 2. Recompute per-camera normals using updated global normals
	// ---------------------------------------------------------------
	for (size_t fi = 0; fi < numFaces; ++fi) {
		uint32_t gf = out.globalFace[fi];  // global face index
		out.normals[fi] = faceNormals[gf];
	}
}

// extract array of faces viewed by each image
void MeshRefine::ListCameraFaces(bool rebuildOctree)
{
	// JPB WIP BUG Restrict multithreading?

	// extract array of faces viewed by each camera
	typedef CLISTDEF2(Mesh::FaceIdxArr) CameraFacesArr;

	static CameraFacesArr arrCameraFaces;
	static std::vector<CameraRenderData> cameraData;
	const int64_t numImages = (int64_t)images.GetSize();
	cameraData.resize(numImages);

	if (rebuildOctree) {
		arrCameraFaces.resize(images.GetSize());
		for (auto& cameraFaces : arrCameraFaces) // Never drop capacity in the inner vectors.
			cameraFaces.clear();

		// Build octree
		{
			Mesh::Octree octree;
			Mesh::FacesInserter::CreateOctree(octree, scene.mesh);
			FOREACH(ID, images) {
				const Image& imageData = images[ID];
				if (!imageData.IsValid())
					continue;
				typedef TFrustum<float, 5> Frustum;
				const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
				Mesh::FacesInserter inserter(arrCameraFaces[ID]);
				octree.Traverse(frustum, inserter);
			}
		}

		scene.mesh.ComputeNormalFaces(); // global normals initial

		// Build per-camera faces.
#pragma omp parallel for
		for (int64_t ID = 0; ID < numImages; ++ID) {
			if (!images[ID].IsValid()) continue;

			const auto& cam = images[ID].camera;

			PreprocessCameraFaces(
				arrCameraFaces[ID],
				faces,
				scene.mesh.vertices,
				scene.mesh.faceNormals,
				cameraData[ID]
			);
		}
	}

	// Compute global face normals
	scene.mesh.ComputeNormalFaces();

#pragma omp parallel for
	for (int64_t ID = 0; ID < numImages; ++ID) {
		if (!images[ID].IsValid()) continue;

		const auto& cam = images[ID].camera;

		// Recompute ONLY the camera-space vertices + normals
		// Faces and remapping remain intact
		UpdateCameraVertsAndNormals(
			scene.mesh.vertices,
			images[ID].camera,
			scene.mesh.faceNormals,
			cameraData[ID]
		);
	}

	// project mesh to each camera plane
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTProjectMesh(idxImage, arrCameraFaces[idxImage], cameraData[idxImage]));
	WaitThreadWorkers(images.GetSize());
}

__forceinline void AtomicInc16(uint16_t& x) noexcept
{
	_InterlockedIncrement16(reinterpret_cast<volatile SHORT*>(&x));
}

__forceinline void AtomicMin16(uint16_t& dest, uint16_t value) noexcept
{
	volatile SHORT* ptr = reinterpret_cast<volatile SHORT*>(&dest);
	SHORT old = *ptr;
	while (old > (SHORT)value) {
		SHORT prev = _InterlockedCompareExchange16(ptr, (SHORT)value, old);
		if (prev == old) break;
		old = prev;
	}
}

// compute for each face the projection area as the maximum area in both images of a pair
// (make sure ListCameraFaces() was called before)
void MeshRefine::ListFaceAreas(Mesh::AreaArr& maxAreas)
{
#if 1
	// for each image, compute the projection area of visible faces
	typedef cList<Mesh::AreaArr> ImageAreaArr;
	ImageAreaArr viewAreas(images.GetSize());
#pragma omp parallel for schedule(dynamic, 1)
	for (int idxImage = 0; idxImage < (int)images.GetSize(); ++idxImage) {
		const Image& imageData = images[idxImage];
		if (!imageData.IsValid())
			continue;
		Mesh::AreaArr& areas = viewAreas[idxImage];
		areas.Resize(faces.GetSize());
		areas.Memset(0);

		const FaceMap& faceMap = views[idxImage].faceMap;
		for (int j = 0; j < faceMap.rows; ++j) {
			const FIndex* facePtr = faceMap.ptr<FIndex>(j);
			for (int i = 0; i < faceMap.cols; ++i) {
				const FIndex idxFace(facePtr[i]);
				if (idxFace == NO_ID)
					continue;
				AtomicInc16(areas[idxFace]); // <-- thread-safe increment
			}
		}
	}

	// for each pair, mark the faces that have big projection areas in both images
	maxAreas.Resize(faces.GetSize());
	maxAreas.Memset(0);

#pragma omp parallel for
	for (int p = 0; p < (int)pairs.size(); ++p) {
		const auto& pair = pairs[p];
		const auto& areasA = viewAreas[pair.i];
		const auto& areasB = viewAreas[pair.j];
		const size_t n = areasA.size();

		for (size_t f = 0; f < n; ++f) {
			const uint16_t v = std::min(areasA[f], areasB[f]);
			AtomicMin16(maxAreas[f], v);
		}
	}
#else
	// original
	ASSERT(maxAreas.IsEmpty());
	// for each image, compute the projection area of visible faces
	typedef cList<Mesh::AreaArr> ImageAreaArr;
	ImageAreaArr viewAreas(images.GetSize());
	FOREACH(idxImage, images) {
		const Image& imageData = images[idxImage];
		if (!imageData.IsValid())
			continue;
		Mesh::AreaArr& areas = viewAreas[idxImage];
		areas.Resize(faces.GetSize());
		areas.Memset(0);
		const FaceMap& faceMap = views[idxImage].faceMap;
		// compute area covered by all vertices (incident faces) viewed by this image
		for (int j = 0; j < faceMap.rows; ++j) {
			for (int i = 0; i < faceMap.cols; ++i) {
				const FIndex idxFace(faceMap(j, i));
				ASSERT((idxFace == NO_ID && views[idxImage].depthMap(j, i) == 0) || (idxFace != NO_ID && views[idxImage].depthMap(j, i) > 0));
				if (idxFace == NO_ID)
					continue;
				++areas[idxFace];
			}
		}
	}

	maxAreas.Resize(faces.GetSize());
	maxAreas.Memset(0);
	FOREACHPTR(pPair, pairs) {
		const Mesh::AreaArr& areasA = viewAreas[pPair->i];
		const Mesh::AreaArr& areasB = viewAreas[pPair->j];
		ASSERT(areasA.GetSize() == areasB.GetSize());
		FOREACH(f, areasA) {
			const uint16_t minArea(MINF(areasA[f], areasB[f]));
			uint16_t& maxArea = maxAreas[f];
			if (maxArea < minArea)
				maxArea = minArea;
		}
	}
#endif
}

// decimate or subdivide mesh such that for each face there is no image pair in which
// its projection area is bigger than the given number of pixels in both images
void MeshRefine::SubdivideMesh(uint32_t maxArea, float fDecimate, unsigned nCloseHoles, unsigned nEnsureEdgeSize)
{
	Mesh::AreaArr maxAreas;

	// first decimate if necessary
	const bool bNoDecimation(fDecimate >= 1.f);
	const bool bNoSimplification(maxArea == 0);
	if (!bNoDecimation) {
		if (fDecimate > 0.f) {
			// decimate to the desired resolution
			scene.mesh.Clean(fDecimate, 0.f, false, nCloseHoles, 0u, 0.f, false);
			scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);

#ifdef MESHOPT_ENSUREEDGESIZE
			// make sure there are no edges too small or too long
			if (nEnsureEdgeSize > 0 && bNoSimplification) {
				scene.mesh.EnsureEdgeSize();
				scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);
			}
#endif

			// re-map vertex and camera faces
			ListVertexFacesPre();
		}
		else {
			// extract array of faces viewed by each camera
			ListCameraFaces();

			// estimate the faces' area that have big projection areas in both images of a pair
			ListFaceAreas(maxAreas);
			ASSERT(!maxAreas.IsEmpty());

			const float fMaxArea((float)(maxArea > 0 ? maxArea : 64));
			const float fMedianArea(6.f * (float)Mesh::AreaArr(maxAreas).GetMedian());
			if (fMedianArea < fMaxArea) {
				maxAreas.Empty();

				// decimate to the auto detected resolution
				scene.mesh.Clean(MAXF(0.1f, fMedianArea / fMaxArea), 0.f, false, nCloseHoles, 0u, 0.f, false);
				scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);

#ifdef MESHOPT_ENSUREEDGESIZE
				// make sure there are no edges too small or too long
				if (nEnsureEdgeSize > 0 && bNoSimplification) {
					scene.mesh.EnsureEdgeSize();
					scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);
				}
#endif

				// re-map vertex and camera faces
				ListVertexFacesPre();
			}
		}
	}
	if (bNoSimplification)
		return;

	if (maxAreas.IsEmpty()) {
		// extract array of faces viewed by each camera
		ListCameraFaces();

		// estimate the faces' area that have big projection areas in both images of a pair
		ListFaceAreas(maxAreas);
	}

	// subdivide mesh faces if its projection area is bigger than the given number of pixels
	const size_t numVertsOld(vertices.GetSize());
	const size_t numFacesOld(faces.GetSize());
	scene.mesh.Subdivide(maxAreas, maxArea);

#ifdef MESHOPT_ENSUREEDGESIZE
	// make sure there are no edges too small or too long
#if MESHOPT_ENSUREEDGESIZE==1
	if ((nEnsureEdgeSize == 1 && !bNoDecimation) || nEnsureEdgeSize > 1)
#endif
	{
		scene.mesh.EnsureEdgeSize();
		scene.mesh.Clean(1.f, 0.f, false, nCloseHoles, 0u, 0.f, true);
	}
#endif

	// re-map vertex and camera faces
	ListVertexFacesPre();

	DEBUG_EXTRA("Mesh subdivided: %u/%u -> %u/%u vertices/faces", numVertsOld, numFacesOld, vertices.GetSize(), faces.GetSize());

#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 3)
		scene.mesh.Save(MAKE_PATH("MeshSubdivided.ply"));
#endif
}


// score mesh using photo-consistency
// and compute vertices gradient using analytical method
double MeshRefine::ScoreMesh(float* gradients, bool rebuildOctree)
{
	// extract array of faces viewed by each camera
	ListCameraFaces(rebuildOctree);

	int64_t numImages = (int64_t) images.size();
#pragma omp parallel for
	for (int64_t ID = 0; ID < numImages; ++ID)
	{
		if (!images[ID].IsValid()) continue;

		View& view = views[ID];

		view.tilesX = (view.width + TILEX - 1) / TILEX;
		view.tilesY = (view.height + TILEY - 1) / TILEY;
		size_t numTiles = view.tilesX * view.tilesY;

		// Reset accumulators (to be filled during ScoreMesh)
		view.tileEnergyAccum.assign(numTiles, 0.f);

		// Only reset tileActive when relevant
		if (iteration == 0 || rebuildOctree) {
			if (view.tileActive.size() != numTiles) {
				view.tileActive.assign(numTiles, 1);
			}
			else {
				std::fill(view.tileActive.begin(), view.tileActive.end(), 1);
			}
		}
	}

	// JPB WIP BUG Nneded twice?
	//scene.mesh.ComputeNormalFaces();

	// for each pair of images, compute a photo-consistency score
	// between the reference image and the pixels of the second image
	// projected in the reference image through the mesh surface
	scorePhoto = 0;
	photoGrad.assign(vertices.GetSize(), Grad(0, 0, 0));
	photoGradNorm.Resize(vertices.GetSize());
	photoGradNorm.Memset(0);
	if (!vertexDepth.IsEmpty()) {
		ASSERT(vertexDepth.GetSize() == vertices.GetSize());
		vertexDepth.MemsetValue(FLT_MAX);
	}

	// JPB WIP BUG Fix this since it is doing pairs of pairs:

#if 1
  // No benefit to trying to reduce i->j and j->i pairs separately.
	// No benefit to grouping i->j i->j2 i->j3... 
#pragma omp parallel
	{
		static thread_local std::vector<std::vector<float>> tileEnergyLocal;
		// Allocate per-thread tileEnergyLocal
		// One-time resize per thread
		if (tileEnergyLocal.size() != views.size()) {
			tileEnergyLocal.resize(views.size());
		}

		// Ensure each view has correctly sized tile buffer
		for (size_t v = 0; v < views.size(); v++) {
			size_t tcount = views[v].tilesX * views[v].tilesY;
			if (tileEnergyLocal[v].size() != tcount)
				tileEnergyLocal[v].assign(tcount, 0.f);
			else
				std::fill(tileEnergyLocal[v].begin(), tileEnergyLocal[v].end(), 0.f);
		}

		static thread_local GradArr localGrad;
		static thread_local std::vector<uint32_t> localNorm;

		// This will be maintained as zero on thread exit.
		const int numVerts = (int)photoGrad.size();
		if (localGrad.empty()) {
			localGrad.resize(numVerts, Grad(0, 0, 0));
			localNorm.resize(numVerts, 0);
		}

		// JPB WIP BUG There is more we can do here with pairi/j
		// Each thread processes its subset of pairs dynamically
#pragma omp for schedule(dynamic)
		for (int i = 0; i < (int)pairs.GetSize(); ++i) {
			const auto& pair = pairs[i];
			ThProcessPair(pair.j, pair.i, localGrad, localNorm, tileEnergyLocal);
			ThProcessPair(pair.i, pair.j, localGrad, localNorm, tileEnergyLocal);
		}

		// Barrier implicit here at end of 'omp for'
#pragma omp for schedule(static)
		for (int v = 0; v < numVerts; ++v) {
			photoGrad[v] += localGrad[v];
			photoGradNorm[v] += localNorm[v];
			localGrad[v] = Grad(0, 0, 0);
			localNorm[v] = 0;
		}

		// -----------------------------
		// REDUCE TILE ENERGIES
		// -----------------------------
#pragma omp for schedule(static)
		for (int vi = 0; vi < (int)views.size(); ++vi) {
			View& vw = views[vi];
			size_t T = vw.tilesX * vw.tilesY;

			for (size_t t = 0; t < T; ++t) {
				vw.tileEnergyAccum[t] += tileEnergyLocal[vi][t];
				tileEnergyLocal[vi][t] = 0.f; // reset for next iteration
			}
		}
	} // end omp parallel

	// --------------------------------------------------
	// DIAGNOSTICS: accumulate photometric energy
	// (must be done BEFORE clearing tileEnergyAccum)
	// --------------------------------------------------
	double photoEnergyIter = 0.0;
	for (size_t vid = 0; vid < views.size(); ++vid) {
		const View& view = views[vid];
		for (float e : view.tileEnergyAccum)
			photoEnergyIter += e;
	}

	photoEnergyLast = photoEnergyIter;

	// -----------------------------------------
	// Update tileActive for the NEXT iteration
	// -----------------------------------------
// Start conservative, get aggressive as we converge
	float tileThreshold = (iteration < 3) ? 5e-4f : 1e-3f;

	for (size_t vid = 0; vid < views.size(); ++vid)	{
		View& view = views[vid];
		size_t tcount = view.tilesX * view.tilesY;

		for (size_t t = 0; t < tcount; ++t) {
			float E = view.tileEnergyAccum[t];

			view.tileActive[t] = (E > tileThreshold ? 1 : 0);

			// Clear AFTER energy was captured
			view.tileEnergyAccum[t] = 0.f;
		}
	}
#else

	ASSERT(events.IsEmpty());
	FOREACHPTR(pPair, pairs) {
		ASSERT(pPair->i < pPair->j);
		switch (nAlternatePair) {
		case 1:
			events.AddEvent(iteration % 2 ? new EVTProcessPair(pPair->j, pPair->i) : new EVTProcessPair(pPair->i, pPair->j));
			break;
		case 2:
			events.AddEvent(new EVTProcessPair(pPair->i, pPair->j));
			break;
		case 3:
			events.AddEvent(new EVTProcessPair(pPair->j, pPair->i));
			break;
		default:
			for (int ip = 0; ip < 2; ++ip)
				events.AddEvent(ip ? new EVTProcessPair(pPair->j, pPair->i) : new EVTProcessPair(pPair->i, pPair->j));
		}
	}
	WaitThreadWorkers(nAlternatePair ? pairs.GetSize() : pairs.GetSize() * 2);
#endif

	// loop through all vertices and compute the smoothing score
	scoreSmooth = 0;
	const VIndex idxStep((vertices.GetSize() + (VIndex)threads.GetSize() - 1) / (VIndex)threads.GetSize());
	smoothGrad1.resize(vertices.GetSize());
	{
		ASSERT(events.IsEmpty());
		VIndex idx(0);
		while (idx < vertices.GetSize()) {
			const VIndex idxNext(MINF(idx + idxStep, vertices.GetSize()));
			events.AddEvent(new EVTSmoothVertices1(idx, idxNext));
			idx = idxNext;
		}
		WaitThreadWorkers(threads.GetSize());
	}
	// loop through all vertices and compute the smoothing gradient
	smoothGrad2.resize(vertices.GetSize());
	{
		ASSERT(events.IsEmpty());
		VIndex idx(0);
		while (idx < vertices.GetSize()) {
			const VIndex idxNext(MINF(idx + idxStep, vertices.GetSize()));
			events.AddEvent(new EVTSmoothVertices2(idx, idxNext));
			idx = idxNext;
		}
		WaitThreadWorkers(threads.GetSize());
	}

	// set the final gradient as the combination of photometric and smoothness gradients
	if (ratioRigidityElasticity >= 1.f) {
		FOREACH(v, vertices)
			((Point3f*)gradients)[v] = photoGradNorm[v] > 0 ?
			Cast<float>(photoGrad[v] / photoGradNorm[v] + smoothGrad2[v] * weightRegularity) :
			Cast<float>(smoothGrad2[v] * weightRegularity);
	} else {
		// compute smoothing gradient as a combination of level 1 and 2 of the Laplacian operator;
		// (see page 105 of "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004)
		const Real rigidity((Real(1) - ratioRigidityElasticity) * weightRegularity);
		const Real elasticity(ratioRigidityElasticity * weightRegularity);
		FOREACH(v, vertices)
			((Point3f*)gradients)[v] = photoGradNorm[v] > 0 ?
			Cast<float>(photoGrad[v] / photoGradNorm[v] + smoothGrad2[v] * elasticity - smoothGrad1[v] * rigidity) :
			Cast<float>(smoothGrad2[v] * elasticity - smoothGrad1[v] * rigidity);
	}
	return (nAlternatePair ? 0.2f : 0.1f) * scorePhoto + 0.01f * scoreSmooth;
}


// given a vertex position and a projection camera, compute the projected position and its derivative
// returns the depth
template <typename TP, typename TX, typename T, typename TJ>
T MeshRefine::ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian)
{
	const TX& x1(X[0]);
	const TX& x2(X[1]);
	const TX& x3(X[2]);

	const TP& p1_1(P[0]);
	const TP& p1_2(P[1]);
	const TP& p1_3(P[2]);
	const TP& p1_4(P[3]);
	const TP& p2_1(P[4]);
	const TP& p2_2(P[5]);
	const TP& p2_3(P[6]);
	const TP& p2_4(P[7]);
	const TP& p3_1(P[8]);
	const TP& p3_2(P[9]);
	const TP& p3_3(P[10]);
	const TP& p3_4(P[11]);

	const TP t5(p3_4 + p3_1 * x1 + p3_2 * x2 + p3_3 * x3);
	const TP t6(1.0 / t5);
	const TP t10(p1_4 + p1_1 * x1 + p1_2 * x2 + p1_3 * x3);
	const TP t11(t10 * t6);
	const TP t15(p2_4 + p2_1 * x1 + p2_2 * x2 + p2_3 * x3);
	const TP t16(t15 * t6);
	x[0] = T(t11);
	x[1] = T(t16);
	if (jacobian) {
		jacobian[0] = TJ((p1_1 - p3_1 * t11) * t6);
		jacobian[1] = TJ((p1_2 - p3_2 * t11) * t6);
		jacobian[2] = TJ((p1_3 - p3_3 * t11) * t6);
		jacobian[3] = TJ((p2_1 - p3_1 * t16) * t6);
		jacobian[4] = TJ((p2_2 - p3_2 * t16) * t6);
		jacobian[5] = TJ((p2_3 - p3_3 * t16) * t6);
	}
	return T(t5);
}


// check if any of the depths surrounding the given coordinate is similar to the given value
bool MeshRefine::IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z)
{
	const ImageRef tl(FLOOR2INT(pt));
	for (int x = 0; x < 2; ++x) {
		for (int y = 0; y < 2; ++y) {
			const ImageRef ir(tl.x + x, tl.y + y);
			if (!depthMap.isInsideWithBorder<int, 3>(ir))
				continue;
			const Depth& depth = depthMap(ir);
#ifndef MESHOPT_DEPTHCONSTBIAS
			if (depth <= 0 || ABS(depth - z) > z * 0.01f /*!IsDepthSimilar(depth, z, 0.01f)*/)
#else
			if (depth <= 0 || depth + MESHOPT_DEPTHCONSTBIAS < z)
#endif
				continue;
			return true;
		}
	}
	return false;
}

#undef VALIDATE_RASTERIZER
#undef VALIDATE_COUNT

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

#undef INVARIANT1
#undef INVARIANT2

// project mesh to the given camera plane
void MeshRefine::ProjectMesh(
	View& view,
	const Camera& camera,
	const CameraRenderData& rd)
{
	DepthMap& depthMap = view.depthMap;
	FaceMap& faceMap = view.faceMap;
	BaryMap& baryMap = view.baryMap;
	const auto& size = view.image.size();
	const size_t numPixels = size.width * size.height;

	static thread_local std::vector<uint16_t> depthGen;
	static thread_local uint16_t depthCurGen = 1;
		// Resize if needed
	if (depthGen.size() != numPixels) {
		depthGen.assign(numPixels, 0);
		depthCurGen = 1;
	}
		// Bump generation
	depthCurGen++;
	if (depthCurGen == 0) {
		std::fill(depthGen.begin(), depthGen.end(), 0);
		depthCurGen = 1;
	}

	// init view data
	// Maps must be completely filled or have a generator.
	depthMap.create(size);
	faceMap.create(size);
	baryMap.create(size);

#ifdef VALIDATE_RASTERIZER
	depthMap.memset(0);
	faceMap.fill(NO_ID);
	baryMap.memset(0);
#endif

	view.isValid.assign(numPixels, 0);

	//depthMap.memset(0);
	//faceMap.memset(NO_ID); JPB WIP BUG Wrong format should be fill()
	//baryMap.memset(0);

	struct Triangle {
		Point2f pti[3];
	} t;

	const int width = size.width;
	const int height = size.height;

	const float minX = 3.f;
	const float minY = 3.f;
	const float maxX = (float)(width - 4);
	const float maxY = (float)(height - 4);

	const float widthMinus1 = (float)(width - 1);
	const float heightMinus1 = (float)(height - 1);

	for (size_t fi = 0, cnt = rd.faces.size(); fi < cnt; ++fi) {
		const Face& face = rd.faces[fi];
		// ==== Camera-space vertices (pre-transformed) ====
		const CamVert& c0 = rd.verts[face[0]];
		const CamVert& c1 = rd.verts[face[1]];
		const CamVert& c2 = rd.verts[face[2]];
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

		// ---- Convert to integer bounding-box & clamp ----
		int minXi = _cvt_ftoi_fast(boxMinX);
		int minYi = _cvt_ftoi_fast(boxMinY);
		int maxXi = _cvt_ftoi_fast(boxMaxX + 1);   // faster than ceil
		int maxYi = _cvt_ftoi_fast(boxMaxY + 1);

		if (minXi < 0) minXi = 0;
		if (minYi < 0) minYi = 0;
		if (maxXi > width)  maxXi = width;
		if (maxYi > height) maxYi = height;

		ImageRef boxMinI(minXi, minYi);
		ImageRef boxMaxI(maxXi - 1, maxYi - 1);   // convert from half-open to inclusive

#ifdef INVARIANT2
		constexpr int border = 0;
		if (boxMinI.x < border)
			boxMinI.x = border;
		if (boxMinI.y < border)
			boxMinI.y = border;
		if (boxMaxI.x >= (size.width - border))
			boxMaxI.x = (size.width - (border + 1));
		if (boxMaxI.y >= (size.height - border))
			boxMaxI.y = (size.height - (border + 1));
#endif

		const float invArea = 1.f / area;

		// edge deltas
		const float w0_dx = (v2.y - v3.y) * invArea;
		const float w0_dy = (v3.x - v2.x) * invArea;
		const float w1_dx = (v3.y - v1.y) * invArea;
		const float w1_dy = (v1.x - v3.x) * invArea;
		const float w2_dx = (v1.y - v2.y) * invArea;
		const float w2_dy = (v2.x - v1.x) * invArea;

		// Original work doesn't use pixel centers - premultiply by invArea
		const float px0 = (float)boxMinI.x;
		const float py0 = (float)boxMinI.y;

		// initial barycentrics for first pixel center
		float w0_row = EdgeFunction2(v2, v3, { px0, py0 }) * invArea;
		float w1_row = EdgeFunction2(v3, v1, { px0, py0 }) * invArea;
		float w2_row = EdgeFunction2(v1, v2, { px0, py0 }) * invArea;

		// vertex depths
		const float z0 = c0.z;
		const float z1 = c1.z;
		const float z2 = c2.z;

		// reciprocal depths
		const float iz0 = c0.invZ;
		const float iz1 = c1.invZ;
		const float iz2 = c2.invZ;

		Depth* __restrict depthPtr = depthMap.ptr<float>(0);
		uint16_t* __restrict depthGenPtr = depthGen.data();
		cuint32_t* __restrict facePtr = faceMap.ptr<cuint32_t>(0);
		Point3f* __restrict baryPtr = baryMap.ptr<Point3f>(0);
		uint8_t* __restrict validPtr = view.isValid.data();

		for (size_t y = boxMinI.y; y <= boxMaxI.y; ++y) {
			size_t base = size_t(y) * width;
			uint16_t* __restrict depthGenRow = depthGenPtr + base;
			Depth* __restrict depthRow = depthPtr + base;
			cuint32_t* __restrict faceRow = facePtr + base;
			Point3f* __restrict baryRow = baryPtr + base;
			uint8_t* __restrict validRow = validPtr + base;

			float w0 = w0_row;
			float w1 = w1_row;
			float w2 = w2_row;

			// -------------------------------------------------------------------
			// Phase 1: advance until entering triangle (cheap rejects only)
			// -------------------------------------------------------------------
			int x = minXi;
			while (x <= maxXi) {
				if (w0 >= 0.f && w1 >= 0.f && w0 + w1 <= 1.f) {
					break; // found first inside pixel
				}

				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;

				++x;
			}

			if (x > maxXi)
				goto next_row; // entire row outside, skip

			// -------------------------------------------------------------------
			// Phase 2: inside span - no inside tests in main loop
			// -------------------------------------------------------------------
			for (; x <= maxXi; ++x) {
				// perspective correct barycentrics
				float denom = w0 * iz0 + w1 * iz1 + w2 * iz2;
				float invDen = 1.f / denom;

				float bx = (w0 * iz0) * invDen;
				float by = (w1 * iz1) * invDen;
				float bz = 1.f - bx - by;

				float z = bx * z0 + by * z1 + bz * z2;

				float old = (depthGenRow[x] == depthCurGen)
					? depthRow[x]
					: std::numeric_limits<float>::infinity();

				if (old > z) {
					depthGenRow[x] = depthCurGen;
					depthRow[x] = z;
					faceRow[x] = (cuint32_t)fi;
					baryRow[x] = { bx, by, bz };
					validRow[x] = 1;
				}

				// advance
				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;

				// exit span
				if (w0 < 0.f || w1 < 0.f || (w0 + w1) > 1.f)
					break;
			}

		next_row:
			w0_row += w0_dy;
			w1_row += w1_dy;
			w2_row += w2_dy;
		}
	}

#ifdef VALIDATE_RASTERIZER

	DepthMap depthMap2;
	FaceMap faceMap2;
	BaryMap baryMap2;

	depthMap2.create(size);
	faceMap2.create(size);
	baryMap2.create(size);

	depthMap2.memset(0);
	faceMap2.memset((uint8_t)NO_ID);
	baryMap2.memset(0);

	// project all triangles on this image and keep the closest ones
	RasterMesh rasterer(vertices, camera, depthMap2, faceMap2, baryMap2);
	RasterMesh::Triangle triangle;
	RasterMesh::TriangleRasterizer triangleRasterizer(triangle, rasterer);
	rasterer.Clear();
	for (auto idxFace : cameraFaces) {
		const Face& facet = faces[idxFace];
		rasterer.idxFace = idxFace;
		rasterer.Project(facet, triangleRasterizer);
	}

	const auto size2 = depthMap2.size();

	size_t diffDepthCount = 0;
	size_t diffFaceCount = 0;
	size_t diffBaryCount = 0;
	double maxDepthDiff = 0.0;
	double maxBaryDiff = 0.0;

	for (int y = 0; y < size2.height; ++y) {
		for (int x = 0; x < size2.width; ++x) {
			const Depth dr = depthMap2(y, x);
			const Depth do_ = depthMap(y, x);

			// Depth difference
			if (dr != 0 || do_ != 0) {
				const double diff = fabs((double)dr - (double)do_);
				if (diff > 1e-4 && !(isnan(dr) && isnan(do_))) {
					++diffDepthCount;
					if (diff > maxDepthDiff) maxDepthDiff = diff;
				}
			}

			// Face difference
			const int fr = faceMap2(y, x);
			const int fo = faceMap(y, x);
			if (fr != fo)
				++diffFaceCount;

			// Barycentric difference
			if (fr == fo) {
				const Point3f& br = baryMap2(y, x);
				const Point3f& bo = baryMap(y, x);
				const double db0 = fabs((double)br.x - (double)bo.x);
				const double db1 = fabs((double)br.y - (double)bo.y);
				const double db2 = fabs((double)br.z - (double)bo.z);
				const double bd = std::max(db0, std::max(db1, db2));
				if (bd > 1e-3) {
					++diffBaryCount;
					if (bd > maxBaryDiff) maxBaryDiff = bd;
				}
			}
		}
	}

	static std::mutex coutMutex;
	{
		std::lock_guard<std::mutex> lock(coutMutex);

		const int total = size2.width * size2.height;
		VERBOSE("Validation results:\n");
		VERBOSE("  Depth differences: %zu / %d (%.6f%%), max delta = %.6g\n",
			diffDepthCount, total, 100.0 * diffDepthCount / total, maxDepthDiff);
		VERBOSE("  Face  differences: %zu / %d (%.6f%%)\n",
			diffFaceCount, total, 100.0 * diffFaceCount / total);
		VERBOSE("  Bary  differences: %zu / %d (%.6f%%), max delta = %.6g\n",
			diffBaryCount, total, 100.0 * diffBaryCount / total, maxBaryDiff);
	}
#endif

  // Some data is stored in tiled format for better memory access later.
	// 1. Calculate Aligned Dimensions
	// We round UP to the nearest tile boundary. 
	// e.g. if width is 100, alignedWidth becomes 128.
	const size_t tilesX = (size.width + TILEX - 1) / TILEX;
	const size_t tilesY = (size.height + TILEY - 1) / TILEY;
	const size_t alignedCount = tilesX * tilesY * TILEX * TILEY;

	const size_t rows = size.height;
	const size_t cols = size.width;
	const size_t count = rows * cols;

	if (view.allocatedSize < alignedCount) {
		// Will leak on exit.
		_aligned_free(view.ray);
		_aligned_free(view.X);
		_aligned_free(view.storedNormal);
		_aligned_free(view.invNd);
		_aligned_free(view.verticesPerPix);
		_aligned_free(view.facesNormalPerPix);
		_aligned_free(view.bary);

		view.ray = (Point3f*)_aligned_malloc(alignedCount * sizeof(Point3f), 16);
		view.X = (Point3f*)_aligned_malloc(alignedCount * sizeof(Point3f), 16);
		view.storedNormal = (Point3f*)_aligned_malloc(alignedCount * sizeof(Point3f), 16);
		view.invNd = (float*)_aligned_malloc(alignedCount * sizeof(float), 16);
		view.verticesPerPix = (cuint32_t*)_aligned_malloc(alignedCount * sizeof(cuint32_t) * 3, 16);
		view.facesNormalPerPix = (Point3f*)_aligned_malloc(alignedCount * sizeof(Point3f), 16);
		view.bary = (Point3f*)_aligned_malloc(alignedCount * sizeof(Point3f), 16);

		view.allocatedSize = alignedCount;
  }

	view.width = cols;
	view.height = rows;

#ifdef VALIDATE_COUNT
	int validCnt = 0;
#endif

	const Point3f cameraC = Cast<float>(camera.C);
	const float r00 = camera.R(0, 0), r01 = camera.R(0, 1), r02 = camera.R(0, 2);
	const float r10 = camera.R(1, 0), r11 = camera.R(1, 1), r12 = camera.R(1, 2);
	const float r20 = camera.R(2, 0), r21 = camera.R(2, 1), r22 = camera.R(2, 2);
	const float fx = camera.K(0, 0);
	const float fy = camera.K(1, 1);
	const float cx = camera.K(0, 2);
	const float cy = camera.K(1, 2);
	const float invFx = 1.0f / fx;
	const float invFy = 1.0f / fy;

#if 1
	//-------------------------------------------------------------------------
		// Tiled Iteration
		// We iterate through valid Tiles, then iterate pixels inside them.
		//-------------------------------------------------------------------------
	for (size_t ty = 0; ty < rows; ty += TILEY) {
		for (size_t tx = 0; tx < cols; tx += TILEX) {

			// Determine the start index for this specific tile in the LINEAR buffers (Output)
			// Math: (BlockRow * BlocksPerRow + BlockCol) * PixelsPerBlock
			const size_t tileIndex = (ty / TILEY) * tilesX + (tx / TILEX);
			const size_t blockStartOffset = tileIndex * (TILEX * TILEY);

			// Pointers to the START of the current tile in the OUTPUT buffers
			Point3f* __restrict tRay = &view.ray[blockStartOffset];
			Point3f* __restrict tX = &view.X[blockStartOffset];
			Point3f* __restrict tNorm = &view.storedNormal[blockStartOffset];
			float* __restrict tInvNd = &view.invNd[blockStartOffset];
			Point3f* __restrict tFaceNorm = &view.facesNormalPerPix[blockStartOffset];
			Point3f* __restrict tBary = &view.bary[blockStartOffset];

			// verticesPerPix is special because it has 3 components per pixel.
			// We offset by blockStartOffset * 3.
			cuint32_t* __restrict tVerts = &view.verticesPerPix[blockStartOffset * 3];

			// Inner Loop: 0..31
			for (size_t ly = 0; ly < TILEY; ++ly) {
				const size_t r = ty + ly;

				// Bounds check: If this tile hangs off the bottom of the image
				if (r >= rows) { break; }

				// Pointers for INPUT arrays (Linear)
				cuint32_t* __restrict faceRow = faceMap.ptr<cuint32_t>(r);
				float* __restrict depthRow = depthMap.ptr<float>(r);
				uint8_t* __restrict isValidRow = &view.isValid[r * cols];
				Point3f* __restrict baryRow = baryMap.ptr<Point3f>(r);

				const float dy = (static_cast<float>(r) - cy) * invFy;

				for (size_t lx = 0; lx < TILEX; ++lx) {
					const size_t c = tx + lx;

					// Bounds check: If this tile hangs off the right of the image
					if (c >= cols) { break; }

					// The local index inside the tile (0..1023)
					// Since we iterate ly then lx, this is sequential: 0, 1, 2...
					const size_t localIdx = ly * TILEX + lx;

					if (!isValidRow[c]) {
						faceRow[c] = NO_ID;
						continue;
					}

					// view.depthMap(r,c) guaranteed > 0
					const float depth = depthRow[c];

					// ---------------------------------------------------------------------
					// Unified Pruning Block (High-Impact Early Rejects)
					// ---------------------------------------------------------------------

					// 0. Basic depth check
					if (depth <= 0.0f) {
						isValidRow[c] = 0;
						continue;
					}

					const FIndex f = faceRow[c];  // Needed for normal check anyway

          // JPB WIP BUG Experiment with depth discontinuity culling
					// 1. Depth discontinuity pruning (A-side)
					// Adaptive threshold: smaller for close, larger for far
					const float depthThresh = 0.05f * depth + 0.01f;  // 5% + 1cm baseline
					const float normalThresh = 0.15f;  // ~81' angle change

					bool isEdge = false;

					// Combined horizontal + vertical depth check (branchless)
					if (c > 0 && c < cols - 1) {
						float dzdx = MAXF(fabs(depth - depthRow[c - 1]),
							fabs(depth - depthRow[c + 1]));

						if (r > 0 && r < rows - 1) {
							float dzdy = MAXF(fabs(depth - depthMap.ptr<float>(r - 1)[c]),
								fabs(depth - depthMap.ptr<float>(r + 1)[c]));

							// Single comparison for both axes
							if (MAXF(dzdx, dzdy) > depthThresh) {
								isEdge = true;
							}
						}
						else if (dzdx > depthThresh) {
							isEdge = true;
						}
					}

					// Normal check (only if depth check passed)
					if (!isEdge && c > 0 && c < cols - 1) {
						const FIndex fLeft = faceRow[c - 1];
						const FIndex fRight = faceRow[c + 1];

						if (isValidRow[c - 1] && isValidRow[c + 1]) {
							const FIndex fLeft = faceRow[c - 1];   // Safe: we know it's valid
							const FIndex fRight = faceRow[c + 1];  // Safe: we know it's valid

							// Now safe to access rd.normals[] with local indices
							const Grad& N = rd.normals[f];
							const float dotLeft = N.dot(rd.normals[fLeft]);
							const float dotRight = N.dot(rd.normals[fRight]);

							// Branchless: compute minimum dot product
							const float minDot = (dotLeft < dotRight ? dotLeft : dotRight);

							if (minDot < (1.0f - normalThresh)) {
								isEdge = true;
							}
						}
					}
					if (isEdge) {
						isValidRow[c] = 0;
						continue;
					}

					// Make vertical symmetric with horizontal
					if (r > 0 && r < rows - 1) {
						float dzdy_above = fabs(depth - depthMap.ptr<float>(r - 1)[c]);
						float dzdy_below = fabs(depth - depthMap.ptr<float>(r + 1)[c]);
						if (dzdy_above > depthThresh && dzdy_below > depthThresh) {
							isValidRow[c] = 0;
							continue;
						}
					}

					// *** EARLY BARYCENTRIC PRUNE *** 
					const Point3f& b = baryRow[c];
					constexpr float epsilon = -1e-7f;  // Allow tiny negatives (numerical error)
					if (b.x < epsilon || b.y < epsilon || b.z < epsilon) {
						isValidRow[c] = 0;
						continue;
					}

					// 2. Texture flatness prune (A-image)
					// Remove low-information pixels before expensive steps.
					// Check 4-connected neighborhood
					const float center = view.image(r, c);
					const float left = (c > 0) ? view.image(r, c - 1) : center;
					const float right = (c < cols - 1) ? view.image(r, c + 1) : center;
					const float up = (r > 0) ? view.image(r - 1, c) : center;
					const float down = (r < rows - 1) ? view.image(r + 1, c) : center;

					const float maxGrad = MAXF(
						MAXF(fabs(center - left), fabs(center - right)),
						MAXF(fabs(center - up), fabs(center - down))
					);

					if (maxGrad < 0.005f) {  // More permissive (0.5% instead of 1%)
						isValidRow[c] = 0;
						continue;
					}

					// At this point we know depth is OK, neighbors are OK, texture exists.
					// Proceed to compute rayW, X, dA, Nd.
					// ---------------------------------------------------------------------

					const float dx = (static_cast<float>(c) - cx) * invFx;
					const Point3f rayW(
						r00 * dx + r10 * dy + r20,
						r01 * dx + r11 * dy + r21,
						r02 * dx + r12 * dy + r22
					);

					const Point3f X = rayW * depth + cameraC;

					const float lenSq = rayW.x * rayW.x + rayW.y * rayW.y + rayW.z * rayW.z;
					const float invLen = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lenSq)));
					const Point3f dA = { rayW.x * invLen, rayW.y * invLen, rayW.z * invLen };
					const Grad& N = rd.normals[f];
					const float Nd = N.dot(dA);

          // JPB WIP BUG Experiment with less strict Nd cull (was 0.1)
					// ---------------------------------------------------------------------
					// Surface Angle & Stability Pruning
					// Nd = N · dA has already been computed here.
					// ---------------------------------------------------------------------

					// 3. Grazing angle reject (relaxed threshold)
					constexpr float minNd = -0.95f;  // ~18' from tangent (safe limit)
					if (Nd > minNd) {  // Includes grazing angles + shallow angles
						isValidRow[c] = 0;
						continue;
					}

					// 5. Compute invNd and check (compact, no-branch math)
					const float invNd = 1.0f / Nd;

					tRay[localIdx] = dA;
					tX[localIdx] = X;
					tNorm[localIdx] = N;
					tInvNd[localIdx] = invNd;

					// Must use local faces.
					const Face& face = rd.faces[f];

					// Vertices stride is 3, so we manually calc offset
					tVerts[localIdx * 3 + 0] = face[0];
					tVerts[localIdx * 3 + 1] = face[1];
					tVerts[localIdx * 3 + 2] = face[2];

					tFaceNorm[localIdx] = N;
					tBary[localIdx] = baryRow[c];

#ifdef VALIDATE_COUNT
					++validCnt;
#endif
				} // end lx
			} // end ly
		} // end tx
	} // end ty
#else
	for (size_t r = 0; r < rows; ++r) {
		cuint32_t* __restrict faceRow = faceMap.ptr<cuint32_t>(r);
		float* __restrict depthRow = depthMap.ptr<float>(r);
		uint8_t* __restrict isValidRow = &view.isValid[r * cols];
		Point3f* __restrict baryRow = baryMap.ptr<Point3f>(r);
		const float dy = (static_cast<float>(r) - cy) * invFy;  // (v - cy)/fy

		for (size_t c = 0; c < cols; ++c) {
			if (!isValidRow[c]) {
				faceRow[c] = NO_ID;
				continue;
			}

			const size_t idx = r * cols + c;

			// view.depthMap(r,c) guaranteed > 0
			const float depth = depthRow[c];

			// Unnormalized direction in camera coords (z=1), rotated to world
			// RayPoint = R^T * TransformPointI2C([ (u-cx)/fx, (v-cy)/fy, 1 ])

			//const Point3f rayW = camera.RayPoint(Point2(c, r));
			// Reconstruct 3D point in world: X = C + (rayW * depth)
			// (equivalently: X = R^T * ([x',y',1] * depth) + C)
			const float dx = (static_cast<float>(c) - cx) * invFx;  // (u - cx)/fx
			const Point3f rayW(
				r00 * dx + r10 * dy + r20,
				r01 * dx + r11 * dy + r21,
				r02 * dx + r12 * dy + r22
			);

			const Point3f X = rayW * depth + cameraC;

			// Face index for this pixel
			const FIndex f = faceRow[c];
			// view.faceMap(r,c) guaranteed not NO_ID

			// Normalized ray ONLY for Nd
			const float lenSq = rayW.x * rayW.x + rayW.y * rayW.y + rayW.z * rayW.z;
			const float invLen = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(lenSq)));
			const Point3f dA = { rayW.x * invLen, rayW.y * invLen, rayW.z * invLen };
			const Grad& N = faceNormals[f];
			const float Nd = N.dot(dA);
			if (Nd > -0.1f || (Nd == 0.f)) {
				isValidRow[c] = 0;
				continue;
			}

			// Store SoA geometry
			// normalized ray for Nd / later Jacobian use
			view.ray[idx] = dA;

			view.X[idx] = X;// full 3D point (world)
			view.storedNormal[idx] = N;
			//view.Nd[idx] = Nd;

			float invNd = 1.0f / Nd;

			if ((invNd < -10.f) || (invNd >= 0.f)) {
				isValidRow[c] = 0;
				continue;
			}

			view.invNd[idx] = invNd;

			const Face& face = faces[f];
			FIndex faceIndexes[3] = { face[0], face[1], face[2] };

			view.verticesPerPix[idx * 3 + 0] = faceIndexes[0];
			view.verticesPerPix[idx * 3 + 1] = faceIndexes[1];
			view.verticesPerPix[idx * 3 + 2] = faceIndexes[2];
			view.facesNormalPerPix[idx] = N;

			// Barycentrics if needed later
			view.bary[idx] = baryRow[c];

			// JPB WIP BUG Not needed isValidRow[c] = 1;
#ifdef VALIDATE_COUNT
			++validCnt;
#endif
		}
	}
#endif

#ifdef VALIDATE_COUNT
	{
		static std::mutex coutMutex;
		VERBOSE("View: %d valid pixels out of %d (%.2f%%)\n",
			validCnt, count, 100.f * validCnt / count);
	}
#endif
}

#if 1
#undef MESHREFINE_WARP_VALIDATE
void MeshRefine::ImageMeshWarp(
	const View& viewA,
	const DepthMap& depthMapA, const Camera& cameraA,
	const DepthMap& depthMapB, const Camera& cameraB,
	const Image32F& imageB,
	TImage<uint16_t>& imageAB,
	std::vector<uint8_t>& mask)
{
	ASSERT(!imageA.empty());

	const size_t rows = depthMapA.rows;
	const size_t cols = depthMapA.cols;

	// Camera A intrinsics
	const float fxA = (float) cameraA.K(0, 0);
	const float fyA = (float) cameraA.K(1, 1);
	const float cxA = (float) cameraA.K(0, 2);
	const float cyA = (float) cameraA.K(1, 2);
	const float rfxA = 1.f / fxA;
	const float rfyA = 1.f / fyA;

	// Camera A extrinsics (R^t and C)
	const float rA00 = (float) cameraA.R(0, 0), rA01 = (float) cameraA.R(0, 1), rA02 = (float) cameraA.R(0, 2);
	const float rA10 = (float) cameraA.R(1, 0), rA11 = (float) cameraA.R(1, 1), rA12 = (float) cameraA.R(1, 2);
	const float rA20 = (float) cameraA.R(2, 0), rA21 = (float) cameraA.R(2, 1), rA22 = (float) cameraA.R(2, 2);

	const float cAx = (float) cameraA.C.x;
	const float cAy = (float) cameraA.C.y;
	const float cAz = (float) cameraA.C.z;

	// Camera B intrinsics
	const float fxB = cameraB.K(0, 0);
	const float fyB = cameraB.K(1, 1);
	const float cxB = cameraB.K(0, 2);
	const float cyB = cameraB.K(1, 2);

	// Camera B extrinsics
	const float rB00 = (float) cameraB.R(0, 0), rB01 = (float) cameraB.R(0, 1), rB02 = (float) cameraB.R(0, 2);
	const float rB10 = (float) cameraB.R(1, 0), rB11 = (float) cameraB.R(1, 1), rB12 = (float) cameraB.R(1, 2);
	const float rB20 = (float) cameraB.R(2, 0), rB21 = (float) cameraB.R(2, 1), rB22 = (float) cameraB.R(2, 2);

	const float cBx = cameraB.C.x;
	const float cBy = cameraB.C.y;
	const float cBz = cameraB.C.z;

	// Compute M = R_B * R_A
	const float m00 = rB00 * rA00 + rB01 * rA01 + rB02 * rA02;
	const float m01 = rB00 * rA10 + rB01 * rA11 + rB02 * rA12;
	const float m02 = rB00 * rA20 + rB01 * rA21 + rB02 * rA22;

	const float m10 = rB10 * rA00 + rB11 * rA01 + rB12 * rA02;
	const float m11 = rB10 * rA10 + rB11 * rA11 + rB12 * rA12;
	const float m12 = rB10 * rA20 + rB11 * rA21 + rB12 * rA22;

	const float m20 = rB20 * rA00 + rB21 * rA01 + rB22 * rA02;
	const float m21 = rB20 * rA10 + rB21 * rA11 + rB22 * rA12;
	const float m22 = rB20 * rA20 + rB21 * rA21 + rB22 * rA22;

	// Compute world-offset transformed by R_B
	// T = R_B * (C_A - C_B)
	const float tcx = cAx - cBx;
	const float tcy = cAy - cBy;
	const float tcz = cAz - cBz;

	const float t0 = rB00 * tcx + rB01 * tcy + rB02 * tcz;
	const float t1 = rB10 * tcx + rB11 * tcy + rB12 * tcz;
	const float t2 = rB20 * tcx + rB21 * tcy + rB22 * tcz;

	boost::container::small_vector<float, 4096> xnRow(cols);
	float x = (-cxA) * rfxA;
	for (size_t i = 0; i < cols; ++i) {
		xnRow[i] = x;
		x += rfxA;
	}

	if (depthMapB.size() != imageB.size()) {
		ERROR("ImageMeshWarp: depth map B and image B have different sizes");
		return;
  }

	const float* __restrict depthMapBPtr = (float*)depthMapB.data;
	const float* __restrict imageBPtr = (float*)imageB.data;

	for (size_t j = 0; j < rows; ++j) {
		const float yn = ((float)j - cyA) * rfyA;

		const float biasX = m01 * yn + m02;
		const float biasY = m11 * yn + m12;
		const float biasZ = m21 * yn + m22;

		const float* __restrict depthRowA = depthMapA.ptr<const float>(j);
		const uint8_t* __restrict validRow = &viewA.isValid[j * cols];
		uint16_t* __restrict outRow = imageAB.ptr<uint16_t>(j);
		uint8_t* __restrict maskRow = &mask[j * cols];

		for (size_t i = 0; i < cols; ++i) {
			const float xn = xnRow[i];

			if (!validRow[i]) {
				outRow[i] = 0;
				maskRow[i] = 0;
				continue;
			}

			const float z = depthRowA[i];
			if (z <= 0.0f) {
				outRow[i] = 0;
				maskRow[i] = 0;
				continue;
			}

			// Compute Z first
			const float termZ = m20 * xn + biasZ;
			const float zcB = z * termZ + t2;

			if (zcB <= 0.0f) {
				outRow[i] = 0;
				maskRow[i] = 0;
				continue;
			}

			const float invZ = 1.0f / zcB;

			// Compute X and Y
			const float termX = m00 * xn + biasX;
			const float termY = m10 * xn + biasY;

			const float xcB = z * termX + t0;
			const float ycB = z * termY + t1;

			const float fxInv = fxB * invZ;
			const float fyInv = fyB * invZ;

			const float u = cxB + fxInv * xcB;
			const float v = cyB + fyInv * ycB;

			// Compute integer pixel index (fast trunc)
			const int x0 = _cvt_ftoi_fast(u);
			const int y0 = _cvt_ftoi_fast(v);

			// Unified OOB test
			if ((unsigned)x0 >= (unsigned)(cols - 1) ||
				(unsigned)y0 >= (unsigned)(rows - 1)) {
				outRow[i] = 0;
				maskRow[i] = 0;
				continue;
			}

			// Precompute depth tolerance
			const float thr = zcB * 0.01f;
			const float zMin = zcB - thr;
			const float zMax = zcB + thr;

			int idx = y0 * cols;

			// Depth rows (depthMapB)
			const float* __restrict dm0 = depthMapBPtr + idx;
			const float* __restrict dm1 = dm0 + cols;

			// Load 4 depths
			const float d00 = dm0[x0];
			const float d10 = dm0[x0 + 1];
			const float d01 = dm1[x0];
			const float d11 = dm1[x0 + 1];

			// Early reject: all 4 depths <= 0?
			if (d00 <= 0.f && d10 <= 0.f && d01 <= 0.f && d11 <= 0.f) {
				outRow[i] = 0;
				maskRow[i] = 0;
				continue;
			}

			// Min/max reduction (branchless)
			float dMin = d00;
			float dMax = d00;

			dMin = (d10 < dMin ? d10 : dMin);
			dMax = (d10 > dMax ? d10 : dMax);

			dMin = (d01 < dMin ? d01 : dMin);
			dMax = (d01 > dMax ? d01 : dMax);

			dMin = (d11 < dMin ? d11 : dMin);
			dMax = (d11 > dMax ? d11 : dMax);

			// Depth validity check
			if (!(dMax >= zMin && dMin <= zMax && dMax > 0.f)) {
				outRow[i] = 0;
				maskRow[i] = 0;
				continue;
			}

			// Fractional bilinear weights
			const float fx = u - (float)x0;
			const float fy = v - (float)y0;

			const float fx1 = 1.f - fx;
			const float fy1 = 1.f - fy;

			// Image rows (imageB)
			const float* i0 = imageBPtr + idx;
			const float* i1 = i0 + cols;

			// Load pixel intensities
			const float a = i0[x0];
			const float b = i0[x0 + 1];
			const float c = i1[x0];
			const float d = i1[x0 + 1];

			// Fused bilinear interpolation
			const float top = a * fx1 + b * fx;
			const float bot = c * fx1 + d * fx;

			float val = top * fy1 + bot * fy;

			// Clamp final value (like your previous code)
			if (val < 0.f) val = 0.f;
			if (val > 1.f) val = 1.f;

			outRow[i] = (uint16_t)(val * 65535.f + 0.5f);
			maskRow[i] = 1;

#ifdef MESHREFINE_WARP_VALIDATE
			{
				const Point3 XwRef = cameraA.TransformPointI2W(Point3(i, j, z));
				const Point3f XcBRef = cameraB.TransformPointW2C(XwRef);
				const Point2f uvRef = cameraB.TransformPointC2I(XcBRef);

				float valRef = imageB.sample<
					Sampler::Linear<float>,
					Sampler::Linear<float>::Type
				>(Sampler::Linear<float>(), uvRef);

				if (fabsf(uvRef.x - u) > 0.01f ||
					fabsf(uvRef.y - v) > 0.01f ||
					fabsf(valRef - val) > 0.02f) {
					DEBUG("Warp mismatch at pixel %zu,%zu\n", j, i);
				}
			}
#endif
		}
	}
}
#else
// project image from view B to view A through the mesh;
// the projected image is stored in imageA
// (imageAB is assumed to be initialize to the right size)
void MeshRefine::ImageMeshWarp(
	const View& viewA,
	const DepthMap& depthMapA, const Camera& cameraA,
	const DepthMap& depthMapB, const Camera& cameraB,
	const Image32F& imageB, TImage<uint16_t>& imageAB, std::vector<uint8_t>& mask)
{
	ASSERT(!imageA.empty());
	typedef Sampler::Linear<float> Sampler;
	const Sampler sampler;
  const size_t cols = depthMapA.cols;
  const size_t rows = depthMapA.rows;
	for (size_t j = 0; j < rows; ++j) {
    uint8_t* __restrict maskRow = &mask[j * depthMapA.cols];
		const uint8_t* __restrict isValidRow = &viewA.isValid[j * cols];
		uint16_t* __restrict outRow = imageAB.ptr<uint16_t>(j);

		for (size_t i = 0; i < depthMapA.cols; ++i) {
			if (isValidRow[i]) {
				const Depth& depthA = depthMapA(j, i);
				const Point3 X(cameraA.TransformPointI2W(Point3(i, j, depthA)));
				const Point3f ptC(cameraB.TransformPointW2C(X));
				const Point2f pt(cameraB.TransformPointC2I(ptC));
				if (!IsDepthSimilar(depthMapB, pt, ptC.z)) {
					outRow[i] = 0;
					maskRow[i] = 0;
					continue;
				}
				float v = imageB.sample<Sampler, Sampler::Type>(sampler, pt);
				if (v < 0.f) v = 0.f;
				if (v > 1.f) v = 1.f;
				outRow[i] = (uint16_t)(v * 65535.0f + 0.5f);
				maskRow[i] = 1;
			}
			else {
				outRow[i] = 0;
        maskRow[i] = 0;
			}
		}
	}
}
#endif

// compute local variance for each image pixel
void MeshRefine::ComputeLocalVariance(
	const Image32F& image,
	const std::vector<uint8_t>& mask,
	TImage<uint16_t>& imageMean,   // fixed-point 0..65535
	TImage<Real>& imageVar)
{
	ASSERT(image.size() == mask.size());
	imageMean.create(image.size());
	imageVar.create(image.size());

	const int rows = image.rows;
	const int cols = image.cols;

	const int hs = HalfSize;
	const int window = 2 * hs + 1;
	const int n = window * window;
	const Real invN = Real(1.0) / Real(n);

	// Zero borders just like OpenMVS
	imageMean.memset(0);
	imageVar.memset(0);

	if (rows == 0 || cols == 0)
		return;

	// Safe interior bounds
	const int rowStart = std::max(0, hs);
	const int rowEnd = std::min(rows, rows - hs);
	const int colStart = std::max(0, hs);
	const int colEnd = std::min(cols, cols - hs);

	if (rowStart >= rowEnd || colStart >= colEnd)
		return;  // window doesn't fit inside image

	// Thread-local buffers
	static thread_local std::vector<Real> colSum;
	static thread_local std::vector<Real> colSumSq;

	colSum.assign(cols, Real(0));
	colSumSq.assign(cols, Real(0));

	// Seed vertical window safely:
	// sum rows [0 .. min(rows-1, 2*hs)]
	const int vStart = 0;
	const int vEnd = std::min(rows - 1, 2 * hs);

	for (int rr = vStart; rr <= vEnd; ++rr) {
		const float* src = image.ptr<float>(rr);
		for (int c = 0; c < cols; ++c) {
			Real v = Real(src[c]);
			colSum[c] += v;
			colSumSq[c] += v * v;
		}
	}

	// Main scanning loop
	for (int r = rowStart; r < rowEnd; ++r) {
		// Compute actual vertical window bounds at this row
		const int vTop = std::max(0, r - hs);
		const int vBot = std::min(rows - 1, r + hs);

		// If this differs from initial window, rebuild vertical window
		if (vTop > vStart || vBot < vEnd) {

			std::fill(colSum.begin(), colSum.end(), Real(0));
			std::fill(colSumSq.begin(), colSumSq.end(), Real(0));

			for (int rr = vTop; rr <= vBot; ++rr) {
				const float* src = image.ptr<float>(rr);
				for (int c = 0; c < cols; ++c) {
					Real v = Real(src[c]);
					colSum[c] += v;
					colSumSq[c] += v * v;
				}
			}
		}

		// Build initial horizontal window at (r, colStart)
		Real winSum = Real(0);
		Real winSumSq = Real(0);

		const int hLeft = std::max(0, colStart - hs);
		const int hRight = std::min(cols - 1, colStart + hs);

		for (int cc = hLeft; cc <= hRight; ++cc) {
			winSum += colSum[cc];
			winSumSq += colSumSq[cc];
		}

		uint16_t* __restrict meanRow = imageMean.ptr<uint16_t>(r);
		Real* __restrict varRow = imageVar.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		// Slide horizontally across interior
		for (int c = colStart; c < colEnd; ++c) {
			if (maskRow[c]) {
				Real mean = winSum * invN;
				Real var = winSumSq * invN - mean * mean;

				if (var < Real(0.0001)) var = Real(0.0001);
				if (mean < Real(0)) mean = Real(0);
				if (mean > Real(1)) mean = Real(1);

				meanRow[c] = uint16_t(mean * 65535.0f + 0.5f);
				varRow[c] = var;
			}

			// Slide window one pixel right, safely
			const int addC = c + hs + 1;
			const int remC = c - hs;

			if (addC < cols) winSum += colSum[addC];
			if (remC >= 0)   winSum -= colSum[remC];

			if (addC < cols) winSumSq += colSumSq[addC];
			if (remC >= 0)   winSumSq -= colSumSq[remC];
		}

		// Slide vertical window
		const int newAddR = r + hs + 1;
		const int newRemR = r - hs;

		if (newAddR < rows && newRemR >= 0) {
			const float* __restrict addRow = image.ptr<float>(newAddR);
			const float* __restrict remRow = image.ptr<float>(newRemR);

			for (int c = 0; c < cols; ++c) {
				Real a = Real(addRow[c]);
				Real d = Real(remRow[c]);
				colSum[c] += (a - d);
				colSumSq[c] += (a * a - d * d);
			}
		}
	}
}

void MeshRefine::ComputeLocalVariance2(
	const TImage<uint16_t>& image,
	const std::vector<uint8_t>& mask,
	TImage<uint16_t>& imageMean,
	TImage<Real>& imageVar)
{
	ASSERT(image.size() == mask.size());
	imageMean.create(image.size());
	imageVar.create(image.size());

	const int rows = image.rows;
	const int cols = image.cols;

	const int hs = HalfSize;
	const int window = 2 * hs + 1;
	const int n = window * window;

	const Real invN = Real(1) / Real(n);
	const Real scale = Real(1.0 / 65535.0);

	// Zero borders for consistency with OpenMVS
	imageMean.memset(0);
	imageVar.memset(0);

	if (rows == 0 || cols == 0)
		return;

	// Clamp valid region to ensure safety even on tiny images
	const int rowStart = std::max(0, hs);
	const int rowEnd = std::min(rows, rows - hs);
	const int colStart = std::max(0, hs);
	const int colEnd = std::min(cols, cols - hs);

	if (rowStart >= rowEnd || colStart >= colEnd)
		return; // image too small for variance window

	// Thread-local column accumulators
	static thread_local std::vector<Real> colSum, colSumSq;
	colSum.assign(cols, Real(0));
	colSumSq.assign(cols, Real(0));

	// Seed vertical window safely:
	// sums rows[max(0, r-hs) .. min(rows-1, r+hs)]
	const int vertStart = 0;
	const int vertEnd = std::min(rows - 1, 2 * hs);

	for (int rr = vertStart; rr <= vertEnd; ++rr) {
		const uint16_t* src = image.ptr<uint16_t>(rr);
		for (int c = 0; c < cols; ++c) {
			Real v = Real(src[c]) * scale;
			colSum[c] += v;
			colSumSq[c] += v * v;
		}
	}

	// Main scanning loop
	for (int r = rowStart; r < rowEnd; ++r) {

		// Compute vertical window bounds
		const int vTop = std::max(0, r - hs);
		const int vBot = std::min(rows - 1, r + hs);

		// Rebuild vertical window when needed
		// (For normal OpenMVS sizes this path is never taken,
		//  but it is required to be safe on tiny images.)
		if (vTop > vertStart || vBot < vertEnd) {
			// Rebuild the vertical window from scratch safely
			std::fill(colSum.begin(), colSum.end(), Real(0));
			std::fill(colSumSq.begin(), colSumSq.end(), Real(0));

			for (int rr = vTop; rr <= vBot; ++rr) {
				const uint16_t* src = image.ptr<uint16_t>(rr);
				for (int c = 0; c < cols; ++c) {
					Real v = Real(src[c]) * scale;
					colSum[c] += v;
					colSumSq[c] += v * v;
				}
			}
		}

		// Build initial horizontal window at (r, colStart)
		Real winSum = Real(0);
		Real winSumSq = Real(0);

		int hLeft = std::max(0, colStart - hs);
		int hRight = std::min(cols - 1, colStart + hs);

		for (int cc = hLeft; cc <= hRight; ++cc) {
			winSum += colSum[cc];
			winSumSq += colSumSq[cc];
		}

		uint16_t* __restrict meanRow = imageMean.ptr<uint16_t>(r);
		Real* __restrict varRow = imageVar.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		for (int c = colStart; c < colEnd; ++c) {

			if (maskRow[c]) {
				Real mean = winSum * invN;
				Real var = winSumSq * invN - mean * mean;

				if (var < Real(0.0001)) var = Real(0.0001);
				mean = std::max<Real>(0, std::min<Real>(1, mean));

				meanRow[c] = uint16_t(mean * 65535.0f + 0.5f);
				varRow[c] = var;
			}

			// Slide window horizontally one pixel, safely
			int addC = c + hs + 1;
			int remC = c - hs;

			if (addC < cols)
				winSum += colSum[addC];
			if (remC >= 0)
				winSum -= colSum[remC];

			if (addC < cols)
				winSumSq += colSumSq[addC];
			if (remC >= 0)
				winSumSq -= colSumSq[remC];
		}

		// Slide vertical window
		int newAddR = r + hs + 1;
		int newRemR = r - hs;

		if (newAddR < rows && newRemR >= 0) {
			const uint16_t* __restrict addRow = image.ptr<uint16_t>(newAddR);
			const uint16_t* __restrict remRow = image.ptr<uint16_t>(newRemR);

			for (int c = 0; c < cols; ++c) {
				Real a = Real(addRow[c]) * scale;
				Real d = Real(remRow[c]) * scale;
				colSum[c] += (a - d);
				colSumSq[c] += (a * a - d * d);
			}
		}
	}
}

// compute local ZNCC and its gradient for each image pixel
float MeshRefine::ComputeLocalZNCC(
	const Image32F& imageA,
	const TImage<uint16_t>& imageMeanA, const TImage<Real>& imageVarA,
	const TImage<uint16_t>& imageB, const TImage<uint16_t>& imageMeanB,
	const TImage<Real>& imageVarB,
	const std::vector<uint8_t>& mask,
	TImage<Real>& imageDZNCC)
{
	ASSERT(imageA.size() == imageB.size());
	ASSERT(imageA.size() == mask.size());

	const int rows = imageA.rows;
	const int cols = imageA.cols;

	const int hs = HalfSize;
	const int rowStart = hs;
	const int rowEnd = rows - hs;
	const int colStart = hs;
	const int colEnd = cols - hs;

	const int n = (2 * hs + 1) * (2 * hs + 1);
	const float invN = 1.0f / float(n);
	const float scale16 = 1.0f / 65535.0f;

	// output
	imageDZNCC.create(rows, cols);
	// no memset needed; we only write valid pixels

	// integral buffer (thread-local to avoid alloc)
	static thread_local cv::Mat integralAB;
	integralAB.create(rows + 1, cols + 1, CV_32F);
	if (integralAB.isContinuous()) {
    memset(integralAB.ptr<float>(0), 0, (rows + 1) * (cols + 1) * sizeof(float));
	} else {
		integralAB.setTo(0);
	}

	// ----------------------------------------------------------------
	// Build integral of A * B
	// ----------------------------------------------------------------
	for (int r = 0; r < rows; ++r) {
		const float* __restrict aPtr = imageA.ptr<float>(r);
		const uint16_t* __restrict bPtr = imageB.ptr<uint16_t>(r);

		float* __restrict dst = integralAB.ptr<float>(r + 1);
		const float* __restrict prev = integralAB.ptr<float>(r);

		float acc = 0.f;
		for (int c = 0; c < cols; ++c) {
			acc += aPtr[c] * (float(bPtr[c]) * scale16);
			dst[c + 1] = prev[c + 1] + acc;
		}
	}

	// ----------------------------------------------------------------
	// Main loop: compute ZNCC & gradient in one pass
	// ----------------------------------------------------------------
	float score = 0.0f;

#if 1 //SSE2?
	for (int r = rowStart; r < rowEnd; ++r) {
		const uint8_t* __restrict maskRow = &mask[r * cols];
		const float* __restrict aRow = imageA.ptr<float>(r);
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(r);

		const float* __restrict up = integralAB.ptr<float>(r - hs);
		const float* __restrict dn = integralAB.ptr<float>(r + hs + 1);

		Real* __restrict gradRow = imageDZNCC.ptr<Real>(r);

		int c = colStart;
		while (c < colEnd) {
			// skip invalids
			while (c < colEnd && !maskRow[c]) ++c;
			if (c >= colEnd) break;

			// valid run [runStart, runEnd)
			const int runStart = c;
			while (c < colEnd && maskRow[c]) ++c;
			const int runEnd = c;

			// process 4 at a time
			int i = runStart;
			const int vecEnd = runStart + ((runEnd - runStart) & ~3);

			for (; i < vecEnd; i += 4) {
				// Gather scalars (cheaper than i32gather here)
				int x0[4], x1[4];
				float sumABv[4], meanAv[4], meanBv[4], aValv[4], bValv[4], varAv[4], varBv[4];

				for (int k = 0; k < 4; ++k) {
					const int cc = i + k;
					x0[k] = cc - hs;
					x1[k] = cc + hs + 1;

					sumABv[k] = dn[x1[k]] - dn[x0[k]] - up[x1[k]] + up[x0[k]];
					meanAv[k] = float(imageMeanA(r, cc)) * scale16;
					meanBv[k] = float(imageMeanB(r, cc)) * scale16;
					aValv[k] = aRow[cc];
					bValv[k] = float(bRow[cc]) * scale16;
					varAv[k] = float(imageVarA(r, cc));
					varBv[k] = float(imageVarB(r, cc));
				}

				// Load into SSE vectors
				__m128 cov = _mm_mul_ps(_mm_loadu_ps(sumABv), _mm_set1_ps(invN));
				__m128 meanA = _mm_loadu_ps(meanAv);
				__m128 meanB = _mm_loadu_ps(meanBv);
				__m128 aVal = _mm_loadu_ps(aValv);
				__m128 bVal = _mm_loadu_ps(bValv);
				__m128 varA = _mm_loadu_ps(varAv);
				__m128 varB = _mm_loadu_ps(varBv);

				// invS = rsqrt(varA*varB), clamp
				__m128 prod = _mm_mul_ps(varA, varB);
				__m128 clamp = _mm_max_ps(prod, _mm_set1_ps(1e-12f));
				__m128 invS = _mm_rsqrt_ps(clamp);
				// refine once: invS *= (1.5 - 0.5*x*invS^2)
				__m128 invSsq = _mm_mul_ps(invS, invS);
				__m128 refine = _mm_sub_ps(_mm_set1_ps(1.5f), _mm_mul_ps(_mm_set1_ps(0.5f), _mm_mul_ps(clamp, invSsq)));
				invS = _mm_mul_ps(invS, refine);

				// zncc = (cov - meanA*meanB) * invS
				__m128 zncc = _mm_mul_ps(_mm_sub_ps(cov, _mm_mul_ps(meanA, meanB)), invS);

				// ZNCCinvVB = zncc / varB (protect small varB)
				__m128 invVB = _mm_div_ps(_mm_set1_ps(1.0f), _mm_max_ps(varB, _mm_set1_ps(1e-12f)));
				__m128 ZNCCinvVB = _mm_mul_ps(zncc, invVB);

				// dZNCC = aVal*invS - bVal*ZNCCinvVB + meanB*ZNCCinvVB - meanA*invS
				__m128 dZNCC = _mm_sub_ps(
					_mm_add_ps(_mm_mul_ps(aVal, invS), _mm_mul_ps(meanB, ZNCCinvVB)),
					_mm_add_ps(_mm_mul_ps(bVal, ZNCCinvVB), _mm_mul_ps(meanA, invS))
				);

				// reliability = min(varA,varB)/(min+0.0015)
				__m128 minV = _mm_min_ps(varA, varB);
				__m128 reliability = _mm_div_ps(minV, _mm_add_ps(minV, _mm_set1_ps(0.0015f)));

				__m128 grad = _mm_mul_ps(_mm_set1_ps(-1.0f), _mm_mul_ps(reliability, dZNCC));
				__m128 term = _mm_mul_ps(reliability, _mm_sub_ps(_mm_set1_ps(1.0f), zncc));

				// Store and accumulate
				float gradOut[4], termOut[4];
				_mm_storeu_ps(gradOut, grad);
				_mm_storeu_ps(termOut, term);

				gradRow[i + 0] = Real(gradOut[0]);
				gradRow[i + 1] = Real(gradOut[1]);
				gradRow[i + 2] = Real(gradOut[2]);
				gradRow[i + 3] = Real(gradOut[3]);

				score += termOut[0] + termOut[1] + termOut[2] + termOut[3];
			}

			// tail
			for (; i < runEnd; ++i) {
				const int x0 = i - hs;
				const int x1 = i + hs + 1;

				const float sumAB = dn[x1] - dn[x0] - up[x1] + up[x0];
				const Real cov = Real(sumAB * invN);

				const Real meanA = Real(imageMeanA(r, i)) * scale16;
				const Real meanB = Real(imageMeanB(r, i)) * scale16;

				const Real varA = imageVarA(r, i);
				const Real varB = imageVarB(r, i);

				float x = float(varA * varB);
				if (x < 1e-12f) x = 1e-12f;
				float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
				invS = invS * (1.5f - 0.5f * x * invS * invS);

				const Real zncc = (cov - meanA * meanB) * invS;

				const Real aVal = Real(aRow[i]);
				const Real bVal = Real(bRow[i]) * scale16;

				const Real invVB = (varB > Real(1e-12) ? Real(1) / varB : Real(1e12));
				const Real ZNCCinvVB = zncc * invVB;

				const Real dZNCC =
					aVal * invS
					- bVal * ZNCCinvVB
					+ meanB * ZNCCinvVB
					- meanA * invS;

				const Real minV = (varA < varB ? varA : varB);
				const Real reliability = minV / (minV + Real(0.0015));

				gradRow[i] = -reliability * dZNCC;
				score += float(reliability * (Real(1) - zncc));
			}
		} // run
	} // rows

#else
	for (int r = rowStart; r < rowEnd; ++r) {
		const uint8_t* __restrict maskRow = &mask[r * cols];
		const float* __restrict aRow = imageA.ptr<float>(r);
		const uint16_t* __restrict bRow = imageB.ptr<uint16_t>(r);

		const float* __restrict up = integralAB.ptr<float>(r - hs);
		const float* __restrict dn = integralAB.ptr<float>(r + hs + 1);

		Real* __restrict gradRow = imageDZNCC.ptr<Real>(r);

		for (int c = colStart; c < colEnd; ++c) {
			if (!maskRow[c])
				continue;

			const int x0 = c - hs;
			const int x1 = c + hs + 1;

			// covariance over window
			const float sumAB = dn[x1] - dn[x0] - up[x1] + up[x0];
			const Real cov = Real(sumAB * invN);

			// unpack means
			const Real meanA = Real(imageMeanA(r, c)) * scale16;
			const Real meanB = Real(imageMeanB(r, c)) * scale16;

			// variances
			const Real varA = imageVarA(r, c);
			const Real varB = imageVarB(r, c);

			// inverse sqrt(varA*varB)
			float x = float(varA * varB);
			if (x < 1e-12f) x = 1e-12f;

			float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
			invS = invS * (1.5f - 0.5f * x * invS * invS);

			// ZNCC
			const Real zncc = (cov - meanA * meanB) * invS;

			// ZNCC gradient
			const Real aVal = Real(aRow[c]);
			const Real bVal = Real(bRow[c]) * scale16;

			const Real ZNCCinvVB = zncc / varB;

			const Real dZNCC =
				aVal * invS
				- bVal * ZNCCinvVB
				+ meanB * ZNCCinvVB
				- meanA * invS;

			// reliability (OpenMVS)
			const Real minV = (varA < varB ? varA : varB);
			const Real reliability = minV / (minV + Real(0.0015));

			gradRow[c] = -reliability * dZNCC;
			score += float(reliability * (Real(1) - zncc));
		}
	}
#endif

	return score;
}

#if 1

#define FMA(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)

void MeshRefine::ComputePhotometricGradient(
	const View& viewA,
	const Camera& cameraA,
	const Camera& cameraB,
	const View& viewB,
	const TImage<Real>& imageDZNCC,
	const std::vector<uint8_t>& mask,
	GradArr& threadGrad,
	std::vector<uint64_t>& localNorm,
	Real RegularizationScale,
	std::vector<float>& tileEnergyLocal)
{
	ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
	//ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());

  size_t cols = viewA.image.cols;
	const size_t RowsEnd = viewA.image.rows - HalfSize;
	const size_t ColsEnd = cols - HalfSize;

#if 1
	thread_local std::vector<uint8_t>  localGen;
	localGen.resize(threadGrad.size());
	thread_local uint8_t localMark = 0;
	if (++localMark == 0) {
		std::fill(localGen.begin(), localGen.end(), 0);
	}
#endif

	static thread_local std::vector<uint8_t> touchedVertices;
	touchedVertices.resize(threadGrad.size(), 0);

	const size_t stride = viewA.width;

	const float* P = &cameraB.Pf[0];
	const float p0 = P[0], p1 = P[1], p2 = P[2], p3 = P[3];
	const float p4 = P[4], p5 = P[5], p6 = P[6], p7 = P[7];
	const float p8 = P[8], p9 = P[9], p10 = P[10], p11 = P[11];

	const __m128 one = _mm_set1_ps(1.0f);
#if 0
	for (int ty = 0; ty < RowsEnd; ty += TILE) {
		const int yEnd = std::min(ty + TILE, RowsEnd);
		for (int tx = 0; tx < ColsEnd; tx += TILE) {
			const int xEnd = std::min(tx + TILE, ColsEnd);

			//------------------------------------------------------------------
			//  Tile inner loop
			//------------------------------------------------------------------
			for (int r = ty; r < yEnd; ++r) {
				const int base = r * stride;
				int c = tx;

				for (; c < xEnd; ++c) {
					if (!mask(r, c)) continue;
					// All image pixels are legal here; we only need to check the view marks.
					const int idx = base + c;
					const FIndex idxFace(viewA.faceMap(r, c));
					ASSERT(idxFace != NO_ID);
					const Grad N(normals[idxFace]);
					const Point3f& dA = viewA.ray[idx];
					const Point3f& storedN = viewA.storedNormal[idx];
					float Nd = viewA.Nd[idx];
					float invNd = viewA.invNd[idx];
#ifdef DEBUGPG
					if ((invNd < -10.f) || (invNd >= 0.f)) {
						__debugbreak();
						continue;
					}
#endif
					const Point3f& X = viewA.X[idx];
					// project point in second image and
					// projection Jacobian matrix in the second image of the 3D point on the surface
					MAYBEUNUSED const float depthB(ProjectVertex(cameraB.P.val, X.ptr(), xB.ptr(), xJac.val));
					ASSERT(depthB > 0);
					// compute gradient in image B
					const TMatrix<Real, 1, 2> gB(viewB.imageGrad.sample<Sampler, View::Grad>(sampler, xB));
					// compute gradient scale
					const Real dZNCC(imageDZNCC(r, c));
					const Real sg((gB * (xJac * (const TMatrix<Real, 3, 1>&)dA))(0) * dZNCC * RegularizationScale * invNd);
					// add gradient to the three vertices
					const Face& face(faces[idxFace]);
					const Point3f& b(viewA.baryMap(r, c));
					for (int v = 0; v < 3; ++v) {
						const Grad g(N * (sg * (Real)b[v]));
						const VIndex idxVert(face[v]);
#ifdef DEBUGPG
						if (idxVert >= photoGrad.size()) {
							__debugbreak();
						}
#endif

						photoGrad[idxVert] += g;
#ifdef DEBUGPG
						if (!isfinite(g.x) || !isfinite(g.y) || !isfinite(g.z)) {
							__debugbreak();
						}
#endif
						++photoGradNorm[idxVert];
					}
				}
			}
		}
	}

#else

#ifdef VALIDATE_GRADIENT
	int touched = 0;
	int count = cols * viewA.image.rows;
	GradArr myNewLocalGrad;
	myNewLocalGrad.resize(threadGrad.size());
	for (int i = 0; i < threadGrad.size(); ++i) {
		myNewLocalGrad[i] = Grad(0, 0, 0);
	}
	GradArr myOrigLocalGrad;
	myOrigLocalGrad.resize(threadGrad.size());
	for (int i = 0; i < threadGrad.size(); ++i) {
		myOrigLocalGrad[i] = Grad(0, 0, 0);
	}

	std::vector<float> gxarr(viewA.image.cols * viewA.image.rows, 0.f);
	std::vector<float> gyarr(viewA.image.cols * viewA.image.rows, 0.f);
	std::vector<float> gbxarr(viewA.image.cols * viewA.image.rows, 0.f);
	std::vector<float> gbyarr(viewA.image.cols * viewA.image.rows, 0.f);
	std::vector<float> gfracx(viewA.image.cols* viewA.image.rows, 0.f);
	std::vector<float> gfracy(viewA.image.cols* viewA.image.rows, 0.f);

#endif
	const size_t tilesX = (cols + TILEX - 1) / TILEX;

	const __m128 invScale = _mm_set1_ps(kInvScale);
  const int maxX = static_cast<int>(viewB.width) - 2;
  const int maxY = static_cast<int>(viewB.height) - 2;

	for (size_t ty = 0; ty < RowsEnd; ty += TILEY) {
		const size_t yEnd = std::min(ty + TILEY, RowsEnd);

		for (size_t tx = 0; tx < ColsEnd; tx += TILEX) {
			const size_t gridRow = ty / TILEY;
			const size_t gridCol = tx / TILEX;
			const size_t tileIndex = (gridRow * viewA.tilesX) + gridCol;
			size_t tileOffset = tileIndex * (TILEX * TILEY);

      // JPB WIP BUG Tile energy check
			if (!viewA.tileActive[tileIndex])
				continue;

			float tileEnergy = 0.f;

			const size_t xEnd = std::min(tx + TILEX, ColsEnd);

			constexpr int LOCAL_CAP = TILEX * TILEY * 4;  // enough for most 32x32 tiles
			struct TGrad { float v[3]; };
			alignas(64) TGrad tileGrad[LOCAL_CAP];

			// tileVertices must be 0xFFFFFFFF initially, but we save from re-initializing it every time
			// by having the flushing of the tile reset this.
			alignas(64)static thread_local uint32_t tileVertices[LOCAL_CAP];
			static thread_local bool initialized = false;
			if (!initialized) {
				for (int i = 0; i < LOCAL_CAP; ++i)
					tileVertices[i] = 0xFFFFFFFF;
				initialized = true;
			}

			alignas(64) uint32_t usedIdx[LOCAL_CAP];
			uint32_t slotVert[LOCAL_CAP];
			size_t usedCount = 0;

			//------------------------------------------------------------------
			//  Tile inner loop
			//------------------------------------------------------------------
			for (size_t r = ty; r < yEnd; ++r) {
				const size_t base = r * stride;
				const size_t localRowOffset = (r - ty) * TILEX;

				const uint8_t* __restrict maskRow = &mask[r * cols];
				const float* __restrict pdZNCC = imageDZNCC.ptr<float>(r);

				for (size_t c = tx; c < xEnd; ++c) {
					if (!maskRow[c]) continue;

					// All image pixels are legal here; we only need to check the view marks.
					const size_t idx = base + c;
					size_t localIdx = localRowOffset + (c - tx);

					const Point3f& X = viewA.X[tileOffset + localIdx];
					const Point3f& dA = viewA.ray[tileOffset + localIdx];
					float invNd = viewA.invNd[tileOffset + localIdx];

					const float X0 = X.x, X1 = X.y, X2 = X.z;

					// Compute projection numerator and denominator
					const float numX = p0 * X0 + p1 * X1 + p2 * X2 + p3;
					const float numY = p4 * X0 + p5 * X1 + p6 * X2 + p7;
					const float denW = p8 * X0 + p9 * X1 + p10 * X2 + p11;

					// Depth is strictly > 0 for valid pixels,
					// Thus denW cannot be 0 unless the point is exactly at the camera center, 
					// which never happens in our geometry.
					const float invW = 1.0f / denW;

					// Final projected pixel coords in B
					const float xB = numX * invW;
					const float yB = numY * invW;

					// ------------------------------------------------------------
					//  Bilinear sample from pre-packed gradient blocks
					// ------------------------------------------------------------
					int xi = _cvt_ftoi_fast(xB);
					int yi = _cvt_ftoi_fast(yB);

          // xi = std::clamp(xi, 0, viewB.width - 2);
					xi = xi & ~(xi >> 31); // clamps negative to 0
					int hi = maxX;
					xi = xi > hi ? hi : xi;

					// yi = std::clamp(yi, 0, viewB.height - 2);
					yi = yi & ~(yi >> 31); // clamps negative to 0
					hi = maxY;
					yi = yi > hi ? hi : yi;

					const float fx = xB - xi;
					const float fy = yB - yi;
					const size_t bi = (yi * (stride- 1) + xi) * 8;

#ifdef VALIDATE_GRADIENT
					gfracx[idx] = fx;
					gfracy[idx] = fy;
					gxarr[idx] = xi;
					gyarr[idx] = yi;
#endif

#if 0 //def VALIDATE_GRADIENT
					float gx00f = viewB.gradBlockInt16[bi + 0] * kInvScale;
					float gx01f = viewB.gradBlockInt16[bi + 1] * kInvScale;
					float gx10f = viewB.gradBlockInt16[bi + 2] * kInvScale;
					float gx11f = viewB.gradBlockInt16[bi + 3] * kInvScale;
												
					float gy00f = viewB.gradBlockInt16[bi + 4] * kInvScale;
					float gy01f = viewB.gradBlockInt16[bi + 5] * kInvScale;
					float gy10f = viewB.gradBlockInt16[bi + 6] * kInvScale;
					float gy11f = viewB.gradBlockInt16[bi + 7] * kInvScale;


					// Vertical interpolation
					float gx0 = gx00f + fy * (gx10f - gx00f);
					float gx1 = gx01f + fy * (gx11f - gx01f);

					float gy0 = gy00f + fy * (gy10f - gy00f);
					float gy1 = gy01f + fy * (gy11f - gy01f);

					// Horizontal interpolation
					float gx = gx0 + fx * (gx1 - gx0);
					float gy = gy0 + fx * (gy1 - gy0);

					float gBx = gx;
					float gBy = gy;


#else
					// load 2 2 patch of gradients
					// load 4 int16s -> 4 int32 -> 4 float
					const __m128i gx16 = _mm_loadl_epi64((__m128i*) & viewB.gradBlockInt16[bi + 0]); // gx00,gx01,gx10,gx11
					const __m128i gy16 = _mm_loadl_epi64((__m128i*) & viewB.gradBlockInt16[bi + 4]); // gy00,gy01,gy10,gy11

					__m128 gxv = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gx16));
					__m128 gyv = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(gy16));

					// rescale back to original float range
					gxv = _mm_mul_ps(gxv, invScale);
					gyv = _mm_mul_ps(gyv, invScale);

					const __m128 fxv = _mm_set1_ps(fx);
					const __m128 fyv = _mm_set1_ps(fy);

					// unpack rows: top (00,01), bottom (10,11)
					const __m128 gx_top = _mm_movelh_ps(gxv, gxv); // 00,01,00,01
					const __m128 gx_bot = _mm_movehl_ps(gxv, gxv); // 10,11,10,11
					const __m128 gy_top = _mm_movelh_ps(gyv, gyv);
					const __m128 gy_bot = _mm_movehl_ps(gyv, gyv);

					// vertical interpolation
					gxv = _mm_add_ps(gx_top, _mm_mul_ps(fyv, _mm_sub_ps(gx_bot, gx_top)));
					gyv = _mm_add_ps(gy_top, _mm_mul_ps(fyv, _mm_sub_ps(gy_bot, gy_top)));

					// horizontal interpolation
					const __m128 gx_shift = _mm_shuffle_ps(gxv, gxv, _MM_SHUFFLE(3, 3, 1, 1));
					const __m128 gy_shift = _mm_shuffle_ps(gyv, gyv, _MM_SHUFFLE(3, 3, 1, 1));

					const __m128 gx_res = _mm_add_ps(gxv, _mm_mul_ps(fxv, _mm_sub_ps(gx_shift, gxv)));
					const __m128 gy_res = _mm_add_ps(gyv, _mm_mul_ps(fxv, _mm_sub_ps(gy_shift, gyv)));

					// gBx, gBy now contain the scalar gradients from the bilinear sample
					const float gBx = _mm_cvtss_f32(gx_res);
					const float gBy = _mm_cvtss_f32(gy_res);
					
          // JPB WIP BUG Prune small gradients to improve performance.
					if ((gBx * gBx + gBy * gBy) < 1e-8f) continue;
#endif

#ifdef VALIDATE_GRADIENT
					gbxarr[idx] = gBx;
					gbyarr[idx] = gBy;
#endif

					// ------------------------------------------------------------
					//  Jacobian partials
					// ------------------------------------------------------------
					const float dx = dA.x;
					const float dy = dA.y;
					const float dz = dA.z;

					// Precompute linear dot products
					const float t0 = p0 * dx + p1 * dy + p2 * dz;     // numerator x part
					const float t1 = p4 * dx + p5 * dy + p6 * dz;     // numerator y part
					const float tW = p8 * dx + p9 * dy + p10 * dz;    // denominator part

					// Final Jacobian dot-products
					const float dot0 = t0 - xB * tW;
					const float dot1 = t1 - yB * tW;

					// ------------------------------------------------------------
					//  Gradient scale
					// ------------------------------------------------------------
					const float dZNCC = pdZNCC[c];

					// JPB WIP BUG
					// This eliminates pixels where the photometric patch correlation is weak, 
					// meaning the pixel is not contributing meaningful refinement signal.
					// Instead of checking dZNCC gradient, check actual ZNCC score
					// (requires passing ZNCC values from ComputeLocalZNCC)
					//if (dZNCC < 0.1f) continue;
					const float sg = (gBx * dot0 + gBy * dot1) * invW * invNd * RegularizationScale * dZNCC;
					tileEnergy += sg * sg;  // per-thread accumulation

					// ------------------------------------------------------------
					//  Accumulate per-vertex gradients
					// ------------------------------------------------------------
					const FIndex v1 = viewA.verticesPerPix[(tileOffset + localIdx) * 3];
					const FIndex v2 = viewA.verticesPerPix[(tileOffset + localIdx) * 3 + 1];
					const FIndex v3 = viewA.verticesPerPix[(tileOffset + localIdx) * 3 + 2];

					const Grad& N = viewA.facesNormalPerPix[tileOffset + localIdx];

					const Point3f& b = viewA.bary[tileOffset + localIdx];

					const Grad Ng = N * sg;      // scale once
					const float bx = b.x, by = b.y, bz = b.z;

#ifdef VALIDATE_GRADIENT
					++touched;
#endif

#if 1
					auto accum = [&](uint32_t vi, const Grad& g)
						{
//							++nTries;

							uint32_t slot = (vi * 0x9E3779B1u) & (LOCAL_CAP - 1);
							uint32_t step = 1;

							uint32_t* __restrict tv = tileVertices;
							TGrad* __restrict tg = tileGrad;

							for (;;) {
//								++nIterations;

								uint32_t v = tv[slot];

								if (v == vi) {
									auto& t = tg[slot];
									t.v[0] += g.x;
									t.v[1] += g.y;
									t.v[2] += g.z;
									return;
								}

								if (v == 0xFFFFFFFF) {
									tv[slot] = vi;
									auto& t = tg[slot];
									t.v[0] = g.x;
									t.v[1] = g.y;
									t.v[2] = g.z;
									usedIdx[usedCount++] = slot;
									return;
								}

								slot = (slot + step) & (LOCAL_CAP - 1);
								step += 1;
							}
						};

					accum(v1, Ng * bx);
					accum(v2, Ng * by);
					accum(v3, Ng * bz);
#else

					photoGrad[fIdx1] += Ng * bx;
					photoGrad[fIdx2] += Ng * by;
					photoGrad[fIdx3] += Ng * bz;

					photoGradNorm[fIdx1] += 1.0f;
					photoGradNorm[fIdx2] += 1.0f;
					photoGradNorm[fIdx3] += 1.0f;
#endif
				}
			} // end rows in tile

			// First pass: accumulate gradient only
			for (size_t i = 0; i < usedCount; ++i) {
				uint32_t slot = usedIdx[i];
				uint32_t vi = tileVertices[slot];
				slotVert[i] = vi;              // store for post-pass
				const auto& g = tileGrad[slot];

				Grad& tg = threadGrad[vi];
				tg.x += g.v[0];
				tg.y += g.v[1];
				tg.z += g.v[2];

				touchedVertices[vi] = 1;

				tileVertices[slot] = 0xFFFFFFFF;
			}

			// Second pass: update localNorm outside hot loop
			for (size_t i = 0; i < usedCount; ++i) {
				uint32_t vi = slotVert[i];

				if (touchedVertices[vi]) {
					touchedVertices[vi] = 0;
					localNorm[vi >> 6] |= (uint64_t(1) << (vi & 63));
				}
			}
			usedCount = 0;
			tileEnergyLocal[tileIndex] += tileEnergy;
		} // end tx
	} // end ty
#endif

#ifdef VALIDATE_GRADIENT
	int origTouched = 0;
	{
		ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
		ASSERT(depthMapA.size() == mask.size() && faceMapA.size() == mask.size() && baryMapA.size() == mask.size() && imageDZNCC.size() == mask.size() && !mask.empty());
		ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());
		const int RowsEnd(viewA.image.rows - HalfSize);
		const int ColsEnd(viewA.image.cols - HalfSize);
		typedef Sampler::Linear<View::Grad::Type> Sampler;
		const Sampler sampler;
		TMatrix<Real, 2, 3> xJac;
		Point2f xB;
		//photoGrad.Memset(0);
		//photoGradNorm.Memset(0);
		for (int r = HalfSize; r < RowsEnd; ++r) {
			for (int c = HalfSize; c < ColsEnd; ++c) {
				if (!mask[r*cols+c])
					continue;
				const FIndex idxFace(viewA.faceMap(r, c));
				if (idxFace == NO_ID) continue;
				//ASSERT(idxFace != NO_ID);
				const Grad N(normals[idxFace]);
				const Point3 rayA(cameraA.RayPoint(Point2(c, r)));
				const Grad dA(normalized(rayA));
				const Real Nd(N.dot(dA));
#if 1
				if (Nd > -0.1)
					continue;
#endif
				const Depth depthA(viewA.depthMap(r, c));
				if (depthA <= 0) continue;

				ASSERT(depthA > 0);
				const Point3 X(rayA * REAL(depthA) + cameraA.C);


#if 0
				// N and dA match.  X very slightly off
				{
					static std::mutex coutMutex;
					int idx = r * cols + c;

					const Point3f& X_new = viewA.X[idx];
					const Point3f& X_orig = X;

					float dx = X_new.x - X_orig.x;
					float dy = X_new.y - X_orig.y;
					float dz = X_new.z - X_orig.z;

					if (fabs(dx) > 1e-7 ||
						fabs(dy) > 1e-7 ||
						fabs(dz) > 1e-7)
					{
						std::lock_guard<std::mutex> lock(coutMutex);

						VERBOSE(
							"X mismatch at pixel %d:\n"
							"  X_orig = [%0.9f %0.9f %0.9f]\n"
							"  X_new  = [%0.9f %0.9f %0.9f]\n"
							"  diff   = [%0.9f %0.9f %0.9f]\n",
							idx,
							X_orig.x, X_orig.y, X_orig.z,
							X_new.x, X_new.y, X_new.z,
							dx, dy, dz
						);
					}
				}
#endif


				// project point in second image and
				// projection Jacobian matrix in the second image of the 3D point on the surface
				MAYBEUNUSED const float depthB(ProjectVertex(cameraB.P.val, X.ptr(), xB.ptr(), xJac.val));
				ASSERT(depthB > 0);
				// compute gradient in image B
				const TMatrix<Real, 1, 2> gB(viewB.imageGrad.sample<Sampler, View::Grad>(sampler, xB));

#if 1
				// N and dA match.
				{
					static std::mutex coutMutex;
					int idx = r * cols + c;
					if (fabs(std::floor(xB[0]) - gxarr[idx] > 0 ||
						fabs(std::floor(xB[1]) - gyarr[idx]) > 0)) {
						VERBOSE("gradient mismatch at pixel %d %f %f, %f %f\n", idx, xB[0], gxarr[idx], xB[1], gyarr[idx]);
					}
				}
				{
					static std::mutex coutMutex;
					int idx = r * cols + c;
					if (fabs(gB[0] - gbxarr[idx]) > 0.00001 ||
						fabs(gB[1] - gbyarr[idx]) > 0.00001) {
						VERBOSE("gradientv mismatch at pixel %d %f %f, %f %f\n", idx, gB[0], gbxarr[idx], gB[1], gbyarr[idx]);
					}
				}
				{
					static std::mutex coutMutex;
					int idx = r * cols + c;
					float gfx = xB[0] - std::floor(xB[0]);
					float gfy = xB[1] - std::floor(xB[1]);
					if (fabs(gfracx[idx] - gfx) > 0.00001 ||
						fabs(gfracy[idx] - gfy) > 0.00001) {
						VERBOSE("gradent frac  mismatch at pixel %d %f %f, %f %f\n", idx, gfx, gfracx[idx], gfy, gfracy[idx]);
					}
				}
#endif


				// compute gradient scale
				const Real dZNCC(imageDZNCC(r, c));
				const Real sg((gB * (xJac * (const TMatrix<Real, 3, 1>&)dA))(0) * dZNCC * RegularizationScale / Nd);
				// add gradient to the three vertices
				const Face& face(faces[idxFace]);
				const Point3f& b(viewA.baryMap[r * cols + c]);
				if (fabs(sg) >= 0.0000001f) ++origTouched;
				for (int v = 0; v < 3; ++v) {
					const Grad g(N * (sg * (Real)b[v]));
					const VIndex idxVert(face[v]);

					myOrigLocalGrad[v] += g;

					//photoGrad[idxVert] += g;
					//++photoGradNorm[idxVert];
				}
			}
		}
	}

#endif

#if 0 // JPB WIP BUG VALIDATE_GRADIENT
	{
		static std::mutex coutMutex;
		VERBOSE("Photometric gradient: touched %d / %d pixels (%.2f%%), orig %d\n", touched, count, 100.f * touched / count, origTouched);

		for (int i = 0; i < myNewLocalGrad.size(); ++i) {
			const Grad& gNew = myNewLocalGrad[i];
			const Grad& gOrig = myOrigLocalGrad[i];
			if (FastAbsS(gNew.x - gOrig.x) > 0.01f ||
				FastAbsS(gNew.y - gOrig.y) > 0.01f ||
				FastAbsS(gNew.z - gOrig.z) > 0.01f) {
				VERBOSE("  Vertex %d: New Grad (%.4f, %.4f, %.4f) vs Orig Grad (%.4f, %.4f, %.4f)\n",
					i,
					gNew.x, gNew.y, gNew.z,
					gOrig.x, gOrig.y, gOrig.z);
      }
		}
	}
#endif
}
#else
//original
// compute the photometric gradient for all vertices seen by an image pair
void MeshRefine::ComputePhotometricGradient(
	const Mesh::FaceArr& faces, const Mesh::NormalArr& normals,
	const DepthMap& depthMapA, const FaceMap& faceMapA, const BaryMap& baryMapA, const Camera& cameraA,
	const Camera& cameraB, const View& viewB,
	const TImage<Real>& imageDZNCC, const BitMatrix& mask, GradArr& photoGrad, UnsignedArr& photoGradNorm, Real RegularizationScale)
{
	ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
	ASSERT(depthMapA.size() == mask.size() && faceMapA.size() == mask.size() && baryMapA.size() == mask.size() && imageDZNCC.size() == mask.size() && !mask.empty());
	ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());
	const int RowsEnd(mask.rows - HalfSize);
	const int ColsEnd(mask.cols - HalfSize);
	typedef Sampler::Linear<View::Grad::Type> Sampler;
	const Sampler sampler;
	TMatrix<Real, 2, 3> xJac;
	Point2f xB;
	photoGrad.Memset(0);
	photoGradNorm.Memset(0);
	for (int r = HalfSize; r < RowsEnd; ++r) {
		for (int c = HalfSize; c < ColsEnd; ++c) {
			if (!mask(r, c))
				continue;
			const FIndex idxFace(faceMapA(r, c));
			ASSERT(idxFace != NO_ID);
			const Grad N(normals[idxFace]);
			const Point3 rayA(cameraA.RayPoint(Point2(c, r)));
			const Grad dA(normalized(rayA));
			const Real Nd(N.dot(dA));
#if 1
			if (Nd > -0.1)
				continue;
#endif
			const Depth depthA(depthMapA(r, c));
			ASSERT(depthA > 0);
			const Point3 X(rayA * REAL(depthA) + cameraA.C);
			// project point in second image and
			// projection Jacobian matrix in the second image of the 3D point on the surface
			MAYBEUNUSED const float depthB(ProjectVertex(cameraB.P.val, X.ptr(), xB.ptr(), xJac.val));
			ASSERT(depthB > 0);
			// compute gradient in image B
			const TMatrix<Real, 1, 2> gB(viewB.imageGrad.sample<Sampler, View::Grad>(sampler, xB));
			// compute gradient scale
			const Real dZNCC(imageDZNCC(r, c));
			const Real sg((gB * (xJac * (const TMatrix<Real, 3, 1>&)dA))(0) * dZNCC * RegularizationScale / Nd);
			// add gradient to the three vertices
			const Face& face(faces[idxFace]);
			const Point3f& b(baryMapA(r, c));
			for (int v = 0; v < 3; ++v) {
				const Grad g(N * (sg * (Real)b[v]));
				const VIndex idxVert(face[v]);
				photoGrad[idxVert] += g;
				++photoGradNorm[idxVert];
			}
		}
	}
}
#endif

// computes the discrete analog of the Laplacian using
// the umbrella-operator on the first triangle ring at each point
float MeshRefine::ComputeSmoothnessGradient1(
	const Mesh::VertexArr& vertices, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
	GradArr& smoothGrad1, VIndex idxStart, VIndex idxEnd)
{
	ASSERT(!vertices.IsEmpty() && vertices.GetSize() == vertexVertices.GetSize() && vertices.GetSize() == smoothGrad1.GetSize());
	float score(0);
	for (VIndex idxV = idxStart; idxV < idxEnd; ++idxV) {
		Grad& grad = smoothGrad1[idxV];
		grad = Grad::ZERO;
#if 1
		if (vertexBoundary[idxV])
			continue;
#endif
		const Mesh::VertexIdxArr& verts = vertexVertices[idxV];
		if (verts.IsEmpty())
			continue;
		FOREACH(v, verts)
			grad += Cast<Real>(vertices[verts[v]]);
		grad = grad / (Real)verts.GetSize() - Cast<Real>(vertices[idxV]);
		const float regularityScore((float)norm(grad));
		ASSERT(ISFINITE(regularityScore));
		score += regularityScore;
	}
	return score;
}
// same as above, but used to compute level 2;
// normalized as in "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004
void MeshRefine::ComputeSmoothnessGradient2(
	const GradArr& smoothGrad1, const Mesh::VertexVerticesArr& vertexVertices, const BoolArr& vertexBoundary,
	GradArr& smoothGrad2, VIndex idxStart, VIndex idxEnd)
{
	ASSERT(!smoothGrad1.IsEmpty() && smoothGrad1.GetSize() == vertexVertices.GetSize() && smoothGrad1.GetSize() == smoothGrad2.GetSize());
	for (VIndex idxV = idxStart; idxV < idxEnd; ++idxV) {
		Grad& grad = smoothGrad2[idxV];
		grad = Grad::ZERO;
#if 1
		if (vertexBoundary[idxV])
			continue;
#endif
		const Mesh::VertexIdxArr& verts = vertexVertices[idxV];
		if (verts.IsEmpty())
			continue;
		Real w(0);
		FOREACH(v, verts) {
			const VIndex idxVert(verts[v]);
			grad += smoothGrad1[idxVert];
			const VIndex numVert(vertexVertices[idxVert].GetSize());
			if (numVert > 0)
				w += Real(1) / (Real)numVert;
		}
		const Real numVert((Real)verts.GetSize());
		const Real nrm(Real(1) / (Real(1) + w / numVert));
		grad = grad * (nrm / numVert) - smoothGrad1[idxV] * nrm;
	}
}


void* MeshRefine::ThreadWorkerTmp(void* arg) {
	MeshRefine& refine = *((MeshRefine*)arg);
	refine.ThreadWorker();
	return NULL;
}
void MeshRefine::ThreadWorker()
{
	while (true) {
		CAutoPtr<Event> evt(events.GetEvent());
		switch (evt->GetID()) {
		case EVT_JOB:
			evt->Run(this);
			break;
		case EVT_CLOSE:
			return;
		default:
			ASSERT("Should not happen!" == NULL);
		}
		sem.Signal();
	}
}
void MeshRefine::WaitThreadWorkers(size_t nJobs)
{
	while (nJobs-- > 0)
		sem.Wait();
	ASSERT(events.IsEmpty());
}
void MeshRefine::ThSelectNeighbors(uint32_t idxImage, std::unordered_set<uint64_t>& mapPairs, unsigned nMaxViews)
{
	// keep only best neighbor views
	const float fMinArea(0.1f);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(2.5f)), fMaxAngle(FD2R(45.f));
	Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	if (imageData.neighbors.IsEmpty()) {
		IndexArr points;
		scene.SelectNeighborViews(idxImage, points);
	}
	ViewScoreArr neighbors(imageData.neighbors);
	Scene::FilterNeighborViews(neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews);
	Lock l(cs);
	FOREACHPTR(pNeighbor, neighbors) {
		ASSERT(images[pNeighbor->idx.ID].IsValid());
		mapPairs.insert(MakePairIdx((uint32_t)idxImage, pNeighbor->ID));
	}
}

__forceinline unsigned ctz64(uint64_t x) {
	unsigned long idx;              // MSVC requires unsigned long
	_BitScanForward64(&idx, x);     // undefined for x==0 (same as builtin)
	return (unsigned)idx;
}

void MeshRefine::ThInitImage(uint32_t idxImage, Real scale, Real sigma)
{
	Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	// load and init image
	unsigned level(nResolutionLevel);
	const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
	if ((imageData.image.empty() || MAXF(imageData.width, imageData.height) != imageSize) && !imageData.ReloadImage(imageSize))
		ABORT("can not load image");
	View& view = views[idxImage];
	Image32F& img = view.image;
	imageData.image.toGray(img, cv::COLOR_BGR2GRAY, true);
	imageData.image.release();
	if (sigma > 0)
		cv::GaussianBlur(img, img, cv::Size(), sigma);
	if (scale < 1.0) {
		cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
		imageData.width = img.width(); imageData.height = img.height();
	}
	imageData.UpdateCamera(scene.platforms);
	if (!nReduceMemory) {
		throw; // Unsupported
#if 0
		// compute image mean and variance
		ComputeLocalVariance(img, std::vector<uint8_t>(img.cols * img.rows, 0xFF), view.imageMean, view.imageVar);
#endif
	}
	// compute image gradient
	typedef View::Grad::Type GradType;
	static thread_local TImage<GradType> grad[2];
	grad[0].create(img.rows, img.cols);
	grad[1].create(img.rows, img.cols);

	const TMatrix<GradType, 3, 5> kernel(CreateDerivativeKernel3x5());
	cv::filter2D(img, grad[0], cv::DataType<GradType>::type, kernel);
	cv::filter2D(img, grad[1], cv::DataType<GradType>::type, kernel.t());
#ifdef VALIDATE_GRADIENT
	cv::merge(grad, 2, view.imageGrad);
#endif

	// ------------------------------------------------------------------
	// Build packed 2x2 gradient blocks (int16 quantized)
	// ------------------------------------------------------------------
	const int width = view.width = grad[0].cols;
	const int height = view.height = grad[0].rows;
	const size_t numElems = static_cast<size_t>(height - 1) * (width - 1) * 8;

	_aligned_free(view.gradBlockInt16); // Will leak on exit.
	view.gradBlockInt16 = (int16_t*)_aligned_malloc(numElems * sizeof(int16_t), 16);

	for (int y = 0; y < height - 1; ++y) {
		const GradType* gxRow0 = grad[0].ptr<GradType>(y);
		const GradType* gxRow1 = grad[0].ptr<GradType>(y + 1);
		const GradType* gyRow0 = grad[1].ptr<GradType>(y);
		const GradType* gyRow1 = grad[1].ptr<GradType>(y + 1);

		for (int x = 0; x < width - 1; ++x) {
			const int bi = (y * (width - 1) + x) * 8;

			auto quant = [](float f) -> int16_t {
				float scaled = f * kScale;
				if (scaled > 32767.f)  scaled = 32767.f;
				if (scaled < -32767.f) scaled = -32767.f;
				return static_cast<int16_t>(scaled);
				};

			// gx 00,01,10,11
			view.gradBlockInt16[bi + 0] = quant(gxRow0[x]);
			view.gradBlockInt16[bi + 1] = quant(gxRow0[x + 1]);
			view.gradBlockInt16[bi + 2] = quant(gxRow1[x]);
			view.gradBlockInt16[bi + 3] = quant(gxRow1[x + 1]);

			// gy 00,01,10,11
			view.gradBlockInt16[bi + 4] = quant(gyRow0[x]);
			view.gradBlockInt16[bi + 5] = quant(gyRow0[x + 1]);
			view.gradBlockInt16[bi + 6] = quant(gyRow1[x]);
			view.gradBlockInt16[bi + 7] = quant(gyRow1[x + 1]);
		}
	}
}

void MeshRefine::ThProjectMesh(uint32_t idxImage, const Mesh::FaceIdxArr& cameraFaces, const CameraRenderData& rd)
{
	const Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	// project mesh to the given camera plane
	View& view = views[idxImage];
	ProjectMesh(view, imageData.camera, rd);

	view.tilesX = (view.width + TILEX - 1) / TILEX;
	view.tilesY = (view.height + TILEY - 1) / TILEY;

	view.tileActive.assign(view.tilesX * view.tilesY, 1);  // initially all tiles active
}
void MeshRefine::ThProcessPair(uint32_t idxImageA, uint32_t idxImageB, GradArr& threadGrad, std::vector<uint32_t>& threadNorm, std::vector<std::vector<float>>& tileEnergyLocal)
{
	// fetch view A data
	const Image& imageDataA = images[idxImageA];
	ASSERT(imageDataA.IsValid());
	const View& viewA = views[idxImageA];
	const BaryMap& baryMapA = viewA.baryMap;
	const FaceMap& faceMapA = viewA.faceMap;
	const DepthMap& depthMapA = viewA.depthMap;
	const Image32F& imageA = viewA.image;
	const Camera& cameraA = imageDataA.camera;
	// fetch view B data
	const Image& imageDataB = images[idxImageB];
	ASSERT(imageDataB.IsValid());
	const View& viewB = views[idxImageB];
	const DepthMap& depthMapB = viewB.depthMap;
	const Image32F& imageB = viewB.image;
	const Camera& cameraB = imageDataB.camera;
	// warp imageB to imageA using the mesh

  size_t numPixels = imageA.cols * imageA.rows;
	static thread_local std::vector<uint8_t> mask;
  mask.resize(numPixels);

  static thread_local TImage<uint16_t> imageAB;
  imageAB.create(imageA.rows, imageA.cols);
	ImageMeshWarp(viewA, depthMapA, cameraA, depthMapB, cameraB, imageB, imageAB, mask);

	// compute ZNCC and its gradient
	const TImage<uint16_t>* imageMeanA;
	const TImage<Real> * imageVarA;
	if (nReduceMemory) {
		static thread_local TImage<uint16_t> _imageMeanA;
		static thread_local TImage<Real> _imageVarA;
		ComputeLocalVariance(viewA.image, mask, _imageMeanA, _imageVarA);
		imageMeanA = &_imageMeanA;
		imageVarA = &_imageVarA;
	}
	else {
		throw; // Unsupported
#if 0
		imageMeanA = &viewA.imageMean;
		imageVarA = &viewA.imageVar;
#endif
	}
	static thread_local TImage<uint16_t> imageMeanAB;
	static thread_local TImage<Real> imageVarAB;
	ComputeLocalVariance2(imageAB, mask, imageMeanAB, imageVarAB);

	static thread_local TImage<Real> imageDZNCC;
	const float score(ComputeLocalZNCC(imageA, *imageMeanA, *imageVarA, imageAB, imageMeanAB, imageVarAB, mask, imageDZNCC));
	// compute field gradient
	const Real RegularizationScale((Real)((REAL)(imageDataA.avgDepth * imageDataB.avgDepth) / (cameraA.GetFocalLength() * cameraB.GetFocalLength())));
	//DEC_BitMatrix(localNorm);
	//jpb wip bug set this here
	//localNorm.memset(0);
	static thread_local std::vector<uint64_t> localNorm;
	localNorm.assign((photoGrad.size() + 63) / 64, 0);
	ComputePhotometricGradient(viewA, cameraA, cameraB, viewB, imageDZNCC, mask, threadGrad, localNorm, RegularizationScale, tileEnergyLocal[idxImageA]);

	// threadGrad has been updated.
	// localNorm must be merged to threadNorm:
	uint32_t* __restrict norm = threadNorm.data();
	const uint64_t* __restrict bits = localNorm.data();
	size_t count = photoGrad.size();
	size_t words = (count + 63) >> 6;

	size_t idx = 0;
	for (size_t w = 0; w < words; ++w) {
		uint64_t mask = bits[w];
		while (mask) {
			uint64_t bit = mask & -mask;       // lowest set bit
			unsigned i = ctz64(mask);
			norm[idx + i] += 1;                // increment norm
			mask ^= bit;                       // clear that bit
		}
		idx += 64;
	}
	//DST_BitMatrix(localNorm);

#if 0
	// JPB WIP BUG Lock l(cs);
	if (vertexDepth.IsEmpty()) {
		FOREACH(i, photoGrad) {
			if (_photoGradNorm[i] > 0) {
				AtomicAddFloat(&photoGrad[i][0], _photoGrad[i][0]);
				AtomicAddFloat(&photoGrad[i][1], _photoGrad[i][1]);
				AtomicAddFloat(&photoGrad[i][2], _photoGrad[i][2]);
				AtomicAddFloat(&photoGradNorm[i], 1.f);
			}
		}
	}
	else {
		Lock l(cs);
		const float depth(MINF(imageDataA.avgDepth, imageDataB.avgDepth));
		FOREACH(i, photoGrad) {
			if (_photoGradNorm[i] > 0) {
				photoGrad[i] += _photoGrad[i];
				photoGradNorm[i] += 1.f;
				if (vertexDepth[i] > depth)
					vertexDepth[i] = depth;
			}
		}
	}
#endif

	scorePhoto += (float)RegularizationScale * score;
}
void MeshRefine::ThSmoothVertices1(VIndex idxStart, VIndex idxEnd)
{
	const float score(ComputeSmoothnessGradient1(vertices, vertexVertices, vertexBoundary, smoothGrad1, idxStart, idxEnd));
	Lock l(cs);
	scoreSmooth += score;
}
void MeshRefine::ThSmoothVertices2(VIndex idxStart, VIndex idxEnd)
{
	ComputeSmoothnessGradient2(smoothGrad1, vertexVertices, vertexBoundary, smoothGrad2, idxStart, idxEnd);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

#ifdef MESHOPT_CERES

#pragma push_macro("LOG")
#undef LOG
#pragma push_macro("CHECK")
#undef CHECK
#pragma push_macro("ERROR")
#undef ERROR
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#pragma pop_macro("ERROR")
#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")

namespace ceres {
	class MeshProblem : public FirstOrderFunction, public IterationCallback
	{
	public:
		MeshProblem(MeshRefine& _refine) : refine(_refine), params(refine.vertices.GetSize() * 3) {
			// init params
			FOREACH(i, refine.vertices)
				* ((Point3d*)params.Begin() + i) = refine.vertices[i];
		}
		virtual ~MeshProblem() {}

		void ApplyParams() const {
			FOREACH(i, refine.vertices)
				refine.vertices[i] = *((Point3d*)params.Begin() + i);
		}
		void ApplyParams(const double* parameters) const {
			memcpy(params.Begin(), parameters, sizeof(double) * params.GetSize());
			ApplyParams();
		}

		bool Evaluate(const double* const parameters, double* cost, double* gradient) const {
			// update surface parameters
			ApplyParams(parameters);
			// evaluate residuals and gradients
			Point3dArr gradients;
			if (!gradient) {
				gradients.Resize(refine.vertices.GetSize());
				gradient = (double*)gradients.Begin();
			}
			*cost = refine.ScoreMesh(gradient);
			return true;
		}

		CallbackReturnType operator()(const IterationSummary& summary) {
			refine.iteration = summary.iteration;
			return ceres::SOLVER_CONTINUE;
		}

		int NumParameters() const { return (int)params.GetSize(); }
		const double* GetParameters() const { return params.Begin(); }
		double* GetParameters() { return params.Begin(); }

	protected:
		MeshRefine& refine;
		DoubleArr params;
	};
} // namespace ceres

#endif // MESHOPT_CERES


// optimize mesh using photo-consistency
// fThPlanarVertex - threshold used to remove vertices on planar patches (percentage of the minimum depth, 0 - disable)
bool Scene::RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews,
	float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea,
	unsigned nScales, float fScaleStep,
	unsigned nReduceMemory, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fThPlanarVertex, float fGradientStep)
{
	cv::setNumThreads(1);// disable OpenCV internal threading to eliminate oversubscription with our threading.

	if (pointcloud.IsEmpty() && !ImagesHaveNeighbors())
		SampleMeshWithVisibility();

	MeshRefine refine(*this, nReduceMemory, nAlternatePair, fRegularityWeight, fRatioRigidityElasticity, nResolutionLevel, nMinResolution, nMaxViews, nMaxThreads);
	if (!refine.IsValid())
		return false;

	// run the mesh optimization on multiple scales (coarse to fine)
	for (unsigned nScale = 0; nScale < nScales; ++nScale) {
		// init images
		const Real scale(POWI(fScaleStep, nScales - nScale - 1));
		const Real step(POWI(2.f, nScales - nScale));
		DEBUG_ULTIMATE("Refine mesh at: %.2f image scale", scale);
		if (!refine.InitImages(scale, Real(0.12) * step + Real(0.2)))
			return false;

		// extract array of triangles incident to each vertex
		refine.ListVertexFacesPre();

		// automatic mesh subdivision
		refine.SubdivideMesh(nMaxFaceArea, nScale == 0 ? fDecimateMesh : 1.f, nCloseHoles, nEnsureEdgeSize);

		// extract array of triangle normals
		refine.ListVertexFacesPost();

#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefine%u.ply", nScales - nScale - 1)));
#endif

		// minimize
#ifdef MESHOPT_CERES
		if (fGradientStep == 0) {
			// DefineProblem
			refine.ratioRigidityElasticity = 1.f;
			ceres::MeshProblem* problemData(new ceres::MeshProblem(refine));
			ceres::GradientProblem problem(problemData);
			// SetMinimizerOptions
			ceres::GradientProblemSolver::Options options;
			if (VERBOSITY_LEVEL > 1) {
				options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
				options.minimizer_progress_to_stdout = true;
			}
			else {
				options.logging_type = ceres::LoggingType::SILENT;
				options.minimizer_progress_to_stdout = false;
			}
			options.function_tolerance = 1e-3;
			options.gradient_tolerance = 1e-7;
			options.max_num_line_search_step_size_iterations = 10;
			options.callbacks.push_back(problemData);
			ceres::GradientProblemSolver::Summary summary;
			// SolveProblem
			ceres::Solve(options, problem, problemData->GetParameters(), &summary);
			DEBUG_ULTIMATE(summary.FullReport().c_str());
			switch (summary.termination_type) {
			case ceres::TerminationType::NO_CONVERGENCE:
				DEBUG_EXTRA("CERES: maximum number of iterations reached!");
			case ceres::TerminationType::CONVERGENCE:
			case ceres::TerminationType::USER_SUCCESS:
				break;
			default:
				VERBOSE("CERES surface refine error: %s!", summary.message.c_str());
				return false;
			}
			ASSERT(summary.IsSolutionUsable());
			problemData->ApplyParams();
		}
		else
#endif // MESHOPT_CERES
		{
#if 1
			uint32_t nVerts = refine.vertices.GetSize();
			int iters;
			if (nScale == 0) iters = 8;      // coarse
			else if (nScale == 1) iters = 6; // mid
			else iters = 4;                  // fine

			double gstep = 1.5;
			double decay = 0.995;

			Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> gradients;
			gradients.resize(nVerts, 3);

			// NEW: momentum
			Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> velocity;
			velocity.resize(nVerts, 3);
			velocity.setZero();

			// NEW: convergence thresholds (currently unused as early-stop)
			double prevCost = std::numeric_limits<double>::max();
			const double costThresh = 1e-4;
			const double gradThresh = 1e-3 * double(nVerts);

			Util::Progress progress(_T("Processed iterations"), iters);
			GET_LOGCONSOLE().Pause();

			for (int iter = 0; iter < iters; ++iter)
			{
				refine.iteration = iter;
				refine.nAlternatePair = (iter + 1 < iters ? nAlternatePair : 0);
				refine.ratioRigidityElasticity =
					(iter <= iters * 7 / 10 ? fRatioRigidityElasticity : 1.f);

				bool rebuildOctree = (iter % 4 == 0);

				double cost = refine.ScoreMesh(gradients.data(), rebuildOctree);

				// -------------------------------------------------------------
				// Momentum update
				// -------------------------------------------------------------
#pragma omp parallel for schedule(static)
				for (int v = 0; v < (int)nVerts; v++) {
					float gx = gradients(v, 0);
					float gy = gradients(v, 1);
					float gz = gradients(v, 2);

					const float beta = 0.75f;
					const float oneMinusBeta = 1.0f - beta;
					float step = float(gstep);

					float vx = velocity(v, 0);
					float vy = velocity(v, 1);
					float vz = velocity(v, 2);

					vx = beta * vx + oneMinusBeta * gx;
					vy = beta * vy + oneMinusBeta * gy;
					vz = beta * vz + oneMinusBeta * gz;

					velocity(v, 0) = vx;
					velocity(v, 1) = vy;
					velocity(v, 2) = vz;

					Vertex& vert = refine.vertices[v];
					vert.x -= vx * step;
					vert.y -= vy * step;
					vert.z -= vz * step;
				}

				double gradNorm = gradients.norm();
				double avgGrad = gradNorm / double(nVerts);

				DEBUG_EXTRA("Iter %d avgGrad = %.6e", iter, avgGrad);

				// =============================================================
				// DIAGNOSTICS BLOCK INSERTED HERE
				// =============================================================

				// 1. Max per-vertex displacement
// 1. Max per-vertex displacement (manual reduction)
				double maxDisp = 0.0;
				{
					int numThreads = 1;
#ifdef _OPENMP
					numThreads = omp_get_max_threads();
#endif
					std::vector<double> localMax(numThreads, 0.0);

#pragma omp parallel
					{
						int tid = 0;
#ifdef _OPENMP
						tid = omp_get_thread_num();
#endif
						double lm = 0.0;

#pragma omp for nowait
						for (int v = 0; v < (int)nVerts; v++) {
							float vx = velocity(v, 0);
							float vy = velocity(v, 1);
							float vz = velocity(v, 2);

							float dx = vx * float(gstep);
							float dy = vy * float(gstep);
							float dz = vz * float(gstep);

							double disp = sqrt(double(dx * dx + dy * dy + dz * dz));
							if (disp > lm)
								lm = disp;
						}

						localMax[tid] = lm;
					}

					for (size_t i = 0; i < localMax.size(); i++)
						if (localMax[i] > maxDisp)
							maxDisp = localMax[i];
				}

				// 2. Active tiles
				int totalTiles = 0;
				int activeTiles = 0;

				for (size_t vi = 0; vi < refine.views.size(); vi++) {
					const MeshRefine::View& vw = refine.views[vi];
					int tiles = (int)vw.tileActive.size();
					totalTiles += tiles;

					for (int t = 0; t < tiles; t++)
						activeTiles += (vw.tileActive[t] ? 1 : 0);
				}

				double activeRatio = (totalTiles > 0)
					? double(activeTiles) / double(totalTiles)
					: 0.0;

				// 3. Photometric energy sum
				double photoEnergy = refine.photoEnergyLast;

				// 4. Smooth/photo gradient ratio (manual reduction)
				double photoGradSum = 0.0;
				double smoothGradSum = 0.0;

				{
					int numThreads = 1;
#ifdef _OPENMP
					numThreads = omp_get_max_threads();
#endif

					std::vector<double> localPhoto(numThreads, 0.0);
					std::vector<double> localSmooth(numThreads, 0.0);

#pragma omp parallel
					{
						int tid = 0;
#ifdef _OPENMP
						tid = omp_get_thread_num();
#endif
						double lp = 0.0;
						double ls = 0.0;

#pragma omp for nowait
						for (int v = 0; v < (int)nVerts; v++) {
							float gx = gradients(v, 0);
							float gy = gradients(v, 1);
							float gz = gradients(v, 2);
							lp += sqrt(double(gx * gx + gy * gy + gz * gz));

							const auto& sg = refine.smoothGrad2[v];
							ls += FastSqrtD(double(sg.x * sg.x + sg.y * sg.y + sg.z * sg.z));
						}

						localPhoto[tid] = lp;
						localSmooth[tid] = ls;
					}

					for (size_t i = 0; i < localPhoto.size(); i++)
						photoGradSum += localPhoto[i];

					for (size_t i = 0; i < localSmooth.size(); i++)
						smoothGradSum += localSmooth[i];
				}

				double smoothPhotoRatio =
					(photoGradSum > 0.0 ? smoothGradSum / photoGradSum : 0.0);

				// 5. Print diagnostics
				DEBUG_EXTRA(
					"Iter %d summary: avgGrad=%.6e  maxDisp=%.6e  activeTiles=%d/%d (%.2f%%)  photoEnergy=%.6e  smooth/photo=%.3f",
					iter,
					avgGrad,
					maxDisp,
					activeTiles,
					totalTiles,
					activeRatio * 100.0,
					photoEnergy,
					smoothPhotoRatio
				);

				// =============================================================
				// END DIAGNOSTICS BLOCK
				// =============================================================

				// Optional safety exit (rare)
				if (iter > 0 && avgGrad < 1e-9) {
					DEBUG_EXTRA("Early exit: avgGrad extremely small (%.6e)", avgGrad);
					break;
				}

				const double energyPerTile =
					(activeTiles > 0)
					? (refine.photoEnergyLast / double(activeTiles))
					: 0.0;

				if (
					iter > 2 &&
					activeRatio < 5e-4 &&
					maxDisp < 0.3 * scale
					) {
					break;
				}

				gstep *= decay;
				progress.display(iter);
			}

			GET_LOGCONSOLE().Play();
			progress.close();
#else
			// loop a constant number of iterations and apply the gradient
			int iters(75);
			double gstep(0.4);
			if (fGradientStep > 1) {
				iters = FLOOR2INT(fGradientStep);
				gstep = (fGradientStep - (float)iters) * 10;
			}
			iters = MAXF(iters / (int)(nScale + 1), 8);
			const int iterStop(iters * 7 / 10);
			const int iterStart(fThPlanarVertex > 0 ? iters * 4 / 10 : INT_MAX);
			Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> gradients(refine.vertices.GetSize(), 3);
			Util::Progress progress(_T("Processed iterations"), iters);
			GET_LOGCONSOLE().Pause();

      constexpr int numGradentApplicationsBeforeOctreeRebuild = 10;
			int numGradientApplications = numGradentApplicationsBeforeOctreeRebuild;
			for (int iter = 0; iter < iters; ++iter) {
				refine.iteration = (unsigned)iter;
				refine.nAlternatePair = (iter + 1 < iters ? nAlternatePair : 0);
				refine.ratioRigidityElasticity = (iter <= iterStop ? fRatioRigidityElasticity : 1.f);
				const bool bAdaptMesh(iter >= iterStart && (iter - iterStart) % 3 == 0 && iters - iter > 5);
				// evaluate residuals and gradients
				if (bAdaptMesh)
					refine.vertexDepth.Resize(refine.vertices.GetSize());

        // Octree rebuilding is very expensive so we limit its frequency.
        // It is technically not accurate to do so, but in practice the vertices move slowly enough for this to be acceptable.
				bool rebuildOctree = numGradientApplications >= numGradentApplicationsBeforeOctreeRebuild;
				const double cost = refine.ScoreMesh(gradients.data(), rebuildOctree);
				if (rebuildOctree) {
					numGradientApplications = 0;
				}
				
				double gv(0);
				VIndex numVertsRemoved(0);
				if (bAdaptMesh) {
					// apply gradients and
					// remove planar vertices (small gradient and almost on the center of their surrounding patch)
					ASSERT(refine.vertexDepth.GetSize() == refine.vertices.GetSize());
					Mesh::VertexIdxArr vertexRemove;
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						vert -= Cast<Vertex::Type>(grad * gstep);
						const double gn(norm(grad));
						gv += gn;
						const float depth(refine.vertexDepth[v]);
						if (depth < FLT_MAX) {
							const float th(depth * fThPlanarVertex);
							if (!refine.vertexBoundary[v] && (float)gn < th && norm(refine.smoothGrad1[v]) < th)
								vertexRemove.Insert(v);
						}
					}
					if (!vertexRemove.IsEmpty()) {
						numVertsRemoved = vertexRemove.GetSize();
						mesh.Decimate(vertexRemove);
						refine.ListVertexFacesPost();
					}
					refine.vertexDepth.Empty();
					numGradientApplications = numGradentApplicationsBeforeOctreeRebuild;
				}
				else {
					// apply gradients
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						vert -= Cast<Vertex::Type>(grad * gstep);
						gv += norm(grad);
					}
					++numGradientApplications;
				}
				DEBUG_EXTRA("\t%2d. f: %.5f (%.4e)\tg: %.5f (%.4e - %.4e)\ts: %.3f\tv: %5u", iter + 1, cost, cost / refine.vertices.GetSize(), gradients.norm(), gradients.norm() / refine.vertices.GetSize(), gv / refine.vertices.GetSize(), gstep, numVertsRemoved);
				gstep *= 0.98;
				progress.display(iter);
			}
			GET_LOGCONSOLE().Play();
			progress.close();
#endif
		}

#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefined%u.ply", nScales - nScale - 1)));
#endif
	}

	// Mesh was mutated and may no longer be valid.

	return true;
} // RefineMesh
/*----------------------------------------------------------------*/
