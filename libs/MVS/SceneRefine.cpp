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
		TImage<Real> imageMean; // image pixels mean
		TImage<Real> imageVar; // image pixels variance
		std::vector<Point3f, AlignedAllocator<Point3f, 16>> ray;
		std::vector<Point3f, AlignedAllocator<Point3f, 16>> X;
		std::vector<float, AlignedAllocator<float, 16>> Nd;
		std::vector<float, AlignedAllocator<float, 16>> invNd;
		std::vector<Point3f, AlignedAllocator<Point3f, 16>> storedNormal;       // 3D point
		std::vector<Point3f, AlignedAllocator<Point3f, 16>> bary;       // 3D point
		std::vector<float, AlignedAllocator<float, 16>> gradX;
		std::vector<float, AlignedAllocator<float, 16>> gradY;
		std::vector<float, AlignedAllocator<float, 16>> gradBlock;
		std::vector<cuint32_t, AlignedAllocator<cuint32_t, 16>> verticesPerPix;
		std::vector<Normal, AlignedAllocator<Normal, 16>> facesNormalPerPix;
		FaceMap faceMap; // remember for each pixel what face projects there
		DepthMap depthMap; // depth-map
		BaryMap baryMap; // barycentric coordinates
		std::vector<uint8_t, AlignedAllocator<uint8_t, 16>> isValid;
		int width, height;
		int blockWidth, blockStride;
		std::vector<uint8_t> marks;
		uint8_t currentMark;
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
			faceMap.memset((uint8_t)NO_ID);
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
	void ListCameraFaces();

	void ListFaceAreas(Mesh::AreaArr& maxAreas);
	void SubdivideMesh(uint32_t maxArea, float fDecimate = 1.f, unsigned nCloseHoles = 15, unsigned nEnsureEdgeSize = 1);

	double ScoreMesh(double* gradients);

	// given a vertex position and a projection camera, compute the projected position and its derivative
	template <typename TP, typename TX, typename T, typename TJ>
	static T ProjectVertex(const TP* P, const TX* X, T* x, TJ* jacobian = NULL);

	static bool IsDepthSimilar(const DepthMap& depthMap, const Point2f& pt, Depth z);
	static void ProjectMesh(
		View& view,
		const Mesh::NormalArr& faceNormals,
		const Mesh::VertexArr& vertices, const Mesh::FaceArr& faces, const Mesh::FaceIdxArr& cameraFaces,
		const Camera& camera, const Image8U::Size& size,
		DepthMap& depthMap, FaceMap& faceMap, BaryMap& baryMap);
	static void ImageMeshWarp(
		const View& viewA,
		const DepthMap& depthMapA, const Camera& cameraA,
		const DepthMap& depthMapB, const Camera& cameraB,
		const Image32F& imageB, Image32F& imageA, std::vector<uint8_t>& mask);
	static void ComputeLocalVariance(
		const Image32F& image, const  std::vector<uint8_t>& mask,
		TImage<Real>& imageMean, TImage<Real>& imageVar);
	static float ComputeLocalZNCC(
		const Image32F& imageA, const TImage<Real>& imageMeanA, const TImage<Real>& imageVarA,
		const Image32F& imageB, const TImage<Real>& imageMeanB, const TImage<Real>& imageVarB,
		const  std::vector<uint8_t>& mask, TImage<Real>& imageDZNCC);
	static void ComputePhotometricGradient(
		const Mesh::FaceArr& faces,
		const Mesh::NormalArr& normals,
		const View& viewA,
		const Camera& cameraA,
		const Camera& cameraB,
		const View& viewB,
		const TImage<Real>& imageDZNCC,
		const  std::vector<uint8_t>& mask,
		GradArr& photoGrad,
		std::vector<bool>& photoGradNorm,
		Real RegularizationScale);
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
	void ThProjectMesh(uint32_t idxImage, const Mesh::FaceIdxArr& cameraFaces);
	void ThProcessPair(uint32_t idxImageA, uint32_t idxImageB, GradArr& localGrad, std::vector<uint32_t>& threadNorm);
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
	bool Run(void* pArgs) {
		((MeshRefine*)pArgs)->ThProjectMesh(idxImage, cameraFaces);
		return true;
	}
	EVTProjectMesh(uint32_t _idxImage, const Mesh::FaceIdxArr& _cameraFaces) : Event(EVT_JOB), idxImage(_idxImage), cameraFaces(_cameraFaces) {}
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

// extract array of faces viewed by each image
void MeshRefine::ListCameraFaces()
{
	// extract array of faces viewed by each camera
	typedef CLISTDEF2(Mesh::FaceIdxArr) CameraFacesArr;
	CameraFacesArr arrCameraFaces(images.GetSize()); {
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

	// compute face normals
	// must occur before ListCameraFaces which prepares pre-calculated data.
	scene.mesh.ComputeNormalFaces();

	// project mesh to each camera plane
	ASSERT(events.IsEmpty());
	FOREACH(idxImage, images)
		events.AddEvent(new EVTProjectMesh(idxImage, arrCameraFaces[idxImage]));
	WaitThreadWorkers(images.GetSize());
}

// compute for each face the projection area as the maximum area in both images of a pair
// (make sure ListCameraFaces() was called before)
void MeshRefine::ListFaceAreas(Mesh::AreaArr& maxAreas)
{
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
	// for each pair, mark the faces that have big projection areas in both images
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
double MeshRefine::ScoreMesh(double* gradients)
{
	// extract array of faces viewed by each camera
	ListCameraFaces();

	// JPB WIP BUG Nneded twice?
	scene.mesh.ComputeNormalFaces();

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
#pragma omp parallel
	{
		const int numVerts = (int)photoGrad.size();
		GradArr localGrad;
    localGrad.assign(numVerts, Grad(0, 0, 0));
		std::vector<uint32_t> localNorm;
    localNorm.assign(numVerts, 0);

		// Each thread processes its subset of pairs dynamically
#pragma omp for schedule(dynamic)
		for (int i = 0; i < (int)pairs.GetSize(); ++i) {
			const auto& pair = pairs[i];
			ThProcessPair(pair.j, pair.i, localGrad, localNorm);
			ThProcessPair(pair.i, pair.j, localGrad, localNorm);
		}

		// Barrier implicit here at end of 'omp for'
#pragma omp for schedule(static)
		for (int v = 0; v < numVerts; ++v) {
			photoGrad[v] += localGrad[v];
			photoGradNorm[v] += localNorm[v];
		}
	} // end omp parallel
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
			((Point3d*)gradients)[v] = photoGradNorm[v] > 0 ?
			Cast<double>(photoGrad[v] / photoGradNorm[v] + smoothGrad2[v] * weightRegularity) :
			Cast<double>(smoothGrad2[v] * weightRegularity);
	}
	else {
		// compute smoothing gradient as a combination of level 1 and 2 of the Laplacian operator;
		// (see page 105 of "Stereo and Silhouette Fusion for 3D Object Modeling from Uncalibrated Images Under Circular Motion" C. Hernandez, 2004)
		const Real rigidity((Real(1) - ratioRigidityElasticity) * weightRegularity);
		const Real elasticity(ratioRigidityElasticity * weightRegularity);
		FOREACH(v, vertices)
			((Point3d*)gradients)[v] = photoGradNorm[v] > 0 ?
			Cast<double>(photoGrad[v] / photoGradNorm[v] + smoothGrad2[v] * elasticity - smoothGrad1[v] * rigidity) :
			Cast<double>(smoothGrad2[v] * elasticity - smoothGrad1[v] * rigidity);
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
TYPE EdgeFunction2(const TPoint2<TYPE>& x0, const TPoint2<TYPE>& x1, const TPoint2<TYPE>& x2) {
	return TYPE((x1 - x0).cross(x2 - x0));  // swapped x1 and x2
}

// project mesh to the given camera plane
void MeshRefine::ProjectMesh(
	View& view,
	const Mesh::NormalArr& faceNormals,
	const Mesh::VertexArr& vertices, const Mesh::FaceArr& faces, const Mesh::FaceIdxArr& cameraFaces,
	const Camera& camera, const Image8U::Size& size,
	DepthMap& depthMap, FaceMap& faceMap, BaryMap& baryMap)
{
	// init view data
	depthMap.create(size);
	faceMap.create(size);
	baryMap.create(size);

	view.isValid.assign(size.width * size.height, 0);

	//depthMap.memset(0);
	//faceMap.memset((uint8_t)NO_ID);
	//baryMap.memset(0);

	struct Triangle {
		Point3 ptc[3];
		Point2f pti[3];
	};
	Triangle t;

	const int width = size.width;
	const int height = size.height;

	const float M00 = camera.Pf(0, 0);
	const float M01 = camera.Pf(0, 1);
	const float M02 = camera.Pf(0, 2);
	const float M03 = camera.Pf(0, 3);

	const float M10 = camera.Pf(1, 0);
	const float M11 = camera.Pf(1, 1);
	const float M12 = camera.Pf(1, 2);
	const float M13 = camera.Pf(1, 3);

	const float M20 = camera.Pf(2, 0);
	const float M21 = camera.Pf(2, 1);
	const float M22 = camera.Pf(2, 2);
	const float M23 = camera.Pf(2, 3);

	for (auto idxFace : cameraFaces) {
		const Face& facet = faces[idxFace];

		bool skipFace = false;
		for (int i = 0; i < 3; ++i) {
			const auto& p = vertices[facet[i]];

			// ------------------------------------------------------------
			//  world to camera projection using 3 4 matrix Pf
			// ------------------------------------------------------------
			float x = M00 * p.x + M01 * p.y + M02 * p.z + M03;
			float y = M10 * p.x + M11 * p.y + M12 * p.z + M13;
			float z = M20 * p.x + M21 * p.y + M22 * p.z + M23;
			if (z <= 0.f) { skipFace = true; break; }

			// normalized image coordinates
			float invZ = 1.f / z;
			float u = x * invZ;   // horizontal coordinate in pixels
			float v = y * invZ;   // vertical coordinate in pixels

			// check pixel bounds
			if (u < 3.f || v < 3.f || u > width - 4.f || v > height - 4.f) {
				skipFace = true;
				break;
			}

			// store results
			t.ptc[i] = { x, y, z };
			t.pti[i] = { u, v };
		}

		if (skipFace)
			continue;

		// draw triangle
		const auto& v1 = t.pti[0];
		const auto& v2 = t.pti[1];
		const auto& v3 = t.pti[2];

		// compute bounding-box fully containing the triangle
		const TPoint2<float> boxMin(MINF3(v1.x, v2.x, v3.x), MINF3(v1.y, v2.y, v3.y));
		const TPoint2<float> boxMax(MAXF3(v1.x, v2.x, v3.x), MAXF3(v1.y, v2.y, v3.y));
		// check the bounding-box intersects the image
		if (boxMax.x < 0.f || boxMin.x >(float)(size.width - 1) ||
			boxMax.y < 0.f || boxMin.y >(float)(size.height - 1))
			continue;
		// clip bounding-box to be fully contained by the image
		ImageRef boxMinI(FLOOR2INT(boxMin));
		ImageRef boxMaxI(CEIL2INT(boxMax));

		constexpr int border = 0;
		if (boxMinI.x < border)
			boxMinI.x = border;
		if (boxMinI.y < border)
			boxMinI.y = border;
		if (boxMaxI.x >= (size.width - border))
			boxMaxI.x = (size.width - (border + 1));
		if (boxMaxI.y >= (size.height - border))
			boxMaxI.y = (size.height - (border + 1));

		// ignore back oriented triangles (negative area)
	// flip winding to match OpenMVS screen-space convention
		const float area = EdgeFunction2(v1, v2, v3);
		if (area >= 0) continue;
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
		float z0 = t.ptc[0].z;
		float z1 = t.ptc[1].z;
		float z2 = t.ptc[2].z;

		// clamp small / invalid z
		z0 = (z0 <= 0.f) ? 1e-6f : z0;
		z1 = (z1 <= 0.f) ? 1e-6f : z1;
		z2 = (z2 <= 0.f) ? 1e-6f : z2;

		// reciprocal depths
		const float iz0 = 1.f / z0;
		const float iz1 = 1.f / z1;
		const float iz2 = 1.f / z2;

		Depth* __restrict depthRow = nullptr;
		cuint32_t* __restrict faceRow = (cuint32_t*)nullptr;
		auto* __restrict baryRow = (Point3f*)nullptr;

		for (size_t y = boxMinI.y; y <= boxMaxI.y; ++y) {
			float w0 = w0_row, w1 = w1_row, w2 = w2_row;

			depthRow = &depthMap[y * width];
			faceRow = &faceMap[y * width];
			baryRow = &baryMap[y * width];
      uint8_t* __restrict isValidRow = &view.isValid[y * width];

			for (size_t x = boxMinI.x; x <= boxMaxI.x; ++x) {
				// inside test (branchless style)
				if ((w0 >= 0.f) & (w1 >= 0.f) & (w2 >= 0.f)) {
					// perspective-correct barycentrics
					const float denom = w0 * iz0 + w1 * iz1 + w2 * iz2;
					if (fabsf(denom) >= 1e-10f) {
						const float invDenom = 1.f / denom;
						const float bx = (w0 * iz0) * invDenom;
						const float by = (w1 * iz1) * invDenom;
						const float bz = 1.f - bx - by;     // third term implicit

						// compute depth (2 FMAs)
						const float z = bx * z0 + by * z1 + bz * z2;

						Depth& depth = depthRow[x];
						if (depth == 0.f || depth > z) {
							depth = z;
							faceRow[x] = idxFace;
							baryRow[x] = { bx, by, bz };
							isValidRow[x] = 1;
						}
					}
				}

				// advance barycentrics horizontally
				w0 += w0_dx;
				w1 += w1_dx;
				w2 += w2_dx;
			}

			// advance barycentrics vertically
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

	const size_t rows = size.height;
	const size_t cols = size.width;
	const size_t count = rows * cols;
	view.width = cols;
	view.height = rows;

	// allocate SoA arrays (aligned)
	view.ray.assign(count, Point3f(0, 0, 0));
	view.X.assign(count, Point3f(0, 0, 0));
	view.storedNormal.assign(count, Point3f(0, 0, 0));
	view.Nd.assign(count, 0.0f);
	view.invNd.assign(count, 0.0f);
	view.verticesPerPix.assign(count * 3, NO_ID);
	view.facesNormalPerPix.assign(count, Point3f(0, 0, 0));
	// optional: barycentric coordinates as SoA if you plan to use them this way
	view.bary.assign(count, TPoint3<float>(0, 0, 0));

	// main pass: fill SoA arrays
#ifdef VALIDATE_COUNT
	int validCnt = 0;
#endif

	for (size_t r = 0; r < rows; ++r)	{
		uint8_t* __restrict isValidRow = &view.isValid[r * cols];
		for (size_t c = 0; c < cols; ++c)	{
			if (!isValidRow[c]) {
				view.faceMap(r, c) = NO_ID;
				continue;
			}

			const size_t idx = r * cols + c;

			// view.depthMap(r,c) guaranteed > 0
			const float depth = view.depthMap(r, c);

			// Unnormalized direction in camera coords (z=1), rotated to world
			// RayPoint = R^T * TransformPointI2C([ (u-cx)/fx, (v-cy)/fy, 1 ])
			const Point3f rayW = camera.RayPoint(Point2(c, r));

			// Reconstruct 3D point in world: X = C + (rayW * depth)
			// (equivalently: X = R^T * ([x',y',1] * depth) + C)
			const Point3f X = rayW * depth + Cast<float>(camera.C);

			// Face index for this pixel
			const FIndex f = view.faceMap(r, c);
			// view.faceMap(r,c) guaranteed not NO_ID

			// Normalized ray ONLY for Nd
			const float invLen = 1.0f / sqrtf(rayW.x * rayW.x + rayW.y * rayW.y + rayW.z * rayW.z);
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
			view.Nd[idx] = Nd;

			float invNd = 1.0f / Nd;

			if ((invNd < -10.f) || (invNd >= 0.f)) {
				isValidRow[c] = 0;
				continue;
			}

			view.invNd[idx] = 1.0f / Nd;

			const Face& face = faces[f];
			FIndex faceIndexes[3] = { face[0], face[1], face[2] };

			view.verticesPerPix[idx * 3 + 0] = faceIndexes[0];
			view.verticesPerPix[idx * 3 + 1] = faceIndexes[1];
			view.verticesPerPix[idx * 3 + 2] = faceIndexes[2];
			view.facesNormalPerPix[idx] = faceNormals[f];

			// Barycentrics if needed later
			const Point3f& b = view.baryMap(r, c);
			view.bary[idx] = b;

			// JPB WIP BUG Not needed isValidRow[c] = 1;
#ifdef VALIDATE_COUNT
			++validCnt;
#endif
		}
	}

#ifdef VALIDATE_COUNT
	{
		static std::mutex coutMutex;
		VERBOSE("View: %d valid pixels out of %d (%.2f%%)\n",
			validCnt, count, 100.f * validCnt / count);
	}
#endif
}

// project image from view B to view A through the mesh;
// the projected image is stored in imageA
// (imageAB is assumed to be initialize to the right size)
void MeshRefine::ImageMeshWarp(
	const View& viewA,
	const DepthMap& depthMapA, const Camera& cameraA,
	const DepthMap& depthMapB, const Camera& cameraB,
	const Image32F& imageB, Image32F& imageA, std::vector<uint8_t>& mask)
{
	ASSERT(!imageA.empty());
	typedef Sampler::Linear<float> Sampler;
	const Sampler sampler;
  const size_t cols = depthMapA.cols;
  const size_t rows = depthMapA.rows;
	for (size_t j = 0; j < rows; ++j) {
    uint8_t* maskRow = &mask[j * depthMapA.cols];
		const uint8_t* __restrict isValidRow = &viewA.isValid[j * cols];

		for (size_t i = 0; i < depthMapA.cols; ++i) {
			if (isValidRow[i]) {
				const Depth& depthA = depthMapA(j, i);
				const Point3 X(cameraA.TransformPointI2W(Point3(i, j, depthA)));
				const Point3f ptC(cameraB.TransformPointW2C(X));
				const Point2f pt(cameraB.TransformPointC2I(ptC));
				if (!IsDepthSimilar(depthMapB, pt, ptC.z)) {
					*maskRow++ = 0;
					continue;
				}
				imageA(j, i) = imageB.sample<Sampler, Sampler::Type>(sampler, pt);
				*maskRow++ = 1;
			}
			else {
        *maskRow++ = 0;
			}
		}
	}
}

// compute local variance for each image pixel
void MeshRefine::ComputeLocalVariance(const Image32F& image,
	const std::vector<uint8_t>& mask,
	TImage<Real>& imageMean,
	TImage<Real>& imageVar) {
	ASSERT(image.size() == mask.size());
	imageMean.create(image.size());
	imageVar.create(image.size());

	const size_t hs = HalfSize;
	const size_t rows = image.rows;
	const size_t cols = image.cols;
	constexpr size_t n = (2 * hs + 1) * (2 * hs + 1);
	constexpr Real invN = 1.0 / static_cast<Real>(n);

	const size_t rowStart = hs;
	const size_t rowEnd = rows - hs;
	const size_t colStart = hs;
	const size_t colEnd = cols - hs;

	// Column running sums for current vertical window [r-hs .. r+hs]
	std::vector<Real> colSum(cols, 0.0);
	std::vector<Real> colSumSq(cols, 0.0);

	// Seed vertical window for the first output row: rows [0 .. 2*hs]
	for (size_t rr = 0; rr < 2 * hs + 1 && rr < rows; ++rr) {
		const float* src = image.ptr<float>(rr);
		for (size_t c = 0; c < cols; ++c) {
			Real v = src[c];
			colSum[c] += v;
			colSumSq[c] += v * v;
		}
	}

	// Process each output row
	for (size_t r = rowStart; r < rowEnd; ++r) {
		Real winSum = 0.0;
		Real winSumSq = 0.0;

		// Initialize horizontal window for column range [0 .. 2*hs]
		for (size_t cc = 0; cc < 2 * hs + 1 && cc < cols; ++cc) {
			winSum += colSum[cc];
			winSumSq += colSumSq[cc];
		}

		Real* __restrict meanRow = imageMean.ptr<Real>(r);
		Real* __restrict varRow = imageVar.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		for (size_t c = colStart; c < colEnd; ++c) {
			if (maskRow[c]) {
				Real mean = winSum * invN;
				Real var = winSumSq * invN - mean * mean;
				if (var < 0.0001) var = 0.0001;
				meanRow[c] = static_cast<Real>(mean);
				varRow[c] = static_cast<Real>(var);
			}

			// Slide horizontal window by one pixel
			if (c + hs + 1 < cols) {
				winSum += colSum[c + hs + 1] - colSum[c - hs];
				winSumSq += colSumSq[c + hs + 1] - colSumSq[c - hs];
			}
		}

		// Advance vertical window after finishing row r:
		// remove row (r - hs) and add row (r + hs + 1)
		if (r + hs + 1 < rows) {
			const float* __restrict addRow = image.ptr<float>(r + hs + 1);
			const float* __restrict remRow = image.ptr<float>(r - hs);
			for (int c = 0; c < cols; ++c) {
				Real a = addRow[c];
				Real d = remRow[c];
				colSum[c] += a - d;
				colSumSq[c] += a * a - d * d;
			}
		}
	}
}

// compute local ZNCC and its gradient for each image pixel
float MeshRefine::ComputeLocalZNCC(
	const Image32F& imageA, const TImage<Real>& imageMeanA, const TImage<Real>& imageVarA,
	const Image32F& imageB, const TImage<Real>& imageMeanB, const TImage<Real>& imageVarB,
	const std::vector<uint8_t>& mask, TImage<Real>& imageDZNCC)
{
	ASSERT(imageA.size() == mask.size() && imageB.size() == mask.size() && !mask.empty());

	const size_t hs = HalfSize;
	const size_t rows = imageA.rows;
	const size_t cols = imageA.cols;
	constexpr size_t n = (2 * hs + 1) * (2 * hs + 1);
	constexpr float invN = 1.0f / static_cast<float>(n);
	const size_t rowEnd = rows - hs;
	const size_t colEnd = cols - hs;

	static thread_local TImage<Real> imageZNCC;
	if (imageZNCC.cols != cols || imageZNCC.rows != rows) {
		imageZNCC.create(rows, cols);
	}
	if (imageDZNCC.cols != cols || imageDZNCC.rows != rows) {
		imageDZNCC.create(rows, cols);
	}

	// Invariant: we will only read/write to imageZNCC(r,c)  and imageDZNCC(r,c) if mask(r,c) is set; other values are undefined.
	// This allows us to avoid initializing these images here.

	// --- build integral of (A * B) in float, single-threaded ---
	static thread_local cv::Mat imageAB, imageABSum;
	cv::multiply(imageA, imageB, imageAB, 1, CV_32F);
	cv::integral(imageAB, imageABSum, CV_32F);

	// --- first pass: compute local covariance (cv) and ZNCC ---
	static thread_local TImage<Real> imageInvSqrtVAVB;
	imageInvSqrtVAVB.create(rows, cols);

	for (size_t r = hs; r < rowEnd; ++r) {
		const float* __restrict sumUp = imageABSum.ptr<float>(r - hs);
		const float* __restrict sumDown = imageABSum.ptr<float>(r + hs + 1);
		Real* __restrict znccRow = imageZNCC.ptr<Real>(r);
		Real* __restrict invRow = imageInvSqrtVAVB.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		for (size_t c = hs; c < colEnd; ++c) {
			if (!maskRow[c]) continue;

			const int x0 = c - hs;
			const int x1 = c + hs + 1;

			const float cov = (sumDown[x1] - sumDown[x0] -
				sumUp[x1] + sumUp[x0]) * invN;

			Real invSqrtVAVB = Real(1) / FastSqrtS(imageVarA(r, c) * imageVarB(r, c));
			Real meanA = imageMeanA(r, c);
			Real meanB = imageMeanB(r, c);

			Real zncc = (cov - meanA * meanB) * invSqrtVAVB;
			znccRow[c] = zncc;
			invRow[c] = invSqrtVAVB;
		}
	}

	// --- second pass: compute gradient and accumulate score ---
	float score = 0.0f;

	for (size_t r = hs; r < rowEnd; ++r) {
		// early reject if an entire row of the mask is zero can t be tested here
		// because mask(r,hs) is only one pixel; so just enter the loop.
		Real* __restrict dRow = imageDZNCC.ptr<Real>(r);
		const Real* __restrict znccRow = imageZNCC.ptr<Real>(r);
		const Real* __restrict invRow = imageInvSqrtVAVB.ptr<Real>(r);
		const uint8_t* __restrict maskRow = &mask[r * cols];

		for (int c = hs; c < colEnd; ++c) {
			if (!maskRow[c]) continue;

			const Real zncc = znccRow[c];
			const Real invS = invRow[c];
			const Real varB = imageVarB(r, c);
			const Real meanA = imageMeanA(r, c);
			const Real meanB = imageMeanB(r, c);
			const Real aVal = static_cast<Real>(imageA(r, c));
			const Real bVal = static_cast<Real>(imageB(r, c));

			const Real znccInvVB = zncc / varB;
			const Real dzncc = aVal * invS - bVal * znccInvVB
				+ meanB * znccInvVB - meanA * invS;

			const Real minVAVB = MINF(imageVarA(r, c), varB);
			const Real reliability = minVAVB / (minVAVB + Real(0.0015));

			const Real grad = -reliability * dzncc;
			dRow[c] = grad;
			score += static_cast<float>(reliability * (Real(1) - zncc));
		}
	}

	return score;
}

#if 1

#define FMA(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)
#undef VALIDATE_GRADIENT

void MeshRefine::ComputePhotometricGradient(
	const Mesh::FaceArr& faces,
	const Mesh::NormalArr& normals,
	const View& viewA,
	const Camera& cameraA,
	const Camera& cameraB,
	const View& viewB,
	const TImage<Real>& imageDZNCC,
	const std::vector<uint8_t>& mask,
	GradArr& threadGrad,
	std::vector<bool>& localNorm,
	Real RegularizationScale)
{
	ASSERT(faces.GetSize() == normals.GetSize() && !faces.IsEmpty());
	ASSERT(viewB.image.size() == viewB.imageGrad.size() && !viewB.image.empty());

  size_t cols = viewA.image.cols;
	const size_t RowsEnd = viewA.image.rows - HalfSize;
	const size_t ColsEnd = cols - HalfSize;

#if 1
	thread_local std::vector<uint8_t>  localGen;
	localGen.resize(faces.GetSize());
	thread_local uint8_t localMark = 0;
	if (++localMark == 0) {
		std::fill(localGen.begin(), localGen.end(), 0);
	}
#endif
	const size_t stride = viewA.width;

	const float* P = &cameraB.Pf[0];

	const __m128 one = _mm_set1_ps(1.0f);

	constexpr int TILEX = 16;
	constexpr int TILEY = 16;


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
#if 1 //local
	static constexpr int LOCAL_CAP = TILEX * TILEY * 4;  // enough for most 32x32 tiles
	uint32_t usedIdx[LOCAL_CAP];
	Grad tileGrad[LOCAL_CAP];
	size_t usedCount = 0;
	uint32_t tileVertices[LOCAL_CAP];
	for (int i = 0; i < LOCAL_CAP; ++i) tileVertices[i] = 0xFFFFFFFF;

#endif
	const uint8_t currentMark = viewA.currentMark;

#ifdef VALIDATE_GRADIENT
	int touched = 0;
#endif

	for (size_t ty = 0; ty < RowsEnd; ty += TILEY) {
		const size_t yEnd = std::min(ty + TILEY, RowsEnd);
		for (size_t tx = 0; tx < ColsEnd; tx += TILEX) {
			const size_t xEnd = std::min(tx + TILEX, ColsEnd);

			//------------------------------------------------------------------
			//  Tile inner loop
			//------------------------------------------------------------------
			for (size_t r = ty; r < yEnd; ++r) {
				const size_t base = r * stride;
				size_t c = tx;
				const uint8_t* __restrict maskRow = &mask[r * cols];

				for (; c < xEnd; ++c) {
					if (!maskRow[c]) continue;
					// All image pixels are legal here; we only need to check the view marks.
					const size_t idx = base + c;
					// -----------------------------------------------------------------
					//  LOAD  STAGE
					// -----------------------------------------------------------------
					// JPB WIP BUG _mm_prefetch((const char*)&viewA.Xx[idx + 16], _MM_HINT_T0);
					// JPB WIP BUG _mm_prefetch((const char*)&viewA.rayX[idx + 16], _MM_HINT_T0);
					const Point3f& X = viewA.X[idx];
					const Point3f& dA = viewA.ray[idx];
					float invNd = viewA.invNd[idx];

					const float X0 = X.x, X1 = X.y, X2 = X.z;

					// w = P20*X0 + P21*X1 + P22*X2 + P23
					float w = P[8] * X0 + P[9] * X1 + P[10] * X2 + P[11];
					if (fabsf(w) < 1e-6f) continue; // skip invalid projections
					float invW = 1.0f / w;

					// projected coordinates
					float xB = (P[0] * X0 + P[1] * X1 + P[2] * X2 + P[3]) * invW;
					float yB = (P[4] * X0 + P[5] * X1 + P[6] * X2 + P[7]) * invW;

					// ------------------------------------------------------------
					//  Bilinear sample from pre-packed gradient blocks
					// ------------------------------------------------------------
					int xi = _cvt_ftoi_fast(xB);
					int yi = _cvt_ftoi_fast(yB);
					xi = std::clamp(xi, 0, viewB.width - 2);
					yi = std::clamp(yi, 0, viewB.height - 2);

					float fx = xB - xi;
					float fy = yB - yi;
					const size_t bi = (yi * (stride- 1) + xi) * 8;

					// load 2 2 patch of gradients
					__m128 gxv = _mm_load_ps(&viewB.gradBlock[bi + 0]); // gx00,gx01,gx10,gx11
					__m128 gyv = _mm_load_ps(&viewB.gradBlock[bi + 4]); // gy00,gy01,gy10,gy11

					__m128 fxv = _mm_set1_ps(fx);
					__m128 fyv = _mm_set1_ps(fy);

					// unpack rows: top (00,01), bottom (10,11)
					__m128 gx_top = _mm_movelh_ps(gxv, gxv); // 00,01,00,01
					__m128 gx_bot = _mm_movehl_ps(gxv, gxv); // 10,11,10,11
					__m128 gy_top = _mm_movelh_ps(gyv, gyv);
					__m128 gy_bot = _mm_movehl_ps(gyv, gyv);

					// vertical interpolation
					gxv = _mm_add_ps(gx_top, _mm_mul_ps(fyv, _mm_sub_ps(gx_bot, gx_top)));
					gyv = _mm_add_ps(gy_top, _mm_mul_ps(fyv, _mm_sub_ps(gy_bot, gy_top)));

					// horizontal interpolation
					__m128 gx_shift = _mm_shuffle_ps(gxv, gxv, _MM_SHUFFLE(3, 3, 1, 1));
					__m128 gy_shift = _mm_shuffle_ps(gyv, gyv, _MM_SHUFFLE(3, 3, 1, 1));

					__m128 gx_res = _mm_add_ps(gxv, _mm_mul_ps(fxv, _mm_sub_ps(gx_shift, gxv)));
					__m128 gy_res = _mm_add_ps(gyv, _mm_mul_ps(fxv, _mm_sub_ps(gy_shift, gyv)));

					// final scalar result
					float gx = _mm_cvtss_f32(gx_res);
					float gy = _mm_cvtss_f32(gy_res);

					// gBx, gBy now contain the scalar gradients from the bilinear sample
					float gBx = gx;   // from the sampler above
					float gBy = gy;

					// ------------------------------------------------------------
					//  Jacobian partials
					// ------------------------------------------------------------
					float xJac00 = (P[0] - P[8] * xB) * invW;
					float xJac01 = (P[1] - P[9] * xB) * invW;
					float xJac02 = (P[2] - P[10] * xB) * invW;
					float xJac10 = (P[4] - P[8] * yB) * invW;
					float xJac11 = (P[5] - P[9] * yB) * invW;
					float xJac12 = (P[6] - P[10] * yB) * invW;

					// dot products (xJac rows dA)
					float dot0 = xJac00 * dA.x + xJac01 * dA.y + xJac02 * dA.z;
					float dot1 = xJac10 * dA.x + xJac11 * dA.y + xJac12 * dA.z;

					// ------------------------------------------------------------
					//  Gradient scale
					// ------------------------------------------------------------
					float dZNCC = imageDZNCC[r * stride + c];
					float sg = (gBx * dot0 + gBy * dot1) * invNd * RegularizationScale * dZNCC;

					// ------------------------------------------------------------
					//  Accumulate per-vertex gradients
					// ------------------------------------------------------------
					const FIndex v1 = viewA.verticesPerPix[idx * 3];
					const FIndex v2 = viewA.verticesPerPix[idx * 3 + 1];
					const FIndex v3 = viewA.verticesPerPix[idx * 3 + 2];

					const Grad& N = viewA.facesNormalPerPix[idx];

					const Point3f& b = viewA.bary[idx];

					const Grad Ng = N * sg;      // scale once
					const float bx = b.x, by = b.y, bz = b.z;

#ifdef VALIDATE_GRADIENT
					++touched;
#endif

#if 1
					auto accum = [&](uint32_t vi, const Grad& g)
						{
							uint32_t slot = vi & (LOCAL_CAP - 1); // hash (requires LOCAL_CAP power of two)
							for (;;)
							{
								if (tileVertices[slot] == vi) {
									// already present -> accumulate
									tileGrad[slot] += g;
									break;
								}
								if (tileVertices[slot] == 0xFFFFFFFF) {
									// new entry
									tileVertices[slot] = vi;
									tileGrad[slot] = g;
									usedIdx[usedCount++] = slot;
									break;
								}
								slot = (slot + 1) & (LOCAL_CAP - 1);
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

			//std::sort(usedIdx, usedIdx + usedCount,
			//	[&](uint32_t a, uint32_t b) { return tileVertices[a] < tileVertices[b]; });

			for (size_t i = 0; i < usedCount; ++i) {
				uint32_t slot = usedIdx[i];
				uint32_t vi = tileVertices[slot];
				const Grad& g = tileGrad[slot];

				threadGrad[vi] += g;

				if (localGen[vi] != localMark) {
					// first write to this face in the current pass
					localGen[vi] = localMark;
					localNorm[vi] = 1;
				}

				tileVertices[slot] = 0xFFFFFFFF;
			}
			usedCount = 0;
		} // end tx
	} // end ty
#endif

#ifdef VALIDATE_GRADIENT
	int origTouched = 0;
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
		//photoGrad.Memset(0);
		//photoGradNorm.Memset(0);
		for (int r = HalfSize; r < RowsEnd; ++r) {
			for (int c = HalfSize; c < ColsEnd; ++c) {
				//if (!mask(r, c))
				//	continue;
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
				//const Point3f& b(baryMapA(r, c));
				if (fabs(sg) >= 0.0000001f) ++origTouched;
#if 0
				for (int v = 0; v < 3; ++v) {
					const Grad g(N * (sg * (Real)b[v]));
					const VIndex idxVert(face[v]);
					//photoGrad[idxVert] += g;
					//++photoGradNorm[idxVert];
				}
#endif
			}
		}
	}

#endif

#ifdef VALIDATE_GRADIENT
	{
		static std::mutex coutMutex;
		VERBOSE("Photometric gradient: touched %d / %d pixels (%.2f%%), orig %d\n", touched, count, 100.f * touched / count, origTouched);
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
		// compute image mean and variance
		ComputeLocalVariance(img, std::vector<uint8_t>(img.cols * img.rows, 0xFF), view.imageMean, view.imageVar);
	}
	// compute image gradient
	typedef View::Grad::Type GradType;
	TImage<GradType> grad[2];
#if 0
	cv::Sobel(img, grad[0], cv::DataType<GradType>::type, 1, 0, 3, 1.0 / 8.0);
	cv::Sobel(img, grad[1], cv::DataType<GradType>::type, 0, 1, 3, 1.0 / 8.0);
#elif 1
	const TMatrix<GradType, 3, 5> kernel(CreateDerivativeKernel3x5());
	cv::filter2D(img, grad[0], cv::DataType<GradType>::type, kernel);
	cv::filter2D(img, grad[1], cv::DataType<GradType>::type, kernel.t());
#else
	const TMatrix<GradType, 5, 7> kernel(CreateDerivativeKernel5x7());
	cv::filter2D(img, grad[0], cv::DataType<GradType>::type, kernel);
	cv::filter2D(img, grad[1], cv::DataType<GradType>::type, kernel.t());
#endif
	cv::merge(grad, 2, view.imageGrad);

	// ------------------------------------------------------------------
	// Build packed 2 2 gradient blocks for SIMD bilinear sampling
	// ------------------------------------------------------------------
	const int width = view.width = grad[0].cols;
	const int height = view.height = grad[0].rows;

	// one block per top-left pixel of each 2 2 region
	view.gradBlock.resize((height - 1) * (width - 1) * 8);

	for (int y = 0; y < height - 1; ++y)
	{
		const GradType* gxRow0 = grad[0].ptr<GradType>(y);
		const GradType* gxRow1 = grad[0].ptr<GradType>(y + 1);
		const GradType* gyRow0 = grad[1].ptr<GradType>(y);
		const GradType* gyRow1 = grad[1].ptr<GradType>(y + 1);

		for (int x = 0; x < width - 1; ++x)
		{
			const int bi = (y * (width - 1) + x) * 8;

			// gx 00,01,10,11
			view.gradBlock[bi + 0] = gxRow0[x];
			view.gradBlock[bi + 1] = gxRow0[x + 1];
			view.gradBlock[bi + 2] = gxRow1[x];
			view.gradBlock[bi + 3] = gxRow1[x + 1];

			// gy 00,01,10,11
			view.gradBlock[bi + 4] = gyRow0[x];
			view.gradBlock[bi + 5] = gyRow0[x + 1];
			view.gradBlock[bi + 6] = gyRow1[x];
			view.gradBlock[bi + 7] = gyRow1[x + 1];
		}
	}
}

void MeshRefine::ThProjectMesh(uint32_t idxImage, const Mesh::FaceIdxArr& cameraFaces)
{
	const Image& imageData = images[idxImage];
	if (!imageData.IsValid())
		return;
	// project mesh to the given camera plane
	View& view = views[idxImage];
	ProjectMesh(view, faceNormals, vertices, faces, cameraFaces, imageData.camera, view.image.size(),
		view.depthMap, view.faceMap, view.baryMap);
}
void MeshRefine::ThProcessPair(uint32_t idxImageA, uint32_t idxImageB, GradArr& threadGrad, std::vector<uint32_t>& threadNorm)
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
	if (mask.size() != imageA.cols * imageA.rows)
    mask.resize(numPixels);

	DEC_Image(float, imageAB);
	imageA.copyTo(imageAB);
	ImageMeshWarp(viewA, depthMapA, cameraA, depthMapB, cameraB, imageB, imageAB, mask);
	// compute ZNCC and its gradient
	const TImage<Real>* imageMeanA, * imageVarA;
	if (nReduceMemory) {
		DEC_Image(Real, _imageMeanA);
		DEC_Image(Real, _imageVarA);
		ComputeLocalVariance(viewA.image, mask, _imageMeanA, _imageVarA);
		imageMeanA = &_imageMeanA;
		imageVarA = &_imageVarA;
	}
	else {
		imageMeanA = &viewA.imageMean;
		imageVarA = &viewA.imageVar;
	}
	DEC_Image(Real, imageMeanAB);
	DEC_Image(Real, imageVarAB);
	ComputeLocalVariance(imageAB, mask, imageMeanAB, imageVarAB);

	static thread_local TImage<Real> imageDZNCC;
	const float score(ComputeLocalZNCC(imageA, *imageMeanA, *imageVarA, imageAB, imageMeanAB, imageVarAB, mask, imageDZNCC));
#ifdef MESHOPT_TYPEPOOL
	DST_Image(imageVarAB);
	DST_Image(imageMeanAB);
	if (nReduceMemory) {
		DST_Image(*((TImage<Real>*)imageMeanA));
		DST_Image(*((TImage<Real>*)imageVarA));
	}
	DST_Image(imageAB);
#endif
	// compute field gradient
	const Real RegularizationScale((Real)((REAL)(imageDataA.avgDepth * imageDataB.avgDepth) / (cameraA.GetFocalLength() * cameraB.GetFocalLength())));
	//DEC_BitMatrix(localNorm);
	//jpb wip bug set this here
	//localNorm.memset(0);
	static thread_local std::vector<bool> localNorm;
  localNorm.assign(photoGrad.size(), false);
	ComputePhotometricGradient(faces, faceNormals, viewA, cameraA, cameraB, viewB, imageDZNCC, mask, threadGrad, localNorm, RegularizationScale);

	// threadGrad has been updated.
	// localNorm must be merged to threadNorm:

	for (size_t i = 0, cnt = localNorm.size(); i < cnt; ++i) {
		if (localNorm[i]) {
			++threadNorm[i];
		}
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
			for (int iter = 0; iter < iters; ++iter) {
				refine.iteration = (unsigned)iter;
				refine.nAlternatePair = (iter + 1 < iters ? nAlternatePair : 0);
				refine.ratioRigidityElasticity = (iter <= iterStop ? fRatioRigidityElasticity : 1.f);
				const bool bAdaptMesh(iter >= iterStart && (iter - iterStart) % 3 == 0 && iters - iter > 5);
				// evaluate residuals and gradients
				if (bAdaptMesh)
					refine.vertexDepth.Resize(refine.vertices.GetSize());
				const double cost = refine.ScoreMesh(gradients.data());
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
				}
				else {
					// apply gradients
					FOREACH(v, refine.vertices) {
						Vertex& vert = refine.vertices[v];
						const Point3d grad(gradients.row(v));
						vert -= Cast<Vertex::Type>(grad * gstep);
						gv += norm(grad);
					}
				}
				DEBUG_EXTRA("\t%2d. f: %.5f (%.4e)\tg: %.5f (%.4e - %.4e)\ts: %.3f\tv: %5u", iter + 1, cost, cost / refine.vertices.GetSize(), gradients.norm(), gradients.norm() / refine.vertices.GetSize(), gv / refine.vertices.GetSize(), gstep, numVertsRemoved);
				gstep *= 0.98;
				progress.display(iter);
			}
			GET_LOGCONSOLE().Play();
			progress.close();
		}

#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			mesh.Save(MAKE_PATH(String::FormatString("MeshRefined%u.ply", nScales - nScale - 1)));
#endif
	}

	return true;
} // RefineMesh
/*----------------------------------------------------------------*/
