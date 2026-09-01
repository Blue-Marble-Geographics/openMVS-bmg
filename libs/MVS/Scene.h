/*
* Scene.h
*
* Copyright (c) 2014-2022 SEACAVE
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

#ifndef _MVS_SCENE_H_
#define _MVS_SCENE_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "SceneDensify.h"
#include "Mesh.h"


// D E F I N E S ///////////////////////////////////////////////////

// Manual build number for this OpenMVS-bmg fork. Bump it by hand when asked.
// It is logged by the ReconstructMesh app and the Poisson reconstruction path
// so any log we produce identifies exactly which build was run.
#ifndef OPENMVS_BMG_BUILD
#define OPENMVS_BMG_BUILD 14
#endif


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

// Forward declarations
struct MVS_API DenseDepthMapData;

class MVS_API Scene
{
public:
	PlatformArr platforms; // camera platforms, each containing the mounted cameras and all known poses
	ImageArr images; // images, each referencing a platform's camera pose

	// JPB Note: Must remain a PointCloudStreaming object to enable a fast stream save.
	PointCloudStreaming pointcloud; // point-cloud (sparse or dense), each containing the point position and the views seeing it
	Mesh mesh; // mesh, represented as vertices and triangles, constructed from the input point-cloud
	OBB3f obb; // optional region-of-interest; oriented bounding box containing the entire scene

	unsigned nCalibratedImages; // number of valid images

	unsigned nMaxThreads; // maximum number of threads used to distribute the work load

public:
	inline Scene(unsigned _nMaxThreads=0)
		: obb(true), nMaxThreads(Thread::getMaxThreads(_nMaxThreads)) {}

	void Release();
	bool IsEmpty() const;
	bool ImagesHaveNeighbors() const;
	bool IsBounded() const { return obb.IsValid(); }

	bool LoadInterface(const String& fileName);
	bool SaveInterface(const String& fileName, int version=-1) const;

	bool LoadDMAP(const String& fileName);
	bool LoadViewNeighbors(const String& fileName);
	bool SaveViewNeighbors(const String& fileName) const;
	bool Import(const String& fileName);

	bool Load(const String& fileName, bool bImport=false);
	bool Save(const String& fileName, ARCHIVE_TYPE type=ARCHIVE_DEFAULT) const;

	bool EstimateNeighborViewsPointCloud(unsigned maxResolution=16);
	void SampleMeshWithVisibility(unsigned maxResolution=320);
	bool ExportMeshToDepthMaps(const String& baseName);

	bool SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews = 3, unsigned nMinPointViews = 2, float fOptimAngle = FD2R(12), unsigned nInsideROI = 1);
	static bool FilterNeighborViews(ViewScoreArr& neighbors, float fMinArea=0.1f, float fMinScale=0.2f, float fMaxScale=2.4f, float fMinAngle=FD2R(3), float fMaxAngle=FD2R(45), unsigned nMaxViews=12);

	bool ExportCamerasMLP(const String& fileName, const String& fileNameScene) const;

	// sub-scene split and save
	struct ImagesChunk {
		std::unordered_set<IIndex> images;
		AABB3f aabb;
	};
	typedef cList<ImagesChunk,const ImagesChunk&,2,16,uint32_t> ImagesChunkArr;
	unsigned Split(ImagesChunkArr& chunks, float maxArea, int depthMapStep=8) const;
	bool ExportChunks(const ImagesChunkArr& chunks, const String& path, ARCHIVE_TYPE type=ARCHIVE_DEFAULT) const;

	// Transform scene
	bool Center(const Point3* pCenter = NULL);
	bool Scale(const REAL* pScale = NULL);
	bool ScaleImages(unsigned nMaxResolution = 0, REAL scale = 0, const String& folderName = String());
	void Transform(const Matrix3x3& rotation, const Point3& translation, REAL scale);
	bool AlignTo(const Scene&);
	REAL ComputeLeveledVolume(float planeThreshold=0, float sampleMesh=-100000, unsigned upAxis=2, bool verbose=true);

	// Estimate and set region-of-interest
	bool EstimateROI(int nEstimateROI=0, float scale=1.f);
	
	// Dense reconstruction
	bool DenseReconstruction(int nFusionMode=0, bool bCrop2ROI=true, float fBorderROI=0);
	bool ComputeDepthMaps(DenseDepthMapData& data);
	void DenseReconstructionEstimate(void*);
	void DenseReconstructionFilter(void*);
	void PointCloudFilter(int thRemove=-1, float maxRemoveFrac=1.f);

	// Mesh reconstruction
	bool ReconstructMesh(float distInsert=2, bool bUseFreeSpaceSupport=true, bool bUseOnlyROI=false, unsigned nItersFixNonManifold=4,
						 float kSigma=2.f, float kQual=1.f, float kb=4.f,
						 float kf=3.f, float kRel=0.1f/*max 0.3*/, float kAbs=1000.f/*min 500*/, float kOutl=400.f/*max 700.f*/,
						 float kInf=(float)(INT_MAX/8));

	// Poisson (Kazhdan PoissonRecon + SurfaceTrimmer, external tools) mesh
	// reconstruction — gated A/B alternative to the graph-cut
	bool ReconstructMeshPoisson(int depth=0, float trimThreshold=7.f, float samplesPerNode=1.5f, float pointWeight=2.f, float islandRatio=0.f, bool releasePointCloud=false);

	// Mesh refinement
	// Deterministic pre-flight sizing check: given this scene and this machine's
	// total RAM, reduces nMaxViews and/or raises nResolutionLevel -- in that order,
	// and only as a last resort -- so RefineMesh's per-batch memory floor fits
	// under this machine's target. No-op on any machine where the scene already
	// fits, which is the common case; call this once, before RefineMesh(CUDA).
	void ResolveRefineMeshSafeSettings(unsigned& nResolutionLevel, unsigned nMinResolution, unsigned& nMaxViews) const;
	bool RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize,
		unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nReduceMemory, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity,
		float fThPlanarVertex, float fGradientStep);
	#ifdef _USE_CUDA
	// Deterministic capability check: true when this machine's CPU refinement path
	// is expected to beat its GPU one, so a CUDA device that was requested should
	// NOT be used. Measured (Richmond Historic): RefineMesh took 44.3 s on a fast
	// desktop CPU (13900K / 7950X class, 64 GB) against 82.6 s on an RTX 3060 12 GB.
	// The bar scales with the device actually installed, so a stronger card still
	// wins; a host with roughly half that CPU throughput, or too little RAM to hold
	// the working set, loses to the 3060 itself. Call this before RefineMeshCUDA; it
	// looks only at static machine and device attributes, never at live load, so the
	// same machine always resolves the same way.
	bool PreferCPUMeshRefinement(unsigned nResolutionLevel, unsigned nMinResolution) const;
	// Pre-flight VRAM estimate for the finest scale of the CUDA path, from scene
	// metadata alone (no decode, no context). False when it cannot be estimated.
	// Used by PreferCPUMeshRefinement so a device too small for the scene is passed
	// over before the run, instead of refusing itself at the last scale.
	bool EstimateRefineMeshCUDAVRAM(unsigned nResolutionLevel, unsigned nMinResolution, uint64_t& needBytes, uint64_t& budgetBytes) const;
	bool RefineMeshCUDA(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize,
		unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fGradientStep);
	#endif

	// Mesh texturing
	bool TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned minCommonCameras=0, float fOutlierThreshold=0.f, float fRatioDataSmoothness=0.3f,
		bool bGlobalSeamLeveling=true, bool bLocalSeamLeveling=true, unsigned nTextureSizeMultiple=0, unsigned nRectPackingHeuristic=3, Pixel8U colEmpty=Pixel8U(255,127,39),
		float fSharpnessWeight=0.5f, int nMaxTextureSize=8192, const IIndexArr& views=IIndexArr());

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ar & platforms;
		ar & images;
		ar & pointcloud;
		ar & mesh;
		ar & obb;
	}
	#endif
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_SCENE_H_
