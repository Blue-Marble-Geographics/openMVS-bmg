/*
 * ReconstructMesh.cpp
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

// TODO: Add archive-type 2
#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include <boost/program_options.hpp>

using namespace MVS;

#undef LOCAL_BUILD // JPB WIP BUG

// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("ReconstructMesh")

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define RECMESH_USE_OPENMP
#endif


// S T R U C T S ///////////////////////////////////////////////////

namespace {

namespace OPT {
String strInputFileName;
String strOutputFileName;
String strMeshFileName;
bool bMeshExport;
float fDistInsert;
bool bUseOnlyROI;
bool bUseConstantWeight;
bool bUseFreeSpaceSupport;
bool bPoisson;
unsigned nPoissonDepth;
float fPoissonTrim;
float fPoissonSamples;
float fPoissonWeight;
float fPoissonIslandRatio;
bool bReleasePointCloud;
float fThicknessFactor;
float fQualityFactor;
float fDecimateMesh;
float fDecimateMeshError;
unsigned nTargetFaceNum;
float fRemoveSpurious;
bool bRemoveSpikes;
unsigned nCloseHoles;
unsigned nSmoothMesh;
float fEdgeLength;
bool bCrop2ROI;
float fBorderROI;
float fSplitMaxArea;
unsigned nArchiveType;
int nProcessPriority;
unsigned nMaxThreads;
String strImagePointsFileName;
String strExportType;
String strConfigFileName;
boost::program_options::variables_map vm;
} // namespace OPT

#ifndef _USE_CUDA
int unused;
#endif

// initialize and parse the command line parameters
bool Initialize(size_t argc, LPCTSTR* argv)
{
	// initialize log and console
	OPEN_LOG();
	OPEN_LOGCONSOLE();

	// group of options allowed only on command line
	boost::program_options::options_description generic("Generic options");
	generic.add_options()
		("help,h", "produce this help message")
		("working-folder,w", boost::program_options::value<std::string>(&WORKING_FOLDER), "working directory (default current directory)")
		("config-file,c", boost::program_options::value<std::string>(&OPT::strConfigFileName)->default_value(APPNAME _T(".cfg")), "file name containing program options")
		("export-type", boost::program_options::value<std::string>(&OPT::strExportType)->default_value(_T("ply")), "file type used to export the 3D scene (ply or obj)")
		("archive-type", boost::program_options::value(&OPT::nArchiveType)->default_value(ARCHIVE_DEFAULT), "project archive type: 0-text, 1-binary, 2-compressed binary")
		("process-priority", boost::program_options::value(&OPT::nProcessPriority)->default_value(-1), "process priority (below normal by default)")
		("max-threads", boost::program_options::value(&OPT::nMaxThreads)->default_value(0), "maximum number of threads (0 for using all available cores)")
		#if TD_VERBOSE != TD_VERBOSE_OFF
		("verbosity,v", boost::program_options::value(&g_nVerbosityLevel)->default_value(
			#if TD_VERBOSE == TD_VERBOSE_DEBUG
			3
			#else
			2
			#endif
			), "verbosity level")
		#endif
		#ifdef _USE_CUDA
		("cuda-device", boost::program_options::value(&SEACAVE::CUDA::desiredDeviceID)->default_value(-1), "CUDA device number to be used to reconstruct the mesh (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
		#else
		("cuda-device", boost::program_options::value(&unused)->default_value(-1), "CUDA device number to be used to reconstruct the mesh (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
		#endif
		;

	// group of options allowed both on command line and in config file
	boost::program_options::options_description config_main("Reconstruct options");
	config_main.add_options()
		("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input filename containing camera poses and image list")
		("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the mesh")
		("min-point-distance,d", boost::program_options::value(&OPT::fDistInsert)->default_value(2.5f), "minimum distance in pixels between the projection of two 3D points to consider them different while triangulating (0 - disabled)")
		("integrate-only-roi", boost::program_options::value(&OPT::bUseOnlyROI)->default_value(false), "use only the points inside the ROI")
		("constant-weight", boost::program_options::value(&OPT::bUseConstantWeight)->default_value(true), "considers all view weights 1 instead of the available weight")
		("free-space-support,f", boost::program_options::value(&OPT::bUseFreeSpaceSupport)->default_value(false), "exploits the free-space support in order to reconstruct weakly-represented surfaces")
		("thickness-factor", boost::program_options::value(&OPT::fThicknessFactor)->default_value(1.f), "multiplier adjusting the minimum thickness considered during visibility weighting")
		("quality-factor", boost::program_options::value(&OPT::fQualityFactor)->default_value(1.f), "multiplier adjusting the quality weight considered during graph-cut")
		("poisson", boost::program_options::value(&OPT::bPoisson)->default_value(false)->implicit_value(true), "use Poisson surface reconstruction (Kazhdan PoissonRecon.exe + SurfaceTrimmer.exe) instead of the Delaunay graph-cut")
		("poisson-depth", boost::program_options::value(&OPT::nPoissonDepth)->default_value(0), "Poisson octree depth (detail; higher = finer/slower); 0 = auto-select from point-cloud density")
		("poisson-trim", boost::program_options::value(&OPT::fPoissonTrim)->default_value(7.f), "SurfaceTrimmer density threshold (>0 trims the open-boundary balloon; 0 - disabled)")
		("poisson-samples", boost::program_options::value(&OPT::fPoissonSamples)->default_value(1.5f), "Poisson samples per node (higher = smoother/coarser, lower = sharper/denser)")
		("poisson-weight", boost::program_options::value(&OPT::fPoissonWeight)->default_value(2.f), "Poisson screened interpolation weight")
		("poisson-island-ratio", boost::program_options::value(&OPT::fPoissonIslandRatio)->default_value(0.f), "SurfaceTrimmer --aRatio + --removeIslands: delete isolated components whose area is below this fraction of the whole mesh (removes small floating blobs); 0 - disabled (stock trimmer behavior)")
		("release-pointcloud", boost::program_options::value(&OPT::bReleasePointCloud)->default_value(true), "release the dense point-cloud once the Poisson solve has taken its geometry, keeping only xyz+normals (frees ~51 bytes/point; measured 2.6GB of a 5.67GB peak at 50.6M points). The saved scene then carries 0 points - the mesh is unaffected, and RefineMesh/TextureMesh do not read them. Set 0 if anything downstream reads points back out of the reconstructed scene (--poisson only)")
		;
	boost::program_options::options_description config_clean("Clean options");
	config_clean.add_options()
		("decimate", boost::program_options::value(&OPT::fDecimateMesh)->default_value(1.f), "decimation factor in range (0..1] to be applied to the reconstructed surface (1 - disabled)")
		("decimate-error", boost::program_options::value(&OPT::fDecimateMeshError)->default_value(0.f), "adaptive error-bounded decimation strength k (0 - disabled, use the --decimate ratio; >0 - stop once the surface has moved ~k*median-edge, so decimation halts while detail remains instead of grinding to the ratio; --decimate is the KEEP FLOOR, so this can only ever keep MORE faces, never fewer). MEASURED: k=1 is far too loose to act -- on a 400-unit scene decimating to 0.4 moved the surface only 0.42mm against a 166mm tolerance, so the floor always won and the setting was inert. Useful values are ~0.001-0.005; read 'DIAG decimate stop' which reports the deviation reached vs allowed in world units and names the k that would bite")
		("target-face-num", boost::program_options::value(&OPT::nTargetFaceNum)->default_value(0), "target number of faces to be applied to the reconstructed surface. (0 - disabled)")
		("remove-spurious", boost::program_options::value(&OPT::fRemoveSpurious)->default_value(20.f), "spurious factor for removing faces with too long edges or isolated components (0 - disabled)")
		("remove-spikes", boost::program_options::value(&OPT::bRemoveSpikes)->default_value(true), "flag controlling the removal of spike faces")
		("close-holes", boost::program_options::value(&OPT::nCloseHoles)->default_value(30), "try to close small holes in the reconstructed surface (0 - disabled)")
		("smooth", boost::program_options::value(&OPT::nSmoothMesh)->default_value(1/* JPB WIP 2*/), "number of iterations to smooth the reconstructed surface (0 - disabled)")
		("edge-length", boost::program_options::value(&OPT::fEdgeLength)->default_value(0.f), "remesh such that the average edge length is this size (0 - disabled)")
		("roi-border", boost::program_options::value(&OPT::fBorderROI)->default_value(0), "add a border to the region-of-interest when cropping the scene (0 - disabled, >0 - percentage, <0 - absolute)")
		("crop-to-roi", boost::program_options::value(&OPT::bCrop2ROI)->default_value(true), "crop scene using the region-of-interest")
		;

	// hidden options, allowed both on command line and
	// in config file, but will not be shown to the user
	boost::program_options::options_description hidden("Hidden options");
	hidden.add_options()
		("mesh-file", boost::program_options::value<std::string>(&OPT::strMeshFileName), "mesh file name to clean (skips the reconstruction step)")
		("mesh-export", boost::program_options::value(&OPT::bMeshExport)->default_value(false), "just export the mesh contained in loaded project")
		("split-max-area", boost::program_options::value(&OPT::fSplitMaxArea)->default_value(0.f), "maximum surface area that a sub-mesh can contain (0 - disabled)")
		("image-points-file", boost::program_options::value<std::string>(&OPT::strImagePointsFileName), "input filename containing the list of points from an image to project on the mesh (optional)")
		;

	boost::program_options::options_description cmdline_options;
	cmdline_options.add(generic).add(config_main).add(config_clean).add(hidden);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config_main).add(config_clean).add(hidden);

	boost::program_options::positional_options_description p;
	p.add("input-file", -1);

	try {
		// parse command line options
		boost::program_options::store(boost::program_options::command_line_parser((int)argc, argv).options(cmdline_options).positional(p).run(), OPT::vm);
		boost::program_options::notify(OPT::vm);
		INIT_WORKING_FOLDER;
		// parse configuration file
		std::ifstream ifs(MAKE_PATH_SAFE(OPT::strConfigFileName));
		if (ifs) {
			boost::program_options::store(parse_config_file(ifs, config_file_options), OPT::vm);
			boost::program_options::notify(OPT::vm);
		}
	}
	catch (const std::exception& e) {
		LOG(e.what());
		return false;
	}

	// initialize the log file
	OPEN_LOGFILE(MAKE_PATH(APPNAME _T("-")+Util::getUniqueName(0)+_T(".log")).c_str());

	// print application details: version and command line
	Util::LogBuild();
	LOG(_T("OpenMVS-bmg build %d"), OPENMVS_BMG_BUILD);
	LOG(_T("Command line: ") APPNAME _T("%s"), Util::CommandLineToString(argc, argv).c_str());

	// validate input
	Util::ensureValidPath(OPT::strInputFileName);
	Util::ensureUnifySlash(OPT::strInputFileName);
	if (OPT::vm.count("help") || OPT::strInputFileName.IsEmpty()) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config_main).add(config_clean);
		GET_LOG() << visible;
	}
	if (OPT::strInputFileName.IsEmpty())
		return false;
	OPT::strExportType = OPT::strExportType.ToLower() == _T("obj") ? _T(".obj") : _T(".ply");

	// initialize optional options
	Util::ensureValidPath(OPT::strOutputFileName);
	Util::ensureUnifySlash(OPT::strOutputFileName);
	Util::ensureValidPath(OPT::strImagePointsFileName);
	if (OPT::strOutputFileName.IsEmpty())
		OPT::strOutputFileName = Util::getFileFullName(OPT::strInputFileName) + _T("_mesh.mvs");

	// initialize global options
	Process::setCurrentProcessPriority((Process::Priority)OPT::nProcessPriority);
	#ifdef _USE_OPENMP
	if (OPT::nMaxThreads != 0)
		omp_set_num_threads(OPT::nMaxThreads);
	#endif

	#ifdef _USE_BREAKPAD
	// start memory dumper
	MiniDumper::Create(APPNAME, WORKING_FOLDER);
	#endif

	Util::Init();
	return true;
}

// finalize application instance
void TFinalize()
{
	MVS::Finalize();

#if TD_VERBOSE != TD_VERBOSE_OFF
	// print memory statistics
	Util::LogMemoryInfo();
	#endif

	CLOSE_LOGFILE();
	CLOSE_LOGCONSOLE();
	CLOSE_LOG();
}

} // unnamed namespace


// export 3D coordinates corresponding to 2D coordinates provided by inputFileName:
// parse image point list; first line is the name of the image to project,
// each consequent line store the xy coordinates to project:
// <image-name> <number-of-points>
// <x-coord1> <y-coord1>
// <x-coord2> <y-coord2>
// ...
// 
// for example:
// N01.JPG 3
// 3090 2680
// 3600 2100
// 3640 2190
bool Export3DProjections(Scene& scene, const String& inputFileName) {
	SML smlPointList(_T("ImagePoints"));
	smlPointList.Load(inputFileName);
	ASSERT(smlPointList.GetArrChildren().size() <= 1);
	IDX idx(0);

	// read image name
	size_t argc;
	CAutoPtrArr<LPSTR> argv;
	while (true) {
		argv = Util::CommandLineToArgvA(smlPointList.GetValue(idx).val, argc);
		if (argc > 0 && argv[0][0] != _T('#'))
			break;
		if (++idx == smlPointList.size())
			return false;
	}
	if (argc < 2)
		return false;
	String imgName(argv[0]);
	IIndex imgID(NO_ID);
	for (const Image& imageData : scene.images) {
		if (!imageData.IsValid())
			continue;
		if (imageData.name.substr(imageData.name.size() - imgName.size()) == imgName) {
			imgID = imageData.ID;
			break;
		}
	}
	if (imgID == NO_ID) {
		VERBOSE("Unable to find image named: %s", imgName.c_str());
		return false;
	}

	// read image points
	std::vector<Point2f> imagePoints;
	while (++idx != smlPointList.size()) {
		// parse image element
		const String& line(smlPointList.GetValue(idx).val);
		argv = Util::CommandLineToArgvA(line, argc);
		if (argc > 0 && argv[0][0] == _T('#'))
			continue;
		if (argc < 2) {
			VERBOSE("Invalid image coordinates: %s", line.c_str());
			continue;
		}
		const Point2f pt(
			String::FromString<float>(argv[0], -1),
			String::FromString<float>(argv[1], -1));
		if (pt.x > 0 && pt.y > 0)
			imagePoints.emplace_back(pt);
	}
	if (imagePoints.empty()) {
		VERBOSE("Unable to read image points from: %s", imgName.c_str());
		return false;
	}

	// prepare output file
	String outFileName(Util::insertBeforeFileExt(inputFileName, "_3D"));
	File oStream(outFileName, File::WRITE, File::CREATE | File::TRUNCATE);
	if (!oStream.isOpen()) {
		VERBOSE("Unable to open output file: %s", outFileName.c_str());
		return false;
	}

	// print image name
	oStream.print("%s %u\n", imgName.c_str(), imagePoints.size());

	// init mesh octree
	const Mesh::Octree octree(scene.mesh.vertices, [](Mesh::Octree::IDX_TYPE size, Mesh::Octree::Type /*radius*/) {
		return size > 256;
	});
	scene.mesh.ListIncidenteFaces();

	// save 3D coord in the output file
	const Image& imgToExport = scene.images[imgID];
	for (const Point2f& pt : imagePoints) {
		// define ray from camera center to each x,y image coord
		const Ray3 ray(imgToExport.camera.C, normalized(imgToExport.camera.RayPoint<REAL>(pt)));
		// find ray intersection with the mesh
		const IntersectRayMesh intRay(octree, ray, scene.mesh);
		if (intRay.pick.IsValid()) {
			const Point3d ptHit(ray.GetPoint(intRay.pick.dist));
			oStream.print("%.7f %.7f %.7f\n", ptHit.x, ptHit.y, ptHit.z);
		} else 
			oStream.print("NA\n");
	}
	return true;
}

void EnableFastFp() {
	_mm_setcsr(_mm_getcsr() | 0x8040); // FTZ | DAZ
}

int main(int argc, LPCTSTR* argv)
{
	#ifdef _DEBUGINFO
	// set _crtBreakAlloc index to stop in <dbgheap.c> at allocation
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);// | _CRTDBG_CHECK_ALWAYS_DF);
	#endif

#pragma omp parallel
		{
			EnableFastFp();
		}

	if (!Initialize(argc, argv))
		return EXIT_FAILURE;

	Scene scene(OPT::nMaxThreads);
	// load project
	if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName), OPT::fSplitMaxArea > 0 || OPT::fDecimateMesh < 1 || OPT::nTargetFaceNum > 0 || OPT::fDecimateMeshError > 0))
		return EXIT_FAILURE;
	const String baseFileName(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName)));
	if (OPT::fSplitMaxArea > 0) {
		// split mesh using max-area constraint
		Mesh::FacesChunkArr chunks;
		if (scene.mesh.Split(chunks, OPT::fSplitMaxArea))
			scene.mesh.Save(chunks, baseFileName);
		TFinalize();
		return EXIT_SUCCESS;
	}

	if (!OPT::strImagePointsFileName.empty() && !scene.mesh.IsEmpty()) {
		Export3DProjections(scene, MAKE_PATH_SAFE(OPT::strImagePointsFileName));
		return EXIT_SUCCESS;
	}

	if (OPT::bMeshExport) {
		// check there is a mesh to export
		if (scene.mesh.IsEmpty())
			return EXIT_FAILURE;
		// save mesh
		const String fileName(MAKE_PATH_SAFE(OPT::strOutputFileName));
		scene.mesh.Save(fileName);
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			scene.ExportCamerasMLP(baseFileName+_T(".mlp"), fileName);
		#endif
	} else {
		const OBB3f initialOBB(scene.obb);
		if (OPT::fBorderROI > 0)
			scene.obb.EnlargePercent(OPT::fBorderROI);
		else if (OPT::fBorderROI < 0)
			scene.obb.Enlarge(-OPT::fBorderROI);
		if (OPT::strMeshFileName.IsEmpty() && scene.mesh.IsEmpty()) {
			// reset image resolution to the original size and
			// make sure the image neighbors are initialized before deleting the point-cloud
#ifdef RECMESH_USE_OPENMP
			bool bAbort(false);
#pragma omp parallel for
			for (int_t idx=0; idx<(int_t)scene.images.GetSize(); ++idx) {
#pragma omp flush (bAbort)
				if (bAbort)
					continue;
				const uint32_t idxImage((uint32_t)idx);
#else
			FOREACH(idxImage, scene.images) {
#endif
				Image& imageData = scene.images[idxImage];
				if (!imageData.IsValid())
					continue;
				// reset image resolution
				if (!imageData.ReloadImage(0, false)) {
#ifdef RECMESH_USE_OPENMP
					bAbort = true;
#pragma omp flush (bAbort)
					continue;
#else
					return EXIT_FAILURE;
#endif
				}
				imageData.UpdateCamera(scene.platforms);
				// select neighbor views
				if (imageData.neighbors.IsEmpty()) {
					IndexArr points;
					scene.SelectNeighborViews(idxImage, points);
				}
			}
#ifdef RECMESH_USE_OPENMP
			if (bAbort)
				return EXIT_FAILURE;
#endif
			// reconstruct a coarse mesh from the given point-cloud
			TD_TIMER_START();
#if 0 // JPB WIP BUG For testing confidence filter
			if (OPT::bUseConstantWeight)
				scene.pointcloud.ReleaseWeights();
#endif
			// JPB TEMP: pass CLI values straight through; tuning has moved
			// to the Mesh::Clean call below (remove-spurious override).
			const bool bMeshOK(OPT::bPoisson
				? scene.ReconstructMeshPoisson((int)OPT::nPoissonDepth, OPT::fPoissonTrim, OPT::fPoissonSamples, OPT::fPoissonWeight, OPT::fPoissonIslandRatio, OPT::bReleasePointCloud)
				: scene.ReconstructMesh(
					OPT::fDistInsert,
					OPT::bUseFreeSpaceSupport,
					OPT::bUseOnlyROI,
					4,
					OPT::fThicknessFactor,
					OPT::fQualityFactor));
			if (!bMeshOK)
				return EXIT_FAILURE;
			VERBOSE("Mesh reconstruction completed: %u vertices, %u faces (%s)", scene.mesh.vertices.GetSize(), scene.mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
			#if TD_VERBOSE != TD_VERBOSE_OFF
			if (VERBOSITY_LEVEL > 2) {
				// dump raw mesh
				scene.mesh.Save(baseFileName+_T("_raw")+OPT::strExportType);
			}
			#endif
		} else if (!OPT::strMeshFileName.IsEmpty()) {
			// load existing mesh to clean
			scene.mesh.Load(MAKE_PATH_SAFE(OPT::strMeshFileName));
		}

		// clean the mesh
		if (OPT::bCrop2ROI && scene.IsBounded()) {
			TD_TIMER_START();
			const size_t numVertices = scene.mesh.vertices.size();
			const size_t numFaces = scene.mesh.faces.size();
			scene.mesh.RemoveFacesOutside(scene.obb);
			VERBOSE("Mesh trimmed to ROI: %u vertices and %u faces removed (%s)",
				numVertices-scene.mesh.vertices.size(), numFaces-scene.mesh.faces.size(), TD_TIMER_GET_FMT().c_str());
		}
		// AUTO FACE CAP for the stages after this one.
		//
		// TextureMesh is by far the most memory-constrained stage in the chain. It holds
		// ~132 observations per face on a 1940-view scene (measured: 975,795,624 obs
		// over 7,388,176 faces) across two structures at 44 B/obs -- perViewOut at 24 B
		// plus facesDatas at 20 B -- so roughly 5.8 KB per FACE. That is ~21x the Poisson
		// solve's measured ~272 B/face, so a mesh that builds comfortably in 22 GB here
		// can need hundreds of GB to texture.
		//
		// Observations per face scale with VIEW COUNT, not with face count: a face is
		// seen by however many cameras happen to cover it, and making faces smaller does
		// not reduce that. So the coefficient is per-view:
		//     obsPerFace ~= kObsPerFacePerView * nViews
		//
		// This is why the cap belongs HERE rather than in the caller: a fixed 132 taken
		// from a 1940-view aerial scene over-decimated a 109-view residential scene by
		// 6x, visibly rounding roof ridges and kerbs. At 109 views the real budget is
		// ~90M faces, i.e. no decimation at all.
		//
		// Only applied when the caller requested neither an explicit target nor a ratio.
		// Applies whenever the caller gave no explicit --target-face-num, INCLUDING when
		// they also gave a --decimate ratio. The two express different intents -- a
		// deliverable-size preference versus a hard downstream memory limit -- so they
		// must coexist rather than one disabling the other, and the more restrictive of
		// the two wins. Previously a --decimate below 1 switched the cap off entirely,
		// which meant a small-scene size preference silently removed the protection the
		// large scenes depend on.
		double capRatio = 1.0;
		if (OPT::nTargetFaceNum == 0 && !scene.mesh.faces.empty()) {
			size_t nViews = 0;
			for (size_t i = 0; i < (size_t)scene.images.GetSize(); ++i)
				if (scene.images[(IIndex)i].IsValid())
					++nViews;
			// Same definition the depth policy bounded itself by, so the two cannot
			// disagree -- see ComputeTextureFaceBudget in SceneReconstruct.cpp, which is
			// at global scope there (that file does `using namespace MVS`, it is not
			// inside the namespace). Declared locally rather than in a header to keep the
			// whole memory model in one translation unit.
			extern double ComputeTextureFaceBudget(size_t nViews, size_t freedBeforeStage);
			// The atlas ceiling, which the RAM one knows nothing about: a face below a
			// few atlas texels carries no texture, so the hardware bounds how many faces
			// are worth keeping regardless of how much memory is free. Surface area
			// cancels out of it, so it holds at any scene scale.
			// PreRefine: the atlas ceiling minus the share reserved for RefineMesh's
			// subdivision (POISSON_ATLAS_REFINE_RESERVE). Identical to the full ceiling at
			// the default 1.0. Both are fetched because the report below has to name the
			// ceiling AND what this stage is holding back from it -- "room for 2.4M faces"
			// on a 4.1M atlas reads as a defect unless the reserve is stated.
			extern double ComputeAtlasFaceBudget(int atlasMaxDim);
			extern double ComputeAtlasFaceBudgetPreRefine(int atlasMaxDim);
			// The atlas dimension plus where it came from -- see ResolveAtlasMaxDimEx.
			// Reported rather than merely used: this is the one decision in the stage
			// that silently shrinks the deliverable, and "atlas 8192" on its own does
			// not tell anyone that this host could have carried 16384 and four times
			// the faces.
			extern int ResolveAtlasMaxDimEx(int atlasMaxDim, int* pHostLimit, int* pEnvPin);
			int atlasHostLimit = 0, atlasEnvPin = 0;
			ResolveAtlasMaxDimEx(0, &atlasHostLimit, &atlasEnvPin);
				// The dimension the SCENE needs, capped by what this host can sample. Derived
				// rather than assumed to be the ceiling: the atlas appetite is
				// surfaceArea/(gsd^2*fill), a scene invariant measured stable to 0.5% across a
				// 2x change in face count. A scene wanting less than the ceiling gets a smaller
				// atlas AND a proportionally smaller face budget, which keeps texels/face on
				// target instead of over-building both. RefineMesh and TextureMesh reach the
				// same number from the same invariant, with nothing passed between the three
				// processes -- no environment variable, no file, no prior run.
				extern int ComputeSceneAtlasDim(const MVS::ImageArr& images, const MVS::Mesh& mesh,
					int* pCeiling, double* pWantDim, double* pSurfaceArea, double* pGsd);
				int atlasCeiling = 0;
				double atlasWantDim = 0.0, atlasSurface = 0.0, atlasGsd = 0.0;
				const int atlasDim = ComputeSceneAtlasDim(scene.images, scene.mesh,
					&atlasCeiling, &atlasWantDim, &atlasSurface, &atlasGsd);
			// 0: the dense cloud has already been released by this point, so availPhys
			// here is what TextureMesh will genuinely see.
			const double budgetMem      = ComputeTextureFaceBudget(nViews, 0);
			// Budgets from the dimension actually CHOSEN, not from the ceiling.
				const double budgetAtlasMax = ComputeAtlasFaceBudget(atlasDim);
			// The atlas figure this stage enforces. Reserve applies to the ATLAS component
			// only: refine's clamp holds the atlas ceiling exactly but knows nothing about
			// RAM, so shrinking the memory budget here would hand the regrowth to a
			// ceiling no later stage checks.
			const double budgetAtlas    = ComputeAtlasFaceBudgetPreRefine(atlasDim);
			const double budget = (budgetMem > 0.0 && budgetAtlas > 0.0)
				? std::min(budgetMem, budgetAtlas)
				: std::max(budgetMem, budgetAtlas);
			if (budget > 0.0) {
				const double nFaces = (double)scene.mesh.faces.size();
				const char* which = (budgetAtlas < budgetMem) ? "ATLAS" : "RAM";
				// WHY this dimension, in one clause. Precedence is env pin > host probe >
				// built-in fallback, so the source also says what the alternative was.
				const char* srcAtlas =
					(atlasEnvPin > 0)     ? "pinned by OPENMVS_ATLAS_MAX_DIM" :
					(atlasHostLimit > 0)  ? "host GPU limit" :
					                        "fallback, no GPU reachable";
				// The largest dimension this host would have allowed. An env pin BELOW the
				// probe is a deliberate operator choice and still worth naming; a pin ABOVE
				// it is an atlas the local GPU cannot sample, which is the more dangerous
				// direction and must not be reported as a win.
				const int atlasCould = atlasCeiling;
				String atlasWant;
				if (atlasWantDim > 0.0 && atlasWantDim <= atlasCould * 1.01)
					atlasWant = String::FormatString(
						", sized to the scene (surface %.4g at gsd %.4g wants %.0f px);"
							" this host allows %d px",
						atlasSurface, atlasGsd, atlasWantDim, atlasCould);
				else if (atlasWantDim > 0.0)
					atlasWant = String::FormatString(
						", CAPPED BY THIS HOST: the scene wants about %.0f px"
							" (%.2fx the texture area) but this GPU samples at most %d px",
							atlasWantDim,
							(atlasWantDim * atlasWantDim) / ((double)atlasDim * atlasDim),
							atlasCould);
					else
						atlasWant = String::FormatString(
							", scene not measurable (no poses or empty mesh) -- using the %d px ceiling",
							atlasCould);
				// NOTE the budget below is enforced against the mesh AS IT STANDS HERE, i.e.
				// BEFORE RefineMesh, which then subdivides by a measured 1.48x-1.88x.
				//
				// That overshoot USED to be unenforced, and this comment documented it as
				// accepted on the grounds that texel SIZE would not improve at all, being
				// set by surface area rather than face count. THAT REASONING IS CORRECT and
				// is now measured: across four runs on the same scene at 7.22M / 4.54M /
				// 4.07M / 3.64M faces, TextureMesh's "full resolution needed about N px"
				// held at 30132 / 30046 / 30028 / 29989 -- a 0.5% spread against a 2x change
				// in face count. Patch size follows the SOURCE PIXELS covering the surface,
				// not the tessellation, so the atlas appetite is a scene constant (~903 Mpx
				// here) and every one of those runs kept the same 0.27-0.28x of the pixels
				// it asked for.
				//
				// What the overshoot DID degrade is texels-per-FACE (37 -> 66 as the count
				// came down), which is what decides whether an individual face has enough
				// atlas area to carry detail, plus patch count and therefore seam length
				// (117583 -> 61648). Those are the wins from enforcing the cap. Texture
				// RESOLUTION is not among them and cannot be bought here -- only a larger
				// atlas moves it.
				//
				// RefineMesh now holds itself inside the same budget instead -- see
				// ClampSubdivideAreaToAtlasBudget in SceneRefine.cpp. It withholds
				// subdivision from the faces with the SMALLEST projected area, so the
				// budget is spent where the imagery supports detail, which is strictly
				// better than either overshooting or decimating back afterwards. So the
				// cap here is the first of two gates rather than the only one, and it no
				// longer needs to anticipate the refine factor.
				// Name the reserve explicitly when one is in force: without it, "room for
				// 2.4M faces" against a 4.1M atlas looks like a miscalculation rather than
				// a deliberate hand-off to the next stage.
				String atlasReserve;
				if (budgetAtlasMax > budgetAtlas)
					atlasReserve = String::FormatString(
						" (atlas ceiling is %.1fM; %.0f%% held back for RefineMesh"
						" subdivision, POISSON_ATLAS_REFINE_RESERVE)",
						budgetAtlasMax * 1e-6,
						100.0 * (1.0 - budgetAtlas / budgetAtlasMax));
				MESH_DIAG("[ATLAS] texture atlas %d px (%s%s) -> room for %.1fM faces%s;"
					" %u views leave room for %.1fM by memory"
					" (enforced here pre-refine; RefineMesh clamps subdivision to the atlas ceiling)",
					atlasDim, srcAtlas, atlasWant.c_str(),
					budgetAtlas * 1e-6, atlasReserve.c_str(),
					(unsigned)nViews, budgetMem * 1e-6);
				if (budget < nFaces) {
					capRatio = budget / nFaces;
					// This CHANGES the deliverable -- it forces decimation the caller did
					// not ask for -- so it stays visible at normal verbosity.
					VERBOSE("[ATLAS] %s-bound: reducing %u faces to %.1fM (x%.3f)",
						which, (unsigned)nFaces, budget * 1e-6, capRatio);
						// WHY the deliverable is smaller than the imagery supports, and what
						// would change it. Without this an operator sees a long densify produce
						// a mesh the atlas then discards most of, with nothing saying that the
						// CEILING -- not the settings -- is what bound it. At a capped atlas the
						// texture resolution is atlas-area / surface-area, so it is the same for
						// every quality tier, and denser densification cannot reach the output.
						if (atlasWantDim > atlasCould * 1.01 && budgetAtlas <= budgetMem) {
							const double wantBudget = ComputeAtlasFaceBudget((int)atlasWantDim);
							VERBOSE("[ATLAS] NOTE: this host's %d px sampling ceiling is the"
								" binding constraint, not the imagery. The scene wants %.0f px,"
								" which would carry %.1fM faces and %.2fx the texture area."
								" At this ceiling %.0f%% of the solved mesh is discarded"
								" regardless of settings, so a denser cloud cannot reach the"
								" output and quality tiers converge here.",
								atlasCould, atlasWantDim, wantBudget * 1e-6,
								(atlasWantDim * atlasWantDim) / ((double)atlasDim * atlasDim),
								100.0 * (1.0 - capRatio));
						}
				} else {
					// nothing was changed, so this is a statement about the budget
					// rather than about the deliverable -- it rides the gate
					MESH_DIAG("[ATLAS] %s-bound at %.1fM faces; mesh has %u -- not reduced",
						which, budget * 1e-6, (unsigned)nFaces);
				}
			}
		}
		float fDecimate(OPT::nTargetFaceNum ? static_cast<float>(OPT::nTargetFaceNum) / scene.mesh.faces.size() : OPT::fDecimateMesh);
		if ((float)capRatio < fDecimate)
			fDecimate = (float)capRatio;
		// Under --poisson, skip ONLY the graph-cut-oriented spurious (long-edge)
		// removal -- the Poisson surface is coherently oriented, and skipping it also
		// preserves the edge extrapolation. Hole-closing STAYS ON: SurfaceTrimmer
		// punches interior holes in low-density regions (textureless roofs, water)
		// that must be filled; disabling it left open interior patches.
		const float cleanSpurious(OPT::bPoisson ? 0.f : OPT::fRemoveSpurious);
		scene.mesh.Clean(fDecimate,
			cleanSpurious, // JPB WIP BUG 15 will give more edges, but not in a good way.
			OPT::bRemoveSpikes,
			OPT::nCloseHoles,
			OPT::nSmoothMesh,
			OPT::fEdgeLength, true, OPT::fDecimateMeshError);

		scene.obb = initialOBB;

		// save the final mesh
#ifdef LOCAL_BUILD
		scene.Save(baseFileName+_T("_rm.mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
#else
		scene.Save(baseFileName + _T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
#endif
		scene.mesh.Save(baseFileName+OPT::strExportType);
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 2)
			scene.ExportCamerasMLP(baseFileName+_T(".mlp"), baseFileName+OPT::strExportType);
		#endif
	}

	if (!OPT::strImagePointsFileName.empty()) {
		Export3DProjections(scene, MAKE_PATH_SAFE(OPT::strImagePointsFileName));
		return EXIT_SUCCESS;
	}

	TFinalize();
	return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/
