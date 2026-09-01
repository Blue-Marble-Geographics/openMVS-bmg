/*
 * RefineMesh.cpp
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

#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include <boost/program_options.hpp>
#include <cstdlib>

using namespace MVS;

// Easier to configure this here.
#pragma comment(linker, "/STACK:0x400000,0x400000")

// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("RefineMesh")

#undef LOCAL_BUILD // JPB WIP BUG

// S T R U C T S ///////////////////////////////////////////////////

namespace {

namespace OPT {
String strInputFileName;
String strOutputFileName;
String strMeshFileName;
unsigned nResolutionLevel;
unsigned nMinResolution;
unsigned nMaxViews;
float fDecimateMesh;
unsigned nCloseHoles;
unsigned nEnsureEdgeSize;
unsigned nScales;
float fScaleStep;
unsigned nReduceMemory;
unsigned nAlternatePair;
float fRegularityWeight;
float fRatioRigidityElasticity;
unsigned nMaxFaceArea;
float fPlanarVertexRatio;
float fGradientStep;
unsigned nCudaPolicy;
unsigned nArchiveType;
int nProcessPriority;
unsigned nMaxThreads;
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
		("cuda-device", boost::program_options::value(&SEACAVE::CUDA::desiredDeviceID)->default_value(-2), "CUDA device number to be used for mesh refinement (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
		#else
		("cuda-device", boost::program_options::value(&unused)->default_value(-2), "CUDA device number to be used for mesh refinement (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
		#endif
		("cuda-policy", boost::program_options::value(&OPT::nCudaPolicy)->default_value(0), "which mesh refinement path to run (0 - auto: weigh this host against the requested device and use the GPU only when it wins, since a fast desktop CPU beats a mid-range card here, 1 - always use the GPU, 2 - always use the CPU); overridable at run time with OPENMVS_REFINE_DEVICE=auto|gpu|cpu")
		;

	// group of options allowed both on command line and in config file
	boost::program_options::options_description config("Refine options");
	config.add_options()
		("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input filename containing camera poses and image list")
		("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the mesh")
		("resolution-level", boost::program_options::value(&OPT::nResolutionLevel)->default_value(0), "how many times to scale down the images before mesh refinement")
		("min-resolution", boost::program_options::value(&OPT::nMinResolution)->default_value(640), "do not scale images lower than this resolution")
		("max-views", boost::program_options::value(&OPT::nMaxViews)->default_value(8/* JPB WIP BUG lower values than 8 start losing information*/), "maximum number of neighbor images used to refine the mesh")
		("decimate", boost::program_options::value(&OPT::fDecimateMesh)->default_value(0.f), "decimation factor in range [0..1] to be applied to the input surface before refinement (0 - auto, 1 - disabled)")
		("close-holes", boost::program_options::value(&OPT::nCloseHoles)->default_value(30), "try to close small holes in the input surface (0 - disabled)")
		("ensure-edge-size", boost::program_options::value(&OPT::nEnsureEdgeSize)->default_value(1), "ensure edge size and improve vertex valence of the input surface (0 - disabled, 1 - auto, 2 - force)")
		("max-face-area", boost::program_options::value(&OPT::nMaxFaceArea)->default_value(32), "maximum face area projected in any pair of images that is not subdivided (0 - disabled)")
		("scales", boost::program_options::value(&OPT::nScales)->default_value(3), "how many iterations to run mesh optimization on multi-scale images")
		("scale-step", boost::program_options::value(&OPT::fScaleStep)->default_value(0.5f), "image scale factor used at each mesh optimization step")
		("reduce-memory", boost::program_options::value(&OPT::nReduceMemory)->default_value(1), "recompute some data in order to reduce memory requirements")
		("alternate-pair", boost::program_options::value(&OPT::nAlternatePair)->default_value(0), "refine mesh using an image pair alternatively as reference (0 - both, 1 - alternate, 2 - only left, 3 - only right; applies to the CPU and GPU paths alike. 1X selects mode X on the GPU while leaving the CPU on 'both')")
		("regularity-weight", boost::program_options::value(&OPT::fRegularityWeight)->default_value(0.2f), "scalar regularity weight to balance between photo-consistency and regularization terms during mesh optimization")
		("rigidity-elasticity-ratio", boost::program_options::value(&OPT::fRatioRigidityElasticity)->default_value(0.9f), "scalar ratio used to compute the regularity gradient as a combination of rigidity and elasticity")
		("gradient-step", boost::program_options::value(&OPT::fGradientStep)->default_value(45.05/* JPB WIP BUG 45.05*/), "gradient step to be used instead (0 - auto)")
		("planar-vertex-ratio", boost::program_options::value(&OPT::fPlanarVertexRatio)->default_value(0.f), "threshold used to remove vertices on planar patches (0 - disabled)")
		;

	// hidden options, allowed both on command line and
	// in config file, but will not be shown to the user
	boost::program_options::options_description hidden("Hidden options");
	hidden.add_options()
		("mesh-file", boost::program_options::value<std::string>(&OPT::strMeshFileName), "mesh file name to refine (overwrite the existing mesh)")
		;

	boost::program_options::options_description cmdline_options;
	cmdline_options.add(generic).add(config).add(hidden);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config).add(hidden);

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
	LOG(_T("Command line: ") APPNAME _T("%s"), Util::CommandLineToString(argc, argv).c_str());

	// validate input
	Util::ensureValidPath(OPT::strInputFileName);
	Util::ensureUnifySlash(OPT::strInputFileName);
	if (OPT::vm.count("help") || OPT::strInputFileName.IsEmpty()) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config);
		GET_LOG() << visible;
	}
	if (OPT::strInputFileName.IsEmpty())
		return false;
	OPT::strExportType = OPT::strExportType.ToLower() == _T("obj") ? _T(".obj") : _T(".ply");

	// initialize optional options
	Util::ensureValidPath(OPT::strOutputFileName);
	Util::ensureUnifySlash(OPT::strOutputFileName);
	if (OPT::strOutputFileName.IsEmpty())
		OPT::strOutputFileName = Util::getFileFullName(OPT::strInputFileName) + _T("_refine.mvs");

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
	// load and refine the coarse mesh
	if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName)))
		return EXIT_FAILURE;
	if (!OPT::strMeshFileName.IsEmpty()) {
		// load given coarse mesh
		scene.mesh.Load(MAKE_PATH_SAFE(OPT::strMeshFileName));
	}
	if (scene.mesh.IsEmpty()) {
		VERBOSE("error: empty initial mesh");
		return EXIT_FAILURE;
	}

	// Pre-flight, deterministic memory-safety check: only touches max-views/
	// resolution-level as a last resort, and only on machines where this scene's
	// per-batch memory floor would not otherwise fit (see
	// Scene::ResolveRefineMeshSafeSettings). No-op, and no added cost, on any
	// adequately sized machine -- the common case.
	scene.ResolveRefineMeshSafeSettings(OPT::nResolutionLevel, OPT::nMinResolution, OPT::nMaxViews);

	// Face count entering refinement, for the [REFINE-FACES] accounting line after it
	// finishes. Declared out here rather than inside the try{} so it is still in scope at
	// the completion log, which sits after the catch.
	//
	// ReconstructMesh's atlas cap is applied to the mesh BEFORE this stage, and this stage
	// then subdivides -- measured 1.48x to 1.88x across scenes -- so the mesh TextureMesh
	// actually receives can be well above the budget that cap enforced. Nothing reported
	// that, which is why it went unnoticed.
	const unsigned nFacesBeforeRefine(scene.mesh.faces.GetSize());

	// The atlas dimension, captured HERE and not after refinement. Scene::RefineMesh
	// rescales the cameras per scale and leaves them at the last scale's resolution, so
	// EstimateSceneGSD run afterwards returns a gsd several times too large and the
	// dimension comes out several times too small -- observed as "atlas 10236 px allows
	// 1.6M faces" on a run whose real answer was 16384 px and 4.07M.
	extern int ComputeSceneAtlasDim(const MVS::ImageArr& images, const MVS::Mesh& mesh,
		int* pCeiling, double* pWantDim, double* pSurfaceArea, double* pGsd);
	int atlasCeiling = 0; double atlasWantDim = 0.0;
	const int atlasDim = ComputeSceneAtlasDim(scene.images, scene.mesh,
		&atlasCeiling, &atlasWantDim, NULL, NULL);

	TD_TIMER_START();
	try {
	#ifdef _USE_CUDA
	// RefineMeshCUDA() decimates, subdivides and displaces scene.mesh in place and
	// can bail out at any scale (e.g. the finest one does not fit this GPU's VRAM),
	// so the CPU path below would otherwise resume from a half-refined, already
	// subdivided mesh and subdivide it again. Snapshot the input geometry and put
	// it back before falling through -- everything else on Mesh is derived data
	// that the CPU path rebuilds itself.
	Mesh::VertexArr meshVerticesBackup;
	Mesh::FaceArr meshFacesBackup;

	// Which path to run (--cuda-policy: 0 auto, 1 GPU, 2 CPU). The environment
	// override exists so the two paths can be A/B'd, or one of them pinned in a
	// deployment, without editing the command line the pipeline builds.
	unsigned nPolicy(OPT::nCudaPolicy);
	if (const char* szDevice = std::getenv("OPENMVS_REFINE_DEVICE")) {
		const String device(String(szDevice).ToLower());
		if (device == _T("auto"))
			nPolicy = 0;
		else if (device == _T("gpu") || device == _T("cuda"))
			nPolicy = 1;
		else if (device == _T("cpu"))
			nPolicy = 2;
		else
			VERBOSE("warning: ignoring unrecognized OPENMVS_REFINE_DEVICE='%s' (expected auto, gpu or cpu)", szDevice);
		if (nPolicy != OPT::nCudaPolicy)
			VERBOSE("Mesh refinement device policy overridden by OPENMVS_REFINE_DEVICE=%s", device.c_str());
	}
	if (nPolicy == 1 && SEACAVE::CUDA::desiredDeviceID < -1)
		SEACAVE::CUDA::desiredDeviceID = -1; // GPU demanded but no device asked for: best available

	// A CUDA device was asked for, but a fast host CPU is measurably faster than a
	// mid-range one -- 44.3 s vs 82.6 s on Richmond Historic against an RTX 3060 --
	// so in auto mode the GPU has to earn the run: enough VRAM for the finest scale
	// AND a host slow enough to lose to it. Note the settings passed here are the
	// ones ResolveRefineMeshSafeSettings just resolved, not the requested ones.
	//
	// Whichever way this goes, the log says WHICH device ran: the three suppressed
	// cases below each name themselves in terms of the switch the user set, and in
	// auto mode PreferCPUMeshRefinement names the device it picked. WHY it picked
	// that one is a property of the tuning model rather than of this run, so it is
	// gated (OPENMVS_REFINE_DIAG=1, or -v 4) along with the rest of the refiner's
	// internals -- this level must not paraphrase it either way.
	bool bTryCUDA(nPolicy != 2 && SEACAVE::CUDA::desiredDeviceID >= -1);
	if (nPolicy == 2) {
		VERBOSE("Mesh refinement: using the CPU (pinned by --cuda-policy 2; any requested CUDA device is ignored)");
	} else if (SEACAVE::CUDA::desiredDeviceID < -1) {
		VERBOSE("Mesh refinement: using the CPU (no CUDA device requested, --cuda-device %d)", SEACAVE::CUDA::desiredDeviceID);
	} else if (nPolicy == 1) {
		VERBOSE("Mesh refinement: using the GPU (pinned by --cuda-policy 1)");
	} else if (scene.PreferCPUMeshRefinement(OPT::nResolutionLevel, OPT::nMinResolution)) {
		// PreferCPUMeshRefinement already logged which device it chose; all this adds
		// is how to overrule it.
		VERBOSE("Mesh refinement: pass --cuda-policy 1, or set OPENMVS_REFINE_DEVICE=gpu, to use the requested device anyway");
		bTryCUDA = false;
	}
	if (bTryCUDA) {
		meshVerticesBackup.CopyOf(scene.mesh.vertices);
		meshFacesBackup.CopyOf(scene.mesh.faces);
	}
	bool bRefinedCUDA(false);
	if (bTryCUDA)
		bRefinedCUDA = scene.RefineMeshCUDA(OPT::nResolutionLevel, OPT::nMinResolution, OPT::nMaxViews,
							  OPT::fDecimateMesh, OPT::nCloseHoles, OPT::nEnsureEdgeSize,
							  OPT::nMaxFaceArea,
							  OPT::nScales, OPT::fScaleStep,
							  // alternate-pair applies to this path too. The legacy 1X form
							  // (11/12/13 - mode X on the GPU, "both" on the CPU) still means
							  // what it did; a plain 0..3 now reaches the GPU instead of being
							  // silently demoted to 0 (= both directions, twice the pair work).
							  OPT::nAlternatePair>10 ? OPT::nAlternatePair%10 : OPT::nAlternatePair,
							  OPT::fRegularityWeight,
							  OPT::fRatioRigidityElasticity,
							  OPT::fGradientStep);
	if (bTryCUDA && !bRefinedCUDA) {
		VERBOSE("Mesh refinement: the GPU did not complete this scene; refining on the CPU instead");
		scene.mesh.EmptyExtra();
		scene.mesh.vertices.CopyOfRemove(meshVerticesBackup);
		scene.mesh.faces.CopyOfRemove(meshFacesBackup);
	}
	if (!bRefinedCUDA)
	#else
	// Same guarantee as the CUDA build above: the log always states which device ran,
	// so "why did this refine on the CPU?" is answerable from the log in every build
	// configuration rather than only where a GPU path exists.
	VERBOSE("Mesh refinement: using the CPU (this build has no CUDA support)");
	#endif
	if (!scene.RefineMesh(OPT::nResolutionLevel, OPT::nMinResolution, OPT::nMaxViews,
						  OPT::fDecimateMesh, OPT::nCloseHoles, OPT::nEnsureEdgeSize,
						  OPT::nMaxFaceArea,
						  OPT::nScales, OPT::fScaleStep,
						  OPT::nReduceMemory, OPT::nAlternatePair,
						  OPT::fRegularityWeight,
						  OPT::fRatioRigidityElasticity,
						  OPT::fPlanarVertexRatio,
						  OPT::fGradientStep))
		return EXIT_FAILURE;
	} catch (const std::exception& e) {
		// Turns a genuine OOM (e.g. the PlaneAlloc backstop in SceneRefine.cpp) into
		// a clean, logged failure instead of a crash -- the pre-flight check above
		// is estimate-based and can be wrong; this is what catches it when it is.
		VERBOSE("error: mesh refinement failed (possibly out of memory): %s", e.what());
		return EXIT_FAILURE;
	}
	VERBOSE("Mesh refinement completed: %u vertices, %u faces (%s)", scene.mesh.vertices.GetSize(), scene.mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	if (REFINE_DIAG_ENABLED()) {
		// [REFINE-FACES] The subdivision factor, and what it means for the atlas budget.
		//
		// ReconstructMesh caps faces against the atlas (budgetAtlas = D^2*0.97/64, i.e. a
		// 64-texel-per-face target) and applies that cap to the mesh BEFORE this stage.
		// Subdivision here then multiplies the count. Print both so the relationship is
		// visible instead of having to be reconstructed from three logs.
		//
		// Subdivision now clamps itself to the same budget -- see
		// ClampSubdivideAreaToAtlasBudget in SceneRefine.cpp, honoured by both the CPU and
		// CUDA refiners -- so the mesh SHOULD land inside it and the OVER-budget notice
		// below should no longer fire. If it does fire, that is real information, not the
		// old known-broken state: the clamp bounds the count it can PREDICT, and the two
		// ways past it are the conformity-split fanout being under-estimated (see
		// MESHOPT_ATLAS_BUDGET_SPLIT_FANOUT, a proven upper bound, so this should be
		// impossible) or a later stage adding faces -- EnsureEdgeSize and hole closing
		// both run after Subdivide. Treat a surviving OVER as a bug worth tracing rather
		// than as the documented status quo.
		//
		// On the REFINE_DIAG gate rather than always-on: it names the atlas budget formula
		// and the 64-texel target, which is mechanism, and it is only actionable next to
		// TextureMesh's [ATLAS-FINAL] -- which sits on TEXTURE_DIAG, so a -DOPENMVS_DIAG=1
		// build carries both halves of the texel-budget accounting or neither.
		extern double ComputeAtlasFaceBudget(int atlasMaxDim);
		extern int ResolveAtlasMaxDimEx(int atlasMaxDim, int* pHostLimit, int* pEnvPin);
		const unsigned nFacesAfter(scene.mesh.faces.GetSize());
		// atlasDim was captured BEFORE refinement (see above) -- the cameras are rescaled by
		// then. Audited against the dimension this scene resolved to, which is the atlas the
		// mesh was sized for and the one TextureMesh converges on from its own patch rects.
		const double budgetAtlas = ComputeAtlasFaceBudget(atlasDim);
		const double factor = nFacesBeforeRefine ? (double)nFacesAfter / (double)nFacesBeforeRefine : 0.0;
		const double texelsPerFace = (nFacesAfter && atlasDim > 0)
			? ((double)atlasDim * (double)atlasDim * 0.97) / (double)nFacesAfter : 0.0;
		REFINE_DIAG("[REFINE-FACES] %u -> %u faces (x%.2f) | atlas %d px allows %.1fM faces at"
			" 64 texels/face -- this mesh gets %.0f texels/face%s",
			nFacesBeforeRefine, nFacesAfter, factor,
			atlasDim, budgetAtlas * 1e-6, texelsPerFace,
			(budgetAtlas > 0.0 && (double)nFacesAfter > budgetAtlas)
				? " | OVER the atlas budget: the cap was applied pre-refine" : "");
	}

	// save the final mesh
	const String baseFileName(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName)));
#ifdef LOCAL_BUILD
	scene.Save(baseFileName + _T("2.mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
#else
	scene.Save(baseFileName + _T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
#endif
	scene.mesh.Save(baseFileName+OPT::strExportType);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2)
		scene.ExportCamerasMLP(baseFileName+_T(".mlp"), baseFileName+OPT::strExportType);
	#endif

	TFinalize();
	return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/
