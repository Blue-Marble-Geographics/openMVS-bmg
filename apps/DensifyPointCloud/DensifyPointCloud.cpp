/*
 * DensifyPointCloud.cpp
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

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("DensifyPointCloud")


// S T R U C T S ///////////////////////////////////////////////////

namespace {

namespace OPT {
String strInputFileName;
String strOutputFileName;
String strViewNeighborsFileName;
String strOutputViewNeighborsFileName;
String strMeshFileName;
String strExportROIFileName;
String strImportROIFileName;
String strDenseConfigFileName;
String strExportDepthMapsName;
float fMaxSubsceneArea;
float fSampleMesh;
float fBorderROI;
bool bCrop2ROI;
int nEstimateROI;
int nFusionMode;
int thFilterPointCloud;
int nExportNumViews;
int nArchiveType;
int nProcessPriority;
unsigned nMaxThreads;
String strConfigFileName;
boost::program_options::variables_map vm;
} // namespace OPT

// Used for testing.
#undef FORCIBLY_DISABLE_CUDA

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
		("cuda-device", boost::program_options::value(&SEACAVE::CUDA::desiredDeviceID)->default_value(-1), "CUDA device number to be used for depth-map estimation (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
		#else
		("cuda-device", boost::program_options::value(&unused)->default_value(-2), "CUDA device number to be used for depth-map estimation (-2 - CPU processing, -1 - best GPU, >=0 - device index)" )
		#endif
		;

	// Previously, the _USE_CUDA pathway set numIters to 4 by default.
	// This has the effect of forcing the CPU to 4 patch-match iterations when
	// CUDA use is compiled in AND CPU processing is enabled.
	// Now both default to DPC_NUM_ITERS (3) here, and when --iters was NOT given
	// explicitly AND a CUDA device is actually present, nEstimationIters is
	// raised back to 4 after option parsing (see the block before Util::Init()):
	// the GPU's parallel checkerboard propagation converges slower per iteration
	// than the CPU's sequential raster sweeps, so it needs the extra iteration
	// (this matches upstream, which always defaulted the CUDA build to 4).
#ifdef FORCIBLY_DISABLE_CUDA
	const unsigned nNumViewsDefault(5); // 0, 1, 8, 12 Doesn't make much of a difference?
	const unsigned numIters(DPC_NUM_ITERS);
#else
	// group of options allowed both on command line and in config file
	#ifdef _USE_CUDA
	const unsigned nNumViewsDefault(8);
	const unsigned numIters(DPC_NUM_ITERS);
	#else
	const unsigned nNumViewsDefault(5);
	const unsigned numIters(DPC_NUM_ITERS);
	#endif
#endif
	unsigned nResolutionLevel;
	unsigned nMaxResolution;
	unsigned nMinResolution;
	unsigned nNumViews;
	unsigned nMinViewsFuse;
	unsigned nSubResolutionLevels;
	unsigned nEstimationIters;
	unsigned nEstimationGeometricIters;
	unsigned nEstimateColors;
	unsigned nEstimateNormals;
	unsigned nOptimize;
	int nIgnoreMaskLabel;
	bool bRemoveDmaps;
	boost::program_options::options_description config("Densify options");
	config.add_options()
		("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input filename containing camera poses and image list")
		("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the dense point-cloud (optional)")
		("view-neighbors-file", boost::program_options::value<std::string>(&OPT::strViewNeighborsFileName), "input filename containing the list of views and their neighbors (optional)")
		("output-view-neighbors-file", boost::program_options::value<std::string>(&OPT::strOutputViewNeighborsFileName), "output filename containing the generated list of views and their neighbors")
		("resolution-level", boost::program_options::value(&nResolutionLevel)->default_value(1), "how many times to scale down the images before point cloud computation")
		("max-resolution", boost::program_options::value(&nMaxResolution)->default_value(2560), "do not scale images higher than this resolution")
		("min-resolution", boost::program_options::value(&nMinResolution)->default_value(640), "do not scale images lower than this resolution")
		("sub-resolution-levels", boost::program_options::value(&nSubResolutionLevels)->default_value(2), "number of patch-match sub-resolution iterations (0 - disabled)")
		("number-views", boost::program_options::value(&nNumViews)->default_value(nNumViewsDefault), "number of views used for depth-map estimation (0 - all neighbor views available)")
		("number-views-fuse", boost::program_options::value(&nMinViewsFuse)->default_value(3), "minimum number of images that agrees with an estimate during fusion in order to consider it inlier (<2 - only merge depth-maps)")
		("ignore-mask-label", boost::program_options::value(&nIgnoreMaskLabel)->default_value(-1), "integer value for the label to ignore in the segmentation mask (<0 - disabled)")
		("iters", boost::program_options::value(&nEstimationIters)->default_value(numIters), "number of patch-match iterations")
		("geometric-iters", boost::program_options::value(&nEstimationGeometricIters)->default_value(4 /* makes the cloud smaller 2*/ ), "number of geometric consistent patch-match iterations (0 - disabled)")
		("estimate-colors", boost::program_options::value(&nEstimateColors)->default_value(2), "estimate the colors for the dense point-cloud (0 - disabled, 1 - final, 2 - estimate)")
		("estimate-normals", boost::program_options::value(&nEstimateNormals)->default_value(2), "estimate the normals for the dense point-cloud (0 - disabled, 1 - final, 2 - estimate)")
		("sub-scene-area", boost::program_options::value(&OPT::fMaxSubsceneArea)->default_value(0.f), "split the scene in sub-scenes such that each sub-scene surface does not exceed the given maximum sampling area (0 - disabled)")
		("sample-mesh", boost::program_options::value(&OPT::fSampleMesh)->default_value(0.f), "uniformly samples points on a mesh (0 - disabled, <0 - number of points, >0 - sample density per square unit)")
		("fusion-mode", boost::program_options::value(&OPT::nFusionMode)->default_value(0), "depth-maps fusion mode (-2 - fuse disparity-maps, -1 - export disparity-maps only, 0 - depth-maps & fusion, 1 - export depth-maps only)")
		("postprocess-dmaps", boost::program_options::value(&nOptimize)->default_value(6 /* JPB WIP BUG Was 4 *//* 2.4 does this JPB WIP BUG  7 */), "flags used to filter the depth-maps after estimation (0 - disabled, 1 - remove-speckles, 2 - fill-gaps, 4 - adjust-filter)")
		("filter-point-cloud", boost::program_options::value(&OPT::thFilterPointCloud)->default_value(0), "filter dense point-cloud based on visibility (0 - disabled, <0 - standalone filter+exit, >0 - inline post-fusion: remove points with visibility<=-N; LARGER N removes fewer, ~20-50 = plume-only)")
		("export-number-views", boost::program_options::value(&OPT::nExportNumViews)->default_value(0), "export points with >= number of views (0 - disabled, <0 - save MVS project too)")
		("roi-border", boost::program_options::value(&OPT::fBorderROI)->default_value(0), "add a border to the region-of-interest when cropping the scene (0 - disabled, >0 - percentage, <0 - absolute)")
		("estimate-roi", boost::program_options::value(&OPT::nEstimateROI)->default_value(2), "estimate and set region-of-interest (0 - disabled, 1 - enabled, 2 - adaptive)")
		("crop-to-roi", boost::program_options::value(&OPT::bCrop2ROI)->default_value(true), "crop scene using the region-of-interest")
		("remove-dmaps", boost::program_options::value(&bRemoveDmaps)->default_value(true), "remove depth-maps after fusion")
		;

	// hidden options, allowed both on command line and
	// in config file, but will not be shown to the user
	boost::program_options::options_description hidden("Hidden options");
	hidden.add_options()
		("mesh-file", boost::program_options::value<std::string>(&OPT::strMeshFileName), "mesh file name used for image pair overlap estimation")
		("export-roi-file", boost::program_options::value<std::string>(&OPT::strExportROIFileName), "ROI file name to be exported form the scene")
		("import-roi-file", boost::program_options::value<std::string>(&OPT::strImportROIFileName), "ROI file name to be imported into the scene")
		("dense-config-file", boost::program_options::value<std::string>(&OPT::strDenseConfigFileName), "optional configuration file for the densifier (overwritten by the command line options)")
		("export-depth-maps-name", boost::program_options::value<std::string>(&OPT::strExportDepthMapsName), "render given mesh and save the depth-map for every image to this file name base (empty - disabled)")
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

#ifdef FORCIBLY_DISABLE_CUDA
#ifdef _USE_CUDA
	VERBOSE("Build was created with CUDA support, but CPU processing will be used regardless of command line setting.");
#else
	VERBOSE("Build was not created with CUDA support and CPU processing will be used.");
#endif
#endif
	// validate input
	Util::ensureValidPath(OPT::strInputFileName);
	if (OPT::vm.count("help") || OPT::strInputFileName.empty()) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config);
		GET_LOG() << visible;
	}
	if (OPT::strInputFileName.empty())
		return false;

	// initialize optional options
	Util::ensureValidPath(OPT::strOutputFileName);
	Util::ensureValidPath(OPT::strViewNeighborsFileName);
	Util::ensureValidPath(OPT::strOutputViewNeighborsFileName);
	Util::ensureValidPath(OPT::strMeshFileName);
	Util::ensureValidPath(OPT::strExportROIFileName);
	Util::ensureValidPath(OPT::strImportROIFileName);
	if (OPT::strOutputFileName.empty())
		OPT::strOutputFileName = Util::getFileFullName(OPT::strInputFileName) + _T("_dense.mvs");

	// init dense options
	if (!OPT::strDenseConfigFileName.empty())
		OPT::strDenseConfigFileName = MAKE_PATH_SAFE(OPT::strDenseConfigFileName);
	OPTDENSE::init();
	const bool bValidConfig(OPTDENSE::oConfig.Load(OPT::strDenseConfigFileName));
	OPTDENSE::update();
	OPTDENSE::nResolutionLevel = nResolutionLevel;
	OPTDENSE::nMaxResolution = nMaxResolution;
	OPTDENSE::nMinResolution = nMinResolution;
	OPTDENSE::nSubResolutionLevels = nSubResolutionLevels;
	OPTDENSE::nNumViews = nNumViews;
	OPTDENSE::nMinViewsFuse = nMinViewsFuse;
	OPTDENSE::fNCCThresholdKeep = 0.45f; // JPB WIP BUG override (default ~0.55) -- loosen depth-map filter to fill holes in textureless / canopy regions
	OPTDENSE::nEstimationIters = nEstimationIters;
	OPTDENSE::nEstimationGeometricIters = nEstimationGeometricIters;
	OPTDENSE::nEstimateColors = nEstimateColors;
	OPTDENSE::nEstimateNormals = nEstimateNormals;
	OPTDENSE::nOptimize = nOptimize;
	OPTDENSE::nIgnoreMaskLabel = nIgnoreMaskLabel;
	OPTDENSE::bRemoveDmaps = bRemoveDmaps;
	if (!bValidConfig && !OPT::strDenseConfigFileName.empty())
		OPTDENSE::oConfig.Save(OPT::strDenseConfigFileName);

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

#ifdef FORCIBLY_DISABLE_CUDA
#ifdef _USE_CUDA
		CUDA::desiredDeviceID = -2;
#endif
#endif

	#ifdef _USE_CUDA
	// GPU checkerboard propagation needs one more photometric iteration than the
	// CPU's sequential sweeps to converge equally (see the numIters note above).
	// Apply it only when the user did not set --iters explicitly (command line or
	// config file) and a usable CUDA device is actually present, so an explicit
	// choice always wins and the CPU fallback keeps DPC_NUM_ITERS.
	if (OPT::vm["iters"].defaulted() && SEACAVE::CUDA::desiredDeviceID >= -1) {
		#ifdef _MSC_VER
		// probe the driver DLL before any cu* call: nvcuda.dll is delay-loaded,
		// so calling into it on a machine without an NVIDIA driver would raise a
		// delay-load exception instead of failing gracefully
		const bool bCudaDriverPresent(GetModuleHandleA("nvcuda.dll") != NULL || LoadLibraryA("nvcuda.dll") != NULL);
		#else
		const bool bCudaDriverPresent(true);
		#endif
		SEACAVE::CUDA::DeviceCapability cap;
		if (bCudaDriverPresent && SEACAVE::CUDA::GetDeviceCapability(SEACAVE::CUDA::desiredDeviceID, cap)) {
			OPTDENSE::nEstimationIters = 4;
			DEBUG("Patch-match iterations raised to 4 for the CUDA estimator (device %d: %s)", cap.deviceID, cap.name);
		}
	}
	#endif // _USE_CUDA

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

// Reorder the final point-cloud in Morton (Z-order) so that spatially-adjacent
// points are stored consecutively in the output PLY. Downstream importers that
// skip re-sorting already well-organized clouds (e.g. Global Mapper's
// "poorly organized" heuristic) can then avoid an expensive spatial sort on
// load. This is a latency shift (the sort happens here, during the long dense
// step, instead of in the consumer) not a total-CPU reduction.
#ifndef DENSIFY_MORTON_SORT_OUTPUT
#define DENSIFY_MORTON_SORT_OUTPUT 1
#endif
#if DENSIFY_MORTON_SORT_OUTPUT
namespace {
// spread the lowest 21 bits of v so each occupies every 3rd output bit
inline uint64_t MortonSpread3(uint64_t v) {
	v &= 0x1FFFFFULL;                          // keep 21 bits
	v = (v | (v << 32)) & 0x1F00000000FFFFULL;
	v = (v | (v << 16)) & 0x1F0000FF0000FFULL;
	v = (v | (v <<  8)) & 0x100F00F00F00F00FULL;
	v = (v | (v <<  4)) & 0x10C30C30C30C30C3ULL;
	v = (v | (v <<  2)) & 0x1249249249249249ULL;
	return v;
}
void ReorderPointCloudMorton(PointCloudStreaming& pc) {
	const size_t numPoints(pc.GetSize());
	if (numPoints < 2)
		return;
	TD_TIMER_START();
	// Raw base pointer to the packed XYZ array (3 floats/point); used for the
	// bounding-box reduction and Morton-key generation below, both of which run
	// before any array is swapped, so the pointer stays valid throughout.
	const float* const __restrict pts = pc.pointsXYZ.data();
	// axis-aligned bounding box of the points (pointsXYZ is 3 floats per point).
	// MSVC's OpenMP is 2.0 (no min/max reduction), so reduce manually with
	// per-thread locals merged under a single critical.
	float minX = std::numeric_limits<float>::max(), minY = minX, minZ = minX;
	float maxX = -minX, maxY = -minX, maxZ = -minX;
	#ifdef _USE_OPENMP
	#pragma omp parallel
	{
		float lMinX = std::numeric_limits<float>::max(), lMinY = lMinX, lMinZ = lMinX;
		float lMaxX = -lMinX, lMaxY = -lMinX, lMaxZ = -lMinX;
		#pragma omp for nowait
		for (int64_t i = 0; i < (int64_t)numPoints; ++i) {
			const float x = pts[i*3+0], y = pts[i*3+1], z = pts[i*3+2];
			if (x < lMinX) lMinX = x; if (x > lMaxX) lMaxX = x;
			if (y < lMinY) lMinY = y; if (y > lMaxY) lMaxY = y;
			if (z < lMinZ) lMinZ = z; if (z > lMaxZ) lMaxZ = z;
		}
		#pragma omp critical
		{
			if (lMinX < minX) minX = lMinX; if (lMaxX > maxX) maxX = lMaxX;
			if (lMinY < minY) minY = lMinY; if (lMaxY > maxY) maxY = lMaxY;
			if (lMinZ < minZ) minZ = lMinZ; if (lMaxZ > maxZ) maxZ = lMaxZ;
		}
	}
	#else
	for (size_t i = 0; i < numPoints; ++i) {
		const float x = pts[i*3+0], y = pts[i*3+1], z = pts[i*3+2];
		if (x < minX) minX = x; if (x > maxX) maxX = x;
		if (y < minY) minY = y; if (y > maxY) maxY = y;
		if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
	}
	#endif
	const double sx = (maxX > minX) ? (2097151.0 / (double)(maxX - minX)) : 0.0;
	const double sy = (maxY > minY) ? (2097151.0 / (double)(maxY - minY)) : 0.0;
	const double sz = (maxZ > minZ) ? (2097151.0 / (double)(maxZ - minZ)) : 0.0;
	// (Morton key, original index) pairs
	std::vector<uint64_t> keys(numPoints);
	std::vector<uint32_t> idx(numPoints);
	uint64_t* const __restrict pKeys = keys.data();
	uint32_t* const __restrict pIdxInit = idx.data();
	#ifdef _USE_OPENMP
	#pragma omp parallel for
	#endif
	for (int64_t i = 0; i < (int64_t)numPoints; ++i) {
		const float x = pts[i*3+0], y = pts[i*3+1], z = pts[i*3+2];
		const uint64_t gx = (uint64_t)((double)(x - minX) * sx);
		const uint64_t gy = (uint64_t)((double)(y - minY) * sy);
		const uint64_t gz = (uint64_t)((double)(z - minZ) * sz);
		pKeys[i] = MortonSpread3(gx) | (MortonSpread3(gy) << 1) | (MortonSpread3(gz) << 2);
		pIdxInit[i] = (uint32_t)i;
	}
	// Stable, fully parallel LSD radix sort on the 64-bit Morton key (8 byte-
	// passes, ping-pong buffers). Linear-time and branch-free vs O(n log n)
	// std::sort. Each pass: (A) per-thread local histograms over disjoint
	// contiguous chunks, (B) a small serial exclusive prefix over the
	// [bucket-major, thread-minor] count matrix that hands every thread a
	// private write cursor per bucket (this keeps the scatter both stable AND
	// lock-free), (C) a fully parallel scatter. Byte-passes whose key bits are
	// all-constant are skipped (the high bytes of a 21-bit-per-axis Morton key
	// are usually uniform). Chunk boundaries are derived solely from nThreads
	// and the thread id, so phase A and phase C partition the input identically
	// (required for the cursors to be correct); MSVC honors num_threads exactly.
	{
		std::vector<uint64_t> keysTmp(numPoints);
		std::vector<uint32_t> idxTmp(numPoints);
		uint64_t* __restrict kIn = keys.data();   uint64_t* __restrict kOut = keysTmp.data();
		uint32_t* __restrict iIn = idx.data();    uint32_t* __restrict iOut = idxTmp.data();
		const int R = 256;
		#ifdef _USE_OPENMP
		const int nThreads = std::max(1, omp_get_max_threads());
		#else
		const int nThreads = 1;
		#endif
		// [thread][bucket] count/cursor matrix (each thread owns one R-wide row)
		std::vector<size_t> hist((size_t)nThreads * R);
		size_t* const __restrict pHist = hist.data();
		for (int pass = 0; pass < 8; ++pass) {
			const int shift = pass * 8;
			std::fill(hist.begin(), hist.end(), (size_t)0);
			// (A) per-thread local histograms over fixed contiguous chunks
			#ifdef _USE_OPENMP
			#pragma omp parallel num_threads(nThreads)
			#endif
			{
				#ifdef _USE_OPENMP
				const int t = omp_get_thread_num();
				#else
				const int t = 0;
				#endif
				const size_t begin = (numPoints * (size_t)t) / nThreads;
				const size_t end   = (numPoints * (size_t)(t + 1)) / nThreads;
				size_t* const __restrict h = pHist + (size_t)t * R;
				for (size_t n = begin; n < end; ++n)
					++h[(kIn[n] >> shift) & 0xFF];
			}
			// totals for the first key's bucket -> constant-byte skip test
			const int firstByte = (int)((kIn[0] >> shift) & 0xFF);
			size_t firstByteTotal = 0;
			for (int t = 0; t < nThreads; ++t)
				firstByteTotal += pHist[(size_t)t * R + firstByte];
			if (firstByteTotal == numPoints)
				continue;
			// (B) exclusive prefix over (bucket-major, thread-minor): each slot
			// becomes that thread's starting output cursor for that bucket
			size_t running = 0;
			for (int b = 0; b < R; ++b)
				for (int t = 0; t < nThreads; ++t) {
					size_t& slot = pHist[(size_t)t * R + b];
					const size_t c = slot;
					slot = running;
					running += c;
				}
			// (C) fully parallel scatter using each thread's private cursors
			#ifdef _USE_OPENMP
			#pragma omp parallel num_threads(nThreads)
			#endif
			{
				#ifdef _USE_OPENMP
				const int t = omp_get_thread_num();
				#else
				const int t = 0;
				#endif
				const size_t begin = (numPoints * (size_t)t) / nThreads;
				const size_t end   = (numPoints * (size_t)(t + 1)) / nThreads;
				size_t* const __restrict cur = pHist + (size_t)t * R;
				for (size_t n = begin; n < end; ++n) {
					const size_t pos = cur[(kIn[n] >> shift) & 0xFF]++;
					kOut[pos] = kIn[n];
					iOut[pos] = iIn[n];
				}
			}
			std::swap(kIn, kOut);
			std::swap(iIn, iOut);
		}
		// skipped passes may leave the result in the temp buffers; ensure idx
		// holds the final permutation.
		if (iIn != idx.data())
			std::copy(iIn, iIn + numPoints, idx.data());
	}
	// gather the dense per-point arrays (points / normals / colors) into Morton
	// order, then swap back. Fused into a single pass: idx[i] is read once and
	// the three independent random reads per iteration expose more memory-level
	// parallelism (better DRAM-latency hiding) than three separate passes. The
	// has-normals/has-colors tests are loop-invariant flags (well-predicted,
	// near-free), so absent optional arrays cost nothing. Each output slot is
	// written once from a distinct source index, so the loop is parallel.
	{
		const bool bHasNormals = (pc.normalsXYZ.size() == numPoints*3);
		const bool bHasColors  = (pc.colorsRGB.size()  == numPoints*3);
		std::vector<float> tmpPts(numPoints*3);
		std::vector<float> tmpNrm(bHasNormals ? numPoints*3 : 0);
		std::vector<uint8_t> tmpCol(bHasColors ? numPoints*3 : 0);
		const float*   const __restrict srcPts = pc.pointsXYZ.data();
		const float*   const __restrict srcNrm = bHasNormals ? pc.normalsXYZ.data() : nullptr;
		const uint8_t* const __restrict srcCol = bHasColors  ? pc.colorsRGB.data()  : nullptr;
		float*   const __restrict dstPts = tmpPts.data();
		float*   const __restrict dstNrm = bHasNormals ? tmpNrm.data() : nullptr;
		uint8_t* const __restrict dstCol = bHasColors  ? tmpCol.data() : nullptr;
		const uint32_t* const __restrict pIdx = idx.data();
		#ifdef _USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i = 0; i < (int64_t)numPoints; ++i) {
			const size_t s = (size_t)pIdx[i]*3;
			const size_t d = (size_t)i*3;
			dstPts[d+0] = srcPts[s+0];
			dstPts[d+1] = srcPts[s+1];
			dstPts[d+2] = srcPts[s+2];
			if (bHasNormals) {
				dstNrm[d+0] = srcNrm[s+0];
				dstNrm[d+1] = srcNrm[s+1];
				dstNrm[d+2] = srcNrm[s+2];
			}
			if (bHasColors) {
				dstCol[d+0] = srcCol[s+0];
				dstCol[d+1] = srcCol[s+1];
				dstCol[d+2] = srcCol[s+2];
			}
		}
		pc.pointsXYZ.swap(tmpPts);
		if (bHasNormals) pc.normalsXYZ.swap(tmpNrm);
		if (bHasColors)  pc.colorsRGB.swap(tmpCol);
	}
	// CSR per-point variable-length arrays: views and weights
	// (rebuild offsets/sizes/memory in the new point order). Two passes: a
	// cheap serial prefix-sum to lay out sizes+offsets, then a parallel
	// block-copy into the pre-sized destination (each block is independent).
	if (pc.pointViewsSizes.size() == numPoints) {
		std::vector<uint32_t> newOffsets(numPoints), newSizes(numPoints);
		std::vector<uint32_t> newMemory(pc.pointViewsMemory.size());
		const uint32_t* const __restrict pIdx = idx.data();
		uint32_t off = 0;
		for (size_t i = 0; i < numPoints; ++i) {
			const uint32_t old = pIdx[i];
			const uint32_t sz = pc.pointViewsSizes[old];
			newSizes[i] = sz;
			newOffsets[i] = off;
			off += sz;
		}
		const uint32_t* const __restrict srcMem  = pc.pointViewsMemory.data();
		const uint32_t* const __restrict srcOff  = pc.pointViewsOffsets.data();
		const uint32_t* const __restrict pSizes  = newSizes.data();
		const uint32_t* const __restrict pOffs   = newOffsets.data();
		uint32_t* const __restrict dstMem = newMemory.data();
		#ifdef _USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i = 0; i < (int64_t)numPoints; ++i) {
			const uint32_t oldOff = srcOff[pIdx[i]];
			const uint32_t sz = pSizes[i];
			const uint32_t dst = pOffs[i];
			for (uint32_t j = 0; j < sz; ++j)
				dstMem[dst+j] = srcMem[oldOff+j];
		}
		pc.pointViewsOffsets.swap(newOffsets);
		pc.pointViewsSizes.swap(newSizes);
		pc.pointViewsMemory.swap(newMemory);
	}
	if (pc.pointWeightsSizes.size() == numPoints) {
		std::vector<uint32_t> newOffsets(numPoints), newSizes(numPoints);
		std::vector<float> newMemory(pc.pointWeightsMemory.size());
		const uint32_t* const __restrict pIdx = idx.data();
		uint32_t off = 0;
		for (size_t i = 0; i < numPoints; ++i) {
			const uint32_t old = pIdx[i];
			const uint32_t sz = pc.pointWeightsSizes[old];
			newSizes[i] = sz;
			newOffsets[i] = off;
			off += sz;
		}
		const float*    const __restrict srcMem = pc.pointWeightsMemory.data();
		const uint32_t* const __restrict srcOff = pc.pointWeightsOffsets.data();
		const uint32_t* const __restrict pSizes = newSizes.data();
		const uint32_t* const __restrict pOffs  = newOffsets.data();
		float* const __restrict dstMem = newMemory.data();
		#ifdef _USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i = 0; i < (int64_t)numPoints; ++i) {
			const uint32_t oldOff = srcOff[pIdx[i]];
			const uint32_t sz = pSizes[i];
			const uint32_t dst = pOffs[i];
			for (uint32_t j = 0; j < sz; ++j)
				dstMem[dst+j] = srcMem[oldOff+j];
		}
		pc.pointWeightsOffsets.swap(newOffsets);
		pc.pointWeightsSizes.swap(newSizes);
		pc.pointWeightsMemory.swap(newMemory);
	}
	// instrumentation for an internal cache-locality reorder: the count and the
	// timing say nothing about the reconstruction, so it rides the diag gate
	DENSIFY_DIAG("Point-cloud reordered in Morton order: %u points (%s)", (unsigned)numPoints, TD_TIMER_GET_FMT().c_str());
}
} // namespace
#endif

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
	if (OPT::fSampleMesh != 0) {
#if 1 // JPB WIP BUG
		throw std::runtime_error("Unsupported");
#else
		// sample input mesh and export the obtained point-cloud
		if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName), true) || scene.mesh.IsEmpty())
			return EXIT_FAILURE;
		TD_TIMER_START();
		PointCloud pointcloud;
		if (OPT::fSampleMesh > 0)
			scene.mesh.SamplePoints(OPT::fSampleMesh, 0, pointcloud);
		else
			scene.mesh.SamplePoints(ROUND2INT<unsigned>(-OPT::fSampleMesh), pointcloud);
		VERBOSE("Sample mesh completed: %u points (%s)", pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
		pointcloud.Save(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName))+_T(".ply"));
		TFinalize();
		return EXIT_SUCCESS;
#endif
	}
	// load and estimate a dense point-cloud
	if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName)))
		return EXIT_FAILURE;
	if (!OPT::strImportROIFileName.empty()) {
		std::ifstream fs(MAKE_PATH_SAFE(OPT::strImportROIFileName));
		if (!fs)
			return EXIT_FAILURE;
		fs >> scene.obb;
		scene.Save(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName))+_T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
		TFinalize();
		return EXIT_SUCCESS;
	}
	if (!scene.IsBounded())
		scene.EstimateROI(OPT::nEstimateROI, 1.1f);
	if (!OPT::strExportROIFileName.empty() && scene.IsBounded()) {
		std::ofstream fs(MAKE_PATH_SAFE(OPT::strExportROIFileName));
		if (!fs)
			return EXIT_FAILURE;
		fs << scene.obb;
		TFinalize();
		return EXIT_SUCCESS;
	}
	if (!OPT::strMeshFileName.empty())
		scene.mesh.Load(MAKE_PATH_SAFE(OPT::strMeshFileName));
	if (!OPT::strViewNeighborsFileName.empty())
		scene.LoadViewNeighbors(MAKE_PATH_SAFE(OPT::strViewNeighborsFileName));
	if (!OPT::strOutputViewNeighborsFileName.empty()) {
		if (!scene.ImagesHaveNeighbors()) {
			VERBOSE("error: neighbor views not computed yet");
			return EXIT_FAILURE;
		}
		scene.SaveViewNeighbors(MAKE_PATH_SAFE(OPT::strOutputViewNeighborsFileName));
		return EXIT_SUCCESS;
	}
	if (!OPT::strExportDepthMapsName.empty() && !scene.mesh.IsEmpty()) {
		// project mesh onto each image and save the resulted depth-maps
		TD_TIMER_START();
		if (!scene.ExportMeshToDepthMaps(MAKE_PATH_SAFE(OPT::strExportDepthMapsName)))
			return EXIT_FAILURE;
		VERBOSE("Mesh projection completed: %u depth-maps (%s)", scene.images.size(), TD_TIMER_GET_FMT().c_str());
		TFinalize();
		return EXIT_SUCCESS;
	}
	if (OPT::fMaxSubsceneArea > 0) {
#if 1 // JPB WIP BUG
		throw std::runtime_error("Unsupported");
#else
		// split the scene in sub-scenes by maximum sampling area
		Scene::ImagesChunkArr chunks;
		scene.Split(chunks, OPT::fMaxSubsceneArea);
		scene.ExportChunks(chunks, GET_PATH_FULL(OPT::strOutputFileName), (ARCHIVE_TYPE)OPT::nArchiveType);
		TFinalize();
		return EXIT_SUCCESS;
#endif
	}
	if (OPT::thFilterPointCloud < 0) {
		// filter point-cloud based on camera-point visibility intersections
		scene.PointCloudFilter(OPT::thFilterPointCloud);
		const String baseFileName(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName))+_T("_filtered"));
		scene.Save(baseFileName+_T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
		scene.pointcloud.Save(baseFileName+_T(".ply"));
		TFinalize();
		return EXIT_SUCCESS;
	}
	if (OPT::nExportNumViews && scene.pointcloud.IsValid()) {
#if 1 // JPB WIP BUG
		throw std::runtime_error("Unsupported");
#else
		// export point-cloud containing only points with N+ views
		const String baseFileName(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName))+
			String::FormatString(_T("_%dviews"), ABS(OPT::nExportNumViews)));
		if (OPT::nExportNumViews > 0) {
			// export point-cloud containing only points with N+ views
			scene.pointcloud.SaveNViews(baseFileName+_T(".ply"), (IIndex)OPT::nExportNumViews);
		} else {
			// save scene and export point-cloud containing only points with N+ views
			scene.pointcloud.RemoveMinViews((IIndex)-OPT::nExportNumViews);
			scene.Save(baseFileName+_T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
			scene.pointcloud.Save(baseFileName+_T(".ply"));
		}
		TFinalize();
		return EXIT_SUCCESS;
#endif
	}
	if ((ARCHIVE_TYPE)OPT::nArchiveType != ARCHIVE_MVS) {
#if 0 // JPB WIP BUG Unsupported
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 1 && !scene.pointcloud.IsEmpty())
			scene.pointcloud.PrintStatistics(scene.images.data(), &scene.obb);
		#endif
#endif

		TD_TIMER_START();
#if 0 // JPB WIP BUG Revisit
		if (OPT::nFusionMode != 0) {
			VERBOSE("error: Negative fusion modes currently unsupported.");
			return EXIT_FAILURE;
		}
#endif
		if (!scene.DenseReconstruction(OPT::nFusionMode, OPT::bCrop2ROI, OPT::fBorderROI)) {
			if (ABS(OPT::nFusionMode) != 1)
				return EXIT_FAILURE;
			VERBOSE("Depth-maps estimated (%s)", TD_TIMER_GET_FMT().c_str());
			TFinalize();
			return EXIT_SUCCESS;
		}
		DENSIFY_DIAG("Densifying point-cloud completed: %u points (%s)", scene.pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// Single deliverable: the multi-view-consensus FUSED cloud (scene.pointcloud),
	// visibility-filtered (floater/plume free-space cull) BEFORE saving so the same
	// clean cloud feeds BOTH the .mvs (mesh stage) and the .ply delivery output.
	const String baseFileName(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName)));

	// Optional inline visibility filter (post-fusion), run FIRST so the filtered
	// cloud is what both outputs contain. Positive --filter-point-cloud runs the
	// camera-point free-space cull. A point's score goes more negative the more real
	// surface it sits IN FRONT of (free-space violation), so floaters / water plumes
	// are strongly negative while ordinary surface is only mildly negative. The value
	// is the removal cutoff MAGNITUDE: a point is removed only if visibility <= -N
	// (LARGER N = gentler, removes only strongly-contradicted floaters; ~20-50 =
	// plume-only; ~1-3 eats real surface). The 0.5f cap makes the filter refuse and
	// leave the cloud untouched if it would remove >50% (threshold too aggressive),
	// so an over-small value can never silently gut the output.
	//   0 = disabled.
	if (OPT::thFilterPointCloud > 0 && !scene.pointcloud.IsEmpty()) {
		const size_t nBefore(scene.pointcloud.GetSize());
		scene.PointCloudFilter(-OPT::thFilterPointCloud, 0.5f);
		DENSIFY_DIAG("Filtered point-cloud (th<=-%d): %u -> %u points",
			OPT::thFilterPointCloud, (unsigned)nBefore, (unsigned)scene.pointcloud.GetSize());
	}

	#if DENSIFY_MORTON_SORT_OUTPUT
	// Keep the Morton reorder AFTER the visibility filter: reordering before it
	// makes spatially-near points index-near, which CONCENTRATES the filter's
	// atomic visibility[] updates onto few cache lines and badly increases
	// cross-thread contention (measured ~2x slower). Order here is output-only.
	ReorderPointCloudMorton(scene.pointcloud);
	#endif

	if (scene.pointcloud.IsEmpty())
		VERBOSE("error: dense point-cloud is EMPTY before save (nothing will be written) -- check fusion / filter threshold");
	if (!scene.Save(baseFileName+_T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType))
		VERBOSE("error: failed to save scene to '%s.mvs'", baseFileName.c_str());
	if (!scene.pointcloud.Save(baseFileName+_T(".ply")))
		VERBOSE("error: failed to save point-cloud to '%s.ply' (%u points)", baseFileName.c_str(), (unsigned)scene.pointcloud.GetSize());
	else
		VERBOSE("Saved dense point-cloud: %s.ply (%u points, filtered fused)", baseFileName.c_str(), (unsigned)scene.pointcloud.GetSize());
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2)
		scene.ExportCamerasMLP(baseFileName+_T(".mlp"), baseFileName+_T(".ply"));
	#endif

	TFinalize();
	return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/
