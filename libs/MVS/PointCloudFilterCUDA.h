/*
* PointCloudFilterCUDA.h
*
* GPU sweep for Scene::PointCloudFilter's camera-point visibility test
* (exploration, 2026-09). See PointCloudFilterCUDA.cu.
*/

#ifndef _MVS_POINTCLOUDFILTERCUDA_H_
#define _MVS_POINTCLOUDFILTERCUDA_H_

// Experimental alternative sweeps for Scene::PointCloudFilter (per-view pixel binning,
// CPU and GPU), selected at run time by OPENMVS_PCF_BINNED only when this is 1. Off by
// default: see the selector in PointCloudFilter. Defined here, not in SceneDensify.cpp,
// because it also decides whether PointCloudFilterCUDA.cu compiles its body -- the
// only CUB user in the project -- so both translation units must see the same value.
#ifndef PCF_EXPERIMENTAL_SWEEPS
#define PCF_EXPERIMENTAL_SWEEPS 0
#endif
// the GPU sweep exists only in a CUDA build with the experimental sweeps enabled
#if defined(_USE_CUDA) && PCF_EXPERIMENTAL_SWEEPS
#define PCF_GPU_SWEEP 1
#else
#define PCF_GPU_SWEEP 0
#endif

#if PCF_GPU_SWEEP

#include <cstdint>
#include <cstddef>
#include <vector>

namespace MVS {
namespace CUDA {

// one view, all in float, as the CPU sweep uses it
struct PCFViewDesc {
	float fx, fy, cx, cy; // intrinsics at image resolution
	float R[9];           // world -> camera rotation, row-major
	float C[3];           // camera center == cone apex (Cast<float>(camera.C), as the CPU)
	float angle;          // cone half-angle, FOV/width
	float cosSq;          // cos(angle)^2, as TConeIntersect
	int width, height;    // image size
};

struct PCFStats {
	double msUpload = 0, msKeys = 0, msSelect = 0, msSort = 0, msBounds = 0, msQuery = 0, msDownload = 0, msTotal = 0;
	size_t nFallback = 0;
	size_t nInView = 0; // sum over views of the points binned (sorted) for that view
	size_t bytesDevice = 0;
};

// Accumulates the visibility scores of every (point, view) query into vis, exactly
// as the CPU sweep would, except for the queries the pixel grid cannot bound; those
// are returned as (point, view) pairs for the caller to run through the octree.
// Returns false on any CUDA failure, in which case vis is untouched and the caller
// must run the CPU sweep instead.
bool PointCloudVisibilityCUDA(
	const float* xyz, size_t numPoints, const uint32_t* viewSizes,
	const PCFViewDesc* views, size_t numViews,
	const uint32_t* queryOffsets /*numViews+1*/, const uint32_t* queryPoints,
	int binSize, float slackPx,
	int* vis,
	std::vector<uint32_t>& fallbackPoints, std::vector<uint32_t>& fallbackViews,
	PCFStats& stats);

} // namespace CUDA
} // namespace MVS

#endif // PCF_GPU_SWEEP

#endif // _MVS_POINTCLOUDFILTERCUDA_H_
