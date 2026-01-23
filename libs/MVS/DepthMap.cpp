/*
* DepthMap.cpp
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
#include "DepthMap.h"
#include "Mesh.h"
#define _USE_OPENCV
#include "Interface.h"
#include "../Common/AutoEstimator.h"
// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
// CGAL: estimate normals
#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/pca_estimate_normals.h>

#include <boost/container/small_vector.hpp>

using namespace MVS;

float firstSpatial;
__declspec( align( 16 ) ) float swSpatials[32];
__declspec( align(16) ) int sImageOffsets[32];
__declspec( align( 16 ) ) ImageRef sRemapImageRef[25];

// D E F I N E S ///////////////////////////////////////////////////

#define DEFVAR_OPTDENSE_string(name, title, desc, ...)  DEFVAR_string(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_bool(name, title, desc, ...)    DEFVAR_bool(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_int32(name, title, desc, ...)   DEFVAR_int32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_uint32(name, title, desc, ...)  DEFVAR_uint32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_flags(name, title, desc, ...)   DEFVAR_flags(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_float(name, title, desc, ...)   DEFVAR_float(OPTDENSE, name, title, desc, __VA_ARGS__)
#define DEFVAR_OPTDENSE_double(name, title, desc, ...)  DEFVAR_double(OPTDENSE, name, title, desc, __VA_ARGS__)

#define MDEFVAR_OPTDENSE_string(name, title, desc, ...) DEFVAR_string(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_bool(name, title, desc, ...)   DEFVAR_bool(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_int32(name, title, desc, ...)  DEFVAR_int32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_uint32(name, title, desc, ...) DEFVAR_uint32(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_flags(name, title, desc, ...)  DEFVAR_flags(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_float(name, title, desc, ...)  DEFVAR_float(OPTDENSE, name, title, desc, __VA_ARGS__)
#define MDEFVAR_OPTDENSE_double(name, title, desc, ...) DEFVAR_double(OPTDENSE, name, title, desc, __VA_ARGS__)

namespace MVS {
DEFOPT_SPACE(OPTDENSE, _T("Dense"))

DEFVAR_OPTDENSE_uint32(nResolutionLevel, "Resolution Level", "How many times to scale down the images before dense reconstruction", "1")
MDEFVAR_OPTDENSE_uint32(nMaxResolution, "Max Resolution", "Do not scale images lower than this resolution", "3200")
MDEFVAR_OPTDENSE_uint32(nMinResolution, "Min Resolution", "Do not scale images lower than this resolution", "640")
DEFVAR_OPTDENSE_uint32(nSubResolutionLevels, "SubResolution levels", "Number of lower resolution levels to estimate the depth and normals", "2")
DEFVAR_OPTDENSE_uint32(nMinViews, "Min Views", "minimum number of agreeing views to validate a depth", "2")
MDEFVAR_OPTDENSE_uint32(nMaxViews, "Max Views", "maximum number of neighbor images used to compute the depth-map for the reference image", "12")
DEFVAR_OPTDENSE_uint32(nMinViewsFuse, "Min Views Fuse", "minimum number of images that agrees with an estimate during fusion in order to consider it inlier (<2 - only merge depth-maps)", "2")
DEFVAR_OPTDENSE_uint32(nMinViewsFilter, "Min Views Filter", "minimum number of images that agrees with an estimate in order to consider it inlier", "2")
MDEFVAR_OPTDENSE_uint32(nMinViewsFilterAdjust, "Min Views Filter Adjust", "minimum number of images that agrees with an estimate in order to consider it inlier (0 - disabled)", "1")
MDEFVAR_OPTDENSE_uint32(nMinViewsTrustPoint, "Min Views Trust Point", "min-number of views so that the point is considered for approximating the depth-maps (<2 - random initialization)", "2")
MDEFVAR_OPTDENSE_uint32(nNumViews, "Num Views", "Number of views used for depth-map estimation (0 - all views available)", "0", "1", "4")
MDEFVAR_OPTDENSE_uint32(nPointInsideROI, "Point Inside ROI", "consider a point shared only if inside ROI when estimating the neighbor views (0 - ignore ROI, 1 - weight more ROI points, 2 - consider only ROI points)", "1")
MDEFVAR_OPTDENSE_bool(bFilterAdjust, "Filter Adjust", "adjust depth estimates during filtering", "1")
MDEFVAR_OPTDENSE_bool(bAddCorners, "Add Corners", "add support points at image corners with nearest neighbor disparities", "0")
MDEFVAR_OPTDENSE_bool(bInitSparse, "Init Sparse", "init depth-map only with the sparse points (no interpolation)", "1")
MDEFVAR_OPTDENSE_bool(bRemoveDmaps, "Remove Dmaps", "remove depth-maps after fusion", "0")
MDEFVAR_OPTDENSE_float(fViewMinScore, "View Min Score", "Min score to consider a neighbor images (0 - disabled)", "2.0")
MDEFVAR_OPTDENSE_float(fViewMinScoreRatio, "View Min Score Ratio", "Min score ratio to consider a neighbor images", "0.03")
MDEFVAR_OPTDENSE_float(fMinArea, "Min Area", "Min shared area for accepting the depth triangulation", "0.05")
MDEFVAR_OPTDENSE_float(fMinAngle, "Min Angle", "Min angle for accepting the depth triangulation", "3.0")
MDEFVAR_OPTDENSE_float(fOptimAngle, "Optim Angle", "Optimal angle for computing the depth triangulation", "12.0")
MDEFVAR_OPTDENSE_float(fMaxAngle, "Max Angle", "Max angle for accepting the depth triangulation", "65.0")
MDEFVAR_OPTDENSE_float(fDescriptorMinMagnitudeThreshold, "Descriptor Min Magnitude Threshold", "minimum patch texture variance accepted when matching two patches (0 - disabled)", "0.02") // 0.02: pixels with patch texture variance below 0.0004 (0.02^2) will be removed from depthmap; 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
MDEFVAR_OPTDENSE_float(fDepthDiffThreshold, "Depth Diff Threshold", "maximum variance allowed for the depths during refinement", "0.01")
MDEFVAR_OPTDENSE_float(fNormalDiffThreshold, "Normal Diff Threshold", "maximum variance allowed for the normal during fusion (degrees)", "25")
MDEFVAR_OPTDENSE_float(fPairwiseMul, "Pairwise Mul", "pairwise cost scale to match the unary cost", "0.3")
MDEFVAR_OPTDENSE_float(fOptimizerEps, "Optimizer Eps", "MRF optimizer stop epsilon", "0.001")
MDEFVAR_OPTDENSE_int32(nOptimizerMaxIters, "Optimizer Max Iters", "MRF optimizer max number of iterations", "80")
MDEFVAR_OPTDENSE_uint32(nSpeckleSize, "Speckle Size", "maximal size of a speckle (small speckles get removed)", "100")
MDEFVAR_OPTDENSE_uint32(nIpolGapSize, "Interpolate Gap Size", "interpolate small gaps (left<->right, top<->bottom)", "7")
MDEFVAR_OPTDENSE_int32(nIgnoreMaskLabel, "Ignore Mask Label", "label id used during ignore mask filter (<0 - disabled)", "-1")
DEFVAR_OPTDENSE_uint32(nOptimize, "Optimize", "should we filter the extracted depth-maps?", "7") // see DepthFlags
MDEFVAR_OPTDENSE_uint32(nEstimateColors, "Estimate Colors", "should we estimate the colors for the dense point-cloud?", "2", "0", "1")
MDEFVAR_OPTDENSE_uint32(nEstimateNormals, "Estimate Normals", "should we estimate the normals for the dense point-cloud?", "0", "1", "2")
MDEFVAR_OPTDENSE_float(fNCCThresholdKeep, "NCC Threshold Keep", "Maximum 1-NCC score accepted for a match", "0.9", "0.5")
DEFVAR_OPTDENSE_uint32(nEstimationIters, "Estimation Iters", "Number of patch-match iterations", "3")
DEFVAR_OPTDENSE_uint32(nEstimationGeometricIters, "Estimation Geometric Iters", "Number of geometric consistent patch-match iterations (0 - disabled)", "2")
MDEFVAR_OPTDENSE_float(fEstimationGeometricWeight, "Estimation Geometric Weight", "pairwise geometric consistency cost weight", "0.1")
MDEFVAR_OPTDENSE_uint32(nRandomIters, "Random Iters", "Number of iterations for random assignment per pixel", "6")
MDEFVAR_OPTDENSE_uint32(nRandomMaxScale, "Random Max Scale", "Maximum number of iterations to skip during random assignment", "2")
MDEFVAR_OPTDENSE_float(fRandomDepthRatio, "Random Depth Ratio", "Depth range ratio of the current estimate for random plane assignment", "0.003", "0.004")
MDEFVAR_OPTDENSE_float(fRandomAngle1Range, "Random Angle1 Range", "Angle 1 range for random plane assignment (degrees)", "16.0", "20.0")
MDEFVAR_OPTDENSE_float(fRandomAngle2Range, "Random Angle2 Range", "Angle 2 range for random plane assignment (degrees)", "10.0", "12.0")
MDEFVAR_OPTDENSE_float(fRandomSmoothDepth, "Random Smooth Depth", "Depth variance used during neighbor smoothness assignment (ratio)", "0.02")
MDEFVAR_OPTDENSE_float(fRandomSmoothNormal, "Random Smooth Normal", "Normal variance used during neighbor smoothness assignment (degrees)", "13")
MDEFVAR_OPTDENSE_float(fRandomSmoothBonus, "Random Smooth Bonus", "Score factor used to encourage smoothness (1 - disabled)", "0.93")
}

// S T R U C T S ///////////////////////////////////////////////////

//constructor from reference of DepthData
DepthData::DepthData(const DepthData& srcDepthData) :
	images(srcDepthData.images),
	neighbors(srcDepthData.neighbors),
	points(srcDepthData.points),
	mask(srcDepthData.mask),
	depthMap(srcDepthData.depthMap),
	normalMap(srcDepthData.normalMap),
	confMap(srcDepthData.confMap),
	dMin(srcDepthData.dMin),
	dMax(srcDepthData.dMax),
	references(srcDepthData.references)
{}

// return normal in world-space for the given pixel
// the 3D points can be precomputed and passed here
void DepthData::GetNormal(const ImageRef& ir, Point3f& N, const TImage<Point3f>* pPointMap) const
{
	ASSERT(!IsEmpty());
	ASSERT(depthMap.pix(ir) > 0);
	const Camera& camera = images.First().camera;
	if (!normalMap.empty()) {
		// set available normal
		N = camera.R.t()*Cast<REAL>(normalMap.pix(ir));
		return;
	}
	// estimate normal based on the neighbor depths
	const int nPointsStep = 2;
	const int nPointsHalf = 2;
	const int nPoints = 2*nPointsHalf+1;
	const int nWindowHalf = nPointsHalf*nPointsStep;
	const int nWindow = 2*nWindowHalf+1;
	const Image8U::Size size(depthMap.size());
	const ImageRef ptCorner(ir.x-nWindowHalf, ir.y-nWindowHalf);
	const ImageRef ptCornerRel(ptCorner.x>=0?0:-ptCorner.x, ptCorner.y>=0?0:-ptCorner.y);
	Point3Arr points(1, nPoints*nPoints);
	if (pPointMap) {
		points[0] = (*pPointMap)(ir);
		for (int j=ptCornerRel.y; j<nWindow; j+=nPointsStep) {
			const int y = ptCorner.y+j;
			if (y >= size.height)
				break;
			for (int i=ptCornerRel.x; i<nWindow; i+=nPointsStep) {
				const int x = ptCorner.x+i;
				if (x >= size.width)
					break;
				if (x==ir.x && y==ir.y)
					continue;
				if (depthMap(y,x) > 0)
					points.Insert((*pPointMap)(y,x));
			}
		}
	} else {
		points[0] = camera.TransformPointI2C(Point3(ir.x,ir.y,depthMap.pix(ir)));
		for (int j=ptCornerRel.y; j<nWindow; j+=nPointsStep) {
			const int y = ptCorner.y+j;
			if (y >= size.height)
				break;
			for (int i=ptCornerRel.x; i<nWindow; i+=nPointsStep) {
				const int x = ptCorner.x+i;
				if (x >= size.width)
					break;
				if (x==ir.x && y==ir.y)
					continue;
				const Depth d = depthMap(y,x);
				if (d > 0)
					points.Insert(camera.TransformPointI2C(Point3(x,y,d)));
			}
		}
	}
	if (points.GetSize() < 3) {
		N = normalized(-points[0]);
		return;
	}
	Plane plane;
	if (EstimatePlaneThLockFirstPoint(points, plane, 0, NULL, 20) < 3) {
		N = normalized(-points[0]);
		return;
	}
	ASSERT(ISEQUAL(plane.m_vN.norm(),REAL(1)));
	// normal is defined up to sign; pick the direction that points to the camera
	if (plane.m_vN.dot((const Point3::EVec)points[0]) > 0)
		plane.Negate();
	N = camera.R.t()*Point3(plane.m_vN);
}
void DepthData::GetNormal(const Point2f& pt, Point3f& N, const TImage<Point3f>* pPointMap) const
{
	const ImageRef ir(ROUND2INT(pt));
	GetNormal(ir, N, pPointMap);
} // GetNormal
/*----------------------------------------------------------------*/


// apply mask to the depth map
void DepthData::ApplyIgnoreMask(const BitMatrix& mask)
{
	ASSERT(IsValid() && !IsEmpty() && mask.size() == depthMap.size());
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			if (mask.isSet(r,c))
				continue;
			// discard depth-map section ignored by mask
			depthMap(r,c) = 0;
			if (!normalMap.empty())
				normalMap(r,c) = Normal::ZERO;
			if (!confMap.empty())
				confMap(r,c) = 0;
		}
	}
} // ApplyIgnoreMask
/*----------------------------------------------------------------*/


bool DepthData::Save(const String& fileName) const
{
#if 1 // Remove extra copy
	ASSERT(IsValid() && !depthMap.empty() && !confMap.empty());
		// serialize out the current state
		IIndexArr IDs(0, images.size());
		for (const ViewData& image: images)
			IDs.push_back(image.GetID());
		const ViewData& image0 = GetView();
		if (!ExportDepthDataRaw(fileName, image0.pImageData->name, IDs, depthMap.size(), image0.camera.K, image0.camera.R, image0.camera.C, dMin, dMax, depthMap, normalMap, confMap, viewsMap))
			return false;

#else
	ASSERT(IsValid() && !depthMap.empty() && !confMap.empty());
	const String fileNameTmp(fileName+".tmp"); {
		// serialize out the current state
		IIndexArr IDs(0, images.size());
		for (const ViewData& image: images)
			IDs.push_back(image.GetID());
		const ViewData& image0 = GetView();
		if (!ExportDepthDataRaw(fileNameTmp, image0.pImageData->name, IDs, depthMap.size(), image0.camera.K, image0.camera.R, image0.camera.C, dMin, dMax, depthMap, normalMap, confMap, viewsMap))
			return false;
	}
	if (!File::renameFile(fileNameTmp, fileName)) {
		DEBUG_EXTRA("error: can not access dmap file '%s'", fileName.c_str());
		File::deleteFile(fileNameTmp);
		return false;
	}
#endif
	return true;
}
bool DepthData::Load(const String& fileName, unsigned flags)
{
	// serialize in the saved state
	String imageFileName;
	IIndexArr IDs;
	cv::Size imageSize;
	Camera camera;
	if (!ImportDepthDataRaw(fileName, imageFileName, IDs, imageSize, camera.K, camera.R, camera.C, dMin, dMax, depthMap, normalMap, confMap, viewsMap, flags))
		return false;
	ASSERT(!IsValid() || (IDs.size() == images.size() && IDs.front() == GetView().GetID()));
	ASSERT(depthMap.size() == imageSize);
	return true;
}
/*----------------------------------------------------------------*/


unsigned DepthData::GetRef()
{
	Lock l(cs);
	return references;
}
unsigned DepthData::IncRef(const String& fileName)
{
	Lock l(cs);
	ASSERT(!IsEmpty() || references==0);
	if (IsEmpty() && !Load(fileName))
		return 0;
	return ++references;
}
unsigned DepthData::DecRef()
{
	Lock l(cs);
	ASSERT(references>0);
	if (--references == 0)
		Release();
	return references;
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

#if 1 // New version
// try to load and apply mask to the depth map;
// the mask for each image is stored in the MVS scene or next to each image with '.mask.png' extension;
// the mask marks as false (or 0) pixels that should be ignored
//  - pMask: optional output mask; if defined, the mask is returned in this image instead of the BitMatrix
bool DepthEstimator::ImportIgnoreMask(const Image& image0, const cv::Size& size, uint8_t nIgnoreMaskLabel, BitMatrix& bmask, Image8U* pMask)
{
	ASSERT(image0.IsValid());
	if (image0.maskName.empty())
		return false;
	Image8U mask;
	if (!mask.Load(image0.maskName)) {
		DEBUG("warning: can not load the segmentation mask '%s'", image0.maskName.c_str());
		return false;
	}
	cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
	if (pMask) {
		*pMask = (mask != nIgnoreMaskLabel);
	} else {
		bmask.create(size);
		bmask.memset(0xFF);
		for (int r=0; r<size.height; ++r) {
			for (int c=0; c<size.width; ++c) {
				if (mask(r,c) == nIgnoreMaskLabel)
					bmask.unset(r,c);
			}
		}
	}
	if (mask.empty())
		mask.release();

	return true;
} // ImportIgnoreMask
#else
// try to load and apply mask to the depth map;
// the mask marks as false pixels that should be ignored
bool DepthEstimator::ImportIgnoreMask(const Image& image0, const Image8U::Size& size, BitMatrix& bmask, uint16_t nIgnoreMaskLabel)
{
	ASSERT(image0.IsValid() && !image0.image.empty());
	if (image0.maskName.empty())
		return false;
	Image16U mask;
	if (!mask.Load(image0.maskName)) {
		DEBUG("warning: can not load the segmentation mask '%s'", image0.maskName.c_str());
		return false;
	}
	cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
	bmask.create(size);
	bmask.memset(0xFF);
	for (int r=0; r<size.height; ++r) {
		for (int c=0; c<size.width; ++c) {
			if (mask(r,c) == nIgnoreMaskLabel)
				bmask.unset(r,c);
		}
	}
	return true;
} // ImportIgnoreMask
#endif

// create the map for converting index to matrix position
//                         1 2 3
//  1 2 4 7 5 3 6 8 9 -->  4 5 6
//                         7 8 9
void DepthEstimator::MapMatrix2ZigzagIdx(const Image8U::Size& size, DepthEstimator::MapRefArr& coords, const BitMatrix& mask, int rawStride)
{
	typedef DepthEstimator::MapRef MapRef;
	const int w = size.width;
	const int w1 = size.width-1;
	coords.Empty();
	coords.Reserve(size.area());
	for (int dy=0, h=rawStride; dy<size.height; dy+=h) {
		if (h*2 > size.height - dy)
			h = size.height - dy;
		int lastX = 0;
		MapRef x(MapRef::ZERO);
		for (int i=0, ei=w*h; i<ei; ++i) {
			const MapRef pt(x.x, x.y+dy);
			if (mask.empty() || mask.isSet(pt))
				coords.Insert(pt);
			if (x.x-- == 0 || ++x.y == h) {
				if (++lastX < w) {
					x.x = lastX;
					x.y = 0;
				} else {
					x.x = w1;
					x.y = lastX - w1;
				}
			}
		}
	}
}

// replace POWI(0.5f, idxScaleRange):           0    1      2       3       4         5         6           7           8             9             10              11
const float DepthEstimator::scaleRanges[12] = {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f};

DepthEstimator::DepthEstimator(
	unsigned nIter,
	DepthData& _depthData0,
#ifndef DPC_EXTENDED_OMP_THREADING
	volatile Thread::safe_t& _idx,
#endif
#if DENSE_NCC == DENSE_NCC_WEIGHTED
	#else
	const Image64F& _image0Sum,
	#endif
	const MapRefArr& _coords)
	:
#ifndef DPC_EXTENDED_OMP_THREADING
	idxPixel(_idx),
#endif
	depthMap0(_depthData0.depthMap), normalMap0(_depthData0.normalMap), confMap0(_depthData0.confMap),
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	#endif
	//nIteration(nIter),
	images(_depthData0.images.begin()+1, _depthData0.images.end()), image0(_depthData0.images[0]),
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	image0Sum(_image0Sum),
	#endif
	coords(_coords), size(_depthData0.images.First().image.size()),
	dMin(_depthData0.dMin), dMax(_depthData0.dMax),
	dMinSqr(SQRT(_depthData0.dMin)), dMaxSqr(SQRT(_depthData0.dMax)),
	dir(nIter%2 ? RB2LT : LT2RB),
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
	idxScore((_depthData0.images.size()-1)/3),
	#elif DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	idxScore(_depthData0.images.size()<=2 ? 0u : 1u),
	#endif
	smoothBonusDepth(1.f-OPTDENSE::fRandomSmoothBonus), smoothBonusNormal((1.f-OPTDENSE::fRandomSmoothBonus)*0.96f),
	smoothSigmaDepth(-1.f/(2.f*SQUARE(OPTDENSE::fRandomSmoothDepth))), // used in exp(-x^2 / (2*(0.02^2)))
	smoothSigmaNormal(-1.f/(2.f*SQUARE(FD2R(OPTDENSE::fRandomSmoothNormal)))), // used in exp(-x^2 / (2*(0.22^2)))
	thMagnitudeSq(OPTDENSE::fDescriptorMinMagnitudeThreshold>0?SQUARE(OPTDENSE::fDescriptorMinMagnitudeThreshold):-1.f),
	angle1Range(FD2R(OPTDENSE::fRandomAngle1Range)), //default 0.279252678=FD2R(20)
	angle2Range(FD2R(OPTDENSE::fRandomAngle2Range)), //default 0.174532920=FD2R(16)
	thConfSmall(OPTDENSE::fNCCThresholdKeep * 0.66f), // default 0.6
	thConfBig(OPTDENSE::fNCCThresholdKeep * 0.9f), // default 0.8
	thConfRand(OPTDENSE::fNCCThresholdKeep * 1.1f), // default 0.99
	thRobust(OPTDENSE::fNCCThresholdKeep * 4.f / 3.f) // default 1.2
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	, thPerturbation(1.f/POW(2.f,float(nIter+1)))
	#endif
	, sh(1.f, images) // Order dependency on images
{
	ASSERT(_depthData0.images.size() >= 1);
	scoreResults.Allocate((int) _depthData0.images.size()-1);
	if (
		(4 != GROUP_SIZE)
		|| (DENSE_NCC != DENSE_NCC_WEIGHTED)) {
		throw std::runtime_error( "Unsupported" );
	}

#ifdef DPC_FASTER_SAMPLING
	constexpr float sigmaSpatial(-1.f/( 2.f*SQUARE((int)nSizeHalfWindow-1) ));

	// ul
	sRemapImageRef[0] = ImageRef(-nSizeHalfWindow, -nSizeHalfWindow);

	// left edge
	int n = 1;
	for (int i=-nSizeHalfWindow + nSizeStep; i<=nSizeHalfWindow; i+=nSizeStep) {
		sRemapImageRef[n++] = ImageRef(-nSizeHalfWindow, i);
	}

	// top row, one offset, downward.
	for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep) {
		for (int j=-nSizeHalfWindow + nSizeStep; j<=nSizeHalfWindow; j+=nSizeStep) {
			sRemapImageRef[n++] = ImageRef(j, i);
		}
	}

	firstSpatial = ( float(SQUARE(sRemapImageRef[0].x) + SQUARE(sRemapImageRef[0].y)) * sigmaSpatial );
	for (auto i = 1; i < nTexels; ++i) {
		const ImageRef& x = sRemapImageRef[i];
		// spatial weight [0..1]
		swSpatials[i-1] = ( float(SQUARE(x.x) + SQUARE(x.y)) * sigmaSpatial );
	}

	for (int i = 0; i < nTexels-1; ++i) {
		sImageOffsets[i] = (sRemapImageRef[i+1].y * ((int)image0.image.step.p[0]/sizeof(float) /* stride */) + sRemapImageRef[i+1].x);
	}
#endif
}

#ifdef _MSC_VER /* visual c++ */
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END 
#else /* gcc or icc */
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
#endif

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, (float) 0.707106781186547524);
_PS_CONST(cephes_log_p0, (float) 7.0376836292E-2);
_PS_CONST(cephes_log_p1, (float) -1.1514610310E-1);
_PS_CONST(cephes_log_p2, (float) 1.1676998740E-1);
_PS_CONST(cephes_log_p3, (float) -1.2420140846E-1);
_PS_CONST(cephes_log_p4, (float) +1.4249322787E-1);
_PS_CONST(cephes_log_p5, (float) -1.6668057665E-1);
_PS_CONST(cephes_log_p6, (float) +2.0000714765E-1);
_PS_CONST(cephes_log_p7, (float) -2.4999993993E-1);
_PS_CONST(cephes_log_p8, (float) +3.3333331174E-1);
_PS_CONST(cephes_log_q1, (float) -2.12194440e-4);
_PS_CONST(cephes_log_q2, (float) 0.693359375);

#ifndef USE_SSE2
typedef union xmm_mm_union {
	__m128 xmm;
	__m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) {          \
    xmm_mm_union u; u.xmm = xmm_;                   \
    mm0_ = u.mm[0];                                 \
    mm1_ = u.mm[1];                                 \
}

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) {                         \
    xmm_mm_union u; u.mm[0]=mm0_; u.mm[1]=mm1_; xmm_ = u.xmm;      \
  }

#endif // USE_SSE2

/* natural logarithm computed for 4 simultaneous float
	 return NaN for x <= 0
*/
v4sf log_ps(v4sf x) {
	v4si emm0;
	v4sf one = *(v4sf*)_ps_1;

	v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

	x = _mm_max_ps(x, *(v4sf*)_ps_min_norm_pos);  /* cut off denormalized stuff */

	emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
	/* keep only the fractional part */
	x = _mm_and_ps(x, *(v4sf*)_ps_inv_mant_mask);
	x = _mm_or_ps(x, *(v4sf*)_ps_0p5);

	emm0 = _mm_sub_epi32(emm0, *(v4si*)_pi32_0x7f);
	v4sf e = _mm_cvtepi32_ps(emm0);

	e = _mm_add_ps(e, one);

	/* part2:
		 if( x < SQRTHF ) {
			 e -= 1;
			 x = x + x - 1.0;
		 } else { x = x - 1.0; }
	*/
	v4sf mask = _mm_cmplt_ps(x, *(v4sf*)_ps_cephes_SQRTHF);
	v4sf tmp = _mm_and_ps(x, mask);
	x = _mm_sub_ps(x, one);
	e = _mm_sub_ps(e, _mm_and_ps(one, mask));
	x = _mm_add_ps(x, tmp);


	v4sf z = _mm_mul_ps(x, x);

	v4sf y = *(v4sf*)_ps_cephes_log_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p5);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p6);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p7);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p8);
	y = _mm_mul_ps(y, x);

	y = _mm_mul_ps(y, z);


	tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q1);
	y = _mm_add_ps(y, tmp);


	tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
	y = _mm_sub_ps(y, tmp);

	tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q2);
	x = _mm_add_ps(x, y);
	x = _mm_add_ps(x, tmp);
	x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
	return x;
}

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, (float) 1.44269504088896341);
_PS_CONST(cephes_exp_C1, (float) 0.693359375);
_PS_CONST(cephes_exp_C2, (float) -2.12194440e-4);

_PS_CONST(cephes_exp_p0, (float) 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, (float) 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, (float) 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, (float) 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, (float) 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, (float) 5.0000001201E-1);

v4sf exp_ps(v4sf x) {
	v4sf tmp = _mm_setzero_ps(), fx;
	v4si emm0;
	v4sf one = *(v4sf*)_ps_1;

	x = _mm_min_ps(x, *(v4sf*)_ps_exp_hi);
	x = _mm_max_ps(x, *(v4sf*)_ps_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm_mul_ps(x, *(v4sf*)_ps_cephes_LOG2EF);
	fx = _mm_add_ps(fx, *(v4sf*)_ps_0p5);

	/* how to perform a floorf with SSE: just below */
	emm0 = _mm_cvttps_epi32(fx);
	tmp  = _mm_cvtepi32_ps(emm0);
	/* if greater, subtract 1 */
	v4sf mask = _mm_cmpgt_ps(tmp, fx);
	mask = _mm_and_ps(mask, one);
	fx = _mm_sub_ps(tmp, mask);

	tmp = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C1);
	v4sf z = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C2);
	x = _mm_sub_ps(x, tmp);
	x = _mm_sub_ps(x, z);

	z = _mm_mul_ps(x, x);

	v4sf y = *(v4sf*)_ps_cephes_exp_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p5);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, x);
	y = _mm_add_ps(y, one);

	/* build 2^n */
	emm0 = _mm_cvttps_epi32(fx);
	emm0 = _mm_add_epi32(emm0, *(v4si*)_pi32_0x7f);
	emm0 = _mm_slli_epi32(emm0, 23);
	v4sf pow2n = _mm_castsi128_ps(emm0);
	y = _mm_mul_ps(y, pow2n);
	return y;
}

_PS_CONST(minus_cephes_DP1, (float) -0.78515625);
_PS_CONST(minus_cephes_DP2, (float)-2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, (float)-3.77489497744594108e-8);
_PS_CONST(sincof_p0, (float)-1.9515295891E-4);
_PS_CONST(sincof_p1, (float)8.3321608736E-3);
_PS_CONST(sincof_p2,(float) -1.6666654611E-1);
_PS_CONST(coscof_p0, (float)2.443315711809948E-005);
_PS_CONST(coscof_p1, (float)-1.388731625493765E-003);
_PS_CONST(coscof_p2, (float)4.166664568298827E-002);
_PS_CONST(cephes_FOPI, (float)1.27323954473516); // 4 / M_PI


/* evaluation of 4 sines at onces, using only SSE1+MMX intrinsics so
	 it runs also on old athlons XPs and the pentium III of your grand
	 mother.

	 The code is the exact rewriting of the cephes sinf function.
	 Precision is excellent as long as x < 8192 (I did not bother to
	 take into account the special handling they have for greater values
	 -- it does not return garbage for arguments over 8192, though, but
	 the extra precision is missing).

	 Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
	 surprising but correct result.

	 Performance is also surprisingly good, 1.33 times faster than the
	 macos vsinf SSE2 function, and 1.5 times faster than the
	 __vrs4_sinf of amd's ACML (which is only available in 64 bits). Not
	 too bad for an SSE1 function (with no special tuning) !
	 However the latter libraries probably have a much better handling of NaN,
	 Inf, denormalized and other special arguments..

	 On my core 1 duo, the execution of this function takes approximately 95 cycles.

	 From what I have observed on the experiments with Intel AMath lib, switching to an
	 SSE2 version would improve the perf by only 10%.

	 Since it is based on SSE intrinsics, it has to be compiled at -O2 to
	 deliver full speed.
*/
v4sf sin_ps(v4sf x) { // any x
	v4sf xmm1, xmm2 = _mm_setzero_ps(), xmm3, sign_bit, y;

	v4si emm0, emm2;
	sign_bit = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(v4sf*)_ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit = _mm_and_ps(sign_bit, *(v4sf*)_ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(v4sf*)_ps_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = _mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(v4si*)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(v4si*)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	/* get the swap sign flag */
	emm0 = _mm_and_si128(emm2, *(v4si*)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	/* get the polynom selection mask
		 there is one polynom for 0 <= x <= Pi/4
		 and another one for Pi/4<x<=Pi/2

		 Both branches will be computed.
	*/
	emm2 = _mm_and_si128(emm2, *(v4si*)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

	v4sf swap_sign_bit = _mm_castsi128_ps(emm0);
	v4sf poly_mask = _mm_castsi128_ps(emm2);
	sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);

	/* The magic pass: "Extended precision modular arithmetic"
		 x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*)_ps_minus_cephes_DP1;
	xmm2 = *(v4sf*)_ps_minus_cephes_DP2;
	xmm3 = *(v4sf*)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v4sf*)_ps_coscof_p0;
	v4sf z = _mm_mul_ps(x, x);

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf*)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf*)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf*)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf*)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf*)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf*)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y2 = _mm_and_ps(xmm3, y2); //, xmm3);
	y = _mm_andnot_ps(xmm3, y);
	y = _mm_add_ps(y, y2);
	/* update the sign */
	y = _mm_xor_ps(y, sign_bit);
	return y;
}

/* almost the same as sin_ps */
v4sf cos_ps(v4sf x) { // any x
	v4sf xmm1, xmm2 = _mm_setzero_ps(), xmm3, y;
	v4si emm0, emm2;
	/* take the absolute value */
	x = _mm_and_ps(x, *(v4sf*)_ps_inv_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(v4sf*)_ps_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = _mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(v4si*)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(v4si*)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm2 = _mm_sub_epi32(emm2, *(v4si*)_pi32_2);

	/* get the swap sign flag */
	emm0 = _mm_andnot_si128(emm2, *(v4si*)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	/* get the polynom selection mask */
	emm2 = _mm_and_si128(emm2, *(v4si*)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

	v4sf sign_bit = _mm_castsi128_ps(emm0);
	v4sf poly_mask = _mm_castsi128_ps(emm2);
	/* The magic pass: "Extended precision modular arithmetic"
		 x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*)_ps_minus_cephes_DP1;
	xmm2 = *(v4sf*)_ps_minus_cephes_DP2;
	xmm3 = *(v4sf*)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v4sf*)_ps_coscof_p0;
	v4sf z = _mm_mul_ps(x, x);

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf*)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf*)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf*)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf*)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf*)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf*)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y2 = _mm_and_ps(xmm3, y2); //, xmm3);
	y = _mm_andnot_ps(xmm3, y);
	y = _mm_add_ps(y, y2);
	/* update the sign */
	y = _mm_xor_ps(y, sign_bit);

	return y;
}

/* since sin_ps and cos_ps are almost identical, sincos_ps could replace both of them..
	 it is almost as fast, and gives you a free cosine with your sine */
void sincos_ps(v4sf x, v4sf* s, v4sf* c) {
	v4sf xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
	v4si emm0, emm2, emm4;
	sign_bit_sin = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(v4sf*)_ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit_sin = _mm_and_ps(sign_bit_sin, *(v4sf*)_ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(v4sf*)_ps_cephes_FOPI);

	/* store the integer part of y in emm2 */
	emm2 = _mm_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(v4si*)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(v4si*)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm4 = emm2;

	/* get the swap sign flag for the sine */
	emm0 = _mm_and_si128(emm2, *(v4si*)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	v4sf swap_sign_bit_sin = _mm_castsi128_ps(emm0);

	/* get the polynom selection mask for the sine*/
	emm2 = _mm_and_si128(emm2, *(v4si*)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
	v4sf poly_mask = _mm_castsi128_ps(emm2);

	/* The magic pass: "Extended precision modular arithmetic"
		 x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf*)_ps_minus_cephes_DP1;
	xmm2 = *(v4sf*)_ps_minus_cephes_DP2;
	xmm3 = *(v4sf*)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	emm4 = _mm_sub_epi32(emm4, *(v4si*)_pi32_2);
	emm4 = _mm_andnot_si128(emm4, *(v4si*)_pi32_4);
	emm4 = _mm_slli_epi32(emm4, 29);
	v4sf sign_bit_cos = _mm_castsi128_ps(emm4);

	sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);


	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	v4sf z = _mm_mul_ps(x, x);
	y = *(v4sf*)_ps_coscof_p0;

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf*)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf*)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf*)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf*)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf*)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf*)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	v4sf ysin2 = _mm_and_ps(xmm3, y2);
	v4sf ysin1 = _mm_andnot_ps(xmm3, y);
	y2 = _mm_sub_ps(y2, ysin2);
	y = _mm_sub_ps(y, ysin1);

	xmm1 = _mm_add_ps(ysin1, ysin2);
	xmm2 = _mm_add_ps(y, y2);

	/* update the sign */
	*s = _mm_xor_ps(xmm1, sign_bit_sin);
	*c = _mm_xor_ps(xmm2, sign_bit_cos);
}

#ifdef DPC_FASTER_SAMPLING

constexpr _DataI vOneThree{ -1,-1,-1,-1, 0x00,0x00,0x00,0x00, -1,-1,-1,-1, 0x00,0x00,0x00,0x00 };

bool __declspec(safebuffers) DepthEstimator::IsScorable3(
	const ImageInfo_t& imageInfo
) noexcept
{
	// 	const Calc_t a = image1.Hl(0,0) + image1.Hm(0) * _AsArrayD(sh.mMat0, 0);
	// const Calc_t j = image1.Hr(0, 0);
	const auto& origImage = imageInfo.image;

	const _Data mat0 = _Set(_AsArray(sh.mMatXYZ, 0));
	const _Data mat1 = _Set(_AsArray(sh.mMatXYZ, 1));
	const _Data mat2 = _Set(_AsArray(sh.mMatXYZ, 2));

	_Data adg = origImage.mHm0Hm1Hm2;
	_Data beh = origImage.mHm0Hm1Hm2;
	_Data cfi = origImage.mHm0Hm1Hm2;

	adg = _Mul(adg, mat0);
	beh = _Mul(beh, mat1);
	cfi = _Mul(cfi, mat2);

	adg = _Add(adg, origImage.mHl00Hl10Hl20);
	beh = _Add(beh, origImage.mHl01Hl11Hl21);
	cfi = _Add(cfi, origImage.mHl02Hl12Hl22);

	const _Data adg_Hr02Hr02Hr02 = _Mul(adg, origImage.mHr02Hr02Hr02);
	const _Data beh_Hr12Hr12Hr12 = _Mul(beh, origImage.mHr12Hr12Hr12);

	const _Data h00h01h02 = _Mul(adg, origImage.mHr00Hr00Hr00);
	const _Data h10h11h12 = _Mul(beh, origImage.mHr11Hr11Hr11);
	const _Data h20h21h22 = _Add(cfi, _Add(adg_Hr02Hr02Hr02, beh_Hr12Hr12Hr12));

	// vXYZ's final component guaranteed 0
	_Data vXYZ = _Mul(h00h01h02, x0ULPatchCorner0);
	vXYZ = _Add(vXYZ, _Mul(h10h11h12, x0ULPatchCorner1));
	vXYZ = _Add(vXYZ, h20h21h22);

	// Prepare H for rasterizing.
	// Determine H's horizontal and vertical basis vectors.
	constexpr _Data vSizeStep ={ (float)nSizeStep, (float)nSizeStep, (float)nSizeStep, (float)nSizeStep };
	const _Data vBasisH = _Mul(h00h01h02, vSizeStep);
	const _Data vBasisV = _Mul(h10h11h12, vSizeStep);

	// To validate IsScorable(imageInfo.image); // JPB WIP BUG

	sh.mVX = vXYZ;
	const _Data vImageWidthWithBorder = _Set((float)imageInfo.widthMinus2);
	const _Data vImageHeightWithBorder = _Set((float)imageInfo.heightMinus2);

	// Epsilon is needed as a+a+a+a isn't necessarily equal to 4*a
	constexpr _Data vFourEpsilon ={ 4.f + FLT_EPSILON, 4.f + FLT_EPSILON, 4.f + FLT_EPSILON, 0.f };
	const _Data vBasisH4 = _Mul(vBasisH, vFourEpsilon);
	const _Data vBasisV4 = _Mul(vBasisV, vFourEpsilon);

	_Data vUR = _Add(sh.mVX, vBasisH4);
	_Data vLL = _Add(sh.mVX, vBasisV4);
	_Data vLR = _Add(vUR, vBasisV4);

	// Use transpose to separate the x, y, and z components.
	// The last register must be 0, so that the corner "w"
	// components become zero.  This is necessary to assist
	// the compares below.
	_MM_TRANSPOSE4_PS(vXYZ, vUR, vLL, vLR);

	const _Data vCornersX = vXYZ;
	const _Data vCornersY = vUR;
	const _Data vCornersZ = vLL;
	const _Data vCornersWidthZ = _Mul(vCornersZ, vImageWidthWithBorder);
	const _Data vCornersHeightZ = _Mul(vCornersZ, vImageHeightWithBorder);

	// Cmp leaves 1's if true, 0 otherwise.
	const _Data vLtX = _CmpLT(vCornersX, vCornersZ);
	const _Data vLtY = _CmpLT(vCornersY, vCornersZ);
	// Although technically this should be GE, we lose the ability
	// to ignore the final component if we do.
	const _Data vGtX = _CmpGT(vCornersX, vCornersWidthZ);
	const _Data vGtY = _CmpGT(vCornersY, vCornersHeightZ);

	const _Data vLtX_vLtY = _Or(vLtX, vLtY);
	const _Data vGtX_vGtY = _Or(vGtX, vGtY);
	// Reject the patch if any of these corners are outside the view boundary.
	// JPB WIP BUG This does not identify patches which completely cover the view boundary.
	const _Data vCmpResult = _Or(vLtX_vLtY, vGtX_vGtY);

#ifdef USE_FAST_COMPARES
	bool scorable = _mm_movemask_ps(vCmpResult) == 0; //AllZerosI(_CastIF(vCmpResult));
#else
	bool scorable = AllZerosI(_CastIF(vCmpResult));
#endif
	if (scorable) {
		const _Data vVXs = _Splat(sh.mVX, 0);
		const _Data vVYs = _Splat(sh.mVX, 1);
		const _Data vVZs = _Splat(sh.mVX, 2);

		const _Data vProj = _Div(sh.mVX, vVZs);
		const _Data vVBasisHX = _Splat(vBasisH, 0);
		const _Data vVBasisHY = _Splat(vBasisH, 1);
		const _Data vVBasisHZ = _Splat(vBasisH, 2);

		const _DataI vProjI = _ConvertIF(vProj);
		constexpr _Data vDeltaFactor { 1.f, 2.f, 3.f, 4.f };
		const _Data vVBasisHX4 = _Mul(vVBasisHX, vDeltaFactor);
		const _Data vVBasisHY4 = _Mul(vVBasisHY, vDeltaFactor);
		const _Data vVBasisHZ4 = _Mul(vVBasisHZ, vDeltaFactor);

		const _Data vProjF = _ConvertFI(vProjI);
		sh.mVTopRowX4 = _Add(vVXs, vVBasisHX4);
		sh.mVTopRowY4 = _Add(vVYs, vVBasisHY4);
		sh.mVTopRowZ4 = _Add(vVZs, vVBasisHZ4);

		const _Data vFracXY = _Sub(vProj, vProjF); // xf yf xx xx
		sh.mvBasisVX = _Splat(vBasisV, 0);
		sh.mvBasisVY = _Splat(vBasisV, 1);
		sh.mvBasisVZ = _Splat(vBasisV, 2);
		const _Data vFracYX = _mm_shuffle_ps(vFracXY, vFracXY, _MM_SHUFFLE(0, 0, 0, 1)); // yf xf xx xx
		const _Data vFracXY_YX = _Mul(vFracXY, vFracYX);

		const uchar*  pSamples =
			imageInfo.data2 + _AsArrayI(vProjI, 1) * imageInfo.rowByteStride
			+ _AsArrayI(vProjI, 0) * 16;
		sh.mvBasisVX4 = _Mul(sh.mvBasisVX, vDeltaFactor);
		sh.mvBasisVY4 = _Mul(sh.mvBasisVY, vDeltaFactor);
		sh.mvBasisVZ4 = _Mul(sh.mvBasisVZ, vDeltaFactor);

		sh.mAddr = pSamples;
		constexpr _Data vOne { 1.f, 1.f, 1.f, 1.f };

		_Data vInterleaved = _UnpackLow(vOne, vFracXY);
		_Data vInterleaved2 = _UnpackLow(vFracXY, vFracXY_YX);
		//	(1.f, xFrac, yFrac, xFrac* yFrac);
		sh.mFirst = _mm_shuffle_ps(vInterleaved, vInterleaved2, _MM_SHUFFLE(3, 2, 1, 0));

		sh.mVLeftColX4 = _Add(vVXs, sh.mvBasisVX4);
		sh.mVLeftColY4 = _Add(vVYs, sh.mvBasisVY4);
		sh.mVLeftColZ4 = _Add(vVZs, sh.mvBasisVZ4);
	}

	return scorable;
}

enum eSPIResult { eNoDepthMap = 1, eMissedDepthMap = 2, eHitDepthMap = 3 };

void __declspec(safebuffers) DepthEstimator::GatherSampleInfo(
	const DepthEstimator::ImageInfo_t& imageInfo,
	float& sumResult,
	float& sumSqResult,
	float& numResult
) const noexcept
{
	// Center a patch of given size on the segment and fetch the pixel values in the target image
	// A variety of cases are culled for now.
	// Hard-coded for nTexels == 25

	//
	// 0  5  6  7  8
	// 1  9 10 11 12
	// 2 13 14 15 16
	// 3 17 18 19 20
	// 4 21 22 23 24
	//

	// A variety of strategies have been attempted to improve the performance of this work.
	// These have not shown a significant benefit:
	// 1) Fetching samples by streaming (avoiding the cache), e.g., vSampleUL = _CastFI( _mm_stream_load_si128((_DataI*) pSampleUL));
	// 2) Interpolating 1/z as opposed to simply dividing per-pixel.
	// 3) Performing the address arithmetic (pSampleUL calculation) in floating-point arithmetic (a variety of different ways
	//    are possible).  Even if successful this may have trouble with large images.
	// 4) Performing the address arithmetic in SIMD integer calculations is slightly slower even though it clearly
	//    does less work.
	// 5) Transposing (_MM_TRANSPOSE4_PS) the fractional coordinates instead of the samples.
	// 6) Unfolding the primary loop of 5 sets of 4 rows.
	// 7) Combining the left-column and primary loop into a single loop.
	// 8) Processing the 5 sets of 4 rows in "twos" in order to potentially reduce instruction latency.

	// These strategies have shown benefit, but introduce problems:
	// 1) Using a lower-precision divide (__mm_rcp_ps).  Nearly works, but very rarely generates an out-of-bounds coordinate due
	//    to precision problems.  This will crash the application.
	// 2) Using AVX/AVX2 to process x and y simultaneously in the primary loop.  Benefit is modest on first attempt (<5%), but
	//    currently requires runtime identification and other support.

	// What worked:
	// 1) Using SSE2.
	// 2) Reordering the weights and tempWeights to retrieve them in scanning order.
	// 3) Pre-computing the address arithmetic for x (vPtxAsIntx16).
	// 4) Using 4x the memory per image to store the samples in a form more friendly to the bilinear-sampling algorithm processing.
	// 5) Using _mm_cvttps_epi32 to truncate towards zero on values that are guaranteed here to be positive.
	// 6) Working on sets of 4 sums, nums, and sums2 and horizontally adding these at the end.
	// 7) Removing all unpredictable branching.
	// 8) Returning the results passed in parameters is faster than alternatives -say as a direct return value (struct).
	// 9) Oddly, moving the divides in the 5x4 sampling loop to the end of the loop speeds up processing several percent.
	//    By doing this, it actually issues one extra pair of unused divides and is still faster.

	const size_t rowByteStride = imageInfo.rowByteStride;

	// Handle the upper-left sample "0"
#ifdef USE_NN
		const float sample = *(float*)pSamples;
		const float weight = w.pixelWeights[0];
		const float tempWeight = w.pixelTempWeights[0];
		const float weightedSample = sample*weight;
		const float tempWeightedSample = sample*tempWeight;

		vSum4 = _And(_SetFirstUnsafe(weightedSample), _CastFI(vMask1));
		vSumSq4 = _And(_SetFirstUnsafe(weightedSample*sample), _CastFI(vMask1));
		vNum4 = _And(_SetFirstUnsafe(tempWeightedSample), _CastFI(vMask1));
#else
	const Weight& w = pWeightMap;

	const uchar* pSamples = sh.mAddr;
	_Data sample = _LoadA((float*) pSamples);

	_Data vPartsFirstSample =_Mul(sample, sh.mFirst);
		// However, vSumSq4 needs the sum of the samples to proceed, e.g.
		// We have: s0 s1 s2 s3
		// and we need (s0 + s1 + s2 + s3)*(s0 + s1 + s2 + s3)*pw[0]
		const float sum = FastHsumS(vPartsFirstSample);
	_Data vSum4 = _Mul(vPartsFirstSample, _Set(w.firstPixelWeight));
	_Data vNum4 = _Mul(vPartsFirstSample, _Set(w.firstPixelTempWeight));
		const float sumSq = sum*sum;

	// The "4" variants will be used to maintain sums/nums for
	// four samples at a time.
	_Data vSumSq4 = _mm_set_ss(sumSq*w.firstPixelWeight);
#endif

	// Handle the 4 left-column samples "1", "2", "3" and "4".

	_Data vPtx = _Div(sh.mVLeftColX4, sh.mVLeftColZ4);
	_Data vPty = _Div(sh.mVLeftColY4, sh.mVLeftColZ4);

	_DataI vPtxAsInt = _TruncateIF(vPtx);
	_DataI vPtyAsInt = _TruncateIF(vPty);
	_DataI vPtxAsIntx16 = _ShiftLI(vPtxAsInt, 4);

	const uchar* pSampleUL =
		imageInfo.data2 + _AsArrayI(vPtyAsInt, 0) * rowByteStride
		+ _AsArrayI(vPtxAsIntx16, 0);
	const uchar* pSampleUR =
		imageInfo.data2 + _AsArrayI(vPtyAsInt, 1) * rowByteStride
		+ _AsArrayI(vPtxAsIntx16, 1);
	const uchar* pSampleLL =
		imageInfo.data2 + _AsArrayI(vPtyAsInt, 2) * rowByteStride
		+ _AsArrayI(vPtxAsIntx16, 2);
	const uchar* pSampleLR =
		imageInfo.data2 + _AsArrayI(vPtyAsInt, 3) * rowByteStride
		+ _AsArrayI(vPtxAsIntx16, 3);

	_Data vSampleUL = _LoadA((float*) pSampleUL);
	_Data vSampleUR = _LoadA((float*) pSampleUR);
	_Data vSampleLL = _LoadA((float*) pSampleLL);
	_Data vSampleLR = _LoadA((float*) pSampleLR);

	_Data vFracX1X2X3X4 = _Sub(vPtx, _ConvertFI(vPtxAsInt)); // pt(i).x - (int) pt(i).x
	_Data vFracY1Y2Y3Y4 = _Sub(vPty, _ConvertFI(vPtyAsInt)); // pt(i).y - (int) pt(i).y

	_MM_TRANSPOSE4_PS(vSampleUL, vSampleUR, vSampleLL, vSampleLR); // 0.7

	_Data vFracXY = _Mul(vFracX1X2X3X4, vFracY1Y2Y3Y4);

	_Data va00 = vSampleUL;;
	_Data va10 = _Mul(vSampleUR, vFracX1X2X3X4);
	_Data va01 = _Mul(vSampleLL, vFracY1Y2Y3Y4);
	_Data va11 = _Mul(vSampleLR, vFracXY);

	_Data va0010 = _Add(va00, va10);
	_Data va0111 = _Add(va01, va11);

	sample = _Add(va0010, va0111);

	_Data sampleSq = _Mul(sample, sample);

	_Data weights =  _LoadA(&w.pixelWeights[0]);
	_Data tempWeights = _LoadA(&w.pixelTempWeights[0]);

	_Data weightedSample, weightedSampleSq, weightedNum;

	weightedSample = _Mul(sample, weights);
	weightedSampleSq = _Mul(sampleSq, weights);
	weightedNum = _Mul(sample, tempWeights);

	vSum4 = _Add(vSum4, weightedSample);
	vSumSq4 = _Add(vSumSq4, weightedSampleSq);
	vNum4 = _Add(vNum4, weightedNum);

	_Data vX1X2X3X4 = sh.mVTopRowX4;
	_Data vY1Y2Y3Y4 = sh.mVTopRowY4;
	_Data vZ1Z2Z3Z4 = sh.mVTopRowZ4;

	vPtx = _Div(vX1X2X3X4, vZ1Z2Z3Z4);
	vPty = _Div(vY1Y2Y3Y4, vZ1Z2Z3Z4);

	const _Data vBasisVX = sh.mvBasisVX;
	const _Data vBasisVY = sh.mvBasisVY;
	const _Data vBasisVZ = sh.mvBasisVZ;

	const _DataI vRowByteStride = _SetI((int) rowByteStride);
	const _DataI vImageBasePtr = _mm_set1_epi64x((__int64) imageInfo.data2);
	// Handle the 5 rows of 4 samples adjacent (everything else of the 5x5 block).
	for (int i = 0; i < 5; ++i) {
		// Get the x,y projection of each of the four samples.

		// Prepare whole and fraction parts of each projection coordinate.
		vPtxAsInt = _TruncateIF(vPtx);
		vPtyAsInt = _TruncateIF(vPty);

		// Prepare the image offset for each sample.
		// We are referencing 16 byte wide samples.
		const _Data vPtxAsIntFloat = _ConvertFI(vPtxAsInt);
		vPtxAsIntx16 = _ShiftLI(vPtxAsInt, 4);
		const _Data vPtyAsIntFloat = _ConvertFI(vPtyAsInt);

#if 0
#if 1
	const _DataI vPtyAsInt13 = _mm_srli_si128(vPtyAsInt, 4);
	// Warning _mm_and_epi32 is an AVX instruction.
	const _DataI vPtxAsIntx1602 = _AndI(vPtxAsIntx16, vOneThree);
	const _DataI vPtxAsIntx1613 =  _mm_srli_epi64(vPtxAsIntx16, 32);
	const _DataI tmp1 = _mm_mul_epu32(vPtyAsInt,vRowByteStride); /* mul 2,0*/
	const _DataI tmp2 = _mm_mul_epu32( vPtyAsInt13, vRowByteStride); /* mul 3,1 */
	const _DataI tmp3 = _mm_add_epi64(tmp1, vPtxAsIntx1602);
	const _DataI tmp4 = _mm_add_epi64(tmp2, vPtxAsIntx1613); // _AndI(_mm_srli_si128(vPtxAsIntx16, 4), vOneThree));
	const _DataI tmp5 = _mm_add_epi64(tmp3, vImageBasePtr);
	const _DataI tmp6 = _mm_add_epi64(tmp4, vImageBasePtr);

	// Fetch the components for each sample (remember 4 floats each).
	vSampleUL = _LoadA((float*) tmp5.m128i_u64[0]);
	vSampleUR = _LoadA((float*) tmp5.m128i_u64[1]);
	vSampleLL = _LoadA((float*) tmp6.m128i_u64[0]);
	vSampleLR = _LoadA((float*) tmp6.m128i_u64[1]);
#else
  _DataI tmp1           = _mm_mul_epu32(vPtyAsInt,vRowByteStride); /* mul 2,0*/
	// tmp1[0,1] has vPtyAsInt[0] * vRowByteStride, tmp[2,3] has vPtyAsInt[2] * vRowByteStride;
  _DataI tmp2           = _mm_mul_epu32( _mm_srli_si128(vPtyAsInt,4), vRowByteStride); /* mul 3,1 */
	// tmp2[0,1] has vPtyAsInt[1] * vRowByteStride, tmp[2,3] has vPtyAsInt[3] * vRowByteStride;
	tmp1 = _mm_add_epi64(tmp1, _mm_and_epi32(vPtxAsIntx16, vOneThree));
	// tmp1[0,1] has vPtyAsInt[0] * vRowByteStride + vPtxAsIntx16[0], tmp[2,3] has vPtyAsInt[2] * vRowByteStride + vPtxAsIntx16[2]
	tmp2 = _mm_add_epi64(tmp2, _mm_srli_epi64(vPtxAsIntx16, 32)); // _mm_and_epi32(_mm_srli_si128(vPtxAsIntx16, 4), vOneThree));
	// tmp2[0,1] has vPtyAsInt[1] * vRowByteStride + vPtxAsIntx16[1], tmp[2,3] has vPtyAsInt[3] * vRowByteStride + vPtxAsIntx16[3];
	tmp1 = _mm_add_epi64(tmp1, vImageBasePtr);
	tmp2 = _mm_add_epi64(tmp2, vImageBasePtr);

		// Fetch the components for each sample (remember 4 floats each).
		vSampleUL = _LoadA((float*) tmp1.m128i_u64[0]);
		vSampleUR = _LoadA((float*) tmp1.m128i_u64[1]);
		vSampleLL = _LoadA((float*) tmp2.m128i_u64[0]);
		vSampleLR = _LoadA((float*) tmp2.m128i_u64[1]);
#endif
#else
		pSampleUL = imageInfo.data2 + _AsArrayI(vPtyAsInt, 0) * rowByteStride + (_AsArrayI(vPtxAsIntx16, 0));
		pSampleUR = imageInfo.data2 + _AsArrayI(vPtyAsInt, 1) * rowByteStride + (_AsArrayI(vPtxAsIntx16, 1));
		pSampleLL = imageInfo.data2 + _AsArrayI(vPtyAsInt, 2) * rowByteStride + (_AsArrayI(vPtxAsIntx16, 2));
		pSampleLR = imageInfo.data2 + _AsArrayI(vPtyAsInt, 3) * rowByteStride + (_AsArrayI(vPtxAsIntx16, 3));

		// Fetch the components for each sample (remember 4 floats each).
		vSampleUL = _LoadA((float*) pSampleUL);
		vSampleUR = _LoadA((float*) pSampleUR);
		vSampleLL = _LoadA((float*) pSampleLL);
		vSampleLR = _LoadA((float*) pSampleLR);
#endif

		vFracX1X2X3X4 = _Sub(vPtx, vPtxAsIntFloat); // pt(i).x - (int) pt(i).x
		vFracY1Y2Y3Y4 = _Sub(vPty, vPtyAsIntFloat); // pt(i).y - (int) pt(i).y

		// Reform the components as SOA for more efficient SIMD processing.
		_MM_TRANSPOSE4_PS(vSampleUL, vSampleUR, vSampleLL, vSampleLR);

		vFracXY = _Mul(vFracX1X2X3X4, vFracY1Y2Y3Y4);

		va00 = vSampleUL;
		va10 = _Mul(vSampleUR, vFracX1X2X3X4);
		va01 = _Mul(vSampleLL, vFracY1Y2Y3Y4);
		va11 = _Mul(vSampleLR, vFracXY);

		va0010 = _Add(va00, va10);
		va0111 = _Add(va01, va11);

		// sample is the full bilinear sample for each of the four samples.
		sample = _Add(va0010, va0111);

		// From this we maintain weighted samples, temp weighted samples and the
		// sample squared of the weighted sample.
		sampleSq = _Mul(sample, sample);

		weights =  _LoadA(&w.pixelWeights[4+i*4]);
		tempWeights = _LoadA(&w.pixelTempWeights[4+i*4]);
		weightedSample = _Mul(sample, weights);
		weightedSampleSq = _Mul(sampleSq, weights);
		weightedNum = _Mul(sample, tempWeights);

		// Sum the contributions for each of the maintained quantities.
		// Remember, we are maintaining groups.
		vSum4 = _Add(vSum4, weightedSample);
		vSumSq4 = _Add(vSumSq4, weightedSampleSq);
		vNum4 = _Add(vNum4, weightedNum);

		// Advance to the next row of the patch.
			vX1X2X3X4 = _Add(vX1X2X3X4, vBasisVX);
			vY1Y2Y3Y4 = _Add(vY1Y2Y3Y4, vBasisVY);
			vZ1Z2Z3Z4 = _Add(vZ1Z2Z3Z4, vBasisVZ);

		vPtx = _Div(vX1X2X3X4, vZ1Z2Z3Z4);
		vPty = _Div(vY1Y2Y3Y4, vZ1Z2Z3Z4);
	}

	// sum4, num4 and sumSq4 are spread across their SIMD registers.
	// Here we have to sum their registers horizontally to get the
	// 5x5 sample block's total contribution.

	// This is purposely unfolded to help deduce dependency stalls.
	// Although vNum4 may not be needed by the caller, it is calculated
	// here as the dependency stalls essentially make this free.
#if 0 // Depending on SSE3 gets a bit more performance, but it isn't significant.
    __m128 shuf1 = _mm_movehdup_ps(vSum4);        // broadcast elements 3,1 to 2,0
    __m128 shuf2 = _mm_movehdup_ps(vSumSq4);        // broadcast elements 3,1 to 2,0
    __m128 shuf3 = _mm_movehdup_ps(vNum4);        // broadcast elements 3,1 to 2,0
    __m128 sums1 = _mm_add_ps(vSum4, shuf1);
    __m128 sums2 = _mm_add_ps(vSumSq4, shuf2);
    __m128 sums3 = _mm_add_ps(vNum4, shuf3);
    shuf1        = _mm_movehl_ps(shuf1, sums1); // high half -> low half
    shuf2        = _mm_movehl_ps(shuf2, sums2); // high half -> low half
    shuf3        = _mm_movehl_ps(shuf3, sums3); // high half -> low half
    sums1        = _mm_add_ss(sums1, shuf1);
    sums2        = _mm_add_ss(sums2, shuf2);
    sums3        = _mm_add_ss(sums3, shuf3);
#else
	_Data shuf1 = _mm_shuffle_ps(vSum4, vSum4, _MM_SHUFFLE(2, 3, 0, 1));     // [ C D | B A ]
	_Data shuf2 = _mm_shuffle_ps(vSumSq4, vSumSq4, _MM_SHUFFLE(2, 3, 0, 1)); // [ C D | B A ]
	_Data shuf3 = _mm_shuffle_ps(vNum4, vNum4, _MM_SHUFFLE(2, 3, 0, 1));     // [ C D | B A ]

	_Data sums1 = _Add(vSum4, shuf1);                                        // sums = [ D+C C+D | B+A A+B ]
	_Data sums2 = _Add(vSumSq4, shuf2);                                      // sums = [ D+C C+D | B+A A+B ]
	_Data sums3 = _Add(vNum4, shuf3);                                        // sums = [ D+C C+D | B+A A+B ]

	shuf1 = _mm_movehl_ps(shuf1, sums1);                                     // [   C   D | D+C C+D ] let the compiler avoid a mov by reusing shuf
	shuf2 = _mm_movehl_ps(shuf2, sums2);                                     // [   C   D | D+C C+D ] let the compiler avoid a mov by reusing shuf
	shuf3 = _mm_movehl_ps(shuf3, sums3);                                     // [ C D | D+C C+D ] let the compiler avoid a mov by reusing shuf

	sums1                  = _mm_add_ss(sums1, shuf1);
	sums2                  = _mm_add_ss(sums2, shuf2);
	sums3 = _mm_add_ss(sums3, shuf3);
#endif

	sumResult = _vFirst(sums1);
	sumSqResult = _vFirst(sums2);
	numResult = _vFirst(sums3);
	}

bool __declspec(safebuffers) DepthEstimator::ScorePixelImage(
	const ImageInfo_t& imageInfo
) noexcept
{
	float sum, sumSq, num;
	GatherSampleInfo(imageInfo, sum, sumSq, num);

	// Score similarity of the reference and target texture patches
	#if DENSE_NCC == DENSE_NCC_FAST
	const float normSq1(sumSq-SQUARE(sum/nSizeWindow));
	#elif DENSE_NCC == DENSE_NCC_WEIGHTED
	const float normSq1(sumSq-SQUARE(sum) * pWeightMap0Info.invSumWeights);
	#else
	const float normSq1(normSqDelta<float,float,nTexels>(texels1.data(), sum/(float)nTexels));
	#endif
	const float nrmSq(normSq0*normSq1);

	// Culls about 1/3 of the scores in limited testing.
	if (nrmSq <= 1e-16f) {
		return false;
	}

	const size_t scoreIdx = scoreResults.numScoreResults;
	scoreResults.pNums[scoreIdx] = num;
	scoreResults.pNrmSqs[scoreIdx] = nrmSq;

	// Is there a depth map contribution?
	// This will add to the score in the following manner:
	// 0 with no depthmap (eNoDepthMap)
	// 4 * OPTDENSE::fEstimationGeometricWeight if we miss (eMissedDepthMap)
	// consistency * OPTDENSE::fEstimationGeometricWeight if we hit (eHitDepthMap), calculation deferred.
	float score;
	if (float* depthMapData = (float*)imageInfo.depthMapData) {
		score = eMissedDepthMap;
		const auto vX1 = imageInfo.image.Tl4.Mul44Vec3(sh.mDepthMapPt);
		// Kj * Rj * (Ri.t() * X + Ci - Cj)
		if (
			(_AsArray(vX1, 2) > 0.f)  // Roughly 5% rejection.
			&	(_AsArray(vX1, 0) >= 1.f * _AsArray(vX1, 2)) // Roughly 1/3 are rejected,
			& (_AsArray(vX1, 0) <= imageInfo.depthMapWidthMinus2*_AsArray(vX1, 2))
			&	(_AsArray(vX1, 1) >= 1.f * _AsArray(vX1, 2)) // 1/2 are rejected.
			& (_AsArray(vX1, 1) <= imageInfo.depthMapHeightMinus2*_AsArray(vX1, 2))
		) {
			const _Data vX1Z = _Splat(vX1, 2);

			const _Data vPAsFloat = _Div(vX1, vX1Z);

			// Depth image may differ in size.
			const size_t byteStride = imageInfo.depthMapRowStride;
			const size_t floatStride = byteStride/sizeof(float);

			const _DataI vPAsInt = _TruncateIF(vPAsFloat);

			const size_t idx = _AsArrayI(vPAsInt, 1) * floatStride + _AsArrayI(vPAsInt, 0);
			const float* __restrict p1 = depthMapData + idx;

			_Data vSamples = _mm_castpd_ps(_mm_load_sd((const double*)p1));
			vSamples = _mm_loadh_pi(vSamples, (__m64*) (p1+floatStride));

			//x0y0, x1y0, x0y1, x1y1);
			//constexpr _Data vSmall = {0.03f, 0.03f, 0.03f, 0.03f};
			const auto vCmp = _CmpLT(FastAbs(_Sub(vX1Z, vSamples)), _Mul(vX1Z, _Set(0.03f)));

#ifdef USE_FAST_COMPARES
			if (auto mask = _mm_movemask_ps(vCmp)) { //!AllZerosI(_CastIF(vCmp))) { // b00 | b10 | b01 | b11) {
#else
			if (!AllZerosI(_CastIF(vCmp))) { // b00 | b10 | b01 | b11) {
#endif
				constexpr _Data vOne ={ 1.f, 1.f, 1.f, 1.f };
				const _Data vS = _Sub(vPAsFloat, _ConvertFI(vPAsInt));
				const _Data vOneMinusS = _Sub(vOne, vS);

				// JPB WIP OPT Tried a variety of different masking techniques including table lookups and bit shifting... none were faster.
				const int b00 =  mask&1; //_AsArrayI(_CastIF(vCmp), 0) == 0xFFFFFFFF;
				const int b10 =  mask&2; //_AsArrayI(_CastIF(vCmp), 1) == 0xFFFFFFFF;
				const int b01 =  mask&4; //_AsArrayI(_CastIF(vCmp), 2) == 0xFFFFFFFF;
				const int b11 =  mask&8; //_AsArrayI(_CastIF(vCmp), 3) == 0xFFFFFFFF;

				const float x0y0 = _AsArray(vSamples, 0);
				const float x1y0 = _AsArray(vSamples, 1);
				const float x0y1 = _AsArray(vSamples, 2);
				const float x1y1 = _AsArray(vSamples, 3);

				const float depth1 =
					_AsArray(vOneMinusS, 1)
					*	(_AsArray(vOneMinusS, 0)*Cast<float>(b00 ? x0y0 : (b10 ? x1y0 : (b01 ? x0y1 : x1y1)))	+ _AsArray(vS, 0)*Cast<float>(b10 ? x1y0 : (b00 ? x0y0 : (b11 ? x1y1 : x0y1))))
					+ _AsArray(vS, 1)
					* (_AsArray(vOneMinusS, 0)*Cast<float>(b01 ? x0y1 : (b11 ? x1y1 : (b00 ? x0y0 : x1y0))) + _AsArray(vS, 0)*Cast<float>(b11 ? x1y1 : (b01 ? x0y1 : (b10 ? x1y0 : x0y0))));

				// Ki * Ri * (Rj.t() * Kj-1 * X + Cj - Ci)

				const _Data vPt = imageInfo.image.Tr4.Mul44Vec3(_Mul(vPAsFloat, _Set(depth1))); //_SetN(X1PxAsFloat * depth1, X1PyAsFloat * depth1, depth1, 1.f));
				// Defer the calculation so consistency can be calculated vectorized.
				score = eHitDepthMap;
				scoreResults.pVs[scoreIdx] = vPt; // Hit depth map pixels need to have to have their consistency calculated.
			}
		}
	} else {
		score = eNoDepthMap;
	}

	scoreResults.pScores[scoreIdx] = score;

	return true;
}
#else // DPC_FASTER_SAMPLING
inline Matrix3x3f ComputeHomographyMatrix(const DepthData::ViewData& img, Depth depth, const Normal& normal, const Point3& X0) {
#if 0
	// compute homography matrix
	const Matrix3x3f H(img.camera.K*HomographyMatrixComposition(image0.camera, img.camera, Vec3(normal), Vec3(X0*depth))*image0.camera.K.inv());
#else
	// compute homography matrix as above, caching some constants
	const Vec3 n(normal);
	return (img.Hl + img.Hm * (n.t()*INVERT(n.dot(X0)*depth))) * img.Hr;
#endif
}

float DepthEstimator::ScorePixelImageOrig(
	ScoreResultsSOA_t& scoreResults,
	const DepthData::ViewData& image1, Depth depth, const Normal& normal)
{
	// center a patch of given size on the segment and fetch the pixel values in the target image
	Matrix3x3f H(ComputeHomographyMatrix(image1, depth, normal, Point3(_AsArray(vX0, 0), _AsArray(vX0, 1), _AsArray(vX0, 2))));
	Point3f X;
	ProjectVertex_3x3_2_3(H.val, Point2f(float(x0.x-nSizeHalfWindow),float(x0.y-nSizeHalfWindow)).ptr(), X.ptr());
	Point3f baseX(X);
	H *= float(nSizeStep);
	int n(0);
	float sum(0);
#if DENSE_NCC != DENSE_NCC_DEFAULT
	float sumSq(0), num(0);
#endif

#if DENSE_NCC == DENSE_NCC_WEIGHTED
	const Weight& w = pWeightMap;
#endif
	for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep) {
		for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep) {
			const Point2f pt(X);
			if (!image1.image.isInsideWithBorder<float,1>(pt))
				return thRobust;

			const int lx((int)pt.x);
			const int ly((int)pt.y);

			const float v(image1.image.sample(pt));

#if DENSE_NCC == DENSE_NCC_FAST
			sum += v;
			sumSq += SQUARE(v);
			num += texels0(n++)*v;
#elif DENSE_NCC == DENSE_NCC_WEIGHTED
			float vw;
			vw = v*w.weights[n];
			sum += vw;
			sumSq += v*vw;
			num += v*w.tempWeights[n];
			++n;
#else
			sum += texels1(n++)=v;
#endif
			X.x += H[0]; X.y += H[3]; X.z += H[6];
		}
		baseX.x += H[1]; baseX.y += H[4]; baseX.z += H[7];
		X = baseX;
	}

	scoreResults.pSum[scoreResults.numScoreResults] = sum;
	scoreResults.pNum[scoreResults.numScoreResults] = num;

	ASSERT(n == nTexels);
	// score similarity of the reference and target texture patches
#if DENSE_NCC == DENSE_NCC_FAST
	const float normSq1(sumSq-SQUARE(sum/nSizeWindow));
#elif DENSE_NCC == DENSE_NCC_WEIGHTED
	const float normSq1(sumSq-SQUARE(sum)/pWeightMap0Info.sumWeights);
#else
	const float normSq1(normSqDelta<float,float,nTexels>(texels1.data(), sum/(float)nTexels));
#endif
	const float nrmSq(normSq0*normSq1);
	if (nrmSq <=1e-16f)
		return thRobust;
#if DENSE_NCC == DENSE_NCC_DEFAULT
	const float num(texels0.dot(texels1));
#endif
	const float ncc(CLAMP(num/SQRT(nrmSq), -1.f, 1.f));
	float score(1.f-ncc);
#if 1
	score *= _vFirst(sh.mVScoreFactor);
#else
#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	// encourage smoothness
	for (const NeighborEstimate& neighbor: neighborsClose) {
		ASSERT(neighbor.depth > 0);
#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		const float factorDepth(DENSE_EXP(SQUARE(plane.Distance(neighbor.X)/depth) * smoothSigmaDepth));
#else
		const float factorDepth(DENSE_EXP(SQUARE((depth-neighbor.depth)/depth) * smoothSigmaDepth));
#endif
		const float factorNormal(DENSE_EXP(SQUARE(ACOS(ComputeAngle(normal.ptr(), neighbor.normal.ptr()))) * smoothSigmaNormal));
		score *= (1.f - smoothBonusDepth * factorDepth) * (1.f - smoothBonusNormal * factorNormal);
	}
#endif
#endif
	if (!image1.depthMap.empty()) {
		ASSERT(OPTDENSE::fEstimationGeometricWeight > 0);
		float consistency(4.f);
		const Point3f X1(image1.Tl*Point3f(float(X0.x)*depth,float(X0.y)*depth,depth)+image1.Tm); // Kj * Rj * (Ri.t() * X + Ci - Cj)
		if (X1.z > 0) {
			const Point2f x1(X1);
			if (image1.depthMap.isInsideWithBorder<float,1>(x1)) {
				Depth depth1;
				if (image1.depthMap.sample(depth1, x1, [&X1](Depth d) { return IsDepthSimilar(X1.z, d, 0.03f); })) {
					const Point2f xb(image1.Tr*Point3f(x1.x*depth1,x1.y*depth1,depth1)+image1.Tn); // Ki * Ri * (Rj.t() * Kj-1 * X + Cj - Ci)
					const float dist(norm(Point2f(float(x0.x)-xb.x, float(x0.y)-xb.y)));
					consistency = MINF(SQRT(dist*(dist+2.f)), consistency);
				}
			}
		}
		score += OPTDENSE::fEstimationGeometricWeight * consistency;
	}
	// apply depth prior weight based on patch textureless
	if (!lowResDepthMap.empty()) {
		const Depth d0 = lowResDepthMap(x0);
		if (d0 > 0) {
			const float deltaDepth(MINF(DepthSimilarity(d0, depth), 0.5f));
			const float smoothSigmaDepth(-1.f / (1.f * 0.02f)); // 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
			const float factorDeltaDepth(DENSE_EXP(normSq0 * smoothSigmaDepth));
			score = (1.f-factorDeltaDepth)*score + factorDeltaDepth*deltaDepth;
		}
	}
	ASSERT(ISFINITE(score));
	return MIN(2.f, score);
}
#endif // DPC_FASTER_SAMPLING

float DepthEstimator::ScorePixel(Depth depth, const Normal4& normal)
{
	ASSERT(depth > 0 && normal.Dot3S(vX0) <= 0);
	// compute score for this pixel as seen in each view
	//ASSERT(scores.size() == images.size());
#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL
	const float dot = normal.Dot3S(vX0);

	if (FastAbsS(dot) < 0.000001f) {
		return thRobust;
	}

	const float factor = 1.f / (depth * dot);
	sh.Detail(
		depth,
		_Mul(normal.data, _Set(factor)),
		_Mul(vX0, _Set(depth)),
		mLowResDepth
	);
#else
	const Vec3 normalD(_AsArray(normal.data, 0), _AsArray(normal.data, 1), _AsArray(normal.data, 2));
	sh.Detail(
		depth,
		Matx13_t(normalD.t()*INVERT(normalD.dot(Vec3(_AsArray(vX0, 0), _AsArray(vX0, 1), _AsArray(vX0, 2))) * depth)), // Always double.
		_Mul(vX0, _Set(depth)),
		mLowResDepth
	);
#endif // DPC_FASTER_SCORE_PIXEL_DETAIL

	// Note: scoreResults is always sized maximally.
	// scoreResults.numScoreResults is used to track scores as we identify them.

#ifndef DPC_FASTER_SAMPLING
	restart :
	std::vector<float> origScores;

	scoreResults.numScoreResults = 0;
	FOREACH(idxView, images) {
		float score = ScorePixelImageOrig(scoreResults, images[idxView], depth, normal.AsNormal());
		scoreResults.pScores[scoreResults.numScoreResults] = score;
		origScores.push_back(score);
		++scoreResults.numScoreResults;
	}
#else
	scoreResults.numScoreResults = 0;
	// Only called for estimating depth-maps (after first pass).
	// Scorable roughly 90% of the time.
	for (const auto& image : sh.imageInfo) {
		if (IsScorable3(image)) {
			scoreResults.numScoreResults += ScorePixelImage(image);
		}
	}

	if (0 == scoreResults.numScoreResults) {
		return thRobust; // JPB WIP May not work for all DENSE_AGGNCC
	}

	// The first pass does not have depth map data (we are creating it).
	const bool hasDepthMapData = sh.imageInfo.front().depthMapData;

	// Notice, thRobust scored images are removed from the score results.

	// JPB WIP OPT Although it appears promising to fold the work above into a single loop (where the following processes
	// up to GROUP_SIZE elements at time), it isn't faster in practice.
	// Maybe this hinders the effectiveness of the instruction cache?
	const _Data vPatchX = _Set((float)x0.x);
	const _Data vPatchY = _Set((float)x0.y);

	// In order to maximize SIMD workload, we process GROUP_SIZE
	// results at a time and handle the various cases without
	// branching.
	const size_t numElems = NumElemsForSize(scoreResults.numScoreResults);
	for (size_t i = 0; i < numElems; i += GROUP_SIZE) {
		// Beware: we may process partial groups where ultimately portions
		// of the registers may be undefined.

		// All calculations need:
		// const float ncc(FastClampS(num/FastSqrtS(nrmSq), -1.f, 1.f));
		// float score(1.f-ncc);
		// #if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		// score *= sh.mVScoreFactor;
		// #endif
		// 
		// if scoreResults.pScores[n] is eHitDepthMap, then it needs a score attenuation based on a weighted depth map consistency calculation.
		//		score += OPTDENSE::fEstimationGeometricWeight * vConsistencies;
		// 
		// If scoreResults.pScores[n] is eMissedDepthMap, then it needs a score attenuation based on a weighted constant.
		//		score += OPTDENSE::fEstimationGeometricWeight * 4.f;
		//
		// If scoreResults.pScores[n] is eNoDepthMap, then it no score attenuation.
		// 
		const _Data vNum = _LoadA(scoreResults.pNums+i);
		const _Data vNrmSq = _LoadA(scoreResults.pNrmSqs+i);
		const _Data vRawScores = _LoadA(scoreResults.pScores+i);

		// Compute for each element:
		//		const float ncc(FastClampS(num/FastSqrtS(nrmSq), -1.f, 1.f));
		//		float score(1.f-ncc);
		//.#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		//		score *= sh.mVScoreFactor;
		// #endif
#ifdef USE_INV_SQRT
#if 1
		_Data vNCC = _Mul(vNum, _mm_rsqrt_ps(vNrmSq));
#else
		_Data xmm0 = _mm_rsqrt_ps(vNrmSq);
		_Data xmm2 = xmm0;
		xmm0 = _Mul(xmm0, xmm0);
		xmm0 = _Mul(xmm0, vNrmSq);
		constexpr _Data vThree ={ 3.f, 3.f, 3.f, 3.f };
		xmm0 = _Sub(xmm0, vThree);
		xmm0 = _Mul(xmm0, xmm2);
		constexpr _Data vMinusHalf ={ -0.5f,-0.5f,-0.5f,-0.5f };
		_Data vNorm = _Mul(xmm0, vMinusHalf);
		_Data vNCC = _Mul(vNum, vNorm);
#endif
#else
		const _Data vNorm = _Sqrt(vNrmSq);
		_Data vNCC = _Div(vNum, vNorm);
#endif
		constexpr _Data vNegOne = { -1.f, -1.f, -1.f, -1.f };
		constexpr _Data vOne = { 1.f, 1.f, 1.f, 1.f };

		vNCC = FastClamp(vNCC, vNegOne, vOne);
		_Data vScores = _Sub(vOne, vNCC);

#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		vScores = _Mul(vScores, sh.mVScoreFactor);
#endif

		if (hasDepthMapData) {
			constexpr _Data vDefaultConsistencies = { 4.f, 4.f, 4.f, 4.f };

			// JPB WIP Hardcoded for GROUP_SIZE=4
			_Data vx0x1x2x3 = scoreResults.pVs[i];
			_Data vy0y1y2y3 = scoreResults.pVs[i + 1];
			_Data vz0z1z2z3 = scoreResults.pVs[i + 2];
			_Data vw0w1w2w3 = scoreResults.pVs[i + 3];

			_MM_TRANSPOSE4_PS( // vw0w1w2w3 will be all zeros and unused.
				vx0x1x2x3,
				vy0y1y2y3,
				vz0z1z2z3,
				vw0w1w2w3
			);
			// vn's now have x0, x1, x2, x3

			const _Data vProjX = _Div(vx0x1x2x3, vz0z1z2z3);
			const _Data vProjY = _Div(vy0y1y2y3, vz0z1z2z3);

			const _Data vDeltaX = _Sub(vPatchX, vProjX);
			const _Data vDeltaY = _Sub(vPatchY, vProjY);
			const _Data vDeltaXSq = _Mul(vDeltaX, vDeltaX);
			const _Data vDeltaYSq = _Mul(vDeltaY, vDeltaY);

			const _Data vDeltaXYSqSum = _Add(vDeltaXSq, vDeltaYSq);

			const _Data vDist = _Sqrt(vDeltaXYSqSum);

			// Constants have a long latency, but the compiler is smart enough to
			// create them far earlier to avoid this.
			constexpr _Data vTwo = { 2.f, 2.f, 2.f, 2.f };
			const _Data vConsistenciesLhs = _Mul(vDist, _Add(vDist, vTwo));
			const _Data vConsistencies = _Min(_Sqrt(vConsistenciesLhs), vDefaultConsistencies);

			constexpr _Data vNoDepthMap = { eNoDepthMap, eNoDepthMap, eNoDepthMap, eNoDepthMap };
			const _Data vNoDepthMapMask = _CmpEQ(vRawScores, vNoDepthMap); // 1's if vScores[i] is eNoDepthMap.

			constexpr _Data vMissedDepthMap = { eMissedDepthMap, eMissedDepthMap, eMissedDepthMap, eMissedDepthMap };
			const _Data vMissedDepthMapMask = _CmpEQ(vRawScores, vMissedDepthMap); // 1's if vScores[i] is eMissedDepthMap.

			constexpr _Data vHitDepthMap = { eHitDepthMap, eHitDepthMap, eHitDepthMap, eHitDepthMap };
			const _Data vHitDepthMapMask = _CmpEQ(vRawScores, vHitDepthMap); // 1's if vScores[i] is eHitDepthMap.

			// Set result[i] to vHitDepthMapPath[i] if vRawScores[i] == eHitDepthMap
			const _Data vHitDepthMapPath = _Add(vScores, _Mul(_Set(OPTDENSE::fEstimationGeometricWeight), vConsistencies));
			_Data vResult = Blend(vScores, vHitDepthMapPath, vHitDepthMapMask);

			// Set result[i] to vMissedDepthMapMask[i] if vRawScores[i] == vMissedDepthMap
			const _Data vMissedDepthMapPath = _Add(vScores, _Set(OPTDENSE::fEstimationGeometricWeight*4.f));
			vResult = Blend(vResult, vMissedDepthMapPath, vMissedDepthMapMask);

			// Set result[i] to vScores[i] if vRawScores[i] == eNoDepthMap
			const _Data vNoDepthMapPath = vScores;
			vScores = Blend(vResult, vNoDepthMapPath, vNoDepthMapMask);
		}

		_StoreA(&scoreResults.pScores[i], vScores);
	}

	if (mLowResDepth > 0.f) {
		// Attenuate score based on the low resolution depth map "apply depth prior weight based on patch textureless".
		// Frequently called during first pass.
		// Scorable almost all of the time.
		// Remember, its possible all of these are defined.
		const _Data vScaledDeltaDepth = _Mul(vFactorDeltaDepth, _Set(sh.mDeltaDepth));

		// Groups of two to reduce stalls.
		size_t i = 0;
		size_t numGroups = numElems / (GROUP_SIZE*2);
		while (numGroups--) {
			const _Data vCurrentScore1 = _LoadA(&scoreResults.pScores[i]);
			const _Data vCurrentScore2 = _LoadA(&scoreResults.pScores[i+GROUP_SIZE]);
			const _Data vFactoredScore1 = _Mul(vCurrentScore1, vOneMinusFactorDeltaDepth);
			const _Data vFactoredScore2 = _Mul(vCurrentScore2, vOneMinusFactorDeltaDepth);
			const _Data vFinalFactoredScore1 = _Add(vFactoredScore1, vScaledDeltaDepth);
			const _Data vFinalFactoredScore2 = _Add(vFactoredScore2, vScaledDeltaDepth);
			_StoreA(&scoreResults.pScores[i], vFinalFactoredScore1);
			_StoreA(&scoreResults.pScores[i+GROUP_SIZE], vFinalFactoredScore2);
			i += GROUP_SIZE*2;
		}

		// Handle the odd group if necessary.
		if (numElems & (GROUP_SIZE*2-1)) {
			const _Data vCurrentScore = _LoadA(&scoreResults.pScores[i]);
			const _Data vFactoredScore = _Add(_Mul(vCurrentScore, vOneMinusFactorDeltaDepth), vScaledDeltaDepth);
			_StoreA(&scoreResults.pScores[i], vFactoredScore);
		}
	}

	// JPB WIP OPT Don't need to clamp since values >= thRobust will be removed anyway.
#ifdef CLAMP_SCORES
	// Be sure the scores are no greater than 2 in any scenario.
	constexpr _Data vTwo ={ 2.f, 2.f, 2.f, 2.f };
	for (size_t i = 0; i < (scoreResults.numScoreResults & ~(GROUP_SIZE-1)); i += GROUP_SIZE) {
		const _Data vCurrentScore = _LoadA(&scoreResults.pScores[i]);
		const _Data vClampedScore = _Min(vCurrentScore, vTwo);
		_StoreA(&scoreResults.pScores[i], vClampedScore);
	}
#endif
#endif // DPC_FASTER_SAMPLING

	// Revisit n'th element for the special case of 2.
	float smallest = scoreResults.pScores[0];
	float secondSmallest = FLT_MAX;
	for (size_t i = 1; i < scoreResults.numScoreResults; ++i) {
		const uint32_t value = (uint32_t&) scoreResults.pScores[i];
		if (value < (uint32_t&) secondSmallest) {
			if (value < (uint32_t&) smallest) {
				(uint32_t&) secondSmallest = (uint32_t&) smallest;
				(uint32_t&) smallest = value;
			}
			else {
				(uint32_t&) secondSmallest = value;
			}
		}
	}

	return 0.5f * (smallest + ( secondSmallest < thRobust ? secondSmallest : smallest ) );
}

// Interestingly moving this inside CalculateScoreFactor causes the mask table
// to be built at runtime.
// First (only available member to initialize) is m128i_i8
constexpr _DataI vSFMasks[] = {
	{0x00,0x00,0x00,0x00,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,-1,-1,-1,-1,-1,-1,-1,-1},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,-1,-1,-1,-1},
	{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}
};

_Data DepthEstimator::CalculateScoreFactor(
	_Data normal,
	float depth,
	size_t numNeighbors,
	const _Data* __restrict neighborsCloseX,
	const _Data* __restrict neighborsCloseNormals
)
{
	// neighborsCloseX and neighborsCloseNormals stored as SOA.
	const _Data vSmoothBonusDepth = _Set(smoothBonusDepth);
	const _Data vSmoothSigmaNormal = _Set(smoothSigmaNormal);
	const _Data vSmoothBonusNormal = _Set(smoothBonusNormal);
	const _Data vDepthFactor = _Set(smoothSigmaDepth/(depth*depth));

	const _Data vPlaneX = _Splat(plane, 0);
	const _Data vPlaneY = _Splat(plane, 1);
	const _Data vPlaneZ = _Splat(plane, 2);
	const _Data vPlaneD = _Splat(plane, 3);

	const _Data vNormalX = _Splat(normal, 0);
	const _Data vNormalY = _Splat(normal, 1);
	const _Data vNormalZ = _Splat(normal, 2);

	_Data neighborsPlanePosDot = _Mul(vPlaneX, neighborsCloseX[0]);
	_Data neighborsCosPlaneNormalAngle = _Mul(vNormalX, neighborsCloseNormals[0]);

	neighborsPlanePosDot = _Add(neighborsPlanePosDot, _Mul(vPlaneY, neighborsCloseX[1]));
	neighborsCosPlaneNormalAngle = _Add(neighborsCosPlaneNormalAngle, _Mul(vNormalY, neighborsCloseNormals[1]));

	neighborsPlanePosDot = _Add(neighborsPlanePosDot, _Mul(vPlaneZ, neighborsCloseX[2]));
	neighborsCosPlaneNormalAngle = _Add(neighborsCosPlaneNormalAngle, _Mul(vNormalZ, neighborsCloseNormals[2]));

	neighborsPlanePosDot = _Add(neighborsPlanePosDot, vPlaneD);
	neighborsCosPlaneNormalAngle = _Add(neighborsCosPlaneNormalAngle, neighborsCloseNormals[3]);

	const _Data vFdSquared = _Mul(neighborsPlanePosDot, neighborsPlanePosDot);
	const _Data vFdSquaredDepthFactor = _Mul(vFdSquared, vDepthFactor);
#ifdef DPC_FASTER_SCORE_FACTOR
	const _Data vExpFdSquaredDepthFactor = FastExpAlwaysNegative(vFdSquaredDepthFactor);
#else
	const _Data vExpFdSquaredDepthFactor = _SetN(DENSE_EXP(_AsArray(vFdSquaredDepthFactor, 0)), DENSE_EXP(_AsArray(vFdSquaredDepthFactor, 1)), DENSE_EXP(_AsArray(vFdSquaredDepthFactor, 2)), DENSE_EXP(_AsArray(vFdSquaredDepthFactor, 3)));
#endif
	const _Data vExpFdSquaredDepthFactorBD = _Mul(vExpFdSquaredDepthFactor, vSmoothBonusDepth);
	constexpr _Data vOne {1.f, 1.f, 1.f, 1.f};
	const _Data vScoreFactorLhs = _Sub(vOne, vExpFdSquaredDepthFactorBD);

	constexpr _Data vNegOne {-1.f, -1.f, -1.f, -1.f};
	const _Data vAnglesClamped = FastClamp(neighborsCosPlaneNormalAngle, vNegOne, vOne);
	const _Data vTmp = FastACos(vAnglesClamped);
	const _Data vTmpSquared = _Mul(vTmp, vTmp);
	const _Data vTmpSquaredvSmoothSigmaNormal = _Mul(vTmpSquared, vSmoothSigmaNormal);
#ifdef DPC_FASTER_SCORE_FACTOR
		const _Data vExpTmpSquaredvSmoothSigmaNormal = FastExpAlwaysPositive(vTmpSquaredvSmoothSigmaNormal);
#else
	const _Data vExpTmpSquaredvSmoothSigmaNormal = _SetN(DENSE_EXP(_AsArray(vTmpSquaredvSmoothSigmaNormal, 0)), DENSE_EXP(_AsArray(vTmpSquaredvSmoothSigmaNormal, 1)), DENSE_EXP(_AsArray(vTmpSquaredvSmoothSigmaNormal, 2)), DENSE_EXP(_AsArray(vTmpSquaredvSmoothSigmaNormal, 3)));
#endif
	const _Data vExpTmpSquaredvSmoothSigmaNormalBN = _Mul(vExpTmpSquaredvSmoothSigmaNormal, vSmoothBonusNormal);
	const _Data vScoreFactorRhs = _Sub(vOne, vExpTmpSquaredvSmoothSigmaNormalBN);

	_Data vScoreFactor = _Mul(vScoreFactorLhs, vScoreFactorRhs);
	vScoreFactor = Blend(vScoreFactor, vOne, _CastFI(vSFMasks[numNeighbors-1]));

	return FastHProduct(vScoreFactor);
}

// run propagation and random refinement cycles;
// the solution belonging to the target image can be also propagated
void DepthEstimator::ProcessPixel(IDX idx)
{
	// compute pixel coordinates from pixel index and its neighbors
	ASSERT(dir == LT2RB || dir == RB2LT);
	if (!PreparePixelPatch(dir == LT2RB ? coords[idx] : coords[coords.GetSize()-1-idx]))
	{
		return;
	}

	if (sh.mLowResDepthMapEmpty) {
		if (!FillPixelPatch<false /* has a low res depth map */>())
			return;
	}	else {
		if (!FillPixelPatch<true /* has a low res depth map */>())
			return;
	}

	// find neighbors
	ImageRef ppNeighbors[2]; //neighbors.Empty();
	int numPPNeighbors = 0;
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	_Data neighborsCloseNormals[4];
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		int numNeighbors = 0;
		_Data neighborsCloseCoord[4];
		 neighborsCloseCoord[3] = _Set(1.f);
		#endif
	Depth neighborsCloseDepths[4];
	#endif

	// dir alternates on each iteration.
	if (dir == LT2RB) {
		// direction from left-top to right-bottom corner
#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		// Will likely calculate x0-1, x0 twice, x0+1 and similar for y0.
		const double denX = 1.f/image0.camera.K(0,0);
		const double tx = (x0.x-image0.camera.K(0,2))*denX;
		const double denY = 1.f/image0.camera.K(1,1);
		const double ty = (x0.y-image0.camera.K(1,2))*denY;
#endif

		if (x0.x > nSizeHalfWindow) {
			const ImageRef nx(x0.x-1, x0.y);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				ppNeighbors[numPPNeighbors] = nx;
				++numPPNeighbors;
				// Notice, here, and below we create a raw pointer to the normal and use only
				// 3 floats of the 4 loaded floats.  This is safe since we are always reading
				// within the boundary of the image.
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x-1.f, x0.y, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( ( tx-denX )*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ty*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				#else
				ppNeighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
				++numNeighbors;
			}
		}
		if (x0.y > nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y-1);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				ppNeighbors[numPPNeighbors] = nx;
				++numPPNeighbors;
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x, x0.y-1.f, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( tx*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ( ty-denY )*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				#else
				ppNeighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
				++numNeighbors;
			}
		}
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		if (x0.x < size.width-nSizeHalfWindow) {
			const ImageRef nx(x0.x+1, x0.y);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x+1.f, x0.y, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( ( tx+denX )*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ty*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				++numNeighbors;
			}
		}
		if (x0.y < size.height-nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y+1);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x, x0.y+1.f, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( tx*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ( ty+denY )*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				++numNeighbors;
			}
		}
		#endif
	} else {
		ASSERT(dir == RB2LT);
		double denX = 1.f/image0.camera.K(0,0);
		double tx = (x0.x-image0.camera.K(0,2))*denX;
		double denY = 1.f/image0.camera.K(1,1);
		double ty = (x0.y-image0.camera.K(1,2))*denY;
		Point3f X;
		// direction from right-bottom to left-top corner
		if (x0.x < size.width-nSizeHalfWindow) {
			const ImageRef nx(x0.x+1, x0.y);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				ppNeighbors[numPPNeighbors] = nx;
				++numPPNeighbors;
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x+1.f, x0.y, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( ( tx+denX )*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ty*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				#else
				ppNeighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
				++numNeighbors;
			}
		}
		if (x0.y < size.height-nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y+1);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				ppNeighbors[numPPNeighbors] = nx;
				++numPPNeighbors;
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x, x0.y+1.f, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( tx*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ( ty+denY )*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				#else
				ppNeighbors.emplace_back(NeighborData{nx,ndepth,normalMap0(nx)});
				#endif
				++numNeighbors;
			}
		}
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		if (x0.x > nSizeHalfWindow) {
			const ImageRef nx(x0.x-1, x0.y);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x-1, x0.y, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( ( tx-denX )*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ty*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				++numNeighbors;
			}
		}
		if (x0.y > nSizeHalfWindow) {
			const ImageRef nx(x0.x, x0.y-1);
			const Depth ndepth(depthMap0.pix(nx));
			if (ndepth > 0) {
				ASSERT(ISEQUAL(norm(normalMap0.pix(nx)), 1.f));
				neighborsCloseNormals[numNeighbors]  = Load3Unsafe(normalMap0.pix(nx).ptr());
					#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
					//const auto Xorig = Cast<float>(image0.camera.TransformPointI2C(Point3(x0.x, x0.y-1.f, ndepth)));
					_AsArray(neighborsCloseCoord[0], numNeighbors) = (float)( tx*ndepth );
					_AsArray(neighborsCloseCoord[1], numNeighbors) = (float)( ( ty-denY )*ndepth );
					_AsArray(neighborsCloseCoord[2], numNeighbors) = (float)ndepth;
					#endif
				neighborsCloseDepths[numNeighbors] = ndepth;
				++numNeighbors;
			}
		}
		#endif
	}

#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
	_Data neighborsCloseNormalsSOA[4];
	for (int i = 0; i < numNeighbors; ++i) {
		neighborsCloseNormalsSOA[i] = neighborsCloseNormals[i];
	}
	_MM_TRANSPOSE4_PS(neighborsCloseNormalsSOA[0], neighborsCloseNormalsSOA[1], neighborsCloseNormalsSOA[2], neighborsCloseNormalsSOA[3]);
#endif

	float& conf = confMap0.pix(x0);
	Depth& depth = depthMap0.pix(x0);
	Normal& normal = normalMap0.pix(x0);
	const float fNCCThresholdKeep = OPTDENSE::fNCCThresholdKeep;
	const Point3f viewDir(_AsArray(vX0,0), _AsArray(vX0,1), _AsArray(vX0,2));
	ASSERT(depth > 0 && normal.dot(viewDir) <= 0);
	#if DENSE_REFINE == DENSE_REFINE_ITER
	// check if any of the neighbor estimates are better then the current estimate
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	for (int n = 0; n < numPPNeighbors; ++n) {
		const ImageRef& nx = ppNeighbors[n];
	#else
	for (NeighborData& neighbor: neighbors) {
		const ImageRef& nx = neighbor.x;
	#endif
		if (confMap0.pix(nx) >= fNCCThresholdKeep)
			continue;
		// I believe only the #if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA works properly.
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		const _Data& neighborNormal = neighborsCloseNormals[n];
		#endif
		Normal nn = Normal(_AsArray(neighborNormal, 0), _AsArray(neighborNormal, 1), _AsArray(neighborNormal, 2));
		ASSERT(ISEQUAL(norm(nn), 1.f)); // JPB WIP BUG 
		const Depth neighborDepth = InterpolatePixel(nx, neighborsCloseDepths[n], nn);
		CorrectNormal(nn);
		//ASSERT(neighbor.depth > 0 && neighbor.normal.dot(viewDir) <= 0);
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(neighborDepth, nn);
		sh.mVScoreFactor = CalculateScoreFactor(
			neighborNormal,
			neighborDepth,
			numNeighbors,
			&neighborsCloseCoord[0],
			&neighborsCloseNormalsSOA[0]
		);
		#endif
		// Any ScorePixel needs sh.mVScoreFactor set accurately.
		const float nconf(ScorePixel(neighborDepth, Normal4(nn)));
		ASSERT(nconf >= 0 && nconf <= 2);
		if (conf > nconf) {
			conf = nconf;
			depth = neighborDepth;
			normal = nn;
		}
	}

#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	bool ignoreNeighbors = false;
#endif
	// try random values around the current estimate in order to refine it
	unsigned idxScaleRange(0);
	RefineIters:
	if (conf <= thConfSmall)
		idxScaleRange = 2;
	else if (conf <= thConfBig)
		idxScaleRange = 1;
	else if (conf >= thConfRand) {
		// try completely random values in order to find an initial estimate
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		ignoreNeighbors = true; // neighborsCloseNormals.clear();
		#endif
		// A scorefactor is 1.f if it has no neighbors.
		sh.mVScoreFactor = _Set(1.f);

#ifdef DPC_FASTER_RANDOM_ITER_CALC
		Point2f p(
			_vFirst(FastATan2(_Set(normal.y), _Set(normal.x))), // atan2(normal.y, normal.x),
			FastACosS(normal.z) // acos(normal.z);
		);

		bool usenp2 = false;
		Normal nnormal;
		Point2f np, np2;
		v4sf vSinResult;
		v4sf vCosResult;

		for (unsigned iter=0, cnt = OPTDENSE::nRandomIters; iter<cnt; ++iter) {
			const Depth ndepth(RandomDepth(rnd, dMinSqr, dMaxSqr));

			// const Normal nnormal(RandomNormal(rnd, viewDir));
			int baseIndex;
			if (usenp2) {
				np = np2;
				baseIndex = 2;
			} else {
				// Get the next two iterations of sin/cos pairs.
				np = Point2f(
					rnd.randomRange(FD2R(0.f), FD2R(180.f)),
					rnd.randomRange(FD2R(90.f), FD2R(180.f))
				);
				// Speculatively calculate np2 in the hope it will be used.
				np2 = Point2f(
					rnd.randomRange(FD2R(0.f), FD2R(180.f)),
					rnd.randomRange(FD2R(90.f), FD2R(180.f))
				);

				sincos_ps(v4sf{ np.x, np.y, np2.x, np2.y }, &vSinResult, &vCosResult);
				baseIndex = 0;
			}
			nnormal = Normal(
				_AsArray(vCosResult, baseIndex)*_AsArray(vSinResult, baseIndex+1),
				_AsArray(vSinResult, baseIndex)*_AsArray(vSinResult, baseIndex+1),
				_AsArray(vCosResult, baseIndex+1)
			);

			if (nnormal.dot(viewDir) > 0) {
				nnormal = -nnormal;
			}
#else
		for (unsigned iter=0; iter<OPTDENSE::nRandomIters; ++iter) {
			const Depth ndepth(RandomDepth(rnd, dMinSqr, dMaxSqr));
			const Normal nnormal(RandomNormal(rnd, viewDir));
#endif // DPC_FASTER_RANDOM_ITER_CALC

			// Any ScorePixel needs sh.mVScoreFactor set accurately.
			// However, here we are ignoring neighbors and sh.mVScoreFactor
			// will be 1.
			const float nconf(ScorePixel(ndepth, Normal4(nnormal)));
			ASSERT(nconf >= 0);
			if (conf > nconf) {
				conf = nconf;
				depth = ndepth;
				normal = nnormal;
				if (conf < thConfRand)
					goto RefineIters;
			}
#ifdef DPC_FASTER_RANDOM_ITER_CALC
			usenp2 = !usenp2;
#endif
		}
		return;
	}

#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
	if (ignoreNeighbors) {
		sh.mVScoreFactor = _Set(1.f);
	}
#endif

	float scaleRange(scaleRanges[idxScaleRange]);
	const float depthRange(MaxDepthDifference(depth, OPTDENSE::fRandomDepthRatio));

#ifdef DPC_FASTER_RANDOM_ITER_CALC
		Point2f p(
		_vFirst(FastATan2(_Set(normal.y), _Set(normal.x))), // atan2(normal.y, normal.x),
		FastACosS(normal.z) // acos(normal.z);
	);
#else
	Point2f p;
	Normal2Dir(normal, p);
#endif

	Normal nnormal;
	bool usenp2 = false;
	Point2f np, np2;
	v4sf vSinResult;
	v4sf vCosResult;

	for (unsigned iter=0, cnt = OPTDENSE::nRandomIters; iter< cnt; ++iter) {
		const Depth ndepth(rnd.randomMeanRange(depth, depthRange*scaleRange));
		if (!ISINSIDE(ndepth, dMin, dMax))
			continue;
		
#ifdef DPC_FASTER_RANDOM_ITER_CALC
		int baseIndex;
		if (usenp2) {
			np = np2;
			baseIndex = 2;
		} else {
			const float randomAngle1Range = angle1Range*scaleRange;
			const float randomAngle2Range = angle2Range*scaleRange;

			// random() is (uint64_t) (state * big num) / nl<uint64_t>::max()
			// 	return p.x + randomAngle1Range * (T(2) * random<T>() - T(1));
			// 	return p.x + randomAngle1Range * (T(2) * (rand()/max()) - T(1));

			np = Point2f(
				rnd.randomMeanRange(p.x, randomAngle1Range),
				rnd.randomMeanRange(p.y, randomAngle2Range)
			);
			// Speculatively calculate np2 in the hope it will be used.
			np2 = Point2f(
				rnd.randomMeanRange(p.x, randomAngle1Range),
				rnd.randomMeanRange(p.y, randomAngle2Range)
			);

			// Gets sin(np.x, np.y, np2.x, np2.y) and cos(same) simultaneously.
			sincos_ps(v4sf{ np.x, np.y, np2.x, np2.y }, &vSinResult, &vCosResult);

			baseIndex = 0;
		}
		nnormal[0] = _AsArray(vCosResult, baseIndex)*_AsArray(vSinResult, baseIndex+1);
		nnormal[1] = _AsArray(vSinResult, baseIndex)*_AsArray(vSinResult, baseIndex+1);
		nnormal[2] = _AsArray(vCosResult, baseIndex+1);
#else
		const Point2f np(rnd.randomMeanRange(p.x, angle1Range*scaleRange), rnd.randomMeanRange(p.y, angle2Range*scaleRange));
		Dir2Normal(np, nnormal);
#endif // DPC_FASTER_RANDOM_ITER_CALC
		if (nnormal.dot(viewDir) >= 0)
			continue;

		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(ndepth, nnormal);
		#endif

		// ScorePixel needs sh.mVScoreFactor set accurately.
		// mVScoreFactor changes with a depth, normal, or neighbor change.
		if (ignoreNeighbors) {
			// sh.mVScoreFactor set previously to 1.f
		} else {
			sh.mVScoreFactor = CalculateScoreFactor(
				_SetN(nnormal[0], nnormal[1], nnormal[2], 0.f),
				ndepth,
				numNeighbors,
				&neighborsCloseCoord[0],
				&neighborsCloseNormalsSOA[0]
			);
		}

		const float nconf(ScorePixel(ndepth, Normal4(nnormal)));
		ASSERT(nconf >= 0);
#ifdef DPC_FASTER_RANDOM_ITER_CALC
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
			p = np;
			scaleRange = scaleRanges[++idxScaleRange];
			usenp2 = false;
		} else {
			usenp2 = !usenp2;
		}
#else
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
			p = np;
			scaleRange = scaleRanges[++idxScaleRange];
		}
#endif // DPC_FASTER_RANDOM_ITER_CALC
	}
	#else
	// current pixel estimate
	PixelEstimate currEstimate{depth, normal};
	// propagate depth estimate from the best neighbor estimate
	PixelEstimate prevEstimate; float prevCost(FLT_MAX);
	#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
	FOREACH(n, ppNeighbors) {
		const ImageRef& nx = ppNeighbors[n];
	#else
	for (const NeighborData& neighbor: ppNeighbors) {
		const ImageRef& nx = neighbor.x;
	#endif
		float nconf(confMap0(nx));
		const unsigned nidxScaleRange(DecodeScoreScale(nconf));
		ASSERT(nconf >= 0 && nconf <= 2);
		if (nconf >= OPTDENSE::fNCCThresholdKeep)
			continue;
		if (prevCost <= nconf)
			continue;
		#if DENSE_SMOOTHNESS != DENSE_SMOOTHNESS_NA
		const NeighborEstimate& neighbor = neighborsClose[n];
		#endif
		if (neighbor.normal.dot(viewDir) >= 0)
			continue;
		prevEstimate.depth = InterpolatePixel(nx, neighbor.depth, neighbor.normal);
		prevEstimate.normal = neighbor.normal;
		CorrectNormal(prevEstimate.normal);
		prevCost = nconf;
	}
	if (prevCost == FLT_MAX)
		prevEstimate = PerturbEstimate(currEstimate, thPerturbation);
	// randomly sampled estimate
	PixelEstimate randEstimate(PerturbEstimate(currEstimate, thPerturbation));
	// select best pixel estimate
	const int numCosts = 5;
	float costs[numCosts] = {0,0,0,0,0};
	const Depth depths[numCosts] = {
		currEstimate.depth, prevEstimate.depth, randEstimate.depth,
		currEstimate.depth, randEstimate.depth};
	const Normal normals[numCosts] = {
		currEstimate.normal, prevEstimate.normal,
		randEstimate.normal, randEstimate.normal,
		currEstimate.normal};
	conf = FLT_MAX;
	for (int idxCost=0; idxCost<numCosts; ++idxCost) {
		const Depth ndepth(depths[idxCost]);
		const Normal nnormal(normals[idxCost]);
		#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
		InitPlane(ndepth, nnormal);
		#endif
		const float nconf(ScorePixel(ndepth, Normal4(nnormal)));
		ASSERT(nconf >= 0);
		if (conf > nconf) {
			conf = nconf;
			depth = ndepth;
			normal = nnormal;
		}
	}
	#endif
}

// interpolate given pixel's estimate to the current position
Depth DepthEstimator::InterpolatePixel(const ImageRef& nx, Depth depth, const Normal& normal) const
{
	ASSERT(depth > 0 && normal.dot(image0.camera.TransformPointI2C(Cast<REAL>(nx))) <= 0);
	Depth depthNew;
	#if 1
	// compute as intersection of the lines
	// {(x1, y1), (x2, y2)} from neighbor's 3D point towards normal direction
	// and
	// {(0, 0), (x4, 1)} from camera center towards current pixel direction
	// in the x or y plane
	if (x0.x == nx.x) {
		const float nx1((float)(((REAL)x0.y - image0.camera.K(1,2)) / image0.camera.K(1,1)));
		const float denom(normal.z + nx1 * normal.y);
		if (FastAbsS(denom) < FZERO_TOLERANCE)
			return depth;
		const float x1((float)(((REAL)nx.y - image0.camera.K(1,2)) / image0.camera.K(1,1)));
		const float nom(depth * (normal.z + x1 * normal.y));
		depthNew = nom / denom;
	}
	else {
		ASSERT(x0.y == nx.y);
		const float nx1((float)(((REAL)x0.x - image0.camera.K(0,2)) / image0.camera.K(0,0)));
		const float denom(normal.z + nx1 * normal.x);
		if (FastAbsS(denom) < FZERO_TOLERANCE)
			return depth;
		const float x1((float)(((REAL)nx.x - image0.camera.K(0,2)) / image0.camera.K(0,0)));
		const float nom(depth * (normal.z + x1 * normal.x));
		depthNew = nom / denom;
	}
	#else
	// compute as the ray - plane intersection
	{
		#if 0
		const Plane plane(Cast<REAL>(normal), image0.camera.TransformPointI2C(Point3(nx, depth)));
		const Ray3 ray(Point3::ZERO, normalized(X0));
		depthNew = (Depth)ray.Intersects(plane).z();
		#else
		const Point3 planeN(normal);
		const REAL planeD(planeN.dot(image0.camera.TransformPointI2C(Point3(nx, depth))));
		depthNew = (Depth)(planeD / planeN.dot(X0));
		#endif
	}
	#endif
	return ISINSIDE(depthNew,dMin,dMax) ? depthNew : depth;
}

#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
// compute plane defined by current depth and normal estimate
void DepthEstimator::InitPlane(Depth depth, const Normal& normal)
{
#if 0
	plane.Set(normal, Normal(Cast<float>(X0)*depth));
#else
	const float D = -depth*(normal[0]*_AsArray(vX0,0) + normal[1]*_AsArray(vX0,1) + normal[2]*_AsArray(vX0,2));
	plane = _SetN(normal[0], normal[1], normal[2], D);
#endif
}
#endif

#if DENSE_REFINE == DENSE_REFINE_EXACT
DepthEstimator::PixelEstimate DepthEstimator::PerturbEstimate(const PixelEstimate& est, float perturbation)
{
	PixelEstimate ptbEst;

	// perturb depth
	const float minDepth = est.depth * (1.f-perturbation);
	const float maxDepth = est.depth * (1.f+perturbation);
	ptbEst.depth = CLAMP(rnd.randomUniform(minDepth, maxDepth), dMin, dMax);

	// perturb normal
	const Normal viewDir(Cast<float>(X0));
	std::uniform_real_distribution<float> urd(-1.f, 1.f);
	const int numMaxTrials = 3;
	int numTrials = 0;
	perturbation *= FHALF_PI;
	while(true) {
		// generate random perturbation rotation
		const RMatrixBaseF R(urd(rnd)*perturbation, urd(rnd)*perturbation, urd(rnd)*perturbation);
		// perturb normal vector
		ptbEst.normal = R * est.normal;
		// make sure the perturbed normal is still looking towards the camera,
		// otherwise try again with a smaller perturbation
		if (ptbEst.normal.dot(viewDir) < 0.f)
			break;
		if (++numTrials == numMaxTrials) {
			ptbEst.normal = est.normal;
			return ptbEst;
		}
		perturbation *= 0.5f;
	}
	ASSERT(ISEQUAL(norm(ptbEst.normal), 1.f));

	return ptbEst;
}
#endif
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

namespace CGAL {
}

// triangulate in-view points, generating a 2D mesh
// return also the estimated depth boundaries (min and max depth)
std::pair<float,float> TriangulatePointsDelaunay(const DepthData::ViewData& image, const PointCloudStreaming& pointcloud, const IndexArr& points, Mesh& mesh, Point2fArr& projs, bool bAddCorners)
{
	typedef CGAL::Simple_cartesian<double> kernel_t;
	typedef CGAL::Triangulation_vertex_base_with_info_2<Mesh::VIndex, kernel_t> vertex_base_t;
	typedef CGAL::Triangulation_data_structure_2<vertex_base_t> triangulation_data_structure_t;
	typedef CGAL::Delaunay_triangulation_2<kernel_t, triangulation_data_structure_t> Delaunay;
	typedef Delaunay::Face_circulator FaceCirculator;
	typedef Delaunay::Face_handle FaceHandle;
	typedef Delaunay::Vertex_handle VertexHandle;
	typedef kernel_t::Point_2 CPoint;

	ASSERT(sizeof(Point3) == sizeof(X3D));
	ASSERT(sizeof(Point2) == sizeof(CPoint));
	std::pair<float,float> depthBounds(FLT_MAX, 0.f);
	mesh.vertices.reserve((Mesh::VIndex)points.size()+4);
	projs.reserve(mesh.vertices.capacity());
	Delaunay delaunay;
	for (uint32_t idx: points) {
		const Point3f pt(image.camera.ProjectPointP3((Point3f&)pointcloud.PointStream()[idx*3]));
		const Point3f x(pt.x/pt.z, pt.y/pt.z, pt.z);
		delaunay.insert(CPoint(x.x, x.y))->info() = mesh.vertices.size();
		mesh.vertices.emplace_back(image.camera.TransformPointI2C(x));
		projs.emplace_back(x.x, x.y);
		if (depthBounds.first > pt.z)
			depthBounds.first = pt.z;
		if (depthBounds.second < pt.z)
			depthBounds.second = pt.z;
	}
	// if full size depth-map requested
	const size_t numPoints(3);
	if (bAddCorners && points.size() >= numPoints) {
		// add the four image corners at the average depth
		ASSERT(image.pImageData->IsValid() && ISINSIDE(image.pImageData->avgDepth, depthBounds.first, depthBounds.second));
		const Mesh::VIndex idxFirstVertex = mesh.vertices.size();
		VertexHandle vcorners[4];
		for (const Point2f x: {Point2i(0, 0), Point2i(image.image.width()-1, 0), Point2i(0, image.image.height()-1), Point2i(image.image.width()-1, image.image.height()-1)}) {
			const Mesh::VIndex i(mesh.vertices.size() - idxFirstVertex);
			(vcorners[i] = delaunay.insert(CPoint(x.x, x.y)))->info() = mesh.vertices.size();
			mesh.vertices.emplace_back(image.camera.TransformPointI2C(Point3f(x, image.pImageData->avgDepth)));
			projs.emplace_back(x);
		}
		// compute average depth from the closest 3 directly connected faces,
		// weighted by the distance
		for (int i=0; i<4; ++i) {
			const VertexHandle vcorner(vcorners[i]);
			FaceCirculator cfc(delaunay.incident_faces(vcorner));
			ASSERT(cfc != 0);
			const FaceCirculator done(cfc);
			const Point2d& posA = reinterpret_cast<const Point2d&>(vcorner->point());
			const Ray3d rayA(Point3d::ZERO, normalized(Cast<REAL>(mesh.vertices[vcorner->info()])));
			typedef TIndexScore<float,float> DepthDist;
			CLISTDEF0(DepthDist) depths(0, numPoints);
			do {
				const FaceHandle fc(cfc->neighbor(cfc->index(vcorner)));
				if (delaunay.is_infinite(fc))
					continue;
				for (int j=0; j<4; ++j)
					if (fc->has_vertex(vcorners[j]))
						continue;
				// compute the depth as the intersection of the corner ray with
				// the plane defined by the face's vertices
				const Planed planeB(
					Cast<REAL>(mesh.vertices[fc->vertex(0)->info()]),
					Cast<REAL>(mesh.vertices[fc->vertex(1)->info()]),
					Cast<REAL>(mesh.vertices[fc->vertex(2)->info()])
				);
				const Point3d poszB(rayA.Intersects(planeB));
				if (poszB.z <= 0)
					continue;
				const Point2d posB((
					reinterpret_cast<const Point2d&>(fc->vertex(0)->point())+
					reinterpret_cast<const Point2d&>(fc->vertex(1)->point())+
					reinterpret_cast<const Point2d&>(fc->vertex(2)->point()))/3.f
				);
				const double dist(norm(posB-posA));
				depths.StoreTop<numPoints>(DepthDist(CLAMP((float)poszB.z,depthBounds.first,depthBounds.second), INVERT((float)dist)));
			} while (++cfc != done);
			ASSERT(depths.size() == numPoints);
			typedef Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<2> > FloatMap;
			FloatMap vecDists(&depths[0].score, numPoints);
			vecDists *= 1.f/vecDists.sum();
			FloatMap vecDepths(&depths[0].idx, numPoints);
			const float depth(vecDepths.dot(vecDists));
			mesh.vertices[idxFirstVertex+i] = image.camera.TransformPointI2C(Point3(posA, depth));
		}
	}
	mesh.faces.reserve(Mesh::FIndex(std::distance(delaunay.finite_faces_begin(),delaunay.finite_faces_end())));
	for (Delaunay::Face_iterator it=delaunay.faces_begin(); it!=delaunay.faces_end(); ++it) {
		const Delaunay::Face& face = *it;
		mesh.faces.emplace_back(face.vertex(2)->info(), face.vertex(1)->info(), face.vertex(0)->info());
	}
	return depthBounds;
}

#if 1 // New version (removed for testing)
// roughly estimate depth and normal maps by triangulating the sparse point-cloud
// and interpolating normal and depth for all pixels
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	Mesh mesh;
	Point2fArr projs;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, mesh, projs, bAddCorners));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	mesh.ComputeNormalVertices();
	depthMap.create(image.image.size());
	normalMap.create(image.image.size());
	if (!bAddCorners || bSparseOnly) {
		depthMap.memset(0);
		normalMap.memset(0);
	}
	if (bSparseOnly) {
		// just project sparse pointcloud onto depthmap
		FOREACH(i, mesh.vertices) {
			const Point2f& x(projs[i]);
			const Point2i ix(FLOOR2INT(x));
			const Depth z(mesh.vertices[i].z);
			const Normal& normal(mesh.vertexNormals[i]);
			for (const Point2i dx : {Point2i(0,0),Point2i(1,0),Point2i(0,1),Point2i(1,1)}) {
				const Point2i ax(ix + dx);
				if (!depthMap.isInside(ax))
					continue;
				depthMap(ax) = z;
				normalMap(ax) = normal;
			}
		}
	} else {
		// rasterize triangles onto depthmap
		struct RasterDepth : TRasterMeshBase<RasterDepth> {
			typedef TRasterMeshBase<RasterDepth> Base;
			using Base::Triangle;
			using Base::camera;
			using Base::depthMap;
			const Mesh::NormalArr& vertexNormals;
			NormalMap& normalMap;
			Mesh::Face face;
			RasterDepth(const Mesh::NormalArr& _vertexNormals, const Camera& _camera, DepthMap& _depthMap, NormalMap& _normalMap)
				: Base(_camera, _depthMap), vertexNormals(_vertexNormals), normalMap(_normalMap) {}
			inline void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
				const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
				const Depth z(ComputeDepth(t, pbary));
				ASSERT(z > Depth(0));
				depthMap(pt) = z;
				normalMap(pt) = normalized(
					vertexNormals[face[0]] * pbary[0]+
					vertexNormals[face[1]] * pbary[1]+
					vertexNormals[face[2]] * pbary[2]
				);
			}
		};
		RasterDepth rasterer {mesh.vertexNormals, camera, depthMap, normalMap};
		RasterDepth::Triangle triangle;
		RasterDepth::TriangleRasterizer triangleRasterizer(triangle, rasterer);
		for (const Mesh::Face& face : mesh.faces) {
			rasterer.face = face;
			triangle.ptc[0].z = mesh.vertices[face[0]].z;
			triangle.ptc[1].z = mesh.vertices[face[1]].z;
			triangle.ptc[2].z = mesh.vertices[face[2]].z;
			Image8U::RasterizeTriangleBary(
				projs[face[0]],
				projs[face[1]],
				projs[face[2]], triangleRasterizer);
		}
	}
	return true;
} // TriangulatePoints2DepthMap
// same as above, but does not estimate the normal-map
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	Mesh mesh;
	Point2fArr projs;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, mesh, projs, bAddCorners));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	depthMap.create(image.image.size());
	if (!bAddCorners || bSparseOnly)
		depthMap.memset(0);
	if (bSparseOnly) {
		// just project sparse pointcloud onto depthmap
		FOREACH(i, mesh.vertices) {
			const Point2f& x(projs[i]);
			const Point2i ix(FLOOR2INT(x));
			const Depth z(mesh.vertices[i].z);
			for (const Point2i dx : {Point2i(0,0),Point2i(1,0),Point2i(0,1),Point2i(1,1)}) {
				const Point2i ax(ix + dx);
				if (!depthMap.isInside(ax))
					continue;
				depthMap(ax) = z;
			}
		}
	} else {
		// rasterize triangles onto depthmap
		struct RasterDepth : TRasterMeshBase<RasterDepth> {
			typedef TRasterMeshBase<RasterDepth> Base;
			using Base::depthMap;
			RasterDepth(const Camera& _camera, DepthMap& _depthMap)
				: Base(_camera, _depthMap) {}
			inline void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
				const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
				const Depth z(ComputeDepth(t, pbary));
				ASSERT(z > Depth(0));
				depthMap(pt) = z;
			}
		};
		RasterDepth rasterer {camera, depthMap};
		RasterDepth::Triangle triangle;
		RasterDepth::TriangleRasterizer triangleRasterizer(triangle, rasterer);
		for (const Mesh::Face& face : mesh.faces) {
			triangle.ptc[0].z = mesh.vertices[face[0]].z;
			triangle.ptc[1].z = mesh.vertices[face[1]].z;
			triangle.ptc[2].z = mesh.vertices[face[2]].z;
			Image8U::RasterizeTriangleBary(
				projs[face[0]],
				projs[face[1]],
				projs[face[2]], triangleRasterizer);
		}
	}
	return true;
} // TriangulatePoints2DepthMap

#else
// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloudStreaming& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	Mesh mesh;
	Point2fArr projs;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, mesh, projs, bAddCorners));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	mesh.ComputeNormalVertices();
	depthMap.create(image.image.size());
	normalMap.create(image.image.size());
	if (!bAddCorners || bSparseOnly) {
		depthMap.memset(0);
		normalMap.memset(0);
	}
	if (bSparseOnly) {
		// just project sparse pointcloud onto depthmap
		FOREACH(i, mesh.vertices) {
			const Point2f& x(projs[i]);
			const Point2i ix(FLOOR2INT(x));
			const Depth z(mesh.vertices[i].z);
			const Normal& normal(mesh.vertexNormals[i]);
			for (const Point2i dx : {Point2i(0,0),Point2i(1,0),Point2i(0,1),Point2i(1,1)}) {
				const Point2i ax(ix + dx);
				if (!depthMap.isInside(ax))
					continue;
				depthMap.pix(ax) = z;
				normalMap.pix(ax) = normal;
			}
		}
	} else {
		// rasterize triangles onto depthmap
		struct RasterDepth : TRasterMeshBase<RasterDepth> {
			typedef TRasterMeshBase<RasterDepth> Base;
			using Base::Triangle;
			using Base::camera;
			using Base::depthMap;
			const Mesh::NormalArr& vertexNormals;
			NormalMap& normalMap;
			std::vector<uint8_t>& marks;
			Mesh::Face face;
			RasterDepth(const Mesh::NormalArr& _vertexNormals, const Camera& _camera, DepthMap& _depthMap, NormalMap& _normalMap, std::vector<uint8_t>& _marks)
				: Base(_camera, _depthMap), vertexNormals(_vertexNormals), normalMap(_normalMap), marks(_marks) {
			}
			inline void Raster(const ImageRef& pt, const Triangle& t, const Point3f& bary) {
				const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(t, bary));
				const Depth z(ComputeDepth(t, pbary));
				ASSERT(z > Depth(0));
				depthMap(pt) = z;
				normalMap(pt) = normalized(
					vertexNormals[face[0]] * pbary[0] +
					vertexNormals[face[1]] * pbary[1] +
					vertexNormals[face[2]] * pbary[2]
				);
			}
		};
		RasterDepth rasterer{ mesh.vertexNormals, camera, depthMap, normalMap, marks };
		RasterDepth::Triangle triangle;
		RasterDepth::TriangleRasterizer triangleRasterizer(triangle, rasterer);
		for (const Mesh::Face& face : mesh.faces) {
			rasterer.face = face;
			triangle.ptc[0].z = mesh.vertices[face[0]].z;
			triangle.ptc[1].z = mesh.vertices[face[1]].z;
			triangle.ptc[2].z = mesh.vertices[face[2]].z;
			Image8U::RasterizeTriangleBary(
				projs[face[0]],
				projs[face[1]],
				projs[face[2]], triangleRasterizer);
		}
	}
	return true;
} // TriangulatePoints2DepthMap
// same as above, but does not estimate the normal-map
bool MVS::TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloudStreaming& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly)
{
	ASSERT(image.pImageData != NULL);

	// triangulate in-view points
	Mesh mesh;
	Point2fArr projs;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(image, pointcloud, points, mesh, projs, bAddCorners));
	dMin = thDepth.first;
	dMax = thDepth.second;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	depthMap.create(image.image.size());
	if (!bAddCorners || bSparseOnly)
		depthMap.memset(0);
	if (bSparseOnly) {
		// just project sparse pointcloud onto depthmap
		FOREACH(i, mesh.vertices) {
			const Point2f& x(projs[i]);
			const Point2i ix(FLOOR2INT(x));
			const Depth z(mesh.vertices[i].z);
			for (const Point2i dx : {Point2i(0,0),Point2i(1,0),Point2i(0,1),Point2i(1,1)}) {
				const Point2i ax(ix + dx);
				if (!depthMap.isInside(ax))
					continue;
				depthMap.pix(ax) = z;
			}
		}
	} else {
		// rasterize triangles onto depthmap
		struct RasterDepth : TRasterMeshBase<RasterDepth> {
			typedef TRasterMeshBase<RasterDepth> Base;
			using Base::depthMap;
			RasterDepth(const Camera& _camera, DepthMap& _depthMap)
				: Base(_camera, _depthMap) {}
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(bary));
				const Depth z(ComputeDepth(pbary));
				ASSERT(z > Depth(0));
				depthMap.pix(pt) = z;
			}
		};
		RasterDepth rasterer = {camera, depthMap};
		for (const Mesh::Face& face : mesh.faces) {
			rasterer.ptc[0].z = mesh.vertices[face[0]].z;
			rasterer.ptc[1].z = mesh.vertices[face[1]].z;
			rasterer.ptc[2].z = mesh.vertices[face[2]].z;
			Image8U::RasterizeTriangleBary(
				projs[face[0]],
				projs[face[1]],
				projs[face[2]], rasterer);
		}
	}
	return true;
} // TriangulatePoints2DepthMap
#endif
/*----------------------------------------------------------------*/


namespace MVS {

// least squares refinement of the given plane to the 3D point set
// (return the number of iterations)
template <typename TYPE>
int OptimizePlane(TPlane<TYPE,3>& plane, const Eigen::Matrix<TYPE,3,1>* points, size_t size, int maxIters, TYPE threshold)
{
	typedef TPlane<TYPE,3> PLANE;
	typedef Eigen::Matrix<TYPE,3,1> POINT;
	ASSERT(size >= PLANE::numParams);
	struct OptimizationFunctor {
		const POINT* points;
		const size_t size;
		const RobustNorm::GemanMcClure<double> robust;
		// construct with the data points
		OptimizationFunctor(const POINT* _points, size_t _size, double _th)
			: points(_points), size(_size), robust(_th) { ASSERT(size < (size_t)std::numeric_limits<int>::max()); }
		static void Residuals(const double* x, int nPoints, const void* pData, double* fvec, double* fjac, int* /*info*/) {
			const OptimizationFunctor& data = *reinterpret_cast<const OptimizationFunctor*>(pData);
			ASSERT((size_t)nPoints == data.size && fvec != NULL && fjac == NULL);
			TPlane<double,3> plane; {
				Point3d N;
				plane.m_fD = x[0];
				Dir2Normal(reinterpret_cast<const Point2d&>(x[1]), N);
				plane.m_vN = N;
			}
			for (size_t i=0; i<data.size; ++i)
				fvec[i] = data.robust(plane.Distance(data.points[i].template cast<double>()));
		}
	} functor(points, size, threshold);
	double arrParams[PLANE::numParams]; {
		arrParams[0] = (double)plane.m_fD;
		const Point3d N(plane.m_vN.x(), plane.m_vN.y(), plane.m_vN.z());
		Normal2Dir(N, reinterpret_cast<Point2d&>(arrParams[1]));
	}
	lm_control_struct control = {1.e-6, 1.e-7, 1.e-8, 1.e-7, 100.0, maxIters}; // lm_control_float;
	lm_status_struct status;
	lmmin(PLANE::numParams, arrParams, (int)size, &functor, OptimizationFunctor::Residuals, &control, &status);
	switch (status.info) {
	//case 4:
	case 5:
	case 6:
	case 7:
	case 8:
	case 9:
	case 10:
	case 11:
	case 12:
		DEBUG_ULTIMATE("error: refine plane: %s", lm_infmsg[status.info]);
		return 0;
	}
	{
		Point3d N;
		plane.m_fD = (TYPE)arrParams[0];
		Dir2Normal(reinterpret_cast<const Point2d&>(arrParams[1]), N);
		plane.m_vN = Cast<TYPE>(N);
	}
	return status.nfev;
}

template <typename TYPE>
class TPlaneSolverAdaptor
{
public:
	enum { MINIMUM_SAMPLES = 3 };
	enum { MAX_MODELS = 1 };

	typedef TYPE Type;
	typedef TPoint3<TYPE> Point;
	typedef CLISTDEF0(Point) Points;
	typedef TPlane<TYPE,3> Model;
	typedef CLISTDEF0(Model) Models;

	TPlaneSolverAdaptor(const Points& points)
		: points_(points)
	{
	}
	TPlaneSolverAdaptor(const Points& points, float w, float h, float d)
		: points_(points)
	{
		// LogAlpha0 is used to make error data scale invariant
		// Ratio of containing diagonal image rectangle over image area
		const float D = SQRT(w*w + h*h + d*d); // diameter
		const float A = w*h*d+1.f; // volume
		logalpha0_ = LOG10(2.f*D/A*0.5f);
	}

	inline bool Fit(const std::vector<size_t>& samples, Models& models) const {
		ASSERT(samples.size() == MINIMUM_SAMPLES);
		Point points[MINIMUM_SAMPLES];
		for (size_t i=0; i<MINIMUM_SAMPLES; ++i)
			points[i] = points_[samples[i]];
		if (CheckCollinearity(points, 3))
			return false;
		models.Resize(1);
		models[0] = Model(points[0], points[1], points[2]);
		return true;
	}

	inline void EvaluateModel(const Model& model) {
		model2evaluate = model;
	}

	inline double Error(size_t sample) const {
		return SQUARE(model2evaluate.Distance(points_[sample]));
	}

	static double Error(const Model& plane, const Points& points) {
		double e(0);
		for (const Point& X: points)
			e += plane.DistanceAbs(X);
		return e/points.size();
	}

	inline size_t NumSamples() const { return static_cast<size_t>(points_.size()); }
	inline double logalpha0() const { return logalpha0_; }
	inline double multError() const { return 0.5; }

protected:
	const Points& points_; // Normalized input data
	double logalpha0_; // Alpha0 is used to make the error adaptive to the image size
	Model model2evaluate; // current model to be evaluated
};

// Robustly estimate the plane that fits best the given points
template <typename TYPE, typename Sampler, bool bFixThreshold>
unsigned TEstimatePlane(const CLISTDEF0(TPoint3<TYPE>)& points, TPlane<TYPE,3>& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	typedef TPlaneSolverAdaptor<TYPE> PlaneSolverAdaptor;

	plane.Invalidate();
	
	const unsigned nPoints = (unsigned)points.size();
	if (nPoints < PlaneSolverAdaptor::MINIMUM_SAMPLES) {
		ASSERT("too few points" == NULL);
		return 0;
	}

	// normalize points
	TMatrix<TYPE,4,4> H;
	typename PlaneSolverAdaptor::Points normPoints;
	NormalizePoints(points, normPoints, &H);

	// plane robust estimation
	std::vector<size_t> vec_inliers;
	Sampler sampler;
	if (bFixThreshold) {
		if (maxThreshold == 0)
			maxThreshold = 0.35/H(0,0);
		PlaneSolverAdaptor kernel(normPoints);
		RANSAC(kernel, sampler, vec_inliers, plane, maxThreshold*H(0,0), 0.99, maxIters);
		DEBUG_LEVEL(3, "Robust plane: %u/%u points", vec_inliers.size(), nPoints);
	} else {
		if (maxThreshold != DBL_MAX)
			maxThreshold *= H(0,0);
		PlaneSolverAdaptor kernel(normPoints, 1, 1, 1);
		const std::pair<double,double> ACRansacOut(ACRANSAC(kernel, sampler, vec_inliers, plane, maxThreshold, 0.99, maxIters));
		const double& thresholdSq = ACRansacOut.first;
		maxThreshold = SQRT(thresholdSq)/H(0,0);
		DEBUG_LEVEL(3, "Auto-robust plane: %u/%u points (%g threshold)", vec_inliers.size(), nPoints, maxThreshold);
	}
	unsigned inliers_count = (unsigned)vec_inliers.size();
	if (inliers_count < PlaneSolverAdaptor::MINIMUM_SAMPLES)
		return 0;

	// fit plane to all the inliers
	FitPlaneOnline<TYPE> fitPlane;
	for (unsigned i=0; i<inliers_count; ++i)
		fitPlane.Update(normPoints[vec_inliers[i]]);
	fitPlane.GetPlane(plane);

	// un-normalize plane
	plane.m_fD = (plane.m_fD+plane.m_vN.dot(typename PlaneSolverAdaptor::Model::POINT(H(0,3),H(1,3),H(2,3))))/H(0,0);

	// if a list of inliers is requested, copy it
	if (arrInliers) {
		inliers_count = 0;
		for (unsigned i=0; i<nPoints; ++i)
			if ((arrInliers[i] = (plane.DistanceAbs(points[i]) <= maxThreshold)) == true)
				++inliers_count;
	}
	return inliers_count;
} // TEstimatePlane

} // namespace MVS

// Robustly estimate the plane that fits best the given points
unsigned MVS::EstimatePlane(const Point3dArr& points, Planed& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSampler,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlane
// Robustly estimate the plane that fits best the given points, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneLockFirstPoint(const Point3dArr& points, Planed& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSamplerLockFirst,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneLockFirstPoint
// Robustly estimate the plane that fits best the given points using a known threshold
unsigned MVS::EstimatePlaneTh(const Point3dArr& points, Planed& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSampler,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneTh
// Robustly estimate the plane that fits best the given points using a known threshold, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneThLockFirstPoint(const Point3dArr& points, Planed& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<double,UniformSamplerLockFirst,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneThLockFirstPoint
// least squares refinement of the given plane to the 3D point set
int MVS::OptimizePlane(Planed& plane, const Eigen::Vector3d* points, size_t size, int maxIters, double threshold)
{
	return OptimizePlane<double>(plane, points, size, maxIters, threshold);
} // OptimizePlane
/*----------------------------------------------------------------*/

// Robustly estimate the plane that fits best the given points
unsigned MVS::EstimatePlane(const Point3fArr& points, Planef& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSampler,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlane
// Robustly estimate the plane that fits best the given points, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneLockFirstPoint(const Point3fArr& points, Planef& plane, double& maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSamplerLockFirst,false>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneLockFirstPoint
// Robustly estimate the plane that fits best the given points using a known threshold
unsigned MVS::EstimatePlaneTh(const Point3fArr& points, Planef& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSampler,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneTh
// Robustly estimate the plane that fits best the given points using a known threshold, making sure the first point is part of the solution (if any)
unsigned MVS::EstimatePlaneThLockFirstPoint(const Point3fArr& points, Planef& plane, double maxThreshold, bool arrInliers[], size_t maxIters)
{
	return TEstimatePlane<float,UniformSamplerLockFirst,true>(points, plane, maxThreshold, arrInliers, maxIters);
} // EstimatePlaneThLockFirstPoint
// least squares refinement of the given plane to the 3D point set
int MVS::OptimizePlane(Planef& plane, const Eigen::Vector3f* points, size_t size, int maxIters, float threshold)
{
	return OptimizePlane<float>(plane, points, size, maxIters, threshold);
} // OptimizePlane
/*----------------------------------------------------------------*/


// estimate the colors of the given dense point cloud
void MVS::EstimatePointColors(const ImageArr& images, PointCloudStreaming& pointcloud)
{
	TD_TIMER_START();

	const size_t cnt = pointcloud.NumPoints();
	pointcloud.ReserveColors(cnt);
	for (size_t i = 0; i < cnt; ++i) {
		const PointCloud::Point& point = (PointCloud::Point&)(pointcloud.PointStream()[i*3]);
		const size_t numViews = pointcloud.ViewsStreamSize(i);
		const auto* views = pointcloud.ViewsStream(i);

		// compute vertex color
		REAL bestDistance(FLT_MAX);
		const Image* pImageData(NULL);
		for (size_t j = 0; j < numViews; ++j) {
			const Image& imageData = images[views[j]];
			ASSERT(imageData.IsValid());
			if (imageData.image.empty())
				continue;
			// compute the distance from the 3D point to the image
			const REAL distance(imageData.camera.PointDepth(point));
			ASSERT(distance > 0);
			if (bestDistance > distance) {
				bestDistance = distance;
				pImageData = &imageData;
			}
		}
		if (pImageData == NULL) {
			// set a dummy color
			pointcloud.AddColor(Pixel8U::WHITE);
		} else {
			// get image color
			const Point2f proj(pImageData->camera.ProjectPointP(point));
			pointcloud.AddColor((pImageData->image.isInsideWithBorder<float, 1>(proj) ? pImageData->image.sample(proj) : Pixel8U::WHITE));
		}
	}

	DEBUG_ULTIMATE("Estimate dense point cloud colors: %u colors (%s)", pointcloud.NumPoints(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointColors
/*----------------------------------------------------------------*/

// estimates the normals through PCA over the K nearest neighbors
void MVS::EstimatePointNormals(const ImageArr& images, PointCloudStreaming& pointcloud, int numNeighbors /*K-nearest neighbors*/)
{
	TD_TIMER_START();

	typedef CGAL::Simple_cartesian<double> kernel_t;
	typedef kernel_t::Point_3 point_t;
	typedef kernel_t::Vector_3 vector_t;
	typedef std::pair<point_t,vector_t> PointVectorPair;
	// fetch the point set
	const size_t numPoints = pointcloud.NumPoints();
	boost::container::vector<PointVectorPair> pointvectors(numPoints, boost::container::default_init); // std::vector<PointVectorPair> pointvectors(numPoints);
	const float* pPoints = pointcloud.PointStream();
	for (size_t i = 0; i < numPoints; ++i, pPoints += 3) {
		pointvectors[i].first = point_t(pPoints[0], pPoints[1], pPoints[2]);
	}
	// estimates normals direction;
	// Note: pca_estimate_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.
	#if CGAL_VERSION_NR < 1041301000
	#if CGAL_VERSION_NR < 1040800000
	CGAL::pca_estimate_normals(
	#else
	CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
	#endif
		pointvectors.begin(), pointvectors.end(),
		CGAL::First_of_pair_property_map<PointVectorPair>(),
		CGAL::Second_of_pair_property_map<PointVectorPair>(),
		numNeighbors
	);
	#else
	#ifdef DPC_FASTER_NORMAL_ESTIMATION
	CGAL::pca_estimate_normals<CGAL::Parallel_tag>(
		pointvectors, numNeighbors,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
		.normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())
	);
#else	
	CGAL::pca_estimate_normals<CGAL::Sequential_tag>(
			pointvectors, numNeighbors,
			CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
			.normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())
		);
#endif
	#endif
	// store the point normals
	pPoints = pointcloud.PointStream();

	pointcloud.normalsXYZ.resize(3 * numPoints);
#pragma omp parallel for num_threads(omp_get_max_threads())
	for (int64_t i = 0; i < (int64_t) numPoints; ++i) {
		const PointCloud::Point& point = (PointCloud::Point&)(pPoints[i*3]);
		const auto* pViews = pointcloud.ViewsStream(i);
		Normal normal = Normal((float) pointvectors[i].second.x(), (float) pointvectors[i].second.y(), (float) pointvectors[i].second.z());
		// correct normal orientation
		//ASSERT(!views.IsEmpty());
		const Image& imageData = images[*pViews];
		if (normal.dot(Cast<float>(imageData.camera.C)-point) < 0)
			normal = -normal;
		// You can't AddNormal.  You have to use stream placement.
		((Normal&) *(pointcloud.NormalStream()+i*3)) = normal;
	}

	DEBUG_ULTIMATE("Estimate dense point cloud normals: %u normals (%s)", pointcloud.NumPoints(), TD_TIMER_GET_FMT().c_str());
} // EstimatePointNormals
/*----------------------------------------------------------------*/

bool MVS::EstimateNormalMap(const Matrix3x3f& K, const DepthMap& depthMap, NormalMap& normalMap)
{
	normalMap.create(depthMap.size());
	struct Tool {
		static bool IsDepthValid(Depth d, Depth nd) {
			return nd > 0 && IsDepthSimilar(d, nd, Depth(0.03f));
		}
		// computes depth gradient (first derivative) at current pixel
		static bool DepthGradient(const DepthMap& depthMap, const ImageRef& ir, Point3f& ws) {
			float& w  = ws[0];
			float& wx = ws[1];
			float& wy = ws[2];
			w = depthMap.pix(ir);
			if (w <= 0)
				return false;
			// loop over neighborhood and finding least squares plane,
			// the coefficients of which give gradient of depth
			int whxx(0), whxy(0), whyy(0);
			float wgx(0), wgy(0);
			const int Radius(1);
			int n(0);
			for (int y = -Radius; y <= Radius; ++y) {
				for (int x = -Radius; x <= Radius; ++x) {
					if (x == 0 && y == 0)
						continue;
					const ImageRef pt(ir.x+x, ir.y+y);
					if (!depthMap.isInside(pt))
						continue;
					const float wi(depthMap.pix(pt));
					if (!IsDepthValid(w, wi))
						continue;
					whxx += x*x; whxy += x*y; whyy += y*y;
					wgx += (wi - w)*x; wgy += (wi - w)*y;
					++n;
				}
			}
			if (n < 3)
				return false;
			// solve 2x2 system, generated from depth gradient
			const int det(whxx*whyy - whxy*whxy);
			if (det == 0)
				return false;
			const float invDet(1.f/float(det));
			wx = (float( whyy)*wgx - float(whxy)*wgy)*invDet;
			wy = (float(-whxy)*wgx + float(whxx)*wgy)*invDet;
			return true;
		}
		// computes normal to the surface given the depth and its gradient
		static Normal ComputeNormal(const Matrix3x3f& K, int x, int y, Depth d, Depth dx, Depth dy) {
			ASSERT(ISZERO(K(0,1)));
			return normalized(Normal(
				K(0,0)*dx,
				K(1,1)*dy,
				(K(0,2)-float(x))*dx+(K(1,2)-float(y))*dy-d
			));
		}
	};
	for (int r=0; r<normalMap.rows; ++r) {
		for (int c=0; c<normalMap.cols; ++c) {
			#if 0
			const Depth d(depthMap(r,c));
			if (d <= 0) {
				normalMap(r,c) = Normal::ZERO;
				continue;
			}
			Depth dl, du;
			if (depthMap.isInside(ImageRef(c-1,r-1)) && Tool::IsDepthValid(d, dl=depthMap(r,c-1)) &&  Tool::IsDepthValid(d, du=depthMap(r-1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, du-d, dl-d);
			else
			if (depthMap.isInside(ImageRef(c+1,r-1)) && Tool::IsDepthValid(d, dl=depthMap(r,c+1)) &&  Tool::IsDepthValid(d, du=depthMap(r-1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, du-d, d-dl);
			else
			if (depthMap.isInside(ImageRef(c+1,r+1)) && Tool::IsDepthValid(d, dl=depthMap(r,c+1)) &&  Tool::IsDepthValid(d, du=depthMap(r+1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, d-du, d-dl);
			else
			if (depthMap.isInside(ImageRef(c-1,r+1)) && Tool::IsDepthValid(d, dl=depthMap(r,c-1)) &&  Tool::IsDepthValid(d, du=depthMap(r+1,c)))
				normalMap(r,c) = Tool::ComputeNormal(K, c, r, d, d-du, dl-d);
			else
				normalMap(r,c) = Normal(0,0,-1);
			#else
			// calculates depth gradient at x
			Normal& n = normalMap(r,c);
			if (Tool::DepthGradient(depthMap, ImageRef(c,r), n))
				n = Tool::ComputeNormal(K, c, r, n.x, n.y, n.z);
			else
				n = Normal::ZERO;
			#endif
			ASSERT(normalMap(r,c).dot(K.inv()*Point3f(float(c),float(r),1.f)) <= 0);
		}
	}
	return true;
} // EstimateNormalMap
/*----------------------------------------------------------------*/


// save the depth map in our .dmap file format
bool MVS::SaveDepthMap(const String& fileName, const DepthMap& depthMap)
{
	ASSERT(!depthMap.empty());
	return SerializeSave(depthMap, fileName, ARCHIVE_BINARY);
} // SaveDepthMap
/*----------------------------------------------------------------*/
// load the depth map from our .dmap file format
bool MVS::LoadDepthMap(const String& fileName, DepthMap& depthMap)
{
	return SerializeLoad(depthMap, fileName, ARCHIVE_BINARY);
} // LoadDepthMap
/*----------------------------------------------------------------*/

// save the normal map in our .nmap file format
bool MVS::SaveNormalMap(const String& fileName, const NormalMap& normalMap)
{
	ASSERT(!normalMap.empty());
	return SerializeSave(normalMap, fileName, ARCHIVE_DEFAULT);
} // SaveNormalMap
/*----------------------------------------------------------------*/
// load the normal map from our .nmap file format
bool MVS::LoadNormalMap(const String& fileName, NormalMap& normalMap)
{
	return SerializeLoad(normalMap, fileName, ARCHIVE_DEFAULT);
} // LoadNormalMap
/*----------------------------------------------------------------*/

// save the confidence map in our .cmap file format
bool MVS::SaveConfidenceMap(const String& fileName, const ConfidenceMap& confMap)
{
	ASSERT(!confMap.empty());
	return SerializeSave(confMap, fileName, ARCHIVE_BINARY);
} // SaveConfidenceMap
/*----------------------------------------------------------------*/
// load the confidence map from our .cmap file format
bool MVS::LoadConfidenceMap(const String& fileName, ConfidenceMap& confMap)
{
	return SerializeLoad(confMap, fileName, ARCHIVE_BINARY);
} // LoadConfidenceMap
/*----------------------------------------------------------------*/



// export depth map as an image (dark - far depth, light - close depth)
Image8U3 MVS::DepthMap2Image(const DepthMap& depthMap, Depth minDepth, Depth maxDepth)
{
	ASSERT(!depthMap.empty());
	// find min and max values
	if (minDepth == FLT_MAX && maxDepth == 0) {
		cList<Depth,Depth,0> depths(0, depthMap.area());
		for (int i=depthMap.area(); --i >= 0; ) {
			const Depth depth = depthMap[i];
			ASSERT(depth == 0 || depth > 0);
			if (depth > 0)
				depths.Insert(depth);
		}
		if (!depths.empty()) {
			const std::pair<Depth,Depth> th(ComputeX84Threshold<Depth,Depth>(depths.data(), depths.size()));
			const std::pair<Depth,Depth> mm(depths.GetMinMax());
			maxDepth = MINF(th.first+th.second, mm.second);
			minDepth = MAXF(th.first-th.second, mm.first);
		}
		DEBUG_ULTIMATE("\tdepth range: [%g, %g]", minDepth, maxDepth);
	}
	const Depth sclDepth(Depth(1)/(maxDepth - minDepth));
	// create color image
	Image8U3 img(depthMap.size());
	for (int i=depthMap.area(); --i >= 0; ) {
		const Depth depth = depthMap[i];
		img[i] = (depth > 0 ? Pixel8U::gray2color(CLAMP((maxDepth-depth)*sclDepth, Depth(0), Depth(1))) : Pixel8U::BLACK);
	}
	return img;
} // DepthMap2Image
bool MVS::ExportDepthMap(const String& fileName, const DepthMap& depthMap, Depth minDepth, Depth maxDepth)
{
	if (depthMap.empty())
		return false;
	return DepthMap2Image(depthMap, minDepth, maxDepth).Save(fileName);
} // ExportDepthMap
/*----------------------------------------------------------------*/

// export normal map as an image
bool MVS::ExportNormalMap(const String& fileName, const NormalMap& normalMap)
{
	if (normalMap.empty())
		return false;
	Image8U3 img(normalMap.size());
	for (int i=normalMap.area(); --i >= 0; ) {
		img[i] = [](const Normal& n) {
			return ISZERO(n) ?
				Image8U3::Type::BLACK :
				Image8U3::Type(
					CLAMP(ROUND2INT((1.f-n.x)*127.5f), 0, 255),
					CLAMP(ROUND2INT((1.f-n.y)*127.5f), 0, 255),
					CLAMP(ROUND2INT(    -n.z *255.0f), 0, 255)
				);
		} (normalMap[i]);
	}
	return img.Save(fileName);
} // ExportNormalMap
/*----------------------------------------------------------------*/

// export confidence map as an image (dark - low confidence, light - high confidence)
bool MVS::ExportConfidenceMap(const String& fileName, const ConfidenceMap& confMap)
{
	// find min and max values
	FloatArr confs(0, confMap.area());
	for (int i=confMap.area(); --i >= 0; ) {
		const float conf = confMap[i];
		ASSERT(conf == 0 || conf > 0);
		if (conf > 0)
			confs.Insert(conf);
	}
	if (confs.IsEmpty())
		return false;
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(confs.Begin(), confs.GetSize()));
	float minConf = th.first-th.second;
	float maxConf = th.first+th.second;
	if (minConf < 0.1f)
		minConf = 0.1f;
	if (maxConf < 0.1f)
		maxConf = 30.f;
	DEBUG_ULTIMATE("\tconfidence range: [%g, %g]", minConf, maxConf);
	const float deltaConf = maxConf - minConf;
	// save image
	Image8U img(confMap.size());
	for (int i=confMap.area(); --i >= 0; ) {
		const float conf = confMap[i];
		img[i] = (conf > 0 ? (uint8_t)CLAMP((conf-minConf)*255.f/deltaConf, 0.f, 255.f) : 0);
	}
	return img.Save(fileName);
} // ExportConfidenceMap
/*----------------------------------------------------------------*/

// export point cloud
bool MVS::ExportPointCloud(const String& fileName, const Image& imageData, const DepthMap& depthMap, const NormalMap& normalMap)
{
	ASSERT(!depthMap.empty());
	const Camera& P0 = imageData.camera;
	if (normalMap.empty()) {
		// vertex definition
		struct Vertex {
			float x,y,z;
			uint8_t r,g,b;
		};
		// list of property information for a vertex
		static PLY::PlyProperty vert_props[] = {
			{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
			{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
			{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
			{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
			{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
			{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
		};
		// list of the kinds of elements in the PLY
		static const char* elem_names[] = {
			"vertex"
		};

		// create PLY object
		ASSERT(!fileName.IsEmpty());
		Util::ensureFolder(fileName);
		const size_t bufferSize = depthMap.area()*(8*3/*pos*/+3*3/*color*/+7/*space*/+2/*eol*/) + 2048/*extra size*/;
		PLY ply;
		if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, bufferSize))
			return false;

		// describe what properties go into the vertex elements
		ply.describe_property("vertex", 6, vert_props);

		// export the array of 3D points
		Vertex vertex;
		const Point2f scaleImage(static_cast<float>(depthMap.cols)/imageData.image.cols, static_cast<float>(depthMap.rows)/imageData.image.rows);
		for (int j=0; j<depthMap.rows; ++j) {
			for (int i=0; i<depthMap.cols; ++i) {
				const Depth& depth = depthMap(j,i);
				ASSERT(depth >= 0);
				if (depth <= 0)
					continue;
				const Point3f X(P0.TransformPointI2W(Point3(i,j,depth)));
				vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
				const Pixel8U c(imageData.image.empty() ? Pixel8U::WHITE : imageData.image(ROUND2INT(scaleImage.y*j),ROUND2INT(scaleImage.x*i)));
				vertex.r = c.r; vertex.g = c.g; vertex.b = c.b;
				ply.put_element(&vertex);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// write to file
		if (!ply.header_complete())
			return false;
	} else {
		// vertex definition
		struct Vertex {
			float x,y,z;
			float nx,ny,nz;
			uint8_t r,g,b;
		};
		// list of property information for a vertex
		static PLY::PlyProperty vert_props[] = {
			{"x", PLY::Float32, PLY::Float32, offsetof(Vertex,x), 0, 0, 0, 0},
			{"y", PLY::Float32, PLY::Float32, offsetof(Vertex,y), 0, 0, 0, 0},
			{"z", PLY::Float32, PLY::Float32, offsetof(Vertex,z), 0, 0, 0, 0},
			{"nx", PLY::Float32, PLY::Float32, offsetof(Vertex,nx), 0, 0, 0, 0},
			{"ny", PLY::Float32, PLY::Float32, offsetof(Vertex,ny), 0, 0, 0, 0},
			{"nz", PLY::Float32, PLY::Float32, offsetof(Vertex,nz), 0, 0, 0, 0},
			{"red", PLY::Uint8, PLY::Uint8, offsetof(Vertex,r), 0, 0, 0, 0},
			{"green", PLY::Uint8, PLY::Uint8, offsetof(Vertex,g), 0, 0, 0, 0},
			{"blue", PLY::Uint8, PLY::Uint8, offsetof(Vertex,b), 0, 0, 0, 0},
		};
		// list of the kinds of elements in the PLY
		static const char* elem_names[] = {
			"vertex"
		};

		// create PLY object
		ASSERT(!fileName.IsEmpty());
		Util::ensureFolder(fileName);
		const size_t bufferSize = depthMap.area()*(8*3/*pos*/+8*3/*normal*/+3*3/*color*/+8/*space*/+2/*eol*/) + 2048/*extra size*/;
		PLY ply;
		if (!ply.write(fileName, 1, elem_names, PLY::BINARY_LE, bufferSize))
			return false;

		// describe what properties go into the vertex elements
		ply.describe_property("vertex", 9, vert_props);

		// export the array of 3D points
		Vertex vertex;
		for (int j=0; j<depthMap.rows; ++j) {
			for (int i=0; i<depthMap.cols; ++i) {
				const Depth& depth = depthMap(j,i);
				ASSERT(depth >= 0);
				if (depth <= 0)
					continue;
				const Point3f X(P0.TransformPointI2W(Point3(i,j,depth)));
				vertex.x = X.x; vertex.y = X.y; vertex.z = X.z;
				const Point3f N(P0.R.t() * Cast<REAL>(normalMap(j,i)));
				vertex.nx = N.x; vertex.ny = N.y; vertex.nz = N.z;
				const Pixel8U c(imageData.image.empty() ? Pixel8U::WHITE : imageData.image(j, i));
				vertex.r = c.r; vertex.g = c.g; vertex.b = c.b;
				ply.put_element(&vertex);
			}
		}
		if (ply.get_current_element_count() == 0)
			return false;

		// write to file
		if (!ply.header_complete())
			return false;
	}
	return true;
} // ExportPointCloud
/*----------------------------------------------------------------*/

#if 1 // JPB WIP BUG

#pragma optimize("", on)
#if 0
#define DBG_BREAK(msg)                              \
    do {                                              \
      OutputDebugStringA(msg "\n");                   \
      __debugbreak();                                 \
    } while (0)

#define DBG_BREAKF(fmt, ...)                        \
    do {                                              \
      char _dbgBuf[512];                              \
      _snprintf_s(_dbgBuf, sizeof(_dbgBuf), _TRUNCATE,\
                  fmt "\n", __VA_ARGS__);             \
      OutputDebugStringA(_dbgBuf);                    \
      __debugbreak();                                 \
    } while (0)

#define DBG_CHECK(cond, msg)                        \
    do { if (!(cond)) DBG_BREAK(msg); } while (0)

#define DBG_CHECKF(cond, fmt, ...)                  \
    do { if (!(cond)) DBG_BREAKF(fmt, __VA_ARGS__); } while (0)
#else
#define DBG_BREAK(msg)          do {} while (0)
#define DBG_BREAKF(fmt, ...)    do {} while (0)
#define DBG_CHECK(cond, msg)    do {} while (0)
#define DBG_CHECKF(cond, fmt, ...) do {} while (0)
#endif


static inline void* PtrAdvance(void*& p, size_t bytes) {
	void* cur = p;
	p = (char*)p + bytes;
	return cur;
}

static inline bool AdvanceChecked(const uint8_t*& p, const uint8_t* end, size_t bytes, const uint8_t** outBegin = nullptr) {
	if ((size_t)(end - p) < bytes) return false;
	if (outBegin) *outBegin = p;
	p += bytes;
	return true;
}

static inline bool ReadU16(const uint8_t*& p, const uint8_t* end, uint16_t& v) {
	const uint8_t* b = nullptr;
	if (!AdvanceChecked(p, end, sizeof(uint16_t), &b)) return false;
	memcpy(&v, b, sizeof(uint16_t));
	return true;
}

static inline bool ReadU32(const uint8_t*& p, const uint8_t* end, uint32_t& v) {
	const uint8_t* b = nullptr;
	if (!AdvanceChecked(p, end, sizeof(uint32_t), &b)) return false;
	memcpy(&v, b, sizeof(uint32_t));
	return true;
}

bool MVS::ExportDepthDataRaw(
	const String& fileName,
	const String& imageFileName,
	const IIndexArr& IDs,
	const cv::Size& imageSize,
	const KMatrix& K,
	const RMatrix& R,
	const CMatrix& C,
	Depth dMin,
	Depth dMax,
	const DepthMap& depthMap,
	const NormalMap& normalMap,
	const ConfidenceMap& confMap,
	const ViewsMap& viewsMap)
{
	DBG_CHECK(IDs.size() > 1 && IDs.size() < 256, "Export: invalid ID count");
	DBG_CHECK(!depthMap.empty(), "Export: depthMap empty");

	const size_t depthArea = depthMap.area();
	DBG_CHECK(depthArea > 0, "Export: zero depth area");

	if (!normalMap.empty())
		DBG_CHECK(normalMap.area() == depthArea, "Export: normalMap area mismatch");
	if (!confMap.empty())
		DBG_CHECK(confMap.area() == depthArea, "Export: confMap area mismatch");
	if (!viewsMap.empty())
		DBG_CHECK(viewsMap.area() == depthArea, "Export: viewsMap area mismatch");

	const bool hasNormal = !normalMap.empty();
	const bool hasConf = !confMap.empty();
	const bool hasViews = !viewsMap.empty();

	const String relImageName(
		MAKE_PATH_REL(
			Util::getFullPath(Util::getFilePath(fileName)),
			Util::getFullPath(imageFileName)));

	const uint16_t nameLen = (uint16_t)relImageName.length();

	size_t fileSize =
		sizeof(HeaderDepthDataRaw) +
		sizeof(uint16_t) + nameLen +
		sizeof(uint32_t) + IDs.size() * sizeof(IIndex) +
		sizeof(REAL) * (9 + 9 + 3) +
		sizeof(float) * depthArea;

	if (hasNormal) fileSize += sizeof(float) * 3 * depthArea;
	if (hasConf)   fileSize += sizeof(float) * depthArea;
	if (hasViews)  fileSize += sizeof(uint8_t) * 4 * depthArea;

	HANDLE hFile = CreateFileA(
		fileName.c_str(),
		GENERIC_READ | GENERIC_WRITE,
		0,
		nullptr,
		CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL,
		nullptr);

	DBG_CHECKF(hFile != INVALID_HANDLE_VALUE,
		"Export: CreateFile failed (err=%lu)", GetLastError());

	LARGE_INTEGER li;
	li.QuadPart = (LONGLONG)fileSize;
	DBG_CHECKF(SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN),
		"Export: SetFilePointerEx failed (err=%lu)", GetLastError());
	DBG_CHECKF(SetEndOfFile(hFile),
		"Export: SetEndOfFile failed (err=%lu)", GetLastError());

	HANDLE hMap = CreateFileMappingA(
		hFile, nullptr, PAGE_READWRITE,
		(DWORD)(fileSize >> 32),
		(DWORD)(fileSize & 0xffffffff),
		nullptr);

	DBG_CHECKF(hMap != nullptr,
		"Export: CreateFileMapping failed (err=%lu)", GetLastError());

	uint8_t* base = (uint8_t*)MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, fileSize);
	DBG_CHECKF(base != nullptr,
		"Export: MapViewOfFile failed (err=%lu)", GetLastError());

	uint8_t* p = base;

	HeaderDepthDataRaw header;
	header.name = HeaderDepthDataRaw::HeaderDepthDataRawName();
	header.type = HeaderDepthDataRaw::HAS_DEPTH |
		(hasNormal ? HeaderDepthDataRaw::HAS_NORMAL : 0) |
		(hasConf ? HeaderDepthDataRaw::HAS_CONF : 0) |
		(hasViews ? HeaderDepthDataRaw::HAS_VIEWS : 0);
	header.imageWidth = imageSize.width;
	header.imageHeight = imageSize.height;
	header.depthWidth = depthMap.cols;
	header.depthHeight = depthMap.rows;
	header.dMin = dMin;
	header.dMax = dMax;

	memcpy(p, &header, sizeof(header)); p += sizeof(header);
	memcpy(p, &nameLen, sizeof(nameLen)); p += sizeof(nameLen);
	memcpy(p, relImageName.data(), nameLen); p += nameLen;

	const uint32_t nIDs = (uint32_t)IDs.size();
	memcpy(p, &nIDs, sizeof(nIDs)); p += sizeof(nIDs);
	memcpy(p, IDs.data(), sizeof(IIndex) * nIDs); p += sizeof(IIndex) * nIDs;

	memcpy(p, K.val, sizeof(REAL) * 9); p += sizeof(REAL) * 9;
	memcpy(p, R.val, sizeof(REAL) * 9); p += sizeof(REAL) * 9;
	memcpy(p, C.ptr(), sizeof(REAL) * 3); p += sizeof(REAL) * 3;

	memcpy(p, depthMap.getData(), sizeof(float) * depthArea);
	p += sizeof(float) * depthArea;

	if (hasNormal) {
		memcpy(p, normalMap.getData(), sizeof(float) * 3 * depthArea);
		p += sizeof(float) * 3 * depthArea;
	}
	if (hasConf) {
		memcpy(p, confMap.getData(), sizeof(float) * depthArea);
		p += sizeof(float) * depthArea;
	}
	if (hasViews) {
		memcpy(p, viewsMap.getData(), sizeof(uint8_t) * 4 * depthArea);
		p += sizeof(uint8_t) * 4 * depthArea;
	}

	DBG_CHECKF(p == base + fileSize,
		"Export: size mismatch (written=%zu expected=%zu)",
		size_t(p - base), fileSize);

	UnmapViewOfFile(base);
	CloseHandle(hMap);
	CloseHandle(hFile);
	return true;
}

bool MVS::ImportDepthDataRaw(
	const String& fileName,
	String& imageFileName,
	IIndexArr& IDs,
	cv::Size& imageSize,
	KMatrix& K,
	RMatrix& R,
	CMatrix& C,
	Depth& dMin,
	Depth& dMax,
	DepthMap& depthMap,
	NormalMap& normalMap,
	ConfidenceMap& confMap,
	ViewsMap& viewsMap,
	unsigned flags)
{
	HANDLE hFile = CreateFileA(
		fileName.c_str(),
		GENERIC_READ,
		FILE_SHARE_READ,
		nullptr,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		nullptr);

	DBG_CHECKF(hFile != INVALID_HANDLE_VALUE,
		"Import: CreateFile failed (err=%lu)", GetLastError());

	LARGE_INTEGER li;
	DBG_CHECKF(GetFileSizeEx(hFile, &li),
		"Import: GetFileSizeEx failed (err=%lu)", GetLastError());

	const size_t fileSize = (size_t)li.QuadPart;
	DBG_CHECK(fileSize >= sizeof(HeaderDepthDataRaw),
		"Import: file too small");

	HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
	DBG_CHECKF(hMap != nullptr,
		"Import: CreateFileMapping failed (err=%lu)", GetLastError());

	const uint8_t* base =
		(const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
	DBG_CHECKF(base != nullptr,
		"Import: MapViewOfFile failed (err=%lu)", GetLastError());

	const uint8_t* p = base;
	const uint8_t* end = base + fileSize;

	HeaderDepthDataRaw header;
	memcpy(&header, p, sizeof(header)); p += sizeof(header);

	uint16_t rawName;
	memcpy(&rawName, base, sizeof(uint16_t));

	DBG_CHECK(header.name == HeaderDepthDataRaw::HeaderDepthDataRawName(),
		"Import: bad magic");
	DBG_CHECK(header.type & HeaderDepthDataRaw::HAS_DEPTH,
		"Import: HAS_DEPTH not set");

	imageSize.width = header.imageWidth;
	imageSize.height = header.imageHeight;
	dMin = header.dMin;
	dMax = header.dMax;

	uint16_t nameLen;
	memcpy(&nameLen, p, sizeof(nameLen)); p += sizeof(nameLen);
	DBG_CHECK(nameLen > 0 && nameLen < 4096, "Import: bad filename length");

	imageFileName.assign((const char*)p, nameLen);
	p += nameLen;

	uint32_t nIDs;
	memcpy(&nIDs, p, sizeof(nIDs)); p += sizeof(nIDs);
	DBG_CHECK(nIDs > 0 && nIDs < 256, "Import: bad nIDs");

	IDs.resize(nIDs);
	memcpy(IDs.data(), p, sizeof(IIndex) * nIDs);
	p += sizeof(IIndex) * nIDs;

	memcpy(K.val, p, sizeof(REAL) * 9); p += sizeof(REAL) * 9;
	memcpy(R.val, p, sizeof(REAL) * 9); p += sizeof(REAL) * 9;
	memcpy(C.ptr(), p, sizeof(REAL) * 3); p += sizeof(REAL) * 3;

	const size_t depthArea =
		(size_t)header.depthWidth * (size_t)header.depthHeight;
	const size_t depthBytes = sizeof(float) * depthArea;

	DBG_CHECK((size_t)(end - p) >= depthBytes, "Import: truncated depth");

	if (flags & HeaderDepthDataRaw::HAS_DEPTH) {
		depthMap.create(header.depthHeight, header.depthWidth);
		memcpy(depthMap.getData(), p, depthBytes);
	}
	p += depthBytes;

	if (header.type & HeaderDepthDataRaw::HAS_NORMAL) {
		const size_t bytes = sizeof(float) * 3 * depthArea;
		DBG_CHECK((size_t)(end - p) >= bytes, "Import: truncated normal");
		if (flags & HeaderDepthDataRaw::HAS_NORMAL) {
			normalMap.create(header.depthHeight, header.depthWidth);
			memcpy(normalMap.getData(), p, bytes);
		}
		p += bytes;
	}

	if (header.type & HeaderDepthDataRaw::HAS_CONF) {
		const size_t bytes = sizeof(float) * depthArea;
		DBG_CHECK((size_t)(end - p) >= bytes, "Import: truncated conf");
		if (flags & HeaderDepthDataRaw::HAS_CONF) {
			confMap.create(header.depthHeight, header.depthWidth);
			memcpy(confMap.getData(), p, bytes);
		}
		p += bytes;
	}

	if (header.type & HeaderDepthDataRaw::HAS_VIEWS) {
		const size_t bytes = sizeof(uint8_t) * 4 * depthArea;
		DBG_CHECK((size_t)(end - p) >= bytes, "Import: truncated views");
		if (flags & HeaderDepthDataRaw::HAS_VIEWS) {
			viewsMap.create(header.depthHeight, header.depthWidth);
			memcpy(viewsMap.getData(), p, bytes);
		}
		p += bytes;
	}

	DBG_CHECKF(p == end,
		"Import: trailing or missing data (%zu bytes)",
		size_t(end - p));

	UnmapViewOfFile(base);
	CloseHandle(hMap);
	CloseHandle(hFile);
	return true;
}
#pragma optimize("", on)

#else
//  - IDs are the reference view ID and neighbor view IDs used to estimate the depth-map (global ID)
bool MVS::ExportDepthDataRaw(const String& fileName, const String& imageFileName,
	const IIndexArr& IDs, const cv::Size& imageSize,
	const KMatrix& K, const RMatrix& R, const CMatrix& C,
	Depth dMin, Depth dMax,
	const DepthMap& depthMap, const NormalMap& normalMap, const ConfidenceMap& confMap, const ViewsMap& viewsMap)
{
	ASSERT(IDs.size() > 1 && IDs.size() < 256);
	ASSERT(!depthMap.empty());
	ASSERT(confMap.empty() || depthMap.size() == confMap.size());
	ASSERT(viewsMap.empty() || depthMap.size() == viewsMap.size());
	ASSERT(depthMap.width() <= imageSize.width && depthMap.height() <= imageSize.height);

	FILE* f = fopen(fileName, "wb");
	if (f == NULL) {
		DEBUG("error: opening file '%s' for writing depth-data", fileName.c_str());
		return false;
	}

	constexpr size_t bufferSize = 65536;
	__declspec(align(32)) char buffer[bufferSize]; // Can't be static, mt aware.
	std::setvbuf(f, buffer, _IOFBF, bufferSize);

	// write header
	HeaderDepthDataRaw header;
	header.name = HeaderDepthDataRaw::HeaderDepthDataRawName();
	header.type = HeaderDepthDataRaw::HAS_DEPTH;
	header.imageWidth = (uint32_t)imageSize.width;
	header.imageHeight = (uint32_t)imageSize.height;
	header.depthWidth = (uint32_t)depthMap.cols;
	header.depthHeight = (uint32_t)depthMap.rows;
	header.dMin = dMin;
	header.dMax = dMax;
	if (!normalMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_NORMAL;
	if (!confMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_CONF;
	if (!viewsMap.empty())
		header.type |= HeaderDepthDataRaw::HAS_VIEWS;
	_fwrite_nolock(&header, sizeof(HeaderDepthDataRaw), 1, f);

	// write image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	const String FileName(MAKE_PATH_REL(Util::getFullPath(Util::getFilePath(fileName)), Util::getFullPath(imageFileName)));
	const uint16_t nFileNameSize((uint16_t)FileName.length());
	_fwrite_nolock(&nFileNameSize, sizeof(uint16_t), 1, f);
	_fwrite_nolock(FileName.c_str(), sizeof(char), nFileNameSize, f);

	// write neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	const uint32_t nIDs(IDs.size());
	_fwrite_nolock(&nIDs, sizeof(IIndex), 1, f);
	_fwrite_nolock(IDs.data(), sizeof(IIndex), nIDs, f);

	// write pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	_fwrite_nolock(K.val, sizeof(REAL), 9, f);
	_fwrite_nolock(R.val, sizeof(REAL), 9, f);
	_fwrite_nolock(C.ptr(), sizeof(REAL), 3, f);

	// write depth-map
	_fwrite_nolock(depthMap.getData(), sizeof(float) * depthMap.area(), 1, f);

	// write normal-map
	if ((header.type & HeaderDepthDataRaw::HAS_NORMAL) != 0)
		_fwrite_nolock(normalMap.getData(), sizeof(float)*3 * normalMap.area(), 1, f);

	// write confidence-map
	if ((header.type & HeaderDepthDataRaw::HAS_CONF) != 0)
		_fwrite_nolock(confMap.getData(), sizeof(float) * confMap.area(), 1, f);

	// write views-map
	if ((header.type & HeaderDepthDataRaw::HAS_VIEWS) != 0)
		_fwrite_nolock(viewsMap.getData(), sizeof(uint8_t)*4 * viewsMap.area(), 1, f);

	const bool bRet(ferror(f) == 0);
	_fclose_nolock(f);
	return bRet;
} // ExportDepthDataRaw

bool MVS::ImportDepthDataRaw(const String& fileName, String& imageFileName,
	IIndexArr& IDs, cv::Size& imageSize,
	KMatrix& K, RMatrix& R, CMatrix& C,
	Depth& dMin, Depth& dMax,
	DepthMap& depthMap, NormalMap& normalMap, ConfidenceMap& confMap, ViewsMap& viewsMap, unsigned flags)
{
	FILE* f = fopen(fileName, "rb");
	if (f == NULL) {
		DEBUG("error: opening file '%s' for reading depth-data", fileName.c_str());
		return false;
	}

	constexpr size_t bufferSize = 65536;
	__declspec(align(32)) char buffer[bufferSize]; // Can't be static, mt aware.

	std::setvbuf(f, buffer, _IOFBF, bufferSize);

	// read header
	HeaderDepthDataRaw header;
	if (_fread_nolock(&header, sizeof(HeaderDepthDataRaw), 1, f) != 1 ||
		header.name != HeaderDepthDataRaw::HeaderDepthDataRawName() ||
		(header.type & HeaderDepthDataRaw::HAS_DEPTH) == 0 ||
		header.depthWidth <= 0 || header.depthHeight <= 0 ||
		header.imageWidth < header.depthWidth || header.imageHeight < header.depthHeight)
	{
		DEBUG("error: invalid depth-data file '%s'", fileName.c_str());
		return false;
	}

	// read image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	uint16_t nFileNameSize;
	_fread_nolock(&nFileNameSize, sizeof(uint16_t), 1, f);
	imageFileName.resize(nFileNameSize);
	_fread_nolock(imageFileName.data(), sizeof(char), nFileNameSize, f);

	// read neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	uint32_t nIDs;
	_fread_nolock(&nIDs, sizeof(IIndex), 1, f);
	ASSERT(nIDs > 0 && nIDs < 256);
	IDs.resize(nIDs);
	_fread_nolock(IDs.data(), sizeof(IIndex), nIDs, f);

	// read pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	_fread_nolock(K.val, sizeof(REAL), 9, f);
	_fread_nolock(R.val, sizeof(REAL), 9, f);
	_fread_nolock(C.ptr(), sizeof(REAL), 3, f);

	// read depth-map
	dMin = header.dMin;
	dMax = header.dMax;
	imageSize.width = header.imageWidth;
	imageSize.height = header.imageHeight;
	if ((flags & HeaderDepthDataRaw::HAS_DEPTH) != 0) {
		depthMap.create(header.depthHeight, header.depthWidth);
		_fread_nolock(depthMap.getData(), sizeof(float) * depthMap.area(), 1, f);
	} else {
		_fseek_nolock(f, sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
	}

	// read normal-map
	if ((header.type & HeaderDepthDataRaw::HAS_NORMAL) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_NORMAL) != 0) {
			normalMap.create(header.depthHeight, header.depthWidth);
			_fread_nolock(normalMap.getData(), sizeof(float)*3 * normalMap.area(), 1, f);
		} else {
			_fseek_nolock(f, sizeof(float)*3*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read confidence-map
	if ((header.type & HeaderDepthDataRaw::HAS_CONF) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_CONF) != 0) {
			confMap.create(header.depthHeight, header.depthWidth);
			_fread_nolock(confMap.getData(), sizeof(float)* confMap.area(), 1, f);
		} else {
			_fseek_nolock(f, sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read visibility-map
	if ((header.type & HeaderDepthDataRaw::HAS_VIEWS) != 0) {
		if ((flags & HeaderDepthDataRaw::HAS_VIEWS) != 0) {
			viewsMap.create(header.depthHeight, header.depthWidth);
			_fread_nolock(viewsMap.getData(), sizeof(uint8_t)*4, viewsMap.area(), f);
		}
	}

	const bool bRet(ferror(f) == 0);
	_fclose_nolock(f);
	return bRet;
} // ImportDepthDataRaw
#endif
/*----------------------------------------------------------------*/


// compare the estimated and ground-truth depth-maps
void MVS::CompareDepthMaps(const DepthMap& depthMap, const DepthMap& depthMapGT, uint32_t idxImage, float threshold)
{
	TD_TIMER_START();
	const uint32_t width = (uint32_t)depthMap.width();
	const uint32_t height = (uint32_t)depthMap.height();
	// compute depth errors for each pixel
	cv::resize(depthMapGT, depthMapGT, depthMap.size());
	unsigned nErrorPixels(0);
	unsigned nExtraPixels(0);
	unsigned nMissingPixels(0);
	FloatArr depths(0, depthMap.area());
	FloatArr depthsGT(0, depthMap.area());
	FloatArr errors(0, depthMap.area());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			const Depth& depth = depthMap(i,j);
			const Depth& depthGT = depthMapGT(i,j);
			if (depth != 0 && depthGT == 0) {
				++nExtraPixels;
				continue;
			}
			if (depth == 0 && depthGT != 0) {
				++nMissingPixels;
				continue;
			}
			depths.Insert(depth);
			depthsGT.Insert(depthGT);
			const float error(depthGT==0 ? 0 : DepthSimilarity(depthGT, depth));
			errors.Insert(error);
		}
	}
	const float fPSNR((float)ComputePSNR(DMatrix32F((int)depths.size(),1,depths.data()), DMatrix32F((int)depthsGT.size(),1,depthsGT.data())));
	const MeanStd<float,double> ms(errors.data(), errors.size());
	const float mean((float)ms.GetMean());
	const float stddev((float)ms.GetStdDev());
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(errors.data(), errors.size()));
	#if TD_VERBOSE != TD_VERBOSE_OFF
	IDX idxPixel = 0;
	Image8U3 errorsVisual(depthMap.size());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			Pixel8U& pix = errorsVisual(i,j);
			const Depth& depth = depthMap(i,j);
			const Depth& depthGT = depthMapGT(i,j);
			if (depth != 0 && depthGT == 0) {
				pix = Pixel8U::GREEN;
				continue;
			}
			if (depth == 0 && depthGT != 0) {
				pix = Pixel8U::BLUE;
				continue;
			}
			const float error = errors[idxPixel++];
			if (depth == 0 && depthGT == 0) {
				pix = Pixel8U::BLACK;
				continue;
			}
			if (error > threshold) {
				pix = Pixel8U::RED;
				++nErrorPixels;
				continue;
			}
			const uint8_t gray((uint8_t)CLAMP((1.f-SAFEDIVIDE(ABS(error), threshold))*255.f, 0.f, 255.f));
			pix = Pixel8U(gray, gray, gray);
		}
	}
	errorsVisual.Save(ComposeDepthFilePath(idxImage, "errors.png"));
	#endif
	VERBOSE("Depth-maps compared for image % 3u: %.4f PSNR; %g median %g mean %g stddev error; %u (%.2f%%%%) error %u (%.2f%%%%) missing %u (%.2f%%%%) extra pixels (%s)",
		idxImage,
		fPSNR,
		th.first, mean, stddev,
		nErrorPixels, (float)nErrorPixels*100.f/depthMap.area(),
		nMissingPixels, (float)nMissingPixels*100.f/depthMap.area(),
		nExtraPixels, (float)nExtraPixels*100.f/depthMap.area(),
		TD_TIMER_GET_FMT().c_str()
	);
}

// compare the estimated and ground-truth normal-maps
void MVS::CompareNormalMaps(const NormalMap& normalMap, const NormalMap& normalMapGT, uint32_t idxImage)
{
	TD_TIMER_START();
	// load normal data
	const uint32_t width = (uint32_t)normalMap.width();
	const uint32_t height = (uint32_t)normalMap.height();
	// compute normal errors for each pixel
	cv::resize(normalMapGT, normalMapGT, normalMap.size());
	FloatArr errors(0, normalMap.area());
	for (uint32_t i=0; i<height; ++i) {
		for (uint32_t j=0; j<width; ++j) {
			const Normal& normal = normalMap(i,j);
			const Normal& normalGT = normalMapGT(i,j);
			if (normal != Normal::ZERO && normalGT == Normal::ZERO)
				continue;
			if (normal == Normal::ZERO && normalGT != Normal::ZERO)
				continue;
			if (normal == Normal::ZERO && normalGT == Normal::ZERO) {
				errors.Insert(0.f);
				continue;
			}
			ASSERT(ISEQUAL(norm(normal),1.f) && ISEQUAL(norm(normalGT),1.f));
			const float error(FR2D(ACOS(CLAMP(normal.dot(normalGT), -1.f, 1.f))));
			errors.Insert(error);
		}
	}
	const MeanStd<float,double> ms(errors.Begin(), errors.GetSize());
	const float mean((float)ms.GetMean());
	const float stddev((float)ms.GetStdDev());
	const std::pair<float,float> th(ComputeX84Threshold<float,float>(errors.Begin(), errors.GetSize()));
	VERBOSE("Normal-maps compared for image % 3u: %.2f median %.2f mean %.2f stddev error (%s)",
		idxImage,
		th.first, mean, stddev,
		TD_TIMER_GET_FMT().c_str()
	);
}
/*----------------------------------------------------------------*/
