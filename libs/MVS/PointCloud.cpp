/*
* PointCloud.cpp
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
#include "PointCloud.h"
#include "DepthMap.h"

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// S T R U C T S ///////////////////////////////////////////////////

PointCloud::PointCloud(const PointCloudStreaming& pcs)
{
	// Convert a PointCloudStreaming to a PointCloud rep.
	Release();

	// Copy points
	const size_t numPoints = pcs.NumPoints();
	points.Reserve(numPoints);
	const float* __restrict ptSrc = pcs.PointStream();
	for (size_t i = 0; i < numPoints; ++i, ptSrc += 3) {
		points.emplace_back(ptSrc[0], ptSrc[1], ptSrc[2]);
	}

	// Copy normals
	const size_t numNormals = pcs.NormalStream() ? numPoints : 0;
	normals.Reserve(numNormals);
	const float* __restrict nrmSrc = pcs.NormalStream();
	for (size_t i = 0; i < numNormals; ++i, nrmSrc += 3) {
		normals.emplace_back(nrmSrc[0], nrmSrc[1], nrmSrc[2]);
	}

	// Copy colors
	const size_t numColors = pcs.ColorStream() ? numPoints : 0;
	colors.Reserve(numColors);
	const uint8_t* __restrict colSrc = pcs.ColorStream();
	for (size_t i = 0; i < numColors; ++i, colSrc += 3) {
		colors.emplace_back(colSrc[0], colSrc[1], colSrc[2]);
	}

	// Point views
	pointViews.Reserve(numPoints);
	for (size_t i = 0; i < numPoints; ++i) {
		const size_t numPointViews = pcs.pointViewsSizes[i];
		const size_t offset = pcs.pointViewsOffsets[i];
		const uint32_t* src = pcs.pointViewsMemory.data() + offset;
		auto& tmp = pointViews.emplace_back();
		tmp.reserve((unsigned) numPointViews);
		for (size_t j = 0; j < numPointViews; ++j) {
			tmp.emplace_back(src[j]);
		}
	}

	// Point weights (may be null)
	if (!pcs.pointWeightsMemory.empty()) {
		pointWeights.Reserve(numPoints);
		for (size_t i = 0; i < numPoints; ++i) {
			const size_t numPointWeights = pcs.pointWeightsSizes[i];
			const size_t offset = pcs.pointWeightsOffsets[i];
			const float* src = pcs.pointWeightsMemory.data() + offset;
			auto& tmp = pointWeights.emplace_back();
			for (size_t j = 0; j < numPointWeights; ++j) {
				tmp.emplace_back(src[j]);
			}
		}
	}
}

void PointCloud::Release()
{
	points.Release();
	pointViews.Release();
	pointWeights.Release();
	normals.Release();
	colors.Release();
}
/*----------------------------------------------------------------*/


void PointCloud::RemovePoint(IDX idx)
{
	ASSERT(pointViews.IsEmpty() || pointViews.GetSize() == points.GetSize());
	if (!pointViews.IsEmpty())
		pointViews.RemoveAt(idx);
	ASSERT(pointWeights.IsEmpty() || pointWeights.GetSize() == points.GetSize());
	if (!pointWeights.IsEmpty())
		pointWeights.RemoveAt(idx);
	ASSERT(normals.IsEmpty() || normals.GetSize() == points.GetSize());
	if (!normals.IsEmpty())
		normals.RemoveAt(idx);
	ASSERT(colors.IsEmpty() || colors.GetSize() == points.GetSize());
	if (!colors.IsEmpty())
		colors.RemoveAt(idx);
	points.RemoveAt(idx);
}
void PointCloud::RemovePointsOutside(const OBB3f& obb) {
	ASSERT(obb.IsValid());
	RFOREACH(i, points)
		if (!obb.Intersects(points[i]))
			RemovePoint(i);
}
void PointCloud::RemoveMinViews(uint32_t thMinViews) {
	ASSERT(!pointViews.empty());
	RFOREACH(i, points)
		if (pointViews[i].size() < thMinViews)
			RemovePoint(i);
}
/*----------------------------------------------------------------*/


// compute the axis-aligned bounding-box of the point-cloud
PointCloud::Box PointCloud::GetAABB() const
{
	Box box(true);
	for (const Point& X: points)
		box.InsertFull(X);
	return box;
}
// same, but only for points inside the given AABB
PointCloud::Box PointCloud::GetAABB(const Box& bound) const
{
	Box box(true);
	for (const Point& X: points)
		if (bound.Intersects(X))
			box.InsertFull(X);
	return box;
}
// compute the axis-aligned bounding-box of the point-cloud
// with more than the given number of views
PointCloud::Box PointCloud::GetAABB(unsigned minViews) const
{
	if (pointViews.empty())
		return GetAABB();
	Box box(true);
	FOREACH(idx, points)
		if (pointViews[idx].size() >= minViews)
			box.InsertFull(points[idx]);
	return box;
}

// compute the center of the point-cloud as the median
PointCloud::Point PointCloud::GetCenter() const
{
	const Index step(5);
	const Index numPoints(points.size()/step);
	if (numPoints == 0)
		return Point::INF;
	typedef CLISTDEF0IDX(Point::Type,Index) Scalars;
	Scalars x(numPoints), y(numPoints), z(numPoints);
	for (Index i=0; i<numPoints; ++i) {
		const Point& X = points[i*step];
		x[i] = X.x;
		y[i] = X.y;
		z[i] = X.z;
	}
	return Point(x.GetMedian(), y.GetMedian(), z.GetMedian());
}
/*----------------------------------------------------------------*/


// estimate the ground-plane as the plane agreeing with most points
//  - planeThreshold: threshold used to estimate the ground plane (0 - auto)
Planef PointCloud::EstimateGroundPlane(const ImageArr& images, float planeThreshold, const String& fileExportPlane) const
{
	ASSERT(!IsEmpty());

	// remove some random points to speed up plane fitting
	const unsigned randMinPoints(150000);
	PointArr workPoints;
	const PointArr* pPoints;
	if (GetSize() > randMinPoints) {
		#ifndef _RELEASE
		SEACAVE::Random rnd(SEACAVE::Random::default_seed());
		#else
		SEACAVE::Random rnd;
		#endif
		const REAL randPointsRatio(MAXF(REAL(1e-4),(REAL)randMinPoints/GetSize()));
		const SEACAVE::Random::result_type randPointsTh(CEIL2INT<SEACAVE::Random::result_type>(randPointsRatio*SEACAVE::Random::max()));
		workPoints.reserve(CEIL2INT<PointArr::IDX>(randPointsRatio*GetSize()));
		for (const Point& X: points)
			if (rnd() <= randPointsTh)
				workPoints.emplace_back(X);
		pPoints = &workPoints;
	} else {
		pPoints = &points;
	}

	// fit plane to the point-cloud
	Planef plane;
	const float minInliersRatio(0.05f);
	double threshold(planeThreshold>0 ? (double)planeThreshold : DBL_MAX);
	const unsigned numInliers(planeThreshold > 0 ? EstimatePlaneTh(*pPoints, plane, threshold) : EstimatePlane(*pPoints, plane, threshold));
	if (numInliers < MINF(ROUND2INT<unsigned>(minInliersRatio*pPoints->size()), 1000u)) {
		plane.Invalidate();
		return plane;
	}
	if (planeThreshold <= 0)
		DEBUG("Ground plane estimated threshold: %g", threshold);

	// refine plane to inliers
	CLISTDEF0(Planef::POINT) inliers;
	const float maxThreshold(static_cast<float>(threshold * 2));
	for (const Point& X: *pPoints)
		if (plane.DistanceAbs(X) < maxThreshold)
			inliers.emplace_back(X);
	const RobustNorm::GemanMcClure<double> robust(threshold);
	plane.Optimize(inliers.data(), inliers.size(), robust);

	// make sure the plane is well oriented, negate plane normal if it faces same direction as cameras on average
	if (!images.empty()) {
		FloatArr cosView(0, images.size());
		for (const Image& imageData: images) {
			if (!imageData.IsValid())
				continue;
			cosView.push_back(plane.m_vN.dot((const Point3f::EVecMap&)Cast<float>(imageData.camera.Direction())));
		}
		if (cosView.GetMedian() > 0)
			plane.Negate();
	}

	// export points on the found plane if requested
	if (!fileExportPlane.empty()) {
		PointCloudStreaming pc;
		pc.ReserveColors(1+pPoints->size());
		pc.ReservePoints(1+pPoints->size());

		const Point orig(Point(plane.m_vN)*-plane.m_fD);
		pc.AddColor(Color::RED);
		pc.AddPoint(orig);
		for (const Point& X: *pPoints) {
			const float dist(plane.DistanceAbs(X));
			if (dist < threshold) {
				pc.AddPoint(X);
				const uint8_t color((uint8_t)(255.f*(1.f-dist/threshold)));
				pc.AddColorGray(color);
			}
		}
		pc.Save(fileExportPlane);
	}
	return plane;
}
/*----------------------------------------------------------------*/


// define a PLY file format composed only of vertices
namespace BasicPLY {
	typedef PointCloud::Point Point;
	typedef PointCloud::Color Color;
	typedef PointCloud::Normal Normal;
	// list of property information for a vertex
	struct PointColNormal {
		Point p;
		Color c;
		Normal n;
	};
	static const PLY::PlyProperty vert_props[] = {
		{"x",     PLY::Float32, PLY::Float32, offsetof(PointColNormal,p.x), 0, 0, 0, 0},
		{"y",     PLY::Float32, PLY::Float32, offsetof(PointColNormal,p.y), 0, 0, 0, 0},
		{"z",     PLY::Float32, PLY::Float32, offsetof(PointColNormal,p.z), 0, 0, 0, 0},
		{"red",   PLY::Uint8,   PLY::Uint8,   offsetof(PointColNormal,c.r), 0, 0, 0, 0},
		{"green", PLY::Uint8,   PLY::Uint8,   offsetof(PointColNormal,c.g), 0, 0, 0, 0},
		{"blue",  PLY::Uint8,   PLY::Uint8,   offsetof(PointColNormal,c.b), 0, 0, 0, 0},
		{"nx",    PLY::Float32, PLY::Float32, offsetof(PointColNormal,n.x), 0, 0, 0, 0},
		{"ny",    PLY::Float32, PLY::Float32, offsetof(PointColNormal,n.y), 0, 0, 0, 0},
		{"nz",    PLY::Float32, PLY::Float32, offsetof(PointColNormal,n.z), 0, 0, 0, 0}
	};
	// list of the kinds of elements in the PLY
	static const char* elem_names[] = {
		"vertex"
	};
} // namespace BasicPLY

#if 0
// load the dense point cloud from a PLY file
bool PointCloud::Load(const String& fileName)
{
	TD_TIMER_STARTD();

	ASSERT(!fileName.IsEmpty());
	Release();

	// open PLY file and read header
	PLY ply;
	if (!ply.read(fileName)) {
		DEBUG_EXTRA("error: invalid PLY file");
		return false;
	}

	// read PLY body
	BasicPLY::PointColNormal vertex;
	for (int i = 0; i < (int)ply.elems.size(); i++) {
		int elem_count;
		LPCSTR elem_name = ply.setup_element_read(i, &elem_count);
		if (PLY::equal_strings(BasicPLY::elem_names[0], elem_name)) {
			PLY::PlyElement* elm = ply.find_element(elem_name);
			const size_t nMaxProps(SizeOfArray(BasicPLY::vert_props));
			for (size_t p=0; p<nMaxProps; ++p) {
				if (ply.find_property(elm, BasicPLY::vert_props[p].name.c_str()) < 0)
					continue;
				ply.setup_property(BasicPLY::vert_props[p]);
				switch (p) {
				case 0: points.Resize((IDX)elem_count); break;
				case 3: colors.Resize((IDX)elem_count); break;
				case 6: normals.Resize((IDX)elem_count); break;
				}
			}
			for (int v=0; v<elem_count; ++v) {
				ply.get_element(&vertex);
				points[v] = vertex.p;
				if (!colors.IsEmpty())
					colors[v] = vertex.c;
				if (!normals.IsEmpty())
					normals[v] = vertex.n;
			}
		} else {
			ply.get_other_element();
		}
	}
	if (points.IsEmpty()) {
		DEBUG_EXTRA("error: invalid point-cloud");
		return false;
	}

	DEBUG_EXTRA("Point-cloud loaded: %u points (%s)", points.GetSize(), TD_TIMER_GET_FMT().c_str());
	return true;
} // Load

// save the dense point cloud as PLY file
bool PointCloud::Save(const String& fileName, bool bLegacyTypes) const
{
	if (points.IsEmpty())
		return false;
	TD_TIMER_STARTD();

	// create PLY object
	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);
	PLY ply;
	if (bLegacyTypes)
		ply.set_legacy_type_names();
	if (!ply.write(fileName, 1, BasicPLY::elem_names, PLY::BINARY_LE, 64*1024))
		return false;

	if (normals.IsEmpty()) {
		// describe what properties go into the vertex elements
		ply.describe_property(BasicPLY::elem_names[0], 6, BasicPLY::vert_props);

		// write the header
		ply.element_count(BasicPLY::elem_names[0], (int)points.GetSize());
		if (!ply.header_complete())
			return false;

		// export the array of 3D points
		BasicPLY::PointColNormal vertex;
		FOREACH(i, points) {
			// export the vertex position, normal and color
			vertex.p = points[i];
			vertex.c = (!colors.IsEmpty() ? colors[i] : Color::WHITE);
			ply.put_element(&vertex);
		}
	} else {
		// describe what properties go into the vertex elements
		ply.describe_property(BasicPLY::elem_names[0], 9, BasicPLY::vert_props);

		// write the header
		ply.element_count(BasicPLY::elem_names[0], (int)points.GetSize());
		if (!ply.header_complete())
			return false;

		// export the array of 3D points
		BasicPLY::PointColNormal vertex;
		FOREACH(i, points) {
			// export the vertex position, normal and color
			vertex.p = points[i];
			vertex.n = normals[i];
			vertex.c = (!colors.IsEmpty() ? colors[i] : Color::WHITE);
			ply.put_element(&vertex);
		}
	}

	DEBUG_EXTRA("Point-cloud saved: %u points (%s)", points.GetSize(), TD_TIMER_GET_FMT().c_str());
	return true;
} // Save

// save the dense point cloud having >=N views as PLY file
bool PointCloud::SaveNViews(const String& fileName, uint32_t minViews, bool bLegacyTypes) const
{
	if (points.IsEmpty())
		return false;
	TD_TIMER_STARTD();

	// create PLY object
	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);
	PLY ply;
	if (bLegacyTypes)
		ply.set_legacy_type_names();
	if (!ply.write(fileName, 1, BasicPLY::elem_names, PLY::BINARY_LE, 64*1024))
		return false;

	if (normals.IsEmpty()) {
		// describe what properties go into the vertex elements
		ply.describe_property(BasicPLY::elem_names[0], 6, BasicPLY::vert_props);

		// export the array of 3D points
		BasicPLY::PointColNormal vertex;
		FOREACH(i, points) {
			if (pointViews[i].size() < minViews)
				continue;
			// export the vertex position, normal and color
			vertex.p = points[i];
			vertex.c = (!colors.IsEmpty() ? colors[i] : Color::WHITE);
			ply.put_element(&vertex);
		}
	} else {
		// describe what properties go into the vertex elements
		ply.describe_property(BasicPLY::elem_names[0], 9, BasicPLY::vert_props);

		// export the array of 3D points
		BasicPLY::PointColNormal vertex;
		FOREACH(i, points) {
			if (pointViews[i].size() < minViews)
				continue;
			// export the vertex position, normal and color
			vertex.p = points[i];
			vertex.n = normals[i];
			vertex.c = (!colors.IsEmpty() ? colors[i] : Color::WHITE);
			ply.put_element(&vertex);
		}
	}
	const int numPoints(ply.get_current_element_count());

	// write the header
	if (!ply.header_complete())
		return false;

	DEBUG_EXTRA("Point-cloud saved: %u points with at least %u views each (%s)", numPoints, minViews, TD_TIMER_GET_FMT().c_str());
	return true;
} // SaveNViews
/*----------------------------------------------------------------*/

#endif
// print various statistics about the point cloud
void PointCloud::PrintStatistics(const Image* pImages, const OBB3f* pObb) const
{
	String strPoints;
	if (pObb && pObb->IsValid()) {
		// print points distribution
		size_t nInsidePoints(0);
		MeanStdMinMax<double> accInside, accOutside;
		FOREACH(idx, points) {
			const bool bInsideROI(pObb->Intersects(points[idx]));
			if (bInsideROI)
				++nInsidePoints;
			if (!pointViews.empty()) {
				if (bInsideROI)
					accInside.Update(pointViews[idx].size());
				else
					accOutside.Update(pointViews[idx].size());
			}
		}
		strPoints = String::FormatString(
			"\n - points info:"
			"\n\t%u points inside ROI (%.2f%%)",
			nInsidePoints, 100.0*nInsidePoints/GetSize()
		);
		if (!pointViews.empty()) {
			strPoints += String::FormatString(
				"\n\t inside ROI track length: %g min / %g mean (%g std) / %g max"
				"\n\toutside ROI track length: %g min / %g mean (%g std) / %g max",
				accInside.minVal, accInside.GetMean(), accInside.GetStdDev(), accInside.maxVal,
				accOutside.minVal, accOutside.GetMean(), accOutside.GetStdDev(), accOutside.maxVal
			);
		}
	}
	String strViews;
	if (!pointViews.empty()) {
		// print views distribution
		size_t nViews(0);
		size_t nPoints1m(0), nPoints2(0), nPoints3(0), nPoints4p(0);
		size_t nPointsOpposedViews(0);
		MeanStdMinMax<double> acc;
		FOREACH(idx, points) {
			const PointCloud::ViewArr& views = pointViews[idx];
			nViews += views.size();
			switch (views.size()) {
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			case 3:
				++nPoints3;
				break;
			default:
				++nPoints4p;
			}
			acc.Update(views.size());
		}
		strViews = String::FormatString(
			"\n - visibility info (%u views - %.2f views/point)%s:"
			"\n\t% 9u points with 1- views (%.2f%%)"
			"\n\t% 9u points with 2  views (%.2f%%)"
			"\n\t% 9u points with 3  views (%.2f%%)"
			"\n\t% 9u points with 4+ views (%.2f%%)"
			"\n\t%g min / %g mean (%g std) / %g max",
			nViews, (REAL)nViews/GetSize(),
			nPointsOpposedViews ? String::FormatString(" (%u (%.2f%%) points with opposed views)", nPointsOpposedViews, 100.f*nPointsOpposedViews/GetSize()).c_str() : "",
			nPoints1m, 100.f*nPoints1m/GetSize(), nPoints2, 100.f*nPoints2/GetSize(), nPoints3, 100.f*nPoints3/GetSize(), nPoints4p, 100.f*nPoints4p/GetSize(),
			acc.minVal, acc.GetMean(), acc.GetStdDev(), acc.maxVal
		);
	}
	String strNormals;
	if (!normals.empty()) {
		if (!pointViews.empty() && pImages != NULL) {
			// print normal/views angle distribution
			size_t nViews(0);
			size_t nPointsm(0), nPoints3(0), nPoints10(0), nPoints25(0), nPoints40(0), nPoints60(0), nPoints90p(0);
			const REAL thCosAngle3(COS(D2R(3.f)));
			const REAL thCosAngle10(COS(D2R(10.f)));
			const REAL thCosAngle25(COS(D2R(25.f)));
			const REAL thCosAngle40(COS(D2R(40.f)));
			const REAL thCosAngle60(COS(D2R(60.f)));
			const REAL thCosAngle90(COS(D2R(90.f)));
			FOREACH(idx, points) {
				const PointCloud::Point& X = points[idx];
				const PointCloud::Normal& N = normals[idx];
				const PointCloud::ViewArr& views = pointViews[idx];
				nViews += views.size();
				for (IIndex idxImage: views) {
					const Point3f X2Cam(Cast<float>(pImages[idxImage].camera.C)-X);
					const REAL cosAngle(ComputeAngle(X2Cam.ptr(), N.ptr()));
					if (cosAngle <= thCosAngle90)
						++nPoints90p;
					else if (cosAngle <= thCosAngle60)
						++nPoints60;
					else if (cosAngle <= thCosAngle40)
						++nPoints40;
					else if (cosAngle <= thCosAngle25)
						++nPoints25;
					else if (cosAngle <= thCosAngle10)
						++nPoints10;
					else if (cosAngle <= thCosAngle3)
						++nPoints3;
					else
						++nPointsm;
				}
			}
			strNormals = String::FormatString(
				"\n - normals visibility info:"
				"\n\t% 9u points with 3- degrees (%.2f%%)"
				"\n\t% 9u points with 10 degrees (%.2f%%)"
				"\n\t% 9u points with 25 degrees (%.2f%%)"
				"\n\t% 9u points with 40 degrees (%.2f%%)"
				"\n\t% 9u points with 60 degrees (%.2f%%)"
				"\n\t% 9u points with 90+ degrees (%.2f%%)",
				nPointsm, 100.f*nPointsm/nViews, nPoints3, 100.f*nPoints3/nViews, nPoints10, 100.f*nPoints10/nViews,
				nPoints40, 100.f*nPoints40/nViews, nPoints60, 100.f*nPoints60/nViews, nPoints90p, 100.f*nPoints90p/nViews
			);
		} else {
			strNormals = "\n - normals info";
		}
	}
	String strWeights;
	if (!pointWeights.empty()) {
		// print weights statistics
		MeanStdMinMax<double> acc;
		for (const PointCloud::WeightArr& weights: pointWeights) {
			float avgWeight(0);
			for (PointCloud::Weight w: weights)
				avgWeight += w;
			acc.Update(avgWeight/weights.size());
		}
		strWeights = String::FormatString(
			"\n - weights info:"
			"\n\t%g min / %g mean (%g std) / %g max",
			acc.minVal, acc.GetMean(), acc.GetStdDev(), acc.maxVal
		);
	}
	String strColors;
	if (!colors.empty()) {
		// print colors statistics
		strColors = "\n - colors";
	}
	VERBOSE("Point-cloud composed of %u points with:%s%s%s%s",
		GetSize(),
		strPoints.c_str(),
		strViews.c_str(),
		strNormals.c_str(),
		strWeights.c_str(),
		strColors.c_str()
	);
} // PrintStatistics
/*----------------------------------------------------------------*/

// Create a streaming cloud from a point cloud.
PointCloudStreaming::PointCloudStreaming(const PointCloud& src)
{
	Release();

	// Convert a PointCloud to a PointCloudStreaming rep.

	// Copy points
	const size_t numPoints = src.GetSize();
	ReservePoints(numPoints);
	for (size_t i = 0; i < numPoints; ++i) {
		AddPoint(src.points[i]);
	}

	// Copy normals
	const size_t numNormals = src.normals.GetSize();
	ReserveNormals(numNormals);
	for (size_t i = 0; i < numNormals; ++i) {
		AddNormal(src.normals[i]);
	}

	// Copy colors
	const size_t numColors = src.normals.GetSize();
	ReserveColors(numColors);
	for (size_t i = 0; i < numColors; ++i) {
			AddColor(src.colors[i]);
	}

	// Calculate the amount of space needed to store the point views and weights in a flat array.

	// Copy pointViews
	if (!src.pointViews.IsEmpty()) {
		ReservePointViewsSizeAndOffset(numPoints);
		const size_t numFlatPointViewItems = (size_t) std::accumulate(
			std::begin(src.pointViews),
			std::end(src.pointViews),
			0,
			[](const size_t cnt, const auto& i)
			{
				return cnt + i.size();
			}
		);
		ReservePointViewsMemory(numFlatPointViewItems);

		for (size_t i = 0; i < numPoints; ++i) {
			AddViews(src.pointViews[i].begin(), src.pointViews[i].end());
		}
	}

	// Copy pointWeights
	if (!src.pointWeights.IsEmpty()) {
		ReservePointWeightsSizeAndOffset(numPoints);
		size_t numFlatPointWeightsItems = std::accumulate(
			std::begin(src.pointWeights),
			std::end(src.pointWeights),
			0,
			[](const size_t cnt, const auto& i)
			{
				return cnt + i.size();
			}
		);
		ReservePointWeightsMemory(numFlatPointWeightsItems);

		for (size_t i = 0; i < numPoints; ++i) {
			AddWeights(src.pointWeights[i].begin(), src.pointWeights[i].end());
		}
	}
}

void PointCloudStreaming::Release()
{
	pointsXYZ.clear();
	pointViewsOffsets.clear();
	pointViewsSizes.clear();
	pointWeightsOffsets.clear();
	pointWeightsSizes.clear();

	pointViewsMemory.clear();
	pointWeightsMemory.clear();

	normalsXYZ.clear();

	colorsRGB.clear();
}

// load the dense point cloud from a PLY file
bool PointCloudStreaming::Load(const String& fileName)
{
	TD_TIMER_STARTD();

	ASSERT(!fileName.IsEmpty());
	Release();

	// open PLY file and read header
	PLY ply;
	if (!ply.read(fileName)) {
		DEBUG_EXTRA("error: invalid PLY file");
		return false;
	}

	// read PLY body
	BasicPLY::PointColNormal vertex;
	for (int i = 0; i < (int)ply.elems.size(); i++) {
		int elem_count;
		LPCSTR elem_name = ply.setup_element_read(i, &elem_count);
		if (PLY::equal_strings(BasicPLY::elem_names[0], elem_name)) {
			PLY::PlyElement* elm = ply.find_element(elem_name);
			const size_t nMaxProps(SizeOfArray(BasicPLY::vert_props));
			for (size_t p=0; p<nMaxProps; ++p) {
				if (ply.find_property(elm, BasicPLY::vert_props[p].name.c_str()) < 0)
					continue;
				ply.setup_property(BasicPLY::vert_props[p]);
				switch (p) {
				case 0: pointsXYZ.resize((IDX)elem_count*3); break;
				case 3: colorsRGB.resize((IDX)elem_count*3); break;
				case 6: normalsXYZ.resize((IDX)elem_count*3); break;
				}
			}

			uint8_t unusedColor[3];
			float unusedNormal[3];
			float* __restrict pPoints = pointsXYZ.data();
			uint8_t* __restrict pColors = colorsRGB.empty() ? unusedColor : colorsRGB.data();
			size_t srcColorOffset = colorsRGB.empty() ? 0 : 3;
			float* __restrict pNormals = normalsXYZ.empty() ? unusedNormal : normalsXYZ.data();
			size_t srcNormalOffset = normalsXYZ.empty() ? 0 : 3;
			for (int v=0; v<elem_count; ++v) {
				ply.get_element(&vertex);

				pPoints[0] = vertex.p.x;
				pPoints[1] = vertex.p.y;
				pPoints[2] = vertex.p.z;
				pPoints += 3;

				pColors[0] = vertex.c.r;
				pColors[1] = vertex.c.g;
				pColors[2] = vertex.c.b;
				pColors += srcColorOffset;

				pNormals[0] = vertex.n.x;
				pNormals[1] = vertex.n.y;
				pNormals[2] = vertex.n.z;
				pNormals += srcNormalOffset;
			}
		} else {
			ply.get_other_element();
		}
	}
	if (pointsXYZ.empty()) {
		DEBUG_EXTRA("error: invalid point-cloud");
		return false;
	}

	DEBUG_EXTRA("Point-cloud loaded: %u points (%s)", NumPoints(), TD_TIMER_GET_FMT().c_str());
	return true;
} // Load

// save the dense point cloud as PLY file
#if 1 // Direct writing
bool PointCloudStreaming::Save(const String& fileName, bool bLegacyTypes) const
{
	if (pointsXYZ.empty())
		return false;

	TD_TIMER_STARTD();

	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);

	std::ofstream out(fileName.c_str(), std::ios::binary);
	if (!out.is_open())
		return false;

	const size_t numPoints = NumPoints();
	const bool hasNormals = !normalsXYZ.empty();
	const bool hasColors = !colorsRGB.empty();

	// ------------------------------------------------------------
	// PLY header
	// ------------------------------------------------------------
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";

	if (bLegacyTypes)
		out << "comment legacy_type_names\n";

	out << "element vertex " << numPoints << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";

	if (hasNormals) {
		out << "property float nx\n";
		out << "property float ny\n";
		out << "property float nz\n";
	}

	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "end_header\n";

	// ------------------------------------------------------------
	// Body: stream binary vertex data
	// ------------------------------------------------------------
	const float* __restrict pPoint = pointsXYZ.data();
	const float* __restrict pNormal = hasNormals ? normalsXYZ.data() : nullptr;

	const uint8_t white[3] = { 255, 255, 255 };
	const uint8_t* __restrict pColor = hasColors ? colorsRGB.data() : white;
	const size_t colorStride = hasColors ? 3 : 0;

	// Optional: large output buffer (helps HDDs)
	constexpr size_t kBufSize = 1 << 20; // 1 MB
	std::vector<char> buffer;
	buffer.reserve(kBufSize);

	for (size_t i = 0; i < numPoints; ++i) {
		// Position
		buffer.insert(buffer.end(),
			reinterpret_cast<const char*>(pPoint),
			reinterpret_cast<const char*>(pPoint + 3));
		pPoint += 3;

		// Normal
		if (hasNormals) {
			buffer.insert(buffer.end(),
				reinterpret_cast<const char*>(pNormal),
				reinterpret_cast<const char*>(pNormal + 3));
			pNormal += 3;
		}

		// Color
		buffer.push_back((char)pColor[0]);
		buffer.push_back((char)pColor[1]);
		buffer.push_back((char)pColor[2]);
		pColor += colorStride;

		// Flush buffer if full
		if (buffer.size() >= kBufSize) {
			out.write(buffer.data(), buffer.size());
			buffer.clear();
		}
	}

	// Flush remainder
	if (!buffer.empty()) {
		out.write(buffer.data(), buffer.size());
		buffer.clear();
	}

	out.flush();
	out.close();

	DEBUG_EXTRA("Point-cloud saved: %u points (%s)",
		NumPoints(),
		TD_TIMER_GET_FMT().c_str());

	return true;
}
#else
bool PointCloudStreaming::Save(const String& fileName, bool bLegacyTypes) const
{
	if (pointsXYZ.empty())
		return false;
	TD_TIMER_STARTD();

	// create PLY object
	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);
	PLY ply;

	if (bLegacyTypes)
		ply.set_legacy_type_names();

	// Make a better guess at the final size...
	// Always binary.
	size_t guessedSize = 0;
	if (!normalsXYZ.empty()) {
		guessedSize += sizeof(float)*3*NumPoints();
	}
	guessedSize += sizeof(float)*3*NumPoints(); // vertices
	guessedSize += sizeof(uint8_t)*3*NumPoints(); // colors
	guessedSize += 64*1024; // From before, far too large, but should cover all headers.

	if (!ply.write(fileName, 1, BasicPLY::elem_names, PLY::BINARY_LE, guessedSize))
		return false;

	if (normalsXYZ.empty()) {
		// describe what properties go into the vertex elements
		ply.describe_property(BasicPLY::elem_names[0], 6, BasicPLY::vert_props);

		// write the header
		ply.element_count(BasicPLY::elem_names[0], (int) NumPoints());
		if (!ply.header_complete())
			return false;

		// export the array of 3D points
		BasicPLY::PointColNormal vertex;

		const float* __restrict pPoint = PointStream();
		uint8_t white[3] = {0xFF, 0xFF, 0xFF};
		const uint8_t* __restrict pColor;
		size_t srcColorOffset;
		if (colorsRGB.empty()) {
			pColor = white;
			srcColorOffset = 0;
		} else {
			pColor = ColorStream();
			srcColorOffset = 3;
		}

		for (size_t i = 0, cnt = NumPoints(); i < cnt; ++i) {
			// export the vertex position, normal and color
			vertex.p = PointCloud::Point(pPoint[0], pPoint[1], pPoint[2]);
			vertex.c = PointCloud::Color(pColor[0], pColor[1], pColor[2]);
			pPoint += 3;
			pColor += srcColorOffset;
			ply.put_element(&vertex);
		}
	} else {
		// describe what properties go into the vertex elements
		ply.describe_property(BasicPLY::elem_names[0], 9, BasicPLY::vert_props);

		// write the header
		ply.element_count(BasicPLY::elem_names[0], (int)NumPoints());
		if (!ply.header_complete())
			return false;

		// export the array of 3D points

		BasicPLY::PointColNormal vertex;
		const float* __restrict pPoint = pointsXYZ.data();
		const float* __restrict pNormal = normalsXYZ.data();
		uint8_t white[3] = {0xFF, 0xFF, 0xFF};
		const uint8_t* __restrict pColor;
		size_t srcColorOffset;
		if (colorsRGB.empty()) {
			pColor = white;
			srcColorOffset = 0;
		} else {
			pColor = colorsRGB.data();
			srcColorOffset = 3;
		}

		for (size_t i = 0, cnt = NumPoints(); i < cnt; ++i) {
			// export the vertex position, normal and color
			const float x = pPoint[0];
			const float y = pPoint[1];
			const float z = pPoint[2];
			const float nx = pNormal[0];
			const float ny = pNormal[1];
			const float nz = pNormal[2];
			const uint8_t r = pColor[0];
			const uint8_t g = pColor[1];
			const uint8_t b = pColor[2];

			vertex.p.x = x;
			vertex.p.y = y;
			vertex.p.z = z;
			vertex.n.x = nx;
			vertex.n.y = ny;
			vertex.n.z = nz;
			vertex.c.r = r;
			vertex.c.g = g;
			vertex.c.b = b;

			//vertex.p = PointCloud::Point(pPoint[0], pPoint[1], pPoint[2]);
			//vertex.n = PointCloud::Normal(pNormal[0], pNormal[1], pNormal[2]);
			pPoint += 3;
			pNormal += 3;
			//vertex.c = PointCloud::Color(pColor[0], pColor[1], pColor[2]);
			pColor += srcColorOffset;

			ply.put_element(&vertex);
		}
	}

	DEBUG_EXTRA("Point-cloud saved: %u points (%s)", NumPoints(), TD_TIMER_GET_FMT().c_str());
	return true;
} // Save
#endif

// save the dense point cloud having >=N views as PLY file
bool PointCloudStreaming::SaveNViews(const String& fileName, uint32_t minViews, bool bLegacyTypes) const
{
	if (pointsXYZ.empty())
		return false;
	TD_TIMER_STARTD();

	// create PLY object
	ASSERT(!fileName.IsEmpty());
	Util::ensureFolder(fileName);
	PLY ply;
	if (bLegacyTypes)
		ply.set_legacy_type_names();
	if (!ply.write(fileName, 1, BasicPLY::elem_names, PLY::BINARY_LE, 64*1024))
		return false;

	const int numProperties = normalsXYZ.empty() ? 6 : 9;
	// describe what properties go into the vertex elements
	ply.describe_property(BasicPLY::elem_names[0], numProperties, BasicPLY::vert_props);

	// export the array of 3D points
	BasicPLY::PointColNormal vertex;

	const float* __restrict pPoint = PointStream();

	const float* __restrict pNormal = NormalStream();
	const size_t srcNormalOffset = pNormal ? 3 : 0;
	const Point3f normal(1.f, 0.f, 0.f);
	if (!pNormal) {
		pNormal = &normal.x;
	}

	uint8_t white[3] = {0xFF, 0xFF, 0xFF};
	const uint8_t* __restrict pColor = ColorStream();
	const size_t srcColorOffset = pColor ? 3 : 0;
	if (!pColor) {
		pColor = white;
	}

	for (size_t i = 0, cnt = NumPoints(); i < cnt; ++i) {
		if (pointViewsSizes[i] < minViews)
			continue;
		// export the vertex position, normal and color
		vertex.p = PointCloud::Point(pPoint[0], pPoint[1], pPoint[2]);
		pPoint += 3;

		vertex.c = PointCloud::Color(pColor[0], pColor[1], pColor[2]);
		pColor += srcColorOffset;

		vertex.n = PointCloud::Normal(pNormal[0], pNormal[1], pNormal[2]);
		pNormal += srcNormalOffset;

		ply.put_element(&vertex);
	}

	const int numPoints(ply.get_current_element_count());

	// write the header
	if (!ply.header_complete())
		return false;

	DEBUG_EXTRA("Point-cloud saved: %u points with at least %u views each (%s)", numPoints, minViews, TD_TIMER_GET_FMT().c_str());
	return true;
} // SaveNViews

void PointCloudStreaming::RemovePoint(IDX idx)
{
	// This does not remove the memory used by pointViewsMemory and pointWeightsMemory.

	pointViewsOffsets.erase(std::begin(pointWeightsSizes) + idx);
	pointViewsSizes.erase(std::begin(pointWeightsSizes) + idx);

	pointWeightsOffsets.erase(std::begin(pointWeightsSizes) + idx);
	pointWeightsSizes.erase(std::begin(pointWeightsSizes) + idx);

	normalsXYZ.erase(std::begin(normalsXYZ) + idx*3, std::begin(normalsXYZ) + (idx+1)*3);
	colorsRGB.erase(std::begin(colorsRGB) + idx*3, std::begin(colorsRGB) + (idx+1)*3);
	pointsXYZ.erase(std::begin(pointsXYZ) + idx*3, std::begin(pointsXYZ) + (idx+1)*3);
}

// compute the axis-aligned bounding-box of the point-cloud
PointCloudStreaming::Box PointCloudStreaming::GetAABB() const
{
	Box box(true);
	for (size_t i = 0, cnt = NumPoints(); i < cnt; ++i) {
		const Point3f& X = Point(i);
		box.InsertFull(X);
	}
	return box;
}
// same, but only for points inside the given AABB
PointCloudStreaming::Box PointCloudStreaming::GetAABB(const Box& bound) const
{
	Box box(true);
	for (size_t i = 0, cnt = NumPoints(); i < cnt; ++i) {
		const Point3f& X = Point(i);
		if (bound.Intersects(X))
			box.InsertFull(X);
	}
	return box;
}
// compute the axis-aligned bounding-box of the point-cloud
// with more than the given number of views
PointCloudStreaming::Box PointCloudStreaming::GetAABB(unsigned minViews) const
{
	if (pointViewsSizes.empty())
		return GetAABB();
	Box box(true);
	for (size_t i = 0, cnt = NumPoints(); i < cnt; ++i) {
		if (pointViewsSizes[i] >= minViews)
			box.InsertFull(Point(i));
	}
	return box;
}

// compute the center of the point-cloud as the median
TPoint3<float> PointCloudStreaming::GetCenter() const
{
	const Index step(5);
	const Index numPoints(NumPoints()/step);
	if (numPoints == 0)
		return TPoint3<float>::INF;
	typedef CLISTDEF0IDX(float,Index) Scalars;
	Scalars x(numPoints), y(numPoints), z(numPoints);
	for (Index i=0; i<numPoints; ++i) {
		const Point3f& X = Point(i*step);
		x[i] = X.x;
		y[i] = X.y;
		z[i] = X.z;
	}
	return Point3f(x.GetMedian(), y.GetMedian(), z.GetMedian());
}
/*----------------------------------------------------------------*/