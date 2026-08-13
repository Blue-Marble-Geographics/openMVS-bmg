/*
* PointCloud.h
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

#ifndef _MVS_POINTCLOUD_H_
#define _MVS_POINTCLOUD_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Image.h"
#ifdef _USE_BOOST
#include <boost/serialization/split_member.hpp>
#endif

// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

class PointCloudStreaming;

// a point-cloud containing the points with the corresponding views
// and optionally weights, normals and colors
// (same size as the number of points or zero)
class MVS_API PointCloud
{
public:
	typedef IDX Index;

	typedef TPoint3<float> Point;
	typedef CLISTDEF0IDX(Point,Index) PointArr;

	typedef uint32_t View;
	typedef SEACAVE::cList<View,const View,0,4,uint32_t> ViewArr;
	typedef CLISTDEFIDX(ViewArr,Index) PointViewArr;

	typedef float Weight;
	typedef SEACAVE::cList<Weight,const Weight,0,4,uint32_t> WeightArr;
	typedef CLISTDEFIDX(WeightArr,Index) PointWeightArr;

	typedef TPoint3<float> Normal;
	typedef CLISTDEF0IDX(Normal,Index) NormalArr;

	typedef Pixel8U Color;
	typedef CLISTDEF0IDX(Color,Index) ColorArr;

	typedef AABB3f Box;

	typedef TOctree<PointArr,Point::Type,3> Octree;

public:
	PointArr points;
	PointViewArr pointViews; // array of views for each point (ordered increasing)
	PointWeightArr pointWeights;
	NormalArr normals;
	ColorArr colors;

public:
	PointCloud() = default;
	PointCloud(const PointCloudStreaming& src);
	void Release();

	inline bool IsEmpty() const { ASSERT(points.GetSize() == pointViews.GetSize() || pointViews.IsEmpty()); return points.IsEmpty(); }
	inline bool IsValid() const { ASSERT(points.GetSize() == pointViews.GetSize() || pointViews.IsEmpty()); return !pointViews.IsEmpty(); }
	inline size_t GetSize() const { ASSERT(points.GetSize() == pointViews.GetSize() || pointViews.IsEmpty()); return points.GetSize(); }

	void RemovePoint(IDX);
	void RemovePointsOutside(const OBB3f&);
	void RemoveMinViews(uint32_t thMinViews);

	Box GetAABB() const;
	Box GetAABB(const Box& bound) const;
	Box GetAABB(unsigned minViews) const;
	Point GetCenter() const;

	Planef EstimateGroundPlane(const ImageArr& images, float planeThreshold=0, const String& fileExportPlane="") const;

	void PrintStatistics(const Image* pImages = NULL, const OBB3f* pObb = NULL) const;
};

class MVS_API PointCloudStreaming
{
public:
	PointCloudStreaming() = default;
	PointCloudStreaming(const PointCloud& src);

	void ReserveColors(size_t cnt)
	{
		colorsRGB.reserve(cnt*3);
	}

	void ReservePoints(size_t cnt)
	{
		pointsXYZ.reserve(cnt*3);
	}

	void ReserveNormals(size_t cnt)
	{
		normalsXYZ.reserve(cnt*3);
	}

	void ReservePointViewsMemory(size_t cnt)
	{
		pointViewsMemory.reserve(cnt);
	}

	void ReservePointViewsSizeAndOffset(size_t cnt)
	{
		pointViewsOffsets.reserve(cnt);
		pointViewsSizes.reserve(cnt);
	}

	void ReservePointWeightsMemory(size_t cnt)
	{
		pointWeightsMemory.reserve(cnt);
	}

	void ReservePointWeightsSizeAndOffset(size_t cnt)
	{
		pointWeightsOffsets.reserve(cnt);
		pointWeightsSizes.reserve(cnt);
	}

	void AddView(uint32_t idx)
	{
		pointViewsSizes.push_back(1);
		pointViewsOffsets.push_back((uint32_t)pointViewsMemory.size());
		pointViewsMemory.push_back(idx);
	}

	template<class T>
	void AddViews(T first, T last)
	{
		size_t cnt = last-first;
		pointViewsSizes.push_back((uint32_t) cnt);
		pointViewsOffsets.push_back((uint32_t) pointViewsMemory.size());

		while (cnt-- > 0) {
			pointViewsMemory.push_back(*first);
			++first;
		}
	}

	void AddWeight(float weight)
	{
		pointWeightsSizes.push_back(1);
		pointWeightsOffsets.push_back((uint32_t) pointWeightsMemory.size());
		pointWeightsMemory.push_back(weight);
	}

	template<class T>
	void AddWeights(T first, T last)
	{
		size_t cnt = last-first;
		pointWeightsSizes.push_back((uint32_t) cnt);
		pointWeightsOffsets.push_back((uint32_t) pointWeightsMemory.size());

		while (cnt-- > 0) {
			pointWeightsMemory.push_back(*first);
			++first;
		}
	}

	void ClearWeights()
	{
		pointWeightsMemory.clear();
		pointWeightsOffsets.clear();
		pointWeightsSizes.clear();
	}

	void AddColor(const Pixel8U& color)
	{
		colorsRGB.emplace_back(color.r);
		colorsRGB.emplace_back(color.g);
		colorsRGB.emplace_back(color.b);
	}

	void AddColorGray(const uint8_t color)
	{
		colorsRGB.emplace_back(color);
		colorsRGB.emplace_back(color);
		colorsRGB.emplace_back(color);
	}

	void AddColor(const uint8_t* color)
	{
		colorsRGB.emplace_back(color[0]);
		colorsRGB.emplace_back(color[1]);
		colorsRGB.emplace_back(color[2]);
	}

	void AddNormal(const Point3f& pt)
	{
		normalsXYZ.emplace_back(pt.x);
		normalsXYZ.emplace_back(pt.y);
		normalsXYZ.emplace_back(pt.z);
	}

	float* AddPoint(const Point3f& pt)
	{
		pointsXYZ.emplace_back(pt.x);
		pointsXYZ.emplace_back(pt.y);
		pointsXYZ.emplace_back(pt.z);
		return &pointsXYZ[pointsXYZ.size() - 3];
	}

	const uint8_t* ColorStream() const noexcept { return colorsRGB.empty() ? nullptr : colorsRGB.data(); }
	const float* NormalStream() const noexcept { return normalsXYZ.empty() ? nullptr : normalsXYZ.data(); }
	const float* PointStream() const noexcept { return pointsXYZ.empty() ? nullptr : pointsXYZ.data(); }

	const Pixel8U& Color(size_t idx) const { return *(Pixel8U*)(ColorStream()+idx*3); }
	Pixel8U& Color(size_t idx) { return *(Pixel8U*)(ColorStream()+idx*3); }

	const Point3f& Normal(size_t idx) const { return *(Point3f*)(NormalStream()+idx*3); }
	Point3f& Normal(size_t idx) { return *(Point3f*)(NormalStream()+idx*3); }

	const Point3f& Point(size_t idx) const { return *(Point3f*)(PointStream()+idx*3); }
	Point3f& Point(size_t idx) { return *(Point3f*)(PointStream()+idx*3); }

	const uint32_t* ViewsStream(size_t idx) const
	{
		return pointViewsMemory.empty() ? nullptr : &pointViewsMemory[pointViewsOffsets[idx]];
	}

	const size_t ViewsStreamSize(size_t idx) const
	{
		return pointViewsSizes.empty() ? 0 : pointViewsSizes[idx];
	}

	const float* WeightsStream(size_t idx) const
	{
		return pointWeightsMemory.empty() ? nullptr : &pointWeightsMemory[pointWeightsOffsets[idx]];
	}

	const size_t WeightsStreamSize(size_t idx) const
	{
		return pointWeightsSizes.empty() ? 0 : pointWeightsSizes[idx];
	}

	void Release();

	void ReleaseWeights()
	{
		// swap-with-empty, NOT clear(): these are POD vectors, so clear() sets size
		// to 0 while keeping the whole capacity allocated -- i.e. a function named
		// Release that releases nothing. At 66M points the weights are ~1.3 GB
		// (offsets + sizes + the flat weight array). Same defect as Release() had.
		std::vector<uint32_t>().swap(pointWeightsOffsets);
		std::vector<uint32_t>().swap(pointWeightsSizes);
		std::vector<float>().swap(pointWeightsMemory);
	}

	void RemovePoint(IDX);

	typedef TOctree<std::vector<Point3f>,TPoint3<float>::Type,3> Octree;

	typedef IDX Index;
	typedef AABB3f Box;

	Box GetAABB() const;
	Box GetAABB(const Box& bound) const;
	Box GetAABB(unsigned minViews) const;
	TPoint3<float> GetCenter() const;

	size_t NumPoints() const { return pointsXYZ.size()/3; }

	inline bool IsEmpty() const { return pointsXYZ.empty(); }
	inline bool IsValid() const { return !pointViewsMemory.empty(); }
	inline size_t GetSize() const { return pointsXYZ.size()/3; }

	bool Load(const String& fileName);
	bool Save(const String& fileName, bool bLegacyTypes=false) const;
	bool SaveNViews(const String& fileName, uint32_t minViews, bool bLegacyTypes=false) const;

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template <class Archive>
	void save(Archive& ar, const unsigned int /*version*/) const {
		uint64_t n;

		n = (uint64_t)pointsXYZ.size();
		ar& n;
		if (n) ar.save_binary(pointsXYZ.data(), n * sizeof(pointsXYZ[0]));

		n = (uint64_t)pointViewsOffsets.size();
		ar& n;
		if (n) ar.save_binary(pointViewsOffsets.data(), n * sizeof(pointViewsOffsets[0]));

		n = (uint64_t)pointViewsSizes.size();
		ar& n;
		if (n) ar.save_binary(pointViewsSizes.data(), n * sizeof(pointViewsSizes[0]));

		n = (uint64_t)pointWeightsOffsets.size();
		ar& n;
		if (n) ar.save_binary(pointWeightsOffsets.data(), n * sizeof(pointWeightsOffsets[0]));

		n = (uint64_t)pointWeightsSizes.size();
		ar& n;
		if (n) ar.save_binary(pointWeightsSizes.data(), n * sizeof(pointWeightsSizes[0]));

		n = (uint64_t)pointViewsMemory.size();
		ar& n;
		if (n) ar.save_binary(pointViewsMemory.data(), n * sizeof(pointViewsMemory[0]));

		n = (uint64_t)pointWeightsMemory.size();
		ar& n;
		if (n) ar.save_binary(pointWeightsMemory.data(), n * sizeof(pointWeightsMemory[0]));

		n = (uint64_t)normalsXYZ.size();
		ar& n;
		if (n) ar.save_binary(normalsXYZ.data(), n * sizeof(normalsXYZ[0]));

		n = (uint64_t)colorsRGB.size();
		ar& n;
		if (n) ar.save_binary(colorsRGB.data(), n * sizeof(colorsRGB[0]));
	}

	template <class Archive>
	void load(Archive& ar, const unsigned int /*version*/) {
		uint64_t n;

		ar& n;
		pointsXYZ.resize((size_t)n);
		if (n) ar.load_binary(pointsXYZ.data(), n * sizeof(pointsXYZ[0]));

		ar& n;
		pointViewsOffsets.resize((size_t)n);
		if (n) ar.load_binary(pointViewsOffsets.data(), n * sizeof(pointViewsOffsets[0]));

		ar& n;
		pointViewsSizes.resize((size_t)n);
		if (n) ar.load_binary(pointViewsSizes.data(), n * sizeof(pointViewsSizes[0]));

		ar& n;
		pointWeightsOffsets.resize((size_t)n);
		if (n) ar.load_binary(pointWeightsOffsets.data(), n * sizeof(pointWeightsOffsets[0]));

		ar& n;
		pointWeightsSizes.resize((size_t)n);
		if (n) ar.load_binary(pointWeightsSizes.data(), n * sizeof(pointWeightsSizes[0]));

		ar& n;
		pointViewsMemory.resize((size_t)n);
		if (n) ar.load_binary(pointViewsMemory.data(), n * sizeof(pointViewsMemory[0]));

		ar& n;
		pointWeightsMemory.resize((size_t)n);
		if (n) ar.load_binary(pointWeightsMemory.data(), n * sizeof(pointWeightsMemory[0]));

		ar& n;
		normalsXYZ.resize((size_t)n);
		if (n) ar.load_binary(normalsXYZ.data(), n * sizeof(normalsXYZ[0]));

		ar& n;
		colorsRGB.resize((size_t)n);
		if (n) ar.load_binary(colorsRGB.data(), n * sizeof(colorsRGB[0]));
	}

	BOOST_SERIALIZATION_SPLIT_MEMBER()
	#endif

	//
	// All points in a cloud are of the format:
	// {xyz} {xyz, normal} {xyz, normal, color} {xyz, color}
	// 
	// If the cloud contains point views and point weights, each point "i"
	// will have set of pointViewsSizes[i] view indexes stored
	// at pointViewsMemory[pointViewsOffsets[i]].  Point weights are stored
	// similarly.
	// 
	std::vector<float> pointsXYZ;

	std::vector<uint32_t> pointViewsOffsets;  // Within pointViewsMemory
	std::vector<uint32_t> pointViewsSizes;
	std::vector<uint32_t> pointWeightsOffsets;// Within pointWeightsMemory
	std::vector<uint32_t> pointWeightsSizes;

	std::vector<uint32_t> pointViewsMemory;
	std::vector<float> pointWeightsMemory;

	std::vector<float> normalsXYZ;

	std::vector<uint8_t> colorsRGB;
};

/*----------------------------------------------------------------*/


struct IndexDist {
	IDX idx;
	REAL dist;

	inline IndexDist() : dist(REAL(FLT_MAX)) {}
	inline bool IsValid() const { return dist < REAL(FLT_MAX); }
};

struct IntersectRayPoints {
	typedef PointCloud::Octree Octree;
	typedef typename Octree::IDX_TYPE IDX;
	typedef TCone<REAL, 3> Cone3;
	typedef TConeIntersect<REAL, 3> Cone3Intersect;

	const PointCloudStreaming& pointcloud;
	const Cone3 cone;
	const Cone3Intersect coneIntersect;
	const unsigned minViews;
	IndexDist pick;

	IntersectRayPoints(const Octree& octree, const Ray3& _ray, const PointCloudStreaming& _pointcloud, unsigned _minViews)
		: pointcloud(_pointcloud), cone(_ray, D2R(REAL(0.5))), coneIntersect(cone), minViews(_minViews)
	{
		octree.Collect(*this, *this);
	}

	inline bool Intersects(const typename Octree::POINT_TYPE& center, typename Octree::Type radius) const {
		return coneIntersect(Sphere3(center.cast<REAL>(), REAL(radius) * SQRT_3));
	}

	void operator () (const IDX* idices, IDX size) {
		// test ray-point intersection and keep the closest
		FOREACHRAWPTR(pIdx, idices, size) {
			const PointCloud::Index idx(*pIdx);
			if (pointcloud.ViewsStreamSize(idx) < minViews)
				continue;
			const PointCloud::Point& X = pointcloud.Point(idx);
			REAL dist;
			if (coneIntersect.Classify(Cast<REAL>(X), dist) == VISIBLE) {
				ASSERT(dist >= 0);
				if (pick.dist > dist) {
					pick.dist = dist;
					pick.idx = idx;
				}
			}
		}
	}
};
/*----------------------------------------------------------------*/


typedef MVS_API float Depth;
typedef MVS_API Point3f Normal;
typedef MVS_API TImage<Depth> DepthMap;
typedef MVS_API TImage<Normal> NormalMap;
typedef MVS_API TImage<float> ConfidenceMap;
typedef MVS_API SEACAVE::cList<Depth,Depth,0> DepthArr;
typedef MVS_API SEACAVE::cList<DepthMap,const DepthMap&,2> DepthMapArr;
typedef MVS_API SEACAVE::cList<NormalMap,const NormalMap&,2> NormalMapArr;
typedef MVS_API SEACAVE::cList<ConfidenceMap,const ConfidenceMap&,2> ConfidenceMapArr;
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_POINTCLOUD_H_
