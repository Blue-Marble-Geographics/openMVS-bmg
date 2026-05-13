/*
* DepthMap.h
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

#ifndef _MVS_DEPTHMAP_H_
#define _MVS_DEPTHMAP_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "P2PUtils.h"
#include "PointCloud.h"
#include <boost/container/small_vector.hpp>
#include <boost/align.hpp>

// D E F I N E S ///////////////////////////////////////////////////

// NCC type used for patch-similarity computation during depth-map estimation
#define DENSE_NCC_DEFAULT 0
#define DENSE_NCC_FAST 1
#define DENSE_NCC_WEIGHTED 2
#define DENSE_NCC DENSE_NCC_WEIGHTED

// NCC score aggregation type used during depth-map estimation
#define DENSE_AGGNCC_NTH 0
#define DENSE_AGGNCC_MEAN 1
#define DENSE_AGGNCC_MIN 2
#define DENSE_AGGNCC_MINMEAN 3
#define DENSE_AGGNCC DENSE_AGGNCC_MINMEAN

// type of smoothness used during depth-map estimation
#define DENSE_SMOOTHNESS_NA 0
#define DENSE_SMOOTHNESS_FAST 1
#define DENSE_SMOOTHNESS_PLANE 2
#define DENSE_SMOOTHNESS DENSE_SMOOTHNESS_PLANE

// type of refinement used during depth-map estimation
#define DENSE_REFINE_ITER 0
#define DENSE_REFINE_EXACT 1
#define DENSE_REFINE DENSE_REFINE_ITER

// exp function type used during depth estimation
#define DENSE_EXP_DEFUALT EXP
#define DENSE_EXP_FAST FEXP<true> // ~10% faster, but slightly less precise
#define DENSE_EXP DENSE_EXP_DEFUALT

#undef SANITY_CHECKS
#define DEBUGME
#undef USE_ORIG
#define USE_INV_SQRT
#define USE_FAST_COMPARES
#undef MORE_ACCURATE_WEIGHTS
#define USE_FAST_SIN_COS
#undef USE_NN
#undef CLAMP_SCORES
#define USE_REMAP

#undef USE_FLOAT_SCORING_ACCURACY

#ifdef USE_FLOAT_SCORING_ACCURACY
using Calc_t = float;
using Vec2_t = Vec2f;
using Vec3_t = Vec3f;
using Matx13_t = cv::Matx13f;
using Matrix3x3_t = Matrix3x3f;
#else
using Calc_t = double;
using Vec2_t = Vec2;
using Vec3_t = Vec3;
using Matx13_t = cv::Matx13d;
using Matrix3x3_t = Matrix3x3;
#endif

#define ComposeDepthFilePath(i, e) MAKE_PATH(String::FormatString(("depth%04u." + String(e)).c_str(), i))


// S T R U C T S ///////////////////////////////////////////////////
namespace MVS {
	typedef TMatrix<uint8_t, 4, 1> ViewsID;
}
DEFINE_CVDATATYPE(MVS::ViewsID)

/* __m128 is ugly to write */
typedef __m128 v4sf;  // vector of 4 float (sse1)

typedef __m128i v4si; // vector of 4 int (sse2)

#if defined(MORE_ACCURATE_WEIGHTS) || defined(DPC_FASTER_SCORE_PIXEL_DETAIL2)
extern v4sf exp_ps(v4sf);
#endif

namespace MVS {

DECOPT_SPACE(OPTDENSE)

namespace OPTDENSE {
// configuration variables
enum DepthFlags {
	REMOVE_SPECKLES	= (1 << 0),
	FILL_GAPS		= (1 << 1),
	ADJUST_FILTER	= (1 << 2),
	OPTIMIZE		= (REMOVE_SPECKLES|FILL_GAPS)
};
extern unsigned nResolutionLevel;
extern unsigned nMaxResolution;
extern unsigned nMinResolution;
extern unsigned nSubResolutionLevels;
extern unsigned nMinViews;
extern unsigned nMaxViews;
extern unsigned nMinViewsFuse;
extern unsigned nMinViewsFilter;
extern unsigned nMinViewsFilterAdjust;
extern unsigned nMinViewsTrustPoint;
extern unsigned nNumViews;
extern unsigned nPointInsideROI;
extern bool bFilterAdjust;
extern bool bAddCorners;
extern bool bInitSparse;
extern bool bRemoveDmaps;
extern bool bDenseFuse;
extern float fViewMinScore;
extern float fViewMinScoreRatio;
extern float fMinArea;
extern float fMinAngle;
extern float fOptimAngle;
extern float fMaxAngle;
extern float fDescriptorMinMagnitudeThreshold;
extern float fDepthDiffThreshold;
extern float fNormalDiffThreshold;
extern float fPairwiseMul;
extern float fOptimizerEps;
extern int nOptimizerMaxIters;
extern unsigned nSpeckleSize;
extern unsigned nIpolGapSize;
extern int nIgnoreMaskLabel;
extern unsigned nOptimize;
extern unsigned nEstimateColors;
extern unsigned nEstimateNormals;
extern float fNCCThresholdKeep;
extern unsigned nEstimationIters;
extern unsigned nEstimationGeometricIters;
extern float fEstimationGeometricWeight;
extern unsigned nRandomIters;
extern unsigned nRandomMaxScale;
extern float fRandomDepthRatio;
extern float fRandomAngle1Range;
extern float fRandomAngle2Range;
extern float fRandomSmoothDepth;
extern float fRandomSmoothNormal;
extern float fRandomSmoothBonus;
} // namespace OPTDENSE
/*----------------------------------------------------------------*/

#ifdef DPC_FASTER_SAMPLING
constexpr int sRemapTbl[25] =
{
	0,
	5, 10, 15, 20,
	1, 2, 3, 4,
	6, 7, 8, 9,
	11, 12, 13, 14,
	16, 17, 18, 19,
	21, 22, 23, 24
};
#endif

enum { nSizeHalfWindow = 4 };

typedef TImage<ViewsID> ViewsMap;

template <int nTexels>
struct WeightedPatchFix {
#ifdef DPC_FASTER_SAMPLING
	float __declspec( align( 16 ) ) pixelWeights[nTexels-1];
	float __declspec( align( 16 ) ) pixelTempWeights[nTexels-1];
	float firstPixelWeight;
	float firstPixelTempWeight;
#else
	float weights[nTexels];
	float tempWeights[nTexels];
#endif
};

// Stored separately for efficiency.
struct WeightedPatchInfo {
	float sumWeights;
#ifdef DPC_FASTER_SAMPLING
	float invSumWeights;
#endif
	float normSq0;
};

struct Normal4
{
	Normal4(const Normal& n)
	{
		data = _SetN(n[0], n[1], n[2], 0.f);
	}

	Normal4(const _Data& v)
	{
		data = v;
	}

	float Dot3S(_Data v) const
	{
		// When v[3] and data[3] are guaranteed 0.
		v = _Mul(v, data);

		// 	https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
			const _Data vT = _mm_add_ps(v, _mm_movehl_ps(v, v));
		const _Data sum = _mm_add_ss(vT, _mm_shuffle_ps(vT, vT, 1));

		return _vFirst(sum);
	}


	Normal AsNormal() const
	{
		return Normal(_AsArray(data, 0), _AsArray(data, 1), _AsArray(data, 2));
	}

	_Data data;
};

struct Normal4D
{
	Normal4D(const Normal4& n)
	{
		vXY = _mm_cvtps_pd(n.data);
		vZ = _mm_set1_pd(_AsArray(n.data,2));
	}

	__m128d vXY;
	__m128d vZ;
};

struct Mat44
{
	_Data vRows[4];

	void SetIdentity()
	{
		vRows[0] = _SetN(1.f, 0.f, 0.f, 0.f);
		vRows[1] = _SetN(0.f, 1.f, 0.f, 0.f);
		vRows[2] = _SetN(0.f, 0.f, 1.f, 0.f);
		vRows[3] = _SetN(0.f, 0.f, 0.f, 1.f);
	}

	void SetRow(int row, _Data v)
	{
		vRows[row] = v;
	}

	_Data Mul44Vec3(_Data v) const noexcept
	{
		// Multiply 4x4 by Vec3
		const _Data vX = _Splat(v, 0);
		const _Data vY = _Splat(v, 1);
		const _Data vZ = _Splat(v, 2);

		const _Data vRow1 = _Mul(vRows[0], vX);
		const _Data vRow2 = _Mul(vRows[1], vY);
		const _Data vRow3 = _Mul(vRows[2], vZ);

		const _Data vRow12 = _Add(vRow1, vRow2);
		const _Data vRow23 = _Add(vRow3, vRows[3]);

		return _Add(vRow12, vRow23);
	}
};

struct MVS_API DepthData {
	struct ViewData {
		// JPB WIP OPT Keep these close for caching.
		Matrix3x3_t Hl; //
		__m128d vHl00_01;
		__m128d vHl02_22;
		__m128d vHl10_11;
		__m128d vHl20_21;
		__m128d vHl00_20;
		__m128d vHl01_21;
		Vec3_t Hm;      // constants during per-pixel loops
		__m128d vHm0Hm2;
		__m128d vHm0;
		__m128d vHm1;
		__m128d vHm2;
		Matrix3x3_t Hr; //
		__m128d vHr00_00;
		__m128d vHr00_11;
		__m128d vHr11_11;
		__m128d vHr02_02;
		__m128d vHr12_12;

		Image32F image; // image float intensities
		Image32F imageBig; // image float intensities

		float scale; // image scale relative to the reference image
		Camera camera; // camera matrix corresponding to this image
		Image* pImageData; // image data

		DepthMap depthMap; // known depth-map (optional)
		Camera cameraDepthMap; // camera matrix corresponding to the depth-map
		Matrix3x3f Tl; //
		Point3f Tm;    // constants during per-pixel geometric-consistent loops
		Matrix3x3f Tr; //
		Point3f Tn;    //
		Mat44 Tr4;
		Mat44 Tl4; //

		_Data mHm0Hm1Hm2;
		_Data mHl00Hl10Hl20;
		_Data mHl01Hl11Hl21;
		_Data mHl02Hl12Hl22;
		_Data mHr00Hr00Hr00;
		_Data mHr11Hr11Hr11;
		_Data mHr02Hr02Hr02;
		_Data mHr12Hr12Hr12;

		_Data mK02K12;
		_Data mInvK00K11;

		_Data mWindowShifted;
		
		inline void Init(const Camera& cameraRef) {
			mWindowShifted = _SetN( (float) (image.width()-2*nSizeHalfWindow-1), (float)(image.height()-2*nSizeHalfWindow-1), 0.f, 0.f);
			mK02K12 = _SetN((float) camera.K(0, 2), (float) camera.K(1, 2), 0.f, 0.f);
			mInvK00K11 = _SetN((float) (1./camera.K(0, 0)), (float) (1./camera.K(1, 1)), 0.f, 0.f);

			Hl = camera.K * camera.R * cameraRef.R.t();
			Hm = camera.K * camera.R * (cameraRef.C - camera.C);
			Hr = cameraRef.K.inv();
			if (!depthMap.empty()) {
				Tl = cameraDepthMap.K * cameraDepthMap.R * cameraRef.R.t();
				Tm = cameraDepthMap.K * cameraDepthMap.R * (cameraRef.C - cameraDepthMap.C);
				Tr = cameraRef.K * cameraRef.R * cameraDepthMap.R.t() * cameraDepthMap.GetInvK();
				Tn = cameraRef.K * cameraRef.R * (cameraDepthMap.C - cameraRef.C);

				// Tr4 is Tr/Tn for SIMD.
				Tr4.SetIdentity();
				Tr4.SetRow(0, _SetN(Tr(0,0), Tr(1,0), Tr(2,0), 0.f));
				Tr4.SetRow(1, _SetN(Tr(0,1), Tr(1,1), Tr(2,1), 0.f));
				Tr4.SetRow(2, _SetN(Tr(0,2), Tr(1,2), Tr(2,2), 0.f));
				Tr4.SetRow(3, _SetN(Tn[0], Tn[1], Tn[2], 0.f));

				// Tl4 is Tl/Tm for SIMD.
				Tl4.SetIdentity();
				Tl4.SetRow(0, _SetN(Tl(0,0), Tl(1,0), Tl(2,0), 0.f));
				Tl4.SetRow(1, _SetN(Tl(0,1), Tl(1,1), Tl(2,1), 0.f));
				Tl4.SetRow(2, _SetN(Tl(0,2), Tl(1,2), Tl(2,2), 0.f));
				Tl4.SetRow(3, _SetN(Tm[0], Tm[1], Tm[2], 0.f));
			}
			vHm0Hm2 = _mm_set_pd(Hm(2), Hm(0));
			vHm0 = _mm_set1_pd(Hm(0));
			vHm1 = _mm_set1_pd(Hm(1));
			vHm2 = _mm_set1_pd(Hm(2));

			vHl00_01 = _mm_set_pd(Hl(0,1), Hl(0,0));
			vHl10_11 = _mm_set_pd(Hl(1,1), Hl(1,0));
			vHl02_22 = _mm_set_pd(Hl(2,2), Hl(0,2));
			vHl20_21 = _mm_set_pd(Hl(2,1), Hl(2,0));
			vHl00_20 = _mm_set_pd(Hl(2,0), Hl(0,0));
			vHl01_21 = _mm_set_pd(Hl(2,1), Hl(0,1));

			vHr00_00 = _mm_set_pd(Hr(0,0), Hr(0,0));
			vHr00_11 = _mm_set_pd(Hr(1,1), Hr(0,0));
			vHr11_11 = _mm_set_pd(Hr(1,1), Hr(1,1));
			vHr02_02 = _mm_set_pd(Hr(0,2), Hr(0,2));
			vHr12_12 = _mm_set_pd(Hr(1,2), Hr(1,2));

			mHm0Hm1Hm2 = _SetN((float) Hm(0), (float)Hm(1), (float)Hm(2), 0.f);
			mHl00Hl10Hl20 = _SetN((float)Hl(0, 0), (float)Hl(1, 0), (float)Hl(2, 0), 0.f);
			mHl01Hl11Hl21 = _SetN((float)Hl(0, 1), (float)Hl(1, 1), (float)Hl(2, 1), 0.f);
			mHl02Hl12Hl22 = _SetN((float)Hl(0, 2), (float)Hl(1, 2), (float)Hl(2, 2), 0.f);
			mHr00Hr00Hr00 = _SetN((float)Hr(0, 0), (float)Hr(0, 0), (float)Hr(0, 0), 0.f);
			mHr11Hr11Hr11 = _SetN((float)Hr(1, 1), (float)Hr(1, 1), (float)Hr(1, 1), 0.f);
			mHr02Hr02Hr02 = _SetN((float)Hr(0, 2), (float)Hr(0, 2), (float)Hr(0, 2), 0.f);
			mHr12Hr12Hr12 = _SetN((float)Hr(1, 2), (float)Hr(1, 2), (float)Hr(1, 2), 0.f);
		}

		inline IIndex GetID() const {
			return pImageData->ID;
		}
		inline IIndex GetLocalID(const ImageArr& images) const {
			return (IIndex)(pImageData - images.begin());
		}

		static bool NeedScaleImage(float scale) {
			ASSERT(scale > 0);
			return ABS(scale-1.f) >= 0.15f;
		}
		template <typename IMAGE>
		static bool ScaleImage(const IMAGE& image, IMAGE& imageScaled, float scale) {
			if (!NeedScaleImage(scale))
				return false;
			cv::resize(image, imageScaled, cv::Size(), scale, scale, scale>1?cv::INTER_CUBIC:cv::INTER_AREA);
			return true;
		}
	};
	typedef CLISTDEF2IDX(ViewData,IIndex) ViewDataArr;

	ViewDataArr images; // array of images used to compute this depth-map (reference image is the first)
	ViewScoreArr neighbors; // array of all images seeing this depth-map (ordered by decreasing importance)
	IndexArr points; // indices of the sparse 3D points seen by the this image
	BitMatrix mask; // mark pixels to be ignored
	DepthMap depthMap; // depth-map
	NormalMap normalMap; // normal-map in camera space
	ConfidenceMap confMap; // confidence-map
	ViewsMap viewsMap; // view-IDs map (indexing images vector starting after first view)
	float dMin, dMax; // global depth range for this image
	cv::Size size; // image size used to estimate this depth-map
	unsigned references; // how many times this depth-map is referenced (on 0 can be safely unloaded)
	CriticalSection cs; // used to count references

	inline DepthData() : references(0) {}
	DepthData(const DepthData&);

	inline void ReleaseImages() {
		for (ViewData& image: images) {
			image.image.release();
			image.depthMap.release();
		}
	}
	inline void Release() {
		depthMap.release();
		normalMap.release();
		confMap.release();
		viewsMap.release();
	}

	inline bool IsValid() const {
		return !images.IsEmpty();
	}
	inline bool IsEmpty() const {
		return depthMap.empty();
	}

	const ViewData& GetView() const { return images.front(); }
	const Camera& GetCamera() const { return GetView().camera; }

	void GetNormal(const ImageRef& ir, Point3f& N, const TImage<Point3f>* pPointMap=NULL) const;
	void GetNormal(const Point2f& x, Point3f& N, const TImage<Point3f>* pPointMap=NULL) const;

	void ApplyIgnoreMask(const BitMatrix&);

	bool Save(const String& fileName) const;
	bool Load(const String& fileName, unsigned flags=15);

	unsigned GetRef();
	unsigned IncRef(const String& fileName);
	unsigned DecRef();

	size_t GetMemorySize() const;

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ASSERT(IsValid());
		ar & depthMap;
		ar & normalMap;
		ar & confMap;
		ar & viewsMap;
		ar & dMin;
		ar & dMax;
	}
	#endif
};
typedef MVS_API CLISTDEFIDX(DepthData,IIndex) DepthDataArr;
/*----------------------------------------------------------------*/

struct MVS_API DepthEstimator {
	enum { nSizeWindow = nSizeHalfWindow*2+1 };
	enum { nSizeStep = 2 };
	enum { TexelChannels = 1 };
	enum { nTexels = SQUARE((nSizeHalfWindow*2+nSizeStep)/nSizeStep)*TexelChannels };

	enum ENDIRECTION {
		LT2RB = 0,
		RB2LT,
		DIRS
	};

	typedef TPoint2<uint16_t> MapRef;
	typedef CLISTDEF0(MapRef) MapRefArr;

	typedef Eigen::Matrix<float,nTexels,1> TexelVec;
#pragma pack(push, 1)
	struct NeighborData {
		ImageRef x;
		Depth depth;
		Normal normal;
	};
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	struct PixelEstimate {
		Depth depth;
		Normal normal;
	};
	#endif
#pragma pack(pop)

	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	typedef WeightedPatchFix<nTexels> Weight;
	typedef CLISTDEFIDX(Weight,int) WeightMap;
	typedef CLISTDEFIDX(WeightedPatchInfo,int) WeightMapInfo_t;
	#endif

	//int idxPixel; // 
#ifndef DPC_EXTENDED_OMP_THREADING
	volatile Thread::safe_t& idxPixel; // current image index to be processed
#endif
#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_NA
	CLISTDEF0IDX(NeighborData,IIndex) neighbors; // neighbor pixels coordinates to be processed
	#else
	// JPB WIP BUG CLISTDEF0IDX(ImageRef,IIndex) neighbors; // neighbor pixels coordinates to be processed
	#endif
	float mLowResDepth;
	_Data vX0;
	ImageRef x0;	  // constants during one pixel loop
	Weight pWeightMap;
	WeightedPatchInfo pWeightMap0Info;
#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL
	_Data x0ULPatchCorner0;
	_Data x0ULPatchCorner1;
#else
	Vec2_t x0ULPatchCorner;
	__m128d x0ULPatchCorner0;
	__m128d x0ULPatchCorner1;
#endif
	Depth lowResDepth;
	float normSq0;	  //
	_Data vFactorDeltaDepth;
	_Data vOneMinusFactorDeltaDepth;
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	TexelVec texels0; //
	#endif
	#if DENSE_NCC == DENSE_NCC_DEFAULT
	TexelVec texels1;
	#endif
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH || DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	// JPB WIP BUG FloatArr scores;
	#else
	// JPB WIP BUG Eigen::VectorXf scores;
	#endif
	#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
	_Data plane; // plane defined by current depth and normal estimate
#endif
	DepthMap& depthMap0;
	NormalMap& normalMap0;
	ConfidenceMap& confMap0;
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	//WeightMap& weightMap0;
	//WeightMapInfo_t& weightMap0Info;
	#endif
	DepthMap lowResDepthMap;

	//const unsigned nIteration; // current PatchMatch iteration
	const DepthData::ViewDataArr images; // neighbor images used

	struct ScoreResultsSOA_t
	{
		void Allocate(int numItems)
		{
			size_t numGroups = NumGroupsForSize(numItems);
			size_t numSIMDItems = numGroups*GROUP_SIZE;
			size_t numBytesForScores = (sizeof(*pScores) * numSIMDItems);
			size_t numBytesForNums = (sizeof(*pNums) * numSIMDItems);
			size_t numBytesForNrmSqs = (sizeof(*pNrmSqs) * numSIMDItems);
			size_t numBytesForVs = (sizeof(*pVs) * numSIMDItems);
			size_t numBytesForSum = (sizeof(*pSum) * numSIMDItems);

			size_t numBytesForNum = (sizeof(*pNum) * numSIMDItems);

			size_t totalBytesNeeded =
				numBytesForScores
				+ numBytesForNums
				+ numBytesForNrmSqs
				+ numBytesForVs
				+ numBytesForSum
				+ numBytesForNum;

			p.resize(totalBytesNeeded);

			uint8_t* tmp = p.data();
			pScores = reinterpret_cast<float*>(tmp); tmp += numBytesForScores;
			pNums = reinterpret_cast<float*>(tmp); tmp += numBytesForNums;
			pNrmSqs = reinterpret_cast<float*>(tmp); tmp += numBytesForNrmSqs;
			pVs = reinterpret_cast<_Data*>(tmp); tmp += numBytesForVs;
			pSum = reinterpret_cast<float*>(tmp); tmp += numBytesForSum;
			pNum = reinterpret_cast<float*>(tmp); tmp += numBytesForNum;
		}

		std::vector<uint8_t, boost::alignment::aligned_allocator<uint8_t, ALIGN_SIZE*4>> p;
		size_t numScoreResults;
		float* pScores;
		float* pNums;
		float* pNrmSqs;
		_Data* pVs;
		float* pSum;
		float* pNum;
	};
	ScoreResultsSOA_t scoreResults;
	SEACAVE::Random rnd;
	const DepthData::ViewData& image0;
	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	const Image64F& image0Sum; // integral image used to fast compute patch mean intensity
	#endif
	const MapRefArr& coords;
	const Image8U::Size size;
	const Depth dMin, dMax;
	const Depth dMinSqr, dMaxSqr;
	const ENDIRECTION dir;
	#if DENSE_AGGNCC == DENSE_AGGNCC_NTH || DENSE_AGGNCC == DENSE_AGGNCC_MINMEAN
	const IDX idxScore;
	#endif

	struct ImageInfo_t
	{
		explicit ImageInfo_t(const DepthData::ViewData& image) :
			image(image),
			data(image.image.data),
			data2(image.imageBig.data),
			rowByteStride((int) image.imageBig.row_stride()),
			widthMinus2((float) (image.image.width() - 2)),
			heightMinus2((float) (image.image.height() - 2))
		{
			if (image.image.elem_stride() != 4) {
				throw std::runtime_error("Unsupported");
			}

			if (!image.depthMap.empty()) {
				depthMapData = image.depthMap.data,
				depthMapElemStride = image.depthMap.elem_stride();
				depthMapRowStride = image.depthMap.row_stride();
				depthMapWidthMinus2 = (float) (image.depthMap.width() - 2);
				depthMapHeightMinus2 = (float) (image.depthMap.height() - 2);
			} else {
				depthMapData = nullptr;
			}
		}

		const DepthData::ViewData& image;
		uchar* data;
		uchar* data2;
		int rowByteStride;
		float widthMinus2;
		float heightMinus2;

		// Optional
		const uchar* depthMapData;
		size_t depthMapElemStride;
		size_t depthMapRowStride;
		float depthMapWidthMinus2;
		float depthMapHeightMinus2;
	};

	struct ScoreHelper
	{
		ScoreHelper(
			const float scoreFactor,
			const DepthData::ViewDataArr& images,
			bool lowResDepthMapEmpty = true
		) :
			mVScoreFactor(_Set(scoreFactor)),
			mLowResDepthMapEmpty(lowResDepthMapEmpty)
		{
			imageInfo.reserve(images.size());
			for (const auto& i : images) {
				imageInfo.emplace_back(i);
			}
		}

		void Detail(
			Depth depth,
#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL
			const _Data mat,
#else
			const Matx13_t& mat,
#endif
			const _Data& depthMapPt,
			float lowResDepth
		)
		{
#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL
			mMatXYZ = _SetN(_AsArray(mat, 0), _AsArray(mat, 1), _AsArray(mat, 2), 0.f);
#else
			mMat01 = _mm_set_pd(mat(1), mat(0));
			mMat0 = _mm_set1_pd(mat(0));
			mMat1 = _mm_set1_pd(mat(1));
			mMat2 = _mm_set1_pd(mat(2));
#endif
			mDepthMapPt = depthMapPt;
			mLowResDepth = lowResDepth;

			if (!mLowResDepthMapEmpty) {
				if (lowResDepth > 0.f) {
					float tmp = FastAbsS(lowResDepth-depth)/lowResDepth;
					mDeltaDepth = FastMinS(tmp, 0.5f);
				}
			}
		}

		std::vector<ImageInfo_t, boost::alignment::aligned_allocator<ImageInfo_t, ALIGN_SIZE>> imageInfo;

#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL
		_Data mMatXYZ;
#else
		__m128d mMat01;
		__m128d mMat0;
		__m128d mMat1;
		__m128d mMat2;
#endif
		_Data mVScoreFactor;
		_Data mDepthMapPt; // w component undefined.

		// Always set before scoring.
		const uchar* mAddr;
		_Data mFirst;
		_Data mVX;
#if 1 // JPB WIP BUG Precision
		_Data mVBasisH;
			_Data mVBasisHX4;
		_Data mVBasisHY4;
		_Data mVBasisHZ4;	
			_Data mVBasisV;	
				_Data mVBasisVX4;
		_Data mVBasisVY4;
		_Data mVBasisVZ4;	
#endif
		_Data mVTopRowX4;
		_Data mVTopRowY4;
		_Data mVTopRowZ4;
		_Data mvBasisVX;
		_Data mvBasisVY;
		_Data mvBasisVZ;
		_Data mvBasisVX4;
		_Data mvBasisVY4;
		_Data mvBasisVZ4;
		_Data mVLeftColX4;
		_Data mVLeftColY4;
#ifdef DPC_FASTER_SAMPLING_USE_INV_Z
		_Data mInvVLeftColZ4;
#else
		_Data mVLeftColZ4;
#endif
		_Data mVBotRowX4;
		_Data mVBotRowY4;
		_Data mVBotRowZ4;

		Depth mLowResDepth;
		float mDeltaDepth;

		mutable bool mLowResDepthMapEmpty;
	};
	ScoreHelper sh;	// Order dependency on images

	DepthEstimator(
		unsigned nIter,
		DepthData& _depthData0,
#ifndef DPC_EXTENDED_OMP_THREADING
		volatile Thread::safe_t& _idx,
#endif
#if DENSE_NCC == DENSE_NCC_WEIGHTED
		#else
		const Image64F& _image0Sum,
		#endif
		const MapRefArr& _coords);

	bool PreparePixelPatch(const ImageRef& x)
	{
		const _DataI vUlxy = _CastIF(_mm_loadl_pi(vX0, (__m64*) &x.x)); // x0.x, x0.y, xx, xx
		const _Data vUlxyAsF = _ConvertFI(vUlxy);
		constexpr _Data vHalfWindowSize { nSizeHalfWindow, nSizeHalfWindow, 0.f, 0.f };
		_Data vZero = _SetZero();
		_Data vCornerXY = _Sub(vUlxyAsF, vHalfWindowSize);
		constexpr _DataI vMask {
			{ -1,-1,-1,-1, -1,-1,-1,-1, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }
		};
		vCornerXY = _And(vCornerXY, _CastFI(vMask));
		const _Data vNegative = _CmpLT(vCornerXY, vZero);
		const _Data vPositive = _CmpGT(vCornerXY, image0.mWindowShifted);
		_mm_storel_pi((__m64*) &x0, _CastFI(vUlxy));
#if 0
		// center a patch of given size on the segment
		int ulx = x.x-nSizeHalfWindow;
		int uly = x.y-nSizeHalfWindow;

		const bool horizontalInWindow = ( (unsigned)( ulx ) ) < ( (unsigned)image0.image.width()-2*nSizeHalfWindow );
		const bool verticalInWindow = ( (unsigned)( uly ) ) < ( (unsigned)image0.image.height()-2*nSizeHalfWindow );
		x0 = x;
#endif

#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL
		x0ULPatchCorner0 = _Set(_AsArray(vCornerXY, 0)); // (float)ulx);
		x0ULPatchCorner1 = _Set(_AsArray(vCornerXY, 1)); // (float)uly);
#else
		x0ULPatchCorner = Vec2_t((Calc_t)( _AsArray(vCornerXY, 0) ), (Calc_t)( _AsArray(vCornerXY, 1) ));
		x0ULPatchCorner0 = _mm_set1_pd(x0ULPatchCorner[0]);
		x0ULPatchCorner1 = _mm_set1_pd(x0ULPatchCorner[1]);
#endif
#if 1
		return _mm_movemask_ps(_Or(vNegative, vPositive)) == 0;
#else
		return horizontalInWindow && verticalInWindow;
#endif

	}

		// fetch the patch pixel values in the main image
	template<bool HasLowResDepthMap>
	bool FillPixelPatch()
	{
#ifndef DPC_FASTER_SAMPLING
		Weight& w = pWeightMap;
		auto& wpi = pWeightMap0Info;

		wpi.sumWeights = 0;
		wpi.normSq0 = 0;
		int n = 0;
		const float colCenter = image0.image(x0);
		for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep) {
			for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep) {
				wpi.normSq0 +=
					(w.tempWeights[n] = image0.image(x0.y+i, x0.x+j)) *
					(w.weights[n] = GetWeight(ImageRef(j, i), colCenter));
				wpi.sumWeights += w.weights[n];
				++n;
			}
		}
		ASSERT(n == nTexels);
		const float tm(wpi.normSq0/wpi.sumWeights);
		wpi.normSq0 = 0;
		n = 0;
		do {
			const float t(w.tempWeights[n] - tm);
			wpi.normSq0 += (w.tempWeights[n] = w.weights[n] * t) * t;
		} while (++n < nTexels);
	normSq0 = wpi.normSq0;
#else
		extern float firstSpatial;
		extern ImageRef __declspec( align( 16 ) ) sRemapImageRef[];
		extern int __declspec( align( 16 ) ) sImageOffsets[];
		extern float __declspec( align( 16 ) ) swSpatials[];

#if DENSE_NCC != DENSE_NCC_WEIGHTED
		STATIC_ASSERT(0); // Unsupported
		const float mean(GetImage0Sum(x0)/nTexels);
		normSq0 = 0;
		float* pTexel0 = texels0.data();
		for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep)
			for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep)
				normSq0 += SQUARE(*pTexel0++ = image0.image(x0.y+i, x0.x+j)-mean);
#else
		// JPB WIP BUG Compare needed?
		Weight& w = pWeightMap;
		WeightedPatchInfo& wpi = pWeightMap0Info;
		if (1) { //wpi.normSq0 == 0) {
			ASSERT(nTexels <= 32); // JPB WIP
			float firstColor;
			float firstPixelTempWeight;

			wpi.normSq0 = 0;
			float sumWeights = 0.f;
			_Data vSumWeights41 = _SetZero();
			_Data vSumWeights42 = _SetZero();
			_Data vNormSqSum41 = _SetZero();
			_Data vNormSqSum42 = _SetZero();

			firstPixelTempWeight = firstColor = image0.image(x0 + sRemapImageRef[0]);

			constexpr float sigmaColor(-1.f/( 2.f*SQUARE(0.1f) ));
			constexpr _Data vSigmaColor ={ sigmaColor, sigmaColor, sigmaColor, sigmaColor };
			const _Data vColCenter = _Set(image0.image.pix(x0));

			// Manually unfold by 4 to break dependency chains.
			int i = 0;
			int numGroups = (nTexels-1) / (GROUP_SIZE*2);
			int baseImageIndex = x0.y * ((int)image0.image.step.p[0]/sizeof(float) /* stride */) + x0.x;
			while (numGroups--) {
				const int i0 = baseImageIndex + sImageOffsets[i];
				const int i1 = baseImageIndex + sImageOffsets[i+1];
				const int i2 = baseImageIndex + sImageOffsets[i+2];
				const int i3 = baseImageIndex + sImageOffsets[i+3];
				const int i4 = baseImageIndex + sImageOffsets[i+4];
				const int i5 = baseImageIndex + sImageOffsets[i+5];
				const int i6 = baseImageIndex + sImageOffsets[i+6];
				const int i7 = baseImageIndex + sImageOffsets[i+7];
				
				// color weight [0..1]
				const float pix0 = image0.image.pix(i0);
				const float pix1 = image0.image.pix(i1);
				const float pix2 = image0.image.pix(i2);
				const float pix3 = image0.image.pix(i3);
				const float pix4 = image0.image.pix(i4);
				const float pix5 = image0.image.pix(i5);
				const float pix6 = image0.image.pix(i6);
				const float pix7 = image0.image.pix(i7);
				
				const _Data vPixelTempWeights1 = _SetN(pix0, pix1, pix2, pix3);
				const _Data vPixelTempWeights2 = _SetN(pix4, pix5, pix6, pix7);
				_Data vColor1 = vPixelTempWeights1;
				_Data vColor2 = vPixelTempWeights2;

				vColor1 = _Sub(vColor1, vColCenter);
				vColor2 = _Sub(vColor2, vColCenter);
				vColor1 = _Mul(vColor1, vColor1);
				vColor2 = _Mul(vColor2, vColor2);
				vColor1 = _Mul(vColor1, vSigmaColor);
				vColor2 = _Mul(vColor2, vSigmaColor);
#ifdef MORE_ACCURATE_WEIGHTS
				// exp(X+Y) = exp(X)*exp(Y) exp(vColor1)*exp(swSpatials[i])
				// = exp(vColor1)*C
				_Data vPixelWeights1 = exp_ps(_Add(vColor1, _LoadA(&swSpatials[i])));
				_Data vPixelWeights2 = exp_ps(_Add(vColor2, _LoadA(&swSpatials[i+GROUP_SIZE])));
#else
				const _Data vPixelWeights1 = FastExp(_Add(vColor1, _LoadA(&swSpatials[i])));
				const _Data vPixelWeights2 = FastExp(_Add(vColor2, _LoadA(&swSpatials[i+GROUP_SIZE])));
#endif
				const _Data vNormSq01 = _Mul(vPixelWeights1, vPixelTempWeights1);
				const _Data vNormSq02 = _Mul(vPixelWeights2, vPixelTempWeights2);

				_StoreA(&w.pixelWeights[i], vPixelWeights1);
				_StoreA(&w.pixelWeights[i+GROUP_SIZE], vPixelWeights2);
				vSumWeights41 = _Add(vSumWeights41, vPixelWeights1);
				vSumWeights42 = _Add(vSumWeights42, vPixelWeights2);
				vNormSqSum41 = _Add(vNormSqSum41, vNormSq01);
				vNormSqSum42 = _Add(vNormSqSum42, vNormSq02);
				_StoreA(&w.pixelTempWeights[i], vPixelTempWeights1);
				_StoreA(&w.pixelTempWeights[i+GROUP_SIZE], vPixelTempWeights2);
				i += GROUP_SIZE*2;
			}

			vSumWeights41 = _Add(vSumWeights41, vSumWeights42);
			vNormSqSum41 = _Add(vNormSqSum41, vNormSqSum42);

			// The first is left over.
			const float pixelTempWeight = firstPixelTempWeight;
			const float color = firstColor - _vFirst(vColCenter);
			const float tmp = color*color*sigmaColor+firstSpatial;
#ifdef MORE_ACCURATE_WEIGHTS
			const float pixelWeight = _vFirst(exp_ps(_Set(tmp)));
#else
			const float pixelWeight = _vFirst(BetterFastExpSse(_Set(tmp)));
#endif
			auto normSq0 = pixelWeight * pixelTempWeight;
			w.firstPixelWeight = pixelWeight;
			sumWeights += pixelWeight;
			wpi.normSq0 += normSq0;
			w.firstPixelTempWeight = pixelTempWeight;

			wpi.normSq0 += FastHsumS(vNormSqSum41);
			sumWeights += FastHsumS(vSumWeights41);

			const float tm = wpi.normSq0/sumWeights;
			wpi.normSq0 = 0;

			i = 0;
			numGroups = (nTexels-1) / (GROUP_SIZE*2);
			const _Data vTm = _Set(tm);
			_Data vNormSq0 = _SetZero();
			_Data vNormSq01 = _SetZero();
			while (numGroups--) {
				const _Data ptw1 = _LoadA(&w.pixelTempWeights[i]);
				const _Data ptw2 = _LoadA(&w.pixelTempWeights[i+GROUP_SIZE]);

				const _Data pw1 = _LoadA(&w.pixelWeights[i]);
				const _Data pw2 = _LoadA(&w.pixelWeights[i+GROUP_SIZE]);

				const _Data vT1 = _Sub(ptw1, vTm);
				const _Data vT2 = _Sub(ptw2, vTm);

				const _Data vPtw1 = _Mul(pw1, vT1);
				const _Data vPtw2 = _Mul(pw2, vT2);

				const _Data vPtw1Sq = _Mul(vPtw1, vT1);
				const _Data vPtw2Sq = _Mul(vPtw2, vT2);

				_StoreA(&w.pixelTempWeights[i], vPtw1);
				_StoreA(&w.pixelTempWeights[i+GROUP_SIZE], vPtw2);

				vNormSq0 = _Add(vNormSq0, vPtw1Sq);
				vNormSq01 = _Add(vNormSq01, vPtw2Sq);

				//const _Data vT = _Sub(_Load(&w.pixelTempWeights[i]), vTm);
				//const _Data vPtw = _Mul(_Load(&w.pixelWeights[i]), vT);
				//_StoreA(&w.pixelTempWeights[i], vPtw);
				//vNormSq0 = _Add(vNormSq0, _Mul(vPtw, vT));
				i += GROUP_SIZE*2;
			}

			vNormSq0 = _Add(vNormSq0, vNormSq01);

			const float t = w.firstPixelTempWeight - tm;
			wpi.normSq0 += ( w.firstPixelTempWeight = w.firstPixelWeight * t ) * t;
		
			wpi.normSq0 += FastHsumS(vNormSq0);

			wpi.sumWeights = sumWeights;
			wpi.invSumWeights = 1.f / sumWeights;
		}
#endif

		normSq0 = wpi.normSq0;
#endif // DPC_FASTER_SAMPLING

		// JPB Notice this differs from the one normally used "smoothSigmaDepth".
		constexpr float smoothSigmaDepthForDepthCalc(-1.f / ( 1.f * 0.02f )); // 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less

		if (
			(normSq0 < thMagnitudeSq)
			&& constexpr(HasLowResDepthMap) 
			&& ( sh.mLowResDepthMapEmpty || lowResDepthMap.pix(x0) <= 0 )
		)
			return false;

		// X0 = image0.camera.TransformPointI2C(Cast<REAL>(x0));
		float x = ((float) x0.x - (float) image0.camera.K(0, 2))/(float) image0.camera.K(0, 0);
		float y = ((float) x0.y - (float) image0.camera.K(1, 2))/(float) image0.camera.K(1, 1);
		float z = 1.f;

		const _DataI vX0Raw = _CastIF(_mm_loadl_pi(vX0, (__m64*) &x0.x)); // x0.x, x0.y, xx, xx
		const _Data vX0AsFloat = _ConvertFI(vX0Raw);
		const _Data vX0AsFloat2 = _Sub(vX0AsFloat, image0.mK02K12);
		vX0 = _Mul(vX0AsFloat2, image0.mInvK00K11);
		_AsArray(vX0, 2) = 1.f;
		_AsArray(vX0, 3) = 0.f; // JPB WIP BUG Remove this debugging
		// vX0 is 0

		if constexpr( HasLowResDepthMap ) {
			mLowResDepth = lowResDepthMap.pix(x0);
			if (mLowResDepth > 0.f) {
				constexpr _Data vOne = { 1.f, 1.f, 1.f, 1.f };
				// vFactorDeltaDepth appears very sensitive to error.  Use the highest precision exp.
#ifdef DPC_FASTER_SCORE_PIXEL_DETAIL2
				const float tmp = normSq0 * smoothSigmaDepthForDepthCalc;
				v4sf vTmp = _Set(tmp);
				vFactorDeltaDepth = exp_ps(vTmp);
#else
				const float tmp = DENSE_EXP(normSq0 * smoothSigmaDepthForDepthCalc);
				vFactorDeltaDepth = _Set(tmp);
#endif
				vOneMinusFactorDeltaDepth = _Sub(vOne, vFactorDeltaDepth);
			}
		}	else {
			mLowResDepth = 0.f;
		}

		return true;
	}

	bool IsScorable(const DepthData::ViewData& image1);
	bool IsScorable2(const ImageInfo_t& image1);
	bool IsScorable3(const ImageInfo_t& image1) noexcept;

	float ScorePixelImageOrig(ScoreResultsSOA_t& scoreResults,
		const DepthData::ViewData& image1, Depth depth, const Normal& normal);

	struct SampleInfo
	{
		float sum;
		float sumSq;
		float num4;
	};

	void GatherSampleInfo(
		const ImageInfo_t& imageInfo,
		float& sumResult,
		float& sumSqResult,
		float& numResult
	) const noexcept;

	bool ScorePixelImage(const ImageInfo_t& imageInfo) noexcept;

	float ScorePixel(Depth depth, const Normal4& normal);
	_Data CalculateScoreFactor(
		_Data normal,
		float depth,
		size_t numNeighbors,
		const _Data* __restrict neighborsCloseX,
		const _Data* __restrict neighborsCloseNormals
	);
	void ProcessPixel(IDX idx);
	Depth InterpolatePixel(const ImageRef&, Depth, const Normal&) const;
	#if DENSE_SMOOTHNESS == DENSE_SMOOTHNESS_PLANE
	void InitPlane(Depth, const Normal&);
	#endif
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	PixelEstimate PerturbEstimate(const PixelEstimate&, float perturbation);
	#endif

	#if DENSE_NCC != DENSE_NCC_WEIGHTED
	inline float GetImage0Sum(const ImageRef& p) const {
		const ImageRef p0(p.x-nSizeHalfWindow, p.y-nSizeHalfWindow);
		const ImageRef p1(p0.x+nSizeWindow, p0.y);
		const ImageRef p2(p0.x, p0.y+nSizeWindow);
		const ImageRef p3(p1.x, p2.y);
		return (float)(image0Sum(p3) - image0Sum(p2) - image0Sum(p1) + image0Sum(p0));
	}
	#endif

	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	float GetWeight(const ImageRef& x, float center) const {
		// color weight [0..1]
		const float sigmaColor(-1.f/(2.f*SQUARE(0.1f)));
		const float wColor(SQUARE(image0.image(x0+x)-center) * sigmaColor);
		// spatial weight [0..1]
		const float sigmaSpatial(-1.f/(2.f*SQUARE((int)nSizeHalfWindow-1)));
		const float wSpatial(float(SQUARE(x.x) + SQUARE(x.y)) * sigmaSpatial);
		return DENSE_EXP(wColor+wSpatial);
	}
	#endif

	static inline Point3 ComputeRelativeC(const DepthData& depthData) {
		return depthData.images[1].camera.R*(depthData.images[0].camera.C-depthData.images[1].camera.C);
	}
	static inline Matrix3x3 ComputeRelativeR(const DepthData& depthData) {
		RMatrix R;
		ComputeRelativeRotation(depthData.images[0].camera.R, depthData.images[1].camera.R, R);
		return R;
	}

	// generate random depth and normal
	inline Depth RandomDepth(SEACAVE::Random& rnd, Depth dMinSqr, Depth dMaxSqr) {
		ASSERT(dMinSqr > 0 && dMinSqr < dMaxSqr);
		return SQUARE(rnd.randomRange(dMinSqr, dMaxSqr));
	}
	inline Normal RandomNormal(SEACAVE::Random& rnd, const Point3f& viewRay) {
		Normal normal;
		Dir2Normal(Point2f(rnd.randomRange(FD2R(0.f),FD2R(180.f)), rnd.randomRange(FD2R(90.f),FD2R(180.f))), normal);
		ASSERT(ISEQUAL(norm(normal), 1.f));
		return normal.dot(viewRay) > 0 ? -normal : normal;
	}

	// adjust normal such that it makes at most 90 degrees with the viewing angle
	inline void CorrectNormal(Normal& normal) const {
		const Normal viewDir(_AsArray(vX0, 0), _AsArray(vX0, 1), _AsArray(vX0,2));
		const float cosAngLen(normal.dot(viewDir));
		if (cosAngLen >= 0)
			normal = RMatrixBaseF(normal.cross(viewDir), MINF((ACOS(cosAngLen/norm(viewDir))-FD2R(90.f))*1.01f, -0.001f)) * normal;
		ASSERT(ISEQUAL(norm(normal), 1.f));
	}

#if 1// New version
	static bool ImportIgnoreMask(const Image&, const cv::Size&, uint8_t nIgnoreMaskLabel, BitMatrix&, Image8U* =NULL);
#else
	static bool ImportIgnoreMask(const Image&, const Image8U::Size&, BitMatrix&, uint16_t nIgnoreMaskLabel);
#endif
	static void MapMatrix2ZigzagIdx(const Image8U::Size& size, DepthEstimator::MapRefArr& coords, const BitMatrix& mask, int rawStride=16);

	const float smoothBonusDepth, smoothBonusNormal;
	const float smoothSigmaDepth, smoothSigmaNormal;
	const float thMagnitudeSq;
	const float angle1Range, angle2Range;
	const float thConfSmall, thConfBig, thConfRand;
	const float thRobust;
	#if DENSE_REFINE == DENSE_REFINE_EXACT
	const float thPerturbation;
	#endif
	static const float scaleRanges[12];
};
/*----------------------------------------------------------------*/


// Tools
#if 1 // New version
// Tools
bool TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly=false);
bool TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloud& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly=false);
#else
bool TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloudStreaming& pointcloud, const IndexArr& points,
	DepthMap& depthMap, NormalMap& normalMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly=false);
bool TriangulatePoints2DepthMap(
	const DepthData::ViewData& image, const PointCloudStreaming& pointcloud, const IndexArr& points,
	DepthMap& depthMap, Depth& dMin, Depth& dMax, bool bAddCorners, bool bSparseOnly=false);
#endif

// Robustly estimate the plane that fits best the given points
MVS_API unsigned EstimatePlane(const Point3Arr&, Plane&, double& maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MVS_API unsigned EstimatePlaneLockFirstPoint(const Point3Arr&, Plane&, double& maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MVS_API unsigned EstimatePlaneTh(const Point3Arr&, Plane&, double maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MVS_API unsigned EstimatePlaneThLockFirstPoint(const Point3Arr&, Plane&, double maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MATH_API int OptimizePlane(Planed& plane, const Eigen::Vector3d* points, size_t size, int maxIters, double threshold);
// same for float points
MATH_API unsigned EstimatePlane(const Point3fArr&, Planef&, double& maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MATH_API unsigned EstimatePlaneLockFirstPoint(const Point3fArr&, Planef&, double& maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MATH_API unsigned EstimatePlaneTh(const Point3fArr&, Planef&, double maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MATH_API unsigned EstimatePlaneThLockFirstPoint(const Point3fArr&, Planef&, double maxThreshold, bool arrInliers[]=NULL, size_t maxIters=0);
MATH_API int OptimizePlane(Planef& plane, const Eigen::Vector3f* points, size_t size, int maxIters, float threshold);

MVS_API void EstimatePointColors(const ImageArr& images, PointCloudStreaming& pointcloud);
MVS_API void EstimatePointNormals(const ImageArr& images, PointCloudStreaming& pointcloud, int numNeighbors=16/*K-nearest neighbors*/);

MVS_API bool EstimateNormalMap(const Matrix3x3f& K, const DepthMap&, NormalMap&);

MVS_API bool SaveDepthMap(const String& fileName, const DepthMap& depthMap);
MVS_API bool LoadDepthMap(const String& fileName, DepthMap& depthMap);
MVS_API bool SaveNormalMap(const String& fileName, const NormalMap& normalMap);
MVS_API bool LoadNormalMap(const String& fileName, NormalMap& normalMap);
MVS_API bool SaveConfidenceMap(const String& fileName, const ConfidenceMap& confMap);
MVS_API bool LoadConfidenceMap(const String& fileName, ConfidenceMap& confMap);

MVS_API Image8U3 DepthMap2Image(const DepthMap& depthMap, Depth minDepth=FLT_MAX, Depth maxDepth=0);
MVS_API bool ExportDepthMap(const String& fileName, const DepthMap& depthMap, Depth minDepth=FLT_MAX, Depth maxDepth=0);
MVS_API bool ExportNormalMap(const String& fileName, const NormalMap& normalMap);
MVS_API bool ExportConfidenceMap(const String& fileName, const ConfidenceMap& confMap);
MVS_API bool ExportPointCloud(const String& fileName, const Image&, const DepthMap&, const NormalMap&);

MVS_API bool ExportDepthDataRaw(const String&, const String& imageFileName,
	const IIndexArr&, const cv::Size& imageSize,
	const KMatrix&, const RMatrix&, const CMatrix&,
	Depth dMin, Depth dMax,
	const DepthMap&, const NormalMap&, const ConfidenceMap&, const ViewsMap&);
MVS_API bool ImportDepthDataRaw(const String&, String& imageFileName,
	IIndexArr&, cv::Size& imageSize,
	KMatrix&, RMatrix&, CMatrix&,
	Depth& dMin, Depth& dMax,
	DepthMap&, NormalMap&, ConfidenceMap&, ViewsMap&, unsigned flags=15);

MVS_API void CompareDepthMaps(const DepthMap& depthMap, const DepthMap& depthMapGT, uint32_t idxImage, float threshold=0.01f);
MVS_API void CompareNormalMaps(const NormalMap& normalMap, const NormalMap& normalMapGT, uint32_t idxImage);
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_DEPTHMAP_H_
