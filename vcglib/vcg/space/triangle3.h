/****************************************************************************
* VCGLib                                                            o o     *
* Visual and Computer Graphics Library                            o     o   *
*                                                                _   O  _   *
* Copyright(C) 2004-2016                                           \/)\/    *
* Visual Computing Lab                                            /\/|      *
* ISTI - Italian National Research Council                           |      *
*                                                                    \      *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
****************************************************************************/

#ifndef __VCG_TRIANGLE3
#define __VCG_TRIANGLE3

#include <vcg/space/box3.h>
#include <vcg/space/point2.h>
#include <vcg/space/point3.h>
#include <vcg/space/plane3.h>
#include <vcg/space/segment3.h>
#include <vcg/space/triangle2.h>

namespace vcg {

/** \addtogroup space */
/*@{*/
/**
        Templated class for storing a generic triangle in a 3D space.
    Note the relation with the Face class of TriMesh complex, both classes provide the P(i) access functions to their points and therefore they share the algorithms on it (e.g. area, normal etc...)
 */
template <class ScalarTriangleType> class Triangle3
{
public:
  typedef ScalarTriangleType ScalarType;
    typedef Point3< ScalarType > CoordType;
    /// The bounding box type
    typedef Box3<ScalarType> BoxType;

/*********************************************
    blah
    blah
**/
    Triangle3(){}
    Triangle3(const CoordType & c0,const CoordType & c1,const CoordType & c2){_v[0]=c0;_v[1]=c1;_v[2]=c2;}
protected:
    /// Vector of vertex pointer incident in the face
    Point3<ScalarType> _v[3];
public:

  /// Shortcut per accedere ai punti delle facce
  inline CoordType & P( const int j ) { return _v[j];}
  inline CoordType & P0( const int j ) { return _v[j];}
  inline CoordType & P1( const int j ) { return _v[(j+1)%3];}
  inline CoordType & P2( const int j ) { return _v[(j+2)%3];}
  inline const CoordType &  P( const int j ) const { return _v[j];}
  inline const CoordType & cP( const int j ) const { return _v[j];}
  inline const CoordType &  P0( const int j ) const { return _v[j];}
  inline const CoordType &  P1( const int j ) const { return _v[(j+1)%3];}
  inline const CoordType &  P2( const int j ) const { return _v[(j+2)%3];}
  inline const CoordType & cP0( const int j ) const { return _v[j];}
  inline const CoordType & cP1( const int j ) const { return _v[(j+1)%3];}
  inline const CoordType & cP2( const int j ) const { return _v[(j+2)%3];}

  inline int VN() const { return 3;}
}; //end Class

/********************** Normal **********************/

/// Returns the normal to the plane passing through p0,p1,p2
template<class TriangleType>
typename TriangleType::CoordType TriangleNormal(const TriangleType &t)
{
  return (( t.cP(1) - t.cP(0)) ^ (t.cP(2) - t.cP(0)));
}
/// Returns the normal to the plane passing through p0,p1,p2
template<class TriangleType>
typename TriangleType::CoordType NormalizedTriangleNormal(const TriangleType &t)
{
  return (( t.cP(1) - t.cP(0)) ^ (t.cP(2) - t.cP(0))).Normalize();
}
template<class Point3Type>
Point3Type Normal( Point3Type const &p0, Point3Type const & p1,  Point3Type const & p2)
{
  return (( p1 - p0) ^ (p2 - p0));
}

/********************** Interpolation **********************/

// The function to computing barycentric coords of a point inside a triangle.
// it requires the knowledge of what is the direction that is more orthogonal to the face plane.
//  ScalarType nx = math::Abs((*fi).cN()[0]);
//  ScalarType ny = math::Abs((*fi).cN()[1]);
//  ScalarType nz = math::Abs((*fi).cN()[2]);
//  if(nx>ny && nx>nz) { axis = 0; }
//  else if(ny>nz)     { axis = 1 }
//  else               { axis = 2 }
//  InterpolationParameters(*fp,axis,Point,L);
//
// This normal direction is used to project the triangle in 2D and solve the problem in 2D where it is simpler and often well defined.

template<class TriangleType, class ScalarType>
bool InterpolationParameters(const TriangleType t, const int Axis, const Point3<ScalarType> & P,  Point3<ScalarType> & L)
{
    typedef Point2<ScalarType> P2;
    if(Axis==0) return InterpolationParameters2( P2(t.cP(0)[1],t.cP(0)[2]), P2(t.cP(1)[1],t.cP(1)[2]), P2(t.cP(2)[1],t.cP(2)[2]), P2(P[1],P[2]), L);
    if(Axis==1) return InterpolationParameters2( P2(t.cP(0)[0],t.cP(0)[2]), P2(t.cP(1)[0],t.cP(1)[2]), P2(t.cP(2)[0],t.cP(2)[2]), P2(P[0],P[2]), L);
    if(Axis==2) return InterpolationParameters2( P2(t.cP(0)[0],t.cP(0)[1]), P2(t.cP(1)[0],t.cP(1)[1]), P2(t.cP(2)[0],t.cP(2)[1]), P2(P[0],P[1]), L);
    return false;
}
/// Handy Wrapper of the above one that uses the passed normal N to choose the right orientation
template<class TriangleType, class ScalarType>
bool InterpolationParameters(const TriangleType t, const Point3<ScalarType> & N, const Point3<ScalarType> & P,  Point3<ScalarType> & L)
{
  if(fabs(N[0])>fabs(N[1]))
    {
    if(fabs(N[0])>fabs(N[2]))
            return InterpolationParameters(t,0,P,L); /* 0 > 1 ? 2 */
        else
            return InterpolationParameters(t,2,P,L); /* 2 > 1 ? 2 */
        }
    else
    {
    if(fabs(N[1])>fabs(N[2]))
            return InterpolationParameters(t,1,P,L); /* 1 > 0 ? 2 */
        else
            return InterpolationParameters(t,2,P,L); /* 2 > 1 ? 2 */
    }
}

// Function that computes the barycentric coords of a 2D triangle.
template<class ScalarType>
bool InterpolationParameters2(const Point2<ScalarType> &V1,
                                                       const Point2<ScalarType> &V2,
                                                         const Point2<ScalarType> &V3,
                                                         const Point2<ScalarType> &P, Point3<ScalarType> &L)
{
    vcg::Triangle2<ScalarType> t2=vcg::Triangle2<ScalarType>(V1,V2,V3);
    return (t2.InterpolationParameters(P,L.X(),L.Y(),L.Z() ));
}

/// Handy Wrapper of the above one that calculate the normal on the triangle
template<class TriangleType, class ScalarType>
bool InterpolationParameters(const TriangleType t, const Point3<ScalarType> & P,  Point3<ScalarType> & L)
{
  vcg::Point3<ScalarType> N=vcg::TriangleNormal<TriangleType>(t);
  return (InterpolationParameters<TriangleType,ScalarType>(t,N,P,L));
}


/********************** Quality **********************/

#define ORIG_RANKING // 43.399
#undef RANKING1 // Isn't good as well.... 40.5?
#undef RANKING2 // Worse than 1 and 3, but faster
#undef RANKING3 // about the same as 1, but isn't good enough
#undef RANKING4 // Good enough I think, but about the same performance as without.
#undef RANKING4b // Better, but still not fast 43.1
#undef RANKING4c // Good enough, but not fast 44.5
#undef RANKING5
#undef RANKING6
#undef RANKING7  // Best choice for us for release
#undef RANKING8
#undef RANKING9  // more accurate, best for release
#undef RANKING10  // 7 more accurate, best for release, better than 10
#undef RANKING11  // 7 more accurate, best for release, better than 10

// Calculates sqrt(area2) / maxedge2 using approximate inverse square root
__forceinline float calculateFastApprox(float area2, float maxedge2) {
  // Load area2 into an SSE register
  __m128 area2_vec = _mm_set_ss(area2);

  // Compute approximate 1/sqrt(area2)
  __m128 inv_sqrt_area2_vec = _mm_rsqrt_ss(area2_vec);

  // To get sqrt(area2) from 1/sqrt(area2), you can do area2 * (1/sqrt(area2))
  // Or, more accurately, refine with one Newton-Raphson iteration
  // The RSQRTSS instruction has about 12 bits of precision. One Newton-Raphson
  // iteration typically gets it to nearly full single-precision accuracy (around 23 bits).
  // The formula for one Newton-Raphson iteration for 1/sqrt(x) is:
  // y = y * (1.5 - 0.5 * x * y * y)

  // For your expression sqrt(area2) / maxedge2, it can be rewritten as
  // sqrt(area2) * (1 / maxedge2)
  // or equivalently (area2 * (1/sqrt(area2))) / maxedge2

  // If you need the sqrt(area2) value specifically, and then divide:
  // __m128 sqrt_area2_vec = _mm_mul_ss(area2_vec, inv_sqrt_area2_vec); // This is approximate sqrt(area2)

  // A common pattern for `X / sqrt(Y)` is `X * (1/sqrt(Y))`
  // Here, you have `sqrt(area2) / maxedge2`.
  // So, we can think of it as `area2 * (1/sqrt(area2)) * (1/maxedge2)`
  // OR: `sqrt(area2) * (1/maxedge2)`

  // Option 1: Directly calculate sqrt(area2) * (1/maxedge2)
  // This is often the most direct if RSQRTSS is sufficient precision for 1/sqrt(area2)

  // First, convert 1/sqrt(area2) to sqrt(area2) (still approximate)
  __m128 approx_sqrt_area2_vec = _mm_mul_ss(area2_vec, inv_sqrt_area2_vec);

  // Load maxedge2 into an SSE register
  __m128 maxedge2_vec = _mm_set_ss(maxedge2);

  // Perform the division using _mm_div_ss (hardware division)
  __m128 result_vec = _mm_div_ss(approx_sqrt_area2_vec, maxedge2_vec);

  return _mm_cvtss_f32(result_vec); // Extract the float result
}

/// Compute a shape quality measure of the triangle composed by points p0,p1,p2
/// It Returns 2*AreaTri/(MaxEdge^2),
/// the range is range [0.0, 0.866]
/// e.g. Equilateral triangle sqrt(3)/2, halfsquare: 1/2, ... up to a line that has zero quality.
__forceinline float Quality(
  float const * __restrict p0, float const * __restrict p1, float const * __restrict p2)
{
#ifdef RANKING1

  const P3ScalarType dx1 = p1[0] - p0[0];
  const P3ScalarType dy1 = p1[1] - p0[1];
  const P3ScalarType dz1 = p1[2] - p0[2];

  const P3ScalarType dx2 = p2[0] - p0[0];
  const P3ScalarType dy2 = p2[1] - p0[1];
  const P3ScalarType dz2 = p2[2] - p0[2];

  // Quick squared edge lengths
  const float l1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const float l2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

  // Use dot product (area projection)
  const float dp = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;

  // Cheap area approximation (orthogonality)
  const float quality = 1.0f - (dp * dp) / (l1 * l2 + 1e-8f); // 1-cos(theta)

  return std::max(0.0f, quality); // Clamp to valid range
#endif

#ifdef RANKING2
  // Edge vectors from p0
  const float dx1 = p1[0] - p0[0];
  const float dy1 = p1[1] - p0[1];
  const float dz1 = p1[2] - p0[2];

  const float dx2 = p2[0] - p0[0];
  const float dy2 = p2[1] - p0[1];
  const float dz2 = p2[2] - p0[2];

  // Cross product (not normalized)
  const float nx = dy1 * dz2 - dz1 * dy2;
  const float ny = dz1 * dx2 - dx1 * dz2;
  const float nz = dx1 * dy2 - dy1 * dx2;

  // Max edge^2 (approximate or skip entirely)
  const float l1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const float l2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

  float area2 = nx * nx + ny * ny + nz * nz;
  float maxEdge2 = std::max(l1, l2);

  // Skip sqrt and divide
  return area2 / (maxEdge2 + 1e-8f);
#endif

#ifdef RANKING3
  const float dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];
  const float dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];
  const float dx3 = p1[0] - p2[0], dy3 = p1[1] - p2[1], dz3 = p1[2] - p2[2];

  // Cross product (d10 ^ d20)
  const float cx = dy1 * dz2 - dz1 * dy2;
  const float cy = dz1 * dx2 - dx1 * dz2;
  const float cz = dx1 * dy2 - dy1 * dx2;

  const float area2 = cx * cx + cy * cy + cz * cz; // = (2 * area)^2

  const float e0 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const float e1 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const float e2 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

  const float maxEdge2 = std::max({ e0, e1, e2 });

  if (maxEdge2 == 0) return 0.f; // degenerate triangle

  constexpr float scale = 0.4330127f; // ~sqrt(3)/4, consistent with 2*area/maxEdge^2
  return scale * area2 / (maxEdge2 * maxEdge2);

#if 0
  // Clamp and quantize into 0..255 for <= 0.3 range
  constexpr ScalarType maxQual = ScalarType(0.3);
  ScalarType clamped = (quality > maxQual) ? maxQual : quality;

  // Map [0.0, 0.3] -> [0, 255]
  return static_cast<uint8_t>(clamped * ScalarType(255.0 / 0.3));
#endif
#endif

#ifdef RANKING4
  float dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];
  float dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];

  float cx = dy1 * dz2 - dz1 * dy2;
  float cy = dz1 * dx2 - dx1 * dz2;
  float cz = dx1 * dy2 - dy1 * dx2;

  float area2 = cx * cx + cy * cy + cz * cz;

  float ex = p1[0] - p2[0], ey = p1[1] - p2[1], ez = p1[2] - p2[2];
  float e0 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  float e1 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  float e2 = ex * ex + ey * ey + ez * ez;
  float maxEdge2 = std::max(e0, std::max(e1, e2));

  if (maxEdge2 <= 0.0f) return 0.0f;

  float q = FastSqrtS(area2) / maxEdge2;
  return (q > 0.3f) ? 0.3f : q;
#endif

#ifdef RANKING4b
  const float dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];
  const float dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];

  const float cx = dy1 * dz2 - dz1 * dy2;
  const float cy = dz1 * dx2 - dx1 * dz2;
  const float cz = dx1 * dy2 - dy1 * dx2;

  const float area2 = cx * cx + cy * cy + cz * cz;

  const float ex = p1[0] - p2[0], ey = p1[1] - p2[1], ez = p1[2] - p2[2];
  const float e0 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const float e1 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const float e2 = ex * ex + ey * ey + ez * ez;
  const float maxEdge2 = FastMaxS(e0, FastMaxS(e1, e2));

  if (maxEdge2 <= 0.0f) return 0.0f;

  return FastSqrtS(area2) / maxEdge2;
#endif


#ifdef RANKING4c
  const float dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];
  const float dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];

  const float cx = dy1 * dz2 - dz1 * dy2;
  const float cy = dz1 * dx2 - dx1 * dz2;
  const float cz = dx1 * dy2 - dy1 * dx2;

  const float area2 = cx * cx + cy * cy + cz * cz;

  const float ex = p1[0] - p2[0], ey = p1[1] - p2[1], ez = p1[2] - p2[2];
  const float e0 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  const float e1 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  const float e2 = ex * ex + ey * ey + ez * ez;
  const float maxEdge2 = FastMaxS(e0, FastMaxS(e1, e2));

  if (maxEdge2 <= 0.0f) return 0.0f;

  return LookupSqrt(area2 / (maxEdge2 * maxEdge2));
#endif

#ifdef RANKING5
  float dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];
  float dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];

  float cx = dy1 * dz2 - dz1 * dy2;
  float cy = dz1 * dx2 - dx1 * dz2;
  float cz = dx1 * dy2 - dy1 * dx2;
  float area2 = cx * cx + cy * cy + cz * cz;

  float dx3 = p1[0] - p2[0], dy3 = p1[1] - p2[1], dz3 = p1[2] - p2[2];
  float e0 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
  float e1 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
  float e2 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;
  float maxEdge2 = std::max(e0, std::max(e1, e2));

  if (maxEdge2 <= 0.0f || area2 <= 0.0f)
    return 0.0f;

  float q = SqrtAccurateNear03(area2 / maxEdge2);
  return (q > 0.3f) ? 0.3f : q;
#endif

#ifdef ORIG_RANKING
    using P3ScalarType = float;

    const P3ScalarType dx1 = p1[0] - p0[0];
    const P3ScalarType dy1 = p1[1] - p0[1];
    const P3ScalarType dz1 = p1[2] - p0[2];

    const P3ScalarType dx2 = p2[0] - p0[0];
    const P3ScalarType dy2 = p2[1] - p0[1];
    const P3ScalarType dz2 = p2[2] - p0[2];

    const P3ScalarType dx3 = p1[0] - p2[0];
    const P3ScalarType dy3 = p1[1] - p2[1];
    const P3ScalarType dz3 = p1[2] - p2[2];

    // Squared edge lengths
    const P3ScalarType l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const P3ScalarType l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    const P3ScalarType l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    P3ScalarType maxEdge2 = l2_10;
    if (l2_20 > maxEdge2) maxEdge2 = l2_20;
    if (l2_12 > maxEdge2) maxEdge2 = l2_12;
    constexpr float eps = 1e-20f;   // for squared edge length
    if (maxEdge2 <= eps)
      return 0.0f;

    // Cross product
    const P3ScalarType nx = dy1 * dz2 - dz1 * dy2;
    const P3ScalarType ny = dz1 * dx2 - dx1 * dz2;
    const P3ScalarType nz = dx1 * dy2 - dy1 * dx2;

    const P3ScalarType area2 = nx * nx + ny * ny + nz * nz;
    // if (area2 == 0) return P3ScalarType(0);

    //return FastSqrtS(area2) / maxEdge2;
    return area2 / (maxEdge2 * maxEdge2); // Square the quality and have the caller adjust.
#endif

#ifdef RANKING6
    // d10 = p1 - p0
    const float dx1 = p1[0] - p0[0], dy1 = p1[1] - p0[1], dz1 = p1[2] - p0[2];

    // d20 = p2 - p0
    const float dx2 = p2[0] - p0[0], dy2 = p2[1] - p0[1], dz2 = p2[2] - p0[2];

    // d12 = p1 - p2
    const float dx3 = dx1 - dx2, dy3 = dy1 - dy2, dz3 = dz1 - dz2;

    // Squared edge lengths
    const float l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const float l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    const float l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    // Branchless max
    float maxEdge2 = l2_10;
    maxEdge2 = (l2_20 > maxEdge2) ? l2_20 : maxEdge2;
    maxEdge2 = (l2_12 > maxEdge2) ? l2_12 : maxEdge2;

    // Early out on degenerate
    if (maxEdge2 <= 0.0f) return 0.0f;

    // Cross product of d10 ^ d20
    const float nx = dy1 * dz2 - dz1 * dy2;
    const float ny = dz1 * dx2 - dx1 * dz2;
    const float nz = dx1 * dy2 - dy1 * dx2;
    const float area2 = nx * nx + ny * ny + nz * nz;

    if (area2 <= 0.0f) return 0.0f;

    const float q = FastSqrt_NR2(area2) / maxEdge2;

    // Clamp to 0.3
    return (q > 0.3f) ? 0.3f : q;
#endif
#ifdef RANKING7
    // Best so far
    const P3ScalarType dx1 = p1[0] - p0[0];
    const P3ScalarType dy1 = p1[1] - p0[1];
    const P3ScalarType dz1 = p1[2] - p0[2];

    const P3ScalarType dx2 = p2[0] - p0[0];
    const P3ScalarType dy2 = p2[1] - p0[1];
    const P3ScalarType dz2 = p2[2] - p0[2];

    const P3ScalarType dx3 = p1[0] - p2[0];
    const P3ScalarType dy3 = p1[1] - p2[1];
    const P3ScalarType dz3 = p1[2] - p2[2];

    // Squared edge lengths
    const P3ScalarType l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const P3ScalarType l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    const P3ScalarType l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    P3ScalarType maxEdge2 = l2_10;
    if (l2_20 > maxEdge2) maxEdge2 = l2_20;
    if (l2_12 > maxEdge2) maxEdge2 = l2_12;

   // const float invMaxEdge2 = 1.f / maxEdge2;

    // Cross product
    const P3ScalarType nx = dy1 * dz2 - dz1 * dy2;
    const P3ScalarType ny = dz1 * dx2 - dx1 * dz2;
    const P3ScalarType nz = dx1 * dy2 - dy1 * dx2;

    const P3ScalarType area2 = nx * nx + ny * ny + nz * nz;

    // Slightly better to remove the compare (43.029)
    //if (area2 > 0.09f * maxEdge2 * maxEdge2) {
   //   return 0.3f;
   // } else {
    return FastSqrtS(area2) / maxEdge2;
  //  }
#endif

#ifdef RANKING8
    // SIMD not faster
    const P3ScalarType dx1 = p1[0] - p0[0];
    const P3ScalarType dy1 = p1[1] - p0[1];
    const P3ScalarType dz1 = p1[2] - p0[2];

    const P3ScalarType dx2 = p2[0] - p0[0];
    const P3ScalarType dy2 = p2[1] - p0[1];
    const P3ScalarType dz2 = p2[2] - p0[2];

    const P3ScalarType dx3 = p1[0] - p2[0];
    const P3ScalarType dy3 = p1[1] - p2[1];
    const P3ScalarType dz3 = p1[2] - p2[2];

    // Squared edge lengths
    const P3ScalarType l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const P3ScalarType l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    const P3ScalarType l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    P3ScalarType maxEdge2 = l2_10;
    if (l2_20 > maxEdge2) maxEdge2 = l2_20;
    if (l2_12 > maxEdge2) maxEdge2 = l2_12;
    *maxEdges = maxEdge2;

    // Cross product
    const P3ScalarType nx = dy1 * dz2 - dz1 * dy2;
    const P3ScalarType ny = dz1 * dx2 - dx1 * dz2;
    const P3ScalarType nz = dx1 * dy2 - dy1 * dx2;

    *areas = nx * nx + ny * ny + nz * nz;
#endif


#ifdef RANKING9 // Gives up some speed, but is more accurate
    // Suggestions to improve RANKING7's accuracy
  // Vector diffs
    const P3ScalarType dx1 = p1[0] - p0[0];
    const P3ScalarType dy1 = p1[1] - p0[1];
    const P3ScalarType dz1 = p1[2] - p0[2];

    const P3ScalarType dx2 = p2[0] - p0[0];
    const P3ScalarType dy2 = p2[1] - p0[1];
    const P3ScalarType dz2 = p2[2] - p0[2];

    // Cross product: (p1 - p0) x (p2 - p0)
    const P3ScalarType nx = dy1 * dz2 - dz1 * dy2;
    const P3ScalarType ny = dz1 * dx2 - dx1 * dz2;
    const P3ScalarType nz = dx1 * dy2 - dy1 * dx2;

    const P3ScalarType area2 = nx * nx + ny * ny + nz * nz;

    // Reuse existing diffs to compute all 3 squared edge lengths
    const P3ScalarType l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const P3ScalarType l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

    const P3ScalarType dx3 = dx1 - dx2;
    const P3ScalarType dy3 = dy1 - dy2;
    const P3ScalarType dz3 = dz1 - dz2;
    const P3ScalarType l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    // Branchless max of three
    P3ScalarType maxEdge2 = l2_10;
    maxEdge2 = (l2_20 > maxEdge2) ? l2_20 : maxEdge2;
    maxEdge2 = (l2_12 > maxEdge2) ? l2_12 : maxEdge2;

    // Use epsilon to avoid divide-by-zero, and clamp area2
    constexpr P3ScalarType eps = P3ScalarType(1e-20f);
    const P3ScalarType safeArea2 = area2 + eps;      // No branch
    const P3ScalarType safeMaxEdge2 = maxEdge2 + eps;

    return FastSqrtS(safeArea2) / safeMaxEdge2;
#endif

#ifdef RANKING10
    // more accurate
    // Edge vectors from p0
    const P3ScalarType dx1 = p1[0] - p0[0];
    const P3ScalarType dy1 = p1[1] - p0[1];
    const P3ScalarType dz1 = p1[2] - p0[2];

    const P3ScalarType dx2 = p2[0] - p0[0];
    const P3ScalarType dy2 = p2[1] - p0[1];
    const P3ScalarType dz2 = p2[2] - p0[2];

    // Edge p1 - p2 from reused diffs
    const P3ScalarType dx3 = dx1 - dx2;
    const P3ScalarType dy3 = dy1 - dy2;
    const P3ScalarType dz3 = dz1 - dz2;

    // Squared edge lengths
    const P3ScalarType l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const P3ScalarType l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    const P3ScalarType l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    P3ScalarType maxEdge2 = l2_10;
    maxEdge2 = (l2_20 > maxEdge2) ? l2_20 : maxEdge2;
    maxEdge2 = (l2_12 > maxEdge2) ? l2_12 : maxEdge2;

    // Cross product (area² of triangle)
    const P3ScalarType nx = dy1 * dz2 - dz1 * dy2;
    const P3ScalarType ny = dz1 * dx2 - dx1 * dz2;
    const P3ScalarType nz = dx1 * dy2 - dy1 * dx2;

    const P3ScalarType area2 = nx * nx + ny * ny + nz * nz;

    // Safe clamp to avoid divide-by-zero
    constexpr P3ScalarType epsilon = P3ScalarType(1e-20f);

    // Return exact 0.0 if degenerate
    if (!(area2 > epsilon) || !(maxEdge2 > epsilon))
      return P3ScalarType(0);

    return FastSqrtS(area2) / maxEdge2;

#endif

#ifdef RANKING11
    // more accurate
    // Edge vectors from p0
    const P3ScalarType dx1 = p1[0] - p0[0];
    const P3ScalarType dy1 = p1[1] - p0[1];
    const P3ScalarType dz1 = p1[2] - p0[2];

    const P3ScalarType dx2 = p2[0] - p0[0];
    const P3ScalarType dy2 = p2[1] - p0[1];
    const P3ScalarType dz2 = p2[2] - p0[2];

    // Edge p1 - p2 from reused diffs
    const P3ScalarType dx3 = dx1 - dx2;
    const P3ScalarType dy3 = dy1 - dy2;
    const P3ScalarType dz3 = dz1 - dz2;

    // Squared edge lengths
    const P3ScalarType l2_10 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
    const P3ScalarType l2_20 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
    const P3ScalarType l2_12 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3;

    P3ScalarType maxEdge2 = l2_10;
    maxEdge2 = (l2_20 > maxEdge2) ? l2_20 : maxEdge2;
    maxEdge2 = (l2_12 > maxEdge2) ? l2_12 : maxEdge2;

    // Cross product (area² of triangle)
    const P3ScalarType nx = dy1 * dz2 - dz1 * dy2;
    const P3ScalarType ny = dz1 * dx2 - dx1 * dz2;
    const P3ScalarType nz = dx1 * dy2 - dy1 * dx2;

    const P3ScalarType area2 = nx * nx + ny * ny + nz * nz;

    // Safe clamp to avoid divide-by-zero
    constexpr P3ScalarType epsilon = P3ScalarType(1e-20f);

    // Return exact 0.0 if degenerate
    if (!(area2 > epsilon) || !(maxEdge2 > epsilon))
      return P3ScalarType(0);

    return area2 / ( maxEdge2 * maxEdge2 );


#endif
}

/// Return the _q of the face, the return value is in [0,sqrt(3)/2] = [0 - 0.866.. ]
template<class TriangleType>
__forceinline float QualityFace(const TriangleType &t)
{
  return Quality((float*) &t.cP(0), (float*)&t.cP(1), (float*)&t.cP(2));
}

/// Compute a shape quality measure of the triangle composed by points p0,p1,p2
/// It Returns inradius/circumradius
/// the range is range [0, 1]
/// e.g. Equilateral triangle 1, halfsquare: 0.81, ... up to a line that has zero quality.
template<class P3ScalarType>
P3ScalarType QualityRadii(Point3<P3ScalarType> const &p0,
                                                    Point3<P3ScalarType> const &p1,
                                                    Point3<P3ScalarType> const &p2) {

    P3ScalarType a=(p1-p0).Norm();
    P3ScalarType b=(p2-p0).Norm();
    P3ScalarType c=(p1-p2).Norm();

    P3ScalarType sum = (a + b + c)*0.5;
    P3ScalarType area2 =  sum*(a+b-sum)*(a+c-sum)*(b+c-sum);
    if(area2 <= 0) return 0;
    //circumradius: (a*b*c)/(4*sqrt(area2))
    //inradius: (a*b*c)/(4*circumradius*sum) => sqrt(area2)/sum;
    return (8*area2)/(a*b*c*sum);
}

/// Compute a shape quality measure of the triangle composed by points p0,p1,p2
/// It Returns mean ratio 2sqrt(a, b)/(a+b) where a+b are the eigenvalues of the M^tM of the
/// transformation matrix into a regular simplex
/// the range is range [0, 1]
template<class P3ScalarType>
P3ScalarType QualityMeanRatio(Point3<P3ScalarType> const &p0,
                                                    Point3<P3ScalarType> const &p1,
                                                    Point3<P3ScalarType> const &p2) {

    P3ScalarType a=(p1-p0).Norm();
    P3ScalarType b=(p2-p0).Norm();
    P3ScalarType c=(p1-p2).Norm();
    P3ScalarType sum = (a + b + c)*0.5; //semiperimeter
    P3ScalarType area2 =  sum*(a+b-sum)*(a+c-sum)*(b+c-sum);
    if(area2 <= 0) return 0;
    return (4.0*sqrt(3.0)*sqrt(area2))/(a*a + b*b + c*c);
}



/// Return the Double of area of the triangle
// NOTE the old Area function has been removed to intentionally
// cause compiling error that will help people to check their code...
// A some  people used Area assuming that it returns the double and some not.
// So please check your codes!!!
// And please DO NOT Insert any Area named function here!

template<class TriangleType>
typename TriangleType::ScalarType DoubleArea(const TriangleType &t)
{
    return Norm( (t.cP(1) - t.cP(0)) ^ (t.cP(2) - t.cP(0)) );
}

template<class TriangleType>
typename TriangleType::ScalarType CosWedge(const TriangleType &t, int k)
{
  typename TriangleType::CoordType
    e0 = t.cP((k+1)%3) - t.cP(k),
    e1 = t.cP((k+2)%3) - t.cP(k);
  return (e0*e1)/(e0.Norm()*e1.Norm());
}

template<class TriangleType>
Point3<typename TriangleType::ScalarType> Barycenter(const TriangleType &t)
{
    return ((t.cP(0)+t.cP(1)+t.cP(2))/(typename TriangleType::ScalarType) 3.0);
}

template<class TriangleType>
typename TriangleType::ScalarType Perimeter(const TriangleType &t)
{
  return Distance(t.cP(0),t.cP(1))+
         Distance(t.cP(1),t.cP(2))+
         Distance(t.cP(2),t.cP(0));
}

template<class TriangleType>
Point3<typename TriangleType::ScalarType> Circumcenter(const TriangleType &t)
{
   typename TriangleType::ScalarType a2 = (t.cP(1) - t.cP(2)).SquaredNorm();
   typename TriangleType::ScalarType b2 = (t.cP(2) - t.cP(0)).SquaredNorm();
   typename TriangleType::ScalarType c2 = (t.cP(0) - t.cP(1)).SquaredNorm();
   Point3<typename TriangleType::ScalarType>c = t.cP(0)*a2*(-a2 + b2 + c2) +
                                                t.cP(1)*b2*( a2 - b2 + c2) +
                                                t.cP(2)*c2*( a2 + b2 - c2);
   c /= 2*(a2*b2 + a2*c2 + b2*c2) - a2*a2 - b2*b2 - c2*c2;
   return c;
}


}	 // end namespace


#endif

