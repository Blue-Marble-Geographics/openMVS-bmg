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

#ifndef __VCG_TRI_UPDATE_NORMALS
#define __VCG_TRI_UPDATE_NORMALS

#include <vcg/space/triangle3.h>
#include <vcg/complex/base.h>

#include <vcg/complex/algorithms/polygon_support.h>

#include "flag.h"

#include "P2PUtils.h"

#define FAST_NORMALIZE_PER_FACE

namespace vcg {
namespace tri {

/// \ingroup trimesh

/// \headerfile normal.h vcg/complex/algorithms/update/normal.h

/// \brief Management, updating and computation of per-vertex, per-face, and per-wedge normals.
/**
This class is used to compute or to update the normals that can be stored in the various component of a mesh.
A number of different algorithms for computing per vertex normals are present.

It must be included \b after complex.h
*/

template <class ComputeMeshType>
class   UpdateNormal
{
public:
typedef ComputeMeshType MeshType;
typedef typename MeshType::VertexType     VertexType;
typedef typename MeshType::CoordType     CoordType;
typedef typename VertexType::NormalType     NormalType;
typedef typename VertexType::ScalarType ScalarType;
typedef typename MeshType::VertexPointer  VertexPointer;
typedef typename MeshType::VertexIterator VertexIterator;
typedef typename MeshType::FaceType       FaceType;
typedef typename MeshType::FacePointer    FacePointer;
typedef typename MeshType::FaceIterator   FaceIterator;

/// \brief Set to zero all the PerVertex normals
/**
 Set to zero all the PerVertex normals. Used by all the face averaging algorithms.
 by default it does not clear the normals of unreferenced vertices because they could be still useful
 */
static void PerVertexClear(ComputeMeshType &m, bool ClearAllVertNormal=false)
{
  RequirePerVertexNormal(m);
  if(ClearAllVertNormal)
    UpdateFlags<ComputeMeshType>::VertexClearV(m);
  else
  {
    UpdateFlags<ComputeMeshType>::VertexSetV(m);
    for(FaceIterator f=m.face.begin();f!=m.face.end();++f)
     if( !(*f).IsD() )
       for(int i=0;i<3;++i) (*f).V(i)->ClearV();
   }
  VertexIterator vi;
  for(vi=m.vert.begin();vi!=m.vert.end();++vi)
     if( !(*vi).IsD() && (*vi).IsRW() && (!(*vi).IsV()) )
         (*vi).N() = NormalType((ScalarType)0,(ScalarType)0,(ScalarType)0);
}

///  \brief Calculates the vertex normal as the classic area weighted average. It does not need or exploit current face normals.
/**
 The normal of a vertex v is the classical area-weigthed average of the normals of the faces incident on v.
 */
static void PerVertex(ComputeMeshType &m)
{
 PerVertexClear(m);
 for(FaceIterator f=m.face.begin();f!=m.face.end();++f)
   if( !(*f).IsD() && (*f).IsR() )
   {
    typename VertexType::NormalType t = vcg::TriangleNormal(*f);

    for(int j=0; j<(*f).VN(); ++j)
     if( !(*f).V(j)->IsD() && (*f).V(j)->IsRW() )
      (*f).V(j)->N() += t;
   }
}

static void PerFacePolygonal(ComputeMeshType &m)
{
  RequirePerFaceNormal(m);  
  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
  {
    if( !(*fi).IsD() )
      fi->N() = PolygonNormal(*fi).Normalize();
  }
}

///  \brief Calculates the vertex normal as an angle weighted average. It does not need or exploit current face normals.
/**
 The normal of a vertex v computed as a weighted sum f the incident face normals.
 The weight is simlply the angle of the involved wedge.  Described in:

G. Thurmer, C. A. Wuthrich
"Computing vertex normals from polygonal facets"
Journal of Graphics Tools, 1998
 */

#pragma intrinsic(_InterlockedCompareExchange)
static __forceinline float AtomicAddFloat(float* __restrict addr, float val)
{
  static_assert(sizeof(float) == sizeof(LONG), "Expected float and LONG to be same size");

  volatile LONG* __restrict intAddr = reinterpret_cast<volatile LONG*>(addr);
  union {
    float f;
    LONG i;
  } oldVal, newVal;

  do {
    oldVal.i = *intAddr;
    newVal.f = oldVal.f + val;
    newVal.i = *reinterpret_cast<LONG*>(&newVal.f); // or just reuse newVal.i = *(LONG*)&newVal.f;
  } while (_InterlockedCompareExchange(intAddr, newVal.i, oldVal.i) != oldVal.i);

  return newVal.f;
}

static void PerVertexAngleWeighted(ComputeMeshType& m)
{
#if 1 // Try again
  using Scalar = float;
  using Point = vcg::Point3<Scalar>;

  const auto* start = &m.vert[0];
#pragma omp parallel for
  for (ptrdiff_t fi = 0; fi < (ptrdiff_t)m.face.size(); ++fi) {
    auto& f = m.face[fi];
    if (f.IsD()) continue;

    const auto& p0 = f.cP(0);
    const auto& p1 = f.cP(1);
    const auto& p2 = f.cP(2);

    // Edge vectors
    const Point e0 = p1 - p0;
    const Point e1 = p2 - p1;
    const Point e2 = p0 - p2;

    // Face normal
    const Point fn = e0 ^ -e2;

    // Squared lengths
    const float e0len2 = e0.X() * e0.X() + e0.Y() * e0.Y() + e0.Z() * e0.Z();
    const float e1len2 = e1.X() * e1.X() + e1.Y() * e1.Y() + e1.Z() * e1.Z();
    const float e2len2 = e2.X() * e2.X() + e2.Y() * e2.Y() + e2.Z() * e2.Z();
    const float fnLen2 = fn.X() * fn.X() + fn.Y() * fn.Y() + fn.Z() * fn.Z();

    // SIMD sqrt of [e0, e1, e2, fn]
    const _Data len2 = _SetN(fnLen2, e2len2, e1len2, e0len2);
    const _Data len = _mm_sqrt_ps(len2);
    const _Data inv = _Div(_Set(1.0f), len);

    float invs[4];
    _Store(invs, inv);

    Point n(0, 0, 0);
    if (fnLen2 > 1e-20f)
      n = fn * invs[3];

    // Edge normals
    const Point e0n = e0 * invs[0];
    const Point e1n = e1 * invs[1];
    const Point e2n = e2 * invs[2];

    // Wedge angles
    const float d0 = e0n * (-e2n);
    const float d1 = (-e0n) * e1n;
    const float d2 = (-e1n) * e2n;

    const _Data vDots = _SetN(d0, d1, d2, 0.0f);
    const _Data vAngles = FastACos(vDots);

    float angles[4];
    _Store(angles, vAngles);

    const Point c0 = angles[0] * n;
    const Point c1 = angles[1] * n;
    const Point c2 = angles[2] * n;

    const int i0 = int(f.V(0) - start);
    const int i1 = int(f.V(1) - start);
    const int i2 = int(f.V(2) - start);

    // ============================
    // Atomic accumulation per vertex
    // ============================
    float* __restrict f0 = &m.vert[i0].N()[0];
    float* __restrict f1 = &m.vert[i1].N()[0];
    float* __restrict f2 = &m.vert[i2].N()[0];

    const float c0x = c0.X();
    const float c0y = c0.Y();
    const float c0z = c0.Z();

    const float c1x = c1.X();
    const float c1y = c1.Y();
    const float c1z = c1.Z();

    const float c2x = c2.X();
    const float c2y = c2.Y();
    const float c2z = c2.Z();

    AtomicAddFloat(f0, c0x);
    AtomicAddFloat(f0 + 1, c0y);
    AtomicAddFloat(f0 + 2, c0z);

    AtomicAddFloat(f1, c1x);
    AtomicAddFloat(f1 + 1, c1y);
    AtomicAddFloat(f1 + 2, c1z);

    AtomicAddFloat(f2, c2x);
    AtomicAddFloat(f2 + 1, c2y);
    AtomicAddFloat(f2 + 2, c2z);
  }

  // Normalize final vertex normals
#pragma omp parallel for
  for (int i = 0; i < (int)m.vert.size(); ++i) {
    if (m.vert[i].IsD()) continue;
    Point n = m.vert[i].N();
    if (Norm(n) > 0) n.Normalize();
    m.vert[i].N() = n;
  }

#else
  using Scalar = float;
  using Point = vcg::Point3<Scalar>;

  const int nVerts = (int)m.vert.size();
  const int nThreads = omp_get_max_threads();

  std::vector<Scalar> soaAccum((size_t)nVerts * nThreads * 3, Scalar(0));

  const auto* start = &m.vert[0];

  if (m.hasDeletedFaces)
  {
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      Scalar* __restrict soa = soaAccum.data();

#pragma omp for
      for (ptrdiff_t fi = 0; fi < (ptrdiff_t)m.face.size(); ++fi) {
        auto& f = m.face[fi];
        if (f.IsD()) continue;

        const auto& p0 = f.cP(0);
        const auto& p1 = f.cP(1);
        const auto& p2 = f.cP(2);

        const Point e0 = p1 - p0;
        const Point e1 = p2 - p1;
        const Point e2 = p0 - p2;

        const Point fn = e0 ^ -e2;
        const Scalar fnLen = Norm(fn);
        if (fnLen <= Scalar(0)) continue;
        const Point n = fn / fnLen;

        // -----------------------------------------
        // Normalize 3 edges in SIMD
        // -----------------------------------------
        const _Data len2 = _mm_set_ps(
          e2.X() * e2.X() + e2.Y() * e2.Y() + e2.Z() * e2.Z(),
          e1.X() * e1.X() + e1.Y() * e1.Y() + e1.Z() * e1.Z(),
          e0.X() * e0.X() + e0.Y() * e0.Y() + e0.Z() * e0.Z(),
          0.0f);

        const _Data len = _mm_sqrt_ps(len2);
        const _Data inv = _Div(_Set(1.0f), len);

        float invs[4];
        _Store(invs, inv);

        const Point e0n = Point(e0.X() * invs[0], e0.Y() * invs[0], e0.Z() * invs[0]);
        const Point e1n = Point(e1.X() * invs[1], e1.Y() * invs[1], e1.Z() * invs[1]);
        const Point e2n = Point(e2.X() * invs[2], e2.Y() * invs[2], e2.Z() * invs[2]);

        const float d0 = e0n * (-e2n);
        const float d1 = (-e0n) * e1n;
        const float d2 = (-e1n) * e2n;

        const _Data vDots = _SetN(d0, d1, d2, 0.0f);
        const _Data vAngles = FastACos(vDots);

        float angles[4];
        _Store(angles, vAngles);

        const Point c0 = angles[0] * n;
        const Point c1 = angles[1] * n;
        const Point c2 = angles[2] * n;

        const size_t i0 = f.V(0) - start;
        const size_t i1 = f.V(1) - start;
        const size_t i2 = f.V(2) - start;

        Scalar* __restrict a0ptr = &soa[i0 * nThreads * 3 + tid * 3];
        Scalar* __restrict a1ptr = &soa[i1 * nThreads * 3 + tid * 3];
        Scalar* __restrict a2ptr = &soa[i2 * nThreads * 3 + tid * 3];

        a0ptr[0] += c0[0]; a0ptr[1] += c0[1]; a0ptr[2] += c0[2];
        a1ptr[0] += c1[0]; a1ptr[1] += c1[1]; a1ptr[2] += c1[2];
        a2ptr[0] += c2[0]; a2ptr[1] += c2[1]; a2ptr[2] += c2[2];
      }
    }
  }
  else
  {
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      Scalar* __restrict soa = soaAccum.data();

#pragma omp for
      for (ptrdiff_t fi = 0; fi < (ptrdiff_t)m.face.size(); ++fi) {
        auto& f = m.face[fi];

        const auto& p0 = f.cP(0);
        const auto& p1 = f.cP(1);
        const auto& p2 = f.cP(2);

        const Point e0 = p1 - p0;
        const Point e1 = p2 - p1;
        const Point e2 = p0 - p2;

        const Point fn = e0 ^ -e2;

        // Compute squared lengths: e0, e1, e2, fn
        const float e0len2 = e0.X() * e0.X() + e0.Y() * e0.Y() + e0.Z() * e0.Z();
        const float e1len2 = e1.X() * e1.X() + e1.Y() * e1.Y() + e1.Z() * e1.Z();
        const float e2len2 = e2.X() * e2.X() + e2.Y() * e2.Y() + e2.Z() * e2.Z();
        const float fnLen2 = fn.X() * fn.X() + fn.Y() * fn.Y() + fn.Z() * fn.Z();

        // Pack into SIMD lanes [e0, e1, e2, fn]
        const _Data len2 = _SetN(fnLen2, e2len2, e1len2, e0len2);

        // sqrt all 4 at once
        const _Data len = _mm_sqrt_ps(len2);

        // invLen = 1.0 / len
        const _Data inv = _Div(_Set(1.0f), len);

        // Extract results
        float invs[4];
        _Store(invs, inv);

        // invs[0] -> 1/|e0|
        // invs[1] -> 1/|e1|
        // invs[2] -> 1/|e2|
        // invs[3] -> 1/|fn|

        // Face normal
        Point n(0, 0, 0);
        if (fnLen2 > 1e-20f) {
          n = fn * invs[3];
        }

        // Edge normals
        const Point e0n = e0 * invs[0];
        const Point e1n = e1 * invs[1];
        const Point e2n = e2 * invs[2];

        const float d0 = e0n * (-e2n);
        const float d1 = (-e0n) * e1n;
        const float d2 = (-e1n) * e2n;

        const _Data vDots = _SetN(d0, d1, d2, 0.0f);
        const _Data vAngles = FastACos(vDots);

        float angles[4];
        _Store(angles, vAngles);

        const Point c0 = angles[0] * n;
        const Point c1 = angles[1] * n;
        const Point c2 = angles[2] * n;

        const size_t i0 = f.V(0) - start;
        const size_t i1 = f.V(1) - start;
        const size_t i2 = f.V(2) - start;

        Scalar* __restrict a0ptr = &soa[i0 * nThreads * 3 + tid * 3];
        Scalar* __restrict a1ptr = &soa[i1 * nThreads * 3 + tid * 3];
        Scalar* __restrict a2ptr = &soa[i2 * nThreads * 3 + tid * 3];

        a0ptr[0] += c0[0]; a0ptr[1] += c0[1]; a0ptr[2] += c0[2];
        a1ptr[0] += c1[0]; a1ptr[1] += c1[1]; a1ptr[2] += c1[2];
        a2ptr[0] += c2[0]; a2ptr[1] += c2[1]; a2ptr[2] += c2[2];
      }
    }
  }

  // =====================================================
  // Reduction + normalize
  // =====================================================
#pragma omp parallel for
  for (int i = 0; i < nVerts; ++i) {
    if (m.vert[i].IsD()) continue;

    Scalar nx = 0, ny = 0, nz = 0;
    for (int t = 0; t < nThreads; ++t) {
      const Scalar* __restrict acc = &soaAccum[i * nThreads * 3 + t * 3];
      nx += acc[0];
      ny += acc[1];
      nz += acc[2];
    }

    Point n(nx, ny, nz);
    if (Norm(n) > 0) n.Normalize();
    m.vert[i].N() = n;
  }
#endif
}

///  \brief Calculates the vertex normal using the Max et al. weighting scheme. It does not need or exploit current face normals.
/**
 The normal of a vertex v is computed according to the formula described by Nelson Max in
 Max, N., "Weights for Computing Vertex Normals from Facet Normals", Journal of Graphics Tools, 4(2) (1999)

 The weight for each wedge is the cross product of the two edge over the product of the square of the two edge lengths.
 According to the original paper it is perfect only for spherical surface, but it should perform well...
 */
static void PerVertexNelsonMaxWeighted(ComputeMeshType &m)
{
 PerVertexClear(m);
 FaceIterator f;
 for(f=m.face.begin();f!=m.face.end();++f)
   if( !(*f).IsD() && (*f).IsR() )
   {
    typename FaceType::NormalType t = TriangleNormal(*f);
        ScalarType e0 = SquaredDistance((*f).V0(0)->cP(),(*f).V1(0)->cP());
        ScalarType e1 = SquaredDistance((*f).V0(1)->cP(),(*f).V1(1)->cP());
        ScalarType e2 = SquaredDistance((*f).V0(2)->cP(),(*f).V1(2)->cP());

        (*f).V(0)->N() += t/(e0*e2);
        (*f).V(1)->N() += t/(e0*e1);
        (*f).V(2)->N() += t/(e1*e2);
   }
}

/// \brief Calculates the face normal
///
/// Not normalized. Use PerFaceNormalized() or call NormalizePerVertex() if you need unit length per face normals.
static void PerFace(ComputeMeshType& m)
{
  RequirePerFaceNormal(m); // JPB WIP BUG What is this doing?

  int64_t numFaces = m.fn;

  if (m.hasDeletedFaces)
  {
#pragma omp parallel for
    for (int64_t i = 0; i < numFaces; ++i) {
      auto& f = m.face[i];
      if (!f.IsD()) {
        f.N() = TriangleNormal(f);
      }
    }
  }
  else
  {
#pragma omp parallel for
    for (int64_t i = 0; i < numFaces; ++i) {
      auto& f = m.face[i];

      auto v0 = f.cP(0);
      auto v1 = f.cP(1);
      auto v2 = f.cP(2);

      auto e0 = v1 - v0;
      auto e1 = v2 - v0;

      auto n = e0 ^ e1;  // cross product

      f.N()[0] = n[0];
      f.N()[1] = n[1];
      f.N()[2] = n[2];
    }
  }
}


/// \brief computePerPolygonalFace computes the normal of each polygonal face.
///
/// Not normalized. Use PerPolygonalFaceNormalized() or call NormalizePerFace() if you need unit length per face normals.
static void PerPolygonalFace(ComputeMeshType &m) {
  tri::RequirePerFaceNormal(m);
  tri::RequirePolygonalMesh(m);
  for(FaceIterator fi = m.face.begin(); fi != m.face.end(); fi++)
    if (!fi->IsD()) {
      fi->N().SetZero();
      for (int i = 0; i < fi->VN(); i++)
        fi->N() += fi->V0(i)->P() ^ fi->V1(i)->P();
    }
}


/// \brief Calculates the vertex normal by averaging the current per-face normals.
/**
    The normal of a vertex v is the average of the un-normalized normals of the faces incident on v.
*/
static void PerVertexFromCurrentFaceNormal(ComputeMeshType &m)
{
  tri::RequirePerVertexNormal(m);

 VertexIterator vi;
 for(vi=m.vert.begin();vi!=m.vert.end();++vi)
   if( !(*vi).IsD() && (*vi).IsRW() )
     (*vi).N()=CoordType(0,0,0);

 FaceIterator fi;
 for(fi=m.face.begin();fi!=m.face.end();++fi)
   if( !(*fi).IsD())
   {
    for(int j=0; j<(*fi).VN(); ++j)
            if( !(*fi).V(j)->IsD())
                    (*fi).V(j)->N() += (*fi).cN();
   }
}

/// \brief Calculates the face normal by averaging the current per-vertex normals.
/**
    The normal of a face f is the average of the normals of the vertices of f.
*/
static void PerFaceFromCurrentVertexNormal(ComputeMeshType &m)
{
  tri::RequirePerVertexNormal(m);
  tri::RequirePerFaceNormal(m);
  for (FaceIterator fi=m.face.begin(); fi!=m.face.end(); ++fi)
   if( !(*fi).IsD())
        {
        NormalType n;
        n.SetZero();
        for(int j=0; j<3; ++j)
            n += fi->V(j)->cN();
        n.Normalize();
        fi->N() = n;
    }
}

/// \brief Normalize the length of the vertex normals.
static void NormalizePerVertex(ComputeMeshType& m)
{
    tri::RequirePerVertexNormal(m); // JPB WIP BUG What is this doing?

    int64_t numVertices = m.vn;
#pragma omp parallel for
    for (int64_t i = 0; i < numVertices; ++i) {
        auto& v = m.vert[i];
        if (!v.IsD() && v.IsRW()) {
            v.N().Normalize();
        }
    }
}

/// \brief Normalize the length of the face normals.
#ifdef FAST_NORMALIZE_PER_FACE
static inline void NormalizePerFace(ComputeMeshType& m)
{
  auto* __restrict faces = &m.face[0];
  const size_t n = m.face.size();
  if (!n) return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i)
  {
    auto& f = faces[i];
    if (!f.IsD())
    {
      auto& nrm = f.N();
      const float len2 = nrm.SquaredNorm();
      if (len2 > 0.0f)
      {
        const float invLen = 1.0f / FastSqrtS(len2);
        nrm *= invLen;
      }
    }
  }
}
#else
static void NormalizePerFace(ComputeMeshType &m)
{
  tri::RequirePerFaceNormal(m);
  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
      if( !(*fi).IsD() )	(*fi).N().Normalize();
}
#endif
/// \brief Set the length of the face normals to their area (without recomputing their directions).
static void NormalizePerFaceByArea(ComputeMeshType &m)
{
  tri::RequirePerFaceNormal(m);
  FaceIterator fi;
  for(fi=m.face.begin();fi!=m.face.end();++fi)
    if( !(*fi).IsD() )
            {
                (*fi).N().Normalize();
                (*fi).N() = (*fi).N() * DoubleArea(*fi);
            }
}

/// \brief Equivalent to PerVertex() and NormalizePerVertex()
static void PerVertexNormalized(ComputeMeshType &m)
{
  PerVertex(m);
  NormalizePerVertex(m);
}

/// \brief Equivalent to PerFace() and NormalizePerFace()
static void PerFaceNormalized(ComputeMeshType &m)
{
  PerFace(m);
  NormalizePerFace(m);
}

/// \brief Equivalent to PerPolygonalFace() and NormalizePerFace()
static void PerPolygonalFaceNormalized(ComputeMeshType &m) {
  PerPolygonalFace(m);
  NormalizePerFace(m);
}

/// \brief Equivalent to PerVertex() and PerFace().
static void PerVertexPerFace(ComputeMeshType &m)
{
 PerFace(m);
 PerVertex(m);
}

/// \brief Equivalent to PerVertexNormalized() and PerFace().
static void PerVertexNormalizedPerFace(ComputeMeshType &m)
{
    PerVertexPerFace(m);
    NormalizePerVertex(m);
}

/// \brief Equivalent to PerVertexNormalizedPerFace() and NormalizePerFace().
static void PerVertexNormalizedPerFaceNormalized(ComputeMeshType &m)
{
    PerVertexNormalizedPerFace(m);
    NormalizePerFace(m);
}

/// \brief Exploit bitquads to compute a per-polygon face normal
static void PerBitQuadFaceNormalized(ComputeMeshType &m)
{
    PerFace(m);
    for(FaceIterator f=m.face.begin();f!=m.face.end();++f) {
      if( !(*f).IsD() )	{
        for (int k=0; k<3; k++) if (f->IsF(k))
        if (&*f < f->FFp(k)) {
          f->N() = f->FFp(k)->N() = (f->FFp(k)->N() + f->N()).Normalize();
        }
      }
  }
}


/// \brief Exploit bitquads to compute a per-polygon face normal
static void PerBitPolygonFaceNormalized(ComputeMeshType &m)
{
  PerFace(m);
  tri::RequireCompactness(m);
  tri::RequireTriangularMesh(m);
  tri::UpdateFlags<ComputeMeshType>::FaceClearV(m);
  std::vector<VertexPointer> vertVec;
  std::vector<FacePointer> faceVec;
  for(size_t i=0;i<m.face.size();++i)
    if(!m.face[i].IsV())
    {
      tri::PolygonSupport<MeshType,MeshType>::ExtractPolygon(&(m.face[i]),vertVec,faceVec);
      CoordType nf(0,0,0);
      for(size_t j=0;j<faceVec.size();++j)
        nf+=faceVec[j]->N().Normalize() * DoubleArea(*faceVec[j]);

      nf.Normalize();

      for(size_t j=0;j<faceVec.size();++j)
        faceVec[j]->N()=nf;
    }
}
/// \brief Multiply the vertex normals by the matrix passed. By default, the scale component is removed.
static void PerVertexMatrix(ComputeMeshType &m, const Matrix44<ScalarType> &mat, bool remove_scaling= true)
{
    tri::RequirePerVertexNormal(m);
    ScalarType scale;

    Matrix33<ScalarType> mat33(mat,3);


    if(remove_scaling){
        scale = pow(mat33.Determinant(),(ScalarType)(1.0/3.0));
        Point3<ScalarType> scaleV(scale,scale,scale);
        Matrix33<ScalarType> S;
        S.SetDiagonal(scaleV.V());
        mat33*=S;
    }

    for(VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
        if( !(*vi).IsD() && (*vi).IsRW() )
            (*vi).N()  = mat33*(*vi).N();
}

/// \brief Multiply the face normals by the matrix passed. By default, the scale component is removed.
static void PerFaceMatrix(ComputeMeshType &m, const Matrix44<ScalarType> &mat, bool remove_scaling= true)
{
    tri::RequirePerFaceNormal(m);
    ScalarType scale;

    Matrix33<ScalarType> mat33(mat,3);

    if( !HasPerFaceNormal(m)) return;

    if(remove_scaling){
        scale = pow(mat33.Determinant(),ScalarType(1.0/3.0));
        mat33[0][0]/=scale;
        mat33[1][1]/=scale;
        mat33[2][2]/=scale;
    }

    for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
        if( !(*fi).IsD() && (*fi).IsRW() )
            (*fi).N() = mat33* (*fi).N();
}

/// \brief Compute per wedge normals taking into account the angle between adjacent faces.
///
/// The PerWedge normals are averaged on common vertexes only if the angle between two faces is \b larger than \p angleRad.
/// It requires FFAdjacency.
static void PerWedgeCrease(ComputeMeshType &m, ScalarType angleRad)
{
  tri::RequirePerFaceWedgeNormal(m);
  tri::RequireFFAdjacency(m);

  ScalarType cosangle=math::Cos(angleRad);

  // Clear the per wedge normals
  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi) if(!(*fi).IsD())
  {
    (*fi).WN(0)=NormalType(0,0,0);
    (*fi).WN(1)=NormalType(0,0,0);
    (*fi).WN(2)=NormalType(0,0,0);
  }

  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)		 if(!(*fi).IsD())
  {
    NormalType nn= TriangleNormal(*fi);
    for(int i=0;i<3;++i)
    {
      const NormalType &na=TriangleNormal(*(*fi).FFp(i));
      if(nn*na > cosangle )
      {
        fi->WN((i+0)%3) +=na;
        fi->WN((i+1)%3) +=na;
      }
    }
  }
}


static void PerFaceRW(ComputeMeshType &m, bool normalize=false)
{
  tri::RequirePerFaceNormal(m);
    FaceIterator f;
    bool cn = true;

    if(normalize)
    {
        for(f=m.m.face.begin();f!=m.m.face.end();++f)
        if( !(*f).IsD() && (*f).IsRW() )
        {
            for(int j=0; j<3; ++j)
                if( !(*f).V(j)->IsR()) 	cn = false;
      if( cn )     f->N() = TriangleNormal(*f).Normalize();
            cn = true;
        }
    }
    else
    {
        for(f=m.m.face.begin();f!=m.m.face.end();++f)
            if( !(*f).IsD() && (*f).IsRW() )
            {
                for(int j=0; j<3; ++j)
                    if( !(*f).V(j)->IsR()) 	cn = false;

                if( cn )
                  f->N() = TriangleNormal(*f).Normalize();
                cn = true;
            }
    }
}


}; // end class

}	// End namespace
}	// End namespace

#endif
