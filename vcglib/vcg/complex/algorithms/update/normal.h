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
#if 0// JPB WIP BUG Revised just bad
static __forceinline int vpIndex(const ComputeMeshType& m, typename ComputeMeshType::VertexPointer vp)
{
    // Works only if vertices are stored contiguously (true for vcglib typical meshes).
    return int(vp - &m.vert[0]);
}

static void PerVertexAngleWeighted(ComputeMeshType& m)
{
    using Scalar = typename ComputeMeshType::ScalarType;
    using Coord = typename ComputeMeshType::CoordType;
    using NormalType = typename ComputeMeshType::VertexType::NormalType;

    const int nV = int(m.vert.size());
    const int nF = int(m.face.size());
    if (nV == 0 || nF == 0) {
        return;
    }

    // Clear destination
    vcg::tri::UpdateNormal<ComputeMeshType>::PerVertexClear(m);

    // Thread-local accumulators
    int nThreads = 1;
#ifdef _OPENMP
    nThreads = omp_get_max_threads();
#endif
    struct alignas(64) PaddedNormal { NormalType n; };
    std::vector<std::vector<PaddedNormal>> tls(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        tls[t].assign(nV, PaddedNormal{ NormalType(0,0,0) });
    }

    // Parallel pass over faces: accumulate into thread-local arrays
#pragma omp parallel for schedule(static)
    for (int f = 0; f < nF; ++f) {
        auto& face = m.face[f];
        if (face.IsD() || !face.IsR()) continue;

        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = tls[tid];

        // Triangle normal (unit)
        NormalType t = vcg::TriangleNormal(face);
        t.Normalize();

        // Edge directions for angle at each corner
        NormalType e0 = (face.V1(0)->cP() - face.V0(0)->cP());
        NormalType e1 = (face.V1(1)->cP() - face.V0(1)->cP());
        NormalType e2 = (face.V1(2)->cP() - face.V0(2)->cP());
        e0.Normalize(); e1.Normalize(); e2.Normalize();

        // Corner angle weights (use the same AngleN as your scalar product-based angle)
        const Scalar a0 = AngleN(e0, -e2);
        const Scalar a1 = AngleN(-e0, e1);
        const Scalar a2 = AngleN(-e1, e2);

        // Indices
        const int i0 = vpIndex(m, face.V(0));
        const int i1 = vpIndex(m, face.V(1));
        const int i2 = vpIndex(m, face.V(2));

        acc[i0].n += t * a0;
        acc[i1].n += t * a1;
        acc[i2].n += t * a2;
    }

    // Reduce thread-local sums into vertex normals
    // Tree-style reduction to improve cache locality
    int stride = 1;
    while (stride < nThreads) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nV; ++i) {
            for (int t = 0; t + stride < nThreads; t += 2 * stride) {
                tls[t][i].n += tls[t + stride][i].n;
            }
        }
        stride <<= 1;
    }

    // Write back and normalize once
    {
        auto& sum = tls[0];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nV; ++i) {
            if (m.vert[i].IsD()) continue;
            m.vert[i].N() = sum[i].n;
            m.vert[i].N().Normalize();
        }
    }
}
#else
static void PerVertexAngleWeighted(ComputeMeshType &m)
{
  PerVertexClear(m);
  FaceIterator f;
  for(f=m.face.begin();f!=m.face.end();++f)
   if( !(*f).IsD() && (*f).IsR() )
   {
        NormalType t = TriangleNormal(*f).Normalize();
        NormalType e0 = ((*f).V1(0)->cP()-(*f).V0(0)->cP()).Normalize();
        NormalType e1 = ((*f).V1(1)->cP()-(*f).V0(1)->cP()).Normalize();
        NormalType e2 = ((*f).V1(2)->cP()-(*f).V0(2)->cP()).Normalize();

        (*f).V(0)->N() += t*AngleN(e0,-e2);
        (*f).V(1)->N() += t*AngleN(-e0,e1);
        (*f).V(2)->N() += t*AngleN(-e1,e2);
   }
}
#endif

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
static void PerFace(ComputeMeshType &m)
{
    RequirePerFaceNormal(m); // JPB WIP BUG What is this doing?

    int64_t numFaces = m.fn;
#pragma omp parallel for
    for (int64_t i = 0; i < numFaces; ++i) {
        auto& f = m.face[i];
        if ( !f.IsD() ) {
            f.N() = TriangleNormal(f);
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
        if (!v.IsD && v.isRW()) {
            v.N().Normalize();
        }
    }
}

/// \brief Normalize the length of the face normals.
static void NormalizePerFace(ComputeMeshType &m)
{
  tri::RequirePerFaceNormal(m);
  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
      if( !(*fi).IsD() )	(*fi).N().Normalize();
}

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
