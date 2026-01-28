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

#ifndef __VCG_TRI_UPDATE_TOPOLOGY
#define __VCG_TRI_UPDATE_TOPOLOGY

#include <cassert>

#include <vcg/complex/base.h>
#include <vcg/simplex/face/topology.h>
#include <vcg/simplex/edge/pos.h>
#include <execution>
#include <tbb/parallel_sort.h>

#define FAST_FILLEDGEVECTOR
#define FAST_VERTEXFACE

namespace vcg {
namespace tri {
/// \ingroup trimesh

/// \headerfile topology.h vcg/complex/algorithms/update/topology.h

/// \brief Generation of per-vertex and per-face topological information.

template <class UpdateMeshType>
class UpdateTopology
{

public:
  typedef UpdateMeshType MeshType;
  typedef typename MeshType::ScalarType     ScalarType;
  typedef typename MeshType::VertexType     VertexType;
  typedef typename MeshType::VertexPointer  VertexPointer;
  typedef typename MeshType::VertexIterator VertexIterator;
  typedef typename MeshType::EdgeType       EdgeType;
  typedef typename MeshType::EdgePointer    EdgePointer;
  typedef typename MeshType::EdgeIterator   EdgeIterator;
  typedef typename MeshType::FaceType       FaceType;
  typedef typename MeshType::FacePointer    FacePointer;
  typedef typename MeshType::FaceIterator   FaceIterator;
  typedef typename MeshType::TetraType      TetraType;
  typedef typename MeshType::TetraPointer   TetraPointer;
  typedef typename MeshType::TetraIterator  TetraIterator;


  /// \headerfile topology.h vcg/complex/algorithms/update/topology.h

  /// \brief Auxiliary data structure for computing tetra tetra adjacency information.
  /**
   * It identifies a face, storing three vertex pointers and a tetra pointer where it belongs.
   */

  class PFace
  {
  public:
    VertexPointer v[3];  //three ordered vertex pointers, identify a face
    TetraPointer  t;     //the pointer to the tetra where this face belongs
    int           z;     //index in [0..3] of the face in the tetra
    bool   isBorder;

    PFace() {}
    PFace(TetraPointer tp, const int nz) { this->Set(tp, nz); }

    void Set(TetraPointer tp /*the tetra pointer*/, const int nz /*the face index*/)
    {
      assert(tp != 0);
      assert(nz >= 0 && nz < 4);

      v[0] = tp->V(Tetra::VofF(nz, 0));
      v[1] = tp->V(Tetra::VofF(nz, 1));
      v[2] = tp->V(Tetra::VofF(nz, 2));

      assert(v[0] != v[1] && v[1] != v[2]); //no degenerate faces

      if (v[0] > v[1])
        std::swap(v[0], v[1]);
      if (v[1] > v[2])
        std::swap(v[1], v[2]);
      if (v[0] > v[1])
        std::swap(v[0], v[1]);

      t = tp;
      z = nz;


    }

    inline bool operator < (const PFace& pf) const
    {
      if (v[0] < pf.v[0])
        return true;
      else
      {
        if (v[0] > pf.v[0]) return false;

        if (v[1] < pf.v[1])
          return true;
        else
        {
          if (v[1] > pf.v[1]) return false;

          return (v[2] < pf.v[2]);
        }
      }
    }

    inline bool operator == (const PFace& pf) const
    {
      return v[0] == pf.v[0] && v[1] == pf.v[1] && v[2] == pf.v[2];
    }
  };

  static void FillFaceVector(MeshType& m, std::vector<PFace>& fvec)
  {
    ForEachTetra(m, [&fvec](TetraType& t) {
      for (int i = 0; i < 4; ++i)
        fvec.push_back(PFace(&t, i));
      });
  }

  static void FillUniqueFaceVector(MeshType& m, std::vector<PFace>& fvec)
  {
    FillFaceVector(m, fvec);
    std::sort(fvec.begin(), fvec.end());
    typename std::vector<PFace>::iterator newEnd = std::unique(fvec.begin(), fvec.end());
  }

  /// \brief Auxiliairy data structure for computing face face adjacency information.
  /**
  It identifies and edge storing two vertex pointer and a face pointer where it belong.
  */
  class PEdge
  {
  public:

    VertexPointer  v[2];  // the two Vertex pointer are ordered!
    FacePointer    f;     // the face where this edge belong
    int            z;     // index in [0..2] of the edge of the face
    bool isBorder;

    PEdge() {}
    PEdge(FacePointer  pf, const int nz) { this->Set(pf, nz); }
    void Set(FacePointer  pf, const int nz)
    {
      assert(pf != 0);
      assert(nz >= 0);
      assert(nz < pf->VN());

      v[0] = pf->V(nz);
      v[1] = pf->V(pf->Next(nz));
      assert(v[0] != v[1]); // The face pointed by 'f' is Degenerate (two coincident vertexes)

      if (v[0] > v[1]) std::swap(v[0], v[1]);
      f = pf;
      z = nz;
    }

    inline bool operator <  (const PEdge& pe) const noexcept
    {
      if (v[0] < pe.v[0]) return true;
      else if (v[0] > pe.v[0]) return false;
      else return v[1] < pe.v[1];
    }

    inline bool operator == (const PEdge& pe) const noexcept
    {
      return v[0] == pe.v[0] && v[1] == pe.v[1];
    }
    /// Convert from edge barycentric coord to the face baricentric coord a point on the current edge.
    /// Face barycentric coordinates are relative to the edge face.
    inline Point3<ScalarType> EdgeBarycentricToFaceBarycentric(ScalarType u) const
    {
      Point3<ScalarType> interp(0, 0, 0);
      interp[this->z] = u;
      interp[(this->z + 1) % 3] = 1.0f - u;
      return interp;
    }
  };

  class PEdge2
  {
  public:

    uint64_t key;     // (uint64_t(v0) << 32) | v1
    FacePointer    f;     // the face where this edge belong
    int            z;     // index in [0..2] of the edge of the face
    bool isBorder;

    __forceinline bool operator <  (const PEdge2& pe) const noexcept
    {
      return key < pe.key;
    }

    __forceinline bool operator == (const PEdge2& pe) const noexcept
    {
      return key == pe.key;
    }
    /// Convert from edge barycentric coord to the face baricentric coord a point on the current edge.
    /// Face barycentric coordinates are relative to the edge face.
    inline Point3<ScalarType> EdgeBarycentricToFaceBarycentric(ScalarType u) const
    {
      Point3<ScalarType> interp(0, 0, 0);
      interp[this->z] = u;
      interp[(this->z + 1) % 3] = 1.0f - u;
      return interp;
    }
  };

#ifdef FAST_FILLEDGEVECTOR
  /// Fill a vector with all the edges of the mesh.
  /// each edge is stored in the vector the number of times that it appears in the mesh, with the referring face.
  /// optionally it can skip the faux edges (to retrieve only the real edges of a triangulated polygonal mesh)
  static constexpr int edgeNext[] = { 1, 2, 0 };
#if  1
  static void FillEdgeVector(MeshType& m, PEdge2*& edgeVec, uint32_t& numEdges)
  {
    const int64_t faceCount = (int64_t)m.face.size();
    if (faceCount == 0) {
      edgeVec = nullptr;
      numEdges = 0;
      return;
    }

    const int threadCount = omp_get_max_threads();
    const size_t upperBound = (size_t)faceCount * 3;

    // Allocate aligned memory (not zeroed)
    edgeVec = (PEdge2*)_aligned_malloc(sizeof(PEdge2) * upperBound, 64);
    if (!edgeVec)
      throw std::bad_alloc();

#if 0
    // Faster to touch each cache line before we begin.
    // Optional warm-up for large buffers (>24 MB)
    if (upperBound > (1 << 20)) {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        const int T = omp_get_num_threads();
        const size_t chunk = (upperBound + T - 1) / T;
        const size_t start = tid * chunk;
        const size_t end = std::min(start + chunk, upperBound);
        const size_t step = 64 / sizeof(PEdge2); // touch once per cache line

        for (size_t i = start; i < end; i += step)
          edgeVec[i].key = 0;
      }
    }
#endif

    auto* const v0 = &m.vert[0];
    std::vector<size_t> threadCountOut(threadCount, 0);
    std::vector<size_t> threadOffset(threadCount, 0);

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int T = omp_get_num_threads();
      const size_t chunk = (faceCount + T - 1) / T;
      const size_t startFace = tid * chunk;
      const size_t endFace = std::min<size_t>(startFace + chunk, faceCount);

      // Each thread owns its own contiguous region
      const size_t startOut = startFace * 3;
      PEdge2* out = edgeVec + startOut;
      size_t outCount = 0;

      PEdge2 localEdges[4096]; // ~96 KB, fits in L2

      for (size_t i = startFace; i < endFace; ++i)
      {
        auto& f = m.face[i];
        if (f.IsD()) continue;

        const int vn = f.VN();
        for (int j = 0; j < vn; ++j)
        {
          const int jNext = edgeNext[j];

          // Although not likely needed, we force define every edge we emit.
          PEdge2 e;
          e.f = &f;
          e.z = (uint8_t)j;
          e.isBorder = false;
          size_t i0 = f.V(j) - v0;
          size_t i1 = f.V(jNext) - v0;
          if (i0 > i1) std::swap(i0, i1);
          e.key = (uint64_t(i0) << 32) | uint32_t(i1);

          localEdges[outCount++] = e;

          // Flush every 4096 edges (~96 KB)
          if (outCount == 4096)
          {
            std::memcpy(out, localEdges, outCount * sizeof(PEdge2));
            out += outCount;
            outCount = 0;
          }
        }
      }

      // Flush any remaining edges
      if (outCount)
      {
        std::memcpy(out, localEdges, outCount * sizeof(PEdge2));
        out += outCount;
      }

      threadOffset[tid] = startOut;
      threadCountOut[tid] = (size_t)(out - (edgeVec + startOut));
    }

    // Compact contiguous regions (rarely needed)
    size_t offset = 0;
    for (int t = 0; t < threadCount; ++t)
    {
      size_t count = threadCountOut[t];
      size_t srcOffset = threadOffset[t];
      if (count > 0 && offset != srcOffset)
        std::memmove(edgeVec + offset, edgeVec + srcOffset, count * sizeof(PEdge2));
      offset += count;
    }

    numEdges = (uint32_t)offset;
  }



#else
  static void FillEdgeVector(MeshType& m, std::vector<PEdge2>& edgeVec)
  {
    const int64_t faceCount = (int64_t)m.face.size();
    const int T = omp_get_max_threads();

    size_t aliveCount;
    if (m.hasDeletedFaces)
    {
      // Step 1: compact
      auto firstDeleted = std::partition(m.face.begin(), m.face.begin() + faceCount,
        [](auto& f) { return !f.IsD(); });
      aliveCount = std::distance(m.face.begin(), firstDeleted);
    }
    else
    {
      aliveCount = faceCount;
    }

    // Step 2: reserve edges (upper bound is 3 per face)
    edgeVec.resize(aliveCount * 3);

    // Step 3: parallel fill
    auto* start = &m.vert[0];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)aliveCount; ++i) {
      auto& f = m.face[i];
      int vn = f.VN();
      size_t base = i * 3; // each face reserves exactly 3 slots
      size_t pos = 0;
      for (int j = 0; j < vn; ++j) {
        const int jNext = edgeNext[j];
        PEdge2& e = edgeVec[base + pos++];
        e.f = &f; e.z = (uint8_t)j;
        size_t i0 = f.V(j) - start;
        size_t i1 = f.V(jNext) - start;
        if (i0 > i1) std::swap(i0, i1);
        e.key = (uint64_t(i0) << 32) | uint32_t(i1);
      }
    }
  }
#endif
#else
static void FillEdgeVector(MeshType& m, std::vector<PEdge>& edgeVec, bool includeFauxEdge=true)
{
  edgeVec.reserve(m.fn*3);
  for (FaceIterator fi=m.face.begin(); fi!=m.face.end(); ++fi)
    if (!(*fi).IsD())
      for (int j=0; j<(*fi).VN(); ++j)
        if (includeFauxEdge || !(*fi).IsF(j))
          edgeVec.push_back(PEdge(&*fi, j));
}
#endif

static void FillUniqueEdgeVector(MeshType& m, PEdge2*& edgeVec, uint32_t& numEdges, bool includeFauxEdge = true, bool computeBorderFlag = false)
{
  if (!includeFauxEdge)
  {
    throw std::runtime_error("Unsupported");
  }

  FillEdgeVector(m, edgeVec, numEdges);
  tbb::parallel_sort(edgeVec, edgeVec+numEdges); // oredering by vertex

  if (computeBorderFlag) {
    for (size_t i = 0; i < numEdges; i++)
      edgeVec[i].isBorder = true;
    for (size_t i = 1; i < numEdges; i++) {
      if (edgeVec[i] == edgeVec[i - 1])
        edgeVec[i].isBorder = edgeVec[i - 1].isBorder = false;
    }
  }

  auto* newEnd = std::unique(std::execution::par_unseq, edgeVec, edgeVec+ numEdges);

  numEdges = newEnd - edgeVec; // redundant! remove?
}

static void FillSelectedFaceEdgeVector(MeshType &m, std::vector<PEdge> &edgeVec)
{
  edgeVec.reserve(m.fn*3);
  ForEachFace(m, [&](FaceType &f){
    for(int j=0;j<f.VN();++j)
      if(f.IsFaceEdgeS(j))
        edgeVec.push_back(PEdge(&f,j));
        });

  sort(edgeVec.begin(), edgeVec.end()); // oredering by vertex
  edgeVec.erase(std::unique(edgeVec.begin(), edgeVec.end()),edgeVec.end()); 
}



/*! \brief Initialize the edge vector all the edges that can be inferred from current face vector, setting up all the current adjacency relations
 *
 *
 */

static void AllocateEdge(MeshType &m)
{
  // Delete all the edges (if any)
  for(EdgeIterator ei=m.edge.begin();ei!=m.edge.end();++ei)
        tri::Allocator<MeshType>::DeleteEdge(m,*ei);
  tri::Allocator<MeshType>::CompactEdgeVector(m);

  // Compute and add edges
  PEdge2* Edges = 0;
  uint32_t numEdges = 0;
  FillUniqueEdgeVector(m,Edges,numEdges,true,tri::HasPerEdgeFlags(m) );
  assert(m.edge.empty());
  tri::Allocator<MeshType>::AddEdges(m,numEdges);
  assert(m.edge.size()== numEdges);

  // Setup adjacency relations
  if(tri::HasEVAdjacency(m))
  {
    const int64_t cnt = (int64_t)numEdges;
    bool hasPerEdgeFlags = tri::HasPerEdgeFlags(m);
#pragma omp parallel for // No conditional
    for (int64_t i = 0; i < cnt; ++i) {
      const auto& srcEdge = Edges[i];
      auto srcV0 = srcEdge.key >> 32;
      auto srcV1 = srcEdge.key & 0xFFFFFFFF;
			auto& dstEdge = m.edge[i];
      const auto v1 = &m.vert[srcV0];
      const auto v2 = &m.vert[srcV1];
      dstEdge.V(0) = v1;
      dstEdge.V(1) = v2;
      if (hasPerEdgeFlags) {
        if (srcEdge.isBorder) dstEdge.SetB(); else dstEdge.ClearB();
      }
    }
  } else {
    if (tri::HasPerEdgeFlags(m)){
      for(size_t i=0; i< numEdges; ++i) {
          if (Edges[i].isBorder) m.edge[i].SetB(); else m.edge[i].ClearB();
      }
    }
  }

  if(tri::HasEFAdjacency(m)) // Note it is an unordered relation.
  {
    for(size_t i=0; i< numEdges; ++i)
    {
      std::vector<FacePointer> fpVec;
      std::vector<int> eiVec;
      face::EFStarFF(Edges[i].f,Edges[i].z,fpVec,eiVec);
      m.edge[i].EFp() = Edges[i].f;
      m.edge[i].EFi() = Edges[i].z;
    }
  }

  if(tri::HasFEAdjacency(m))
  {
    for(size_t i=0; i< numEdges; ++i)
    {
      std::vector<FacePointer> fpVec;
      std::vector<int> eiVec;
      face::EFStarFF(Edges[i].f,Edges[i].z,fpVec,eiVec);
      for(size_t j=0;j<fpVec.size();++j)
        fpVec[j]->FEp(eiVec[j])=&(m.edge[i]);

//      Edges[i].f->FE(Edges[i].z) = &(m.edge[i]);
//      Connect in loop the non manifold
//      FaceType* fpit=fp;
//      int eit=ei;

//      do
//      {
//        faceVec.push_back(fpit);
//        indVed.push_back(eit);
//        FaceType *new_fpit = fpit->FFp(eit);
//        int       new_eit  = fpit->FFi(eit);
//        fpit=new_fpit;
//        eit=new_eit;
//      } while(fpit != fp);


//      m.edge[i].EFp() = Edges[i].f;
//      m.edge[i].EFi() = ;
    }
  }
  
  _aligned_free(Edges);
}

/// \brief Clear the tetra-tetra topological relation, setting each involved pointer to null.
/// useful when you passed a mesh with tt adjacency to an algorithm that does not use it and chould have messed it
static void ClearTetraTetra (MeshType & m)
{
  RequireTTAdjacency(m);
  ForEachTetra(m, [] (TetraType & t) {
      for (int i = 0; i < 4; ++i)
      {
        t.TTp(i) = NULL;
        t.TTi(i) = -1;
      }
  });
}

/// \brief Updates the Tetra-Tetra topological relation by allowing to retrieve for each tetra what other tetras share their faces.
static void TetraTetra (MeshType & m)
{
  RequireTTAdjacency(m);
  if (m.tn == 0) return;

  std::vector<PFace> fvec;
  FillFaceVector(m, fvec);
  std::sort(fvec.begin(), fvec.end());

  int nf = 0;
  typename std::vector<PFace>::iterator pback, pfront;
  pback  = fvec.begin();
  pfront = fvec.begin();

  do 
  {
    if (pfront == fvec.end() || !(*pfront == *pback))
    {
      typename std::vector<PFace>::iterator q, q_next;
      for (q = pback; q < pfront - 1; ++q)
      {
        assert((*q).z >= 0);
        q_next = q;
        ++q_next;
        assert((*q_next).z >= 0 && (*q_next).z < 4);
        
        (*q).t->TTp(q->z) = (*q_next).t;
        (*q).t->TTi(q->z) = (*q_next).z;
      }
      
      (*q).t->TTp(q->z) = pback->t;
      (*q).t->TTi(q->z) = pback->z;
      pback = pfront;
      ++nf;
    }
    if (pfront == fvec.end()) break;
    ++pfront;
  } while (true);
}
/// \brief Clear the Face-Face topological relation setting each involved pointer to null.
/// useful when you passed a mesh with ff adjacency to an algorithm that does not use it and could have messed it.
static void ClearFaceFace(MeshType &m)
{
  RequireFFAdjacency(m);
  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
  {
    if( ! (*fi).IsD() )
    {
      for(int j=0;j<fi->VN();++j)
      {
        fi->FFp(j)=0;
        fi->FFi(j)=-1;
      }
    }
  }
}

/// \brief Update the Face-Face topological relation by allowing to retrieve for each face what other faces shares their edges.
static void FaceFace(MeshType& m)
{
  RequireFFAdjacency(m);
  if (m.fn == 0) return;

  PEdge2* edges;
  uint32_t numEdges = 0;
  FillEdgeVector(m, edges, numEdges);   // or <false>, depending on need

  // Sort by canonical vertex pair
  tbb::parallel_sort(edges, edges+numEdges);

  // Find run boundaries
  std::vector<size_t> runStarts;
  runStarts.reserve(numEdges+1);
  runStarts.push_back(0);
  for (uint32_t i = 1; i < numEdges; ++i)
  {
    if (!(edges[i] == edges[i - 1]))
      runStarts.push_back(i);
  }
  runStarts.push_back(numEdges);

  // Parallel wiring
  const ptrdiff_t nRuns = (ptrdiff_t)runStarts.size() - 1;

#pragma omp parallel for schedule(static, 10000)
  for (ptrdiff_t r = 0; r < nRuns; ++r)
  {
    uint32_t i = runStarts[r];
    uint32_t j = runStarts[r + 1];

    uint32_t next = i + 1;
    for (uint32_t k = i; k < j; ++k)
    {
      if (next == j)
        next = i;

      PEdge2& a = edges[k];
      PEdge2& b = edges[next];

      auto* __restrict fa = a.f;
      fa->FFp(a.z) = b.f;
      fa->FFi(a.z) = b.z;

      ++next;
    }
  }

  _aligned_free(edges);
}

/// \brief Update the vertex-tetra topological relation.
static void VertexTetra(MeshType & m)
{
  RequireVTAdjacency(m);

  
  ForEachVertex(m, [] (VertexType & v) {
      v.VTp() = NULL;
      v.VTi() = 0;
  });

  ForEachTetra(m, [] (TetraType & t) {
    //this works like this: the first iteration defines the end of the chain
    //then it backwards chains everything
      for (int i = 0; i < 4; ++i)
      {
        t.VTp(i) = t.V(i)->VTp();
        t.VTi(i) = t.V(i)->VTi();
        t.V(i)->VTp() = &t;
        t.V(i)->VTi() = i;
      }
  });
}
/// \brief Update the Vertex-Face topological relation.
/**
The function allows to retrieve for each vertex the list of faces sharing this vertex.
After this call all the VF component are initialized. Isolated vertices have a null list of faces.
\sa vcg::vertex::VFAdj
\sa vcg::face::VFAdj
*/

static void VertexFace(MeshType &m)
{
  RequireVFAdjacency(m);

#ifdef FAST_VERTEXFACE
  const int64_t numVertices = (int64_t)m.vert.size();
#pragma omp parallel for
   for (int64_t i = 0; i < numVertices; ++i) {
      auto& vi = m.vert[i];
      vi.VFp() = 0;
      vi.VFi() = 0; // note that (0,-1) means uninitiazlied while 0,0 is the valid initialized values for isolated vertices.
   }
#else
  for(VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
  {
    (*vi).VFp() = 0;
    (*vi).VFi() = 0; // note that (0,-1) means uninitiazlied while 0,0 is the valid initialized values for isolated vertices.
  }
#endif

  if (m.hasDeletedFaces)
  {
    for (auto fi = m.face.begin(), fe = m.face.end(); fi != fe; ++fi) {
      auto& f = *fi;
      if (f.IsD()) continue;

      const int cnt = f.VN();
      for (int j = 0; j < cnt; ++j) {
        auto v = f.V(j);   // pointer to vertex

        f.VFp(j) = v->VFp();
        f.VFi(j) = v->VFi();
        v->VFp() = &f;
        v->VFi() = j;
      }
    }
  }
  else
  {
    for (auto fi = m.face.begin(), fe = m.face.end(); fi != fe; ++fi) {
      auto& f = *fi;
      const int cnt = f.VN();
      for (int j = 0; j < cnt; ++j) {
        auto v = f.V(j);   // pointer to vertex

        f.VFp(j) = v->VFp();
        f.VFi(j) = v->VFi();
        v->VFp() = &f;
        v->VFi() = j;
      }
    }
  }
}


/// \headerfile topology.h vcg/complex/algorithms/update/topology.h

/// \brief Auxiliairy data structure for computing face face adjacency information.
/**
It identifies and edge storing two vertex pointer and a face pointer where it belong.
*/

class PEdgeTex
{
public:

  typename FaceType::TexCoordType  v[2];		// the two TexCoord are ordered!
  FacePointer    f;                       // the face where this edge belong
  int      z;				      // index in [0..2] of the edge of the face

  PEdgeTex() {}

  void Set( FacePointer  pf, const int nz )
  {
    assert(pf!=0);
    assert(nz>=0);
    assert(nz<3);

    v[0] = pf->WT(nz);
    v[1] = pf->WT(pf->Next(nz));
    assert(v[0] != v[1]); // The face pointed by 'f' is Degenerate (two coincident vertexes)

    if( v[1] < v[0] ) std::swap(v[0],v[1]);
    f    = pf;
    z    = nz;
  }

  inline bool operator <  ( const PEdgeTex & pe ) const
  {
    if( v[0]<pe.v[0] ) return true;
    else if( pe.v[0]<v[0] ) return false;
    else return v[1] < pe.v[1];
  }
  inline bool operator == ( const PEdgeTex & pe ) const
  {
    return (v[0]==pe.v[0]) && (v[1]==pe.v[1]);
  }
  inline bool operator != ( const PEdgeTex & pe ) const
  {
    return (v[0]!=pe.v[0]) || (v[1]!=pe.v[1]);
  }

};


/// \brief Update the Face-Face topological relation so that it reflects the per-wedge texture connectivity

/**
Using this function two faces are adjacent along the FF relation IFF the two faces have matching texture coords along the involved edge.
In other words F1->FFp(i) == F2 iff F1 and F2 have the same tex coords along edge i
*/

static void FaceFaceFromTexCoord(MeshType &m)
{
  RequireFFAdjacency(m);
  RequirePerFaceWedgeTexCoord(m);
  vcg::tri::UpdateTopology<MeshType>::FaceFace(m);
  for (FaceIterator fi = m.face.begin(); fi != m.face.end(); ++fi)
  {
    if (!(*fi).IsD())
    {
      for (int i = 0; i < (*fi).VN(); i++)
      {
        if (!vcg::face::IsBorder((*fi), i))
        {
          typename MeshType::FacePointer nextFace = (*fi).FFp(i);
          int nextEdgeIndex = (*fi).FFi(i);
          bool border = false;
          if ((*fi).cV(i) == nextFace->cV(nextEdgeIndex))
          {
            if ((*fi).WT(i) != nextFace->WT(nextEdgeIndex) || (*fi).WT((*fi).Next(i)) != nextFace->WT(nextFace->Next(nextEdgeIndex)))
              border = true;
          }
          else
          {
            if ((*fi).WT(i) != nextFace->WT(nextFace->Next(nextEdgeIndex)) || (*fi).WT((*fi).Next(i)) != nextFace->WT(nextEdgeIndex))
              border = true;
          }
          if (border)
            vcg::face::FFDetach((*fi), i);

        }
      }
    }
  }
}

/// \brief Test correctness of VEtopology
static void TestVertexEdge(MeshType &m)
{
  std::vector<int> numVertex(m.vert.size(),0);
  
  tri::RequireVEAdjacency(m);
  
  for(EdgeIterator ei=m.edge.begin();ei!=m.edge.end();++ei)
  {
      if (!(*ei).IsD())
      {
        assert(tri::IsValidPointer(m,ei->V(0)));
        assert(tri::IsValidPointer(m,ei->V(1)));
        if(ei->VEp(0)) assert(tri::IsValidPointer(m,ei->VEp(0)));
        if(ei->VEp(1)) assert(tri::IsValidPointer(m,ei->VEp(1)));
        numVertex[tri::Index(m,(*ei).V(0))]++;
        numVertex[tri::Index(m,(*ei).V(1))]++;
      }
  }
  
  for(VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
  {
      if (!vi->IsD())
      {
        int cnt =0;
        for(edge::VEIterator<EdgeType> vei(&*vi);!vei.End();++vei)
          cnt++;
        assert((numVertex[tri::Index(m,*vi)] == 0) == (vi->VEp()==0) );
        assert(cnt==numVertex[tri::Index(m,*vi)]);        
      }
  }  
}


/// \brief Test correctness of VFtopology
static void TestVertexFace(MeshType &m)
{
    SimpleTempData<typename MeshType::VertContainer, int > numVertex(m.vert,0);

  assert(tri::HasPerVertexVFAdjacency(m));

    for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
    {
        if (!(*fi).IsD())
        {
            numVertex[(*fi).V0(0)]++;
            numVertex[(*fi).V1(0)]++;
            numVertex[(*fi).V2(0)]++;
        }
    }

    vcg::face::VFIterator<FaceType> VFi;

    for(VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
    {
        if (!vi->IsD())
        if(vi->VFp()!=0) // unreferenced vertices MUST have VF == 0;
        {
            int num=0;
            assert(tri::IsValidPointer(m, vi->VFp()));
            VFi.f=vi->VFp();
            VFi.z=vi->VFi();
            while (!VFi.End())
            {
                num++;
                assert(!VFi.F()->IsD());
                assert((VFi.F()->V(VFi.I()))==&(*vi));
                ++VFi;
            }
            assert(num==numVertex[&(*vi)]);
        }
    }
}

/// \brief Test correctness of FFtopology (only for 2Manifold Meshes!)
static void TestFaceFace(MeshType &m)
{
  assert(HasFFAdjacency(m));

  for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
    {
    if (!fi->IsD())
        {
      for (int i=0;i<(*fi).VN();i++)
            {
        FaceType *ffpi=fi->FFp(i);
        int e=fi->FFi(i);
        //invariant property of FF topology for two manifold meshes
        assert(ffpi->FFp(e) == &(*fi));
        assert(ffpi->FFi(e) == i);

        // Test that the two faces shares the same edge
        // Vertices of the i-th edges of the first face
        VertexPointer v0i= fi->V0(i);
        VertexPointer v1i= fi->V1(i);
        // Vertices of the corresponding edge on the other face
        VertexPointer ffv0i= ffpi->V0(e);
        VertexPointer ffv1i= ffpi->V1(e);

        assert( (ffv0i==v0i) || (ffv0i==v1i) );
        assert( (ffv1i==v0i) || (ffv1i==v1i) );
            }

        }
    }
}

/// Auxiliairy data structure for computing edge edge adjacency information.
/// It identifies an edge storing a vertex pointer and a edge pointer where it belong.
class PVertexEdge
{
public:

  VertexPointer  v;		// the two Vertex pointer are ordered!
  EdgePointer    e;		  // the edge where this vertex belong
  int      z;				      // index in [0..1] of the vertex of the edge

  PVertexEdge(  ) {}
  PVertexEdge( EdgePointer  pe, const int nz )
{
  assert(pe!=0);
  assert(nz>=0);
  assert(nz<2);

  v= pe->V(nz);
  e    = pe;
  z    = nz;
}
inline bool operator  <  ( const PVertexEdge & pe ) const { return ( v<pe.v ); }
inline bool operator ==  ( const PVertexEdge & pe ) const { return ( v==pe.v ); }
inline bool operator !=  ( const PVertexEdge & pe ) const { return ( v!=pe.v ); }
};



static void EdgeEdge(MeshType &m)
{
  RequireEEAdjacency(m);
  std::vector<PVertexEdge> v;
  if( m.en == 0 ) return;

//  printf("Inserting Edges\n");
  for(EdgeIterator pf=m.edge.begin(); pf!=m.edge.end(); ++pf)			// Lo riempio con i dati delle facce
    if( ! (*pf).IsD() )
      for(int j=0;j<2;++j)
      {
//        printf("egde %i ind %i (%i %i)\n",tri::Index(m,&*pf),j,tri::Index(m,pf->V(0)),tri::Index(m,pf->V(1)));
        v.push_back(PVertexEdge(&*pf,j));
      }

//  printf("en = %i (%i)\n",m.en,m.edge.size());
  sort(v.begin(), v.end());							// Lo ordino per vertici

  int ne = 0;											// Numero di edge reali

  typename std::vector<PVertexEdge>::iterator pe,ps;
  // for(ps = v.begin(),pe=v.begin();pe<=v.end();++pe)	// Scansione vettore ausiliario
  ps = v.begin();pe=v.begin();
  do
  {
//    printf("v %i -> e %i\n",tri::Index(m,(*ps).v),tri::Index(m,(*ps).e));
    if( pe==v.end() || !(*pe == *ps) )					// Trovo blocco di edge uguali
    {
      typename std::vector<PVertexEdge>::iterator q,q_next;
      for (q=ps;q<pe-1;++q)						// Scansione edge associati
      {
        assert((*q).z>=0);
        assert((*q).z< 2);
        q_next = q;
        ++q_next;
        assert((*q_next).z>=0);
        assert((*q_next).z< 2);
        (*q).e->EEp(q->z) = (*q_next).e;				// Collegamento in lista delle facce
        (*q).e->EEi(q->z) = (*q_next).z;
      }
      assert((*q).z>=0);
      assert((*q).z< 2);
      (*q).e->EEp((*q).z) = ps->e;
      (*q).e->EEi((*q).z) = ps->z;
      ps = pe;
      ++ne;										// Aggiorno il numero di edge
    }
    if(pe==v.end()) break;
    ++pe;
   } while(true);
}

static void VertexEdge(MeshType &m)
{
  RequireVEAdjacency(m);

  for(VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
  {
    (*vi).VEp() = 0;
    (*vi).VEi() = 0;
  }

  for(EdgeIterator ei=m.edge.begin();ei!=m.edge.end();++ei)
  if( ! (*ei).IsD() )
  {
    for(int j=0;j<2;++j)
    { assert(tri::IsValidPointer(m,ei->V(j)));
      (*ei).VEp(j) = (*ei).V(j)->VEp();
      (*ei).VEi(j) = (*ei).V(j)->VEi();
      (*ei).V(j)->VEp() = &(*ei);
      (*ei).V(j)->VEi() = j;
    }
  }
}

}; // end class

}	// End namespace
}	// End namespace


#endif
