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

#ifndef __VCG_TRIMESHCOLLAPSE_QUADRIC__
#define __VCG_TRIMESHCOLLAPSE_QUADRIC__

#include<vcg/math/quadric.h>
#include<vcg/complex/algorithms/update/bounding.h>
#include<vcg/complex/algorithms/local_optimization/tri_edge_collapse.h>
#include<vcg/complex/algorithms/local_optimization.h>
#include<vcg/complex/algorithms/stat.h>
#include <cstdlib>   // getenv/atoi -- OPENMVS_MESH_QUADRIC_GATHER / _CHECK
#include <cstring>
#include <cmath>     // std::fabs -- [QUADRIC-CHECK]
#include <vector>

// MESH_DIAG comes from libs/MVS/Common.h. This header is only ever instantiated from
// libs/MVS/Mesh.cpp, which includes that first, so it is available in practice -- but
// vcglib had never referenced an OpenMVS macro before [QUADRIC-CHECK], so keep the
// header self-contained rather than relying on include order.
#ifndef MESH_DIAG
#define MESH_DIAG(...) ((void)0)
#endif

// Normally, all collapses are generated to find a "best" one.
// Enabling this forces the logic to take the first one it find that
// fulfills a quality tolerance of "likely good enough".
// Definitely changes the topology of the result.
#undef TAKE_FIRST_GOOD_COLLAPSE

// Don't collpase candidates (add to the heap), if they look poor.
#undef REJECT_BAD_CANDIDATES // Not as helpful as it seems.

double g_ScaleFactor;
constexpr uint8_t nextPacked[3] = { 0x12, 0x20, 0x01 };

namespace vcg{
namespace tri{


/**
  This class describe Quadric based collapse operation.

    Requirements:

    Vertex
    must have:
   incremental mark
   VF topology

    must have:
        members

      QuadricType Qd();

            ScalarType W() const;
                A per-vertex Weight that can be used in simplification
                lower weight means that error is lowered,
                standard: return W==1.0

            void Merge(MESH_TYPE::vertex_type const & v);
                Merges the attributes of the current vertex with the ones of v
                (e.g. its weight with the one of the given vertex, the color ect).
                Standard: void function;

      OtherWise the class should be templated with a static helper class that helps to retrieve these functions.
      If the vertex class exposes these functions a default static helper class is provided.

*/
        //**Helper CLASSES**//
        template <class VERTEX_TYPE>
        class QInfoStandard
        {
        public:
      QInfoStandard(){}
      static void Init(){}
      static math::Quadric<double> &Qd(VERTEX_TYPE &v) {return v.Qd();}
      static math::Quadric<double> &Qd(VERTEX_TYPE *v) {return v->Qd();}
      static typename VERTEX_TYPE::ScalarType W(VERTEX_TYPE * /*v*/) {return 1.0;}
      static typename VERTEX_TYPE::ScalarType W(VERTEX_TYPE &/*v*/) {return 1.0;}
      static void Merge(VERTEX_TYPE & /*v_dest*/, VERTEX_TYPE const & /*v_del*/){}
        };


class TriEdgeCollapseQuadricParameter : public BaseParameterClass
{
public:
  double    BoundaryQuadricWeight = 0.5;
  bool      FastPreserveBoundary  = false;
  bool      AreaCheck           = false;
  bool      HardQualityCheck = false;
  double    HardQualityThr = 0.1;
  bool      HardNormalCheck =  false;
  bool      NormalCheck           = false;
  double    NormalThrRad          = M_PI/2.0;
  double    CosineThr             = 0 ; // ~ cos(pi/2) 
  bool      OptimalPlacement =true;
  bool      SVDPlacement = false;
  bool      PreserveTopology =false;
  bool      PreserveBoundary = false;
  double    QuadricEpsilon = 1e-15;
  bool      QualityCheck =true;
  double    QualityThr =.3;     // Collapsed that generate faces with quality LOWER than this value are penalized. So higher the value -> better the quality of the accepted triangles
  bool      QualityQuadric =false; // During the initialization manage all the edges as border edges adding a set of additional quadrics that are useful mostly for keeping face aspect ratio good.
  double    QualityQuadricWeight = 0.001f; // During the initialization manage all the edges as border edges adding a set of additional quadrics that are useful mostly for keeping face aspect ratio good.
  bool      QualityWeight=false;
  double    QualityWeightFactor=100.0;
  double    ScaleFactor=1.0;
  bool      ScaleIndependent=true;
  bool      UseArea =true;
  bool      UseVertexWeight=false;
  //float     MaxError = 0.f;

  TriEdgeCollapseQuadricParameter() {}
};
static volatile int g_doit = 0;
static volatile uint64_t tries = 0;
static volatile double g_total = 0.;
static volatile double g_avg = 0.;
static volatile double g_totalCyclesBefore = 0.;
static volatile double g_avgCyclesBefore = 0.;

#pragma pack(push, 1)
template<class TriMeshType, class VertexPair, class MYTYPE, class HelperType = QInfoStandard<typename TriMeshType::VertexType> >
class TriEdgeCollapseQuadric: public TriEdgeCollapse< TriMeshType, VertexPair, MYTYPE, HelperType>
{
public:
  typedef typename vcg::tri::TriEdgeCollapse< TriMeshType, VertexPair, MYTYPE, HelperType> TEC;
  typedef typename TriEdgeCollapse<TriMeshType, VertexPair, MYTYPE, HelperType>::HeapType HeapType;
  typedef typename TriEdgeCollapse<TriMeshType, VertexPair, MYTYPE, HelperType>::HeapElem HeapElem;
    
  typedef typename TriMeshType::CoordType CoordType;
  typedef typename TriMeshType::ScalarType ScalarType;
  typedef typename TriMeshType::FaceType FaceType;
  typedef typename TriMeshType::VertexType VertexType;
  typedef typename TriMeshType::VertexIterator VertexIterator;
  typedef typename TriMeshType::FaceIterator FaceIterator;
  typedef typename vcg::face::VFIterator<FaceType> VFIterator;
  typedef  math::Quadric< double > QuadricType;
  typedef TriEdgeCollapseQuadricParameter QParameter;
  //typedef HelperType QH;
  
  CoordType optimalPos;  // Local storage of the once computed optimal position of the collapse.
  
  // Pointer to the vector that store the Write flags. Used to preserve them if you ask to preserve for the boundaries.
  static std::vector<typename TriMeshType::VertexPointer>  & WV(){
    static std::vector<typename TriMeshType::VertexPointer> _WV; return _WV;
  }
  
  inline TriEdgeCollapseQuadric(){}
  
  inline TriEdgeCollapseQuadric(const VertexPair &p, int i)
  {
    this->localMark = i;
    this->pos=p;
    //this->_priority = ComputePriority();
  }
  

  // ---- Staleness, split in two -------------------------------------------------
  //
  // Upstream has ONE test: the entry is usable iff neither endpoint's IMark has moved
  // since the entry was created. UpdateHeap bumps IMark on the survivor AND every ring
  // neighbour, so ~50% of entries are invalidated purely because a neighbour moved --
  // even though a collapse only changes the SURVIVOR's quadric and position, so those
  // entries' priorities are provably still correct. They then get thrown away and
  // rebuilt, which is the 32.9M redundant ComputePriority calls per run.
  //
  // Here IMark keeps meaning "the local topology was touched" and a second per-vertex
  // mark, Gm(v) (the GlobalMark at which v last SURVIVED a collapse), means "this
  // vertex's quadric/position changed". No per-entry storage is needed: localMark is
  // already the creation timestamp. Bit layout is untouched -- MYTYPE must stay exactly
  // 36 bytes, which the allocator and PtrToOffset both hardcode.
  //
  // IsUpToDateFast: cheap, const, conservative. Only rejects definitely-dead entries.
  // Used by the parallel heap compaction, which must not walk rings or mutate state.
  __forceinline bool IsUpToDateFast() const
  {
    const VertexType* __restrict v0 = this->pos.cV(0);
    if ((v0->cFlags() & 1) || this->localMark < QH::Gm(v0)) return false;
    const VertexType* __restrict v1 = this->pos.cV(1);
    if ((v1->cFlags() & 1) || this->localMark < QH::Gm(v1)) return false;
    if (!g_deciLazy) {
      // classic behaviour: topology touch alone invalidates
      if (this->localMark < v0->IMark() || this->localMark < v1->IMark()) return false;
    }
    return true;
  }

  // Full test, pop site only (single-threaded there).
  __forceinline bool IsUpToDate()
  {
    VertexType* __restrict v0 = this->pos.V(0);
    VertexType* __restrict v1 = this->pos.V(1);
    if ((v0->cFlags() & 1) || (v1->cFlags() & 1)) return false;
    if (!g_deciLazy)
      return !(this->localMark < v0->IMark() || this->localMark < v1->IMark());

    // priority validity: did either endpoint's own geometry change?
    if (this->localMark < QH::Gm(v0) || this->localMark < QH::Gm(v1)) return false;

    // topology validity: IMark is a conservative proxy. If it fired, ask the real
    // question instead of discarding -- are the two still joined by a face? That is
    // exactly what EdgeCollapser::Do needs, and it re-derives everything else from the
    // live VF adjacency at call time.
    if (this->localMark < v0->IMark() || this->localMark < v1->IMark()) {
      bool adj = false;
      vcg::face::VFIterator<FaceType> it;
      for (it.F() = v0->VFp(), it.I() = v0->VFi(); it.F() != 0; ++it) {
        const FaceType &f = *it.F();
        if (f.IsD()) continue;
        if (f.V(0) == v1 || f.V(1) == v1 || f.V(2) == v1) { adj = true; break; }
      }
      if (!adj) return false;
      this->localMark = this->GlobalMark();   // verified; don't re-walk next time
    }
    return true;
  }

  inline bool IsFeasible(BaseParameterClass *_pp){
    QParameter *pp=(QParameter *)_pp;
    if(!pp->PreserveTopology) return true;
    
    bool res = ( EdgeCollapser<TriMeshType, VertexPair>::LinkConditions(this->pos) );
    if(!res) ++( TEC::FailStat::LinkConditionEdge() );
    return res;
  }
  
  // Solves for optimalPos and leaves the SUMMED quadric q0+q1 in abc[], which the caller
  // may then evaluate itself.
  //
  // Split out of ComputePosition() so that ComputePriority can stop paying for two things
  // it threw away. MEASURED: prio = 49.6% of updateHeap = 2.218 s, 69.6 ns across 31.9M
  // calls, the single largest item in the collapse loop.
  //   1. ComputePosition's RETURN VALUE is a full ten-term Apply() at v1's position.
  //      ComputePriority stored it in `optimalError` and never read it.
  //   2. ComputePriority then re-summed the SAME two quadrics into a QuadricType -- an
  //      80-byte copy construct plus ten adds, and a second pair of loads -- purely to
  //      call Apply(), when abc[] already held exactly that sum.
  // Both removals are EXACT. abc[i] = q0.array[i] + q1.array[i] is elementwise, which is
  // what operator+= does, and the caller's inline Apply below reproduces vcg
  // Quadric::Apply term for term in the same association order, so the result is
  // bit-identical rather than merely equivalent.
  __forceinline void ComputePositionAbc(double abc[10])
  {
#if 0 // original
    CoordType newPos = (this->pos.V(0)->P() + this->pos.V(1)->P()) / 2.0;

      if ((QH::Qd(this->pos.V(0)).Apply(newPos) + QH::Qd(this->pos.V(1)).Apply(newPos)) > 2.0 * 1e-15)
      {
        QuadricType q = QH::Qd(this->pos.V(0));
        q += QH::Qd(this->pos.V(1));

        Point3<QuadricType::ScalarType> x;
          q.Minimum(x);
        newPos = CoordType::Construct(x);
      }
    this->optimalPos = newPos;
#else
    // Rework to avoid explicit degenerate checking.
    // Here we just try to find the minimum and if that
    // fails we fall back to the midpoint.
    const VertexType* __restrict v0 = this->pos.V(0);
    const VertexType* __restrict v1 = this->pos.V(1);

    const QuadricType& q0 = QH::Qd(v0);
    const QuadricType& q1 = QH::Qd(v1);

   // if (g_doit)

//    std::cout << "Using q0 q1: " << (uintptr_t)q0.array << " " << (uintptr_t)q1.array << "\n";


    // Float significantly reduces accuracy.
    using ReturnScalarType = double;

    // abc is the caller's; the local declaration that used to sit here would shadow it.
    const double* __restrict pq0 = q0.array;
    const double* __restrict pq1 = q1.array;

    for (size_t i = 0; i < 10; ++i) {
      abc[i] = pq0[i] + pq1[i]; // Not store bound
    }

    // Matrix A (symmetric)
    // Vector b (divided by 2)
    constexpr ReturnScalarType negHalf = ReturnScalarType(-0.5);
    constexpr ReturnScalarType one = ReturnScalarType(1.0);

    const ReturnScalarType b0 = abc[6+0] * negHalf;
    const ReturnScalarType b1 = abc[6+1] * negHalf;
    const ReturnScalarType b2 = abc[6+2] * negHalf;

    // Cholesky decomposition (A = L * Lt)
    const ReturnScalarType L00 = FastSqrtD(abc[0]);
    const ReturnScalarType invL00 = one / L00;

    const ReturnScalarType L10 = abc[1] * invL00;
    const ReturnScalarType L20 = abc[2] * invL00;

    const ReturnScalarType d11 = abc[3] - L10 * L10;

    const ReturnScalarType L11 = FastSqrtD(d11);
    const ReturnScalarType invL11 = one / L11;

    const ReturnScalarType L21 = (abc[4] - L10 * L20) * invL11;

    const ReturnScalarType d22 = abc[5] - L20 * L20 - L21 * L21;
    const ReturnScalarType L22 = FastSqrtD(d22);
    const ReturnScalarType invL22 = one / L22;

    // Forward substitution: solve L * y = b
    const ReturnScalarType y0 = b0 * invL00;
    const ReturnScalarType y1 = (b1 - L10 * y0) * invL11;
    const ReturnScalarType y2 = (b2 - L20 * y0 - L21 * y1) * invL22;

    // Backward substitution: solve Lt * x = y
    const ReturnScalarType x2 = y2 * invL22;
    const ReturnScalarType x1 = (y1 - L21 * x2) * invL11;
    const ReturnScalarType x0 = (y0 - L10 * x1 - L20 * x2) * invL00;

    constexpr ReturnScalarType two = ReturnScalarType(2.0);

    if ((L00 >= ReturnScalarType(1e-12)) && (d11 > ReturnScalarType(0.0)) && (d22 > ReturnScalarType(0.0))) {
      this->optimalPos = Point3f(x0, x1, x2); // accept it   
    } else {
      constexpr ReturnScalarType half = ReturnScalarType(0.5);
      this->optimalPos = (v0->cP() + v1->cP()) * half; // fallback to midpoint
    }
#endif
  }

  // Preserved for the call sites that DO want the value: identical arithmetic to before.
  double ComputePosition()
  {
    double abc[10];
    ComputePositionAbc(abc);
    constexpr double two = 2.0;
    const VertexType* __restrict v1 = this->pos.V(1);
    const double v1x = v1->cP()[0];
    const double v1y = v1->cP()[1];
    const double v1z = v1->cP()[2];
    return double(
      v1x * v1x * abc[0] + two * v1x * v1y * abc[1] + two * v1x * v1z * abc[2] + v1x * abc[6 + 0]
      + v1y * v1y * abc[3] + two * v1y * v1z * abc[4] + v1y * abc[6 + 1]
      + v1z * v1z * abc[5] + v1z * abc[6 + 2] + abc[9]
    );
  }

  void Execute(TriMeshType &m)
  {
    auto* __restrict v0 = this->pos.V(0);
    auto* __restrict v1 = this->pos.V(1);

    CoordType newPos = this->optimalPos;
    if (g_deciRecentred) {
      // Both quadrics must end up relative to where v1 is ABOUT to be. Translate each
      // from its own vertex onto newPos and sum. Both displacements are sub-edge-length.
      auto & Q1 = QH::Rq(v1);
      const auto & Q0 = QH::Rq(v0);
      double acc[10];
      for (int pass = 0; pass < 2; ++pass) {
        const auto & q = pass ? Q1 : Q0;
        const auto & r = pass ? v1->cP() : v0->cP();
        const double tx = (double)newPos[0] - (double)r[0];
        const double ty = (double)newPos[1] - (double)r[1];
        const double tz = (double)newPos[2] - (double)r[2];
        const double A0 = (double)q.a[0]*tx + (double)q.a[1]*ty + (double)q.a[2]*tz;
        const double A1 = (double)q.a[1]*tx + (double)q.a[3]*ty + (double)q.a[4]*tz;
        const double A2 = (double)q.a[2]*tx + (double)q.a[4]*ty + (double)q.a[5]*tz;
        const double nb0 = (double)q.b[0] + 2.0*A0;
        const double nb1 = (double)q.b[1] + 2.0*A1;
        const double nb2 = (double)q.b[2] + 2.0*A2;
        const double nc  = (tx*A0 + ty*A1 + tz*A2)
                         + ((double)q.b[0]*tx + (double)q.b[1]*ty + (double)q.b[2]*tz)
                         + (double)q.c;
        if (!pass) {
          for (int k = 0; k < 6; ++k) acc[k] = (double)q.a[k];
          acc[6] = nb0; acc[7] = nb1; acc[8] = nb2; acc[9] = nc;
        } else {
          for (int k = 0; k < 6; ++k) acc[k] += (double)q.a[k];
          acc[6] += nb0; acc[7] += nb1; acc[8] += nb2; acc[9] += nc;
        }
      }
      for (int k = 0; k < 6; ++k) Q1.a[k] = (float)acc[k];
      Q1.b[0] = (float)acc[6]; Q1.b[1] = (float)acc[7]; Q1.b[2] = (float)acc[8];
      Q1.c = (float)acc[9];
    } else {
    auto& q0 = QH::Qd(v1);
    q0 += QH::Qd(v0); // v0 is deleted and v1 take the new position
    }
    // v1 is the ONLY vertex whose quadric and position change, so it is the only one
    // whose dependent priorities go stale.
    //
    // GlobalMark()+1, NOT GlobalMark(). UpdateHeap opens with `mark = ++GlobalMark()`
    // and stamps IMark with that POST-increment value, giving the entries it creates the
    // same localMark -- which is why `localMark < IMark` is correctly false for them.
    // Execute runs BEFORE that increment, so stamping the current value left a
    // one-iteration window: entries created at mark M, then a collapse into one of their
    // endpoints while GlobalMark is still M, tested `M < M` and were wrongly judged
    // valid. MEASURED by [DECI-AUDIT]: 36,324 of 4,640,584 collapses (0.78%) ran on a
    // stale priority, wrong by up to 1789x. +1 is the mark UpdateHeap is about to assign,
    // so entries created after this collapse compare equal (valid) and entries created
    // before compare less (stale).
    QH::Gm(v1) = this->GlobalMark() + 1;
    EdgeCollapser<TriMeshType,VertexPair>::Do(m, this->pos, newPos); 
  }
  
  // Final Clean up after the end of the simplification process
  static void Finalize(TriMeshType &m, HeapType& /*h_ret*/, BaseParameterClass *_pp)
  {
    QParameter *pp=(QParameter *)_pp;
    
    // If we had the boundary preservation we should clean up the writable flags
    if(pp->FastPreserveBoundary)
    {
      typename 	TriMeshType::VertexIterator  vi;
      for(vi=m.vert.begin();vi!=m.vert.end();++vi)
        if(!(*vi).IsD()) (*vi).SetW();
    }
    if(pp->PreserveBoundary)
    {
      typename 	std::vector<typename TriMeshType::VertexPointer>::iterator wvi;
      for(wvi=WV().begin();wvi!=WV().end();++wvi)
        if(!(*wvi)->IsD()) (*wvi)->SetW();
    }
  }
  

  static __forceinline void fastPushHeap(HeapType& heap) {
    size_t i = heap.size() - 1;
    using Elem = typename HeapType::value_type;

    Elem val = std::move(heap[i]);
    const uint64_t valCode = val.code;

    while (i > 0) {
      const size_t parent = (i - 1) >> 1;
      const uint64_t parentCode = heap[parent].code;

      if (!(valCode < parentCode)) break;
      heap[i] = std::move(heap[parent]);
      i = parent;
    }

    heap[i] = std::move(val);
  }

  static void Init(TriMeshType &m, HeapType &h_ret, BaseParameterClass *_pp)
  {
    QParameter *pp=(QParameter *)_pp;    
    pp->CosineThr=cos(pp->NormalThrRad);
    h_ret.clear();
    {
      DeciPhaseTimer _t( g_deciSetup.topo );
    vcg::tri::UpdateTopology<TriMeshType>::VertexFace(m);
    vcg::tri::UpdateFlags<TriMeshType>::FaceBorderFromVF(m);
    }
    
    {
    DeciPhaseTimer _tb( g_deciSetup.boundary );
    if(pp->FastPreserveBoundary)
    {
      for(auto pf=m.face.begin();pf!=m.face.end();++pf)
        if( !(*pf).IsD() && (*pf).IsW() )
          for(int j=0;j<3;++j)
            if((*pf).IsB(j))
            {
              (*pf).V(j)->ClearW();
              (*pf).V1(j)->ClearW();
            }
    }
    
    if(pp->PreserveBoundary)
    {
      WV().clear();
      for(auto pf=m.face.begin();pf!=m.face.end();++pf)
        if( !(*pf).IsD() && (*pf).IsW() )
          for(int j=0;j<3;++j)
            if((*pf).IsB(j))
            {
              if((*pf).V(j)->IsW())  {(*pf).V(j)->ClearW(); WV().push_back((*pf).V(j));}
              if((*pf).V1(j)->IsW()) {(*pf).V1(j)->ClearW();WV().push_back((*pf).V1(j));}
            }
    }
    
    }   // ends the boundary-scan timing scope

    { DeciPhaseTimer _t( g_deciSetup.quadric ); InitQuadric(m,pp); }

    // Initialize the heap with all the possible collapses
    DeciPhaseTimer _th( g_deciSetup.heapBuild );   // ends with Init()
    if(IsSymmetric(pp))
    { // if the collapse is symmetric (e.g. u->v == v->u)
      h_ret.reserve(8 * m.vn);

#if 1
      thread_local std::vector<uint32_t> seenGen;
      thread_local uint32_t genCounter = 1;

      if (seenGen.size() < m.vert.size())
        seenGen.resize(m.vert.size(), 0);

      VertexType* baseVert = &m.vert[0];
      auto vi = m.vert.begin();
      auto viEnd = m.vert.end();

      for (; vi != viEnd; ++vi) {

        if (vi->IsD() || !vi->IsRW())
          continue;

        const uint32_t gen = ++genCounter;
        VertexType* v0 = &*vi;

        vcg::face::VFIterator<FaceType> x;
        for (x.F() = v0->VFp(), x.I() = v0->VFi(); x.F() != 0; ++x) {

          VertexType* v1 = x.V1();
          if (v0 < v1 && v1->IsRW()) {
            const uint32_t i1 = uint32_t(v1 - baseVert);
            if (seenGen[i1] != gen) {
              seenGen[i1] = gen;

              auto* mod = new MYTYPE(
                VertexPair(v0, v1),
                TriEdgeCollapseQuadric<TriMeshType, VertexPair, MYTYPE>::GlobalMark()
              );
              // Priority deliberately left 0 here; every entry's is computed in ONE
              // parallel pass after the enumeration finishes (see the loop at the end
              // of Init). PackedHeapElem packs priority into the high 32 bits and the
              // pool offset into the low 32, and CodeToPtr casts to uint32_t -- so a
              // zero priority still decodes to the correct object.
              h_ret.emplace_back(mod, 0u);
            }
          }

          VertexType* v2 = x.V2();
          if (v0 < v2 && v2->IsRW()) {
            const uint32_t i2 = uint32_t(v2 - baseVert);
            if (seenGen[i2] != gen) {
              seenGen[i2] = gen;

              auto* mod = new MYTYPE(
                VertexPair(v0, v2),
                TriEdgeCollapseQuadric<TriMeshType, VertexPair, MYTYPE>::GlobalMark()
              );
              // Priority deliberately left 0 here; every entry's is computed in ONE
              // parallel pass after the enumeration finishes (see the loop at the end
              // of Init). PackedHeapElem packs priority into the high 32 bits and the
              // pool offset into the low 32, and CodeToPtr casts to uint32_t -- so a
              // zero priority still decodes to the correct object.
              h_ret.emplace_back(mod, 0u);
            }
          }
        }
      }
#else
      for(auto vi=m.vert.begin();vi!=m.vert.end();++vi)
        if(!(*vi).IsD() && (*vi).IsRW())
        {
          vcg::face::VFIterator<FaceType> x;
          for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++ x){
            x.V1()->ClearV();
            x.V2()->ClearV();
          }
          for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++x )
          {
            if((x.V0()<x.V1()) && x.V1()->IsRW() && !x.V1()->IsV()){
              x.V1()->SetV();

              auto* mod = new MYTYPE(VertexPair(x.V0(), x.V1()), TriEdgeCollapseQuadric< TriMeshType, VertexPair, MYTYPE>::GlobalMark());
              // Priority deliberately left 0 here; every entry's is computed in ONE
              // parallel pass after the enumeration finishes (see the loop at the end
              // of Init). PackedHeapElem packs priority into the high 32 bits and the
              // pool offset into the low 32, and CodeToPtr casts to uint32_t -- so a
              // zero priority still decodes to the correct object.
              h_ret.emplace_back(mod, 0u);
              //fastPushHeap(h_ret);
            }
            if((x.V0()<x.V2()) && x.V2()->IsRW()&& !x.V2()->IsV()){
              x.V2()->SetV();
              auto* mod = new MYTYPE(VertexPair(x.V0(), x.V2()), TriEdgeCollapseQuadric< TriMeshType, VertexPair, MYTYPE>::GlobalMark());
              // Priority deliberately left 0 here; every entry's is computed in ONE
              // parallel pass after the enumeration finishes (see the loop at the end
              // of Init). PackedHeapElem packs priority into the high 32 bits and the
              // pool offset into the low 32, and CodeToPtr casts to uint32_t -- so a
              // zero priority still decodes to the correct object.
              h_ret.emplace_back(mod, 0u);
              //fastPushHeap(h_ret);
            }
          }
        }
#endif
    }
#if 0
    else
    { // if the collapse is A-symmetric (e.g. u->v != v->u)
      for(auto vi=m.vert.begin();vi!=m.vert.end();++vi)
        if(!(*vi).IsD() && (*vi).IsRW())
        {
          vcg::face::VFIterator<FaceType> x;
          UnMarkAll(m);
          for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++ x)
          {
            if(x.V()->IsRW() && x.V1()->IsRW() && !IsMarked(m,x.F()->V1(x.I()))){
              h_ret.push_back( HeapElem( new MYTYPE( VertexPair (x.V(),x.V1()),TriEdgeCollapse< TriMeshType,VertexPair,MYTYPE>::GlobalMark())));
            }
            if(x.V()->IsRW() && x.V2()->IsRW() && !IsMarked(m,x.F()->V2(x.I()))){
              h_ret.push_back( HeapElem( new MYTYPE( VertexPair (x.V(),x.V2()),TriEdgeCollapse< TriMeshType,VertexPair,MYTYPE>::GlobalMark())));
            }
          }
        }
    }
#endif

    // ---- Priorities, in parallel ------------------------------------------------
    //
    // The enumeration above must stay serial: it bump-allocates each MYTYPE from the
    // shared pool (g_qOffset++ / g_qBlocks.push_back are unsynchronised) and dedups
    // neighbours through a thread_local generation array. But ComputePriority is the
    // expensive part -- two Quadric<double> loads (80 B each) out of a ~533 MB
    // per-vertex array plus a 3x3 Cholesky -- and it is PURE: it reads the two
    // endpoint quadrics and writes only the object's own optimalPos.
    //
    // So the loop above records the object and leaves the priority 0, and every entry
    // is evaluated here in one parallel sweep. Entries are independent, each thread
    // touches a disjoint set, and no allocation happens.
    //
    // Bit layout (PackedHeapElem): priority in the HIGH 32 bits, pool offset in the
    // LOW 32. CodeToPtr does `static_cast<uint32_t>(code)`, so the placeholder decodes
    // to the right object, and patching only the high half leaves the offset intact.
    //
    // Order-independent, so the result is IDENTICAL to computing it inline -- unlike
    // the InitQuadric gather, this one changes no arithmetic at all. makeHeapUltraFast
    // runs after Init returns, so the heap is built from the finished codes.
    {
      const ptrdiff_t _n = (ptrdiff_t)h_ret.size();
      auto * const _h = h_ret.data();   // HeapType's element type, not spelled out here
#pragma omp parallel for schedule(static)
      for (ptrdiff_t _i = 0; _i < _n; ++_i)
      {
        MYTYPE * const mod = CodeToPtr<MYTYPE>(_h[_i].code);
        float pri = mod->ComputePriority();
        // (uint32_t&) on a non-const float is the same bit-reinterpretation the
        // enumeration used before this change, and that AddCollapseToHeap still uses.
        _h[_i].code = (((uint64_t)(uint32_t&)pri) << 32) | (_h[_i].code & 0xFFFFFFFFull);
      }
      // Exact, serial, and free -- one add instead of 20M racing increments.
      if (DECI_PROFILE) g_deciSetup.nPriority += (unsigned long long)_n;
    }

  }
//  static float HeapSimplexRatio(BaseParameterClass *_pp) {return IsSymmetric(_pp)?5.0f:9.0f;}
  static float HeapSimplexRatio(BaseParameterClass *_pp) {return IsSymmetric(_pp)?4.0f:8.0f;}
  static bool IsSymmetric(BaseParameterClass *_pp) {return ((QParameter *)_pp)->OptimalPlacement;}
  static bool IsVertexStable(BaseParameterClass *_pp) {return !((QParameter *)_pp)->OptimalPlacement;}

  /** Evaluate the priority (error) for an edge collapse
  *
  * It simulate the collapse and compute the quadric error 
  * generated by this collapse. This error is weighted with 
  * - aspect ratio of involved triangles
  * - normal variation
  */
  // ---- Recentred-quadric helpers (OPENMVS_MESH_QUADRIC_RECENTRED) ---------------
  //
  // Each vertex's quadric is stored relative to ITS OWN position (CLEAN::RQuadric).
  // To combine a pair they must share a frame, so Q0 is translated into Q1's -- a
  // displacement of exactly one edge length, which is why this stays conditioned.
  // Translation is EXACT: A' = A, b' = b + 2At, c' = Q(t).
  //
  // Layout matches vcg::math::Quadric: abc[0..5] = A (a11 a12 a13 a22 a23 a33),
  // abc[6..8] = b, abc[9] = c, and Apply doubles the off-diagonal A terms.
  __forceinline void RqSumInV1Frame(double abc[10]) const
  {
    const VertexType * __restrict v0 = this->pos.cV(0);
    const VertexType * __restrict v1 = this->pos.cV(1);
    const auto & q0 = QH::Rq(v0);
    const auto & q1 = QH::Rq(v1);
    const double tx = (double)v1->cP()[0] - (double)v0->cP()[0];
    const double ty = (double)v1->cP()[1] - (double)v0->cP()[1];
    const double tz = (double)v1->cP()[2] - (double)v0->cP()[2];
    const double A0 = (double)q0.a[0]*tx + (double)q0.a[1]*ty + (double)q0.a[2]*tz;
    const double A1 = (double)q0.a[1]*tx + (double)q0.a[3]*ty + (double)q0.a[4]*tz;
    const double A2 = (double)q0.a[2]*tx + (double)q0.a[4]*ty + (double)q0.a[5]*tz;
    for (int k = 0; k < 6; ++k) abc[k] = (double)q0.a[k] + (double)q1.a[k];
    abc[6] = ((double)q0.b[0] + 2.0*A0) + (double)q1.b[0];
    abc[7] = ((double)q0.b[1] + 2.0*A1) + (double)q1.b[1];
    abc[8] = ((double)q0.b[2] + 2.0*A2) + (double)q1.b[2];
    abc[9] = ( (tx*A0 + ty*A1 + tz*A2)
             + ((double)q0.b[0]*tx + (double)q0.b[1]*ty + (double)q0.b[2]*tz)
             + (double)q0.c ) + (double)q1.c;
  }

  // Q(y) for the summed quadric, y measured from V(1). Mirrors vcg Quadric::Apply.
  static __forceinline double RqApply(const double abc[10], double y0, double y1, double y2)
  {
    return y0*y0*abc[0] + 2.0*y0*y1*abc[1] + 2.0*y0*y2*abc[2] + y0*abc[6]
         + y1*y1*abc[3] + 2.0*y1*y2*abc[4] + y1*abc[7]
         + y2*y2*abc[5] + y2*abc[8] + abc[9];
  }

  // Same Cholesky as ComputePosition, but solving in V(1)'s frame: the result is an
  // OFFSET from V(1), so optimalPos = P(v1) + y. Degenerate fallback is the edge
  // midpoint, which in this frame is -t/2.
  __forceinline void RqComputePosition(const double abc[10])
  {
    const VertexType * __restrict v0 = this->pos.cV(0);
    const VertexType * __restrict v1 = this->pos.cV(1);
    const double b0 = abc[6] * -0.5, b1 = abc[7] * -0.5, b2 = abc[8] * -0.5;
    const double L00 = FastSqrtD(abc[0]);
    const double invL00 = 1.0 / L00;
    const double L10 = abc[1] * invL00;
    const double L20 = abc[2] * invL00;
    const double d11 = abc[3] - L10 * L10;
    const double L11 = FastSqrtD(d11);
    const double invL11 = 1.0 / L11;
    const double L21 = (abc[4] - L10 * L20) * invL11;
    const double d22 = abc[5] - L20 * L20 - L21 * L21;
    const double L22 = FastSqrtD(d22);
    const double invL22 = 1.0 / L22;
    const double z0 = b0 * invL00;
    const double z1 = (b1 - L10 * z0) * invL11;
    const double z2 = (b2 - L20 * z0 - L21 * z1) * invL22;
    const double x2 = z2 * invL22;
    const double x1 = (z1 - L21 * x2) * invL11;
    const double x0 = (z0 - L10 * x1 - L20 * x2) * invL00;
    if ((L00 >= 1e-12) && (d11 > 0.0) && (d22 > 0.0))
      this->optimalPos = CoordType((ScalarType)(v1->cP()[0] + x0),
                                   (ScalarType)(v1->cP()[1] + x1),
                                   (ScalarType)(v1->cP()[2] + x2));
    else
      this->optimalPos = (v0->cP() + v1->cP()) / (ScalarType)2.0;
  }

  ScalarType ComputePriority()
  {
    if (g_deciRecentred) {
      // One load of each 40-byte quadric, summed into V(1)'s frame, solved there.
      double abc[10];
      RqSumInV1Frame(abc);
      RqComputePosition(abc);
      const VertexType * __restrict rv1 = this->pos.cV(1);
      const double y0 = (double)this->optimalPos[0] - (double)rv1->cP()[0];
      const double y1 = (double)this->optimalPos[1] - (double)rv1->cP()[1];
      const double y2 = (double)this->optimalPos[2] - (double)rv1->cP()[2];
      extern double g_ScaleFactor;
      double qe = g_ScaleFactor * RqApply(abc, y0, y1, y2);
      if (qe <= 1e-15) {
        const VertexType * __restrict rv0 = this->pos.cV(0);
        const float dx = rv0->cP()[0] - rv1->cP()[0];
        const float dy = rv0->cP()[1] - rv1->cP()[1];
        const float dz = rv0->cP()[2] - rv1->cP()[2];
        qe = 1e-15 * FastSqrtS(dx*dx + dy*dy + dz*dz);
      }
      return (ScalarType)qe;
    }
    // NO DECI_COUNT here: this is called from the parallel priority sweep in Init,
    // and the counters are deliberately non-atomic. Counted at the two SERIAL sites
    // instead -- exactly, and for free.
#if 1

    VertexType* __restrict v0 = this->pos.V(0);
    VertexType* __restrict v1 = this->pos.V(1);

    CoordType oldPos0 = v0->P();
    CoordType oldPos1 = v1->P();

    // ComputePositionAbc, not ComputePosition: see the comment on it. The old call
    // discarded a full ten-term Apply() at v1 into an unread `optimalError`, then re-summed
    // the same two quadrics into `qq` to Apply() again at optimalPos. abc[] already holds
    // that sum, so only the second Apply is real work.
    double abc[10];
    ComputePositionAbc(abc);

    // Point3d::Construct was doing exactly this float -> double widening.
    const double p0 = (double)this->optimalPos[0];
    const double p1 = (double)this->optimalPos[1];
    const double p2 = (double)this->optimalPos[2];

    extern double g_ScaleFactor;
    // Term for term and in the same association order as vcg Quadric::Apply, which maps
    // a[0..5] -> array[0..5], b[0..2] -> array[6..8], c -> array[9].
    double quadErr = g_ScaleFactor * (
            p0*p0*abc[0] + 2*p0*p1*abc[1] + 2*p0*p2*abc[2] + p0*abc[6]
        +   p1*p1*abc[3] + 2*p1*p2*abc[4] + p1*abc[7]
        +   p2*p2*abc[5] + p2*abc[8] + abc[9]);

    if (quadErr <= 1e-15) {
      const float dx = oldPos0[0] - oldPos1[0];
      const float dy = oldPos0[1] - oldPos1[1];
      const float dz = oldPos0[2] - oldPos1[2];
      quadErr = 1e-15 * FastSqrtS(dx * dx + dy * dy + dz * dz);
    }

    return (ScalarType)quadErr;
#else
    // In P2P, the geometry has been run through both DPC and the first part of ReconstructMesh,
    // so we can expect the points of the mesh to be relatively uniform.  Therefore, checking for
    // long edges or other similar metrics to cull on isn't effective.
    VertexType* __restrict v[2];
    v[0] = this->pos.V(0);
    v[1] = this->pos.V(1);

#if 0 //original work


    ScalarType origQual = std::numeric_limits<double>::max();


    //// Move the two vertexes into new position (storing the old ones)
    CoordType OldPos0 = v[0]->P();
    CoordType OldPos1 = v[1]->P();
    ComputePosition();
    // Now Simulate the collapse 
    v[0]->P() = v[1]->P() = this->optimalPos;

    ScalarType newQual = std::numeric_limits<ScalarType>::max();  // 
      for (VFIterator x(v[0]); !x.End(); ++x)  // for all faces in v0
        if (x.V1() != v[1] && x.V2() != v[1])
          newQual = std::min(newQual, QualityFace(*x.F()));
      for (VFIterator x(v[1]); !x.End(); ++x)	 // for all faces in v1
        if (x.V1() != v[0] && x.V2() != v[0]) // skip faces with v0
          newQual = std::min(newQual, QualityFace(*x.F()));

    QuadricType qq = QH::Qd(v[0]);
    qq += QH::Qd(v[1]);

    double QuadErr = g_ScaleFactor * qq.Apply(Point3d::Construct(v[1]->P()));

    assert(!math::IsNAN(QuadErr));
    // All collapses involving triangles with quality larger than <QualityThr> have no penalty;
    if (newQual > 0.3) newQual = 0.3;


    QuadErr = std::max(QuadErr, 1e-15);
    if (QuadErr <= 1e-15)
    {
      QuadErr *= Distance(OldPos0, OldPos1);
    }


    ScalarType error;
    error = (ScalarType)(QuadErr / newQual);

    // Restore old position of v0 and v1
    v[0]->P() = OldPos0;
    v[1]->P() = OldPos1;

    return error;


#else

#if 1 // Provably better 33.630
    _mm_prefetch((char*)(v[0]->VFp()), _MM_HINT_T1);
    _mm_prefetch((char*)(&v[0]->VFi()), _MM_HINT_T1);
    _mm_prefetch((char*)(v[1]->VFp()), _MM_HINT_T1);
    _mm_prefetch((char*)(&v[1]->VFi()), _MM_HINT_T1);
#endif
       
    ScalarType origQual= std::numeric_limits<double>::max();

    //// Move the two vertexes into new position (storing the old ones)
    CoordType OldPos0 = v[0]->P();
    CoordType OldPos1 = v[1]->P();

    const double optimalError = ComputePosition();

    // Now Simulate the collapse 
    v[0]->P() = v[1]->P() = this->optimalPos;
    
    ScalarType newQual = 0.3f * 0.3f; // Squared now
    static int cntr = 1;
    for (int vi = 0; vi < 2; ++vi) {
      const VertexType* __restrict current = (vi == 0 ? v[0] : v[1]);
      const VertexType* __restrict other = (vi == 0 ? v[1] : v[0]);
      for (VFIterator x(const_cast<VertexType*>(current)); !x.End(); ++x) {
        FaceType* f = x.F();
        int z = x.I();

#if 1 // Provably better 33.630
        FaceType* nextFace = f->VFp(z);
        if (nextFace) {
          _mm_prefetch((char*)nextFace, _MM_HINT_T0);
          _mm_prefetch((char*)nextFace + 64, _MM_HINT_T0);
        }
#endif
        // Fast early-out: skip already seen faces
        if (f->IMark() != cntr) {
          f->IMark() = cntr;

          const uint8_t packed = nextPacked[z];
          const int aIndex = packed >> 4;
          const int bIndex = packed & 0xF;
          const VertexType* __restrict a = f->V(aIndex);
          const VertexType* __restrict b = f->V(bIndex);
          if (a != other && b != other) {
            // Notice we actually calculate QualityFace^2 and compensate below.
            newQual = FastMinS(QualityFace(*f), newQual);
#ifdef TAKE_FIRST_GOOD_COLLAPSE
            if (newQual < 0.09f) goto earlyExit;
#endif
          }
        }
      }
    }
    ++cntr;

#ifdef TAKE_FIRST_GOOD_COLLAPSE
earlyExit:
#endif

#if 0
#if 1
double QuadErr = g_ScaleFactor * optimalError;
if (QuadErr <= 1e-15)
QuadErr = 1e-15 * Distance(OldPos0, OldPos1);

// Restore original vertex positions
v[0]->P() = OldPos0;
v[1]->P() = OldPos1;

// Invert priority logic — better triangles get lower cost
return ScalarType(QuadErr * (1.0f - newQual));
#else
    constexpr double kMinError = 1e-15;
    constexpr ScalarType kMinQual = 1e-4f;
    constexpr ScalarType kMaxQual = 0.3f;

    // Clamp quality fast, no call to std::clamp
    const ScalarType qual = (newQual < kMinQual) ? kMinQual :
      (newQual > kMaxQual) ? kMaxQual : newQual;

    double scaledError = g_ScaleFactor * optimalError;

    // Only do distance calculation if really needed
    if (scaledError <= kMinError) {
      // This path is very rare — it’s OK to branch here
      const float dx = OldPos0[0] - OldPos1[0];
      const float dy = OldPos0[1] - OldPos1[1];
      const float dz = OldPos0[2] - OldPos1[2];
      scaledError = kMinError * std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    // Restore old position of v0 and v1
    v[0]->P() = OldPos0;
    v[1]->P() = OldPos1;

    return ScalarType(scaledError / qual);
#endif
#else

  //  newQual = FastSqrtS(newQual);

    QuadricType qq = QH::Qd(v[0]);
    qq += QH::Qd(v[1]);

    extern double g_ScaleFactor;
    double QuadErr = std::min(g_ScaleFactor * qq.Apply(Point3d::Construct(v[1]->P())), 0.3);

    assert(!math::IsNAN(QuadErr));
    // All collapses involving triangles with quality larger than <QualityThr> (0.3) have no penalty;
    
    if (QuadErr <= 1e-15) {
      QuadErr = 1e-15 * Distance(OldPos0, OldPos1);
    }

    // Restore old position of v0 and v1
    v[0]->P()=OldPos0;
    v[1]->P()=OldPos1;
    
    // Square again since this expression squared is monotonic and preserves ordering: return (ScalarType)(QuadErr / FastSqrtS(newQual));
    return (ScalarType)(QuadErr * QuadErr / newQual);
#endif
#endif
#endif
  }
  
  
  bool CheckForFlippedFaceOverVertex(VertexType *vp, ScalarType angleThrRad =  math::ToRad(150.))
  {
    std::map<VertexType *, CoordType>  edgeNormMap; 
    ScalarType maxAngle=0;
    
    for(VFIterator x(vp); !x.End(); ++x )	 // for all faces in v1
    {
      if(QualityFace(*x.F()) <0.01 ) return true; 
        for(int i=0;i<2;++i)
        { 
          VertexType *vv= i==0?x.V1():x.V2();
          assert(vv!=vp);
          auto ni = edgeNormMap.find(vv);
          if(ni==edgeNormMap.end()) edgeNormMap[vv] = NormalizedTriangleNormal(*x.F());
          else maxAngle = std::max(maxAngle,AngleN(NormalizedTriangleNormal(*x.F()),ni->second));
        }
    }
    
    return (maxAngle > angleThrRad);          
  }  
  
  // This function return true if, after an edge collapse, 
  // among the surviving faces, there are two adjacent faces forming a 
  // diedral angle larger than the given threshold
  // It assumes that the two vertexes of the collapsing edge 
  // have been already moved to the new position but the topolgy has not yet been changed (e.g. there are two zero-area faces)
  
#if 0 // JPB WIP BUG
  bool CheckForFlip(ScalarType angleThrRad =  math::ToRad(150.))
  {
    std::map<VertexType *, CoordType>  edgeNormMap; 
    VertexType * v[2];
    v[0] = this->pos.V(0);
    v[1] = this->pos.V(1);
    ScalarType maxAngle=0;
    assert (v[0]->P()==v[1]->P());
    
    for(VFIterator x(v[0]); !x.End(); ++x )	 // for all faces in v0
      if( x.V1()!=v[1] && x.V2()!=v[1] )     // skip faces with v1
      {
        if(QualityFace(*x.F()) <0.01 ) return true;         
        for(int i=0;i<2;++i)
        { 
          VertexType *vv= (i==0)?x.V1():x.V2();
          assert(vv!=v[0]);
          auto ni = edgeNormMap.find(vv);
          if(ni==edgeNormMap.end()) edgeNormMap[vv] = NormalizedTriangleNormal(*x.F());
          else maxAngle = std::max(maxAngle,AngleN(NormalizedTriangleNormal(*x.F()),ni->second));
        }
      }
    for(VFIterator x(v[1]); !x.End(); ++x )	 // for all faces in v1
      if( x.V1()!=v[0] && x.V2()!=v[0] )     // skip faces with v0
      {
        if(QualityFace(*x.F()) <0.01 ) return true;         
        for(int i=0;i<2;++i)
        { 
          VertexType *vv= i==0?x.V1():x.V2();
          assert(vv!=v[1]);
          auto ni = edgeNormMap.find(vv);
          if(ni==edgeNormMap.end()) edgeNormMap[vv] = NormalizedTriangleNormal(*x.F());
          else maxAngle = std::max(maxAngle,AngleN(NormalizedTriangleNormal(*x.F()),ni->second));
        }
    }
    return (maxAngle > angleThrRad);      
  }
#endif
 
#if 0
  template <class VertexType>
  bool HasSharedFace2(VertexType* v0, VertexType* v1) {
    vcg::face::VFIterator<typename VertexType::FaceType> it;
    for (it.F() = v0->VFp(), it.I() = v0->VFi(); it.F(); ++it) {
      auto* f = it.F();
      if (f->IsD()) continue;

      // check if v1 is one of the other two vertices in the face
      if (f->V(0) == v1 || f->V(1) == v1 || f->V(2) == v1)
        return true;
    }
    return false;
  }

  template <class VertexType>
  bool OnBoundary2(VertexType* v) {
    using FaceType = typename VertexType::FaceType;

    vcg::face::VFIterator<FaceType> it;
    for (it.F() = v->VFp(), it.I() = v->VFi(); it.F(); ++it) {
      auto* f = it.F();
      if (f->IsD()) continue;

      // Get index of vertex 'v' in the face
      int vi = -1;
      if (f->V(0) == v) vi = 0;
      else if (f->V(1) == v) vi = 1;
      else if (f->V(2) == v) vi = 2;
      if (vi == -1) continue; // not found (should not happen)

      if (f->IsB(vi))
        return true;
    }
    return false;
  }

  bool IsCollapseLegal2(VertexType* v0, VertexType* v1) {
    if (v0 == v1) return false;
    if (v0->IsD() || v1->IsD()) return false;
    if (!HasSharedFace2(v0, v1)) return false;
    if (OnBoundary2(v0) != OnBoundary2(v1)) return false;
    return true;
  }
#endif


  inline static float EstimateMergeCost(const QuadricType& q0, const QuadricType& q1) {
    float a = q0.array[0] + q1.array[0];
    float b = q0.array[1] + q1.array[1];
    float c = q0.array[2] + q1.array[2];
    return a * a + b * b + c * c; // example upper bound
  }

#if 0 // original
  inline void AddCollapseToHeap(HeapType& h_ret, VertexType* v0, VertexType* v1)
  {
    auto* mod = new MYTYPE(VertexPair(v0, v1), this->GlobalMark());
    const float priority = mod->ComputePriority();
    const uint32_t priBits = (uint32_t&)priority;
    h_ret.emplace_back(mod, priBits);
    std::push_heap(h_ret.begin(), h_ret.end());
  }

#else
inline void AddCollapseToHeap(void* ph, void* phBuffer, VertexType* v0, VertexType* v1)
{
    DECI_COUNT(nAddCollapse);
    DECI_COUNT(nPriority);   // this call site always goes on to ComputePriority
    auto& hBuffer = *static_cast<HeapType*>(phBuffer);

    // See tscAlloc/tscPrio/tscPush in DeciSetupTimes. This overload is LOOP-ONLY -- the
    // heap build uses the HeapType& overload and a parallel priority sweep, which is why
    // addCollapse setup= reads 0 -- so these counters need no thread safety.
    const bool _apOn = g_deciLoopProf;
    unsigned long long _apT = _apOn ? __rdtsc() : 0ull;

    // Try v0 v1 collapse
    auto* mod = new MYTYPE(VertexPair(v0, v1), this->GlobalMark());

    if (_apOn) { const unsigned long long t = __rdtsc(); g_deciSetup.tscAlloc += t - _apT; _apT = t; }

    const float priority = mod->ComputePriority();

    if (_apOn) { const unsigned long long t = __rdtsc(); g_deciSetup.tscPrio += t - _apT; _apT = t; }
#ifdef REJECT_BAD_CANDIDATES
    auto& h = *static_cast<HeapType*>(ph);
    // min of 10 rejects 5%, 5 rejects 10%
    if (!h.empty()) {
      uint32_t heapMin = h.front().code >> 32;
      float hMin = reinterpret_cast<float&>(heapMin);
      if (priority > hMin * 5.f) {
        return;
      }
    }
#endif
    // Add the collapse to the mini heap.
    // Remember, the first element is the true min of this heap.
    const uint32_t priBits = (uint32_t&)priority;
    hBuffer.emplace_back(mod, priBits);
    if (hBuffer.size() > 1) {
      if (hBuffer.back().code < hBuffer[0].code) {
        std::swap(hBuffer[0], hBuffer.back());
      }
    }

    if (_apOn) g_deciSetup.tscPush += __rdtsc() - _apT;
}
#endif
#if 0 // Original
inline  void UpdateHeap(HeapType& h_ret)
{
  this->GlobalMark()++;
  VertexType* v[2];
  v[0] = this->pos.V(0);
  v[1] = this->pos.V(1);
  v[1]->IMark() = this->GlobalMark();

  // First loop around the surviving vertex to unmark the Visit flags
  for (VFIterator vfi(v[1]); !vfi.End(); ++vfi) {
    vfi.V1()->ClearV();
    vfi.V2()->ClearV();
    vfi.V1()->IMark() = this->GlobalMark();
    vfi.V2()->IMark() = this->GlobalMark();
  }

  // Second Loop
  for (VFIterator vfi(v[1]); !vfi.End(); ++vfi) {
    if (!(vfi.V1()->IsV()) && vfi.V1()->IsRW())
    {
      vfi.V1()->SetV();
      AddCollapseToHeap(h_ret, vfi.V0(), vfi.V1());
    }
    if (!(vfi.V2()->IsV()) && vfi.V2()->IsRW())
    {
      vfi.V2()->SetV();
      AddCollapseToHeap(h_ret, vfi.V2(), vfi.V0());
    }
    if (vfi.V1()->IsRW() && vfi.V2()->IsRW())
      AddCollapseToHeap(h_ret, vfi.V1(), vfi.V2());
  } // end second loop around surviving vertex.
}

#else

  __forceinline void UpdateHeap(
    void* __restrict h,
    void* __restrict hBuffer,
    std::vector<void*>& __restrict pairsScratch,
    std::vector<void*>& __restrict toAddScratch
  )
  {
    const int mark = ++this->GlobalMark();

    VertexType* __restrict v1 = this->pos.V(1);

    v1->IMark() = mark;

    pairsScratch.clear();

    // First loop: clear visited flags and mark all incident vertices
    for (VFIterator vfi(v1); !vfi.End(); ++vfi) {
      FaceType& f = (*vfi.f);
      int z = vfi.I();
      const int current = z + 1 - 3 * (z == 2);     // (z + 1) % 3
      const int next = z + 2 - 3 * (z >= 1);     // (z + 2) % 3
      VertexType* __restrict a = f.V(current);
      VertexType* __restrict b = f.V(next);

      a->ClearV();
      b->ClearV();
      a->IMark() = mark;
      b->IMark() = mark;

      pairsScratch.push_back(a);
      pairsScratch.push_back(b);
    }

    // Second loop: test and add candidate collapses.
    //
    // This does NOT re-walk the VF ring, and must not. The ring is a linked list threaded
    // through faces, so a second traversal is a second pointer chase -- MEASURED as part of
    // walk=35.7% of updateHeap (1.596 s, 344 ns per collapse for two passes). The old code
    // drove this loop with a VFIterator purely to read vfi.V0() as the "anchor", but
    // VFIterator::V0() is f->V0(z) where z is the corner index of the vertex the iterator
    // is CENTRED on -- so it is v1 on every single iteration, invariant. a and b already
    // come out of pairsScratch, which pass 1 filled. Nothing else touched vfi.
    //
    // Bit-identical: same iteration count (one entry pair per ring face), same order, same
    // values. Purely the removal of a redundant traversal.
    toAddScratch.clear();

    const size_t nPairs = pairsScratch.size();
    for (size_t pi = 0; pi < nPairs; pi += 2) {
      VertexType* __restrict a = (VertexType*)pairsScratch[pi];
      VertexType* __restrict b = (VertexType*)pairsScratch[pi + 1];
      VertexType* __restrict c = v1; // anchor vertex: the surviving vertex, by construction

      bool aVisited = a->IsV();
      bool bVisited = b->IsV();

      if (!aVisited) { // Always rw && a->IsRW()) {
        a->SetV();
        toAddScratch.push_back(c);
        toAddScratch.push_back(a);
        DECI_COUNT(nAddSurvivor);   // c is the survivor: quadric changed
      }

      if (!bVisited) { // Always rw && b->IsRW()) {
        b->SetV();
        toAddScratch.push_back(b);
        toAddScratch.push_back(c);
        DECI_COUNT(nAddSurvivor);   // c is the survivor: quadric changed
      }

      // The opposite edge: neither endpoint is the survivor, so neither quadric nor
      // position changed and ComputePriority would return exactly the value already in
      // the heap. Under lazy revalidation the existing entry stays valid (IsUpToDate
      // checks Gm, not just IMark), so re-adding it is pure duplication -- ~50% of all
      // loop priority calls, plus the matching pool allocations and heap pushes.
      if (!g_deciLazy) {
        toAddScratch.push_back(a);
        toAddScratch.push_back(b);
        DECI_COUNT(nAddOpposite);
      }
    }

    int toAddCnt = (int) toAddScratch.size();
    VertexType* c0 = nullptr;
    VertexType* c1 = nullptr;
    VertexType* c2 = nullptr;
    VertexType* c3 = nullptr;

    auto pvToAdd = toAddScratch.begin();
    for (int i = 0; i < toAddCnt; i += 2) {
#if 1 // Provably better 33.630
      if (i + 3 < toAddCnt) {
        auto* nextV0 = (VertexType*) pvToAdd[i + 2];
        auto* nextV1 = (VertexType*) pvToAdd[i + 3];

        if (nextV0 != c0 && nextV0 != c1 && nextV0 != c2 && nextV0 != c3) {
          if (g_deciRecentred) {
            const auto& r0 = QH::Rq(nextV0);
            _mm_prefetch((const char*)&r0, _MM_HINT_T1);
            _mm_prefetch(((const char*)&r0) + 39, _MM_HINT_T1);
          } else {
          const QuadricType& q0 = QH::Qd(nextV0);
          _mm_prefetch((char*)q0.array, _MM_HINT_T1);
          _mm_prefetch(((char*)q0.array) + 64, _MM_HINT_T1);
          }
        }
        if (nextV1 != c0 && nextV1 != c1 && nextV1 != c2 && nextV1 != c3) {
          if (g_deciRecentred) {
            const auto& r1 = QH::Rq(nextV1);
            _mm_prefetch((const char*)&r1, _MM_HINT_T1);
            _mm_prefetch(((const char*)&r1) + 39, _MM_HINT_T1);
          } else {
          const QuadricType& q1 = QH::Qd(nextV1);
          _mm_prefetch((char*)q1.array, _MM_HINT_T1);
          _mm_prefetch(((char*)q1.array) + 64, _MM_HINT_T1);
          }
        }
      }
#endif

      VertexType* __restrict currV0 = (VertexType*) pvToAdd[i];
      VertexType* __restrict currV1 = (VertexType*) pvToAdd[i + 1];
      AddCollapseToHeap(h, hBuffer, currV0, currV1);

      // Notice, currV0 and currV1 on the first iteration
      // are not prefetched, but they are cached because
      // we have incurred the penalty of accessing them.
      c3 = c1;
      c2 = c0;
      c1 = currV0;
      c0 = currV1;
    }
  }
#endif

  static void InitQuadric(TriMeshType &m,BaseParameterClass *_pp)
  {
    QParameter *pp=(QParameter *)_pp;
    QH::Init();

    // Two implementations of the same accumulation, selected at runtime.
    //
    // SCATTER is upstream: loop faces, add each face quadric into its three vertices.
    // Serial, because neighbouring faces share vertices and the += collide.
    // GATHER walks each vertex VF ring into a local quadric instead -- disjoint
    // writes, threadable, ~3x the plane arithmetic spread over all cores. Measured
    // quadric=0.651 -> 0.109 s on RichmondHistoric (13.3M faces).
    //
    // Gather is the DEFAULT, VALIDATED 2026-09-13; set OPENMVS_MESH_QUADRIC_GATHER=0 to
    // revert. It could not be validated the obvious way: decimation stops on a face-count
    // FLOOR, so wrong quadrics still deliver ~4.06M faces -- just the wrong ones -- and the
    // border term touches only 0.1% of vertices on these scenes, so it is invisible in any
    // aggregate. OPENMVS_MESH_QUADRIC_CHECK=1 therefore runs BOTH in one process and
    // compares every coefficient of every vertex, keeping the SCATTER result so the check
    // never changes behaviour.
    //
    // RESULT (RichmondHistoric, 6,669,050 vertices compared):
    //   max rel diff 2.882e-12   max abs diff 2.910e-11
    // Not an algebraic error -- every way this could be wrong is O(1) (merging the two
    // border ifs halves a term, a dropped guard gives ~1). It IS ~1000x more than naive
    // reordering of 6-12 doubles predicts; most likely cancellation in that off-diagonal
    // cross term, or FMA contraction differing between the two inlining contexts under
    // /fp:precise. Either way it propagates to the collapse priority at the same ~3e-12,
    // so it can only reorder collapses whose priorities are already that close -- far less
    // perturbation than the Morton vertex reorder, which is default-on and shuffles
    // tie-breaking wholesale.
    const char *_eG = std::getenv("OPENMVS_MESH_QUADRIC_GATHER");
    const char *_eC = std::getenv("OPENMVS_MESH_QUADRIC_CHECK");
    // The scatter/gather check compares double world-frame quadrics; it cannot run on
    // the recentred path, which does not allocate them.
    const bool _check  = (_eC && std::atoi(_eC) > 0) && !g_deciRecentred;
    // Gather is now the DEFAULT; OPENMVS_MESH_QUADRIC_GATHER=0 reverts to the scatter.
    const bool _gather = _check || !(_eG && std::atoi(_eG) == 0);

    // ---- SCATTER (upstream, unchanged) ----------------------------------------
    auto _scatter = [&]() {
    for(VertexIterator pv=m.vert.begin();pv!=m.vert.end();++pv)
      if( ! (*pv).IsD() && (*pv).IsW())
        QH::Qd(*pv).SetZero();

    for(FaceIterator fi=m.face.begin();fi!=m.face.end();++fi)
      if( !(*fi).IsD() && (*fi).IsR() )
        if((*fi).V(0)->IsR() &&(*fi).V(1)->IsR() &&(*fi).V(2)->IsR())
        {
          Plane3<ScalarType,false> facePlane;
          facePlane.SetDirection( ( (*fi).V(1)->cP() - (*fi).V(0)->cP() ) ^  ( (*fi).V(2)->cP() - (*fi).V(0)->cP() ));
          if(!pp->UseArea)
            facePlane.Normalize();
          facePlane.SetOffset( facePlane.Direction().dot((*fi).V(0)->cP()));

          QuadricType q;
          q.ByPlane(facePlane);

          // The basic < add face quadric to each vertex > loop
          for(int j=0;j<3;++j)
            if( (*fi).V(j)->IsW() )
              QH::Qd((*fi).V(j)) += q;

          for(int j=0;j<3;++j)
            if( (*fi).IsB(j) || pp->QualityQuadric )
            {
              Plane3<ScalarType,false> borderPlane;
              QuadricType bq;
              // Border quadric record the squared distance from the plane orthogonal to the face and passing
              // through the edge.
              borderPlane.SetDirection(facePlane.Direction() ^ (( (*fi).V1(j)->cP() - (*fi).V(j)->cP() ).normalized()));
              if(  (*fi).IsB(j) ) borderPlane.SetDirection(borderPlane.Direction()* (ScalarType)(pp->BoundaryQuadricWeight ));        // amplify border planes
              else                borderPlane.SetDirection(borderPlane.Direction()* (ScalarType)(pp->QualityQuadricWeight ));   // and consider much less quadric for quality
              borderPlane.SetOffset(borderPlane.Direction().dot((*fi).V(j)->cP()));
              bq.ByPlane(borderPlane);

              if( (*fi).V (j)->IsW() )	QH::Qd((*fi).V (j)) += bq;
              if( (*fi).V1(j)->IsW() )	QH::Qd((*fi).V1(j)) += bq;
            }
        }
    };

    // ---- GATHER (threadable) --------------------------------------------------
    // Equivalence, term by term: the separate SetZero pass is folded in (acc starts
    // zero and is assigned, so non-writable/deleted vertices keep the QZero that the
    // QuadricTemp constructor wrote); the face quadric is added once per ring entry,
    // which is once per corner exactly as the scatter did; the border term keeps its
    // TWO INDEPENDENT ifs on V(j) and V1(j) -- merging them would halve the border
    // weight; every guard (IsD/IsR, the three IsR corners, IsB||QualityQuadric,
    // UseArea) is carried over unchanged.
    auto _gatherRun = [&]() {
      const int _nv = (int)m.vert.size();
#pragma omp parallel for schedule(dynamic, 4096)
      for (int _vi = 0; _vi < _nv; ++_vi)
      {
        VertexType &v = m.vert[_vi];
        if( v.IsD() || !v.IsW() ) continue;
        VertexType * const pv = &v;

        QuadricType acc;
        acc.SetZero();
        // Recentred accumulation, built directly in the vertex's own frame -- for plane
        // (n,d) and reference r, s = n.r - d gives A = nn^T, b' = 2ns, c' = s^2. The
        // large world-frame terms (b = -2dn, c = d^2) are never formed at all.
        double racc[10] = {0,0,0,0,0,0,0,0,0,0};
        const double rx = (double)v.cP()[0], ry = (double)v.cP()[1], rz = (double)v.cP()[2];

        vcg::face::VFIterator<FaceType> x;
        for( x.F() = v.VFp(), x.I() = v.VFi() ; x.F()!=0 ; ++x )
        {
          FaceType &f = *x.F();
          if( f.IsD() || !f.IsR() ) continue;
          if( !( f.V(0)->IsR() && f.V(1)->IsR() && f.V(2)->IsR() ) ) continue;

          Plane3<ScalarType,false> facePlane;
          facePlane.SetDirection( ( f.V(1)->cP() - f.V(0)->cP() ) ^ ( f.V(2)->cP() - f.V(0)->cP() ) );
          if(!pp->UseArea)
            facePlane.Normalize();
          facePlane.SetOffset( facePlane.Direction().dot( f.V(0)->cP() ) );

          if (g_deciRecentred) {
            const double n0 = (double)facePlane.Direction()[0];
            const double n1 = (double)facePlane.Direction()[1];
            const double n2 = (double)facePlane.Direction()[2];
            const double sv = n0*rx + n1*ry + n2*rz - (double)facePlane.Offset();
            racc[0]+=n0*n0; racc[1]+=n1*n0; racc[2]+=n2*n0;
            racc[3]+=n1*n1; racc[4]+=n2*n1; racc[5]+=n2*n2;
            racc[6]+=2.0*n0*sv; racc[7]+=2.0*n1*sv; racc[8]+=2.0*n2*sv; racc[9]+=sv*sv;
          } else {
          QuadricType q;
          q.ByPlane(facePlane);
          acc += q;   // v is a corner of f, and v.IsW() was checked above
          }

          for(int j=0;j<3;++j)
            if( f.IsB(j) || pp->QualityQuadric )
            {
              const bool hitsV  = ( f.V (j) == pv );
              const bool hitsV1 = ( f.V1(j) == pv );
              if( !hitsV && !hitsV1 ) continue;

              Plane3<ScalarType,false> borderPlane;
              QuadricType bq;
              borderPlane.SetDirection( facePlane.Direction() ^ ( ( f.V1(j)->cP() - f.V(j)->cP() ).normalized() ) );
              if( f.IsB(j) ) borderPlane.SetDirection(borderPlane.Direction()* (ScalarType)(pp->BoundaryQuadricWeight ));
              else           borderPlane.SetDirection(borderPlane.Direction()* (ScalarType)(pp->QualityQuadricWeight ));
              borderPlane.SetOffset( borderPlane.Direction().dot( f.V(j)->cP() ) );
              if (g_deciRecentred) {
                const double m0 = (double)borderPlane.Direction()[0];
                const double m1 = (double)borderPlane.Direction()[1];
                const double m2 = (double)borderPlane.Direction()[2];
                const double sb = m0*rx + m1*ry + m2*rz - (double)borderPlane.Offset();
                const int reps = (hitsV ? 1 : 0) + (hitsV1 ? 1 : 0);  // two independent adds
                for (int rep = 0; rep < reps; ++rep) {
                  racc[0]+=m0*m0; racc[1]+=m1*m0; racc[2]+=m2*m0;
                  racc[3]+=m1*m1; racc[4]+=m2*m1; racc[5]+=m2*m2;
                  racc[6]+=2.0*m0*sb; racc[7]+=2.0*m1*sb; racc[8]+=2.0*m2*sb; racc[9]+=sb*sb;
                }
              } else {
              bq.ByPlane(borderPlane);

              if( hitsV  ) acc += bq;   // two independent adds, exactly as the scatter had
              if( hitsV1 ) acc += bq;
              }
            }
        }
        if (g_deciRecentred) {
          auto & rq = QH::Rq(v);
          for (int k = 0; k < 6; ++k) rq.a[k] = (float)racc[k];
          rq.b[0] = (float)racc[6]; rq.b[1] = (float)racc[7]; rq.b[2] = (float)racc[8];
          rq.c = (float)racc[9];
        } else
        QH::Qd(v) = acc;
      }
    };

    if( !_gather )      _scatter();
    else if( !_check )  _gatherRun();
    else
    {
      // Run both, compare, keep the scatter answer.
      _scatter();
      const size_t _nv = m.vert.size();
      std::vector<QuadricType> _ref(_nv);
      for(size_t i=0;i<_nv;++i) _ref[i] = QH::Qd(m.vert[i]);

      _gatherRun();

      double _maxRel = 0.0, _maxAbs = 0.0;
      size_t _worst = 0; int _worstK = 0; size_t _nCmp = 0;
      for(size_t i=0;i<_nv;++i)
      {
        if( m.vert[i].IsD() || !m.vert[i].IsW() ) continue;
        ++_nCmp;
        const QuadricType &a = _ref[i];
        const QuadricType &b = QH::Qd(m.vert[i]);
        for(int k=0;k<10;++k)
        {
          const double d = std::fabs(a.array[k]-b.array[k]);
          const double s = std::fabs(a.array[k]);
          const double r = (s > 0.0) ? d/s : (d > 0.0 ? 1.0 : 0.0);
          if( d > _maxAbs ) _maxAbs = d;
          if( r > _maxRel ) { _maxRel = r; _worst = i; _worstK = k; }
        }
        QH::Qd(m.vert[i]) = _ref[i];   // keep the scatter result
      }
      MESH_DIAG("[QUADRIC-CHECK] %llu vertices compared | max rel diff %.3e (vertex %llu coeff %d)"
        " | max abs diff %.3e -- summation order alone should give ~1e-15;"
        " anything above ~1e-9 means the gather is NOT equivalent",
        (unsigned long long)_nCmp, _maxRel, (unsigned long long)_worst, _worstK, _maxAbs);
    }

    if(pp->ScaleIndependent)
    {
      vcg::tri::UpdateBounding<TriMeshType>::Box(m);
      //Make all quadric independent from mesh size

      const double diag = m.bbox.Diag();
      if (diag <= 0.0) {
        // pathological mesh, disable early reject safely
        g_ScaleFactor = 1.0;
      }
      else {
        g_ScaleFactor = 1e8 * std::pow(1.0 / diag, 6);
      }
    }
    if(pp->QualityWeight) // we map quality range into a squared 01 and than this into the 1..QualityWeightFactor range
    {
      ScalarType minQ, maxQ;
      tri::Stat<TriMeshType>::ComputePerVertexQualityMinMax(m,minQ,maxQ);      
      for(VertexIterator vi=m.vert.begin();vi!=m.vert.end();++vi)
        if( ! (*vi).IsD() && (*vi).IsW())
        {
          const double quality01squared = pow((double)((vi->Q()-minQ)/(maxQ-minQ)),2.0);
          QH::Qd(*vi) *= 1.0 + quality01squared * (pp->QualityWeightFactor-1.0); 
        }
    }
  }
  
  CoordType ComputeMinimal()
  {
  }
  
CoordType ComputeMinimalOld()
{
   VertexType* &v0 = this->pos.V(0);
   VertexType* &v1 = this->pos.V(1);
   QuadricType q=QH::Qd(v0);
   q+=QH::Qd(v1);
   
   Point3<QuadricType::ScalarType> x;
   
   bool rt=q.Minimum(x);
   if(!rt) { // if the computation of the minimum fails we choose between the two edge points and the middle one.
     Point3<QuadricType::ScalarType> x0=Point3d::Construct(v0->P());
     Point3<QuadricType::ScalarType> x1=Point3d::Construct(v1->P());
     x.Import((v1->P()+v1->P())/2);
     double qvx=q.Apply(x);
     double qv0=q.Apply(x0);
     double qv1=q.Apply(x1);
     if(qv0<qvx) x=x0;
     if(qv1<qvx && qv1<qv0) x=x1;
   }
   
   return CoordType::Construct(x);
 }
};
#pragma pack(pop)

} // namespace tri
} // namespace vcg
#endif
