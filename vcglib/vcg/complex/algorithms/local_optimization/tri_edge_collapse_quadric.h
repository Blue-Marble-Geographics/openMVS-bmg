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

// Normally, all collapses are generated to find a "best" one.
// Enabling this forces the logic to take the first one it find that
// fulfills a quality tolerance of "likely good enough".
// Definitely changes the topology of the result.
#undef TAKE_FIRST_GOOD_COLLAPSE

// Don't collpase candidates (add to the heap), if they look poor.
#undef REJECT_BAD_CANDIDATES // Not as helpful as it seems.

float g_ScaleFactor;
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
  

  inline bool IsFeasible(BaseParameterClass *_pp){
    QParameter *pp=(QParameter *)_pp;
    if(!pp->PreserveTopology) return true;
    
    bool res = ( EdgeCollapser<TriMeshType, VertexPair>::LinkConditions(this->pos) );
    if(!res) ++( TEC::FailStat::LinkConditionEdge() );
    return res;
  }
  
  double ComputePosition()
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

    ReturnScalarType abc[10];
    const double* __restrict pq0 = q0.array;
    const double* __restrict pq1 = q1.array;

    for (size_t i = 0; i < 10; ++i) {
      abc[i] = pq0[i] + pq1[i]; // Not store bound
    }

    const double v1x = v1->cP()[0];
    const double v1y = v1->cP()[1];
    const double v1z = v1->cP()[2];

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

    return ReturnScalarType(
      v1x * v1x * abc[0] + two * v1x * v1y * abc[1] + two * v1x * v1z * abc[2] + v1x * abc[6 + 0]
      + v1y * v1y * abc[3] + two * v1y * v1z * abc[4] + v1y * abc[6 + 1]
      + v1z * v1z * abc[5] + v1z * abc[6 + 2] + abc[9]
    );
#endif
  }

  void Execute(TriMeshType &m)
  {
    auto* __restrict v0 = this->pos.V(0);
    auto* __restrict v1 = this->pos.V(1);

    CoordType newPos = this->optimalPos;
    auto& q0 = QH::Qd(v1);
    q0 += QH::Qd(v0); // v0 is deleted and v1 take the new position
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
    vcg::tri::UpdateTopology<TriMeshType>::VertexFace(m);
    vcg::tri::UpdateFlags<TriMeshType>::FaceBorderFromVF(m);
    
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
    
    InitQuadric(m,pp);

    // Initialize the heap with all the possible collapses
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
              float priority = mod->ComputePriority();
              h_ret.emplace_back(mod, (uint32_t&)priority);
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
              float priority = mod->ComputePriority();
              h_ret.emplace_back(mod, (uint32_t&)priority);
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
              float priority = mod->ComputePriority();
              h_ret.emplace_back(mod, (uint32_t&)priority);
              //fastPushHeap(h_ret);
            }
            if((x.V0()<x.V2()) && x.V2()->IsRW()&& !x.V2()->IsV()){
              x.V2()->SetV();
              auto* mod = new MYTYPE(VertexPair(x.V0(), x.V2()), TriEdgeCollapseQuadric< TriMeshType, VertexPair, MYTYPE>::GlobalMark());
              float priority = mod->ComputePriority();
              h_ret.emplace_back(mod, (uint32_t&)priority);
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
  ScalarType ComputePriority()
  {
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
    
    ScalarType newQual = 0.3f;
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

    extern float g_ScaleFactor;
    double QuadErr = std::min(g_ScaleFactor * qq.Apply(Point3d::Construct(v[1]->P())), 0.3);

    assert(!math::IsNAN(QuadErr));
    // All collapses involving triangles with quality larger than <QualityThr> (0.3) have no penalty;
    
    if (QuadErr <= 1e-15) {
      QuadErr = 1e-15 * Distance(OldPos0, OldPos1);
    }

    // Restore old position of v0 and v1
    v[0]->P()=OldPos0;
    v[1]->P()=OldPos1;
    
    return (ScalarType)(QuadErr / newQual);
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
    auto& hBuffer = *static_cast<HeapType*>(phBuffer);

    // Try v0 v1 collapse
    auto* mod = new MYTYPE(VertexPair(v0, v1), this->GlobalMark());

    const float priority = mod->ComputePriority();
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

  __forceinline void UpdateHeap(void*h, void* hBuffer, std::vector<void*>& pairsScratch, std::vector<void*>& toAddScratch)
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

    // Second loop: test and add candidate collapses
    auto itPairs = pairsScratch.begin();
    
    toAddScratch.clear();

    int toAddCnt = 0;

    for (VFIterator vfi(v1); !vfi.End(); ++vfi) {
      VertexType* __restrict a = (VertexType*) *itPairs++;
      VertexType* __restrict b = (VertexType*)*itPairs++;
      VertexType* __restrict c = vfi.V0(); // anchor vertex

      if (!a->IsV()) { // Always rw && a->IsRW()) {
        a->SetV();
        toAddScratch.push_back(c);
        toAddScratch.push_back(a);
      }

      if (!b->IsV()) { // Always rw && b->IsRW()) {
        b->SetV();
        toAddScratch.push_back(b);
        toAddScratch.push_back(c);
      }

      // Always rw if (a->IsRW() && b->IsRW()) {
      toAddScratch.push_back(a);
      toAddScratch.push_back(b);
    }

    VertexType* cache[4] = {}; // track last 4 seen vertices
    auto notInCache = [&](VertexType* v) {
      return v != cache[0] && v != cache[1] && v != cache[2] && v != cache[3];
      };

    auto pvToAdd = toAddScratch.begin();
    for (int i = 0, cnt = (int) toAddScratch.size(); i < cnt; i += 2) {
#if 1 // Provably better 33.630
      if (i + 2 < toAddCnt) {
        auto* nextV0 = (VertexType*) pvToAdd[i + 2];
        auto* nextV1 = (VertexType*) pvToAdd[i + 3];

        if (notInCache(nextV0)) {
          const QuadricType& q0 = QH::Qd(nextV0);
          _mm_prefetch((char*)q0.array, _MM_HINT_T1);
          _mm_prefetch(((char*)q0.array) + 64, _MM_HINT_T1);
        }
        if (notInCache(nextV1)) {
          const QuadricType& q1 = QH::Qd(nextV1);
          _mm_prefetch((char*)q1.array, _MM_HINT_T1);
          _mm_prefetch(((char*)q1.array) + 64, _MM_HINT_T1);
        }
      }
#endif

      VertexType* __restrict currV0 = (VertexType*) pvToAdd[i];
      VertexType* __restrict currV1 = (VertexType*) pvToAdd[i + 1];
      AddCollapseToHeap(h, hBuffer, currV0, currV1);

      // Notice, currV0 and currV1 on the first iteration
      // are not prefetched, but they are cached because
      // we have incurred the penalty of accessing them.
      cache[3] = cache[1];
      cache[2] = cache[0];
      cache[1] = currV0;
      cache[0] = currV1;
    }
  }
#endif

  static void InitQuadric(TriMeshType &m,BaseParameterClass *_pp)
  {
    QParameter *pp=(QParameter *)_pp;
    QH::Init();
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
    
    if(pp->ScaleIndependent)
    {
      vcg::tri::UpdateBounding<TriMeshType>::Box(m);
      //Make all quadric independent from mesh size
      g_ScaleFactor = 1e8*pow(1.0/m.bbox.Diag(),6); // scaling factor
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
