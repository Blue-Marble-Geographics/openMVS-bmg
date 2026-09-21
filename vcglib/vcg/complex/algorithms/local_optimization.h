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

#ifndef __VCGLIB_LOCALOPTIMIZATION
#define __VCGLIB_LOCALOPTIMIZATION

#include <vcg/complex/complex.h>
#include <time.h>
#include <chrono>
#include <algorithm>
#include <intrin.h>   // __rdtsc -- [LOOP-PROFILE]
using Clock = std::chrono::steady_clock;

// Gate for the decimation profiling instrumentation.
//
// Mirrors MESH_DIAG_ENABLED() from libs/MVS/Common.h, which is a COMPILE-TIME constant
// keyed off -DOPENMVS_MESH_DIAG -- and whose own comment states the gate is deliberately
// meant to skip the COMPUTATION, not just the printing. That applies here: the counters
// increment ~86M times per run (once per ComputePriority, once per AddCollapseToHeap)
// and feed nothing but a log line.
//
// Defers to MESH_DIAG_ENABLED() itself where it is visible, so this cannot drift from
// the real gate -- that also folds in TD_VERBOSE, which the raw flag does not. Mesh.cpp
// includes MVS/Common.h before the vcg headers, so that is the path taken in practice.
// The fallbacks cover this header being reached without MVS/Common.h; an undefined
// OPENMVS_MESH_DIAG means off, matching Common.h.
#if defined(MESH_DIAG_ENABLED)
#define DECI_PROFILE MESH_DIAG_ENABLED()
#elif defined(OPENMVS_MESH_DIAG) && OPENMVS_MESH_DIAG
#define DECI_PROFILE 1
#else
#define DECI_PROFILE 0
#endif

// Compiles to nothing when the gate is off.
#define DECI_COUNT(field) do { if (DECI_PROFILE) ++g_deciSetup.field; } while(0)

// Per-phase wall clock for decimation SETUP, which is entirely single-threaded
// while the DoOptimization loop that follows it is not (see HeapThreadPool below).
// Filled by TriEdgeCollapseQuadric::Init and LocalOptimization::Init, reported by
// the caller as [DECI-PROFILE]. Defined in libs/MVS/Mesh.cpp alongside g_qBlocks.
// Not thread-safe and does not need to be -- everything it measures is serial.
struct DeciSetupTimes
{
  double topo;       // VertexFace + FaceBorderFromVF inside Init
  double boundary;   // the Fast/PreserveBoundary face scans
  double quadric;    // InitQuadric: per-face quadric accumulation
  double heapBuild;  // candidate enumeration + ComputePriority + emplace_back
  double heapify;    // makeHeapUltraFast

  // Call counts for the two functions the timing implicates. The [DECI-PROFILE]
  // arithmetic says the collapse loop is ~12 candidates' worth of work per collapse
  // and therefore almost entirely AddCollapseToHeap -> ComputePriority; these confirm
  // or refute that directly instead of by inference.
  //
  // Plain uint64, NOT atomic. That is only safe because every DECI_COUNT site is on a
  // SERIAL path -- which stopped being true once Init's priority sweep was threaded, and
  // the counter promptly reported 2,520,325 of 19,996,738 (12.6%, 32 threads racing one
  // ++). ComputePriority therefore no longer counts itself; the two serial sites
  // (AddCollapseToHeap, and one exact add for the sweep's h_ret.size()) do. If you add a
  // DECI_COUNT, check the caller is serial or the number silently becomes fiction.
  unsigned long long nPriority;      // ComputePriority() calls
  unsigned long long nAddCollapse;   // AddCollapseToHeap() calls

  // Split of nAddCollapse by call site inside UpdateHeap, testing whether roughly half
  // the loop's priority work is recomputing values that cannot have changed:
  //   nAddSurvivor -- one endpoint is the SURVIVING vertex, whose quadric was just
  //                   replaced by Q0+Q1. The priority genuinely changed; must recompute.
  //   nAddOpposite -- the (a,b) pair opposite the survivor, pushed unconditionally once
  //                   per incident face. Neither endpoint's quadric or position changed,
  //                   so ComputePriority necessarily returns what the heap already holds.
  unsigned long long nAddSurvivor;
  unsigned long long nAddOpposite;

  // Heap pops and how many were stale. The class's own heapPopCount/stalePopCount are
  // RESET whenever compaction fires, so they cannot answer 'how much of the heap was
  // garbage over the whole run'; these accumulate.
  unsigned long long nHeapPop;
  unsigned long long nHeapStale;

  // [DECI-AUDIT] -- validates the two invariants the greedy loop rests on, at every
  // collapse actually executed. Opt-in (OPENMVS_MESH_DECI_AUDIT=1) because it costs one
  // extra ComputePriority plus a VF ring walk per pop.
  //
  //   nAuditNonEdge   collapses whose two endpoints no longer share a face. Today this
  //                   MUST be 0: legality is guaranteed solely by the IMark bump-and-
  //                   re-add cycle, since IsFeasible is commented out at the pop site
  //                   AND returns true unconditionally when PreserveTopology is false,
  //                   which is what runDecimate sets. Nothing else checks it.
  //   nAuditPriStale  collapses whose stored heap priority no longer matches a fresh
  //                   ComputePriority beyond 1e-6 relative. Today this MUST be 0 too:
  //                   every entry is re-added with a fresh priority after any touch.
  //
  // Both are the invariants a geometry-mark split would trade away, so measure them on
  // the CURRENT code first -- a validator that has never read zero proves nothing.
  unsigned long long nAudited;
  unsigned long long nAuditNonEdge;
  unsigned long long nAuditPriStale;
  double             auditMaxPriRel;

  // [FP32-PROBE] -- would storing the quadrics as float instead of double work?
  //
  // The motive is cache: Quadric<double> is 10 doubles = 80 B/vertex, so the array is
  // 533 MB on RichmondHistoric and 779 MB on Marco, and the decimate loop's cost is
  // dominated by two random loads out of it per ComputePriority. Halving it is the
  // obvious lever.
  //
  // The risk is cancellation. ByPlane stores a2..c2 at O(1), ad..cd at O(d), and d2 at
  // O(d^2), where d = n.p -- so on a 1224-unit scene d^2 ~ 1e6, and Apply() sums terms
  // of that size down to a squared distance of ~1e-2. Roughly 8 digits vanish. double
  // has 16 and survives; float has 7 and may not.
  //
  // This measures it instead of arguing about it: at each collapse, round BOTH endpoint
  // quadrics to float precision, re-run the real ComputePriority, and compare with the
  // double answer. No storage change, no risk -- it exercises the actual code path.
  // Also records the coefficient dynamic range that drives the cancellation.
  unsigned long long nFp32Probed;
  unsigned long long nFp32Bad;        // relative error > 1e-3: ordering would move
  double             fp32MaxRel;
  double             fp32MaxAbsA9;    // largest |d^2| term seen
  double             fp32MaxAbsA0;    // largest |a^2| term seen

  // Same probe, but with each quadric RECENTRED on its own vertex first. Translating a
  // quadric is exact: for y = x - r,  A' = A,  b' = b + 2Ar,  c' = Q(r). Since r sits on
  // the surface, b' becomes O(displacement) and c' O(displacement^2) instead of O(d) and
  // O(d^2) -- which is the whole problem, because d = n.p is the distance from the
  // WORLD ORIGIN to the plane and this scene sits ~3000 units off it.
  unsigned long long nRcBad;       // recentred float error > 1e-3
  double             rcMaxRel;
  double             rcMaxAbsC;    // largest |c'| after recentring (was max|d^2|)

  // [LOOP-PROFILE] -- where the collapse loop's ~11.8 s actually goes.
  //
  // Four hypotheses about this loop have now died (OptimalPlacement, quadric locality,
  // float storage, sift-down prefetch), each costing a build. ns/prio is loop_time
  // divided by priority calls, which attributes EVERYTHING to ComputePriority by
  // construction and so cannot distinguish them. These split the iteration for real.
  //
  // __rdtsc, not chrono: ~4 ns versus ~25, against 9.2M iterations x 4 reads. Ticks are
  // reported as percentages, so the TSC frequency never has to be known. Opt-in
  // (OPENMVS_MESH_DECI_LOOPPROF=1) and NOT to be combined with the audit or fp32 probe --
  // both call ComputePriority on the pop path and would land inside tscExec.
  unsigned long long tscPop;     // selection + popHeapUltraFast (the sift-down)
  unsigned long long tscStale;   // pops rejected by IsUpToDate
  unsigned long long tscExec;    // Execute: EdgeCollapser::Do
  unsigned long long tscUpd;     // UpdateHeap: VF walks + AddCollapseToHeap + pushes
  unsigned long long tscOther;   // compaction, hBuffer bookkeeping, GoalReached

  // updateHeap is 35.7% of the loop and is 31.9M AddCollapseToHeap calls at ~131 ns each
  // (MEASURED: loop priority calls 31,878,462 vs 4,641,654 collapses = 6.87 per collapse,
  // the survivor's valence). That 131 ns is THREE things and they want different fixes:
  //   alloc -> the block pool is too slow / touching cold pages
  //   prio  -> the quadric sum + Cholesky is the work; only fewer CALLS can help, and
  //            ~70% of pushed entries are never popped (9.3M pops vs 31.9M pushes)
  //   push  -> the hBuffer emplace + mini-heap compare
  // Splitting them is the only way to choose. NOTE this adds 4 rdtsc reads to a call that
  // costs ~131 ns, so it inflates updateHeap's share of the loop by roughly a tenth; read
  // the RATIO between the three, not their absolute share.
  unsigned long long tscAlloc;   // new MYTYPE (block pool)
  unsigned long long tscPrio;    // ComputePriority
  unsigned long long tscPush;    // hBuffer emplace_back + mini-heap swap

  void Clear( void ){ topo = boundary = quadric = heapBuild = heapify = 0.0; nPriority = nAddCollapse = 0; nAddSurvivor = nAddOpposite = 0; nHeapPop = nHeapStale = 0;
                     nAudited = nAuditNonEdge = nAuditPriStale = 0; auditMaxPriRel = 0.0;
                     nFp32Probed = nFp32Bad = 0; fp32MaxRel = fp32MaxAbsA9 = fp32MaxAbsA0 = 0.0;
                     nRcBad = 0; rcMaxRel = rcMaxAbsC = 0.0;
                     tscPop = tscStale = tscExec = tscUpd = tscOther = 0;
                     tscAlloc = tscPrio = tscPush = 0;
                     }
};
extern DeciSetupTimes g_deciSetup;
extern bool g_deciAudit;   // OPENMVS_MESH_DECI_AUDIT=1; see DECI-AUDIT above
extern bool g_deciLazy;    // OPENMVS_MESH_DECI_LAZY=1; see TriEdgeCollapseQuadric
extern bool g_deciFp32Probe;   // OPENMVS_MESH_QUADRIC_FP32_PROBE=1; see FP32-PROBE
extern bool g_deciRecentred;   // OPENMVS_MESH_QUADRIC_RECENTRED=1; see CLEAN::RQuadric
extern bool g_deciLoopProf;    // OPENMVS_MESH_DECI_LOOPPROF=1; see LOOP-PROFILE

// [EXEC-PROFILE] -- Execute is 27.3% of the loop (~3.3 s) and, across a whole session spent
// on this loop, was never once examined. Execute is EdgeCollapser::Do: FindSets walks v0's
// VF ring; the av01 loop VFDetaches each shared face from its two non-v0 corners and deletes
// it; the av0 loop re-points v0's remaining faces at v1 and prepends them to v1's VF list.
//
// These are EXACT COUNTS, not timings, deliberately: the rdtsc sub-splits above are
// documented as untrustworthy below ~100 ns and Execute's parts are exactly that size,
// whereas counts cannot be reordered by the CPU. The one that matters is g_vfDetachSteps --
// VFDetach is a singly-linked-list unlink that, whenever the face is not already at the
// head, SCANS the vertex's VF list for the predecessor. Size it (steps x a few ns per
// dependent pointer chase) BEFORE writing any fix.
//
// Declared in vcg/simplex/face/topology.h at global scope, because edge_collapse.h and
// topology.h both need them and sit below this header. Defined in libs/MVS/Mesh.cpp.
extern unsigned long long g_execCollapses;    // calls to EdgeCollapser::Do
extern unsigned long long g_execRingV0;       // VF ring length of v0 (FindSets iterations)
extern unsigned long long g_execAv01;         // faces incident on both v0 and v1
extern unsigned long long g_execAv0;          // faces incident on v0 only (relinked to v1)
extern unsigned long long g_vfDetachCalls;    // VFDetach calls
extern unsigned long long g_vfDetachHead;     // ... that hit the O(1) head case
extern unsigned long long g_vfDetachSteps;    // ... total list nodes walked on the scan path

// Scoped accumulator: adds its lifetime into the named slot.
#if DECI_PROFILE
struct DeciPhaseTimer
{
  explicit DeciPhaseTimer( double &sink ) : _sink(sink) , _t0(Clock::now()) {}
  ~DeciPhaseTimer( void ){ _sink += std::chrono::duration<double>( Clock::now() - _t0 ).count(); }
  DeciPhaseTimer( const DeciPhaseTimer & ) = delete;
  DeciPhaseTimer &operator=( const DeciPhaseTimer & ) = delete;
private:
  double &_sink;
  Clock::time_point _t0;
};
#else
// Same name and construction so every call site stays unchanged; optimises away whole.
struct DeciPhaseTimer { explicit DeciPhaseTimer( double & ){} };
#endif

//#pragma optimization("", off) // JPB WIP BUG

// This is (((uint64_t)((uint32_t&)std::numeric_limits<float>::max())) << 32);
constexpr uint64_t kMaxCode = uint64_t(0x7F7FFFFF) << 32;

template<typename T>
T* CodeToPtr(uint64_t code)
{
  extern std::vector<void*> g_qBlocks;

  constexpr uint64_t kBlockShift = 23;        // block index is in bits 23..31
  constexpr uint64_t kIndexMask = (1ULL << kBlockShift) - 1; // lower 23 bits
  constexpr uint64_t kItemSize = (36);          // size of each item in bytes

  // Strip high 32 bits
  const uint32_t code32 = static_cast<uint32_t>(code);  // safe, only using low bits

  const uint32_t blockIndex = code32 >> kBlockShift;
  const uint32_t indexInBlock = code32 & static_cast<uint32_t>(kIndexMask);

  uint8_t* base = static_cast<uint8_t*>(g_qBlocks[blockIndex]);
  return reinterpret_cast<T*>(base + indexInBlock * kItemSize);
}

// Encode a pooled MYTYPE* as (block index, index within block).
//
// This USED to assume the pointer lay in g_qBlocks.back(), the most recently allocated
// block. That holds whenever PtrToOffset is called straight after `new MYTYPE(...)`,
// which was the only caller -- but it is not a property of the pointer, it is a property
// of that one call site, and nothing said so.
//
// OPENMVS_MESH_DECI_FULLLAZY re-inserts an EXISTING entry, allocated long before and
// therefore usually in an earlier block. The old code then computed
// ptr - lastBlockBase, which is negative, wrapped it through uintptr_t, and handed
// CodeToPtr an index far outside the block -- reconstructing a wild pointer that faults
// on the next pop. RichmondHistoric's 20M Init entries span 3 blocks at
// BLOCK_SIZE = 8M objects, so two thirds of all re-keys were corrupt; the run died about
// two thirds of the way in, once one of them surfaced.
//
// Searching from the back keeps the freshly-allocated case at one iteration, and the pool
// is only a handful of blocks (8M objects each), so the miss case is a few compares.
template<typename T>
uint64_t PtrToOffset(T* ptr)
{
  extern std::vector<void*> g_qBlocks;

  constexpr uint64_t kBlockShift = 23;                      // must match CodeToPtr
  constexpr size_t   kItemSize = 36;                        // sizeof(MYTYPE), hardcoded there too
  // Mesh.cpp: BLOCK_SIZE = 8 * 1024 * 1024 objects == 1 << kBlockShift. The index field
  // is exactly kBlockShift bits wide, so a block may hold no more than this.
  constexpr size_t   kBlockBytes = (size_t(1) << kBlockShift) * kItemSize;

  const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  for (size_t bi = g_qBlocks.size(); bi-- > 0; ) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(g_qBlocks[bi]);
    if (p < base)
      continue;
    const uintptr_t off = p - base;
    if (off < kBlockBytes)
      return (uint64_t(bi) << kBlockShift) | uint64_t(off / kItemSize);
  }
  assert(false && "PtrToOffset: pointer is not inside any pool block");
  return 0;
}

namespace vcg{
// Base class for Parameters
// all parameters must be derived from this.
class BaseParameterClass { };

template<class MeshType>
class LocalOptimization;

enum ModifierType{	TetraEdgeCollapseOp, TriEdgeSwapOp, TriVertexSplitOp,
				TriEdgeCollapseOp,TetraEdgeSpliOpt,TetraEdgeSwapOp, TriEdgeFlipOp,
				QuadDiagCollapseOp, QuadEdgeCollapseOp};
/** \addtogroup tetramesh */
/*@{*/
/// This abstract class define which functions a local modification class must have to be used in the LocalOptimization framework.
template <class MeshType>
class LocalModification
{
 public:
   typedef typename LocalOptimization<MeshType>::HeapType HeapType;
   typedef typename MeshType::ScalarType ScalarType;

  inline LocalModification(){}
  ~LocalModification(){}
  
	/// return the type of operation
	ModifierType IsOfType() ;

	/// return true if the data have not changed since it was created
  bool IsUpToDate() const;

	/// return true if no constraint disallow this operation to be performed (ex: change of topology in edge collapses)
  bool IsFeasible(BaseParameterClass *pp);

	/// Compute the priority to be used in the heap
  ScalarType ComputePriority();

	/// Return the priority to be used in the heap (implement static priority)
	ScalarType Priority() const;

  /// Perform the operation
  void Execute(MeshType &m);

	/// perform initialization
  static void Init(MeshType &m, HeapType&, BaseParameterClass *pp);

	/// An approximation of the size of the heap with respect of the number of simplex
    /// of the mesh. When this number is exceeded a clear heap purging is performed. 
    /// so it is should be reasonably larger than the minimum expected size to avoid too frequent clear heap
    /// For example for symmetric edge collapse a 5 is a good guess. 
    /// while for non symmetric edge collapse a larger number like 9 is a better choice
  static float HeapSimplexRatio(BaseParameterClass *) {return 6.0f;}

  const char *Info(MeshType &) {return 0;}
	/// Update the heap as a consequence of this operation
  void UpdateHeap(void*);
};	//end class local modification

/// LocalOptimization:
/// This class implements the algorihms running on 0-1-2-3-simplicial complex that are based on local modification
/// The local modification can be and edge_collpase, or an edge_swap, a vertex plit...as far as they implement
/// the interface defined in LocalModification.
/// Implementation note: in order to keep the local modification itself indepented by its use in this class, they are not
/// really derived by LocalModification. Instead, a wrapper is done to this purpose (see vcg/complex/tetramesh/decimation/collapse.h)

template<class MeshType>
class LocalOptimization
{
public:
  LocalOptimization(MeshType& mm, BaseParameterClass* _pp) : m(mm)
  {
    ClearTermination();
    HeapSimplexRatio = 5;
    pp = _pp;
    heapPopCount = stalePopCount = 1;
    cntr = 0;
  }

  struct PackedHeapElem;
  typedef typename MeshType::ScalarType ScalarType;
  typedef typename std::vector<PackedHeapElem> HeapType;
  typedef LocalModification <MeshType>  LocModType;

#if 0
  using Leaf = vcg::tri::TriEdgeCollapse<
    CLEAN::Mesh,
    vcg::tri::BasicVertexPair<CLEAN::Vertex>,
    CLEAN::TriEdgeCollapse
    >;
#endif
	/// termination conditions	
	 enum LOTermination {	
      LOnSimplices	= 0x01,	// test number of simplicies	
			LOnVertices		= 0x02, // test number of verticies
			LOnOps			= 0x04, // test number of operations
			LOMetric		= 0x08, // test Metric (error, quality...instance dependent)
			LOTime			= 0x10  // test how much time is passed since the start
		} ;

	int tf; // Termination Flag
	
  int nPerformedOps,
		nTargetOps,
		nTargetSimplices,
		nTargetVertices;

	float	timeBudget;
  Clock::time_point	startTime;
	ScalarType currMetric;
	ScalarType targetMetric;
  BaseParameterClass *pp;

  // The ratio between Heap size and the number of simplices in the current mesh
  // When this value is exceeded a ClearHeap Start;

  float HeapSimplexRatio;
  int64_t heapPopCount;
  int64_t stalePopCount;
  int64_t cntr;

	void SetTerminationFlag		(int v){tf |= v;}
	void ClearTerminationFlag	(int v){tf &= ~v;}
	bool IsTerminationFlag		(int v){return ((tf & v)!=0);}

	void SetTargetSimplices	(int ts			){nTargetSimplices	= ts;	SetTerminationFlag(LOnSimplices);	}	 	
	void SetTargetVertices	(int tv			){nTargetVertices	= tv;	SetTerminationFlag(LOnVertices);	} 
	void SetTargetOperations(int to			){nTargetOps		= to;	SetTerminationFlag(LOnOps);			} 

	void SetTargetMetric	(ScalarType tm	){targetMetric		= tm;	SetTerminationFlag(LOMetric);		} 
	void SetTimeBudget		(float tb		){timeBudget		= tb;	SetTerminationFlag(LOTime);			} 

  void ClearTermination()
  {
    tf=0;
    nTargetSimplices=0;
    nTargetOps=0;
    targetMetric=0;
    currMetric=0;
    timeBudget=0;
    nTargetVertices=0;
  }
	/// the mesh to optimize
	MeshType & m;

	///the heap of operations
	HeapType h;
  HeapType hBuffer;
  ///the element of the heap
  // it is just a wrapper of the pointer to the localMod. 
  // std heap does not work for
  // pointers and we want pointers to have heterogenous heaps. 

  struct PackedHeapElem
  {
    ///pointer to instance of local modifier
    uint64_t code; // priority stored as a float in upper 32, locModPtr is an offset in the lower 32.
    // Offset 0 is essentially a nullptr

    PackedHeapElem() {}

    __forceinline PackedHeapElem(LocModType* p, uint32_t priority) :
      code((((uint64_t)priority) << 32) + PtrToOffset(p))
   	{}

    __forceinline explicit PackedHeapElem(uint64_t _code) :
      code(_code)
    {}

    /// STL heap has the largest element as the first one.
    /// usually we mean priority as an error so we should invert the comparison
    inline bool operator <(const PackedHeapElem& h) const
    { 
		  return (code > h.code);
		  //return (locModPtr->Priority() < h.locModPtr->Priority());
	  }
  };

  inline void makeHeapUltraFast(HeapType& heap)
  {
    const size_t n = heap.size();
    if (n < 2) return;

    auto* __restrict data = heap.data();

    for (size_t i = (n - 1) / 2 + 1; i-- > 0;) {
      PackedHeapElem tmp = data[i];
      auto tmpPri = tmp.code;

      size_t hole = i;

      while (true) {
        size_t left = 2 * hole + 1;
        if (left >= n) break;

        // Manually compute min child without branching
        const size_t right = left + 1;
        const size_t child = right < n && data[right].code < data[left].code ? right : left;

        const auto childPri = data[child].code;

        // Branchless break condition (if tmpPri <= childPri)
        if (!(tmpPri > childPri)) break;

        data[hole] = data[child]; // move child up
        hole = child;
      }

      data[hole] = tmp; // single final write
    }
  }

  template<typename Heap>
  __forceinline void siftDown(Heap& heap, size_t i, size_t count)
  {
    using Elem = typename Heap::value_type;
    Elem val = std::move(heap[i]);
    uint64_t valCode = val.code;

    while (true) {
      size_t left = 2 * i + 1;
      if (left >= count) break;

      size_t right = left + 1;

      size_t best = left;
      if (right < count && heap[right].code < heap[left].code)
        best = right;

      if (!(heap[best].code < valCode))
        break;

      heap[i] = std::move(heap[best]);
      i = best;
    }

    heap[i] = std::move(val);
  }

  template<typename HeapType>
  __forceinline void popHeapUltraFast(HeapType& heap)
  {
    const size_t n = heap.size();
    if (n <= 1) {
      if (n == 1) heap.pop_back();
      return;
    }

    using Elem = typename HeapType::value_type;
    Elem* __restrict data = heap.data();

    Elem last = std::move(data[n - 1]);
    heap.pop_back();

    size_t hole = 0;
    const size_t limit = heap.size(); // after pop_back

    while (true) {
      size_t left = 2 * hole + 1;
      if (left >= limit) break;

      // Prefetch the NEXT level while comparing this one.
      //
      // The sift-down is the loop's memory problem: ~24 levels per pop, each a random
      // 8-byte read out of a heap that reaches 160-400 MB, 9.2M times a run. That also
      // explains why halving the entry count (lazy revalidation) bought only ~10% -- a
      // binary heap is log-depth, so half the entries removes ONE level of 24.
      //
      // It prefetches cleanly because the descent is only two-way: from `hole` we move to
      // 2h+1 or 2h+2, whose children together occupy indices 4h+3 .. 4h+6 -- four
      // consecutive 8-byte codes, 32 bytes, normally one cache line. So a single prefetch
      // covers whichever branch the comparison takes, with no speculation about which.
      {
        const size_t gk = 4 * hole + 3;
        if (gk < limit) {
          _mm_prefetch((const char*)(data + gk), _MM_HINT_T0);
          // 4h+3..4h+6 straddles a line when the base is badly aligned; +3 covers it.
          const size_t gk2 = gk + 3;
          if (gk2 < limit) _mm_prefetch((const char*)(data + gk2), _MM_HINT_T0);
        }
      }

      size_t right = left + 1;
      size_t child = (right < limit && data[right].code < data[left].code) ? right : left;

      if (!(last.code > data[child].code)) break;

      data[hole] = std::move(data[child]);
      hole = child;
    }

    data[hole] = std::move(last);
  }

  __forceinline void fastPushHeap(HeapType& heap) {
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

  __forceinline void fastPushHeap(HeapType& heap, size_t index)
  {
    auto val = std::move(heap[index]);
    const uint64_t valCode = val.code;

    while (index > 0) {
      size_t parent = (index - 1) >> 1;
      const uint64_t parentCode = heap[parent].code;
      if (!(valCode < parentCode)) break;
      heap[index] = heap[parent];
      index = parent;
    }

    heap[index] = val;
  }

  template<typename HeapType>
  __forceinline void fastPushHeapPartial(HeapType& heap, size_t i) 
  {
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


  template <typename T>
  struct alignas(64) AlignedSlot {
    T value;
    char padding[64 - sizeof(T)];

    AlignedSlot() {}
    explicit AlignedSlot(const T& v) : value(v) {}

    // Allow move, disallow copy (required for atomics)
    AlignedSlot(AlignedSlot&&) = default;
    AlignedSlot& operator=(AlignedSlot&&) = default;

    AlignedSlot(const AlignedSlot&) = delete;
    AlignedSlot& operator=(const AlignedSlot&) = delete;
  };

  class HeapThreadPool {
  public:
    explicit HeapThreadPool(int numThreads) :
      threadCount(numThreads),
      doneCount(0),
      shuttingDown(false),
      generation(0),
      bigBuffer(nullptr),
      bigBufferSize(0)
    {
      threadBuffers.resize(numThreads);
      threadOffsets.resize(numThreads + 1);

      start();
    }

    ~HeapThreadPool()
    {
      shutdown();
    }

    void shutdown()
    {
      {
        std::lock_guard<std::mutex> lock(mtx);
        shuttingDown = true;
        cvStart.notify_all();
      }
      for (auto& t : workers) {
        if (t.joinable()) t.join();
      }
      workers.clear();
    }

    size_t filterValidHeap(HeapType& out, const HeapType& h)
    {
      {
        std::lock_guard<std::mutex> lock(mtx);
        originalHeap = &h;
        doneCount = 0;
        const size_t numElementsNeeded = h.size();
        if (numElementsNeeded > bigBufferSize) {
          _aligned_free(bigBuffer);
          bigBuffer = (PackedHeapElem*)_aligned_malloc(numElementsNeeded * sizeof(PackedHeapElem), 64);
          bigBufferSize = numElementsNeeded;
        }
        generation++;
      }

      cvStart.notify_all();

      std::unique_lock<std::mutex> lock(mtx);
      cvDone.wait(lock, [this]() {
        return doneCount.load() == threadCount;
        });

      originalHeap = nullptr;

      const size_t cnt = workers.size();
      size_t totalSize = 0;
      for (size_t i = 0; i < cnt; ++i) {
        totalSize += threadBuffers[i].value.second;
      }

      out.reserve(h.capacity());
      out.resize(totalSize);
      threadOffsets[0] = 0;
      for (size_t i = 0; i < cnt; ++i) {
        threadOffsets[i + 1] = threadOffsets[i] + threadBuffers[i].value.second;
      }

      for (int i = 0; i < (int) threadCount; ++i) {
        std::memcpy(&out[threadOffsets[i]], threadBuffers[i].value.first, threadBuffers[i].value.second * sizeof(PackedHeapElem));
      }

      return totalSize;
    }

  private:
    void pinThreadToCore(int threadIndex)
    {
      DWORD_PTR mask = 1ull << (threadIndex * 2);
      SetThreadAffinityMask(GetCurrentThread(), mask);
      // JPB WIP BUG Must be restored.
    }

    void start()
    {
      for (int i = 0; i < threadCount; ++i) {
        workers.emplace_back([this, i]() { workerLoop(i); });
      }
    }

    void workerLoop(int tid)
    {
      using Leaf = vcg::tri::TriEdgeCollapseQuadric<CLEAN::Mesh,
        vcg::tri::BasicVertexPair<CLEAN::Vertex>,
        CLEAN::TriEdgeCollapse,
        CLEAN::QHelper>;

      pinThreadToCore(tid);

      int localGen = 0;

      while (true) {
        {
          std::unique_lock<std::mutex> lock(mtx);
          cvStart.wait(lock, [this, localGen]() {
            return generation != localGen || shuttingDown;
        });

          if (shuttingDown) return;

          localGen = generation; // latch the new generation
        }

        // Process a portion of the workload and write the results to
        // the bigbuffer.  Our portion starts at tid*heapSize/# workers
        const HeapType& h = *originalHeap;
        const size_t heapSize = h.size();
        const size_t chunkSize = heapSize / workers.size();
        const size_t remainder = heapSize % workers.size();
        const size_t startIdx = tid * chunkSize + std::min((size_t) tid, remainder);
        const size_t endIdx = startIdx + chunkSize + (tid < remainder ? 1 : 0);
        PackedHeapElem* __restrict dst = bigBuffer + startIdx;
        threadBuffers[tid].value.first = dst;
        threadBuffers[tid].value.second = 0;

        // It is faster to copy to stack memory first.
        // Prefetch on the element and its used values is not helpful.
        // movsq is faster then memcpy.
        // Blocksizes 64 is slower, 128 better, 256 slower
        enum { kBlockSize = 128 };

        PackedHeapElem alignas(64) tmp[kBlockSize];

        for (size_t i = startIdx; i + (kBlockSize-1) < endIdx; i += kBlockSize) {
          int count = 0;
          for (int j = 0; j < kBlockSize; ++j) {
            const Leaf* locMod = CodeToPtr<Leaf>(h[i + j].code);
            // IsUpToDateFast: const and allocation-free. The full test can walk a VF
            // ring and refresh localMark, neither of which is safe from these
            // workers. Conservative in the SAFE direction -- it only drops entries
            // that are definitely dead, so nothing the pop would accept is lost.
            if (locMod->IsUpToDateFast())
              tmp[count++] = h[i + j]; // Use address from vector directly
          }
          memcpy(dst, tmp, count * sizeof(PackedHeapElem));
          dst += count;
        }

        const size_t tailStart = endIdx - ((endIdx - startIdx) & (kBlockSize-1));
        if (tailStart < endIdx) {
          int count = 0;
          for (size_t i = tailStart; i < endIdx; ++i) {
            const PackedHeapElem& el = h[i];
            const Leaf* locMod = CodeToPtr<Leaf>(el.code);
            if (locMod->IsUpToDateFast())
              tmp[count++] = el;
          }

          memcpy(dst, tmp, count * sizeof(PackedHeapElem));
          dst += count;
        }

        threadBuffers[tid].value.second = dst - threadBuffers[tid].value.first;

        if (doneCount.fetch_add(1, std::memory_order_acq_rel) + 1 == threadCount) {
          std::lock_guard<std::mutex> lock(mtx);
          cvDone.notify_one();
        }
      }
    }

    // Thread-safe worker state
    std::mutex mtx;
    std::vector<std::thread> workers;
    std::vector<AlignedSlot<std::pair<PackedHeapElem*, size_t> /* span */>> threadBuffers;
    std::vector<size_t> threadOffsets;
    PackedHeapElem* bigBuffer;
    size_t bigBufferSize;

    const HeapType* originalHeap = nullptr;

    std::condition_variable cvStart;
    std::condition_variable cvDone;

    int threadCount;

    std::atomic<int> doneCount;
    std::atomic<bool> shuttingDown;
    std::atomic<int> generation;
  };


 static  double CalibrateCPUFrequency()
 {
    LARGE_INTEGER qpc_start, qpc_end, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&qpc_start);
    uint64_t tsc_start = __rdtsc();

    Sleep(100); // 100 ms

    uint64_t tsc_end = __rdtsc();
    QueryPerformanceCounter(&qpc_end);

    double seconds = (qpc_end.QuadPart - qpc_start.QuadPart) / (double)freq.QuadPart;
    return (tsc_end - tsc_start) / seconds;
  }


  // Integrated DoOptimization using the refactored pool
#if 0 // Original
 void ClearHeap()
 {
   using Leaf = vcg::tri::TriEdgeCollapseQuadric<CLEAN::Mesh,
     vcg::tri::BasicVertexPair<CLEAN::Vertex>,
     CLEAN::TriEdgeCollapse,
     CLEAN::QHelper>;

   //    int sz=h.size(); int t0=clock();
   for (auto hi = h.begin(); hi != h.end();)
   {
     Leaf* __restrict locMod = CodeToPtr<Leaf>(hi->code);

     if (!locMod->IsUpToDate())
     {
       *hi = h.back();
       if (&*hi == &h.back())
       {
         hi = h.end();
         h.pop_back();
         break;
       }
       h.pop_back();
       continue;
     }
     ++hi;
   }
   //    printf("\nReduced heap from %7i to %7i (fn %7i) in %7.2f \n",sz,h.size(),m.fn,float(clock()-t0)/CLOCKS_PER_SEC);
   make_heap(h.begin(), h.end());
 }

 bool DoOptimization()
 {
   using Leaf = vcg::tri::TriEdgeCollapseQuadric<CLEAN::Mesh,
     vcg::tri::BasicVertexPair<CLEAN::Vertex>,
     CLEAN::TriEdgeCollapse,
     CLEAN::QHelper>;

   startTime = Clock::now();
   nPerformedOps = 0;
   while (!GoalReached() && !h.empty())
   {
     if (h.size() > m.SimplexNumber() * HeapSimplexRatio)  ClearHeap();
     std::pop_heap(h.begin(), h.end());
     Leaf* __restrict locMod = CodeToPtr<Leaf>(h.back().code);
     h.pop_back();

     if (locMod->IsUpToDate())
     {
       //printf("popped out: %s\n",locMod->Info(m));
       //if (locMod->IsFeasible(this->pp))
       {
         nPerformedOps++;
         locMod->Execute(m);
         locMod->UpdateHeap(h);
       }
     }
   }
   return !h.empty();
 }
#else
  bool DoOptimization(size_t maxVertices)
  {
    using Leaf = vcg::tri::TriEdgeCollapseQuadric<CLEAN::Mesh,
      vcg::tri::BasicVertexPair<CLEAN::Vertex>,
      CLEAN::TriEdgeCollapse,
      CLEAN::QHelper>;

    assert(((tf & LOnSimplices) == 0) || (nTargetSimplices != -1));
    assert(((tf & LOnVertices) == 0) || (nTargetVertices != -1));
    assert(((tf & LOnOps) == 0) || (nTargetOps != -1));
    assert(((tf & LOMetric) == 0) || (targetMetric != -1));
    assert(((tf & LOTime) == 0) || (timeBudget != -1));

    static std::vector<void*> pairsScratch;
    static std::vector<void*> toAddScratch;
    pairsScratch.reserve(maxVertices);
    toAddScratch.reserve(maxVertices);

    startTime = Clock::now();
    nPerformedOps = 0;
    currMetric = 0;                                     // reset error tracker for this pass
    const bool useMetric = IsTerminationFlag(LOMetric); // error-bounded stop enabled?

    // JPB WIP BUG Double check this.
    // Faster to use all the threads and not worry about pinning. 43.213
    static HeapThreadPool pool(std::thread::hardware_concurrency() / 4);

    // Drain the mini-buffer into the main heap once it is worth a partial re-heapify.
    // HOISTED to a lambda because it is no longer reachable from one place only: the
    // full-lazy path re-inserts a re-keyed entry and then `continue`s, skipping the rest
    // of the iteration. Left inline, hBuffer would grow without bound and the
    // min_element rescan done on every buffer pop would go linear.
    auto drainBuffer = [&]() {
      if (hBuffer.size() < 64)
        return;
      const size_t oldSize = h.size();
      const size_t numNew = hBuffer.size();

      h.reserve(oldSize + numNew);
      h.insert(h.end(),
        std::make_move_iterator(hBuffer.begin()),
        std::make_move_iterator(hBuffer.end()));
      hBuffer.clear();

      const size_t totalSize = h.size();

      // Only fix the parents of the newly added range
      const int64_t firstParent = (oldSize - 1) >> 1;
      const int64_t lastParent = (totalSize - 1) >> 1;

      for (int64_t i = lastParent; i >= firstParent; --i)
        siftDown(h, size_t(i), totalSize);
    };

    while (!GoalReached()) {
      unsigned long long _lpT = g_deciLoopProf ? __rdtsc() : 0ull;
      // Check top element without popping
      // Find the min
      bool minIsInBuffer;
      if (h.empty()) {
        if (hBuffer.empty()) {
          break;
        }
        minIsInBuffer = true;
      } else {
        // heap not empty
        if (!hBuffer.empty()) {
          minIsInBuffer = hBuffer[0].code < h.front().code;
        } else {
          minIsInBuffer = false;
        }
      }
      
      Leaf* locMod;
      uint64_t selCode = 0;                       // packed code of the selected min (error in hi 32b)
      if (minIsInBuffer) {
        selCode = hBuffer.front().code;
        locMod = CodeToPtr<Leaf>(selCode);
        hBuffer.erase(hBuffer.begin());

        if (!hBuffer.empty()) {
          auto minIt = std::min_element(hBuffer.begin(), hBuffer.end(),
            [](const auto& a, const auto& b) { return a.code < b.code; });

          if (minIt != hBuffer.begin()) {
            std::iter_swap(hBuffer.begin(), minIt);
          }
        }
      } else {
        // Min is in the heap, ignore the buffer
        selCode = h.front().code;
        locMod = CodeToPtr<Leaf>(selCode);
        auto* v0 = locMod->pos.V(0);
        auto* v1 = locMod->pos.V(1);

       // std::cout << "stored: v0 v1: " << &Leaf::QH::Qd(v0) << " " << &Leaf::QH::Qd(v1) << "\n";

#if 1 // JPB WIP BUG
        _mm_prefetch((char*)v0, _MM_HINT_T1); // start loading Face* line
        _mm_prefetch((char*)v1, _MM_HINT_T1); // start loading Face* line
        // QH::Qd(v) is (*TDp())[v] -- taking its address dereferences TDp, which is NULL
        // on the recentred path. These must branch, not just be skipped.
        if (g_deciRecentred) {
          _mm_prefetch((char*)&Leaf::QH::Rq(v0), _MM_HINT_T1);          // 40 B: usually
          _mm_prefetch(((char*)&Leaf::QH::Rq(v0)) + 39, _MM_HINT_T1);   // one line, two
          _mm_prefetch((char*)&Leaf::QH::Rq(v1), _MM_HINT_T1);          // when it straddles
          _mm_prefetch(((char*)&Leaf::QH::Rq(v1)) + 39, _MM_HINT_T1);
        } else {
        _mm_prefetch((char*)&Leaf::QH::Qd(v0), _MM_HINT_T1);
        _mm_prefetch(((char*)&Leaf::QH::Qd(v0))+64, _MM_HINT_T1);
        _mm_prefetch((char*)&Leaf::QH::Qd(v1), _MM_HINT_T1);
        _mm_prefetch(((char*)&Leaf::QH::Qd(v1)) + 64, _MM_HINT_T1);
        }
#endif
        popHeapUltraFast(h);
        // The moment you pop the heap, it is likely a new minimum will replace it.
        // Begin prefetching its data now.
        if (!h.empty()) {
          Leaf* likelyMinimum = CodeToPtr<Leaf>(h.front().code);
#if 1 // JPB WIP BUG
          _mm_prefetch((char*)likelyMinimum, _MM_HINT_T1);
#endif
        }
      }

      // track the geometric error of the collapse about to be applied so the
      // LOMetric goal (error-bounded decimation) can stop at a tolerance
      if (useMetric && selCode != 0) {
        uint32_t priBits = (uint32_t)(selCode >> 32);
        currMetric = (ScalarType)(float&)priBits; // priority is the float error bit-packed in hi 32b
      }
      if (g_deciLoopProf) { const unsigned long long _t = __rdtsc(); g_deciSetup.tscPop += _t - _lpT; _lpT = _t; }
      ++heapPopCount;
      DECI_COUNT(nHeapPop);
      if (!locMod->IsUpToDate()) {
        ++stalePopCount;
        DECI_COUNT(nHeapStale);
        if (g_deciLoopProf) g_deciSetup.tscStale += __rdtsc() - _lpT;
        continue;
      }
      if (g_deciLoopProf) { const unsigned long long _t = __rdtsc(); g_deciSetup.tscStale += _t - _lpT; _lpT = _t; }

      // [DECI-AUDIT] -- see DeciSetupTimes. Runs only with OPENMVS_MESH_DECI_AUDIT=1.
      //
      // NOTE it calls ComputePriority, which also refreshes locMod->optimalPos. So with
      // the audit ON a stale stored position is silently corrected before Execute uses
      // it: the audit REPORTS staleness without letting it affect the output mesh. That
      // isolation is deliberate, and it is also why this must not be left enabled.
      if (DECI_PROFILE && g_deciAudit) {
        typedef typename MeshType::FaceType   AuditFaceType;
        typedef typename MeshType::VertexType AuditVertType;
        AuditVertType * const av0 = locMod->pos.V(0);
        AuditVertType * const av1 = locMod->pos.V(1);

        // (a) are the endpoints still adjacent? A collapse of a non-edge merges two
        //     unrelated vertices -- silent corruption, not a crash.
        bool adj = false;
        {
          vcg::face::VFIterator<AuditFaceType> ax;
          for (ax.F() = av0->VFp(), ax.I() = av0->VFi(); ax.F() != 0; ++ax) {
            const AuditFaceType &af = *ax.F();
            if (af.IsD()) continue;
            if (af.V(0) == av1 || af.V(1) == av1 || af.V(2) == av1) { adj = true; break; }
          }
        }
        if (!adj) ++g_deciSetup.nAuditNonEdge;

        // (b) is the priority the heap ordered on still the right one?
        uint32_t _spBits = (uint32_t)(selCode >> 32);
        const float storedPri = (float&)_spBits;
        const float freshPri  = locMod->ComputePriority();
        const double _d = std::fabs((double)freshPri - (double)storedPri);
        const double _s = std::fabs((double)storedPri);
        const double _r = (_s > 0.0) ? (_d / _s) : (_d > 0.0 ? 1.0 : 0.0);
        if (_r > g_deciSetup.auditMaxPriRel) g_deciSetup.auditMaxPriRel = _r;
        if (_r > 1e-6) ++g_deciSetup.nAuditPriStale;
        ++g_deciSetup.nAudited;
      }

      // [FP32-PROBE] -- see DeciSetupTimes. OPENMVS_MESH_QUADRIC_FP32_PROBE=1.
      // Restores the double quadrics AND re-runs ComputePriority afterwards, so
      // optimalPos is left exactly as the double path would have it: the probe measures
      // without perturbing the collapse.
      // !g_deciRecentred: this probe reads QH::Qd, which is not allocated there.
      if (DECI_PROFILE && g_deciFp32Probe && !g_deciRecentred) {
        typedef typename MeshType::VertexType ProbeVertType;
        ProbeVertType * const pv0 = locMod->pos.V(0);
        ProbeVertType * const pv1 = locMod->pos.V(1);
        auto &Q0 = Leaf::QH::Qd(pv0);
        auto &Q1 = Leaf::QH::Qd(pv1);

        double save0[10], save1[10];
        for (int k = 0; k < 10; ++k) {
          save0[k] = Q0.array[k];
          save1[k] = Q1.array[k];
        }
        const double a9 = std::fabs(save0[9]), a0 = std::fabs(save0[0]);
        if (a9 > g_deciSetup.fp32MaxAbsA9) g_deciSetup.fp32MaxAbsA9 = a9;
        if (a0 > g_deciSetup.fp32MaxAbsA0) g_deciSetup.fp32MaxAbsA0 = a0;

        const float priD = locMod->ComputePriority();          // true, double
        for (int k = 0; k < 10; ++k) {                          // round-trip through float
          Q0.array[k] = (double)(float)save0[k];
          Q1.array[k] = (double)(float)save1[k];
        }
        const float priF = locMod->ComputePriority();          // as float storage would give
        for (int k = 0; k < 10; ++k) {                          // restore
          Q0.array[k] = save0[k];
          Q1.array[k] = save1[k];
        }
        locMod->ComputePriority();                              // restore optimalPos

        const double dd = std::fabs((double)priF - (double)priD);
        const double ss = std::fabs((double)priD);
        const double rr = (ss > 0.0) ? (dd / ss) : (dd > 0.0 ? 1.0 : 0.0);
        if (rr > g_deciSetup.fp32MaxRel) g_deciSetup.fp32MaxRel = rr;
        if (rr > 1e-3) ++g_deciSetup.nFp32Bad;

        // ---- and again, RECENTRED on each vertex before rounding -----------------
        // Exact translation by r: A unchanged, b' = b + 2Ar, c' = Apply(r). Round the
        // recentred form to float, translate back with -r in double, and re-run the real
        // ComputePriority. That simulates "store recentred float, expand for the solve"
        // and is an UPPER bound on the error: a real implementation would solve in the
        // recentred frame and never rebuild the large world-frame terms at all.
        {
          double w0[10], w1[10];
          for (int pass = 0; pass < 2; ++pass) {
            const double *src = pass ? save1 : save0;
            double *dst = pass ? w1 : w0;
            const auto &P = pass ? pv1->cP() : pv0->cP();
            const double rx = (double)P[0], ry = (double)P[1], rz = (double)P[2];
            // A.r  (a is the packed symmetric 3x3: a11 a12 a13 a22 a23 a33)
            const double Ar0 = src[0]*rx + src[1]*ry + src[2]*rz;
            const double Ar1 = src[1]*rx + src[3]*ry + src[4]*rz;
            const double Ar2 = src[2]*rx + src[4]*ry + src[5]*rz;
            // c' = Q(r) = r.A.r + b.r + c
            const double cR = (rx*Ar0 + ry*Ar1 + rz*Ar2)
                            + (src[6]*rx + src[7]*ry + src[8]*rz) + src[9];
            double q[10];
            for (int k = 0; k < 6; ++k) q[k] = src[k];
            q[6] = src[6] + 2.0*Ar0;
            q[7] = src[7] + 2.0*Ar1;
            q[8] = src[8] + 2.0*Ar2;
            q[9] = cR;
            const double ac = std::fabs(cR);
            if (ac > g_deciSetup.rcMaxAbsC) g_deciSetup.rcMaxAbsC = ac;
            for (int k = 0; k < 10; ++k) q[k] = (double)(float)q[k];   // <-- float storage
            // translate back by -r, in double
            const double Br0 = q[0]*rx + q[1]*ry + q[2]*rz;
            const double Br1 = q[1]*rx + q[3]*ry + q[4]*rz;
            const double Br2 = q[2]*rx + q[4]*ry + q[5]*rz;
            for (int k = 0; k < 6; ++k) dst[k] = q[k];
            dst[6] = q[6] - 2.0*Br0;
            dst[7] = q[7] - 2.0*Br1;
            dst[8] = q[8] - 2.0*Br2;
            dst[9] = (rx*Br0 + ry*Br1 + rz*Br2) - (q[6]*rx + q[7]*ry + q[8]*rz) + q[9];
          }
          for (int k = 0; k < 10; ++k) { Q0.array[k] = w0[k]; Q1.array[k] = w1[k]; }
          const float priR = locMod->ComputePriority();
          const double dr = std::fabs((double)priR - (double)priD);
          const double rrr = (ss > 0.0) ? (dr / ss) : (dr > 0.0 ? 1.0 : 0.0);
          if (rrr > g_deciSetup.rcMaxRel) g_deciSetup.rcMaxRel = rrr;
          if (rrr > 1e-3) ++g_deciSetup.nRcBad;
        }
        ++g_deciSetup.nFp32Probed;
      }

      // I think exeucte and updateheap share FindSets info.
      if (g_deciLoopProf) _lpT = __rdtsc();   // exclude any audit/probe above
      locMod->Execute(m);
      if (g_deciLoopProf) { const unsigned long long _t = __rdtsc(); g_deciSetup.tscExec += _t - _lpT; _lpT = _t; }
      locMod->UpdateHeap((void*) &h, (void*)&hBuffer, pairsScratch, toAddScratch);
      if (g_deciLoopProf) { const unsigned long long _t = __rdtsc(); g_deciSetup.tscUpd += _t - _lpT; _lpT = _t; }

      drainBuffer();


      // --- Periodic cleanup decision ---
      // Uses a sliding window and will attempt a heap update when
      // the ratio of stale pops reaches a certain tolerance.
      // 512 much slower, 128 much slower than that.
      //if ((heapPopCount & ((1024 * 256) - 1)) == 0) { // Cheap modulo for power-of-2
        const bool staleHeavy = heapPopCount >= 262144 && ( stalePopCount > heapPopCount / 2 );

        if (staleHeavy) {
          std::move(std::begin(hBuffer), std::end(hBuffer), std::back_inserter(h));
          hBuffer.clear();

          // Clean up main heap
          HeapType newHeap;
          //auto st = __rdtsc();
          size_t finalSize = pool.filterValidHeap(newHeap, h);
          //auto endme = __rdtsc();

          //double cpu_hz = CalibrateCPUFrequency();

          //std::cout << "clean " << (endme - st) / cpu_hz << "\n";

          std::swap(h, newHeap);
          makeHeapUltraFast(h);

          heapPopCount = 0;
          stalePopCount = 0;
       // }
      }

      // Whatever is left of the iteration: compaction, hBuffer bookkeeping, GoalReached.
      if (g_deciLoopProf) g_deciSetup.tscOther += __rdtsc() - _lpT;
    }

    // Stop the outer decimation loop when a HARD goal (target simplices or error
    // metric) is reached; keep going (return true) if we merely yielded on the time
    // budget with work still queued.
    const bool workLeft = !h.empty() || !hBuffer.empty();
    const bool hardGoal =
        (IsTerminationFlag(LOnSimplices) && m.SimplexNumber() <= nTargetSimplices) ||
        (IsTerminationFlag(LOMetric)     && currMetric > targetMetric);
    return workLeft && !hardGoal;
  }
#endif

	///initialize for all vertex the temporary mark must call only at the start of decimation
	///by default it takes the first element in the heap and calls Init (static funcion) of that type
	///of local modification. 
  template <class LocalModificationType> void Init()
	{
    omp_set_nested(0); // JPB WIP BUG

    // JPB WIP BUG omp can ignore my thread request.
    omp_set_dynamic(0);  // must be called before any parallel region

    vcg::tri::InitVertexIMark(m);
		
    // The expected size of heap depends on the type of the local modification we are using..
    HeapSimplexRatio = LocalModificationType::HeapSimplexRatio(pp);

    LocalModificationType::Init(m,h,pp);

    //std::make_heap(h.begin(),h.end());
    { DeciPhaseTimer _t( g_deciSetup.heapify ); makeHeapUltraFast(h); }

    // Unused if(!h.empty()) currMetric=h.front().pri;
	}


	template <class LocalModificationType> void Finalize()
	{
    LocalModificationType::Finalize(m,h,pp);

    LocalModificationType::Release();
	}


	/// say if the process is to end or not: the process ends when any of the termination conditions is verified
	/// override this function to implemetn other tests
	bool GoalReached(){
    if ( IsTerminationFlag(LOnSimplices) &&	( m.SimplexNumber()<= nTargetSimplices)) return true;
    // Unused if ( IsTerminationFlag(LOnVertices)  &&  ( m.VertexNumber() <= nTargetVertices)) return true;
    // Unused if ( IsTerminationFlag(LOnOps)		   && (nPerformedOps	== nTargetOps)) return true;
    // Unused if ( IsTerminationFlag(LOMetric)		 &&  ( currMetric		> targetMetric)) return true;
    if ( IsTerminationFlag(LOMetric) && ( currMetric > targetMetric)) return true;
    if ( IsTerminationFlag(LOTime) )
    {
      static int cntr = 1;
      ++cntr;
      if (!(cntr & 255)) {
        auto now = Clock::now();
        if (now < startTime) return true;  // overflow not needed, but safe
        if (std::chrono::duration<double>(now - startTime).count() > timeBudget)
          return true;
    }
    }
		return false;
	}

};//end class decimation

}//end namespace

#endif