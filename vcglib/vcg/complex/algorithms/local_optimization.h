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
using Clock = std::chrono::steady_clock;



std::atomic<uint64_t> tries = 0;
std::atomic<uint64_t> hits = 0;

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

template<typename T>
uint64_t PtrToOffset(T* ptr)
{
  // Assume g_qBlocks.back() is non-empty and ptr is within that last block
  void* base = g_qBlocks.back();  // Most recent block
  size_t blockIndex = g_qBlocks.size() - 1;

  // Avoid repeated casts, compute offset in one step
  uintptr_t byteOffset = reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(base);

  uint64_t indexInBlock = byteOffset / (36);

  // Combine into 64-bit ID: bits 23.. are block ID, bits 0..6 are index
  return (blockIndex << 23) | indexInBlock;
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
            if (locMod->IsUpToDate())
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
            if (locMod->IsUpToDate())
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
  bool DoOptimization()
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

    startTime = Clock::now();
    nPerformedOps = 0;

    // Faster to use all the threads and not worry about pinning. 43.213
    static HeapThreadPool pool(std::thread::hardware_concurrency() / 4);

    while (!GoalReached()) {
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
      if (minIsInBuffer) {
        locMod = CodeToPtr<Leaf>(hBuffer.front().code);
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
        locMod = CodeToPtr<Leaf>(h.front().code);
        auto* v0 = locMod->pos.V(0);
        auto* v1 = locMod->pos.V(1);

       // std::cout << "stored: v0 v1: " << &Leaf::QH::Qd(v0) << " " << &Leaf::QH::Qd(v1) << "\n";

        _mm_prefetch((char*)v0, _MM_HINT_T1); // start loading Face* line
        _mm_prefetch((char*)v1, _MM_HINT_T1); // start loading Face* line
        _mm_prefetch((char*)&Leaf::QH::Qd(v0), _MM_HINT_T1);
        _mm_prefetch(((char*)&Leaf::QH::Qd(v0))+64, _MM_HINT_T1);
        _mm_prefetch((char*)&Leaf::QH::Qd(v1), _MM_HINT_T1);
        _mm_prefetch(((char*)&Leaf::QH::Qd(v1)) + 64, _MM_HINT_T1);
        popHeapUltraFast(h);
        // The moment you pop the heap, it is likely a new minimum will replace it.
        // Begin prefetching its data now.
        if (!h.empty()) {
          Leaf* likelyMinimum = CodeToPtr<Leaf>(h.front().code);
          _mm_prefetch((char*)likelyMinimum, _MM_HINT_T1);
        }
      }

      ++heapPopCount;
      if (!locMod->IsUpToDate()) {
        ++stalePopCount;
        continue;
      }

      // I think exeucte and updateheap share FindSets info.
      locMod->Execute(m);
      locMod->UpdateHeap((void*) &h, (void*)&hBuffer);

      // always pushes to the buffer.
      // Is buffer too big?
// Is buffer too big?
      if (hBuffer.size() >= 64) {
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
      }


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
    }

    return !h.empty() || !hBuffer.empty();
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
    makeHeapUltraFast(h);

    // Unused if(!h.empty()) currMetric=h.front().pri;
	}


	template <class LocalModificationType> void Finalize()
	{
    LocalModificationType::Finalize(m,h,pp);
	}


	/// say if the process is to end or not: the process ends when any of the termination conditions is verified
	/// override this function to implemetn other tests
	bool GoalReached(){
    if ( IsTerminationFlag(LOnSimplices) &&	( m.SimplexNumber()<= nTargetSimplices)) return true;
    // Unused if ( IsTerminationFlag(LOnVertices)  &&  ( m.VertexNumber() <= nTargetVertices)) return true;
    // Unused if ( IsTerminationFlag(LOnOps)		   && (nPerformedOps	== nTargetOps)) return true;
    // Unused if ( IsTerminationFlag(LOMetric)		 &&  ( currMetric		> targetMetric)) return true;
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