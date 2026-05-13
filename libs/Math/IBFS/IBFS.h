/*
#########################################################
#                                                       #
#  IBFSGraph -  Software for solving                    #
#               Maximum s-t Flow / Minimum s-t Cut      #
#               using the IBFS algorithm                #
#                                                       #
#  http://www.cs.tau.ac.il/~sagihed/ibfs/               #
#                                                       #
#  Haim Kaplan (haimk@cs.tau.ac.il)                     #
#  Sagi Hed (sagihed@post.tau.ac.il)                    #
#                                                       #
#  2015 - modified by cDc@seacave                       #
#                                                       #
#########################################################

This software implements the IBFS (Incremental Breadth First Search) maximum flow algorithm from
	"Maximum flows by incremental breadth-first search"
	Andrew V. Goldberg, Sagi Hed, Haim Kaplan, Robert E. Tarjan, and Renato F. Werneck.
	In Proceedings of the 19th European conference on Algorithms, ESA'11, pages 457-468.
	ISBN 978-3-642-23718-8
	2011

Copyright Haim Kaplan (haimk@cs.tau.ac.il) and Sagi Hed (sagihed@post.tau.ac.il)

###########
# LICENSE #
###########
This software can be used for research purposes only.
If you use this software for research purposes, you should cite the aforementioned paper
in any resulting publication and appropriately credit it.

If you require another license, please contact the above.

###########
#  USAGE  #
###########

	IBFSGraph g = new IBFSGraph();

	// g.initSize(numNodes, numEdges) indicate the number of nodes and edges in the graph.
	// Number of edges does not include edges from the source and to the sink!
	g->initSize(4, 5);

	// g.addNode(nodeID, capFromSource, capToSink) indicate node is connected
	// to source and sink with appropriate capacities
	// nodeID is between 0 ... numNodes as supplied in initSize.
	g->addNode(0, 500, 100);
	g->addNode(1, 200, 0);
	g->addNode(2, 50, 50);
	g->addNode(3, 0, 0); // can discard this line

	// g.addEdge(fromNodeID, toNodeID, capForward, capReverse) indicate edge
	// connect fromNodeID and toNodeID
	// with appropriate forward capacity and reverse capacity.
	g->addEdge(0, 1, 100, 40);
	g->addEdge(0, 2, 200, 800);
	g->addEdge(0, 3, 500, 500);
	g->addEdge(1, 3, 300, 100);
	g->addEdge(2, 3, 500, 500);
	
	long startTime = getTime();
	g->initGraph();
	g->computeMaxFlow();
	long time = getTime()-startTime;

	fprintf(stdout, "time=%d\n", time);
	fprintf(stdout, "flow=%d\n", g->getFlow());
	for (int i=0; i < 3; i++)
		fprintf("node %d has label %d\n", i, g->isNodeOnSrcSide(i));

*/


#ifndef _IBFS_H__
#define _IBFS_H__

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <memory>
#include <atomic>
#include <vector>


#ifndef IBIO
#define IBIO 0
#endif
#ifndef IBTEST
#define IBTEST 0
#endif
#ifndef IBSTATS
#define IBSTATS 0
#endif
#ifndef IBDEBUG
#define IBDEBUG(X) fprintf(stdout, X"\n"); fflush(stdout)
#endif

#define IB_ALTERNATE_SMART 1

// Sort each node's arcs by headIdx after finalizeGraph(). Combined with the
// upstream Morton cell-id ordering, this makes the inner BFS / adoption /
// augment loops walk monotonically increasing memory addresses, which the
// hardware prefetcher loves. Two-pass parallel impl is race-free; pure
// permutation, no semantic change.
#ifndef IBFS_OPT_ARC_SORT
#define IBFS_OPT_ARC_SORT 1
#endif


namespace IBFS {

typedef float Real;
typedef float EdgeCap;

// 32-bit index types for compact Node/Arc layout.
// kNullIdx       : "no node" sentinel (e.g. unset firstSon, end-of-list).
// kOrphansEndIdx : terminator used inside orphan/bucket linked lists,
//                  distinguishable from kNullIdx for legacy semantics.
// kNullParent    : "no parent" sentinel for ParentRef.
typedef uint32_t NodeIdx;
typedef uint32_t ParentRef;     // packed (nodeIdx << 2) | arcSlot(0..3)

static constexpr NodeIdx   kNullIdx       = 0xFFFFFFFFu;
static constexpr NodeIdx   kOrphansEndIdx = 0xFFFFFFFEu;
static constexpr ParentRef kNullParent    = 0xFFFFFFFFu;

// Maximum supported node count given 30 bits of headIdx packed inside Arc.
// Bits [31..30] = revIdx (2 bits), [29..0] = headIdx (30 bits, max ~1.07B).
static constexpr uint32_t  kMaxNodes      = (1u << 30) - 2u; // leave room for sentinels

class IBFSStats
{
public:
	IBFSStats()
	{
		Real C = (IBSTATS ? 0 : -1);
		augs=C;
		growthS=C;
		growthT=C;
		orphans=C;
		growthArcs=C;
		pushes=C;
		orphanArcs1=C;
		orphanArcs2=C;
		orphanArcs3=C;
		if (IBSTATS) augLenMin = (1 << 30);
		else augLenMin=C;
		augLenMax=C;
	}
	void inline incAugs() {if (IBSTATS) augs++;}
	Real inline getAugs() {return augs;}
	void inline incGrowthS() {if (IBSTATS) growthS++;}
	Real inline getGrowthS() {return growthS;}
	void inline incGrowthT() {if (IBSTATS) growthT++;}
	Real inline getGrowthT() {return growthT;}
	void inline incOrphans() {if (IBSTATS) orphans++;}
	Real inline getOrphans() {return orphans;}
	void inline incGrowthArcs() {if (IBSTATS) growthArcs++;}
	Real inline getGrowthArcs() {return growthArcs;}
	void inline incPushes() {if (IBSTATS) pushes++;}
	Real inline getPushes() {return pushes;}
	void inline incOrphanArcs1() {if (IBSTATS) orphanArcs1++;}
	Real inline getOrphanArcs1() {return orphanArcs1;}
	void inline incOrphanArcs2() {if (IBSTATS) orphanArcs2++;}
	Real inline getOrphanArcs2() {return orphanArcs2;}
	void inline incOrphanArcs3() {if (IBSTATS) orphanArcs3++;}
	Real inline getOrphanArcs3() {return orphanArcs3;}
	void inline addAugLen(Real len) {
		if (IBSTATS) {
			if (len > augLenMax) augLenMax = len;
			if (len < augLenMin) augLenMin = len;
		}
	}
	Real inline getAugLenMin() {return augLenMin;}
	Real inline getAugLenMax() {return augLenMax;}

private:
	Real augs;
	Real growthS;
	Real growthT;
	Real orphans;
	Real growthArcs;
	Real pushes;
	Real orphanArcs1;
	Real orphanArcs2;
	Real orphanArcs3;
	Real augLenMin;
	Real augLenMax;
};



class IBFSGraph
{
public:
	IBFSGraph();
	~IBFSGraph();
	void setVerbose(bool a_verbose) {
		verbose = a_verbose;
	}

	void initSize(int numNodes, int numEdges);
	void addEdge(int nodeIndexFrom, int nodeIndexTo, EdgeCap capacity, EdgeCap reverseCapacity);
	void addNode(int nodeIndex, EdgeCap capacityFromSource, EdgeCap capacityToSink);

	void setCompactSlowInitMode(bool a_compactSlowInitMode) {
		compactSlowInitMode = a_compactSlowInitMode;
	}

	void finalizeGraph()
	{
#pragma omp parallel for schedule(static)
		for (int i = 0; i < numNodes; ++i) {
			nodes[i].arcCount = static_cast<uint8_t>(
				arcCountBuild[i].load(std::memory_order_relaxed));
		}

		// Optional but documents the phase boundary
		std::atomic_thread_fence(std::memory_order_acquire);

#if IBFS_OPT_ARC_SORT
		// Per-node arc sort by headIdx. Each node has ≤4 arcs; selection sort
		// in registers. Carries the entire Arc struct (incl. stale revIdx).
		// Pass A: sort arcs locally, record old→new permutation packed into a
		//         single byte (4 × 2-bit fields).
		// Pass B: rewrite each arc's revIdx by looking up the *peer* node's
		//         permutation table — race-free because each thread only
		//         writes to its own node's arcs in this pass.
		std::unique_ptr<uint8_t[]> invP(new uint8_t[numNodes]);

		// Identity packing 0xE4 = 11_10_01_00 maps slot i → i for all i ∈ {0..3}.
		const uint8_t kIdentity = 0xE4u;

#pragma omp parallel for schedule(static)
		for (int i = 0; i < numNodes; ++i) {
			Node& u = nodes[i];
			const int cnt = u.arcCount;
			if (cnt <= 1) { invP[i] = kIdentity; continue; }

			// permutation indices into u.arcs[]; selection sort by headIdx.
			uint8_t p[4] = { 0, 1, 2, 3 };
			for (int a = 0; a < cnt - 1; ++a) {
				int best = a;
				NodeIdx bestKey = u.arcs[p[a]].headIdx();
				for (int b = a + 1; b < cnt; ++b) {
					const NodeIdx kHead = u.arcs[p[b]].headIdx();
					if (kHead < bestKey) { best = b; bestKey = kHead; }
				}
				if (best != a) { uint8_t t = p[a]; p[a] = p[best]; p[best] = t; }
			}

			// reorder Arc structs in place via a stack copy.
			Arc tmp[4];
			for (int a = 0; a < cnt; ++a) tmp[a] = u.arcs[p[a]];
			for (int a = 0; a < cnt; ++a) u.arcs[a] = tmp[a];

			// Also permute residBits to follow the new arc order.
			{
				const uint8_t oldBits = u.residBits;
				uint8_t newBits = 0;
				for (int a = 0; a < cnt; ++a)
					newBits |= (uint8_t)(((oldBits >> p[a]) & 1u) << a);
				u.residBits = newBits;
			}

			// pack: invMap[oldSlot] = newSlot. Initialize as identity, then
			// overwrite the active slots so unused-arc slots remain identity.
			uint8_t pack = kIdentity;
			for (int newSlot = 0; newSlot < cnt; ++newSlot) {
				const int oldSlot = p[newSlot];
				pack &= ~(uint8_t)(0x3u << (oldSlot * 2));
				pack |=  (uint8_t)((newSlot & 0x3) << (oldSlot * 2));
			}
			invP[i] = pack;
		}

		// Pass B: rewrite each arc.revIdx via peer's permutation table.
#pragma omp parallel for schedule(static)
		for (int i = 0; i < numNodes; ++i) {
			Node& u = nodes[i];
			const int cnt = u.arcCount;
			for (int a = 0; a < cnt; ++a) {
				Arc& arc = u.arcs[a];
				const uint8_t pkPeer = invP[arc.headIdx()];
				const int oldRev = arc.revIdx();
				const int newRev = (pkPeer >> (oldRev * 2)) & 0x3;
				arc.setRev(static_cast<uint8_t>(newRev));
			}
		}
#endif // IBFS_OPT_ARC_SORT
	}

	// Variant of finalizeGraph for callers that have already written every
	// node's arcs[] directly and set arcCount themselves (bypassing addEdge
	// and arcCountBuild). Skips the atomic-counter copy, runs only the
	// optional arc-sort pass. Used by the OpenMVS Delaunay graph build,
	// where every cell has exactly 4 neighbors and slot assignment is
	// deterministic from the neighbor index — no atomics required.
	void finalizeGraphPrebuilt()
	{
		std::atomic_thread_fence(std::memory_order_acquire);

#if IBFS_OPT_ARC_SORT
		std::unique_ptr<uint8_t[]> invP(new uint8_t[numNodes]);
		const uint8_t kIdentity = 0xE4u;

#pragma omp parallel for schedule(static)
		for (int i = 0; i < numNodes; ++i) {
			Node& u = nodes[i];
			const int cnt = u.arcCount;
			if (cnt <= 1) { invP[i] = kIdentity; continue; }

			uint8_t p[4] = { 0, 1, 2, 3 };
			for (int a = 0; a < cnt - 1; ++a) {
				int best = a;
				NodeIdx bestKey = u.arcs[p[a]].headIdx();
				for (int b = a + 1; b < cnt; ++b) {
					const NodeIdx kHead = u.arcs[p[b]].headIdx();
					if (kHead < bestKey) { best = b; bestKey = kHead; }
				}
				if (best != a) { uint8_t t = p[a]; p[a] = p[best]; p[best] = t; }
			}

			Arc tmp[4];
			for (int a = 0; a < cnt; ++a) tmp[a] = u.arcs[p[a]];
			for (int a = 0; a < cnt; ++a) u.arcs[a] = tmp[a];

			// Permute residBits to follow the new arc order.
			{
				const uint8_t oldBits = u.residBits;
				uint8_t newBits = 0;
				for (int a = 0; a < cnt; ++a)
					newBits |= (uint8_t)(((oldBits >> p[a]) & 1u) << a);
				u.residBits = newBits;
			}

			uint8_t pack = kIdentity;
			for (int newSlot = 0; newSlot < cnt; ++newSlot) {
				const int oldSlot = p[newSlot];
				pack &= ~(uint8_t)(0x3u << (oldSlot * 2));
				pack |=  (uint8_t)((newSlot & 0x3) << (oldSlot * 2));
			}
			invP[i] = pack;
		}

#pragma omp parallel for schedule(static)
		for (int i = 0; i < numNodes; ++i) {
			Node& u = nodes[i];
			const int cnt = u.arcCount;
			for (int a = 0; a < cnt; ++a) {
				Arc& arc = u.arcs[a];
				const uint8_t pkPeer = invP[arc.headIdx()];
				const int oldRev = arc.revIdx();
				const int newRev = (pkPeer >> (oldRev * 2)) & 0x3;
				arc.setRev(static_cast<uint8_t>(newRev));
			}
		}
#endif // IBFS_OPT_ARC_SORT
	}

	void initGraph();
	EdgeCap computeMaxFlow();

	inline IBFSStats getStats() {
		return stats;
	}
	inline EdgeCap getFlow() {
		return flow;
	}

	bool isNodeOnSrcSide(int nodeIndex) const;

	struct Node;

#if 0 // BLock quantize
	struct Arc {
		Node* head;
		Arc* rev;
		uint16_t	rCap;
		unsigned char	isRevResidual;
	};

	struct Node {
		static constexpr int kMaxArcs = 4;

		// Group together arcCount + arcs for locality
		std::atomic<int> arcCount;               // 4
		Arc arcs[kMaxArcs];         // 32 (assuming Arc = 8 bytes)

		EdgeCap excess;             // 4
		Arc* parent;                // 8

		Node* firstSon;             // 8
		Node* nextPtr;              // 8

		int lastAugTimestamp : 31;  // 4 (bitfield with next)
		int isParentCurr : 1;

		int label;                  // 4
	};

	__forceinline uint16_t EncodeCap(float cap, float invScale) noexcept {
		float x = std::min(std::max(cap * invScale, 0.0f), 65535.0f);
		return static_cast<uint16_t>(x + 0.5f);
	}
	__forceinline float DecodeCap(uint16_t q, float scale) noexcept {
		return scale * static_cast<float>(q);
	}
#else
	struct Arc {
		// Packed: [31..30]=revIdx (2 bits), [29..0]=headIdx (30 bits).
		// isRevResidual lives in the owning Node's residBits field (same
		// cache line), freeing the full 30-bit range for headIdx.
		// 8 B total (was 12), so Node arcs[4] = 32 B; Node fits in one
		// 64-byte cache line.
		uint32_t pack;           // 4
		EdgeCap  rCap;           // 4

		static constexpr uint32_t kHeadMask  = (1u << 30) - 1u;       // [29..0]
		static constexpr uint32_t kRevShift  = 30u;
		static constexpr uint32_t kRevMask   = 0x3u << kRevShift;

		__forceinline NodeIdx  headIdx()       const { return pack & kHeadMask; }
		__forceinline uint8_t  revIdx()        const { return (uint8_t)(pack >> kRevShift); }

		__forceinline void setHead (NodeIdx h)  { pack = (pack & ~kHeadMask)  | (h & kHeadMask); }
		__forceinline void setRev  (uint8_t r)  { pack = (pack & ~kRevMask)   | ((uint32_t)(r & 0x3) << kRevShift); }

		// Build-time single-shot init (no resid — that's on the Node now).
		__forceinline void initFields(NodeIdx h, uint8_t r) {
			pack = (h & kHeadMask)
			     | ((uint32_t)(r & 0x3) << kRevShift);
		}
	};                           // = 8 B
	static_assert(sizeof(Arc) == 8, "Arc must be 8 bytes after compaction");

	struct Node {
		static constexpr int kMaxArcs = 4;

		// ---- hot path ----
		EdgeCap   excess;            // 4
		int       label;             // 4

		ParentRef parentRef;         // 4   (was Arc* parent)
		NodeIdx   firstSonIdx;       // 4   (was Node* firstSon)
		NodeIdx   nextPtrIdx;        // 4   (was Node* nextPtr)

		uint32_t  lastAugTimestamp;  // 4
		uint8_t   isParentCurr;      // 1
		uint8_t   arcCount;          // 1
		uint8_t   residBits;         // 1  bits [0..3] = isRevResidual for arcs[0..3]
		uint8_t   pad0;              // 1  -> header: 28 B

		// ---- arc data (hot but secondary) ----
		Arc       arcs[kMaxArcs];    // 4 * 8 = 32 B  -> 60 B

		// Pad to 64 B = one cache line per Node.
		uint8_t   pad1[4];
	};

	static_assert(Node::kMaxArcs <= 4,
		"ParentRef encodes the arc slot in 2 bits; kMaxArcs must be <= 4");

	std::atomic<int>* arcCountBuild;  // allocated as separate array in initSize
#endif

	class ActiveList
	{
	public:
		inline ActiveList() { len = 0; }
		inline void init(int /*numNodes*/) {
			list_vec.reserve(1 << 20);
			len = 0;
		}
		inline void release() { std::vector<Node*>().swap(list_vec); }
		inline void clear() { len = 0; }
		inline void add(Node* x) {
			if ((size_t)len < list_vec.size())
				list_vec[len] = x;
			else
				list_vec.push_back(x);
			len++;
		}
		inline static void swapLists(ActiveList *a, ActiveList *b) {
			std::swap(a->list_vec, b->list_vec);
			std::swap(a->len, b->len);
		}
		std::vector<Node*> list_vec;
		int len;
	};

	class Buckets
	{
	public:
		inline Buckets() { maxBucket = 0; nodes = NULL; numNodesTotal = 0; }
		inline void init(Node *a_nodes, int numNodes) {
			nodes = a_nodes;
			numNodesTotal = numNodes;
			maxBucket = 0;
			// prevPtrs is indexed by node offset (x - nodes), so it must
			// cover all nodes, not just the number of bucket levels.
			prevPtrs.assign(numNodes, kNullIdx);
		}
		inline void ensureSize(int bucket) {
			if (bucket >= (int)buckets.size()) {
				int newSize = std::max(bucket + 1, (int)buckets.size() * 2);
				newSize = std::min(newSize, numNodesTotal);
				buckets.resize(newSize, kNullIdx);
			}
		}
		inline void release() {
			std::vector<NodeIdx>().swap(buckets);
			std::vector<NodeIdx>().swap(prevPtrs);
		}
		template <bool sTree> inline void add(Node* x) {
			int bucket = (sTree ? (x->label) : (-x->label));
			ensureSize(bucket);
			NodeIdx headI = buckets[bucket];
			NodeIdx xi    = static_cast<NodeIdx>(x - nodes);
			if (headI == kNullIdx || headI == kOrphansEndIdx) {
				x->nextPtrIdx = kOrphansEndIdx;
			} else {
				x->nextPtrIdx = headI;
				prevPtrs[headI] = xi;
			}
			buckets[bucket] = xi;
			if (bucket > maxBucket) maxBucket = bucket;
		}
		inline Node* popFront(int bucket) {
			if (bucket >= (int)buckets.size()) return NULL;
			NodeIdx i = buckets[bucket];
			if (i == kNullIdx || i == kOrphansEndIdx) return NULL;
			Node *x = nodes + i;
			buckets[bucket] = x->nextPtrIdx;
			return x;
		}
		template <bool sTree> inline void remove(Node *x) {
			int bucket = (sTree ? (x->label) : (-x->label));
			if (bucket >= (int)buckets.size()) return;
			NodeIdx xi = static_cast<NodeIdx>(x - nodes);
			if (buckets[bucket] == xi) {
				buckets[bucket] = x->nextPtrIdx;
			} else {
				NodeIdx prev = prevPtrs[xi];
				nodes[prev].nextPtrIdx = x->nextPtrIdx;
				if (x->nextPtrIdx != kOrphansEndIdx)
					prevPtrs[x->nextPtrIdx] = prev;
			}
		}

		std::vector<NodeIdx> buckets;
		std::vector<NodeIdx> prevPtrs;
		Node *nodes;
		int numNodesTotal;
		int maxBucket;
	};

	// members
	IBFSStats stats;
  Node* nodes;
  Node* nodeEnd;
	int 	numNodes;
	EdgeCap	flow;

	unsigned short augTimestamp;
	unsigned int uniqOrphansS, uniqOrphansT;
	NodeIdx orphanFirstIdx;
	NodeIdx orphanLastIdx;
	int topLevelS, topLevelT;

	ActiveList active0, activeS1, activeT1;
	Buckets orphanBuckets;
	bool verbose;

	// ---- index helpers ----
	inline NodeIdx idxOf(const Node* n) const noexcept {
		return static_cast<NodeIdx>(n - nodes);
	}
	inline ParentRef packParent(const Arc* a, const Node* owner) const noexcept {
		if (a == nullptr) return kNullParent;
		uint32_t slot = static_cast<uint32_t>(a - owner->arcs);
		return (idxOf(owner) << 2) | slot;
	}
	inline Arc* unpackParent(ParentRef r) const noexcept {
		if (r == kNullParent) return nullptr;
		return &nodes[r >> 2].arcs[r & 0x3u];
	}
	inline Node* parentOwner(ParentRef r) const noexcept {
		return (r == kNullParent) ? nullptr : &nodes[r >> 2];
	}

	void augment(Arc* __restrict bridge);

	struct NodeArcPair
	{
		NodeArcPair() {}
		NodeArcPair(Node* _n, Arc* _a) : n(_n), a(_a) {}
		Node* n;
		Arc* a;
	};

	template <bool sTree> void augmentTree(
		EdgeCap bottleneck,
		Node** __restrict nodePath,
		Arc** __restrict arcPath,
		int nodeCount
	);
	template <bool sTree> void adoption();
	template <bool sTree> void adoption3Pass();
	template <bool dirS> void growth();

	#if IBIO>0
	bool readFromFile(char *filename);
	bool readFromFileCompile(char *filename);
	bool readFromFile(char *filename, bool checkCompile);
	bool readCompiled(FILE *pFile);
	#endif

	//
	// Initialization
	//
	struct TmpEdge
	{
		Node*		head;
		Node*		tail;
		EdgeCap		cap;
		EdgeCap		revCap;
	};
	struct TmpArc
	{
		TmpArc		*rev;
		EdgeCap		cap;
	};
	char	*memArcs;
	TmpEdge	*tmpEdges, *tmpEdgeLast;
	TmpArc	*tmpArcs;
	bool compactSlowInitMode;
	void initGraphFast();
	void initGraphCompact();

	//
	// Testing
	//
	void testTree();
	void testExit() {
		exit(1);
	}
	inline void testNode(Node *x) {
		if (IBTEST && x-nodes == -1) {
			IBDEBUG("*");
		}
	}
};


inline void IBFSGraph::addNode(int nodeIndex, EdgeCap capFromSource, EdgeCap capToSink)
{
	EdgeCap f = nodes[nodeIndex].excess;
	if (f > 0) capFromSource += f;
	else capToSink -= f;
	flow += (capFromSource < capToSink ? capFromSource : capToSink);
	nodes[nodeIndex].excess = capFromSource - capToSink;
}

inline void IBFSGraph::addEdge(int from, int to, EdgeCap cap, EdgeCap revCap) {
	Node* __restrict u = &nodes[from];
	Node* __restrict v = &nodes[to];

#if 1 // JPB WIP Faster than serial code by 25%
	// Atomically get the next arc index for each node
	int uArcIdx = arcCountBuild[from].fetch_add(1, std::memory_order_relaxed);
	int vArcIdx = arcCountBuild[to].fetch_add(1, std::memory_order_relaxed);

	Arc* __restrict uv = &u->arcs[uArcIdx];
	Arc* __restrict vu = &v->arcs[vArcIdx];
#else
	Arc* uv = &u->arcs[u->arcCount++];
	Arc* vu = &v->arcs[v->arcCount++];
#endif

	// forward arc
	uv->initFields(static_cast<NodeIdx>(to), static_cast<uint8_t>(vArcIdx));
	uv->rCap = cap;
	if (revCap > 0) u->residBits |= (1u << uArcIdx);

	// reverse arc
	vu->initFields(static_cast<NodeIdx>(from), static_cast<uint8_t>(uArcIdx));
	vu->rCap = revCap;
	if (cap > 0) v->residBits |= (1u << vArcIdx);
}

inline bool IBFSGraph::isNodeOnSrcSide(int nodeIndex) const
{
	const Node& x = nodes[nodeIndex];
	if (x.label == numNodes || x.label == 0) return activeT1.len == 0;
	return x.label > 0;
}

} // namespace IBFS

#endif
