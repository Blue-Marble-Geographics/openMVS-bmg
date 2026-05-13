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

*/

#include "Common.h"
#include "IBFS.h"
#include "boost/container/small_vector.hpp"

// Toggle perf optimizations for A/B test:
//   IBFS_OPT_PREFETCH : software prefetch in growth / adoption / augment / augmentTree
// Set to 0 to disable.
#ifndef IBFS_OPT_PREFETCH
#define IBFS_OPT_PREFETCH 1
#endif

using namespace IBFS;

//
// Orphan handling (index-based)
//
#define ADD_ORPHAN_BACK(n) do {                              \
	NodeIdx _ni = static_cast<NodeIdx>((n) - nodes);         \
	if (orphanFirstIdx != kOrphansEndIdx) {                  \
		nodes[orphanLastIdx].nextPtrIdx = _ni;               \
		orphanLastIdx = _ni;                                 \
	} else {                                                 \
		orphanFirstIdx = orphanLastIdx = _ni;                \
	}                                                        \
	(n)->nextPtrIdx = kOrphansEndIdx;                        \
} while (0)

#define ADD_ORPHAN_FRONT(n) do {                             \
	NodeIdx _ni = static_cast<NodeIdx>((n) - nodes);         \
	if (orphanFirstIdx == kOrphansEndIdx) {                  \
		(n)->nextPtrIdx = kOrphansEndIdx;                    \
		orphanFirstIdx = orphanLastIdx = _ni;                \
	} else {                                                 \
		(n)->nextPtrIdx = orphanFirstIdx;                    \
		orphanFirstIdx = _ni;                                \
	}                                                        \
} while (0)




IBFSGraph::IBFSGraph() {
	numNodes = 0;
	uniqOrphansS = uniqOrphansT = 0;
	augTimestamp = 0;
	verbose = IBTEST;
	flow = 0;
  orphanFirstIdx = orphanLastIdx = kOrphansEndIdx;
  nodes = nodeEnd = nullptr;
  arcCountBuild = nullptr;
}

IBFSGraph::~IBFSGraph() {
	delete[] arcCountBuild;
	active0.release();
	activeS1.release();
	activeT1.release();
	orphanBuckets.release();
	_aligned_free(nodes);
	nodes = nullptr;
}


void IBFSGraph::initGraph() {
	// Must be after nodes are added.
  for (Node* x = nodes; x < nodeEnd; ++x) {
		if (x->excess == 0) {
			x->label = numNodes;
			continue;
		}
		if (x->excess > 0) {
			x->label = 1;
			activeS1.add(x);
		} else {
			x->label = -1;
			activeT1.add(x);
		}
	}
  topLevelS = topLevelT = 1;
}


void IBFSGraph::initSize(int n, int)
{
	assert(n >= 0 && static_cast<uint32_t>(n) <= kMaxNodes);
	numNodes = n;
	nodes = static_cast<Node*>(_aligned_malloc(sizeof(Node) * static_cast<size_t>(n), 64));
	arcCountBuild = new std::atomic<int>[n];

#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; ++i) {
		arcCountBuild[i].store(0, std::memory_order_relaxed);
		Node& node = nodes[i];
		node.excess = 0;
		node.parentRef = kNullParent;
		node.firstSonIdx = kNullIdx;
		node.nextPtrIdx = kNullIdx;
		node.lastAugTimestamp = 0;
		node.isParentCurr = 0;
		node.arcCount = 0;
		node.residBits = 0;
		node.label = 0;
	}

	nodeEnd = nodes + n;
	active0.init(n);
	activeS1.init(n);
	activeT1.init(n);
	orphanBuckets.init(nodes, n);
	flow = 0;
}

template <bool sTree>
__forceinline void IBFSGraph::augmentTree(
	EdgeCap bottleneck,
	Node** __restrict nodePath,
	Arc** __restrict arcPath,
	int nodeCount
) {
	Node* x = nodePath[0];

	for (int i = 0; i < nodeCount - 1; ++i) {
		Arc* a = arcPath[i];
#if IBFS_OPT_PREFETCH
		// Prefetch the next iteration's node and arc while we mutate this one.
		if (i + 1 < nodeCount - 1) {
			_mm_prefetch((const char*)nodePath[i + 1], _MM_HINT_T0);
			_mm_prefetch((const char*)arcPath[i + 1], _MM_HINT_T0);
		}
#endif
		Node* parentNode = nodes + a->headIdx();
		Arc& rev = parentNode->arcs[a->revIdx()];

		if (sTree) {
			// used reverse residual
			a->rCap += bottleneck;
			parentNode->residBits |= (1u << a->revIdx());
			rev.rCap -= bottleneck;
		}
		else {
			// used forward residual
			rev.rCap += bottleneck;
			x->residBits |= (1u << (int)(a - x->arcs));
			a->rCap -= bottleneck;
		}

		if ((sTree ? rev.rCap : a->rCap) == 0) {
			if (sTree) x->residBits &= ~(1u << (int)(a - x->arcs));
			else parentNode->residBits &= ~(1u << a->revIdx());

			NodeIdx xi = static_cast<NodeIdx>(x - nodes);
			NodeIdx yi = parentNode->firstSonIdx;
			if (yi == xi) {
				parentNode->firstSonIdx = x->nextPtrIdx;
			}
			else {
				while (yi != kNullIdx && yi != kOrphansEndIdx
					&& nodes[yi].nextPtrIdx != xi) {
					yi = nodes[yi].nextPtrIdx;
				}
				if (yi != kNullIdx && yi != kOrphansEndIdx)
					nodes[yi].nextPtrIdx = x->nextPtrIdx;
			}

			// orphan creation
			ADD_ORPHAN_BACK(x);
		}
		x = nodePath[i + 1];
	}

	// terminal
	x->excess += (sTree ? -bottleneck : bottleneck);
	if (x->excess == 0) {
		ADD_ORPHAN_BACK(x);
	}
}

enum { kSmallSize = 1024 };

struct Scratch
{
	IBFSGraph::Node* smallSNode[kSmallSize];
	IBFSGraph::Arc* smallSArc[kSmallSize];

	IBFSGraph::Node* smallTNode[kSmallSize];
	IBFSGraph::Arc* smallTArc[kSmallSize];
};

static std::vector<IBFSGraph::Node*> bigSNode;
static std::vector<IBFSGraph::Arc*> bigSArc;

static std::vector<IBFSGraph::Node*> bigTNode;
static std::vector<IBFSGraph::Arc*> bigTArc;

static Scratch scratch;

void IBFSGraph::augment(Arc * __restrict bridge)
{
	Node* x;
	Arc* a;
	EdgeCap bottleneck;
	Real pushesBefore;

	// stats
	if (IBSTATS) pushesBefore=stats.getPushes();
	stats.incAugs();
	stats.incPushes();

	Node** __restrict sNode = scratch.smallSNode;
	Arc** __restrict sArc = scratch.smallSArc;

	// bottleneck in S
	bottleneck = bridge->rCap;
	int lenS = 0;
	bool overflow = false;

	{
		Node* bridgeHead = nodes + bridge->headIdx();
		x = nodes + bridgeHead->arcs[bridge->revIdx()].headIdx();
	}
	for (;;) {
		// ---- overflow check once per full iteration ----
		if (lenS + 1 >= kSmallSize) {
			overflow = true;
			break;
		}

		// ================== step 0 ==================
		sNode[lenS] = x;

		if (x->excess) {
			++lenS;
			break;
		}

		a = unpackParent(x->parentRef);
		sArc[lenS] = a;

		Node* aHead0 = nodes + a->headIdx();
#if IBFS_OPT_PREFETCH
		// Prefetch the rev-arc cache line on parent (if arcs[3], it's a 2nd line).
		_mm_prefetch((const char*)aHead0, _MM_HINT_T0);
#endif

		{
			const Arc& rev0 = aHead0->arcs[a->revIdx()];
			if (bottleneck > rev0.rCap)
				bottleneck = rev0.rCap;
		}

		++lenS;
		x = aHead0;

		// ================== step 1 ==================
		sNode[lenS] = x;

		if (x->excess) {
			++lenS;
			break;
		}

		a = unpackParent(x->parentRef);
		sArc[lenS] = a;

		Node* aHead1 = nodes + a->headIdx();
#if IBFS_OPT_PREFETCH
		_mm_prefetch((const char*)aHead1, _MM_HINT_T0);
#endif

		{
			const Arc& rev1 = aHead1->arcs[a->revIdx()];
			if (bottleneck > rev1.rCap)
				bottleneck = rev1.rCap;
		}

		++lenS;
		x = aHead1;
	}

	if (!overflow && bottleneck > x->excess)
		bottleneck = x->excess;


	Node** __restrict tNode = scratch.smallTNode;
	Arc** __restrict tArc = scratch.smallTArc;

	int lenT = 0;

	for (x = nodes + bridge->headIdx();;) {
		// ---- overflow check once per full iteration ----
		if (lenT + 1 >= kSmallSize) {
			overflow = true;
			break;
		}

		// ================== step 0 ==================
		tNode[lenT] = x;

		if (x->excess) {
			++lenT;
			break;
		}

		a = unpackParent(x->parentRef);
		tArc[lenT] = a;

		if (bottleneck > a->rCap)
			bottleneck = a->rCap;

		++lenT;
#if IBFS_OPT_PREFETCH
		// Prefetch the next parent node before we land on it.
		_mm_prefetch((const char*)(nodes + a->headIdx()), _MM_HINT_T0);
#endif
		x = nodes + a->headIdx();

		// ================== step 1 ==================
		tNode[lenT] = x;

		if (x->excess) {
			++lenT;
			break;
		}

		a = unpackParent(x->parentRef);
		tArc[lenT] = a;

		if (bottleneck > a->rCap)
			bottleneck = a->rCap;

		++lenT;
#if IBFS_OPT_PREFETCH
		_mm_prefetch((const char*)(nodes + a->headIdx()), _MM_HINT_T0);
#endif
		x = nodes + a->headIdx();
	}

	if (!overflow && bottleneck > (-x->excess))
		bottleneck = (-x->excess);

	if (overflow) {
		bottleneck = bridge->rCap;

		bigSNode.clear();
		bigSNode.reserve(1024);
		bigSArc.clear();
		bigSArc.reserve(1024);

		{
			Node* bridgeHead = nodes + bridge->headIdx();
			x = nodes + bridgeHead->arcs[bridge->revIdx()].headIdx();
		}
		for (;; x = nodes + a->headIdx()) {
			bigSNode.push_back(x);

			if (x->excess)
				break;

			a = unpackParent(x->parentRef);
			bigSArc.push_back(a);

			Arc& rev = nodes[a->headIdx()].arcs[a->revIdx()];
			if (bottleneck > rev.rCap)
				bottleneck = rev.rCap;
		}

		if (bottleneck > x->excess)
			bottleneck = x->excess;

		bigTNode.clear();
		bigTNode.reserve(1024);
		bigTArc.clear();
		bigTArc.reserve(1024);

		for (x = nodes + bridge->headIdx(); ; x = nodes + a->headIdx()) {
			bigTNode.push_back(x);
			if (x->excess) break;
			a = unpackParent(x->parentRef);
			bigTArc.push_back(a);
			if (bottleneck > a->rCap)
				bottleneck = a->rCap;
		}

		if (bottleneck > (-x->excess))
			bottleneck = (-x->excess);
	}
#if 0
	bottleneck = bridge->rCap;
	for (x=bridge->rev->head; ; x=a->head)
	{
		sScratch.push_back(x);
		stats.incPushes();
		if (x->excess) break;
		a = x->parent;
		if (bottleneck > a->rev->rCap) {
			bottleneck = a->rev->rCap;
		}
	}
	if (bottleneck > x->excess) {
		bottleneck = x->excess;
	}

	// bottleneck in T
	for (x=bridge->head; ; x=a->head)
	{
		tScratch.push_back(x);
		stats.incPushes();
		if (x->excess) break;
		a = x->parent;
		if (bottleneck > a->rCap) {
			bottleneck = a->rCap;
		}
	}
	if (bottleneck > (-x->excess)) {
		bottleneck = (-x->excess);
	}
#endif

	// stats
	if (IBSTATS) {
		Real augLen = stats.getPushes() - pushesBefore;
		stats.addAugLen(augLen);
	}

	// augment connecting arc
	Node& bridgeTgt = nodes[bridge->headIdx()];
	Arc& rev = bridgeTgt.arcs[bridge->revIdx()];
	rev.rCap += bottleneck;
	nodes[rev.headIdx()].residBits |= (1u << rev.revIdx());
	bridge->rCap -= bottleneck;
	if (bridge->rCap == 0) {
		bridgeTgt.residBits &= ~(1u << bridge->revIdx());
	}


	// augment T
	augTimestamp++;
	augmentTree<false>(
		bottleneck,
		overflow ? bigTNode.data() : tNode,
		overflow ? bigTArc.data() : tArc,
		overflow ? (int)bigTNode.size() : lenT
	);
	adoption<false>();

	// augment S
	augTimestamp++;
	augmentTree<true>(
		bottleneck,
		overflow ? bigSNode.data() : sNode,
		overflow ? bigSArc.data() : sArc,
		overflow ? (int)bigSNode.size() : lenS
	);
	adoption<true>();

	flow += bottleneck;
}


template <bool sTree>
void IBFSGraph::adoption()
{
	Node* x, * y;
	Arc*  a;
	bool threePass = false;
	int minLabel, numOrphans = 0, numOrphansUniq = 0;

	while (orphanFirstIdx != kOrphansEndIdx) {
		x = nodes + orphanFirstIdx;
		orphanFirstIdx = x->nextPtrIdx;
		testNode(x);
		stats.incOrphans();
		numOrphans++;

		if (x->lastAugTimestamp != augTimestamp) {
			x->lastAugTimestamp = augTimestamp;
			if (sTree) uniqOrphansS++;
			else uniqOrphansT++;
			numOrphansUniq++;
		}
		if (numOrphans >= 3 * numOrphansUniq) {
			threePass = true;
		}

		// check for same-level parent
		if (x->isParentCurr) {
			a = unpackParent(x->parentRef);
		}
		else {
			x->isParentCurr = 1;
			a = nullptr;
		}
		x->parentRef = kNullParent;

		if (x->label != (sTree ? 1 : -1)) {
			minLabel = x->label - (sTree ? 1 : -1);
			const int cnt = x->arcCount;
			__assume(cnt <= 4);
			const uint8_t resBits = x->residBits;
			for (int i = 0; i < cnt; ++i) {
				a = &x->arcs[i];
				stats.incOrphanArcs1();
				y = nodes + a->headIdx();

#if IBFS_OPT_PREFETCH
				// prefetch next arc's target node
				if (i + 1 < cnt)
					_mm_prefetch((const char*)(nodes + x->arcs[i + 1].headIdx()), _MM_HINT_T0);
#endif

				if ((sTree ? ((resBits >> i) & 1u) : a->rCap) && y->label == minLabel) {
					x->parentRef = packParent(a, x);
					x->nextPtrIdx = y->firstSonIdx;
					y->firstSonIdx = idxOf(x);
					break;
				}
			}
		}
		if (x->parentRef != kNullParent) continue;

		// orphan children
		{
			NodeIdx yi = x->firstSonIdx;
			while (yi != kNullIdx && yi != kOrphansEndIdx) {
				y = nodes + yi;
				stats.incOrphanArcs3();
				NodeIdx zi = y->nextPtrIdx;
				ADD_ORPHAN_BACK(y);
				yi = zi;
			}
		}
		x->firstSonIdx = kNullIdx;

		if (x->label == (sTree ? topLevelS : -topLevelT)) {
			x->label = numNodes;
			continue;
		}

		if (threePass) {
			x->label += (sTree ? 1 : -1);
			orphanBuckets.add<sTree>(x);
			continue;
		}

		// relabel
		minLabel = (sTree ? topLevelS : -topLevelT);
		if (x->label != minLabel) {
			const int cnt = x->arcCount;
			__assume(cnt <= 4);
			const uint8_t resBits = x->residBits;
			for (int i = 0; i < cnt; ++i) {
				a = &x->arcs[i];
				stats.incOrphanArcs2();
				y = nodes + a->headIdx();


#if IBFS_OPT_PREFETCH
				// prefetch next arc's target node
				if (i + 1 < cnt)
					_mm_prefetch((const char*)(nodes + x->arcs[i + 1].headIdx()), _MM_HINT_T0);
#endif

				if ((sTree ? ((resBits >> i) & 1u) : a->rCap) &&
					(sTree ? y->label > 0 : y->label < 0) &&
					(sTree ? y->label < minLabel : y->label > minLabel)) {
					minLabel = y->label;
					x->parentRef = packParent(a, x);
					if (minLabel == x->label) break;
				}
			}
		}

		if (x->parentRef != kNullParent) {
			x->label = minLabel + (sTree ? 1 : -1);
			Node* po = nodes + unpackParent(x->parentRef)->headIdx();
			x->nextPtrIdx = po->firstSonIdx;
			po->firstSonIdx = idxOf(x);
			if (sTree && x->label == topLevelS) activeS1.add(x);
			else if (!sTree && x->label == -topLevelT) activeT1.add(x);
		}
		else {
			x->label = numNodes;
		}
	}

	if (threePass) {
		adoption3Pass<sTree>();
	}
}

template <bool sTree>
void IBFSGraph::adoption3Pass()
{
	Arc* a;
	Node* x, * y;
	int minLabel, destLabel;

	for (int level = 2; level <= orphanBuckets.maxBucket; level++) {
		while ((x = orphanBuckets.popFront(level)) != nullptr) {
			testNode(x);

			// pass 2: find lowest level parent
			if (x->parentRef == kNullParent) {
				minLabel = (sTree ? topLevelS : -topLevelT);
				destLabel = x->label - (sTree ? 1 : -1);

				const int cnt = x->arcCount;
				__assume(cnt <= 4);
				const uint8_t resBits = x->residBits;
				for (int i = 0; i < cnt; ++i) {
					a = &x->arcs[i];
					y = nodes + a->headIdx();
					if ((sTree ? ((resBits >> i) & 1u) : a->rCap) &&
						(y->excess || y->parentRef != kNullParent) &&
						//!y->isOrphan() &&
						(sTree ? (y->label > 0) : (y->label < 0)) &&
						(sTree ? (y->label < minLabel) : (y->label > minLabel)))
					{
						x->parentRef = packParent(a, x);
						if ((minLabel = y->label) == destLabel) break;
					}
				}

				if (x->parentRef == kNullParent) {
					x->label = numNodes;
					continue;
				}

				x->label = minLabel + (sTree ? 1 : -1);
				if (x->label != (sTree ? level : -level)) {
					orphanBuckets.add<sTree>(x);
					continue;
				}
			}

			// pass 3: lower potential sons and/or find first parent
			if (x->label != (sTree ? topLevelS : -topLevelT)) {
				minLabel = x->label + (sTree ? 1 : -1);

				const int cnt = x->arcCount;
				__assume(cnt <= 4);
				const uint8_t resBits = x->residBits;
				for (int i = 0; i < cnt; ++i) {
					a = &x->arcs[i];
					y = nodes + a->headIdx();

					Arc& rev = y->arcs[a->revIdx()];
					if ((sTree ? a->rCap : ((resBits >> i) & 1u)) &&
							((!sTree && y->label == numNodes) ||
								// the above implicitly holds by condition below when sTree=true
								(sTree ? (minLabel < y->label) : (minLabel > y->label))))
						{
						if (y->label != numNodes)
							orphanBuckets.remove<sTree>(y);

						y->label = minLabel;
						y->parentRef = packParent(&rev, y);
						orphanBuckets.add<sTree>(y);
					}
				}
			}

			// relabel onto new parent
			{
				Node* po = nodes + unpackParent(x->parentRef)->headIdx();
				x->nextPtrIdx = po->firstSonIdx;
				po->firstSonIdx = idxOf(x);
			}
			x->isParentCurr = 0;

			// add to active list of the next growth phase
			if (sTree) {
				if (x->label == topLevelS)
					activeS1.add(x);
			}
			else {
				if (x->label == -topLevelT)
					activeT1.add(x);
			}
		}
	}

	orphanBuckets.maxBucket = 0;
}



template <bool dirS>
void IBFSGraph::growth()
{
	Node* x, * y;

	for (Node** active = active0.list_vec.data();
		active != active0.list_vec.data() + active0.len;
		++active) {

		x = *active;

#if IBFS_OPT_PREFETCH
		// At the start of each outer iteration in growth(), prefetch the next active node
		if (active + 1 < active0.list_vec.data() + active0.len)
			_mm_prefetch((const char*)*(active + 1), _MM_HINT_T0);
#endif

		if (x->label != (dirS ? topLevelS - 1 : -(topLevelT - 1)))
			continue;

		if (dirS) stats.incGrowthS();
		else stats.incGrowthT();

		for (int i = 0, cnt = x->arcCount; i < cnt; ++i) {
			Arc* a = &x->arcs[i];

			if ((dirS ? a->rCap : (EdgeCap)((x->residBits >> i) & 1u)) == 0) continue;

			y = nodes + a->headIdx();

#if IBFS_OPT_PREFETCH
			// prefetch next arc's head node while we process this one
			if (i + 1 < cnt) {
				_mm_prefetch((const char*)(nodes + x->arcs[i + 1].headIdx()), _MM_HINT_T0);
			}
#endif

			Arc& rev = y->arcs[a->revIdx()];
			if (y->label == numNodes) {
				y->isParentCurr = 0;
				y->label = x->label + (dirS ? 1 : -1);

				// parent assignment
				y->parentRef = packParent(&rev, y);

				y->nextPtrIdx = x->firstSonIdx;
				x->firstSonIdx = idxOf(y);

				if (dirS) activeS1.add(y);
				else activeT1.add(y);

			}
			else if (dirS ? (y->label < 0) : (y->label > 0)) {

				// found augmenting bridge
				augment(dirS ? a : &rev);

				if (x->label != (dirS ? topLevelS - 1 : -(topLevelT - 1)))
					break;

				// recheck same arc if it still has residual
				if (dirS ? a->rCap : (EdgeCap)((x->residBits >> i) & 1u))
					--i;
			}
		}
	}

	active0.clear();
}


EdgeCap IBFSGraph::computeMaxFlow() {
	orphanFirstIdx = kOrphansEndIdx;
	orphanLastIdx  = kOrphansEndIdx;
	bool dirS = true;
	ActiveList::swapLists(&active0, &activeS1);

  while (true) {
		if (dirS) topLevelS++;
		else topLevelT++;

		if (dirS) growth<true>();
		else growth<false>();

    if (activeS1.len == 0 || activeT1.len == 0) break;

		if ((!IB_ALTERNATE_SMART && dirS) ||
			(IB_ALTERNATE_SMART && uniqOrphansT == uniqOrphansS && dirS) ||
			(IB_ALTERNATE_SMART && uniqOrphansT < uniqOrphansS)) {
			ActiveList::swapLists(&active0, &activeT1);
			dirS=false;
		} else {
			ActiveList::swapLists(&active0, &activeS1);
			dirS=true;
		}
	}
	
  return flow;
}


#if IBIO>0
bool IBFSGraph::readFromFile(char *filename)
{
	return readFromFile(filename, false);
}
bool IBFSGraph::readFromFileCompile(char *filename)
{
	return readFromFile(filename, true);
}
bool IBFSGraph::readFromFile(char *filename, bool checkCompile)
{
	const int MAX_LINE_LEN = 100;
	char line[MAX_LINE_LEN];
	int declaredNumOfNodes, declaredNumOfEdges, nodeId1, nodeId2;
	int currentNumOfEdges = 0;
	char c, c1, c2, c3;
	EdgeCap capacity, capacity2;
	int numLines=0;
	// only for compile mode
	const int bufferSize = sizeof(char) + sizeof(EdgeCap)*4;
	char buffer[bufferSize];

	char *filenameCompiled = new char[strlen(filename) + strlen(".compiled") + 1];
	strcpy(filenameCompiled, filename);
	strcat(filenameCompiled, ".compiled");

	FILE *pFile;
	FILE *pFileCompiled = NULL;
	if (checkCompile) {
		if ((pFileCompiled = fopen(filenameCompiled, "rb")) != NULL) {
			delete[] filenameCompiled;
			return readCompiled(pFileCompiled);
		}
		fclose(pFileCompiled);
	}
	if ((pFile = fopen(filename, "r")) == NULL) {
		fprintf(stdout, "Could not open file %s\n", filename);
		delete[] filenameCompiled;
		return false;
	}
	if (checkCompile && (pFileCompiled = fopen(filenameCompiled, "wb")) == NULL) {
		fprintf(stdout, "Could not open file %s\n", filenameCompiled);
		delete[] filenameCompiled;
		fclose(pFile);
		return false;
	}
	delete[] filenameCompiled;

	// read from file into temporary structure
	while (fgets(line, MAX_LINE_LEN, pFile) != NULL)
	{
		numLines++;
		switch (line[0])
	    {
			case 'c':
			case '\n':
			case '\0':
			default:
				break;
			case 'p':
				sscanf(line, "%c %c%c%c", &c, &c1, &c2, &c3);
				if (c1=='m' && c2=='a' && c3=='x') {
					sscanf(line, "%c %c%c%c %d %d", &c, &c1, &c2, &c3, &declaredNumOfNodes, &declaredNumOfEdges);
				} else {
					sscanf(line, "%c %d %d", &c, &declaredNumOfNodes, &declaredNumOfEdges);
				}
				initSize(declaredNumOfNodes, declaredNumOfEdges);
				if (checkCompile) {
					fwrite(&declaredNumOfNodes, sizeof(int), 1, pFileCompiled);
					fwrite(&declaredNumOfEdges, sizeof(int), 1, pFileCompiled);
				}
				break;

			case 'n':
				sscanf(line, "%c %d %d %d ", &c, &nodeId1, &capacity, &capacity2);
				if (capacity != 0 || capacity2 != 0) {
					addNode(nodeId1, capacity, capacity2);
					if (checkCompile) {
						buffer[0] = 'n';
						memcpy(buffer+sizeof(char), &nodeId1, sizeof(int));
						memcpy(buffer+sizeof(char)+sizeof(int), &nodeId1, sizeof(int));
						memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int), &capacity, sizeof(int));
						memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int)+sizeof(int), &capacity2, sizeof(int));
						fwrite(&buffer, 1, bufferSize, pFileCompiled);
					}
				}
				break;

			case 'a':
				sscanf(line, "%c %d %d %d %d", &c,
					&nodeId1, &nodeId2, &capacity, &capacity2);
				if (nodeId1 < 0 ||
					nodeId1 >= declaredNumOfNodes ||
					nodeId2 < 0 ||
					nodeId2 >= declaredNumOfNodes)
				{
					fprintf(stdout, "inconsistent node index (Line %d)\n", numLines);
					return false;
				}
				if (currentNumOfEdges >= declaredNumOfEdges)
				{
					fprintf(stdout, "inconsistent number of edges (Line %d)\n", numLines);
					return false;
				}
				addEdge(nodeId1, nodeId2, capacity, capacity2);
				currentNumOfEdges++;
				if (checkCompile) {
					buffer[0] = 'a';
					memcpy(buffer+sizeof(char), &nodeId1, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int), &nodeId2, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int), &capacity, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int)+sizeof(int), &capacity2, sizeof(int));
					fwrite(&buffer, 1, bufferSize, pFileCompiled);
				}
				break;
		}
	}

	fclose(pFile);
	if (checkCompile) {
		buffer[0] = 'x';
		fwrite(&buffer, 1, bufferSize, pFileCompiled);
	}
	if (currentNumOfEdges != declaredNumOfEdges) {
		fprintf(stdout, "inconsistent number of edges: differs from declared %d != %d\n",
				currentNumOfEdges, declaredNumOfEdges);
		return false;
	}
	return true;
}


bool IBFSGraph::readCompiled(FILE *pFile)
{
	int declaredNumOfNodes, declaredNumOfEdges, nodeId1, nodeId2;
	EdgeCap capacity, capacity2;
	const int bufferSize = sizeof(char)+sizeof(EdgeCap)*4;
	char buffer[bufferSize];

	// read from file into htemporary structure
	fprintf(stdout, "c reading compiled file\n");
	if (fread(&declaredNumOfNodes, sizeof(int), 1, (pFile)) < 1 ||
			fread(&declaredNumOfEdges, sizeof(int), 1, (pFile)) < 1) {
		fprintf(stdout, "ERROR while reading compiled num nodes/edges, EOF=%d\n", feof(pFile));
		fclose(pFile);
		return false;
	}
	initSize(declaredNumOfNodes, declaredNumOfEdges);
	for (int line=0; !feof(pFile); line++) {
		if (fread(&buffer, 1, bufferSize, pFile) < bufferSize) {
			fprintf(stdout, "ERROR while reading compiled line %d, EOF=%d\n", line, feof(pFile));
			fclose(pFile);
			return false;
		}
		memcpy(&nodeId1,   buffer+sizeof(char), sizeof(int));
		memcpy(&nodeId2,   buffer+sizeof(char)+sizeof(int), sizeof(int));
		memcpy(&capacity,  buffer+sizeof(char)+sizeof(int)+sizeof(int), sizeof(int));
		memcpy(&capacity2, buffer+sizeof(char)+sizeof(int)+sizeof(int)+sizeof(int), sizeof(int));
		if (buffer[0] == 'n') {
			if (capacity != 0 || capacity2 != 0) {
				addNode(nodeId1, capacity, capacity2);
			}
		} else if (buffer[0] == 'a') {
			if (nodeId1 < 0 ||
				nodeId1 >= declaredNumOfNodes ||
			nodeId2 < 0 ||
				nodeId2 >= declaredNumOfNodes)
			{
				fprintf(stdout, "inconsistent node index in compiled file %d,%d line %d\n", nodeId1, nodeId2, line);
				return false;
			}
			addEdge(nodeId1, nodeId2, capacity, capacity2);
		} else if (buffer[0] == 'x') {
			break;
		}
	}
	fclose(pFile);
	return true;
}
#endif