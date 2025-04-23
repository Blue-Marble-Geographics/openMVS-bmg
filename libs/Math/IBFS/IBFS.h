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

#define GC_OPTS

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
#define IBDEBUG(X) /*fprintf(stdout, X"\n"); fflush(stdout) */
#endif

#define IB_ALTERNATE_SMART 1
#define IB_ORPHANS_END ((Node*)1)

#pragma pack(push, 1)
namespace IBFS {

typedef float Real;
typedef float EdgeCap;

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

double __forceinline edgeCapToDouble(EdgeCap f)
{
	if constexpr (std::is_same_v<EdgeCap, int32_t>) {
		constexpr double C = (1. / 16.);
		return f * C;
	} else if constexpr(std::is_same_v<EdgeCap, int64_t>) {
		constexpr double C = (1. / (65536.*65536.));
		return f * C;
	} else if constexpr(std::is_same_v<EdgeCap, float>) {
		return f;
	} else if constexpr(std::is_same_v<EdgeCap, double>) {
		return f;
	}
}

EdgeCap __forceinline doubleToEdgeCap(double f)
{
	if constexpr (std::is_same_v<EdgeCap, int32_t>) {
		constexpr double C = 16.;
		return f * C;
	} else if constexpr (std::is_same_v<EdgeCap, int64_t>) {
		constexpr double C = (65536.*65536.);
		return f * C;
	} else if constexpr(std::is_same_v<EdgeCap, float>) {
		return f;
	} else if constexpr(std::is_same_v<EdgeCap, double>) {
		return f;
	}
}

class IBFSGraph
{
public:
	IBFSGraph();
	~IBFSGraph();

	void setVerbose(bool a_verbose) {
		verbose = a_verbose;
	}

	void initSize(int numNodes, int numEdges);
#ifdef GC_OPTS
	void addEdge(const void* nodeIndexFrom, const void* nodeIndexTo, double capacity, double reverseCapacity);
	EdgeCap addNode(const void* node, double capacityFromSource, double capacityToSink);
	void initFlow(double f) { flow = doubleToEdgeCap(f); }
#else
	void addEdge(int nodeIndexFrom, int nodeIndexTo, EdgeCap capacity, EdgeCap reverseCapacity);
	void addNode(int nodeIndex, EdgeCap capacityFromSource, EdgeCap capacityToSink);
#endif

	void setCompactSlowInitMode(bool a_compactSlowInitMode) {
		compactSlowInitMode = a_compactSlowInitMode;
	}
	void initGraph();
	double computeMaxFlow();

	inline IBFSStats getStats() {
		return stats;
	}
	inline double getFlow() {
		return edgeCapToDouble(flow);
	}
	inline size_t getNumNodes() {
		return nodeEnd-nodes;
	}
	inline size_t getNumArcs() {
		return arcEnd-arcs;
	}

#ifdef GC_OPTS
	bool isNodeOnSrcSide(const void* node) const;
#else
	bool isNodeOnSrcSide(int nodeIndex) const;
#endif

#ifdef GC_OPTS
	const void* nodeHandle(int n) const { return nodes + n; }
#endif

private:
	struct Node;
	struct Arc;

	struct Arc
	{
		Node*		head;
		Arc*		rev;
		#if 0
		int			isRevResidual :1;
		int			rCap :31;
		#else
		EdgeCap			rCap;
		unsigned char	isRevResidual;
		#endif
	};

	struct alignas(64) Node // 229 -> 222 without this
	{
		int			lastAugTimestamp:31;
		int			isParentCurr:1;
		Arc			*firstArc;
		Arc         *endArc;
		Arc			*parent;
		Node		*firstSon;
		Node		*nextPtr;
		std::atomic<int>			label;	// label > 0: distance from s, label < 0: -distance from t
		EdgeCap		excess;	 // excess > 0: capacity from s, excess < 0: -capacity to t
	};

	class ActiveList
	{
	public:
		inline ActiveList() {
			list = NULL;
			len = 0;
		}
		inline void init(int numNodes) {
			list = new Node*[numNodes];
			len = 0;
		}
		inline void release() {
			if (list != NULL) {
				delete[] list;
				list = NULL;
			}
		}
		inline void clear() {
			len = 0;
		}
		inline void add(Node* x) {
			list[len] = x;
			len++;
		}
		inline static void swapLists(ActiveList *a, ActiveList *b) {
			ActiveList tmp = (*a);
			(*a) = (*b);
			(*b) = tmp;
		}
		Node **list;
		int len;
	};

	class Buckets
	{
	public:
		inline Buckets() {
			buckets = NULL;
			prevPtrs = NULL;
			maxBucket = 0;
			nodes = NULL;
		}
		inline void init(Node *a_nodes, int numNodes) {
			nodes = a_nodes;
			buckets = new Node*[numNodes];
			memset(buckets, 0, sizeof(Node*)*numNodes);
			prevPtrs = new Node*[numNodes];
			memset(prevPtrs, 0, sizeof(Node*)*numNodes);
			maxBucket = 0;
		}
		inline void release() {
			if (buckets != NULL) {
				delete[] buckets;
				buckets = NULL;
			}
			if (prevPtrs != NULL) {
				delete[] prevPtrs;
				prevPtrs = NULL;
			}
		}
		template <bool sTree> inline void add(Node* x) {
			int bucket = (sTree ? (x->label.load()) : (-x->label.load()));
			if (buckets[bucket] == NULL || buckets[bucket] == IB_ORPHANS_END) {
				x->nextPtr = IB_ORPHANS_END;
			} else {
				x->nextPtr = buckets[bucket];
				prevPtrs[x->nextPtr-nodes] = x;
			}
			buckets[bucket] = x;
			if (bucket > maxBucket) maxBucket = bucket;
		}
		inline Node* popFront(int bucket) {
			Node *x = buckets[bucket];
			if (x == NULL || x == IB_ORPHANS_END) return NULL;
			buckets[bucket] = x->nextPtr;
			//x->nextOrphan = NULL;
			return x;
		}
		template <bool sTree> inline void remove(Node *x) {
			int bucket = (sTree ? (x->label.load()) : (-x->label.load()));
			if (buckets[bucket] == x) {
				buckets[bucket] = x->nextPtr;
			} else {
				prevPtrs[x-nodes]->nextPtr = x->nextPtr;
				if (x->nextPtr != IB_ORPHANS_END) prevPtrs[x->nextPtr-nodes] = prevPtrs[x-nodes];
			}
			//x->nextOrphan = NULL;
		}

		Node **buckets;
		Node **prevPtrs;
		Node *nodes;
		int maxBucket;
	};

	// members
	IBFSStats stats;
	Node	*nodes, *nodeEnd;
	Arc		*arcs, *arcEnd;
	int 	numNodes;
public: // JPB WIP BUG
	EdgeCap	flow;
private:
	unsigned short augTimestamp;
	unsigned int uniqOrphansS, uniqOrphansT;
	Node* orphanFirst;
	Node* orphanLast;
	int topLevelS, topLevelT;
	ActiveList active0, activeS1, activeT1;
	Buckets orphanBuckets;
	bool verbose;

	void augment(Arc *bridge);
	template <bool sTree> void augmentTree(Node *x, EdgeCap bottleneck);
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
	TmpEdge	*tmpEdges;
#ifdef GC_OPTS
	std::atomic<TmpEdge*> tmpEdgeLast;
#else
	TmpEdge *tmpEdgeLast;
#endif
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

#ifdef GC_OPTS
inline EdgeCap IBFSGraph::addNode(const void* n, double capacitySource, double capacitySink)
{
	Node* node = (Node*) n;
	const double f = edgeCapToDouble(node->excess);
	if (f > 0) {
		capacitySource += f;
	} else {
		capacitySink -= f;
	}
	if (capacitySource < capacitySink) {
		flow += capacitySource;
	} else {
		flow += capacitySink;
	}

	node->excess = doubleToEdgeCap(capacitySource - capacitySink);
	return flow;
}
#else
inline void IBFSGraph::addNode(int nodeIndex, EdgeCap capacitySource, EdgeCap capacitySink)
{
	EdgeCap f = nodes[nodeIndex].excess;
	if (f > 0) {
		capacitySource += f;
	} else {
		capacitySink -= f;
	}
	if (capacitySource < capacitySink) {
		flow += capacitySource;
	} else {
		flow += capacitySink;
	}
	nodes[nodeIndex].excess = capacitySource - capacitySink;
}
#endif

#ifdef GC_OPTS
inline void IBFSGraph::addEdge(const void* nodeIndexFrom, const void* nodeIndexTo, double capacity, double reverseCapacity)
{
	assert((void*)tmpEdgeLast < (void*)tmpArcs);
	TmpEdge* t = tmpEdgeLast.fetch_add(1); // increment atomically.
	t->tail = (Node*) nodeIndexFrom;
	t->head = (Node*) nodeIndexTo;
	t->cap = doubleToEdgeCap(capacity);
	t->revCap = doubleToEdgeCap(reverseCapacity);

	// use label as a temporary storage
	// to count the out degree of nodes
	((Node*) nodeIndexFrom)->label++;
	((Node*) nodeIndexTo)->label++;

	/*
	Arc *aFwd = arcLast;
	arcLast++;
	Arc *aRev = arcLast;
	arcLast++;

	Node* x = nodes + nodeIndexFrom;
	x->label++;
	Node* y = nodes + nodeIndexTo;
	y->label++;

	aRev->rev = aFwd;
	aFwd->rev = aRev;
	aFwd->rCap = capacity;
	aRev->rCap = reverseCapacity;
	aFwd->head = y;
	aRev->head = x;*/
}
#else
inline void IBFSGraph::addEdge(int nodeIndexFrom, int nodeIndexTo, EdgeCap capacity, EdgeCap reverseCapacity)
{
	assert((void*)tmpEdgeLast < (void*)tmpArcs);
	tmpEdgeLast->tail = nodes + nodeIndexFrom;
	tmpEdgeLast->head = nodes + nodeIndexTo;
	tmpEdgeLast->cap = capacity;
	tmpEdgeLast->revCap = reverseCapacity;
	tmpEdgeLast++;

	// use label as a temporary storage
	// to count the out degree of nodes
	nodes[nodeIndexFrom].label++;
	nodes[nodeIndexTo].label++;

	/*
	Arc *aFwd = arcLast;
	arcLast++;
	Arc *aRev = arcLast;
	arcLast++;

	Node* x = nodes + nodeIndexFrom;
	x->label++;
	Node* y = nodes + nodeIndexTo;
	y->label++;

	aRev->rev = aFwd;
	aFwd->rev = aRev;
	aFwd->rCap = capacity;
	aRev->rCap = reverseCapacity;
	aFwd->head = y;
	aRev->head = x;*/
}
#endif

#ifdef GC_OPTS
inline bool IBFSGraph::isNodeOnSrcSide(const void* node) const
{
	if (((Node*) node)->label == numNodes || ((Node*) node)->label == 0) {
		return activeT1.len == 0;
	}
	return ((Node*) node)->label > 0;
}
#else
inline bool IBFSGraph::isNodeOnSrcSide(int nodeIndex) const
{
	if (nodes[nodeIndex].label == numNodes || nodes[nodeIndex].label == 0) {
		return activeT1.len == 0;
	}
	return (nodes[nodeIndex].label > 0);
}
#endif

} // namespace IBFS

#pragma pack(pop)
#endif
