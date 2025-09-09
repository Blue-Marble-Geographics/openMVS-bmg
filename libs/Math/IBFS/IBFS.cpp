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

using namespace IBFS;

constexpr float kMinResidual = 0.025f;  // Half of the smallest quantized step (0.05)

//
// Orphan handling
//
#define ADD_ORPHAN_BACK(n)							\
if (orphanFirst != IB_ORPHANS_END)					\
{													\
	orphanLast = (orphanLast->nextPtr = (n));		\
}													\
else												\
{													\
	orphanLast = (orphanFirst = (n));				\
}													\
(n)->nextPtr = IB_ORPHANS_END





#define ADD_ORPHAN_FRONT(n)							\
if (orphanFirst == IB_ORPHANS_END)					\
{													\
	(n)->nextPtr = IB_ORPHANS_END;					\
	orphanLast = (orphanFirst = (n));				\
}													\
else												\
{													\
	(n)->nextPtr = orphanFirst;						\
	orphanFirst = (n);								\
}




IBFSGraph::IBFSGraph() {
	numNodes = 0;
	uniqOrphansS = uniqOrphansT = 0;
	augTimestamp = 0;
	verbose = IBTEST;
	flow = 0;
  orphanFirst = orphanLast = nullptr;
  nodes = nodeEnd = nullptr;
}

IBFSGraph::~IBFSGraph() {
	active0.release();
	activeS1.release();
	activeT1.release();
	orphanBuckets.release();
	delete[] nodes;
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
	numNodes = n;
	nodes = new Node[n];

#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int tcount = omp_get_num_threads();

		const int chunkSize = (n + tcount - 1) / tcount;
		const int begin = tid * chunkSize;
		const int end = std::min(begin + chunkSize, n);

		for (int i = begin; i < end; ++i) {
			Node& node = nodes[i];
			node.arcCount = 0;
			node.excess = 0;
			node.parent = nullptr;
			node.firstSon = nullptr;
			node.nextPtr = nullptr;
			node.lastAugTimestamp = 0;
			node.isParentCurr = 0;
			node.label = 0;
		}
	}

	nodeEnd = nodes + n;
	active0.init(n);
	activeS1.init(n);
	activeT1.init(n);
	orphanBuckets.init(nodes, n);
	flow = 0;
}

template <bool sTree>
void IBFSGraph::augmentTree(Node* x, EdgeCap bottleneck) {
	Node* y;
	Arc* a;
	int hopCount = 0;

	while (true) {
		if (x->excess) break;

		a = x->parent;

		if (sTree) {
			a->rCap += bottleneck;
			a->rev->isRevResidual = 1;
			a->rev->rCap -= bottleneck;
		} else {
			a->rev->rCap += bottleneck;
			a->isRevResidual = 1;
			a->rCap -= bottleneck;
		}

		if ((sTree ? a->rev->rCap : a->rCap) == 0) {
			if (sTree) a->isRevResidual = 0;
			else a->rev->isRevResidual = 0;

			y = a->head->firstSon;
			if (y == x) {
				a->head->firstSon = x->nextPtr;
			}
			else {
				for (; y && y->nextPtr != x; y = y->nextPtr);
				if (y) y->nextPtr = x->nextPtr;
			}

			x->nextPtr = IB_ORPHANS_END;
			if (orphanFirst != IB_ORPHANS_END) orphanLast = orphanLast->nextPtr = x;
			else orphanFirst = orphanLast = x;
		}

		x = a->head;
	}

	x->excess += (sTree ? -bottleneck : bottleneck);
	if (x->excess == 0) {
		x->nextPtr = IB_ORPHANS_END;
		if (orphanFirst != IB_ORPHANS_END) orphanLast = orphanLast->nextPtr = x;
		else orphanFirst = orphanLast = x;
	}
}

void IBFSGraph::augment(Arc *bridge)
{
	Node* x;
	Arc *a;
	EdgeCap bottleneck;
	Real pushesBefore;

	// stats
	if (IBSTATS) pushesBefore=stats.getPushes();
	stats.incAugs();
	stats.incPushes();

	// bottleneck in S
	bottleneck = bridge->rCap;
	for (x=bridge->rev->head; ; x=a->head)
	{
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

	// stats
	if (IBSTATS) {
		Real augLen = stats.getPushes() - pushesBefore;
		stats.addAugLen(augLen);
	}

	// augment connecting arc
	bridge->rev->rCap += bottleneck;
	bridge->isRevResidual = 1;
	bridge->rCap -= bottleneck;
	if (bridge->rCap == 0) {
		bridge->rev->isRevResidual = 0;
	}

	// augment T
	augTimestamp++;
	augmentTree<false>(bridge->head, bottleneck);
	adoption<false>();

	// augment S
	augTimestamp++;
	augmentTree<true>(bridge->rev->head, bottleneck);
	adoption<true>();

	flow += bottleneck;
}


template <bool sTree>
void IBFSGraph::adoption()
{
	Node *x, *y, *z;
  Arc *a;
  bool threePass = false;
  int minLabel, numOrphans = 0, numOrphansUniq = 0;

  while (orphanFirst != IB_ORPHANS_END) {
		x = orphanFirst;
		orphanFirst = x->nextPtr;
		testNode(x);
		stats.incOrphans();
		numOrphans++;

		if (x->lastAugTimestamp != augTimestamp) {
			x->lastAugTimestamp = augTimestamp;
			if (sTree) uniqOrphansS++;
			else uniqOrphansT++;
			numOrphansUniq++;
		}
		if (numOrphans >= 3*numOrphansUniq) {
			threePass = true;
		}

    // check for same-level parent
		if (x->isParentCurr) {
			a = x->parent;
		} else {
			x->isParentCurr = 1;
      a = nullptr;
		}
    x->parent = nullptr;

    if (x->label != (sTree ? 1 : -1)) {
      minLabel = x->label - (sTree ? 1 : -1);
      for (int i = 0; i < x->arcCount; ++i) {
        a = &x->arcs[i];
				stats.incOrphanArcs1();
				y = a->head;
        if ((sTree ? a->isRevResidual : a->rCap) && y->label == minLabel) {
					x->parent = a;
					x->nextPtr = y->firstSon;
					y->firstSon = x;
					break;
				}
			}
		}
    if (x->parent != nullptr) continue;

    // orphan children
    for (y = x->firstSon; y != nullptr; y = z) {
			stats.incOrphanArcs3();
			z=y->nextPtr;
			ADD_ORPHAN_BACK(y);
		}
    x->firstSon = nullptr;

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
      for (int i = 0; i < x->arcCount; ++i) {
        a = &x->arcs[i];
			stats.incOrphanArcs2();
			y = a->head;
			if ((sTree ? a->isRevResidual : a->rCap) &&
            (sTree ? y->label > 0 : y->label < 0) &&
            (sTree ? y->label < minLabel : y->label > minLabel)) {
          minLabel = y->label;
				x->parent = a;
          if (minLabel == x->label) break;
        }
			}
		}

    if (x->parent != nullptr) {
      x->label = minLabel + (sTree ? 1 : -1);
			x->nextPtr = x->parent->head->firstSon;
			x->parent->head->firstSon = x;
      if (sTree && x->label == topLevelS) activeS1.add(x);
      else if (!sTree && x->label == -topLevelT) activeT1.add(x);
			} else {
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
	Arc *a;
	Node *x, *y;
	int minLabel, destLabel;

	for (int level=2; level <= orphanBuckets.maxBucket; level++) {
		while ((x = orphanBuckets.popFront(level)) != NULL) {
			testNode(x);
			// pass 2: find lowest level parent
			if (x->parent == NULL) {
				minLabel = (sTree ? topLevelS : -topLevelT);
				destLabel = x->label - (sTree ? 1 : -1);
				for (int i = 0; i < x->arcCount; ++i) {
					a = &x->arcs[i];
					y = a->head;
					if ((sTree ? a->isRevResidual : a->rCap) &&
						(y->excess || y->parent != NULL) &&
						//!y->isOrphan() &&
						(sTree ? (y->label > 0) : (y->label < 0)) &&
						(sTree ? (y->label < minLabel) : (y->label > minLabel)))
					{
						x->parent = a;
						if ((minLabel = y->label) == destLabel) break;
					}
				}
				if (x->parent == NULL) {
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
				for (int i = 0; i < x->arcCount; ++i) {
					a = &x->arcs[i];
					y = a->head;

					// lower potential sons
					if ((sTree ? a->rCap : a->isRevResidual) &&
						((!sTree && y->label == numNodes) ||
						// the above implicitly holds by condition below when sTree=true
						(sTree ? (minLabel < y->label) : (minLabel > y->label))))
					{
						if (y->label != numNodes) orphanBuckets.remove<sTree>(y);
						y->label = minLabel;
						y->parent = a->rev;
						orphanBuckets.add<sTree>(y);
					}
				}
			}

			// relabel onto new parent
			x->nextPtr = x->parent->head->firstSon;
			x->parent->head->firstSon = x;
			x->isParentCurr = 0;
			// add to active list of the next growth phase
			if (sTree) {
				if (x->label == topLevelS) activeS1.add(x);
			} else {
				if (x->label == -topLevelT) activeT1.add(x);
			}
		}
	}

	orphanBuckets.maxBucket = 0;
}


template <bool dirS>
void IBFSGraph::growth() {
	Node *x, *y;

  for (Node** active = active0.list; active != active0.list + active0.len; ++active) {
    x = *active;

    if (x->label != (dirS ? topLevelS - 1 : -(topLevelT - 1))) continue;

		if (dirS) stats.incGrowthS();
		else stats.incGrowthT();

    for (int i = 0; i < x->arcCount; ++i) {
      Arc* a = &x->arcs[i];

			if ((dirS ? a->rCap : a->isRevResidual) == 0) continue;

			y = a->head;

      if (y->label == numNodes) {
				y->isParentCurr = 0;
        y->label = x->label + (dirS ? 1 : -1);
				y->parent = a->rev;
				y->nextPtr = x->firstSon;
				x->firstSon = y;
				if (dirS) activeS1.add(y);
				else activeT1.add(y);
      } else if (dirS ? (y->label < 0) : (y->label > 0)) {
        augment(dirS ? a : a->rev);

        if (x->label != (dirS ? topLevelS - 1 : -(topLevelT - 1))) break;
        if ((dirS ? a->rCap : a->isRevResidual)) --i;
			}
		}
	}

	active0.clear();
}


EdgeCap IBFSGraph::computeMaxFlow() {
	orphanFirst = IB_ORPHANS_END;
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