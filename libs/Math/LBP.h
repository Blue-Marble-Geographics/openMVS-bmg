////////////////////////////////////////////////////////////////////
// LBA.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __SEACAVE_LBA_H__
#define __SEACAVE_LBA_H__


// I N C L U D E S /////////////////////////////////////////////////

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define LBP_USE_OPENMP
#endif

#ifdef LBP_USE_OPENMP
#include <omp.h>
#endif

// Iteration budget / convergence controls for LBPInference::Optimize().
//  - LBP_MAX_ITERS: hard cap on message-passing sweeps. On large texture MRFs
//    (millions of faces) the label-flip count decays smoothly (~0.93x/iter) and
//    often does NOT reach the convergence threshold within the cap, so the run
//    burns the full budget. Each sweep is O(msgBuf) memory traffic (~145 ms on a
//    2.75M-face / 8.2M-edge / 242 MB-message graph). The tail sweeps only flip
//    marginal-tie faces (two views nearly equal quality) whose choice is visually
//    near-equivalent and whose seams are smoothed by seam leveling anyway.
//    Lower this to trade a little view-selection optimality for a near-linear
//    speedup (e.g. 25 ~= halves LBP time). 50 = original behaviour.
//  - LBP_CONVERGENCE_FRAC: early-out when a sweep flips fewer than this fraction
//    of nodes. 0.005 (0.5%) = original. Raising it (e.g. 0.01) stops the tail a
//    few sweeps sooner.
#ifndef LBP_MAX_ITERS
#define LBP_MAX_ITERS 50u
#endif
#ifndef LBP_CONVERGENCE_FRAC
#define LBP_CONVERGENCE_FRAC 0.005f
#endif

// The [LBP-DIAG] trace rides TEXTURE_DIAG (MVS/Common.h): this solver is only ever
// used by the texturing MRF, and its instrumentation used to sit behind
// "#ifdef DEBUG_EXTRA" -- which is ALWAYS defined, so the guard did nothing and the
// per-sweep line (plus its full-graph energy evaluation) printed at the default
// verbosity. SceneTexture.cpp includes this header after MVS/Common.h, so the macro
// is in scope; the fallback below only keeps the header standalone-includable.
#ifndef TEXTURE_DIAG
#define TEXTURE_DIAG(...)
#define TEXTURE_DIAG_ENABLED()	false
#endif

namespace SEACAVE {

// S T R U C T S ///////////////////////////////////////////////////

// basic implementation of the loopy belief propagation algorithm
// based on a code originally written by Michael Waechter:
// https://github.com/nmoehrle/mvs-texturing
// Copyright(c) Michael Waechter
// Licensed under the BSD 3-Clause license
class MATH_API LBPInference
{
public:
	typedef unsigned NodeID;
	typedef unsigned LabelID;
	typedef unsigned EdgeID;
	typedef float EnergyType;

	struct DataCost {
		NodeID nodeID;
		EnergyType cost;
	};

	typedef EnergyType (STCALL *FncSmoothCost)(NodeID, NodeID, LabelID, LabelID);

	enum { MaxEnergy = 1000 };
	// Historic inline capacity of the per-node label vectors. Those vectors are gone
	// (labels/dataCosts are CSR now -- see Node), so this no longer sizes anything and
	// there is no reallocate-vs-waste trade-off left to tune. Kept only because it is
	// public; nothing reads it.
	enum { kApproxMaxNumSharedViews = 32 };

public:
	// Messages do NOT live here: they are in the flat ping-pong msgBuf[] arrays
	// (see PrepareMessageBuffers). This struct is instantiated once per directed
	// edge -- millions of times -- so it must stay small.
	struct DirectedEdge {
		NodeID nodeID1;
		NodeID nodeID2;
		inline DirectedEdge(NodeID _nodeID1, NodeID _nodeID2) : nodeID1(_nodeID1), nodeID2(_nodeID2) {}
	};

	// 8 bytes. Everything else about a node -- its candidate labels, their data
	// costs, and its incoming edges -- lives in the flat CSR arrays below.
	//
	// WHY. This used to carry two small_vector<...,kApproxMaxNumSharedViews> plus a
	// small_vector<EdgeID,3>, which put sizeof(Node) at ~352 B: 24 B of header and
	// 128 B of inline storage per label vector, times two, whether or not it was
	// used. Measured avgLabels on a real scene is 8.84 against an inline capacity of
	// 32, so ~110 B of every 352 B was live. That would be merely wasteful if the
	// array were touched sparsely -- but the message-passing sweep walks EVERY node,
	// and the three fields it reads are spread across ~5 of the 6 cache lines, so
	// each sweep streamed the whole array: ~1.3 GB at 3.8M nodes, which is five times
	// the entire message buffer the loop was supposedly bound on. In CSR the same
	// content is ~344 MB in four contiguous streams that prefetch perfectly.
	struct Node {
		LabelID label;
		EnergyType dataCost;
		inline Node() : label(0), dataCost(MaxEnergy) {}
	};

	std::vector<DirectedEdge> edges;
	std::vector<Node> nodes;
	std::vector<LabelID> finalLabels;
	FncSmoothCost fncSmoothCost;

	// ---- per-node candidate labels + data costs, CSR ----
	// nodeLabelOfs has nodes.size()+1 entries; node n owns [ofs[n], ofs[n+1]) in both
	// nodeLabels and nodeDataCosts, which are parallel. Built by the two-phase
	// ReserveDataCosts() / SetDataCost() pair, because a node's label count is known
	// up front at both call sites and that avoids any per-node growable container.
	std::vector<uint32_t> nodeLabelOfs;
	std::vector<LabelID> nodeLabels;
	std::vector<EnergyType> nodeDataCosts;
	// Build-time only: how many labels have been written to each node so far.
	// Freed by PrepareTopology once the graph is complete.
	std::vector<uint32_t> labelFill;

	// ---- per-node incoming edges, CSR ----
	// Derived in PrepareTopology by scanning `edges` ascending. That reproduces the
	// exact order the old per-node push_back produced -- SetNeighbors appends edge
	// 2k to nodes[nodeID2] and 2k+1 to nodes[nodeID1] on its k-th call, so a node's
	// list was already in ascending edge order -- which matters because the order of
	// this list is the order the incoming messages are summed, and float addition is
	// not associative.
	std::vector<uint32_t> nodeInEdgeOfs;
	std::vector<EdgeID> nodeInEdges;

	inline uint32_t NumLabels(NodeID n) const { return nodeLabelOfs[n+1] - nodeLabelOfs[n]; }
	inline const LabelID* Labels(NodeID n) const { return nodeLabels.data() + nodeLabelOfs[n]; }
	inline LabelID* Labels(NodeID n) { return nodeLabels.data() + nodeLabelOfs[n]; }
	inline const EnergyType* DataCosts(NodeID n) const { return nodeDataCosts.data() + nodeLabelOfs[n]; }
	inline EnergyType* DataCosts(NodeID n) { return nodeDataCosts.data() + nodeLabelOfs[n]; }
	// When true, fncSmoothCost is assumed to be a (generalized) Potts model:
	// cost is 0 for l1==l2 and a single per-edge constant otherwise (independent
	// of the specific label values). Enables the O(L1+L2) message fast path in
	// Optimize() that skips the L1*L2 fncSmoothCost evaluations. Leave false for
	// arbitrary pairwise costs. Bit-identical to the general path when the model
	// really is Potts.
	bool bPottsSmoothness = false;

	// Cached topology
	//
	// There is deliberately NO outEdges array. SetNeighbors is the only producer of
	// `edges` and always appends the pair (n1->n2) at e0 and (n2->n1) at e0+1 into a
	// vector that starts empty and only ever grows in twos, so e0 is always even and
	// the reverse of any directed edge e is exactly e^1. incomingEdges[u] and the
	// outgoing set of u are therefore the same list under an XOR, and materializing
	// the second one cost a small_vector per node (~120 MB at 4M nodes) plus one more
	// random-access stream in the hot loop for nothing. See the emit loop in Optimize.
	LabelID globalMaxLabel = 0;
	size_t maxLabelsPerNode = 0;
	bool topologyPrepared = false;

	// Per-(directed-)edge Potts weight, precomputed once when bPottsSmoothness is
	// set. The Potts constant depends only on the edge's endpoints (e.g. the two
	// face normals), which are fixed for the whole solve, so evaluating
	// fncSmoothCost every iteration is pure waste; cache it here instead.
	std::vector<EnergyType> edgeWeight;

public:
	LBPInference() {}
	LBPInference(NodeID nNodes) : nodes(nNodes) {}

	void PrepareTopology()
	{
		if (topologyPrepared)
			return;

		const size_t numNodes = nodes.size();

		// The data-cost build is complete by now, so the fill cursors are dead.
		labelFill.clear();
		labelFill.shrink_to_fit();

		// Build the per-node incoming-edge CSR ONCE, by counting then scattering.
		// Scanning e ascending reproduces the order the old per-node push_back gave
		// (see nodeInEdges), which the float summation order depends on.
		nodeInEdgeOfs.assign(numNodes + 1, 0);
		for (EdgeID e = 0; e < edges.size(); ++e)
			++nodeInEdgeOfs[edges[e].nodeID2 + 1];
		for (size_t i = 0; i < numNodes; ++i)
			nodeInEdgeOfs[i+1] += nodeInEdgeOfs[i];
		nodeInEdges.resize(edges.size());
		{
			std::vector<uint32_t> fill(numNodes, 0);
			for (EdgeID e = 0; e < edges.size(); ++e) {
				const NodeID v = edges[e].nodeID2;
				nodeInEdges[nodeInEdgeOfs[v] + fill[v]++] = e;
			}
		}

		// Compute label bounds ONCE
		globalMaxLabel = 0;
		maxLabelsPerNode = 0;

		for (size_t n = 0; n < numNodes; ++n)
			maxLabelsPerNode = std::max(maxLabelsPerNode, (size_t)NumLabels((NodeID)n));
		for (const LabelID l : nodeLabels)
			globalMaxLabel = std::max(globalMaxLabel, l);

		// Precompute the constant per-edge Potts weight ONCE (fncSmoothCost with
		// any two distinct labels). Removes ~numIterations redundant evaluations
		// (each a random faceNormals[] fetch + dot product) per directed edge.
		if (bPottsSmoothness) {
			edgeWeight.resize(edges.size());
			// A non-finite pairwise weight is not survivable and fails SILENTLY, so it
			// is scrubbed here rather than trusted. The failure chain, observed as
			// "[LBP-DIAG] ... energy=-nan data=<finite> smooth=-nan":
			//   minPlusW = minAll + W is NaN -> every `energyBuf[k] < best` test is
			//   false -> the whole outgoing message is NaN -> the neighbour's incoming
			//   sum is NaN -> its own messages are NaN. The poison advances one ring
			//   per sweep. Normalization cannot clear it (minMsg stays FLT_MAX), and in
			//   the final labeling `e < bestE` never fires, so the node silently pins
			//   labels[0] and stops flipping -- which SUPPRESSES `changed=` and makes a
			//   dead region look like convergence.
			// 0 is the correct neutral: the only producer seen in practice is a
			// degenerate (zero-area) face, whose seams have no visible area to cost.
			size_t numBadWeights = 0;
			for (EdgeID e = 0; e < edges.size(); ++e) {
				const EnergyType w = fncSmoothCost(edges[e].nodeID1, edges[e].nodeID2, 0, 1);
				// `!(finite)` rather than a NaN/inf test, so both take the fallback
				if (!(w > -std::numeric_limits<EnergyType>::max() && w < std::numeric_limits<EnergyType>::max())) {
					edgeWeight[e] = EnergyType(0);
					++numBadWeights;
				} else {
					edgeWeight[e] = w;
				}
			}
			if (numBadWeights)
				TEXTURE_DIAG("[LBP-DIAG] warning: %zu/%zu directed edges had a non-finite smoothness weight (scrubbed to 0) -- check the face normals",
					numBadWeights, edges.size());

			// Precompute the label cross-reference ONCE (see msgLabelXRef). This is
			// exactly the work the Potts inner loop used to redo every sweep; one pass
			// here replaces ~LBP_MAX_ITERS of it.
			ASSERT(maxLabelsPerNode < (size_t)kNoLabelPos16); // uint16 slot index
			ASSERT(msgOffset.size() == edges.size() + 1);     // PrepareMessageBuffers ran
			msgLabelXRef.assign(msgOffset[edges.size()], kNoLabelPos16);
			{
				const uint32_t* __restrict offs = msgOffset.data();
			#ifdef LBP_USE_OPENMP
			#pragma omp parallel
			#endif
				{
					// Private scatter table, kept all-sentinel between nodes so each
					// node costs O(L1) to fill and O(L1) to clear.
					std::vector<uint32_t> lp((size_t)globalMaxLabel + 1, kNoLabelPos);
			#ifdef LBP_USE_OPENMP
			#pragma omp for schedule(dynamic, 256)
			#endif
					for (int_t u = 0; u < (int_t)numNodes; ++u) {
						const uint32_t labOfs = nodeLabelOfs[u];
						const size_t L1 = nodeLabelOfs[u+1] - labOfs;
						const uint32_t inOfs = nodeInEdgeOfs[u];
						const size_t deg = nodeInEdgeOfs[u+1] - inOfs;
						if (L1 == 0 || deg == 0)
							continue;
						const LabelID* __restrict labels1 = nodeLabels.data() + labOfs;
						// backwards, so the LOWEST k wins on duplicate labels --
						// matching the first-match "break" of the original scan
						for (size_t k = L1; k-- > 0; )
							lp[labels1[k]] = (uint32_t)k;
						for (size_t t = 0; t < deg; ++t) {
							const EdgeID ie = nodeInEdges[inOfs + t];
							const EdgeID eid = ie ^ 1;
							const uint32_t msgBase = offs[eid];
							const size_t L2 = offs[eid+1] - msgBase;
							const uint32_t labOfs2 = nodeLabelOfs[edges[ie].nodeID1];
							const LabelID* __restrict labels2 = nodeLabels.data() + labOfs2;
							for (size_t j = 0; j < L2; ++j) {
								const uint32_t k = lp[labels2[j]];
								msgLabelXRef[msgBase + j] =
									(k == kNoLabelPos) ? kNoLabelPos16 : (uint16_t)k;
							}
						}
						for (size_t k = 0; k < L1; ++k)
							lp[labels1[k]] = kNoLabelPos;
					}
				}
			}
		}

		topologyPrepared = true;
	}

	inline void SetNumNodes(NodeID nNodes) {
		nodes.resize(nNodes);
	}
	inline NodeID GetNumNodes() const {
		return (NodeID)nodes.size();
	}

	inline void SetNeighbors(NodeID nodeID1, NodeID nodeID2) {
		// Only the edge pair is recorded; the per-node incoming lists are derived
		// from it in PrepareTopology (see nodeInEdges).
		edges.emplace_back(nodeID1, nodeID2);
		edges.emplace_back(nodeID2, nodeID1);
	}

	// Phase 1 of the two-phase data-cost build: the exact number of labels each node
	// will be given. Must be called once, before any SetDataCost, and after the node
	// count is set. counts must have GetNumNodes() entries.
	void ReserveDataCosts(const uint32_t* counts) {
		const size_t n = nodes.size();
		nodeLabelOfs.resize(n + 1);
		size_t total = 0;
		for (size_t i = 0; i < n; ++i) {
			nodeLabelOfs[i] = (uint32_t)total;
			total += counts[i];
		}
		ASSERT(total <= (size_t)std::numeric_limits<uint32_t>::max());
		nodeLabelOfs[n] = (uint32_t)total;
		nodeLabels.resize(total);
		nodeDataCosts.resize(total);
		labelFill.assign(n, 0);
	}

	// Phase 2. Appends in call order, exactly as the old push_back did, so the
	// within-node label order (and hence every downstream float summation order) is
	// unchanged.
	inline void SetDataCost(LabelID label, NodeID nodeID, EnergyType cost) {
		Node& node = nodes[nodeID];
		const uint32_t slot = nodeLabelOfs[nodeID] + labelFill[nodeID]++;
		ASSERT(slot < nodeLabelOfs[nodeID+1]); // ReserveDataCosts under-counted this node
		nodeLabels[slot] = label;
		const EnergyType dataCost(cost);
		nodeDataCosts[slot] = dataCost;
		if (dataCost < node.dataCost) {
			node.label = label;
			node.dataCost = dataCost;
		}
	}
	inline void SetDataCost(LabelID label, const DataCost& cost) {
		SetDataCost(label, cost.nodeID, cost.cost);
	}
	inline void SetDataCosts(LabelID label, const std::vector<DataCost>& costs) {
		for (const DataCost& cost: costs)
			SetDataCost(label, cost);
	}

	inline void SetSmoothCost(FncSmoothCost func) {
		fncSmoothCost = func;
	}

	// Declare that fncSmoothCost is a (generalized) Potts model so Optimize()
	// can use the O(L1+L2) message fast path (see bPottsSmoothness).
	inline void SetPottsSmoothness(bool b) {
		bPottsSmoothness = b;
	}

	EnergyType ComputeEnergy() const {
		EnergyType energy(0);
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for reduction(+:energy)
		#endif
		for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID)
			energy += nodes[nodeID].dataCost;
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for reduction(+:energy)
		#endif
		for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
			const DirectedEdge& edge = edges[edgeID];
			energy += fncSmoothCost(edge.nodeID1, edge.nodeID2, nodes[edge.nodeID1].label, nodes[edge.nodeID2].label);
		}
		return energy;
	}

	// Diagnostic breakdown of the same objective ComputeEnergy() sums up.
	// The total alone cannot tell the two failure modes apart:
	//  - data:   how far down its quality ranking each face's chosen view sits.
	//            Excess here is a REAL fidelity loss (more oblique / lower
	//            resolution source), which seam leveling cannot undo.
	//  - smooth: the Potts penalty for neighbours picking different views, i.e.
	//            patch fragmentation and seam length -- largely cosmetic, and
	//            what the later seam leveling / blending pass exists to hide.
	//  - numUndefined: nodes still on label 0, which become NO_ID faces and are
	//            left untextured. Should be only faces with no candidate view;
	//            if it is still falling late in the solve, stopping early is
	//            costing real coverage, not just view-selection optimality.
	// Accumulated in double: EnergyType is float, and at a total near 1e10 its
	// ULP is ~1024, i.e. bigger than a single node's contribution.
	//
	// This re-evaluates fncSmoothCost RAW -- it does not reuse the scrubbed
	// edgeWeight cache -- so it stays an honest evaluation of the model as the
	// caller defined it, and a "smooth=-nan while data=<finite>" line remains the
	// signal that fncSmoothCost is returning non-finite values (bad face normals).
	// PrepareTopology's warning names the count; this line is the symptom.
	struct EnergySplit {
		double data = 0;
		double smooth = 0;
		int_t numUndefined = 0;
	};

	EnergySplit ComputeEnergySplit() const {
		double dataE(0), smoothE(0);
		int_t nUndefined(0);
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for reduction(+:dataE,nUndefined)
		#endif
		for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID) {
			const Node& node = nodes[nodeID];
			dataE += (double)node.dataCost;
			if (node.label == 0)
				++nUndefined;
		}
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for reduction(+:smoothE)
		#endif
		for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
			const DirectedEdge& edge = edges[edgeID];
			smoothE += (double)fncSmoothCost(edge.nodeID1, edge.nodeID2, nodes[edge.nodeID1].label, nodes[edge.nodeID2].label);
		}
		EnergySplit split;
		split.data = dataE;
		split.smooth = smoothE;
		split.numUndefined = nUndefined;
		return split;
	}

	// Two aligned buffers for ping-pong messaging
	std::vector<EnergyType, AlignedAllocator<EnergyType, 32>> msgBuf[2];
	// uint32, not size_t. This is indexed randomly in the hot loop, once per directed
	// edge, so its width is real bandwidth: at 8.2M edges size_t cost 66 MB against
	// 33 MB here. The element count it addresses is sum(labels per edge) -- ~72M on a
	// 2.75M-face scene -- and uint32 still covers 4.29e9 elements, i.e. a 17 GB message
	// buffer, far past anything that fits in RAM. PrepareMessageBuffers asserts it.
	// It also pairs well with the e^1 reverse-edge trick: offs[e] and offs[e^1] are
	// 4 bytes apart and a pair is 8-byte aligned, so both come from one cache line.
	std::vector<uint32_t> msgOffset;
	bool buffersInitialized = false;
	size_t msgParity = 0; // 0 = read buf0, 1 = write buf1

	inline size_t EdgeMsgLen(size_t e) noexcept {
		return NumLabels(edges[e].nodeID2);
	}

	// (There was an edgeMsgLen array here. Its only consumer was the standalone
	// message-normalization pass, which is now fused into the emit loop -- where the
	// length is already in hand from the label CSR -- so the array and the 16 MB it
	// occupied are gone.)

	// Per-thread scratch for Optimize(), one row per OpenMP thread.
	// Lifetime is tied to this object (freed on destruction and by
	// ReleaseScratch()), unlike static thread_local which would linger
	// for the whole lifetime of the pooled worker threads.
	std::vector<std::vector<EnergyType>> tlSumAll;
	std::vector<std::vector<EnergyType>> tlEnergyBuf;

	static const uint32_t kNoLabelPos = (uint32_t)-1;
	static const uint16_t kNoLabelPos16 = (uint16_t)-1;

	// Precomputed label cross-reference, parallel to the message buffer: for the
	// directed edge e and its j-th message slot, msgLabelXRef[msgOffset[e]+j] is the
	// index k within the SOURCE node's label list of the label that the destination's
	// j-th label refers to, or kNoLabelPos16 if the source does not offer it.
	//
	// This mapping depends only on the label sets, which never change during the
	// solve, so recomputing it every sweep was pure waste -- and expensive waste: the
	// Potts path used to reach edges[ie].nodeID1 -> nodeLabelOfs[v] -> nodeLabels[..]
	// (three DEPENDENT random loads, so their latencies serialize) and additionally
	// build and tear down a per-node scatter table. All of that is now one sequential
	// read at an offset the loop has already loaded for the message write itself.
	// Costs ~2 bytes per message element (~144 MB at 72M elements); only built when
	// bPottsSmoothness, since the general path needs the real label values anyway.
	std::vector<uint16_t> msgLabelXRef;

	// Release the per-thread scratch buffers (called once optimization
	// finishes, so the memory does not outlive the last Optimize call).
	void ReleaseScratch() {
		tlSumAll.clear();
		tlSumAll.shrink_to_fit();
		tlEnergyBuf.clear();
		tlEnergyBuf.shrink_to_fit();
	}

	// Release the message buffers and cached topology once the sweeps are done.
	// Only finalLabels (plus nodes/edges, which ComputeEnergy still needs) matter
	// after the solve, while msgBuf alone is 2x the message-buffer size reported by
	// [LBP-DIAG] -- gigabytes on a full-resolution mesh. Everything freed here is
	// rebuilt on demand by PrepareMessageBuffers()/PrepareTopology(), so calling
	// Optimize() again on the same object stays correct (just pays the rebuild).
	void ReleaseSolveBuffers() {
		msgBuf[0].clear();
		msgBuf[0].shrink_to_fit();
		msgBuf[1].clear();
		msgBuf[1].shrink_to_fit();
		msgOffset.clear();
		msgOffset.shrink_to_fit();
		edgeWeight.clear();
		edgeWeight.shrink_to_fit();
		msgLabelXRef.clear();
		msgLabelXRef.shrink_to_fit();
		// Topology, rebuilt by PrepareTopology. The label/data-cost CSR is NOT freed
		// here: unlike these, it is caller-supplied and cannot be regenerated.
		nodeInEdgeOfs.clear();
		nodeInEdgeOfs.shrink_to_fit();
		nodeInEdges.clear();
		nodeInEdges.shrink_to_fit();
		buffersInitialized = false;
		topologyPrepared = false;
		msgParity = 0; // rebuilt buffers start zeroed, so read from buf0 again
	}

	void PrepareMessageBuffers()
	{
		const size_t numEdges = edges.size();

		// ReserveDataCosts must have run: the label CSR is what sizes every message.
		ASSERT(nodeLabelOfs.size() == nodes.size() + 1);

		// numEdges+1: the trailing sentinel makes offs[e+1]-offs[e] the length of edge
		// e's message. That is why there is no per-edge length array any more, and --
		// more importantly -- why the hot loop no longer has to reach through
		// edges[e].nodeID2 into nodeLabelOfs to find it, which was a chain of two
		// dependent random loads per edge.
		msgOffset.resize(numEdges + 1);

		size_t total = 0;
		for (size_t e = 0; e < numEdges; ++e) {
			const size_t L = NumLabels(edges[e].nodeID2);

			msgOffset[e] = (uint32_t)total;
			total += L;
		}
		msgOffset[numEdges] = (uint32_t)total;
		// msgOffset is uint32: see its declaration. 4.29e9 elements is a 17 GB message
		// buffer, so this cannot trip on any graph that fits in memory -- but it is the
		// one assumption that would corrupt silently, so it is checked rather than
		// assumed.
		ASSERT(total <= (size_t)std::numeric_limits<uint32_t>::max());

		msgBuf[0].assign(total, EnergyType(0));
		msgBuf[1].assign(total, EnergyType(0));
		buffersInitialized = true;
	} 

// -----------------------------------------------------------------
	// One sweep. The labeling is FUSED into the accumulation: the quantity the old
	// separate labeling pass minimized, data[j] + sum of incoming messages, is exactly
	// the `sum` this loop already builds, so that pass was a second full traversal of
	// the message buffer and of every CSR stream to recompute what was in registers.
	//
	// The consequence is a one-sweep shift: the labels written during sweep t come
	// from the messages of sweep t-1, and `changed` likewise reports the previous
	// sweep's flips. Optimize() closes the gap with a final bEmitMessages=false call,
	// which does the accumulation and argmin without emitting -- so the labels it ends
	// with are identical to the old code's, at N+1 message-buffer traversals instead
	// of 2N. The convergence test now trips one sweep later, which is the only visible
	// difference and costs one extra (much cheaper) sweep.
	int Optimize(unsigned numIterations /* always 1 */, bool bEmitMessages = true)
	{
		const uint32_t* __restrict offs = msgOffset.data();

		int changed = 0;

		// ------------------------------------------------------------
		// Message passing iterations
		// ------------------------------------------------------------
		EnergyType* __restrict readMsgs = msgBuf[msgParity].data();
		EnergyType* __restrict writeMsgs = msgBuf[msgParity ^ 1].data();

		// Ensure one scratch row per thread (allocated on first use, reused
		// on later Optimize calls, freed by ReleaseScratch()).
	#ifdef LBP_USE_OPENMP
		const int numThreads = omp_get_max_threads();
	#else
		const int numThreads = 1;
	#endif
		if ((int)tlSumAll.size() < numThreads) {
			tlSumAll.resize(numThreads);
			tlEnergyBuf.resize(numThreads);
		}

	#ifdef LBP_USE_OPENMP
	#pragma omp parallel
	#endif
		{
			// ------------------------------------------------------------
			// Per-thread scratch (persists across Optimize calls, owned by
			// this object). Each thread indexes its own row => no race.
			// ------------------------------------------------------------
	#ifdef LBP_USE_OPENMP
			const int tid = omp_get_thread_num();
	#else
			const int tid = 0;
	#endif
			std::vector<EnergyType>& sumAll = tlSumAll[tid];
			std::vector<EnergyType>& energyBuf = tlEnergyBuf[tid];

			// per-node scratch
			if (sumAll.size() < maxLabelsPerNode)
				sumAll.resize(maxLabelsPerNode);

			if (energyBuf.size() < maxLabelsPerNode)
				energyBuf.resize(maxLabelsPerNode);

	#ifdef LBP_USE_OPENMP
	#pragma omp for schedule(dynamic, 256) reduction(+:changed) nowait // Better than static
	#endif
			for (int_t u = 0; u < (int_t)nodes.size(); ++u) {
				// Contiguous streams walked in node order -- no Node struct is touched
				// until the argmin writes the label (see the Node comment).
				const uint32_t labOfs = nodeLabelOfs[u];
				const size_t L1 = nodeLabelOfs[u+1] - labOfs;
				const uint32_t inOfs = nodeInEdgeOfs[u];
				const size_t deg = nodeInEdgeOfs[u+1] - inOfs;
				if (L1 == 0)
					continue;

				const LabelID* __restrict labels1 = nodeLabels.data() + labOfs;
				const EnergyType* __restrict dataCosts1 = nodeDataCosts.data() + labOfs;
				const EdgeID* __restrict inEdges1 = nodeInEdges.data() + inOfs;

				// add all incoming ONCE
				// We fuse as we can.
				EnergyType* __restrict sum = sumAll.data();

				// Fast path: deg == 3 (almost always)
				if (deg == 3) {
					const EnergyType* __restrict u = dataCosts1;

					const EnergyType* __restrict p0 = readMsgs + offs[inEdges1[0]];
					const EnergyType* __restrict p1 = readMsgs + offs[inEdges1[1]];
					const EnergyType* __restrict p2 = readMsgs + offs[inEdges1[2]];
					EnergyType* __restrict out = sum;

					size_t k = 0;
					for (; k + 3 < L1; k += 4) {
						out[0] = u[0] + p0[0] + p1[0] + p2[0];
						out[1] = u[1] + p0[1] + p1[1] + p2[1];
						out[2] = u[2] + p0[2] + p1[2] + p2[2];
						out[3] = u[3] + p0[3] + p1[3] + p2[3];

						u += 4;
						p0 += 4;
						p1 += 4;
						p2 += 4;
						out += 4;
					}

					for (; k < L1; ++k) {
						*out++ = *u++ + *p0++ + *p1++ + *p2++;
					}
				}
				else {
					// Generic fallback: start from unary
					memcpy(sum, dataCosts1, L1 * sizeof(EnergyType));

					// Accumulate all incoming messages
					for (size_t t = 0; t < deg; ++t) {
						const EnergyType* __restrict msg = readMsgs + offs[inEdges1[t]];
						for (size_t k = 0; k < L1; ++k)
							sum[k] += msg[k];
					}
				}

				// --------------------------------------------------------
				// Labeling, fused. `sum` IS data[j] + all incoming messages, the
				// exact quantity the old separate pass re-derived by re-reading the
				// whole message buffer and every CSR stream a second time. Doing it
				// here costs an argmin over ~L1 floats already in registers.
				// --------------------------------------------------------
				{
					Node& node = nodes[u];
					const LabelID oldLabel = node.label;

					EnergyType bestE = std::numeric_limits<EnergyType>::max();
					size_t bestIdx = 0;
					LabelID bestL = labels1[0];

					for (size_t j = 0; j < L1; ++j) {
						if (sum[j] < bestE) {
							bestE = sum[j];
							bestIdx = j;
							bestL = labels1[j];
						}
					}

					node.label = bestL;
					node.dataCost = dataCosts1[bestIdx];
					if (bestL != oldLabel)
						++changed;
				}

				// A node with no incoming edge has no outgoing one either (the two
				// sets are the same list XOR 1), so there is nothing to emit. It was
				// still labeled above, which is what the old separate pass did for it.
				if (!bEmitMessages || deg == 0)
					continue;

				// --------------------------------------------------------
				// Emit message for each outgoing edge
				// --------------------------------------------------------
				// Iterate the INCOMING edges and flip each one, instead of walking a
				// separate outEdges list. reverse(e) == e^1 (see the topology comment
				// on why that holds unconditionally), which buys three things:
				//   - outEdges is not built or stored at all;
				//   - the message v->u that has to be subtracted for the edge u->v is
				//     the very edge being iterated, so the O(deg) search that used to
				//     find it -- with a random edges[pe] load per probe, up to 9 per
				//     node -- disappears, and with it the `sub == nullptr` branch,
				//     which the pairing invariant makes unreachable;
				//   - offs[ie]/offs[ie^1], edges[ie]/edges[ie^1] and
				//     edgeWeight[ie]/edgeWeight[ie^1] are adjacent and pair-aligned, so
				//     each lands in a single cache line.
				for (size_t t = 0; t < deg; ++t) {
					const EdgeID ie = inEdges1[t];           // ie:  v -> u
					const EdgeID eid = ie ^ 1;               // eid: u -> v
					// The message length comes from the offset array's sentinel, so the
					// Potts path never has to touch edges[] or the destination node's
					// label list at all -- that was a three-deep dependent random-load
					// chain (edges -> nodeLabelOfs -> nodeLabels) per edge per sweep.
					const uint32_t msgBase = offs[eid];
					const size_t L2 = offs[eid+1] - msgBase;

					// message v -> u: the edge we are standing on, no search needed
					const EnergyType* __restrict sub = readMsgs + offs[ie];

					// energyBuf = sumAll - sub
					for (size_t k = 0; k < L1; ++k) {
						energyBuf[k] = sumAll[k] - sub[k];
					}

					EnergyType* __restrict msgOut = writeMsgs + msgBase;

					// Normalization (subtract the per-edge minimum) is FUSED into the
					// writes below rather than done by a second pass over the whole
					// message buffer. That pass read and wrote all 242 MB of it every
					// sweep to touch values that were live in registers here moments
					// earlier; msgOut is ~L2 floats and is hot in L1 at this point. The
					// running min is exact and order-independent, and the subtraction is
					// applied to the same values in the same order, so the result is
					// bit-identical to the separate pass.
					EnergyType minMsg = std::numeric_limits<EnergyType>::max();

					// Potts fast path: pairwise cost is 0 when l1==l2 and a single
					// per-edge constant W otherwise (independent of the label values),
					// so the message min-convolution is
					//   msg(l2) = min( min_k energyBuf[k] + W,  energyBuf[k : label==l2] )
					// This collapses O(L1*L2) to O(L1+L2) and evaluates fncSmoothCost
					// (a normal dot-product) ONCE per edge instead of L1*L2 times -- the
					// actual hot spot. Result is bit-identical to the general loop below.
					if (bPottsSmoothness) {
						const EnergyType W = edgeWeight[eid];
						EnergyType minAll = energyBuf[0];
						for (size_t k = 1; k < L1; ++k)
							if (energyBuf[k] < minAll) minAll = energyBuf[k];
						const EnergyType minPlusW = minAll + W;
						// One sequential read at an offset already loaded for msgOut,
						// instead of rebuilding a scatter table per node and chasing
						// the destination's labels through two random loads per edge.
						const uint16_t* __restrict xr = msgLabelXRef.data() + msgBase;
						for (size_t j = 0; j < L2; ++j) {
							const uint16_t k = xr[j];
							EnergyType best = minPlusW;
							// kNoLabelPos16 means node u has no such label, i.e.
							// the old scan would have fallen through without a match
							if (k != kNoLabelPos16 && energyBuf[k] < best)
								best = energyBuf[k];
							msgOut[j] = best;
							if (best < minMsg) minMsg = best;
						}
					}
					else {
						// General pairwise cost: O(L1*L2). Only this path needs the real
						// label values, so it pays for the neighbour lookup itself.
						const NodeID idxV = edges[ie].nodeID1;
						const LabelID* __restrict labels2 = nodeLabels.data() + nodeLabelOfs[idxV];
						for (size_t j = 0; j < L2; ++j) {
							const LabelID l2 = labels2[j];
							EnergyType best = std::numeric_limits<EnergyType>::max();

							for (size_t k = 0; k < L1; ++k) {
								const LabelID l1 = labels1[k];
								const EnergyType v = fncSmoothCost((NodeID)u, idxV, l1, l2);
								const EnergyType e = energyBuf[k] + v;
								if (e < best) {
									best = e;
								}
							}

							msgOut[j] = best;
							if (best < minMsg) minMsg = best;
						}
					}

					// L2 == 0 leaves minMsg at max and this loop empty, matching what
					// the old standalone pass did for a zero-length message.
					for (size_t j = 0; j < L2; ++j)
						msgOut[j] -= minMsg;
				}

			}

			// ------------------------------------------------------------
			// Flip buffers
			// ------------------------------------------------------------
			// bEmitMessages is a plain function parameter, identical in every thread,
			// so the `single` below is either encountered by all of them or by none --
			// which is what OpenMP requires of a worksharing construct. The implicit
			// barrier at the end of `single` is also what makes the `nowait` on the
			// sweep loop safe: it guarantees every emit has landed before the flip is
			// read. When nothing was emitted there is no flip to make, and the implicit
			// barrier closing the parallel region covers the labeling writes.
			if (bEmitMessages) {
	#ifdef LBP_USE_OPENMP
	#pragma omp single
	#endif
				{
					msgParity ^= 1;
				}
			}

			// Two full passes used to live here: a message-normalization sweep and a
			// final labeling sweep, each re-reading the entire message buffer (and the
			// labeling, every CSR stream as well). Normalization is fused into the emit
			// above; labeling is fused into the accumulation above. Nothing is left.
		} // omp parallel

		return changed;
	}

	EnergyType Optimize() 
	{
		if (!buffersInitialized)
			PrepareMessageBuffers();

		PrepareTopology();

		unsigned maxIters = LBP_MAX_ITERS;
		int lastChanged = INT_MAX;
		unsigned stall = 0;

		// [LBP-DIAG] perf diagnostic: how many iterations run, the per-iteration
		// "changed" trajectory, and the message-buffer size (the bandwidth the message
		// passing must stream each iteration). Entirely observational, so the trace
		// buffer is not even allocated unless the gate is open.
		const bool bDiag = TEXTURE_DIAG_ENABLED();
		unsigned itersRun = 0;
		const unsigned kTraceMax = maxIters;
		std::vector<int> changedTrace(bDiag ? kTraceMax : 0u, 0);

		for (unsigned it = 0; it < maxIters; ++it) {
			int changed = Optimize(1);
			if (bDiag && it < kTraceMax) changedTrace[it] = changed;
			itersRun = it + 1;

			// Track the OBJECTIVE every few sweeps, not just the flip count: with
			// dozens of near-tied labels per node the tail sweeps can keep flipping
			// marginal ties (or oscillate between them) long after the energy has
			// gone flat. Energy is what decides whether raising LBP_MAX_ITERS buys
			// any real quality. O(nodes+edges) -- negligible next to a sweep, but it
			// runs ~10x per solve and feeds nothing but this line, so it is gated.
			//
			// Note the labeling lags the messages by one sweep (see the fusion comment
			// on Optimize(unsigned,bool)), so this line reports the energy of iteration
			// it-1's messages. The trajectory is what it is read for, and the catch-up
			// call after the loop means the FINAL labeling is not shifted.
			if (bDiag && (it == 0 || (it % 5) == 4)) {
				const EnergySplit es = ComputeEnergySplit();
				TEXTURE_DIAG("[LBP-DIAG] iter=%u changed=%d energy=%.9g data=%.9g smooth=%.9g undefined=%lld",
					itersRun, changed, es.data + es.smooth, es.data, es.smooth,
					(long long)es.numUndefined);
			}

			if (changed < nodes.size() * LBP_CONVERGENCE_FRAC)
				break;

			if (changed >= lastChanged) {
				if (++stall >= 5)
					break;          // oscillating / stalled
			}
			else {
				stall = 0;
			}

			lastChanged = changed;
		}

		// Catch-up labeling. The sweeps label from the PREVIOUS sweep's messages (see
		// Optimize(unsigned,bool)), so at this point the labels are one set behind the
		// messages that were last emitted. This applies them: accumulation + argmin,
		// no emit. It is the ONLY extra pass the fusion costs, against one saved per
		// sweep, and it leaves the labeling identical to what the old separate pass
		// produced after the same number of sweeps.
		Optimize(1, false);

		if (bDiag) {
			// total labels over all nodes + message-buffer element count (== sum of
			// per-edge label counts). This is what the memory-bound cost scales with.
			size_t totalLabels = 0;
			totalLabels = nodeLabels.size();
			const size_t msgElems = buffersInitialized ? msgBuf[0].size() : 0;
			TEXTURE_DIAG("[LBP-DIAG] nodes=%zu edges=%zu iters=%u avgLabels=%.2f msgBufMB=%.1f (x2)",
				nodes.size(), edges.size(), itersRun,
				nodes.empty() ? 0.0 : (double)totalLabels / (double)nodes.size(),
				(double)(msgElems * sizeof(EnergyType)) / (1024.0*1024.0));
			char buf[512]; int off = 0;
			off += snprintf(buf+off, sizeof(buf)-off, "[LBP-DIAG] changed/iter:");
			for (unsigned i = 0; i < itersRun && i < kTraceMax && off < (int)sizeof(buf)-16; ++i)
				off += snprintf(buf+off, sizeof(buf)-off, " %d", changedTrace[i]);
			TEXTURE_DIAG("%s", buf);
		}

		// Build compact label array for fast external access
		finalLabels.resize(nodes.size());

		for (size_t i = 0, cnt = (size_t) nodes.size(); i < cnt; ++i)
			finalLabels[i] = nodes[i].label;

		// Optimization done: release the per-thread scratch so it does not
		// outlive the last Optimize call, and drop the message buffers/topology
		// now that finalLabels holds everything the caller needs. Must come after
		// the [LBP-DIAG] block above, which reads msgBuf[0].size().
		ReleaseScratch();
		ReleaseSolveBuffers();

		// Still computed unconditionally: it is this function's RETURN VALUE, not a
		// diagnostic, and it is one O(nodes+edges) pass against ~50 sweeps.
		const EnergyType finalEnergy = ComputeEnergy();
		// Final MRF energy: the objective quality of the labeling. Compare across
		// LBP_MAX_ITERS values -- if energy@25 is within a fraction of a percent of
		// energy@50, the shorter budget costs essentially no view-selection quality.
		TEXTURE_DIAG("[LBP-DIAG] finalEnergy=%.6g", (double)finalEnergy);
		return finalEnergy;
	}

	inline LabelID GetLabel(NodeID nodeID) const {
		return nodes[nodeID].label;
	}
};
/*----------------------------------------------------------------*/
} // namespace SEACAVE

#endif // __SEACAVE_LBP_H__
