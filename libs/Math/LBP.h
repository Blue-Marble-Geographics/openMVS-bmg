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
  enum { kApproxMaxNumSharedViews = 32 }; // Too low and we reallocate; too high and we waste memory and slow down too.

public:
	struct DirectedEdge {
		NodeID nodeID1;
		NodeID nodeID2;
		boost::container::small_vector<EnergyType, kApproxMaxNumSharedViews> newMsgs;
		boost::container::small_vector<EnergyType, kApproxMaxNumSharedViews> oldMsgs;
		inline DirectedEdge(NodeID _nodeID1, NodeID _nodeID2) : nodeID1(_nodeID1), nodeID2(_nodeID2) {}
	};

	struct Node {
		LabelID label;
		EnergyType dataCost;
		boost::container::small_vector<LabelID, kApproxMaxNumSharedViews> labels;
		boost::container::small_vector<EnergyType, kApproxMaxNumSharedViews> dataCosts;
		boost::container::small_vector<EdgeID, 3> incomingEdges; // Frequently 3 in length.
		inline Node() : label(0), dataCost(MaxEnergy) {}
	};

	std::vector<DirectedEdge> edges;
	std::vector<Node> nodes;
	std::vector<LabelID> finalLabels;
	FncSmoothCost fncSmoothCost;
	// When true, fncSmoothCost is assumed to be a (generalized) Potts model:
	// cost is 0 for l1==l2 and a single per-edge constant otherwise (independent
	// of the specific label values). Enables the O(L1+L2) message fast path in
	// Optimize() that skips the L1*L2 fncSmoothCost evaluations. Leave false for
	// arbitrary pairwise costs. Bit-identical to the general path when the model
	// really is Potts.
	bool bPottsSmoothness = false;

	// Cached topology
	std::vector<boost::container::small_vector<EdgeID, 3>> outEdges;
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

		// Build outEdges ONCE
		outEdges.assign(nodes.size(), {});
		for (EdgeID e = 0; e < edges.size(); ++e)
			outEdges[edges[e].nodeID1].push_back(e);

		// Compute label bounds ONCE
		globalMaxLabel = 0;
		maxLabelsPerNode = 0;

		for (const Node& n : nodes) {
			maxLabelsPerNode = std::max(maxLabelsPerNode, n.labels.size());
			for (LabelID l : n.labels)
				globalMaxLabel = std::max(globalMaxLabel, l);
		}

		// Precompute the constant per-edge Potts weight ONCE (fncSmoothCost with
		// any two distinct labels). Removes ~numIterations redundant evaluations
		// (each a random faceNormals[] fetch + dot product) per directed edge.
		if (bPottsSmoothness) {
			edgeWeight.resize(edges.size());
			for (EdgeID e = 0; e < edges.size(); ++e)
				edgeWeight[e] = fncSmoothCost(edges[e].nodeID1, edges[e].nodeID2, 0, 1);
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
		EdgeID e0 = (EdgeID)edges.size();

		edges.emplace_back(nodeID1, nodeID2);
		edges.emplace_back(nodeID2, nodeID1);

		nodes[nodeID2].incomingEdges.push_back(e0);
		nodes[nodeID1].incomingEdges.push_back(e0 + 1);
	}

	inline void SetDataCost(LabelID label, NodeID nodeID, EnergyType cost) {
		Node& node = nodes[nodeID];
		node.labels.push_back(label);
		const EnergyType dataCost(cost);
		node.dataCosts.push_back(dataCost);
		if (dataCost < node.dataCost) {
			node.label = label;
			node.dataCost = dataCost;
		}
		for (EdgeID edgeID: node.incomingEdges) {
			DirectedEdge& incomingEdge = edges[edgeID];
			incomingEdge.oldMsgs.push_back(0);
			incomingEdge.newMsgs.push_back(0);
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

	// Two aligned buffers for ping-pong messaging
	std::vector<EnergyType, AlignedAllocator<EnergyType, 32>> msgBuf[2];
	std::vector<size_t> msgOffset;
	bool buffersInitialized = false;
	size_t msgParity = 0; // 0 = read buf0, 1 = write buf1

	inline size_t EdgeMsgLen(size_t e) noexcept {
		return nodes[edges[e].nodeID2].labels.size();
	}

	std::vector<uint16_t> edgeMsgLen; // uint16_t is usually enough

	// Per-thread scratch for Optimize(), one row per OpenMP thread.
	// Lifetime is tied to this object (freed on destruction and by
	// ReleaseScratch()), unlike static thread_local which would linger
	// for the whole lifetime of the pooled worker threads.
	std::vector<std::vector<EnergyType>> tlSumAll;
	std::vector<std::vector<EnergyType>> tlEnergyBuf;

	// Release the per-thread scratch buffers (called once optimization
	// finishes, so the memory does not outlive the last Optimize call).
	void ReleaseScratch() {
		tlSumAll.clear();
		tlSumAll.shrink_to_fit();
		tlEnergyBuf.clear();
		tlEnergyBuf.shrink_to_fit();
	}

	void PrepareMessageBuffers()
	{
		const size_t numEdges = edges.size();

		msgOffset.resize(numEdges);
		edgeMsgLen.resize(numEdges);

		size_t total = 0;
		for (size_t e = 0; e < numEdges; ++e) {
			const size_t L =
				nodes[edges[e].nodeID2].labels.size();

			msgOffset[e] = total;
			edgeMsgLen[e] = (uint16_t)L;
			total += L;
		}

		msgBuf[0].assign(total, EnergyType(0));
		msgBuf[1].assign(total, EnergyType(0));
		buffersInitialized = true;
	} 

// -----------------------------------------------------------------
	int Optimize(unsigned numIterations /* always 1 */)
	{
		const size_t* __restrict offs = msgOffset.data();
		const EnergyType maxE = (EnergyType)LBPInference::MaxEnergy;

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
	#pragma omp for schedule(dynamic, 256) nowait // Better than static
	#endif
			for (int_t u = 0; u < (int_t)nodes.size(); ++u) {
				const Node& n1 = nodes[u];
				const auto& labels1 = n1.labels;
				const auto& inEdges1 = n1.incomingEdges;
				const auto& outE = outEdges[u];

				const size_t L1 = labels1.size();
				if (L1 == 0 || outE.empty())
					continue;

				// add all incoming ONCE
				// We fuse as we can.
				EnergyType* __restrict sum = sumAll.data();
				const size_t deg = inEdges1.size();

				// Fast path: deg == 3 (almost always)
				if (deg == 3) {
					const EnergyType* __restrict u = n1.dataCosts.data();

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
					memcpy(sum, n1.dataCosts.data(), L1 * sizeof(EnergyType));

					// Accumulate all incoming messages
					for (EdgeID eid : inEdges1) {
						const EnergyType* __restrict msg = readMsgs + offs[eid];
						for (size_t k = 0; k < L1; ++k)
							sum[k] += msg[k];
					}
				}

				// --------------------------------------------------------
				// Emit message for each outgoing edge
				// --------------------------------------------------------
				for (EdgeID eid : outE) {
					const DirectedEdge& edge = edges[eid];
					const Node& n2 = nodes[edge.nodeID2];
					const LabelID* __restrict labels2 = n2.labels.data();
					const size_t L2 = (size_t)edgeMsgLen[eid];

					// Find message from v -> u (subtraction term)
					const EnergyType* __restrict sub = nullptr;
					for (EdgeID pe : inEdges1) {
						if (edges[pe].nodeID1 == edge.nodeID2) {
							sub = readMsgs + offs[pe];
							break;
						}
					}

					// energyBuf = sumAll - sub
					if (sub) {
						for (size_t k = 0; k < L1; ++k) {
							energyBuf[k] = sumAll[k] - sub[k];
						}
					}
					else {
						memcpy(energyBuf.data(), sumAll.data(), L1 * sizeof(EnergyType));
					}

					EnergyType* __restrict msgOut = writeMsgs + offs[eid];

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
						for (size_t j = 0; j < L2; ++j) {
							const LabelID l2 = labels2[j];
							EnergyType best = minPlusW;
							for (size_t k = 0; k < L1; ++k) {
								if (labels1[k] == l2) {
									if (energyBuf[k] < best)
										best = energyBuf[k];
									break;
								}
							}
							msgOut[j] = best;
						}
						continue;
					}

					// General pairwise cost: O(L1*L2)
					// This is correct for any fncSmoothCost, including your SmoothnessPottsStrong.
					for (size_t j = 0; j < L2; ++j) {
						const LabelID l2 = labels2[j];
						EnergyType best = std::numeric_limits<EnergyType>::max();

						for (size_t k = 0; k < L1; ++k) {
							const LabelID l1 = labels1[k];
							const EnergyType v = fncSmoothCost(edge.nodeID1, edge.nodeID2, l1, l2);
							const EnergyType e = energyBuf[k] + v;
							if (e < best) {
								best = e;
							}
						}

						msgOut[j] = best;
					}
				}
			}

			// ------------------------------------------------------------
			// Flip buffers
			// ------------------------------------------------------------
	#ifdef LBP_USE_OPENMP
	#pragma omp single
	#endif
			{
				msgParity ^= 1;
			}

	#ifdef LBP_USE_OPENMP
	#pragma omp barrier   // REQUIRED
	#endif

			// Restore for stability
			// ------------------------------------------------------------
			// Normalize messages
			// ------------------------------------------------------------
			EnergyType* __restrict normMsgs = msgBuf[msgParity].data();

#ifdef LBP_USE_OPENMP
#pragma omp for schedule(static)
#endif
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				EnergyType* __restrict m = normMsgs + offs[edgeID];
				const size_t L = edgeMsgLen[edgeID];

				EnergyType minMsg = std::numeric_limits<EnergyType>::max();
				for (size_t k = 0; k < L; ++k) {
					if (m[k] < minMsg) {
						minMsg = m[k];
					}
				}
				for (size_t k = 0; k < L; ++k) {
					m[k] -= minMsg;
				}
			}


#ifdef LBP_USE_OPENMP
#pragma omp barrier
#endif

			// ------------------------------------------------------------
			// Final labeling
			// ------------------------------------------------------------
			EnergyType* __restrict msgs = msgBuf[msgParity].data();

	#ifdef LBP_USE_OPENMP
	#pragma omp for schedule(static) reduction(+:changed)
	#endif
			for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID) {
				Node& node = nodes[nodeID];
				const LabelID oldLabel = node.label;
				const auto& inEdges = node.incomingEdges;
				const EnergyType* __restrict data = node.dataCosts.data();
				const LabelID* __restrict labels = node.labels.data();
				const size_t L = node.labels.size();
				if (L == 0)
					continue; // inactive
				const size_t deg = inEdges.size();

				// Unrolling this like the one above is not currently effective.
				EnergyType bestE = std::numeric_limits<EnergyType>::max();
				size_t bestIdx = 0;
				LabelID bestL = labels[0];
				const EnergyType invQ = (EnergyType)(1.0f / 1024.0f);

				if (deg == 3) {
					const EnergyType* __restrict m0 = msgs + offs[inEdges[0]];
					const EnergyType* __restrict m1 = msgs + offs[inEdges[1]];
					const EnergyType* __restrict m2 = msgs + offs[inEdges[2]];

					for (size_t j = 0; j < L; ++j) {
						EnergyType e = data[j] + m0[j] + m1[j] + m2[j];
						if (e < bestE) {
							bestE = e;
							bestIdx = j;
							bestL = labels[j];
						}
					}
				}
				else {
					for (size_t j = 0; j < L; ++j) {
						EnergyType e = data[j];
						for (EdgeID eid : inEdges)
							e += msgs[offs[eid] + j];

						if (e < bestE) {
							bestE = e;
							bestIdx = j;
							bestL = labels[j];
						}
					}
				}

				node.label = bestL;
				node.dataCost = data[bestIdx];
				if (bestL != oldLabel)
					++changed;
			}
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

		// [LBP-DIAG] one-off perf diagnostic: how many iterations run, the
		// per-iteration "changed" trajectory, and the message-buffer size (the
		// bandwidth the message passing must stream each iteration). Cheap;
		// remove once the LBP cost model is understood.
		unsigned itersRun = 0;
		const unsigned kTraceMax = maxIters;
		std::vector<int> changedTrace(kTraceMax, 0);

		for (unsigned it = 0; it < maxIters; ++it) {
			int changed = Optimize(1);
			if (it < kTraceMax) changedTrace[it] = changed;
			itersRun = it + 1;

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

		#ifdef DEBUG_EXTRA
		{
			// total labels over all nodes + message-buffer element count (== sum of
			// per-edge label counts). This is what the memory-bound cost scales with.
			size_t totalLabels = 0;
			for (const Node& n : nodes) totalLabels += n.labels.size();
			const size_t msgElems = buffersInitialized ? msgBuf[0].size() : 0;
			DEBUG_EXTRA("[LBP-DIAG] nodes=%zu edges=%zu iters=%u avgLabels=%.2f msgBufMB=%.1f (x2)",
				nodes.size(), edges.size(), itersRun,
				nodes.empty() ? 0.0 : (double)totalLabels / (double)nodes.size(),
				(double)(msgElems * sizeof(EnergyType)) / (1024.0*1024.0));
			char buf[512]; int off = 0;
			off += snprintf(buf+off, sizeof(buf)-off, "[LBP-DIAG] changed/iter:");
			for (unsigned i = 0; i < itersRun && i < kTraceMax && off < (int)sizeof(buf)-16; ++i)
				off += snprintf(buf+off, sizeof(buf)-off, " %d", changedTrace[i]);
			DEBUG_EXTRA("%s", buf);
		}
		#endif

		// Build compact label array for fast external access
		finalLabels.resize(nodes.size());

		for (size_t i = 0, cnt = (size_t) nodes.size(); i < cnt; ++i)
			finalLabels[i] = nodes[i].label;

		// Optimization done: release the per-thread scratch so it does not
		// outlive the last Optimize call.
		ReleaseScratch();

		const EnergyType finalEnergy = ComputeEnergy();
		#ifdef DEBUG_EXTRA
		// Final MRF energy: the objective quality of the labeling. Compare across
		// LBP_MAX_ITERS values -- if energy@25 is within a fraction of a percent of
		// energy@50, the shorter budget costs essentially no view-selection quality.
		DEBUG_EXTRA("[LBP-DIAG] finalEnergy=%.6g", (double)finalEnergy);
		#endif
		return finalEnergy;
	}

	inline LabelID GetLabel(NodeID nodeID) const {
		return nodes[nodeID].label;
	}
};
/*----------------------------------------------------------------*/
} // namespace SEACAVE

#endif // __SEACAVE_LBP_H__
