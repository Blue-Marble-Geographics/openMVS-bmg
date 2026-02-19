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

	// Cached topology
	std::vector<boost::container::small_vector<EdgeID, 3>> outEdges;
	LabelID globalMaxLabel = 0;
	size_t maxLabelsPerNode = 0;
	bool topologyPrepared = false;

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

		// ------------------------------------------------------------
		// Thread-local scratch (persistent across calls)
		// ------------------------------------------------------------
		static thread_local std::vector<EnergyType> sumAll;
		static thread_local std::vector<EnergyType> energyBuf;
		static thread_local std::vector<EnergyType> perLabelMin;
		static thread_local std::vector<uint32_t> perLabelGen;
		static thread_local uint32_t curGen = 1;

		int changed = 0;

		// ------------------------------------------------------------
		// Message passing iterations
		// ------------------------------------------------------------
		EnergyType* __restrict readMsgs = msgBuf[msgParity].data();
		EnergyType* __restrict writeMsgs = msgBuf[msgParity ^ 1].data();

	#ifdef LBP_USE_OPENMP
	#pragma omp parallel
	#endif
		{
			const size_t need = (size_t)globalMaxLabel + 1;

			// Step 1: ensure capacity (no reallocation later)
			if (perLabelMin.capacity() < need) {
				perLabelMin.reserve(need);
				perLabelGen.reserve(need);
			}

			// Step 2: ensure size (no reallocation now)
			if (perLabelMin.size() < need) {
				const size_t oldSize = perLabelMin.size();
				perLabelMin.resize(need);
				perLabelGen.resize(need);

				// Zero only the newly-added range
				memset(perLabelGen.data() + oldSize, 0,
					(need - oldSize) * sizeof(uint32_t));
			}

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
					const auto& labels2 = n2.labels;
					const size_t L2 = edgeMsgLen[eid];

					// find message from v -> u (subtraction term)
					const EnergyType* __restrict sub = nullptr;
					for (EdgeID pe : inEdges1) {
						if (edges[pe].nodeID1 == edge.nodeID2) {
							sub = readMsgs + offs[pe];
							break;
						}
					}

					// energyBuf = sumAll - sub
					if (sub) {
						for (size_t k = 0; k < L1; ++k)
							energyBuf[k] = sumAll[k] - sub[k];
					}
					else {
						memcpy(energyBuf.data(), sumAll.data(), L1 * sizeof(EnergyType));
					}

					// generation bump
					if (++curGen == 0) {
						memset(perLabelGen.data(), 0, perLabelGen.size() * sizeof(uint32_t));
						curGen = 1;
					}

					EnergyType globalMin = std::numeric_limits<EnergyType>::max();

					// compute minima
					for (size_t k = 0; k < L1; ++k) {
						EnergyType e = energyBuf[k];
						if (e < globalMin)
							globalMin = e;

						LabelID l = labels1[k];
						if (l != 0) {
							if (perLabelGen[l] != curGen) {
								perLabelGen[l] = curGen;
								perLabelMin[l] = e;
							}
							else if (e < perLabelMin[l]) {
								perLabelMin[l] = e;
							}
						}
					}

					const EnergyType base = globalMin + maxE;
					EnergyType* __restrict msgOut = writeMsgs + offs[eid];
					const uint32_t* __restrict gen = perLabelGen.data();
					const EnergyType* __restrict minv = perLabelMin.data();
					for (size_t j = 0; j < L2; ++j) {
						LabelID l2 = labels2[j];
						EnergyType best = base;

						if (l2 != 0 && gen[l2] == curGen) {
							best = std::min(best, minv[l2]);
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

	#if 0 // Optional?
			// ------------------------------------------------------------
			// Normalize messages
			// ------------------------------------------------------------
			EnergyType* __restrict normMsgs = msgBuf[msgParity].data();

	#ifdef LBP_USE_OPENMP
	#pragma omp for schedule(static) nowait
	#endif
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				EnergyType* __restrict m = normMsgs + offs[edgeID];
				const size_t L = edgeMsgLen[edgeID];

				EnergyType minMsg = std::numeric_limits<EnergyType>::max();
				for (size_t k = 0; k < L; ++k)
					if (m[k] < minMsg)
						minMsg = m[k];

				const size_t L4 = L & ~size_t(3);
				__m128 vmin = _mm_set1_ps(minMsg);
				__m128 vzero = _mm_setzero_ps();

				size_t k = 0;
				for (; k < L4; k += 4) {
					__m128 v = _mm_loadu_ps(m + k);
					v = _mm_sub_ps(v, vmin);
					v = _mm_max_ps(v, vzero);
					_mm_storeu_ps(m + k, v);
				}
				for (; k < L; ++k) {
					EnergyType v = m[k] - minMsg;
					if (v < (EnergyType)0)
						v = (EnergyType)0;
					m[k] = v;
				}
			}
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

		unsigned maxIters = 50;
		int lastChanged = INT_MAX;
		unsigned stall = 0;

		for (unsigned it = 0; it < maxIters; ++it) {
			int changed = Optimize(1);

			if (changed < nodes.size() * 0.005f)
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

		// Build compact label array for fast external access
		finalLabels.resize(nodes.size());

		for (size_t i = 0, cnt = (size_t) nodes.size(); i < cnt; ++i)
			finalLabels[i] = nodes[i].label;

		return ComputeEnergy();
	}

	inline LabelID GetLabel(NodeID nodeID) const {
		return nodes[nodeID].label;
	}
};
/*----------------------------------------------------------------*/
} // namespace SEACAVE

#endif // __SEACAVE_LBP_H__
