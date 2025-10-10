////////////////////////////////////////////////////////////////////
// LBA.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef __SEACAVE_LBA_H__
#define __SEACAVE_LBA_H__


// I N C L U D E S /////////////////////////////////////////////////
#pragma optimize("", on) // JPB WIP BUG


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define LBP_USE_OPENMP
#endif

#include <emmintrin.h> // for alignment hints if compiler uses them
#include <malloc.h>   // for _aligned_malloc / _aligned_free
#include <new>        // for std::bad_alloc
#include <cstddef>    // for std::size_t

template <typename T, std::size_t Alignment>
struct AlignedAllocator
{
	using value_type = T;

	AlignedAllocator() noexcept {}
	template <class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

	T* allocate(std::size_t n)
	{
		void* p = _aligned_malloc(n * sizeof(T), Alignment);
		if (!p) throw std::bad_alloc();
		return static_cast<T*>(p);
	}

	void deallocate(T* p, std::size_t) noexcept
	{
		_aligned_free(p);
	}

	// required by MSVC allocator traits
	template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template <class T1, std::size_t A1, class T2, std::size_t A2>
inline bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept { return A1 == A2; }

template <class T1, std::size_t A1, class T2, std::size_t A2>
inline bool operator!=(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept { return A1 != A2; }


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

protected:
	struct DirectedEdge {
		NodeID nodeID1;
		NodeID nodeID2;
		std::vector<EnergyType> newMsgs;
		std::vector<EnergyType> oldMsgs;
		inline DirectedEdge(NodeID _nodeID1, NodeID _nodeID2) : nodeID1(_nodeID1), nodeID2(_nodeID2) {}
	};

	struct Node {
		LabelID label;
		EnergyType dataCost;
		std::vector<LabelID> labels;
		std::vector<EnergyType> dataCosts;
		std::vector<EdgeID> incomingEdges;
		inline Node() : label(0), dataCost(MaxEnergy) {}
	};

	std::vector<DirectedEdge> edges;
	std::vector<Node> nodes;
	FncSmoothCost fncSmoothCost;

public:
	LBPInference() {}
	LBPInference(NodeID nNodes) : nodes(nNodes) {}

	inline void SetNumNodes(NodeID nNodes) {
		nodes.resize(nNodes);
	}
	inline NodeID GetNumNodes() const {
		return (NodeID)nodes.size();
	}

	inline void SetNeighbors(NodeID nodeID1, NodeID nodeID2) {
		nodes[nodeID2].incomingEdges.push_back((EdgeID)edges.size());
		edges.push_back(DirectedEdge(nodeID1, nodeID2));
		nodes[nodeID1].incomingEdges.push_back((EdgeID)edges.size());
		edges.push_back(DirectedEdge(nodeID2, nodeID1));
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

#if 1

#if 0 //  claude
	void Optimize(unsigned num_iterations)
	{
		for (unsigned iter = 0; iter < num_iterations; ++iter) {
#ifdef LBP_USE_OPENMP
#pragma omp parallel
			{
				std::vector<EnergyType> energyBuf;
				energyBuf.reserve(128);
#pragma omp for schedule(static) nowait
#endif
				for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
					DirectedEdge& edge = edges[edgeID];
					const Node& n1 = nodes[edge.nodeID1];
					const Node& n2 = nodes[edge.nodeID2];
					const LabelID* __restrict labels1 = n1.labels.data();
					const LabelID* __restrict labels2 = n2.labels.data();
					const EnergyType* __restrict dataCosts1 = n1.dataCosts.data();
					const EdgeID* __restrict inEdges1 = n1.incomingEdges.data();
					const size_t numInEdges = n1.incomingEdges.size();
					const size_t L1 = n1.labels.size();
					const size_t L2 = n2.labels.size();
					const NodeID nodeID2 = edge.nodeID2;

					energyBuf.resize(L1);
					EnergyType* __restrict energyPtr = energyBuf.data();

					// Precompute unary + incoming message sum for node1
					// Initialize with data costs
					std::memcpy(energyPtr, dataCosts1, L1 * sizeof(EnergyType));

					// Add incoming messages
					for (size_t i = 0; i < numInEdges; ++i) {
						const DirectedEdge& pre = edges[inEdges1[i]];
						if (pre.nodeID1 == nodeID2) continue;

						const EnergyType* __restrict oldMsgs = pre.oldMsgs.data();
						const size_t L1_4 = L1 & ~size_t(3);
						size_t k = 0;

						// SIMD addition of messages
						for (; k < L1_4; k += 4) {
							__m128 ve = _mm_loadu_ps(&energyPtr[k]);
							__m128 vm = _mm_loadu_ps(&oldMsgs[k]);
							ve = _mm_add_ps(ve, vm);
							_mm_storeu_ps(&energyPtr[k], ve);
						}
						for (; k < L1; ++k) {
							energyPtr[k] += oldMsgs[k];
						}
					}

					// Compute new messages
					EnergyType* __restrict newMsgs = edge.newMsgs.data();
					const EnergyType maxEnergy = (EnergyType)LBPInference::MaxEnergy;

					// Precompute which labels match for vectorization
					for (size_t j = 0; j < L2; ++j) {
						const LabelID label2 = labels2[j];
						const bool label2_nonzero = (label2 != 0);
						EnergyType minEnergy = std::numeric_limits<EnergyType>::max();

						// Process 4 labels at a time
						const size_t L1_4 = L1 & ~size_t(3);
						__m128 vmin = _mm_set1_ps(std::numeric_limits<EnergyType>::max());
						__m128 vmaxE = _mm_set1_ps(maxEnergy);
						__m128 vzero = _mm_setzero_ps();

						size_t k = 0;
						for (; k < L1_4; k += 4) {
							// Load energies
							__m128 ve = _mm_loadu_ps(&energyPtr[k]);

							// Compute smoothness costs for 4 labels
							const LabelID l0 = labels1[k];
							const LabelID l1 = labels1[k + 1];
							const LabelID l2 = labels1[k + 2];
							const LabelID l3 = labels1[k + 3];

							const float s0 = (l0 == label2 && l0 != 0 && label2_nonzero) ? 0.0f : maxEnergy;
							const float s1 = (l1 == label2 && l1 != 0 && label2_nonzero) ? 0.0f : maxEnergy;
							const float s2 = (l2 == label2 && l2 != 0 && label2_nonzero) ? 0.0f : maxEnergy;
							const float s3 = (l3 == label2 && l3 != 0 && label2_nonzero) ? 0.0f : maxEnergy;

							__m128 vsmooth = _mm_set_ps(s3, s2, s1, s0);
							ve = _mm_add_ps(ve, vsmooth);
							vmin = _mm_min_ps(vmin, ve);
						}

						// Horizontal minimum of SIMD register
						vmin = _mm_min_ps(vmin, _mm_shuffle_ps(vmin, vmin, _MM_SHUFFLE(2, 3, 0, 1)));
						vmin = _mm_min_ps(vmin, _mm_shuffle_ps(vmin, vmin, _MM_SHUFFLE(1, 0, 3, 2)));
						_mm_store_ss(&minEnergy, vmin);

						// Handle remaining elements
						for (; k < L1; ++k) {
							const LabelID label1 = labels1[k];
							EnergyType smooth = (label1 == label2 && label1 != 0 && label2_nonzero) ? 0 : maxEnergy;
							EnergyType energy = energyPtr[k] + smooth;
							if (energy < minEnergy) minEnergy = energy;
						}

						newMsgs[j] = minEnergy;
					}
				}
#ifdef LBP_USE_OPENMP
			}
#endif

			// Message normalization pass
#ifdef LBP_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				DirectedEdge& edge = edges[edgeID];
				edge.newMsgs.swap(edge.oldMsgs);

				EnergyType* __restrict oldMsgs = edge.oldMsgs.data();
				const size_t L = edge.oldMsgs.size();

				// SIMD minimum finding (8-wide for better ILP)
				const size_t L8 = L & ~size_t(7);
				__m128 vmin1 = _mm_set1_ps(std::numeric_limits<EnergyType>::max());
				__m128 vmin2 = vmin1;

				size_t k = 0;
				for (; k < L8; k += 8) {
					__m128 v1 = _mm_loadu_ps(&oldMsgs[k]);
					__m128 v2 = _mm_loadu_ps(&oldMsgs[k + 4]);
					vmin1 = _mm_min_ps(vmin1, v1);
					vmin2 = _mm_min_ps(vmin2, v2);
				}

				vmin1 = _mm_min_ps(vmin1, vmin2);
				vmin1 = _mm_min_ps(vmin1, _mm_shuffle_ps(vmin1, vmin1, _MM_SHUFFLE(2, 3, 0, 1)));
				vmin1 = _mm_min_ps(vmin1, _mm_shuffle_ps(vmin1, vmin1, _MM_SHUFFLE(1, 0, 3, 2)));

				EnergyType minMsg;
				_mm_store_ss(&minMsg, vmin1);

				// Scalar minimum for remainder
				for (; k < L; ++k) {
					if (oldMsgs[k] < minMsg) minMsg = oldMsgs[k];
				}

				// SIMD normalization (8-wide)
				const size_t L8_norm = L & ~size_t(7);
				__m128 vmin_broadcast = _mm_set1_ps(minMsg);

				k = 0;
				for (; k < L8_norm; k += 8) {
					__m128 v1 = _mm_loadu_ps(&oldMsgs[k]);
					__m128 v2 = _mm_loadu_ps(&oldMsgs[k + 4]);
					v1 = _mm_sub_ps(v1, vmin_broadcast);
					v2 = _mm_sub_ps(v2, vmin_broadcast);
					_mm_storeu_ps(&oldMsgs[k], v1);
					_mm_storeu_ps(&oldMsgs[k + 4], v2);
				}

				for (; k < L; ++k) {
					oldMsgs[k] -= minMsg;
				}
			}
		}

		// Final labeling phase
#ifdef LBP_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID) {
			Node& node = nodes[nodeID];
			const EdgeID* __restrict inEdges = node.incomingEdges.data();
			const size_t numInEdges = node.incomingEdges.size();
			const EnergyType* __restrict dataCosts = node.dataCosts.data();
			const size_t L = node.labels.size();

			EnergyType bestE = std::numeric_limits<EnergyType>::max();
			size_t bestIdx = 0;

			for (size_t j = 0; j < L; ++j) {
				EnergyType e = dataCosts[j];
				for (size_t i = 0; i < numInEdges; ++i) {
					e += edges[inEdges[i]].oldMsgs[j];
				}
				if (e < bestE) {
					bestE = e;
					bestIdx = j;
				}
			}

			node.label = node.labels[bestIdx];
			node.dataCost = dataCosts[bestIdx];
		}
	}
#else
// Two aligned buffers for ping-pong messaging
std::vector<EnergyType, AlignedAllocator<EnergyType, 32>> msgBuf[2];
std::vector<size_t> msgOffset;
bool buffersInitialized = false;
size_t msgParity = 0; // 0 = read buf0, 1 = write buf1

inline size_t EdgeMsgLen(size_t e) noexcept {
	return nodes[edges[e].nodeID2].labels.size();
}

void PrepareMessageBuffers()
{
	msgOffset.resize(edges.size());
	size_t total = 0;
	for (size_t e = 0; e < edges.size(); ++e) {
		msgOffset[e] = total;
		total += EdgeMsgLen(e);
	}
	msgBuf[0].assign(total, EnergyType(0));
	msgBuf[1].assign(total, EnergyType(0));
	buffersInitialized = true;
}

// -----------------------------------------------------------------
void Optimize(unsigned num_iterations)
{
	if (!buffersInitialized)
		PrepareMessageBuffers();

	const size_t* __restrict offs = msgOffset.data();
	const size_t numEdges = edges.size();

	for (unsigned iter = 0; iter < num_iterations; ++iter)
	{
		EnergyType* __restrict readMsgs = msgBuf[msgParity].data();
		EnergyType* __restrict writeMsgs = msgBuf[msgParity ^ 1].data();

#ifdef LBP_USE_OPENMP
#pragma omp parallel
		{
			std::vector<EnergyType> energyBuf;
			energyBuf.reserve(128);

#pragma omp for schedule(static, 2048)
#endif
			for (int_t edgeID = 0; edgeID < (int_t)numEdges; ++edgeID)
			{
				const DirectedEdge& edge = edges[edgeID];
				const Node& n1 = nodes[edge.nodeID1];
				const Node& n2 = nodes[edge.nodeID2];

				const auto& labels1 = n1.labels;
				const auto& labels2 = n2.labels;
				const auto& inEdges1 = n1.incomingEdges;

				const size_t L1 = labels1.size();
				const size_t L2 = labels2.size();

				EnergyType* __restrict msgOut = writeMsgs + offs[edgeID];
				energyBuf.resize(L1);

				// unary + incoming (read from previous normalized buffer)
				const EdgeID* __restrict inE = inEdges1.data();
				const size_t nIn = inEdges1.size();
				const EnergyType* __restrict data1 = n1.dataCosts.data();
				const int32_t edgeNode2 = edge.nodeID2;

				for (size_t k = 0; k < L1; ++k)
				{
					// start with unary cost
					EnergyType e = data1[k];

					// manual unroll for small, typical degrees (2–6)
					size_t t = 0;

					// unrolled body
					for (; t + 3 < nIn; t += 4)
					{
						const DirectedEdge& pre0 = edges[inE[t + 0]];
						const DirectedEdge& pre1 = edges[inE[t + 1]];
						const DirectedEdge& pre2 = edges[inE[t + 2]];
						const DirectedEdge& pre3 = edges[inE[t + 3]];

						if (pre0.nodeID1 != edgeNode2) e += readMsgs[offs[inE[t + 0]] + k];
						if (pre1.nodeID1 != edgeNode2) e += readMsgs[offs[inE[t + 1]] + k];
						if (pre2.nodeID1 != edgeNode2) e += readMsgs[offs[inE[t + 2]] + k];
						if (pre3.nodeID1 != edgeNode2) e += readMsgs[offs[inE[t + 3]] + k];
					}

					// remainder
					for (; t < nIn; ++t)
					{
						const DirectedEdge& pre = edges[inE[t]];
						if (pre.nodeID1 != edgeNode2)
							e += readMsgs[offs[inE[t]] + k];
					}

					energyBuf[k] = e;
				}

				// Potts message computation
				const EnergyType maxE = (EnergyType)LBPInference::MaxEnergy;

				for (size_t j = 0; j < L2; ++j)
				{
					const LabelID l2 = labels2[j];
					const bool v2 = (l2 != 0);

					EnergyType minE = std::numeric_limits<EnergyType>::max();

					const EnergyType* __restrict Eb = energyBuf.data();   // E[k]
					const LabelID* __restrict L1p = labels1.data();      // labels1[k]

					size_t k = 0;
					const size_t L4 = L1 & ~size_t(3); // largest multiple of 4 <= L1

					// unrolled by 4
					for (; k < L4; k += 4)
					{
						const LabelID l10 = L1p[k + 0];
						const LabelID l11 = L1p[k + 1];
						const LabelID l12 = L1p[k + 2];
						const LabelID l13 = L1p[k + 3];

						const EnergyType s0 = (l10 == l2 && l10 != 0 && v2) ? (EnergyType)0 : maxE;
						const EnergyType s1 = (l11 == l2 && l11 != 0 && v2) ? (EnergyType)0 : maxE;
						const EnergyType s2 = (l12 == l2 && l12 != 0 && v2) ? (EnergyType)0 : maxE;
						const EnergyType s3 = (l13 == l2 && l13 != 0 && v2) ? (EnergyType)0 : maxE;

						const EnergyType e0 = Eb[k + 0] + s0;
						const EnergyType e1 = Eb[k + 1] + s1;
						const EnergyType e2 = Eb[k + 2] + s2;
						const EnergyType e3 = Eb[k + 3] + s3;

						if (e0 < minE) minE = e0;
						if (e1 < minE) minE = e1;
						if (e2 < minE) minE = e2;
						if (e3 < minE) minE = e3;
					}

					// tail
					for (; k < L1; ++k)
					{
						const LabelID l1 = L1p[k];
						const EnergyType smooth = (l1 == l2 && l1 != 0 && v2) ? (EnergyType)0 : maxE;
						const EnergyType e = Eb[k] + smooth;
						if (e < minE) minE = e;
					}

					msgOut[j] = minE;
				}
			}
#ifdef LBP_USE_OPENMP
		} // end parallel region
#endif

				// --- flip buffers: write -> new read
		msgParity ^= 1;

		// --- normalize the *new* read buffer (formerly write buffer)
		EnergyType* __restrict normMsgs = msgBuf[msgParity].data();
#ifdef LBP_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (int_t edgeID = 0; edgeID < (int_t)numEdges; ++edgeID)
		{
			EnergyType* __restrict m = normMsgs + offs[edgeID];
			const size_t L = EdgeMsgLen(edgeID);

			EnergyType minMsg = std::numeric_limits<EnergyType>::max();
			for (size_t k = 0; k < L; ++k)
				if (m[k] < minMsg) minMsg = m[k];

			const size_t L4 = L & ~size_t(3);
			__m128 vmin = _mm_set1_ps(minMsg);
			size_t k = 0;
			for (; k < L4; k += 4) {
				__m128 v = _mm_loadu_ps(m + k);
				v = _mm_sub_ps(v, vmin);
				_mm_storeu_ps(m + k, v);
			}
			for (; k < L; ++k)
				m[k] -= minMsg;
		}
	}

	// -----------------------------------------------------------------
	// Final labeling (reads from last normalized buffer)
	EnergyType* __restrict msgs = msgBuf[msgParity].data();
#ifdef LBP_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID)
	{
		Node& node = nodes[nodeID];
		const auto& inEdges = node.incomingEdges;
		const size_t L = node.labels.size();

		EnergyType bestE = std::numeric_limits<EnergyType>::max();
		LabelID bestL = node.labels.front();
		size_t bestIdx = 0;

		for (size_t j = 0; j < L; ++j)
		{
			EnergyType e = node.dataCosts[j];
			for (EdgeID eid : inEdges)
				e += msgs[offs[eid] + j];
			if (e < bestE) { bestE = e; bestIdx = j; bestL = node.labels[j]; }
		}

		node.label = bestL;
		node.dataCost = node.dataCosts[bestIdx];
	}
}

#endif




#else
	void Optimize(unsigned num_iterations) {
		for (unsigned i = 0; i < num_iterations; ++i) {
			#ifdef LBP_USE_OPENMP
			#pragma omp parallel for
			#endif
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				DirectedEdge& edge = edges[edgeID];
				const std::vector<LabelID>& labels1 = nodes[edge.nodeID1].labels;
				const std::vector<LabelID>& labels2 = nodes[edge.nodeID2].labels;
				for (size_t j = 0; j < labels2.size(); ++j) {
					const LabelID label2(labels2[j]);
					EnergyType minEnergy(std::numeric_limits<EnergyType>::max());
					for (size_t k = 0; k < labels1.size(); ++k) {
						const LabelID label1(labels1[k]);
						EnergyType energy(nodes[edge.nodeID1].dataCosts[k] + fncSmoothCost(edge.nodeID1, edge.nodeID2, label1, label2));
						const std::vector<EdgeID>& incoming_edges1 = nodes[edge.nodeID1].incomingEdges;
						for (size_t n = 0; n < incoming_edges1.size(); ++n) {
							const DirectedEdge& pre_edge = edges[incoming_edges1[n]];
							if (pre_edge.nodeID1 == edge.nodeID2) continue;
							energy += pre_edge.oldMsgs[k];
						}
						if (minEnergy > energy)
							minEnergy = energy;
					}
					edge.newMsgs[j] = minEnergy;
				}
			}
			#ifdef LBP_USE_OPENMP
			#pragma omp parallel for
			#endif
			for (int_t edgeID = 0; edgeID < (int_t)edges.size(); ++edgeID) {
				DirectedEdge& edge = edges[edgeID];
				edge.newMsgs.swap(edge.oldMsgs);
				EnergyType minMsg(std::numeric_limits<EnergyType>::max());
				for (EnergyType msg: edge.oldMsgs)
					if (minMsg > msg)
						minMsg = msg;
				for (EnergyType& msg: edge.oldMsgs)
					msg -= minMsg;
			}
		}
		#ifdef LBP_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int_t nodeID = 0; nodeID < (int_t)nodes.size(); ++nodeID) {
			Node& node = nodes[nodeID];
			EnergyType minEnergy(std::numeric_limits<EnergyType>::max());
			for (size_t j = 0; j < node.labels.size(); ++j) {
				EnergyType energy(node.dataCosts[j]);
				for (EdgeID incoming_edge_idx : node.incomingEdges)
					energy += edges[incoming_edge_idx].oldMsgs[j];
				if (energy < minEnergy) {
					minEnergy = energy;
					node.label = node.labels[j];
					node.dataCost = node.dataCosts[j];
				}
			}
		}
	}
#endif
	EnergyType Optimize() {
		TD_TIMER_STARTD();
		EnergyType energy(ComputeEnergy());
		EnergyType diff(energy);
		unsigned i(0);
		#if 1
		unsigned nIncreases(0), nTotalIncreases(0);
		#endif

		const EnergyType eps = (EnergyType)1e-6;

		while (true) {
			TD_TIMER_STARTD();
			const EnergyType last_energy(energy);
			Optimize(1);
			energy = ComputeEnergy();
			diff = last_energy - energy;
			DEBUG_ULTIMATE("\t%2u. e: %g\td: %g\tt: %s", i, last_energy, diff, TD_TIMER_GET_FMT().c_str());
			if (++i > 100 || fabs((double)diff) < eps)
				break;
			#if 1
			if (diff < EnergyType(0)) {
				++nTotalIncreases;
				if (++nIncreases > 3)
					break;
			} else {
				if (nTotalIncreases > 5)
					break;
				nIncreases = 0;
			}
			#else
			if (diff < EnergyType(0))
				break;
			#endif
		}
		if (diff == EnergyType(0)) {
			DEBUG_ULTIMATE("Inference converged in %u iterations: %g energy (%s)", i, energy, TD_TIMER_GET_FMT().c_str());
		} else {
			DEBUG_ULTIMATE("Inference aborted (energy increased): %u iterations, %g energy (%s)", i, energy, TD_TIMER_GET_FMT().c_str());
		}
		return energy;
	}

	inline LabelID GetLabel(NodeID nodeID) const {
		return nodes[nodeID].label;
	}
};
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // __SEACAVE_LBP_H__
