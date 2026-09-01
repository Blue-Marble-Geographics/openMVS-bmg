#include "SceneRefineAVX2.h"

#include <immintrin.h>
#include <cstring>

// FMA3 is available under /arch:AVX2 on this TU (see libs/MVS/CMakeLists.txt),
// but MSVC never auto-contracts separate mul+add intrinsics into fmadd
// (intrinsics are fixed ops, not reassociable expressions like source "a*b+c"
// under /fp:fast) -- so it has to be requested explicitly. Single-rounding FMA
// is NOT bit-identical to the double-rounded mul-then-add this file otherwise
// keeps exact parity with; this is an accepted, real departure for speed. Set
// to 0 to fall back to the original separate mul/add/sub sequence for an A/B.
#ifndef MESHOPT_AVX2_FMA
#define MESHOPT_AVX2_FMA 1
#endif
#if MESHOPT_AVX2_FMA
#define FMADD(a, b, c)  _mm256_fmadd_ps((a), (b), (c))   // a*b + c
#define FMSUB(a, b, c)  _mm256_fmsub_ps((a), (b), (c))   // a*b - c
#define FNMADD(a, b, c) _mm256_fnmadd_ps((a), (b), (c))  // c - a*b
#else
#define FMADD(a, b, c)  _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#define FMSUB(a, b, c)  _mm256_sub_ps(_mm256_mul_ps((a), (b)), (c))
#define FNMADD(a, b, c) _mm256_sub_ps((c), _mm256_mul_ps((a), (b)))
#endif

// approximate reciprocal (rcp + one Newton-Raphson step, ~23 bits of mantissa):
// same rcp/rsqrt+Newton pattern already used below for invS, and the scalar
// FastRecip precedent in SceneRefine.cpp -- these kernels already gave up exact
// IEEE division parity for FMA, so the 3 plain _mm256_div_ps sites below are
// the same category of missed win. Gated with MESHOPT_AVX2_FMA (same "give up
// bit-identical for speed" contract); set 0 for an exact-division A/B.
#if MESHOPT_AVX2_FMA
static inline __m256 FastRecipAVX2(__m256 a) {
	const __m256 twov = _mm256_set1_ps(2.f);
	__m256 y = _mm256_rcp_ps(a);
	y = _mm256_mul_ps(y, FNMADD(a, y, twov));
	return y;
}
#endif

namespace MVS {

int SceneRefineZNCCRowAVX2(
	const uint8_t* __restrict maskRow,
	const uint16_t* __restrict imageARow,
	const uint16_t* __restrict imageBRow,
	const uint16_t* __restrict meanARow,
	const float* __restrict varARow,
	const float* __restrict colB,
	const float* __restrict colB2,
	const float* __restrict colAB,
	float* __restrict gradRow,
	int colStart,
	int colEnd,
	float invWindowArea,
	float* score)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 invNv = _mm256_set1_ps(invWindowArea);
	const __m256 scale16v = _mm256_set1_ps(scale16);
	const __m256 scale65535v = _mm256_set1_ps(65535.f);
	const __m256 halfv = _mm256_set1_ps(0.5f);
	const __m256 minVarv = _mm256_set1_ps(0.0001f);
	const __m256 epsv = _mm256_set1_ps(1e-12f);
	const __m256 reliabilityFloorv = _mm256_set1_ps(0.0015f);
	const __m256 oneAndHalfv = _mm256_set1_ps(1.5f);
#ifdef _USE_CERES
	// accumulated in-register across the whole row and reduced once at the end,
	// instead of a per-iteration store-to-stack + scalar reload + memory
	// read-modify-write into *score (a serializing dependency every iteration).
	// Reassociates the summation order vs. the old immediate-per-group add into
	// *score -- not bit-identical, same accepted tradeoff as the FMA/rcp changes
	// above, and a far smaller effect than either.
	// Gated at COMPILE time (not just the runtime `if (score)` below): score is
	// only ever non-null in a CERES build, but without this #ifdef the compiler
	// still has to keep this __m256 alive across the whole loop "just in case",
	// adding to this kernel's already-tight register pressure for nothing.
	__m256 scoreAcc = _mm256_setzero_ps();
#endif

	int i = colStart;
	const int vecEnd = colStart + ((colEnd - colStart) & ~7);
	for (; i < vecEnd; i += 8) {
		uint64_t mask8;
		std::memcpy(&mask8, maskRow + i, sizeof(mask8));
		if (mask8 == 0)
			continue;
		const __m128i maskBytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(maskRow + i));
		const __m256i laneMask = _mm256_cmpgt_epi32(
			_mm256_cvtepu8_epi32(maskBytes), _mm256_setzero_si256());

		// raw loads hoisted well ahead of their first use: MSVC's intrinsic
		// scheduling stays close to source order (unlike GCC/Clang, which would
		// freely interleave these), so issuing them before the colB/colB2/colAB
		// sum trees below (independent of these four) gives their L1 load
		// latency room to hide behind that unrelated work instead of sitting
		// immediately in front of the widen/convert chain that consumes each.
		const __m128i meanARaw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(meanARow + i));
		const __m256 varA = _mm256_loadu_ps(varARow + i);
		const __m128i aRaw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(imageARow + i));
		const __m128i bRaw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(imageBRow + i));

		const __m256 sumB = _mm256_add_ps(
			_mm256_add_ps(_mm256_loadu_ps(colB + i - 2), _mm256_loadu_ps(colB + i - 1)),
			_mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(colB + i), _mm256_loadu_ps(colB + i + 1)), _mm256_loadu_ps(colB + i + 2)));
		const __m256 sumB2 = _mm256_add_ps(
			_mm256_add_ps(_mm256_loadu_ps(colB2 + i - 2), _mm256_loadu_ps(colB2 + i - 1)),
			_mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(colB2 + i), _mm256_loadu_ps(colB2 + i + 1)), _mm256_loadu_ps(colB2 + i + 2)));
		const __m256 sumAB = _mm256_add_ps(
			_mm256_add_ps(_mm256_loadu_ps(colAB + i - 2), _mm256_loadu_ps(colAB + i - 1)),
			_mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(colAB + i), _mm256_loadu_ps(colAB + i + 1)), _mm256_loadu_ps(colAB + i + 2)));

		const __m256 meanBRaw = _mm256_mul_ps(sumB, invNv);
		const __m256 meanBScaled = FMADD(meanBRaw, scale65535v, halfv);
		const __m256i meanBInt = _mm256_cvttps_epi32(meanBScaled);
		const __m256 meanB = _mm256_mul_ps(_mm256_cvtepi32_ps(meanBInt), scale16v);
		const __m256 varB = _mm256_max_ps(
			FMSUB(sumB2, invNv, _mm256_mul_ps(meanBRaw, meanBRaw)),
			minVarv);
		const __m256 meanA = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(meanARaw)), scale16v);
		const __m256 cov = _mm256_mul_ps(sumAB, invNv);

		const __m256 prod = _mm256_max_ps(_mm256_mul_ps(varA, varB), epsv);
		__m256 invS = _mm256_rsqrt_ps(prod);
		const __m256 invSsq = _mm256_mul_ps(invS, invS);
		invS = _mm256_mul_ps(invS, FNMADD(halfv, _mm256_mul_ps(prod, invSsq), oneAndHalfv));
		const __m256 zncc = _mm256_mul_ps(FNMADD(meanA, meanB, cov), invS);
		// invVB = 1/varB = invS^2 * varA, since invS = 1/sqrt(varA*varB); reuses
		// the already-refined invS instead of a second rcp/div chain. Not exact
		// where the prod-clamp and a varB-only clamp would diverge (only one of
		// varA/varB near-zero), but that's exactly where reliability (below,
		// via minV=min(varA,varB)) already pushes grad toward zero regardless.
		const __m256 invVB = _mm256_mul_ps(_mm256_mul_ps(invS, invS), varA);
		const __m256 znccInvVB = _mm256_mul_ps(zncc, invVB);
		const __m256 aVal = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(aRaw)), scale16v);
		const __m256 bVal = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(bRaw)), scale16v);
		// grad needs -dZNCC (= -(aVal*invS + meanB*znccInvVB - bVal*znccInvVB - meanA*invS));
		// chain is built sign-flipped from the start so no separate negation/negOnev is needed
		__m256 negDZNCC = FMADD(meanA, invS, _mm256_mul_ps(bVal, znccInvVB));
		negDZNCC = FNMADD(meanB, znccInvVB, negDZNCC);
		negDZNCC = FNMADD(aVal, invS, negDZNCC);
		const __m256 minV = _mm256_min_ps(varA, varB);
#if MESHOPT_AVX2_FMA
		const __m256 reliability = _mm256_mul_ps(minV, FastRecipAVX2(_mm256_add_ps(minV, reliabilityFloorv)));
#else
		const __m256 reliability = _mm256_div_ps(minV, _mm256_add_ps(minV, reliabilityFloorv));
#endif
		const __m256 grad = _mm256_mul_ps(reliability, negDZNCC);
		if (mask8 == UINT64_C(0x0101010101010101))
			_mm256_storeu_ps(gradRow + i, grad);
		else
			_mm256_maskstore_ps(gradRow + i, laneMask, grad);

#ifdef _USE_CERES
		if (score) {
			scoreAcc = _mm256_add_ps(scoreAcc, _mm256_and_ps(
				_mm256_castsi256_ps(laneMask), FNMADD(reliability, zncc, reliability)));
		}
#endif
	}
#ifdef _USE_CERES
	if (score) {
		alignas(32) float terms[8];
		_mm256_store_ps(terms, scoreAcc);
		*score += terms[0] + terms[1] + terms[2] + terms[3];
		*score += terms[4] + terms[5] + terms[6] + terms[7];
	}
#endif
	_mm256_zeroupper();
	return i;
}

int SceneRefineWarpRowAVX2(
	const float* __restrict xnRow,
	const float* __restrict depthRowA,
	const uint32_t* __restrict faceRowA,
	const uint16_t* __restrict imageRowA,
	const float* __restrict depthMapB,
	const uint16_t* __restrict imageB,
	int cols,
	int colsB,
	int rowsB,
	float biasX, float biasY, float biasZ,
	float m00, float m10, float m20,
	float t0, float t1, float t2,
	float fxB, float fyB, float cxB, float cyB,
	uint32_t invalidId,
	uint16_t* __restrict outRow,
	uint8_t* __restrict maskRow)
{
	const __m256 zerops = _mm256_setzero_ps();
	const __m256 onev = _mm256_set1_ps(1.f);
	const __m256 tenv = _mm256_set1_ps(10.f);
	const __m256 uMaxv = _mm256_set1_ps((float)(colsB - 10));
	const __m256 vMaxv = _mm256_set1_ps((float)(rowsB - 10));
	const __m256i xLimit = _mm256_set1_epi32(colsB - 1);
	const __m256i yLimit = _mm256_set1_epi32(rowsB - 1);
	const __m256i colsBv = _mm256_set1_epi32(colsB);
	const __m256i invalidv = _mm256_set1_epi32((int)invalidId);
	const __m256 biasXv = _mm256_set1_ps(biasX);
	const __m256 biasYv = _mm256_set1_ps(biasY);
	const __m256 biasZv = _mm256_set1_ps(biasZ);
	const __m256 m00v = _mm256_set1_ps(m00);
	const __m256 m10v = _mm256_set1_ps(m10);
	const __m256 m20v = _mm256_set1_ps(m20);
	const __m256 t0v = _mm256_set1_ps(t0);
	const __m256 t1v = _mm256_set1_ps(t1);
	const __m256 t2v = _mm256_set1_ps(t2);
	const __m256 fxBv = _mm256_set1_ps(fxB);
	const __m256 fyBv = _mm256_set1_ps(fyB);
	const __m256 cxBv = _mm256_set1_ps(cxB);
	const __m256 cyBv = _mm256_set1_ps(cyB);
	const __m256 scale16v = _mm256_set1_ps(1.f / 65535.f);
	const __m256 scale65535v = _mm256_set1_ps(65535.f);
	const __m256 halfv = _mm256_set1_ps(0.5f);
	const __m256 hundredthv = _mm256_set1_ps(0.01f);
	const __m256i low16 = _mm256_set1_epi32(0xFFFF);
	const __m256i allOnes = _mm256_set1_epi32(-1);
	const __m128i one8 = _mm_set1_epi8(1);

	int i = 0;
	const int vecEnd = cols & ~7;
	for (; i < vecEnd; i += 8) {
		const __m128i fallback16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(imageRowA + i));
		const __m256i face = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(faceRowA + i));
		__m256i aliveI = _mm256_xor_si256(_mm256_cmpeq_epi32(face, invalidv), allOnes);
		if (_mm256_testz_si256(aliveI, aliveI)) {
			// whole group uncovered: fallback copy, mask stays clear
			_mm_storeu_si128(reinterpret_cast<__m128i*>(outRow + i), fallback16);
			std::memset(maskRow + i, 0, 8);
			continue;
		}
		const __m256 xn = _mm256_loadu_ps(xnRow + i);
		const __m256 z = _mm256_loadu_ps(depthRowA + i);
		// scalar gates are "continue on z<=0", so alive = !(z<=0) (NaN stays alive
		// here and dies at the border check, exactly like the scalar path)
		__m256 alive = _mm256_and_ps(_mm256_castsi256_ps(aliveI), _mm256_cmp_ps(z, zerops, _CMP_NLE_UQ));
		const __m256 termZ = FMADD(m20v, xn, biasZv);
		const __m256 zcB = FMADD(z, termZ, t2v);
		alive = _mm256_and_ps(alive, _mm256_cmp_ps(zcB, zerops, _CMP_NLE_UQ));
#if MESHOPT_AVX2_FMA
		const __m256 invZ = FastRecipAVX2(zcB);
#else
		const __m256 invZ = _mm256_div_ps(onev, zcB);
#endif
		const __m256 termX = FMADD(m00v, xn, biasXv);
		const __m256 termY = FMADD(m10v, xn, biasYv);
		const __m256 xcB = FMADD(z, termX, t0v);
		const __m256 ycB = FMADD(z, termY, t1v);
		const __m256 fxInv = _mm256_mul_ps(fxBv, invZ);
		const __m256 fyInv = _mm256_mul_ps(fyBv, invZ);
		const __m256 u = FMADD(fxInv, xcB, cxBv);
		const __m256 v = FMADD(fyInv, ycB, cyBv);
		__m256 border = _mm256_and_ps(
			_mm256_cmp_ps(u, tenv, _CMP_GT_OQ), _mm256_cmp_ps(u, uMaxv, _CMP_LT_OQ));
		border = _mm256_and_ps(border, _mm256_and_ps(
			_mm256_cmp_ps(v, tenv, _CMP_GT_OQ), _mm256_cmp_ps(v, vMaxv, _CMP_LT_OQ)));
		alive = _mm256_and_ps(alive, border);
		const __m256i x0 = _mm256_cvttps_epi32(u);
		const __m256i y0 = _mm256_cvttps_epi32(v);
		// unsigned x0 >= colsB-1 (resp. y0/rowsB-1) rejects, matching the scalar
		// (unsigned) casts; a >= b  <=>  max_epu32(a,b) == a
		const __m256i geX = _mm256_cmpeq_epi32(_mm256_max_epu32(x0, xLimit), x0);
		const __m256i geY = _mm256_cmpeq_epi32(_mm256_max_epu32(y0, yLimit), y0);
		alive = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_or_si256(geX, geY)), alive);
		aliveI = _mm256_castps_si256(alive);
		if (_mm256_testz_si256(aliveI, aliveI)) {
			_mm_storeu_si128(reinterpret_cast<__m128i*>(outRow + i), fallback16);
			std::memset(maskRow + i, 0, 8);
			continue;
		}
		// zero dead-lane indices so plain (unmasked) gathers stay in-bounds;
		// vgather rewrites its mask operand, so masked gathers would cost MSVC a
		// register copy per gather AND chain every gather on the alive mask
		const __m256i idx = _mm256_and_si256(
			_mm256_add_epi32(_mm256_mullo_epi32(y0, colsBv), x0), aliveI);
		const __m256i idxRow1 = _mm256_add_epi32(idx, colsBv);
		const __m256 thr = _mm256_mul_ps(zcB, hundredthv);
		const __m256 zMin = _mm256_sub_ps(zcB, thr);
		const __m256 zMax = _mm256_add_ps(zcB, thr);
		// fetch the adjacent (x0, x0+1) float pair per row with 4-lane 64-bit
		// gathers (scale 4 over the float grid): same bytes as two 8-lane ps
		// gathers at roughly half the gather micro-ops
		const __m128i idxLo = _mm256_castsi256_si128(idx);
		const __m128i idxHi = _mm256_extracti128_si256(idx, 1);
		const __m128i idxR1Lo = _mm256_castsi256_si128(idxRow1);
		const __m128i idxR1Hi = _mm256_extracti128_si256(idxRow1, 1);
		const __m256 row0Lo = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(depthMapB), idxLo, 4));
		const __m256 row0Hi = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(depthMapB), idxHi, 4));
		const __m256 row1Lo = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(depthMapB), idxR1Lo, 4));
		const __m256 row1Hi = _mm256_castpd_ps(_mm256_i32gather_pd(reinterpret_cast<const double*>(depthMapB), idxR1Hi, 4));
		// imageB gather issued here (right after the depth gathers, BEFORE the
		// deinterleave/similarity chain that depends on them) so MSVC's
		// close-to-source-order scheduling puts these two independent gather
		// streams back-to-back instead of serializing behind the depth chain --
		// only depends on idx/idxRow1, computed above, so hoisting is safe.
		const __m256i top2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(imageB), idx, 2);
		const __m256i bot2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(imageB), idxRow1, 2);
		// deinterleave the (even,odd) pairs back into lane order 0..7
		const auto deinterleave = [](const __m256 lo, const __m256 hi, __m256& even, __m256& odd) {
			const __m256 e = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
			const __m256 o = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
			even = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), _MM_SHUFFLE(3, 1, 2, 0)));
			odd = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(o), _MM_SHUFFLE(3, 1, 2, 0)));
		};
		__m256 d00, d10, d01, d11;
		deinterleave(row0Lo, row0Hi, d00, d10);
		deinterleave(row1Lo, row1Hi, d01, d11);
		const auto similar = [&](const __m256 d) {
			return _mm256_and_ps(
				_mm256_and_ps(_mm256_cmp_ps(d, zerops, _CMP_GT_OQ), _mm256_cmp_ps(d, zMin, _CMP_GE_OQ)),
				_mm256_cmp_ps(d, zMax, _CMP_LE_OQ));
		};
		alive = _mm256_and_ps(alive, _mm256_or_ps(
			_mm256_or_ps(similar(d00), similar(d10)),
			_mm256_or_ps(similar(d01), similar(d11))));
		aliveI = _mm256_castps_si256(alive);
		const __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(top2, low16)), scale16v);
		const __m256 b = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(top2, 16)), scale16v);
		const __m256 c = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(bot2, low16)), scale16v);
		const __m256 d = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(bot2, 16)), scale16v);
		const __m256 fx = _mm256_sub_ps(u, _mm256_cvtepi32_ps(x0));
		const __m256 fy = _mm256_sub_ps(v, _mm256_cvtepi32_ps(y0));
		const __m256 fx1 = _mm256_sub_ps(onev, fx);
		const __m256 fy1 = _mm256_sub_ps(onev, fy);
		const __m256 top = FMADD(b, fx, _mm256_mul_ps(a, fx1));
		const __m256 bot = FMADD(d, fx, _mm256_mul_ps(c, fx1));
		__m256 val = FMADD(bot, fy, _mm256_mul_ps(top, fy1));
		val = _mm256_min_ps(_mm256_max_ps(val, zerops), onev);
		const __m256i vi = _mm256_cvttps_epi32(FMADD(val, scale65535v, halfv));
		const __m256i fb32 = _mm256_cvtepu16_epi32(fallback16);
		const __m256i res = _mm256_blendv_epi8(fb32, vi, aliveI);
		const __m128i packed = _mm_packus_epi32(
			_mm256_castsi256_si128(res), _mm256_extracti128_si256(res, 1));
		_mm_storeu_si128(reinterpret_cast<__m128i*>(outRow + i), packed);
		const __m128i m16 = _mm_packs_epi32(
			_mm256_castsi256_si128(aliveI), _mm256_extracti128_si256(aliveI, 1));
		const __m128i m8 = _mm_packs_epi16(m16, _mm_setzero_si128());
		_mm_storel_epi64(reinterpret_cast<__m128i*>(maskRow + i), _mm_and_si128(m8, one8));
	}
	_mm256_zeroupper();
	return i;
}

// load 8 consecutive u16 and widen to 8 floats (exact: every u16 is representable)
static inline __m256 LoadU16x8AsF32(const uint16_t* __restrict p) {
	return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}

// Eight-wide priming pass for the fused-ZNCC rolling column sums:
//   colB += b, colB2 += b*b, colAB += a*b   over columns [x0,x1).
// Deliberately NO FMA here, unlike the kernels above: these are running sums
// accumulated across thousands of row slides, so this mirrors the scalar loop
// op-for-op (u16 widen, one mul by 1/65535, separate mul/add) and the result
// is BIT-IDENTICAL to the scalar path -- the memory traffic dominates anyway.
void SceneRefineZNCCPrimeRowAVX2(
	const uint16_t* __restrict aRow,
	const uint16_t* __restrict bRow,
	float* __restrict colB,
	float* __restrict colB2,
	float* __restrict colAB,
	int x0,
	int x1)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 scale16v = _mm256_set1_ps(scale16);
	int x = x0;
	for (; x + 8 <= x1; x += 8) {
		const __m256 b = _mm256_mul_ps(LoadU16x8AsF32(bRow + x), scale16v);
		const __m256 a = _mm256_mul_ps(LoadU16x8AsF32(aRow + x), scale16v);
		_mm256_storeu_ps(colB + x, _mm256_add_ps(_mm256_loadu_ps(colB + x), b));
		_mm256_storeu_ps(colB2 + x, _mm256_add_ps(_mm256_loadu_ps(colB2 + x), _mm256_mul_ps(b, b)));
		_mm256_storeu_ps(colAB + x, _mm256_add_ps(_mm256_loadu_ps(colAB + x), _mm256_mul_ps(a, b)));
	}
	for (; x < x1; ++x) {
		const float b = float(bRow[x]) * scale16;
		const float a = float(aRow[x]) * scale16;
		colB[x] += b;
		colB2[x] += b * b;
		colAB[x] += a * b;
	}
	_mm256_zeroupper();
}

// Eight-wide rolling update for the fused-ZNCC column sums (add row addR,
// remove row remR) over columns [x0,x1):
//   colB  += bAdd - bRem
//   colB2 += bAdd*bAdd - bRem*bRem
//   colAB += aAdd*bAdd - aRem*bRem
// Same contract as the priming kernel above: no FMA, exact scalar mirror,
// BIT-IDENTICAL results.
void SceneRefineZNCCRollRowAVX2(
	const uint16_t* __restrict addA,
	const uint16_t* __restrict remA,
	const uint16_t* __restrict addB,
	const uint16_t* __restrict remB,
	float* __restrict colB,
	float* __restrict colB2,
	float* __restrict colAB,
	int x0,
	int x1)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 scale16v = _mm256_set1_ps(scale16);
	int x = x0;
	for (; x + 8 <= x1; x += 8) {
		const __m256 bAdd = _mm256_mul_ps(LoadU16x8AsF32(addB + x), scale16v);
		const __m256 bRem = _mm256_mul_ps(LoadU16x8AsF32(remB + x), scale16v);
		const __m256 aAdd = _mm256_mul_ps(LoadU16x8AsF32(addA + x), scale16v);
		const __m256 aRem = _mm256_mul_ps(LoadU16x8AsF32(remA + x), scale16v);
		_mm256_storeu_ps(colB + x, _mm256_add_ps(_mm256_loadu_ps(colB + x),
			_mm256_sub_ps(bAdd, bRem)));
		_mm256_storeu_ps(colB2 + x, _mm256_add_ps(_mm256_loadu_ps(colB2 + x),
			_mm256_sub_ps(_mm256_mul_ps(bAdd, bAdd), _mm256_mul_ps(bRem, bRem))));
		_mm256_storeu_ps(colAB + x, _mm256_add_ps(_mm256_loadu_ps(colAB + x),
			_mm256_sub_ps(_mm256_mul_ps(aAdd, bAdd), _mm256_mul_ps(aRem, bRem))));
	}
	for (; x < x1; ++x) {
		const float bAdd = float(addB[x]) * scale16;
		const float bRem = float(remB[x]) * scale16;
		const float aAdd = float(addA[x]) * scale16;
		const float aRem = float(remA[x]) * scale16;
		colB[x] += bAdd - bRem;
		colB2[x] += bAdd * bAdd - bRem * bRem;
		colAB[x] += aAdd * bAdd - aRem * bRem;
	}
	_mm256_zeroupper();
}

// ===========================================================================
// Windowed-derivative ZNCC kernels (MESHOPT_DZNCC_WINDOWED=1, the LIVE config)
// ===========================================================================
// Same contract as the prime/roll kernels above: NO FMA, every operation
// mirrors the scalar loop op-for-op in the same order, all stores are masked
// blends where the scalar path left values unwritten -- byte-identical output.
// The rolling column sums stay in DOUBLE (drift protection across thousands of
// row slides, matching the scalar kernel); rsqrt parity holds because VRSQRTPS
// and RSQRTSS use the same hardware approximation per element.

// widen the two 4-lane halves of an 8-float vector to doubles
static inline void WidenF32x8(__m256 v, __m256d& lo, __m256d& hi) {
	lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
	hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
}

void SceneRefineWZNCCPrimeRowAVX2(
	const uint16_t* __restrict aRow,
	const uint16_t* __restrict bRow,
	double* __restrict colB,
	double* __restrict colB2,
	double* __restrict colAB,
	int cols)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 scale16v = _mm256_set1_ps(scale16);
	int x = 0;
	for (; x + 8 <= cols; x += 8) {
		const __m256 b = _mm256_mul_ps(LoadU16x8AsF32(bRow + x), scale16v);
		const __m256 a = _mm256_mul_ps(LoadU16x8AsF32(aRow + x), scale16v);
		__m256d bLo, bHi, aLo, aHi;
		WidenF32x8(b, bLo, bHi);
		WidenF32x8(a, aLo, aHi);
		// scalar: colB[c] += b; colB2[c] += (double)b*b; colAB[c] += (double)a*b
		_mm256_storeu_pd(colB + x, _mm256_add_pd(_mm256_loadu_pd(colB + x), bLo));
		_mm256_storeu_pd(colB + x + 4, _mm256_add_pd(_mm256_loadu_pd(colB + x + 4), bHi));
		_mm256_storeu_pd(colB2 + x, _mm256_add_pd(_mm256_loadu_pd(colB2 + x), _mm256_mul_pd(bLo, bLo)));
		_mm256_storeu_pd(colB2 + x + 4, _mm256_add_pd(_mm256_loadu_pd(colB2 + x + 4), _mm256_mul_pd(bHi, bHi)));
		_mm256_storeu_pd(colAB + x, _mm256_add_pd(_mm256_loadu_pd(colAB + x), _mm256_mul_pd(aLo, bLo)));
		_mm256_storeu_pd(colAB + x + 4, _mm256_add_pd(_mm256_loadu_pd(colAB + x + 4), _mm256_mul_pd(aHi, bHi)));
	}
	for (; x < cols; ++x) {
		const float b = float(bRow[x]) * scale16;
		const float a = float(aRow[x]) * scale16;
		colB[x] += b;
		colB2[x] += (double)b * b;
		colAB[x] += (double)a * b;
	}
	_mm256_zeroupper();
}

void SceneRefineWZNCCRollRowAVX2(
	const uint16_t* __restrict addA,
	const uint16_t* __restrict remA,
	const uint16_t* __restrict addB,
	const uint16_t* __restrict remB,
	double* __restrict colB,
	double* __restrict colB2,
	double* __restrict colAB,
	int cols)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 scale16v = _mm256_set1_ps(scale16);
	int x = 0;
	for (; x + 8 <= cols; x += 8) {
		const __m256 bAdd = _mm256_mul_ps(LoadU16x8AsF32(addB + x), scale16v);
		const __m256 bRem = _mm256_mul_ps(LoadU16x8AsF32(remB + x), scale16v);
		const __m256 aAdd = _mm256_mul_ps(LoadU16x8AsF32(addA + x), scale16v);
		const __m256 aRem = _mm256_mul_ps(LoadU16x8AsF32(remA + x), scale16v);
		__m256d bAddLo, bAddHi, bRemLo, bRemHi, aAddLo, aAddHi, aRemLo, aRemHi;
		WidenF32x8(bAdd, bAddLo, bAddHi);
		WidenF32x8(bRem, bRemLo, bRemHi);
		WidenF32x8(aAdd, aAddLo, aAddHi);
		WidenF32x8(aRem, aRemLo, aRemHi);
		// scalar: colB[x] += (double)bAdd - bRem;
		//         colB2[x] += (double)bAdd*bAdd - (double)bRem*bRem;
		//         colAB[x] += (double)aAdd*bAdd - (double)aRem*bRem;
		_mm256_storeu_pd(colB + x, _mm256_add_pd(_mm256_loadu_pd(colB + x), _mm256_sub_pd(bAddLo, bRemLo)));
		_mm256_storeu_pd(colB + x + 4, _mm256_add_pd(_mm256_loadu_pd(colB + x + 4), _mm256_sub_pd(bAddHi, bRemHi)));
		_mm256_storeu_pd(colB2 + x, _mm256_add_pd(_mm256_loadu_pd(colB2 + x),
			_mm256_sub_pd(_mm256_mul_pd(bAddLo, bAddLo), _mm256_mul_pd(bRemLo, bRemLo))));
		_mm256_storeu_pd(colB2 + x + 4, _mm256_add_pd(_mm256_loadu_pd(colB2 + x + 4),
			_mm256_sub_pd(_mm256_mul_pd(bAddHi, bAddHi), _mm256_mul_pd(bRemHi, bRemHi))));
		_mm256_storeu_pd(colAB + x, _mm256_add_pd(_mm256_loadu_pd(colAB + x),
			_mm256_sub_pd(_mm256_mul_pd(aAddLo, bAddLo), _mm256_mul_pd(aRemLo, bRemLo))));
		_mm256_storeu_pd(colAB + x + 4, _mm256_add_pd(_mm256_loadu_pd(colAB + x + 4),
			_mm256_sub_pd(_mm256_mul_pd(aAddHi, bAddHi), _mm256_mul_pd(aRemHi, bRemHi))));
	}
	for (; x < cols; ++x) {
		const float bAdd = float(addB[x]) * scale16;
		const float bRem = float(remB[x]) * scale16;
		const float aAdd = float(addA[x]) * scale16;
		const float aRem = float(remA[x]) * scale16;
		colB[x] += (double)bAdd - bRem;
		colB2[x] += (double)bAdd * bAdd - (double)bRem * bRem;
		colAB[x] += (double)aAdd * bAdd - (double)aRem * bRem;
	}
	_mm256_zeroupper();
}

// per-pixel derivative-term pass for one row: 5-tap double window sums of the
// rolling columns, mean/var/invS/zncc, masked stores into the ring rows.
// Skipped (unmasked) lanes store the memset zero for InvS/Zovb/Mean/Valid and
// the prior contents for VarB -- exactly what the scalar loop leaves behind.
void SceneRefineWZNCCTermsRowAVX2(
	const uint8_t* __restrict maskRow,
	const uint16_t* __restrict meanARow,
	const float* __restrict varARow,
	const double* __restrict colB,
	const double* __restrict colB2,
	const double* __restrict colAB,
	float* __restrict rowInvS,
	float* __restrict rowZovb,
	float* __restrict rowMean,
	float* __restrict rowVarB,
	float* __restrict rowValid,
	int colStart,
	int colEnd,
	float invN,
	float* score)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 scale16v = _mm256_set1_ps(scale16);
	const __m256d invNd = _mm256_set1_pd((double)invN);
	const __m256 minVarv = _mm256_set1_ps(0.0001f);
	const __m256 epsv = _mm256_set1_ps(1e-12f);
	const __m256 onev = _mm256_set1_ps(1.f);
	const __m256i zero256 = _mm256_setzero_si256();

	// sequential 5-tap double sum (o = -2..+2), one 4-lane half at a time --
	// per lane this is the same add order as the scalar loop
	const auto sum5 = [](const double* __restrict base, int i) -> __m256d {
		__m256d s = _mm256_loadu_pd(base + i - 2);
		s = _mm256_add_pd(s, _mm256_loadu_pd(base + i - 1));
		s = _mm256_add_pd(s, _mm256_loadu_pd(base + i));
		s = _mm256_add_pd(s, _mm256_loadu_pd(base + i + 1));
		s = _mm256_add_pd(s, _mm256_loadu_pd(base + i + 2));
		return s;
	};
	const auto narrow = [](__m256d lo, __m256d hi) -> __m256 {
		return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(lo)), _mm256_cvtpd_ps(hi), 1);
	};

	int i = colStart;
	const int vecEnd = colStart + ((colEnd - colStart) & ~7);
	for (; i < vecEnd; i += 8) {
		uint64_t mask8;
		std::memcpy(&mask8, maskRow + i, sizeof(mask8));
		if (mask8 == 0)
			continue;

		const __m256d sumBLo = sum5(colB, i), sumBHi = sum5(colB, i + 4);
		const __m256d sumB2Lo = sum5(colB2, i), sumB2Hi = sum5(colB2, i + 4);
		const __m256d sumABLo = sum5(colAB, i), sumABHi = sum5(colAB, i + 4);

		// meanBRawD = sumB * invN (double); meanBRaw = (float)meanBRawD
		const __m256d meanBRawDLo = _mm256_mul_pd(sumBLo, invNd);
		const __m256d meanBRawDHi = _mm256_mul_pd(sumBHi, invNd);
		const __m256 meanBRaw = narrow(meanBRawDLo, meanBRawDHi);
		// meanB = float((uint16_t)(meanBRaw * 65535.f + 0.5f)) * scale16  (truncating cast)
		const __m256i meanBInt = _mm256_cvttps_epi32(
			_mm256_add_ps(_mm256_mul_ps(meanBRaw, _mm256_set1_ps(65535.f)), _mm256_set1_ps(0.5f)));
		const __m256 meanB = _mm256_mul_ps(_mm256_cvtepi32_ps(meanBInt), scale16v);
		// varB = MAXF((float)(sumB2*invN - meanBRawD*meanBRawD), 0.0001f)  (narrow BEFORE max)
		const __m256 varB = _mm256_max_ps(narrow(
			_mm256_sub_pd(_mm256_mul_pd(sumB2Lo, invNd), _mm256_mul_pd(meanBRawDLo, meanBRawDLo)),
			_mm256_sub_pd(_mm256_mul_pd(sumB2Hi, invNd), _mm256_mul_pd(meanBRawDHi, meanBRawDHi))), minVarv);
		const __m256 meanA = _mm256_mul_ps(_mm256_cvtepi32_ps(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(meanARow + i)))), scale16v);
		const __m256 varA = _mm256_loadu_ps(varARow + i);
		// product = MAXF(varA*varB, 1e-12f); invS = rsqrt + one Newton step,
		// mirrored order: invS *= 1.5f - 0.5f*product*invS*invS
		const __m256 product = _mm256_max_ps(_mm256_mul_ps(varA, varB), epsv);
		__m256 invS = _mm256_rsqrt_ps(product);
		invS = _mm256_mul_ps(invS, _mm256_sub_ps(_mm256_set1_ps(1.5f),
			_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), product), invS), invS)));
		// zncc = (float)(sumAB*invN - (double)meanA*meanB) * invS
		__m256d mALo, mAHi, mBLo, mBHi;
		WidenF32x8(meanA, mALo, mAHi);
		WidenF32x8(meanB, mBLo, mBHi);
		const __m256 zncc = _mm256_mul_ps(narrow(
			_mm256_sub_pd(_mm256_mul_pd(sumABLo, invNd), _mm256_mul_pd(mALo, mBLo)),
			_mm256_sub_pd(_mm256_mul_pd(sumABHi, invNd), _mm256_mul_pd(mAHi, mBHi))), invS);
		const __m256 zovb = _mm256_div_ps(zncc, varB); // IEEE divide, as scalar
		// rowMean = meanA*invS - meanB*zovb
		const __m256 meanTerm = _mm256_sub_ps(_mm256_mul_ps(meanA, invS), _mm256_mul_ps(meanB, zovb));

		// lane mask from the 8 mask bytes: all-ones where mask != 0
		const __m256i laneMaskI = _mm256_xor_si256(_mm256_cmpeq_epi32(
			_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)mask8)), zero256), _mm256_set1_epi32(-1));
		const __m256 laneMask = _mm256_castsi256_ps(laneMaskI);
		// masked stores: zeros stay zeros (memset) for the box-summed rings,
		// VarB keeps whatever the scalar path would have left in place
		_mm256_storeu_ps(rowInvS + i, _mm256_and_ps(laneMask, invS));
		_mm256_storeu_ps(rowZovb + i, _mm256_and_ps(laneMask, zovb));
		_mm256_storeu_ps(rowMean + i, _mm256_and_ps(laneMask, meanTerm));
		_mm256_storeu_ps(rowValid + i, _mm256_and_ps(laneMask, onev));
		_mm256_storeu_ps(rowVarB + i, _mm256_blendv_ps(_mm256_loadu_ps(rowVarB + i), varB, laneMask));

		if (score) {
			// scalar: score += (minV / (minV + 0.0015f)) * (1.f - zncc) at masked pixels
			const __m256 minV = _mm256_min_ps(varA, varB);
			const __m256 rel = _mm256_div_ps(minV, _mm256_add_ps(minV, _mm256_set1_ps(0.0015f)));
			alignas(32) float terms[8];
			_mm256_store_ps(terms, _mm256_and_ps(laneMask, _mm256_mul_ps(rel, _mm256_sub_ps(onev, zncc))));
			for (int k = 0; k < 8; ++k)
				*score += terms[k];
		}
	}
	// scalar tail: exact copy of the scalar term loop
	for (; i < colEnd; ++i) {
		if (!maskRow[i])
			continue;
		double sumB = 0.0, sumB2 = 0.0, sumAB = 0.0;
		for (int o = -2; o <= 2; ++o) {
			sumB += colB[i + o];
			sumB2 += colB2[i + o];
			sumAB += colAB[i + o];
		}
		const double meanBRawD = sumB * invN;
		const float meanBRaw = (float)meanBRawD;
		const float meanB = float((uint16_t)(meanBRaw * 65535.f + 0.5f)) * scale16;
		const float varB = (float)(sumB2 * invN - meanBRawD * meanBRawD) > 0.0001f
			? (float)(sumB2 * invN - meanBRawD * meanBRawD) : 0.0001f;
		const float meanA = float(meanARow[i]) * scale16;
		const float varA = varARow[i];
		float product = varA * varB > 1e-12f ? varA * varB : 1e-12f;
		float invS = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(product)));
		invS *= 1.5f - 0.5f * product * invS * invS;
		const float zncc = (float)(sumAB * invN - (double)meanA * meanB) * invS;
		const float zovb = zncc / varB;
		rowInvS[i] = invS;
		rowZovb[i] = zovb;
		rowMean[i] = meanA * invS - meanB * zovb;
		rowVarB[i] = varB;
		rowValid[i] = 1.f;
		if (score) {
			const float minV = varA < varB ? varA : varB;
			*score += (minV / (minV + 0.0015f)) * (1.f - zncc);
		}
	}
	_mm256_zeroupper();
}

// vertical ring accumulation for one emitted row: colX[c] += ringX[c] over the
// four planes, full width -- the same add order per column as the scalar loop
void SceneRefineWZNCCEmitAccumAVX2(
	float* __restrict ci, float* __restrict cz, float* __restrict cm, float* __restrict cn,
	const float* __restrict ri, const float* __restrict rz, const float* __restrict rm, const float* __restrict rv,
	int cols)
{
	int c = 0;
	for (; c + 8 <= cols; c += 8) {
		_mm256_storeu_ps(ci + c, _mm256_add_ps(_mm256_loadu_ps(ci + c), _mm256_loadu_ps(ri + c)));
		_mm256_storeu_ps(cz + c, _mm256_add_ps(_mm256_loadu_ps(cz + c), _mm256_loadu_ps(rz + c)));
		_mm256_storeu_ps(cm + c, _mm256_add_ps(_mm256_loadu_ps(cm + c), _mm256_loadu_ps(rm + c)));
		_mm256_storeu_ps(cn + c, _mm256_add_ps(_mm256_loadu_ps(cn + c), _mm256_loadu_ps(rv + c)));
	}
	for (; c < cols; ++c) {
		ci[c] += ri[c];
		cz[c] += rz[c];
		cm[c] += rm[c];
		cn[c] += rv[c];
	}
	_mm256_zeroupper();
}

// derivative emission for one output row: 5-tap horizontal sums of the four
// accumulated columns, then the windowed-average gradient. gradRow stores are
// masked blends, so unmasked pixels keep their previous (unread) bytes exactly
// like the scalar path.
void SceneRefineWZNCCEmitRowAVX2(
	const uint8_t* __restrict maskRow,
	const uint16_t* __restrict aRow,
	const uint16_t* __restrict bRow,
	const float* __restrict varARow,
	const float* __restrict varBRow,
	const float* __restrict colInv,
	const float* __restrict colZ,
	const float* __restrict colM,
	const float* __restrict colN,
	float* __restrict gradRow,
	int colStart,
	int colEnd)
{
	const float scale16 = 1.f / 65535.f;
	const __m256 scale16v = _mm256_set1_ps(scale16);
	const __m256 onev = _mm256_set1_ps(1.f);
	const __m256i zero256 = _mm256_setzero_si256();
	const auto sum5f = [](const float* __restrict base, int i) -> __m256 {
		__m256 s = _mm256_loadu_ps(base + i - 2);
		s = _mm256_add_ps(s, _mm256_loadu_ps(base + i - 1));
		s = _mm256_add_ps(s, _mm256_loadu_ps(base + i));
		s = _mm256_add_ps(s, _mm256_loadu_ps(base + i + 1));
		s = _mm256_add_ps(s, _mm256_loadu_ps(base + i + 2));
		return s;
	};
	int i = colStart;
	const int vecEnd = colStart + ((colEnd - colStart) & ~7);
	for (; i < vecEnd; i += 8) {
		uint64_t mask8;
		std::memcpy(&mask8, maskRow + i, sizeof(mask8));
		if (mask8 == 0)
			continue;
		const __m256 sInv = sum5f(colInv, i);
		const __m256 sZ = sum5f(colZ, i);
		const __m256 sM = sum5f(colM, i);
		const __m256 sN = sum5f(colN, i);
		// scalar: invCnt = 1.f / sN (sN >= 1 at masked pixels; garbage lanes are blended away)
		const __m256 invCnt = _mm256_div_ps(onev, sN);
		const __m256 aVal = _mm256_mul_ps(_mm256_cvtepi32_ps(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(aRow + i)))), scale16v);
		const __m256 bVal = _mm256_mul_ps(_mm256_cvtepi32_ps(
			_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(bRow + i)))), scale16v);
		const __m256 minV = _mm256_min_ps(_mm256_loadu_ps(varARow + i), _mm256_loadu_ps(varBRow + i));
		const __m256 reliability = _mm256_div_ps(minV, _mm256_add_ps(minV, _mm256_set1_ps(0.0015f)));
		// scalar: reliability * (bVal*sZ - aVal*sInv + sM) * invCnt
		const __m256 val = _mm256_mul_ps(_mm256_mul_ps(reliability,
			_mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(bVal, sZ), _mm256_mul_ps(aVal, sInv)), sM)), invCnt);
		const __m256i laneMaskI = _mm256_xor_si256(_mm256_cmpeq_epi32(
			_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)mask8)), zero256), _mm256_set1_epi32(-1));
		_mm256_storeu_ps(gradRow + i, _mm256_blendv_ps(_mm256_loadu_ps(gradRow + i), val, _mm256_castsi256_ps(laneMaskI)));
	}
	for (; i < colEnd; ++i) {
		if (!maskRow[i])
			continue;
		float sInv = 0.f, sZ = 0.f, sM = 0.f, sN = 0.f;
		for (int o = -2; o <= 2; ++o) {
			sInv += colInv[i + o];
			sZ += colZ[i + o];
			sM += colM[i + o];
			sN += colN[i + o];
		}
		const float invCnt = 1.f / sN;
		const float aVal = float(aRow[i]) * scale16;
		const float bVal = float(bRow[i]) * scale16;
		const float minV = varARow[i] < varBRow[i] ? varARow[i] : varBRow[i];
		const float reliability = minV / (minV + 0.0015f);
		gradRow[i] = reliability * (bVal * sZ - aVal * sInv + sM) * invCnt;
	}
	_mm256_zeroupper();
}

// ===========================================================================
// Photometric-gradient group kernel (see the header for the split contract)
// ===========================================================================
uint32_t SceneRefinePGGroupAVX2(
	const PGPairCtxAVX2& ctx,
	const uint8_t* __restrict maskRow,
	const float* __restrict depthRow,
	const float* __restrict dZNCCRow,
	float rowF, float rayRowMulX, float rayRowMulY, float rayRowMulZ,
	const float* __restrict nx8, const float* __restrict ny8, const float* __restrict nz8,
	int c0,
	float* __restrict sgOut)
{
	(void)rowF;
	// lane column coordinates, exact int -> float like the scalar (float)(int)c
	const __m256i laneIdx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	const __m256i cI = _mm256_add_epi32(_mm256_set1_epi32(c0), laneIdx);
	const __m256 cF = _mm256_cvtepi32_ps(cI);

	// mask bytes -> lane mask (nonzero byte == valid warp sample)
	uint64_t m8;
	std::memcpy(&m8, maskRow + c0, sizeof(m8));
	const __m256i zeroI = _mm256_setzero_si256();
	__m256i keepI = _mm256_xor_si256(_mm256_cmpeq_epi32(
		_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)m8)), zeroI), _mm256_set1_epi32(-1));

	// scalar: dxA = ((float)(int)c - cxA) * invFxA;
	//         ray = (rA0k*dxA + rayRowMulK) + rA2k
	const __m256 dxA = _mm256_mul_ps(_mm256_sub_ps(cF, _mm256_set1_ps(ctx.cxA)), _mm256_set1_ps(ctx.invFxA));
	const __m256 rayX = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ctx.rA00), dxA), _mm256_set1_ps(rayRowMulX)), _mm256_set1_ps(ctx.rA20));
	const __m256 rayY = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ctx.rA01), dxA), _mm256_set1_ps(rayRowMulY)), _mm256_set1_ps(ctx.rA21));
	const __m256 rayZ = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ctx.rA02), dxA), _mm256_set1_ps(rayRowMulZ)), _mm256_set1_ps(ctx.rA22));

	// scalar: lenSq = (x*x + y*y) + z*z; NdU = (Nx*x + Ny*y) + Nz*z
	const __m256 lenSq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rayX, rayX), _mm256_mul_ps(rayY, rayY)), _mm256_mul_ps(rayZ, rayZ));
	const __m256 Nx = _mm256_loadu_ps(nx8), Ny = _mm256_loadu_ps(ny8), Nz = _mm256_loadu_ps(nz8);
	const __m256 NdU = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Nx, rayX), _mm256_mul_ps(Ny, rayY)), _mm256_mul_ps(Nz, rayZ));
	// scalar rejects: NdU >= 0 || NdU*NdU < 0.01f*lenSq  =>  keep: NdU < 0 && NdU*NdU >= 0.01f*lenSq
	keepI = _mm256_and_si256(keepI, _mm256_castps_si256(_mm256_and_ps(
		_mm256_cmp_ps(NdU, _mm256_setzero_ps(), _CMP_LT_OQ),
		_mm256_cmp_ps(_mm256_mul_ps(NdU, NdU), _mm256_mul_ps(_mm256_set1_ps(0.01f), lenSq), _CMP_GE_OQ))));
	const __m256 invNd = _mm256_div_ps(_mm256_set1_ps(1.0f), NdU); // IEEE divide, dead lanes blended away

	// scalar: t0 = (p0*dx + p1*dy) + p2*dz (etc.); numX = t0*depthA + projCX
	const __m256 t0 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ctx.p0), rayX), _mm256_mul_ps(_mm256_set1_ps(ctx.p1), rayY)), _mm256_mul_ps(_mm256_set1_ps(ctx.p2), rayZ));
	const __m256 t1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ctx.p4), rayX), _mm256_mul_ps(_mm256_set1_ps(ctx.p5), rayY)), _mm256_mul_ps(_mm256_set1_ps(ctx.p6), rayZ));
	const __m256 tW = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ctx.p8), rayX), _mm256_mul_ps(_mm256_set1_ps(ctx.p9), rayY)), _mm256_mul_ps(_mm256_set1_ps(ctx.p10), rayZ));
	const __m256 depthA = _mm256_loadu_ps(depthRow + c0);
	const __m256 numX = _mm256_add_ps(_mm256_mul_ps(t0, depthA), _mm256_set1_ps(ctx.projCX));
	const __m256 numY = _mm256_add_ps(_mm256_mul_ps(t1, depthA), _mm256_set1_ps(ctx.projCY));
	const __m256 denW = _mm256_add_ps(_mm256_mul_ps(tW, depthA), _mm256_set1_ps(ctx.projCW));
	const __m256 invW = _mm256_div_ps(_mm256_set1_ps(1.0f), denW);
	const __m256 xB = _mm256_mul_ps(numX, invW);
	const __m256 yB = _mm256_mul_ps(numY, invW);

	// sample coords (gradShift is 0.f in the centered-kernel config; x - 0.f is exact)
	const __m256 gsx = _mm256_sub_ps(xB, _mm256_set1_ps(ctx.gradShift));
	const __m256 gsy = _mm256_sub_ps(yB, _mm256_set1_ps(ctx.gradShift));
	const __m256i xi = _mm256_cvttps_epi32(gsx); // == _cvt_ftoi_fast truncation
	const __m256i yi = _mm256_cvttps_epi32(gsy);
	// scalar: (unsigned)xi > (unsigned)maxX rejects negatives and overflow alike
	keepI = _mm256_and_si256(keepI, _mm256_cmpgt_epi32(xi, _mm256_set1_epi32(-1)));
	keepI = _mm256_and_si256(keepI, _mm256_cmpgt_epi32(_mm256_set1_epi32(ctx.maxX + 1), xi));
	keepI = _mm256_and_si256(keepI, _mm256_cmpgt_epi32(yi, _mm256_set1_epi32(-1)));
	keepI = _mm256_and_si256(keepI, _mm256_cmpgt_epi32(_mm256_set1_epi32(ctx.maxY + 1), yi));
	const uint32_t valid = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(keepI));
	if (valid == 0) {
		_mm256_zeroupper();
		return 0;
	}

	const __m256 fx = _mm256_sub_ps(gsx, _mm256_cvtepi32_ps(xi));
	const __m256 fy = _mm256_sub_ps(gsy, _mm256_cvtepi32_ps(yi));
	// int16-plane element offsets; a 32-bit gather at scale 2 reads the tap
	// pair {off, off+1}, exactly like the scalar *(const int*)(plane + off)
	__m256i off = _mm256_add_epi32(_mm256_mullo_epi32(yi, _mm256_set1_epi32(ctx.wB)), xi);
	off = _mm256_and_si256(off, keepI); // clamp dead lanes to 0 (their gather is masked anyway)
	const __m256i offBot = _mm256_add_epi32(off, _mm256_set1_epi32(ctx.wB));
	const __m256i gxTop16 = _mm256_mask_i32gather_epi32(zeroI, (const int*)ctx.gradXB, off, keepI, 2);
	const __m256i gxBot16 = _mm256_mask_i32gather_epi32(zeroI, (const int*)ctx.gradXB, offBot, keepI, 2);
	const __m256i gyTop16 = _mm256_mask_i32gather_epi32(zeroI, (const int*)ctx.gradYB, off, keepI, 2);
	const __m256i gyBot16 = _mm256_mask_i32gather_epi32(zeroI, (const int*)ctx.gradYB, offBot, keepI, 2);

	// dequantize the two int16 taps of each gather: low tap via shift-up/down
	// sign extension, high tap via arithmetic shift -- then * invScale, exactly
	// like the scalar cvtepi16_epi32 + cvtepi32_ps + mul
	const __m256 invScalev = _mm256_set1_ps(ctx.invScale);
	const auto lowTap = [&invScalev](__m256i g) -> __m256 {
		return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_slli_epi32(g, 16), 16)), invScalev);
	};
	const auto highTap = [&invScalev](__m256i g) -> __m256 {
		return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(g, 16)), invScalev);
	};
	// merged-bilerp per-pixel order: vertical lerp per tap, then horizontal
	//   a = v00 + fy*(v10 - v00); b = v01 + fy*(v11 - v01); g = a + fx*(b - a)
	const auto bilerp = [&fx, &fy](__m256 v00, __m256 v01, __m256 v10, __m256 v11) -> __m256 {
		const __m256 a = _mm256_add_ps(v00, _mm256_mul_ps(fy, _mm256_sub_ps(v10, v00)));
		const __m256 b = _mm256_add_ps(v01, _mm256_mul_ps(fy, _mm256_sub_ps(v11, v01)));
		return _mm256_add_ps(a, _mm256_mul_ps(fx, _mm256_sub_ps(b, a)));
	};
	const __m256 gBx = bilerp(lowTap(gxTop16), highTap(gxTop16), lowTap(gxBot16), highTap(gxBot16));
	const __m256 gBy = bilerp(lowTap(gyTop16), highTap(gyTop16), lowTap(gyBot16), highTap(gyBot16));

	// scalar: dot0 = t0 - xB*tW; dot1 = t1 - yB*tW;
	//         sg = ((((gBx*dot0 + gBy*dot1) * invW) * invNd) * Reg) * dZNCC
	const __m256 dot0 = _mm256_sub_ps(t0, _mm256_mul_ps(xB, tW));
	const __m256 dot1 = _mm256_sub_ps(t1, _mm256_mul_ps(yB, tW));
	const __m256 dZNCC = _mm256_loadu_ps(dZNCCRow + c0);
	const __m256 sg = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(
		_mm256_add_ps(_mm256_mul_ps(gBx, dot0), _mm256_mul_ps(gBy, dot1)),
		invW), invNd), _mm256_set1_ps(ctx.regScale)), dZNCC);
	_mm256_storeu_ps(sgOut, sg);
	_mm256_zeroupper();
	return valid;
}

} // namespace MVS
