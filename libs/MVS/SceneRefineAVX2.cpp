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

} // namespace MVS
