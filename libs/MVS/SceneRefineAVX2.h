#pragma once

#include <cstdint>

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
	float* score);

// Eight-wide ImageMeshWarp row kernel (direct-validity + u16 image config).
// Mirrors the scalar per-pixel pipeline op-for-op (mul/add kept separate, IEEE
// div, truncating converts) so the result is bit-identical; the early-out
// cascade becomes progressive lane masks and all B-side reads use masked
// gathers (dead lanes never touch memory). Returns the first unprocessed
// column; the caller runs the scalar tail from there.
// Eight-wide priming / rolling updates for the fused-ZNCC column sums
// (u16-image config). Unlike the kernels above these use NO FMA: they mirror
// the scalar loops op-for-op and are BIT-IDENTICAL to them -- the sums they
// maintain accumulate across thousands of row slides.
void SceneRefineZNCCPrimeRowAVX2(
	const uint16_t* __restrict aRow,
	const uint16_t* __restrict bRow,
	float* __restrict colB,
	float* __restrict colB2,
	float* __restrict colAB,
	int x0,
	int x1);
void SceneRefineZNCCRollRowAVX2(
	const uint16_t* __restrict addA,
	const uint16_t* __restrict remA,
	const uint16_t* __restrict addB,
	const uint16_t* __restrict remB,
	float* __restrict colB,
	float* __restrict colB2,
	float* __restrict colAB,
	int x0,
	int x1);

// Windowed-derivative ZNCC kernels (MESHOPT_DZNCC_WINDOWED config): priming /
// rolling of the DOUBLE column sums, the per-row derivative-term pass, the
// emit-side ring accumulation and the windowed-average emission. All mirror
// their scalar loops op-for-op with masked blend-stores: byte-identical
// output. They hardcode the 5-tap (HalfSize==2) ZNCC window -- the call site
// static_asserts that.
void SceneRefineWZNCCPrimeRowAVX2(
	const uint16_t* __restrict aRow,
	const uint16_t* __restrict bRow,
	double* __restrict colB,
	double* __restrict colB2,
	double* __restrict colAB,
	int cols);
void SceneRefineWZNCCRollRowAVX2(
	const uint16_t* __restrict addA,
	const uint16_t* __restrict remA,
	const uint16_t* __restrict addB,
	const uint16_t* __restrict remB,
	double* __restrict colB,
	double* __restrict colB2,
	double* __restrict colAB,
	int cols);
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
	float* score);
void SceneRefineWZNCCEmitAccumAVX2(
	float* __restrict ci, float* __restrict cz, float* __restrict cm, float* __restrict cn,
	const float* __restrict ri, const float* __restrict rz, const float* __restrict rm, const float* __restrict rv,
	int cols);
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
	int colEnd);

// Eight-wide core of ComputePhotometricGradient's per-pixel pipeline (the
// unnorm-ray + face-cache + int16-gradient-planes + merged-bilerp config):
// ray reconstruction, squared-form grazing test, projection into B, masked
// 2x2 bilinear gradient gathers, Jacobian dots and the gradient scale sg.
// Mirrors the scalar pipeline op-for-op (no FMA, IEEE divides, cvtt
// truncation): each surviving lane's sg is bit-identical to the scalar value.
// The caller does the scalar prepass (face-setup cache, per-lane normals) and
// the scalar per-lane tail (barycentric + tile-slot accumulation) in pixel
// order, so the accumulation order -- and therefore the result -- is
// byte-identical to the scalar path.
struct PGPairCtxAVX2 {
	const int16_t* gradXB;   // planar int16 gradient planes of view B
	const int16_t* gradYB;
	float rA00, rA01, rA02;  // camera A rotation rows (ray reconstruction)
	float rA20, rA21, rA22;
	float cxA, invFxA;
	float p0, p1, p2, p4, p5, p6, p8, p9, p10; // camera B projection rows
	float projCX, projCY, projCW;              // P*C + last column, folded
	float invScale;          // int16 gradient dequantization (kInvScale)
	float regScale;          // RegularizationScale
	float gradShift;         // 0.5f when MESHOPT_PG_GRAD_CENTERED else 0.f (x-0.f is exact)
	int maxX, maxY, wB;
};
// Processes lanes c0..c0+7 of one row. nx/ny/nz hold the per-lane face
// normals from the caller's prepass (zeros at unmasked lanes). Returns the
// 8-bit validity mask of lanes that survived mask + grazing + bounds;
// sgOut[k] is meaningful only for valid lanes. Dead lanes never touch the
// gradient planes (masked gathers).
uint32_t SceneRefinePGGroupAVX2(
	const PGPairCtxAVX2& ctx,
	const uint8_t* __restrict maskRow,
	const float* __restrict depthRow,
	const float* __restrict dZNCCRow,
	float rowF, float rayRowMulX, float rayRowMulY, float rayRowMulZ,
	const float* __restrict nx8, const float* __restrict ny8, const float* __restrict nz8,
	int c0,
	float* __restrict sgOut);

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
	uint8_t* __restrict maskRow);

} // namespace MVS
