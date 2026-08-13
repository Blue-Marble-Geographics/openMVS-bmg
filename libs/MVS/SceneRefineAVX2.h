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
