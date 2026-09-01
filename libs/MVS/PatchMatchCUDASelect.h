/*
* PatchMatchCUDASelect.h
*
* Copyright (c) 2014-2021 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#ifndef _MVS_PATCHMATCHCUDASELECT_H_
#define _MVS_PATCHMATCHCUDASELECT_H_


// D E F I N E S ///////////////////////////////////////////////////

// A/B switch for DensifyPointCloud's CUDA PatchMatch depth-map estimator.
//
//   0 (default) - the current, reworked estimator:
//                 PatchMatchCUDA.inl / .cpp / .cu
//                 VRAM texture LRU cache across reference images, fp16 image
//                 textures, pinned staging + async stream, stateless xorshift
//                 RNG, on-device pack/unpack kernels, per-pair precomputed
//                 plane-homography constants.
//
//   1           - the original estimator, restored verbatim from commit
//                 235b9712: PatchMatchCUDALegacy.inl / .cpp / .cu
//                 per-image upload-and-free on every call, fp32 textures,
//                 curand state array in VRAM, host-side pack/unpack loops.
//
// Both variants live side by side and expose the same MVS::CUDA::PatchMatch
// interface; exactly one is compiled. Every file of the unselected variant
// preprocesses away to a single typedef, so switching needs no CMake edit
// beyond the option below and no change at any call site.
//
// Select the legacy build with either
//   cmake -DOpenMVS_PATCHMATCH_CUDA_LEGACY=ON ...
// or by defining PATCHMATCH_CUDA_LEGACY=1 on the compiler command line.
//
// SCOPE: DensifyPointCloud's depth-map estimator only. The RefineMesh CUDA
// path (SceneRefineCUDA) and everything else in SceneDensify -- image caching,
// worker threading, fusion, filtering -- are outside this switch and stay as
// they are in both configurations. That is deliberate: it keeps the A/B
// comparison to the estimator itself.
#ifndef PATCHMATCH_CUDA_LEGACY
#define PATCHMATCH_CUDA_LEGACY 0
#endif

#endif // _MVS_PATCHMATCHCUDASELECT_H_
