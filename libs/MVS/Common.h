/*
* Common.h
*
* Copyright (c) 2014-2015 SEACAVE
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

#ifndef _MVS_COMMON_H_
#define _MVS_COMMON_H_


// I N C L U D E S /////////////////////////////////////////////////

#if defined(MVS_EXPORTS) && !defined(Common_EXPORTS)
#define Common_EXPORTS
#endif

#include "../Common/Common.h"
#include "../IO/Common.h"
#include "../Math/Common.h"

#ifndef MVS_API
#define MVS_API GENERAL_API
#endif
#ifndef MVS_TPL
#define MVS_TPL GENERAL_TPL
#endif


// D E F I N E S ///////////////////////////////////////////////////
// 3 is noticeably better than 2, 4 not so.
#define DPC_NUM_ITERS (3) // Was 4 in _USE_CUDA pathway, 3 not.
#define DPC_IMAGE_CACHE
#undef DPC_EXTENDED_OMP_THREADING // Changes threading order and drastically affects the result.
#define DPC_EXTENDED_OMP_THREADING2
#define DPC_NEW_FUSING
#define DPC_FLUSH_DENORMALS

// Use a faster, but less accurate, exp function in score factor calculation?
#define DPC_FASTER_SCORE_FACTOR

// Reduce the precision in Dir2Normal and Normal2Dir to improve performance?
#define DPC_FASTER_RANDOM_ITER_CALC

#define DPC_FASTER_SAMPLING

// Disabled due to precision problems in sampling.
#undef DPC_FASTER_SAMPLING_USE_INV_Z

// Reduce the detail calculation accuracy to improve performance?
#define DPC_FASTER_SCORE_PIXEL_DETAIL
#define DPC_FASTER_SCORE_PIXEL_DETAIL2

// Use a parallel version of pca_estimate_normals (requires TBB)?
// JPB WIP OPT Restore when the support for this can be added to the build.
#define DPC_FASTER_NORMAL_ESTIMATION

// DPC_FASTER_SAMPLING related:


// P R O T O T Y P E S /////////////////////////////////////////////

using namespace SEACAVE;

#define _USE_OPENCV
#define _DISABLE_NO_ID
#include "Interface.h"

namespace MVS {

// Initialize / close the library; should be called at the beginning and end of the program
void Initialize(LPCTSTR appname, unsigned nMaxThreads=0, int nProcessPriority=0);
void Finalize();
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_COMMON_H_
