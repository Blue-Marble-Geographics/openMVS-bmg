# Find required packages
FIND_PACKAGE(CGAL REQUIRED)
if(CGAL_FOUND)
	include_directories(${CGAL_INCLUDE_DIRS})
	add_definitions(${CGAL_DEFINITIONS})
	link_directories(${CGAL_LIBRARY_DIRS})
endif()

FIND_PACKAGE(VCG REQUIRED)
if(VCG_FOUND)
	include_directories(${VCG_INCLUDE_DIRS})
	add_definitions(${VCG_DEFINITIONS})
endif()

set(CERES_LIBS "")
if(OpenMVS_USE_CERES)
	FIND_PACKAGE(Ceres)
	if(CERES_FOUND)
		include_directories(${CERES_INCLUDE_DIRS})
		add_definitions(${CERES_DEFINITIONS})
	else()
		set(OpenMVS_USE_CERES OFF)
		message("-- Can't find CERES. Continuing without it.")
	endif()
endif()

# List sources files
FILE(GLOB LIBRARY_FILES_C "*.cpp")
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")
# Local source snapshots are not part of the library. FILE(GLOB) otherwise
# picks them up on every CMake regeneration when a legitimate source is added.
list(FILTER LIBRARY_FILES_C EXCLUDE REGEX ".*( -|Copy).*")
list(FILTER LIBRARY_FILES_C EXCLUDE REGEX ".*[.]bak$")

# In-process Kazhdan PoissonRecon + SurfaceTrimmer, replacing the external-exe
# shell-out. Built as an isolated static library so that only these two wrapper
# translation units see the (heavily-templated) PoissonRecon headers and its
# PreProcessor.h macros; the rest of MVS is unaffected. SurfaceTrimmer.cpp is
# compiled with POISSONRECON_LIB_BUILD so its main()/CmdLine globals are excluded.
set(POISSONRECON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../PoissonRecon/Src")
add_library(PoissonReconLib STATIC
	"${POISSONRECON_SRC_DIR}/PoissonReconLib.cpp"
	"${POISSONRECON_SRC_DIR}/SurfaceTrimmer.cpp"
)
target_include_directories(PoissonReconLib PRIVATE "${POISSONRECON_SRC_DIR}")
target_compile_definitions(PoissonReconLib PRIVATE POISSONRECON_LIB_BUILD NOMINMAX _CRT_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE)
if(MSVC)
	target_compile_options(PoissonReconLib PRIVATE /bigobj)
	# Debug info on the optimized Release build so the generated machine code can be
	# inspected (PDB usable by the VS debugger's disassembly view or `dumpbin /DISASM`).
	# /O2 is kept. Release-only, so other configs are unaffected.
	# NOTE: do NOT add /FA* assembly-listing flags here — this build uses LTCG (/GL),
	# which defers code generation to link time, so the compile-time .cod write fails
	# (C1083 'Cannot open ... .cod' -> LNK1257). Use the debugger/dumpbin instead.
	target_compile_options(PoissonReconLib PRIVATE $<$<CONFIG:Release>:/Zi>)
endif()
if(_USE_CUDA)
	FILE(GLOB LIBRARY_FILES_CUDA "*.cu")
	LIST(APPEND LIBRARY_FILES_C ${LIBRARY_FILES_CUDA})
endif()
FILE(GLOB CUDA_LIBRARY_FILES_C "CUDA/*.cpp")
FILE(GLOB CUDA_LIBRARY_FILES_H "CUDA/*.h" "CUDA/*.inl")
if(_USE_CUDA)
	FILE(GLOB CUDA_LIBRARY_FILES_CUDA "CUDA/*.cu")
	LIST(APPEND CUDA_LIBRARY_FILES_C ${CUDA_LIBRARY_FILES_CUDA})
endif()
SOURCE_GROUP("CUDA" FILES ${CUDA_LIBRARY_FILES_C} ${CUDA_LIBRARY_FILES_H})

cxx_library_with_type(MVS "Libs" "" "${cxx_default}"
	${LIBRARY_FILES_C} ${LIBRARY_FILES_H}
	${CUDA_LIBRARY_FILES_C} ${CUDA_LIBRARY_FILES_H}
)

# A/B switch for DensifyPointCloud's CUDA PatchMatch depth-map estimator:
# OFF (default) builds the reworked estimator (PatchMatchCUDA.inl/.cpp/.cu), ON
# builds the original one restored from commit 235b9712 (PatchMatchCUDALegacy.*).
# Both variants are always in the source list -- the unselected one preprocesses
# away -- so flipping this needs only a re-configure, not a file-list edit.
# See PatchMatchCUDASelect.h for what actually differs. PUBLIC because
# PatchMatchCUDA.h is an installed public header whose class definition depends
# on the value.
option(OpenMVS_PATCHMATCH_CUDA_LEGACY "Build the original (pre-rework) CUDA PatchMatch estimator for A/B testing" OFF)

# Make nvcc report per-kernel register counts and local-memory spill. The
# checkerboard kernels carry enough per-thread state to spill, and spill traffic
# (not texture throughput) is the suspected limit -- see the occupancy notes in
# PatchMatchCUDA.cu. Off by default because it is noisy; turn it on when tuning
# PMCUDA_BLOCK_H / PMCUDA_MIN_BLOCKS and read the "spill stores"/"registers"
# lines for BlackPixelProcess and RedPixelProcess.
option(OpenMVS_CUDA_PTXAS_VERBOSE "Report CUDA register usage and local-memory spill at compile time" OFF)
if(_USE_CUDA AND OpenMVS_CUDA_PTXAS_VERBOSE)
	target_compile_options(MVS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()
if(_USE_CUDA AND OpenMVS_PATCHMATCH_CUDA_LEGACY)
	target_compile_definitions(MVS PUBLIC PATCHMATCH_CUDA_LEGACY=1)
	message(STATUS "MVS: CUDA PatchMatch estimator = LEGACY (original, 235b9712)")
elseif(_USE_CUDA)
	message(STATUS "MVS: CUDA PatchMatch estimator = current (reworked)")
endif()

# Keep AVX2 code in one translation unit so the rest of MVS retains the
# baseline x64/SSE2 instruction set. Disable LTCG for this source to prevent
# cross-module inlining from leaking AVX2 instructions into baseline code.
if(MSVC)
	set_source_files_properties(SceneRefineAVX2.cpp PROPERTIES
		COMPILE_OPTIONS "/arch:AVX2;/GL-"
		SKIP_PRECOMPILE_HEADERS ON)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	set_source_files_properties(SceneRefineAVX2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mno-fma")
endif()

# Manually set Common.h as the precompiled header
IF(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16.0)
	TARGET_PRECOMPILE_HEADERS(MVS PRIVATE "Common.h")
endif()

# Link its dependencies
if(_USE_CUDA)
	# Native SASS for every desktop generation end-users actually run, gated by
	# what the building toolkit supports. The former hardcoded "50;72;75" left
	# every Ampere/Ada/Blackwell card JIT-compiling the compute_75 PTX at first
	# launch (start-up stalls, re-JITs whenever the driver cache evicts, no
	# arch-tuned code), and sm_50/sm_72 are deprecated in CUDA 12 and REMOVED in
	# CUDA 13, where they break the build outright (sm_72 is an embedded Tegra
	# target no desktop Windows user has). Plain numeric entries embed both SASS
	# and PTX per arch, so the newest PTX keeps forward-compatibility with GPUs
	# newer than the toolkit.
	# An explicit -DCMAKE_CUDA_ARCHITECTURES is respected; the value "75" is
	# indistinguishable from the root CMakeLists' fallback default, so it is
	# treated as unset (native 75 stays in the list regardless). When
	# OpenMVS_MAX_CUDA_COMPATIBILITY is on, the -gencode flags assembled in the
	# root CMakeLists carry the arch list instead.
	if(OpenMVS_MAX_CUDA_COMPATIBILITY OR (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "75"))
		set_target_properties(MVS PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
	else()
		set(MVS_CUDA_ARCHS 75)
		if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
			list(APPEND MVS_CUDA_ARCHS 86)
		endif()
		if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8")
			list(APPEND MVS_CUDA_ARCHS 89 90)
		endif()
		if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
			list(APPEND MVS_CUDA_ARCHS 100 120)
		endif()
		message(STATUS "MVS CUDA architectures: ${MVS_CUDA_ARCHS}")
		set_target_properties(MVS PROPERTIES CUDA_ARCHITECTURES "${MVS_CUDA_ARCHS}")
	endif()
endif()
# TODO: CGAL::TBB_support should be controlled by a configuration option
TARGET_LINK_LIBRARIES(MVS PRIVATE Common Math IO CGAL::CGAL CGAL::TBB_support ${CERES_LIBRARIES} ${CUDA_CUDA_LIBRARY} PoissonReconLib)

# Delay-load the CUDA driver library. Statically bound, nvcuda.dll is resolved by
# the loader before main() runs, so a machine with no NVIDIA driver fails the
# process with 0xC0000135 and one whose driver dropped the legacy entry points we
# still import fails it with 0xC0000139 -- in both cases killing CPU-only runs
# (--cuda-device -2) that never wanted CUDA at all. Delay-loading binds each
# entry point on first call instead, so the CPU path never touches the driver and
# the GPU path behaves exactly as before on machines where it already works.
# CUDA::HasLegacyDriverAPI() is what keeps a first call from ever happening on an
# incompatible driver.
# PUBLIC, not PRIVATE: MVS is normally a static library, so the flag has to reach
# the executables' link line, which is where the delay-load stubs are generated.
if(_USE_CUDA AND MSVC)
	if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13.0)
		target_link_libraries(MVS PUBLIC delayimp)
		target_link_options(MVS PUBLIC "/DELAYLOAD:nvcuda.dll")
	else()
		message(WARNING "CMake < 3.13 cannot propagate /DELAYLOAD:nvcuda.dll to the apps; "
						"RefineMesh will keep a hard load-time dependency on the CUDA driver.")
	endif()
endif()

# Install
SET_TARGET_PROPERTIES(MVS PROPERTIES
	PUBLIC_HEADER "${LIBRARY_FILES_H}")
INSTALL(TARGETS MVS
	EXPORT OpenMVSTargets
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/MVS")

# PoissonReconLib is a static library statically linked into MVS. Because MVS is
# exported via OpenMVSTargets and its objects are not absorbed into the MVS
# archive, consumers of the static MVS library must also link PoissonReconLib;
# therefore it has to be part of the same export set or the install(EXPORT ...)
# step fails with "requires target PoissonReconLib that is not in any export set".
INSTALL(TARGETS PoissonReconLib
	EXPORT OpenMVSTargets
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
