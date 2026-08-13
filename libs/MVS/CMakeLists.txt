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
	set_target_properties(MVS PROPERTIES CUDA_ARCHITECTURES "50;72;75")
endif()
# TODO: CGAL::TBB_support should be controlled by a configuration option
TARGET_LINK_LIBRARIES(MVS PRIVATE Common Math IO CGAL::CGAL CGAL::TBB_support ${CERES_LIBRARIES} ${CUDA_CUDA_LIBRARY} PoissonReconLib)

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
