# Kazhdan PoissonRecon + SurfaceTrimmer (MIT-licensed) — used by
# Scene::ReconstructMeshPoisson via shell-out (ReconstructMesh --poisson).
#
# These tools ship with their OWN Visual Studio projects. Rather than re-author a
# CMake build for them (PoissonRecon.cpp pulls in the bundled PNG/JPEG/ZLIB image
# libs, plus OpenMP / /bigobj / AVX settings), we pull the existing .vcxproj files
# directly into the CMake-generated OpenMVS solution with include_external_msproject,
# so they build with their proven settings and appear alongside the OpenMVS targets.
#
# CAVEATS (read before enabling):
#  * VS-generator only (include_external_msproject). MSVC required.
#  * The vcxproj define only Debug|Release (Win32/x64); the OpenMVS solution also
#    has RelWithDebInfo/MinSizeRel. Build the solution in Release|x64 (or Debug).
#    The non-matching configs simply won't build these tool projects.
#  * The exes land in each vcxproj's own OutDir, NOT next to ReconstructMesh.exe.
#    After building, either copy PoissonRecon.exe + SurfaceTrimmer.exe next to
#    ReconstructMesh.exe, or set the OPENMVS_POISSONRECON_DIR environment variable
#    to their output folder (the shell-out checks that first).

if(NOT MSVC)
	message(WARNING "PoissonRecon tools integration is MSVC-only; skipping.")
	return()
endif()

set(PR_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Bundled image/compression libs (PoissonRecon.vcxproj links ZLIB.lib, PNG.lib,
# and JPEG-turbo.lib — the JPEG lib is the libjpeg-turbo project in the
# JPEG-turbo/ subfolder, NOT the unused root JPEG.vcxproj).
include_external_msproject(ZLIB           "${PR_DIR}/ZLIB.vcxproj")
include_external_msproject(PNG            "${PR_DIR}/PNG.vcxproj"  ZLIB)
include_external_msproject(JPEG-turbo     "${PR_DIR}/JPEG-turbo/JPEG-turbo.vcxproj")

# The two tools we actually invoke. PoissonRecon links the image libs (build-order
# dependency); SurfaceTrimmer is PLY-only and needs none.
include_external_msproject(PoissonRecon   "${PR_DIR}/PoissonRecon.vcxproj"   ZLIB PNG JPEG-turbo)
include_external_msproject(SurfaceTrimmer "${PR_DIR}/SurfaceTrimmer.vcxproj")

# include_external_msproject builds each tool into its own vcxproj OutDir, NOT the
# OpenMVS runtime bin dir (build*/bin/vcNN/x64/Release). Copy the two exes next to
# ReconstructMesh.exe after they build, so the --poisson shell-out finds them via
# Util::getProcessFolder(). DST is resolved from the ReconstructMesh target so it
# always matches the real bin dir regardless of generator/toolset.
if(TARGET ReconstructMesh)
	add_custom_target(PoissonReconToolsCopy ALL
		COMMAND ${CMAKE_COMMAND}
			-DPR_DIR=${CMAKE_BINARY_DIR}
			-DDST_DIR=$<TARGET_FILE_DIR:ReconstructMesh>
			-DCFG=$<CONFIG>
			-P "${CMAKE_CURRENT_SOURCE_DIR}/copy_tools.cmake"
		COMMENT "Copying PoissonRecon + SurfaceTrimmer exes next to ReconstructMesh.exe"
		VERBATIM)
	add_dependencies(PoissonReconToolsCopy PoissonRecon SurfaceTrimmer)
else()
	message(WARNING "PoissonRecon tools: ReconstructMesh target not found; exes will build to their own OutDir (set OPENMVS_POISSONRECON_DIR at runtime).")
endif()
