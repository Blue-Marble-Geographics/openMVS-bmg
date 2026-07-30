# Copy the freshly-built PoissonRecon.exe + SurfaceTrimmer.exe into DST_DIR
# (the OpenMVS runtime bin dir), wherever their own vcxproj placed them.
# Invoked at build time via:  cmake -DPR_DIR=.. -DDST_DIR=.. -DCFG=.. -P copy_tools.cmake
#
# We glob all .exe under PR_DIR and match by filename, preferring a build path
# that contains the active configuration (Release/Debug), so we don't have to
# hard-code the vcxproj OutDir layout.

file(GLOB_RECURSE _all_exes "${PR_DIR}/*.exe")

foreach(_tool PoissonRecon SurfaceTrimmer)
	set(_picked "")
	# first pass: a candidate whose path contains the active config
	foreach(_exe ${_all_exes})
		get_filename_component(_name "${_exe}" NAME)
		if(_name STREQUAL "${_tool}.exe" AND _exe MATCHES "[/\\]${CFG}[/\\]")
			set(_picked "${_exe}")
			break()
		endif()
	endforeach()
	# fallback: any candidate with the right name
	if(NOT _picked)
		foreach(_exe ${_all_exes})
			get_filename_component(_name "${_exe}" NAME)
			if(_name STREQUAL "${_tool}.exe")
				set(_picked "${_exe}")
				break()
			endif()
		endforeach()
	endif()

	if(_picked)
		file(COPY "${_picked}" DESTINATION "${DST_DIR}")
		message(STATUS "PoissonRecon tools: copied ${_picked} -> ${DST_DIR}")
	else()
		message(WARNING "PoissonRecon tools: ${_tool}.exe not found under ${PR_DIR} (was it built?)")
	endif()
endforeach()
