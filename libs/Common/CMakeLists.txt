# List sources files
FILE(GLOB LIBRARY_FILES_C "*.cpp")
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")

cxx_library_with_type(Common "Libs" "" "${cxx_default}"
	${LIBRARY_FILES_C} ${LIBRARY_FILES_H}
)

# Manually set Common.h as the precompiled header
IF(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16.0)
	TARGET_PRECOMPILE_HEADERS(Common PRIVATE "Common.h")
endif()

# Link its dependencies
# Prefer the vcpkg CONFIG imported target (carries full lib path + include dirs,
# portable across machines); fall back to the bare library name for environments
# where libjpeg-turbo is provided outside vcpkg and the imported target does not
# exist (otherwise CMake passes "libjpeg-turbo::turbojpeg" to the linker verbatim
# -> LNK1104 cannot open file 'libjpeg-turbo::turbojpeg.lib').
if(TARGET libjpeg-turbo::turbojpeg)
	TARGET_LINK_LIBRARIES(Common ${Boost_LIBRARIES} ${OpenCV_LIBS} libjpeg-turbo::turbojpeg)
else()
	TARGET_LINK_LIBRARIES(Common ${Boost_LIBRARIES} ${OpenCV_LIBS} turbojpeg)
endif()

# Install
SET_TARGET_PROPERTIES(Common PROPERTIES
	PUBLIC_HEADER "${LIBRARY_FILES_H}")
INSTALL(TARGETS Common
	EXPORT OpenMVSTargets
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Common")
