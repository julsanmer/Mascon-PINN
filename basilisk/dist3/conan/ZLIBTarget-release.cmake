
set(ZLIB_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include")
set(ZLIB_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include")
set(ZLIB_INCLUDES_RELEASE "/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include")
set(ZLIB_RES_DIRS_RELEASE )
set(ZLIB_DEFINITIONS_RELEASE )
set(ZLIB_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(ZLIB_COMPILE_DEFINITIONS_RELEASE )
set(ZLIB_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(ZLIB_COMPILE_OPTIONS_C_RELEASE "")
set(ZLIB_COMPILE_OPTIONS_CXX_RELEASE "")
set(ZLIB_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(ZLIB_LIBRARIES_RELEASE "") # Will be filled later
set(ZLIB_LIBS_RELEASE "") # Same as ZLIB_LIBRARIES
set(ZLIB_SYSTEM_LIBS_RELEASE )
set(ZLIB_FRAMEWORK_DIRS_RELEASE )
set(ZLIB_FRAMEWORKS_RELEASE )
set(ZLIB_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(ZLIB_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(ZLIB_FRAMEWORKS_FOUND_RELEASE "${ZLIB_FRAMEWORKS_RELEASE}" "${ZLIB_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(ZLIB_INCLUDE_DIRS_RELEASE
                 ZLIB_INCLUDE_DIR_RELEASE
                 ZLIB_INCLUDES_RELEASE
                 ZLIB_DEFINITIONS_RELEASE
                 ZLIB_LINKER_FLAGS_RELEASE_LIST
                 ZLIB_COMPILE_DEFINITIONS_RELEASE
                 ZLIB_COMPILE_OPTIONS_RELEASE_LIST
                 ZLIB_LIBRARIES_RELEASE
                 ZLIB_LIBS_RELEASE
                 ZLIB_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to ZLIB_LIBS and ZLIB_LIBRARY_LIST
set(ZLIB_LIBRARY_LIST_RELEASE z)
set(ZLIB_LIB_DIRS_RELEASE "/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_ZLIB_DEPENDENCIES_RELEASE "${ZLIB_FRAMEWORKS_FOUND_RELEASE} ${ZLIB_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${ZLIB_LIBRARY_LIST_RELEASE}"  # libraries
                              "${ZLIB_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_ZLIB_DEPENDENCIES_RELEASE}"  # deps
                              ZLIB_LIBRARIES_RELEASE            # out_libraries
                              ZLIB_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "ZLIB")                                      # package_name

set(ZLIB_LIBS_RELEASE ${ZLIB_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${ZLIB_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND ZLIB_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND ZLIB_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${ZLIB_SYSTEM_LIBS_RELEASE})
    list(APPEND ZLIB_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND ZLIB_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(ZLIB_LIBRARIES_TARGETS_RELEASE "${ZLIB_LIBRARIES_TARGETS_RELEASE};")
set(ZLIB_LIBRARIES_RELEASE "${ZLIB_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/" ${CMAKE_PREFIX_PATH})
