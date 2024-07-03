
set(libbacktrace_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/libbacktrace/cci.20210118/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include")
set(libbacktrace_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/libbacktrace/cci.20210118/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include")
set(libbacktrace_INCLUDES_RELEASE "/Users/julio/.conan/data/libbacktrace/cci.20210118/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include")
set(libbacktrace_RES_DIRS_RELEASE )
set(libbacktrace_DEFINITIONS_RELEASE )
set(libbacktrace_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(libbacktrace_COMPILE_DEFINITIONS_RELEASE )
set(libbacktrace_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(libbacktrace_COMPILE_OPTIONS_C_RELEASE "")
set(libbacktrace_COMPILE_OPTIONS_CXX_RELEASE "")
set(libbacktrace_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(libbacktrace_LIBRARIES_RELEASE "") # Will be filled later
set(libbacktrace_LIBS_RELEASE "") # Same as libbacktrace_LIBRARIES
set(libbacktrace_SYSTEM_LIBS_RELEASE )
set(libbacktrace_FRAMEWORK_DIRS_RELEASE )
set(libbacktrace_FRAMEWORKS_RELEASE )
set(libbacktrace_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(libbacktrace_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(libbacktrace_FRAMEWORKS_FOUND_RELEASE "${libbacktrace_FRAMEWORKS_RELEASE}" "${libbacktrace_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(libbacktrace_INCLUDE_DIRS_RELEASE
                 libbacktrace_INCLUDE_DIR_RELEASE
                 libbacktrace_INCLUDES_RELEASE
                 libbacktrace_DEFINITIONS_RELEASE
                 libbacktrace_LINKER_FLAGS_RELEASE_LIST
                 libbacktrace_COMPILE_DEFINITIONS_RELEASE
                 libbacktrace_COMPILE_OPTIONS_RELEASE_LIST
                 libbacktrace_LIBRARIES_RELEASE
                 libbacktrace_LIBS_RELEASE
                 libbacktrace_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to libbacktrace_LIBS and libbacktrace_LIBRARY_LIST
set(libbacktrace_LIBRARY_LIST_RELEASE backtrace)
set(libbacktrace_LIB_DIRS_RELEASE "/Users/julio/.conan/data/libbacktrace/cci.20210118/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_libbacktrace_DEPENDENCIES_RELEASE "${libbacktrace_FRAMEWORKS_FOUND_RELEASE} ${libbacktrace_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${libbacktrace_LIBRARY_LIST_RELEASE}"  # libraries
                              "${libbacktrace_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_libbacktrace_DEPENDENCIES_RELEASE}"  # deps
                              libbacktrace_LIBRARIES_RELEASE            # out_libraries
                              libbacktrace_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "libbacktrace")                                      # package_name

set(libbacktrace_LIBS_RELEASE ${libbacktrace_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${libbacktrace_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND libbacktrace_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND libbacktrace_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${libbacktrace_SYSTEM_LIBS_RELEASE})
    list(APPEND libbacktrace_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND libbacktrace_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(libbacktrace_LIBRARIES_TARGETS_RELEASE "${libbacktrace_LIBRARIES_TARGETS_RELEASE};")
set(libbacktrace_LIBRARIES_RELEASE "${libbacktrace_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/libbacktrace/cci.20210118/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/libbacktrace/cci.20210118/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/" ${CMAKE_PREFIX_PATH})
