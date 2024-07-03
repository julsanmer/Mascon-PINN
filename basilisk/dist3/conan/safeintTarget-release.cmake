
set(safeint_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/safeint/3.0.28/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(safeint_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/safeint/3.0.28/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(safeint_INCLUDES_RELEASE "/Users/julio/.conan/data/safeint/3.0.28/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(safeint_RES_DIRS_RELEASE )
set(safeint_DEFINITIONS_RELEASE )
set(safeint_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(safeint_COMPILE_DEFINITIONS_RELEASE )
set(safeint_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(safeint_COMPILE_OPTIONS_C_RELEASE "")
set(safeint_COMPILE_OPTIONS_CXX_RELEASE "")
set(safeint_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(safeint_LIBRARIES_RELEASE "") # Will be filled later
set(safeint_LIBS_RELEASE "") # Same as safeint_LIBRARIES
set(safeint_SYSTEM_LIBS_RELEASE )
set(safeint_FRAMEWORK_DIRS_RELEASE )
set(safeint_FRAMEWORKS_RELEASE )
set(safeint_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(safeint_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(safeint_FRAMEWORKS_FOUND_RELEASE "${safeint_FRAMEWORKS_RELEASE}" "${safeint_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(safeint_INCLUDE_DIRS_RELEASE
                 safeint_INCLUDE_DIR_RELEASE
                 safeint_INCLUDES_RELEASE
                 safeint_DEFINITIONS_RELEASE
                 safeint_LINKER_FLAGS_RELEASE_LIST
                 safeint_COMPILE_DEFINITIONS_RELEASE
                 safeint_COMPILE_OPTIONS_RELEASE_LIST
                 safeint_LIBRARIES_RELEASE
                 safeint_LIBS_RELEASE
                 safeint_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to safeint_LIBS and safeint_LIBRARY_LIST
set(safeint_LIBRARY_LIST_RELEASE )
set(safeint_LIB_DIRS_RELEASE )

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_safeint_DEPENDENCIES_RELEASE "${safeint_FRAMEWORKS_FOUND_RELEASE} ${safeint_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${safeint_LIBRARY_LIST_RELEASE}"  # libraries
                              "${safeint_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_safeint_DEPENDENCIES_RELEASE}"  # deps
                              safeint_LIBRARIES_RELEASE            # out_libraries
                              safeint_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "safeint")                                      # package_name

set(safeint_LIBS_RELEASE ${safeint_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${safeint_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND safeint_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND safeint_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${safeint_SYSTEM_LIBS_RELEASE})
    list(APPEND safeint_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND safeint_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(safeint_LIBRARIES_TARGETS_RELEASE "${safeint_LIBRARIES_TARGETS_RELEASE};")
set(safeint_LIBRARIES_RELEASE "${safeint_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/safeint/3.0.28/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/safeint/3.0.28/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/" ${CMAKE_PREFIX_PATH})
