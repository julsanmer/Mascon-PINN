
set(re2_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/include")
set(re2_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/include")
set(re2_INCLUDES_RELEASE "/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/include")
set(re2_RES_DIRS_RELEASE )
set(re2_DEFINITIONS_RELEASE )
set(re2_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(re2_COMPILE_DEFINITIONS_RELEASE )
set(re2_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(re2_COMPILE_OPTIONS_C_RELEASE "")
set(re2_COMPILE_OPTIONS_CXX_RELEASE "")
set(re2_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(re2_LIBRARIES_RELEASE "") # Will be filled later
set(re2_LIBS_RELEASE "") # Same as re2_LIBRARIES
set(re2_SYSTEM_LIBS_RELEASE )
set(re2_FRAMEWORK_DIRS_RELEASE )
set(re2_FRAMEWORKS_RELEASE )
set(re2_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(re2_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(re2_FRAMEWORKS_FOUND_RELEASE "${re2_FRAMEWORKS_RELEASE}" "${re2_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(re2_INCLUDE_DIRS_RELEASE
                 re2_INCLUDE_DIR_RELEASE
                 re2_INCLUDES_RELEASE
                 re2_DEFINITIONS_RELEASE
                 re2_LINKER_FLAGS_RELEASE_LIST
                 re2_COMPILE_DEFINITIONS_RELEASE
                 re2_COMPILE_OPTIONS_RELEASE_LIST
                 re2_LIBRARIES_RELEASE
                 re2_LIBS_RELEASE
                 re2_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to re2_LIBS and re2_LIBRARY_LIST
set(re2_LIBRARY_LIST_RELEASE re2)
set(re2_LIB_DIRS_RELEASE "/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_re2_DEPENDENCIES_RELEASE "${re2_FRAMEWORKS_FOUND_RELEASE} ${re2_SYSTEM_LIBS_RELEASE} absl::absl")

conan_package_library_targets("${re2_LIBRARY_LIST_RELEASE}"  # libraries
                              "${re2_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_re2_DEPENDENCIES_RELEASE}"  # deps
                              re2_LIBRARIES_RELEASE            # out_libraries
                              re2_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "re2")                                      # package_name

set(re2_LIBS_RELEASE ${re2_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${re2_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND re2_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND re2_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${re2_SYSTEM_LIBS_RELEASE})
    list(APPEND re2_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND re2_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(re2_LIBRARIES_TARGETS_RELEASE "${re2_LIBRARIES_TARGETS_RELEASE};absl::absl")
set(re2_LIBRARIES_RELEASE "${re2_LIBRARIES_RELEASE};absl::absl")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/" ${CMAKE_PREFIX_PATH})
