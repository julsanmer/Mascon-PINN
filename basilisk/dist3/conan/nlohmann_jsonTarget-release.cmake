
set(nlohmann_json_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(nlohmann_json_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(nlohmann_json_INCLUDES_RELEASE "/Users/julio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(nlohmann_json_RES_DIRS_RELEASE )
set(nlohmann_json_DEFINITIONS_RELEASE )
set(nlohmann_json_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(nlohmann_json_COMPILE_DEFINITIONS_RELEASE )
set(nlohmann_json_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(nlohmann_json_COMPILE_OPTIONS_C_RELEASE "")
set(nlohmann_json_COMPILE_OPTIONS_CXX_RELEASE "")
set(nlohmann_json_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(nlohmann_json_LIBRARIES_RELEASE "") # Will be filled later
set(nlohmann_json_LIBS_RELEASE "") # Same as nlohmann_json_LIBRARIES
set(nlohmann_json_SYSTEM_LIBS_RELEASE )
set(nlohmann_json_FRAMEWORK_DIRS_RELEASE )
set(nlohmann_json_FRAMEWORKS_RELEASE )
set(nlohmann_json_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(nlohmann_json_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(nlohmann_json_FRAMEWORKS_FOUND_RELEASE "${nlohmann_json_FRAMEWORKS_RELEASE}" "${nlohmann_json_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(nlohmann_json_INCLUDE_DIRS_RELEASE
                 nlohmann_json_INCLUDE_DIR_RELEASE
                 nlohmann_json_INCLUDES_RELEASE
                 nlohmann_json_DEFINITIONS_RELEASE
                 nlohmann_json_LINKER_FLAGS_RELEASE_LIST
                 nlohmann_json_COMPILE_DEFINITIONS_RELEASE
                 nlohmann_json_COMPILE_OPTIONS_RELEASE_LIST
                 nlohmann_json_LIBRARIES_RELEASE
                 nlohmann_json_LIBS_RELEASE
                 nlohmann_json_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to nlohmann_json_LIBS and nlohmann_json_LIBRARY_LIST
set(nlohmann_json_LIBRARY_LIST_RELEASE )
set(nlohmann_json_LIB_DIRS_RELEASE )

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_nlohmann_json_DEPENDENCIES_RELEASE "${nlohmann_json_FRAMEWORKS_FOUND_RELEASE} ${nlohmann_json_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${nlohmann_json_LIBRARY_LIST_RELEASE}"  # libraries
                              "${nlohmann_json_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_nlohmann_json_DEPENDENCIES_RELEASE}"  # deps
                              nlohmann_json_LIBRARIES_RELEASE            # out_libraries
                              nlohmann_json_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "nlohmann_json")                                      # package_name

set(nlohmann_json_LIBS_RELEASE ${nlohmann_json_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${nlohmann_json_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND nlohmann_json_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND nlohmann_json_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${nlohmann_json_SYSTEM_LIBS_RELEASE})
    list(APPEND nlohmann_json_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND nlohmann_json_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(nlohmann_json_LIBRARIES_TARGETS_RELEASE "${nlohmann_json_LIBRARIES_TARGETS_RELEASE};")
set(nlohmann_json_LIBRARIES_RELEASE "${nlohmann_json_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/" ${CMAKE_PREFIX_PATH})
