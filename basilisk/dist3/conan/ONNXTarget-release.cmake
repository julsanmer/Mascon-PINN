########## MACROS ###########################################################################
#############################################################################################


macro(conan_find_apple_frameworks FRAMEWORKS_FOUND FRAMEWORKS FRAMEWORKS_DIRS)
    if(APPLE)
        foreach(_FRAMEWORK ${FRAMEWORKS})
            # https://cmake.org/pipermail/cmake-developers/2017-August/030199.html
            find_library(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND NAMES ${_FRAMEWORK} PATHS ${FRAMEWORKS_DIRS} CMAKE_FIND_ROOT_PATH_BOTH)
            if(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND)
                list(APPEND ${FRAMEWORKS_FOUND} ${CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND})
            else()
                message(FATAL_ERROR "Framework library ${_FRAMEWORK} not found in paths: ${FRAMEWORKS_DIRS}")
            endif()
        endforeach()
    endif()
endmacro()


function(conan_package_library_targets libraries package_libdir deps out_libraries out_libraries_target build_type package_name)
    unset(_CONAN_ACTUAL_TARGETS CACHE)
    unset(_CONAN_FOUND_SYSTEM_LIBS CACHE)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAMES ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            list(APPEND _out_libraries ${CONAN_FOUND_LIBRARY})
            if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
                # Create a micro-target for each lib/a found
                string(REGEX REPLACE "[^A-Za-z0-9.+_-]" "_" _LIBRARY_NAME ${_LIBRARY_NAME})
                set(_LIB_NAME CONAN_LIB::${package_name}_${_LIBRARY_NAME}${build_type})
                if(NOT TARGET ${_LIB_NAME})
                    # Create a micro-target for each lib/a found
                    add_library(${_LIB_NAME} UNKNOWN IMPORTED)
                    set_target_properties(${_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${CONAN_FOUND_LIBRARY})
                    set(_CONAN_ACTUAL_TARGETS ${_CONAN_ACTUAL_TARGETS} ${_LIB_NAME})
                else()
                    conan_message(STATUS "Skipping already existing target: ${_LIB_NAME}")
                endif()
                list(APPEND _out_libraries_target ${_LIB_NAME})
            endif()
            conan_message(STATUS "Found: ${CONAN_FOUND_LIBRARY}")
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            list(APPEND _out_libraries_target ${_LIBRARY_NAME})
            list(APPEND _out_libraries ${_LIBRARY_NAME})
            set(_CONAN_FOUND_SYSTEM_LIBS "${_CONAN_FOUND_SYSTEM_LIBS};${_LIBRARY_NAME}")
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()

    if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
        # Add all dependencies to all targets
        string(REPLACE " " ";" deps_list "${deps}")
        foreach(_CONAN_ACTUAL_TARGET ${_CONAN_ACTUAL_TARGETS})
            set_property(TARGET ${_CONAN_ACTUAL_TARGET} PROPERTY INTERFACE_LINK_LIBRARIES "${_CONAN_FOUND_SYSTEM_LIBS};${deps_list}")
        endforeach()
    endif()

    set(${out_libraries} ${_out_libraries} PARENT_SCOPE)
    set(${out_libraries_target} ${_out_libraries_target} PARENT_SCOPE)
endfunction()


########### VARIABLES #######################################################################
#############################################################################################


set(ONNX_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_INCLUDES_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_RES_DIRS_RELEASE )
set(ONNX_DEFINITIONS_RELEASE "-DONNX_NAMESPACE=onnx"
			"-DONNX_ML=1"
			"-D__STDC_FORMAT_MACROS")
set(ONNX_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(ONNX_COMPILE_DEFINITIONS_RELEASE "ONNX_NAMESPACE=onnx"
			"ONNX_ML=1"
			"__STDC_FORMAT_MACROS")
set(ONNX_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(ONNX_COMPILE_OPTIONS_C_RELEASE "")
set(ONNX_COMPILE_OPTIONS_CXX_RELEASE "")
set(ONNX_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(ONNX_LIBRARIES_RELEASE "") # Will be filled later
set(ONNX_LIBS_RELEASE "") # Same as ONNX_LIBRARIES
set(ONNX_SYSTEM_LIBS_RELEASE )
set(ONNX_FRAMEWORK_DIRS_RELEASE )
set(ONNX_FRAMEWORKS_RELEASE )
set(ONNX_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(ONNX_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/lib/cmake/conan-official-onnx-targets.cmake")

conan_find_apple_frameworks(ONNX_FRAMEWORKS_FOUND_RELEASE "${ONNX_FRAMEWORKS_RELEASE}" "${ONNX_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(ONNX_INCLUDE_DIRS_RELEASE
                 ONNX_INCLUDE_DIR_RELEASE
                 ONNX_INCLUDES_RELEASE
                 ONNX_DEFINITIONS_RELEASE
                 ONNX_LINKER_FLAGS_RELEASE_LIST
                 ONNX_COMPILE_DEFINITIONS_RELEASE
                 ONNX_COMPILE_OPTIONS_RELEASE_LIST
                 ONNX_LIBRARIES_RELEASE
                 ONNX_LIBS_RELEASE
                 ONNX_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to ONNX_LIBS and ONNX_LIBRARY_LIST
set(ONNX_LIBRARY_LIST_RELEASE onnx onnx_proto)
set(ONNX_LIB_DIRS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_ONNX_DEPENDENCIES_RELEASE "${ONNX_FRAMEWORKS_FOUND_RELEASE} ${ONNX_SYSTEM_LIBS_RELEASE} protobuf::libprotobuf")

conan_package_library_targets("${ONNX_LIBRARY_LIST_RELEASE}"  # libraries
                              "${ONNX_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_ONNX_DEPENDENCIES_RELEASE}"  # deps
                              ONNX_LIBRARIES_RELEASE            # out_libraries
                              ONNX_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "ONNX")                                      # package_name

set(ONNX_LIBS_RELEASE ${ONNX_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${ONNX_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND ONNX_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND ONNX_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${ONNX_SYSTEM_LIBS_RELEASE})
    list(APPEND ONNX_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND ONNX_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(ONNX_LIBRARIES_TARGETS_RELEASE "${ONNX_LIBRARIES_TARGETS_RELEASE};protobuf::libprotobuf")
set(ONNX_LIBRARIES_RELEASE "${ONNX_LIBRARIES_RELEASE};protobuf::libprotobuf")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(ONNX_COMPONENTS_RELEASE ONNX::onnx ONNX::onnx_proto)

########### COMPONENT onnx_proto VARIABLES #############################################

set(ONNX_onnx_proto_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_onnx_proto_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_onnx_proto_INCLUDES_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_onnx_proto_LIB_DIRS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/lib")
set(ONNX_onnx_proto_RES_DIRS_RELEASE )
set(ONNX_onnx_proto_DEFINITIONS_RELEASE "-DONNX_NAMESPACE=onnx"
			"-DONNX_ML=1")
set(ONNX_onnx_proto_COMPILE_DEFINITIONS_RELEASE "ONNX_NAMESPACE=onnx"
			"ONNX_ML=1")
set(ONNX_onnx_proto_COMPILE_OPTIONS_C_RELEASE "")
set(ONNX_onnx_proto_COMPILE_OPTIONS_CXX_RELEASE "")
set(ONNX_onnx_proto_LIBS_RELEASE onnx_proto)
set(ONNX_onnx_proto_SYSTEM_LIBS_RELEASE )
set(ONNX_onnx_proto_FRAMEWORK_DIRS_RELEASE )
set(ONNX_onnx_proto_FRAMEWORKS_RELEASE )
set(ONNX_onnx_proto_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/lib/cmake/conan-official-onnx-targets.cmake")
set(ONNX_onnx_proto_DEPENDENCIES_RELEASE protobuf::libprotobuf)
set(ONNX_onnx_proto_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT onnx_proto FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(ONNX_onnx_proto_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(ONNX_onnx_proto_FRAMEWORKS_FOUND_RELEASE "${ONNX_onnx_proto_FRAMEWORKS_RELEASE}" "${ONNX_onnx_proto_FRAMEWORK_DIRS_RELEASE}")

set(ONNX_onnx_proto_LIB_TARGETS_RELEASE "")
set(ONNX_onnx_proto_NOT_USED_RELEASE "")
set(ONNX_onnx_proto_LIBS_FRAMEWORKS_DEPS_RELEASE ${ONNX_onnx_proto_FRAMEWORKS_FOUND_RELEASE} ${ONNX_onnx_proto_SYSTEM_LIBS_RELEASE} ${ONNX_onnx_proto_DEPENDENCIES_RELEASE})
conan_package_library_targets("${ONNX_onnx_proto_LIBS_RELEASE}"
                              "${ONNX_onnx_proto_LIB_DIRS_RELEASE}"
                              "${ONNX_onnx_proto_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              ONNX_onnx_proto_NOT_USED_RELEASE
                              ONNX_onnx_proto_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "ONNX_onnx_proto")

set(ONNX_onnx_proto_LINK_LIBS_RELEASE ${ONNX_onnx_proto_LIB_TARGETS_RELEASE} ${ONNX_onnx_proto_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT onnx VARIABLES #############################################

set(ONNX_onnx_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_onnx_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_onnx_INCLUDES_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include")
set(ONNX_onnx_LIB_DIRS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/lib")
set(ONNX_onnx_RES_DIRS_RELEASE )
set(ONNX_onnx_DEFINITIONS_RELEASE "-DONNX_NAMESPACE=onnx"
			"-DONNX_ML=1"
			"-D__STDC_FORMAT_MACROS")
set(ONNX_onnx_COMPILE_DEFINITIONS_RELEASE "ONNX_NAMESPACE=onnx"
			"ONNX_ML=1"
			"__STDC_FORMAT_MACROS")
set(ONNX_onnx_COMPILE_OPTIONS_C_RELEASE "")
set(ONNX_onnx_COMPILE_OPTIONS_CXX_RELEASE "")
set(ONNX_onnx_LIBS_RELEASE onnx)
set(ONNX_onnx_SYSTEM_LIBS_RELEASE )
set(ONNX_onnx_FRAMEWORK_DIRS_RELEASE )
set(ONNX_onnx_FRAMEWORKS_RELEASE )
set(ONNX_onnx_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/lib/cmake/conan-official-onnx-targets.cmake")
set(ONNX_onnx_DEPENDENCIES_RELEASE ONNX::onnx_proto)
set(ONNX_onnx_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT onnx FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(ONNX_onnx_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(ONNX_onnx_FRAMEWORKS_FOUND_RELEASE "${ONNX_onnx_FRAMEWORKS_RELEASE}" "${ONNX_onnx_FRAMEWORK_DIRS_RELEASE}")

set(ONNX_onnx_LIB_TARGETS_RELEASE "")
set(ONNX_onnx_NOT_USED_RELEASE "")
set(ONNX_onnx_LIBS_FRAMEWORKS_DEPS_RELEASE ${ONNX_onnx_FRAMEWORKS_FOUND_RELEASE} ${ONNX_onnx_SYSTEM_LIBS_RELEASE} ${ONNX_onnx_DEPENDENCIES_RELEASE})
conan_package_library_targets("${ONNX_onnx_LIBS_RELEASE}"
                              "${ONNX_onnx_LIB_DIRS_RELEASE}"
                              "${ONNX_onnx_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              ONNX_onnx_NOT_USED_RELEASE
                              ONNX_onnx_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "ONNX_onnx")

set(ONNX_onnx_LINK_LIBS_RELEASE ${ONNX_onnx_LIB_TARGETS_RELEASE} ${ONNX_onnx_LIBS_FRAMEWORKS_DEPS_RELEASE})