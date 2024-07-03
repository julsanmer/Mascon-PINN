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


set(protobuf_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_INCLUDES_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_RES_DIRS_RELEASE )
set(protobuf_DEFINITIONS_RELEASE )
set(protobuf_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(protobuf_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(protobuf_COMPILE_OPTIONS_C_RELEASE "")
set(protobuf_COMPILE_OPTIONS_CXX_RELEASE "")
set(protobuf_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(protobuf_LIBRARIES_RELEASE "") # Will be filled later
set(protobuf_LIBS_RELEASE "") # Same as protobuf_LIBRARIES
set(protobuf_SYSTEM_LIBS_RELEASE )
set(protobuf_FRAMEWORK_DIRS_RELEASE )
set(protobuf_FRAMEWORKS_RELEASE )
set(protobuf_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(protobuf_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf/protobuf-generate.cmake"
			"/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf/protobuf-module.cmake"
			"/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf/protobuf-options.cmake")

conan_find_apple_frameworks(protobuf_FRAMEWORKS_FOUND_RELEASE "${protobuf_FRAMEWORKS_RELEASE}" "${protobuf_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(protobuf_INCLUDE_DIRS_RELEASE
                 protobuf_INCLUDE_DIR_RELEASE
                 protobuf_INCLUDES_RELEASE
                 protobuf_DEFINITIONS_RELEASE
                 protobuf_LINKER_FLAGS_RELEASE_LIST
                 protobuf_COMPILE_DEFINITIONS_RELEASE
                 protobuf_COMPILE_OPTIONS_RELEASE_LIST
                 protobuf_LIBRARIES_RELEASE
                 protobuf_LIBS_RELEASE
                 protobuf_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to protobuf_LIBS and protobuf_LIBRARY_LIST
set(protobuf_LIBRARY_LIST_RELEASE protoc protobuf)
set(protobuf_LIB_DIRS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_protobuf_DEPENDENCIES_RELEASE "${protobuf_FRAMEWORKS_FOUND_RELEASE} ${protobuf_SYSTEM_LIBS_RELEASE} ZLIB::ZLIB")

conan_package_library_targets("${protobuf_LIBRARY_LIST_RELEASE}"  # libraries
                              "${protobuf_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_protobuf_DEPENDENCIES_RELEASE}"  # deps
                              protobuf_LIBRARIES_RELEASE            # out_libraries
                              protobuf_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "protobuf")                                      # package_name

set(protobuf_LIBS_RELEASE ${protobuf_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${protobuf_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND protobuf_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND protobuf_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${protobuf_SYSTEM_LIBS_RELEASE})
    list(APPEND protobuf_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND protobuf_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(protobuf_LIBRARIES_TARGETS_RELEASE "${protobuf_LIBRARIES_TARGETS_RELEASE};ZLIB::ZLIB")
set(protobuf_LIBRARIES_RELEASE "${protobuf_LIBRARIES_RELEASE};ZLIB::ZLIB")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf" ${CMAKE_PREFIX_PATH})

set(protobuf_COMPONENTS_RELEASE protobuf::libprotoc protobuf::libprotobuf)

########### COMPONENT libprotobuf VARIABLES #############################################

set(protobuf_libprotobuf_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_libprotobuf_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_libprotobuf_INCLUDES_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_libprotobuf_LIB_DIRS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib")
set(protobuf_libprotobuf_RES_DIRS_RELEASE )
set(protobuf_libprotobuf_DEFINITIONS_RELEASE )
set(protobuf_libprotobuf_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_libprotobuf_COMPILE_OPTIONS_C_RELEASE "")
set(protobuf_libprotobuf_COMPILE_OPTIONS_CXX_RELEASE "")
set(protobuf_libprotobuf_LIBS_RELEASE protobuf)
set(protobuf_libprotobuf_SYSTEM_LIBS_RELEASE )
set(protobuf_libprotobuf_FRAMEWORK_DIRS_RELEASE )
set(protobuf_libprotobuf_FRAMEWORKS_RELEASE )
set(protobuf_libprotobuf_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf/protobuf-generate.cmake"
			"/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf/protobuf-module.cmake"
			"/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib/cmake/protobuf/protobuf-options.cmake")
set(protobuf_libprotobuf_DEPENDENCIES_RELEASE ZLIB::ZLIB)
set(protobuf_libprotobuf_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT libprotobuf FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(protobuf_libprotobuf_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(protobuf_libprotobuf_FRAMEWORKS_FOUND_RELEASE "${protobuf_libprotobuf_FRAMEWORKS_RELEASE}" "${protobuf_libprotobuf_FRAMEWORK_DIRS_RELEASE}")

set(protobuf_libprotobuf_LIB_TARGETS_RELEASE "")
set(protobuf_libprotobuf_NOT_USED_RELEASE "")
set(protobuf_libprotobuf_LIBS_FRAMEWORKS_DEPS_RELEASE ${protobuf_libprotobuf_FRAMEWORKS_FOUND_RELEASE} ${protobuf_libprotobuf_SYSTEM_LIBS_RELEASE} ${protobuf_libprotobuf_DEPENDENCIES_RELEASE})
conan_package_library_targets("${protobuf_libprotobuf_LIBS_RELEASE}"
                              "${protobuf_libprotobuf_LIB_DIRS_RELEASE}"
                              "${protobuf_libprotobuf_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              protobuf_libprotobuf_NOT_USED_RELEASE
                              protobuf_libprotobuf_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "protobuf_libprotobuf")

set(protobuf_libprotobuf_LINK_LIBS_RELEASE ${protobuf_libprotobuf_LIB_TARGETS_RELEASE} ${protobuf_libprotobuf_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT libprotoc VARIABLES #############################################

set(protobuf_libprotoc_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_libprotoc_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_libprotoc_INCLUDES_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include")
set(protobuf_libprotoc_LIB_DIRS_RELEASE "/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/lib")
set(protobuf_libprotoc_RES_DIRS_RELEASE )
set(protobuf_libprotoc_DEFINITIONS_RELEASE )
set(protobuf_libprotoc_COMPILE_DEFINITIONS_RELEASE )
set(protobuf_libprotoc_COMPILE_OPTIONS_C_RELEASE "")
set(protobuf_libprotoc_COMPILE_OPTIONS_CXX_RELEASE "")
set(protobuf_libprotoc_LIBS_RELEASE protoc)
set(protobuf_libprotoc_SYSTEM_LIBS_RELEASE )
set(protobuf_libprotoc_FRAMEWORK_DIRS_RELEASE )
set(protobuf_libprotoc_FRAMEWORKS_RELEASE )
set(protobuf_libprotoc_BUILD_MODULES_PATHS_RELEASE )
set(protobuf_libprotoc_DEPENDENCIES_RELEASE protobuf::libprotobuf)
set(protobuf_libprotoc_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT libprotoc FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(protobuf_libprotoc_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(protobuf_libprotoc_FRAMEWORKS_FOUND_RELEASE "${protobuf_libprotoc_FRAMEWORKS_RELEASE}" "${protobuf_libprotoc_FRAMEWORK_DIRS_RELEASE}")

set(protobuf_libprotoc_LIB_TARGETS_RELEASE "")
set(protobuf_libprotoc_NOT_USED_RELEASE "")
set(protobuf_libprotoc_LIBS_FRAMEWORKS_DEPS_RELEASE ${protobuf_libprotoc_FRAMEWORKS_FOUND_RELEASE} ${protobuf_libprotoc_SYSTEM_LIBS_RELEASE} ${protobuf_libprotoc_DEPENDENCIES_RELEASE})
conan_package_library_targets("${protobuf_libprotoc_LIBS_RELEASE}"
                              "${protobuf_libprotoc_LIB_DIRS_RELEASE}"
                              "${protobuf_libprotoc_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              protobuf_libprotoc_NOT_USED_RELEASE
                              protobuf_libprotoc_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "protobuf_libprotoc")

set(protobuf_libprotoc_LINK_LIBS_RELEASE ${protobuf_libprotoc_LIB_TARGETS_RELEASE} ${protobuf_libprotoc_LIBS_FRAMEWORKS_DEPS_RELEASE})