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


set(flatbuffers_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(flatbuffers_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(flatbuffers_INCLUDES_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(flatbuffers_RES_DIRS_RELEASE )
set(flatbuffers_DEFINITIONS_RELEASE )
set(flatbuffers_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(flatbuffers_COMPILE_DEFINITIONS_RELEASE )
set(flatbuffers_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(flatbuffers_COMPILE_OPTIONS_C_RELEASE "")
set(flatbuffers_COMPILE_OPTIONS_CXX_RELEASE "")
set(flatbuffers_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(flatbuffers_LIBRARIES_RELEASE "") # Will be filled later
set(flatbuffers_LIBS_RELEASE "") # Same as flatbuffers_LIBRARIES
set(flatbuffers_SYSTEM_LIBS_RELEASE )
set(flatbuffers_FRAMEWORK_DIRS_RELEASE )
set(flatbuffers_FRAMEWORKS_RELEASE )
set(flatbuffers_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(flatbuffers_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib/cmake/FlatcTargets.cmake"
			"/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib/cmake/BuildFlatBuffers.cmake")

conan_find_apple_frameworks(flatbuffers_FRAMEWORKS_FOUND_RELEASE "${flatbuffers_FRAMEWORKS_RELEASE}" "${flatbuffers_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(flatbuffers_INCLUDE_DIRS_RELEASE
                 flatbuffers_INCLUDE_DIR_RELEASE
                 flatbuffers_INCLUDES_RELEASE
                 flatbuffers_DEFINITIONS_RELEASE
                 flatbuffers_LINKER_FLAGS_RELEASE_LIST
                 flatbuffers_COMPILE_DEFINITIONS_RELEASE
                 flatbuffers_COMPILE_OPTIONS_RELEASE_LIST
                 flatbuffers_LIBRARIES_RELEASE
                 flatbuffers_LIBS_RELEASE
                 flatbuffers_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to flatbuffers_LIBS and flatbuffers_LIBRARY_LIST
set(flatbuffers_LIBRARY_LIST_RELEASE flatbuffers)
set(flatbuffers_LIB_DIRS_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_flatbuffers_DEPENDENCIES_RELEASE "${flatbuffers_FRAMEWORKS_FOUND_RELEASE} ${flatbuffers_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${flatbuffers_LIBRARY_LIST_RELEASE}"  # libraries
                              "${flatbuffers_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_flatbuffers_DEPENDENCIES_RELEASE}"  # deps
                              flatbuffers_LIBRARIES_RELEASE            # out_libraries
                              flatbuffers_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "flatbuffers")                                      # package_name

set(flatbuffers_LIBS_RELEASE ${flatbuffers_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${flatbuffers_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND flatbuffers_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND flatbuffers_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${flatbuffers_SYSTEM_LIBS_RELEASE})
    list(APPEND flatbuffers_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND flatbuffers_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(flatbuffers_LIBRARIES_TARGETS_RELEASE "${flatbuffers_LIBRARIES_TARGETS_RELEASE};")
set(flatbuffers_LIBRARIES_RELEASE "${flatbuffers_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(flatbuffers_COMPONENTS_RELEASE flatbuffers::flatbuffers)

########### COMPONENT flatbuffers VARIABLES #############################################

set(flatbuffers_flatbuffers_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(flatbuffers_flatbuffers_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(flatbuffers_flatbuffers_INCLUDES_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(flatbuffers_flatbuffers_LIB_DIRS_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(flatbuffers_flatbuffers_RES_DIRS_RELEASE )
set(flatbuffers_flatbuffers_DEFINITIONS_RELEASE )
set(flatbuffers_flatbuffers_COMPILE_DEFINITIONS_RELEASE )
set(flatbuffers_flatbuffers_COMPILE_OPTIONS_C_RELEASE "")
set(flatbuffers_flatbuffers_COMPILE_OPTIONS_CXX_RELEASE "")
set(flatbuffers_flatbuffers_LIBS_RELEASE flatbuffers)
set(flatbuffers_flatbuffers_SYSTEM_LIBS_RELEASE )
set(flatbuffers_flatbuffers_FRAMEWORK_DIRS_RELEASE )
set(flatbuffers_flatbuffers_FRAMEWORKS_RELEASE )
set(flatbuffers_flatbuffers_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib/cmake/FlatcTargets.cmake"
			"/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib/cmake/BuildFlatBuffers.cmake")
set(flatbuffers_flatbuffers_DEPENDENCIES_RELEASE )
set(flatbuffers_flatbuffers_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flatbuffers FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(flatbuffers_flatbuffers_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(flatbuffers_flatbuffers_FRAMEWORKS_FOUND_RELEASE "${flatbuffers_flatbuffers_FRAMEWORKS_RELEASE}" "${flatbuffers_flatbuffers_FRAMEWORK_DIRS_RELEASE}")

set(flatbuffers_flatbuffers_LIB_TARGETS_RELEASE "")
set(flatbuffers_flatbuffers_NOT_USED_RELEASE "")
set(flatbuffers_flatbuffers_LIBS_FRAMEWORKS_DEPS_RELEASE ${flatbuffers_flatbuffers_FRAMEWORKS_FOUND_RELEASE} ${flatbuffers_flatbuffers_SYSTEM_LIBS_RELEASE} ${flatbuffers_flatbuffers_DEPENDENCIES_RELEASE})
conan_package_library_targets("${flatbuffers_flatbuffers_LIBS_RELEASE}"
                              "${flatbuffers_flatbuffers_LIB_DIRS_RELEASE}"
                              "${flatbuffers_flatbuffers_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              flatbuffers_flatbuffers_NOT_USED_RELEASE
                              flatbuffers_flatbuffers_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "flatbuffers_flatbuffers")

set(flatbuffers_flatbuffers_LINK_LIBS_RELEASE ${flatbuffers_flatbuffers_LIB_TARGETS_RELEASE} ${flatbuffers_flatbuffers_LIBS_FRAMEWORKS_DEPS_RELEASE})