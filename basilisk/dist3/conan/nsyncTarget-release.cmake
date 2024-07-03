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


set(nsync_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_INCLUDES_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_RES_DIRS_RELEASE )
set(nsync_DEFINITIONS_RELEASE )
set(nsync_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(nsync_COMPILE_DEFINITIONS_RELEASE )
set(nsync_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(nsync_COMPILE_OPTIONS_C_RELEASE "")
set(nsync_COMPILE_OPTIONS_CXX_RELEASE "")
set(nsync_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(nsync_LIBRARIES_RELEASE "") # Will be filled later
set(nsync_LIBS_RELEASE "") # Same as nsync_LIBRARIES
set(nsync_SYSTEM_LIBS_RELEASE )
set(nsync_FRAMEWORK_DIRS_RELEASE )
set(nsync_FRAMEWORKS_RELEASE )
set(nsync_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(nsync_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(nsync_FRAMEWORKS_FOUND_RELEASE "${nsync_FRAMEWORKS_RELEASE}" "${nsync_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(nsync_INCLUDE_DIRS_RELEASE
                 nsync_INCLUDE_DIR_RELEASE
                 nsync_INCLUDES_RELEASE
                 nsync_DEFINITIONS_RELEASE
                 nsync_LINKER_FLAGS_RELEASE_LIST
                 nsync_COMPILE_DEFINITIONS_RELEASE
                 nsync_COMPILE_OPTIONS_RELEASE_LIST
                 nsync_LIBRARIES_RELEASE
                 nsync_LIBS_RELEASE
                 nsync_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to nsync_LIBS and nsync_LIBRARY_LIST
set(nsync_LIBRARY_LIST_RELEASE nsync nsync_cpp)
set(nsync_LIB_DIRS_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_nsync_DEPENDENCIES_RELEASE "${nsync_FRAMEWORKS_FOUND_RELEASE} ${nsync_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${nsync_LIBRARY_LIST_RELEASE}"  # libraries
                              "${nsync_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_nsync_DEPENDENCIES_RELEASE}"  # deps
                              nsync_LIBRARIES_RELEASE            # out_libraries
                              nsync_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "nsync")                                      # package_name

set(nsync_LIBS_RELEASE ${nsync_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${nsync_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND nsync_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND nsync_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${nsync_SYSTEM_LIBS_RELEASE})
    list(APPEND nsync_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND nsync_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(nsync_LIBRARIES_TARGETS_RELEASE "${nsync_LIBRARIES_TARGETS_RELEASE};")
set(nsync_LIBRARIES_RELEASE "${nsync_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(nsync_COMPONENTS_RELEASE nsync::nsync_c nsync::nsync_cpp)

########### COMPONENT nsync_cpp VARIABLES #############################################

set(nsync_nsync_cpp_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_nsync_cpp_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_nsync_cpp_INCLUDES_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_nsync_cpp_LIB_DIRS_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(nsync_nsync_cpp_RES_DIRS_RELEASE )
set(nsync_nsync_cpp_DEFINITIONS_RELEASE )
set(nsync_nsync_cpp_COMPILE_DEFINITIONS_RELEASE )
set(nsync_nsync_cpp_COMPILE_OPTIONS_C_RELEASE "")
set(nsync_nsync_cpp_COMPILE_OPTIONS_CXX_RELEASE "")
set(nsync_nsync_cpp_LIBS_RELEASE nsync_cpp)
set(nsync_nsync_cpp_SYSTEM_LIBS_RELEASE )
set(nsync_nsync_cpp_FRAMEWORK_DIRS_RELEASE )
set(nsync_nsync_cpp_FRAMEWORKS_RELEASE )
set(nsync_nsync_cpp_BUILD_MODULES_PATHS_RELEASE )
set(nsync_nsync_cpp_DEPENDENCIES_RELEASE )
set(nsync_nsync_cpp_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT nsync_cpp FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(nsync_nsync_cpp_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(nsync_nsync_cpp_FRAMEWORKS_FOUND_RELEASE "${nsync_nsync_cpp_FRAMEWORKS_RELEASE}" "${nsync_nsync_cpp_FRAMEWORK_DIRS_RELEASE}")

set(nsync_nsync_cpp_LIB_TARGETS_RELEASE "")
set(nsync_nsync_cpp_NOT_USED_RELEASE "")
set(nsync_nsync_cpp_LIBS_FRAMEWORKS_DEPS_RELEASE ${nsync_nsync_cpp_FRAMEWORKS_FOUND_RELEASE} ${nsync_nsync_cpp_SYSTEM_LIBS_RELEASE} ${nsync_nsync_cpp_DEPENDENCIES_RELEASE})
conan_package_library_targets("${nsync_nsync_cpp_LIBS_RELEASE}"
                              "${nsync_nsync_cpp_LIB_DIRS_RELEASE}"
                              "${nsync_nsync_cpp_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              nsync_nsync_cpp_NOT_USED_RELEASE
                              nsync_nsync_cpp_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "nsync_nsync_cpp")

set(nsync_nsync_cpp_LINK_LIBS_RELEASE ${nsync_nsync_cpp_LIB_TARGETS_RELEASE} ${nsync_nsync_cpp_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT nsync_c VARIABLES #############################################

set(nsync_nsync_c_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_nsync_c_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_nsync_c_INCLUDES_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(nsync_nsync_c_LIB_DIRS_RELEASE "/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(nsync_nsync_c_RES_DIRS_RELEASE )
set(nsync_nsync_c_DEFINITIONS_RELEASE )
set(nsync_nsync_c_COMPILE_DEFINITIONS_RELEASE )
set(nsync_nsync_c_COMPILE_OPTIONS_C_RELEASE "")
set(nsync_nsync_c_COMPILE_OPTIONS_CXX_RELEASE "")
set(nsync_nsync_c_LIBS_RELEASE nsync)
set(nsync_nsync_c_SYSTEM_LIBS_RELEASE )
set(nsync_nsync_c_FRAMEWORK_DIRS_RELEASE )
set(nsync_nsync_c_FRAMEWORKS_RELEASE )
set(nsync_nsync_c_BUILD_MODULES_PATHS_RELEASE )
set(nsync_nsync_c_DEPENDENCIES_RELEASE )
set(nsync_nsync_c_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT nsync_c FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(nsync_nsync_c_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(nsync_nsync_c_FRAMEWORKS_FOUND_RELEASE "${nsync_nsync_c_FRAMEWORKS_RELEASE}" "${nsync_nsync_c_FRAMEWORK_DIRS_RELEASE}")

set(nsync_nsync_c_LIB_TARGETS_RELEASE "")
set(nsync_nsync_c_NOT_USED_RELEASE "")
set(nsync_nsync_c_LIBS_FRAMEWORKS_DEPS_RELEASE ${nsync_nsync_c_FRAMEWORKS_FOUND_RELEASE} ${nsync_nsync_c_SYSTEM_LIBS_RELEASE} ${nsync_nsync_c_DEPENDENCIES_RELEASE})
conan_package_library_targets("${nsync_nsync_c_LIBS_RELEASE}"
                              "${nsync_nsync_c_LIB_DIRS_RELEASE}"
                              "${nsync_nsync_c_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              nsync_nsync_c_NOT_USED_RELEASE
                              nsync_nsync_c_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "nsync_nsync_c")

set(nsync_nsync_c_LINK_LIBS_RELEASE ${nsync_nsync_c_LIB_TARGETS_RELEASE} ${nsync_nsync_c_LIBS_FRAMEWORKS_DEPS_RELEASE})