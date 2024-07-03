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


set(cpuinfo_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_INCLUDES_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_RES_DIRS_RELEASE )
set(cpuinfo_DEFINITIONS_RELEASE )
set(cpuinfo_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(cpuinfo_COMPILE_DEFINITIONS_RELEASE )
set(cpuinfo_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(cpuinfo_COMPILE_OPTIONS_C_RELEASE "")
set(cpuinfo_COMPILE_OPTIONS_CXX_RELEASE "")
set(cpuinfo_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(cpuinfo_LIBRARIES_RELEASE "") # Will be filled later
set(cpuinfo_LIBS_RELEASE "") # Same as cpuinfo_LIBRARIES
set(cpuinfo_SYSTEM_LIBS_RELEASE )
set(cpuinfo_FRAMEWORK_DIRS_RELEASE )
set(cpuinfo_FRAMEWORKS_RELEASE )
set(cpuinfo_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(cpuinfo_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(cpuinfo_FRAMEWORKS_FOUND_RELEASE "${cpuinfo_FRAMEWORKS_RELEASE}" "${cpuinfo_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(cpuinfo_INCLUDE_DIRS_RELEASE
                 cpuinfo_INCLUDE_DIR_RELEASE
                 cpuinfo_INCLUDES_RELEASE
                 cpuinfo_DEFINITIONS_RELEASE
                 cpuinfo_LINKER_FLAGS_RELEASE_LIST
                 cpuinfo_COMPILE_DEFINITIONS_RELEASE
                 cpuinfo_COMPILE_OPTIONS_RELEASE_LIST
                 cpuinfo_LIBRARIES_RELEASE
                 cpuinfo_LIBS_RELEASE
                 cpuinfo_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to cpuinfo_LIBS and cpuinfo_LIBRARY_LIST
set(cpuinfo_LIBRARY_LIST_RELEASE cpuinfo clog)
set(cpuinfo_LIB_DIRS_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_cpuinfo_DEPENDENCIES_RELEASE "${cpuinfo_FRAMEWORKS_FOUND_RELEASE} ${cpuinfo_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${cpuinfo_LIBRARY_LIST_RELEASE}"  # libraries
                              "${cpuinfo_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_cpuinfo_DEPENDENCIES_RELEASE}"  # deps
                              cpuinfo_LIBRARIES_RELEASE            # out_libraries
                              cpuinfo_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "cpuinfo")                                      # package_name

set(cpuinfo_LIBS_RELEASE ${cpuinfo_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${cpuinfo_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND cpuinfo_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND cpuinfo_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${cpuinfo_SYSTEM_LIBS_RELEASE})
    list(APPEND cpuinfo_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND cpuinfo_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(cpuinfo_LIBRARIES_TARGETS_RELEASE "${cpuinfo_LIBRARIES_TARGETS_RELEASE};")
set(cpuinfo_LIBRARIES_RELEASE "${cpuinfo_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(cpuinfo_COMPONENTS_RELEASE cpuinfo::cpuinfo cpuinfo::clog)

########### COMPONENT clog VARIABLES #############################################

set(cpuinfo_clog_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_clog_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_clog_INCLUDES_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_clog_LIB_DIRS_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/lib")
set(cpuinfo_clog_RES_DIRS_RELEASE )
set(cpuinfo_clog_DEFINITIONS_RELEASE )
set(cpuinfo_clog_COMPILE_DEFINITIONS_RELEASE )
set(cpuinfo_clog_COMPILE_OPTIONS_C_RELEASE "")
set(cpuinfo_clog_COMPILE_OPTIONS_CXX_RELEASE "")
set(cpuinfo_clog_LIBS_RELEASE clog)
set(cpuinfo_clog_SYSTEM_LIBS_RELEASE )
set(cpuinfo_clog_FRAMEWORK_DIRS_RELEASE )
set(cpuinfo_clog_FRAMEWORKS_RELEASE )
set(cpuinfo_clog_BUILD_MODULES_PATHS_RELEASE )
set(cpuinfo_clog_DEPENDENCIES_RELEASE )
set(cpuinfo_clog_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT clog FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(cpuinfo_clog_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(cpuinfo_clog_FRAMEWORKS_FOUND_RELEASE "${cpuinfo_clog_FRAMEWORKS_RELEASE}" "${cpuinfo_clog_FRAMEWORK_DIRS_RELEASE}")

set(cpuinfo_clog_LIB_TARGETS_RELEASE "")
set(cpuinfo_clog_NOT_USED_RELEASE "")
set(cpuinfo_clog_LIBS_FRAMEWORKS_DEPS_RELEASE ${cpuinfo_clog_FRAMEWORKS_FOUND_RELEASE} ${cpuinfo_clog_SYSTEM_LIBS_RELEASE} ${cpuinfo_clog_DEPENDENCIES_RELEASE})
conan_package_library_targets("${cpuinfo_clog_LIBS_RELEASE}"
                              "${cpuinfo_clog_LIB_DIRS_RELEASE}"
                              "${cpuinfo_clog_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              cpuinfo_clog_NOT_USED_RELEASE
                              cpuinfo_clog_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "cpuinfo_clog")

set(cpuinfo_clog_LINK_LIBS_RELEASE ${cpuinfo_clog_LIB_TARGETS_RELEASE} ${cpuinfo_clog_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cpuinfo VARIABLES #############################################

set(cpuinfo_cpuinfo_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_cpuinfo_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_cpuinfo_INCLUDES_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include")
set(cpuinfo_cpuinfo_LIB_DIRS_RELEASE "/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/lib")
set(cpuinfo_cpuinfo_RES_DIRS_RELEASE )
set(cpuinfo_cpuinfo_DEFINITIONS_RELEASE )
set(cpuinfo_cpuinfo_COMPILE_DEFINITIONS_RELEASE )
set(cpuinfo_cpuinfo_COMPILE_OPTIONS_C_RELEASE "")
set(cpuinfo_cpuinfo_COMPILE_OPTIONS_CXX_RELEASE "")
set(cpuinfo_cpuinfo_LIBS_RELEASE cpuinfo)
set(cpuinfo_cpuinfo_SYSTEM_LIBS_RELEASE )
set(cpuinfo_cpuinfo_FRAMEWORK_DIRS_RELEASE )
set(cpuinfo_cpuinfo_FRAMEWORKS_RELEASE )
set(cpuinfo_cpuinfo_BUILD_MODULES_PATHS_RELEASE )
set(cpuinfo_cpuinfo_DEPENDENCIES_RELEASE cpuinfo::clog)
set(cpuinfo_cpuinfo_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cpuinfo FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(cpuinfo_cpuinfo_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(cpuinfo_cpuinfo_FRAMEWORKS_FOUND_RELEASE "${cpuinfo_cpuinfo_FRAMEWORKS_RELEASE}" "${cpuinfo_cpuinfo_FRAMEWORK_DIRS_RELEASE}")

set(cpuinfo_cpuinfo_LIB_TARGETS_RELEASE "")
set(cpuinfo_cpuinfo_NOT_USED_RELEASE "")
set(cpuinfo_cpuinfo_LIBS_FRAMEWORKS_DEPS_RELEASE ${cpuinfo_cpuinfo_FRAMEWORKS_FOUND_RELEASE} ${cpuinfo_cpuinfo_SYSTEM_LIBS_RELEASE} ${cpuinfo_cpuinfo_DEPENDENCIES_RELEASE})
conan_package_library_targets("${cpuinfo_cpuinfo_LIBS_RELEASE}"
                              "${cpuinfo_cpuinfo_LIB_DIRS_RELEASE}"
                              "${cpuinfo_cpuinfo_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              cpuinfo_cpuinfo_NOT_USED_RELEASE
                              cpuinfo_cpuinfo_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "cpuinfo_cpuinfo")

set(cpuinfo_cpuinfo_LINK_LIBS_RELEASE ${cpuinfo_cpuinfo_LIB_TARGETS_RELEASE} ${cpuinfo_cpuinfo_LIBS_FRAMEWORKS_DEPS_RELEASE})