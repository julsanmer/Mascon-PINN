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


set(Microsoft.GSL_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(Microsoft.GSL_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(Microsoft.GSL_INCLUDES_RELEASE "/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(Microsoft.GSL_RES_DIRS_RELEASE )
set(Microsoft.GSL_DEFINITIONS_RELEASE )
set(Microsoft.GSL_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(Microsoft.GSL_COMPILE_DEFINITIONS_RELEASE )
set(Microsoft.GSL_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(Microsoft.GSL_COMPILE_OPTIONS_C_RELEASE "")
set(Microsoft.GSL_COMPILE_OPTIONS_CXX_RELEASE "")
set(Microsoft.GSL_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(Microsoft.GSL_LIBRARIES_RELEASE "") # Will be filled later
set(Microsoft.GSL_LIBS_RELEASE "") # Same as Microsoft.GSL_LIBRARIES
set(Microsoft.GSL_SYSTEM_LIBS_RELEASE )
set(Microsoft.GSL_FRAMEWORK_DIRS_RELEASE )
set(Microsoft.GSL_FRAMEWORKS_RELEASE )
set(Microsoft.GSL_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(Microsoft.GSL_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(Microsoft.GSL_FRAMEWORKS_FOUND_RELEASE "${Microsoft.GSL_FRAMEWORKS_RELEASE}" "${Microsoft.GSL_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(Microsoft.GSL_INCLUDE_DIRS_RELEASE
                 Microsoft.GSL_INCLUDE_DIR_RELEASE
                 Microsoft.GSL_INCLUDES_RELEASE
                 Microsoft.GSL_DEFINITIONS_RELEASE
                 Microsoft.GSL_LINKER_FLAGS_RELEASE_LIST
                 Microsoft.GSL_COMPILE_DEFINITIONS_RELEASE
                 Microsoft.GSL_COMPILE_OPTIONS_RELEASE_LIST
                 Microsoft.GSL_LIBRARIES_RELEASE
                 Microsoft.GSL_LIBS_RELEASE
                 Microsoft.GSL_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to Microsoft.GSL_LIBS and Microsoft.GSL_LIBRARY_LIST
set(Microsoft.GSL_LIBRARY_LIST_RELEASE )
set(Microsoft.GSL_LIB_DIRS_RELEASE )

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_Microsoft.GSL_DEPENDENCIES_RELEASE "${Microsoft.GSL_FRAMEWORKS_FOUND_RELEASE} ${Microsoft.GSL_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${Microsoft.GSL_LIBRARY_LIST_RELEASE}"  # libraries
                              "${Microsoft.GSL_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_Microsoft.GSL_DEPENDENCIES_RELEASE}"  # deps
                              Microsoft.GSL_LIBRARIES_RELEASE            # out_libraries
                              Microsoft.GSL_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "Microsoft.GSL")                                      # package_name

set(Microsoft.GSL_LIBS_RELEASE ${Microsoft.GSL_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${Microsoft.GSL_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND Microsoft.GSL_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND Microsoft.GSL_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${Microsoft.GSL_SYSTEM_LIBS_RELEASE})
    list(APPEND Microsoft.GSL_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND Microsoft.GSL_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(Microsoft.GSL_LIBRARIES_TARGETS_RELEASE "${Microsoft.GSL_LIBRARIES_TARGETS_RELEASE};")
set(Microsoft.GSL_LIBRARIES_RELEASE "${Microsoft.GSL_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(Microsoft.GSL_COMPONENTS_RELEASE Microsoft.GSL::GSL)

########### COMPONENT GSL VARIABLES #############################################

set(Microsoft.GSL_GSL_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(Microsoft.GSL_GSL_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(Microsoft.GSL_GSL_INCLUDES_RELEASE "/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include")
set(Microsoft.GSL_GSL_LIB_DIRS_RELEASE )
set(Microsoft.GSL_GSL_RES_DIRS_RELEASE )
set(Microsoft.GSL_GSL_DEFINITIONS_RELEASE )
set(Microsoft.GSL_GSL_COMPILE_DEFINITIONS_RELEASE )
set(Microsoft.GSL_GSL_COMPILE_OPTIONS_C_RELEASE "")
set(Microsoft.GSL_GSL_COMPILE_OPTIONS_CXX_RELEASE "")
set(Microsoft.GSL_GSL_LIBS_RELEASE )
set(Microsoft.GSL_GSL_SYSTEM_LIBS_RELEASE )
set(Microsoft.GSL_GSL_FRAMEWORK_DIRS_RELEASE )
set(Microsoft.GSL_GSL_FRAMEWORKS_RELEASE )
set(Microsoft.GSL_GSL_BUILD_MODULES_PATHS_RELEASE )
set(Microsoft.GSL_GSL_DEPENDENCIES_RELEASE )
set(Microsoft.GSL_GSL_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT GSL FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(Microsoft.GSL_GSL_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(Microsoft.GSL_GSL_FRAMEWORKS_FOUND_RELEASE "${Microsoft.GSL_GSL_FRAMEWORKS_RELEASE}" "${Microsoft.GSL_GSL_FRAMEWORK_DIRS_RELEASE}")

set(Microsoft.GSL_GSL_LIB_TARGETS_RELEASE "")
set(Microsoft.GSL_GSL_NOT_USED_RELEASE "")
set(Microsoft.GSL_GSL_LIBS_FRAMEWORKS_DEPS_RELEASE ${Microsoft.GSL_GSL_FRAMEWORKS_FOUND_RELEASE} ${Microsoft.GSL_GSL_SYSTEM_LIBS_RELEASE} ${Microsoft.GSL_GSL_DEPENDENCIES_RELEASE})
conan_package_library_targets("${Microsoft.GSL_GSL_LIBS_RELEASE}"
                              "${Microsoft.GSL_GSL_LIB_DIRS_RELEASE}"
                              "${Microsoft.GSL_GSL_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              Microsoft.GSL_GSL_NOT_USED_RELEASE
                              Microsoft.GSL_GSL_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "Microsoft.GSL_GSL")

set(Microsoft.GSL_GSL_LINK_LIBS_RELEASE ${Microsoft.GSL_GSL_LIB_TARGETS_RELEASE} ${Microsoft.GSL_GSL_LIBS_FRAMEWORKS_DEPS_RELEASE})