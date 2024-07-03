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


set(Eigen3_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(Eigen3_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(Eigen3_INCLUDES_RELEASE "/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(Eigen3_RES_DIRS_RELEASE )
set(Eigen3_DEFINITIONS_RELEASE )
set(Eigen3_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(Eigen3_COMPILE_DEFINITIONS_RELEASE )
set(Eigen3_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(Eigen3_COMPILE_OPTIONS_C_RELEASE "")
set(Eigen3_COMPILE_OPTIONS_CXX_RELEASE "")
set(Eigen3_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(Eigen3_LIBRARIES_RELEASE "") # Will be filled later
set(Eigen3_LIBS_RELEASE "") # Same as Eigen3_LIBRARIES
set(Eigen3_SYSTEM_LIBS_RELEASE )
set(Eigen3_FRAMEWORK_DIRS_RELEASE )
set(Eigen3_FRAMEWORKS_RELEASE )
set(Eigen3_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(Eigen3_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(Eigen3_FRAMEWORKS_FOUND_RELEASE "${Eigen3_FRAMEWORKS_RELEASE}" "${Eigen3_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(Eigen3_INCLUDE_DIRS_RELEASE
                 Eigen3_INCLUDE_DIR_RELEASE
                 Eigen3_INCLUDES_RELEASE
                 Eigen3_DEFINITIONS_RELEASE
                 Eigen3_LINKER_FLAGS_RELEASE_LIST
                 Eigen3_COMPILE_DEFINITIONS_RELEASE
                 Eigen3_COMPILE_OPTIONS_RELEASE_LIST
                 Eigen3_LIBRARIES_RELEASE
                 Eigen3_LIBS_RELEASE
                 Eigen3_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to Eigen3_LIBS and Eigen3_LIBRARY_LIST
set(Eigen3_LIBRARY_LIST_RELEASE )
set(Eigen3_LIB_DIRS_RELEASE )

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_Eigen3_DEPENDENCIES_RELEASE "${Eigen3_FRAMEWORKS_FOUND_RELEASE} ${Eigen3_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${Eigen3_LIBRARY_LIST_RELEASE}"  # libraries
                              "${Eigen3_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_Eigen3_DEPENDENCIES_RELEASE}"  # deps
                              Eigen3_LIBRARIES_RELEASE            # out_libraries
                              Eigen3_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "Eigen3")                                      # package_name

set(Eigen3_LIBS_RELEASE ${Eigen3_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${Eigen3_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND Eigen3_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND Eigen3_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${Eigen3_SYSTEM_LIBS_RELEASE})
    list(APPEND Eigen3_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND Eigen3_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(Eigen3_LIBRARIES_TARGETS_RELEASE "${Eigen3_LIBRARIES_TARGETS_RELEASE};")
set(Eigen3_LIBRARIES_RELEASE "${Eigen3_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(Eigen3_COMPONENTS_RELEASE Eigen3::Eigen)

########### COMPONENT Eigen VARIABLES #############################################

set(Eigen3_Eigen_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(Eigen3_Eigen_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(Eigen3_Eigen_INCLUDES_RELEASE "/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3")
set(Eigen3_Eigen_LIB_DIRS_RELEASE )
set(Eigen3_Eigen_RES_DIRS_RELEASE )
set(Eigen3_Eigen_DEFINITIONS_RELEASE )
set(Eigen3_Eigen_COMPILE_DEFINITIONS_RELEASE )
set(Eigen3_Eigen_COMPILE_OPTIONS_C_RELEASE "")
set(Eigen3_Eigen_COMPILE_OPTIONS_CXX_RELEASE "")
set(Eigen3_Eigen_LIBS_RELEASE )
set(Eigen3_Eigen_SYSTEM_LIBS_RELEASE )
set(Eigen3_Eigen_FRAMEWORK_DIRS_RELEASE )
set(Eigen3_Eigen_FRAMEWORKS_RELEASE )
set(Eigen3_Eigen_BUILD_MODULES_PATHS_RELEASE )
set(Eigen3_Eigen_DEPENDENCIES_RELEASE )
set(Eigen3_Eigen_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT Eigen FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(Eigen3_Eigen_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(Eigen3_Eigen_FRAMEWORKS_FOUND_RELEASE "${Eigen3_Eigen_FRAMEWORKS_RELEASE}" "${Eigen3_Eigen_FRAMEWORK_DIRS_RELEASE}")

set(Eigen3_Eigen_LIB_TARGETS_RELEASE "")
set(Eigen3_Eigen_NOT_USED_RELEASE "")
set(Eigen3_Eigen_LIBS_FRAMEWORKS_DEPS_RELEASE ${Eigen3_Eigen_FRAMEWORKS_FOUND_RELEASE} ${Eigen3_Eigen_SYSTEM_LIBS_RELEASE} ${Eigen3_Eigen_DEPENDENCIES_RELEASE})
conan_package_library_targets("${Eigen3_Eigen_LIBS_RELEASE}"
                              "${Eigen3_Eigen_LIB_DIRS_RELEASE}"
                              "${Eigen3_Eigen_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              Eigen3_Eigen_NOT_USED_RELEASE
                              Eigen3_Eigen_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "Eigen3_Eigen")

set(Eigen3_Eigen_LINK_LIBS_RELEASE ${Eigen3_Eigen_LIB_TARGETS_RELEASE} ${Eigen3_Eigen_LIBS_FRAMEWORKS_DEPS_RELEASE})