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


set(date_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include")
set(date_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include")
set(date_INCLUDES_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include")
set(date_RES_DIRS_RELEASE )
set(date_DEFINITIONS_RELEASE "-DUSE_OS_TZDB=0")
set(date_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(date_COMPILE_DEFINITIONS_RELEASE "USE_OS_TZDB=0")
set(date_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(date_COMPILE_OPTIONS_C_RELEASE "")
set(date_COMPILE_OPTIONS_CXX_RELEASE "")
set(date_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(date_LIBRARIES_RELEASE "") # Will be filled later
set(date_LIBS_RELEASE "") # Same as date_LIBRARIES
set(date_SYSTEM_LIBS_RELEASE )
set(date_FRAMEWORK_DIRS_RELEASE )
set(date_FRAMEWORKS_RELEASE )
set(date_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(date_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(date_FRAMEWORKS_FOUND_RELEASE "${date_FRAMEWORKS_RELEASE}" "${date_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(date_INCLUDE_DIRS_RELEASE
                 date_INCLUDE_DIR_RELEASE
                 date_INCLUDES_RELEASE
                 date_DEFINITIONS_RELEASE
                 date_LINKER_FLAGS_RELEASE_LIST
                 date_COMPILE_DEFINITIONS_RELEASE
                 date_COMPILE_OPTIONS_RELEASE_LIST
                 date_LIBRARIES_RELEASE
                 date_LIBS_RELEASE
                 date_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to date_LIBS and date_LIBRARY_LIST
set(date_LIBRARY_LIST_RELEASE date-tz)
set(date_LIB_DIRS_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_date_DEPENDENCIES_RELEASE "${date_FRAMEWORKS_FOUND_RELEASE} ${date_SYSTEM_LIBS_RELEASE} CURL::CURL")

conan_package_library_targets("${date_LIBRARY_LIST_RELEASE}"  # libraries
                              "${date_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_date_DEPENDENCIES_RELEASE}"  # deps
                              date_LIBRARIES_RELEASE            # out_libraries
                              date_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "date")                                      # package_name

set(date_LIBS_RELEASE ${date_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${date_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND date_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND date_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${date_SYSTEM_LIBS_RELEASE})
    list(APPEND date_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND date_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(date_LIBRARIES_TARGETS_RELEASE "${date_LIBRARIES_TARGETS_RELEASE};CURL::CURL")
set(date_LIBRARIES_RELEASE "${date_LIBRARIES_RELEASE};CURL::CURL")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(date_COMPONENTS_RELEASE date::date-tz)

########### COMPONENT date-tz VARIABLES #############################################

set(date_date-tz_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include")
set(date_date-tz_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include")
set(date_date-tz_INCLUDES_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include")
set(date_date-tz_LIB_DIRS_RELEASE "/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/lib")
set(date_date-tz_RES_DIRS_RELEASE )
set(date_date-tz_DEFINITIONS_RELEASE "-DUSE_OS_TZDB=0")
set(date_date-tz_COMPILE_DEFINITIONS_RELEASE "USE_OS_TZDB=0")
set(date_date-tz_COMPILE_OPTIONS_C_RELEASE "")
set(date_date-tz_COMPILE_OPTIONS_CXX_RELEASE "")
set(date_date-tz_LIBS_RELEASE date-tz)
set(date_date-tz_SYSTEM_LIBS_RELEASE )
set(date_date-tz_FRAMEWORK_DIRS_RELEASE )
set(date_date-tz_FRAMEWORKS_RELEASE )
set(date_date-tz_BUILD_MODULES_PATHS_RELEASE )
set(date_date-tz_DEPENDENCIES_RELEASE CURL::CURL)
set(date_date-tz_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT date-tz FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(date_date-tz_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(date_date-tz_FRAMEWORKS_FOUND_RELEASE "${date_date-tz_FRAMEWORKS_RELEASE}" "${date_date-tz_FRAMEWORK_DIRS_RELEASE}")

set(date_date-tz_LIB_TARGETS_RELEASE "")
set(date_date-tz_NOT_USED_RELEASE "")
set(date_date-tz_LIBS_FRAMEWORKS_DEPS_RELEASE ${date_date-tz_FRAMEWORKS_FOUND_RELEASE} ${date_date-tz_SYSTEM_LIBS_RELEASE} ${date_date-tz_DEPENDENCIES_RELEASE})
conan_package_library_targets("${date_date-tz_LIBS_RELEASE}"
                              "${date_date-tz_LIB_DIRS_RELEASE}"
                              "${date_date-tz_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              date_date-tz_NOT_USED_RELEASE
                              date_date-tz_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "date_date-tz")

set(date_date-tz_LINK_LIBS_RELEASE ${date_date-tz_LIB_TARGETS_RELEASE} ${date_date-tz_LIBS_FRAMEWORKS_DEPS_RELEASE})