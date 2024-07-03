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


set(CURL_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include")
set(CURL_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include")
set(CURL_INCLUDES_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include")
set(CURL_RES_DIRS_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/res")
set(CURL_DEFINITIONS_RELEASE "-DCURL_STATICLIB=1")
set(CURL_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(CURL_COMPILE_DEFINITIONS_RELEASE "CURL_STATICLIB=1")
set(CURL_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(CURL_COMPILE_OPTIONS_C_RELEASE "")
set(CURL_COMPILE_OPTIONS_CXX_RELEASE "")
set(CURL_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(CURL_LIBRARIES_RELEASE "") # Will be filled later
set(CURL_LIBS_RELEASE "") # Same as CURL_LIBRARIES
set(CURL_SYSTEM_LIBS_RELEASE )
set(CURL_FRAMEWORK_DIRS_RELEASE )
set(CURL_FRAMEWORKS_RELEASE CoreFoundation SystemConfiguration Security)
set(CURL_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(CURL_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(CURL_FRAMEWORKS_FOUND_RELEASE "${CURL_FRAMEWORKS_RELEASE}" "${CURL_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(CURL_INCLUDE_DIRS_RELEASE
                 CURL_INCLUDE_DIR_RELEASE
                 CURL_INCLUDES_RELEASE
                 CURL_DEFINITIONS_RELEASE
                 CURL_LINKER_FLAGS_RELEASE_LIST
                 CURL_COMPILE_DEFINITIONS_RELEASE
                 CURL_COMPILE_OPTIONS_RELEASE_LIST
                 CURL_LIBRARIES_RELEASE
                 CURL_LIBS_RELEASE
                 CURL_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to CURL_LIBS and CURL_LIBRARY_LIST
set(CURL_LIBRARY_LIST_RELEASE curl)
set(CURL_LIB_DIRS_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_CURL_DEPENDENCIES_RELEASE "${CURL_FRAMEWORKS_FOUND_RELEASE} ${CURL_SYSTEM_LIBS_RELEASE} ZLIB::ZLIB")

conan_package_library_targets("${CURL_LIBRARY_LIST_RELEASE}"  # libraries
                              "${CURL_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_CURL_DEPENDENCIES_RELEASE}"  # deps
                              CURL_LIBRARIES_RELEASE            # out_libraries
                              CURL_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "CURL")                                      # package_name

set(CURL_LIBS_RELEASE ${CURL_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${CURL_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND CURL_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND CURL_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${CURL_SYSTEM_LIBS_RELEASE})
    list(APPEND CURL_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND CURL_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(CURL_LIBRARIES_TARGETS_RELEASE "${CURL_LIBRARIES_TARGETS_RELEASE};ZLIB::ZLIB")
set(CURL_LIBRARIES_RELEASE "${CURL_LIBRARIES_RELEASE};ZLIB::ZLIB")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(CURL_COMPONENTS_RELEASE CURL::libcurl)

########### COMPONENT libcurl VARIABLES #############################################

set(CURL_libcurl_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include")
set(CURL_libcurl_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include")
set(CURL_libcurl_INCLUDES_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include")
set(CURL_libcurl_LIB_DIRS_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/lib")
set(CURL_libcurl_RES_DIRS_RELEASE "/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/res")
set(CURL_libcurl_DEFINITIONS_RELEASE "-DCURL_STATICLIB=1")
set(CURL_libcurl_COMPILE_DEFINITIONS_RELEASE "CURL_STATICLIB=1")
set(CURL_libcurl_COMPILE_OPTIONS_C_RELEASE "")
set(CURL_libcurl_COMPILE_OPTIONS_CXX_RELEASE "")
set(CURL_libcurl_LIBS_RELEASE curl)
set(CURL_libcurl_SYSTEM_LIBS_RELEASE )
set(CURL_libcurl_FRAMEWORK_DIRS_RELEASE )
set(CURL_libcurl_FRAMEWORKS_RELEASE CoreFoundation SystemConfiguration Security)
set(CURL_libcurl_BUILD_MODULES_PATHS_RELEASE )
set(CURL_libcurl_DEPENDENCIES_RELEASE ZLIB::ZLIB)
set(CURL_libcurl_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT libcurl FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(CURL_libcurl_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(CURL_libcurl_FRAMEWORKS_FOUND_RELEASE "${CURL_libcurl_FRAMEWORKS_RELEASE}" "${CURL_libcurl_FRAMEWORK_DIRS_RELEASE}")

set(CURL_libcurl_LIB_TARGETS_RELEASE "")
set(CURL_libcurl_NOT_USED_RELEASE "")
set(CURL_libcurl_LIBS_FRAMEWORKS_DEPS_RELEASE ${CURL_libcurl_FRAMEWORKS_FOUND_RELEASE} ${CURL_libcurl_SYSTEM_LIBS_RELEASE} ${CURL_libcurl_DEPENDENCIES_RELEASE})
conan_package_library_targets("${CURL_libcurl_LIBS_RELEASE}"
                              "${CURL_libcurl_LIB_DIRS_RELEASE}"
                              "${CURL_libcurl_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              CURL_libcurl_NOT_USED_RELEASE
                              CURL_libcurl_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "CURL_libcurl")

set(CURL_libcurl_LINK_LIBS_RELEASE ${CURL_libcurl_LIB_TARGETS_RELEASE} ${CURL_libcurl_LIBS_FRAMEWORKS_DEPS_RELEASE})