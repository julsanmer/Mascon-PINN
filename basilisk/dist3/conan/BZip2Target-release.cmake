
set(BZip2_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/bzip2/1.0.8/_/_/package/238a93dc813ca1550968399f1f8925565feeff8e/include")
set(BZip2_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/bzip2/1.0.8/_/_/package/238a93dc813ca1550968399f1f8925565feeff8e/include")
set(BZip2_INCLUDES_RELEASE "/Users/julio/.conan/data/bzip2/1.0.8/_/_/package/238a93dc813ca1550968399f1f8925565feeff8e/include")
set(BZip2_RES_DIRS_RELEASE )
set(BZip2_DEFINITIONS_RELEASE )
set(BZip2_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(BZip2_COMPILE_DEFINITIONS_RELEASE )
set(BZip2_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(BZip2_COMPILE_OPTIONS_C_RELEASE "")
set(BZip2_COMPILE_OPTIONS_CXX_RELEASE "")
set(BZip2_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(BZip2_LIBRARIES_RELEASE "") # Will be filled later
set(BZip2_LIBS_RELEASE "") # Same as BZip2_LIBRARIES
set(BZip2_SYSTEM_LIBS_RELEASE )
set(BZip2_FRAMEWORK_DIRS_RELEASE )
set(BZip2_FRAMEWORKS_RELEASE )
set(BZip2_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(BZip2_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(BZip2_FRAMEWORKS_FOUND_RELEASE "${BZip2_FRAMEWORKS_RELEASE}" "${BZip2_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(BZip2_INCLUDE_DIRS_RELEASE
                 BZip2_INCLUDE_DIR_RELEASE
                 BZip2_INCLUDES_RELEASE
                 BZip2_DEFINITIONS_RELEASE
                 BZip2_LINKER_FLAGS_RELEASE_LIST
                 BZip2_COMPILE_DEFINITIONS_RELEASE
                 BZip2_COMPILE_OPTIONS_RELEASE_LIST
                 BZip2_LIBRARIES_RELEASE
                 BZip2_LIBS_RELEASE
                 BZip2_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to BZip2_LIBS and BZip2_LIBRARY_LIST
set(BZip2_LIBRARY_LIST_RELEASE bz2)
set(BZip2_LIB_DIRS_RELEASE "/Users/julio/.conan/data/bzip2/1.0.8/_/_/package/238a93dc813ca1550968399f1f8925565feeff8e/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_BZip2_DEPENDENCIES_RELEASE "${BZip2_FRAMEWORKS_FOUND_RELEASE} ${BZip2_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${BZip2_LIBRARY_LIST_RELEASE}"  # libraries
                              "${BZip2_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_BZip2_DEPENDENCIES_RELEASE}"  # deps
                              BZip2_LIBRARIES_RELEASE            # out_libraries
                              BZip2_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "BZip2")                                      # package_name

set(BZip2_LIBS_RELEASE ${BZip2_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${BZip2_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND BZip2_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND BZip2_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${BZip2_SYSTEM_LIBS_RELEASE})
    list(APPEND BZip2_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND BZip2_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(BZip2_LIBRARIES_TARGETS_RELEASE "${BZip2_LIBRARIES_TARGETS_RELEASE};")
set(BZip2_LIBRARIES_RELEASE "${BZip2_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/bzip2/1.0.8/_/_/package/238a93dc813ca1550968399f1f8925565feeff8e/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/bzip2/1.0.8/_/_/package/238a93dc813ca1550968399f1f8925565feeff8e/" ${CMAKE_PREFIX_PATH})
