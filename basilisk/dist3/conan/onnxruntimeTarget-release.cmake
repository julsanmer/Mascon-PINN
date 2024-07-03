
set(onnxruntime_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include"
			"/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session")
set(onnxruntime_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include;/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session")
set(onnxruntime_INCLUDES_RELEASE "/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include"
			"/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session")
set(onnxruntime_RES_DIRS_RELEASE )
set(onnxruntime_DEFINITIONS_RELEASE )
set(onnxruntime_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(onnxruntime_COMPILE_DEFINITIONS_RELEASE )
set(onnxruntime_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(onnxruntime_COMPILE_OPTIONS_C_RELEASE "")
set(onnxruntime_COMPILE_OPTIONS_CXX_RELEASE "")
set(onnxruntime_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(onnxruntime_LIBRARIES_RELEASE "") # Will be filled later
set(onnxruntime_LIBS_RELEASE "") # Same as onnxruntime_LIBRARIES
set(onnxruntime_SYSTEM_LIBS_RELEASE )
set(onnxruntime_FRAMEWORK_DIRS_RELEASE )
set(onnxruntime_FRAMEWORKS_RELEASE Foundation)
set(onnxruntime_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(onnxruntime_BUILD_MODULES_PATHS_RELEASE )

conan_find_apple_frameworks(onnxruntime_FRAMEWORKS_FOUND_RELEASE "${onnxruntime_FRAMEWORKS_RELEASE}" "${onnxruntime_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(onnxruntime_INCLUDE_DIRS_RELEASE
                 onnxruntime_INCLUDE_DIR_RELEASE
                 onnxruntime_INCLUDES_RELEASE
                 onnxruntime_DEFINITIONS_RELEASE
                 onnxruntime_LINKER_FLAGS_RELEASE_LIST
                 onnxruntime_COMPILE_DEFINITIONS_RELEASE
                 onnxruntime_COMPILE_OPTIONS_RELEASE_LIST
                 onnxruntime_LIBRARIES_RELEASE
                 onnxruntime_LIBS_RELEASE
                 onnxruntime_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to onnxruntime_LIBS and onnxruntime_LIBRARY_LIST
set(onnxruntime_LIBRARY_LIST_RELEASE onnxruntime_session onnxruntime_optimizer onnxruntime_providers onnxruntime_framework onnxruntime_graph onnxruntime_util onnxruntime_mlas onnxruntime_common onnxruntime_flatbuffers)
set(onnxruntime_LIB_DIRS_RELEASE "/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_onnxruntime_DEPENDENCIES_RELEASE "${onnxruntime_FRAMEWORKS_FOUND_RELEASE} ${onnxruntime_SYSTEM_LIBS_RELEASE} absl::absl;protobuf::protobuf;date::date;re2::re2;ONNX::ONNX;flatbuffers::flatbuffers;Boost::headers;safeint::safeint;nlohmann_json::nlohmann_json;Eigen3::Eigen3;Microsoft.GSL::Microsoft.GSL;cpuinfo::cpuinfo;nsync::nsync")

conan_package_library_targets("${onnxruntime_LIBRARY_LIST_RELEASE}"  # libraries
                              "${onnxruntime_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_onnxruntime_DEPENDENCIES_RELEASE}"  # deps
                              onnxruntime_LIBRARIES_RELEASE            # out_libraries
                              onnxruntime_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "onnxruntime")                                      # package_name

set(onnxruntime_LIBS_RELEASE ${onnxruntime_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${onnxruntime_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND onnxruntime_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND onnxruntime_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${onnxruntime_SYSTEM_LIBS_RELEASE})
    list(APPEND onnxruntime_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND onnxruntime_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(onnxruntime_LIBRARIES_TARGETS_RELEASE "${onnxruntime_LIBRARIES_TARGETS_RELEASE};absl::absl;protobuf::protobuf;date::date;re2::re2;ONNX::ONNX;flatbuffers::flatbuffers;Boost::headers;safeint::safeint;nlohmann_json::nlohmann_json;Eigen3::Eigen3;Microsoft.GSL::Microsoft.GSL;cpuinfo::cpuinfo;nsync::nsync")
set(onnxruntime_LIBRARIES_RELEASE "${onnxruntime_LIBRARIES_RELEASE};absl::absl;protobuf::protobuf;date::date;re2::re2;ONNX::ONNX;flatbuffers::flatbuffers;Boost::headers;safeint::safeint;nlohmann_json::nlohmann_json;Eigen3::Eigen3;Microsoft.GSL::Microsoft.GSL;cpuinfo::cpuinfo;nsync::nsync")

set(CMAKE_MODULE_PATH "/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/" ${CMAKE_PREFIX_PATH})
