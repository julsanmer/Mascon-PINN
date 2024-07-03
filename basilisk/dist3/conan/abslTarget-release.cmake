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


set(absl_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_RES_DIRS_RELEASE )
set(absl_DEFINITIONS_RELEASE )
set(absl_LINKER_FLAGS_RELEASE_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(absl_COMPILE_DEFINITIONS_RELEASE )
set(absl_COMPILE_OPTIONS_RELEASE_LIST "" "")
set(absl_COMPILE_OPTIONS_C_RELEASE "")
set(absl_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_LIBRARIES_TARGETS_RELEASE "") # Will be filled later, if CMake 3
set(absl_LIBRARIES_RELEASE "") # Will be filled later
set(absl_LIBS_RELEASE "") # Same as absl_LIBRARIES
set(absl_SYSTEM_LIBS_RELEASE )
set(absl_FRAMEWORK_DIRS_RELEASE )
set(absl_FRAMEWORKS_RELEASE CoreFoundation)
set(absl_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
set(absl_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib/cmake/conan_trick/cxx_std.cmake")

conan_find_apple_frameworks(absl_FRAMEWORKS_FOUND_RELEASE "${absl_FRAMEWORKS_RELEASE}" "${absl_FRAMEWORK_DIRS_RELEASE}")

mark_as_advanced(absl_INCLUDE_DIRS_RELEASE
                 absl_INCLUDE_DIR_RELEASE
                 absl_INCLUDES_RELEASE
                 absl_DEFINITIONS_RELEASE
                 absl_LINKER_FLAGS_RELEASE_LIST
                 absl_COMPILE_DEFINITIONS_RELEASE
                 absl_COMPILE_OPTIONS_RELEASE_LIST
                 absl_LIBRARIES_RELEASE
                 absl_LIBS_RELEASE
                 absl_LIBRARIES_TARGETS_RELEASE)

# Find the real .lib/.a and add them to absl_LIBS and absl_LIBRARY_LIST
set(absl_LIBRARY_LIST_RELEASE absl_scoped_set_env absl_failure_signal_handler absl_leak_check absl_flags_parse absl_flags_usage absl_flags_usage_internal absl_log_internal_check_op absl_die_if_null absl_log_flags absl_flags absl_flags_reflection absl_raw_hash_set absl_hashtablez_sampler absl_flags_private_handle_accessor absl_flags_internal absl_flags_config absl_flags_program_name absl_flags_marshalling absl_flags_commandlineflag absl_flags_commandlineflag_internal absl_log_initialize absl_log_internal_conditions absl_log_internal_message absl_examine_stack absl_log_internal_format absl_log_internal_proto absl_log_internal_nullguard absl_log_internal_log_sink_set absl_log_internal_globals absl_log_globals absl_hash absl_city absl_low_level_hash absl_log_sink absl_log_entry absl_periodic_sampler absl_random_distributions absl_random_seed_sequences absl_random_internal_pool_urbg absl_random_seed_gen_exception absl_random_internal_seed_material absl_random_internal_randen absl_random_internal_randen_slow absl_random_internal_randen_hwaes absl_random_internal_randen_hwaes_impl absl_random_internal_platform absl_random_internal_distribution_test_util absl_statusor absl_status absl_strerror absl_str_format_internal absl_cordz_sample_token absl_cord absl_cordz_info absl_cord_internal absl_crc_cord_state absl_crc32c absl_crc_internal absl_crc_cpu_detect absl_cordz_functions absl_exponential_biased absl_cordz_handle absl_synchronization absl_stacktrace absl_symbolize absl_debugging_internal absl_demangle_internal absl_graphcycles_internal absl_malloc_internal absl_time absl_strings absl_int128 absl_strings_internal absl_base absl_spinlock_wait absl_civil_time absl_time_zone absl_bad_any_cast_impl absl_throw_delegate absl_bad_optional_access absl_bad_variant_access absl_raw_logging_internal absl_log_severity)
set(absl_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_absl_DEPENDENCIES_RELEASE "${absl_FRAMEWORKS_FOUND_RELEASE} ${absl_SYSTEM_LIBS_RELEASE} ")

conan_package_library_targets("${absl_LIBRARY_LIST_RELEASE}"  # libraries
                              "${absl_LIB_DIRS_RELEASE}"      # package_libdir
                              "${_absl_DEPENDENCIES_RELEASE}"  # deps
                              absl_LIBRARIES_RELEASE            # out_libraries
                              absl_LIBRARIES_TARGETS_RELEASE    # out_libraries_targets
                              "_RELEASE"                          # build_type
                              "absl")                                      # package_name

set(absl_LIBS_RELEASE ${absl_LIBRARIES_RELEASE})

foreach(_FRAMEWORK ${absl_FRAMEWORKS_FOUND_RELEASE})
    list(APPEND absl_LIBRARIES_TARGETS_RELEASE ${_FRAMEWORK})
    list(APPEND absl_LIBRARIES_RELEASE ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${absl_SYSTEM_LIBS_RELEASE})
    list(APPEND absl_LIBRARIES_TARGETS_RELEASE ${_SYSTEM_LIB})
    list(APPEND absl_LIBRARIES_RELEASE ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(absl_LIBRARIES_TARGETS_RELEASE "${absl_LIBRARIES_TARGETS_RELEASE};")
set(absl_LIBRARIES_RELEASE "${absl_LIBRARIES_RELEASE};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

set(absl_COMPONENTS_RELEASE absl::pretty_function absl::scoped_set_env absl::btree absl::counting_allocator absl::flat_hash_set absl::node_hash_map absl::node_hash_set absl::hashtable_debug absl::node_slot_policy absl::failure_signal_handler absl::debugging absl::leak_check absl::flags_parse absl::flags_usage absl::flags_usage_internal absl::any_invocable absl::bind_front absl::absl_check absl::check absl::log_internal_check_impl absl::log_internal_check_op absl::die_if_null absl::log_flags absl::log_internal_flags absl::flags absl::flags_reflection absl::flat_hash_map absl::algorithm_container absl::hash_function_defaults absl::raw_hash_map absl::raw_hash_set absl::hash_policy_traits absl::common_policy_traits absl::hashtablez_sampler absl::hashtable_debug_hooks absl::container_common absl::flags_private_handle_accessor absl::flags_internal absl::flags_config absl::flags_program_name absl::flags_path_util absl::flags_marshalling absl::flags_commandlineflag absl::flags_commandlineflag_internal absl::log_initialize absl::log absl::log_streamer absl::absl_log absl::log_internal_log_impl absl::log_internal_conditions absl::log_internal_strip absl::log_internal_nullstream absl::log_internal_voidify absl::log_structured absl::log_internal_structured absl::log_internal_message absl::examine_stack absl::log_internal_format absl::log_internal_proto absl::log_internal_nullguard absl::log_internal_append_truncated absl::log_sink_registry absl::log_internal_log_sink_set absl::cleanup absl::cleanup_internal absl::log_internal_globals absl::log_globals absl::hash absl::city absl::low_level_hash absl::log_sink absl::log_entry absl::log_internal_config absl::numeric absl::sample_recorder absl::periodic_sampler absl::random_random absl::random_bit_gen_ref absl::random_internal_mock_helpers absl::random_distributions absl::random_seed_sequences absl::random_internal_distribution_caller absl::random_internal_generate_real absl::random_internal_wide_multiply absl::random_internal_nonsecure_base absl::random_internal_pool_urbg absl::random_seed_gen_exception absl::random_internal_salted_seed_seq absl::random_internal_seed_material absl::random_internal_fast_uniform_bits absl::random_internal_pcg_engine absl::random_internal_fastmath absl::random_internal_randen_engine absl::random_internal_iostream_state_saver absl::random_internal_randen absl::random_internal_randen_slow absl::random_internal_randen_hwaes absl::random_internal_randen_hwaes_impl absl::random_internal_platform absl::random_internal_distribution_test_util absl::random_internal_uniform_helper absl::random_internal_traits absl::statusor absl::status absl::strerror absl::str_format absl::str_format_internal absl::numeric_representation absl::cordz_sample_token absl::cord absl::fixed_array absl::function_ref absl::cordz_update_scope absl::cordz_info absl::cord_internal absl::inlined_vector absl::inlined_vector_internal absl::compressed_tuple absl::container_memory absl::layout absl::crc_cord_state absl::crc32c absl::crc_internal absl::prefetch absl::crc_cpu_detect absl::non_temporal_memcpy absl::non_temporal_arm_intrinsics absl::cordz_functions absl::exponential_biased absl::cordz_statistics absl::cordz_update_tracker absl::cordz_handle absl::synchronization absl::stacktrace absl::symbolize absl::debugging_internal absl::demangle_internal absl::graphcycles_internal absl::malloc_internal absl::kernel_timeout_internal absl::time absl::strings absl::int128 absl::bits absl::strings_internal absl::endian absl::base absl::spinlock_wait absl::dynamic_annotations absl::civil_time absl::time_zone absl::any absl::fast_type_id absl::bad_any_cast absl::bad_any_cast_impl absl::span absl::throw_delegate absl::algorithm absl::optional absl::memory absl::meta absl::bad_optional_access absl::variant absl::bad_variant_access absl::raw_logging_internal absl::atomic_hook absl::errno_saver absl::log_severity absl::compare absl::core_headers absl::utility absl::base_internal absl::type_traits absl::config)

########### COMPONENT config VARIABLES #############################################

set(absl_config_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_config_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_config_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_config_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_config_RES_DIRS_RELEASE )
set(absl_config_DEFINITIONS_RELEASE )
set(absl_config_COMPILE_DEFINITIONS_RELEASE )
set(absl_config_COMPILE_OPTIONS_C_RELEASE "")
set(absl_config_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_config_LIBS_RELEASE )
set(absl_config_SYSTEM_LIBS_RELEASE )
set(absl_config_FRAMEWORK_DIRS_RELEASE )
set(absl_config_FRAMEWORKS_RELEASE )
set(absl_config_BUILD_MODULES_PATHS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib/cmake/conan_trick/cxx_std.cmake")
set(absl_config_DEPENDENCIES_RELEASE )
set(absl_config_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT config FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_config_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_config_FRAMEWORKS_FOUND_RELEASE "${absl_config_FRAMEWORKS_RELEASE}" "${absl_config_FRAMEWORK_DIRS_RELEASE}")

set(absl_config_LIB_TARGETS_RELEASE "")
set(absl_config_NOT_USED_RELEASE "")
set(absl_config_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_config_FRAMEWORKS_FOUND_RELEASE} ${absl_config_SYSTEM_LIBS_RELEASE} ${absl_config_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_config_LIBS_RELEASE}"
                              "${absl_config_LIB_DIRS_RELEASE}"
                              "${absl_config_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_config_NOT_USED_RELEASE
                              absl_config_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_config")

set(absl_config_LINK_LIBS_RELEASE ${absl_config_LIB_TARGETS_RELEASE} ${absl_config_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT type_traits VARIABLES #############################################

set(absl_type_traits_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_type_traits_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_type_traits_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_type_traits_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_type_traits_RES_DIRS_RELEASE )
set(absl_type_traits_DEFINITIONS_RELEASE )
set(absl_type_traits_COMPILE_DEFINITIONS_RELEASE )
set(absl_type_traits_COMPILE_OPTIONS_C_RELEASE "")
set(absl_type_traits_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_type_traits_LIBS_RELEASE )
set(absl_type_traits_SYSTEM_LIBS_RELEASE )
set(absl_type_traits_FRAMEWORK_DIRS_RELEASE )
set(absl_type_traits_FRAMEWORKS_RELEASE )
set(absl_type_traits_BUILD_MODULES_PATHS_RELEASE )
set(absl_type_traits_DEPENDENCIES_RELEASE absl::config)
set(absl_type_traits_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT type_traits FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_type_traits_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_type_traits_FRAMEWORKS_FOUND_RELEASE "${absl_type_traits_FRAMEWORKS_RELEASE}" "${absl_type_traits_FRAMEWORK_DIRS_RELEASE}")

set(absl_type_traits_LIB_TARGETS_RELEASE "")
set(absl_type_traits_NOT_USED_RELEASE "")
set(absl_type_traits_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_type_traits_FRAMEWORKS_FOUND_RELEASE} ${absl_type_traits_SYSTEM_LIBS_RELEASE} ${absl_type_traits_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_type_traits_LIBS_RELEASE}"
                              "${absl_type_traits_LIB_DIRS_RELEASE}"
                              "${absl_type_traits_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_type_traits_NOT_USED_RELEASE
                              absl_type_traits_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_type_traits")

set(absl_type_traits_LINK_LIBS_RELEASE ${absl_type_traits_LIB_TARGETS_RELEASE} ${absl_type_traits_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT base_internal VARIABLES #############################################

set(absl_base_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_base_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_base_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_base_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_base_internal_RES_DIRS_RELEASE )
set(absl_base_internal_DEFINITIONS_RELEASE )
set(absl_base_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_base_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_base_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_base_internal_LIBS_RELEASE )
set(absl_base_internal_SYSTEM_LIBS_RELEASE )
set(absl_base_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_base_internal_FRAMEWORKS_RELEASE )
set(absl_base_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_base_internal_DEPENDENCIES_RELEASE absl::config absl::type_traits)
set(absl_base_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT base_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_base_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_base_internal_FRAMEWORKS_FOUND_RELEASE "${absl_base_internal_FRAMEWORKS_RELEASE}" "${absl_base_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_base_internal_LIB_TARGETS_RELEASE "")
set(absl_base_internal_NOT_USED_RELEASE "")
set(absl_base_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_base_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_base_internal_SYSTEM_LIBS_RELEASE} ${absl_base_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_base_internal_LIBS_RELEASE}"
                              "${absl_base_internal_LIB_DIRS_RELEASE}"
                              "${absl_base_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_base_internal_NOT_USED_RELEASE
                              absl_base_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_base_internal")

set(absl_base_internal_LINK_LIBS_RELEASE ${absl_base_internal_LIB_TARGETS_RELEASE} ${absl_base_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT utility VARIABLES #############################################

set(absl_utility_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_utility_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_utility_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_utility_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_utility_RES_DIRS_RELEASE )
set(absl_utility_DEFINITIONS_RELEASE )
set(absl_utility_COMPILE_DEFINITIONS_RELEASE )
set(absl_utility_COMPILE_OPTIONS_C_RELEASE "")
set(absl_utility_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_utility_LIBS_RELEASE )
set(absl_utility_SYSTEM_LIBS_RELEASE )
set(absl_utility_FRAMEWORK_DIRS_RELEASE )
set(absl_utility_FRAMEWORKS_RELEASE )
set(absl_utility_BUILD_MODULES_PATHS_RELEASE )
set(absl_utility_DEPENDENCIES_RELEASE absl::base_internal absl::config absl::type_traits)
set(absl_utility_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT utility FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_utility_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_utility_FRAMEWORKS_FOUND_RELEASE "${absl_utility_FRAMEWORKS_RELEASE}" "${absl_utility_FRAMEWORK_DIRS_RELEASE}")

set(absl_utility_LIB_TARGETS_RELEASE "")
set(absl_utility_NOT_USED_RELEASE "")
set(absl_utility_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_utility_FRAMEWORKS_FOUND_RELEASE} ${absl_utility_SYSTEM_LIBS_RELEASE} ${absl_utility_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_utility_LIBS_RELEASE}"
                              "${absl_utility_LIB_DIRS_RELEASE}"
                              "${absl_utility_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_utility_NOT_USED_RELEASE
                              absl_utility_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_utility")

set(absl_utility_LINK_LIBS_RELEASE ${absl_utility_LIB_TARGETS_RELEASE} ${absl_utility_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT core_headers VARIABLES #############################################

set(absl_core_headers_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_core_headers_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_core_headers_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_core_headers_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_core_headers_RES_DIRS_RELEASE )
set(absl_core_headers_DEFINITIONS_RELEASE )
set(absl_core_headers_COMPILE_DEFINITIONS_RELEASE )
set(absl_core_headers_COMPILE_OPTIONS_C_RELEASE "")
set(absl_core_headers_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_core_headers_LIBS_RELEASE )
set(absl_core_headers_SYSTEM_LIBS_RELEASE )
set(absl_core_headers_FRAMEWORK_DIRS_RELEASE )
set(absl_core_headers_FRAMEWORKS_RELEASE )
set(absl_core_headers_BUILD_MODULES_PATHS_RELEASE )
set(absl_core_headers_DEPENDENCIES_RELEASE absl::config)
set(absl_core_headers_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT core_headers FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_core_headers_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_core_headers_FRAMEWORKS_FOUND_RELEASE "${absl_core_headers_FRAMEWORKS_RELEASE}" "${absl_core_headers_FRAMEWORK_DIRS_RELEASE}")

set(absl_core_headers_LIB_TARGETS_RELEASE "")
set(absl_core_headers_NOT_USED_RELEASE "")
set(absl_core_headers_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_core_headers_FRAMEWORKS_FOUND_RELEASE} ${absl_core_headers_SYSTEM_LIBS_RELEASE} ${absl_core_headers_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_core_headers_LIBS_RELEASE}"
                              "${absl_core_headers_LIB_DIRS_RELEASE}"
                              "${absl_core_headers_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_core_headers_NOT_USED_RELEASE
                              absl_core_headers_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_core_headers")

set(absl_core_headers_LINK_LIBS_RELEASE ${absl_core_headers_LIB_TARGETS_RELEASE} ${absl_core_headers_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT compare VARIABLES #############################################

set(absl_compare_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_compare_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_compare_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_compare_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_compare_RES_DIRS_RELEASE )
set(absl_compare_DEFINITIONS_RELEASE )
set(absl_compare_COMPILE_DEFINITIONS_RELEASE )
set(absl_compare_COMPILE_OPTIONS_C_RELEASE "")
set(absl_compare_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_compare_LIBS_RELEASE )
set(absl_compare_SYSTEM_LIBS_RELEASE )
set(absl_compare_FRAMEWORK_DIRS_RELEASE )
set(absl_compare_FRAMEWORKS_RELEASE )
set(absl_compare_BUILD_MODULES_PATHS_RELEASE )
set(absl_compare_DEPENDENCIES_RELEASE absl::core_headers absl::type_traits)
set(absl_compare_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT compare FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_compare_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_compare_FRAMEWORKS_FOUND_RELEASE "${absl_compare_FRAMEWORKS_RELEASE}" "${absl_compare_FRAMEWORK_DIRS_RELEASE}")

set(absl_compare_LIB_TARGETS_RELEASE "")
set(absl_compare_NOT_USED_RELEASE "")
set(absl_compare_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_compare_FRAMEWORKS_FOUND_RELEASE} ${absl_compare_SYSTEM_LIBS_RELEASE} ${absl_compare_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_compare_LIBS_RELEASE}"
                              "${absl_compare_LIB_DIRS_RELEASE}"
                              "${absl_compare_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_compare_NOT_USED_RELEASE
                              absl_compare_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_compare")

set(absl_compare_LINK_LIBS_RELEASE ${absl_compare_LIB_TARGETS_RELEASE} ${absl_compare_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_severity VARIABLES #############################################

set(absl_log_severity_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_severity_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_severity_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_severity_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_severity_RES_DIRS_RELEASE )
set(absl_log_severity_DEFINITIONS_RELEASE )
set(absl_log_severity_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_severity_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_severity_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_severity_LIBS_RELEASE absl_log_severity)
set(absl_log_severity_SYSTEM_LIBS_RELEASE )
set(absl_log_severity_FRAMEWORK_DIRS_RELEASE )
set(absl_log_severity_FRAMEWORKS_RELEASE )
set(absl_log_severity_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_severity_DEPENDENCIES_RELEASE absl::core_headers)
set(absl_log_severity_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_severity FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_severity_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_severity_FRAMEWORKS_FOUND_RELEASE "${absl_log_severity_FRAMEWORKS_RELEASE}" "${absl_log_severity_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_severity_LIB_TARGETS_RELEASE "")
set(absl_log_severity_NOT_USED_RELEASE "")
set(absl_log_severity_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_severity_FRAMEWORKS_FOUND_RELEASE} ${absl_log_severity_SYSTEM_LIBS_RELEASE} ${absl_log_severity_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_severity_LIBS_RELEASE}"
                              "${absl_log_severity_LIB_DIRS_RELEASE}"
                              "${absl_log_severity_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_severity_NOT_USED_RELEASE
                              absl_log_severity_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_severity")

set(absl_log_severity_LINK_LIBS_RELEASE ${absl_log_severity_LIB_TARGETS_RELEASE} ${absl_log_severity_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT errno_saver VARIABLES #############################################

set(absl_errno_saver_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_errno_saver_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_errno_saver_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_errno_saver_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_errno_saver_RES_DIRS_RELEASE )
set(absl_errno_saver_DEFINITIONS_RELEASE )
set(absl_errno_saver_COMPILE_DEFINITIONS_RELEASE )
set(absl_errno_saver_COMPILE_OPTIONS_C_RELEASE "")
set(absl_errno_saver_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_errno_saver_LIBS_RELEASE )
set(absl_errno_saver_SYSTEM_LIBS_RELEASE )
set(absl_errno_saver_FRAMEWORK_DIRS_RELEASE )
set(absl_errno_saver_FRAMEWORKS_RELEASE )
set(absl_errno_saver_BUILD_MODULES_PATHS_RELEASE )
set(absl_errno_saver_DEPENDENCIES_RELEASE absl::config)
set(absl_errno_saver_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT errno_saver FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_errno_saver_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_errno_saver_FRAMEWORKS_FOUND_RELEASE "${absl_errno_saver_FRAMEWORKS_RELEASE}" "${absl_errno_saver_FRAMEWORK_DIRS_RELEASE}")

set(absl_errno_saver_LIB_TARGETS_RELEASE "")
set(absl_errno_saver_NOT_USED_RELEASE "")
set(absl_errno_saver_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_errno_saver_FRAMEWORKS_FOUND_RELEASE} ${absl_errno_saver_SYSTEM_LIBS_RELEASE} ${absl_errno_saver_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_errno_saver_LIBS_RELEASE}"
                              "${absl_errno_saver_LIB_DIRS_RELEASE}"
                              "${absl_errno_saver_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_errno_saver_NOT_USED_RELEASE
                              absl_errno_saver_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_errno_saver")

set(absl_errno_saver_LINK_LIBS_RELEASE ${absl_errno_saver_LIB_TARGETS_RELEASE} ${absl_errno_saver_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT atomic_hook VARIABLES #############################################

set(absl_atomic_hook_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_atomic_hook_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_atomic_hook_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_atomic_hook_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_atomic_hook_RES_DIRS_RELEASE )
set(absl_atomic_hook_DEFINITIONS_RELEASE )
set(absl_atomic_hook_COMPILE_DEFINITIONS_RELEASE )
set(absl_atomic_hook_COMPILE_OPTIONS_C_RELEASE "")
set(absl_atomic_hook_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_atomic_hook_LIBS_RELEASE )
set(absl_atomic_hook_SYSTEM_LIBS_RELEASE )
set(absl_atomic_hook_FRAMEWORK_DIRS_RELEASE )
set(absl_atomic_hook_FRAMEWORKS_RELEASE )
set(absl_atomic_hook_BUILD_MODULES_PATHS_RELEASE )
set(absl_atomic_hook_DEPENDENCIES_RELEASE absl::config absl::core_headers)
set(absl_atomic_hook_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT atomic_hook FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_atomic_hook_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_atomic_hook_FRAMEWORKS_FOUND_RELEASE "${absl_atomic_hook_FRAMEWORKS_RELEASE}" "${absl_atomic_hook_FRAMEWORK_DIRS_RELEASE}")

set(absl_atomic_hook_LIB_TARGETS_RELEASE "")
set(absl_atomic_hook_NOT_USED_RELEASE "")
set(absl_atomic_hook_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_atomic_hook_FRAMEWORKS_FOUND_RELEASE} ${absl_atomic_hook_SYSTEM_LIBS_RELEASE} ${absl_atomic_hook_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_atomic_hook_LIBS_RELEASE}"
                              "${absl_atomic_hook_LIB_DIRS_RELEASE}"
                              "${absl_atomic_hook_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_atomic_hook_NOT_USED_RELEASE
                              absl_atomic_hook_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_atomic_hook")

set(absl_atomic_hook_LINK_LIBS_RELEASE ${absl_atomic_hook_LIB_TARGETS_RELEASE} ${absl_atomic_hook_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT raw_logging_internal VARIABLES #############################################

set(absl_raw_logging_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_logging_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_logging_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_logging_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_raw_logging_internal_RES_DIRS_RELEASE )
set(absl_raw_logging_internal_DEFINITIONS_RELEASE )
set(absl_raw_logging_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_raw_logging_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_raw_logging_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_raw_logging_internal_LIBS_RELEASE absl_raw_logging_internal)
set(absl_raw_logging_internal_SYSTEM_LIBS_RELEASE )
set(absl_raw_logging_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_raw_logging_internal_FRAMEWORKS_RELEASE )
set(absl_raw_logging_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_raw_logging_internal_DEPENDENCIES_RELEASE absl::atomic_hook absl::config absl::core_headers absl::errno_saver absl::log_severity)
set(absl_raw_logging_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT raw_logging_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_raw_logging_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_raw_logging_internal_FRAMEWORKS_FOUND_RELEASE "${absl_raw_logging_internal_FRAMEWORKS_RELEASE}" "${absl_raw_logging_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_raw_logging_internal_LIB_TARGETS_RELEASE "")
set(absl_raw_logging_internal_NOT_USED_RELEASE "")
set(absl_raw_logging_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_raw_logging_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_raw_logging_internal_SYSTEM_LIBS_RELEASE} ${absl_raw_logging_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_raw_logging_internal_LIBS_RELEASE}"
                              "${absl_raw_logging_internal_LIB_DIRS_RELEASE}"
                              "${absl_raw_logging_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_raw_logging_internal_NOT_USED_RELEASE
                              absl_raw_logging_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_raw_logging_internal")

set(absl_raw_logging_internal_LINK_LIBS_RELEASE ${absl_raw_logging_internal_LIB_TARGETS_RELEASE} ${absl_raw_logging_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT bad_variant_access VARIABLES #############################################

set(absl_bad_variant_access_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_variant_access_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_variant_access_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_variant_access_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_bad_variant_access_RES_DIRS_RELEASE )
set(absl_bad_variant_access_DEFINITIONS_RELEASE )
set(absl_bad_variant_access_COMPILE_DEFINITIONS_RELEASE )
set(absl_bad_variant_access_COMPILE_OPTIONS_C_RELEASE "")
set(absl_bad_variant_access_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_bad_variant_access_LIBS_RELEASE absl_bad_variant_access)
set(absl_bad_variant_access_SYSTEM_LIBS_RELEASE )
set(absl_bad_variant_access_FRAMEWORK_DIRS_RELEASE )
set(absl_bad_variant_access_FRAMEWORKS_RELEASE )
set(absl_bad_variant_access_BUILD_MODULES_PATHS_RELEASE )
set(absl_bad_variant_access_DEPENDENCIES_RELEASE absl::config absl::raw_logging_internal)
set(absl_bad_variant_access_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT bad_variant_access FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_bad_variant_access_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_bad_variant_access_FRAMEWORKS_FOUND_RELEASE "${absl_bad_variant_access_FRAMEWORKS_RELEASE}" "${absl_bad_variant_access_FRAMEWORK_DIRS_RELEASE}")

set(absl_bad_variant_access_LIB_TARGETS_RELEASE "")
set(absl_bad_variant_access_NOT_USED_RELEASE "")
set(absl_bad_variant_access_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_bad_variant_access_FRAMEWORKS_FOUND_RELEASE} ${absl_bad_variant_access_SYSTEM_LIBS_RELEASE} ${absl_bad_variant_access_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_bad_variant_access_LIBS_RELEASE}"
                              "${absl_bad_variant_access_LIB_DIRS_RELEASE}"
                              "${absl_bad_variant_access_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_bad_variant_access_NOT_USED_RELEASE
                              absl_bad_variant_access_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_bad_variant_access")

set(absl_bad_variant_access_LINK_LIBS_RELEASE ${absl_bad_variant_access_LIB_TARGETS_RELEASE} ${absl_bad_variant_access_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT variant VARIABLES #############################################

set(absl_variant_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_variant_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_variant_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_variant_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_variant_RES_DIRS_RELEASE )
set(absl_variant_DEFINITIONS_RELEASE )
set(absl_variant_COMPILE_DEFINITIONS_RELEASE )
set(absl_variant_COMPILE_OPTIONS_C_RELEASE "")
set(absl_variant_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_variant_LIBS_RELEASE )
set(absl_variant_SYSTEM_LIBS_RELEASE )
set(absl_variant_FRAMEWORK_DIRS_RELEASE )
set(absl_variant_FRAMEWORKS_RELEASE )
set(absl_variant_BUILD_MODULES_PATHS_RELEASE )
set(absl_variant_DEPENDENCIES_RELEASE absl::bad_variant_access absl::base_internal absl::config absl::core_headers absl::type_traits absl::utility)
set(absl_variant_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT variant FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_variant_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_variant_FRAMEWORKS_FOUND_RELEASE "${absl_variant_FRAMEWORKS_RELEASE}" "${absl_variant_FRAMEWORK_DIRS_RELEASE}")

set(absl_variant_LIB_TARGETS_RELEASE "")
set(absl_variant_NOT_USED_RELEASE "")
set(absl_variant_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_variant_FRAMEWORKS_FOUND_RELEASE} ${absl_variant_SYSTEM_LIBS_RELEASE} ${absl_variant_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_variant_LIBS_RELEASE}"
                              "${absl_variant_LIB_DIRS_RELEASE}"
                              "${absl_variant_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_variant_NOT_USED_RELEASE
                              absl_variant_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_variant")

set(absl_variant_LINK_LIBS_RELEASE ${absl_variant_LIB_TARGETS_RELEASE} ${absl_variant_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT bad_optional_access VARIABLES #############################################

set(absl_bad_optional_access_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_optional_access_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_optional_access_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_optional_access_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_bad_optional_access_RES_DIRS_RELEASE )
set(absl_bad_optional_access_DEFINITIONS_RELEASE )
set(absl_bad_optional_access_COMPILE_DEFINITIONS_RELEASE )
set(absl_bad_optional_access_COMPILE_OPTIONS_C_RELEASE "")
set(absl_bad_optional_access_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_bad_optional_access_LIBS_RELEASE absl_bad_optional_access)
set(absl_bad_optional_access_SYSTEM_LIBS_RELEASE )
set(absl_bad_optional_access_FRAMEWORK_DIRS_RELEASE )
set(absl_bad_optional_access_FRAMEWORKS_RELEASE )
set(absl_bad_optional_access_BUILD_MODULES_PATHS_RELEASE )
set(absl_bad_optional_access_DEPENDENCIES_RELEASE absl::config absl::raw_logging_internal)
set(absl_bad_optional_access_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT bad_optional_access FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_bad_optional_access_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_bad_optional_access_FRAMEWORKS_FOUND_RELEASE "${absl_bad_optional_access_FRAMEWORKS_RELEASE}" "${absl_bad_optional_access_FRAMEWORK_DIRS_RELEASE}")

set(absl_bad_optional_access_LIB_TARGETS_RELEASE "")
set(absl_bad_optional_access_NOT_USED_RELEASE "")
set(absl_bad_optional_access_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_bad_optional_access_FRAMEWORKS_FOUND_RELEASE} ${absl_bad_optional_access_SYSTEM_LIBS_RELEASE} ${absl_bad_optional_access_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_bad_optional_access_LIBS_RELEASE}"
                              "${absl_bad_optional_access_LIB_DIRS_RELEASE}"
                              "${absl_bad_optional_access_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_bad_optional_access_NOT_USED_RELEASE
                              absl_bad_optional_access_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_bad_optional_access")

set(absl_bad_optional_access_LINK_LIBS_RELEASE ${absl_bad_optional_access_LIB_TARGETS_RELEASE} ${absl_bad_optional_access_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT meta VARIABLES #############################################

set(absl_meta_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_meta_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_meta_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_meta_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_meta_RES_DIRS_RELEASE )
set(absl_meta_DEFINITIONS_RELEASE )
set(absl_meta_COMPILE_DEFINITIONS_RELEASE )
set(absl_meta_COMPILE_OPTIONS_C_RELEASE "")
set(absl_meta_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_meta_LIBS_RELEASE )
set(absl_meta_SYSTEM_LIBS_RELEASE )
set(absl_meta_FRAMEWORK_DIRS_RELEASE )
set(absl_meta_FRAMEWORKS_RELEASE )
set(absl_meta_BUILD_MODULES_PATHS_RELEASE )
set(absl_meta_DEPENDENCIES_RELEASE absl::type_traits)
set(absl_meta_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT meta FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_meta_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_meta_FRAMEWORKS_FOUND_RELEASE "${absl_meta_FRAMEWORKS_RELEASE}" "${absl_meta_FRAMEWORK_DIRS_RELEASE}")

set(absl_meta_LIB_TARGETS_RELEASE "")
set(absl_meta_NOT_USED_RELEASE "")
set(absl_meta_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_meta_FRAMEWORKS_FOUND_RELEASE} ${absl_meta_SYSTEM_LIBS_RELEASE} ${absl_meta_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_meta_LIBS_RELEASE}"
                              "${absl_meta_LIB_DIRS_RELEASE}"
                              "${absl_meta_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_meta_NOT_USED_RELEASE
                              absl_meta_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_meta")

set(absl_meta_LINK_LIBS_RELEASE ${absl_meta_LIB_TARGETS_RELEASE} ${absl_meta_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT memory VARIABLES #############################################

set(absl_memory_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_memory_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_memory_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_memory_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_memory_RES_DIRS_RELEASE )
set(absl_memory_DEFINITIONS_RELEASE )
set(absl_memory_COMPILE_DEFINITIONS_RELEASE )
set(absl_memory_COMPILE_OPTIONS_C_RELEASE "")
set(absl_memory_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_memory_LIBS_RELEASE )
set(absl_memory_SYSTEM_LIBS_RELEASE )
set(absl_memory_FRAMEWORK_DIRS_RELEASE )
set(absl_memory_FRAMEWORKS_RELEASE )
set(absl_memory_BUILD_MODULES_PATHS_RELEASE )
set(absl_memory_DEPENDENCIES_RELEASE absl::core_headers absl::meta)
set(absl_memory_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT memory FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_memory_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_memory_FRAMEWORKS_FOUND_RELEASE "${absl_memory_FRAMEWORKS_RELEASE}" "${absl_memory_FRAMEWORK_DIRS_RELEASE}")

set(absl_memory_LIB_TARGETS_RELEASE "")
set(absl_memory_NOT_USED_RELEASE "")
set(absl_memory_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_memory_FRAMEWORKS_FOUND_RELEASE} ${absl_memory_SYSTEM_LIBS_RELEASE} ${absl_memory_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_memory_LIBS_RELEASE}"
                              "${absl_memory_LIB_DIRS_RELEASE}"
                              "${absl_memory_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_memory_NOT_USED_RELEASE
                              absl_memory_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_memory")

set(absl_memory_LINK_LIBS_RELEASE ${absl_memory_LIB_TARGETS_RELEASE} ${absl_memory_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT optional VARIABLES #############################################

set(absl_optional_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_optional_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_optional_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_optional_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_optional_RES_DIRS_RELEASE )
set(absl_optional_DEFINITIONS_RELEASE )
set(absl_optional_COMPILE_DEFINITIONS_RELEASE )
set(absl_optional_COMPILE_OPTIONS_C_RELEASE "")
set(absl_optional_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_optional_LIBS_RELEASE )
set(absl_optional_SYSTEM_LIBS_RELEASE )
set(absl_optional_FRAMEWORK_DIRS_RELEASE )
set(absl_optional_FRAMEWORKS_RELEASE )
set(absl_optional_BUILD_MODULES_PATHS_RELEASE )
set(absl_optional_DEPENDENCIES_RELEASE absl::bad_optional_access absl::base_internal absl::config absl::core_headers absl::memory absl::type_traits absl::utility)
set(absl_optional_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT optional FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_optional_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_optional_FRAMEWORKS_FOUND_RELEASE "${absl_optional_FRAMEWORKS_RELEASE}" "${absl_optional_FRAMEWORK_DIRS_RELEASE}")

set(absl_optional_LIB_TARGETS_RELEASE "")
set(absl_optional_NOT_USED_RELEASE "")
set(absl_optional_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_optional_FRAMEWORKS_FOUND_RELEASE} ${absl_optional_SYSTEM_LIBS_RELEASE} ${absl_optional_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_optional_LIBS_RELEASE}"
                              "${absl_optional_LIB_DIRS_RELEASE}"
                              "${absl_optional_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_optional_NOT_USED_RELEASE
                              absl_optional_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_optional")

set(absl_optional_LINK_LIBS_RELEASE ${absl_optional_LIB_TARGETS_RELEASE} ${absl_optional_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT algorithm VARIABLES #############################################

set(absl_algorithm_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_algorithm_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_algorithm_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_algorithm_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_algorithm_RES_DIRS_RELEASE )
set(absl_algorithm_DEFINITIONS_RELEASE )
set(absl_algorithm_COMPILE_DEFINITIONS_RELEASE )
set(absl_algorithm_COMPILE_OPTIONS_C_RELEASE "")
set(absl_algorithm_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_algorithm_LIBS_RELEASE )
set(absl_algorithm_SYSTEM_LIBS_RELEASE )
set(absl_algorithm_FRAMEWORK_DIRS_RELEASE )
set(absl_algorithm_FRAMEWORKS_RELEASE )
set(absl_algorithm_BUILD_MODULES_PATHS_RELEASE )
set(absl_algorithm_DEPENDENCIES_RELEASE absl::config)
set(absl_algorithm_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT algorithm FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_algorithm_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_algorithm_FRAMEWORKS_FOUND_RELEASE "${absl_algorithm_FRAMEWORKS_RELEASE}" "${absl_algorithm_FRAMEWORK_DIRS_RELEASE}")

set(absl_algorithm_LIB_TARGETS_RELEASE "")
set(absl_algorithm_NOT_USED_RELEASE "")
set(absl_algorithm_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_algorithm_FRAMEWORKS_FOUND_RELEASE} ${absl_algorithm_SYSTEM_LIBS_RELEASE} ${absl_algorithm_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_algorithm_LIBS_RELEASE}"
                              "${absl_algorithm_LIB_DIRS_RELEASE}"
                              "${absl_algorithm_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_algorithm_NOT_USED_RELEASE
                              absl_algorithm_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_algorithm")

set(absl_algorithm_LINK_LIBS_RELEASE ${absl_algorithm_LIB_TARGETS_RELEASE} ${absl_algorithm_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT throw_delegate VARIABLES #############################################

set(absl_throw_delegate_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_throw_delegate_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_throw_delegate_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_throw_delegate_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_throw_delegate_RES_DIRS_RELEASE )
set(absl_throw_delegate_DEFINITIONS_RELEASE )
set(absl_throw_delegate_COMPILE_DEFINITIONS_RELEASE )
set(absl_throw_delegate_COMPILE_OPTIONS_C_RELEASE "")
set(absl_throw_delegate_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_throw_delegate_LIBS_RELEASE absl_throw_delegate)
set(absl_throw_delegate_SYSTEM_LIBS_RELEASE )
set(absl_throw_delegate_FRAMEWORK_DIRS_RELEASE )
set(absl_throw_delegate_FRAMEWORKS_RELEASE )
set(absl_throw_delegate_BUILD_MODULES_PATHS_RELEASE )
set(absl_throw_delegate_DEPENDENCIES_RELEASE absl::config absl::raw_logging_internal)
set(absl_throw_delegate_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT throw_delegate FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_throw_delegate_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_throw_delegate_FRAMEWORKS_FOUND_RELEASE "${absl_throw_delegate_FRAMEWORKS_RELEASE}" "${absl_throw_delegate_FRAMEWORK_DIRS_RELEASE}")

set(absl_throw_delegate_LIB_TARGETS_RELEASE "")
set(absl_throw_delegate_NOT_USED_RELEASE "")
set(absl_throw_delegate_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_throw_delegate_FRAMEWORKS_FOUND_RELEASE} ${absl_throw_delegate_SYSTEM_LIBS_RELEASE} ${absl_throw_delegate_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_throw_delegate_LIBS_RELEASE}"
                              "${absl_throw_delegate_LIB_DIRS_RELEASE}"
                              "${absl_throw_delegate_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_throw_delegate_NOT_USED_RELEASE
                              absl_throw_delegate_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_throw_delegate")

set(absl_throw_delegate_LINK_LIBS_RELEASE ${absl_throw_delegate_LIB_TARGETS_RELEASE} ${absl_throw_delegate_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT span VARIABLES #############################################

set(absl_span_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_span_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_span_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_span_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_span_RES_DIRS_RELEASE )
set(absl_span_DEFINITIONS_RELEASE )
set(absl_span_COMPILE_DEFINITIONS_RELEASE )
set(absl_span_COMPILE_OPTIONS_C_RELEASE "")
set(absl_span_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_span_LIBS_RELEASE )
set(absl_span_SYSTEM_LIBS_RELEASE )
set(absl_span_FRAMEWORK_DIRS_RELEASE )
set(absl_span_FRAMEWORKS_RELEASE )
set(absl_span_BUILD_MODULES_PATHS_RELEASE )
set(absl_span_DEPENDENCIES_RELEASE absl::algorithm absl::core_headers absl::throw_delegate absl::type_traits)
set(absl_span_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT span FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_span_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_span_FRAMEWORKS_FOUND_RELEASE "${absl_span_FRAMEWORKS_RELEASE}" "${absl_span_FRAMEWORK_DIRS_RELEASE}")

set(absl_span_LIB_TARGETS_RELEASE "")
set(absl_span_NOT_USED_RELEASE "")
set(absl_span_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_span_FRAMEWORKS_FOUND_RELEASE} ${absl_span_SYSTEM_LIBS_RELEASE} ${absl_span_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_span_LIBS_RELEASE}"
                              "${absl_span_LIB_DIRS_RELEASE}"
                              "${absl_span_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_span_NOT_USED_RELEASE
                              absl_span_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_span")

set(absl_span_LINK_LIBS_RELEASE ${absl_span_LIB_TARGETS_RELEASE} ${absl_span_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT bad_any_cast_impl VARIABLES #############################################

set(absl_bad_any_cast_impl_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_any_cast_impl_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_any_cast_impl_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_any_cast_impl_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_bad_any_cast_impl_RES_DIRS_RELEASE )
set(absl_bad_any_cast_impl_DEFINITIONS_RELEASE )
set(absl_bad_any_cast_impl_COMPILE_DEFINITIONS_RELEASE )
set(absl_bad_any_cast_impl_COMPILE_OPTIONS_C_RELEASE "")
set(absl_bad_any_cast_impl_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_bad_any_cast_impl_LIBS_RELEASE absl_bad_any_cast_impl)
set(absl_bad_any_cast_impl_SYSTEM_LIBS_RELEASE )
set(absl_bad_any_cast_impl_FRAMEWORK_DIRS_RELEASE )
set(absl_bad_any_cast_impl_FRAMEWORKS_RELEASE )
set(absl_bad_any_cast_impl_BUILD_MODULES_PATHS_RELEASE )
set(absl_bad_any_cast_impl_DEPENDENCIES_RELEASE absl::config absl::raw_logging_internal)
set(absl_bad_any_cast_impl_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT bad_any_cast_impl FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_bad_any_cast_impl_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_bad_any_cast_impl_FRAMEWORKS_FOUND_RELEASE "${absl_bad_any_cast_impl_FRAMEWORKS_RELEASE}" "${absl_bad_any_cast_impl_FRAMEWORK_DIRS_RELEASE}")

set(absl_bad_any_cast_impl_LIB_TARGETS_RELEASE "")
set(absl_bad_any_cast_impl_NOT_USED_RELEASE "")
set(absl_bad_any_cast_impl_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_bad_any_cast_impl_FRAMEWORKS_FOUND_RELEASE} ${absl_bad_any_cast_impl_SYSTEM_LIBS_RELEASE} ${absl_bad_any_cast_impl_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_bad_any_cast_impl_LIBS_RELEASE}"
                              "${absl_bad_any_cast_impl_LIB_DIRS_RELEASE}"
                              "${absl_bad_any_cast_impl_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_bad_any_cast_impl_NOT_USED_RELEASE
                              absl_bad_any_cast_impl_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_bad_any_cast_impl")

set(absl_bad_any_cast_impl_LINK_LIBS_RELEASE ${absl_bad_any_cast_impl_LIB_TARGETS_RELEASE} ${absl_bad_any_cast_impl_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT bad_any_cast VARIABLES #############################################

set(absl_bad_any_cast_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_any_cast_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_any_cast_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bad_any_cast_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_bad_any_cast_RES_DIRS_RELEASE )
set(absl_bad_any_cast_DEFINITIONS_RELEASE )
set(absl_bad_any_cast_COMPILE_DEFINITIONS_RELEASE )
set(absl_bad_any_cast_COMPILE_OPTIONS_C_RELEASE "")
set(absl_bad_any_cast_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_bad_any_cast_LIBS_RELEASE )
set(absl_bad_any_cast_SYSTEM_LIBS_RELEASE )
set(absl_bad_any_cast_FRAMEWORK_DIRS_RELEASE )
set(absl_bad_any_cast_FRAMEWORKS_RELEASE )
set(absl_bad_any_cast_BUILD_MODULES_PATHS_RELEASE )
set(absl_bad_any_cast_DEPENDENCIES_RELEASE absl::bad_any_cast_impl absl::config)
set(absl_bad_any_cast_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT bad_any_cast FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_bad_any_cast_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_bad_any_cast_FRAMEWORKS_FOUND_RELEASE "${absl_bad_any_cast_FRAMEWORKS_RELEASE}" "${absl_bad_any_cast_FRAMEWORK_DIRS_RELEASE}")

set(absl_bad_any_cast_LIB_TARGETS_RELEASE "")
set(absl_bad_any_cast_NOT_USED_RELEASE "")
set(absl_bad_any_cast_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_bad_any_cast_FRAMEWORKS_FOUND_RELEASE} ${absl_bad_any_cast_SYSTEM_LIBS_RELEASE} ${absl_bad_any_cast_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_bad_any_cast_LIBS_RELEASE}"
                              "${absl_bad_any_cast_LIB_DIRS_RELEASE}"
                              "${absl_bad_any_cast_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_bad_any_cast_NOT_USED_RELEASE
                              absl_bad_any_cast_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_bad_any_cast")

set(absl_bad_any_cast_LINK_LIBS_RELEASE ${absl_bad_any_cast_LIB_TARGETS_RELEASE} ${absl_bad_any_cast_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT fast_type_id VARIABLES #############################################

set(absl_fast_type_id_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_fast_type_id_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_fast_type_id_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_fast_type_id_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_fast_type_id_RES_DIRS_RELEASE )
set(absl_fast_type_id_DEFINITIONS_RELEASE )
set(absl_fast_type_id_COMPILE_DEFINITIONS_RELEASE )
set(absl_fast_type_id_COMPILE_OPTIONS_C_RELEASE "")
set(absl_fast_type_id_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_fast_type_id_LIBS_RELEASE )
set(absl_fast_type_id_SYSTEM_LIBS_RELEASE )
set(absl_fast_type_id_FRAMEWORK_DIRS_RELEASE )
set(absl_fast_type_id_FRAMEWORKS_RELEASE )
set(absl_fast_type_id_BUILD_MODULES_PATHS_RELEASE )
set(absl_fast_type_id_DEPENDENCIES_RELEASE absl::config)
set(absl_fast_type_id_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT fast_type_id FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_fast_type_id_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_fast_type_id_FRAMEWORKS_FOUND_RELEASE "${absl_fast_type_id_FRAMEWORKS_RELEASE}" "${absl_fast_type_id_FRAMEWORK_DIRS_RELEASE}")

set(absl_fast_type_id_LIB_TARGETS_RELEASE "")
set(absl_fast_type_id_NOT_USED_RELEASE "")
set(absl_fast_type_id_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_fast_type_id_FRAMEWORKS_FOUND_RELEASE} ${absl_fast_type_id_SYSTEM_LIBS_RELEASE} ${absl_fast_type_id_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_fast_type_id_LIBS_RELEASE}"
                              "${absl_fast_type_id_LIB_DIRS_RELEASE}"
                              "${absl_fast_type_id_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_fast_type_id_NOT_USED_RELEASE
                              absl_fast_type_id_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_fast_type_id")

set(absl_fast_type_id_LINK_LIBS_RELEASE ${absl_fast_type_id_LIB_TARGETS_RELEASE} ${absl_fast_type_id_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT any VARIABLES #############################################

set(absl_any_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_any_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_any_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_any_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_any_RES_DIRS_RELEASE )
set(absl_any_DEFINITIONS_RELEASE )
set(absl_any_COMPILE_DEFINITIONS_RELEASE )
set(absl_any_COMPILE_OPTIONS_C_RELEASE "")
set(absl_any_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_any_LIBS_RELEASE )
set(absl_any_SYSTEM_LIBS_RELEASE )
set(absl_any_FRAMEWORK_DIRS_RELEASE )
set(absl_any_FRAMEWORKS_RELEASE )
set(absl_any_BUILD_MODULES_PATHS_RELEASE )
set(absl_any_DEPENDENCIES_RELEASE absl::bad_any_cast absl::config absl::core_headers absl::fast_type_id absl::type_traits absl::utility)
set(absl_any_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT any FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_any_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_any_FRAMEWORKS_FOUND_RELEASE "${absl_any_FRAMEWORKS_RELEASE}" "${absl_any_FRAMEWORK_DIRS_RELEASE}")

set(absl_any_LIB_TARGETS_RELEASE "")
set(absl_any_NOT_USED_RELEASE "")
set(absl_any_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_any_FRAMEWORKS_FOUND_RELEASE} ${absl_any_SYSTEM_LIBS_RELEASE} ${absl_any_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_any_LIBS_RELEASE}"
                              "${absl_any_LIB_DIRS_RELEASE}"
                              "${absl_any_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_any_NOT_USED_RELEASE
                              absl_any_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_any")

set(absl_any_LINK_LIBS_RELEASE ${absl_any_LIB_TARGETS_RELEASE} ${absl_any_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT time_zone VARIABLES #############################################

set(absl_time_zone_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_time_zone_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_time_zone_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_time_zone_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_time_zone_RES_DIRS_RELEASE )
set(absl_time_zone_DEFINITIONS_RELEASE )
set(absl_time_zone_COMPILE_DEFINITIONS_RELEASE )
set(absl_time_zone_COMPILE_OPTIONS_C_RELEASE "")
set(absl_time_zone_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_time_zone_LIBS_RELEASE absl_time_zone)
set(absl_time_zone_SYSTEM_LIBS_RELEASE )
set(absl_time_zone_FRAMEWORK_DIRS_RELEASE )
set(absl_time_zone_FRAMEWORKS_RELEASE CoreFoundation)
set(absl_time_zone_BUILD_MODULES_PATHS_RELEASE )
set(absl_time_zone_DEPENDENCIES_RELEASE )
set(absl_time_zone_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT time_zone FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_time_zone_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_time_zone_FRAMEWORKS_FOUND_RELEASE "${absl_time_zone_FRAMEWORKS_RELEASE}" "${absl_time_zone_FRAMEWORK_DIRS_RELEASE}")

set(absl_time_zone_LIB_TARGETS_RELEASE "")
set(absl_time_zone_NOT_USED_RELEASE "")
set(absl_time_zone_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_time_zone_FRAMEWORKS_FOUND_RELEASE} ${absl_time_zone_SYSTEM_LIBS_RELEASE} ${absl_time_zone_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_time_zone_LIBS_RELEASE}"
                              "${absl_time_zone_LIB_DIRS_RELEASE}"
                              "${absl_time_zone_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_time_zone_NOT_USED_RELEASE
                              absl_time_zone_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_time_zone")

set(absl_time_zone_LINK_LIBS_RELEASE ${absl_time_zone_LIB_TARGETS_RELEASE} ${absl_time_zone_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT civil_time VARIABLES #############################################

set(absl_civil_time_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_civil_time_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_civil_time_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_civil_time_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_civil_time_RES_DIRS_RELEASE )
set(absl_civil_time_DEFINITIONS_RELEASE )
set(absl_civil_time_COMPILE_DEFINITIONS_RELEASE )
set(absl_civil_time_COMPILE_OPTIONS_C_RELEASE "")
set(absl_civil_time_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_civil_time_LIBS_RELEASE absl_civil_time)
set(absl_civil_time_SYSTEM_LIBS_RELEASE )
set(absl_civil_time_FRAMEWORK_DIRS_RELEASE )
set(absl_civil_time_FRAMEWORKS_RELEASE )
set(absl_civil_time_BUILD_MODULES_PATHS_RELEASE )
set(absl_civil_time_DEPENDENCIES_RELEASE )
set(absl_civil_time_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT civil_time FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_civil_time_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_civil_time_FRAMEWORKS_FOUND_RELEASE "${absl_civil_time_FRAMEWORKS_RELEASE}" "${absl_civil_time_FRAMEWORK_DIRS_RELEASE}")

set(absl_civil_time_LIB_TARGETS_RELEASE "")
set(absl_civil_time_NOT_USED_RELEASE "")
set(absl_civil_time_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_civil_time_FRAMEWORKS_FOUND_RELEASE} ${absl_civil_time_SYSTEM_LIBS_RELEASE} ${absl_civil_time_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_civil_time_LIBS_RELEASE}"
                              "${absl_civil_time_LIB_DIRS_RELEASE}"
                              "${absl_civil_time_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_civil_time_NOT_USED_RELEASE
                              absl_civil_time_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_civil_time")

set(absl_civil_time_LINK_LIBS_RELEASE ${absl_civil_time_LIB_TARGETS_RELEASE} ${absl_civil_time_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT dynamic_annotations VARIABLES #############################################

set(absl_dynamic_annotations_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_dynamic_annotations_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_dynamic_annotations_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_dynamic_annotations_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_dynamic_annotations_RES_DIRS_RELEASE )
set(absl_dynamic_annotations_DEFINITIONS_RELEASE )
set(absl_dynamic_annotations_COMPILE_DEFINITIONS_RELEASE )
set(absl_dynamic_annotations_COMPILE_OPTIONS_C_RELEASE "")
set(absl_dynamic_annotations_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_dynamic_annotations_LIBS_RELEASE )
set(absl_dynamic_annotations_SYSTEM_LIBS_RELEASE )
set(absl_dynamic_annotations_FRAMEWORK_DIRS_RELEASE )
set(absl_dynamic_annotations_FRAMEWORKS_RELEASE )
set(absl_dynamic_annotations_BUILD_MODULES_PATHS_RELEASE )
set(absl_dynamic_annotations_DEPENDENCIES_RELEASE absl::config)
set(absl_dynamic_annotations_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT dynamic_annotations FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_dynamic_annotations_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_dynamic_annotations_FRAMEWORKS_FOUND_RELEASE "${absl_dynamic_annotations_FRAMEWORKS_RELEASE}" "${absl_dynamic_annotations_FRAMEWORK_DIRS_RELEASE}")

set(absl_dynamic_annotations_LIB_TARGETS_RELEASE "")
set(absl_dynamic_annotations_NOT_USED_RELEASE "")
set(absl_dynamic_annotations_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_dynamic_annotations_FRAMEWORKS_FOUND_RELEASE} ${absl_dynamic_annotations_SYSTEM_LIBS_RELEASE} ${absl_dynamic_annotations_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_dynamic_annotations_LIBS_RELEASE}"
                              "${absl_dynamic_annotations_LIB_DIRS_RELEASE}"
                              "${absl_dynamic_annotations_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_dynamic_annotations_NOT_USED_RELEASE
                              absl_dynamic_annotations_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_dynamic_annotations")

set(absl_dynamic_annotations_LINK_LIBS_RELEASE ${absl_dynamic_annotations_LIB_TARGETS_RELEASE} ${absl_dynamic_annotations_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT spinlock_wait VARIABLES #############################################

set(absl_spinlock_wait_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_spinlock_wait_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_spinlock_wait_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_spinlock_wait_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_spinlock_wait_RES_DIRS_RELEASE )
set(absl_spinlock_wait_DEFINITIONS_RELEASE )
set(absl_spinlock_wait_COMPILE_DEFINITIONS_RELEASE )
set(absl_spinlock_wait_COMPILE_OPTIONS_C_RELEASE "")
set(absl_spinlock_wait_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_spinlock_wait_LIBS_RELEASE absl_spinlock_wait)
set(absl_spinlock_wait_SYSTEM_LIBS_RELEASE )
set(absl_spinlock_wait_FRAMEWORK_DIRS_RELEASE )
set(absl_spinlock_wait_FRAMEWORKS_RELEASE )
set(absl_spinlock_wait_BUILD_MODULES_PATHS_RELEASE )
set(absl_spinlock_wait_DEPENDENCIES_RELEASE absl::base_internal absl::core_headers absl::errno_saver)
set(absl_spinlock_wait_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT spinlock_wait FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_spinlock_wait_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_spinlock_wait_FRAMEWORKS_FOUND_RELEASE "${absl_spinlock_wait_FRAMEWORKS_RELEASE}" "${absl_spinlock_wait_FRAMEWORK_DIRS_RELEASE}")

set(absl_spinlock_wait_LIB_TARGETS_RELEASE "")
set(absl_spinlock_wait_NOT_USED_RELEASE "")
set(absl_spinlock_wait_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_spinlock_wait_FRAMEWORKS_FOUND_RELEASE} ${absl_spinlock_wait_SYSTEM_LIBS_RELEASE} ${absl_spinlock_wait_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_spinlock_wait_LIBS_RELEASE}"
                              "${absl_spinlock_wait_LIB_DIRS_RELEASE}"
                              "${absl_spinlock_wait_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_spinlock_wait_NOT_USED_RELEASE
                              absl_spinlock_wait_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_spinlock_wait")

set(absl_spinlock_wait_LINK_LIBS_RELEASE ${absl_spinlock_wait_LIB_TARGETS_RELEASE} ${absl_spinlock_wait_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT base VARIABLES #############################################

set(absl_base_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_base_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_base_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_base_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_base_RES_DIRS_RELEASE )
set(absl_base_DEFINITIONS_RELEASE )
set(absl_base_COMPILE_DEFINITIONS_RELEASE )
set(absl_base_COMPILE_OPTIONS_C_RELEASE "")
set(absl_base_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_base_LIBS_RELEASE absl_base)
set(absl_base_SYSTEM_LIBS_RELEASE )
set(absl_base_FRAMEWORK_DIRS_RELEASE )
set(absl_base_FRAMEWORKS_RELEASE )
set(absl_base_BUILD_MODULES_PATHS_RELEASE )
set(absl_base_DEPENDENCIES_RELEASE absl::atomic_hook absl::base_internal absl::config absl::core_headers absl::dynamic_annotations absl::log_severity absl::raw_logging_internal absl::spinlock_wait absl::type_traits)
set(absl_base_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT base FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_base_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_base_FRAMEWORKS_FOUND_RELEASE "${absl_base_FRAMEWORKS_RELEASE}" "${absl_base_FRAMEWORK_DIRS_RELEASE}")

set(absl_base_LIB_TARGETS_RELEASE "")
set(absl_base_NOT_USED_RELEASE "")
set(absl_base_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_base_FRAMEWORKS_FOUND_RELEASE} ${absl_base_SYSTEM_LIBS_RELEASE} ${absl_base_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_base_LIBS_RELEASE}"
                              "${absl_base_LIB_DIRS_RELEASE}"
                              "${absl_base_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_base_NOT_USED_RELEASE
                              absl_base_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_base")

set(absl_base_LINK_LIBS_RELEASE ${absl_base_LIB_TARGETS_RELEASE} ${absl_base_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT endian VARIABLES #############################################

set(absl_endian_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_endian_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_endian_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_endian_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_endian_RES_DIRS_RELEASE )
set(absl_endian_DEFINITIONS_RELEASE )
set(absl_endian_COMPILE_DEFINITIONS_RELEASE )
set(absl_endian_COMPILE_OPTIONS_C_RELEASE "")
set(absl_endian_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_endian_LIBS_RELEASE )
set(absl_endian_SYSTEM_LIBS_RELEASE )
set(absl_endian_FRAMEWORK_DIRS_RELEASE )
set(absl_endian_FRAMEWORKS_RELEASE )
set(absl_endian_BUILD_MODULES_PATHS_RELEASE )
set(absl_endian_DEPENDENCIES_RELEASE absl::base absl::config absl::core_headers)
set(absl_endian_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT endian FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_endian_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_endian_FRAMEWORKS_FOUND_RELEASE "${absl_endian_FRAMEWORKS_RELEASE}" "${absl_endian_FRAMEWORK_DIRS_RELEASE}")

set(absl_endian_LIB_TARGETS_RELEASE "")
set(absl_endian_NOT_USED_RELEASE "")
set(absl_endian_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_endian_FRAMEWORKS_FOUND_RELEASE} ${absl_endian_SYSTEM_LIBS_RELEASE} ${absl_endian_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_endian_LIBS_RELEASE}"
                              "${absl_endian_LIB_DIRS_RELEASE}"
                              "${absl_endian_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_endian_NOT_USED_RELEASE
                              absl_endian_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_endian")

set(absl_endian_LINK_LIBS_RELEASE ${absl_endian_LIB_TARGETS_RELEASE} ${absl_endian_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT strings_internal VARIABLES #############################################

set(absl_strings_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strings_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strings_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strings_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_strings_internal_RES_DIRS_RELEASE )
set(absl_strings_internal_DEFINITIONS_RELEASE )
set(absl_strings_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_strings_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_strings_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_strings_internal_LIBS_RELEASE absl_strings_internal)
set(absl_strings_internal_SYSTEM_LIBS_RELEASE )
set(absl_strings_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_strings_internal_FRAMEWORKS_RELEASE )
set(absl_strings_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_strings_internal_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::endian absl::raw_logging_internal absl::type_traits)
set(absl_strings_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT strings_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_strings_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_strings_internal_FRAMEWORKS_FOUND_RELEASE "${absl_strings_internal_FRAMEWORKS_RELEASE}" "${absl_strings_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_strings_internal_LIB_TARGETS_RELEASE "")
set(absl_strings_internal_NOT_USED_RELEASE "")
set(absl_strings_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_strings_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_strings_internal_SYSTEM_LIBS_RELEASE} ${absl_strings_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_strings_internal_LIBS_RELEASE}"
                              "${absl_strings_internal_LIB_DIRS_RELEASE}"
                              "${absl_strings_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_strings_internal_NOT_USED_RELEASE
                              absl_strings_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_strings_internal")

set(absl_strings_internal_LINK_LIBS_RELEASE ${absl_strings_internal_LIB_TARGETS_RELEASE} ${absl_strings_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT bits VARIABLES #############################################

set(absl_bits_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bits_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bits_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bits_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_bits_RES_DIRS_RELEASE )
set(absl_bits_DEFINITIONS_RELEASE )
set(absl_bits_COMPILE_DEFINITIONS_RELEASE )
set(absl_bits_COMPILE_OPTIONS_C_RELEASE "")
set(absl_bits_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_bits_LIBS_RELEASE )
set(absl_bits_SYSTEM_LIBS_RELEASE )
set(absl_bits_FRAMEWORK_DIRS_RELEASE )
set(absl_bits_FRAMEWORKS_RELEASE )
set(absl_bits_BUILD_MODULES_PATHS_RELEASE )
set(absl_bits_DEPENDENCIES_RELEASE absl::core_headers)
set(absl_bits_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT bits FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_bits_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_bits_FRAMEWORKS_FOUND_RELEASE "${absl_bits_FRAMEWORKS_RELEASE}" "${absl_bits_FRAMEWORK_DIRS_RELEASE}")

set(absl_bits_LIB_TARGETS_RELEASE "")
set(absl_bits_NOT_USED_RELEASE "")
set(absl_bits_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_bits_FRAMEWORKS_FOUND_RELEASE} ${absl_bits_SYSTEM_LIBS_RELEASE} ${absl_bits_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_bits_LIBS_RELEASE}"
                              "${absl_bits_LIB_DIRS_RELEASE}"
                              "${absl_bits_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_bits_NOT_USED_RELEASE
                              absl_bits_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_bits")

set(absl_bits_LINK_LIBS_RELEASE ${absl_bits_LIB_TARGETS_RELEASE} ${absl_bits_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT int128 VARIABLES #############################################

set(absl_int128_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_int128_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_int128_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_int128_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_int128_RES_DIRS_RELEASE )
set(absl_int128_DEFINITIONS_RELEASE )
set(absl_int128_COMPILE_DEFINITIONS_RELEASE )
set(absl_int128_COMPILE_OPTIONS_C_RELEASE "")
set(absl_int128_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_int128_LIBS_RELEASE absl_int128)
set(absl_int128_SYSTEM_LIBS_RELEASE )
set(absl_int128_FRAMEWORK_DIRS_RELEASE )
set(absl_int128_FRAMEWORKS_RELEASE )
set(absl_int128_BUILD_MODULES_PATHS_RELEASE )
set(absl_int128_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::bits)
set(absl_int128_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT int128 FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_int128_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_int128_FRAMEWORKS_FOUND_RELEASE "${absl_int128_FRAMEWORKS_RELEASE}" "${absl_int128_FRAMEWORK_DIRS_RELEASE}")

set(absl_int128_LIB_TARGETS_RELEASE "")
set(absl_int128_NOT_USED_RELEASE "")
set(absl_int128_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_int128_FRAMEWORKS_FOUND_RELEASE} ${absl_int128_SYSTEM_LIBS_RELEASE} ${absl_int128_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_int128_LIBS_RELEASE}"
                              "${absl_int128_LIB_DIRS_RELEASE}"
                              "${absl_int128_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_int128_NOT_USED_RELEASE
                              absl_int128_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_int128")

set(absl_int128_LINK_LIBS_RELEASE ${absl_int128_LIB_TARGETS_RELEASE} ${absl_int128_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT strings VARIABLES #############################################

set(absl_strings_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strings_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strings_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strings_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_strings_RES_DIRS_RELEASE )
set(absl_strings_DEFINITIONS_RELEASE )
set(absl_strings_COMPILE_DEFINITIONS_RELEASE )
set(absl_strings_COMPILE_OPTIONS_C_RELEASE "")
set(absl_strings_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_strings_LIBS_RELEASE absl_strings)
set(absl_strings_SYSTEM_LIBS_RELEASE )
set(absl_strings_FRAMEWORK_DIRS_RELEASE )
set(absl_strings_FRAMEWORKS_RELEASE )
set(absl_strings_BUILD_MODULES_PATHS_RELEASE )
set(absl_strings_DEPENDENCIES_RELEASE absl::strings_internal absl::base absl::bits absl::config absl::core_headers absl::endian absl::int128 absl::memory absl::raw_logging_internal absl::throw_delegate absl::type_traits)
set(absl_strings_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT strings FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_strings_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_strings_FRAMEWORKS_FOUND_RELEASE "${absl_strings_FRAMEWORKS_RELEASE}" "${absl_strings_FRAMEWORK_DIRS_RELEASE}")

set(absl_strings_LIB_TARGETS_RELEASE "")
set(absl_strings_NOT_USED_RELEASE "")
set(absl_strings_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_strings_FRAMEWORKS_FOUND_RELEASE} ${absl_strings_SYSTEM_LIBS_RELEASE} ${absl_strings_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_strings_LIBS_RELEASE}"
                              "${absl_strings_LIB_DIRS_RELEASE}"
                              "${absl_strings_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_strings_NOT_USED_RELEASE
                              absl_strings_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_strings")

set(absl_strings_LINK_LIBS_RELEASE ${absl_strings_LIB_TARGETS_RELEASE} ${absl_strings_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT time VARIABLES #############################################

set(absl_time_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_time_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_time_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_time_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_time_RES_DIRS_RELEASE )
set(absl_time_DEFINITIONS_RELEASE )
set(absl_time_COMPILE_DEFINITIONS_RELEASE )
set(absl_time_COMPILE_OPTIONS_C_RELEASE "")
set(absl_time_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_time_LIBS_RELEASE absl_time)
set(absl_time_SYSTEM_LIBS_RELEASE )
set(absl_time_FRAMEWORK_DIRS_RELEASE )
set(absl_time_FRAMEWORKS_RELEASE )
set(absl_time_BUILD_MODULES_PATHS_RELEASE )
set(absl_time_DEPENDENCIES_RELEASE absl::base absl::civil_time absl::core_headers absl::int128 absl::raw_logging_internal absl::strings absl::time_zone)
set(absl_time_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT time FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_time_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_time_FRAMEWORKS_FOUND_RELEASE "${absl_time_FRAMEWORKS_RELEASE}" "${absl_time_FRAMEWORK_DIRS_RELEASE}")

set(absl_time_LIB_TARGETS_RELEASE "")
set(absl_time_NOT_USED_RELEASE "")
set(absl_time_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_time_FRAMEWORKS_FOUND_RELEASE} ${absl_time_SYSTEM_LIBS_RELEASE} ${absl_time_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_time_LIBS_RELEASE}"
                              "${absl_time_LIB_DIRS_RELEASE}"
                              "${absl_time_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_time_NOT_USED_RELEASE
                              absl_time_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_time")

set(absl_time_LINK_LIBS_RELEASE ${absl_time_LIB_TARGETS_RELEASE} ${absl_time_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT kernel_timeout_internal VARIABLES #############################################

set(absl_kernel_timeout_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_kernel_timeout_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_kernel_timeout_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_kernel_timeout_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_kernel_timeout_internal_RES_DIRS_RELEASE )
set(absl_kernel_timeout_internal_DEFINITIONS_RELEASE )
set(absl_kernel_timeout_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_kernel_timeout_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_kernel_timeout_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_kernel_timeout_internal_LIBS_RELEASE )
set(absl_kernel_timeout_internal_SYSTEM_LIBS_RELEASE )
set(absl_kernel_timeout_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_kernel_timeout_internal_FRAMEWORKS_RELEASE )
set(absl_kernel_timeout_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_kernel_timeout_internal_DEPENDENCIES_RELEASE absl::core_headers absl::raw_logging_internal absl::time)
set(absl_kernel_timeout_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT kernel_timeout_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_kernel_timeout_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_kernel_timeout_internal_FRAMEWORKS_FOUND_RELEASE "${absl_kernel_timeout_internal_FRAMEWORKS_RELEASE}" "${absl_kernel_timeout_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_kernel_timeout_internal_LIB_TARGETS_RELEASE "")
set(absl_kernel_timeout_internal_NOT_USED_RELEASE "")
set(absl_kernel_timeout_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_kernel_timeout_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_kernel_timeout_internal_SYSTEM_LIBS_RELEASE} ${absl_kernel_timeout_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_kernel_timeout_internal_LIBS_RELEASE}"
                              "${absl_kernel_timeout_internal_LIB_DIRS_RELEASE}"
                              "${absl_kernel_timeout_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_kernel_timeout_internal_NOT_USED_RELEASE
                              absl_kernel_timeout_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_kernel_timeout_internal")

set(absl_kernel_timeout_internal_LINK_LIBS_RELEASE ${absl_kernel_timeout_internal_LIB_TARGETS_RELEASE} ${absl_kernel_timeout_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT malloc_internal VARIABLES #############################################

set(absl_malloc_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_malloc_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_malloc_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_malloc_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_malloc_internal_RES_DIRS_RELEASE )
set(absl_malloc_internal_DEFINITIONS_RELEASE )
set(absl_malloc_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_malloc_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_malloc_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_malloc_internal_LIBS_RELEASE absl_malloc_internal)
set(absl_malloc_internal_SYSTEM_LIBS_RELEASE )
set(absl_malloc_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_malloc_internal_FRAMEWORKS_RELEASE )
set(absl_malloc_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_malloc_internal_DEPENDENCIES_RELEASE absl::base absl::base_internal absl::config absl::core_headers absl::dynamic_annotations absl::raw_logging_internal)
set(absl_malloc_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT malloc_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_malloc_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_malloc_internal_FRAMEWORKS_FOUND_RELEASE "${absl_malloc_internal_FRAMEWORKS_RELEASE}" "${absl_malloc_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_malloc_internal_LIB_TARGETS_RELEASE "")
set(absl_malloc_internal_NOT_USED_RELEASE "")
set(absl_malloc_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_malloc_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_malloc_internal_SYSTEM_LIBS_RELEASE} ${absl_malloc_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_malloc_internal_LIBS_RELEASE}"
                              "${absl_malloc_internal_LIB_DIRS_RELEASE}"
                              "${absl_malloc_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_malloc_internal_NOT_USED_RELEASE
                              absl_malloc_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_malloc_internal")

set(absl_malloc_internal_LINK_LIBS_RELEASE ${absl_malloc_internal_LIB_TARGETS_RELEASE} ${absl_malloc_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT graphcycles_internal VARIABLES #############################################

set(absl_graphcycles_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_graphcycles_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_graphcycles_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_graphcycles_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_graphcycles_internal_RES_DIRS_RELEASE )
set(absl_graphcycles_internal_DEFINITIONS_RELEASE )
set(absl_graphcycles_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_graphcycles_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_graphcycles_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_graphcycles_internal_LIBS_RELEASE absl_graphcycles_internal)
set(absl_graphcycles_internal_SYSTEM_LIBS_RELEASE )
set(absl_graphcycles_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_graphcycles_internal_FRAMEWORKS_RELEASE )
set(absl_graphcycles_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_graphcycles_internal_DEPENDENCIES_RELEASE absl::base absl::base_internal absl::config absl::core_headers absl::malloc_internal absl::raw_logging_internal)
set(absl_graphcycles_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT graphcycles_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_graphcycles_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_graphcycles_internal_FRAMEWORKS_FOUND_RELEASE "${absl_graphcycles_internal_FRAMEWORKS_RELEASE}" "${absl_graphcycles_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_graphcycles_internal_LIB_TARGETS_RELEASE "")
set(absl_graphcycles_internal_NOT_USED_RELEASE "")
set(absl_graphcycles_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_graphcycles_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_graphcycles_internal_SYSTEM_LIBS_RELEASE} ${absl_graphcycles_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_graphcycles_internal_LIBS_RELEASE}"
                              "${absl_graphcycles_internal_LIB_DIRS_RELEASE}"
                              "${absl_graphcycles_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_graphcycles_internal_NOT_USED_RELEASE
                              absl_graphcycles_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_graphcycles_internal")

set(absl_graphcycles_internal_LINK_LIBS_RELEASE ${absl_graphcycles_internal_LIB_TARGETS_RELEASE} ${absl_graphcycles_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT demangle_internal VARIABLES #############################################

set(absl_demangle_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_demangle_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_demangle_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_demangle_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_demangle_internal_RES_DIRS_RELEASE )
set(absl_demangle_internal_DEFINITIONS_RELEASE )
set(absl_demangle_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_demangle_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_demangle_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_demangle_internal_LIBS_RELEASE absl_demangle_internal)
set(absl_demangle_internal_SYSTEM_LIBS_RELEASE )
set(absl_demangle_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_demangle_internal_FRAMEWORKS_RELEASE )
set(absl_demangle_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_demangle_internal_DEPENDENCIES_RELEASE absl::base absl::core_headers)
set(absl_demangle_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT demangle_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_demangle_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_demangle_internal_FRAMEWORKS_FOUND_RELEASE "${absl_demangle_internal_FRAMEWORKS_RELEASE}" "${absl_demangle_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_demangle_internal_LIB_TARGETS_RELEASE "")
set(absl_demangle_internal_NOT_USED_RELEASE "")
set(absl_demangle_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_demangle_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_demangle_internal_SYSTEM_LIBS_RELEASE} ${absl_demangle_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_demangle_internal_LIBS_RELEASE}"
                              "${absl_demangle_internal_LIB_DIRS_RELEASE}"
                              "${absl_demangle_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_demangle_internal_NOT_USED_RELEASE
                              absl_demangle_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_demangle_internal")

set(absl_demangle_internal_LINK_LIBS_RELEASE ${absl_demangle_internal_LIB_TARGETS_RELEASE} ${absl_demangle_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT debugging_internal VARIABLES #############################################

set(absl_debugging_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_debugging_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_debugging_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_debugging_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_debugging_internal_RES_DIRS_RELEASE )
set(absl_debugging_internal_DEFINITIONS_RELEASE )
set(absl_debugging_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_debugging_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_debugging_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_debugging_internal_LIBS_RELEASE absl_debugging_internal)
set(absl_debugging_internal_SYSTEM_LIBS_RELEASE )
set(absl_debugging_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_debugging_internal_FRAMEWORKS_RELEASE )
set(absl_debugging_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_debugging_internal_DEPENDENCIES_RELEASE absl::core_headers absl::config absl::dynamic_annotations absl::errno_saver absl::raw_logging_internal)
set(absl_debugging_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT debugging_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_debugging_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_debugging_internal_FRAMEWORKS_FOUND_RELEASE "${absl_debugging_internal_FRAMEWORKS_RELEASE}" "${absl_debugging_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_debugging_internal_LIB_TARGETS_RELEASE "")
set(absl_debugging_internal_NOT_USED_RELEASE "")
set(absl_debugging_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_debugging_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_debugging_internal_SYSTEM_LIBS_RELEASE} ${absl_debugging_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_debugging_internal_LIBS_RELEASE}"
                              "${absl_debugging_internal_LIB_DIRS_RELEASE}"
                              "${absl_debugging_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_debugging_internal_NOT_USED_RELEASE
                              absl_debugging_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_debugging_internal")

set(absl_debugging_internal_LINK_LIBS_RELEASE ${absl_debugging_internal_LIB_TARGETS_RELEASE} ${absl_debugging_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT symbolize VARIABLES #############################################

set(absl_symbolize_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_symbolize_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_symbolize_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_symbolize_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_symbolize_RES_DIRS_RELEASE )
set(absl_symbolize_DEFINITIONS_RELEASE )
set(absl_symbolize_COMPILE_DEFINITIONS_RELEASE )
set(absl_symbolize_COMPILE_OPTIONS_C_RELEASE "")
set(absl_symbolize_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_symbolize_LIBS_RELEASE absl_symbolize)
set(absl_symbolize_SYSTEM_LIBS_RELEASE )
set(absl_symbolize_FRAMEWORK_DIRS_RELEASE )
set(absl_symbolize_FRAMEWORKS_RELEASE )
set(absl_symbolize_BUILD_MODULES_PATHS_RELEASE )
set(absl_symbolize_DEPENDENCIES_RELEASE absl::debugging_internal absl::demangle_internal absl::base absl::config absl::core_headers absl::dynamic_annotations absl::malloc_internal absl::raw_logging_internal absl::strings)
set(absl_symbolize_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT symbolize FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_symbolize_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_symbolize_FRAMEWORKS_FOUND_RELEASE "${absl_symbolize_FRAMEWORKS_RELEASE}" "${absl_symbolize_FRAMEWORK_DIRS_RELEASE}")

set(absl_symbolize_LIB_TARGETS_RELEASE "")
set(absl_symbolize_NOT_USED_RELEASE "")
set(absl_symbolize_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_symbolize_FRAMEWORKS_FOUND_RELEASE} ${absl_symbolize_SYSTEM_LIBS_RELEASE} ${absl_symbolize_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_symbolize_LIBS_RELEASE}"
                              "${absl_symbolize_LIB_DIRS_RELEASE}"
                              "${absl_symbolize_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_symbolize_NOT_USED_RELEASE
                              absl_symbolize_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_symbolize")

set(absl_symbolize_LINK_LIBS_RELEASE ${absl_symbolize_LIB_TARGETS_RELEASE} ${absl_symbolize_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT stacktrace VARIABLES #############################################

set(absl_stacktrace_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_stacktrace_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_stacktrace_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_stacktrace_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_stacktrace_RES_DIRS_RELEASE )
set(absl_stacktrace_DEFINITIONS_RELEASE )
set(absl_stacktrace_COMPILE_DEFINITIONS_RELEASE )
set(absl_stacktrace_COMPILE_OPTIONS_C_RELEASE "")
set(absl_stacktrace_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_stacktrace_LIBS_RELEASE absl_stacktrace)
set(absl_stacktrace_SYSTEM_LIBS_RELEASE )
set(absl_stacktrace_FRAMEWORK_DIRS_RELEASE )
set(absl_stacktrace_FRAMEWORKS_RELEASE )
set(absl_stacktrace_BUILD_MODULES_PATHS_RELEASE )
set(absl_stacktrace_DEPENDENCIES_RELEASE absl::debugging_internal absl::config absl::core_headers absl::raw_logging_internal)
set(absl_stacktrace_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT stacktrace FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_stacktrace_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_stacktrace_FRAMEWORKS_FOUND_RELEASE "${absl_stacktrace_FRAMEWORKS_RELEASE}" "${absl_stacktrace_FRAMEWORK_DIRS_RELEASE}")

set(absl_stacktrace_LIB_TARGETS_RELEASE "")
set(absl_stacktrace_NOT_USED_RELEASE "")
set(absl_stacktrace_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_stacktrace_FRAMEWORKS_FOUND_RELEASE} ${absl_stacktrace_SYSTEM_LIBS_RELEASE} ${absl_stacktrace_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_stacktrace_LIBS_RELEASE}"
                              "${absl_stacktrace_LIB_DIRS_RELEASE}"
                              "${absl_stacktrace_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_stacktrace_NOT_USED_RELEASE
                              absl_stacktrace_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_stacktrace")

set(absl_stacktrace_LINK_LIBS_RELEASE ${absl_stacktrace_LIB_TARGETS_RELEASE} ${absl_stacktrace_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT synchronization VARIABLES #############################################

set(absl_synchronization_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_synchronization_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_synchronization_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_synchronization_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_synchronization_RES_DIRS_RELEASE )
set(absl_synchronization_DEFINITIONS_RELEASE )
set(absl_synchronization_COMPILE_DEFINITIONS_RELEASE )
set(absl_synchronization_COMPILE_OPTIONS_C_RELEASE "")
set(absl_synchronization_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_synchronization_LIBS_RELEASE absl_synchronization)
set(absl_synchronization_SYSTEM_LIBS_RELEASE )
set(absl_synchronization_FRAMEWORK_DIRS_RELEASE )
set(absl_synchronization_FRAMEWORKS_RELEASE )
set(absl_synchronization_BUILD_MODULES_PATHS_RELEASE )
set(absl_synchronization_DEPENDENCIES_RELEASE absl::graphcycles_internal absl::kernel_timeout_internal absl::atomic_hook absl::base absl::base_internal absl::config absl::core_headers absl::dynamic_annotations absl::malloc_internal absl::raw_logging_internal absl::stacktrace absl::symbolize absl::time)
set(absl_synchronization_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT synchronization FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_synchronization_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_synchronization_FRAMEWORKS_FOUND_RELEASE "${absl_synchronization_FRAMEWORKS_RELEASE}" "${absl_synchronization_FRAMEWORK_DIRS_RELEASE}")

set(absl_synchronization_LIB_TARGETS_RELEASE "")
set(absl_synchronization_NOT_USED_RELEASE "")
set(absl_synchronization_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_synchronization_FRAMEWORKS_FOUND_RELEASE} ${absl_synchronization_SYSTEM_LIBS_RELEASE} ${absl_synchronization_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_synchronization_LIBS_RELEASE}"
                              "${absl_synchronization_LIB_DIRS_RELEASE}"
                              "${absl_synchronization_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_synchronization_NOT_USED_RELEASE
                              absl_synchronization_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_synchronization")

set(absl_synchronization_LINK_LIBS_RELEASE ${absl_synchronization_LIB_TARGETS_RELEASE} ${absl_synchronization_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_handle VARIABLES #############################################

set(absl_cordz_handle_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_handle_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_handle_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_handle_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_handle_RES_DIRS_RELEASE )
set(absl_cordz_handle_DEFINITIONS_RELEASE )
set(absl_cordz_handle_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_handle_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_handle_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_handle_LIBS_RELEASE absl_cordz_handle)
set(absl_cordz_handle_SYSTEM_LIBS_RELEASE )
set(absl_cordz_handle_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_handle_FRAMEWORKS_RELEASE )
set(absl_cordz_handle_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_handle_DEPENDENCIES_RELEASE absl::base absl::config absl::raw_logging_internal absl::synchronization)
set(absl_cordz_handle_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_handle FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_handle_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_handle_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_handle_FRAMEWORKS_RELEASE}" "${absl_cordz_handle_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_handle_LIB_TARGETS_RELEASE "")
set(absl_cordz_handle_NOT_USED_RELEASE "")
set(absl_cordz_handle_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_handle_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_handle_SYSTEM_LIBS_RELEASE} ${absl_cordz_handle_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_handle_LIBS_RELEASE}"
                              "${absl_cordz_handle_LIB_DIRS_RELEASE}"
                              "${absl_cordz_handle_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_handle_NOT_USED_RELEASE
                              absl_cordz_handle_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_handle")

set(absl_cordz_handle_LINK_LIBS_RELEASE ${absl_cordz_handle_LIB_TARGETS_RELEASE} ${absl_cordz_handle_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_update_tracker VARIABLES #############################################

set(absl_cordz_update_tracker_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_update_tracker_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_update_tracker_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_update_tracker_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_update_tracker_RES_DIRS_RELEASE )
set(absl_cordz_update_tracker_DEFINITIONS_RELEASE )
set(absl_cordz_update_tracker_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_update_tracker_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_update_tracker_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_update_tracker_LIBS_RELEASE )
set(absl_cordz_update_tracker_SYSTEM_LIBS_RELEASE )
set(absl_cordz_update_tracker_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_update_tracker_FRAMEWORKS_RELEASE )
set(absl_cordz_update_tracker_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_update_tracker_DEPENDENCIES_RELEASE absl::config)
set(absl_cordz_update_tracker_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_update_tracker FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_update_tracker_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_update_tracker_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_update_tracker_FRAMEWORKS_RELEASE}" "${absl_cordz_update_tracker_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_update_tracker_LIB_TARGETS_RELEASE "")
set(absl_cordz_update_tracker_NOT_USED_RELEASE "")
set(absl_cordz_update_tracker_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_update_tracker_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_update_tracker_SYSTEM_LIBS_RELEASE} ${absl_cordz_update_tracker_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_update_tracker_LIBS_RELEASE}"
                              "${absl_cordz_update_tracker_LIB_DIRS_RELEASE}"
                              "${absl_cordz_update_tracker_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_update_tracker_NOT_USED_RELEASE
                              absl_cordz_update_tracker_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_update_tracker")

set(absl_cordz_update_tracker_LINK_LIBS_RELEASE ${absl_cordz_update_tracker_LIB_TARGETS_RELEASE} ${absl_cordz_update_tracker_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_statistics VARIABLES #############################################

set(absl_cordz_statistics_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_statistics_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_statistics_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_statistics_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_statistics_RES_DIRS_RELEASE )
set(absl_cordz_statistics_DEFINITIONS_RELEASE )
set(absl_cordz_statistics_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_statistics_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_statistics_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_statistics_LIBS_RELEASE )
set(absl_cordz_statistics_SYSTEM_LIBS_RELEASE )
set(absl_cordz_statistics_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_statistics_FRAMEWORKS_RELEASE )
set(absl_cordz_statistics_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_statistics_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::cordz_update_tracker absl::synchronization)
set(absl_cordz_statistics_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_statistics FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_statistics_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_statistics_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_statistics_FRAMEWORKS_RELEASE}" "${absl_cordz_statistics_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_statistics_LIB_TARGETS_RELEASE "")
set(absl_cordz_statistics_NOT_USED_RELEASE "")
set(absl_cordz_statistics_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_statistics_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_statistics_SYSTEM_LIBS_RELEASE} ${absl_cordz_statistics_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_statistics_LIBS_RELEASE}"
                              "${absl_cordz_statistics_LIB_DIRS_RELEASE}"
                              "${absl_cordz_statistics_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_statistics_NOT_USED_RELEASE
                              absl_cordz_statistics_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_statistics")

set(absl_cordz_statistics_LINK_LIBS_RELEASE ${absl_cordz_statistics_LIB_TARGETS_RELEASE} ${absl_cordz_statistics_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT exponential_biased VARIABLES #############################################

set(absl_exponential_biased_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_exponential_biased_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_exponential_biased_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_exponential_biased_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_exponential_biased_RES_DIRS_RELEASE )
set(absl_exponential_biased_DEFINITIONS_RELEASE )
set(absl_exponential_biased_COMPILE_DEFINITIONS_RELEASE )
set(absl_exponential_biased_COMPILE_OPTIONS_C_RELEASE "")
set(absl_exponential_biased_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_exponential_biased_LIBS_RELEASE absl_exponential_biased)
set(absl_exponential_biased_SYSTEM_LIBS_RELEASE )
set(absl_exponential_biased_FRAMEWORK_DIRS_RELEASE )
set(absl_exponential_biased_FRAMEWORKS_RELEASE )
set(absl_exponential_biased_BUILD_MODULES_PATHS_RELEASE )
set(absl_exponential_biased_DEPENDENCIES_RELEASE absl::config absl::core_headers)
set(absl_exponential_biased_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT exponential_biased FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_exponential_biased_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_exponential_biased_FRAMEWORKS_FOUND_RELEASE "${absl_exponential_biased_FRAMEWORKS_RELEASE}" "${absl_exponential_biased_FRAMEWORK_DIRS_RELEASE}")

set(absl_exponential_biased_LIB_TARGETS_RELEASE "")
set(absl_exponential_biased_NOT_USED_RELEASE "")
set(absl_exponential_biased_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_exponential_biased_FRAMEWORKS_FOUND_RELEASE} ${absl_exponential_biased_SYSTEM_LIBS_RELEASE} ${absl_exponential_biased_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_exponential_biased_LIBS_RELEASE}"
                              "${absl_exponential_biased_LIB_DIRS_RELEASE}"
                              "${absl_exponential_biased_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_exponential_biased_NOT_USED_RELEASE
                              absl_exponential_biased_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_exponential_biased")

set(absl_exponential_biased_LINK_LIBS_RELEASE ${absl_exponential_biased_LIB_TARGETS_RELEASE} ${absl_exponential_biased_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_functions VARIABLES #############################################

set(absl_cordz_functions_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_functions_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_functions_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_functions_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_functions_RES_DIRS_RELEASE )
set(absl_cordz_functions_DEFINITIONS_RELEASE )
set(absl_cordz_functions_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_functions_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_functions_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_functions_LIBS_RELEASE absl_cordz_functions)
set(absl_cordz_functions_SYSTEM_LIBS_RELEASE )
set(absl_cordz_functions_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_functions_FRAMEWORKS_RELEASE )
set(absl_cordz_functions_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_functions_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::exponential_biased absl::raw_logging_internal)
set(absl_cordz_functions_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_functions FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_functions_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_functions_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_functions_FRAMEWORKS_RELEASE}" "${absl_cordz_functions_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_functions_LIB_TARGETS_RELEASE "")
set(absl_cordz_functions_NOT_USED_RELEASE "")
set(absl_cordz_functions_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_functions_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_functions_SYSTEM_LIBS_RELEASE} ${absl_cordz_functions_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_functions_LIBS_RELEASE}"
                              "${absl_cordz_functions_LIB_DIRS_RELEASE}"
                              "${absl_cordz_functions_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_functions_NOT_USED_RELEASE
                              absl_cordz_functions_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_functions")

set(absl_cordz_functions_LINK_LIBS_RELEASE ${absl_cordz_functions_LIB_TARGETS_RELEASE} ${absl_cordz_functions_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT non_temporal_arm_intrinsics VARIABLES #############################################

set(absl_non_temporal_arm_intrinsics_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_non_temporal_arm_intrinsics_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_non_temporal_arm_intrinsics_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_non_temporal_arm_intrinsics_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_non_temporal_arm_intrinsics_RES_DIRS_RELEASE )
set(absl_non_temporal_arm_intrinsics_DEFINITIONS_RELEASE )
set(absl_non_temporal_arm_intrinsics_COMPILE_DEFINITIONS_RELEASE )
set(absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_C_RELEASE "")
set(absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_non_temporal_arm_intrinsics_LIBS_RELEASE )
set(absl_non_temporal_arm_intrinsics_SYSTEM_LIBS_RELEASE )
set(absl_non_temporal_arm_intrinsics_FRAMEWORK_DIRS_RELEASE )
set(absl_non_temporal_arm_intrinsics_FRAMEWORKS_RELEASE )
set(absl_non_temporal_arm_intrinsics_BUILD_MODULES_PATHS_RELEASE )
set(absl_non_temporal_arm_intrinsics_DEPENDENCIES_RELEASE absl::config)
set(absl_non_temporal_arm_intrinsics_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT non_temporal_arm_intrinsics FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_non_temporal_arm_intrinsics_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_non_temporal_arm_intrinsics_FRAMEWORKS_FOUND_RELEASE "${absl_non_temporal_arm_intrinsics_FRAMEWORKS_RELEASE}" "${absl_non_temporal_arm_intrinsics_FRAMEWORK_DIRS_RELEASE}")

set(absl_non_temporal_arm_intrinsics_LIB_TARGETS_RELEASE "")
set(absl_non_temporal_arm_intrinsics_NOT_USED_RELEASE "")
set(absl_non_temporal_arm_intrinsics_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_non_temporal_arm_intrinsics_FRAMEWORKS_FOUND_RELEASE} ${absl_non_temporal_arm_intrinsics_SYSTEM_LIBS_RELEASE} ${absl_non_temporal_arm_intrinsics_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_non_temporal_arm_intrinsics_LIBS_RELEASE}"
                              "${absl_non_temporal_arm_intrinsics_LIB_DIRS_RELEASE}"
                              "${absl_non_temporal_arm_intrinsics_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_non_temporal_arm_intrinsics_NOT_USED_RELEASE
                              absl_non_temporal_arm_intrinsics_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_non_temporal_arm_intrinsics")

set(absl_non_temporal_arm_intrinsics_LINK_LIBS_RELEASE ${absl_non_temporal_arm_intrinsics_LIB_TARGETS_RELEASE} ${absl_non_temporal_arm_intrinsics_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT non_temporal_memcpy VARIABLES #############################################

set(absl_non_temporal_memcpy_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_non_temporal_memcpy_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_non_temporal_memcpy_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_non_temporal_memcpy_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_non_temporal_memcpy_RES_DIRS_RELEASE )
set(absl_non_temporal_memcpy_DEFINITIONS_RELEASE )
set(absl_non_temporal_memcpy_COMPILE_DEFINITIONS_RELEASE )
set(absl_non_temporal_memcpy_COMPILE_OPTIONS_C_RELEASE "")
set(absl_non_temporal_memcpy_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_non_temporal_memcpy_LIBS_RELEASE )
set(absl_non_temporal_memcpy_SYSTEM_LIBS_RELEASE )
set(absl_non_temporal_memcpy_FRAMEWORK_DIRS_RELEASE )
set(absl_non_temporal_memcpy_FRAMEWORKS_RELEASE )
set(absl_non_temporal_memcpy_BUILD_MODULES_PATHS_RELEASE )
set(absl_non_temporal_memcpy_DEPENDENCIES_RELEASE absl::non_temporal_arm_intrinsics absl::config absl::core_headers)
set(absl_non_temporal_memcpy_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT non_temporal_memcpy FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_non_temporal_memcpy_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_non_temporal_memcpy_FRAMEWORKS_FOUND_RELEASE "${absl_non_temporal_memcpy_FRAMEWORKS_RELEASE}" "${absl_non_temporal_memcpy_FRAMEWORK_DIRS_RELEASE}")

set(absl_non_temporal_memcpy_LIB_TARGETS_RELEASE "")
set(absl_non_temporal_memcpy_NOT_USED_RELEASE "")
set(absl_non_temporal_memcpy_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_non_temporal_memcpy_FRAMEWORKS_FOUND_RELEASE} ${absl_non_temporal_memcpy_SYSTEM_LIBS_RELEASE} ${absl_non_temporal_memcpy_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_non_temporal_memcpy_LIBS_RELEASE}"
                              "${absl_non_temporal_memcpy_LIB_DIRS_RELEASE}"
                              "${absl_non_temporal_memcpy_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_non_temporal_memcpy_NOT_USED_RELEASE
                              absl_non_temporal_memcpy_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_non_temporal_memcpy")

set(absl_non_temporal_memcpy_LINK_LIBS_RELEASE ${absl_non_temporal_memcpy_LIB_TARGETS_RELEASE} ${absl_non_temporal_memcpy_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT crc_cpu_detect VARIABLES #############################################

set(absl_crc_cpu_detect_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_cpu_detect_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_cpu_detect_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_cpu_detect_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_crc_cpu_detect_RES_DIRS_RELEASE )
set(absl_crc_cpu_detect_DEFINITIONS_RELEASE )
set(absl_crc_cpu_detect_COMPILE_DEFINITIONS_RELEASE )
set(absl_crc_cpu_detect_COMPILE_OPTIONS_C_RELEASE "")
set(absl_crc_cpu_detect_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_crc_cpu_detect_LIBS_RELEASE absl_crc_cpu_detect)
set(absl_crc_cpu_detect_SYSTEM_LIBS_RELEASE )
set(absl_crc_cpu_detect_FRAMEWORK_DIRS_RELEASE )
set(absl_crc_cpu_detect_FRAMEWORKS_RELEASE )
set(absl_crc_cpu_detect_BUILD_MODULES_PATHS_RELEASE )
set(absl_crc_cpu_detect_DEPENDENCIES_RELEASE absl::base absl::config)
set(absl_crc_cpu_detect_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT crc_cpu_detect FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_crc_cpu_detect_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_crc_cpu_detect_FRAMEWORKS_FOUND_RELEASE "${absl_crc_cpu_detect_FRAMEWORKS_RELEASE}" "${absl_crc_cpu_detect_FRAMEWORK_DIRS_RELEASE}")

set(absl_crc_cpu_detect_LIB_TARGETS_RELEASE "")
set(absl_crc_cpu_detect_NOT_USED_RELEASE "")
set(absl_crc_cpu_detect_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_crc_cpu_detect_FRAMEWORKS_FOUND_RELEASE} ${absl_crc_cpu_detect_SYSTEM_LIBS_RELEASE} ${absl_crc_cpu_detect_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_crc_cpu_detect_LIBS_RELEASE}"
                              "${absl_crc_cpu_detect_LIB_DIRS_RELEASE}"
                              "${absl_crc_cpu_detect_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_crc_cpu_detect_NOT_USED_RELEASE
                              absl_crc_cpu_detect_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_crc_cpu_detect")

set(absl_crc_cpu_detect_LINK_LIBS_RELEASE ${absl_crc_cpu_detect_LIB_TARGETS_RELEASE} ${absl_crc_cpu_detect_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT prefetch VARIABLES #############################################

set(absl_prefetch_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_prefetch_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_prefetch_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_prefetch_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_prefetch_RES_DIRS_RELEASE )
set(absl_prefetch_DEFINITIONS_RELEASE )
set(absl_prefetch_COMPILE_DEFINITIONS_RELEASE )
set(absl_prefetch_COMPILE_OPTIONS_C_RELEASE "")
set(absl_prefetch_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_prefetch_LIBS_RELEASE )
set(absl_prefetch_SYSTEM_LIBS_RELEASE )
set(absl_prefetch_FRAMEWORK_DIRS_RELEASE )
set(absl_prefetch_FRAMEWORKS_RELEASE )
set(absl_prefetch_BUILD_MODULES_PATHS_RELEASE )
set(absl_prefetch_DEPENDENCIES_RELEASE absl::config)
set(absl_prefetch_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT prefetch FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_prefetch_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_prefetch_FRAMEWORKS_FOUND_RELEASE "${absl_prefetch_FRAMEWORKS_RELEASE}" "${absl_prefetch_FRAMEWORK_DIRS_RELEASE}")

set(absl_prefetch_LIB_TARGETS_RELEASE "")
set(absl_prefetch_NOT_USED_RELEASE "")
set(absl_prefetch_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_prefetch_FRAMEWORKS_FOUND_RELEASE} ${absl_prefetch_SYSTEM_LIBS_RELEASE} ${absl_prefetch_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_prefetch_LIBS_RELEASE}"
                              "${absl_prefetch_LIB_DIRS_RELEASE}"
                              "${absl_prefetch_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_prefetch_NOT_USED_RELEASE
                              absl_prefetch_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_prefetch")

set(absl_prefetch_LINK_LIBS_RELEASE ${absl_prefetch_LIB_TARGETS_RELEASE} ${absl_prefetch_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT crc_internal VARIABLES #############################################

set(absl_crc_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_crc_internal_RES_DIRS_RELEASE )
set(absl_crc_internal_DEFINITIONS_RELEASE )
set(absl_crc_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_crc_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_crc_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_crc_internal_LIBS_RELEASE absl_crc_internal)
set(absl_crc_internal_SYSTEM_LIBS_RELEASE )
set(absl_crc_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_crc_internal_FRAMEWORKS_RELEASE )
set(absl_crc_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_crc_internal_DEPENDENCIES_RELEASE absl::crc_cpu_detect absl::base absl::config absl::core_headers absl::dynamic_annotations absl::endian absl::prefetch absl::raw_logging_internal absl::memory absl::bits)
set(absl_crc_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT crc_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_crc_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_crc_internal_FRAMEWORKS_FOUND_RELEASE "${absl_crc_internal_FRAMEWORKS_RELEASE}" "${absl_crc_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_crc_internal_LIB_TARGETS_RELEASE "")
set(absl_crc_internal_NOT_USED_RELEASE "")
set(absl_crc_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_crc_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_crc_internal_SYSTEM_LIBS_RELEASE} ${absl_crc_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_crc_internal_LIBS_RELEASE}"
                              "${absl_crc_internal_LIB_DIRS_RELEASE}"
                              "${absl_crc_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_crc_internal_NOT_USED_RELEASE
                              absl_crc_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_crc_internal")

set(absl_crc_internal_LINK_LIBS_RELEASE ${absl_crc_internal_LIB_TARGETS_RELEASE} ${absl_crc_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT crc32c VARIABLES #############################################

set(absl_crc32c_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc32c_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc32c_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc32c_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_crc32c_RES_DIRS_RELEASE )
set(absl_crc32c_DEFINITIONS_RELEASE )
set(absl_crc32c_COMPILE_DEFINITIONS_RELEASE )
set(absl_crc32c_COMPILE_OPTIONS_C_RELEASE "")
set(absl_crc32c_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_crc32c_LIBS_RELEASE absl_crc32c)
set(absl_crc32c_SYSTEM_LIBS_RELEASE )
set(absl_crc32c_FRAMEWORK_DIRS_RELEASE )
set(absl_crc32c_FRAMEWORKS_RELEASE )
set(absl_crc32c_BUILD_MODULES_PATHS_RELEASE )
set(absl_crc32c_DEPENDENCIES_RELEASE absl::crc_cpu_detect absl::crc_internal absl::non_temporal_memcpy absl::config absl::core_headers absl::dynamic_annotations absl::endian absl::prefetch absl::strings)
set(absl_crc32c_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT crc32c FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_crc32c_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_crc32c_FRAMEWORKS_FOUND_RELEASE "${absl_crc32c_FRAMEWORKS_RELEASE}" "${absl_crc32c_FRAMEWORK_DIRS_RELEASE}")

set(absl_crc32c_LIB_TARGETS_RELEASE "")
set(absl_crc32c_NOT_USED_RELEASE "")
set(absl_crc32c_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_crc32c_FRAMEWORKS_FOUND_RELEASE} ${absl_crc32c_SYSTEM_LIBS_RELEASE} ${absl_crc32c_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_crc32c_LIBS_RELEASE}"
                              "${absl_crc32c_LIB_DIRS_RELEASE}"
                              "${absl_crc32c_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_crc32c_NOT_USED_RELEASE
                              absl_crc32c_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_crc32c")

set(absl_crc32c_LINK_LIBS_RELEASE ${absl_crc32c_LIB_TARGETS_RELEASE} ${absl_crc32c_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT crc_cord_state VARIABLES #############################################

set(absl_crc_cord_state_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_cord_state_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_cord_state_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_crc_cord_state_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_crc_cord_state_RES_DIRS_RELEASE )
set(absl_crc_cord_state_DEFINITIONS_RELEASE )
set(absl_crc_cord_state_COMPILE_DEFINITIONS_RELEASE )
set(absl_crc_cord_state_COMPILE_OPTIONS_C_RELEASE "")
set(absl_crc_cord_state_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_crc_cord_state_LIBS_RELEASE absl_crc_cord_state)
set(absl_crc_cord_state_SYSTEM_LIBS_RELEASE )
set(absl_crc_cord_state_FRAMEWORK_DIRS_RELEASE )
set(absl_crc_cord_state_FRAMEWORKS_RELEASE )
set(absl_crc_cord_state_BUILD_MODULES_PATHS_RELEASE )
set(absl_crc_cord_state_DEPENDENCIES_RELEASE absl::crc32c absl::config absl::strings)
set(absl_crc_cord_state_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT crc_cord_state FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_crc_cord_state_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_crc_cord_state_FRAMEWORKS_FOUND_RELEASE "${absl_crc_cord_state_FRAMEWORKS_RELEASE}" "${absl_crc_cord_state_FRAMEWORK_DIRS_RELEASE}")

set(absl_crc_cord_state_LIB_TARGETS_RELEASE "")
set(absl_crc_cord_state_NOT_USED_RELEASE "")
set(absl_crc_cord_state_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_crc_cord_state_FRAMEWORKS_FOUND_RELEASE} ${absl_crc_cord_state_SYSTEM_LIBS_RELEASE} ${absl_crc_cord_state_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_crc_cord_state_LIBS_RELEASE}"
                              "${absl_crc_cord_state_LIB_DIRS_RELEASE}"
                              "${absl_crc_cord_state_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_crc_cord_state_NOT_USED_RELEASE
                              absl_crc_cord_state_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_crc_cord_state")

set(absl_crc_cord_state_LINK_LIBS_RELEASE ${absl_crc_cord_state_LIB_TARGETS_RELEASE} ${absl_crc_cord_state_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT layout VARIABLES #############################################

set(absl_layout_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_layout_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_layout_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_layout_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_layout_RES_DIRS_RELEASE )
set(absl_layout_DEFINITIONS_RELEASE )
set(absl_layout_COMPILE_DEFINITIONS_RELEASE )
set(absl_layout_COMPILE_OPTIONS_C_RELEASE "")
set(absl_layout_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_layout_LIBS_RELEASE )
set(absl_layout_SYSTEM_LIBS_RELEASE )
set(absl_layout_FRAMEWORK_DIRS_RELEASE )
set(absl_layout_FRAMEWORKS_RELEASE )
set(absl_layout_BUILD_MODULES_PATHS_RELEASE )
set(absl_layout_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::meta absl::strings absl::span absl::utility)
set(absl_layout_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT layout FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_layout_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_layout_FRAMEWORKS_FOUND_RELEASE "${absl_layout_FRAMEWORKS_RELEASE}" "${absl_layout_FRAMEWORK_DIRS_RELEASE}")

set(absl_layout_LIB_TARGETS_RELEASE "")
set(absl_layout_NOT_USED_RELEASE "")
set(absl_layout_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_layout_FRAMEWORKS_FOUND_RELEASE} ${absl_layout_SYSTEM_LIBS_RELEASE} ${absl_layout_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_layout_LIBS_RELEASE}"
                              "${absl_layout_LIB_DIRS_RELEASE}"
                              "${absl_layout_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_layout_NOT_USED_RELEASE
                              absl_layout_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_layout")

set(absl_layout_LINK_LIBS_RELEASE ${absl_layout_LIB_TARGETS_RELEASE} ${absl_layout_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT container_memory VARIABLES #############################################

set(absl_container_memory_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_container_memory_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_container_memory_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_container_memory_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_container_memory_RES_DIRS_RELEASE )
set(absl_container_memory_DEFINITIONS_RELEASE )
set(absl_container_memory_COMPILE_DEFINITIONS_RELEASE )
set(absl_container_memory_COMPILE_OPTIONS_C_RELEASE "")
set(absl_container_memory_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_container_memory_LIBS_RELEASE )
set(absl_container_memory_SYSTEM_LIBS_RELEASE )
set(absl_container_memory_FRAMEWORK_DIRS_RELEASE )
set(absl_container_memory_FRAMEWORKS_RELEASE )
set(absl_container_memory_BUILD_MODULES_PATHS_RELEASE )
set(absl_container_memory_DEPENDENCIES_RELEASE absl::config absl::memory absl::type_traits absl::utility)
set(absl_container_memory_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT container_memory FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_container_memory_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_container_memory_FRAMEWORKS_FOUND_RELEASE "${absl_container_memory_FRAMEWORKS_RELEASE}" "${absl_container_memory_FRAMEWORK_DIRS_RELEASE}")

set(absl_container_memory_LIB_TARGETS_RELEASE "")
set(absl_container_memory_NOT_USED_RELEASE "")
set(absl_container_memory_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_container_memory_FRAMEWORKS_FOUND_RELEASE} ${absl_container_memory_SYSTEM_LIBS_RELEASE} ${absl_container_memory_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_container_memory_LIBS_RELEASE}"
                              "${absl_container_memory_LIB_DIRS_RELEASE}"
                              "${absl_container_memory_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_container_memory_NOT_USED_RELEASE
                              absl_container_memory_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_container_memory")

set(absl_container_memory_LINK_LIBS_RELEASE ${absl_container_memory_LIB_TARGETS_RELEASE} ${absl_container_memory_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT compressed_tuple VARIABLES #############################################

set(absl_compressed_tuple_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_compressed_tuple_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_compressed_tuple_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_compressed_tuple_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_compressed_tuple_RES_DIRS_RELEASE )
set(absl_compressed_tuple_DEFINITIONS_RELEASE )
set(absl_compressed_tuple_COMPILE_DEFINITIONS_RELEASE )
set(absl_compressed_tuple_COMPILE_OPTIONS_C_RELEASE "")
set(absl_compressed_tuple_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_compressed_tuple_LIBS_RELEASE )
set(absl_compressed_tuple_SYSTEM_LIBS_RELEASE )
set(absl_compressed_tuple_FRAMEWORK_DIRS_RELEASE )
set(absl_compressed_tuple_FRAMEWORKS_RELEASE )
set(absl_compressed_tuple_BUILD_MODULES_PATHS_RELEASE )
set(absl_compressed_tuple_DEPENDENCIES_RELEASE absl::utility)
set(absl_compressed_tuple_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT compressed_tuple FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_compressed_tuple_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_compressed_tuple_FRAMEWORKS_FOUND_RELEASE "${absl_compressed_tuple_FRAMEWORKS_RELEASE}" "${absl_compressed_tuple_FRAMEWORK_DIRS_RELEASE}")

set(absl_compressed_tuple_LIB_TARGETS_RELEASE "")
set(absl_compressed_tuple_NOT_USED_RELEASE "")
set(absl_compressed_tuple_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_compressed_tuple_FRAMEWORKS_FOUND_RELEASE} ${absl_compressed_tuple_SYSTEM_LIBS_RELEASE} ${absl_compressed_tuple_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_compressed_tuple_LIBS_RELEASE}"
                              "${absl_compressed_tuple_LIB_DIRS_RELEASE}"
                              "${absl_compressed_tuple_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_compressed_tuple_NOT_USED_RELEASE
                              absl_compressed_tuple_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_compressed_tuple")

set(absl_compressed_tuple_LINK_LIBS_RELEASE ${absl_compressed_tuple_LIB_TARGETS_RELEASE} ${absl_compressed_tuple_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT inlined_vector_internal VARIABLES #############################################

set(absl_inlined_vector_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_inlined_vector_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_inlined_vector_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_inlined_vector_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_inlined_vector_internal_RES_DIRS_RELEASE )
set(absl_inlined_vector_internal_DEFINITIONS_RELEASE )
set(absl_inlined_vector_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_inlined_vector_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_inlined_vector_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_inlined_vector_internal_LIBS_RELEASE )
set(absl_inlined_vector_internal_SYSTEM_LIBS_RELEASE )
set(absl_inlined_vector_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_inlined_vector_internal_FRAMEWORKS_RELEASE )
set(absl_inlined_vector_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_inlined_vector_internal_DEPENDENCIES_RELEASE absl::compressed_tuple absl::core_headers absl::memory absl::span absl::type_traits)
set(absl_inlined_vector_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT inlined_vector_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_inlined_vector_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_inlined_vector_internal_FRAMEWORKS_FOUND_RELEASE "${absl_inlined_vector_internal_FRAMEWORKS_RELEASE}" "${absl_inlined_vector_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_inlined_vector_internal_LIB_TARGETS_RELEASE "")
set(absl_inlined_vector_internal_NOT_USED_RELEASE "")
set(absl_inlined_vector_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_inlined_vector_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_inlined_vector_internal_SYSTEM_LIBS_RELEASE} ${absl_inlined_vector_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_inlined_vector_internal_LIBS_RELEASE}"
                              "${absl_inlined_vector_internal_LIB_DIRS_RELEASE}"
                              "${absl_inlined_vector_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_inlined_vector_internal_NOT_USED_RELEASE
                              absl_inlined_vector_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_inlined_vector_internal")

set(absl_inlined_vector_internal_LINK_LIBS_RELEASE ${absl_inlined_vector_internal_LIB_TARGETS_RELEASE} ${absl_inlined_vector_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT inlined_vector VARIABLES #############################################

set(absl_inlined_vector_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_inlined_vector_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_inlined_vector_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_inlined_vector_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_inlined_vector_RES_DIRS_RELEASE )
set(absl_inlined_vector_DEFINITIONS_RELEASE )
set(absl_inlined_vector_COMPILE_DEFINITIONS_RELEASE )
set(absl_inlined_vector_COMPILE_OPTIONS_C_RELEASE "")
set(absl_inlined_vector_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_inlined_vector_LIBS_RELEASE )
set(absl_inlined_vector_SYSTEM_LIBS_RELEASE )
set(absl_inlined_vector_FRAMEWORK_DIRS_RELEASE )
set(absl_inlined_vector_FRAMEWORKS_RELEASE )
set(absl_inlined_vector_BUILD_MODULES_PATHS_RELEASE )
set(absl_inlined_vector_DEPENDENCIES_RELEASE absl::algorithm absl::core_headers absl::inlined_vector_internal absl::throw_delegate absl::memory absl::type_traits)
set(absl_inlined_vector_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT inlined_vector FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_inlined_vector_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_inlined_vector_FRAMEWORKS_FOUND_RELEASE "${absl_inlined_vector_FRAMEWORKS_RELEASE}" "${absl_inlined_vector_FRAMEWORK_DIRS_RELEASE}")

set(absl_inlined_vector_LIB_TARGETS_RELEASE "")
set(absl_inlined_vector_NOT_USED_RELEASE "")
set(absl_inlined_vector_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_inlined_vector_FRAMEWORKS_FOUND_RELEASE} ${absl_inlined_vector_SYSTEM_LIBS_RELEASE} ${absl_inlined_vector_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_inlined_vector_LIBS_RELEASE}"
                              "${absl_inlined_vector_LIB_DIRS_RELEASE}"
                              "${absl_inlined_vector_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_inlined_vector_NOT_USED_RELEASE
                              absl_inlined_vector_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_inlined_vector")

set(absl_inlined_vector_LINK_LIBS_RELEASE ${absl_inlined_vector_LIB_TARGETS_RELEASE} ${absl_inlined_vector_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cord_internal VARIABLES #############################################

set(absl_cord_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cord_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cord_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cord_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cord_internal_RES_DIRS_RELEASE )
set(absl_cord_internal_DEFINITIONS_RELEASE )
set(absl_cord_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_cord_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cord_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cord_internal_LIBS_RELEASE absl_cord_internal)
set(absl_cord_internal_SYSTEM_LIBS_RELEASE )
set(absl_cord_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_cord_internal_FRAMEWORKS_RELEASE )
set(absl_cord_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_cord_internal_DEPENDENCIES_RELEASE absl::base_internal absl::compressed_tuple absl::config absl::container_memory absl::core_headers absl::crc_cord_state absl::endian absl::inlined_vector absl::layout absl::raw_logging_internal absl::strings absl::throw_delegate absl::type_traits)
set(absl_cord_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cord_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cord_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cord_internal_FRAMEWORKS_FOUND_RELEASE "${absl_cord_internal_FRAMEWORKS_RELEASE}" "${absl_cord_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_cord_internal_LIB_TARGETS_RELEASE "")
set(absl_cord_internal_NOT_USED_RELEASE "")
set(absl_cord_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cord_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_cord_internal_SYSTEM_LIBS_RELEASE} ${absl_cord_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cord_internal_LIBS_RELEASE}"
                              "${absl_cord_internal_LIB_DIRS_RELEASE}"
                              "${absl_cord_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cord_internal_NOT_USED_RELEASE
                              absl_cord_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cord_internal")

set(absl_cord_internal_LINK_LIBS_RELEASE ${absl_cord_internal_LIB_TARGETS_RELEASE} ${absl_cord_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_info VARIABLES #############################################

set(absl_cordz_info_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_info_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_info_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_info_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_info_RES_DIRS_RELEASE )
set(absl_cordz_info_DEFINITIONS_RELEASE )
set(absl_cordz_info_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_info_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_info_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_info_LIBS_RELEASE absl_cordz_info)
set(absl_cordz_info_SYSTEM_LIBS_RELEASE )
set(absl_cordz_info_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_info_FRAMEWORKS_RELEASE )
set(absl_cordz_info_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_info_DEPENDENCIES_RELEASE absl::base absl::config absl::cord_internal absl::cordz_functions absl::cordz_handle absl::cordz_statistics absl::cordz_update_tracker absl::core_headers absl::inlined_vector absl::span absl::raw_logging_internal absl::stacktrace absl::synchronization)
set(absl_cordz_info_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_info FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_info_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_info_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_info_FRAMEWORKS_RELEASE}" "${absl_cordz_info_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_info_LIB_TARGETS_RELEASE "")
set(absl_cordz_info_NOT_USED_RELEASE "")
set(absl_cordz_info_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_info_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_info_SYSTEM_LIBS_RELEASE} ${absl_cordz_info_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_info_LIBS_RELEASE}"
                              "${absl_cordz_info_LIB_DIRS_RELEASE}"
                              "${absl_cordz_info_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_info_NOT_USED_RELEASE
                              absl_cordz_info_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_info")

set(absl_cordz_info_LINK_LIBS_RELEASE ${absl_cordz_info_LIB_TARGETS_RELEASE} ${absl_cordz_info_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_update_scope VARIABLES #############################################

set(absl_cordz_update_scope_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_update_scope_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_update_scope_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_update_scope_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_update_scope_RES_DIRS_RELEASE )
set(absl_cordz_update_scope_DEFINITIONS_RELEASE )
set(absl_cordz_update_scope_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_update_scope_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_update_scope_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_update_scope_LIBS_RELEASE )
set(absl_cordz_update_scope_SYSTEM_LIBS_RELEASE )
set(absl_cordz_update_scope_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_update_scope_FRAMEWORKS_RELEASE )
set(absl_cordz_update_scope_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_update_scope_DEPENDENCIES_RELEASE absl::config absl::cord_internal absl::cordz_info absl::cordz_update_tracker absl::core_headers)
set(absl_cordz_update_scope_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_update_scope FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_update_scope_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_update_scope_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_update_scope_FRAMEWORKS_RELEASE}" "${absl_cordz_update_scope_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_update_scope_LIB_TARGETS_RELEASE "")
set(absl_cordz_update_scope_NOT_USED_RELEASE "")
set(absl_cordz_update_scope_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_update_scope_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_update_scope_SYSTEM_LIBS_RELEASE} ${absl_cordz_update_scope_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_update_scope_LIBS_RELEASE}"
                              "${absl_cordz_update_scope_LIB_DIRS_RELEASE}"
                              "${absl_cordz_update_scope_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_update_scope_NOT_USED_RELEASE
                              absl_cordz_update_scope_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_update_scope")

set(absl_cordz_update_scope_LINK_LIBS_RELEASE ${absl_cordz_update_scope_LIB_TARGETS_RELEASE} ${absl_cordz_update_scope_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT function_ref VARIABLES #############################################

set(absl_function_ref_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_function_ref_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_function_ref_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_function_ref_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_function_ref_RES_DIRS_RELEASE )
set(absl_function_ref_DEFINITIONS_RELEASE )
set(absl_function_ref_COMPILE_DEFINITIONS_RELEASE )
set(absl_function_ref_COMPILE_OPTIONS_C_RELEASE "")
set(absl_function_ref_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_function_ref_LIBS_RELEASE )
set(absl_function_ref_SYSTEM_LIBS_RELEASE )
set(absl_function_ref_FRAMEWORK_DIRS_RELEASE )
set(absl_function_ref_FRAMEWORKS_RELEASE )
set(absl_function_ref_BUILD_MODULES_PATHS_RELEASE )
set(absl_function_ref_DEPENDENCIES_RELEASE absl::base_internal absl::core_headers absl::meta)
set(absl_function_ref_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT function_ref FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_function_ref_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_function_ref_FRAMEWORKS_FOUND_RELEASE "${absl_function_ref_FRAMEWORKS_RELEASE}" "${absl_function_ref_FRAMEWORK_DIRS_RELEASE}")

set(absl_function_ref_LIB_TARGETS_RELEASE "")
set(absl_function_ref_NOT_USED_RELEASE "")
set(absl_function_ref_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_function_ref_FRAMEWORKS_FOUND_RELEASE} ${absl_function_ref_SYSTEM_LIBS_RELEASE} ${absl_function_ref_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_function_ref_LIBS_RELEASE}"
                              "${absl_function_ref_LIB_DIRS_RELEASE}"
                              "${absl_function_ref_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_function_ref_NOT_USED_RELEASE
                              absl_function_ref_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_function_ref")

set(absl_function_ref_LINK_LIBS_RELEASE ${absl_function_ref_LIB_TARGETS_RELEASE} ${absl_function_ref_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT fixed_array VARIABLES #############################################

set(absl_fixed_array_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_fixed_array_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_fixed_array_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_fixed_array_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_fixed_array_RES_DIRS_RELEASE )
set(absl_fixed_array_DEFINITIONS_RELEASE )
set(absl_fixed_array_COMPILE_DEFINITIONS_RELEASE )
set(absl_fixed_array_COMPILE_OPTIONS_C_RELEASE "")
set(absl_fixed_array_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_fixed_array_LIBS_RELEASE )
set(absl_fixed_array_SYSTEM_LIBS_RELEASE )
set(absl_fixed_array_FRAMEWORK_DIRS_RELEASE )
set(absl_fixed_array_FRAMEWORKS_RELEASE )
set(absl_fixed_array_BUILD_MODULES_PATHS_RELEASE )
set(absl_fixed_array_DEPENDENCIES_RELEASE absl::compressed_tuple absl::algorithm absl::config absl::core_headers absl::dynamic_annotations absl::throw_delegate absl::memory)
set(absl_fixed_array_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT fixed_array FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_fixed_array_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_fixed_array_FRAMEWORKS_FOUND_RELEASE "${absl_fixed_array_FRAMEWORKS_RELEASE}" "${absl_fixed_array_FRAMEWORK_DIRS_RELEASE}")

set(absl_fixed_array_LIB_TARGETS_RELEASE "")
set(absl_fixed_array_NOT_USED_RELEASE "")
set(absl_fixed_array_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_fixed_array_FRAMEWORKS_FOUND_RELEASE} ${absl_fixed_array_SYSTEM_LIBS_RELEASE} ${absl_fixed_array_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_fixed_array_LIBS_RELEASE}"
                              "${absl_fixed_array_LIB_DIRS_RELEASE}"
                              "${absl_fixed_array_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_fixed_array_NOT_USED_RELEASE
                              absl_fixed_array_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_fixed_array")

set(absl_fixed_array_LINK_LIBS_RELEASE ${absl_fixed_array_LIB_TARGETS_RELEASE} ${absl_fixed_array_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cord VARIABLES #############################################

set(absl_cord_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cord_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cord_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cord_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cord_RES_DIRS_RELEASE )
set(absl_cord_DEFINITIONS_RELEASE )
set(absl_cord_COMPILE_DEFINITIONS_RELEASE )
set(absl_cord_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cord_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cord_LIBS_RELEASE absl_cord)
set(absl_cord_SYSTEM_LIBS_RELEASE )
set(absl_cord_FRAMEWORK_DIRS_RELEASE )
set(absl_cord_FRAMEWORKS_RELEASE )
set(absl_cord_BUILD_MODULES_PATHS_RELEASE )
set(absl_cord_DEPENDENCIES_RELEASE absl::base absl::config absl::cord_internal absl::cordz_functions absl::cordz_info absl::cordz_update_scope absl::cordz_update_tracker absl::core_headers absl::crc_cord_state absl::endian absl::fixed_array absl::function_ref absl::inlined_vector absl::optional absl::raw_logging_internal absl::span absl::strings absl::type_traits)
set(absl_cord_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cord FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cord_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cord_FRAMEWORKS_FOUND_RELEASE "${absl_cord_FRAMEWORKS_RELEASE}" "${absl_cord_FRAMEWORK_DIRS_RELEASE}")

set(absl_cord_LIB_TARGETS_RELEASE "")
set(absl_cord_NOT_USED_RELEASE "")
set(absl_cord_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cord_FRAMEWORKS_FOUND_RELEASE} ${absl_cord_SYSTEM_LIBS_RELEASE} ${absl_cord_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cord_LIBS_RELEASE}"
                              "${absl_cord_LIB_DIRS_RELEASE}"
                              "${absl_cord_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cord_NOT_USED_RELEASE
                              absl_cord_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cord")

set(absl_cord_LINK_LIBS_RELEASE ${absl_cord_LIB_TARGETS_RELEASE} ${absl_cord_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cordz_sample_token VARIABLES #############################################

set(absl_cordz_sample_token_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_sample_token_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_sample_token_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cordz_sample_token_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cordz_sample_token_RES_DIRS_RELEASE )
set(absl_cordz_sample_token_DEFINITIONS_RELEASE )
set(absl_cordz_sample_token_COMPILE_DEFINITIONS_RELEASE )
set(absl_cordz_sample_token_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cordz_sample_token_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cordz_sample_token_LIBS_RELEASE absl_cordz_sample_token)
set(absl_cordz_sample_token_SYSTEM_LIBS_RELEASE )
set(absl_cordz_sample_token_FRAMEWORK_DIRS_RELEASE )
set(absl_cordz_sample_token_FRAMEWORKS_RELEASE )
set(absl_cordz_sample_token_BUILD_MODULES_PATHS_RELEASE )
set(absl_cordz_sample_token_DEPENDENCIES_RELEASE absl::config absl::cordz_handle absl::cordz_info)
set(absl_cordz_sample_token_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cordz_sample_token FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cordz_sample_token_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cordz_sample_token_FRAMEWORKS_FOUND_RELEASE "${absl_cordz_sample_token_FRAMEWORKS_RELEASE}" "${absl_cordz_sample_token_FRAMEWORK_DIRS_RELEASE}")

set(absl_cordz_sample_token_LIB_TARGETS_RELEASE "")
set(absl_cordz_sample_token_NOT_USED_RELEASE "")
set(absl_cordz_sample_token_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cordz_sample_token_FRAMEWORKS_FOUND_RELEASE} ${absl_cordz_sample_token_SYSTEM_LIBS_RELEASE} ${absl_cordz_sample_token_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cordz_sample_token_LIBS_RELEASE}"
                              "${absl_cordz_sample_token_LIB_DIRS_RELEASE}"
                              "${absl_cordz_sample_token_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cordz_sample_token_NOT_USED_RELEASE
                              absl_cordz_sample_token_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cordz_sample_token")

set(absl_cordz_sample_token_LINK_LIBS_RELEASE ${absl_cordz_sample_token_LIB_TARGETS_RELEASE} ${absl_cordz_sample_token_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT numeric_representation VARIABLES #############################################

set(absl_numeric_representation_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_numeric_representation_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_numeric_representation_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_numeric_representation_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_numeric_representation_RES_DIRS_RELEASE )
set(absl_numeric_representation_DEFINITIONS_RELEASE )
set(absl_numeric_representation_COMPILE_DEFINITIONS_RELEASE )
set(absl_numeric_representation_COMPILE_OPTIONS_C_RELEASE "")
set(absl_numeric_representation_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_numeric_representation_LIBS_RELEASE )
set(absl_numeric_representation_SYSTEM_LIBS_RELEASE )
set(absl_numeric_representation_FRAMEWORK_DIRS_RELEASE )
set(absl_numeric_representation_FRAMEWORKS_RELEASE )
set(absl_numeric_representation_BUILD_MODULES_PATHS_RELEASE )
set(absl_numeric_representation_DEPENDENCIES_RELEASE absl::config)
set(absl_numeric_representation_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT numeric_representation FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_numeric_representation_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_numeric_representation_FRAMEWORKS_FOUND_RELEASE "${absl_numeric_representation_FRAMEWORKS_RELEASE}" "${absl_numeric_representation_FRAMEWORK_DIRS_RELEASE}")

set(absl_numeric_representation_LIB_TARGETS_RELEASE "")
set(absl_numeric_representation_NOT_USED_RELEASE "")
set(absl_numeric_representation_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_numeric_representation_FRAMEWORKS_FOUND_RELEASE} ${absl_numeric_representation_SYSTEM_LIBS_RELEASE} ${absl_numeric_representation_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_numeric_representation_LIBS_RELEASE}"
                              "${absl_numeric_representation_LIB_DIRS_RELEASE}"
                              "${absl_numeric_representation_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_numeric_representation_NOT_USED_RELEASE
                              absl_numeric_representation_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_numeric_representation")

set(absl_numeric_representation_LINK_LIBS_RELEASE ${absl_numeric_representation_LIB_TARGETS_RELEASE} ${absl_numeric_representation_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT str_format_internal VARIABLES #############################################

set(absl_str_format_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_str_format_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_str_format_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_str_format_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_str_format_internal_RES_DIRS_RELEASE )
set(absl_str_format_internal_DEFINITIONS_RELEASE )
set(absl_str_format_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_str_format_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_str_format_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_str_format_internal_LIBS_RELEASE absl_str_format_internal)
set(absl_str_format_internal_SYSTEM_LIBS_RELEASE )
set(absl_str_format_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_str_format_internal_FRAMEWORKS_RELEASE )
set(absl_str_format_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_str_format_internal_DEPENDENCIES_RELEASE absl::bits absl::strings absl::config absl::core_headers absl::numeric_representation absl::type_traits absl::utility absl::int128 absl::span)
set(absl_str_format_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT str_format_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_str_format_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_str_format_internal_FRAMEWORKS_FOUND_RELEASE "${absl_str_format_internal_FRAMEWORKS_RELEASE}" "${absl_str_format_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_str_format_internal_LIB_TARGETS_RELEASE "")
set(absl_str_format_internal_NOT_USED_RELEASE "")
set(absl_str_format_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_str_format_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_str_format_internal_SYSTEM_LIBS_RELEASE} ${absl_str_format_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_str_format_internal_LIBS_RELEASE}"
                              "${absl_str_format_internal_LIB_DIRS_RELEASE}"
                              "${absl_str_format_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_str_format_internal_NOT_USED_RELEASE
                              absl_str_format_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_str_format_internal")

set(absl_str_format_internal_LINK_LIBS_RELEASE ${absl_str_format_internal_LIB_TARGETS_RELEASE} ${absl_str_format_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT str_format VARIABLES #############################################

set(absl_str_format_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_str_format_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_str_format_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_str_format_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_str_format_RES_DIRS_RELEASE )
set(absl_str_format_DEFINITIONS_RELEASE )
set(absl_str_format_COMPILE_DEFINITIONS_RELEASE )
set(absl_str_format_COMPILE_OPTIONS_C_RELEASE "")
set(absl_str_format_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_str_format_LIBS_RELEASE )
set(absl_str_format_SYSTEM_LIBS_RELEASE )
set(absl_str_format_FRAMEWORK_DIRS_RELEASE )
set(absl_str_format_FRAMEWORKS_RELEASE )
set(absl_str_format_BUILD_MODULES_PATHS_RELEASE )
set(absl_str_format_DEPENDENCIES_RELEASE absl::str_format_internal)
set(absl_str_format_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT str_format FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_str_format_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_str_format_FRAMEWORKS_FOUND_RELEASE "${absl_str_format_FRAMEWORKS_RELEASE}" "${absl_str_format_FRAMEWORK_DIRS_RELEASE}")

set(absl_str_format_LIB_TARGETS_RELEASE "")
set(absl_str_format_NOT_USED_RELEASE "")
set(absl_str_format_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_str_format_FRAMEWORKS_FOUND_RELEASE} ${absl_str_format_SYSTEM_LIBS_RELEASE} ${absl_str_format_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_str_format_LIBS_RELEASE}"
                              "${absl_str_format_LIB_DIRS_RELEASE}"
                              "${absl_str_format_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_str_format_NOT_USED_RELEASE
                              absl_str_format_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_str_format")

set(absl_str_format_LINK_LIBS_RELEASE ${absl_str_format_LIB_TARGETS_RELEASE} ${absl_str_format_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT strerror VARIABLES #############################################

set(absl_strerror_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strerror_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strerror_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_strerror_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_strerror_RES_DIRS_RELEASE )
set(absl_strerror_DEFINITIONS_RELEASE )
set(absl_strerror_COMPILE_DEFINITIONS_RELEASE )
set(absl_strerror_COMPILE_OPTIONS_C_RELEASE "")
set(absl_strerror_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_strerror_LIBS_RELEASE absl_strerror)
set(absl_strerror_SYSTEM_LIBS_RELEASE )
set(absl_strerror_FRAMEWORK_DIRS_RELEASE )
set(absl_strerror_FRAMEWORKS_RELEASE )
set(absl_strerror_BUILD_MODULES_PATHS_RELEASE )
set(absl_strerror_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::errno_saver)
set(absl_strerror_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT strerror FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_strerror_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_strerror_FRAMEWORKS_FOUND_RELEASE "${absl_strerror_FRAMEWORKS_RELEASE}" "${absl_strerror_FRAMEWORK_DIRS_RELEASE}")

set(absl_strerror_LIB_TARGETS_RELEASE "")
set(absl_strerror_NOT_USED_RELEASE "")
set(absl_strerror_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_strerror_FRAMEWORKS_FOUND_RELEASE} ${absl_strerror_SYSTEM_LIBS_RELEASE} ${absl_strerror_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_strerror_LIBS_RELEASE}"
                              "${absl_strerror_LIB_DIRS_RELEASE}"
                              "${absl_strerror_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_strerror_NOT_USED_RELEASE
                              absl_strerror_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_strerror")

set(absl_strerror_LINK_LIBS_RELEASE ${absl_strerror_LIB_TARGETS_RELEASE} ${absl_strerror_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT status VARIABLES #############################################

set(absl_status_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_status_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_status_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_status_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_status_RES_DIRS_RELEASE )
set(absl_status_DEFINITIONS_RELEASE )
set(absl_status_COMPILE_DEFINITIONS_RELEASE )
set(absl_status_COMPILE_OPTIONS_C_RELEASE "")
set(absl_status_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_status_LIBS_RELEASE absl_status)
set(absl_status_SYSTEM_LIBS_RELEASE )
set(absl_status_FRAMEWORK_DIRS_RELEASE )
set(absl_status_FRAMEWORKS_RELEASE )
set(absl_status_BUILD_MODULES_PATHS_RELEASE )
set(absl_status_DEPENDENCIES_RELEASE absl::atomic_hook absl::config absl::cord absl::core_headers absl::function_ref absl::inlined_vector absl::optional absl::raw_logging_internal absl::stacktrace absl::str_format absl::strerror absl::strings absl::symbolize)
set(absl_status_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT status FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_status_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_status_FRAMEWORKS_FOUND_RELEASE "${absl_status_FRAMEWORKS_RELEASE}" "${absl_status_FRAMEWORK_DIRS_RELEASE}")

set(absl_status_LIB_TARGETS_RELEASE "")
set(absl_status_NOT_USED_RELEASE "")
set(absl_status_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_status_FRAMEWORKS_FOUND_RELEASE} ${absl_status_SYSTEM_LIBS_RELEASE} ${absl_status_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_status_LIBS_RELEASE}"
                              "${absl_status_LIB_DIRS_RELEASE}"
                              "${absl_status_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_status_NOT_USED_RELEASE
                              absl_status_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_status")

set(absl_status_LINK_LIBS_RELEASE ${absl_status_LIB_TARGETS_RELEASE} ${absl_status_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT statusor VARIABLES #############################################

set(absl_statusor_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_statusor_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_statusor_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_statusor_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_statusor_RES_DIRS_RELEASE )
set(absl_statusor_DEFINITIONS_RELEASE )
set(absl_statusor_COMPILE_DEFINITIONS_RELEASE )
set(absl_statusor_COMPILE_OPTIONS_C_RELEASE "")
set(absl_statusor_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_statusor_LIBS_RELEASE absl_statusor)
set(absl_statusor_SYSTEM_LIBS_RELEASE )
set(absl_statusor_FRAMEWORK_DIRS_RELEASE )
set(absl_statusor_FRAMEWORKS_RELEASE )
set(absl_statusor_BUILD_MODULES_PATHS_RELEASE )
set(absl_statusor_DEPENDENCIES_RELEASE absl::base absl::status absl::core_headers absl::raw_logging_internal absl::type_traits absl::strings absl::utility absl::variant)
set(absl_statusor_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT statusor FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_statusor_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_statusor_FRAMEWORKS_FOUND_RELEASE "${absl_statusor_FRAMEWORKS_RELEASE}" "${absl_statusor_FRAMEWORK_DIRS_RELEASE}")

set(absl_statusor_LIB_TARGETS_RELEASE "")
set(absl_statusor_NOT_USED_RELEASE "")
set(absl_statusor_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_statusor_FRAMEWORKS_FOUND_RELEASE} ${absl_statusor_SYSTEM_LIBS_RELEASE} ${absl_statusor_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_statusor_LIBS_RELEASE}"
                              "${absl_statusor_LIB_DIRS_RELEASE}"
                              "${absl_statusor_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_statusor_NOT_USED_RELEASE
                              absl_statusor_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_statusor")

set(absl_statusor_LINK_LIBS_RELEASE ${absl_statusor_LIB_TARGETS_RELEASE} ${absl_statusor_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_traits VARIABLES #############################################

set(absl_random_internal_traits_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_traits_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_traits_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_traits_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_traits_RES_DIRS_RELEASE )
set(absl_random_internal_traits_DEFINITIONS_RELEASE )
set(absl_random_internal_traits_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_traits_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_traits_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_traits_LIBS_RELEASE )
set(absl_random_internal_traits_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_traits_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_traits_FRAMEWORKS_RELEASE )
set(absl_random_internal_traits_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_traits_DEPENDENCIES_RELEASE absl::config)
set(absl_random_internal_traits_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_traits FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_traits_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_traits_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_traits_FRAMEWORKS_RELEASE}" "${absl_random_internal_traits_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_traits_LIB_TARGETS_RELEASE "")
set(absl_random_internal_traits_NOT_USED_RELEASE "")
set(absl_random_internal_traits_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_traits_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_traits_SYSTEM_LIBS_RELEASE} ${absl_random_internal_traits_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_traits_LIBS_RELEASE}"
                              "${absl_random_internal_traits_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_traits_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_traits_NOT_USED_RELEASE
                              absl_random_internal_traits_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_traits")

set(absl_random_internal_traits_LINK_LIBS_RELEASE ${absl_random_internal_traits_LIB_TARGETS_RELEASE} ${absl_random_internal_traits_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_uniform_helper VARIABLES #############################################

set(absl_random_internal_uniform_helper_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_uniform_helper_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_uniform_helper_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_uniform_helper_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_uniform_helper_RES_DIRS_RELEASE )
set(absl_random_internal_uniform_helper_DEFINITIONS_RELEASE )
set(absl_random_internal_uniform_helper_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_uniform_helper_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_uniform_helper_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_uniform_helper_LIBS_RELEASE )
set(absl_random_internal_uniform_helper_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_uniform_helper_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_uniform_helper_FRAMEWORKS_RELEASE )
set(absl_random_internal_uniform_helper_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_uniform_helper_DEPENDENCIES_RELEASE absl::config absl::random_internal_traits absl::type_traits)
set(absl_random_internal_uniform_helper_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_uniform_helper FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_uniform_helper_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_uniform_helper_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_uniform_helper_FRAMEWORKS_RELEASE}" "${absl_random_internal_uniform_helper_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_uniform_helper_LIB_TARGETS_RELEASE "")
set(absl_random_internal_uniform_helper_NOT_USED_RELEASE "")
set(absl_random_internal_uniform_helper_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_uniform_helper_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_uniform_helper_SYSTEM_LIBS_RELEASE} ${absl_random_internal_uniform_helper_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_uniform_helper_LIBS_RELEASE}"
                              "${absl_random_internal_uniform_helper_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_uniform_helper_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_uniform_helper_NOT_USED_RELEASE
                              absl_random_internal_uniform_helper_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_uniform_helper")

set(absl_random_internal_uniform_helper_LINK_LIBS_RELEASE ${absl_random_internal_uniform_helper_LIB_TARGETS_RELEASE} ${absl_random_internal_uniform_helper_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_distribution_test_util VARIABLES #############################################

set(absl_random_internal_distribution_test_util_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_distribution_test_util_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_distribution_test_util_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_distribution_test_util_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_distribution_test_util_RES_DIRS_RELEASE )
set(absl_random_internal_distribution_test_util_DEFINITIONS_RELEASE )
set(absl_random_internal_distribution_test_util_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_distribution_test_util_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_distribution_test_util_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_distribution_test_util_LIBS_RELEASE absl_random_internal_distribution_test_util)
set(absl_random_internal_distribution_test_util_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_distribution_test_util_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_distribution_test_util_FRAMEWORKS_RELEASE )
set(absl_random_internal_distribution_test_util_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_distribution_test_util_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::raw_logging_internal absl::strings absl::str_format absl::span)
set(absl_random_internal_distribution_test_util_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_distribution_test_util FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_distribution_test_util_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_distribution_test_util_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_distribution_test_util_FRAMEWORKS_RELEASE}" "${absl_random_internal_distribution_test_util_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_distribution_test_util_LIB_TARGETS_RELEASE "")
set(absl_random_internal_distribution_test_util_NOT_USED_RELEASE "")
set(absl_random_internal_distribution_test_util_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_distribution_test_util_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_distribution_test_util_SYSTEM_LIBS_RELEASE} ${absl_random_internal_distribution_test_util_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_distribution_test_util_LIBS_RELEASE}"
                              "${absl_random_internal_distribution_test_util_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_distribution_test_util_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_distribution_test_util_NOT_USED_RELEASE
                              absl_random_internal_distribution_test_util_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_distribution_test_util")

set(absl_random_internal_distribution_test_util_LINK_LIBS_RELEASE ${absl_random_internal_distribution_test_util_LIB_TARGETS_RELEASE} ${absl_random_internal_distribution_test_util_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_platform VARIABLES #############################################

set(absl_random_internal_platform_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_platform_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_platform_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_platform_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_platform_RES_DIRS_RELEASE )
set(absl_random_internal_platform_DEFINITIONS_RELEASE )
set(absl_random_internal_platform_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_platform_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_platform_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_platform_LIBS_RELEASE absl_random_internal_platform)
set(absl_random_internal_platform_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_platform_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_platform_FRAMEWORKS_RELEASE )
set(absl_random_internal_platform_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_platform_DEPENDENCIES_RELEASE absl::config)
set(absl_random_internal_platform_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_platform FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_platform_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_platform_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_platform_FRAMEWORKS_RELEASE}" "${absl_random_internal_platform_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_platform_LIB_TARGETS_RELEASE "")
set(absl_random_internal_platform_NOT_USED_RELEASE "")
set(absl_random_internal_platform_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_platform_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_platform_SYSTEM_LIBS_RELEASE} ${absl_random_internal_platform_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_platform_LIBS_RELEASE}"
                              "${absl_random_internal_platform_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_platform_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_platform_NOT_USED_RELEASE
                              absl_random_internal_platform_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_platform")

set(absl_random_internal_platform_LINK_LIBS_RELEASE ${absl_random_internal_platform_LIB_TARGETS_RELEASE} ${absl_random_internal_platform_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_randen_hwaes_impl VARIABLES #############################################

set(absl_random_internal_randen_hwaes_impl_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_hwaes_impl_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_hwaes_impl_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_hwaes_impl_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_randen_hwaes_impl_RES_DIRS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_randen_hwaes_impl_LIBS_RELEASE absl_random_internal_randen_hwaes_impl)
set(absl_random_internal_randen_hwaes_impl_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_FRAMEWORKS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_randen_hwaes_impl_DEPENDENCIES_RELEASE absl::random_internal_platform absl::config)
set(absl_random_internal_randen_hwaes_impl_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_randen_hwaes_impl FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_randen_hwaes_impl_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_randen_hwaes_impl_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_randen_hwaes_impl_FRAMEWORKS_RELEASE}" "${absl_random_internal_randen_hwaes_impl_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_randen_hwaes_impl_LIB_TARGETS_RELEASE "")
set(absl_random_internal_randen_hwaes_impl_NOT_USED_RELEASE "")
set(absl_random_internal_randen_hwaes_impl_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_randen_hwaes_impl_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_randen_hwaes_impl_SYSTEM_LIBS_RELEASE} ${absl_random_internal_randen_hwaes_impl_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_randen_hwaes_impl_LIBS_RELEASE}"
                              "${absl_random_internal_randen_hwaes_impl_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_randen_hwaes_impl_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_randen_hwaes_impl_NOT_USED_RELEASE
                              absl_random_internal_randen_hwaes_impl_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_randen_hwaes_impl")

set(absl_random_internal_randen_hwaes_impl_LINK_LIBS_RELEASE ${absl_random_internal_randen_hwaes_impl_LIB_TARGETS_RELEASE} ${absl_random_internal_randen_hwaes_impl_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_randen_hwaes VARIABLES #############################################

set(absl_random_internal_randen_hwaes_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_hwaes_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_hwaes_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_hwaes_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_randen_hwaes_RES_DIRS_RELEASE )
set(absl_random_internal_randen_hwaes_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_hwaes_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_hwaes_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_randen_hwaes_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_randen_hwaes_LIBS_RELEASE absl_random_internal_randen_hwaes)
set(absl_random_internal_randen_hwaes_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_randen_hwaes_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_randen_hwaes_FRAMEWORKS_RELEASE )
set(absl_random_internal_randen_hwaes_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_randen_hwaes_DEPENDENCIES_RELEASE absl::random_internal_platform absl::random_internal_randen_hwaes_impl absl::config)
set(absl_random_internal_randen_hwaes_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_randen_hwaes FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_randen_hwaes_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_randen_hwaes_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_randen_hwaes_FRAMEWORKS_RELEASE}" "${absl_random_internal_randen_hwaes_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_randen_hwaes_LIB_TARGETS_RELEASE "")
set(absl_random_internal_randen_hwaes_NOT_USED_RELEASE "")
set(absl_random_internal_randen_hwaes_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_randen_hwaes_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_randen_hwaes_SYSTEM_LIBS_RELEASE} ${absl_random_internal_randen_hwaes_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_randen_hwaes_LIBS_RELEASE}"
                              "${absl_random_internal_randen_hwaes_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_randen_hwaes_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_randen_hwaes_NOT_USED_RELEASE
                              absl_random_internal_randen_hwaes_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_randen_hwaes")

set(absl_random_internal_randen_hwaes_LINK_LIBS_RELEASE ${absl_random_internal_randen_hwaes_LIB_TARGETS_RELEASE} ${absl_random_internal_randen_hwaes_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_randen_slow VARIABLES #############################################

set(absl_random_internal_randen_slow_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_slow_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_slow_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_slow_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_randen_slow_RES_DIRS_RELEASE )
set(absl_random_internal_randen_slow_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_slow_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_slow_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_randen_slow_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_randen_slow_LIBS_RELEASE absl_random_internal_randen_slow)
set(absl_random_internal_randen_slow_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_randen_slow_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_randen_slow_FRAMEWORKS_RELEASE )
set(absl_random_internal_randen_slow_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_randen_slow_DEPENDENCIES_RELEASE absl::random_internal_platform absl::config)
set(absl_random_internal_randen_slow_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_randen_slow FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_randen_slow_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_randen_slow_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_randen_slow_FRAMEWORKS_RELEASE}" "${absl_random_internal_randen_slow_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_randen_slow_LIB_TARGETS_RELEASE "")
set(absl_random_internal_randen_slow_NOT_USED_RELEASE "")
set(absl_random_internal_randen_slow_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_randen_slow_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_randen_slow_SYSTEM_LIBS_RELEASE} ${absl_random_internal_randen_slow_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_randen_slow_LIBS_RELEASE}"
                              "${absl_random_internal_randen_slow_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_randen_slow_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_randen_slow_NOT_USED_RELEASE
                              absl_random_internal_randen_slow_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_randen_slow")

set(absl_random_internal_randen_slow_LINK_LIBS_RELEASE ${absl_random_internal_randen_slow_LIB_TARGETS_RELEASE} ${absl_random_internal_randen_slow_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_randen VARIABLES #############################################

set(absl_random_internal_randen_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_randen_RES_DIRS_RELEASE )
set(absl_random_internal_randen_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_randen_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_randen_LIBS_RELEASE absl_random_internal_randen)
set(absl_random_internal_randen_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_randen_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_randen_FRAMEWORKS_RELEASE )
set(absl_random_internal_randen_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_randen_DEPENDENCIES_RELEASE absl::random_internal_platform absl::random_internal_randen_hwaes absl::random_internal_randen_slow)
set(absl_random_internal_randen_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_randen FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_randen_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_randen_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_randen_FRAMEWORKS_RELEASE}" "${absl_random_internal_randen_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_randen_LIB_TARGETS_RELEASE "")
set(absl_random_internal_randen_NOT_USED_RELEASE "")
set(absl_random_internal_randen_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_randen_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_randen_SYSTEM_LIBS_RELEASE} ${absl_random_internal_randen_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_randen_LIBS_RELEASE}"
                              "${absl_random_internal_randen_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_randen_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_randen_NOT_USED_RELEASE
                              absl_random_internal_randen_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_randen")

set(absl_random_internal_randen_LINK_LIBS_RELEASE ${absl_random_internal_randen_LIB_TARGETS_RELEASE} ${absl_random_internal_randen_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_iostream_state_saver VARIABLES #############################################

set(absl_random_internal_iostream_state_saver_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_iostream_state_saver_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_iostream_state_saver_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_iostream_state_saver_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_iostream_state_saver_RES_DIRS_RELEASE )
set(absl_random_internal_iostream_state_saver_DEFINITIONS_RELEASE )
set(absl_random_internal_iostream_state_saver_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_iostream_state_saver_LIBS_RELEASE )
set(absl_random_internal_iostream_state_saver_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_iostream_state_saver_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_iostream_state_saver_FRAMEWORKS_RELEASE )
set(absl_random_internal_iostream_state_saver_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_iostream_state_saver_DEPENDENCIES_RELEASE absl::int128 absl::type_traits)
set(absl_random_internal_iostream_state_saver_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_iostream_state_saver FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_iostream_state_saver_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_iostream_state_saver_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_iostream_state_saver_FRAMEWORKS_RELEASE}" "${absl_random_internal_iostream_state_saver_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_iostream_state_saver_LIB_TARGETS_RELEASE "")
set(absl_random_internal_iostream_state_saver_NOT_USED_RELEASE "")
set(absl_random_internal_iostream_state_saver_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_iostream_state_saver_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_iostream_state_saver_SYSTEM_LIBS_RELEASE} ${absl_random_internal_iostream_state_saver_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_iostream_state_saver_LIBS_RELEASE}"
                              "${absl_random_internal_iostream_state_saver_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_iostream_state_saver_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_iostream_state_saver_NOT_USED_RELEASE
                              absl_random_internal_iostream_state_saver_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_iostream_state_saver")

set(absl_random_internal_iostream_state_saver_LINK_LIBS_RELEASE ${absl_random_internal_iostream_state_saver_LIB_TARGETS_RELEASE} ${absl_random_internal_iostream_state_saver_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_randen_engine VARIABLES #############################################

set(absl_random_internal_randen_engine_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_engine_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_engine_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_randen_engine_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_randen_engine_RES_DIRS_RELEASE )
set(absl_random_internal_randen_engine_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_engine_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_randen_engine_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_randen_engine_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_randen_engine_LIBS_RELEASE )
set(absl_random_internal_randen_engine_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_randen_engine_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_randen_engine_FRAMEWORKS_RELEASE )
set(absl_random_internal_randen_engine_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_randen_engine_DEPENDENCIES_RELEASE absl::endian absl::random_internal_iostream_state_saver absl::random_internal_randen absl::raw_logging_internal absl::type_traits)
set(absl_random_internal_randen_engine_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_randen_engine FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_randen_engine_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_randen_engine_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_randen_engine_FRAMEWORKS_RELEASE}" "${absl_random_internal_randen_engine_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_randen_engine_LIB_TARGETS_RELEASE "")
set(absl_random_internal_randen_engine_NOT_USED_RELEASE "")
set(absl_random_internal_randen_engine_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_randen_engine_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_randen_engine_SYSTEM_LIBS_RELEASE} ${absl_random_internal_randen_engine_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_randen_engine_LIBS_RELEASE}"
                              "${absl_random_internal_randen_engine_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_randen_engine_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_randen_engine_NOT_USED_RELEASE
                              absl_random_internal_randen_engine_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_randen_engine")

set(absl_random_internal_randen_engine_LINK_LIBS_RELEASE ${absl_random_internal_randen_engine_LIB_TARGETS_RELEASE} ${absl_random_internal_randen_engine_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_fastmath VARIABLES #############################################

set(absl_random_internal_fastmath_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_fastmath_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_fastmath_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_fastmath_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_fastmath_RES_DIRS_RELEASE )
set(absl_random_internal_fastmath_DEFINITIONS_RELEASE )
set(absl_random_internal_fastmath_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_fastmath_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_fastmath_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_fastmath_LIBS_RELEASE )
set(absl_random_internal_fastmath_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_fastmath_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_fastmath_FRAMEWORKS_RELEASE )
set(absl_random_internal_fastmath_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_fastmath_DEPENDENCIES_RELEASE absl::bits)
set(absl_random_internal_fastmath_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_fastmath FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_fastmath_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_fastmath_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_fastmath_FRAMEWORKS_RELEASE}" "${absl_random_internal_fastmath_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_fastmath_LIB_TARGETS_RELEASE "")
set(absl_random_internal_fastmath_NOT_USED_RELEASE "")
set(absl_random_internal_fastmath_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_fastmath_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_fastmath_SYSTEM_LIBS_RELEASE} ${absl_random_internal_fastmath_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_fastmath_LIBS_RELEASE}"
                              "${absl_random_internal_fastmath_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_fastmath_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_fastmath_NOT_USED_RELEASE
                              absl_random_internal_fastmath_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_fastmath")

set(absl_random_internal_fastmath_LINK_LIBS_RELEASE ${absl_random_internal_fastmath_LIB_TARGETS_RELEASE} ${absl_random_internal_fastmath_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_pcg_engine VARIABLES #############################################

set(absl_random_internal_pcg_engine_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_pcg_engine_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_pcg_engine_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_pcg_engine_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_pcg_engine_RES_DIRS_RELEASE )
set(absl_random_internal_pcg_engine_DEFINITIONS_RELEASE )
set(absl_random_internal_pcg_engine_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_pcg_engine_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_pcg_engine_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_pcg_engine_LIBS_RELEASE )
set(absl_random_internal_pcg_engine_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_pcg_engine_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_pcg_engine_FRAMEWORKS_RELEASE )
set(absl_random_internal_pcg_engine_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_pcg_engine_DEPENDENCIES_RELEASE absl::config absl::int128 absl::random_internal_fastmath absl::random_internal_iostream_state_saver absl::type_traits)
set(absl_random_internal_pcg_engine_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_pcg_engine FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_pcg_engine_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_pcg_engine_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_pcg_engine_FRAMEWORKS_RELEASE}" "${absl_random_internal_pcg_engine_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_pcg_engine_LIB_TARGETS_RELEASE "")
set(absl_random_internal_pcg_engine_NOT_USED_RELEASE "")
set(absl_random_internal_pcg_engine_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_pcg_engine_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_pcg_engine_SYSTEM_LIBS_RELEASE} ${absl_random_internal_pcg_engine_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_pcg_engine_LIBS_RELEASE}"
                              "${absl_random_internal_pcg_engine_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_pcg_engine_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_pcg_engine_NOT_USED_RELEASE
                              absl_random_internal_pcg_engine_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_pcg_engine")

set(absl_random_internal_pcg_engine_LINK_LIBS_RELEASE ${absl_random_internal_pcg_engine_LIB_TARGETS_RELEASE} ${absl_random_internal_pcg_engine_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_fast_uniform_bits VARIABLES #############################################

set(absl_random_internal_fast_uniform_bits_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_fast_uniform_bits_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_fast_uniform_bits_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_fast_uniform_bits_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_fast_uniform_bits_RES_DIRS_RELEASE )
set(absl_random_internal_fast_uniform_bits_DEFINITIONS_RELEASE )
set(absl_random_internal_fast_uniform_bits_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_fast_uniform_bits_LIBS_RELEASE )
set(absl_random_internal_fast_uniform_bits_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_fast_uniform_bits_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_fast_uniform_bits_FRAMEWORKS_RELEASE )
set(absl_random_internal_fast_uniform_bits_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_fast_uniform_bits_DEPENDENCIES_RELEASE absl::config)
set(absl_random_internal_fast_uniform_bits_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_fast_uniform_bits FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_fast_uniform_bits_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_fast_uniform_bits_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_fast_uniform_bits_FRAMEWORKS_RELEASE}" "${absl_random_internal_fast_uniform_bits_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_fast_uniform_bits_LIB_TARGETS_RELEASE "")
set(absl_random_internal_fast_uniform_bits_NOT_USED_RELEASE "")
set(absl_random_internal_fast_uniform_bits_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_fast_uniform_bits_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_fast_uniform_bits_SYSTEM_LIBS_RELEASE} ${absl_random_internal_fast_uniform_bits_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_fast_uniform_bits_LIBS_RELEASE}"
                              "${absl_random_internal_fast_uniform_bits_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_fast_uniform_bits_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_fast_uniform_bits_NOT_USED_RELEASE
                              absl_random_internal_fast_uniform_bits_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_fast_uniform_bits")

set(absl_random_internal_fast_uniform_bits_LINK_LIBS_RELEASE ${absl_random_internal_fast_uniform_bits_LIB_TARGETS_RELEASE} ${absl_random_internal_fast_uniform_bits_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_seed_material VARIABLES #############################################

set(absl_random_internal_seed_material_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_seed_material_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_seed_material_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_seed_material_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_seed_material_RES_DIRS_RELEASE )
set(absl_random_internal_seed_material_DEFINITIONS_RELEASE )
set(absl_random_internal_seed_material_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_seed_material_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_seed_material_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_seed_material_LIBS_RELEASE absl_random_internal_seed_material)
set(absl_random_internal_seed_material_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_seed_material_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_seed_material_FRAMEWORKS_RELEASE )
set(absl_random_internal_seed_material_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_seed_material_DEPENDENCIES_RELEASE absl::core_headers absl::optional absl::random_internal_fast_uniform_bits absl::raw_logging_internal absl::span absl::strings)
set(absl_random_internal_seed_material_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_seed_material FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_seed_material_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_seed_material_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_seed_material_FRAMEWORKS_RELEASE}" "${absl_random_internal_seed_material_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_seed_material_LIB_TARGETS_RELEASE "")
set(absl_random_internal_seed_material_NOT_USED_RELEASE "")
set(absl_random_internal_seed_material_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_seed_material_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_seed_material_SYSTEM_LIBS_RELEASE} ${absl_random_internal_seed_material_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_seed_material_LIBS_RELEASE}"
                              "${absl_random_internal_seed_material_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_seed_material_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_seed_material_NOT_USED_RELEASE
                              absl_random_internal_seed_material_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_seed_material")

set(absl_random_internal_seed_material_LINK_LIBS_RELEASE ${absl_random_internal_seed_material_LIB_TARGETS_RELEASE} ${absl_random_internal_seed_material_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_salted_seed_seq VARIABLES #############################################

set(absl_random_internal_salted_seed_seq_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_salted_seed_seq_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_salted_seed_seq_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_salted_seed_seq_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_salted_seed_seq_RES_DIRS_RELEASE )
set(absl_random_internal_salted_seed_seq_DEFINITIONS_RELEASE )
set(absl_random_internal_salted_seed_seq_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_salted_seed_seq_LIBS_RELEASE )
set(absl_random_internal_salted_seed_seq_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_salted_seed_seq_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_salted_seed_seq_FRAMEWORKS_RELEASE )
set(absl_random_internal_salted_seed_seq_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_salted_seed_seq_DEPENDENCIES_RELEASE absl::inlined_vector absl::optional absl::span absl::random_internal_seed_material absl::type_traits)
set(absl_random_internal_salted_seed_seq_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_salted_seed_seq FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_salted_seed_seq_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_salted_seed_seq_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_salted_seed_seq_FRAMEWORKS_RELEASE}" "${absl_random_internal_salted_seed_seq_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_salted_seed_seq_LIB_TARGETS_RELEASE "")
set(absl_random_internal_salted_seed_seq_NOT_USED_RELEASE "")
set(absl_random_internal_salted_seed_seq_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_salted_seed_seq_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_salted_seed_seq_SYSTEM_LIBS_RELEASE} ${absl_random_internal_salted_seed_seq_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_salted_seed_seq_LIBS_RELEASE}"
                              "${absl_random_internal_salted_seed_seq_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_salted_seed_seq_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_salted_seed_seq_NOT_USED_RELEASE
                              absl_random_internal_salted_seed_seq_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_salted_seed_seq")

set(absl_random_internal_salted_seed_seq_LINK_LIBS_RELEASE ${absl_random_internal_salted_seed_seq_LIB_TARGETS_RELEASE} ${absl_random_internal_salted_seed_seq_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_seed_gen_exception VARIABLES #############################################

set(absl_random_seed_gen_exception_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_seed_gen_exception_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_seed_gen_exception_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_seed_gen_exception_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_seed_gen_exception_RES_DIRS_RELEASE )
set(absl_random_seed_gen_exception_DEFINITIONS_RELEASE )
set(absl_random_seed_gen_exception_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_seed_gen_exception_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_seed_gen_exception_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_seed_gen_exception_LIBS_RELEASE absl_random_seed_gen_exception)
set(absl_random_seed_gen_exception_SYSTEM_LIBS_RELEASE )
set(absl_random_seed_gen_exception_FRAMEWORK_DIRS_RELEASE )
set(absl_random_seed_gen_exception_FRAMEWORKS_RELEASE )
set(absl_random_seed_gen_exception_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_seed_gen_exception_DEPENDENCIES_RELEASE absl::config)
set(absl_random_seed_gen_exception_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_seed_gen_exception FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_seed_gen_exception_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_seed_gen_exception_FRAMEWORKS_FOUND_RELEASE "${absl_random_seed_gen_exception_FRAMEWORKS_RELEASE}" "${absl_random_seed_gen_exception_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_seed_gen_exception_LIB_TARGETS_RELEASE "")
set(absl_random_seed_gen_exception_NOT_USED_RELEASE "")
set(absl_random_seed_gen_exception_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_seed_gen_exception_FRAMEWORKS_FOUND_RELEASE} ${absl_random_seed_gen_exception_SYSTEM_LIBS_RELEASE} ${absl_random_seed_gen_exception_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_seed_gen_exception_LIBS_RELEASE}"
                              "${absl_random_seed_gen_exception_LIB_DIRS_RELEASE}"
                              "${absl_random_seed_gen_exception_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_seed_gen_exception_NOT_USED_RELEASE
                              absl_random_seed_gen_exception_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_seed_gen_exception")

set(absl_random_seed_gen_exception_LINK_LIBS_RELEASE ${absl_random_seed_gen_exception_LIB_TARGETS_RELEASE} ${absl_random_seed_gen_exception_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_pool_urbg VARIABLES #############################################

set(absl_random_internal_pool_urbg_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_pool_urbg_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_pool_urbg_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_pool_urbg_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_pool_urbg_RES_DIRS_RELEASE )
set(absl_random_internal_pool_urbg_DEFINITIONS_RELEASE )
set(absl_random_internal_pool_urbg_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_pool_urbg_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_pool_urbg_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_pool_urbg_LIBS_RELEASE absl_random_internal_pool_urbg)
set(absl_random_internal_pool_urbg_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_pool_urbg_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_pool_urbg_FRAMEWORKS_RELEASE )
set(absl_random_internal_pool_urbg_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_pool_urbg_DEPENDENCIES_RELEASE absl::base absl::config absl::core_headers absl::endian absl::random_internal_randen absl::random_internal_seed_material absl::random_internal_traits absl::random_seed_gen_exception absl::raw_logging_internal absl::span)
set(absl_random_internal_pool_urbg_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_pool_urbg FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_pool_urbg_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_pool_urbg_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_pool_urbg_FRAMEWORKS_RELEASE}" "${absl_random_internal_pool_urbg_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_pool_urbg_LIB_TARGETS_RELEASE "")
set(absl_random_internal_pool_urbg_NOT_USED_RELEASE "")
set(absl_random_internal_pool_urbg_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_pool_urbg_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_pool_urbg_SYSTEM_LIBS_RELEASE} ${absl_random_internal_pool_urbg_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_pool_urbg_LIBS_RELEASE}"
                              "${absl_random_internal_pool_urbg_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_pool_urbg_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_pool_urbg_NOT_USED_RELEASE
                              absl_random_internal_pool_urbg_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_pool_urbg")

set(absl_random_internal_pool_urbg_LINK_LIBS_RELEASE ${absl_random_internal_pool_urbg_LIB_TARGETS_RELEASE} ${absl_random_internal_pool_urbg_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_nonsecure_base VARIABLES #############################################

set(absl_random_internal_nonsecure_base_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_nonsecure_base_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_nonsecure_base_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_nonsecure_base_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_nonsecure_base_RES_DIRS_RELEASE )
set(absl_random_internal_nonsecure_base_DEFINITIONS_RELEASE )
set(absl_random_internal_nonsecure_base_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_nonsecure_base_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_nonsecure_base_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_nonsecure_base_LIBS_RELEASE )
set(absl_random_internal_nonsecure_base_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_nonsecure_base_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_nonsecure_base_FRAMEWORKS_RELEASE )
set(absl_random_internal_nonsecure_base_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_nonsecure_base_DEPENDENCIES_RELEASE absl::core_headers absl::inlined_vector absl::random_internal_pool_urbg absl::random_internal_salted_seed_seq absl::random_internal_seed_material absl::span absl::type_traits)
set(absl_random_internal_nonsecure_base_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_nonsecure_base FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_nonsecure_base_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_nonsecure_base_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_nonsecure_base_FRAMEWORKS_RELEASE}" "${absl_random_internal_nonsecure_base_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_nonsecure_base_LIB_TARGETS_RELEASE "")
set(absl_random_internal_nonsecure_base_NOT_USED_RELEASE "")
set(absl_random_internal_nonsecure_base_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_nonsecure_base_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_nonsecure_base_SYSTEM_LIBS_RELEASE} ${absl_random_internal_nonsecure_base_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_nonsecure_base_LIBS_RELEASE}"
                              "${absl_random_internal_nonsecure_base_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_nonsecure_base_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_nonsecure_base_NOT_USED_RELEASE
                              absl_random_internal_nonsecure_base_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_nonsecure_base")

set(absl_random_internal_nonsecure_base_LINK_LIBS_RELEASE ${absl_random_internal_nonsecure_base_LIB_TARGETS_RELEASE} ${absl_random_internal_nonsecure_base_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_wide_multiply VARIABLES #############################################

set(absl_random_internal_wide_multiply_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_wide_multiply_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_wide_multiply_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_wide_multiply_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_wide_multiply_RES_DIRS_RELEASE )
set(absl_random_internal_wide_multiply_DEFINITIONS_RELEASE )
set(absl_random_internal_wide_multiply_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_wide_multiply_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_wide_multiply_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_wide_multiply_LIBS_RELEASE )
set(absl_random_internal_wide_multiply_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_wide_multiply_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_wide_multiply_FRAMEWORKS_RELEASE )
set(absl_random_internal_wide_multiply_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_wide_multiply_DEPENDENCIES_RELEASE absl::bits absl::config absl::int128)
set(absl_random_internal_wide_multiply_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_wide_multiply FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_wide_multiply_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_wide_multiply_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_wide_multiply_FRAMEWORKS_RELEASE}" "${absl_random_internal_wide_multiply_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_wide_multiply_LIB_TARGETS_RELEASE "")
set(absl_random_internal_wide_multiply_NOT_USED_RELEASE "")
set(absl_random_internal_wide_multiply_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_wide_multiply_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_wide_multiply_SYSTEM_LIBS_RELEASE} ${absl_random_internal_wide_multiply_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_wide_multiply_LIBS_RELEASE}"
                              "${absl_random_internal_wide_multiply_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_wide_multiply_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_wide_multiply_NOT_USED_RELEASE
                              absl_random_internal_wide_multiply_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_wide_multiply")

set(absl_random_internal_wide_multiply_LINK_LIBS_RELEASE ${absl_random_internal_wide_multiply_LIB_TARGETS_RELEASE} ${absl_random_internal_wide_multiply_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_generate_real VARIABLES #############################################

set(absl_random_internal_generate_real_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_generate_real_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_generate_real_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_generate_real_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_generate_real_RES_DIRS_RELEASE )
set(absl_random_internal_generate_real_DEFINITIONS_RELEASE )
set(absl_random_internal_generate_real_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_generate_real_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_generate_real_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_generate_real_LIBS_RELEASE )
set(absl_random_internal_generate_real_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_generate_real_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_generate_real_FRAMEWORKS_RELEASE )
set(absl_random_internal_generate_real_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_generate_real_DEPENDENCIES_RELEASE absl::bits absl::random_internal_fastmath absl::random_internal_traits absl::type_traits)
set(absl_random_internal_generate_real_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_generate_real FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_generate_real_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_generate_real_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_generate_real_FRAMEWORKS_RELEASE}" "${absl_random_internal_generate_real_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_generate_real_LIB_TARGETS_RELEASE "")
set(absl_random_internal_generate_real_NOT_USED_RELEASE "")
set(absl_random_internal_generate_real_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_generate_real_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_generate_real_SYSTEM_LIBS_RELEASE} ${absl_random_internal_generate_real_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_generate_real_LIBS_RELEASE}"
                              "${absl_random_internal_generate_real_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_generate_real_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_generate_real_NOT_USED_RELEASE
                              absl_random_internal_generate_real_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_generate_real")

set(absl_random_internal_generate_real_LINK_LIBS_RELEASE ${absl_random_internal_generate_real_LIB_TARGETS_RELEASE} ${absl_random_internal_generate_real_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_distribution_caller VARIABLES #############################################

set(absl_random_internal_distribution_caller_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_distribution_caller_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_distribution_caller_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_distribution_caller_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_distribution_caller_RES_DIRS_RELEASE )
set(absl_random_internal_distribution_caller_DEFINITIONS_RELEASE )
set(absl_random_internal_distribution_caller_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_distribution_caller_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_distribution_caller_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_distribution_caller_LIBS_RELEASE )
set(absl_random_internal_distribution_caller_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_distribution_caller_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_distribution_caller_FRAMEWORKS_RELEASE )
set(absl_random_internal_distribution_caller_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_distribution_caller_DEPENDENCIES_RELEASE absl::config absl::utility absl::fast_type_id)
set(absl_random_internal_distribution_caller_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_distribution_caller FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_distribution_caller_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_distribution_caller_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_distribution_caller_FRAMEWORKS_RELEASE}" "${absl_random_internal_distribution_caller_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_distribution_caller_LIB_TARGETS_RELEASE "")
set(absl_random_internal_distribution_caller_NOT_USED_RELEASE "")
set(absl_random_internal_distribution_caller_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_distribution_caller_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_distribution_caller_SYSTEM_LIBS_RELEASE} ${absl_random_internal_distribution_caller_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_distribution_caller_LIBS_RELEASE}"
                              "${absl_random_internal_distribution_caller_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_distribution_caller_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_distribution_caller_NOT_USED_RELEASE
                              absl_random_internal_distribution_caller_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_distribution_caller")

set(absl_random_internal_distribution_caller_LINK_LIBS_RELEASE ${absl_random_internal_distribution_caller_LIB_TARGETS_RELEASE} ${absl_random_internal_distribution_caller_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_seed_sequences VARIABLES #############################################

set(absl_random_seed_sequences_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_seed_sequences_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_seed_sequences_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_seed_sequences_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_seed_sequences_RES_DIRS_RELEASE )
set(absl_random_seed_sequences_DEFINITIONS_RELEASE )
set(absl_random_seed_sequences_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_seed_sequences_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_seed_sequences_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_seed_sequences_LIBS_RELEASE absl_random_seed_sequences)
set(absl_random_seed_sequences_SYSTEM_LIBS_RELEASE )
set(absl_random_seed_sequences_FRAMEWORK_DIRS_RELEASE )
set(absl_random_seed_sequences_FRAMEWORKS_RELEASE )
set(absl_random_seed_sequences_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_seed_sequences_DEPENDENCIES_RELEASE absl::config absl::inlined_vector absl::random_internal_pool_urbg absl::random_internal_salted_seed_seq absl::random_internal_seed_material absl::random_seed_gen_exception absl::span)
set(absl_random_seed_sequences_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_seed_sequences FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_seed_sequences_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_seed_sequences_FRAMEWORKS_FOUND_RELEASE "${absl_random_seed_sequences_FRAMEWORKS_RELEASE}" "${absl_random_seed_sequences_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_seed_sequences_LIB_TARGETS_RELEASE "")
set(absl_random_seed_sequences_NOT_USED_RELEASE "")
set(absl_random_seed_sequences_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_seed_sequences_FRAMEWORKS_FOUND_RELEASE} ${absl_random_seed_sequences_SYSTEM_LIBS_RELEASE} ${absl_random_seed_sequences_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_seed_sequences_LIBS_RELEASE}"
                              "${absl_random_seed_sequences_LIB_DIRS_RELEASE}"
                              "${absl_random_seed_sequences_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_seed_sequences_NOT_USED_RELEASE
                              absl_random_seed_sequences_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_seed_sequences")

set(absl_random_seed_sequences_LINK_LIBS_RELEASE ${absl_random_seed_sequences_LIB_TARGETS_RELEASE} ${absl_random_seed_sequences_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_distributions VARIABLES #############################################

set(absl_random_distributions_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_distributions_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_distributions_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_distributions_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_distributions_RES_DIRS_RELEASE )
set(absl_random_distributions_DEFINITIONS_RELEASE )
set(absl_random_distributions_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_distributions_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_distributions_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_distributions_LIBS_RELEASE absl_random_distributions)
set(absl_random_distributions_SYSTEM_LIBS_RELEASE )
set(absl_random_distributions_FRAMEWORK_DIRS_RELEASE )
set(absl_random_distributions_FRAMEWORKS_RELEASE )
set(absl_random_distributions_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_distributions_DEPENDENCIES_RELEASE absl::base_internal absl::config absl::core_headers absl::random_internal_generate_real absl::random_internal_distribution_caller absl::random_internal_fast_uniform_bits absl::random_internal_fastmath absl::random_internal_iostream_state_saver absl::random_internal_traits absl::random_internal_uniform_helper absl::random_internal_wide_multiply absl::strings absl::type_traits)
set(absl_random_distributions_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_distributions FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_distributions_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_distributions_FRAMEWORKS_FOUND_RELEASE "${absl_random_distributions_FRAMEWORKS_RELEASE}" "${absl_random_distributions_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_distributions_LIB_TARGETS_RELEASE "")
set(absl_random_distributions_NOT_USED_RELEASE "")
set(absl_random_distributions_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_distributions_FRAMEWORKS_FOUND_RELEASE} ${absl_random_distributions_SYSTEM_LIBS_RELEASE} ${absl_random_distributions_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_distributions_LIBS_RELEASE}"
                              "${absl_random_distributions_LIB_DIRS_RELEASE}"
                              "${absl_random_distributions_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_distributions_NOT_USED_RELEASE
                              absl_random_distributions_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_distributions")

set(absl_random_distributions_LINK_LIBS_RELEASE ${absl_random_distributions_LIB_TARGETS_RELEASE} ${absl_random_distributions_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_internal_mock_helpers VARIABLES #############################################

set(absl_random_internal_mock_helpers_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_mock_helpers_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_mock_helpers_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_internal_mock_helpers_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_internal_mock_helpers_RES_DIRS_RELEASE )
set(absl_random_internal_mock_helpers_DEFINITIONS_RELEASE )
set(absl_random_internal_mock_helpers_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_internal_mock_helpers_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_internal_mock_helpers_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_internal_mock_helpers_LIBS_RELEASE )
set(absl_random_internal_mock_helpers_SYSTEM_LIBS_RELEASE )
set(absl_random_internal_mock_helpers_FRAMEWORK_DIRS_RELEASE )
set(absl_random_internal_mock_helpers_FRAMEWORKS_RELEASE )
set(absl_random_internal_mock_helpers_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_internal_mock_helpers_DEPENDENCIES_RELEASE absl::fast_type_id absl::optional)
set(absl_random_internal_mock_helpers_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_internal_mock_helpers FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_internal_mock_helpers_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_internal_mock_helpers_FRAMEWORKS_FOUND_RELEASE "${absl_random_internal_mock_helpers_FRAMEWORKS_RELEASE}" "${absl_random_internal_mock_helpers_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_internal_mock_helpers_LIB_TARGETS_RELEASE "")
set(absl_random_internal_mock_helpers_NOT_USED_RELEASE "")
set(absl_random_internal_mock_helpers_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_internal_mock_helpers_FRAMEWORKS_FOUND_RELEASE} ${absl_random_internal_mock_helpers_SYSTEM_LIBS_RELEASE} ${absl_random_internal_mock_helpers_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_internal_mock_helpers_LIBS_RELEASE}"
                              "${absl_random_internal_mock_helpers_LIB_DIRS_RELEASE}"
                              "${absl_random_internal_mock_helpers_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_internal_mock_helpers_NOT_USED_RELEASE
                              absl_random_internal_mock_helpers_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_internal_mock_helpers")

set(absl_random_internal_mock_helpers_LINK_LIBS_RELEASE ${absl_random_internal_mock_helpers_LIB_TARGETS_RELEASE} ${absl_random_internal_mock_helpers_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_bit_gen_ref VARIABLES #############################################

set(absl_random_bit_gen_ref_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_bit_gen_ref_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_bit_gen_ref_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_bit_gen_ref_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_bit_gen_ref_RES_DIRS_RELEASE )
set(absl_random_bit_gen_ref_DEFINITIONS_RELEASE )
set(absl_random_bit_gen_ref_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_bit_gen_ref_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_bit_gen_ref_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_bit_gen_ref_LIBS_RELEASE )
set(absl_random_bit_gen_ref_SYSTEM_LIBS_RELEASE )
set(absl_random_bit_gen_ref_FRAMEWORK_DIRS_RELEASE )
set(absl_random_bit_gen_ref_FRAMEWORKS_RELEASE )
set(absl_random_bit_gen_ref_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_bit_gen_ref_DEPENDENCIES_RELEASE absl::core_headers absl::random_internal_distribution_caller absl::random_internal_fast_uniform_bits absl::type_traits)
set(absl_random_bit_gen_ref_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_bit_gen_ref FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_bit_gen_ref_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_bit_gen_ref_FRAMEWORKS_FOUND_RELEASE "${absl_random_bit_gen_ref_FRAMEWORKS_RELEASE}" "${absl_random_bit_gen_ref_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_bit_gen_ref_LIB_TARGETS_RELEASE "")
set(absl_random_bit_gen_ref_NOT_USED_RELEASE "")
set(absl_random_bit_gen_ref_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_bit_gen_ref_FRAMEWORKS_FOUND_RELEASE} ${absl_random_bit_gen_ref_SYSTEM_LIBS_RELEASE} ${absl_random_bit_gen_ref_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_bit_gen_ref_LIBS_RELEASE}"
                              "${absl_random_bit_gen_ref_LIB_DIRS_RELEASE}"
                              "${absl_random_bit_gen_ref_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_bit_gen_ref_NOT_USED_RELEASE
                              absl_random_bit_gen_ref_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_bit_gen_ref")

set(absl_random_bit_gen_ref_LINK_LIBS_RELEASE ${absl_random_bit_gen_ref_LIB_TARGETS_RELEASE} ${absl_random_bit_gen_ref_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT random_random VARIABLES #############################################

set(absl_random_random_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_random_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_random_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_random_random_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_random_random_RES_DIRS_RELEASE )
set(absl_random_random_DEFINITIONS_RELEASE )
set(absl_random_random_COMPILE_DEFINITIONS_RELEASE )
set(absl_random_random_COMPILE_OPTIONS_C_RELEASE "")
set(absl_random_random_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_random_random_LIBS_RELEASE )
set(absl_random_random_SYSTEM_LIBS_RELEASE )
set(absl_random_random_FRAMEWORK_DIRS_RELEASE )
set(absl_random_random_FRAMEWORKS_RELEASE )
set(absl_random_random_BUILD_MODULES_PATHS_RELEASE )
set(absl_random_random_DEPENDENCIES_RELEASE absl::random_distributions absl::random_internal_nonsecure_base absl::random_internal_pcg_engine absl::random_internal_pool_urbg absl::random_internal_randen_engine absl::random_seed_sequences)
set(absl_random_random_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT random_random FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_random_random_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_random_random_FRAMEWORKS_FOUND_RELEASE "${absl_random_random_FRAMEWORKS_RELEASE}" "${absl_random_random_FRAMEWORK_DIRS_RELEASE}")

set(absl_random_random_LIB_TARGETS_RELEASE "")
set(absl_random_random_NOT_USED_RELEASE "")
set(absl_random_random_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_random_random_FRAMEWORKS_FOUND_RELEASE} ${absl_random_random_SYSTEM_LIBS_RELEASE} ${absl_random_random_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_random_random_LIBS_RELEASE}"
                              "${absl_random_random_LIB_DIRS_RELEASE}"
                              "${absl_random_random_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_random_random_NOT_USED_RELEASE
                              absl_random_random_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_random_random")

set(absl_random_random_LINK_LIBS_RELEASE ${absl_random_random_LIB_TARGETS_RELEASE} ${absl_random_random_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT periodic_sampler VARIABLES #############################################

set(absl_periodic_sampler_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_periodic_sampler_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_periodic_sampler_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_periodic_sampler_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_periodic_sampler_RES_DIRS_RELEASE )
set(absl_periodic_sampler_DEFINITIONS_RELEASE )
set(absl_periodic_sampler_COMPILE_DEFINITIONS_RELEASE )
set(absl_periodic_sampler_COMPILE_OPTIONS_C_RELEASE "")
set(absl_periodic_sampler_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_periodic_sampler_LIBS_RELEASE absl_periodic_sampler)
set(absl_periodic_sampler_SYSTEM_LIBS_RELEASE )
set(absl_periodic_sampler_FRAMEWORK_DIRS_RELEASE )
set(absl_periodic_sampler_FRAMEWORKS_RELEASE )
set(absl_periodic_sampler_BUILD_MODULES_PATHS_RELEASE )
set(absl_periodic_sampler_DEPENDENCIES_RELEASE absl::core_headers absl::exponential_biased)
set(absl_periodic_sampler_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT periodic_sampler FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_periodic_sampler_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_periodic_sampler_FRAMEWORKS_FOUND_RELEASE "${absl_periodic_sampler_FRAMEWORKS_RELEASE}" "${absl_periodic_sampler_FRAMEWORK_DIRS_RELEASE}")

set(absl_periodic_sampler_LIB_TARGETS_RELEASE "")
set(absl_periodic_sampler_NOT_USED_RELEASE "")
set(absl_periodic_sampler_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_periodic_sampler_FRAMEWORKS_FOUND_RELEASE} ${absl_periodic_sampler_SYSTEM_LIBS_RELEASE} ${absl_periodic_sampler_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_periodic_sampler_LIBS_RELEASE}"
                              "${absl_periodic_sampler_LIB_DIRS_RELEASE}"
                              "${absl_periodic_sampler_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_periodic_sampler_NOT_USED_RELEASE
                              absl_periodic_sampler_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_periodic_sampler")

set(absl_periodic_sampler_LINK_LIBS_RELEASE ${absl_periodic_sampler_LIB_TARGETS_RELEASE} ${absl_periodic_sampler_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT sample_recorder VARIABLES #############################################

set(absl_sample_recorder_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_sample_recorder_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_sample_recorder_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_sample_recorder_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_sample_recorder_RES_DIRS_RELEASE )
set(absl_sample_recorder_DEFINITIONS_RELEASE )
set(absl_sample_recorder_COMPILE_DEFINITIONS_RELEASE )
set(absl_sample_recorder_COMPILE_OPTIONS_C_RELEASE "")
set(absl_sample_recorder_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_sample_recorder_LIBS_RELEASE )
set(absl_sample_recorder_SYSTEM_LIBS_RELEASE )
set(absl_sample_recorder_FRAMEWORK_DIRS_RELEASE )
set(absl_sample_recorder_FRAMEWORKS_RELEASE )
set(absl_sample_recorder_BUILD_MODULES_PATHS_RELEASE )
set(absl_sample_recorder_DEPENDENCIES_RELEASE absl::base absl::synchronization)
set(absl_sample_recorder_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT sample_recorder FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_sample_recorder_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_sample_recorder_FRAMEWORKS_FOUND_RELEASE "${absl_sample_recorder_FRAMEWORKS_RELEASE}" "${absl_sample_recorder_FRAMEWORK_DIRS_RELEASE}")

set(absl_sample_recorder_LIB_TARGETS_RELEASE "")
set(absl_sample_recorder_NOT_USED_RELEASE "")
set(absl_sample_recorder_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_sample_recorder_FRAMEWORKS_FOUND_RELEASE} ${absl_sample_recorder_SYSTEM_LIBS_RELEASE} ${absl_sample_recorder_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_sample_recorder_LIBS_RELEASE}"
                              "${absl_sample_recorder_LIB_DIRS_RELEASE}"
                              "${absl_sample_recorder_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_sample_recorder_NOT_USED_RELEASE
                              absl_sample_recorder_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_sample_recorder")

set(absl_sample_recorder_LINK_LIBS_RELEASE ${absl_sample_recorder_LIB_TARGETS_RELEASE} ${absl_sample_recorder_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT numeric VARIABLES #############################################

set(absl_numeric_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_numeric_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_numeric_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_numeric_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_numeric_RES_DIRS_RELEASE )
set(absl_numeric_DEFINITIONS_RELEASE )
set(absl_numeric_COMPILE_DEFINITIONS_RELEASE )
set(absl_numeric_COMPILE_OPTIONS_C_RELEASE "")
set(absl_numeric_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_numeric_LIBS_RELEASE )
set(absl_numeric_SYSTEM_LIBS_RELEASE )
set(absl_numeric_FRAMEWORK_DIRS_RELEASE )
set(absl_numeric_FRAMEWORKS_RELEASE )
set(absl_numeric_BUILD_MODULES_PATHS_RELEASE )
set(absl_numeric_DEPENDENCIES_RELEASE absl::int128)
set(absl_numeric_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT numeric FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_numeric_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_numeric_FRAMEWORKS_FOUND_RELEASE "${absl_numeric_FRAMEWORKS_RELEASE}" "${absl_numeric_FRAMEWORK_DIRS_RELEASE}")

set(absl_numeric_LIB_TARGETS_RELEASE "")
set(absl_numeric_NOT_USED_RELEASE "")
set(absl_numeric_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_numeric_FRAMEWORKS_FOUND_RELEASE} ${absl_numeric_SYSTEM_LIBS_RELEASE} ${absl_numeric_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_numeric_LIBS_RELEASE}"
                              "${absl_numeric_LIB_DIRS_RELEASE}"
                              "${absl_numeric_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_numeric_NOT_USED_RELEASE
                              absl_numeric_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_numeric")

set(absl_numeric_LINK_LIBS_RELEASE ${absl_numeric_LIB_TARGETS_RELEASE} ${absl_numeric_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_config VARIABLES #############################################

set(absl_log_internal_config_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_config_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_config_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_config_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_config_RES_DIRS_RELEASE )
set(absl_log_internal_config_DEFINITIONS_RELEASE )
set(absl_log_internal_config_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_config_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_config_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_config_LIBS_RELEASE )
set(absl_log_internal_config_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_config_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_config_FRAMEWORKS_RELEASE )
set(absl_log_internal_config_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_config_DEPENDENCIES_RELEASE absl::config absl::core_headers)
set(absl_log_internal_config_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_config FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_config_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_config_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_config_FRAMEWORKS_RELEASE}" "${absl_log_internal_config_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_config_LIB_TARGETS_RELEASE "")
set(absl_log_internal_config_NOT_USED_RELEASE "")
set(absl_log_internal_config_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_config_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_config_SYSTEM_LIBS_RELEASE} ${absl_log_internal_config_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_config_LIBS_RELEASE}"
                              "${absl_log_internal_config_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_config_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_config_NOT_USED_RELEASE
                              absl_log_internal_config_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_config")

set(absl_log_internal_config_LINK_LIBS_RELEASE ${absl_log_internal_config_LIB_TARGETS_RELEASE} ${absl_log_internal_config_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_entry VARIABLES #############################################

set(absl_log_entry_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_entry_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_entry_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_entry_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_entry_RES_DIRS_RELEASE )
set(absl_log_entry_DEFINITIONS_RELEASE )
set(absl_log_entry_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_entry_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_entry_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_entry_LIBS_RELEASE absl_log_entry)
set(absl_log_entry_SYSTEM_LIBS_RELEASE )
set(absl_log_entry_FRAMEWORK_DIRS_RELEASE )
set(absl_log_entry_FRAMEWORKS_RELEASE )
set(absl_log_entry_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_entry_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_internal_config absl::log_severity absl::span absl::strings absl::time)
set(absl_log_entry_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_entry FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_entry_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_entry_FRAMEWORKS_FOUND_RELEASE "${absl_log_entry_FRAMEWORKS_RELEASE}" "${absl_log_entry_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_entry_LIB_TARGETS_RELEASE "")
set(absl_log_entry_NOT_USED_RELEASE "")
set(absl_log_entry_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_entry_FRAMEWORKS_FOUND_RELEASE} ${absl_log_entry_SYSTEM_LIBS_RELEASE} ${absl_log_entry_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_entry_LIBS_RELEASE}"
                              "${absl_log_entry_LIB_DIRS_RELEASE}"
                              "${absl_log_entry_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_entry_NOT_USED_RELEASE
                              absl_log_entry_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_entry")

set(absl_log_entry_LINK_LIBS_RELEASE ${absl_log_entry_LIB_TARGETS_RELEASE} ${absl_log_entry_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_sink VARIABLES #############################################

set(absl_log_sink_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_sink_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_sink_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_sink_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_sink_RES_DIRS_RELEASE )
set(absl_log_sink_DEFINITIONS_RELEASE )
set(absl_log_sink_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_sink_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_sink_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_sink_LIBS_RELEASE absl_log_sink)
set(absl_log_sink_SYSTEM_LIBS_RELEASE )
set(absl_log_sink_FRAMEWORK_DIRS_RELEASE )
set(absl_log_sink_FRAMEWORKS_RELEASE )
set(absl_log_sink_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_sink_DEPENDENCIES_RELEASE absl::config absl::log_entry)
set(absl_log_sink_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_sink FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_sink_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_sink_FRAMEWORKS_FOUND_RELEASE "${absl_log_sink_FRAMEWORKS_RELEASE}" "${absl_log_sink_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_sink_LIB_TARGETS_RELEASE "")
set(absl_log_sink_NOT_USED_RELEASE "")
set(absl_log_sink_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_sink_FRAMEWORKS_FOUND_RELEASE} ${absl_log_sink_SYSTEM_LIBS_RELEASE} ${absl_log_sink_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_sink_LIBS_RELEASE}"
                              "${absl_log_sink_LIB_DIRS_RELEASE}"
                              "${absl_log_sink_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_sink_NOT_USED_RELEASE
                              absl_log_sink_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_sink")

set(absl_log_sink_LINK_LIBS_RELEASE ${absl_log_sink_LIB_TARGETS_RELEASE} ${absl_log_sink_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT low_level_hash VARIABLES #############################################

set(absl_low_level_hash_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_low_level_hash_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_low_level_hash_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_low_level_hash_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_low_level_hash_RES_DIRS_RELEASE )
set(absl_low_level_hash_DEFINITIONS_RELEASE )
set(absl_low_level_hash_COMPILE_DEFINITIONS_RELEASE )
set(absl_low_level_hash_COMPILE_OPTIONS_C_RELEASE "")
set(absl_low_level_hash_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_low_level_hash_LIBS_RELEASE absl_low_level_hash)
set(absl_low_level_hash_SYSTEM_LIBS_RELEASE )
set(absl_low_level_hash_FRAMEWORK_DIRS_RELEASE )
set(absl_low_level_hash_FRAMEWORKS_RELEASE )
set(absl_low_level_hash_BUILD_MODULES_PATHS_RELEASE )
set(absl_low_level_hash_DEPENDENCIES_RELEASE absl::config absl::endian absl::int128)
set(absl_low_level_hash_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT low_level_hash FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_low_level_hash_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_low_level_hash_FRAMEWORKS_FOUND_RELEASE "${absl_low_level_hash_FRAMEWORKS_RELEASE}" "${absl_low_level_hash_FRAMEWORK_DIRS_RELEASE}")

set(absl_low_level_hash_LIB_TARGETS_RELEASE "")
set(absl_low_level_hash_NOT_USED_RELEASE "")
set(absl_low_level_hash_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_low_level_hash_FRAMEWORKS_FOUND_RELEASE} ${absl_low_level_hash_SYSTEM_LIBS_RELEASE} ${absl_low_level_hash_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_low_level_hash_LIBS_RELEASE}"
                              "${absl_low_level_hash_LIB_DIRS_RELEASE}"
                              "${absl_low_level_hash_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_low_level_hash_NOT_USED_RELEASE
                              absl_low_level_hash_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_low_level_hash")

set(absl_low_level_hash_LINK_LIBS_RELEASE ${absl_low_level_hash_LIB_TARGETS_RELEASE} ${absl_low_level_hash_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT city VARIABLES #############################################

set(absl_city_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_city_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_city_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_city_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_city_RES_DIRS_RELEASE )
set(absl_city_DEFINITIONS_RELEASE )
set(absl_city_COMPILE_DEFINITIONS_RELEASE )
set(absl_city_COMPILE_OPTIONS_C_RELEASE "")
set(absl_city_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_city_LIBS_RELEASE absl_city)
set(absl_city_SYSTEM_LIBS_RELEASE )
set(absl_city_FRAMEWORK_DIRS_RELEASE )
set(absl_city_FRAMEWORKS_RELEASE )
set(absl_city_BUILD_MODULES_PATHS_RELEASE )
set(absl_city_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::endian)
set(absl_city_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT city FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_city_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_city_FRAMEWORKS_FOUND_RELEASE "${absl_city_FRAMEWORKS_RELEASE}" "${absl_city_FRAMEWORK_DIRS_RELEASE}")

set(absl_city_LIB_TARGETS_RELEASE "")
set(absl_city_NOT_USED_RELEASE "")
set(absl_city_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_city_FRAMEWORKS_FOUND_RELEASE} ${absl_city_SYSTEM_LIBS_RELEASE} ${absl_city_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_city_LIBS_RELEASE}"
                              "${absl_city_LIB_DIRS_RELEASE}"
                              "${absl_city_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_city_NOT_USED_RELEASE
                              absl_city_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_city")

set(absl_city_LINK_LIBS_RELEASE ${absl_city_LIB_TARGETS_RELEASE} ${absl_city_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT hash VARIABLES #############################################

set(absl_hash_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_hash_RES_DIRS_RELEASE )
set(absl_hash_DEFINITIONS_RELEASE )
set(absl_hash_COMPILE_DEFINITIONS_RELEASE )
set(absl_hash_COMPILE_OPTIONS_C_RELEASE "")
set(absl_hash_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_hash_LIBS_RELEASE absl_hash)
set(absl_hash_SYSTEM_LIBS_RELEASE )
set(absl_hash_FRAMEWORK_DIRS_RELEASE )
set(absl_hash_FRAMEWORKS_RELEASE )
set(absl_hash_BUILD_MODULES_PATHS_RELEASE )
set(absl_hash_DEPENDENCIES_RELEASE absl::bits absl::city absl::config absl::core_headers absl::endian absl::fixed_array absl::function_ref absl::meta absl::int128 absl::strings absl::optional absl::variant absl::utility absl::low_level_hash)
set(absl_hash_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT hash FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_hash_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_hash_FRAMEWORKS_FOUND_RELEASE "${absl_hash_FRAMEWORKS_RELEASE}" "${absl_hash_FRAMEWORK_DIRS_RELEASE}")

set(absl_hash_LIB_TARGETS_RELEASE "")
set(absl_hash_NOT_USED_RELEASE "")
set(absl_hash_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_hash_FRAMEWORKS_FOUND_RELEASE} ${absl_hash_SYSTEM_LIBS_RELEASE} ${absl_hash_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_hash_LIBS_RELEASE}"
                              "${absl_hash_LIB_DIRS_RELEASE}"
                              "${absl_hash_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_hash_NOT_USED_RELEASE
                              absl_hash_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_hash")

set(absl_hash_LINK_LIBS_RELEASE ${absl_hash_LIB_TARGETS_RELEASE} ${absl_hash_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_globals VARIABLES #############################################

set(absl_log_globals_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_globals_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_globals_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_globals_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_globals_RES_DIRS_RELEASE )
set(absl_log_globals_DEFINITIONS_RELEASE )
set(absl_log_globals_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_globals_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_globals_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_globals_LIBS_RELEASE absl_log_globals)
set(absl_log_globals_SYSTEM_LIBS_RELEASE )
set(absl_log_globals_FRAMEWORK_DIRS_RELEASE )
set(absl_log_globals_FRAMEWORKS_RELEASE )
set(absl_log_globals_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_globals_DEPENDENCIES_RELEASE absl::atomic_hook absl::config absl::core_headers absl::hash absl::log_severity absl::strings)
set(absl_log_globals_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_globals FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_globals_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_globals_FRAMEWORKS_FOUND_RELEASE "${absl_log_globals_FRAMEWORKS_RELEASE}" "${absl_log_globals_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_globals_LIB_TARGETS_RELEASE "")
set(absl_log_globals_NOT_USED_RELEASE "")
set(absl_log_globals_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_globals_FRAMEWORKS_FOUND_RELEASE} ${absl_log_globals_SYSTEM_LIBS_RELEASE} ${absl_log_globals_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_globals_LIBS_RELEASE}"
                              "${absl_log_globals_LIB_DIRS_RELEASE}"
                              "${absl_log_globals_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_globals_NOT_USED_RELEASE
                              absl_log_globals_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_globals")

set(absl_log_globals_LINK_LIBS_RELEASE ${absl_log_globals_LIB_TARGETS_RELEASE} ${absl_log_globals_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_globals VARIABLES #############################################

set(absl_log_internal_globals_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_globals_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_globals_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_globals_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_globals_RES_DIRS_RELEASE )
set(absl_log_internal_globals_DEFINITIONS_RELEASE )
set(absl_log_internal_globals_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_globals_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_globals_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_globals_LIBS_RELEASE absl_log_internal_globals)
set(absl_log_internal_globals_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_globals_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_globals_FRAMEWORKS_RELEASE )
set(absl_log_internal_globals_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_globals_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_severity absl::raw_logging_internal absl::strings absl::time)
set(absl_log_internal_globals_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_globals FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_globals_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_globals_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_globals_FRAMEWORKS_RELEASE}" "${absl_log_internal_globals_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_globals_LIB_TARGETS_RELEASE "")
set(absl_log_internal_globals_NOT_USED_RELEASE "")
set(absl_log_internal_globals_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_globals_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_globals_SYSTEM_LIBS_RELEASE} ${absl_log_internal_globals_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_globals_LIBS_RELEASE}"
                              "${absl_log_internal_globals_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_globals_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_globals_NOT_USED_RELEASE
                              absl_log_internal_globals_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_globals")

set(absl_log_internal_globals_LINK_LIBS_RELEASE ${absl_log_internal_globals_LIB_TARGETS_RELEASE} ${absl_log_internal_globals_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cleanup_internal VARIABLES #############################################

set(absl_cleanup_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cleanup_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cleanup_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cleanup_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cleanup_internal_RES_DIRS_RELEASE )
set(absl_cleanup_internal_DEFINITIONS_RELEASE )
set(absl_cleanup_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_cleanup_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cleanup_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cleanup_internal_LIBS_RELEASE )
set(absl_cleanup_internal_SYSTEM_LIBS_RELEASE )
set(absl_cleanup_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_cleanup_internal_FRAMEWORKS_RELEASE )
set(absl_cleanup_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_cleanup_internal_DEPENDENCIES_RELEASE absl::base_internal absl::core_headers absl::utility)
set(absl_cleanup_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cleanup_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cleanup_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cleanup_internal_FRAMEWORKS_FOUND_RELEASE "${absl_cleanup_internal_FRAMEWORKS_RELEASE}" "${absl_cleanup_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_cleanup_internal_LIB_TARGETS_RELEASE "")
set(absl_cleanup_internal_NOT_USED_RELEASE "")
set(absl_cleanup_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cleanup_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_cleanup_internal_SYSTEM_LIBS_RELEASE} ${absl_cleanup_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cleanup_internal_LIBS_RELEASE}"
                              "${absl_cleanup_internal_LIB_DIRS_RELEASE}"
                              "${absl_cleanup_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cleanup_internal_NOT_USED_RELEASE
                              absl_cleanup_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cleanup_internal")

set(absl_cleanup_internal_LINK_LIBS_RELEASE ${absl_cleanup_internal_LIB_TARGETS_RELEASE} ${absl_cleanup_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT cleanup VARIABLES #############################################

set(absl_cleanup_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cleanup_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cleanup_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_cleanup_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_cleanup_RES_DIRS_RELEASE )
set(absl_cleanup_DEFINITIONS_RELEASE )
set(absl_cleanup_COMPILE_DEFINITIONS_RELEASE )
set(absl_cleanup_COMPILE_OPTIONS_C_RELEASE "")
set(absl_cleanup_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_cleanup_LIBS_RELEASE )
set(absl_cleanup_SYSTEM_LIBS_RELEASE )
set(absl_cleanup_FRAMEWORK_DIRS_RELEASE )
set(absl_cleanup_FRAMEWORKS_RELEASE )
set(absl_cleanup_BUILD_MODULES_PATHS_RELEASE )
set(absl_cleanup_DEPENDENCIES_RELEASE absl::cleanup_internal absl::config absl::core_headers)
set(absl_cleanup_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT cleanup FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_cleanup_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_cleanup_FRAMEWORKS_FOUND_RELEASE "${absl_cleanup_FRAMEWORKS_RELEASE}" "${absl_cleanup_FRAMEWORK_DIRS_RELEASE}")

set(absl_cleanup_LIB_TARGETS_RELEASE "")
set(absl_cleanup_NOT_USED_RELEASE "")
set(absl_cleanup_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_cleanup_FRAMEWORKS_FOUND_RELEASE} ${absl_cleanup_SYSTEM_LIBS_RELEASE} ${absl_cleanup_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_cleanup_LIBS_RELEASE}"
                              "${absl_cleanup_LIB_DIRS_RELEASE}"
                              "${absl_cleanup_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_cleanup_NOT_USED_RELEASE
                              absl_cleanup_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_cleanup")

set(absl_cleanup_LINK_LIBS_RELEASE ${absl_cleanup_LIB_TARGETS_RELEASE} ${absl_cleanup_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_log_sink_set VARIABLES #############################################

set(absl_log_internal_log_sink_set_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_log_sink_set_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_log_sink_set_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_log_sink_set_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_log_sink_set_RES_DIRS_RELEASE )
set(absl_log_internal_log_sink_set_DEFINITIONS_RELEASE )
set(absl_log_internal_log_sink_set_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_log_sink_set_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_log_sink_set_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_log_sink_set_LIBS_RELEASE absl_log_internal_log_sink_set)
set(absl_log_internal_log_sink_set_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_log_sink_set_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_log_sink_set_FRAMEWORKS_RELEASE )
set(absl_log_internal_log_sink_set_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_log_sink_set_DEPENDENCIES_RELEASE absl::base absl::cleanup absl::config absl::core_headers absl::log_internal_config absl::log_internal_globals absl::log_globals absl::log_entry absl::log_severity absl::log_sink absl::raw_logging_internal absl::synchronization absl::span absl::strings)
set(absl_log_internal_log_sink_set_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_log_sink_set FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_log_sink_set_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_log_sink_set_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_log_sink_set_FRAMEWORKS_RELEASE}" "${absl_log_internal_log_sink_set_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_log_sink_set_LIB_TARGETS_RELEASE "")
set(absl_log_internal_log_sink_set_NOT_USED_RELEASE "")
set(absl_log_internal_log_sink_set_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_log_sink_set_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_log_sink_set_SYSTEM_LIBS_RELEASE} ${absl_log_internal_log_sink_set_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_log_sink_set_LIBS_RELEASE}"
                              "${absl_log_internal_log_sink_set_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_log_sink_set_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_log_sink_set_NOT_USED_RELEASE
                              absl_log_internal_log_sink_set_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_log_sink_set")

set(absl_log_internal_log_sink_set_LINK_LIBS_RELEASE ${absl_log_internal_log_sink_set_LIB_TARGETS_RELEASE} ${absl_log_internal_log_sink_set_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_sink_registry VARIABLES #############################################

set(absl_log_sink_registry_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_sink_registry_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_sink_registry_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_sink_registry_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_sink_registry_RES_DIRS_RELEASE )
set(absl_log_sink_registry_DEFINITIONS_RELEASE )
set(absl_log_sink_registry_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_sink_registry_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_sink_registry_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_sink_registry_LIBS_RELEASE )
set(absl_log_sink_registry_SYSTEM_LIBS_RELEASE )
set(absl_log_sink_registry_FRAMEWORK_DIRS_RELEASE )
set(absl_log_sink_registry_FRAMEWORKS_RELEASE )
set(absl_log_sink_registry_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_sink_registry_DEPENDENCIES_RELEASE absl::config absl::log_sink absl::log_internal_log_sink_set)
set(absl_log_sink_registry_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_sink_registry FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_sink_registry_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_sink_registry_FRAMEWORKS_FOUND_RELEASE "${absl_log_sink_registry_FRAMEWORKS_RELEASE}" "${absl_log_sink_registry_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_sink_registry_LIB_TARGETS_RELEASE "")
set(absl_log_sink_registry_NOT_USED_RELEASE "")
set(absl_log_sink_registry_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_sink_registry_FRAMEWORKS_FOUND_RELEASE} ${absl_log_sink_registry_SYSTEM_LIBS_RELEASE} ${absl_log_sink_registry_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_sink_registry_LIBS_RELEASE}"
                              "${absl_log_sink_registry_LIB_DIRS_RELEASE}"
                              "${absl_log_sink_registry_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_sink_registry_NOT_USED_RELEASE
                              absl_log_sink_registry_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_sink_registry")

set(absl_log_sink_registry_LINK_LIBS_RELEASE ${absl_log_sink_registry_LIB_TARGETS_RELEASE} ${absl_log_sink_registry_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_append_truncated VARIABLES #############################################

set(absl_log_internal_append_truncated_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_append_truncated_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_append_truncated_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_append_truncated_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_append_truncated_RES_DIRS_RELEASE )
set(absl_log_internal_append_truncated_DEFINITIONS_RELEASE )
set(absl_log_internal_append_truncated_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_append_truncated_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_append_truncated_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_append_truncated_LIBS_RELEASE )
set(absl_log_internal_append_truncated_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_append_truncated_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_append_truncated_FRAMEWORKS_RELEASE )
set(absl_log_internal_append_truncated_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_append_truncated_DEPENDENCIES_RELEASE absl::config absl::strings absl::span)
set(absl_log_internal_append_truncated_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_append_truncated FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_append_truncated_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_append_truncated_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_append_truncated_FRAMEWORKS_RELEASE}" "${absl_log_internal_append_truncated_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_append_truncated_LIB_TARGETS_RELEASE "")
set(absl_log_internal_append_truncated_NOT_USED_RELEASE "")
set(absl_log_internal_append_truncated_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_append_truncated_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_append_truncated_SYSTEM_LIBS_RELEASE} ${absl_log_internal_append_truncated_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_append_truncated_LIBS_RELEASE}"
                              "${absl_log_internal_append_truncated_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_append_truncated_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_append_truncated_NOT_USED_RELEASE
                              absl_log_internal_append_truncated_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_append_truncated")

set(absl_log_internal_append_truncated_LINK_LIBS_RELEASE ${absl_log_internal_append_truncated_LIB_TARGETS_RELEASE} ${absl_log_internal_append_truncated_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_nullguard VARIABLES #############################################

set(absl_log_internal_nullguard_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_nullguard_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_nullguard_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_nullguard_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_nullguard_RES_DIRS_RELEASE )
set(absl_log_internal_nullguard_DEFINITIONS_RELEASE )
set(absl_log_internal_nullguard_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_nullguard_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_nullguard_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_nullguard_LIBS_RELEASE absl_log_internal_nullguard)
set(absl_log_internal_nullguard_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_nullguard_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_nullguard_FRAMEWORKS_RELEASE )
set(absl_log_internal_nullguard_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_nullguard_DEPENDENCIES_RELEASE absl::config absl::core_headers)
set(absl_log_internal_nullguard_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_nullguard FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_nullguard_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_nullguard_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_nullguard_FRAMEWORKS_RELEASE}" "${absl_log_internal_nullguard_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_nullguard_LIB_TARGETS_RELEASE "")
set(absl_log_internal_nullguard_NOT_USED_RELEASE "")
set(absl_log_internal_nullguard_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_nullguard_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_nullguard_SYSTEM_LIBS_RELEASE} ${absl_log_internal_nullguard_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_nullguard_LIBS_RELEASE}"
                              "${absl_log_internal_nullguard_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_nullguard_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_nullguard_NOT_USED_RELEASE
                              absl_log_internal_nullguard_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_nullguard")

set(absl_log_internal_nullguard_LINK_LIBS_RELEASE ${absl_log_internal_nullguard_LIB_TARGETS_RELEASE} ${absl_log_internal_nullguard_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_proto VARIABLES #############################################

set(absl_log_internal_proto_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_proto_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_proto_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_proto_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_proto_RES_DIRS_RELEASE )
set(absl_log_internal_proto_DEFINITIONS_RELEASE )
set(absl_log_internal_proto_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_proto_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_proto_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_proto_LIBS_RELEASE absl_log_internal_proto)
set(absl_log_internal_proto_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_proto_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_proto_FRAMEWORKS_RELEASE )
set(absl_log_internal_proto_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_proto_DEPENDENCIES_RELEASE absl::base absl::config absl::core_headers absl::strings absl::span)
set(absl_log_internal_proto_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_proto FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_proto_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_proto_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_proto_FRAMEWORKS_RELEASE}" "${absl_log_internal_proto_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_proto_LIB_TARGETS_RELEASE "")
set(absl_log_internal_proto_NOT_USED_RELEASE "")
set(absl_log_internal_proto_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_proto_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_proto_SYSTEM_LIBS_RELEASE} ${absl_log_internal_proto_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_proto_LIBS_RELEASE}"
                              "${absl_log_internal_proto_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_proto_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_proto_NOT_USED_RELEASE
                              absl_log_internal_proto_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_proto")

set(absl_log_internal_proto_LINK_LIBS_RELEASE ${absl_log_internal_proto_LIB_TARGETS_RELEASE} ${absl_log_internal_proto_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_format VARIABLES #############################################

set(absl_log_internal_format_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_format_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_format_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_format_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_format_RES_DIRS_RELEASE )
set(absl_log_internal_format_DEFINITIONS_RELEASE )
set(absl_log_internal_format_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_format_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_format_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_format_LIBS_RELEASE absl_log_internal_format)
set(absl_log_internal_format_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_format_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_format_FRAMEWORKS_RELEASE )
set(absl_log_internal_format_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_format_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_internal_append_truncated absl::log_internal_config absl::log_internal_globals absl::log_severity absl::strings absl::str_format absl::time absl::span)
set(absl_log_internal_format_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_format FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_format_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_format_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_format_FRAMEWORKS_RELEASE}" "${absl_log_internal_format_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_format_LIB_TARGETS_RELEASE "")
set(absl_log_internal_format_NOT_USED_RELEASE "")
set(absl_log_internal_format_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_format_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_format_SYSTEM_LIBS_RELEASE} ${absl_log_internal_format_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_format_LIBS_RELEASE}"
                              "${absl_log_internal_format_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_format_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_format_NOT_USED_RELEASE
                              absl_log_internal_format_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_format")

set(absl_log_internal_format_LINK_LIBS_RELEASE ${absl_log_internal_format_LIB_TARGETS_RELEASE} ${absl_log_internal_format_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT examine_stack VARIABLES #############################################

set(absl_examine_stack_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_examine_stack_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_examine_stack_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_examine_stack_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_examine_stack_RES_DIRS_RELEASE )
set(absl_examine_stack_DEFINITIONS_RELEASE )
set(absl_examine_stack_COMPILE_DEFINITIONS_RELEASE )
set(absl_examine_stack_COMPILE_OPTIONS_C_RELEASE "")
set(absl_examine_stack_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_examine_stack_LIBS_RELEASE absl_examine_stack)
set(absl_examine_stack_SYSTEM_LIBS_RELEASE )
set(absl_examine_stack_FRAMEWORK_DIRS_RELEASE )
set(absl_examine_stack_FRAMEWORKS_RELEASE )
set(absl_examine_stack_BUILD_MODULES_PATHS_RELEASE )
set(absl_examine_stack_DEPENDENCIES_RELEASE absl::stacktrace absl::symbolize absl::config absl::core_headers absl::raw_logging_internal)
set(absl_examine_stack_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT examine_stack FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_examine_stack_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_examine_stack_FRAMEWORKS_FOUND_RELEASE "${absl_examine_stack_FRAMEWORKS_RELEASE}" "${absl_examine_stack_FRAMEWORK_DIRS_RELEASE}")

set(absl_examine_stack_LIB_TARGETS_RELEASE "")
set(absl_examine_stack_NOT_USED_RELEASE "")
set(absl_examine_stack_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_examine_stack_FRAMEWORKS_FOUND_RELEASE} ${absl_examine_stack_SYSTEM_LIBS_RELEASE} ${absl_examine_stack_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_examine_stack_LIBS_RELEASE}"
                              "${absl_examine_stack_LIB_DIRS_RELEASE}"
                              "${absl_examine_stack_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_examine_stack_NOT_USED_RELEASE
                              absl_examine_stack_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_examine_stack")

set(absl_examine_stack_LINK_LIBS_RELEASE ${absl_examine_stack_LIB_TARGETS_RELEASE} ${absl_examine_stack_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_message VARIABLES #############################################

set(absl_log_internal_message_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_message_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_message_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_message_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_message_RES_DIRS_RELEASE )
set(absl_log_internal_message_DEFINITIONS_RELEASE )
set(absl_log_internal_message_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_message_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_message_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_message_LIBS_RELEASE absl_log_internal_message)
set(absl_log_internal_message_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_message_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_message_FRAMEWORKS_RELEASE )
set(absl_log_internal_message_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_message_DEPENDENCIES_RELEASE absl::base absl::config absl::core_headers absl::errno_saver absl::inlined_vector absl::examine_stack absl::log_internal_append_truncated absl::log_internal_format absl::log_internal_globals absl::log_internal_proto absl::log_internal_log_sink_set absl::log_internal_nullguard absl::log_globals absl::log_entry absl::log_severity absl::log_sink absl::log_sink_registry absl::memory absl::raw_logging_internal absl::strings absl::strerror absl::time absl::span)
set(absl_log_internal_message_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_message FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_message_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_message_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_message_FRAMEWORKS_RELEASE}" "${absl_log_internal_message_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_message_LIB_TARGETS_RELEASE "")
set(absl_log_internal_message_NOT_USED_RELEASE "")
set(absl_log_internal_message_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_message_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_message_SYSTEM_LIBS_RELEASE} ${absl_log_internal_message_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_message_LIBS_RELEASE}"
                              "${absl_log_internal_message_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_message_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_message_NOT_USED_RELEASE
                              absl_log_internal_message_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_message")

set(absl_log_internal_message_LINK_LIBS_RELEASE ${absl_log_internal_message_LIB_TARGETS_RELEASE} ${absl_log_internal_message_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_structured VARIABLES #############################################

set(absl_log_internal_structured_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_structured_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_structured_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_structured_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_structured_RES_DIRS_RELEASE )
set(absl_log_internal_structured_DEFINITIONS_RELEASE )
set(absl_log_internal_structured_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_structured_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_structured_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_structured_LIBS_RELEASE )
set(absl_log_internal_structured_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_structured_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_structured_FRAMEWORKS_RELEASE )
set(absl_log_internal_structured_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_structured_DEPENDENCIES_RELEASE absl::config absl::log_internal_message absl::strings)
set(absl_log_internal_structured_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_structured FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_structured_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_structured_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_structured_FRAMEWORKS_RELEASE}" "${absl_log_internal_structured_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_structured_LIB_TARGETS_RELEASE "")
set(absl_log_internal_structured_NOT_USED_RELEASE "")
set(absl_log_internal_structured_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_structured_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_structured_SYSTEM_LIBS_RELEASE} ${absl_log_internal_structured_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_structured_LIBS_RELEASE}"
                              "${absl_log_internal_structured_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_structured_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_structured_NOT_USED_RELEASE
                              absl_log_internal_structured_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_structured")

set(absl_log_internal_structured_LINK_LIBS_RELEASE ${absl_log_internal_structured_LIB_TARGETS_RELEASE} ${absl_log_internal_structured_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_structured VARIABLES #############################################

set(absl_log_structured_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_structured_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_structured_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_structured_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_structured_RES_DIRS_RELEASE )
set(absl_log_structured_DEFINITIONS_RELEASE )
set(absl_log_structured_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_structured_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_structured_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_structured_LIBS_RELEASE )
set(absl_log_structured_SYSTEM_LIBS_RELEASE )
set(absl_log_structured_FRAMEWORK_DIRS_RELEASE )
set(absl_log_structured_FRAMEWORKS_RELEASE )
set(absl_log_structured_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_structured_DEPENDENCIES_RELEASE absl::config absl::log_internal_structured absl::strings)
set(absl_log_structured_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_structured FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_structured_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_structured_FRAMEWORKS_FOUND_RELEASE "${absl_log_structured_FRAMEWORKS_RELEASE}" "${absl_log_structured_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_structured_LIB_TARGETS_RELEASE "")
set(absl_log_structured_NOT_USED_RELEASE "")
set(absl_log_structured_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_structured_FRAMEWORKS_FOUND_RELEASE} ${absl_log_structured_SYSTEM_LIBS_RELEASE} ${absl_log_structured_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_structured_LIBS_RELEASE}"
                              "${absl_log_structured_LIB_DIRS_RELEASE}"
                              "${absl_log_structured_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_structured_NOT_USED_RELEASE
                              absl_log_structured_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_structured")

set(absl_log_structured_LINK_LIBS_RELEASE ${absl_log_structured_LIB_TARGETS_RELEASE} ${absl_log_structured_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_voidify VARIABLES #############################################

set(absl_log_internal_voidify_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_voidify_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_voidify_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_voidify_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_voidify_RES_DIRS_RELEASE )
set(absl_log_internal_voidify_DEFINITIONS_RELEASE )
set(absl_log_internal_voidify_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_voidify_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_voidify_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_voidify_LIBS_RELEASE )
set(absl_log_internal_voidify_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_voidify_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_voidify_FRAMEWORKS_RELEASE )
set(absl_log_internal_voidify_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_voidify_DEPENDENCIES_RELEASE absl::config)
set(absl_log_internal_voidify_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_voidify FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_voidify_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_voidify_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_voidify_FRAMEWORKS_RELEASE}" "${absl_log_internal_voidify_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_voidify_LIB_TARGETS_RELEASE "")
set(absl_log_internal_voidify_NOT_USED_RELEASE "")
set(absl_log_internal_voidify_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_voidify_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_voidify_SYSTEM_LIBS_RELEASE} ${absl_log_internal_voidify_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_voidify_LIBS_RELEASE}"
                              "${absl_log_internal_voidify_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_voidify_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_voidify_NOT_USED_RELEASE
                              absl_log_internal_voidify_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_voidify")

set(absl_log_internal_voidify_LINK_LIBS_RELEASE ${absl_log_internal_voidify_LIB_TARGETS_RELEASE} ${absl_log_internal_voidify_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_nullstream VARIABLES #############################################

set(absl_log_internal_nullstream_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_nullstream_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_nullstream_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_nullstream_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_nullstream_RES_DIRS_RELEASE )
set(absl_log_internal_nullstream_DEFINITIONS_RELEASE )
set(absl_log_internal_nullstream_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_nullstream_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_nullstream_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_nullstream_LIBS_RELEASE )
set(absl_log_internal_nullstream_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_nullstream_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_nullstream_FRAMEWORKS_RELEASE )
set(absl_log_internal_nullstream_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_nullstream_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_severity absl::strings)
set(absl_log_internal_nullstream_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_nullstream FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_nullstream_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_nullstream_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_nullstream_FRAMEWORKS_RELEASE}" "${absl_log_internal_nullstream_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_nullstream_LIB_TARGETS_RELEASE "")
set(absl_log_internal_nullstream_NOT_USED_RELEASE "")
set(absl_log_internal_nullstream_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_nullstream_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_nullstream_SYSTEM_LIBS_RELEASE} ${absl_log_internal_nullstream_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_nullstream_LIBS_RELEASE}"
                              "${absl_log_internal_nullstream_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_nullstream_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_nullstream_NOT_USED_RELEASE
                              absl_log_internal_nullstream_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_nullstream")

set(absl_log_internal_nullstream_LINK_LIBS_RELEASE ${absl_log_internal_nullstream_LIB_TARGETS_RELEASE} ${absl_log_internal_nullstream_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_strip VARIABLES #############################################

set(absl_log_internal_strip_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_strip_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_strip_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_strip_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_strip_RES_DIRS_RELEASE )
set(absl_log_internal_strip_DEFINITIONS_RELEASE )
set(absl_log_internal_strip_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_strip_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_strip_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_strip_LIBS_RELEASE )
set(absl_log_internal_strip_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_strip_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_strip_FRAMEWORKS_RELEASE )
set(absl_log_internal_strip_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_strip_DEPENDENCIES_RELEASE absl::log_internal_message absl::log_internal_nullstream absl::log_severity)
set(absl_log_internal_strip_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_strip FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_strip_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_strip_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_strip_FRAMEWORKS_RELEASE}" "${absl_log_internal_strip_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_strip_LIB_TARGETS_RELEASE "")
set(absl_log_internal_strip_NOT_USED_RELEASE "")
set(absl_log_internal_strip_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_strip_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_strip_SYSTEM_LIBS_RELEASE} ${absl_log_internal_strip_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_strip_LIBS_RELEASE}"
                              "${absl_log_internal_strip_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_strip_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_strip_NOT_USED_RELEASE
                              absl_log_internal_strip_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_strip")

set(absl_log_internal_strip_LINK_LIBS_RELEASE ${absl_log_internal_strip_LIB_TARGETS_RELEASE} ${absl_log_internal_strip_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_conditions VARIABLES #############################################

set(absl_log_internal_conditions_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_conditions_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_conditions_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_conditions_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_conditions_RES_DIRS_RELEASE )
set(absl_log_internal_conditions_DEFINITIONS_RELEASE )
set(absl_log_internal_conditions_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_conditions_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_conditions_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_conditions_LIBS_RELEASE absl_log_internal_conditions)
set(absl_log_internal_conditions_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_conditions_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_conditions_FRAMEWORKS_RELEASE )
set(absl_log_internal_conditions_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_conditions_DEPENDENCIES_RELEASE absl::base absl::config absl::core_headers absl::log_internal_voidify)
set(absl_log_internal_conditions_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_conditions FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_conditions_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_conditions_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_conditions_FRAMEWORKS_RELEASE}" "${absl_log_internal_conditions_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_conditions_LIB_TARGETS_RELEASE "")
set(absl_log_internal_conditions_NOT_USED_RELEASE "")
set(absl_log_internal_conditions_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_conditions_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_conditions_SYSTEM_LIBS_RELEASE} ${absl_log_internal_conditions_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_conditions_LIBS_RELEASE}"
                              "${absl_log_internal_conditions_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_conditions_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_conditions_NOT_USED_RELEASE
                              absl_log_internal_conditions_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_conditions")

set(absl_log_internal_conditions_LINK_LIBS_RELEASE ${absl_log_internal_conditions_LIB_TARGETS_RELEASE} ${absl_log_internal_conditions_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_log_impl VARIABLES #############################################

set(absl_log_internal_log_impl_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_log_impl_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_log_impl_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_log_impl_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_log_impl_RES_DIRS_RELEASE )
set(absl_log_internal_log_impl_DEFINITIONS_RELEASE )
set(absl_log_internal_log_impl_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_log_impl_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_log_impl_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_log_impl_LIBS_RELEASE )
set(absl_log_internal_log_impl_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_log_impl_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_log_impl_FRAMEWORKS_RELEASE )
set(absl_log_internal_log_impl_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_log_impl_DEPENDENCIES_RELEASE absl::log_internal_conditions absl::log_internal_message absl::log_internal_strip)
set(absl_log_internal_log_impl_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_log_impl FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_log_impl_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_log_impl_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_log_impl_FRAMEWORKS_RELEASE}" "${absl_log_internal_log_impl_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_log_impl_LIB_TARGETS_RELEASE "")
set(absl_log_internal_log_impl_NOT_USED_RELEASE "")
set(absl_log_internal_log_impl_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_log_impl_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_log_impl_SYSTEM_LIBS_RELEASE} ${absl_log_internal_log_impl_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_log_impl_LIBS_RELEASE}"
                              "${absl_log_internal_log_impl_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_log_impl_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_log_impl_NOT_USED_RELEASE
                              absl_log_internal_log_impl_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_log_impl")

set(absl_log_internal_log_impl_LINK_LIBS_RELEASE ${absl_log_internal_log_impl_LIB_TARGETS_RELEASE} ${absl_log_internal_log_impl_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT absl_log VARIABLES #############################################

set(absl_absl_log_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_absl_log_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_absl_log_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_absl_log_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_absl_log_RES_DIRS_RELEASE )
set(absl_absl_log_DEFINITIONS_RELEASE )
set(absl_absl_log_COMPILE_DEFINITIONS_RELEASE )
set(absl_absl_log_COMPILE_OPTIONS_C_RELEASE "")
set(absl_absl_log_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_absl_log_LIBS_RELEASE )
set(absl_absl_log_SYSTEM_LIBS_RELEASE )
set(absl_absl_log_FRAMEWORK_DIRS_RELEASE )
set(absl_absl_log_FRAMEWORKS_RELEASE )
set(absl_absl_log_BUILD_MODULES_PATHS_RELEASE )
set(absl_absl_log_DEPENDENCIES_RELEASE absl::log_internal_log_impl)
set(absl_absl_log_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT absl_log FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_absl_log_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_absl_log_FRAMEWORKS_FOUND_RELEASE "${absl_absl_log_FRAMEWORKS_RELEASE}" "${absl_absl_log_FRAMEWORK_DIRS_RELEASE}")

set(absl_absl_log_LIB_TARGETS_RELEASE "")
set(absl_absl_log_NOT_USED_RELEASE "")
set(absl_absl_log_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_absl_log_FRAMEWORKS_FOUND_RELEASE} ${absl_absl_log_SYSTEM_LIBS_RELEASE} ${absl_absl_log_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_absl_log_LIBS_RELEASE}"
                              "${absl_absl_log_LIB_DIRS_RELEASE}"
                              "${absl_absl_log_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_absl_log_NOT_USED_RELEASE
                              absl_absl_log_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_absl_log")

set(absl_absl_log_LINK_LIBS_RELEASE ${absl_absl_log_LIB_TARGETS_RELEASE} ${absl_absl_log_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_streamer VARIABLES #############################################

set(absl_log_streamer_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_streamer_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_streamer_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_streamer_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_streamer_RES_DIRS_RELEASE )
set(absl_log_streamer_DEFINITIONS_RELEASE )
set(absl_log_streamer_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_streamer_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_streamer_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_streamer_LIBS_RELEASE )
set(absl_log_streamer_SYSTEM_LIBS_RELEASE )
set(absl_log_streamer_FRAMEWORK_DIRS_RELEASE )
set(absl_log_streamer_FRAMEWORKS_RELEASE )
set(absl_log_streamer_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_streamer_DEPENDENCIES_RELEASE absl::config absl::absl_log absl::log_severity absl::optional absl::strings absl::strings_internal absl::utility)
set(absl_log_streamer_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_streamer FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_streamer_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_streamer_FRAMEWORKS_FOUND_RELEASE "${absl_log_streamer_FRAMEWORKS_RELEASE}" "${absl_log_streamer_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_streamer_LIB_TARGETS_RELEASE "")
set(absl_log_streamer_NOT_USED_RELEASE "")
set(absl_log_streamer_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_streamer_FRAMEWORKS_FOUND_RELEASE} ${absl_log_streamer_SYSTEM_LIBS_RELEASE} ${absl_log_streamer_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_streamer_LIBS_RELEASE}"
                              "${absl_log_streamer_LIB_DIRS_RELEASE}"
                              "${absl_log_streamer_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_streamer_NOT_USED_RELEASE
                              absl_log_streamer_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_streamer")

set(absl_log_streamer_LINK_LIBS_RELEASE ${absl_log_streamer_LIB_TARGETS_RELEASE} ${absl_log_streamer_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log VARIABLES #############################################

set(absl_log_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_RES_DIRS_RELEASE )
set(absl_log_DEFINITIONS_RELEASE )
set(absl_log_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_LIBS_RELEASE )
set(absl_log_SYSTEM_LIBS_RELEASE )
set(absl_log_FRAMEWORK_DIRS_RELEASE )
set(absl_log_FRAMEWORKS_RELEASE )
set(absl_log_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_DEPENDENCIES_RELEASE absl::log_internal_log_impl)
set(absl_log_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_FRAMEWORKS_FOUND_RELEASE "${absl_log_FRAMEWORKS_RELEASE}" "${absl_log_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_LIB_TARGETS_RELEASE "")
set(absl_log_NOT_USED_RELEASE "")
set(absl_log_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_FRAMEWORKS_FOUND_RELEASE} ${absl_log_SYSTEM_LIBS_RELEASE} ${absl_log_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_LIBS_RELEASE}"
                              "${absl_log_LIB_DIRS_RELEASE}"
                              "${absl_log_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_NOT_USED_RELEASE
                              absl_log_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log")

set(absl_log_LINK_LIBS_RELEASE ${absl_log_LIB_TARGETS_RELEASE} ${absl_log_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_initialize VARIABLES #############################################

set(absl_log_initialize_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_initialize_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_initialize_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_initialize_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_initialize_RES_DIRS_RELEASE )
set(absl_log_initialize_DEFINITIONS_RELEASE )
set(absl_log_initialize_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_initialize_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_initialize_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_initialize_LIBS_RELEASE absl_log_initialize)
set(absl_log_initialize_SYSTEM_LIBS_RELEASE )
set(absl_log_initialize_FRAMEWORK_DIRS_RELEASE )
set(absl_log_initialize_FRAMEWORKS_RELEASE )
set(absl_log_initialize_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_initialize_DEPENDENCIES_RELEASE absl::config absl::log_globals absl::log_internal_globals absl::time)
set(absl_log_initialize_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_initialize FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_initialize_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_initialize_FRAMEWORKS_FOUND_RELEASE "${absl_log_initialize_FRAMEWORKS_RELEASE}" "${absl_log_initialize_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_initialize_LIB_TARGETS_RELEASE "")
set(absl_log_initialize_NOT_USED_RELEASE "")
set(absl_log_initialize_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_initialize_FRAMEWORKS_FOUND_RELEASE} ${absl_log_initialize_SYSTEM_LIBS_RELEASE} ${absl_log_initialize_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_initialize_LIBS_RELEASE}"
                              "${absl_log_initialize_LIB_DIRS_RELEASE}"
                              "${absl_log_initialize_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_initialize_NOT_USED_RELEASE
                              absl_log_initialize_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_initialize")

set(absl_log_initialize_LINK_LIBS_RELEASE ${absl_log_initialize_LIB_TARGETS_RELEASE} ${absl_log_initialize_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_commandlineflag_internal VARIABLES #############################################

set(absl_flags_commandlineflag_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_commandlineflag_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_commandlineflag_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_commandlineflag_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_commandlineflag_internal_RES_DIRS_RELEASE )
set(absl_flags_commandlineflag_internal_DEFINITIONS_RELEASE )
set(absl_flags_commandlineflag_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_commandlineflag_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_commandlineflag_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_commandlineflag_internal_LIBS_RELEASE absl_flags_commandlineflag_internal)
set(absl_flags_commandlineflag_internal_SYSTEM_LIBS_RELEASE )
set(absl_flags_commandlineflag_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_commandlineflag_internal_FRAMEWORKS_RELEASE )
set(absl_flags_commandlineflag_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_commandlineflag_internal_DEPENDENCIES_RELEASE absl::config absl::dynamic_annotations absl::fast_type_id)
set(absl_flags_commandlineflag_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_commandlineflag_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_commandlineflag_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_commandlineflag_internal_FRAMEWORKS_FOUND_RELEASE "${absl_flags_commandlineflag_internal_FRAMEWORKS_RELEASE}" "${absl_flags_commandlineflag_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_commandlineflag_internal_LIB_TARGETS_RELEASE "")
set(absl_flags_commandlineflag_internal_NOT_USED_RELEASE "")
set(absl_flags_commandlineflag_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_commandlineflag_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_commandlineflag_internal_SYSTEM_LIBS_RELEASE} ${absl_flags_commandlineflag_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_commandlineflag_internal_LIBS_RELEASE}"
                              "${absl_flags_commandlineflag_internal_LIB_DIRS_RELEASE}"
                              "${absl_flags_commandlineflag_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_commandlineflag_internal_NOT_USED_RELEASE
                              absl_flags_commandlineflag_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_commandlineflag_internal")

set(absl_flags_commandlineflag_internal_LINK_LIBS_RELEASE ${absl_flags_commandlineflag_internal_LIB_TARGETS_RELEASE} ${absl_flags_commandlineflag_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_commandlineflag VARIABLES #############################################

set(absl_flags_commandlineflag_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_commandlineflag_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_commandlineflag_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_commandlineflag_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_commandlineflag_RES_DIRS_RELEASE )
set(absl_flags_commandlineflag_DEFINITIONS_RELEASE )
set(absl_flags_commandlineflag_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_commandlineflag_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_commandlineflag_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_commandlineflag_LIBS_RELEASE absl_flags_commandlineflag)
set(absl_flags_commandlineflag_SYSTEM_LIBS_RELEASE )
set(absl_flags_commandlineflag_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_commandlineflag_FRAMEWORKS_RELEASE )
set(absl_flags_commandlineflag_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_commandlineflag_DEPENDENCIES_RELEASE absl::config absl::fast_type_id absl::flags_commandlineflag_internal absl::optional absl::strings)
set(absl_flags_commandlineflag_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_commandlineflag FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_commandlineflag_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_commandlineflag_FRAMEWORKS_FOUND_RELEASE "${absl_flags_commandlineflag_FRAMEWORKS_RELEASE}" "${absl_flags_commandlineflag_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_commandlineflag_LIB_TARGETS_RELEASE "")
set(absl_flags_commandlineflag_NOT_USED_RELEASE "")
set(absl_flags_commandlineflag_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_commandlineflag_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_commandlineflag_SYSTEM_LIBS_RELEASE} ${absl_flags_commandlineflag_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_commandlineflag_LIBS_RELEASE}"
                              "${absl_flags_commandlineflag_LIB_DIRS_RELEASE}"
                              "${absl_flags_commandlineflag_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_commandlineflag_NOT_USED_RELEASE
                              absl_flags_commandlineflag_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_commandlineflag")

set(absl_flags_commandlineflag_LINK_LIBS_RELEASE ${absl_flags_commandlineflag_LIB_TARGETS_RELEASE} ${absl_flags_commandlineflag_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_marshalling VARIABLES #############################################

set(absl_flags_marshalling_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_marshalling_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_marshalling_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_marshalling_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_marshalling_RES_DIRS_RELEASE )
set(absl_flags_marshalling_DEFINITIONS_RELEASE )
set(absl_flags_marshalling_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_marshalling_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_marshalling_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_marshalling_LIBS_RELEASE absl_flags_marshalling)
set(absl_flags_marshalling_SYSTEM_LIBS_RELEASE )
set(absl_flags_marshalling_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_marshalling_FRAMEWORKS_RELEASE )
set(absl_flags_marshalling_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_marshalling_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_severity absl::optional absl::strings absl::str_format)
set(absl_flags_marshalling_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_marshalling FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_marshalling_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_marshalling_FRAMEWORKS_FOUND_RELEASE "${absl_flags_marshalling_FRAMEWORKS_RELEASE}" "${absl_flags_marshalling_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_marshalling_LIB_TARGETS_RELEASE "")
set(absl_flags_marshalling_NOT_USED_RELEASE "")
set(absl_flags_marshalling_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_marshalling_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_marshalling_SYSTEM_LIBS_RELEASE} ${absl_flags_marshalling_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_marshalling_LIBS_RELEASE}"
                              "${absl_flags_marshalling_LIB_DIRS_RELEASE}"
                              "${absl_flags_marshalling_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_marshalling_NOT_USED_RELEASE
                              absl_flags_marshalling_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_marshalling")

set(absl_flags_marshalling_LINK_LIBS_RELEASE ${absl_flags_marshalling_LIB_TARGETS_RELEASE} ${absl_flags_marshalling_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_path_util VARIABLES #############################################

set(absl_flags_path_util_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_path_util_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_path_util_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_path_util_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_path_util_RES_DIRS_RELEASE )
set(absl_flags_path_util_DEFINITIONS_RELEASE )
set(absl_flags_path_util_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_path_util_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_path_util_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_path_util_LIBS_RELEASE )
set(absl_flags_path_util_SYSTEM_LIBS_RELEASE )
set(absl_flags_path_util_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_path_util_FRAMEWORKS_RELEASE )
set(absl_flags_path_util_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_path_util_DEPENDENCIES_RELEASE absl::config absl::strings)
set(absl_flags_path_util_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_path_util FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_path_util_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_path_util_FRAMEWORKS_FOUND_RELEASE "${absl_flags_path_util_FRAMEWORKS_RELEASE}" "${absl_flags_path_util_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_path_util_LIB_TARGETS_RELEASE "")
set(absl_flags_path_util_NOT_USED_RELEASE "")
set(absl_flags_path_util_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_path_util_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_path_util_SYSTEM_LIBS_RELEASE} ${absl_flags_path_util_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_path_util_LIBS_RELEASE}"
                              "${absl_flags_path_util_LIB_DIRS_RELEASE}"
                              "${absl_flags_path_util_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_path_util_NOT_USED_RELEASE
                              absl_flags_path_util_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_path_util")

set(absl_flags_path_util_LINK_LIBS_RELEASE ${absl_flags_path_util_LIB_TARGETS_RELEASE} ${absl_flags_path_util_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_program_name VARIABLES #############################################

set(absl_flags_program_name_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_program_name_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_program_name_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_program_name_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_program_name_RES_DIRS_RELEASE )
set(absl_flags_program_name_DEFINITIONS_RELEASE )
set(absl_flags_program_name_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_program_name_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_program_name_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_program_name_LIBS_RELEASE absl_flags_program_name)
set(absl_flags_program_name_SYSTEM_LIBS_RELEASE )
set(absl_flags_program_name_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_program_name_FRAMEWORKS_RELEASE )
set(absl_flags_program_name_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_program_name_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::flags_path_util absl::strings absl::synchronization)
set(absl_flags_program_name_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_program_name FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_program_name_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_program_name_FRAMEWORKS_FOUND_RELEASE "${absl_flags_program_name_FRAMEWORKS_RELEASE}" "${absl_flags_program_name_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_program_name_LIB_TARGETS_RELEASE "")
set(absl_flags_program_name_NOT_USED_RELEASE "")
set(absl_flags_program_name_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_program_name_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_program_name_SYSTEM_LIBS_RELEASE} ${absl_flags_program_name_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_program_name_LIBS_RELEASE}"
                              "${absl_flags_program_name_LIB_DIRS_RELEASE}"
                              "${absl_flags_program_name_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_program_name_NOT_USED_RELEASE
                              absl_flags_program_name_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_program_name")

set(absl_flags_program_name_LINK_LIBS_RELEASE ${absl_flags_program_name_LIB_TARGETS_RELEASE} ${absl_flags_program_name_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_config VARIABLES #############################################

set(absl_flags_config_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_config_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_config_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_config_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_config_RES_DIRS_RELEASE )
set(absl_flags_config_DEFINITIONS_RELEASE )
set(absl_flags_config_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_config_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_config_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_config_LIBS_RELEASE absl_flags_config)
set(absl_flags_config_SYSTEM_LIBS_RELEASE )
set(absl_flags_config_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_config_FRAMEWORKS_RELEASE )
set(absl_flags_config_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_config_DEPENDENCIES_RELEASE absl::config absl::flags_path_util absl::flags_program_name absl::core_headers absl::strings absl::synchronization)
set(absl_flags_config_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_config FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_config_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_config_FRAMEWORKS_FOUND_RELEASE "${absl_flags_config_FRAMEWORKS_RELEASE}" "${absl_flags_config_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_config_LIB_TARGETS_RELEASE "")
set(absl_flags_config_NOT_USED_RELEASE "")
set(absl_flags_config_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_config_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_config_SYSTEM_LIBS_RELEASE} ${absl_flags_config_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_config_LIBS_RELEASE}"
                              "${absl_flags_config_LIB_DIRS_RELEASE}"
                              "${absl_flags_config_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_config_NOT_USED_RELEASE
                              absl_flags_config_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_config")

set(absl_flags_config_LINK_LIBS_RELEASE ${absl_flags_config_LIB_TARGETS_RELEASE} ${absl_flags_config_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_internal VARIABLES #############################################

set(absl_flags_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_internal_RES_DIRS_RELEASE )
set(absl_flags_internal_DEFINITIONS_RELEASE )
set(absl_flags_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_internal_LIBS_RELEASE absl_flags_internal)
set(absl_flags_internal_SYSTEM_LIBS_RELEASE )
set(absl_flags_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_internal_FRAMEWORKS_RELEASE )
set(absl_flags_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_internal_DEPENDENCIES_RELEASE absl::base absl::config absl::flags_commandlineflag absl::flags_commandlineflag_internal absl::flags_config absl::flags_marshalling absl::synchronization absl::meta absl::utility)
set(absl_flags_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_internal_FRAMEWORKS_FOUND_RELEASE "${absl_flags_internal_FRAMEWORKS_RELEASE}" "${absl_flags_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_internal_LIB_TARGETS_RELEASE "")
set(absl_flags_internal_NOT_USED_RELEASE "")
set(absl_flags_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_internal_SYSTEM_LIBS_RELEASE} ${absl_flags_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_internal_LIBS_RELEASE}"
                              "${absl_flags_internal_LIB_DIRS_RELEASE}"
                              "${absl_flags_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_internal_NOT_USED_RELEASE
                              absl_flags_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_internal")

set(absl_flags_internal_LINK_LIBS_RELEASE ${absl_flags_internal_LIB_TARGETS_RELEASE} ${absl_flags_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_private_handle_accessor VARIABLES #############################################

set(absl_flags_private_handle_accessor_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_private_handle_accessor_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_private_handle_accessor_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_private_handle_accessor_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_private_handle_accessor_RES_DIRS_RELEASE )
set(absl_flags_private_handle_accessor_DEFINITIONS_RELEASE )
set(absl_flags_private_handle_accessor_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_private_handle_accessor_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_private_handle_accessor_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_private_handle_accessor_LIBS_RELEASE absl_flags_private_handle_accessor)
set(absl_flags_private_handle_accessor_SYSTEM_LIBS_RELEASE )
set(absl_flags_private_handle_accessor_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_private_handle_accessor_FRAMEWORKS_RELEASE )
set(absl_flags_private_handle_accessor_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_private_handle_accessor_DEPENDENCIES_RELEASE absl::config absl::flags_commandlineflag absl::flags_commandlineflag_internal absl::strings)
set(absl_flags_private_handle_accessor_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_private_handle_accessor FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_private_handle_accessor_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_private_handle_accessor_FRAMEWORKS_FOUND_RELEASE "${absl_flags_private_handle_accessor_FRAMEWORKS_RELEASE}" "${absl_flags_private_handle_accessor_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_private_handle_accessor_LIB_TARGETS_RELEASE "")
set(absl_flags_private_handle_accessor_NOT_USED_RELEASE "")
set(absl_flags_private_handle_accessor_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_private_handle_accessor_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_private_handle_accessor_SYSTEM_LIBS_RELEASE} ${absl_flags_private_handle_accessor_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_private_handle_accessor_LIBS_RELEASE}"
                              "${absl_flags_private_handle_accessor_LIB_DIRS_RELEASE}"
                              "${absl_flags_private_handle_accessor_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_private_handle_accessor_NOT_USED_RELEASE
                              absl_flags_private_handle_accessor_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_private_handle_accessor")

set(absl_flags_private_handle_accessor_LINK_LIBS_RELEASE ${absl_flags_private_handle_accessor_LIB_TARGETS_RELEASE} ${absl_flags_private_handle_accessor_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT container_common VARIABLES #############################################

set(absl_container_common_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_container_common_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_container_common_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_container_common_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_container_common_RES_DIRS_RELEASE )
set(absl_container_common_DEFINITIONS_RELEASE )
set(absl_container_common_COMPILE_DEFINITIONS_RELEASE )
set(absl_container_common_COMPILE_OPTIONS_C_RELEASE "")
set(absl_container_common_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_container_common_LIBS_RELEASE )
set(absl_container_common_SYSTEM_LIBS_RELEASE )
set(absl_container_common_FRAMEWORK_DIRS_RELEASE )
set(absl_container_common_FRAMEWORKS_RELEASE )
set(absl_container_common_BUILD_MODULES_PATHS_RELEASE )
set(absl_container_common_DEPENDENCIES_RELEASE absl::type_traits)
set(absl_container_common_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT container_common FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_container_common_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_container_common_FRAMEWORKS_FOUND_RELEASE "${absl_container_common_FRAMEWORKS_RELEASE}" "${absl_container_common_FRAMEWORK_DIRS_RELEASE}")

set(absl_container_common_LIB_TARGETS_RELEASE "")
set(absl_container_common_NOT_USED_RELEASE "")
set(absl_container_common_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_container_common_FRAMEWORKS_FOUND_RELEASE} ${absl_container_common_SYSTEM_LIBS_RELEASE} ${absl_container_common_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_container_common_LIBS_RELEASE}"
                              "${absl_container_common_LIB_DIRS_RELEASE}"
                              "${absl_container_common_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_container_common_NOT_USED_RELEASE
                              absl_container_common_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_container_common")

set(absl_container_common_LINK_LIBS_RELEASE ${absl_container_common_LIB_TARGETS_RELEASE} ${absl_container_common_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT hashtable_debug_hooks VARIABLES #############################################

set(absl_hashtable_debug_hooks_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtable_debug_hooks_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtable_debug_hooks_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtable_debug_hooks_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_hashtable_debug_hooks_RES_DIRS_RELEASE )
set(absl_hashtable_debug_hooks_DEFINITIONS_RELEASE )
set(absl_hashtable_debug_hooks_COMPILE_DEFINITIONS_RELEASE )
set(absl_hashtable_debug_hooks_COMPILE_OPTIONS_C_RELEASE "")
set(absl_hashtable_debug_hooks_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_hashtable_debug_hooks_LIBS_RELEASE )
set(absl_hashtable_debug_hooks_SYSTEM_LIBS_RELEASE )
set(absl_hashtable_debug_hooks_FRAMEWORK_DIRS_RELEASE )
set(absl_hashtable_debug_hooks_FRAMEWORKS_RELEASE )
set(absl_hashtable_debug_hooks_BUILD_MODULES_PATHS_RELEASE )
set(absl_hashtable_debug_hooks_DEPENDENCIES_RELEASE absl::config)
set(absl_hashtable_debug_hooks_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT hashtable_debug_hooks FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_hashtable_debug_hooks_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_hashtable_debug_hooks_FRAMEWORKS_FOUND_RELEASE "${absl_hashtable_debug_hooks_FRAMEWORKS_RELEASE}" "${absl_hashtable_debug_hooks_FRAMEWORK_DIRS_RELEASE}")

set(absl_hashtable_debug_hooks_LIB_TARGETS_RELEASE "")
set(absl_hashtable_debug_hooks_NOT_USED_RELEASE "")
set(absl_hashtable_debug_hooks_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_hashtable_debug_hooks_FRAMEWORKS_FOUND_RELEASE} ${absl_hashtable_debug_hooks_SYSTEM_LIBS_RELEASE} ${absl_hashtable_debug_hooks_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_hashtable_debug_hooks_LIBS_RELEASE}"
                              "${absl_hashtable_debug_hooks_LIB_DIRS_RELEASE}"
                              "${absl_hashtable_debug_hooks_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_hashtable_debug_hooks_NOT_USED_RELEASE
                              absl_hashtable_debug_hooks_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_hashtable_debug_hooks")

set(absl_hashtable_debug_hooks_LINK_LIBS_RELEASE ${absl_hashtable_debug_hooks_LIB_TARGETS_RELEASE} ${absl_hashtable_debug_hooks_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT hashtablez_sampler VARIABLES #############################################

set(absl_hashtablez_sampler_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtablez_sampler_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtablez_sampler_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtablez_sampler_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_hashtablez_sampler_RES_DIRS_RELEASE )
set(absl_hashtablez_sampler_DEFINITIONS_RELEASE )
set(absl_hashtablez_sampler_COMPILE_DEFINITIONS_RELEASE )
set(absl_hashtablez_sampler_COMPILE_OPTIONS_C_RELEASE "")
set(absl_hashtablez_sampler_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_hashtablez_sampler_LIBS_RELEASE absl_hashtablez_sampler)
set(absl_hashtablez_sampler_SYSTEM_LIBS_RELEASE )
set(absl_hashtablez_sampler_FRAMEWORK_DIRS_RELEASE )
set(absl_hashtablez_sampler_FRAMEWORKS_RELEASE )
set(absl_hashtablez_sampler_BUILD_MODULES_PATHS_RELEASE )
set(absl_hashtablez_sampler_DEPENDENCIES_RELEASE absl::base absl::config absl::exponential_biased absl::sample_recorder absl::synchronization)
set(absl_hashtablez_sampler_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT hashtablez_sampler FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_hashtablez_sampler_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_hashtablez_sampler_FRAMEWORKS_FOUND_RELEASE "${absl_hashtablez_sampler_FRAMEWORKS_RELEASE}" "${absl_hashtablez_sampler_FRAMEWORK_DIRS_RELEASE}")

set(absl_hashtablez_sampler_LIB_TARGETS_RELEASE "")
set(absl_hashtablez_sampler_NOT_USED_RELEASE "")
set(absl_hashtablez_sampler_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_hashtablez_sampler_FRAMEWORKS_FOUND_RELEASE} ${absl_hashtablez_sampler_SYSTEM_LIBS_RELEASE} ${absl_hashtablez_sampler_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_hashtablez_sampler_LIBS_RELEASE}"
                              "${absl_hashtablez_sampler_LIB_DIRS_RELEASE}"
                              "${absl_hashtablez_sampler_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_hashtablez_sampler_NOT_USED_RELEASE
                              absl_hashtablez_sampler_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_hashtablez_sampler")

set(absl_hashtablez_sampler_LINK_LIBS_RELEASE ${absl_hashtablez_sampler_LIB_TARGETS_RELEASE} ${absl_hashtablez_sampler_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT common_policy_traits VARIABLES #############################################

set(absl_common_policy_traits_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_common_policy_traits_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_common_policy_traits_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_common_policy_traits_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_common_policy_traits_RES_DIRS_RELEASE )
set(absl_common_policy_traits_DEFINITIONS_RELEASE )
set(absl_common_policy_traits_COMPILE_DEFINITIONS_RELEASE )
set(absl_common_policy_traits_COMPILE_OPTIONS_C_RELEASE "")
set(absl_common_policy_traits_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_common_policy_traits_LIBS_RELEASE )
set(absl_common_policy_traits_SYSTEM_LIBS_RELEASE )
set(absl_common_policy_traits_FRAMEWORK_DIRS_RELEASE )
set(absl_common_policy_traits_FRAMEWORKS_RELEASE )
set(absl_common_policy_traits_BUILD_MODULES_PATHS_RELEASE )
set(absl_common_policy_traits_DEPENDENCIES_RELEASE absl::meta)
set(absl_common_policy_traits_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT common_policy_traits FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_common_policy_traits_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_common_policy_traits_FRAMEWORKS_FOUND_RELEASE "${absl_common_policy_traits_FRAMEWORKS_RELEASE}" "${absl_common_policy_traits_FRAMEWORK_DIRS_RELEASE}")

set(absl_common_policy_traits_LIB_TARGETS_RELEASE "")
set(absl_common_policy_traits_NOT_USED_RELEASE "")
set(absl_common_policy_traits_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_common_policy_traits_FRAMEWORKS_FOUND_RELEASE} ${absl_common_policy_traits_SYSTEM_LIBS_RELEASE} ${absl_common_policy_traits_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_common_policy_traits_LIBS_RELEASE}"
                              "${absl_common_policy_traits_LIB_DIRS_RELEASE}"
                              "${absl_common_policy_traits_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_common_policy_traits_NOT_USED_RELEASE
                              absl_common_policy_traits_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_common_policy_traits")

set(absl_common_policy_traits_LINK_LIBS_RELEASE ${absl_common_policy_traits_LIB_TARGETS_RELEASE} ${absl_common_policy_traits_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT hash_policy_traits VARIABLES #############################################

set(absl_hash_policy_traits_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_policy_traits_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_policy_traits_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_policy_traits_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_hash_policy_traits_RES_DIRS_RELEASE )
set(absl_hash_policy_traits_DEFINITIONS_RELEASE )
set(absl_hash_policy_traits_COMPILE_DEFINITIONS_RELEASE )
set(absl_hash_policy_traits_COMPILE_OPTIONS_C_RELEASE "")
set(absl_hash_policy_traits_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_hash_policy_traits_LIBS_RELEASE )
set(absl_hash_policy_traits_SYSTEM_LIBS_RELEASE )
set(absl_hash_policy_traits_FRAMEWORK_DIRS_RELEASE )
set(absl_hash_policy_traits_FRAMEWORKS_RELEASE )
set(absl_hash_policy_traits_BUILD_MODULES_PATHS_RELEASE )
set(absl_hash_policy_traits_DEPENDENCIES_RELEASE absl::common_policy_traits absl::meta)
set(absl_hash_policy_traits_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT hash_policy_traits FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_hash_policy_traits_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_hash_policy_traits_FRAMEWORKS_FOUND_RELEASE "${absl_hash_policy_traits_FRAMEWORKS_RELEASE}" "${absl_hash_policy_traits_FRAMEWORK_DIRS_RELEASE}")

set(absl_hash_policy_traits_LIB_TARGETS_RELEASE "")
set(absl_hash_policy_traits_NOT_USED_RELEASE "")
set(absl_hash_policy_traits_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_hash_policy_traits_FRAMEWORKS_FOUND_RELEASE} ${absl_hash_policy_traits_SYSTEM_LIBS_RELEASE} ${absl_hash_policy_traits_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_hash_policy_traits_LIBS_RELEASE}"
                              "${absl_hash_policy_traits_LIB_DIRS_RELEASE}"
                              "${absl_hash_policy_traits_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_hash_policy_traits_NOT_USED_RELEASE
                              absl_hash_policy_traits_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_hash_policy_traits")

set(absl_hash_policy_traits_LINK_LIBS_RELEASE ${absl_hash_policy_traits_LIB_TARGETS_RELEASE} ${absl_hash_policy_traits_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT raw_hash_set VARIABLES #############################################

set(absl_raw_hash_set_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_hash_set_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_hash_set_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_hash_set_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_raw_hash_set_RES_DIRS_RELEASE )
set(absl_raw_hash_set_DEFINITIONS_RELEASE )
set(absl_raw_hash_set_COMPILE_DEFINITIONS_RELEASE )
set(absl_raw_hash_set_COMPILE_OPTIONS_C_RELEASE "")
set(absl_raw_hash_set_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_raw_hash_set_LIBS_RELEASE absl_raw_hash_set)
set(absl_raw_hash_set_SYSTEM_LIBS_RELEASE )
set(absl_raw_hash_set_FRAMEWORK_DIRS_RELEASE )
set(absl_raw_hash_set_FRAMEWORKS_RELEASE )
set(absl_raw_hash_set_BUILD_MODULES_PATHS_RELEASE )
set(absl_raw_hash_set_DEPENDENCIES_RELEASE absl::bits absl::compressed_tuple absl::config absl::container_common absl::container_memory absl::core_headers absl::endian absl::hash_policy_traits absl::hashtable_debug_hooks absl::hashtablez_sampler absl::memory absl::meta absl::optional absl::prefetch absl::raw_logging_internal absl::utility)
set(absl_raw_hash_set_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT raw_hash_set FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_raw_hash_set_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_raw_hash_set_FRAMEWORKS_FOUND_RELEASE "${absl_raw_hash_set_FRAMEWORKS_RELEASE}" "${absl_raw_hash_set_FRAMEWORK_DIRS_RELEASE}")

set(absl_raw_hash_set_LIB_TARGETS_RELEASE "")
set(absl_raw_hash_set_NOT_USED_RELEASE "")
set(absl_raw_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_raw_hash_set_FRAMEWORKS_FOUND_RELEASE} ${absl_raw_hash_set_SYSTEM_LIBS_RELEASE} ${absl_raw_hash_set_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_raw_hash_set_LIBS_RELEASE}"
                              "${absl_raw_hash_set_LIB_DIRS_RELEASE}"
                              "${absl_raw_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_raw_hash_set_NOT_USED_RELEASE
                              absl_raw_hash_set_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_raw_hash_set")

set(absl_raw_hash_set_LINK_LIBS_RELEASE ${absl_raw_hash_set_LIB_TARGETS_RELEASE} ${absl_raw_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT raw_hash_map VARIABLES #############################################

set(absl_raw_hash_map_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_hash_map_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_hash_map_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_raw_hash_map_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_raw_hash_map_RES_DIRS_RELEASE )
set(absl_raw_hash_map_DEFINITIONS_RELEASE )
set(absl_raw_hash_map_COMPILE_DEFINITIONS_RELEASE )
set(absl_raw_hash_map_COMPILE_OPTIONS_C_RELEASE "")
set(absl_raw_hash_map_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_raw_hash_map_LIBS_RELEASE )
set(absl_raw_hash_map_SYSTEM_LIBS_RELEASE )
set(absl_raw_hash_map_FRAMEWORK_DIRS_RELEASE )
set(absl_raw_hash_map_FRAMEWORKS_RELEASE )
set(absl_raw_hash_map_BUILD_MODULES_PATHS_RELEASE )
set(absl_raw_hash_map_DEPENDENCIES_RELEASE absl::container_memory absl::raw_hash_set absl::throw_delegate)
set(absl_raw_hash_map_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT raw_hash_map FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_raw_hash_map_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_raw_hash_map_FRAMEWORKS_FOUND_RELEASE "${absl_raw_hash_map_FRAMEWORKS_RELEASE}" "${absl_raw_hash_map_FRAMEWORK_DIRS_RELEASE}")

set(absl_raw_hash_map_LIB_TARGETS_RELEASE "")
set(absl_raw_hash_map_NOT_USED_RELEASE "")
set(absl_raw_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_raw_hash_map_FRAMEWORKS_FOUND_RELEASE} ${absl_raw_hash_map_SYSTEM_LIBS_RELEASE} ${absl_raw_hash_map_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_raw_hash_map_LIBS_RELEASE}"
                              "${absl_raw_hash_map_LIB_DIRS_RELEASE}"
                              "${absl_raw_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_raw_hash_map_NOT_USED_RELEASE
                              absl_raw_hash_map_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_raw_hash_map")

set(absl_raw_hash_map_LINK_LIBS_RELEASE ${absl_raw_hash_map_LIB_TARGETS_RELEASE} ${absl_raw_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT hash_function_defaults VARIABLES #############################################

set(absl_hash_function_defaults_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_function_defaults_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_function_defaults_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hash_function_defaults_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_hash_function_defaults_RES_DIRS_RELEASE )
set(absl_hash_function_defaults_DEFINITIONS_RELEASE )
set(absl_hash_function_defaults_COMPILE_DEFINITIONS_RELEASE )
set(absl_hash_function_defaults_COMPILE_OPTIONS_C_RELEASE "")
set(absl_hash_function_defaults_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_hash_function_defaults_LIBS_RELEASE )
set(absl_hash_function_defaults_SYSTEM_LIBS_RELEASE )
set(absl_hash_function_defaults_FRAMEWORK_DIRS_RELEASE )
set(absl_hash_function_defaults_FRAMEWORKS_RELEASE )
set(absl_hash_function_defaults_BUILD_MODULES_PATHS_RELEASE )
set(absl_hash_function_defaults_DEPENDENCIES_RELEASE absl::config absl::cord absl::hash absl::strings)
set(absl_hash_function_defaults_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT hash_function_defaults FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_hash_function_defaults_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_hash_function_defaults_FRAMEWORKS_FOUND_RELEASE "${absl_hash_function_defaults_FRAMEWORKS_RELEASE}" "${absl_hash_function_defaults_FRAMEWORK_DIRS_RELEASE}")

set(absl_hash_function_defaults_LIB_TARGETS_RELEASE "")
set(absl_hash_function_defaults_NOT_USED_RELEASE "")
set(absl_hash_function_defaults_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_hash_function_defaults_FRAMEWORKS_FOUND_RELEASE} ${absl_hash_function_defaults_SYSTEM_LIBS_RELEASE} ${absl_hash_function_defaults_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_hash_function_defaults_LIBS_RELEASE}"
                              "${absl_hash_function_defaults_LIB_DIRS_RELEASE}"
                              "${absl_hash_function_defaults_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_hash_function_defaults_NOT_USED_RELEASE
                              absl_hash_function_defaults_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_hash_function_defaults")

set(absl_hash_function_defaults_LINK_LIBS_RELEASE ${absl_hash_function_defaults_LIB_TARGETS_RELEASE} ${absl_hash_function_defaults_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT algorithm_container VARIABLES #############################################

set(absl_algorithm_container_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_algorithm_container_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_algorithm_container_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_algorithm_container_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_algorithm_container_RES_DIRS_RELEASE )
set(absl_algorithm_container_DEFINITIONS_RELEASE )
set(absl_algorithm_container_COMPILE_DEFINITIONS_RELEASE )
set(absl_algorithm_container_COMPILE_OPTIONS_C_RELEASE "")
set(absl_algorithm_container_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_algorithm_container_LIBS_RELEASE )
set(absl_algorithm_container_SYSTEM_LIBS_RELEASE )
set(absl_algorithm_container_FRAMEWORK_DIRS_RELEASE )
set(absl_algorithm_container_FRAMEWORKS_RELEASE )
set(absl_algorithm_container_BUILD_MODULES_PATHS_RELEASE )
set(absl_algorithm_container_DEPENDENCIES_RELEASE absl::algorithm absl::core_headers absl::meta)
set(absl_algorithm_container_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT algorithm_container FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_algorithm_container_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_algorithm_container_FRAMEWORKS_FOUND_RELEASE "${absl_algorithm_container_FRAMEWORKS_RELEASE}" "${absl_algorithm_container_FRAMEWORK_DIRS_RELEASE}")

set(absl_algorithm_container_LIB_TARGETS_RELEASE "")
set(absl_algorithm_container_NOT_USED_RELEASE "")
set(absl_algorithm_container_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_algorithm_container_FRAMEWORKS_FOUND_RELEASE} ${absl_algorithm_container_SYSTEM_LIBS_RELEASE} ${absl_algorithm_container_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_algorithm_container_LIBS_RELEASE}"
                              "${absl_algorithm_container_LIB_DIRS_RELEASE}"
                              "${absl_algorithm_container_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_algorithm_container_NOT_USED_RELEASE
                              absl_algorithm_container_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_algorithm_container")

set(absl_algorithm_container_LINK_LIBS_RELEASE ${absl_algorithm_container_LIB_TARGETS_RELEASE} ${absl_algorithm_container_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flat_hash_map VARIABLES #############################################

set(absl_flat_hash_map_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flat_hash_map_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flat_hash_map_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flat_hash_map_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flat_hash_map_RES_DIRS_RELEASE )
set(absl_flat_hash_map_DEFINITIONS_RELEASE )
set(absl_flat_hash_map_COMPILE_DEFINITIONS_RELEASE )
set(absl_flat_hash_map_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flat_hash_map_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flat_hash_map_LIBS_RELEASE )
set(absl_flat_hash_map_SYSTEM_LIBS_RELEASE )
set(absl_flat_hash_map_FRAMEWORK_DIRS_RELEASE )
set(absl_flat_hash_map_FRAMEWORKS_RELEASE )
set(absl_flat_hash_map_BUILD_MODULES_PATHS_RELEASE )
set(absl_flat_hash_map_DEPENDENCIES_RELEASE absl::container_memory absl::core_headers absl::hash_function_defaults absl::raw_hash_map absl::algorithm_container absl::memory)
set(absl_flat_hash_map_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flat_hash_map FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flat_hash_map_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flat_hash_map_FRAMEWORKS_FOUND_RELEASE "${absl_flat_hash_map_FRAMEWORKS_RELEASE}" "${absl_flat_hash_map_FRAMEWORK_DIRS_RELEASE}")

set(absl_flat_hash_map_LIB_TARGETS_RELEASE "")
set(absl_flat_hash_map_NOT_USED_RELEASE "")
set(absl_flat_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flat_hash_map_FRAMEWORKS_FOUND_RELEASE} ${absl_flat_hash_map_SYSTEM_LIBS_RELEASE} ${absl_flat_hash_map_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flat_hash_map_LIBS_RELEASE}"
                              "${absl_flat_hash_map_LIB_DIRS_RELEASE}"
                              "${absl_flat_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flat_hash_map_NOT_USED_RELEASE
                              absl_flat_hash_map_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flat_hash_map")

set(absl_flat_hash_map_LINK_LIBS_RELEASE ${absl_flat_hash_map_LIB_TARGETS_RELEASE} ${absl_flat_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_reflection VARIABLES #############################################

set(absl_flags_reflection_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_reflection_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_reflection_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_reflection_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_reflection_RES_DIRS_RELEASE )
set(absl_flags_reflection_DEFINITIONS_RELEASE )
set(absl_flags_reflection_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_reflection_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_reflection_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_reflection_LIBS_RELEASE absl_flags_reflection)
set(absl_flags_reflection_SYSTEM_LIBS_RELEASE )
set(absl_flags_reflection_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_reflection_FRAMEWORKS_RELEASE )
set(absl_flags_reflection_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_reflection_DEPENDENCIES_RELEASE absl::config absl::flags_commandlineflag absl::flags_private_handle_accessor absl::flags_config absl::strings absl::synchronization absl::flat_hash_map)
set(absl_flags_reflection_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_reflection FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_reflection_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_reflection_FRAMEWORKS_FOUND_RELEASE "${absl_flags_reflection_FRAMEWORKS_RELEASE}" "${absl_flags_reflection_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_reflection_LIB_TARGETS_RELEASE "")
set(absl_flags_reflection_NOT_USED_RELEASE "")
set(absl_flags_reflection_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_reflection_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_reflection_SYSTEM_LIBS_RELEASE} ${absl_flags_reflection_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_reflection_LIBS_RELEASE}"
                              "${absl_flags_reflection_LIB_DIRS_RELEASE}"
                              "${absl_flags_reflection_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_reflection_NOT_USED_RELEASE
                              absl_flags_reflection_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_reflection")

set(absl_flags_reflection_LINK_LIBS_RELEASE ${absl_flags_reflection_LIB_TARGETS_RELEASE} ${absl_flags_reflection_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags VARIABLES #############################################

set(absl_flags_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_RES_DIRS_RELEASE )
set(absl_flags_DEFINITIONS_RELEASE )
set(absl_flags_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_LIBS_RELEASE absl_flags)
set(absl_flags_SYSTEM_LIBS_RELEASE )
set(absl_flags_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_FRAMEWORKS_RELEASE )
set(absl_flags_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_DEPENDENCIES_RELEASE absl::config absl::flags_commandlineflag absl::flags_config absl::flags_internal absl::flags_reflection absl::base absl::core_headers absl::strings)
set(absl_flags_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_FRAMEWORKS_FOUND_RELEASE "${absl_flags_FRAMEWORKS_RELEASE}" "${absl_flags_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_LIB_TARGETS_RELEASE "")
set(absl_flags_NOT_USED_RELEASE "")
set(absl_flags_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_SYSTEM_LIBS_RELEASE} ${absl_flags_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_LIBS_RELEASE}"
                              "${absl_flags_LIB_DIRS_RELEASE}"
                              "${absl_flags_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_NOT_USED_RELEASE
                              absl_flags_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags")

set(absl_flags_LINK_LIBS_RELEASE ${absl_flags_LIB_TARGETS_RELEASE} ${absl_flags_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_flags VARIABLES #############################################

set(absl_log_internal_flags_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_flags_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_flags_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_flags_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_flags_RES_DIRS_RELEASE )
set(absl_log_internal_flags_DEFINITIONS_RELEASE )
set(absl_log_internal_flags_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_flags_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_flags_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_flags_LIBS_RELEASE )
set(absl_log_internal_flags_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_flags_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_flags_FRAMEWORKS_RELEASE )
set(absl_log_internal_flags_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_flags_DEPENDENCIES_RELEASE absl::flags)
set(absl_log_internal_flags_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_flags FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_flags_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_flags_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_flags_FRAMEWORKS_RELEASE}" "${absl_log_internal_flags_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_flags_LIB_TARGETS_RELEASE "")
set(absl_log_internal_flags_NOT_USED_RELEASE "")
set(absl_log_internal_flags_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_flags_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_flags_SYSTEM_LIBS_RELEASE} ${absl_log_internal_flags_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_flags_LIBS_RELEASE}"
                              "${absl_log_internal_flags_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_flags_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_flags_NOT_USED_RELEASE
                              absl_log_internal_flags_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_flags")

set(absl_log_internal_flags_LINK_LIBS_RELEASE ${absl_log_internal_flags_LIB_TARGETS_RELEASE} ${absl_log_internal_flags_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_flags VARIABLES #############################################

set(absl_log_flags_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_flags_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_flags_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_flags_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_flags_RES_DIRS_RELEASE )
set(absl_log_flags_DEFINITIONS_RELEASE )
set(absl_log_flags_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_flags_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_flags_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_flags_LIBS_RELEASE absl_log_flags)
set(absl_log_flags_SYSTEM_LIBS_RELEASE )
set(absl_log_flags_FRAMEWORK_DIRS_RELEASE )
set(absl_log_flags_FRAMEWORKS_RELEASE )
set(absl_log_flags_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_flags_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_globals absl::log_severity absl::log_internal_config absl::log_internal_flags absl::flags absl::flags_marshalling absl::strings)
set(absl_log_flags_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_flags FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_flags_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_flags_FRAMEWORKS_FOUND_RELEASE "${absl_log_flags_FRAMEWORKS_RELEASE}" "${absl_log_flags_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_flags_LIB_TARGETS_RELEASE "")
set(absl_log_flags_NOT_USED_RELEASE "")
set(absl_log_flags_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_flags_FRAMEWORKS_FOUND_RELEASE} ${absl_log_flags_SYSTEM_LIBS_RELEASE} ${absl_log_flags_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_flags_LIBS_RELEASE}"
                              "${absl_log_flags_LIB_DIRS_RELEASE}"
                              "${absl_log_flags_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_flags_NOT_USED_RELEASE
                              absl_log_flags_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_flags")

set(absl_log_flags_LINK_LIBS_RELEASE ${absl_log_flags_LIB_TARGETS_RELEASE} ${absl_log_flags_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT die_if_null VARIABLES #############################################

set(absl_die_if_null_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_die_if_null_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_die_if_null_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_die_if_null_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_die_if_null_RES_DIRS_RELEASE )
set(absl_die_if_null_DEFINITIONS_RELEASE )
set(absl_die_if_null_COMPILE_DEFINITIONS_RELEASE )
set(absl_die_if_null_COMPILE_OPTIONS_C_RELEASE "")
set(absl_die_if_null_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_die_if_null_LIBS_RELEASE absl_die_if_null)
set(absl_die_if_null_SYSTEM_LIBS_RELEASE )
set(absl_die_if_null_FRAMEWORK_DIRS_RELEASE )
set(absl_die_if_null_FRAMEWORKS_RELEASE )
set(absl_die_if_null_BUILD_MODULES_PATHS_RELEASE )
set(absl_die_if_null_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log absl::strings)
set(absl_die_if_null_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT die_if_null FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_die_if_null_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_die_if_null_FRAMEWORKS_FOUND_RELEASE "${absl_die_if_null_FRAMEWORKS_RELEASE}" "${absl_die_if_null_FRAMEWORK_DIRS_RELEASE}")

set(absl_die_if_null_LIB_TARGETS_RELEASE "")
set(absl_die_if_null_NOT_USED_RELEASE "")
set(absl_die_if_null_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_die_if_null_FRAMEWORKS_FOUND_RELEASE} ${absl_die_if_null_SYSTEM_LIBS_RELEASE} ${absl_die_if_null_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_die_if_null_LIBS_RELEASE}"
                              "${absl_die_if_null_LIB_DIRS_RELEASE}"
                              "${absl_die_if_null_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_die_if_null_NOT_USED_RELEASE
                              absl_die_if_null_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_die_if_null")

set(absl_die_if_null_LINK_LIBS_RELEASE ${absl_die_if_null_LIB_TARGETS_RELEASE} ${absl_die_if_null_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_check_op VARIABLES #############################################

set(absl_log_internal_check_op_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_check_op_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_check_op_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_check_op_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_check_op_RES_DIRS_RELEASE )
set(absl_log_internal_check_op_DEFINITIONS_RELEASE )
set(absl_log_internal_check_op_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_check_op_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_check_op_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_check_op_LIBS_RELEASE absl_log_internal_check_op)
set(absl_log_internal_check_op_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_check_op_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_check_op_FRAMEWORKS_RELEASE )
set(absl_log_internal_check_op_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_check_op_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::log_internal_nullguard absl::log_internal_nullstream absl::log_internal_strip absl::strings)
set(absl_log_internal_check_op_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_check_op FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_check_op_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_check_op_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_check_op_FRAMEWORKS_RELEASE}" "${absl_log_internal_check_op_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_check_op_LIB_TARGETS_RELEASE "")
set(absl_log_internal_check_op_NOT_USED_RELEASE "")
set(absl_log_internal_check_op_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_check_op_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_check_op_SYSTEM_LIBS_RELEASE} ${absl_log_internal_check_op_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_check_op_LIBS_RELEASE}"
                              "${absl_log_internal_check_op_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_check_op_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_check_op_NOT_USED_RELEASE
                              absl_log_internal_check_op_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_check_op")

set(absl_log_internal_check_op_LINK_LIBS_RELEASE ${absl_log_internal_check_op_LIB_TARGETS_RELEASE} ${absl_log_internal_check_op_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT log_internal_check_impl VARIABLES #############################################

set(absl_log_internal_check_impl_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_check_impl_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_check_impl_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_log_internal_check_impl_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_log_internal_check_impl_RES_DIRS_RELEASE )
set(absl_log_internal_check_impl_DEFINITIONS_RELEASE )
set(absl_log_internal_check_impl_COMPILE_DEFINITIONS_RELEASE )
set(absl_log_internal_check_impl_COMPILE_OPTIONS_C_RELEASE "")
set(absl_log_internal_check_impl_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_log_internal_check_impl_LIBS_RELEASE )
set(absl_log_internal_check_impl_SYSTEM_LIBS_RELEASE )
set(absl_log_internal_check_impl_FRAMEWORK_DIRS_RELEASE )
set(absl_log_internal_check_impl_FRAMEWORKS_RELEASE )
set(absl_log_internal_check_impl_BUILD_MODULES_PATHS_RELEASE )
set(absl_log_internal_check_impl_DEPENDENCIES_RELEASE absl::core_headers absl::log_internal_check_op absl::log_internal_conditions absl::log_internal_message absl::log_internal_strip)
set(absl_log_internal_check_impl_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT log_internal_check_impl FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_log_internal_check_impl_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_log_internal_check_impl_FRAMEWORKS_FOUND_RELEASE "${absl_log_internal_check_impl_FRAMEWORKS_RELEASE}" "${absl_log_internal_check_impl_FRAMEWORK_DIRS_RELEASE}")

set(absl_log_internal_check_impl_LIB_TARGETS_RELEASE "")
set(absl_log_internal_check_impl_NOT_USED_RELEASE "")
set(absl_log_internal_check_impl_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_log_internal_check_impl_FRAMEWORKS_FOUND_RELEASE} ${absl_log_internal_check_impl_SYSTEM_LIBS_RELEASE} ${absl_log_internal_check_impl_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_log_internal_check_impl_LIBS_RELEASE}"
                              "${absl_log_internal_check_impl_LIB_DIRS_RELEASE}"
                              "${absl_log_internal_check_impl_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_log_internal_check_impl_NOT_USED_RELEASE
                              absl_log_internal_check_impl_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_log_internal_check_impl")

set(absl_log_internal_check_impl_LINK_LIBS_RELEASE ${absl_log_internal_check_impl_LIB_TARGETS_RELEASE} ${absl_log_internal_check_impl_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT check VARIABLES #############################################

set(absl_check_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_check_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_check_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_check_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_check_RES_DIRS_RELEASE )
set(absl_check_DEFINITIONS_RELEASE )
set(absl_check_COMPILE_DEFINITIONS_RELEASE )
set(absl_check_COMPILE_OPTIONS_C_RELEASE "")
set(absl_check_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_check_LIBS_RELEASE )
set(absl_check_SYSTEM_LIBS_RELEASE )
set(absl_check_FRAMEWORK_DIRS_RELEASE )
set(absl_check_FRAMEWORKS_RELEASE )
set(absl_check_BUILD_MODULES_PATHS_RELEASE )
set(absl_check_DEPENDENCIES_RELEASE absl::log_internal_check_impl absl::core_headers absl::log_internal_check_op absl::log_internal_conditions absl::log_internal_message absl::log_internal_strip)
set(absl_check_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT check FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_check_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_check_FRAMEWORKS_FOUND_RELEASE "${absl_check_FRAMEWORKS_RELEASE}" "${absl_check_FRAMEWORK_DIRS_RELEASE}")

set(absl_check_LIB_TARGETS_RELEASE "")
set(absl_check_NOT_USED_RELEASE "")
set(absl_check_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_check_FRAMEWORKS_FOUND_RELEASE} ${absl_check_SYSTEM_LIBS_RELEASE} ${absl_check_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_check_LIBS_RELEASE}"
                              "${absl_check_LIB_DIRS_RELEASE}"
                              "${absl_check_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_check_NOT_USED_RELEASE
                              absl_check_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_check")

set(absl_check_LINK_LIBS_RELEASE ${absl_check_LIB_TARGETS_RELEASE} ${absl_check_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT absl_check VARIABLES #############################################

set(absl_absl_check_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_absl_check_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_absl_check_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_absl_check_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_absl_check_RES_DIRS_RELEASE )
set(absl_absl_check_DEFINITIONS_RELEASE )
set(absl_absl_check_COMPILE_DEFINITIONS_RELEASE )
set(absl_absl_check_COMPILE_OPTIONS_C_RELEASE "")
set(absl_absl_check_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_absl_check_LIBS_RELEASE )
set(absl_absl_check_SYSTEM_LIBS_RELEASE )
set(absl_absl_check_FRAMEWORK_DIRS_RELEASE )
set(absl_absl_check_FRAMEWORKS_RELEASE )
set(absl_absl_check_BUILD_MODULES_PATHS_RELEASE )
set(absl_absl_check_DEPENDENCIES_RELEASE absl::log_internal_check_impl)
set(absl_absl_check_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT absl_check FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_absl_check_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_absl_check_FRAMEWORKS_FOUND_RELEASE "${absl_absl_check_FRAMEWORKS_RELEASE}" "${absl_absl_check_FRAMEWORK_DIRS_RELEASE}")

set(absl_absl_check_LIB_TARGETS_RELEASE "")
set(absl_absl_check_NOT_USED_RELEASE "")
set(absl_absl_check_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_absl_check_FRAMEWORKS_FOUND_RELEASE} ${absl_absl_check_SYSTEM_LIBS_RELEASE} ${absl_absl_check_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_absl_check_LIBS_RELEASE}"
                              "${absl_absl_check_LIB_DIRS_RELEASE}"
                              "${absl_absl_check_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_absl_check_NOT_USED_RELEASE
                              absl_absl_check_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_absl_check")

set(absl_absl_check_LINK_LIBS_RELEASE ${absl_absl_check_LIB_TARGETS_RELEASE} ${absl_absl_check_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT bind_front VARIABLES #############################################

set(absl_bind_front_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bind_front_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bind_front_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_bind_front_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_bind_front_RES_DIRS_RELEASE )
set(absl_bind_front_DEFINITIONS_RELEASE )
set(absl_bind_front_COMPILE_DEFINITIONS_RELEASE )
set(absl_bind_front_COMPILE_OPTIONS_C_RELEASE "")
set(absl_bind_front_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_bind_front_LIBS_RELEASE )
set(absl_bind_front_SYSTEM_LIBS_RELEASE )
set(absl_bind_front_FRAMEWORK_DIRS_RELEASE )
set(absl_bind_front_FRAMEWORKS_RELEASE )
set(absl_bind_front_BUILD_MODULES_PATHS_RELEASE )
set(absl_bind_front_DEPENDENCIES_RELEASE absl::base_internal absl::compressed_tuple)
set(absl_bind_front_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT bind_front FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_bind_front_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_bind_front_FRAMEWORKS_FOUND_RELEASE "${absl_bind_front_FRAMEWORKS_RELEASE}" "${absl_bind_front_FRAMEWORK_DIRS_RELEASE}")

set(absl_bind_front_LIB_TARGETS_RELEASE "")
set(absl_bind_front_NOT_USED_RELEASE "")
set(absl_bind_front_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_bind_front_FRAMEWORKS_FOUND_RELEASE} ${absl_bind_front_SYSTEM_LIBS_RELEASE} ${absl_bind_front_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_bind_front_LIBS_RELEASE}"
                              "${absl_bind_front_LIB_DIRS_RELEASE}"
                              "${absl_bind_front_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_bind_front_NOT_USED_RELEASE
                              absl_bind_front_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_bind_front")

set(absl_bind_front_LINK_LIBS_RELEASE ${absl_bind_front_LIB_TARGETS_RELEASE} ${absl_bind_front_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT any_invocable VARIABLES #############################################

set(absl_any_invocable_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_any_invocable_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_any_invocable_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_any_invocable_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_any_invocable_RES_DIRS_RELEASE )
set(absl_any_invocable_DEFINITIONS_RELEASE )
set(absl_any_invocable_COMPILE_DEFINITIONS_RELEASE )
set(absl_any_invocable_COMPILE_OPTIONS_C_RELEASE "")
set(absl_any_invocable_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_any_invocable_LIBS_RELEASE )
set(absl_any_invocable_SYSTEM_LIBS_RELEASE )
set(absl_any_invocable_FRAMEWORK_DIRS_RELEASE )
set(absl_any_invocable_FRAMEWORKS_RELEASE )
set(absl_any_invocable_BUILD_MODULES_PATHS_RELEASE )
set(absl_any_invocable_DEPENDENCIES_RELEASE absl::base_internal absl::config absl::core_headers absl::type_traits absl::utility)
set(absl_any_invocable_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT any_invocable FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_any_invocable_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_any_invocable_FRAMEWORKS_FOUND_RELEASE "${absl_any_invocable_FRAMEWORKS_RELEASE}" "${absl_any_invocable_FRAMEWORK_DIRS_RELEASE}")

set(absl_any_invocable_LIB_TARGETS_RELEASE "")
set(absl_any_invocable_NOT_USED_RELEASE "")
set(absl_any_invocable_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_any_invocable_FRAMEWORKS_FOUND_RELEASE} ${absl_any_invocable_SYSTEM_LIBS_RELEASE} ${absl_any_invocable_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_any_invocable_LIBS_RELEASE}"
                              "${absl_any_invocable_LIB_DIRS_RELEASE}"
                              "${absl_any_invocable_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_any_invocable_NOT_USED_RELEASE
                              absl_any_invocable_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_any_invocable")

set(absl_any_invocable_LINK_LIBS_RELEASE ${absl_any_invocable_LIB_TARGETS_RELEASE} ${absl_any_invocable_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_usage_internal VARIABLES #############################################

set(absl_flags_usage_internal_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_usage_internal_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_usage_internal_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_usage_internal_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_usage_internal_RES_DIRS_RELEASE )
set(absl_flags_usage_internal_DEFINITIONS_RELEASE )
set(absl_flags_usage_internal_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_usage_internal_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_usage_internal_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_usage_internal_LIBS_RELEASE absl_flags_usage_internal)
set(absl_flags_usage_internal_SYSTEM_LIBS_RELEASE )
set(absl_flags_usage_internal_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_usage_internal_FRAMEWORKS_RELEASE )
set(absl_flags_usage_internal_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_usage_internal_DEPENDENCIES_RELEASE absl::config absl::flags_config absl::flags absl::flags_commandlineflag absl::flags_internal absl::flags_path_util absl::flags_private_handle_accessor absl::flags_program_name absl::flags_reflection absl::flat_hash_map absl::strings absl::synchronization)
set(absl_flags_usage_internal_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_usage_internal FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_usage_internal_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_usage_internal_FRAMEWORKS_FOUND_RELEASE "${absl_flags_usage_internal_FRAMEWORKS_RELEASE}" "${absl_flags_usage_internal_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_usage_internal_LIB_TARGETS_RELEASE "")
set(absl_flags_usage_internal_NOT_USED_RELEASE "")
set(absl_flags_usage_internal_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_usage_internal_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_usage_internal_SYSTEM_LIBS_RELEASE} ${absl_flags_usage_internal_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_usage_internal_LIBS_RELEASE}"
                              "${absl_flags_usage_internal_LIB_DIRS_RELEASE}"
                              "${absl_flags_usage_internal_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_usage_internal_NOT_USED_RELEASE
                              absl_flags_usage_internal_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_usage_internal")

set(absl_flags_usage_internal_LINK_LIBS_RELEASE ${absl_flags_usage_internal_LIB_TARGETS_RELEASE} ${absl_flags_usage_internal_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_usage VARIABLES #############################################

set(absl_flags_usage_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_usage_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_usage_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_usage_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_usage_RES_DIRS_RELEASE )
set(absl_flags_usage_DEFINITIONS_RELEASE )
set(absl_flags_usage_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_usage_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_usage_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_usage_LIBS_RELEASE absl_flags_usage)
set(absl_flags_usage_SYSTEM_LIBS_RELEASE )
set(absl_flags_usage_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_usage_FRAMEWORKS_RELEASE )
set(absl_flags_usage_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_usage_DEPENDENCIES_RELEASE absl::config absl::core_headers absl::flags_usage_internal absl::strings absl::synchronization)
set(absl_flags_usage_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_usage FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_usage_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_usage_FRAMEWORKS_FOUND_RELEASE "${absl_flags_usage_FRAMEWORKS_RELEASE}" "${absl_flags_usage_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_usage_LIB_TARGETS_RELEASE "")
set(absl_flags_usage_NOT_USED_RELEASE "")
set(absl_flags_usage_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_usage_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_usage_SYSTEM_LIBS_RELEASE} ${absl_flags_usage_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_usage_LIBS_RELEASE}"
                              "${absl_flags_usage_LIB_DIRS_RELEASE}"
                              "${absl_flags_usage_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_usage_NOT_USED_RELEASE
                              absl_flags_usage_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_usage")

set(absl_flags_usage_LINK_LIBS_RELEASE ${absl_flags_usage_LIB_TARGETS_RELEASE} ${absl_flags_usage_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flags_parse VARIABLES #############################################

set(absl_flags_parse_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_parse_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_parse_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flags_parse_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flags_parse_RES_DIRS_RELEASE )
set(absl_flags_parse_DEFINITIONS_RELEASE )
set(absl_flags_parse_COMPILE_DEFINITIONS_RELEASE )
set(absl_flags_parse_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flags_parse_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flags_parse_LIBS_RELEASE absl_flags_parse)
set(absl_flags_parse_SYSTEM_LIBS_RELEASE )
set(absl_flags_parse_FRAMEWORK_DIRS_RELEASE )
set(absl_flags_parse_FRAMEWORKS_RELEASE )
set(absl_flags_parse_BUILD_MODULES_PATHS_RELEASE )
set(absl_flags_parse_DEPENDENCIES_RELEASE absl::algorithm_container absl::config absl::core_headers absl::flags_config absl::flags absl::flags_commandlineflag absl::flags_commandlineflag_internal absl::flags_internal absl::flags_private_handle_accessor absl::flags_program_name absl::flags_reflection absl::flags_usage absl::strings absl::synchronization)
set(absl_flags_parse_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flags_parse FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flags_parse_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flags_parse_FRAMEWORKS_FOUND_RELEASE "${absl_flags_parse_FRAMEWORKS_RELEASE}" "${absl_flags_parse_FRAMEWORK_DIRS_RELEASE}")

set(absl_flags_parse_LIB_TARGETS_RELEASE "")
set(absl_flags_parse_NOT_USED_RELEASE "")
set(absl_flags_parse_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flags_parse_FRAMEWORKS_FOUND_RELEASE} ${absl_flags_parse_SYSTEM_LIBS_RELEASE} ${absl_flags_parse_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flags_parse_LIBS_RELEASE}"
                              "${absl_flags_parse_LIB_DIRS_RELEASE}"
                              "${absl_flags_parse_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flags_parse_NOT_USED_RELEASE
                              absl_flags_parse_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flags_parse")

set(absl_flags_parse_LINK_LIBS_RELEASE ${absl_flags_parse_LIB_TARGETS_RELEASE} ${absl_flags_parse_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT leak_check VARIABLES #############################################

set(absl_leak_check_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_leak_check_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_leak_check_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_leak_check_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_leak_check_RES_DIRS_RELEASE )
set(absl_leak_check_DEFINITIONS_RELEASE )
set(absl_leak_check_COMPILE_DEFINITIONS_RELEASE )
set(absl_leak_check_COMPILE_OPTIONS_C_RELEASE "")
set(absl_leak_check_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_leak_check_LIBS_RELEASE absl_leak_check)
set(absl_leak_check_SYSTEM_LIBS_RELEASE )
set(absl_leak_check_FRAMEWORK_DIRS_RELEASE )
set(absl_leak_check_FRAMEWORKS_RELEASE )
set(absl_leak_check_BUILD_MODULES_PATHS_RELEASE )
set(absl_leak_check_DEPENDENCIES_RELEASE absl::config absl::core_headers)
set(absl_leak_check_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT leak_check FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_leak_check_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_leak_check_FRAMEWORKS_FOUND_RELEASE "${absl_leak_check_FRAMEWORKS_RELEASE}" "${absl_leak_check_FRAMEWORK_DIRS_RELEASE}")

set(absl_leak_check_LIB_TARGETS_RELEASE "")
set(absl_leak_check_NOT_USED_RELEASE "")
set(absl_leak_check_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_leak_check_FRAMEWORKS_FOUND_RELEASE} ${absl_leak_check_SYSTEM_LIBS_RELEASE} ${absl_leak_check_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_leak_check_LIBS_RELEASE}"
                              "${absl_leak_check_LIB_DIRS_RELEASE}"
                              "${absl_leak_check_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_leak_check_NOT_USED_RELEASE
                              absl_leak_check_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_leak_check")

set(absl_leak_check_LINK_LIBS_RELEASE ${absl_leak_check_LIB_TARGETS_RELEASE} ${absl_leak_check_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT debugging VARIABLES #############################################

set(absl_debugging_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_debugging_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_debugging_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_debugging_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_debugging_RES_DIRS_RELEASE )
set(absl_debugging_DEFINITIONS_RELEASE )
set(absl_debugging_COMPILE_DEFINITIONS_RELEASE )
set(absl_debugging_COMPILE_OPTIONS_C_RELEASE "")
set(absl_debugging_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_debugging_LIBS_RELEASE )
set(absl_debugging_SYSTEM_LIBS_RELEASE )
set(absl_debugging_FRAMEWORK_DIRS_RELEASE )
set(absl_debugging_FRAMEWORKS_RELEASE )
set(absl_debugging_BUILD_MODULES_PATHS_RELEASE )
set(absl_debugging_DEPENDENCIES_RELEASE absl::stacktrace absl::leak_check)
set(absl_debugging_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT debugging FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_debugging_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_debugging_FRAMEWORKS_FOUND_RELEASE "${absl_debugging_FRAMEWORKS_RELEASE}" "${absl_debugging_FRAMEWORK_DIRS_RELEASE}")

set(absl_debugging_LIB_TARGETS_RELEASE "")
set(absl_debugging_NOT_USED_RELEASE "")
set(absl_debugging_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_debugging_FRAMEWORKS_FOUND_RELEASE} ${absl_debugging_SYSTEM_LIBS_RELEASE} ${absl_debugging_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_debugging_LIBS_RELEASE}"
                              "${absl_debugging_LIB_DIRS_RELEASE}"
                              "${absl_debugging_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_debugging_NOT_USED_RELEASE
                              absl_debugging_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_debugging")

set(absl_debugging_LINK_LIBS_RELEASE ${absl_debugging_LIB_TARGETS_RELEASE} ${absl_debugging_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT failure_signal_handler VARIABLES #############################################

set(absl_failure_signal_handler_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_failure_signal_handler_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_failure_signal_handler_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_failure_signal_handler_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_failure_signal_handler_RES_DIRS_RELEASE )
set(absl_failure_signal_handler_DEFINITIONS_RELEASE )
set(absl_failure_signal_handler_COMPILE_DEFINITIONS_RELEASE )
set(absl_failure_signal_handler_COMPILE_OPTIONS_C_RELEASE "")
set(absl_failure_signal_handler_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_failure_signal_handler_LIBS_RELEASE absl_failure_signal_handler)
set(absl_failure_signal_handler_SYSTEM_LIBS_RELEASE )
set(absl_failure_signal_handler_FRAMEWORK_DIRS_RELEASE )
set(absl_failure_signal_handler_FRAMEWORKS_RELEASE )
set(absl_failure_signal_handler_BUILD_MODULES_PATHS_RELEASE )
set(absl_failure_signal_handler_DEPENDENCIES_RELEASE absl::examine_stack absl::stacktrace absl::base absl::config absl::core_headers absl::raw_logging_internal)
set(absl_failure_signal_handler_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT failure_signal_handler FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_failure_signal_handler_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_failure_signal_handler_FRAMEWORKS_FOUND_RELEASE "${absl_failure_signal_handler_FRAMEWORKS_RELEASE}" "${absl_failure_signal_handler_FRAMEWORK_DIRS_RELEASE}")

set(absl_failure_signal_handler_LIB_TARGETS_RELEASE "")
set(absl_failure_signal_handler_NOT_USED_RELEASE "")
set(absl_failure_signal_handler_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_failure_signal_handler_FRAMEWORKS_FOUND_RELEASE} ${absl_failure_signal_handler_SYSTEM_LIBS_RELEASE} ${absl_failure_signal_handler_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_failure_signal_handler_LIBS_RELEASE}"
                              "${absl_failure_signal_handler_LIB_DIRS_RELEASE}"
                              "${absl_failure_signal_handler_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_failure_signal_handler_NOT_USED_RELEASE
                              absl_failure_signal_handler_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_failure_signal_handler")

set(absl_failure_signal_handler_LINK_LIBS_RELEASE ${absl_failure_signal_handler_LIB_TARGETS_RELEASE} ${absl_failure_signal_handler_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT node_slot_policy VARIABLES #############################################

set(absl_node_slot_policy_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_slot_policy_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_slot_policy_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_slot_policy_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_node_slot_policy_RES_DIRS_RELEASE )
set(absl_node_slot_policy_DEFINITIONS_RELEASE )
set(absl_node_slot_policy_COMPILE_DEFINITIONS_RELEASE )
set(absl_node_slot_policy_COMPILE_OPTIONS_C_RELEASE "")
set(absl_node_slot_policy_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_node_slot_policy_LIBS_RELEASE )
set(absl_node_slot_policy_SYSTEM_LIBS_RELEASE )
set(absl_node_slot_policy_FRAMEWORK_DIRS_RELEASE )
set(absl_node_slot_policy_FRAMEWORKS_RELEASE )
set(absl_node_slot_policy_BUILD_MODULES_PATHS_RELEASE )
set(absl_node_slot_policy_DEPENDENCIES_RELEASE absl::config)
set(absl_node_slot_policy_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT node_slot_policy FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_node_slot_policy_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_node_slot_policy_FRAMEWORKS_FOUND_RELEASE "${absl_node_slot_policy_FRAMEWORKS_RELEASE}" "${absl_node_slot_policy_FRAMEWORK_DIRS_RELEASE}")

set(absl_node_slot_policy_LIB_TARGETS_RELEASE "")
set(absl_node_slot_policy_NOT_USED_RELEASE "")
set(absl_node_slot_policy_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_node_slot_policy_FRAMEWORKS_FOUND_RELEASE} ${absl_node_slot_policy_SYSTEM_LIBS_RELEASE} ${absl_node_slot_policy_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_node_slot_policy_LIBS_RELEASE}"
                              "${absl_node_slot_policy_LIB_DIRS_RELEASE}"
                              "${absl_node_slot_policy_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_node_slot_policy_NOT_USED_RELEASE
                              absl_node_slot_policy_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_node_slot_policy")

set(absl_node_slot_policy_LINK_LIBS_RELEASE ${absl_node_slot_policy_LIB_TARGETS_RELEASE} ${absl_node_slot_policy_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT hashtable_debug VARIABLES #############################################

set(absl_hashtable_debug_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtable_debug_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtable_debug_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_hashtable_debug_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_hashtable_debug_RES_DIRS_RELEASE )
set(absl_hashtable_debug_DEFINITIONS_RELEASE )
set(absl_hashtable_debug_COMPILE_DEFINITIONS_RELEASE )
set(absl_hashtable_debug_COMPILE_OPTIONS_C_RELEASE "")
set(absl_hashtable_debug_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_hashtable_debug_LIBS_RELEASE )
set(absl_hashtable_debug_SYSTEM_LIBS_RELEASE )
set(absl_hashtable_debug_FRAMEWORK_DIRS_RELEASE )
set(absl_hashtable_debug_FRAMEWORKS_RELEASE )
set(absl_hashtable_debug_BUILD_MODULES_PATHS_RELEASE )
set(absl_hashtable_debug_DEPENDENCIES_RELEASE absl::hashtable_debug_hooks)
set(absl_hashtable_debug_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT hashtable_debug FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_hashtable_debug_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_hashtable_debug_FRAMEWORKS_FOUND_RELEASE "${absl_hashtable_debug_FRAMEWORKS_RELEASE}" "${absl_hashtable_debug_FRAMEWORK_DIRS_RELEASE}")

set(absl_hashtable_debug_LIB_TARGETS_RELEASE "")
set(absl_hashtable_debug_NOT_USED_RELEASE "")
set(absl_hashtable_debug_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_hashtable_debug_FRAMEWORKS_FOUND_RELEASE} ${absl_hashtable_debug_SYSTEM_LIBS_RELEASE} ${absl_hashtable_debug_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_hashtable_debug_LIBS_RELEASE}"
                              "${absl_hashtable_debug_LIB_DIRS_RELEASE}"
                              "${absl_hashtable_debug_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_hashtable_debug_NOT_USED_RELEASE
                              absl_hashtable_debug_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_hashtable_debug")

set(absl_hashtable_debug_LINK_LIBS_RELEASE ${absl_hashtable_debug_LIB_TARGETS_RELEASE} ${absl_hashtable_debug_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT node_hash_set VARIABLES #############################################

set(absl_node_hash_set_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_hash_set_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_hash_set_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_hash_set_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_node_hash_set_RES_DIRS_RELEASE )
set(absl_node_hash_set_DEFINITIONS_RELEASE )
set(absl_node_hash_set_COMPILE_DEFINITIONS_RELEASE )
set(absl_node_hash_set_COMPILE_OPTIONS_C_RELEASE "")
set(absl_node_hash_set_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_node_hash_set_LIBS_RELEASE )
set(absl_node_hash_set_SYSTEM_LIBS_RELEASE )
set(absl_node_hash_set_FRAMEWORK_DIRS_RELEASE )
set(absl_node_hash_set_FRAMEWORKS_RELEASE )
set(absl_node_hash_set_BUILD_MODULES_PATHS_RELEASE )
set(absl_node_hash_set_DEPENDENCIES_RELEASE absl::core_headers absl::hash_function_defaults absl::node_slot_policy absl::raw_hash_set absl::algorithm_container absl::memory)
set(absl_node_hash_set_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT node_hash_set FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_node_hash_set_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_node_hash_set_FRAMEWORKS_FOUND_RELEASE "${absl_node_hash_set_FRAMEWORKS_RELEASE}" "${absl_node_hash_set_FRAMEWORK_DIRS_RELEASE}")

set(absl_node_hash_set_LIB_TARGETS_RELEASE "")
set(absl_node_hash_set_NOT_USED_RELEASE "")
set(absl_node_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_node_hash_set_FRAMEWORKS_FOUND_RELEASE} ${absl_node_hash_set_SYSTEM_LIBS_RELEASE} ${absl_node_hash_set_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_node_hash_set_LIBS_RELEASE}"
                              "${absl_node_hash_set_LIB_DIRS_RELEASE}"
                              "${absl_node_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_node_hash_set_NOT_USED_RELEASE
                              absl_node_hash_set_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_node_hash_set")

set(absl_node_hash_set_LINK_LIBS_RELEASE ${absl_node_hash_set_LIB_TARGETS_RELEASE} ${absl_node_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT node_hash_map VARIABLES #############################################

set(absl_node_hash_map_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_hash_map_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_hash_map_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_node_hash_map_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_node_hash_map_RES_DIRS_RELEASE )
set(absl_node_hash_map_DEFINITIONS_RELEASE )
set(absl_node_hash_map_COMPILE_DEFINITIONS_RELEASE )
set(absl_node_hash_map_COMPILE_OPTIONS_C_RELEASE "")
set(absl_node_hash_map_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_node_hash_map_LIBS_RELEASE )
set(absl_node_hash_map_SYSTEM_LIBS_RELEASE )
set(absl_node_hash_map_FRAMEWORK_DIRS_RELEASE )
set(absl_node_hash_map_FRAMEWORKS_RELEASE )
set(absl_node_hash_map_BUILD_MODULES_PATHS_RELEASE )
set(absl_node_hash_map_DEPENDENCIES_RELEASE absl::container_memory absl::core_headers absl::hash_function_defaults absl::node_slot_policy absl::raw_hash_map absl::algorithm_container absl::memory)
set(absl_node_hash_map_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT node_hash_map FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_node_hash_map_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_node_hash_map_FRAMEWORKS_FOUND_RELEASE "${absl_node_hash_map_FRAMEWORKS_RELEASE}" "${absl_node_hash_map_FRAMEWORK_DIRS_RELEASE}")

set(absl_node_hash_map_LIB_TARGETS_RELEASE "")
set(absl_node_hash_map_NOT_USED_RELEASE "")
set(absl_node_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_node_hash_map_FRAMEWORKS_FOUND_RELEASE} ${absl_node_hash_map_SYSTEM_LIBS_RELEASE} ${absl_node_hash_map_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_node_hash_map_LIBS_RELEASE}"
                              "${absl_node_hash_map_LIB_DIRS_RELEASE}"
                              "${absl_node_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_node_hash_map_NOT_USED_RELEASE
                              absl_node_hash_map_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_node_hash_map")

set(absl_node_hash_map_LINK_LIBS_RELEASE ${absl_node_hash_map_LIB_TARGETS_RELEASE} ${absl_node_hash_map_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT flat_hash_set VARIABLES #############################################

set(absl_flat_hash_set_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flat_hash_set_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flat_hash_set_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_flat_hash_set_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_flat_hash_set_RES_DIRS_RELEASE )
set(absl_flat_hash_set_DEFINITIONS_RELEASE )
set(absl_flat_hash_set_COMPILE_DEFINITIONS_RELEASE )
set(absl_flat_hash_set_COMPILE_OPTIONS_C_RELEASE "")
set(absl_flat_hash_set_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_flat_hash_set_LIBS_RELEASE )
set(absl_flat_hash_set_SYSTEM_LIBS_RELEASE )
set(absl_flat_hash_set_FRAMEWORK_DIRS_RELEASE )
set(absl_flat_hash_set_FRAMEWORKS_RELEASE )
set(absl_flat_hash_set_BUILD_MODULES_PATHS_RELEASE )
set(absl_flat_hash_set_DEPENDENCIES_RELEASE absl::container_memory absl::hash_function_defaults absl::raw_hash_set absl::algorithm_container absl::core_headers absl::memory)
set(absl_flat_hash_set_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT flat_hash_set FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_flat_hash_set_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_flat_hash_set_FRAMEWORKS_FOUND_RELEASE "${absl_flat_hash_set_FRAMEWORKS_RELEASE}" "${absl_flat_hash_set_FRAMEWORK_DIRS_RELEASE}")

set(absl_flat_hash_set_LIB_TARGETS_RELEASE "")
set(absl_flat_hash_set_NOT_USED_RELEASE "")
set(absl_flat_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_flat_hash_set_FRAMEWORKS_FOUND_RELEASE} ${absl_flat_hash_set_SYSTEM_LIBS_RELEASE} ${absl_flat_hash_set_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_flat_hash_set_LIBS_RELEASE}"
                              "${absl_flat_hash_set_LIB_DIRS_RELEASE}"
                              "${absl_flat_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_flat_hash_set_NOT_USED_RELEASE
                              absl_flat_hash_set_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_flat_hash_set")

set(absl_flat_hash_set_LINK_LIBS_RELEASE ${absl_flat_hash_set_LIB_TARGETS_RELEASE} ${absl_flat_hash_set_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT counting_allocator VARIABLES #############################################

set(absl_counting_allocator_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_counting_allocator_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_counting_allocator_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_counting_allocator_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_counting_allocator_RES_DIRS_RELEASE )
set(absl_counting_allocator_DEFINITIONS_RELEASE )
set(absl_counting_allocator_COMPILE_DEFINITIONS_RELEASE )
set(absl_counting_allocator_COMPILE_OPTIONS_C_RELEASE "")
set(absl_counting_allocator_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_counting_allocator_LIBS_RELEASE )
set(absl_counting_allocator_SYSTEM_LIBS_RELEASE )
set(absl_counting_allocator_FRAMEWORK_DIRS_RELEASE )
set(absl_counting_allocator_FRAMEWORKS_RELEASE )
set(absl_counting_allocator_BUILD_MODULES_PATHS_RELEASE )
set(absl_counting_allocator_DEPENDENCIES_RELEASE absl::config)
set(absl_counting_allocator_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT counting_allocator FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_counting_allocator_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_counting_allocator_FRAMEWORKS_FOUND_RELEASE "${absl_counting_allocator_FRAMEWORKS_RELEASE}" "${absl_counting_allocator_FRAMEWORK_DIRS_RELEASE}")

set(absl_counting_allocator_LIB_TARGETS_RELEASE "")
set(absl_counting_allocator_NOT_USED_RELEASE "")
set(absl_counting_allocator_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_counting_allocator_FRAMEWORKS_FOUND_RELEASE} ${absl_counting_allocator_SYSTEM_LIBS_RELEASE} ${absl_counting_allocator_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_counting_allocator_LIBS_RELEASE}"
                              "${absl_counting_allocator_LIB_DIRS_RELEASE}"
                              "${absl_counting_allocator_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_counting_allocator_NOT_USED_RELEASE
                              absl_counting_allocator_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_counting_allocator")

set(absl_counting_allocator_LINK_LIBS_RELEASE ${absl_counting_allocator_LIB_TARGETS_RELEASE} ${absl_counting_allocator_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT btree VARIABLES #############################################

set(absl_btree_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_btree_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_btree_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_btree_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_btree_RES_DIRS_RELEASE )
set(absl_btree_DEFINITIONS_RELEASE )
set(absl_btree_COMPILE_DEFINITIONS_RELEASE )
set(absl_btree_COMPILE_OPTIONS_C_RELEASE "")
set(absl_btree_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_btree_LIBS_RELEASE )
set(absl_btree_SYSTEM_LIBS_RELEASE )
set(absl_btree_FRAMEWORK_DIRS_RELEASE )
set(absl_btree_FRAMEWORKS_RELEASE )
set(absl_btree_BUILD_MODULES_PATHS_RELEASE )
set(absl_btree_DEPENDENCIES_RELEASE absl::container_common absl::common_policy_traits absl::compare absl::compressed_tuple absl::container_memory absl::cord absl::core_headers absl::layout absl::memory absl::raw_logging_internal absl::strings absl::throw_delegate absl::type_traits absl::utility)
set(absl_btree_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT btree FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_btree_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_btree_FRAMEWORKS_FOUND_RELEASE "${absl_btree_FRAMEWORKS_RELEASE}" "${absl_btree_FRAMEWORK_DIRS_RELEASE}")

set(absl_btree_LIB_TARGETS_RELEASE "")
set(absl_btree_NOT_USED_RELEASE "")
set(absl_btree_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_btree_FRAMEWORKS_FOUND_RELEASE} ${absl_btree_SYSTEM_LIBS_RELEASE} ${absl_btree_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_btree_LIBS_RELEASE}"
                              "${absl_btree_LIB_DIRS_RELEASE}"
                              "${absl_btree_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_btree_NOT_USED_RELEASE
                              absl_btree_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_btree")

set(absl_btree_LINK_LIBS_RELEASE ${absl_btree_LIB_TARGETS_RELEASE} ${absl_btree_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT scoped_set_env VARIABLES #############################################

set(absl_scoped_set_env_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_scoped_set_env_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_scoped_set_env_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_scoped_set_env_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_scoped_set_env_RES_DIRS_RELEASE )
set(absl_scoped_set_env_DEFINITIONS_RELEASE )
set(absl_scoped_set_env_COMPILE_DEFINITIONS_RELEASE )
set(absl_scoped_set_env_COMPILE_OPTIONS_C_RELEASE "")
set(absl_scoped_set_env_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_scoped_set_env_LIBS_RELEASE absl_scoped_set_env)
set(absl_scoped_set_env_SYSTEM_LIBS_RELEASE )
set(absl_scoped_set_env_FRAMEWORK_DIRS_RELEASE )
set(absl_scoped_set_env_FRAMEWORKS_RELEASE )
set(absl_scoped_set_env_BUILD_MODULES_PATHS_RELEASE )
set(absl_scoped_set_env_DEPENDENCIES_RELEASE absl::config absl::raw_logging_internal)
set(absl_scoped_set_env_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT scoped_set_env FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_scoped_set_env_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_scoped_set_env_FRAMEWORKS_FOUND_RELEASE "${absl_scoped_set_env_FRAMEWORKS_RELEASE}" "${absl_scoped_set_env_FRAMEWORK_DIRS_RELEASE}")

set(absl_scoped_set_env_LIB_TARGETS_RELEASE "")
set(absl_scoped_set_env_NOT_USED_RELEASE "")
set(absl_scoped_set_env_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_scoped_set_env_FRAMEWORKS_FOUND_RELEASE} ${absl_scoped_set_env_SYSTEM_LIBS_RELEASE} ${absl_scoped_set_env_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_scoped_set_env_LIBS_RELEASE}"
                              "${absl_scoped_set_env_LIB_DIRS_RELEASE}"
                              "${absl_scoped_set_env_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_scoped_set_env_NOT_USED_RELEASE
                              absl_scoped_set_env_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_scoped_set_env")

set(absl_scoped_set_env_LINK_LIBS_RELEASE ${absl_scoped_set_env_LIB_TARGETS_RELEASE} ${absl_scoped_set_env_LIBS_FRAMEWORKS_DEPS_RELEASE})

########### COMPONENT pretty_function VARIABLES #############################################

set(absl_pretty_function_INCLUDE_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_pretty_function_INCLUDE_DIR_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_pretty_function_INCLUDES_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include")
set(absl_pretty_function_LIB_DIRS_RELEASE "/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/lib")
set(absl_pretty_function_RES_DIRS_RELEASE )
set(absl_pretty_function_DEFINITIONS_RELEASE )
set(absl_pretty_function_COMPILE_DEFINITIONS_RELEASE )
set(absl_pretty_function_COMPILE_OPTIONS_C_RELEASE "")
set(absl_pretty_function_COMPILE_OPTIONS_CXX_RELEASE "")
set(absl_pretty_function_LIBS_RELEASE )
set(absl_pretty_function_SYSTEM_LIBS_RELEASE )
set(absl_pretty_function_FRAMEWORK_DIRS_RELEASE )
set(absl_pretty_function_FRAMEWORKS_RELEASE )
set(absl_pretty_function_BUILD_MODULES_PATHS_RELEASE )
set(absl_pretty_function_DEPENDENCIES_RELEASE )
set(absl_pretty_function_LINKER_FLAGS_LIST_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>
)

########## COMPONENT pretty_function FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(absl_pretty_function_FRAMEWORKS_FOUND_RELEASE "")
conan_find_apple_frameworks(absl_pretty_function_FRAMEWORKS_FOUND_RELEASE "${absl_pretty_function_FRAMEWORKS_RELEASE}" "${absl_pretty_function_FRAMEWORK_DIRS_RELEASE}")

set(absl_pretty_function_LIB_TARGETS_RELEASE "")
set(absl_pretty_function_NOT_USED_RELEASE "")
set(absl_pretty_function_LIBS_FRAMEWORKS_DEPS_RELEASE ${absl_pretty_function_FRAMEWORKS_FOUND_RELEASE} ${absl_pretty_function_SYSTEM_LIBS_RELEASE} ${absl_pretty_function_DEPENDENCIES_RELEASE})
conan_package_library_targets("${absl_pretty_function_LIBS_RELEASE}"
                              "${absl_pretty_function_LIB_DIRS_RELEASE}"
                              "${absl_pretty_function_LIBS_FRAMEWORKS_DEPS_RELEASE}"
                              absl_pretty_function_NOT_USED_RELEASE
                              absl_pretty_function_LIB_TARGETS_RELEASE
                              "RELEASE"
                              "absl_pretty_function")

set(absl_pretty_function_LINK_LIBS_RELEASE ${absl_pretty_function_LIB_TARGETS_RELEASE} ${absl_pretty_function_LIBS_FRAMEWORKS_DEPS_RELEASE})