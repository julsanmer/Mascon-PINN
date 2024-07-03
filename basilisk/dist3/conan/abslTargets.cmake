

if(NOT TARGET absl::config)
    add_library(absl::config INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::type_traits)
    add_library(absl::type_traits INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::base_internal)
    add_library(absl::base_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::utility)
    add_library(absl::utility INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::core_headers)
    add_library(absl::core_headers INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::compare)
    add_library(absl::compare INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_severity)
    add_library(absl::log_severity INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::errno_saver)
    add_library(absl::errno_saver INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::atomic_hook)
    add_library(absl::atomic_hook INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::raw_logging_internal)
    add_library(absl::raw_logging_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::bad_variant_access)
    add_library(absl::bad_variant_access INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::variant)
    add_library(absl::variant INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::bad_optional_access)
    add_library(absl::bad_optional_access INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::meta)
    add_library(absl::meta INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::memory)
    add_library(absl::memory INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::optional)
    add_library(absl::optional INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::algorithm)
    add_library(absl::algorithm INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::throw_delegate)
    add_library(absl::throw_delegate INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::span)
    add_library(absl::span INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::bad_any_cast_impl)
    add_library(absl::bad_any_cast_impl INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::bad_any_cast)
    add_library(absl::bad_any_cast INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::fast_type_id)
    add_library(absl::fast_type_id INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::any)
    add_library(absl::any INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::time_zone)
    add_library(absl::time_zone INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::civil_time)
    add_library(absl::civil_time INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::dynamic_annotations)
    add_library(absl::dynamic_annotations INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::spinlock_wait)
    add_library(absl::spinlock_wait INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::base)
    add_library(absl::base INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::endian)
    add_library(absl::endian INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::strings_internal)
    add_library(absl::strings_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::bits)
    add_library(absl::bits INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::int128)
    add_library(absl::int128 INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::strings)
    add_library(absl::strings INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::time)
    add_library(absl::time INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::kernel_timeout_internal)
    add_library(absl::kernel_timeout_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::malloc_internal)
    add_library(absl::malloc_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::graphcycles_internal)
    add_library(absl::graphcycles_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::demangle_internal)
    add_library(absl::demangle_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::debugging_internal)
    add_library(absl::debugging_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::symbolize)
    add_library(absl::symbolize INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::stacktrace)
    add_library(absl::stacktrace INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::synchronization)
    add_library(absl::synchronization INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_handle)
    add_library(absl::cordz_handle INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_update_tracker)
    add_library(absl::cordz_update_tracker INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_statistics)
    add_library(absl::cordz_statistics INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::exponential_biased)
    add_library(absl::exponential_biased INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_functions)
    add_library(absl::cordz_functions INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::non_temporal_arm_intrinsics)
    add_library(absl::non_temporal_arm_intrinsics INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::non_temporal_memcpy)
    add_library(absl::non_temporal_memcpy INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::crc_cpu_detect)
    add_library(absl::crc_cpu_detect INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::prefetch)
    add_library(absl::prefetch INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::crc_internal)
    add_library(absl::crc_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::crc32c)
    add_library(absl::crc32c INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::crc_cord_state)
    add_library(absl::crc_cord_state INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::layout)
    add_library(absl::layout INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::container_memory)
    add_library(absl::container_memory INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::compressed_tuple)
    add_library(absl::compressed_tuple INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::inlined_vector_internal)
    add_library(absl::inlined_vector_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::inlined_vector)
    add_library(absl::inlined_vector INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cord_internal)
    add_library(absl::cord_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_info)
    add_library(absl::cordz_info INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_update_scope)
    add_library(absl::cordz_update_scope INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::function_ref)
    add_library(absl::function_ref INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::fixed_array)
    add_library(absl::fixed_array INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cord)
    add_library(absl::cord INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cordz_sample_token)
    add_library(absl::cordz_sample_token INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::numeric_representation)
    add_library(absl::numeric_representation INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::str_format_internal)
    add_library(absl::str_format_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::str_format)
    add_library(absl::str_format INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::strerror)
    add_library(absl::strerror INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::status)
    add_library(absl::status INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::statusor)
    add_library(absl::statusor INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_traits)
    add_library(absl::random_internal_traits INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_uniform_helper)
    add_library(absl::random_internal_uniform_helper INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_distribution_test_util)
    add_library(absl::random_internal_distribution_test_util INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_platform)
    add_library(absl::random_internal_platform INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_randen_hwaes_impl)
    add_library(absl::random_internal_randen_hwaes_impl INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_randen_hwaes)
    add_library(absl::random_internal_randen_hwaes INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_randen_slow)
    add_library(absl::random_internal_randen_slow INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_randen)
    add_library(absl::random_internal_randen INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_iostream_state_saver)
    add_library(absl::random_internal_iostream_state_saver INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_randen_engine)
    add_library(absl::random_internal_randen_engine INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_fastmath)
    add_library(absl::random_internal_fastmath INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_pcg_engine)
    add_library(absl::random_internal_pcg_engine INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_fast_uniform_bits)
    add_library(absl::random_internal_fast_uniform_bits INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_seed_material)
    add_library(absl::random_internal_seed_material INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_salted_seed_seq)
    add_library(absl::random_internal_salted_seed_seq INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_seed_gen_exception)
    add_library(absl::random_seed_gen_exception INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_pool_urbg)
    add_library(absl::random_internal_pool_urbg INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_nonsecure_base)
    add_library(absl::random_internal_nonsecure_base INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_wide_multiply)
    add_library(absl::random_internal_wide_multiply INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_generate_real)
    add_library(absl::random_internal_generate_real INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_distribution_caller)
    add_library(absl::random_internal_distribution_caller INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_seed_sequences)
    add_library(absl::random_seed_sequences INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_distributions)
    add_library(absl::random_distributions INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_internal_mock_helpers)
    add_library(absl::random_internal_mock_helpers INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_bit_gen_ref)
    add_library(absl::random_bit_gen_ref INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::random_random)
    add_library(absl::random_random INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::periodic_sampler)
    add_library(absl::periodic_sampler INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::sample_recorder)
    add_library(absl::sample_recorder INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::numeric)
    add_library(absl::numeric INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_config)
    add_library(absl::log_internal_config INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_entry)
    add_library(absl::log_entry INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_sink)
    add_library(absl::log_sink INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::low_level_hash)
    add_library(absl::low_level_hash INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::city)
    add_library(absl::city INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::hash)
    add_library(absl::hash INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_globals)
    add_library(absl::log_globals INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_globals)
    add_library(absl::log_internal_globals INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cleanup_internal)
    add_library(absl::cleanup_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::cleanup)
    add_library(absl::cleanup INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_log_sink_set)
    add_library(absl::log_internal_log_sink_set INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_sink_registry)
    add_library(absl::log_sink_registry INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_append_truncated)
    add_library(absl::log_internal_append_truncated INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_nullguard)
    add_library(absl::log_internal_nullguard INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_proto)
    add_library(absl::log_internal_proto INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_format)
    add_library(absl::log_internal_format INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::examine_stack)
    add_library(absl::examine_stack INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_message)
    add_library(absl::log_internal_message INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_structured)
    add_library(absl::log_internal_structured INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_structured)
    add_library(absl::log_structured INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_voidify)
    add_library(absl::log_internal_voidify INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_nullstream)
    add_library(absl::log_internal_nullstream INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_strip)
    add_library(absl::log_internal_strip INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_conditions)
    add_library(absl::log_internal_conditions INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_log_impl)
    add_library(absl::log_internal_log_impl INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::absl_log)
    add_library(absl::absl_log INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_streamer)
    add_library(absl::log_streamer INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log)
    add_library(absl::log INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_initialize)
    add_library(absl::log_initialize INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_commandlineflag_internal)
    add_library(absl::flags_commandlineflag_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_commandlineflag)
    add_library(absl::flags_commandlineflag INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_marshalling)
    add_library(absl::flags_marshalling INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_path_util)
    add_library(absl::flags_path_util INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_program_name)
    add_library(absl::flags_program_name INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_config)
    add_library(absl::flags_config INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_internal)
    add_library(absl::flags_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_private_handle_accessor)
    add_library(absl::flags_private_handle_accessor INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::container_common)
    add_library(absl::container_common INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::hashtable_debug_hooks)
    add_library(absl::hashtable_debug_hooks INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::hashtablez_sampler)
    add_library(absl::hashtablez_sampler INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::common_policy_traits)
    add_library(absl::common_policy_traits INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::hash_policy_traits)
    add_library(absl::hash_policy_traits INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::raw_hash_set)
    add_library(absl::raw_hash_set INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::raw_hash_map)
    add_library(absl::raw_hash_map INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::hash_function_defaults)
    add_library(absl::hash_function_defaults INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::algorithm_container)
    add_library(absl::algorithm_container INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flat_hash_map)
    add_library(absl::flat_hash_map INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_reflection)
    add_library(absl::flags_reflection INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags)
    add_library(absl::flags INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_flags)
    add_library(absl::log_internal_flags INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_flags)
    add_library(absl::log_flags INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::die_if_null)
    add_library(absl::die_if_null INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_check_op)
    add_library(absl::log_internal_check_op INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::log_internal_check_impl)
    add_library(absl::log_internal_check_impl INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::check)
    add_library(absl::check INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::absl_check)
    add_library(absl::absl_check INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::bind_front)
    add_library(absl::bind_front INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::any_invocable)
    add_library(absl::any_invocable INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_usage_internal)
    add_library(absl::flags_usage_internal INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_usage)
    add_library(absl::flags_usage INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flags_parse)
    add_library(absl::flags_parse INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::leak_check)
    add_library(absl::leak_check INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::debugging)
    add_library(absl::debugging INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::failure_signal_handler)
    add_library(absl::failure_signal_handler INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::node_slot_policy)
    add_library(absl::node_slot_policy INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::hashtable_debug)
    add_library(absl::hashtable_debug INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::node_hash_set)
    add_library(absl::node_hash_set INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::node_hash_map)
    add_library(absl::node_hash_map INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::flat_hash_set)
    add_library(absl::flat_hash_set INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::counting_allocator)
    add_library(absl::counting_allocator INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::btree)
    add_library(absl::btree INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::scoped_set_env)
    add_library(absl::scoped_set_env INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::pretty_function)
    add_library(absl::pretty_function INTERFACE IMPORTED)
endif()

if(NOT TARGET absl::absl)
    add_library(absl::absl INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/abslTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(absl_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${absl_FIND_COMPONENTS})
        list(FIND absl_COMPONENTS_RELEASE "absl::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'absl'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'absl'")
        endif()
    endforeach()
endif()