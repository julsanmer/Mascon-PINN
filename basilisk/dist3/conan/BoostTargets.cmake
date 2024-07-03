

if(NOT TARGET Boost::dynamic_linking)
    add_library(Boost::dynamic_linking INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::disable_autolinking)
    add_library(Boost::disable_autolinking INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::diagnostic_definitions)
    add_library(Boost::diagnostic_definitions INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::headers)
    add_library(Boost::headers INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::_libboost)
    add_library(Boost::_libboost INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::serialization)
    add_library(Boost::serialization INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::wserialization)
    add_library(Boost::wserialization INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::exception)
    add_library(Boost::exception INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::test)
    add_library(Boost::test INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::test_exec_monitor)
    add_library(Boost::test_exec_monitor INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::prg_exec_monitor)
    add_library(Boost::prg_exec_monitor INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::unit_test_framework)
    add_library(Boost::unit_test_framework INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::system)
    add_library(Boost::system INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::date_time)
    add_library(Boost::date_time INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::container)
    add_library(Boost::container INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::chrono)
    add_library(Boost::chrono INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::atomic)
    add_library(Boost::atomic INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::thread)
    add_library(Boost::thread INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::type_erasure)
    add_library(Boost::type_erasure INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::timer)
    add_library(Boost::timer INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::stacktrace)
    add_library(Boost::stacktrace INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::stacktrace_noop)
    add_library(Boost::stacktrace_noop INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::stacktrace_basic)
    add_library(Boost::stacktrace_basic INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::stacktrace_backtrace)
    add_library(Boost::stacktrace_backtrace INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::stacktrace_addr2line)
    add_library(Boost::stacktrace_addr2line INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::regex)
    add_library(Boost::regex INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::random)
    add_library(Boost::random INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::program_options)
    add_library(Boost::program_options INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::filesystem)
    add_library(Boost::filesystem INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::log)
    add_library(Boost::log INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::log_setup)
    add_library(Boost::log_setup INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::iostreams)
    add_library(Boost::iostreams INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::context)
    add_library(Boost::context INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::coroutine)
    add_library(Boost::coroutine INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::contract)
    add_library(Boost::contract INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED)
endif()

if(NOT TARGET Boost::Boost)
    add_library(Boost::Boost INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/BoostTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(Boost_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${Boost_FIND_COMPONENTS})
        list(FIND Boost_COMPONENTS_RELEASE "Boost::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'Boost'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'Boost'")
        endif()
    endforeach()
endif()