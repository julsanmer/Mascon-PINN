if(EXISTS "/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/Debug/test_gaussMarkov")
  if(NOT EXISTS "/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/test_gaussMarkov[1]_tests-Debug.cmake" OR
     NOT "/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/test_gaussMarkov[1]_tests-Debug.cmake" IS_NEWER_THAN "/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/Debug/test_gaussMarkov" OR
     NOT "/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/test_gaussMarkov[1]_tests-Debug.cmake" IS_NEWER_THAN "${CMAKE_CURRENT_LIST_FILE}")
    include("/opt/homebrew/Cellar/cmake/3.26.4/share/cmake/Modules/GoogleTestAddTests.cmake")
    gtest_discover_tests_impl(
      TEST_EXECUTABLE [==[/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/Debug/test_gaussMarkov]==]
      TEST_EXECUTOR [==[]==]
      TEST_WORKING_DIR [==[/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests]==]
      TEST_EXTRA_ARGS [==[]==]
      TEST_PROPERTIES [==[]==]
      TEST_PREFIX [==[]==]
      TEST_SUFFIX [==[]==]
      TEST_FILTER [==[]==]
      NO_PRETTY_TYPES [==[FALSE]==]
      NO_PRETTY_VALUES [==[FALSE]==]
      TEST_LIST [==[test_gaussMarkov_TESTS]==]
      CTEST_FILE [==[/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/test_gaussMarkov[1]_tests-Debug.cmake]==]
      TEST_DISCOVERY_TIMEOUT [==[5]==]
      TEST_XML_OUTPUT_DIR [==[]==]
    )
  endif()
  include("/Users/julio/Desktop/basilisk/dist3/architecture/utilities/tests/test_gaussMarkov[1]_tests-Debug.cmake")
else()
  add_test(test_gaussMarkov_NOT_BUILT test_gaussMarkov_NOT_BUILT)
endif()
