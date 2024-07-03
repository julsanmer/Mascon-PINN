#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/CMakeFiles/thrusterPlatformReference.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/architecture/_GeneralModuleFiles -I/Users/julio/Desktop/basilisk/src/../libs -I/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3 -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms -c++ -interface _thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -o /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms/thrusterPlatformReferencePYTHON_wrap.cxx /Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference/thrusterPlatformReference.i
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/CMakeFiles/thrusterPlatformReference.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/architecture/_GeneralModuleFiles -I/Users/julio/Desktop/basilisk/src/../libs -I/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3 -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session -I/Users/julio/.conan/data/abseil/20230125.3/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include -I/Users/julio/.conan/data/protobuf/3.21.12/_/_/package/be2a0a2807bb180398bafcdcd02b31ceea4093ed/include -I/Users/julio/.conan/data/zlib/1.2.13/_/_/package/240c2182163325b213ca6886a7614c8ed2bf1738/include -I/Users/julio/.conan/data/date/3.0.1/_/_/package/33916d95da6210bd8de70ffe57e95c68c8983732/include -I/Users/julio/.conan/data/libcurl/8.4.0/_/_/package/d699a8117ee89877a5435732a284bd66e73e8db3/include -I/Users/julio/.conan/data/re2/20230801/_/_/package/89057090356509c67c8b93e2281b62f169649ce4/include -I/Users/julio/.conan/data/onnx/1.14.1/_/_/package/48f89f03519a83ba9e20c54b894f2541c29e8f5c/include -I/Users/julio/.conan/data/flatbuffers/1.12.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include -I/Users/julio/.conan/data/boost/1.83.0/_/_/package/dd7f5f958c7381cfd81e611a16062de0c827160a/include -I/Users/julio/.conan/data/safeint/3.0.28/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include -I/Users/julio/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include -I/Users/julio/.conan/data/ms-gsl/4.0.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include -I/Users/julio/.conan/data/cpuinfo/cci.20220618/_/_/package/2ce689326568bd3a5257168c238ca5487c40c13f/include -I/Users/julio/.conan/data/nsync/1.26.0/_/_/package/2f2de4e3345f667bb03ed16a03f45c72c978d397/include -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms -c++ -interface _thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -o /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms/thrusterPlatformReferencePYTHON_wrap.cxx /Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference/thrusterPlatformReference.i
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/CMakeFiles/thrusterPlatformReference.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/architecture/_GeneralModuleFiles -I/Users/julio/Desktop/basilisk/src/../libs -I/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3 -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms -c++ -interface _thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -o /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms/thrusterPlatformReferencePYTHON_wrap.cxx /Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference/thrusterPlatformReference.i
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/CMakeFiles/thrusterPlatformReference.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/architecture/_GeneralModuleFiles -I/Users/julio/Desktop/basilisk/src/../libs -I/Users/julio/.conan/data/eigen/3.4.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/eigen3 -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include -I/Users/julio/.conan/data/onnxruntime/1.16.2/_/_/package/a254a010f70acd9347074f333eb19a470758b9f6/include/onnxruntime/core/session -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms -c++ -interface _thrusterPlatformReference -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -o /Users/julio/Desktop/basilisk/dist3/Basilisk/fswAlgorithms/thrusterPlatformReferencePYTHON_wrap.cxx /Users/julio/Desktop/basilisk/src/fswAlgorithms/effectorInterfaces/thrusterPlatformReference/thrusterPlatformReference.i
fi

