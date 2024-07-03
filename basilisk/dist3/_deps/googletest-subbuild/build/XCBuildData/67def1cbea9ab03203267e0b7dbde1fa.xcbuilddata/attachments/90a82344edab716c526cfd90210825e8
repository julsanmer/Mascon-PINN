#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3/_deps
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -P /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/download-googletest-populate.cmake
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -P /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/verify-googletest-populate.cmake
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -P /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/extract-googletest-populate.cmake
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E touch /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/googletest-populate-download
fi

