#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E touch /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/CMakeFiles/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/googletest-populate-complete
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E touch /Users/julio/Desktop/basilisk/dist3/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/$CONFIGURATION$EFFECTIVE_PLATFORM_NAME/googletest-populate-done
fi

