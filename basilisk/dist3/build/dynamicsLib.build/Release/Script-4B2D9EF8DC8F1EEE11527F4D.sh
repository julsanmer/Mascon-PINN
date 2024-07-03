#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libdynamicsLib.dylib
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libdynamicsLib.dylib
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libdynamicsLib.dylib
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libdynamicsLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libdynamicsLib.dylib
fi

