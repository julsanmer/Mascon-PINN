#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libExternalModulesLib.dylib
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libExternalModulesLib.dylib
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libExternalModulesLib.dylib
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libExternalModulesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libExternalModulesLib.dylib
fi

