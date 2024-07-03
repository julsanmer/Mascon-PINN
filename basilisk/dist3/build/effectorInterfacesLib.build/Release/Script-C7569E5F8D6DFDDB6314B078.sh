#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libeffectorInterfacesLib.dylib
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/libeffectorInterfacesLib.dylib
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/MinSizeRel/libeffectorInterfacesLib.dylib
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/dist3
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E cmake_symlink_library /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libeffectorInterfacesLib.dylib /Users/julio/Desktop/basilisk/dist3/Basilisk/RelWithDebInfo/libeffectorInterfacesLib.dylib
fi

