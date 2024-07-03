#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/src
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 utilities/overwriteSwig.py /Users/julio/Desktop/basilisk/dist3 4.1.1
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/src
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 utilities/overwriteSwig.py /Users/julio/Desktop/basilisk/dist3 4.1.1
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/src
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 utilities/overwriteSwig.py /Users/julio/Desktop/basilisk/dist3 4.1.1
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/src
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 utilities/overwriteSwig.py /Users/julio/Desktop/basilisk/dist3 4.1.1
fi

