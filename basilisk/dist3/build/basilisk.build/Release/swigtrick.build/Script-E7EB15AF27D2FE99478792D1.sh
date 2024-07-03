#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generatePackageInit.py /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ ../../msgPayloadDefC/ ../../msgPayloadDefCpp/ /Users/julio/Desktop/basilisk/External/msgPayloadDefC/ /Users/julio/Desktop/basilisk/External/msgPayloadDefCpp/
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generatePackageInit.py /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ ../../msgPayloadDefC/ ../../msgPayloadDefCpp/ /Users/julio/Desktop/basilisk/External/msgPayloadDefC/ /Users/julio/Desktop/basilisk/External/msgPayloadDefCpp/
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generatePackageInit.py /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ ../../msgPayloadDefC/ ../../msgPayloadDefCpp/ /Users/julio/Desktop/basilisk/External/msgPayloadDefC/ /Users/julio/Desktop/basilisk/External/msgPayloadDefCpp/
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generatePackageInit.py /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ ../../msgPayloadDefC/ ../../msgPayloadDefCpp/ /Users/julio/Desktop/basilisk/External/msgPayloadDefC/ /Users/julio/Desktop/basilisk/External/msgPayloadDefCpp/
fi

