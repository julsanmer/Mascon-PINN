#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNav2MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNav2MsgPayload.h SmallBodyNav2MsgPayload msgPayloadDefC True
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNav2MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNav2MsgPayload.h SmallBodyNav2MsgPayload msgPayloadDefC True
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNav2MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNav2MsgPayload.h SmallBodyNav2MsgPayload msgPayloadDefC True
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNav2MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNav2MsgPayload.h SmallBodyNav2MsgPayload msgPayloadDefC True
fi

