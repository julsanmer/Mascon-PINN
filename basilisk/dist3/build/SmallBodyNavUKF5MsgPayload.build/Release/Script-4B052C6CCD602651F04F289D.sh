#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNavUKF5MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNavUKF5MsgPayload.h SmallBodyNavUKF5MsgPayload msgPayloadDefC True
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNavUKF5MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNavUKF5MsgPayload.h SmallBodyNavUKF5MsgPayload msgPayloadDefC True
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNavUKF5MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNavUKF5MsgPayload.h SmallBodyNavUKF5MsgPayload msgPayloadDefC True
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/src/architecture/messaging/msgAutoSource
  /Users/julio/Desktop/python_scripts/.venv/bin/python3 generateSWIGModules.py /Users/julio/Desktop/basilisk/src/architecture/messaging/../../../dist3/autoSource/SmallBodyNavUKF5MsgPayload.i ../../../External/msgPayloadDefC/SmallBodyNavUKF5MsgPayload.h SmallBodyNavUKF5MsgPayload msgPayloadDefC True
fi

