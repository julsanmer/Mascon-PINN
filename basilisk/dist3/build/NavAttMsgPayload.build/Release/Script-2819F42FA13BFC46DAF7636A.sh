#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/julio/Desktop/basilisk/dist3/architecture/messaging
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/architecture/messaging/CMakeFiles/NavAttMsgPayload.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../ -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../../ -I/Users/julio/Desktop/basilisk/External/ -I/Users/julio/Desktop/basilisk/dist3/autoSource/ -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ -c++ -interface _NavAttMsgPayload -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/architecture/messaging/.. -I/Users/julio/Desktop/basilisk/src/architecture/messaging -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../.. -o /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging//NavAttMsgPayloadPYTHON_wrap.cxx /Users/julio/Desktop/basilisk/dist3/autoSource/NavAttMsgPayload.i
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/julio/Desktop/basilisk/dist3/architecture/messaging
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/architecture/messaging/CMakeFiles/NavAttMsgPayload.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../ -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../../ -I/Users/julio/Desktop/basilisk/External/ -I/Users/julio/Desktop/basilisk/dist3/autoSource/ -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ -c++ -interface _NavAttMsgPayload -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/architecture/messaging/.. -I/Users/julio/Desktop/basilisk/src/architecture/messaging -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../.. -o /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging//NavAttMsgPayloadPYTHON_wrap.cxx /Users/julio/Desktop/basilisk/dist3/autoSource/NavAttMsgPayload.i
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/julio/Desktop/basilisk/dist3/architecture/messaging
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/architecture/messaging/CMakeFiles/NavAttMsgPayload.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../ -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../../ -I/Users/julio/Desktop/basilisk/External/ -I/Users/julio/Desktop/basilisk/dist3/autoSource/ -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ -c++ -interface _NavAttMsgPayload -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/architecture/messaging/.. -I/Users/julio/Desktop/basilisk/src/architecture/messaging -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../.. -o /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging//NavAttMsgPayloadPYTHON_wrap.cxx /Users/julio/Desktop/basilisk/dist3/autoSource/NavAttMsgPayload.i
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/julio/Desktop/basilisk/dist3/architecture/messaging
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E make_directory /Users/julio/Desktop/basilisk/dist3/architecture/messaging/CMakeFiles/NavAttMsgPayload.dir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/
  /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E env SWIG_LIB=/opt/homebrew/Cellar/swig/4.1.1/share/swig/4.1.1 /opt/homebrew/bin/swig -python -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../ -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../../ -I/Users/julio/Desktop/basilisk/External/ -I/Users/julio/Desktop/basilisk/dist3/autoSource/ -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -outdir /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging/ -c++ -interface _NavAttMsgPayload -I/Users/julio/Desktop/basilisk/src/../dist3/autoSource -I/Users/julio/Desktop/basilisk/src -I/Users/julio/Desktop/basilisk/External -I/opt/homebrew/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Users/julio/Desktop/basilisk/src/architecture/messaging/.. -I/Users/julio/Desktop/basilisk/src/architecture/messaging -I/Users/julio/Desktop/basilisk/src/architecture/messaging/../.. -o /Users/julio/Desktop/basilisk/dist3/Basilisk/architecture/messaging//NavAttMsgPayloadPYTHON_wrap.cxx /Users/julio/Desktop/basilisk/dist3/autoSource/NavAttMsgPayload.i
fi

