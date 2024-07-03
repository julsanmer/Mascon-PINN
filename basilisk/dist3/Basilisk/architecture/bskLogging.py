# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _bskLogging
else:
    import _bskLogging

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "this":
            set(self, name, value)
        elif name == "thisown":
            self.this.own(value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


MAX_LOGGING_LENGTH = _bskLogging.MAX_LOGGING_LENGTH
BSK_DEBUG = _bskLogging.BSK_DEBUG
BSK_INFORMATION = _bskLogging.BSK_INFORMATION
BSK_WARNING = _bskLogging.BSK_WARNING
BSK_ERROR = _bskLogging.BSK_ERROR
BSK_SILENT = _bskLogging.BSK_SILENT

def printDefaultLogLevel():
    return _bskLogging.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _bskLogging.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _bskLogging.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _bskLogging.BSKLogger_swiginit(self, _bskLogging.new_BSKLogger(*args))
    __swig_destroy__ = _bskLogging.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _bskLogging.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _bskLogging.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _bskLogging.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _bskLogging.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_bskLogging.BSKLogger_logLevelMap_get, _bskLogging.BSKLogger_logLevelMap_set)

# Register BSKLogger in _bskLogging:
_bskLogging.BSKLogger_swigregister(BSKLogger)
cvar = _bskLogging.cvar


def _BSKLogger():
    return _bskLogging._BSKLogger()

def _BSKLogger_d(arg1):
    return _bskLogging._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _bskLogging._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _bskLogging._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _bskLogging._bskLog(arg1, arg2, arg3)

from Basilisk.architecture.swig_common_model import *


import sys
protectAllClasses(sys.modules[__name__])


