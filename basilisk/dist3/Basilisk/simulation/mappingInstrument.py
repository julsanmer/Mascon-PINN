# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.


from typing import Union, Iterable
from Basilisk.utilities import pythonVariableLogger



from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _mappingInstrument
else:
    import _mappingInstrument

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



from Basilisk.architecture.swig_common_model import *


def new_doubleArray(nelements):
    return _mappingInstrument.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _mappingInstrument.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _mappingInstrument.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _mappingInstrument.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _mappingInstrument.new_longArray(nelements)

def delete_longArray(ary):
    return _mappingInstrument.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _mappingInstrument.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _mappingInstrument.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _mappingInstrument.new_intArray(nelements)

def delete_intArray(ary):
    return _mappingInstrument.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _mappingInstrument.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _mappingInstrument.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _mappingInstrument.new_shortArray(nelements)

def delete_shortArray(ary):
    return _mappingInstrument.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _mappingInstrument.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _mappingInstrument.shortArray_setitem(ary, index, value)


def getStructSize(self):
    try:
        return eval('sizeof_' + repr(self).split(';')[0].split('.')[-1])
    except (NameError) as e:
        typeString = 'sizeof_' + repr(self).split(';')[0].split('.')[-1]
        raise NameError(e.message + '\nYou tried to get this size macro: ' + typeString + 
            '\n It appears to be undefined.  \nYou need to run the SWIG GEN_SIZEOF' +  
            ' SWIG macro against the class/struct in your SWIG file if you want to ' + 
            ' make this call.\n')


def protectSetAttr(self, name, value):
    if(hasattr(self, name) or name == 'this' or name.find('swig') >= 0):
        object.__setattr__(self, name, value)
    else:
        raise ValueError('You tried to add this variable: ' + name + '\n' + 
            'To this class: ' + str(self))

def protectAllClasses(moduleType):
    import inspect
    import sys
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for member in clsmembers:
        try:
            exec(str(member[0]) + '.__setattr__ = protectSetAttr')
            exec(str(member[0]) + '.getStructSize = getStructSize') 
        except (AttributeError, TypeError) as e:
            pass


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _mappingInstrument.delete_SwigPyIterator

    def value(self):
        return _mappingInstrument.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _mappingInstrument.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _mappingInstrument.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _mappingInstrument.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _mappingInstrument.SwigPyIterator_equal(self, x)

    def copy(self):
        return _mappingInstrument.SwigPyIterator_copy(self)

    def next(self):
        return _mappingInstrument.SwigPyIterator_next(self)

    def __next__(self):
        return _mappingInstrument.SwigPyIterator___next__(self)

    def previous(self):
        return _mappingInstrument.SwigPyIterator_previous(self)

    def advance(self, n):
        return _mappingInstrument.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _mappingInstrument.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _mappingInstrument.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _mappingInstrument.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _mappingInstrument.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _mappingInstrument.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _mappingInstrument.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _mappingInstrument:
_mappingInstrument.SwigPyIterator_swigregister(SwigPyIterator)

from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _mappingInstrument.MAX_LOGGING_LENGTH
BSK_DEBUG = _mappingInstrument.BSK_DEBUG
BSK_INFORMATION = _mappingInstrument.BSK_INFORMATION
BSK_WARNING = _mappingInstrument.BSK_WARNING
BSK_ERROR = _mappingInstrument.BSK_ERROR
BSK_SILENT = _mappingInstrument.BSK_SILENT

def printDefaultLogLevel():
    return _mappingInstrument.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _mappingInstrument.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _mappingInstrument.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _mappingInstrument.BSKLogger_swiginit(self, _mappingInstrument.new_BSKLogger(*args))
    __swig_destroy__ = _mappingInstrument.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _mappingInstrument.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _mappingInstrument.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _mappingInstrument.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _mappingInstrument.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_mappingInstrument.BSKLogger_logLevelMap_get, _mappingInstrument.BSKLogger_logLevelMap_set)

# Register BSKLogger in _mappingInstrument:
_mappingInstrument.BSKLogger_swigregister(BSKLogger)
cvar = _mappingInstrument.cvar


def _BSKLogger():
    return _mappingInstrument._BSKLogger()

def _BSKLogger_d(arg1):
    return _mappingInstrument._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _mappingInstrument._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _mappingInstrument._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _mappingInstrument._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _mappingInstrument.SysModel_swiginit(self, _mappingInstrument.new_SysModel(*args))
    __swig_destroy__ = _mappingInstrument.delete_SysModel

    def SelfInit(self):
        return _mappingInstrument.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _mappingInstrument.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _mappingInstrument.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _mappingInstrument.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_mappingInstrument.SysModel_ModelTag_get, _mappingInstrument.SysModel_ModelTag_set)
    CallCounts = property(_mappingInstrument.SysModel_CallCounts_get, _mappingInstrument.SysModel_CallCounts_set)
    RNGSeed = property(_mappingInstrument.SysModel_RNGSeed_get, _mappingInstrument.SysModel_RNGSeed_set)
    moduleID = property(_mappingInstrument.SysModel_moduleID_get, _mappingInstrument.SysModel_moduleID_set)

    def logger(self, *args, **kwargs):
        raise TypeError(
            f"The 'logger' function is not supported for this type ('{type(self).__qualname__}'). "
            "To fix this, update the SWIG file for this module. Change "
            """'%include "sys_model.h"' to '%include "sys_model.i"'"""
        )


    def logger(self, variableNames: Union[str, Iterable[str]], recordingTime: int = 0):
        if isinstance(variableNames, str):
            variableNames = [variableNames]

        logging_functions = {
            variable_name: lambda _, vn=variable_name: getattr(self, vn)
            for variable_name in variableNames
        }

        for variable_name, log_fun in logging_functions.items():
            try:
                log_fun(0)
            except AttributeError:
                raise ValueError(f"Cannot log {variable_name} as it is not a "
                                f"variable of {type(self).__name__}")

        return pythonVariableLogger.PythonVariableLogger(logging_functions, recordingTime)


# Register SysModel in _mappingInstrument:
_mappingInstrument.SysModel_swigregister(SysModel)
class MappingInstrument(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _mappingInstrument.MappingInstrument_swiginit(self, _mappingInstrument.new_MappingInstrument())
    __swig_destroy__ = _mappingInstrument.delete_MappingInstrument

    def Reset(self, CurrentSimNanos):
        return _mappingInstrument.MappingInstrument_Reset(self, CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos):
        return _mappingInstrument.MappingInstrument_UpdateState(self, CurrentSimNanos)

    def addMappingPoint(self, tmpAccessMsg, dataName):
        return _mappingInstrument.MappingInstrument_addMappingPoint(self, tmpAccessMsg, dataName)
    dataNodeOutMsgs = property(_mappingInstrument.MappingInstrument_dataNodeOutMsgs_get, _mappingInstrument.MappingInstrument_dataNodeOutMsgs_set)
    accessInMsgs = property(_mappingInstrument.MappingInstrument_accessInMsgs_get, _mappingInstrument.MappingInstrument_accessInMsgs_set)
    bskLogger = property(_mappingInstrument.MappingInstrument_bskLogger_get, _mappingInstrument.MappingInstrument_bskLogger_set)
    nodeBaudRate = property(_mappingInstrument.MappingInstrument_nodeBaudRate_get, _mappingInstrument.MappingInstrument_nodeBaudRate_set)

# Register MappingInstrument in _mappingInstrument:
_mappingInstrument.MappingInstrument_swigregister(MappingInstrument)
class AccessMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    hasAccess = property(_mappingInstrument.AccessMsgPayload_hasAccess_get, _mappingInstrument.AccessMsgPayload_hasAccess_set)
    slantRange = property(_mappingInstrument.AccessMsgPayload_slantRange_get, _mappingInstrument.AccessMsgPayload_slantRange_set)
    elevation = property(_mappingInstrument.AccessMsgPayload_elevation_get, _mappingInstrument.AccessMsgPayload_elevation_set)
    azimuth = property(_mappingInstrument.AccessMsgPayload_azimuth_get, _mappingInstrument.AccessMsgPayload_azimuth_set)
    range_dot = property(_mappingInstrument.AccessMsgPayload_range_dot_get, _mappingInstrument.AccessMsgPayload_range_dot_set)
    el_dot = property(_mappingInstrument.AccessMsgPayload_el_dot_get, _mappingInstrument.AccessMsgPayload_el_dot_set)
    az_dot = property(_mappingInstrument.AccessMsgPayload_az_dot_get, _mappingInstrument.AccessMsgPayload_az_dot_set)
    r_BL_L = property(_mappingInstrument.AccessMsgPayload_r_BL_L_get, _mappingInstrument.AccessMsgPayload_r_BL_L_set)
    v_BL_L = property(_mappingInstrument.AccessMsgPayload_v_BL_L_get, _mappingInstrument.AccessMsgPayload_v_BL_L_set)

    def __init__(self):
        _mappingInstrument.AccessMsgPayload_swiginit(self, _mappingInstrument.new_AccessMsgPayload())
    __swig_destroy__ = _mappingInstrument.delete_AccessMsgPayload

# Register AccessMsgPayload in _mappingInstrument:
_mappingInstrument.AccessMsgPayload_swigregister(AccessMsgPayload)
class DataNodeUsageMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dataName = property(_mappingInstrument.DataNodeUsageMsgPayload_dataName_get, _mappingInstrument.DataNodeUsageMsgPayload_dataName_set)
    baudRate = property(_mappingInstrument.DataNodeUsageMsgPayload_baudRate_get, _mappingInstrument.DataNodeUsageMsgPayload_baudRate_set)

    def __init__(self):
        _mappingInstrument.DataNodeUsageMsgPayload_swiginit(self, _mappingInstrument.new_DataNodeUsageMsgPayload())
    __swig_destroy__ = _mappingInstrument.delete_DataNodeUsageMsgPayload

# Register DataNodeUsageMsgPayload in _mappingInstrument:
_mappingInstrument.DataNodeUsageMsgPayload_swigregister(DataNodeUsageMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


