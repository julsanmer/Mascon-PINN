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
    from . import _simpleStorageUnit
else:
    import _simpleStorageUnit

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
    return _simpleStorageUnit.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _simpleStorageUnit.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _simpleStorageUnit.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _simpleStorageUnit.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _simpleStorageUnit.new_longArray(nelements)

def delete_longArray(ary):
    return _simpleStorageUnit.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _simpleStorageUnit.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _simpleStorageUnit.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _simpleStorageUnit.new_intArray(nelements)

def delete_intArray(ary):
    return _simpleStorageUnit.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _simpleStorageUnit.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _simpleStorageUnit.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _simpleStorageUnit.new_shortArray(nelements)

def delete_shortArray(ary):
    return _simpleStorageUnit.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _simpleStorageUnit.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _simpleStorageUnit.shortArray_setitem(ary, index, value)


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
    __swig_destroy__ = _simpleStorageUnit.delete_SwigPyIterator

    def value(self):
        return _simpleStorageUnit.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _simpleStorageUnit.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _simpleStorageUnit.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _simpleStorageUnit.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _simpleStorageUnit.SwigPyIterator_equal(self, x)

    def copy(self):
        return _simpleStorageUnit.SwigPyIterator_copy(self)

    def next(self):
        return _simpleStorageUnit.SwigPyIterator_next(self)

    def __next__(self):
        return _simpleStorageUnit.SwigPyIterator___next__(self)

    def previous(self):
        return _simpleStorageUnit.SwigPyIterator_previous(self)

    def advance(self, n):
        return _simpleStorageUnit.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _simpleStorageUnit.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _simpleStorageUnit.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _simpleStorageUnit.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _simpleStorageUnit.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _simpleStorageUnit.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _simpleStorageUnit.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _simpleStorageUnit:
_simpleStorageUnit.SwigPyIterator_swigregister(SwigPyIterator)

from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _simpleStorageUnit.MAX_LOGGING_LENGTH
BSK_DEBUG = _simpleStorageUnit.BSK_DEBUG
BSK_INFORMATION = _simpleStorageUnit.BSK_INFORMATION
BSK_WARNING = _simpleStorageUnit.BSK_WARNING
BSK_ERROR = _simpleStorageUnit.BSK_ERROR
BSK_SILENT = _simpleStorageUnit.BSK_SILENT

def printDefaultLogLevel():
    return _simpleStorageUnit.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _simpleStorageUnit.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _simpleStorageUnit.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _simpleStorageUnit.BSKLogger_swiginit(self, _simpleStorageUnit.new_BSKLogger(*args))
    __swig_destroy__ = _simpleStorageUnit.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _simpleStorageUnit.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _simpleStorageUnit.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _simpleStorageUnit.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _simpleStorageUnit.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_simpleStorageUnit.BSKLogger_logLevelMap_get, _simpleStorageUnit.BSKLogger_logLevelMap_set)

# Register BSKLogger in _simpleStorageUnit:
_simpleStorageUnit.BSKLogger_swigregister(BSKLogger)
cvar = _simpleStorageUnit.cvar


def _BSKLogger():
    return _simpleStorageUnit._BSKLogger()

def _BSKLogger_d(arg1):
    return _simpleStorageUnit._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _simpleStorageUnit._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _simpleStorageUnit._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _simpleStorageUnit._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _simpleStorageUnit.SysModel_swiginit(self, _simpleStorageUnit.new_SysModel(*args))
    __swig_destroy__ = _simpleStorageUnit.delete_SysModel

    def SelfInit(self):
        return _simpleStorageUnit.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _simpleStorageUnit.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _simpleStorageUnit.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _simpleStorageUnit.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_simpleStorageUnit.SysModel_ModelTag_get, _simpleStorageUnit.SysModel_ModelTag_set)
    CallCounts = property(_simpleStorageUnit.SysModel_CallCounts_get, _simpleStorageUnit.SysModel_CallCounts_set)
    RNGSeed = property(_simpleStorageUnit.SysModel_RNGSeed_get, _simpleStorageUnit.SysModel_RNGSeed_set)
    moduleID = property(_simpleStorageUnit.SysModel_moduleID_get, _simpleStorageUnit.SysModel_moduleID_set)

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


# Register SysModel in _simpleStorageUnit:
_simpleStorageUnit.SysModel_swigregister(SysModel)
class dataInstance(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dataInstanceName = property(_simpleStorageUnit.dataInstance_dataInstanceName_get, _simpleStorageUnit.dataInstance_dataInstanceName_set)
    dataInstanceSum = property(_simpleStorageUnit.dataInstance_dataInstanceSum_get, _simpleStorageUnit.dataInstance_dataInstanceSum_set)

    def __init__(self):
        _simpleStorageUnit.dataInstance_swiginit(self, _simpleStorageUnit.new_dataInstance())
    __swig_destroy__ = _simpleStorageUnit.delete_dataInstance

# Register dataInstance in _simpleStorageUnit:
_simpleStorageUnit.dataInstance_swigregister(dataInstance)
class DataStorageUnitBase(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _simpleStorageUnit.DataStorageUnitBase_swiginit(self, _simpleStorageUnit.new_DataStorageUnitBase())
    __swig_destroy__ = _simpleStorageUnit.delete_DataStorageUnitBase

    def Reset(self, CurrentSimNanos):
        return _simpleStorageUnit.DataStorageUnitBase_Reset(self, CurrentSimNanos)

    def addDataNodeToModel(self, tmpNodeMsg):
        return _simpleStorageUnit.DataStorageUnitBase_addDataNodeToModel(self, tmpNodeMsg)

    def UpdateState(self, CurrentSimNanos):
        return _simpleStorageUnit.DataStorageUnitBase_UpdateState(self, CurrentSimNanos)

    def setDataBuffer(self, partitionName, data):
        return _simpleStorageUnit.DataStorageUnitBase_setDataBuffer(self, partitionName, data)
    nodeDataUseInMsgs = property(_simpleStorageUnit.DataStorageUnitBase_nodeDataUseInMsgs_get, _simpleStorageUnit.DataStorageUnitBase_nodeDataUseInMsgs_set)
    storageUnitDataOutMsg = property(_simpleStorageUnit.DataStorageUnitBase_storageUnitDataOutMsg_get, _simpleStorageUnit.DataStorageUnitBase_storageUnitDataOutMsg_set)
    storageCapacity = property(_simpleStorageUnit.DataStorageUnitBase_storageCapacity_get, _simpleStorageUnit.DataStorageUnitBase_storageCapacity_set)
    bskLogger = property(_simpleStorageUnit.DataStorageUnitBase_bskLogger_get, _simpleStorageUnit.DataStorageUnitBase_bskLogger_set)

# Register DataStorageUnitBase in _simpleStorageUnit:
_simpleStorageUnit.DataStorageUnitBase_swigregister(DataStorageUnitBase)
class SimpleStorageUnit(DataStorageUnitBase):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _simpleStorageUnit.SimpleStorageUnit_swiginit(self, _simpleStorageUnit.new_SimpleStorageUnit())
    __swig_destroy__ = _simpleStorageUnit.delete_SimpleStorageUnit

    def setDataBuffer(self, data):
        return _simpleStorageUnit.SimpleStorageUnit_setDataBuffer(self, data)

# Register SimpleStorageUnit in _simpleStorageUnit:
_simpleStorageUnit.SimpleStorageUnit_swigregister(SimpleStorageUnit)
class DataNodeUsageMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dataName = property(_simpleStorageUnit.DataNodeUsageMsgPayload_dataName_get, _simpleStorageUnit.DataNodeUsageMsgPayload_dataName_set)
    baudRate = property(_simpleStorageUnit.DataNodeUsageMsgPayload_baudRate_get, _simpleStorageUnit.DataNodeUsageMsgPayload_baudRate_set)

    def __init__(self):
        _simpleStorageUnit.DataNodeUsageMsgPayload_swiginit(self, _simpleStorageUnit.new_DataNodeUsageMsgPayload())
    __swig_destroy__ = _simpleStorageUnit.delete_DataNodeUsageMsgPayload

# Register DataNodeUsageMsgPayload in _simpleStorageUnit:
_simpleStorageUnit.DataNodeUsageMsgPayload_swigregister(DataNodeUsageMsgPayload)
class DataStorageStatusMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    storageLevel = property(_simpleStorageUnit.DataStorageStatusMsgPayload_storageLevel_get, _simpleStorageUnit.DataStorageStatusMsgPayload_storageLevel_set)
    storageCapacity = property(_simpleStorageUnit.DataStorageStatusMsgPayload_storageCapacity_get, _simpleStorageUnit.DataStorageStatusMsgPayload_storageCapacity_set)
    currentNetBaud = property(_simpleStorageUnit.DataStorageStatusMsgPayload_currentNetBaud_get, _simpleStorageUnit.DataStorageStatusMsgPayload_currentNetBaud_set)
    storedDataName = property(_simpleStorageUnit.DataStorageStatusMsgPayload_storedDataName_get, _simpleStorageUnit.DataStorageStatusMsgPayload_storedDataName_set)
    storedData = property(_simpleStorageUnit.DataStorageStatusMsgPayload_storedData_get, _simpleStorageUnit.DataStorageStatusMsgPayload_storedData_set)

    def __init__(self):
        _simpleStorageUnit.DataStorageStatusMsgPayload_swiginit(self, _simpleStorageUnit.new_DataStorageStatusMsgPayload())
    __swig_destroy__ = _simpleStorageUnit.delete_DataStorageStatusMsgPayload

# Register DataStorageStatusMsgPayload in _simpleStorageUnit:
_simpleStorageUnit.DataStorageStatusMsgPayload_swigregister(DataStorageStatusMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


