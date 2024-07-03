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
    from . import _simpleInstrument
else:
    import _simpleInstrument

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


from Basilisk.architecture.swig_common_model import *


def new_doubleArray(nelements):
    return _simpleInstrument.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _simpleInstrument.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _simpleInstrument.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _simpleInstrument.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _simpleInstrument.new_longArray(nelements)

def delete_longArray(ary):
    return _simpleInstrument.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _simpleInstrument.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _simpleInstrument.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _simpleInstrument.new_intArray(nelements)

def delete_intArray(ary):
    return _simpleInstrument.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _simpleInstrument.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _simpleInstrument.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _simpleInstrument.new_shortArray(nelements)

def delete_shortArray(ary):
    return _simpleInstrument.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _simpleInstrument.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _simpleInstrument.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _simpleInstrument.MAX_LOGGING_LENGTH
BSK_DEBUG = _simpleInstrument.BSK_DEBUG
BSK_INFORMATION = _simpleInstrument.BSK_INFORMATION
BSK_WARNING = _simpleInstrument.BSK_WARNING
BSK_ERROR = _simpleInstrument.BSK_ERROR
BSK_SILENT = _simpleInstrument.BSK_SILENT

def printDefaultLogLevel():
    return _simpleInstrument.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _simpleInstrument.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _simpleInstrument.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _simpleInstrument.BSKLogger_swiginit(self, _simpleInstrument.new_BSKLogger(*args))
    __swig_destroy__ = _simpleInstrument.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _simpleInstrument.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _simpleInstrument.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _simpleInstrument.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _simpleInstrument.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_simpleInstrument.BSKLogger_logLevelMap_get, _simpleInstrument.BSKLogger_logLevelMap_set)

# Register BSKLogger in _simpleInstrument:
_simpleInstrument.BSKLogger_swigregister(BSKLogger)
cvar = _simpleInstrument.cvar


def _BSKLogger():
    return _simpleInstrument._BSKLogger()

def _BSKLogger_d(arg1):
    return _simpleInstrument._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _simpleInstrument._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _simpleInstrument._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _simpleInstrument._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _simpleInstrument.SysModel_swiginit(self, _simpleInstrument.new_SysModel(*args))
    __swig_destroy__ = _simpleInstrument.delete_SysModel

    def SelfInit(self):
        return _simpleInstrument.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _simpleInstrument.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _simpleInstrument.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _simpleInstrument.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_simpleInstrument.SysModel_ModelTag_get, _simpleInstrument.SysModel_ModelTag_set)
    CallCounts = property(_simpleInstrument.SysModel_CallCounts_get, _simpleInstrument.SysModel_CallCounts_set)
    RNGSeed = property(_simpleInstrument.SysModel_RNGSeed_get, _simpleInstrument.SysModel_RNGSeed_set)
    moduleID = property(_simpleInstrument.SysModel_moduleID_get, _simpleInstrument.SysModel_moduleID_set)

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


# Register SysModel in _simpleInstrument:
_simpleInstrument.SysModel_swigregister(SysModel)
class DataNodeBase(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _simpleInstrument.delete_DataNodeBase

    def Reset(self, CurrentSimNanos):
        return _simpleInstrument.DataNodeBase_Reset(self, CurrentSimNanos)

    def computeDataStatus(self, currentTime):
        return _simpleInstrument.DataNodeBase_computeDataStatus(self, currentTime)

    def UpdateState(self, CurrentSimNanos):
        return _simpleInstrument.DataNodeBase_UpdateState(self, CurrentSimNanos)
    nodeDataOutMsg = property(_simpleInstrument.DataNodeBase_nodeDataOutMsg_get, _simpleInstrument.DataNodeBase_nodeDataOutMsg_set)
    nodeStatusInMsg = property(_simpleInstrument.DataNodeBase_nodeStatusInMsg_get, _simpleInstrument.DataNodeBase_nodeStatusInMsg_set)
    nodeBaudRate = property(_simpleInstrument.DataNodeBase_nodeBaudRate_get, _simpleInstrument.DataNodeBase_nodeBaudRate_set)
    nodeDataName = property(_simpleInstrument.DataNodeBase_nodeDataName_get, _simpleInstrument.DataNodeBase_nodeDataName_set)
    dataStatus = property(_simpleInstrument.DataNodeBase_dataStatus_get, _simpleInstrument.DataNodeBase_dataStatus_set)

# Register DataNodeBase in _simpleInstrument:
_simpleInstrument.DataNodeBase_swigregister(DataNodeBase)
class SimpleInstrument(DataNodeBase):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _simpleInstrument.SimpleInstrument_swiginit(self, _simpleInstrument.new_SimpleInstrument())
    __swig_destroy__ = _simpleInstrument.delete_SimpleInstrument

# Register SimpleInstrument in _simpleInstrument:
_simpleInstrument.SimpleInstrument_swigregister(SimpleInstrument)
class DataNodeUsageMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dataName = property(_simpleInstrument.DataNodeUsageMsgPayload_dataName_get, _simpleInstrument.DataNodeUsageMsgPayload_dataName_set)
    baudRate = property(_simpleInstrument.DataNodeUsageMsgPayload_baudRate_get, _simpleInstrument.DataNodeUsageMsgPayload_baudRate_set)

    def __init__(self):
        _simpleInstrument.DataNodeUsageMsgPayload_swiginit(self, _simpleInstrument.new_DataNodeUsageMsgPayload())
    __swig_destroy__ = _simpleInstrument.delete_DataNodeUsageMsgPayload

# Register DataNodeUsageMsgPayload in _simpleInstrument:
_simpleInstrument.DataNodeUsageMsgPayload_swigregister(DataNodeUsageMsgPayload)
class DeviceCmdMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    deviceCmd = property(_simpleInstrument.DeviceCmdMsgPayload_deviceCmd_get, _simpleInstrument.DeviceCmdMsgPayload_deviceCmd_set)

    def __init__(self):
        _simpleInstrument.DeviceCmdMsgPayload_swiginit(self, _simpleInstrument.new_DeviceCmdMsgPayload())
    __swig_destroy__ = _simpleInstrument.delete_DeviceCmdMsgPayload

# Register DeviceCmdMsgPayload in _simpleInstrument:
_simpleInstrument.DeviceCmdMsgPayload_swigregister(DeviceCmdMsgPayload)
class DataStorageStatusMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    storageLevel = property(_simpleInstrument.DataStorageStatusMsgPayload_storageLevel_get, _simpleInstrument.DataStorageStatusMsgPayload_storageLevel_set)
    storageCapacity = property(_simpleInstrument.DataStorageStatusMsgPayload_storageCapacity_get, _simpleInstrument.DataStorageStatusMsgPayload_storageCapacity_set)
    currentNetBaud = property(_simpleInstrument.DataStorageStatusMsgPayload_currentNetBaud_get, _simpleInstrument.DataStorageStatusMsgPayload_currentNetBaud_set)
    storedDataName = property(_simpleInstrument.DataStorageStatusMsgPayload_storedDataName_get, _simpleInstrument.DataStorageStatusMsgPayload_storedDataName_set)
    storedData = property(_simpleInstrument.DataStorageStatusMsgPayload_storedData_get, _simpleInstrument.DataStorageStatusMsgPayload_storedData_set)

    def __init__(self):
        _simpleInstrument.DataStorageStatusMsgPayload_swiginit(self, _simpleInstrument.new_DataStorageStatusMsgPayload())
    __swig_destroy__ = _simpleInstrument.delete_DataStorageStatusMsgPayload

# Register DataStorageStatusMsgPayload in _simpleInstrument:
_simpleInstrument.DataStorageStatusMsgPayload_swigregister(DataStorageStatusMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


