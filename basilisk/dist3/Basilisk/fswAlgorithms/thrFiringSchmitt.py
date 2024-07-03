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
    from . import _thrFiringSchmitt
else:
    import _thrFiringSchmitt

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
    return _thrFiringSchmitt.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _thrFiringSchmitt.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _thrFiringSchmitt.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _thrFiringSchmitt.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _thrFiringSchmitt.new_longArray(nelements)

def delete_longArray(ary):
    return _thrFiringSchmitt.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _thrFiringSchmitt.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _thrFiringSchmitt.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _thrFiringSchmitt.new_intArray(nelements)

def delete_intArray(ary):
    return _thrFiringSchmitt.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _thrFiringSchmitt.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _thrFiringSchmitt.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _thrFiringSchmitt.new_shortArray(nelements)

def delete_shortArray(ary):
    return _thrFiringSchmitt.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _thrFiringSchmitt.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _thrFiringSchmitt.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _thrFiringSchmitt.MAX_LOGGING_LENGTH
BSK_DEBUG = _thrFiringSchmitt.BSK_DEBUG
BSK_INFORMATION = _thrFiringSchmitt.BSK_INFORMATION
BSK_WARNING = _thrFiringSchmitt.BSK_WARNING
BSK_ERROR = _thrFiringSchmitt.BSK_ERROR
BSK_SILENT = _thrFiringSchmitt.BSK_SILENT

def printDefaultLogLevel():
    return _thrFiringSchmitt.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _thrFiringSchmitt.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _thrFiringSchmitt.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _thrFiringSchmitt.BSKLogger_swiginit(self, _thrFiringSchmitt.new_BSKLogger(*args))
    __swig_destroy__ = _thrFiringSchmitt.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _thrFiringSchmitt.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _thrFiringSchmitt.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _thrFiringSchmitt.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _thrFiringSchmitt.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_thrFiringSchmitt.BSKLogger_logLevelMap_get, _thrFiringSchmitt.BSKLogger_logLevelMap_set)

# Register BSKLogger in _thrFiringSchmitt:
_thrFiringSchmitt.BSKLogger_swigregister(BSKLogger)
cvar = _thrFiringSchmitt.cvar


def _BSKLogger():
    return _thrFiringSchmitt._BSKLogger()

def _BSKLogger_d(arg1):
    return _thrFiringSchmitt._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _thrFiringSchmitt._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _thrFiringSchmitt._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _thrFiringSchmitt._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _thrFiringSchmitt.SysModel_swiginit(self, _thrFiringSchmitt.new_SysModel(*args))
    __swig_destroy__ = _thrFiringSchmitt.delete_SysModel

    def SelfInit(self):
        return _thrFiringSchmitt.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _thrFiringSchmitt.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _thrFiringSchmitt.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _thrFiringSchmitt.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_thrFiringSchmitt.SysModel_ModelTag_get, _thrFiringSchmitt.SysModel_ModelTag_set)
    CallCounts = property(_thrFiringSchmitt.SysModel_CallCounts_get, _thrFiringSchmitt.SysModel_CallCounts_set)
    RNGSeed = property(_thrFiringSchmitt.SysModel_RNGSeed_get, _thrFiringSchmitt.SysModel_RNGSeed_set)
    moduleID = property(_thrFiringSchmitt.SysModel_moduleID_get, _thrFiringSchmitt.SysModel_moduleID_set)

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


# Register SysModel in _thrFiringSchmitt:
_thrFiringSchmitt.SysModel_swigregister(SysModel)
class thrFiringSchmittConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    level_on = property(_thrFiringSchmitt.thrFiringSchmittConfig_level_on_get, _thrFiringSchmitt.thrFiringSchmittConfig_level_on_set)
    level_off = property(_thrFiringSchmitt.thrFiringSchmittConfig_level_off_get, _thrFiringSchmitt.thrFiringSchmittConfig_level_off_set)
    thrMinFireTime = property(_thrFiringSchmitt.thrFiringSchmittConfig_thrMinFireTime_get, _thrFiringSchmitt.thrFiringSchmittConfig_thrMinFireTime_set)
    baseThrustState = property(_thrFiringSchmitt.thrFiringSchmittConfig_baseThrustState_get, _thrFiringSchmitt.thrFiringSchmittConfig_baseThrustState_set)
    numThrusters = property(_thrFiringSchmitt.thrFiringSchmittConfig_numThrusters_get, _thrFiringSchmitt.thrFiringSchmittConfig_numThrusters_set)
    maxThrust = property(_thrFiringSchmitt.thrFiringSchmittConfig_maxThrust_get, _thrFiringSchmitt.thrFiringSchmittConfig_maxThrust_set)
    lastThrustState = property(_thrFiringSchmitt.thrFiringSchmittConfig_lastThrustState_get, _thrFiringSchmitt.thrFiringSchmittConfig_lastThrustState_set)
    prevCallTime = property(_thrFiringSchmitt.thrFiringSchmittConfig_prevCallTime_get, _thrFiringSchmitt.thrFiringSchmittConfig_prevCallTime_set)
    thrForceInMsg = property(_thrFiringSchmitt.thrFiringSchmittConfig_thrForceInMsg_get, _thrFiringSchmitt.thrFiringSchmittConfig_thrForceInMsg_set)
    onTimeOutMsg = property(_thrFiringSchmitt.thrFiringSchmittConfig_onTimeOutMsg_get, _thrFiringSchmitt.thrFiringSchmittConfig_onTimeOutMsg_set)
    thrConfInMsg = property(_thrFiringSchmitt.thrFiringSchmittConfig_thrConfInMsg_get, _thrFiringSchmitt.thrFiringSchmittConfig_thrConfInMsg_set)
    bskLogger = property(_thrFiringSchmitt.thrFiringSchmittConfig_bskLogger_get, _thrFiringSchmitt.thrFiringSchmittConfig_bskLogger_set)

    def createWrapper(self):
        return thrFiringSchmitt(self)


    def __init__(self):
        _thrFiringSchmitt.thrFiringSchmittConfig_swiginit(self, _thrFiringSchmitt.new_thrFiringSchmittConfig())
    __swig_destroy__ = _thrFiringSchmitt.delete_thrFiringSchmittConfig

# Register thrFiringSchmittConfig in _thrFiringSchmitt:
_thrFiringSchmitt.thrFiringSchmittConfig_swigregister(thrFiringSchmittConfig)
class thrFiringSchmitt(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _thrFiringSchmitt.thrFiringSchmitt_swiginit(self, _thrFiringSchmitt.new_thrFiringSchmitt(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _thrFiringSchmitt.thrFiringSchmitt_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _thrFiringSchmitt.thrFiringSchmitt_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _thrFiringSchmitt.thrFiringSchmitt_Reset(self, currentSimNanos)

    def __deref__(self):
        return _thrFiringSchmitt.thrFiringSchmitt___deref__(self)

    def getConfig(self):
        return _thrFiringSchmitt.thrFiringSchmitt_getConfig(self)
    __swig_destroy__ = _thrFiringSchmitt.delete_thrFiringSchmitt
    level_on = property(_thrFiringSchmitt.thrFiringSchmitt_level_on_get, _thrFiringSchmitt.thrFiringSchmitt_level_on_set)
    level_off = property(_thrFiringSchmitt.thrFiringSchmitt_level_off_get, _thrFiringSchmitt.thrFiringSchmitt_level_off_set)
    thrMinFireTime = property(_thrFiringSchmitt.thrFiringSchmitt_thrMinFireTime_get, _thrFiringSchmitt.thrFiringSchmitt_thrMinFireTime_set)
    baseThrustState = property(_thrFiringSchmitt.thrFiringSchmitt_baseThrustState_get, _thrFiringSchmitt.thrFiringSchmitt_baseThrustState_set)
    numThrusters = property(_thrFiringSchmitt.thrFiringSchmitt_numThrusters_get, _thrFiringSchmitt.thrFiringSchmitt_numThrusters_set)
    maxThrust = property(_thrFiringSchmitt.thrFiringSchmitt_maxThrust_get, _thrFiringSchmitt.thrFiringSchmitt_maxThrust_set)
    lastThrustState = property(_thrFiringSchmitt.thrFiringSchmitt_lastThrustState_get, _thrFiringSchmitt.thrFiringSchmitt_lastThrustState_set)
    prevCallTime = property(_thrFiringSchmitt.thrFiringSchmitt_prevCallTime_get, _thrFiringSchmitt.thrFiringSchmitt_prevCallTime_set)
    thrForceInMsg = property(_thrFiringSchmitt.thrFiringSchmitt_thrForceInMsg_get, _thrFiringSchmitt.thrFiringSchmitt_thrForceInMsg_set)
    onTimeOutMsg = property(_thrFiringSchmitt.thrFiringSchmitt_onTimeOutMsg_get, _thrFiringSchmitt.thrFiringSchmitt_onTimeOutMsg_set)
    thrConfInMsg = property(_thrFiringSchmitt.thrFiringSchmitt_thrConfInMsg_get, _thrFiringSchmitt.thrFiringSchmitt_thrConfInMsg_set)
    bskLogger = property(_thrFiringSchmitt.thrFiringSchmitt_bskLogger_get, _thrFiringSchmitt.thrFiringSchmitt_bskLogger_set)

# Register thrFiringSchmitt in _thrFiringSchmitt:
_thrFiringSchmitt.thrFiringSchmitt_swigregister(thrFiringSchmitt)
class THRArrayConfigMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    numThrusters = property(_thrFiringSchmitt.THRArrayConfigMsgPayload_numThrusters_get, _thrFiringSchmitt.THRArrayConfigMsgPayload_numThrusters_set)
    thrusters = property(_thrFiringSchmitt.THRArrayConfigMsgPayload_thrusters_get, _thrFiringSchmitt.THRArrayConfigMsgPayload_thrusters_set)

    def __init__(self):
        _thrFiringSchmitt.THRArrayConfigMsgPayload_swiginit(self, _thrFiringSchmitt.new_THRArrayConfigMsgPayload())
    __swig_destroy__ = _thrFiringSchmitt.delete_THRArrayConfigMsgPayload

# Register THRArrayConfigMsgPayload in _thrFiringSchmitt:
_thrFiringSchmitt.THRArrayConfigMsgPayload_swigregister(THRArrayConfigMsgPayload)
class THRArrayCmdForceMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    thrForce = property(_thrFiringSchmitt.THRArrayCmdForceMsgPayload_thrForce_get, _thrFiringSchmitt.THRArrayCmdForceMsgPayload_thrForce_set)

    def __init__(self):
        _thrFiringSchmitt.THRArrayCmdForceMsgPayload_swiginit(self, _thrFiringSchmitt.new_THRArrayCmdForceMsgPayload())
    __swig_destroy__ = _thrFiringSchmitt.delete_THRArrayCmdForceMsgPayload

# Register THRArrayCmdForceMsgPayload in _thrFiringSchmitt:
_thrFiringSchmitt.THRArrayCmdForceMsgPayload_swigregister(THRArrayCmdForceMsgPayload)
class THRArrayOnTimeCmdMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    OnTimeRequest = property(_thrFiringSchmitt.THRArrayOnTimeCmdMsgPayload_OnTimeRequest_get, _thrFiringSchmitt.THRArrayOnTimeCmdMsgPayload_OnTimeRequest_set)

    def __init__(self):
        _thrFiringSchmitt.THRArrayOnTimeCmdMsgPayload_swiginit(self, _thrFiringSchmitt.new_THRArrayOnTimeCmdMsgPayload())
    __swig_destroy__ = _thrFiringSchmitt.delete_THRArrayOnTimeCmdMsgPayload

# Register THRArrayOnTimeCmdMsgPayload in _thrFiringSchmitt:
_thrFiringSchmitt.THRArrayOnTimeCmdMsgPayload_swigregister(THRArrayOnTimeCmdMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


