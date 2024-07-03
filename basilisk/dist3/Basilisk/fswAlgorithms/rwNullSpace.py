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
    from . import _rwNullSpace
else:
    import _rwNullSpace

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
    return _rwNullSpace.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _rwNullSpace.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _rwNullSpace.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _rwNullSpace.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _rwNullSpace.new_longArray(nelements)

def delete_longArray(ary):
    return _rwNullSpace.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _rwNullSpace.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _rwNullSpace.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _rwNullSpace.new_intArray(nelements)

def delete_intArray(ary):
    return _rwNullSpace.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _rwNullSpace.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _rwNullSpace.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _rwNullSpace.new_shortArray(nelements)

def delete_shortArray(ary):
    return _rwNullSpace.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _rwNullSpace.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _rwNullSpace.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _rwNullSpace.MAX_LOGGING_LENGTH
BSK_DEBUG = _rwNullSpace.BSK_DEBUG
BSK_INFORMATION = _rwNullSpace.BSK_INFORMATION
BSK_WARNING = _rwNullSpace.BSK_WARNING
BSK_ERROR = _rwNullSpace.BSK_ERROR
BSK_SILENT = _rwNullSpace.BSK_SILENT

def printDefaultLogLevel():
    return _rwNullSpace.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _rwNullSpace.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _rwNullSpace.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _rwNullSpace.BSKLogger_swiginit(self, _rwNullSpace.new_BSKLogger(*args))
    __swig_destroy__ = _rwNullSpace.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _rwNullSpace.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _rwNullSpace.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _rwNullSpace.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _rwNullSpace.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_rwNullSpace.BSKLogger_logLevelMap_get, _rwNullSpace.BSKLogger_logLevelMap_set)

# Register BSKLogger in _rwNullSpace:
_rwNullSpace.BSKLogger_swigregister(BSKLogger)
cvar = _rwNullSpace.cvar


def _BSKLogger():
    return _rwNullSpace._BSKLogger()

def _BSKLogger_d(arg1):
    return _rwNullSpace._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _rwNullSpace._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _rwNullSpace._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _rwNullSpace._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _rwNullSpace.SysModel_swiginit(self, _rwNullSpace.new_SysModel(*args))
    __swig_destroy__ = _rwNullSpace.delete_SysModel

    def SelfInit(self):
        return _rwNullSpace.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _rwNullSpace.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _rwNullSpace.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _rwNullSpace.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_rwNullSpace.SysModel_ModelTag_get, _rwNullSpace.SysModel_ModelTag_set)
    CallCounts = property(_rwNullSpace.SysModel_CallCounts_get, _rwNullSpace.SysModel_CallCounts_set)
    RNGSeed = property(_rwNullSpace.SysModel_RNGSeed_get, _rwNullSpace.SysModel_RNGSeed_set)
    moduleID = property(_rwNullSpace.SysModel_moduleID_get, _rwNullSpace.SysModel_moduleID_set)

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


# Register SysModel in _rwNullSpace:
_rwNullSpace.SysModel_swigregister(SysModel)
class rwNullSpaceConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    rwMotorTorqueInMsg = property(_rwNullSpace.rwNullSpaceConfig_rwMotorTorqueInMsg_get, _rwNullSpace.rwNullSpaceConfig_rwMotorTorqueInMsg_set)
    rwSpeedsInMsg = property(_rwNullSpace.rwNullSpaceConfig_rwSpeedsInMsg_get, _rwNullSpace.rwNullSpaceConfig_rwSpeedsInMsg_set)
    rwDesiredSpeedsInMsg = property(_rwNullSpace.rwNullSpaceConfig_rwDesiredSpeedsInMsg_get, _rwNullSpace.rwNullSpaceConfig_rwDesiredSpeedsInMsg_set)
    rwConfigInMsg = property(_rwNullSpace.rwNullSpaceConfig_rwConfigInMsg_get, _rwNullSpace.rwNullSpaceConfig_rwConfigInMsg_set)
    rwMotorTorqueOutMsg = property(_rwNullSpace.rwNullSpaceConfig_rwMotorTorqueOutMsg_get, _rwNullSpace.rwNullSpaceConfig_rwMotorTorqueOutMsg_set)
    tau = property(_rwNullSpace.rwNullSpaceConfig_tau_get, _rwNullSpace.rwNullSpaceConfig_tau_set)
    OmegaGain = property(_rwNullSpace.rwNullSpaceConfig_OmegaGain_get, _rwNullSpace.rwNullSpaceConfig_OmegaGain_set)
    numWheels = property(_rwNullSpace.rwNullSpaceConfig_numWheels_get, _rwNullSpace.rwNullSpaceConfig_numWheels_set)
    bskLogger = property(_rwNullSpace.rwNullSpaceConfig_bskLogger_get, _rwNullSpace.rwNullSpaceConfig_bskLogger_set)

    def createWrapper(self):
        return rwNullSpace(self)


    def __init__(self):
        _rwNullSpace.rwNullSpaceConfig_swiginit(self, _rwNullSpace.new_rwNullSpaceConfig())
    __swig_destroy__ = _rwNullSpace.delete_rwNullSpaceConfig

# Register rwNullSpaceConfig in _rwNullSpace:
_rwNullSpace.rwNullSpaceConfig_swigregister(rwNullSpaceConfig)
class rwNullSpace(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _rwNullSpace.rwNullSpace_swiginit(self, _rwNullSpace.new_rwNullSpace(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _rwNullSpace.rwNullSpace_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _rwNullSpace.rwNullSpace_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _rwNullSpace.rwNullSpace_Reset(self, currentSimNanos)

    def __deref__(self):
        return _rwNullSpace.rwNullSpace___deref__(self)

    def getConfig(self):
        return _rwNullSpace.rwNullSpace_getConfig(self)
    __swig_destroy__ = _rwNullSpace.delete_rwNullSpace
    rwMotorTorqueInMsg = property(_rwNullSpace.rwNullSpace_rwMotorTorqueInMsg_get, _rwNullSpace.rwNullSpace_rwMotorTorqueInMsg_set)
    rwSpeedsInMsg = property(_rwNullSpace.rwNullSpace_rwSpeedsInMsg_get, _rwNullSpace.rwNullSpace_rwSpeedsInMsg_set)
    rwDesiredSpeedsInMsg = property(_rwNullSpace.rwNullSpace_rwDesiredSpeedsInMsg_get, _rwNullSpace.rwNullSpace_rwDesiredSpeedsInMsg_set)
    rwConfigInMsg = property(_rwNullSpace.rwNullSpace_rwConfigInMsg_get, _rwNullSpace.rwNullSpace_rwConfigInMsg_set)
    rwMotorTorqueOutMsg = property(_rwNullSpace.rwNullSpace_rwMotorTorqueOutMsg_get, _rwNullSpace.rwNullSpace_rwMotorTorqueOutMsg_set)
    tau = property(_rwNullSpace.rwNullSpace_tau_get, _rwNullSpace.rwNullSpace_tau_set)
    OmegaGain = property(_rwNullSpace.rwNullSpace_OmegaGain_get, _rwNullSpace.rwNullSpace_OmegaGain_set)
    numWheels = property(_rwNullSpace.rwNullSpace_numWheels_get, _rwNullSpace.rwNullSpace_numWheels_set)
    bskLogger = property(_rwNullSpace.rwNullSpace_bskLogger_get, _rwNullSpace.rwNullSpace_bskLogger_set)

# Register rwNullSpace in _rwNullSpace:
_rwNullSpace.rwNullSpace_swigregister(rwNullSpace)
class ArrayMotorTorqueMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    motorTorque = property(_rwNullSpace.ArrayMotorTorqueMsgPayload_motorTorque_get, _rwNullSpace.ArrayMotorTorqueMsgPayload_motorTorque_set)

    def __init__(self):
        _rwNullSpace.ArrayMotorTorqueMsgPayload_swiginit(self, _rwNullSpace.new_ArrayMotorTorqueMsgPayload())
    __swig_destroy__ = _rwNullSpace.delete_ArrayMotorTorqueMsgPayload

# Register ArrayMotorTorqueMsgPayload in _rwNullSpace:
_rwNullSpace.ArrayMotorTorqueMsgPayload_swigregister(ArrayMotorTorqueMsgPayload)
class RWSpeedMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    wheelSpeeds = property(_rwNullSpace.RWSpeedMsgPayload_wheelSpeeds_get, _rwNullSpace.RWSpeedMsgPayload_wheelSpeeds_set)
    wheelThetas = property(_rwNullSpace.RWSpeedMsgPayload_wheelThetas_get, _rwNullSpace.RWSpeedMsgPayload_wheelThetas_set)

    def __init__(self):
        _rwNullSpace.RWSpeedMsgPayload_swiginit(self, _rwNullSpace.new_RWSpeedMsgPayload())
    __swig_destroy__ = _rwNullSpace.delete_RWSpeedMsgPayload

# Register RWSpeedMsgPayload in _rwNullSpace:
_rwNullSpace.RWSpeedMsgPayload_swigregister(RWSpeedMsgPayload)
class RWConstellationMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    numRW = property(_rwNullSpace.RWConstellationMsgPayload_numRW_get, _rwNullSpace.RWConstellationMsgPayload_numRW_set)
    reactionWheels = property(_rwNullSpace.RWConstellationMsgPayload_reactionWheels_get, _rwNullSpace.RWConstellationMsgPayload_reactionWheels_set)

    def __init__(self):
        _rwNullSpace.RWConstellationMsgPayload_swiginit(self, _rwNullSpace.new_RWConstellationMsgPayload())
    __swig_destroy__ = _rwNullSpace.delete_RWConstellationMsgPayload

# Register RWConstellationMsgPayload in _rwNullSpace:
_rwNullSpace.RWConstellationMsgPayload_swigregister(RWConstellationMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


