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
    from . import _rwMotorVoltage
else:
    import _rwMotorVoltage

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
    return _rwMotorVoltage.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _rwMotorVoltage.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _rwMotorVoltage.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _rwMotorVoltage.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _rwMotorVoltage.new_longArray(nelements)

def delete_longArray(ary):
    return _rwMotorVoltage.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _rwMotorVoltage.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _rwMotorVoltage.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _rwMotorVoltage.new_intArray(nelements)

def delete_intArray(ary):
    return _rwMotorVoltage.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _rwMotorVoltage.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _rwMotorVoltage.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _rwMotorVoltage.new_shortArray(nelements)

def delete_shortArray(ary):
    return _rwMotorVoltage.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _rwMotorVoltage.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _rwMotorVoltage.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _rwMotorVoltage.MAX_LOGGING_LENGTH
BSK_DEBUG = _rwMotorVoltage.BSK_DEBUG
BSK_INFORMATION = _rwMotorVoltage.BSK_INFORMATION
BSK_WARNING = _rwMotorVoltage.BSK_WARNING
BSK_ERROR = _rwMotorVoltage.BSK_ERROR
BSK_SILENT = _rwMotorVoltage.BSK_SILENT

def printDefaultLogLevel():
    return _rwMotorVoltage.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _rwMotorVoltage.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _rwMotorVoltage.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _rwMotorVoltage.BSKLogger_swiginit(self, _rwMotorVoltage.new_BSKLogger(*args))
    __swig_destroy__ = _rwMotorVoltage.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _rwMotorVoltage.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _rwMotorVoltage.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _rwMotorVoltage.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _rwMotorVoltage.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_rwMotorVoltage.BSKLogger_logLevelMap_get, _rwMotorVoltage.BSKLogger_logLevelMap_set)

# Register BSKLogger in _rwMotorVoltage:
_rwMotorVoltage.BSKLogger_swigregister(BSKLogger)
cvar = _rwMotorVoltage.cvar


def _BSKLogger():
    return _rwMotorVoltage._BSKLogger()

def _BSKLogger_d(arg1):
    return _rwMotorVoltage._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _rwMotorVoltage._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _rwMotorVoltage._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _rwMotorVoltage._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _rwMotorVoltage.SysModel_swiginit(self, _rwMotorVoltage.new_SysModel(*args))
    __swig_destroy__ = _rwMotorVoltage.delete_SysModel

    def SelfInit(self):
        return _rwMotorVoltage.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _rwMotorVoltage.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _rwMotorVoltage.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _rwMotorVoltage.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_rwMotorVoltage.SysModel_ModelTag_get, _rwMotorVoltage.SysModel_ModelTag_set)
    CallCounts = property(_rwMotorVoltage.SysModel_CallCounts_get, _rwMotorVoltage.SysModel_CallCounts_set)
    RNGSeed = property(_rwMotorVoltage.SysModel_RNGSeed_get, _rwMotorVoltage.SysModel_RNGSeed_set)
    moduleID = property(_rwMotorVoltage.SysModel_moduleID_get, _rwMotorVoltage.SysModel_moduleID_set)

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


# Register SysModel in _rwMotorVoltage:
_rwMotorVoltage.SysModel_swigregister(SysModel)
class rwMotorVoltageConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    VMin = property(_rwMotorVoltage.rwMotorVoltageConfig_VMin_get, _rwMotorVoltage.rwMotorVoltageConfig_VMin_set)
    VMax = property(_rwMotorVoltage.rwMotorVoltageConfig_VMax_get, _rwMotorVoltage.rwMotorVoltageConfig_VMax_set)
    K = property(_rwMotorVoltage.rwMotorVoltageConfig_K_get, _rwMotorVoltage.rwMotorVoltageConfig_K_set)
    rwSpeedOld = property(_rwMotorVoltage.rwMotorVoltageConfig_rwSpeedOld_get, _rwMotorVoltage.rwMotorVoltageConfig_rwSpeedOld_set)
    priorTime = property(_rwMotorVoltage.rwMotorVoltageConfig_priorTime_get, _rwMotorVoltage.rwMotorVoltageConfig_priorTime_set)
    resetFlag = property(_rwMotorVoltage.rwMotorVoltageConfig_resetFlag_get, _rwMotorVoltage.rwMotorVoltageConfig_resetFlag_set)
    voltageOutMsg = property(_rwMotorVoltage.rwMotorVoltageConfig_voltageOutMsg_get, _rwMotorVoltage.rwMotorVoltageConfig_voltageOutMsg_set)
    torqueInMsg = property(_rwMotorVoltage.rwMotorVoltageConfig_torqueInMsg_get, _rwMotorVoltage.rwMotorVoltageConfig_torqueInMsg_set)
    rwParamsInMsg = property(_rwMotorVoltage.rwMotorVoltageConfig_rwParamsInMsg_get, _rwMotorVoltage.rwMotorVoltageConfig_rwParamsInMsg_set)
    rwSpeedInMsg = property(_rwMotorVoltage.rwMotorVoltageConfig_rwSpeedInMsg_get, _rwMotorVoltage.rwMotorVoltageConfig_rwSpeedInMsg_set)
    rwAvailInMsg = property(_rwMotorVoltage.rwMotorVoltageConfig_rwAvailInMsg_get, _rwMotorVoltage.rwMotorVoltageConfig_rwAvailInMsg_set)
    rwConfigParams = property(_rwMotorVoltage.rwMotorVoltageConfig_rwConfigParams_get, _rwMotorVoltage.rwMotorVoltageConfig_rwConfigParams_set)
    bskLogger = property(_rwMotorVoltage.rwMotorVoltageConfig_bskLogger_get, _rwMotorVoltage.rwMotorVoltageConfig_bskLogger_set)

    def createWrapper(self):
        return rwMotorVoltage(self)


    def __init__(self):
        _rwMotorVoltage.rwMotorVoltageConfig_swiginit(self, _rwMotorVoltage.new_rwMotorVoltageConfig())
    __swig_destroy__ = _rwMotorVoltage.delete_rwMotorVoltageConfig

# Register rwMotorVoltageConfig in _rwMotorVoltage:
_rwMotorVoltage.rwMotorVoltageConfig_swigregister(rwMotorVoltageConfig)
class rwMotorVoltage(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _rwMotorVoltage.rwMotorVoltage_swiginit(self, _rwMotorVoltage.new_rwMotorVoltage(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _rwMotorVoltage.rwMotorVoltage_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _rwMotorVoltage.rwMotorVoltage_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _rwMotorVoltage.rwMotorVoltage_Reset(self, currentSimNanos)

    def __deref__(self):
        return _rwMotorVoltage.rwMotorVoltage___deref__(self)

    def getConfig(self):
        return _rwMotorVoltage.rwMotorVoltage_getConfig(self)
    __swig_destroy__ = _rwMotorVoltage.delete_rwMotorVoltage
    VMin = property(_rwMotorVoltage.rwMotorVoltage_VMin_get, _rwMotorVoltage.rwMotorVoltage_VMin_set)
    VMax = property(_rwMotorVoltage.rwMotorVoltage_VMax_get, _rwMotorVoltage.rwMotorVoltage_VMax_set)
    K = property(_rwMotorVoltage.rwMotorVoltage_K_get, _rwMotorVoltage.rwMotorVoltage_K_set)
    rwSpeedOld = property(_rwMotorVoltage.rwMotorVoltage_rwSpeedOld_get, _rwMotorVoltage.rwMotorVoltage_rwSpeedOld_set)
    priorTime = property(_rwMotorVoltage.rwMotorVoltage_priorTime_get, _rwMotorVoltage.rwMotorVoltage_priorTime_set)
    resetFlag = property(_rwMotorVoltage.rwMotorVoltage_resetFlag_get, _rwMotorVoltage.rwMotorVoltage_resetFlag_set)
    voltageOutMsg = property(_rwMotorVoltage.rwMotorVoltage_voltageOutMsg_get, _rwMotorVoltage.rwMotorVoltage_voltageOutMsg_set)
    torqueInMsg = property(_rwMotorVoltage.rwMotorVoltage_torqueInMsg_get, _rwMotorVoltage.rwMotorVoltage_torqueInMsg_set)
    rwParamsInMsg = property(_rwMotorVoltage.rwMotorVoltage_rwParamsInMsg_get, _rwMotorVoltage.rwMotorVoltage_rwParamsInMsg_set)
    rwSpeedInMsg = property(_rwMotorVoltage.rwMotorVoltage_rwSpeedInMsg_get, _rwMotorVoltage.rwMotorVoltage_rwSpeedInMsg_set)
    rwAvailInMsg = property(_rwMotorVoltage.rwMotorVoltage_rwAvailInMsg_get, _rwMotorVoltage.rwMotorVoltage_rwAvailInMsg_set)
    rwConfigParams = property(_rwMotorVoltage.rwMotorVoltage_rwConfigParams_get, _rwMotorVoltage.rwMotorVoltage_rwConfigParams_set)
    bskLogger = property(_rwMotorVoltage.rwMotorVoltage_bskLogger_get, _rwMotorVoltage.rwMotorVoltage_bskLogger_set)

# Register rwMotorVoltage in _rwMotorVoltage:
_rwMotorVoltage.rwMotorVoltage_swigregister(rwMotorVoltage)
class ArrayMotorTorqueMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    motorTorque = property(_rwMotorVoltage.ArrayMotorTorqueMsgPayload_motorTorque_get, _rwMotorVoltage.ArrayMotorTorqueMsgPayload_motorTorque_set)

    def __init__(self):
        _rwMotorVoltage.ArrayMotorTorqueMsgPayload_swiginit(self, _rwMotorVoltage.new_ArrayMotorTorqueMsgPayload())
    __swig_destroy__ = _rwMotorVoltage.delete_ArrayMotorTorqueMsgPayload

# Register ArrayMotorTorqueMsgPayload in _rwMotorVoltage:
_rwMotorVoltage.ArrayMotorTorqueMsgPayload_swigregister(ArrayMotorTorqueMsgPayload)
class RWAvailabilityMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    wheelAvailability = property(_rwMotorVoltage.RWAvailabilityMsgPayload_wheelAvailability_get, _rwMotorVoltage.RWAvailabilityMsgPayload_wheelAvailability_set)

    def __init__(self):
        _rwMotorVoltage.RWAvailabilityMsgPayload_swiginit(self, _rwMotorVoltage.new_RWAvailabilityMsgPayload())
    __swig_destroy__ = _rwMotorVoltage.delete_RWAvailabilityMsgPayload

# Register RWAvailabilityMsgPayload in _rwMotorVoltage:
_rwMotorVoltage.RWAvailabilityMsgPayload_swigregister(RWAvailabilityMsgPayload)
class RWArrayConfigMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    GsMatrix_B = property(_rwMotorVoltage.RWArrayConfigMsgPayload_GsMatrix_B_get, _rwMotorVoltage.RWArrayConfigMsgPayload_GsMatrix_B_set)
    JsList = property(_rwMotorVoltage.RWArrayConfigMsgPayload_JsList_get, _rwMotorVoltage.RWArrayConfigMsgPayload_JsList_set)
    numRW = property(_rwMotorVoltage.RWArrayConfigMsgPayload_numRW_get, _rwMotorVoltage.RWArrayConfigMsgPayload_numRW_set)
    uMax = property(_rwMotorVoltage.RWArrayConfigMsgPayload_uMax_get, _rwMotorVoltage.RWArrayConfigMsgPayload_uMax_set)

    def __init__(self):
        _rwMotorVoltage.RWArrayConfigMsgPayload_swiginit(self, _rwMotorVoltage.new_RWArrayConfigMsgPayload())
    __swig_destroy__ = _rwMotorVoltage.delete_RWArrayConfigMsgPayload

# Register RWArrayConfigMsgPayload in _rwMotorVoltage:
_rwMotorVoltage.RWArrayConfigMsgPayload_swigregister(RWArrayConfigMsgPayload)
class RWSpeedMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    wheelSpeeds = property(_rwMotorVoltage.RWSpeedMsgPayload_wheelSpeeds_get, _rwMotorVoltage.RWSpeedMsgPayload_wheelSpeeds_set)
    wheelThetas = property(_rwMotorVoltage.RWSpeedMsgPayload_wheelThetas_get, _rwMotorVoltage.RWSpeedMsgPayload_wheelThetas_set)

    def __init__(self):
        _rwMotorVoltage.RWSpeedMsgPayload_swiginit(self, _rwMotorVoltage.new_RWSpeedMsgPayload())
    __swig_destroy__ = _rwMotorVoltage.delete_RWSpeedMsgPayload

# Register RWSpeedMsgPayload in _rwMotorVoltage:
_rwMotorVoltage.RWSpeedMsgPayload_swigregister(RWSpeedMsgPayload)
class ArrayMotorVoltageMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    voltage = property(_rwMotorVoltage.ArrayMotorVoltageMsgPayload_voltage_get, _rwMotorVoltage.ArrayMotorVoltageMsgPayload_voltage_set)

    def __init__(self):
        _rwMotorVoltage.ArrayMotorVoltageMsgPayload_swiginit(self, _rwMotorVoltage.new_ArrayMotorVoltageMsgPayload())
    __swig_destroy__ = _rwMotorVoltage.delete_ArrayMotorVoltageMsgPayload

# Register ArrayMotorVoltageMsgPayload in _rwMotorVoltage:
_rwMotorVoltage.ArrayMotorVoltageMsgPayload_swigregister(ArrayMotorVoltageMsgPayload)
BOOL_FALSE = _rwMotorVoltage.BOOL_FALSE
BOOL_TRUE = _rwMotorVoltage.BOOL_TRUE
AVAILABLE = _rwMotorVoltage.AVAILABLE
UNAVAILABLE = _rwMotorVoltage.UNAVAILABLE
MAX_CIRCLE_NUM = _rwMotorVoltage.MAX_CIRCLE_NUM
MAX_LIMB_PNTS = _rwMotorVoltage.MAX_LIMB_PNTS
MAX_EFF_CNT = _rwMotorVoltage.MAX_EFF_CNT
MAX_NUM_CSS_SENSORS = _rwMotorVoltage.MAX_NUM_CSS_SENSORS
MAX_ST_VEH_COUNT = _rwMotorVoltage.MAX_ST_VEH_COUNT
NANO2SEC = _rwMotorVoltage.NANO2SEC
SEC2NANO = _rwMotorVoltage.SEC2NANO
SEC2HOUR = _rwMotorVoltage.SEC2HOUR

import sys
protectAllClasses(sys.modules[__name__])


