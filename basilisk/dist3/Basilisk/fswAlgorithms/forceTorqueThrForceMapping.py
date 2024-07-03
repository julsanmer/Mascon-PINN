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
    from . import _forceTorqueThrForceMapping
else:
    import _forceTorqueThrForceMapping

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
    return _forceTorqueThrForceMapping.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _forceTorqueThrForceMapping.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _forceTorqueThrForceMapping.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _forceTorqueThrForceMapping.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _forceTorqueThrForceMapping.new_longArray(nelements)

def delete_longArray(ary):
    return _forceTorqueThrForceMapping.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _forceTorqueThrForceMapping.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _forceTorqueThrForceMapping.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _forceTorqueThrForceMapping.new_intArray(nelements)

def delete_intArray(ary):
    return _forceTorqueThrForceMapping.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _forceTorqueThrForceMapping.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _forceTorqueThrForceMapping.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _forceTorqueThrForceMapping.new_shortArray(nelements)

def delete_shortArray(ary):
    return _forceTorqueThrForceMapping.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _forceTorqueThrForceMapping.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _forceTorqueThrForceMapping.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _forceTorqueThrForceMapping.MAX_LOGGING_LENGTH
BSK_DEBUG = _forceTorqueThrForceMapping.BSK_DEBUG
BSK_INFORMATION = _forceTorqueThrForceMapping.BSK_INFORMATION
BSK_WARNING = _forceTorqueThrForceMapping.BSK_WARNING
BSK_ERROR = _forceTorqueThrForceMapping.BSK_ERROR
BSK_SILENT = _forceTorqueThrForceMapping.BSK_SILENT

def printDefaultLogLevel():
    return _forceTorqueThrForceMapping.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _forceTorqueThrForceMapping.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _forceTorqueThrForceMapping.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _forceTorqueThrForceMapping.BSKLogger_swiginit(self, _forceTorqueThrForceMapping.new_BSKLogger(*args))
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _forceTorqueThrForceMapping.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _forceTorqueThrForceMapping.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _forceTorqueThrForceMapping.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _forceTorqueThrForceMapping.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_forceTorqueThrForceMapping.BSKLogger_logLevelMap_get, _forceTorqueThrForceMapping.BSKLogger_logLevelMap_set)

# Register BSKLogger in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.BSKLogger_swigregister(BSKLogger)
cvar = _forceTorqueThrForceMapping.cvar


def _BSKLogger():
    return _forceTorqueThrForceMapping._BSKLogger()

def _BSKLogger_d(arg1):
    return _forceTorqueThrForceMapping._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _forceTorqueThrForceMapping._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _forceTorqueThrForceMapping._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _forceTorqueThrForceMapping._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _forceTorqueThrForceMapping.SysModel_swiginit(self, _forceTorqueThrForceMapping.new_SysModel(*args))
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_SysModel

    def SelfInit(self):
        return _forceTorqueThrForceMapping.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _forceTorqueThrForceMapping.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _forceTorqueThrForceMapping.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _forceTorqueThrForceMapping.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_forceTorqueThrForceMapping.SysModel_ModelTag_get, _forceTorqueThrForceMapping.SysModel_ModelTag_set)
    CallCounts = property(_forceTorqueThrForceMapping.SysModel_CallCounts_get, _forceTorqueThrForceMapping.SysModel_CallCounts_set)
    RNGSeed = property(_forceTorqueThrForceMapping.SysModel_RNGSeed_get, _forceTorqueThrForceMapping.SysModel_RNGSeed_set)
    moduleID = property(_forceTorqueThrForceMapping.SysModel_moduleID_get, _forceTorqueThrForceMapping.SysModel_moduleID_set)

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


# Register SysModel in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.SysModel_swigregister(SysModel)
class forceTorqueThrForceMappingConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    rThruster_B = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_rThruster_B_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_rThruster_B_set)
    gtThruster_B = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_gtThruster_B_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_gtThruster_B_set)
    numThrusters = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_numThrusters_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_numThrusters_set)
    CoM_B = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_CoM_B_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_CoM_B_set)
    cmdTorqueInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_cmdTorqueInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_cmdTorqueInMsg_set)
    cmdForceInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_cmdForceInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_cmdForceInMsg_set)
    thrConfigInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_thrConfigInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_thrConfigInMsg_set)
    vehConfigInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_vehConfigInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_vehConfigInMsg_set)
    thrForceCmdOutMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_thrForceCmdOutMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_thrForceCmdOutMsg_set)
    bskLogger = property(_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_bskLogger_get, _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_bskLogger_set)

    def createWrapper(self):
        return forceTorqueThrForceMapping(self)


    def __init__(self):
        _forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_swiginit(self, _forceTorqueThrForceMapping.new_forceTorqueThrForceMappingConfig())
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_forceTorqueThrForceMappingConfig

# Register forceTorqueThrForceMappingConfig in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.forceTorqueThrForceMappingConfig_swigregister(forceTorqueThrForceMappingConfig)
class forceTorqueThrForceMapping(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _forceTorqueThrForceMapping.forceTorqueThrForceMapping_swiginit(self, _forceTorqueThrForceMapping.new_forceTorqueThrForceMapping(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _forceTorqueThrForceMapping.forceTorqueThrForceMapping_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _forceTorqueThrForceMapping.forceTorqueThrForceMapping_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _forceTorqueThrForceMapping.forceTorqueThrForceMapping_Reset(self, currentSimNanos)

    def __deref__(self):
        return _forceTorqueThrForceMapping.forceTorqueThrForceMapping___deref__(self)

    def getConfig(self):
        return _forceTorqueThrForceMapping.forceTorqueThrForceMapping_getConfig(self)
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_forceTorqueThrForceMapping
    rThruster_B = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_rThruster_B_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_rThruster_B_set)
    gtThruster_B = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_gtThruster_B_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_gtThruster_B_set)
    numThrusters = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_numThrusters_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_numThrusters_set)
    CoM_B = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_CoM_B_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_CoM_B_set)
    cmdTorqueInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_cmdTorqueInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_cmdTorqueInMsg_set)
    cmdForceInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_cmdForceInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_cmdForceInMsg_set)
    thrConfigInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_thrConfigInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_thrConfigInMsg_set)
    vehConfigInMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_vehConfigInMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_vehConfigInMsg_set)
    thrForceCmdOutMsg = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_thrForceCmdOutMsg_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_thrForceCmdOutMsg_set)
    bskLogger = property(_forceTorqueThrForceMapping.forceTorqueThrForceMapping_bskLogger_get, _forceTorqueThrForceMapping.forceTorqueThrForceMapping_bskLogger_set)

# Register forceTorqueThrForceMapping in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.forceTorqueThrForceMapping_swigregister(forceTorqueThrForceMapping)

from Basilisk.architecture.swig_common_model import *

class CmdTorqueBodyMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    torqueRequestBody = property(_forceTorqueThrForceMapping.CmdTorqueBodyMsgPayload_torqueRequestBody_get, _forceTorqueThrForceMapping.CmdTorqueBodyMsgPayload_torqueRequestBody_set)

    def __init__(self):
        _forceTorqueThrForceMapping.CmdTorqueBodyMsgPayload_swiginit(self, _forceTorqueThrForceMapping.new_CmdTorqueBodyMsgPayload())
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_CmdTorqueBodyMsgPayload

# Register CmdTorqueBodyMsgPayload in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.CmdTorqueBodyMsgPayload_swigregister(CmdTorqueBodyMsgPayload)
class CmdForceBodyMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    forceRequestBody = property(_forceTorqueThrForceMapping.CmdForceBodyMsgPayload_forceRequestBody_get, _forceTorqueThrForceMapping.CmdForceBodyMsgPayload_forceRequestBody_set)

    def __init__(self):
        _forceTorqueThrForceMapping.CmdForceBodyMsgPayload_swiginit(self, _forceTorqueThrForceMapping.new_CmdForceBodyMsgPayload())
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_CmdForceBodyMsgPayload

# Register CmdForceBodyMsgPayload in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.CmdForceBodyMsgPayload_swigregister(CmdForceBodyMsgPayload)
class THRArrayConfigMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    numThrusters = property(_forceTorqueThrForceMapping.THRArrayConfigMsgPayload_numThrusters_get, _forceTorqueThrForceMapping.THRArrayConfigMsgPayload_numThrusters_set)
    thrusters = property(_forceTorqueThrForceMapping.THRArrayConfigMsgPayload_thrusters_get, _forceTorqueThrForceMapping.THRArrayConfigMsgPayload_thrusters_set)

    def __init__(self):
        _forceTorqueThrForceMapping.THRArrayConfigMsgPayload_swiginit(self, _forceTorqueThrForceMapping.new_THRArrayConfigMsgPayload())
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_THRArrayConfigMsgPayload

# Register THRArrayConfigMsgPayload in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.THRArrayConfigMsgPayload_swigregister(THRArrayConfigMsgPayload)
class VehicleConfigMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    ISCPntB_B = property(_forceTorqueThrForceMapping.VehicleConfigMsgPayload_ISCPntB_B_get, _forceTorqueThrForceMapping.VehicleConfigMsgPayload_ISCPntB_B_set)
    CoM_B = property(_forceTorqueThrForceMapping.VehicleConfigMsgPayload_CoM_B_get, _forceTorqueThrForceMapping.VehicleConfigMsgPayload_CoM_B_set)
    massSC = property(_forceTorqueThrForceMapping.VehicleConfigMsgPayload_massSC_get, _forceTorqueThrForceMapping.VehicleConfigMsgPayload_massSC_set)
    CurrentADCSState = property(_forceTorqueThrForceMapping.VehicleConfigMsgPayload_CurrentADCSState_get, _forceTorqueThrForceMapping.VehicleConfigMsgPayload_CurrentADCSState_set)

    def __init__(self):
        _forceTorqueThrForceMapping.VehicleConfigMsgPayload_swiginit(self, _forceTorqueThrForceMapping.new_VehicleConfigMsgPayload())
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_VehicleConfigMsgPayload

# Register VehicleConfigMsgPayload in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.VehicleConfigMsgPayload_swigregister(VehicleConfigMsgPayload)
class THRArrayCmdForceMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    thrForce = property(_forceTorqueThrForceMapping.THRArrayCmdForceMsgPayload_thrForce_get, _forceTorqueThrForceMapping.THRArrayCmdForceMsgPayload_thrForce_set)

    def __init__(self):
        _forceTorqueThrForceMapping.THRArrayCmdForceMsgPayload_swiginit(self, _forceTorqueThrForceMapping.new_THRArrayCmdForceMsgPayload())
    __swig_destroy__ = _forceTorqueThrForceMapping.delete_THRArrayCmdForceMsgPayload

# Register THRArrayCmdForceMsgPayload in _forceTorqueThrForceMapping:
_forceTorqueThrForceMapping.THRArrayCmdForceMsgPayload_swigregister(THRArrayCmdForceMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


