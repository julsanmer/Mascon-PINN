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
    from . import _hingedRigidBodyPIDMotor
else:
    import _hingedRigidBodyPIDMotor

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
    return _hingedRigidBodyPIDMotor.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _hingedRigidBodyPIDMotor.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _hingedRigidBodyPIDMotor.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _hingedRigidBodyPIDMotor.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _hingedRigidBodyPIDMotor.new_longArray(nelements)

def delete_longArray(ary):
    return _hingedRigidBodyPIDMotor.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _hingedRigidBodyPIDMotor.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _hingedRigidBodyPIDMotor.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _hingedRigidBodyPIDMotor.new_intArray(nelements)

def delete_intArray(ary):
    return _hingedRigidBodyPIDMotor.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _hingedRigidBodyPIDMotor.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _hingedRigidBodyPIDMotor.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _hingedRigidBodyPIDMotor.new_shortArray(nelements)

def delete_shortArray(ary):
    return _hingedRigidBodyPIDMotor.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _hingedRigidBodyPIDMotor.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _hingedRigidBodyPIDMotor.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _hingedRigidBodyPIDMotor.MAX_LOGGING_LENGTH
BSK_DEBUG = _hingedRigidBodyPIDMotor.BSK_DEBUG
BSK_INFORMATION = _hingedRigidBodyPIDMotor.BSK_INFORMATION
BSK_WARNING = _hingedRigidBodyPIDMotor.BSK_WARNING
BSK_ERROR = _hingedRigidBodyPIDMotor.BSK_ERROR
BSK_SILENT = _hingedRigidBodyPIDMotor.BSK_SILENT

def printDefaultLogLevel():
    return _hingedRigidBodyPIDMotor.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _hingedRigidBodyPIDMotor.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _hingedRigidBodyPIDMotor.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _hingedRigidBodyPIDMotor.BSKLogger_swiginit(self, _hingedRigidBodyPIDMotor.new_BSKLogger(*args))
    __swig_destroy__ = _hingedRigidBodyPIDMotor.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _hingedRigidBodyPIDMotor.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _hingedRigidBodyPIDMotor.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _hingedRigidBodyPIDMotor.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _hingedRigidBodyPIDMotor.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_hingedRigidBodyPIDMotor.BSKLogger_logLevelMap_get, _hingedRigidBodyPIDMotor.BSKLogger_logLevelMap_set)

# Register BSKLogger in _hingedRigidBodyPIDMotor:
_hingedRigidBodyPIDMotor.BSKLogger_swigregister(BSKLogger)
cvar = _hingedRigidBodyPIDMotor.cvar


def _BSKLogger():
    return _hingedRigidBodyPIDMotor._BSKLogger()

def _BSKLogger_d(arg1):
    return _hingedRigidBodyPIDMotor._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _hingedRigidBodyPIDMotor._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _hingedRigidBodyPIDMotor._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _hingedRigidBodyPIDMotor._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _hingedRigidBodyPIDMotor.SysModel_swiginit(self, _hingedRigidBodyPIDMotor.new_SysModel(*args))
    __swig_destroy__ = _hingedRigidBodyPIDMotor.delete_SysModel

    def SelfInit(self):
        return _hingedRigidBodyPIDMotor.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _hingedRigidBodyPIDMotor.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _hingedRigidBodyPIDMotor.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _hingedRigidBodyPIDMotor.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_hingedRigidBodyPIDMotor.SysModel_ModelTag_get, _hingedRigidBodyPIDMotor.SysModel_ModelTag_set)
    CallCounts = property(_hingedRigidBodyPIDMotor.SysModel_CallCounts_get, _hingedRigidBodyPIDMotor.SysModel_CallCounts_set)
    RNGSeed = property(_hingedRigidBodyPIDMotor.SysModel_RNGSeed_get, _hingedRigidBodyPIDMotor.SysModel_RNGSeed_set)
    moduleID = property(_hingedRigidBodyPIDMotor.SysModel_moduleID_get, _hingedRigidBodyPIDMotor.SysModel_moduleID_set)

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


# Register SysModel in _hingedRigidBodyPIDMotor:
_hingedRigidBodyPIDMotor.SysModel_swigregister(SysModel)
class hingedRigidBodyPIDMotorConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    K = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_K_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_K_set)
    P = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_P_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_P_set)
    I = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_I_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_I_set)
    priorTime = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_priorTime_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_priorTime_set)
    priorThetaError = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_priorThetaError_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_priorThetaError_set)
    intError = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_intError_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_intError_set)
    hingedRigidBodyInMsg = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_hingedRigidBodyInMsg_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_hingedRigidBodyInMsg_set)
    hingedRigidBodyRefInMsg = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_hingedRigidBodyRefInMsg_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_hingedRigidBodyRefInMsg_set)
    motorTorqueOutMsg = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_motorTorqueOutMsg_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_motorTorqueOutMsg_set)
    bskLogger = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_bskLogger_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_bskLogger_set)

    def createWrapper(self):
        return hingedRigidBodyPIDMotor(self)


    def __init__(self):
        _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_swiginit(self, _hingedRigidBodyPIDMotor.new_hingedRigidBodyPIDMotorConfig())
    __swig_destroy__ = _hingedRigidBodyPIDMotor.delete_hingedRigidBodyPIDMotorConfig

# Register hingedRigidBodyPIDMotorConfig in _hingedRigidBodyPIDMotor:
_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotorConfig_swigregister(hingedRigidBodyPIDMotorConfig)
class hingedRigidBodyPIDMotor(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_swiginit(self, _hingedRigidBodyPIDMotor.new_hingedRigidBodyPIDMotor(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_Reset(self, currentSimNanos)

    def __deref__(self):
        return _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor___deref__(self)

    def getConfig(self):
        return _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_getConfig(self)
    __swig_destroy__ = _hingedRigidBodyPIDMotor.delete_hingedRigidBodyPIDMotor
    K = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_K_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_K_set)
    P = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_P_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_P_set)
    I = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_I_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_I_set)
    priorTime = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_priorTime_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_priorTime_set)
    priorThetaError = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_priorThetaError_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_priorThetaError_set)
    intError = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_intError_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_intError_set)
    hingedRigidBodyInMsg = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_hingedRigidBodyInMsg_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_hingedRigidBodyInMsg_set)
    hingedRigidBodyRefInMsg = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_hingedRigidBodyRefInMsg_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_hingedRigidBodyRefInMsg_set)
    motorTorqueOutMsg = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_motorTorqueOutMsg_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_motorTorqueOutMsg_set)
    bskLogger = property(_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_bskLogger_get, _hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_bskLogger_set)

# Register hingedRigidBodyPIDMotor in _hingedRigidBodyPIDMotor:
_hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor_swigregister(hingedRigidBodyPIDMotor)
class HingedRigidBodyMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    theta = property(_hingedRigidBodyPIDMotor.HingedRigidBodyMsgPayload_theta_get, _hingedRigidBodyPIDMotor.HingedRigidBodyMsgPayload_theta_set)
    thetaDot = property(_hingedRigidBodyPIDMotor.HingedRigidBodyMsgPayload_thetaDot_get, _hingedRigidBodyPIDMotor.HingedRigidBodyMsgPayload_thetaDot_set)

    def __init__(self):
        _hingedRigidBodyPIDMotor.HingedRigidBodyMsgPayload_swiginit(self, _hingedRigidBodyPIDMotor.new_HingedRigidBodyMsgPayload())
    __swig_destroy__ = _hingedRigidBodyPIDMotor.delete_HingedRigidBodyMsgPayload

# Register HingedRigidBodyMsgPayload in _hingedRigidBodyPIDMotor:
_hingedRigidBodyPIDMotor.HingedRigidBodyMsgPayload_swigregister(HingedRigidBodyMsgPayload)
class ArrayMotorTorqueMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    motorTorque = property(_hingedRigidBodyPIDMotor.ArrayMotorTorqueMsgPayload_motorTorque_get, _hingedRigidBodyPIDMotor.ArrayMotorTorqueMsgPayload_motorTorque_set)

    def __init__(self):
        _hingedRigidBodyPIDMotor.ArrayMotorTorqueMsgPayload_swiginit(self, _hingedRigidBodyPIDMotor.new_ArrayMotorTorqueMsgPayload())
    __swig_destroy__ = _hingedRigidBodyPIDMotor.delete_ArrayMotorTorqueMsgPayload

# Register ArrayMotorTorqueMsgPayload in _hingedRigidBodyPIDMotor:
_hingedRigidBodyPIDMotor.ArrayMotorTorqueMsgPayload_swigregister(ArrayMotorTorqueMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


