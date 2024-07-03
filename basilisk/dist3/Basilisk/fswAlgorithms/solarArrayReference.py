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
    from . import _solarArrayReference
else:
    import _solarArrayReference

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
    return _solarArrayReference.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _solarArrayReference.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _solarArrayReference.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _solarArrayReference.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _solarArrayReference.new_longArray(nelements)

def delete_longArray(ary):
    return _solarArrayReference.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _solarArrayReference.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _solarArrayReference.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _solarArrayReference.new_intArray(nelements)

def delete_intArray(ary):
    return _solarArrayReference.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _solarArrayReference.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _solarArrayReference.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _solarArrayReference.new_shortArray(nelements)

def delete_shortArray(ary):
    return _solarArrayReference.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _solarArrayReference.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _solarArrayReference.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _solarArrayReference.MAX_LOGGING_LENGTH
BSK_DEBUG = _solarArrayReference.BSK_DEBUG
BSK_INFORMATION = _solarArrayReference.BSK_INFORMATION
BSK_WARNING = _solarArrayReference.BSK_WARNING
BSK_ERROR = _solarArrayReference.BSK_ERROR
BSK_SILENT = _solarArrayReference.BSK_SILENT

def printDefaultLogLevel():
    return _solarArrayReference.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _solarArrayReference.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _solarArrayReference.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _solarArrayReference.BSKLogger_swiginit(self, _solarArrayReference.new_BSKLogger(*args))
    __swig_destroy__ = _solarArrayReference.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _solarArrayReference.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _solarArrayReference.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _solarArrayReference.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _solarArrayReference.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_solarArrayReference.BSKLogger_logLevelMap_get, _solarArrayReference.BSKLogger_logLevelMap_set)

# Register BSKLogger in _solarArrayReference:
_solarArrayReference.BSKLogger_swigregister(BSKLogger)
cvar = _solarArrayReference.cvar


def _BSKLogger():
    return _solarArrayReference._BSKLogger()

def _BSKLogger_d(arg1):
    return _solarArrayReference._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _solarArrayReference._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _solarArrayReference._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _solarArrayReference._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _solarArrayReference.SysModel_swiginit(self, _solarArrayReference.new_SysModel(*args))
    __swig_destroy__ = _solarArrayReference.delete_SysModel

    def SelfInit(self):
        return _solarArrayReference.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _solarArrayReference.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _solarArrayReference.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _solarArrayReference.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_solarArrayReference.SysModel_ModelTag_get, _solarArrayReference.SysModel_ModelTag_set)
    CallCounts = property(_solarArrayReference.SysModel_CallCounts_get, _solarArrayReference.SysModel_CallCounts_set)
    RNGSeed = property(_solarArrayReference.SysModel_RNGSeed_get, _solarArrayReference.SysModel_RNGSeed_set)
    moduleID = property(_solarArrayReference.SysModel_moduleID_get, _solarArrayReference.SysModel_moduleID_set)

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


# Register SysModel in _solarArrayReference:
_solarArrayReference.SysModel_swigregister(SysModel)
referenceFrame = _solarArrayReference.referenceFrame
bodyFrame = _solarArrayReference.bodyFrame
class solarArrayReferenceConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    a1Hat_B = property(_solarArrayReference.solarArrayReferenceConfig_a1Hat_B_get, _solarArrayReference.solarArrayReferenceConfig_a1Hat_B_set)
    a2Hat_B = property(_solarArrayReference.solarArrayReferenceConfig_a2Hat_B_get, _solarArrayReference.solarArrayReferenceConfig_a2Hat_B_set)
    attitudeFrame = property(_solarArrayReference.solarArrayReferenceConfig_attitudeFrame_get, _solarArrayReference.solarArrayReferenceConfig_attitudeFrame_set)
    count = property(_solarArrayReference.solarArrayReferenceConfig_count_get, _solarArrayReference.solarArrayReferenceConfig_count_set)
    priorT = property(_solarArrayReference.solarArrayReferenceConfig_priorT_get, _solarArrayReference.solarArrayReferenceConfig_priorT_set)
    priorThetaR = property(_solarArrayReference.solarArrayReferenceConfig_priorThetaR_get, _solarArrayReference.solarArrayReferenceConfig_priorThetaR_set)
    attNavInMsg = property(_solarArrayReference.solarArrayReferenceConfig_attNavInMsg_get, _solarArrayReference.solarArrayReferenceConfig_attNavInMsg_set)
    attRefInMsg = property(_solarArrayReference.solarArrayReferenceConfig_attRefInMsg_get, _solarArrayReference.solarArrayReferenceConfig_attRefInMsg_set)
    hingedRigidBodyInMsg = property(_solarArrayReference.solarArrayReferenceConfig_hingedRigidBodyInMsg_get, _solarArrayReference.solarArrayReferenceConfig_hingedRigidBodyInMsg_set)
    hingedRigidBodyRefOutMsg = property(_solarArrayReference.solarArrayReferenceConfig_hingedRigidBodyRefOutMsg_get, _solarArrayReference.solarArrayReferenceConfig_hingedRigidBodyRefOutMsg_set)
    bskLogger = property(_solarArrayReference.solarArrayReferenceConfig_bskLogger_get, _solarArrayReference.solarArrayReferenceConfig_bskLogger_set)

    def createWrapper(self):
        return solarArrayReference(self)


    def __init__(self):
        _solarArrayReference.solarArrayReferenceConfig_swiginit(self, _solarArrayReference.new_solarArrayReferenceConfig())
    __swig_destroy__ = _solarArrayReference.delete_solarArrayReferenceConfig

# Register solarArrayReferenceConfig in _solarArrayReference:
_solarArrayReference.solarArrayReferenceConfig_swigregister(solarArrayReferenceConfig)
class solarArrayReference(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _solarArrayReference.solarArrayReference_swiginit(self, _solarArrayReference.new_solarArrayReference(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _solarArrayReference.solarArrayReference_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _solarArrayReference.solarArrayReference_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _solarArrayReference.solarArrayReference_Reset(self, currentSimNanos)

    def __deref__(self):
        return _solarArrayReference.solarArrayReference___deref__(self)

    def getConfig(self):
        return _solarArrayReference.solarArrayReference_getConfig(self)
    __swig_destroy__ = _solarArrayReference.delete_solarArrayReference
    a1Hat_B = property(_solarArrayReference.solarArrayReference_a1Hat_B_get, _solarArrayReference.solarArrayReference_a1Hat_B_set)
    a2Hat_B = property(_solarArrayReference.solarArrayReference_a2Hat_B_get, _solarArrayReference.solarArrayReference_a2Hat_B_set)
    attitudeFrame = property(_solarArrayReference.solarArrayReference_attitudeFrame_get, _solarArrayReference.solarArrayReference_attitudeFrame_set)
    count = property(_solarArrayReference.solarArrayReference_count_get, _solarArrayReference.solarArrayReference_count_set)
    priorT = property(_solarArrayReference.solarArrayReference_priorT_get, _solarArrayReference.solarArrayReference_priorT_set)
    priorThetaR = property(_solarArrayReference.solarArrayReference_priorThetaR_get, _solarArrayReference.solarArrayReference_priorThetaR_set)
    attNavInMsg = property(_solarArrayReference.solarArrayReference_attNavInMsg_get, _solarArrayReference.solarArrayReference_attNavInMsg_set)
    attRefInMsg = property(_solarArrayReference.solarArrayReference_attRefInMsg_get, _solarArrayReference.solarArrayReference_attRefInMsg_set)
    hingedRigidBodyInMsg = property(_solarArrayReference.solarArrayReference_hingedRigidBodyInMsg_get, _solarArrayReference.solarArrayReference_hingedRigidBodyInMsg_set)
    hingedRigidBodyRefOutMsg = property(_solarArrayReference.solarArrayReference_hingedRigidBodyRefOutMsg_get, _solarArrayReference.solarArrayReference_hingedRigidBodyRefOutMsg_set)
    bskLogger = property(_solarArrayReference.solarArrayReference_bskLogger_get, _solarArrayReference.solarArrayReference_bskLogger_set)

# Register solarArrayReference in _solarArrayReference:
_solarArrayReference.solarArrayReference_swigregister(solarArrayReference)
class NavAttMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    timeTag = property(_solarArrayReference.NavAttMsgPayload_timeTag_get, _solarArrayReference.NavAttMsgPayload_timeTag_set)
    sigma_BN = property(_solarArrayReference.NavAttMsgPayload_sigma_BN_get, _solarArrayReference.NavAttMsgPayload_sigma_BN_set)
    omega_BN_B = property(_solarArrayReference.NavAttMsgPayload_omega_BN_B_get, _solarArrayReference.NavAttMsgPayload_omega_BN_B_set)
    vehSunPntBdy = property(_solarArrayReference.NavAttMsgPayload_vehSunPntBdy_get, _solarArrayReference.NavAttMsgPayload_vehSunPntBdy_set)

    def __init__(self):
        _solarArrayReference.NavAttMsgPayload_swiginit(self, _solarArrayReference.new_NavAttMsgPayload())
    __swig_destroy__ = _solarArrayReference.delete_NavAttMsgPayload

# Register NavAttMsgPayload in _solarArrayReference:
_solarArrayReference.NavAttMsgPayload_swigregister(NavAttMsgPayload)
class AttRefMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    sigma_RN = property(_solarArrayReference.AttRefMsgPayload_sigma_RN_get, _solarArrayReference.AttRefMsgPayload_sigma_RN_set)
    omega_RN_N = property(_solarArrayReference.AttRefMsgPayload_omega_RN_N_get, _solarArrayReference.AttRefMsgPayload_omega_RN_N_set)
    domega_RN_N = property(_solarArrayReference.AttRefMsgPayload_domega_RN_N_get, _solarArrayReference.AttRefMsgPayload_domega_RN_N_set)

    def __init__(self):
        _solarArrayReference.AttRefMsgPayload_swiginit(self, _solarArrayReference.new_AttRefMsgPayload())
    __swig_destroy__ = _solarArrayReference.delete_AttRefMsgPayload

# Register AttRefMsgPayload in _solarArrayReference:
_solarArrayReference.AttRefMsgPayload_swigregister(AttRefMsgPayload)
class HingedRigidBodyMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    theta = property(_solarArrayReference.HingedRigidBodyMsgPayload_theta_get, _solarArrayReference.HingedRigidBodyMsgPayload_theta_set)
    thetaDot = property(_solarArrayReference.HingedRigidBodyMsgPayload_thetaDot_get, _solarArrayReference.HingedRigidBodyMsgPayload_thetaDot_set)

    def __init__(self):
        _solarArrayReference.HingedRigidBodyMsgPayload_swiginit(self, _solarArrayReference.new_HingedRigidBodyMsgPayload())
    __swig_destroy__ = _solarArrayReference.delete_HingedRigidBodyMsgPayload

# Register HingedRigidBodyMsgPayload in _solarArrayReference:
_solarArrayReference.HingedRigidBodyMsgPayload_swigregister(HingedRigidBodyMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


