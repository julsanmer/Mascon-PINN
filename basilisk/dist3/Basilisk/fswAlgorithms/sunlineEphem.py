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
    from . import _sunlineEphem
else:
    import _sunlineEphem

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
    return _sunlineEphem.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _sunlineEphem.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _sunlineEphem.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _sunlineEphem.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _sunlineEphem.new_longArray(nelements)

def delete_longArray(ary):
    return _sunlineEphem.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _sunlineEphem.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _sunlineEphem.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _sunlineEphem.new_intArray(nelements)

def delete_intArray(ary):
    return _sunlineEphem.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _sunlineEphem.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _sunlineEphem.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _sunlineEphem.new_shortArray(nelements)

def delete_shortArray(ary):
    return _sunlineEphem.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _sunlineEphem.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _sunlineEphem.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _sunlineEphem.MAX_LOGGING_LENGTH
BSK_DEBUG = _sunlineEphem.BSK_DEBUG
BSK_INFORMATION = _sunlineEphem.BSK_INFORMATION
BSK_WARNING = _sunlineEphem.BSK_WARNING
BSK_ERROR = _sunlineEphem.BSK_ERROR
BSK_SILENT = _sunlineEphem.BSK_SILENT

def printDefaultLogLevel():
    return _sunlineEphem.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _sunlineEphem.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _sunlineEphem.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _sunlineEphem.BSKLogger_swiginit(self, _sunlineEphem.new_BSKLogger(*args))
    __swig_destroy__ = _sunlineEphem.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _sunlineEphem.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _sunlineEphem.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _sunlineEphem.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _sunlineEphem.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_sunlineEphem.BSKLogger_logLevelMap_get, _sunlineEphem.BSKLogger_logLevelMap_set)

# Register BSKLogger in _sunlineEphem:
_sunlineEphem.BSKLogger_swigregister(BSKLogger)
cvar = _sunlineEphem.cvar


def _BSKLogger():
    return _sunlineEphem._BSKLogger()

def _BSKLogger_d(arg1):
    return _sunlineEphem._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _sunlineEphem._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _sunlineEphem._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _sunlineEphem._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _sunlineEphem.SysModel_swiginit(self, _sunlineEphem.new_SysModel(*args))
    __swig_destroy__ = _sunlineEphem.delete_SysModel

    def SelfInit(self):
        return _sunlineEphem.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _sunlineEphem.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _sunlineEphem.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _sunlineEphem.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_sunlineEphem.SysModel_ModelTag_get, _sunlineEphem.SysModel_ModelTag_set)
    CallCounts = property(_sunlineEphem.SysModel_CallCounts_get, _sunlineEphem.SysModel_CallCounts_set)
    RNGSeed = property(_sunlineEphem.SysModel_RNGSeed_get, _sunlineEphem.SysModel_RNGSeed_set)
    moduleID = property(_sunlineEphem.SysModel_moduleID_get, _sunlineEphem.SysModel_moduleID_set)

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


# Register SysModel in _sunlineEphem:
_sunlineEphem.SysModel_swigregister(SysModel)
class sunlineEphemConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    navStateOutMsg = property(_sunlineEphem.sunlineEphemConfig_navStateOutMsg_get, _sunlineEphem.sunlineEphemConfig_navStateOutMsg_set)
    sunPositionInMsg = property(_sunlineEphem.sunlineEphemConfig_sunPositionInMsg_get, _sunlineEphem.sunlineEphemConfig_sunPositionInMsg_set)
    scPositionInMsg = property(_sunlineEphem.sunlineEphemConfig_scPositionInMsg_get, _sunlineEphem.sunlineEphemConfig_scPositionInMsg_set)
    scAttitudeInMsg = property(_sunlineEphem.sunlineEphemConfig_scAttitudeInMsg_get, _sunlineEphem.sunlineEphemConfig_scAttitudeInMsg_set)
    bskLogger = property(_sunlineEphem.sunlineEphemConfig_bskLogger_get, _sunlineEphem.sunlineEphemConfig_bskLogger_set)

    def createWrapper(self):
        return sunlineEphem(self)


    def __init__(self):
        _sunlineEphem.sunlineEphemConfig_swiginit(self, _sunlineEphem.new_sunlineEphemConfig())
    __swig_destroy__ = _sunlineEphem.delete_sunlineEphemConfig

# Register sunlineEphemConfig in _sunlineEphem:
_sunlineEphem.sunlineEphemConfig_swigregister(sunlineEphemConfig)
class sunlineEphem(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _sunlineEphem.sunlineEphem_swiginit(self, _sunlineEphem.new_sunlineEphem(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _sunlineEphem.sunlineEphem_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _sunlineEphem.sunlineEphem_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _sunlineEphem.sunlineEphem_Reset(self, currentSimNanos)

    def __deref__(self):
        return _sunlineEphem.sunlineEphem___deref__(self)

    def getConfig(self):
        return _sunlineEphem.sunlineEphem_getConfig(self)
    __swig_destroy__ = _sunlineEphem.delete_sunlineEphem
    navStateOutMsg = property(_sunlineEphem.sunlineEphem_navStateOutMsg_get, _sunlineEphem.sunlineEphem_navStateOutMsg_set)
    sunPositionInMsg = property(_sunlineEphem.sunlineEphem_sunPositionInMsg_get, _sunlineEphem.sunlineEphem_sunPositionInMsg_set)
    scPositionInMsg = property(_sunlineEphem.sunlineEphem_scPositionInMsg_get, _sunlineEphem.sunlineEphem_scPositionInMsg_set)
    scAttitudeInMsg = property(_sunlineEphem.sunlineEphem_scAttitudeInMsg_get, _sunlineEphem.sunlineEphem_scAttitudeInMsg_set)
    bskLogger = property(_sunlineEphem.sunlineEphem_bskLogger_get, _sunlineEphem.sunlineEphem_bskLogger_set)

# Register sunlineEphem in _sunlineEphem:
_sunlineEphem.sunlineEphem_swigregister(sunlineEphem)
class NavAttMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    timeTag = property(_sunlineEphem.NavAttMsgPayload_timeTag_get, _sunlineEphem.NavAttMsgPayload_timeTag_set)
    sigma_BN = property(_sunlineEphem.NavAttMsgPayload_sigma_BN_get, _sunlineEphem.NavAttMsgPayload_sigma_BN_set)
    omega_BN_B = property(_sunlineEphem.NavAttMsgPayload_omega_BN_B_get, _sunlineEphem.NavAttMsgPayload_omega_BN_B_set)
    vehSunPntBdy = property(_sunlineEphem.NavAttMsgPayload_vehSunPntBdy_get, _sunlineEphem.NavAttMsgPayload_vehSunPntBdy_set)

    def __init__(self):
        _sunlineEphem.NavAttMsgPayload_swiginit(self, _sunlineEphem.new_NavAttMsgPayload())
    __swig_destroy__ = _sunlineEphem.delete_NavAttMsgPayload

# Register NavAttMsgPayload in _sunlineEphem:
_sunlineEphem.NavAttMsgPayload_swigregister(NavAttMsgPayload)
class NavTransMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    timeTag = property(_sunlineEphem.NavTransMsgPayload_timeTag_get, _sunlineEphem.NavTransMsgPayload_timeTag_set)
    r_BN_N = property(_sunlineEphem.NavTransMsgPayload_r_BN_N_get, _sunlineEphem.NavTransMsgPayload_r_BN_N_set)
    v_BN_N = property(_sunlineEphem.NavTransMsgPayload_v_BN_N_get, _sunlineEphem.NavTransMsgPayload_v_BN_N_set)
    vehAccumDV = property(_sunlineEphem.NavTransMsgPayload_vehAccumDV_get, _sunlineEphem.NavTransMsgPayload_vehAccumDV_set)

    def __init__(self):
        _sunlineEphem.NavTransMsgPayload_swiginit(self, _sunlineEphem.new_NavTransMsgPayload())
    __swig_destroy__ = _sunlineEphem.delete_NavTransMsgPayload

# Register NavTransMsgPayload in _sunlineEphem:
_sunlineEphem.NavTransMsgPayload_swigregister(NavTransMsgPayload)
class EphemerisMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    r_BdyZero_N = property(_sunlineEphem.EphemerisMsgPayload_r_BdyZero_N_get, _sunlineEphem.EphemerisMsgPayload_r_BdyZero_N_set)
    v_BdyZero_N = property(_sunlineEphem.EphemerisMsgPayload_v_BdyZero_N_get, _sunlineEphem.EphemerisMsgPayload_v_BdyZero_N_set)
    sigma_BN = property(_sunlineEphem.EphemerisMsgPayload_sigma_BN_get, _sunlineEphem.EphemerisMsgPayload_sigma_BN_set)
    omega_BN_B = property(_sunlineEphem.EphemerisMsgPayload_omega_BN_B_get, _sunlineEphem.EphemerisMsgPayload_omega_BN_B_set)
    timeTag = property(_sunlineEphem.EphemerisMsgPayload_timeTag_get, _sunlineEphem.EphemerisMsgPayload_timeTag_set)

    def __init__(self):
        _sunlineEphem.EphemerisMsgPayload_swiginit(self, _sunlineEphem.new_EphemerisMsgPayload())
    __swig_destroy__ = _sunlineEphem.delete_EphemerisMsgPayload

# Register EphemerisMsgPayload in _sunlineEphem:
_sunlineEphem.EphemerisMsgPayload_swigregister(EphemerisMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


