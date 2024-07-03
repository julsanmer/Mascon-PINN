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
    from . import _magneticFieldCenteredDipole
else:
    import _magneticFieldCenteredDipole

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
    return _magneticFieldCenteredDipole.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _magneticFieldCenteredDipole.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _magneticFieldCenteredDipole.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _magneticFieldCenteredDipole.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _magneticFieldCenteredDipole.new_longArray(nelements)

def delete_longArray(ary):
    return _magneticFieldCenteredDipole.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _magneticFieldCenteredDipole.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _magneticFieldCenteredDipole.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _magneticFieldCenteredDipole.new_intArray(nelements)

def delete_intArray(ary):
    return _magneticFieldCenteredDipole.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _magneticFieldCenteredDipole.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _magneticFieldCenteredDipole.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _magneticFieldCenteredDipole.new_shortArray(nelements)

def delete_shortArray(ary):
    return _magneticFieldCenteredDipole.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _magneticFieldCenteredDipole.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _magneticFieldCenteredDipole.shortArray_setitem(ary, index, value)


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
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_SwigPyIterator

    def value(self):
        return _magneticFieldCenteredDipole.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _magneticFieldCenteredDipole.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _magneticFieldCenteredDipole.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _magneticFieldCenteredDipole.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _magneticFieldCenteredDipole.SwigPyIterator_equal(self, x)

    def copy(self):
        return _magneticFieldCenteredDipole.SwigPyIterator_copy(self)

    def next(self):
        return _magneticFieldCenteredDipole.SwigPyIterator_next(self)

    def __next__(self):
        return _magneticFieldCenteredDipole.SwigPyIterator___next__(self)

    def previous(self):
        return _magneticFieldCenteredDipole.SwigPyIterator_previous(self)

    def advance(self, n):
        return _magneticFieldCenteredDipole.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _magneticFieldCenteredDipole.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _magneticFieldCenteredDipole.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _magneticFieldCenteredDipole.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _magneticFieldCenteredDipole.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _magneticFieldCenteredDipole.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _magneticFieldCenteredDipole.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.SwigPyIterator_swigregister(SwigPyIterator)

from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _magneticFieldCenteredDipole.MAX_LOGGING_LENGTH
BSK_DEBUG = _magneticFieldCenteredDipole.BSK_DEBUG
BSK_INFORMATION = _magneticFieldCenteredDipole.BSK_INFORMATION
BSK_WARNING = _magneticFieldCenteredDipole.BSK_WARNING
BSK_ERROR = _magneticFieldCenteredDipole.BSK_ERROR
BSK_SILENT = _magneticFieldCenteredDipole.BSK_SILENT

def printDefaultLogLevel():
    return _magneticFieldCenteredDipole.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _magneticFieldCenteredDipole.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _magneticFieldCenteredDipole.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _magneticFieldCenteredDipole.BSKLogger_swiginit(self, _magneticFieldCenteredDipole.new_BSKLogger(*args))
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _magneticFieldCenteredDipole.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _magneticFieldCenteredDipole.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _magneticFieldCenteredDipole.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _magneticFieldCenteredDipole.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_magneticFieldCenteredDipole.BSKLogger_logLevelMap_get, _magneticFieldCenteredDipole.BSKLogger_logLevelMap_set)

# Register BSKLogger in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.BSKLogger_swigregister(BSKLogger)
cvar = _magneticFieldCenteredDipole.cvar


def _BSKLogger():
    return _magneticFieldCenteredDipole._BSKLogger()

def _BSKLogger_d(arg1):
    return _magneticFieldCenteredDipole._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _magneticFieldCenteredDipole._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _magneticFieldCenteredDipole._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _magneticFieldCenteredDipole._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _magneticFieldCenteredDipole.SysModel_swiginit(self, _magneticFieldCenteredDipole.new_SysModel(*args))
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_SysModel

    def SelfInit(self):
        return _magneticFieldCenteredDipole.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _magneticFieldCenteredDipole.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _magneticFieldCenteredDipole.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _magneticFieldCenteredDipole.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_magneticFieldCenteredDipole.SysModel_ModelTag_get, _magneticFieldCenteredDipole.SysModel_ModelTag_set)
    CallCounts = property(_magneticFieldCenteredDipole.SysModel_CallCounts_get, _magneticFieldCenteredDipole.SysModel_CallCounts_set)
    RNGSeed = property(_magneticFieldCenteredDipole.SysModel_RNGSeed_get, _magneticFieldCenteredDipole.SysModel_RNGSeed_set)
    moduleID = property(_magneticFieldCenteredDipole.SysModel_moduleID_get, _magneticFieldCenteredDipole.SysModel_moduleID_set)

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


# Register SysModel in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.SysModel_swigregister(SysModel)
class MagneticFieldBase(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_MagneticFieldBase

    def Reset(self, CurrentSimNanos):
        return _magneticFieldCenteredDipole.MagneticFieldBase_Reset(self, CurrentSimNanos)

    def addSpacecraftToModel(self, tmpScMsg):
        return _magneticFieldCenteredDipole.MagneticFieldBase_addSpacecraftToModel(self, tmpScMsg)

    def UpdateState(self, CurrentSimNanos):
        return _magneticFieldCenteredDipole.MagneticFieldBase_UpdateState(self, CurrentSimNanos)
    scStateInMsgs = property(_magneticFieldCenteredDipole.MagneticFieldBase_scStateInMsgs_get, _magneticFieldCenteredDipole.MagneticFieldBase_scStateInMsgs_set)
    envOutMsgs = property(_magneticFieldCenteredDipole.MagneticFieldBase_envOutMsgs_get, _magneticFieldCenteredDipole.MagneticFieldBase_envOutMsgs_set)
    planetPosInMsg = property(_magneticFieldCenteredDipole.MagneticFieldBase_planetPosInMsg_get, _magneticFieldCenteredDipole.MagneticFieldBase_planetPosInMsg_set)
    epochInMsg = property(_magneticFieldCenteredDipole.MagneticFieldBase_epochInMsg_get, _magneticFieldCenteredDipole.MagneticFieldBase_epochInMsg_set)
    envMinReach = property(_magneticFieldCenteredDipole.MagneticFieldBase_envMinReach_get, _magneticFieldCenteredDipole.MagneticFieldBase_envMinReach_set)
    envMaxReach = property(_magneticFieldCenteredDipole.MagneticFieldBase_envMaxReach_get, _magneticFieldCenteredDipole.MagneticFieldBase_envMaxReach_set)
    planetRadius = property(_magneticFieldCenteredDipole.MagneticFieldBase_planetRadius_get, _magneticFieldCenteredDipole.MagneticFieldBase_planetRadius_set)
    bskLogger = property(_magneticFieldCenteredDipole.MagneticFieldBase_bskLogger_get, _magneticFieldCenteredDipole.MagneticFieldBase_bskLogger_set)

# Register MagneticFieldBase in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.MagneticFieldBase_swigregister(MagneticFieldBase)
class MagneticFieldCenteredDipole(MagneticFieldBase):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _magneticFieldCenteredDipole.MagneticFieldCenteredDipole_swiginit(self, _magneticFieldCenteredDipole.new_MagneticFieldCenteredDipole())
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_MagneticFieldCenteredDipole
    g10 = property(_magneticFieldCenteredDipole.MagneticFieldCenteredDipole_g10_get, _magneticFieldCenteredDipole.MagneticFieldCenteredDipole_g10_set)
    g11 = property(_magneticFieldCenteredDipole.MagneticFieldCenteredDipole_g11_get, _magneticFieldCenteredDipole.MagneticFieldCenteredDipole_g11_set)
    h11 = property(_magneticFieldCenteredDipole.MagneticFieldCenteredDipole_h11_get, _magneticFieldCenteredDipole.MagneticFieldCenteredDipole_h11_set)
    bskLogger = property(_magneticFieldCenteredDipole.MagneticFieldCenteredDipole_bskLogger_get, _magneticFieldCenteredDipole.MagneticFieldCenteredDipole_bskLogger_set)

# Register MagneticFieldCenteredDipole in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.MagneticFieldCenteredDipole_swigregister(MagneticFieldCenteredDipole)
MAX_BODY_NAME_LENGTH = _magneticFieldCenteredDipole.MAX_BODY_NAME_LENGTH
class SpicePlanetStateMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    J2000Current = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_J2000Current_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_J2000Current_set)
    PositionVector = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_PositionVector_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_PositionVector_set)
    VelocityVector = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_VelocityVector_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_VelocityVector_set)
    J20002Pfix = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_J20002Pfix_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_J20002Pfix_set)
    J20002Pfix_dot = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_J20002Pfix_dot_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_J20002Pfix_dot_set)
    computeOrient = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_computeOrient_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_computeOrient_set)
    PlanetName = property(_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_PlanetName_get, _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_PlanetName_set)

    def __init__(self):
        _magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_swiginit(self, _magneticFieldCenteredDipole.new_SpicePlanetStateMsgPayload())
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_SpicePlanetStateMsgPayload

# Register SpicePlanetStateMsgPayload in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.SpicePlanetStateMsgPayload_swigregister(SpicePlanetStateMsgPayload)
class SCStatesMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    r_BN_N = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_r_BN_N_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_r_BN_N_set)
    v_BN_N = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_v_BN_N_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_v_BN_N_set)
    r_CN_N = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_r_CN_N_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_r_CN_N_set)
    v_CN_N = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_v_CN_N_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_v_CN_N_set)
    sigma_BN = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_sigma_BN_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_sigma_BN_set)
    omega_BN_B = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_omega_BN_B_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_omega_BN_B_set)
    omegaDot_BN_B = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_omegaDot_BN_B_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_omegaDot_BN_B_set)
    TotalAccumDVBdy = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_TotalAccumDVBdy_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_TotalAccumDVBdy_set)
    TotalAccumDV_BN_B = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_TotalAccumDV_BN_B_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_TotalAccumDV_BN_B_set)
    nonConservativeAccelpntB_B = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_nonConservativeAccelpntB_B_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_nonConservativeAccelpntB_B_set)
    MRPSwitchCount = property(_magneticFieldCenteredDipole.SCStatesMsgPayload_MRPSwitchCount_get, _magneticFieldCenteredDipole.SCStatesMsgPayload_MRPSwitchCount_set)

    def __init__(self):
        _magneticFieldCenteredDipole.SCStatesMsgPayload_swiginit(self, _magneticFieldCenteredDipole.new_SCStatesMsgPayload())
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_SCStatesMsgPayload

# Register SCStatesMsgPayload in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.SCStatesMsgPayload_swigregister(SCStatesMsgPayload)
class MagneticFieldMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    magField_N = property(_magneticFieldCenteredDipole.MagneticFieldMsgPayload_magField_N_get, _magneticFieldCenteredDipole.MagneticFieldMsgPayload_magField_N_set)

    def __init__(self):
        _magneticFieldCenteredDipole.MagneticFieldMsgPayload_swiginit(self, _magneticFieldCenteredDipole.new_MagneticFieldMsgPayload())
    __swig_destroy__ = _magneticFieldCenteredDipole.delete_MagneticFieldMsgPayload

# Register MagneticFieldMsgPayload in _magneticFieldCenteredDipole:
_magneticFieldCenteredDipole.MagneticFieldMsgPayload_swigregister(MagneticFieldMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


