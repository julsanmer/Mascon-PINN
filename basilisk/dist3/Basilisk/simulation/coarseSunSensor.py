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
    from . import _coarseSunSensor
else:
    import _coarseSunSensor

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



def new_doubleArray(nelements):
    return _coarseSunSensor.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _coarseSunSensor.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _coarseSunSensor.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _coarseSunSensor.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _coarseSunSensor.new_longArray(nelements)

def delete_longArray(ary):
    return _coarseSunSensor.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _coarseSunSensor.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _coarseSunSensor.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _coarseSunSensor.new_intArray(nelements)

def delete_intArray(ary):
    return _coarseSunSensor.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _coarseSunSensor.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _coarseSunSensor.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _coarseSunSensor.new_shortArray(nelements)

def delete_shortArray(ary):
    return _coarseSunSensor.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _coarseSunSensor.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _coarseSunSensor.shortArray_setitem(ary, index, value)


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
    __swig_destroy__ = _coarseSunSensor.delete_SwigPyIterator

    def value(self):
        return _coarseSunSensor.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _coarseSunSensor.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _coarseSunSensor.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _coarseSunSensor.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _coarseSunSensor.SwigPyIterator_equal(self, x)

    def copy(self):
        return _coarseSunSensor.SwigPyIterator_copy(self)

    def next(self):
        return _coarseSunSensor.SwigPyIterator_next(self)

    def __next__(self):
        return _coarseSunSensor.SwigPyIterator___next__(self)

    def previous(self):
        return _coarseSunSensor.SwigPyIterator_previous(self)

    def advance(self, n):
        return _coarseSunSensor.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _coarseSunSensor.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _coarseSunSensor.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _coarseSunSensor.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _coarseSunSensor.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _coarseSunSensor.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _coarseSunSensor.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _coarseSunSensor:
_coarseSunSensor.SwigPyIterator_swigregister(SwigPyIterator)

from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _coarseSunSensor.MAX_LOGGING_LENGTH
BSK_DEBUG = _coarseSunSensor.BSK_DEBUG
BSK_INFORMATION = _coarseSunSensor.BSK_INFORMATION
BSK_WARNING = _coarseSunSensor.BSK_WARNING
BSK_ERROR = _coarseSunSensor.BSK_ERROR
BSK_SILENT = _coarseSunSensor.BSK_SILENT

def printDefaultLogLevel():
    return _coarseSunSensor.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _coarseSunSensor.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _coarseSunSensor.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _coarseSunSensor.BSKLogger_swiginit(self, _coarseSunSensor.new_BSKLogger(*args))
    __swig_destroy__ = _coarseSunSensor.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _coarseSunSensor.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _coarseSunSensor.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _coarseSunSensor.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _coarseSunSensor.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_coarseSunSensor.BSKLogger_logLevelMap_get, _coarseSunSensor.BSKLogger_logLevelMap_set)

# Register BSKLogger in _coarseSunSensor:
_coarseSunSensor.BSKLogger_swigregister(BSKLogger)
cvar = _coarseSunSensor.cvar


def _BSKLogger():
    return _coarseSunSensor._BSKLogger()

def _BSKLogger_d(arg1):
    return _coarseSunSensor._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _coarseSunSensor._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _coarseSunSensor._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _coarseSunSensor._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _coarseSunSensor.SysModel_swiginit(self, _coarseSunSensor.new_SysModel(*args))
    __swig_destroy__ = _coarseSunSensor.delete_SysModel

    def SelfInit(self):
        return _coarseSunSensor.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _coarseSunSensor.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _coarseSunSensor.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _coarseSunSensor.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_coarseSunSensor.SysModel_ModelTag_get, _coarseSunSensor.SysModel_ModelTag_set)
    CallCounts = property(_coarseSunSensor.SysModel_CallCounts_get, _coarseSunSensor.SysModel_CallCounts_set)
    RNGSeed = property(_coarseSunSensor.SysModel_RNGSeed_get, _coarseSunSensor.SysModel_RNGSeed_set)
    moduleID = property(_coarseSunSensor.SysModel_moduleID_get, _coarseSunSensor.SysModel_moduleID_set)

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


# Register SysModel in _coarseSunSensor:
_coarseSunSensor.SysModel_swigregister(SysModel)
CSSFAULT_OFF = _coarseSunSensor.CSSFAULT_OFF
CSSFAULT_STUCK_CURRENT = _coarseSunSensor.CSSFAULT_STUCK_CURRENT
CSSFAULT_STUCK_MAX = _coarseSunSensor.CSSFAULT_STUCK_MAX
CSSFAULT_STUCK_RAND = _coarseSunSensor.CSSFAULT_STUCK_RAND
CSSFAULT_RAND = _coarseSunSensor.CSSFAULT_RAND
NOMINAL = _coarseSunSensor.NOMINAL
class CoarseSunSensor(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _coarseSunSensor.CoarseSunSensor_swiginit(self, _coarseSunSensor.new_CoarseSunSensor())
    __swig_destroy__ = _coarseSunSensor.delete_CoarseSunSensor

    def Reset(self, CurrentClock):
        return _coarseSunSensor.CoarseSunSensor_Reset(self, CurrentClock)

    def UpdateState(self, CurrentSimNanos):
        return _coarseSunSensor.CoarseSunSensor_UpdateState(self, CurrentSimNanos)

    def setUnitDirectionVectorWithPerturbation(self, cssThetaPerturb, cssPhiPerturb):
        return _coarseSunSensor.CoarseSunSensor_setUnitDirectionVectorWithPerturbation(self, cssThetaPerturb, cssPhiPerturb)

    def setBodyToPlatformDCM(self, yaw, pitch, roll):
        return _coarseSunSensor.CoarseSunSensor_setBodyToPlatformDCM(self, yaw, pitch, roll)

    def readInputMessages(self):
        return _coarseSunSensor.CoarseSunSensor_readInputMessages(self)

    def computeSunData(self):
        return _coarseSunSensor.CoarseSunSensor_computeSunData(self)

    def computeTrueOutput(self):
        return _coarseSunSensor.CoarseSunSensor_computeTrueOutput(self)

    def applySensorErrors(self):
        return _coarseSunSensor.CoarseSunSensor_applySensorErrors(self)

    def scaleSensorValues(self):
        return _coarseSunSensor.CoarseSunSensor_scaleSensorValues(self)

    def applySaturation(self):
        return _coarseSunSensor.CoarseSunSensor_applySaturation(self)

    def writeOutputMessages(self, Clock):
        return _coarseSunSensor.CoarseSunSensor_writeOutputMessages(self, Clock)
    sunInMsg = property(_coarseSunSensor.CoarseSunSensor_sunInMsg_get, _coarseSunSensor.CoarseSunSensor_sunInMsg_set)
    stateInMsg = property(_coarseSunSensor.CoarseSunSensor_stateInMsg_get, _coarseSunSensor.CoarseSunSensor_stateInMsg_set)
    cssDataOutMsg = property(_coarseSunSensor.CoarseSunSensor_cssDataOutMsg_get, _coarseSunSensor.CoarseSunSensor_cssDataOutMsg_set)
    cssConfigLogOutMsg = property(_coarseSunSensor.CoarseSunSensor_cssConfigLogOutMsg_get, _coarseSunSensor.CoarseSunSensor_cssConfigLogOutMsg_set)
    sunEclipseInMsg = property(_coarseSunSensor.CoarseSunSensor_sunEclipseInMsg_get, _coarseSunSensor.CoarseSunSensor_sunEclipseInMsg_set)
    albedoInMsg = property(_coarseSunSensor.CoarseSunSensor_albedoInMsg_get, _coarseSunSensor.CoarseSunSensor_albedoInMsg_set)
    faultState = property(_coarseSunSensor.CoarseSunSensor_faultState_get, _coarseSunSensor.CoarseSunSensor_faultState_set)
    theta = property(_coarseSunSensor.CoarseSunSensor_theta_get, _coarseSunSensor.CoarseSunSensor_theta_set)
    phi = property(_coarseSunSensor.CoarseSunSensor_phi_get, _coarseSunSensor.CoarseSunSensor_phi_set)
    B2P321Angles = property(_coarseSunSensor.CoarseSunSensor_B2P321Angles_get, _coarseSunSensor.CoarseSunSensor_B2P321Angles_set)
    dcm_PB = property(_coarseSunSensor.CoarseSunSensor_dcm_PB_get, _coarseSunSensor.CoarseSunSensor_dcm_PB_set)
    nHat_B = property(_coarseSunSensor.CoarseSunSensor_nHat_B_get, _coarseSunSensor.CoarseSunSensor_nHat_B_set)
    sHat_B = property(_coarseSunSensor.CoarseSunSensor_sHat_B_get, _coarseSunSensor.CoarseSunSensor_sHat_B_set)
    albedoValue = property(_coarseSunSensor.CoarseSunSensor_albedoValue_get, _coarseSunSensor.CoarseSunSensor_albedoValue_set)
    scaleFactor = property(_coarseSunSensor.CoarseSunSensor_scaleFactor_get, _coarseSunSensor.CoarseSunSensor_scaleFactor_set)
    pastValue = property(_coarseSunSensor.CoarseSunSensor_pastValue_get, _coarseSunSensor.CoarseSunSensor_pastValue_set)
    trueValue = property(_coarseSunSensor.CoarseSunSensor_trueValue_get, _coarseSunSensor.CoarseSunSensor_trueValue_set)
    sensedValue = property(_coarseSunSensor.CoarseSunSensor_sensedValue_get, _coarseSunSensor.CoarseSunSensor_sensedValue_set)
    kellyFactor = property(_coarseSunSensor.CoarseSunSensor_kellyFactor_get, _coarseSunSensor.CoarseSunSensor_kellyFactor_set)
    fov = property(_coarseSunSensor.CoarseSunSensor_fov_get, _coarseSunSensor.CoarseSunSensor_fov_set)
    r_B = property(_coarseSunSensor.CoarseSunSensor_r_B_get, _coarseSunSensor.CoarseSunSensor_r_B_set)
    r_PB_B = property(_coarseSunSensor.CoarseSunSensor_r_PB_B_get, _coarseSunSensor.CoarseSunSensor_r_PB_B_set)
    senBias = property(_coarseSunSensor.CoarseSunSensor_senBias_get, _coarseSunSensor.CoarseSunSensor_senBias_set)
    senNoiseStd = property(_coarseSunSensor.CoarseSunSensor_senNoiseStd_get, _coarseSunSensor.CoarseSunSensor_senNoiseStd_set)
    faultNoiseStd = property(_coarseSunSensor.CoarseSunSensor_faultNoiseStd_get, _coarseSunSensor.CoarseSunSensor_faultNoiseStd_set)
    maxOutput = property(_coarseSunSensor.CoarseSunSensor_maxOutput_get, _coarseSunSensor.CoarseSunSensor_maxOutput_set)
    minOutput = property(_coarseSunSensor.CoarseSunSensor_minOutput_get, _coarseSunSensor.CoarseSunSensor_minOutput_set)
    walkBounds = property(_coarseSunSensor.CoarseSunSensor_walkBounds_get, _coarseSunSensor.CoarseSunSensor_walkBounds_set)
    kPower = property(_coarseSunSensor.CoarseSunSensor_kPower_get, _coarseSunSensor.CoarseSunSensor_kPower_set)
    CSSGroupID = property(_coarseSunSensor.CoarseSunSensor_CSSGroupID_get, _coarseSunSensor.CoarseSunSensor_CSSGroupID_set)
    bskLogger = property(_coarseSunSensor.CoarseSunSensor_bskLogger_get, _coarseSunSensor.CoarseSunSensor_bskLogger_set)

# Register CoarseSunSensor in _coarseSunSensor:
_coarseSunSensor.CoarseSunSensor_swigregister(CoarseSunSensor)
class CSSConstellation(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _coarseSunSensor.CSSConstellation_swiginit(self, _coarseSunSensor.new_CSSConstellation())
    __swig_destroy__ = _coarseSunSensor.delete_CSSConstellation

    def Reset(self, CurrentClock):
        return _coarseSunSensor.CSSConstellation_Reset(self, CurrentClock)

    def UpdateState(self, CurrentSimNanos):
        return _coarseSunSensor.CSSConstellation_UpdateState(self, CurrentSimNanos)

    def appendCSS(self, newSensor):
        return _coarseSunSensor.CSSConstellation_appendCSS(self, newSensor)
    constellationOutMsg = property(_coarseSunSensor.CSSConstellation_constellationOutMsg_get, _coarseSunSensor.CSSConstellation_constellationOutMsg_set)
    sensorList = property(_coarseSunSensor.CSSConstellation_sensorList_get, _coarseSunSensor.CSSConstellation_sensorList_set)

# Register CSSConstellation in _coarseSunSensor:
_coarseSunSensor.CSSConstellation_swigregister(CSSConstellation)
class SCStatesMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    r_BN_N = property(_coarseSunSensor.SCStatesMsgPayload_r_BN_N_get, _coarseSunSensor.SCStatesMsgPayload_r_BN_N_set)
    v_BN_N = property(_coarseSunSensor.SCStatesMsgPayload_v_BN_N_get, _coarseSunSensor.SCStatesMsgPayload_v_BN_N_set)
    r_CN_N = property(_coarseSunSensor.SCStatesMsgPayload_r_CN_N_get, _coarseSunSensor.SCStatesMsgPayload_r_CN_N_set)
    v_CN_N = property(_coarseSunSensor.SCStatesMsgPayload_v_CN_N_get, _coarseSunSensor.SCStatesMsgPayload_v_CN_N_set)
    sigma_BN = property(_coarseSunSensor.SCStatesMsgPayload_sigma_BN_get, _coarseSunSensor.SCStatesMsgPayload_sigma_BN_set)
    omega_BN_B = property(_coarseSunSensor.SCStatesMsgPayload_omega_BN_B_get, _coarseSunSensor.SCStatesMsgPayload_omega_BN_B_set)
    omegaDot_BN_B = property(_coarseSunSensor.SCStatesMsgPayload_omegaDot_BN_B_get, _coarseSunSensor.SCStatesMsgPayload_omegaDot_BN_B_set)
    TotalAccumDVBdy = property(_coarseSunSensor.SCStatesMsgPayload_TotalAccumDVBdy_get, _coarseSunSensor.SCStatesMsgPayload_TotalAccumDVBdy_set)
    TotalAccumDV_BN_B = property(_coarseSunSensor.SCStatesMsgPayload_TotalAccumDV_BN_B_get, _coarseSunSensor.SCStatesMsgPayload_TotalAccumDV_BN_B_set)
    nonConservativeAccelpntB_B = property(_coarseSunSensor.SCStatesMsgPayload_nonConservativeAccelpntB_B_get, _coarseSunSensor.SCStatesMsgPayload_nonConservativeAccelpntB_B_set)
    MRPSwitchCount = property(_coarseSunSensor.SCStatesMsgPayload_MRPSwitchCount_get, _coarseSunSensor.SCStatesMsgPayload_MRPSwitchCount_set)

    def __init__(self):
        _coarseSunSensor.SCStatesMsgPayload_swiginit(self, _coarseSunSensor.new_SCStatesMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_SCStatesMsgPayload

# Register SCStatesMsgPayload in _coarseSunSensor:
_coarseSunSensor.SCStatesMsgPayload_swigregister(SCStatesMsgPayload)
MAX_BODY_NAME_LENGTH = _coarseSunSensor.MAX_BODY_NAME_LENGTH
class SpicePlanetStateMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    J2000Current = property(_coarseSunSensor.SpicePlanetStateMsgPayload_J2000Current_get, _coarseSunSensor.SpicePlanetStateMsgPayload_J2000Current_set)
    PositionVector = property(_coarseSunSensor.SpicePlanetStateMsgPayload_PositionVector_get, _coarseSunSensor.SpicePlanetStateMsgPayload_PositionVector_set)
    VelocityVector = property(_coarseSunSensor.SpicePlanetStateMsgPayload_VelocityVector_get, _coarseSunSensor.SpicePlanetStateMsgPayload_VelocityVector_set)
    J20002Pfix = property(_coarseSunSensor.SpicePlanetStateMsgPayload_J20002Pfix_get, _coarseSunSensor.SpicePlanetStateMsgPayload_J20002Pfix_set)
    J20002Pfix_dot = property(_coarseSunSensor.SpicePlanetStateMsgPayload_J20002Pfix_dot_get, _coarseSunSensor.SpicePlanetStateMsgPayload_J20002Pfix_dot_set)
    computeOrient = property(_coarseSunSensor.SpicePlanetStateMsgPayload_computeOrient_get, _coarseSunSensor.SpicePlanetStateMsgPayload_computeOrient_set)
    PlanetName = property(_coarseSunSensor.SpicePlanetStateMsgPayload_PlanetName_get, _coarseSunSensor.SpicePlanetStateMsgPayload_PlanetName_set)

    def __init__(self):
        _coarseSunSensor.SpicePlanetStateMsgPayload_swiginit(self, _coarseSunSensor.new_SpicePlanetStateMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_SpicePlanetStateMsgPayload

# Register SpicePlanetStateMsgPayload in _coarseSunSensor:
_coarseSunSensor.SpicePlanetStateMsgPayload_swigregister(SpicePlanetStateMsgPayload)
class CSSRawDataMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    OutputData = property(_coarseSunSensor.CSSRawDataMsgPayload_OutputData_get, _coarseSunSensor.CSSRawDataMsgPayload_OutputData_set)

    def __init__(self):
        _coarseSunSensor.CSSRawDataMsgPayload_swiginit(self, _coarseSunSensor.new_CSSRawDataMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_CSSRawDataMsgPayload

# Register CSSRawDataMsgPayload in _coarseSunSensor:
_coarseSunSensor.CSSRawDataMsgPayload_swigregister(CSSRawDataMsgPayload)
class AlbedoMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    albedoAtInstrumentMax = property(_coarseSunSensor.AlbedoMsgPayload_albedoAtInstrumentMax_get, _coarseSunSensor.AlbedoMsgPayload_albedoAtInstrumentMax_set)
    AfluxAtInstrumentMax = property(_coarseSunSensor.AlbedoMsgPayload_AfluxAtInstrumentMax_get, _coarseSunSensor.AlbedoMsgPayload_AfluxAtInstrumentMax_set)
    albedoAtInstrument = property(_coarseSunSensor.AlbedoMsgPayload_albedoAtInstrument_get, _coarseSunSensor.AlbedoMsgPayload_albedoAtInstrument_set)
    AfluxAtInstrument = property(_coarseSunSensor.AlbedoMsgPayload_AfluxAtInstrument_get, _coarseSunSensor.AlbedoMsgPayload_AfluxAtInstrument_set)

    def __init__(self):
        _coarseSunSensor.AlbedoMsgPayload_swiginit(self, _coarseSunSensor.new_AlbedoMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_AlbedoMsgPayload

# Register AlbedoMsgPayload in _coarseSunSensor:
_coarseSunSensor.AlbedoMsgPayload_swigregister(AlbedoMsgPayload)
class EclipseMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    shadowFactor = property(_coarseSunSensor.EclipseMsgPayload_shadowFactor_get, _coarseSunSensor.EclipseMsgPayload_shadowFactor_set)

    def __init__(self):
        _coarseSunSensor.EclipseMsgPayload_swiginit(self, _coarseSunSensor.new_EclipseMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_EclipseMsgPayload

# Register EclipseMsgPayload in _coarseSunSensor:
_coarseSunSensor.EclipseMsgPayload_swigregister(EclipseMsgPayload)
class CSSArraySensorMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    CosValue = property(_coarseSunSensor.CSSArraySensorMsgPayload_CosValue_get, _coarseSunSensor.CSSArraySensorMsgPayload_CosValue_set)

    def __init__(self):
        _coarseSunSensor.CSSArraySensorMsgPayload_swiginit(self, _coarseSunSensor.new_CSSArraySensorMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_CSSArraySensorMsgPayload

# Register CSSArraySensorMsgPayload in _coarseSunSensor:
_coarseSunSensor.CSSArraySensorMsgPayload_swigregister(CSSArraySensorMsgPayload)
class CSSConfigLogMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    r_B = property(_coarseSunSensor.CSSConfigLogMsgPayload_r_B_get, _coarseSunSensor.CSSConfigLogMsgPayload_r_B_set)
    nHat_B = property(_coarseSunSensor.CSSConfigLogMsgPayload_nHat_B_get, _coarseSunSensor.CSSConfigLogMsgPayload_nHat_B_set)
    fov = property(_coarseSunSensor.CSSConfigLogMsgPayload_fov_get, _coarseSunSensor.CSSConfigLogMsgPayload_fov_set)
    signal = property(_coarseSunSensor.CSSConfigLogMsgPayload_signal_get, _coarseSunSensor.CSSConfigLogMsgPayload_signal_set)
    maxSignal = property(_coarseSunSensor.CSSConfigLogMsgPayload_maxSignal_get, _coarseSunSensor.CSSConfigLogMsgPayload_maxSignal_set)
    minSignal = property(_coarseSunSensor.CSSConfigLogMsgPayload_minSignal_get, _coarseSunSensor.CSSConfigLogMsgPayload_minSignal_set)
    CSSGroupID = property(_coarseSunSensor.CSSConfigLogMsgPayload_CSSGroupID_get, _coarseSunSensor.CSSConfigLogMsgPayload_CSSGroupID_set)

    def __init__(self):
        _coarseSunSensor.CSSConfigLogMsgPayload_swiginit(self, _coarseSunSensor.new_CSSConfigLogMsgPayload())
    __swig_destroy__ = _coarseSunSensor.delete_CSSConfigLogMsgPayload

# Register CSSConfigLogMsgPayload in _coarseSunSensor:
_coarseSunSensor.CSSConfigLogMsgPayload_swigregister(CSSConfigLogMsgPayload)
class CSSVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _coarseSunSensor.CSSVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _coarseSunSensor.CSSVector___nonzero__(self)

    def __bool__(self):
        return _coarseSunSensor.CSSVector___bool__(self)

    def __len__(self):
        return _coarseSunSensor.CSSVector___len__(self)

    def __getslice__(self, i, j):
        return _coarseSunSensor.CSSVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _coarseSunSensor.CSSVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _coarseSunSensor.CSSVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _coarseSunSensor.CSSVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _coarseSunSensor.CSSVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _coarseSunSensor.CSSVector___setitem__(self, *args)

    def pop(self):
        return _coarseSunSensor.CSSVector_pop(self)

    def append(self, x):
        return _coarseSunSensor.CSSVector_append(self, x)

    def empty(self):
        return _coarseSunSensor.CSSVector_empty(self)

    def size(self):
        return _coarseSunSensor.CSSVector_size(self)

    def swap(self, v):
        return _coarseSunSensor.CSSVector_swap(self, v)

    def begin(self):
        return _coarseSunSensor.CSSVector_begin(self)

    def end(self):
        return _coarseSunSensor.CSSVector_end(self)

    def rbegin(self):
        return _coarseSunSensor.CSSVector_rbegin(self)

    def rend(self):
        return _coarseSunSensor.CSSVector_rend(self)

    def clear(self):
        return _coarseSunSensor.CSSVector_clear(self)

    def get_allocator(self):
        return _coarseSunSensor.CSSVector_get_allocator(self)

    def pop_back(self):
        return _coarseSunSensor.CSSVector_pop_back(self)

    def erase(self, *args):
        return _coarseSunSensor.CSSVector_erase(self, *args)

    def __init__(self, *args):
        _coarseSunSensor.CSSVector_swiginit(self, _coarseSunSensor.new_CSSVector(*args))

    def push_back(self, x):
        return _coarseSunSensor.CSSVector_push_back(self, x)

    def front(self):
        return _coarseSunSensor.CSSVector_front(self)

    def back(self):
        return _coarseSunSensor.CSSVector_back(self)

    def assign(self, n, x):
        return _coarseSunSensor.CSSVector_assign(self, n, x)

    def resize(self, *args):
        return _coarseSunSensor.CSSVector_resize(self, *args)

    def insert(self, *args):
        return _coarseSunSensor.CSSVector_insert(self, *args)

    def reserve(self, n):
        return _coarseSunSensor.CSSVector_reserve(self, n)

    def capacity(self):
        return _coarseSunSensor.CSSVector_capacity(self)
    __swig_destroy__ = _coarseSunSensor.delete_CSSVector

# Register CSSVector in _coarseSunSensor:
_coarseSunSensor.CSSVector_swigregister(CSSVector)

import sys
protectAllClasses(sys.modules[__name__])


