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
    from . import _relativeODuKF
else:
    import _relativeODuKF

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
    return _relativeODuKF.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _relativeODuKF.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _relativeODuKF.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _relativeODuKF.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _relativeODuKF.new_longArray(nelements)

def delete_longArray(ary):
    return _relativeODuKF.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _relativeODuKF.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _relativeODuKF.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _relativeODuKF.new_intArray(nelements)

def delete_intArray(ary):
    return _relativeODuKF.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _relativeODuKF.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _relativeODuKF.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _relativeODuKF.new_shortArray(nelements)

def delete_shortArray(ary):
    return _relativeODuKF.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _relativeODuKF.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _relativeODuKF.shortArray_setitem(ary, index, value)


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


MAX_LOGGING_LENGTH = _relativeODuKF.MAX_LOGGING_LENGTH
BSK_DEBUG = _relativeODuKF.BSK_DEBUG
BSK_INFORMATION = _relativeODuKF.BSK_INFORMATION
BSK_WARNING = _relativeODuKF.BSK_WARNING
BSK_ERROR = _relativeODuKF.BSK_ERROR
BSK_SILENT = _relativeODuKF.BSK_SILENT

def printDefaultLogLevel():
    return _relativeODuKF.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _relativeODuKF.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _relativeODuKF.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _relativeODuKF.BSKLogger_swiginit(self, _relativeODuKF.new_BSKLogger(*args))
    __swig_destroy__ = _relativeODuKF.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _relativeODuKF.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _relativeODuKF.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _relativeODuKF.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _relativeODuKF.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_relativeODuKF.BSKLogger_logLevelMap_get, _relativeODuKF.BSKLogger_logLevelMap_set)

# Register BSKLogger in _relativeODuKF:
_relativeODuKF.BSKLogger_swigregister(BSKLogger)
cvar = _relativeODuKF.cvar


def _BSKLogger():
    return _relativeODuKF._BSKLogger()

def _BSKLogger_d(arg1):
    return _relativeODuKF._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _relativeODuKF._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _relativeODuKF._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _relativeODuKF._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _relativeODuKF.SysModel_swiginit(self, _relativeODuKF.new_SysModel(*args))
    __swig_destroy__ = _relativeODuKF.delete_SysModel

    def SelfInit(self):
        return _relativeODuKF.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _relativeODuKF.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _relativeODuKF.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _relativeODuKF.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_relativeODuKF.SysModel_ModelTag_get, _relativeODuKF.SysModel_ModelTag_set)
    CallCounts = property(_relativeODuKF.SysModel_CallCounts_get, _relativeODuKF.SysModel_CallCounts_set)
    RNGSeed = property(_relativeODuKF.SysModel_RNGSeed_get, _relativeODuKF.SysModel_RNGSeed_set)
    moduleID = property(_relativeODuKF.SysModel_moduleID_get, _relativeODuKF.SysModel_moduleID_set)

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


# Register SysModel in _relativeODuKF:
_relativeODuKF.SysModel_swigregister(SysModel)
class RelODuKFConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    navStateOutMsg = property(_relativeODuKF.RelODuKFConfig_navStateOutMsg_get, _relativeODuKF.RelODuKFConfig_navStateOutMsg_set)
    filtDataOutMsg = property(_relativeODuKF.RelODuKFConfig_filtDataOutMsg_get, _relativeODuKF.RelODuKFConfig_filtDataOutMsg_set)
    opNavInMsg = property(_relativeODuKF.RelODuKFConfig_opNavInMsg_get, _relativeODuKF.RelODuKFConfig_opNavInMsg_set)
    numStates = property(_relativeODuKF.RelODuKFConfig_numStates_get, _relativeODuKF.RelODuKFConfig_numStates_set)
    countHalfSPs = property(_relativeODuKF.RelODuKFConfig_countHalfSPs_get, _relativeODuKF.RelODuKFConfig_countHalfSPs_set)
    numObs = property(_relativeODuKF.RelODuKFConfig_numObs_get, _relativeODuKF.RelODuKFConfig_numObs_set)
    beta = property(_relativeODuKF.RelODuKFConfig_beta_get, _relativeODuKF.RelODuKFConfig_beta_set)
    alpha = property(_relativeODuKF.RelODuKFConfig_alpha_get, _relativeODuKF.RelODuKFConfig_alpha_set)
    kappa = property(_relativeODuKF.RelODuKFConfig_kappa_get, _relativeODuKF.RelODuKFConfig_kappa_set)
    lambdaVal = property(_relativeODuKF.RelODuKFConfig_lambdaVal_get, _relativeODuKF.RelODuKFConfig_lambdaVal_set)
    gamma = property(_relativeODuKF.RelODuKFConfig_gamma_get, _relativeODuKF.RelODuKFConfig_gamma_set)
    switchMag = property(_relativeODuKF.RelODuKFConfig_switchMag_get, _relativeODuKF.RelODuKFConfig_switchMag_set)
    dt = property(_relativeODuKF.RelODuKFConfig_dt_get, _relativeODuKF.RelODuKFConfig_dt_set)
    timeTag = property(_relativeODuKF.RelODuKFConfig_timeTag_get, _relativeODuKF.RelODuKFConfig_timeTag_set)
    gyrAggTimeTag = property(_relativeODuKF.RelODuKFConfig_gyrAggTimeTag_get, _relativeODuKF.RelODuKFConfig_gyrAggTimeTag_set)
    aggSigma_b2b1 = property(_relativeODuKF.RelODuKFConfig_aggSigma_b2b1_get, _relativeODuKF.RelODuKFConfig_aggSigma_b2b1_set)
    dcm_BdyGyrpltf = property(_relativeODuKF.RelODuKFConfig_dcm_BdyGyrpltf_get, _relativeODuKF.RelODuKFConfig_dcm_BdyGyrpltf_set)
    wM = property(_relativeODuKF.RelODuKFConfig_wM_get, _relativeODuKF.RelODuKFConfig_wM_set)
    wC = property(_relativeODuKF.RelODuKFConfig_wC_get, _relativeODuKF.RelODuKFConfig_wC_set)
    stateInit = property(_relativeODuKF.RelODuKFConfig_stateInit_get, _relativeODuKF.RelODuKFConfig_stateInit_set)
    state = property(_relativeODuKF.RelODuKFConfig_state_get, _relativeODuKF.RelODuKFConfig_state_set)
    statePrev = property(_relativeODuKF.RelODuKFConfig_statePrev_get, _relativeODuKF.RelODuKFConfig_statePrev_set)
    sBar = property(_relativeODuKF.RelODuKFConfig_sBar_get, _relativeODuKF.RelODuKFConfig_sBar_set)
    sBarPrev = property(_relativeODuKF.RelODuKFConfig_sBarPrev_get, _relativeODuKF.RelODuKFConfig_sBarPrev_set)
    covar = property(_relativeODuKF.RelODuKFConfig_covar_get, _relativeODuKF.RelODuKFConfig_covar_set)
    covarPrev = property(_relativeODuKF.RelODuKFConfig_covarPrev_get, _relativeODuKF.RelODuKFConfig_covarPrev_set)
    covarInit = property(_relativeODuKF.RelODuKFConfig_covarInit_get, _relativeODuKF.RelODuKFConfig_covarInit_set)
    xBar = property(_relativeODuKF.RelODuKFConfig_xBar_get, _relativeODuKF.RelODuKFConfig_xBar_set)
    obs = property(_relativeODuKF.RelODuKFConfig_obs_get, _relativeODuKF.RelODuKFConfig_obs_set)
    yMeas = property(_relativeODuKF.RelODuKFConfig_yMeas_get, _relativeODuKF.RelODuKFConfig_yMeas_set)
    SP = property(_relativeODuKF.RelODuKFConfig_SP_get, _relativeODuKF.RelODuKFConfig_SP_set)
    qNoise = property(_relativeODuKF.RelODuKFConfig_qNoise_get, _relativeODuKF.RelODuKFConfig_qNoise_set)
    sQnoise = property(_relativeODuKF.RelODuKFConfig_sQnoise_get, _relativeODuKF.RelODuKFConfig_sQnoise_set)
    measNoise = property(_relativeODuKF.RelODuKFConfig_measNoise_get, _relativeODuKF.RelODuKFConfig_measNoise_set)
    noiseSF = property(_relativeODuKF.RelODuKFConfig_noiseSF_get, _relativeODuKF.RelODuKFConfig_noiseSF_set)
    planetIdInit = property(_relativeODuKF.RelODuKFConfig_planetIdInit_get, _relativeODuKF.RelODuKFConfig_planetIdInit_set)
    planetId = property(_relativeODuKF.RelODuKFConfig_planetId_get, _relativeODuKF.RelODuKFConfig_planetId_set)
    firstPassComplete = property(_relativeODuKF.RelODuKFConfig_firstPassComplete_get, _relativeODuKF.RelODuKFConfig_firstPassComplete_set)
    postFits = property(_relativeODuKF.RelODuKFConfig_postFits_get, _relativeODuKF.RelODuKFConfig_postFits_set)
    timeTagOut = property(_relativeODuKF.RelODuKFConfig_timeTagOut_get, _relativeODuKF.RelODuKFConfig_timeTagOut_set)
    maxTimeJump = property(_relativeODuKF.RelODuKFConfig_maxTimeJump_get, _relativeODuKF.RelODuKFConfig_maxTimeJump_set)
    opNavInBuffer = property(_relativeODuKF.RelODuKFConfig_opNavInBuffer_get, _relativeODuKF.RelODuKFConfig_opNavInBuffer_set)
    bskLogger = property(_relativeODuKF.RelODuKFConfig_bskLogger_get, _relativeODuKF.RelODuKFConfig_bskLogger_set)

    def createWrapper(self):
        return relativeODuKF(self)


    def __init__(self):
        _relativeODuKF.RelODuKFConfig_swiginit(self, _relativeODuKF.new_RelODuKFConfig())
    __swig_destroy__ = _relativeODuKF.delete_RelODuKFConfig

# Register RelODuKFConfig in _relativeODuKF:
_relativeODuKF.RelODuKFConfig_swigregister(RelODuKFConfig)

def relODuKFTwoBodyDyn(state, mu, stateDeriv):
    return _relativeODuKF.relODuKFTwoBodyDyn(state, mu, stateDeriv)

def relODuKFTimeUpdate(configData, updateTime):
    return _relativeODuKF.relODuKFTimeUpdate(configData, updateTime)

def relODuKFMeasUpdate(configData):
    return _relativeODuKF.relODuKFMeasUpdate(configData)

def relODuKFCleanUpdate(configData):
    return _relativeODuKF.relODuKFCleanUpdate(configData)

def relODStateProp(configData, stateInOut, dt):
    return _relativeODuKF.relODStateProp(configData, stateInOut, dt)

def relODuKFMeasModel(configData):
    return _relativeODuKF.relODuKFMeasModel(configData)
class relativeODuKF(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _relativeODuKF.relativeODuKF_swiginit(self, _relativeODuKF.new_relativeODuKF(*args))

        if (len(args)) > 0:
            args[0].thisown = False




    def SelfInit(self):
        return _relativeODuKF.relativeODuKF_SelfInit(self)

    def UpdateState(self, currentSimNanos):
        return _relativeODuKF.relativeODuKF_UpdateState(self, currentSimNanos)

    def Reset(self, currentSimNanos):
        return _relativeODuKF.relativeODuKF_Reset(self, currentSimNanos)

    def __deref__(self):
        return _relativeODuKF.relativeODuKF___deref__(self)

    def getConfig(self):
        return _relativeODuKF.relativeODuKF_getConfig(self)
    __swig_destroy__ = _relativeODuKF.delete_relativeODuKF
    navStateOutMsg = property(_relativeODuKF.relativeODuKF_navStateOutMsg_get, _relativeODuKF.relativeODuKF_navStateOutMsg_set)
    filtDataOutMsg = property(_relativeODuKF.relativeODuKF_filtDataOutMsg_get, _relativeODuKF.relativeODuKF_filtDataOutMsg_set)
    opNavInMsg = property(_relativeODuKF.relativeODuKF_opNavInMsg_get, _relativeODuKF.relativeODuKF_opNavInMsg_set)
    numStates = property(_relativeODuKF.relativeODuKF_numStates_get, _relativeODuKF.relativeODuKF_numStates_set)
    countHalfSPs = property(_relativeODuKF.relativeODuKF_countHalfSPs_get, _relativeODuKF.relativeODuKF_countHalfSPs_set)
    numObs = property(_relativeODuKF.relativeODuKF_numObs_get, _relativeODuKF.relativeODuKF_numObs_set)
    beta = property(_relativeODuKF.relativeODuKF_beta_get, _relativeODuKF.relativeODuKF_beta_set)
    alpha = property(_relativeODuKF.relativeODuKF_alpha_get, _relativeODuKF.relativeODuKF_alpha_set)
    kappa = property(_relativeODuKF.relativeODuKF_kappa_get, _relativeODuKF.relativeODuKF_kappa_set)
    lambdaVal = property(_relativeODuKF.relativeODuKF_lambdaVal_get, _relativeODuKF.relativeODuKF_lambdaVal_set)
    gamma = property(_relativeODuKF.relativeODuKF_gamma_get, _relativeODuKF.relativeODuKF_gamma_set)
    switchMag = property(_relativeODuKF.relativeODuKF_switchMag_get, _relativeODuKF.relativeODuKF_switchMag_set)
    dt = property(_relativeODuKF.relativeODuKF_dt_get, _relativeODuKF.relativeODuKF_dt_set)
    timeTag = property(_relativeODuKF.relativeODuKF_timeTag_get, _relativeODuKF.relativeODuKF_timeTag_set)
    gyrAggTimeTag = property(_relativeODuKF.relativeODuKF_gyrAggTimeTag_get, _relativeODuKF.relativeODuKF_gyrAggTimeTag_set)
    aggSigma_b2b1 = property(_relativeODuKF.relativeODuKF_aggSigma_b2b1_get, _relativeODuKF.relativeODuKF_aggSigma_b2b1_set)
    dcm_BdyGyrpltf = property(_relativeODuKF.relativeODuKF_dcm_BdyGyrpltf_get, _relativeODuKF.relativeODuKF_dcm_BdyGyrpltf_set)
    wM = property(_relativeODuKF.relativeODuKF_wM_get, _relativeODuKF.relativeODuKF_wM_set)
    wC = property(_relativeODuKF.relativeODuKF_wC_get, _relativeODuKF.relativeODuKF_wC_set)
    stateInit = property(_relativeODuKF.relativeODuKF_stateInit_get, _relativeODuKF.relativeODuKF_stateInit_set)
    state = property(_relativeODuKF.relativeODuKF_state_get, _relativeODuKF.relativeODuKF_state_set)
    statePrev = property(_relativeODuKF.relativeODuKF_statePrev_get, _relativeODuKF.relativeODuKF_statePrev_set)
    sBar = property(_relativeODuKF.relativeODuKF_sBar_get, _relativeODuKF.relativeODuKF_sBar_set)
    sBarPrev = property(_relativeODuKF.relativeODuKF_sBarPrev_get, _relativeODuKF.relativeODuKF_sBarPrev_set)
    covar = property(_relativeODuKF.relativeODuKF_covar_get, _relativeODuKF.relativeODuKF_covar_set)
    covarPrev = property(_relativeODuKF.relativeODuKF_covarPrev_get, _relativeODuKF.relativeODuKF_covarPrev_set)
    covarInit = property(_relativeODuKF.relativeODuKF_covarInit_get, _relativeODuKF.relativeODuKF_covarInit_set)
    xBar = property(_relativeODuKF.relativeODuKF_xBar_get, _relativeODuKF.relativeODuKF_xBar_set)
    obs = property(_relativeODuKF.relativeODuKF_obs_get, _relativeODuKF.relativeODuKF_obs_set)
    yMeas = property(_relativeODuKF.relativeODuKF_yMeas_get, _relativeODuKF.relativeODuKF_yMeas_set)
    SP = property(_relativeODuKF.relativeODuKF_SP_get, _relativeODuKF.relativeODuKF_SP_set)
    qNoise = property(_relativeODuKF.relativeODuKF_qNoise_get, _relativeODuKF.relativeODuKF_qNoise_set)
    sQnoise = property(_relativeODuKF.relativeODuKF_sQnoise_get, _relativeODuKF.relativeODuKF_sQnoise_set)
    measNoise = property(_relativeODuKF.relativeODuKF_measNoise_get, _relativeODuKF.relativeODuKF_measNoise_set)
    noiseSF = property(_relativeODuKF.relativeODuKF_noiseSF_get, _relativeODuKF.relativeODuKF_noiseSF_set)
    planetIdInit = property(_relativeODuKF.relativeODuKF_planetIdInit_get, _relativeODuKF.relativeODuKF_planetIdInit_set)
    planetId = property(_relativeODuKF.relativeODuKF_planetId_get, _relativeODuKF.relativeODuKF_planetId_set)
    firstPassComplete = property(_relativeODuKF.relativeODuKF_firstPassComplete_get, _relativeODuKF.relativeODuKF_firstPassComplete_set)
    postFits = property(_relativeODuKF.relativeODuKF_postFits_get, _relativeODuKF.relativeODuKF_postFits_set)
    timeTagOut = property(_relativeODuKF.relativeODuKF_timeTagOut_get, _relativeODuKF.relativeODuKF_timeTagOut_set)
    maxTimeJump = property(_relativeODuKF.relativeODuKF_maxTimeJump_get, _relativeODuKF.relativeODuKF_maxTimeJump_set)
    opNavInBuffer = property(_relativeODuKF.relativeODuKF_opNavInBuffer_get, _relativeODuKF.relativeODuKF_opNavInBuffer_set)
    bskLogger = property(_relativeODuKF.relativeODuKF_bskLogger_get, _relativeODuKF.relativeODuKF_bskLogger_set)

# Register relativeODuKF in _relativeODuKF:
_relativeODuKF.relativeODuKF_swigregister(relativeODuKF)
UKF_MAX_DIM = _relativeODuKF.UKF_MAX_DIM

def ukfQRDJustR(sourceMat, nRow, nCol, destMat):
    return _relativeODuKF.ukfQRDJustR(sourceMat, nRow, nCol, destMat)

def ukfLInv(sourceMat, nRow, nCol, destMat):
    return _relativeODuKF.ukfLInv(sourceMat, nRow, nCol, destMat)

def ukfUInv(sourceMat, nRow, nCol, destMat):
    return _relativeODuKF.ukfUInv(sourceMat, nRow, nCol, destMat)

def ukfLUD(sourceMat, nRow, nCol, destMat, indx):
    return _relativeODuKF.ukfLUD(sourceMat, nRow, nCol, destMat, indx)

def ukfLUBckSlv(sourceMat, nRow, nCol, indx, bmat, destMat):
    return _relativeODuKF.ukfLUBckSlv(sourceMat, nRow, nCol, indx, bmat, destMat)

def ukfMatInv(sourceMat, nRow, nCol, destMat):
    return _relativeODuKF.ukfMatInv(sourceMat, nRow, nCol, destMat)

def ukfCholDecomp(sourceMat, nRow, nCol, destMat):
    return _relativeODuKF.ukfCholDecomp(sourceMat, nRow, nCol, destMat)

def ukfCholDownDate(rMat, xVec, beta, nStates, rOut):
    return _relativeODuKF.ukfCholDownDate(rMat, xVec, beta, nStates, rOut)
class NavTransMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    timeTag = property(_relativeODuKF.NavTransMsgPayload_timeTag_get, _relativeODuKF.NavTransMsgPayload_timeTag_set)
    r_BN_N = property(_relativeODuKF.NavTransMsgPayload_r_BN_N_get, _relativeODuKF.NavTransMsgPayload_r_BN_N_set)
    v_BN_N = property(_relativeODuKF.NavTransMsgPayload_v_BN_N_get, _relativeODuKF.NavTransMsgPayload_v_BN_N_set)
    vehAccumDV = property(_relativeODuKF.NavTransMsgPayload_vehAccumDV_get, _relativeODuKF.NavTransMsgPayload_vehAccumDV_set)

    def __init__(self):
        _relativeODuKF.NavTransMsgPayload_swiginit(self, _relativeODuKF.new_NavTransMsgPayload())
    __swig_destroy__ = _relativeODuKF.delete_NavTransMsgPayload

# Register NavTransMsgPayload in _relativeODuKF:
_relativeODuKF.NavTransMsgPayload_swigregister(NavTransMsgPayload)
class OpNavMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    timeTag = property(_relativeODuKF.OpNavMsgPayload_timeTag_get, _relativeODuKF.OpNavMsgPayload_timeTag_set)
    valid = property(_relativeODuKF.OpNavMsgPayload_valid_get, _relativeODuKF.OpNavMsgPayload_valid_set)
    covar_N = property(_relativeODuKF.OpNavMsgPayload_covar_N_get, _relativeODuKF.OpNavMsgPayload_covar_N_set)
    covar_B = property(_relativeODuKF.OpNavMsgPayload_covar_B_get, _relativeODuKF.OpNavMsgPayload_covar_B_set)
    covar_C = property(_relativeODuKF.OpNavMsgPayload_covar_C_get, _relativeODuKF.OpNavMsgPayload_covar_C_set)
    r_BN_N = property(_relativeODuKF.OpNavMsgPayload_r_BN_N_get, _relativeODuKF.OpNavMsgPayload_r_BN_N_set)
    r_BN_B = property(_relativeODuKF.OpNavMsgPayload_r_BN_B_get, _relativeODuKF.OpNavMsgPayload_r_BN_B_set)
    r_BN_C = property(_relativeODuKF.OpNavMsgPayload_r_BN_C_get, _relativeODuKF.OpNavMsgPayload_r_BN_C_set)
    planetID = property(_relativeODuKF.OpNavMsgPayload_planetID_get, _relativeODuKF.OpNavMsgPayload_planetID_set)
    faultDetected = property(_relativeODuKF.OpNavMsgPayload_faultDetected_get, _relativeODuKF.OpNavMsgPayload_faultDetected_set)

    def __init__(self):
        _relativeODuKF.OpNavMsgPayload_swiginit(self, _relativeODuKF.new_OpNavMsgPayload())
    __swig_destroy__ = _relativeODuKF.delete_OpNavMsgPayload

# Register OpNavMsgPayload in _relativeODuKF:
_relativeODuKF.OpNavMsgPayload_swigregister(OpNavMsgPayload)
ODUKF_N_STATES = _relativeODuKF.ODUKF_N_STATES
ODUKF_N_MEAS = _relativeODuKF.ODUKF_N_MEAS
class OpNavFilterMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    timeTag = property(_relativeODuKF.OpNavFilterMsgPayload_timeTag_get, _relativeODuKF.OpNavFilterMsgPayload_timeTag_set)
    covar = property(_relativeODuKF.OpNavFilterMsgPayload_covar_get, _relativeODuKF.OpNavFilterMsgPayload_covar_set)
    state = property(_relativeODuKF.OpNavFilterMsgPayload_state_get, _relativeODuKF.OpNavFilterMsgPayload_state_set)
    stateError = property(_relativeODuKF.OpNavFilterMsgPayload_stateError_get, _relativeODuKF.OpNavFilterMsgPayload_stateError_set)
    postFitRes = property(_relativeODuKF.OpNavFilterMsgPayload_postFitRes_get, _relativeODuKF.OpNavFilterMsgPayload_postFitRes_set)
    numObs = property(_relativeODuKF.OpNavFilterMsgPayload_numObs_get, _relativeODuKF.OpNavFilterMsgPayload_numObs_set)

    def __init__(self):
        _relativeODuKF.OpNavFilterMsgPayload_swiginit(self, _relativeODuKF.new_OpNavFilterMsgPayload())
    __swig_destroy__ = _relativeODuKF.delete_OpNavFilterMsgPayload

# Register OpNavFilterMsgPayload in _relativeODuKF:
_relativeODuKF.OpNavFilterMsgPayload_swigregister(OpNavFilterMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


