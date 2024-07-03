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
    from . import _thrusterStateEffector
else:
    import _thrusterStateEffector

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
    return _thrusterStateEffector.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _thrusterStateEffector.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _thrusterStateEffector.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _thrusterStateEffector.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _thrusterStateEffector.new_longArray(nelements)

def delete_longArray(ary):
    return _thrusterStateEffector.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _thrusterStateEffector.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _thrusterStateEffector.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _thrusterStateEffector.new_intArray(nelements)

def delete_intArray(ary):
    return _thrusterStateEffector.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _thrusterStateEffector.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _thrusterStateEffector.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _thrusterStateEffector.new_shortArray(nelements)

def delete_shortArray(ary):
    return _thrusterStateEffector.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _thrusterStateEffector.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _thrusterStateEffector.shortArray_setitem(ary, index, value)


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
    __swig_destroy__ = _thrusterStateEffector.delete_SwigPyIterator

    def value(self):
        return _thrusterStateEffector.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _thrusterStateEffector.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _thrusterStateEffector.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _thrusterStateEffector.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _thrusterStateEffector.SwigPyIterator_equal(self, x)

    def copy(self):
        return _thrusterStateEffector.SwigPyIterator_copy(self)

    def next(self):
        return _thrusterStateEffector.SwigPyIterator_next(self)

    def __next__(self):
        return _thrusterStateEffector.SwigPyIterator___next__(self)

    def previous(self):
        return _thrusterStateEffector.SwigPyIterator_previous(self)

    def advance(self, n):
        return _thrusterStateEffector.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _thrusterStateEffector.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _thrusterStateEffector.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _thrusterStateEffector.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _thrusterStateEffector.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _thrusterStateEffector.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _thrusterStateEffector.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _thrusterStateEffector:
_thrusterStateEffector.SwigPyIterator_swigregister(SwigPyIterator)

from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _thrusterStateEffector.MAX_LOGGING_LENGTH
BSK_DEBUG = _thrusterStateEffector.BSK_DEBUG
BSK_INFORMATION = _thrusterStateEffector.BSK_INFORMATION
BSK_WARNING = _thrusterStateEffector.BSK_WARNING
BSK_ERROR = _thrusterStateEffector.BSK_ERROR
BSK_SILENT = _thrusterStateEffector.BSK_SILENT

def printDefaultLogLevel():
    return _thrusterStateEffector.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _thrusterStateEffector.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _thrusterStateEffector.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _thrusterStateEffector.BSKLogger_swiginit(self, _thrusterStateEffector.new_BSKLogger(*args))
    __swig_destroy__ = _thrusterStateEffector.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _thrusterStateEffector.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _thrusterStateEffector.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _thrusterStateEffector.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _thrusterStateEffector.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_thrusterStateEffector.BSKLogger_logLevelMap_get, _thrusterStateEffector.BSKLogger_logLevelMap_set)

# Register BSKLogger in _thrusterStateEffector:
_thrusterStateEffector.BSKLogger_swigregister(BSKLogger)
cvar = _thrusterStateEffector.cvar


def _BSKLogger():
    return _thrusterStateEffector._BSKLogger()

def _BSKLogger_d(arg1):
    return _thrusterStateEffector._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _thrusterStateEffector._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _thrusterStateEffector._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _thrusterStateEffector._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _thrusterStateEffector.SysModel_swiginit(self, _thrusterStateEffector.new_SysModel(*args))
    __swig_destroy__ = _thrusterStateEffector.delete_SysModel

    def SelfInit(self):
        return _thrusterStateEffector.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _thrusterStateEffector.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _thrusterStateEffector.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _thrusterStateEffector.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_thrusterStateEffector.SysModel_ModelTag_get, _thrusterStateEffector.SysModel_ModelTag_set)
    CallCounts = property(_thrusterStateEffector.SysModel_CallCounts_get, _thrusterStateEffector.SysModel_CallCounts_set)
    RNGSeed = property(_thrusterStateEffector.SysModel_RNGSeed_get, _thrusterStateEffector.SysModel_RNGSeed_set)
    moduleID = property(_thrusterStateEffector.SysModel_moduleID_get, _thrusterStateEffector.SysModel_moduleID_set)

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


# Register SysModel in _thrusterStateEffector:
_thrusterStateEffector.SysModel_swigregister(SysModel)
class StateData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    state = property(_thrusterStateEffector.StateData_state_get, _thrusterStateEffector.StateData_state_set)
    stateDeriv = property(_thrusterStateEffector.StateData_stateDeriv_get, _thrusterStateEffector.StateData_stateDeriv_set)
    stateName = property(_thrusterStateEffector.StateData_stateName_get, _thrusterStateEffector.StateData_stateName_set)
    stateEnabled = property(_thrusterStateEffector.StateData_stateEnabled_get, _thrusterStateEffector.StateData_stateEnabled_set)
    bskLogger = property(_thrusterStateEffector.StateData_bskLogger_get, _thrusterStateEffector.StateData_bskLogger_set)

    def __init__(self, *args):
        _thrusterStateEffector.StateData_swiginit(self, _thrusterStateEffector.new_StateData(*args))
    __swig_destroy__ = _thrusterStateEffector.delete_StateData

    def setState(self, newState):
        return _thrusterStateEffector.StateData_setState(self, newState)

    def propagateState(self, dt):
        return _thrusterStateEffector.StateData_propagateState(self, dt)

    def setDerivative(self, newDeriv):
        return _thrusterStateEffector.StateData_setDerivative(self, newDeriv)

    def getState(self):
        return _thrusterStateEffector.StateData_getState(self)

    def getStateDeriv(self):
        return _thrusterStateEffector.StateData_getStateDeriv(self)

    def getName(self):
        return _thrusterStateEffector.StateData_getName(self)

    def getRowSize(self):
        return _thrusterStateEffector.StateData_getRowSize(self)

    def getColumnSize(self):
        return _thrusterStateEffector.StateData_getColumnSize(self)

    def isStateActive(self):
        return _thrusterStateEffector.StateData_isStateActive(self)

    def disable(self):
        return _thrusterStateEffector.StateData_disable(self)

    def enable(self):
        return _thrusterStateEffector.StateData_enable(self)

    def scaleState(self, scaleFactor):
        return _thrusterStateEffector.StateData_scaleState(self, scaleFactor)

    def __add__(self, operand):
        return _thrusterStateEffector.StateData___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _thrusterStateEffector.StateData___mul__(self, scaleFactor)

# Register StateData in _thrusterStateEffector:
_thrusterStateEffector.StateData_swigregister(StateData)
class BackSubMatrices(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    matrixA = property(_thrusterStateEffector.BackSubMatrices_matrixA_get, _thrusterStateEffector.BackSubMatrices_matrixA_set)
    matrixB = property(_thrusterStateEffector.BackSubMatrices_matrixB_get, _thrusterStateEffector.BackSubMatrices_matrixB_set)
    matrixC = property(_thrusterStateEffector.BackSubMatrices_matrixC_get, _thrusterStateEffector.BackSubMatrices_matrixC_set)
    matrixD = property(_thrusterStateEffector.BackSubMatrices_matrixD_get, _thrusterStateEffector.BackSubMatrices_matrixD_set)
    vecTrans = property(_thrusterStateEffector.BackSubMatrices_vecTrans_get, _thrusterStateEffector.BackSubMatrices_vecTrans_set)
    vecRot = property(_thrusterStateEffector.BackSubMatrices_vecRot_get, _thrusterStateEffector.BackSubMatrices_vecRot_set)

    def __init__(self):
        _thrusterStateEffector.BackSubMatrices_swiginit(self, _thrusterStateEffector.new_BackSubMatrices())
    __swig_destroy__ = _thrusterStateEffector.delete_BackSubMatrices

# Register BackSubMatrices in _thrusterStateEffector:
_thrusterStateEffector.BackSubMatrices_swigregister(BackSubMatrices)
class EffectorMassProps(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    mEff = property(_thrusterStateEffector.EffectorMassProps_mEff_get, _thrusterStateEffector.EffectorMassProps_mEff_set)
    mEffDot = property(_thrusterStateEffector.EffectorMassProps_mEffDot_get, _thrusterStateEffector.EffectorMassProps_mEffDot_set)
    IEffPntB_B = property(_thrusterStateEffector.EffectorMassProps_IEffPntB_B_get, _thrusterStateEffector.EffectorMassProps_IEffPntB_B_set)
    rEff_CB_B = property(_thrusterStateEffector.EffectorMassProps_rEff_CB_B_get, _thrusterStateEffector.EffectorMassProps_rEff_CB_B_set)
    rEffPrime_CB_B = property(_thrusterStateEffector.EffectorMassProps_rEffPrime_CB_B_get, _thrusterStateEffector.EffectorMassProps_rEffPrime_CB_B_set)
    IEffPrimePntB_B = property(_thrusterStateEffector.EffectorMassProps_IEffPrimePntB_B_get, _thrusterStateEffector.EffectorMassProps_IEffPrimePntB_B_set)

    def __init__(self):
        _thrusterStateEffector.EffectorMassProps_swiginit(self, _thrusterStateEffector.new_EffectorMassProps())
    __swig_destroy__ = _thrusterStateEffector.delete_EffectorMassProps

# Register EffectorMassProps in _thrusterStateEffector:
_thrusterStateEffector.EffectorMassProps_swigregister(EffectorMassProps)
class StateEffector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    nameOfSpacecraftAttachedTo = property(_thrusterStateEffector.StateEffector_nameOfSpacecraftAttachedTo_get, _thrusterStateEffector.StateEffector_nameOfSpacecraftAttachedTo_set)
    parentSpacecraftName = property(_thrusterStateEffector.StateEffector_parentSpacecraftName_get, _thrusterStateEffector.StateEffector_parentSpacecraftName_set)
    effProps = property(_thrusterStateEffector.StateEffector_effProps_get, _thrusterStateEffector.StateEffector_effProps_set)
    stateDerivContribution = property(_thrusterStateEffector.StateEffector_stateDerivContribution_get, _thrusterStateEffector.StateEffector_stateDerivContribution_set)
    forceOnBody_B = property(_thrusterStateEffector.StateEffector_forceOnBody_B_get, _thrusterStateEffector.StateEffector_forceOnBody_B_set)
    torqueOnBodyPntB_B = property(_thrusterStateEffector.StateEffector_torqueOnBodyPntB_B_get, _thrusterStateEffector.StateEffector_torqueOnBodyPntB_B_set)
    torqueOnBodyPntC_B = property(_thrusterStateEffector.StateEffector_torqueOnBodyPntC_B_get, _thrusterStateEffector.StateEffector_torqueOnBodyPntC_B_set)
    r_BP_P = property(_thrusterStateEffector.StateEffector_r_BP_P_get, _thrusterStateEffector.StateEffector_r_BP_P_set)
    dcm_BP = property(_thrusterStateEffector.StateEffector_dcm_BP_get, _thrusterStateEffector.StateEffector_dcm_BP_set)
    bskLogger = property(_thrusterStateEffector.StateEffector_bskLogger_get, _thrusterStateEffector.StateEffector_bskLogger_set)
    __swig_destroy__ = _thrusterStateEffector.delete_StateEffector

    def updateEffectorMassProps(self, integTime):
        return _thrusterStateEffector.StateEffector_updateEffectorMassProps(self, integTime)

    def updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N):
        return _thrusterStateEffector.StateEffector_updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N)

    def updateEnergyMomContributions(self, integTime, rotAngMomPntCContr_B, rotEnergyContr, omega_BN_B):
        return _thrusterStateEffector.StateEffector_updateEnergyMomContributions(self, integTime, rotAngMomPntCContr_B, rotEnergyContr, omega_BN_B)

    def modifyStates(self, integTime):
        return _thrusterStateEffector.StateEffector_modifyStates(self, integTime)

    def calcForceTorqueOnBody(self, integTime, omega_BN_B):
        return _thrusterStateEffector.StateEffector_calcForceTorqueOnBody(self, integTime, omega_BN_B)

    def writeOutputStateMessages(self, integTimeNanos):
        return _thrusterStateEffector.StateEffector_writeOutputStateMessages(self, integTimeNanos)

    def registerStates(self, states):
        return _thrusterStateEffector.StateEffector_registerStates(self, states)

    def linkInStates(self, states):
        return _thrusterStateEffector.StateEffector_linkInStates(self, states)

    def computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN):
        return _thrusterStateEffector.StateEffector_computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN)

    def prependSpacecraftNameToStates(self):
        return _thrusterStateEffector.StateEffector_prependSpacecraftNameToStates(self)

    def receiveMotherSpacecraftData(self, rSC_BP_P, dcmSC_BP):
        return _thrusterStateEffector.StateEffector_receiveMotherSpacecraftData(self, rSC_BP_P, dcmSC_BP)

# Register StateEffector in _thrusterStateEffector:
_thrusterStateEffector.StateEffector_swigregister(StateEffector)
class StateVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    stateMap = property(_thrusterStateEffector.StateVector_stateMap_get, _thrusterStateEffector.StateVector_stateMap_set)

    def __add__(self, operand):
        return _thrusterStateEffector.StateVector___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _thrusterStateEffector.StateVector___mul__(self, scaleFactor)

    def __init__(self):
        _thrusterStateEffector.StateVector_swiginit(self, _thrusterStateEffector.new_StateVector())
    __swig_destroy__ = _thrusterStateEffector.delete_StateVector

# Register StateVector in _thrusterStateEffector:
_thrusterStateEffector.StateVector_swigregister(StateVector)
class DynParamManager(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dynProperties = property(_thrusterStateEffector.DynParamManager_dynProperties_get, _thrusterStateEffector.DynParamManager_dynProperties_set)
    stateContainer = property(_thrusterStateEffector.DynParamManager_stateContainer_get, _thrusterStateEffector.DynParamManager_stateContainer_set)
    bskLogger = property(_thrusterStateEffector.DynParamManager_bskLogger_get, _thrusterStateEffector.DynParamManager_bskLogger_set)

    def __init__(self):
        _thrusterStateEffector.DynParamManager_swiginit(self, _thrusterStateEffector.new_DynParamManager())
    __swig_destroy__ = _thrusterStateEffector.delete_DynParamManager

    def registerState(self, nRow, nCol, stateName):
        return _thrusterStateEffector.DynParamManager_registerState(self, nRow, nCol, stateName)

    def getStateObject(self, stateName):
        return _thrusterStateEffector.DynParamManager_getStateObject(self, stateName)

    def getStateVector(self):
        return _thrusterStateEffector.DynParamManager_getStateVector(self)

    def updateStateVector(self, newState):
        return _thrusterStateEffector.DynParamManager_updateStateVector(self, newState)

    def propagateStateVector(self, dt):
        return _thrusterStateEffector.DynParamManager_propagateStateVector(self, dt)

    def createProperty(self, propName, propValue):
        return _thrusterStateEffector.DynParamManager_createProperty(self, propName, propValue)

    def getPropertyReference(self, propName):
        return _thrusterStateEffector.DynParamManager_getPropertyReference(self, propName)

    def setPropertyValue(self, propName, propValue):
        return _thrusterStateEffector.DynParamManager_setPropertyValue(self, propName, propValue)

# Register DynParamManager in _thrusterStateEffector:
_thrusterStateEffector.DynParamManager_swigregister(DynParamManager)
class THRSimConfig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    thrLoc_B = property(_thrusterStateEffector.THRSimConfig_thrLoc_B_get, _thrusterStateEffector.THRSimConfig_thrLoc_B_set)
    thrDir_B = property(_thrusterStateEffector.THRSimConfig_thrDir_B_get, _thrusterStateEffector.THRSimConfig_thrDir_B_set)
    ThrusterOnRamp = property(_thrusterStateEffector.THRSimConfig_ThrusterOnRamp_get, _thrusterStateEffector.THRSimConfig_ThrusterOnRamp_set)
    ThrusterOffRamp = property(_thrusterStateEffector.THRSimConfig_ThrusterOffRamp_get, _thrusterStateEffector.THRSimConfig_ThrusterOffRamp_set)
    areaNozzle = property(_thrusterStateEffector.THRSimConfig_areaNozzle_get, _thrusterStateEffector.THRSimConfig_areaNozzle_set)
    MaxThrust = property(_thrusterStateEffector.THRSimConfig_MaxThrust_get, _thrusterStateEffector.THRSimConfig_MaxThrust_set)
    steadyIsp = property(_thrusterStateEffector.THRSimConfig_steadyIsp_get, _thrusterStateEffector.THRSimConfig_steadyIsp_set)
    MinOnTime = property(_thrusterStateEffector.THRSimConfig_MinOnTime_get, _thrusterStateEffector.THRSimConfig_MinOnTime_set)
    ThrustOps = property(_thrusterStateEffector.THRSimConfig_ThrustOps_get, _thrusterStateEffector.THRSimConfig_ThrustOps_set)
    thrusterMagDisp = property(_thrusterStateEffector.THRSimConfig_thrusterMagDisp_get, _thrusterStateEffector.THRSimConfig_thrusterMagDisp_set)
    thrusterDirectionDisp = property(_thrusterStateEffector.THRSimConfig_thrusterDirectionDisp_get, _thrusterStateEffector.THRSimConfig_thrusterDirectionDisp_set)
    updateOnly = property(_thrusterStateEffector.THRSimConfig_updateOnly_get, _thrusterStateEffector.THRSimConfig_updateOnly_set)
    label = property(_thrusterStateEffector.THRSimConfig_label_get, _thrusterStateEffector.THRSimConfig_label_set)
    cutoffFrequency = property(_thrusterStateEffector.THRSimConfig_cutoffFrequency_get, _thrusterStateEffector.THRSimConfig_cutoffFrequency_set)
    MaxSwirlTorque = property(_thrusterStateEffector.THRSimConfig_MaxSwirlTorque_get, _thrusterStateEffector.THRSimConfig_MaxSwirlTorque_set)

    def __init__(self):
        _thrusterStateEffector.THRSimConfig_swiginit(self, _thrusterStateEffector.new_THRSimConfig())
    __swig_destroy__ = _thrusterStateEffector.delete_THRSimConfig

# Register THRSimConfig in _thrusterStateEffector:
_thrusterStateEffector.THRSimConfig_swigregister(THRSimConfig)
class ThrusterStateEffector(StateEffector, SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _thrusterStateEffector.ThrusterStateEffector_swiginit(self, _thrusterStateEffector.new_ThrusterStateEffector())
    __swig_destroy__ = _thrusterStateEffector.delete_ThrusterStateEffector

    def Reset(self, CurrentSimNanos):
        return _thrusterStateEffector.ThrusterStateEffector_Reset(self, CurrentSimNanos)

    def ReadInputs(self):
        return _thrusterStateEffector.ThrusterStateEffector_ReadInputs(self)

    def writeOutputStateMessages(self, CurrentClock):
        return _thrusterStateEffector.ThrusterStateEffector_writeOutputStateMessages(self, CurrentClock)

    def registerStates(self, states):
        return _thrusterStateEffector.ThrusterStateEffector_registerStates(self, states)

    def linkInStates(self, states):
        return _thrusterStateEffector.ThrusterStateEffector_linkInStates(self, states)

    def computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN):
        return _thrusterStateEffector.ThrusterStateEffector_computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN)

    def calcForceTorqueOnBody(self, integTime, omega_BN_B):
        return _thrusterStateEffector.ThrusterStateEffector_calcForceTorqueOnBody(self, integTime, omega_BN_B)

    def updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N):
        return _thrusterStateEffector.ThrusterStateEffector_updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N)

    def updateEffectorMassProps(self, integTime):
        return _thrusterStateEffector.ThrusterStateEffector_updateEffectorMassProps(self, integTime)

    def UpdateState(self, CurrentSimNanos):
        return _thrusterStateEffector.ThrusterStateEffector_UpdateState(self, CurrentSimNanos)

    def addThruster(self, *args):
        return _thrusterStateEffector.ThrusterStateEffector_addThruster(self, *args)

    def ConfigureThrustRequests(self):
        return _thrusterStateEffector.ThrusterStateEffector_ConfigureThrustRequests(self)

    def UpdateThrusterProperties(self):
        return _thrusterStateEffector.ThrusterStateEffector_UpdateThrusterProperties(self)
    cmdsInMsg = property(_thrusterStateEffector.ThrusterStateEffector_cmdsInMsg_get, _thrusterStateEffector.ThrusterStateEffector_cmdsInMsg_set)
    thrusterOutMsgs = property(_thrusterStateEffector.ThrusterStateEffector_thrusterOutMsgs_get, _thrusterStateEffector.ThrusterStateEffector_thrusterOutMsgs_set)
    thrusterData = property(_thrusterStateEffector.ThrusterStateEffector_thrusterData_get, _thrusterStateEffector.ThrusterStateEffector_thrusterData_set)
    NewThrustCmds = property(_thrusterStateEffector.ThrusterStateEffector_NewThrustCmds_get, _thrusterStateEffector.ThrusterStateEffector_NewThrustCmds_set)
    kappaInit = property(_thrusterStateEffector.ThrusterStateEffector_kappaInit_get, _thrusterStateEffector.ThrusterStateEffector_kappaInit_set)
    nameOfKappaState = property(_thrusterStateEffector.ThrusterStateEffector_nameOfKappaState_get, _thrusterStateEffector.ThrusterStateEffector_nameOfKappaState_set)
    hubSigma = property(_thrusterStateEffector.ThrusterStateEffector_hubSigma_get, _thrusterStateEffector.ThrusterStateEffector_hubSigma_set)
    hubOmega = property(_thrusterStateEffector.ThrusterStateEffector_hubOmega_get, _thrusterStateEffector.ThrusterStateEffector_hubOmega_set)
    kappaState = property(_thrusterStateEffector.ThrusterStateEffector_kappaState_get, _thrusterStateEffector.ThrusterStateEffector_kappaState_set)
    inertialPositionProperty = property(_thrusterStateEffector.ThrusterStateEffector_inertialPositionProperty_get, _thrusterStateEffector.ThrusterStateEffector_inertialPositionProperty_set)
    bskLogger = property(_thrusterStateEffector.ThrusterStateEffector_bskLogger_get, _thrusterStateEffector.ThrusterStateEffector_bskLogger_set)
    mDotTotal = property(_thrusterStateEffector.ThrusterStateEffector_mDotTotal_get, _thrusterStateEffector.ThrusterStateEffector_mDotTotal_set)

# Register ThrusterStateEffector in _thrusterStateEffector:
_thrusterStateEffector.ThrusterStateEffector_swigregister(ThrusterStateEffector)
class THRArrayOnTimeCmdMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    OnTimeRequest = property(_thrusterStateEffector.THRArrayOnTimeCmdMsgPayload_OnTimeRequest_get, _thrusterStateEffector.THRArrayOnTimeCmdMsgPayload_OnTimeRequest_set)

    def __init__(self):
        _thrusterStateEffector.THRArrayOnTimeCmdMsgPayload_swiginit(self, _thrusterStateEffector.new_THRArrayOnTimeCmdMsgPayload())
    __swig_destroy__ = _thrusterStateEffector.delete_THRArrayOnTimeCmdMsgPayload

# Register THRArrayOnTimeCmdMsgPayload in _thrusterStateEffector:
_thrusterStateEffector.THRArrayOnTimeCmdMsgPayload_swigregister(THRArrayOnTimeCmdMsgPayload)
class THROutputMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    maxThrust = property(_thrusterStateEffector.THROutputMsgPayload_maxThrust_get, _thrusterStateEffector.THROutputMsgPayload_maxThrust_set)
    thrustFactor = property(_thrusterStateEffector.THROutputMsgPayload_thrustFactor_get, _thrusterStateEffector.THROutputMsgPayload_thrustFactor_set)
    thrustForce = property(_thrusterStateEffector.THROutputMsgPayload_thrustForce_get, _thrusterStateEffector.THROutputMsgPayload_thrustForce_set)
    thrustForce_B = property(_thrusterStateEffector.THROutputMsgPayload_thrustForce_B_get, _thrusterStateEffector.THROutputMsgPayload_thrustForce_B_set)
    thrustTorquePntB_B = property(_thrusterStateEffector.THROutputMsgPayload_thrustTorquePntB_B_get, _thrusterStateEffector.THROutputMsgPayload_thrustTorquePntB_B_set)
    thrusterLocation = property(_thrusterStateEffector.THROutputMsgPayload_thrusterLocation_get, _thrusterStateEffector.THROutputMsgPayload_thrusterLocation_set)
    thrusterDirection = property(_thrusterStateEffector.THROutputMsgPayload_thrusterDirection_get, _thrusterStateEffector.THROutputMsgPayload_thrusterDirection_set)

    def __init__(self):
        _thrusterStateEffector.THROutputMsgPayload_swiginit(self, _thrusterStateEffector.new_THROutputMsgPayload())
    __swig_destroy__ = _thrusterStateEffector.delete_THROutputMsgPayload

# Register THROutputMsgPayload in _thrusterStateEffector:
_thrusterStateEffector.THROutputMsgPayload_swigregister(THROutputMsgPayload)
class SCStatesMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    r_BN_N = property(_thrusterStateEffector.SCStatesMsgPayload_r_BN_N_get, _thrusterStateEffector.SCStatesMsgPayload_r_BN_N_set)
    v_BN_N = property(_thrusterStateEffector.SCStatesMsgPayload_v_BN_N_get, _thrusterStateEffector.SCStatesMsgPayload_v_BN_N_set)
    r_CN_N = property(_thrusterStateEffector.SCStatesMsgPayload_r_CN_N_get, _thrusterStateEffector.SCStatesMsgPayload_r_CN_N_set)
    v_CN_N = property(_thrusterStateEffector.SCStatesMsgPayload_v_CN_N_get, _thrusterStateEffector.SCStatesMsgPayload_v_CN_N_set)
    sigma_BN = property(_thrusterStateEffector.SCStatesMsgPayload_sigma_BN_get, _thrusterStateEffector.SCStatesMsgPayload_sigma_BN_set)
    omega_BN_B = property(_thrusterStateEffector.SCStatesMsgPayload_omega_BN_B_get, _thrusterStateEffector.SCStatesMsgPayload_omega_BN_B_set)
    omegaDot_BN_B = property(_thrusterStateEffector.SCStatesMsgPayload_omegaDot_BN_B_get, _thrusterStateEffector.SCStatesMsgPayload_omegaDot_BN_B_set)
    TotalAccumDVBdy = property(_thrusterStateEffector.SCStatesMsgPayload_TotalAccumDVBdy_get, _thrusterStateEffector.SCStatesMsgPayload_TotalAccumDVBdy_set)
    TotalAccumDV_BN_B = property(_thrusterStateEffector.SCStatesMsgPayload_TotalAccumDV_BN_B_get, _thrusterStateEffector.SCStatesMsgPayload_TotalAccumDV_BN_B_set)
    nonConservativeAccelpntB_B = property(_thrusterStateEffector.SCStatesMsgPayload_nonConservativeAccelpntB_B_get, _thrusterStateEffector.SCStatesMsgPayload_nonConservativeAccelpntB_B_set)
    MRPSwitchCount = property(_thrusterStateEffector.SCStatesMsgPayload_MRPSwitchCount_get, _thrusterStateEffector.SCStatesMsgPayload_MRPSwitchCount_set)

    def __init__(self):
        _thrusterStateEffector.SCStatesMsgPayload_swiginit(self, _thrusterStateEffector.new_SCStatesMsgPayload())
    __swig_destroy__ = _thrusterStateEffector.delete_SCStatesMsgPayload

# Register SCStatesMsgPayload in _thrusterStateEffector:
_thrusterStateEffector.SCStatesMsgPayload_swigregister(SCStatesMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


