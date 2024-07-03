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
    from . import _spinningBodyTwoDOFStateEffector
else:
    import _spinningBodyTwoDOFStateEffector

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

class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_SwigPyIterator

    def value(self):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_equal(self, x)

    def copy(self):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_copy(self)

    def next(self):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_next(self)

    def __next__(self):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___next__(self)

    def previous(self):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_previous(self)

    def advance(self, n):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _spinningBodyTwoDOFStateEffector.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.SwigPyIterator_swigregister(SwigPyIterator)

def new_doubleArray(nelements):
    return _spinningBodyTwoDOFStateEffector.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _spinningBodyTwoDOFStateEffector.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _spinningBodyTwoDOFStateEffector.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _spinningBodyTwoDOFStateEffector.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _spinningBodyTwoDOFStateEffector.new_longArray(nelements)

def delete_longArray(ary):
    return _spinningBodyTwoDOFStateEffector.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _spinningBodyTwoDOFStateEffector.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _spinningBodyTwoDOFStateEffector.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _spinningBodyTwoDOFStateEffector.new_intArray(nelements)

def delete_intArray(ary):
    return _spinningBodyTwoDOFStateEffector.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _spinningBodyTwoDOFStateEffector.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _spinningBodyTwoDOFStateEffector.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _spinningBodyTwoDOFStateEffector.new_shortArray(nelements)

def delete_shortArray(ary):
    return _spinningBodyTwoDOFStateEffector.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _spinningBodyTwoDOFStateEffector.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _spinningBodyTwoDOFStateEffector.shortArray_setitem(ary, index, value)


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



from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _spinningBodyTwoDOFStateEffector.MAX_LOGGING_LENGTH
BSK_DEBUG = _spinningBodyTwoDOFStateEffector.BSK_DEBUG
BSK_INFORMATION = _spinningBodyTwoDOFStateEffector.BSK_INFORMATION
BSK_WARNING = _spinningBodyTwoDOFStateEffector.BSK_WARNING
BSK_ERROR = _spinningBodyTwoDOFStateEffector.BSK_ERROR
BSK_SILENT = _spinningBodyTwoDOFStateEffector.BSK_SILENT

def printDefaultLogLevel():
    return _spinningBodyTwoDOFStateEffector.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _spinningBodyTwoDOFStateEffector.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _spinningBodyTwoDOFStateEffector.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _spinningBodyTwoDOFStateEffector.BSKLogger_swiginit(self, _spinningBodyTwoDOFStateEffector.new_BSKLogger(*args))
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _spinningBodyTwoDOFStateEffector.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _spinningBodyTwoDOFStateEffector.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _spinningBodyTwoDOFStateEffector.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _spinningBodyTwoDOFStateEffector.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_spinningBodyTwoDOFStateEffector.BSKLogger_logLevelMap_get, _spinningBodyTwoDOFStateEffector.BSKLogger_logLevelMap_set)

# Register BSKLogger in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.BSKLogger_swigregister(BSKLogger)
cvar = _spinningBodyTwoDOFStateEffector.cvar


def _BSKLogger():
    return _spinningBodyTwoDOFStateEffector._BSKLogger()

def _BSKLogger_d(arg1):
    return _spinningBodyTwoDOFStateEffector._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _spinningBodyTwoDOFStateEffector._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _spinningBodyTwoDOFStateEffector._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _spinningBodyTwoDOFStateEffector._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _spinningBodyTwoDOFStateEffector.SysModel_swiginit(self, _spinningBodyTwoDOFStateEffector.new_SysModel(*args))
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_SysModel

    def SelfInit(self):
        return _spinningBodyTwoDOFStateEffector.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _spinningBodyTwoDOFStateEffector.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _spinningBodyTwoDOFStateEffector.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _spinningBodyTwoDOFStateEffector.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_spinningBodyTwoDOFStateEffector.SysModel_ModelTag_get, _spinningBodyTwoDOFStateEffector.SysModel_ModelTag_set)
    CallCounts = property(_spinningBodyTwoDOFStateEffector.SysModel_CallCounts_get, _spinningBodyTwoDOFStateEffector.SysModel_CallCounts_set)
    RNGSeed = property(_spinningBodyTwoDOFStateEffector.SysModel_RNGSeed_get, _spinningBodyTwoDOFStateEffector.SysModel_RNGSeed_set)
    moduleID = property(_spinningBodyTwoDOFStateEffector.SysModel_moduleID_get, _spinningBodyTwoDOFStateEffector.SysModel_moduleID_set)

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


# Register SysModel in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.SysModel_swigregister(SysModel)
class StateData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    state = property(_spinningBodyTwoDOFStateEffector.StateData_state_get, _spinningBodyTwoDOFStateEffector.StateData_state_set)
    stateDeriv = property(_spinningBodyTwoDOFStateEffector.StateData_stateDeriv_get, _spinningBodyTwoDOFStateEffector.StateData_stateDeriv_set)
    stateName = property(_spinningBodyTwoDOFStateEffector.StateData_stateName_get, _spinningBodyTwoDOFStateEffector.StateData_stateName_set)
    stateEnabled = property(_spinningBodyTwoDOFStateEffector.StateData_stateEnabled_get, _spinningBodyTwoDOFStateEffector.StateData_stateEnabled_set)
    bskLogger = property(_spinningBodyTwoDOFStateEffector.StateData_bskLogger_get, _spinningBodyTwoDOFStateEffector.StateData_bskLogger_set)

    def __init__(self, *args):
        _spinningBodyTwoDOFStateEffector.StateData_swiginit(self, _spinningBodyTwoDOFStateEffector.new_StateData(*args))
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_StateData

    def setState(self, newState):
        return _spinningBodyTwoDOFStateEffector.StateData_setState(self, newState)

    def propagateState(self, dt):
        return _spinningBodyTwoDOFStateEffector.StateData_propagateState(self, dt)

    def setDerivative(self, newDeriv):
        return _spinningBodyTwoDOFStateEffector.StateData_setDerivative(self, newDeriv)

    def getState(self):
        return _spinningBodyTwoDOFStateEffector.StateData_getState(self)

    def getStateDeriv(self):
        return _spinningBodyTwoDOFStateEffector.StateData_getStateDeriv(self)

    def getName(self):
        return _spinningBodyTwoDOFStateEffector.StateData_getName(self)

    def getRowSize(self):
        return _spinningBodyTwoDOFStateEffector.StateData_getRowSize(self)

    def getColumnSize(self):
        return _spinningBodyTwoDOFStateEffector.StateData_getColumnSize(self)

    def isStateActive(self):
        return _spinningBodyTwoDOFStateEffector.StateData_isStateActive(self)

    def disable(self):
        return _spinningBodyTwoDOFStateEffector.StateData_disable(self)

    def enable(self):
        return _spinningBodyTwoDOFStateEffector.StateData_enable(self)

    def scaleState(self, scaleFactor):
        return _spinningBodyTwoDOFStateEffector.StateData_scaleState(self, scaleFactor)

    def __add__(self, operand):
        return _spinningBodyTwoDOFStateEffector.StateData___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _spinningBodyTwoDOFStateEffector.StateData___mul__(self, scaleFactor)

# Register StateData in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.StateData_swigregister(StateData)
class BackSubMatrices(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    matrixA = property(_spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixA_get, _spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixA_set)
    matrixB = property(_spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixB_get, _spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixB_set)
    matrixC = property(_spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixC_get, _spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixC_set)
    matrixD = property(_spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixD_get, _spinningBodyTwoDOFStateEffector.BackSubMatrices_matrixD_set)
    vecTrans = property(_spinningBodyTwoDOFStateEffector.BackSubMatrices_vecTrans_get, _spinningBodyTwoDOFStateEffector.BackSubMatrices_vecTrans_set)
    vecRot = property(_spinningBodyTwoDOFStateEffector.BackSubMatrices_vecRot_get, _spinningBodyTwoDOFStateEffector.BackSubMatrices_vecRot_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.BackSubMatrices_swiginit(self, _spinningBodyTwoDOFStateEffector.new_BackSubMatrices())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_BackSubMatrices

# Register BackSubMatrices in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.BackSubMatrices_swigregister(BackSubMatrices)
class EffectorMassProps(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    mEff = property(_spinningBodyTwoDOFStateEffector.EffectorMassProps_mEff_get, _spinningBodyTwoDOFStateEffector.EffectorMassProps_mEff_set)
    mEffDot = property(_spinningBodyTwoDOFStateEffector.EffectorMassProps_mEffDot_get, _spinningBodyTwoDOFStateEffector.EffectorMassProps_mEffDot_set)
    IEffPntB_B = property(_spinningBodyTwoDOFStateEffector.EffectorMassProps_IEffPntB_B_get, _spinningBodyTwoDOFStateEffector.EffectorMassProps_IEffPntB_B_set)
    rEff_CB_B = property(_spinningBodyTwoDOFStateEffector.EffectorMassProps_rEff_CB_B_get, _spinningBodyTwoDOFStateEffector.EffectorMassProps_rEff_CB_B_set)
    rEffPrime_CB_B = property(_spinningBodyTwoDOFStateEffector.EffectorMassProps_rEffPrime_CB_B_get, _spinningBodyTwoDOFStateEffector.EffectorMassProps_rEffPrime_CB_B_set)
    IEffPrimePntB_B = property(_spinningBodyTwoDOFStateEffector.EffectorMassProps_IEffPrimePntB_B_get, _spinningBodyTwoDOFStateEffector.EffectorMassProps_IEffPrimePntB_B_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.EffectorMassProps_swiginit(self, _spinningBodyTwoDOFStateEffector.new_EffectorMassProps())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_EffectorMassProps

# Register EffectorMassProps in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.EffectorMassProps_swigregister(EffectorMassProps)
class StateEffector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    nameOfSpacecraftAttachedTo = property(_spinningBodyTwoDOFStateEffector.StateEffector_nameOfSpacecraftAttachedTo_get, _spinningBodyTwoDOFStateEffector.StateEffector_nameOfSpacecraftAttachedTo_set)
    parentSpacecraftName = property(_spinningBodyTwoDOFStateEffector.StateEffector_parentSpacecraftName_get, _spinningBodyTwoDOFStateEffector.StateEffector_parentSpacecraftName_set)
    effProps = property(_spinningBodyTwoDOFStateEffector.StateEffector_effProps_get, _spinningBodyTwoDOFStateEffector.StateEffector_effProps_set)
    stateDerivContribution = property(_spinningBodyTwoDOFStateEffector.StateEffector_stateDerivContribution_get, _spinningBodyTwoDOFStateEffector.StateEffector_stateDerivContribution_set)
    forceOnBody_B = property(_spinningBodyTwoDOFStateEffector.StateEffector_forceOnBody_B_get, _spinningBodyTwoDOFStateEffector.StateEffector_forceOnBody_B_set)
    torqueOnBodyPntB_B = property(_spinningBodyTwoDOFStateEffector.StateEffector_torqueOnBodyPntB_B_get, _spinningBodyTwoDOFStateEffector.StateEffector_torqueOnBodyPntB_B_set)
    torqueOnBodyPntC_B = property(_spinningBodyTwoDOFStateEffector.StateEffector_torqueOnBodyPntC_B_get, _spinningBodyTwoDOFStateEffector.StateEffector_torqueOnBodyPntC_B_set)
    r_BP_P = property(_spinningBodyTwoDOFStateEffector.StateEffector_r_BP_P_get, _spinningBodyTwoDOFStateEffector.StateEffector_r_BP_P_set)
    dcm_BP = property(_spinningBodyTwoDOFStateEffector.StateEffector_dcm_BP_get, _spinningBodyTwoDOFStateEffector.StateEffector_dcm_BP_set)
    bskLogger = property(_spinningBodyTwoDOFStateEffector.StateEffector_bskLogger_get, _spinningBodyTwoDOFStateEffector.StateEffector_bskLogger_set)
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_StateEffector

    def updateEffectorMassProps(self, integTime):
        return _spinningBodyTwoDOFStateEffector.StateEffector_updateEffectorMassProps(self, integTime)

    def updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N):
        return _spinningBodyTwoDOFStateEffector.StateEffector_updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N)

    def updateEnergyMomContributions(self, integTime, rotAngMomPntCContr_B, rotEnergyContr, omega_BN_B):
        return _spinningBodyTwoDOFStateEffector.StateEffector_updateEnergyMomContributions(self, integTime, rotAngMomPntCContr_B, rotEnergyContr, omega_BN_B)

    def modifyStates(self, integTime):
        return _spinningBodyTwoDOFStateEffector.StateEffector_modifyStates(self, integTime)

    def calcForceTorqueOnBody(self, integTime, omega_BN_B):
        return _spinningBodyTwoDOFStateEffector.StateEffector_calcForceTorqueOnBody(self, integTime, omega_BN_B)

    def writeOutputStateMessages(self, integTimeNanos):
        return _spinningBodyTwoDOFStateEffector.StateEffector_writeOutputStateMessages(self, integTimeNanos)

    def registerStates(self, states):
        return _spinningBodyTwoDOFStateEffector.StateEffector_registerStates(self, states)

    def linkInStates(self, states):
        return _spinningBodyTwoDOFStateEffector.StateEffector_linkInStates(self, states)

    def computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN):
        return _spinningBodyTwoDOFStateEffector.StateEffector_computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN)

    def prependSpacecraftNameToStates(self):
        return _spinningBodyTwoDOFStateEffector.StateEffector_prependSpacecraftNameToStates(self)

    def receiveMotherSpacecraftData(self, rSC_BP_P, dcmSC_BP):
        return _spinningBodyTwoDOFStateEffector.StateEffector_receiveMotherSpacecraftData(self, rSC_BP_P, dcmSC_BP)

# Register StateEffector in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.StateEffector_swigregister(StateEffector)
class StateVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    stateMap = property(_spinningBodyTwoDOFStateEffector.StateVector_stateMap_get, _spinningBodyTwoDOFStateEffector.StateVector_stateMap_set)

    def __add__(self, operand):
        return _spinningBodyTwoDOFStateEffector.StateVector___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _spinningBodyTwoDOFStateEffector.StateVector___mul__(self, scaleFactor)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.StateVector_swiginit(self, _spinningBodyTwoDOFStateEffector.new_StateVector())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_StateVector

# Register StateVector in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.StateVector_swigregister(StateVector)
class DynParamManager(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dynProperties = property(_spinningBodyTwoDOFStateEffector.DynParamManager_dynProperties_get, _spinningBodyTwoDOFStateEffector.DynParamManager_dynProperties_set)
    stateContainer = property(_spinningBodyTwoDOFStateEffector.DynParamManager_stateContainer_get, _spinningBodyTwoDOFStateEffector.DynParamManager_stateContainer_set)
    bskLogger = property(_spinningBodyTwoDOFStateEffector.DynParamManager_bskLogger_get, _spinningBodyTwoDOFStateEffector.DynParamManager_bskLogger_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.DynParamManager_swiginit(self, _spinningBodyTwoDOFStateEffector.new_DynParamManager())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_DynParamManager

    def registerState(self, nRow, nCol, stateName):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_registerState(self, nRow, nCol, stateName)

    def getStateObject(self, stateName):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_getStateObject(self, stateName)

    def getStateVector(self):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_getStateVector(self)

    def updateStateVector(self, newState):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_updateStateVector(self, newState)

    def propagateStateVector(self, dt):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_propagateStateVector(self, dt)

    def createProperty(self, propName, propValue):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_createProperty(self, propName, propValue)

    def getPropertyReference(self, propName):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_getPropertyReference(self, propName)

    def setPropertyValue(self, propName, propValue):
        return _spinningBodyTwoDOFStateEffector.DynParamManager_setPropertyValue(self, propName, propValue)

# Register DynParamManager in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.DynParamManager_swigregister(DynParamManager)
class SpinningBodyTwoDOFStateEffector(StateEffector, SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    mass1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_mass1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_mass1_set)
    mass2 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_mass2_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_mass2_set)
    k1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_k1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_k1_set)
    k2 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_k2_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_k2_set)
    c1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_c1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_c1_set)
    c2 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_c2_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_c2_set)
    theta1Init = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta1Init_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta1Init_set)
    theta1DotInit = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta1DotInit_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta1DotInit_set)
    theta2Init = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta2Init_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta2Init_set)
    theta2DotInit = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta2DotInit_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_theta2DotInit_set)
    nameOfTheta1State = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta1State_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta1State_set)
    nameOfTheta1DotState = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta1DotState_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta1DotState_set)
    nameOfTheta2State = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta2State_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta2State_set)
    nameOfTheta2DotState = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta2DotState_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_nameOfTheta2DotState_set)
    r_S1B_B = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_S1B_B_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_S1B_B_set)
    r_S2S1_S1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_S2S1_S1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_S2S1_S1_set)
    r_Sc1S1_S1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_Sc1S1_S1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_Sc1S1_S1_set)
    r_Sc2S2_S2 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_Sc2S2_S2_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_r_Sc2S2_S2_set)
    s1Hat_S1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_s1Hat_S1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_s1Hat_S1_set)
    s2Hat_S2 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_s2Hat_S2_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_s2Hat_S2_set)
    IS1PntSc1_S1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_IS1PntSc1_S1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_IS1PntSc1_S1_set)
    IS2PntSc2_S2 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_IS2PntSc2_S2_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_IS2PntSc2_S2_set)
    dcm_S10B = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_dcm_S10B_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_dcm_S10B_set)
    dcm_S20S1 = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_dcm_S20S1_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_dcm_S20S1_set)
    spinningBodyOutMsgs = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_spinningBodyOutMsgs_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_spinningBodyOutMsgs_set)
    spinningBodyConfigLogOutMsgs = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_spinningBodyConfigLogOutMsgs_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_spinningBodyConfigLogOutMsgs_set)
    motorTorqueInMsg = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_motorTorqueInMsg_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_motorTorqueInMsg_set)
    motorLockInMsg = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_motorLockInMsg_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_motorLockInMsg_set)
    spinningBodyRefInMsgs = property(_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_spinningBodyRefInMsgs_get, _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_spinningBodyRefInMsgs_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_swiginit(self, _spinningBodyTwoDOFStateEffector.new_SpinningBodyTwoDOFStateEffector())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_SpinningBodyTwoDOFStateEffector

    def Reset(self, CurrentClock):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_Reset(self, CurrentClock)

    def writeOutputStateMessages(self, CurrentClock):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_writeOutputStateMessages(self, CurrentClock)

    def UpdateState(self, CurrentSimNanos):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_UpdateState(self, CurrentSimNanos)

    def registerStates(self, statesIn):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_registerStates(self, statesIn)

    def linkInStates(self, states):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_linkInStates(self, states)

    def updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_updateContributions(self, integTime, backSubContr, sigma_BN, omega_BN_B, g_N)

    def computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_computeDerivatives(self, integTime, rDDot_BN_N, omegaDot_BN_B, sigma_BN)

    def updateEffectorMassProps(self, integTime):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_updateEffectorMassProps(self, integTime)

    def updateEnergyMomContributions(self, integTime, rotAngMomPntCContr_B, rotEnergyContr, omega_BN_B):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_updateEnergyMomContributions(self, integTime, rotAngMomPntCContr_B, rotEnergyContr, omega_BN_B)

    def prependSpacecraftNameToStates(self):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_prependSpacecraftNameToStates(self)

    def computeSpinningBodyInertialStates(self):
        return _spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_computeSpinningBodyInertialStates(self)

# Register SpinningBodyTwoDOFStateEffector in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.SpinningBodyTwoDOFStateEffector_swigregister(SpinningBodyTwoDOFStateEffector)
class SCStatesMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    r_BN_N = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_r_BN_N_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_r_BN_N_set)
    v_BN_N = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_v_BN_N_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_v_BN_N_set)
    r_CN_N = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_r_CN_N_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_r_CN_N_set)
    v_CN_N = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_v_CN_N_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_v_CN_N_set)
    sigma_BN = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_sigma_BN_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_sigma_BN_set)
    omega_BN_B = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_omega_BN_B_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_omega_BN_B_set)
    omegaDot_BN_B = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_omegaDot_BN_B_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_omegaDot_BN_B_set)
    TotalAccumDVBdy = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_TotalAccumDVBdy_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_TotalAccumDVBdy_set)
    TotalAccumDV_BN_B = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_TotalAccumDV_BN_B_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_TotalAccumDV_BN_B_set)
    nonConservativeAccelpntB_B = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_nonConservativeAccelpntB_B_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_nonConservativeAccelpntB_B_set)
    MRPSwitchCount = property(_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_MRPSwitchCount_get, _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_MRPSwitchCount_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_swiginit(self, _spinningBodyTwoDOFStateEffector.new_SCStatesMsgPayload())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_SCStatesMsgPayload

# Register SCStatesMsgPayload in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.SCStatesMsgPayload_swigregister(SCStatesMsgPayload)
class ArrayMotorTorqueMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    motorTorque = property(_spinningBodyTwoDOFStateEffector.ArrayMotorTorqueMsgPayload_motorTorque_get, _spinningBodyTwoDOFStateEffector.ArrayMotorTorqueMsgPayload_motorTorque_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.ArrayMotorTorqueMsgPayload_swiginit(self, _spinningBodyTwoDOFStateEffector.new_ArrayMotorTorqueMsgPayload())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_ArrayMotorTorqueMsgPayload

# Register ArrayMotorTorqueMsgPayload in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.ArrayMotorTorqueMsgPayload_swigregister(ArrayMotorTorqueMsgPayload)
class ArrayEffectorLockMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    effectorLockFlag = property(_spinningBodyTwoDOFStateEffector.ArrayEffectorLockMsgPayload_effectorLockFlag_get, _spinningBodyTwoDOFStateEffector.ArrayEffectorLockMsgPayload_effectorLockFlag_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.ArrayEffectorLockMsgPayload_swiginit(self, _spinningBodyTwoDOFStateEffector.new_ArrayEffectorLockMsgPayload())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_ArrayEffectorLockMsgPayload

# Register ArrayEffectorLockMsgPayload in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.ArrayEffectorLockMsgPayload_swigregister(ArrayEffectorLockMsgPayload)
class HingedRigidBodyMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    theta = property(_spinningBodyTwoDOFStateEffector.HingedRigidBodyMsgPayload_theta_get, _spinningBodyTwoDOFStateEffector.HingedRigidBodyMsgPayload_theta_set)
    thetaDot = property(_spinningBodyTwoDOFStateEffector.HingedRigidBodyMsgPayload_thetaDot_get, _spinningBodyTwoDOFStateEffector.HingedRigidBodyMsgPayload_thetaDot_set)

    def __init__(self):
        _spinningBodyTwoDOFStateEffector.HingedRigidBodyMsgPayload_swiginit(self, _spinningBodyTwoDOFStateEffector.new_HingedRigidBodyMsgPayload())
    __swig_destroy__ = _spinningBodyTwoDOFStateEffector.delete_HingedRigidBodyMsgPayload

# Register HingedRigidBodyMsgPayload in _spinningBodyTwoDOFStateEffector:
_spinningBodyTwoDOFStateEffector.HingedRigidBodyMsgPayload_swigregister(HingedRigidBodyMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


