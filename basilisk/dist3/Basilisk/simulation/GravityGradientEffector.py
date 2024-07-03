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
    from . import _GravityGradientEffector
else:
    import _GravityGradientEffector

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
    return _GravityGradientEffector.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _GravityGradientEffector.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _GravityGradientEffector.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _GravityGradientEffector.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _GravityGradientEffector.new_longArray(nelements)

def delete_longArray(ary):
    return _GravityGradientEffector.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _GravityGradientEffector.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _GravityGradientEffector.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _GravityGradientEffector.new_intArray(nelements)

def delete_intArray(ary):
    return _GravityGradientEffector.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _GravityGradientEffector.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _GravityGradientEffector.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _GravityGradientEffector.new_shortArray(nelements)

def delete_shortArray(ary):
    return _GravityGradientEffector.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _GravityGradientEffector.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _GravityGradientEffector.shortArray_setitem(ary, index, value)


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

MAX_LOGGING_LENGTH = _GravityGradientEffector.MAX_LOGGING_LENGTH
BSK_DEBUG = _GravityGradientEffector.BSK_DEBUG
BSK_INFORMATION = _GravityGradientEffector.BSK_INFORMATION
BSK_WARNING = _GravityGradientEffector.BSK_WARNING
BSK_ERROR = _GravityGradientEffector.BSK_ERROR
BSK_SILENT = _GravityGradientEffector.BSK_SILENT

def printDefaultLogLevel():
    return _GravityGradientEffector.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _GravityGradientEffector.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _GravityGradientEffector.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _GravityGradientEffector.BSKLogger_swiginit(self, _GravityGradientEffector.new_BSKLogger(*args))
    __swig_destroy__ = _GravityGradientEffector.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _GravityGradientEffector.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _GravityGradientEffector.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _GravityGradientEffector.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _GravityGradientEffector.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_GravityGradientEffector.BSKLogger_logLevelMap_get, _GravityGradientEffector.BSKLogger_logLevelMap_set)

# Register BSKLogger in _GravityGradientEffector:
_GravityGradientEffector.BSKLogger_swigregister(BSKLogger)
cvar = _GravityGradientEffector.cvar


def _BSKLogger():
    return _GravityGradientEffector._BSKLogger()

def _BSKLogger_d(arg1):
    return _GravityGradientEffector._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _GravityGradientEffector._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _GravityGradientEffector._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _GravityGradientEffector._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _GravityGradientEffector.SysModel_swiginit(self, _GravityGradientEffector.new_SysModel(*args))
    __swig_destroy__ = _GravityGradientEffector.delete_SysModel

    def SelfInit(self):
        return _GravityGradientEffector.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _GravityGradientEffector.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _GravityGradientEffector.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _GravityGradientEffector.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_GravityGradientEffector.SysModel_ModelTag_get, _GravityGradientEffector.SysModel_ModelTag_set)
    CallCounts = property(_GravityGradientEffector.SysModel_CallCounts_get, _GravityGradientEffector.SysModel_CallCounts_set)
    RNGSeed = property(_GravityGradientEffector.SysModel_RNGSeed_get, _GravityGradientEffector.SysModel_RNGSeed_set)
    moduleID = property(_GravityGradientEffector.SysModel_moduleID_get, _GravityGradientEffector.SysModel_moduleID_set)

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


# Register SysModel in _GravityGradientEffector:
_GravityGradientEffector.SysModel_swigregister(SysModel)
class StateData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    state = property(_GravityGradientEffector.StateData_state_get, _GravityGradientEffector.StateData_state_set)
    stateDeriv = property(_GravityGradientEffector.StateData_stateDeriv_get, _GravityGradientEffector.StateData_stateDeriv_set)
    stateName = property(_GravityGradientEffector.StateData_stateName_get, _GravityGradientEffector.StateData_stateName_set)
    stateEnabled = property(_GravityGradientEffector.StateData_stateEnabled_get, _GravityGradientEffector.StateData_stateEnabled_set)
    bskLogger = property(_GravityGradientEffector.StateData_bskLogger_get, _GravityGradientEffector.StateData_bskLogger_set)

    def __init__(self, *args):
        _GravityGradientEffector.StateData_swiginit(self, _GravityGradientEffector.new_StateData(*args))
    __swig_destroy__ = _GravityGradientEffector.delete_StateData

    def setState(self, newState):
        return _GravityGradientEffector.StateData_setState(self, newState)

    def propagateState(self, dt):
        return _GravityGradientEffector.StateData_propagateState(self, dt)

    def setDerivative(self, newDeriv):
        return _GravityGradientEffector.StateData_setDerivative(self, newDeriv)

    def getState(self):
        return _GravityGradientEffector.StateData_getState(self)

    def getStateDeriv(self):
        return _GravityGradientEffector.StateData_getStateDeriv(self)

    def getName(self):
        return _GravityGradientEffector.StateData_getName(self)

    def getRowSize(self):
        return _GravityGradientEffector.StateData_getRowSize(self)

    def getColumnSize(self):
        return _GravityGradientEffector.StateData_getColumnSize(self)

    def isStateActive(self):
        return _GravityGradientEffector.StateData_isStateActive(self)

    def disable(self):
        return _GravityGradientEffector.StateData_disable(self)

    def enable(self):
        return _GravityGradientEffector.StateData_enable(self)

    def scaleState(self, scaleFactor):
        return _GravityGradientEffector.StateData_scaleState(self, scaleFactor)

    def __add__(self, operand):
        return _GravityGradientEffector.StateData___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _GravityGradientEffector.StateData___mul__(self, scaleFactor)

# Register StateData in _GravityGradientEffector:
_GravityGradientEffector.StateData_swigregister(StateData)
class DynamicEffector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _GravityGradientEffector.delete_DynamicEffector

    def computeStateContribution(self, integTime):
        return _GravityGradientEffector.DynamicEffector_computeStateContribution(self, integTime)

    def linkInStates(self, states):
        return _GravityGradientEffector.DynamicEffector_linkInStates(self, states)

    def computeForceTorque(self, integTime, timeStep):
        return _GravityGradientEffector.DynamicEffector_computeForceTorque(self, integTime, timeStep)
    stateDerivContribution = property(_GravityGradientEffector.DynamicEffector_stateDerivContribution_get, _GravityGradientEffector.DynamicEffector_stateDerivContribution_set)
    forceExternal_N = property(_GravityGradientEffector.DynamicEffector_forceExternal_N_get, _GravityGradientEffector.DynamicEffector_forceExternal_N_set)
    forceExternal_B = property(_GravityGradientEffector.DynamicEffector_forceExternal_B_get, _GravityGradientEffector.DynamicEffector_forceExternal_B_set)
    torqueExternalPntB_B = property(_GravityGradientEffector.DynamicEffector_torqueExternalPntB_B_get, _GravityGradientEffector.DynamicEffector_torqueExternalPntB_B_set)
    bskLogger = property(_GravityGradientEffector.DynamicEffector_bskLogger_get, _GravityGradientEffector.DynamicEffector_bskLogger_set)

# Register DynamicEffector in _GravityGradientEffector:
_GravityGradientEffector.DynamicEffector_swigregister(DynamicEffector)
class StateVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    stateMap = property(_GravityGradientEffector.StateVector_stateMap_get, _GravityGradientEffector.StateVector_stateMap_set)

    def __add__(self, operand):
        return _GravityGradientEffector.StateVector___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _GravityGradientEffector.StateVector___mul__(self, scaleFactor)

    def __init__(self):
        _GravityGradientEffector.StateVector_swiginit(self, _GravityGradientEffector.new_StateVector())
    __swig_destroy__ = _GravityGradientEffector.delete_StateVector

# Register StateVector in _GravityGradientEffector:
_GravityGradientEffector.StateVector_swigregister(StateVector)
class DynParamManager(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dynProperties = property(_GravityGradientEffector.DynParamManager_dynProperties_get, _GravityGradientEffector.DynParamManager_dynProperties_set)
    stateContainer = property(_GravityGradientEffector.DynParamManager_stateContainer_get, _GravityGradientEffector.DynParamManager_stateContainer_set)
    bskLogger = property(_GravityGradientEffector.DynParamManager_bskLogger_get, _GravityGradientEffector.DynParamManager_bskLogger_set)

    def __init__(self):
        _GravityGradientEffector.DynParamManager_swiginit(self, _GravityGradientEffector.new_DynParamManager())
    __swig_destroy__ = _GravityGradientEffector.delete_DynParamManager

    def registerState(self, nRow, nCol, stateName):
        return _GravityGradientEffector.DynParamManager_registerState(self, nRow, nCol, stateName)

    def getStateObject(self, stateName):
        return _GravityGradientEffector.DynParamManager_getStateObject(self, stateName)

    def getStateVector(self):
        return _GravityGradientEffector.DynParamManager_getStateVector(self)

    def updateStateVector(self, newState):
        return _GravityGradientEffector.DynParamManager_updateStateVector(self, newState)

    def propagateStateVector(self, dt):
        return _GravityGradientEffector.DynParamManager_propagateStateVector(self, dt)

    def createProperty(self, propName, propValue):
        return _GravityGradientEffector.DynParamManager_createProperty(self, propName, propValue)

    def getPropertyReference(self, propName):
        return _GravityGradientEffector.DynParamManager_getPropertyReference(self, propName)

    def setPropertyValue(self, propName, propValue):
        return _GravityGradientEffector.DynParamManager_setPropertyValue(self, propName, propValue)

# Register DynParamManager in _GravityGradientEffector:
_GravityGradientEffector.DynParamManager_swigregister(DynParamManager)
class GravityGradientEffector(SysModel, DynamicEffector):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _GravityGradientEffector.GravityGradientEffector_swiginit(self, _GravityGradientEffector.new_GravityGradientEffector())
    __swig_destroy__ = _GravityGradientEffector.delete_GravityGradientEffector

    def linkInStates(self, states):
        return _GravityGradientEffector.GravityGradientEffector_linkInStates(self, states)

    def computeForceTorque(self, integTime, timeStep):
        return _GravityGradientEffector.GravityGradientEffector_computeForceTorque(self, integTime, timeStep)

    def Reset(self, CurrentSimNanos):
        return _GravityGradientEffector.GravityGradientEffector_Reset(self, CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos):
        return _GravityGradientEffector.GravityGradientEffector_UpdateState(self, CurrentSimNanos)

    def WriteOutputMessages(self, CurrentClock):
        return _GravityGradientEffector.GravityGradientEffector_WriteOutputMessages(self, CurrentClock)

    def addPlanetName(self, planetName):
        return _GravityGradientEffector.GravityGradientEffector_addPlanetName(self, planetName)
    gravityGradientOutMsg = property(_GravityGradientEffector.GravityGradientEffector_gravityGradientOutMsg_get, _GravityGradientEffector.GravityGradientEffector_gravityGradientOutMsg_set)
    hubSigma = property(_GravityGradientEffector.GravityGradientEffector_hubSigma_get, _GravityGradientEffector.GravityGradientEffector_hubSigma_set)
    r_BN_N = property(_GravityGradientEffector.GravityGradientEffector_r_BN_N_get, _GravityGradientEffector.GravityGradientEffector_r_BN_N_set)
    ISCPntB_B = property(_GravityGradientEffector.GravityGradientEffector_ISCPntB_B_get, _GravityGradientEffector.GravityGradientEffector_ISCPntB_B_set)
    c_B = property(_GravityGradientEffector.GravityGradientEffector_c_B_get, _GravityGradientEffector.GravityGradientEffector_c_B_set)
    m_SC = property(_GravityGradientEffector.GravityGradientEffector_m_SC_get, _GravityGradientEffector.GravityGradientEffector_m_SC_set)
    r_PN_N = property(_GravityGradientEffector.GravityGradientEffector_r_PN_N_get, _GravityGradientEffector.GravityGradientEffector_r_PN_N_set)
    muPlanet = property(_GravityGradientEffector.GravityGradientEffector_muPlanet_get, _GravityGradientEffector.GravityGradientEffector_muPlanet_set)
    bskLogger = property(_GravityGradientEffector.GravityGradientEffector_bskLogger_get, _GravityGradientEffector.GravityGradientEffector_bskLogger_set)

# Register GravityGradientEffector in _GravityGradientEffector:
_GravityGradientEffector.GravityGradientEffector_swigregister(GravityGradientEffector)
class GravityGradientMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    gravityGradientTorque_B = property(_GravityGradientEffector.GravityGradientMsgPayload_gravityGradientTorque_B_get, _GravityGradientEffector.GravityGradientMsgPayload_gravityGradientTorque_B_set)

    def __init__(self):
        _GravityGradientEffector.GravityGradientMsgPayload_swiginit(self, _GravityGradientEffector.new_GravityGradientMsgPayload())
    __swig_destroy__ = _GravityGradientEffector.delete_GravityGradientMsgPayload

# Register GravityGradientMsgPayload in _GravityGradientEffector:
_GravityGradientEffector.GravityGradientMsgPayload_swigregister(GravityGradientMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])


