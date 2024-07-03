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
    from . import _gravityEffector
else:
    import _gravityEffector

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

from Basilisk.simulation.pointMassGravityModel import PointMassGravityModel
from Basilisk.simulation.masconGravityModel import MasconGravityModel
from Basilisk.simulation.pinnGravityModel import PINNGravityModel
from Basilisk.simulation.pinn2GravityModel import PINN2GravityModel
from Basilisk.simulation.polyhedralGravityModel import PolyhedralGravityModel
from Basilisk.simulation.sphericalHarmonicsGravityModel import SphericalHarmonicsGravityModel

from Basilisk.utilities import deprecated

Mascon = MasconGravityModel
PINN = PINNGravityModel
Polyhedral = PolyhedralGravityModel
SphericalHarmonics = SphericalHarmonicsGravityModel

from typing import Optional, Union



def new_doubleArray(nelements):
    return _gravityEffector.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _gravityEffector.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _gravityEffector.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _gravityEffector.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _gravityEffector.new_longArray(nelements)

def delete_longArray(ary):
    return _gravityEffector.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _gravityEffector.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _gravityEffector.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _gravityEffector.new_intArray(nelements)

def delete_intArray(ary):
    return _gravityEffector.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _gravityEffector.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _gravityEffector.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _gravityEffector.new_shortArray(nelements)

def delete_shortArray(ary):
    return _gravityEffector.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _gravityEffector.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _gravityEffector.shortArray_setitem(ary, index, value)


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


SHARED_PTR_DISOWN = _gravityEffector.SHARED_PTR_DISOWN
class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _gravityEffector.delete_SwigPyIterator

    def value(self):
        return _gravityEffector.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _gravityEffector.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _gravityEffector.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _gravityEffector.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _gravityEffector.SwigPyIterator_equal(self, x)

    def copy(self):
        return _gravityEffector.SwigPyIterator_copy(self)

    def next(self):
        return _gravityEffector.SwigPyIterator_next(self)

    def __next__(self):
        return _gravityEffector.SwigPyIterator___next__(self)

    def previous(self):
        return _gravityEffector.SwigPyIterator_previous(self)

    def advance(self, n):
        return _gravityEffector.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _gravityEffector.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _gravityEffector.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _gravityEffector.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _gravityEffector.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _gravityEffector.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _gravityEffector.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _gravityEffector:
_gravityEffector.SwigPyIterator_swigregister(SwigPyIterator)
class GravBodyVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _gravityEffector.GravBodyVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _gravityEffector.GravBodyVector___nonzero__(self)

    def __bool__(self):
        return _gravityEffector.GravBodyVector___bool__(self)

    def __len__(self):
        return _gravityEffector.GravBodyVector___len__(self)

    def __getslice__(self, i, j):
        return _gravityEffector.GravBodyVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _gravityEffector.GravBodyVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _gravityEffector.GravBodyVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _gravityEffector.GravBodyVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _gravityEffector.GravBodyVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _gravityEffector.GravBodyVector___setitem__(self, *args)

    def pop(self):
        return _gravityEffector.GravBodyVector_pop(self)

    def append(self, x):
        return _gravityEffector.GravBodyVector_append(self, x)

    def empty(self):
        return _gravityEffector.GravBodyVector_empty(self)

    def size(self):
        return _gravityEffector.GravBodyVector_size(self)

    def swap(self, v):
        return _gravityEffector.GravBodyVector_swap(self, v)

    def begin(self):
        return _gravityEffector.GravBodyVector_begin(self)

    def end(self):
        return _gravityEffector.GravBodyVector_end(self)

    def rbegin(self):
        return _gravityEffector.GravBodyVector_rbegin(self)

    def rend(self):
        return _gravityEffector.GravBodyVector_rend(self)

    def clear(self):
        return _gravityEffector.GravBodyVector_clear(self)

    def get_allocator(self):
        return _gravityEffector.GravBodyVector_get_allocator(self)

    def pop_back(self):
        return _gravityEffector.GravBodyVector_pop_back(self)

    def erase(self, *args):
        return _gravityEffector.GravBodyVector_erase(self, *args)

    def __init__(self, *args):
        _gravityEffector.GravBodyVector_swiginit(self, _gravityEffector.new_GravBodyVector(*args))

    def push_back(self, x):
        return _gravityEffector.GravBodyVector_push_back(self, x)

    def front(self):
        return _gravityEffector.GravBodyVector_front(self)

    def back(self):
        return _gravityEffector.GravBodyVector_back(self)

    def assign(self, n, x):
        return _gravityEffector.GravBodyVector_assign(self, n, x)

    def resize(self, *args):
        return _gravityEffector.GravBodyVector_resize(self, *args)

    def insert(self, *args):
        return _gravityEffector.GravBodyVector_insert(self, *args)

    def reserve(self, n):
        return _gravityEffector.GravBodyVector_reserve(self, n)

    def capacity(self):
        return _gravityEffector.GravBodyVector_capacity(self)
    __swig_destroy__ = _gravityEffector.delete_GravBodyVector

# Register GravBodyVector in _gravityEffector:
_gravityEffector.GravBodyVector_swigregister(GravBodyVector)
import Basilisk.simulation.gravityModel
class DynamicEffector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _gravityEffector.delete_DynamicEffector

    def computeStateContribution(self, integTime):
        return _gravityEffector.DynamicEffector_computeStateContribution(self, integTime)

    def linkInStates(self, states):
        return _gravityEffector.DynamicEffector_linkInStates(self, states)

    def computeForceTorque(self, integTime, timeStep):
        return _gravityEffector.DynamicEffector_computeForceTorque(self, integTime, timeStep)
    stateDerivContribution = property(_gravityEffector.DynamicEffector_stateDerivContribution_get, _gravityEffector.DynamicEffector_stateDerivContribution_set)
    forceExternal_N = property(_gravityEffector.DynamicEffector_forceExternal_N_get, _gravityEffector.DynamicEffector_forceExternal_N_set)
    forceExternal_B = property(_gravityEffector.DynamicEffector_forceExternal_B_get, _gravityEffector.DynamicEffector_forceExternal_B_set)
    torqueExternalPntB_B = property(_gravityEffector.DynamicEffector_torqueExternalPntB_B_get, _gravityEffector.DynamicEffector_torqueExternalPntB_B_set)
    bskLogger = property(_gravityEffector.DynamicEffector_bskLogger_get, _gravityEffector.DynamicEffector_bskLogger_set)

# Register DynamicEffector in _gravityEffector:
_gravityEffector.DynamicEffector_swigregister(DynamicEffector)
class StateData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    state = property(_gravityEffector.StateData_state_get, _gravityEffector.StateData_state_set)
    stateDeriv = property(_gravityEffector.StateData_stateDeriv_get, _gravityEffector.StateData_stateDeriv_set)
    stateName = property(_gravityEffector.StateData_stateName_get, _gravityEffector.StateData_stateName_set)
    stateEnabled = property(_gravityEffector.StateData_stateEnabled_get, _gravityEffector.StateData_stateEnabled_set)
    bskLogger = property(_gravityEffector.StateData_bskLogger_get, _gravityEffector.StateData_bskLogger_set)

    def __init__(self, *args):
        _gravityEffector.StateData_swiginit(self, _gravityEffector.new_StateData(*args))
    __swig_destroy__ = _gravityEffector.delete_StateData

    def setState(self, newState):
        return _gravityEffector.StateData_setState(self, newState)

    def propagateState(self, dt):
        return _gravityEffector.StateData_propagateState(self, dt)

    def setDerivative(self, newDeriv):
        return _gravityEffector.StateData_setDerivative(self, newDeriv)

    def getState(self):
        return _gravityEffector.StateData_getState(self)

    def getStateDeriv(self):
        return _gravityEffector.StateData_getStateDeriv(self)

    def getName(self):
        return _gravityEffector.StateData_getName(self)

    def getRowSize(self):
        return _gravityEffector.StateData_getRowSize(self)

    def getColumnSize(self):
        return _gravityEffector.StateData_getColumnSize(self)

    def isStateActive(self):
        return _gravityEffector.StateData_isStateActive(self)

    def disable(self):
        return _gravityEffector.StateData_disable(self)

    def enable(self):
        return _gravityEffector.StateData_enable(self)

    def scaleState(self, scaleFactor):
        return _gravityEffector.StateData_scaleState(self, scaleFactor)

    def __add__(self, operand):
        return _gravityEffector.StateData___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _gravityEffector.StateData___mul__(self, scaleFactor)

# Register StateData in _gravityEffector:
_gravityEffector.StateData_swigregister(StateData)

from Basilisk.architecture.swig_common_model import *

MAX_LOGGING_LENGTH = _gravityEffector.MAX_LOGGING_LENGTH
BSK_DEBUG = _gravityEffector.BSK_DEBUG
BSK_INFORMATION = _gravityEffector.BSK_INFORMATION
BSK_WARNING = _gravityEffector.BSK_WARNING
BSK_ERROR = _gravityEffector.BSK_ERROR
BSK_SILENT = _gravityEffector.BSK_SILENT

def printDefaultLogLevel():
    return _gravityEffector.printDefaultLogLevel()

def setDefaultLogLevel(logLevel):
    return _gravityEffector.setDefaultLogLevel(logLevel)

def getDefaultLogLevel():
    return _gravityEffector.getDefaultLogLevel()
class BSKLogger(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _gravityEffector.BSKLogger_swiginit(self, _gravityEffector.new_BSKLogger(*args))
    __swig_destroy__ = _gravityEffector.delete_BSKLogger

    def setLogLevel(self, logLevel):
        return _gravityEffector.BSKLogger_setLogLevel(self, logLevel)

    def printLogLevel(self):
        return _gravityEffector.BSKLogger_printLogLevel(self)

    def getLogLevel(self):
        return _gravityEffector.BSKLogger_getLogLevel(self)

    def bskLog(self, targetLevel, info):
        return _gravityEffector.BSKLogger_bskLog(self, targetLevel, info)
    logLevelMap = property(_gravityEffector.BSKLogger_logLevelMap_get, _gravityEffector.BSKLogger_logLevelMap_set)

# Register BSKLogger in _gravityEffector:
_gravityEffector.BSKLogger_swigregister(BSKLogger)
cvar = _gravityEffector.cvar


def _BSKLogger():
    return _gravityEffector._BSKLogger()

def _BSKLogger_d(arg1):
    return _gravityEffector._BSKLogger_d(arg1)

def _printLogLevel(arg1):
    return _gravityEffector._printLogLevel(arg1)

def _setLogLevel(arg1, arg2):
    return _gravityEffector._setLogLevel(arg1, arg2)

def _bskLog(arg1, arg2, arg3):
    return _gravityEffector._bskLog(arg1, arg2, arg3)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _gravityEffector.SysModel_swiginit(self, _gravityEffector.new_SysModel(*args))
    __swig_destroy__ = _gravityEffector.delete_SysModel

    def SelfInit(self):
        return _gravityEffector.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _gravityEffector.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _gravityEffector.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _gravityEffector.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_gravityEffector.SysModel_ModelTag_get, _gravityEffector.SysModel_ModelTag_set)
    CallCounts = property(_gravityEffector.SysModel_CallCounts_get, _gravityEffector.SysModel_CallCounts_set)
    RNGSeed = property(_gravityEffector.SysModel_RNGSeed_get, _gravityEffector.SysModel_RNGSeed_set)
    moduleID = property(_gravityEffector.SysModel_moduleID_get, _gravityEffector.SysModel_moduleID_set)

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


# Register SysModel in _gravityEffector:
_gravityEffector.SysModel_swigregister(SysModel)
class GravBodyData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _gravityEffector.GravBodyData_swiginit(self, _gravityEffector.new_GravBodyData())

        object.__setattr__(self, "_pyGravityModel", None) # Enable setting _pyGravityModel
        self.gravityModel = PointMassGravityModel() # Re-set gravityModel to populate the _pyGravityModel



    def initBody(self, moduleID):
        return _gravityEffector.GravBodyData_initBody(self, moduleID)

    def computeGravityInertial(self, r_I, simTimeNanos):
        return _gravityEffector.GravBodyData_computeGravityInertial(self, r_I, simTimeNanos)

    def loadEphemeris(self):
        return _gravityEffector.GravBodyData_loadEphemeris(self)

    def registerProperties(self, statesIn):
        return _gravityEffector.GravBodyData_registerProperties(self, statesIn)
    isCentralBody = property(_gravityEffector.GravBodyData_isCentralBody_get, _gravityEffector.GravBodyData_isCentralBody_set)
    gravityModel = property(_gravityEffector.GravBodyData_gravityModel_get, _gravityEffector.GravBodyData_gravityModel_set)
    mu = property(_gravityEffector.GravBodyData_mu_get, _gravityEffector.GravBodyData_mu_set)
    radEquator = property(_gravityEffector.GravBodyData_radEquator_get, _gravityEffector.GravBodyData_radEquator_set)
    radiusRatio = property(_gravityEffector.GravBodyData_radiusRatio_get, _gravityEffector.GravBodyData_radiusRatio_set)
    planetName = property(_gravityEffector.GravBodyData_planetName_get, _gravityEffector.GravBodyData_planetName_set)
    displayName = property(_gravityEffector.GravBodyData_displayName_get, _gravityEffector.GravBodyData_displayName_set)
    modelDictionaryKey = property(_gravityEffector.GravBodyData_modelDictionaryKey_get, _gravityEffector.GravBodyData_modelDictionaryKey_set)
    planetBodyInMsg = property(_gravityEffector.GravBodyData_planetBodyInMsg_get, _gravityEffector.GravBodyData_planetBodyInMsg_set)
    localPlanet = property(_gravityEffector.GravBodyData_localPlanet_get, _gravityEffector.GravBodyData_localPlanet_set)
    bskLogger = property(_gravityEffector.GravBodyData_bskLogger_get, _gravityEffector.GravBodyData_bskLogger_set)


    """
    If we were to call GravBodyData::gravityModel we would obtain a pointer to the
    parent object GravityModel, as this is what is stored in the GravBodyData C++
    class (the concrete type is "lost"). To overcome this, we store a copy of the
    set object in _pyGravityModel and use the gravityModel property to keep the
    Python and C++ objects synchronized. _pyGravityModel does retain the concrete
    type (PointMassGravityModel, SphericalHarmonicsGravityModel...)
    """
    _gravityModel = gravityModel
    @property
    def gravityModel(self):
        return self._pyGravityModel

    @gravityModel.setter
    def gravityModel(self, value):
        self._gravityModel = value
        self._pyGravityModel = value

    @property
    def useSphericalHarmParams(self):
        return isinstance(self.gravityModel, SphericalHarmonicsGravityModel)

    @useSphericalHarmParams.setter
    def useSphericalHarmParams(self, value: bool):
        deprecated.deprecationWarn(
            "GravBodyData.useSphericalHarmParams setter",
            "2024/09/07",
            "Using 'useSphericalHarmParams = True/False' to turn on/off the spherical harmonics"
            " is deprecated. Prefer the following syntax:\n"
            "\tplanet.useSphericalHarmonicsGravityModel('GGM2BData.txt', 100)\n"
            "Over:\n"
            "\tplanet.useSphericalHarmParams = True\n"
            "\tsimIncludeGravBody.loadGravFromFile('GGM2BData.txt', planet.spherHarm, 100)"
        )
        if self.useSphericalHarmParams and not value:
            self.gravityModel = PointMassGravityModel()
        elif not self.useSphericalHarmParams and value:
            self.gravityModel = SphericalHarmonicsGravityModel()

    @property
    def usePolyhedral(self):
        return isinstance(self.gravityModel, PolyhedralGravityModel)

    @usePolyhedral.setter
    def usePolyhedral(self, value: bool):
        deprecated.deprecationWarn(
            "GravBodyData.usePolyhedral setter",
            "2024/09/07",
            "Using 'usePolyhedral = True/False' to turn on/off the polyhedral model"
            " is deprecated. Prefer the following syntax:\n"
            "\tplanet.usePolyhedralGravityModel('eros.txt')\n"
            "Over:\n"
            "\tplanet.usePolyhedral = True\n"
            "\tsimIncludeGravBody.loadPolyFromFile('eros.txt', planet.poly)"
        )
        if self.usePolyhedral and not value:
            self.gravityModel = PointMassGravityModel()
        elif not self.usePolyhedral and value:
            self.gravityModel = PolyhedralGravityModel()

    @property
    def mascon(self) -> MasconGravityModel:
        return self.gravityModel

    @mascon.setter
    def mascon(self, value: MasconGravityModel):
        self.gravityModel = value

    @property
    def pinn(self) -> PINNGravityModel:
        return self.gravityModel

    @pinn.setter
    def pinn(self, value: PINNGravityModel):
        self.gravityModel = value

    @property
    def pinn2(self) -> PINN2GravityModel:
        return self.gravityModel

    @pinn.setter
    def pinn2(self, value: PINN2GravityModel):
        self.gravityModel = value

    @property
    def spherHarm(self) -> SphericalHarmonicsGravityModel:
        if self.useSphericalHarmParams:
            return self.gravityModel
        else:
            raise ValueError("GravBodyData is not using spherical harmonics as a gravity model. "
                "Call 'useSphericalHarmonicsGravityModel(...)' or set 'useSphericalHarmParams' to 'True' before retrieving 'spherHarm'.")

    @spherHarm.setter
    def spherHarm(self, value: SphericalHarmonicsGravityModel):
        self.gravityModel = value

    @property
    def poly(self) -> PolyhedralGravityModel:
        if self.usePolyhedral:
            return self.gravityModel
        else:
            raise ValueError("GravBodyData is not using the polyhedral gravity model. "
                "Call 'usePolyhedralGravityModel(...)' or set 'usePolyhedral' to 'True' before retrieving 'poly'.")

    @poly.setter
    def poly(self, value: PolyhedralGravityModel):
        self.gravityModel = value

    def usePointMassGravityModel(self):
        self.gravityModel = PointMassGravityModel()

    def useMasconGravityModel(self):
        self.gravityModel = MasconGravityModel()

    def usePINNGravityModel(self):
        self.gravityModel = PINNGravityModel()

    def usePINN2GravityModel(self):
        self.gravityModel = PINN2GravityModel()

    def useSphericalHarmonicsGravityModel(self, file: str, maxDeg: int):
        """Makes the GravBodyData use Spherical Harmonics as its gravity model.

        Args:
            file (str): The file that contains the spherical harmonics data in the
                JPL format.
            maxDeg (int): The maximum degree to use in the spherical harmonics.
        """
        self.gravityModel = SphericalHarmonicsGravityModel().loadFromFile(file, maxDeg)

    def usePolyhedralGravityModel(self, file: str):
        """Makes the GravBodyData use the Polyhedral gravity model.

        Args:
            file (str): The file that contains the vertices and facet
                data for the polyhedral.
        """
        self.gravityModel = PolyhedralGravityModel().loadFromFile(file)


    __swig_destroy__ = _gravityEffector.delete_GravBodyData

# Register GravBodyData in _gravityEffector:
_gravityEffector.GravBodyData_swigregister(GravBodyData)
class GravityEffector(SysModel):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def Reset(self, currentSimNanos):
        return _gravityEffector.GravityEffector_Reset(self, currentSimNanos)

    def UpdateState(self, currentSimNanos):
        return _gravityEffector.GravityEffector_UpdateState(self, currentSimNanos)

    def linkInStates(self, statesIn):
        return _gravityEffector.GravityEffector_linkInStates(self, statesIn)

    def registerProperties(self, statesIn):
        return _gravityEffector.GravityEffector_registerProperties(self, statesIn)

    def computeGravityField(self, r_cF_N, rDot_cF_N):
        return _gravityEffector.GravityEffector_computeGravityField(self, r_cF_N, rDot_cF_N)

    def setGravBodies(self, gravBodies):
        return _gravityEffector.GravityEffector_setGravBodies(self, gravBodies)

    def addGravBody(self, gravBody):
        return _gravityEffector.GravityEffector_addGravBody(self, gravBody)
    gravBodies = property(_gravityEffector.GravityEffector_gravBodies_get, _gravityEffector.GravityEffector_gravBodies_set)
    centralBody = property(_gravityEffector.GravityEffector_centralBody_get)
    vehicleGravityPropName = property(_gravityEffector.GravityEffector_vehicleGravityPropName_get)
    systemTimeCorrPropName = property(_gravityEffector.GravityEffector_systemTimeCorrPropName_get)
    inertialPositionPropName = property(_gravityEffector.GravityEffector_inertialPositionPropName_get)
    inertialVelocityPropName = property(_gravityEffector.GravityEffector_inertialVelocityPropName_get)
    nameOfSpacecraftAttachedTo = property(_gravityEffector.GravityEffector_nameOfSpacecraftAttachedTo_get)
    centralBodyOutMsg = property(_gravityEffector.GravityEffector_centralBodyOutMsg_get, _gravityEffector.GravityEffector_centralBodyOutMsg_set)
    bskLogger = property(_gravityEffector.GravityEffector_bskLogger_get, _gravityEffector.GravityEffector_bskLogger_set)

    def __init__(self):
        _gravityEffector.GravityEffector_swiginit(self, _gravityEffector.new_GravityEffector())
    __swig_destroy__ = _gravityEffector.delete_GravityEffector

# Register GravityEffector in _gravityEffector:
_gravityEffector.GravityEffector_swigregister(GravityEffector)
MAX_BODY_NAME_LENGTH = _gravityEffector.MAX_BODY_NAME_LENGTH
class SpicePlanetStateMsgPayload(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    J2000Current = property(_gravityEffector.SpicePlanetStateMsgPayload_J2000Current_get, _gravityEffector.SpicePlanetStateMsgPayload_J2000Current_set)
    PositionVector = property(_gravityEffector.SpicePlanetStateMsgPayload_PositionVector_get, _gravityEffector.SpicePlanetStateMsgPayload_PositionVector_set)
    VelocityVector = property(_gravityEffector.SpicePlanetStateMsgPayload_VelocityVector_get, _gravityEffector.SpicePlanetStateMsgPayload_VelocityVector_set)
    J20002Pfix = property(_gravityEffector.SpicePlanetStateMsgPayload_J20002Pfix_get, _gravityEffector.SpicePlanetStateMsgPayload_J20002Pfix_set)
    J20002Pfix_dot = property(_gravityEffector.SpicePlanetStateMsgPayload_J20002Pfix_dot_get, _gravityEffector.SpicePlanetStateMsgPayload_J20002Pfix_dot_set)
    computeOrient = property(_gravityEffector.SpicePlanetStateMsgPayload_computeOrient_get, _gravityEffector.SpicePlanetStateMsgPayload_computeOrient_set)
    PlanetName = property(_gravityEffector.SpicePlanetStateMsgPayload_PlanetName_get, _gravityEffector.SpicePlanetStateMsgPayload_PlanetName_set)

    def __init__(self):
        _gravityEffector.SpicePlanetStateMsgPayload_swiginit(self, _gravityEffector.new_SpicePlanetStateMsgPayload())
    __swig_destroy__ = _gravityEffector.delete_SpicePlanetStateMsgPayload

# Register SpicePlanetStateMsgPayload in _gravityEffector:
_gravityEffector.SpicePlanetStateMsgPayload_swigregister(SpicePlanetStateMsgPayload)

import sys
protectAllClasses(sys.modules[__name__])        

#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
import csv

from Basilisk import __path__

def loadGravFromFile(
        fileName: str, 
        spherHarm: "SphericalHarmonicsGravityModel", 
        maxDeg: int = 2
    ):

    [clmList, slmList, mu, radEquator] = loadGravFromFileToList(fileName, maxDeg=2)

    spherHarm.muBody = mu
    spherHarm.radEquator = radEquator
    spherHarm.cBar = clmList
    spherHarm.sBar = slmList
    spherHarm.maxDeg = maxDeg

def loadGravFromFileToList(fileName: str, maxDeg: int = 2):
    with open(fileName, 'r') as csvfile:
        gravReader = csv.reader(csvfile, delimiter=',')
        firstRow = next(gravReader)
        clmList = []
        slmList = []

        try:
            radEquator = float(firstRow[0])
            mu = float(firstRow[1])
# firstRow[2] is uncertainty in mu, not needed for Basilisk
            maxDegreeFile = int(firstRow[3])
            maxOrderFile = int(firstRow[4])
            coefficientsNormalized = int(firstRow[5]) == 1
            refLong = float(firstRow[6])
            refLat = float(firstRow[7])
        except Exception as ex:
            raise ValueError("File is not in the expected JPL format for "
                             "spherical Harmonics", ex)

        if maxDegreeFile < maxDeg or maxOrderFile < maxDeg:
            raise ValueError(f"Requested using Spherical Harmonics of degree {maxDeg}"
                             f", but file '{fileName}' has maximum degree/order of"
                             f"{min(maxDegreeFile, maxOrderFile)}")

        if not coefficientsNormalized:
            raise ValueError("Coefficients in given file are not normalized. This is "
                            "not currently supported in Basilisk.")

        if refLong != 0 or refLat != 0:
            raise ValueError("Coefficients in given file use a reference longitude"
                             " or latitude that is not zero. This is not currently "
                             "supported in Basilisk.")

        clmRow = []
        slmRow = []
        currDeg = 0
        for gravRow in gravReader:
            while int(gravRow[0]) > currDeg:
                if (len(clmRow) < currDeg + 1):
                    clmRow.extend([0.0] * (currDeg + 1 - len(clmRow)))
                    slmRow.extend([0.0] * (currDeg + 1 - len(slmRow)))
                clmList.append(clmRow)
                slmList.append(slmRow)
                clmRow = []
                slmRow = []
                currDeg += 1
            clmRow.append(float(gravRow[2]))
            slmRow.append(float(gravRow[3]))

        return [clmList, slmList, mu, radEquator]


def loadPolyFromFile(fileName: str, poly: "PolyhedralGravityModel"):
    [vertList, faceList, _, _] = loadPolyFromFileToList(fileName)
    poly.xyzVertex = vertList
    poly.orderFacet = faceList

def loadPolyFromFileToList(fileName: str):
    with open(fileName) as polyFile:
        if fileName.endswith('.tab'):
            try:
                nVertex, nFacet = [int(x) for x in next(polyFile).split()] # read first line
                fileType = 'gaskell'
            except:
                polyFile.seek(0)
                fileType = 'pds3'

            if fileType == 'gaskell':
                vertList = []
                faceList = []

                contLines = 0
                for line in polyFile:
                    arrtemp = []

                    for x in line.split():
                        arrtemp.append(float(x))

                    if contLines < nVertex:
                        vertList.append([float(arrtemp[1]*1e3),float(arrtemp[2]*1e3),float(arrtemp[3]*1e3)])
                    else:
                        faceList.append([int(arrtemp[1]),int(arrtemp[2]),int(arrtemp[3])])

                    contLines += 1
            elif fileType == 'pds3':
                nVertex = 0
                nFacet = 0
                vertList = []
                faceList = []
                for line in polyFile:
                    arrtemp = line.split()
                    if arrtemp:
                        if arrtemp[0] == 'v':
                            nVertex += 1
                            vertList.append([float(arrtemp[1])*1e3, float(arrtemp[2])*1e3, float(arrtemp[3])*1e3])
                        elif arrtemp[0] == 'f':
                            nFacet += 1
                            faceList.append([int(arrtemp[1])+1, int(arrtemp[2])+1, int(arrtemp[3])+1])
        elif fileName.endswith('.obj'):
            nVertex = 0
            nFacet = 0
            vertList = []
            faceList = []
            for line in polyFile:
                arrtemp = line.split()
                if arrtemp:
                    if arrtemp[0] == 'v':
                        nVertex += 1
                        vertList.append([float(arrtemp[1])*1e3, float(arrtemp[2])*1e3, float(arrtemp[3])*1e3])
                    elif arrtemp[0] == 'f':
                        nFacet += 1
                        faceList.append([int(arrtemp[1]), int(arrtemp[2]), int(arrtemp[3])])
        elif fileName.endswith('.txt'):
            nVertex, nFacet = [int(x) for x in next(polyFile).split()] # read first line
            vertList = []
            faceList = []

            contLines = 0
            for line in polyFile:
                arrtemp = []

                for x in line.split():
                    arrtemp.append(float(x))

                if contLines < nVertex:
                    vertList.append([float(arrtemp[0]*1e3),float(arrtemp[1]*1e3),float(arrtemp[2]*1e3)])
                else:
                    faceList.append([int(arrtemp[0]),int(arrtemp[1]),int(arrtemp[2])])

                contLines += 1
        else:
            raise ValueError("Unrecognized file extension. Valid extensions are "
                             "'.tab', '.obj', and '.txt'")

        return [vertList, faceList, nVertex, nFacet]


