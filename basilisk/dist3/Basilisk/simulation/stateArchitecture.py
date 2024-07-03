# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _stateArchitecture
else:
    import _stateArchitecture

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
    return _stateArchitecture.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _stateArchitecture.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _stateArchitecture.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _stateArchitecture.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _stateArchitecture.new_longArray(nelements)

def delete_longArray(ary):
    return _stateArchitecture.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _stateArchitecture.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _stateArchitecture.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _stateArchitecture.new_intArray(nelements)

def delete_intArray(ary):
    return _stateArchitecture.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _stateArchitecture.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _stateArchitecture.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _stateArchitecture.new_shortArray(nelements)

def delete_shortArray(ary):
    return _stateArchitecture.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _stateArchitecture.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _stateArchitecture.shortArray_setitem(ary, index, value)


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


class StateVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    stateMap = property(_stateArchitecture.StateVector_stateMap_get, _stateArchitecture.StateVector_stateMap_set)

    def __add__(self, operand):
        return _stateArchitecture.StateVector___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _stateArchitecture.StateVector___mul__(self, scaleFactor)

    def __init__(self):
        _stateArchitecture.StateVector_swiginit(self, _stateArchitecture.new_StateVector())
    __swig_destroy__ = _stateArchitecture.delete_StateVector

# Register StateVector in _stateArchitecture:
_stateArchitecture.StateVector_swigregister(StateVector)
class DynParamManager(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    dynProperties = property(_stateArchitecture.DynParamManager_dynProperties_get, _stateArchitecture.DynParamManager_dynProperties_set)
    stateContainer = property(_stateArchitecture.DynParamManager_stateContainer_get, _stateArchitecture.DynParamManager_stateContainer_set)
    bskLogger = property(_stateArchitecture.DynParamManager_bskLogger_get, _stateArchitecture.DynParamManager_bskLogger_set)

    def __init__(self):
        _stateArchitecture.DynParamManager_swiginit(self, _stateArchitecture.new_DynParamManager())
    __swig_destroy__ = _stateArchitecture.delete_DynParamManager

    def registerState(self, nRow, nCol, stateName):
        return _stateArchitecture.DynParamManager_registerState(self, nRow, nCol, stateName)

    def getStateObject(self, stateName):
        return _stateArchitecture.DynParamManager_getStateObject(self, stateName)

    def getStateVector(self):
        return _stateArchitecture.DynParamManager_getStateVector(self)

    def updateStateVector(self, newState):
        return _stateArchitecture.DynParamManager_updateStateVector(self, newState)

    def propagateStateVector(self, dt):
        return _stateArchitecture.DynParamManager_propagateStateVector(self, dt)

    def createProperty(self, propName, propValue):
        return _stateArchitecture.DynParamManager_createProperty(self, propName, propValue)

    def getPropertyReference(self, propName):
        return _stateArchitecture.DynParamManager_getPropertyReference(self, propName)

    def setPropertyValue(self, propName, propValue):
        return _stateArchitecture.DynParamManager_setPropertyValue(self, propName, propValue)

# Register DynParamManager in _stateArchitecture:
_stateArchitecture.DynParamManager_swigregister(DynParamManager)
class StateData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    state = property(_stateArchitecture.StateData_state_get, _stateArchitecture.StateData_state_set)
    stateDeriv = property(_stateArchitecture.StateData_stateDeriv_get, _stateArchitecture.StateData_stateDeriv_set)
    stateName = property(_stateArchitecture.StateData_stateName_get, _stateArchitecture.StateData_stateName_set)
    stateEnabled = property(_stateArchitecture.StateData_stateEnabled_get, _stateArchitecture.StateData_stateEnabled_set)
    bskLogger = property(_stateArchitecture.StateData_bskLogger_get, _stateArchitecture.StateData_bskLogger_set)

    def __init__(self, *args):
        _stateArchitecture.StateData_swiginit(self, _stateArchitecture.new_StateData(*args))
    __swig_destroy__ = _stateArchitecture.delete_StateData

    def setState(self, newState):
        return _stateArchitecture.StateData_setState(self, newState)

    def propagateState(self, dt):
        return _stateArchitecture.StateData_propagateState(self, dt)

    def setDerivative(self, newDeriv):
        return _stateArchitecture.StateData_setDerivative(self, newDeriv)

    def getState(self):
        return _stateArchitecture.StateData_getState(self)

    def getStateDeriv(self):
        return _stateArchitecture.StateData_getStateDeriv(self)

    def getName(self):
        return _stateArchitecture.StateData_getName(self)

    def getRowSize(self):
        return _stateArchitecture.StateData_getRowSize(self)

    def getColumnSize(self):
        return _stateArchitecture.StateData_getColumnSize(self)

    def isStateActive(self):
        return _stateArchitecture.StateData_isStateActive(self)

    def disable(self):
        return _stateArchitecture.StateData_disable(self)

    def enable(self):
        return _stateArchitecture.StateData_enable(self)

    def scaleState(self, scaleFactor):
        return _stateArchitecture.StateData_scaleState(self, scaleFactor)

    def __add__(self, operand):
        return _stateArchitecture.StateData___add__(self, operand)

    def __mul__(self, scaleFactor):
        return _stateArchitecture.StateData___mul__(self, scaleFactor)

# Register StateData in _stateArchitecture:
_stateArchitecture.StateData_swigregister(StateData)

def eigenMatrixXd2CArray(inMat, outArray):
    return _stateArchitecture.eigenMatrixXd2CArray(inMat, outArray)

def eigenMatrixXi2CArray(inMat, outArray):
    return _stateArchitecture.eigenMatrixXi2CArray(inMat, outArray)

def eigenVector3d2CArray(inMat, outArray):
    return _stateArchitecture.eigenVector3d2CArray(inMat, outArray)

def eigenMRPd2CArray(inMat, outArray):
    return _stateArchitecture.eigenMRPd2CArray(inMat, outArray)

def eigenMatrix3d2CArray(inMat, outArray):
    return _stateArchitecture.eigenMatrix3d2CArray(inMat, outArray)

def cArray2EigenMatrixXd(inArray, nRows, nCols):
    return _stateArchitecture.cArray2EigenMatrixXd(inArray, nRows, nCols)

def cArray2EigenVector3d(inArray):
    return _stateArchitecture.cArray2EigenVector3d(inArray)

def cArray2EigenMRPd(inArray):
    return _stateArchitecture.cArray2EigenMRPd(inArray)

def cArray2EigenMatrix3d(inArray):
    return _stateArchitecture.cArray2EigenMatrix3d(inArray)

def c2DArray2EigenMatrix3d(in2DArray):
    return _stateArchitecture.c2DArray2EigenMatrix3d(in2DArray)

def eigenM1(angle):
    return _stateArchitecture.eigenM1(angle)

def eigenM2(angle):
    return _stateArchitecture.eigenM2(angle)

def eigenM3(angle):
    return _stateArchitecture.eigenM3(angle)

def eigenTilde(vec):
    return _stateArchitecture.eigenTilde(vec)

def eigenMRPd2Vector3d(vec):
    return _stateArchitecture.eigenMRPd2Vector3d(vec)

def eigenC2MRP(arg1):
    return _stateArchitecture.eigenC2MRP(arg1)

def newtonRaphsonSolve(initialEstimate, accuracy, f, fPrime):
    return _stateArchitecture.newtonRaphsonSolve(initialEstimate, accuracy, f, fPrime)

import sys
protectAllClasses(sys.modules[__name__])


