# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _gravityModel
else:
    import _gravityModel

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
    return _gravityModel.new_doubleArray(nelements)

def delete_doubleArray(ary):
    return _gravityModel.delete_doubleArray(ary)

def doubleArray_getitem(ary, index):
    return _gravityModel.doubleArray_getitem(ary, index)

def doubleArray_setitem(ary, index, value):
    return _gravityModel.doubleArray_setitem(ary, index, value)

def new_longArray(nelements):
    return _gravityModel.new_longArray(nelements)

def delete_longArray(ary):
    return _gravityModel.delete_longArray(ary)

def longArray_getitem(ary, index):
    return _gravityModel.longArray_getitem(ary, index)

def longArray_setitem(ary, index, value):
    return _gravityModel.longArray_setitem(ary, index, value)

def new_intArray(nelements):
    return _gravityModel.new_intArray(nelements)

def delete_intArray(ary):
    return _gravityModel.delete_intArray(ary)

def intArray_getitem(ary, index):
    return _gravityModel.intArray_getitem(ary, index)

def intArray_setitem(ary, index, value):
    return _gravityModel.intArray_setitem(ary, index, value)

def new_shortArray(nelements):
    return _gravityModel.new_shortArray(nelements)

def delete_shortArray(ary):
    return _gravityModel.delete_shortArray(ary)

def shortArray_getitem(ary, index):
    return _gravityModel.shortArray_getitem(ary, index)

def shortArray_setitem(ary, index, value):
    return _gravityModel.shortArray_setitem(ary, index, value)


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


SHARED_PTR_DISOWN = _gravityModel.SHARED_PTR_DISOWN
class GravityModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _gravityModel.delete_GravityModel

    def initializeParameters(self, *args):
        return _gravityModel.GravityModel_initializeParameters(self, *args)

    def computeField(self, position_planetFixed):
        return _gravityModel.GravityModel_computeField(self, position_planetFixed)

    def computePotentialEnergy(self, positionWrtPlanet_N):
        return _gravityModel.GravityModel_computePotentialEnergy(self, positionWrtPlanet_N)
    bskLogger = property(_gravityModel.GravityModel_bskLogger_get, _gravityModel.GravityModel_bskLogger_set)

# Register GravityModel in _gravityModel:
_gravityModel.GravityModel_swigregister(GravityModel)

