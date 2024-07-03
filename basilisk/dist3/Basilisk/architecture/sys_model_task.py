# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.1.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _sys_model_task
else:
    import _sys_model_task

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


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _sys_model_task.delete_SwigPyIterator

    def value(self):
        return _sys_model_task.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _sys_model_task.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _sys_model_task.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _sys_model_task.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _sys_model_task.SwigPyIterator_equal(self, x)

    def copy(self):
        return _sys_model_task.SwigPyIterator_copy(self)

    def next(self):
        return _sys_model_task.SwigPyIterator_next(self)

    def __next__(self):
        return _sys_model_task.SwigPyIterator___next__(self)

    def previous(self):
        return _sys_model_task.SwigPyIterator_previous(self)

    def advance(self, n):
        return _sys_model_task.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _sys_model_task.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _sys_model_task.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _sys_model_task.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _sys_model_task.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _sys_model_task.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _sys_model_task.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _sys_model_task:
_sys_model_task.SwigPyIterator_swigregister(SwigPyIterator)
class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _sys_model_task.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _sys_model_task.IntVector___nonzero__(self)

    def __bool__(self):
        return _sys_model_task.IntVector___bool__(self)

    def __len__(self):
        return _sys_model_task.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _sys_model_task.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _sys_model_task.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _sys_model_task.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _sys_model_task.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _sys_model_task.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _sys_model_task.IntVector___setitem__(self, *args)

    def pop(self):
        return _sys_model_task.IntVector_pop(self)

    def append(self, x):
        return _sys_model_task.IntVector_append(self, x)

    def empty(self):
        return _sys_model_task.IntVector_empty(self)

    def size(self):
        return _sys_model_task.IntVector_size(self)

    def swap(self, v):
        return _sys_model_task.IntVector_swap(self, v)

    def begin(self):
        return _sys_model_task.IntVector_begin(self)

    def end(self):
        return _sys_model_task.IntVector_end(self)

    def rbegin(self):
        return _sys_model_task.IntVector_rbegin(self)

    def rend(self):
        return _sys_model_task.IntVector_rend(self)

    def clear(self):
        return _sys_model_task.IntVector_clear(self)

    def get_allocator(self):
        return _sys_model_task.IntVector_get_allocator(self)

    def pop_back(self):
        return _sys_model_task.IntVector_pop_back(self)

    def erase(self, *args):
        return _sys_model_task.IntVector_erase(self, *args)

    def __init__(self, *args):
        _sys_model_task.IntVector_swiginit(self, _sys_model_task.new_IntVector(*args))

    def push_back(self, x):
        return _sys_model_task.IntVector_push_back(self, x)

    def front(self):
        return _sys_model_task.IntVector_front(self)

    def back(self):
        return _sys_model_task.IntVector_back(self)

    def assign(self, n, x):
        return _sys_model_task.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _sys_model_task.IntVector_resize(self, *args)

    def insert(self, *args):
        return _sys_model_task.IntVector_insert(self, *args)

    def reserve(self, n):
        return _sys_model_task.IntVector_reserve(self, n)

    def capacity(self):
        return _sys_model_task.IntVector_capacity(self)
    __swig_destroy__ = _sys_model_task.delete_IntVector

# Register IntVector in _sys_model_task:
_sys_model_task.IntVector_swigregister(IntVector)
class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _sys_model_task.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _sys_model_task.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _sys_model_task.DoubleVector___bool__(self)

    def __len__(self):
        return _sys_model_task.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _sys_model_task.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _sys_model_task.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _sys_model_task.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _sys_model_task.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _sys_model_task.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _sys_model_task.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _sys_model_task.DoubleVector_pop(self)

    def append(self, x):
        return _sys_model_task.DoubleVector_append(self, x)

    def empty(self):
        return _sys_model_task.DoubleVector_empty(self)

    def size(self):
        return _sys_model_task.DoubleVector_size(self)

    def swap(self, v):
        return _sys_model_task.DoubleVector_swap(self, v)

    def begin(self):
        return _sys_model_task.DoubleVector_begin(self)

    def end(self):
        return _sys_model_task.DoubleVector_end(self)

    def rbegin(self):
        return _sys_model_task.DoubleVector_rbegin(self)

    def rend(self):
        return _sys_model_task.DoubleVector_rend(self)

    def clear(self):
        return _sys_model_task.DoubleVector_clear(self)

    def get_allocator(self):
        return _sys_model_task.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _sys_model_task.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _sys_model_task.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _sys_model_task.DoubleVector_swiginit(self, _sys_model_task.new_DoubleVector(*args))

    def push_back(self, x):
        return _sys_model_task.DoubleVector_push_back(self, x)

    def front(self):
        return _sys_model_task.DoubleVector_front(self)

    def back(self):
        return _sys_model_task.DoubleVector_back(self)

    def assign(self, n, x):
        return _sys_model_task.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _sys_model_task.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _sys_model_task.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _sys_model_task.DoubleVector_reserve(self, n)

    def capacity(self):
        return _sys_model_task.DoubleVector_capacity(self)
    __swig_destroy__ = _sys_model_task.delete_DoubleVector

# Register DoubleVector in _sys_model_task:
_sys_model_task.DoubleVector_swigregister(DoubleVector)
class StringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _sys_model_task.StringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _sys_model_task.StringVector___nonzero__(self)

    def __bool__(self):
        return _sys_model_task.StringVector___bool__(self)

    def __len__(self):
        return _sys_model_task.StringVector___len__(self)

    def __getslice__(self, i, j):
        return _sys_model_task.StringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _sys_model_task.StringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _sys_model_task.StringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _sys_model_task.StringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _sys_model_task.StringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _sys_model_task.StringVector___setitem__(self, *args)

    def pop(self):
        return _sys_model_task.StringVector_pop(self)

    def append(self, x):
        return _sys_model_task.StringVector_append(self, x)

    def empty(self):
        return _sys_model_task.StringVector_empty(self)

    def size(self):
        return _sys_model_task.StringVector_size(self)

    def swap(self, v):
        return _sys_model_task.StringVector_swap(self, v)

    def begin(self):
        return _sys_model_task.StringVector_begin(self)

    def end(self):
        return _sys_model_task.StringVector_end(self)

    def rbegin(self):
        return _sys_model_task.StringVector_rbegin(self)

    def rend(self):
        return _sys_model_task.StringVector_rend(self)

    def clear(self):
        return _sys_model_task.StringVector_clear(self)

    def get_allocator(self):
        return _sys_model_task.StringVector_get_allocator(self)

    def pop_back(self):
        return _sys_model_task.StringVector_pop_back(self)

    def erase(self, *args):
        return _sys_model_task.StringVector_erase(self, *args)

    def __init__(self, *args):
        _sys_model_task.StringVector_swiginit(self, _sys_model_task.new_StringVector(*args))

    def push_back(self, x):
        return _sys_model_task.StringVector_push_back(self, x)

    def front(self):
        return _sys_model_task.StringVector_front(self)

    def back(self):
        return _sys_model_task.StringVector_back(self)

    def assign(self, n, x):
        return _sys_model_task.StringVector_assign(self, n, x)

    def resize(self, *args):
        return _sys_model_task.StringVector_resize(self, *args)

    def insert(self, *args):
        return _sys_model_task.StringVector_insert(self, *args)

    def reserve(self, n):
        return _sys_model_task.StringVector_reserve(self, n)

    def capacity(self):
        return _sys_model_task.StringVector_capacity(self)
    __swig_destroy__ = _sys_model_task.delete_StringVector

# Register StringVector in _sys_model_task:
_sys_model_task.StringVector_swigregister(StringVector)
class ConstCharVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _sys_model_task.ConstCharVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _sys_model_task.ConstCharVector___nonzero__(self)

    def __bool__(self):
        return _sys_model_task.ConstCharVector___bool__(self)

    def __len__(self):
        return _sys_model_task.ConstCharVector___len__(self)

    def __getslice__(self, i, j):
        return _sys_model_task.ConstCharVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _sys_model_task.ConstCharVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _sys_model_task.ConstCharVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _sys_model_task.ConstCharVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _sys_model_task.ConstCharVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _sys_model_task.ConstCharVector___setitem__(self, *args)

    def pop(self):
        return _sys_model_task.ConstCharVector_pop(self)

    def append(self, x):
        return _sys_model_task.ConstCharVector_append(self, x)

    def empty(self):
        return _sys_model_task.ConstCharVector_empty(self)

    def size(self):
        return _sys_model_task.ConstCharVector_size(self)

    def swap(self, v):
        return _sys_model_task.ConstCharVector_swap(self, v)

    def begin(self):
        return _sys_model_task.ConstCharVector_begin(self)

    def end(self):
        return _sys_model_task.ConstCharVector_end(self)

    def rbegin(self):
        return _sys_model_task.ConstCharVector_rbegin(self)

    def rend(self):
        return _sys_model_task.ConstCharVector_rend(self)

    def clear(self):
        return _sys_model_task.ConstCharVector_clear(self)

    def get_allocator(self):
        return _sys_model_task.ConstCharVector_get_allocator(self)

    def pop_back(self):
        return _sys_model_task.ConstCharVector_pop_back(self)

    def erase(self, *args):
        return _sys_model_task.ConstCharVector_erase(self, *args)

    def __init__(self, *args):
        _sys_model_task.ConstCharVector_swiginit(self, _sys_model_task.new_ConstCharVector(*args))

    def push_back(self, x):
        return _sys_model_task.ConstCharVector_push_back(self, x)

    def front(self):
        return _sys_model_task.ConstCharVector_front(self)

    def back(self):
        return _sys_model_task.ConstCharVector_back(self)

    def assign(self, n, x):
        return _sys_model_task.ConstCharVector_assign(self, n, x)

    def resize(self, *args):
        return _sys_model_task.ConstCharVector_resize(self, *args)

    def insert(self, *args):
        return _sys_model_task.ConstCharVector_insert(self, *args)

    def reserve(self, n):
        return _sys_model_task.ConstCharVector_reserve(self, n)

    def capacity(self):
        return _sys_model_task.ConstCharVector_capacity(self)
    __swig_destroy__ = _sys_model_task.delete_ConstCharVector

# Register ConstCharVector in _sys_model_task:
_sys_model_task.ConstCharVector_swigregister(ConstCharVector)
class SysModel(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _sys_model_task.SysModel_swiginit(self, _sys_model_task.new_SysModel(*args))
    __swig_destroy__ = _sys_model_task.delete_SysModel

    def SelfInit(self):
        return _sys_model_task.SysModel_SelfInit(self)

    def IntegratedInit(self):
        return _sys_model_task.SysModel_IntegratedInit(self)

    def UpdateState(self, CurrentSimNanos):
        return _sys_model_task.SysModel_UpdateState(self, CurrentSimNanos)

    def Reset(self, CurrentSimNanos):
        return _sys_model_task.SysModel_Reset(self, CurrentSimNanos)
    ModelTag = property(_sys_model_task.SysModel_ModelTag_get, _sys_model_task.SysModel_ModelTag_set)
    CallCounts = property(_sys_model_task.SysModel_CallCounts_get, _sys_model_task.SysModel_CallCounts_set)
    RNGSeed = property(_sys_model_task.SysModel_RNGSeed_get, _sys_model_task.SysModel_RNGSeed_set)
    moduleID = property(_sys_model_task.SysModel_moduleID_get, _sys_model_task.SysModel_moduleID_set)

    def logger(self, *args, **kwargs):
        raise TypeError(
            f"The 'logger' function is not supported for this type ('{type(self).__qualname__}'). "
            "To fix this, update the SWIG file for this module. Change "
            """'%include "sys_model.h"' to '%include "sys_model.i"'"""
        )


# Register SysModel in _sys_model_task:
_sys_model_task.SysModel_swigregister(SysModel)
class ModelPriorityPair(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    CurrentModelPriority = property(_sys_model_task.ModelPriorityPair_CurrentModelPriority_get, _sys_model_task.ModelPriorityPair_CurrentModelPriority_set)
    ModelPtr = property(_sys_model_task.ModelPriorityPair_ModelPtr_get, _sys_model_task.ModelPriorityPair_ModelPtr_set)

    def __init__(self):
        _sys_model_task.ModelPriorityPair_swiginit(self, _sys_model_task.new_ModelPriorityPair())
    __swig_destroy__ = _sys_model_task.delete_ModelPriorityPair

# Register ModelPriorityPair in _sys_model_task:
_sys_model_task.ModelPriorityPair_swigregister(ModelPriorityPair)
class SysModelTask(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _sys_model_task.SysModelTask_swiginit(self, _sys_model_task.new_SysModelTask(*args))
    __swig_destroy__ = _sys_model_task.delete_SysModelTask

    def AddNewObject(self, NewModel, Priority=-1):
        return _sys_model_task.SysModelTask_AddNewObject(self, NewModel, Priority)

    def SelfInitTaskList(self):
        return _sys_model_task.SysModelTask_SelfInitTaskList(self)

    def ExecuteTaskList(self, CurrentSimTime):
        return _sys_model_task.SysModelTask_ExecuteTaskList(self, CurrentSimTime)

    def ResetTaskList(self, CurrentSimTime):
        return _sys_model_task.SysModelTask_ResetTaskList(self, CurrentSimTime)

    def ResetTask(self):
        return _sys_model_task.SysModelTask_ResetTask(self)

    def enableTask(self):
        return _sys_model_task.SysModelTask_enableTask(self)

    def disableTask(self):
        return _sys_model_task.SysModelTask_disableTask(self)

    def updatePeriod(self, newPeriod):
        return _sys_model_task.SysModelTask_updatePeriod(self, newPeriod)

    def updateParentProc(self, parent):
        return _sys_model_task.SysModelTask_updateParentProc(self, parent)
    TaskModels = property(_sys_model_task.SysModelTask_TaskModels_get, _sys_model_task.SysModelTask_TaskModels_set)
    TaskName = property(_sys_model_task.SysModelTask_TaskName_get, _sys_model_task.SysModelTask_TaskName_set)
    parentProc = property(_sys_model_task.SysModelTask_parentProc_get, _sys_model_task.SysModelTask_parentProc_set)
    NextStartTime = property(_sys_model_task.SysModelTask_NextStartTime_get, _sys_model_task.SysModelTask_NextStartTime_set)
    NextPickupTime = property(_sys_model_task.SysModelTask_NextPickupTime_get, _sys_model_task.SysModelTask_NextPickupTime_set)
    TaskPeriod = property(_sys_model_task.SysModelTask_TaskPeriod_get, _sys_model_task.SysModelTask_TaskPeriod_set)
    PickupDelay = property(_sys_model_task.SysModelTask_PickupDelay_get, _sys_model_task.SysModelTask_PickupDelay_set)
    FirstTaskTime = property(_sys_model_task.SysModelTask_FirstTaskTime_get, _sys_model_task.SysModelTask_FirstTaskTime_set)
    taskActive = property(_sys_model_task.SysModelTask_taskActive_get, _sys_model_task.SysModelTask_taskActive_set)
    bskLogger = property(_sys_model_task.SysModelTask_bskLogger_get, _sys_model_task.SysModelTask_bskLogger_set)

# Register SysModelTask in _sys_model_task:
_sys_model_task.SysModelTask_swigregister(SysModelTask)

