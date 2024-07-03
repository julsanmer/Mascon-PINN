/*
 ISC License

 Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

*/

%module masconFit
%{
    #include "masconFit.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *

from Basilisk.fswAlgorithms.polyhedralShapeModel import PolyhedralShapeModel

from Basilisk.utilities import deprecated

PolyhedralShape = PolyhedralShapeModel

from typing import Optional, Union

%}

%include "std_string.i"
%include "swig_conly_data.i"
%include "swig_eigen.i"

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}

%pythonappend MasconFit::MasconFit() %{
    object.__setattr__(self, "_pyShapeModel", None) # Enable setting _pyShapeModel
    self.shapeModel = PolyhedralShapeModel() # Re-set gravityModel to populate the _pyShapeModel%}

%include "masconFit.h"

%extend MasconFit {
    %pythoncode %{
    
    """
    """
    _shapeModel = shapeModel
    @property
    def shapeModel(self):
        return self._pyShapeModel
    
    @shapeModel.setter
    def shapeModel(self, value):
        self._shapeModel = value
        self._pyShapeModel = value

    %}
}
