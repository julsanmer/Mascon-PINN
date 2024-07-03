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

%module dmcukf
%{
    #include "dmcukf.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *

from Basilisk.simulation.pointMassGravityModel import PointMassGravityModel
from Basilisk.simulation.masconGravityModel import MasconGravityModel
from Basilisk.simulation.pinnGravityModel import PINNGravityModel
from Basilisk.simulation.polyhedralGravityModel import PolyhedralGravityModel
from Basilisk.simulation.sphericalHarmonicsGravityModel import SphericalHarmonicsGravityModel

from Basilisk.utilities import deprecated

Mascon = MasconGravityModel
PINN = PINNGravityModel
Polyhedral = PolyhedralGravityModel
SphericalHarmonics = SphericalHarmonicsGravityModel

from typing import Optional, Union

%}

%include "std_string.i"
%include "swig_conly_data.i"
%include "swig_eigen.i"

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}

%pythonappend DMCUKF::DMCUKF() %{
    object.__setattr__(self, "_pyGravityModel", None) # Enable setting _pyGravityModel
    self.gravityModel = PointMassGravityModel() # Re-set gravityModel to populate the _pyGravityModel%}

%include "sys_model.h"
%include "dmcukf.h"

%include "architecture/msgPayloadDefC/EphemerisMsgPayload.h"
struct EphemerisMsg_C;
%include "architecture/msgPayloadDefC/NavAttMsgPayload.h"
struct NavAttMsg_C;
%include "architecture/msgPayloadDefC/LandmarkMsgPayload.h"
struct LandmarkMsg_C;
%include "msgPayloadDefC/DMCUKFMsgPayload.h"
struct DMCUKFMsg_C;


%extend DMCUKF {
    %pythoncode %{
    
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
        
    %}
}