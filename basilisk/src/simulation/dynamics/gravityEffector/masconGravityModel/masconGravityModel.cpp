/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "masconGravityModel.h"
#include "simulation/dynamics/_GeneralModuleFiles/gravityEffector.h"

std::optional<std::string> MasconGravityModel::initializeParameters()
{
    // If data hasn't been loaded, quit and return failure
    if (this->xyzMascon.size() == 0 || this->muMascon.size() == 0) {
        return "Could not initialize mascon model: the positions (xyzMascon) or standard gravity vector (muMascon) "
               "were not provided.";
    }

    // Initialize total mass
    this->muBody = muMascon.sum();
    
    return {};
}

std::optional<std::string> MasconGravityModel::initializeParameters(const GravBodyData& body)
{
    return this->initializeParameters();
}

Eigen::Vector3d
MasconGravityModel::computeField(const Eigen::Vector3d& position_planetFixed) const
{
    const size_t nMascon = this->muMascon.size();
    
    // Declare relative position spacecraft-mass
    // and preallocate mascon gravity
    Eigen::Vector3d dpos_m, acc;
    acc.setZero(3);
    
    // Loop through point-masses
    for (unsigned int m = 0; m < nMascon; m++) {
        // Compute spacecraft relative position w.r.t. point-mass
        dpos_m = position_planetFixed - this->xyzMascon.row(m).transpose();
        
        // Add mth mass gravity contribution
        acc += -this->muMascon(m) * dpos_m/pow(dpos_m.norm(),3);
    }

    return acc;
}

double
MasconGravityModel::computePotentialEnergy(const Eigen::Vector3d& positionWrtPlanet_N) const
{
    const size_t nMascon = this->muMascon.size();
    
    // Declare relative position spacecraft-mass
    // and preallocate mascon potential
    Eigen::Vector3d dpos_m;
    double U;
    U = 0;
    
    // Loop through point-masses
    for (unsigned int m = 0; m < nMascon; m++) {
        // Compute spacecraft relative position w.r.t. point-mass
        dpos_m = positionWrtPlanet_N - this->xyzMascon.row(m).transpose();
        
        // Add mth mass potential contribution
        U += this->muMascon(m) / dpos_m.norm();
    }

    return U;
}
