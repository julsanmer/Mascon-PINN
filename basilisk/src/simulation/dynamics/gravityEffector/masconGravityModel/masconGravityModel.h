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

#ifndef MASCON_GRAVITY_MODEL_H
#define MASCON_GRAVITY_MODEL_H

#include "simulation/dynamics/_GeneralModuleFiles/gravityModel.h"

/** The Mascon gravity model.
 *
 * In this class, a point-mass mascon distribution is declared.
 * Each point-mass is defined by its position relative to the
 * center mass of the planet and its standard gravity parameter.
 */
class MasconGravityModel : public GravityModel {
  public:

    /** Initialize all parameters necessary for the computation of gravity.
     *
     * The attribute `muBody` equals the sum of `muMascon`.
     *
     * Will return an error message (string) if `xyzMascon` or `muMascon` were not set.
     * Otherwise, returns an empty optional.
     */
    std::optional<std::string> initializeParameters() override;

    /** Initialize all parameters necessary for the computation of gravity.
     *
     * Will return an error message (string) if `xyzMascon` or `muMascon` were not set.
     * Otherwise, returns an empty optional.
     */
    std::optional<std::string> initializeParameters(const GravBodyData&) override;

    /** Returns the gravity acceleration at a position around this body.
     *
     * The position is given in the body-fixed reference frame.
     * Likewise, the resulting acceleration should be given in the
     * body-fixed reference frame.
     */
    Eigen::Vector3d computeField(const Eigen::Vector3d& position_planetFixed) const override;

    /** Returns the gravitational potential energy at a position around this body.
     *
     * The current implementation returns the potential energy of a point-mass
     * (the mascon distribution is ignored)
     *
     * The position is given relative to the body and in the inertial
     * reference frame.
     */
    double computePotentialEnergy(const Eigen::Vector3d& positionWrtPlanet_N) const override;

  public:
    double muBody = 0;  /**< [m^3/s^2] Gravitation parameter for the planet */

    /**
     * This matrix contains the position of every point-mass
     * in meters. Each row corresponds to a different point-mass,
     * while each column corresponds to x, y, z respectively.
     */
    Eigen::MatrixX3d xyzMascon;

    /**
     * This vector contains the gravity parameter of every
     * point-mass in m^3/s^2. Each entry corresponds
     * to a different point-mass.
     */
    Eigen::VectorXd muMascon;
};

#endif /* MASCON_GRAVITY_MODEL_H */
