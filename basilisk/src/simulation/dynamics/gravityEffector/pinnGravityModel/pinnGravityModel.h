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

#ifndef PINN_GRAVITY_MODEL_H
#define PINN_GRAVITY_MODEL_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "simulation/dynamics/_GeneralModuleFiles/gravityModel.h"

/** The physics-informed neural network gravity model.
 *
 * In this class, a PINN is defined by loading a ONNX file.
 * Open neural network exchange allows to do inference
 * on a gravity potential neural network
 */
class PINNGravityModel : public GravityModel {
  public:

    /** Initialize all parameters necessary for the computation of gravity.
     *
     * The attribute `muBody` must be set separately.
     *
     * Will return an error message (string) if `xyzVertex` or `orderFacet` were not set.
     * Otherwise, returns an empty optional.
     */
    std::optional<std::string> initializeParameters() override;

    /** Initialize all parameters necessary for the computation of gravity.
     *
     * The attribute `muBody` is read from the given `GravBodyData`.
     *
     * Will return an error message (string) if `xyzVertex` or `orderFacet` were not set.
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
     * (the polyhedral shape of the body is ignored)
     *
     * The position is given relative to the body and in the inertial
     * reference frame.
     */
    double computePotentialEnergy(const Eigen::Vector3d& positionWrtPlanet_N) const override;

  public:
    double muBody = 0;  /**< [m^3/s^2] Gravitation parameter for the planet */

    /**
     * This string defines the path where the ONNX PINN
     * is stored
     */
    std::string PINNPath;
    
    /**
     * This is the step for the finite differentiating of the
     * PINN potential [m]
     */
    double eps = 1;
    
  private:
    // ORT Environment
    std::shared_ptr<Ort::Env> env;

    // Session
    std::shared_ptr<Ort::Session> session;
    
    // Inputs
    std::vector<std::string> inputName;
    std::vector<std::int64_t> inputShape;

    // Outputs
    std::vector<std::string> outputName;
    std::vector<std::int64_t> outputShape;
    
    // Finite differences step
    Eigen::Vector3d dx;
    Eigen::Vector3d dy;
    Eigen::Vector3d dz;
};

#endif /* PINN_GRAVITY_MODEL_H */
