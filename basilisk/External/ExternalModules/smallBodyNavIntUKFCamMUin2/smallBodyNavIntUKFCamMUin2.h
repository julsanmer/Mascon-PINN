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


#ifndef SMALLBODYNAVINTUKFCAMMUIN2_H
#define SMALLBODYNAVINTUKFCAMMUIN2_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "cMsgCInterface/CameraNavMsg_C.h"
#include "cMsgCInterface/EphemerisMsg_C.h"
#include "cMsgCInterface/SmallBodyNavIntMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"

/*! @brief state batch UKF class */
class StateBatchIntUKFCamMUin2
{
public:
    Eigen::VectorXd tcpu;
    Eigen::MatrixXd Xhat;
    Eigen::MatrixXd Pxx;
    
    Eigen::MatrixXd resZ;
    
    Eigen::MatrixXd rGrav;
    Eigen::MatrixXd aGrav;

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    StateBatchIntUKFCamMUin2();
    ~StateBatchIntUKFCamMUin2();
};

/*! @brief spherical harmonics class */
class SphericalHarmonicsIntUKFCamMUin2
{
public:
    double radEquator;    //!< [-] Reference radius for the planet
    double muBody;        //!< [-] Gravitation parameter for the planet
    int degSpher;      //!< [-] Inhomogeneous gravity field degree
    
    int nCS, nC;           //!< [-] Number of coefficients
    Eigen::MatrixXi CSidx; //!< [-] Matrix with spherharm coefficients
    Eigen::VectorXi nCarray, nSarray;
    
    Eigen::MatrixXd cBar;  //!< [-] C coefficient set
    Eigen::MatrixXd sBar;  //!< [-] S coefficient set
    std::vector<std::vector<double>> aBar;  //!< [-] Normalized 'derived' Assoc. Legendre
    std::vector<std::vector<double>> n1;    //!< [-] What am I
    std::vector<std::vector<double>> n2;    //!< [-] What am I
    std::vector<std::vector<double>> nQuot1;//!< [-] What am I
    std::vector<std::vector<double>> nQuot2;//!< [-] What am I

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:

    SphericalHarmonicsIntUKFCamMUin2();
    ~SphericalHarmonicsIntUKFCamMUin2();
    void initializeParameters();            //!< [-] configure all spher-harm based on inputs
    double getK(const unsigned int degree); //!< class method
    Eigen::Vector3d computeField(const Eigen::Vector3d pos_Pfix);
    void CSVec2csBar(Eigen::VectorXd CSVec);
    void CSIndex();
};

/*! @brief This module estimates relative spacecraft position, velocity with respect to the body, and the non-Keplerian acceleration perturbing the spacecraft motion, using an unscented Kalman filter (UKF)
 */
class SmallBodyNavIntUKFCamMUin2: public SysModel {
public:
    SmallBodyNavIntUKFCamMUin2();
    ~SmallBodyNavIntUKFCamMUin2();

    void SelfInit();  //!< Self initialization for C-wrapped messages
    void Reset(uint64_t CurrentSimNanos);  //!< Resets module
    void UpdateState(uint64_t CurrentSimNanos);  //!< Updates state
    
    void updateMeasBatch(Eigen::VectorXd t, Eigen::MatrixXd Z, Eigen::MatrixXd PvvZ, Eigen::VectorXd Zsol, Eigen::MatrixXd r_AS);  //!< Updates state with a meas batch

private:
    void readMessages();  //!< Reads input messages
    void writeMessages(uint64_t CurrentSimNanos);  //!< Writes output messages
    void predict();  //!< Predict process
    void update();  //!< Update state
    Eigen::VectorXd dynamics(double t0, double t, Eigen::VectorXd X);  //!< Process dynamics
    Eigen::VectorXd measurements(Eigen::VectorXd X);  //!< Measurements transformation

public:
    ReadFunctor<CameraNavMsgPayload> cameraNavInMsg;  //!< Camera nav input message
    ReadFunctor<EphemerisMsgPayload> ephemerisInMsg;  //!< Small body ephemeris input message
    Message<SmallBodyNavIntMsgPayload> smallBodyNavIntOutMsg;  //!< Small body nav UKF output msg - states and covariances
    SmallBodyNavIntMsg_C smallBodyNavIntOutMsgC = {};  //!< C-wrapped Small body nav UKF output msg - states and covariances

    BSKLogger bskLogger;  //!< -- BSK Logging
    
    int measMode; //!< Let choose camera measurements or simple ones

    double mu_ast;  //!< Gravitational constant of the small body
    Eigen::VectorXd xhat_k;  //!< Initial state
    Eigen::MatrixXd Pxx_k;  //!< Initial state covariance
    Eigen::MatrixXd Pww;  //!< Process noise covariance
    Eigen::MatrixXd Pvv;  //!< Measurement noise covariance
    
    Eigen::VectorXd resz_k;

    Eigen::Matrix3d dcm_AN;  //!< Small body dcm
    double rotRate;  //!< Small body angular velocity

    int nx, nz;  //!< Dimensions of state and measurement
    
    double alpha;  //!< Filter hyperparameter
    double beta;  //!< Filter hyperparameter
    double kappa;  //!< Filter hyperparameter

    int Nint;  //!< Integration steps
    double t_ad;  //!< Adimensioanalization time

    double lst0; //!< Small body initial local sidereal time
    
    bool useSpherharm; //!< Flag to set mascons model
    bool useMascons; //!< Flag to set spherical harmonics model
    bool useSRP; //!< Flag to set SRP model
    bool useSun; //!< Flag to set Sun 3rd body gravity
    
    double m; //!< Spacecraft mass
    double CR; //!< Reflectivity coefficient
    double ASRP; //!< SRP area
    
    SphericalHarmonicsIntUKFCamMUin2 spherharm;  //!< Object that computes the spherical harmonics gravity field

    StateBatchIntUKFCamMUin2 stateBatch;  //!< Object that stores batch of measurements data


private:
    CameraNavMsgPayload cameraNavInMsgBuffer;  //!< Message buffer for input translational nav message
    EphemerisMsgPayload ephemerisInMsgBuffer;  //!< Message buffer for asteroid ephemeris

    double tk;  //!< Previous time, ns
    double tk1;  //!< Current time, ns
    
    Eigen::VectorXd cx;  //!< Vector with sigma points spread
    Eigen::VectorXd wgt_m;  //!< Vector with mean weights
    Eigen::VectorXd wgt_c;  //!< Vector with covariance weights
    
    // Distribution type
    class Dist {
        public:
            int n;

            Eigen::VectorXd mean;
            Eigen::MatrixXd cov, covL, Xmean;

            // Generate distribution from sigma points & weights
            Dist(Eigen::MatrixXd& X, Eigen::VectorXd wgt_m, Eigen::VectorXd wgt_c);
        
            // Generate zero-mean Gaussian distribution
            Dist(Eigen::MatrixXd& S);
    };
    
    // Augmented state sigma point type
    class Sigma {
        public:
            int n_state, n_pts;
            Eigen::MatrixXd state;

            Sigma(const Dist& distX, Eigen::VectorXd cx);
    };
    
    // Predicted state
    Eigen::VectorXd xhat_k1_;  //!< Apriori predicted state, skewness and kurtosis
    Eigen::MatrixXd Pxx_k1_;  //!< Apriori predicted state covariance
    
    // Measurements
    Eigen::VectorXd z;  //!< Measurements vector
    int zsol; //!< Flag inidicating if measurement is available
    
    // Small body variables
    Eigen::Vector3d r_AS_k;
    
    // Computation time
    double tcpu;  //!< Computational time

    int nLvisible, nL;
    Eigen::MatrixXd pL;
    Eigen::VectorXd Llabels;
};


#endif
