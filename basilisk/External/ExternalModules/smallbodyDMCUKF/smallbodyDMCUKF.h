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


#ifndef SMALLBODYDMCUKF_H
#define SMALLBODYDMCUKF_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "cMsgCInterface/CamNav3Msg_C.h"
#include "cMsgCInterface/EphemerisMsg_C.h"
#include "cMsgCInterface/SmallbodyDMCUKFMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"

/*! @brief state batch UKF class */
class StatemeasBatchDMCUKF
{
public:
    Eigen::VectorXd tcpu;
    Eigen::MatrixXd Xhat;
    Eigen::MatrixXd Pxx;
    
    Eigen::VectorXd tBatch;
    Eigen::MatrixXd pixelBatch;
    Eigen::VectorXd nvisibleBatch;
    Eigen::VectorXd visibleBatch;
    Eigen::MatrixXd latlonBatch;
    Eigen::MatrixXd rBatch_AS;
    Eigen::MatrixXd eul323Batch_AN;
    int nSegment;
    
    Eigen::MatrixXd resZ;
    
    Eigen::VectorXd tGrav;
    Eigen::MatrixXd rGrav;
    Eigen::MatrixXd aGrav;

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    StatemeasBatchDMCUKF();
    ~StatemeasBatchDMCUKF();
};

/*! @brief mascons class */
class MasconDMCUKF
{
public:
    int nM;
    Eigen::MatrixXd posM;
    Eigen::VectorXd muM;
    
    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    MasconDMCUKF();
    ~MasconDMCUKF();
    void initializeParameters();  
    Eigen::Vector3d computeField(const Eigen::Vector3d pos_Pfix);
};

/*! @brief spherical harmonics class */
class SpherharmDMCUKF
{
public:
    double radEquator;    //!< [-] Reference radius for the planet
    double muBody;        //!< [-] Gravitation parameter for the planet
    int degSpher;      //!< [-] Inhomogeneous gravity field degree
    
    Eigen::MatrixXd cBar;  //!< [-] C coefficient set
    Eigen::MatrixXd sBar;  //!< [-] S coefficient set
    std::vector<std::vector<double>> aBar;  //!< [-] Normalized 'derived' Assoc. Legendre
    std::vector<std::vector<double>> n1;    //!< [-] What am I
    std::vector<std::vector<double>> n2;    //!< [-] What am I
    std::vector<std::vector<double>> nQuot1;//!< [-] What am I
    std::vector<std::vector<double>> nQuot2;//!< [-] What am I
    
    int nC;  //!< [-] number of C coefficients
    int nS;  //!< [-] number of S coefficients
    int nCS;  //!< [-] number of total coefficients

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    SpherharmDMCUKF();
    ~SpherharmDMCUKF();
    void initializeParameters();            //!< [-] configure all spher-harm based on inputs
    double getK(const unsigned int degree); //!< class method
    Eigen::Vector3d computeField(const Eigen::Vector3d pos_Pfix);
    void CSvec2CSmat(Eigen::VectorXd Csvec, Eigen::VectorXd CSad);
    Eigen::VectorXd CSmat2CSvec(Eigen::MatrixXd Cmat, Eigen::MatrixXd Smat, Eigen::VectorXd CSad);
};

/*! @brief This module estimates relative spacecraft position, velocity with respect to the body, and the non-Keplerian acceleration perturbing the spacecraft motion, using an unscented Kalman filter (UKF)
 */
class SmallbodyDMCUKF: public SysModel {
public:
    SmallbodyDMCUKF();
    ~SmallbodyDMCUKF();

    void SelfInit();  //!< Self initialization for C-wrapped messages
    void Reset(uint64_t CurrentSimNanos);  //!< Resets module
    void UpdateState(uint64_t CurrentSimNanos);  //!< Updates state
    void updateMeasBatch();  //!< Updates state with a meas batch
private:
    void readMessages();  //!< Reads input messages
    void writeMessages(uint64_t CurrentSimNanos);  //!< Writes output messages
    void predict();  //!< Predict process
    void update();  //!< Update state
    void adimensionalize(double tau);  //!< Adimensionalize state
    Eigen::VectorXd dynamics(double t0, double t, Eigen::VectorXd X);  //!< Process dynamics
    Eigen::VectorXd measurements(Eigen::VectorXd X);  //!< Measurements transformation
    void processPixelMeas(Eigen::VectorXd pixelVec, int nvisible);

public:
    ReadFunctor<CamNav3MsgPayload> camNavInMsg;  //!< Camera nav input message
    ReadFunctor<EphemerisMsgPayload> ephemerisInMsg;  //!< Small body ephemeris input message
    Message<SmallbodyDMCUKFMsgPayload> smallbodyDMCUKFOutMsg;  //!< Small body nav UKF output msg - states and covariances
    SmallbodyDMCUKFMsg_C smallbodyDMCUKFOutMsgC = {};  //!< C-wrapped Small body nav UKF output msg - states and covariances

    BSKLogger bskLogger;  //!< -- BSK Logging
    
    bool useMeasSimple;
    bool useMeasPoscam;
    bool useMeasPixel;
    
    Eigen::VectorXd xhat_k;  //!< Initial state
    Eigen::MatrixXd Pxx_k;  //!< Initial state covariance
    Eigen::MatrixXd Pww;  //!< Process noise covariance
    Eigen::MatrixXd Pvv;  //!< Measurement noise covariance
    
    Eigen::VectorXd resz_k;
    Eigen::VectorXd innx_k;
    
    Eigen::Vector2d latlon_k;
    double lst_k;
    Eigen::Matrix3d dcm_AN;  //!< Small body dcm
    double rotRate;  //!< Small body angular velocity
    
    double alpha;  //!< Filter hyperparameter
    double beta;  //!< Filter hyperparameter
    double kappa;  //!< Filter hyperparameter
    
    int Nint;  //!< Integration steps
    
    double lst0; //!< Small body initial local sidereal time
    
    bool useSH;
    bool useM;
    bool useSRP; //!< Flag to set SRP model
    bool useSun; //!< Flag to set Sun 3rd body gravity
    
    double m; //!< Spacecraft mass
    double CR; //!< Reflectivity coefficient
    double ASRP; //!< SRP area
    
    int nLandmarks;
    Eigen::MatrixXd xyzLandmarks;
    Eigen::MatrixXd pixel;
    double nVisible;
    Eigen::VectorXd idxVisible;
    double f;
    double wPixel;
    
    MasconDMCUKF mascon;  //!< Object that computes the mascons gravity field
    SpherharmDMCUKF spherharm;  //!< Object that computes the spherical harmonics gravity field
    
    StatemeasBatchDMCUKF statemeasBatch;  //!< Object that stores batch of measurements data

    //!    // Distribution type
    class Dist {
        public:
            int n;

            Eigen::VectorXd mean;
            Eigen::MatrixXd cov, covL, Xmean;

            // Generate distribution from sigma points & weights
            Dist(Eigen::MatrixXd& X, Eigen::VectorXd wgt_m, Eigen::VectorXd wgt_c);
        
            // Generate zero-mean Gaussian distribution
            Dist(Eigen::MatrixXd& S);
            
            void func(SmallbodyDMCUKF* smallbodyDMCUKF);
    };


private:
    CamNav3MsgPayload camNavInMsgBuffer;  //!< Message buffer for input translational nav message
    EphemerisMsgPayload ephemerisInMsgBuffer;  //!< Message buffer for asteroid ephemeris

    double tk;  //!< Previous time, ns
    double tk1;  //!< Current time, ns
    
    Eigen::VectorXd cx;  //!< Vector with sigma points spread
    Eigen::VectorXd wgt_m;  //!< Vector with mean weights
    Eigen::VectorXd wgt_c;  //!< Vector with covariance weights
    
    // Augmented state sigma point type
    class Sigma {
        public:
            int n_state, n_pts;
            Eigen::MatrixXd state;

            Sigma(const Dist& distX, Eigen::VectorXd cx);
    };
    
    // Dimensions
    int nx, nz;  //!< Dimensions of state and measurement
    
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
    
    //
    int nLvisible, nL;
    Eigen::MatrixXd pL;
    Eigen::VectorXd Llabels;
};


#endif
