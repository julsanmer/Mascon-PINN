/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#ifndef DMCUKF_H
#define DMCUKF_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/msgPayloadDefC/EphemerisMsgPayload.h"
#include "architecture/msgPayloadDefC/NavAttMsgPayload.h"
#include "architecture/msgPayloadDefC/LandmarkMsgPayload.h"
#include "msgPayloadDefC/DMCUKFMsgPayload.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"
#include "simulation/dynamics/_GeneralModuleFiles/gravityModel.h"


/*! @brief DMC-UKF batch class */
class Batch
{
public:
    Batch();
    ~Batch();
    
public:
    /* Output batches */
    Eigen::VectorXd t;              //!< [s] time
    Eigen::MatrixXd x;              //!< [-] DMC-UKF state
    Eigen::MatrixXd Pxx;            //!< [-] DMC-UKF covariance
    
    /* Measurement batches */
    Eigen::MatrixXi pLmk;           //!< [-] landmark pixels
    Eigen::MatrixXi isvisibleLmk;   //!< [-] landmark visibility status
    
    /* Small body data batches */
    Eigen::MatrixXd r_PS_N1;        //!< [m] small body position w.r.t. Sun in small body centred inertial frame
    Eigen::MatrixXd mrp_PN0;        //!< [-] small body mrp w.r.t. inertial frame
    
    /* Spacecraft attitude data batch */
    Eigen::MatrixXd mrp_BP;         //!< [-] spacecraft mrp w.r.t. small body rotating frame

    BSKLogger bskLogger;            //!< -- BSK Logging
};

/*! @brief This module estimates spacecraft position, velocity and unmodeled acceleration w.r.t. small body by using a dynamical model compensated unscented Kalman filter (DMC-UKF) */
class DMCUKF: public SysModel
{
public:
    DMCUKF();
    ~DMCUKF();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);
    
    void processBatch();
private:
    void readInputMessages();
    void processInputs();
    void writeOutputMessages(uint64_t CurrentSimNanos);
    
    Eigen::VectorXd funcProc(Eigen::VectorXd x);
    Eigen::VectorXd dynamics(Eigen::VectorXd x);
    Eigen::VectorXd funcMeas(Eigen::VectorXd x);
    void prepareMeas(Eigen::VectorXi pixelVec, Eigen::VectorXi isVisible);
    Eigen::Matrix3d mrp2dcm(Eigen::Vector3d mrp);
    
public:
    BSKLogger bskLogger;  //!< -- BSK Logging
    
    /* DMC-UKF variables */
    int nx;                         //!< [-] state dimension
    Eigen::VectorXd x;              //!< [-] state
    Eigen::MatrixXd Pxx;            //!< [-] state covariance
    Eigen::MatrixXd Pproc;          //!< [-] process noise covariance
    Eigen::MatrixXd Pmeas;          //!< [-] measurement noise covariance
    Eigen::MatrixXd Ppixel;         //!< [-] pixel noise covariance
    
    /* Static dcm */
    Eigen::Matrix3d dcm_N1N0;       //!< [-] dcm from heliocentric to equatorial small body inertial
    Eigen::Matrix3d dcm_CB;         //!< [-] dcm from spacecraft to camera frame
        
    /* Process parameters */
    int Nint;                       //!< [-] number of integration steps
    bool useSRP;                    //!< [-] bool flag to set SRP model
    bool useSun;                    //!< [-] bool flag to set Sun 3rd body gravity
    
    /* Spacecraft parameters */
    double m;                       //!< [kg] spacecraft mass
    double CR;                      //!< [-] spacecraft reflection coefficient
    double ASRP;                    //!< [m^2] spacecraft SRP exposed area
    
    /* Camera parameters */
    double f;                       //!< [m] camera focal length
    double wPixel;                  //!< [m] camera pixel width
    
    /* Landmark distribution data */
    int nLmk;                       //!< [-] number of landmarks
    Eigen::MatrixXd rLmk;           //!< [m] landmark positions in small body rotating frame
    
    std::vector<ReadFunctor<LandmarkMsgPayload>> landmarkInMsgs;    //!< vector of other sc state input messages
    ReadFunctor<EphemerisMsgPayload> ephemerisInMsg;                //!< planet ephemeris input message
    ReadFunctor<NavAttMsgPayload> attitudeInMsg;                    //!< spacecraft state input msg
    
    std::vector<LandmarkMsgPayload> landmarkMsgBuffer;              //!< buffer of landmark input data
    NavAttMsgPayload spacecraftAtt;                                 //!< input spacecraft attitude
    EphemerisMsgPayload ephemerisPlanet;                            //!< input planet ephemeris
    Message<DMCUKFMsgPayload> DMCUKFOutMsg;                         //!< output dmc-ukf message
    
    /*! @brief This nested class encodes the generic UKF algorithm */
    class UKF
    {
    public:
        UKF();
        ~UKF();
        
        void initWeights(int n);
        void setState(Eigen::VectorXd x, Eigen::MatrixXd Pxx, Eigen::MatrixXd Pproc);
        
        void processUT(DMCUKF* dmcukf);
        void measUT(DMCUKF* dmcukf);
        void update(DMCUKF* dmcukf);
        
    public:
        /* State variables */
        int nx;                         //!< [-] state dimension
        Eigen::VectorXd x_k;            //!< [-] current state
        Eigen::MatrixXd Pxx_k;          //!< [-] current state covariance
        Eigen::MatrixXd Pproc;          //!< [-] process noise covariance
        
        /* Measurement variables */
        int nz;                         //!< [-] measurement dimension
        Eigen::VectorXd z;              //!< [-] incoming measurement
        Eigen::MatrixXd Pmeas;          //!< [-] cross-correlation between state and measurement
        
        /* Hyperparameters */
        double alpha;                   //!< [-] hyperparameter
        double beta;                    //!< [-] hyperparameter
        double kappa;                   //!< [-] hyperparameter
        
    private:
        /* Weigths */
        Eigen::VectorXd wgt_m;          //!< [-] mean weights
        Eigen::VectorXd wgt_c;          //!< [-] covariance weights
        Eigen::VectorXd cx;             //!< [-] sigma points spread factor
        
        /* A-priori UKF variables */
        Eigen::VectorXd x_k1_;          //!< [-] a-priori state
        Eigen::MatrixXd Pxx_k1_;        //!< [-] a-priori state covariance
        Eigen::VectorXd z_k1_;          //!< [-] a-priori measurement
        Eigen::MatrixXd Pzz;            //!< [-] a-priori measurement covariance
        Eigen::MatrixXd Pzx;            //!< [-] cross-correlation between state and measurement
        
        /*! @brief Statistical distribution class */
        class Dist
        {
        public:
            /* Distribution variables */
            int n;                      //!< [-] distribution dimension
            Eigen::VectorXd mean;       //!< [-] distribution mean
            Eigen::MatrixXd cov;        //!< [-] distribution covariance
            Eigen::MatrixXd covL;       //!< [-] distribution square root covariance
            Eigen::MatrixXd Xmean;      //!< [-] stacks distribution mean per sample point

            /* Create distribution from sigma points & weights */
            Dist(Eigen::MatrixXd& X, Eigen::VectorXd wgt_m, Eigen::VectorXd wgt_c);
            
            /* Create a Gaussian distribution */
            Dist(Eigen::VectorXd& x, Eigen::MatrixXd& Pxx);
        };
        
        /*! @brief Sigma points class */
        class Sigma
        {
        public:
            /* Sigma points variables */
            int n_state;                //!< [-] state dimension
            int n_pts;                  //!< [-] number of sigma points
            Eigen::MatrixXd state;      //!< [-] sigma points

            Sigma(const Dist& distX, Eigen::VectorXd cx);
        };
    };
    
    /* Objects */
    std::shared_ptr<GravityModel> gravityModel; /**< Model used to compute the gravity of the object */
    Batch batch;                        //!< batch object
    UKF ukf;                            //!< UKF object

private:
    /* Time variables */
    double tk;                          //!< [s] previous time
    double tk1;                         //!< [s] current time
    Eigen::Vector2d tspan;              //!< [s] timespan initial and final time
        
    /* Measurement variables */
    int nz;                             //!< [-] measurement dimension
    Eigen::VectorXd z;                  //!< [-] measurement
    int nVisible;                       //!< [-] number of visible landmarks
    Eigen::MatrixXd rLmkVisible;        //!< [m] positions of visible landmarks in small body rotating frame
    bool flagMeas;                      //!< [-] boolean flag to tell if a measurement is available
    
    /* Small body variables */
    Eigen::Vector3d r_PS_N1;            //!< [m] small body position w.r.t. Sun in small body centred inertial frame
    Eigen::Matrix3d dcm_PN1;            //!< [-] dcm from equatorial small body inertial to rotating small body frame
    
    /* Spacecraft variables */
    Eigen::Matrix3d dcm_BP;             //!< [-] dcm from rotating small body to spacecraft frame
    Eigen::Matrix3d dcm_CP;             //!< [-] dcm from rotating small body to camera frame
};


#endif
