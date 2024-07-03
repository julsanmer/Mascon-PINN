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


#include "dmcukf.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <iostream>
#include <cstring>
#include <math.h>
#include <time.h>

/*! @brief Creates an instance of the batch class.
 */
Batch::Batch()
{
}

/*! Empty destructor method.
 */
Batch::~Batch()
{
}

/*! @brief Creates an instance of the DMC-UKF class.
 */
DMCUKF::DMCUKF()
{
    /* Initialize time */
    this->tk = 0;
    
    /* Set state dimension */
    this->nx = 9;
    
    /* Declare current state and its covariance */
    this->x(this->nx);
    this->Pxx(this->nx, this->nx);
    
    /* Declare process noise covariance */
    this->Pproc(this->nx, this->nx);
}

/*! Empty destructor method.
 */
DMCUKF::~DMCUKF()
{
}

/*! Resets the module
 */
void DMCUKF::Reset(uint64_t CurrentSimNanos)
{    
    /* Initialize UKF weights */
    this->ukf.initWeights(this->nx);
    
    /* Set initial state distribution and process noise in the UKF */
    this->ukf.setState(this->x, this->Pxx, this->Pproc);
}

/*! Process input messages to module variables
 */
void DMCUKF::processInputs()
{
    /* Extract dcm of small body w.r.t. inertial frame */
    Eigen::Matrix3d dcm_PN0;
    double dcm_PN0_array[3][3];
    MRP2C(this->ephemerisPlanet.sigma_BN, dcm_PN0_array);
    dcm_PN0 = cArray2EigenMatrix3d(*dcm_PN0_array);
    
    /* Compute dcm of small body w.r.t. small body centred inertial frame */
    this->dcm_PN1 = dcm_PN0 * this->dcm_N1N0.transpose();
    
    /* Extract small body position w.r.t. inertial frame */
    Eigen::Vector3d r_PS_N0;
    r_PS_N0 = cArray2EigenVector3d(this->ephemerisPlanet.r_BdyZero_N);
    
    /* Compute small body position w.r.t. Sun in small body centred inertial frame */
    this->r_PS_N1 = this->dcm_N1N0 * r_PS_N0;

    /* Extract dcm of spacecraft w.r.t. inertial frame */
    double dcm_BN0_array[3][3];
    MRP2C(this->spacecraftAtt.sigma_BN, dcm_BN0_array);
    
    /* Compute dcm of spacecraft w.r.t. small body rotating frame */
    this->dcm_BP = cArray2EigenMatrix3d(*dcm_BN0_array) * dcm_PN0.transpose();
    
    /* Get number of landmarks */
    this->nLmk = int(this->landmarkInMsgs.size());
    
    /* Preallocate pixel and visibility status vectors */
    Eigen::VectorXi pixelVec, isVisible;
    pixelVec.setZero(2*this->nLmk);
    isVisible.setZero(this->nLmk);
    
    /* Loop through landmarks */
    for (unsigned int i = 0; i < this->nLmk; i++){
        /* Save landmark pixel */
        pixelVec(i) = this->landmarkMsgBuffer[i].pL[0];
        pixelVec(this->nLmk + i) = this->landmarkMsgBuffer[i].pL[1];
        
        /* Save visibility status */
        isVisible(i) = this->landmarkMsgBuffer[i].isVisible;
    }
    
    /* Prepare measurements for the ukf step */
    this->prepareMeas(pixelVec, isVisible);
}

/*! This method computes the state dynamics derivative
    @param x: state vector
 */
Eigen::VectorXd DMCUKF::dynamics(Eigen::VectorXd x)
{
    /* Initialize state derivative */
    Eigen::VectorXd xDot;
    xDot.setZero(this->nx);
    
    /* Declare position, velocity and unmodeled acceleration
     (all expressed in small body centred inertial frame) */
    Eigen::Vector3d r, v, a;
    
    /* Define small body gravity, srp and Sun 3rd body gravity accelerations */
    Eigen::Vector3d a_grav,  a_SRP, a_Sun;
    
    /* Define spacecraft position w.r.t. Sun */
    Eigen::Vector3d r_BS_N1;
    
    /* Extract position, velocity and unmodeled acceleration */
    r << x.segment(0, 3);
    v << x.segment(3, 3);
    a << x.segment(6, 3);
        
    /* Compute small body gravity acceleration */
    a_grav = this->gravityModel->computeField(this->dcm_PN1 * r);
        
    /* Preallocate solar perturbations */
    a_SRP.setZero(3);
    a_Sun.setZero(3);
        
    /* Compute spacecraft position w.r.t. Sun */
    r_BS_N1 = this->r_PS_N1 + r;
        
    /* Add srp acceleration if required */
    if (this->useSRP == true){
        a_SRP = this->CR * this->ASRP * SOLAR_FLUX_EARTH*pow(AU*1000.,2) / (SPEED_LIGHT * pow(r_BS_N1.norm(),3) * this->m)*(r_BS_N1);
    }
        
    /* Add Sun 3rd body gravity if required */
    if (this->useSun == true){
        a_Sun = -(MU_SUN*1e9) * (r_BS_N1/pow(r_BS_N1.norm(),3) - this->r_PS_N1/pow(this->r_PS_N1.norm(),3));
    }
        
    /* Compute dynamics time derivative */
    xDot.segment(0, 3) = v;
    xDot.segment(3, 3) = a + this->dcm_PN1.transpose() * a_grav + a_SRP + a_Sun;
    
    return xDot;
}

/*! This method encodes the process function
    @param x: state vector
 */
Eigen::VectorXd DMCUKF::funcProc(Eigen::VectorXd x){
    /* Compute timestep for integration */
    double dt = (this->tspan(1) - this->tspan(0)) / this->Nint;
    
    /* Initialize state derivative */
    Eigen::VectorXd k1, k2, k3, k4;
    
    /* Do integration loop */
    for (int i = 0; i < this->Nint; i++){
        /* Get derivative */
        k1 = this->dynamics(x);
        k2 = this->dynamics(x + dt*k1/2);
        k3 = this->dynamics(x + dt*k2/2);
        k4 = this->dynamics(x + dt*k3);
        
        /* Do forward Euler integration step */
        x += dt*(k1 + 2*k2 + 2*k3 + k4) / 6;
    }
    
    return x;
}

/*! This method encodes state to measurements function
    @param x: state vector
 */
Eigen::VectorXd DMCUKF::funcMeas(Eigen::VectorXd x){
    /* Initialize measurement vector */
    Eigen::VectorXd z;
    z.setZero(this->nz);
    
    double px, py;
    
    /* Obtain spacecraft position in small body rotating frame */
    Eigen::Vector3d r;
    r = this->dcm_PN1 * x.segment(0, 3);
    
    /* Declare ith landmark position and relative position w.r.t. spacecraft */
    Eigen::Vector3d rLmk_i, dr_i;

    /* */
    for (int i = 0; i < this->nVisible; i++){
        /* Extract index */
        rLmk_i = this->rLmkVisible.row(i).transpose();
            
        /* Compute relative distance between facet center and spacecraft */
        dr_i = this->dcm_CP * (rLmk_i - r);
            
        /* Compute pixel location */
        px = this->f * dr_i(0) / (dr_i(2) * this->wPixel);
        py = this->f * dr_i(1) / (dr_i(2) * this->wPixel);
        z.segment(2*i, 2) << px, py;
    }
    
    return z;
}

/*! This method transforms the raw measurements to the ukf form
 */
void DMCUKF::prepareMeas(Eigen::VectorXi pixelVec, Eigen::VectorXi isVisible){
    /* Set measurement flag to false by default */
    this->flagMeas = false;
    
    /* Get number of visible landmarks */
    this->nVisible = isVisible.sum();
    
    /* If there is at least one visible landmark */
    if (this->nVisible > 0){
        /* Set measurement flag to true */
        this->flagMeas = true;
        
        /* Declare measurements dimension */
        this->nz = 2*this->nVisible;
        
        /* Resize measurements, noise covariance and visible landmarks positions */
        this->z.resize(2*this->nVisible);
        this->Pmeas.resize(2*this->nVisible, 2*this->nVisible);
        this->Pmeas.setZero(2*this->nVisible, 2*this->nVisible);
        this->rLmkVisible.resize(this->nVisible, 3);
        
        /* Declare horizontal and vertical pixel */
        double px, py;
        
        /* Set internal counter for visible landmarks */
        int cont = 0;
        
        /* Loop through landmarks */
        for (int i = 0; i < this->nLmk; i++){
            /* Check if ith landmark is visible */
            if (isVisible(i) == 1){
                /* Get pixel */
                px = pixelVec(i);
                py = pixelVec(this->nLmk + i);
                
                /* Center the pixel to mitigate dispersion */
                if (px > 0){
                    px -= 0.5;
                }
                else{
                    px += 0.5;
                }
                if (py > 0){
                    py -= 0.5;
                }
                else{
                    py += 0.5;
                }
                
                /* Fill measurement vector and covariance */
                this->z.segment(2*cont, 2) << px, py;
                this->Pmeas.block(2*cont, 2*cont, 2, 2) = this->Ppixel;
                
                /* Fill visible landmark position */
                this->rLmkVisible.row(cont) = this->rLmk.row(i);
                
                /* Update visible landmarks counter */
                cont += 1;
            }
        }
    }
}

/*! This method transforms a MRP to a direction cosine matrix
 @param mrp: modified Rodrigues parameters
 @return dcm: direction cosine matrix
 */
Eigen::Matrix3d DMCUKF::mrp2dcm(Eigen::Vector3d mrp){
    /* Declare dcm */
    Eigen::Matrix3d dcm;
    
    /* Declare mrp and dcm as arrays */
    double mrp_array[3];
    double dcm_array[3][3];
    
    /* Set mrp array and obtain dcm array */
    v3Set(mrp(0), mrp(1), mrp(2), mrp_array);
    MRP2C(mrp_array, dcm_array);
    
    /* Obtain dcm matrix */
    dcm = cArray2EigenMatrix3d(*dcm_array);
    
    return dcm;
}

/*! This method is used to read the input messages.
    @return void
 */
void DMCUKF::readInputMessages(){
    /* Read landmark messages */
    bool flagRead;
    LandmarkMsgPayload lmkMsg;
    
    /* Check if there are landmark messages */
    if(!this->landmarkInMsgs.empty())
    {
        flagRead = true;
        
        /* Loop through landmarks*/
        for (int i = 0; i < this->landmarkInMsgs.size(); i++) {
            /* Save ith landmark message in buffer */
            lmkMsg = this->landmarkInMsgs.at(i)();
            flagRead = flagRead && this->landmarkInMsgs.at(i).isWritten();
            this->landmarkMsgBuffer.push_back(lmkMsg);
        }
    }
    else {
        /* Output error */
        flagRead = false;
        bskLogger.bskLog(BSK_ERROR, "There is no input landmark messages.");
    }
    
    /* Read planet ephemeris message */
    this->ephemerisPlanet = this->ephemerisInMsg();
    
    /* Read spacecraft attitude message */
    this->spacecraftAtt = this->attitudeInMsg();
}

/*! This method writes the output messages
 */
void DMCUKF::writeOutputMessages(uint64_t CurrentSimNanos){
    /* Create output msg buffers */
    DMCUKFMsgPayload DMCUKFMsgBuffer;
    
    /* Zero the output message buffers before assigning values */
    DMCUKFMsgBuffer = this->DMCUKFOutMsg.zeroMsgPayload;
    
    /* Assign values to the DMC-UKF buffer message */
    eigenMatrixXd2CArray(this->x, DMCUKFMsgBuffer.x);
    eigenMatrixXd2CArray(this->Pxx, *DMCUKFMsgBuffer.Pxx);
    
    /* Write the buffer to the DMC-UKF C++ output message */
    this->DMCUKFOutMsg.write(&DMCUKFMsgBuffer, this->moduleID, CurrentSimNanos);
}

/*! This is the main method that gets called every time the module is updated.
 */
void DMCUKF::UpdateState(uint64_t CurrentSimNanos)
{
    /* Get current time */
    this->tk1 = CurrentSimNanos * NANO2SEC;
    
    /* Read input messages */
    this->readInputMessages();
    
    /* Process inputs to module variables */
    this->processInputs();
    
    /* Do ukf update */
    this->ukf.update(this);
    
    /* Write output messages */
    this->writeOutputMessages(CurrentSimNanos);
    
    /* Update initial time for next iteration */
    this->tk = CurrentSimNanos * NANO2SEC;
}

/*! Process a batch of measurements from a initial state and covariance
*/
void DMCUKF::processBatch(){
    /* Obtain number of measurements */
    int nZ = int(this->batch.t.size());

    /* Initialize outputs */
    this->batch.x.setZero(nZ, this->nx);
    this->batch.Pxx.setZero(nZ, this->nx * this->nx);
    
    this->ukf.setState(this->x, this->Pxx, this->Pproc);
    
    /* Save ukf initial condition */
    this->batch.x.row(0) = this->ukf.x_k.transpose();
    Eigen::Map<Eigen::RowVectorXd> PxxRow(this->ukf.Pxx_k.data(), this->ukf.Pxx_k.size());
    this->batch.Pxx.row(0) = PxxRow;
    
    /* Declare computational time */
    Eigen::VectorXd tcpu;
    tcpu.setZero(nZ);
    
    /* Loop to update */
    for (int i = 1; i < nZ; i++){
        /* Obtain timespan */
        this->tspan = this->batch.t.segment(i-1, 2);
        
        /* Update time-varying variables (small body heliocentric position and orientation, and camera orientation w.r.t. small body) */
        this->r_PS_N1 = this->batch.r_PS_N1.row(i).transpose();
        this->dcm_PN1 = this->mrp2dcm(this->batch.mrp_PN0.row(i).transpose()) * this->dcm_N1N0.transpose();
        this->dcm_CP = this->dcm_CB * this->mrp2dcm(this->batch.mrp_BP.row(i).transpose());
                
        /* Prepare measurement */
        this->prepareMeas(this->batch.pLmk.row(i).transpose(), this->batch.isvisibleLmk.row(i).transpose());
        
        /* Do ukf update */
        this->ukf.update(this);
                
        /* Save current estimate and its covariance */
        this->batch.x.row(i) = this->ukf.x_k.transpose();
        Eigen::Map<Eigen::RowVectorXd> PxxRow(this->ukf.Pxx_k.data(), this->ukf.Pxx_k.size());
        this->batch.Pxx.row(i) = PxxRow;
    }
    
    /* Set last estimate */
    this->x = this->ukf.x_k;
    this->Pxx = this->ukf.Pxx_k;
}

/*! @brief Creates an instance of the UKF nested class.
 */
DMCUKF::UKF::UKF(){
    /* Set hyperparameters */
    this->alpha = 0;
    this->beta = 2;
    this->kappa = 1e-3;
}

/*! Empty destructor method.
 */
DMCUKF::UKF::~UKF()
{
}

/*! This method initializes UKF weights and sigma points spread factor.
    @param n: state dimension
 */
void DMCUKF::UKF::initWeights(int n)
{
    /* Save state dimension */
    this->nx = n;
    
    /* Preallocate weights and sigma spread factor */
    this->cx.setZero(this->nx);
    this->wgt_m.setZero(2*this->nx+1);
    this->wgt_c.setZero(2*this->nx+1);
    
    /* Compute weights of the central sigma point */
    this->wgt_m(0) = this->kappa / (this->kappa + this->nx);
    this->wgt_c(0) = this->wgt_m(0) + 1 - pow(this->alpha, 2) + this->beta;
    
    /* Loop through state dimension */
    for (int i = 0; i < this->nx; i++) {
        /* Fill sigma point spread factor */
        this->cx(i) = sqrt(this->nx + this->kappa);
        
        /* Fill sigma point weights */
        this->wgt_m(i+1) = 1 / (2*(this->nx + this->kappa));
        this->wgt_m(this->nx+i+1) = this->wgt_m(i+1);
        this->wgt_c(i+1) = this->wgt_m(i+1);
        this->wgt_c(this->nx+i+1) = this->wgt_m(i+1);
    }
}

/*! This method sets the state distribution and process noise.
    @param x: state
    @param Pxx: state covairance
    @param Pproc: process noise covariance
 */
void DMCUKF::UKF::setState(Eigen::VectorXd x, Eigen::MatrixXd Pxx, Eigen::MatrixXd Pproc)
{
    /* Set state distribution */
    this->x_k = x;
    this->Pxx_k = Pxx;
    
    /* Set process noise */
    this->Pproc = Pproc;
}

/*! This method does the UKF update step.
    @param dmcukf: main DMCUKF class
 */
void DMCUKF::UKF::update(DMCUKF* dmcukf)
{
    /* Do process UT */
    this->processUT(dmcukf);
    
    /* If there is an available measurement, do the UKF update step */
    if (dmcukf->flagMeas) {
        /* Declare Kalman gain */
        Eigen::MatrixXd K;
        
        /* Get measurement dimension and resize measurement-related variables */
        this->nz = dmcukf->nz;
        this->z.resize(this->nz);
        this->Pmeas.resize(this->nz, this->nz);
        this->Pzx.resize(this->nx, this->nz);
        this->Pzz.resize(this->nz, this->nz);
        
        /* Get measurement and its noise covariance */
        this->z = dmcukf->z;
        this->Pmeas = dmcukf->Pmeas;
                
        /* Do UT of a-priori state to measurements */
        this->measUT(dmcukf);
        
        /* Compute Kalman gain */
        K = this->Pzz.llt().solve(this->Pzx).transpose();

        /* Update state mean and covariance with Kalman gain */
        this->x_k = this->x_k1_ + K * (this->z - this->z_k1_);
        this->Pxx_k = this->Pxx_k1_ - K * this->Pzz * K.transpose();
    }
    else{
        /* Update with a-priori state */
        this->x_k = this->x_k1_;
        this->Pxx_k = this->Pxx_k1_;
    }
}

/*! This method computes the a-priori state distribution.
    @param dmcukf: main DMCUKF class
 */
void DMCUKF::UKF::processUT(DMCUKF* dmcukf){
    /* Create state Gaussian distribution */
    Dist distx_k(this->x_k, this->Pxx_k);
    
    /* Create sigma points distribution */
    Sigma sigx_k(distx_k, this->cx);

    /* Declare a-priori state of sigma points */
    Eigen::MatrixXd Xproc(this->nx, sigx_k.n_pts);
    
    /* Loop through sigma points */
    for (int i = 0; i < sigx_k.n_pts; i++){
        /* Process sigma point */
        Xproc.col(i) = dmcukf->funcProc(sigx_k.state.col(i));
    }

    /* Create a-priori state Gaussian distribution */
    Dist distx_k1_(Xproc, this->wgt_m, this->wgt_c);
    
    /* Update a-priori state and covariance */
    this->x_k1_ = distx_k1_.mean;
    this->Pxx_k1_ = distx_k1_.cov + this->Pproc;
}

/*! This method computes the a-priori measurement distribution.
    @param dmcukf: main DMCUKF class
 */
void DMCUKF::UKF::measUT(DMCUKF* dmcukf){
    /* Create a-priori state Gaussian distribution */
    Dist distx_k1_(this->x_k1_, this->Pxx_k1_);
        
    /* Create sigma points distribution */
    Sigma sigx_k1_(distx_k1_, this->cx);
    
    /* Declare a-priori measurements of sigma points */
    Eigen::MatrixXd Z(this->nz, sigx_k1_.n_pts);
    
    /* Loop through sigma points */
    for (int i = 0; i < sigx_k1_.n_pts; i++){
        /* Transform state sigma point to a measurement */
        Z.col(i) = dmcukf->funcMeas(sigx_k1_.state.col(i));
    }

    /* Save a-priori state and measurement */
    Eigen::VectorXd xm, zm;
    xm = distx_k1_.mean;
    zm = Z * this->wgt_m;
    
    /* Stack a-priori state and measurement */
    Eigen::MatrixXd Xm, Zm;
    Xm = xm.rowwise().replicate(sigx_k1_.n_pts);
    Zm = zm.rowwise().replicate(sigx_k1_.n_pts);
    
    /* Compute a-priori measurement covariance and measurement-state correlation */
    this->Pzz = (Z - Zm) * this->wgt_c.asDiagonal() * (Z - Zm).transpose();
    this->Pzx = (Z - Zm) * this->wgt_c.asDiagonal() * (sigx_k1_.state - Xm).transpose();
    
    /* Update a-priori measurement and covariance */
    this->z_k1_ = zm;
    this->Pzz += this->Pmeas;
}

/*! This method creates a sigma points distribution.
 */
DMCUKF::UKF::Sigma::Sigma(const Dist& distX, Eigen::VectorXd cx){
    /* Define number of sigma points from state dimension */
    int nx;
    nx = distX.n;
    this->n_pts = 2*nx + 1;
    
    /* Fill the state with the mean value */
    this->state = distX.mean.rowwise().replicate(this->n_pts);
    
    /* Spread from the mean to create sigma points */
    this->state.block(0, 1,    nx, nx) += distX.covL * cx.asDiagonal();
    this->state.block(0, nx+1, nx, nx) -= distX.covL * cx.asDiagonal();
}

/*! This method creates a Gaussian distribution from sigma points.
 */
DMCUKF::UKF::Dist::Dist(Eigen::MatrixXd& X, Eigen::VectorXd wgt_m, Eigen::VectorXd wgt_c){
    /* Get distribution dimension and number of samples */
    this->n = int(X.rows());
    int n_pts = int(X.cols());
    
    /* Compute weighted mean and stack */
    this->mean = X * wgt_m;
    this->Xmean = mean.rowwise().replicate(n_pts);
    
    /* Compute weighted covariance and its square-root */
    this->cov = (X - this->Xmean) * wgt_c.asDiagonal() * (X - this->Xmean).transpose();
    this->covL = this->cov.llt().matrixL();
}

/*! This method creates a Gaussian distribution from mean and covariance
 */
DMCUKF::UKF::Dist::Dist(Eigen::VectorXd& x, Eigen::MatrixXd& Pxx){
    /* Get dimension and set zero-mean */
    this->n = int(Pxx.rows());
    this->mean = x;
    
    /* Set covariance and compute its square-root */
    this->cov = Pxx;
    this->covL = Pxx.llt().matrixL();
}
