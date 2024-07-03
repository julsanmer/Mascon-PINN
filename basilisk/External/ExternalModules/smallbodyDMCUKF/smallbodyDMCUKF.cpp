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


#include "smallbodyDMCUKF.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <iostream>
#include <cstring>
#include <math.h>
#include <random>
#include <time.h>

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
SmallbodyDMCUKF::SmallbodyDMCUKF()
{
    /* Dimensions */
    this->nx = 9;
    
    /* Initial time */
    this->tk = 0;
    
    /* Predicted state */
    this->xhat_k1_(this->nx);
    this->Pxx_k1_(this->nx, this->nx);
    
    /* Process noise */
    this->Pww(this->nx,this->nx);
    
    /* Sigma points spread and weights */
    this->cx.setZero(this->nx);
    this->wgt_m.setZero(2*this->nx+1);
    this->wgt_c.setZero(2*this->nx+1);
    
    /* Filter hyperparameters */
    this->alpha = 0;
    this->beta = 2;
    this->kappa = 1e-3;
    
    return;
}

/*! Module Destructor */
SmallbodyDMCUKF::~SmallbodyDMCUKF()
{
}

/*! Initialize C-wrapped output messages */
void SmallbodyDMCUKF::SelfInit(){
    SmallbodyDMCUKFMsg_C_init(&this->smallbodyDMCUKFOutMsgC);
}

/*! This method is used to reset the module, check that required input messages are connect and compute weigths.
    @return void
*/
void SmallbodyDMCUKF::Reset(uint64_t CurrentSimNanos)
{
    /* Compute sigma spread and weights to be used in the UT */
    this->wgt_m(0) = this->kappa / (this->kappa + this->nx);
    this->wgt_c(0) = this->wgt_m(0) + 1 - pow(this->alpha,2) + this->beta;
    for (int i = 0; i < this->nx; i++) {
        /* Sigma points spread */
        this->cx(i) = sqrt(this->nx + this->kappa);
        
        /* Assign weigths */
        this->wgt_m(i+1) = 1 / (2*(this->nx + this->kappa));
        this->wgt_m(this->nx+i+1) = this->wgt_m(i+1);
        this->wgt_c(i+1) = this->wgt_m(i+1);
        this->wgt_c(this->nx+i+1) = this->wgt_m(i+1);
    }
    
    /* Initialize gravity properties */
    if (this->useSH == true){
        /* Initialize spherical harmonics */
        this->spherharm.initializeParameters();
    }
    if (this->useM == true){
        /* Initialize mascons */
        this->mascon.initializeParameters();
    }

    
    /* Set number of tracked landmarks and maximum number measurement */
    this->nz = 3;
    /*this->resz_k(this->nz);*/
    
    Dist distx_k(this->Pxx_k);
    distx_k.func(this);
}


/*! This method is used to read the input messages.
    @return void
*/
void SmallbodyDMCUKF::readMessages(){
    /* Read in the input messages */
    this->camNavInMsgBuffer = this->camNavInMsg();
    this->ephemerisInMsgBuffer = this->ephemerisInMsg();
    
    /* Extract dcm and angular velocity of the small body */
    double dcm_AN_array[3][3];
    MRP2C(ephemerisInMsgBuffer.sigma_BN, dcm_AN_array);
    this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);
    /*this->omega_AN_A = cArray2EigenVector3d(this->ephemerisInMsgBuffer.omega_BN_B);*/
}

/*! This method does the prediction step
    @param CurrentSimNanos
    @return void
*/
void SmallbodyDMCUKF::predict(){
    /* Generate distribution */
    Dist distx_k(this->Pxx_k);
    distx_k.mean = this->xhat_k;
    
    /* Generate sigma points */
    Sigma sig(distx_k, this->cx);

    /* Compute prediction */
    Eigen::MatrixXd Xp(this->nx, sig.n_pts);
    for (int i = 0; i < sig.n_pts; i++){
        Xp.col(i) = this->dynamics(this->tk, this->tk1, sig.state.col(i));
    }

    /* Generate final distribution */
    Dist distXp(Xp, this->wgt_m, this->wgt_c);
    
    /* Update values */
    this->xhat_k1_ = distXp.mean;
    this->Pxx_k1_ = distXp.cov + this->Pww;
}

/*! This method updates the state
    @param CurrentSimNanos
    @return void
*/
void SmallbodyDMCUKF::update(){
    /* Generate distributions */
    Dist distx_k1_(this->Pxx_k1_);
    distx_k1_.mean = this->xhat_k1_;
    
    /* Generate sigma points */
    Sigma sig(distx_k1_, this->cx);
    
    /* Compute expected measurements */
    Eigen::MatrixXd Z(this->nz, sig.n_pts), Pzz(this->nz, this->nz), Pzx(this->nz, this->nx), K(this->nx, this->nz), Xu(this->nx, sig.n_pts);
    Eigen::VectorXd xm, zm;
    for (int i = 0; i < sig.n_pts; i++){
        Z.col(i) = measurements(sig.state.col(i));
    }
    
    /* Compute mean and covariances */
    xm = distx_k1_.mean;
    zm = Z * this->wgt_m;
    
    Eigen::MatrixXd Xm, Zm;
    Xm = xm.rowwise().replicate(sig.n_pts);
    Zm = zm.rowwise().replicate(sig.n_pts);
    Pzz = (Z-Zm)*this->wgt_c.asDiagonal()*(Z-Zm).transpose();
    Pzx = (Z-Zm)*this->wgt_c.asDiagonal()*(sig.state-Xm).transpose();
    
    /* Add measurements noise */
    if (this->useMeasSimple == true){
        Pzz += this->dcm_AN.transpose()*this->Pvv*this->dcm_AN;
    }
    else if (this->useMeasPoscam == true or this->useMeasPixel == true){
        Pzz += this->Pvv;
    }

    /* Compute Kalman gain */
    K = Pzz.llt().solve(Pzx).transpose();
    /*if (this->nVisible == 25){
        std::cout << this->Pvv;
    }*/
    
    /*K = Pzx.transpose()*Pzz.inverse();*/
        
    /* Update state distribution */
    Xu = sig.state - K*(Z.colwise() - zm);
    Dist distXu(Xu, this->wgt_m, this->wgt_c);
    distXu.mean = xm + K*(this->z - zm);

    /*std::cout << this->xhat_k1_ - xm;*/
    /*std::cout << this->Pxx_k1_ - K*Pzx*this->Pxx_k1_*/
    
    /* Update state */
    this->xhat_k = distXu.mean;
    this->Pxx_k = distXu.cov;
    
    /*std::cout << this->xhat_k1_;
    std::cout << this->xhat_k;*/
    
    /* Compute residual */
    this->innx_k = K*(this->z - zm);
    /*this->resz_k = this->measurements(this->xhat_k) - this->z;*/
    this->innx_k = (this->Pww - K*Pzz*K.transpose() + this->innx_k*this->innx_k.transpose()).diagonal();
}

/*! This method propagates the dynamics
    @param CurrentSimNanos
    @return void
*/
Eigen::VectorXd SmallbodyDMCUKF::dynamics(double t0, double tf, Eigen::VectorXd X){
    /* Compute timestep */
    double dt = (tf-t0) / this->Nint;
    
    /* Declare output */
    Eigen::VectorXd Xp, Xdot, Xproc;
    Xp.setZero(this->nx), Xdot.setZero(this->nx), Xproc.setZero(this->nx);
    
    /* Define local sidereal time*/
    double lst;
    double theta_AN[3];
    double dcm_AN_array[3][3];

    /* Extract variables */
    Eigen::Vector3d r, v, a;
    Eigen::Vector3d a_grav, r_CS, a_SRP, a_Sun;
    double sunDist;
    for (int i = 0; i < this->Nint; i++){
        /* Initialize states */
        r << X.segment(0,3);
        v << X.segment(3,3);
        a << X.segment(6,3);
        
        /* Propagate rotation */
        lst = this->lst_k + i*dt*this->rotRate;
        v3Set(0,0,lst, theta_AN);
        Euler3232C(theta_AN, dcm_AN_array);
        this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);
        
        /* Choose gravity model */
        if (this->useSH == true){
            /* Compute inhomogeneous gravity */
            a_grav = this->spherharm.computeField(this->dcm_AN*r);
        }
        if (this->useM == true){
            /* Compute inhomogeneous gravity */
            a_grav = this->mascon.computeField(this->dcm_AN*r);
        }
        
        /* Add solar perturbations */
        a_SRP.setZero(3);
        a_Sun.setZero(3);
        r_CS = this->r_AS_k + r;
        if (this->useSRP == true){
            a_SRP = this->CR*this->ASRP*SOLAR_FLUX_EARTH*pow(AU*1000.,2) / (SPEED_LIGHT*pow(r_CS.norm(),3)*this->m)*(r_CS);
            /*std::cout << a_SRP.norm();*/
        }
        if (this->useSun == true){
            a_Sun = -(MU_SUN*1e9)*(r_CS/pow(r_CS.norm(),3) - this->r_AS_k/pow(this->r_AS_k.norm(),3));
        }
    
        /* Compute dynamics derivative */
        Xdot.segment(0,3) = v;
        if (this->useM == true){
            Xdot.segment(3,3) = a + this->dcm_AN.transpose()*a_grav + a_SRP + a_Sun;
        }
        else if (this->useSH == true){
            Xdot.segment(3,3) = - this->spherharm.muBody*r/pow(r.norm(),3) + a + this->dcm_AN.transpose()*a_grav + a_SRP + a_Sun;
        }
    
        /* Use Euler integration to propagate */
        X += Xdot*dt;
    }
    
    /* Prepare output */
    Xp = X;
    return Xp;
}


/*! This method transforms state to measurements
    @param CurrentSimNanos
    @return void
*/
Eigen::VectorXd SmallbodyDMCUKF::measurements(Eigen::VectorXd X){
    Eigen::VectorXd Z;
    if (useMeasPixel == true){
        int idx;
        double px, py;
        Eigen::Vector3d xyzLandmark_i, r, dr_i;
        r = this->dcm_AN*X.segment(0,3);
        Z.setZero(2*this->nVisible);
        
        /* Compute radius, longitude and latitude */
        double radB, lonB, latB;
        radB = r.norm();
        lonB = atan2(r(1),r(0));
        latB = asin(r(2)/radB);
        
        /* Compute dcm for change to camera coordinates */
        double dcm_array[3][3];
        double theta[3];
        Eigen::Matrix3d dcm;
        v3Set(this->latlon_k(1), -(M_PI_2 + this->latlon_k(0)), M_PI_2, theta);
        Euler3232C(theta, dcm_array);
        dcm = cArray2EigenMatrix3d(*dcm_array);
        
        for (int i = 0; i < this->nVisible; i++){
            /* Extract index */
            idx = this->idxVisible(i);
            xyzLandmark_i = this->xyzLandmarks.row(idx).transpose();
            
            /* Compute relative distance between facet center and spacecraft */
            dr_i = dcm*(xyzLandmark_i - r);
            
            /* Compute pixel location */
            px = this->f*dr_i(0)/(dr_i(2)*this->wPixel);
            py = this->f*dr_i(1)/(dr_i(2)*this->wPixel);
            Z.segment(2*i,2) << px, py;
        }
        /*std::cout << this->nVisible << '\n';
        std::cout << px << '\n';*/
    }
    else if (useMeasPoscam == true or useMeasSimple == true){
        /* Declare output */
        Z.setZero(this->nz);
        
        /* Extract position */
        Eigen::Vector3d r;
        r << X.segment(0,3);
        Z = this->dcm_AN*r;
    }
    return Z;
}

void SmallbodyDMCUKF::processPixelMeas(Eigen::VectorXd pixelVec, int nvisible){
    /* Extract number of visible numbers */
    this->nVisible = nvisible;
    
    /* Preallocate and resize variables */
    this->nz = 2*this->nVisible;
    this->z.resize(2*this->nVisible);
    this->idxVisible.resize(this->nVisible);
    this->Pvv.resize(2*this->nVisible,2*this->nVisible);
    this->Pvv.setZero(2*this->nVisible,2*this->nVisible);
    int contVisible;
    contVisible = 0;
    
    /* Loop to fill variables */
    for (int i = 0; i < this->nLandmarks; i++){
        if (pixelVec(2*i) != 0){
            this->idxVisible(contVisible) = i;
            this->z.segment(2*contVisible,2) << pixelVec.segment(2*i,2);
            this->Pvv.block(2*contVisible,2*contVisible,2,2) << pow(1,2), 0, 0, pow(1,2);
            contVisible += 1;
        }
    }
}

SmallbodyDMCUKF::Sigma::Sigma(const Dist& distX, Eigen::VectorXd cx){
    /* Extract dimensions */
    int nx;
    nx = distX.n;
    
    /* Declare variables */
    n_pts = 2*nx+1;
    
    /* Fill all with mean values */
    state = distX.mean.rowwise().replicate(n_pts);
    
    /* Spread sigma points */
    state.block(0, 1,    nx, nx) += distX.covL * cx.asDiagonal();
    state.block(0, 1+nx, nx, nx) -= distX.covL * cx.asDiagonal();
}

/*! This method creates a distribution with skewness and kurtosis
    @return void
*/
SmallbodyDMCUKF::Dist::Dist(Eigen::MatrixXd& X, Eigen::VectorXd wgt_m, Eigen::VectorXd wgt_c){
    /* Compute mean and covariance */
    n = X.rows();
    int n_pts = X.cols();
    mean = X * wgt_m;
    
    Xmean = mean.rowwise().replicate(n_pts);
    cov = (X - Xmean)*wgt_c.asDiagonal()*(X-Xmean).transpose();
    covL = cov.llt().matrixL();
}

/*! This method creates a zero mean Gaussian distribution
    @return void
*/
SmallbodyDMCUKF::Dist::Dist(Eigen::MatrixXd& S){
    /* Compute mean and covariance */
    n = S.rows();
    mean.setZero(n);
    cov = S;
    covL = S.llt().matrixL();
}

void SmallbodyDMCUKF::Dist::func(SmallbodyDMCUKF* smallbodyDMCUKF){
    smallbodyDMCUKF->updateMeasBatch();
}

/*! This is the main method that gets called every time the module is updated.
    @return void
*/
void SmallbodyDMCUKF::updateMeasBatch(){
    /* Obtain number of measurements and t0 */
    int nZ = this->statemeasBatch.tBatch.size();
    this->tk = this->statemeasBatch.tBatch(0);

    /* Initialize outputs */
    Eigen::MatrixXd Xhat;
    Eigen::MatrixXd Pxx;
    Eigen::MatrixXd resZ;
    Eigen::MatrixXd innX;
    Xhat.setZero(nZ,this->nx);
    Pxx.setZero(nZ,this->nx*this->nx);
    /*resZ.setZero(nZ,this->nz);*/
    innX.setZero(nZ,this->nx);

    /* Initialize filling estimated state and covariance */
    Xhat.row(0) = this->xhat_k.transpose();
    Eigen::Map<Eigen::RowVectorXd> PxxRow(this->Pxx_k.data(), this->Pxx_k.size());
    Pxx.row(0) = PxxRow;
    
    /* Compute rotation matrix */
    double theta_AN[3];
    double dcm_AN_array[3][3];
    /*lst = this->lst0 + this->rotRate*tZ(0);
    v3Set(0,0,lst, theta_AN);
    Euler3232C(theta_AN, dcm_AN_array);
    this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);*/
    this->lst_k = this->statemeasBatch.eul323Batch_AN(0,2);
    v3Set(0,0,this->lst_k, theta_AN);
    Euler3232C(theta_AN, dcm_AN_array);
    this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);
    
    /* Preallocate training variables */
    Eigen::MatrixXd rGrav;
    Eigen::MatrixXd aGrav;
    rGrav.setZero(nZ,3);
    aGrav.setZero(nZ,3);

    rGrav.row(0) = (this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
    aGrav.row(0) = (this->dcm_AN*this->xhat_k.segment(6,9)).transpose();
    
    /* Declare computational time */
    Eigen::VectorXd tcpu;
    tcpu.setZero(nZ);
            
    /* Loop to update */
    for (int i = 1; i < nZ; i++){
        /* Start measuring time */
        clock_t start, end;
        start = clock();
        
        /* Get time and measurements */
        this->tk1 = this->statemeasBatch.tBatch(i);
        
        /* Get small body w.r.t. Sun */
        this->r_AS_k = this->statemeasBatch.rBatch_AS.row(i);
        this->latlon_k = this->statemeasBatch.latlonBatch.row(i);
        
        /* Get measurement covariance */
        /*this->Pvv << PvvZ(i,0), 0, 0, 0, PvvZ(i,4), 0, 0, 0, PvvZ(i,8);*/
        /*this->Pvv << 25, 0, 0, 0, 25, 0, 0, 0, 25;*/
            
        /* Update prediction */
        this->predict();
        this->lst_k = this->statemeasBatch.eul323Batch_AN(i,2);
        v3Set(0,0,this->lst_k, theta_AN);
        Euler3232C(theta_AN, dcm_AN_array);
        this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);
        if (this->statemeasBatch.visibleBatch(i) == 1){
            /* Process measurement and update */
            processPixelMeas(this->statemeasBatch.pixelBatch.row(i).transpose(), this->statemeasBatch.nvisibleBatch(i));
            this->update();
        }
        else{
            this->xhat_k = this->xhat_k1_;
            this->xhat_k.segment(6,3) << 0, 0, 0;
            this->Pxx_k = this->Pxx_k1_;
            /*this->resz_k.setZero(3);*/
            this->innx_k.setZero(9);
        }
        
        /* Fill gravity training */
        rGrav.row(i) = (this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
        aGrav.row(i) = (this->dcm_AN*(this->xhat_k.segment(6,3))).transpose();
        if (this->useSH == true){
            aGrav.row(i) += this->spherharm.computeField(this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
        }
        if (this->useM == true){
            aGrav.row(i) += this->mascon.computeField(this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
        }
                
        /* Save outputs */
        Xhat.row(i) = this->xhat_k.transpose();
        /*resZ.row(i) = this->resz_k.transpose();
        innX.row(i) = this->innx_k.transpose().array().abs();*/

        /* Save covariance matrix */
        Eigen::Map<Eigen::RowVectorXd> PxxRow(this->Pxx_k.data(), this->Pxx_k.size());
        Pxx.row(i) = PxxRow;
        
        /* Prepare next iteration */
        this->tk = this->statemeasBatch.tBatch(i);
        
        /* Stop measuring time and calculate the elapsed time */
        end = clock();
        tcpu(i) = double(end - start) / double(CLOCKS_PER_SEC);
    }
    /* Fill outputs */
    this->statemeasBatch.tcpu = tcpu;
    this->statemeasBatch.Xhat = Xhat;
    this->statemeasBatch.Pxx = Pxx;
    /*this->statebatch.resZ = resZ;*/
    
    this->statemeasBatch.rGrav = rGrav;
    this->statemeasBatch.aGrav = aGrav;
}


/*! This method writes the output messages
    @return void
*/
void SmallbodyDMCUKF::writeMessages(uint64_t CurrentSimNanos){
    /* Create output msg buffers */
    SmallbodyDMCUKFMsgPayload smallbodyDMCUKFOutMsgBuffer;
    
    /* Zero the output message buffers before assigning values */
    smallbodyDMCUKFOutMsgBuffer = this->smallbodyDMCUKFOutMsg.zeroMsgPayload;
    
    /* Assign values to the small body navigation output message */
    eigenMatrixXd2CArray(this->xhat_k, smallbodyDMCUKFOutMsgBuffer.state);
    eigenMatrixXd2CArray(this->Pxx_k, *smallbodyDMCUKFOutMsgBuffer.covar);
    /*eigenMatrixXd2CArray(this->z, smallBodyNav1OutMsgBuffer.meas);*/
    smallbodyDMCUKFOutMsgBuffer.tcpu = this->tcpu;
    
    /* Write to the C++-wrapped output messages */
    this->smallbodyDMCUKFOutMsg.write(&smallbodyDMCUKFOutMsgBuffer, this->moduleID, CurrentSimNanos);

    /* Write to the C-wrapped output messages */
    SmallbodyDMCUKFMsg_C_write(&smallbodyDMCUKFOutMsgBuffer, &this->smallbodyDMCUKFOutMsgC, this->moduleID, CurrentSimNanos);
}

/*! This is the main method that gets called every time the module is updated.
    @return void
*/
void SmallbodyDMCUKF::UpdateState(uint64_t CurrentSimNanos)
{
    /* Start measuring time */
    clock_t start, end;
    start = clock();
    
    /* Execute filter */
    this->tk1 = CurrentSimNanos*NANO2SEC;
    this->readMessages();
    this->predict();
    this->update();
    this->writeMessages(CurrentSimNanos);
    this->tk = CurrentSimNanos*NANO2SEC;
    
    /* Stop measuring time and calculate the elapsed time */
    end = clock();
    this->tcpu = double(end - start) / double(CLOCKS_PER_SEC);
}


StatemeasBatchDMCUKF::StatemeasBatchDMCUKF()
{
    return;
}

StatemeasBatchDMCUKF::~StatemeasBatchDMCUKF()
{
    return;
}

SpherharmDMCUKF::SpherharmDMCUKF()
{
    return;
}

SpherharmDMCUKF::~SpherharmDMCUKF()
{
    return;
}

/*
@brief Computes the term (2 - d_l), where d_l is the kronecker delta.
*/
double SpherharmDMCUKF::getK(const unsigned int degree)
{
    return ((degree == 0) ? 1.0 : 2.0);
}

void SpherharmDMCUKF::initializeParameters()
{
    for(unsigned int i = 0; i <= this->degSpher + 1; i++)
    {
        std::vector<double> aRow, n1Row, n2Row;
        aRow.resize(i+1, 0.0);
        // Diagonal elements of A_bar
        if (i == 0)
        {
             aRow[i] = 1.0;
        }
        else
        {
            aRow[i] = sqrt(double((2*i+1)*getK(i))/(2*i*getK(i-1))) * aBar[i-1][i-1];
        }
        n1Row.resize(i+1, 0.0);
        n2Row.resize(i+1, 0.0);
        for (unsigned int m = 0; m <= i; m++)
        {
            if (i >= m + 2)
            {
                n1Row[m] = sqrt(double((2*i+1)*(2*i-1))/((i-m)*(i+m)));
                n2Row[m] = sqrt(double((i+m-1)*(2*i+1)*(i-m-1))/((i+m)*(i-m)*(2*i-3)));

            }
        }
        n1.push_back(n1Row);
        n2.push_back(n2Row);
        aBar.push_back(aRow);
    }
    for (unsigned int l = 0; l <= this->degSpher; l++) // up to _maxDegree-1
    {
        std::vector<double> nq1Row, nq2Row;
        nq1Row.resize(l+1, 0.0);
        nq2Row.resize(l+1, 0.0);
        for (unsigned int m = 0; m <= l; m++)
        {
            if (m < l)
            {
                nq1Row[m] = sqrt(double((l-m)*getK(m)*(l+m+1))/getK(m+1));
            }
            nq2Row[m] = sqrt(double((l+m+2)*(l+m+1)*(2*l+1)*getK(m))/((2*l+3)*getK(m+1)));
        }
        nQuot1.push_back(nq1Row);
        nQuot2.push_back(nq2Row);
    }
    
    /* Compute number of coefficients */
    this->nC = 0;
    this->nS = 0;
    this->nCS = 0;
    int nC_ii = 3;
    int nS_ii = 2;
    
    for (unsigned int i = 2; i <= this->degSpher; i++)
    {
        this->nC += nC_ii;
        this->nS += nS_ii;
        nC_ii += 1;
        nS_ii += 1;
    }
    this->nC -= 1;
    this->nS -= 1;
    this->nCS = this->nC + this->nS;
}


/*! This method computes gravity field
    @param CurrentSimNanos
    @return void
*/
Eigen::Vector3d SpherharmDMCUKF::computeField(const Eigen::Vector3d pos_Pfix)
{
    double x = pos_Pfix[0];
    double y = pos_Pfix[1];
    double z = pos_Pfix[2];
    double r, s, t, u;
    double order;
    double rho;
    double a1, a2, a3, a4, sum_a1, sum_a2, sum_a3, sum_a4;
    std::vector<double> rE, iM, rhol;
    Eigen::Vector3d acc;
    acc.fill(0.0);

    // Change of variables: direction cosines
    r = sqrt(x*x + y*y + z*z);
    s = x/r;
    t = y/r;
    u = z/r;

    order = this->degSpher;

    for (unsigned int l = 1; l <= this->degSpher+1; l++)
    {
        //Diagonal terms are computed in initialize()
        // Low diagonal terms
        this->aBar[l][l-1] = sqrt(double((2*l)*getK(l-1))/getK(l)) * this->aBar[l][l] * u;
    }

    // Lower terms of A_bar
    for (unsigned int m = 0; m <= order+1; m++)
    {
        for(unsigned int l = m + 2; l <= this->degSpher+1; l++)
        {
            this->aBar[l][m] = u * this->n1[l][m] * aBar[l-1][m] - this->n2[l][m] * this->aBar[l-2][m];
        }

        // Computation of real and imaginary parts of (2+j*t)^m
        if (m == 0)
        {
            rE.push_back(1.0);
            iM.push_back(0.0);
        }
        else
        {
            rE.push_back(s * rE[m-1] - t * iM[m-1]);
            iM.push_back(s * iM[m-1] + t * rE[m-1]);
        }
    }

    rho = this->radEquator/r;
    rhol.resize(this->degSpher+2, 0.0);
    rhol[0] = this->muBody/r;
    rhol[1] = rhol[0]*rho;

    // Degree 0

    // Gravity field and potential of degree l = 0
    // Gravity components
    a1 = 0.0;
    a2 = 0.0;
    a3 = 0.0;
    a4 = 0.0;

    for (unsigned int l = 1; l <= this->degSpher; l++) // does not include l = maxDegree
    {
        rhol[l+1] =  rho * rhol[l]; // rho_l computed

        sum_a1 = 0.0;
        sum_a2 = 0.0;
        sum_a3 = 0.0;
        sum_a4 = 0.0;

        for(unsigned int m = 0; m <= l; m++)
        {
            double D, E, F;
            D = cBar(l,m) * rE[m] + sBar(l,m) * iM[m];
            if (m == 0)
            {
                E = 0.0;
                F = 0.0;
            }
            else
            {
                E = cBar(l,m) * rE[m-1] + sBar(l,m) * iM[m-1];
                F = sBar(l,m) * rE[m-1] - cBar(l,m) * iM[m-1];
            }

            //            if (l < degree)   // Gravity contains up to max_degree-1 harmonics
            //            {
            sum_a1 = sum_a1 + m * this->aBar[l][m] * E;
            sum_a2 = sum_a2 + m * this->aBar[l][m] * F;
            if (m < l)
            {
                sum_a3 = sum_a3 + this->nQuot1[l][m] * this->aBar[l][m+1] * D;
            }
            sum_a4 = sum_a4 + this->nQuot2[l][m] * this->aBar[l+1][m+1] * D;
            //            }

        }

        //        if (l < degree)   // Gravity contains up to max_degree-1 harmonics
        //        {
        a1 = a1 + rhol[l+1]/this->radEquator * sum_a1;
        a2 = a2 + rhol[l+1]/this->radEquator * sum_a2;
        a3 = a3 + rhol[l+1]/this->radEquator * sum_a3;
        a4 = a4 - rhol[l+1]/this->radEquator * sum_a4;
        //        }
    }

    acc[0] = a1 + s * a4;
    acc[1] = a2 + t * a4;
    acc[2] = a3 + u * a4;

    return acc;
}

Eigen::VectorXd SpherharmDMCUKF::CSmat2CSvec(Eigen::MatrixXd Cmat, Eigen::MatrixXd Smat, Eigen::VectorXd CSad){
    /* Initialize spherical harmonics vector */
    Eigen::VectorXd CSvec;
    CSvec.setZero(this->nCS);
    
    /* Compute initial vector */
    int contC = 0;
    int contS = 0;
    for (unsigned int i = 2; i <= this->degSpher; i++){
        for (unsigned int j = 0; j <= i; j++){
            if (i != 2 or j != 1){
                CSvec[contC] = Cmat(i,j)/CSad(i-2);
                contC += 1;
                if (j > 0){
                    CSvec[this->nC+contS] = Smat(i,j)/CSad(i-2);
                    contS += 1;
                }
            }
        }
    }
    return CSvec;
}

void SpherharmDMCUKF::CSvec2CSmat(Eigen::VectorXd CSvec, Eigen::VectorXd CSad)
{
    /* Initialize counters */
    int contC = 0;
    int contS = 0;
    
    /* Transform CS vector to matrices */
    for (unsigned int i = 2; i <= this->degSpher; i++){
        for (unsigned int j = 0; j <= i; j++){
            if (i != 2 or j != 1){
                this->cBar(i,j) = CSvec[contC]*CSad(i-2);
                contC += 1;
                if (j > 0){
                    this->sBar(i,j) = CSvec[this->nC+contS]*CSad(i-2);
                    contS += 1;
                }
            }
        }
    }
}


MasconDMCUKF::MasconDMCUKF()
{
    return;
}

MasconDMCUKF::~MasconDMCUKF()
{
    return;
}

void MasconDMCUKF::initializeParameters()
{
    /* Set number of mascons */
    this->nM = this->muM.size();
    
    return;
}

/*! This method computes gravity field
    @param CurrentSimNanos
    @return void
*/
Eigen::Vector3d MasconDMCUKF::computeField(const Eigen::Vector3d pos_Pfix)
{
    /* Preallocate variables */
    Eigen::Vector3d posMii;
    Eigen::Vector3d dr;
    Eigen::Vector3d acc;
    acc.setZero(3);
    
    /* Loop through mascons */
    for (unsigned int i = 0; i < this->nM; i++)
    {
        /* Relative position with each mascons */
        posMii = this->posM.row(i);
        dr = pos_Pfix - posMii;
        acc += -this->muM(i)*dr/pow(dr.norm(),3);
    }
    return acc;
}
