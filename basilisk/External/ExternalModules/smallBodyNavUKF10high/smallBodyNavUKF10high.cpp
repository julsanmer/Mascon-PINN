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


#include "smallBodyNavUKF10high.h"
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
SmallBodyNavUKF10high::SmallBodyNavUKF10high()
{
    /* Dimensions */
    this->nx = 10;
    
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
SmallBodyNavUKF10high::~SmallBodyNavUKF10high()
{
}

/*! Initialize C-wrapped output messages */
void SmallBodyNavUKF10high::SelfInit(){
    SmallBodyNav2Msg_C_init(&this->smallBodyNav2OutMsgC);
}

/*! This method is used to reset the module, check that required input messages are connect and compute weigths.
    @return void
*/
void SmallBodyNavUKF10high::Reset(uint64_t CurrentSimNanos)
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
        this->mascons.initializeParameters();
    }

    
    /* Set number of tracked landmarks and maximum number measurement */
    this->nz = 3;
    this->resz_k(this->nz);
}


/*! This method is used to read the input messages.
    @return void
*/
void SmallBodyNavUKF10high::readMessages(){
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
void SmallBodyNavUKF10high::predict(){
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
void SmallBodyNavUKF10high::update(){
    /* Generate distributions */
    Dist distx_k1_(this->Pxx_k1_);
    distx_k1_.mean = this->xhat_k1_;
    
    /* Generate sigma points */
    Sigma sig(distx_k1_, this->cx);
    
    /* Compute expected measurements */
    Eigen::MatrixXd Z(this->nz, sig.n_pts), Pzz(this->nz, this->nz), Pzx(this->nz, this->nx), K(this->nx, this->nz), Xu(this->nx, sig.n_pts);
    Eigen::VectorXd xm, zm;
    for (int i = 0; i < sig.n_pts; i++){
        Z.col(i) = this->measurements(sig.state.col(i));
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
    if (this->measMode == 0){
        Pzz += this->dcm_AN.transpose()*this->Pvv*this->dcm_AN;
    }
    else if (this->measMode == 1){
        Pzz += this->Pvv;
    }

    /* Compute Kalman gain */
    K = Pzz.llt().solve(Pzx).transpose();
    /*K = Pzx.transpose()*Pzz.inverse();*/
        
    /* Update state distribution */
    Xu = sig.state - K*(Z.colwise() - zm);
    Dist distXu(Xu, this->wgt_m, this->wgt_c);
    distXu.mean = xm + K*(this->z - zm);
    
    /*std::cout << this->Pxx_k1_ - K*Pzx*this->Pxx_k1_*/
    
    /* Update state */
    this->xhat_k = distXu.mean;
    this->Pxx_k = distXu.cov;
    
    /*this->Pxx_k = (this->Pxx_k1_ - K*Pzz*K.transpose());*/
    
    /*Eigen::VectorXd deltax;
    deltax = K*(this->z - zm);
    std::cout << deltax*deltax.transpose() << '\n';*/
    
    /* Compute residual */
    this->innx_k = K*(this->z - zm);
    this->resz_k = this->measurements(this->xhat_k) - this->z;
    /*std::cout << (this->Pww - K*Pzz*K.transpose() + this->innx_k*this->innx_k
                  .transpose()).diagonal();*/
    this->innx_k = (this->Pww - K*Pzz*K.transpose() + this->innx_k*this->innx_k.transpose()).diagonal();
}

/*! This method propagates the dynamics
    @param CurrentSimNanos
    @return void
*/
Eigen::VectorXd SmallBodyNavUKF10high::dynamics(double t0, double tf, Eigen::VectorXd X){
    /* Compute timestep */
    double dt = ((tf - t0)/this->t_ad) / this->Nint;
    double mu_ast;
    
    /* Declare output */
    Eigen::VectorXd Xp, Xdot, Xproc;
    Xp.setZero(this->nx), Xdot.setZero(this->nx), Xproc.setZero(this->nx);
    
    double lst;
    double theta_AN[3];
    double dcm_AN_array[3][3];
    lst = this->lst0 + this->rotRate*t0;
    v3Set(0,0,lst, theta_AN);
    Euler3232C(theta_AN, dcm_AN_array);
    this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);

    /* Extract variables */
    Eigen::Vector3d r, v, a;
    Eigen::Vector3d a_grav, r_CS, a_SRP, a_Sun;
    double sunDist;
    for (int i = 0; i < this->Nint; i++){
        r << X.segment(0,3);
        v << X.segment(3,3);
        a << X.segment(6,3);
        mu_ast = X(9);
        
        /* Choose gravity model */
        if (this->useSH == true){
            /* Set standard gravity parameter */
            this->spherharm.muBody = mu_ast;
    
            /* Compute inhomogeneous gravity */
            a_grav = this->spherharm.computeField(this->dcm_AN*r);
        }
        if (this->useM == true){
            /* Compute inhomogeneous gravity */
            a_grav = this->mascons.computeField(this->dcm_AN*r);
        }
        
        /* Add solar perturbations */
        a_SRP.setZero(3);
        a_Sun.setZero(3);
        r_CS = this->r_AS_k + r;
        if (this->useSRP == true){
            a_SRP = this->CR*this->ASRP*SOLAR_FLUX_EARTH*pow(AU*1000.,2) / (SPEED_LIGHT*pow(r_CS.norm(),3)*this->m)*(r_CS);
        }
        if (this->useSun == true){
            a_Sun = -(MU_SUN*1e9)*(r_CS/pow(r_CS.norm(),3) - this->r_AS_k/pow(this->r_AS_k.norm(),3));
        }
    
        /* Compute dynamics derivative */
        Xdot.segment(0,3) = v;
        Xdot.segment(3,3) = - mu_ast*r/pow(r.norm(),3)*pow(this->t_ad,2)
                            + a + this->dcm_AN.transpose()*a_grav*pow(this->t_ad,2)
                            + a_SRP + a_Sun;
        if (this->useM == true){
            Xdot.segment(3,3) = a + this->dcm_AN.transpose()*a_grav*pow(this->t_ad,2)
                                + a_SRP + a_Sun;
        }
    
        /* Use Euler integration to propagate */
        X += Xdot*dt;
        lst += this->rotRate*dt;
        
        v3Set(0,0,lst, theta_AN);
        Euler3232C(theta_AN, dcm_AN_array);
        this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);
    }
    
    /* Prepare output */
    Xp = X;
    return Xp;
}


/*! This method transforms state to measurements
    @param CurrentSimNanos
    @return void
*/
Eigen::VectorXd SmallBodyNavUKF10high::measurements(Eigen::VectorXd X){
    /* Declare output */
    Eigen::VectorXd Z;
    Z.setZero(this->nz);
    
    /* Extract position */
    Eigen::Vector3d r;
    r << X.segment(0,3);
    Z = this->dcm_AN*r;

    return Z;
}

SmallBodyNavUKF10high::Sigma::Sigma(const Dist& distX, Eigen::VectorXd cx){
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
SmallBodyNavUKF10high::Dist::Dist(Eigen::MatrixXd& X, Eigen::VectorXd wgt_m, Eigen::VectorXd wgt_c){
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
SmallBodyNavUKF10high::Dist::Dist(Eigen::MatrixXd& S){
    /* Compute mean and covariance */
    n = S.rows();
    mean.setZero(n);
    cov = S;
    covL = S.llt().matrixL();
}

/*! This is the main method that gets called every time the module is updated.
    @return void
*/
void SmallBodyNavUKF10high::updateMeasBatch(Eigen::VectorXd tZ, Eigen::MatrixXd Z, Eigen::MatrixXd PvvZ, Eigen::VectorXd Zsol, Eigen::MatrixXd r_AS, int nSegment){
    /* Obtain number of measurements and t0 */
    int nZ = tZ.size();
    this->tk = tZ(0);

    /* Initialize outputs */
    Eigen::MatrixXd Xhat;
    Eigen::MatrixXd Pxx;
    Eigen::MatrixXd resZ;
    Eigen::MatrixXd innX;
    Xhat.setZero(nZ,this->nx);
    Pxx.setZero(nZ,this->nx*this->nx);
    resZ.setZero(nZ,this->nz);
    innX.setZero(nZ,this->nx);

    /* Initialize filling estimated state and covariance */
    Xhat.row(0) = this->xhat_k.transpose();
    Eigen::Map<Eigen::RowVectorXd> PxxRow(this->Pxx_k.data(), this->Pxx_k.size());
    Pxx.row(0) = PxxRow;
    
    /* Compute rotation matrix */
    double theta_AN[3];
    double dcm_AN_array[3][3];
    double lst;
    lst = this->lst0 + this->rotRate*tZ(0);
    v3Set(0,0,lst, theta_AN);
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
        this->tk1 = tZ(i);
        this->z = Z.row(i).transpose();
        this->zsol = Zsol(i);
        
        /* Get small body w.r.t. Sun */
        this->r_AS_k = r_AS.row(i);
        
        /* Get measurement covariance */
        this->Pvv << PvvZ(i,0), PvvZ(i,1), PvvZ(i,2), PvvZ(i,3), PvvZ(i,4), PvvZ(i,5), PvvZ(i,6), PvvZ(i,7), PvvZ(i,8);
        /*this->Pvv << PvvZ(i,0), 0, 0, 0, PvvZ(i,4), 0, 0, 0, PvvZ(i,8);*/
        /*this->Pvv << 25, 0, 0, 0, 25, 0, 0, 0, 25;*/
            
        /* Update prediction */
        this->predict();
        if (this->zsol == 1){
            this->update();
        }
        else{
            this->xhat_k = this->xhat_k1_;
            this->xhat_k.segment(6,3) << 0, 0, 0;
            this->Pxx_k = this->Pxx_k1_;
            this->resz_k.setZero(3);
            this->innx_k.setZero(10);
        }
        
        /* Fill gravity training */
        rGrav.row(i) = (this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
        aGrav.row(i) = (this->dcm_AN*(this->xhat_k.segment(6,3))).transpose();
        if (this->useSH == true){
            aGrav.row(i) += this->spherharm.computeField(this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
        }
        if (this->useM == true){
            aGrav.row(i) += this->mascons.computeField(this->dcm_AN*this->xhat_k.segment(0,3)).transpose();
        }
                
        /* Save outputs */
        Xhat.row(i) = this->xhat_k.transpose();
        resZ.row(i) = this->resz_k.transpose();
        innX.row(i) = this->innx_k.transpose().array().abs();
        int nWindow;
        nWindow = 50;
        /*if (nSegment == 1 and i==1){
            this->Pww(6,6) = this->Pww(6,6)*0.01;
            this->Pww(7,7) = this->Pww(7,7)*0.01;
            this->Pww(8,8) = this->Pww(8,8)*0.01;
            this->Pww(3,3) -= 0.9*this->Pww(6,6)*pow(60,3)/3;
            this->Pww(4,4) -= 0.9*this->Pww(7,7)*pow(60,3)/3;
            this->Pww(5,5) -= 0.9*this->Pww(8,8)*pow(60,3)/3;
            this->Pww(0,0) -= 0.9*this->Pww(6,6)*pow(60,6)/6;
            this->Pww(1,1) -= 0.9*this->Pww(7,7)*pow(60,6)/6;
            this->Pww(2,2) -= 0.9*this->Pww(8,8)*pow(60,6)/6;
        }*/
        if (nSegment > 0){
            if (i >= nWindow){
                Eigen::MatrixXd Pest;
                Pest.setZero(this->nx,this->nx);
                for (int j=0; j < nWindow; j++){
                    /*if (i == nWindow){
                        std::cout << (innX.block(i-j,0,1,this->nx)) << '\n';
                    }*/
                    Pest.diagonal() += (innX.block(i-j,0,1,this->nx)).transpose()/nWindow; /*innX.block(i-j,0,1,this->nx).transpose()*innX.block(i-j,0,1,this->nx)/nWindow;*/
                }
                /*std::cout << Pest << '\n';*/
                /*this->Pww = Pest;*/
                /*if (nSegment == 1){
                    this->Pww(6,6) = Pest(6,6)*0.1;
                    this->Pww(7,7) = Pest(7,7)*0.1;
                    this->Pww(8,8) = Pest(8,8)*0.1;
                }*/
            }
            /*if (i == 980){
                std::cout << this->Pww << '\n';
            }*/
        }
            
        
        /* Compute rotation matrix */
        this->spherharm.muBody = this->xhat_k(9);
        
        /* Save covariance matrix */
        Eigen::Map<Eigen::RowVectorXd> PxxRow(this->Pxx_k.data(), this->Pxx_k.size());
        Pxx.row(i) = PxxRow;
        
        /* Prepare next iteration */
        this->tk = tZ(i);
        
        /* Stop measuring time and calculate the elapsed time */
        end = clock();
        tcpu(i) = double(end - start) / double(CLOCKS_PER_SEC);
    }
    /* Fill outputs */
    this->stateBatch.tcpu = tcpu;
    this->stateBatch.Xhat = Xhat;
    this->stateBatch.Pxx = Pxx;
    this->stateBatch.resZ = resZ;
    
    this->stateBatch.rGrav = rGrav;
    this->stateBatch.aGrav = aGrav;
}


/*! This method writes the output messages
    @return void
*/
void SmallBodyNavUKF10high::writeMessages(uint64_t CurrentSimNanos){
    /* Create output msg buffers */
    SmallBodyNav2MsgPayload smallBodyNav2OutMsgBuffer;
    
    /* Zero the output message buffers before assigning values */
    smallBodyNav2OutMsgBuffer = this->smallBodyNav2OutMsg.zeroMsgPayload;
    
    /* Assign values to the small body navigation output message */
    eigenMatrixXd2CArray(this->xhat_k, smallBodyNav2OutMsgBuffer.state);
    eigenMatrixXd2CArray(this->Pxx_k, *smallBodyNav2OutMsgBuffer.covar);
    /*eigenMatrixXd2CArray(this->z, smallBodyNav1OutMsgBuffer.meas);*/
    smallBodyNav2OutMsgBuffer.tcpu = this->tcpu;
    
    /* Write to the C++-wrapped output messages */
    this->smallBodyNav2OutMsg.write(&smallBodyNav2OutMsgBuffer, this->moduleID, CurrentSimNanos);

    /* Write to the C-wrapped output messages */
    SmallBodyNav2Msg_C_write(&smallBodyNav2OutMsgBuffer, &this->smallBodyNav2OutMsgC, this->moduleID, CurrentSimNanos);
}

/*! This is the main method that gets called every time the module is updated.
    @return void
*/
void SmallBodyNavUKF10high::UpdateState(uint64_t CurrentSimNanos)
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


StateBatchUKF10high::StateBatchUKF10high()
{
    return;
}

StateBatchUKF10high::~StateBatchUKF10high()
{
    return;
}

SphericalharmonicsUKF10high::SphericalharmonicsUKF10high()
{
    return;
}

SphericalharmonicsUKF10high::~SphericalharmonicsUKF10high()
{
    return;
}

/*
@brief Computes the term (2 - d_l), where d_l is the kronecker delta.
*/
double SphericalharmonicsUKF10high::getK(const unsigned int degree)
{
    return ((degree == 0) ? 1.0 : 2.0);
}

void SphericalharmonicsUKF10high::initializeParameters()
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
Eigen::Vector3d SphericalharmonicsUKF10high::computeField(const Eigen::Vector3d pos_Pfix)
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

Eigen::VectorXd SphericalharmonicsUKF10high::CSmat2CSvec(Eigen::MatrixXd Cmat, Eigen::MatrixXd Smat, Eigen::VectorXd CSad){
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

void SphericalharmonicsUKF10high::CSvec2CSmat(Eigen::VectorXd CSvec, Eigen::VectorXd CSad)
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


MasconsUKF10high::MasconsUKF10high()
{
    return;
}

MasconsUKF10high::~MasconsUKF10high()
{
    return;
}

void MasconsUKF10high::initializeParameters()
{
    /* Set number of mascons */
    this->nM = this->muM.size();
    
    return;
}

/*! This method computes gravity field
    @param CurrentSimNanos
    @return void
*/
Eigen::Vector3d MasconsUKF10high::computeField(const Eigen::Vector3d pos_Pfix)
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
