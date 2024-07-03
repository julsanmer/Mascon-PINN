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


#include "gravEst.h"
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
GravEst::GravEst()
{
    this->useNAGD = false;
    this->useAdam = true;
    this->useAdagrad = false;
    this->stop = false;
    this->beta1 = 0.9;
    this->beta2 = 0.99;
    this->eta = 0.9;
    this->w = 1e-10;
    return;
}

/*! Module Destructor */
GravEst::~GravEst()
{
    return;
}

/*! Initialize C-wrapped output messages */
void GravEst::SelfInit(){
}

/*! This method is used to reset the module, check that required input messages are connect and compute weigths.
    @return void
*/
void GravEst::Reset(uint64_t CurrentSimNanos)
{
}

void GravEst::UpdateState(uint64_t CurrentSimNanos)
{
}

/*! This method updates gravity solution
    @param CurrentSimNanos
    @return void
*/
void GravEst::trainGravity(Eigen::MatrixXd posData, Eigen::MatrixXd accData, Eigen::VectorXd W){
    /* Start measuring time */
    clock_t start, end;
    start = clock();
    
    /* Preallocate objective function and data */
    this->L.setZero(this->maxIter);
    
    /* Do z-score normalization */
    this->preprocessData(posData, accData);
    
    /* Declare variables */
    Eigen::VectorXd x0, x, upd0, upd, dLdacc, gradL, xMod;
    Eigen::VectorXd gradLsum, mt, mt0, vt, vt0, mtmod, vtmod;
    double dmu;
    
    /* Declare specific variables for mascons */
    Eigen::VectorXd sqrtmuM0, muM0, posM0Vec;
    Eigen::MatrixXd posM0, posM0T;
    
    /* According to model */
    Eigen::MatrixXd daccdx, daccdmuM;
    if (this->useSH == true){
        /* Initial condition */
        x0 = this->spherharm.CSmat2CSvec();
        upd0.setZero(this->spherharm.nCS);
        gradLsum.setZero(this->spherharm.nCS);
        
        /* Compute Jacobian */
        daccdx = this->spherharm.daccdCS(this->posDataVec, this->nData, this->accStd);
    }
    else if (this->useM == true){
        /* Set conditions */
        this->mascon.useMSE = this->useMSE;
        this->mascon.useMLE = this->useMLE;
        
        if (this->mascon.MU == true){
            /* Initial condition */
            x0 = sqrt((this->mascon.muM.segment(1,this->mascon.nM-1)/this->mascon.muMad).array());
            upd0.setZero(this->mascon.nM-1);
            gradLsum.setZero(this->mascon.nM-1);
            
            /* Compute Jacobian */
            daccdmuM = this->mascon.daccdmu(this->posDataVec, this->nData, this->accNormVec);
        }
        else if (this->mascon.MUPOS == true){
            upd0.setZero(4*(this->mascon.nM-1));
            
            /* Preallocate current variables */
            muM0 = this->mascon.muM.segment(1,this->mascon.nM-1)/this->mascon.muMad;
            sqrtmuM0 = sqrt(muM0.array());
            posM0 = this->mascon.posM.block(1,0,this->mascon.nM-1,3);
            
            /* Stack mascon position in a vector */
            posM0T = posM0.transpose();
            Eigen::Map<Eigen::RowVectorXd> posM0VecRow(posM0T.data(), posM0T.size());
            Eigen::VectorXd posM0Vec = posM0VecRow.transpose().array()/this->mascon.posMadVec.array();
            x0.setZero(4*(this->mascon.nM-1));
            x0.segment(0,this->mascon.nM-1) = sqrt(muM0.array());
            x0.segment(this->mascon.nM-1,3*(this->mascon.nM-1)) = posM0Vec;
        }
    }
    
    if (this->useAdam == true){
        mt0.setZero(x0.rows());
        vt0.setZero(x0.rows());
    }
    
    /* Training loop */
    for (unsigned int i=0; i < this->maxIter; i++){
        /*std::cout << i << "\n";*/
        /*if (i % 10000 == 0){
         std::cout << i << "\n";
         }*/
        
        /* Compute loss function gradient */
        if (this->useSH == true){
            gradL = this->spherharm.gradlossCS(daccdx, x0, this->accDataVec, this->accMeanVec, this->accStdVec, this->L0);
            this->L(i) = this->spherharm.lossCS(daccdx, x0, this->accDataVec, this->accMeanVec, this->accStdVec, this->L0);
        }
        else if (this->useM == true){
            if (this->mascon.MU == true){
                daccdx = this->mascon.daccdsqrmu(daccdmuM, x0, this->nData);
                dLdacc = this->mascon.dlossdacc(daccdx, x0, this->nData, this->accDataVec);
                gradL = this->mascon.gradlossMU(daccdx, dLdacc, this->nData);
                this->L(i) = this->mascon.loss(daccdx, x0, this->nData, this->accDataVec);
            }
            else if (this->mascon.MUPOS == true){
                daccdx = this->mascon.daccdsqrmupos(x0, this->posDataVec, this->nData, this->accNormVec);
                dLdacc = this->mascon.dlossdacc(daccdx.block(0,0,3*this->nData,this->mascon.nM-1), x0.segment(0,this->mascon.nM-1), this->nData, this->accDataVec);
                gradL = this->mascon.gradlossMUPOS(daccdx, dLdacc, this->nData);
                /*std::cout << gradL << '\n';*/
                /*if (i == 0){
                    std::cout << gradL;
                }*/
                this->L(i) = this->mascon.loss(daccdx.block(0,0,3*this->nData,this->mascon.nM-1), x0.segment(0,this->mascon.nM-1), this->nData, this->accDataVec);
            }
        }
        
        /* Choose algorithm */
        if (this->useNAGD == true){
            upd = this->eta*upd0 + this->lam*gradL;
            
            /* Do update */
            x = x0 - upd;
        }
        else if (this->useAdagrad == true){
            gradLsum.array() += gradL.array()*gradL.array();
            upd = this->lam*gradL.array()/(sqrt(gradLsum.array())+1e-6);
            
            /* Do update */
            x = x0 - upd;
        }
        else if (this->useAdam == true){
            mt = this->beta1*mt0 + (1-this->beta1)*gradL;
            vt = this->beta2*vt0.array() + (1-this->beta2)*gradL.array()*gradL.array();
            mtmod = mt / (1-pow(this->beta1,i+1));
            vtmod = vt / (1-pow(this->beta2,i+1));
            upd = this->lam*mtmod.array()/(sqrt(vtmod.array())+1e-6);
            mt0 = mt;
            vt0 = vt;
            
            /* Do update */
            x = x0 - upd;
        }
        else if (this->useLevMar == true){
            bool next;
            next = false;
            double Lnew;
            while (next==false){
                Eigen::MatrixXd invLevMar, LerMar, identity(4*(this->mascon.nM-1),4*(this->mascon.nM-1));
                identity.setIdentity();
                LerMar = daccdx.transpose()*daccdx;
                invLevMar = (LerMar + this->lam*identity).inverse();
                /*upd = invLevMar.colPivHouseholderQr().solve(daccdx.transpose()*(this->mascon.accVec - this->accDataVec));*/
                upd = invLevMar*(daccdx.transpose()*(this->mascon.accVec - this->accDataVec));
                x = x0 - upd;
                
                daccdx = this->mascon.daccdsqrmupos(x, this->posDataVec, this->nData, this->accStd);
                Lnew = this->mascon.loss(daccdx, x, this->nData, this->accDataVec);
                std::cout << Lnew << '\n';
                if (Lnew < this->L(i)){
                    this->lam /= 3;
                    next = true;
                }
                else{
                    this->lam *= 2;
                }
            }
        }
        /*std::cout << i << '\n';*/
        
        

        /*if (i == 0){
         std::cout << this->trainData.x0.segment(this->trainData.nM-1,3*(this->trainData.nM-1)).array()*this->trainData.posMadVec.array();
         }*/

        
        /* Check mass violation */
        if (this->useM == true){
            /* Check interior points if needed */
            if (this->mascon.MUPOS == true){
                xMod = this->poly.interiorConstraint(x.segment(this->mascon.nM-1,3*(this->mascon.nM-1)).array()*this->mascon.posMadVec.array(),this->mascon.nM-1);
                x.segment(this->mascon.nM-1,3*(this->mascon.nM-1)) = xMod.array()/this->mascon.posMadVec.array();
            }
            
            /* Check mass excess */
            dmu = sqrt(x.segment(0,this->mascon.nM-1).transpose()*x.segment(0,this->mascon.nM-1)) - sqrt(this->mascon.mu/this->mascon.muMad);
            if (dmu > 0){
                x.segment(0,this->mascon.nM-1) -= dmu*x.segment(0,this->mascon.nM-1)/x.segment(0,this->mascon.nM-1).norm();
            }
        }
        
        /* Prepare next iteration */
        x0 = x;
        upd0 = upd;
        
        if (i > 1 and this->stop == true){
            if (abs(this->L(i) - this->L(i-1))/this->L(i) < this->tolStop){
                break;
            }
        }
    }
    
    /* Translate variables to DMC-UKF */
    if (this->useSH == true){
        /* Transform to C,S matrices */
        this->spherharm.CSvec2CSmat(x0, this->spherharm.CSad);
    }
    else if (this->useM == true){
        if (this->mascon.MU == true){
            /* Transform to muM */
            muM0 = x0.array()*x0.array()*this->mascon.muMad;
            this->mascon.muM << this->mascon.mu - muM0.sum(), muM0;
        }
        else if (this->mascon.MUPOS == true){
            /* */
            sqrtmuM0 = x0.segment(0,this->mascon.nM-1);
            muM0 = sqrtmuM0.array()*sqrtmuM0.array()*this->mascon.muMad;
            
            /* Unfold positions into matrix */
            posM0Vec = x0.segment(this->mascon.nM-1,3*(this->mascon.nM-1)).array()*this->mascon.posMadVec.array();
            posM0T = Eigen::Map<Eigen::MatrixXd>(posM0Vec.transpose().data(),3,this->mascon.nM-1);
            posM0 = posM0T.transpose();
            
            this->mascon.muM << this->mascon.mu - muM0.sum(), muM0;
            this->mascon.posM.block(1,0,this->mascon.nM-1,3) = posM0;
        }
    }
    
    /* Stop measuring time and calculate the elapsed time */
    end = clock();
    this->tcpu = double(end - start) / double(CLOCKS_PER_SEC);
}
                        
void GravEst::preprocessData(Eigen::MatrixXd posData_i, Eigen::MatrixXd accData_i)
{
    /* Count number of data */
    this->nData = int(accData_i.rows());

    /* Eliminate 0th mass */
    if (this->useM == true){
        Eigen::Vector3d pos_k;
        double normpos_k;
        for (unsigned int i=0; i < this->nData; i++){
            pos_k = posData_i.row(i);
            normpos_k = pos_k.norm();
            accData_i.row(i) += this->mascon.mu*pos_k/pow(normpos_k,3);
        }
    }
    
    /* Preallocate terms */
    this->accNormVec.setZero(this->nData);
    this->posDataVec.setZero(3*this->nData);
    this->accDataVec.setZero(3*this->nData);
    for (unsigned int i = 0; i < this->nData; i++){
        /* Compute norm */
        this->accNormVec(i) = accData_i.row(i).norm();
        
        /* Fill */
        this->posDataVec.segment(3*i,3) = posData_i.row(i).transpose();
        this->accDataVec.segment(3*i,3) = accData_i.row(i).transpose()/this->accNormVec(i);
    }
    
    /* Fill if mascon position */
    if (this->mascon.MUPOS == true){
        this->mascon.posMadVec.setZero(3*(this->mascon.nM-1));
        for (unsigned int k = 0; k < this->mascon.nM-1; k++){
            this->mascon.posMadVec.segment(3*k,3) = this->mascon.posMad;
        }
    }
}


SpherharmGravEst::SpherharmGravEst()
{
    return;
}

SpherharmGravEst::~SpherharmGravEst()
{
    return;
}

void SpherharmGravEst::initializeParameters()
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
            aRow[i] = sqrt(double((2*i+1)*this->getK(i))/(2*i*this->getK(i-1)))*this->aBar[i-1][i-1];
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
        this->n1.push_back(n1Row);
        this->n2.push_back(n2Row);
        this->aBar.push_back(aRow);
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
                nq1Row[m] = sqrt(double((l-m)*this->getK(m)*(l+m+1))/this->getK(m+1));
            }
            nq2Row[m] = sqrt(double((l+m+2)*(l+m+1)*(2*l+1)*this->getK(m))/((2*l+3)*this->getK(m+1)));
        }
        this->nQuot1.push_back(nq1Row);
        this->nQuot2.push_back(nq2Row);
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

/*
@brief Computes the term (2 - d_l), where d_l is the kronecker delta.
*/
double SpherharmGravEst::getK(const unsigned int degree)
{
    return ((degree == 0) ? 1.0 : 2.0);
}

Eigen::MatrixXd SpherharmGravEst::daccdCS(Eigen::VectorXd posDataVec, int nData, Eigen::Vector3d accStd)
{
    /* Preallocate Jacobian */
    Eigen::MatrixXd daccdCS;
    daccdCS.setZero(3*nData,this->nCS);
    
    /* Declare auxiliary variables*/
    Eigen::Vector3d pos;
    double x, y, z;
    double r, s, t, u;
    
    double rho;
    Eigen::VectorXd rE, iM, rhol;
    
    double dsa1dCij, dsa2dCij, dsa3dCij, dsa4dCij;
    double dsa1dSij, dsa2dSij, dsa3dSij, dsa4dSij;
    double da0dCij, da1dCij, da2dCij;
    double da0dSij, da1dSij, da2dSij;
    
    Eigen::Vector3d dadCij;
    Eigen::Vector3d dadSij;
    
    /* Loop through positions */
    for (unsigned int k = 0; k < nData; k++){
        /* Reinitialize these variables*/
        rE.setZero(this->degSpher+2);
        iM.setZero(this->degSpher+2);
        rhol.setZero(this->degSpher+2);
        
        /* Extract position coordiantes */
        pos = posDataVec.segment(3*k,3);
        x = pos[0];
        y = pos[1];
        z = pos[2];
        
        /* Change variables: direction cosines*/
        r = sqrt(x*x + y*y + z*z);
        s = x/r;
        t = y/r;
        u = z/r;
        
        /* Fill low diagonal terms of aBar */
        for (unsigned int i = 1; i <= this->degSpher+1; i++){
            this->aBar[i][i-1] = sqrt((2*i)*this->getK(i-1)/this->getK(i))*this->aBar[i][i]*u;
        }
        
        /* Fill lower terms of aBar, rE and iM */
        for (unsigned int j = 0; j <= this->degSpher+1; j++){
            for (unsigned int i = j+2; i <= this->degSpher+1; i++){
                this->aBar[i][j] = u*this->n1[i][j]*this->aBar[i-1][j] - this->n2[i][j]*this->aBar[i-2][j];
            }
            if (j == 0){
                rE[j] = 1.0;
                iM[j] = 0.0;
            }
            else{
                rE[j] = s*rE[j-1] - t*iM[j-1];
                iM[j] = s*iM[j-1] + t*rE[j-1];
            }
        }
        
        /* Initialize rhol */
        rho = this->radEquator/r;
        rhol[0] = this->muBody/r;
        rhol[1] = rhol[0]*rho;
        rhol[2] = rhol[1]*rho;

        /* Set counters */
        int cont_C = 0;
        int cont_S = 0;
        
        /* Loop through degree */
        for (unsigned int i = 2; i <= this->degSpher; i++){
            /* Add term to rhol */
            rhol[i+1] = rho * rhol[i];

            /* Loop through order */
            for (unsigned int j = 0; j <= i; j++){
                if (j == 0){
                    dsa1dCij = 0.;
                    dsa1dSij = 0.;
                    dsa2dCij = 0.;
                    dsa2dSij = 0.;
                }
                else{
                    dsa1dCij = j*this->aBar[i][j]*rE[j-1];
                    dsa1dSij = j*this->aBar[i][j]*iM[j-1];
                    dsa2dCij = j*this->aBar[i][j]*(-iM[j-1]);
                    dsa2dSij = j*this->aBar[i][j]*rE[j-1];
                }

                if (j < i){
                    dsa3dCij = this->nQuot1[i][j]*this->aBar[i][j+1]*rE[j];
                    dsa3dSij = this->nQuot1[i][j]*this->aBar[i][j+1]*iM[j];
                }
                else{
                    dsa3dCij = 0.;
                    dsa3dSij = 0.;
                }
                dsa4dCij = this->nQuot2[i][j]*this->aBar[i+1][j+1]*rE[j];
                dsa4dSij = this->nQuot2[i][j]*this->aBar[i+1][j+1]*iM[j];

                da0dCij = (dsa1dCij - s*dsa4dCij)*rhol[i+1]/this->radEquator;
                da0dSij = (dsa1dSij - s*dsa4dSij)*rhol[i+1]/this->radEquator;
                da1dCij = (dsa2dCij - t*dsa4dCij)*rhol[i+1]/this->radEquator;
                da1dSij = (dsa2dSij - t*dsa4dSij)*rhol[i+1]/this->radEquator;
                da2dCij = (dsa3dCij - u*dsa4dCij)*rhol[i+1]/this->radEquator;
                da2dSij = (dsa3dSij - u*dsa4dSij)*rhol[i+1]/this->radEquator;
                
                /* Do not fill 2nd order terms */
                if (i != 2 or j != 1){
                    daccdCS.block(3*k,cont_C,3,1) << da0dCij/accStd(0)*this->CSad(i-2), da1dCij/accStd(1)*this->CSad(i-2), da2dCij/accStd(2)*this->CSad(i-2);
                    cont_C += 1;
                    if (j > 0){
                        daccdCS.block(3*k,this->nC+cont_S,3,1) << da0dSij/accStd(0)*this->CSad(i-2), da1dSij/accStd(1)*this->CSad(i-2), da2dSij/accStd(2)*this->CSad*(i-2);
                        cont_S += 1;
                }
                }
            }
        }
    }
    return daccdCS;
}

Eigen::VectorXd SpherharmGravEst::gradlossCS(Eigen::MatrixXd daccdCS, Eigen::VectorXd CSVec, Eigen::VectorXd accDataVec, Eigen::VectorXd accMeanVec, Eigen::VectorXd accStdVec, double L0)
{
    /* Predict acceleration */
    Eigen::VectorXd accVec, gradL, daccdmuM_k, daccdxM_k, daccdyM_k, daccdzM_k;
    accVec = (daccdCS*CSVec).array() - accMeanVec.array()/accStdVec.array();
    gradL = 2*daccdCS.transpose()*(accVec - accDataVec)/L0;
    
    return gradL;
}

double SpherharmGravEst::lossCS(Eigen::MatrixXd daccdCS, Eigen::VectorXd CSVec, Eigen::VectorXd accDataVec, Eigen::VectorXd accMeanVec, Eigen::VectorXd accStdVec, double L0)
{
    /* Compute loss function */
    double L;
    Eigen::VectorXd accVec;
    accVec = (daccdCS*CSVec).array() - accMeanVec.array()/accStdVec.array();
    L = (accVec-accDataVec).transpose()*(accVec-accDataVec);
    
    return L/L0;
}

Eigen::VectorXd SpherharmGravEst::CSmat2CSvec(){
    /* Initialize spherical harmonics vector */
    Eigen::VectorXd CSvec;
    CSvec.setZero(this->nCS);
    
    /* Compute initial vector */
    int contC = 0;
    int contS = 0;
    for (unsigned int i = 2; i <= this->degSpher; i++){
        for (unsigned int j = 0; j <= i; j++){
            if (i != 2 or j != 1){
                CSvec[contC] = this->cBar(i,j)/this->CSad(i-2);
                contC += 1;
                if (j > 0){
                    CSvec[this->nC+contS] = this->sBar(i,j)/this->CSad(i-2);
                    contS += 1;
                }
            }
        }
    }
    return CSvec;
}

void SpherharmGravEst::CSvec2CSmat(Eigen::VectorXd CSvec, Eigen::VectorXd CSad)
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


MasconGravEst::MasconGravEst()
{
    return;
}

MasconGravEst::~MasconGravEst()
{
    return;
}

Eigen::Vector3d MasconGravEst::computeField(Eigen::Vector3d pos)
{
    /* Declare auxiliary variables */
    Eigen::Vector3d dri, posMi, acc;
    acc.setZero(3);
    
    /* Loop through mascon */
    for (unsigned int i = 0; i < this->nM; i++){
        /* Relative position with each mascons */
        posMi = this->posM.row(i);
        dri = pos - posMi;
        acc += -this->muM(i)*dri/pow(dri.norm(),3);
    }

    return acc;
}

Eigen::MatrixXd MasconGravEst::daccdsqrmupos(Eigen::VectorXd sqrmuposMVec, Eigen::VectorXd posDataVec, int nData, Eigen::VectorXd accNormVec)
{
    /* Preallocate dimensional variables */
    Eigen::VectorXd sqrtmuM, muM, posM;
    sqrtmuM = sqrmuposMVec.segment(0,this->nM-1)*sqrt(this->muMad);
    muM = sqrtmuM.array()*sqrtmuM.array();
    posM = sqrmuposMVec.segment(this->nM-1,3*(this->nM-1)).array()*this->posMadVec.array();

    /* Preallocate Jacobian */
    Eigen::MatrixXd daccdsqrmuposM;
    daccdsqrmuposM.setZero(3*nData,4*(this->nM-1));
    
    /* Declare auxiliary variables*/
    Eigen::Vector3d posData_j, posM_k, dr_jk, daccdsqrmuM_jk;
    double sqrtmuM_k, muM_k, normdr_jk, normdr3_jk, normdr5_jk, normpos3_j, accNorm_j;
    Eigen::Vector3d daxdposM_jk, daydposM_jk, dazdposM_jk;
    
    /* Loop through data */
    for (unsigned int j = 0; j < nData; j++){
        /* Extract data position */
        posData_j = posDataVec.segment(3*j,3);
        normpos3_j = pow(posData_j.norm(),3);
        accNorm_j = accNormVec(j);
        
        /* Loop through mascons */
        for (unsigned int k = 0; k < this->nM-1; k++){
            /* Extract mascon position and mass */
            posM_k = posM.segment(3*k,3);
            muM_k = muM(k);
            sqrtmuM_k = sqrtmuM(k);
            
            /* Compute relative position */
            dr_jk = posData_j - posM_k;
            normdr_jk = dr_jk.norm();
            normdr3_jk = pow(normdr_jk,3);
            normdr5_jk = pow(normdr_jk,5);
            
            /* Compute terms for mass (añadir anchor mass) */
            daccdsqrmuM_jk = 2*sqrtmuM_k*(-dr_jk/normdr3_jk + posData_j/normpos3_j)*sqrt(this->muMad);
            
            /* Compute terms for position */
            daxdposM_jk << (1/normdr3_jk - 3*dr_jk(0)*dr_jk(0)/normdr5_jk)*this->posMad(0), -3*dr_jk(0)*dr_jk(1)/normdr5_jk*this->posMad(1), -3*dr_jk(0)*dr_jk(2)/normdr5_jk*this->posMad(2);
            daydposM_jk << -3*dr_jk(0)*dr_jk(1)/normdr5_jk*this->posMad(0), (1/normdr3_jk -3*dr_jk(1)*dr_jk(1)/normdr5_jk)*this->posMad(1), -3*dr_jk(1)*dr_jk(2)/normdr5_jk*this->posMad(2);
            dazdposM_jk << -3*dr_jk(0)*dr_jk(2)/normdr5_jk*this->posMad(0), -3*dr_jk(1)*dr_jk(2)/normdr5_jk*this->posMad(1), (1/normdr3_jk -3*dr_jk(2)*dr_jk(2)/normdr5_jk)*this->posMad(2);
            
            /* Fill block */
            daccdsqrmuposM.block(3*j,k,3,1) << daccdsqrmuM_jk.array()/accNorm_j;
            daccdsqrmuposM.block(3*j,this->nM-1+3*k,1,3) << muM_k*daxdposM_jk.array()/accNorm_j;
            daccdsqrmuposM.block(3*j,this->nM-1+3*k+1,1,3) << muM_k*daydposM_jk.array()/accNorm_j;
            daccdsqrmuposM.block(3*j,this->nM-1+3*k+2,1,3) << muM_k*dazdposM_jk.array()/accNorm_j;
        }
    }
    return daccdsqrmuposM;
}

Eigen::VectorXd MasconGravEst::dlossdacc(Eigen::MatrixXd daccdsqrmuM, Eigen::VectorXd sqrmuM, int nData, Eigen::VectorXd accDataVec)
{
    /* Preallocate variables */
    Eigen::VectorXd dLdacc, accVec;
    Eigen::VectorXd dacc;
    accVec = (daccdsqrmuM*sqrmuM/2).array();
    dLdacc.setZero(3*nData);
    dacc = accVec - accDataVec;
    
    /* Fill matrix */
    if (this->useMLE){
        for (unsigned int k = 0; k < nData; k++){
            /* Compute acceleration difference and fill */
            dLdacc.segment(3*k,3) = dacc.segment(3*k,3)/dacc.segment(3*k,3).norm();
        }
    }
    else if (this->useMSE == true){
        dLdacc = 2*dacc;
    }
    return dLdacc;
}

Eigen::VectorXd MasconGravEst::gradlossMUPOS(Eigen::MatrixXd daccdsqrmuposM, Eigen::VectorXd dLdacc, int nData)
{
    /* Predict acceleration */
    Eigen::VectorXd gradL, daccdsqrmuM_k, daccdxM_k, daccdyM_k, daccdzM_k;
    gradL.setZero(4*(this->nM-1));

    /* Compute gradient */
    for (unsigned int k = 0; k < this->nM-1; k++){
        /* Extract columns */
        daccdsqrmuM_k = daccdsqrmuposM.col(k);
        daccdxM_k = daccdsqrmuposM.col(this->nM-1+3*k);
        daccdyM_k = daccdsqrmuposM.col(this->nM-1+3*k+1);
        daccdzM_k = daccdsqrmuposM.col(this->nM-1+3*k+2);
        
        /* Fill gradient */
        gradL.segment(k,1) << (daccdsqrmuM_k.array()*dLdacc.array()).sum();
        gradL.segment(this->nM-1+3*k,3) << (daccdxM_k.array()*dLdacc.array()).sum(),
            (daccdyM_k.array()*dLdacc.array()).sum(),
            (daccdzM_k.array()*dLdacc.array()).sum();
    }
    /*std::cout <<  gradL/nData << '\n';*/
    return gradL/nData;
}

double MasconGravEst::loss(Eigen::MatrixXd daccdsqrmuM, Eigen::VectorXd sqrmuM, int nData, Eigen::VectorXd accDataVec)
{
    /* Compute loss function */
    double L;
    Eigen::VectorXd accVec;
    accVec = (daccdsqrmuM*sqrmuM/2).array();
    L = 0;
    if (this->useMLE == true){
        L = 0;
        for (unsigned int i=0; i < nData; i++){
            L += (accVec.segment(3*i,3)-accDataVec.segment(3*i,3)).norm();
        }
    }
    else if (this->useMSE == true){
        L = (accVec-accDataVec).transpose()*(accVec-accDataVec);
    }
    
    return L/nData;
}

Eigen::MatrixXd MasconGravEst::daccdmu(Eigen::VectorXd posDataVec, int nData, Eigen::VectorXd accNormVec)
{
    /* Preallocate Jacobian */
    Eigen::MatrixXd daccdmuM;
    daccdmuM.setZero(3*nData,this->nM-1);
    
    /* Declare auxiliary variables*/
    Eigen::Vector3d posData_j, posM_k, dr_jk, daccdmuM_jk;
    double normdr_jk, normdr3_jk, normpos3_j, accNorm_j;
    
    /* Loop through data */
    for (unsigned int j = 0; j < nData; j++){
        /* Extract data position */
        posData_j = posDataVec.segment(3*j,3);
        normpos3_j = pow(posData_j.norm(),3);
        accNorm_j = accNormVec(j);
        
        /* Loop through mascons */
        for (unsigned int k = 0; k < this->nM-1; k++){
            /* Extract mascon position and mass */
            posM_k = this->posM.row(k+1).transpose();
            
            /* Compute relative position */
            dr_jk = posData_j - posM_k;
            normdr_jk = dr_jk.norm();
            normdr3_jk = pow(normdr_jk,3);
            
            /* Compute terms for mass (añadir anchor mass) */
            daccdmuM_jk = -dr_jk/normdr3_jk + posData_j/normpos3_j;
            
            /* Fill block */
            daccdmuM.block(3*j,k,3,1) << daccdmuM_jk.array()/accNorm_j;
        }
    }
    return daccdmuM;
}

Eigen::MatrixXd MasconGravEst::daccdsqrmu(Eigen::MatrixXd daccdmuM, Eigen::VectorXd sqrmuM, int nData)
{
    /* Preallocate Jacobian */
    Eigen::MatrixXd daccdsqrmuM;
    daccdsqrmuM.setZero(3*nData,this->nM-1);
    sqrmuM *= sqrt(this->muMad);
    
    /* Loop through data */
    for (unsigned int k = 0; k < this->nM-1; k++){
        daccdsqrmuM.col(k) = 2*daccdmuM.col(k)*sqrmuM(k)*sqrt(this->muMad);
    }
    return daccdsqrmuM;
}

Eigen::VectorXd MasconGravEst::gradlossMU(Eigen::MatrixXd daccdsqrmuM, Eigen::VectorXd dLdacc, int nData)
{
    /* Predict acceleration */
    Eigen::VectorXd gradL, daccdsqrmuM_k;
    gradL.setZero(this->nM-1);

    /* Compute gradient */
    for (unsigned int k = 0; k < this->nM-1; k++){
        /* Extract columns */
        daccdsqrmuM_k = daccdsqrmuM.col(k);
        
        /* Fill gradient */
        gradL.segment(k,1) << (daccdsqrmuM_k.array()*dLdacc.array()).sum();
    }
    return gradL/nData;
}

PolyGravEst::PolyGravEst()
{
    return;
}

PolyGravEst::~PolyGravEst()
{
    return;
}

void PolyGravEst::initializeParameters()
{
    int i, j, k;
    Eigen::Vector3d v, xyz1, xyz2, xyz3, e21, e32;
    
    /* Initialize volume and normal */
    this->volPoly = 0.0;
    this->normalFacet.setZero(this->nFacet,3);
    this->xyzFacet.setZero(this->nFacet,3);
    
    /* Loop through each facet to compute volume */
    for (unsigned int m = 0; m < this->nFacet; m++)
    {
        /* Fill auxiliary variables with vertex order on each facet */
        v = this->orderFacet.row(m);
        i = v[0];
        j = v[1];
        k = v[2];
        
        xyz1 = this->xyzVertex.row(i);
        xyz2 = this->xyzVertex.row(j);
        xyz3 = this->xyzVertex.row(k);
        
        this->xyzFacet.row(m) = (xyz1 + xyz2 + xyz3)/3;
        
        /* Compute two edge vectors and normal to facet */
        e21 = xyz2 - xyz1;
        e32 = xyz3 - xyz2;
        this->normalFacet.row(m) = e21.cross(e32) / e21.cross(e32).norm();
        
        /* Add volume contribution */
        this->volPoly += abs(xyz1.cross(xyz2).transpose()*xyz3)/6;
    }
}

Eigen::Vector3d PolyGravEst::computeField(Eigen::Vector3d pos)
{
    int i, j, k;
    Eigen::Vector3d v;
    Eigen::Vector3d ri, rj, rk;
    Eigen::Vector3d nf;
    Eigen::Vector3d r1, r2, re;
    Eigen::Vector3d r21, n21;
    Eigen::Matrix3d Ee;
    
    int idx_min;
    double a, b, e, Le;
    double wy, wx, wf;
    
    Eigen::Vector3d dUe, dUf, acc;
    dUe.setZero(3);
    dUf.setZero(3);
    
    /* Loop through each facet */
    for (unsigned int m = 0; m < this->nFacet; m++){
        /* Fill auxiliary variables with vertex order on each facet */
        v = this->orderFacet.row(m);
        i = v[0];
        j = v[1];
        k = v[2];
        
        /* Compute vectors and norm from each vertex to the evaluation position */
        ri = this->xyzVertex.row(i).transpose() - pos;
        rj = this->xyzVertex.row(j).transpose() - pos;
        rk = this->xyzVertex.row(k).transpose() - pos;
        
        /* Extract normal to facet */
        nf = this->normalFacet.row(m).transpose();
        
        /* Loop through each facet edge */
        for (unsigned int n = 0; n <= 2; n++){
            switch(n){
                case 0:
                    idx_min = fmin(i,j);
                    r1 = ri;
                    r2 = rj;
                    re = this->xyzVertex.row(idx_min).transpose() - pos;
                    
                    a = ri.norm();
                    b = rj.norm();
                    break;
                case 1:
                    idx_min = fmin(j,k);
                    r1 = rj;
                    r2 = rk;
                    re = xyzVertex.row(idx_min).transpose() - pos;
                    
                    a = rj.norm();
                    b = rk.norm();
                    break;
                case 2:
                    idx_min = fmin(i,k);
                    r1 = rk;
                    r2 = ri;
                    re = xyzVertex.row(idx_min).transpose() - pos;
                    
                    a = rk.norm();
                    b = ri.norm();
                    break;
            }
        
            /* Compute along edge vector and norm */
            r21 = r2 - r1;
            e = r21.norm();
            n21 = r21.cross(nf) / r21.cross(nf).norm();
        
            /* Dimensionless per edge factor */
            Le = log((a+b+e) / (a+b-e));
        
            /* Compute dyad product */
            Ee = nf*n21.transpose();
        
            /* Add current facet distribution */
            dUe += Ee*re*Le;
        }
        
        /* Compute solid angle for the current facet */
        wy = ri.transpose()*rj.cross(rk);
        wx = ri.norm()*rj.norm()*rk.norm() + ri.norm()*rj.transpose()*rk
            + rj.norm()*rk.transpose()*ri + rk.norm()*ri.transpose()*rj;
        wf = 2*atan2(wy, wx);
        
        /* Add current solid angle facet */
        dUf += nf*(nf.transpose()*ri)*wf;
    }
    
    /* Compute acceleration contribution */
    acc = (this->muBody/this->volPoly)*(-dUe + dUf);

    return acc;
}

Eigen::VectorXd PolyGravEst::interiorConstraint(Eigen::VectorXd posVec, int nBatch)
{
    /* Preallocate output */
    double lU;
    Eigen::VectorXd posOutput;
    posOutput = posVec;
    
    /* Declare auxiliary variables */
    Eigen::Vector3d pos;
    int i, j, k;
    Eigen::Vector3d v;
    Eigen::Vector3d ri, rj, rk;
    double wy, wx, wf;
    
    /* Keep track of closest point */
    Eigen::Vector3d rFClose, rF;
    double normdrClose, normdrF;
    
    int idx;
    
    /* Loop through all positions */
    for (unsigned int l = 0; l < nBatch; l++){
        /* Extract position */
        pos = posVec.segment(3*l,3);
        
        /* Initialize laplacian */
        lU = 0;
        
        /* Loop through each facet */
        for (unsigned int m = 0; m < this->nFacet; m++){
            /* Fill auxiliary variables with vertex order on each facet */
            v = this->orderFacet.row(m);
            i = v[0];
            j = v[1];
            k = v[2];
            
            /* Compute vectors and norm from each vertex to the evaluation position */
            ri = this->xyzVertex.row(i).transpose() - pos;
            rj = this->xyzVertex.row(j).transpose() - pos;
            rk = this->xyzVertex.row(k).transpose() - pos;
            
            /* Keep track of closest point on surface */
            rF = (this->xyzVertex.row(i).transpose()+this->xyzVertex.row(j).transpose()+this->xyzVertex.row(k).transpose())/3;
            /*std::cout << rF << '\n';*/
            normdrF = (pos - rF).norm();
            if (m == 0){
                rFClose = rF;
                normdrClose = normdrF;
                idx = m;
            }
            else{
                if (normdrF < normdrClose){
                    rFClose = rF;
                    normdrClose = normdrF;
                    idx = m;
                }
            }
            
            /* Compute solid angle for the current facet */
            wy = (ri.transpose())*rj.cross(rk);
            wx = ri.norm()*rj.norm()*rk.norm() + ri.norm()*rj.transpose()*rk + rj.norm()*rk.transpose()*ri + rk.norm()*ri.transpose()*rj;
            wf = -2*atan2(wy, wx);
            
            /* Add term */
            lU += wf;
        }
        /* Check if point is interior or not */
        if (abs(lU) < 2*M_PI_2){
            posOutput.segment(3*l,3) = rFClose;
            /*std::cout << posOutput.segment(3*l, 3) << '\n';*/
            /*std::cout << idx << '\n';*/
        }
    }
    return posOutput;
}

Eigen::VectorXd PolyGravEst::computeLaplacian(Eigen::MatrixXd posBatch)
{
    /* Extract dimensions */
    int nBatch;
    nBatch = int(posBatch.rows());
    
    /* Preallocate output */
    Eigen::VectorXd lU;
    lU.setZero(nBatch);
    
    /* Declare auxiliary variables */
    Eigen::Vector3d pos;
    int i, j, k;
    Eigen::Vector3d v;
    Eigen::Vector3d ri, rj, rk;
    double wy, wx, wf;
        
    /* Loop through all positions */
    for (unsigned int l = 0; l < nBatch; l++){
        /* Extract position */
        pos = posBatch.row(l).transpose();
        
        /* Loop through each facet */
        for (unsigned int m = 0; m < this->nFacet; m++){
            /* Fill auxiliary variables with vertex order on each facet */
            v = this->orderFacet.row(m);
            i = v[0];
            j = v[1];
            k = v[2];
            
            /* Compute vectors and norm from each vertex to the evaluation position */
            ri = this->xyzVertex.row(i).transpose() - pos;
            rj = this->xyzVertex.row(j).transpose() - pos;
            rk = this->xyzVertex.row(k).transpose() - pos;
            
            /* Compute solid angle for the current facet */
            wy = (ri.transpose())*rj.cross(rk);
            wx = ri.norm()*rj.norm()*rk.norm() + ri.norm()*rj.transpose()*rk + rj.norm()*rk.transpose()*ri + rk.norm()*ri.transpose()*rj;
            wf = -2*atan2(wy, wx);
            
            /* Add term */
            lU(l) += wf;
        }
    }
    return lU;
}

double PolyGravEst::computeAltitude(Eigen::Vector3d pos){
    /* Declare distance */
    Eigen::VectorXd distanceFacet;
    distanceFacet.setZero(this->nFacet);
    
    /* Loop through all facets */
    for (unsigned int i = 0; i < this->nFacet; i++){
        /* Compute distance */
        distanceFacet(i) = (pos - xyzFacet.row(i).transpose()).norm();
    }
    
    /* Choose minimum */
    double altitude;
    altitude = distanceFacet.minCoeff();
    
    return altitude;
}
