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

#include <functional>

#include "masconFit.h"
#include "architecture/utilities/linearAlgebra.h"
#include <iostream>
#include <filesystem>
#include <cstring>
#include <math.h>

#include <random>
#include <algorithm>

/*!
 @brief Creates an instance of the mascon fit class that by default trains only masses, uses mean-squared error as loss and dimensional mascon parameters.
 */
MasconFit::MasconFit()
: shapeModel{std::make_shared<PolyhedralShapeModel>()}
{
    // Set mean squared error loss
    this->lossType = "quadratic";
    
    // Set only mass training
    this->trainXYZ = false;
    
    // Set unity adimensionalization
    this->muMad = 1;
    this->xyzMad << 1, 1, 1;
    
    // Check shape model
    std::optional<std::string> errorMessage;
    if (this->shapeModel) {
        this->shapeModel->bskLogger = &bskLogger;
        errorMessage = this->shapeModel->initializeParameters();
    }
    else {
        errorMessage = "Gravity model is null";
    }
}

/*! Empty destructor method.
 */
MasconFit::~MasconFit()
{
}

/*! This method sets maximum iterations for Adam gradient descent.
 @param maxIter: maximum number of iterations
 */
void MasconFit::setMaxIter(unsigned int maxIter)
{
    // Set maximum iterations
    this->graddescent.maxIter = maxIter;
}

/*! This method sets learning rate for Adam gradient descent.
 @param lr: learning rate
 */
void MasconFit::setLR(double lr)
{
    // Set learning rate
    this->graddescent.lr = lr;
}

/*! This method sets batch size for Adam gradient descent.
 @param batchSize: batch size
 */
void MasconFit::setBatchSize(unsigned int batchSize)
{
    // Set batch size
    this->graddescent.batchSize = batchSize;
}

/*! This method sets customized hyperparameters for Adam gradient descent.
    @param beta1: average gradient decay rate
    @param beta2: average squared gradient decay rate
    @param eps: numerical stability constant
 */
void MasconFit::setHyperparam(double beta1, double beta2, double eps)
{
    // Set custom hyperparameters
    this->graddescent.beta1 = beta1;
    this->graddescent.beta2 = beta2;
    this->graddescent.eps = eps;
}

/*! This method prepares the initial decision variable.
    @return x0: initial decision variable
 */
Eigen::VectorXd MasconFit::getInitCond()
{
    // Declare initial decision variable
    Eigen::VectorXd x0;
    
    if (this->trainXYZ){
        // Do square-root of masses
        Eigen::VectorXd muM0 = this->muM.segment(1, this->nM-1) / this->muMad;
        Eigen::VectorXd sqrtmuM0 = sqrt(muM0.array());
            
        // Stack mascon positions in a vector
        Eigen::MatrixXd xyzM0 = this->xyzM.block(1, 0, this->nM-1, 3);
        Eigen::MatrixXd xyzM0T = xyzM0.transpose();
        Eigen::Map<Eigen::RowVectorXd> xyzM0VecRow(xyzM0T.data(), xyzM0T.size());
        Eigen::VectorXd xyzM0Vec = xyzM0VecRow.transpose().array() / this->xyzMadVec.array();
        
        // Set dimension and initial condition in Adam gradient descent
        x0.setZero(4*(this->nM-1));
        x0.segment(0, this->nM-1) = sqrt(muM0.array());
        x0.segment(this->nM-1, 3*(this->nM-1)) = xyzM0Vec;
    }
    else{
        // Set dimension and initial condition in Adam gradient descent
        x0 = sqrt((this->muM.segment(1, this->nM-1) / this->muMad).array());
    }
    
    return x0;
}

/*! This method returns the loss history.
    @return loss:  loss function history
 */
Eigen::VectorXd MasconFit::getLoss()
{
    return this->graddescent.loss;
}

/*! This method updates the mascon distribution.
    @param x: trained decision variable
 */
void MasconFit::setMasconDist(Eigen::VectorXd x)
{
    // Declare temporary variables to store new mascon distribution
    Eigen::VectorXd muMtemp, sqrmuMtemp, xyzMtempVec;
    Eigen::MatrixXd xyzMtemp, xyzMtempT;
    
    // Translate variables to DMC-UKF
    if (this->trainXYZ){
        // Get new square-root mass and mass
        sqrmuMtemp = x.segment(0, this->nM-1);
        muMtemp = sqrmuMtemp.array()*sqrmuMtemp.array() * this->muMad;
            
        // Get new mass positions
        xyzMtempVec = x.segment(this->nM-1, 3*(this->nM-1)).array() * this->xyzMadVec.array();
        xyzMtempT = Eigen::Map<Eigen::MatrixXd>(xyzMtempVec.transpose().data(), 3, this->nM-1);
        xyzMtemp = xyzMtempT.transpose();
            
        // Update mascon distribution
        this->muM << this->mu - muMtemp.sum(), muMtemp;
        this->xyzM.block(1, 0, this->nM-1, 3) = xyzMtemp;
    }
    else{
        // Update mascon distribution
        muMtemp = x.array()*x.array() * this->muMad;
        this->muM << this->mu - muMtemp.sum(), muMtemp;
    }
}

/*! This method prepares the data to be trained.
    @param posData: position dataset
    @param accData: gravity acceleration dataset
 */
void MasconFit::prepareData(Eigen::MatrixXd posData, Eigen::MatrixXd accData)
{
    this->accMax = accData.array().maxCoeff();
    
    // Declare jth data position and acceleration
    Eigen::Vector3d pos_j, acc_j;
    
    // Preallocate position and acceleration (adimensional) data as vectors
    this->accDataNorm.setZero(this->nData);
    this->accDataAd.setZero(this->nData, 3);
    
    // Declare jth mass position w.r.t. 0th mass
    Eigen::Vector3d dr_j0;
    
    // Loop through data
    for (unsigned int j = 0; j < this->nData; j++){
        // Extract jth data and add 0th mass contribution to acceleration
        pos_j = posData.row(j).transpose();
        dr_j0 = pos_j - this->xyzM.row(0).transpose();
        acc_j = accData.row(j).transpose() + this->mu * dr_j0/pow(dr_j0.norm(), 3);
        
        // Fill position and acceleration (adimensional) data vectors
        this->accDataNorm(j) = acc_j.norm();
        this->accDataAd.row(j) = acc_j.transpose();
        
        /*this->posDataAd.row(j) = pos_j;
        this->accDataAd.row(j) = acc_j / acc_j.norm();*/
    }

    // Set 0th mass position
    this->xyzM_0 = this->xyzM.row(0);
    
    // Adimensionalize mass positions if they are trained
    if (this->trainXYZ){
        // Preallocate stack vector with adimensional position factor
        this->xyzMadVec.setZero(3*(this->nM-1));
        
        // Loop through masses
        for (unsigned int k = 0; k < this->nM-1; k++){
            // Fill stack vector
            this->xyzMadVec.segment(3*k, 3) = this->xyzMad;
        }
    }
    else{
        // Preallocate mass positions in vector
        this->xyzMVec.setZero(3*(this->nM-1));
        
        // Loop through masses
        for (unsigned int k = 0; k < this->nM-1; k++){
            // Stack kth mass position
            this->xyzMVec.segment(3*k, 3) = this->xyzM.row(k+1).transpose();
        }
    }
}

/*! This method computes the gradient of the loss function w.r.t. mascon distribution
    @param x: decision variable
    @return grad: gradient
 */
Eigen::VectorXd MasconFit::computeGrad(Eigen::VectorXd x, Eigen::MatrixXd posData, Eigen::MatrixXd accData)
{
    int nData = int(posData.rows());
    
    // Declare mascon Jacobian, loss gradient w.r.t. predicted acceleration
    // and loss gradient w.r.t. mascon decision variables
    Eigen::MatrixXd daccdx;
    Eigen::VectorXd dlossdacc, grad;
    
    // Compute Jacobian of acceleration data
    // w.r.t. mascon decision variables
    if (this->trainXYZ){
        daccdx = this->computeJacobian(x, posData, accData);
        this->daccdsqrmuM = daccdx.block(0, 0, 3*nData, this->nM-1);
    }
    else{
        this->daccdsqrmuM = 2*(this->daccdmuM.array().rowwise()
                               * x.transpose().array());
    }
    
    // Compute loss gradient w.r.t. acceleration data
    dlossdacc = this->computedLossdAcc(x.segment(0, this->nM-1),
                                       accData);
    
    // Preallocate loss function gradient
    if (this->trainXYZ){
        grad.setZero(4*(this->nM-1));
    }
    else{
        grad.setZero(this->nM-1);
    }
        
    // Declare Jacobian column of kth mascon
    Eigen::VectorXd daccdsqrmuM_k, daccdxM_k, daccdyM_k, daccdzM_k;
        
    // Compute gradient
    for (unsigned int k = 0; k < this->nM-1; k++){
        if (this->trainXYZ){
            // Extract Jacobian columns for kth mascon
            daccdxM_k = daccdx.col(this->nM-1 + 3*k);
            daccdyM_k = daccdx.col(this->nM-1 + 3*k+1);
            daccdzM_k = daccdx.col(this->nM-1 + 3*k+2);
            
            // Fill kth element of loss gradient
            grad.segment(this->nM-1 + 3*k, 3) << (daccdxM_k.array() * dlossdacc.array()).sum(),
            (daccdyM_k.array() * dlossdacc.array()).sum(),
            (daccdzM_k.array() * dlossdacc.array()).sum();
        }
        
        // Extract Jacobian columns for kth mascon
        daccdsqrmuM_k = this->daccdsqrmuM.col(k);

        // Fill kth element of loss gradient
        grad(k) = (daccdsqrmuM_k.array() * dlossdacc.array()).sum();
    }

    return grad;
}

/*! This method computes the gradient of the loss function w.r.t. predicted acceleration
    @param sqrmuM: square-root masses
    @return dlossdacc: gradient of loss w.r.t. predicted acceleration
 */
Eigen::VectorXd MasconFit::computedLossdAcc(Eigen::VectorXd sqrmuM, Eigen::MatrixXd accData)
{
    int nData = int(accData.rows());
    
    // Preallocate gradient of loss w.r.t. predicted acceleration
    Eigen::VectorXd dlossdacc, dMSEdacc, dMAEdacc;
    dlossdacc.setZero(3*nData);
    dMSEdacc.setZero(3*nData);
    dMAEdacc.setZero(3*nData);
    
    // Compute predicted adimensional acceleration
    Eigen::VectorXd accVec = (this->daccdsqrmuM * sqrmuM / 2).array();
    Eigen::MatrixXd acc = accVec.transpose().reshaped(3, nData).transpose();
    
    // Preallocate jth data variables;
    Eigen::Vector3d acc_j, accData_j, dacc_j;
    double normaccData_j;
    
    // Loop through data
    for (unsigned int j = 0; j < nData; j++){
        // Get jth acceleration data
        accData_j = accData.row(j);
        normaccData_j = accData_j.norm();
        
        // Get jth acceleration predict
        acc_j = acc.row(j).transpose();
        
        // Compute error
        dacc_j = acc_j - accData_j;
        
        // Fill gradient
        if (this->lossType == "linear"){
            dMAEdacc.segment(3*j, 3) = dacc_j / (dacc_j.norm() * normaccData_j);
            dMSEdacc.segment(3*j, 3) = dacc_j / (dacc_j.norm() * this->accMax);
        }
        else if (this->lossType == "quadratic"){
            dMAEdacc.segment(3*j, 3) = 2*dacc_j / pow(normaccData_j, 2);
            dMSEdacc.segment(3*j, 3) = 2*dacc_j / pow(this->accMax, 2);
        }
    }
    
    // Compute total loss gradient
    dlossdacc = (dMAEdacc + dMSEdacc) / nData;
    
    return dlossdacc;
}

/*! This method computes the Jacobian of the data w.r.t. mascon distribution
    @param x: decision variable
    @return daccdx: Jacobian
 */
Eigen::MatrixXd MasconFit::computeJacobian(Eigen::VectorXd x, Eigen::MatrixXd posData, Eigen::MatrixXd accData)
{
    int nData = int(posData.rows());
    
    // Declare Jacobian of acceleration data w.r.t. mascon distribution
    Eigen::MatrixXd daccdx;
    
    // Preallocate dimensional square-root masses, masses and positions
    Eigen::VectorXd sqrtmuM, muM, xyzM;
    sqrtmuM = x.segment(0, this->nM-1) * sqrt(this->muMad);
    muM = sqrtmuM.array() * sqrtmuM.array();
    
    /*  */
    Eigen::Matrix3d I, xyzMadMat;
    Eigen::Matrix3d daccdxyzM_jk;
    
    // If the mass positions are considered in the Jacobian
    if (this->trainXYZ){
        // Extract masses positions
        xyzM = x.segment(this->nM-1, 3*(this->nM-1)).array() * this->xyzMadVec.array();
        
        // Set identity and diagonal of adimensional masses positions matrices
        I.setIdentity();
        xyzMadMat.setZero();
        xyzMadMat.diagonal() << this->xyzMad.array();
        
        // Preallocate Jacobian
        daccdx.setZero(3*nData, 4*(this->nM-1));
    }
    else{
        // Get stacked mass positions
        xyzM = this->xyzMVec;
        
        // Preallocate Jacobian
        daccdx.setZero(3*nData, this->nM-1);
    }
        
    // Declare jth position data, kth mass position and their relative position
    Eigen::Vector3d pos_j, xyzM_k, dr_j0, dr_jk;
        
    // Declare jth acceleration derivative w.r.t. kth square-root mass
    Eigen::Vector3d daccdsqrmuM_jk;
    
    // Declare kth square-root mass and mass
    double sqrtmuM_k, muM_k;
        
    // Declare relative position norm variables between jth data and kth mass
    // Declare norm variables of position-acceleration data
    double normdr_jk, normdr3_jk, normdr5_jk, normdr3_j0;
        
    // Loop through data
    for (unsigned int j = 0; j < nData; j++){
        // Extract data position
        /*posData_j = this->posDataVec.segment(3*j, 3);
        normacc_j = this->accDataNorm(j);*/
        pos_j = posData.row(j).transpose();
        dr_j0 = pos_j - this->xyzM_0;
        normdr3_j0 = pow(dr_j0.norm(), 3);
        //normacc_j = accData.row(j).norm();
            
        // Loop through masses
        for (unsigned int k = 0; k < this->nM-1; k++){
            // Extract kth mass, its square-root and position
            muM_k = muM(k);
            sqrtmuM_k = sqrtmuM(k);
            xyzM_k = xyzM.segment(3*k, 3);
                
            // Compute relative position (and norms) between jth data and kth mass
            dr_jk = pos_j - xyzM_k;
            normdr_jk = dr_jk.norm();
            normdr3_jk = pow(normdr_jk, 3);
            normdr5_jk = pow(normdr_jk, 5);
                                
            // Compute gradient of jth data w.r.t. kth square-root mass
            // Fill gradient block
            daccdsqrmuM_jk = 2 * sqrtmuM_k * (-dr_jk/normdr3_jk + dr_j0/normdr3_j0) * sqrt(this->muMad);
            daccdx.block(3*j, k, 3, 1) << daccdsqrmuM_jk.array();
            
            // Add mass positions component
            if (this->trainXYZ){
                // Compute gradient of jth data w.r.t. kth mass and position
                // Fill gradient block
                daccdxyzM_jk = muM_k * (I/normdr3_jk - 3*dr_jk*dr_jk.transpose()/normdr5_jk) * xyzMadMat;
                daccdx.block(3*j, this->nM-1 + 3*k, 3, 3) = daccdxyzM_jk.transpose();
            }
        }
    }

    return daccdx;
}

/*! This method computes the loss function
    @param x: decision variable
    @return loss: loss function
 */
double MasconFit::computeLoss(Eigen::VectorXd x, Eigen::MatrixXd accData)
{
    int nData = int(accData.rows());
    
    // Retrieve sqrmuM
    Eigen::VectorXd sqrmuM = x.segment(0, this->nM-1);
    
    // Compute predicted adimensional acceleration
    Eigen::VectorXd accVec = (this->daccdsqrmuM * sqrmuM / 2).array();
    Eigen::MatrixXd acc = accVec.transpose().reshaped(3, nData).transpose();
    
    // Initialize loss terms
    double MAE, MSE, loss;
    MAE = 0;
    MSE = 0;
    
    // Preallocate jth data variables;
    Eigen::Vector3d acc_j, accData_j, dacc_j;
    double normaccData_j;
    
    // Loop through data
    for (unsigned int j = 0; j < nData; j++){
        // Get jth acceleration data
        accData_j = accData.row(j);
        normaccData_j = accData_j.norm();
        
        // Get jth acceleration predict
        acc_j = acc.row(j).transpose();
        
        // Compute error
        dacc_j = acc_j - accData_j;
        
        // Fill gradient
        if (this->lossType == "linear"){
            MAE += dacc_j.norm() / normaccData_j;
            MSE += dacc_j.norm() / this->accMax;
        }
        else if (this->lossType == "quadratic"){
            MAE += pow(dacc_j.norm() / normaccData_j, 2);
            MSE += pow(dacc_j.norm() / this->accMax, 2);
        }
    }
    
    // Compute loss
    loss = (MAE + MSE) / nData;
    
    return loss;
}

/*! This method checks if the decision variable does not violate the constraints and applies
    the constraint projection step in the affirmative case
    @param x: decision variable
    @return x: decision variable
 */
Eigen::VectorXd MasconFit::projectConstraint(Eigen::VectorXd x)
{
    // Do interior constraint
    if (this->trainXYZ){
        // Declare kth mascon laplcian, position and its projection
        Eigen::Vector3d xyzM, xyzMProj;
        
        // Preallocate minimum distance index and kth position stack matrix
        Eigen::MatrixXd xyzMat;
        bool isExterior;
                
        // Loop through masses
        for (unsigned int k = 0; k < this->nM-1; k++){
            xyzM = x.segment(this->nM-1 + 3*k, 3).array() * this->xyzMad.array();
            isExterior = this->shapeModel->isExterior(xyzM);
            
            // If the mass position is exterior
            if (isExterior == true){
                // Project to minimum distance facet
                xyzMProj = this->shapeModel->closestPoint(xyzM);
            }
            else{
                // Keep interior point
                xyzMProj = xyzM;
            }
            
            // Update decision variable
            x.segment(this->nM-1 + 3*k, 3) = xyzMProj.array() / this->xyzMad.array();
        }
    }
        
    // Compute mass excess
    double dmu = sqrt(x.segment(0, this->nM-1).transpose() * x.segment(0, this->nM-1)) - sqrt(this->mu / this->muMad);
    
    // Check if there is a mass excess
    if (dmu > 0){
        // Reduce masses in the same proportion
        x.segment(0, this->nM-1) -= dmu * x.segment(0, this->nM-1)
        / x.segment(0, this->nM-1).norm();
    }
    
    return x;
}

/*! This method trains a mascon distribution using the input position-acceleration dataset
    @param posData: position dataset
    @param accData: gravity acceleration dataset
 */
void MasconFit::train(Eigen::MatrixXd posData, Eigen::MatrixXd accData,
                      bool show_progress)
{
    // Set number of masses and total mass
    // (preserved during training)
    this->nM = int(this->muM.size());
    
    // Obtain number of data and preprocess
    this->nData = int(accData.rows());
    this->prepareData(posData, accData);
        
    // Declare initial and trained decision variables
    Eigen::VectorXd x0, x;
    
    // Get initial condition from the mascon distribution
    x0 = this->getInitCond();
    
    // Compute Jacobian if only masses are to be trained
    if (this->trainXYZ == false){
        Eigen::VectorXd onesVec;
        onesVec.setOnes(this->nM-1);
        this->daccdmuM = this->computeJacobian(onesVec, posData, accData)/2;
    }
    
    // Train the mascon distribution
    this->graddescent.show_progress = show_progress;
    x = this->graddescent.trainLoop(this, x0, posData, this->accDataAd);
    
    // Update the mascon distribution
    this->setMasconDist(x);
}

/*! @brief Creates an instance of the Adam gradient descent.
 */
MasconFit::AdamGD::AdamGD()
{
    // Set default Adam hyperparameters
    this->beta1 = 0.9;
    this->beta2 = 0.99;
    this->lr = 1e-3;
    this->eps = 1e-6;
    this->maxIter = 1000;
}
 
/*! Empty destructor method.
 */
MasconFit::AdamGD::~AdamGD()
{
    return;
}

void MasconFit::AdamGD::shuffleDataset(Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs)
{
    // Get the number of rows in the matrices
    int nData = int(outputs.rows());

    // Create an index array and initialize it with sequential values
    std::vector<int> idx(nData);
    for (int i = 0; i < nData; ++i) {
        idx[i] = i;
    }

    // Shuffle the index array
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);

    // Shuffle both matrices using the same shuffling order
    inputs = inputs(idx, Eigen::all);
    outputs = outputs(idx, Eigen::all);
}

// Function to organize data into mini-batches
std::vector<Eigen::MatrixXd> MasconFit::AdamGD::splitData(const Eigen::MatrixXd& data, int batchSize)
{
    std::vector<Eigen::MatrixXd> miniBatches;
    int nData = int(data.rows());
    int n = int(data.cols());
    int nBatches = nData / this->batchSize;

    // Loop through the shuffled data and create mini-batches
    for (int i = 0; i < nBatches; ++i) {
        Eigen::MatrixXd batch(this->batchSize, n);
        for (int j = 0; j < this->batchSize; ++j) {
            batch.row(j) = data.row(i * this->batchSize + j);
        }
        miniBatches.push_back(batch);
    }

    // Add the last mini-batch if there are remaining samples
    if (nData % this->batchSize != 0) {
        int remainingData = nData % this->batchSize;
        Eigen::MatrixXd batch(remainingData, n);
        for (int i = 0; i < remainingData; ++i) {
            batch.row(i) = data.row(nBatches * this->batchSize + i);
        }
        miniBatches.push_back(batch);
    }

    return miniBatches;
}

void MasconFit::AdamGD::loadDataset(Eigen::MatrixXd inputs, Eigen::MatrixXd outputs)
{
    // Number of batches
    this->nBatches = int(inputs.rows() / this->batchSize);
    
    // Shuffle dataset
    this->shuffleDataset(inputs, outputs);
    
    // Set minibatches of data
    this->inputBatches = this->splitData(inputs, this->batchSize);
    this->outputBatches = this->splitData(outputs, this->batchSize);
}

/*! This method executes the Adam gradient descent training loop
    @param masconfit: main MasconFit class
    @param x0: initial decision variable
    @return x: trained decision variable
 */
Eigen::VectorXd MasconFit::AdamGD::trainLoop(MasconFit* masconfit, Eigen::VectorXd x0, Eigen::MatrixXd inputs, Eigen::MatrixXd outputs)
{
    // Get decision variable dimension
    int nx = int(x0.size());
    
    // Initialize biased first and second-order moments
    Eigen::VectorXd mt, vt, mt0, vt0;
    mt0.setZero(nx);
    vt0.setZero(nx);
    
    // Declare non-biased first and second-order moments
    Eigen::VectorXd mtmod, vtmod;
    
    // Declare loss gradient, update term and decision variable
    Eigen::VectorXd grad, upd, x;
    
    // Preallocate loss function
    this->loss.setZero(this->maxIter);
    
    // Set decimal precision for output
    std::cout.precision(4);
    
    // Loop until maximum iterations
    for (unsigned int i = 0; i < this->maxIter; i++){
        // Shuffle dataset
        this->loadDataset(inputs, outputs);
        this->loss(i) = 0;
        
        // Loop through data batches
        for (unsigned int j = 0; j < this->nBatches; j++){
            // Compute loss gradient
            grad = masconfit->computeGrad(x0,
                                          this->inputBatches[j],
                                          this->outputBatches[j]);
            
            // Update first and second-order moments
            mt = this->beta1 * mt0 + (1-this->beta1) * grad;
            vt = this->beta2 * vt0.array() + (1-this->beta2) * grad.array() * grad.array();
            mtmod = mt / (1 - pow(this->beta1, i+1));
            vtmod = vt / (1 - pow(this->beta2, i+1));
            mt0 = mt;
            vt0 = vt;
            
            // Update decision variable
            upd = this->lr * mtmod.array() / (sqrt(vtmod.array()) + this->eps);
            x = x0 - upd;
            
            // Do constraints projection
            x = masconfit->projectConstraint(x);
            
            // Prepare next iteration
            x0 = x;
            
            // Save current loss
            this->loss(i) += masconfit->computeLoss(x,
                                                    this->outputBatches[j]) / this->nBatches;
        }
            
        // Print progress after each 5% of training is completed
        if (this->show_progress){
            if ((i+1) % (int(this->maxIter/20)) == 0 || i==0){
                std::cout << i+1 << "/" << this->maxIter << " iter, loss="
                << this->loss(i) << '\n';
            }
        }
    }
    
    return x;
}
