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


#ifndef MASCONFIT_H
#define MASCONFIT_H

#include "architecture/utilities/macroDefinitions.h"
#include <memory>
#include <vector>
#include "shapeModel.h"
#include "polyhedralShapeModel.h"


/*! @brief MasconFit class */
class MasconFit
{
public:
    MasconFit();
    ~MasconFit();
    
    void setMaxIter(unsigned int maxIter);
    void setLR(double lr);
    void setBatchSize(unsigned int batchSize);
    void setHyperparam(double beta1, double beta2, double eps);
    void setMasconDist(Eigen::VectorXd x);
    
    void train(Eigen::MatrixXd posData, Eigen::MatrixXd accData, bool show_progress);
    
    Eigen::VectorXd getLoss();
        
private:
    Eigen::VectorXd getInitCond();
    void prepareData(Eigen::MatrixXd posData, Eigen::MatrixXd accData);
    
    double computeLoss(Eigen::VectorXd x, Eigen::MatrixXd accData);
    Eigen::VectorXd computeGrad(Eigen::VectorXd x, Eigen::MatrixXd posData, Eigen::MatrixXd accData);
    Eigen::MatrixXd computeJacobian(Eigen::VectorXd x, Eigen::MatrixXd posData, Eigen::MatrixXd accData);
    Eigen::VectorXd computedLossdAcc(Eigen::VectorXd sqrmuM, Eigen::MatrixXd accData);
    Eigen::VectorXd projectConstraint(Eigen::VectorXd x);

public:
    /* Mascon distribution */
    double mu;                  //!< [m^3/s^2] total standard gravity
    int nM;                     //!< [-] number of masses
    Eigen::VectorXd muM;        //!< [m^3/s^2] masses standard gravity stack vector
    Eigen::MatrixXd xyzM;       //!< [m] masses position stack matrix

    /* Adimensionalization */
    double muMad;               //!< [m^3/s^2] adimensional factor for standard gravity
    Eigen::Vector3d xyzMad;     //!< [m] adimensional factor for masses position
    double accMax;
    
    /* Flags to set training type */
    bool trainXYZ;              //!< [-] bool flag that also trains masses position
    std::string lossType;       //!< [-] string with loss type

    BSKLogger bskLogger;        //!< -- BSK Logging
    
    /*! @brief This nested class encodes the Adam gradient descent algorithm */
    class AdamGD
    {
    public:
        AdamGD();
        ~AdamGD();
        
        void loadDataset(Eigen::MatrixXd inputs, Eigen::MatrixXd outputs);
        Eigen::VectorXd trainLoop(MasconFit* mascon, Eigen::VectorXd x0, Eigen::MatrixXd inputs, Eigen::MatrixXd outputs);
        
    private:
        void shuffleDataset(Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs);        
        std::vector<Eigen::MatrixXd> splitData(const Eigen::MatrixXd& data, int batchSize);
        
    public:
        /* Hyperparameters */
        int maxIter;            //!< [-] maximum number of iterations
        double lr;              //!< [-] learning rate
        double beta1;           //!< [-] average gradient decay rate
        double beta2;           //!< [-] average squared gradient decay rate
        double eps;             //!< [-] numerical stability constant
        
        /* Data batches */
        int batchSize;                                //!< [-] batch size of dataset
        int nBatches;                                 //!< [-] number of batches
        std::vector<Eigen::MatrixXd> inputBatches;    //!< [-] inputs data batches
        std::vector<Eigen::MatrixXd> outputBatches;   //!< [-] outputs data batches
        
        /* Training progress */
        bool show_progress;     //!< [-] flag that prints loss evolution
        Eigen::VectorXd loss;   //!< [-] loss history
    };
    
    AdamGD graddescent;             //!< object storing Adam gradient descent algorithm
    std::shared_ptr<ShapeModel> shapeModel; //!< object storing shape model

private:
    /* Dataset */
    int nData;                  //!< [-] number of data samples
    Eigen::VectorXd posDataVec; //!< [m] data positions stack vector
    Eigen::VectorXd accDataVec; //!< [m/s^2] data accelerations stack vector
    Eigen::VectorXd accDataNorm;//!< [m/s^2] norm of each data acceleration
    
    Eigen::MatrixXd posDataAd;
    Eigen::MatrixXd accDataAd;
    
    /* Stack vectors */
    Eigen::VectorXd xyzMVec;    //!< [m] masses position stack vector
    Eigen::VectorXd xyzMadVec;  //!< [m] stacked adimensional factor for masses position
    Eigen::Vector3d xyzM_0;     //!< [m] 0th mass position
    
    /* Jacobian */
    Eigen::MatrixXd daccdsqrmuM; //!< [-] Jacobian of square-root masses
    Eigen::MatrixXd daccdmuM;    //!< [-] Jacobian of
};

#endif
