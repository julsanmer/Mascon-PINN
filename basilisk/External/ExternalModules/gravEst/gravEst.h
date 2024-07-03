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


#ifndef GRAVEST_H
#define GRAVEST_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"

/*! @brief mascons class */
class PolyGravEst
{
public:
    unsigned int nVertex;         //!< [-] Number of vertexes
    unsigned int nFacet;          //!< [-] Number of facets
    
    Eigen::MatrixXd xyzVertex;    //!< [m] Position of vertex
    Eigen::MatrixXd orderFacet;   //!< [-] Vertexes of a facet
    
    Eigen::MatrixXd xyzFacet;
    Eigen::MatrixXd normalFacet;
    
    double muBody;
    double volPoly;

    BSKLogger bskLogger;          //!< -- BSK Logging

public:
    PolyGravEst();
    ~PolyGravEst();
    void initializeParameters();
    Eigen::Vector3d computeField(Eigen::Vector3d pos);
    Eigen::VectorXd interiorConstraint(Eigen::VectorXd posVec, int nBatch);
    Eigen::VectorXd computeLaplacian(Eigen::MatrixXd posBatch);
    double computeAltitude(Eigen::Vector3d pos);
};



/*! @brief mascons class */
class MasconGravEst
{
public:
    int nM;
    Eigen::MatrixXd posM;
    Eigen::VectorXd muM;
    
    bool MUPOS;
    bool MU;
    
    bool useMSE;
    bool useMLE;
    
    double mu;
    double muMad;
    Eigen::Vector3d posMad;
    Eigen::VectorXd posMadVec;
    
    Eigen::VectorXd accVec;

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    MasconGravEst();
    ~MasconGravEst();
    
    Eigen::Vector3d computeField(Eigen::Vector3d pos);
    
    Eigen::MatrixXd daccdsqrmupos(Eigen::VectorXd sqrmuposMVec, Eigen::VectorXd posDataVec, int nData, Eigen::VectorXd accNormVec);
    Eigen::VectorXd dlossdacc(Eigen::MatrixXd daccdsqrmuM, Eigen::VectorXd sqrmuM, int nData, Eigen::VectorXd accDataVec);
    /*Eigen::VectorXd gradlossMUPOS(Eigen::MatrixXd daccdmuposM, Eigen::VectorXd muposM, int nData, Eigen::VectorXd accDataVec, Eigen::VectorXd accMeanVec, Eigen::VectorXd accStdVec, double L0, bool useMSE, bool useMLE);*/
    Eigen::VectorXd gradlossMUPOS(Eigen::MatrixXd daccdsqrmuposM, Eigen::VectorXd dLdacc, int nData);
    double loss(Eigen::MatrixXd daccdsqrmuM, Eigen::VectorXd sqrmuM, int nData, Eigen::VectorXd accDataVec);
    
    Eigen::MatrixXd daccdmu(Eigen::VectorXd posDataVec, int nData, Eigen::VectorXd accNormVec);
    Eigen::MatrixXd daccdsqrmu(Eigen::MatrixXd daccdmuM, Eigen::VectorXd sqrmuMVec, int nData);
    Eigen::VectorXd gradlossMU(Eigen::MatrixXd daccdsqrmuM, Eigen::VectorXd dLdacc, int nData);
    /*double lossMU(Eigen::MatrixXd daccdmuM, Eigen::VectorXd muMVec, Eigen::VectorXd accDataVec, Eigen::VectorXd accMeanVec, Eigen::VectorXd accStdVec, double L0, bool useMSE, bool useMLE);*/
};

/*! @brief spherical harmonics class */
class SpherharmGravEst
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
    
    Eigen::VectorXd CSad;

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    SpherharmGravEst();
    ~SpherharmGravEst();
    
    void initializeParameters();            //!< [-] configure all spher-harm based on inputs*/
    double getK(const unsigned int degree); //!< class method
    void CSvec2CSmat(Eigen::VectorXd Csvec, Eigen::VectorXd CSad);
    Eigen::VectorXd CSmat2CSvec();
    
    Eigen::MatrixXd daccdCS(Eigen::VectorXd posDataVec, int nData, Eigen::Vector3d accStd);
    Eigen::VectorXd gradlossCS(Eigen::MatrixXd daccdCS, Eigen::VectorXd CSVec, Eigen::VectorXd accDataVec, Eigen::VectorXd accMeanVec, Eigen::VectorXd accStdVec, double L0);
    double lossCS(Eigen::MatrixXd daccdCS, Eigen::VectorXd CSVec, Eigen::VectorXd accDataVec, Eigen::VectorXd accMeanVec, Eigen::VectorXd accStdVec, double L0);
};

/*! @brief This module estimates relative spacecraft position, velocity with respect to the body, and the non-Keplerian acceleration perturbing the spacecraft motion, using an unscented Kalman filter (UKF)
 */
class GravEst: public SysModel {
public:
    GravEst();
    ~GravEst();

    void SelfInit();  //!< Self initialization for C-wrapped messages
    void Reset(uint64_t CurrentSimNanos);  //!< Resets module
    void UpdateState(uint64_t CurrentSimNanos);  //!< Updates state
    
    void trainGravity(Eigen::MatrixXd posBatch, Eigen::MatrixXd accBatch, Eigen::VectorXd W); //!< Updates gravity with pos and acc pairs
    void preprocessData(Eigen::MatrixXd posData, Eigen::MatrixXd accData);
private:

public:    
    Eigen::VectorXd accFit; //!< Fitted accelerations
    Eigen::MatrixXd W; //!< Weight matrix
    
    int nData; //!< Number of training samples
    Eigen::VectorXd posDataVec; //!< Data position vector
    Eigen::VectorXd accDataVec; //!< Data acceleration vector
    Eigen::VectorXd accNormVec;
    Eigen::VectorXd accMeanVec;
    Eigen::VectorXd accStdVec;
    Eigen::Vector3d accMean;
    Eigen::Vector3d accStd;
    
    Eigen::VectorXd L; //!< Loss function
    double L0; //!< Initial loss
    
    double tcpu; //!< Computational time for least-squares
    
    bool useSH;
    bool useM;
    
    bool useAdagrad;
    bool useAdam;
    bool useNAGD;
    bool useLevMar;
    bool stop;
    int maxIter; //!< Gradient descent maximum iterations
    double eta; //!< Gradient descent momentum parameter
    double lam; //!< Gradient descent learning rate
    double beta1;
    double beta2;
    double w;
    double tolStop;
    
    bool useMSE; //!< Mean squared error
    bool useMLE; //!< Mean linear error
    
    BSKLogger bskLogger;  //!< -- BSK Logging
    
    MasconGravEst mascon;  //!< Object that computes the mascons gravity field
    SpherharmGravEst spherharm;  //!< Object that computes the spherical harmonics gravity field
    PolyGravEst poly;  //!< Object that computes the mascons gravity field

private:
};


#endif
