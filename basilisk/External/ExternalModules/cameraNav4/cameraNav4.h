/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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


#ifndef CAMERA_NAV4_H
#define CAMERA_NAV4_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "architecture/_GeneralModuleFiles/sys_model.h"

#include "cMsgCInterface/EphemerisMsg_C.h"
#include "cMsgCInterface/SCStatesMsg_C.h"
#include "cMsgCInterface/CamNav3Msg_C.h"

#include "architecture/messaging/messaging.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/bskLogging.h"

/*! @brief spherical harmonics class */
class CamObjectShape
{
public:
    int nFacet; //!< [-] number of shape facets
    Eigen::MatrixXd orderFacet; //!< [-] facets order
    Eigen::MatrixXd xyzFacet; //!< [m] center points of each facet
    Eigen::MatrixXd normalFacet; //!< [-] normals of each facet
    Eigen::MatrixXd xyzVertex; //!< [m] vertexes points
    
    int nLandmark; //!< [-] number of landmarks in map
    Eigen::VectorXd idxLandmark; //!< [-] index of landmarks
    Eigen::MatrixXd xyzLandmark; //!< [-] map of landmarks
    Eigen::MatrixXd normalLandmark; //!< [-] normals of each landmark facet

    BSKLogger bskLogger;                      //!< -- BSK Logging

public:
    CamObjectShape();
    ~CamObjectShape();
    void initializeParameters();            //!< [-] configure all spher-harm based on inputs
    Eigen::MatrixXd computeSnapshot(Eigen::Vector3d r, Eigen::Vector3d eS, double f, double camFOV, double nxPixel, double nyPixel);
    Eigen::VectorXd computeLaplacian(Eigen::MatrixXd posBatch);
};

/*! @brief ground location class */
class CameraNav4:  public SysModel {
public:
    CameraNav4();
    ~CameraNav4();
    void SelfInit();  //!< Self initialization for C-wrapped messages
    void UpdateState(uint64_t CurrentSimNanos);
    void Reset(uint64_t CurrentSimNanos);
    void ReadMessages();
    void WriteMessages(uint64_t CurrentClock);
    
    void computePixelBatch(Eigen::MatrixXd posBatch, Eigen::MatrixXd eSBatch);
    void computePosBatch();
    
private:
    void computePixel();
    void computePosLSQ();

public:
    double f; //!< [m] camera focal lenght
    double FOVx; //!< [rad] horizontal field of view
    double FOVy; //!< [rad] vertical field of view
    double nxPixel; //!< [-] number of horizontal pixels
    double nyPixel; //!< [-] number of vertical pixels
    double wPixel; //!< [m] pixel width
    
    int nLandmark; //!< [-] number of landmarks
    Eigen::MatrixXd xyzLandmark; //!< [-]  landmarks map for navigation
    Eigen::MatrixXd dxyzLandmark; //!< [-] error in nav. landmarks map
    
    int nVisibleLandmark; //!< [-] number of visible landmarks
    Eigen::VectorXd visibleLandmark; //!< [-] flag telling if a landmark is visible
    Eigen::MatrixXd pixelLandmark; //!< [-] pixels for landmarks
    
    int navSolution; //!< [-] flag telling if a navigation solution is being computed
    
    double maskangleCam; //!< [-] minimum slant range for camera vision
    double maskangleSun; //!< [-] minimum slant range for Sun visibility
        
    Eigen::Vector3d rLSQ_BP_P; //!< [m] least-squares solution
    Eigen::Matrix3d PLSQ; //!< [m^2] uncertainty on least-squares solution
    
    Eigen::VectorXd nVisibleBatch;
    Eigen::VectorXd visibleBatch;
    Eigen::MatrixXd pixelBatch;
    Eigen::MatrixXd latlonBatch;
    Eigen::MatrixXd rNavBatch;
    
    double tcpu; //!< [s] computational time
    
    CamObjectShape body;  //!< Object encoding planet shape

    ReadFunctor<EphemerisMsgPayload> ephemerisInMsg; //!< small body ephemeris input message
    ReadFunctor<SCStatesMsgPayload> scStateInMsg; //!< spacecraft state input msg
    Message<CamNav3MsgPayload> camNav3OutMsg;  //!< Camera nav output msg
    CamNav3Msg_C camNav3OutMsgC = {};  //!< C-wrapped camera nav output msg

    SCStatesMsgPayload spacecraftState; //!< -- input inertial state
    EphemerisMsgPayload ephemeris;  //!< asteroid ephemeris

    BSKLogger bskLogger;         //!< -- BSK Logging

private:
    Eigen::Vector3d r_PN_N;
    Eigen::Vector3d r_BP_P;
    Eigen::Vector3d e_SP_P;
    Eigen::Matrix3d dcm_BP;
    Eigen::Matrix3d dcm_PN;
};

#endif
