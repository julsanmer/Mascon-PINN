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


#ifndef INTERRANGING_MEAS_H
#define INTERRANGING_MEAS_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "architecture/_GeneralModuleFiles/sys_model.h"

#include "cMsgCInterface/InterRangeMsg_C.h"

#include "architecture/msgPayloadDefC/SpicePlanetStateMsgPayload.h"
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include "architecture/messaging/messaging.h"

#include "architecture/utilities/bskLogging.h"

/*! @brief ground location class */
class interRangingMeas:  public SysModel {
public:
    interRangingMeas();
    ~interRangingMeas();
    
    void SelfInit();  //!< Self initialization for C-wrapped messages
    void UpdateState(uint64_t CurrentSimNanos);
    void Reset(uint64_t CurrentSimNanos);
    bool ReadMessages();
    void WriteMessages(uint64_t CurrentClock);
    void addSpacecraftToModel(Message<SCStatesMsgPayload> *tmpScMsg);
    
private:
    void computeRange();

public:
    int nSat;
    Eigen::MatrixXd range;

    ReadFunctor<SpicePlanetStateMsgPayload> planetInMsg;            //!< planet state input message
    std::vector<ReadFunctor<SCStatesMsgPayload>> scStateInMsgs; //!< vector of linked sc state input messages
    Message<InterRangeMsgPayload> interRangeOutMsg;  //!< Message buffer for input translational nav message
    InterRangeMsg_C interRangeOutMsgC = {};  //!< C-wrapped camera nav output msg
    std::vector<SCStatesMsgPayload> scStatesBuffer;             //!< buffer of linked spacecraft states
    SpicePlanetStateMsgPayload planetState;                         //!< buffer of planet data
    
    BSKLogger bskLogger;         //!< -- BSK Logging

private:
    int nBuffer;
    Eigen::MatrixXd rangeBuffer;
    
    Eigen::Matrix3d dcm_PN; //!< Rotation matrix from inertial frame N to planet-centered to planet-fixed frame P
    Eigen::Vector3d r_PN_N; //!< [m] Planet to inertial frame origin vector.
};


#endif
