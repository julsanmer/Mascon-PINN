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

#include "interRangingMeas.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/astroConstants.h"

#include <iostream>

/*! @brief Creates an instance of the SpacecraftLocation class
 @return void
 */
interRangingMeas::interRangingMeas()
{
    /* Set buffer storage number */
    this->nBuffer = 16;
    this->rangeBuffer.setZero(this->nBuffer,this->nBuffer);
}

/*! Empty destructor method.
 @return void
 */
interRangingMeas::~interRangingMeas()
{
}

/*! Initialize C-wrapped output messages */
void interRangingMeas::SelfInit(){
    InterRangeMsg_C_init(&this->interRangeOutMsgC);
}

/*! Resets the internal position to the specified initial position.*/
void interRangingMeas::Reset(uint64_t CurrentSimNanos)
{
    /* Preallocate linked satellites */
    this->nSat = this->scStateInMsgs.size();
    this->range.setZero(this->nSat,this->nSat);
}


/*! Adds a scState message name to the vector of names to be subscribed to. Also creates a corresponding access message output name.
*/
void interRangingMeas::addSpacecraftToModel(Message<SCStatesMsgPayload> *tmpScMsg)
{
    this->scStateInMsgs.push_back(tmpScMsg->addSubscriber());
}


/*! Read module messages
*/
bool interRangingMeas::ReadMessages()
{
    SCStatesMsgPayload scMsg;

    /* clear out the vector of spacecraft states.  This is created freshly below. */
    this->scStatesBuffer.clear();

    // read in the spacecraft state messages
    bool scRead;
    if(!this->scStateInMsgs.empty())
    {
        scRead = true;
        for (long unsigned int c = 0; c < this->scStateInMsgs.size(); c++) {
            scMsg = this->scStateInMsgs.at(c)();
            scRead = scRead && this->scStateInMsgs.at(c).isWritten();
            this->scStatesBuffer.push_back(scMsg);
        }
    } else {
        bskLogger.bskLog(BSK_ERROR, "Spacecraft location has no other spacecraft to track.");
        scRead = false;
    }
    //! - Read in the optional planet message.  if no planet message is set, then a zero planet position, velocity and orientation is assumed
    bool planetRead = true;
    if(this->planetInMsg.isLinked())
    {
        planetRead = this->planetInMsg.isWritten();
        this->planetState = this->planetInMsg();
    }

    return(planetRead && scRead);
}

/*! write module messages
*/
void interRangingMeas::WriteMessages(uint64_t CurrentClock)
{
    /* Add current buffer values */
    this->rangeBuffer.block(0,0,this->nSat,this->nSat) = this->range;
    
    /* Create output msg buffers */
    InterRangeMsgPayload interRangeOutMsgBuffer;
    
    /* Zero the output message buffers before assigning values */
    interRangeOutMsgBuffer = this->interRangeOutMsg.zeroMsgPayload;
    
    /* Assign values to the small body navigation output message */
    eigenMatrixXd2CArray(this->rangeBuffer, *interRangeOutMsgBuffer.range);
    
    /* Write to the C++-wrapped output messages */
    this->interRangeOutMsg.write(&interRangeOutMsgBuffer, this->moduleID, CurrentClock);

    /* Write to the C-wrapped output messages */
    InterRangeMsg_C_write(&interRangeOutMsgBuffer, &this->interRangeOutMsgC, this->moduleID, CurrentClock);
}

/*! compute the spacecraft to spacecraft access messages
 */
void interRangingMeas::computeRange()
{
    // get planet position and orientation relative to inertial frame
    this->dcm_PN = cArray2EigenMatrix3d(*this->planetState.J20002Pfix);
    this->r_PN_N = cArray2EigenVector3d(this->planetState.PositionVector);
    
    Eigen::Vector3d ri_SN_N;     // ith satellite position relative to inertial
    Eigen::Vector3d ri_SP_P;     // ith satellite position relative to planet
    Eigen::Vector3d rj_SN_N;     // jth satellite position relative to inertial
    Eigen::Vector3d rj_SP_P;     // jth satellite position relative to planet
    Eigen::Vector3d rij_P;       // ith-jth relative position in planet
    
    // compute sc-sc ranging
    for (long unsigned int i=0; i < this->nSat; i++){
        /* Read ith satellite position */
        ri_SN_N = cArray2EigenVector3d(this->scStatesBuffer.at(i).r_BN_N);
        ri_SP_P = this->dcm_PN*(ri_SN_N - this->r_PN_N);
        for (long unsigned int j=i; j < this->nSat; j++){
            /* Read jth satellite position */
            rj_SN_N = cArray2EigenVector3d(this->scStatesBuffer.at(j).r_BN_N);
            rj_SP_P = this->dcm_PN*(rj_SN_N - this->r_PN_N);
            
            /* Compute relative position */
            rij_P = rj_SP_P - ri_SP_P;
            this->range(i,j) = rij_P.norm();
        }
    }
}

/*!
 update module 
 @param CurrentSimNanos
 */
void interRangingMeas::UpdateState(uint64_t CurrentSimNanos)
{
    this->ReadMessages();
    this->computeRange();
    this->WriteMessages(CurrentSimNanos);

}
