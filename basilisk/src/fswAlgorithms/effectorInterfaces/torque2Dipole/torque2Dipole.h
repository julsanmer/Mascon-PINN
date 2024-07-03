/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#ifndef TORQUE2DIPOLE_H
#define TORQUE2DIPOLE_H

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/TAMSensorBodyMsg_C.h"
#include "cMsgCInterface/DipoleRequestBodyMsg_C.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* Inputs.*/
    TAMSensorBodyMsg_C tamSensorBodyInMsg;          //!< [Tesla] input message for magnetic field sensor data in the Body frame
    CmdTorqueBodyMsg_C tauRequestInMsg;             //!< [N-m] input message containing control torque in the Body frame
    
    /* Outputs.*/
    DipoleRequestBodyMsg_C dipoleRequestOutMsg;     //!< [A-m2] output message containing dipole request in the Body frame
    
    /* Other. */
    BSKLogger *bskLogger;                           //!< BSK Logging
}torque2DipoleConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_torque2Dipole(torque2DipoleConfig *configData, int64_t moduleID);
    void Update_torque2Dipole(torque2DipoleConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_torque2Dipole(torque2DipoleConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
