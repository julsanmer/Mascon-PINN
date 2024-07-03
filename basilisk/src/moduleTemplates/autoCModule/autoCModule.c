/*
 ISC License

 Copyright (c) 2024, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#include "moduleTemplates//autoCModule/autoCModule.h"
#include "string.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_autoCModule(autoCModuleConfig  *configData, int64_t moduleID)
{
    AttRefMsg_C_init(&configData->some2OutMsg);
    SCStatesMsg_C_init(&configData->someOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_autoCModule(autoCModuleConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!AttRefMsg_C_isLinked(&configData->someInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: autoCModule.someInMsg was not connected.");
    }
    if (!AttRefMsg_C_isLinked(&configData->some2InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: autoCModule.some2InMsg was not connected.");
    }
    if (!CSSConfigMsg_C_isLinked(&configData->anotherInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: autoCModule.anotherInMsg was not connected.");
    }
}


/*! Add a description of what this main Update() routine does for this module
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_autoCModule(autoCModuleConfig *configData, uint64_t callTime, int64_t moduleID)
{
    AttRefMsgPayload someInMsgBuffer;  //!< local copy of message buffer
    AttRefMsgPayload some2InMsgBuffer;  //!< local copy of message buffer
    CSSConfigMsgPayload anotherInMsgBuffer;  //!< local copy of message buffer
    AttRefMsgPayload some2OutMsgBuffer;  //!< local copy of message buffer
    SCStatesMsgPayload someOutMsgBuffer;  //!< local copy of message buffer

    // always zero the output message buffers before assigning values
    some2OutMsgBuffer = AttRefMsg_C_zeroMsgPayload();
    someOutMsgBuffer = SCStatesMsg_C_zeroMsgPayload();

    // read in the input messages
    someInMsgBuffer = AttRefMsg_C_read(&configData->someInMsg);
    some2InMsgBuffer = AttRefMsg_C_read(&configData->some2InMsg);
    anotherInMsgBuffer = CSSConfigMsg_C_read(&configData->anotherInMsg);

    // do some math and stuff to populate the output messages

    // write to the output messages
    AttRefMsg_C_write(&some2OutMsgBuffer, &configData->some2OutMsg, moduleID, callTime);
    SCStatesMsg_C_write(&someOutMsgBuffer, &configData->someOutMsg, moduleID, callTime);
}

