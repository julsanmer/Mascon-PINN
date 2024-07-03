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


#include "moduleTemplates//autoCppModule/autoCppModule.h"
#include <iostream>
#include <cstring>

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
AutoCppModule::AutoCppModule()
{
}

/*! Module Destructor */
AutoCppModule::~AutoCppModule()
{
}

/*! This method is used to reset the module and checks that required input messages are connect.
    @return void
*/
void AutoCppModule::Reset(uint64_t CurrentSimNanos)
{
    // check that required input messages are connected
    if (!this->someInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "AutoCppModule.someInMsg was not linked.");
    }
    if (!this->some2InMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "AutoCppModule.some2InMsg was not linked.");
    }
    if (!this->anotherInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "AutoCppModule.anotherInMsg was not linked.");
    }
    if (!this->anotherCppInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "AutoCppModule.anotherCppInMsg was not linked.");
    }

}


/*! This is the main method that gets called every time the module is updated.  Provide an appropriate description.
    @return void
*/
void AutoCppModule::UpdateState(uint64_t CurrentSimNanos)
{
    AttRefMsgPayload someInMsgBuffer;  //!< local copy of message buffer
    AttRefMsgPayload some2InMsgBuffer;  //!< local copy of message buffer
    CSSConfigMsgPayload anotherInMsgBuffer;  //!< local copy of message buffer
    CSSConfigLogMsgPayload anotherCppInMsgBuffer;  //!< local copy of message buffer
    AttRefMsgPayload some2OutMsgBuffer;  //!< local copy of message buffer
    SCStatesMsgPayload someOutMsgBuffer;  //!< local copy of message buffer
    RWConfigMsgPayload anotherCppOutMsgBuffer;  //!< local copy of message buffer

    // always zero the output message buffers before assigning values
    some2OutMsgBuffer = this->some2OutMsg.zeroMsgPayload;
    someOutMsgBuffer = this->someOutMsg.zeroMsgPayload;
    anotherCppOutMsgBuffer = this->anotherCppOutMsg.zeroMsgPayload;

    // read in the input messages
    someInMsgBuffer = this->someInMsg();
    some2InMsgBuffer = this->some2InMsg();
    anotherInMsgBuffer = this->anotherInMsg();
    anotherCppInMsgBuffer = this->anotherCppInMsg();

    // do some math and stuff to populate the output messages

    // write to the output messages
    this->some2OutMsg.write(&some2OutMsgBuffer, this->moduleID, CurrentSimNanos);
    this->someOutMsg.write(&someOutMsgBuffer, this->moduleID, CurrentSimNanos);
    this->anotherCppOutMsg.write(&anotherCppOutMsgBuffer, this->moduleID, CurrentSimNanos);
}

