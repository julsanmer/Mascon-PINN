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


#ifndef AUTOCPPMODULE_H
#define AUTOCPPMODULE_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/msgPayloadDefC/AttRefMsgPayload.h"
#include "architecture/msgPayloadDefC/CSSConfigMsgPayload.h"
#include "architecture/msgPayloadDefCpp/CSSConfigLogMsgPayload.h"
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include "architecture/msgPayloadDefCpp/RWConfigMsgPayload.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"

/*! @brief This is an auto-created sample C++ module.  The description is included with the module class definition
 */
class AutoCppModule: public SysModel {
public:
    AutoCppModule();
    ~AutoCppModule();

    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);

public:
    ReadFunctor<AttRefMsgPayload> someInMsg;  //!< input msg description
    ReadFunctor<AttRefMsgPayload> some2InMsg;  //!< input msg description
    ReadFunctor<CSSConfigMsgPayload> anotherInMsg;  //!< input msg description
    ReadFunctor<CSSConfigLogMsgPayload> anotherCppInMsg;  //!< input msg description

    Message<AttRefMsgPayload> some2OutMsg;  //!< output msg description
    Message<SCStatesMsgPayload> someOutMsg;  //!< output msg description
    Message<RWConfigMsgPayload> anotherCppOutMsg;  //!< output msg description

    BSKLogger bskLogger;              //!< -- BSK Logging

};


#endif
