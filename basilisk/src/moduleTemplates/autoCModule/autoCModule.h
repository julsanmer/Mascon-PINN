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


#ifndef AUTOCMODULE_H
#define AUTOCMODULE_H

#include <stdint.h>
#include "cMsgCInterface/AttRefMsg_C.h"
#include "cMsgCInterface/CSSConfigMsg_C.h"
#include "cMsgCInterface/SCStatesMsg_C.h"
#include "architecture/utilities/bskLogging.h"

/*! @brief This is an auto-created sample C module.  The description is included with the module class definition
 */
typedef struct {

    /* declare module IO interfaces */
    AttRefMsg_C someInMsg;  //!< input msg description
    AttRefMsg_C some2InMsg;  //!< input msg description
    CSSConfigMsg_C anotherInMsg;  //!< input msg description
    AttRefMsg_C some2OutMsg;  //!< output msg description
    SCStatesMsg_C someOutMsg;  //!< output msg description

    BSKLogger *bskLogger;  //!< BSK Logging
}autoCModuleConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_autoCModule(autoCModuleConfig *configData, int64_t moduleID);
    void Update_autoCModule(autoCModuleConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_autoCModule(autoCModuleConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
