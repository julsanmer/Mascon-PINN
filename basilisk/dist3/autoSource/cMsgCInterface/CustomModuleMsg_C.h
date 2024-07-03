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

/* All of the files in this folder (dist3/autoSource) are autocoded by the script
architecture/messaging/msgAutoSource/GenCMessages.py.
The script checks for the line "INSTANTIATE_TEMPLATES" in the file architecture/messaging/messaging.i. This
ensures that if a c++ message is instantiated that we also have a C equivalent of that message.
*/

#ifndef CustomModuleMsg_C_H
#define CustomModuleMsg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/CustomModuleMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CustomModuleMsgPayload payload;		        //!< message copy, zero'd on construction
    CustomModuleMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CustomModuleMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CustomModuleMsg_cpp_subscribe(CustomModuleMsg_C *subscriber, void* source);

void CustomModuleMsg_C_subscribe(CustomModuleMsg_C *subscriber, CustomModuleMsg_C *source);

int8_t CustomModuleMsg_C_isSubscribedTo(CustomModuleMsg_C *subscriber, CustomModuleMsg_C *source);
int8_t CustomModuleMsg_cpp_isSubscribedTo(CustomModuleMsg_C *subscriber, void* source);

void CustomModuleMsg_C_addAuthor(CustomModuleMsg_C *coowner, CustomModuleMsg_C *data);

void CustomModuleMsg_C_init(CustomModuleMsg_C *owner);

int CustomModuleMsg_C_isLinked(CustomModuleMsg_C *data);

int CustomModuleMsg_C_isWritten(CustomModuleMsg_C *data);

uint64_t CustomModuleMsg_C_timeWritten(CustomModuleMsg_C *data);

int64_t CustomModuleMsg_C_moduleID(CustomModuleMsg_C *data);

void CustomModuleMsg_C_write(CustomModuleMsgPayload *data, CustomModuleMsg_C *destination, int64_t moduleID, uint64_t callTime);

CustomModuleMsgPayload CustomModuleMsg_C_read(CustomModuleMsg_C *source);

CustomModuleMsgPayload CustomModuleMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif