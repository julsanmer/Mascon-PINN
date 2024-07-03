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

#ifndef VSCMGCmdMsg_C_H
#define VSCMGCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/VSCMGCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    VSCMGCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    VSCMGCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} VSCMGCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void VSCMGCmdMsg_cpp_subscribe(VSCMGCmdMsg_C *subscriber, void* source);

void VSCMGCmdMsg_C_subscribe(VSCMGCmdMsg_C *subscriber, VSCMGCmdMsg_C *source);

int8_t VSCMGCmdMsg_C_isSubscribedTo(VSCMGCmdMsg_C *subscriber, VSCMGCmdMsg_C *source);
int8_t VSCMGCmdMsg_cpp_isSubscribedTo(VSCMGCmdMsg_C *subscriber, void* source);

void VSCMGCmdMsg_C_addAuthor(VSCMGCmdMsg_C *coowner, VSCMGCmdMsg_C *data);

void VSCMGCmdMsg_C_init(VSCMGCmdMsg_C *owner);

int VSCMGCmdMsg_C_isLinked(VSCMGCmdMsg_C *data);

int VSCMGCmdMsg_C_isWritten(VSCMGCmdMsg_C *data);

uint64_t VSCMGCmdMsg_C_timeWritten(VSCMGCmdMsg_C *data);

int64_t VSCMGCmdMsg_C_moduleID(VSCMGCmdMsg_C *data);

void VSCMGCmdMsg_C_write(VSCMGCmdMsgPayload *data, VSCMGCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

VSCMGCmdMsgPayload VSCMGCmdMsg_C_read(VSCMGCmdMsg_C *source);

VSCMGCmdMsgPayload VSCMGCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif