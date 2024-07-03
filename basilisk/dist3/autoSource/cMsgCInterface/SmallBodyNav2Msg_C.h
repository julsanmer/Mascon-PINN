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

#ifndef SmallBodyNav2Msg_C_H
#define SmallBodyNav2Msg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/SmallBodyNav2MsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SmallBodyNav2MsgPayload payload;		        //!< message copy, zero'd on construction
    SmallBodyNav2MsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SmallBodyNav2Msg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SmallBodyNav2Msg_cpp_subscribe(SmallBodyNav2Msg_C *subscriber, void* source);

void SmallBodyNav2Msg_C_subscribe(SmallBodyNav2Msg_C *subscriber, SmallBodyNav2Msg_C *source);

int8_t SmallBodyNav2Msg_C_isSubscribedTo(SmallBodyNav2Msg_C *subscriber, SmallBodyNav2Msg_C *source);
int8_t SmallBodyNav2Msg_cpp_isSubscribedTo(SmallBodyNav2Msg_C *subscriber, void* source);

void SmallBodyNav2Msg_C_addAuthor(SmallBodyNav2Msg_C *coowner, SmallBodyNav2Msg_C *data);

void SmallBodyNav2Msg_C_init(SmallBodyNav2Msg_C *owner);

int SmallBodyNav2Msg_C_isLinked(SmallBodyNav2Msg_C *data);

int SmallBodyNav2Msg_C_isWritten(SmallBodyNav2Msg_C *data);

uint64_t SmallBodyNav2Msg_C_timeWritten(SmallBodyNav2Msg_C *data);

int64_t SmallBodyNav2Msg_C_moduleID(SmallBodyNav2Msg_C *data);

void SmallBodyNav2Msg_C_write(SmallBodyNav2MsgPayload *data, SmallBodyNav2Msg_C *destination, int64_t moduleID, uint64_t callTime);

SmallBodyNav2MsgPayload SmallBodyNav2Msg_C_read(SmallBodyNav2Msg_C *source);

SmallBodyNav2MsgPayload SmallBodyNav2Msg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif