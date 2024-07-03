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

#ifndef SmallBodyNav1Msg_C_H
#define SmallBodyNav1Msg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/SmallBodyNav1MsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SmallBodyNav1MsgPayload payload;		        //!< message copy, zero'd on construction
    SmallBodyNav1MsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SmallBodyNav1Msg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SmallBodyNav1Msg_cpp_subscribe(SmallBodyNav1Msg_C *subscriber, void* source);

void SmallBodyNav1Msg_C_subscribe(SmallBodyNav1Msg_C *subscriber, SmallBodyNav1Msg_C *source);

int8_t SmallBodyNav1Msg_C_isSubscribedTo(SmallBodyNav1Msg_C *subscriber, SmallBodyNav1Msg_C *source);
int8_t SmallBodyNav1Msg_cpp_isSubscribedTo(SmallBodyNav1Msg_C *subscriber, void* source);

void SmallBodyNav1Msg_C_addAuthor(SmallBodyNav1Msg_C *coowner, SmallBodyNav1Msg_C *data);

void SmallBodyNav1Msg_C_init(SmallBodyNav1Msg_C *owner);

int SmallBodyNav1Msg_C_isLinked(SmallBodyNav1Msg_C *data);

int SmallBodyNav1Msg_C_isWritten(SmallBodyNav1Msg_C *data);

uint64_t SmallBodyNav1Msg_C_timeWritten(SmallBodyNav1Msg_C *data);

int64_t SmallBodyNav1Msg_C_moduleID(SmallBodyNav1Msg_C *data);

void SmallBodyNav1Msg_C_write(SmallBodyNav1MsgPayload *data, SmallBodyNav1Msg_C *destination, int64_t moduleID, uint64_t callTime);

SmallBodyNav1MsgPayload SmallBodyNav1Msg_C_read(SmallBodyNav1Msg_C *source);

SmallBodyNav1MsgPayload SmallBodyNav1Msg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif