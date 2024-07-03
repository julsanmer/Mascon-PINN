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

#ifndef SmallBodyNavUKF2Msg_C_H
#define SmallBodyNavUKF2Msg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/SmallBodyNavUKF2MsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SmallBodyNavUKF2MsgPayload payload;		        //!< message copy, zero'd on construction
    SmallBodyNavUKF2MsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SmallBodyNavUKF2Msg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SmallBodyNavUKF2Msg_cpp_subscribe(SmallBodyNavUKF2Msg_C *subscriber, void* source);

void SmallBodyNavUKF2Msg_C_subscribe(SmallBodyNavUKF2Msg_C *subscriber, SmallBodyNavUKF2Msg_C *source);

int8_t SmallBodyNavUKF2Msg_C_isSubscribedTo(SmallBodyNavUKF2Msg_C *subscriber, SmallBodyNavUKF2Msg_C *source);
int8_t SmallBodyNavUKF2Msg_cpp_isSubscribedTo(SmallBodyNavUKF2Msg_C *subscriber, void* source);

void SmallBodyNavUKF2Msg_C_addAuthor(SmallBodyNavUKF2Msg_C *coowner, SmallBodyNavUKF2Msg_C *data);

void SmallBodyNavUKF2Msg_C_init(SmallBodyNavUKF2Msg_C *owner);

int SmallBodyNavUKF2Msg_C_isLinked(SmallBodyNavUKF2Msg_C *data);

int SmallBodyNavUKF2Msg_C_isWritten(SmallBodyNavUKF2Msg_C *data);

uint64_t SmallBodyNavUKF2Msg_C_timeWritten(SmallBodyNavUKF2Msg_C *data);

int64_t SmallBodyNavUKF2Msg_C_moduleID(SmallBodyNavUKF2Msg_C *data);

void SmallBodyNavUKF2Msg_C_write(SmallBodyNavUKF2MsgPayload *data, SmallBodyNavUKF2Msg_C *destination, int64_t moduleID, uint64_t callTime);

SmallBodyNavUKF2MsgPayload SmallBodyNavUKF2Msg_C_read(SmallBodyNavUKF2Msg_C *source);

SmallBodyNavUKF2MsgPayload SmallBodyNavUKF2Msg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif