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

#ifndef SmallBodyNavHOUMsg_C_H
#define SmallBodyNavHOUMsg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/SmallBodyNavHOUMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SmallBodyNavHOUMsgPayload payload;		        //!< message copy, zero'd on construction
    SmallBodyNavHOUMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SmallBodyNavHOUMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SmallBodyNavHOUMsg_cpp_subscribe(SmallBodyNavHOUMsg_C *subscriber, void* source);

void SmallBodyNavHOUMsg_C_subscribe(SmallBodyNavHOUMsg_C *subscriber, SmallBodyNavHOUMsg_C *source);

int8_t SmallBodyNavHOUMsg_C_isSubscribedTo(SmallBodyNavHOUMsg_C *subscriber, SmallBodyNavHOUMsg_C *source);
int8_t SmallBodyNavHOUMsg_cpp_isSubscribedTo(SmallBodyNavHOUMsg_C *subscriber, void* source);

void SmallBodyNavHOUMsg_C_addAuthor(SmallBodyNavHOUMsg_C *coowner, SmallBodyNavHOUMsg_C *data);

void SmallBodyNavHOUMsg_C_init(SmallBodyNavHOUMsg_C *owner);

int SmallBodyNavHOUMsg_C_isLinked(SmallBodyNavHOUMsg_C *data);

int SmallBodyNavHOUMsg_C_isWritten(SmallBodyNavHOUMsg_C *data);

uint64_t SmallBodyNavHOUMsg_C_timeWritten(SmallBodyNavHOUMsg_C *data);

int64_t SmallBodyNavHOUMsg_C_moduleID(SmallBodyNavHOUMsg_C *data);

void SmallBodyNavHOUMsg_C_write(SmallBodyNavHOUMsgPayload *data, SmallBodyNavHOUMsg_C *destination, int64_t moduleID, uint64_t callTime);

SmallBodyNavHOUMsgPayload SmallBodyNavHOUMsg_C_read(SmallBodyNavHOUMsg_C *source);

SmallBodyNavHOUMsgPayload SmallBodyNavHOUMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif