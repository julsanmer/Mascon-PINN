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

#ifndef GravityGradientMsg_C_H
#define GravityGradientMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/GravityGradientMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    GravityGradientMsgPayload payload;		        //!< message copy, zero'd on construction
    GravityGradientMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} GravityGradientMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void GravityGradientMsg_cpp_subscribe(GravityGradientMsg_C *subscriber, void* source);

void GravityGradientMsg_C_subscribe(GravityGradientMsg_C *subscriber, GravityGradientMsg_C *source);

int8_t GravityGradientMsg_C_isSubscribedTo(GravityGradientMsg_C *subscriber, GravityGradientMsg_C *source);
int8_t GravityGradientMsg_cpp_isSubscribedTo(GravityGradientMsg_C *subscriber, void* source);

void GravityGradientMsg_C_addAuthor(GravityGradientMsg_C *coowner, GravityGradientMsg_C *data);

void GravityGradientMsg_C_init(GravityGradientMsg_C *owner);

int GravityGradientMsg_C_isLinked(GravityGradientMsg_C *data);

int GravityGradientMsg_C_isWritten(GravityGradientMsg_C *data);

uint64_t GravityGradientMsg_C_timeWritten(GravityGradientMsg_C *data);

int64_t GravityGradientMsg_C_moduleID(GravityGradientMsg_C *data);

void GravityGradientMsg_C_write(GravityGradientMsgPayload *data, GravityGradientMsg_C *destination, int64_t moduleID, uint64_t callTime);

GravityGradientMsgPayload GravityGradientMsg_C_read(GravityGradientMsg_C *source);

GravityGradientMsgPayload GravityGradientMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif