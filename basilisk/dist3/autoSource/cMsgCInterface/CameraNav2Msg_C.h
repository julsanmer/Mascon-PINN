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

#ifndef CameraNav2Msg_C_H
#define CameraNav2Msg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/CameraNav2MsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CameraNav2MsgPayload payload;		        //!< message copy, zero'd on construction
    CameraNav2MsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CameraNav2Msg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CameraNav2Msg_cpp_subscribe(CameraNav2Msg_C *subscriber, void* source);

void CameraNav2Msg_C_subscribe(CameraNav2Msg_C *subscriber, CameraNav2Msg_C *source);

int8_t CameraNav2Msg_C_isSubscribedTo(CameraNav2Msg_C *subscriber, CameraNav2Msg_C *source);
int8_t CameraNav2Msg_cpp_isSubscribedTo(CameraNav2Msg_C *subscriber, void* source);

void CameraNav2Msg_C_addAuthor(CameraNav2Msg_C *coowner, CameraNav2Msg_C *data);

void CameraNav2Msg_C_init(CameraNav2Msg_C *owner);

int CameraNav2Msg_C_isLinked(CameraNav2Msg_C *data);

int CameraNav2Msg_C_isWritten(CameraNav2Msg_C *data);

uint64_t CameraNav2Msg_C_timeWritten(CameraNav2Msg_C *data);

int64_t CameraNav2Msg_C_moduleID(CameraNav2Msg_C *data);

void CameraNav2Msg_C_write(CameraNav2MsgPayload *data, CameraNav2Msg_C *destination, int64_t moduleID, uint64_t callTime);

CameraNav2MsgPayload CameraNav2Msg_C_read(CameraNav2Msg_C *source);

CameraNav2MsgPayload CameraNav2Msg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif