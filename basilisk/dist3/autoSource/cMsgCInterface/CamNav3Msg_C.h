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

#ifndef CamNav3Msg_C_H
#define CamNav3Msg_C_H

#include <stdint.h>
#include "architecture/../../External/msgPayloadDefC/CamNav3MsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CamNav3MsgPayload payload;		        //!< message copy, zero'd on construction
    CamNav3MsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CamNav3Msg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CamNav3Msg_cpp_subscribe(CamNav3Msg_C *subscriber, void* source);

void CamNav3Msg_C_subscribe(CamNav3Msg_C *subscriber, CamNav3Msg_C *source);

int8_t CamNav3Msg_C_isSubscribedTo(CamNav3Msg_C *subscriber, CamNav3Msg_C *source);
int8_t CamNav3Msg_cpp_isSubscribedTo(CamNav3Msg_C *subscriber, void* source);

void CamNav3Msg_C_addAuthor(CamNav3Msg_C *coowner, CamNav3Msg_C *data);

void CamNav3Msg_C_init(CamNav3Msg_C *owner);

int CamNav3Msg_C_isLinked(CamNav3Msg_C *data);

int CamNav3Msg_C_isWritten(CamNav3Msg_C *data);

uint64_t CamNav3Msg_C_timeWritten(CamNav3Msg_C *data);

int64_t CamNav3Msg_C_moduleID(CamNav3Msg_C *data);

void CamNav3Msg_C_write(CamNav3MsgPayload *data, CamNav3Msg_C *destination, int64_t moduleID, uint64_t callTime);

CamNav3MsgPayload CamNav3Msg_C_read(CamNav3Msg_C *source);

CamNav3MsgPayload CamNav3Msg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif