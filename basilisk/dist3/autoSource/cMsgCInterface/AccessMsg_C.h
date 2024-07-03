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

#ifndef AccessMsg_C_H
#define AccessMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AccessMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AccessMsgPayload payload;		        //!< message copy, zero'd on construction
    AccessMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AccessMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AccessMsg_cpp_subscribe(AccessMsg_C *subscriber, void* source);

void AccessMsg_C_subscribe(AccessMsg_C *subscriber, AccessMsg_C *source);

int8_t AccessMsg_C_isSubscribedTo(AccessMsg_C *subscriber, AccessMsg_C *source);
int8_t AccessMsg_cpp_isSubscribedTo(AccessMsg_C *subscriber, void* source);

void AccessMsg_C_addAuthor(AccessMsg_C *coowner, AccessMsg_C *data);

void AccessMsg_C_init(AccessMsg_C *owner);

int AccessMsg_C_isLinked(AccessMsg_C *data);

int AccessMsg_C_isWritten(AccessMsg_C *data);

uint64_t AccessMsg_C_timeWritten(AccessMsg_C *data);

int64_t AccessMsg_C_moduleID(AccessMsg_C *data);

void AccessMsg_C_write(AccessMsgPayload *data, AccessMsg_C *destination, int64_t moduleID, uint64_t callTime);

AccessMsgPayload AccessMsg_C_read(AccessMsg_C *source);

AccessMsgPayload AccessMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif