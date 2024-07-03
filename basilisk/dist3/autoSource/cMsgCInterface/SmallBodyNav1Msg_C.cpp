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

#include "SmallBodyNav1Msg_C.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/bsk_Print.h"
#include<string.h>

//! C interface to subscribe to a message
void SmallBodyNav1Msg_C_subscribe(SmallBodyNav1Msg_C *subscriber, SmallBodyNav1Msg_C *source) {
    subscriber->payloadPointer = &(source->payload);
    subscriber->headerPointer = &(source->header);
    subscriber->header.isLinked = 1;            // set input message as linked
    subscriber->headerPointer->isLinked = 1;        // set output message as linked
};

//! C interface to check if subscriber is indeed subscribed to a message (1: subscribed, 0: not subscribed)
int8_t SmallBodyNav1Msg_C_isSubscribedTo(SmallBodyNav1Msg_C *subscriber, SmallBodyNav1Msg_C *source) {

    return ((subscriber->payloadPointer == &(source->payload))&&(subscriber->headerPointer == &(source->header)));
    
};

//! C interface to claim authorship to a message
void SmallBodyNav1Msg_C_addAuthor(SmallBodyNav1Msg_C *coownerMsg, SmallBodyNav1Msg_C *targetMsg) {
    coownerMsg->payloadPointer = &(targetMsg->payload);
    coownerMsg->headerPointer = &(targetMsg->header);
};

//! C interface to initialize the module output message
void SmallBodyNav1Msg_C_init(SmallBodyNav1Msg_C *owner) {
    //! check if the msg pointer is not assigned already.  If not, then connect message to itself.
    if (owner->payloadPointer == 0) {
        SmallBodyNav1Msg_C_addAuthor(owner, owner);
    }
};

//! C interface to write to a message
void SmallBodyNav1Msg_C_write(SmallBodyNav1MsgPayload *data, SmallBodyNav1Msg_C *destination, int64_t moduleID, uint64_t callTime) {
    *destination->payloadPointer = *data;
    destination->headerPointer->isWritten = 1;
    destination->headerPointer->timeWritten = callTime;
    destination->headerPointer->moduleID = moduleID;
    return;
};

//! C interface to return a zero'd copy of the message payload
SmallBodyNav1MsgPayload SmallBodyNav1Msg_C_zeroMsgPayload() {
    SmallBodyNav1MsgPayload zeroMsg;
    memset(&zeroMsg, 0x0, sizeof(SmallBodyNav1MsgPayload));
    return zeroMsg;
};


//! C interface to read to a message
SmallBodyNav1MsgPayload SmallBodyNav1Msg_C_read(SmallBodyNav1Msg_C *source) {
    if (!source->headerPointer->isWritten) {
        BSK_PRINT(MSG_ERROR,"In C input msg, you are trying to read an un-written message of type SmallBodyNav1Msg.");
    }
    //! ensure the current message container has a copy of a subscribed message.
    //! Does nothing if the message is writing to itself
    source->payload = *source->payloadPointer;

    return *source->payloadPointer;
};

//! C interface to see if this message container has been subscribed to
int SmallBodyNav1Msg_C_isLinked(SmallBodyNav1Msg_C *data) {
    return (int) data->header.isLinked;
};

//! C interface to see if this message container ever been written to
int SmallBodyNav1Msg_C_isWritten(SmallBodyNav1Msg_C *data) {
    if (data->header.isLinked) {
        return (int) data->headerPointer->isWritten;
    }
    BSK_PRINT(MSG_ERROR,"In C input msg, you are checking if an unconnected msg of type SmallBodyNav1Msg is written.");
    return 0;
};

//! C interface to see if this message container ever been written to
uint64_t SmallBodyNav1Msg_C_timeWritten(SmallBodyNav1Msg_C *data) {
    if (data->header.isLinked) {
        return data->headerPointer->timeWritten;
    }
    BSK_PRINT(MSG_ERROR,"In C input msg, you are requesting the write time of an unconnected msg of type SmallBodyNav1Msg.");
    return 0;
};

//! C interface to get the moduleID of who wrote the message
int64_t SmallBodyNav1Msg_C_moduleID(SmallBodyNav1Msg_C *data) {
    if (!data->header.isLinked) {
        BSK_PRINT(MSG_ERROR,"In C input msg, you are requesting moduleID of an unconnected msg of type SmallBodyNav1Msg.");
        return 0;
    }
    if (!data->headerPointer->isWritten) {
        BSK_PRINT(MSG_ERROR,"In C input msg, you are requesting moduleID of an unwritten msg of type SmallBodyNav1Msg.");
        return 0;
    }
    return data->headerPointer->moduleID;
};

//! method description
void SmallBodyNav1Msg_cpp_subscribe(SmallBodyNav1Msg_C *subscriber, void* source){
    Message<SmallBodyNav1MsgPayload>* source_t = (Message<SmallBodyNav1MsgPayload>*) source;
    MsgHeader *msgPtr;
    subscriber->payloadPointer = source_t->subscribeRaw(&(msgPtr));
    subscriber->headerPointer = msgPtr;
    subscriber->header.isLinked = 1;    // set input message as linked
    subscriber->headerPointer->isLinked = 1;    // set output message as linked
};


//! Cpp interface to check if subscriber is indeed subscribed to a message (1: subscribed, 0: not subscribed)
int8_t SmallBodyNav1Msg_cpp_isSubscribedTo(SmallBodyNav1Msg_C *subscriber, void* source) {

    MsgHeader *dummyMsgPtr;
    Message<SmallBodyNav1MsgPayload>* source_t = (Message<SmallBodyNav1MsgPayload>*) source;
    int8_t firstCheck = (subscriber->payloadPointer == source_t->getMsgPointers(&(dummyMsgPtr)));
    int8_t secondCheck = (subscriber->headerPointer == dummyMsgPtr);

    return (firstCheck && secondCheck);
    
};
