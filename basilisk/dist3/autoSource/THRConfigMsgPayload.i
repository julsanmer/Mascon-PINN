#undef SWIGPYTHON_BUILTIN

%module THRConfigMsgPayload
%{
    #include "msgPayloadDefC/THRConfigMsgPayload.h"
    #include "architecture/messaging/messaging.h"
    #include "architecture/msgPayloadDefC/ReconfigBurnInfoMsgPayload.h"
    #include "architecture/msgPayloadDefC/RWConfigElementMsgPayload.h"
    #include "architecture/msgPayloadDefC/THRConfigMsgPayload.h"
    #include "simulation/dynamics/reactionWheels/reactionWheelSupport.h"
    #include <stdint.h>
    #include <vector>
    #include <string>
%}
%include "messaging/newMessaging.ih"

%include "std_vector.i"
%include "std_string.i"
%include "_GeneralModuleFiles/swig_eigen.i"
%include "_GeneralModuleFiles/swig_conly_data.i"
%include "stdint.i"
%template(TimeVector) std::vector<unsigned long long>;
%template(DoubleVector) std::vector<double>;
%template(StringVector) std::vector<std::string>;

%include "architecture/utilities/macroDefinitions.h"
%include "fswAlgorithms/fswUtilities/fswDefinitions.h"
%include "simulation/dynamics/reactionWheels/reactionWheelSupport.h"
ARRAYINTASLIST(FSWdeviceAvailability)
STRUCTASLIST(CSSUnitConfigMsgPayload)
STRUCTASLIST(AccPktDataMsgPayload)
STRUCTASLIST(RWConfigElementMsgPayload)
STRUCTASLIST(CSSArraySensorMsgPayload)

%include "messaging/messaging.h"
%include "_GeneralModuleFiles/sys_model.h"

%array_functions(THRConfigMsgPayload, ThrustConfigArray);
%array_functions(RWConfigElementMsgPayload, RWConfigArray);
%array_functions(ReconfigBurnInfoMsgPayload, ReconfigBurnArray);

%rename(__subscribe_to) subscribeTo;  // we want the users to have a unified "subscribeTo" interface
%rename(__subscribe_to_C) subscribeToC;  // we want the users to have a unified "subscribeTo" interface
%rename(__is_subscribed_to) isSubscribedTo;  // we want the users to have a unified "isSubscribedTo" interface
%rename(__is_subscribed_to_C) isSubscribedToC;  // we want the users to have a unified "isSubscribedTo" interface
%rename(__time_vector) times;  // It's not really useful to give the user back a time vector
%rename(__timeWritten_vector) timesWritten;
%rename(__record_vector) record;

%pythoncode %{
import numpy as np
%};
%include "msgPayloadDefC/THRConfigMsgPayload.h"
INSTANTIATE_TEMPLATES(THRConfigMsg, THRConfigMsgPayload, msgPayloadDefC)
%template(THRConfigMsgOutMsgsVector) std::vector<Message<THRConfigMsgPayload>>;
%template(THRConfigMsgOutMsgsPtrVector) std::vector<Message<THRConfigMsgPayload>*>;
%template(THRConfigMsgInMsgsVector) std::vector<ReadFunctor<THRConfigMsgPayload>>;


%{
#include "cMsgCInterface/THRConfigMsg_C.h"
%}
%include "cMsgCInterface/THRConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct THRConfigMsg;
%extend THRConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import THRConfigMsg
        if type(source) == type(self):
            THRConfigMsg_C_subscribe(self, source)
        elif type(source) == THRConfigMsg:
            THRConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe THRConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import THRConfigMsg
        if type(source) == type(self):
            return (THRConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == THRConfigMsg:
            return (THRConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import THRConfigMsgRecorder
        self.header.isLinked = 1
        return THRConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        THRConfigMsg_C_addAuthor(self, self)
        if data:
            THRConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        THRConfigMsg_C_addAuthor(self, self)
        THRConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return THRConfigMsg_C_read(self)
    %}
};
