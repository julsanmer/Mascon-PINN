%{
#include "cMsgCInterface/OpNavCirclesMsg_C.h"
%}
%include "cMsgCInterface/OpNavCirclesMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct OpNavCirclesMsg;
%extend OpNavCirclesMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import OpNavCirclesMsg
        if type(source) == type(self):
            OpNavCirclesMsg_C_subscribe(self, source)
        elif type(source) == OpNavCirclesMsg:
            OpNavCirclesMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe OpNavCirclesMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import OpNavCirclesMsg
        if type(source) == type(self):
            return (OpNavCirclesMsg_C_isSubscribedTo(self, source))
        elif type(source) == OpNavCirclesMsg:
            return (OpNavCirclesMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import OpNavCirclesMsgRecorder
        self.header.isLinked = 1
        return OpNavCirclesMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        OpNavCirclesMsg_C_addAuthor(self, self)
        if data:
            OpNavCirclesMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        OpNavCirclesMsg_C_addAuthor(self, self)
        OpNavCirclesMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return OpNavCirclesMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DeviceCmdMsg_C.h"
%}
%include "cMsgCInterface/DeviceCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DeviceCmdMsg;
%extend DeviceCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DeviceCmdMsg
        if type(source) == type(self):
            DeviceCmdMsg_C_subscribe(self, source)
        elif type(source) == DeviceCmdMsg:
            DeviceCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DeviceCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DeviceCmdMsg
        if type(source) == type(self):
            return (DeviceCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == DeviceCmdMsg:
            return (DeviceCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DeviceCmdMsgRecorder
        self.header.isLinked = 1
        return DeviceCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DeviceCmdMsg_C_addAuthor(self, self)
        if data:
            DeviceCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DeviceCmdMsg_C_addAuthor(self, self)
        DeviceCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DeviceCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/STSensorMsg_C.h"
%}
%include "cMsgCInterface/STSensorMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct STSensorMsg;
%extend STSensorMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import STSensorMsg
        if type(source) == type(self):
            STSensorMsg_C_subscribe(self, source)
        elif type(source) == STSensorMsg:
            STSensorMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe STSensorMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import STSensorMsg
        if type(source) == type(self):
            return (STSensorMsg_C_isSubscribedTo(self, source))
        elif type(source) == STSensorMsg:
            return (STSensorMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import STSensorMsgRecorder
        self.header.isLinked = 1
        return STSensorMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        STSensorMsg_C_addAuthor(self, self)
        if data:
            STSensorMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        STSensorMsg_C_addAuthor(self, self)
        STSensorMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return STSensorMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWSpeedMsg_C.h"
%}
%include "cMsgCInterface/RWSpeedMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWSpeedMsg;
%extend RWSpeedMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWSpeedMsg
        if type(source) == type(self):
            RWSpeedMsg_C_subscribe(self, source)
        elif type(source) == RWSpeedMsg:
            RWSpeedMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWSpeedMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWSpeedMsg
        if type(source) == type(self):
            return (RWSpeedMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWSpeedMsg:
            return (RWSpeedMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWSpeedMsgRecorder
        self.header.isLinked = 1
        return RWSpeedMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWSpeedMsg_C_addAuthor(self, self)
        if data:
            RWSpeedMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWSpeedMsg_C_addAuthor(self, self)
        RWSpeedMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWSpeedMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavMsg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavMsg;
%extend SmallBodyNavMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavMsg
        if type(source) == type(self):
            SmallBodyNavMsg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavMsg:
            SmallBodyNavMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavMsg
        if type(source) == type(self):
            return (SmallBodyNavMsg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavMsg:
            return (SmallBodyNavMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavMsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavMsg_C_addAuthor(self, self)
        if data:
            SmallBodyNavMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavMsg_C_addAuthor(self, self)
        SmallBodyNavMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PrescribedTransMsg_C.h"
%}
%include "cMsgCInterface/PrescribedTransMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PrescribedTransMsg;
%extend PrescribedTransMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PrescribedTransMsg
        if type(source) == type(self):
            PrescribedTransMsg_C_subscribe(self, source)
        elif type(source) == PrescribedTransMsg:
            PrescribedTransMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PrescribedTransMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PrescribedTransMsg
        if type(source) == type(self):
            return (PrescribedTransMsg_C_isSubscribedTo(self, source))
        elif type(source) == PrescribedTransMsg:
            return (PrescribedTransMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PrescribedTransMsgRecorder
        self.header.isLinked = 1
        return PrescribedTransMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PrescribedTransMsg_C_addAuthor(self, self)
        if data:
            PrescribedTransMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PrescribedTransMsg_C_addAuthor(self, self)
        PrescribedTransMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PrescribedTransMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWConstellationMsg_C.h"
%}
%include "cMsgCInterface/RWConstellationMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWConstellationMsg;
%extend RWConstellationMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWConstellationMsg
        if type(source) == type(self):
            RWConstellationMsg_C_subscribe(self, source)
        elif type(source) == RWConstellationMsg:
            RWConstellationMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWConstellationMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWConstellationMsg
        if type(source) == type(self):
            return (RWConstellationMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWConstellationMsg:
            return (RWConstellationMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWConstellationMsgRecorder
        self.header.isLinked = 1
        return RWConstellationMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWConstellationMsg_C_addAuthor(self, self)
        if data:
            RWConstellationMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWConstellationMsg_C_addAuthor(self, self)
        RWConstellationMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWConstellationMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/GravityGradientMsg_C.h"
%}
%include "cMsgCInterface/GravityGradientMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct GravityGradientMsg;
%extend GravityGradientMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import GravityGradientMsg
        if type(source) == type(self):
            GravityGradientMsg_C_subscribe(self, source)
        elif type(source) == GravityGradientMsg:
            GravityGradientMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe GravityGradientMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import GravityGradientMsg
        if type(source) == type(self):
            return (GravityGradientMsg_C_isSubscribedTo(self, source))
        elif type(source) == GravityGradientMsg:
            return (GravityGradientMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import GravityGradientMsgRecorder
        self.header.isLinked = 1
        return GravityGradientMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        GravityGradientMsg_C_addAuthor(self, self)
        if data:
            GravityGradientMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        GravityGradientMsg_C_addAuthor(self, self)
        GravityGradientMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return GravityGradientMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
%}
%include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CmdTorqueBodyMsg;
%extend CmdTorqueBodyMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CmdTorqueBodyMsg
        if type(source) == type(self):
            CmdTorqueBodyMsg_C_subscribe(self, source)
        elif type(source) == CmdTorqueBodyMsg:
            CmdTorqueBodyMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CmdTorqueBodyMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CmdTorqueBodyMsg
        if type(source) == type(self):
            return (CmdTorqueBodyMsg_C_isSubscribedTo(self, source))
        elif type(source) == CmdTorqueBodyMsg:
            return (CmdTorqueBodyMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CmdTorqueBodyMsgRecorder
        self.header.isLinked = 1
        return CmdTorqueBodyMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CmdTorqueBodyMsg_C_addAuthor(self, self)
        if data:
            CmdTorqueBodyMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CmdTorqueBodyMsg_C_addAuthor(self, self)
        CmdTorqueBodyMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CmdTorqueBodyMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/OpNavLimbMsg_C.h"
%}
%include "cMsgCInterface/OpNavLimbMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct OpNavLimbMsg;
%extend OpNavLimbMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import OpNavLimbMsg
        if type(source) == type(self):
            OpNavLimbMsg_C_subscribe(self, source)
        elif type(source) == OpNavLimbMsg:
            OpNavLimbMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe OpNavLimbMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import OpNavLimbMsg
        if type(source) == type(self):
            return (OpNavLimbMsg_C_isSubscribedTo(self, source))
        elif type(source) == OpNavLimbMsg:
            return (OpNavLimbMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import OpNavLimbMsgRecorder
        self.header.isLinked = 1
        return OpNavLimbMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        OpNavLimbMsg_C_addAuthor(self, self)
        if data:
            OpNavLimbMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        OpNavLimbMsg_C_addAuthor(self, self)
        OpNavLimbMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return OpNavLimbMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DvBurnCmdMsg_C.h"
%}
%include "cMsgCInterface/DvBurnCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DvBurnCmdMsg;
%extend DvBurnCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DvBurnCmdMsg
        if type(source) == type(self):
            DvBurnCmdMsg_C_subscribe(self, source)
        elif type(source) == DvBurnCmdMsg:
            DvBurnCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DvBurnCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DvBurnCmdMsg
        if type(source) == type(self):
            return (DvBurnCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == DvBurnCmdMsg:
            return (DvBurnCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DvBurnCmdMsgRecorder
        self.header.isLinked = 1
        return DvBurnCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DvBurnCmdMsg_C_addAuthor(self, self)
        if data:
            DvBurnCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DvBurnCmdMsg_C_addAuthor(self, self)
        DvBurnCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DvBurnCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWCmdMsg_C.h"
%}
%include "cMsgCInterface/RWCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWCmdMsg;
%extend RWCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWCmdMsg
        if type(source) == type(self):
            RWCmdMsg_C_subscribe(self, source)
        elif type(source) == RWCmdMsg:
            RWCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWCmdMsg
        if type(source) == type(self):
            return (RWCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWCmdMsg:
            return (RWCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWCmdMsgRecorder
        self.header.isLinked = 1
        return RWCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWCmdMsg_C_addAuthor(self, self)
        if data:
            RWCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWCmdMsg_C_addAuthor(self, self)
        RWCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/GroundStateMsg_C.h"
%}
%include "cMsgCInterface/GroundStateMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct GroundStateMsg;
%extend GroundStateMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import GroundStateMsg
        if type(source) == type(self):
            GroundStateMsg_C_subscribe(self, source)
        elif type(source) == GroundStateMsg:
            GroundStateMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe GroundStateMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import GroundStateMsg
        if type(source) == type(self):
            return (GroundStateMsg_C_isSubscribedTo(self, source))
        elif type(source) == GroundStateMsg:
            return (GroundStateMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import GroundStateMsgRecorder
        self.header.isLinked = 1
        return GroundStateMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        GroundStateMsg_C_addAuthor(self, self)
        if data:
            GroundStateMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        GroundStateMsg_C_addAuthor(self, self)
        GroundStateMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return GroundStateMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/EphemerisMsg_C.h"
%}
%include "cMsgCInterface/EphemerisMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct EphemerisMsg;
%extend EphemerisMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import EphemerisMsg
        if type(source) == type(self):
            EphemerisMsg_C_subscribe(self, source)
        elif type(source) == EphemerisMsg:
            EphemerisMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe EphemerisMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import EphemerisMsg
        if type(source) == type(self):
            return (EphemerisMsg_C_isSubscribedTo(self, source))
        elif type(source) == EphemerisMsg:
            return (EphemerisMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import EphemerisMsgRecorder
        self.header.isLinked = 1
        return EphemerisMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        EphemerisMsg_C_addAuthor(self, self)
        if data:
            EphemerisMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        EphemerisMsg_C_addAuthor(self, self)
        EphemerisMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return EphemerisMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/IMUSensorMsg_C.h"
%}
%include "cMsgCInterface/IMUSensorMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct IMUSensorMsg;
%extend IMUSensorMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import IMUSensorMsg
        if type(source) == type(self):
            IMUSensorMsg_C_subscribe(self, source)
        elif type(source) == IMUSensorMsg:
            IMUSensorMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe IMUSensorMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import IMUSensorMsg
        if type(source) == type(self):
            return (IMUSensorMsg_C_isSubscribedTo(self, source))
        elif type(source) == IMUSensorMsg:
            return (IMUSensorMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import IMUSensorMsgRecorder
        self.header.isLinked = 1
        return IMUSensorMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        IMUSensorMsg_C_addAuthor(self, self)
        if data:
            IMUSensorMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        IMUSensorMsg_C_addAuthor(self, self)
        IMUSensorMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return IMUSensorMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DipoleRequestBodyMsg_C.h"
%}
%include "cMsgCInterface/DipoleRequestBodyMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DipoleRequestBodyMsg;
%extend DipoleRequestBodyMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DipoleRequestBodyMsg
        if type(source) == type(self):
            DipoleRequestBodyMsg_C_subscribe(self, source)
        elif type(source) == DipoleRequestBodyMsg:
            DipoleRequestBodyMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DipoleRequestBodyMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DipoleRequestBodyMsg
        if type(source) == type(self):
            return (DipoleRequestBodyMsg_C_isSubscribedTo(self, source))
        elif type(source) == DipoleRequestBodyMsg:
            return (DipoleRequestBodyMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DipoleRequestBodyMsgRecorder
        self.header.isLinked = 1
        return DipoleRequestBodyMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DipoleRequestBodyMsg_C_addAuthor(self, self)
        if data:
            DipoleRequestBodyMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DipoleRequestBodyMsg_C_addAuthor(self, self)
        DipoleRequestBodyMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DipoleRequestBodyMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RateCmdMsg_C.h"
%}
%include "cMsgCInterface/RateCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RateCmdMsg;
%extend RateCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RateCmdMsg
        if type(source) == type(self):
            RateCmdMsg_C_subscribe(self, source)
        elif type(source) == RateCmdMsg:
            RateCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RateCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RateCmdMsg
        if type(source) == type(self):
            return (RateCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == RateCmdMsg:
            return (RateCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RateCmdMsgRecorder
        self.header.isLinked = 1
        return RateCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RateCmdMsg_C_addAuthor(self, self)
        if data:
            RateCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RateCmdMsg_C_addAuthor(self, self)
        RateCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RateCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWAvailabilityMsg_C.h"
%}
%include "cMsgCInterface/RWAvailabilityMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWAvailabilityMsg;
%extend RWAvailabilityMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWAvailabilityMsg
        if type(source) == type(self):
            RWAvailabilityMsg_C_subscribe(self, source)
        elif type(source) == RWAvailabilityMsg:
            RWAvailabilityMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWAvailabilityMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWAvailabilityMsg
        if type(source) == type(self):
            return (RWAvailabilityMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWAvailabilityMsg:
            return (RWAvailabilityMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWAvailabilityMsgRecorder
        self.header.isLinked = 1
        return RWAvailabilityMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWAvailabilityMsg_C_addAuthor(self, self)
        if data:
            RWAvailabilityMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWAvailabilityMsg_C_addAuthor(self, self)
        RWAvailabilityMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWAvailabilityMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWArrayConfigMsg_C.h"
%}
%include "cMsgCInterface/RWArrayConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWArrayConfigMsg;
%extend RWArrayConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWArrayConfigMsg
        if type(source) == type(self):
            RWArrayConfigMsg_C_subscribe(self, source)
        elif type(source) == RWArrayConfigMsg:
            RWArrayConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWArrayConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWArrayConfigMsg
        if type(source) == type(self):
            return (RWArrayConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWArrayConfigMsg:
            return (RWArrayConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWArrayConfigMsgRecorder
        self.header.isLinked = 1
        return RWArrayConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWArrayConfigMsg_C_addAuthor(self, self)
        if data:
            RWArrayConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWArrayConfigMsg_C_addAuthor(self, self)
        RWArrayConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWArrayConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CSSRawDataMsg_C.h"
%}
%include "cMsgCInterface/CSSRawDataMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CSSRawDataMsg;
%extend CSSRawDataMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CSSRawDataMsg
        if type(source) == type(self):
            CSSRawDataMsg_C_subscribe(self, source)
        elif type(source) == CSSRawDataMsg:
            CSSRawDataMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CSSRawDataMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CSSRawDataMsg
        if type(source) == type(self):
            return (CSSRawDataMsg_C_isSubscribedTo(self, source))
        elif type(source) == CSSRawDataMsg:
            return (CSSRawDataMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CSSRawDataMsgRecorder
        self.header.isLinked = 1
        return CSSRawDataMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CSSRawDataMsg_C_addAuthor(self, self)
        if data:
            CSSRawDataMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CSSRawDataMsg_C_addAuthor(self, self)
        CSSRawDataMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CSSRawDataMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CSSConfigMsg_C.h"
%}
%include "cMsgCInterface/CSSConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CSSConfigMsg;
%extend CSSConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CSSConfigMsg
        if type(source) == type(self):
            CSSConfigMsg_C_subscribe(self, source)
        elif type(source) == CSSConfigMsg:
            CSSConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CSSConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CSSConfigMsg
        if type(source) == type(self):
            return (CSSConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == CSSConfigMsg:
            return (CSSConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CSSConfigMsgRecorder
        self.header.isLinked = 1
        return CSSConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CSSConfigMsg_C_addAuthor(self, self)
        if data:
            CSSConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CSSConfigMsg_C_addAuthor(self, self)
        CSSConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CSSConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/VSCMGCmdMsg_C.h"
%}
%include "cMsgCInterface/VSCMGCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct VSCMGCmdMsg;
%extend VSCMGCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import VSCMGCmdMsg
        if type(source) == type(self):
            VSCMGCmdMsg_C_subscribe(self, source)
        elif type(source) == VSCMGCmdMsg:
            VSCMGCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe VSCMGCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import VSCMGCmdMsg
        if type(source) == type(self):
            return (VSCMGCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == VSCMGCmdMsg:
            return (VSCMGCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import VSCMGCmdMsgRecorder
        self.header.isLinked = 1
        return VSCMGCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        VSCMGCmdMsg_C_addAuthor(self, self)
        if data:
            VSCMGCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        VSCMGCmdMsg_C_addAuthor(self, self)
        VSCMGCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return VSCMGCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ReconfigBurnArrayInfoMsg_C.h"
%}
%include "cMsgCInterface/ReconfigBurnArrayInfoMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ReconfigBurnArrayInfoMsg;
%extend ReconfigBurnArrayInfoMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ReconfigBurnArrayInfoMsg
        if type(source) == type(self):
            ReconfigBurnArrayInfoMsg_C_subscribe(self, source)
        elif type(source) == ReconfigBurnArrayInfoMsg:
            ReconfigBurnArrayInfoMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ReconfigBurnArrayInfoMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ReconfigBurnArrayInfoMsg
        if type(source) == type(self):
            return (ReconfigBurnArrayInfoMsg_C_isSubscribedTo(self, source))
        elif type(source) == ReconfigBurnArrayInfoMsg:
            return (ReconfigBurnArrayInfoMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ReconfigBurnArrayInfoMsgRecorder
        self.header.isLinked = 1
        return ReconfigBurnArrayInfoMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ReconfigBurnArrayInfoMsg_C_addAuthor(self, self)
        if data:
            ReconfigBurnArrayInfoMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ReconfigBurnArrayInfoMsg_C_addAuthor(self, self)
        ReconfigBurnArrayInfoMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ReconfigBurnArrayInfoMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/VehicleConfigMsg_C.h"
%}
%include "cMsgCInterface/VehicleConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct VehicleConfigMsg;
%extend VehicleConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import VehicleConfigMsg
        if type(source) == type(self):
            VehicleConfigMsg_C_subscribe(self, source)
        elif type(source) == VehicleConfigMsg:
            VehicleConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe VehicleConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import VehicleConfigMsg
        if type(source) == type(self):
            return (VehicleConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == VehicleConfigMsg:
            return (VehicleConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import VehicleConfigMsgRecorder
        self.header.isLinked = 1
        return VehicleConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        VehicleConfigMsg_C_addAuthor(self, self)
        if data:
            VehicleConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        VehicleConfigMsg_C_addAuthor(self, self)
        VehicleConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return VehicleConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWConfigLogMsg_C.h"
%}
%include "cMsgCInterface/RWConfigLogMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWConfigLogMsg;
%extend RWConfigLogMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWConfigLogMsg
        if type(source) == type(self):
            RWConfigLogMsg_C_subscribe(self, source)
        elif type(source) == RWConfigLogMsg:
            RWConfigLogMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWConfigLogMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWConfigLogMsg
        if type(source) == type(self):
            return (RWConfigLogMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWConfigLogMsg:
            return (RWConfigLogMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWConfigLogMsgRecorder
        self.header.isLinked = 1
        return RWConfigLogMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWConfigLogMsg_C_addAuthor(self, self)
        if data:
            RWConfigLogMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWConfigLogMsg_C_addAuthor(self, self)
        RWConfigLogMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWConfigLogMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/MagneticFieldMsg_C.h"
%}
%include "cMsgCInterface/MagneticFieldMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct MagneticFieldMsg;
%extend MagneticFieldMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import MagneticFieldMsg
        if type(source) == type(self):
            MagneticFieldMsg_C_subscribe(self, source)
        elif type(source) == MagneticFieldMsg:
            MagneticFieldMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe MagneticFieldMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import MagneticFieldMsg
        if type(source) == type(self):
            return (MagneticFieldMsg_C_isSubscribedTo(self, source))
        elif type(source) == MagneticFieldMsg:
            return (MagneticFieldMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import MagneticFieldMsgRecorder
        self.header.isLinked = 1
        return MagneticFieldMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        MagneticFieldMsg_C_addAuthor(self, self)
        if data:
            MagneticFieldMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        MagneticFieldMsg_C_addAuthor(self, self)
        MagneticFieldMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return MagneticFieldMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AttGuidMsg_C.h"
%}
%include "cMsgCInterface/AttGuidMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AttGuidMsg;
%extend AttGuidMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AttGuidMsg
        if type(source) == type(self):
            AttGuidMsg_C_subscribe(self, source)
        elif type(source) == AttGuidMsg:
            AttGuidMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AttGuidMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AttGuidMsg
        if type(source) == type(self):
            return (AttGuidMsg_C_isSubscribedTo(self, source))
        elif type(source) == AttGuidMsg:
            return (AttGuidMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AttGuidMsgRecorder
        self.header.isLinked = 1
        return AttGuidMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AttGuidMsg_C_addAuthor(self, self)
        if data:
            AttGuidMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AttGuidMsg_C_addAuthor(self, self)
        AttGuidMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AttGuidMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavUKFMsg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavUKFMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavUKFMsg;
%extend SmallBodyNavUKFMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKFMsg
        if type(source) == type(self):
            SmallBodyNavUKFMsg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavUKFMsg:
            SmallBodyNavUKFMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavUKFMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKFMsg
        if type(source) == type(self):
            return (SmallBodyNavUKFMsg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavUKFMsg:
            return (SmallBodyNavUKFMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavUKFMsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavUKFMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavUKFMsg_C_addAuthor(self, self)
        if data:
            SmallBodyNavUKFMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavUKFMsg_C_addAuthor(self, self)
        SmallBodyNavUKFMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavUKFMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/TemperatureMsg_C.h"
%}
%include "cMsgCInterface/TemperatureMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct TemperatureMsg;
%extend TemperatureMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import TemperatureMsg
        if type(source) == type(self):
            TemperatureMsg_C_subscribe(self, source)
        elif type(source) == TemperatureMsg:
            TemperatureMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe TemperatureMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import TemperatureMsg
        if type(source) == type(self):
            return (TemperatureMsg_C_isSubscribedTo(self, source))
        elif type(source) == TemperatureMsg:
            return (TemperatureMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import TemperatureMsgRecorder
        self.header.isLinked = 1
        return TemperatureMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        TemperatureMsg_C_addAuthor(self, self)
        if data:
            TemperatureMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        TemperatureMsg_C_addAuthor(self, self)
        TemperatureMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return TemperatureMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DataNodeUsageMsg_C.h"
%}
%include "cMsgCInterface/DataNodeUsageMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DataNodeUsageMsg;
%extend DataNodeUsageMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DataNodeUsageMsg
        if type(source) == type(self):
            DataNodeUsageMsg_C_subscribe(self, source)
        elif type(source) == DataNodeUsageMsg:
            DataNodeUsageMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DataNodeUsageMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DataNodeUsageMsg
        if type(source) == type(self):
            return (DataNodeUsageMsg_C_isSubscribedTo(self, source))
        elif type(source) == DataNodeUsageMsg:
            return (DataNodeUsageMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DataNodeUsageMsgRecorder
        self.header.isLinked = 1
        return DataNodeUsageMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DataNodeUsageMsg_C_addAuthor(self, self)
        if data:
            DataNodeUsageMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DataNodeUsageMsg_C_addAuthor(self, self)
        DataNodeUsageMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DataNodeUsageMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/TAMSensorBodyMsg_C.h"
%}
%include "cMsgCInterface/TAMSensorBodyMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct TAMSensorBodyMsg;
%extend TAMSensorBodyMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import TAMSensorBodyMsg
        if type(source) == type(self):
            TAMSensorBodyMsg_C_subscribe(self, source)
        elif type(source) == TAMSensorBodyMsg:
            TAMSensorBodyMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe TAMSensorBodyMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import TAMSensorBodyMsg
        if type(source) == type(self):
            return (TAMSensorBodyMsg_C_isSubscribedTo(self, source))
        elif type(source) == TAMSensorBodyMsg:
            return (TAMSensorBodyMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import TAMSensorBodyMsgRecorder
        self.header.isLinked = 1
        return TAMSensorBodyMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        TAMSensorBodyMsg_C_addAuthor(self, self)
        if data:
            TAMSensorBodyMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        TAMSensorBodyMsg_C_addAuthor(self, self)
        TAMSensorBodyMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return TAMSensorBodyMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/THRArrayCmdForceMsg_C.h"
%}
%include "cMsgCInterface/THRArrayCmdForceMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct THRArrayCmdForceMsg;
%extend THRArrayCmdForceMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import THRArrayCmdForceMsg
        if type(source) == type(self):
            THRArrayCmdForceMsg_C_subscribe(self, source)
        elif type(source) == THRArrayCmdForceMsg:
            THRArrayCmdForceMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe THRArrayCmdForceMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import THRArrayCmdForceMsg
        if type(source) == type(self):
            return (THRArrayCmdForceMsg_C_isSubscribedTo(self, source))
        elif type(source) == THRArrayCmdForceMsg:
            return (THRArrayCmdForceMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import THRArrayCmdForceMsgRecorder
        self.header.isLinked = 1
        return THRArrayCmdForceMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        THRArrayCmdForceMsg_C_addAuthor(self, self)
        if data:
            THRArrayCmdForceMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        THRArrayCmdForceMsg_C_addAuthor(self, self)
        THRArrayCmdForceMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return THRArrayCmdForceMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/TransRefMsg_C.h"
%}
%include "cMsgCInterface/TransRefMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct TransRefMsg;
%extend TransRefMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import TransRefMsg
        if type(source) == type(self):
            TransRefMsg_C_subscribe(self, source)
        elif type(source) == TransRefMsg:
            TransRefMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe TransRefMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import TransRefMsg
        if type(source) == type(self):
            return (TransRefMsg_C_isSubscribedTo(self, source))
        elif type(source) == TransRefMsg:
            return (TransRefMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import TransRefMsgRecorder
        self.header.isLinked = 1
        return TransRefMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        TransRefMsg_C_addAuthor(self, self)
        if data:
            TransRefMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        TransRefMsg_C_addAuthor(self, self)
        TransRefMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return TransRefMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AccDataMsg_C.h"
%}
%include "cMsgCInterface/AccDataMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AccDataMsg;
%extend AccDataMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AccDataMsg
        if type(source) == type(self):
            AccDataMsg_C_subscribe(self, source)
        elif type(source) == AccDataMsg:
            AccDataMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AccDataMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AccDataMsg
        if type(source) == type(self):
            return (AccDataMsg_C_isSubscribedTo(self, source))
        elif type(source) == AccDataMsg:
            return (AccDataMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AccDataMsgRecorder
        self.header.isLinked = 1
        return AccDataMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AccDataMsg_C_addAuthor(self, self)
        if data:
            AccDataMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AccDataMsg_C_addAuthor(self, self)
        AccDataMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AccDataMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PyBatteryMsg_C.h"
%}
%include "cMsgCInterface/PyBatteryMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PyBatteryMsg;
%extend PyBatteryMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PyBatteryMsg
        if type(source) == type(self):
            PyBatteryMsg_C_subscribe(self, source)
        elif type(source) == PyBatteryMsg:
            PyBatteryMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PyBatteryMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PyBatteryMsg
        if type(source) == type(self):
            return (PyBatteryMsg_C_isSubscribedTo(self, source))
        elif type(source) == PyBatteryMsg:
            return (PyBatteryMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PyBatteryMsgRecorder
        self.header.isLinked = 1
        return PyBatteryMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PyBatteryMsg_C_addAuthor(self, self)
        if data:
            PyBatteryMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PyBatteryMsg_C_addAuthor(self, self)
        PyBatteryMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PyBatteryMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/THRArrayOnTimeCmdMsg_C.h"
%}
%include "cMsgCInterface/THRArrayOnTimeCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct THRArrayOnTimeCmdMsg;
%extend THRArrayOnTimeCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import THRArrayOnTimeCmdMsg
        if type(source) == type(self):
            THRArrayOnTimeCmdMsg_C_subscribe(self, source)
        elif type(source) == THRArrayOnTimeCmdMsg:
            THRArrayOnTimeCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe THRArrayOnTimeCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import THRArrayOnTimeCmdMsg
        if type(source) == type(self):
            return (THRArrayOnTimeCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == THRArrayOnTimeCmdMsg:
            return (THRArrayOnTimeCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import THRArrayOnTimeCmdMsgRecorder
        self.header.isLinked = 1
        return THRArrayOnTimeCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        THRArrayOnTimeCmdMsg_C_addAuthor(self, self)
        if data:
            THRArrayOnTimeCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        THRArrayOnTimeCmdMsg_C_addAuthor(self, self)
        THRArrayOnTimeCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return THRArrayOnTimeCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PowerStorageStatusMsg_C.h"
%}
%include "cMsgCInterface/PowerStorageStatusMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PowerStorageStatusMsg;
%extend PowerStorageStatusMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PowerStorageStatusMsg
        if type(source) == type(self):
            PowerStorageStatusMsg_C_subscribe(self, source)
        elif type(source) == PowerStorageStatusMsg:
            PowerStorageStatusMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PowerStorageStatusMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PowerStorageStatusMsg
        if type(source) == type(self):
            return (PowerStorageStatusMsg_C_isSubscribedTo(self, source))
        elif type(source) == PowerStorageStatusMsg:
            return (PowerStorageStatusMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PowerStorageStatusMsgRecorder
        self.header.isLinked = 1
        return PowerStorageStatusMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PowerStorageStatusMsg_C_addAuthor(self, self)
        if data:
            PowerStorageStatusMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PowerStorageStatusMsg_C_addAuthor(self, self)
        PowerStorageStatusMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PowerStorageStatusMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/HillRelStateMsg_C.h"
%}
%include "cMsgCInterface/HillRelStateMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct HillRelStateMsg;
%extend HillRelStateMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import HillRelStateMsg
        if type(source) == type(self):
            HillRelStateMsg_C_subscribe(self, source)
        elif type(source) == HillRelStateMsg:
            HillRelStateMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe HillRelStateMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import HillRelStateMsg
        if type(source) == type(self):
            return (HillRelStateMsg_C_isSubscribedTo(self, source))
        elif type(source) == HillRelStateMsg:
            return (HillRelStateMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import HillRelStateMsgRecorder
        self.header.isLinked = 1
        return HillRelStateMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        HillRelStateMsg_C_addAuthor(self, self)
        if data:
            HillRelStateMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        HillRelStateMsg_C_addAuthor(self, self)
        HillRelStateMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return HillRelStateMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/InertialHeadingMsg_C.h"
%}
%include "cMsgCInterface/InertialHeadingMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct InertialHeadingMsg;
%extend InertialHeadingMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import InertialHeadingMsg
        if type(source) == type(self):
            InertialHeadingMsg_C_subscribe(self, source)
        elif type(source) == InertialHeadingMsg:
            InertialHeadingMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe InertialHeadingMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import InertialHeadingMsg
        if type(source) == type(self):
            return (InertialHeadingMsg_C_isSubscribedTo(self, source))
        elif type(source) == InertialHeadingMsg:
            return (InertialHeadingMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import InertialHeadingMsgRecorder
        self.header.isLinked = 1
        return InertialHeadingMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        InertialHeadingMsg_C_addAuthor(self, self)
        if data:
            InertialHeadingMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        InertialHeadingMsg_C_addAuthor(self, self)
        InertialHeadingMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return InertialHeadingMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CSSArraySensorMsg_C.h"
%}
%include "cMsgCInterface/CSSArraySensorMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CSSArraySensorMsg;
%extend CSSArraySensorMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CSSArraySensorMsg
        if type(source) == type(self):
            CSSArraySensorMsg_C_subscribe(self, source)
        elif type(source) == CSSArraySensorMsg:
            CSSArraySensorMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CSSArraySensorMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CSSArraySensorMsg
        if type(source) == type(self):
            return (CSSArraySensorMsg_C_isSubscribedTo(self, source))
        elif type(source) == CSSArraySensorMsg:
            return (CSSArraySensorMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CSSArraySensorMsgRecorder
        self.header.isLinked = 1
        return CSSArraySensorMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CSSArraySensorMsg_C_addAuthor(self, self)
        if data:
            CSSArraySensorMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CSSArraySensorMsg_C_addAuthor(self, self)
        CSSArraySensorMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CSSArraySensorMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SynchClockMsg_C.h"
%}
%include "cMsgCInterface/SynchClockMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SynchClockMsg;
%extend SynchClockMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SynchClockMsg
        if type(source) == type(self):
            SynchClockMsg_C_subscribe(self, source)
        elif type(source) == SynchClockMsg:
            SynchClockMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SynchClockMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SynchClockMsg
        if type(source) == type(self):
            return (SynchClockMsg_C_isSubscribedTo(self, source))
        elif type(source) == SynchClockMsg:
            return (SynchClockMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SynchClockMsgRecorder
        self.header.isLinked = 1
        return SynchClockMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SynchClockMsg_C_addAuthor(self, self)
        if data:
            SynchClockMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SynchClockMsg_C_addAuthor(self, self)
        SynchClockMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SynchClockMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CmdForceBodyMsg_C.h"
%}
%include "cMsgCInterface/CmdForceBodyMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CmdForceBodyMsg;
%extend CmdForceBodyMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CmdForceBodyMsg
        if type(source) == type(self):
            CmdForceBodyMsg_C_subscribe(self, source)
        elif type(source) == CmdForceBodyMsg:
            CmdForceBodyMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CmdForceBodyMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CmdForceBodyMsg
        if type(source) == type(self):
            return (CmdForceBodyMsg_C_isSubscribedTo(self, source))
        elif type(source) == CmdForceBodyMsg:
            return (CmdForceBodyMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CmdForceBodyMsgRecorder
        self.header.isLinked = 1
        return CmdForceBodyMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CmdForceBodyMsg_C_addAuthor(self, self)
        if data:
            CmdForceBodyMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CmdForceBodyMsg_C_addAuthor(self, self)
        CmdForceBodyMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CmdForceBodyMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ClassicElementsMsg_C.h"
%}
%include "cMsgCInterface/ClassicElementsMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ClassicElementsMsg;
%extend ClassicElementsMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ClassicElementsMsg
        if type(source) == type(self):
            ClassicElementsMsg_C_subscribe(self, source)
        elif type(source) == ClassicElementsMsg:
            ClassicElementsMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ClassicElementsMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ClassicElementsMsg
        if type(source) == type(self):
            return (ClassicElementsMsg_C_isSubscribedTo(self, source))
        elif type(source) == ClassicElementsMsg:
            return (ClassicElementsMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ClassicElementsMsgRecorder
        self.header.isLinked = 1
        return ClassicElementsMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ClassicElementsMsg_C_addAuthor(self, self)
        if data:
            ClassicElementsMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ClassicElementsMsg_C_addAuthor(self, self)
        ClassicElementsMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ClassicElementsMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SCMassPropsMsg_C.h"
%}
%include "cMsgCInterface/SCMassPropsMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SCMassPropsMsg;
%extend SCMassPropsMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SCMassPropsMsg
        if type(source) == type(self):
            SCMassPropsMsg_C_subscribe(self, source)
        elif type(source) == SCMassPropsMsg:
            SCMassPropsMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SCMassPropsMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SCMassPropsMsg
        if type(source) == type(self):
            return (SCMassPropsMsg_C_isSubscribedTo(self, source))
        elif type(source) == SCMassPropsMsg:
            return (SCMassPropsMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SCMassPropsMsgRecorder
        self.header.isLinked = 1
        return SCMassPropsMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SCMassPropsMsg_C_addAuthor(self, self)
        if data:
            SCMassPropsMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SCMassPropsMsg_C_addAuthor(self, self)
        SCMassPropsMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SCMassPropsMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AlbedoMsg_C.h"
%}
%include "cMsgCInterface/AlbedoMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AlbedoMsg;
%extend AlbedoMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AlbedoMsg
        if type(source) == type(self):
            AlbedoMsg_C_subscribe(self, source)
        elif type(source) == AlbedoMsg:
            AlbedoMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AlbedoMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AlbedoMsg
        if type(source) == type(self):
            return (AlbedoMsg_C_isSubscribedTo(self, source))
        elif type(source) == AlbedoMsg:
            return (AlbedoMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AlbedoMsgRecorder
        self.header.isLinked = 1
        return AlbedoMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AlbedoMsg_C_addAuthor(self, self)
        if data:
            AlbedoMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AlbedoMsg_C_addAuthor(self, self)
        AlbedoMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AlbedoMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ColorMsg_C.h"
%}
%include "cMsgCInterface/ColorMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ColorMsg;
%extend ColorMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ColorMsg
        if type(source) == type(self):
            ColorMsg_C_subscribe(self, source)
        elif type(source) == ColorMsg:
            ColorMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ColorMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ColorMsg
        if type(source) == type(self):
            return (ColorMsg_C_isSubscribedTo(self, source))
        elif type(source) == ColorMsg:
            return (ColorMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ColorMsgRecorder
        self.header.isLinked = 1
        return ColorMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ColorMsg_C_addAuthor(self, self)
        if data:
            ColorMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ColorMsg_C_addAuthor(self, self)
        ColorMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ColorMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CmdForceInertialMsg_C.h"
%}
%include "cMsgCInterface/CmdForceInertialMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CmdForceInertialMsg;
%extend CmdForceInertialMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CmdForceInertialMsg
        if type(source) == type(self):
            CmdForceInertialMsg_C_subscribe(self, source)
        elif type(source) == CmdForceInertialMsg:
            CmdForceInertialMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CmdForceInertialMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CmdForceInertialMsg
        if type(source) == type(self):
            return (CmdForceInertialMsg_C_isSubscribedTo(self, source))
        elif type(source) == CmdForceInertialMsg:
            return (CmdForceInertialMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CmdForceInertialMsgRecorder
        self.header.isLinked = 1
        return CmdForceInertialMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CmdForceInertialMsg_C_addAuthor(self, self)
        if data:
            CmdForceInertialMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CmdForceInertialMsg_C_addAuthor(self, self)
        CmdForceInertialMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CmdForceInertialMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/BoreAngleMsg_C.h"
%}
%include "cMsgCInterface/BoreAngleMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct BoreAngleMsg;
%extend BoreAngleMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import BoreAngleMsg
        if type(source) == type(self):
            BoreAngleMsg_C_subscribe(self, source)
        elif type(source) == BoreAngleMsg:
            BoreAngleMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe BoreAngleMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import BoreAngleMsg
        if type(source) == type(self):
            return (BoreAngleMsg_C_isSubscribedTo(self, source))
        elif type(source) == BoreAngleMsg:
            return (BoreAngleMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import BoreAngleMsgRecorder
        self.header.isLinked = 1
        return BoreAngleMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        BoreAngleMsg_C_addAuthor(self, self)
        if data:
            BoreAngleMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        BoreAngleMsg_C_addAuthor(self, self)
        BoreAngleMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return BoreAngleMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/NavAttMsg_C.h"
%}
%include "cMsgCInterface/NavAttMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct NavAttMsg;
%extend NavAttMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import NavAttMsg
        if type(source) == type(self):
            NavAttMsg_C_subscribe(self, source)
        elif type(source) == NavAttMsg:
            NavAttMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe NavAttMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import NavAttMsg
        if type(source) == type(self):
            return (NavAttMsg_C_isSubscribedTo(self, source))
        elif type(source) == NavAttMsg:
            return (NavAttMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import NavAttMsgRecorder
        self.header.isLinked = 1
        return NavAttMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        NavAttMsg_C_addAuthor(self, self)
        if data:
            NavAttMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        NavAttMsg_C_addAuthor(self, self)
        NavAttMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return NavAttMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SolarFluxMsg_C.h"
%}
%include "cMsgCInterface/SolarFluxMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SolarFluxMsg;
%extend SolarFluxMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SolarFluxMsg
        if type(source) == type(self):
            SolarFluxMsg_C_subscribe(self, source)
        elif type(source) == SolarFluxMsg:
            SolarFluxMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SolarFluxMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SolarFluxMsg
        if type(source) == type(self):
            return (SolarFluxMsg_C_isSubscribedTo(self, source))
        elif type(source) == SolarFluxMsg:
            return (SolarFluxMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SolarFluxMsgRecorder
        self.header.isLinked = 1
        return SolarFluxMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SolarFluxMsg_C_addAuthor(self, self)
        if data:
            SolarFluxMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SolarFluxMsg_C_addAuthor(self, self)
        SolarFluxMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SolarFluxMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/NavTransMsg_C.h"
%}
%include "cMsgCInterface/NavTransMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct NavTransMsg;
%extend NavTransMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import NavTransMsg
        if type(source) == type(self):
            NavTransMsg_C_subscribe(self, source)
        elif type(source) == NavTransMsg:
            NavTransMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe NavTransMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import NavTransMsg
        if type(source) == type(self):
            return (NavTransMsg_C_isSubscribedTo(self, source))
        elif type(source) == NavTransMsg:
            return (NavTransMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import NavTransMsgRecorder
        self.header.isLinked = 1
        return NavTransMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        NavTransMsg_C_addAuthor(self, self)
        if data:
            NavTransMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        NavTransMsg_C_addAuthor(self, self)
        NavTransMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return NavTransMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RWConfigElementMsg_C.h"
%}
%include "cMsgCInterface/RWConfigElementMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RWConfigElementMsg;
%extend RWConfigElementMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RWConfigElementMsg
        if type(source) == type(self):
            RWConfigElementMsg_C_subscribe(self, source)
        elif type(source) == RWConfigElementMsg:
            RWConfigElementMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RWConfigElementMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RWConfigElementMsg
        if type(source) == type(self):
            return (RWConfigElementMsg_C_isSubscribedTo(self, source))
        elif type(source) == RWConfigElementMsg:
            return (RWConfigElementMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RWConfigElementMsgRecorder
        self.header.isLinked = 1
        return RWConfigElementMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RWConfigElementMsg_C_addAuthor(self, self)
        if data:
            RWConfigElementMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RWConfigElementMsg_C_addAuthor(self, self)
        RWConfigElementMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RWConfigElementMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/HeadingFilterMsg_C.h"
%}
%include "cMsgCInterface/HeadingFilterMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct HeadingFilterMsg;
%extend HeadingFilterMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import HeadingFilterMsg
        if type(source) == type(self):
            HeadingFilterMsg_C_subscribe(self, source)
        elif type(source) == HeadingFilterMsg:
            HeadingFilterMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe HeadingFilterMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import HeadingFilterMsg
        if type(source) == type(self):
            return (HeadingFilterMsg_C_isSubscribedTo(self, source))
        elif type(source) == HeadingFilterMsg:
            return (HeadingFilterMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import HeadingFilterMsgRecorder
        self.header.isLinked = 1
        return HeadingFilterMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        HeadingFilterMsg_C_addAuthor(self, self)
        if data:
            HeadingFilterMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        HeadingFilterMsg_C_addAuthor(self, self)
        HeadingFilterMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return HeadingFilterMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SCStatesMsg_C.h"
%}
%include "cMsgCInterface/SCStatesMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SCStatesMsg;
%extend SCStatesMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SCStatesMsg
        if type(source) == type(self):
            SCStatesMsg_C_subscribe(self, source)
        elif type(source) == SCStatesMsg:
            SCStatesMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SCStatesMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SCStatesMsg
        if type(source) == type(self):
            return (SCStatesMsg_C_isSubscribedTo(self, source))
        elif type(source) == SCStatesMsg:
            return (SCStatesMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SCStatesMsgRecorder
        self.header.isLinked = 1
        return SCStatesMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SCStatesMsg_C_addAuthor(self, self)
        if data:
            SCStatesMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SCStatesMsg_C_addAuthor(self, self)
        SCStatesMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SCStatesMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DvExecutionDataMsg_C.h"
%}
%include "cMsgCInterface/DvExecutionDataMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DvExecutionDataMsg;
%extend DvExecutionDataMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DvExecutionDataMsg
        if type(source) == type(self):
            DvExecutionDataMsg_C_subscribe(self, source)
        elif type(source) == DvExecutionDataMsg:
            DvExecutionDataMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DvExecutionDataMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DvExecutionDataMsg
        if type(source) == type(self):
            return (DvExecutionDataMsg_C_isSubscribedTo(self, source))
        elif type(source) == DvExecutionDataMsg:
            return (DvExecutionDataMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DvExecutionDataMsgRecorder
        self.header.isLinked = 1
        return DvExecutionDataMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DvExecutionDataMsg_C_addAuthor(self, self)
        if data:
            DvExecutionDataMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DvExecutionDataMsg_C_addAuthor(self, self)
        DvExecutionDataMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DvExecutionDataMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CameraImageMsg_C.h"
%}
%include "cMsgCInterface/CameraImageMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CameraImageMsg;
%extend CameraImageMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CameraImageMsg
        if type(source) == type(self):
            CameraImageMsg_C_subscribe(self, source)
        elif type(source) == CameraImageMsg:
            CameraImageMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CameraImageMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CameraImageMsg
        if type(source) == type(self):
            return (CameraImageMsg_C_isSubscribedTo(self, source))
        elif type(source) == CameraImageMsg:
            return (CameraImageMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CameraImageMsgRecorder
        self.header.isLinked = 1
        return CameraImageMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CameraImageMsg_C_addAuthor(self, self)
        if data:
            CameraImageMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CameraImageMsg_C_addAuthor(self, self)
        CameraImageMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CameraImageMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AttStateMsg_C.h"
%}
%include "cMsgCInterface/AttStateMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AttStateMsg;
%extend AttStateMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AttStateMsg
        if type(source) == type(self):
            AttStateMsg_C_subscribe(self, source)
        elif type(source) == AttStateMsg:
            AttStateMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AttStateMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AttStateMsg
        if type(source) == type(self):
            return (AttStateMsg_C_isSubscribedTo(self, source))
        elif type(source) == AttStateMsg:
            return (AttStateMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AttStateMsgRecorder
        self.header.isLinked = 1
        return AttStateMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AttStateMsg_C_addAuthor(self, self)
        if data:
            AttStateMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AttStateMsg_C_addAuthor(self, self)
        AttStateMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AttStateMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/THRArrayConfigMsg_C.h"
%}
%include "cMsgCInterface/THRArrayConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct THRArrayConfigMsg;
%extend THRArrayConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import THRArrayConfigMsg
        if type(source) == type(self):
            THRArrayConfigMsg_C_subscribe(self, source)
        elif type(source) == THRArrayConfigMsg:
            THRArrayConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe THRArrayConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import THRArrayConfigMsg
        if type(source) == type(self):
            return (THRArrayConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == THRArrayConfigMsg:
            return (THRArrayConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import THRArrayConfigMsgRecorder
        self.header.isLinked = 1
        return THRArrayConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        THRArrayConfigMsg_C_addAuthor(self, self)
        if data:
            THRArrayConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        THRArrayConfigMsg_C_addAuthor(self, self)
        THRArrayConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return THRArrayConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/EclipseMsg_C.h"
%}
%include "cMsgCInterface/EclipseMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct EclipseMsg;
%extend EclipseMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import EclipseMsg
        if type(source) == type(self):
            EclipseMsg_C_subscribe(self, source)
        elif type(source) == EclipseMsg:
            EclipseMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe EclipseMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import EclipseMsg
        if type(source) == type(self):
            return (EclipseMsg_C_isSubscribedTo(self, source))
        elif type(source) == EclipseMsg:
            return (EclipseMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import EclipseMsgRecorder
        self.header.isLinked = 1
        return EclipseMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        EclipseMsg_C_addAuthor(self, self)
        if data:
            EclipseMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        EclipseMsg_C_addAuthor(self, self)
        EclipseMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return EclipseMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DeviceStatusMsg_C.h"
%}
%include "cMsgCInterface/DeviceStatusMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DeviceStatusMsg;
%extend DeviceStatusMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DeviceStatusMsg
        if type(source) == type(self):
            DeviceStatusMsg_C_subscribe(self, source)
        elif type(source) == DeviceStatusMsg:
            DeviceStatusMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DeviceStatusMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DeviceStatusMsg
        if type(source) == type(self):
            return (DeviceStatusMsg_C_isSubscribedTo(self, source))
        elif type(source) == DeviceStatusMsg:
            return (DeviceStatusMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DeviceStatusMsgRecorder
        self.header.isLinked = 1
        return DeviceStatusMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DeviceStatusMsg_C_addAuthor(self, self)
        if data:
            DeviceStatusMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DeviceStatusMsg_C_addAuthor(self, self)
        DeviceStatusMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DeviceStatusMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/VoltMsg_C.h"
%}
%include "cMsgCInterface/VoltMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct VoltMsg;
%extend VoltMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import VoltMsg
        if type(source) == type(self):
            VoltMsg_C_subscribe(self, source)
        elif type(source) == VoltMsg:
            VoltMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe VoltMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import VoltMsg
        if type(source) == type(self):
            return (VoltMsg_C_isSubscribedTo(self, source))
        elif type(source) == VoltMsg:
            return (VoltMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import VoltMsgRecorder
        self.header.isLinked = 1
        return VoltMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        VoltMsg_C_addAuthor(self, self)
        if data:
            VoltMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        VoltMsg_C_addAuthor(self, self)
        VoltMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return VoltMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SCEnergyMomentumMsg_C.h"
%}
%include "cMsgCInterface/SCEnergyMomentumMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SCEnergyMomentumMsg;
%extend SCEnergyMomentumMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SCEnergyMomentumMsg
        if type(source) == type(self):
            SCEnergyMomentumMsg_C_subscribe(self, source)
        elif type(source) == SCEnergyMomentumMsg:
            SCEnergyMomentumMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SCEnergyMomentumMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SCEnergyMomentumMsg
        if type(source) == type(self):
            return (SCEnergyMomentumMsg_C_isSubscribedTo(self, source))
        elif type(source) == SCEnergyMomentumMsg:
            return (SCEnergyMomentumMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SCEnergyMomentumMsgRecorder
        self.header.isLinked = 1
        return SCEnergyMomentumMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SCEnergyMomentumMsg_C_addAuthor(self, self)
        if data:
            SCEnergyMomentumMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SCEnergyMomentumMsg_C_addAuthor(self, self)
        SCEnergyMomentumMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SCEnergyMomentumMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SunlineFilterMsg_C.h"
%}
%include "cMsgCInterface/SunlineFilterMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SunlineFilterMsg;
%extend SunlineFilterMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SunlineFilterMsg
        if type(source) == type(self):
            SunlineFilterMsg_C_subscribe(self, source)
        elif type(source) == SunlineFilterMsg:
            SunlineFilterMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SunlineFilterMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SunlineFilterMsg
        if type(source) == type(self):
            return (SunlineFilterMsg_C_isSubscribedTo(self, source))
        elif type(source) == SunlineFilterMsg:
            return (SunlineFilterMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SunlineFilterMsgRecorder
        self.header.isLinked = 1
        return SunlineFilterMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SunlineFilterMsg_C_addAuthor(self, self)
        if data:
            SunlineFilterMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SunlineFilterMsg_C_addAuthor(self, self)
        SunlineFilterMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SunlineFilterMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/OpNavMsg_C.h"
%}
%include "cMsgCInterface/OpNavMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct OpNavMsg;
%extend OpNavMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import OpNavMsg
        if type(source) == type(self):
            OpNavMsg_C_subscribe(self, source)
        elif type(source) == OpNavMsg:
            OpNavMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe OpNavMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import OpNavMsg
        if type(source) == type(self):
            return (OpNavMsg_C_isSubscribedTo(self, source))
        elif type(source) == OpNavMsg:
            return (OpNavMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import OpNavMsgRecorder
        self.header.isLinked = 1
        return OpNavMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        OpNavMsg_C_addAuthor(self, self)
        if data:
            OpNavMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        OpNavMsg_C_addAuthor(self, self)
        OpNavMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return OpNavMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SpicePlanetStateMsg_C.h"
%}
%include "cMsgCInterface/SpicePlanetStateMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SpicePlanetStateMsg;
%extend SpicePlanetStateMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SpicePlanetStateMsg
        if type(source) == type(self):
            SpicePlanetStateMsg_C_subscribe(self, source)
        elif type(source) == SpicePlanetStateMsg:
            SpicePlanetStateMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SpicePlanetStateMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SpicePlanetStateMsg
        if type(source) == type(self):
            return (SpicePlanetStateMsg_C_isSubscribedTo(self, source))
        elif type(source) == SpicePlanetStateMsg:
            return (SpicePlanetStateMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SpicePlanetStateMsgRecorder
        self.header.isLinked = 1
        return SpicePlanetStateMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SpicePlanetStateMsg_C_addAuthor(self, self)
        if data:
            SpicePlanetStateMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SpicePlanetStateMsg_C_addAuthor(self, self)
        SpicePlanetStateMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SpicePlanetStateMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PrescribedMotionMsg_C.h"
%}
%include "cMsgCInterface/PrescribedMotionMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PrescribedMotionMsg;
%extend PrescribedMotionMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PrescribedMotionMsg
        if type(source) == type(self):
            PrescribedMotionMsg_C_subscribe(self, source)
        elif type(source) == PrescribedMotionMsg:
            PrescribedMotionMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PrescribedMotionMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PrescribedMotionMsg
        if type(source) == type(self):
            return (PrescribedMotionMsg_C_isSubscribedTo(self, source))
        elif type(source) == PrescribedMotionMsg:
            return (PrescribedMotionMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PrescribedMotionMsgRecorder
        self.header.isLinked = 1
        return PrescribedMotionMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PrescribedMotionMsg_C_addAuthor(self, self)
        if data:
            PrescribedMotionMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PrescribedMotionMsg_C_addAuthor(self, self)
        PrescribedMotionMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PrescribedMotionMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SpiceTimeMsg_C.h"
%}
%include "cMsgCInterface/SpiceTimeMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SpiceTimeMsg;
%extend SpiceTimeMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SpiceTimeMsg
        if type(source) == type(self):
            SpiceTimeMsg_C_subscribe(self, source)
        elif type(source) == SpiceTimeMsg:
            SpiceTimeMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SpiceTimeMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SpiceTimeMsg
        if type(source) == type(self):
            return (SpiceTimeMsg_C_isSubscribedTo(self, source))
        elif type(source) == SpiceTimeMsg:
            return (SpiceTimeMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SpiceTimeMsgRecorder
        self.header.isLinked = 1
        return SpiceTimeMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SpiceTimeMsg_C_addAuthor(self, self)
        if data:
            SpiceTimeMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SpiceTimeMsg_C_addAuthor(self, self)
        SpiceTimeMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SpiceTimeMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/MTBMsg_C.h"
%}
%include "cMsgCInterface/MTBMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct MTBMsg;
%extend MTBMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import MTBMsg
        if type(source) == type(self):
            MTBMsg_C_subscribe(self, source)
        elif type(source) == MTBMsg:
            MTBMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe MTBMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import MTBMsg
        if type(source) == type(self):
            return (MTBMsg_C_isSubscribedTo(self, source))
        elif type(source) == MTBMsg:
            return (MTBMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import MTBMsgRecorder
        self.header.isLinked = 1
        return MTBMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        MTBMsg_C_addAuthor(self, self)
        if data:
            MTBMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        MTBMsg_C_addAuthor(self, self)
        MTBMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return MTBMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/MTBArrayConfigMsg_C.h"
%}
%include "cMsgCInterface/MTBArrayConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct MTBArrayConfigMsg;
%extend MTBArrayConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import MTBArrayConfigMsg
        if type(source) == type(self):
            MTBArrayConfigMsg_C_subscribe(self, source)
        elif type(source) == MTBArrayConfigMsg:
            MTBArrayConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe MTBArrayConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import MTBArrayConfigMsg
        if type(source) == type(self):
            return (MTBArrayConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == MTBArrayConfigMsg:
            return (MTBArrayConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import MTBArrayConfigMsgRecorder
        self.header.isLinked = 1
        return MTBArrayConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        MTBArrayConfigMsg_C_addAuthor(self, self)
        if data:
            MTBArrayConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        MTBArrayConfigMsg_C_addAuthor(self, self)
        MTBArrayConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return MTBArrayConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ReconfigBurnInfoMsg_C.h"
%}
%include "cMsgCInterface/ReconfigBurnInfoMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ReconfigBurnInfoMsg;
%extend ReconfigBurnInfoMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ReconfigBurnInfoMsg
        if type(source) == type(self):
            ReconfigBurnInfoMsg_C_subscribe(self, source)
        elif type(source) == ReconfigBurnInfoMsg:
            ReconfigBurnInfoMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ReconfigBurnInfoMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ReconfigBurnInfoMsg
        if type(source) == type(self):
            return (ReconfigBurnInfoMsg_C_isSubscribedTo(self, source))
        elif type(source) == ReconfigBurnInfoMsg:
            return (ReconfigBurnInfoMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ReconfigBurnInfoMsgRecorder
        self.header.isLinked = 1
        return ReconfigBurnInfoMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ReconfigBurnInfoMsg_C_addAuthor(self, self)
        if data:
            ReconfigBurnInfoMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ReconfigBurnInfoMsg_C_addAuthor(self, self)
        ReconfigBurnInfoMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ReconfigBurnInfoMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/VSCMGSpeedMsg_C.h"
%}
%include "cMsgCInterface/VSCMGSpeedMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct VSCMGSpeedMsg;
%extend VSCMGSpeedMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import VSCMGSpeedMsg
        if type(source) == type(self):
            VSCMGSpeedMsg_C_subscribe(self, source)
        elif type(source) == VSCMGSpeedMsg:
            VSCMGSpeedMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe VSCMGSpeedMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import VSCMGSpeedMsg
        if type(source) == type(self):
            return (VSCMGSpeedMsg_C_isSubscribedTo(self, source))
        elif type(source) == VSCMGSpeedMsg:
            return (VSCMGSpeedMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import VSCMGSpeedMsgRecorder
        self.header.isLinked = 1
        return VSCMGSpeedMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        VSCMGSpeedMsg_C_addAuthor(self, self)
        if data:
            VSCMGSpeedMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        VSCMGSpeedMsg_C_addAuthor(self, self)
        VSCMGSpeedMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return VSCMGSpeedMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AccessMsg_C.h"
%}
%include "cMsgCInterface/AccessMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AccessMsg;
%extend AccessMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AccessMsg
        if type(source) == type(self):
            AccessMsg_C_subscribe(self, source)
        elif type(source) == AccessMsg:
            AccessMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AccessMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AccessMsg
        if type(source) == type(self):
            return (AccessMsg_C_isSubscribedTo(self, source))
        elif type(source) == AccessMsg:
            return (AccessMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AccessMsgRecorder
        self.header.isLinked = 1
        return AccessMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AccessMsg_C_addAuthor(self, self)
        if data:
            AccessMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AccessMsg_C_addAuthor(self, self)
        AccessMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AccessMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/HingedRigidBodyMsg_C.h"
%}
%include "cMsgCInterface/HingedRigidBodyMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct HingedRigidBodyMsg;
%extend HingedRigidBodyMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import HingedRigidBodyMsg
        if type(source) == type(self):
            HingedRigidBodyMsg_C_subscribe(self, source)
        elif type(source) == HingedRigidBodyMsg:
            HingedRigidBodyMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe HingedRigidBodyMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import HingedRigidBodyMsg
        if type(source) == type(self):
            return (HingedRigidBodyMsg_C_isSubscribedTo(self, source))
        elif type(source) == HingedRigidBodyMsg:
            return (HingedRigidBodyMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import HingedRigidBodyMsgRecorder
        self.header.isLinked = 1
        return HingedRigidBodyMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        HingedRigidBodyMsg_C_addAuthor(self, self)
        if data:
            HingedRigidBodyMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        HingedRigidBodyMsg_C_addAuthor(self, self)
        HingedRigidBodyMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return HingedRigidBodyMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CModuleTemplateMsg_C.h"
%}
%include "cMsgCInterface/CModuleTemplateMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CModuleTemplateMsg;
%extend CModuleTemplateMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CModuleTemplateMsg
        if type(source) == type(self):
            CModuleTemplateMsg_C_subscribe(self, source)
        elif type(source) == CModuleTemplateMsg:
            CModuleTemplateMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CModuleTemplateMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CModuleTemplateMsg
        if type(source) == type(self):
            return (CModuleTemplateMsg_C_isSubscribedTo(self, source))
        elif type(source) == CModuleTemplateMsg:
            return (CModuleTemplateMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CModuleTemplateMsgRecorder
        self.header.isLinked = 1
        return CModuleTemplateMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CModuleTemplateMsg_C_addAuthor(self, self)
        if data:
            CModuleTemplateMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CModuleTemplateMsg_C_addAuthor(self, self)
        CModuleTemplateMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CModuleTemplateMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/STAttMsg_C.h"
%}
%include "cMsgCInterface/STAttMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct STAttMsg;
%extend STAttMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import STAttMsg
        if type(source) == type(self):
            STAttMsg_C_subscribe(self, source)
        elif type(source) == STAttMsg:
            STAttMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe STAttMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import STAttMsg
        if type(source) == type(self):
            return (STAttMsg_C_isSubscribedTo(self, source))
        elif type(source) == STAttMsg:
            return (STAttMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import STAttMsgRecorder
        self.header.isLinked = 1
        return STAttMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        STAttMsg_C_addAuthor(self, self)
        if data:
            STAttMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        STAttMsg_C_addAuthor(self, self)
        STAttMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return STAttMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/FuelTankMsg_C.h"
%}
%include "cMsgCInterface/FuelTankMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct FuelTankMsg;
%extend FuelTankMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import FuelTankMsg
        if type(source) == type(self):
            FuelTankMsg_C_subscribe(self, source)
        elif type(source) == FuelTankMsg:
            FuelTankMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe FuelTankMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import FuelTankMsg
        if type(source) == type(self):
            return (FuelTankMsg_C_isSubscribedTo(self, source))
        elif type(source) == FuelTankMsg:
            return (FuelTankMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import FuelTankMsgRecorder
        self.header.isLinked = 1
        return FuelTankMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        FuelTankMsg_C_addAuthor(self, self)
        if data:
            FuelTankMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        FuelTankMsg_C_addAuthor(self, self)
        FuelTankMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return FuelTankMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
%}
%include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ArrayMotorTorqueMsg;
%extend ArrayMotorTorqueMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ArrayMotorTorqueMsg
        if type(source) == type(self):
            ArrayMotorTorqueMsg_C_subscribe(self, source)
        elif type(source) == ArrayMotorTorqueMsg:
            ArrayMotorTorqueMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ArrayMotorTorqueMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ArrayMotorTorqueMsg
        if type(source) == type(self):
            return (ArrayMotorTorqueMsg_C_isSubscribedTo(self, source))
        elif type(source) == ArrayMotorTorqueMsg:
            return (ArrayMotorTorqueMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ArrayMotorTorqueMsgRecorder
        self.header.isLinked = 1
        return ArrayMotorTorqueMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ArrayMotorTorqueMsg_C_addAuthor(self, self)
        if data:
            ArrayMotorTorqueMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ArrayMotorTorqueMsg_C_addAuthor(self, self)
        ArrayMotorTorqueMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ArrayMotorTorqueMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AccPktDataMsg_C.h"
%}
%include "cMsgCInterface/AccPktDataMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AccPktDataMsg;
%extend AccPktDataMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AccPktDataMsg
        if type(source) == type(self):
            AccPktDataMsg_C_subscribe(self, source)
        elif type(source) == AccPktDataMsg:
            AccPktDataMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AccPktDataMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AccPktDataMsg
        if type(source) == type(self):
            return (AccPktDataMsg_C_isSubscribedTo(self, source))
        elif type(source) == AccPktDataMsg:
            return (AccPktDataMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AccPktDataMsgRecorder
        self.header.isLinked = 1
        return AccPktDataMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AccPktDataMsg_C_addAuthor(self, self)
        if data:
            AccPktDataMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AccPktDataMsg_C_addAuthor(self, self)
        AccPktDataMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AccPktDataMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/BodyHeadingMsg_C.h"
%}
%include "cMsgCInterface/BodyHeadingMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct BodyHeadingMsg;
%extend BodyHeadingMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import BodyHeadingMsg
        if type(source) == type(self):
            BodyHeadingMsg_C_subscribe(self, source)
        elif type(source) == BodyHeadingMsg:
            BodyHeadingMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe BodyHeadingMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import BodyHeadingMsg
        if type(source) == type(self):
            return (BodyHeadingMsg_C_isSubscribedTo(self, source))
        elif type(source) == BodyHeadingMsg:
            return (BodyHeadingMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import BodyHeadingMsgRecorder
        self.header.isLinked = 1
        return BodyHeadingMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        BodyHeadingMsg_C_addAuthor(self, self)
        if data:
            BodyHeadingMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        BodyHeadingMsg_C_addAuthor(self, self)
        BodyHeadingMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return BodyHeadingMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CameraConfigMsg_C.h"
%}
%include "cMsgCInterface/CameraConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CameraConfigMsg;
%extend CameraConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CameraConfigMsg
        if type(source) == type(self):
            CameraConfigMsg_C_subscribe(self, source)
        elif type(source) == CameraConfigMsg:
            CameraConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CameraConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CameraConfigMsg
        if type(source) == type(self):
            return (CameraConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == CameraConfigMsg:
            return (CameraConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CameraConfigMsgRecorder
        self.header.isLinked = 1
        return CameraConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CameraConfigMsg_C_addAuthor(self, self)
        if data:
            CameraConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CameraConfigMsg_C_addAuthor(self, self)
        CameraConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CameraConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PlasmaFluxMsg_C.h"
%}
%include "cMsgCInterface/PlasmaFluxMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PlasmaFluxMsg;
%extend PlasmaFluxMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PlasmaFluxMsg
        if type(source) == type(self):
            PlasmaFluxMsg_C_subscribe(self, source)
        elif type(source) == PlasmaFluxMsg:
            PlasmaFluxMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PlasmaFluxMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PlasmaFluxMsg
        if type(source) == type(self):
            return (PlasmaFluxMsg_C_isSubscribedTo(self, source))
        elif type(source) == PlasmaFluxMsg:
            return (PlasmaFluxMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PlasmaFluxMsgRecorder
        self.header.isLinked = 1
        return PlasmaFluxMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PlasmaFluxMsg_C_addAuthor(self, self)
        if data:
            PlasmaFluxMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PlasmaFluxMsg_C_addAuthor(self, self)
        PlasmaFluxMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PlasmaFluxMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/TAMSensorMsg_C.h"
%}
%include "cMsgCInterface/TAMSensorMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct TAMSensorMsg;
%extend TAMSensorMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import TAMSensorMsg
        if type(source) == type(self):
            TAMSensorMsg_C_subscribe(self, source)
        elif type(source) == TAMSensorMsg:
            TAMSensorMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe TAMSensorMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import TAMSensorMsg
        if type(source) == type(self):
            return (TAMSensorMsg_C_isSubscribedTo(self, source))
        elif type(source) == TAMSensorMsg:
            return (TAMSensorMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import TAMSensorMsgRecorder
        self.header.isLinked = 1
        return TAMSensorMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        TAMSensorMsg_C_addAuthor(self, self)
        if data:
            TAMSensorMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        TAMSensorMsg_C_addAuthor(self, self)
        TAMSensorMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return TAMSensorMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/LandmarkMsg_C.h"
%}
%include "cMsgCInterface/LandmarkMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct LandmarkMsg;
%extend LandmarkMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import LandmarkMsg
        if type(source) == type(self):
            LandmarkMsg_C_subscribe(self, source)
        elif type(source) == LandmarkMsg:
            LandmarkMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe LandmarkMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import LandmarkMsg
        if type(source) == type(self):
            return (LandmarkMsg_C_isSubscribedTo(self, source))
        elif type(source) == LandmarkMsg:
            return (LandmarkMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import LandmarkMsgRecorder
        self.header.isLinked = 1
        return LandmarkMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        LandmarkMsg_C_addAuthor(self, self)
        if data:
            LandmarkMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        LandmarkMsg_C_addAuthor(self, self)
        LandmarkMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return LandmarkMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/EpochMsg_C.h"
%}
%include "cMsgCInterface/EpochMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct EpochMsg;
%extend EpochMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import EpochMsg
        if type(source) == type(self):
            EpochMsg_C_subscribe(self, source)
        elif type(source) == EpochMsg:
            EpochMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe EpochMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import EpochMsg
        if type(source) == type(self):
            return (EpochMsg_C_isSubscribedTo(self, source))
        elif type(source) == EpochMsg:
            return (EpochMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import EpochMsgRecorder
        self.header.isLinked = 1
        return EpochMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        EpochMsg_C_addAuthor(self, self)
        if data:
            EpochMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        EpochMsg_C_addAuthor(self, self)
        EpochMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return EpochMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/IMUSensorBodyMsg_C.h"
%}
%include "cMsgCInterface/IMUSensorBodyMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct IMUSensorBodyMsg;
%extend IMUSensorBodyMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import IMUSensorBodyMsg
        if type(source) == type(self):
            IMUSensorBodyMsg_C_subscribe(self, source)
        elif type(source) == IMUSensorBodyMsg:
            IMUSensorBodyMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe IMUSensorBodyMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import IMUSensorBodyMsg
        if type(source) == type(self):
            return (IMUSensorBodyMsg_C_isSubscribedTo(self, source))
        elif type(source) == IMUSensorBodyMsg:
            return (IMUSensorBodyMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import IMUSensorBodyMsgRecorder
        self.header.isLinked = 1
        return IMUSensorBodyMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        IMUSensorBodyMsg_C_addAuthor(self, self)
        if data:
            IMUSensorBodyMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        IMUSensorBodyMsg_C_addAuthor(self, self)
        IMUSensorBodyMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return IMUSensorBodyMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ArrayEffectorLockMsg_C.h"
%}
%include "cMsgCInterface/ArrayEffectorLockMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ArrayEffectorLockMsg;
%extend ArrayEffectorLockMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ArrayEffectorLockMsg
        if type(source) == type(self):
            ArrayEffectorLockMsg_C_subscribe(self, source)
        elif type(source) == ArrayEffectorLockMsg:
            ArrayEffectorLockMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ArrayEffectorLockMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ArrayEffectorLockMsg
        if type(source) == type(self):
            return (ArrayEffectorLockMsg_C_isSubscribedTo(self, source))
        elif type(source) == ArrayEffectorLockMsg:
            return (ArrayEffectorLockMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ArrayEffectorLockMsgRecorder
        self.header.isLinked = 1
        return ArrayEffectorLockMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ArrayEffectorLockMsg_C_addAuthor(self, self)
        if data:
            ArrayEffectorLockMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ArrayEffectorLockMsg_C_addAuthor(self, self)
        ArrayEffectorLockMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ArrayEffectorLockMsg_C_read(self)
    %}
};
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
%{
#include "cMsgCInterface/PowerNodeUsageMsg_C.h"
%}
%include "cMsgCInterface/PowerNodeUsageMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PowerNodeUsageMsg;
%extend PowerNodeUsageMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PowerNodeUsageMsg
        if type(source) == type(self):
            PowerNodeUsageMsg_C_subscribe(self, source)
        elif type(source) == PowerNodeUsageMsg:
            PowerNodeUsageMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PowerNodeUsageMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PowerNodeUsageMsg
        if type(source) == type(self):
            return (PowerNodeUsageMsg_C_isSubscribedTo(self, source))
        elif type(source) == PowerNodeUsageMsg:
            return (PowerNodeUsageMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PowerNodeUsageMsgRecorder
        self.header.isLinked = 1
        return PowerNodeUsageMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PowerNodeUsageMsg_C_addAuthor(self, self)
        if data:
            PowerNodeUsageMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PowerNodeUsageMsg_C_addAuthor(self, self)
        PowerNodeUsageMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PowerNodeUsageMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AttRefMsg_C.h"
%}
%include "cMsgCInterface/AttRefMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AttRefMsg;
%extend AttRefMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AttRefMsg
        if type(source) == type(self):
            AttRefMsg_C_subscribe(self, source)
        elif type(source) == AttRefMsg:
            AttRefMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AttRefMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AttRefMsg
        if type(source) == type(self):
            return (AttRefMsg_C_isSubscribedTo(self, source))
        elif type(source) == AttRefMsg:
            return (AttRefMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AttRefMsgRecorder
        self.header.isLinked = 1
        return AttRefMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AttRefMsg_C_addAuthor(self, self)
        if data:
            AttRefMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AttRefMsg_C_addAuthor(self, self)
        AttRefMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AttRefMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/InertialFilterMsg_C.h"
%}
%include "cMsgCInterface/InertialFilterMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct InertialFilterMsg;
%extend InertialFilterMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import InertialFilterMsg
        if type(source) == type(self):
            InertialFilterMsg_C_subscribe(self, source)
        elif type(source) == InertialFilterMsg:
            InertialFilterMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe InertialFilterMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import InertialFilterMsg
        if type(source) == type(self):
            return (InertialFilterMsg_C_isSubscribedTo(self, source))
        elif type(source) == InertialFilterMsg:
            return (InertialFilterMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import InertialFilterMsgRecorder
        self.header.isLinked = 1
        return InertialFilterMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        InertialFilterMsg_C_addAuthor(self, self)
        if data:
            InertialFilterMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        InertialFilterMsg_C_addAuthor(self, self)
        InertialFilterMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return InertialFilterMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/TDBVehicleClockCorrelationMsg_C.h"
%}
%include "cMsgCInterface/TDBVehicleClockCorrelationMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct TDBVehicleClockCorrelationMsg;
%extend TDBVehicleClockCorrelationMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import TDBVehicleClockCorrelationMsg
        if type(source) == type(self):
            TDBVehicleClockCorrelationMsg_C_subscribe(self, source)
        elif type(source) == TDBVehicleClockCorrelationMsg:
            TDBVehicleClockCorrelationMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe TDBVehicleClockCorrelationMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import TDBVehicleClockCorrelationMsg
        if type(source) == type(self):
            return (TDBVehicleClockCorrelationMsg_C_isSubscribedTo(self, source))
        elif type(source) == TDBVehicleClockCorrelationMsg:
            return (TDBVehicleClockCorrelationMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import TDBVehicleClockCorrelationMsgRecorder
        self.header.isLinked = 1
        return TDBVehicleClockCorrelationMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        TDBVehicleClockCorrelationMsg_C_addAuthor(self, self)
        if data:
            TDBVehicleClockCorrelationMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        TDBVehicleClockCorrelationMsg_C_addAuthor(self, self)
        TDBVehicleClockCorrelationMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return TDBVehicleClockCorrelationMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/ArrayMotorVoltageMsg_C.h"
%}
%include "cMsgCInterface/ArrayMotorVoltageMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct ArrayMotorVoltageMsg;
%extend ArrayMotorVoltageMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import ArrayMotorVoltageMsg
        if type(source) == type(self):
            ArrayMotorVoltageMsg_C_subscribe(self, source)
        elif type(source) == ArrayMotorVoltageMsg:
            ArrayMotorVoltageMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe ArrayMotorVoltageMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import ArrayMotorVoltageMsg
        if type(source) == type(self):
            return (ArrayMotorVoltageMsg_C_isSubscribedTo(self, source))
        elif type(source) == ArrayMotorVoltageMsg:
            return (ArrayMotorVoltageMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import ArrayMotorVoltageMsgRecorder
        self.header.isLinked = 1
        return ArrayMotorVoltageMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        ArrayMotorVoltageMsg_C_addAuthor(self, self)
        if data:
            ArrayMotorVoltageMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        ArrayMotorVoltageMsg_C_addAuthor(self, self)
        ArrayMotorVoltageMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return ArrayMotorVoltageMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/MTBCmdMsg_C.h"
%}
%include "cMsgCInterface/MTBCmdMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct MTBCmdMsg;
%extend MTBCmdMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import MTBCmdMsg
        if type(source) == type(self):
            MTBCmdMsg_C_subscribe(self, source)
        elif type(source) == MTBCmdMsg:
            MTBCmdMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe MTBCmdMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import MTBCmdMsg
        if type(source) == type(self):
            return (MTBCmdMsg_C_isSubscribedTo(self, source))
        elif type(source) == MTBCmdMsg:
            return (MTBCmdMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import MTBCmdMsgRecorder
        self.header.isLinked = 1
        return MTBCmdMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        MTBCmdMsg_C_addAuthor(self, self)
        if data:
            MTBCmdMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        MTBCmdMsg_C_addAuthor(self, self)
        MTBCmdMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return MTBCmdMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/VSCMGArrayTorqueMsg_C.h"
%}
%include "cMsgCInterface/VSCMGArrayTorqueMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct VSCMGArrayTorqueMsg;
%extend VSCMGArrayTorqueMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import VSCMGArrayTorqueMsg
        if type(source) == type(self):
            VSCMGArrayTorqueMsg_C_subscribe(self, source)
        elif type(source) == VSCMGArrayTorqueMsg:
            VSCMGArrayTorqueMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe VSCMGArrayTorqueMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import VSCMGArrayTorqueMsg
        if type(source) == type(self):
            return (VSCMGArrayTorqueMsg_C_isSubscribedTo(self, source))
        elif type(source) == VSCMGArrayTorqueMsg:
            return (VSCMGArrayTorqueMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import VSCMGArrayTorqueMsgRecorder
        self.header.isLinked = 1
        return VSCMGArrayTorqueMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        VSCMGArrayTorqueMsg_C_addAuthor(self, self)
        if data:
            VSCMGArrayTorqueMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        VSCMGArrayTorqueMsg_C_addAuthor(self, self)
        VSCMGArrayTorqueMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return VSCMGArrayTorqueMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PixelLineFilterMsg_C.h"
%}
%include "cMsgCInterface/PixelLineFilterMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PixelLineFilterMsg;
%extend PixelLineFilterMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PixelLineFilterMsg
        if type(source) == type(self):
            PixelLineFilterMsg_C_subscribe(self, source)
        elif type(source) == PixelLineFilterMsg:
            PixelLineFilterMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PixelLineFilterMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PixelLineFilterMsg
        if type(source) == type(self):
            return (PixelLineFilterMsg_C_isSubscribedTo(self, source))
        elif type(source) == PixelLineFilterMsg:
            return (PixelLineFilterMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PixelLineFilterMsgRecorder
        self.header.isLinked = 1
        return PixelLineFilterMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PixelLineFilterMsg_C_addAuthor(self, self)
        if data:
            PixelLineFilterMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PixelLineFilterMsg_C_addAuthor(self, self)
        PixelLineFilterMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PixelLineFilterMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/OpNavFilterMsg_C.h"
%}
%include "cMsgCInterface/OpNavFilterMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct OpNavFilterMsg;
%extend OpNavFilterMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import OpNavFilterMsg
        if type(source) == type(self):
            OpNavFilterMsg_C_subscribe(self, source)
        elif type(source) == OpNavFilterMsg:
            OpNavFilterMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe OpNavFilterMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import OpNavFilterMsg
        if type(source) == type(self):
            return (OpNavFilterMsg_C_isSubscribedTo(self, source))
        elif type(source) == OpNavFilterMsg:
            return (OpNavFilterMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import OpNavFilterMsgRecorder
        self.header.isLinked = 1
        return OpNavFilterMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        OpNavFilterMsg_C_addAuthor(self, self)
        if data:
            OpNavFilterMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        OpNavFilterMsg_C_addAuthor(self, self)
        OpNavFilterMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return OpNavFilterMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SwDataMsg_C.h"
%}
%include "cMsgCInterface/SwDataMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SwDataMsg;
%extend SwDataMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SwDataMsg
        if type(source) == type(self):
            SwDataMsg_C_subscribe(self, source)
        elif type(source) == SwDataMsg:
            SwDataMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SwDataMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SwDataMsg
        if type(source) == type(self):
            return (SwDataMsg_C_isSubscribedTo(self, source))
        elif type(source) == SwDataMsg:
            return (SwDataMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SwDataMsgRecorder
        self.header.isLinked = 1
        return SwDataMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SwDataMsg_C_addAuthor(self, self)
        if data:
            SwDataMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SwDataMsg_C_addAuthor(self, self)
        SwDataMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SwDataMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CSSUnitConfigMsg_C.h"
%}
%include "cMsgCInterface/CSSUnitConfigMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CSSUnitConfigMsg;
%extend CSSUnitConfigMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CSSUnitConfigMsg
        if type(source) == type(self):
            CSSUnitConfigMsg_C_subscribe(self, source)
        elif type(source) == CSSUnitConfigMsg:
            CSSUnitConfigMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CSSUnitConfigMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CSSUnitConfigMsg
        if type(source) == type(self):
            return (CSSUnitConfigMsg_C_isSubscribedTo(self, source))
        elif type(source) == CSSUnitConfigMsg:
            return (CSSUnitConfigMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CSSUnitConfigMsgRecorder
        self.header.isLinked = 1
        return CSSUnitConfigMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CSSUnitConfigMsg_C_addAuthor(self, self)
        if data:
            CSSUnitConfigMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CSSUnitConfigMsg_C_addAuthor(self, self)
        CSSUnitConfigMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CSSUnitConfigMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/PowerNodeStatusMsg_C.h"
%}
%include "cMsgCInterface/PowerNodeStatusMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct PowerNodeStatusMsg;
%extend PowerNodeStatusMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import PowerNodeStatusMsg
        if type(source) == type(self):
            PowerNodeStatusMsg_C_subscribe(self, source)
        elif type(source) == PowerNodeStatusMsg:
            PowerNodeStatusMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe PowerNodeStatusMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import PowerNodeStatusMsg
        if type(source) == type(self):
            return (PowerNodeStatusMsg_C_isSubscribedTo(self, source))
        elif type(source) == PowerNodeStatusMsg:
            return (PowerNodeStatusMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import PowerNodeStatusMsgRecorder
        self.header.isLinked = 1
        return PowerNodeStatusMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        PowerNodeStatusMsg_C_addAuthor(self, self)
        if data:
            PowerNodeStatusMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        PowerNodeStatusMsg_C_addAuthor(self, self)
        PowerNodeStatusMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return PowerNodeStatusMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/AtmoPropsMsg_C.h"
%}
%include "cMsgCInterface/AtmoPropsMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct AtmoPropsMsg;
%extend AtmoPropsMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import AtmoPropsMsg
        if type(source) == type(self):
            AtmoPropsMsg_C_subscribe(self, source)
        elif type(source) == AtmoPropsMsg:
            AtmoPropsMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe AtmoPropsMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import AtmoPropsMsg
        if type(source) == type(self):
            return (AtmoPropsMsg_C_isSubscribedTo(self, source))
        elif type(source) == AtmoPropsMsg:
            return (AtmoPropsMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import AtmoPropsMsgRecorder
        self.header.isLinked = 1
        return AtmoPropsMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        AtmoPropsMsg_C_addAuthor(self, self)
        if data:
            AtmoPropsMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        AtmoPropsMsg_C_addAuthor(self, self)
        AtmoPropsMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return AtmoPropsMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CMEstDataMsg_C.h"
%}
%include "cMsgCInterface/CMEstDataMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CMEstDataMsg;
%extend CMEstDataMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CMEstDataMsg
        if type(source) == type(self):
            CMEstDataMsg_C_subscribe(self, source)
        elif type(source) == CMEstDataMsg:
            CMEstDataMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CMEstDataMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CMEstDataMsg
        if type(source) == type(self):
            return (CMEstDataMsg_C_isSubscribedTo(self, source))
        elif type(source) == CMEstDataMsg:
            return (CMEstDataMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CMEstDataMsgRecorder
        self.header.isLinked = 1
        return CMEstDataMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CMEstDataMsg_C_addAuthor(self, self)
        if data:
            CMEstDataMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CMEstDataMsg_C_addAuthor(self, self)
        CMEstDataMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CMEstDataMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/RealTimeFactorMsg_C.h"
%}
%include "cMsgCInterface/RealTimeFactorMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct RealTimeFactorMsg;
%extend RealTimeFactorMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import RealTimeFactorMsg
        if type(source) == type(self):
            RealTimeFactorMsg_C_subscribe(self, source)
        elif type(source) == RealTimeFactorMsg:
            RealTimeFactorMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe RealTimeFactorMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import RealTimeFactorMsg
        if type(source) == type(self):
            return (RealTimeFactorMsg_C_isSubscribedTo(self, source))
        elif type(source) == RealTimeFactorMsg:
            return (RealTimeFactorMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import RealTimeFactorMsgRecorder
        self.header.isLinked = 1
        return RealTimeFactorMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        RealTimeFactorMsg_C_addAuthor(self, self)
        if data:
            RealTimeFactorMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        RealTimeFactorMsg_C_addAuthor(self, self)
        RealTimeFactorMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return RealTimeFactorMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CamNav3Msg_C.h"
%}
%include "cMsgCInterface/CamNav3Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CamNav3Msg;
%extend CamNav3Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CamNav3Msg
        if type(source) == type(self):
            CamNav3Msg_C_subscribe(self, source)
        elif type(source) == CamNav3Msg:
            CamNav3Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CamNav3Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CamNav3Msg
        if type(source) == type(self):
            return (CamNav3Msg_C_isSubscribedTo(self, source))
        elif type(source) == CamNav3Msg:
            return (CamNav3Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CamNav3MsgRecorder
        self.header.isLinked = 1
        return CamNav3MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CamNav3Msg_C_addAuthor(self, self)
        if data:
            CamNav3Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CamNav3Msg_C_addAuthor(self, self)
        CamNav3Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CamNav3Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/InterRangeMsg_C.h"
%}
%include "cMsgCInterface/InterRangeMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct InterRangeMsg;
%extend InterRangeMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import InterRangeMsg
        if type(source) == type(self):
            InterRangeMsg_C_subscribe(self, source)
        elif type(source) == InterRangeMsg:
            InterRangeMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe InterRangeMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import InterRangeMsg
        if type(source) == type(self):
            return (InterRangeMsg_C_isSubscribedTo(self, source))
        elif type(source) == InterRangeMsg:
            return (InterRangeMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import InterRangeMsgRecorder
        self.header.isLinked = 1
        return InterRangeMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        InterRangeMsg_C_addAuthor(self, self)
        if data:
            InterRangeMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        InterRangeMsg_C_addAuthor(self, self)
        InterRangeMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return InterRangeMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNav4Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNav4Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNav4Msg;
%extend SmallBodyNav4Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav4Msg
        if type(source) == type(self):
            SmallBodyNav4Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNav4Msg:
            SmallBodyNav4Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNav4Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav4Msg
        if type(source) == type(self):
            return (SmallBodyNav4Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNav4Msg:
            return (SmallBodyNav4Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNav4MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNav4MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNav4Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNav4Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNav4Msg_C_addAuthor(self, self)
        SmallBodyNav4Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNav4Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallbodyDMCUKFMsg_C.h"
%}
%include "cMsgCInterface/SmallbodyDMCUKFMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallbodyDMCUKFMsg;
%extend SmallbodyDMCUKFMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallbodyDMCUKFMsg
        if type(source) == type(self):
            SmallbodyDMCUKFMsg_C_subscribe(self, source)
        elif type(source) == SmallbodyDMCUKFMsg:
            SmallbodyDMCUKFMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallbodyDMCUKFMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallbodyDMCUKFMsg
        if type(source) == type(self):
            return (SmallbodyDMCUKFMsg_C_isSubscribedTo(self, source))
        elif type(source) == SmallbodyDMCUKFMsg:
            return (SmallbodyDMCUKFMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallbodyDMCUKFMsgRecorder
        self.header.isLinked = 1
        return SmallbodyDMCUKFMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallbodyDMCUKFMsg_C_addAuthor(self, self)
        if data:
            SmallbodyDMCUKFMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallbodyDMCUKFMsg_C_addAuthor(self, self)
        SmallbodyDMCUKFMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallbodyDMCUKFMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavIntMsg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavIntMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavIntMsg;
%extend SmallBodyNavIntMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavIntMsg
        if type(source) == type(self):
            SmallBodyNavIntMsg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavIntMsg:
            SmallBodyNavIntMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavIntMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavIntMsg
        if type(source) == type(self):
            return (SmallBodyNavIntMsg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavIntMsg:
            return (SmallBodyNavIntMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavIntMsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavIntMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavIntMsg_C_addAuthor(self, self)
        if data:
            SmallBodyNavIntMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavIntMsg_C_addAuthor(self, self)
        SmallBodyNavIntMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavIntMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNav1Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNav1Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNav1Msg;
%extend SmallBodyNav1Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav1Msg
        if type(source) == type(self):
            SmallBodyNav1Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNav1Msg:
            SmallBodyNav1Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNav1Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav1Msg
        if type(source) == type(self):
            return (SmallBodyNav1Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNav1Msg:
            return (SmallBodyNav1Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNav1MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNav1MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNav1Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNav1Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNav1Msg_C_addAuthor(self, self)
        SmallBodyNav1Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNav1Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/DMCUKFMsg_C.h"
%}
%include "cMsgCInterface/DMCUKFMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct DMCUKFMsg;
%extend DMCUKFMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import DMCUKFMsg
        if type(source) == type(self):
            DMCUKFMsg_C_subscribe(self, source)
        elif type(source) == DMCUKFMsg:
            DMCUKFMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe DMCUKFMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import DMCUKFMsg
        if type(source) == type(self):
            return (DMCUKFMsg_C_isSubscribedTo(self, source))
        elif type(source) == DMCUKFMsg:
            return (DMCUKFMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import DMCUKFMsgRecorder
        self.header.isLinked = 1
        return DMCUKFMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        DMCUKFMsg_C_addAuthor(self, self)
        if data:
            DMCUKFMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        DMCUKFMsg_C_addAuthor(self, self)
        DMCUKFMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return DMCUKFMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CameraNavMsg_C.h"
%}
%include "cMsgCInterface/CameraNavMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CameraNavMsg;
%extend CameraNavMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CameraNavMsg
        if type(source) == type(self):
            CameraNavMsg_C_subscribe(self, source)
        elif type(source) == CameraNavMsg:
            CameraNavMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CameraNavMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CameraNavMsg
        if type(source) == type(self):
            return (CameraNavMsg_C_isSubscribedTo(self, source))
        elif type(source) == CameraNavMsg:
            return (CameraNavMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CameraNavMsgRecorder
        self.header.isLinked = 1
        return CameraNavMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CameraNavMsg_C_addAuthor(self, self)
        if data:
            CameraNavMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CameraNavMsg_C_addAuthor(self, self)
        CameraNavMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CameraNavMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavUKF4Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavUKF4Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavUKF4Msg;
%extend SmallBodyNavUKF4Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF4Msg
        if type(source) == type(self):
            SmallBodyNavUKF4Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavUKF4Msg:
            SmallBodyNavUKF4Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavUKF4Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF4Msg
        if type(source) == type(self):
            return (SmallBodyNavUKF4Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavUKF4Msg:
            return (SmallBodyNavUKF4Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF4MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavUKF4MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavUKF4Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNavUKF4Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavUKF4Msg_C_addAuthor(self, self)
        SmallBodyNavUKF4Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavUKF4Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNav2Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNav2Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNav2Msg;
%extend SmallBodyNav2Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav2Msg
        if type(source) == type(self):
            SmallBodyNav2Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNav2Msg:
            SmallBodyNav2Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNav2Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav2Msg
        if type(source) == type(self):
            return (SmallBodyNav2Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNav2Msg:
            return (SmallBodyNav2Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNav2MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNav2MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNav2Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNav2Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNav2Msg_C_addAuthor(self, self)
        SmallBodyNav2Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNav2Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavHOUMsg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavHOUMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavHOUMsg;
%extend SmallBodyNavHOUMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavHOUMsg
        if type(source) == type(self):
            SmallBodyNavHOUMsg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavHOUMsg:
            SmallBodyNavHOUMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavHOUMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavHOUMsg
        if type(source) == type(self):
            return (SmallBodyNavHOUMsg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavHOUMsg:
            return (SmallBodyNavHOUMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavHOUMsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavHOUMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavHOUMsg_C_addAuthor(self, self)
        if data:
            SmallBodyNavHOUMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavHOUMsg_C_addAuthor(self, self)
        SmallBodyNavHOUMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavHOUMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavUKF2Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavUKF2Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavUKF2Msg;
%extend SmallBodyNavUKF2Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF2Msg
        if type(source) == type(self):
            SmallBodyNavUKF2Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavUKF2Msg:
            SmallBodyNavUKF2Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavUKF2Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF2Msg
        if type(source) == type(self):
            return (SmallBodyNavUKF2Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavUKF2Msg:
            return (SmallBodyNavUKF2Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF2MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavUKF2MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavUKF2Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNavUKF2Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavUKF2Msg_C_addAuthor(self, self)
        SmallBodyNavUKF2Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavUKF2Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CameraNav2Msg_C.h"
%}
%include "cMsgCInterface/CameraNav2Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CameraNav2Msg;
%extend CameraNav2Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CameraNav2Msg
        if type(source) == type(self):
            CameraNav2Msg_C_subscribe(self, source)
        elif type(source) == CameraNav2Msg:
            CameraNav2Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CameraNav2Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CameraNav2Msg
        if type(source) == type(self):
            return (CameraNav2Msg_C_isSubscribedTo(self, source))
        elif type(source) == CameraNav2Msg:
            return (CameraNav2Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CameraNav2MsgRecorder
        self.header.isLinked = 1
        return CameraNav2MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CameraNav2Msg_C_addAuthor(self, self)
        if data:
            CameraNav2Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CameraNav2Msg_C_addAuthor(self, self)
        CameraNav2Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CameraNav2Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNavUKF5Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNavUKF5Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNavUKF5Msg;
%extend SmallBodyNavUKF5Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF5Msg
        if type(source) == type(self):
            SmallBodyNavUKF5Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNavUKF5Msg:
            SmallBodyNavUKF5Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNavUKF5Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF5Msg
        if type(source) == type(self):
            return (SmallBodyNavUKF5Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNavUKF5Msg:
            return (SmallBodyNavUKF5Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNavUKF5MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNavUKF5MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNavUKF5Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNavUKF5Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNavUKF5Msg_C_addAuthor(self, self)
        SmallBodyNavUKF5Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNavUKF5Msg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/CustomModuleMsg_C.h"
%}
%include "cMsgCInterface/CustomModuleMsg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct CustomModuleMsg;
%extend CustomModuleMsg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import CustomModuleMsg
        if type(source) == type(self):
            CustomModuleMsg_C_subscribe(self, source)
        elif type(source) == CustomModuleMsg:
            CustomModuleMsg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe CustomModuleMsg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import CustomModuleMsg
        if type(source) == type(self):
            return (CustomModuleMsg_C_isSubscribedTo(self, source))
        elif type(source) == CustomModuleMsg:
            return (CustomModuleMsg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import CustomModuleMsgRecorder
        self.header.isLinked = 1
        return CustomModuleMsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        CustomModuleMsg_C_addAuthor(self, self)
        if data:
            CustomModuleMsg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        CustomModuleMsg_C_addAuthor(self, self)
        CustomModuleMsg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return CustomModuleMsg_C_read(self)
    %}
};
%{
#include "cMsgCInterface/SmallBodyNav3Msg_C.h"
%}
%include "cMsgCInterface/SmallBodyNav3Msg_C.h"
%include "architecture/messaging/msgHeader.h"
typedef struct SmallBodyNav3Msg;
%extend SmallBodyNav3Msg_C {
    %pythoncode %{

    def subscribeTo(self, source):
        """subscribe to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav3Msg
        if type(source) == type(self):
            SmallBodyNav3Msg_C_subscribe(self, source)
        elif type(source) == SmallBodyNav3Msg:
            SmallBodyNav3Msg_cpp_subscribe(self, source)
        else:
            raise Exception('tried to subscribe SmallBodyNav3Msg to another message type')



    def isSubscribedTo(self, source):
        """check if self is subscribed to another message source"""
        from Basilisk.architecture.messaging import SmallBodyNav3Msg
        if type(source) == type(self):
            return (SmallBodyNav3Msg_C_isSubscribedTo(self, source))
        elif type(source) == SmallBodyNav3Msg:
            return (SmallBodyNav3Msg_cpp_isSubscribedTo(self, source))
        else:
            return 0


    def recorder(self, timeDiff=0):
        """create a recorder module for this message"""
        from Basilisk.architecture.messaging import SmallBodyNav3MsgRecorder
        self.header.isLinked = 1
        return SmallBodyNav3MsgRecorder(self, timeDiff)

    def init(self, data=None):
        """returns a Msg copy connected to itself"""
        SmallBodyNav3Msg_C_addAuthor(self, self)
        if data:
            SmallBodyNav3Msg_C_write(data, self, -1, 0)
        return self

    def write(self, payload, time=0, moduleID=0):
        """write the message payload.
        The 2nd argument is time in nanoseconds.  It is optional and defaults to 0.
        The 3rd argument is the module ID which defaults to 0.
        """
        SmallBodyNav3Msg_C_addAuthor(self, self)
        SmallBodyNav3Msg_C_write(payload, self, moduleID, time)  # msgs written in Python have 0 module ID
        return self

    def read(self):
        """read the message payload."""
        return SmallBodyNav3Msg_C_read(self)
    %}
};
