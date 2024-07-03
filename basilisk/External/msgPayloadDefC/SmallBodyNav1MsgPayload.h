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

#ifndef SMALLBODY_NAV1_MESSAGE_H
#define SMALLBODY_NAV1_MESSAGE_H

/*! @brief Structure used to define the output of the sub-module.  This is the same
    output message that is used by all sub-modules in the module folder. */
typedef struct{
    double state[9];     //!< [units] state
    double covar[9][9];  //!< [units] covariance
    double skew[9];      //!< [units] skewness
    double kurt[9];      //!< [units] kurtosis
    double meas[3];      //!< [units] measurement
    double tcpu;         //!< [units] computation time
}SmallBodyNav1MsgPayload;


#endif
