
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

#include "cameraNav4.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <iostream>
#include <math.h>
#include "architecture/utilities/linearAlgebra.h"
#include <time.h>

CamObjectShape::CamObjectShape()
{
    return;
}

CamObjectShape::~CamObjectShape()
{
    return;
}

void CamObjectShape::initializeParameters()
{
    /* Compute facet center and normal */
    Eigen::Vector3d v, xyz1, xyz2, xyz3, e21, e32;
    int i, j, k;
    this->nFacet = this->orderFacet.rows();
    this->xyzFacet.setZero(this->nFacet,3);
    this->normalFacet.setZero(this->nFacet,3);
    
    /* Preallocate variables associated to landmarks */
    this->nLandmark = idxLandmark.size();
    this->xyzLandmark.setZero(this->nLandmark,3);
    this->normalLandmark.setZero(this->nLandmark,3);
    int contLandmark;
    contLandmark = 0;
            
    /* Loop through shape facets */
    for (unsigned int m=0; m < this->nFacet; m++){
        /* Fill auxiliary variables with vertex order on each facet */
        v = orderFacet.row(m);
        i = v[0];
        j = v[1];
        k = v[2];
        
        xyz1 = this->xyzVertex.row(i);
        xyz2 = this->xyzVertex.row(j);
        xyz3 = this->xyzVertex.row(k);
        this->xyzFacet.row(m) = (xyz1 + xyz2 + xyz3)/3;
        
        /* Compute two edge vectors and normal to facet */
        e21 = xyz2 - xyz1;
        e32 = xyz3 - xyz2;
        this->normalFacet.row(m) = e21.cross(e32) / e21.cross(e32).norm();
                
        /* Check if it is landmark */
        if (m == this->idxLandmark(contLandmark)){
            /* Add error to landmark */
            this->xyzLandmark.row(contLandmark) = this->xyzFacet.row(m);
            this->normalLandmark.row(contLandmark) = this->normalFacet.row(m);
            
            /* Sum landmarks counter */
            contLandmark += 1;
        }
    }
}

Eigen::MatrixXd CamObjectShape::computeSnapshot(Eigen::Vector3d r, Eigen::Vector3d eS, double f, double wPixel, double nxPixel, double nyPixel)
{
    /* Compute radius, longitude and latitude */
    double rad, lon, lat;
    rad = r.norm();
    lon = atan2(r(1),r(0));
    lat = asin(r(2)/rad);
    
    /* Compute dcm for change to camera coordinates */
    Eigen::Matrix3d dcm;
    double dcm_array[3][3];
    double theta[3];
    v3Set(lon, -(M_PI_2 + lat), M_PI_2, theta);
    Euler3232C(theta, dcm_array);
    dcm = cArray2EigenMatrix3d(*dcm_array);
    
    /* Compute pixel width */
    double FOVx;
    double FOVy;
    FOVx = 2*atan2(nxPixel*wPixel,2*f);
    FOVy = 2*atan2(nyPixel*wPixel,2*f);
    
    /* Preallocate pixels */
    Eigen::MatrixXd pixel;
    pixel.setZero(this->nFacet,2);
    
    /* Declare elevation w.r.t. camera, pixels, landmark current relative and absolute positions */
    double angCam, angSun;
    double px, py;
    Eigen::Vector3d xyzFacet_i, dri;
    
    /* Loop through landmarks */
    for (int i=0; i < this->nFacet; i++){
        /* Extract landmark position to be evaluated */
        xyzFacet_i = this->xyzFacet.row(i).transpose();
        
        /* Compute elevation angle between spacecraft and landmark for camera */
        angCam = M_PI_2 - safeAcos(xyzFacet_i.dot(r)/(r.norm()*xyzFacet_i.norm()));
        
        /* Compute sun incident angle */
        angSun = M_PI_2 - safeAcos(xyzFacet_i.dot(eS)/(eS.norm()*xyzFacet_i.norm()));
            
        /* Check if it is visible by camera and compute pixel */
        if (angCam > 0){
            /* Compute relative distance between landmark and spacecraft */
            dri = dcm*(xyzFacet_i - r);
            
            /* Compute pixel location */
            px = f*dri(0)/(dri(2)*wPixel);
            py = f*dri(1)/(dri(2)*wPixel);
            
            /* Apply pixelization error*/
            if (px > 0){
                px = ceil(px);
            }
            else if(px < 0){
                px = floor(px);
            }
            if (py > 0){
                py = ceil(py);
            }
            else if(py < 0){
                py = floor(py);
            }
            
            /* Check if pixel is within resolution */
            if ((abs(px) <= nxPixel/2) && (abs(py) <= nyPixel/2)){
                /* Check if pixel is illuminated */
                if (angSun > 0){
                    /* Add pixel */
                    pixel.row(i) << px, py;
                }
            }
        }
    }
    return pixel;
}

Eigen::VectorXd CamObjectShape::computeLaplacian(Eigen::MatrixXd posBatch)
{
    /* Extract dimensions */
    int nBatch;
    nBatch = posBatch.rows();
    
    /* Preallocate output */
    Eigen::VectorXd lU;
    lU.setZero(nBatch);
    
    /* Declare auxiliary variables */
    Eigen::Vector3d pos;
    int i, j, k;
    Eigen::Vector3d v;
    Eigen::Vector3d ri, rj, rk;
    double wy, wx, wf;
        
    /* Loop through all positions */
    for (unsigned int l = 0; l < nBatch; l++){
        /* Extract position */
        pos = posBatch.row(l).transpose();
        
        /* Loop through each facet */
        for (unsigned int m = 0; m < this->nFacet; m++){
            /* Fill auxiliary variables with vertex order on each facet */
            v = this->orderFacet.row(m);
            i = v[0];
            j = v[1];
            k = v[2];
            
            /* Compute vectors and norm from each vertex to the evaluation position */
            ri = this->xyzVertex.row(i).transpose() - pos;
            rj = this->xyzVertex.row(j).transpose() - pos;
            rk = this->xyzVertex.row(k).transpose() - pos;
            
            /* Compute solid angle for the current facet */
            wy = (ri.transpose())*rj.cross(rk);
            wx = ri.norm()*rj.norm()*rk.norm() + ri.norm()*rj.transpose()*rk + rj.norm()*rk.transpose()*ri + rk.norm()*ri.transpose()*rj;
            wf = -2*atan2(wy, wx);
            
            /* Add term */
            lU(l) += wf;
        }
    }
    return lU;
}

/*! @brief Creates an instance of the GroundLocation class with a minimum elevation of 10 degrees,
 @return void
 */
CameraNav4::CameraNav4()
{
    this->rLSQ_BP_P(3);
    this->PLSQ(3,3);
    
    this->maskangleCam = 0;
    this->maskangleSun = 0;
}

/*! Empty destructor method.
 @return void
 */
CameraNav4::~CameraNav4()
{
}

/*! Initialize C-wrapped output messages */
void CameraNav4::SelfInit(){
    CamNav3Msg_C_init(&this->camNav3OutMsgC);
}

/*! Resets the internal position to the specified initial position.*/
void CameraNav4::Reset(uint64_t CurrentSimNanos)
{
    /* Compute field of views */
    this->FOVx = 2*atan2(this->nxPixel*this->wPixel,2*this->f);
    this->FOVy = 2*atan2(this->nyPixel*this->wPixel,2*this->f);
        
    /* Initialize body object */
    this->body.initializeParameters();
        
    /* Initialize landmarks database */
    this->nLandmark = this->body.nLandmark;
    this->xyzLandmark = this->body.xyzLandmark + this->dxyzLandmark;

    /* Initialize landmarks pixels and visible flag */
    this->pixelLandmark(this->nLandmark,2);
    this->visibleLandmark(this->nLandmark);
}

void CameraNav4::computePixelBatch(Eigen::MatrixXd posBatch, Eigen::MatrixXd eSBatch){
    /* Preallocate output */
    int nBatch;
    nBatch = posBatch.rows();
    
    this->pixelBatch.setZero(nBatch,2*this->nLandmark);
    this->visibleBatch.setZero(nBatch);
    this->nVisibleBatch.setZero(nBatch);
    this->latlonBatch.setZero(nBatch,2);
    
    /* Compute dcm for change to camera coordinates */
    double radB, lonB, latB;
    Eigen::Matrix3d dcm;
    double dcm_array[3][3];
    double theta[3];
    
    /* Define auxiliary variables */
    int nVisible;
    Eigen::Vector3d r, eS, xyzLandmark_i, normalLandmark_i, dr_i;
    double angleCam, angleSun, px, py;
    
    /* Loop through bathc */
    for (int j=0; j < nBatch; j++){
        if (j % (nBatch/100) == 0){
            std::cout << j/(nBatch/100) << "% completed" << '\n';
        }
        
        /* Extract spacecraft position and Sun's vector */
        r = posBatch.row(j).transpose();
        eS = eSBatch.row(j).transpose();
        
        /* Obtain orientation w.r.t. camera */
        radB = r.norm();
        lonB = atan2(r(1),r(0));
        latB = asin(r(2)/radB);
        v3Set(lonB, -(M_PI_2 + latB), M_PI_2, theta);
        Euler3232C(theta, dcm_array);
        dcm = cArray2EigenMatrix3d(*dcm_array);
        
        /* Reset visible landmarks */
        nVisible = 0;
                
        /* Loop through landmarks */
        for (int i=0; i < this->nLandmark; i++){
            /* Obtain facet center normal to be evaluated */
            xyzLandmark_i = this->body.xyzLandmark.row(i).transpose();
            normalLandmark_i = this->body.normalLandmark.row(i).transpose();
            
            /* Compute elevation angle between spacecraft and facet as view from camera by camera */
            angleCam = M_PI_2 - safeAcos(normalLandmark_i.dot(r)/(r.norm()*normalLandmark_i.norm()));
            
            /* Compute sun incident angle */
            angleSun = M_PI_2 - safeAcos(normalLandmark_i.dot(eS)/(eS.norm()*normalLandmark_i.norm()));
            
            /* Check if it is visible by camera and Sun */
            if (angleCam > this->maskangleCam){
                /* Compute relative distance between facet center and spacecraft */
                dr_i = dcm*(xyzLandmark_i - r);
                
                /* Compute pixel location */
                px = this->f*dr_i(0)/(dr_i(2)*this->wPixel);
                py = this->f*dr_i(1)/(dr_i(2)*this->wPixel);
                
                /* Check if pixel is within camera resolution */
                if ((abs(px) <= this->nxPixel/2) && (abs(py) <= this->nyPixel/2)){
                    /* Check if facet pixel is illuminated by Sun */
                    if (angleSun > this->maskangleSun){
                        /* Markers */
                        Eigen::MatrixXd dposRay;
                        Eigen::VectorXd dray;
                        dray.setZero(4);
                        Eigen::VectorXd lap;
                        dray << 10, 50, 250, 1000;
                        dposRay.setZero(4,3);
                        for (int k=0; k<4; k++){
                            dposRay.row(k) = xyzLandmark_i - dray(k)*(xyzLandmark_i - r)/((xyzLandmark_i - r).norm());
                        }
                        lap = body.computeLaplacian(dposRay);
                        /*std::cout << lap << '\n';*/
                        
                        if (abs(lap.sum()) < 2*MPI){
                            /* Fill that it is visible by Sun */
                            /*this->visibleLandmark(i) = 1;*/
                            nVisible += 1;
                            /* Apply pixelization error*/
                            if (px > 0){
                                this->pixelBatch(j,2*i) = ceil(px) - 0.5;
                            }
                            else if(px < 0){
                                this->pixelBatch(j,2*i) = floor(px) + 0.5;
                            }
                            if (py > 0){
                                this->pixelBatch(j,2*i+1) = ceil(py) - 0.5;
                            }
                            else if(py < 0){
                                this->pixelBatch(j,2*i+1) = floor(py) + 0.5;
                            }
                        }
                    }
                }
            }
        }
        if (nVisible > 2){
            this->visibleBatch(j) = 1;
        }
        this->nVisibleBatch(j) = nVisible;
        this->latlonBatch.row(j) << latB, lonB;
    }
}

/* */
void CameraNav4::computePosBatch(){
    /* Preallocate variables */
    int n, nBatch;
    Eigen::Vector3d rL_P, vecLOS_C, vecLOS_P, b;
    Eigen::Matrix3d A, dcm;
    double px, py, u, v, r, lonB, latB;
    double dcm_array[3][3];
    double theta[3];
    
    /* Preallocate output */
    nBatch = this->pixelBatch.rows();
    this->rNavBatch.setZero(nBatch,3);
    
    /* Loop through batch */
    for (int j = 0; j < nBatch; j++){
        /* Check if there are enough landmarks */
        if (this->visibleBatch(j) == 1){
            /* Obtain orientation w.r.t camera */
            latB = this->latlonBatch(j,0);
            lonB = this->latlonBatch(j,1);
            v3Set(lonB, -(M_PI_2 + latB), M_PI_2, theta);
            Euler3232C(theta, dcm_array);
            dcm = cArray2EigenMatrix3d(*dcm_array);
            
            /* Check number of landmarks to be evaluated */
            n = this->nVisibleBatch(j);
            
            /* Define A*r_P = b matrix and vector*/
            A.setIdentity();
            A *= n;
            b.setZero(3);
            
            /* Loop through landmarks */
            for (int i = 0; i < this->nLandmark; i++){
                if (abs(this->pixelBatch(j,2*i)) > 0.25){
                    rL_P = this->xyzLandmark.row(i);
                    px = this->pixelBatch(j,2*i);
                    py = this->pixelBatch(j,2*i+1);
                    
                    /* Transform pixel to los directions from camera */
                    u = px*this->wPixel/this->f;
                    v = py*this->wPixel/this->f;
                    
                    /* Make LOS vector unitary */
                    vecLOS_C << u, v, 1;
                    vecLOS_C /= vecLOS_C.norm();
                    
                    /* Rotate LOS to planet rotating frame */
                    vecLOS_P = dcm.transpose()*vecLOS_C;
                    
                    /* Add line to least-squares matrix and vector */
                    A -= vecLOS_P*vecLOS_P.transpose();
                    b += rL_P - (rL_P.transpose()*vecLOS_P)*vecLOS_P;
                }
            }
            
            /* Solve the linear system */
            this->rNavBatch.row(j) = A.lu().solve(b);
        }
    }
}


/*! Computes eclipsed area in the image plane.
 *
 */
void CameraNav4::computePixel(){
    /* Preallocate pixels and visible landmarks */
    this->pixelLandmark.setZero(this->nLandmark,2);
    this->visibleLandmark.setZero(this->nLandmark);
    
    /* Declare elevation angles w.r.t. camera, pixels, landmark current relative and absolute positions */
    double angleCam, angleSun;
    double px, py;
    Eigen::Vector3d xyzLandmark_i, normalLandmark_i, dr_i;
    
    /* Loop through facets */
    for (int i=0; i < this->nLandmark; i++){
        /* Obtain facet center normal to be evaluated */
        xyzLandmark_i = this->body.xyzLandmark.row(i).transpose();
        normalLandmark_i = this->body.normalLandmark.row(i).transpose();
        
        /* Compute elevation angle between spacecraft and facet as view from camera by camera */
        angleCam = M_PI_2 - safeAcos(normalLandmark_i.dot(this->r_BP_P)/(this->r_BP_P.norm()*normalLandmark_i.norm()));
        
        /* Compute sun incident angle */
        angleSun = M_PI_2 - safeAcos(normalLandmark_i.dot(this->e_SP_P)/(this->e_SP_P.norm()*normalLandmark_i.norm()));
            
        /* Check if it is visible by camera and Sun */
        if (angleCam > this->maskangleCam){
            /* Compute relative distance between facet center and spacecraft */
            dr_i = this->dcm_BP*(xyzLandmark_i - this->r_BP_P);
            
            /* Compute pixel location */
            px = this->f*dr_i(0)/(dr_i(2)*this->wPixel);
            py = this->f*dr_i(1)/(dr_i(2)*this->wPixel);
            
            /* Check if pixel is within camera resolution */
            if ((abs(px) <= this->nxPixel/2) && (abs(py) <= this->nyPixel/2)){
                /* Check if facet pixel is illuminated by Sun */
                if (angleSun > this->maskangleSun){
                    /* Fill that it is visible by Sun */
                    this->visibleLandmark(i) = 1;
                    
                    /* Apply pixelization error*/
                    if (px > 0){
                        this->pixelLandmark(i,0) = ceil(px);
                    }
                    else if(px < 0){
                        this->pixelLandmark(i,0) = floor(px);
                    }
                    if (py > 0){
                        this->pixelLandmark(i,1) = ceil(py);
                    }
                    else if(py < 0){
                        this->pixelLandmark(i,1) = floor(py);
                    }
                }
            }
        }
    }
    /* Compute number of visible landmarks */
    this->nVisibleLandmark = this->visibleLandmark.sum();
}


/*! Computes position and uncertainty based on least-squares..
 *
 */
void CameraNav4::computePosLSQ(){
    /* Start measuring time */
    clock_t start, end;
    start = clock();
    
    /* Check if there is enough brigthness to compute a position */
    if (this->nVisibleLandmark > 2){
        this->navSolution = 1;
        
        /* Check number of landmarks to be evaluated */
        int n;
        n = this->nVisibleLandmark;
        int contVisibleLandmark;
        contVisibleLandmark = 0;
    
        /* Define current landmark index, landmark, centred pixels and LOS directions */
        Eigen::Vector3d rL_P;
        double pxC, pyC;
        double u, v;
    
        /* Define current LOS vector in camera frame and planet */
        Eigen::Vector3d vecLOS_C;
        Eigen::Vector3d vecLOS_P;
    
        /* Define A*r_P = b matrix and vector*/
        Eigen::Matrix3d A;
        Eigen::Vector3d b;
        A.setIdentity();
        A *= n;
        b.setZero(3);
    
        /* Define auxiliary variables for uncertainty computation */
        Eigen::MatrixXd ALSQ;
        Eigen::VectorXd bLSQ;
        ALSQ.setZero(3*n,3);
        bLSQ.setZero(3*n);
        double xi, J;
        
        /* Preallocate centred pixel */
        Eigen::Vector2d pL;
    
        /* Loop through landmarks */
        for (int i=0; i < this->nLandmark; i++){
            if (this->visibleLandmark(i) == 1){
                rL_P = this->xyzLandmark.row(i);
                pL = this->pixelLandmark.row(i);
                                
                /* Compute the pixel centre location */
                if (pL(0) > 0){
                    pxC = pL(0) - 0.5;
                }
                else{
                    pxC = pL(0) + 0.5;
                }
                if (pL(1) > 0){
                    pyC = pL(1) - 0.5;
                }
                else{
                    pyC = pL(1) + 0.5;
                }
            
                /* Untangle pixel to los directions from camera */
                u = pxC*this->wPixel / this->f;
                v = pyC*this->wPixel / this->f;
            
                /* Make LOS vector unitary */
                vecLOS_C << u, v, 1;
                vecLOS_C /= vecLOS_C.norm();
            
                /* Rotate LOS to planet rotating frame */
                vecLOS_P = this->dcm_BP.transpose()*vecLOS_C;
            
                /* Add line to least-squares matrix and vector */
                A -= vecLOS_P*vecLOS_P.transpose();
                b += rL_P - (rL_P.transpose()*vecLOS_P)*vecLOS_P;
            
                /* Compute jacobian */
                ALSQ.block(3*contVisibleLandmark,0,3,3) << 0, vecLOS_P(2), -vecLOS_P(1), -vecLOS_P(2), 0, vecLOS_P(0), vecLOS_P(1), -vecLOS_P(0), 0;
                bLSQ.segment(3*contVisibleLandmark,3) << rL_P(1)*vecLOS_P(2) - rL_P(2)*vecLOS_P(1), rL_P(2)*vecLOS_P(0) - rL_P(0)*vecLOS_P(2), rL_P(0)*vecLOS_P(1) - rL_P(1)*vecLOS_P(0);
                contVisibleLandmark += 1;
            }
        }
    
        /* Solve the linear system */
        this->rLSQ_BP_P = A.lu().solve(b);
        /*this->rLSQ_BP_P = ALSQ.colPivHouseholderQr().solve(bLSQ);*/
        /*std::cout << this->rLSQ_BP_P;
        std::cout << this->r_BP_P;*/
    
        /* Compute objective function, xi square and covariance */
        J = (ALSQ*this->rLSQ_BP_P - bLSQ).transpose()*(ALSQ*this->rLSQ_BP_P - bLSQ);
        xi = sqrt(J / (3*n - 3));
        this->PLSQ = pow(xi,2)*(ALSQ.transpose()*ALSQ).inverse();
    }
    else {
        /* Set zero output */
        this->navSolution = 0;
        this->rLSQ_BP_P.setZero(3);
        this->PLSQ.setZero(3,3);
    }
    
    /* Stop measuring time and calculate the elapsed time */
    end = clock();
    this->tcpu = double(end - start) / double(CLOCKS_PER_SEC);
}


/*! Read module messages
*/
void CameraNav4::ReadMessages()
{
    /* Read in the input messages */
    this->spacecraftState = this->scStateInMsg();
    this->ephemeris = this->ephemerisInMsg();
    
    /* Extract dcm and angular velocity of the small body */
    double dcm_PN_array[3][3];
    MRP2C(this->ephemeris.sigma_BN, dcm_PN_array);
    this->dcm_PN = cArray2EigenMatrix3d(*dcm_PN_array);
    
    /* Compute sun line projection on small body centred fixed frame */
    Eigen::Vector3d r_PS_N;
    r_PS_N = cArray2EigenVector3d(this->ephemeris.r_BdyZero_N);
    this->e_SP_P = this->dcm_PN*(-r_PS_N / r_PS_N.norm());
    /*std::cout << this->e_SP_P << '\n';*/
    
    /* Relative state between spacecraft and small body in small body centred fixed frame */
    this->r_BP_P = this->dcm_PN*(cArray2EigenVector3d(this->spacecraftState.r_BN_N) -  cArray2EigenVector3d(this->ephemeris.r_BdyZero_N));
    
    /* Compute radius, longitude and latitude */
    double radB, lonB, latB;
    radB = r_BP_P.norm();
    lonB = atan2(r_BP_P(1),r_BP_P(0));
    latB = asin(r_BP_P(2)/radB);
    
    /* Compute dcm for change to camera coordinates */
    double dcm_BP_array[3][3];
    double theta_BP[3];
    v3Set(lonB, -(M_PI_2 + latB), M_PI_2, theta_BP);
    Euler3232C(theta_BP, dcm_BP_array);
    this->dcm_BP = cArray2EigenMatrix3d(*dcm_BP_array);
}

/*! write module messages
*/
void CameraNav4::WriteMessages(uint64_t CurrentClock)
{
    /* Create output msg buffers */
    CamNav3MsgPayload camNav3OutMsgBuffer;
    
    /* Zero the output message buffers before assigning values */
    camNav3OutMsgBuffer = this->camNav3OutMsg.zeroMsgPayload;
    
    /* Assign values to the small body camera output message */
    eigenMatrixXd2CArray(this->rLSQ_BP_P, camNav3OutMsgBuffer.r_BP_P);
    eigenMatrixXd2CArray(this->PLSQ, *camNav3OutMsgBuffer.P);
    camNav3OutMsgBuffer.nLvisible = this->nVisibleLandmark;
    camNav3OutMsgBuffer.navSolution = this->navSolution;
    camNav3OutMsgBuffer.tcpu = this->tcpu;
    
    /* Write to the C++-wrapped output messages */
    this->camNav3OutMsg.write(&camNav3OutMsgBuffer, this->moduleID, CurrentClock);

    /* Write to the C-wrapped output messages */
    CamNav3Msg_C_write(&camNav3OutMsgBuffer, &this->camNav3OutMsgC, this->moduleID, CurrentClock);
}


/*!
 update module 
 @param CurrentSimNanos
 */
void CameraNav4::UpdateState(uint64_t CurrentSimNanos)
{
    /* Read message */
    this->ReadMessages();
    
    /* Compute equivalent pixels */
    this->computePixel();
    
    /* Compute least-squares */
    this->computePosLSQ();
    
    /* Write mesage */
    this->WriteMessages(CurrentSimNanos);
}
