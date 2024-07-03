/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "polyhedralShapeModel.h"

std::optional<std::string> PolyhedralShapeModel::initializeParameters()
{
    // If data hasn't been loaded, quit and return failure
    if(this->xyzVertex.size() == 0 || this->orderFacet.size() == 0)
    {
        return "Could not initialize polyhedral data: the vertex (xyzVertex) or facet (orderFacet) "
        "were not provided.";;
    }
    
    const size_t nFacet = this->orderFacet.rows();
    
    // Preallocate facet normal and center
    Eigen::Vector3d nf;
    this->normalFacet.setZero(nFacet, 3);
    this->xyzFacet.setZero(nFacet, 3);
    
    // For the facet in loop: declare
    // vertex indexes and positions
    Eigen::Vector3i v;
    int i, j, k;
    Eigen::Vector3d xyz1, xyz2, xyz3, e21, e32;;
    
    // Loop through each facet
    for (unsigned int m = 0; m < nFacet; m++){
        // Obtain vertex indexes of the facet
        v = this->orderFacet.row(m);
        i = v[0] - 1;
        j = v[1] - 1;
        k = v[2] - 1;
        
        // Extract vertexes position
        xyz1 = this->xyzVertex.row(i);
        xyz2 = this->xyzVertex.row(j);
        xyz3 = this->xyzVertex.row(k);
        
        // Compute facet normal and center
        e21 = xyz2 - xyz1;
        e32 = xyz3 - xyz2;
        nf = e21.cross(e32) / e21.cross(e32).norm();
        this->normalFacet.row(m) = nf.transpose();        
        this->xyzFacet.row(m) = (xyz1 + xyz2 + xyz3) / 3;
    }
    
    // Assign cuboid limits
    this->xyzLimit.setZero(3,2);
    this->xyzLimit << this->xyzVertex.col(0).minCoeff(), this->xyzVertex.col(0).maxCoeff(),
                      this->xyzVertex.col(1).minCoeff(), this->xyzVertex.col(1).maxCoeff(),
                      this->xyzVertex.col(2).minCoeff(), this->xyzVertex.col(2).maxCoeff();
    
    return {};
}

Eigen::Vector3d
PolyhedralShapeModel::closestPoint(const Eigen::Vector3d& position_planetFixed) const
{    
    int idxClosest;
    idxClosest = this->closestFacet(position_planetFixed);

    // Get the closest facet center
    Eigen::Vector3d xyzClosest;
    xyzClosest = this->xyzFacet.row(idxClosest);
    
    return xyzClosest;
}

double
PolyhedralShapeModel::computeAltitude(const Eigen::Vector3d& position_planetFixed) const
{
    const size_t nFacet = this->orderFacet.rows();

    // Stack spacecraft position
    Eigen::MatrixXd positionStack;
    positionStack = position_planetFixed.rowwise().replicate(nFacet).transpose();
    
    // Compute squared distance w.r.t. polyhedron facets
    //Eigen::VectorXd distFacet;
    //distFacet = (positionStack - this->xyzFacet).rowwise().squaredNorm();
    
    // Compute closest facet
    int idxClosest;
    idxClosest = this->closestFacet(position_planetFixed);
    
    // Get the closest facet center
    Eigen::Vector3d xyzClosest, normalClosest, dxyzClosest;
    xyzClosest = this->xyzFacet.row(idxClosest);
    normalClosest = this->normalFacet.row(idxClosest);
    dxyzClosest = position_planetFixed - xyzClosest;
    
    Eigen::Vector3i v;
    int i, j, k;
    Eigen::Vector3d xyz1, xyz2, xyz3;
    
    // Obtain minimum distance w.r.t. facets
    double altitude, distance1, distance2, distance3;
    altitude = abs(dxyzClosest.transpose() * normalClosest);
    
    // Obtain minimum distance w.r.t. facets
    //double altitude;
    //altitude = sqrt(distFacet.minCoeff());
    
    return altitude;
}

int
PolyhedralShapeModel::closestFacet(const Eigen::Vector3d& position_planetFixed) const
{
    const size_t nFacet = this->orderFacet.rows();
    
    // Stack spacecraft position
    Eigen::MatrixXd position_stack;
    position_stack = position_planetFixed.rowwise().replicate(nFacet).transpose();

    // Compute index of closest facet
    int idxClosest;
    (position_stack - this->xyzFacet).rowwise().squaredNorm().minCoeff(&idxClosest);

    // Get the closest facet center
    Eigen::Vector3d xyzClosest;
    xyzClosest = this->xyzFacet.row(idxClosest);
    
    // Initialize closest facet index
    //int idxClosest = 0;
    
    // Initialize closest distance
    //double distanceClosest = position_planetFixed.norm();
    
    // Declare current facet variables
    //double distance;
    //Eigen::Vector3d normal, xyz;
    
    // Loop through facets
    //for (unsigned int m = 0; m < nFacet; m++){
        // Retrieve facet center and normal
        //xyz = this->xyzFacet.row(m);
        //normal = this->normalFacet.row(m);
        
        // Compute distance to facet
        //distance = (position_planetFixed - xyz).transpose();// * normal;
        
        // If distance is not negative and is less than
        // current closest value
        //if (distance > 0 && distance < distanceClosest){
            // Save current index as closest
            //idxClosest = m;
            //distanceClosest = distance;
        //}
    //}
    
    return idxClosest;
}

bool
PolyhedralShapeModel::isExterior(const Eigen::Vector3d &position_planetFixed) const
{
    // Preallocate isExterior flag
    bool isExterior;
    isExterior = true;
    
    // Extract cuboid limits
    double xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = this->xyzLimit(0,0);
    xmax = this->xyzLimit(0,1);
    ymin = this->xyzLimit(1,0);
    ymax = this->xyzLimit(1,1);
    zmin = this->xyzLimit(2,0);
    zmax = this->xyzLimit(2,1);
    
    // Check if spacecraft is outside cuboid
    if (position_planetFixed(0) > xmax or position_planetFixed(0) < xmin){
        return isExterior;
    }
    if (position_planetFixed(1) > ymax or position_planetFixed(1) < ymin){
        return isExterior;
    }
    if (position_planetFixed(2) > zmax or position_planetFixed(2) < zmin){
        return isExterior;
    }
    
    const size_t nFacet = this->orderFacet.rows();
    
    // Initialize laplacian
    double laplacian = 0;
    
    // For the facet in loop: declare vertex indexes
    // and relative positions w.r.t. spacecraft
    Eigen::Vector3i v;
    int i, j, k;
    Eigen::Vector3d ri, rj, rk;
    
    // Initialize solid angle variables
    double wy, wx, wf;
        
    // Loop through each facet
    for (unsigned int m = 0; m < nFacet; m++){
        // Obtain vertex indexes of the facet
        v = this->orderFacet.row(m);
        i = v[0] - 1;
        j = v[1] - 1;
        k = v[2] - 1;
        
        // Compute facet vertexes relative position w.r.t. spacecraft
        ri = this->xyzVertex.row(i).transpose() - position_planetFixed;
        rj = this->xyzVertex.row(j).transpose() - position_planetFixed;
        rk = this->xyzVertex.row(k).transpose() - position_planetFixed;
        
        // Compute facet solid angle
        wy = ri.transpose()*rj.cross(rk);
        wx = ri.norm()*rj.norm()*rk.norm()
             + ri.norm()*rj.transpose()*rk
             + rj.norm()*rk.transpose()*ri
             + rk.norm()*ri.transpose()*rj;
        wf = 2*atan2(wy, wx);
        
        // Add solid angle to laplacian
        laplacian -= wf;
    }
    
    // Assign isExterior to false
    // if laplacian = -4*pi
    if (laplacian < -2*M_PI){
        isExterior = false;
    }
    
    return isExterior;
}
