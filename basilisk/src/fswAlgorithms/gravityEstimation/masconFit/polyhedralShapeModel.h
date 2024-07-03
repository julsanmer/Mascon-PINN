/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at
 Boulder

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

#ifndef POLYHEDRAL_SHAPE_MODEL_H
#define POLYHEDRAL_SHAPE_MODEL_H

#include "shapeModel.h"

/**
 * The polyhedron shape class
 *
 * In this class, a polyhedron is defined by its triangular facets.
 * Each facet is defined by three vertices (they are triangles), and
 * each vertex is defined by its position relative to the center of
 * mass of the body.
 */
class PolyhedralShapeModel : public ShapeModel {
  public:
    /** Initialize all parameters necessary for the shape model.
     *
     * Will return an error message (string) if `xyzVertex` or `orderFacet` were not set.
     * Otherwise, returns an empty optional.
     */
    std::optional<std::string> initializeParameters() override;
    
    /** Returns the altitude w.r.t. body.
     *
     * The position is given in the body-fixed reference frame.
     */
    double computeAltitude(const Eigen::Vector3d& position_planetFixed) const override;

    /** Returns the closest point in the polyhedron.
     *
     * The position is given in the body-fixed reference frame.
     */
    Eigen::Vector3d closestPoint(const Eigen::Vector3d& position_planetFixed) const override;
    
    /** Returns the closest face in the polyhedron.
     *
     * The position is given in the body-fixed reference frame.
     */
    int closestFacet(const Eigen::Vector3d& position_planetFixed) const;
    
    /** Returns a bool that is true when the point is exterior.
     *
     * The position is given in the body-fixed reference frame.
     */
    bool isExterior(const Eigen::Vector3d& position_planetFixed) const override;
    
  public:    
    /**
     * This matrix contains the position of every vertex of this
     * polyhedron, in meters. Each row corresponds to a different
     * vertex, while each column corresponds to x, y, z respectively.
     */
    Eigen::MatrixX3d xyzVertex;

    /**
     * This matrix defines the facets of the matrix. Each row
     * contains three numbers, each of them corresponding to the
     * index of a vertex, as defined in xyzVertex. These three
     * vertices define a single facet of the polyhedron.
     *
     * Note that the order of the vertex index is important: the facets
     * must all be outward pointing.
     */
    Eigen::MatrixX3i orderFacet;
private:
    /**
     * This matrix contains the center of each facet [m].
     */
    Eigen::MatrixX3d xyzFacet;
    
    /**
     * This matrix contains the normal of each facet [-].
     */
    Eigen::MatrixX3d normalFacet;
    
    /**
     * This matrix is 3x2 and contains the cirscumcribing
     * cuboid limits [m].
     */
    Eigen::Matrix3Xd xyzLimit;
};

#endif /* POLYHEDRAL_SHAPE_MODEL_H */
