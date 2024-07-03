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

#include "pinn2GravityModel.h"
#include <iostream>
#include "simulation/dynamics/_GeneralModuleFiles/gravityEffector.h"


std::optional<std::string> PINN2GravityModel::initializeParameters()
{
    // Import the PyTorch module
    this->torch_module = PyImport_ImportModule("torch");
    
    // If torch object is not null
    if (this->torch_module != nullptr) {
        // Get PyTorch load
        PyObject* torch_load_func = PyObject_GetAttrString(this->torch_module, "load");
        
        //
        if (torch_load_func != nullptr && PyCallable_Check(torch_load_func)) {
            // Load PINN model
            this->pinn_obj = PyObject_CallFunction(torch_load_func, "s",
                                                   this->PINNPath.c_str());;
            
            // Get the gradient and potential methods of the model
            this->grad_func = PyObject_GetAttrString(this->pinn_obj, "gradient");
            this->U_func = PyObject_GetAttrString(this->pinn_obj, "forward");
        }
    }
    
    return {};
}

std::optional<std::string> PINN2GravityModel::initializeParameters(const GravBodyData& body)
{
    return this->initializeParameters();
}

Eigen::Vector3d
PINN2GravityModel::computeField(const Eigen::Vector3d& position_planetFixed) const
{
    // Preallocate acceleration
    Eigen::Vector3d acc;
    
    // Declare float for position
    float x = position_planetFixed(0);
    float y = position_planetFixed(1);
    float z = position_planetFixed(2);
                    
    // Call tensor attribute and declare input tensor
    PyObject* torch_tensor_func = PyObject_GetAttrString(this->torch_module, "tensor");
    PyObject* input_tensor = PyObject_CallFunction(torch_tensor_func,
                                                   "[(fff)]",
                                                   x, y, z);

    // Call gradient method
    PyObject* output_tensor = PyObject_CallFunctionObjArgs(this->grad_func,
                                                           input_tensor,
                                                           nullptr);
                    
    // Convert the output tensor to a Python list using .tolist()
    PyObject* output_list = PyObject_CallMethod(output_tensor,
                                                "tolist",
                                                nullptr);
    
    // Loop through list items
    for (int i = 0; i < 3; ++i) {
        // Get list element
        PyObject* elem = PyList_GetItem(output_list, i);
        
        // Save list element in Eigen acceleration
        acc(i) = PyFloat_AsDouble(elem);
    }
    /*std::cout << acc << '\n';*/
    
    
    return acc;
}

double
PINN2GravityModel::computePotentialEnergy(const Eigen::Vector3d& positionWrtPlanet_N) const
{
    return 0;
}
