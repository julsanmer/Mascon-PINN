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

#include "pinnGravityModel.h"
#include <iostream>
#include "simulation/dynamics/_GeneralModuleFiles/gravityEffector.h"
#include <Python.h>

namespace {
// Transform Eigen::Vector into Ort Tensor
template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape)
{
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    
  return tensor;
}
}

std::optional<std::string> PINNGravityModel::initializeParameters()
{
    // Initialize the Python interpreter
    Py_Initialize();

    // Import the torch module
    PyObject* torch_module = PyImport_ImportModule("torch");
    if (torch_module != nullptr) {
        PyObject* torch_load_func = PyObject_GetAttrString(torch_module, "load");
        if (torch_load_func != nullptr && PyCallable_Check(torch_load_func)) {
            PyObject* model_obj = PyObject_CallFunction(torch_load_func, "s", "/Users/julio/Desktop/python_scripts/THOR/scripts/Results/eros/results/poly/ideal/dense_alt50km_100000samples/pinn_models/pinn6x40SIREN_mascon100.pt"); // Replace "model.pt" with the path to your model file
            if (model_obj != nullptr) {
                std::cout << 4;
                // Get the forward method of the model
                PyObject* forward_func = PyObject_GetAttrString(model_obj, "gradient");
                if (forward_func != nullptr && PyCallable_Check(forward_func)) {
                    std::cout << 5;
                    float a = 1000;
                    float b = 2000;
                    float c = 24000;
                    
                    // Prepare input tensor (example: a tensor of size [1, 10] filled with zeros)
                    PyObject* torch_tensor_func = PyObject_GetAttrString(torch_module, "tensor");
                    PyObject* input_tensor = PyObject_CallFunction(torch_tensor_func, "[(fff)]", a, b, c); // Example: create a tensor of size [1, 10] filled with zeros
                    std::cout << 6;

                    // Perform a forward pass
                    PyObject* output_tensor = PyObject_CallFunctionObjArgs(forward_func, input_tensor, nullptr);
                    std::cout << 7;
                    
                    // Print the output tensor
                    PyObject* repr = PyObject_Repr(output_tensor);
                    const char* output_str = PyUnicode_AsUTF8(repr);
                    printf("Output tensor: %s\n", output_str);
                    
                    // Convert the output tensor to a Python list using .tolist()
                    PyObject* output_list = PyObject_CallMethod(output_tensor, "tolist", nullptr);
                    if (output_list == nullptr) {
                        // Handle error: failed to convert to list
                        std::cerr << "Failed to convert output tensor to list" << std::endl;
                    }

                    Eigen::Vector3d vec;
                    double value;
                    for (int i = 0; i < 3; ++i) {
                        PyObject* elem = PyList_GetItem(output_list, i);
                        std::cout << 999;
                        if (!PyFloat_Check(elem)) {
                            // Handle error: element is not a float
                            throw std::runtime_error("Invalid element in torch tensor");
                        }
                        value = PyFloat_AsDouble(elem);
                        vec(i) = value;
                    }
                    std::cout << vec;
                    

                    // Convert the output tensor to a NumPy array
                    /*PyObject* numpy_array = PyObject_GetAttrString(output_tensor, "numpy");
                    PyObject* numpy_array_data = PyObject_GetAttrString(numpy_array, "data");*/

                    // Print the output tensor

                    // Cleanup
                }
                Py_DECREF(forward_func);
                Py_DECREF(model_obj);
            }
            Py_DECREF(torch_load_func);
        }
        Py_DECREF(torch_module);
    }

    std::cout << 4;
    // Finalize the Python interpreter
    //Py_Finalize();
    
    
    // Initialize ONNXRuntime environment
    std::string instanceName{"PINN gravity model"};
    this->env = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                           instanceName.c_str());
    
    // Create a session by loading ONNX model
    Ort::SessionOptions sessionOptions;
    this->session = std::make_shared<Ort::Session>(*this->env, this->PINNPath.c_str(),
                                                   sessionOptions);
    
    // Allocate memory by default
    Ort::AllocatorWithDefaultOptions allocator;

    // Declare PINN inputs size and shape
    for (std::size_t i = 0; i < this->session->GetInputCount(); i++) {
      this->inputName.emplace_back(this->session->GetInputNameAllocated(i, allocator).get());
      this->inputShape = this->session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }
    
    // Declare PINN outputs size and shape
    for (std::size_t i = 0; i < this->session->GetOutputCount(); i++) {
      this->outputName.emplace_back(this->session->GetOutputNameAllocated(i, allocator).get());
      this->outputShape = this->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }
    
    // Finite difference vectors
    this->dx << this->eps, 0, 0;
    this->dy << 0, this->eps, 0;
    this->dz << 0, 0, this->eps;
    
    return {};
}

std::optional<std::string> PINNGravityModel::initializeParameters(const GravBodyData& body)
{
    return this->initializeParameters();
}

Eigen::Vector3d
PINNGravityModel::computeField(const Eigen::Vector3d& position_planetFixed) const
{
    // Declare acceleration
    Eigen::Vector3d acc;
    
    // Compute dUx
    double Ux1, Ux0;
    Ux0 = computePotentialEnergy(position_planetFixed - this->dx);
    Ux1 = computePotentialEnergy(position_planetFixed + this->dx);
    acc(0) = (Ux1 - Ux0) / 2*this->eps;
    
    // Compute dUy
    double Uy1, Uy0;
    Uy0 = computePotentialEnergy(position_planetFixed - this->dy);
    Uy1 = computePotentialEnergy(position_planetFixed + this->dy);
    acc(1) = (Uy1 - Uy0) / 2*this->eps;
    
    // Compute dUz
    double Uz1, Uz0;
    Uz0 = computePotentialEnergy(position_planetFixed - this->dz);
    Uz1 = computePotentialEnergy(position_planetFixed + this->dz);
    acc(2) = (Uz1 - Uz0) / 2*this->eps;
    
    return acc;
}

double
PINNGravityModel::computePotentialEnergy(const Eigen::Vector3d& positionWrtPlanet_N) const
{
    // Declare input to PINN
    Eigen::Vector3d input_NN;
    input_NN = positionWrtPlanet_N;
    
    // Assert input and output names sizes
    assert(this->inputName.size() == 1 && this->outputName.size() == 1);

    // Retrieve PINN input shape
    auto input_shape = this->inputShape;
    
    // Save PINN input as Ort tensor
    std::vector<float> stdVector(input_NN.data(),  input_NN.data() + input_NN.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor<float>(stdVector, input_shape));

    // Check PINN input is a tensor and its dimension
    assert(input_tensors[0].IsTensor() && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    
    // It is required to set input/output names in Ort
    std::vector<const char*> input_names_char(this->inputName.size(), nullptr);
    std::transform(std::begin(this->inputName), std::end(this->inputName), std::begin(input_names_char), [&](const std::string& str) { return str.c_str(); });
    std::vector<const char*> output_names_char(this->outputName.size(), nullptr);
    std::transform(std::begin(this->outputName), std::end(this->outputName), std::begin(output_names_char), [&](const std::string& str) { return str.c_str(); });
    
    // Run PINN
    auto output_tensors = this->session->Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_names_char.size(), output_names_char.data(), output_names_char.size());
    
    // Retrieve PINN output info
    Ort::TensorTypeAndShapeInfo output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    // Save PINNN output into double
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    double output_NN;
    output_NN = output_data[0];
    
    return output_NN;
}
