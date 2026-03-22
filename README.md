# DL_cpp

A custom C++ Deep Learning library implementing fundamental neural network components from scratch. This project is inspired from PyTorch and provides a flexible object-oriented framework for building, training, and testing neural networks using a custom matrix library and modular layer design.


## Features

-   **Layer Types**:
    -   `Dense`: Fully Connected (Dense) layers with configurable activation functions.
    -   `Dropout`: Regularization layer to prevent overfitting.
-   **Optimizers**:
    -   `Adam`: Adaptive Moment Estimation.
    -   `gd_momentum`: Gradient Descent with Momentum.
    -   `rmsprop`: Root Mean Square Propagation.
-   **Activation Functions**:
    -   Sigmoid
    -   Tanh
    -   Linear (None)
-   **Core Components**:
    -   Custom `Matrix` library for efficient linear algebra operations.
    -   `wrapper` class to manage model architecture, training loops, and predictions.
    -   `Batch_Handler` for managing data batches.

## Requirements

-   C++17 or higher
-   CMake 3.10 or higher
-   Windows OS

## Build Instructions

### Run Project(Windows PowerShell)

A PowerShell script is provided to automate the build and run process. It handles directory creation, CMake configuration, compilation, and execution.

```powershell
./run.ps1
```


## Architecture & OOP Design

This project demonstrates strong Object-Oriented Programming (OOP) principles and C++ design patterns:

-   **Polymorphism & Interfaces**: The core architecture relies on abstract base classes. `nn::Layer` and `nn::optimizer` serve as interfaces, allowing for seamless extension of new layer types and optimization algorithms without modifying the core training logic.
-   **Strategy Pattern**: The `nn::wrapper` class implements the Strategy Pattern for optimization. Different optimization strategies (Adam, RMSProp, SGD) can be injected into the model at runtime. THe `nn::Batch_Handler` class handles batching of data and allows flexibility to the user to adapt batch size for Mini-Batch Gradient Descent. 
-   **Encapsulation & Abstraction**: Complex linear algebra operations are encapsulated within the custom `batx::Matrix` class, providing a clean high-level API for neural network mathematical operations.
-   **Modern C++ Features**: severeal usages of modern C++ (C++17) features including templates, smart memory management consideration etc.

## Usage Example

Here is a simple example of how to create, train, and use a model (based on `main/main.cpp`):

```cpp
#include "Dense.hpp"
#include "Dropout.hpp"
#include "Adam.hpp"
#include "wrapper.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    // 1. Prepare data as a templated matx::Matrix object
    std::vector<std::vector<double>> X_data = {{0.0, -0.889, 1.414}, {-1.225, 1.397, -0.707}};
    std::vector<std::vector<double>> y_data = {{2.5}, {-6.2}};
    
    matx::Matrix<double> X(X_data);
    matx::Matrix<double> y(y_data);

    // 2. Define Architecture
    // Eg : Input -> Dense(3, tanh) -> Dense(4, sigmoid) -> Dropout(0.01) -> Dense(1, linear)
    nn::Layer* layer1 = new nn::Dense(3, "tanh");
    nn::Layer* layer2 = new nn::Dense(4, "sigmoid");
    nn::Layer* drop = new nn::Dropout(0.01);
    nn::Layer* layer3 = new nn::Dense(1, "none");

    // 3. Select Optimizer with appropriate hyperparams 
    nn::optimizer* optim = new nn::Adam(0.15); // Learning rate 0.15

    // 4. Create Model Wrapper
    // Pass layers as an initializer list and the optimizer
    nn::wrapper model({layer1, layer2, drop, layer3}, optim);

    // 5. Train
    model.train(X, y, 100); // Trains for 100 epochs

    // 6. Predict
    matx::Matrix<double> y_pred = model.predict(X);
    std::cout << "Predictions:\n";
    y_pred.print_Matrix();

    return 0;
}
```

## Project Structure

```
DL_cpp/
├── CMakeLists.txt              # CMake build configuration
├── run.ps1                     # PowerShell build automation script
├── README.md                   # Project documentation
├── include/                    # Header files (.hpp)
│   ├── Adam.hpp                # Adam optimizer definition
│   ├── Batch_Handler.hpp       # Batch processing utilities
│   ├── Dropout.hpp             # Dropout layer definition
│   ├── Layer.hpp               # Abstract base Layer interface
│   ├── Dense.hpp           # Dense core layer definition
│   ├── Matrix.hpp              # Templated custom Matrix library
│   ├── activations.hpp         # Activation functions (sigmoid, tanh, etc.)
│   ├── gd_momentum.hpp         # Gradient Descent with Momentum optimizer
│   ├── loss.hpp                # Loss functions (MSE, CrossEntropy, etc.)
│   ├── optimizer.hpp           # Abstract base Optimizer interface
│   ├── rmsprop.hpp             # RMSProp optimizer definition
│   └── wrapper.hpp             # High-level model wrapper (Strategy Context)
├── src/                        # Source files (.cpp)
│   ├── Adam.cpp                # Adam implementation
│   ├── Batch_Handler.cpp       # Batch logic implementation
│   ├── Dropout.cpp             # Dropout implementation
│   ├── Dense.cpp           # Dense layer implementation
│   ├── gd_momentum.cpp         # GD Momentum implementation
│   ├── rmsprop.cpp             # RMSProp implementation
│   └── wrapper.cpp             # Model wrapper implementation
├── main/                       # Application entry point
│   └── main.cpp                # Testing and demonstration code
├── build/                      # Build artifacts (created by CMake)
└── exe/                        # Compiled executables
```

## Implementation Details

-   **Weight Initialization**: The model uses He/Glorot initialization for better convergence.
-   **Error Handling**: Uses C++ exceptions for error management.
-   **Extensibility**: New layers(like LSTM/CNN) can be added by inheriting from the `nn::Layer` interface defined in `include/Layer.hpp`.


## Extensions & Experimental Modules

-   **Graph Neural Networks (GNN)**: A specialized implementation residing in the `GNN` branch. It introduces a custom Graph Data structure and specialized batch handling logic to support graph-based deep learning tasks.. 


## Instructions for user 

- 1. Currently you can add a dataset consisting of only floating point numbers by making a directory named "dataset" in the project root. Make sure the last column is the target column and use a csv_parser object to get corresponding X and y. An example is shown in main.cpp

- 2. Currently data preprocessing such as scaling and NaN value handling must be entirely done by the user before putting it in the dataset directory. Note that presence of NaN values will throw errors.


## Benchmarking on California Housing Dataset
![alt text](image.png)