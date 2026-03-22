#include "Adam.hpp"
#include "Dense.hpp"
#include "Dropout.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "gd_momentum.hpp"
#include "loss.hpp"
#include "wrapper.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

int main() {

	std::cout << "new code\n";
	std::vector<std::vector<double>> X = {{0.0, -0.889, 1.414}, {-1.225, 1.397, -0.707}, {1.225, -0.508, -0.707}};

	std::vector<std::vector<double>> y = {{2.5}, {-6.2}, {13.1}};
	// std::vector<std::vector<double>> X = {{2}};
	// std::vector<std::vector<double>> y = {{3}};
	matx::Matrix<double> a(X);
	matx::Matrix<double> b(y);

	//(a * (double)5).print_Matrix();
	nn::Layer* layer1 = new nn::Dense(3, "tanh");
	nn::Layer* layer2 = new nn::Dense(4, "sigmoid");
	nn::Layer* drop = new nn::Dropout(0.01);
	nn::Layer* layer3 = new nn::Dense(1, "none");

	nn::Adam optim(0.15);
	// std::cout << "np\n";
	nn::wrapper model({layer1, layer2, drop, layer3}, &optim);

	model.train(a, b, 100);
	// std::cout << "what\n;";
	matx::Matrix<double> y_pred = model.predict(a);
	y_pred.print_Matrix();
}

// Note --to make sure the wrapper object is deleted it is the user's responsibility to provide its scope
// particularly when the same optimizer is used for different models.

/*
to change ---
1) add He/glorot initialization. Model is highly sensitive to weight initialization. -- DONE
2) add a element wise function which takes a func ptr as input in matx::Matrix -- DONE
3) replace exit 1's with std::throw everywhere , use size_t wherever warning -- DONE
4) separate nn.cpp into diff files , one for each class -- DONE


5) add optimizers like adam , sgd -- DONE
6) add option to print loss at each epoch -- DONE
7) add features like dropout!! -> complicated because layer type can cause issues :( ----------------- DONEEE

** Changed layer interactions -- updated both forward and backprop of Dense.. see interfaces defined in Layer.hpp.
Use those templates to build more layers in the future.

dont forget to define is_trainable for each layer!!

8) think about the Matrix constructor how i want to manage it

*/