#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "wrapper_templated.hpp"
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

	std::cout << "bye\n";
	std::vector<std::vector<double>> X = {{0.0, -0.889, 1.414}, {-1.225, 1.397, -0.707}, {1.225, -0.508, -0.707}};

	std::vector<std::vector<double>> y = {{2.5}, {-6.2}, {13.1}};
	// std::vector<std::vector<double>> X = {{2}};
	// std::vector<std::vector<double>> y = {{3}};
	matx::Matrix<double> a(X);
	matx::Matrix<double> b(y);

	//(a * (double)5).print_Matrix();
	nn::MLP_Layer* layer1 = new nn::MLP_Layer(3, "tanh");
	nn::MLP_Layer* layer2 = new nn::MLP_Layer(4, "sigmoid");
	nn::MLP_Layer* layer3 = new nn::MLP_Layer(1, "none");

	nn::optimizer::GD_Momentum* optim = new nn::optimizer::GD_Momentum(0.2, 0.5);
	// std::cout << "np\n";
	nn::wrapper<nn::optimizer::GD_Momentum> model({layer1, layer2, layer3}, optim);

	model.train(a, b, 30, 100);
	matx::Matrix<double> y_pred = model.predict(a);
	y_pred.print_Matrix();
}

/*
to change ---
1) add He/glorot initialization. Model is highly sensitive to weight initialization. -- DONE
2) add a element wise function which takes a func ptr as input in matx::Matrix -- DONE
3) replace exit 1's with std::throw everywhere , use size_t wherever warning -- DONE
4) separate nn.cpp into diff files , one for each class -- DONE


5) add optimizers like adam , sgd
6) add features like dropout!! -> complicated because layer type can cause issues :(
7) think about the Matrix constructor how i want to manage it

*/