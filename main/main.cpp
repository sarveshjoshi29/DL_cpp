#include "Adam.hpp"
#include "Attention_Pool.hpp"
#include "Dropout.hpp"
#include "GNN.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "gd_momentum.hpp"
#include "loss.hpp"
#include "wrapper_GNN.hpp"
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

	// include self loops in adjacency matrix
	std::vector<std::vector<int>> adj = {{1, 1, 0}, {1, 1, 1}, {0, 1, 1}};
	matx::Matrix<int> adj_matrix(adj);
	std::vector<std::vector<double>> feature_vec = {{1.5, 3.0, 2.1}, {2.3, 4.1, 3.9}, {4.2, 2.3, 3.1}};
	matx::Matrix<double> f(feature_vec);

	nn::Graph g(adj_matrix, f);
	std::vector<nn::Graph*> x;
	x.push_back(&g);
	nn::Layer* layer1 = new nn::GNN("sigmoid", 4);
	nn::Layer* layer2 = new nn::GNN("sigmoid", 2);
	nn::Layer* layer3 = new nn::AttentionPool();
	nn::Layer* layer4 = new nn::MLP_Layer(4, "sigmoid");
	nn::Layer* layer5 = new nn::MLP_Layer(1, "sigmoid");
	nn::optimizer* optim = new nn::Adam(0.15);
	matx::Matrix<double> y_temp(1, 1, 1);
	std::vector<matx::Matrix<double>> y_train;
	y_train.push_back(y_temp);
	nn::wrapper_GNN model({layer1, layer2, layer3, layer4, layer5}, optim, "cross_entropy");
	model.train(x, y_train, 100, 1);
	std::vector<matx::Matrix<double>> preds = model.predict(x);
	preds[0].print_Matrix();
	// std::cout << "nice\n";
	//  std::cout << "new code\n";
	//  std::vector<std::vector<double>> X = {{0.0, -0.889, 1.414}, {-1.225, 1.397, -0.707}, {1.225, -0.508, -0.707}};

	// std::vector<std::vector<double>> y = {{2.5}, {-6.2}, {13.1}};
	// // std::vector<std::vector<double>> X = {{2}};
	// // std::vector<std::vector<double>> y = {{3}};
	// matx::Matrix<double> a(X);
	// matx::Matrix<double> b(y);

	// //(a * (double)5).print_Matrix();
	// nn::Layer* layer1 = new nn::MLP_Layer(3, "tanh");
	// nn::Layer* layer2 = new nn::MLP_Layer(4, "sigmoid");
	// nn::Layer* drop = new nn::Dropout(0.01);
	// nn::Layer* layer3 = new nn::MLP_Layer(1, "none");

	// nn::optimizer* optim = new nn::Adam(0.15);
	// // std::cout << "np\n";
	// nn::wrapper model({layer1, layer2, drop, layer3}, optim);

	// model.train(a, b, 100);
	// // std::cout << "what\n;";
	// matx::Matrix<double> y_pred = model.predict(a);
	// y_pred.print_Matrix();
}

/*
to change ---
1) add He/glorot initialization. Model is highly sensitive to weight initialization. -- DONE
2) add a element wise function which takes a func ptr as input in matx::Matrix -- DONE
3) replace exit 1's with std::throw everywhere , use size_t wherever warning -- DONE
4) separate nn.cpp into diff files , one for each class -- DONE


5) add optimizers like adam , sgd -- DONE
6) add option to print loss at each epoch -- DONE
7) add features like dropout!! -> complicated because layer type can cause issues :( ----------------- DONEEE

** Changed layer interactions -- updated both forward and backprop of MLP_Layer.. see interfaces defined in Layer.hpp.
Use those templates to build more layers in the future.

dont forget to define is_trainable for each layer!!

8) think about the Matrix constructor how i want to manage it

*/