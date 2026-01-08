#pragma once
#include "GNN_BatchHandler.hpp"
#include "Graph.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
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

namespace nn {
class wrapper_GNN {
	private:
		nn::GNN_BatchHandler* batch_handler;

	public:
		std::vector<nn::Layer*> layers;
		std::mt19937 gen;
		std::vector<nn::Graph*> data;
		std::vector<matx::Matrix<double>> y;
		double (*loss)(matx::Matrix<double>, matx::Matrix<double>);
		matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>);
		nn::optimizer* update_manager;

		~wrapper_GNN();

		wrapper_GNN(std::initializer_list<nn::Layer*> inp_layers, nn::optimizer* optimizer, string loss = "mse");

		void train(std::vector<nn::Graph*> X_train, std::vector<matx::Matrix<double>> y_train, int epochs = 100, int verbose = 1);

		void printwts();

		std::vector<matx::Matrix<double>> predict(std::vector<Graph*> X_test);

	private:
		void init_batch_handler(std::vector<nn::Graph*> data, std::vector<matx::Matrix<double>>& y,
								matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), optimizer* update_manager,
								double (*loss)(matx::Matrix<double>, matx::Matrix<double>));
};
} // namespace nn
