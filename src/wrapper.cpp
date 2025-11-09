#include "wrapper.hpp"
#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
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

namespace nn {

wrapper::~wrapper() {
	if(batch_handler != nullptr) {
		delete batch_handler;
	}
	if(update_manager != nullptr) {
		delete update_manager;
	}
	for(auto k : layers) {
		if(k != nullptr) {
			delete k;
		}
	}
	layers.clear();
}

wrapper::wrapper(std::initializer_list<nn::MLP_Layer*> inp_layers, nn::optimizer* optimizer, string loss) {
	gen.seed(std::random_device{}());
	if(loss == "mse") {
		this->loss = nn::loss::mse;
		this->lossprime = nn::loss::mseprime;
	}
	for(auto l : inp_layers) {
		this->layers.push_back(l);
	}

	for(size_t i = 1; i < this->layers.size(); i++) {
		(*layers[i]).set_weights((*layers[i - 1]).num_neurons, gen);
		(*layers[i]).set_bias();
	}
	update_manager = optimizer;
}

void wrapper::train(const matx::Matrix<double> X_train, const matx::Matrix<double> y_train, int epochs, size_t batch_size) {
	data = X_train;
	y = y_train;
	//	std::cout << "done\n";
	init_batch_handler(batch_size, X_train, y_train, lossprime, update_manager);
	std::vector<int> inp_dims = data.transpose().shape();

	layers[0]->set_weights(inp_dims[0], gen);
	layers[0]->set_bias();
	// printwts();
	// std::cout << "all good";
	for(int i = 1; i <= epochs; i++) {
		if(i == 1) {
			batch_handler->init_params_updmanager(layers);
			//	std::cout << "no\n";
		}
		// std::cout << "nah\n";
		batch_handler->one_epoch(layers);
	}
}

matx::Matrix<double> wrapper::predict(matx::Matrix<double> X_test) {
	matx::Matrix<double> test = X_test.transpose();
	for(size_t i = 0; i < layers.size(); i++) {
		if(i == 0) {
			layers[i]->forward_pass(test);
			continue;
		}
		layers[i]->forward_pass(layers[i - 1]->a);
	}
	return layers[layers.size() - 1]->a;
}

void wrapper::printwts() {
	for(size_t i = 0; i < layers.size(); i++) {
		layers[i]->w.print_Matrix();
		//	std::cout << "\n\n\n";
	}
}

void wrapper::init_batch_handler(size_t batch_size, const matx::Matrix<double>& data, const matx::Matrix<double>& y,
								 matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), optimizer* update_manager) {
	//	std::cout << "reached bh_init\n";
	this->batch_handler = new nn::BatchHandler(batch_size, data, y, lossprime, update_manager);
	// std::cout << "exited\n";
}
} // namespace nn
