#include "wrapper.hpp"
#include "Layer.hpp"
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
	// it is responsibility of user to delete optimizer because same optimizer can be used for different models.
	for(auto k : layers) {
		if(k != nullptr) {
			delete k;
		}
	}
	layers.clear();
}

wrapper::wrapper(std::initializer_list<nn::Layer*> inp_layers, nn::optimizer* optimizer, string loss) {
	gen.seed(std::random_device{}());
	if(loss == "mse") {
		this->loss = nn::loss::mse;
		this->lossprime = nn::loss::mseprime;
	}
	for(auto l : inp_layers) {
		this->layers.push_back(l);
	}
	update_manager = optimizer;
	batch_handler = nullptr;
}

void wrapper::train(const matx::Matrix<double> X_train, const matx::Matrix<double> y_train, int epochs, size_t batch_size, int verbose) {
	data = X_train;
	y = y_train;
	//	std::cout << "done\n";
	// if train is called multiple times we want to delete the previous training object
	if(batch_handler != nullptr) {
		delete batch_handler;
	}
	init_batch_handler(batch_size, X_train, y_train, lossprime, update_manager, loss);
	std::vector<int> inp_dims = data.transpose().shape();
	int curr_dims = inp_dims[0];

	for(size_t i = 0; i < this->layers.size(); i++) {
		curr_dims = layers[i]->initialize(curr_dims);
	}
	// printwts();
	//  std::cout << "all good";
	for(int i = 1; i <= epochs; i++) {
		if(verbose) {
			std::cout << "Epoch " << i << " -- ";
		}
		if(i == 1) {
			//	std::cout << "reachee\n";
			batch_handler->init_params_updmanager(layers);
			//	std::cout << "no\n";
		}
		// std::cout << "nah\n";
		batch_handler->one_epoch(layers, verbose);
	}
}

matx::Matrix<double> wrapper::predict(matx::Matrix<double> X_test) {
	matx::Matrix<double> test = X_test.transpose();
	for(size_t i = 0; i < layers.size(); i++) {
		test = (*layers[i]).forward_pass(test);
	}
	return test;
}

void wrapper::printwts() {
	for(size_t i = 0; i < layers.size(); i++) {
		layers[i]->w.print_Matrix();
		//	std::cout << "\n\n\n";
	}
}

void wrapper::init_batch_handler(size_t batch_size, const matx::Matrix<double>& data, const matx::Matrix<double>& y,
								 matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), optimizer* update_manager,
								 double (*loss)(matx::Matrix<double>, matx::Matrix<double>)) {
	//	std::cout << "reached bh_init\n";
	this->batch_handler = new nn::BatchHandler(batch_size, data, y, lossprime, update_manager, loss);
	// std::cout << "exited\n";
}

} // namespace nn
