#include "wrapper_GNN.hpp"
#include "GNN.hpp"
#include "GNN_BatchHandler.hpp"
#include "MLP_Layer.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
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

wrapper_GNN::~wrapper_GNN() {
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

wrapper_GNN::wrapper_GNN(std::initializer_list<nn::Layer*> inp_layers, nn::optimizer* optimizer, string loss) {
	gen.seed(std::random_device{}());
	if(loss == "mse") {
		this->loss = nn::loss::mse;
		this->lossprime = nn::loss::mseprime;
	} else if(loss == "cross_entropy") {
		this->loss = nn::loss::cross_entropy;
		this->lossprime = nn::loss::cross_entropy_prime;
	}
	for(auto l : inp_layers) {
		this->layers.push_back(l);
	}
	update_manager = optimizer;
}

void wrapper_GNN::init_batch_handler(std::vector<nn::Graph*> data, std::vector<matx::Matrix<double>>& y,
									 matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), nn::optimizer* update_manager,
									 double (*loss)(matx::Matrix<double>, matx::Matrix<double>)) {
	//	std::cout << "reached bh_init\n";

	this->batch_handler = new nn::GNN_BatchHandler(data, y, lossprime, update_manager, loss);
	// std::cout << "exited\n";
}

void wrapper_GNN::train(std::vector<nn::Graph*> X_train, std::vector<matx::Matrix<double>> y_train, int epochs, int verbose) {
	data = X_train;
	y = y_train;
	// std::cout << "done\n";
	init_batch_handler(X_train, y_train, lossprime, update_manager, loss);
	std::vector<int> inp_dims(2, 0);
	for(auto ptr : X_train) {
		ptr->compute_A_hat();
	}
	GNN* pt = dynamic_cast<GNN*>(layers[0]);

	inp_dims[0] = data[0]->feature_vectors[0].size();
	inp_dims[1] = pt->length;
	std::vector<int> curr_dims = inp_dims;

	for(size_t i = 0; i < this->layers.size(); i++) {
		// std::cout << i << "okk";
		GNN* pt = dynamic_cast<GNN*>(layers[i]);
		if(pt) {
			curr_dims[1] = pt->length;
			curr_dims = layers[i]->initialize(curr_dims);
		} else {
			curr_dims = layers[i]->initialize(curr_dims);
		}
	}
	// printwts();
	// std::cout << "all good";
	for(int i = 1; i <= epochs; i++) {
		if(verbose) {
			std::cout << "Epoch " << i << " -- ";
		}
		if(i == 1) {
			// std::cout << "reachee\n";
			batch_handler->init_params_updmanager(layers);
			// std::cout << "no\n";
		}
		// std::cout << "nah\n";
		batch_handler->one_epoch(layers, verbose);
		//	std::cout << "f9\n";
	}
}

std::vector<matx::Matrix<double>> wrapper_GNN::predict(std::vector<Graph*> X_test) {
	std::vector<matx::Matrix<double>> y;
	// std::cout << "okk?\n";
	for(int j = 0; j < X_test.size(); j++) {
		matx::Matrix<double> y_pred = X_test[j]->feature_vectors;
		// std::cout << "in\n";
		for(size_t i = 0; i < layers.size(); i++) {
			GNN* pt = dynamic_cast<GNN*>(layers[i]);
			if(pt) {
				pt->A_hat = X_test[j]->A_hat;
			}
			y_pred = (*layers[i]).forward_pass(y_pred);
			
		}
		y.push_back(y_pred);
		//std::cout << y_pred[0][0] << " done\n";
	}
	return y;
}

} // namespace nn