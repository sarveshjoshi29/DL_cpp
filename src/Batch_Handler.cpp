#include "Batch_Handler.hpp"
#include "Layer.hpp"
#include "MLP_Layer.hpp"
#include "Matrix.hpp"
#include "loss.hpp"
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

// IDEA IS -- BATCH HANDLER WILL TAKE CARE OF BATCHES... FOR RMSPROP/MOMENTUM/ADAM I NEED TO CREATE CLASSES TO CHANGE MY UPDATE_PARAMS FUNCTION! SO I
// THINK I SHOULD PLAN TO REMOVE ONLY UPDATE_PARAMS FROM BATCH HANDLER BUT KEEP THE REST.
//
//
// have defined class optimizer as an interface which must have two functions - init_params() and update_params();
//
//
// DEFINE UPDATE_PARAMS(LAYERS,LEARNING_RATE) IN DIFF CLASSES AND PASS POINTER THROUGH CONSTRUCTOR OF BATCH HANDLER.

// need to debug batch handler --- done
// and all init_param , update_param providers :) --- ADAM rem.

// Structure -- templated wrapper and batch handler to handle diff optims. You can make a custom optimizer which MUST have two functions -
// 1) init_params(layers) -- this should initialized all the internal params required (can use size of each layer).
// 2) update_params(layers) -- this should update w,b of EACH Layer when dw,db are known and using internal parameters

// any other functions for custom optim will be used within that class only!

namespace nn {

BatchHandler::BatchHandler(size_t batch_size, const matx::Matrix<double>& data, const matx::Matrix<double>& y,
						   matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), nn::optimizer* update_manager,
						   double (*loss)(matx::Matrix<double>, matx::Matrix<double>))
	: batch_size{batch_size}, lossprime{lossprime}, update_manager{update_manager}, loss{loss} {
	split_data(data, y);
	curr_loss = 0;
}

void BatchHandler::split_data(const matx::Matrix<double>& data, const matx::Matrix<double>& y) {

	size_t rows = data.size();
	size_t cols = data[0].size();
	size_t y_cols = y[0].size();
	num_matx = rows / batch_size;
	num_matx = (num_matx == 0 ? 1 : num_matx);
	// std::cout << "cols is " << cols << "\n";

	for(size_t i = 0; i < num_matx; i++) {
		matx::Matrix<double> temp = data(i * batch_size, min((i + 1) * batch_size, rows), 0, cols);
		// std::cout << "batch handler --- shape is --- " << temp.shape()[0] << " " << temp.shape()[1] << " ----\n";
		batches_X.push_back(temp);
		matx::Matrix<double> tempy = y(i * batch_size, min((i + 1) * batch_size, rows), 0, y_cols);
		batches_Y.push_back(tempy);
	}
}

void BatchHandler::forward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers) {
	//	std::cout << "forwin\n";
	matx::Matrix<double> curr_passed_object = batches_X[itr].transpose();
	for(size_t i = 0; i < layers.size(); i++) {
		curr_passed_object = (*layers[i]).forward_pass(curr_passed_object);
	}
	out_last = curr_passed_object;
	//	std::cout << "forwout\n";
}

void BatchHandler::backward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers) {
	//	std::cout << "backwin\n";
	matx::Matrix<double> dJ_da = lossprime(batches_Y[itr].transpose(), out_last);
	for(int i = layers.size() - 1; i >= 0; i--) {
		dJ_da = layers[i]->backward_pass(dJ_da);
	}
	//	std::cout << "backout\n";
}

// independent of itr , keep memory of required variables in the other provider class itself!
void BatchHandler::update_params(std::vector<nn::Layer*>& layers) {
	// std::cout << "update in\n";
	update_manager->update_params(layers);
}

void BatchHandler::init_params_updmanager(std::vector<nn::Layer*>& layers) {

	update_manager->init_params(layers);
	//	std::cout << "exited ";
}

double BatchHandler::compute_loss(matx::Matrix<double>& y) {
	return loss(y, out_last);
}

void BatchHandler::one_epoch(std::vector<nn::Layer*>& layers, int verbose) {
	// std::cout << num_matx << "\n";
	curr_loss = 0;
	for(size_t itr = 0; itr < num_matx; itr++) {

		// std::cout << " itr is " << itr << "\n";
		forward_prop_itr(itr, layers);
		// out_last.print_Matrix();
		matx::Matrix<double> y_temp = batches_Y[itr].transpose();
		curr_loss += compute_loss(y_temp);
		backward_prop_itr(itr, layers);
		update_params(layers);
	}
	curr_loss /= num_matx;
	if(verbose) {
		std::cout << "Loss = " << curr_loss << "\n";
	}
}

} // namespace nn