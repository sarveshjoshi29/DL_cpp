#include "GNN_BatchHandler.hpp"
#include "Attention_Pool.hpp"
#include "GNN.hpp"
#include "Graph.hpp"
#include "Matrix.hpp"
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

// todo -- manually write forw, back passes and also handle constructor! ... change calls in one_epoch accordingly

namespace nn {

GNN_BatchHandler::GNN_BatchHandler(std::vector<nn::Graph*> disjoint_graphs, std::vector<matx::Matrix<double>> batches_Y,
								   matx::Matrix<double> (*lossprime)(matx::Matrix<double>, matx::Matrix<double>), nn::optimizer* update_manager,
								   double (*loss)(matx::Matrix<double>, matx::Matrix<double>))
	: graphs{disjoint_graphs}, batches_Y{batches_Y}, lossprime{lossprime}, update_manager{update_manager}, loss{loss} {
	{ curr_loss = 0; }
}

void GNN_BatchHandler::forward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers) {
	matx::Matrix<double> A_hat = graphs[itr]->A_hat;
	// A_hat.print_Matrix();
	matx::Matrix<double> curr_passed_object = graphs[itr]->feature_vectors;
	// curr_passed_object.print_Matrix();

	for(int i = 0; i < layers.size(); i++) {
		GNN* pt = dynamic_cast<GNN*>(layers[i]);
		AttentionPool* pt2 = dynamic_cast<AttentionPool*>(layers[i]);
		// if(pt2) {
		// 	std::cout << "here\n";
		// }
		// std::cout << "k till " << i << "\n";
		if(pt) {
			pt->A_hat = A_hat;
		}

		curr_passed_object = layers[i]->forward_pass(curr_passed_object);
		// curr_passed_object.print_Matrix();
	}
	out_last = curr_passed_object;
	// std::cout << "forw done";
}

void GNN_BatchHandler::backward_prop_itr(size_t itr, std::vector<nn::Layer*>& layers) {
	matx::Matrix<double> dJ_da = lossprime(batches_Y[itr].transpose(), out_last);
	for(int i = layers.size() - 1; i >= 0; i--) {
		dJ_da = layers[i]->backward_pass(dJ_da);
	}
}

// independent of itr , keep memory of required variables in the other provider class itself!
void GNN_BatchHandler::update_params(std::vector<nn::Layer*>& layers) {
	// std::cout << "update in\n";
	update_manager->update_params(layers);
}

void GNN_BatchHandler::init_params_updmanager(std::vector<nn::Layer*>& layers) {

	update_manager->init_params(layers);
	//	std::cout << "exited ";
}

double GNN_BatchHandler::compute_loss(matx::Matrix<double>& y) {
	return loss(y, out_last);
}

void GNN_BatchHandler::one_epoch(std::vector<nn::Layer*>& layers, int verbose) {
	// std::cout << num_matx << "\n";
	curr_loss = 0;
	for(size_t itr = 0; itr < graphs.size(); itr++) {

		// std::cout << " itr is " << itr << "\n";
		forward_prop_itr(itr, layers);
		// std::cout << "forw prop ok\n";
		//  out_last.print_Matrix();
		matx::Matrix<double> y_temp = batches_Y[itr].transpose();
		curr_loss += compute_loss(y_temp);
		backward_prop_itr(itr, layers);
		// std::cout << "backprop ok\n";
		update_params(layers);
	}
	curr_loss /= graphs.size();
	if(verbose) {
		std::cout << "Loss = " << curr_loss << "\n";
	}
}

} // namespace nn