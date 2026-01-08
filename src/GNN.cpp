#include "GNN.hpp"
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

// figure out a way to get D-1 and A.
GNN::GNN(string activation_str, int length) : length{length} {
	if(activation_str == "sigmoid") {
		activation = nn::activations::sigmoid;
		activation_prime = nn::activations::sigmoidprime;
	}
	l = 3;

	is_trainable = true;
}

matx::Matrix<double> GNN::forward_pass(matx::Matrix<double> x) {
	// d-1 a h w + h b
	// std::cout << "entered\n";
	// w.print_Matrix();
	//  std::vector<int> shapew = w.shape();
	//  std::vector<int> shapeA = A_hat.shape();
	//  std::vector<int> shapex = x.shape();
	//  std::vector<int> shapeb = b.shape();

	// std::cout << "shape of w is " << shapew[0] << " " << shapew[1] << "\n";
	// std::cout << "shape of A is " << shapeA[0] << " " << shapeA[1] << "\n";
	// std::cout << "shape of x is " << shapex[0] << " " << shapex[1] << "\n";
	// std::cout << "shape of b is " << shapeb[0] << " " << shapeb[1] << "\n";
	input = x;
	H = (A_hat % x % w.transpose()) + (x % b.transpose());
	// H.print_Matrix();
	return H.apply_elem_wise(activation);
}

matx::Matrix<double> GNN::backward_pass(matx::Matrix<double> dJ_dy) {
	matx::Matrix<double> deriv_act = dJ_dy * H.apply_elem_wise(activation_prime);
	dw = deriv_act.transpose() % A_hat % input;
	db = deriv_act.transpose() % input;
	return A_hat.transpose() % deriv_act % w + deriv_act % b;
}

std::vector<int> GNN::initialize(std::vector<int> dims) {
	// dims = (Fin , Fout) which is dims of x
	w = matx::Matrix<double>(dims[1], dims[0], 1);
	b = matx::Matrix<double>(dims[1], dims[0], 1);
	dw = matx::Matrix<double>(dims[1], dims[0], 0.1);
	db = matx::Matrix<double>(dims[1], dims[0], 0.1);
	std::vector<int> to_ret = {dims[1], 0};
	return to_ret;
}
} // namespace nn
