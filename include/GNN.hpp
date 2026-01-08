#pragma once
#include "Layer.hpp"
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

// prob1 -- we need to store vector of grads for each w,b ... i.e dJ/dW[i] so would need to change update manager for this!
// one way to solve above problem is to create that many layers in main itself so that GNN Layers are stacked!

namespace nn {
class GNN : public Layer {
	public:
		int l;
		int length;
		matx::Matrix<double> input;
		matx::Matrix<double> D_inv;
		matx::Matrix<double> adj_matrix;
		matx::Matrix<double> A_hat;
		matx::Matrix<double> H;
		double (*activation)(double);
		double (*activation_prime)(double);
		std::vector<int> initialize(std::vector<int> input_dims) override;
		matx::Matrix<double> forward_pass(matx::Matrix<double> x) override;
		matx::Matrix<double> backward_pass(matx::Matrix<double> dJ_da) override;
		GNN(string activation_str, int length);

		
};
} // namespace nn
