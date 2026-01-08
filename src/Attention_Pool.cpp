#include "Attention_Pool.hpp"
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

namespace nn {
std::vector<int> AttentionPool::initialize(std::vector<int> input_dims) {
	// do random or 0 init of wts?
	matx::Matrix<double> temp(input_dims[0], 1, 1);
	w = temp;
	b = matx::Matrix<double>(1, 1, 1);
	return input_dims;
}

matx::Matrix<double> AttentionPool::forward_pass(matx::Matrix<double> x) {
	// std::vector<int> shapex = x.shape();
	// std::vector<int> shapew = w.shape();
	// std::vector<int> shapeb = b.shape();
	// std::cout << "shape x is " << shapex[0] << " " << shapex[1] << "\n";
	// std::cout << "shape w is " << shapew[0] << " " << shapew[1] << "\n";
	// std::cout << "shape b is " << shapeb[0] << " " << shapeb[1] << "\n";
	matx::Matrix<double> scores = x % w + b;
	softmax_scores = activations::softmax(scores);
	H = x;
	return x.transpose() % softmax_scores;
}

matx::Matrix<double> AttentionPool::backward_pass(matx::Matrix<double> dJ_dy) {
	// plz check calc once
	matx::Matrix<double> dJ_dalpha = H % dJ_dy;
	matx::Matrix<double> unit(softmax_scores.size(), softmax_scores[0].size(), (softmax_scores.transpose() % dJ_dalpha)[0][0]);
	matx::Matrix<double> dJ_dz = softmax_scores * (dJ_dalpha - unit);
	dw = H.transpose() % dJ_dz;
	double bias_grad{};
	for(int i = 0; i < dJ_dz.size(); i++) {
		bias_grad += dJ_dz[i][0];
	}
	db = matx::Matrix<double>(dJ_dz.size(), dJ_dz[0].size(), bias_grad);
	return (softmax_scores % (dJ_dy.transpose()) + dJ_dz % w.transpose());
}

} // namespace nn