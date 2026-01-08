#include "Dropout.hpp"
#include "Layer.hpp"
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
Dropout::Dropout(double frac) : gen(std::random_device{}()), dist(1 - frac), dropout_rate{frac} {
	// std::cout << "constructed\n";
	is_trainable = false;
}

std::vector<int> Dropout::initialize(std::vector<int> input_dims) {
	return {input_dims};
}

matx::Matrix<double> Dropout::forward_pass(matx::Matrix<double> x) {
	// std::cout << "forw start\n";
	matx::Matrix<double> d(x.size(), x[0].size(), 1);
	for(int i = 0; i < d.size(); i++) {
		for(int j = 0; j < d[0].size(); j++) {
			bool keep = dist(gen);
			d[i][j] = keep ? 1 : 0;
		}
	}
	drop_matrix = d;
	// std::cout << "forw end\n";
	return (x * drop_matrix) / (1 - dropout_rate);
}

matx::Matrix<double> Dropout::backward_pass(matx::Matrix<double> dJ_da) {
	return (dJ_da * drop_matrix) / (1 - dropout_rate);
}
} // namespace nn