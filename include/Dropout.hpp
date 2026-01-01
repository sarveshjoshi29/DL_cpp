#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
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
class Dropout : public Layer {
	public:
		double dropout_rate;
		matx::Matrix<double> drop_matrix;
		std::random_device rd;
		std::mt19937 gen;
		std::bernoulli_distribution dist;

		Dropout(double frac);
		int initialize(int input_dims) override;
		matx::Matrix<double> forward_pass(matx::Matrix<double> x) override;
		matx::Matrix<double> backward_pass(matx::Matrix<double> dJ_da) override;
};
} // namespace nn