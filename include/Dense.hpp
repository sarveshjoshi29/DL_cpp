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

namespace nn { // namespace nn

class Dense : public Layer {
	public:
		matx::Matrix<double> z;
		matx::Matrix<double> a;
		int num_neurons;
		// matx::Matrix<double> w;
		// matx::Matrix<double> b;
		matx::Matrix<double> (*activation)(matx::Matrix<double>);
		matx::Matrix<double> (*activationprime)(matx::Matrix<double>);
		// matx::Matrix<double> dw;
		// matx::Matrix<double> db;
		matx::Matrix<double> a_prev;
		Dense(int nl, string act);
		void set_weights(int dims_prev_layer, std::mt19937& gen);
		void set_bias();

		int initialize(int dims) override;
		matx::Matrix<double> forward_pass(matx::Matrix<double> x) override;
		matx::Matrix<double> backward_pass(matx::Matrix<double> dJ_da) override;
};
} // namespace nn