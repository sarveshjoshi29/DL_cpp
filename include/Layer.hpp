#pragma once

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
class Layer {
	public:
		matx::Matrix<double> w;
		matx::Matrix<double> b;
		matx::Matrix<double> dw;
		matx::Matrix<double> db;
		bool is_trainable;

		virtual int initialize(int input_dims) = 0;
		virtual matx::Matrix<double> forward_pass(matx::Matrix<double> x) = 0;
		virtual matx::Matrix<double> backward_pass(matx::Matrix<double> dJ_da) = 0;
		virtual ~Layer() = default;
};

} // namespace nn