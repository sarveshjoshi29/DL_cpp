#pragma once

#include "Matrix.hpp"
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
class activations {
	public:
		static double sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}

		static double sigmoidprime(double x) {
			double s = sigmoid(x);
			return s - (s * s);
		}

		static matx::Matrix<double> sigmoid(matx::Matrix<double> x) {
			return x.apply_elem_wise(sigmoid);
		}

		static matx::Matrix<double> sigmoidprime(matx::Matrix<double> x) {
			return x.apply_elem_wise(sigmoidprime);
		}

		static double relu(double x) {
			return (x > 0 ? x : 0);
		}

		static double reluprime(double x) {
			return (x > 0 ? 1 : 0);
		}

		static matx::Matrix<double> relu(matx::Matrix<double> x) {
			return x.apply_elem_wise(relu);
		}

		static matx::Matrix<double> reluprime(matx::Matrix<double> x) {
			return x.apply_elem_wise(reluprime);
		}

		static double tanh(double x) {
			return std::tanh(x);
		}

		static double tanhprime(double x) {
			double c = cosh(x);
			return 1 / (c * c);
		}

		static matx::Matrix<double> tanh(matx::Matrix<double> x) {
			return x.apply_elem_wise(tanh);
		}

		static matx::Matrix<double> tanhprime(matx::Matrix<double> x) {
			return x.apply_elem_wise(tanhprime);
		}

		static matx::Matrix<double> none(matx::Matrix<double> x) {
			return x;
		}

		static matx::Matrix<double> noneprime(matx::Matrix<double> x) {
			std::vector<std::vector<double>> ans(x.shape()[0], std::vector<double>(x.shape()[1], 1));
			return matx::Matrix<double>(ans);
		}
};
} // namespace nn