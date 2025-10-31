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
class loss {
	public:
		static double mse(double y, double y_pred) {
			double t = y - y_pred;
			return t * t;
		}

		static double mseprime(double y, double y_pred) {
			double t = y_pred - y;
			return t;
		}

		static double mse(matx::Matrix<double> y, matx::Matrix<double> y_pred) {
			std::vector<int> y_shape = y.shape();
			std::vector<int> y_pred_shape = y_pred.shape();
			if(y_shape[0] != y_pred_shape[0] || y_shape[1] != y_pred_shape[1]) {
				// std::cout << "here\n";
				std::cerr << "Incompatible shapes!\n";
				exit(1);
			}
			double ans{};
			for(int i = 0; i < y_shape[0]; i++) {
				for(int j = 0; j < y_shape[1]; j++) {
					ans += mse(y[i][j], y_pred[i][j]);
				}
			}
			ans /= 2 * (y_shape[0] * y_shape[1]);
			return ans;
		}

		static matx::Matrix<double> mseprime(matx::Matrix<double> y, matx::Matrix<double> y_pred) {
			return y_pred - y;
		}
};
} // namespace nn