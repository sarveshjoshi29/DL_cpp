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

		static double mse(const matx::Matrix<double>& y, const matx::Matrix<double>& y_pred) {
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

		static matx::Matrix<double> mseprime(const matx::Matrix<double>& y, const matx::Matrix<double>& y_pred) {
			return (y_pred - y);
		}

		static double bce(double y, double y_pred) {
			double epsilon = 1e-15;
			if(y_pred < epsilon)
				y_pred = epsilon;
			if(y_pred > 1.0 - epsilon)
				y_pred = 1.0 - epsilon;
			return -(1 - y) * log(1 - y_pred) - y * (log(y_pred));
		}

		static double bceprime(double y, double y_pred) {
			double epsilon = 1e-15;
			y_pred = std::clamp(y_pred, epsilon, 1.0 - epsilon); // Added for safety!
			return (1 - y) / (1 - y_pred) - y / (y_pred);
		}

		static double bce(const matx::Matrix<double>& y, const matx::Matrix<double>& y_pred) {
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
					ans += bce(y[i][j], y_pred[i][j]);
				}
			}
			ans /= (y_shape[0] * y_shape[1]);
			return ans;
		}

		static matx::Matrix<double> bceprime(const matx::Matrix<double>& y, const matx::Matrix<double>& y_pred) {
			matx::Matrix<double> ans(y.shape()[0], y.shape()[1]);
			std::vector<int> y_shape = y.shape();
			std::vector<int> y_pred_shape = y_pred.shape();
			if(y_shape[0] != y_pred_shape[0] || y_shape[1] != y_pred_shape[1]) {
				// std::cout << "here\n";
				std::cerr << "Incompatible shapes!\n";
				exit(1);
			}
			for(int i = 0; i < y_shape[0]; i++) {
				for(int j = 0; j < y_shape[1]; j++) {
					ans[i][j] = bceprime(y[i][j], y_pred[i][j]);
				}
			}
			ans = ans / (y_shape[0] * y_shape[1]);
			return ans;
		}
};
} // namespace nn