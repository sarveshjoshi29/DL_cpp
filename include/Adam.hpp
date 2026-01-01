#pragma once
#include "Matrix.hpp"
#include "optimizer.hpp"
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
class Adam : public optimizer {
	public:
		double alpha;
		double beta1;
		double beta2;
		static double rt(double x);
		static double square(double x);
		std::vector<matx::Matrix<double>> m;
		std::vector<matx::Matrix<double>> v;
		std::vector<matx::Matrix<double>> m_bias;
		std::vector<matx::Matrix<double>> v_bias;
		std::vector<matx::Matrix<double>> m_corrected;
		std::vector<matx::Matrix<double>> m_bias_corrected;
		std::vector<matx::Matrix<double>> v_corrected;
		std::vector<matx::Matrix<double>> v_bias_corrected;
		size_t time;
		Adam(double alpha = 0.01, double beta1 = 0.9, double beta2 = 0.99);
		void init_params(std::vector<nn::Layer*>& layers);
		void update_params(std::vector<nn::Layer*>& layers);
};

} // namespace nn