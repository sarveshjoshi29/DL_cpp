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
class RMSProp : public optimizer {
	public:
		static double square(double x);
		static double rt(double x);

	public:
		std::vector<matx::Matrix<double>> sdw;
		std::vector<matx::Matrix<double>> sdb;
		double alpha;
		double beta;

		RMSProp(double beta = 0.5, double alpha = 0.01);
		void init_params(std::vector<nn::Layer*>& layers);
		void update_params(std::vector<nn::Layer*>& layers);
};
} // namespace nn