#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include "loss.hpp"
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
class GD_Momentum : public nn::optimizer {
	public:
		std::vector<matx::Matrix<double>> vdw;
		std::vector<matx::Matrix<double>> vdb;
		double beta;
		double alpha;

		GD_Momentum(double beta = 0.9, double alpha = 0.01);
		void init_params(std::vector<nn::Layer*>& layers) override;
		void update_params(std::vector<nn::Layer*>& layers) override;
};
} // namespace nn