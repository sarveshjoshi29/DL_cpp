#include "Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
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
class AttentionPool : public Layer {
		matx::Matrix<double> softmax_scores;
		matx::Matrix<double> H;

	public:
		matx::Matrix<double> forward_pass(matx::Matrix<double> x) override;
		std::vector<int> initialize(std::vector<int> input_dims) override;
		matx::Matrix<double> backward_pass(matx::Matrix<double> dJ_da) override;
};
} // namespace nn
