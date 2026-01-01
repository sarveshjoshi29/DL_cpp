#include "gd_momentum.hpp"
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

GD_Momentum::GD_Momentum(double beta, double alpha) : beta{beta}, alpha{alpha} {
}

void GD_Momentum::init_params(std::vector<nn::Layer*>& layers) {
	vdw.resize(layers.size());
	vdb.resize(layers.size());
	// std::cout << "fine\n";
	for(int i = 0; i < vdw.size(); i++) {
		//	std::cout << layers[0]->dw.size() << "\n";
		if(!layers[i]->is_trainable) {
			continue;
		}
		std::vector<std::vector<double>> tempvec(layers[i]->dw.size(), std::vector<double>(layers[i]->dw[0].size(), 0));
		// std::cout << "okk\n";
		matx::Matrix<double> temp(tempvec);
		vdw[i] = temp;

		std::vector<std::vector<double>> tempvec2(layers[i]->db.size(), std::vector<double>(layers[i]->db[0].size(), 0));
		matx::Matrix<double> temp2(tempvec2);
		vdb[i] = temp2;
	}
}

void GD_Momentum::update_params(std::vector<nn::Layer*>& layers) {

	for(size_t i = 0; i < layers.size(); i++) {
		if(!layers[i]->is_trainable) {
			continue;
		}
		vdw[i] = vdw[i] * beta + (layers[i]->dw) * (1 - beta);
		layers[i]->w = layers[i]->w - (vdw[i] * alpha);

		vdb[i] = vdb[i] * beta + (layers[i]->db) * (1 - beta);
		layers[i]->b = layers[i]->b - (vdb[i] * alpha);
	}
}

} // namespace nn