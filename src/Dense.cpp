#include "Dense.hpp"
#include "Layer.hpp"
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

Dense::Dense(int nl, string act) {
	if(act == "relu") {
		activation = nn::activations::relu;
		activationprime = nn::activations::reluprime;
	}
	if(act == "sigmoid") {
		activation = nn::activations::sigmoid;
		activationprime = nn::activations::sigmoidprime;
	}
	if(act == "tanh") {
		activation = nn::activations::tanh;
		activationprime = nn::activations::tanhprime;
	}
	if(act == "none") {
		activation = nn::activations::none;
		activationprime = nn::activations::noneprime;
	}
	num_neurons = nl;
	z.resize(nl, 1);
	a = activation(z);
}

void Dense::set_weights(int dims_prev_layer, std::mt19937& gen) {

	w.resize(num_neurons, dims_prev_layer);
	// std::cout << (activation == static_cast<matx::Matrix<double> (*)(matx::Matrix<double>)>(nn::activations::relu)) << " dumb\n";
	if(activation == static_cast<matx::Matrix<double> (*)(matx::Matrix<double>)>(nn::activations::relu)) {
		double std = sqrt(2.0 / (dims_prev_layer));
		std::normal_distribution<> dis(0.0, std);
		for(int i = 0; i < num_neurons; i++) {
			for(int j = 0; j < dims_prev_layer; j++) {
				w[i][j] = dis(gen);
			}
		}
	} else {
		// std::cout << "reached\n";
		double std = sqrt(2.0 / (num_neurons + dims_prev_layer));
		std::normal_distribution<double> dis(0.0, std);
		for(int i = 0; i < num_neurons; i++) {
			for(int j = 0; j < dims_prev_layer; j++) {
				w[i][j] = dis(gen);
			}
		}
	}
	dw.resize(w.shape()[0], w.shape()[1]);
	// std::cout << "set wts exited\n";
}

void Dense::set_bias() {
	b.resize(num_neurons, 1);
	db.resize(b.shape()[0], b.shape()[1]);
	// std::cout << "set bias exited\n";
}

int Dense::initialize(int dims) {
	is_trainable = true;
	std::random_device rd;
	std::mt19937 gen{rd()};
	set_weights(dims, gen);
	set_bias();
	return num_neurons;
}

matx::Matrix<double> Dense::forward_pass(matx::Matrix<double> x) {

	// std::cout << w.shape()[0] << " " << w.shape()[1] << "\n";
	a_prev = x;
	z = (w % x);
	// std::cout << z.shape()[0] << " " << z.shape()[1] << "\n";
	z = z + b;
	// std::cout << b.shape()[0] << " " << b.shape()[1] << "\n";
	a = activation(z);
	return a;
}

matx::Matrix<double> Dense::backward_pass(matx::Matrix<double> dJ_da) {
	double m = a.shape()[1];
	matx::Matrix<double> delta_l = dJ_da * activationprime(z);
	dw = (delta_l % (a_prev.transpose())) * (1.0 / m);
	std::vector<std::vector<double>> temp_b(num_neurons, std::vector<double>(1, 0));
	for(int i = 0; i < num_neurons; i++) {
		double t = 0;
		for(int j = 0; j < m; j++) {
			t += delta_l[i][j];
		}
		t /= m;
		temp_b[i][0] = t;
	}
	db = matx::Matrix<double>(temp_b);
	return w.transpose() % delta_l;
}

} // namespace nn