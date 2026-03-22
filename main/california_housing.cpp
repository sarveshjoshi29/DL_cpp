#include "Adam.hpp"
#include "Dense.hpp"
#include "Dropout.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "activations.hpp"
#include "dataset_loader.hpp"
#include "file_manager.hpp"
#include "gd_momentum.hpp"
#include "loss.hpp"
#include "wrapper.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
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

int main() {

	// std::cout << "new code\n";

	auto start_time = std::chrono::high_resolution_clock::now();
	std::cout << "Loading dataset california housing..." << std::endl;

	utils::csv_parser parser("../dataset/california_housing_train.csv");
	utils::csv_parser parser_test("../dataset/california_housing_test.csv");
	matx::Matrix<double> X(std::move(parser.data_X));
	matx::Matrix<double> y(std::move(parser.data_y));

	matx::Matrix<double> X_test(std::move(parser_test.data_X));
	matx::Matrix<double> y_test(std::move(parser_test.data_y));

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;

	std::cout << "-----------------------------------\n";
	std::cout << "Data loaded successfully!" << std::endl;
	std::cout << "Total rows: " << X.shape()[0] << "   Total columns: " << X.shape()[1] << std::endl;
	std::cout << "Time taken: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
	std::cout << "-----------------------------------\n";

	// //(a * (double)5).print_Matrix();
	nn::Layer* layer1 = new nn::Dense(20, "tanh");
	nn::Layer* layer2 = new nn::Dense(20, "sigmoid");
	nn::Layer* drop = new nn::Dropout(0.01);
	nn::Layer* layer3 = new nn::Dense(1, "none");

	nn::Adam optim(0.005);
	// // std::cout << "np\n";
	nn::wrapper model({layer1, layer2, drop, layer3}, &optim);
	model.train(X, y, 20);
	// // std::cout << "what\n;";
	matx::Matrix<double> y_pred = model.predict(X_test);
	// std::cout << y_pred.shape()[0] << " " << y_pred.shape()[1] << "\n";
	// std::cout << y_test.shape()[0] << " " << y_test.shape()[1] << "\n";
	double test_loss = nn::loss::mse(y_test.transpose(), y_pred);
	std::cout << "Loss on test dataset is " << test_loss << "\n";

	utils::file_manager f;
	f.save<double>("../predictions/cali_predicted.csv", y_pred.transpose());
	// y_pred.print_Matrix();
}

// Note --to make sure the wrapper object is deleted it is the user's responsibility to provide its scope
// particularly when the same optimizer is used for different models.

/*
to change ---
1) add He/glorot initialization. Model is highly sensitive to weight initialization. -- DONE
2) add a element wise function which takes a func ptr as input in matx::Matrix -- DONE
3) replace exit 1's with std::throw everywhere , use size_t wherever warning -- DONE
4) separate nn.cpp into diff files , one for each class -- DONE


5) add optimizers like adam , sgd -- DONE
6) add option to print loss at each epoch -- DONE
7) add features like dropout!! -> complicated because layer type can cause issues :( ----------------- DONEEE

** Changed layer interactions -- updated both forward and backprop of Dense.. see interfaces defined in Layer.hpp.
Use those templates to build more layers in the future.

dont forget to define is_trainable for each layer!!

8) think about the Matrix constructor how i want to manage it


9) Save weights of a model in binary and implement reconstruction when model.load() is called!
*/