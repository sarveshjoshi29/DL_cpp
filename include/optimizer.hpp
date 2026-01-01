#pragma once
#include "Layer.hpp"
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

// IDEA IS -- BATCH HANDLER WILL TAKE CARE OF BATCHES... FOR RMSPROP/MOMENTUM/ADAM I NEED TO CREATE CLASSES TO CHANGE MY UPDATE_PARAMS FUNCTION! SO I
// THINK I SHOULD PLAN TO REMOVE ONLY UPDATE_PARAMS FROM BATCH HANDLER BUT KEEP THE REST.
//
//
// have defined class optimizer as an interface which must have two functions - init_params() and update_params();
//
//
// DEFINE UPDATE_PARAMS(LAYERS,LEARNING_RATE) IN DIFF CLASSES AND PASS POINTER THROUGH CONSTRUCTOR OF BATCH HANDLER.

// need to debug batch handler --- done
// and all init_param , update_param providers :) --- ADAM rem.

// Structure -- templated wrapper and batch handler to handle diff optims. You can make a custom optimizer which MUST have two functions -
// 1) init_params(layers) -- this should initialized all the internal params required (can use size of each layer).
// 2) update_params(layers) -- this should update w,b of EACH Layer when dw,db are known and using internal parameters

// any other functions for custom optim will be used within that class only!

namespace nn {
class optimizer {
	public:
		virtual void init_params(std::vector<nn::Layer*>&) = 0;
		virtual void update_params(std::vector<nn::Layer*>&) = 0;
		virtual ~optimizer() {};
		// this calls destructor of child class
};

} // namespace nn