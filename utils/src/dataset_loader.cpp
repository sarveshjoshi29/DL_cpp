#include "dataset_loader.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace utils {
csv_parser::csv_parser(std::string file_name) {
	std::ifstream infile(file_name);
	if(!infile.is_open()) {
		throw std::runtime_error("Could not open file: " + file_name);
	}
	parse(infile);
}

void csv_parser::parse(std::ifstream& infile) {
	std::string line;
	bool ct{};

	while(std::getline(infile, line)) {
		if(!ct) {
			ct = 1;
			continue;
		}
		std::stringstream ss(line);
		std::string curr_token;
		std::vector<double> ans;
		while(std::getline(ss, curr_token, ',')) {
			ans.push_back(std::stod(curr_token));
		}

		double y = ans.back();
		ans.pop_back();
		data_y.push_back({y});
		data_X.emplace_back(std::move(ans));
	}
}
} // namespace utils
