#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;

const double percent_tests = 20.0;
const double INF = 1e9;

// parameter 0 is class 
vector<vector<int>> data, test_data, learn_data;
vector<vector<string>> string_data, attributes; 
vector<map<string, int>> mapping; 

void read_data(string file){
	ifstream f(file);
	string line;

	while(getline(f, line)){
		vector<string> current_data;

		int last = 0;
		while(true){
			int ind = line.find(",", last);
			if(ind == -1){
				break;
			}

			string cur = line.substr(last, ind - last);
			current_data.push_back(cur);
			last = ind + 1; 
		}

		string cur = line.substr(last, line.length() - last);
		current_data.push_back(cur);
		string_data.push_back(current_data);
	}

	attributes.resize(string_data[0].size());
	for(int i = 0; i < string_data.size(); i++){
		for(int j = 0; j < string_data[i].size(); j++){
			attributes[j].push_back(string_data[i][j]);
		}
	}

	mapping.resize(attributes.size());
	for(int i = 0; i < attributes.size(); i++){
		sort(attributes[i].begin(), attributes[i].end());
		attributes[i].resize(std::distance(attributes[i].begin(), unique(attributes[i].begin(), attributes[i].end())));
		for(int j = 0; j < attributes[i].size(); j++){
			mapping[i][attributes[i][j]] = j;
		}
	}

	data.resize(string_data.size());
	for(int i = 0; i < string_data.size(); i++){
		data[i].resize(string_data[i].size());
		for(int j = 0; j < string_data[i].size(); j++){
			data[i][j] = mapping[j][string_data[i][j]];
		}
	}

	f.close();
}

void separate_data(){
	vector<int> permutation, used(data.size(), 0);
	permutation.resize(data.size());
	for(int i = 0; i < data.size(); i++){
		permutation[i] = i;
	}

	random_shuffle(permutation.begin(), permutation.end());

	int test_number = data.size() * percent_tests / 100.0;
	for(int i = 0; i < test_number; i++){
		test_data.push_back(data[permutation[i]]);
		used[permutation[i]] = 1;
	}

	for(int i = 0; i < data.size(); i++){
		if(!used[i]){
			learn_data.push_back(data[i]);
		}
	}
}

struct Node{
	int n, p;
	int prediction;
	int attribute;
	vector<shared_ptr<Node>> children;

	Node(){
		n = 0;
		p = 0;
		prediction = 0;
		attribute = 0,
		children.clear();
	}
};


class DecisionTree{
public:
	shared_ptr<Node> root;

	DecisionTree(): root(make_shared<Node>()){}


private:
	long double I(shared_ptr<Node> node){
		long double probability_p = 1.0 * node -> p / (node -> p + node -> n); 
		long double probability_n = 1.0 * node -> n / (node -> p + node -> n);
		long double entropy_p = 0;
		if(probability_p != 0){
			entropy_p = -1.0 * probability_p * log2(probability_p);
		}

		long double entropy_n = 0;
		if(probability_n != 0){
			entropy_n = -1.0 * probability_n * log2(probability_n);
		}
		long double entropy = entropy_p + entropy_n;
		return entropy;
	}


	long double calculate_attribute_entropy(int n, int p, vector<vector<int> > &data, int attribute){
		vector<shared_ptr<Node>> nodes(mapping[attribute].size(), NULL);
		for(int i = 0; i < data.size(); i++){
			int attribute_id = data[i][attribute];
			if(!nodes[attribute_id]){
				nodes[attribute_id] = make_shared<Node>();
			}

			if(data[i][0] == 1){
				(nodes[attribute_id] -> p)++;
			}
			else{
				(nodes[attribute_id] -> n)++;
			}
		}

		long double expected_information = 0.0;
		for(int i = 0; i < nodes.size(); i++){
			if(nodes[i] == NULL){
				continue;
			}

			long double probability = 1.0 * (nodes[i] -> p + nodes[i] -> n) / (n + p);
			expected_information += probability * I(nodes[i]);
		}
		return expected_information;
	}


public:

	void create_decision_tree(vector<vector<int> > &data, shared_ptr<Node> node){
		// Set values for current node
		node -> attribute = -1;
		for(int i = 0; i < data.size(); i++){
			if(data[i][0] == 0){
				(node -> n)++;
			}
			else{
				(node -> p)++;
			}
		}

		if(node -> p >= node -> n){
			node -> prediction = 1;
		}
		else{
			node -> prediction = 0; 
		}

		// Whole subtree has same value so we don't need to branch it anymore
		if(node -> p == 0 || node -> n == 0){
			return;
		}

		// Find best attribute according to entropy
		int attribute = -1;
		long double best_value = INF;
		for(int i = 1; i < data[0].size(); i++){
			if(data[0][i] == -1){
				continue;
			}


			long double current_value = calculate_attribute_entropy(node -> p, node -> n, data, i);
			
			if(current_value < best_value){
				best_value = current_value;
				attribute = i;
			}
		}


		node -> attribute = attribute; 
		if(attribute == -1){
			return;
		}

		// Partition data and create child nodes
		vector<vector<vector<int>>> data_children(mapping[attribute].size());
		node -> children.resize(mapping[attribute].size(), NULL);
		for(int i = 0; i < data.size(); i++){
			int attribute_id = data[i][attribute];
			data[i][attribute] = -1;

			if(!node -> children[attribute_id]){
				node -> children[attribute_id] = make_shared<Node>();
			}

			
			if(data[i][0] == 1){
				(node -> children[attribute_id] -> p)++;
			}
			else{
				(node -> children[attribute_id] -> n)++;
			}

			data_children[attribute_id].push_back(data[i]);
		}

		for(int i = 0; i < data_children.size(); i++){
			if(data_children[i].size() == 0){
				continue;
			}

			create_decision_tree(data_children[i], node -> children[i]);
		}
	}

	int make_prediction(vector<int> &test, shared_ptr<Node> node){
		if(node -> attribute == -1){
			return node -> prediction;
		}

		if(node -> children[test[node -> attribute]] == NULL){
			return node -> prediction;
		}

		return make_prediction(test, node -> children[test[node -> attribute]]);
	}
};



int main(){
	srand(time(NULL));

	read_data("data/balance-scale.data");
	separate_data();

	DecisionTree tree;
	tree.create_decision_tree(learn_data, tree.root);

	int correct = 0;
	for(int i = 0; i < test_data.size(); i++){
		int predicted_value = tree.make_prediction(test_data[i], tree.root);
		if(predicted_value == test_data[i][0]){
			correct++;
		}
	}

	cout << fixed << setprecision(2) << "Predictions accuracy: " << 100.0 * correct / test_data.size() << endl;
	return 0;
}