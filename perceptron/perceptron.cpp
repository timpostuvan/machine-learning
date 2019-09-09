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

const double INF = 1e14;

struct Data{
	vector<long double> parameters;
	int label;

	Data(vector<long double> par, int lab): label(lab), parameters(par){}
};

vector<Data> test_data, learn_data;

void read_data(string file, vector<Data> &data){
	vector<vector<string>> string_data;
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

	for(int i = 0; i < string_data.size(); i++){
		vector<long double> par(string_data[i].size()); 

		// First parameter is label
		int lab = stoi(string_data[i][0]);

		// First parameter should be 1 (for threshold)
		par[0] = 1.0;
		for(int j = 1; j < string_data[i].size(); j++){
			par[j] = stold(string_data[i][j]);
		}

		data.push_back(Data(par, lab));
	}

	f.close();
}

class Perceptron{
	int number;
	vector<Data>& data;
	vector<long double> w; 

	long long dot_product(vector<long double> a, vector<long double> b){
		long long product = 0;
		for(int i = 0; i < a.size(); i++){
			product += a[i] * b[i];
		}
		
		return product;
	}

	int correct_label(Data &x){
		return (x.label == number);
	}

	void update_weights(Data &x, long double training_rate){
		int prediction = predict_label(x);
		int correct_prediction = correct_label(x);
		
		for(int i = 0; i < x.parameters.size(); i++){
			long double add = training_rate * (correct_prediction - prediction) * x.parameters[i];
			w[i] += add;
		}
	}

public:

	Perceptron(int num, vector<Data> &dat): number(num), data(dat){
		// Initialise w
		w.resize(data[0].parameters.size(), 0);
	}

	long double prediction_strength(Data &x){
		return dot_product(w, x.parameters);
	}

	int predict_label(Data &x){
		long double strength = prediction_strength(x);
		int prediction;
		if(strength > 0){
			prediction = 1;
		}
		else{
			prediction = 0;
		}

		return prediction;
	}

	void train_perceptron(int iterations, long double training_rate){
		for(int it = 0; it < iterations; it++){
			for(int i = 0; i < data.size(); i++){
				update_weights(data[i], training_rate);
			}
		}
	}
};

int predict_number(vector<Perceptron> &perceptrons, Data &x){
	int prediction = 0;
	long double best_strength = -INF;
	for(int i = 0; i < perceptrons.size(); i++){
		if(perceptrons[i].predict_label(x) == 1){
			long double strength = perceptrons[i].prediction_strength(x);
			if(best_strength < strength){
				best_strength = strength;
				prediction = i;
			}
		}
	}

	return prediction;
}



int main(){
	read_data("data/mnist_train.csv", learn_data);
	read_data("data/mnist_test.csv", test_data);

	const int number_of_iterations = 200;
	const long double learning_rate = 0.0000007;
	
	vector<Perceptron> perceptrons;
	for(int number = 0; number < 10; number++){
		perceptrons.push_back(Perceptron(number, learn_data));
		perceptrons.back().train_perceptron(number_of_iterations, learning_rate);
	}

	int correct = 0;
	for(int i = 0; i < test_data.size(); i++){
		int prediction = predict_number(perceptrons, test_data[i]);
		correct += (prediction == test_data[i].label);
	}

	cout << "Number of iterations: " << number_of_iterations << endl
		 << fixed << setprecision(7) << "Learning rate: " << learning_rate << endl;

	cout << fixed << setprecision(2) << "Predictions accuracy: " << 100.0 * correct / test_data.size() << endl << endl;

	return 0;
}