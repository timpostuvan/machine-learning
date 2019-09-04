#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <tuple>

using namespace std;

typedef tuple<long double, long double, long double, long double, string> DataType;

const int RESTARTS_NUMBER = 10000;
const int MAX_ITERATIONS = 10000;
const double INF = 1e14;

// Parameter 4 is class
vector<DataType> test_data; 
vector<vector<long double>> data;


void read_iris_dataset(string file){
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

		// Data to check accuracy
		DataType current_formatted = make_tuple(stold(current_data[0]), stold(current_data[1]), 
								stold(current_data[2]), stold(current_data[3]), current_data[4]); 
		test_data.push_back(current_formatted);

		// Data for clustering
		vector<long double> current_learn;
		for(int i = 0; i < current_data.size() - 1; i++){
			current_learn.push_back(stold(current_data[i]));
		}
		data.push_back(current_learn);
	}

	f.close();
}

vector<vector<long double>> initial_clusters(vector<vector<long double>> &data, int k){
	vector<int> permutation;
	for(int i = 0; i < data.size(); i++){
		permutation.push_back(i);
	}

	random_shuffle(permutation.begin(), permutation.end());

	vector<vector<long double>> clusters;
	for(int i = 0; i < k; i++){
		clusters.push_back(data[permutation[i]]);
	}
	return clusters;
}

long double distance(vector<long double> &a, vector<long double> &b){
	long double ret = 0.0;
	for(int i = 0; i < a.size(); i++){
		ret += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return ret;
}

vector<int> assign_to_centroids(vector<vector<long double>> &data, vector<vector<long double>> &centroids){
	vector<int> centroid_id(data.size());
	int k = centroids.size();

	for(int i = 0; i < data.size(); i++){
		int best_id = 0;
		long double best_distance = INF;
		for(int j = 0; j < k; j++){
			long double cur_distance = distance(data[i], centroids[j]);
			if(cur_distance < best_distance){
				best_distance = cur_distance;
				best_id = j;
			}
		}
		centroid_id[i] = best_id;
	}
	return centroid_id;
}

vector<vector<long double>> recalculate_centroids(vector<vector<long double>> &data, vector<int> centroid_id, int k){
	vector<vector<vector<long double>>> data_clusters(k);
	for(int i = 0; i < data.size(); i++){
		data_clusters[centroid_id[i]].push_back(data[i]);
	}

	vector<vector<long double>> clusters; 
	for(int i = 0; i < k; i++){

		// Add a new random centroid start since this cluster is empty
		if(data_clusters[i].size() == 0){
			clusters.push_back(data[rand() % data.size()]);
			continue;
		}

		vector<long double> mean(data_clusters[i][0].size(), 0.0);
		for(vector<long double> &cur : data_clusters[i]){
			for(int j = 0; j < cur.size(); j++){
				mean[j] += cur[j];
			}
		}

		for(int j = 0; j < mean.size(); j++){
			mean[j] /= 1.0 * data_clusters[i].size();
		}		
		clusters.push_back(mean);
	}

	return clusters;
}


vector<vector<long double>> k_means_clustering(vector<vector<long double>> &data, int k){
	vector<vector<long double>> centroids = initial_clusters(data, k);
	vector<int> previous_centroid_id(data.size(), -1);
	for(int i = 0; i < MAX_ITERATIONS; i++){
		vector<int> centroid_id = assign_to_centroids(data, centroids);
		if(centroid_id == previous_centroid_id){
			return centroids;
		}
		centroids = recalculate_centroids(data, centroid_id, k);
		previous_centroid_id = centroid_id;
	}

	return centroids;
}

long double cost_function(vector<vector<long double>> &data, vector<vector<long double>> &centroids){
	vector<int> centroid_id = assign_to_centroids(data, centroids);

	long double ret = 0.0;
	for(int i = 0; i < data.size(); i++){
		ret += distance(data[i], centroids[centroid_id[i]]);
	}

	return ret;
}

void check_success(vector<vector<long double>> &best_centroids, vector<vector<long double>> &data, vector<DataType> &test_data){
	int k = best_centroids.size();
	vector<int> centroid_id = assign_to_centroids(data, best_centroids);
	

	vector<vector<string>> cluster_values(k);
	for(int i = 0; i < data.size(); i++){
		cluster_values[centroid_id[i]].push_back(get<4>(test_data[i]));
	}

	for(int i = 0; i < k; i++){
		map<string, int> value_count;

		for(int j = 0; j < cluster_values[i].size(); j++){
			value_count[cluster_values[i][j]]++;
		}

		cout << "Cluster " << i + 1 << endl;
		for(auto it : value_count){
			long double percent = 1.0 * it.second / cluster_values[i].size();
			cout << it.first << ": " << it.second << "  " << percent << endl;
		}
	}
}


int main(){
	srand(time(NULL));

	read_iris_dataset("data/iris.data");

	int k = 3;

	long double best_value = INF;
	vector<vector<long double>> best_centroids;
	for(int iteration = 0; iteration < RESTARTS_NUMBER; iteration++){
		cout << "Iteration: " << iteration << endl;
		vector<vector<long double>> current_centroids = k_means_clustering(data, k);
		long double current_value = cost_function(data, current_centroids);
		if(current_value < best_value){
			best_value = current_value;
			best_centroids = current_centroids;
		}
	}

	check_success(best_centroids, data, test_data);

	return 0;
}