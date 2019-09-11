#include <unistd.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct Data{
	vector<double> parameters;
	vector<double> result;
	int label;

	Data(vector<double> par, vector<double> res, int lab): label(lab), result(res), parameters(par){}
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
		vector<double> par(string_data[i].size() - 1); 

		// First parameter is label
		int lab = stoi(string_data[i][0]);
		vector<double> res(10, 0.0);
		res[lab] = 1.0;


		for(int j = 0; j < string_data[i].size() - 1; j++){
			par[j] = 1.0 * stold(string_data[i][j + 1]) / 255.0;
		}

		data.push_back(Data(par, res, lab));
	}

	f.close();
}

// ----------------------------------------------------------------------------


struct Connection{
    double weight;
    double delta_weight;
};


class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
    Neuron(unsigned num_outputs, unsigned my_index);
    void set_output_val(double val) { m_output_val = val; }
    double get_output_val() const { return m_output_val; }
    void feed_forward(const Layer &prev_layer);
    void calc_output_gradients(double target_val);
    void calc_hidden_gradients(const Layer &next_layer);
    void update_input_weights(Layer &prev_layer);

private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transfer_function(double x);
    static double transfer_function_derivative(double x);
    static double random_weight() { return (rand() - (RAND_MAX / 2.0)) / double(RAND_MAX * 5.0); }
    double sum_DOW(const Layer &next_layer) const;
    double m_output_val;
    vector<Connection> m_output_weights;
    unsigned m_my_index;
    double m_gradient;
};

double Neuron::eta = 0.001;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.9;   // momentum, multiplier of last delta_weight, [0.0..1.0]


void Neuron::update_input_weights(Layer &prev_layer){
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prev_layer.size(); ++n) {
        Neuron &neuron = prev_layer[n];
        double old_delta_weight = neuron.m_output_weights[m_my_index].delta_weight;

        double new_delta_weight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.get_output_val()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * old_delta_weight;

        neuron.m_output_weights[m_my_index].delta_weight = new_delta_weight;
        neuron.m_output_weights[m_my_index].weight += new_delta_weight;
    }
}

double Neuron::sum_DOW(const Layer &next_layer) const{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < next_layer.size() - 1; ++n) {
        sum += m_output_weights[n].weight * next_layer[n].m_gradient;
    }

    return sum;
}

void Neuron::calc_hidden_gradients(const Layer &next_layer){
    double dow = sum_DOW(next_layer);
    m_gradient = dow * Neuron::transfer_function_derivative(m_output_val);
}

void Neuron::calc_output_gradients(double target_val){
    double delta = target_val - m_output_val;
    m_gradient = delta * Neuron::transfer_function_derivative(m_output_val);
}

double Neuron::transfer_function(double x){
    // tanh - output range [-1.0..1.0]

    return tanh(x);
}

double Neuron::transfer_function_derivative(double x){
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::feed_forward(const Layer &prev_layer){
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prev_layer.size(); ++n) {
        sum += prev_layer[n].get_output_val() *
                prev_layer[n].m_output_weights[m_my_index].weight;
    }

    m_output_val = Neuron::transfer_function(sum);
}

Neuron::Neuron(unsigned num_outputs, unsigned my_index){
    for (unsigned c = 0; c < num_outputs; ++c) {
        m_output_weights.push_back(Connection());
        m_output_weights.back().weight = random_weight();
    }

    m_my_index = my_index;
}


// ****************** class NeuralNetwork ******************
class NeuralNetwork
{
public:
    NeuralNetwork(const vector<unsigned> &topology);
    void feed_forward(const vector<double> &input_vals);
    void back_prop(const vector<double> &target_vals);
    void get_results(vector<double> &result_vals) const;
    double get_recent_average_error(void) const { return m_recent_average_error; }

private:
    vector<Layer> m_layers; // m_layers[layer_num][neuron_num]
    double m_error;
    double m_recent_average_error;
    static double m_recent_average_smoothing_factor;
};


double NeuralNetwork::m_recent_average_smoothing_factor = 100.0; // Number of training samples to average over


void NeuralNetwork::get_results(vector<double> &result_vals) const{
    result_vals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        result_vals.push_back(m_layers.back()[n].get_output_val());
    }
}

void NeuralNetwork::back_prop(const vector<double> &target_vals){
    // Calculate overall net error (RMS of output neuron errors)

    Layer &output_layer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
        double delta = target_vals[n] - output_layer[n].get_output_val();
        m_error += delta * delta;
    }
    m_error /= output_layer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement

    m_recent_average_error =
            (m_recent_average_error * m_recent_average_smoothing_factor + m_error)
            / (m_recent_average_smoothing_factor + 1.0);

    // Calculate output layer gradients

    for (unsigned n = 0; n < output_layer.size() - 1; ++n) {
        output_layer[n].calc_output_gradients(target_vals[n]);
    }

    // Calculate hidden layer gradients

    for (unsigned layer_num = m_layers.size() - 2; layer_num > 0; --layer_num) {
        Layer &hidden_layer = m_layers[layer_num];
        Layer &next_layer = m_layers[layer_num + 1];

        for (unsigned n = 0; n < hidden_layer.size(); ++n) {
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layer_num = m_layers.size() - 1; layer_num > 0; --layer_num) {
        Layer &layer = m_layers[layer_num];
        Layer &prev_layer = m_layers[layer_num - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].update_input_weights(prev_layer);
        }
    }
}

void NeuralNetwork::feed_forward(const vector<double> &input_vals){
    assert(input_vals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < input_vals.size(); ++i) {
        m_layers[0][i].set_output_val(input_vals[i]);
    }

    // forward propagate
    for (unsigned layer_num = 1; layer_num < m_layers.size(); ++layer_num) {
        Layer &prev_layer = m_layers[layer_num - 1];
        for (unsigned n = 0; n < m_layers[layer_num].size() - 1; ++n) {
            m_layers[layer_num][n].feed_forward(prev_layer);
        }
    }
}

NeuralNetwork::NeuralNetwork(const vector<unsigned> &topology)
{
    unsigned num_layers = topology.size();
    for (unsigned layer_num = 0; layer_num < num_layers; ++layer_num) {
        m_layers.push_back(Layer());
        unsigned num_outputs = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; ++neuron_num) {
            m_layers.back().push_back(Neuron(num_outputs, neuron_num));
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().set_output_val(1.0);
    }
}


void show_vector_vals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << "\n";
}


int main()
{
    srand(time(NULL));

    read_data("data/mnist_train.csv", learn_data);
	read_data("data/mnist_test.csv", test_data);

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology = {784, 128, 50, 20, 10};
    NeuralNetwork neural_network(topology);

    vector<double> input_vals, target_vals, result_vals;
    int trainingPass = 0;

    for(int i = 0; i < 500; i++){
        int correct = 0;
        for(int j = 0; j < learn_data.size(); j++) {
            Data &x = learn_data[j];
            cout << "Pass " << j;
            
            // Feed forward
            neural_network.feed_forward(x.parameters);

            // Collect the net's actual output results:
            neural_network.get_results(result_vals);
            show_vector_vals("Outputs:", result_vals);

            int prediction = 0;
			for(int i = 0; i < result_vals.size(); i++){
				if(result_vals[i] > result_vals[prediction]){
					prediction = i;
				}
			}

            cout << prediction << "  " << x.label << "\n";

            correct += (prediction == x.label);

            // Train the net what the outputs should have been
            show_vector_vals("Targets:", x.result);
            neural_network.back_prop(x.result);

            // Report how well the training is working, average over recent samples:
            cout << "Network recent average error: "
                    << neural_network.get_recent_average_error() << "\n\n";
        }

        // Report overall success on training data        
        ofstream res("results.txt", std::ios_base::app);
        res << "TRAINING RESULT " << i + 1 << ": " << 100.0 * correct / learn_data.size() << "\n\n";
        res.close();



        correct = 0;
        for(int j = 0; j < test_data.size(); j++) {
            Data &x = test_data[j];
            cout << "Pass " << j << "\n";
            
            // Feed forward
            neural_network.feed_forward(x.parameters);

            // Collect the net's actual output results:
            neural_network.get_results(result_vals);
            show_vector_vals("Outputs:", result_vals);

            int prediction = 0;
			for(int i = 0; i < result_vals.size(); i++){
				if(result_vals[i] > result_vals[prediction]){
					prediction = i;
				}
			}

            cout << prediction << "  " << x.label << "\n";

            correct += (prediction == x.label);

            // Train the net what the outputs should have been
            show_vector_vals("Targets:", x.result);
        }

        // Report overall success on testing data        
        res.open("results.txt", std::ios_base::app);
        res << "TESTING RESULT " << i + 1 << ": " << 100.0 * correct / test_data.size() << "\n\n";
        res.close();
    } 

    cout << endl << "Done" << endl;
}