#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

const double ONE = 1;
const double TWO = 2;
const double learningRate = 0.001;
const double MAX_HOUSE_PRICE = 20000000;
const double MAX_AREA = 20000;
const double MAX_BEDROOMS = 6;
const double MAX_BATHROOMS = 4;
const double MAX_STORIES = 4;
const double MAX_PARKING = 3;
/*********** Convention ************
*
* Always use Layers, Neurons as the order of dimensions
* previous layer to next layer as the order of dimensions
* weight exists from every neuron of the current layer to every neuron of next layer
* bias exists for every neuron
* Differentiation constants will have same dimensions as values, activation values
*
*********** Convention ************/

/*********** General Methods ************/
bool isNumber(const string& s) {
    double f;
    istringstream iss(s);
    // Read the double. noskipws ensures we don't accidentally ignore trailing spaces.
    iss >> noskipws >> f; 
    // It's a valid number only if we successfully read it AND reached the end of the string
    return iss.eof() && !iss.fail(); 
}

void displayData(vector<double> data) {
    int i;
    for (i = 0; i < data.size(); i++) {
        cout<<fixed<<setprecision(6)<<data[i]<<", ";
        cout<<endl;
    }
}

void displayData(vector<vector<double>> data) {
    int i, j;
    for (i = 0; i < data.size(); i++) {
        for (j = 0; j < data[i].size(); j++) {
            cout<<fixed<<setprecision(6)<<data[i][j]<<", ";
        }
        cout<<endl;
    }
}

void displayData(vector<vector<vector<double>>> data) {
    int i, j, k;
    for (i = 0; i < data.size(); i++) {
        cout<<endl<<"data of "<<i<<"th index is: "<<endl<<endl;
        for (j = 0; j < data[i].size(); j++) {
            for (k = 0; k < data[i][j].size(); k++) {
                cout<<fixed<<setprecision(6)<<data[i][j][k]<<", ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}
/*********** General Methods ************/


// Reads a CSV file and converts the data into a 2D vector of doubles
vector<vector<double>> readCSV(string filename, bool skipHeader) {
    vector<vector<double>> dataset;
    ifstream file(filename);
    string line, cellValue;
    
    // Check if file opened successfully
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << "\n";
        return dataset; 
    }
    
    // Skip the first line if it contains column names
    if (skipHeader) {
        getline(file, line);
    }
    
    // Read the file line by line
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        
        // Split the line by commas
        while (getline(ss, cellValue, ',')) {
            try {
                // Convert string to double and push to the row
                if (isNumber(cellValue)) {
                    row.push_back(stof(cellValue));
                }
                else if (cellValue == "yes" || cellValue == "no") {
                    row.push_back(cellValue == "yes" ? 1 : 0);
                }
            } catch (const invalid_argument& e) {
                // If a cell is blank or contains text, default to 0.0f
                row.push_back(0.0f);
            }
        }
        
        // Add the completed row to our dataset
        if (!row.empty()) {
            dataset.push_back(row);
        }
    }
    
    file.close();
    return dataset;
}

double weightConstraints() {
    double limit = 0.679;
    return limit;
}

double generateRandomNumber(double lower_bound, double upper_bound) {
    // 1. random_device pulls a truly random seed from your OS hardware
    static random_device rd;  

    // 2. mt19937 is the Mersenne Twister engine. 
    // We make it 'static' so it is only seeded ONCE during the program's lifetime.
    static mt19937 gen(rd()); 

    // 3. uniform_real_distribution ensures flat, unbiased doubleing-point numbers
    uniform_real_distribution<double> dist(lower_bound, upper_bound);

    return dist(gen);
}

vector<vector<vector<double>>> generateRandomWeights(int numberOfLayers, int numberOfNeurons) {
    int i, j, k;
    double limit = weightConstraints();
    vector<vector<vector<double>>> weights = vector<vector<vector<double>>>(
        numberOfLayers - 1, vector<vector<double>>(
            numberOfNeurons, vector<double>(numberOfNeurons)
        )
    );

    vector<vector<double>>outputWeights = vector<vector<double>>(numberOfNeurons, vector<double>(1));

    for (i = 0; i  + 1 < numberOfLayers; i++) {
        for (j = 0; j < numberOfNeurons; j++) {
            for (k = 0; k < numberOfNeurons; k++) {
                weights[i][j][k] = generateRandomNumber(-limit, limit);
            }
        }
    }

    for (i = 0; i < numberOfNeurons; i++) {
        outputWeights[i][0] = generateRandomNumber(-limit, limit);
    }

    weights.push_back(outputWeights);
    return weights;
}

vector<vector<double>> generateBiases(int layers, int numberOfNeurons) {
    // numberOfLayers - 1 because only output layer neurons will have bias and output bias is added seperately
    vector<vector<double>> biases = vector<vector<double>>(layers - 1, vector<double>(numberOfNeurons, 0));
    vector<double> outputBias = vector<double>(1, 0);
    biases.push_back(outputBias);

    return biases;
}

vector<vector<double>> initializeValues(int numberOfLayers, int numberOfNeurons) {
    vector<vector<double>> res = vector<vector<double>>(numberOfLayers, vector<double>(numberOfNeurons, 0));
    res.push_back(vector<double>(1, 0));

    return res;
}

vector<vector<double>> processData(vector<vector<double>> data) {
    int i, n = data.size();
    for (i = 0; i < n; i++) {
        int l = data[i].size();
        data[i][0] /= MAX_HOUSE_PRICE;
        data[i][1] /= MAX_AREA;
        data[i][2] /= MAX_BEDROOMS;
        data[i][3] /= MAX_BATHROOMS;
        data[i][4] /= MAX_STORIES;
        data[i][10] /= MAX_PARKING;
        swap(data[i][0], data[i][l - 1]);
    }

    return data;
}

double activationFunction(double x) {
    //using sigmoid function as activation function (1 / (1 + e ^ -x))
    double res = ONE / (ONE + exp(-x));
    return res;
}

vector<int> defineNeuralNetwork() {
    //Number of layers in the neural network
    int numberOfLayers = 5;
    //Number of neurons in each layer
    int numberOfNeurons = 11;
    return {numberOfLayers, numberOfNeurons};
}

void neuralNetwork(vector<vector<double>> data) {
    vector<int> nnDimensions = defineNeuralNetwork();
    int numberOfLayers = nnDimensions[0];
    int numberOfNeurons = nnDimensions[1];

    //weights[i][j][k] = weight of the edge joining j th neuron in i-1 th layer and k th neuron in i th layer
    vector<vector<vector<double>>> weights = generateRandomWeights(numberOfLayers, numberOfNeurons);
    vector<vector<double>> biases = generateBiases(numberOfLayers, numberOfNeurons);
    vector<vector<double>> values = initializeValues(numberOfLayers, numberOfNeurons);
    vector<vector<double>> activationValues = initializeValues(numberOfLayers, numberOfNeurons);

    int i, i1, i2, j, k1, k2;
    int noOfBatches = 1000;

    // cout<<"Displaying weights"<<endl;
    // displayData(weights);
    // cout<<endl<<"Displaying biases"<<endl;
    // displayData(biases);
    // cout<<"Starting loop through data records"<<endl;
    for (i2 = 0; i2 < noOfBatches; i2++) {
        cout<<endl<<endl<<"Batch number: "<<i2<<endl<<endl;
        for (i1 = 0; i1 < data.size(); i1++) {
            // cout<<"current data size is: "<<data[i1].size()<<endl;
            // displayData(data[i1]);
            // cout<<endl<<"data is: "<<data[i1][11]<<endl;
            // Assuming numberOfNeurons = data[i1].size() - 1, as last column would be output
            values = activationValues = initializeValues(numberOfLayers, numberOfNeurons);

            if (numberOfNeurons != (data[i1].size() - 1)) {
                cout<<"Critical Error"<<endl;
                cout<<"Number of neurons are: "<<numberOfNeurons<<" while data row size is: "<<data[i1].size()<<endl;
                return;
            }

            // Assigning input values to the values array 0th column.
            for (j = 0; j < numberOfNeurons; j++) {
                activationValues[0][j] = values[0][j] = data[i1][j];
            }

            for (j = 1; j < numberOfLayers; j++) {
                //k1 -> loops through neurons of current layer
                for (k1 = 0; k1 < numberOfNeurons; k1++) {
                    //k2 -> loops through neurons of previous layer to calcualte values of current layer
                    for (k2 = 0; k2 < numberOfNeurons; k2++) {
                        values[j][k1] += ((activationValues[j - 1][k2] * weights[j - 1][k2][k1]) + biases[j - 1][k1]);
                    }
                    activationValues[j][k1] = activationFunction(values[j][k1]);
                }
            }

            int n = numberOfLayers;
            for (j = 0; j < numberOfNeurons; j++) {
                activationValues[n][0] += (activationValues[n - 1][j] * weights[n - 1][j][0]);
            }

            double errorValue = (activationValues[n][0] - data[i1][11]) * (activationValues[n][0] - data[i1][11]);
            cout<<"Calculated output is: "<<fixed<<setprecision(6)<<activationValues[n][0]<<" Expected output is: "<<data[i1][11]<<" Error is: "<<errorValue<<endl;

            // Calculate Differentiattion constants
            vector<vector<double>> diffConstants = activationValues;
            diffConstants[n][0] = TWO * (activationValues[n][0] - data[i1][11]);

            for (i = n - 1; i >= 0; i--) {
                for (j = 0; j < numberOfNeurons; j++) {
                    diffConstants[i][j] = 0;
                    for (k1 = 0; k1 < diffConstants[i + 1].size(); k1++) {
                        diffConstants[i][j] += (weights[i][j][k1] * diffConstants[i + 1][k1]);
                    }
                    diffConstants[i][j] *= (activationValues[i][j] * (ONE - activationValues[i][j]));
                }
            }

            // Calculate differentiation values w.r.t weights
            vector<vector<vector<double>>> diffValues = weights;
            for (i = n - 1; i >= 0; i--) {
                for (j = 0; j < numberOfNeurons; j++) {
                    for (k1 = 0; k1 < weights[i][j].size(); k1++) {
                        diffValues[i][j][k1] = activationValues[i][j] * diffConstants[i + 1][k1];
                    }
                }
            }

            //updating the weights
            vector<vector<vector<double>>> newWeights = weights;
            for (i = 0; i < n; i++) {
                for (j = 0; j < weights[i].size(); j++) {
                    for (k1 = 0; k1 < weights[i][j].size(); k1++) {
                        newWeights[i][j][k1] = weights[i][j][k1] - (learningRate * diffValues[i][j][k1]);
                    }
                }
            }

            //updating biases
            vector<vector<double>> newBiases = biases;
            for (i = 0; i < biases.size(); i++) {
                for (j = 0; j < biases[i].size(); j++) {
                    newBiases[i][j] = biases[i][j] - (learningRate * diffConstants[i][j]);
                }
            }

            biases = newBiases;
            weights = newWeights;
        }
    }
}

int main() {
    string filename = "Housing1.csv";

    cout<<"Running time: 9"<<endl;
    cout << "Loading dataset from " << filename << "...\n";
    vector<vector<double>> data = readCSV(filename, true);
    vector<vector<double>> processedData = processData(data);
    cout << ">>>>>>> Processed records: " << data.size() << endl;
    // cout<<"Printing processed data "<<endl;
    // displayData(processedData);
    // cout<<endl<<endl;
    cout<<">>>>>>>>>>>>> 2. Processing Neural Network"<<endl;
    neuralNetwork(processedData);
}
