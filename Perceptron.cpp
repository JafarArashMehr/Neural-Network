#include <random>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;


vector<vector<double>> input_set = {{ 0,1,0 },{ 0,0,1 },{ 1,0,0 },{ 1,1,0 },{ 1,1,1 },{ 0,1,1 },{ 0,1,0 }};

vector<double> y_values = {1,0,0,1,1,0,1};

int number_of_data = 7;
int dimensions = 3;

random_device rd;
default_random_engine eng(rd());
uniform_real_distribution<> distr{0, 1};

vector<double> weights;

void W(){

    for (int i = 0 ; i < dimensions; i++){

    weights.push_back(distr(eng));
}
}

double sigmoid (double x ){

      return  1/(1+exp(-x));
}

vector<double> sigmoid_values;

double derivative_sigmoid (double x ){

      return  sigmoid(x)*(1-sigmoid(x));
}

vector<double> derivative_sigmoid_values;


double bias = distr(eng);
double lr = 0.05;
double gamma;

double predict (vector<double> weight_values,vector<double> inputs_values, double bias){

double result;
   for (int p = 0 ; p < number_of_data; p++){
      result += weight_values[p] * inputs_values[p];
   }
   return sigmoid(result + bias);
}

////////////////////////////////////////////////////
vector<vector<double>> matrix_x;

vector<double> matrix_y_real;

vector<double> matrix_y_predict;

vector<double> derivatives_matrix_y_predict;
////////////////////////////////////////////////////

int training_iteration = 50000;

int main(){

    W();

    for (int i = 0 ; i < number_of_data ; i++){

      matrix_x.push_back(vector<double>());
   
         for (int j = 0 ; j < dimensions ; j++){

            matrix_x[i].push_back(input_set[i][j]);
   }
}

for (int k = 0 ; k < number_of_data ; k++){
      matrix_y_real.push_back(y_values[k]);
      matrix_y_predict.push_back(0);
      derivatives_matrix_y_predict.push_back(0);
}

for (int epoch = 0 ; epoch < training_iteration ;  epoch++){


   for (int m = 0 ; m < number_of_data ; m++){

   double result = 0;
   
      for (int n = 0 ; n < dimensions ; n++){

        result += weights[n] * matrix_x[m][n];
      }
        result += bias;
        matrix_y_predict[m] = sigmoid(result) ;
        derivatives_matrix_y_predict[m] = derivative_sigmoid(result);
   }

/////////////////////////////////////////////////////////////////////////
   for (int p = 0 ; p < dimensions; p++){
      for (int q = 0 ; q < number_of_data ; q++){
            gamma +=  derivatives_matrix_y_predict[q]  * (matrix_y_predict[q] - matrix_y_real[q]) * matrix_x[q][p];
      }
      weights [p] = weights [p] -  (2./number_of_data) * lr * gamma;
      gamma = 0;
}
    for (int b = 0 ; b < number_of_data ; b++){
        bias = bias - lr * derivatives_matrix_y_predict[b] * (matrix_y_predict[b] - matrix_y_real[b]);
    }
}
///////////////////   TEST   /////////////////////////
vector<double> new_values_to_predict = {0,0,1};

double prediction = predict(weights , new_values_to_predict , bias);
cout << prediction << endl;

return 0;
}