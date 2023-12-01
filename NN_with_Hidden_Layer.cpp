#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;


vector<vector<double>> input_data = {{ 0,0 },{ 0,1 },{ 1,0 },{ 1,1 }};
vector<double> target_values = {0,1,1,0};

int number_of_data = 4;
int dimensions = 2;
int hidden_layer_nodes = 3;
// Defining learning rate
double lr = 1;

random_device rd;
default_random_engine eng(rd());
uniform_real_distribution<> distr{0, 1};


int training_iteration = 5000;

///////////////// Required Vectors/Matrices for Feed Forward /////////////////

vector<vector<double>> weights_1;
vector<double> weights_2;

vector<vector<double>> matrix_x;

vector<double> bias_1 ;
vector<double> bias_2 ;

vector<vector<double>> hidden_values;
vector<vector<double>> hidden_values_activated;
vector<vector<double>> hidden_values_activated_derivative;

vector<double> output_values;
vector<double> output_values_activated;

///////////////// Required Vectors/Matrices for Back Propagation /////////////////

vector<double> front_output_activated;
vector<double> behind_output_activated;

vector<vector<double>> Gradient_wrt_W_2;
vector<double> Calculate_Grad_W2_Vector;
vector<double> Calculate_Grad_Bias1_Vector;
vector<vector<double>> BP_behind_hidden_activated;

vector<vector<double>> Gradient_wrt_W_1;
vector<vector<double>> BP_front_hidden_activated;
vector<vector<double>> Calculate_Grad_W1_Matrix;

////////////////////////////////////////////////////////////////////

void Vectors_Matrices_Builder(){

    for (int i = 0 ; i < hidden_layer_nodes ; i++){
      weights_2.push_back(distr(eng));
      weights_1.push_back(vector<double>());
        for (int j  = 0 ; j < dimensions ; j++){
            weights_1[i].push_back(distr(eng));
        }
    }

    for (int i = 0 ; i < hidden_layer_nodes ; i++){

      bias_1.push_back(0);
      Calculate_Grad_Bias1_Vector.push_back(0); 
    }

    bias_2.push_back(0);

    for (int i = 0 ; i < number_of_data ; i++){

      output_values.push_back(0); 
      output_values_activated.push_back(0);

      front_output_activated.push_back(0); 
      behind_output_activated.push_back(0); 
    }

    for (int i = 0 ; i < hidden_layer_nodes ; i++){

      hidden_values.push_back(vector<double>());
      hidden_values_activated.push_back(vector<double>());
      hidden_values_activated_derivative.push_back(vector<double>());

      BP_front_hidden_activated.push_back(vector<double>());
      BP_behind_hidden_activated.push_back(vector<double>());

        for (int j = 0 ; j < number_of_data ; j++){
            hidden_values[i].push_back(0);
            hidden_values_activated[i].push_back(0);
            BP_front_hidden_activated[i].push_back(0);
            hidden_values_activated_derivative[i].push_back(0);
            BP_behind_hidden_activated[i].push_back(0);
      }
    }

    for (int i = 0 ; i < number_of_data ; i++){

      Gradient_wrt_W_2.push_back(vector<double>());

        for (int j = 0 ; j < hidden_layer_nodes ; j++){
            Gradient_wrt_W_2[i].push_back(0);
      }
    }

    for (int j = 0 ; j < hidden_layer_nodes ; j++){
            Calculate_Grad_W2_Vector.push_back(0);
      }


    for (int i = 0 ; i < number_of_data ; i++){

      Gradient_wrt_W_1.push_back(vector<double>());

        for (int j = 0 ; j < dimensions ; j++){
            Gradient_wrt_W_1[i].push_back(0);
      }
    }

   for (int i = 0 ; i < hidden_layer_nodes ; i++){

      Calculate_Grad_W1_Matrix.push_back(vector<double>());

        for (int j = 0 ; j < dimensions ; j++){
            Calculate_Grad_W1_Matrix[i].push_back(0);
      }
    }
}
/////////////////////// Defines required function including sigmoid, derivative of the sigmoid and prediction ///////////////////////////////////////////////

class Required_Functions{
    public:

    double sigmoid (double x ){
          return  1/(1+exp(-x));
    }

    double derivative_sigmoid (double x ){
          return  sigmoid(x)*(1-sigmoid(x));
    }

  double predict (vector<double> Input_to_Predict , vector<vector<double>> W1,vector<double> W2, vector<double> bias1 , double bias2){

  vector<double> W1X;
  vector<double> W1X_activated;

   for (int m = 0 ; m < hidden_layer_nodes ; m++){

   double result_hidden_layer = 0;
   
      for (int n = 0 ; n < 1 ; n++){

        for (int k = 0 ; k < dimensions ; k++){

        result_hidden_layer += W1[m][k] * Input_to_Predict[k];  // 3x1 Output
      }
        //derivatives_matrix_y_predict[m] = derivative_sigmoid(result);

        result_hidden_layer += bias1[m];
                
        W1X.push_back(result_hidden_layer) ;
        W1X_activated.push_back(sigmoid(result_hidden_layer)) ;
        }
   }

    double result_output = 0;
    for (int m = 0 ; m < hidden_layer_nodes ; m++){
        result_output += W2[m] * W1X_activated[m];  
      }
      result_output += bias2;

    return sigmoid(result_output);
}
};

Required_Functions Start; 
Required_Functions *Function = &Start;

///////////////////////////////////////////////////
vector<double> Training_Number;
vector<double> Error_Values;
////////////////////////////////////////////////////

int main(){

Vectors_Matrices_Builder();

    // Forming the training set
    for (int i = 0 ; i < dimensions ; i++){

      matrix_x.push_back(vector<double>());
   
         for (int j = 0 ; j < number_of_data ; j++){

            matrix_x[i].push_back(input_data[j][i]);  // dimension x number_of_data (2 x 4)
   }
}

/////////////////////////// Training Begins! ////////////////////////////////////

for (int epoch = 0 ; epoch < training_iteration ;  epoch++){

/////////////////////////// Feed Forwards //////////////////////////

 /////////////////////////// W_1 * X //////////////////////////////

   for (int m = 0 ; m < hidden_layer_nodes ; m++){

   double result_hidden_layer = 0;
   
      for (int n = 0 ; n < number_of_data ; n++){

        for (int k = 0 ; k < dimensions ; k++){

        result_hidden_layer += weights_1[m][k] * matrix_x[k][n];
      }

        result_hidden_layer += bias_1[m];
                
        hidden_values[m][n]= result_hidden_layer ;
        hidden_values_activated[m][n]= Function->sigmoid(result_hidden_layer) ;
        result_hidden_layer = 0;    
        
        }
   }
   
 /////////////////////////// W_2 * SIGMOID (W_1 * X) ////////////////////////////////////

  for (int r = 0 ; r < number_of_data ; r++){

   double result_output_layer = 0;
   
      for (int p = 0 ; p < weights_2.size() ; p++){

        result_output_layer += weights_2[p] * hidden_values_activated[p][r];
      }

        result_output_layer += bias_2[0];
                
        output_values[r] = result_output_layer ;
        output_values_activated[r] = Function->sigmoid(result_output_layer) ;
        
        }

/////////////////////////// Error Calculation //////////////////////////

double error = 0;
    for (int k = 0 ; k < number_of_data ; k++){
        
        error +=  pow((target_values[k] - output_values_activated[k]), 2 );
    }
     error  = error / (2. * number_of_data);

cout << "In Training Number: " << epoch+1 << " ->" << " The Error is: " << error << endl;

Error_Values.push_back(error);
Training_Number.push_back(epoch+1);

/////////////////////////// Back Propagation //////////////////////////

    for (int t = 0 ; t < number_of_data ; t++){
        front_output_activated[t] = (-1. / number_of_data) * (target_values[t] - output_values_activated[t]);
    }

    for (int w = 0 ; w < number_of_data ; w++){
        behind_output_activated[w] = front_output_activated[w] * Function->derivative_sigmoid(output_values_activated[w]);
    }

 /////////////////////////// Form Grad W2 ////////////////////////////////////

 for (int r = 0 ; r < number_of_data ; r++){
   
      for (int p = 0 ; p < hidden_layer_nodes ; p++){

        Gradient_wrt_W_2[r][p] = hidden_values_activated[p][r];
      }
 }
 /////////////////////////// Calculation Grad(W2)) ////////////////////////////////////

  for (int r = 0 ; r < hidden_layer_nodes ; r++){

   double result_1 = 0;
   
      for (int p = 0 ; p < number_of_data ; p++){

        result_1 += behind_output_activated[p] * Gradient_wrt_W_2[p][r];
      }

        Calculate_Grad_W2_Vector[r] = result_1 ;
        }
 /////////////////////////// Calculation Grad(bias2)) ////////////////////////////////////

 double new_bias2 = 0;

 for (int r = 0 ; r < number_of_data ; r++){
      new_bias2 += behind_output_activated[r];
    }

/////////////////////////// Moving in front of hidden_activated ///////////////////////////

/////////////////////////// W2 * behind_output_activated ////////////////////////////////////

  for (int r = 0 ; r < hidden_layer_nodes ; r++){
   
      for (int p = 0 ; p < number_of_data ; p++){

        BP_front_hidden_activated[r][p] = weights_2[r] * behind_output_activated[p];
      }
  }

/////////////////////////// Derivative of hidden values activated ////////////////////////////////////

  for (int r = 0 ; r < hidden_layer_nodes ; r++){
   
      for (int p = 0 ; p < number_of_data ; p++){

        hidden_values_activated_derivative[r][p] =  Function->derivative_sigmoid(hidden_values[r][p]);
      }
  }
/////////////////////////// Moving behind hidden_activated ///////////////////////////

///////////////// (W2 * behind_output_activated) * (Derivative of hidden values activated) ////////////////////
  
  for (int r = 0 ; r < hidden_layer_nodes ; r++){
   
      for (int p = 0 ; p < number_of_data ; p++){

        BP_behind_hidden_activated[r][p] =  BP_front_hidden_activated[r][p] * hidden_values_activated_derivative[r][p];
      }

  }

////////////////////////// Form Grad W1 ////////////////////////////////////

  for (int r = 0 ; r < dimensions ; r++){
   
      for (int p = 0 ; p < number_of_data ; p++){

        Gradient_wrt_W_1[p][r] = matrix_x[r][p];
      }
  }

/////////////////////////// Calculation Grad(W1)) ////////////////////////////////////

   for (int m = 0 ; m < hidden_layer_nodes ; m++){

   double result_4 = 0;
   
      for (int n = 0 ; n < dimensions ; n++){

        for (int k = 0 ; k < number_of_data ; k++){

        result_4 += BP_behind_hidden_activated[m][k] * Gradient_wrt_W_1[k][n];
      }

        Calculate_Grad_W1_Matrix[m][n]= result_4 ;
        result_4 = 0;    
        
        }
   }
 /////////////////////////// Calculation Grad(bias1)) ////////////////////////////////////

    for (int i = 0 ; i < hidden_layer_nodes ; i++){

      double new_bias1 = 0;

        for (int j = 0 ; j < number_of_data ; j++){

          new_bias1 += BP_behind_hidden_activated[i][j];
        }

      Calculate_Grad_Bias1_Vector[i] = new_bias1;
    }

////////////////////////// Gradient Descent ////////////////////////////////////
////////////////////////// UPDATE W2 ////////////////////////////////////

for (int i = 0 ; i < hidden_layer_nodes ; i++){

  weights_2[i] = weights_2[i] - lr * Calculate_Grad_W2_Vector[i];
}
////////////////////////// UPDATE Bias2 ////////////////////////////////////

bias_2[0] = bias_2[0] - lr * new_bias2;

////////////////////////// UPDATE W1 ////////////////////////////////////

for (int i = 0 ; i < hidden_layer_nodes ; i++){

  for (int j = 0 ; j < dimensions ; j++){

  weights_1[i][j] = weights_1[i][j] - lr * Calculate_Grad_W1_Matrix[i][j];

  }
}
////////////////////////// UPDATE Bias1 ////////////////////////////////////

for (int i = 0 ; i < hidden_layer_nodes ; i++){

  bias_1[i] = bias_1[i] - lr * Calculate_Grad_Bias1_Vector[i]; 
}
}

////////////////////////// Saving in a text file ////////////////////////////////////

    string filename("Results-CPP.txt");
    fstream file_out;
    file_out.open(filename, std::ios::out);

    for (int r=0 ; r < training_iteration; r++){

        file_out << Training_Number[r] << "       " << Error_Values[r] << endl;     
    };

    file_out.close();
////////////////////////// Prediction ////////////////////////////////////

vector<double> Predict_value = { 1,1 };
double prediction = Function->predict(Predict_value, weights_1 , weights_2 , bias_1 , bias_2[0]);

cout << endl ;
cout << "The predicted Value is: " << prediction << endl;

        return 0 ;
   }