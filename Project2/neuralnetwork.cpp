#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

typedef float (*func)(float);
typedef float (*func2)(float, float);

float myTanh(float x)
{
  return tanh(x);
}

float derivMyTanh(float x)
{
  float t = tanh(x);
  return 1 + t*t;
}

float sig(float x)
{
  return 1/(1 + exp(-x));
}

float derivSig(float x)
{
  float e = exp(-x);
  return e/((1 + e)*(1 + e));
}

float crossEntropy(float y_pred, float y)
{
  return (y_pred - y) / (y_pred * (1 - y_pred));
}


class NN
{

public:

  vector<int> dim;
  vector<func> acf;
  vector<func> dacf;

  int length;

  float*** weights;
  float** bias;

  float** z;
  float** a;

  float** grad;
  float*** delta;
  func2 cost;

  NN(vector<int> _dim, vector<func> _acf, vector<func> _dacf, func2 _cost)
  {
    dim = _dim;
    acf = _acf;
    dacf = _dacf;
    cost = _cost;
    length = dim.size()-1;
    default_random_engine generator;
    normal_distribution<float> distribution(0, 1);

    weights = new float**[length];
    bias = new float*[length];
    z = new float*[length];
    a = new float*[length];
    grad = new float*[length];
    delta = new float**[length];

    for(int i = 0; i<length; i++)
    {
      int m = dim[i+1];
      int n = dim[i];
      weights[i] = new float*[m];
      delta[i] = new float*[m];
      bias[i] = new float[m];
      z[i] = new float[m];
      a[i] = new float[m];
      grad[i] = new float[m];

      for(int j = 0; j<m; j++)
      {
        bias[i][j] = 0.01;
        z[i][j] =  a[i][j] = 0;

        weights[i][j] = new float[n];
        delta[i][j] = new float[n];
        for(int k = 0; k<n; k++)
        {
          weights[i][j][k] = distribution(generator);
        }
      }
    }
  }

  void forward(float* x)
  {
    int m = dim[1];
    int n = dim[0];
    for(int i=0; i<m; i++)
    {
      z[0][i] = bias[0][i];
      for(int j=0; j<n; j++)
      {
        z[0][i] += weights[0][i][j]*x[j];
      }
    }

    for(int i=1; i<length; i++)
    {
      int m = dim[i+1];
      int n = dim[i];

      for(int j=0; j<m; j++)
      {
        z[i][j] = bias[i][j];
        a[i][j] = 0;
        for(int k=0; k<n; k++)
        {
          z[i][j] += weights[i][j][k]*a[i-1][k];
        }
        a[i][j] = acf[i-1](z[i][j]);
      }
    }
  }

  void backward(float* x, float* y)
  {
    forward(x);
    int m;
    int n = dim[length];

    for(int i=0; i<n; i++)
    {
      grad[length-1][i] = dacf[length-1](z[length-1][i])*cost(a[length-1][i], *y);
    }

    for(int i=length-1; i>0; i--)
    {
      m = dim[i+1];
      n = dim[i];

      for(int j=0; j<n; j++)
      {
        grad[i-1][j] = 0;
        for(int k=0; k<m; k++)
        {
          grad[i-1][j] += weights[i][k][j]*grad[i][k];
        }
        grad[i-1][j] *= dacf[i-1](z[length-1][j]);
      }
    }
  }

    void train(float** x, float** y, int n)
    {
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<dim.size()-1; j++)
            {
                int m = dim[j+1];
                int n = dim[j];
                for(int k=0; k<m; k++)
                {
                    for(int l=0; l<n; l++)
                    {
                        delta[j][k][l] = 0;
                    }
                }
            }
        }

        for(int i=0; i<n; i++)
        {
            backward(x[0], y[0]);
            for(int j=0; j<dim.size()-1; j++)
            {
                int m = dim[j+1];
                int n = dim[j];
                for(int k=0; k<m; k++)
                {
                    for(int l=0; l<n; l++)
                    {
                        delta[j][k][l] += grad[j][k]*a[j][l];
                    }
                }
            }
        }

        for(int j=0; j<dim.size()-1; j++)
        {
            int m = dim[j+1];
            int n = dim[j];
            for(int k=0; k<m; k++)
            {
                for(int l=0; l<n; l++)
                {
                    weights[j][k][l] -= 0.001*delta[j][k][l];
                }
                bias[j][k] -= 0.001*grad[j][k];
            }
        }
    }
};

int main()
{
    vector<int> dim{2, 4, 1};
    vector<func> acf{myTanh, sig};
    vector<func> dacf{derivMyTanh, derivSig};

    NN nn(dim, acf, dacf, crossEntropy);


    float** x = new float*[1];
    float** y = new float*[1];
    x[0] = new float[2];
    y[0] = new float[1];
    x[0][0] = 1; x[0][1] = 2;
    y[0][0] = 2;

    int n = 2000;
    nn.forward(x[0]);
    cout << nn.a[1][0] << endl;
    for(int i=0; i<n; i++)
    {
        nn.train(x, y, 1);
    }

    nn.forward(x[0]);
    cout << nn.a[1][0] << endl;


return 0;
}
