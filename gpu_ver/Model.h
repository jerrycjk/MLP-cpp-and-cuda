#include "Dense.h"
#include <iostream>

class Model
{
private:
    Dense **layers ; // array of pointers to layers

    int layer_num ; // record number of layer, eg: [784, 128, 10], layer_num = 2
    int batch_size ;

public:
    Model(int *layer_dims, int layer_num, int batch_size, float lr) ;
    ~Model();

    // do training steps: forward pass, backprop, update
    void Train(const float *input, const float *ans) ;

    // get predict result
    const float* Predict(const float *input) ;

    // evaluate preformance
    float Evaluate(const float *input, const float *ans, int batch_num) ;

    // debug
    void Debug_msg() ;
};