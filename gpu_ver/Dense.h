#include <iostream>
#include <cmath>
#include <random>

class Dense
{
private:
    int activation_type ; // 0 for relu, 1 for softmax
    int in_num, out_num ; // in_num: input node number, out_num: output node number
    int batch_size ;
    float *weights ; // #row = out_num, #column = in_num
    float *bias ; // #size = out_num
    float lr ; // learning rate

    float *Z ; // storage use to save the result of WX+b, #row = batch size, #column = out_num
    float *A ; // storage use to save the output of activation(Z), #row = batch size, #column = out_num
    float *A_prev ; // storage use to save input from previous layer(X), #row = batch_size, #column = in_num

    float *dW ; // storage use to save "dLoss/dWeight", shape = weights
    float *db ; // storage use to save "dLoss/db", shape = bias
    float *dA_prev ; // storage use to save "dLoss/dA_prev", shape = A_prev
    float *dZ ; // storage use to save "dLoss/dZ", shape = Z

    // idx: #row to cal
    // void cal_activation(int idx) ;

    void activation_backprop(const float *dA) ;
public:
    Dense(int batch_size, int in_num, int out_num, int activation_type, float learning_rate = 0.001);
    ~Dense();

    // input: #batch_size * #dimension(in_num)
    void Forward(const float *input) ; 
    
    // if this is last layer, arg is Y, if not, is dA. 
    // dA: #batch_size * #out_num 
    void Backprop(const float *dA) ;

    //
    void Update_params() ;

    // for debug
    void PrintAllInstance() ;    

    // return float *A 
    const float* Get_A() ;

    // return float *dA_prev 
    const float* Get_dAPrev() ;

    // return out_num
    int Get_out_num() ;
};