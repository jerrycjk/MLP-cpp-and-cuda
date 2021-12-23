#include "Model.h"

Model::Model(int *layer_dims, int layer_num, int batch_size, float lr)
:layer_num(layer_num), batch_size(batch_size)
{
    layers = new Dense*[layer_num] ;
    for (int i=0; i<layer_num-1; i++) {
        layers[i] = new Dense(batch_size, layer_dims[i], layer_dims[i+1], 0, lr) ;
    }
    layers[layer_num-1] = new Dense(batch_size, layer_dims[layer_num-1], layer_dims[layer_num], 1, lr) ;
}

Model::~Model()
{
    for (int i=0; i<layer_num; i++) {
        delete layers[i] ;
    }
    delete [] layers ;
}

void Model::Train(const float *input, const float *ans) {
    const float *X = input ;
    const float *dA = ans ;

    // forward pass
    for (int i=0; i<layer_num; i++) {
        layers[i]->Forward(X) ;

        X = layers[i]->Get_A() ;
    }

    // backprop
    for (int i=layer_num-1; i>=0; i--) {
        layers[i]->Backprop(dA) ;

        dA = layers[i]->Get_dAPrev() ;
    }

    // update
    for (int i=0; i<layer_num; i++) {
        layers[i]->Update_params() ;
    }
}

const float* Model::Predict(const float *input) {
    const float *X = input ;
    for (int i=0; i<layer_num; i++) {
        layers[i]->Forward(X) ;

        X = layers[i]->Get_A() ;
    }

    return X ;
}

const float Model::Evaluate(const float *input, const float *ans, int batch_num) {
    int acc=0 ;
    const float *pred ;
    float max_pred, max_ans ;
    int idx_pred, idx_ans ;
    int num_class = layers[layer_num-1]->Get_out_num() ;

    for (int i=0; i<batch_num; i++) {
        pred = Predict(&input[i*batch_size*28*28]) ;

        for (int j=0; j<batch_size; j++) {
            idx_pred = idx_ans = 0 ;
            max_pred = max_ans = 0.0 ;
            for (int k=0; k<num_class; k++) {
                if (pred[j*num_class+k] > max_pred) {
                    max_pred = pred[j*num_class+k] ;
                    idx_pred = k ;
                }
                if (ans[(i*batch_size+j)*num_class+k] > max_ans) {
                    max_ans = ans[(i*batch_size+j)*num_class+k] ;
                    idx_ans = k ;
                }
            }
            if (idx_pred == idx_ans) {
                acc += 1 ;
            }
        }
    }

    return (float)acc/(batch_num*batch_size) ;
}

void Model::Debug_msg() {
    for (int i=0; i<layer_num; i++) {
        std::cout << "Layer " << i << "=========================================================" << std::endl ;
        layers[i]->PrintAllInstance() ;
    }
}