#include "Dense.h"

Dense::Dense(int batch_size, int in_num, int out_num, int activation_type, float learning_rate)
:batch_size(batch_size), in_num(in_num), out_num(out_num), activation_type(activation_type), lr(learning_rate)
{
    weights = new float[in_num*out_num] ;
    bias = new float[out_num]{0.0} ;

    Z = new float[batch_size*out_num]{0.0} ;
    A = new float[batch_size*out_num]{0.0} ;
    A_prev = new float[batch_size*in_num]{0.0} ;

    dW = new float[in_num*out_num]{0.0} ;
    db = new float[out_num]{0.0} ;
    dA_prev = new float[batch_size*in_num]{0.0} ;
    dZ = new float[batch_size*out_num]{0.0} ;

    std::random_device rd;
    std::mt19937 gen(rd()) ;
    std::normal_distribution<float> dis(0.0, 1.0);

    for (int i=0; i<out_num; i++) {
        for (int j=0; j<in_num; j++) {
            weights[i*in_num+j] = dis(gen) ;
            weights[i*in_num+j] *= sqrt(2.0/in_num) ;
        }
        bias[i] = 0. ;
    }
}

Dense::~Dense()
{
    delete [] weights ;
    delete [] bias ;

    delete [] Z ;
    delete [] A ;
    delete [] A_prev ;

    delete [] dW ;
    delete [] db ;
    delete [] dA_prev ;
    delete [] dZ ;
}

void Dense::Forward(float *input) {
    float net ;

    // calculate forward pass
    for (int i=0; i<batch_size; i++) { // i-th input
        for (int j=0; j<out_num; j++) { // j-th output node
            Z[i*out_num+j] = 0 ;
            for (int k=0; k<in_num; k++) {
                net = input[i*in_num+k] * weights[j*in_num+k] + bias[j]; // WX+b
                Z[i*out_num+j] += net ;
            }
        }
        cal_activation(i) ;
    }

    // save input for calculate backprop
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<in_num; j++) {
            A_prev[i*in_num+j] = input[i*in_num+j] ;
        }
    }
}

void Dense::Backprop(float *dA) {
    // dL/dZ
    activation_backprop(dA) ;

    // dL/dW = 1/m * dL/dZ * A_prev
    for (int i=0; i<out_num; i++) {
        for (int j=0; j<in_num; j++) {
            dW[i*in_num+j] = 0 ;
            for (int k=0; k<batch_size; k++) {
                dW[i*in_num+j] += dZ[k*out_num+i] * A_prev[k*in_num+j] ;
            }
            dW[i*in_num+j] /= batch_size ;
        }
    }

    // dL/db = 1/m * dZ
    for (int i=0; i<out_num; i++) {
        db[i] = 0;
        for (int j=0; j<batch_size; j++) {
            db[i] += dZ[j*out_num+i] ;
        }
        db[i] /= batch_size ;
    }

    // dA_prev = W * dZ
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<in_num; j++) {
            dA_prev[i*in_num+j] = 0 ;
            for (int k=0; k<out_num; k++) {
                dA_prev[i*in_num+j] += dZ[i*out_num+k] * weights[k*in_num+j] ;
            }
        }
    }
}

void Dense::activation_backprop(float *dA) {
    // dL/dZ = dL/dA * dA/dZ
    if (activation_type == 0) { // relu
        for (int i=0; i<batch_size; i++) {
            for (int j=0; j<out_num; j++) {
                if (Z[i*out_num+j] <= 0) dZ[i*out_num+j] = 0.0 ;
                else dZ[i*out_num+j] = dA[i*out_num+j] ;
            }
        }
    }
    else { // softmax, this is last layer
        for (int i=0; i<batch_size; i++) {
            for (int j=0; j<out_num; j++) {
                dZ[i*out_num+j] = A[i*out_num+j] - dA[i*out_num+j] ;
            }
        }
    }
}

void Dense::cal_activation(int idx){
    float temp = 0.0 ;
    float max = 0.0 ;
    float sum = 0.0 ;

    if (activation_type == 0) { // relu
        for (int i=0; i<out_num; i++) {
            temp = Z[idx*out_num+i] ;
            A[idx*out_num+i] = (temp > 0.0) ? temp : 0.0 ;
        }
    }
    else { // softmax
        // find max to prevent overflow
        for (int i=0; i<out_num; i++) {
            if (Z[idx*out_num+i] > max) {
                max = Z[idx*out_num+i] ;
            }
        }
        // sum = sum(e^{i-b})
        for (int i=0; i<out_num; i++) {
            temp = exp(Z[idx*out_num+i] - max) ;
            sum += temp ;
            A[idx*out_num+i] = temp ;
        }
        // softmax(v) = e^{v_i - max} / sum
        for (int i=0; i<out_num; i++) {
            A[idx*out_num+i] = A[idx*out_num+i] / sum ;
        }
    }
}

void Dense::PrintAllInstance() {
    std::cout << "weights: " << std::endl ;
    for (int i=0; i<out_num; i++) {
        for (int j=0; j<in_num; j++) {
            std::cout << weights[i*in_num+j] << " " ;
        }
        std::cout << std::endl ;
    }

    std::cout << std::endl << "bias: " << std::endl ;
    for (int i=0; i<out_num; i++) {
        std::cout << bias[i] << " " ;
    }

    std::cout << std::endl << "Z: " << std::endl ;
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<out_num; j++) {
            std::cout << Z[i*out_num+j] << " " ;
        }
        std::cout << std::endl ;
    }

    std::cout << std::endl << "A: " << std::endl ;
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<out_num; j++) {
            std::cout << A[i*out_num+j] << " " ;
        }
        std::cout << std::endl ;
    }

    std::cout << std::endl << "A_prev: " << std::endl ;
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<in_num; j++) {
            std::cout << A_prev[i*in_num+j] << " " ;
        }
        std::cout << std::endl ;
    }



    std::cout << "dW: " << std::endl ;
    for (int i=0; i<out_num; i++) {
        for (int j=0; j<in_num; j++) {
            std::cout << dW[i*in_num+j] << " " ;
        }
        std::cout << std::endl ;
    }

    std::cout << std::endl << "dB: " << std::endl ;
    for (int i=0; i<out_num; i++) {
        std::cout << db[i] << " " ;
    }

    std::cout << std::endl << "dZ: " << std::endl ;
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<out_num; j++) {
            std::cout << dZ[i*out_num+j] << " " ;
        }
        std::cout << std::endl ;
    }

    std::cout << std::endl << "dA_prev: " << std::endl ;
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<in_num; j++) {
            std::cout << dA_prev[i*in_num+j] << " " ;
        }
        std::cout << std::endl ;
    }

}