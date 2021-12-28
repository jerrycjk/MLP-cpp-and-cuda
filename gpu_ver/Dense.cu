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
    std::normal_distribution<float> dis ;

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

// ==================================      gpu codes      =========================================

__global__ void cal_foward(int in_num, float *input_d, float *w_d, float *b_d, float *z_d) {
    int i_th_input = blockIdx.x ;
    int j_th_out = threadIdx.x ;
    int out_num = blockDim.x ;

    float z = 0 ;
    for (int k=0; k<in_num; k++) {
        z += input_d[i_th_input*in_num+k] * w_d[j_th_out*in_num+k] + b_d[j_th_out] ;
    }
    z_d[i_th_input*out_num+j_th_out] = z ;
}

__global__ void cal_relu_activation(float *z_d, float *a_d) {
    int idx = blockIdx.x ;
    int i = threadIdx.x ;
    int out_num = blockDim.x ;
    float temp = z_d[idx*out_num+i] ;
    if (temp <= 0.0) {
        temp = 0.0 ;
    }
    a_d[idx*out_num+i] = temp ;
}

__global__ void cal_softmax_activation(int out_num, float *z_d, float *a_d) {
    int idx = threadIdx.x ;
    float max = 0 ;
    float temp ;
    float sum = 0 ;
    // find max to prevent overflow
    for (int i=0; i<out_num; i++) {
        if (z_d[idx*out_num+i] > max) {
            max = z_d[idx*out_num+i] ;
        }
    }
    // sum = sum(e^{i-b})
    for (int i=0; i<out_num; i++) {
        temp = expf(z_d[idx*out_num+i] - max) ;
        sum += temp ;
        a_d[idx*out_num+i] = temp ;
    }
    // softmax(v) = e^{v_i - max} / sum
    for (int i=0; i<out_num; i++) {
        a_d[idx*out_num+i] = a_d[idx*out_num+i] / sum ;
    }
}

__global__ void cal_relu_backprop(float *z_d, float *da_d, float *dz_d) {
    int i = blockIdx.x ;
    int j = threadIdx.x ;
    int out_num = blockDim.x ;
    
    if (z_d[i*out_num+j] <= 0) dz_d[i*out_num+j] = 0.0 ;
    else dz_d[i*out_num+j] = da_d[i*out_num+j] ;
}

__global__ void cal_softmax_backprop(float *a_d, float *da_d, float *dz_d) {
    int i = blockIdx.x ;
    int j = threadIdx.x ;
    int out_num = blockDim.x ;
    
    dz_d[i*out_num+j] = a_d[i*out_num+j] - da_d[i*out_num+j] ;
}

__global__ void cal_dw_backprop(int batch_size, float *dz_d, float *a_prev_d, float *dw_d) {
    int i = blockIdx.x ;
    int j = threadIdx.x ;
    int out_num = gridDim.x ;
    int in_num = blockDim.x ;

    float cal = 0 ;
    
    for (int k=0; k<batch_size; k++) {
        cal += dz_d[k*out_num+i] * a_prev_d[k*in_num+j] ;
    }
    dw_d[i*in_num+j] = cal / batch_size ;
}

__global__ void cal_db_backprop(int batch_size, float *dz_d, float *db_d) {
    int i = threadIdx.x ;
    int out_num = blockDim.x ;

    float cal = 0 ;
    
    for (int j=0; j<batch_size; j++) {
        cal += dz_d[j*out_num+i] ;
    }
    db_d[i] = cal / batch_size ;
}

__global__ void cal_da_prev_backprop(int out_num, float *dz_d, float *w_d, float *da_prev_d) {
    int i = blockIdx.x ;
    int j = threadIdx.x ;
    int in_num = blockDim.x ;

    float cal = 0 ;
    
    for (int k=0; k<out_num; k++) {
        cal += dz_d[i*out_num+k] * w_d[k*in_num+j] ;
    }
    da_prev_d[i*in_num+j] = cal ;
}

__global__ void update_weight(float lr, float *dw_d, float *w_d) {
    int i = blockIdx.x ;
    int j = threadIdx.x ;
    int in_num = blockDim.x ;

    w_d[i*in_num+j] -= lr*dw_d[i*in_num+j] ;
}

__global__ void update_bias(float lr, float *db_d, float *b_d) {
    int i = threadIdx.x ;
    
    b_d[i] -= lr*db_d[i] ;
}

// ================================================================================================

void Dense::Forward(const float *input) {
    // calculate forward pass
    float *input_d, *w_d, *b_d, *z_d ;
    float *a_d ;
    // allocate device space for 'input' and copy the content on host to it
    cudaMalloc(&input_d, batch_size*in_num*sizeof(float)) ;
    cudaMemcpy(input_d, input, batch_size*in_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // same for weight
    cudaMalloc(&w_d, out_num*in_num*sizeof(float)) ;
    cudaMemcpy(w_d, weights, out_num*in_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // same for bias
    cudaMalloc(&b_d, out_num*sizeof(float)) ;
    cudaMemcpy(b_d, bias, out_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // only allocate device space for Z
    cudaMalloc(&z_d, batch_size*out_num*sizeof(float)) ;
    // only allocate device space for A
    cudaMalloc(&a_d, batch_size*out_num*sizeof(float)) ;

    // forward pass, need: input, weight, bias, output: Z (all on device)
    cal_foward <<<batch_size, out_num>>> (in_num, input_d, w_d, b_d, z_d) ;

    // save content of Z
    cudaMemcpy(Z, z_d, batch_size*out_num*sizeof(float), cudaMemcpyDeviceToHost) ;

    // activate, need: Z, output: A (on device)
    if (activation_type == 0) {
        cal_relu_activation <<<batch_size, out_num>>> (z_d, a_d) ;
    }
    else {
        cal_softmax_activation <<<1, batch_size>>> (out_num, z_d, a_d) ;
    }

    // save content of A
    cudaMemcpy(A, a_d, batch_size*out_num*sizeof(float), cudaMemcpyDeviceToHost) ;

    // free
    cudaFree(input_d) ;
    cudaFree(w_d) ;
    cudaFree(b_d) ;
    cudaFree(z_d) ;
    cudaFree(a_d) ;

    // save input for calculate backprop
    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<in_num; j++) {
            A_prev[i*in_num+j] = input[i*in_num+j] ;
        }
    }
}

void Dense::Backprop(const float *dA) {
    // dL/dZ
    activation_backprop(dA) ;

    float *dz_d, *a_prev_d, *w_d ;
    float *dw_d, *db_d, *da_prev_d ;

    // allocate device space for 'dZ' and copy the content on host to it
    cudaMalloc(&dz_d, batch_size*out_num*sizeof(float)) ;
    cudaMemcpy(dz_d, dZ, batch_size*out_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // A_prev
    cudaMalloc(&a_prev_d, batch_size*in_num*sizeof(float)) ;
    cudaMemcpy(a_prev_d, A_prev, batch_size*in_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // weights
    cudaMalloc(&w_d, out_num*in_num*sizeof(float)) ;
    cudaMemcpy(w_d, weights, out_num*in_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // only allocate device space for dW
    cudaMalloc(&dw_d, out_num*in_num*sizeof(float)) ;
    // only allocate device space for db
    cudaMalloc(&db_d, out_num*sizeof(float)) ;
    // only allocate device space for dA_prev
    cudaMalloc(&da_prev_d, batch_size*in_num*sizeof(float)) ;

    // dL/dW = 1/m * dL/dZ * A_prev
    cal_dw_backprop <<<out_num, in_num>>> (batch_size, dz_d, a_prev_d, dw_d) ;
    // dL/db = 1/m * dZ
    cal_db_backprop <<<1, out_num>>> (batch_size, dz_d, db_d) ;
    // dA_prev = W * dZ
    cal_da_prev_backprop <<<batch_size, in_num>>> (out_num, dz_d, w_d, da_prev_d) ;

    // save content of dW
    cudaMemcpy(dW, dw_d, out_num*in_num*sizeof(float), cudaMemcpyDeviceToHost) ;
    // save content of db
    cudaMemcpy(db, db_d, out_num*sizeof(float), cudaMemcpyDeviceToHost) ;
    // save content of dA_prev
    cudaMemcpy(dA_prev, da_prev_d, batch_size*in_num*sizeof(float), cudaMemcpyDeviceToHost) ;

    // free
    cudaFree(dz_d) ;
    cudaFree(a_prev_d) ;
    cudaFree(w_d) ;
    cudaFree(dw_d) ;
    cudaFree(db_d) ;
    cudaFree(da_prev_d) ;
}

void Dense::Update_params() {
    float *dw_d, *db_d ;
    float *w_d, *b_d ;

    // allocate device space for 'dW' and copy the content on host to it
    cudaMalloc(&dw_d, out_num*in_num*sizeof(float)) ;
    cudaMemcpy(dw_d, dW, out_num*in_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // dB
    cudaMalloc(&db_d, out_num*sizeof(float)) ;
    cudaMemcpy(db_d, db, out_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // weight
    cudaMalloc(&w_d, out_num*in_num*sizeof(float)) ;
    cudaMemcpy(w_d, weights, out_num*in_num*sizeof(float), cudaMemcpyHostToDevice) ;
    // bias
    cudaMalloc(&b_d, out_num*sizeof(float)) ;
    cudaMemcpy(b_d, bias, out_num*sizeof(float), cudaMemcpyHostToDevice) ;

    update_weight <<<out_num, in_num>>> (lr, dw_d, w_d) ;
    update_bias <<<1, out_num>>> (lr, db_d, b_d) ;

    // save content of weights
    cudaMemcpy(weights, w_d, out_num*in_num*sizeof(float), cudaMemcpyDeviceToHost) ;
    // save content of bias
    cudaMemcpy(bias, b_d, out_num*sizeof(float), cudaMemcpyDeviceToHost) ;

    // free
    cudaFree(dw_d) ;
    cudaFree(db_d) ;
    cudaFree(w_d) ;
    cudaFree(b_d) ;
}

void Dense::activation_backprop(const float *dA) {
    // dL/dZ = dL/dA * dA/dZ
    if (activation_type == 0) { // relu
        float *z_d, *dz_d, *da_d ;

        // allocate device space for 'Z' and copy the content on host to it
        cudaMalloc(&z_d, batch_size*out_num*sizeof(float)) ;
        cudaMemcpy(z_d, Z, batch_size*out_num*sizeof(float), cudaMemcpyHostToDevice) ;
        // dA
        cudaMalloc(&da_d, batch_size*out_num*sizeof(float)) ;
        cudaMemcpy(da_d, dA, batch_size*out_num*sizeof(float), cudaMemcpyHostToDevice) ;
        // only allocate device space for dZ
        cudaMalloc(&dz_d, batch_size*out_num*sizeof(float)) ;

        cal_relu_backprop <<<batch_size, out_num>>> (z_d, da_d, dz_d) ;

        // save content of dZ
        cudaMemcpy(dZ, dz_d, batch_size*out_num*sizeof(float), cudaMemcpyDeviceToHost) ;

        // free
        cudaFree(z_d) ;
        cudaFree(dz_d) ;
        cudaFree(da_d) ;
    }
    else { // softmax, this is last layer
        float *a_d, *dz_d, *da_d ;

        // allocate device space for 'A' and copy the content on host to it
        cudaMalloc(&a_d, batch_size*out_num*sizeof(float)) ;
        cudaMemcpy(a_d, A, batch_size*out_num*sizeof(float), cudaMemcpyHostToDevice) ;
        // dA
        cudaMalloc(&da_d, batch_size*out_num*sizeof(float)) ;
        cudaMemcpy(da_d, dA, batch_size*out_num*sizeof(float), cudaMemcpyHostToDevice) ;
        // only allocate device space for dZ
        cudaMalloc(&dz_d, batch_size*out_num*sizeof(float)) ;

        cal_softmax_backprop <<<batch_size, out_num>>> (a_d, da_d, dz_d) ;

        // save content of dZ
        cudaMemcpy(dZ, dz_d, batch_size*out_num*sizeof(float), cudaMemcpyDeviceToHost) ;

        // free
        cudaFree(a_d) ;
        cudaFree(dz_d) ;
        cudaFree(da_d) ;
    }
}

// void Dense::cal_activation(int idx){
//     float temp = 0.0 ;
//     float max = 0.0 ;
//     float sum = 0.0 ;

//     if (activation_type == 0) { // relu
//         for (int i=0; i<out_num; i++) {
//             temp = Z[idx*out_num+i] ;
//             A[idx*out_num+i] = (temp > 0.0) ? temp : 0.0 ;
//         }
//     }
//     else { // softmax
//         // find max to prevent overflow
//         for (int i=0; i<out_num; i++) {
//             if (Z[idx*out_num+i] > max) {
//                 max = Z[idx*out_num+i] ;
//             }
//         }
//         // sum = sum(e^{i-b})
//         for (int i=0; i<out_num; i++) {
//             temp = exp(Z[idx*out_num+i] - max) ;
//             sum += temp ;
//             A[idx*out_num+i] = temp ;
//         }
//         // softmax(v) = e^{v_i - max} / sum
//         for (int i=0; i<out_num; i++) {
//             A[idx*out_num+i] = A[idx*out_num+i] / sum ;
//         }
//     }
// }

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
    std::cout << std::endl ;

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



    std::cout << std::endl << "dW: " << std::endl ;
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
    std::cout << std::endl ;

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

const float* Dense::Get_A() {
    return this->A ;
}

const float* Dense::Get_dAPrev() {
    return this->dA_prev ;
}

int Dense::Get_out_num() {
    return this->out_num ;
}