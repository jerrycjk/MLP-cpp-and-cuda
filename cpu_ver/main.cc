#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <iomanip>

#include "Model.h"

#define EPOCH 5

using namespace std ;

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist(string path, float *arr) {
    ifstream file(path.c_str(), ios::in|ios::binary) ;
    if (file.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number, sizeof(magic_number)) ;
        magic_number = reverseInt(magic_number) ;

        file.read((char*)&number_of_images, sizeof(number_of_images)) ;
        number_of_images = reverseInt(number_of_images) ;

        file.read((char*)&n_rows, sizeof(n_rows)) ;
        n_rows = reverseInt(n_rows) ;

        file.read((char*)&n_cols, sizeof(n_cols)) ;
        n_cols = reverseInt(n_cols) ;

        for (int i=0; i<number_of_images; i++) {
            for (int r=0; r<n_rows; r++) {
                for (int c=0; c<n_cols; c++) {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i*n_rows*n_cols+r*n_cols+c] = (float)temp ;
                    arr[i*n_rows*n_cols+r*n_cols+c] /= 255. ;
                }
            }
        }
    }
}

void read_label(string path, float *arr) {
    ifstream file(path.c_str(), ios::in|ios::binary) ;
    if (file.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number, sizeof(magic_number)) ;
        magic_number = reverseInt(magic_number) ;

        file.read((char*)&number_of_images, sizeof(number_of_images)) ;
        number_of_images = reverseInt(number_of_images) ;

        for (int i=0; i<number_of_images; i++) {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            int one_hot = (int)temp ;
            arr[i*10+one_hot] = 1 ;
        }
    }
}

int main(int argc, char *argv[]) {
    // get data
    chrono::steady_clock::time_point t_io_start = chrono::steady_clock::now();

    string x_train_path = "../mnist/train-images-idx3-ubyte" ;
    string y_train_path = "../mnist/train-labels-idx1-ubyte" ;
    string x_test_path = "../mnist/t10k-images-idx3-ubyte" ;
    string y_test_path = "../mnist/t10k-labels-idx1-ubyte" ;

    float *X_train = new float[60000*28*28]{0.0};
    float *y_train = new float[60000*10]{0.0};
    float *X_test = new float[10000*28*28]{0.0};
    float *y_test = new float[10000*10]{0.0};

    read_mnist(x_train_path, X_train) ;
    read_label(y_train_path, y_train) ;
    read_mnist(x_test_path, X_test) ;
    read_label(y_test_path, y_test) ;

    chrono::steady_clock::time_point t_io_end = chrono::steady_clock::now();
    cout << "Read mnist take " << chrono::duration_cast<chrono::milliseconds>(t_io_end - t_io_start).count() << " ms.\n";

    // cout << "Read: " << std::fixed << endl ;

    // for (int k=0; k<2; k++) {
    //     for (int i=0; i<28; i++) {
    //         for (int j=0; j<28; j++) {
    //             if (X_train[k*28*28+i*28+j] >= 0.0 && X_train[k*28*28+i*28+j] <= 0.0) cout << setw(4) << 0 << " " ;
    //             else cout << setw(4) << setprecision(2) << X_train[k*28*28+i*28+j] << " ";
    //         }
    //         cout << endl ;
    //     }
    //     cout << endl ;
    // }
    // cout.unsetf(std::ios_base::floatfield);

    // cout << "label = " << endl ;
    // for (int k=0; k<2; k++) {
    //     for (int i=0; i<10; i++) {
    //         cout << y_train[k*10+i] << " " ;
    //     }
    //     cout << endl ;
    // }
    // cout << endl ;

    // build model, 784->128->10
    chrono::steady_clock::time_point t_cal_start = chrono::steady_clock::now();

    int layer_dims[3] {784, 128, 10} ;
    int batch_size = 100 ;
    Model model(layer_dims, 2, batch_size, 0.05) ;

    // train
    for (int i=0; i<EPOCH; i++) {
        cout << "Epoch " << i << ":"<< endl ;
        cout << "\tTrain acc: " << model.Evaluate(X_train, y_train, 60000/batch_size) << endl ;
        cout << "\t Test acc: " << model.Evaluate(X_test, y_test, 10000/batch_size) << endl ;

        for (int j=0; j<60000/batch_size; j++) {
            model.Train(&X_train[j*batch_size*28*28], &y_train[j*batch_size*10]) ;
        }
    }

    // result
    cout << "Final Test Accuracy: " << model.Evaluate(X_test, y_test, 10000/batch_size) << endl ;

    chrono::steady_clock::time_point t_cal_end = chrono::steady_clock::now();
    cout << "Train take " << chrono::duration_cast<chrono::seconds>(t_cal_end - t_cal_start).count() << " s.\n";

    delete [] X_train ;
    delete [] y_train ;
    delete [] X_test ;
    delete [] y_test ;

    return 0 ;
}