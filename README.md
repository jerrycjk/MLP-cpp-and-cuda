# MLP-cpp-and-cuda

This is a basic implementation of multi-layer perceptron by using c++ and cuda. It use mnist dataset to test the time and get a 10 times speed up.

## CPU version

- Class: Dense

This is a class for fully connected layer. It provides the functions: forward pass, backward pass, and update weight. 
I only implement relu and softmax activation functions.

- Class: Model

This is a class that connects layers into a model. It provides trainning, prediction, and evaluation functions.

- main

In main, I download and read mnist dataset from [here](http://yann.lecun.com/exdb/mnist/), roughly set batch_size = 100, lr = 0.05, epoch = 5, then start trainnig.

After 126s, I get about 90% accuracy.

## GPU version

Almost the same as cpu ver, except that the computation part in Dense is changed to cuda code. (not optimized yet)

All settings are the same as cpu ver, but it only takes 12s for training and get the accuracy about the same.
