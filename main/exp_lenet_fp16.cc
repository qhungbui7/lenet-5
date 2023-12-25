
/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
*/

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/custom/fashion_mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/custom/fp16/fp16.h"


int main() {
  // data
  FASHION_MNIST dataset("../data/fashion-mnist/");
  dataset.read();
  std::cout << "FP16 Convolution LeNet-5 implementation" << std::endl;
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn

  // Conv(int channel_in, int height_in, int width_in, int channel_out,
  //      int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
  //      int pad_h = 0)


  // MaxPooling(int channel_in, int height_in, int width_in,
  //            int height_pool, int width_pool, int stride = 1) :

  // [(W−K+2P)/S]+1

  Network lenet5;
  Layer* conv1 = new FP16Conv(1, 28, 28, 6, 5, 5, 1, 0, 0);
  Layer* relu1 = new ReLU;
  // (28 - 5 + 2 * 0) / 1 + 1 = 24

  Layer* pool2 = new MaxPooling(6, 24, 24, 2, 2, 1);
  // (24 - 2 + 2 * 0) / 1 + 1 = 23

  Layer* conv3 = new FP16Conv(6, 23, 23, 16, 5, 5, 1, 0, 0);
  Layer* relu3 = new ReLU;
  // (23 - 5 + 2 * 0) / 1 + 1 = 19

  Layer* pool4 = new MaxPooling(16, 19, 19, 2, 2, 1);
  // (23 - 2 + 2 * 0) / 1 + 1 = 18

  Layer* fc6 = new FullyConnected(pool4->output_dim(), 120);
  Layer* relu6 = new ReLU;
  // 19 * 19 * 16 = 5776


  Layer* fc7 = new FullyConnected(120, 84);
  Layer* relu7 = new ReLU;

  Layer* fc8 = new FullyConnected(84, 10);
  Layer* softmax8 = new Softmax;

  lenet5.add_layer(conv1);
  lenet5.add_layer(relu1);

  lenet5.add_layer(pool2);

  lenet5.add_layer(conv3);
  lenet5.add_layer(relu3);

  lenet5.add_layer(pool4);

  lenet5.add_layer(fc6);
  lenet5.add_layer(relu6);

  lenet5.add_layer(fc7);
  lenet5.add_layer(relu7);

  lenet5.add_layer(fc8);
  lenet5.add_layer(softmax8);

  // loss
  Loss* loss = new CrossEntropy;
  lenet5.add_loss(loss);
  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;
  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                    std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                    std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1) {
        std::cout << ith_batch << "-th grad: " << std::endl;
        lenet5.check_gradient(x_batch, target_batch, 10);
      }
      lenet5.forward(x_batch);
      //lenet5.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0) {
        std::cout << ith_batch << "-th batch, loss: " << lenet5.get_loss()
        << std::endl;
      }
      // optimize
      //lenet5.update(opt);
    }
    // test
    lenet5.forward(dataset.test_data);
    float acc = compute_accuracy(lenet5.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  return 0;
}

