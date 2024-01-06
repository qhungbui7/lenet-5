#ifndef SRC_FASHION_MNIST_H_
#define SRC_FASHION_MNIST_H_

#include <fstream>
#include <iostream>

using namespace std;
#include <string>
#include "../utils.h"

class FASHION_MNIST {
 private:
  string data_dir;

 public:
  Matrix train_data;
  Matrix train_labels;
  Matrix test_data;
  Matrix test_labels;

  void read_mnist_data(string filename, Matrix& data);
  void read_mnist_label(string filename, Matrix& labels);

  explicit FASHION_MNIST(string data_dir) : data_dir(data_dir) {}
  void read();
};

#endif  // SRC_FASHION_MNIST_H_
