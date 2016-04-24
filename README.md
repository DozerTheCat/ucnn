# μCNN
# (micro Convolutional Neural Network)

A fairly bare bones C++ CNN implementation that was built with the goal to balance hack-ability, functionality, and speed.  It was a learning exercise inspired partially by tiny-cnn, which is a wonderful alternative to this code, and partially by my frustration trying to find a CNN package that easily builds in Visual Studio.  μCNN is in readable C++ with only old fashioned C tricks for optimization.  It is not designed to use GPUs or to scale over a cluster to train very deep models. For that, go with Caffe, TensorFlow, CMTK, Torch, etc…  μCNN is competitive with other CPU training options and can train usable models for most general object detection and object recognition problems.

Thought it is easy to dive into the code and modify layers or loss functions, at the same time the API provides a smart training option which smartly manages training.  μCNN includes the standard MNIST and CIFAR-10 examples and with a laptop CPU, smart training gives 99% accuracy on MNIST in about a minute. 

It was tested with MS Developer Studio 2010, 2015, and Cygwin g++ 5.3.0. It should be fairly portable with little work. 

Features Supported:
+ Layers:  Input, Fully Connected, Convolution, Max Pool, Fractional Max Pool (in progress), Stocastic Pooling (in progress), Concatenation (in progress)
+ Activation Functions: Identity, Hyperbolic Tangent (tanh), Exponential Linear Unit (ELU), Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (LReLU), Very Leaky Rectified Linear Unitv (VLReLU), Sigmoid, Softmax (in progress)
+ Optimization: Stochastic Gradient Descent, RMSProp, AdaGrad
+ Loss Functions: Mean Squared Error, Cross Entropy
+ Threading: optional and externally controlled at the application level using OpenMP
+ Architecture: Branching allowed
+ Smart Solver: Optimizes parameters and speeds training
+ Image Support: optional OpenCV utilities (in progress)

API Examples:
Load model and perform prediction:
```
#include <ucnn.h>

ucnn::network cnn; 
cnn.read("../models/uCNN_CIFAR-10.txt");
const int predicted_class=cnn.predict_class(float_image.data());

```

Construction of a new CNN for MNIST, and train records with OpenMP threading:  
```
#include <ucnn_omp.h>

ucnn::network cnn("adagrad");
cnn.set_smart_train(true);
cnn.allow_threads(8);  
cnn.set_mini_batch_size(25);
	
// add layer definitions	
cnn.push_back("I1","input 28 28 1");            // MNIST is 28x28x1
cnn.push_back("C1","convolution 5 5 15 relu");  // 5x5 kernel, 12 maps.  out size is 28-5+1=24
cnn.push_back("P1","max_pool 4 4");             // pool 4x4 blocks, stride 4. out size is 6
cnn.push_back("C2","convolution 5 5 150 relu"); // 5x5 kernel, 150 maps.  out size is 6-5+1=2
cnn.push_back("P2","max_pool 2 2");             // pool 2x2 blocks. out size is 2/2=1 
cnn.push_back("FC1","fully_connected 100 identity");// fully connected 100 nodes, ReLU 
cnn.push_back("FC2","fully_connected 10 tanh"); 
 
cnn.connect_all(); // connect layers automatically (no branches)

// train with OpenMP threading
cnn.start_epoch();

#pragma omp parallel num_threads(8) 
#pragma omp for schedule(dynamic)
for(int k=0; k<train_samples; k++) 
	cnn.train_class(train_images[k].data(), train_labels[k]);

cnn.end_epoch();

std::cout << "estimated accuracy:" << cnn.estimated_accuracy << "%" << std::endl;

cnn.write("ucnn_model_mnist.txt");

```

Example output from sample training application (configuration above):

```
==  MNIST  ============================================================== 0:0:0
  epoch:                1 of 100
  training time:        19.058 seconds
  skipped:              55112 samples of 60000 (91%)
  train accuracy:       98.8033% (1.19666% error)
  test accuracy:        98.65% (1.35% error)
  saved model:          ../models/tmp_0.bin

==  MNIST  ============================================================= 0:0:29
  epoch:                2 of 100
  training time:        16.411 seconds
  skipped:              56700 samples of 60000 (94%)
  train accuracy:       99.0683% (0.931671% error)
  test accuracy:        98.83% (1.17% error)
  saved model:          ../models/tmp_1.bin

==  MNIST  ============================================================= 0:0:56
  epoch:                3 of 100
  training time:        16.221 seconds
  skipped:              56740 samples of 60000 (94%)
  train accuracy:       99.2917% (0.708336% error)
  test accuracy:        99.04% (0.959999% error)
  saved model:          ../models/tmp_2.bin

==  MNIST  ============================================================= 0:1:23

```
