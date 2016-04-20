# μCNN
(micro convolutional neural network)

A fairly bare bones CNN implementation that was built with the goal to balance hack-ability, functionality, and speed.  It was a learning exercise inspired by tiny-cnn, which is a wonderful alternative to this code.  μCNN is only C++ with only old fashioned C tricks for optimization.  Therefore is not meant to train deep models. For that, go with Caffe, TensorFlow, CMTK, Torch, etc…  However it is still able to train usable models for many general object detection and object recognition problems.

μCNN includes the standard MNIST and CIFAR-10 examples. Laptop CPU training  gives 99% accuracy on MNIST in about 52secs of training using both cores. The Windows executable size is around 100KB.

It was tested with MS Developer Studio 2010 and 2015. It should be fairly portable with little work. 

Features Supported:
+ Layers:  Input, Fully Connected, Convolution, Max Pool, Fractional Max Pool (in progress), Concatenation (in progress)
+ Activation Functions: Identity, Hyperbolic Tangent (tanh), Exponential Linear Unit (ELU), Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (LReLU), Very Leaky Rectified Linear Unitv (VLReLU), Sigmoid, Softmax (in progress)
+ Optimization: Stochastic Gradient Descent, RMSProp, AdaGrad
+ Loss Functions: only mean squared error has been tested
+ Threading: optional and externally controlled at the application level using OpenMP
+ Architecture: Non-standard network topologies like resnet or an inception module can be constructed
+ Smart training: Just an option to skip training when forward prediction is already good. This can dramatically speed up training for some problems. 
+ Image Support: optional OpenCV utilities

API Examples:
Load model and perform prediction:
```
#include <ucnn.h>

ucnn::network cnn; 
cnn.read("../models/uCNN_CIFAR-10.txt");
const float *out=cnn.predict(float_image.data());

```

Construction of a new CNN for MNIST, and train records with OpenMP threading:  
```
#include <ucnn_omp.h>

ucnn::network cnn("adagrad");
cnn.set_smart_train_level(0.05f);
cnn.allow_threads(thread_count);  
cnn.allow_mini_batches(mini_batch_size);
	
// add layer definitions	
cnn.push_back("I1","input 28 28 1");              // MNIST is 28x28x1
cnn.push_back("C1","convolution 5 5 12 lrelu");   // 5x5 kernel, 12 maps.  output size is 28-5+1=24
cnn.push_back("P1","fractional_max_pool 12");     // max pool. output size is 12
cnn.push_back("C2","convolution 5 5 20 lrelu");   // 5x5 kernel, 20 maps.  output size is 10-5+1=6
cnn.push_back("P2","fractional_max_pool 5");      // fractional pool. output size is 5 
cnn.push_back("C3","convolution 5 5 150 lrelu");  // 5x5 kernel, 150 maps.  output size is 5-5+1=1
cnn.push_back("FC1","fully_connected 100 lrelu"); // fully connected 100 nodes, Leaky ReLU 
cnn.push_back("FC2","fully_connected 10 tanh"); 

// connect layers automatically
cnn.connect_all();

// train with OpenMP threading
#pragma omp parallel num_threads(thread_count) 
#pragma omp for schedule(dynamic)
for(int k=0; k<train_samples; k++) 
	cnn.train(train_images[k].data(), target[k].data());

cnn.sync_mini_batch();
cnn.write(model_file);
	
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
