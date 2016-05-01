# μcnn (micro convolutional neural network)

μcnn is an efficient C++ CNN implementation that was built with the goal to balance hack-ability, functionality, and speed.  Consisting of only a handful of header files, μcnn is in portable C++ with old fashioned C tricks for optimization. With optional OpenMP and SSE3 speedups enabled it's speed is competitive with other CPU based CNN frameworks. Being a minimal CPU solution, it is not designed to scale over a cluster to train very deep models (for that, go with GPUs and Caffe, TensorFlow, CMTK, Torch, etc…)

The μcnn API provides a 'smart training' option which abstracts the management of the training process but still provides the flexibility to handle the threading and input data as you'd like. Just make a loop and pass in training samples until μcnn says stop. On the standard MNIST handwritten digit database, μcnn's 'smart training' gives 99% accuracy in about a 20 seconds. 

Latest change status is on the [μcnn wiki](https://github.com/DozerTheCat/ucnn/wiki). 

Features:
+ Layers:  Input, Fully Connected, Convolution, Max Pool, Dropout, (Fractional Max Pool, Stocastic Pooling, Concatenation all in progress)
+ Activation Functions: Identity, Hyperbolic Tangent (tanh), Exponential Linear Unit (ELU), Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (LReLU), Very Leaky Rectified Linear Unitv (VLReLU), Sigmoid, Softmax (in progress)
+ Optimization: Stochastic Gradient Descent, RMSProp, AdaGrad, Adam
+ Loss Functions: Mean Squared Error, Cross Entropy
+ Threading: optional and externally controlled at the application level using OpenMP
+ Architecture: Branching allowed
+ Solver: Smart training optimizes parameters, speeds up training, and provides exit criteria.
+ Image Support: optional OpenCV utilities (in progress)
+ Portable: tested with MS Developer Studio 2010, 2015, and Cygwin g++ 5.3.0. 
+ Logging: html training report

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

ucnn::network cnn("adam");
cnn.set_smart_train(true);
cnn.allow_threads(8);  
cnn.set_mini_batch_size(24);
	
// add layer definitions	
cnn.push_back("I1","input 28 28 1");            // MNIST is 28x28x1
cnn.push_back("C1","convolution 5 5 15 elu");   // 5x5 kernel, 12 maps.  out size is 28-5+1=24
cnn.push_back("P1","max_pool 4 4");             // pool 4x4 blocks, stride 4. out size is 6
cnn.push_back("C2","convolution 5 5 150 elu");  // 5x5 kernel, 150 maps.  out size is 6-5+1=2
cnn.push_back("P2","max_pool 2 2");             // pool 2x2 blocks. out size is 2/2=1 
cnn.push_back("FC1","fully_connected 100 identity");// fully connected 100 nodes 
cnn.push_back("FC2","fully_connected 10 tanh"); 
 
cnn.connect_all(); // connect layers automatically (no branches)

while(1)
{
	// train with OpenMP threading
	cnn.start_epoch("cross_entropy");
	
	#pragma omp parallel num_threads(8) 
	#pragma omp for schedule(dynamic)
	for(int k=0; k<train_samples; k++) cnn.train_class(train_images[k].data(), train_labels[k]);
	
	cnn.end_epoch();
	
	std::cout << "estimated accuracy:" << cnn.estimated_accuracy << "%" << std::endl;
	
	cnn.write("ucnn_model_mnist.txt");
	
	if (cnn.elvis_left_the_building()) break;
};

```

Example output from sample application:

```
==  MNIST  Epoch  1  ==================================================== 0:0:0
  mini batch:           24
  training time:        8.99951 seconds on 8 threads
  model updates:        384 (15% of records)
  estimated accuracy:   95.8583%
  test accuracy:        98.44% (1.56001% error)
  saved model:          ../models/tmp_1.txt

==  MNIST  Epoch  2  ==================================================== 0:0:9
  mini batch:           24
  training time:        6.67038 seconds on 8 threads
  model updates:        193 (7% of records)
  estimated accuracy:   98.1418%
  test accuracy:        98.77% (1.23% error)
  saved model:          ../models/tmp_2.txt

==  MNIST  Epoch  3  =================================================== 0:0:17
  mini batch:           24
  training time:        6.32536 seconds on 8 threads
  model updates:        156 (6% of records)
  estimated accuracy:   98.4768%
  test accuracy:        98.98% (1.02% error)
  saved model:          ../models/tmp_3.txt

==  MNIST  Epoch  4  =================================================== 0:0:24
  mini batch:           24
  training time:        6.10435 seconds on 8 threads
  model updates:        136 (5% of records)
  estimated accuracy:   98.6808%
  test accuracy:        99% (1% error)
  saved model:          ../models/tmp_4.txt

==  MNIST  Epoch  5  =================================================== 0:0:30

```
