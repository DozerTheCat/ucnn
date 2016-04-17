# μCNN
(micro convolutional neural network)

A fairly bare bones CNN implementatoin that was built with the goal to balance hackability, functionality, and speed.  It was inspired by tiny-cnn, which is a wonderful alternative to this code. μCNN is only C++ with only standard C coding tricks for optimizaiton. Advanced CPU or GPU optimizations will be through optional packages.

μCNN includes the standard MNIST and CIFAR-10 examples. Single laptop core CPU training gives 70% accuracy on CIFAR-10 in 30mins and 98.7% accuracy on MNIST in 30secs.

Features Supported:
  Layers:  Input, Fully Connected, Convolution, Max Pool
  Activation Functions: Identity, Hyperbolic Tangent (tanh), Exponential Linear Unit (ELU), Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (LReLU), Very Leaky Rectified Linear Unitv (VLReLU), Sigmoid
  Optimization: Stochastic Gradient Descent, RMSProp, AdaGrad
  
  

