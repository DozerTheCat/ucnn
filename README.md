# uCNN
micro cnn (convolutional neural network)

A fairly bare bones CNN implementatoin that was built with the goal to balance simplicity, functionality, and speed.  It was inspired by tiny-cnn, which is a wonderful alternative to this code. uCNN is only C++ with no CPU or GPU specific optimizations. 

uCNN includes the standard MNIST and CIFAR-10 examples. Single threaded CPU training gives 70% accuracy on CIFAR-10 in 30mins and 98.7% accuracy on MNIST in 30secs. 

