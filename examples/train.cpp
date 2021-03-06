// == ucnn ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
// 
//    This file is part of ucnn.
//
//    uncc is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Affero General Public License as published
//    by the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    ucnn is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Affero General Public License for more details.
//
//    You should have received a copy of the GNU Affero General Public License
//    along with ucnn.  If not, see <http://www.gnu.org/licenses/>.
//
// ============================================================================
//    train.cpp: demonstrates configuring and training a new model
//
//    Instructions: 
//	  Add the "ucnn" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Download CIFAR-10 data and unzip locally on your machine:
//		(http://www.cs.toronto.edu/~kriz/cifar.html)
//    Set the data_path variable in the code to point to your data location.
//
// ==================================================================== ucnn ==

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <tchar.h>

#include <ucnn.h>  
#include "MNIST.h"
#include "CIFAR.h"

//*
using namespace MNIST;
std::string data_path="../data/mnist/";
/*/
using namespace CIFAR10;
std::string data_path="../data/cifar-10-batches-bin/";
//*/

float test(ucnn::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	// use progress object for simple timing and status updating
	ucnn::progress progress((int)test_images.size(), "  testing:\t\t");

	int out_size=cnn.out_size(); // we know this to be 10 for MNIST
	int correct_predictions=0;
	const int record_cnt= (int)test_images.size();

	for(int k=0; k<record_cnt; k++)
	{
		// uccn returns a pointer to internally managed memmory (pointer to output of final layer- do not delete it)
		const int prediction=cnn.predict_class(test_images[k].data());

		// this utility funciton finds the max
		if(prediction ==test_labels[k]) correct_predictions+=1;
	
		if(k%1000==0) progress.draw_progress(k);
	}

	float dt = progress.elapsed_seconds();
	float error = 100.f-(float)correct_predictions/record_cnt*100.f;
	return error;
}

void remove_cifar_mean(std::vector<std::vector<float>> &train_images, std::vector<std::vector<float>> &test_images)
{
	// calculate the mean for every pixel position 
	ucnn::matrix mean(32, 32, 3);
	mean.fill(0);
	for (int i = 0; i < train_images.size(); i++) mean += ucnn::matrix(32, 32, 3, train_images[i].data());
	mean *= (float)(1.f / train_images.size());

	// remove mean from data
	for (int i = 0; i < train_images.size(); i++)
	{
		ucnn::matrix img(32, 32, 3, train_images[i].data());
		img -= mean;
		memcpy(train_images[i].data(), img.x, sizeof(float)*img.size());
	}
	for (int i = 0; i < test_images.size(); i++)
	{
		ucnn::matrix img(32, 32, 3, test_images[i].data());
		img -= mean;
		memcpy(test_images[i].data(), img.x, sizeof(float)*img.size());
	}
}

int main()
{
	// == parse data
	// array to hold image data (note that ucnn does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	std::vector<int> test_labels;
	std::vector<std::vector<float>> train_images;
	std::vector<int> train_labels;

	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if(!parse_test_data(data_path, test_images, test_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}
	if(!parse_train_data(data_path, train_images, train_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}

	// == setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad")
	ucnn::network cnn("adagrad"); 
	cnn.set_smart_training(false); // speed up training
	
	// configure network 
	if(data_name().compare("MNIST")==0)
	{
		cnn.push_back("I1","input 28 28 1");			// MNIST is 28x28x1
		cnn.push_back("C1","convolution 5 5 15 elu");	// 5x5 kernel, 12 maps.  out size is 28-5+1=24
		cnn.push_back("P1","max_pool 4");				// pool 4x4 blocks. outsize is 6
		cnn.push_back("C2","convolution 5 5 150 elu");	// 5x5 kernel, 150 maps.  out size is 6-5+1=2
		cnn.push_back("P2","max_pool 2");				// pool 2x2 blocks. outsize is 2/2=1 
		cnn.push_back("FC1","fully_connected 100 identity");// fully connected 100 nodes, ReLU 
		cnn.push_back("FC2","fully_connected 10 tanh"); 
	}
	else // CIFAR
	{
		cnn.push_back("I1","input 32 32 3");				// CIFAR is 32x32x3
		cnn.push_back("C1","convolution 5 5 32 relu");		// 5x5 kernel, 32 maps.  out size is 32-5+1=28
		cnn.push_back("P1","max_pool 3 2");					// 3x3 pool stride 2. out size is (28-3)/2+1=13
		cnn.push_back("C2","convolution 3 3 32 relu");		// 3x3 kernel, 32 maps.  out size is 11
		cnn.push_back("P2","max_pool 3 2");					// 3x3 pool stride 2. outsize is (11-3)/2+1=5
		cnn.push_back("C3","convolution 3 3 64 relu");		// 3x3 kernel, 64 maps.  out size is 5-3+1=3
		//cnn.push_back("P3", "max_pool 3");					// 3x3 pool stride 1. outsize is 1
		cnn.push_back("FC1","fully_connected 64 identity");	// fully connected 100 nodes, ReLU 
		cnn.push_back("FC2","fully_connected 10 tanh"); 
	}

	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();

	const int train_samples=(int)train_images.size();

	//training epochs
	// setup timer/progress for overall training
	ucnn::progress overall_progress(-1, "  overall:\t\t");
	while (1)
	{
		overall_progress.draw_header(data_name() + "  Epoch  " + std::to_string((long long)cnn.get_epoch() + 1), true);
		// setup timer / progress for this one epoch
		ucnn::progress progress(train_samples, "  training:\t\t");

		cnn.start_epoch("cross_entropy");

		for (int k = 0; k<train_samples; k++)
		{
			// for CIFAR, can augment data random with mirror flips
			//ucnn::matrix m(32, 32, 3, train_images[k].data()); if(rand()%2==0) m = m.flip_cols(); 
			//cnn.train_class(m.x, train_labels[k]);
			cnn.train_class(train_images[k].data(), train_labels[k]);
			if (k % 1000 == 0) progress.draw_progress(k);
		}

		cnn.end_epoch();
		std::cout << "  mini batch:\t\t" << 1 << "                                  " << std::endl;
		std::cout << "  training time:\t" << progress.elapsed_seconds() << " seconds on " << 1 << " thread" << std::endl;
		std::cout << "  model updates:\t" << cnn.train_updates << " (" << (int)(100.f*(1. - (float)cnn.train_skipped / cnn.train_samples)) << "% of records)" << std::endl;
		std::cout << "  estimated accuracy:\t" << cnn.estimated_accuracy << "%" << std::endl;


		float error_rate = 0;

		/* if you want to run in-sample testing on the training set, include this code
		// == run training set
		progress.reset((int)train_images.size(), "  testing in-sample:\t");
		error_rate=test(cnn, train_images, train_labels);
		std::cout << "  train accuracy:\t"<<100.f-error_rate<<"% ("<< error_rate<<"% error)      "<<std::endl;
		*/

		// == run testing set
		progress.reset((int)test_images.size(), "  testing out-of-sample:\t");
		error_rate = test(cnn, test_images, test_labels);
		std::cout << "  test accuracy:\t" << 100.f - error_rate << "% (" << error_rate << "% error)      " << std::endl;

		// save model
		std::string model_file = "../models/tmp_" + std::to_string((long long)cnn.get_epoch()) + ".txt";
		cnn.write(model_file);
		std::cout << "  saved model:\t\t" << model_file << std::endl << std::endl;

		// can't seem to improve
		if (cnn.elvis_left_the_building())
		{
			std::cout << "Elvis just left the building. No further improvement in training found.\nStopping.." << std::endl;
			break;
		}

	};
	std::cout << std::endl;
	return 0;
}
