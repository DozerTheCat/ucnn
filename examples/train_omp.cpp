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
//    train_omp.cpp: demonstrates configuring and training a new model
//    using openmp
//
//    Instructions: 
//	  Add the "ucnn" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Download CIFAR-10 data and unzip locally on your machine:
//		(http://www.cs.toronto.edu/~kriz/cifar.html)
//    Set the data_path variable in the code to point to your data location.
//	  Enable OpenMP support in your project configuration/properties
//
// ==================================================================== ucnn ==

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <tchar.h>

#include <ucnn_omp.h>
#include "MNIST.h"
#include "CIFAR.h"

const int thread_count = 8; 
const int mini_batch_size = 16;
const float initial_learning_rate = 0.025;
// #define USE_MNIST

#ifdef USE_MNIST
using namespace MNIST;
std::string data_path="../data/mnist/";
std::string optimizer = "adam";
#else // USE_MNIST
using namespace CIFAR10;
std::string data_path="../data/cifar-10-batches-bin/";
std::string optimizer = "adam";
#endif //USE_MNIST


float test(ucnn::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	// use progress object for simple timing and status updating
	ucnn::progress progress((int)test_images.size(), "  testing:\t\t");

	int out_size=cnn.out_size(); // we know this to be 10 for MNIST
	int correct_predictions=0;
	const int record_cnt= (int)test_images.size();

	#pragma omp parallel num_threads(thread_count) 
	#pragma omp for reduction(+:correct_predictions) schedule(dynamic)
	for(int k=0; k<record_cnt; k++)
	{
		// uccn returns a pointer to internally managed memmory (pointer to output of final layer- do not delete it)
		const int prediction=cnn.predict_class(test_images[k].data());

		// this utility funciton finds the max
		if(prediction ==test_labels[k]) correct_predictions+=1;
		if(k%1000==0) progress.draw_progress(k);
	}

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
		img.clip(-1, 1);
		memcpy(train_images[i].data(), img.x, sizeof(float)*img.size());
		//img.min_max(&fmin, &fmax);
		//std::cout << fmin << "," << fmax << "|";
	}
	for (int i = 0; i < test_images.size(); i++)
	{
		ucnn::matrix img(32, 32, 3, test_images[i].data());
		img -= mean;
		img.clip(-1, 1);
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
	
	remove_cifar_mean(test_images, train_images);  
	
	// == setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad", "adam")
	ucnn::network cnn(optimizer.c_str());
	// !! the thread count must be set prior to loading or creating a model !!
	cnn.allow_threads(thread_count);  
	cnn.set_mini_batch_size(mini_batch_size);
	cnn.set_smart_training(true); // automate training
	cnn.set_learning_rate(initial_learning_rate);
	
	// configure network 
#ifdef USE_MNIST
	cnn.push_back("I1", "input 28 28 1");			// MNIST is 28x28x1
	cnn.push_back("C1", "convolution 5 5 20 elu");	// 5x5 kernel, 12 maps.  out size is 28-5+1=24
	cnn.push_back("P1", "max_pool 4");				// pool 4x4 blocks. outsize is 6
	cnn.push_back("C2", "convolution 5 5 200 elu");	// 5x5 kernel, 150 maps.  out size is 6-5+1=2
	cnn.push_back("P2", "max_pool 2");				// pool 2x2 blocks. outsize is 2/2=1 
	cnn.push_back("FC1", "fully_connected 80 identity");// fully connected 100 nodes, ReLU 
	cnn.push_back("FC2", "fully_connected 10 tanh");


#else // CIFAR
	cnn.push_back("I1","input 32 32 3");				// CIFAR is 32x32x3
	cnn.push_back("C1","convolution 5 5 32 elu");		// 5x5 kernel, 32 maps.  out size is 32-5+1=28
	cnn.push_back("P1","max_pool 2");					// 3x3 pool stride 2. out size is (28-3)/2+1=14
	cnn.push_back("C2","convolution 5 5 32 elu");		// 3x3 kernel, 32 maps.  out size is 10
	cnn.push_back("P2","max_pool 2");					// 3x3 pool stride 2. outsize is (9-3)/2+1=5
	cnn.push_back("D1", "dropout 0.1");             // dropout 
	cnn.push_back("C3","convolution 5 5 64 elu");		// 3x3 kernel, 64 maps.  out size is 5-3+1=1
	cnn.push_back("D2", "dropout 0.25");             // dropout 
	cnn.push_back("FC1","fully_connected 100 identity");	// fully connected 100 nodes 
	cnn.push_back("D3", "dropout 0.5");             // dropout 
	cnn.push_back("FC2","fully_connected 10 tanh");
#endif
	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();
	std::cout << "==  Network Configuration  ====================================================" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;


	// add headers for table of values we want to log out
	ucnn::html_log log;
	log.set_table_header("epoch\ttest accuracy(%)\testimated accuracy(%)\tepoch time(s)\ttotal time(s)\tlearn rate\tmodel");
	log.set_note(cnn.get_configuration());

	// setup timer/progress for overall training
	ucnn::progress overall_progress(-1, "  overall:\t\t");
	const int train_samples = (int)train_images.size();
	while(1)
	{
		overall_progress.draw_header(data_name() + "  Epoch  " + std::to_string((long long)cnn.get_epoch() + 1) , true);
		// setup timer / progress for this one epoch
		ucnn::progress progress(train_samples, "  training:\t\t");

		cnn.start_epoch("cross_entropy");

		#pragma omp parallel num_threads(thread_count) 
		#pragma omp for schedule(dynamic)
		for(int k=0; k<train_samples; k++) 
		{
			// for CIFAR, can augment data random with mirror flips, for MNIST shift only
#ifndef USE_MNIST
			ucnn::matrix m(32, 32, 3, train_images[k].data()); if(rand()%2==0) m = m.flip_cols(); 
#else
			ucnn::matrix m(28, 28, 1, train_images[k].data()); 
#endif
			m = m.shift((rand() % 5) - 2, (rand() % 5) - 2, 1);
			cnn.train_class(m.x, train_labels[k]);
			if(k%1000==0) progress.draw_progress(k);
		}

		cnn.end_epoch();
		//cnn.set_learning_rate(0.5f*cnn.get_learning_rate());
		float dt = progress.elapsed_seconds();
		std::cout << "  mini batch:\t\t" << mini_batch_size << "                               " << std::endl;
		std::cout << "  training time:\t" << dt << " seconds on "<< thread_count << " threads"<< std::endl;
		std::cout << "  model updates:\t" << cnn.train_updates << " (" << (int)(100.f*(1. - (float)cnn.train_skipped / cnn.train_samples)) << "% of records)" << std::endl;
		std::cout << "  estimated accuracy:\t" << cnn.estimated_accuracy << "%" << std::endl;

		float error_rate=0;

		/* if you want to run in-sample testing on the training set, include this code
		// == run training set
		progress.reset((int)train_images.size(), "  testing in-sample:\t");
		error_rate=test(cnn, train_images, train_labels);
		std::cout << "  train accuracy:\t"<<100.f-error_rate<<"% ("<< error_rate<<"% error)      "<<std::endl;
		*/

		// == run testing set
		progress.reset((int)test_images.size(), "  testing out-of-sample:\t");
		error_rate=test(cnn, test_images, test_labels);
		std::cout << "  test accuracy:\t"<<100.f-error_rate<<"% ("<< error_rate<<"% error)      "<<std::endl;

		// save model
		std::string model_file="../models/tmp_"+std::to_string((long long)cnn.get_epoch()) +".txt";
		cnn.write(model_file);
		std::cout << "  saved model:\t\t"<<model_file<<std::endl<< std::endl;

		// write log file
		std::string log_out;
		log_out += float2str(dt) + "\t";
		log_out += float2str(overall_progress.elapsed_seconds()) + "\t";
		log_out += float2str(cnn.get_learning_rate()) + "\t";
		log_out += model_file;
		log.add_table_row(cnn.estimated_accuracy, 100.f - error_rate, log_out);
		// will write this every epoch
		log.write("../models/ucnn_log.htm");

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
