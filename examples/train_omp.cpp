// == uCNN ====================================================================
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
//	  Enable OpenMP support in your project configuration/properties
//
// ==================================================================== uCNN ==

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <tchar.h>

#include "ucnn_omp.h"  
#include "MNIST.h"
#include "CIFAR.h"

const int thread_count = 4; 
const int mini_batch_size = 4;

/*
using namespace MNIST;
std::string data_path="../data/mnist/";
std::string model_file="../models/uCNN_MNIST.txt";
/*/
using namespace CIFAR10;
std::string data_path="../data/cifar-10-batches-bin/";
std::string model_file="../models/uCNN_CIFAR-10.txt";
//*/

// just to make the output easier to read
void draw_header(std::string name, int _seconds)
{
	std::string header="==  "+ name + "  "; 
	int seconds = _seconds;
	int minutes  =(int)(seconds/60);
	int hours = (int)(minutes/60);
	seconds = seconds-minutes*60;
	minutes = minutes-hours*60;
	std::string elapsed = " "+std::to_string((long long)hours)+":"+std::to_string((long long)minutes)+":"+std::to_string((long long)seconds);
	int L = 79-(int)header.length()-(int)elapsed.length();
	for(int i=0; i<L; i++) header+="=";
	std::cout << header<< elapsed<<std::endl;	
}

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
		const float *out=cnn.predict(test_images[k].data());

		// this utility funciton finds the max
		if(ucnn::max_index(out,out_size)==test_labels[k]) correct_predictions+=1;
	
		if(k%1000==0) progress.draw_progress(k);
	}

	float dt = progress.elapsed_seconds();
	float error = 100.f-(float)correct_predictions/record_cnt*100.f;
	//std::cout << "  testing time: " << dt << " seconds ("<<test_images.size() <<" records at "<<(float)record_cnt/dt << " records/second)      "<< std::endl;
	//std::cout << "  accuracy: " << (float)correct_predictions/record_cnt*100.f <<"%" << std::endl;
	return error;
}

int _tmain()
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

	// since labels are integers but network output is really an array, need to make appropriate target outputs
	std::vector<std::vector<float>> target;
	for(int i=0; i<10; i++)
	{
		// target array is -1 except at label index where it is 1
		std::vector<float> t(10,-1);
		t[i]=1;
		target.push_back(t);
	}

	// == setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad")
	ucnn::network cnn("adagrad"); 
	// !! the thread count must be set prior to loading or creating a model !!
	cnn.allow_threads(thread_count);  
	cnn.allow_mini_batches(mini_batch_size);
	cnn.set_smart_train_level(0.05f); // this skips back propagation if prediction is good - set to zero to turn off
	
	const int epochs=100;
	float learning_rate_decay=0.05f;
	// configure network 
	if(data_name().compare("MNIST")==0)
	{
		cnn.push_back("I1","input 28 28 1");			// MNIST is 28x28x1
		cnn.push_back("C1","convolution 5 5 12 relu");	// 5x5 kernel, 12 maps.  out size is 28-5+1=24
		cnn.push_back("P1","max_pool 4 ");				// pool 4x4 blocks. outsize is 6
		cnn.push_back("C2","convolution 5 5 150 relu");	// 5x5 kernel, 150 maps.  out size is 6-5+1=2
		cnn.push_back("P2","max_pool 2 2");				// pool 2x2 blocks. outsize is 2/2=1 
		cnn.push_back("FC1","fully_connected 100 relu");// fully connected 100 nodes, ReLU 
		cnn.push_back("FC2","fully_connected 10 tanh"); 
	}
	else // CIFAR
	{
		cnn.push_back("I1","input 32 32 3");				// CIFAR is 32x32x3
		cnn.push_back("C1","convolution 3 3 32 lrelu");	// 3x3 kernel, 32 maps.  out size is 32-3+1=30
		cnn.push_back("P1","max_pool 3");					// pool 3x3 blocks. outsize is 10
		cnn.push_back("C2","convolution 3 3 64 lrelu");	// 3x3 kernel, 64 maps.  out size is 10-3+1=8
		cnn.push_back("P2","max_pool 2");					// pool 2x2 blocks. outsize is 8/2=4 
		cnn.push_back("C3","convolution 3 3 64 lrelu");	// 3x3 kernel, 64 maps.  out size is 4-3+1=2
		cnn.push_back("FC1","fully_connected 128 tanh");	// fully connected 100 nodes, ReLU 
		cnn.push_back("FC2","fully_connected 10 tanh"); 
	}

	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();


	const int train_samples=(int)train_images.size();

	ucnn::progress overall_progress(epochs, "  overall:\t\t");
	//training epochs
	for(int epoch=0; epoch<epochs; epoch++)
	{
		draw_header(data_name(), (int)overall_progress.elapsed_seconds());
		std::cout << "  epoch:\t\t"<< epoch+1<< " of " << epochs <<std::endl;

		ucnn::progress progress(train_samples, "  training:\t\t");

		// lower learning rate every so often
		if((epoch+1)%10==0) 
			cnn.set_learning_rate((1.f-learning_rate_decay)*cnn.get_learning_rate() ) ;
	
		int skipped=0;
		#pragma omp parallel num_threads(thread_count) 
		#pragma omp for reduction(+:skipped) schedule(dynamic)
		for(int k=0; k<train_samples; k++) 
		{
			const bool trained=cnn.train(train_images[k].data(), target[train_labels[k]].data());
			if(!trained) skipped++; 
			if(k%1000==0) progress.draw_progress(k);
		}	
		cnn.sync_mini_batch();

		std::cout << "  training time:\t" << progress.elapsed_seconds() << " seconds                            "<< std::endl;
		
		if(cnn.get_smart_train_level()>0)
		{
			std::cout << "  skipped:\t\t" << skipped << " samples of "<<train_samples << " (" << (int)(100.f*skipped/train_samples)<<"%)"<< std::endl;
			// if skipped too many records, lower the skip energy threshold
			if(100*skipped/train_samples > 90) 
			{
				//std::cout << "  reducing smart train, was "<< (float)cnn.get_smart_train_level() << std::endl; 
				cnn.set_smart_train_level(0.5f*cnn.get_smart_train_level());
			}
		}

		float error_rate=0;
		// == run training set
		progress.reset((int)train_images.size(), "  testing in-sample:\t");
		error_rate=test(cnn, train_images, train_labels);
		std::cout << "  train accuracy:\t"<<100.f-error_rate<<"% ("<< error_rate<<"% error)      "<<std::endl;

		// == run testing set
		progress.reset((int)test_images.size(), "  testing out-of-sample:\t");
		error_rate=test(cnn, test_images, test_labels);
		std::cout << "  test accuracy:\t"<<100.f-error_rate<<"% ("<< error_rate<<"% error)      "<<std::endl;

		std::string model_file="../models/tmp_"+std::to_string((long long)epoch) +".bin";
		cnn.write(model_file,true);
		std::cout << "  saved model:\t\t"<<model_file<<std::endl<< std::endl;

	}
	std::cout << std::endl;
	return 0;
}
