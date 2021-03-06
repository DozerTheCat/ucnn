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
//    test.cpp:  Simple example using pre-trained model to test ucnn
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

// by selecting a different namespace, we'll call different data parsing functions below
/*
using namespace MNIST;
std::string data_path="../data/mnist/";
std::string model_file="../models/ucnn_mnist.txt";
/*/
using namespace CIFAR10;
std::string data_path="../data/cifar-10-batches-bin/";
std::string model_file="../models/ucnn_cifar.txt";
//*/

void test(ucnn::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	// use progress object for simple timing and status updating
	ucnn::progress progress((int)test_images.size(), "  testing : ");

	int out_size=cnn.out_size(); // we know this to be 10 for MNIST
	int correct_predictions=0;
	const int record_cnt= (int)test_images.size();

	for(int k=0; k<record_cnt; k++)
	{
		// predict_class returnes the output index of the highest response
		const int prediction=cnn.predict_class(test_images[k].data());

		if(prediction ==test_labels[k]) correct_predictions++;

		if(k%1000==0) progress.draw_progress(k);
	}

	float dt = progress.elapsed_seconds();
	std::cout << "  test time: " << dt << " seconds                                          "<< std::endl;
	std::cout << "  records: " << test_images.size() << std::endl;
	std::cout << "  speed: " << (float)record_cnt/dt << " records/second" << std::endl;
	std::cout << "  accuracy: " << (float)correct_predictions/record_cnt*100.f <<"%" << std::endl;
}


int main()
{
	// == parse data
	// array to hold image data (note that ucnn does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	// array to hold image labels 
	std::vector<int> test_labels;
	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if(!parse_test_data(data_path, test_images, test_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}

	// == setup the network  
	ucnn::network cnn; 
	// load model
	if(!cnn.read(model_file)) {std::cerr << "error: could not read model.\n"; return 1;}
	std::cout << "ucnn configuration:" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

	// == run the test
	std::cout << "Testing " << data_name() << ":" << std::endl;
	// this function will loop through all images, call predict, and print out stats
	test(cnn, test_images, test_labels);	
	std::cout << std::endl;
	return 0;
}
