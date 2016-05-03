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
//    test_omp.cpp:  extends test.cpp to use threads with OpenMP
//
//    Instructions: 
//	  Add the "ucnn" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Download CIFAR-10 data and unzip locally on your machine:
//		(http://www.cs.toronto.edu/~kriz/cifar.html)
//    Set the data_path variable in the code to point to your data location.
//	  Enable OpenMP support in your project configuration/properties
//    "<><><><><><>" will preceed changes from the non-threaded version
//
// ==================================================================== ucnn ==

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cstdio>
#include <tchar.h>

// <><><><><><<><><><><><> instead of "ucnn.h" include "ucnn_omp.h"
#include <ucnn_omp.h> 
#include "MNIST.h"
#include "CIFAR.h"

// <><><><><><><><><><><><> define desired thread count (though you can do things dynamically if needed)
const int thread_count = 8; 

// by selecting a different namespace, we'll call different data parsing functions below
/*
using namespace MNIST;
std::string data_path="../data/mnist/";
std::string model_file="../models/ucnn_mnist.txt";
/*/
using namespace CIFAR10;
std::string data_path="../data/cifar-10-batches-bin/";
std::string model_file = "../models/ucnn_cifar.txt";
//*/

void test(ucnn::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels)
{
	// use progress object for simple timing and status updating
	ucnn::progress progress((int)test_images.size(), "  testing : ");

	int out_size=cnn.out_size(); // we know this to be 10 for MNIST
	int correct_predictions=0;
	const int record_cnt = 1000;// (int)test_images.size();

	std::vector<int> single;
	int a, b;
	FILE *f = fopen("./single.txt", "rt");
	for (int i = 0; i < 1000; i++)
	{
		std::fscanf(f, "%d\t%d\n", &a, &b);
		single.push_back(b);
	}fclose(f);
	// <><><><><><<><><><><><> use standard omp parallel for loop, with schedule dynamic the screen output will still somewhat work
	#pragma omp parallel num_threads(thread_count) 
	#pragma omp for reduction(+:correct_predictions) schedule(dynamic)
	for (int k = 0; k < record_cnt; k++)
	{
		int prediction;
			// predict_class returnes the output index of the highest response
			prediction = cnn.predict_class(test_images[k].data());
			//			FILE *f = fopen("./single.txt", "at");
			//				std::fprintf(f, "%d\t%d\n", k, prediction);
			//			fclose(f);
		//	if (single[k] != prediction) std::cout << "pisser\n";

			if (prediction == test_labels[k]) correct_predictions++;

			if (k % 1000 == 0) progress.draw_progress(k);
		
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
	// <><><><><><<><><><><><> add network.allow_threads() 
	// here we need to prepare uccn to store data from multiple threads
	// alternatively you can create a different ucnn::network for each thread and use each network in a separate thread
	// !! the thread count must be set prior to loading or creating a model !!
	cnn.allow_threads(thread_count);  

	// load model 
	if (!cnn.read(model_file)) { std::cerr << "error: could not read model.\n"; return 1; }
	std::cout << "ucnn configuration:" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

	// == run the test 
	std::cout << "Testing " << data_name() << ":" << std::endl;
	// this function will loop through all images, call predict, and print out stats
	test(cnn, test_images, test_labels);
	cnn.clear();

	std::cout << std::endl;
	return 0;
}
