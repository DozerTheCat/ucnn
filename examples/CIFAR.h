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
//    CIFAR.h: prepares CIFAR data for testing/training
//
//    See http://www.cs.toronto.edu/~kriz/cifar.html for info about CIFAR-10
//
// ==================================================================== uCNN ==

#pragma once


#include <iostream> // cout
#include <sstream>
#include <fstream>
#include <iomanip> //setw
#include <random>

#include <stdio.h>
#include <tchar.h>

namespace CIFAR10
{

std::string data_name() {return std::string("CIFAR-10");}

bool parse_cifar_data(const std::string& cifar_file, 
	std::vector<std::vector<float>> *images,
	std::vector<int> *labels,
	float scale_min = -1.0, float scale_max = 1.0,
	int x_padding = 0, int y_padding = 0) 
{
	std::ifstream ifs(cifar_file.c_str(), std::ios::in | std::ios::binary);

	if (ifs.bad() || ifs.fail()) 
	{
		//std::cout << "failed to open file:" + cifar_file;
		return false;
	}

	// format is 1byte class, 1024b (32x32) R, 1024b (32x32) G, 1024b (32x32) B
	// 10,000 items in each file

	for (size_t i = 0; i < 10000; i++) 
	{
		// read label
		unsigned char label;
		ifs.read((char*) &label, 1);
		labels->push_back((int) label);
		
		// read image
		std::vector<unsigned char> image_c(32*32*3);
		ifs.read((char*) &image_c[0], 32*32*3);
		int width = 32+2*x_padding;
		int height = 32+2*y_padding;
		std::vector<float> image(height*width*3);

		// convert from RGB to BGR
		for (size_t c = 0; c < 3; c++)
		for (size_t y = 0; y < 32; y++)
		for (size_t x = 0; x < 32; x++)
			image[width * (y + y_padding) + x + x_padding + (3-c-1)*width*height] =
			(image_c[y * 32 + x+c*32*32] / 255.0f) * (scale_max - scale_min) + scale_min;
		
		images->push_back(image);
		
	}	
	return true;
}

bool parse_test_data(std::string &data_path, std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels, 
	float min_val=-1.f, float max_val=1.f, int padx=0, int pady=0)
{
	return parse_cifar_data(data_path+"test_batch.bin", &test_images, &test_labels, min_val, max_val, padx, pady);
}

bool parse_train_data(std::string &data_path, std::vector<std::vector<float>> &train_images, std::vector<int> &train_labels, 
	float min_val=-1.f, float max_val=1.f, int padx=0, int pady=0)
{
	if(!parse_cifar_data(data_path+"data_batch_1.bin", &train_images, &train_labels, min_val, max_val, padx, pady)) return false;
	if(!parse_cifar_data(data_path+"data_batch_2.bin", &train_images, &train_labels, min_val, max_val, padx, pady)) return false;
	if(!parse_cifar_data(data_path+"data_batch_3.bin", &train_images, &train_labels, min_val, max_val, padx, pady)) return false;
	if(!parse_cifar_data(data_path+"data_batch_4.bin", &train_images, &train_labels, min_val, max_val, padx, pady)) return false;
	if(!parse_cifar_data(data_path+"data_batch_5.bin", &train_images, &train_labels, min_val, max_val, padx, pady)) return false;
	return true;
}

} // namespace

