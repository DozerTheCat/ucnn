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
//
//    ucnn.h:  include this one file to use ucnn without OpenMP
//
// ==================================================================== uCNN ==
#pragma once

// undef if no plans to use training
#define INCLUDE_TRAINING_CODE



//#define USE_OMP
//#define USE_AF
//#define USE_CUDA


#ifdef USE_AF
#include <arrayfire.h>
#define AF_RELEASE
	#ifdef USE_CUDA
		#define AF_CUDA
		#pragma comment(lib, "afcuda")
	#else
		#define AF_CPU
		#pragma comment(lib, "afcpu")
	#endif
#endif


#include <time.h>
#include <string>
#include "network.h"


namespace ucnn
{

// returns Energy (euclidian distance / 2) and max index
float match_labels(const float *out, const float *target, const int size, int *best_index=NULL)
{
	float E=0;
	int max_j=0;
	for(int j=0; j<size; j++)
	{
		E+=(out[j]-target[j])*(out[j]-target[j]);
		if(out[max_j]<out[j]) max_j=j;
	}
	if(best_index) *best_index=max_j;
	E*=0.5;
	return E;
}

int max_index(const float *out, const int size)
{
	int max_j=0;
	for(int j=0; j<size; j++)
	{
		if(out[max_j]<out[j]) max_j=j;
	}
	return max_j;
}

// class to handle timing and drawing text progress output
class progress
{
public:
	progress(int size=-1, char *label=NULL ) {reset(size, label);}

	unsigned int start_progress_time;
	unsigned int total_progress_items;
	std::string label_progress;
	// if default values used, the values won't be changed from last call
	void reset(int size=-1, char *label=NULL ) 
	{start_progress_time=clock(); if(size>0) total_progress_items=size; if(label!=NULL) label_progress=label;}

	float elapsed_seconds() {return (float)(clock()-start_progress_time)/CLOCKS_PER_SEC;}
	float remaining_seconds(int item_index)
	{
		float elapsed_dt = elapsed_seconds();
		float percent_complete = 100.f*item_index/total_progress_items;
		if(percent_complete>0) return ((elapsed_dt/percent_complete*100.f)-elapsed_dt);
		return 0.f;
	}
	void draw_progress(int item_index)
	{
		int time_remaining = (int)remaining_seconds(item_index);
		float percent_complete = 100.f*item_index/total_progress_items;
		if(percent_complete>0) std::cout << label_progress << (int)percent_complete <<"% (" << (int)time_remaining << "sec remaining)     \r";
	}
};

}// namespace