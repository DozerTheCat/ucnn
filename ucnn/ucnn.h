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
#include <chrono>
#include "core_math.h"
#include "network.h" // this is the important thing
// this other stuff may be moved to utils


namespace ucnn
{

// class to handle timing and drawing text progress output
class progress
{
public:
	progress(int size=-1, const char *label=NULL ) {reset(size, label);}

	std::chrono::time_point<std::chrono::system_clock>  start_progress_time;
	unsigned int total_progress_items;
	std::string label_progress;
	// if default values used, the values won't be changed from last call
	void reset(int size=-1, const char *label=NULL ) 
	{
		start_progress_time= std::chrono::system_clock::now();
		if(size>0) total_progress_items=size; if(label!=NULL) label_progress=label;
	}
	float elapsed_seconds() 
	{	
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_progress_time);
		return (float)time_span.count();
	}
	float remaining_seconds(int item_index)
	{
		float elapsed_dt = elapsed_seconds();
		float percent_complete = 100.f*item_index/total_progress_items;
		if(percent_complete>0) return ((elapsed_dt/percent_complete*100.f)-elapsed_dt);
		return 0.f;
	}
	// this doesn't work correctly with g++/Cygwin
	// the carriage return seems to delete the text... 
	void draw_progress(int item_index)
	{
		int time_remaining = (int)remaining_seconds(item_index);
		float percent_complete = 100.f*item_index/total_progress_items;
		if (percent_complete > 0)
		{
			std::cout << label_progress << (int)percent_complete << "% (" << (int)time_remaining << "sec remaining)              \r"<<std::flush;
		}
	}
	void draw_header(std::string name, bool _time=false)
	{
		std::string header = "==  " + name + "  ";

		int seconds = 0;
		std::string elapsed;
		int L = 79 - (int)header.length();
		if (_time)
		{
			seconds = (int)elapsed_seconds();
			int minutes = (int)(seconds / 60);
			int hours = (int)(minutes / 60);
			seconds = seconds - minutes * 60;
			minutes = minutes - hours * 60;
			elapsed = " " + std::to_string((long long)hours) + ":" + std::to_string((long long)minutes) + ":" + std::to_string((long long)seconds);
			L-= (int)elapsed.length();
		}
		for (int i = 0; i<L; i++) header += "=";
		if (_time)
			std::cout << header << elapsed << std::endl;
		else 
			std::cout << header << std::endl;
	}
};

// just to make the output easier to read


}// namespace