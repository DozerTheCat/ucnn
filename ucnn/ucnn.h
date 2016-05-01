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
#define UCNN_SSE3

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
	void replaceAll(std::string& str, const std::string& from, const std::string& to) {
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}

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
			std::string min_string = std::to_string((long long)minutes);
			if (min_string.length() < 2) min_string = "0" + min_string;
			std::string sec_string = std::to_string((long long)seconds);
			if (sec_string.length() < 2) sec_string = "0" + sec_string;
			elapsed = " " + std::to_string((long long)hours) + ":" + min_string + ":" + sec_string;
			L-= (int)elapsed.length();
		}
		for (int i = 0; i<L; i++) header += "=";
		if (_time)
			std::cout << header << elapsed << std::endl;
		else 
			std::cout << header << std::endl;
	}
};

class html_log
{
	struct log_stuff
	{
		std::string str;
		float test_accurracy;
		float train_accurracy_est;
	};
	std::vector <log_stuff> log;
	std::string header;
	std::string notes;
public:
	html_log() {};

	void add_header(std::string tab_header) { header=tab_header;}
	void add_row(float train_acccuracy, float test_accuracy, std::string tab_row)
	{
		log_stuff s;
		s.str = tab_row; s.test_accurracy = test_accuracy; s.train_accurracy_est = train_acccuracy;
		log.push_back(s);
	}
	void add_note(std::string msg) {notes = msg;}
	bool write(std::string filename) {

		std::string top = "<!DOCTYPE html><html><head><meta http-equiv=\"content - type\" content=\"text/html; charset = UTF - 8\"><style>table, th, td{border: 1px solid black; border - collapse: collapse; } th, td{ padding: 5px;}</style><meta name=\"robots\" content=\"noindex, nofollow\"><meta name=\"googlebot\" content=\"noindex, nofollow\"><meta http-equiv=\"refresh\" content=\"30\"/><script type=\"text/javascript\" src=\"/js/lib/dummy.js\"></script><link rel=\"stylesheet\" type=\"text/css\" href=\"/css/result-light.css\"><style type=\"text/css\"></style><title>Micro CNN Training Report</title></head><body><script type=\"text/javascript\" src=\"https://www.gstatic.com/charts/loader.js\"></script>Training Summary <script type=\"text/javascript\">document.write(Date());</script>:<div id = \"chart_div\"></div><script type = 'text/javascript'>//<![CDATA[\n";
		top += "google.charts.load('current', { packages: ['corechart', 'line'] });\ngoogle.charts.setOnLoadCallback(drawLineColors);\nfunction drawLineColors() {";
		top += "\nvar data = new google.visualization.DataTable();data.addColumn('number', 'Epoch');data.addColumn('number', 'Training Estimate');data.addColumn('number', 'Validation Testing');data.addRows([";
		std::string data = "";
		float min = 100;
		for (int i = 0; i < log.size(); i++)
		{
			if ((100. - log[i].train_accurracy_est) < min) min = (100. - log[i].train_accurracy_est);
			if ((100. - log[i].test_accurracy) < min) min = (100. - log[i].test_accurracy);
			data += "[" + int2str(i) + "," + float2str(100. - log[i].train_accurracy_est) + "," + float2str(100. - log[i].test_accurracy) + "],";
		}
		float min_10 = min;
//		while (min_10 > min) min_10 /= 10.f;

		std::string mid = "]);var options = { 'height':400, hAxis: {title: 'Epoch', logScale: true},vAxis : {title: 'Error (%)', logScale: true, viewWindow: {min:"+float2str(min_10)+",max: 100},ticks: [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2, 5, 10, 20, 50, 100] },colors : ['#313055','#F52B00'] };var chart = new google.visualization.LineChart(document.getElementById('chart_div')); chart.draw(data, options);}//]]>\n </script>";

		std::string msg = "<table style='width:100 %'>";
		int N = (int)log.size();
		msg += "<tr><td>" + header + "</td></tr>";
		for (int i = N - 1; i >=0; i--)
			msg += "<tr><td>" + log[i].str + "</td></tr>";

		replaceAll(msg, "\t", "</td><td>");
		replaceAll(notes, "\n", "<br>");

		std::string bottom = "</tr></table><br>"+notes+"</body></html>";

		std::ofstream f(filename.c_str());
		f << top; f << data; f << mid; f << msg; f << bottom;

		f.close();
		return true;
	}

};

}// namespace