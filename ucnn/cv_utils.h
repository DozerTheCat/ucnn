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
//    cv_utils.cpp:  helper functions using opencv 
//
// ==================================================================== uCNN ==


#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#pragma comment(lib, "opencv_core249")
#pragma comment(lib, "opencv_highgui249")
#pragma comment(lib, "opencv_imgproc249")
#pragma comment(lib, "opencv_contrib249")

#include "ucnn.h"

namespace ucnn
{

cv::Mat matrix2cv(ucnn::matrix &m)
{
	if(m.chans==1 || m.chans>3)
	{
	cv::Mat cv_m(m.cols,m.rows,CV_32FC1,m.x);
	return cv_m;
	}
	if(m.chans==2)
	{
	cv::Mat cv_m(m.cols,m.rows,CV_32FC2,m.x);
	return cv_m;
	}
	if(m.chans==3)
	{
	cv::Mat cv_m(m.cols,m.rows,CV_32FC3,m.x);
	return cv_m;
	}
}

void show(ucnn::matrix &m, float zoom=1.0f, const char *win_name="")
{
	cv::Mat cv_m=matrix2cv(m);
	if(m.chans==1)
	{
		double min_, max_;
		cv::minMaxIdx(cv_m, &min_,&max_);
		cv_m=cv_m-min_;
		max_=max_-min_;
		cv_m/=max_;
	}

	if(zoom!=1.f)
		cv::resize(cv_m, cv_m,cv::Size(0,0),zoom,zoom);
	cv::imshow(win_name,cv_m);
	cv::waitKey(1);
}
void hide(const char *win_name="")
{
	if(win_name==NULL) cv::destroyAllWindows();
	else cv::destroyWindow(win_name);
}

}