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
//
//    cv_utils.cpp:  helper functions using opencv 
//
// ==================================================================== ucnn ==


#define OPENCV_3

#ifdef OPENCV_2
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#pragma comment(lib, "opencv_core249")
#pragma comment(lib, "opencv_highgui249")
#pragma comment(lib, "opencv_imgproc249")
#pragma comment(lib, "opencv_contrib249")
#endif

#ifdef OPENCV_3
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#pragma comment(lib, "opencv_world310")
#endif


#include "core_math.h"
#include "network.h"

namespace ucnn
{
cv::Mat matrix2cv(ucnn::matrix &m, bool uc8 = false)
{
	cv::Mat cv_m;
	if (m.chans != 3)
	{
		cv_m = cv::Mat(m.cols, m.rows, CV_32FC1, m.x);
	}
	if (m.chans == 3)
	{
		cv::Mat in[3];
		in[0] = cv::Mat(m.cols, m.rows, CV_32FC1, m.x);
		in[1] = cv::Mat(m.cols, m.rows, CV_32FC1, &m.x[m.cols*m.rows]);
		in[2] = cv::Mat(m.cols, m.rows, CV_32FC1, &m.x[2 * m.cols*m.rows]);
		cv::merge(in, 3, cv_m);
	}
	if (uc8)
	{
		double min_, max_;
		cv_m = cv_m.reshape(1);
		cv::minMaxIdx(cv_m, &min_, &max_);
		cv_m = cv_m - min_;
		max_ = max_ - min_;
		cv_m /= max_;
		cv_m *= 255;
		cv_m = cv_m.reshape(m.chans, m.rows);
		if (m.chans != 3)
			cv_m.convertTo(cv_m, CV_8UC1);
		else
			cv_m.convertTo(cv_m, CV_8UC3);
	}
	return cv_m;
}

ucnn::matrix cv2matrix(cv::Mat &m)
{
	if (m.type() == CV_8UC1)
	{
		m.convertTo(m, CV_32FC1);
		m = m / 255.;
	}
	if (m.type() == CV_8UC3)
	{
		m.convertTo(m, CV_32FC3);
	}
	if (m.type() == CV_32FC1)
	{
		return ucnn::matrix(m.cols, m.rows, 1, (float*)m.data);
	}
	if (m.type() == CV_32FC3)
	{
		cv::Mat in[3];
		cv::split(m, in);
		ucnn::matrix out(m.cols, m.rows, 3);
		memcpy(out.x, in[0].data, m.cols*m.rows * sizeof(float));
		memcpy(&out.x[m.cols*m.rows], in[1].data, m.cols*m.rows * sizeof(float));
		memcpy(&out.x[2 * m.cols*m.rows], in[2].data, m.cols*m.rows * sizeof(float));
		return out;
	}
	return  ucnn::matrix(0, 0, 0);
}
ucnn::matrix bgr2ycrcb(ucnn::matrix &m)
{
	cv::Mat cv_m = matrix2cv(m);
	double min_, max_;
	cv_m = cv_m.reshape(1);
	cv::minMaxIdx(cv_m, &min_, &max_);
	cv_m = cv_m - min_;
	max_ = max_ - min_;
	cv_m /= max_;

	cv_m = cv_m.reshape(m.chans, m.rows);
	cv::Mat cv_Y;
	cv::cvtColor(cv_m, cv_Y, CV_BGR2YCrCb);
	cv_Y = cv_Y.reshape(1);
	cv_Y -= 0.5f;
	cv_Y *= 2.f;
	cv_Y = cv_Y.reshape(m.chans);

	m = cv2matrix(cv_Y);
	return m;
}

void show(ucnn::matrix &m, float zoom = 1.0f, const char *win_name = "")
{
	cv::Mat cv_m = matrix2cv(m);

	double min_, max_;
	cv_m = cv_m.reshape(1);
	cv::minMaxIdx(cv_m, &min_, &max_);
	cv_m = cv_m - min_;
	max_ = max_ - min_;
	cv_m /= max_;
	//	cv_m += 1.f;
	//	cv_m *= 0.5;
	cv_m = cv_m.reshape(m.chans, m.rows);

	if (zoom != 1.f) cv::resize(cv_m, cv_m, cv::Size(0, 0), zoom, zoom);
	cv::imshow(win_name, cv_m);
	cv::waitKey(1);
}

// null name hides all windows	
void hide(const char *win_name="")
{
	if(win_name==NULL) cv::destroyAllWindows();
	else cv::destroyWindow(win_name);
}

ucnn::matrix draw_cnn_weights(ucnn::network &cnn)
{
	int w = (int)cnn.W.size();
	cv::Mat im;
	
	std::vector <cv::Mat> im_layers;

	int layers = (int)cnn.layer_sets[0].size();
	for (int k = 0; k < layers; k++)
	{
		base_layer *layer = cnn.layer_sets[0][k];
		if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

		__for__(auto &link __in__ layer->forward_linked_layers)
		{
			int connection_index = link.first;
			base_layer *p_bottom = link.second;

			for (auto i = 0; i < cnn.W[connection_index]->chans; i++)
			{
				cv::Mat im = matrix2cv(cnn.W[connection_index]->get_chan(i), true);
				cv::resize(im, im, cv::Size(0, 0), 4., 4., 0);
				im_layers.push_back(im);
			}
			// draw these nicely
			int s = im_layers[0].cols;
			cv::Mat tmp(s,p_bottom->node.chans*s, CV_32FC1);// = im.clone();
			for(int i=0; i<im_layers.size(); i++)
				im_layers[i].copyTo(tmp(cv::Rect(i*s, 0, s, s)));
			im = tmp;

		}
		break;
	}
	/*
	int imgs = (int)im_layers.size();
	cv::Mat im;
	if (imgs <= 0) return im;

	im = im_layers[0].clone(); //(im_layers[0].rows, im_layers[0].cols, CV_8UC1);
	int W = im.cols;

	if (W<400)
	{
	W = 400;
	float S = (float)W / (float)im.cols;
	cv::resize(im, im, cv::Size(W, (int)(S*im.rows)), 0, 0, 0);
	}

	for (auto i = 1; i<imgs; i++)
	{
	float S = (float)W / (float)im_layers[i].cols;
	cv::Mat mout;
	cv::resize(im_layers[i], mout, cv::Size(W, (int)(S*im_layers[i].rows)), 0, 0, 0);

	// new output image
	cv::Mat tmp(im.rows + mout.rows, im.cols, CV_8UC1);// = im.clone();
	//std::cout << "H=" << im.rows << ", W=" << im.cols << std::endl;
	//std::cout << "H2=" << mout.rows << ", W2=" << mout.cols << std::endl;
	//std::cout << "im copy";
	im.copyTo(tmp(cv::Rect(0, 0, im.cols, im.rows)));
	//std::cout << "mout copy";
	mout.copyTo(tmp(cv::Rect(0, im.rows, mout.cols, mout.rows)));
	//std::cout << "tmp clone";
	im = tmp.clone();

	}
	*/
	if (im.cols>0 && im.rows>0)
	{
		cv::applyColorMap(im, im, cv::COLORMAP_JET);// COLORMAP_HOT); // cv::COLORMAP_JET); COLORMAP_RAINBOW
	}
	return cv2matrix(im);
}

}