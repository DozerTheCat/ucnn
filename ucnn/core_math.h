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
//    core_math.h: defines matrix class and math functions
//
// ==================================================================== uCNN ==


#pragma once

#include <math.h>
#include <string.h>
#include <string>
#include <cstdlib>

namespace ucnn
{

inline float dot(const float *x1, const float *x2, const int size)
{
	switch (size)
	{
	case 1: return x1[0] * x2[0];
	case 2: return x1[0] * x2[0] + x1[1] * x2[1];
	case 3: return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2];
	case 4: return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3];
	case 5: return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3] + x1[4] * x2[4];
	default:
		float v = 0;
		for (int i = 0; i<size; i++) v += x1[i] * x2[i];
		return v;
	};
}


inline void unwrap_5x5(float *aligned_out, const float *in, const int in_size)
{
	float *img = aligned_out;
	const int node_size = in_size - 5 + 1;
	for (int j = 0; j < node_size; j += 1)//stride) // intput w
	{
		for (int i = 0; i < node_size; i += 1)//stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			memcpy(img, tn, 5 * sizeof(float)); tn += in_size; img += 5;
			memcpy(img, tn, 5 * sizeof(float)); tn += in_size; img += 5;
			memcpy(img, tn, 5 * sizeof(float)); tn += in_size; img += 5;
			memcpy(img, tn, 5 * sizeof(float)); tn += in_size; img += 5;
			memcpy(img, tn, 5 * sizeof(float)); tn += in_size; img += 5;
			memset(img, 0, 3 * sizeof(float)); img += 3; // junk pad
		}
	}
}

inline void dot_unwrapped_5x5(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	const float *_filt = filter_ptr;
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		float c0 = _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0];// +_img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt = filter_ptr;
		out[j] = c0;
	}
}

inline void dot_unwrapped_5x5_sse(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm_prefetch((const char *)(out), _MM_HINT_T0);
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		__m128 a, b, c0, c1;
		//_mm_prefetch((const char *)(filter_ptr + 12), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);

		a = _mm_load_ps(_img + 4); b = _mm_load_ps(filter_ptr + 4);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 8); b = _mm_load_ps(filter_ptr + 8);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 12); b = _mm_load_ps(filter_ptr + 12);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 16); b = _mm_load_ps(filter_ptr + 16);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 20); b = _mm_load_ps(filter_ptr + 20);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 24); b = _mm_load_ps(filter_ptr + 24);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);

		_mm_store_ss(&out[j], c0);
		_img += 28;
	}
}

inline void unwrap_3x3(float *aligned_out, const float *in, const int in_size)
{
	float *img = aligned_out;
	const int node_size = in_size - 3 + 1;
	for (int j = 0; j < node_size; j += 1)//stride) // intput w
	{
		for (int i = 0; i < node_size; i += 1)//stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			memcpy(img, tn, 3 * sizeof(float)); tn += in_size; img += 3;
			memcpy(img, tn, 3 * sizeof(float)); tn += in_size; img += 3;
			memcpy(img, tn, 3 * sizeof(float)); tn += in_size; img += 3;
			memset(img, 0, 3 * sizeof(float)); img += 3; // junk pad
		}
	}
}

inline void dot_unwrapped_3x3_sse(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm_prefetch((const char *)(out), _MM_HINT_T0);
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		__m128 a, b, c0, c1;
		_mm_prefetch((const char *)(out + j), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);

		a = _mm_load_ps(_img + 4); b = _mm_load_ps(filter_ptr + 4);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 8); b = _mm_load_ps(filter_ptr + 8);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);

		_mm_store_ss(&out[j], c0);

		_img += 12;
	}
}

inline void dot_unwrapped_sse(const float *_img, const float *filter_ptr, float *out, const int outsize, const int filtersize)
{
	__m128 a, b, c0, c1;
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		_mm_prefetch((const char *)(out + j), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);
		for (int i = 4; i < filtersize; i += 4)
		{
			a = _mm_load_ps(_img + i); b = _mm_load_ps(filter_ptr + i);
			c1 = _mm_mul_ps(a, b);
			c0 = _mm_add_ps(c0, c1);
		}

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);
		_mm_store_ss(&out[j], c0);

		_img += filtersize;
	}
}

// second item is rotated 180 (this is a convolution)
inline float dot_rot180(const float *x1, const float *x2, const int size)	
{	
	switch(size)
	{
	case 1:  return x1[0]*x2[0]; 
	case 2:  return x1[0]*x2[1]+x1[1]*x2[0]; 
	case 3:  return x1[0]*x2[2]+x1[1]*x2[1]+x1[2]*x2[0]; 
	case 4:  return x1[0]*x2[3]+x1[1]*x2[2]+x1[2]*x2[1]+x1[3]*x2[0]; 
	case 5:  return x1[0]*x2[4]+x1[1]*x2[3]+x1[2]*x2[2]+x1[3]*x2[1]+x1[4]*x2[0]; 
	default: 
		float v=0;
		for(int i=0; i<size; i++) v+=x1[i]*x2[size-i-1];
		return v;	
	};

}

inline float unwrap_2d_dot(const float *x1, const float *x2, const int size, int stride1, int stride2)	
{	
	float v=0;	

	for(int j=0; j<size; j++) 
		v+= dot(&x1[stride1*j],&x2[stride2*j],size);
	return v;
}

//#include <smmintrin.h>

inline float unwrap_2d_dot_5x5(const float *x1, const float *x2,  int stride1, int stride2)	
{
	const float *f1 = &x1[0]; const float *f2 = &x2[0];
	float v;
	v = f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; 
	return v;
}

inline float unwrap_2d_dot_3x3(const float *x1, const float *x2,  int stride1, int stride2)	
{	
	const float *f1=&x1[0]; const float *f2=&x2[0];
	float v;
	v = f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]; 
	return v;
}

// second item is rotated 180
inline float unwrap_2d_dot_rot180_5x5(const float *x1, const float *x2,  int stride1, int stride2)	
{	
	const float *f1=&x1[0]; const float *f2=&x2[stride2*4];
	float v = f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; 
	return v;
}

// second item is rotated 180
inline float unwrap_2d_dot_rot180_3x3(const float *x1, const float *x2,  int stride1, int stride2)	
{	
	const float *f1=&x1[0]; const float *f2=&x2[stride2*2];
	float v = f1[0]*f2[2]+f1[1]*f2[1]+f1[2]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[2]+f1[1]*f2[1]+f1[2]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[2]+f1[1]*f2[1]+f1[2]*f2[0]; 
	return v;
}

// second item is rotated 180
inline float unwrap_2d_dot_rot180(const float *x1, const float *x2, const int size, int stride1, int stride2)	
{	
	float v=0;	
	for(int j=0; j<size; j++) 
	{
		v+= dot_rot180(&x1[stride1*j],&x2[stride2*(size-j-1)],size);
	}
	return v;
}

// matrix class ---------------------------------------------------
// should use opencv if available
//
class matrix
{
	int _size;
	int _capacity;
public:
	std::string _name;
	int cols, rows, chans;
	float *x;

	matrix( ): cols(0), rows(0), chans(0), _size(0), _capacity(0), x(NULL)  {} 

	matrix( int _w, int _h, int _c=1, float *data=NULL): cols(_w), rows(_h), chans(_c) 
	{
		_size=cols*rows*chans; _capacity=_size; x = new float[_size]; 
		if(data!=NULL) memcpy(x,data,_size*sizeof(float));
	}

	// copy constructor - deep copy
	matrix( const matrix &m) : cols(m.cols), rows(m.rows), chans(m.chans), _size(m._size), _capacity(m._size)   {x = new float[_size]; memcpy(x,m.x,sizeof(float)*_size); } // { v=m.v; x=(float*)v.data();}
	// copy and pad constructor
	matrix( const matrix &m, int pad_cols, int pad_rows) : cols(m.cols+2*pad_cols), rows(m.rows+2*pad_rows), chans(m.chans)
	{
		_size = cols*rows*chans;
		_capacity = _size;
		x = new float[_size]; 
		fill(0);
		for(int c=0; c<m.chans; c++)
		for(int j=0; j<m.rows; j++)
		{
			memcpy(x+pad_cols+(pad_rows+j)*cols+c*cols*rows,m.x+j*m.cols +c*m.cols*m.rows,sizeof(float)*m.cols);
		}
		 
	} // { v=m.v; x=(float*)v.data();}

	~matrix() { if(x) delete [] x; x=NULL;}
	
	matrix get_chan(int channel) const
	{
		return matrix(cols,rows,1,&x[channel*cols*rows]);	
	}

	// if edge_pad==0, then the padded area is just 0. Otherwise it fills with edge pixel colors
	matrix pad(int dx, int dy, int edge_pad=0)
	{
		matrix v(cols+2*dx,rows+2*dy,chans);
		v.fill(0);
		//float *new_x = new float[chans*w*h]; 
		for(int k=0; k<chans; k++)
		{
			int v_chan_offset=k*v.rows*v.cols;
			int chan_offset=k*cols*rows;
			for(int j=0; j<rows; j++)
			{
				memcpy(&v.x[dx+(j+dy)*v.cols+v_chan_offset], &x[j*cols+chan_offset], sizeof(float)*cols);
				if(edge_pad)
				{
					// do left/right side
					for(int i=0; i<dx; i++)
					{
						v.x[i+(j+dy)*v.cols+v_chan_offset]=x[0+j*cols+chan_offset];
						v.x[i+dx+cols+(j+dy)*v.cols+v_chan_offset]=x[(cols-1)+j*cols+chan_offset];
					}
				}
			}
			// top bottom pad
			if(edge_pad)
			{
				for(int j=0; j<dy; j++)
				{
					memcpy(&v.x[(j)*v.cols+v_chan_offset],&v.x[(dy)*v.cols+v_chan_offset], sizeof(float)*v.cols);
					memcpy(&v.x[(j+dy+rows)*v.cols+v_chan_offset], &v.x[(rows-1+dy)*v.cols+v_chan_offset], sizeof(float)*v.cols);
				}
			}
		}

		return v;
	}

	matrix crop(int dx, int dy, int w, int h)
	{
		matrix v(w,h,chans);

		//float *new_x = new float[chans*w*h]; 
		for(int k=0; k<chans; k++)
		for(int j=0; j<h; j++)
		{
			memcpy(&v.x[j*w+k*w*h], &x[dx+(j+dy)*cols+k*rows*cols], sizeof(float)*w);
		}

		return v;
	}

	ucnn::matrix shift(int dx, int dy, int edge_pad)
	{
		int orig_cols=cols;
		int orig_rows=rows;
		int off_x=abs(dx);
		int off_y=abs(dy);

		ucnn::matrix shifted= pad(off_x, off_y, edge_pad);

		return shifted.crop(off_x-dx, off_y-dy,orig_cols,orig_rows);
	}

	ucnn::matrix flip_cols ()
	{
		ucnn::matrix v(cols,rows,chans);
		for(int k=0; k<chans; k++)
			for(int j=0; j<rows; j++)
				for(int i=0; i<cols; i++)
					v.x[i+j*cols+k*cols*rows]=x[(cols-i-1)+j*cols+k*cols*rows];

		return v;
	}

	void clip(float min, float max)
	{
		int s = rows*cols*chans;
		for (int i = 0; i < s; i++)
		{
			if (x[i] < min) x[i] = min;
			if (x[i] > max) x[i]=max;
		}
	}


	void min_max(float *min, float *max, int *min_i=NULL, int *max_i=NULL)
	{
		int s = rows*cols*chans;
		int mini = 0;
		int maxi = 0; 
		for (int i = 0; i < s; i++)
		{
			if (x[i] < x[mini]) mini = i;
			if (x[i] > x[maxi]) maxi = i;
		}
		*min = x[mini];
		*max = x[maxi];
		if (min_i) *min_i = mini;
		if (max_i) *max_i = maxi;
	}

	float remove_mean(int channel)
	{
		int s = rows*cols;
		int offset = channel*s;
		float average=0;
		for(int i=0; i<s; i++) average+=x[i+offset];		
		average= average/(float)s;
		for(int i=0; i<s; i++) x[i+offset]-=average;		
		return average;
	}

	float remove_mean()
	{
		int s = rows*cols*chans;
		//int offset = channel*s;
		float average=0;
		for(int i=0; i<s; i++) average+=x[i];		
		average= average/(float)s;
		for(int i=0; i<s; i++) x[i]-=average;		
		return average;
	}
	void fill(float val) { for(int i=0; i<_size; i++) x[i]=val; }
	// deep copy
	inline matrix& operator =(const matrix &m)
	{
		resize(m.cols, m.rows, m.chans);
		memcpy(x,m.x,sizeof(float)*_size);
		return *this;
	}

	int  size() const {return _size;} 
	
	void resize(int _w, int _h, int _c) { 
		int s = _w*_h*_c;
		if(s>_capacity) { if(_capacity>0) delete [] x; _size = s; _capacity=_size; x = new float[_size];}
		cols=_w; rows=_h; chans=_c; _size=s;
	} 
	
	// dot vector to 2d mat
	inline matrix dot_1dx2d(const matrix &m_2d) const
	{
		ucnn::matrix v(m_2d.rows, 1, 1);
		for(int j=0; j<m_2d.rows; j++)	v.x[j]=dot(x,&m_2d.x[j*m_2d.cols],_size);
		return v;
	}

	// +=
	inline matrix& operator+=(const matrix &m2){
	  for(int i = 0; i < _size; i++) x[i] += m2.x[i];
	  return *this;
	}
	// -=
	inline matrix& operator-=(const matrix &m2) {
		for (int i = 0; i < _size; i++) x[i] -= m2.x[i];
		return *this;
	}
	// *= float
	inline matrix operator *=(const float v) {
		for (int i = 0; i < _size; i++) x[i] = x[i] * v;
		return *this;
	}
	// * float
	inline matrix operator *(const float v){
		matrix T(cols,rows,1);
	  for(int i = 0; i < _size; i++) T.x[i] = x[i] * v;
	  return T;
	}

	// +
	inline matrix operator +(matrix m2)
	{
		matrix T(cols,rows,chans);
		for(int i = 0; i < _size; i++) T.x[i] = x[i] + m2.x[i]; 
		return T;
	}
};

}// namespace

