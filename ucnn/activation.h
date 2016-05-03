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
//    activation.h:  neuron activation functions
//
// ==================================================================== ucnn ==

#pragma once


#include <math.h>
#include <algorithm>
#include <string>

namespace ucnn {

// not using class because I thought this may be faster than vptrs
namespace tan_h 
{
	inline float  f(float *in, int i, int size, float bias) // this is activation f(x)
	{
        const float ep = std::exp(in[i]+bias);
        const float em = std::exp(-in[i]+bias); 
        return (ep - em) / (ep + em);
    }

	inline float  df(float *in, int i, int size) { return 1.f - in[i]*in[i]; }  // this is df(x), but we pass in the activated value f(x) and not x 
	const char name[]="tanh";
}

namespace elu 
{
	inline float  f(float *in, int i, int size, float bias) {  if(in[i]+bias < 0) return 0.1f*(std::exp(in[i]+bias)- 1.f); return in[i]+bias; }
	inline float  df(float *in, int i, int size) { if(in[i] > 0) return 1.f; else return 0.1f*std::exp(in[i]);}
	const char name[]="elu";
}

namespace identity 
{
	inline float  f(float *in, int i, const int size, const float bias) {  return bias+in[i]; }
	inline float  df(float *in, int i, const int size){return 1.f;};
	const char name[]="identity";
}
namespace relu 
{
	inline float  f(float *in, int i, const int size, const float bias) {  if(in[i]+bias < 0) return 0; return in[i]+bias; }
	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.0f; }
	const char name[]="relu";
};
namespace lrelu 
{
	inline float  f(float *in, int i, const int size, const float bias) {  if(in[i]+bias < 0) return 0.01f*(in[i]+bias); return in[i]+bias; }
	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.01f; }
	const char name[]="lrelu";
};
namespace vlrelu 
{
	inline float  f(float *in, int i, const int size, const float bias) {  if(in[i]+bias < 0) return 0.33f*(in[i]+bias); return in[i]+bias; }
	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.33f; }
	const char name[]="vlrelu";
};

namespace sigmoid
{
	inline float  f(float *in, int i, const int size, const float bias) { return in[i] =(1.0f/(1.0f+exp(-(in[i]+bias))));}
	inline float df(float *in, int i, const int size) {return in[i]*(1.f-in[i]); }
	const char name[]="sigmoid";
};

/*
namespace softmax 
{
	void f(float *in_out, int size=0) 
	{ 
		float max= in_out[0];
		for(int i=1; i<size; i++) if(in_out[i] > max) max= in_out[i];
	
		float denom=0;
		for(int i=0; i<size; i++) denom+= std::exp(in_out[i] - max);
		
		for(int i=0; i<size; i++)
			in_out[i] = std::exp(in_out[i] - max)/ denom;

	}

	void df(float *in_out, int size=0) 
	{
		for(int i=0; i<size; i++) 
		{
			in_out[i] = in_out[i]*(1.f-in_out[i]);
		}
	}

	// oj = exp (zj - m - log{sum_i{exp(zi-m)}})


};
*/
namespace none
{
	inline float f(float *in, int i, int size, float bias) {return 0;};
	inline float df(float *in, int i, int size) {return 0;};
	const char name[]="none";
};

typedef struct 
{
public:
	float (*f)(float *, int, const int, const float);
	float (*df)(float *, int, const int);
	const char *name;
} activation_function;

activation_function* new_activation_function(std::string act)
{
	activation_function *p = new activation_function;
	if(act.compare(tan_h::name)==0) { p->f = &tan_h::f; p->df = &tan_h::df; p->name=tan_h::name;return p;}
	if(act.compare(identity::name)==0) { p->f = &identity::f; p->df = &identity::df; p->name=identity::name; return p;}
	if(act.compare(vlrelu::name)==0) { p->f = &vlrelu::f; p->df = &vlrelu::df; p->name=vlrelu::name; return p;}
	if(act.compare(lrelu::name)==0) { p->f = &lrelu::f; p->df = &lrelu::df; p->name=lrelu::name; return p;}
	if(act.compare(relu::name)==0) { p->f = &relu::f; p->df = &relu::df; p->name=relu::name;return p;}
	if(act.compare(sigmoid::name)==0) { p->f = &sigmoid::f; p->df = &sigmoid::df; p->name=sigmoid::name; return p;}
	if(act.compare(elu::name)==0) { p->f = &elu::f; p->df = &elu::df; p->name=elu::name; return p;}
	if(act.compare(none::name)==0) { p->f = &none::f; p->df = &none::df; p->name=none::name; return p;}
	delete p;
	return NULL;
}

activation_function* new_activation_function(const char *type)
{
	std::string act(type);
	return new_activation_function(act);
}

} // namespace