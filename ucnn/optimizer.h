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
//    optimizer.h: stochastic optimization approaches
//
// ==================================================================== uCNN ==

#pragma once

#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>

#include "core_math.h"

// hack for VS2010 to handle c++11 for(:)
#if (_MSC_VER  == 1600)
	#ifndef __for__
	#define __for__ for each
	#define __in__ in
	#endif
#else
	#ifndef __for__
	#define __for__ for
	#define __in__ :
	#endif
#endif

namespace ucnn {



class optimizer
{
public:
	// learning rates are tweaked in inc_w function so that they can be similar for all optimizers
	float learning_rate;
	optimizer(): learning_rate(0.01f) {}
	virtual ~optimizer(){}
	virtual void reset() {}
	// this increments the weight matrix w, which corresponds to connection index 'g'
	// bottom is the number of grads coming up from the lower layer
	// top is the current output node value of the upper layer
	virtual void increment_w(matrix *w,  int g, const matrix &dW){}//, matrix *top){}
	virtual void push_back(int w, int h, int c){}	
};

#ifndef NO_TRAINING_CODE


class sgd: public optimizer
{
public:
	static const char *name(){return "sgd";}

	virtual void increment_w(matrix *w,  int g, const matrix &dW)
	{
		const float w_decay=0.01f;//1;
		for(int s=0; s<w->size(); s++)	
			w->x[s] -= (dW.x[s] + w_decay*w->x[s])*learning_rate;
	}
};

class adagrad: public optimizer
{
	// persistent variables that mirror size of weight matrix
	std::vector<matrix *> G1;
public:
	static const char *name(){return "adagrad";}

	virtual ~adagrad(){__for__(auto g __in__ G1) delete g;}
	virtual void push_back(int w, int h, int c) { G1.push_back(new matrix(w, h, c)); G1[G1.size() - 1]->fill(0); }
	
	virtual void reset() { __for__(auto g __in__ G1) g->fill(0.f);}
	virtual void increment_w(matrix *w,  int g, const matrix &dW)
	{
		float *g1 = G1[g]->x;
		//float min, max;
		//G1[g]->min_max(&min, &max);
		//std::cout << "((" << min << "," << max << ")";
		const float eps = 1.e-8f;
		// if (G1[g]->size() != w->size()) throw;
		for(int s=0; s<w->size(); s++) 
		{
			g1[s] += dW.x[s] * dW.x[s];
			//if (g1[s] < 1) throw;
			w->x[s] -= learning_rate*dW.x[s]/(std::sqrt(g1[s]) + eps);
		}	
	};
};

class rmsprop: public optimizer
{
	// persistent variables that mirror size of weight matrix
	std::vector<matrix *> G1;
public:
	static const char *name(){return "rmsprop";}
	virtual ~rmsprop(){__for__(auto g __in__ G1) delete g;}

	virtual void push_back(int w, int h, int c){ G1.push_back(new matrix(w,h,c)); G1[G1.size() - 1]->fill(0);}
	virtual void reset() { __for__(auto g __in__ G1) g->fill(0.f);}
	virtual void increment_w(matrix *w,  int g, const matrix &dW)
	{
		float *g1 = G1[g]->x;
		const float eps = 1.e-8f;
		const float mu = 0.999f;
		for(int s=0; s<(int)w->size(); s++)
		{
			g1[s] = mu * g1[s]+(1-mu) * dW.x[s] * dW.x[s];
			w->x[s] -= 0.01f*learning_rate*dW.x[s]/(std::sqrt(g1[s]) + eps);
		}	
	};

};

class adam: public optimizer
{
	float b1_t, b2_t;
	const float b1, b2;
	// persistent variables that mirror size of weight matrix
	std::vector<matrix *> G1;
	std::vector<matrix *> G2;
public:
	static const char *name(){return "adam";}
	adam(): b1(0.9f), b1_t(0.9f), b2(0.999f), b2_t(0.999f), optimizer()	{}
	virtual ~adam(){__for__(auto g __in__ G1) delete g; __for__(auto g __in__ G2) delete g;}

	virtual void reset()
	{
		b1_t*=b1; b2_t*=b2;
		__for__(auto g __in__ G1) g->fill(0.f);
		__for__(auto g __in__ G2) g->fill(0.f);
	}

	virtual void push_back(int w, int h, int c){G1.push_back(new matrix(w,h,c)); G1[G1.size() - 1]->fill(0); G2.push_back(new matrix(w,h,c)); G2[G2.size() - 1]->fill(0);
	}

	virtual void increment_w(matrix *w,  int g, const matrix &dW)
	{
		float *g1 = G1[g]->x;
		float *g2 = G2[g]->x;
		const float eps = 1.e-8f;
		const float b1=0.9f, b2=0.999f;
		for(int s=0; s<(int)w->size(); s++)
			{
				g1[s] = b1* g1[s]+(1-b1) * dW.x[s];
				g2[s] = b2* g2[s]+(1-b2) * dW.x[s]*dW.x[s];
				w->x[s] -= 0.1f*learning_rate* (g1[s]/(1.f-b1_t)) / ((float)std::sqrt(g2[s]/(1.-b2_t)) + eps);
			}	
	};

};


optimizer* new_optimizer(const char *type)
{
	if(type==NULL) return NULL;
	std::string act(type);
	if(act.compare(sgd::name())==0) { return new sgd();}
	if(act.compare(rmsprop::name())==0) { return new rmsprop();}
	if(act.compare(adagrad::name())==0) { return new adagrad();}
	if(act.compare(adam::name())==0) { return new adam();}

	return NULL;
}

#else


optimizer* new_optimizer(const char *type) {return NULL;}
optimizer* new_optimizer(std::string act){return NULL;}

#endif


} // namespace