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
//    cost.h:  cost/loss function for training
//
// ==================================================================== ucnn ==

#pragma once

#include <math.h>
#include <algorithm>
#include <string>

namespace ucnn {

namespace mse 
{
	inline float  cost(float out, float target) {return 0.5f*(out-target)*(out-target);};
	inline float  d_cost(float out, float target) {return (out-target);};
	const char name[]="mse";
}
/*
namespace triplet_loss 
{
	inline float  E(float out1, float out2, float out3) {return 0.5f*(out-target)*(out-target);};
	inline float  dE(float out, float target) {return (out-target);};
	const char name[]="triplet_loss";
}
*/
namespace cross_entropy 
{
	inline float  cost(float out, float target) {return (-target * std::log(out) - (1.f - target) * std::log(1.f - out));};
	inline float  d_cost(float out, float target) {return ((out - target) / (out*(1.f - out)));};
	const char name[]="cross_entropy";
}


typedef struct 
{
public:
	float (*cost)(float, float);
	float (*d_cost)(float, float);
	const char *name;
} cost_function;

cost_function* new_cost_function(std::string loss)
{
	cost_function *p = new cost_function;
	if(loss.compare(cross_entropy::name)==0) { p->cost = &cross_entropy::cost; p->d_cost = &cross_entropy::d_cost; p->name=cross_entropy::name;return p;}
	if(loss.compare(mse::name)==0) { p->cost = &mse::cost; p->d_cost = &mse::d_cost; p->name=mse::name; return p;}
	//if(loss.compare(triplet_loss::name)==0) { p->E = &triplet_loss::E; p->dE = &triplet_loss::dE; p->name=triplet_loss::name; return p;}
	delete p;
	return NULL;
}

cost_function* new_cost_function(const char *type)
{
	std::string loss(type);
	return new_cost_function(loss);
}

}