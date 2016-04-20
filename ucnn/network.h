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
//    network.h: The main artificial neural network graph for uCNN
//
// ==================================================================== uCNN ==

#pragma once

#include <random>
#include <string>
#include <iostream> // cout
#include <sstream>
#include <map>
#include <vector>

#include "activation.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"

#ifdef _WIN32
    #include <windows.h>
    void sleep(unsigned milliseconds){ Sleep(milliseconds);}
#else
    #include <unistd.h>
    void sleep(unsigned milliseconds) {usleep(milliseconds * 1000);}
#endif

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

//----------------------------------------------------------------------
// net - class that holds all the layers and connection information
//	- runs forward prediction
//  - with INCLUDE_TRAINING_CODE define can run backpropogation

class network
{
	
	int _size;  // output size
	int _thread_count; // determines number of layer sets (copys of layers)
	int _batch_size;   // determines number of dW sets 
	float _skip_energy_level;
	std::vector <float> _running_E;
	optimizer *_optimizer;
	const int MAIN_LAYER_SET;
	const int BATCH_RESERVED, BATCH_FREE, BATCH_COMPLETE;
#ifdef UCNN_OMP
	omp_lock_t _lock_batch;
	void lock_batch() {omp_set_lock(&_lock_batch);}
	void unlock_batch() {omp_unset_lock(&_lock_batch);}
	void init_lock() {omp_init_lock(&_lock_batch);}
	void destroy_lock() {omp_destroy_lock(&_lock_batch);}
	int get_thread_num() {return omp_get_thread_num();}
#else
	void lock_batch() {}
	void unlock_batch() {}
	void init_lock(){}
	void destroy_lock() {}
	int get_thread_num() {return 0;}
#endif

public:	

	// here we have multiple sets of the layers to allow threading and batch processing 
	std::vector< std::vector<base_layer *>> layer_sets;
	// name to index of layers for layer management
	std::map<std::string, int> layer_map; 
	// these are the weights between layers (not stored in the layer cause I look at them as the graph in the net)
	std::vector<matrix *> W;
	double _running_sum_E;
	std::vector< std::vector<matrix>> dW_sets; // only for training, will have _batch_size of these
	std::vector< std::vector<matrix>> dbias_sets; // only for training, will have _batch_size of these
	std::vector< unsigned char > batch_open; // only for training, will have _batch_size of these
	
	std::vector<std::pair<std::string,std::string>> layer_graph;

	network(const char* opt_name=NULL):MAIN_LAYER_SET(0), BATCH_FREE(0), BATCH_RESERVED(1), BATCH_COMPLETE(2), _thread_count(1), _skip_energy_level(0.f), _batch_size(1) //, MAX_UCNN_BATCH(1) 
	{ 
		_size=0;  
		_optimizer = new_optimizer(opt_name);
		//std::vector<base_layer *> layer_set;
		//layer_sets.push_back(layer_set);
		layer_sets.resize(1);
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size);
		_running_sum_E = 0.;
		init_lock();

#ifdef USE_AF
		af::setDevice(0);
        af::info();
#endif
	}
	
	~network() 
	{
		clear(); 
		if(_optimizer) delete _optimizer; 
		destroy_lock();	
	}

	void clear()
	{
		for(int i=0; i<(int)layer_sets.size(); i++)
		{
			__for__(auto l __in__ layer_sets[i]) delete l;
			layer_sets.clear();
		}
		layer_sets.clear();
		__for__(auto w __in__ W) delete w;  
		W.clear();
		layer_map.clear();
		layer_graph.clear();
	}

	// output size of final layer;
	int out_size() {return _size;}

	// get input size 
	bool get_input_size(int *w, int *h, int *c)
	{
		if(layer_sets[MAIN_LAYER_SET].size()<1) return false; 
		*w=layer_sets[MAIN_LAYER_SET][0]->node.cols;*h=layer_sets[MAIN_LAYER_SET][0]->node.rows;*c=layer_sets[MAIN_LAYER_SET][0]->node.chans;
		return true;
	}

	// sets up number of layer copies to run over multiple threads
	void allow_threads(int threads=1)
	{
		if(threads<1) _thread_count=1; else _thread_count=threads;
		int layer_cnt = (int)layer_sets.size();

		if (layer_cnt<_thread_count) 
		{
			layer_sets.resize(_thread_count);
		}
		// ToDo: add shrink back /  else if(layer_cnt>_thread_count)
		sync_layer_sets();
	}

	// when using threads, need to get bias data synched between all layer sets, call after bias update in main layer set
	void sync_layer_sets()
	{
		for(int i=1; i<(int)layer_sets.size();i++)
		{
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
			{
				for(int k=0; k<layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++) 
					(layer_sets[i])[j]->bias.x[k]=(layer_sets[MAIN_LAYER_SET])[j]->bias.x[k];
			}
		}
	}

	// sets up number of mini batches (storage for sets of weight deltas)
	// for now we will not persist these dW in memory, 
	void allow_mini_batches(int batch_cnt)
	{
		if(batch_cnt<1) batch_cnt=1;
		_batch_size = batch_cnt;
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size); reset_mini_batch();
		//_batch_index=0;
	}
	// return index of next free batch
	// or returns -2 if no free batches - all complete (need a sync call)
	// or returns -1 if no free batches - some still in progress (wait to see if one frees)
	int get_next_open_batch()
	{
		int reserved=0;
		int filled=0;
		for(int i=0; i<batch_open.size(); i++)
		{
			if(batch_open[i]==BATCH_FREE) return i;
			if(batch_open[i]==BATCH_RESERVED) reserved++;
			if(batch_open[i]==BATCH_COMPLETE) filled++;
		}
		if(reserved>0) return -1; // all filled but wainting for reserves
		if(filled==batch_open.size()) return -2; // all filled and complete
		throw;
			//return -3; // ?
	}

	void reset_mini_batch()
	{
		memset(batch_open.data(),BATCH_FREE,batch_open.size());
	}

	// apply all weights to first set of dW, then apply to model weights 
	void sync_mini_batch(int last_batch_index=-1) 
	{
		// need to ensure no batches in progress (reserved)
		int next = get_next_open_batch();
		if(next==-1) throw;


		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
		
		//if(last_batch_index==-1) last_batch_index=_batch_index-1;
		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		base_layer *layer;// = layer_sets[MAIN_LAYER_SET][layer_cnt-1];

		// sum contributions 
		for(int k=layer_cnt-1; k>=0; k--)	
		{
			layer = layer_sets[MAIN_LAYER_SET][k];
			__for__ (auto &link __in__ layer->backward_linked_layers)
			{
				int w_index = (int)link.first;

				if(batch_open[0]==BATCH_FREE) dW_sets[0][w_index].fill(0);
				
				for(int b=1; b< _batch_size; b++)
				{
					if(batch_open[b]==BATCH_COMPLETE)
						dW_sets[0][w_index]+=dW_sets[b][w_index];
				}
			}
			if( dynamic_cast<convolution_layer*> (layer) != NULL)  continue;
			
			if(batch_open[0]==BATCH_FREE) dbias_sets[0][k].fill(0);
			for(int b=1; b< _batch_size; b++)
			{
				if(batch_open[b]==BATCH_COMPLETE)
					dbias_sets[0][k]+=dbias_sets[b][k];
			}
		}

		// update weights
		for(int k=layer_cnt-1; k>=0; k--)	
		{
			layer = layer_sets[MAIN_LAYER_SET][k];
			__for__ (auto &link __in__ layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				if(dW_sets[0][w_index].size()>0)
					_optimizer->increment_w( W[w_index],w_index,  dW_sets[0][w_index]);  // -- 10%
			}
			if( dynamic_cast<convolution_layer*> (layer) != NULL)  continue;
			for(int j=0; j<layer->bias.size(); j++)
				layer->bias.x[j] -= dbias_sets[0][k].x[j]* _optimizer->learning_rate;
		}

		// start over
		reset_mini_batch();
		//_batch_index=0;
		sync_layer_sets();

	}

	// used to remove mean from weight
	void normalize_weights()
	{
		__for__(auto w __in__ W) 
		{
			if(w->chans>1) 
			w->remove_mean();
	//		if(w->chans>1) 	for(int c=0; c<w->chans; c++) w->remove_mean(c);
		}
	}

	// used to add some noise to weights
	void heat_weights(float std=0.001f)
	{
		static std::mt19937 gen(1);
		//std::uniform_real_distribution<float> dst(-std, std);
		std::normal_distribution<float> dst(0, std);
		__for__(auto w __in__ W) 
			for(int c=0; c<w->size(); c++) 
				w->x[c]*=(1.f+dst(gen));  //w->x[c]+=dst(gen);
	}

	// used to push a layer back in the ORDERED list of layers
	// if connect_all() is used, then the order of the push_back is used to connect the layers
	// when forward or backward propogation, this order is used for serialized order of calculations 
	bool push_back(const char *layer_name, const char *layer_config)
	{
		if(layer_map[layer_name]!=NULL) return false;
		base_layer *l=new_layer(layer_name, layer_config);
		// set map to index

		// make sure there is a 'set' to add layers to
		if(layer_sets.size()<1)
		{
			std::vector<base_layer *> layer_set;
			layer_sets.push_back(layer_set);
		}
		allow_threads(_thread_count);

		layer_map[layer_name] = (int)layer_sets[MAIN_LAYER_SET].size();
		layer_sets[MAIN_LAYER_SET].push_back(l);
		// upadate as potential last layer - so it sets the out size
		_size=l->fan_size();
		for(int i=1; i<(int)layer_sets.size();i++)
			layer_sets[i].push_back(new_layer(layer_name, layer_config));
		return true;
	}

	// connect 2 layers together and initialize weights
	void connect(const char *layer_name_top, const char *layer_name_bottom) 
	{
		size_t i_top=layer_map[layer_name_top];
		size_t i_bottom=layer_map[layer_name_bottom];

		base_layer *l_top= layer_sets[MAIN_LAYER_SET][i_top];
		base_layer *l_bottom= layer_sets[MAIN_LAYER_SET][i_bottom];
		
		int w_i=(int)W.size();
		matrix *w = l_bottom->new_connection(*l_top, w_i);
		W.push_back(w);
		layer_graph.push_back(std::make_pair(layer_name_top,layer_name_bottom));
		// need to build connections for other batches/threads
		for(int i=1; i<(int)layer_sets.size(); i++)
		{
			l_top= layer_sets[i][i_top];
			l_bottom= layer_sets[i][i_bottom];
			delete l_bottom->new_connection(*l_top, w_i);
		}

		// we need to let optimizer prepare space for stateful information 
		if (_optimizer)	_optimizer->push_back(w->cols, w->rows, w->chans);

		int fan_in=l_bottom->fan_size();
		int fan_out=l_top->fan_size();

		// ToDo: this may be broke when 2 layers connect to one. need to fix (i.e. resnet)
		// after all connections, run through and do weights with correct fan count

		// initialize weights
		static std::mt19937 gen(1);
		if(strcmp(l_bottom->p_act->name,"tanh")==0)
		{
			// xavier : for tanh
			float weight_base = (float)(std::sqrt(6./( (double)fan_in+(double)fan_out)));
//			float weight_base = (float)(std::sqrt(.25/( (double)fan_in)));
			std::uniform_real_distribution<float> dst(-weight_base, weight_base);
			for(int i=0; i<w->size(); i++) w->x[i]=dst(gen);
		}
		else if(strcmp(l_bottom->p_act->name,"sigmoid")==0)
		{
			// xavier : for sigmoid
			float weight_base = 4.f*(float)(std::sqrt(6./( (double)fan_in+(double)fan_out)));
			std::uniform_real_distribution<float> dst(-weight_base, weight_base);
			for(int i=0; i<w->size(); i++) w->x[i]=dst(gen);
		}
		else if((strcmp(l_bottom->p_act->name,"lrelu")==0) || (strcmp(l_bottom->p_act->name,"relu")==0)
			|| (strcmp(l_bottom->p_act->name,"vlrelu")==0) || (strcmp(l_bottom->p_act->name,"elu")==0))
		{
			// he : for relu
			float weight_base = (float)(std::sqrt(2./(double)fan_in));
			std::normal_distribution<float> dst(0, weight_base);
			for(int i=0; i<w->size(); i++) w->x[i]=dst(gen);
		}
		else
		{
			// lecun : orig
			float weight_base = (float)(std::sqrt(1./(double)fan_in));
			std::uniform_real_distribution<float> dst(-weight_base, weight_base);
			for(int i=0; i<w->size(); i++) w->x[i]=dst(gen);
		}
	}

	// automatically connect all layers in the order they were provided 
	// easy way to go, but can't deal with highway/resnet/inception types of architectures
	void connect_all()
	{	
		for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size()-1; j++) connect(layer_sets[MAIN_LAYER_SET][j]->name.c_str(), layer_sets[MAIN_LAYER_SET][j+1]->name.c_str());
	}

	// do not delete or modify the returned pointer. it is a live pointer to the last layer in the network
	float* predict(const float *in, int _thread_number=-1) 
	{

		if(_thread_number<0) _thread_number=get_thread_num();
		if(_thread_number>_thread_count) throw;

		const int thread_number=_thread_number;

		if(thread_number>=(int)layer_sets.size()) return NULL;
		// add bias to outputs - just init with bias
		__for__(auto layer __in__ layer_sets[thread_number]) layer->node.fill(0.f);

		// first layer assumed input. copy it (but could just attach to save time)
		memcpy(layer_sets[thread_number][0]->node.x, in, sizeof(float)*layer_sets[thread_number][0]->node.size());

		__for__(auto layer __in__ layer_sets[thread_number])
		{
			// add bias and activate these outputs (should all be summed up) from the last set of layers
			// we activate before using the layer instead of after getting the signal to allow for more flexibility in the topology
			layer->activate_nodes(); 

			// send output signal downstream (note in this code 'top' is input layer, 'bottom' is output - bucking tradition
			__for__ (auto &link __in__ layer->forward_linked_layers)
			{
				// instead of having a list of paired connections, just use the shape of W to determine connections
				// this is harder to read, but requires less lookups
				// the 'link' variable is a std::pair created during the connect() call for the layers
				int connection_index = link.first; 
				base_layer *p_bottom = link.second;
				// weighted distribution of the signal to layers under it
				p_bottom->accumulate_signal(*layer, *W[connection_index],thread_number);
			}

		}
		// return pointer to result from last layer
		return layer_sets[thread_number][layer_sets[thread_number].size()-1]->node.x;
	}

	// write parameters to fie
	// note that this does not persist intermediate training information that could be needed to 'pickup where you left off'
	// but you can still load and resume training
	bool write(std::string filename, bool binary=false) { return write(std::ofstream(filename), binary); }
	bool write(std::ofstream &ofs, bool binary=false) 
	{
		// save layers
		ofs<<(int)layer_sets[MAIN_LAYER_SET].size()<<std::endl;
		for(int j=0; j<(int)layer_sets[0].size(); j++)
			ofs<<layer_sets[MAIN_LAYER_SET][j]->name<<std::endl<<layer_sets[MAIN_LAYER_SET][j]->get_config_string();

		// save graph
		ofs<<(int)layer_graph.size()<<std::endl;
		for(int j=0; j<(int)layer_graph.size(); j++)
			ofs<<layer_graph[j].first << std::endl << layer_graph[j].second << std::endl;

		if(binary)
		{
			ofs<<(int)1<<std::endl;
			// binary version to save space if needed
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				ofs.write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
			// save weights
			for(int j=0; j<(int)W.size(); j++)
				ofs.write((char*) W[j]->x, W[j]->size()*sizeof(float));
		}
		else
		{
			ofs<<(int)0<<std::endl;
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
			{
				for(int k=0; k<layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  ofs << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << " ";
				ofs <<std::endl;
			}
			// save weights
			for(int j=0; j<(int)W.size(); j++)
			{
				for(int i=0; i<W[j]->size(); i++) ofs << W[j]->x[i] << " ";
				ofs <<std::endl;
			}
		}
		ofs.flush();
		
		return true;
	}

	// read network from a file
	bool read(std::string filename) {	std::ifstream fs(filename); if(fs.is_open()) return read(std::ifstream(filename)); else return false;}
	bool read(std::istream &ifs)
	{
		if(!ifs.good()) return false;
		// read layer def
		int layer_count;
		ifs>>layer_count;

		std::string s;
		getline(ifs,s); // get endline
		std::string layer_name;
		std::string layer_def;
		for (auto i=0; i<layer_count; i++)
		{
			getline(ifs,layer_name);
			getline(ifs,layer_def);
			push_back(layer_name.c_str(),layer_def.c_str());
		}

		// read graph
		int graph_count;
		ifs>>graph_count;
		getline(ifs,s); // get endline

		std::string layer_name1;
		std::string layer_name2;
		for (auto i=0; i<graph_count; i++)
		{
			getline(ifs,layer_name1);
			getline(ifs,layer_name2);
			connect(layer_name1.c_str(),layer_name2.c_str());
		}

		int binary;
		ifs>>binary;
		getline(ifs,s); // get endline

		// binary version to save space if needed
		if(binary)
		{
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				ifs.read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
			for(int j=0; j<(int)W.size(); j++)
				ifs.read((char*) W[j]->x, W[j]->size()*sizeof(float));
		}
		else // text version
		{
			// read bias
			for(int j=0; j<layer_count; j++)
			{
				for(int k=0; k<layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  ifs >> layer_sets[MAIN_LAYER_SET][j]->bias.x[k];
				getline(ifs,s); // get endline
			}

			// read weights
			for (auto j=0; j<(int)W.size(); j++)
			{
				for(int i=0; i<W[j]->size(); i++) ifs >> W[j]->x[i] ;
				getline(ifs,s); // get endline
			}
		}
		// copies batch=0 stuff to other batches
		sync_layer_sets();

		return true;
	}

#ifdef INCLUDE_TRAINING_CODE

	float get_learning_rate() {if(!_optimizer) throw; return _optimizer->learning_rate;}
	void set_learning_rate(float alpha) {if(!_optimizer) throw; _optimizer->learning_rate=alpha;}
	void reset() {if(!_optimizer) throw; _optimizer->reset();}
	float get_smart_train_level() {return _skip_energy_level;}
	void set_smart_train_level(float skip_energy_level) {_skip_energy_level=skip_energy_level;}
	// goal here is to update the weights W. 
	// use w_new = w_old - alpha dE/dw
	// E = sum: 1/2*||y-target||^2
	// note y = f(x*w)
	// dE = (target-y)*dy/dw = (target-y)*df/dw = (target-y)*df/dx* dx/dw = (target-y) * df * y_prev  

// ===========================================================================
// training part
// ===========================================================================
	bool train(float *in, float *target, int _thread_number = -1)
	{
		if (_optimizer == NULL) throw;
		//_optimizer->reset();

		if (_thread_number < 0) _thread_number = get_thread_num();

		if (_thread_number > _thread_count) throw;

		const int thread_number = _thread_number;

		lock_batch();

		int my_batch_index = -3;
		while (my_batch_index < 0)
		{
			my_batch_index = get_next_open_batch();

			if (my_batch_index >= 0)
			{
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				break;
			}
			else if (my_batch_index == -2)
			{
				sync_mini_batch(); // resets _batch_index to 0
				my_batch_index = get_next_open_batch();
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				break;
			}
			// need to wait for ones in progress to finish
			unlock_batch();
			sleep(1);
			lock_batch();
		}

		// run through forward to get nodes activated
		predict(in, thread_number);  //--- 10% of time

		// set all deltas to zero
		__for__(auto layer __in__ layer_sets[thread_number]) layer->delta.fill(0.f);

		int layer_cnt = (int)layer_sets[thread_number].size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		base_layer *layer = layer_sets[thread_number][layer_cnt - 1];
		float E = 0;
		//std::vector<float> vE(layer->node.size());
		for (int j = 0; j < layer->node.size(); j++)
		{
			//layer->delta.x[j] =  cross_entropy::dE(layer->node.x[j],target[j]);
			layer->delta.x[j] = mse::dE(layer->node.x[j], target[j]);
			float e = mse::E(layer->node.x[j], target[j]);
			//vE[j]=e;
			E += e;
			//std::cout << "<" << layer->node.x[j] << ">";
		}
		//std::sort(vE.begin(), vE.end());
		//float dE = vE[1]-vE[0];
		E /= (float)layer->node.size();
		//std::cout << "skip " << _skip_energy_level <<", E " << E << ",";
		if (0)///(_skip_energy_level > 0 && E > 0)
		{
#ifdef UCNN_OMP	
			#pragma omp critical
#endif
		{
			std::cout << "E " << E << ",";
				_running_E.push_back(E);
				int s = _running_E.size();
				if (s >= 1000)
				{
					std::cout << ">100";
					float oldE = _running_E.front();
					//std::cout << "b " << _running_E[10] << "," << _running_E[0] << std::endl;
				
					std::sort(_running_E.begin(), _running_E.end());
					int index = s *9 / 10;
					std::cout << "e " << _running_E[s-1] << "," << _running_E[0] << std::endl;
					if(_running_E[index]>0)
						if (_skip_energy_level < _running_E[index]) _skip_energy_level = _running_E[index];
					_running_E.clear();
				}

			}
		}
		
		//std::cout << dE << "," << E<< "\n";
		if(E>0 && E<_skip_energy_level)
		{
			lock_batch();
			batch_open[my_batch_index]=BATCH_FREE;
			unlock_batch();
			return false;
		}


		// update hidden layers
		// start at lower layer and push information up to previous layer
		for(int k=layer_cnt-1; k>=0; k--)
		{
			layer = layer_sets[thread_number][k];
			// all the signals should be summed up to this layer by now, so we go through and take the grad of activiation
			int nodes=layer->node.size();
			for (int i=0; i< nodes; i++) layer->delta.x[i] *= layer->df(layer->node.x, i, nodes);

			// now pass that signal upstream
			__for__ (auto &link __in__ layer->backward_linked_layers) // --- 50% of time this loop
			{
				base_layer *p_top=link.second;
				// note all the delta[connections[i].second] should have been calculated by time we get here
				layer->distribute_delta(*p_top, *W[link.first],_thread_count);
			}
		}
		

		// update weights - shouldn't matter the direction we update these 
		// we can stay in backwards direction...
		// it was not faster to combine distribute_delta and increment_w into the same loop
		// matrix dw(1,1,1);
		int size_W=(int)W.size();
		dW_sets[my_batch_index].resize(size_W);
		dbias_sets[my_batch_index].resize(layer_cnt);
		for(int k=layer_cnt-1; k>=0; k--)	
		{
			layer = layer_sets[thread_number][k];
			
			__for__ (auto &link __in__ layer->backward_linked_layers)
			{
				base_layer *p_top =link.second;
				int w_index = (int)link.first;
				//if (dynamic_cast<max_pooling_layer*> (layer) != NULL)  continue;
				layer->calculate_dw(*p_top, dW_sets[my_batch_index][w_index],_thread_count);// --- 20%
				// moved this out to sync_mini_batch();
				//_optimizer->increment_w( W[w_index],w_index, dW_sets[_batch_index][w_index]);  // -- 10%
			}
			if( dynamic_cast<convolution_layer*> (layer) != NULL)  continue;
	
			dbias_sets[my_batch_index][k] = layer->delta;
		}
		// if all batches finished, update weights

		lock_batch();
		batch_open[my_batch_index]=BATCH_COMPLETE;
		int next_index=get_next_open_batch();
		if(next_index==-2) // all complete
			sync_mini_batch(); // resets _batch_index to 0
		unlock_batch();

		

		return true;
	}
	
#else

	float get_learning_rate() {return 0;}
	void set_learning_rate(float alpha) {}
	void train(float *in, float *target){}
	void reset() {}
	float get_smart_train_level() {return 0;}
	void set_smart_train_level(float skip_energy_level) {}

#endif

};

}
