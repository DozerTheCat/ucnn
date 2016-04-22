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

#include "ucnn.h"
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


// returns Energy (euclidian distance / 2) and max index
float match_labels(const float *out, const float *target, const int size, int *best_index = NULL)
{
	float E = 0;
	int max_j = 0;
	for (int j = 0; j<size; j++)
	{
		E += (out[j] - target[j])*(out[j] - target[j]);
		if (out[max_j]<out[j]) max_j = j;
	}
	if (best_index) *best_index = max_j;
	E *= 0.5;
	return E;
}

int max_index(const float *out, const int size)
{
	int max_j = 0;
	for (int j = 0; j<size; j++)
	{
		if (out[max_j]<out[j]) max_j = j;
	}
	return max_j;
}

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
	bool _smart_train;
	std::vector <float> _running_E;
	double _running_sum_E;

	optimizer *_optimizer;
	const int MAIN_LAYER_SET=0;
	const unsigned char BATCH_RESERVED=1, BATCH_FREE=0, BATCH_COMPLETE=2;
	const int BATCH_FILLED_COMPLETE=-2, BATCH_FILLED_IN_PROCESS=-1;
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

	int train_correct;
	int train_skipped;
	int train_updates;
	int train_samples;
	int epoch_count;
	float old_estimated_accuracy;
	float estimated_accuracy;

	// here we have multiple sets of the layers to allow threading and batch processing 
	std::vector< std::vector<base_layer *>> layer_sets;
	// name to index of layers for layer management
	std::map<std::string, int> layer_map; 
	// these are the weights between layers (not stored in the layer cause I look at them as the graph in the net)
	std::vector<matrix *> W;
	std::vector< std::vector<matrix>> dW_sets; // only for training, will have _batch_size of these
	std::vector< std::vector<matrix>> dbias_sets; // only for training, will have _batch_size of these
	std::vector< unsigned char > batch_open; // only for training, will have _batch_size of these
	
	std::vector<std::pair<std::string,std::string>> layer_graph;

	network(const char* opt_name=NULL): _thread_count(1), _skip_energy_level(0.f), _batch_size(1) //, MAX_UCNN_BATCH(1) 
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
		train_correct = 0;
		train_samples = 0;
		train_skipped = 0;
		epoch_count = 0; 
		train_updates = 0;
		estimated_accuracy = 0;
		old_estimated_accuracy = 0;
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
	void set_mini_batch_size(int batch_cnt)
	{
		if(batch_cnt<1) batch_cnt=1;
		_batch_size = batch_cnt;
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size); reset_mini_batch();
		//_batch_index=0;
	}
	int get_mini_batch_size() {return _batch_size;}

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
		if(reserved>0) return BATCH_FILLED_IN_PROCESS; // all filled but wainting for reserves
		if(filled==batch_open.size()) return BATCH_FILLED_COMPLETE; // all filled and complete
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
		train_updates++; // sometimes may have no updates, .. so this is not exact
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
	int predict_class(const float *in, int _thread_number = -1)
	{
		const float* out = forward(in, _thread_number);
		return max_index(out, out_size());

	}
	float* forward(const float *in, int _thread_number=-1)
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

	int reserve_next_batch()
	{

		lock_batch();
		int my_batch_index = -3;
		while (my_batch_index < 0)
		{
			my_batch_index = get_next_open_batch();

			if (my_batch_index >= 0) // valid index
			{
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				return my_batch_index;
			}
			else if (my_batch_index == BATCH_FILLED_COMPLETE) // all index are complete
			{
				sync_mini_batch(); // resets _batch_index to 0
				my_batch_index = get_next_open_batch();
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				return my_batch_index;
			}
			// need to wait for ones in progress to finish
			unlock_batch();
			sleep(1);
			lock_batch();
		}
	}

	float get_learning_rate() {if(!_optimizer) throw; return _optimizer->learning_rate;}
	void set_learning_rate(float alpha) {if(!_optimizer) throw; _optimizer->learning_rate=alpha;}
	void reset() {if(!_optimizer) throw; _optimizer->reset();}
	bool get_smart_train() {return _smart_train;}
	void set_smart_train(bool _use_train) { _smart_train = _use_train;}
	float get_smart_train_level() { return _skip_energy_level; }
	void set_smart_train_level(float _level) { _skip_energy_level = _level; }
	
	// goal here is to update the weights W. 
	// use w_new = w_old - alpha dE/dw
	// E = sum: 1/2*||y-target||^2
	// note y = f(x*w)
	// dE = (target-y)*dy/dw = (target-y)*df/dw = (target-y)*df/dx* dx/dw = (target-y) * df * y_prev  

// ===========================================================================
// training part
// ===========================================================================

	void start_epoch(std::string loss_function="mse")
	{
		train_correct = 0;
		train_skipped = 0;
		train_updates = 0;
		train_samples = 0;
		if (estimated_accuracy > 0 && _smart_train && old_estimated_accuracy>0)
		{
			if ((old_estimated_accuracy - estimated_accuracy) == 0)
			{
				heat_weights();
			}
		}
		old_estimated_accuracy = estimated_accuracy;
		estimated_accuracy = 0;
		//_skip_energy_level = 0.05;
		_running_sum_E = 0;
	}
	
	void end_epoch()
	{
		// run leftovers
		sync_mini_batch();
		epoch_count++;

		estimated_accuracy = 100.f*train_correct / train_samples;
		estimated_accuracy = (float)((int)(0.995f*estimated_accuracy * 10)) / 10.f; // reduce precision

	}

	bool train_class(float *in, int label_index, int _thread_number = -1)
	{
		if (_optimizer == NULL) throw;
		//_optimizer->reset();

		if (_thread_number < 0) _thread_number = get_thread_num();

		if (_thread_number > _thread_count) throw;

		const int thread_number = _thread_number;


		// get next free mini_batch slot
		// this is tied to the current state of the model
		int my_batch_index = reserve_next_batch();

		// run through forward to get nodes activated
		forward(in, thread_number);  //--- 10% of time

		// set all deltas to zero
		__for__(auto layer __in__ layer_sets[thread_number]) layer->delta.fill(0.f);

		int layer_cnt = (int)layer_sets[thread_number].size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		base_layer *layer = layer_sets[thread_number][layer_cnt - 1];
		float E = 0;
		int max_j_out = 0;
		int max_j_target = label_index;
		
		// was passing this in, but may as well just create it 
		std::vector<float> target(layer->node.size(), -1);
		if(label_index>=0 && label_index<layer->node.size()) target[label_index] = 1;

		int layer_node_size = layer->node.size();
		for (int j = 0; j < layer_node_size; j++)
		{
			//layer->delta.x[j] = cross_entropy::dE(layer->node.x[j], target[j]);
			
			layer->delta.x[j] = mse::dE(layer->node.x[j], target[j])*layer->df(layer->node.x, j, layer_node_size);;
			float e = mse::E(layer->node.x[j], target[j]);

			if (layer->node.x[max_j_out] < layer->node.x[j]) max_j_out = j;
			// for better E maybe just look at 2 highest scores so zeros don't dominate 
			E += e;
		}

		E /= (float)layer->node.size();
		if (E != E)
		{
			std::cerr << "network blew up" << std::endl;
			throw;
		}
		
		// to do: put data in separate objects for each thread to reduct critcal section
#ifdef UCNN_OMP	
#pragma omp critical
#endif
		{
			train_samples++;

			if (max_j_target == max_j_out)
			{
				train_correct++;
			}

			if (_smart_train)
			{
				_running_E.push_back(E);
				_running_sum_E += E;
				const int SMART_TRAIN_SAMPLE_SIZE = 1000;

				int s = (int)_running_E.size();
				if (s >= SMART_TRAIN_SAMPLE_SIZE)
				{

					_running_sum_E /= (double)s;
					std::sort(_running_E.begin(), _running_E.end());
					float top_fraction = (float)_running_sum_E*10.f;
					if (top_fraction > 0.75f) top_fraction = 0.75f;
					if (top_fraction < 0.05f) top_fraction = 0.05f;
					int index = s - 1 - (int)(top_fraction*(s - 1));

					//std::cout << "e " << _running_E[s-1] << "," << _running_E[0] << std::endl;
					if (_running_E[index] > 0)
						_skip_energy_level = _running_E[index];

					_running_E.clear();
					//_running_sum_E = 0;
				}
			}
			if (E>0 && E<_skip_energy_level)
				train_skipped++;

		}  // omp critical




		if (E>0 && E<_skip_energy_level && _smart_train)
		{
			lock_batch();
			batch_open[my_batch_index] = BATCH_FREE;
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
			// already did last layer
			if(k<layer_cnt - 1)
				for (int i=0; i< nodes; i++) 
					layer->delta.x[i] *= layer->df(layer->node.x, i, nodes);

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
		if(next_index==BATCH_FILLED_COMPLETE) // all complete
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
	void set_smart_train_level(float _level) {}
	bool get_smart_train() { return false; }
	void set_smart_train(bool _use) {}

#endif

};

}
