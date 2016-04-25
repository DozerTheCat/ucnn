CC=g++
CFLAGS= -I../ucnn/ -std=c++11 -O3
CFLAGS_OMP= -I../ucnn/ -std=c++11 -O3 -DUCNN_OPM
UCNN_HEADERS =  ../ucnn/ucnn.h ../ucnn/network.h ../ucnn/layer.h\
../ucnn/activation.h ../ucnn/cost.h ../ucnn/optimizer.h ../ucnn/core_math.h

UCNN_HEADERS_OMP =  ../ucnn/ucnn_omp.h ../ucnn/ucnn.h ../ucnn/network.h ../ucnn/layer.h\
../ucnn/activation.h ../ucnn/cost.h ../ucnn/optimizer.h ../ucnn/core_math.h

all: test test_omp train train_omp

test: test.cpp $(UCNN_HEADERS)
	$(CC) $(CFLAGS) test.cpp $(UCNN_HEADERS) -o test

test_omp: test_omp.cpp $(UCNN_HEADERS_OMP)
	$(CC) $(CFLAGS) test_omp.cpp $(UCNN_HEADERS) -fopenmp -o test_omp

train: train.cpp $(UCNN_HEADERS)
	$(CC) $(CFLAGS) train.cpp $(UCNN_HEADERS) -o train

train_omp: train_omp.cpp $(UCNN_HEADERS_OMP)
	$(CC) $(CFLAGS) train_omp.cpp $(UCNN_HEADERS_OMP) -fopenmp -o train_omp

clean:
	-rm -f test
	-rm -f test_omp
	-rm -f train
	-rm -f train_omp
