#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;


/******** Declaring Cuda Functions ********/


// Function for Hashing 32-bit integer
__device__ uint32_t hash(uint32_t key);

// Function to determine number of blocks and threads
__host__ void getBlocksThreads(int *blocks, int *threads, int entries);

// Function for reshaping hashmap
__global__ void reshape(HashMap newHM, HashMap oldHM);

/******** HashMap Methods ********/


/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) 
	: capacity(size), entries(0)
{
	glbGpuAllocator->_cudaMalloc((void **) &hashMap, this->capacity * sizeof(Entry));
	cudaCheckError();
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree((void *) hashMap);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	/* Getting number of blocks and threads */
	int blocks, threads;
	getBlocksThreads(&blocks, &threads, capacity);

	/* Alloccing new hashmap */
	HashMap newHashMap;
	glbGpuAllocator->_cudaMalloc((void **) &newHashMap, numBucketsReshape * sizeof(Entry));
	cudaCheckError();
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	return false;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	return NULL;
}


/******** Defining Cuda Functions ********/


__device__ uint32_t hash(uint32_t key) {
	unsigned long hash = 5381;
	int c;

	while (key != 0) {
		c = key % 10;
		hash = ((hash << 5) + hash) + c; /* hash * 33  + c */

		key /= 10;
	}

	return hash;
}

__host__ void getBlocksThreads(int *blocks, int *threads, int entries) {
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, 0);
	cudaCheckError();

	*threads = devProps.maxThreadsPerBlock;
	*blocks = entries / (*threads) + ((entries % (*threads) == 0 ) ? 0 : 1);
}