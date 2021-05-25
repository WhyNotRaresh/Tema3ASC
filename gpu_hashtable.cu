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

// Function sets all hashmap entries to the given one
__global__ void setHashMap(HashMap hashMap, Entry entry, int capacity);

// Function for reshaping hashmap
__global__ void reshapeHashMap(HashMap newHM, HashMap oldHM, int newCap, int oldCap);


/******** HashMap Methods ********/


/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) 
	: capacity(size), entries(0)
{
	size_t total_bytes = this->capacity * sizeof(Entry);

	glbGpuAllocator->_cudaMalloc((void **) &hashMap, total_bytes);
	cudaCheckError();

	int blocks, threads;
	getBlocksThreads(&blocks, &threads, capacity);
	setHashMap<<<blocks, threads>>>(hashMap, Entry(), capacity);
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree((void *) hashMap);
	cudaCheckError();
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int blocks, threads;

	/* Alloccing new hashmap */
	HashMap newHashMap;
	size_t total_bytes =  numBucketsReshape * sizeof(Entry);

	glbGpuAllocator->_cudaMalloc((void **) &newHashMap, total_bytes);
	cudaCheckError();

	getBlocksThreads(&blocks, &threads, capacity);
	setHashMap<<<blocks, threads>>>(newHashMap, Entry(), numBucketsReshape);

	/* Writing to new hashmap */
	getBlocksThreads(&blocks, &threads, capacity);

	reshapeHashMap<<<blocks, threads>>>(newHashMap, hashMap, numBucketsReshape, capacity);
	cudaDeviceSynchronize();
	cudaCheckError();
	
	glbGpuAllocator->_cudaFree((void *) hashMap);
	cudaCheckError();
	hashMap = newHashMap;
	capacity = numBucketsReshape;
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

__global__ void setHashMap(HashMap hashMap, Entry entry, int capacity) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < capacity) {
		hashMap[idx] = entry;
	}
}

__global__ void reshapeHashMap(HashMap newHM, HashMap oldHM, int newCap, int oldCap) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx > oldCap) {
		return;
	}

	// TODO
}