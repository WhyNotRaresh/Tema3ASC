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
__device__ uint32_t hashKey(uint32_t key);

// Function to determine number of blocks and threads
__host__ void getBlocksThreads(int *blocks, int *threads, int entries);

// Function for reshaping hashmap
__global__ void reshapeHashMap(HashTable newHM, HashTable oldHM, int newCap, int oldCap);

// Function for inserting into hashmap
__global__ void insertIntoHashMap(HashTable hashMap, int *keys, int *values, int *updates, int noEntries, int capacity);


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

	cudaMemset((void *) hashMap, 0, total_bytes);
	cudaCheckError();
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
	HashTable newHashMap;
	size_t total_bytes =  numBucketsReshape * sizeof(Entry);

	glbGpuAllocator->_cudaMalloc((void **) &newHashMap, total_bytes);
	cudaCheckError();

	cudaMemset((void *) newHashMap, 0, total_bytes);
	cudaCheckError();

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
	int blocks, threads;
	getBlocksThreads(&blocks, &threads, numKeys);

	/* Setting device entries */
	int *deviceKeys, *deviceValues;

	glbGpuAllocator->_cudaMalloc((void **) &deviceKeys, numKeys * sizeof(int));
	cudaCheckError();
	glbGpuAllocator->_cudaMalloc((void **) &deviceValues, numKeys * sizeof(int));
	cudaCheckError();

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* Reshaping HashMap */
	if ((entries + numKeys) / ((float) capacity) >= 0.9f) {
		this->reshape((int) ((entries + numKeys) / 0.8f));
	}

	/* Number of updated keys */
	int *keyUpdates;
	glbGpuAllocator->_cudaMallocManaged((void **) &keyUpdates, sizeof(int));
	cudaCheckError();

	/* Inserting values */
	insertIntoHashMap<<<blocks, threads>>>(hashMap, deviceKeys, deviceValues, keyUpdates, numKeys, capacity);

	cudaDeviceSynchronize();
	cudaCheckError();

	entries += numKeys - (*keyUpdates);

	glbGpuAllocator->_cudaFree(deviceEntries);
	cudaCheckError();
	free(hostEntries);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	return NULL;
}


/******** Defining Cuda Functions ********/


__device__ uint32_t hashKey(uint32_t key) {
	unsigned long hash = 5381;
	int c;

	while (key != 0) {
		c = key % 10;
		hash = ((hash << 5) + hash) + c; /* hash * 33  + c */

		key /= 10;
	}

	return (uint32_t) hash;
}

__host__ void getBlocksThreads(int *blocks, int *threads, int entries) {
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, 0);
	cudaCheckError();

	*threads = devProps.maxThreadsPerBlock;
	*blocks = entries / (*threads) + ((entries % (*threads) == 0 ) ? 0 : 1);
}

__global__ void reshapeHashMap(HashTable newHM, HashTable oldHM, int newCap, int oldCap) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < oldCap && idx < oldCap && oldHM[idx].key != KEY_INVALID) {
		uint32_t hash = hashKey(oldHM[idx].key) % newCap;

		while(atomicCAS(&(newHM[hash].key), KEY_INVALID, oldHM[idx].key) == KEY_INVALID) {
			hash = (hash + 1) % newCap;
		}

		newHM[hash].value = oldHM[idx].value;
	}
}

__global__ void insertIntoHashMap(HashTable hashMap, int *keys, int *values, int *updates, int noEntries, int capacity) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < noEntries) {
		uint32_t hash = hashKey(keys[idx]) % capacity;
		int oldKey = atomicCAS(&(hashMap[hash].key), KEY_INVALID, keys[idx]);

		while(oldKey != KEY_INVALID && oldKey != keys[idx]) {
			hash = (hash + 1) % capacity;
			oldKey = atomicCAS(&(hashMap[hash].key), KEY_INVALID, keys[idx]);
		}

		if (oldKey == keys[idx]) {
			atomicAdd(updates, 1);
		}

		hashMap[hash].value = vales[idx];
	}
}
