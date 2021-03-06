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
__global__ void insertIntoHashMap(HashTable hashMap, Entry *newEntries, int *updates, int noEntries, int capacity);

// Function for searching the hashmap
__global__ void getFromHashMap(HashTable hashMap, int *keys, int *retValues, int noKeys, int capacity);


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
	getBlocksThreads(&blocks, &threads, capacity);

	/* Alloccing new hashmap */
	HashTable newHashMap;
	size_t total_bytes =  numBucketsReshape * sizeof(Entry);

	glbGpuAllocator->_cudaMalloc((void **) &newHashMap, total_bytes);
	cudaCheckError();

	cudaMemset((void *) newHashMap, 0, total_bytes);
	cudaCheckError();

	/* Writing to new hashmap */
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

	/* Setting new entries */
	Entry *newEntries;
	glbGpuAllocator->_cudaMallocManaged((void **) &newEntries, numKeys * sizeof(Entry));
	cudaCheckError();

	for (int i = 0; i < numKeys; i++) {
		newEntries[i] = Entry(keys[i], values[i]);	
	}

	/* Reshaping HashMap */
	if ((entries + numKeys) / ((float) capacity) >= 0.9f) {
		reshape((int) ((entries + numKeys) / 0.7f));
	}

	/* Number of updated keys */
	int *keyUpdates;
	glbGpuAllocator->_cudaMallocManaged((void **) &keyUpdates, sizeof(int));
	cudaCheckError();

	/* Inserting entries */
	insertIntoHashMap<<<blocks, threads>>>(hashMap, newEntries, keyUpdates, numKeys, capacity);
	cudaDeviceSynchronize();
	cudaCheckError();

	entries += numKeys - (*keyUpdates);

	glbGpuAllocator->_cudaFree(newEntries);
	cudaCheckError();
	glbGpuAllocator->_cudaFree(keyUpdates);
	cudaCheckError();

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int blocks, threads;
	getBlocksThreads(&blocks, &threads, numKeys);

	int total_bytes = numKeys * sizeof(int);
	
	/* Keys to search for */
	int *deviceKeys;
	glbGpuAllocator->_cudaMalloc((void **) &deviceKeys, total_bytes);
	cudaCheckError();
	cudaMemcpy(deviceKeys, keys, total_bytes, cudaMemcpyHostToDevice);
	cudaCheckError();

	/* Values to be returned */
	int *retValues;
	glbGpuAllocator->_cudaMallocManaged((void **) &retValues, total_bytes);
	cudaCheckError();

	getFromHashMap<<<blocks, threads>>>(hashMap, deviceKeys, retValues, numKeys, capacity);
	cudaDeviceSynchronize();
	cudaCheckError();

	glbGpuAllocator->_cudaFree((void *) deviceKeys);
	cudaCheckError();

	return retValues;
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

__global__ void insertIntoHashMap(HashTable hashMap, Entry *newEntries, int *updates, int noEntries, int capacity) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < noEntries) {
		uint32_t hash = hashKey(newEntries[idx].key) % capacity;
		uint32_t oldKey = atomicCAS(&(hashMap[hash].key), KEY_INVALID, newEntries[idx].key);

		while(oldKey != KEY_INVALID && oldKey != newEntries[idx].key) {
			hash = (hash + 1) % capacity;
			oldKey = atomicCAS(&(hashMap[hash].key), KEY_INVALID, newEntries[idx].key);
		}

		if (oldKey == newEntries[idx].key) {
			atomicAdd(updates, 1);
		}

		hashMap[hash].value = newEntries[idx].value;
	}
}

__global__ void getFromHashMap(HashTable hashMap, int *keys, int *retValues, int noKeys, int capacity) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < noKeys) {
		uint32_t hash = hashKey(keys[idx]) % capacity;

		while (hashMap[hash].key != keys[idx] && hashMap[hash].key != KEY_INVALID) {
			hash = (hash + 1) % capacity;
		}

		retValues[idx] = hashMap[hash].value;
	}
}
