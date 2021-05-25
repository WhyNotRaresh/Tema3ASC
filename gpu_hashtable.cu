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

// Function sets all hashmap entries to the given one
__global__ void setHashMap(HashTable hashMap, Entry entry, int capacity);

// Function for reshaping hashmap
__global__ void reshapeHashMap(HashTable newHM, HashTable oldHM, int newCap, int oldCap);

// Function for inserting into hashmap
__global__ void insertIntoHashMap(HashTable hashMap, Entry *newEntries, int *updates, int noEntries, int capacity);


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
	HashTable newHashMap;
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
	int blocks, threads;
	getBlocksThreads(&blocks, &threads, numKeys);

	/* Setting device entries */
	Entry *hostEntries, *deviceEntries;
	size_t total_bytes = numKeys * sizeof(Entry);

	hostEntries = (Entry *) malloc(total_bytes);
	if (hostEntries == NULL) {
		return false;
	}

	for (int i = 0; i < numKeys; i++) {
		hostEntries[i] = Entry(keys[i], values[i]);
	}
	
	glbGpuAllocator->_cudaMalloc((void **) &deviceEntries, total_bytes);
	cudaCheckError();
	cudaMemcpy(deviceEntries, hostEntries, total_bytes, cudaMemcpyHostToDevice);
	cudaCheckError();

	/* Number of updated keys */
	int *keyUpdates;
	glbGpuAllocator->_cudaMallocManaged((void **) &keyUpdates, sizeof(int));
	cudaCheckError();

	/* Inserting values */
	printf("%d, %d", blocks, threads);
	insertIntoHashMap<<<blocks, threads>>>(hashMap, deviceEntries, keyUpdates, numKeys, capacity);

	cudaDeviceSynchronize();
	cudaCheckError();
	printf("finished gpu work");

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

__global__ void setHashMap(HashTable hashMap, Entry entry, int capacity) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < capacity) {
		hashMap[idx] = entry;
	}
}

__global__ void reshapeHashMap(HashTable newHM, HashTable oldHM, int newCap, int oldCap) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < oldCap && idx < oldCap && oldHM[idx].key != INVALID_KEY) {
		uint32_t hash = hashKey(oldHM[idx].key) % newCap;

		while(atomicCAS(&(newHM[hash].key), INVALID_KEY, oldHM[idx].key) == INVALID_KEY) {
			hash = (++hash) % newCap;
		}

		newHM[hash].value = oldHM[idx].value;
	}
}

__global__ void insertIntoHashMap(HashTable hashMap, Entry *newEntries, int *updates, int noEntries, int capacity) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < noEntries && idx < capacity) {
		uint32_t hash = hashKey(newEntries[idx].key) % capacity;
		uint32_t oldKey = atomicCAS(&(hashMap[hash].key), INVALID_KEY, newEntries[idx].key);

		while(oldKey != INVALID_KEY && oldKey != newEntries[idx].key) {
			hash = (++hash) % capacity;
			oldKey = atomicCAS(&(hashMap[hash].key), INVALID_KEY, newEntries[idx].key);
		}

		if (oldKey == newEntries[idx].key) {
			atomicAdd(updates, 1);
		}

		hashMap[hash].value = newEntries[idx].value;
	}
}