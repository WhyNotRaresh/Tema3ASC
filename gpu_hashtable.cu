#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <stdio.h>

#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

/**
 * Function for Hashing 32-bit integer
 */
static uint32_t hash(uint32_t key) {
	unsigned long hash = 5381;
	int c;

	while (key != 0) {
		c = key % 10;
		hash = ((hash << 5) + hash) + c; /* hash * 33  + c */

		key /= 10;
	}

	return hash;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) 
	: capacity(size), entries(0)
{
	cudaError_t err = glbGpuAllocator->_cudaMalloc((void **) &hashMap, this->capacity * sizeof(Entry));
	if (err != cudaSuccess) {
		fprinf(stderr, "cudaMalloc fail on init");
	}
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