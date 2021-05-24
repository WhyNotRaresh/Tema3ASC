#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}

/**
 * Hashmap entry type
 */
#define INVALID_KEY		UINT32_MAX

typedef struct Entry {
	uint32_t key, value;

	Entry() : key(INVALID_KEY), value(0) {}
	Entry(uint32_t k, uint32_t v) : key(k), value(v) {}
} Entry, * HashMap;

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
private: 
	HashMap hashMap;
	int capacity;
	int entries;

public:
	GpuHashTable(int size);
	~GpuHashTable();

	void reshape(int sizeReshape);
		
	bool insertBatch(int *keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);
	
};

#endif

