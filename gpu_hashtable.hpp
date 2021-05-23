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
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
private: 
	typedef struct {
		uint32_t key, value;
	} Entry, *HashMap;

	HashMap hashMap;
	unsigned long capacity;
	unsigned long entries;

public:
	GpuHashTable(int size);
	~GpuHashTable();

	void reshape(int sizeReshape);
		
	bool insertBatch(int *keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);
	
};

#endif

