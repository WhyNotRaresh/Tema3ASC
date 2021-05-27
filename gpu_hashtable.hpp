#ifndef _HASHCPU_
#define _HASHCPU_

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
typedef struct Entry {
	uint32_t key, value;

	Entry() : key(0), value(0) {}
	Entry(uint32_t k, uint32_t v) : key(k), value(v) {}

	inline bool operator==(const Entry& other) {
		return other.key == key;
	}

	inline bool operator!=(const Entry& other) {
		return !(other.key == key);
	}
} Entry, * HashTable;

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
private: 
	HashTable hashMap;
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

