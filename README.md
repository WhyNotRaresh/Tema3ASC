Nume: Badita Rares Octavian
Grupa: 333CB

# Tema 3 Arhitectura Sistemelor de Calcul

Organizare
-

### Structura de Date

Hashtabel-ul este implementat drept un vector de structuri ```Entry```.
Structura de date are 2 camputri de tipul uint32_t pentru salvarea cheii si valorii asociate.
Pentru gasirea/inserarea unei valori in hashtable se foloseste metoda 'linear probing', care implica:
 1. calculul valorii de hash a cheii;
 1. cautarea in mod liniar, pecand de la valoarea de hash a cheii
 1. cautarea se opreste odata ce se intalneste un ```Entry``` cu cheia egala cu ```KEY_INVALID``` sau cu cheia cauta.

### Functia de Hash

Functia de hash este updatata dupa cea de pe acest [link](http://www.cse.yorku.ca/~oz/hash.html) care calculeaza hashul pentru un string.

Functia updatat imi calculeaz hashul cu formula ```hash = hash * 33 + c``` unde hash este initializat cu 5381 si c este fiecare cifra a numarului luata pe rand.
Astfel pentru 125, de exemplu, hashul este egal cu ((5381) * 33 + 5) * 33 + 2) * 33 + 1.

Calculul se face in O(1) pentru ca un unsigned int are maxim 10 cifre, deci numarul de operatii este limitat.

### Blocks si Threads

Pentru apelarea functiilor de CUDA kernel, folosesc functia ```__host__ void getBlocksThreads(int *blocks, int *threads, int entries)``` pentru a gasi numarul de blockuri si threaduri.

Functia salveaza numarul de threaduri, aflat din proprietatiile deviceului. Apoi, in functie de numarul de structuri ```Entry``` ce vor fi prelucrate, se calculeaza si numarul de blockuri.

### Insert, Get si Reshape

Fiecare metoda a clasei de HashTable apeleaza o functie CUDA kernel pentru divizara cantitatii de munca.

Toate cele 3 functii de kernel obtin idul prin ```size_t idx = blockIdx.x * blockDim.x + threadIdx.x;``` si aplica metoda de 'linear probing' pentru inserare/gasirea valorilor.

Pentru inserarea de noi valori (la Insert si Reshape) folosesc operatii atomice pentru a ma asigura ca nu exista probleme de sincronizare a threadurilor.

#### Insert:

Aloca cu ```cudaMallocManaged``` un vector de structuri ```Entry``` pentru care seteaza valorile din cei 2 vectori de chei si valori.

Daca numarul de intari in hashmap insumat cu numarul de chei noi depaseste 90% din capacitatea hashmap-ului, atunci este apelata functia de reshape, inaintea apelului functiei de kernel.

De asemenea, functia de kernel calculeaza si numarulde chei updatate, nu inserate, pentru a inregistra corect numarul de intrari ocupate in hashmap (variabila ```keyUpdates```).
In final, numarul de intrai ocupate este egal cu ```entries += numKeys - keyUpdates```.


#### Get:

Aloc un vector pentru functia de kernel si copiez cheile primite ca input.

Vectorul de valori obtinute este alocat cu ```cudaMallocManaged```.

#### Reshape:

Aloc noul hashmap cu dimensiunea egala cu ```numBucketsReshape```, apoi apelez functia de kernel care face un insert pentru toate elementele vechiului hashmap.

Implementare
-

* Tema este realizata integral.
* Tema imi da punctaj maxim pe coada ibm, dar pe coada hp se blocheaza de la testul 2.

Resurse Utile
-

* [Functia de hash](http://www.cse.yorku.ca/~oz/hash.html) folosita.
* [Functii CUDA](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8).
* [Diferenta intre functile](https://stackoverflow.com/questions/12373940/difference-between-global-and-device-functions#:~:text=Global%20functions%20are%20also%20called%20%22kernels%22.&text=Device%20functions%20can%20only%20be,be%20called%20from%20host%20code.) __host__, __device__ si __global__.

[Github](https://github.com/WhyNotRaresh/Team3ASC)
-