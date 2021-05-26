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

Implementare
-

* Tema este realizata integral.


Resurse Utile
-

* [Functia de hash](http://www.cse.yorku.ca/~oz/hash.html) folosita.
* [Functii CUDA](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8).

[Github](https://github.com/WhyNotRaresh/Team3ASC)
-