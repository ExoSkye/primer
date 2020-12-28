#include <cstdio>
#include <cuda_runtime.h>
#include <Tracy.hpp>
#include <TracyC.h>
#include <cuda.h>

typedef unsigned long long int ulli;

void* operator new(std::size_t count) {
    auto ptr = malloc(count);
    TracyAlloc(ptr, count);
    return ptr;
}

void operator delete(void* ptr) noexcept {
    TracyFree(ptr);
    free(ptr);
}

__global__ void checkPrime(bool* divisible, ulli tocheck,ulli offset) {
    ulli i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (i < tocheck) {
        if (tocheck % i == 0) {
            divisible[i] = true;
        }
        else {
            divisible[i] = false;
        }
    }

}

__host__ int checkPrime(ulli num, ulli step) {
    ZoneScopedN("Prime checking loop")
    TracyCZoneN(alloc,"Allocating memory on:",true)
    int divisibles = 0;
    TracyCZoneN(hostalloc,"HOST",true)
    bool *divarray = (bool *) malloc(step * sizeof(bool));
    TracyCZoneEnd(hostalloc)
    TracyCZoneN(gpualloc,"GPU",true)
    bool *device_divarray = nullptr;
    cudaMalloc((void **) &device_divarray, step * sizeof(bool));
    TracyCZoneEnd(gpualloc)
    TracyCZoneEnd(alloc)
    for (ulli i = 0; i < num; i+=step) {
        TracyCZoneN(loop,"Prime check loop",true);
        int threadsPerBlock = 256;
        int blocksPerGrid = (step + threadsPerBlock - 1) / threadsPerBlock;
        TracyCZoneN(kernelrun,"Running kernel",true)
        checkPrime<<<blocksPerGrid, threadsPerBlock>>>(device_divarray, num,i);
        TracyCZoneEnd(kernelrun)
        TracyCZoneN(memcpygpu,"Copy results over",true)
        cudaMemcpy(divarray, device_divarray, step * sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(device_divarray);
        TracyCZoneEnd(memcpygpu)
        TracyCZoneN(interpret,"Interpret results",true)
#pragma omp parallel for
        for (ulli j = 0; j < step; j++) {
            if (divarray[j]) {
                divisibles++;
            }
        }
        TracyCZoneEnd(interpret)
        TracyCZoneEnd(loop)
        FrameMark;
    }
    if (divisibles == 0) {
        printf("%llu is prime :)",num);
    }
    else {
        printf("%llu is not prime :'(",num);
    }
    free(divarray);
    return 0;
}

__host__ int main(int argc, char** argv) {
    ZoneScopedN("Program")
    if (argc < 2 || argc > 2) {
        printf("Inavlid parameters\n"
                    "Usage: %s number\n"
                    "number:\t\tNumber to check for being prime",argv[0]);
    }
    else if (argc == 2) {
        char* ptr;
        ulli i = 2147483648;
        size_t freemem, total;
        cuMemGetInfo(&freemem,&total);
        ulli numtocheck = strtol(argv[1],&ptr,10);
        printf("Working out optimum step size:\n");
        void* testobj;
        while (true) {
            i+=1073741824;
            testobj = malloc(i*sizeof(bool));
            if (testobj == nullptr || i > freemem) {
                printf("[✖] %llu\n",i);
                i-=1073741824;
                break;
            }
            else if (i > numtocheck) {
                printf("[✔] %llu\n", i);
                break;
            }
            else {
                printf("[✔] %llu\n",i);
            }
            free(testobj);
        }
        return checkPrime(numtocheck,i);
    }
}
