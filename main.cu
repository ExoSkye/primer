#include <cstdio>
#include <cuda_runtime.h>
#include <Tracy.hpp>
#include <TracyC.h>
#include <cuda.h>
#include <mutex>
#include <thread>

typedef unsigned long long int ulli;
/*
void* operator new(std::size_t count) {
    auto ptr = malloc(count);
    TracyAlloc(ptr, count);
    return ptr;
}

void operator delete(void* ptr) noexcept {
    TracyFree(ptr);
    free(ptr);
}
*/
__global__ void checkPrime(bool* divisible, ulli tocheck,ulli offset) {
    ulli i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (i < tocheck) {
        if (tocheck % i == 0) {
            divisible[i-offset] = true;
        }
        else {
            divisible[i-offset] = false;
        }
    }

}

std::mutex m;

struct divstruct {
    int divisible;
};

__global__ void parallelfunc(bool* divarray,ulli* divisibles) {
    ulli i = blockDim.x * blockIdx.x + threadIdx.x;
    if (divarray[i]) {
        atomicAdd(divisibles, 1);
    }
}

__host__ int checkPrime(ulli num, ulli step) {
    ZoneScopedN("Prime checking loop")
    TracyCZoneN(alloc,"Allocating memory on GPU",true)
    bool *device_divarray = nullptr;
    cudaMalloc((void **) &device_divarray, step * sizeof(bool));
    ulli* d_divisibles = nullptr;
    cudaMalloc((void**)&d_divisibles, sizeof(ulli));
    TracyCZoneEnd(alloc)
    for (ulli i = 0; i < num; i+=step) {
        TracyCZoneN(loop,"Prime check loop",true);
        int threadsPerBlock = 256;
        int blocksPerGrid = (step + threadsPerBlock - 1) / threadsPerBlock;
        TracyCZoneN(kernelrun,"Running kernel",true)
        checkPrime<<<blocksPerGrid, threadsPerBlock>>>(device_divarray, num,i);
        TracyCZoneEnd(kernelrun)
        TracyCZoneN(interpret,"Interpret results",true)
        parallelfunc<<<blocksPerGrid, threadsPerBlock>>>(device_divarray,d_divisibles);
        TracyCZoneEnd(interpret)
        TracyCZoneEnd(loop)
        FrameMark;
    }
    ulli* divisibles = (ulli*)malloc(sizeof(ulli));
    cudaMemcpy(divisibles,d_divisibles,sizeof(ulli),cudaMemcpyDeviceToHost);
    if (*divisibles == 0) {
        printf("%llu is prime :)",num);
    }
    else {
        printf("%llu is not prime :'(",num);
    }
    cudaFree(device_divarray);
    return 0;
}

__host__ int main(int argc, char** argv) {
    ZoneScopedN("Program")
    if (argc < 2 || argc > 3) {
        printf("Inavlid parameters\n"
                    "Usage: %s number gpumemory\n"
                    "number:\t\tNumber to check for being prime\n"
                    "gpumemory:\t\tAmount of GPU Memory you have, used in finding best size parameter",argv[0]);
    }
    else if (argc == 3) {
        ulli i = 2147483648;
        ulli numtocheck;
        ulli gpumem;
        sscanf(argv[2],"%llu",&gpumem);
        sscanf(argv[1],"%llu",&numtocheck);
        printf("Working out optimum step size:\n");
        void* testobj;
        while (true) {
            i+=1073741824;
            testobj = malloc(i*sizeof(bool));
            if (testobj == nullptr || i > gpumem) {
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
