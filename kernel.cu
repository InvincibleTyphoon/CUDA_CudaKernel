#include <cstdio>
#include <cuda_runtime.h>

// GPU를 위한 커널 프로그램(NVCC가 컴파일함)
__global__ void addKernel(int* c, const int * a, const int * b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__host__ int main(void){
	const int SIZE = 5;
	const int a[SIZE] = { 1,2,3,4,5 };
	const int b[SIZE] = { 10,20,30,40,50 };
	int c[SIZE] = { 0 };

	//디바이스(GPU) 측 데이터
	int * dev_a = 0;
	int * dev_b = 0;
	int * dev_c = 0;

	//VRAM에 메모리 할당
	cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_c, SIZE * sizeof(int));

	//값 복사(host -> device)
	cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	//커널 함수 수행
	//dev_c = dev_a + dev_b
	addKernel <<<1, SIZE >>> (dev_c, dev_a, dev_b);

	//값 복사(device -> host)
	cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	//메모리 공간 해제
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	//결과 출력
	printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n",
		a[0], a[1], a[2], a[3], a[4],
		b[0], b[1], b[2], b[3], b[4],
		c[0], c[1], c[2], c[3], c[4]);

	return 0;
}