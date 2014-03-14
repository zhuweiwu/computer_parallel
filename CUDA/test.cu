#include "stdio.h" 
#include <math.h>
#define COLUMNS 4 
#define ROWS 4 
 
__global__ void add(float *a, float *b, float *c) 
{ 
 int x = blockIdx.x; 
 int y = blockIdx.y; 
 int i = (COLUMNS*y) + x; 
 c[i] = a[i]; 
} 
 
int main() 
{ 
 float a[ROWS][COLUMNS], b[ROWS][COLUMNS], c[ROWS][COLUMNS]; 
 float *dev_a, *dev_b, *dev_c; 
 
 cudaMalloc((void **) &dev_a, ROWS*COLUMNS*sizeof(float)); 
 cudaMalloc((void **) &dev_b, ROWS*COLUMNS*sizeof(float)); 
 cudaMalloc((void **) &dev_c, ROWS*COLUMNS*sizeof(float)); 
 
 for (int y = 0; y < ROWS; y++) // Fill Arrays 
 for (int x = 0; x < COLUMNS; x++) 
 { 
 a[y][x] = (float)rand() / 32768.0; 
 b[y][x] = (float)rand() / 32768.0; 
 } 
 
 cudaMemcpy(dev_a, a, ROWS*COLUMNS*sizeof(float), 
cudaMemcpyHostToDevice); 
 cudaMemcpy(dev_b, b, ROWS*COLUMNS*sizeof(float), 
cudaMemcpyHostToDevice); 
 
 dim3 grid(COLUMNS,ROWS); 
 add<<<grid,1>>>(dev_a, dev_b, dev_c); 
 
 cudaMemcpy(c, dev_c, ROWS*COLUMNS*sizeof(float), 
cudaMemcpyDeviceToHost); 
 
 for (int y = 0; y < ROWS; y++) // Output Arrays 
 { 
 for (int x = 0; x < COLUMNS; x++) 
 { 
 printf("[%d][%d]=%d ",y,x,c[y][x]); 
 } 
 printf("\n"); 
 } 
 return 0; 
}