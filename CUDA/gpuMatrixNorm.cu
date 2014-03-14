/* 
 * The function of this file implements Matrix Norm in GPU using CUDA.
 * 
 * input: N: the Matrix Size (required) 
 *        Random Seed : use for initial matrix value (not required)
 *
 * __global__ void kernel(float *d_A, float *d_B, int n) finish the function in GPU
 * 
 * d_A transfer data from host to device
 * d_B get the result from device to host
 *
 * @date : 3/9/2014
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
	    cudaGetErrorString(res),__FILE__,__LINE__);	\
    exit(-1);						\
  }                                                     \
  
int N;  /* Matrix size */
/* Matrices */
float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
      B[row][col] = 0.0;
    }
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	    printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}
//
__global__ void kernel(float *d_A, float *d_B, int n) {
	int tId = threadIdx.x;
	int blockId = blockIdx.x;
	float mu;
	float sigma;
	int i;
	
	//int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	if( blockId < n){
			
		for(i=0; i<n; i++){
			mu += d_A[tId + i*n];
		}
		mu /= n;
		for(i=0; i<n; i++){
			sigma += powf((d_A[tId + i*n] - mu), 2.0);
		}
		
		sigma /= n;
		sigma = powf(sigma, 0.5);
		
		for(i=0; i<n; i++){
			if(sigma == 0.0){
				d_B[tId + i*n] = 0.0;
			}else{
				d_B[tId + i*n] = (d_A[tId + i*n] - mu)/sigma;
			}
		}
	
	}

}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  matrixNorm();
  
  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_B();

  /* Display timing results */
  printf("\n Total Elapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);
  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][] and B[][],
 * defined in the beginning of this code.  B[][] is initialized to zeros.
 */
void matrixNorm(){
	cudaError_t res;
	float *d_A, *d_B;	
	float *h_A, *h_B;
	int i,j;
	int size = N*N*sizeof(float);
	
	h_A = (float*) malloc(size);
	h_B = (float*) malloc(size);
	
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			h_A[i*N + j] = A[i][j];
		}
	}
	
	//allocate memory for d_A and d_B in the device
	res = cudaMalloc((void **) &d_A, size);
	CHECK_ERR(res);
	res = cudaMalloc((void **) &d_B, size);
	CHECK_ERR(res);

	//initial memory 
	//res = cudaMemset(d_A, 0, size);
	//CHECK_ERR(res);
	//res = cudaMemset(d_B, 0, size);
	//CHECK_ERR(res);

	//copy memory to device
	res = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CHECK_ERR(res);
	res = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	CHECK_ERR(res);

	//calling the kernel
	int threadPerBlock = 128; 
	int blockPerGrid = ceil((float)(N*N)/(float)threadPerBlock);
	
	//printf("test... %d \n", blockPerGrid);
	kernel<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, N);
	
	res = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
	CHECK_ERR(res);
	
	//free memeory
	cudaFree(d_A);
	cudaFree(d_B);
	
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			B[i][j] = h_B[i*N+j];
		}
	}
 }
 
 
 

