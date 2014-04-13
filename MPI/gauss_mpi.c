/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c" 
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
#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */
int seed;

/* Matrices and vectors */
volatile float **A, *B, X[MAXN];
/* A * X = B, solve for X */


/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
				* It is this routine that is timed.
				* It is called only on the parent.
				*/

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Process the parameters */
void parameters() {
  /* Random seed */
  //int seed = 0;

  char uid[32]; /*User name */
  
  srand(time_seed());  /* Randomize */
  srand(seed);
  printf("Random seed = %i\n", seed);

  if (N < 1 || N > MAXN) {
	 printf("N = %i is out of range.\n", N);
	 exit(0);
  } 
  
  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;
  
  /* allocate memory to matrix A and B */
  
  int i;
  A = (volatile float**) malloc(N * sizeof(float *));
  for(i=0; i < N; i++){
	A[i] = malloc(N * sizeof(float));
  } 
  B = (float*) malloc(N * sizeof(float));
  

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
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
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

/*main function*/
int main(int argc, char **argv) {
	int          my_rank;   /*My process rank           */
	int          p;         /*The number of processes   */
	int          source;    /*Messages sending  source  */
	int          dest;      /*Messages recv destination */
	int          tag = 0;   
	double       start_time;/*MPI start time            */
	double       end_time;  /*MPI end time              */
	MPI_Status   status;

	/* Let the system do what it needs to start up MPI */
	MPI_Init(&argc, &argv);
	
	/* Get process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	/* Find out how many processes are being used */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	/*Rank 0 is responsible for sending rows to other processes*/
	if(my_rank == 0){
		/*Get Matrix Size and Seed from keyboard*/
		printf("Input Matrix A Size and Seed : \n");
		
		scanf("%d %d", &N, &seed);
		
		/* Get the parameters we need: matrix size and seed */
		parameters();
		
		/* allocate memory and initialize A and B*/
		initialize_inputs();
		
		/* Display A and B if matrix size is less than 10 */
		print_inputs();
		
		/* Clock Sart*/
		start_time = MPI_Wtime();
		
		/* Gauss Elimination */
		gauss(my_rank, p);
		
		/* Clock Stop*/
		end_time = MPI_Wtime();
		
		/* Display the resilt X if N is less than 10 */
		print_X();
		
		/* Display cost time */
		printf("Cost Time = %.2f ms \n", (end_time - start_time) * 1000);
		
	}else{
		gauss(my_rank, p);
	}
	
	MPI_Finalize();	
	exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss(int my_rank, int p) {
  int norm, row, col;  /* Normalization row, and zeroing
						  element row and col */
  
  /* Rank 0 knows N, so it needs to send N to other processes*/  
  MPI_Bcast((void*)&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  /* Allocate memory for other processes */
  if(my_rank != 0){
	int i;
	A = (volatile float**) malloc(N * sizeof(float *));
	for(i=0; i < N; i++){
		A[i] = malloc(N * sizeof(float));
	} 
	B = (float*) malloc(N * sizeof(float));
  }
  
  /* Make sure all processes finish allocating memory */
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Status status;
  
  for(row = 0; row < N; row++){
	/* rank 0 send corresponding data to other processes */
	if(my_rank == 0){
		// send first row to other processes
		if(row == 0){
			int rank;
			for(rank = 1; rank < p; rank++){
				MPI_Send((void*)A[row], N, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
				MPI_Send((void*)&B[row], 1, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
			}
		}
		// use mod to divide matrix A and each process recv some of the matrix A 
		else if(row % p != 0){
			int rank = row % p;
			MPI_Send((void*)A[row], N, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
			MPI_Send((void*)&B[row], 1, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
		}
	}
	/* other ranks receive data from rank 0*/
	else{
		// other processes need row 0 
		if(row == 0){
			MPI_Recv((void*)A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv((void*)&B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		}
		// other rows are received by corresponding rank
		else if(row % p == my_rank){
			MPI_Recv((void*)A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv((void*)&B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		}
	}
  }
  /* All the data have been alloctaed here */
  MPI_Barrier(MPI_COMM_WORLD);
  

  /* Gaussian elimination */
  
  for (norm = 0; norm < N - 1; norm++) {
	for (row = norm + 1; row < N; row++) {
		// each processor processes their own rows 
		if(my_rank == row % p){			
			float multiplier = A[row][norm] / A[norm][norm];
			for (col = norm; col < N; col++) {
				A[row][col] -= A[norm][col] * multiplier;
			}
			B[row] -= B[norm] * multiplier;
		}
    }
	/* each norm we boardcast the (norm + 1) row to other processors  */
	MPI_Bcast((void*)A[norm + 1], N, MPI_FLOAT, (norm + 1) % p, MPI_COMM_WORLD);
	MPI_Bcast((void*)&B[norm + 1], 1, MPI_FLOAT, (norm + 1) % p, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
  }
  /* After we reach the end of loof norm, we do not need send data back to rank 0.
   * Because the matrix A and B are the final result.
   */

  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */


  /* Back substitution */
  if(my_rank == 0){
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N-1; col > row; col--) {
			X[row] -= A[row][col] * X[col];
		}
		X[row] /= A[row][row];
	}
  } 
}
