/**
Implement 2D convolution model using SPMD model using hybrid programming (MPIopenMP
or MPI-Pthreads, Lecture 18). You need to run your program on 1, 2, 4 and 8
processors, with 8 threads per processors. You need to provide speedups as well as
computation and communication timings.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>
#include <pthread.h>

#define C_SWAP(a,b) { ctmp = (a); (a) = (b); (b) = ctmp; }
//size of the image
#define     N       512
#define     PR      8

/* struct complex_s {
    float r;
    float i;
};

typedef struct complex_s complex;
static complex ctmp; */

typedef struct {float r; float i;} complex;
static complex ctmp;

//matrix
static complex image1[N][N];
static complex image2[N][N];
static complex result[N][N];
static complex temp1[N][N];
static complex temp2[N][N];

static int stripN;

void c_fft1d(complex* r, int n, int isign)
{
    int     m, i, i1, j, k, i2, l, l1, l2;
    float   c1, c2, z;
    complex t, u;
    
    if (isign == 0) return;
    
    /* Do the bit reversal */
    i2 = n >> 1;
    j = 0;
    for (i = 0; i < n - 1; i ++) {
        if (i < j)
            C_SWAP(r[i], r[j]);
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    
    /* m = (int) log2((double)n); */
    for (i = n, m = 0; i > 1; m ++, i /= 2);
    
    /* Compute the FFT */
    c1 = -1.0;
    c2 =  0.0;
    l2 =  1;
    for (l = 0; l < m; l ++) {
        l1  = l2;
        l2 <<= 1;
        u.r = 1.0;
        u.i = 0.0;
        for (j = 0; j < l1; j ++) {
            for (i = j; i < n; i += l2) {
                i1 = i + l1;
                
                /* t = u * r[i1] */
                t.r = u.r * r[i1].r - u.i * r[i1].i;
                t.i = u.r * r[i1].i + u.i * r[i1].r;
                
                /* r[i1] = r[i] - t */
                r[i1].r = r[i].r - t.r;
                r[i1].i = r[i].i - t.i;
                
                /* r[i] = r[i] + t */
                r[i].r += t.r;
                r[i].i += t.i;
            }
            z =  u.r * c1 - u.i * c2;
            
            u.i = u.r * c2 + u.i * c1;
            u.r = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (isign == -1) /* FWD FFT */
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }
    
    /* Scaling for inverse transform */
    if (isign == 1) {       /* IFFT*/
        for (i = 0; i < n;i ++) {
            r[i].r /= n;
            r[i].i /= n;
        }
    }
}

void read()
{
    FILE* fp = NULL;
    
	if ((fp = fopen("1_im1", "r")) == NULL) {
		printf("Cannot find the goal file.\n");
		exit(0);
	}
    
	int i, j;
    
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			fscanf(fp, "%g", &image1[i][j].r);
			image1[i][j].i = 0;
		}
	}
    
	fclose(fp);
    
	if ((fp = fopen("1_im2", "r")) == NULL) {
		printf("Cannot find the goal file.\n");
		exit(0);
	}
    
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			fscanf(fp, "%g", &image2[i][j].r);
			image2[i][j].i = 0;
		}
	}
    
	fclose(fp);
	printf("Finish reading data.\n");
}

void write()
{
    FILE* fp = NULL;
    
	if ((fp = fopen("my_out_1", "w+")) == NULL) {
		printf("Cannot find the goal file.\n");
		exit(0);
	}
    
	int i, j;
    
	for (i = 0; i < N; i ++) {
		for (j = 0; j < N; j ++) {
			fprintf(fp, "%6.2g", result[i][j].r);
		}
		fprintf(fp, "\n");
	}
	
	fclose(fp);
	printf("Finish writing data.\n");
}

void* ffter(void* pId)
{
    int id = (int)pId;
    int s, t;
    
    s = id * stripN;
    t = s + stripN;
    
    for (; s < t; ++ s) {
        c_fft1d(&temp1[s][0], N, -1);
        c_fft1d(&temp2[s][0], N, -1);
    }
    
    return 0;
}

void* ffter_s(void* pId)
{
    int id = (int)pId;
    int s, t;
    
    s = id * stripN;
    t = s + stripN;
    
    for (; s < t; ++ s) {
        c_fft1d(&temp1[s][0], N, 1);
    }
    
    return 0;
}

void fft(int size, int sign)
{
    stripN = N / size / PR;
    long i;
    int rc;
    void* status;
    
    pthread_attr_t attr;
    pthread_t ffterid[PR];
    
    // set global thread attributes
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    
    if (sign == -1) {
        for (i = 0; i < PR; ++ i) {
            pthread_create(&ffterid[i], &attr, ffter, (void *)i);
        }
    }
    else {
        for (i = 0; i < PR; ++ i) {
            pthread_create(&ffterid[i], &attr, ffter_s, (void *)i);
        }
    }
    
    pthread_attr_destroy(&attr);
    for (i = 0; i < PR; ++ i) {
        rc = pthread_join(ffterid[i], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
#ifdef DEBUG
        printf("FFT: completed join with thread %ld having a status of %ld\n", i, (long)status);
#endif
    }
	
	
}

void flip(int row)
{
    complex ctmp1, ctmp2;
    int counter;
    
    for (counter = row + 1; counter < N; ++ counter) {
        ctmp1 = image1[counter][row];
        ctmp2 = image2[counter][row];
        image1[counter][row] = image1[row][counter];
        image2[counter][row] = image2[row][counter];
        image1[row][counter] = ctmp1;
        image2[row][counter] = ctmp2;
    }	
}

void flip_s(int row)
{
    complex ctmp_s;
    int counter;
    
    for (counter = row + 1; counter < N; ++ counter) {
        ctmp_s = result[counter][row];
        result[counter][row] = result[row][counter];
        result[row][counter] = ctmp_s;
    }
}

void* flipper(void *arg)
{
    
    int id =(int) arg;
    int s, t;
    
    s = id * stripN;
    t = s + stripN;
    
    for (; s < t; ++ s) {
        flip(s);
    }
    
    return 0;
}

void* flipper_s(void *arg)
{
    int id =(int) arg;
    int s, t;
    
    s = id * stripN;
    t = s + stripN;
    
    for (; s < t; ++ s) {
        flip_s(s);
    }
    
    return 0;
}

void transpose(int sign)
{
    stripN = N / PR;
    long i;
    int rc;
    void* status;
    
    pthread_attr_t attr;
    pthread_t flipperid[PR];
    
    // set global thread attributes
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    
    // create flippers
    if(sign == 1) {
        for (i = 0; i < PR; ++ i) {
            pthread_create(&flipperid[i], &attr, flipper, (void *)i);
        }
    }
    else {
        for (i = 0; i < PR; ++ i) {
            pthread_create(&flipperid[i], &attr, flipper_s, (void *)i);
        }
    }
    
    pthread_attr_destroy(&attr);
    for (i = 0; i < PR; ++ i) {
        rc = pthread_join(flipperid[i], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
#ifdef DEBUG
        printf("TRANSPOSE: completed join with thread %ld having a status of %ld\n", i, (long)status);
#endif
    }
}

void multiply(int row)
{
    int i, j = 0;
    for(i = row; j < N; ++ j) {
        temp1[i][j].r = temp1[i][j].r * temp2[i][j].r;
        temp1[i][j].i = temp1[i][j].i * temp2[i][j].i;
    }
}

void* multiplier(void* arg)
{
    int id =(int) arg;
    int s, t;
    
    s = id * stripN;
    t = s + stripN;
    
    for (; s < t; ++ s) {
        multiply(s);
    }
    
    return 0;
}

void mul(int size)
{
    stripN = N / size / PR;
    long i;
    int rc;
    void* status;
    
    pthread_attr_t attr;
    pthread_t multiplierid[PR];
    
    // set global thread attributes
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    
    // create flippers
    for (i = 0; i < PR; ++ i) {
        pthread_create(&multiplierid[i], &attr, multiplier, (void *)i);
    }
    
    pthread_attr_destroy(&attr);
    for (i = 0; i < PR; ++ i) {
        rc = pthread_join(multiplierid[i], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
#ifdef DEBUG
        printf("MUL: completed join with thread %ld having a status of %ld\n", i, (long)status);
#endif
    }
}

int main(int argc, char* argv[])
{
    int rank = 0;
	int size = 0;
	int i = 0, j = 0;
	double start_time = 0, end_time = 0;
	double mpi_start_time = 0, mpi_end_time = 0;
    
    /* Initial */
	MPI_Init(&argc, &argv);
	
	start_time = MPI_Wtime();
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//make own type of complex
	int blen[3] = { 1, 1, 1 };
	MPI_Aint indices[3];
	MPI_Datatype mystruct;
	MPI_Datatype old_types[3];
	
	old_types[0] = MPI_FLOAT;
	old_types[1] = MPI_FLOAT;
	old_types[2] = MPI_UB;
	
	indices[0] = 0;
	indices[1] = sizeof(float);
	indices[2] = sizeof(complex);
	
	MPI_Type_struct(3, blen, indices, old_types, &mystruct);
	MPI_Type_commit(&mystruct);
    
    if(rank == 0) {
		//read file
		read();
	}
    mpi_start_time = MPI_Wtime();
    
    /* start 2D-FFT */
	//Row FFT
	MPI_Scatter(&image1[0][0], N * (N / size), mystruct, &temp1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	MPI_Scatter(&image2[0][0], N * (N / size), mystruct, &temp2[0][0], N * (N / size), mystruct, 0 ,MPI_COMM_WORLD);
    
    // pthread
    fft(size, -1);

   /*  for(i = 0; i < N/size; i++){
		c_fft1d(&temp1[i][0],N,-1);
		c_fft1d(&temp2[i][0],N,-1);
	} */

    
    MPI_Gather(&temp1[0][0], N * (N / size), mystruct, &image1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	MPI_Gather(&temp2[0][0], N * (N / size), mystruct, &image2[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
    
	//int j = 0;
    for (i = 0; i < N; ++ i) {
        for (j = 0; j < N; ++ j) {
            temp1[i][j].r = 0; temp1[i][j].i = 0;
            temp2[i][j].r = 0; temp2[i][j].i = 0;
        }
    }
	
    //Transpose in processor 0
	if(rank == 0) {
        /* for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				temp1[i][j] = image1[i][j];
				temp2[i][j] = image2[i][j];
			}
		}
        
		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				image1[j][i] = temp1[i][j];
				image2[j][i] = temp2[i][j];
			}
		} */
        transpose(1);
	}
    
    //Col FFT
	MPI_Scatter(&image1[0][0], N * (N / size), mystruct, &temp1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	MPI_Scatter(&image2[0][0], N * (N / size), mystruct, &temp2[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	
    // pthread
    fft(size, -1);
    /* for(i = 0; i < N/size; i++){
		c_fft1d(&temp1[i][0],N,-1);
		c_fft1d(&temp2[i][0],N,-1);
	} */
	
	MPI_Gather(&temp1[0][0], N * (N / size), mystruct, &image1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	MPI_Gather(&temp2[0][0], N * (N / size), mystruct, &image2[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	/* end of 2D-FFT*/
    
	
	/* Point-Wise Multiplication */
	MPI_Scatter(&image1[0][0], N * (N / size), mystruct, &temp1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	MPI_Scatter(&image2[0][0], N * (N / size), mystruct, &temp2[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
    
    // pthread mul
    mul(size);
    /* for( i = 0; i < N/size; i++){
		for(j = 0; j < N; j++){
			temp1[i][j].r = temp1[i][j].r * temp2[i][j].r;
			temp1[i][j].i = temp1[i][j].i * temp2[i][j].i;
		}
	} */
    
    MPI_Gather(&temp1[0][0], N * (N / size), mystruct, &result[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	/* point wise multiplication end */
	
    
	/* start inverse 2D-FFT */
	MPI_Scatter(&result[0][0], N * (N / size), mystruct, &temp1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	
    // pthread iFFT per r
    fft(size, 1);
    /* for(i = 0; i < N/size; i++){
		c_fft1d(&temp1[i][0],N,1);
	} */
	
	MPI_Gather(&temp1[0][0], N * (N / size), mystruct, &result[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
    
    // pthread transpose
	if(rank == 0) {
        /* for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				temp1[i][j] = result[i][j];
			}
		}
        
		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				result[j][i] = temp1[i][j];
			}
		} */
        transpose(2);
	}
    
	MPI_Scatter(&result[0][0], N * (N / size), mystruct, &temp1[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	
    // pthread iFFT per r
    fft(size, 1);
    /* for(i = 0; i < N/size; i++){
		c_fft1d(&temp1[i][0],N,1);
	} */
	
	MPI_Gather(&temp1[0][0], N * (N / size), mystruct, &result[0][0], N * (N / size), mystruct, 0, MPI_COMM_WORLD);
	/* end of 2D-inverse-FFT*/
	
	if(rank == 0) {
		mpi_end_time = MPI_Wtime();
		
		write();
		
		end_time = MPI_Wtime();
		
		printf("MPI Cost time: %f (ms) \n", (mpi_end_time-mpi_start_time) * 1000);
		printf("Total time: %f (ms) \n", (end_time-start_time) * 1000);
        
		printf("Finish All!!!\n");
	}
    
	MPI_Type_free(&mystruct);
	MPI_Finalize();
    
    return 0;
}

