/**
	Implement the 2D convolution using SPMD model and use MPI send and receive
	operations to perform communication. You need to run your program on 1, 2, 4, and 8
	processors and provide speedups as well as computation and communication timings.
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

typedef struct {float r; float i;} complex;
static complex ctmp;

//size of the image
#define N 512
//file
FILE* fp;
//matrix
complex image1[N][N];
complex image2[N][N];
complex result[N][N];
complex temp1[N][N];
complex temp2[N][N];

#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}

void c_fft1d(complex *r, int      n, int      isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
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
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
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
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}

void read(){
	if((fp = fopen("1_im1","r")) == NULL){
		printf("Cannot find the goal file.\n");
		exit(0);
	}
	int i,j;
	for(i = 0;i < N; i++){
		for(j = 0;j < N; j++){
			fscanf(fp,"%g",&image1[i][j].r);
			image1[i][j].i = 0;
		}
	}

	fclose(fp);
 
	if((fp = fopen("1_im2","r")) == NULL){
		printf("Cannot find the goal file.\n");
		exit(0);
	}
	for(i = 0;i < N; i++){
		for(j = 0;j < N; j++){
			fscanf(fp,"%g",&image2[i][j].r);
			image2[i][j].i = 0;
		}
	}
	fclose(fp); 
	printf("Finish reading data.\n");
}

void write(){
	if((fp = fopen("my_out_1","w+")) == NULL){
		printf("Cannot find the goal file.\n");
		exit(0);
	}
	int i,j;
	for(i = 0;i < N; i++){
		for(j = 0;j < N; j++){
			fprintf(fp,"%6.2g",result[i][j].r);
		}
		fprintf(fp,"\n");
	}
	
	fclose(fp);
	printf("Finish writing data.\n");
}

int main(int argc, char **argv){
	int rank;
	int size;
	int i, j;
	double start_time, end_time;
	double mpi_start_time, mpi_end_time;

	
	/* Initial */
	MPI_Init(&argc, &argv);
	
	start_time = MPI_Wtime();
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	//make own type of complex
	int blen[3] = {1,1,1};
	MPI_Aint indices[3];
	MPI_Datatype mystruct;
	MPI_Datatype myvector;
	MPI_Datatype old_types[3];
	
	old_types[0] = MPI_FLOAT;
	old_types[1] = MPI_FLOAT;
	old_types[2] = MPI_UB;
	
	indices[0] = 0;
	indices[1] = sizeof(float);
	indices[2] = sizeof(complex);
	
	MPI_Type_struct(3,blen,indices,old_types,&mystruct);
	MPI_Type_commit(&mystruct);
	MPI_Status status;
	
	/* read data file */
	if(rank == 0){
		//read file
		read();
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	mpi_start_time = MPI_Wtime();
	/* start 2D-FFT */
	//Row FFT
	if(rank == 0){
		
		//send data to other processes
		for(i = 1; i < size; i++){
			MPI_Send(&image1[i*(N/size)][0], N*N/size, mystruct, i, 0, MPI_COMM_WORLD);
			MPI_Send(&image2[i*(N/size)][0], N*N/size, mystruct, i, 1, MPI_COMM_WORLD);
		}
		
		//1D-fft
		for(i= 0; i < N/size; i++){
			c_fft1d(&image1[i][0], N, -1);
			c_fft1d(&image2[i][0], N, -1);
		}
		
		for(i=1; i<size; i++){
			MPI_Recv(&image1[i*N/size][0],N*N/size,mystruct,i,i, MPI_COMM_WORLD,&status);
			MPI_Recv(&image2[i*N/size][0],N*N/size,mystruct,i,i+N, MPI_COMM_WORLD,&status);
		}
		 
		
		//printf("Processor 0 recieve all data. Row FFT Finish.\n");
		
	}else{
		MPI_Recv(&image1[0][0],N*N/size,mystruct,0,0, MPI_COMM_WORLD,&status);
		MPI_Recv(&image2[0][0],N*N/size,mystruct,0,1, MPI_COMM_WORLD,&status);
		
		//1D-fft
		for(i= 0; i < N/size; i++){
			c_fft1d(&image1[i][0], N, -1);
			c_fft1d(&image2[i][0], N, -1);
		}

		MPI_Send(&image1[0][0],N*N/size,mystruct,0,rank, MPI_COMM_WORLD);
		MPI_Send(&image2[0][0],N*N/size,mystruct,0,rank+N, MPI_COMM_WORLD);
	}//Row FFT finish
	
	if(rank == 0){
		for(i = 0; i < N; i++){
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
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	//Col FFT	
	if(rank == 0){
		//send data to other processes
		for(i = 1; i < size; i++){
			MPI_Send(&image1[i*(N/size)][0], N*N/size, mystruct, i, 0, MPI_COMM_WORLD);
			MPI_Send(&image2[i*(N/size)][0], N*N/size, mystruct, i, 1, MPI_COMM_WORLD);
		}
		
		//1D-fft
		for(i= 0; i < N/size; i++){
			c_fft1d(&image1[i][0], N, -1);
			c_fft1d(&image2[i][0], N, -1);
		}
		
		//send back
		for(i = 1;i < size; i++){
			MPI_Recv(&image1[i*N/size][0],N*N/size,mystruct,i,i, MPI_COMM_WORLD,&status);
			MPI_Recv(&image2[i*N/size][0],N*N/size,mystruct,i,i+N, MPI_COMM_WORLD,&status);
		}
		
		//printf("Processor 0 recieve all data. Col FFT Finish.\n");
		
	}else{
		MPI_Recv(&image1[0][0],N*N/size,mystruct,0,0, MPI_COMM_WORLD,&status);
		MPI_Recv(&image2[0][0],N*N/size,mystruct,0,1, MPI_COMM_WORLD,&status);
		
		for(i = 0; i < N/size; i++)
		{
			c_fft1d(&image1[i][0],N, -1);
			c_fft1d(&image2[i][0],N, -1);
		}
		
		MPI_Send(&image1[0][0],N*N/size,mystruct,0,rank, MPI_COMM_WORLD);	
		MPI_Send(&image2[0][0],N*N/size,mystruct,0,rank+N, MPI_COMM_WORLD);
	}//Col FFT finish
	/* end of 2D-FFT*/
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	
	/* Point-Wise Multiplication */
	if(rank == 0){
		//send data
		for(i = 1; i < size; i++){
			MPI_Send(&image1[i*(N/size)][0], N*N/size, mystruct, i, 0, MPI_COMM_WORLD);
			MPI_Send(&image2[i*(N/size)][0], N*N/size, mystruct, i, 1, MPI_COMM_WORLD);
		}
		
		for(i = 0; i < N/size; i++){
			for(j = 0; j < N; j++){
				result[i][j].r = image1[i][j].r * image2[i][j].r;
				result[i][j].i = image1[i][j].i * image2[i][j].i;
			}
		}
		
		for(i = 1; i < size; i++){
			MPI_Recv(&result[i*N/size][0], N*N/size, mystruct, i, i, MPI_COMM_WORLD, &status);
		}
		
		//printf("Finish Point-Wise Multiplication \n");
		
	}else{
		//receive data from processor 0
		MPI_Recv(&image1[0][0],N*N/size,mystruct,0,0, MPI_COMM_WORLD,&status);
		MPI_Recv(&image2[0][0],N*N/size,mystruct,0,1, MPI_COMM_WORLD,&status);
		
		for(i = 0; i < N/size; i++){
			for(j = 0; j < N; j++){
				result[i][j].r = image1[i][j].r * image2[i][j].r;
				result[i][j].i = image1[i][j].i * image2[i][j].i;
			}
		}
		
		MPI_Send(&result[0][0],N*N/size,mystruct,0,rank, MPI_COMM_WORLD);
	}//point-wise multiplication end
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	/* start inverse 2D-FFT */
	//Row inverse-FFT
	if(rank == 0){
		//send data to other processes
		for(i = 1; i < size; i++){
			MPI_Send(&result[i*(N/size)][0], N*N/size, mystruct, i, 0, MPI_COMM_WORLD);
		}
		
		//1D-fft
		for(i= 0; i < N/size; i++){
			c_fft1d(&result[i][0], N, 1);
		}
		 
		//recieve data from other processes
		for(i=1; i<size; i++){
			MPI_Recv(&result[i*N/size][0],N*N/size,mystruct,i,i, MPI_COMM_WORLD,&status);
		}
		
		//printf("Processor 0 recieve all data. Row inverse-FFT Finish.\n");
		
	}else{
		MPI_Recv(&result[0][0],N*N/size,mystruct,0,0, MPI_COMM_WORLD,&status);
		
		//1D-fft
		for(i= 0; i < N/size; i++){
			c_fft1d(&result[i][0], N, 1);
		}

		MPI_Send(&result[0][0],N*N/size,mystruct,0,rank, MPI_COMM_WORLD);
	}//Row inverse-FFT end
	
	if(rank == 0){
		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				temp1[i][j] = result[i][j];
			}
		}

		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				result[j][i] = temp1[i][j];
			}
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Col inverse-FFT	
	if(rank == 0){
		//send data to other processes
		for(i = 1; i < size; i++){
			MPI_Send(&result[i*(N/size)][0], N*N/size, mystruct, i, 0, MPI_COMM_WORLD);
		}
		
		//1D-fft
		for(i= 0; i < N/size; i++){
			c_fft1d(&result[i][0], N, 1);
		}
		
		//send back
		for(i = 1;i < size; i++){
			MPI_Recv(&result[i*N/size][0],N*N/size,mystruct,i,i, MPI_COMM_WORLD,&status);
		}
		
		//printf("Processor 0 recieve all data. Col inverse-FFT Finish.\n");
		
	}else{
		MPI_Recv(&result[0][0],N*N/size,mystruct,0,0, MPI_COMM_WORLD,&status);		
		for(i = 0; i < N/size; i++)
		{
			c_fft1d(&result[i][0],N, 1);
		}
		
		MPI_Send(&result[0][0],N*N/size,mystruct,0,rank, MPI_COMM_WORLD);	
	}//Col FFT finish
	/* end of 2D-inverse-FFT*/
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(rank == 0){
		mpi_end_time = MPI_Wtime();
		
		write();
		
		end_time = MPI_Wtime();
		
		printf("MPI Cost time: %f (ms) \n", (mpi_end_time-mpi_start_time)*1000);
		printf("Total time: %f (ms) \n", (end_time-start_time)*1000);
			
		printf("Finish All!!!\n");
	}
	
	MPI_Type_free(&mystruct);
	//MPI_Type_free(&myvector);
	MPI_Finalize();
	exit(0);

}