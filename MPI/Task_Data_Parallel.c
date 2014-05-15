/**
	Implement 2D convolution model using a Task and Data Parallel Model. You also need to
	show the use of communicators in MPI. Let’s say we divide the P processors into four groups:
	P1, P2, P3, and P4. You will run Task 1 on P1 processors, Task 2 on P2 processors, Task 3
	on P3 processors, and Task 4 on P4 processors. The Following figure illustrates this case.
	Report computation and communication results for P1=P2=P3=P4=2.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>

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

void read_image1(){
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
	printf("Finish reading data from image1.\n");
}

void read_image2(){
	if((fp = fopen("1_im2","r")) == NULL){
		printf("Cannot find the goal file.\n");
		exit(0);
	}
	int i,j;
	for(i = 0;i < N; i++){
		for(j = 0;j < N; j++){
			fscanf(fp,"%g",&image2[i][j].r);
			image2[i][j].i = 0;
		}
	}
	fclose(fp); 
	printf("Finish reading data from image2.\n");
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
	int rank, size; //gobal rank and size
	int rank_local, size_local; //local rank and size
	int color; //control of subset assignment (nonnegative integer). 
			   //Processes with the same color are in the same new communicator
	int i, j;
	double start_time, end_time;
	double mpi_start_time, mpi_end_time;
	MPI_Comm mycomm;
	MPI_Status status;
	
	/* Initial */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	start_time = MPI_Wtime();
	/*
	 * (rank 0,1),(rank 2,3), (rank 4,5) (rank 6,7)
	 *  color=0    color=1     color=2    color=3
	 * split 8 processes into 4 group
	 */
	color = rank/2;
	MPI_Comm_split(MPI_COMM_WORLD,color,rank,&mycomm);
	MPI_Comm_rank(mycomm,&rank_local);
	MPI_Comm_size(mycomm,&size_local);
	
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
	//MPI_Type_vector(N,N/size,N,mystruct,&myvector);
	//MPI_Type_commit(&myvector);
	
	/* read file, rank0 and rank2 read image1 and image2, respectively */
	if(color == 0){
		//rank0 read image1
		if(rank_local == 0){
			read_image1();
		}
	}
	
	if(color == 1){
		if(rank_local == 0){
			read_image2();
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	mpi_start_time = MPI_Wtime();
	
	/* *******************
	*	     Task 1      *
	*	 2D-FFT Image1   *	
	* ********************/
	if(color == 0){
		/* rank0 and rank1 finish 2D-FFT of image1 */
		//Row FFT of image1
		MPI_Scatter(&image1[0][0],N*N/size_local,mystruct,&temp1[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local;i++){
			c_fft1d(&temp1[i][0],N, -1);
		}
		MPI_Gather(&temp1[0][0],N*N/size_local,mystruct,&image1[0][0],N*N/size_local,mystruct,0,mycomm);
		
		//Transpose of image1
		if(rank_local == 0){
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					temp1[i][j] = image1[i][j];
				}
			}
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					image1[j][i] = temp1[i][j];
				}
			}
		}
		
		//Col FFT of image1
		MPI_Scatter(&image1[0][0],N*N/size_local,mystruct,&temp1[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local;i++){
			c_fft1d(&temp1[i][0],N, -1);
		}
		MPI_Gather(&temp1[0][0],N*N/size_local,mystruct,&image1[0][0],N*N/size_local,mystruct,0,mycomm);
		
		if(rank_local == 0){
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					temp1[i][j] = image1[i][j];
				}
			}
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					image1[j][i] = temp1[i][j];
				}
			}
		}
		
		//printf("Task1 finish!! \n");
	}
	
	/* *******************
	*	     Task 2      * 
	*	 2D-FFT Image2   *	
	* ********************/
	if(color == 1){
		/* rank2 and rank3 finish 2D-FFT of image2 */
		//Row FFT of image2
		MPI_Scatter(&image2[0][0],N*N/size_local,mystruct,&temp2[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local;i++){
			c_fft1d(&temp2[i][0],N, -1);
		}
		MPI_Gather(&temp2[0][0],N*N/size_local,mystruct,&image2[0][0],N*N/size_local,mystruct,0,mycomm);
		
		//Transpose of image2
		if(rank_local == 0){
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					temp2[i][j] = image2[i][j];
				}
			}
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					image2[j][i] = temp2[i][j];
				}
			}
		}
		
		//Col FFT of image2
		MPI_Scatter(&image2[0][0],N*N/size_local,mystruct,&temp2[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local;i++){
			c_fft1d(&temp2[i][0],N, -1);
		}
		MPI_Gather(&temp2[0][0],N*N/size_local,mystruct,&image2[0][0],N*N/size_local,mystruct,0,mycomm);
		
		if(rank_local == 0){
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					temp2[i][j] = image2[i][j];
				}
			}
			for(i = 0; i < N; i++){
				for(j = 0; j < N; j++){
					image2[j][i] = temp2[i][j];
				}
			}
		}
		
		//printf("Task2 finish!! \n");
		
	}
	
	//continue until task1 and task2 are finished
	//MPI_Barrier(MPI_COMM_WORLD);
	
	/* *******************
	*	     Task 3      *  
	*   Point-Wise Multi *
	* ********************/
	//before do point-wise multiplication, send data to rank4
	if(color == 0){
		if(rank_local == 0){
			MPI_Send(&image1[0][0],N*N,mystruct,4,0,MPI_COMM_WORLD);
		}
	}
	
	if(color == 1){
		if(rank_local == 0){
			MPI_Send(&image2[0][0],N*N,mystruct,4,0,MPI_COMM_WORLD);
		}
	}
	
	// start point-wise multiplication
	if(color == 2){
		if(rank_local == 0){
			MPI_Recv(&image1[0][0],N*N,mystruct,0,0,MPI_COMM_WORLD,&status);
			MPI_Recv(&image2[0][0],N*N,mystruct,2,0,MPI_COMM_WORLD,&status);
		}
		
		MPI_Scatter(&image1[0][0],N*N/size_local,mystruct,&temp1[0][0],N*N/size_local,mystruct,0,mycomm);
		MPI_Scatter(&image2[0][0],N*N/size_local,mystruct,&temp2[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local; i++){
			for(j = 0; j < N; j++){
				temp1[i][j].r = temp1[i][j].r * temp2[i][j].r;
				temp1[i][j].i = temp1[i][j].i * temp2[i][j].i;
			}
		}	
		MPI_Gather(&temp1[0][0],N*N/size_local,mystruct,&result[0][0],N*N/size_local,mystruct,0,mycomm);
		//printf("Task3 finish!!! \n");
		if(rank_local==0){
			MPI_Send(&result[0][0],N*N,mystruct,6,0,MPI_COMM_WORLD);
		}	
	}
	
	/* *******************
	*	     Task 4      *  
	*   Inverse 2D-FFT   *
	* ********************/	
	
	//before inverse FFT, receive the data from rank4
	if(color == 3){
		if(rank_local == 0){
			MPI_Recv(&result[0][0],N*N,mystruct,4,0,MPI_COMM_WORLD,&status);
		}
		
		/* rank6 and rank7 finish inverse 2D-FFT of result */
		MPI_Scatter(&result[0][0],N*N/size_local,mystruct,&temp1[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local;i++){
			c_fft1d(&temp1[i][0],N, 1);
		}
		MPI_Gather(&temp1[0][0],N*N/size_local,mystruct,&result[0][0],N*N/size_local,mystruct,0,mycomm);
		
		//Transpose of result
		if(rank_local == 0){
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

		MPI_Scatter(&result[0][0],N*N/size_local,mystruct,&temp1[0][0],N*N/size_local,mystruct,0,mycomm);	
		for(i = 0; i < N/size_local;i++){
			c_fft1d(&temp1[i][0],N, 1);
		}
		MPI_Gather(&temp1[0][0],N*N/size_local,mystruct,&result[0][0],N*N/size_local,mystruct,0,mycomm);
		
		if(rank_local == 0){
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
			
			mpi_end_time = MPI_Wtime();
		}
		
		//printf("Task4 finish!! \n");
		
		if(rank_local == 0){
			
			write();
			end_time = MPI_Wtime();
			
			printf("MPI Cost time: %f (ms) \n", (mpi_end_time-mpi_start_time)*1000);
			printf("Total time: %f (ms) \n", (end_time-start_time)*1000);
		}
	}
	
	MPI_Type_free(&mystruct);
	MPI_Finalize();
	exit(0);	
}