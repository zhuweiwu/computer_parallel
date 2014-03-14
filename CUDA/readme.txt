In this homework, our target is to implement the matrix normlization using parallel computation.
Here we use cuda to design code to finish our computing on GPU.

Code Design:

In order to get the matrix we need, the equation is necessary:
	B[row][col] = (A[row][col] - mean) / standard_deviation
And need three steps to reach the goal:
1. mean of each column: (A[0][col] + A[1][col] + ... + A[N][col])/N , 0 <= col <= N
2. standard_deviation of each column: (A[i][col]-mean(col))^2 and sum all terms of this column.
                                       After diviing N and square root, we get the result.
3. the last step is to calculate the matrix B

The obvious difference between sequential computation is that we need to allocate memory 
and copy datas from host(CPU) to device(GPU) and then if we want to get the results, 
we also need to copy data from device(GPU) to host(CPU).
At the first glance, it will be more complex than just using CPU to compute. But as we know
GPU have many advantages than CPU in computing large number of data, like simple construction, more arithmetic units.

And now, how can we tell GPU what to calculate and how to calculate to get what we want.
The first thing is to allocate our data to the GPU memory. And we should decide three important units: grid, block and thread.
In this homework, we only use one-dimension block and thread.

 | Block0  --> Thread0 | Thread1 | Thread2 | ... |
 | Block1  --> Thread0 | Thread1 | Thread2 | ... |
 | Block2  --> Thread0 | Thread1 | Thread2 | ... | 
 | Block3  --> Thread0 | Thread1 | Thread2 | ... |

The syntax to get this kind of structure is :
	kernel_method<<<blockPerGrid, threadPerBlock>>>(args...);
Here, the values of blockPerGrid and threadPerBlock is the key of what we need to decide:
For example:
	We have a matrix A[N][N] and the size of this matrix is N*N. So we can assign threadPerBlock = 256.
	And blockPerGrid = ceil(N*N/256) ; The arm of this is that we can get an unique thread Id in the GPU.

__global__ void kernel(float *d_A, float *d_B, int n): this function is the whole computation process we design.

The postion of a thread can be found by threadIdx.x + blockIdx.x * blockDim.x. The first term is thread Id, and the next term is block Id.
So we can get every element of matrix A by the index. And what we need is each column, hence, we can search index, like 
	for(i=0; i<n; i++){
			mu += d_A[tId + i*n];
		}
In this part of code, we can get each column of data by the index, tId + i*n, which tId is threadIdx.x and n is the value of matrix size N.

Because each column is independent with others. We divide whole matrix into columns.

Performance Analysis:
									   

|  Matrix Size  |   Sequential(ms)  |  Parallel(ms)  |
|      4        |       0.011       |      351.359   |
|	   8        |       0.014       |	   345.739   |
|      20       |       0.031       |      336.385   |
|      40       |       0.077       |      325.574   |  
|      100      |       0.452       |      344.048   |
|      200      |       0.604       |      333.528   |
|      500      |       3.498       |      350.199   |
|      1000     |       14.187      |      390.303   |
|      2000     |       59.107      |      500.866   |
|      6000     |		651.759     |      684.689   | 
|      8000     |       1280.65     |      922.362   |

From the table, we can get some results.
1.When the matrix is small, the cost time of GPU is more than CPU. The reason is most of the cost time
  use for I/O operation to copy data from CPU to GPU.
2.When the matrix becomes larger and larger, advantages of GPU is obvious. It cost less time than CPU.

In conclusion, GPU is more suitable for large size matrix calculation.

