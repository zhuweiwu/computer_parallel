


In this homework, we use two methods to implement parallel gaussian elimination.
One is Pthread, the other is openMP.

1.Design:

Before starting to write code, we should clearly understand about how does Gaussian Elimination work.
And then we can implement parallel computing correctly. Let us analyse the sequential code, guass.c.
Here is an example:

|  A[0][0]  A[0][1]  A[0][2] |   |X[0]|   |B[0]|
|  A[1][0]  A[1][1]  A[1][2] | * |X[1]| = |B[1]|
|  A[2][0]  A[2][1]  A[2][2] |   |X[2]|   |B[2]|

And we need change these matrix to:

|  C[0][0]  C[0][1]  C[0][2] |   |X[0]|   |D[0]|
|     0     C[1][1]  C[1][2] | * |X[1]| = |D[1]|
|     0        0     C[2][2] |   |X[2]|   |D[2]|

Based on the algorithm of Gaussian Elimination, we find that this part can be parallel. 
And then we look at the sequential computing code:

	for (norm = 0; norm < N - 1; norm++) {
		for (row = norm + 1; row < N; row++) {
			 multiplier = A[row][norm] / A[norm][norm];
			 for (col = norm; col < N; col++) {
				A[row][col] -= A[norm][col] * multiplier;
			 }
			B[row] -= B[norm] * multiplier;
		}

	}

From the code, we find that the computation between different normalization has dependence, 
but the computation in each norm are independent. Now we can decide the design of parallel.

2.Code Decision:

--openMP:
	After find the part which we need to be parallel, we implement this function using openMP. It is
	very easy to finish the code. We just need to add a special sentence after the out-loop, like: 
			for(norm = 0; norm < N -1; norm++){
		#pragma omp parallel for private(row, col, multiplier) num_threads(nThreads)
				for(row = norm + 1; row < N; row++)
				...
			}			
	Here the variables(row, col, and multiplier) are private, because they can not shared by other threads.
	And we also need to tell openMP how many threads you want to create, so nThreads is needed.
	

--pThread
	The paralle part is the same as openMP. But we should create threads and join them by ourselves.
	In my design, I assign one thread to one row according to the thread ID
	After all rows finished, we need a barrier to synchronize all threads.
	The detail of code can be found in gauss_pthread.c.


3.Performance Analysis:

If you want to get the result, we should compile all my .c files.The command is:
	gcc gauss.c  ---> ./a.out N  (Here N is an integer <= 2000)
	gcc gauss_pthread.c -lpthread ---> ./a.out N M (N is the same as above and M is the number of threads)
	gcc gauss_openmp.c -fopenmp ---> ./a.out N M (N,M is the same meaning as above)

N = 2000

time_sequential = 19600.2


number of      time_pthread(ms)     time_openMP(ms)
threads(M)
-----------------------------------------------------
   1               19614                21086.8     |
-----------------------------------------------------
   2               10010.9              10642.7     |
-----------------------------------------------------
   4               8583.05              9146.04     |
-----------------------------------------------------
   8               9377.17              9466.83     |
-----------------------------------------------------
   10              9411.24              9813.42     |
-----------------------------------------------------
   12              9230.02              9360.36     |
-----------------------------------------------------   
   16              9222.24              9433.24     |
-----------------------------------------------------
   20              9213.22              9573.94     | 
-----------------------------------------------------
   50              9708.79              9796.15     |
-----------------------------------------------------
   100             11293.9              10101.8     |
-----------------------------------------------------

Result Analysis:

From the data of table, we find that there is no obvious difference between pThread and openMP.
But pThread is a little faster than openMP from the time comparison. And the fastest is when we choose 4 threads. 
The basic reason is our computer has multi-core. And it can execute several operation at the same time.
However, after we continuously increase the number of threads, computer costs more time to finish the process.
Because the processor is busy during the whole computing. So if the threads is more than 4, the time is longer and longer.
Hence, in my conclusion, when the threads are equal to the number of processors, we can get the best performance.
Here, 4 threads are the best choose.

Thread Number = 4
speedup_pthread =  19600.2/8583.05 = 2.28
speedup_openMp =   19600.2/9146.04 = 2.14





