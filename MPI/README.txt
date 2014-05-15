Files:
4 .c files
4 .o files
4 execute files
4 .bash files


Nodes:
You can use command to submit job:
	qsub -P cs546_s14_project -pe mpich 1 (follow bash fileName)
	
	
If you want to change the number of processors, use vim to open the bash file and edit the number after -np:
	#!/bin/bash
	mpirun -np 8 ./solutionA
	
	
