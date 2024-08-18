default:
	nvcc sgemm.cu -o sgemm.exe -lcublas -arch=sm_80 -O3  --ptxas-options=-v #-maxrregcount=128
	#nvcc sgemm.cu -o sgemm.exe -arch=compute_61 -code=sm_61 -lcublas --ptxas-options=-v -maxrregcount=128

run:
	./sgemm.exe

ut:
	nvcc ut.cu -o ut.exe -lcublas -arch=sm_80