block_limit_per_sm:
	nvcc block_limit_per_sm.cu  -o block_limit_per_sm.exe -O3 -arch=sm_80
	bash block_limit_per_sm.sh
	
block_schedule:
	nvcc block_schedule.cu -o block_schedule.exe -O3 -arch=sm_80
	./block_schedule.exe 16
	./block_schedule.exe 32
	./block_schedule.exe 64

