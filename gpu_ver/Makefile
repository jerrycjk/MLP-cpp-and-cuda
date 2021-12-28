CXX = clang++
NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm
EXES     := net


net: main.cu Dense.cu Model.cu
	nvcc $(NVFLAGS) $(LDFLAGS) main.cu Dense.cu Model.cu -o $@