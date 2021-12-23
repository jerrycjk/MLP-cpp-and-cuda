CXX = clang++
CXXFLAGS = -std=c++11 -O3
EXE = net

.all: $(EXE)

net: main.cc Dense.cc Model.cc
	$(CXX) $(CXXFLAGS) main.cc Dense.cc Model.cc -o $@