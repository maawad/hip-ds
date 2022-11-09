INC=-Iinclude/ -Iinclude/detail/
COMMPILER=hipcc
COMMPILER_FLAGS=--std=c++17
CUDA_FLGAS=--expt-relaxed-constexpr -gencode arch=compute_60,code=sm_75
default: cu

test:	tests/test.cpp
		mkdir -p bin/
		$(COMMPILER) $(INC) -c tests/test.cpp -o bin/test

cu:	tests/test.cu
	mkdir -p bin/
	$(COMMPILER) $(COMMPILER_FLAGS) $(CUDA_FLGAS) tests/test.cu  $(INC) -o bin/test

clean:
		rm -rf bin/*



