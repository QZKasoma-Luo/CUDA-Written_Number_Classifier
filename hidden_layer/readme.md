## Compile the library

nvcc -c hidden_layer.cu -o hidden_layer.o
ar rcs libhiddenlayer.a hidden_layer.o
