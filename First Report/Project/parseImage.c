#include <stdio.h>
#include <stdlib.h>

// Function to swap endianness (if needed)
unsigned int swap_endian(unsigned int val) {
    return ((val >> 24) & 0x000000ff) |
           ((val >>  8) & 0x0000ff00) |
           ((val <<  8) & 0x00ff0000) |
           ((val << 24) & 0xff000000);
}

// Function to read IDX3-UBYTE file and convert it to a 1D vector
float* read_idx3_ubyte_to_vector(const char* filename, int* image_size, int* num_images) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Unable to open file!\n");
        return NULL;
    }

    // Read the header information
    unsigned int magic_number, number_of_images, rows, cols;
    fread(&magic_number, sizeof(unsigned int), 1, file);
    fread(&number_of_images, sizeof(unsigned int), 1, file);
    fread(&rows, sizeof(unsigned int), 1, file);
    fread(&cols, sizeof(unsigned int), 1, file);

    // Convert endian if necessary
    magic_number = swap_endian(magic_number);
    number_of_images = swap_endian(number_of_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    // Verify if it's an IDX3 file (magic number for images should be 2051)
    if (magic_number != 2051) {
        printf("Invalid IDX3 file (magic number mismatch)!\n");
        fclose(file);
        return NULL;
    }

    *num_images = number_of_images;
    *image_size = rows * cols;  // Total pixels in one image

    // Allocate memory for 1D vector
    float *vector = (float*)malloc(number_of_images * (*image_size) * sizeof(float));
    if (!vector) {
        printf("Memory allocation failed!\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data and normalize to [0, 1]
    for (int i = 0; i < number_of_images * (*image_size); i++) {
        unsigned char pixel;
        fread(&pixel, sizeof(unsigned char), 1, file);
        vector[i] = pixel / 255.0f;  // Normalize the pixel value to [0, 1]
    }

    fclose(file);
    return vector;
}

int main() {
    const char* filename = "checkerboard.idx3-ubyte";
    int image_size, num_images;
    
    float* vector = read_idx3_ubyte_to_vector(filename, &image_size, &num_images);
    
    if (vector) {
        printf("Successfully read %d images, each with %d pixels.\n", num_images, image_size);
        for ( int i=0; i<image_size; i++ ){
            printf("%f", vector[i]);
        }
        
        free(vector);  // Free the allocated memory
    }

    return 0;
}