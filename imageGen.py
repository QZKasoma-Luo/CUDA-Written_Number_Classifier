import numpy as np

def save_as_idx_format(filename, image):
    # Ensure the image is in uint8 format
    image = image.astype(np.uint8)
    
    # Define the IDX header
    magic_number = 0x00000803  # Magic number for 3-dimensional unsigned byte array
    num_images = 1             # We're saving just one image
    rows = image.shape[0]      # 28
    cols = image.shape[1]      # 28
    
    # Create the header as per IDX file format
    header = np.array([
        0x00, 0x00, 0x08, 0x03,  # Magic number (for unsigned byte images)
        0x00, 0x00, 0x00, num_images,  # Number of images (in this case, 1)
        0x00, 0x00, 0x00, rows,   # Number of rows (28 for MNIST)
        0x00, 0x00, 0x00, cols    # Number of columns (28 for MNIST)
    ], dtype=np.uint8)

    # Open the file in binary write mode
    with open(filename, 'wb') as f:
        # Write the header to the file
        f.write(header.tobytes())
        # Write the image data to the file
        f.write(image.tobytes())

# Set image size to 28x28
size = 28

# Create a 2D array for the alternating pattern (checkerboard)
checkerboard = np.fromfunction(lambda x, y: (x + y) % 2 * 255, (size, size))

# Save the checkerboard image as an IDX file
save_as_idx_format('checkerboard.idx3-ubyte', checkerboard)