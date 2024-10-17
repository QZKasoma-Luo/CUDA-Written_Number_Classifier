import numpy as np

''' this script creates a sample image with a checkerboard pattern to test mechanics ''' 

def save_as_idx_format(filename, image):
    image = image.astype(np.uint8)
    
    # Define the IDX header
    magic_number = 0x00000803  #
    num_images = 1           
    rows = image.shape[0]     
    cols = image.shape[1]      
    
    # Create the header as per IDX file format
    header = np.array([
        0x00, 0x00, 0x08, 0x03,  # Magic number 
        0x00, 0x00, 0x00, num_images,  # Number of images
        0x00, 0x00, 0x00, rows,   # Number of rows 
        0x00, 0x00, 0x00, cols    # Number of columns 
    ], dtype=np.uint8)

    # Open the file in binary write mode
    with open(filename, 'wb') as f:
        f.write(header.tobytes())
        f.write(image.tobytes())

# Set image size to 28x28
size = 28

# Create a 2D array for the alternating pattern 
checkerboard = np.fromfunction(lambda x, y: (x + y) % 2 * 255, (size, size))

# Save the checkerboard image as an IDX file
save_as_idx_format('checkerboard.idx3-ubyte', checkerboard)
