import numpy as np
from tensorflow.keras.datasets import mnist

def save_training_data(images, labels, filename):
    num_samples = images.shape[0]
    images = images.astype(np.float32)  # Ensure consistent data type
    labels = labels.astype(np.int32)

    with open(filename, 'wb') as f:
        # Write the number of samples as int32
        f.write(np.array([num_samples], dtype=np.int32).tobytes())

        # Write each label and image vector to the file
        for i in range(num_samples):
            # Write label (int32)
            f.write(labels[i].tobytes())
            # Write image vector (float32 array)
            f.write(images[i].tobytes())

if __name__ == "__main__":
    # Load MNIST dataset
    (images, labels), _ = mnist.load_data()

    # Select the first 1,000 images and labels
    train_samples = 1000
    test_samples = 500
    trainimages = images[:train_samples]
    trainlabels = labels[:train_samples]

    testimages = images[train_samples:test_samples]
    testlabels = labels[train_samples:test_samples]

    # Flatten the images from 28x28 to 784 vectors
    trainimages = trainimages.reshape((train_samples, 28 * 28))
    testimages = testimages.reshape((test_samples, 28 * 28))

    # Normalize pixel values to [0, 1]
    trainimages = trainimages / 255.0
    testimages = testimages / 255.0

    # Save to binary file
    save_training_data(trainimages, trainlabels, 'train_mnist.bin')
    save_training_data(testimages, testlabels, 'test_mnist.bin')

