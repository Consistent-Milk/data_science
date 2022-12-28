# Run this script to download MNIST dataset
# to the root directory of the script.

from torchvision import datasets

image_path = './'
mnist_dataset = datasets.MNIST(
    image_path,
    'train',
    download=True
)
