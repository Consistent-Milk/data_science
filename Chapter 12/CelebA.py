# This script downloads the CelebA dataset into the root directory
# of the script.
from torchvision import datasets

image_path = './'
celeba_dataset = datasets.CelebA(
    image_path,
    split='train',
    target_type='attr',
    download=True
)
