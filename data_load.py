import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding facial keypoints.
    
    This class reads image file paths and keypoints from a CSV file and provides 
    functionality to apply optional transformations on the samples.
    
    Args:
        csv_file_address (str): Path to the CSV file containing image file names and keypoints.
        root_dir (str): Directory where the images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_file_address: str, root_dir: str, transform=None):
        """
        Initialize the FacialKeypointsDataset with the given parameters.
        
        Args:
            csv_file_address (str): Path to the CSV file containing image file names and keypoints.
            root_dir (str): Directory where the images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file_address)  # Read the CSV file into a DataFrame
        self.root_dir = root_dir  # Store the root directory path
        self.transform = transform  # Store the transform function (if any)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.key_pts_frame)  # The length is the number of rows in the DataFrame
    
    def __getitem__(self, idx):
        """
        Retrieve the sample (image and keypoints) at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing:
                - 'image': The loaded image as a NumPy array.
                - 'keypoints': The corresponding facial keypoints as a NumPy array.
        """
        # Construct the full image file path
        image_address = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        
        # Read the image from the file
        image = mpimg.imread(image_address)

        # Check if the image has an alpha channel (RGBA), and if so, remove it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]  # Keep only the RGB channels
        
        # Extract the keypoints from the DataFrame
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)  # Convert keypoints to a float NumPy array and reshape

        # Create a sample dictionary with the image and keypoints
        sample = {'image': image, 'keypoints': key_pts}

        # Apply the transform to the sample if one is provided
        if self.transform:
            sample = self.transform(sample)

        return sample  # Return the sample with the image and keypoints
    
# tranforms

class Normalize(object):
    """
    Custom normalization class for preprocessing images and their corresponding keypoints.
    """

    # The call method makes the class instance callable like a function
    def __call__(self, sample):
        """
        Apply normalization to a sample containing an image and its keypoints.

        Args:
            sample (dict): A dictionary containing:
                - 'image': The input image as a NumPy array.
                - 'keypoints': The keypoints associated with the image as a NumPy array.

        Returns:
            dict: A dictionary with normalized 'image' and 'keypoints'.
        """

        # Extract image and keypoints from the sample dictionary
        image, key_pts = sample['image'], sample['keypoints']
        
        # Make copies of the image and keypoints to avoid altering the original data
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # Convert the image to grayscale using OpenCV's color conversion function
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        
        # Normalize the image to a range between 0 and 1 by dividing by 255 (since image pixel values are typically in [0, 255])
        image_copy = image_copy / 255.0

        # Normalize keypoints to a range roughly between -2 and 2
        # Assuming keypoints are centered around 100 and have a rough spread of 50
        key_pts_copy = (key_pts_copy - 100) / 50.0

        # Return the normalized image and keypoints in a dictionary
        return {'image': image_copy, 'keypoints': key_pts_copy}
    
class Rescale(object):
    """
    A class to rescale the image in a sample to a given size.
    
    Args:
        output_size (tuple or int): Desired output size for rescaling the image.
            If tuple, the image will be resized to match the dimensions specified by output_size.
            If int, the smaller dimension of the image will be resized to output_size, 
            and the larger dimension will be scaled to maintain the aspect ratio.
    """
    
    def __init__(self, output_size):
        """
        Initialize the Rescale class with the specified output size.
        
        Args:
            output_size (tuple or int): Desired output size.
                - If a tuple (width, height), the image will be resized to these dimensions.
                - If an integer, the smaller edge of the image will be resized to this size, 
                  and the other edge will be scaled to maintain the aspect ratio.
        """
        # Check that output_size is either an int or a tuple
        assert isinstance(output_size, (int, tuple)), "Output size must be an integer or a tuple"
        self.output_size = output_size  # Store the provided output size

    def __call__(self, sample):
        """
        Apply the rescaling to a sample containing an image and its keypoints.
        
        Args:
            sample (dict): A dictionary containing:
                - 'image': The input image as a NumPy array.
                - 'keypoints': The keypoints associated with the image as a NumPy array.
        
        Returns:
            dict: A dictionary containing:
                - 'image': The resized image as a NumPy array.
                - 'keypoints': The adjusted keypoints scaled to the resized image.
        """
        
        # Extract the image and keypoints from the sample dictionary
        image, key_pts = sample['image'], sample['keypoints']

        # Get the height and width of the original image
        h, w = image.shape[:2]
        
        # Determine the new dimensions for the image
        if isinstance(self.output_size, int):
            # If output_size is an integer, calculate the new dimensions while maintaining the aspect ratio
            if h > w:
                # If the height is greater than the width, set the new width to output_size and scale the height
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                # If the width is greater than or equal to the height, set the new height to output_size and scale the width
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            # If output_size is a tuple, use it directly as the new dimensions
            new_h, new_w = self.output_size

        # Convert the new dimensions to integers
        new_h, new_w = int(new_h), int(new_w)
        
        # Resize the image to the new dimensions using OpenCV's resize function
        img = cv2.resize(image, (new_w, new_h))
        
        # Scale the keypoints to match the resized image
        # The keypoints are adjusted by multiplying with the ratios of the new dimensions to the original dimensions
        key_pts = key_pts * [new_w / w, new_h / h]

        # Return the resized image and adjusted keypoints in a dictionary
        return {'image': img, 'keypoints': key_pts}
    
class RandomCrop(object):
    """
    A class to randomly crop the image in a sample to a given size.
    
    Args:
        output_size (tuple or int): Desired output size for cropping the image.
            If int, a square crop of size (output_size, output_size) will be made.
            If tuple, the crop will have the dimensions specified by (height, width).
    """
    
    def __init__(self, output_size):
        """
        Initialize the RandomCrop class with the specified output size.
        
        Args:
            output_size (tuple or int): Desired output size.
                - If a tuple (height, width), the crop will match these dimensions.
                - If an integer, a square crop of size (output_size, output_size) will be made.
        """
        # Check that output_size is either an int or a tuple
        assert isinstance(output_size, (int, tuple)), "Output size must be an integer or a tuple"
        if isinstance(output_size, int):
            # If an integer is provided, convert it to a tuple of (output_size, output_size) for a square crop
            self.output_size = (output_size, output_size)
        else:
            # If a tuple is provided, ensure it has exactly two elements
            assert len(output_size) == 2, "Tuple output size must have two elements"
            self.output_size = output_size  # Store the provided output size

    def __call__(self, sample):
        """
        Apply random cropping to a sample containing an image and its keypoints.
        
        Args:
            sample (dict): A dictionary containing:
                - 'image': The input image as a NumPy array.
                - 'keypoints': The keypoints associated with the image as a NumPy array.
        
        Returns:
            dict: A dictionary containing:
                - 'image': The cropped image as a NumPy array.
                - 'keypoints': The adjusted keypoints relative to the cropped image.
        """
        
        # Extract the image and keypoints from the sample dictionary
        image, key_pts = sample['image'], sample['keypoints']

        # Get the height and width of the original image
        h, w = image.shape[:2]
        
        # Desired crop dimensions (new height and new width)
        new_h, new_w = self.output_size

        # Randomly select the top-left corner coordinates (top, left) for cropping
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Crop the image to the new dimensions starting from (top, left)
        image = image[top: top + new_h, left: left + new_w]

        # Adjust the keypoints to match the new origin (0,0) of the cropped image
        key_pts = key_pts - [left, top]

        # Return the cropped image and adjusted keypoints in a dictionary
        return {'image': image, 'keypoints': key_pts}
    

class ToTensor(object):
    """
    A class to convert NumPy ndarrays in a sample to PyTorch Tensors.
    
    This is useful for preparing image data and keypoints for input into a PyTorch model,
    as PyTorch models require input data to be in the form of Tensors.
    """

    def __call__(self, sample):
        """
        Apply the transformation to convert image and keypoints in the sample to PyTorch Tensors.
        
        Args:
            sample (dict): A dictionary containing:
                - 'image': The input image as a NumPy array.
                - 'keypoints': The keypoints associated with the image as a NumPy array.
        
        Returns:
            dict: A dictionary containing:
                - 'image': The image converted to a PyTorch Tensor.
                - 'keypoints': The keypoints converted to a PyTorch Tensor.
        """
        
        # Extract the image and keypoints from the sample dictionary
        image, key_pts = sample['image'], sample['keypoints']

        # Check if the image is grayscale (i.e., a 2D array)
        if len(image.shape) == 2:
            # If the image is grayscale, reshape it to add a channel dimension
            # This changes the shape from (H, W) to (H, W, 1)
            image = image.reshape(image.shape[0], image.shape[1], 1)
        
        # Transpose the image from H x W x C to C x H x W
        # This rearranges the image dimensions to the format expected by PyTorch (Channels, Height, Width)
        image = image.transpose((2, 0, 1))

        # Convert the image and keypoints from NumPy arrays to PyTorch Tensors
        image_tensor = torch.from_numpy(image)  # Convert image to a PyTorch Tensor
        keypoints_tensor = torch.from_numpy(key_pts)  # Convert keypoints to a PyTorch Tensor

        # Return the transformed image and keypoints as a dictionary
        return {'image': image_tensor, 'keypoints': keypoints_tensor}