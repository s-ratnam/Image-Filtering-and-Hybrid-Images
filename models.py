"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_1D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    """Creates a 1D Gaussian kernel using the specified standard deviation.

    Note: ensure that the value of the kernel sums to 1.

    Args:
        standard_deviation (float): standard deviation of the gaussian

    Returns:
        torch.FloatTensor: required kernel as a row vector
    """

    kernel = torch.FloatTensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    

    standard_deviation = int(standard_deviation)
    k = int((4. * standard_deviation) + 1.)
    mean = int(math.floor(k/2.))
    
  
    x = torch.linspace(0, k-1, steps = k)
    
  
   
    kernel = (torch.exp(torch.neg(torch.pow(x - mean, 2.0) / (2.0 * standard_deviation**2.))) *
    (1.0 / (standard_deviation * math.sqrt(2.0 * 3.1415))))


    kernel = kernel / torch.sum(kernel)
    
    # print(kernel)
  
    

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return kernel
  


def create_2D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    """Creates a 2D Gaussian kernel using the specified standard deviation in
    each dimension, and no cross-correlation between dimensions,

    i.e. 
    sigma_matrix = [standard_deviation^2    0
                    0                       standard_deviation^2]


    The kernel should have:
    - shape (k, k) where k = standard_deviation * 4 + 1
    - mean = floor(k / 2)
    - values that sum to 1

    Args:
        standard_deviation (float): the standard deviation along a dimension

    Returns:
        torch.FloatTensor: 2D Gaussian kernel

    HINT:
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      vectors drawn from 1D Gaussian distributions.
    """

    kernel_2d = torch.Tensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################


    # take the outer product of two vectors 
    kernel_2d = torch.ger(create_1D_Gaussian_kernel(standard_deviation), create_1D_Gaussian_kernel(standard_deviation))

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return kernel_2d
    # eturn output


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff standard deviation.

        PyTorch requires the kernel to be of a particular shape in order to apply
        it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        where c is the # channels in the image. Start by getting a 2D Gaussian
        kernel using your implementation from Part 1, which will be of shape
        (k, k). Then, let's say you have an RGB image, you will need to turn this
        into a Tensor of shape (3, 1, k, k) by stacking the Gaussian kernel 3
        times.

        Args
        - cutoff_standarddeviation: int specifying the cutoff standard deviation
        Returns
        - kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel() function from part1.py in this
          function.
        - Since the # channels may differ across each image in the dataset, make
          sure you don't hardcode the dimensions you reshape the kernel to. There
          is a variable defined in this class to give you channel information.
        - You can use torch.reshape() to change the dimensions of the tensor.
        - You can use torch's repeat() to repeat a tensor along specified axes.
        """
        kernel = torch.Tensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################

        # start by getting a 2D gaussian Kernel with implementation from part1 
        kernel = create_2D_Gaussian_kernel(cutoff_standarddeviation)

        # goal is to reshape kernel as (c, 1, k, k)
        # Find c 
        c = self.n_channels
        # Find k of kernel returned by create_2D_Gaussian_kernel
        k = list(kernel.shape)[0]
        # reshape and repeat 
        kernel = torch.reshape(kernel, (1, 1, k, k))
        kernel = kernel.repeat((c, 1, 1, 1))

        

        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return kernel

    def low_pass(self, x, kernel):
        """
        Applies low pass filter to the input image.

        Args:
        - x: Tensor of shape (b, c, m, n) where b is batch size
        - kernel: low pass filter to be applied to the image
        Returns:
        - filtered_image: Tensor of shape (b, c, m, n)

        HINT:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the filter
          will be applied to.
        """

        filtered_image = torch.Tensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################

        pad = (list(kernel.shape)[2]//2 , list(kernel.shape)[3]//2)
        filtered_image = torch.nn.functional.conv2d(x, kernel, padding = pad, groups = self.n_channels)

        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the hybrid
        image.

        Args
        - image1: Tensor of shape (b, m, n, c)
        - image2: Tensor of shape (b, m, n, c)
        - cutoff_standarddeviation: Tensor of shape (b)
        Returns:
        - low_frequencies: Tensor of shape (b, m, n, c)
        - high_frequencies: Tensor of shape (b, m, n, c)
        - hybrid_image: Tensor of shape (b, m, n, c)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function in
          this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
          use torch.clamp().
        - If you want to use images with different dimensions, you should resize
          them in the HybridImageDataset class using torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        low_frequencies = torch.Tensor()
        high_frequencies = torch.Tensor()
        hybrid_image = torch.Tensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################

        filter_f = self.get_kernel(cutoff_standarddeviation)
        low_frequencies = self.low_pass(image1, filter_f)
        high_frequencies = torch.sub(image2, self.low_pass(image2, filter_f))
        image_to_conv = torch.add(low_frequencies, high_frequencies)
        hybrid_image = torch.clamp(image_to_conv, 0, 1)

        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return low_frequencies, high_frequencies, hybrid_image
