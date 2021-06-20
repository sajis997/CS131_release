import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out

def down_sample(image,x_factor,y_factor):
    
    gray_image = rgb2gray(image)

    # get the number of the rows and columns of the image
    nrows,ncols = gray_image.shape
    '''
    Specify the new image dimensions we want for our smallest output image

    In this case we shall downsample the image by a fixed ratio
    Since the ratios are different, the mage will appear distorted

    We can also set the x_new and y_new to arbitrary values, it will NOT 
    work if they are larger than nrows and ncols. That would be upsampling / interpolation
    '''
    assert x_factor < ncols, 'X factor is larger than the total number of columns'
    assert y_factor < nrows, 'Y factory is larger than the total number of rows'
    
    new_nrows = nrows // y_factor
    new_ncols = ncols // x_factor
    
    # Determine the ratio of the old dimension compared to the new dimensions
    y_scale = nrows / new_nrows
    x_scale = ncols / new_ncols
    
    # Declare and initialize an output image buffer
    out = np.zeros((new_nrows,new_ncols))

    
    for y in range(new_nrows):
        for x in range(new_ncols):
            out[y,x] = gray_image[int(y*y_scale),int(x*x_scale)]

    
    return out

def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = np.power(image,2) * 0.5
    ### END YOUR CODE

    return out

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989, 0.5870, 0.1140])


def bilinear_resize(input_image,output_rows, output_cols):
    """Resize an image using the bilinear interpolation method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """    
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    x_ratio = float(input_cols - 1) / (output_cols - 1) if output_cols > 1 else 0
    y_ratio = float(input_rows - 1) / (output_rows - 1) if output_rows > 1 else 0

    for i in range(output_rows): # height
        for j in range(output_cols): # width

            x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
            x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = input_image[y_l,x_l]
            b = input_image[y_l,x_h]
            c = input_image[y_h,x_l]
            d = input_image[y_h,x_h]

            output_image[i,j] = a * (1 - x_weight) * (1 - y_weight)  \
                                + b * x_weight * (1 - y_weight) + \
                                  c * y_weight * (1 - x_weight) + \
                                  d * x_weight * y_weight

    return output_image    

def linear_resize(in_array, out_size):
    """
        `in_array` is the input array
        `out_size` is the desired size
    """

    ratio = (len(in_array) - 1) / (out_size - 1)
    out_array = []

    for i in range(out_size):
        low = math.floor(ratio * i)
        high = math.ceil(ratio * i)

        weight = ratio * i - low

        a = in_array[low]
        b = in_array[high]

        out_array.append(a * (1 - weight) + b * weight)

    return out_array

def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    '''
    for i in range(0,output_rows):     # height
        for j in range(0,output_cols): # width
            row_index = min(round(i * (input_rows/output_rows)), input_rows - 1)  # row index
            col_index = min(round(j * (input_cols/output_cols)), input_cols - 1)  # column index
            output_image[i,j] = input_image[row_index,col_index]
    '''
    
    for i in range(output_rows):     # height
        for j in range(output_cols): # width
            row_index = min(int(round( float(i) / float(output_rows) * float(input_rows))), input_rows - 1)  # row index
            col_index = min(int(round( float(j) / float(output_cols) * float(input_cols))), input_cols - 1)  # column index
            output_image[i,j] = input_image[row_index,col_index]        
    
    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!
    R = np.array([
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1] ] )
    ## YOUR CODE HERE
    pass
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    pass
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
