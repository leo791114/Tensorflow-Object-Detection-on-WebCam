import numpy as np

#
# Transform image to numpy array
#


def load_image_into_numpy_array(image):
    '''
    Function to transform image opened by PIL's Image to numpy array
    Args:
    image: image opened by PIL's Image
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
