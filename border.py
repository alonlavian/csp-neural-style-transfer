"""
Module that adds a magnified border to the image
"""
import PIL
from PIL import Image
import numpy as np

def get_magnified_img(image, resize_by=20):
    """
    Gets the magnified image
    Arguments:
        image: PIL image object
        resize_by: num pixels by which to resize the image. 
    Returns:
        PIL image object with larger image for the border
    """
    resized_image = image.resize(tuple(map(lambda dim: dim+resize_by*2, image.size)))
    return resized_image


def make_border(image, border_size): 
    """
    Makes a border of the image by making placing the image over a magnified 
    version of the image. 
    Arguments: 
        image: PIL image object 
        border_size: integer (specifies width in pixels of border)
    Returns: 
        PIL image object the size of the border image created by get_magnified_img
    """ 
    border_img = get_magnified_img(image, border_size)
    border_img.paste(image, (border_size, border_size), mask=None)
    return border_img
    
def main():
    IMAGE_PATH = "content_images/Mona_Lisa.jpg"
    img = PIL.Image.open(IMAGE_PATH)
    new_img = make_border(img, 75)
    new_img.save("test.jpg")
if __name__ == '__main__':
    main()    