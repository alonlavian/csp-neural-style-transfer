"""
Module that adds a magnified border to the image

Usage:
    --image: Image to apply border on
"""
import argparse
import PIL
from PIL import Image, ImageFilter


def get_magnified_img(image, resize_by=20):
    """
    Gets the magnified image
    Arguments:
        image: PIL image object
        resize_by: num pixels by which to resize the image.
    Returns:
        PIL image object with larger image for the border
    """
    resized_image = image.resize(
        tuple(map(lambda dim: dim + resize_by * 2, image.size)))
    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(10))
    return blurred_image


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


def main(image_path):
    """
    Function for command line execution
    """
    img = Image.open(image_path)
    new_img = make_border(img, 75).convert("RGBA")
    new_img.save("test.png")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--image", "-i")
    ARGS = PARSER.parse_args()
    main(ARGS.image)
