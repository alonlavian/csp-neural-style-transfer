#!env/bin/python
"""
Module containing functions to apply nst on images in a directory

Arguments:
    --interactive: Launches the interactive CLI for modifying images
    --max_resolution: The resolution to scale both images to, Default: 512
    --border_size: The border to apply on the image in pixels, Default: 75
    --content_path: The path to the content image
    --content_directory: The path to the directory of the content images
    --iterations: The number of iterations to run the style transfer, Default: 1000
    --style_path: The path to the style image
    --help: Displays help message
Usage:
    ./modify_images.py --interactive
    ./modify_images.py --style_path [IMAGE] --content_path [IMAGE]
    ./modify_images.py --style_path [IMAGE] --content_directory [DIRECTORY]
"""

import os.path
import sys
from collections import namedtuple
import PIL
import tensorflow as tf
import numpy as np
from PyInquirer import prompt

import nst_model as nst
import add_border
import cli

C9_HOME_DIR = "/home/ubuntu/workspace/csp-python-source/csp-neural-style-transfer"
WINDOWS_HOME_DIR = "D:\\csp-neural-style-transfer"

def run_nst(content_path,
            style_path,
            model,
            num_iterations,
            max_resolution=512,
            border_size=75):
    """
    Function that runs nst
    Arguments:
        content_path: Path to content image
        style_path: Path to style image
        model: Instance of nst.NSTModel
        num_iterations: Total number of optimization steps to run
    Returns:
        Optimized image as np.array
    """
    images = nst.ContentAndStyleImage(content_path, style_path, max_resolution)
    best_img, _ = model.run_style_transfer(
        images, num_iterations=num_iterations)
    best_img_border = np.array(add_border.make_border(
        PIL.Image.fromarray(best_img), border_size))
    return best_img_border


def modify_directory(directory, style_path, iterations, max_resolution, border_size):
    """
    Function that runs run_nst on all images on a directory
    Arguments:
        directory: Directory containing image files
        style_path: Path to style image
        iterations: Total number of optimization steps to run
    Returns:
        None
    """
    model = nst.NSTModel()
    # obtains content image and all other images in its directory
    # directory = os.path.dirname(os.path.abspath(content_path))

    # new directory to store modified images
    _, style_tail = os.path.split(style_path)
    new_dir = os.path.join(directory, f"{style_tail[:-4]}_modified")
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        pass
    # for every file, modify and place in new directory
    for file in os.listdir(directory):
        if not os.path.isdir(file) and "modified" not in file:
            # ipdb.set_trace()
            try:
                new_img = run_nst(f"{directory}/{file}",
                                  style_path, model, iterations,
                                  border_size=border_size,
                                  max_resolution=max_resolution)
                _, tail = os.path.split(file)
                new_img = PIL.Image.fromarray(new_img)
                new_img_filename = os.path.join(
                    new_dir, f"modified_{tail}")  # f"{tail}_modified"
                new_img.save(new_img_filename)
            except KeyboardInterrupt:
                print("Action Cancelled By User")
                sys.exit()
            except FileNotFoundError:
                print("Please enter valid file paths")


def modify_image(content_path, style_path, iterations, max_resolution, border_size):
    """
    Function that runs run_nst on one image
    Arguments:
        content_path: Path to content image
        style_path: Path to style image
        iterations: Total number of optimization steps to run
    Returns:
        None
    """
    model = nst.NSTModel()
    try:
        new_img = run_nst(content_path,
                          style_path, model, iterations,
                          max_resolution=max_resolution, border_size=border_size)
        new_img = PIL.Image.fromarray(new_img)
        head, tail = os.path.split(content_path)
        new_img_filename = f"./{head}/modified_{tail}"  # f"{tail}_modified"
        new_img.save(new_img_filename)
    except KeyboardInterrupt:
        print("Action Cancelled By User")
        sys.exit()
    except FileNotFoundError:
        print("Please enter valid file paths")


# def run_all():
#     """
#     Runs style transfer for all content images and style images
#     """
#     for i in os.listdir("./style_images_2"):
#         modify_directory("./content_images", f"./style_images_2/{i}", 1000)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string(
        "content_directory", None, "Directory of images to apply transformation")
    tf.app.flags.DEFINE_string(
        "style_path", None,
        "Path to the style image")
    tf.app.flags.DEFINE_string(
        "content_path", None, "Image to apply transformation on")
    tf.app.flags.DEFINE_integer(
        "max_resolution", 512, "Maximum Resolution")
    tf.app.flags.DEFINE_integer(
        "iterations", 1000, "Number of iterations to run optimizations for")
    tf.app.flags.DEFINE_integer(
        "border_size", 75, "border size to add, 0 for none")
    tf.app.flags.DEFINE_bool("interactive", False, "enable interactive prompt")
    tf.app.flags.DEFINE_bool("help", False, "flag to get usage help")
    ARGS = tf.app.flags.FLAGS
    try:
        # if ARGS.all:
        #     run_all()
        if os.getcwd() not in [C9_HOME_DIR, WINDOWS_HOME_DIR]:
            sys.exit("Please run modify_images.py from the directory it resides in")
        elif ARGS.help:
            print(__doc__)
        elif ARGS.interactive:
            ANSWER = prompt(**cli.return_cli())
            NAMED_ANSWERS = namedtuple("Arguments", ANSWER.keys())(*ANSWER.values())
            modify_directory(NAMED_ANSWERS.content_directory,
                                             NAMED_ANSWERS.style_path,
                                             NAMED_ANSWERS.iterations,
                                             NAMED_ANSWERS.max_resolution,
                                             NAMED_ANSWERS.border_size)
        elif ARGS.style_path:
            if ARGS.content_directory:
                modify_directory(ARGS.content_directory,
                                 ARGS.style_path,
                                 ARGS.iterations,
                                 ARGS.max_resolution,
                                 ARGS.border_size)
            elif ARGS.content_path:
                modify_image(ARGS.content_path,
                             ARGS.style_path,
                             ARGS.iterations,
                             ARGS.max_resolution,
                             ARGS.border_size)
            else:
                print("Please specify either content image or content directory")
        else:
            print("Please specify style path or use flag interactive")
            print(__doc__)
    except AttributeError:
        print("Cleaning up...")
