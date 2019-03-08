"""
Module containing functions to apply nst on images in a directory
"""

import argparse
import os.path
import sys
import PIL
import tensorflow as tf
import test as nst


def run_nst(content_path, style_path, model, num_iterations):
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
    images = nst.ContentAndStyleImage(content_path, style_path)
    best_img, _ = model.run_style_transfer(
        images, num_iterations=num_iterations)
    return best_img


def modify_directory(directory, style_path, num_iterations):
    """
    Function that runs run_nst on all images on a directory
    Arguments:
        directory: Directory containing image files
        style_path: Path to style image
        num_iterations: Total number of optimization steps to run
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
                                  style_path, model, num_iterations)
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


def modify_image(content_path, style_path, num_iterations):
    """
    Function that runs run_nst on one image
    Arguments:
        content_path: Path to content image
        style_path: Path to style image
        num_iterations: Total number of optimization steps to run
    Returns:
        None
    """
    model = nst.NSTModel()
    try:
        new_img = run_nst(f"{content_path}",
                          style_path, model, num_iterations)
        new_img = PIL.Image.fromarray(new_img)
        head, tail = os.path.split(content_path)
        new_img_filename = f"./{head}/modified_{tail}"  # f"{tail}_modified"
        new_img.save(new_img_filename)
    except KeyboardInterrupt:
        print("Action Cancelled By User")
        sys.exit()
    except FileNotFoundError:
        print("Please enter valid file paths")


def run_all():
    for i in os.listdir("./style_images"):
        modify_directory("./content_images", f"./style_images/{i}", 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    tf.app.flags.DEFINE_string(
        "content_directory", None, "Directory of images to apply transformation")
    tf.app.flags.DEFINE_string(
        "style_path", None,
        "Path to the style image")
    tf.app.flags.DEFINE_string(
        "content_path", None, "Image to apply transformation on")
    tf.app.flags.DEFINE_integer(
        "iterations", 1000, "Number of iterations to run optimizations for")
    tf.app.flags.DEFINE_bool("all", False, "")
    args = tf.app.flags.FLAGS
    # parser.add_argument("--directory", "-d", type=str)
    # parser.add_argument("--style_path", "-s", type=str)
    # parser.add_argument("--iterations", "-i", type=int, default=1000)
    # args = parser.parse_args()
    if args.all:
        run_all()
    if args.style_path:
        if args.content_directory:
            modify_directory(args.content_directory,
                             args.style_path, args.iterations)
        elif args.content_path:
            modify_image(args.content_path, args.style_path, args.iterations)
        else:
            print("Please specify either content image or content directory")
    else:
        print("Please specify style path")
