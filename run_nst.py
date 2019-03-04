"""
Module containing functions to apply nst on images in a directory
"""

import argparse
import os.path
import sys
import PIL
import tensorflow as tf
import nst_model as nst


def run_nst(content_path, style_path, model, num_iterations):
    ''' docstring :) '''
    images = nst.ContentAndStyleImage(content_path, style_path)
    best_img, _ = model.run_style_transfer(
        images, num_iterations=num_iterations)
    return best_img


def modify(directory, style_path, num_iterations):
    ''' docstring :) '''
    model = nst.NSTModel()
    tf.logging.set_verbosity(tf.logging.FATAL)
    # obtains content image and all other images in its directory
    # directory = os.path.dirname(os.path.abspath(content_path))

    # new directory to store modified images
    new_dir = os.path.join(directory, 'modified')
    try:
        os.mkdir(new_dir)
    except FileExistsError:
        pass
    # for every file, modify and place in new directory
    for file in os.listdir(directory):
        if not os.path.isdir(file) and file != "modified":
            # ipdb.set_trace()
            try:
                new_img = run_nst(f"{directory}/{file}",
                                  style_path, model, num_iterations)
                _, tail = os.path.split(file)
                new_img = PIL.Image.fromarray(new_img)
                new_img_filename = os.path.join(
                    new_dir, tail)  # f"{tail}_modified"
                new_img.save(new_img_filename)
            except KeyboardInterrupt:
                print("Action Cancelled By User")
                sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    tf.app.flags.DEFINE_string(
        "directory", "./tmp/nst/", "Directory of images to apply transformation")
    tf.app.flags.DEFINE_string(
        "style_path", "./tmp/nst/The_Great_Wave_off_Kanagawa.jpg",
        "Path to the style image")
    tf.app.flags.DEFINE_integer(
        "iterations", 1000, "Number of iterations to run optimizations for")
    args = tf.app.flags.FLAGS
    # parser.add_argument("--directory", "-d", type=str)
    # parser.add_argument("--style_path", "-s", type=str)
    # parser.add_argument("--iterations", "-i", type=int, default=1000)
    # args = parser.parse_args()
    modify(args.directory, args.style_path, args.iterations)
