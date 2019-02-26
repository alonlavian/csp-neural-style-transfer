from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image as keras_image_process
import tensorflow.contrib.eager as tfe
import tensorflow as tf
import functools
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


def load_img_file(path_to_img):
    max_dimension = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize(
        (round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = keras_image_process.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def show_image_with_title(img, title=None):
    output_image = np.squeeze(img, axis=0)
    output_image = out.astype(np.uint32)
    if title is not None:
        plt.title(title)
    plt.imshow(output_image)


class ContentImage(object):
    def __init__(self, path_to_img):
        max_dimension = 512
        img = Image.open(path_to_img)
        long = max(img.size)
        scale = max_dim / long
        img = img.resize(
            (round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
        img = keras_image_process.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        self.image = img

    def show_image(self):
        output_image = np.squeeze(self.image, axis=0)
        output_image = output_image.astype(np.uint32)
        plt.imshow(output_image)

    def process_image(self):
        self.processed_image = tf.keras.applications.vgg19.preprocess_input(
            self.image)

    def deprocess_image(self):
        img_to_unprocess = processed_img.copy()
        if len(img_to_unprocess.shape) == 4:
            img_to_unprocess = np.squeeze(img_to_unprocess, 0)
        assert len(img_to_unprocess.shape) == 3, ("Input to deprocess image must be an image of "
                                                  "dimension [1, height, width, channel] or [height, width, channel]")
        if len(img_to_unprocess.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")
        img_to_unprocess[:, :, 0] += 103.939
        img_to_unprocess[:, :, 1] += 116.779
        img_to_unprocess[:, :, 2] += 123.68
        img_to_unprocess = img_to_unprocess[:, :, ::-1]

        img_to_unprocess = np.clip(img_to_unprocess, 0, 255).astype('uint8')
        self.unprocessed_image = img_to_unprocess


class Model(object):
    def __init__(self):
        self.content_layers = ['block5_conv2']
        # Style layer we are interested in
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
        ]

        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

    def get_vgg_model(self):
        """ Creates our model with access to intermediate layers.

        This function will load the VGG19 model and access the intermediate layers.
        These layers will then be used to create a new model that will take input image
        and return the outputs from these intermediate layers from the VGG model.

        Returns:
          returns a keras model that takes image inputs and outputs the style and
            content intermediate layers.
        """
        # Load our model. We load pretrained VGG, trained on imagenet data
        self.vgg_model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet')
        self.vgg_model.trainable = False
        # Get output layers corresponding to style and content layers
        self.style_outputs = [self.vgg_model.get_layer(
            name).output for name in self.style_layers]
        self.content_outputs = [self.vgg_model.get_layer(
            name).output for name in self.content_layers]
        self.model_outputs = self.style_outputs + self.content_outputs
        # Build model
        self.model = models.Model(self.vgg_model.input, self.model_outputs)

    def _get_content_loss(self, content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def gram_matrix(input_tensor):
        num_channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)
