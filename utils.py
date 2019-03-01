from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image as keras_image_process
from tqdm import tqdm
import tensorflow.contrib.eager as tfe
import tensorflow as tf
import functools
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
tf.enable_eager_execution()


class ContentAndStyleImage(object):
    def __init__(self, path_to_content_img, path_to_style_img):
        self.content_image = self._get_image(path_to_content_img)
        self.style_image = self._get_image(path_to_style_img)
        self.process_images()

    def _get_image(self, path_to_img):
        max_dimension = 512
        img = Image.open(path_to_img)
        long = max(img.size)
        scale = max_dimension / long
        img = img.resize(
            (round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
        img = keras_image_process.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def show_image(self):
        output_image = np.squeeze(self.image, axis=0)
        output_image = output_image.astype(np.uint32)
        plt.imshow(output_image)

    def process_images(self):
        self.processed_content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image)
        self.processed_style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image)

    def deprocess_image(self, processed_img):
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
        return img_to_unprocess


class Model(object):
    def __init__(self):
        self.content_layers = ['block5_conv2', 'block5_conv3']
        # Style layer we are interested in
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
            # 'block2_conv2',
        ]

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.get_vgg_model()

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
        return tf.reduce_mean(tf.square(content - target))

    def _get_gram_matrix(self, input_tensor):
        num_channels = int(input_tensor.shape[-1])
        input_vectors = tf.reshape(input_tensor, [-1, num_channels])
        num_vectors = tf.shape(input_vectors)[0]
        gram = tf.matmul(input_vectors, input_vectors, transpose_a=True)
        return gram / tf.cast(num_vectors, tf.float32)

    def _get_style_loss(self, style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = style.get_shape().as_list()
        gram_style = self._get_gram_matrix(style)

        # / (4. * (channels ** 2) * (width * height) ** 2)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def _get_total_variational_loss(self, content, ):
        return tf.reduce_sum(tf.image.total_variation(content))

    def _get_feature_representations(self, content_and_style_class):
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess both the content and style
      images from their path. Then it will feed them through the network to obtain
        the outputs of the intermediate layers.

        Arguments:
          model: The model that we are using.
          content_path: The path to the content image.
          style_path: The path to the style image

        Returns:
          returns the style features and the content features.
        """
        # Load our images in
        content_image = content_and_style_class.processed_content_image
        style_image = content_and_style_class.processed_style_image

        # batch compute content and style features
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        # Get the style and content feature representations from our model
        style_features = [style_layer[0]
                          for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer[0]
                            for content_layer in content_outputs[self.num_style_layers:]]
        return style_features, content_features

    def _compute_loss(self, loss_weights, init_image, gram_style_features, content_features, TA_weight):
        """This function will compute the loss total loss.
            ****Taken From Code Implementation****
        Arguments:
          model: The model that will give us access to the intermediate layers
          loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
          content_features: Precomputed outputs from defined content layers of
            interest.

        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = self.model(init_image)

        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]

        total_style_score = 0
        total_content_score = 0
        total_TA_score = 0
        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        averge_style_weight = 1.0 / float(self.num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            total_style_score += averge_style_weight * \
                self._get_style_loss(comb_style[0], target_style)

        # Accumulate content losses from all layers
        average_content_weight = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            total_content_score += average_content_weight * \
                self._get_content_loss(comb_content[0], target_content)
        average_TA_weight = 1.0 / float(self.num_content_layers)
        total_TA_score = self._get_total_variational_loss(
            init_image) * TA_weight
        total_style_score *= style_weight
        total_content_score *= content_weight

        # Get total loss
        total_loss = total_style_score + total_content_score + total_TA_score
        return total_loss, total_style_score, total_content_score

    def _compute_gradients(self, config):
        with tf.GradientTape() as tape:
            all_loss = self._compute_loss(**config)
        # Compute gradients wrt input image
            total_loss = all_loss[0]
            return tape.gradient(total_loss, config['init_image']), all_loss

    def run_style_transfer(self, content_and_style_class,
                           num_iterations=2000,
                           content_weight=1e0,
                           style_weight=1e2,
                           ta_weight=1):
        # trainable to false.
        # We don't need to (or want to) train any layers of our model, so we set their
        for layer in self.model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = self._get_feature_representations(
            content_and_style_class)
        gram_style_features = [self._get_gram_matrix(style_feature)
                               for style_feature in style_features]

        # Set initial image
        init_image = content_and_style_class.processed_content_image
        init_image = tf.Variable(init_image, dtype=tf.float32)
        # Create our optimizer
        opt = tf.train.AdamOptimizer(
            learning_rate=5, beta1=0.99, epsilon=1e-1)

        # For displaying intermediate images
        iter_count = 1

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (style_weight, content_weight)
        config = {
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features,
            "TA_weight": ta_weight,
        }

        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iterations / (num_rows * num_cols)
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        imgs = []
        for i in tqdm(range(num_iterations)):
            grads, all_loss = self._compute_gradients(config)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end_time = time.time()
            if loss < best_loss:

                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = content_and_style_class.deprocess_image(
                    init_image.numpy())
            if i % 100 == 0:
                imgs.append(content_and_style_class.deprocess_image(
                    (init_image.numpy())))
        print('Total time: {:.4f}s'.format(time.time() - global_start))
        plt.figure(figsize=(14, 4))
        fig, ax = plt.subplots(num_iterations // 100, 1)
        for i, img in enumerate(imgs):
            ax[i].imshow(img)
            # ax[i].xticks([])
            # ax[i].yticks([])
        fig.savefig("image")
        fig_best, ax_best = plt.subplots(1, 1)
        ax_best.imshow(best_img)
        fig_best.savefig("image_best")
        return best_img, best_loss
