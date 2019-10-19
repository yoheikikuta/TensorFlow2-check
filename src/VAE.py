import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.input_layer = layers.InputLayer(input_shape=(28, 28, 1))
        self.conv2d_1 = layers.Conv2D(filters=32, kernel_size=3,
                                      strides=(2, 2), activation='relu')
        self.conv2d_2 = layers.Conv2D(filters=64, kernel_size=3,
                                      strides=(2, 2), activation='relu')
        self.flatten = layers.Flatten()
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.flatten(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        return z_mean, z_log_var


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, latent_dim=32, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.input_layer = layers.Dense(7 * 7 * 32, activation='relu',
                                        input_shape=(latent_dim,))
        self.reshaped_input = layers.Reshape((7, 7, 32))
        self.convtr2d_1 = layers.Conv2DTranspose(filters=64,
                                                 kernel_size=3,
                                                 strides=(2, 2),
                                                 padding="SAME",
                                                 activation='relu')
        self.convtr2d_2 = layers.Conv2DTranspose(filters=32,
                                                 kernel_size=3,
                                                 strides=(2, 2),
                                                 padding="SAME",
                                                 activation='relu')
        self.convtr2d_3 = layers.Conv2DTranspose(filters=1,
                                                 kernel_size=3,
                                                 strides=(1, 1),
                                                 padding="SAME")

    def call(self, inputs, apply_sigmoid=False):
        x = self.input_layer(inputs)
        x = self.reshaped_input(x)
        x = self.convtr2d_1(x)
        x = self.convtr2d_2(x)
        x = self.convtr2d_3(x)
        if apply_sigmoid:
            return tf.sigmoid(x)
        return x


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, latent_dim=32, name='autoencoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.sampling = Sampling()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z)
        return z_mean, z_log_var, z, reconstructed


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    z_mean, z_log_var, z, reconstructed = model(x)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=reconstructed, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, z_mean, z_log_var)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train(vae, optimizer, test_loss, train_dataset, test_dataset):
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            loss = compute_loss(vae, x_batch_train)

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    for test_x in test_dataset:
        test_loss(compute_loss(vae, test_x))


if __name__ == "__main__":
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(
        test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

    # Training
    start = time.time()
    vae = VariationalAutoEncoder(50)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(3):
        print(f"Start of epoch {epoch + 1}")
        test_loss = tf.keras.metrics.Mean()
        train(vae, optimizer, test_loss, train_dataset, test_dataset)
        print(f"  test loss: {test_loss.result()}")

    end = time.time()
    print(f"TRAINING TIME: {end - start} [sec]")
