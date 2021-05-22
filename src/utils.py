from skimage import io
import tensorflow as tf


def load_image(path: str):
  image = io.imread(path)
  expanded = tf.expand_dims(tf.cast(tf.convert_to_tensor(image),
                                    tf.float32) / 255., axis = 0)

  return expanded