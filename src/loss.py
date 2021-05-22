import tensorflow as tf


class Loss(object):
  
  @classmethod
  def gram_matrix(cls, arr):
    """Gramian matrix for calculating style loss"""
    x = tf.transpose(arr, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))

    return gram


  @classmethod
  def content_loss(cls, content, generated):
    """1/2 * sum of (generated - original) ** 2"""
    content_loss = tf.reduce_sum(tf.square((generated - content)))
    
    return content_loss * 5e-1


  @classmethod
  def style_loss(cls, style, generated):
    style_gram = cls.gram_matrix(style)
    generated_gram = cls.gram_matrix(generated)

    style_loss = tf.reduce_mean(tf.square(generated_gram - style_gram))

    return style_loss