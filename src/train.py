import tensorflow as tf

from loss import Loss
from feature_extractor import FeatureExtractor


class Constants(object):
  CONTENT_WEIGHT = 2e-5
  STYLE_WEIGHT = 1e-4


class Train(Loss, FeatureExtractor):
  
  @classmethod
  def calculate_step_loss(cls, model, content, style, generated):
    tensor = tf.concat([content, style, generated], axis = 0)
    features = cls.extract(image_stack = tensor, model = model)
    content_act, style_act = cls.get_layers(features)

    content_loss = cls.content_loss(content_act[0], content_act[-1])

    style_loss = 0.
    for layer in style_act:
      layer_loss = cls.style_loss(layer[1], layer[-1])
      style_loss += layer_loss

    loss = (content_loss * Constants.CONTENT_WEIGHT) \
           + (style_loss * Constants.STYLE_WEIGHT)

    return loss


def train(model, content, style, generated, epochs = 10):
  optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-4) 
  for epoch in range(epochs):
    with tf.GradientTape() as GT:
      loss = Train.calculate_step_loss(model, content, style, generated)

    print(f"EPOCH: {epoch + 1} \nLOSS: {loss}\n" + ("---" * 15))

    gradients = GT.gradient(loss, generated)
    optimizer.apply_gradients([(gradients, generated)])

  return generated