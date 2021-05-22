import tensorflow as tf


class FeatureExtractor(object):
  
  @classmethod
  def custom_model(cls):
    filters = [64, 128, 64, 32]
    layer_names = [f"conv_block_{i}" for i in range(2, 2 + len(filters))]

    input_layer = tf.keras.layers.Input(shape = (256, 256, 3),
                                        name = "input_layer")

    x = tf.keras.layers.Conv2D(filters = 32,
                               kernel_size = (2, 2),
                               strides = (1, 1),
                               padding = "same",
                               name = "conv_block_1")(input_layer)
                   
    layer_num = len(layer_names) or len(filters)
    activation_layers = list()
    for i in range(layer_num):
      if i % 2 == 1 and i != 0:
        x = tf.keras.layers.ReLU(name = f"relu_layer_{i}th_iter")(x)
        x = tf.keras.layers.BatchNormalization()(x)

      x = tf.keras.layers.Conv2D(filters = filters[i],
                               kernel_size = (3, 3),
                               strides = (1, 1),
                               padding = "same",
                               name = layer_names[i])(x)
      
      activation_layers.append(x)
                                  
    out = tf.keras.layers.Conv2D(filters = 3,
                               kernel_size = (3, 3),
                               strides = (2, 2),
                               padding = "same",
                               name = "conv_block_out")(x)

    activation_layers.append(out)

    model = tf.keras.models.Model(inputs = input_layer,
                                  outputs = activation_layers,
                                  name = "feature_extractor_model")

    return model         


  @classmethod
  def vgg_extractor_model(cls):
    vgg_model = tf.keras.applications.VGG19(include_top = False,
                                            weights = "imagenet")
    
    style_conv_blocks = [f"block{i}_conv1" for i in range(1, 6)]
    content_conv_block = ["block5_conv2"]
    all_activation_layers = style_conv_blocks + content_conv_block

    input_layer = vgg_model.inputs
    output_layers = [vgg_model.get_layer(i).output for i in all_activation_layers]
    
    model = tf.keras.models.Model(inputs = input_layer,
                                  outputs = output_layers,
                                  name = "vgg19_extractor")

    return model

  
  @classmethod
  def vgg_with_recurrent(cls):
    input_layer = tf.keras.layers.Input(shape = (256, 256, 3))
    vgg_layer = tf.keras.applications.VGG19(include_top = False,
                                            weights = "imagenet")(input_layer)

    content_activation = tf.keras.layers.ConvLSTM2D(64, (1, 1), (1, 1),
                                                    padding = "same")    
    
    num_layers = 1
    recurrent_layer_names = [f"rnn_conv_{i}" for i in range(1, num_layers)]
    recurrent_layers = list()
    for layer_name in recurrent_layer_names:
      rnn_layer = tf.keras.layers.ConvLSTM2D(64, (1, 1), (1, 1), padding = "same",
                                             name = layer_name)
      recurrent_layers.append(rnn_layer)

    x = input_layer
    model_outputs = list()
    for rnn_layer in recurrent_layers:
      x = tf.expand_dims(x, axis = 0)
      x = rnn_layer(x)
      model_outputs.append(x)

    x = tf.expand_dims(x, axis = 0)
    out = content_activation(x)
    model_outputs.append(out)
    
    model = tf.keras.models.Model(inputs = input_layer,
                                  outputs = model_outputs,
                                  name = "vgg_19_with_recurrent")
    
    return model
    

  @classmethod
  def extract(cls, image_stack, model):
    """Image stack is (3, None, None, 3)
    shaped image data which contains
    content, style and generated images"""
    return model()(image_stack)

  
  @staticmethod
  def get_layers(features):
    content = features[-1]
    style = features[:-1]
    
    return content, style