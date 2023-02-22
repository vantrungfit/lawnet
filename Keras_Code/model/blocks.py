import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras import regularizers
from model.spp import SpatialPyramidPooling

class NetBlock:
    def __init__(self, config):
        self.use_bias = False if config.use_bias==0 else True
        self.weight_decay = config.weight_decay
        self.kernel_initializer = config.kernel_initializer
        self.kernel_regularizer = regularizers.l2(self.weight_decay)
        self.kernel_constraint = MaxNorm(1.0)
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.shared_axes = [2, 3] if K.image_data_format() == 'channels_first' else [1, 2]
        
    def _norm_layer(self, inputs, norm='batch', name=None):
        if(norm=='batch'):
            x = BatchNormalization(axis=self.channel_axis, name=name)(inputs)
        else:
            x = inputs
        return x
    
    def make_divisible(self, v, divisor, min_value=None):
        # This function is taken from the original tf repo.
        # It ensures that all layers have a channel number that is divisible by 8
        # It can be seen here:
        # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _hard_swish(self, x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0
    
    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[channel_axis])

        x = GlobalAveragePooling2D()(inputs)
        x = Reshape((1, 1, input_channels))(x)
        x = conv_block(x, input_channels//4, 1, 1, padding='same', norm=None, activation='relu')
        x = conv_block(x, input_channels, 1, 1, padding='same', norm=None, activation='hard_sigmoid')
        x = Multiply()([inputs, x])

        return x
    
    def _activation_layer(self, x, activation='relu'):

        if activation == 'relu':
            x = ReLU()(x)
        elif activation == 'relu6':
            x = ReLU(max_value=6.0)(x)
        elif activation == 'prelu':
            x = PReLU(shared_axes=self.shared_axes)(x)
        elif activation == 'mish':
            x = tfa.activations.mish(x)
        elif activation == 'hard_swish':
            x = self._hard_swish(x)
        elif activation == None:
            x = x
        else:
            x = Activation(activation)(x)

        return x

    def cnn_block(self, inputs, filters, maxpooling=False, dropout=0.0, name=None):

        x = Conv2D(filters, 3, strides = 1, 
                   padding = "same",
                   activation='relu',
                   use_bias = self.use_bias,
                   kernel_initializer = self.kernel_initializer,
                   kernel_regularizer = self.kernel_regularizer,
                   kernel_constraint = self.kernel_constraint,
                   name=name
                  )(inputs)
       
        x = self._norm_layer(x, norm=norm)
        
        if(maxpooling):
            x = MaxPooling2D((2,2), strides=(2,2))(x)
        
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
    
    return x
    
    def conv_block(self, inputs, filters, kernel_size, strides, padding='same', norm='batch', activation='relu', dropout=0.0):

        x = Conv2D(filters, kernel_size, strides = strides, 
                   padding = padding, use_bias = self.use_bias,
                   kernel_initializer = self.kernel_initializer,
                   kernel_regularizer = self.kernel_regularizer,
                   kernel_constraint = self.kernel_constraint
                  )(inputs)
       
        x = self._norm_layer(x, norm=norm)
        x = self._activation_layer(x, activation=activation)
        
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
            
        return x

    def separable_conv_block(self, inputs, filters, kernel_size, strides, padding='same', norm='batch', activation='relu', dropout=0.0):

        x = SeparableConv2D(filters, kernel_size, strides = strides, 
                            padding = padding, use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            kernel_regularizer = self.kernel_regularizer,
                            kernel_constraint = self.kernel_constraint
                           )(inputs)
        x = self._norm_layer(x, norm=norm)
        x = self._activation_layer(x, activation=activation)
            
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
        return x

    def depthwise_conv_block(self, inputs, kernel_size, strides, padding='same', norm='batch', activation='relu', dropout=0.0):
        
        x = DepthwiseConv2D(kernel_size, strides = strides, padding = padding, 
                            depth_multiplier = 1, use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            kernel_regularizer = self.kernel_regularizer,
                            kernel_constraint = self.kernel_constraint
                           )(inputs)

        x = self._norm_layer(x, norm=norm)
        x = self._activation_layer(x, activation=activation)
        
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
            
        return x
    
    def spp_block(self, inputs, pool_list=[1, 2, 4], pool_type='max', norm='batch', fc=128):
        x = SpatialPyramidPooling(pool_list=pool_list, pool_type=pool_type, keepdims=True)(inputs)
        x = self._norm_layer(x, norm=norm)
        x = self.conv_block(x, fc, kernel_size=1, strides=1, padding='valid', norm=norm, activation=None)
        
        return x
    
    def pspp_block(self, inputs, pool_list=[1, 2, 4], pool_type='max', norm='batch', fc=128):
        
        out = []
        
        for bins in pool_list:
            x = SpatialPyramidPooling(pool_list=[bins], pool_type=pool_type, keepdims=True)(inputs)
            x = self._norm_layer(x, norm=norm)
            x = self.conv_block(x, fc, kernel_size=1, strides=1, padding='valid', norm=norm, activation=None)
            out.append(x)
            
        x = Concatenate()(out)

        return x
    
    def bottleneck_v1(self, inputs, c, ks, s, alpha=1.0, norm='batch', activation='relu'):
        
        c = int(c*alpha)
        
        x = self.depthwise_conv_block(inputs, ks, s, padding='same', norm=norm, activation=activation)
        x = self.conv_block(x, c, kernel_size=1, strides=1, norm=norm, activation=activation)

        return x

    def bottleneck_v2(self, inputs, c, ks, t, s, r = False, norm='batch', activation='relu6'):
        
        tchannel = K.int_shape(inputs)[self.channel_axis] * t
        
        x = self.conv_block(inputs, tchannel, kernel_size=1, strides=1, padding='same', norm=norm, activation=activation)
        x = self.depthwise_conv_block(x, ks, s, padding='same', norm=norm, activation=activation)
        x = self.conv_block(x, c, kernel_size=1, strides = 1, padding = 'same', norm=norm, activation=None)
        
        if r:
            x = add([x, inputs])
        
        return x

    def inverted_residual_block(self, inputs, c, ks, t, s, n, norm='batch', activation='relu', alpha=1.0):
        
        c = self.make_divisible(c*alpha, 8)
        x = self.bottleneck_v2(inputs, c, ks, t, s, r=False, norm=norm, activation=activation)
        
        for i in range(1, n):
            x = self.bottleneck_v2(x, c, ks, t, 1, r=True, norm=norm, activation=activation)
        
        return x
    
    def bottleneck_v3(self, inputs, c, ks, t, s, alpha=1.0, squeeze=False, norm='batch', activation='relu'):

        input_shape = K.int_shape(inputs)
        tchannel = int(t)
        cchannel = int(alpha*filters)
        r = s==1 and input_shape[channel_axis]==c

        x = self.conv_block(inputs, tchannel, 1, 1, norm=norm, activation=activation)
        x = self.depthwise_conv_block(x, ks, s, padding='same', norm=norm, activation=activation)

        if squeeze:
            x = self._squeeze(x)

        x = self.conv_block(x, cchannel, kernel_size=1, strides=1, norm=norm, activation=None)

        if r:
            x = Add()([x, inputs])

        return x

    def CA(self, x, ratio=4):
        [n, h, w, c] = x.shape
        reduction_channels = max(8, c//ratio)
        output_channels = c
        residual = x

        x_h = AveragePooling2D(pool_size=(h, 1), strides=1)(x)
        x_w = AveragePooling2D(pool_size=(1, w), strides=1)(x)

        x_w = Permute((2, 1, 3))(x_w)
        y = K.concatenate((x_h, x_w), axis=2)

        # Reduction
        y = self.conv_block(y, filters=reduction_channels, kernel_size=1, strides=1, padding='valid', norm='batch', activation='hard_swish')
        
        # Split
        x_h, x_w = tf.split(y, axis=2, num_or_size_splits=[h, w])
        x_w = Permute((2, 1, 3))(x_w)

        # Expansion
        a_h = self.conv_block(x_h, filters=output_channels, kernel_size=1, strides=1, padding='valid', norm=None, activation='sigmoid')
        a_w = self.conv_block(x_w, filters=output_channels, kernel_size=1, strides=1, padding='valid', norm=None, activation='sigmoid')

        # Reweight
        a_h = tf.tile(a_h, [1, h, 1, 1])
        a_w = tf.tile(a_w, [1, 1, w, 1])
        out = multiply([residual, a_w, a_h])
    
        return out
    
    def MPCA(self, x, ratio=4):
        [n, h, w, c] = x.shape
        reduction_channels = max(8, c//ratio)
        output_channels = c
        residual = x

        x_h = MaxPooling2D(pool_size=(h, 1), strides=1)(x) + AveragePooling2D(pool_size=(h, 1), strides=1)(x)
        x_w = MaxPooling2D(pool_size=(1, w), strides=1)(x) + (x)AveragePooling2D(pool_size=(1, w), strides=1)(x)

        x_w = Permute((2, 1, 3))(x_w)
        y = K.concatenate((x_h, x_w), axis=2)

        # Reduction
        y = self.conv_block(y, filters=reduction_channels, kernel_size=1, strides=1, padding='valid', norm='batch', activation='hard_swish')
        
        # Split
        x_h, x_w = tf.split(y, axis=2, num_or_size_splits=[h, w])
        x_w = Permute((2, 1, 3))(x_w)

        # Expansion
        a_h = self.conv_block(x_h, filters=output_channels, kernel_size=1, strides=1, padding='valid', norm=None, activation='sigmoid')
        a_w = self.conv_block(x_w, filters=output_channels, kernel_size=1, strides=1, padding='valid', norm=None, activation='sigmoid')

        # Reweight
        a_h = tf.tile(a_h, [1, h, 1, 1])
        a_w = tf.tile(a_w, [1, 1, w, 1])
        out = multiply([residual, a_w, a_h])
    
        return out

    def attach_attention_module(self, x, module_name='ca', ratio=4):
        if(module_name=='ca'):
            return self.CA(x, ratio=ratio)
        elif(module_name=='mpca'):
            return self.MPCA(x, ratio=ratio)
        
        return x
    
    def cut_out_layer(self, input_tensor, cut_pos):
        return Lambda(lambda x: x[:, :, :, :cut_pos])(input_tensor)

    def cut_in_layer(self, input_tensor, cut_pos):
        return Lambda(lambda x: x[:, :, :, cut_pos:])(input_tensor)

    def sand_glass_bottleneck(self, x, filters, strides, 
                              reduction_ratio=4, 
                              multiplier=1.0, 
                              norm='batch', 
                              activation='mish',
                              attention_module='cam',
                              attention_ratio=4):
        assert strides in [1, 2]
    
        residual = x
        inp = x.shape[self.channel_axis]
        use_identity = False if multiplier == 1.0 else True

        identity_filters = int(round(inp * multiplier))
        reduction_filters = int(round(inp / reduction_ratio))

        if(not(strides == 1 and inp == filters)):
            residual = self.separable_conv_block(x, filters=filters, kernel_size=3, strides=strides, norm=norm, activation=None)

        # depthwise
        x = self.depthwise_conv_block(x, kernel_size=3, strides=1, norm=norm, activation=activation)
        
        # attention
        x = self.attach_attention_module(x, module_name=attention_module, ratio = attention_ratio)

        # pointwise reduction
        if reduction_ratio != 1:
            x = self.conv_block(x, filters=reduction_filters, kernel_size=1, strides=1, norm=norm, activation=None)

        # pointwise expansion
        x = self.conv_block(x, filters=filters, kernel_size=1, strides=1, norm=norm, activation=activation)

        # depthwise linear
        x = self.depthwise_conv_block(x, kernel_size=3, strides=strides, norm=norm, activation=None)

        if use_identity:
            identity_tensor = Add()([self.cut_out_layer(x, identity_filters), self.cut_out_layer(residual, identity_filters)])
            x = Concatenate()([identity_tensor, self.cut_in_layer(x, identity_filters)])
        else:
            x = add([x, residual])

        return x
    
    def sand_glass_block(self, x, t, c, n, s,
                         width_mult=1.0,
                         multiplier=1.0,
                         round_nearest=8,
                         attention_module='',
                         attention_ratio=4,
                         norm='batch',
                         activation='mish'
                        ):

        filters = self.make_divisible(c * width_mult, round_nearest)

        for i in range(n):
            strides = s if i == 0 else 1
            name = block_name + '_i' + str(i+1)

            x = self.sand_glass_bottleneck(x, filters=filters, strides=strides, 
                                           reduction_ratio=t,
                                           multiplier=multiplier, 
                                           norm=norm, activation=activation,
                                           attention_module=attention_module,
                                           attention_ratio=attention_ratio
                                         )

        return x
