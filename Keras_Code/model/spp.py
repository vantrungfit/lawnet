import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: 
            Either a collection of integers or a collection of collections of 2 integers. 
            Each element in the inner collection must contain 2 integers, (pooled_rows, pooled_cols)
            For example, providing [1, 2, 4] or [[1, 1], [2, 2], [4, 4]] preforms pooling using three 
            different pooling layers, having outputs with dimensions 1x1, 2x2 and 4x4 respectively.
            These are flattened along height and width to give an output of shape 
            [batch_size, (1 + 4 + 16) * channels] = [batch_size, 21*channels] if keepdims=False else
            [batch_size, 21*channels, 1, 1] for 'channels_first' or
            [batch_size, 1, 1, 21*channels] for 'channels_last'
        pool_type: A string to specify type of pooling.
            'max': max pooling (default)
            'avg': average pooling
            'mix': max + average pooling
        keepdims: A boolean, whether to keep the dimensions or not. 
            If keepdims is False, the rank of the tensor is reduced by 1. (default)
            If keepdims is True, the reduced dimension is retained with length 1.
    # Input shape
        4D tensor with shape:
        '(batch_size, channels, rows, cols)` if image_data_format = 'channels_first'
        or 4D tensor with shape:
        `(batch_size, rows, cols, channels)` if image_data_format = 'channels_last'
    # Output shape
        2D or 4D tensor with shape:
        `(batch_size, channels * sum([i * i for i in bins]))` if keepdims = False
        `(batch_size, channels * sum([i * i for i in bins]), 1, 1)` if image_data_format = 'channels_first' and keepdims = True
        `(batch_size, 1, 1, channels * sum([i * i for i in bins]))` if image_data_format = 'channels_last' and keepdims = True
    """

    def __init__(self, pool_list, pool_type, keepdims=False, **kwargs):
        self.pool_list = pool_list
        self.pool_type = pool_type
        self.keepdims = keepdims
        super(SpatialPyramidPooling, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        #num_output_per_channel = sum([i * i for i in self.pool_list]) 
        num_output_per_channel = 0
        
        for pool_size in self.pool_list:
            if type(pool_size) != int:
                pooled_rows, pooled_cols = pool_size
            else:
                pooled_rows, pooled_cols = pool_size, pool_size

            num_output_per_channel += pooled_rows*pooled_cols
        
        if(K.image_data_format()=='channels_first'):
            output_channels = num_output_per_channel * input_shape[1]

            if(self.keepdims):
                output_shape = (input_shape[0], output_channels, 1, 1)
            else:
                output_shape = (input_shape[0], output_channels)
        else:
            output_channels = num_output_per_channel * input_shape[-1]

            if(self.keepdims):
                output_shape = (input_shape[0], 1, 1, output_channels)
            else:
                output_shape = (input_shape[0], output_channels)
                
        return output_shape

    def get_config(self):
        config = {'pool_list': self.pool_list,
                  'pool_type': self.pool_type,
                  'keepdims': self.keepdims
                 }
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):

        image_data_format = K.image_data_format()
        input_shape = K.int_shape(inputs)
        outputs = []
        epsilon = K.epsilon()
        
        def pool(x, pool_type='max', axis=-1, keepdims=False):
            if(pool_type=='avg'):
                return K.mean(x, axis=axis, keepdims=keepdims)
            elif(pool_type=='mix'):
                return K.max(x, axis=axis, keepdims=keepdims) + K.mean(x, axis=axis, keepdims=keepdims)
            else:
                return K.max(x, axis=axis, keepdims=keepdims)

        if image_data_format == 'channels_first':
            axis = (2, 3)
            channels = input_shape[1]
            
            for pool_size in self.pool_list:
                if type(pool_size) != int:
                    pooled_rows, pooled_cols = pool_size
                else:
                    pooled_rows, pooled_cols = pool_size, pool_size
                    
                win_rows = input_shape[axis[0]]//pooled_rows
                win_cols = input_shape[axis[1]]//pooled_cols
                
                for jy in range(pooled_cols):
                    
                    for ix in range(pooled_rows):
                        
                        x = int(ix*win_rows)
                        y = int(jy*win_cols)
                        
                        pooled_vals = pool(inputs[:,:,x:x+win_rows,y:y+win_cols], 
                                           pool_type=self.pool_type,
                                           axis=axis, keepdims=self.keepdims)
                        outputs.append(pooled_vals)
            outputs = K.concatenate(outputs, axis=1)
        elif image_data_format == 'channels_last':
            channels = input_shape[-1]
            axis=[1, 2]
            for pool_size in self.pool_list:

                if type(pool_size) != int:
                    pooled_rows, pooled_cols = pool_size
                else:
                    pooled_rows, pooled_cols = pool_size, pool_size
                    
                win_rows = input_shape[axis[0]]//pooled_rows
                win_cols = input_shape[axis[1]]//pooled_cols
                
                for jy in range(pooled_cols):
                    
                    for ix in range(pooled_rows):
                        
                        x = int(ix*win_rows)
                        y = int(jy*win_cols)
                        
                        pooled_vals = pool(inputs[:,x:x+win_rows,y:y+win_cols,:], 
                                           pool_type=self.pool_type,
                                           axis=axis, keepdims=self.keepdims)
                        outputs.append(pooled_vals)
            outputs = K.concatenate(outputs, axis=-1)

        return outputs