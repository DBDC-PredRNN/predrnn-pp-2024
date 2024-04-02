__author__ = 'yunbo'
__editor__ = 'songhune'

import tensorflow as tf
import tensorflow.compat.v1 as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm
tf.disable_v2_behavior()
def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True):
    
    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in range(seq_length-1):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-10]*images[:,t] + (1-mask_true[:,t-10])*x_gen

            # Forward direction cell
            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
        
        with tf.variable_scop('bpredrnn_pp'):
            # Backward direction cell
            hidden_back, cell_back, mem_back = lstm[0](inputs[::-1], hidden[0][::-1], cell[0][::-1], mem)
            z_t_back = gradient_highway(hidden_back, z_t)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)
                hidden_back, cell_back, mem_back = lstm[i](hidden[i-1][::-1], hidden_back, cell_back, mem_back)

            
            
            
            x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")
            x_gen_back = tf.layers.conv2d(inputs=hidden_back[num_layers-1],
                                          filters=output_channels,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          name="back_to_pixel_back")

            gen_images.append((x_gen+x_gen_back)/2)

    gen_images = tf.stack(gen_images)
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images - images[:,1:])
    return [gen_images, loss]