from tealayer2 import Tea
import tensorflow as tf
import numpy as np

layer = Tea(units=4, name='test')
out = layer(np.random.rand(2, 8).astype('float32'))
print('OK:', out.shape)