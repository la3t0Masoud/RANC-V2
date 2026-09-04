from tealayer2 import Tea, AdditivePooling
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras import Model

inputs = Input(shape=(28, 28, 1))
flattened = Flatten()(inputs)
core0 = Tea(units=64, name='tea_1_1')(flattened)
model = Model(inputs=inputs, outputs=core0)

for layer in model.layers:
    for weight in layer.weights:
        print(f"layer.name={layer.name!r:20s}  weight.name={weight.name!r:30s}")