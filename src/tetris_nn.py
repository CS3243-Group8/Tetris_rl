import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from src.Tetris_Env import TetrisEnv
from src.tetris_generator import TetrisGenerator

Window_Length = 1
env = TetrisEnv()
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n


model = Sequential()
input_shape = (Window_Length,) + (224,)

model.add(Dense(128, input_shape = input_shape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())

model.compile(optimizer=Adam(lr=0.00025), metrics=['mae'], loss='categorical_crossentropy')
model.fit_generator(generator=TetrisGenerator(), steps_per_epoch = 1000000, epochs = 1, verbose = 1)
model.save_weights(filepath="weight.h5f", overwrite= True)

