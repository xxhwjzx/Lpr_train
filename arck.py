# -- encoding:utf-8 --
import main2
from keras.utils import plot_model
import os

os.environ["PATH"] += os.pathsep + 'G:/Graphviz2.38/bin/'

model = main2.build_model(40, 160, 1, 10)
plot_model(model, to_file='googlenet1.png', show_shapes=True)
