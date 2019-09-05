import os
import glob
import argparse
import time
from PIL import Image
import numpy as np
import PIL

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, display_images, displayImage
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='my_examples/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
start = time.time()
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

def load_images_with_resize(image_files):
    loaded_images = []
    width, height = 0,0
    t = ()
    for file in image_files:
        im = Image.open( file )
        width, height = im.size
        t0 = (width, height)
        t = t + t0
        im = im.resize((640, 480), PIL.Image.ANTIALIAS)
        x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0), t

# Input images
inputs,t =load_images_with_resize(glob.glob(args.input))
print('\nHeight: {0}.'.format(t))
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)


# Display results
viz = display_images(outputs.copy(), inputs.copy(),  t=t)

end = time.time()
print('It took: ', end - start)
