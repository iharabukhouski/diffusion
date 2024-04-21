import math
import io
import PIL
from PIL import Image

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data import x_to_PIL

def generate_plt(
  images, # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
):

  T = images.shape[0]

  fig, fig_ax = plt.subplots(
    figsize = (
      15,
      2,
    ),
  )

  fig_ax.axis('off') 
  fig_ax.set_title(f'T: {T}')

  num_images_to_display = 10
  stepsize = int(math.ceil(T / num_images_to_display))

  for i in range(0, num_images_to_display):

    # making sure that the first image is from t = 0
    if i == 0:

      t = 0

    # making sure that the last image is from t = T
    elif i == num_images_to_display - 1:

      t = T - 1

    else:

      t = i * stepsize

    subplot_ax = fig.add_subplot(
      1, # Number of rows
      num_images_to_display, # Number of columns
      i + 1,
    )

    subplot_ax.axis('off') 
    subplot_ax.set_title(f't: {t}')

    image_PIL = x_to_PIL(images[t, 0])

    subplot_ax.imshow(
      image_PIL,
    )

  return fig

def plt_gausian():

  pass

def plt_images(
  images, # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
):

  generate_plt(images)

  plt.show()

def save_images(
  images, # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
):

  generate_plt(images)

  plt.savefig('./sample.png')

import wandb

def as_PIL(
  images, # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
):

  fig = generate_plt(images)

  image_buffer = io.BytesIO()
  
  plt.savefig(
    image_buffer,
    format='png',
  )

  image = Image.open(image_buffer)

  wandb_image = wandb.Image(image)

  image_buffer.close()

  return wandb_image
