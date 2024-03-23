import math
import matplotlib.pyplot as plt
from data import x_to_PIL

def plt_gausian():

  pass

def plt_images(
  images, # (T, BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
):

  T = images.shape[0]

  plt.figure(
    figsize = (
      15,
      2,
    ),
  )
  plt.axis('off')

  num_images_to_display = 10
  stepsize = int(math.ceil(T / num_images_to_display))

  for t in range (0, T):

    plt.subplot(
      1, # Number of rows
      num_images_to_display, # Number of columns
      int(t / stepsize + 1) # Index
    )

    image_PIL = x_to_PIL(images[t, 0])

    plt.imshow(
      image_PIL,
    )

  plt.show()
