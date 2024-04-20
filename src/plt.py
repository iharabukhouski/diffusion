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
  plt.title(f'T: {T}')

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

    plt.subplot(
      1, # Number of rows
      num_images_to_display, # Number of columns
      i + 1,
    )

    plt.title(f't: {t}')

    image_PIL = x_to_PIL(images[t, 0])

    plt.imshow(
      image_PIL,
    )

  plt.show()
