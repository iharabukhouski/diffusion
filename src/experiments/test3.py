
# - Generative Adversarial Networks (GANs)
# - Denoising Diffusion Models
# - Variational Autoencoders (VAEs), Normalizing Flows

# - Latent Distribution
# - Latent Space
# - Vanishing Gradients
# - Mode Collapse
# - Normalization Layers
# - Residual Connections
# - Markov Chain
# - Stochastic Events

# Imagen - is text-to-image model by Google Brain (2022)
# DALL-E - is text-to-image model by OpenAI

# The task of the model is to predict the noise that was added to each of the images
# Old Image + Noise (???) -> New Image
# What is `???`

# - Forward Process - progressively add noise to image
#   - Noise scheduler
# - Backward Process - pregressively restore image from noise
#   - U-net

# Typical number of steps is 1K

# What we need:
# - Noise Scheduler
# - Model that predicts noise in the image (UNet)
# - Way to encode the current timestep (Timestep Encoding)

# Vocabulary

# - x_0 - initial image
# - x_t - image at timestep `t`
# - x_T - final image
# - timestep == `t`


# `alphas` - ???
# `betas` - variance schedule; describes how much noise we want to add in each of the timesteps; describes noise level at each timestep; range between 0 and 1
# `posterior variance`
# `posterior` - P(A|B) - updated `prior`
# `isotropic` - looks the same in every direction
# Why do we need to multiply by identity matrix


# torch.Tensor([1]).dtype

# images = next(iter(dataloader))[0]

# plt.figure(figsize = (15,15))
# plt.axis('off')

# num_images_to_display = 10
# stepsize = int(T / num_images_to_display)

# timesteps = torch.arange(
#   start = 0,
#   end = T,
#   step = stepsize,
#   dtype = torch.int64,
# )

# for (
#   timestep_index,
#   timesteps, # a.k.a. `t`
# ) in enumerate(timesteps):

#   for batch_index in range(0, BATCH_SIZE):

#     images_with_noise, noises = images_to_images_with_noise_at_timesteps(images, timesteps)

#     # `image_with_noise` a.k.a. `x_t`
#     images_with_noise_at_timesteps = images_with_noise[batch_index, :, :, :]

#     image_with_noise_PIL = x_to_PIL(images_with_noise_at_timesteps)

#     plt.subplot(
#       BATCH_SIZE, # number of rows
#       num_images_to_display, # number of columns
#       batch_index * len(timesteps) + (timestep_index + 1) # index
#     )

#     plt.imshow(
#       image_with_noise_PIL,
    # )

# Backward Process (UNet)
# UNet has similar structure to `AutoEncoder`
# - Convolution
# - Residual Connections
# - Batch / Group Normalization
# - Attention Modules

# Backward Process
# Neural Network (U-Net)

# Group Normalization
# Attention Module

# Down sampling
# Up sampling
# Residual connections

# - Convolution
# - Relu
# - Batch Normalization
# - Positional Embeddings
# - ReLU
# - Convolution
# - ReLU
# - Batch Normalization


# TODO
# - Refactor code
# - Understand math
# - Connect performance analyzer
# - Follow shapes
# - build graphs that help to show intuition of internals
# - read papers
# - train in cloud with credits
# - increase batch size

# print(model.state_dict())

# Saving model weights
# torch.save(model.state_dict(), './weights')

# Loading model weights
# model.load_state_dict(torch.load('./weights'))
# model.eval()

# Saving model architecture and weights
# torch.save(model, './model')

# Loading model architecture and weights
# model = torch.load('./model')

# # CHECKPOINTS
# SEE: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
# torch.save(
#   {
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss
#   },
#   './checkpoint',
# )

# # load
# checkpoint = torch.load('./checkpoint')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# model.eval() # sets model in eval mode
# # -- or --
# model.train() # sets model in train mode

# PROFILING

# with profile(
#   activities = [
#     ProfilerActivity.CPU,
#     ProfilerActivity.
#   ],
#   record_shapes = True
# ) as prof:
#   with record_function("model_inference"):
#     sample_image()

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
