import setuptools

setuptools.setup(
  name = 'diffusion',
  version = 0.1,
  install_requires = [
    'numpy', # Needed because without it 'torch' gives warnings :(
    'torch',

    # Below dependencies are needed for the model
    'matplotlib',
    'torchvision',

    'wandb',

    'lovely_tensors',

    # Needed for Standford Cars dataset
    'scipy',

    'onnx',
    'onnxscript',

    'python-dotenv',

    'safetensors',

    'nvitop',
  ],
)
