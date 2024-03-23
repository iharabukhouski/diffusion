- [Papers](#papers)
- [Architecture](#architecture)
- [Vocabulary](#vocabulary)
- [Links](#links)

# Papers

[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) - original paper
[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf) - improved version by OpenAI

# Architecture

- Forward Process
- Reverse Process
  - UNet
    - Time Embedding
      - SinusoidalPositionalEmbedding
      - Linear
      - ReLU
    - Initial Projection / Conv2d(3, 64)
    - Down
      - Block
        - Conv
        - ReLU
        - BatchNorm

        - Time Embedding
        - ReLU

        - Conv
        - ReLU
        - BatchNorm
    - Up
      - Block
        - Conv
        - ReLU
        - BatchNorm

        - Time Embedding
        - ReLU

        - Conv
        - ReLU
        - BatchNorm
    - Final Projection

# Vocabulary

# Links

https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=717s
https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1338s
