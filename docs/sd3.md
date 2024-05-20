
GAN
VAE

---

ordinary differential equations - involve a single independent variables
partial differential equations - involve multiple independent variables

linear ODEs
nonlinear ODEs

SDE - stochastic differential equations

---

rectified flow

two distributions p1 and p0

---

score-based generative models

[Score-Based Generative Modeling through SDE](https://github.com/yang-song/score_sde_pytorch)

---

numerical instability - when a network is way to sensitive to small errors in input data

mode collapse - generator in GAN generates a very limited diversity examples

---

mode - value which occures the most frequently in the dataset

---

reflow operation

--

distill

--

distributions are the central object in statistics and machine learning
finding a transport map to transfer one distribution to another
learning an ordinary differential equation (ODE) a.k.a. flow
traveling in straight path as much as possible
neural ODE model
stochastic differential equations (SDE) model


---

solving numerically vs. solving analytically

solving numerically - using computational methods and algorithms to approximate a solution

solving analytically - finding an exact analytical solution

numerical solutions are used when analytical solution is not possible or too difficult to obtain

deriving a closed-form analytical expression

analytical methods when possible provide an exact symbolic solution

using mathematical techniques to derive an exact symbolic solution rather than approximating a solution computationally

---

p0 - standard gaussian distribution
p1 - unknown distributions (eg images)

generative modeling - a nonlinear transform that turns a point drawn from p0 to point that follows p1

---

p0 and p1 are both unknown distributions

transfer modeling


---

E[*] - expectation


---

clip - contrastive language image  pre-training
t5 xxl - google; text-to-text transfer transformer


sinusoidal encoding
positional embedding
mlp
linear
patching
modulation
unpatchingт
noised latent


MM-DiT - multimodal diffusion transformer

SiLU - sigmoid linear unit
layer norm
rms-norm - root mean square layer normalization

---

gan - single step
vae - single step
diffusion - multi-step; forward process ends up with noise from standard gausian
flow matching - learn how to morph p0 into p1 / q; it is a more general way; the goal is to learn time dependent vector field

---

gausian mixture model - defines a gausian at each data point and them merges them into a single distribution

---

ODE solver
