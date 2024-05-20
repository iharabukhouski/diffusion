#! /usr/bin/env python3

# import open_clip

# print(open_clip.list_pretrained())

# import clip

# from clip import available_models

# print(clip.tokenize('11'))


import torch
from PIL import Image

import clip

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model, preprocess = clip.load(
  "ViT-L/14",
  device=device,
)

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():

  image_features = model.encode_image(image)
  text_features = model.encode_text(text)

  logits_per_image, logits_per_text = model(image, text)
  probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("CLIP Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
   'ViT-bigG-14',
   pretrained='laion2b_s39b_b160k',
   device = device,
)
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Open CLIP Label probs:", text_probs)  # prints: [[1., 0., 0.]]
