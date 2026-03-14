# An Image is Worth 16×16 Words

Minimal implementation of the **Vision Transformer (ViT)** from the paper:

> **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**  
> Alexey Dosovitskiy et al. (Google Research, 2020)  
> [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

Pure PyTorch re-implementation of the original ViT architecture.

## Architecture Overview

The model treats an image as a sequence of fixed-size patches (default 16×16), embeds them linearly, adds position embeddings + a learnable [CLS] token, and feeds the sequence into a standard Transformer encoder. The final [CLS] representation is used for classification.

**model architecture:

![Vision Transformer (ViT) architecture diagram](https://github.com/google-research/vision_transformer/raw/main/vit_figure.png)

*(Image source: official Google Research ViT repo — shows patch splitting, embedding, position encoding, Transformer encoder stack, and MLP head.)*

## Features

- Pure Transformer encoder (no convolutions)
- Configurable variants: ViT-Tiny, ViT-Small, ViT-Base, ViT-Large
- Patch size 16×16 (as in the paper title)
- [CLS] token for classification
- Supports ImageNet-style training / fine-tuning

## Requirements

```bash
torch >= 2.0
torchvision
einops        

