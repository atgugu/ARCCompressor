# ARCCompressor

A Python implementation of a compression‑based approach to solving ARC/ARC‑AGI tasks using neural networks.  
This project models the ARC problem as an information compression task and trains a neural compressor to infer missing outputs by minimizing reconstruction error and latent information content.

> The core idea: *Useful structure in a task corresponds to compressible information*.  
> By optimizing a neural model to compress a puzzle while reconstructing the given examples, the learned representation can be decoded to produce correct answers.  

---

## 🧠 Overview

ARCCompressor is inspired by recent work showing that **lossless compression objectives can drive intelligent behavior** in abstract reasoning tasks. It trains a neural network at inference time — without external pre‑training or search — to *compress* the ARC task and uses that compressed representation to infer solutions. :contentReference[oaicite:0]{index=0}

The repository includes:

- A neural model architecture tailored for ARC structures  
- Training, evaluation, and visualization code  
- Support for preprocessing ARC task data  
- Metrics and logging for performance analysis  

---

## 🚀 Features

✔ Compression‑based task solving (inference‑time training)  
✔ Equivariant neural network architecture  
✔ Task preprocessing & model visualization  
✔ CLI scripts for training and solving tasks  
✔ Works on ARC and related ARC‑AGI datasets

