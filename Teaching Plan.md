
---

## Module I — Modern Vision Architectures

### Weeks 1–4 (8 Sessions)

### Session 1

**Introduction and Course Overview**

- Evolution of deep learning
    
- Why CNNs replaced MLPs for images
    
- Modern DL landscape
    
- CNNs, Transformers, Generative AI
    

Reading:

- Prince Ch.10.1
    

---

### Session 2

**Translation Equivariance and Convolution**

- Invariance vs equivariance
    
- 1D convolution
    
- 2D convolution
    
- Feature maps
    
- Parameter sharing
    

Mathematical emphasis:

- Proof of convolution equivariance
    

Reading:

- Prince Ch.10.1–10.3
    

---

### Session 3

**CNN Building Blocks**

- Padding
    
- Stride
    
- Pooling
    
- Receptive fields
    
- Dilated convolutions
    

Lab:

- CNN from scratch in PyTorch
    

Reading:

- Prince Ch.10.3–10.4
    

---

### Session 4

**Classical CNN Architectures**

- LeNet
    
- AlexNet
    
- VGG
    
- Inception
    

Supplementary lecture (outside Prince)

---

### Session 5

**Deep CNN Training Challenges**

- Vanishing gradients
    
- Degradation problem
    
- Batch normalization
    

Reading:

- Prince Ch.11.1–11.4
    

---

### Session 6

**Residual Networks**

- Residual blocks
    
- Skip connections
    
- ResNet architectures
    

Reading:

- Prince Ch.11
    

---

### Session 7

**Modern CNNs**

- ResNet
    
- DenseNet
    
- EfficientNet
    
- ConvNeXt
    

Supplementary lecture

---

### Session 8

**Vision Tasks**

- Classification
    
- Detection
    
- Segmentation
    

Project Discussion #1

---

# Module II — Attention and Transformers

### Weeks 5–8 (8 Sessions)

---

### Session 9

**Sequence Modeling Before Transformers**

- RNN
    
- LSTM
    
- GRU
    

Motivation only

Reading:

- Instructor notes
    

---

### Session 10

**Attention Mechanisms**

- Encoder-decoder attention
    
- Self-attention
    

Reading:

- Prince Ch.12.2
    

---

### Session 11

**Self-Attention Mathematics**

- Query
    
- Key
    
- Value
    

Attention(Q,K,V)=softmax(QKT/d)VAttention(Q,K,V) = softmax(QK^T/\sqrt{d})V

Reading:

- Prince Ch.12.2–12.3
    

---

### Session 12

**Multi-Head Attention and Positional Encoding**

Reading:

- Prince Ch.12.3–12.4
    

Lab:

- Implement self-attention
    

---

### Session 13

**Transformer Architecture**

- Encoder
    
- Decoder
    
- Residual connections
    
- Layer normalization
    

Reading:

- Prince Ch.12.4
    

---

### Session 14

**BERT and Representation Learning**

- Masked language modeling
    
- Encoder models
    

Reading:

- Prince Ch.12.6
    

---

### Session 15

**GPT and Generative Pretraining**

- Autoregressive models
    
- Scaling laws
    
- Emergence
    

Reading:

- Prince Ch.12.7
    

---

### Session 16

**Vision Transformers and Modern LLMs**

- ViT
    
- Llama
    
- Instruction tuning
    
- LoRA
    
- RAG (overview)
    

Project Discussion #2

Reading:

- Prince Ch.12.10 + supplementary papers
    

---

# Module III — Graph Learning

### Week 9 (2 Sessions)

---

### Session 17

**Introduction to Graphs and Graph Learning**

Reading:

- Prince Ch.13.1–13.4
    

---

### Session 18

**Graph Neural Networks**

- GCN
    
- Node classification
    
- Graph classification
    

Reading:

- Prince Ch.13.4–13.9
    

---

# Module IV — Representation Learning & Generative Models

### Weeks 10–13 (8 Sessions)

---

### Session 19

**Representation Learning**

- Why unsupervised learning?
    
- Latent representations
    
- Generative models taxonomy
    

Reading:

- Prince Ch.14
    

---

### Session 20

**Autoencoders**

- Undercomplete AE
    
- Denoising AE
    
- Sparse AE
    

Supplementary lecture

Important because many students later work on VAEs.

---

### Session 21

**Variational Autoencoders I**

- Latent variable models
    
- ELBO
    

Reading:

- Prince Ch.17.1–17.5
    

---

### Session 22

**Variational Autoencoders II**

- Reparameterization trick
    
- Training VAEs
    

Reading:

- Prince Ch.17.6–17.8
    

Lab:

- Implement VAE
    

---

### Session 23

**Generative Adversarial Networks**

- Generator
    
- Discriminator
    
- Minimax optimization
    

Reading:

- Prince Ch.15
    

---

### Session 24

**Advanced GANs**

- Conditional GANs
    
- StyleGAN
    
- Image translation
    

Reading:

- Prince Ch.15
    

---

### Session 25

**Normalizing Flows**

- Change of variables
    
- Invertibility
    
- Flow architectures
    

Reading:

- Prince Ch.16
    

---

### Session 26

**Diffusion Models**

- Forward process
    
- Reverse process
    
- Denoising
    

Reading:

- Prince Ch.18.1–18.4
    

---

# Module V — Research Topics and Theory

### Weeks 14–15 (4 Sessions)

---

### Session 27

**Diffusion Models II**

- Training
    
- Stable Diffusion
    
- Latent Diffusion
    

Reading:

- Prince Ch.18.5–18.7
    

---

### Session 28

**Why Deep Learning Works**

- Overparameterization
    
- Double descent
    
- Generalization
    

Reading:

- Prince Ch.20
    

---

### Session 29

**Current Research Frontiers**

Survey lecture:

- Multimodal Models
    
- Vision-Language Models
    
- Agentic AI
    
- Diffusion Transformers
    
- World Models
    

Recent papers discussion

---

### Session 30

**Student Research Presentations**

Each student presents:

- One recent paper (2024–2026)
    
- Mathematical formulation
    
- Architecture
    
- Experimental results
    
- Critical discussion
    

---

|Week|Session|Topic|Main Contents|Reference|
|---|---|---|---|---|
|1|1|Course Introduction & Deep Learning Landscape|Course overview, evolution of deep learning, modern architectures, AI trends|Intro + Ch.10|
|1|2|Convolution Fundamentals|Translation equivariance, parameter sharing, convolution operation, feature maps|Ch.10|
|2|3|CNN Building Blocks|Padding, stride, pooling, receptive fields, dilated convolutions|Ch.10|
|2|4|Classical CNN Architectures|LeNet, AlexNet, VGG, Inception|Supplementary|
|3|5|Training Deep CNNs|Vanishing gradients, degradation problem, batch normalization|Ch.11|
|3|6|Residual Networks|Residual learning, skip connections, ResNet architectures|Ch.11|
|4|7|Modern CNN Architectures|DenseNet, EfficientNet, ConvNeXt|Supplementary|
|4|8|Computer Vision Applications|Classification, detection, segmentation, transfer learning|Ch.10 + Supplementary|
|5|9|Sequence Modeling Background|RNNs, LSTMs, GRUs, limitations of recurrent models|Supplementary|
|5|10|Introduction to Attention|Encoder-decoder attention, self-attention motivation|Ch.12|
|6|11|Self-Attention Mathematics|Query-Key-Value framework, attention computation|Ch.12|
|6|12|Multi-Head Attention|Multi-head attention, positional encoding, implementation|Ch.12|
|7|13|Transformer Architecture|Encoder, decoder, residual connections, layer normalization|Ch.12|
|7|14|BERT and Representation Learning|Masked language modeling, encoder-based transformers|Ch.12|
|8|15|GPT and Autoregressive Models|Language modeling, autoregressive generation, scaling laws|Ch.12|
|8|16|Vision Transformers & Modern LLMs|ViT, instruction tuning, LoRA, PEFT, RAG overview|Ch.12 + Supplementary|
|9|17|Graph Representation Learning|Graph data, graph structures, graph tasks|Ch.13|
|9|18|Graph Neural Networks|GCNs, message passing, node and graph classification|Ch.13|
|10|19|Representation Learning|Unsupervised learning, latent representations, generative modeling overview|Ch.14|
|10|20|Autoencoders|Basic AE, sparse AE, denoising AE, latent spaces|Supplementary|
|11|21|Variational Autoencoders I|Probabilistic latent-variable models, ELBO|Ch.17|
|11|22|Variational Autoencoders II|Reparameterization trick, VAE training and applications|Ch.17|
|12|23|Generative Adversarial Networks|GAN framework, generator and discriminator|Ch.15|
|12|24|Advanced GAN Models|Conditional GANs, StyleGAN, image translation|Ch.15|
|13|25|Normalizing Flows|Change-of-variable theorem, invertible networks|Ch.16|
|13|26|Diffusion Models I|Forward diffusion, reverse denoising, score matching|Ch.18|
|14|27|Diffusion Models II|Stable Diffusion, latent diffusion, image generation systems|Ch.18|
|14|28|Why Deep Learning Works|Generalization, overparameterization, double descent|Ch.20|
|15|29|Current Research Frontiers|Multimodal AI, vision-language models, agentic AI, diffusion transformers|Recent Papers|
|15|30|Student Research Presentations|Paper presentation and discussion|Recent Papers|

###  Programming Assignments

|Assignment|Topic|
|---|---|
|Assignment 1|CNN implementation and comparison with MLP on CIFAR-10|
|Assignment 2|Implement self-attention and a mini Transformer|
|Assignment 3|Implement a VAE and compare latent representations|
|Assignment 4 (Optional)|Diffusion model or fine-tuning a pretrained Transformer|

###  Evaluation

|Component|Weight|
|---|---|
|Homework Assignments|15%|
|Programming Assignments|25%|
|Midterm Exam|20%|
|Paper Presentation|10%|
|Final Project|30%|

