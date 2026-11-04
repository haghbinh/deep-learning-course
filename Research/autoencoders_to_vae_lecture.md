# From Autoencoders to Variational Autoencoders
## Workshop Lecture Notes

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Classical Autoencoders](#2-classical-autoencoders)
3. [The Limitation of Classical Autoencoders](#3-the-limitation-of-classical-autoencoders)
4. [Probabilistic Foundations](#4-probabilistic-foundations)
5. [Variational Autoencoders (VAE)](#5-variational-autoencoders-vae)
6. [The ELBO Derivation](#6-the-elbo-derivation)
7. [The Reparameterization Trick](#7-the-reparameterization-trick)
8. [VAE Loss Function in Practice](#8-vae-loss-function-in-practice)
9. [PyTorch Implementation on MNIST](#9-pytorch-implementation-on-mnist)
10. [Experiments & Visualizations](#10-experiments--visualizations)
11. [Summary & Further Reading](#11-summary--further-reading)

---

## 1. Introduction & Motivation

### What Problem Are We Solving?

Suppose we have a dataset $\mathcal{D} = \{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\}$ of high-dimensional observations (e.g., 28×28 MNIST images, $x \in \mathbb{R}^{784}$). We want to:

1. **Compress** each $x$ into a compact, meaningful representation $z \in \mathbb{R}^d$ where $d \ll 784$.
2. **Generate** new, realistic samples that look like they came from the same distribution as $\mathcal{D}$.
3. **Understand** the underlying structure (latent space) of the data.

Classical autoencoders achieve (1), but fail at (2) and (3) in a principled way. Variational Autoencoders (VAEs) fix this by placing a **probabilistic framework** over the latent space.

---

## 2. Classical Autoencoders

### 2.1 Architecture

A classical autoencoder consists of two neural networks:

- **Encoder**: $f_\phi : \mathcal{X} \rightarrow \mathcal{Z}$, mapping input $x$ to latent code $z = f_\phi(x)$
- **Decoder**: $g_\theta : \mathcal{Z} \rightarrow \mathcal{X}$, mapping latent code $z$ back to reconstruction $\hat{x} = g_\theta(z)$

```
Input x  →  [Encoder f_φ]  →  Latent z  →  [Decoder g_θ]  →  Reconstruction x̂
  (784)          ↓               (d)               ↓               (784)
             Bottleneck       (e.g., d=2)       Expansion
```

### 2.2 Objective Function

The autoencoder is trained by minimizing the **reconstruction loss**:

$$\mathcal{L}_{\text{AE}}(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N} \| x^{(i)} - g_\theta(f_\phi(x^{(i)})) \|^2$$

For binary inputs (like binarized MNIST), we use **binary cross-entropy**:

$$\mathcal{L}_{\text{AE}}(\phi, \theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{D} \left[ x_j^{(i)} \log \hat{x}_j^{(i)} + (1 - x_j^{(i)}) \log(1 - \hat{x}_j^{(i)}) \right]$$

where $D = 784$ is the input dimension and $\hat{x} = g_\theta(f_\phi(x))$.

### 2.3 Training via Backpropagation

Parameters $\phi$ and $\theta$ are updated jointly:

$$\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}_{\text{AE}}, \qquad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{AE}}$$

Since the encoder output $z$ is deterministic and differentiable, gradients flow through the bottleneck seamlessly.

---

## 3. The Limitation of Classical Autoencoders

### 3.1 The Discontinuous Latent Space Problem

The encoder maps each input to a **single point** in latent space:

$$z = f_\phi(x) \in \mathbb{R}^d \quad \text{(a fixed vector, no uncertainty)}$$

The latent space has **no regularization**. Points between learned encodings may decode to garbage:

```
z₁ = f_φ(digit_3)  →  Region A
z₂ = f_φ(digit_7)  →  Region B

Interpolation: (z₁ + z₂)/2  →  ???  (likely meaningless)
```

### 3.2 The Generation Problem

To generate new samples, we would need to sample $z$ from some distribution. But we have **no principled way** to define what distribution to sample from — the latent space can be arbitrarily shaped.

### 3.3 Root Cause

The autoencoder has no incentive to organize the latent space in any structured way. It simply learns a lookup-table-like compression.

---

## 4. Probabilistic Foundations

### 4.1 The Generative Model

We posit a **latent variable model** with:

- **Prior**: $p(z)$ — our belief about latent codes before seeing data
- **Likelihood**: $p_\theta(x \mid z)$ — the decoder (a conditional distribution)
- **Joint**: $p_\theta(x, z) = p_\theta(x \mid z) \cdot p(z)$

We choose a simple Gaussian prior:

$$p(z) = \mathcal{N}(z; \mathbf{0}, \mathbf{I})$$

The marginal likelihood (evidence) of a data point is:

$$p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz$$

### 4.2 The Inference Problem

Given an observation $x$, we want the **posterior**:

$$p_\theta(z \mid x) = \frac{p_\theta(x \mid z) \, p(z)}{p_\theta(x)}$$

**Problem**: Computing $p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$ is **intractable** for neural network decoders. The integral has no closed form.

### 4.3 Variational Inference

We approximate the true posterior $p_\theta(z \mid x)$ with a tractable distribution $q_\phi(z \mid x)$ called the **variational posterior** or **approximate posterior**. We want $q_\phi$ to be as close as possible to $p_\theta(z \mid x)$, measured by **KL divergence**:

$$\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

We choose:

$$q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$$

where $\mu_\phi(x)$ and $\sigma^2_\phi(x)$ are outputs of the encoder neural network.

---

## 5. Variational Autoencoders (VAE)

### 5.1 The VAE Architecture

The VAE encoder does **not** output a single point but a **distribution**:

```
Input x  →  [Encoder q_φ(z|x)]  →  μ(x), σ²(x)
                                        ↓
                               z ~ N(μ(x), diag(σ²(x)))   ← Sampling step
                                        ↓
                          [Decoder p_θ(x|z)]  →  Reconstruction x̂
```

Concretely:

- **Encoder** outputs two vectors: $\mu_\phi(x) \in \mathbb{R}^d$ and $\log \sigma^2_\phi(x) \in \mathbb{R}^d$
- **Latent sample**: $z \sim q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$
- **Decoder**: $p_\theta(x \mid z)$ maps $z$ back to the data space

### 5.2 The Gaussian Parameterization

We output $\log \sigma^2$ instead of $\sigma^2$ for numerical stability (it can take any real value):

$$\mu = \text{EncoderMu}(x) \in \mathbb{R}^d$$
$$\log \sigma^2 = \text{EncoderLogVar}(x) \in \mathbb{R}^d$$
$$\sigma = \exp\left(\frac{1}{2} \log \sigma^2\right) = \exp\left(\frac{\text{logvar}}{2}\right)$$

---

## 6. The ELBO Derivation

### 6.1 Starting from Log-Likelihood

We want to maximize $\log p_\theta(x)$. Begin with Jensen's inequality:

$$\log p_\theta(x) = \log \int p_\theta(x, z) \, dz = \log \int \frac{p_\theta(x, z)}{q_\phi(z \mid x)} q_\phi(z \mid x) \, dz$$

By Jensen's inequality (since $\log$ is concave):

$$\log p_\theta(x) \geq \int q_\phi(z \mid x) \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \, dz = \mathcal{L}(\phi, \theta; x)$$

This lower bound $\mathcal{L}$ is called the **Evidence Lower BOund (ELBO)**.

### 6.2 Decomposing the ELBO

$$\mathcal{L}(\phi, \theta; x) = \mathbb{E}_{q_\phi(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]$$

Since $p_\theta(x, z) = p_\theta(x \mid z) \cdot p(z)$:

$$\mathcal{L}(\phi, \theta; x) = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z) + \log p(z) - \log q_\phi(z \mid x)\right]$$

$$\boxed{\mathcal{L}(\phi, \theta; x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]}_{\text{Reconstruction Term}} - \underbrace{\text{KL}\left(q_\phi(z \mid x) \| p(z)\right)}_{\text{Regularization Term}}}$$

### 6.3 Relationship to the True Log-Likelihood

We can also show:

$$\log p_\theta(x) = \mathcal{L}(\phi, \theta; x) + \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

Since KL divergence is always $\geq 0$, we confirm $\log p_\theta(x) \geq \mathcal{L}$. We maximize $\mathcal{L}$ as a surrogate for the intractable $\log p_\theta(x)$.

### 6.4 Closed-Form KL for Gaussians

With $q_\phi(z \mid x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$, the KL has a **closed form**:

$$\text{KL}(q_\phi(z \mid x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**Derivation**:

$$\text{KL}(\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(0, I)) = \frac{1}{2}\left[\text{tr}(\Sigma) + \mu^\top \mu - d - \log \det(\Sigma)\right]$$

For a diagonal covariance $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$:

$$= \frac{1}{2}\left[\sum_j \sigma_j^2 + \sum_j \mu_j^2 - d - \sum_j \log \sigma_j^2\right] = -\frac{1}{2}\sum_j \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## 7. The Reparameterization Trick

### 7.1 The Problem with Sampling

We need $\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$, but **sampling is not differentiable**:

$$z \sim \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$$

The gradient $\nabla_\phi$ cannot flow through the stochastic node.

### 7.2 The Reparameterization

Rewrite the sampling as a **deterministic function** of a fixed-distribution noise variable $\epsilon$:

$$\epsilon \sim \mathcal{N}(0, I)$$
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$$

where $\odot$ denotes element-wise multiplication.

Now $z$ is a **deterministic, differentiable** function of $\phi$ (through $\mu_\phi$ and $\sigma_\phi$), and of $\epsilon$ (which does not depend on $\phi$):

```
Stochastic node (no gradient):      Reparameterized (gradient flows):
                                     
  μ, σ  →  z ~ N(μ, σ²)              ε ~ N(0, I)
              ↓                           ↓
           loss                      z = μ + σ⊙ε  →  loss
                                     ↑
                                  ∇φ flows through μ and σ
```

### 7.3 The ELBO Gradient

The Monte Carlo estimate of the gradient becomes:

$$\nabla_{\phi, \theta} \mathcal{L} \approx \frac{1}{L} \sum_{l=1}^{L} \nabla_{\phi, \theta} \left[\log p_\theta(x \mid z^{(l)}) \right] + \nabla_\phi \left[-\text{KL}(q_\phi \| p)\right]$$

where $z^{(l)} = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon^{(l)}$, $\epsilon^{(l)} \sim \mathcal{N}(0, I)$.

In practice, $L = 1$ sample suffices when the mini-batch size is large enough.

---

## 8. VAE Loss Function in Practice

### 8.1 Total Loss (Negative ELBO)

We minimize (negate the ELBO):

$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] + \text{KL}\left(q_\phi(z \mid x) \| p(z)\right)$$

### 8.2 Reconstruction Loss

For **binary data** (Bernoulli decoder):

$$\mathcal{L}_{\text{recon}} = -\sum_{j=1}^{D} \left[x_j \log \hat{x}_j + (1 - x_j) \log (1 - \hat{x}_j)\right]$$

For **continuous data** (Gaussian decoder):

$$\mathcal{L}_{\text{recon}} = \frac{1}{2} \sum_{j=1}^{D} (x_j - \hat{x}_j)^2$$

### 8.3 KL Divergence Loss

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 8.4 Final Loss

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}$$

Per data point. For a mini-batch of size $M$:

$$\mathcal{L}_{\text{batch}} = \frac{1}{M} \sum_{i=1}^{M} \left[\mathcal{L}_{\text{recon}}^{(i)} + \mathcal{L}_{\text{KL}}^{(i)}\right]$$

### 8.5 The $\beta$-VAE Variant

A useful extension is the $\beta$-VAE, which weights the KL term:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}, \quad \beta > 1$$

Higher $\beta$ encourages more **disentangled** representations at the cost of reconstruction quality.

---

## 9. PyTorch Implementation on MNIST

### 9.1 Setup and Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 9.2 Hyperparameters

```python
# ── Hyperparameters ───────────────────────────────────────────────
INPUT_DIM    = 784       # 28×28 flattened
HIDDEN_DIM   = 400       # hidden layer size
LATENT_DIM   = 20        # dimensionality of z
BATCH_SIZE   = 128
EPOCHS       = 30
LEARNING_RATE = 1e-3
BETA         = 1.0       # β-VAE weight on KL term (1.0 = standard VAE)
```

### 9.3 Data Loading

```python
# ── Data Loading ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    # Flatten 28×28 → 784 in the model forward pass
])

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
```

### 9.4 Classical Autoencoder

```python
# ── Classical Autoencoder ─────────────────────────────────────────
class Autoencoder(nn.Module):
    """
    Classical Autoencoder:
      Encoder: x (784) → hidden → z (LATENT_DIM)
      Decoder: z (LATENT_DIM) → hidden → x̂ (784)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder: f_φ
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: g_θ
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   # output in [0,1] for BCE loss
        )

    def forward(self, x):
        x = x.view(-1, INPUT_DIM)  # flatten
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def ae_loss(x_hat, x):
    """Binary Cross-Entropy reconstruction loss."""
    x = x.view(-1, INPUT_DIM)
    return F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
```

### 9.5 Variational Autoencoder

```python
# ── Variational Autoencoder ───────────────────────────────────────
class VAE(nn.Module):
    """
    Variational Autoencoder:
      Encoder q_φ(z|x): x → μ(x), log σ²(x)
      Reparameterize:   z = μ + σ ⊙ ε,  ε ~ N(0, I)
      Decoder p_θ(x|z): z → x̂
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # ── Encoder body ──────────────────────────────────────────
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Two heads: mean and log-variance
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)   # μ_φ(x)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)   # log σ²_φ(x)

        # ── Decoder ───────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """
        Returns μ and log σ² for the approximate posterior q_φ(z|x).
        """
        h      = self.encoder_fc(x)
        mu     = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick:
          z = μ + σ ⊙ ε,   ε ~ N(0, I)
          σ = exp(log_var / 2)

        During INFERENCE (eval mode), we use z = μ (no noise).
        """
        if self.training:
            # σ = exp(0.5 * log σ²)
            std = torch.exp(0.5 * log_var)
            # Sample ε ~ N(0, I), same shape as std
            eps = torch.randn_like(std)
            # z = μ + σ ⊙ ε
            return mu + std * eps
        else:
            return mu   # deterministic at test time

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(-1, INPUT_DIM)     # flatten: (B, 784)
        mu, log_var = self.encode(x)  # (B, d), (B, d)
        z           = self.reparameterize(mu, log_var)  # (B, d)
        x_hat       = self.decode(z)  # (B, 784)
        return x_hat, mu, log_var


def vae_loss(x_hat, x, mu, log_var, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence

    Reconstruction: Binary Cross-Entropy
      ℒ_recon = -Σ_j [ x_j log x̂_j + (1-x_j) log(1-x̂_j) ]

    KL Divergence (closed form for Gaussians):
      KL(q_φ(z|x) ‖ p(z)) = -½ Σ_j (1 + log σ²_j - μ²_j - σ²_j)

    Both are summed over dimensions and averaged over the batch.
    """
    x = x.view(-1, INPUT_DIM)

    # ── Reconstruction loss ───────────────────────────────────────
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # ── KL divergence ─────────────────────────────────────────────
    # = -½ Σ_j (1 + log_var_j - μ²_j - exp(log_var_j))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # ── Total loss (averaged over batch) ─────────────────────────
    batch_size = x.size(0)
    total_loss = (recon_loss + beta * kl_loss) / batch_size

    return total_loss, recon_loss / batch_size, kl_loss / batch_size
```

### 9.6 Training Loop

```python
# ── Training Utilities ────────────────────────────────────────────
def train_vae(model, optimizer, loader, epoch, beta=1.0):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        x_hat, mu, log_var = model(x)

        # Compute loss
        loss, recon, kl = vae_loss(x_hat, x, mu, log_var, beta=beta)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()

    n = len(loader)
    print(f"Epoch {epoch:3d} | Loss: {total_loss/n:.4f} | "
          f"Recon: {total_recon/n:.4f} | KL: {total_kl/n:.4f}")
    return total_loss / n, total_recon / n, total_kl / n


def evaluate_vae(model, loader, beta=1.0):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            loss, _, _ = vae_loss(x_hat, x, mu, log_var, beta=beta)
            total_loss += loss.item()

    return total_loss / len(loader)


# ── Instantiate and Train ─────────────────────────────────────────
vae = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

print("=" * 60)
print("Training Variational Autoencoder on MNIST")
print("=" * 60)

history = {'train_loss': [], 'recon_loss': [], 'kl_loss': [], 'test_loss': []}

for epoch in range(1, EPOCHS + 1):
    train_loss, recon_loss, kl_loss = train_vae(
        vae, optimizer, train_loader, epoch, beta=BETA
    )
    test_loss = evaluate_vae(vae, test_loader, beta=BETA)

    history['train_loss'].append(train_loss)
    history['recon_loss'].append(recon_loss)
    history['kl_loss'].append(kl_loss)
    history['test_loss'].append(test_loss)

print("\nTraining complete!")
```

### 9.7 Visualization: Reconstructions

```python
# ── Reconstruction Visualization ──────────────────────────────────
def show_reconstructions(model, loader, n=10, title="Reconstructions"):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    with torch.no_grad():
        x_hat, mu, log_var = model(x)

    x     = x.cpu().view(-1, 28, 28).numpy()
    x_hat = x_hat.cpu().view(-1, 28, 28).numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i], cmap='gray');     axes[0, i].axis('off')
        axes[1, i].imshow(x_hat[i], cmap='gray'); axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    plt.show()


show_reconstructions(vae, test_loader, title="VAE Reconstructions (MNIST)")
```

### 9.8 Visualization: Generated Samples

```python
# ── Sampling from the Prior ───────────────────────────────────────
def generate_samples(model, n=100, title="Generated Samples"):
    model.eval()
    with torch.no_grad():
        # Sample z ~ N(0, I)
        z = torch.randn(n, LATENT_DIM).to(device)
        # Decode
        samples = model.decode(z).cpu().view(-1, 28, 28).numpy()

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("generated_samples.png", dpi=150)
    plt.show()


generate_samples(vae, title="Samples from VAE Prior p(z) = N(0, I)")
```

### 9.9 Visualization: 2D Latent Space (LATENT_DIM = 2)

```python
# ── 2D Latent Space Visualization ────────────────────────────────
# NOTE: Set LATENT_DIM = 2 and retrain for this visualization

def plot_latent_space(model, loader, title="2D Latent Space"):
    model.eval()
    mus, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, _ = model.encode(x.view(-1, INPUT_DIM))
            mus.append(mu.cpu().numpy())
            labels.append(y.numpy())

    mus    = np.concatenate(mus,    axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mus[:, 0], mus[:, 1],
                          c=labels, cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel("z₁"); plt.ylabel("z₂")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("latent_space_2d.png", dpi=150)
    plt.show()


# plot_latent_space(vae, test_loader)   # uncomment if LATENT_DIM = 2


# ── 2D Manifold: Grid Traversal ───────────────────────────────────
def plot_2d_manifold(model, n=20, r=3.0, title="2D Latent Manifold"):
    """
    Decode a regular grid of z values in 2D latent space.
    n: grid resolution per axis
    r: range [-r, r] for each dimension
    """
    model.eval()
    grid = np.linspace(-r, r, n)
    canvas = np.zeros((n * 28, n * 28))

    with torch.no_grad():
        for i, z1 in enumerate(grid):
            for j, z2 in enumerate(grid):
                z = torch.tensor([[z1, z2]], dtype=torch.float32).to(device)
                img = model.decode(z).cpu().view(28, 28).numpy()
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("latent_manifold.png", dpi=150)
    plt.show()


# plot_2d_manifold(vae)   # uncomment if LATENT_DIM = 2
```

### 9.10 Visualization: Loss Curves

```python
# ── Training Curves ───────────────────────────────────────────────
def plot_loss_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['test_loss'],  label='Test',  color='orange', linestyle='--')
    axes[0].set_title("Total ELBO Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history['recon_loss'], color='green')
    axes[1].set_title("Reconstruction Loss (BCE)"); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history['kl_loss'], color='red')
    axes[2].set_title("KL Divergence"); axes[2].set_xlabel("Epoch")

    plt.suptitle("VAE Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()


plot_loss_curves(history)
```

### 9.11 Saving and Loading the Model

```python
# ── Save & Load ───────────────────────────────────────────────────
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
    'config': {
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'latent_dim': LATENT_DIM,
        'beta': BETA,
    }
}, 'vae_mnist.pth')

# Load:
checkpoint = torch.load('vae_mnist.pth', map_location=device)
vae_loaded = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
vae_loaded.load_state_dict(checkpoint['model_state_dict'])
vae_loaded.eval()
print("Model loaded successfully.")
```

---

## 10. Experiments & Visualizations

### 10.1 Expected Training Behavior

| Epoch | Total Loss | Recon Loss | KL Loss | Notes |
|------:|----------:|----------:|-------:|-------|
| 1     | ~180       | ~170       | ~10    | KL starts small (posterior ≈ prior) |
| 5     | ~130       | ~110       | ~20    | KL grows as encoder becomes informative |
| 15    | ~105       | ~82        | ~23    | Reconstruction improves |
| 30    | ~98        | ~74        | ~24    | Near convergence |

### 10.2 Effect of Latent Dimension $d$

| $d$ | Reconstruction Quality | Generation Quality | Interpretability |
|----:|:----------------------:|:-----------------:|:----------------:|
| 2   | Low                    | High              | Excellent (visualizable) |
| 10  | Medium                 | High              | Good |
| 20  | High                   | High              | Moderate |
| 100 | Very High              | Medium            | Low |

### 10.3 Effect of $\beta$ in $\beta$-VAE

| $\beta$ | Reconstruction | Disentanglement | Typical Use |
|--------:|:--------------:|:---------------:|:------------|
| 0.5     | Excellent      | Poor            | When reconstruction is priority |
| 1.0     | Good           | Moderate        | Standard VAE |
| 4.0     | Fair           | Good            | Disentanglement research |
| 10.0    | Poor           | Excellent       | Maximum disentanglement |

### 10.4 Interpolation Between Digits

```python
def interpolate(model, x1, x2, steps=10):
    """
    Linearly interpolate between two digits in latent space.
    """
    model.eval()
    with torch.no_grad():
        x1 = x1.to(device).view(1, -1)
        x2 = x2.to(device).view(1, -1)
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # Linear interpolation: z(t) = (1-t)*μ₁ + t*μ₂
        alphas = torch.linspace(0, 1, steps)
        images = []
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            img = model.decode(z_interp).cpu().view(28, 28).numpy()
            images.append(img)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.2, 1.5))
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray'); ax.axis('off')
    plt.suptitle("Latent Space Interpolation", fontsize=12)
    plt.tight_layout()
    plt.savefig("interpolation.png", dpi=150)
    plt.show()


# Example: interpolate between first two test images
test_iter = iter(test_loader)
x_batch, y_batch = next(test_iter)
interpolate(vae, x_batch[0], x_batch[1])
```

---

## 11. Summary & Further Reading

### 11.1 Key Takeaways

| Concept | Classical AE | VAE |
|:--------|:------------|:----|
| Encoder output | Point $z = f_\phi(x)$ | Distribution $q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2 I)$ |
| Latent space | Unstructured | Regularized (close to $\mathcal{N}(0, I)$) |
| Generation | Not principled | Sample $z \sim \mathcal{N}(0,I)$, decode |
| Objective | Reconstruction only | ELBO = Reconstruction − KL |
| Gradient | Straightforward | Requires reparameterization trick |
| Interpolation | Often breaks | Smooth (continuous latent space) |

### 11.2 The VAE at a Glance

$$\underbrace{q_\phi(z \mid x)}_{\text{Encoder}} \approx p_\theta(z \mid x), \qquad \underbrace{p_\theta(x \mid z)}_{\text{Decoder}} = \text{Neural Network}$$

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]}_{\text{BCE/MSE Loss}} + \underbrace{\text{KL}(q_\phi(z \mid x) \| p(z))}_{\frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2)}$$

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \quad \leftarrow \text{Reparameterization}$$

### 11.3 Common Pitfalls

1. **Posterior Collapse**: KL term goes to zero, encoder ignores input. Fix: KL annealing (warm-up from $\beta=0$ to $\beta=1$ over early epochs).
2. **Blurry Reconstructions**: BCE loss favors averages. Fix: Use perceptual loss or adversarial components (VAE-GAN).
3. **Prior Mismatch**: True aggregate posterior $\int q_\phi(z \mid x) p(x) dx \neq \mathcal{N}(0,I)$. Fix: VampPrior, Flows.

### 11.4 Extensions

| Model | Key Idea |
|:------|:---------|
| **$\beta$-VAE** | Stronger KL penalty → disentanglement |
| **IWAE** | Importance-weighted ELBO (tighter bound) |
| **VAE-GAN** | Replace reconstruction loss with discriminator |
| **CVAE** | Conditional VAE: $q_\phi(z \mid x, c)$, $p_\theta(x \mid z, c)$ |
| **VQ-VAE** | Discrete latent codes via vector quantization |
| **Normalizing Flow VAE** | Richer posterior via invertible transformations |

### 11.5 References

1. Kingma, D.P. & Welling, M. (2014). **Auto-Encoding Variational Bayes**. *ICLR 2014*. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
2. Rezende, D.J., Mohamed, S. & Wierstra, D. (2014). **Stochastic Backpropagation and Approximate Inference in Deep Generative Models**. *ICML 2014*.
3. Higgins, I. et al. (2017). **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**. *ICLR 2017*.
4. Burda, Y., Grosse, R. & Salakhutdinov, R. (2016). **Importance Weighted Autoencoders**. *ICLR 2016*.
5. Doersch, C. (2016). **Tutorial on Variational Autoencoders**. [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)

---

*Workshop Notes — From Autoencoders to Variational Autoencoders*  
*All mathematical derivations are self-contained. PyTorch ≥ 2.0 recommended.*
