# CAM, Grad-CAM and Their Extensions

Symbols are defined where they are first needed. Indices start at 0, as in PyTorch.

---

## The network as a function, and what "target layer" means

### A network is a composition of functions
Denote the input (an image or a signal) by $X$. A network with $n$ layers is the function
$$\hat y = f_n\circ f_{n-1}\circ\cdots\circ f_1\,(X)$$
Here $f_\ell$ is the transformation of layer $\ell$ (convolution, activation, pooling, linear layer, and so on). The final output is a vector $\hat y\in\mathbb R^{C}$, where $C$ is the number of classes. Its $c$-th entry is written $y^c$. It is the **raw score** (logit) of class $c$, before softmax.

### Splitting the network at one point
Choose a layer index $\ell^\*$ and split the network into two functions:
$$A=f(X)=f_{\ell^\*}\circ\cdots\circ f_1(X),\qquad y=g(A)=f_n\circ\cdots\circ f_{\ell^\*+1}(A)$$
- $A$ is the output of layer $\ell^\*$, called the **activation tensor**.
- $f$ is the **feature extractor** and $g$ is the **decision part**.
- Layer $\ell^\*$ is the **target layer**.

### What "the last layer" means
In CAM and Grad-CAM, "the last layer" means the **last convolutional layer** (together with its activation). It is the last layer whose output still has spatial (or temporal) structure. Why this layer?

1. **Spatial structure:** a convolution output has shape $(K,H,W)$, and entry $(i,j)$ still corresponds to a region of the input. After Flatten or a linear layer, this link is lost.
2. **High-level meaning:** deeper layers have larger receptive fields, so features move from edges and textures to object parts and then to concepts.
3. **Result:** the last conv layer gives the best trade-off between "where" and "what". The cost is low resolution. For example, in ResNet-18 with a 224×224 input, this layer has a 7×7 map.

**Example (ResNet-18):**
```
input (3×224×224) → conv1 → layer1 → layer2 → layer3 → layer4 → avgpool → fc → y
                                                          ↑
                                           target layer: A with shape (512, 7, 7)
```
Here $f$ runs from the input to `layer4`, and $g$ consists of `avgpool` and `fc`.

```python
import torch, torchvision
m = torchvision.models.resnet18(weights=None).eval()
x = torch.randn(1, 3, 224, 224)
f = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
g = lambda A: m.fc(torch.flatten(m.avgpool(A), 1))
A = f(x); y = g(A)
print(A.shape, y.shape)      # (1,512,7,7)  (1,1000)
print(torch.allclose(y, m(x), atol=1e-5))   # True
```
This code shows that $g(f(X))$ is the original network.

---

## Convolution, channels and feature maps

### 2D convolution
The layer input is a tensor with $M$ channels and size $H_{in}\times W_{in}$. Its entry is $X^m_{a,b}$ ($m$ is the channel, $a$ the row, $b$ the column). The layer has $K$ filters with these parameters:
- $\theta^k_{m,p,q}$: the weight of filter $k$ at input channel $m$ and kernel position $(p,q)$. The kernel size is $P\times Q$.
- $b_k$: the bias of filter $k$.
- $s$: the stride.
- $d$: the zero-padding on each side. $\tilde X$ is the padded input.
- $\sigma$: the activation function (for example ReLU).

The output of filter $k$ at position $(i,j)$ is
$$\boxed{A^k_{ij}=\sigma\Big(b_k+\sum_{m=0}^{M-1}\sum_{p=0}^{P-1}\sum_{q=0}^{Q-1}\theta^k_{m,p,q}\,\tilde X^m_{\,s i+p,\;s j+q}\Big)}$$
Mathematically this is cross-correlation, because the kernel is not flipped. PyTorch implements exactly this.

**Output size:**
$$H=\Big\lfloor\frac{H_{in}+2d-P}{s}\Big\rfloor+1,\qquad W=\Big\lfloor\frac{W_{in}+2d-Q}{s}\Big\rfloor+1$$

### Channel and feature map
- Each filter $k\in\{0,\dots,K-1\}$ produces a matrix $A^k\in\mathbb R^{H\times W}$. This matrix is a **feature map**.
- Stacking the $K$ maps gives the tensor $A\in\mathbb R^{K\times H\times W}$. Its first axis is the **channel axis**.
- For a batch of $N$ samples, the output shape is $(N,K,H,W)$.

"Channel" emphasizes the **position** in the tensor (the second axis). "Feature map" emphasizes the **content**: where the pattern of filter $k$ was found in the input. The layer has $K\cdot M\cdot P\cdot Q+K$ parameters.

**Hand-computed example.** Take $M=1$, $s=1$, $d=0$, $\sigma$ the identity, $b=0$:
$$X=\begin{bmatrix}1&2&0\\0&1&3\\1&0&2\end{bmatrix},\quad \theta=\begin{bmatrix}1&0\\0&1\end{bmatrix}$$
The output size is $H=W=3-2+1=2$, and
$$A_{00}=X_{00}+X_{11}=2,\; A_{01}=X_{01}+X_{12}=5,\; A_{10}=X_{10}+X_{21}=0,\; A_{11}=X_{11}+X_{22}=3$$
So $A=\begin{bmatrix}2&5\\0&3\end{bmatrix}$.
```python
import torch, torch.nn as nn
X = torch.tensor([[1.,2,0],[0,1,3],[1,0,2]])[None, None]       # (N=1, M=1, 3, 3)
conv = nn.Conv2d(1, 1, 2, bias=False)
conv.weight.data = torch.tensor([[[[1.,0],[0,1]]]])            # (K, M, P, Q)
print(conv(X))      # [[2,5],[0,3]]
conv8 = nn.Conv2d(1, 8, 2); print(conv8(X).shape)   # (1, 8, 2, 2): eight channels
```

### 1D convolution
The input is a signal with $M$ channels and length $L$. Its entry is $X^m_a$, and the kernel has length $P$:
$$\boxed{A^k_t=\sigma\Big(b_k+\sum_{m=0}^{M-1}\sum_{p=0}^{P-1}\theta^k_{m,p}\,\tilde X^m_{\,s t+p}\Big)},\qquad T=\Big\lfloor\frac{L+2d-P}{s}\Big\rfloor+1$$
The output shape is $(N,K,T)$.

**Example:** take $x=[1,2,0,3,1]$ and $\theta=[1,-1,2]$ ($s=1$, $d=0$). Then $T=3$ and
$$A_0=1-2+0=-1,\quad A_1=2-0+6=8,\quad A_2=0-3+2=-1$$
```python
x = torch.tensor([[[1.,2,0,3,1]]])        # (N=1, M=1, L=5)
c1 = nn.Conv1d(1, 1, 3, bias=False); c1.weight.data = torch.tensor([[[1.,-1,2]]])
print(c1(x))                              # [-1, 8, -1]
```

---

## Global average pooling and the linear layer

### Definitions
- $Z=H\cdot W$ is the number of spatial positions in a map.
- **GAP** (Global Average Pooling) turns each map into a single number:
$$F^k=\frac1Z\sum_{i=0}^{H-1}\sum_{j=0}^{W-1}A^k_{ij}$$
- The **linear layer** has weight $w^c_k$ (between feature $k$ and class $c$) and bias $\beta_c$:
$$y^c=\sum_{k=0}^{K-1}w^c_kF^k+\beta_c$$

### Numerical example
We reuse these maps later ($K=2$, $Z=4$):
$$A^0=\begin{bmatrix}2&0\\1&0\end{bmatrix},\quad A^1=\begin{bmatrix}0&1\\0&3\end{bmatrix}$$
Then $F^0=3/4=0.75$ and $F^1=4/4=1$. If $w^c=(4,-1)$ and $\beta_c=0$:
$$y^c=4(0.75)+(-1)(1)=2$$
```python
A = torch.tensor([[[2.,0],[1,0]], [[0.,1],[0,3]]])      # (K=2, 2, 2)
F_k = A.mean(dim=(1,2)); w = torch.tensor([4., -1.])
print(F_k, (w*F_k).sum())      # [0.75, 1.0]  2.0
```

---

## CAM (Class Activation Mapping)

### Architectural assumption
CAM is defined only when $g$ is exactly GAP followed by one linear layer:
$$A\xrightarrow{\text{GAP}}F\xrightarrow{\text{Linear}}y$$
### Step-by-step derivation
**Step 1.** From the definitions of $y^c$ and $F^k$:
$$y^c=\sum_k w^c_k\,\frac1Z\sum_{i,j}A^k_{ij}+\beta_c$$
**Step 2.** The factor $\frac1Z$ does not depend on $k$, and finite sums can be swapped:
$$y^c=\frac1Z\sum_{i,j}\Big(\sum_k w^c_kA^k_{ij}\Big)+\beta_c$$
**Step 3.** Define the CAM map as
$$\boxed{M^c_{ij}=\sum_{k}w^c_kA^k_{ij}}\quad\Longrightarrow\quad y^c=\frac1Z\sum_{i,j}M^c_{ij}+\beta_c$$
So the class score is the **mean** of the map $M^c$ plus the bias. $M^c_{ij}$ is the contribution of position $(i,j)$ to the score. This decomposition is exact, not approximate.

**Step 4 (display).** For display, upsample $M^c$ to the input size (interpolation) and rescale it to $[0,1]$:
$$\bar M^c=\frac{M^c-\min M^c}{\max M^c-\min M^c}$$

### Numerical example
With the data above:
$$M^c=4A^0-1\cdot A^1=\begin{bmatrix}8&-1\\4&-3\end{bmatrix}$$
The mean of $M^c$ is $(8-1+4-3)/4=2$, which equals $y^c=2$. This confirms the decomposition. The top-left position contributes most.

### Implementation
```python
import torch, torch.nn as nn, torch.nn.functional as F

class TinyNet(nn.Module):
    def __init__(s, K=8, C=3):
        super().__init__()
        s.features = nn.Sequential(nn.Conv2d(1,4,3,padding=1), nn.ReLU(),
                                   nn.Conv2d(4,K,3,padding=1), nn.ReLU())  # target layer
        s.fc = nn.Linear(K, C)
    def head(s, A): return s.fc(A.mean(dim=(2,3)))          # g
    def forward(s, x): return s.head(s.features(x))

net = TinyNet().eval(); x = torch.randn(1,1,16,16)
A = net.features(x); y = net.head(A); c = y.argmax(1).item()
M = torch.einsum('k,khw->hw', net.fc.weight[c], A[0])      # M^c_ij
# check: mean of M plus the bias equals the score
print(torch.allclose(M.mean() + net.fc.bias[c], y[0, c], atol=1e-5))   # True
```

### Limitation
CAM needs the GAP+Linear architecture. For other networks you would have to change the structure and retrain. Grad-CAM removes this limitation.

---

## Grad-CAM

### Idea
In CAM, the weight $w^c_k$ says how important channel $k$ is for class $c$. A general network has no such weight. But $g$ is differentiable, so we can build the importance from the **gradient**. Denote the gradient of the score with respect to the activation by $G$ (to avoid confusion with the function $g(\cdot)$):
$$G^{c,k}_{ij}=\frac{\partial y^c}{\partial A^k_{ij}}$$

### Definition
$$\alpha^c_k=\frac1Z\sum_{i,j}G^{c,k}_{ij},\qquad \boxed{L^c=\mathrm{ReLU}\Big(\sum_k\alpha^c_kA^k\Big)}$$
Then upsample and normalize as in CAM.

### Proof: in the CAM architecture, Grad-CAM equals CAM
In the CAM architecture, $y^c=\frac1Z\sum_{i,j}\sum_k w^c_kA^k_{ij}+\beta_c$.

**Step 1.** In this sum, the variable $A^k_{ij}$ appears once, with coefficient $w^c_k/Z$. So
$$G^{c,k}_{ij}=\frac{w^c_k}{Z}$$
**Step 2.** This value does not depend on $(i,j)$. Averaging over the $Z$ positions gives
$$\alpha^c_k=\frac1Z\sum_{i,j}\frac{w^c_k}{Z}=\frac1Z\cdot Z\cdot\frac{w^c_k}{Z}=\frac{w^c_k}{Z}$$
**Step 3.** Therefore
$$\sum_k\alpha^c_kA^k=\frac1Z M^c$$
The Grad-CAM map (before ReLU) is the CAM map divided by the constant $Z$. Normalization removes positive constants, so after normalization the two maps are identical. **The correct weight is $w^c_k/Z$, not $w^c_k$.**

### Why is the mean gradient a measure of importance? (Taylor expansion)
**Step 1.** Expand $y^c=g_c(A)$ to first order around the current activation $A$. If we move the activation to $A+\Delta A$:
$$y^c(A+\Delta A)\approx y^c(A)+\sum_{k,i,j}G^{c,k}_{ij}\,\Delta A^k_{ij}$$
**Step 2.** Suppose we remove the features of channel $k$ entirely, i.e. $\Delta A^k=-A^k$. The change in score is approximately
$$\Delta y^c\approx-\sum_{i,j}G^{c,k}_{ij}A^k_{ij}$$
So $\sum_{i,j}G^{c,k}_{ij}A^k_{ij}$ is the **contribution** of channel $k$ to the score (to first order).

**Step 3.** The total contribution of all channels is
$$\sum_k\sum_{i,j}G^{c,k}_{ij}A^k_{ij}$$
(If the network uses ReLU and has no biases, this is exact by Euler's theorem for homogeneous functions. In general it is an approximation.)

**Step 4 (the Grad-CAM approximation).** Replace the position-dependent gradient $G^{c,k}_{ij}$ with its spatial mean $\alpha^c_k$:
$$\sum_k\sum_{i,j}\alpha^c_kA^k_{ij}=\sum_{i,j}\Big(\sum_k\alpha^c_kA^k_{ij}\Big)$$
The expression in parentheses is the Grad-CAM map before ReLU. So this map shows each position's contribution to the score, under the approximation that a channel's effect strength is the same at every position.

**Step 5 (ReLU).** Positions with negative values are evidence **against** class $c$. ReLU keeps only the evidence in favor of the class.

### Why use $y^c$ before softmax?
With $p^c=\mathrm{softmax}(y)_c$ we have $\frac{\partial p^c}{\partial y^c}=p^c(1-p^c)$ and $\frac{\partial p^c}{\partial y^{c'}}=-p^cp^{c'}$. These derivatives bring in the other classes, and when $p^c\to1$ the gradient goes to zero (saturation). The raw score gives the pure effect of class $c$.

### Numerical example 1 (same data as CAM)
Earlier we had $w^c=(4,-1)$ and $Z=4$. So $G^{c,0}\equiv1$ and $G^{c,1}\equiv-0.25$. The weights are $\alpha=(1,-0.25)$, i.e. $w/Z$:
$$\alpha_0A^0+\alpha_1A^1=\begin{bmatrix}2&0\\1&0\end{bmatrix}-0.25\begin{bmatrix}0&1\\0&3\end{bmatrix}=\begin{bmatrix}2&-0.25\\1&-0.75\end{bmatrix}\xrightarrow{\text{ReLU}}\begin{bmatrix}2&0\\1&0\end{bmatrix}$$
CAM after ReLU was $\begin{bmatrix}8&0\\4&0\end{bmatrix}$, i.e. four times larger ($Z=4$). After normalization the two are identical.

### Numerical example 2 (non-uniform gradient)
Suppose
$$G^{0}=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad G^{1}=\begin{bmatrix}0&0\\0&-0.4\end{bmatrix}$$
Then $\alpha_0=0.25$, $\alpha_1=-0.1$, and
$$0.25A^0-0.1A^1=\begin{bmatrix}0.5&-0.1\\0.25&-0.3\end{bmatrix}\xrightarrow{\text{ReLU}}\begin{bmatrix}0.5&0\\0.25&0\end{bmatrix}$$
Only one position of channel 0 had a nonzero gradient, but averaging spread it over all four positions. This weakness is addressed by Grad-CAM++.

### Implementation and verification
```python
def grad_cam(features, head, x, c=None):
    A = features(x)                                   # (1,K,H,W)
    y = head(A)                                       # (1,C) raw scores
    c = y.argmax(1).item() if c is None else c
    G, = torch.autograd.grad(y[0, c], A)              # (1,K,H,W)
    alpha = G.mean(dim=(2,3), keepdim=True)           # (1,K,1,1)
    cam = F.relu((alpha * A).sum(1, keepdim=True))    # (1,1,H,W)
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam[0, 0].detach()

# check on TinyNet: alpha equals w/Z
A = net.features(x); y = net.head(A); c = y.argmax(1).item()
G, = torch.autograd.grad(y[0, c], A)
Z = A.shape[2] * A.shape[3]
print(torch.allclose(G.mean(dim=(2,3))[0], net.fc.weight[c] / Z, atol=1e-6))   # True
cam_gc = grad_cam(net.features, net.head, x, c)       # same size as x
```

For pretrained models, use hooks:
```python
def grad_cam_hook(model, layer, x, c=None):
    store = {}
    def fh(m, i, o): store['A'] = o; o.retain_grad()
    h = layer.register_forward_hook(fh)
    y = model(x); c = y.argmax(1).item() if c is None else c
    model.zero_grad(); y[0, c].backward(); h.remove()
    A, G = store['A'], store['A'].grad
    alpha = G.mean(dim=(2,3), keepdim=True)
    cam = F.relu((alpha * A).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))[0, 0]
# cam = grad_cam_hook(m, m.layer4, x)   # m is the ResNet-18 from the first section
```

---

## Extensions

### Grad-CAM++

**Motivation.** Numerical example 2 showed that uniform averaging of gradients down-weights positions with large gradients. If an object is small or appears several times, coverage is incomplete.

**Goal.** We want a separate weight $\alpha^{c,k}_{ij}$ for each position, such that
$$w^c_k=\sum_{i,j}\alpha^{c,k}_{ij}\,\mathrm{ReLU}(G^{c,k}_{ij}),\qquad L^c=\mathrm{ReLU}\Big(\sum_kw^c_kA^k\Big)$$

**Derivation (following the paper; heuristic).** Fix channel $k$ and drop the indices $k$ and $c$. Write $Y$ for the class score and assume the CAM-like form $Y=\sum_k w_k\sum_{a,b}A^k_{ab}$, with $w_k=\sum_{a,b}\alpha_{ab}\,\mathrm{ReLU}(\partial Y/\partial A_{ab})$. Then
$$Y=\Big(\sum_{a,b}\alpha_{ab}\frac{\partial Y}{\partial A_{ab}}\Big)\sum_{a,b}A_{ab}$$
Differentiating twice with respect to $A_{ij}$ (treating $\alpha$ as locally constant, as the paper does) and solving for $\alpha_{ij}$ gives
$$\boxed{\alpha^{c,k}_{ij}=\frac{\dfrac{\partial^2y^c}{(\partial A^k_{ij})^2}}{2\dfrac{\partial^2y^c}{(\partial A^k_{ij})^2}+\sum_{a,b}A^k_{ab}\dfrac{\partial^3y^c}{(\partial A^k_{ij})^3}}}$$
This derivation is heuristic and depends on simplifying assumptions. The paper does not claim full rigor, and I have not expanded the intermediate steps here.

**Higher derivatives.** The paper assumes $Y^c=\exp(S^c)$ with $S^c$ linear in $A$. With $G=\partial S^c/\partial A_{ij}$:
$$\frac{\partial Y}{\partial A_{ij}}=e^{S}G,\quad\frac{\partial^2Y}{\partial A_{ij}^2}=e^{S}G^2,\quad\frac{\partial^3Y}{\partial A_{ij}^3}=e^{S}G^3$$
The factor $e^{S}$ cancels between numerator and denominator:
$$\alpha_{ij}=\frac{G^2}{2G^2+\sum_{a,b}A_{ab}G^3}$$
The $e^S$ factor in $\mathrm{ReLU}(\partial Y/\partial A)$ is a constant that disappears after normalization. In practice, $G$ is the gradient of the raw score.

**Numerical example.** For channel 0 above, with constant gradient $G=0.5$ and $\sum A=3$:
$$\alpha=\frac{0.25}{0.5+3(0.125)}=\frac{0.25}{0.875}\approx0.2857$$
$$w=\sum_{i,j}\alpha\cdot\mathrm{ReLU}(0.5)=4\times0.2857\times0.5\approx0.571$$
(In Grad-CAM the value was $0.5$.) For a channel with negative gradient, $\mathrm{ReLU}(G)=0$, so $w=0$.
```python
def grad_cam_pp(A, G):                 # A, G: (1,K,H,W)
    g2, g3 = G**2, G**3
    denom = 2*g2 + A.sum(dim=(2,3), keepdim=True) * g3
    alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
    w = (alpha * F.relu(G)).sum(dim=(2,3), keepdim=True)
    return F.relu((w * A).sum(1, keepdim=True))
# numerical check of the example above:
A0 = torch.tensor([[[[2.,0],[1,0]]]]); G0 = torch.full_like(A0, 0.5)
print(grad_cam_pp(A0, G0))   # w ≈ 0.571, map = 0.571 * A0
```

### Score-CAM

**Motivation.** Gradients can be noisy or saturated, and when they vanish the map becomes meaningless. Score-CAM uses no gradients. It measures each channel's importance by its **direct effect on the output**.

**Symbols.** $f^c(\cdot)$ is the score of class $c$ as a function of the network **input**. $\odot$ is element-wise multiplication. $X_0$ is a baseline input (for example all zeros).

**Algorithm:**
1. Upsample each map $A^k$ to the input size and normalize it to $[0,1]$. The result is the mask $\tilde M^k$.
2. Feed the masked input $X\odot\tilde M^k$ to the network.
3. The channel's importance is the score increase over the baseline:
$$s^c_k=f^c(X\odot\tilde M^k)-f^c(X_0)$$
4. Normalize the weights with a softmax over $k$: $\alpha^c_k=\dfrac{e^{s^c_k}}{\sum_{k'}e^{s^c_{k'}}}$.
5. The final map is $L^c=\mathrm{ReLU}\big(\sum_k\alpha^c_kA^k\big)$.

**Intuition.** If the mask of a channel that lights up a "cat ear" raises the "cat" score a lot, that channel gets a large $s^c_k$ and a high weight, with no gradient involved.

**Cost:** $K$ forward passes (512 for ResNet-18). To save memory, feed the masks in batches.
```python
@torch.no_grad()
def score_cam(features, model_fn, x, c):       # x: (1,3,H,W)
    A = features(x)                            # (1,K,h,w)
    M = F.interpolate(A, size=x.shape[2:], mode='bilinear', align_corners=False)
    mn, mx = M.amin(dim=(2,3), keepdim=True), M.amax(dim=(2,3), keepdim=True)
    M = (M - mn) / (mx - mn + 1e-8)            # (1,K,H,W)
    masked = x * M.transpose(0, 1)             # (K,3,H,W)
    s = model_fn(masked)[:, c] - model_fn(torch.zeros_like(x))[:, c]
    alpha = torch.softmax(s, dim=0)            # (K,)
    return F.relu((alpha[:, None, None] * A[0]).sum(0))
```

### Comparison

| Method | Channel weight | Gradient | Cost | Strength |
|---|---|---|---|---|
| CAM | $w^c_k$ | No | Low | Exact, but needs a specific architecture |
| Grad-CAM | Mean of $G$ | Yes | Low | Any architecture |
| Grad-CAM++ | Spatial weights from higher derivatives | Yes | Low | Multiple objects, better coverage |
| Score-CAM | Score increase under masking | No | $K$ forward passes | Independent of gradient noise |

---

## Application to single-channel signals

### Mathematics
A single-lead signal (such as ECG) has input shape $(N,1,L)$. "Single-channel" refers only to the input. As in the 1D convolution section, the output of the first Conv1d has several channels, so at the target layer $A\in\mathbb R^{K\times T}$. The substitutions are:

- $Z=T$, and position $(i,j)$ becomes time $t$.
- $G^{c,k}_t=\partial y^c/\partial A^k_t$.

$$\alpha^c_k=\frac1T\sum_{t=0}^{T-1}G^{c,k}_t,\qquad L^c_t=\mathrm{ReLU}\Big(\sum_k\alpha^c_kA^k_t\Big)$$

The proof of equality with CAM is exactly the same as in the Grad-CAM section, with $Z=T$. The result is a **temporal importance curve** of length $T$, which is brought to length $L$ by `linear` interpolation.

**Numerical example.** Take $K=2$, $T=4$, $A^0=[1,3,0,0]$, $A^1=[0,0,2,2]$ and $\alpha=(0.5,-0.5)$:
$$0.5A^0-0.5A^1=[0.5,1.5,-1,-1]\xrightarrow{\text{ReLU}}[0.5,1.5,0,0]$$
The importance lies in the first two time steps.

### Implementation and a pulse experiment
```python
class Net1D(nn.Module):
    def __init__(s, K=8, C=2):
        super().__init__()
        s.features = nn.Sequential(nn.Conv1d(1,4,7,padding=3), nn.ReLU(),
                                   nn.Conv1d(4,K,7,padding=3), nn.ReLU(),
                                   nn.MaxPool1d(2))               # length: L → L/2
        s.fc = nn.Linear(K, C)
    def head(s, A): return s.fc(A.mean(dim=2))
    def forward(s, x): return s.head(s.features(x))

def grad_cam_1d(model, x, c=None):
    A = model.features(x)                                  # (1,K,T)
    y = model.head(A); c = y.argmax(1).item() if c is None else c
    G, = torch.autograd.grad(y[0, c], A)                   # (1,K,T)
    alpha = G.mean(dim=2, keepdim=True)
    cam = F.relu((alpha * A).sum(1, keepdim=True))         # (1,1,T)
    cam = F.interpolate(cam, size=x.shape[-1], mode='linear', align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam[0, 0].detach()

def make(cls, L=128):                                      # class 1: pulse at samples 60 to 69
    s = 0.1 * torch.randn(L)
    if cls == 1: s[60:70] += 2.0
    return s

model = Net1D(); opt = torch.optim.Adam(model.parameters(), 1e-2)
for _ in range(200):
    ys = torch.randint(0, 2, (32,))
    xb = torch.stack([make(int(c)) for c in ys])[:, None]
    loss = F.cross_entropy(model(xb), ys); opt.zero_grad(); loss.backward(); opt.step()

cam = grad_cam_1d(model, make(1)[None, None], c=1)
print(cam.argmax().item())      # expected: a number near 60 to 70
```

### Notes
- **Temporal resolution** equals $T$. Pooling and stride reduce it. For short events, choose a layer with larger $T$.
- **Multi-channel input** (for example 12-lead ECG): the same method works and gives a shared temporal curve. For per-lead importance, take the gradient with respect to the input itself.
- **Display:** plot $L^c_t$ as a background color under the signal.

---

# Note: Multiple FC Layers After GAP

After the last convolutional layer and GAP, a network may have several fully connected (FC) layers before the output. In that case, we must decide what counts as $y^c$ in CAM.

**The rule:** $y^c$ is always the **final raw class score** (the logit before softmax). It is never the output of a hidden FC layer, because hidden units are not class scores.

**The condition for exact CAM:** The CAM decomposition needs the path from GAP to the score to be a single linear map. Two cases follow.

**Case 1: Several linear layers with no activation between them.** Linear maps combine into one linear map. If $F$ is the GAP output, $h_1=W_1F+b_1$, and $y=W_2h_1+b_2$, then

$$y=(W_2W_1)F+(W_2b_1+b_2).$$

So we can treat the stack as one linear layer with weight matrix $W_2W_1$, and take the CAM weights $w^c$ from row $c$ of this matrix.

*Example:* $W_1=\begin{bmatrix}1&0\\0&2\end{bmatrix}$ and $W_2=\begin{bmatrix}3&1\\0&1\end{bmatrix}$. Then $W_2W_1=\begin{bmatrix}3&2\\0&2\end{bmatrix}$. For class 1, the CAM weights are $w^1=(3,2)$.

**Case 2: Several FC layers with a non-linear activation between them.** This is the most common case, for example with ReLU. The composition is no longer a fixed matrix, so the score is not a linear function of $F$. The CAM decomposition is then not defined. Use **Grad-CAM** instead. It needs only that $y^c$ is differentiable with respect to the activation $A$. The gradient flows back through all FC layers, so $G^{c,k}_{ij}=\partial y^c/\partial A^k_{ij}$ is still well defined.

**Summary table:**

| Layers after GAP | Score $y^c$ | Exact CAM? | Recommended method |
|---|---|---|---|
| One linear layer | Final output | Yes | CAM or Grad-CAM (they are equivalent) |
| Several linear layers, no activation | Final output | Yes, after combining the matrices | CAM with $W_2W_1$ |
| Several layers with activation | Final output | No | Grad-CAM |

**Warning:** The outputs of hidden FC layers are not class scores. Do not use them in place of $y^c$.

---

## Example: Grad-CAM with a pretrained model on a benchmark dataset

This example uses a ResNet-18 model pretrained on ImageNet, and the `grad-cam` package. The dataset is the ImageNet validation set, arranged in one folder per class (the layout that `ImageFolder` expects).

```python
# pip install grad-cam torch torchvision
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms, datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

DATA_DIR = "/path/to/imagenet/val"        # one sub-folder per class

# 1. Load a pretrained model
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights).eval()
categories = weights.meta["categories"]   # names of the 1000 classes

# 2. Preprocessing: one version for the model, one for display
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
resize_crop = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224)])
to_model_input = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])

# 3. Load the benchmark dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=to_model_input)

# 4. Grad-CAM on the last convolutional block (not on fc)
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# 5. Take one image; keep its path so we can show the same image
idx = 0
path, label = dataset.samples[idx]
x = dataset[idx][0].unsqueeze(0)          # shape (1, 3, 224, 224)

with torch.no_grad():
    pred = model(x).argmax(dim=1).item()

# 6. Heatmap for the predicted class
heatmap = cam(input_tensor=x,
              targets=[ClassifierOutputTarget(pred)])[0]   # shape (224, 224)

# 7. Overlay the heatmap on the original image
img = resize_crop(Image.open(path).convert("RGB"))
rgb = np.asarray(img, dtype=np.float32) / 255.0
overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)

print("Ground truth:", dataset.classes[label])
print("Prediction:  ", categories[pred])
Image.fromarray(overlay).save("gradcam_example.png")
```

**Notes for students:**
- The target layer is `model.layer4[-1]`, the last convolutional block. The `fc` layer is not used, because its output has no spatial map.
- `ClassifierOutputTarget(pred)` tells the package to use the raw score of the predicted class $y^c$ (before softmax). This is the score defined in the handout.
- Red areas in the overlay show where the network found the most evidence for the predicted class.

*Caution: this code has not been run. Check it with real data before using it in the handout. Also check the names and arguments of `show_cam_on_image` and `ClassifierOutputTarget` against the version you have installed.*