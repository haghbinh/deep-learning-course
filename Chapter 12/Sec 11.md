# فصل ۱۲: ترنسفورمرها (Transformers) ادامه...
## جلسه 11
---

## بخش چهارم: گسترش‌های خودتوجهی ضرب نقطه‌ای

در جلسهٔ ۱۰ دیدیم که خودتوجهی ضرب نقطه‌ای در شکل ماتریسی به صورت زیر است:

$$
\mathbf{Sa}[\mathbf{X}] = \mathbf{V}[\mathbf{X}]\; \text{Softmax}\Big[\mathbf{K}[\mathbf{X}]^{\top} \mathbf{Q}[\mathbf{X}]\Big]
$$

این سازوکار در عمل تقریباً همیشه با سه گسترش به کار می‌رود: **رمزگذاری موقعیت** که ترتیب نشانه‌ها را وارد مدل می‌کند، **مقیاس‌دهی** ضرب‌های نقطه‌ای که آموزش را پایدار می‌کند، و **چند سر** که چند سازوکار توجه را به‌طور موازی اجرا می‌کند.

### رمزگذاری موقعیت (Positional encoding)

#### مشکل: خودتوجهی ترتیب را نمی‌بیند

در خودتوجهی، پارامترهای $\boldsymbol{\Omega}_v, \boldsymbol{\Omega}_q, \boldsymbol{\Omega}_k$ و $\boldsymbol{\beta}_v, \boldsymbol{\beta}_q, \boldsymbol{\beta}_k$ بین همهٔ جایگاه‌ها مشترک‌اند و وزن‌های توجه فقط به **محتوای** ورودی‌ها بستگی دارند، نه به جای آن‌ها. نتیجه این است که اگر ترتیب ورودی‌ها را عوض کنیم، خروجی‌ها هم دقیقاً به همان ترتیب جابه‌جا می‌شوند. اگر $\mathbf{P}$ یک **ماتریس جایگشت** (Permutation matrix) $N \times N$ باشد، $\mathbf{X}\mathbf{P}$ همان ستون‌های $\mathbf{X}$ با ترتیبی دیگر است و:

$$
\mathbf{Sa}[\mathbf{X}\mathbf{P}] = \mathbf{Sa}[\mathbf{X}]\,\mathbf{P}
$$

یعنی خودتوجهی نسبت به جایگشت ورودی‌ها **هم‌وردا** (Equivariant) است (مفهوم هم‌وردایی را در جلسهٔ ۲ دیدیم). دلیل آن:

- چون همهٔ ستون‌های $\boldsymbol{\beta}_v\mathbf{1}^{\top}$ یکسان‌اند، $\boldsymbol{\beta}_v\mathbf{1}^{\top}\mathbf{P} = \boldsymbol{\beta}_v\mathbf{1}^{\top}$ و در نتیجه $\mathbf{V}[\mathbf{X}\mathbf{P}] = \mathbf{V}[\mathbf{X}]\mathbf{P}$؛ برای $\mathbf{Q}$ و $\mathbf{K}$ هم همین‌طور.
- ماتریس ضرب‌های نقطه‌ای به $\mathbf{P}^{\top}\mathbf{K}^{\top}\mathbf{Q}\mathbf{P}$ تبدیل می‌شود، یعنی سطرها و ستون‌هایش به یک ترتیب جابه‌جا می‌شوند. `softmax` ستونی با این جابه‌جایی جابه‌جا می‌شود: $\text{Softmax}[\mathbf{P}^{\top}\mathbf{A}\mathbf{P}] = \mathbf{P}^{\top}\text{Softmax}[\mathbf{A}]\mathbf{P}$.
- بنابراین $\mathbf{Sa}[\mathbf{X}\mathbf{P}] = \mathbf{V}\mathbf{P}\,\mathbf{P}^{\top}\text{Softmax}[\mathbf{K}^{\top}\mathbf{Q}]\,\mathbf{P} = \mathbf{Sa}[\mathbf{X}]\,\mathbf{P}$، چون $\mathbf{P}\mathbf{P}^{\top} = \mathbf{I}$.

اما ترتیب کلمه‌ها در متن معنادار است: دو جملهٔ «علی به رضا زنگ زد» و «رضا به علی زنگ زد» دقیقاً همان کلمه‌ها را دارند، ولی معنایشان متفاوت است. خودتوجهی به‌تنهایی برای کلمهٔ «علی» در هر دو جمله **همان خروجی** را می‌سازد. پس باید اطلاعات موقعیت را به‌طور صریح به مدل بدهیم.

> [!note] مقایسه با کانولوشن
> جایگشت $\mathbf{X} \mapsto \mathbf{X}\mathbf{P}$ یک تبدیل **خطی** و وارون‌پذیر است ($\mathbf{P}^{-1} = \mathbf{P}^{\top}$). انتقال دایره‌ای در CNN (جلسهٔ ۲) هم یک جایگشت خاص، یعنی جایگشت **چرخشی**، است. کانولوشن فقط نسبت به این جایگشت‌های خاص هم‌وردا است و ساختار همسایگی را حفظ می‌کند؛ اما خودتوجهی نسبت به **همهٔ** $N!$ جایگشت هم‌وردا است و هیچ اطلاعاتی از ترتیب ندارد.

#### رمزگذاری موقعیت مطلق (Absolute positional encoding)

ساده‌ترین راه این است که یک ماتریس $\boldsymbol{\Pi} \in \mathbb{R}^{D \times N}$ به ورودی اضافه کنیم:

$$
\mathbf{X} \;\longrightarrow\; \mathbf{X} + \boldsymbol{\Pi}
$$

ستون $n$ام $\boldsymbol{\Pi}$ برداری است که فقط به موقعیت $n$ بستگی دارد و برای هر موقعیت متفاوت است؛ بنابراین یک کلمه در دو موقعیت مختلف، دو ورودی متفاوت خواهد داشت. ماتریس $\boldsymbol{\Pi}$ را می‌توان به دو صورت ساخت.

**از پیش تعیین‌شده:** رایج‌ترین انتخاب، رمزگذاری **سینوسی** مقالهٔ اصلی ترنسفورمر است. برای موقعیت $n$ و $i = 0, 1, \ldots, D/2 - 1$:

$$
\Pi_{2i,\,n} = \sin\!\left(\frac{n}{10000^{2i/D}}\right),
\qquad
\Pi_{2i+1,\,n} = \cos\!\left(\frac{n}{10000^{2i/D}}\right)
$$

هر جفت سطر یک موج سینوسی با بسامد متفاوت است: سطرهای بالایی با تغییر $n$ سریع نوسان می‌کنند و سطرهای پایینی بسیار کند؛ مانند عقربه‌های ثانیه‌شمار، دقیقه‌شمار و ساعت‌شمار که با هم یک زمان را به‌طور یکتا نشان می‌دهند. این رمزگذاری پارامتر ندارد و برای هر طولی تعریف می‌شود.

> [!example] مثال
> فرض کنید $D = 4$ و موقعیت‌ها را از صفر شماره‌گذاری کنیم. با فرمول بالا، سه ستون اول $\boldsymbol{\Pi}$ به صورت زیر است (سطرهای ۱ و ۲ با بسامد $1$ و سطرهای ۳ و ۴ با بسامد $1/100$):
>
> $$
> \boldsymbol{\pi}_0 = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 1 \end{bmatrix},
> \qquad
> \boldsymbol{\pi}_1 = \begin{bmatrix} 0.841 \\ 0.540 \\ 0.010 \\ 1.000 \end{bmatrix},
> \qquad
> \boldsymbol{\pi}_2 = \begin{bmatrix} 0.909 \\ -0.416 \\ 0.020 \\ 1.000 \end{bmatrix}
> $$
>
> اگر نهفتهٔ کلمهٔ «علی» $\mathbf{x} = [0.5,\, -0.2,\, 0.1,\, 0.3]^{\top}$ باشد، این کلمه در جملهٔ «علی به رضا زنگ زد» در موقعیت ۰ و در جملهٔ «رضا به علی زنگ زد» در موقعیت ۲ است و ورودی شبکه در دو جمله متفاوت می‌شود:
>
> $$
> \mathbf{x} + \boldsymbol{\pi}_0 = \begin{bmatrix} 0.5 \\ 0.8 \\ 0.1 \\ 1.3 \end{bmatrix},
> \qquad
> \mathbf{x} + \boldsymbol{\pi}_2 = \begin{bmatrix} 1.409 \\ -0.616 \\ 0.120 \\ 1.300 \end{bmatrix}
> $$
>
> پس خودتوجهی دیگر برای «علی» در دو جمله خروجی یکسان نمی‌سازد. همچنین دیده می‌شود که سطرهای ۱ و ۲ از یک موقعیت به موقعیت بعد زیاد تغییر می‌کنند، ولی سطرهای ۳ و ۴ بسیار کند؛ سطرهای سریع موقعیت‌های نزدیک را از هم جدا می‌کنند و سطرهای کند موقعیت‌های دور را.

**یادگرفتنی:** درایه‌های $\boldsymbol{\Pi}$ پارامترهایی هستند که همراه با بقیهٔ شبکه آموزش می‌بینند. در این حالت طول دنباله نمی‌تواند از بیشینهٔ $N$ در زمان آموزش بیشتر شود.

> [!note] جای اضافه کردن $\boldsymbol{\Pi}$
> رمزگذاری موقعیت معمولاً یک بار به ورودی شبکه اضافه می‌شود، ولی می‌توان آن را در هر لایه هم اضافه کرد. گاهی هم $\boldsymbol{\Pi}$ فقط هنگام محاسبهٔ پرسش‌ها و کلیدها به $\mathbf{X}$ اضافه می‌شود و در محاسبهٔ مقدارها نه.

#### رمزگذاری موقعیت نسبی (Relative positional encoding)

موقعیت مطلق یک کلمه اغلب چندان مهم نیست؛ مثلاً یک جمله چه در ابتدای متن باشد و چه در انتهای آن، معنایش یکی است. آنچه مهم است **فاصلهٔ نسبی** کلمه‌هاست. درایهٔ $(m, n)$ ماتریس توجه به کلید موقعیت $m$ و پرسش موقعیت $n$ مربوط است؛ پس به یک فاصلهٔ مشخص $m - n$ مربوط است. در رمزگذاری نسبی، برای هر فاصله یک پارامتر $\pi_{m-n}$ یاد گرفته می‌شود و با آن ماتریس توجه تغییر داده می‌شود؛ مثلاً $\pi_{m-n}$ پیش از `softmax` به $\mathbf{k}_m^{\top}\mathbf{q}_n$ اضافه می‌شود (یا در آن ضرب می‌شود). به این ترتیب مدل می‌تواند الگوهایی مانند «به کلمهٔ قبلی توجه کن» را مستقل از جای جمله در متن یاد بگیرد.

کد زیر هم‌وردایی خودتوجهی را بررسی می‌کند و نشان می‌دهد که افزودن رمزگذاری سینوسی آن را از بین می‌برد:

```python
import torch

torch.manual_seed(0)

def self_attention(X, Ov, Oq, Ok, bv, bq, bk):
    V, Q, K = bv + Ov @ X, bq + Oq @ X, bk + Ok @ X
    return V @ torch.softmax(K.T @ Q, dim=0)

D, N = 4, 3
X = torch.randn(D, N)
params = [0.5 * torch.randn(D, D) for _ in range(3)] + [0.5 * torch.randn(D, 1) for _ in range(3)]

P = torch.eye(N)[:, [2, 0, 1]]             # permutation matrix: reorders the columns of X
print("Sa[XP] == Sa[X]P :", torch.allclose(self_attention(X @ P, *params),
                                           self_attention(X, *params) @ P, atol=1e-6))

# sinusoidal positional encoding Pi (D x N), added to the input
pos = torch.arange(N).float()                            # positions 0, ..., N-1
freq = 1.0 / 10000 ** (torch.arange(0, D, 2).float() / D)
Pi = torch.zeros(D, N)
Pi[0::2] = torch.sin(freq[:, None] * pos)
Pi[1::2] = torch.cos(freq[:, None] * pos)

print("with Pi:", torch.allclose(self_attention(X@P+Pi,*params),
                                           self_attention(X + Pi, *params) @ P, atol=1e-6))
```

```
Sa[XP] == Sa[X]P : True
with Pi          : False
```

### خودتوجهی ضرب نقطه‌ای مقیاس‌شده (Scaled dot-product self-attention)

ضرب‌های نقطه‌ای $\mathbf{k}_m^{\top}\mathbf{q}_n$ می‌توانند اندازهٔ بزرگی داشته باشند. در این حالت `softmax` **اشباع** می‌شود: بزرگ‌ترین مقدار وزنی نزدیک ۱ و بقیه وزنی نزدیک صفر می‌گیرند. اگر $s_j$ ورودی‌های `softmax` و $a_m$ خروجی‌های آن باشند، مشتق آن $\partial a_m / \partial s_j = a_m(\delta_{mj} - a_j)$ است و وقتی بردار وزن‌ها تقریباً یک‌داغ باشد، همهٔ این مشتق‌ها تقریباً صفرند. یعنی تغییرهای کوچک در ورودی‌ها تقریباً هیچ اثری بر خروجی ندارند و آموزش شبکه دشوار می‌شود.

چرا ضرب‌های نقطه‌ای بزرگ می‌شوند؟ اگر درایه‌های $\mathbf{q}$ و $\mathbf{k}$ مستقل، با میانگین صفر و واریانس ۱ باشند، $\mathbf{k}^{\top}\mathbf{q} = \sum_{d=1}^{D_q} k_d\, q_d$ مجموع $D_q$ جملهٔ مستقل با واریانس ۱ است؛ پس واریانس آن $D_q$ و انحراف معیارش $\sqrt{D_q}$ است ($D_q$ بُعد پرسش‌ها و کلیدهاست). برای $D_q = 1024$ این انحراف معیار ۳۲ است. راه‌حل این است که ضرب‌های نقطه‌ای را بر $\sqrt{D_q}$ تقسیم کنیم:

$$
\mathbf{Sa}[\mathbf{X}] = \mathbf{V}[\mathbf{X}]\; \text{Softmax}\!\left[\frac{\mathbf{K}[\mathbf{X}]^{\top} \mathbf{Q}[\mathbf{X}]}{\sqrt{D_q}}\right]
$$

این شکل را **خودتوجهی ضرب نقطه‌ای مقیاس‌شده** می‌نامند و تقریباً همهٔ ترنسفورمرها از آن استفاده می‌کنند.

```python
import torch

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)

for Dq in [4, 64, 1024]:
    q, k = torch.randn(Dq, 10000), torch.randn(Dq, 10000)    # 10000 random (q, k) pairs
    dots = (k * q).sum(dim=0)                                 # k^T q
    print(f"D_q = {Dq:4d}:  std(k^T q) = {dots.std():5.2f},  std(k^T q / sqrt(D_q)) = {(dots / Dq**0.5).std():.2f}")

s = torch.randn(6)                         # scaled scores of one query against N = 6 keys
print("softmax[s]     :", torch.softmax(s, dim=0))
print("softmax[32 * s]:", torch.softmax(32 * s, dim=0))      # the same scores without scaling, D_q = 1024
```

```
D_q =    4:  std(k^T q) =  1.99,  std(k^T q / sqrt(D_q)) = 1.00
D_q =   64:  std(k^T q) =  7.91,  std(k^T q / sqrt(D_q)) = 0.99
D_q = 1024:  std(k^T q) = 32.05,  std(k^T q / sqrt(D_q)) = 1.00
softmax[s]     : tensor([0.276, 0.070, 0.111, 0.226, 0.111, 0.207])
softmax[32 * s]: tensor([0.998, 0.000, 0.000, 0.002, 0.000, 0.000])
```

بدون مقیاس‌دهی، تقریباً همهٔ وزن به یک ورودی می‌رسد و `softmax` اشباع شده است؛ با مقیاس‌دهی، وزن‌ها بین ورودی‌ها پخش‌اند.

### چند سر (Multiple heads)

در عمل معمولاً چند سازوکار خودتوجهی به‌طور **موازی** اجرا می‌شوند که هر کدام را یک **سر** (Head) می‌نامند. با $H$ سر، سر $h$ام پارامترهای مخصوص خودش را دارد:

$$
\mathbf{V}_h = \boldsymbol{\beta}_{vh}\mathbf{1}^{\top} + \boldsymbol{\Omega}_{vh}\mathbf{X},
\qquad
\mathbf{Q}_h = \boldsymbol{\beta}_{qh}\mathbf{1}^{\top} + \boldsymbol{\Omega}_{qh}\mathbf{X},
\qquad
\mathbf{K}_h = \boldsymbol{\beta}_{kh}\mathbf{1}^{\top} + \boldsymbol{\Omega}_{kh}\mathbf{X}
$$

$$
\mathbf{Sa}_h[\mathbf{X}] = \mathbf{V}_h\; \text{Softmax}\!\left[\frac{\mathbf{K}_h^{\top} \mathbf{Q}_h}{\sqrt{D_q}}\right]
$$

اگر بُعد ورودی‌ها $D$ باشد، معمولاً هر سر پرسش‌ها، کلیدها و مقدارهایی با بُعد $D/H$ دارد؛ یعنی $\boldsymbol{\Omega}_{vh}, \boldsymbol{\Omega}_{qh}, \boldsymbol{\Omega}_{kh} \in \mathbb{R}^{(D/H) \times D}$ و $D_q = D/H$. خروجی‌های سرها **زیر هم** قرار می‌گیرند و یک تبدیل خطی دیگر، $\boldsymbol{\Omega}_c \in \mathbb{R}^{D \times D}$، آن‌ها را با هم ترکیب می‌کند:

$$
\mathbf{MhSa}[\mathbf{X}] = \boldsymbol{\Omega}_c \Big[\mathbf{Sa}_1[\mathbf{X}]^{\top}, \mathbf{Sa}_2[\mathbf{X}]^{\top}, \ldots, \mathbf{Sa}_H[\mathbf{X}]^{\top}\Big]^{\top}
$$

![udl_fig12_6](../img/udl_fig12_6.png)

شکل بالا خودتوجهی چندسر را با دو سر نشان می‌دهد:

- خودتوجهی به‌طور موازی در چند «سر» انجام می‌شود؛ اینجا دو سر در کادرهای فیروزه‌ای و نارنجی رسم شده‌اند.
- هر سر پرسش‌ها، کلیدها و مقدارهای **مخصوص خودش** را دارد که هر کدام $(D/H) \times N$ هستند، و ماتریس توجه $N \times N$ خودش را می‌سازد.
- خروجی هر سر، $\mathbf{Sa}_h[\mathbf{X}]$، یک ماتریس $(D/H) \times N$ است.
- خروجی سرها **به‌صورت عمودی** به هم الحاق می‌شوند و ماتریسی $D \times N$ می‌سازند؛ سپس تبدیل خطی $\boldsymbol{\Omega}_c$ آن‌ها را دوباره ترکیب می‌کند. خروجی نهایی هم‌اندازهٔ ورودی است.

چرا چند سر؟ هر سر می‌تواند به نوع متفاوتی از رابطه بین کلمه‌ها توجه کند؛ مثلاً یک سر ضمیر را به اسم مرجعش (it → restaurant) وصل کند و سری دیگر فعل را به فاعلش. در عمل به نظر می‌رسد چند سر برای خوب کار کردن ترنسفورمر لازم است؛ یکی از توضیح‌های پیشنهادی این است که چند سر خودتوجهی را در برابر مقداردهی اولیهٔ بد مقاوم‌تر می‌کند.

> [!note] هزینهٔ چند سر
> چون هر سر بُعد $D/H$ دارد، تعداد پارامترهای پرسش‌ها، کلیدها و مقدارها همان $3D^2 + 3D$ حالت تک‌سر است؛ فقط $D^2$ پارامتر $\boldsymbol{\Omega}_c$ اضافه می‌شود. برای مثال نقد رستوران با $D = 1024$ و $H = 16$ سر (هر سر با بُعد 64)، تعداد پارامترها $3{,}148{,}800 + 1{,}048{,}576 = 4{,}197{,}376$ است. در پیاده‌سازی هم ماتریس‌های $\boldsymbol{\Omega}_{qh}$ همهٔ سرها زیر هم یک ماتریس $D \times D$ می‌سازند؛ پس پرسش‌های همهٔ سرها با یک ضرب ماتریسی محاسبه و سپس بین سرها تقسیم می‌شوند.

کد زیر خودتوجهی چندسر را از روی فرمول بالا پیاده‌سازی می‌کند و نتیجه را با `nn.MultiheadAttention` مقایسه می‌کند. توجه کنید که PyTorch ورودی‌ها را در **سطرها** نگه می‌دارد و ماتریس‌های $\boldsymbol{\Omega}_q, \boldsymbol{\Omega}_k, \boldsymbol{\Omega}_v$ همهٔ سرها را زیر هم در `in_proj_weight` ذخیره می‌کند:

```python
import torch

torch.manual_seed(0)

D, N, H = 8, 5, 2
Dh = D // H                                   # size of each head: D/H
X = torch.randn(D, N)                         # columns are the inputs

mha = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H)
torch.nn.init.normal_(mha.in_proj_bias)       # make the biases beta_q, beta_k, beta_v nonzero
torch.nn.init.zeros_(mha.out_proj.bias)       # the formula above has no bias after Omega_c

Omega_q, Omega_k, Omega_v = mha.in_proj_weight.chunk(3)      # each D x D (rows = all heads)
beta_q, beta_k, beta_v = (b[:, None] for b in mha.in_proj_bias.chunk(3))
Omega_c = mha.out_proj.weight                                 # D x D

heads = []
for h in range(H):
    r = slice(h * Dh, (h + 1) * Dh)           # rows that belong to head h
    Q = beta_q[r] + Omega_q[r] @ X            # (D/H, N)
    K = beta_k[r] + Omega_k[r] @ X
    V = beta_v[r] + Omega_v[r] @ X
    heads.append(V @ torch.softmax(K.T @ Q / Dh ** 0.5, dim=0))   # Sa_h[X]  (D/H, N)

MhSa = Omega_c @ torch.cat(heads, dim=0)      # stack vertically -> (D, N), then recombine

# PyTorch stores the inputs as ROWS: shape (N, batch, D)
out, _ = mha(X.T[:, None], X.T[:, None], X.T[:, None])
print("output shape:", tuple(MhSa.shape))
print("same as nn.MultiheadAttention:", torch.allclose(MhSa, out[:, 0].T, atol=1e-6))
print("parameters:", sum(p.numel() for p in mha.parameters()) - D)   # minus the out_proj bias
```

```
output shape: (8, 5)
same as nn.MultiheadAttention: True
parameters: 280
```

برای $D = 8$ تعداد پارامترها $3 \cdot 8^2 + 3 \cdot 8 + 8^2 = 280$ است که با فرمول بالا می‌خواند.

---

## بخش پنجم: لایه‌های ترنسفورمر (Transformer layers)

خودتوجهی چندسر به‌تنهایی یک شبکه نیست؛ در ترنسفورمر، این سازوکار بخشی از یک واحد بزرگ‌تر به نام **لایهٔ ترنسفورمر** است. ورودی یک لایه ماتریس $\mathbf{X} \in \mathbb{R}^{D \times N}$ و خروجی آن ماتریسی هم‌اندازه است. لایه از چهار عمل پشت سر هم تشکیل شده است:

$$
\begin{aligned}
\mathbf{X} &\leftarrow \mathbf{X} + \mathbf{MhSa}[\mathbf{X}] \\
\mathbf{X} &\leftarrow \text{LayerNorm}[\mathbf{X}] \\
\mathbf{x}_n &\leftarrow \mathbf{x}_n + \mathbf{mlp}[\mathbf{x}_n] \qquad n = 1, \ldots, N \\
\mathbf{X} &\leftarrow \text{LayerNorm}[\mathbf{X}]
\end{aligned}
$$

![udl_fig12_7](../img/udl_fig12_7.png)

شکل بالا این چهار مرحله را نشان می‌دهد:

- **ورودی و خروجی:** ورودی ماتریسی $D \times N$ شامل نهفته‌های $D$بعدی $N$ نشانه است و خروجی ماتریسی با همان اندازه است. به همین دلیل می‌توان لایه‌ها را پشت سر هم چید.
- **بلوک خودتوجهی چندسر:** در این مرحله نهفته‌های کلمه‌ها با یکدیگر **تعامل** می‌کنند. این بلوک بخش پردازشی یک **بلوک باقیمانده** (فصل ۱۱) است؛ پس ورودی دوباره به خروجی آن اضافه می‌شود.
- **LayerNorm:** روی هر نهفته (هر ستون) **جداگانه** اعمال می‌شود.
- **بلوک باقیماندهٔ دوم:** یک شبکهٔ تمام‌متصل **یکسان** به‌طور جداگانه روی هر یک از $N$ ستون اعمال می‌شود و دوباره ورودی به خروجی آن اضافه می‌شود.
- **LayerNorm نهایی:** در پایان دوباره LayerNorm اعمال می‌شود.

دو بلوک نقش‌های مکمل دارند: خودتوجهی اطلاعات را **بین نشانه‌ها** جابه‌جا می‌کند ولی (جز `softmax`) تبدیل غیرخطی قوی ندارد؛ شبکهٔ تمام‌متصل هر نشانه را **به‌تنهایی** و به‌طور غیرخطی پردازش می‌کند ولی نشانه‌ها را با هم ترکیب نمی‌کند.

### نرمال‌سازی لایه‌ای (LayerNorm)

LayerNorm شبیه نرمال‌سازی دسته‌ای (جلسهٔ ۶) است، با این تفاوت که میانگین و واریانس را روی $D$ درایهٔ **یک نهفته** حساب می‌کند، نه روی نمونه‌های یک دسته:

$$
\text{LayerNorm}[\mathbf{x}_n] = \boldsymbol{\gamma} \odot \frac{\mathbf{x}_n - \mu_n}{\sigma_n} + \boldsymbol{\delta},
\qquad
\mu_n = \frac{1}{D}\sum_{d=1}^{D} x_{dn},
\qquad
\sigma_n^2 = \frac{1}{D}\sum_{d=1}^{D} (x_{dn} - \mu_n)^2
$$

که $\boldsymbol{\gamma}, \boldsymbol{\delta} \in \mathbb{R}^{D}$ پارامترهای یادگرفتنی مقیاس و جابه‌جایی‌اند. چون محاسبه به نمونه‌های دیگر دسته وابسته نیست، برای دنباله‌هایی با طول متغیر و دسته‌های کوچک مناسب است و در آموزش و استنتاج یکسان رفتار می‌کند.

### شبکهٔ تمام‌متصل موازی

شبکه $\mathbf{mlp}[\cdot]$ معمولاً یک شبکهٔ دولایه با تابع فعال‌سازی ReLU است که لایهٔ پنهانش بزرگ‌تر از ورودی است (معمولاً $4D$):

$$
\mathbf{mlp}[\mathbf{x}_n] = \boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2\, \text{ReLU}\big[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \mathbf{x}_n\big],
\qquad
\boldsymbol{\Omega}_1 \in \mathbb{R}^{4D \times D},\;
\boldsymbol{\Omega}_2 \in \mathbb{R}^{D \times 4D}
$$

همان پارامترها برای همهٔ $N$ ستون به کار می‌روند؛ پس تعداد پارامترها به طول متن بستگی ندارد.

> [!note] تعداد پارامترهای یک لایه
> برای $D = 1024$ و $H = 16$ (با بایاس خروجی خودتوجهی، مانند PyTorch): خودتوجهی چندسر $4{,}198{,}400$، شبکهٔ تمام‌متصل $8{,}393{,}728$ و دو LayerNorm $4{,}096$ پارامتر؛ در مجموع $12{,}596{,}224$ پارامتر، یعنی حدود ۱۲٫۶ میلیون. حدود دوسوم پارامترها در شبکهٔ تمام‌متصل است.

> [!note] ترتیب LayerNorm
> ترتیب بالا (LayerNorm **پس از** جمع باقیمانده) همان طرح مقالهٔ اصلی ترنسفورمر است. در بسیاری از مدل‌های جدید LayerNorm **پیش از** خودتوجهی و شبکهٔ تمام‌متصل، درون شاخهٔ باقیمانده، قرار می‌گیرد (Pre-LN) که آموزش شبکه‌های عمیق را پایدارتر می‌کند؛ در PyTorch با `norm_first=True` انتخاب می‌شود.

کد زیر یک لایهٔ ترنسفورمر را از روی چهار مرحلهٔ بالا می‌سازد و با `nn.TransformerEncoderLayer` مقایسه می‌کند:

```python
import torch

torch.manual_seed(0)

D, N, H = 8, 5, 2
X = torch.randn(D, N)                          # columns are the word embeddings

layer = torch.nn.TransformerEncoderLayer(d_model=D, nhead=H, dim_feedforward=4 * D,
                                         dropout=0.0, activation="relu")
layer.eval()
mha, ln1, ln2 = layer.self_attn, layer.norm1, layer.norm2
mlp = torch.nn.Sequential(layer.linear1, torch.nn.ReLU(), layer.linear2)

def layer_norm(X, ln):                         # normalize each column separately
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.var(dim=0, unbiased=False, keepdim=True).add(ln.eps).sqrt()
    return ln.weight[:, None] * (X - mu) / sigma + ln.bias[:, None]   # gamma, delta

with torch.no_grad():
    A, _ = mha(X.T, X.T, X.T)                  # PyTorch: rows are the tokens
    Y = layer_norm(X + A.T, ln1)               # 1) residual block with MhSa, 2) LayerNorm
    Y = layer_norm(Y + mlp(Y.T).T, ln2)        # 3) residual block with the same MLP per column, 4) LayerNorm
    out = layer(X.T).T

print("output shape:", tuple(Y.shape))
print("same as nn.TransformerEncoderLayer:", torch.allclose(Y, out, atol=1e-5))
```

```
output shape: (8, 5)
same as nn.TransformerEncoderLayer: True
```

یک ترنسفورمر از چندین لایهٔ این‌چنینی که پشت سر هم قرار گرفته‌اند ساخته می‌شود؛ مثلاً BERT پایه ۱۲ و GPT-3 ۹۶ لایه دارد.

---

## بخش ششم: ترنسفورمر برای پردازش زبان طبیعی

اکنون همهٔ اجزای لازم را داریم و می‌توانیم ببینیم یک ترنسفورمر چگونه یک متن را پردازش می‌کند. مسیر کامل چنین است:

$$
\text{متن}
\;\xrightarrow{\text{نشانه‌گذاری}}\;
\mathbf{T}
\;\xrightarrow{\;\boldsymbol{\Omega}_e\;}\;
\mathbf{X}
\;\xrightarrow{\;+\,\boldsymbol{\Pi}\;}\;
\underbrace{\text{لایهٔ ترنسفورمر} \to \cdots \to \text{لایهٔ ترنسفورمر}}_{K\ \text{لایه}}
\;\longrightarrow\;
\text{خروجی}
$$

1. **نشانه‌گذاری:** متن با یک نشانه‌گذار زیرکلمه‌ای (مانند BPE) به $N$ نشانه شکسته می‌شود و هر نشانه یک شماره در واژه‌نامه می‌گیرد (جلسهٔ ۱۰، بخش دوم).
2. **نهفته‌سازی:** با $\mathbf{X} = \boldsymbol{\Omega}_e\mathbf{T}$ هر نشانه به یک بردار $D$بعدی تبدیل می‌شود؛ $\boldsymbol{\Omega}_e$ همراه با بقیهٔ شبکه آموزش می‌بیند.
3. **رمزگذاری موقعیت:** اطلاعات جایگاه نشانه‌ها به $\mathbf{X}$ افزوده می‌شود (بخش چهارم).
4. **لایه‌های ترنسفورمر:** ماتریس $\mathbf{X}$ از $K$ لایهٔ ترنسفورمر (بخش پنجم) می‌گذرد. اندازهٔ $D \times N$ در همهٔ لایه‌ها ثابت می‌ماند.

### یک مثال: دنبال کردن ابعاد

متن «an aardvark ate an ant» را با $|\mathcal{V}| = 30{,}000$، $D = 1024$، $H = 16$ و $K = 12$ لایه در نظر بگیرید و فرض کنید هر کلمه یک نشانه است ($N = 5$):

| مرحله | ماتریس | ابعاد | تعداد پارامترها |
| --- | --- | --- | --- |
| نشانه‌گذاری | $\mathbf{T}$ | $30{,}000 \times 5$ | — |
| نهفته‌سازی | $\boldsymbol{\Omega}_e$ | $1024 \times 30{,}000$ | $30{,}720{,}000$ |
| ورودی شبکه | $\mathbf{X}$ | $1024 \times 5$ | — |
| ۱۲ لایهٔ ترنسفورمر | خروجی هر لایه | $1024 \times 5$ | $12 \times 12{,}596{,}224 = 151{,}154{,}688$ |

در مجموع حدود ۱۸۲ میلیون پارامتر به دست می‌آید. دو نکته از این جدول:

- تعداد پارامترها به طول متن $N$ بستگی **ندارد**؛ همان شبکه متن ۵ نشانه‌ای و ۵۰۰ نشانه‌ای را پردازش می‌کند. فقط حجم محاسبات با $N^2$ زیاد می‌شود.
- بخش بزرگی از پارامترها (اینجا حدود یک‌ششم) در ماتریس نهفته‌سازی است.

### نهفته‌های وابسته به بافت

در شکل 12.9 (جلسهٔ ۱۰) دیدیم که دو کلمهٔ «an» در ورودی $\mathbf{X}$ بردار **یکسانی** دارند. پس از گذر از لایه‌های ترنسفورمر، هر ستون خروجی ترکیبی از اطلاعات همهٔ نشانه‌هاست و به **بافت** (Context) کلمه و جای آن بستگی دارد؛ به همین دلیل خروجی‌ها را **نهفته‌های وابسته به بافت** (Contextual embeddings) می‌نامند. کلمهٔ «شیر» در «شیر را در یخچال گذاشتم» و «شیر در جنگل غرید» در ورودی یک بردار دارد، اما در خروجی دو بردار متفاوت.

کد زیر این را برای مثال بالا (با یک مدل کوچک و آموزش‌ندیده) نشان می‌دهد. نتیجهٔ سطر دوم، همان هم‌وردایی بخش چهارم است: بدون رمزگذاری موقعیت، دو «an» که ورودی یکسان دارند، خروجی یکسان هم می‌گیرند:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

vocab = ["a", "aardvark", "an", "ant", "ate"]          # a tiny vocabulary
D, H, K = 8, 2, 2                                       # embedding size, heads, layers

embed = nn.Embedding(len(vocab), D)                     # Omega_e (stored transposed)
layer = nn.TransformerEncoderLayer(d_model=D, nhead=H, dim_feedforward=4 * D, dropout=0.0)
model = nn.TransformerEncoder(layer, num_layers=K, enable_nested_tensor=False).eval()

text = "an aardvark ate an ant".split()
tokens = torch.tensor([vocab.index(w) for w in text])   # token ids, N = 5
N = len(tokens)

pos = torch.arange(N).float()                           # sinusoidal positional encoding (N, D)
freq = 1.0 / 10000 ** (torch.arange(0, D, 2).float() / D)
Pi = torch.zeros(N, D)
Pi[:, 0::2] = torch.sin(pos[:, None] * freq)
Pi[:, 1::2] = torch.cos(pos[:, None] * freq)

with torch.no_grad():
    X = embed(tokens)                                   # (N, D): row n = embedding of token n
    out_no_pos = model(X[:, None])[:, 0]                # K transformer layers, no position
    out = model((X + Pi)[:, None])[:, 0]                # K transformer layers, with position

# tokens 0 and 3 are both "an"
print("input:            same vector for both 'an':", torch.equal(X[0], X[3]))
print("output, no Pi:    same vector for both 'an':", torch.allclose(out_no_pos[0], out_no_pos[3], atol=1e-6))
print("output, with Pi:  same vector for both 'an':", torch.allclose(out[0], out[3], atol=1e-6))
```

```
input:            same vector for both 'an': True
output, no Pi:    same vector for both 'an': True
output, with Pi:  same vector for both 'an': False
```

### سه نوع مدل ترنسفورمر

خروجی لایه‌ها بسته به کاربرد به سه شکل استفاده می‌شود و سه خانوادهٔ اصلی مدل را می‌سازد:

| نوع | کار اصلی | نمونهٔ کاربرد | مدل مشهور |
| --- | --- | --- | --- |
| **رمزگذار** (Encoder) | تبدیل متن به نمایشی که برای کارهای گوناگون مفید است | تحلیل احساس، طبقه‌بندی متن، برچسب‌گذاری کلمه‌ها | BERT |
| **رمزگشا** (Decoder) | پیش‌بینی نشانهٔ بعدی برای ادامه دادن متن | تولید متن، دستیارهای گفت‌وگو | GPT |
| **رمزگذار–رمزگشا** (Encoder-decoder) | تبدیل یک دنباله به دنبالهٔ دیگر | ترجمهٔ ماشینی، خلاصه‌سازی | ترنسفورمر اصلی |

این سه نوع با دسته‌های جلسهٔ ۸ (بخش دوم) متناظرند: رمزگذار برای کارهای چندبه‌یک و چندبه‌چند هم‌زمان، رمزگشا برای تولید دنباله (مانند یک‌به‌چند) و رمزگذار–رمزگشا برای چندبه‌چند با تأخیر. در جلسهٔ بعد هر کدام را جداگانه بررسی می‌کنیم.

---
