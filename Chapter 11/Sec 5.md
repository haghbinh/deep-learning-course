# فصل ۱۱: شبکه‌های باقیمانده (Residual Networks)
## جلسه 5

---
## بخش اول: چرا به بلوک‌های باقیمانده نیاز داریم؟

در فصل قبل **شبکه‌های کانولوشنی** را ساختیم: لایه‌ها (کانولوشن ← تابع فعال‌سازی ← ادغام) را پشت سر هم قرار می‌دهیم و هر لایه خروجی لایهٔ قبلی را پردازش می‌کند.

یک سؤال طبیعی پیش می‌آید: **اگر چند لایه خوب کار می‌کنند، آیا لایه‌های بیشتر بهتر کار می‌کنند؟**

این ایده منطقی به نظر می‌رسد:

- شبکه‌های عمیق‌تر می‌توانند **توابع پیچیده‌تری** را نمایش دهند.
- هر لایهٔ اضافه می‌تواند **ویژگی‌های انتزاعی‌تری** بسازد (لبه‌ها ← بافت‌ها ← اجزا ← اشیاء).
- از نظر تئوری، شبکهٔ عمیق‌تر **دست‌کم به خوبی** شبکهٔ کم‌عمق‌تر عمل می‌کند (دلیلش را در ادامه می‌بینیم).

در عمل، صرفاً اضافه کردن لایه‌های بیشتر از یک عمق مشخص به بعد **جواب نمی‌دهد**. این فصل دربارهٔ دلیل این اتفاق و روش رفع آن با اتصال‌های باقیمانده است.

---

### پردازش ترتیبی: شبکهٔ «ساده»

تا اینجا همهٔ شبکه‌هایی که دیده‌ایم داده را به‌صورت **ترتیبی** پردازش می‌کنند. با ورودی $\mathbf{x}$ و لایه‌های $$\mathbf{f}_1, \mathbf{f}_2, \dots$$ که هر کدام پارامترهای $\boldsymbol{\phi}_k$ دارند:

$$  
\begin{aligned}  
\mathbf{h}_1 &= \mathbf{f}_1[\mathbf{x}, \boldsymbol{\phi}_1] \\  
\mathbf{h}_2 &= \mathbf{f}_2[\mathbf{h}_1, \boldsymbol{\phi}_2] \\  
\mathbf{h}_3 &= \mathbf{f}_3[\mathbf{h}_2, \boldsymbol{\phi}_3] \\  
\mathbf{y} &= \mathbf{f}_4[\mathbf{h}_3, \boldsymbol{\phi}_4]  
\end{aligned}  
$$

به‌طور معادل، کل شبکه یک **ترکیب** (Composition) بزرگ از توابع است:

$$  
\mathbf{y} = \mathbf{f}_4\Big[\mathbf{f}_3\big[\mathbf{f}_2[\mathbf{f}_1[\mathbf{x}, \boldsymbol{\phi}_1], \boldsymbol{\phi}_2], \boldsymbol{\phi}_3\big], \boldsymbol{\phi}_4\Big]  
$$

> [!note] نکتهٔ کلیدی  
> در یک شبکهٔ ساده، اطلاعات **فقط یک مسیر** از ورودی تا خروجی دارد: باید از **تک‌تک** لایه‌ها عبور کند. هر لایه ممکن است اطلاعات را مخدوش کند یا از دست بدهد، و هیچ راهی برای پرش از روی لایه‌ها وجود ندارد.

---

### آزمایش شگفت‌انگیز: عمیق‌تر، بدتر است

در سال ۲۰۱۶، He و همکارانش شبکه‌های کانولوشنی ساده با عمق‌های مختلف را روی مجموعه‌دادهٔ تصویری CIFAR-10 آموزش دادند و یک شبکهٔ **۲۰ لایه** را با یک شبکهٔ **۵۶ لایه** مقایسه کردند.

| شبکه       | ۵۶ لایه | ۲۰ لایه   |
| ---------- | ------- | --------- |
| خطای آموزش | کمتر    | **بیشتر** |
| خطای آزمون | کمتر    | **بیشتر** |

شبکهٔ عمیق‌تر روی **هر دو** دادهٔ آموزش و آزمون بدتر بود.
![Pasted image 20261010131730](../img/Pasted%20image%2020261010131730.png)

> [!warning] این بیش‌برازش (Overfitting) نیست  
> بیش‌برازش یعنی مدل دادهٔ آموزش را _بیش از حد خوب_ یاد می‌گیرد و تعمیم ضعیفی دارد؛ پس خطای آموزش کم و خطای آزمون زیاد است.  
> اینجا شبکهٔ عمیق‌تر **خطای آموزش بیشتری** هم دارد؛ یعنی حتی نمی‌تواند دادهٔ آموزش را به خوبی شبکهٔ کم‌عمق برازش کند. این نشان‌دهندهٔ یک **مشکل بهینه‌سازی** است: فرایند آموزش نمی‌تواند پارامترهای خوب را پیدا کند.

---

### آزمایش فکری: عمیق‌تر هرگز نباید بدتر باشد

شبکهٔ ۲۰ لایهٔ آموزش‌دیده را در نظر بگیرید و ۳۶ لایهٔ دیگر به آن اضافه کنید که هر کدام فقط **تابع همانی** (Identity) را محاسبه می‌کنند:

$$  
\mathbf{f}_k[\mathbf{h}] = \mathbf{h}  
$$

این لایه‌های اضافه ورودی خود را بدون تغییر عبور می‌دهند؛ پس شبکهٔ ۵۶ لایه **دقیقاً همان تابع** شبکهٔ ۲۰ لایه را محاسبه می‌کند و همان خطا را دارد.

بنابراین یک جواب ۵۶ لایه وجود دارد که **دست‌کم به خوبی** جواب ۲۰ لایه است. فقط گرادیان کاهشی **نمی‌تواند آن را پیدا کند**.

> [!question] چرا؟  
> برای دنباله‌ای از لایه‌های غیرخطی (کانولوشن + ReLU)، یادگرفتن حتی نگاشت «هیچ کاری نکن» (همانی) به‌طرز شگفت‌آوری سخت است. بهینه‌ساز باید همهٔ وزن‌ها را بسیار دقیق تنظیم کند تا به $$\mathbf{f}_k[\mathbf{h}] \approx \mathbf{h}$$ برسد.

---

### چه مشکلی در گرادیان‌ها پیش می‌آید؟

آموزش با پس‌انتشار (Backpropagation) انجام می‌شود. طبق قاعدهٔ زنجیره‌ای، گرادیان تابع زیان $L$ نسبت به خروجی یک لایهٔ ابتدایی، **حاصل‌ضرب** مشتق‌ها در همهٔ لایه‌های بعدی است:

$$  
\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_3}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{h}_3} \cdot \frac{\partial L}{\partial \mathbf{y}}  
$$

در شبکه‌ای با $K$ لایه، این حاصل‌ضرب حدوداً $K$ جمله دارد. این موضوع دو مشکل ایجاد می‌کند.

#### مشکل الف: گرادیان‌های محوشونده و انفجاری

ضرب کردن تعداد زیادی عدد در هم ناپایدار است. به‌عنوان یک تشبیه اسکالر:

$$  
0.9^{50} \approx 0.005 \qquad \qquad 1.1^{50} \approx 117  
$$

- اگر ضرایب عمدتاً **کوچک‌تر از ۱** باشند، گرادیان **محو** می‌شود و لایه‌های ابتدایی تقریباً چیزی یاد نمی‌گیرند.
- اگر ضرایب عمدتاً **بزرگ‌تر از ۱** باشند، گرادیان **منفجر** می‌شود و آموزش ناپایدار می‌شود.

قبلاً راه‌حل‌های نسبی را دیده‌ایم: **مقداردهی اولیهٔ دقیق** (مثل مقداردهی He) و **نرمال‌سازی دسته‌ای**. این‌ها کمک می‌کنند، اما مشکل شبکه‌های عمیق را کاملاً حل نمی‌کنند.

#### مشکل ب: گرادیان‌های خردشده (Shattered Gradients)

این مشکل ظریف‌تر است و انگیزهٔ اصلی این فصل  است. 

در یک شبکهٔ **کم‌عمق**، اگر ورودی $\mathbf{x}$ را کمی تغییر دهیم، گرادیان هم کمی تغییر می‌کند. ورودی‌های نزدیک به هم **گرادیان‌های مشابهی** دارند؛ پس سطح زیان نسبتاً هموار است و گرادیان کاهشی می‌تواند آن را دنبال کند.

در یک شبکهٔ **عمیق ساده**، گرادیان حتی با تغییرات بسیار کوچک ورودی به‌شدت **تغییر می‌کند**. گرادیان‌ها در نقاط نزدیک به هم تقریباً **بی‌ارتباط** به نظر می‌رسند، مثل نویز تصادفی. به این پدیده **گرادیان‌های خردشده** می‌گویند (Balduzzi و همکاران، ۲۰۱۷).

> [!example] شهود  
> تصور کنید در مه از کوهی پایین می‌آیید و فقط شیب زیر پایتان را حس می‌کنید.
> 
> - **شبکهٔ کم‌عمق:** شیب به‌نرمی تغییر می‌کند، پس یک قدم رو به پایین واقعاً شما را پایین‌تر می‌برد.
> - **شبکهٔ عمیق ساده:** زمین مثل شیشهٔ خردشده است و شیب هر چند سانتی‌متر به یک جهت تصادفی دیگر اشاره می‌کند. یک قدم «رو به پایین» تقریباً هیچ اطلاعاتی دربارهٔ مسیر بعدی به شما نمی‌دهد.

چرا عمق باعث این پدیده می‌شود؟ هر لایه می‌تواند تابع را کمی خم کند و تا بزند (به یاد بیاورید که شبکه‌های ReLU نواحی خطی تکه‌ای می‌سازند). ترکیب لایه‌های زیاد تعداد این تاها را چند برابر می‌کند، پس تابع و گرادیان آن با تغییر ورودی بسیار سریع تغییر می‌کنند.

> [!note] پیامد  
> گرادیان کاهشی فرض می‌کند گرادیان در نقطهٔ فعلی راهنمای خوبی برای یک قدم کوچک است. با گرادیان‌های خردشده این فرض از بین می‌رود، پس آموزش کند می‌شود یا گیر می‌کند. این توضیح می‌دهد که چرا شبکهٔ ۵۶ لایه **خطای آموزش بیشتری** داشت.

---

### ایده‌ای که مشکل را حل می‌کند: افزودن یک میان‌بر

چه می‌شود اگر هر بلوک فقط **تغییر** ورودی‌اش را یاد بگیرد، به‌جای اینکه یک خروجی کاملاً جدید تولید کند؟

به‌جای:

$$  
\mathbf{h}_{k} = \mathbf{f}_k[\mathbf{h}_{k-1}]  
$$

از این استفاده می‌کنیم:

$$  
\boxed{\mathbf{h}_{k} = \mathbf{h}_{k-1} + \mathbf{f}_k[\mathbf{h}_{k-1}]}  
$$

ورودی $\mathbf{h}_{k-1}$ از طریق یک **اتصال پرشی** (Skip connection) که **اتصال باقیمانده** (Residual connection) هم نامیده می‌شود، دوباره به خروجی بلوک **اضافه می‌شود**. حالا بلوک $\mathbf{f}_k$ فقط **باقیمانده** را یاد می‌گیرد: یعنی تفاوت بین خروجی مطلوب و ورودی.

به‌طور خلاصه، چرا این کار کمک می‌کند:

۱. **یادگیری همانی آسان است.** برای اینکه بلوک «کاری نکند»، کافی است $$\mathbf{f}_k \approx \mathbf{0}$$ باشد (مثلاً با وزن‌های کوچک). دیگر شبکه‌های عمیق‌تر در وضعیت بدتری نیستند و مشکل آزمایش فکری بالا حل می‌شود.

۲. **گرادیان‌ها یک بزرگراه پیدا می‌کنند.** با مشتق‌گیری داریم:

$$  
\frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \mathbf{I} + \frac{\partial \mathbf{f}_k}{\partial \mathbf{h}_{k-1}}  
$$

جملهٔ همانی $\mathbf{I}$ یعنی گرادیان می‌تواند مستقیماً به عقب برگردد، حتی اگر $$\frac{\partial \mathbf{f}_k}{\partial \mathbf{h}_{k-1}}$$ کوچک باشد. این مشکل گرادیان‌های محوشونده را تا حد زیادی کاهش می‌دهد.

۳. **گرادیان‌های هموارتر.** چون اطلاعات می‌تواند از روی لایه‌ها عبور کند، شبکه تا حدی مثل ترکیبی از شبکه‌های کم‌عمق‌تر رفتار می‌کند و گرادیان‌هایش بسیار کمتر خرد می‌شوند.

> [!success] نتیجه  
> با اتصال‌های باقیمانده، He و همکارانش شبکه‌هایی با **بیش از ۱۰۰ لایه** (و در آزمایش‌ها بیش از ۱۰۰۰ لایه) را آموزش دادند و شبکه‌های عمیق‌تر **بهتر** شدند، نه بدتر. شبکه‌های ResNet در سال ۲۰۱۵ برندهٔ مسابقهٔ ImageNet شدند.


> [!tip] خلاصه در یک جمله  
> مشکل شبکه‌های عمیق ساده **بهینه‌سازی** است، نه ناتوانی در نمایش جواب‌های خوب؛ اتصال‌های باقیمانده با این کار که هر بلوک فقط یک اصلاح کوچک روی ورودی‌اش یاد بگیرد، آموزش شبکه‌های عمیق را ممکن می‌کنند.


```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

DEPTH = 50   # number of layers / blocks
WIDTH = 64   # hidden units per layer

# ---------------------------------------------------------------
# Two networks with the same layers. Only difference: the "+ h"
# ---------------------------------------------------------------
class PlainNet(nn.Module):
    # h_k = f_k(h_{k-1})
    def __init__(self, depth=DEPTH, width=WIDTH):
        super().__init__()
        self.inp = nn.Linear(1, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, 1)
        for layer in self.layers:                      # He initialization
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.out(h)


class ResNet(nn.Module):
    # h_k = h_{k-1} + f_k(h_{k-1})
    def __init__(self, depth=DEPTH, width=WIDTH):
        super().__init__()
        self.inp = nn.Linear(1, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, 1)
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            layer.weight.data *= 0.1   # start each block close to identity (f_k ≈ 0)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = h + torch.relu(layer(h))   # <-- the skip connection
        return self.out(h)


# ---------------------------------------------------------------
# Experiment 1: shattered gradients
# Compute dy/dx for 400 closely spaced inputs x in [-2, 2].
# ---------------------------------------------------------------
x = torch.linspace(-2, 2, 400).unsqueeze(1).requires_grad_(True)

def input_gradient(net, x):
    y = net(x).sum()
    (g,) = torch.autograd.grad(y, x)
    return g.squeeze().detach()

g_plain = input_gradient(PlainNet(), x)
g_res = input_gradient(ResNet(), x)

# ---------------------------------------------------------------
# Experiment 2: how smooth is the gradient as depth grows?
# Correlation between gradients at neighbouring inputs:
#   ~1  -> smooth (useful for gradient descent)
#   ~0  -> noise-like ("shattered")
# ---------------------------------------------------------------
def neighbour_corr(g):
    return torch.corrcoef(torch.stack([g[:-1], g[1:]]))[0, 1].item()

depths = [1, 2, 5, 10, 25, 50, 100]
N_NETS = 10   # average over several random networks per depth
corr_plain, corr_res = [], []
for d in depths:
    corr_plain.append(sum(neighbour_corr(input_gradient(PlainNet(depth=d), x))
                          for _ in range(N_NETS)) / N_NETS)
    corr_res.append(sum(neighbour_corr(input_gradient(ResNet(depth=d), x))
                        for _ in range(N_NETS)) / N_NETS)
    print(f"depth {d:3d}   plain: {corr_plain[-1]:.3f}   residual: {corr_res[-1]:.3f}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
xs = x.detach().squeeze()

ax[0].plot(xs, g_plain)
ax[0].set_title(f"Plain net ({DEPTH} layers): dy/dx")
ax[1].plot(xs, g_res, color="C1")
ax[1].set_title(f"Residual net ({DEPTH} blocks): dy/dx")
for a in ax[:2]:
    a.set_xlabel("input x")

ax[2].plot(depths, corr_plain, "o-", label="plain")
ax[2].plot(depths, corr_res, "o-", label="residual")
ax[2].set_xscale("log")
ax[2].set_xlabel("depth")
ax[2].set_ylabel("neighbour gradient correlation")
ax[2].set_title("Gradient smoothness vs depth")
ax[2].legend()

plt.tight_layout()
plt.show()
```

```
depth   1   plain: 0.991   residual: 0.990
depth   2   plain: 0.989   residual: 0.988
depth   5   plain: 0.976   residual: 0.991
depth  10   plain: 0.964   residual: 0.993
depth  25   plain: 0.842   residual: 0.997
depth  50   plain: 0.695   residual: 0.998
depth 100   plain: 0.492   residual: 1.000
```

![Pasted image 20261010130824](../img/Pasted%20image%2020261010130824.png)

---
