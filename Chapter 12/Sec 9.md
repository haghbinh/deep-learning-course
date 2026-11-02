# شبکه‌های عصبی بازگشتی ( ادامه...)
## جلسه 9
---
## بخش پنجم: پس‌انتشار در طول زمان (BPTT)

در بخش سوم دیدیم که RNN بازشده یک **شبکهٔ پیش‌خور عمیق** است که در همهٔ لایه‌هایش از **وزن‌های مشترک** استفاده می‌کند. پس برای محاسبهٔ گرادیان به الگوریتم جدیدی نیاز نداریم: همان **پس‌انتشار** را روی شبکهٔ بازشده اجرا می‌کنیم. به این کار **پس‌انتشار در طول زمان** (Backpropagation Through Time, BPTT) می‌گویند.

الگوریتم دو مرحله دارد:

1. **گذر رو به جلو:** دنباله را از $1=t$ تا $T=t$ می‌خوانیم، حالت‌های پنهان $\mathbf{h}_1, \dots, \mathbf{h}_T$ و خروجی‌ها را محاسبه و **ذخیره** می‌کنیم و زیان را به دست می‌آوریم.
2. **گذر رو به عقب:** گرادیان را از $T=t$ به سمت $1=t$ برمی‌گردانیم.

---

### گرادیان نسبت به حالت‌های پنهان

برای راحتی، ورودی تابع فعال‌سازی را $\mathbf{z}_t$ می‌نامیم:

$$
\mathbf{z}_t = \boldsymbol{\beta}_h + \boldsymbol{\Omega}_{xh}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hh}\,\mathbf{h}_{t-1}, \qquad \mathbf{h}_t = \mathbf{a}[\mathbf{z}_t]
$$

زیان کل، جمع زیان گام‌هاست: $$L = \sum_{t} L_t$$ که در آن $$L_t = \ell\big[\hat{\mathbf{y}}_t,\ \mathbf{y}_t\big]$$ (در مسئلهٔ چندبه‌یک فقط $L_T$ وجود دارد).

نکتهٔ کلیدی این است که حالت پنهان $\mathbf{h}_t$ از **دو مسیر** روی زیان اثر می‌گذارد:

- **مستقیم:** از طریق خروجی همان گام، $\hat{\mathbf{y}}_t$ (پیکان رو به بالا در شکل باز شدن شبکه)،
- **غیرمستقیم:** از طریق حالت پنهان گام بعد، $\mathbf{h}_{t+1}$، و در نتیجه همهٔ گام‌های آینده (پیکان افقی).

بنابراین گرادیان به‌صورت بازگشتی، از آخر به اول، محاسبه می‌شود:

$$
\frac{\partial L}{\partial \mathbf{h}_t} = \underbrace{\frac{\partial L_t}{\partial \mathbf{h}_t}}_{\text{مسیر مستقیم}} + \underbrace{\mathbf{J}_{t+1}^{\top}\, \frac{\partial L}{\partial \mathbf{h}_{t+1}}}_{\text{مسیر از آینده}}
$$

که در آن $\mathbf{J}_{t+1}$ **ژاکوبی** (Jacobian) حالت پنهان نسبت به حالت قبلی است:

$$
\mathbf{J}_{t+1} = \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \text{diag}\big(\mathbf{a}'[\mathbf{z}_{t+1}]\big)\,\boldsymbol{\Omega}_{hh}
$$

در گام آخر جملهٔ «مسیر از آینده» وجود ندارد و بازگشت از $$\frac{\partial L}{\partial \mathbf{h}_T} = \frac{\partial L_T}{\partial \mathbf{h}_T}$$ شروع می‌شود.

---

### گرادیان نسبت به وزن‌های مشترک

چون یک ماتریس $\boldsymbol{\Omega}_{hh}$ در **همهٔ** گام‌ها به کار رفته است، گرادیان آن **جمع سهم همهٔ گام‌ها** است:

$$
\frac{\partial L}{\partial \boldsymbol{\Omega}_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{z}_t}\, \mathbf{h}_{t-1}^{\top},
\qquad
\frac{\partial L}{\partial \mathbf{z}_t} = \mathbf{a}'[\mathbf{z}_t] \odot \frac{\partial L}{\partial \mathbf{h}_t}
$$

به همین ترتیب:

$$
\frac{\partial L}{\partial \boldsymbol{\Omega}_{xh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{z}_t}\, \mathbf{x}_t^{\top},
\qquad
\frac{\partial L}{\partial \boldsymbol{\beta}_h} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{z}_t}
$$

> [!note] نکتهٔ کلیدی
> جمع شدن گرادیان روی گام‌های زمانی، نتیجهٔ مستقیم **اشتراک پارامتر** است. در CNN هم (فصل ۱۰) گرادیان یک هسته، جمع سهم آن در **همهٔ موقعیت‌های مکانی** بود؛ اینجا جمع سهم در **همهٔ گام‌های زمانی** است.

---

### حاصل‌ضرب ژاکوبی‌ها

اگر بازگشت بالا را باز کنیم، اثر حالت پنهان یک گام دور، $\mathbf{h}_k$، روی حالت پنهان گام $T$ برابر است با **حاصل‌ضرب** ژاکوبی‌های همهٔ گام‌های میان آن‌ها:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k} = \mathbf{J}_T\, \mathbf{J}_{T-1} \cdots \mathbf{J}_{k+1} = \prod_{t=k+1}^{T} \text{diag}\big(\mathbf{a}'[\mathbf{z}_t]\big)\,\boldsymbol{\Omega}_{hh}
$$

هرچه فاصلهٔ $T-k$ بیشتر باشد، تعداد جمله‌های این حاصل‌ضرب بیشتر است. این همان ساختاری است که در فصل ۱۱ برای شبکه‌های عمیق دیدیم؛ با این تفاوت که اینجا **همان** ماتریس $\boldsymbol{\Omega}_{hh}$ بارها در خودش ضرب می‌شود.

> [!example] مثال عددی
> در مثال اسکالر بخش سوم  $0.5=\omega_{hh}$  و $0.364=h_2$ و $0.180=h_3$، با توجه به اینکه $$\tanh'(z) = 1 - \tanh^2(z)$$:
>
> $$
> \frac{\partial h_3}{\partial h_1} = \underbrace{(1 - 0.364^2) \times 0.5}_{J_2 \,\approx\, 0.434} \times \underbrace{(1 - 0.180^2) \times 0.5}_{J_3 \,\approx\, 0.484} \approx 0.21
> $$
>
> یعنی پس از تنها دو گام، گرادیان به حدود یک‌پنجم کاهش یافته است. در بخش بعد می‌بینیم که این حاصل‌ضرب برای دنباله‌های بلند چه مشکلی ایجاد می‌کند.

---

### پس‌انتشار بریده‌شده در طول زمان (Truncated BPTT)

در BPTT کامل، همهٔ حالت‌های پنهان $\mathbf{h}_1, \dots, \mathbf{h}_T$ باید در حافظه نگه داشته شوند و هر به‌روزرسانی وزن‌ها به یک گذر کامل روی کل دنباله نیاز دارد. برای دنباله‌های بسیار بلند (مثلاً چند ده هزار گام) این کار پرهزینه است.

در **BPTT بریده‌شده**، دنبالهٔ بلند را به قطعه‌های کوتاه‌تری با طول $k$ تقسیم می‌کنیم:

- در **گذر رو به جلو**، حالت پنهان از یک قطعه به قطعهٔ بعد **منتقل می‌شود**؛ پس مدل همچنان کل گذشته را «می‌بیند».
- در **گذر رو به عقب**، گرادیان فقط تا **ابتدای همان قطعه** برمی‌گردد و پس از هر قطعه وزن‌ها به‌روز می‌شوند.

در PyTorch این کار با جدا کردن حالت پنهان از گراف محاسباتی در ابتدای هر قطعه انجام می‌شود (متد `detach` روی حالت پنهان).

> [!warning] بهای بریدن
> با BPTT بریده‌شده، مدل نمی‌تواند وابستگی‌هایی را **یاد بگیرد** که طولشان از $k$ گام بیشتر است، چون گرادیانی از آن فاصله به عقب نمی‌رسد؛ هرچند در گذر رو به جلو اطلاعات قدیمی‌تر هنوز در حالت پنهان حضور دارد.

---
## بخش ششم: گرادیان‌های محوشونده و انفجاری در RNN

### ریشهٔ مشکل: ضرب مکرر یک ماتریس

در بخش پنجم دیدیم که اثر حالت پنهان گام $k$ روی حالت پنهان گام $T$، حاصل‌ضرب $T-k$ ژاکوبی است:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k} = \prod_{t=k+1}^{T} \text{diag}\big(\mathbf{a}'[\mathbf{z}_t]\big)\,\boldsymbol{\Omega}_{hh}
$$

در فصل ۱۱ با همین ساختار برای شبکه‌های عمیق آشنا شدیم: ضرب تعداد زیادی جمله در هم ناپایدار است. با همان تشبیه اسکالر:

$$
0.9^{50} \approx 0.005 \qquad \qquad 1.1^{50} \approx 117
$$

اما در RNN وضعیت حادتر است: در یک شبکهٔ پیش‌خور عمیق هر لایه ماتریس وزن **خودش** را دارد، ولی در RNN **همان** ماتریس $\boldsymbol{\Omega}_{hh}$ در هر گام دوباره ضرب می‌شود. بنابراین اندازهٔ این حاصل‌ضرب عمدتاً به **بزرگ‌ترین مقدار تکین** (Largest singular value) ماتریس $\boldsymbol{\Omega}_{hh}$، یعنی $\sigma_{\max}$، بستگی دارد. از آنجا که مشتق $\tanh$ هرگز از ۱ بیشتر نیست:

$$
\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k} \right\| \;\le\; \big(\sigma_{\max}(\boldsymbol{\Omega}_{hh})\big)^{T-k}
$$

- اگر $\sigma_{\max} < 1$ باشد، گرادیان با افزایش فاصلهٔ $T-k$ به‌صورت **نمایی** کوچک می‌شود: **گرادیان محوشونده** (Vanishing gradient).
- اگر $\sigma_{\max} > 1$ باشد، گرادیان **ممکن است** به‌صورت نمایی بزرگ شود: **گرادیان انفجاری** (Exploding gradient).

> [!note] نکتهٔ کلیدی
> اشباع تابع $\tanh$ مشکل محوشدگی را شدیدتر می‌کند: وقتی $|\mathbf{z}_t|$ بزرگ است، $\tanh'(\mathbf{z}_t)$ نزدیک صفر است و ژاکوبی آن گام تقریباً صفر می‌شود. در مثال عددی بخش پنجم، هر ژاکوبی کمتر از $0.5$ بود و گرادیان پس از تنها دو گام به حدود $0.21$ رسید؛ پس از ۲۰ گام با همین نرخ، کمتر از یک‌میلیونیم باقی می‌ماند.

---

### پیامد: وابستگی‌های بلندمدت یاد گرفته نمی‌شوند

در بخش پنجم دیدیم که گرادیان نسبت به وزن‌های مشترک، **جمع** سهم همهٔ گام‌هاست:

$$
\frac{\partial L}{\partial \boldsymbol{\Omega}_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{z}_t}\, \mathbf{h}_{t-1}^{\top}
$$

وقتی گرادیان محو می‌شود، سهم گام‌های **دور** در این جمع تقریباً صفر است و جمع تقریباً فقط از گام‌های **نزدیک** ساخته می‌شود. نتیجه این است که وزن‌ها طوری به‌روز می‌شوند که مدل وابستگی‌های **کوتاه‌مدت** را یاد بگیرد و وابستگی‌های **بلندمدت** عملاً در آموزش دیده نمی‌شوند.

> [!example] شهود
> در جملهٔ بخش اول («**علی** که سال‌ها در خارج از کشور زندگی کرده بود و تازه به شهرش برگشته بود، از دیدن دوستان قدیمی‌**اش** خوشحال شد»)، برای اینکه مدل یاد بگیرد «ـَش» به «علی» برمی‌گردد، خطای پیش‌بینی در انتهای جمله باید از روی ده‌ها گام به عقب برسد. با گرادیان محوشونده، این سیگنال پیش از رسیدن به «علی» تقریباً از بین رفته است.

---

### راه‌حل گرادیان انفجاری: برش گرادیان

گرادیان انفجاری معمولاً به‌صورت **ناگهانی** رخ می‌دهد: زیان در طول آموزش ناگهان جهش می‌کند یا به `nan` می‌رسد. راه‌حل ساده و رایج آن **برش گرادیان** (Gradient clipping) است. اگر نُرم بردار گرادیان $\mathbf{g}$ (شامل همهٔ پارامترها) از یک آستانهٔ $\tau$ بیشتر شود، آن را کوچک می‌کنیم:

$$
\mathbf{g} \;\leftarrow\; \begin{cases} \mathbf{g} & \|\mathbf{g}\| \le \tau \\[4pt] \tau\,\dfrac{\mathbf{g}}{\|\mathbf{g}\|} & \|\mathbf{g}\| > \tau \end{cases}
$$

جهت گرادیان تغییر نمی‌کند و فقط **اندازهٔ گام** محدود می‌شود. در PyTorch این کار بین محاسبهٔ گرادیان و به‌روزرسانی وزن‌ها انجام می‌شود:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### راه‌حل گرادیان محوشونده

برش گرادیان فقط گرادیان‌های **بزرگ** را کوچک می‌کند و برای گرادیان محوشونده کاری انجام نمی‌دهد. راه‌حل‌های جزئی مانند مقداردهی اولیهٔ دقیق $\boldsymbol{\Omega}_{hh}$ (مثلاً با یک ماتریس متعامد) کمک می‌کنند، اما مشکل را حل نمی‌کنند.

راه‌حل اصلی، تغییر **معماری** است؛ همان ایده‌ای که در فصل ۱۱ مشکل شبکه‌های عمیق را حل کرد. در شبکه‌های باقیمانده دیدیم:

$$
\frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \mathbf{I} + \frac{\partial \mathbf{f}_k}{\partial \mathbf{h}_{k-1}}
$$

جملهٔ همانی $\mathbf{I}$ یک «بزرگراه» برای عبور گرادیان می‌ساخت. شبکه‌های **LSTM** و **GRU** (بخش‌های هفتم و هشتم) همین ایده را در طول **زمان** پیاده می‌کنند: به‌جای اینکه حالت پنهان در هر گام کاملاً بازنویسی شود، حافظه با یک **به‌روزرسانی جمعی** از گامی به گام بعد منتقل می‌شود و گرادیان می‌تواند بدون ضرب مکرر در $\boldsymbol{\Omega}_{hh}$ به گام‌های دور برسد.

---
## بخش هفتم: حافظهٔ کوتاه‌مدت طولانی (LSTM)

در بخش ششم دیدیم که مشکل RNN ساده این است که حالت پنهان در **هر گام کاملاً بازنویسی** می‌شود و گرادیان برای رسیدن به گام‌های دور باید بارها در $\boldsymbol{\Omega}_{hh}$ و $\tanh'$ ضرب شود. شبکهٔ **حافظهٔ کوتاه‌مدت طولانی** (Long Short-Term Memory, LSTM) این مشکل را با دو ایده حل می‌کند:

- یک حافظهٔ جداگانه به نام **حالت سلول** (Cell state) $\mathbf{c}_t$ که به‌صورت **جمعی** به‌روز می‌شود، نه با بازنویسی کامل.
- سه **دروازه** (Gate) که یاد می‌گیرند در هر گام چه چیزی از حافظه پاک شود، چه چیزی به آن اضافه شود و چه چیزی از آن خوانده شود.

---

### دروازه‌ها

هر دروازه برداری با مقادیر بین ۰ و ۱ است که از ورودی فعلی و حالت پنهان قبلی ساخته می‌شود. برای این کار از تابع **سیگموید** $\text{sig}[\cdot]$ استفاده می‌کنیم که خروجی آن همیشه در بازهٔ $(0, 1)$ است:

$$
\begin{aligned}
\mathbf{f}_t &= \text{sig}\Big[\boldsymbol{\beta}_f + \boldsymbol{\Omega}_{xf}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hf}\,\mathbf{h}_{t-1}\Big] && \text{دروازهٔ فراموشی (Forget gate)} \\
\mathbf{i}_t &= \text{sig}\Big[\boldsymbol{\beta}_i + \boldsymbol{\Omega}_{xi}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hi}\,\mathbf{h}_{t-1}\Big] && \text{دروازهٔ ورودی (Input gate)} \\
\mathbf{o}_t &= \text{sig}\Big[\boldsymbol{\beta}_o + \boldsymbol{\Omega}_{xo}\,\mathbf{x}_t + \boldsymbol{\Omega}_{ho}\,\mathbf{h}_{t-1}\Big] && \text{دروازهٔ خروجی (Output gate)}
\end{aligned}
$$

دروازه‌ها به‌صورت **عنصربه‌عنصر** ($\odot$) در بردارهای دیگر ضرب می‌شوند: مقدار نزدیک ۱ یعنی «عبور کامل» و مقدار نزدیک ۰ یعنی «مسدود».

---

### به‌روزرسانی حافظه

ابتدا یک **محتوای پیشنهادی** (Candidate) برای افزودن به حافظه ساخته می‌شود؛ این همان محاسبهٔ RNN ساده است:

$$
\tilde{\mathbf{c}}_t = \tanh\Big[\boldsymbol{\beta}_c + \boldsymbol{\Omega}_{xc}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hc}\,\mathbf{h}_{t-1}\Big]
$$

سپس حالت سلول و حالت پنهان به‌روز می‌شوند:

$$
\begin{aligned}
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} \;+\; \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh\big[\mathbf{c}_t\big]
\end{aligned}
$$

- دروازهٔ **فراموشی** تعیین می‌کند چه بخشی از حافظهٔ قبلی $\mathbf{c}_{t-1}$ **نگه داشته** شود.
- دروازهٔ **ورودی** تعیین می‌کند چه بخشی از محتوای جدید $\tilde{\mathbf{c}}_t$ به حافظه **اضافه** شود.
- دروازهٔ **خروجی** تعیین می‌کند چه بخشی از حافظه در حالت پنهان $\mathbf{h}_t$ **ظاهر** شود.

شکل زیر یک سلول LSTM را نشان می‌دهد. خط پررنگ بالایی مسیر **حالت سلول** است: از $\mathbf{c}_{t-1}$ تا $\mathbf{c}_t$ فقط یک ضرب در دروازهٔ فراموشی و یک جمع با محتوای جدید روی آن انجام می‌شود. جعبه‌های بنفش سه دروازه و محتوای پیشنهادی هستند که همگی از $\mathbf{h}_{t-1}$ و $\mathbf{x}_t$ ساخته می‌شوند.

![lstm_cell](../img/lstm_cell.png)

خروجی مدل مثل قبل از حالت پنهان محاسبه می‌شود: $$\hat{\mathbf{y}}_t = \boldsymbol{\beta}_y + \boldsymbol{\Omega}_{hy}\,\mathbf{h}_t$$ و حالت‌های اولیه صفر در نظر گرفته می‌شوند: $$\mathbf{h}_0 = \mathbf{c}_0 = \mathbf{0}$$

> [!example] شهود
> در جملهٔ «**علی** که سال‌ها در خارج از کشور زندگی کرده بود ... از دیدن دوستان قدیمی‌**اش** خوشحال شد»، مدل می‌تواند با دیدن «علی» اطلاعات «فاعل جمله مفرد و مذکر است» را در یکی از مؤلفه‌های $\mathbf{c}_t$ بنویسد (دروازهٔ ورودی باز)، در طول جملهٔ میانی آن را دست‌نخورده نگه دارد (دروازهٔ فراموشی نزدیک ۱ و دروازهٔ ورودی نزدیک ۰) و در انتهای جمله آن را بخواند (دروازهٔ خروجی باز).

---

### چرا گرادیان محو نمی‌شود؟

مسیر اصلی حافظه از $\mathbf{c}_{t-1}$ به $\mathbf{c}_t$ فقط یک ضرب عنصربه‌عنصر در دروازهٔ فراموشی است. اگر فقط همین مسیر مستقیم را در نظر بگیریم:

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \text{diag}\big(\mathbf{f}_t\big)
\qquad \Longrightarrow \qquad
\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_k} = \prod_{t=k+1}^{T} \text{diag}\big(\mathbf{f}_t\big)
$$

این را با RNN ساده (بخش ششم) مقایسه کنید:

- در این مسیر **هیچ ضربی در ماتریس وزن** $\boldsymbol{\Omega}_{hh}$ وجود ندارد.
- در این مسیر **هیچ مشتق** $\tanh'$ (و در نتیجه هیچ اشباعی) وجود ندارد.
- مقدار $\mathbf{f}_t$ **یاد گرفته می‌شود**: اگر نگه داشتن یک اطلاعات برای مسئله مفید باشد، مدل یاد می‌گیرد دروازهٔ فراموشی آن مؤلفه را نزدیک ۱ نگه دارد و گرادیان تقریباً بدون تضعیف از روی ده‌ها گام عبور می‌کند.

> [!note] ارتباط با شبکه‌های باقیمانده
> اگر $\mathbf{f}_t = \mathbf{1}$ باشد، به‌روزرسانی حافظه به شکل $$\mathbf{c}_t = \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$ درمی‌آید که دقیقاً ساختار **اتصال باقیمانده** در فصل ۱۱ است ($$\mathbf{h}_k = \mathbf{h}_{k-1} + \mathbf{f}_k[\mathbf{h}_{k-1}]$$؛ در آنجا $\mathbf{f}_k$ نماد بلوک باقیمانده بود، نه دروازهٔ فراموشی)، با این تفاوت که این بار در بُعد **زمان** است، نه عمق. دروازهٔ فراموشی این اتصال را **قابل‌یادگیری** می‌کند: مدل می‌تواند هر جا که لازم است حافظه را پاک کند.

---

### هزینهٔ LSTM

یک LSTM به‌جای یک تبدیل خطی، **چهار** تبدیل خطی دارد (سه دروازه و محتوای پیشنهادی)؛ پس تعداد پارامترهای آن تقریباً **چهار برابر** RNN ساده با همان اندازهٔ حالت پنهان است. برای مثال MNIST بخش چهارم ($D = 28$ و $H = 128$):

| لایه        | تعداد پارامترها |
| ----------- | --------------- |
| `nn.RNN`    | 20,224          |
| `nn.LSTM`   | 80,896          |

---

### LSTM در PyTorch

استفاده از LSTM در PyTorch تقریباً مثل `nn.RNN` است؛ تنها تفاوت این است که خروجی دوم به‌جای `h_n` یک زوج شامل آخرین حالت پنهان و آخرین حالت سلول است. در مدل بخش چهارم کافی است دو خط عوض شود:

```python
self.rnn = nn.LSTM(D, H, batch_first=True)   # instead of nn.RNN

out, (h_n, c_n) = self.rnn(x)                # h_n: last hidden state, c_n: last cell state
```

> [!note] نکتهٔ کلیدی
> برای ادامه دادن یک دنباله در فراخوانی بعدی (بخش چهارم)، در LSTM باید **هر دو** حالت را منتقل کرد: حالت پنهان $\mathbf{h}$ و حالت سلول $\mathbf{c}$. حافظهٔ بلندمدت در $\mathbf{c}$ نگه داشته می‌شود.

---
## بخش هشتم: واحد بازگشتی دروازه‌دار (GRU)

شبکهٔ **واحد بازگشتی دروازه‌دار** (Gated Recurrent Unit, GRU) نسخهٔ ساده‌شدهٔ LSTM است. همان ایدهٔ اصلی، یعنی **به‌روزرسانی جمعی** حافظه و کنترل آن با دروازه‌ها، را نگه می‌دارد، اما:

- حالت سلول جداگانه ندارد؛ حافظه همان حالت پنهان $\mathbf{h}_t$ است.
- به‌جای سه دروازه، **دو** دروازه دارد: دروازهٔ **به‌روزرسانی** (Update gate) و دروازهٔ **بازنشانی** (Reset gate).

---

### دروازه‌ها

$$
\begin{aligned}
\mathbf{u}_t &= \text{sig}\Big[\boldsymbol{\beta}_u + \boldsymbol{\Omega}_{xu}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hu}\,\mathbf{h}_{t-1}\Big] && \text{دروازهٔ به‌روزرسانی} \\
\mathbf{r}_t &= \text{sig}\Big[\boldsymbol{\beta}_r + \boldsymbol{\Omega}_{xr}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hr}\,\mathbf{h}_{t-1}\Big] && \text{دروازهٔ بازنشانی}
\end{aligned}
$$

> [!note] نکتهٔ نمادگذاری
> در بسیاری از منابع دروازهٔ به‌روزرسانی را با $\mathbf{z}_t$ نشان می‌دهند. ما از $\mathbf{u}_t$ استفاده می‌کنیم، چون $\mathbf{z}_t$ را در بخش پنجم برای ورودی تابع فعال‌سازی به کار بردیم.

---

### به‌روزرسانی حالت پنهان

ابتدا یک **حالت پنهان پیشنهادی** $\tilde{\mathbf{h}}_t$ ساخته می‌شود. این دقیقاً معادلهٔ RNN ساده در بخش سوم است، با این تفاوت که حالت قبلی ابتدا در دروازهٔ بازنشانی ضرب می‌شود:

$$
\tilde{\mathbf{h}}_t = \tanh\Big[\boldsymbol{\beta}_h + \boldsymbol{\Omega}_{xh}\,\mathbf{x}_t + \boldsymbol{\Omega}_{hh}\,\big(\mathbf{r}_t \odot \mathbf{h}_{t-1}\big)\Big]
$$

سپس حالت پنهان جدید، **میانگین وزنی** حالت قبلی و حالت پیشنهادی است:

$$
\mathbf{h}_t = (\mathbf{1} - \mathbf{u}_t) \odot \mathbf{h}_{t-1} \;+\; \mathbf{u}_t \odot \tilde{\mathbf{h}}_t
$$

- دروازهٔ **به‌روزرسانی** تعیین می‌کند چه مقدار از حافظه **بازنویسی** شود: اگر $\mathbf{u}_t \approx \mathbf{0}$ باشد، حالت قبلی بدون تغییر کپی می‌شود؛ اگر $\mathbf{u}_t \approx \mathbf{1}$ باشد، حالت پیشنهادی جایگزین آن می‌شود.
- دروازهٔ **بازنشانی** تعیین می‌کند گذشته در ساختن حالت پیشنهادی چقدر **نادیده گرفته** شود: اگر $\mathbf{r}_t \approx \mathbf{0}$ باشد، $\tilde{\mathbf{h}}_t$ فقط از ورودی فعلی ساخته می‌شود؛ مثلاً در شروع یک جملهٔ تازه.

شکل زیر یک سلول GRU را نشان می‌دهد. خط پررنگ بالایی مسیر حافظه از $\mathbf{h}_{t-1}$ تا $\mathbf{h}_t$ است: یک ضرب در $\mathbf{1} - \mathbf{u}_t$ و یک جمع با سهم حالت پیشنهادی.

![gru_cell](../img/gru_cell.png)

> [!note] ارتباط با LSTM و شبکه‌های باقیمانده
> مسیر مستقیم حافظه در GRU مانند LSTM است: در این مسیر، مشتق $\mathbf{h}_t$ نسبت به $\mathbf{h}_{t-1}$ برابر $$\text{diag}\big(\mathbf{1} - \mathbf{u}_t\big)$$ است و هیچ ضربی در $\boldsymbol{\Omega}_{hh}$ یا $\tanh'$ ندارد. در واقع $\mathbf{1} - \mathbf{u}_t$ نقش دروازهٔ فراموشی و $\mathbf{u}_t$ نقش دروازهٔ ورودی LSTM را دارد، با این قید که این دو **به هم وابسته‌اند**: هرچه بیشتر اضافه شود، همان‌قدر بیشتر فراموش می‌شود.

---

### مقایسهٔ LSTM و GRU

| ویژگی                          | LSTM                                   | GRU                                    |
| ------------------------------ | -------------------------------------- | -------------------------------------- |
| تعداد دروازه‌ها                 | ۳ (فراموشی، ورودی، خروجی)              | ۲ (به‌روزرسانی، بازنشانی)               |
| حالت‌ها                         | $\mathbf{h}_t$ و $\mathbf{c}_t$        | فقط $\mathbf{h}_t$                     |
| تعداد تبدیل‌های خطی             | ۴                                      | ۳                                      |
| پارامترهای لایه ($D = 28$، $H = 128$) | 80,896                                 | 60,672                                 |

در عمل دقت این دو مدل معمولاً نزدیک به هم است و هیچ‌کدام همیشه برتر نیست. GRU پارامتر کمتری دارد و کمی سریع‌تر است؛ LSTM به‌دلیل حافظهٔ جداگانه گاهی در دنباله‌های بسیار بلند بهتر عمل می‌کند.

> [!warning] تفاوت قرارداد در PyTorch
> در مستندات PyTorch برای `nn.GRU`، نماد $z$ برای دروازه‌ای به کار رفته که سهم **حالت قبلی** را تعیین می‌کند، یعنی همان $\mathbf{1} - \mathbf{u}_t$ در نماد ما. همچنین در پیاده‌سازی PyTorch دروازهٔ بازنشانی **بعد** از ضرب ماتریسی اعمال می‌شود. این دو تفاوت فقط در نحوهٔ نوشتن است و ایدهٔ مدل را تغییر نمی‌دهد.

---

### مثال: مقایسهٔ RNN، LSTM و GRU

مدل چندبه‌یک بخش چهارم را با هر سه نوع لایهٔ بازگشتی، با تنظیمات یکسان ($H = 128$، ۵ دوره، بهینه‌ساز Adam)، روی MNIST سطربه‌سطر آموزش می‌دهیم:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

tf = transforms.ToTensor()
train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
test_set = datasets.MNIST("data", train=False, download=True, transform=tf)
test_loader = DataLoader(test_set, batch_size=1000)

# ---------------------------------------------------------------
# Same many-to-one model as Section 4, with a choice of recurrent layer
# ---------------------------------------------------------------
class SeqClassifier(nn.Module):
    def __init__(self, cell, D=28, H=128, K=10):
        super().__init__()
        self.rnn = cell(D, H, batch_first=True)   # nn.RNN, nn.LSTM or nn.GRU
        self.fc = nn.Linear(H, K)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in loader:
            correct += (model(img.squeeze(1)).argmax(1) == label).sum().item()
    return correct / len(loader.dataset)

for name, cell in [("RNN", nn.RNN), ("LSTM", nn.LSTM), ("GRU", nn.GRU)]:
    torch.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    model = SeqClassifier(cell)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        for img, label in train_loader:
            loss = loss_fn(model(img.squeeze(1)), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:5s} parameters {n_params:6d}  test accuracy {accuracy(model, test_loader):.4f}")
```

```
RNN   parameters  21514  test accuracy 0.9471
LSTM  parameters  82186  test accuracy 0.9779
GRU   parameters  61962  test accuracy 0.9766
```

شبکه‌های دروازه‌دار با تنظیمات یکسان حدود ۳ درصد دقیق‌ترند و LSTM و GRU نتیجهٔ تقریباً یکسانی دارند. حتی در این دنبالهٔ کوتاه ($T = 28$)، RNN ساده برای استفاده از اطلاعات سطرهای بالای تصویر مشکل دارد. دقت RNN کمی با بخش چهارم فرق دارد، چون ترتیب تولید اعداد تصادفی در این کد متفاوت است.

---
## بخش نهم: یک مدل کامل LSTM و GRU در PyTorch

در عمل یک لایهٔ بازگشتی به‌تنهایی به کار نمی‌رود و بخشی از یک شبکهٔ بزرگ‌تر است. در این بخش یک معماری استاندارد برای **طبقه‌بندی متن** (مثلاً تحلیل احساسات؛ یک مسئلهٔ چندبه‌یک) را می‌سازیم که اجزای اصلی آن در بیشتر مدل‌های بازگشتی تکرار می‌شوند:

1. **لایهٔ نهفته‌سازی** (Embedding): هر کلمه با یک شمارهٔ صحیح (Token id) مشخص است؛ لایهٔ `nn.Embedding` هر شماره را به یک بردار ویژگی قابل‌یادگیری با $D$ بعد تبدیل می‌کند. این بردارها همان $\mathbf{x}_t$ هستند.
2. **چند لایهٔ بازگشتی روی هم** (شکل RNN چندلایه در بخش سوم) با حذف تصادفی (Dropout) بین لایه‌ها.
3. **پردازش دوطرفه** (Bidirectional).
4. **دنباله‌های با طول متفاوت**: جمله‌های یک دسته طول‌های متفاوتی دارند و با صفر لایه‌گذاری (Padding) می‌شوند. با تابع `pack_padded_sequence` لایهٔ بازگشتی گام‌های لایه‌گذاری را پردازش نمی‌کند و `h_n` آخرین گام **واقعی** هر جمله را برمی‌گرداند.
5. **لایهٔ خروجی**: آخرین حالت پنهان لایهٔ آخر (در هر دو جهت) به یک لایهٔ تمام‌متصل داده می‌شود.

> [!note] RNN دوطرفه
> یک RNN دوطرفه دو RNN جداگانه دارد: یکی دنباله را از ابتدا به انتها و دیگری از انتها به ابتدا می‌خواند. حالت پنهان هر گام از کنار هم گذاشتن حالت‌های دو جهت ساخته می‌شود؛ پس هر گام هم از **گذشته** و هم از **آینده** خبر دارد. این کار زمانی مفید است که کل دنباله از قبل در دسترس باشد (مثل طبقه‌بندی یک جملهٔ کامل)، اما برای پیش‌بینی برخط یا تولید گام‌به‌گام قابل استفاده نیست، چون آینده هنوز در دسترس نیست.

### تعریف مدل

با یک آرگومان می‌توان بین LSTM و GRU جابه‌جا شد:

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class TextClassifier(nn.Module):
    """Embedding -> 2-layer bidirectional LSTM/GRU -> dropout -> linear."""
    def __init__(self, vocab_size, num_classes, cell="lstm",
                 emb_dim=100, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.cell = cell
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                       bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, tokens, lengths):
        x = self.embedding(tokens)                          # (B, T, emb_dim)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h_n = self.rnn(packed)
        if self.cell == "lstm":
            h_n = h_n[0]                                    # (h_n, c_n) -> h_n
        # h_n: (num_layers * 2, B, hidden); last layer = forward [-2], backward [-1]
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)       # (B, 2 * hidden)
        return self.fc(self.dropout(h_last))
```

> [!note] نکتهٔ کلیدی
> در یک مدل دوطرفهٔ چندلایه، شکل `h_n` برابر `(num_layers * 2, B, hidden)` است و حالت‌های هر لایه به ترتیب «جهت رو به جلو، جهت معکوس» پشت سر هم قرار دارند. بنابراین دو سطر آخر آن، حالت نهایی **لایهٔ آخر** در دو جهت است. در LSTM خروجی دوم یک زوج `(h_n, c_n)` است و فقط `h_n` را برمی‌داریم.

### به‌کارگیری مدل

یک دسته با سه جملهٔ کوتاه (شمارهٔ کلمه‌ها؛ صفر برای لایه‌گذاری) به هر دو مدل داده می‌شود:

```python
torch.manual_seed(0)

# A mini-batch of 3 sentences (token ids); 0 = padding
tokens = torch.tensor([[12, 45,  7, 89, 3],
                       [ 5, 61,  0,  0, 0],
                       [33,  2, 98,  0, 0]])
lengths = torch.tensor([5, 2, 3])

for cell in ["lstm", "gru"]:
    model = TextClassifier(vocab_size=10000, num_classes=2, cell=cell)
    logits = model(tokens, lengths)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cell}: logits {tuple(logits.shape)}  parameters {n_params:,}")
```

```
lstm: logits (3, 2)  parameters 1,631,298
gru: logits (3, 2)  parameters 1,473,602
```

حدود یک میلیون پارامتر از این تعداد مربوط به لایهٔ نهفته‌سازی است ($10000 \times 100$).

آموزش این مدل دقیقاً مانند بخش چهارم است؛ تنها تفاوت، استفاده از برش گرادیان (بخش ششم) است که برای مدل‌های بازگشتی یک روال استاندارد به شمار می‌رود:

```python
model = TextClassifier(vocab_size=10000, num_classes=2, cell="gru")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for tokens, lengths, labels in train_loader:
    loss = loss_fn(model(tokens, lengths), labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Section 6
    optimizer.step()
```

برای دنباله‌های عددی (مثل سیگنال‌ها یا سری‌های زمانی) لایهٔ نهفته‌سازی لازم نیست: ورودی با شکل `(B, T, D)` مستقیماً به لایهٔ بازگشتی داده می‌شود و بقیهٔ مدل تغییری نمی‌کند.

---
