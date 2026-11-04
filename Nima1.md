ابتدا ساختار مدل فعلی را حساب کردم. شمارنده‌ام برای مدل شما دقیقاً ۱٬۴۹۵٬۴۱۱ پارامتر داد، که با عدد گزارش‌شده یکی است؛ پس محاسبات زیر قابل اتکاست.

## مهم‌ترین ضعف مدل فعلی: میدان دید کوچک

میدان دید (Receptive Field) آخرین لایه کانولوشنی فقط **۱۸۶ نمونه** است. در فرکانس نمونه‌برداری ۶۴ کیلوهرتز، این یعنی حدود **۳ میلی‌ثانیه**.

- یک سیکل جریان تغذیه در ۲۰ هرتز حدود ۳۲۰۰ نمونه است و در ۶۰ هرتز حدود ۱۰۷۰ نمونه.
- پس هر ویژگی فقط کسر کوچکی از یک سیکل را می‌بیند.
- `AdaptiveAvgPool` در انتها فقط این ویژگی‌های محلی را میانگین می‌گیرد. رابطه‌های بلندمدت، مثل مدولاسیون دامنه و باندهای کناری که امضای اصلی خطا در جریان هستند، تقریباً از دست می‌روند.
- حدود ۱٫۱۸ میلیون از ۱٫۵ میلیون پارامتر هم در دو لایه آخر جمع شده‌اند. این پارامترها عمق می‌سازند، نه میدان دید.

## پیشنهادها، به ترتیب اثر احتمالی

**۱. کاهش نرخ نمونه‌برداری ورودی (احتمالاً بیشترین اثر).**
- سیگنال را با فیلتر ضد‌الیاسینگ به حدود ۸ کیلوهرتز کاهش دهید، مثلاً با `scipy.signal.decimate(x, 8, ftype='fir')`.
- با این کار، پنجره ۴۰۹۶ نمونه‌ای به جای ۶۴ میلی‌ثانیه، ۰٫۵ ثانیه را پوشش می‌دهد و وضوح فرکانسی از ۱۵٫۶ هرتز به ۲ هرتز می‌رسد. این برای دیدن باندهای کناری خطا لازم است.
- هزینه‌اش این است که تعداد پنجره‌ها کم می‌شود. برای جبران، پنجره‌ها را با ۵۰ تا ۷۵ درصد هم‌پوشانی ببرید.
- ۸ و ۴ کیلوهرتز را هر دو امتحان کنید تا معلوم شود محتوای فرکانس بالا، مثل سوئیچینگ اینورتر، لازم است یا نه.

**۲. اصلاح نحوه آموزش.** این تغییرات معماری را دست نمی‌زنند و همان نوسان اعتبارسنجی را هم درمان می‌کنند:
- بهینه‌ساز AdamW با `OneCycleLR` یا زمان‌بند cosine
- `CrossEntropyLoss(label_smoothing=0.1)`
- EMA وزن‌ها
- استانداردسازی ثابت هر کانال ورودی

**۳. افزایش داده، فقط روی داده آموزش.**
- برش تصادفی: از یک بازه کمی بلندتر، ۴۰۹۶ نمونه انتخاب شود.
- مقیاس دامنه تصادفی بین ۰٫۹ و ۱٫۱.
- نویز گاوسی ضعیف.

برش تصادفی برای سیگنال‌های تناوبی معمولاً بیشترین کمک را می‌کند.

**۴. معماری با میدان دید بزرگ‌تر.** دو مدل زیر را با همان سقف پارامتر طراحی کردم.

**۵. تصمیم‌گیری بر اساس چند پنجره.** میانگین خروجی چند پنجره متوالی از یک رکورد معمولاً دقت را بالا می‌برد. این یک روش ارزیابی جداست و باید جداگانه گزارش شود.

## مدل‌های جایگزین با پارامتر کمتر یا مساوی

| مدل | پارامتر | میدان دید آخرین لایه |
|---|---|---|
| CNN1D فعلی | ۱٬۴۹۵٬۴۱۱ | ۱۸۶ نمونه |
| ResNet1D-SE | ۱٬۴۷۶٬۶۳۹ | ۱۲۹۱ نمونه (حدود ۷ برابر) |
| ResCNN-BiGRU-Attention | ۱٬۰۰۲٬۶۳۶ | تمام پنجره (از طریق GRU) |

پارامترها را با فرمول حساب کرده‌ام، چون نصب PyTorch در این محیط ممکن نشد. بعد از اولین اجرا با `count_params` تأییدشان کنید.

```python
import torch
import torch.nn as nn


class SE1D(nn.Module):
    """Squeeze-and-Excitation: وزن‌دهی به کانال‌ها"""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):                         # [B, C, T]
        return x * self.fc(x.mean(dim=2)).unsqueeze(2)


class BasicBlock1D(nn.Module):
    def __init__(self, c_in, c_out, k=7, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.se = SE1D(c_out)
        self.act = nn.GELU()
        self.short = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(c_out))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.act(out + self.short(x))


def stem(in_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, 32, 15, stride=2, padding=7, bias=False),
        nn.BatchNorm1d(32), nn.GELU(),
        nn.MaxPool1d(3, stride=2, padding=1))      # 4096 -> 1024


# ---------------- مدل ۱: ResNet1D-SE (حدود ۱٫۴۸ میلیون پارامتر) ----------------
class ResNet1D_SE(nn.Module):
    def __init__(self, in_channels=3, num_classes=51, widths=(48, 96, 128, 160), k=7, p_drop=0.2):
        super().__init__()
        layers, c = [stem(in_channels)], 32
        for i, w in enumerate(widths):              # 1024 -> 1024 -> 512 -> 256 -> 128
            layers += [BasicBlock1D(c, w, k, stride=1 if i == 0 else 2), BasicBlock1D(w, w, k)]
            c = w
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(c, num_classes))

    def forward_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward_features(x)


# ---------------- مدل ۲: ResCNN-BiGRU-Attention (حدود ۱٫۰ میلیون پارامتر) ----------------
class AttentionPooling1D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, x):                          # [B, T, D]
        w = torch.softmax(self.attn(x).float(), dim=1).to(x.dtype)
        return (w * x).sum(dim=1)


class ResCNN_BiGRU(nn.Module):
    def __init__(self, in_channels=3, num_classes=51, widths=(64, 96, 128, 160), k=7, hidden=128, p_drop=0.2):
        super().__init__()
        layers, c = [stem(in_channels)], 32
        for i, w in enumerate(widths):              # 4096 -> 1024 -> ... -> 128 گام
            layers.append(BasicBlock1D(c, w, k, stride=1 if i == 0 else 2))
            c = w
        self.cnn = nn.Sequential(*layers)
        self.gru = nn.GRU(c, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling1D(2 * hidden)
        self.classifier = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(p_drop),
                                        nn.Linear(2 * hidden, num_classes))

    def forward_features(self, x):
        h = self.cnn(x).transpose(1, 2)             # [B, 128, 160]
        h, _ = self.gru(h)                         # [B, 128, 256]
        return self.pool(h)                        # [B, 256]

    def forward(self, x):
        return self.classifier(self.forward_features(x))

    def extract_features(self, x):
        with torch.no_grad():
            return self.forward_features(x)
```

**چرا این دو مدل؟**
- **ResNet1D-SE:**
  - اتصال‌های میان‌بُر (residual) آموزش شبکه عمیق‌تر را پایدار می‌کنند.
  - هسته ۷ در همه لایه‌ها میدان دید را حدود ۷ برابر می‌کند.
  - بلوک SE به مدل اجازه می‌دهد کانال‌های مهم برای هر کلاس را پررنگ‌تر کند.
- **ResCNN-BiGRU:**
  - بخش کانولوشنی ویژگی‌های محلی را می‌سازد و طول را به ۱۲۸ گام می‌رساند.
  - GRU رابطه‌های بلندمدت را در کل پنجره مدل می‌کند، کاری که `AvgPool` نمی‌کند.
  - با ۱۲۸ گام، مشکل ناپایداری و کندی نسخه قبلی هم کمتر است.

**گزینه‌های دیگر:**
- **InceptionTime:** حدود ۴۰۰ هزار پارامتر دارد و روی معیارهای طبقه‌بندی سری زمانی بسیار قوی است.
- **مدل دوشاخه زمان و فرکانس:** یک شاخه کوچک روی طیف FFT هر پنجره کار می‌کند. برای سیگنال جریان معمولاً مفید است، چون امضای خطا ماهیت فرکانسی دارد.

## ترتیب پیشنهادی آزمایش‌ها

هر بار فقط یک تغییر بدهید تا اثر هرکدام معلوم شود:

1. اصلاحات آموزش (بند ۲) روی همین CNN.
2. کاهش نرخ نمونه‌برداری (بند ۱).
3. افزایش داده (بند ۳).
4. جایگزینی معماری با ResNet1D-SE، و بعد ResCNN-BiGRU.

نمی‌توانم عدد دقیقی برای بهبود تضمین کنم. تجربه کلی در تشخیص خطا با سیگنال جریان نشان می‌دهد بند ۱ و میدان دید بزرگ‌تر معمولاً بیشترین اثر را دارند.

اگر این نتایج قرار است در همین مقاله برود، همه مدل‌ها را با همان تقسیم داده، همان پنج بذر و همان تنظیمات آموزش اجرا کنید تا مقایسه منصفانه بماند.