# Deep Learning Course — Persian Lecture Notes

جزوه‌های درس **یادگیری عمیق** به زبان فارسی، بر پایهٔ کتاب
Simon J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023) — [udlbook.github.io](https://udlbook.github.io/udlbook/).

Lecture notes for a Deep Learning course, written in Persian. The notes follow Prince's *Understanding Deep Learning* and are written in Markdown (authored in [Obsidian](https://obsidian.md)); math is LaTeX (`$...$`, `$$...$$`).

## Contents

| Folder | Topic | Files |
|---|---|---|
| `Chapter 10/` | Convolutional networks | `Sec 1.md` – `Sec 4.md`, `Sec 4_hw1.md` (project 1: Fashion-MNIST CNN) |
| `Chapter 11/` | Residual networks | `Sec 5.md` – `Sec 7.md`, `Sec 7_hw2.md` (project 2: transfer learning on Oxford Flowers-102) |
| `Chapter 12/` | Recurrent networks (RNN, LSTM, GRU) — background for Transformers | `Sec 8.md`, `Sec 9.md` |
| — | Supplementary | `Grad-CAM.md`, `autoencoders_to_vae_lecture.md` |
| `img/` | Figures used in the notes | |
| `Codes/` | Jupyter notebooks for the sessions | `Ch10/` |

## Reading the notes

- **On GitHub:** open any `.md` file; GitHub renders the math and images.
- **In Obsidian (best):** clone the repo and open the folder as a vault. Right-to-left layout is easiest to read there.

```bash
git clone https://github.com/haghbinh/deep-learning-course.git
```

## Data

Datasets are not stored in the repo. `sec4.ipynb` downloads MNIST through `torchvision` into `Codes/Ch10/data/` on first run. `sec3_1.ipynb` expects `data/mnist1d_data.pkl` from the [MNIST-1D](https://github.com/greydanus/mnist1d) project.

## Credits

Some figures are taken from Prince, *Understanding Deep Learning* (figures © the author, used for teaching); the remaining figures were made for this course.
