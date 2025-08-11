# Brain Tumor MRI Classifier (PyTorch)

A lightweight CNN that classifies brain MRI images into tumor classes.  
Works on CPU or NVIDIA GPU (CUDA 12.x). Training + evaluation live in a single Jupyter notebook.

**My result:** **96.57%** test accuracy (128×128 images, 25 epochs, AMP on, RTX 5060 Ti).

---

## Project Structure
```
.
├─ Main.ipynb               # training + evaluation
├─ main.py                  # optional script version (same pipeline)
├─ requirements.txt
├─ .gitignore
├─ models/                  # (ignored) saved weights
└─ data/                    # (ignored) dataset goes here
```

## Data Layout
Put your dataset in `data/` like this (folder names = class names):
```
data/
  Training/
    class_0/ *.jpg|*.png
    class_1/ *.jpg|*.png
    ...
  Testing/
    class_0/ *.jpg|*.png
    class_1/ *.jpg|*.png
    ...
```
**Dataset source:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  
**Num classes:** default is **4** (change in code if different).

## Setup

> Python 3.9+ recommended. Use either **uv** or plain **pip**.

### Using uv
```bash
uv venv
uv pip install -r requirements.txt
# Install PyTorch with CUDA (adjust cu129 to your CUDA runtime if needed)
uv pip install --index-url https://download.pytorch.org/whl/cu129 torch torchvision
```

### Using pip
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu129 torch torchvision
```

> No NVIDIA GPU? Use the CPU wheel instead:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

**Verify GPU**
```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Train & Evaluate

### Notebook
1. `uv run jupyter lab` (or `jupyter lab`)
2. Open `Main.ipynb` → Run All

Notes:
- On Windows notebooks use `num_workers=0` to avoid DataLoader hangs.
- Mixed precision (`torch.cuda.amp`) auto-enables when CUDA is available.
- Small speedups: `torch.backends.cudnn.benchmark = True`, use `channels_last`.

### Script (optional)
```bash
python main.py --data-root data --epochs 25 --batch-size 32 --num-classes 4
```

## Model
- **Backbone:** Simple CNN — 3×Conv (32→64→128) + ReLU + MaxPool → Flatten → Linear(256) → Dropout(0.5) → Linear(num_classes)
- **Input:** 128×128 RGB, normalized to mean/std = `[0.5, 0.5, 0.5]`
- **Optimizer:** Adam (lr=1e-4)
- **Loss:** CrossEntropyLoss
- **Speedups:** AMP, `channels_last`, cuDNN benchmark

## Results (my run)
- **Test Accuracy:** 96.57%
- **Test Loss:** ~0.133

_(Your numbers may vary with split/seed/GPU.)_

## Inference Example
```python
import torch, torchvision.transforms as T
from PIL import Image
from main import SimpleCNN  # if the model class is in main.py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=4).to(device)
model.load_state_dict(torch.load('models/best.pt', map_location=device))
model.eval()

tf = T.Compose([
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

img = tf(Image.open('path/to/mri.jpg')).unsqueeze(0).to(device)
with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
    pred = model(img).argmax(dim=1).item()
print('Predicted class:', pred)
```

## Troubleshooting
- **`cuda: False`** → install the CUDA wheel (`+cu129` etc.), update NVIDIA driver, restart kernel.
- **DataLoader hang on Windows** → set `num_workers=0` in notebooks.
- **CUDA OOM** → lower `batch_size`, use smaller `img_size`, or try gradient accumulation.

## License
MIT — add a `LICENSE` file if you want to use MIT.
