import torch, numpy as np, cv2
from PIL import Image
import matplotlib.pyplot as plt

# load model using the local UNet implementation
from training.models.unet_colorizer import UNetColorizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNetColorizer(base=64).to(device)
ckpt = torch.load('training/runs/best.pt', map_location=device)
# extract likely state dict
if isinstance(ckpt, dict) and 'model' in ckpt:
    state = ckpt['model']
elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state = ckpt['model_state_dict']
else:
    state = ckpt

# try non-strict load first, fallback to a simple name mapping if necessary
try:
    res = model.load_state_dict(state, strict=False)
    if getattr(res, 'missing_keys', None) or getattr(res, 'unexpected_keys', None):
        print('load_state_dict result:', res)
except RuntimeError as e:
    print('Initial load failed, attempting mapping:', e)
    mapped = {}
    for k, v in state.items():
        new_k = k.replace('.net', '')
        new_k = new_k.replace('mid', 'bottleneck')
        new_k = new_k.replace('out_conv', 'head.0')
        mapped[new_k] = v
    res = model.load_state_dict(mapped, strict=False)
    print('Mapped load result:', res)
model.eval()

# load input L (use local test image in repo)
# img_path = 'candle.png'
img_path = 'apple-modified.jpg'
orig = Image.open(img_path).convert('L')
import numpy as np, cv2
orig_np = np.array(orig)
H, W = 256, 256
L_resized = cv2.resize(orig_np, (W, H)).astype(np.float32)
L_norm = L_resized / 255.0            # training likely used [0,1] for L
L_t = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(L_t)           # shape [1,2,H,W]
out_np = out.squeeze(0).cpu().numpy().transpose(1,2,0)  # [H,W,2]
print("out min/max/mean:", out.min().item(), out.max().item(), out.mean().item())

def make_rgb_from_ab(ab_map, L_norm):
    # ab_map expected in actual ab units (not normalized)
    lab = np.zeros((H, W, 3), dtype=np.float32)
    lab[:,:,0] = L_norm * 100.0         # L in [0,100]
    lab[:,:,1:] = ab_map
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb, 0, 1)
    return rgb

# --- Option A: assume out in [0,1] representing (ab+128)/255 ---
ab_A = out_np * 255.0 - 128.0
ab_A = np.clip(ab_A, -128, 127)
rgb_A = make_rgb_from_ab(ab_A, L_norm)

# --- Option B: assume out in [-1,1] representing ab/128 ---
ab_B = out_np * 128.0
ab_B = np.clip(ab_B, -128, 127)
rgb_B = make_rgb_from_ab(ab_B, L_norm)

# show side-by-side
plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.title('Input (Gray)'); plt.imshow(L_resized, cmap='gray'); plt.axis('off')
# plt.subplot(1,3,2); plt.title('Option A: out*255-128'); plt.imshow(rgb_A); plt.axis('off')
plt.subplot(1,2,2); plt.title('Output (Colorized)'); plt.imshow(rgb_B); plt.axis('off')
plt.show()

# save outputs locally
# Image.fromarray((rgb_A*255).astype('uint8')).save('colorized_optionA.png')
Image.fromarray((rgb_B*255).astype('uint8')).save('colorized_optionB.png')
print("Saved colorized_optionA.png and colorized_optionB.png")
