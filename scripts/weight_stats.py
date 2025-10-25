import os
import sys
import torch, numpy as np
# ensure workspace root is on sys.path so colorizer_v1 can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ck = torch.load('colorizer_v1/best.pt', map_location='cpu')
state = ck.get('model', ck)
from colorizer_v1.unet_colorizer import UNetColorizer
cfg = ck.get('cfg', {})
model = UNetColorizer(base=cfg.get('base_channels',64))
model.load_state_dict(state, strict=False)
all_weights = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
print('count', all_weights.size)
print('mean', float(all_weights.mean()))
print('std', float(all_weights.std()))
print('min', float(all_weights.min()))
print('max', float(all_weights.max()))
