import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when Streamlit's cwd is the app folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import local model
from training.models.unet_colorizer import UNetColorizer

st.set_page_config(page_title="Colorizer", layout="centered")

@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetColorizer(base=64).to(device)
    ckpt_path = '../training/runs/best1.pt'
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        st.error(f"Failed to load checkpoint {ckpt_path}: {e}")
        raise

    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        mapped = {}
        for k, v in state.items():
            new_k = k.replace('.net', '')
            new_k = new_k.replace('mid', 'bottleneck')
            new_k = new_k.replace('out_conv', 'head.0')
            mapped[new_k] = v
        model.load_state_dict(mapped, strict=False)

    model.eval()
    return model, device


def preprocess(img_pil, device):
    # convert to L channel and resize to 256x256
    gray = img_pil.convert('L')
    arr = np.array(gray)
    H, W = 256, 256
    L_resized = cv2.resize(arr, (W, H)).astype(np.float32)
    L_norm = L_resized / 255.0
    L_t = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device).float()
    return L_t, L_norm, L_resized


def postprocess(out_tensor, L_norm):
    out_np = out_tensor.squeeze(0).cpu().numpy().transpose(1,2,0)
    # assume model outputs in [-1,1] mapping to ab/128
    ab = out_np * 128.0
    ab = np.clip(ab, -128, 127)
    H, W = L_norm.shape
    lab = np.zeros((H, W, 3), dtype=np.float32)
    lab[:,:,0] = L_norm * 100.0
    lab[:,:,1:] = ab
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb, 0, 1)
    rgb_u8 = (rgb * 255).astype('uint8')
    return rgb_u8


def pil_bytes_from_array(arr):
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


st.title("Image Colorizer — UNet")
st.write("Upload a grayscale image (or color) and press Colorize to run the model.")

uploaded = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

col1, col2 = st.columns(2)

if uploaded is not None:
    input_img = Image.open(uploaded)
    with col1:
        st.image(input_img, caption='Input image', use_column_width=True)

    if st.button('Colorize'):
        with st.spinner('Loading model and running inference...'):
            model, device = load_model()
            L_t, L_norm, L_resized = preprocess(input_img, device)
            with torch.no_grad():
                out = model(L_t)
            rgb = postprocess(out, L_norm)

        with col2:
            st.image(rgb, caption='Colorized output', use_column_width=True)
            png_buf = pil_bytes_from_array(rgb)
            st.download_button('Download colorized PNG', data=png_buf, file_name='colorized.png', mime='image/png')

else:
    st.info('Upload an image to get started.')
