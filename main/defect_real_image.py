import sys
sys.path.insert(0, "/home/aiosman/miniconda3/envs/diffusion/lib/python3.10/site-packages")
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from main.train_diffusion import UNetModel, GaussianDiffusion

# ─── Configuration & Parameters ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_SIZE = 128
CHANNELS = 3
TIMESTEPS = 500
NUM_CLASSES = 5 

BASE_DIR = "/project/community/aiosman"
# Path to the real image
REAL_IMAGE_PATH = os.path.join(BASE_DIR, "real_dataset", "0804c5e3-6ca2-4fff-90a3-4db3d2d4820b.jpg")
# Path to the low density checkpoint as decided
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints_low_density", "model_epoch_150.pt")
# Output directory
SAVE_DIR = os.path.join(BASE_DIR, "real_image_results")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── MASK PARAMETERS (EDIT THESE TO CHANGE THE MASK LOCATION) ────────────────
# Defaulting to a central rectangle, but you can adjust these coordinates
MASK_Y_START = 48
MASK_Y_END   = 80
MASK_X_START = 48
MASK_X_END   = 80

# The starting timestep for reverse diffusion (500 = completely redraw defect from pure noise)
START_STEP = 500

# ─── Dataset Preprocessing ───────────────────────────────────────────────────
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Loading real image...")
try:
    pil_image = Image.open(REAL_IMAGE_PATH).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

tensor_image = transform(pil_image).unsqueeze(0).to(device) # Shape [1, 3, 128, 128]

# We want to test all 5 aspect ratios, so we duplicate the image 5 times in a batch
batch_images = tensor_image.repeat(NUM_CLASSES, 1, 1, 1) # Shape [5, 3, 128, 128]
batch_labels = torch.arange(NUM_CLASSES, device=device)  # Shape [5]

# ─── Mask Creation ───────────────────────────────────────────────────────────
mask = torch.zeros_like(batch_images)
# Set the defect area to 1.0
mask[:, :, MASK_Y_START:MASK_Y_END, MASK_X_START:MASK_X_END] = 1.0

# Create a "masked" display image (greyed out where defect is)
display_masked_img = tensor_image.clone()
display_masked_img[:, :, MASK_Y_START:MASK_Y_END, MASK_X_START:MASK_X_END] = 0.0 # grey (normalized 0)


# ─── Model Initialization ────────────────────────────────────────────────────
print("Initializing model architecture...")
model = UNetModel(
    in_channels=CHANNELS,
    model_channels=128,
    out_channels=CHANNELS,
    channel_mult=(1, 2, 4, 8),
    attention_resolutions=[16, 8],
    num_classes=NUM_CLASSES
).to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=TIMESTEPS)

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

model.eval() 


# ─── Inpainting Routine (RePaint Algorithm) ──────────────────────────────────
def inpaint_image(model, diffusion, real_images, labels, mask, start_step):
    print(f"Starting inpainting from step {start_step}...")
    num_imgs = real_images.shape[0]

    if start_step == diffusion.timesteps:
        # Start from pure noise
        img = torch.randn_like(real_images)
    else:
        # Start from partially noised real image
        t_start = torch.full((num_imgs,), start_step - 1, device=device, dtype=torch.long)
        img = diffusion.q_sample(real_images, t_start)

    with torch.no_grad():
        for i in reversed(range(start_step)):
            t_curr = torch.full((num_imgs,), i, device=device, dtype=torch.long)
            
            # Predict previous step
            model_pred = diffusion.p_sample(model, img, t_curr, labels=labels)
            
            # Get the ground truth noise level for the known regions
            if i > 0:
                t_next = torch.full((num_imgs,), i - 1, device=device, dtype=torch.long)
                known_noisy_real = diffusion.q_sample(real_images, t_next)
            else:
                known_noisy_real = real_images 

            # Composite: mask region comes from model, non-mask region comes from ground truth
            img = mask * model_pred + (1.0 - mask) * known_noisy_real

    return img

cleaned_images = inpaint_image(model, gaussian_diffusion, batch_images, batch_labels, mask, START_STEP)

# ─── Plotting Results ────────────────────────────────────────────────────────
print("Generating final plot...")
fig, axes = plt.subplots(1, 2 + NUM_CLASSES, figsize=(4 * (2 + NUM_CLASSES), 5))
fig.suptitle(f"Real Image Inpainting | Timesteps: {START_STEP}", fontsize=18)

def process_for_plot(t):
    # Convert from [-1, 1] to [0, 1] for matplotlib
    img = (t.cpu().squeeze(0).permute(1, 2, 0).numpy() + 1.0) / 2.0
    return np.clip(img, 0, 1)

# Plot Original
axes[0].imshow(process_for_plot(tensor_image))
axes[0].set_title("Original Real Image", fontsize=14)
axes[0].axis('off')

# Plot Masked
axes[1].imshow(process_for_plot(display_masked_img))
# Draw a red rectangle boundary on the masked plot to show exactly where it is
rect = plt.Rectangle((MASK_X_START, MASK_Y_START), MASK_X_END - MASK_X_START, MASK_Y_END - MASK_Y_START, 
                     linewidth=2, edgecolor='red', facecolor='none')
axes[1].add_patch(rect)
axes[1].set_title("Input with Mask Box", fontsize=14)
axes[1].axis('off')

# Plot 5 generations
for i in range(NUM_CLASSES):
    ax = axes[2 + i]
    ax.imshow(process_for_plot(cleaned_images[i].unsqueeze(0)))
    ax.set_title(f"Inpainted (Aspect Ratio {i+1})", fontsize=14)
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, "real_image_inpainting_results.png")
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close()

print(f"Success! Results saved to {save_path}")
