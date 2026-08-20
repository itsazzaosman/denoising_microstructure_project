import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from abc import abstractmethod
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt

from main.train_diffusion import UNetModel, GaussianDiffusion

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Config ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 128
CHANNELS = 3
TIMESTEPS = 500
NUM_CLASSES = 5 

BASE_DIR = "/project/community/aiosman/diffusion_project"
test_dataset_path  = os.path.join(BASE_DIR, "dataset_split/test")
sample_dir         = os.path.join(BASE_DIR, "noise_level_accuracy")
ckpt_dir           = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(ckpt_dir, "model_epoch_150.pt")

os.makedirs(sample_dir, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
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

test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)

# ─── NEW: Fetch exactly one image per Aspect Ratio ────────────────────────────
def get_one_image_per_class(dataset, num_classes=NUM_CLASSES):
    imgs = []
    lbls = []
    found_classes = set()
    
    for img, label in dataset:
        if label not in found_classes:
            imgs.append(img)
            lbls.append(label)
            found_classes.add(label)
        if len(found_classes) == num_classes:
            break
            
    # Sort them so they always appear in order (Class 0, 1, 2, 3...)
    sorted_pairs = sorted(zip(lbls, imgs), key=lambda x: x[0])
    sorted_lbls, sorted_imgs = zip(*sorted_pairs)
    
    return torch.stack(sorted_imgs), torch.tensor(sorted_lbls)

# ─── Model Initialization ─────────────────────────────────────────────────────
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

# ─── Quantitative Metric ──────────────────────────────────────────────────────
def calculate_masked_pixel_accuracy(ground_truth, restored_img, mask, tolerance=0.10):
    diff = torch.abs(ground_truth - restored_img)
    matched_pixels = (diff < tolerance).all(dim=1) 
    
    spatial_mask = mask[:, 0, :, :].bool()
    matched_in_mask = matched_pixels & spatial_mask
    
    total_defect_pixels = spatial_mask.sum(dim=(1, 2)).float()
    accuracy = (matched_in_mask.sum(dim=(1, 2)).float() / total_defect_pixels) * 100.0
    return accuracy.mean().item()

# ─── Blur Ablation Test ───────────────────────────────────────────────────────
def test_artifact_cleanup_blur(model, diffusion, real_images, labels, start_step, save_dir):
    print(f"\n--- Running Ablation Test for t={start_step} ---")
    
    num_imgs = real_images.shape[0] # Dynamically get number of classes (e.g., 5)
    
    blurred_images = TF.gaussian_blur(real_images, kernel_size=[15, 15], sigma=[5.0, 5.0])
    corrupted_images = real_images.clone()
    
    mask = torch.zeros_like(real_images)
    mask[:, :, 56:72, 56:72] = 1.0 
    corrupted_images[:, :, 56:72, 56:72] = blurred_images[:, :, 56:72, 56:72]

    if start_step == 0:
        noisy_corrupted_display = corrupted_images.clone()
        cleaned_images = corrupted_images.clone() 
    else:
        t_start = torch.full((num_imgs,), start_step - 1, device=device, dtype=torch.long)
        img = diffusion.q_sample(corrupted_images, t_start)
        noisy_corrupted_display = img.clone() 

        with torch.no_grad():
            for i in reversed(range(start_step)):
                t_curr = torch.full((num_imgs,), i, device=device, dtype=torch.long)
                model_pred = diffusion.p_sample(model, img, t_curr, labels=labels)
                
                if i > 0:
                    t_next = torch.full((num_imgs,), i - 1, device=device, dtype=torch.long)
                    known_noisy_real = diffusion.q_sample(real_images, t_next)
                else:
                    known_noisy_real = real_images 

                img = mask * model_pred + (1.0 - mask) * known_noisy_real

        cleaned_images = img

    accuracy = calculate_masked_pixel_accuracy(real_images, cleaned_images, mask)
    print(f"Restoration Accuracy: {accuracy:.2f}%")

    # Plotting dynamically based on number of classes
    fig, axes = plt.subplots(4, num_imgs, figsize=(3 * num_imgs, 8))
    fig.suptitle(f"Blur Cleanup | Noise Step {start_step}/{diffusion.timesteps} | Accuracy: {accuracy:.2f}%", fontsize=16)

    row_titles = ["Original", "Blurred", f"Noised (t={start_step})", "AI Restored"]
    plot_data = [real_images, corrupted_images, noisy_corrupted_display, cleaned_images]

    for row_idx in range(4):
        axes[row_idx, 0].set_ylabel(row_titles[row_idx], fontsize=12, rotation=0, labelpad=40, ha='center')
        for col_idx in range(num_imgs):
            disp_img = (plot_data[row_idx][col_idx].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
            disp_img = np.clip(disp_img, 0, 1) 
            
            axes[row_idx, col_idx].imshow(disp_img)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            
            # Add Class label above the Original row
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"Aspect Ratio {labels[col_idx].item() + 1}")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"blur_ablation_t{start_step}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

    return accuracy

# ─── Execution Routine ────────────────────────────────────────────────────────
print("\nFetching specific aspect ratio images...")
batch_images, batch_labels = get_one_image_per_class(test_dataset, num_classes=NUM_CLASSES)
batch_images = batch_images.to(device)
batch_labels = batch_labels.to(device)

timesteps_to_test = [0, 10, 25,50, 75, 100, 250, 500]
recorded_accuracies = []

for t in timesteps_to_test:
    acc = test_artifact_cleanup_blur(
        model=model, 
        diffusion=gaussian_diffusion, 
        real_images=batch_images, 
        labels=batch_labels, 
        start_step=t,
        save_dir=sample_dir
    )
    recorded_accuracies.append(acc)

# print("\nAblation study complete! Check the 'noise_level_accuracy' folder for results.")

print("\nGenerating Accuracy vs. Timestep graph...")
plt.figure(figsize=(10, 6))

# Plot the line with circular markers
plt.plot(timesteps_to_test, recorded_accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# Styling the graph
plt.title('Microstructure Restoration Accuracy vs. Forward Noise Timestep', fontsize=14, fontweight='bold')
plt.xlabel('Noise Added (Timesteps $t$)', fontsize=12)
plt.ylabel('Masked Pixel Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(timesteps_to_test) # Force the X-axis to show exactly the steps you tested
plt.ylim(0, 100) # Lock Y-axis from 0 to 100%

# Highlight the peak accuracy
max_acc = max(recorded_accuracies)
best_t = timesteps_to_test[recorded_accuracies.index(max_acc)]
plt.annotate(f'Peak: {max_acc:.1f}%\n(t={best_t})', 
             xy=(best_t, max_acc), 
             xytext=(best_t, max_acc + 5),
             ha='center',
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

# Save the graph
graph_path = os.path.join(sample_dir, "accuracy_vs_timestep_plot.png")
plt.savefig(graph_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Ablation study complete! Check the '{sample_dir}' folder for the images and the summary graph.")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import numpy as np
# from abc import abstractmethod
# from torchvision import datasets, transforms
# import torchvision.transforms.functional as TF
# import os
# from torch.utils.data import DataLoader
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend for cluster
# import matplotlib.pyplot as plt

# from train_diffusion import UNetModel, GaussianDiffusion  # Ensure these are defined in train_diffusion.py

# # ─── Device ───────────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # ─── Config ───────────────────────────────────────────────────────────────────
# IMAGE_SIZE = 128
# CHANNELS = 3
# BATCH_SIZE = 8  # Only need a small batch for testing/visualization
# TIMESTEPS = 500
# NUM_CLASSES = 5 # Aspect ratios

# BASE_DIR = "/project/community/aiosman/diffusion_project"
# test_dataset_path  = os.path.join(BASE_DIR, "dataset_split/test")
# sample_dir         = os.path.join(BASE_DIR, "samples")
# ckpt_dir           = os.path.join(BASE_DIR, "checkpoints")

# # IMPORTANT: Update this to match your exact saved checkpoint name (e.g., model_epoch_150.pt)
# CHECKPOINT_PATH = os.path.join(ckpt_dir, "model_epoch_150.pt")

# os.makedirs(sample_dir, exist_ok=True)

# # ─── Dataset & Dataloader ─────────────────────────────────────────────────────
# class PadToSquare:
#     def __call__(self, img):
#         w, h = img.size
#         max_dim = max(w, h)
#         pad_left = (max_dim - w) // 2
#         pad_top = (max_dim - h) // 2
#         pad_right = max_dim - w - pad_left
#         pad_bottom = max_dim - h - pad_top
#         return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

# transform = transforms.Compose([
#     PadToSquare(),
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


# print("Initializing model architecture...")
# model = UNetModel(
#     in_channels=CHANNELS,
#     model_channels=128,
#     out_channels=CHANNELS,
#     channel_mult=(1, 2, 4, 8),
#     attention_resolutions=[16, 8],
#     num_classes=NUM_CLASSES
# ).to(device)

# gaussian_diffusion = GaussianDiffusion(timesteps=TIMESTEPS)

# print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
# if os.path.exists(CHECKPOINT_PATH):
#     # map_location=device ensures it loads correctly whether on GPU or CPU
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
#     # Extract the model weights from the saved dictionary
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f"Successfully loaded model from Epoch {checkpoint['epoch']} with Loss {checkpoint['loss']:.4f}")
# else:
#     raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Please verify the filename.")

# model.eval() # Strictly set to evaluation mode

# # ─── Quantitative Metric ──────────────────────────────────────────────────────
# def calculate_masked_pixel_accuracy(ground_truth, restored_img, mask, tolerance=0.10):
#     """Calculates the percentage of structurally matched pixels ONLY inside the defect mask."""
#     diff = torch.abs(ground_truth - restored_img)
#     matched_pixels = (diff < tolerance).all(dim=1) 
    
#     # Extract the 2D spatial mask (since your mask has 3 channels, we just grab the first one)
#     spatial_mask = mask[:, 0, :, :].bool()
    
#     # Count how many pixels match ONLY inside the True regions of the mask
#     matched_in_mask = matched_pixels & spatial_mask
    
#     # Calculate percentage based on the size of the defect, not the whole image
#     total_defect_pixels = spatial_mask.sum(dim=(1, 2)).float()
#     accuracy = (matched_in_mask.sum(dim=(1, 2)).float() / total_defect_pixels) * 100.0
    
#     return accuracy.mean().item()

# # ─── Blur Ablation Test ───────────────────────────────────────────────────────
# def test_artifact_cleanup_blur(model, diffusion, dataloader, start_step=250, save_dir="samples"):
#     print(f"Starting Blur Cleanup Test at Step {start_step}...")
    
#     real_images, labels = next(iter(dataloader))
#     real_images = real_images[:8].to(device) 
#     labels = labels[:8].to(device)

#     blurred_images = TF.gaussian_blur(real_images, kernel_size=[15, 15], sigma=[5.0, 5.0])
#     corrupted_images = real_images.clone()
    
#     mask = torch.zeros_like(real_images)
#     mask[:, :, 56:72, 56:72] = 1.0 
#     corrupted_images[:, :, 56:72, 56:72] = blurred_images[:, :, 56:72, 56:72]

#     t_start = torch.full((8,), start_step - 1, device=device, dtype=torch.long)
#     img = diffusion.q_sample(corrupted_images, t_start)
#     noisy_corrupted_display = img.clone() 

#     with torch.no_grad(): # Disable gradients for inference
#         for i in reversed(range(start_step)):
#             t_curr = torch.full((8,), i, device=device, dtype=torch.long)
            
#             model_pred = diffusion.p_sample(model, img, t_curr, labels=labels)
            
#             if i > 0:
#                 t_next = torch.full((8,), i - 1, device=device, dtype=torch.long)
#                 known_noisy_real = diffusion.q_sample(real_images, t_next)
#             else:
#                 known_noisy_real = real_images 

#             img = mask * model_pred + (1.0 - mask) * known_noisy_real

#     cleaned_images = img

#     accuracy = calculate_masked_pixel_accuracy(real_images, cleaned_images, mask)
#     print(f"--- Blur Ablation Results ---")
#     print(f"Noise Level: Step {start_step}/{diffusion.timesteps}")
#     print(f"Restoration Accuracy: {accuracy:.2f}%\n")

#     fig, axes = plt.subplots(4, 8, figsize=(16, 8))
#     fig.suptitle(f"Blur Cleanup | Noise Step {start_step} | Accuracy: {accuracy:.2f}%", fontsize=16)

#     row_titles = ["Original Image", " Blurred Image", f" Noised (t={start_step})", "4. AI Restored"]
#     plot_data = [real_images, corrupted_images, noisy_corrupted_display, cleaned_images]

#     for row_idx in range(4):
#         axes[row_idx, 0].set_ylabel(row_titles[row_idx], fontsize=12, rotation=0, labelpad=60, ha='center')
#         for col_idx in range(8):
#             disp_img = (plot_data[row_idx][col_idx].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
#             disp_img = np.clip(disp_img, 0, 1) 8
            
#             axes[row_idx, col_idx].imshow(disp_img)
#             axes[row_idx, col_idx].set_xticks([])
#             axes[row_idx, col_idx].set_yticks([])

#     plt.tight_layout()
#     save_path = os.path.join(save_dir, f"blur_cleanup_eval_t{start_step}.png")
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"Saved visualization to {save_path}")

# # Run the experiment starting at step 250
# test_artifact_cleanup_blur(model, gaussian_diffusion, test_loader, start_step=250)