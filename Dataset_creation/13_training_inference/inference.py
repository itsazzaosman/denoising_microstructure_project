import torch
import torchvision.transforms as transforms
from PIL import Image
from architecture import EBSD_VAE

def denoise_map(model_path, noisy_image_path, output_path):
    model = EBSD_VAE()
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    model.cuda()
    
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    
    image = Image.open(noisy_image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).cuda() 
    
    with torch.no_grad():
        cleaned_tensor, _, _ = model(input_tensor)
    
    cleaned_tensor = cleaned_tensor.squeeze(0).cpu() 
    cleaned_image = transforms.ToPILImage()(cleaned_tensor)
    cleaned_image.save(output_path)
    print(f"Cleaned map saved to {output_path}")

if __name__ == "__main__":
    # denoise_map("ebsd_vae_weights.pth", "test_noisy.png", "test_clean.png")
    pass