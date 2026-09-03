import torch
import torch.optim as optim
from architecture import EBSD_VAE, vae_loss_function
from torch.utils.data import DataLoader
from dataset import EBSD_Dataset

def train_vae(model, dataloader, epochs=50, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train() 

    for epoch in range(epochs):
        total_loss = 0
        
        for noisy_batch, clean_batch in dataloader:
            noisy_batch = noisy_batch.cuda() 
            clean_batch = clean_batch.cuda()

            optimizer.zero_grad()
            reconstructed, mu, logvar = model(noisy_batch)
            
            loss = vae_loss_function(reconstructed, clean_batch, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "ebsd_vae_weights.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    # Dataset lives outside the git repo - see 13_training_inference/README.md
    CLEAN_DIR = "/project/community/aiosman/datasets/ni_vae/clean"
    NOISY_DIR = "/project/community/aiosman/datasets/ni_vae/noisy"
    
    print("Loading dataset...")
    dataset = EBSD_Dataset(CLEAN_DIR, NOISY_DIR)
    
    # Batch size of 64 or 128 is usually optimal for 128x128 images on a standard GPU
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    print("Initializing model...")
    model = EBSD_VAE().cuda()
    
    print("Starting training...")
    train_vae(model, dataloader, epochs=50)