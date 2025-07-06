# %%
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
  def __init__(self, latent_dim=20):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim
    # Encoder
    self.enc_fc1 = nn.Linear(28 * 28, 400)
    self.enc_fc2_mu = nn.Linear(400, latent_dim)
    self.enc_fc2_logvar = nn.Linear(400, latent_dim)
    # Decoder
    self.dec_fc1 = nn.Linear(latent_dim, 400)
    self.dec_fc2 = nn.Linear(400, 28 * 28)

  def encoder(self, x):
    x = x.view(-1, 28 * 28)
    h1 = F.relu(self.enc_fc1(x))
    mu = self.enc_fc2_mu(h1)
    logvar = self.enc_fc2_logvar(h1)
    # For inference, just return mu (mean of latent)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decoder(self, z):
    h3 = F.relu(self.dec_fc1(z))
    return torch.sigmoid(self.dec_fc2(h3)).view(-1, 28, 28)

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    return self.decoder(z), mu, logvar


class SignMNIST(torch.utils.data.Dataset):
  def __init__(self, path: Path):
    self.df = pd.read_csv(path)
    self.target: np.array = self.df["label"].values
    self.data = self.df.drop(columns=["label"]).values
    self.data = self.data.reshape(self.data.shape[0], 28, 28)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: list[int]) -> tuple:
    image = self.data[index]  # shape (28, 28)

    image = image.astype(np.float32) / 255.0  # min max Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add channel dimension (C x H x W)

    label = self.target[index]
    label = torch.tensor(label, dtype=torch.int64)

    return image, label

  @staticmethod
  def label_to_letter(label: int) -> str:
    return chr(label + 65)

  @property
  def shape(self) -> tuple:
    return self.data.shape, self.target.shape


def interpolate_latent_space(model, z1, z2, steps=10):
  """
  Linearly interpolate between two latent vectors z1 and z2.
  Returns a list of interpolated latent vectors.
  """
  return [z1 * (1 - t) + z2 * t for t in np.linspace(0, 1, steps)]


def visualize_interpolation(model, z1, z2, steps=10, device="cpu"):
  """
  Interpolates between z1 and z2 in latent space and visualizes the decoded outputs.
  """
  model.eval()
  interpolated = interpolate_latent_space(model, z1, z2, steps)
  decoded_imgs = []
  with torch.no_grad():
    for z in interpolated:
      z = z.to(device)
      recon = model.decoder(z.unsqueeze(0))
      decoded_imgs.append(recon.cpu().squeeze().numpy())
  # Plot the results
  fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
  for i, img in enumerate(decoded_imgs):
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
  plt.show()


# Example usage:
# z1 = model.encoder(torch.tensor(img1).unsqueeze(0).to(device))[0].squeeze()
# z2 = model.encoder(torch.tensor(img2).unsqueeze(0).to(device))[0].squeeze()
# visualize_interpolation(model, z1, z2, steps=10, device=device)

if __name__ == "__main__":
  from torchvision import datasets, transforms
  from torch.utils.data import DataLoader
  from pyprojroot import here

  # Download dataset if not already present
  if not (here(r"data/sign_language_mnist/sign_mnist_train.csv").exists()):
    import kagglehub

    dwnl_path: str = kagglehub.dataset_download(
      "datamunge/sign-language-mnist",
    )
    # move the downloaded files to the correct location
    import shutil

    dest_dir = here(r"data/sign_language_mnist")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(
      Path(dwnl_path) / "sign_mnist_train.csv", dest_dir / "sign_mnist_train.csv"
    )
    shutil.move(
      Path(dwnl_path) / "sign_mnist_test.csv", dest_dir / "sign_mnist_test.csv"
    )
  # Load train and test set
  np.random.seed(1)
  torch.manual_seed(1)
  train_loader = DataLoader(
    train := SignMNIST(here(r"data/sign_language_mnist/sign_mnist_train.csv")),
    batch_size=64,
    shuffle=True,
  )
  test_loader = DataLoader(
    test := SignMNIST(here(r"data/sign_language_mnist/sign_mnist_test.csv")),
    batch_size=64,
    shuffle=True,
  )

  # Define and load the VAE model
  latent_dim = 20
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = VAE(latent_dim=latent_dim).to(device)

  # Training function
  def loss_function(recon_x, x, mu, logvar):
    # Flatten recon_x to match x.view(-1, 28*28)
    recon_x = recon_x.view(-1, 28 * 28)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction="sum")
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

  import os
  from pyprojroot import here

  if os.path.exists(here("models/vae_signmnist.pth")):
    model.load_state_dict(
      torch.load(here("models/vae_signmnist.pth"), map_location=device)
    )
    print(f"Loaded VAE weights from {here('models/vae_signmnist.pth')}")
  else:
    print("Training VAE from scratch...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    epochs = 5  # For demonstration; increase for better results
    for epoch in range(epochs):
      train_loss = 0
      for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
      print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")
    torch.save(model.state_dict(), here("models/vae_signmnist.pth"))
    print(f"Saved trained VAE to {here('models/vae_signmnist.pth')}")

  model.eval()

  # Get two images from test set with different classes
  data_iter = iter(test_loader)
  found = False
  while not found:
    imgs, labels = next(data_iter)
    if labels[0] != labels[1]:
      img1, img2 = imgs[0], imgs[1]
      found = True

  img1, img2 = img1.to(device), img2.to(device)
  with torch.no_grad():
    z1 = model.encoder(img1.unsqueeze(0))[0].squeeze()
    z2 = model.encoder(img2.unsqueeze(0))[0].squeeze()
  visualize_interpolation(model, z1, z2, steps=10, device=device)

  print(
    "Loaded two SignMNIST images for interpolation example. VAE model is trained and loaded."
  )
# %%
