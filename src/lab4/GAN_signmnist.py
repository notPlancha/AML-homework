# %%
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
  def __init__(self, latent_dim=100):
    super(Generator, self).__init__()
    self.latent_dim = latent_dim
    self.main = nn.Sequential(
      nn.Linear(latent_dim, 256),
      nn.ReLU(True),
      nn.Linear(256, 512),
      nn.ReLU(True),
      nn.Linear(512, 784),
      nn.Tanh(),
    )

  def forward(self, z):
    return self.main(z).view(-1, 28, 28)


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      nn.Linear(784, 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = x.view(-1, 784)
    return self.main(x)


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


def interpolate_latent_space(generator, z1, z2, steps=10):
  """
  Linearly interpolate between two latent vectors z1 and z2.
  Returns a list of interpolated latent vectors.
  """
  return [z1 * (1 - t) + z2 * t for t in np.linspace(0, 1, steps)]


def visualize_interpolation(generator, z1, z2, steps=10, device="cpu"):
  """
  Interpolates between z1 and z2 in latent space and visualizes the generated outputs.
  """
  generator.eval()
  interpolated = interpolate_latent_space(generator, z1, z2, steps)
  generated_imgs = []
  with torch.no_grad():
    for z in interpolated:
      z = z.to(device)
      gen_img = generator(z.unsqueeze(0))
      generated_imgs.append(gen_img.cpu().squeeze().numpy())
  # Plot the results
  fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
  for i, img in enumerate(generated_imgs):
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
  plt.show()


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

  # Define and load the GAN model
  latent_dim = 100
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  generator = Generator(latent_dim=latent_dim).to(device)
  discriminator = Discriminator().to(device)

  # Loss function
  criterion = nn.BCELoss()

  import os
  from pyprojroot import here

  if os.path.exists(here("models/gan_generator_signmnist.pth")) and os.path.exists(
    here("models/gan_discriminator_signmnist.pth")
  ):
    generator.load_state_dict(
      torch.load(here("models/gan_generator_signmnist.pth"), map_location=device)
    )
    discriminator.load_state_dict(
      torch.load(here("models/gan_discriminator_signmnist.pth"), map_location=device)
    )
    print(f"Loaded GAN weights from {here('models/')}")
  else:
    print("Training GAN from scratch...")
    optimizer_G = torch.optim.Adam(
      generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
      discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    epochs = 50
    for epoch in range(epochs):
      for batch_idx, (real_data, _) in enumerate(train_loader):
        batch_size = real_data.size(0)
        real_data = real_data.to(device)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real data
        real_output = discriminator(real_data)
        d_loss_real = criterion(real_output, real_labels)

        # Fake data
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()

      print(
        f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
      )

    # Save models
    os.makedirs(here("models"), exist_ok=True)
    torch.save(generator.state_dict(), here("models/gan_generator_signmnist.pth"))
    torch.save(
      discriminator.state_dict(), here("models/gan_discriminator_signmnist.pth")
    )
    print(f"Saved trained GAN to {here('models/')}")

  generator.eval()
  discriminator.eval()

  # Generate two random latent vectors for interpolation
  z1 = torch.randn(latent_dim).to(device)
  z2 = torch.randn(latent_dim).to(device)

  visualize_interpolation(generator, z1, z2, steps=10, device=device)

  print(
    "GAN model is trained and loaded. Generated interpolation between random latent vectors."
  )
