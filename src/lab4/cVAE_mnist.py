# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class cVAE(nn.Module):
  def __init__(self, latent_dim=20, n_classes=10):
    super(cVAE, self).__init__()
    self.latent_dim = latent_dim
    self.n_classes = n_classes
    # Encoder: input is image + one-hot label
    self.enc_fc1 = nn.Linear(28 * 28 + n_classes, 400)
    self.enc_fc2_mu = nn.Linear(400, latent_dim)
    self.enc_fc2_logvar = nn.Linear(400, latent_dim)
    # Decoder: input is latent + one-hot label
    self.dec_fc1 = nn.Linear(latent_dim + n_classes, 400)
    self.dec_fc2 = nn.Linear(400, 28 * 28)

  def encoder(self, x, y):
    # x: [batch, 1, 28, 28], y: [batch, n_classes]
    x = x.view(-1, 28 * 28)
    h = torch.cat([x, y], dim=1)
    h1 = F.relu(self.enc_fc1(h))
    mu = self.enc_fc2_mu(h1)
    logvar = self.enc_fc2_logvar(h1)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decoder(self, z, y):
    # z: [batch, latent_dim], y: [batch, n_classes]
    h = torch.cat([z, y], dim=1)
    h3 = F.relu(self.dec_fc1(h))
    return torch.sigmoid(self.dec_fc2(h3)).view(-1, 28, 28)

  def forward(self, x, y):
    mu, logvar = self.encoder(x, y)
    z = self.reparameterize(mu, logvar)
    return self.decoder(z, y), mu, logvar


def interpolate_latent_space(model, z1, z2, steps=10):
  return [z1 * (1 - t) + z2 * t for t in np.linspace(0, 1, steps)]


def visualize_interpolation(model, z1, z2, y, steps=10, device="cpu"):
  """
  Interpolates between z1 and z2 in latent space and visualizes the decoded outputs.
  y: one-hot label vector (same for all interpolations)
  """
  model.eval()
  interpolated = interpolate_latent_space(model, z1, z2, steps)
  decoded_imgs = []
  with torch.no_grad():
    for z in interpolated:
      z = z.to(device)
      recon = model.decoder(z.unsqueeze(0), y.unsqueeze(0))
      decoded_imgs.append(recon.cpu().squeeze().numpy())
  fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
  for i, img in enumerate(decoded_imgs):
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
  plt.show()


if __name__ == "__main__":
  from torchvision import datasets, transforms
  from torch.utils.data import DataLoader
  from pyprojroot import here

  # Load MNIST train and test set
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
    ]
  )
  mnist_train = datasets.MNIST(
    root=here("data"), train=True, download=True, transform=transform
  )
  mnist_test = datasets.MNIST(
    root=here("data"), train=False, download=True, transform=transform
  )
  train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
  test_loader = DataLoader(mnist_test, batch_size=2, shuffle=True)

  latent_dim = 20
  n_classes = 10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = cVAE(latent_dim=latent_dim, n_classes=n_classes).to(device)

  def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.view(-1, 28 * 28)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

  import os
  from pyprojroot import here

  model_path = here("models/cvae_mnist.pth")
  if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded cVAE weights from {model_path}")
  else:
    print("Training cVAE from scratch...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    epochs = 5
    for epoch in range(epochs):
      train_loss = 0
      for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        # Convert targets to one-hot
        y = F.one_hot(targets, num_classes=n_classes).float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, y)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
      print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained cVAE to {model_path}")

  model.eval()

  # Get two images from test set with different classes
  data_iter = iter(test_loader)
  found = False
  while not found:
    imgs, labels = next(data_iter)
    if labels[0] != labels[1]:
      img1, img2 = imgs[0], imgs[1]
      label1, label2 = labels[0], labels[1]
      found = True

  img1, img2 = img1.to(device), img2.to(device)
  y1 = (
    F.one_hot(label1.unsqueeze(0), num_classes=n_classes).float().to(device).squeeze(0)
  )
  y2 = (
    F.one_hot(label2.unsqueeze(0), num_classes=n_classes).float().to(device).squeeze(0)
  )

  with torch.no_grad():
    z1 = model.encoder(img1.unsqueeze(0), y1.unsqueeze(0))[0].squeeze()
    z2 = model.encoder(img2.unsqueeze(0), y2.unsqueeze(0))[0].squeeze()
  # For visualization, pick one label to condition on (e.g., y1)
  visualize_interpolation(model, z1, z2, y1, steps=10, device=device)

  print(
    "Loaded two MNIST images for interpolation example. cVAE model is trained and loaded."
  )
