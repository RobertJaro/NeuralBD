import argparse
import os

import torch
import yaml
from torch import nn
from tqdm import tqdm

from nbd.data_module import NeuralBDDataModule
from nbd.model import ImageModel


def pretrain_on_mean_image(image_model, mean_image, coords,
                           epochs=300, batch_size=1024, lr=1e-4):
    """
    Pretrain the ImageModel on the mean image from the datamodule.

    Args:
        image_model: instance of ImageModel
        mean_image: numpy array or torch tensor of shape [H, W, C]
        coords: numpy array or torch tensor of shape [H, W, 2] with coordinates
        epochs: number of epochs
        batch_size: batch size
        lr: learning rate
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    image_model = image_model.to(device)
    optimizer = torch.optim.Adam(image_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert to tensors if necessary
    if not isinstance(mean_image, torch.Tensor):
        mean_image = torch.tensor(mean_image, dtype=torch.float32)
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)

    mean_image = mean_image.to(device)
    coords = coords.to(device)

    # Flatten coordinates and pixels
    H, W, C = mean_image.shape
    coords_flat = coords.view(-1, 2)
    pixels_flat = mean_image.view(-1, C)

    for epoch in tqdm(range(epochs)):
        perm = torch.randperm(coords_flat.size(0))
        epoch_loss = 0.0

        for i in range(0, coords_flat.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_coords = coords_flat[idx]
            batch_pixels = pixels_flat[idx]

            optimizer.zero_grad()
            output = image_model(batch_coords)
            loss_val = criterion(output, batch_pixels)
            loss_val.backward()
            optimizer.step()

            epoch_loss += loss_val.item() * batch_coords.size(0)

        epoch_loss /= coords_flat.size(0)
        if epoch % 10 == 0:
            print(f"[Mean Image Pretrain] Epoch {epoch}, Loss: {epoch_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, overwrite_args = parser.parse_known_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    base_dir = config['base_dir']
    os.makedirs(base_dir, exist_ok=True)

    data_config = config['data']
    data_module = NeuralBDDataModule(**data_config)

    save_path = config['base_dir']

    image = data_module.image_mean
    coords = data_module.img_coords

    image_model = ImageModel()
    pretrain_on_mean_image(image_model, image, coords)
    torch.save(image_model.state_dict(), save_path+'/meta_model.pth')
    print(f"Pretrained ImageModel saved to {save_path}")