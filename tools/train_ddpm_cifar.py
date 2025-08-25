import torch
import torchvision.datasets as datasets
from torchvision import transforms
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.unet_base import Unet
from scheduler.noise_scheduler import LinearNoiseScheduler, CosineNoiseScheduler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # # Create the noise scheduler
    # scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
    #                                  beta_start=diffusion_config['beta_start'],
    #                                  beta_end=diffusion_config['beta_end'])

    NoiseScheduler = train_config['noise_scheduler']
    # # Create the noise scheduler
    if NoiseScheduler ==  "linear":
        scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'], beta_start=diffusion_config['beta_start'], beta_end=diffusion_config['beta_end'])
    elif NoiseScheduler == "cosine":
        scheduler =  CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'])
    else:
        raise ValueError("Unknown scheduler name.")

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])

    train_dataset = datasets.CIFAR10(root="dataset/", train=True, transform= transform,  download=True)
    train_dataset_raw = datasets.CIFAR10(root="dataset/", train=True, download=True)
    # test_dataset = datasets.CIFAR10(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)

    print("Number of training images:", len(train_dataset))     # CIFAR: 50000
    print("Number of batches:", len(train_loader))  # num_of_images/ batch_size 50000/32


    ## visualize the progressive noise
    # visualize_noise_progression(scheduler, train_dataset, device, diffusion_config)

    logger = SummaryWriter(os.path.join("runs", "ddpm"))
    # Instantiate the model
    model = Unet(model_config).to(device)
    model.train()

    # # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])


    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im, lbl in tqdm(train_loader):
            optimizer.zero_grad()
            # im = im.float().to(device)
            im = im.to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        logger.add_scalar("MSE", np.mean(losses), global_step=epoch_idx)
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        # torch.save(model.state_dict(), os.path.join(train_config['task_name'],
        #                                             train_config['ckpt_name']))
        
        if (epoch_idx + 1) % 10 == 0:
              torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # optional
                    'loss': np.mean(losses)
                },  os.path.join(train_config['task_name'],train_config['ckpt_name']))

    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
