import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision
import argparse
import yaml
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.noise_scheduler import LinearNoiseScheduler
from scheduler.noise_scheduler import CosineNoiseScheduler


torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unnormalize(img_tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean

def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        # noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        timestep = torch.full((xt.size(0),), i, device=device, dtype=torch.long) # return shape (num_samples,)
        noise_pred = model(xt, timestep)

        # Use scheduler to get x0 and xt-1
        # xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, timestep)



        # Save x0
        # ims = unnormalize(x0_pred)
        # ims = ims.clamp(0,1).detach().cpu()
        # # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # # ims = (ims + 1) / 2
        # grid = make_grid(ims, nrow=train_config['num_grid_rows'])

        ims = unnormalize(x0_pred)
        ims = ims.clamp(0, 1).detach().cpu().float()  # force float32
        grid = make_grid(ims, nrow=ims.size(0), padding=0)

        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
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
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    # model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                               train_config['ckpt_name']), map_location=device))

    checkpoint = torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # NoiseScheduler = train_config['noise_scheduler']
    NoiseScheduler = train_config.get('noise_scheduler', 'linear').lower()
    # # Create the noise scheduler
    if NoiseScheduler ==  "linear":
        scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'], beta_start=diffusion_config['beta_start'], beta_end=diffusion_config['beta_end'])
    elif NoiseScheduler == "cosine":
        scheduler =  CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'])
    else:
        raise ValueError("Unknown scheduler name.")

    # # Create the noise scheduler
    # scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
    #                                  beta
    #                                  _start=diffusion_config['beta_start'],
    #                                  beta_end=diffusion_config['beta_end'])


    # Create the noise scheduler
    # scheduler = CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'])

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
