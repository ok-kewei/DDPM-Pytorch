import torch


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)


    @torch.no_grad()
    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Sample x_{t-1} given x_t and predicted noise.

        Args:
            xt: [B, C, H, W] current noisy image at timestep t
            noise_pred: [B, C, H, W] predicted noise (epsilon_theta)
            t: int or 0-dim tensor, current timestep

        Returns:
            xt_prev: [B, C, H, W] sampled x_{t-1}
            x0_pred: [B, C, H, W] predicted x_0
        """
        B = xt.size(0)

        # Ensure t is a Python integer
        if torch.is_tensor(t):
            t = int(t.item()) if t.numel() == 1 else int(t[0].item())

        # Create uniform timestep tensor for the batch
        t_batch = torch.full((B,), t, device=xt.device, dtype=torch.long)
        # print("t_batch:", t_batch.shape, "device:", t_batch.device)



        # Move alpha/beta tensors to same device and reshape for broadcasting
        alpha_t = self.alphas.to(xt.device)[t_batch].view(B, 1, 1, 1)
        alpha_cumprod_t = self.alpha_cum_prod.to(xt.device)[t_batch].view(B, 1, 1, 1)
        alpha_cumprod_prev_t = self.alpha_cum_prod.to(xt.device)[torch.clamp(t_batch - 1, 0)].view(B, 1, 1, 1)
        beta_t = self.betas.to(xt.device)[t_batch].view(B, 1, 1, 1)

        # print("xt:", xt.shape, "device:", xt.device)
        # print("noise_pred:", noise_pred.shape, "device:", noise_pred.device)
        # print("alpha_t:", alpha_t.shape, "device:", alpha_t.device)
        # print("alpha_cumprod_t:", alpha_cumprod_t.shape, "device:", alpha_cumprod_t.device)
        # print("alpha_cumprod_prev_t:", alpha_cumprod_prev_t.shape, "device:", alpha_cumprod_prev_t.device)
        # print("beta_t:", beta_t.shape, "device:", beta_t.device)

        # Predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        x0_pred = torch.clamp(x0_pred, -1., 1.).to(xt.device)



        # Posterior mean and variance (DDPM formulas)
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t) * x0_pred +
            torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * xt
        )
        posterior_var = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        # Sample noise (no noise at t=0)
        noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
        xt_prev = posterior_mean + torch.sqrt(posterior_var) * noise


        # print("xt range:", xt.min().item(), xt.max().item())
        # print("x0_pred range:", x0_pred.min().item(), x0_pred.max().item())
        # print("posterior_mean range:", posterior_mean.min().item(), posterior_mean.max().item())
        # print("posterior_var range:", posterior_var.min().item(), posterior_var.max().item())

        return xt_prev, x0_pred

    
#     # def sample_prev_timestep(self, xt, noise_pred, t):
#     #     r"""
#     #         Use the noise prediction by model to get
#     #         xt-1 using xt and the noise predicted
#     #     :param xt: current timestep sample
#     #     :param noise_pred: model noise prediction
#     #     :param t: current timestep we are at
#     #     :return:
#     #     """
#     #     x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
#     #           torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
#     #     x0 = torch.clamp(x0, -1., 1.)

#     #     mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
#     #     mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

#     #     if t == 0:
#     #         return mean, x0
#     #     else:
#     #         variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
#     #         variance = variance * self.betas.to(xt.device)[t]
#     #         sigma = variance ** 0.5
#     #         z = torch.randn(xt.shape).to(xt.device)

#     #         # OR
#     #         # variance = self.betas[t]
#     #         # sigma = variance ** 0.5
#     #         # z = torch.randn(xt.shape).to(xt.device)
#     #         return mean + sigma * z, x0



class CosineNoiseScheduler:
    r"""
    Cosine noise schedule as proposed by Nichol & Dhariwal (2021).
    """

    def __init__(self, num_timesteps, s=0.008):
        self.num_timesteps = num_timesteps

        # Calculate alphā_t using the cosine schedule
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        t = steps / num_timesteps
        f_t = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]  # Normalize to start at 1.0

        # Compute betas from alpha_bar
        betas = []
        for i in range(1, len(alpha_bar)):
            beta = min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999)
            betas.append(beta)
        self.betas = torch.tensor(betas, dtype=torch.float32)

        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0) #bar_alpha[t] = alpha_1 * alpha_2 * ... * alpha_t
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        batch_size = original.size(0)
        original_shape = original.shape

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    @torch.no_grad()
    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Sample x_{t-1} given x_t and predicted noise.

        Args:
            xt: [B, C, H, W] current noisy image at timestep t
            noise_pred: [B, C, H, W] predicted noise (epsilon_theta)
            t: int or 0-dim tensor, current timestep

        Returns:
            xt_prev: [B, C, H, W] sampled x_{t-1}
            x0_pred: [B, C, H, W] predicted x_0
        """
        B = xt.size(0)

        # Ensure t is a Python integer
        if torch.is_tensor(t):
            t = int(t.item()) if t.numel() == 1 else int(t[0].item())

        # Create uniform timestep tensor for the batch
        t_batch = torch.full((B,), t, device=xt.device, dtype=torch.long)
        # print("t_batch:", t_batch.shape, "device:", t_batch.device)



        # Move alpha/beta tensors to same device and reshape for broadcasting
        alpha_t = self.alphas.to(xt.device)[t_batch].view(B, 1, 1, 1)
        alpha_cumprod_t = self.alpha_cum_prod.to(xt.device)[t_batch].view(B, 1, 1, 1)
        alpha_cumprod_prev_t = self.alpha_cum_prod.to(xt.device)[torch.clamp(t_batch - 1, 0)].view(B, 1, 1, 1)
        beta_t = self.betas.to(xt.device)[t_batch].view(B, 1, 1, 1)

        # print("xt:", xt.shape, "device:", xt.device)
        # print("noise_pred:", noise_pred.shape, "device:", noise_pred.device)
        # print("alpha_t:", alpha_t.shape, "device:", alpha_t.device)
        # print("alpha_cumprod_t:", alpha_cumprod_t.shape, "device:", alpha_cumprod_t.device)
        # print("alpha_cumprod_prev_t:", alpha_cumprod_prev_t.shape, "device:", alpha_cumprod_prev_t.device)
        # print("beta_t:", beta_t.shape, "device:", beta_t.device)

        # Predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        x0_pred = torch.clamp(x0_pred, -1., 1.).to(xt.device)



        # Posterior mean and variance (DDPM formulas)
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t) * x0_pred +
            torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * xt
        )
        posterior_var = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

        # Sample noise (no noise at t=0)
        noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
        xt_prev = posterior_mean + torch.sqrt(posterior_var) * noise


        # print("xt range:", xt.min().item(), xt.max().item())
        # print("x0_pred range:", x0_pred.min().item(), x0_pred.max().item())
        # print("posterior_mean range:", posterior_mean.min().item(), posterior_mean.max().item())
        # print("posterior_var range:", posterior_var.min().item(), posterior_var.max().item())

        return xt_prev, x0_pred










  


def visualize_noise_progression(scheduler, dataset, device, diffusion_config):
    import matplotlib.pyplot as plt

    sample_image, _ = dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)
    sample_noise = torch.randn_like(sample_image).to(device)

    timesteps = [0, 50, 100, 200, 300, 400, 500, 600, 700,800, 900, diffusion_config['num_timesteps'] - 1]
    noisy_imgs = []
    for t_val in timesteps:
        t_tensor = torch.tensor([t_val], device=device)
        noisy = scheduler.add_noise(sample_image, sample_noise, t_tensor)
        noisy_imgs.append(noisy.squeeze(0).cpu())

    plt.figure(figsize=(15, 3))
    for i, noisy_img in enumerate(noisy_imgs):
        plt.subplot(1, len(timesteps), i + 1)
        plt.imshow(noisy_img.squeeze(), cmap='gray')
        plt.title(f"t = {timesteps[i]}")
        plt.axis('off')
    plt.suptitle("Noise Progression Across Timesteps", fontsize=14)
    plt.tight_layout()
    plt.show()






        
