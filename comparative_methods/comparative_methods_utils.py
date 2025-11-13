import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import prox_tv
import pywt
from scipy.fftpack import dct, idct
import sys
sys.path.append('..')
from CryoGEN.core import measurement_operator


def tv_minimize_batch(target_measurements_batch, masks, img_size=128, lmbda=1e-3, max_iter=200, lr=1e-1, block_size=2):
    """
    Total Variation (TV) minimization-based image reconstruction.
    Performs iterative gradient-based optimization followed by TV denoising using `prox_tv`.
    Tries to reconstruct an image that matches the target measurements under a blockwise operator.

    Args:
        target_measurements_batch: List of measurement tensors for each mask.
        masks: Measurement masks used during sensing.
        img_size: Size of the reconstructed image (assumed square).
        lmbda: TV regularization strength.
        max_iter: Number of optimization iterations.
        lr: Learning rate for Adam optimizer.
        block_size: Block size used in the measurement operator.

    Returns:
        Tensor of reconstructed images with shape [B, 1, H, W].
    """
    batch_size = target_measurements_batch[0].shape[0]
    num_masks = len(target_measurements_batch)
    device = masks.device
    upsampled_images = torch.zeros((batch_size, 1, img_size, img_size), device=device)

    for b in range(batch_size):
        x = torch.randn((1, 1, img_size, img_size), requires_grad=True, device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            simulated = measurement_operator(x, masks, block_size)
            sim_tensor = torch.stack([m[0] for m in simulated], dim=0).float()
            target_tensor = torch.stack([target_measurements_batch[m][b] for m in range(num_masks)], dim=0).float()
            loss = F.mse_loss(sim_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_np = x.detach().cpu().numpy()[0, 0].astype(np.float64)
                x_np_tv = prox_tv.tv1_2d(x_np, lmbda)
                x.data = torch.tensor(x_np_tv, device=device).unsqueeze(0).unsqueeze(0)

        upsampled_images[b] = x.detach()

    return upsampled_images


def dct2d(x):
    """Applies 2D Discrete Cosine Transform (DCT) to the input."""
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2d(x):
    """Applies 2D Inverse Discrete Cosine Transform (IDCT) to the input."""
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def soft_threshold(x, thresh):
    """Applies soft-thresholding for proximal gradient algorithms."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def dct_sparse_minimize_batch(target_measurements_batch, masks, img_size=128, lmbda=1e-3, max_iter=200, lr=1e-1, block_size=2):
    """
    DCT-based sparse image reconstruction.
    Uses iterative optimization with DCT-domain soft-thresholding to promote sparsity.

    Args:
        Same as `tv_minimize_batch`

    Returns:
        Tensor of reconstructed images with shape [B, 1, H, W].
    """
    batch_size = target_measurements_batch[0].shape[0]
    num_masks = len(target_measurements_batch)
    device = masks.device
    upsampled_images = torch.zeros((batch_size, 1, img_size, img_size), device=device)

    for b in range(batch_size):
        x = torch.randn((1, 1, img_size, img_size), requires_grad=True, device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            simulated = measurement_operator(x, masks, block_size)
            sim_tensor = torch.stack([m[0] for m in simulated], dim=0).float()
            target_tensor = torch.stack([target_measurements_batch[m][b] for m in range(num_masks)], dim=0).float()
            loss = F.mse_loss(sim_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_np = x.detach().cpu().numpy()[0, 0].astype(np.float64)
                dct_coeff = dct2d(x_np)
                dct_thresh = soft_threshold(dct_coeff, lmbda)
                x_np_denoised = idct2d(dct_thresh)
                x.data = torch.tensor(x_np_denoised, device=device).unsqueeze(0).unsqueeze(0)

        upsampled_images[b] = x.detach()

    return upsampled_images


def wavelet_decompose(x, wavelet='db1', level=None):
    """Performs 2D wavelet decomposition and flattens the result into a single array."""
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)
    coeffs_flat, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeffs_flat, coeff_slices

def wavelet_reconstruct(coeffs_flat, coeff_slices, wavelet='db1'):
    """Reconstructs 2D data from flattened wavelet coefficients."""
    coeffs = pywt.array_to_coeffs(coeffs_flat, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet=wavelet)

def wavelet_sparse_minimize_batch(target_measurements_batch, masks, img_size=128, lmbda=1e-3, max_iter=200, lr=1e-1, wavelet='db1', level=None, block_size=2):
    """
    Wavelet-based sparse image reconstruction.
    Promotes sparsity in the wavelet domain using soft-thresholding.

    Args:
        Same as `tv_minimize_batch`, plus:
        wavelet: Name of the wavelet filter to use.
        level: Number of decomposition levels.

    Returns:
        Tensor of reconstructed images with shape [B, 1, H, W].
    """
    batch_size = target_measurements_batch[0].shape[0]
    num_masks = len(target_measurements_batch)
    device = masks.device
    upsampled_images = torch.zeros((batch_size, 1, img_size, img_size), device=device)

    for b in range(batch_size):
        x = torch.randn((1, 1, img_size, img_size), requires_grad=True, device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            simulated = measurement_operator(x, masks, block_size)
            sim_tensor = torch.stack([m[0] for m in simulated], dim=0).float()
            target_tensor = torch.stack([target_measurements_batch[m][b] for m in range(num_masks)], dim=0).float()
            loss = F.mse_loss(sim_tensor, target_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_np = x.detach().cpu().numpy()[0, 0].astype(np.float64)
                coeffs_flat, coeff_slices = wavelet_decompose(x_np, wavelet, level)
                coeffs_thresh = soft_threshold(coeffs_flat, lmbda)
                x_np_denoised = wavelet_reconstruct(coeffs_thresh, coeff_slices, wavelet)
                x.data = torch.tensor(x_np_denoised, device=device).unsqueeze(0).unsqueeze(0)

        upsampled_images[b] = x.detach()

    return upsampled_images


def dmplug_batch(model, target_measurements_batch, masks, img_size=128, max_iter=1000, lr=1e-2, block_size=2):
    """
    Diffusion-based reconstruction using DDIM (DMPlug).

    Args:
        model: A diffusion model implementing the `sample = model(Z, t).sample` interface.
        target_measurements_batch: List of target measurement tensors.
        masks: Binary masks used for measurements.
        img_size: Size of the latent image.
        max_iter: Number of gradient optimization steps.
        lr: Learning rate for optimization.
        block_size: Block size used in measurement operator.

    Returns:
        Reconstructed tensor `x_t` representing the final denoised image.
    """
    device = masks.device
    channels = model.config.in_channels
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(3)

    batch = torch.stack(target_measurements_batch, dim=0).permute(1, 0, 2, 3, 4)
    num_images = batch.shape[0]
    y_n = batch.reshape(-1, channels, batch.shape[3], batch.shape[4]).to(device)
    y_n.requires_grad = False

    Z = torch.randn((num_images, channels, img_size, img_size), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([{'params': Z, 'lr': lr}])
    criterion = torch.nn.MSELoss().to(device)

    pbar = tqdm(range(max_iter))
    for t in pbar:
        model.eval()
        optimizer.zero_grad()
        x_current = Z
        for tt in scheduler.timesteps:
            t_i = torch.full((Z.shape[0],), tt, device=device, dtype=torch.long)
            noise_pred = model(x_current, t_i).sample
            x_current = scheduler.step(noise_pred, tt, x_current).prev_sample
        x_t = x_current
        pred = measurement_operator(x_t, masks, block_size)
        pred_stack = torch.stack(pred, dim=1)
        loss = criterion(pred_stack, batch)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Step {t} - Loss {loss.item():.4f}")

    return x_t.detach()


"""
Diffusion functions
"""
# This code is from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, deprecate
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        next_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t+1}) of previous timestep. `next_sample` should be used as next model input in the
            reverse denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: Optional[torch.FloatTensor] = None
    next_sample: Optional[torch.FloatTensor] = None
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2010.02502

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.config.steps_offset
        
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): TODO
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device if torch.is_tensor(model_output) else "cpu"
            noise = torch.randn(model_output.shape, generator=generator).to(device)
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def reverse_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): TODO
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        next_timestep = min(self.config.num_train_timesteps - 2,
                            timestep + self.config.num_train_timesteps // self.num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_next = self.alphas_cumprod[next_timestep] if next_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. TODO: simple noising implementatiom
        next_sample = self.add_noise(pred_original_sample,
                                     model_output,
                                     torch.LongTensor([next_timestep]))

        # # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self._get_variance(next_timestep, timestep)
        # std_dev_t = eta * variance ** (0.5)

        # if use_clipped_model_output:
        #     # the model_output is always re-derived from the clipped x_0 in Glide
        #     model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_next - std_dev_t**2) ** (0.5) * model_output

        # # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # next_sample = alpha_prod_t_next ** (0.5) * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (next_sample,)

        return DDIMSchedulerOutput(next_sample=next_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        if self.alphas_cumprod.device != original_samples.device:
            self.alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        if timesteps.device != original_samples.device:
            timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
