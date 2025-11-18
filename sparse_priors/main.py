import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from CryoSENSE.data import load_cryoem_batch
from CryoSENSE.masks import create_binary_masks
from CryoSENSE.core import measurement_operator
from CryoSENSE.evaluation import analyze_reconstruction
from sparse_utils import tv_minimize_batch, dct_sparse_minimize_batch, wavelet_sparse_minimize_batch, dmplug_batch


def main():
    parser = argparse.ArgumentParser(description='Run prior reconstructions')
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument('--num_masks', type=int, default=64)
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--mask_type', type=str, default="random_binary")
    parser.add_argument('--cryoem_path', type=str, required=True)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_cryoem', action='store_true')
    parser.add_argument('--result_dir', type=str, default="results")
    parser.add_argument('--prior', type=str, default="dct")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--lambda_', type=float, default=1e-1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--noise_level', type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Results will be saved to: {args.result_dir}")

    if args.model == "":
        pass
    else:
        pipeline = DDPMPipeline.from_pretrained(args.model).to("cuda")
        model = pipeline.unet

    img_size = args.img_size
    in_channels = args.in_channels

    masks = create_binary_masks(
        num_masks=args.num_masks,
        mask_prob=args.mask_prob,
        mask_type=args.mask_type,
        img_size=img_size,
    )

    if args.use_cryoem:
        total_images = args.end_id - args.start_id + 1
        num_batches = math.ceil(total_images / args.batch_size)

        for batch_idx in range(num_batches):
            batch_start = args.start_id + batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, args.end_id + 1)
            batch_ids = list(range(batch_start, batch_end))
            current_batch_size = len(batch_ids)

            print(f"\nBatch {batch_idx+1}/{num_batches}: IDs {batch_ids}")
            image_batch = load_cryoem_batch(
                args.cryoem_path, batch_ids, img_size, "cuda"
            )

            target_measurements_batch = measurement_operator(image_batch, masks, args.block_size)

            print('NOISE LEVEL', args.noise_level)
            if args.noise_level > 0:
                noise = [torch.randn_like(m) * args.noise_level for m in target_measurements_batch]
                target_measurements_batch = [m + n for m, n in zip(target_measurements_batch, noise)]

            if args.prior == "tv_minimize":
                output = tv_minimize_batch(
                    target_measurements_batch, masks,
                    lr=args.learning_rate, lmbda=args.lambda_, max_iter=args.num_epochs, block_size=args.block_size, img_size=img_size
                )
            elif args.prior == "dct":
                output = dct_sparse_minimize_batch(
                    target_measurements_batch, masks,
                    lr=args.learning_rate, lmbda=args.lambda_, max_iter=args.num_epochs, block_size=args.block_size, img_size=img_size
                )
            elif args.prior == "wavelet":
                output = wavelet_sparse_minimize_batch(
                    target_measurements_batch, masks,
                    lr=args.learning_rate, lmbda=args.lambda_, max_iter=args.num_epochs, block_size=args.block_size, img_size=img_size
                )
            elif args.prior == "dmplug":
                output = dmplug_batch(
                    model, target_measurements_batch, masks, img_size, max_iter=1000, block_size=args.block_size, img_size=img_size
                )
            else:
                raise ValueError(f"Unknown prior: {args.prior}")

            analyze_reconstruction(
                image_batch,
                output,
                target_measurements_batch,
                masks,
                current_batch_ids=batch_ids,
                result_dir=args.result_dir,
                experiment_params={
                    'mask_type': args.mask_type,
                    'noise_level': args.noise_level,
                    'num_masks': args.num_masks
                },
                block_size=args.block_size
            )

            for i, image_id in enumerate(batch_ids):
                recon_vis = (output[i:i+1] + 1) / 2  # [0, 1] range for plotting

                # Save visualization as PNG
                plt.figure(figsize=(8, 8), frameon=False)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                if args.in_channels == 1:
                    plt.imshow(recon_vis[0, 0].cpu().numpy(), cmap='gray')
                else:
                    plt.imshow(recon_vis[0, :min(3, args.in_channels)].permute(1, 2, 0).cpu().numpy())

                plt.axis('off')
                plt.savefig(f"{args.result_dir}/reconstructed_image_{image_id}.png", bbox_inches='tight', pad_inches=0)
                plt.close()

                # Save tensor
                torch.save(output[i:i+1].cpu(), f"{args.result_dir}/reconstruction_raw_image_{image_id}.pt")

            del image_batch, target_measurements_batch, output
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
