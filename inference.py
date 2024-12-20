import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision.utils  import make_grid
from torchvision import datasets, transforms

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_start=args.beta_start, 
        beta_end=args.beta_end, beta_schedule=args.beta_schedule, variance_type=args.variance_type, 
        prediction_type=args.prediction_type, clip_sample=args.clip_sample, clip_sample_range=args.clip_sample_range
    )
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(embed_dim=128, n_classes=args.num_classes)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        args.num_inference_steps = args.num_train_timesteps
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(
        num_train_timesteps=args.num_train_timesteps, beta_start=args.beta_start, 
        beta_end=args.beta_end, beta_schedule=args.beta_schedule, variance_type=args.variance_type, 
        prediction_type=args.prediction_type, clip_sample=args.clip_sample, clip_sample_range=args.clip_sample_range
    )

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline =  DDPMPipeline(
    unet=unet,
    scheduler=scheduler,
    vae=vae,
    class_embedder = class_embedder
    )

    
    logger.info("***** Running Infrence *****")

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize to match generated image size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dataset = datasets.ImageFolder(root="/content/hw5_code/data/imagenet100_128x128/validation", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    wandb_logger = wandb.init(
            project='ddpm', 
            name="Inference 1", 
            config=vars(args))

    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 25
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images_pil = pipeline(batch_size=batch_size, num_inference_steps=args.num_inference_steps, classes=classes, guidance_scale=args.cfg_guidance_scale, generator=generator, device=device)
            gen_images_tensor = torch.stack([val_transform(img) for img in gen_images_pil])
            all_images.append(gen_images_tensor)
    else:
        # generate 5000 images
        batch_size = args.batch_size 
        for i in tqdm(range(0, 5000, batch_size)):
            gen_images_pil = pipeline(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps, generator=generator, device=device)
            gen_images_tensor = torch.stack([val_transform(img) for img in gen_images_pil])
            all_images.append(gen_images_tensor)
            logger.info(f"Batch at index {i} done")
    
    # TODO: load validation images as reference batch

    all_images = torch.cat(all_images, dim=0)
    
    
    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # TODO: compute FID and IS

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore(feature=2048).to(device)

    with torch.no_grad():
      # Move all_images to the device before updating the FID metric
      all_images = all_images.to(device)
      
      # Update with real images (validation set)
      for images, _ in val_loader:
          images = (images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).to(device)
          fid_metric.update(images, real=True)
      
      # Update with generated images
      all_images = (all_images * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).to(device)
      fid_metric.update(all_images, real=False)
      is_metric.update(all_images)
    
    # Compute FID and IS
    fid_score = fid_metric.compute()
    is_score, _ = is_metric.compute()

    logger.info(f"Frechet Inception Distance (FID): {fid_score}")
    logger.info(f"Inception Score (IS): {is_score}")

    wandb_logger.finish()

        

if __name__ == '__main__':
    main()