import os
import sys
import torch

from diffusers import DiffusionPipeline


def main(dataset, clsname, seed):
    # 1. load both base & refiner
    base = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', 
                                            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    base.to("cuda:1")

    refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                text_encoder_2=base.text_encoder_2, 
                                                vae=base.vae,
                                                torch_dtype=torch.float16,
                                                use_safetensors=True,
                                                variant="fp16",)
    refiner.to("cuda:1")

    # 2. Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    if '_' in clsname:
        cls = clsname.replace('_', ' ')
    else:
        cls = clsname
        
    if cls == 'airplane':
        prompt = f"a single object of an {cls} with clean background"
    else:
        prompt = f"a single object of a {cls} with clean background"

    # 3. run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    os.makedirs(f'data/diffusion/{dataset}/{clsname}', exist_ok=True)
    image.save(f'data/diffusion/{dataset}/{clsname}/{seed}.png')
    print(f'>>> SAVE data/diffusion/{dataset}/{clsname}/{seed}.png Done!')


if __name__ == '__main__':
    dataset = sys.argv[1]
    
    if dataset == 'modelnet_c':
        cls_file = 'data/modelnet_c/shape_names.txt'
        with open(cls_file) as fin:
            lines = fin.readlines()
            classnames = [line.strip() for line in lines if line.strip() != ""]
    
    for seed in [1,2,3]:
        for clsname in classnames:
            main(dataset, clsname, seed)
