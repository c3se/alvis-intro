import os
import argparse
from functools import wraps
from glob import glob

import torch
import torchvision
from datasets import concatenate_datasets, Dataset
from torchvision import models
from torchvision.transforms import v2, InterpolationMode


parser = argparse.ArgumentParser(description="Trains a ResNet-50 on ImageNet-1k")

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--max-steps-per-epoch", type=int, default=0)

torch.set_float32_matmul_precision("high")

# Performance options
def get_dtype(arg: str):
    try:
        dtype = getattr(torch, arg)
    except AttributeError:
        raise ValueError(f'"{arg}" is not a known torch.dtype')

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f'Expected a torch.dtype object got "{type(dtype)}"')

    return dtype

parser.add_argument(
    "--fp32-matmul-precision",
    default="high",
    choices=["highest", "high", "medium"],
)
parser.add_argument("--use-amp", action="store_true")
parser.add_argument("--amp-dtype", default=torch.float16, type=get_dtype)
parser.add_argument(
    "--num-workers",
    type=int,
    default=(int(os.getenv("SLURM_CPUS_ON_NODE", "2")) - 2),
)
parser.add_argument("--pin-memory", default=True, action="store_true")
parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
parser.add_argument("--device", type=torch.device, default="cuda")

# Misc. options
parser.add_argument(
    "--dataroot",
    default=(
        "/mimer/NOBACKUP/Datasets/"
        "ImageNet/hf-cache/imagenet-1k/default/1.0.0/"
        "09dbb3153f1ac686bac1f40d24f307c383b383bc171f2df5d9e91c1ad57455b9/"
    ),
)
parser.add_argument('--validation', default=True, action=argparse.BooleanOptionalAction)


def per_sample(cls: torch.nn.Module):
    '''Wraps a transform to perform one forward call per sample.'''

    class PerSampleTransform(cls):
        '''Wrapper to do one forward call per sample.

        Original doc:
        ''' + f"{cls.__doc__}"

        @wraps(cls.forward)
        def forward(self, inputs):
            imgs = inputs["image"]
            for ix, img in enumerate(imgs):
                imgs[ix] = super().forward(img)
            return inputs

    return PerSampleTransform


class ToRGB(torch.nn.Module):
    '''Transform tensor image to 3-channel RGB.'''

    def forward(self, img: torch.Tensor):
        return (
            img[..., :3, :, :]  # RGB or RGBA to RBG
            if img.size(-3) > 1
            else img.repeat(3, 1, 1)  # greyscale to RGB
        )


def get_dataloader(args: argparse.Namespace, train: bool):
    '''Initializes and returns a dataloader.'''

    # Init dataset
    split = "train" if train else "validation"
    dataset = concatenate_datasets(
        [
            Dataset.from_file(fn) for fn in glob(
                f"{args.dataroot}/imagenet-1k-{split}-00???-of-00???.arrow"
            )
        ]
    ).with_format("torch")

    # Init transforms
    if train:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            per_sample(ToRGB)(),
            per_sample(v2.RandomResizedCrop)(
                size=(224, 224),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            per_sample(ToRGB)(),
            per_sample(v2.Resize)(
                size=(224, 224),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset.set_transform(transforms, columns="image", output_all_columns=True)

    # Init dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=train,
        batch_size=(args.batch_size if train else 2 * args.batch_size),
        num_workers=(args.num_workers if train else min(args.num_workers, 1)),
        pin_memory=args.pin_memory,
    )
    return dataloader


def main():
    args = parser.parse_args()

    if not os.access(args.dataroot, os.R_OK):
        raise RuntimeError(
            'No read access to dataroot, see'
            ' https://www.c3se.chalmers.se/documentation/applications/datasets/#imagenet'
        )
    torch.set_float32_matmul_precision(args.fp32_matmul_precision)



    trainloader = get_dataloader(args, train=True)
    valloader = get_dataloader(args, train=False) if args.validation else []

    # Initialize model stuff
    model = models.resnet50(weights=None).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Training
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(trainloader):
            images = batch["image"].to(device=args.device)
            labels = batch["label"].to(device=args.device)
            optimizer.zero_grad()

            # Calculate loss
            with torch.autocast(
                device_type=args.device.type,
                dtype=args.amp_dtype,
                enabled=args.use_amp,
            ):
                logits = model(images)
                loss = loss_fn(logits, labels)

            # Update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break

        print(f"Epoch {epoch+1}/{args.num_epochs}", f"Loss: {loss}", end=" ")

        model.eval()
        with torch.no_grad():
            n_top1 = 0
            n_top5 = 0
            n_samples = 0
            for step, batch in enumerate(valloader):
                images = batch["image"].to(device=args.device)
                labels = batch["label"].to(device=args.device)

                # Calculate loss
                logits = model(images)
                loss = loss_fn(logits, labels)

                # Calculate Top-N accuracies
                n_samples += images.size(0)
                labels = labels.view(-1, 1)
                n_top1 += (
                    logits.topk(k=1, dim=1).indices == labels
                ).any(dim=1).sum(dim=0)
                n_top5 += (
                    logits.topk(k=5, dim=1).indices == labels
                ).any(dim=1).sum(dim=0)

                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break

        if args.validation:
            print(
                f"Val. loss: {loss}",
                f"Val. top-1 acc.: {n_top1 / n_samples}",
                f"Val. top-5 acc.: {n_top5 / n_samples}",
            )


if __name__ == '__main__':
    main()
