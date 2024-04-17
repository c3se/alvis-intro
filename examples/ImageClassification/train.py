import argparse
from glob import glob

import torch
import torchvision
from torchvision import models
from torchvision.transforms import v2
from datasets import concatenate_datasets, Dataset


parser = argparse.ArgumentParser(description="Trains a ResNet-50 on ImageNet-1k")

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--num-epochs", type=int, default=10)


# Performance options
parser.add_argument("--use-tf32", action="store_true")
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--pin-memory", action="store_true")
parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))

# Misc. options
parser.add_argument(
    "--dataroot",
    default=(
        "/mimer/NOBACKUP/Datasets/"
        "ImageNet/hf-cache/imagenet-1k/default/1.0.0/"
        "09dbb3153f1ac686bac1f40d24f307c383b383bc171f2df5d9e91c1ad57455b9/"
    ),
)


class PerSampleRandomResizedCrop(v2.RandomResizedCrop):
    '''Wrapper for RandomResizedCrop to deal with inputs of different sizes.'''
    def forward(self, inputs):
        imgs = inputs["image"]
        out_imgs = None
        for ix, img in enumerate(imgs):
            imgs[ix] = super().forward(img)
        return inputs


def main():
    args = parser.parse_args()

    # Load raw data
    trainset = concatenate_datasets(
        [
            Dataset.from_file(fn) for fn in glob(
                    f"{args.dataroot}/imagenet-1k-train-00???-of-00257.arrow",
            )
        ]
    ).with_format("torch")

    # Initialize dataloading
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        PerSampleRandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    trainset.set_transform(transforms)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Initialize model stuff
    model = models.resnet50(weights=None).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(args.num_epochs):
        for batch in trainloader:
            images = batch["image"].to(device=args.device)
            labels = batch["label"].to(device=args.device)
            optimizer.zero_grad()

            # Calculate loss
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Update weights
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.num_epochs}", f"Loss: {loss}")


if __name__ == '__main__':
    main()
