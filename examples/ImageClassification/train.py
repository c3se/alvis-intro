import argparse
from glob import glob


import torch
import torchvision
from datasets import concatenate_datasets, Dataset


parser = argparse.ArgumentParser(description="Trains a ResNet-50 on ImageNet-1k")

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--num-epochs", type=int, default=10)


# Performance options
parser.add_argument("--use-tf32", action="store_true")
parser.add_argument("--num-workers", type=int, default=0)
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


def main():
    args = parser.parse_args()

    # Initialize dataloading
    trainset = concatenate_datasets(
        [
            Dataset.from_file(fn) for fn in glob(
                    f"{args.dataroot}/imagenet-1k-train-00???-of-00257.arrow",
            )
        ]
    ).with_format("torch", device=args.device)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Initialize model stuff
    model = torchvision.models.resnet50(weights=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(args.num_epochs):
        for inputs, labels in trainloader:
            optimizer.zero_grad()

            # Calculate loss
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Update weights
            loss.backward()
            optimizer.step()

            break

        #TODO check performance on validation set
        print(f"Epoch {epoch+1}/{args.num_epochs}", f"Loss: {loss}")


if __name__ == '__main__':
    main()
