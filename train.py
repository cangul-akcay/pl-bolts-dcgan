"""
PL Bolts DCGAN Train Script
"""

from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from pl_bolts.models.gans import DCGAN
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10, MNIST


def main(args=None):
    """
    Main function to train DCGAN

    Args:
        args ([ArgumentParser], optional): Training Arguments. Defaults to None.
    """
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, choices=["cifar10", "mnist"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
        transforms = transform_lib.Compose(
            [
                transform_lib.Resize(script_args.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = CIFAR10(root=script_args.data_dir, download=True, transform=transforms)
        image_channels = 3
    elif script_args.dataset == "mnist":
        transforms = transform_lib.Compose(
            [
                transform_lib.Resize(script_args.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize((0.5,), (0.5,)),
            ]
        )
        dataset = MNIST(root=script_args.data_dir, download=True, transform=transforms)
        image_channels = 1

    dataloader = DataLoader(
        dataset,
        batch_size=script_args.batch_size,
        shuffle=True,
        num_workers=script_args.num_workers,
    )

    parser = DCGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    model = DCGAN(**vars(args), image_channels=image_channels)
    callbacks = [
        TensorboardGenerativeModelImageSampler(num_samples=script_args.batch_size, normalize=True),
        LatentDimInterpolator(interpolate_epoch_interval=5),
    ]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
