# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from compressai.datasets import ImageFolder
from compressai.zoo import models
from torch.utils.tensorboard import SummaryWriter
from datasets import build_dataset
from timm.data import Mixup
from timm.utils import accuracy, AverageMeter


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.class_acc = nn.CrossEntropyLoss()

    def forward(self, output, target,labels):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["cls_loss"] = self.class_acc(output["y_class"],labels)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]+out["cls_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, mixup_fn, clip_max_norm
):
    model.train()
    # device = next(model.parameters()).device
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    for i, (samples, targets) in enumerate(train_dataloader):
        # d = d.to(device)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            out_criterion = criterion(outputs,samples, targets)

        # out_net = model(d)
        # out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(samples)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\tBpp loss: {out_criterion["cls_loss"].item():.3f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
    return out_criterion,aux_loss

def tf_imwrite(imgs, path):
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    imgs=imgs.cpu()
    imagenet_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, 3, 1, 1)
    imagenet_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, 3, 1, 1)

    imgs = torchvision.utils.make_grid(torch.clamp(imgs * imagenet_std + imagenet_mean, min=0., max=1.))

    imgs_pil = torchvision.transforms.ToPILImage()(imgs)
    imgs_pil.save(path)


def test_epoch(epoch, test_dataloader, model, criterion,output_dir):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    cls_loss = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    write_img = True
    with torch.no_grad():
        for images, target in test_dataloader:
            # d = d.to(device)
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            with torch.cuda.amp.autocast():
                out_net = model(images)
                out_criterion = criterion(out_net, images,target)
            if write_img:
                tf_imwrite(images[:4], os.path.join(output_dir, 'example_org.png'))
                tf_imwrite(out_net["x_hat"], os.path.join(output_dir, 'example_rec.png'))
                write_img = False

            acc1, acc5 = accuracy(out_net["y_class"], target, topk=(1, 5))

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            cls_loss.update(out_criterion["cls_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            acc1_meter.update(acc1.item())
            acc5_meter.update(acc5.item())
            loss.update(out_criterion["loss"])


    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAcc1 loss: {acc1_meter.avg:.2f} |"
        f"\tAcc5 loss: {acc5_meter.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg,mse_loss.avg,bpp_loss.avg,aux_loss.avg,acc1_meter.avg,acc5_meter.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    # parser.add_argument(
    #     "-d", "--dataset", type=str, required=True, help="Training dataset"
    # )
    parser.add_argument('--data-path', type=str, required=True,
                        help='dataset path')
    # Dataset parameters
    parser.add_argument('--data-set', default='IMNET', choices=['IMNET', 'INAT19'],
                        type=str, help='Image Net dataset path')

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument('--input-size', default=256, type=int, help='images input size')
    # parser.add_argument(
    #     "--patch-size",
    #     type=int,
    #     nargs=2,
    #     default=(256, 256),
    #     help="Size of the patches to be cropped (default: %(default)s)",
    # )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    output_dir = '/yuanxx_9F/stf/output'
    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    # )
    #
    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    # )

    # train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    # test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        dataset_val,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    net = models[args.model]()
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    writer = SummaryWriter(log_dir='/taofei/stf1/logs1')
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_out_criterion,train_aux_loss =train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            mixup_fn,
            args.clip_max_norm,
        )
        loss,test_mse_loss,test_bpp_loss,test_aux_loss,acc1,acc5 = test_epoch(epoch, test_dataloader, net, criterion,output_dir)
        lr_scheduler.step(loss)
        # with SummaryWriter(log_dir='/taofei/stf1/logs') as writer:
        writer.add_scalar(tag='test/loss',scalar_value=loss,global_step=epoch)
        writer.add_scalar(tag='test/mse_loss',scalar_value=test_mse_loss* 255 ** 2 / 3,global_step=epoch)
        writer.add_scalar(tag='test/bpp_loss',scalar_value=test_bpp_loss,global_step=epoch)
        writer.add_scalar(tag='test/aux_loss',scalar_value=test_aux_loss,global_step=epoch)
        writer.add_scalar(tag='test/Acc1', scalar_value=acc1, global_step=epoch)
        writer.add_scalar(tag='test/Acc5', scalar_value=acc5, global_step=epoch)

        writer.add_scalar(tag='train/loss',scalar_value=train_out_criterion['loss'].item(),global_step=epoch)
        writer.add_scalar(tag='train/mse_loss', scalar_value=train_out_criterion['mse_loss'].item()* 255 ** 2 / 3, global_step=epoch)
        writer.add_scalar(tag='train/bpp_loss', scalar_value=train_out_criterion['bpp_loss'].item(), global_step=epoch)
        writer.add_scalar(tag='train/aux_loss', scalar_value=train_aux_loss.item(), global_step=epoch)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                args.save_path,
            )
    writer.close()


if __name__ == "__main__":
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    main(sys.argv[1:])
