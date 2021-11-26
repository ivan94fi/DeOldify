#!/usr/bin/env python
# coding: utf-8

# ## Stable Model Training

# #### NOTES:
# * This is "NoGAN" based training, described in the DeOldify readme.
# * This model prioritizes stable and reliable renderings.  It does particularly well on portraits and landscapes.  It's not as colorful as the artistic model.

# NOTE:  This must be the first call in order to work properly!
import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module=r"torch.nn.functional",
)
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module=r"fastai.data_block",
)

from deoldify import device
from deoldify.device_id import DeviceId

# choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

# isort: split

import os

import fastai
from fastai import *
from fastai.callbacks.tensorboard import *
from fastai.vision import *
from fastai.vision.gan import *
from PIL import Image, ImageDraw, ImageFile, ImageFont

from deoldify.critics import *
from deoldify.dataset import get_colorize_data
from deoldify.generators import *
from deoldify.loss import *
from deoldify.save import *

# ## Setup

# Path to hr images
path = Path("data/imagenet/ILSVRC/Data/CLS-LOC")
path_hr = path
# Path to lr images
path_lr = path / "degraded"

print("hr images (path): {}".format(path))
print("hr images (path_hr): {}".format(path_hr))
print("lr images (path_lr): {}".format(path_lr))

proj_id = "StableModel"

# Path to folder containing generated images
# Name of generator checkpoints excuted in GAN mode (StabelModel_gen_<number>.pth)
gen_name = proj_id + "_gen"
# Name of pretrained generator checkpoint (always ends in _0, rewritten when changing input size)
pre_gen_name = gen_name + "_0"
# Name of discriminator checkpoints
crit_name = proj_id + "_crit"

print("Normal generator checkpoint name (gen_name): {}".format(gen_name))
print("Pretrained generator checkpoint name (pre_gen_name): {}".format(pre_gen_name))
print("Discriminator checkpoint name (crit_name): {}".format(crit_name))

# Generated images directory (StableModel_image_gen)
name_gen = proj_id + "_image_gen"
# Generated images path
path_gen = path / name_gen
print("Generated images path (path_gen): {}".format(path_gen))

TENSORBOARD_PATH = Path("data/tensorboard/" + proj_id)

nf_factor = 2
pct_start = 1e-8


def get_data(bs: int, sz: int, keep_pct: float):
    return get_colorize_data(
        sz=sz,
        bs=bs,
        crappy_path=path_lr,
        good_path=path_hr,
        random_seed=None,
        keep_pct=keep_pct,
    )


def get_crit_data(classes, bs, sz):
    src = ImageList.from_folder(path, include=classes, recurse=True).split_by_rand_pct(
        0.1, seed=42
    )
    ll = src.label_from_folder(classes=classes)
    data = (
        ll.transform(get_transforms(max_zoom=2.0), size=sz)
        .databunch(bs=bs)
        .normalize(imagenet_stats)
    )
    return data


def create_training_images(fn, i):
    """Construct the grayscale samples needed for training."""
    dest = path_lr / fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # converts image to grayscale (with alpha), then back to 3 channels
    img = PIL.Image.open(fn).convert("LA").convert("RGB")
    img.save(dest)


def save_preds(dl):
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1


def save_gen_images():
    if path_gen.exists():
        shutil.rmtree(path_gen)
    path_gen.mkdir(exist_ok=True)
    data_gen = get_data(bs=bs, sz=sz, keep_pct=0.085)
    save_preds(data_gen.fix_dl)
    PIL.Image.open(path_gen.ls()[0])


# ## Create black and white training images

# Only runs if the directory isn't already created.
if not path_lr.exists():
    print("Creating degraded images")
    il = ImageList.from_folder(path_hr)
    parallel(create_training_images, il.items)

# ## Pre-train generator

do_64 = True
do_128 = False
do_192 = False
# #### NOTE
# Most of the training takes place here in pretraining for NoGAN.  The goal here is to take the generator as far as possible with conventional training, as that is much easier to control and obtain glitch-free results compared to GAN training.

# #######################################################
# ### 64px version

if do_64:
    bs = 88
    sz = 64
    keep_pct = 1.0

    # dataloader, gets data from crappy/good paths
    # get all dataset (pct == 1), with 0.1% of validation
    data_gen = get_data(bs=bs, sz=sz, keep_pct=keep_pct)

    # generator learner: unet_wide with vgg16 feature loss
    learn_gen = gen_learner_wide(data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor)

    learn_gen.callback_fns.append(
        partial(ImageGenTensorboardWriter, base_dir=TENSORBOARD_PATH, name="GenPre")
    )

    # train for one epoch, using one-cycle lr scheduling policy
    # The encoder part is a pretrained, frozen resnet101
    print("Generator pretrain: {}".format(sz))
    learn_gen.fit_one_cycle(1, pct_start=0.8, max_lr=slice(1e-3))

    # save the weights
    learn_gen.save(pre_gen_name)

    # unfreeze the encoder
    learn_gen.unfreeze()

    # train for one epoch: now the whole net is unfrozen
    learn_gen.fit_one_cycle(1, pct_start=pct_start, max_lr=slice(3e-7, 3e-4))

    # save the weights
    learn_gen.save(pre_gen_name)

# #######################################################
# ### 128px version

if do_128:
    bs = 20
    sz = 128
    keep_pct = 1.0

    # Construct new dataloader with different batch size and input size
    learn_gen.data = get_data(sz=sz, bs=bs, keep_pct=keep_pct)

    # This should be useless, model is already unfrozen
    learn_gen.unfreeze()

    # Continue training, different lr parameters
    print("Generator pretrain: {}".format(sz))
    learn_gen.fit_one_cycle(1, pct_start=pct_start, max_lr=slice(1e-7, 1e-4))

    learn_gen.save(pre_gen_name)

# #######################################################
# ### 192px version
# Same as before but only use half dataset, lower lrs?

if do_192:
    bs = 8
    sz = 192  # final size
    keep_pct = 0.50  # keep half dataset

    learn_gen.data = get_data(sz=sz, bs=bs, keep_pct=keep_pct)

    learn_gen.unfreeze()

    print("Generator pretrain: {}".format(sz))
    learn_gen.fit_one_cycle(1, pct_start=pct_start, max_lr=slice(5e-8, 5e-5))

    learn_gen.save(pre_gen_name)

# #######################################################
# ## Repeatable GAN Cycle

# #### NOTE
# Best results so far have been based on repeating the cycle below a few times (about 5-8?), until diminishing returns are hit (no improvement in image quality).  Each time you repeat the cycle, you want to increment that old_checkpoint_num by 1 so that new check points don't overwrite the old.

old_checkpoint_num = 0
checkpoint_num = old_checkpoint_num + 1
gen_old_checkpoint_name = gen_name + "_" + str(old_checkpoint_num)
gen_new_checkpoint_name = gen_name + "_" + str(checkpoint_num)
crit_old_checkpoint_name = crit_name + "_" + str(old_checkpoint_num)
crit_new_checkpoint_name = crit_name + "_" + str(checkpoint_num)

# ### Save Generated Images

bs = 8
sz = 192  # final size

# final generator
learn_gen = gen_learner_wide(
    data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor
).load(gen_old_checkpoint_name, with_opt=False)

print("Saving generated images")
save_gen_images()

# ### Pretrain Critic

# ##### Only need full pretraining of critic when starting from scratch.  Otherwise, just finetune!

# On the first iteration train only discriminator for 6 epochs
if old_checkpoint_num == 0:
    bs = 64
    sz = 128
    learn_gen = None
    gc.collect()
    data_crit = get_crit_data([name_gen, "test"], bs=bs, sz=sz)
    data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)
    learn_critic = colorize_crit_learner(data=data_crit, nf=256)
    learn_critic.callback_fns.append(
        partial(LearnerTensorboardWriter, base_dir=TENSORBOARD_PATH, name="CriticPre")
    )
    print("Discriminator pretrain: {}".format(sz))
    learn_critic.fit_one_cycle(6, 1e-3)
    learn_critic.save(crit_old_checkpoint_name)

bs = 16
sz = 192

data_crit = get_crit_data([name_gen, "test"], bs=bs, sz=sz)

data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)

learn_critic = colorize_crit_learner(data=data_crit, nf=256).load(
    crit_old_checkpoint_name, with_opt=False
)

learn_critic.callback_fns.append(
    partial(LearnerTensorboardWriter, base_dir=TENSORBOARD_PATH, name="CriticPre")
)

learn_critic.fit_one_cycle(4, 1e-4)

learn_critic.save(crit_new_checkpoint_name)

# ### GAN

learn_crit = None
learn_gen = None
gc.collect()

lr = 2e-5
sz = 192
bs = 5

data_crit = get_crit_data([name_gen, "test"], bs=bs, sz=sz)

learn_crit = colorize_crit_learner(data=data_crit, nf=256).load(
    crit_new_checkpoint_name, with_opt=False
)

learn_gen = gen_learner_wide(
    data=data_gen, gen_loss=FeatureLoss(), nf_factor=nf_factor
).load(gen_old_checkpoint_name, with_opt=False)

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(
    learn_gen,
    learn_crit,
    weights_gen=(1.0, 1.5),
    show_img=False,
    switcher=switcher,
    opt_func=partial(optim.Adam, betas=(0.0, 0.9)),
    wd=1e-3,
)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.0))
learn.callback_fns.append(
    partial(
        GANTensorboardWriter,
        base_dir=TENSORBOARD_PATH,
        name="GanLearner",
        visual_iters=100,
    )
)
learn.callback_fns.append(
    partial(
        GANSaveCallback,
        learn_gen=learn_gen,
        filename=gen_new_checkpoint_name,
        save_iters=100,
    )
)

# Find the checkpoint just before where glitches start to be introduced.  This is all very new so you may need to play around with just how far you go here with keep_pct.

learn.data = get_data(sz=sz, bs=bs, keep_pct=0.03)
learn_gen.freeze_to(-1)
learn.fit(1, lr)
