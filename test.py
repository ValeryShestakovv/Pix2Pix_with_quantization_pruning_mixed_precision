import time

import torch
import torch.quantization.qconfig
from torch import optim
from torch.nn.utils import prune
from torchvision.transforms import transforms

from generator_model import Generator
from discriminator_model import Discriminator
import config
from torchvision.utils import save_image
from dataset import MapDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import save_some_examples, test_some_examples, load_checkpoint, print_size_of_model


def test():

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    BCE = torch.nn.BCEWithLogitsLoss()

    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )


    #Глобальный прунинг модели
    """print("GLOBAL_PRUNING_MODEL______________________")
    parameters_to_prune = (
        (gen.initial_down[0], 'weight'),
        (gen.down1.conv[0], 'weight'),
        (gen.down2.conv[0], 'weight'),
        (gen.down3.conv[0], 'weight'),
        (gen.down4.conv[0], 'weight'),
        (gen.down5.conv[0], 'weight'),
        (gen.down6.conv[0], 'weight'),
        (gen.up1.conv[0], 'weight'),
        (gen.up2.conv[0], 'weight'),
        (gen.up3.conv[0], 'weight'),
        (gen.up4.conv[0], 'weight'),
        (gen.up5.conv[0], 'weight'),
        (gen.up6.conv[0], 'weight'),
        (gen.up7.conv[0], 'weight'),
        (gen.final_up[0], 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.90,
    )
    prune.remove(gen.initial_down[0], 'weight')
    prune.remove(gen.down1.conv[0], 'weight')
    prune.remove(gen.down2.conv[0], 'weight')
    prune.remove(gen.down3.conv[0], 'weight')
    prune.remove(gen.down4.conv[0], 'weight')
    prune.remove(gen.down5.conv[0], 'weight')
    prune.remove(gen.down6.conv[0], 'weight')
    prune.remove(gen.up1.conv[0], 'weight')
    prune.remove(gen.up2.conv[0], 'weight')
    prune.remove(gen.up3.conv[0], 'weight')
    prune.remove(gen.up4.conv[0], 'weight')
    prune.remove(gen.up5.conv[0], 'weight')
    prune.remove(gen.up6.conv[0], 'weight')
    prune.remove(gen.up7.conv[0], 'weight')
    prune.remove(gen.final_up[0], 'weight')"""

    #Локальный прунинг для слоя conv2d (рандом)
    """print("LOCAL_PRUNING_CONV2D______________________")
    module = gen.initial_down[0]
    prune.random_unstructured(module, name='weight', amount=0.3)
    print(module.weight)
    module._forward_pre_hooks
    print(list(gen.initial_down[0].named_parameters()))
    print("END_PRUNING______________________")
    #если нужно юзать с квантизацией, то необходимо сделать "Remove pruning re-parametrization", т.е. присваеваем веса обрезаной модоли изначальной.
    print("Remove pruning re-parametrization")
    prune.remove(module, 'weight')"""

    # static (post-train) quan
    # (pytorch конфиг "fbgemm" в pytorch 1.8 пока еще не поддерживает квантование по каналам для conv_transpose2d (https://github.com/pytorch/pytorch/issues/54816),
    # который есть в модели pix2pix ("qnnpack" только для arm))
    """backend = "fbgemm"
    gen.config = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_fp32_fused = torch.quantization.fuse_modules(gen, [['conv', 'relu']])
    model_fp32_prepared = torch.quantization.prepare(gen)
    print(type(val_dataset))
    for data, target in enumerate(val_loader):
        model_fp32_prepared(data)
    model_int8 = torch.quantization.convert(model_fp32_prepared)"""

    #Динамическая квантизация
    """gen_int8 = torch.quantization.quantize_dynamic(gen, {torch.nn.Sequential}, dtype=torch.qint8)"""

    #Смотрим размер модели
    """f = print_size_of_model(gen, "fp32")
    q = print_size_of_model(gen32, "fp32")"""

    test_some_examples(gen, disc, val_loader, epoch_num=1000, folder="evaluation")


test()