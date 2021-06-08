import os
import time

import torch
import torch.quantization.qconfig
import config
from torchvision.utils import save_image

def test_some_examples(gen, disc, val_loader, epoch_num, folder):
    sum_time = 0
    D_fake_sum = 0
    f = 0
    BCE = torch.nn.BCEWithLogitsLoss()
    for epoch in range(epoch_num):
        x, y = next(iter(val_loader))
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            print("______________________________")
            iter_start_time = time.time()
            y_fake = gen(x)
            iter_end_time = time.time() - iter_start_time

            if epoch == 0:
                save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
            else:
                sum_time = sum_time + iter_end_time
                print('Inference time of one iter:', iter_end_time)
                D_fake = disc(x, y_fake)
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                f = f + D_fake_loss
                print('D_fake_loss:', D_fake_loss)
                D_fake = torch.sigmoid(disc(x, y_fake)).mean().item()
                D_fake_sum = D_fake_sum + D_fake
                print('D_fake_sigmoid:', D_fake)


                y_fake = y_fake * 0.5 + 0.5  # remove normalization#
                save_image(y_fake, folder + f"/y_gen_{epoch}.png")

        gen.train()
    print("______________________________")
    print('Mean inference time of one iter:', sum_time / (epoch_num-1)) #-1 потому что первая генерация почему то очень долгая
    print('Mean D_fake_sigmoid:', D_fake_sum / (epoch_num-1))
    print('Mean D_LOSS:', f / (epoch_num - 1))

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size


