import math
import torch
from torch import optim, nn
from torch.optim import lr_scheduler


def basic(dataset, model):
    opt = optim.Adam(model.parameters())
    schedule = lr_scheduler.ExponentialLR(opt, gamma=0.9)

    for epoch in range(1):
        for i, target in dataset:

            opt.step()
            opt.zero_grad()
        schedule.step()


def lr_warmup(dataset, model, warmup_steps=4000):
    opt = optim.AdamW(model.parameters())
    stop_lr = 1 / math.sqrt(warmup_steps)
    end_factor = stop_lr / opt.param_groups[0]['lr']
    schedule_1 = lr_scheduler.LinearLR(opt, start_factor=1, end_factor=1 / math.sqrt(warmup_steps), total_iters=warmup_steps)
    schedule_2 = lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda x: 1 / math.sqrt(x))

    for epoch in range(1):
        for i, target in dataset:
            opt.step()
            opt.zero_grad()

        if epoch < warmup_steps:
            schedule_1.step()
        else:
            schedule_2.step()