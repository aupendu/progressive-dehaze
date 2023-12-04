import torch.nn as nn
from importlib import import_module

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()

            elif loss_type == 'L1':
                loss_function = nn.L1Loss()

            elif loss_type.find('VGG20') >= 0:
                module = import_module('loss.vgg20')
                loss_function = getattr(module, 'VGG')(loss_type[5:], rgb_range=1)
                loss_function = loss_function.cuda()

            elif loss_type.find('SSIM') >= 0:
                module = import_module('loss.ssim')
                loss_function = getattr(module, 'SSIM')()

            elif loss_type.find('MSSIM') >= 0:
                module = import_module('loss.ssim')
                loss_function = getattr(module, 'MSSSIM')()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})


    def forward(self, dehaze, GT):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:

                loss = l['function'](dehaze, GT)

            effective_loss = l['weight'] * loss
            losses.append(effective_loss)
        loss_sum = sum(losses)

        return loss_sum

