from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from utils import VggGenerator, VggDiscriminator, MnistDataset, format_Gx, create_figure
from RAdam import RAdam

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--data_dir', default='/home/ted/Projects/CVPR/mnist_png/training')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

dataset = MnistDataset(data_dir=opt.data_dir)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = VggGenerator()

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = VggDiscriminator()

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCEWithLogitsLoss()

fixed_noise, _, _, _ = next(iter(dataloader))
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = RAdam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = RAdam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

import shutil
shutil.rmtree('runs', ignore_errors=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

wait_D = 0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # unpack img and its label
        in_img, out_img, in_class, out_class = data

        in_img = in_img.to(device)
        out_img = out_img.to(device)

        # train with fake
        fake = netG(in_img)
        if wait_D == 4 or i == 0:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = out_img.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item() # should stay close to 1 and converge to 0.5

            # train with fake
            # fake = netG(in_img)

            if i % 10 == 0:
                fig = create_figure(in_img, out_img, format_Gx(fake))
                writer.add_figure("Images", fig, i, close=True)

            label.fill_(fake_label)
            output = netD(fake.detach()) # no gradients flow back Generator
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item() # before update, should stay close to 0 and converge to 0.5
            errD = errD_real + errD_fake
            
            torch.nn.utils.clip_grad_value_(netD.parameters(), clip_value=0.01)
            optimizerD.step()
            if i!= 0:
                wait_D = 0

        wait_D += 1
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator loss
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item() # after update, should stay close to 0 and converge to 0.5
        optimizerG.step()
        
        

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, opt.niter, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise.to(device))
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

        if i % 10 == 0:
            writer.add_scalars('Loss', {'D': errD.item(),
                                        'G': errG.item()}, i)
            writer.add_scalars('D Ouput', {'real': D_x,
                                           'fake_before': D_G_z1,
                                           'fake_after': D_G_z2}, i)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

writer.close()


