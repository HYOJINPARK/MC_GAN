import argparse
import torchtext.vocab as vocab
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler

from model_MCb import VisualSemanticEmbedding
from model_MCb import G_NET, D_NET64
from data_MCb import ReedICML2016
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################################   NOTICE   ###########################################

# This code is based from stackGAN ++
# https://github.com/hanzhanggit/StackGAN-v2
# https://github.com/woozzu/dong_iccv_2017

#######################################################################################


img_normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, default='../../../DATA/cub_200_2011/cub/images',
                    help='root directory that contains images')

parser.add_argument('--caption_root', type=str, default='../../../DATA/cub_200_2011/cub/cub_ReedScott',
                    help='root directory that contains captions')
parser.add_argument('--trainclasses_file', type=str, default='trainvalclasses.txt',
                    help='text file that contains training classes')
parser.add_argument('--NET_G', type=str, default='',
                    help='Net_G')
parser.add_argument('--NET_D', type=str, default='',
                    help='Net_D')

parser.add_argument('--text_embedding_model', type=str, default='text_emb/birds.pth',
                    help='pretrained text embedding model')
parser.add_argument('--save_filename', type=str, default='G_model',
                    help='checkpoint file')
parser.add_argument('--num_threads', type=int, default=4,
                    help='number of threads for fetching data (default: 4)')
parser.add_argument('--num_epochs', type=int, default=600,
                    help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                    help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
parser.add_argument('--z_dim', type = int, default=100,
                    help='noise dimension for FG')
parser.add_argument('--FG_emb', type = int, default=128,
                    help='noise dimension for FG')
parser.add_argument('--ngf', type = int, default=1024,
                    help='number of generator base feature')
parser.add_argument('--ndf', type = int, default=64,
                    help='number of discriminator base feature')
parser.add_argument('--no_cuda', type = bool, default=False,
                    help='do not use cuda')
parser.add_argument('--gpu', dest='gpu_id', type=str, default='1')

args = parser.parse_args()
cudnn.benchmark = True

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True
    cudnn.benchmark = False

# s_gpus = args.gpu_id.split(',')
# gpus = [int(ix) for ix in s_gpus]


def preprocess(img, desc, len_desc, txt_encoder):
    img = Variable(img.cuda() if not args.no_cuda else img)
    desc = Variable(desc.cuda() if not args.no_cuda else desc)

    # len_desc = len_desc.numpy()
    # sorted_indices = np.argsort(len_desc)[::-1]
    len_desc, indices = torch.sort(len_desc, 0, True)
    sorted_indices = indices.numpy()
    original_indices = np.argsort(sorted_indices)
    desc = desc[sorted_indices, ...].transpose(0, 1)
    #len_desc = len_desc[sorted_indices]
    packed_desc = nn.utils.rnn.pack_padded_sequence(
        desc,len_desc.numpy() )
    _, txt_feat = txt_encoder(packed_desc)
    txt_feat = txt_feat.squeeze()
    txt_feat = txt_feat[original_indices, ...]

    txt_feat_np = txt_feat.data.cpu().numpy() if not args.no_cuda else txt_feat.data.numpy()
    txt_feat_mismatch = torch.Tensor(np.roll(txt_feat_np, 1, axis=0))
    txt_feat_mismatch = Variable(txt_feat_mismatch.cuda() if not args.no_cuda else txt_feat_mismatch)

    txt_feat_np_split = np.split(txt_feat_np, [txt_feat_np.shape[0] // 2])
    txt_feat_relevant = torch.Tensor(np.concatenate([
        np.roll(txt_feat_np_split[0], -1, axis=0),
        txt_feat_np_split[1]
    ]))
    txt_feat_relevant = Variable(txt_feat_relevant.cuda() if not args.no_cuda else txt_feat_relevant)

    return img, txt_feat, txt_feat_mismatch, txt_feat_relevant

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network():
    netG = G_NET(args)
    netG.apply(weights_init)
    print(netG)

    netD = D_NET64(args)
    netD.apply(weights_init)


    count = 0
    if args.NET_G != '':
        state_dict = torch.load(args.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', args.NET_G)

        istart = args.NET_G.rfind('_') + 1
        iend = args.NET_G.rfind('.')
        count = args.NET_G[istart:iend]
        count = int(count) + 1

    if args.NET_D != '':
        print('Load %s' % (args.NET_D))
        state_dict = torch.load(args.NET_D)
        netD.load_state_dict(state_dict)


    if not args.no_cuda:
        netG.cuda()
        netD.cuda()


    return netG, netD, count

if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = vocab.FastText(language="en")

    print('Loading a dataset...')
    train_data = ReedICML2016(args.img_root,
                              args.caption_root,
                              args.trainclasses_file,
                              word_embedding,
                              args.max_nwords,
                              transforms.Compose([
                                  transforms.Scale(74),
                                  transforms.RandomCrop(64),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                              ]))

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    word_embedding = None

    # pretrained text embedding model
    print('Loading a pretrained text embedding model...')
    txt_encoder = VisualSemanticEmbedding(args.embed_ndim)
    txt_encoder.load_state_dict(torch.load(args.text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder
    for param in txt_encoder.parameters():
        param.requires_grad = False

    if not args.no_cuda:
        print('use cuda accelerate')

    G, D, start_count = load_network()
    nz = args.z_dim
    noise = Variable(torch.FloatTensor(args.batch_size, nz))

    if not args.no_cuda:
        txt_encoder.cuda()
        G.cuda()
        D.cuda()
        noise = noise.cuda()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, args.lr_decay)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, args.lr_decay)

    criterion = nn.MSELoss()
    predictions = []
    start_t = time.time()
    start_epoch = start_count // (args.batch_size)

    for epoch in range(args.num_epochs):

        for i, (img, desc, len_desc) in enumerate(train_loader):
            img, txt_feat, txt_mismatch, txt_relv = \
                preprocess(img, desc, len_desc, txt_encoder)

            batch_size = img.size(0)
            ONES = Variable(torch.ones(batch_size))
            ZEROS = Variable(torch.zeros(batch_size))
            if not args.no_cuda:
                ONES, ZEROS = ONES.cuda(), ZEROS.cuda()

            noise.data.normal_(0, 1)
            this_noise = noise[:batch_size,:]
            fake_img, z_mean, z_log_var = G(this_noise, txt_relv, img)

            # UPDATE DISCRIMINATOR
            D.zero_grad()

            real_logits = D(img, txt_feat.detach())
            wtxt_logits = D(img, txt_mismatch.detach())
            fake_logits = D(fake_img.detach(), txt_relv.detach())


            errD_real = 1*criterion(real_logits[0], ONES)
            errD_wtxt = 0.5*criterion(wtxt_logits[0], ZEROS)
            errD_fake = 0.5*criterion(fake_logits[0], ZEROS)


            errD = errD_real + errD_wtxt + errD_fake
            errD.backward()
            d_optimizer.step()


            # UPDATE GENERATOR
            G.zero_grad()

            outputs = D(fake_img, txt_relv)
            errG_IST = criterion(outputs[0], ONES)
            errG_total = errG_IST

            kl_loss = KL_loss(z_mean, z_log_var)
            errG_total = errG_total + kl_loss

            errG_total.backward()
            g_optimizer.step()


            if i % 50 == 0:
                end_t = time.time()
                print('Epoch [%d/%d], Iter [%d/%d], errD: %.4f, G_pair: %.4f, KL: %.4f, time: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, len(train_loader), errD.data[0],
                         errG_IST.data[0],kl_loss.data[0],  end_t - start_t))
                start_t = time.time()

        if (epoch+1) % 50 ==0:
            save_image((fake_img.data + 1) * 0.5, './examples/MC_%d.png' % (epoch + 1))

            torch.save(G.state_dict(),"{}/MC_G_bird_{}.pth".format(args.save_filename, epoch+1))
            torch.save(D.state_dict(),"{}/NC_D_bird.pth".format(args.save_filename))




