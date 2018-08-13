from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
import os
import time
from PIL import Image
import cv2
from copy import deepcopy
import random

from miscc.config import cfg
from miscc.utils import mkdir_p

from model import G_NET, D_NET, INCEPTION_V3
import torchvision.transforms as transforms

################################   NOTICE   ###########################################

# This code is based from stackGAN ++
# https://github.com/hanzhanggit/StackGAN-v2

#######################################################################################


Torchtransform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])
Tensor2Img = transforms.Compose([transforms.ToPILImage()])

# ################## Shared functions ###################
def compute_mean_covariance(img):

    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    img_hat_transpose = img_hat.transpose(1, 2)
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):

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


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):

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

    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET())

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        # print(netsD[i])
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    inception_model = INCEPTION_V3()

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()

    return netG, netsD, len(netsD), inception_model, count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save( netG.state_dict(),'%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(netD.state_dict(),'%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')



# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.output_dir = output_dir
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.SNAPSHOT_INTERVAL = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.img_size = imsize // (2 ** (cfg.TREE.BRANCH_NUM - 1))

        if cfg.DATASET_NAME.find('flower') != -1:
            self.base_num = 1217

    def prepare_data(self, data):
        imgs, segs, w_imgs, w_segs, t_embedding, _, = data

        this_batch = imgs[0].size(0)
        crop_vbase= []

        for N_d in range(self.num_Ds):
            crop_base_imgs = torch.zeros(this_batch, 3, self.img_size, self.img_size)
            for step, (base_img_list) in enumerate(data[5]):
                if cfg.DATASET_NAME.find('flower') != -1:
                    base_ix = random.randint(1, self.base_num)
                    base_img_name = '%s/%s.jpg' % (base_img_list, str(base_ix))
                else:
                    temp_base_list = os.listdir(base_img_list)
                    base_ix = random.randint(0, len(temp_base_list) - 1)
                    base_img_name = '%s/%s.jpg' % (base_img_list, str(base_ix))
                base_img = Image.open(base_img_name).convert('RGB')
                crop_base = base_img.resize([self.img_size, self.img_size])

                crop_base = Torchtransform(crop_base)
                crop_base_imgs[step, :] = crop_base

            if cfg.CUDA:
                crop_vbase.append(Variable(crop_base_imgs).cuda())
            else:
                crop_vbase.append(Variable(crop_base_imgs))

        real_vimgs, real_vsegs, wrong_vimgs, wrong_vsegs = [], [], [], []
        if cfg.CUDA:
            vembedding = Variable(t_embedding).cuda()
        else:
            vembedding = Variable(t_embedding)

        for i in range(self.num_Ds):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                real_vsegs.append(Variable(segs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
                wrong_vsegs.append(Variable(w_segs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                real_vsegs.append(Variable(segs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
                wrong_vsegs.append(Variable(w_segs[i]))

        return real_vimgs, real_vsegs, wrong_vimgs, wrong_vsegs, vembedding, crop_vbase

    def train_Dnet(self, idx):

        batch_size = self.real_imgs[idx].size(0)
        criterion, mu = self.criterion, self.mu

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]

        # Forward
        netD.zero_grad()
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        errD = 0

        fake_imgs = self.fake_imgs[idx]
        real_segs = self.real_segs[idx]
        wrong_segs = self.wrong_segs[idx]
        fake_segs = self.fake_segs[idx]
        # Discriminant

        real_logits = netD(real_imgs, mu.detach(), real_segs)
        wtxt_logits = netD(wrong_imgs, mu.detach(), wrong_segs)
        wseg_logits = netD(real_imgs, mu.detach(), wrong_segs)
        fake_logits = netD(fake_imgs.detach(), mu.detach(), fake_segs.detach())

        # img-seg-txt
        errD_real = cfg.TRAIN.COEFF.ITS_LOSS*criterion(real_logits[0], real_labels)
        errD_wtxt = cfg.TRAIN.COEFF.ITS_LOSS*criterion(wtxt_logits[0], fake_labels)
        errD_wseg = cfg.TRAIN.COEFF.ITS_LOSS*criterion(wseg_logits[0], fake_labels)
        errD_fake = cfg.TRAIN.COEFF.ITS_LOSS*criterion(fake_logits[0], fake_labels)

        # img?
        if len(real_logits) > 0 and cfg.TRAIN.COEFF.I_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.I_LOSS * \
                               criterion(real_logits[1], real_labels)
            errD_wtxt_uncond = cfg.TRAIN.COEFF.I_LOSS * \
                               criterion(wtxt_logits[1], real_labels)
            errD_wseg_uncond = cfg.TRAIN.COEFF.I_LOSS * \
                               criterion(wseg_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.I_LOSS * \
                               criterion(fake_logits[1], fake_labels)

            #
            errD_real = errD_real + errD_real_uncond
            errD_wtxt = errD_wtxt + errD_wtxt_uncond
            errD_wseg = errD_wseg + errD_wseg_uncond
            errD_fake = errD_fake + errD_fake_uncond


        # img-seg pair
        if len(real_logits) > 0 and cfg.TRAIN.COEFF.IS_LOSS > 0:
            errD_real_seg = cfg.TRAIN.COEFF.IS_LOSS * \
                            criterion(real_logits[2], real_labels)
            errD_wtxt_seg = cfg.TRAIN.COEFF.IS_LOSS * \
                            criterion(wtxt_logits[2], real_labels)
            errD_wseg_seg = cfg.TRAIN.COEFF.IS_LOSS * \
                            criterion(wseg_logits[2], fake_labels)
            errD_fake_seg = cfg.TRAIN.COEFF.IS_LOSS * \
                            criterion(fake_logits[2], fake_labels)
            #
            errD_real = errD_real + errD_real_seg
            errD_wtxt = errD_wtxt + errD_wtxt_seg
            errD_wseg = errD_wseg + errD_wseg_seg
            errD_fake = errD_fake + errD_fake_seg
            #
            errD = errD + errD_real + errD_wtxt + errD_wseg + errD_fake

      # backward
        errD.backward()
        # update parameters
        optD.step()

        return errD


    def train_Gnet(self):
        self.netG.zero_grad()
        errG_total = 0
        errG_pair =0
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]

        for i in range(self.num_Ds):

            outputs = self.netsD[i](self.fake_imgs[i], mu, self.fake_segs[i])

            errG = cfg.TRAIN.COEFF.ITS_LOSS*criterion(outputs[0], real_labels)
            errG_pair += errG

            if batch_size > 0 and cfg.TRAIN.COEFF.I_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.I_LOSS * \
                             criterion(outputs[1], real_labels)
                errG = errG + errG_patch

            if batch_size > 0 and cfg.TRAIN.COEFF.IS_LOSS > 0:
                errG_seg = cfg.TRAIN.COEFF.IS_LOSS * \
                           criterion(outputs[2], real_labels)
                errG = errG + errG_seg

                errG_total = errG_total + errG

            if cfg.TRAIN.COEFF.RC_LOSS > 0:
                # img64 - only BG reconstruction
                temp_seg = (self.fake_segs[0].detach())
                BG_mask = (temp_seg < 0).type(torch.FloatTensor)

                erode_mask = BG_mask.permute(0, 2, 3, 1).data.cpu().numpy()
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                for b_idx in range(batch_size):
                    erode_mask[b_idx, :, :, 0] = cv2.erode(erode_mask[b_idx, :, :, 0], kernel, iterations=1)

                BG_mask = Variable(torch.FloatTensor(erode_mask)).cuda()
                BG_mask = BG_mask.permute(0, 3, 1, 2)
                BG_mask = BG_mask.repeat(1, 3, 1, 1)
                BG_fake = torch.mul(self.fake_imgs[i], BG_mask)
                BG_img = torch.mul(self.crop_base[i], BG_mask)

                err_BG = cfg.TRAIN.COEFF.RC_LOSS * self.RC_criterion(BG_fake, BG_img)
                errG_total = errG_total + err_BG

        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()

        return kl_loss, errG_total, errG_pair, err_BG

    def train(self):
        self.netG, self.netsD, self.num_Ds,\
            self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.MSELoss()
        self.RC_criterion = nn.L1Loss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        self.SNAPSHOT_INTERVAL = self.num_batches * 35;
        print('save model each %i' %self.SNAPSHOT_INTERVAL)

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.real_imgs, self.real_segs, self.wrong_imgs, self.wrong_segs, \
                self.txt_embedding, self.crop_base = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                self.fake_imgs, self.fake_segs, self.mu, self.logvar = \
                    self.netG(noise, self.txt_embedding, self.crop_base)
                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                kl_loss, errG_total, errG_pair, errG_BG = self.train_Gnet()
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # for inception score
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                count = count + 1

                if count % self.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    self.fake_imgs, self.fake_segs, _, _ = \
                        self.netG(fixed_noise, self.txt_embedding, self.crop_base)
                    save_image((self.fake_imgs[-1].data + 1) * 0.5,'../examples/{}/Image/Image_ep{}.png'.format(self.output_dir, epoch + 1))
                    save_image((self.fake_segs[-1].data + 1) * 0.5, '../examples/{}/Image/Seg_ep{}.png'.format(self.output_dir, epoch + 1))
                    save_image((self.crop_base[-1].data + 1) * 0.5, '../examples/{}/Image/BASE_ep{}.png'.format(self.output_dir, epoch + 1))

                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions) > 50:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)

                        print('Inception mean:', mean, 'std', std, 'NLPP_mean', mean_nlpp)
                        predictions = []

            end_t = time.time()
            print('''[%d/%d][%d]
            Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Loss_pair: %.2f Loss_BG: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data[0], errG_total.data[0], kl_loss.data[0],
                     errG_pair.data[0], errG_BG.data[0], end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)


    def save_singleimages(self, images, segs, base_img, filenames,
                          save_dir, sentenceID, imsize):

        for i in range(segs.size(0)):
            result_img = Image.new('RGB', [3 * imsize, imsize])
            s_tmp = '%s/%s' % \
                    (save_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[0][i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            seg = segs[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            seg = torch.squeeze(seg, 0)
            segarr = seg.data.cpu().numpy()

            fake_im = Image.fromarray(ndarr)
            fake_seg = Image.fromarray(segarr)

            base = base_img[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            base_arr = base.permute(1, 2, 0).data.cpu().numpy()
            base_pil = Image.fromarray(base_arr)

            result_img.paste(im=base_pil, box=(0, 0))
            result_img.paste(im=fake_im, box=(imsize, 0))
            result_img.paste(im=fake_seg, box=(imsize * 2, 0))

            result_img.save(fullpath)

    def evaluate(self):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator

            self.num_Ds = cfg.TREE.BRANCH_NUM
            self.base_num = 135
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames, _ = data
                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)

                crop_vbase = []


                crop_base_imgs = torch.zeros(batch_size, 3, self.img_size, self.img_size)
                for step, (base_img_list) in enumerate(data[3]):
                    if cfg.DATASET_NAME.find('flower') != -1:
                        base_ix = random.randint(1, self.base_num)
                        base_img_name = '%s/%s.jpg' % (base_img_list, str(base_ix))
                    else:
                        temp_base_list = os.listdir(base_img_list)
                        base_ix = random.randint(0, len(temp_base_list) - 1)
                        base_img_name = '%s/%s.jpg' % (base_img_list, str(base_ix))

                    base_img = Image.open(base_img_name).convert('RGB')
                    crop_base = base_img.resize([self.img_size, self.img_size])

                    crop_base = Torchtransform(crop_base)
                    crop_base_imgs[step, :] = crop_base

                if cfg.CUDA:
                    crop_vbase.append(Variable(crop_base_imgs).cuda())
                else:
                    crop_vbase.append(Variable(crop_base_imgs))

                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                for i in range(embedding_dim):
                    fake_imgs, fake_segs, _, _ = netG(noise, t_embeddings[:, i, :], crop_vbase)
                    self.save_singleimages(fake_imgs, fake_segs[-1], crop_vbase[0],
                                           filenames, save_dir, i, self.img_size)
