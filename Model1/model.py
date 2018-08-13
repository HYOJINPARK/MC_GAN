
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo


################################   NOTICE   ###########################################

# This code is based from stackGAN ++
# https://github.com/hanzhanggit/StackGAN-v2

#######################################################################################



# ############################## For Compute inception score ##############################
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.
class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax()(x)
        return x


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class Synthesis_Block(nn.Module):
    def __init__(self, channel_num):
        super(Synthesis_Block, self).__init__()
        self.fg_block = nn.Sequential(
            conv3x3(channel_num, channel_num*2),
            nn.BatchNorm2d(channel_num*2),
            nn.LeakyReLU(0.02, inplace=True),
            conv3x3(channel_num*2, channel_num*2),
            nn.BatchNorm2d(channel_num*2)
        )
        self.channel = channel_num

    def forward(self, fg, bg):
        out = self.fg_block(fg)
        out_ = out[:,:self.channel]
        out_switch = F.sigmoid(out[:, self.channel:])
        actmap = torch.mean(out_switch,1)
        residual = torch.mul(bg, out_switch)
        out_block = out_ + residual

        return out_block


class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)

        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM

        self.k_size = cfg.TREE.BASE_SIZE // (2**4)
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        k_size =self.k_size
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size * 2, bias=False),
            nn.BatchNorm1d(ngf * k_size * k_size * 2),
            GLU())

        self.img_block1 = nn.Sequential(
            nn.Conv2d(3, ngf // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 8)
        )
        self.img_block2 = nn.Sequential(
            nn.Conv2d(ngf // 8, ngf // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 4)
        )
        self.img_block3 = nn.Sequential(
            nn.Conv2d(ngf // 4, ngf // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 2)
        )
        self.img_block4 = nn.Sequential(
            nn.Conv2d(ngf // 2, ngf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        self.synthesis1 = Synthesis_Block(ngf)
        self.upsample1 = upBlock(ngf, ngf // 2)

        self.synthesis2 = Synthesis_Block(ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

        self.synthesis3 = Synthesis_Block(ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)

        self.synthesis4 = Synthesis_Block(ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code, base_img):
        in_code = torch.cat((c_code, z_code), 1)


        fg_code = self.fc(in_code)
        fg_code = fg_code.view(-1, self.gf_dim, self.k_size, self.k_size)

        bg_code1 = self.img_block1(base_img)
        bg_code2 = self.img_block2(bg_code1)
        bg_code3 = self.img_block3(bg_code2)
        bg_code4 = self.img_block4(bg_code3)

        out_code1 = self.synthesis1(fg_code, bg_code4)
        out_code1 = self.upsample1(out_code1)

        out_code2 = self.synthesis2(out_code1, bg_code3)
        out_code2 = self.upsample2(out_code2)

        out_code3 = self.synthesis3(out_code2, bg_code2)
        out_code3 = self.upsample3(out_code3)

        out_code4 = self.synthesis4(out_code3, bg_code1)
        out_code4 = self.upsample4(out_code4)

        return out_code4


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf, out_dim):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.out_dim = out_dim
        self.img = nn.Sequential(
            conv3x3(ngf, out_dim),
            nn.Tanh()
        )

    def forward(self, h_code):
        img_set = self.img(h_code)
        if self.out_dim > 3:
            out_img = img_set[:, :3, :]
            out_seg = img_set[:, 3, :]
            out_seg = torch.unsqueeze(out_seg, 1)
            return out_img, out_seg
        else:
            return img_set


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

    def define_module(self):

        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim, 4)


    def forward(self, z_code, text_embedding, base_img):

        c_code, mu, logvar = self.ca_net(text_embedding)

        fake_imgs, fake_segs = [], []
        h_code1 = self.h_net1(z_code, c_code, base_img[0])
        fake_img1, fake_seg1 = self.img_net1(h_code1)

        fake_imgs.append(fake_img1)
        fake_segs.append(fake_seg1)

        return fake_imgs, fake_segs, mu, logvar


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image(ndf, indim=3):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(indim, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For 64 x 64 images
class D_NET(nn.Module):
    def __init__(self):
        super(D_NET, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.k_size = cfg.TREE.BASE_SIZE // (2**4)
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        k_size = self.k_size
        self.img_code_s16 = encode_image(ndf)
        self.imgseg_code_s16 = encode_image(ndf, 4)

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)

        self.logits_IST = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=k_size, stride=1))
        self.logits_I = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=k_size, stride=1))
        self.logits_IS = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=k_size, stride=1))

    def forward(self, img, txt, seg):
        imgseg_pair = torch.cat((img, seg), 1)
        txt_code = txt.view(-1, self.ef_dim, 1, 1)
        txt_code = txt_code.repeat(1, 1, self.k_size, self.k_size)

        # only img
        img_code = self.img_code_s16(img)
        # img seg pair
        imgseg_code = self.imgseg_code_s16(imgseg_pair)

        # imgsegtxt
        h_c_code1 = torch.cat((txt_code, imgseg_code), 1)
        h_c_code1 = self.jointConv(h_c_code1)

        D_imgsegtxt = self.logits_IST(h_c_code1)
        D_img = self.logits_I(img_code)
        D_imgseg = self.logits_IS(imgseg_code)
        return [D_imgsegtxt.view(-1), D_img.view(-1), D_imgseg.view(-1)]

