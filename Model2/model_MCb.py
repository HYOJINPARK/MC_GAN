import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

################################   NOTICE   ###########################################

# This code is based from stackGAN ++
# https://github.com/hanzhanggit/StackGAN-v2
# https://github.com/woozzu/dong_iccv_2017

#######################################################################################

class VisualSemanticEmbedding(nn.Module):
    def __init__(self, embed_ndim):
        super(VisualSemanticEmbedding, self).__init__()
        self.embed_ndim = embed_ndim

        # image feature
        self.img_encoder = models.vgg16(pretrained=True)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.feat_extractor = nn.Sequential(*(self.img_encoder.classifier[i] for i in range(6)))
        self.W = nn.Linear(4096, embed_ndim, False)

        # text feature
        self.txt_encoder = nn.GRU(embed_ndim, embed_ndim, 1)

    def forward(self, img, txt):
        # image feature
        img_feat = self.img_encoder.features(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.feat_extractor(img_feat)
        img_feat = self.W(img_feat)

        # text feature
        h0 = torch.zeros(1, img.size(0), self.embed_ndim)
        h0 = Variable(h0.cuda() if txt.data.is_cuda else h0)
        _, txt_feat = self.txt_encoder(txt, h0)
        txt_feat = txt_feat.squeeze()

        return img_feat, txt_feat


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

    # Upsale the spatial size by a factor of 2

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, args):
        super(CA_NET, self).__init__()
        self.t_dim = args.embed_ndim
        self.ef_dim = args.FG_emb
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()
        self.is_cuda = not (args.no_cuda)

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        # print('CA net')
        # print(mu.size())
        # print(logvar.size())
        # print(c_code.size())
        return c_code, mu, logvar

class Synthesis_Block(nn.Module):
    def __init__(self, channel_num):
        super(Synthesis_Block, self).__init__()
        self.fg_block = nn.Sequential(
            conv3x3(channel_num, channel_num*2),
            nn.BatchNorm2d(channel_num*2),
            nn.LeakyReLU(0.2, inplace=True),
            conv3x3(channel_num*2, channel_num*2),
            nn.BatchNorm2d(channel_num*2)
        )

        self.channel = channel_num

    def forward(self, fg, bg):
        out = self.fg_block(fg)
        out_ = out[:,:self.channel]
        out_switch = F.sigmoid(out[:, self.channel:])
        residual = torch.mul(bg, out_switch)
        out_block = out_ + residual
        return out_block

class INIT_STAGE_G(nn.Module):
    def __init__(self, args):
        super(INIT_STAGE_G, self).__init__()
        self.in_dim = args.FG_emb + args.z_dim
        self.txt_dim = args.FG_emb
        self.gf_dim = args.ngf

        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
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


    def forward(self, z_code, txt_feat, base_img):
        # print(txt_feat.size())
        # print(z_code.size())
        in_code = torch.cat((txt_feat, z_code), 1)

        # state size 16ngf x 4 x 4
        fg_code = self.fc(in_code)
        fg_code = fg_code.view(-1, self.gf_dim, 4, 4)

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
    def __init__(self,args):
        super(G_NET, self).__init__()

        self.ca_net = CA_NET(args)
        self.h_net1 = INIT_STAGE_G(args)
        self.img_net1 = GET_IMAGE_G(args.ngf// 16, 3)

    def forward(self, z_code, text_embedding, base_img):
        c_code, mu, logvar = self.ca_net(text_embedding)
        h_code1 = self.h_net1(z_code, c_code, base_img)
        fake_img = self.img_net1(h_code1)

        return fake_img, mu, logvar


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
def encode_image_by_16times(ndf, indim=3):
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
class D_NET64(nn.Module):
    def __init__(self, args):
        super(D_NET64, self).__init__()
        self.df_dim = args.ndf
        self.ef_dim = args.embed_ndim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.imgseg_code_s16 = encode_image_by_16times(ndf, 4)

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.logits_IT = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1))

    def forward(self, img, txt):
        txt_code = txt.view(-1, self.ef_dim, 1, 1)
        txt_code = txt_code.repeat(1, 1, 4, 4)

        # only img
        img_code = self.img_code_s16(img)

        # imgtxt
        h_c_code = torch.cat((txt_code, img_code), 1)
        h_c_code = self.jointConv(h_c_code)
        D_imgtxt = self.logits_IT(h_c_code)

        return [D_imgtxt.view(-1)]

