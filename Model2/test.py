import os
import argparse
import torchtext.vocab as vocab

from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from model_MCb import VisualSemanticEmbedding
# from model import Generator
from model_MCb import G_NET
from data_MCb import split_sentence_into_words
import os
import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, default='test/birds',
                    help='root directory that contains images')
parser.add_argument('--text_file', type=str, default='test/text_birds.txt',
                    help='text file that contains descriptions')
parser.add_argument('--caption_root', type=str, default='../../../DATA/cub_200_2011/cub/cub_ReedScott',
                    help='root directory that contains captions')

parser.add_argument('--NET_G', type=str, default='G_model/MC_bird_600.pth',
                    help='Net_G')
parser.add_argument('--NET_D', type=str, default='',
                    help='Net_D')

parser.add_argument('--text_embedding_model', type=str, default='text_emb/birds.pth',
                    help='pretrained text embedding model')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 64)')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--z_dim', type = int, default=100,
                    help='noise dimension for FG')
parser.add_argument('--FG_emb', type = int, default=128,
                    help='noise dimension for FG')
parser.add_argument('--ngf', type = int, default=1024,
                    help='number of generator base feature')
parser.add_argument('--output_root', type=str, default ='test/MCb',
                    help='root directory of output')
parser.add_argument('--no_cuda', type = bool, default=False,
                    help='do not use cuda')
parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
args = parser.parse_args()
cudnn.benchmark = True

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True
    cudnn.benchmark = False
s_gpus = args.gpu_id.split(',')
gpus = [int(ix) for ix in s_gpus]

if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = vocab.FastText(language="en")

    print('Loading a pretrained model...')
    txt_encoder = VisualSemanticEmbedding(args.embed_ndim)
    txt_encoder.load_state_dict(torch.load(args.text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder

    #G = Generator(args)
    G = G_NET(args)
    # G = torch.nn.DataParallel(G, device_ids=gpus)
    state_dict = torch.load(args.NET_G)
    G.load_state_dict(state_dict)
    print('Load ', args.NET_G)
    nz = args.z_dim
    noise = Variable(torch.FloatTensor(args.batch_size, nz))

    if not args.no_cuda:
        txt_encoder.cuda()
        G.cuda()
        noise = noise.cuda()


    transform = transforms.Compose([
        transforms.Scale(74),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print('Loading test data...')
    filenames = os.listdir(args.img_root)
    img = []
    for fn in filenames:
        im = Image.open(os.path.join(args.img_root, fn))
        im = transform(im)
        img.append(im)
    img = torch.stack(img)
    mkdir_p(args.output_root)
    save_image((img+1)*0.5, os.path.join(args.output_root, 'original.jpg'))
    img = Variable(img.cuda() if not args.no_cuda else img, volatile=True)

    html = '<html><body><h1>Manipulated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Description</b></td><td><b>Image</b></td></tr>'
    html += '\n<tr><td>ORIGINAL</td><td><img src="{}"></td></tr>'.format('original.jpg')
    with open(args.text_file, 'r') as f:
        texts = f.readlines()
    G.eval()
    for i, txt in enumerate(texts):
        txt = txt.replace('\n', '')
        temp_desc = split_sentence_into_words(txt)
        word_vecs = torch.FloatTensor(len(temp_desc), 300).zero_()
        step = 0

        for w in temp_desc:
            if w in word_embedding.stoi:
                temp = word_embedding.vectors[word_embedding.stoi[w]]
                word_vecs[step, :] = torch.FloatTensor(temp)
                step = step + 1

        desc = word_vecs.unsqueeze(1)
        desc = desc.repeat(1, img.size(0), 1)
        desc = Variable(desc.cuda() if not args.no_cuda else desc, volatile=True)

        _, txt_feat = txt_encoder(desc)
        txt_feat = txt_feat.squeeze(0)
        noise.data.normal_(0, 1)
        this_noise = noise[:img.size(0), :]
        if(128< img.size(0)):
            print('have problem!!!!!!!!!!!!!!')
        output, _, _ = G(this_noise, txt_feat, img)

        out_filename = 'output_%d.jpg' % i
        save_image((output.data + 1) * 0.5, os.path.join(args.output_root, out_filename))
        html += '\n<tr><td>{}</td><td><img src="{}"></td></tr>'.format(txt, out_filename)

    with open(os.path.join(args.output_root, 'index.html'), 'w') as f:
        f.write(html)
    print('Done. The results were saved in %s.' % args.output_root)
