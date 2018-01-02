import torch
import pdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# TODO: Speed up dynamic routing, still takes an insane amount of time
# TODO: GPU Support
# TODO: Validation loop
# TODO: Pytorch dataloaders

class CapsuleNet(nn.Module):
    def __init__(self, **kwargs):
        super(CapsuleNet, self).__init__()
        inp_h, inp_w = 28, 28
        # Conv 1 info
        self.conv1_s = 1
        self.conv1_c = 256
        self.conv1_f = 9
        self.conv1 = nn.Conv2d(
            1, self.conv1_c, self.conv1_f, stride=self.conv1_s)

        conv_h = self.get_conv_dim(inp_h, self.conv1_f, self.conv1_s)  # 20
        conv_w = self.get_conv_dim(inp_w, self.conv1_f, self.conv1_s)  # 20
        # PrimaryCaps info
        self.pcaps_n = 32
        self.pcaps_s = 2
        self.pcaps_d = 8
        self.pcaps_f = 9

        self.pcaps_h = self.get_conv_dim(
            conv_h, self.pcaps_f, self.pcaps_s)  # 6
        self.pcaps_w = self.get_conv_dim(
            conv_w, self.pcaps_f, self.pcaps_s)  # 6

        self.pcap = nn.Conv2d(self.conv1_c, self.pcaps_n * self.pcaps_d,
                              kernel_size=self.pcaps_f, stride=self.pcaps_s)

        # DigiCaps Layer
        self.dcaps_n = 10
        self.dcaps_d = 16
        self.n_iter = 3

        # W = 10 x 1152 x (8 x 16)
        self.W = nn.Parameter(torch.Tensor(np.random.normal(0, 0.01, (self.dcaps_n,
                                                                      self.pcaps_h * self.pcaps_w * self.pcaps_n,
                                                                      self.pcaps_d,
                                                                      self.dcaps_d))))
        self.register_parameter('W_ij', self.W)
        self.reconstruction_dims = [
            ('relu', 512), ('relu', 1024), ('sigmoid', 784)]
        in_dim = self.dcaps_d * self.dcaps_n
        for ix, (activation, out_dim) in enumerate(self.reconstruction_dims):
            layer_name = 'reconstruction_%d' % (ix)
            layer = nn.Linear(in_dim, out_dim)
            setattr(self, layer_name, layer)
            in_dim = out_dim

    def get_conv_dim(self, l, fs, s, p=0):
        """
        Gets dimension after a filter application.
            :param l: Length along the dimension
            :param fs: Filter size
            :param s: Stride
            :param p: Padding
        """
        return ((l + 2 * p - fs) // s) + 1

    def squash(self, t, axis):
        """
        Performs the squashing operation
            :param t: Tensor to squash
            :param axis: axis along which to squash
        """
        norm_t_sq = (t * t).sum(axis, keepdim=True)
        norm_t = torch.sqrt(norm_t_sq + 1e-6)
        ret_t = norm_t_sq / (1. + norm_t_sq) * (t / norm_t)
        return ret_t

    def forward(self, inp):
        """
        The forward pass
            :param inp: batch x 1 x 28 x 28: The input image
            :return v: batch x 10 x 16: The digicaps layer
        """

        # print(inp.size())
        conv1 = F.relu(self.conv1(inp))  # batch x 256 x 20 x 20

        # print(conv1.size())
        pcaps = self.pcap(conv1)  # batch x 256 x 6 x 6

        # print(pcaps.size())
        pcaps = pcaps.view(-1, self.pcaps_n, self.pcaps_d,
                           self.pcaps_h, self.pcaps_w)
        # BATCH X 32 X 8 X 6 X 6

        caps = self.squash(pcaps, axis=2)
        caps = caps.transpose(2, -1)  # batch x 32 x 6 x 6 x 8
        caps = caps.contiguous().view(caps.size(0), -1, caps.size(-1)
                                      ).unsqueeze(1)  # batch x 1152 x 8

        # Now the DigiCaps Layer
        caps_prime = caps.expand(
            caps.size(0), self.dcaps_n, caps.size(2), caps.size(3)).contiguous(
        ).view(-1, caps.size(-1)).unsqueeze(1)  # batch * 10 * 1152 x 1 x 8

        w_prime = self.W.unsqueeze(0).expand(caps.size(0), self.W.size(0), self.W.size(1), self.W.size(2), self.W.size(3)).contiguous(
        ).view(-1, self.W.size(-2), self.W.size(-1))  # batch * 10 *1152 x 8 x 16

        u_hat = torch.bmm(caps_prime, w_prime).squeeze(1).view(
            caps.size(0), self.dcaps_n, caps.size(2), -1)  # batch x 10 x 1152 x 16

        # print(u_hat.size())
        # batch x 10 x 1152
        b = Variable(torch.zeros((caps.size(0), self.dcaps_n, caps.size(2))))
        # print("b  : ", b.size())
        if torch.cuda.is_available():
            b = b.cuda()
        # Setting up routing
        for i in xrange(self.n_iter):
            c = F.softmax(b, dim=-1)
            s = (c.unsqueeze(-1) * u_hat).sum(2)
            v = self.squash(s, -1)  # batch x 10 x 16
            a = (u_hat * v.unsqueeze(2)).sum(-1)
            b = b + a
            # print("v : ", v.size())
        return v

    def reconstruct(self, digicaps, gold_labels):
        """
        Reconstructs the image based on the gold label
            :param digicaps: batch x 10 x 16: The digicaps layer
            :param gold_labels: batch x 10 : The one hot gold labels (Torch Variable)
            :return masked_v: batch x 784: The image
        """
        # idx = gold_labels.unsqueeze(
        #    1).expand(gold_labels.size(0), digicaps.size(-1)).unsqueeze(1)
        # masked_v = torch.gather(digicaps, 1, idx).squeeze(1)
        masked_v = (digicaps * gold_labels.unsqueeze(-1)
                    ).view(digicaps.size(0), -1)  # batch x 160
        # self.reconstruction_dims = [('relu', 512), ('relu', 1024), ('sigmoid', 784)]
        for ix, (activation, _) in enumerate(self.reconstruction_dims):
            layer_name = 'reconstruction_%d' % (ix)
            masked_v = getattr(F, activation)(
                getattr(self, layer_name)(masked_v))
        return masked_v
