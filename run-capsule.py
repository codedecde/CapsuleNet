import scipy.io as sio
import argparse
import numpy as np
from capsule import CapsuleNet
from torch.autograd import Variable
import torch.optim as optim
import pdb
import torch
from tqdm import tqdm

NEPOCHS = 20
BATCH_SIZE = 1000

parser = argparse.ArgumentParser(
    description='CapsuleNet with dynamic routing')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--train', action='store_true', default=False,
                    help='train the Capsnet (default: False)')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# ========= Set Seed for replication ==== #
np.random.seed(42)

# ========= MNIST DATALOADER ============ #
# train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True,
#                                                           download=True,
#                                                           transform=transforms.ToTensor()),
#                                            batch_size=BATCH_SIZE,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False,
#                                                          transform=transforms.ToTensor()),
#                                           batch_size=BATCH_SIZE,
#                                           shuffle=True)

# ========= LOAD MNIST DATA ============= #
mnist = sio.loadmat('Data/mnist-original.mat')
data = mnist['data'].transpose()
labels = mnist['label'].transpose()
index = range(data.shape[0])
np.random.shuffle(index)
data = data[index]
labels = labels[index].flatten()

# TODO: Small data for debugging
# data = data[:320]

VAL_FRAC = 0.9
train_x, train_y = data[:int(VAL_FRAC * data.shape[0])
                        ], labels[: int(VAL_FRAC * data.shape[0])]
val_x, val_y = data[int(VAL_FRAC * data.shape[0])
                        :], labels[int(VAL_FRAC * data.shape[0]):]


def get_onehot(tensor, labels=10):
    """
    Converts tensor to onehot tensor
        :param tensor: batch, : The indices of labels
        :param labels: int: Number of labels
        :returns one_hot: batch x labels: One hot encoding for the labels
    """
    one_hot = np.zeros((tensor.shape[0], labels))
    one_hot[range(tensor.shape[0]), tensor.astype(int)] = 1.
    return one_hot


def margin_loss(digicaps, labels_onehot):
    def margin_mask(tensor):
        """
        computes max(0, tensor_val)
        """
        mask = torch.gt(tensor, 0).float()
        return tensor * mask
    """
    The margin loss, as defined in the paper.
        :param digicaps: batch x 10 x 16
        :param labels: batch x num_labels
    """
    norm_vec = torch.sqrt((digicaps ** 2).sum(-1))
    m_pos = 0.9
    m_neg = 0.1
    lbda = 0.5
    # pdb.set_trace()
    m1 = margin_mask(m_pos - norm_vec) ** 2
    m2 = margin_mask(norm_vec - m_neg) ** 2
    mloss = ((labels_onehot * m1) + (lbda * (1. - labels_onehot) * m2)).sum(-1)
    mloss = mloss.sum(-1).mean()
    return mloss


def reconstruction_loss(reconstruction, orig):
    """
    Computes the reconstruction loss
        :param reconstruction: batch x im_size
        :param orig: batch x im_size
        :return rloss: The reconstruction loss
    """
    rloss = ((reconstruction - orig) ** 2).mean(-1)
    rloss = rloss.mean()
    return rloss


# ========= CapsuleNet ================== #
model = CapsuleNet()
if args.cuda:
    model.cuda()

opt = optim.Adam(model.parameters(), lr=0.01)

# ========= TrainingLoop ================ #

# for epoch in xrange(NEPOCHS):
#     for (batch_x, batch_y) in train_loader:
#         if args.cuda:
#             batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

#         batch_y_onehot = get_onehot(batch_y)

#         batch_x, batch_y, batch_y_onehot = Variable(batch_x), \
#             Variable(batch_y), Variable(batch_y_onehot)

#         digicaps = model(batch_x)
#         reconstruction = model.reconstruct(digicaps, batch_y)
#         mloss = margin_loss(digicaps, batch_y_onehot)
#         rloss = reconstruction_loss(reconstruction, batch_x)
#         loss = mloss + 0.0005 * rloss
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#         print loss.data.numpy()[0]

if args.train:
    loss_list = []
    for epoch in xrange(NEPOCHS):
        index = range(train_x.shape[0])
        np.random.shuffle(index)
        train_x, train_y = train_x[index], train_y[index]
        steps = -(- train_x.shape[0] // BATCH_SIZE)
        for step, ix in enumerate(xrange(0, train_x.shape[0], BATCH_SIZE)):
            batch_x = train_x[ix: ix + BATCH_SIZE]
            batch_x = Variable(torch.Tensor(batch_x.astype('float')))
            batch_y = train_y[ix: ix + BATCH_SIZE]
            batch_y_onehot = get_onehot(batch_y)
            batch_y = Variable(torch.LongTensor(batch_y.astype('int')))
            batch_y_onehot = Variable(torch.FloatTensor(batch_y_onehot))

            if args.cuda:
                batch_x, batch_y, batch_y_onehot = batch_x.cuda(
                ), batch_y.cuda(), batch_y_onehot.cuda()

            digicaps = model(batch_x.view(-1, 28, 28).unsqueeze(1))
            reconstruction = model.reconstruct(digicaps, batch_y_onehot)
            mloss = margin_loss(digicaps, batch_y_onehot)
            rloss = reconstruction_loss(reconstruction, batch_x)
            rloss = 0.0005 * rloss
            loss = mloss + rloss
            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_list.append(loss.data.numpy()[0])
            print "R Loss: %.4f\tM Loss: %.4f" % (rloss.data.numpy()[0], mloss.data.numpy()[0])

    steps = -(- val_x.shape[0] // BATCH_SIZE)
    for step, ix in enumerate(xrange(0, val_x.shape[0], BATCH_SIZE)):
        batch_x = val_x[ix: ix + BATCH_SIZE]
        batch_x = Variable(torch.Tensor(
            batch_x.astype('float')))
        batch_y = val_y[ix: ix + BATCH_SIZE]
        batch_y_onehot = get_onehot(batch_y)
        # batch_y = Variable(torch.LongTensor(batch_y.astype('int')))
        batch_y_onehot = Variable(torch.FloatTensor(batch_y_onehot))

        if args.cuda:
            batch_x, batch_y, batch_y_onehot = batch_x.cuda(
            ), batch_y.cuda(), batch_y_onehot.cuda()

        digicaps = model(batch_x.view(-1, 28, 28).unsqueeze(1))
        _, pred = torch.max(torch.norm(digicaps, p=2, dim=2), dim=1)
        pred_onehot = Variable(torch.FloatTensor(
            get_onehot(pred.data.numpy())))
        reconstruction = model.reconstruct(digicaps, pred_onehot)
        # print("batch_y_onehot : ", batch_y_onehot.size())
        # print("digicaps : ", digicaps.size())
        # print("pred_onehot : ", pred_onehot.size())
        mloss = margin_loss(digicaps, batch_y_onehot)
        rloss = reconstruction_loss(reconstruction, batch_x)
        rloss = 0.0005 * rloss
        loss = mloss + rloss

    loss /= steps
    print "Avg Loss: %.4f\t" % (loss.data.numpy()[0])
    # ==== Do validation stuff ============ #
