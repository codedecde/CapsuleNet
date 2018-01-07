import scipy.io as sio
import argparse
import numpy as np
from capsule import CapsuleNet
from torch.autograd import Variable
import torch.optim as optim
import pdb
import torch
from tqdm import tqdm
import torch.utils.data as data
from torchvision import datasets, transforms

NEPOCHS = 20
BATCH_SIZE = 32

parser = argparse.ArgumentParser(
    description='CapsuleNet with dynamic routing')
parser.add_argument('--cuda',
                    action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--train',
                    action='store_true', default=False,
                    help='train the Capsnet (default: False)')
parser.add_argument('--batch-size',
                    type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs',
                    type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr',
                    type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# ========= Set Seed for replication ==== #
np.random.seed(42)

# ========= MNIST DATALOADER ============ #
train_loader = data.DataLoader(datasets.MNIST('./data', train=True,
                                              download=True,
                                              transform=transforms.ToTensor()),
                               batch_size=BATCH_SIZE,
                               shuffle=True)
test_loader = data.DataLoader(datasets.MNIST('./data', train=False,
                                             transform=transforms.ToTensor()),
                              batch_size=BATCH_SIZE,
                              shuffle=False)

# ========= LOAD MNIST DATA ============= #
# mnist = sio.loadmat('Data/mnist-original.mat')
# data = mnist['data'].transpose()
# labels = mnist['label'].transpose()
# index = range(data.shape[0])
# np.random.shuffle(index)
# data = data[index]
# labels = labels[index].flatten()

# # TODO: Small data for debugging
# # data = data[:320]

# VAL_FRAC = 0.9
# train_x, train_y = data[:int(VAL_FRAC * data.shape[0])
#                         ], labels[: int(VAL_FRAC * data.shape[0])]
# val_x, val_y = data[int(VAL_FRAC * data.shape[0])
#                         :], labels[int(VAL_FRAC * data.shape[0]):]


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
        :param orig: batch x im_ht x im_wd
        :return rloss: The reconstruction loss
    """
    rloss = ((reconstruction - orig.view(orig.size(0),
                                         orig.size(1), -1)) ** 2).mean(-1)
    rloss = rloss.mean()
    return rloss


# ========= CapsuleNet ================== #
model = CapsuleNet()
if args.cuda:
    model.cuda()

opt = optim.Adam(model.parameters(), lr=args.lr)

# ========= TrainingLoop ================ #
loss_list = []
for epoch in xrange(args.epochs):
    for (batch_x, batch_y) in train_loader:
        batch_y_onehot = torch.FloatTensor(get_onehot(batch_y.numpy()))

        if args.cuda:
            batch_x, batch_y, batch_y_onehot = batch_x.cuda(
            ), batch_y.cuda(), batch_y_onehot.cuda()

        batch_x, batch_y, batch_y_onehot = Variable(batch_x), \
            Variable(batch_y), Variable(batch_y_onehot)

        digicaps = model(batch_x)

        print("digicaps : ", digicaps.size())
        print("batch_x : ", batch_x.size())
        print("batch_y : ", batch_y.size())
        print("batch_y_onehot : ", batch_y_onehot.size())
        reconstruction = model.reconstruct(digicaps, batch_y_onehot)

        print("reconstruction : ", reconstruction.size())
        mloss = margin_loss(digicaps, batch_y_onehot)
        rloss = reconstruction_loss(reconstruction, batch_x)
        loss = mloss + 0.0005 * rloss
        loss_list.append(loss.data[0])

        loss.backward()
        opt.step()
        opt.zero_grad()

        print "R Loss: %.4f\tM Loss: %.4f" % (rloss.data.numpy()[0], mloss.data.numpy()[0])

# ========= Validation ================ #
for (batch_x, batch_y) in test_loader:
    batch_y_onehot = torch.FloatTensor(get_onehot(batch_y.numpy()))

    if args.cuda:
        batch_x, batch_y, batch_y_onehot = batch_x.cuda(
        ), batch_y.cuda(), batch_y_onehot.cuda()

    batch_x, batch_y, batch_y_onehot = Variable(batch_x.no_grad()), \
        Variable(batch_y.no_grad()), Variable(batch_y_onehot.no_grad())

    digicaps = model(batch_x)

    _, pred = torch.max(torch.norm(digicaps, p=2, dim=2), dim=1)
    pred_onehot = Variable(torch.FloatTensor(get_onehot(pred.data.numpy())))
    reconstruction = model.reconstruct(digicaps, pred)

    mloss = margin_loss(digicaps, pred_onehot)
    rloss = reconstruction_loss(reconstruction, batch_x)
    loss += mloss + 0.0005 * rloss

print("Average Loss: %.4f".format(loss.data.numpy()[0] / len(test_loader)))
# if args.train:
#     loss_list = []
#     for epoch in xrange(NEPOCHS):
#         index = range(train_x.shape[0])
#         np.random.shuffle(index)
#         train_x, train_y = train_x[index], train_y[index]
#         steps = -(- train_x.shape[0] // BATCH_SIZE)
#         for step, ix in enumerate(xrange(0, train_x.shape[0], BATCH_SIZE)):
#             batch_x = train_x[ix: ix + BATCH_SIZE]
#             batch_x = Variable(torch.Tensor(batch_x.astype('float')))
#             batch_y = train_y[ix: ix + BATCH_SIZE]
#             batch_y_onehot = get_onehot(batch_y)
#             batch_y = Variable(torch.LongTensor(batch_y.astype('int')))
#             batch_y_onehot = Variable(torch.FloatTensor(batch_y_onehot))

#             if args.cuda:
#                 batch_x, batch_y, batch_y_onehot = batch_x.cuda(
#                 ), batch_y.cuda(), batch_y_onehot.cuda()

#             digicaps = model(batch_x.view(-1, 28, 28).unsqueeze(1))
#             reconstruction = model.reconstruct(digicaps, batch_y_onehot)
#             mloss = margin_loss(digicaps, batch_y_onehot)
#             rloss = reconstruction_loss(reconstruction, batch_x)
#             rloss = 0.0005 * rloss
#             loss = mloss + rloss
#             loss.backward()
#             opt.step()
#             opt.zero_grad()

#             loss_list.append(loss.data.numpy()[0])
#             print "R Loss: %.4f\tM Loss: %.4f" % (rloss.data.numpy()[0], mloss.data.numpy()[0])

# steps = -(- val_x.shape[0] // BATCH_SIZE)
# for step, ix in enumerate(xrange(0, val_x.shape[0], BATCH_SIZE)):
#     batch_x = val_x[ix: ix + BATCH_SIZE]
#     batch_x = Variable(torch.Tensor(
#         batch_x.astype('float')))
#     batch_y = val_y[ix: ix + BATCH_SIZE]
#     batch_y_onehot = get_onehot(batch_y)
#     # batch_y = Variable(torch.LongTensor(batch_y.astype('int')))
#     batch_y_onehot = Variable(torch.FloatTensor(batch_y_onehot))

#     if args.cuda:
#         batch_x, batch_y, batch_y_onehot = batch_x.cuda(
#         ), batch_y.cuda(), batch_y_onehot.cuda()

#     digicaps = model(batch_x.view(-1, 28, 28).unsqueeze(1))
#     _, pred = torch.max(torch.norm(digicaps, p=2, dim=2), dim=1)
#     pred_onehot = Variable(torch.FloatTensor(
#         get_onehot(pred.data.numpy())))
#     reconstruction = model.reconstruct(digicaps, pred_onehot)
#     # print("batch_y_onehot : ", batch_y_onehot.size())
#     # print("digicaps : ", digicaps.size())
#     # print("pred_onehot : ", pred_onehot.size())
#     mloss = margin_loss(digicaps, batch_y_onehot)
#     rloss = reconstruction_loss(reconstruction, batch_x)
#     rloss = 0.0005 * rloss
#     loss = mloss + rloss

# loss /= steps
# print "Avg Loss: %.4f\t" % (loss.data.numpy()[0])
# ==== Do validation stuff ============ #
