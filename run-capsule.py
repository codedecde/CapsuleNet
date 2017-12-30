import scipy.io as sio
import numpy as np
from capsule import CapsuleNet
from torch.autograd import Variable
import torch.optim as optim
import pdb
import torch

# ========= Set Seed for replication ==== #
np.random.seed(42)
# ========= LOAD MNIST DATA ============= #
mnist = sio.loadmat('Data/mnist-original.mat')
data = mnist['data'].transpose()
labels = mnist['label'].transpose()
index = range(data.shape[0])
np.random.shuffle(index)
data = data[index]
labels = labels[index].flatten()

VAL_FRAC = 0.9
train_x, train_y = data[:int(VAL_FRAC * data.shape[0])
                        ], labels[:int(VAL_FRAC * data.shape[0])]
val_x, val_y = data[int(VAL_FRAC * data.shape[0])                    :], labels[int(VAL_FRAC * data.shape[0]):]


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
    rloss = ((reconstruction - orig) ** 2).sum(-1)
    rloss = rloss.mean()
    return rloss


# ========= CapsuleNet ================== #
model = CapsuleNet()
opt = optim.Adam(model.parameters(), lr=0.01)
# ========= TrainingLoop ================ #
NEPOCHS = 20
BATCH_SIZE = 16
for epoch in xrange(NEPOCHS):
    index = range(train_x.shape[0])
    np.random.shuffle(index)
    train_x, train_y = train_x[index], train_y[index]
    steps = -(- train_x.shape[0] // BATCH_SIZE)
    for step, ix in enumerate(xrange(0, train_x.shape[0], BATCH_SIZE)):
        batch_x = train_x[ix: ix + BATCH_SIZE]
        batch_x = Variable(torch.Tensor(batch_x))
        batch_y = train_y[ix: ix + BATCH_SIZE]
        batch_y_onehot = get_onehot(batch_y)
        batch_y = Variable(torch.LongTensor(batch_y))
        batch_y_onehot = Variable(torch.FloatTensor(batch_y_onehot))
        digicaps = model(batch_x.view(-1, 28, 28).unsqueeze(1))
        reconstruction = model.reconstruct(digicaps, batch_y)
        mloss = margin_loss(digicaps, batch_y_onehot)
        rloss = reconstruction_loss(reconstruction, batch_x)
        loss = mloss + 0.0005 * rloss
        loss.backward()
        opt.step()
        opt.zero_grad()
        print loss.data.numpy()[0]
    # ==== Do validation stuff ============ #
