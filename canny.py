from scipy.misc import imread, imsave
import torch
from torch.autograd import Variable
from net_canny import Net
import numpy as np


def canny(raw_img, use_cuda=False, save_all_img=False):
    if isinstance(raw_img, np.ndarray):
        img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    else:
        img = raw_img
    if len(img.shape) == 3:
        batch = img.unsqueeze(0).float()
    else:
        batch = img.float()

    if img.max() > 1:
        thr = 1000
    else:
        thr = 3

    net = Net(threshold=thr, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()

    net.eval()

    if use_cuda:
        data = Variable(batch).cuda()
    else:
        data = Variable(batch)

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)
    if save_all_img:
        imsave('gradient_magnitude.png', grad_mag.data.cpu().numpy()[0, 0])
        imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
        imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
        imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])

    return (thresholded.data.cpu() > 0.0).to(torch.float)


if __name__ == '__main__':
    img = imread('fb_profile.jpg') / 255.0

    # canny(img, use_cuda=False)
    canny(img, use_cuda=True)
