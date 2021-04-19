'''Help plot things, mostly images, in an IPython notebook'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sqrd_dims(n, nrows=None):
    '''Smallest sqaured bounds containing area n'''
    if nrows is None:
        nrows = int(np.floor(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    return nrows, ncols

def format_plot(ax):
    # hides noise, also making unused plots invisible
    ax.axis('off')
    ax.set_aspect('equal')

def split_at(items, i):
    return items[:i], items[i:]

def imshow_bgr2rgb(img, ax):
    # my most common plotting need

    # ax.imshow(img[:,:,::-1])

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)

def make_grid_of_plots(nplots, nrows=None):
    if nplots == 1:
        fig, ax = plt.subplots()
        return [ax,], (1, 1)

    nrows, ncols = sqrd_dims(nplots, nrows)

    fig, axs = plt.subplots(nrows, ncols)
    desired, extra = split_at( axs.flatten(), nplots )
    # hide extras
    for ax in extra:
        format_plot(ax)
    return desired, (nrows, ncols)

def imshow(*imgs, show_axis=False):
    # convenience to show any number of images, whether in grayscale or bgr

    n = len(imgs)
    axes, (nrows, ncols) = make_grid_of_plots(n)
    # prep imgs
    for img, ax in zip(imgs, axes):
        if len(img.shape) == 2:
            # grayscale, lacking dimension
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # rgb_img = np.stack((img,)*3, axis=-1)
        else:
            # opencv default img is bgr
            rgb_img = img[:,:,::-1]

        if show_axis:
            ax.axis('on')
        else:
            ax.axis('off')
        ax.imshow(rgb_img)

    # plt.figure(figsize=(1.5*nrows, 1.5*ncols))
    # default_width = 6
    # adjusted_height = default_width * nrows/ncols
    # plt.rcParams['figure.figsize'] = (2*ncols, 2*nrows)
    # plt.show()

def plot_a_lot(items, plot_fn, nrows=None, vert_spacing=None):
    # more flexible version of imshow, for any plots

    # items: list of things to show
    # plot_fn(item, ax): do whatever necessary to show the thing
    # nrows: set if manually changing metaplot dimension
    # vert_spacing: set when many subplots skew margins; try 0.7?

    axs, (nrows, ncols) = make_grid_of_plots(len(items), nrows)

    # Plot the things
    for item, ax in zip(items, axs):
        format_plot(ax)
        plot_fn(item, ax)

    if vert_spacing:
        plt.subplots_adjust(hspace=vert_spacing)
#     plt.subplots_adjust(hspace=0)
    plt.rcParams['figure.figsize'] = (2*ncols, 2*nrows)
    plt.show()
