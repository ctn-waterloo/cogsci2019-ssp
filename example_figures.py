# This script will generate many of the explanatory figures used in the paper

import matplotlib.pyplot as plt
import nengo.spa as spa
import seaborn as sns
import numpy as np
from utils import encode_point, spatial_dot, get_heatmap_vectors, make_good_unitary
import os

if not os.path.exists('figures'):
    os.makedirs('figures')


plot_types = [
    'Single Item',
    'Two Items Decoded',
    'Animal Icons',
    'Sliding Objects',
]

seed = 13
dim = 512
limit = 5
res = 256

vmin=-1
vmax=1
cmap='plasma'

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

x_axis_sp = make_good_unitary(dim)
y_axis_sp = make_good_unitary(dim)

heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)


def plt_heatmap(vec, heatmap_vectors, name='', vmin=-1, vmax=1, cmap='plasma'):
    # vec has shape (dim) and heatmap_vectors have shape (xs, ys, dim) so the result will be (xs, ys)
    # the output is transposed and flipped so that it is displayed intuitively on the image plot
    vs = np.flip(np.tensordot(vec, heatmap_vectors, axes=([0], [2])).T, axis=0)

    if cmap == 'diverging':
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

    plt.imshow(vs, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()

    if name:
        plt.suptitle(name)


def heatmap(vec, heatmap_vectors, ax, name='', vmin=-1, vmax=1, cmap='plasma'):
    # vec has shape (dim) and heatmap_vectors have shape (xs, ys, dim) so the result will be (xs, ys)
    # the output is transposed and flipped so that it is displayed intuitively on the image plot
    vs = np.flip(np.tensordot(vec, heatmap_vectors, axes=([0], [2])).T, axis=0)

    if cmap == 'diverging':
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

    img = ax.imshow(vs, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_title(name)

    return img


# Same as the other function, but using the plt interface so the colorbar can be used
def plt_plot_similarity(vec, xs, ys, x_axis_sp, y_axis_sp, name='', vmin=-1, vmax=1, cmap='plasma'):
    # fig, ax = plt.subplots()

    vs = spatial_dot(
        vec=vec,
        xs=xs,
        ys=ys,
        x_axis_sp=x_axis_sp,
        y_axis_sp=y_axis_sp
    )

    if cmap == 'diverging':
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

    plt.imshow(vs, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()

    if name:
        plt.suptitle(name)


###############
# Single Item #
###############
if "Single Item" in plot_types:
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))

    coord_sp = encode_point(3, -2, x_axis_sp, y_axis_sp)

    heatmap(
        coord_sp.v,
        heatmap_vectors,
        ax,
        name="Single Object",
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    fig.savefig('figures/single_item.pdf', dpi=600, bbox_inches='tight')

#####################
# Two Items Decoded #
#####################
if "Two Items Decoded" in plot_types:
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))

    pos1 = encode_point(3, -2, x_axis_sp, y_axis_sp)
    pos2 = encode_point(-.3, 1.5, x_axis_sp, y_axis_sp)
    item1 = spa.SemanticPointer(dim)
    item2 = spa.SemanticPointer(dim)

    mem = pos1*item1 + pos2*item2

    decode1 = mem *~ item1
    decode2 = mem *~ item2

    heatmap(
        (decode1 + decode2).v,
        heatmap_vectors,
        ax,
        name='',
        vmin=vmin, vmax=vmax, cmap=cmap,
    )

    fig.savefig('figures/two_items.pdf', dpi=600, bbox_inches='tight')


if 'Sliding Objects' in plot_types:

    fig, ax = plt.subplots(1, 3, sharey='row', tight_layout=True, figsize=(9, 3))

    pos1 = encode_point(3, -2, x_axis_sp, y_axis_sp)
    pos2 = encode_point(4, 1, x_axis_sp, y_axis_sp)
    pos3 = encode_point(-1, 2, x_axis_sp, y_axis_sp)

    mem = pos1 + pos2 + pos3
    mem.normalize()

    # sliding all objects
    mem_moved = mem * encode_point(-2, 1, x_axis_sp, y_axis_sp)

    title_font_size = 16

    heatmap(
        vec=mem.v,
        heatmap_vectors=heatmap_vectors,
        ax=ax[0],
        name="Original Memory Contents",
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    ax[0].set_title("Original Memory Contents", fontsize=title_font_size)

    heatmap(
        vec=mem_moved.v,
        heatmap_vectors=heatmap_vectors,
        ax=ax[2],
        name="All Items Moved",
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    ax[2].set_title("All Items Moved", fontsize=title_font_size)

    # sliding single object
    new_pos = pos3 * encode_point(-2, 1, x_axis_sp, y_axis_sp)
    mem_single_moved = mem - pos3 + new_pos

    img = heatmap(
        vec=mem_single_moved.v,
        heatmap_vectors=heatmap_vectors,
        ax=ax[1],
        name="Single Item Moved",
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    ax[1].set_title("Single Item Moved", fontsize=title_font_size)

    fig.savefig('figures/sliding_objects.pdf', dpi=600, bbox_inches='tight')

if 'Animal Icons' in plot_types:

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    def getImage(path, zoom=0.3):
        return OffsetImage(plt.imread(path), zoom=zoom)

    root = 'images'
    paths = [
        root + '/icons8-fox-96.png',
        root + '/icons8-fox-96.png',
        root + '/icons8-pug-96.png',
        root + '/icons8-badger-96.png',
        root + '/icons8-bear-96.png',
    ]

    x = np.array([1.2, -3.4, 1.7, 4.1, 2.1])
    y = np.array([1.3, -1.1, -1.1, 3.2, 2.4])

    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    ax.scatter(y, -x)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal', 'box')

    artists = []
    for x0, y0, path in zip(x, y,paths):
        ab = AnnotationBbox(getImage(path), (y0, -x0), frameon=False)
        artists.append(ax.add_artist(ab))

    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('Memory Contents')

    fig.savefig('figures/example_image.pdf', dpi=600, bbox_inches='tight')

