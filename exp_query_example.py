import matplotlib.pyplot as plt
import nengo.spa as spa
import seaborn as sns
import numpy as np
import pandas as pd
from utils import circular_region, arc_region, encode_point, get_heatmap_vectors, \
    generate_region_vector, make_good_unitary
import os

dim = 512
limit = 5
res = 128
n_seeds = 50

if not os.path.exists('figures'):
    os.makedirs('figures')

folder = 'figures/example_{0}D_{1}seeds'.format(dim, n_seeds)

vmin=-1
vmax=1
cmap='plasma'

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

desired_circle = circular_region(xs=xs, ys=xs, radius=2, x_offset=0, y_offset=0)
desired_arc = arc_region(xs=xs, ys=ys, center=2 * np.pi / 8, arc=np.pi / 2.)

vocab_labels = ['Fox', 'Dog', 'Badger', 'Bear', 'Wolf']
location_labels = ['Fox', 'Dog', 'Badger', 'Bear', 'None']
n_animals = len(vocab_labels)
vocab_vectors = np.zeros((n_animals, dim))

# Heatmaps for object queries
avg_heatmaps = []

# Similarity to location queries
sim_locations = []

for i in range(len(vocab_labels)):
    avg_heatmaps.append(np.zeros((len(xs), len(ys))))
    sim_locations.append(np.zeros((n_seeds, n_animals)))

avg_heatmap_circle = np.zeros((len(xs), len(ys)))
avg_heatmap_arc = np.zeros((len(xs), len(ys)))

sim_circles = np.zeros((n_seeds, n_animals))
sim_arcs = np.zeros((n_seeds, n_animals))

if not os.path.exists(folder):
    # Data is not saved already, generate it now and save it after

    for seed in range(n_seeds):

        rstate = np.random.RandomState(seed=seed)
        x_axis_sp = make_good_unitary(dim, rng=rstate)
        y_axis_sp = make_good_unitary(dim, rng=rstate)

        heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

        vocab_sps = {}
        for i, animal in enumerate(vocab_labels):
            vocab_sps[animal] = spa.SemanticPointer(dim)
            vocab_vectors[i, :] = vocab_sps[animal].v

        mem = spa.SemanticPointer(data=np.zeros(dim))

        fox_pos1 = encode_point(1.2, 1.3, x_axis_sp, y_axis_sp)
        fox_pos2 = encode_point(-3.4, -1.1, x_axis_sp, y_axis_sp)
        dog_pos = encode_point(1.7, -1.1, x_axis_sp, y_axis_sp)
        badger_pos = encode_point(4.1, 3.2, x_axis_sp, y_axis_sp)
        bear_pos = encode_point(2.1, 2.4, x_axis_sp, y_axis_sp)
        none_pos = encode_point(0, 0, x_axis_sp, y_axis_sp)

        mem += vocab_sps['Fox'] * fox_pos1
        mem += vocab_sps['Fox'] * fox_pos2
        mem += vocab_sps['Dog'] * dog_pos
        mem += vocab_sps['Badger'] * badger_pos
        mem += vocab_sps['Bear'] * bear_pos

        mem.normalize()

        for i, animal in enumerate(vocab_labels):
            avg_heatmaps[i] += np.tensordot((mem * ~vocab_sps[animal]).v, heatmap_vectors, axes=([0], [2]))

        sim_locations[0][seed, :] = np.tensordot((mem * ~fox_pos1).v, vocab_vectors, axes=([0], [1]))  # asking for only one of the cat positions
        sim_locations[1][seed, :] = np.tensordot((mem * ~dog_pos).v, vocab_vectors, axes=([0], [1]))
        sim_locations[2][seed, :] = np.tensordot((mem * ~badger_pos).v, vocab_vectors, axes=([0], [1]))
        sim_locations[3][seed, :] = np.tensordot((mem * ~bear_pos).v, vocab_vectors, axes=([0], [1]))
        sim_locations[4][seed, :] = np.tensordot((mem * ~none_pos).v, vocab_vectors, axes=([0], [1]))  # asking for a location where no animal exists

        # Note: x and y are sometimes mixed up on the plots
        desired_circle = circular_region(xs=xs, ys=xs, radius=2, x_offset=0, y_offset=0)
        desired_arc = arc_region(xs=xs, ys=ys, center=2 * np.pi / 8, arc=np.pi / 2.)

        # note that the threshold may need to change based on the area of the region
        circle_sp = generate_region_vector(desired_circle, xs, ys, x_axis_sp, y_axis_sp)
        arc_sp = generate_region_vector(desired_arc, xs*1.1, ys*1.1, x_axis_sp, y_axis_sp)

        sim_circles[seed, :] = np.tensordot((mem * ~circle_sp).v, vocab_vectors, axes=([0], [1]))
        sim_arcs[seed, :] = np.tensordot((mem * ~arc_sp).v, vocab_vectors, axes=([0], [1]))

        avg_heatmap_circle += np.tensordot((circle_sp).v, heatmap_vectors, axes=([0], [2]))
        avg_heatmap_arc += np.tensordot((arc_sp).v, heatmap_vectors, axes=([0], [2]))

    for i in range(len(vocab_labels)):
        avg_heatmaps[i] /= n_seeds

    avg_heatmap_circle /= n_seeds
    avg_heatmap_arc /= n_seeds

    animal_list = []
    for animal in vocab_labels:
        animal_list += [animal] * n_seeds

    animal_series = pd.Series(animal_list)

    sim_dfs = []

    for i in range(len(vocab_labels)):
        sim_dfs.append(
            pd.DataFrame(
                data={'Similarity': sim_locations[i].T.flatten(), 'Animal': animal_series}
            )
        )

    sim_circle_df = pd.DataFrame(
        data={'Similarity': sim_circles.T.flatten(), 'Animal': animal_series}
    )
    sim_arc_df = pd.DataFrame(
        data={'Similarity': sim_arcs.T.flatten(), 'Animal': animal_series}
    )

    # Save all data so it doesn't have to be regenerated next time
    os.makedirs(folder)

    for i in range(5):
        sim_dfs[i].to_csv(os.path.join(folder, 'sim_df_{0}.csv'.format(i)))

    sim_circle_df.to_csv(os.path.join(folder, 'sim_circle_df.csv'))
    sim_arc_df.to_csv(os.path.join(folder, 'sim_arc_df.csv'))

    np.savez(
        os.path.join(folder, 'heatmaps.npz'),
        avg_heatmaps_0=avg_heatmaps[0],
        avg_heatmaps_1=avg_heatmaps[1],
        avg_heatmaps_2=avg_heatmaps[2],
        avg_heatmaps_3=avg_heatmaps[3],
        avg_heatmaps_4=avg_heatmaps[4],
        avg_heatmap_circle=avg_heatmap_circle,
        avg_heatmap_arc=avg_heatmap_arc,
    )

else:
    # Data exists already, just load it
    data = np.load(os.path.join(folder, 'heatmaps.npz'))
    avg_heatmaps[0] = data['avg_heatmaps_0']
    avg_heatmaps[1] = data['avg_heatmaps_1']
    avg_heatmaps[2] = data['avg_heatmaps_2']
    avg_heatmaps[3] = data['avg_heatmaps_3']
    avg_heatmaps[4] = data['avg_heatmaps_4']
    avg_heatmap_circle = data['avg_heatmap_circle']
    avg_heatmap_arc = data['avg_heatmap_arc']

    sim_dfs = [None]*5
    for i in range(5):
        sim_dfs[i] = pd.read_csv(os.path.join(folder, 'sim_df_{0}.csv'.format(i)))

    sim_circle_df = pd.read_csv(os.path.join(folder, 'sim_circle_df.csv'))
    sim_arc_df = pd.read_csv(os.path.join(folder, 'sim_arc_df.csv'))


title_fontsize = 18
label_fontsize = 16

fig, ax = plt.subplots(
    2, 5,
    sharey='row',
    figsize=(16, 8)
)

for i, animal in enumerate(vocab_labels):
    img = ax[0, i].imshow(avg_heatmaps[i], cmap=cmap, vmin=vmin, vmax=vmax, extent=(xs[0], xs[-1], ys[0], ys[-1]))
    sns.barplot(x='Animal', y='Similarity', data=sim_dfs[i], ax=ax[1, i])

    ax[0, i].set_title('Where is the {0}?'.format(animal), fontsize=title_fontsize)

    ax[1, i].set_xticklabels(vocab_labels)
    ax[1, i].set_xlabel('')  # Clean the plot up a bit, it's obvious they are animals

    # Remove internal y-labels to clean up the space
    if i != 0:
        ax[1, i].set_ylabel('')


# Fixing labels for flipped axes and inverted y-axis
ax[1, 0].set_title('What is at (1.3, -1.2)?', fontsize=title_fontsize)
ax[1, 1].set_title('What is at (1.1, 1.7)?', fontsize=title_fontsize)
ax[1, 2].set_title('What is at (3.2, -4.1)?', fontsize=title_fontsize)
ax[1, 3].set_title('What is at (2.4, -2.1)?', fontsize=title_fontsize)
ax[1, 4].set_title('What is at (0.0, 0.0)?', fontsize=title_fontsize)

ax[1, 0].set_ylabel('Similarity', fontsize=label_fontsize)


fig.subplots_adjust(
    bottom=0.07,
    top=1.0,
    left=0.05,
    right=0.92,
    wspace=0.1,
    hspace=0.1
)

cb_ax = fig.add_axes([0.94, 0.6, 0.015, 0.35])
cbar = fig.colorbar(img, cax=cb_ax)

fig.savefig("figures/example_queries_{}seeds.pdf".format(n_seeds), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots(
    2, 2,
    sharey='row',
    figsize=(8, 8)
)

fig.subplots_adjust(
    bottom=0.07,
    top=0.99,
    left=0.11,
    right=0.87,
    wspace=0.1,
    hspace=0.1
)

vmin_region = -.4
vmax_region = .4

img1 = ax[0, 0].imshow(avg_heatmap_circle, cmap=cmap, vmin=vmin_region, vmax=vmax_region, extent=(xs[0], xs[-1], ys[0], ys[-1]))
img2 = ax[0, 1].imshow(avg_heatmap_arc, cmap=cmap, vmin=vmin_region, vmax=vmax_region, extent=(xs[0], xs[-1], ys[0], ys[-1]))
sns.barplot(x='Animal', y='Similarity', data=sim_circle_df, ax=ax[1, 0])
sns.barplot(x='Animal', y='Similarity', data=sim_arc_df, ax=ax[1, 1])

ax[0, 0].set_title('Circular Region Similarity Map', fontsize=title_fontsize)
ax[0, 1].set_title('Rectangular Region Similarity Map', fontsize=title_fontsize)

ax[1, 0].set_title('What is within the circular region?', fontsize=title_fontsize)
ax[1, 1].set_title('What is in the bottom right quadrant?', fontsize=title_fontsize)

ax[0, 0].set_title('Circular Region', fontsize=title_fontsize)
ax[0, 1].set_title('Rectangular Region', fontsize=title_fontsize)

ax[1, 0].set_title('', fontsize=title_fontsize)
ax[1, 1].set_title('', fontsize=title_fontsize)

# Remove the internal y-label
ax[1, 1].set_ylabel('')

ax[1, 0].set_ylabel('Similarity', fontsize=label_fontsize)

ax[1, 0].set_xticklabels(vocab_labels)
ax[1, 0].set_xlabel('')  # Clean the plot up a bit, it's obvious they are animals
ax[1, 1].set_xticklabels(vocab_labels)
ax[1, 1].set_xlabel('')  # Clean the plot up a bit, it's obvious they are animals

cb_region_ax2 = fig.add_axes([0.91, 0.585, 0.025, 0.375])
cbar_region2 = fig.colorbar(img2, cax=cb_region_ax2)

fig.savefig("figures/example_region_queries_{}seeds.pdf".format(n_seeds), dpi=600, bbox_inches='tight')

plt.show()
