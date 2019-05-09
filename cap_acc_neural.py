import nengo
import nengo.spa as spa
import numpy as np
import pandas as pd
from utils import make_good_unitary, MemoryDataset, get_heatmap_vectors, loc_match, item_match
import os

import argparse

parser = argparse.ArgumentParser('Capacity and Accuracy experiment using neurons')

parser.add_argument('--n-samples', type=int, default=100,
                    help='Number of samples to test with')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
parser.add_argument('--neurons-per-dim', type=int, default=15)
parser.add_argument('--limit', type=int, default=5, help='The absolute min and max of the space')
# NOTE: this is set low so it returns the best match unless the result is really bad
parser.add_argument('--similarity-threshold', type=float, default=0.01)
parser.add_argument('--large-item-range', action='store_true', help='if set use a lot more item amounts')
parser.add_argument('--query-type', type=str, default='Both', choices=['Location', 'Item', 'Both'])
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

if args.large_item_range:
    item_amounts = [2**n for n in range(12)]
else:
    item_amounts = [1, 2, 4, 8, 16, 32]

df = pd.DataFrame()

D = args.dim
seed = args.seed
n_cconv_neurons = args.neurons_per_dim * 2
limit = args.limit

dt = 0.001
time_per_sample = 100

if not os.path.exists('output/neural_capacity'):
    os.makedirs('output/neural_capacity')

fname = 'output/neural_capacity/query_experiment_npd{0}_{1}D_{2}samples'.format(
    args.neurons_per_dim, D, args.n_samples
)

# Sample period in seconds
sample_period = time_per_sample * dt

rstate = np.random.RandomState(seed=args.seed)
x_axis_sp = make_good_unitary(args.dim, rng=rstate)
y_axis_sp = make_good_unitary(args.dim, rng=rstate)

xs = np.linspace(-args.limit, args.limit, 128)
ys = np.linspace(-args.limit, args.limit, 128)

heatmap_vectors = get_heatmap_vectors(xs=xs, ys=ys, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

vocab = spa.Vocabulary(args.dim)

if args.large_item_range:
    n_vocab_vectors = item_amounts[-1]
else:
    # vocab size of twice the maximum number of items used
    n_vocab_vectors = item_amounts[-1] * 2

vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

print("Generating {0} vocab items".format(n_vocab_vectors))
for i in range(n_vocab_vectors):
    p = vocab.create_pointer()
    vocab_vectors[i, :] = p.v
print("Vocab generation complete")

# A copy that will get shuffled around in MemoryDataset
vocab_vectors_copy = vocab_vectors.copy()


class Experiment(object):

    def __init__(self, dim, n_items, limit, x_axis_sp, y_axis_sp, dt=0.001, time_per_sample=100):

        self.dt = dt
        self.time_per_sample = time_per_sample

        self.dataset = MemoryDataset(
            dim=dim,
            n_items=n_items,
            allow_duplicate_items=False,
            limits=(-limit, limit, -limit, limit),
            normalize_memory=True,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
        )
        self.data_gen = self.dataset.sample_generator(return_coord_sp=True, item_set=vocab_vectors_copy)

        self.mem, self.item, self.coord_sp, self.coord = self.data_gen.__next__()

        self.step = 0

    def __call__(self, t):

        # Convert from seconds to number of timesteps, and then scale by time per sample
        i = int((t / self.dt) / self.time_per_sample)

        # Change the sample to evaluate if required
        if i > self.step:
            self.step += 1
            self.mem, self.item, self.coord_sp, self.coord = self.data_gen.__next__()

        return np.concatenate([self.mem, self.item, self.coord_sp, self.coord])


for n_items in item_amounts:
    print("Running experiment for {} items in memory".format(n_items))

    exp = Experiment(
        dim=D,
        n_items=n_items,
        limit=limit,
        x_axis_sp=x_axis_sp,
        y_axis_sp=y_axis_sp,
        dt=dt,
        time_per_sample=time_per_sample,  # Timesteps per sample
    )

    model = nengo.Network('Nengo Capacity Experiment')
    with model:
        exp_node = nengo.Node(exp, size_in=0, size_out=3 * D + 2)

        memory = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D)
        true_coord_sp = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
        item_output = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
        coord_sp_output = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

        true_item = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
        true_coord = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

        # Item Query
        cconv_item_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
        nengo.Connection(memory, cconv_item_query.input_a)
        nengo.Connection(true_item, cconv_item_query.input_b)
        nengo.Connection(cconv_item_query.output, coord_sp_output)

        # Location Query
        cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
        nengo.Connection(memory, cconv_loc_query.input_a)
        nengo.Connection(true_coord_sp, cconv_loc_query.input_b)
        nengo.Connection(cconv_loc_query.output, item_output)

        # Connect up the inputs
        nengo.Connection(exp_node[0:D], memory, synapse=None)
        nengo.Connection(exp_node[D * 2:D * 3], true_coord_sp)
        nengo.Connection(exp_node[D:2 * D], true_item)
        nengo.Connection(exp_node[D * 3:], true_coord)

        # Sample with the probes at the middle and end points of each transition.
        # Only the middle value will be saved, to reduce memory requirements
        p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
        p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
        p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
        p_truth_coord = nengo.Probe(true_coord, synapse=None, sample_every=sample_period / 2.)
        p_truth_coord_sp = nengo.Probe(true_coord_sp, synapse=None, sample_every=sample_period / 2.)

    sim = nengo.Simulator(model, dt=dt)

    run_time = args.n_samples * time_per_sample * dt

    print("Running for {0} simulated seconds".format(run_time))

    sim.run(run_time)

    print("Simulation Complete.")

    pred_item = sim.data[p_pred_item]
    truth_item = sim.data[p_truth_item]
    pred_coord_sp = sim.data[p_pred_coord_sp]
    truth_coord = sim.data[p_truth_coord]
    truth_coord_sp = sim.data[p_truth_coord_sp]

    # Only keep the center timesteps from the period
    trimmed_pred_item = pred_item[1::2]
    trimmed_truth_item = truth_item[1::2]
    trimmed_pred_coord_sp = pred_coord_sp[1::2]
    trimmed_truth_coord = truth_coord[1::2]
    trimmed_truth_coord_sp = truth_coord_sp[1::2]

    # Add datapoints to the pandas dataframe
    for i in range(trimmed_pred_item.shape[0]):

        if args.query_type == 'Location' or args.query_type == 'Both':
            acc = item_match(
                sp=trimmed_pred_item[i, :],
                vocab_vectors=vocab_vectors,
                item=trimmed_truth_item[i, :],
                sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Similarity": np.dot(trimmed_pred_item[i, :], trimmed_truth_item[i, :]),
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Correct Item": True,
                    "Query Type": 'Location',
                    "Limit": limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": time_per_sample,
                },
                ignore_index=True
            )

        if args.query_type == 'Item' or args.query_type == 'Both':
            acc = loc_match(
                sp=trimmed_pred_coord_sp[i, :],
                heatmap_vectors=heatmap_vectors,
                coord=trimmed_truth_coord[i, :],
                xs=xs,
                ys=ys,
                distance_threshold=0.5,
                sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Similarity": np.dot(trimmed_pred_coord_sp[i, :], trimmed_truth_coord_sp[i, :]),
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Correct Item": True,
                    "Query Type": 'Item',
                    "Limit": limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": time_per_sample,
                },
                ignore_index=True
            )

df.to_csv(fname + '.csv')
