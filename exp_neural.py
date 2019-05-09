# Measuring the performance of various capabilities of spatial semantic pointers
# such as querying objects, querying locations, moving objects, etc
# using a spiking neural network
import numpy as np
import nengo
import nengo.spa as spa
from utils import item_match_neural, loc_match, loc_match_duplicate, region_item_match, \
    encode_point, get_heatmap_vectors, MemoryDataset, make_good_unitary
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser('Measuring the performance of various capabilities of spatial semantic pointers')

parser.add_argument('--n-samples', type=int, default=100, help='Number of samples to evaluate')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
parser.add_argument('--neurons-per-dim', type=int, default=50)
parser.add_argument('--limit', type=int, default=5, help='The absolute min and max of the space')
parser.add_argument('--res', type=int, default=128, help='Resolution for the linspace')
parser.add_argument('--n-items-min', type=int, default=2, help='Lowest number of items in a memory')
parser.add_argument('--n-items-max', type=int, default=24, help='Highest number of items in a memory')
# parser.add_argument('--n-items', type=int, default=8)
# parser.add_argument('--similarity-threshold', type=float, default=0.1, help='Similarity must be above this value to count')
parser.add_argument('--similarity-threshold', type=float, default=0.01,
                    help='Similarity must be above this value to count')
parser.add_argument('--experiment', type=str, default='missing_object',
                    choices=['single_object', 'missing_object', 'duplicate_object', 'location',
                             'sliding_object', 'sliding_group', 'region'])
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--time-per-sample', type=int, default=500)
parser.add_argument('--synapse', type=float, default=0.05)
parser.add_argument('--direct-mode', action='store_true', help='direct mode for debugging')
parser.add_argument('--folder', default='output/neural_results', help='folder to save results')
parser.add_argument('--experiments', type=str, default='all', choices=['all', 'region', 'non-region', 'sliding'])

args = parser.parse_args()

if args.synapse == 0:
    synapse = None
else:
    synapse = args.synapse

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

rstate = np.random.RandomState(seed=args.seed)
x_axis_sp = make_good_unitary(args.dim, rng=rstate)
y_axis_sp = make_good_unitary(args.dim, rng=rstate)

heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

# These are for dealing with shifted memories, that could potentially go outside the normal range
larger_heatmap_vectors = get_heatmap_vectors(xs * 2, ys * 2, x_axis_sp, y_axis_sp)

dt = 0.001

D = args.dim

n_cconv_neurons = args.neurons_per_dim * 2

# Sample period in seconds
sample_period = args.time_per_sample * dt

# memory_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
# memory_sizes = [2, 6, 12, 24]
memory_sizes = list(range(args.n_items_min, args.n_items_max + 1))

if not os.path.exists(args.folder):
    os.makedirs(args.folder)

# fname = 'output/neural_capabilities/neural_{}_npd{}_tps{}_seed{}_dim{}'.format(
#     args.experiment, args.neurons_per_dim, args.time_per_sample, args.seed, args.dim
# )

fname = '{}/seed{}_dim{}_min{}_max{}_th{}'.format(
    args.folder, args.seed, args.dim, args.n_items_min, args.n_items_max, args.similarity_threshold
)

if args.direct_mode:
    fname += '_direct'

# modify name if only some experiments are run
if args.experiments == 'region':
    fname += '_region_only'
    experiment_list = [
        'region'
    ]
elif args.experiments == 'non-region':
    fname += '_nonregion_only'
    experiment_list = [
        'single_object',
        'missing_object',
        'duplicate_object',
        'location',
        'sliding_object',
        'sliding_group',
    ]
elif args.experiments == 'sliding':
    fname += '_nonregion_only'
    experiment_list = [
        'sliding_object',
        'sliding_group',
    ]
else:
    experiment_list = [
        'single_object',
        'missing_object',
        'duplicate_object',
        'location',
        'sliding_object',
        'sliding_group',
        'region'
    ]


def main(experiment_list):

    df = pd.DataFrame()

    for exp_name in experiment_list:

        print("Running Experiment: {}".format(exp_name))

        if exp_name == 'single_object':
            exp_df = single_object()
        elif exp_name == 'missing_object':
            exp_df = missing_object()
        elif exp_name == 'duplicate_object':
            exp_df = duplicate_object()
        elif exp_name == 'location':
            exp_df = location()
        elif exp_name == 'sliding_object':
            exp_df = sliding_object()
        elif exp_name == 'sliding_group':
            exp_df = sliding_group()
        elif exp_name == 'region':
            exp_df = region()
        else:
            raise NotImplementedError

        df = df.append(exp_df)

    df.to_csv(fname + '.csv')


def single_object():

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
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, self.item, self.coord_sp, self.coord = self.data_gen.__next__()

            return np.concatenate([self.mem, self.item, self.coord_sp, self.coord])

    df = pd.DataFrame()

    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
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

            # # Location Query
            # cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            # nengo.Connection(memory, cconv_loc_query.input_a)
            # nengo.Connection(true_coord_sp, cconv_loc_query.input_b)
            # nengo.Connection(cconv_loc_query.output, item_output)

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            nengo.Connection(exp_node[D * 2:D * 3], true_coord_sp)
            nengo.Connection(exp_node[D:2 * D], true_item)
            nengo.Connection(exp_node[D * 3:], true_coord)


            # Sample with the probes at the middle and end points of each transition.
            # Only the middle value will be saved, to reduce memory requirements
            p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
            p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
            p_truth_coord = nengo.Probe(true_coord, synapse=None, sample_every=sample_period / 2.)
            # p_truth_coord_sp = nengo.Probe(true_coord_sp, synapse=None, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        pred_item = sim.data[p_pred_item]
        # truth_item = sim.data[p_truth_item]
        pred_coord_sp = sim.data[p_pred_coord_sp]
        truth_coord = sim.data[p_truth_coord]
        # truth_coord_sp = sim.data[p_truth_coord_sp]

        # Only keep the center timesteps from the period
        # trimmed_pred = pred[0::2]
        # trimmed_truth = truth[0::2]
        # Looks like this way keeps it centered properly
        trimmed_pred_item = pred_item[1::2]
        # trimmed_truth_item = truth_item[1::2]
        trimmed_pred_coord_sp = pred_coord_sp[1::2]
        trimmed_truth_coord = truth_coord[1::2]
        # trimmed_truth_coord_sp = truth_coord_sp[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_item.shape[0]):

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
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Single Object',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


def missing_object():

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
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, self.item, self.coord_sp, self.coord = self.data_gen.__next__()

                # replace item with a random pointer, it should not be found in the memory
                self.item = spa.SemanticPointer(args.dim).v

            return np.concatenate([self.mem, self.item, self.coord_sp, self.coord])

    df = pd.DataFrame()

    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
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

            # # Location Query
            # cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            # nengo.Connection(memory, cconv_loc_query.input_a)
            # nengo.Connection(true_coord_sp, cconv_loc_query.input_b)
            # nengo.Connection(cconv_loc_query.output, item_output)

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            nengo.Connection(exp_node[D * 2:D * 3], true_coord_sp)
            nengo.Connection(exp_node[D:2 * D], true_item)
            nengo.Connection(exp_node[D * 3:], true_coord)


            # Sample with the probes at the middle and end points of each transition.
            # Only the middle value will be saved, to reduce memory requirements
            p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
            p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
            p_truth_coord = nengo.Probe(true_coord, synapse=None, sample_every=sample_period / 2.)
            # p_truth_coord_sp = nengo.Probe(true_coord_sp, synapse=None, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        pred_item = sim.data[p_pred_item]
        # truth_item = sim.data[p_truth_item]
        pred_coord_sp = sim.data[p_pred_coord_sp]
        truth_coord = sim.data[p_truth_coord]
        # truth_coord_sp = sim.data[p_truth_coord_sp]

        # Only keep the center timesteps from the period
        # trimmed_pred = pred[0::2]
        # trimmed_truth = truth[0::2]
        # Looks like this way keeps it centered properly
        trimmed_pred_item = pred_item[1::2]
        # trimmed_truth_item = truth_item[1::2]
        trimmed_pred_coord_sp = pred_coord_sp[1::2]
        trimmed_truth_coord = truth_coord[1::2]
        # trimmed_truth_coord_sp = truth_coord_sp[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_item.shape[0]):

            acc = 1 - loc_match(
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
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Missing Object',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


def duplicate_object():

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

            self.data_gen = self.dataset.duplicates_sample_generator(
                item_set=vocab_vectors_copy,
                n_items_min=max(2, n_items),
                n_items_max=max(2, n_items),
            )

            self.mem, self.item, self.coord1, self.coord2 = self.data_gen.__next__()

            self.step = 0

        def __call__(self, t):
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, self.item, self.coord1, self.coord2 = self.data_gen.__next__()

            return np.concatenate([self.mem, self.item, self.coord1, self.coord2])

    df = pd.DataFrame()
    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
        with model:
            exp_node = nengo.Node(exp, size_in=0, size_out=2 * D + 4)

            memory = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D)
            # true_coord_sp = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            # item_output = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            coord_sp_output = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

            true_item = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            true_coord1 = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())
            true_coord2 = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

            # Item Query
            cconv_item_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            nengo.Connection(memory, cconv_item_query.input_a)
            nengo.Connection(true_item, cconv_item_query.input_b)
            nengo.Connection(cconv_item_query.output, coord_sp_output)

            # # Location Query
            # cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            # nengo.Connection(memory, cconv_loc_query.input_a)
            # nengo.Connection(true_coord_sp, cconv_loc_query.input_b)
            # nengo.Connection(cconv_loc_query.output, item_output)

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            # nengo.Connection(exp_node[D * 2:D * 3], true_coord_sp)
            nengo.Connection(exp_node[D:2 * D], true_item)
            nengo.Connection(exp_node[D * 2:D * 2 + 2], true_coord1)
            nengo.Connection(exp_node[D * 2 + 2:D * 2 + 4], true_coord2)


            # Sample with the probes at the middle and end points of each transition.
            # Only the middle value will be saved, to reduce memory requirements
            # p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
            p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
            p_truth_coord1 = nengo.Probe(true_coord1, synapse=None, sample_every=sample_period / 2.)
            p_truth_coord2 = nengo.Probe(true_coord2, synapse=None, sample_every=sample_period / 2.)
            # p_truth_coord_sp = nengo.Probe(true_coord_sp, synapse=None, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        # pred_item = sim.data[p_pred_item]
        # truth_item = sim.data[p_truth_item]
        pred_coord_sp = sim.data[p_pred_coord_sp]
        truth_coord1 = sim.data[p_truth_coord1]
        truth_coord2 = sim.data[p_truth_coord2]
        # truth_coord_sp = sim.data[p_truth_coord_sp]

        # Only keep the center timesteps from the period
        # trimmed_pred = pred[0::2]
        # trimmed_truth = truth[0::2]
        # Looks like this way keeps it centered properly
        # trimmed_pred_item = pred_item[1::2]
        # trimmed_truth_item = truth_item[1::2]
        trimmed_pred_coord_sp = pred_coord_sp[1::2]
        trimmed_truth_coord1 = truth_coord1[1::2]
        trimmed_truth_coord2 = truth_coord2[1::2]
        # trimmed_truth_coord_sp = truth_coord_sp[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_coord_sp.shape[0]):

            acc = loc_match_duplicate(
                trimmed_pred_coord_sp[i, :], heatmap_vectors,
                coord1=trimmed_truth_coord1[i, :], coord2=trimmed_truth_coord2[i, :],
                xs=xs, ys=ys, sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Duplicate Object',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


def location():

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
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, self.item, self.coord_sp, self.coord = self.data_gen.__next__()

            return np.concatenate([self.mem, self.item, self.coord_sp, self.coord])

    df = pd.DataFrame()

    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
        with model:
            exp_node = nengo.Node(exp, size_in=0, size_out=3 * D + 2)

            memory = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D)
            true_coord_sp = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            item_output = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            # coord_sp_output = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

            true_item = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            # true_coord = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

            # Item Query
            # cconv_item_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            # nengo.Connection(memory, cconv_item_query.input_a)
            # nengo.Connection(true_item, cconv_item_query.input_b)
            # nengo.Connection(cconv_item_query.output, coord_sp_output)

            # # Location Query
            cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            nengo.Connection(memory, cconv_loc_query.input_a)
            nengo.Connection(true_coord_sp, cconv_loc_query.input_b)
            nengo.Connection(cconv_loc_query.output, item_output)

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            nengo.Connection(exp_node[D * 2:D * 3], true_coord_sp)
            nengo.Connection(exp_node[D:2 * D], true_item)
            # nengo.Connection(exp_node[D * 3:], true_coord)


            # Sample with the probes at the middle and end points of each transition.
            # Only the middle value will be saved, to reduce memory requirements
            p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
            p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
            # p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_coord = nengo.Probe(true_coord, synapse=None, sample_every=sample_period / 2.)
            # p_truth_coord_sp = nengo.Probe(true_coord_sp, synapse=None, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        pred_item = sim.data[p_pred_item]
        truth_item = sim.data[p_truth_item]
        # pred_coord_sp = sim.data[p_pred_coord_sp]
        # truth_coord = sim.data[p_truth_coord]
        # truth_coord_sp = sim.data[p_truth_coord_sp]

        # Only keep the center timesteps from the period
        # trimmed_pred = pred[0::2]
        # trimmed_truth = truth[0::2]
        # Looks like this way keeps it centered properly
        trimmed_pred_item = pred_item[1::2]
        trimmed_truth_item = truth_item[1::2]
        # trimmed_pred_coord_sp = pred_coord_sp[1::2]
        # trimmed_truth_coord = truth_coord[1::2]
        # trimmed_truth_coord_sp = truth_coord_sp[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_item.shape[0]):

            # acc = item_match(
            acc = item_match_neural(
                sp=trimmed_pred_item[i, :],
                vocab_vectors=vocab_vectors,
                item=trimmed_truth_item[i, :],
                sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Location',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


def region():

    class Experiment(object):

        def __init__(self, dim, n_items, limit, x_axis_sp, y_axis_sp, vocab_vectors, dt=0.001, time_per_sample=100):
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
            self.data_gen_region = self.dataset.region_sample_generator(
                vocab_vectors=vocab_vectors,
                xs=xs,
                ys=ys,
                n_items_min=n_items,
                n_items_max=n_items,
                rad_min=1,
                rad_max=3
            )

            self.mem, self.items, self.coords, self.region, self.vocab_indices = self.data_gen_region.__next__()

            # Convert to a fixed length one-hot vector
            self.vocab_vec = np.zeros((vocab_vectors.shape[0],))
            for i in self.vocab_indices:
                self.vocab_vec[i] = 1

            self.step = 0

        def __call__(self, t):
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, self.items, self.coords, self.region, self.vocab_indices = self.data_gen_region.__next__()

                # Set up the one hot vector that can be probed
                self.vocab_vec[:] = 0
                for i in self.vocab_indices:
                    self.vocab_vec[i] = 1

            return np.concatenate([self.mem, self.region, self.vocab_vec])

    df = pd.DataFrame()

    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            vocab_vectors=vocab_vectors,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
        with model:
            exp_node = nengo.Node(exp, size_in=0, size_out=2 * D + vocab_vectors.shape[0])

            memory = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D)
            true_region_sp = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            item_output = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            # coord_sp_output = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

            vocab_indices = nengo.Ensemble(n_neurons=1, dimensions=vocab_vectors.shape[0], neuron_type=nengo.Direct())

            # true_item = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            # true_coord = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

            # Item Query
            # cconv_item_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            # nengo.Connection(memory, cconv_item_query.input_a)
            # nengo.Connection(true_item, cconv_item_query.input_b)
            # nengo.Connection(cconv_item_query.output, coord_sp_output)

            # # Location Query
            cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            nengo.Connection(memory, cconv_loc_query.input_a)
            nengo.Connection(true_region_sp, cconv_loc_query.input_b)
            nengo.Connection(cconv_loc_query.output, item_output)

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            nengo.Connection(exp_node[D:D * 2], true_region_sp)
            nengo.Connection(exp_node[D*2:], vocab_indices)
            # nengo.Connection(exp_node[D:2 * D], true_item)
            # nengo.Connection(exp_node[D * 3:], true_coord)


            # Sample with the probes at the middle and end points of each transition.
            # Only the middle value will be saved, to reduce memory requirements
            p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
            # p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_coord = nengo.Probe(true_coord, synapse=None, sample_every=sample_period / 2.)
            # p_truth_region_sp = nengo.Probe(true_region_sp, synapse=None, sample_every=sample_period / 2.)
            p_truth_vocab_indices = nengo.Probe(vocab_indices, synapse=None, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        pred_item = sim.data[p_pred_item]
        # truth_item = sim.data[p_truth_item]
        # pred_coord_sp = sim.data[p_pred_coord_sp]
        # truth_coord = sim.data[p_truth_coord]
        # truth_region_sp = sim.data[p_truth_region_sp]
        truth_vocab_indices = sim.data[p_truth_vocab_indices]

        # Only keep the center timesteps from the period
        # trimmed_pred = pred[0::2]
        # trimmed_truth = truth[0::2]
        # Looks like this way keeps it centered properly
        trimmed_pred_item = pred_item[1::2]
        # trimmed_truth_item = truth_item[1::2]
        # trimmed_pred_coord_sp = pred_coord_sp[1::2]
        # trimmed_truth_coord = truth_coord[1::2]
        # trimmed_truth_region_sp = truth_region_sp[1::2]
        trimmed_truth_vocab_indices = truth_vocab_indices[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_item.shape[0]):

            # Convert one-hot encoding back to a list for the function
            vocab_indices_list = []
            for j in range(vocab_vectors.shape[0]):
                if trimmed_truth_vocab_indices[i, j] > 0.5:
                    vocab_indices_list.append(j)

            acc = region_item_match(
                sp=trimmed_pred_item[i, :],
                vocab_vectors=vocab_vectors,
                vocab_indices=vocab_indices_list,
                sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Region',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


def sliding_group():

    class Experiment(object):

        def __init__(self, dim, n_items, limit, x_axis_sp, y_axis_sp, dt=0.001, time_per_sample=100):
            self.dt = dt
            self.time_per_sample = time_per_sample

            self.x_axis_sp = x_axis_sp
            self.y_axis_sp = y_axis_sp

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

            dx = np.random.uniform(-args.limit / 2., args.limit / 2.)
            dy = np.random.uniform(-args.limit / 2., args.limit / 2.)
            self.slide_sp = encode_point(dx, dy, x_axis_sp, y_axis_sp)

            self.coord[0] += dx
            self.coord[1] += dy

            self.step = 0

        def __call__(self, t):
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, self.item, self.coord_sp, self.coord = self.data_gen.__next__()

                dx = np.random.uniform(-args.limit / 2., args.limit / 2.)
                dy = np.random.uniform(-args.limit / 2., args.limit / 2.)
                self.slide_sp = encode_point(dx, dy, x_axis_sp, y_axis_sp)

                self.coord[0] += dx
                self.coord[1] += dy

            return np.concatenate([self.mem, self.item, self.slide_sp.v, self.coord])

    df = pd.DataFrame()

    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
        with model:
            exp_node = nengo.Node(exp, size_in=0, size_out=3 * D + 2)

            memory = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D)
            # memory_slid = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D)
            true_slide_sp = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            item_output = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            coord_sp_output = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

            true_item = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            true_coord = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

            # Sliding
            cconv_slide = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D)
            nengo.Connection(memory, cconv_slide.input_a)
            nengo.Connection(true_slide_sp, cconv_slide.input_b)


            # Item Query
            cconv_item_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            nengo.Connection(cconv_slide.output, cconv_item_query.input_a)  # Note: directly connecting, no intermediate
            # nengo.Connection(memory_slid, cconv_item_query.input_a)
            nengo.Connection(true_item, cconv_item_query.input_b)
            nengo.Connection(cconv_item_query.output, coord_sp_output)

            # # Location Query
            # cconv_loc_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            # nengo.Connection(memory, cconv_loc_query.input_a)
            # nengo.Connection(true_coord_sp, cconv_loc_query.input_b)
            # nengo.Connection(cconv_loc_query.output, item_output)

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            nengo.Connection(exp_node[D * 2:D * 3], true_slide_sp)
            nengo.Connection(exp_node[D:2 * D], true_item)
            nengo.Connection(exp_node[D * 3:], true_coord)


            # Sample with the probes at the middle and end points of each transition.
            # Only the middle value will be saved, to reduce memory requirements
            p_pred_item = nengo.Probe(item_output, synapse=None, sample_every=sample_period / 2.)
            # p_truth_item = nengo.Probe(true_item, synapse=None, sample_every=sample_period / 2.)
            p_pred_coord_sp = nengo.Probe(coord_sp_output, synapse=None, sample_every=sample_period / 2.)
            p_truth_coord = nengo.Probe(true_coord, synapse=None, sample_every=sample_period / 2.)
            # p_truth_slide_sp = nengo.Probe(true_slide_sp, synapse=None, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        pred_item = sim.data[p_pred_item]
        # truth_item = sim.data[p_truth_item]
        pred_coord_sp = sim.data[p_pred_coord_sp]
        truth_coord = sim.data[p_truth_coord]
        # truth_slide_sp = sim.data[p_truth_slide_sp]

        # Only keep the center timesteps from the period
        # trimmed_pred = pred[0::2]
        # trimmed_truth = truth[0::2]
        # Looks like this way keeps it centered properly
        trimmed_pred_item = pred_item[1::2]
        # trimmed_truth_item = truth_item[1::2]
        trimmed_pred_coord_sp = pred_coord_sp[1::2]
        trimmed_truth_coord = truth_coord[1::2]
        # trimmed_truth_slide_sp = truth_slide_sp[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_item.shape[0]):

            acc = loc_match(
                sp=trimmed_pred_coord_sp[i, :],
                heatmap_vectors=larger_heatmap_vectors,
                coord=trimmed_truth_coord[i, :],
                xs=xs * 2,
                ys=ys * 2,
                distance_threshold=0.5,
                sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Accuracy": acc,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Sliding Group',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


def sliding_object():
    class Experiment(object):

        def __init__(self, dim, n_items, limit, x_axis_sp, y_axis_sp, dt=0.001, time_per_sample=100):
            self.dt = dt
            self.time_per_sample = time_per_sample

            self.x_axis_sp = x_axis_sp
            self.y_axis_sp = y_axis_sp

            self.dataset = MemoryDataset(
                dim=dim,
                n_items=n_items,
                allow_duplicate_items=False,
                limits=(-limit, limit, -limit, limit),
                normalize_memory=True,
                x_axis_sp=x_axis_sp,
                y_axis_sp=y_axis_sp,
            )
            self.data_gen_multi = self.dataset.multi_return_sample_generator(
                return_coord_sp=False,
                item_set=vocab_vectors_copy,
                n_items=n_items,
                allow_duplicate_items=False,
            )

            self.mem, items, coords = self.data_gen_multi.__next__()

            self.item_move = items[0, :]
            self.item_stay = items[1, :]
            self.coord_move = coords[0, :]
            self.coord_stay = coords[1, :]
            self.coord_sp_original = encode_point(self.coord_move[0], self.coord_move[1], x_axis_sp, y_axis_sp)

            dx = np.random.uniform(-args.limit / 2., args.limit / 2.)
            dy = np.random.uniform(-args.limit / 2., args.limit / 2.)
            self.slide_sp = encode_point(dx, dy, x_axis_sp, y_axis_sp)

            # used for doing the add and subtract in a single convolution
            self.swap_v = self.slide_sp.v - encode_point(0, 0, x_axis_sp, y_axis_sp).v
            # swap_v[0] -= 1

            self.coord_move[0] += dx
            self.coord_move[1] += dy

            self.coord_sp_moved = encode_point(self.coord_move[0], self.coord_move[1], x_axis_sp, y_axis_sp)

            self.step = 0

        def __call__(self, t):
            # i = int(t * 10) % self.n_items

            # Convert from seconds to number of timesteps, and then scale by time per sample
            i = int((t / self.dt) / self.time_per_sample)

            # Change the sample to evaluate if required
            if i > self.step:
                self.step += 1
                self.mem, items, coords = self.data_gen_multi.__next__()

                self.item_move = items[0, :]
                self.item_stay = items[1, :]
                self.coord_move = coords[0, :]
                self.coord_stay = coords[1, :]
                self.coord_sp_original = encode_point(self.coord_move[0], self.coord_move[1], x_axis_sp, y_axis_sp)

                dx = np.random.uniform(-args.limit / 2., args.limit / 2.)
                dy = np.random.uniform(-args.limit / 2., args.limit / 2.)
                self.slide_sp = encode_point(dx, dy, x_axis_sp, y_axis_sp)

                # used for doing the add and subtract in a single convolution
                self.swap_v = self.slide_sp.v - encode_point(0, 0, x_axis_sp, y_axis_sp).v
                # swap_v[0] -= 1

                self.coord_move[0] += dx
                self.coord_move[1] += dy

                self.coord_sp_moved = encode_point(self.coord_move[0], self.coord_move[1], x_axis_sp, y_axis_sp)

            return np.concatenate([self.mem, self.item_move, self.item_stay, self.coord_sp_original.v,
                                   self.coord_sp_moved.v, self.swap_v, self.coord_move, self.coord_stay])

    df = pd.DataFrame()

    for n_items in memory_sizes:

        vocab = spa.Vocabulary(args.dim)

        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        exp = Experiment(
            dim=D,
            n_items=n_items,
            limit=args.limit,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            dt=dt,
            time_per_sample=args.time_per_sample,  # Timesteps per sample
        )

        model = nengo.Network()
        if args.direct_mode:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()
        with model:
            exp_node = nengo.Node(exp, size_in=0, size_out=6 * D + 4)

            # Two items are kept track of, the one that is moved and one of the ones that stays
            # many others also stay, but only doing two queries total per sample in this experiment
            memory = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D)
            true_item_moved = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            true_item_stay = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            true_coord_sp_original = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            # true_coord_sp_moved = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            true_swap_sp = nengo.Ensemble(n_neurons=1, dimensions=D, neuron_type=nengo.Direct())
            true_coord_moved = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())
            true_coord_stay = nengo.Ensemble(n_neurons=1, dimensions=2, neuron_type=nengo.Direct())

            # Connect up the inputs
            nengo.Connection(exp_node[0:D], memory, synapse=None)
            nengo.Connection(exp_node[D * 1:D * 2], true_item_moved, synapse=None)
            nengo.Connection(exp_node[D * 2:D * 3], true_item_stay, synapse=None)
            nengo.Connection(exp_node[D * 3:D * 4], true_coord_sp_original, synapse=None)
            # nengo.Connection(exp_node[D * 4:D * 5], true_coord_sp_moved, synapse=None)
            nengo.Connection(exp_node[D * 5:D * 6], true_swap_sp, synapse=None)
            nengo.Connection(exp_node[D * 6:D * 6 + 2], true_coord_moved, synapse=None)
            nengo.Connection(exp_node[D * 6 + 2: D * 6 + 4], true_coord_stay, synapse=None)

            # This is the memory where one item has been moved, queries will be done from here
            memory_slid = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D)
            # The value that is to be added to the original memory to produce a memory with one item slid
            # sum_term = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D)  #note, using cconv output directly
            # true_slide_sp = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            # item_output = nengo.Ensemble(n_neurons=D*args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

            # The coordinate predictions for the moved and stay item
            coord_sp_output_moved = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())
            coord_sp_output_stay = nengo.Ensemble(n_neurons=D * args.neurons_per_dim, dimensions=D, neuron_type=nengo.Direct())

            # Calculate the sum term, it is: item_moved*pos_sp_original*swap_sp

            cconv_pos = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D)
            cconv_swap = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D)

            nengo.Connection(true_item_moved, cconv_pos.input_a, synapse=None)
            nengo.Connection(true_coord_sp_original, cconv_pos.input_b, synapse=None)

            nengo.Connection(cconv_pos.output, cconv_swap.input_a, synapse=synapse)
            nengo.Connection(true_swap_sp, cconv_swap.input_b, synapse=None)

            # nengo.Connect(cconv_swap.output, sum_term)

            # Perform the summation of the original memory and the sum term into memory_slid
            nengo.Connection(memory, memory_slid, synapse=synapse)
            nengo.Connection(cconv_swap.output, memory_slid, synapse=synapse)

            # Item Query
            cconv_item_moved_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            nengo.Connection(memory_slid, cconv_item_moved_query.input_a, synapse=synapse)
            nengo.Connection(true_item_moved, cconv_item_moved_query.input_b, synapse=synapse)
            nengo.Connection(cconv_item_moved_query.output, coord_sp_output_moved, synapse=synapse)

            cconv_item_stay_query = nengo.networks.CircularConvolution(n_cconv_neurons, dimensions=D, invert_b=True)
            nengo.Connection(memory_slid, cconv_item_stay_query.input_a, synapse=synapse)
            nengo.Connection(true_item_stay, cconv_item_stay_query.input_b, synapse=synapse)
            nengo.Connection(cconv_item_stay_query.output, coord_sp_output_stay, synapse=synapse)

            # Sample with the probes at the middle and end points of each transition.
            p_pred_coord_sp_moved = nengo.Probe(coord_sp_output_moved, synapse=synapse, sample_every=sample_period / 2.)
            p_pred_coord_sp_stay = nengo.Probe(coord_sp_output_stay, synapse=synapse, sample_every=sample_period / 2.)
            p_truth_coord_moved = nengo.Probe(true_coord_moved, synapse=synapse, sample_every=sample_period / 2.)
            p_truth_coord_stay = nengo.Probe(true_coord_stay, synapse=synapse, sample_every=sample_period / 2.)

        sim = nengo.Simulator(model, dt=dt)

        run_time = args.n_samples * args.time_per_sample * dt

        print("Running for {0} simulated seconds".format(run_time))

        sim.run(run_time)

        print("Simulation Complete.")

        pred_coord_sp_moved = sim.data[p_pred_coord_sp_moved]
        pred_coord_sp_stay = sim.data[p_pred_coord_sp_stay]
        truth_coord_moved = sim.data[p_truth_coord_moved]
        truth_coord_stay = sim.data[p_truth_coord_stay]

        # Only keep the center timesteps from the period
        trimmed_pred_coord_sp_moved = pred_coord_sp_moved[1::2]
        trimmed_pred_coord_sp_stay = pred_coord_sp_stay[1::2]
        trimmed_truth_coord_moved = truth_coord_moved[1::2]
        trimmed_truth_coord_stay = truth_coord_stay[1::2]

        # Add datapoints to the pandas dataframe
        for i in range(trimmed_pred_coord_sp_moved.shape[0]):

            acc_moved = loc_match(
                sp=trimmed_pred_coord_sp_moved[i, :],
                heatmap_vectors=larger_heatmap_vectors,
                coord=trimmed_truth_coord_moved[i, :],
                xs=xs * 2,
                ys=ys * 2,
                distance_threshold=0.5,
                sim_threshold=args.similarity_threshold,
            )

            acc_stay = loc_match(
                sp=trimmed_pred_coord_sp_stay[i, :],
                heatmap_vectors=larger_heatmap_vectors,
                coord=trimmed_truth_coord_stay[i, :],
                xs=xs * 2,
                ys=ys * 2,
                distance_threshold=0.5,
                sim_threshold=args.similarity_threshold,
            )

            df = df.append(
                {
                    "Accuracy (moved)": acc_moved,
                    "Accuracy (not moved)": acc_stay,
                    "Accuracy": (acc_moved + acc_stay)/2.,
                    "Items": n_items,
                    "Dimensionality": D,
                    "Query Type": 'Sliding Object',
                    "Limit": args.limit,
                    "Circular Convolution Neurons": n_cconv_neurons,
                    "Time Per Sample": args.time_per_sample,
                },
                ignore_index=True
            )

    return df
    # df.to_csv(fname + '.csv')


if __name__ == '__main__':
    main(experiment_list)
