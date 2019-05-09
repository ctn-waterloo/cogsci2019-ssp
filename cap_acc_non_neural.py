import nengo.spa as spa
import numpy as np
import pandas as pd
import argparse
from utils import make_good_unitary, encode_point, MemoryDataset
import os.path as osp
import os

parser = argparse.ArgumentParser('Perform a many query experiments')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--res', type=int, default=128)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--n-samples', type=int, default=500, help='number of memories created at a specific item amount')
parser.add_argument('--normalize-memory', type=int, default=1, choices=[0, 1])
parser.add_argument('--folder', type=str, default='output/non_neural_capacity', help='folder to save the output data')

args = parser.parse_args()

name = 'query_experiment_limit5_{0}_samples_{1}D.npz'.format(args.n_samples, args.dim)

if not osp.exists(args.folder):
    os.makedirs(args.folder)

rstate = np.random.RandomState(seed=args.seed)
x_axis_sp = make_good_unitary(args.dim, rng=rstate)
y_axis_sp = make_good_unitary(args.dim, rng=rstate)

vocab = spa.Vocabulary(args.dim)

item_amounts = [2**n for n in range(12)]
# item_amounts = [2**n for n in range(10)]

n_item_amounts = len(item_amounts)

max_items = 2**11
# max_items = 2**9

vocab_vectors = np.zeros((max_items, args.dim))

print("Generating {0} vocab items".format(max_items))
for i in range(max_items):
    p = vocab.create_pointer()
    vocab_vectors[i, :] = p.v
print("Vocab generation complete")

# A copy that will get shuffled around in MemoryDataset
vocab_vectors_copy = vocab_vectors.copy()

# limits = [.25, .5, 1.0, 2.0, 4.0, 8.0, 16.0]
limits = [5.0]

df = pd.DataFrame()

# location query, nearest neighbors accuracy
lq_nn_accuracy = np.zeros((len(limits), n_item_amounts, args.n_samples))
# item query, nearest neighbors accuracy
iq_nn_accuracy = np.zeros((len(limits), n_item_amounts, args.n_samples))

# location query raw similarity
lq_similarity = np.zeros((len(limits), n_item_amounts, args.n_samples))
# item query raw similarity
iq_similarity = np.zeros((len(limits), n_item_amounts, args.n_samples))

items_used = np.zeros((len(limits), n_item_amounts, args.n_samples, args.dim))
loc_sp_used = np.zeros((len(limits), n_item_amounts, args.n_samples, args.dim))
coord_used = np.zeros((len(limits), n_item_amounts, args.n_samples, 2))

extract_items = np.zeros((len(limits), n_item_amounts, args.n_samples, args.dim))
extract_locs = np.zeros((len(limits), n_item_amounts, args.n_samples, args.dim))

for li, limit in enumerate(limits):
    print("Using Limit of {0}".format(limit))
    for n, n_items in enumerate(item_amounts):
        print("Running Capacity Experiment for {0} Items in Memory".format(n_items))
        # memory will have shape (n_samples, D)
        # items will have shape (n_samples, n_items, D)
        # coords will have shape (n_samples, n_items, 2)
        dataset = MemoryDataset(
            dim=args.dim,
            n_items=n_items,
            allow_duplicate_items=False,
            limits=(-limit, limit, -limit, limit),
            normalize_memory=args.normalize_memory,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
        )
        data_gen = dataset.sample_generator(item_set=vocab_vectors_copy)

        for s in range(args.n_samples):

            # Acquire the next sample
            mem_v, item_v, coord_v = data_gen.__next__()

            mem = spa.SemanticPointer(data=mem_v)

            # Pick one item that is in the memory (in this case the first one)
            item_loc = encode_point(coord_v[0], coord_v[1], x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)
            item_sp = spa.SemanticPointer(data=item_v)

            items_used[li, n, s, :] = item_v
            loc_sp_used[li, n, s, :] = item_loc.v
            coord_used[li, n, s, :] = coord_v

            extract_item = (mem * ~item_loc).v
            extract_loc = (mem * ~item_sp).v

            extract_items[li, n, s, :] = extract_item
            extract_locs[li, n, s, :] = extract_loc

            lq_similarity[li, n, s] = np.dot(extract_item, item_v)
            iq_similarity[li, n, s] = np.dot(extract_loc, item_loc.v)


fname = osp.join(args.folder, name)
print("Saving data to {0}".format(fname))

with open(fname, 'wb') as f:
    np.savez(
        f,
        items_used=items_used,
        loc_sp_used=loc_sp_used,
        coord_used=coord_used,
        lq_similarity=lq_similarity,
        iq_similarity=iq_similarity,
        dim=args.dim,
        theta=np.pi/2.,
        seed=args.seed,
        vocab_vectors=vocab_vectors,
        limits=np.array(limits),
        item_amounts=np.array(item_amounts),
        extract_locs=extract_locs,
        extract_items=extract_items,
        x_axis_vec=x_axis_sp.v,
        y_axis_vec=y_axis_sp.v,
        normalize_memory=args.normalize_memory,
    )
