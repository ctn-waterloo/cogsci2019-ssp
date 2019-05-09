# Measuring the performance of various capabilities of spatial semantic pointers
# such as querying objects, querying locations, moving objects, etc
import numpy as np
import nengo.spa as spa
import matplotlib.pyplot as plt
from utils import item_match, loc_match, loc_match_duplicate, region_item_match, \
    encode_point, get_heatmap_vectors, MemoryDataset, make_good_unitary
import argparse
import os


def main():

    parser = argparse.ArgumentParser('Measuring the performance of various capabilities of spatial semantic pointers')

    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples to evaluate per item number')
    parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
    parser.add_argument('--neurons-per-dim', type=int, default=15)
    parser.add_argument('--limit', type=int, default=5, help='The absolute min and max of the space')
    parser.add_argument('--res', type=int, default=128, help='Resolution for the linspace')
    parser.add_argument('--n-items-min', type=int, default=2, help='Lowest number of items in a memory')
    parser.add_argument('--n-items-max', type=int, default=24, help='Highest number of items in a memory')
    # One threshold is best for region queries, the other for the other queries, TODO: use them in the appropriate places
    parser.add_argument('--similarity-threshold', type=float, default=0.1, help='Similarity must be above this value to count')
    # parser.add_argument('--similarity-threshold', type=float, default=0.25, help='Similarity must be above this value to count')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--folder', default='output/non_neural_results', help='folder to save results')

    args = parser.parse_args()

    fname = 'seed{}_dim{}_min{}_max{}.npz'.format(args.seed, args.dim, args.n_items_min, args.n_items_max)

    # Range of item sizes to try
    item_range = list(range(args.n_items_min, args.n_items_max + 1))
    n_item_range = len(item_range)

    xs = np.linspace(-args.limit, args.limit, args.res)
    ys = np.linspace(-args.limit, args.limit, args.res)

    rstate = np.random.RandomState(seed=args.seed)
    x_axis_sp = make_good_unitary(args.dim, rng=rstate)
    y_axis_sp = make_good_unitary(args.dim, rng=rstate)

    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

    # These are for dealing with shifted memories, that could potentially go outside the normal range
    larger_heatmap_vectors = get_heatmap_vectors(xs*2, ys*2, x_axis_sp, y_axis_sp)

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    results = {
        'single_object': np.zeros((n_item_range, args.n_samples)),
        'missing_object': np.zeros((n_item_range, args.n_samples)),
        'duplicate_object': np.zeros((n_item_range, args.n_samples)),
        'location': np.zeros((n_item_range, args.n_samples)),
        'sliding_group': np.zeros((n_item_range, args.n_samples)),
        'sliding_object': np.zeros((n_item_range, args.n_samples)),
        'sliding_object_moved_only': np.zeros((n_item_range, args.n_samples)),
        'sliding_object_scaled': np.zeros((n_item_range, args.n_samples)),
        'sliding_object_scaled_moved_only': np.zeros((n_item_range, args.n_samples)),
        'region': np.zeros((n_item_range, args.n_samples)),
    }

    for n, n_items in enumerate(item_range):
        print("Running experiments for n_items={}".format(n_items))

        vocab = spa.Vocabulary(args.dim)

        # n_vocab_vectors = args.n_items_max * 2
        n_vocab_vectors = n_items * 2

        vocab_vectors = np.zeros((n_vocab_vectors, args.dim))

        # print("Generating {0} vocab items".format(n_vocab_vectors))
        for i in range(n_vocab_vectors):
            p = vocab.create_pointer()
            vocab_vectors[i, :] = p.v
        # print("Vocab generation complete")

        # A copy that will get shuffled around in MemoryDataset
        vocab_vectors_copy = vocab_vectors.copy()

        dataset = MemoryDataset(
            dim=args.dim,
            n_items=0,  # unused,
            allow_duplicate_items=False,
            limits=(-args.limit, args.limit, -args.limit, args.limit),
            normalize_memory=True,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
        )
        # data_gen = dataset.sample_generator(item_set=vocab_vectors_copy)
        data_gen_var_item = dataset.variable_item_sample_generator(
            item_set=vocab_vectors_copy,
            n_items_min=n_items,
            n_items_max=n_items,
        )
        data_gen_duplicate = dataset.duplicates_sample_generator(
            item_set=vocab_vectors_copy,
            n_items_min=max(2, n_items),
            n_items_max=n_items,
        )

        data_gen_multi = dataset.multi_return_sample_generator(
            item_set=vocab_vectors_copy,
            n_items=n_items,
            allow_duplicate_items=False,
        )

        # Generates circular regions
        data_gen_region = dataset.region_sample_generator(
            vocab_vectors=vocab_vectors,
            xs=xs,
            ys=ys,
            n_items_min=n_items,
            n_items_max=n_items,
            rad_min=1,
            rad_max=3
        )

        # Query Single Object and Query Location
        for s in range(args.n_samples):
            # Acquire the next sample
            mem_v, item_v, coord_v, n_items = data_gen_var_item.__next__()

            item_loc = encode_point(coord_v[0], coord_v[1], x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

            mem_sp = spa.SemanticPointer(data=mem_v)
            loc_result = mem_sp * ~ spa.SemanticPointer(data=item_v)
            item_result = mem_sp * ~ item_loc

            # using a random semantic pointer here
            loc_missing_result = spa.SemanticPointer(data=mem_v) * ~ spa.SemanticPointer(args.dim)

            # TODO: find the grid coordinate of the top location, count as correct it matches the real coordinate
            results['single_object'][n, s] = loc_match(
                sp=loc_result,
                heatmap_vectors=heatmap_vectors,
                coord=coord_v,
                xs=xs,
                ys=ys,
                distance_threshold=0.5,
                sim_threshold=args.similarity_threshold,
            )

            results['location'][n, s] = item_match(
                sp=item_result,
                vocab_vectors=vocab_vectors,
                item=item_v,
                sim_threshold=args.similarity_threshold,
            )

            results['missing_object'][n, s] = 1 - loc_match(
                sp=loc_missing_result,
                heatmap_vectors=heatmap_vectors,
                coord=coord_v,
                xs=xs,
                ys=ys,
                distance_threshold=0.5,
                sim_threshold=args.similarity_threshold,
            )

        # Query Duplicate Objects
        for s in range(args.n_samples):
            # Acquire the next sample for duplicates
            mem_v, item_v, coord1_v, coord2_v = data_gen_duplicate.__next__()

            loc_results = spa.SemanticPointer(data=mem_v) *~ spa.SemanticPointer(data=item_v)

            # TODO: find the grid coordinates of the top two locations, count as correct if they match the real coordinates
            results['duplicate_object'][n, s] = loc_match_duplicate(
                loc_results, heatmap_vectors,
                coord1=coord1_v, coord2=coord2_v, xs=xs, ys=ys, sim_threshold=args.similarity_threshold,
            )

        # Query Region
        # NOTE: threshold will depend on region size
        # TODO: redo that old region experiment with better region generation
        for s in range(args.n_samples):
            mem_v, items, coords, region_v, vocab_indices = data_gen_region.__next__()

            mem_sp = spa.SemanticPointer(data=mem_v)
            region_sp = spa.SemanticPointer(data=region_v)

            region_results = mem_sp * ~region_sp

            results['region'][n, s] = region_item_match(
                region_results, vocab_vectors, vocab_indices, sim_threshold=args.similarity_threshold
            )

        # Sliding Whole Group and Sliding Single Object
        # accuracy will be the number of matches in the end
        for s in range(args.n_samples):
            mem_v, item_vs, coord_vs = data_gen_multi.__next__()

            mem_sp = spa.SemanticPointer(data=mem_v)

            # Choose random amount to move by
            dx = np.random.uniform(-args.limit / 2., args.limit / 2.)
            dy = np.random.uniform(-args.limit / 2., args.limit / 2.)
            slide_vec = np.array([dx, dy])
            # slide_vec = np.array([dy, dx])

            d_coord = encode_point(dx, dy, x_axis_sp, y_axis_sp)

            slide_mem_sp = mem_sp * d_coord

            first_item = spa.SemanticPointer(data=item_vs[0, :])
            first_coord = encode_point(coord_vs[0, 0], coord_vs[0, 1], x_axis_sp, y_axis_sp)
            single_slide_mem_sp = mem_sp + first_item*first_coord*d_coord - first_item*first_coord
            single_slide_mem_sp.normalize()

            # scaling to account for normalization
            scaling = 1 / np.sqrt(n_items)
            single_slide_scaled_mem_sp = mem_sp + scaling*first_item*first_coord*d_coord - scaling*first_item*first_coord
            single_slide_scaled_mem_sp.normalize()

            res_group = 0
            res_single = 0
            res_single_move_only = 0
            res_single_scaled = 0
            res_single_scaled_move_only = 0

            for i in range(n_items):

                loc_result = slide_mem_sp * ~ spa.SemanticPointer(data=item_vs[i, :])

                res_group += loc_match(
                    sp=loc_result,
                    heatmap_vectors=larger_heatmap_vectors,
                    coord=coord_vs[i, :] + slide_vec,
                    xs=xs*2,
                    ys=ys*2,
                    distance_threshold=0.5,
                    sim_threshold=args.similarity_threshold,
                )

                single_loc_result = single_slide_mem_sp * ~ spa.SemanticPointer(data=item_vs[i, :])
                single_loc_scaled_result = single_slide_scaled_mem_sp * ~ spa.SemanticPointer(data=item_vs[i, :])

                # Only the first item has moved for the single movement case
                if i == 0:
                    res_single_move_only = loc_match(
                        sp=single_loc_result,
                        heatmap_vectors=larger_heatmap_vectors,
                        coord=coord_vs[i, :] + slide_vec,
                        xs=xs*2,
                        ys=ys*2,
                        distance_threshold=0.5,
                        sim_threshold=args.similarity_threshold,
                    )
                    res_single += res_single_move_only

                    res_single_scaled_move_only = loc_match(
                        sp=single_loc_scaled_result,
                        heatmap_vectors=larger_heatmap_vectors,
                        coord=coord_vs[i, :] + slide_vec,
                        xs=xs*2,
                        ys=ys*2,
                        distance_threshold=0.5,
                        sim_threshold=args.similarity_threshold,
                    )
                    res_single_scaled += res_single_scaled_move_only
                else:
                    res_single += loc_match(
                        sp=single_loc_result,
                        heatmap_vectors=larger_heatmap_vectors,
                        coord=coord_vs[i, :],
                        xs=xs*2,
                        ys=ys*2,
                        distance_threshold=0.5,
                        sim_threshold=args.similarity_threshold,
                    )

                    res_single_scaled += loc_match(
                        sp=single_loc_scaled_result,
                        heatmap_vectors=larger_heatmap_vectors,
                        coord=coord_vs[i, :],
                        xs=xs*2,
                        ys=ys*2,
                        distance_threshold=0.5,
                        sim_threshold=args.similarity_threshold,
                    )

            res_group /= n_items
            res_single /= n_items
            res_single_scaled /= n_items

            results['sliding_group'][n, s] = res_group
            results['sliding_object'][n, s] = res_single
            results['sliding_object_moved_only'][n, s] = res_single_move_only
            results['sliding_object_scaled'][n, s] = res_single_scaled
            results['sliding_object_scaled_moved_only'][n, s] = res_single_scaled_move_only

    np.savez(
        os.path.join(args.folder, fname),
        item_range=np.array(item_range),
        **results
    )


if __name__ == "__main__":
    main()
