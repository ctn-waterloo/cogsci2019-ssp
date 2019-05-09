# Capacity and Accuracy figure for the paper
# Show neural and ideal approach

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    ideal_fnames = [
        # 'query_suite_results/query_suite_500_samples_512D.npz',
        'output/non_neural_capacity/query_experiment_limit5_500_samples_128D.npz',
        'output/non_neural_capacity/query_experiment_limit5_500_samples_256D.npz',
        'output/non_neural_capacity/query_experiment_limit5_500_samples_512D.npz',
    ]

    neural_fnames = [
        # 'output/neural_capacity/neural_loc_query_experiment_npd50_512D_2000samples.csv',
        'output/neural_capacity/query_experiment_npd50_128D_200samples.csv',
        'output/neural_capacity/query_experiment_npd50_256D_200samples.csv',
        'output/neural_capacity/query_experiment_npd50_512D_200samples.csv',
        # 'output/neural_capacity/query_experiment_npd15_512D_100samples.csv',
    ]

    df_ideal = pd.DataFrame()
    df_neural = pd.DataFrame()

    for fname in ideal_fnames:
        df_ideal = df_ideal.append(load_non_neural(fname), ignore_index=True)

    for fname in neural_fnames:
        df_neural = df_neural.append(load_neural(fname), ignore_index=True)

    # Add a new column with the implementation
    df_ideal['Implementation'] = 'Non Neural'
    df_neural['Implementation'] = 'Neural'

    df_neural_location = df_neural[df_neural['Query Type'] == 'Location']
    df_neural_item = df_neural[df_neural['Query Type'] == 'Item']

    # rename columns to be consistent with the other dataframe
    df_neural_location = df_neural_location.rename(
        index=str, columns={
            "Similarity": "Location Query Similarity",
            "Accuracy": "Location Query Correct",
        }
    )
    df_neural_item = df_neural_item.rename(
        index=str, columns={
            "Similarity": "Item Query Similarity",
            "Accuracy": "Item Query Within Threshold",
        }
    )

    df_location = df_ideal.append(df_neural_location, ignore_index=True)
    df_item = df_ideal.append(df_neural_item, ignore_index=True)

    title_font_size = 16
    label_font_size = 14

    fig, ax = plt.subplots(2, 2, sharey='row', sharex='col', tight_layout=True, figsize=(8, 6))

    # only use one limit for the plots
    # df_ideal = df_ideal[df_ideal['Limit'] == 5]

    # Location query capacity
    ax[0, 0].set(xscale="log")
    sns.lineplot(
        data=df_location,
        x="Items",
        y="Location Query Similarity",
        style="Dimensionality",
        hue="Implementation",
        ax=ax[0, 0],
    )
    ax[0, 0].set_title('Location Query Capacity', fontsize=title_font_size)
    ax[0, 0].set_ylabel("Similarity", fontsize=label_font_size)

    ax[0, 0].plot(
        [df_location["Items"].min(), df_location["Items"].max()],
        [0.133, 0.133],
        linewidth=1, color='black', linestyle='--'
    )

    # Location query accuracy
    ax[1, 0].set(xscale="log")
    sns.lineplot(
        data=df_location,
        x="Items",
        y="Location Query Correct",
        style="Dimensionality",
        hue="Implementation",
        ax=ax[1, 0],
    )
    ax[1, 0].set_title("Location Query Accuracy", fontsize=title_font_size)
    ax[1, 0].set_xlabel("Number of Stored Items", fontsize=label_font_size)
    ax[1, 0].set_ylabel("Accuracy", fontsize=label_font_size)

    # Item query capacity
    ax[0, 1].set(xscale="log")
    sns.lineplot(
        data=df_item,
        x="Items",
        y="Item Query Similarity",
        style="Dimensionality",
        hue="Implementation",
        ax=ax[0, 1],
    )
    ax[0, 1].set_title('Item Query Capacity', fontsize=title_font_size)

    ax[0, 1].plot(
        [df_location["Items"].min(), df_location["Items"].max()],
        [0.154, 0.154],
        linewidth=1, color='black', linestyle='--'
    )

    # ax[0, 1].legend(['Non-Neural', 'Neural'])

    # Item query accuracy
    ax[1, 1].set(xscale="log")
    sns.lineplot(
        data=df_item,
        x="Items",
        y="Item Query Within Threshold",
        style="Dimensionality",
        hue="Implementation",
        ax=ax[1, 1],
    )
    ax[1, 1].set_title("Item Query Accuracy", fontsize=title_font_size)
    ax[1, 1].set_xlabel("Number of Stored Items", fontsize=label_font_size)

    # Remove all legends except the top right
    ax[0, 0].get_legend().remove()
    ax[1, 0].get_legend().remove()
    ax[1, 1].get_legend().remove()

    fig.savefig('output/cap_acc.pdf', dpi=600, bbox_inches='tight')

    plt.show()


def load_neural(fname_neural):
    df_neural = pd.read_csv(fname_neural)

    return df_neural


########################
# Load non-neural data #
########################

def load_non_neural(fname_ideal):
    data = np.load(fname_ideal)

    # Some processing is needed to get the data into the correct format for the plots

    items_used = data['items_used']
    loc_sp_used = data['loc_sp_used']
    coord_used = data['coord_used']
    lq_similarity = data['lq_similarity']
    iq_similarity = data['iq_similarity']
    dim = data['dim']
    vocab_vectors = data['vocab_vectors']
    item_amounts = data['item_amounts']
    limits = data['limits']
    extract_locs = data['extract_locs']
    extract_items = data['extract_items']

    n_samples = lq_similarity.shape[2]

    res = 128

    # location query, nearest neighbors accuracy, all of the vocab
    lq_nn_accuracy = np.zeros((len(limits), len(item_amounts), n_samples))
    # location query, nearest neighbors accuracy, only items used in the memory
    lq_mem_accuracy = np.zeros((len(limits), len(item_amounts), n_samples))
    # item query, nearest neighbors accuracy
    iq_nn_accuracy = np.zeros((len(limits), len(item_amounts), n_samples))
    # item query, within threshold accuracy
    iq_thresh_accuracy = np.zeros((len(limits), len(item_amounts), n_samples))

    threshold = 0.5

    for li, limit in enumerate(limits):

        for ni, n_items in enumerate(item_amounts):

            # Similarity to all items in the vocab
            sim_items = np.dot(extract_items[li, ni, :, :], vocab_vectors.T)
            # Similarity to all items that were present in the memory
            sim_items_used = np.dot(extract_items[li, ni, :, :], items_used[li, ni, :, :].T)
            # Similarity to all locations that were present in the memory
            sim_locs = np.dot(extract_locs[li, ni, :, :], loc_sp_used[li, ni, :, :].T)

            closest_indx_items_used = np.argmax(sim_items_used, axis=1)
            closest_indx_items = np.argmax(sim_items, axis=1)
            closest_indx_locs = np.argmax(sim_locs, axis=1)
            for s in range(n_samples):
                if np.allclose(vocab_vectors[closest_indx_items[s]], items_used[li, ni, s, :]):
                    lq_nn_accuracy[li, ni, s] = 1
                else:
                    lq_nn_accuracy[li, ni, s] = 0

                if np.allclose(items_used[li, ni, closest_indx_items_used[s], :], items_used[li, ni, s, :]):
                    lq_mem_accuracy[li, ni, s] = 1
                else:
                    lq_mem_accuracy[li, ni, s] = 0

                if np.allclose(loc_sp_used[li, ni, closest_indx_locs[s], :], loc_sp_used[li, ni, s, :]):
                    iq_nn_accuracy[li, ni, s] = 1
                else:
                    iq_nn_accuracy[li, ni, s] = 0

                coord_recall = coord_used[li, ni, closest_indx_locs[s], :]
                correct_coord = coord_used[li, ni, s, :]

                if np.linalg.norm(coord_recall - correct_coord) < threshold:
                    iq_thresh_accuracy[li, ni, s] = 1
                else:
                    iq_thresh_accuracy[li, ni, s] = 0

    # build a pandas dataframe efficiently
    shape = lq_similarity.shape
    limit_column = np.zeros(shape)
    items_column = np.zeros(shape)

    dimensionality_column = np.ones(shape).astype(np.int32) * dim

    for li, limit in enumerate(limits):
        limit_column[li, :, :] = limit

    for ii, n_items in enumerate(item_amounts):
        items_column[:, ii, :] = n_items

    ds = np.vstack(
        [lq_similarity.flatten(),
         iq_similarity.flatten(),
         lq_nn_accuracy.flatten(),
         lq_mem_accuracy.flatten(),
         iq_nn_accuracy.flatten(),
         iq_thresh_accuracy.flatten(),
         limit_column.flatten(),
         items_column.flatten(),
         dimensionality_column.flatten(),
         ]
    )
    column_names = [
        'Location Query Similarity',
        'Item Query Similarity',
        'Location Query Correct',
        'Location Query Correct Within Memory',
        'Item Query Correct',
        'Item Query Within Threshold',
        'Limit',
        'Items',
        'Dimensionality',
    ]

    df = pd.DataFrame(
        data=ds.T,
        columns=column_names,
    )

    return df


if __name__ == '__main__':
    main()
