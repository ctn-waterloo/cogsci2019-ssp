# plot results from either 'exp_non_neural.py' or 'exp_neural.py'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

parser = argparse.ArgumentParser('Plotting the performance of various capabilities of spatial semantic pointers')

parser.add_argument('--datafile', default='output/non_neural_results/seed13_dim512_min2_max12.npz', help='data file to plot')

args = parser.parse_args()

exp_names = [
    'Single Object',
    'Missing Object',
    'Duplicate Object',
    'Location',
    'Sliding Group',
    'Sliding Object',
    'Sliding Object (moved only)',  # this one only counts the accuracy of the one object that has moved
    'Sliding Object Scaled',
    'Sliding Object Scaled (moved only)',  # this one only counts the accuracy of the one object that has moved
    'Region',
]

dict_keys = [
    'single_object',
    'missing_object',
    'duplicate_object',
    'location',
    'sliding_group',
    'sliding_object',
    'sliding_object_moved_only',
    'sliding_object_scaled',
    'sliding_object_scaled_moved_only',
    'region',
]

if '.npz' in args.datafile:
    # Non-neural data is saved in this format
    data = np.load(args.datafile)

    item_range = data['item_range']

    # Format the data into a pandas dataframe for plotting with seaborn
    df = pd.DataFrame()

    for i in range(len(dict_keys)):
        for n, n_items in enumerate(item_range):
            df = df.append(
                pd.DataFrame(
                    data={'Accuracy': data[dict_keys[i]][n, :], 'Query': exp_names[i], 'Items': n_items}
                )
            )
elif '.csv' in args.datafile:
    # Neural data is saved in this format
    df = pd.read_csv(args.datafile)
    df.rename(columns={'Query Type': 'Query'}, inplace=True)
    pass  # TODO:
else:
    raise NotImplementedError("Data format not recognized for: {}\n"
                              "Use .npz for numpy arrays and .csv for pandas dataframes".format(args.datafile))

print(df.columns.tolist())

# Print out means and standard deviations across n_items
print("Mean\t    Std\t\t    Query")
for label in exp_names:
    if '.csv' in args.datafile and label == 'Sliding Object':
        acc_name = 'Accuracy (not moved)'
    elif '.csv' in args.datafile and label == 'Sliding Object (moved only)':
        acc_name = 'Accuracy (moved)'
        label = 'Sliding Object'
    else:
        acc_name = 'Accuracy'
    mean = df.loc[df['Query'] == label][acc_name].mean()
    std = df.loc[df['Query'] == label][acc_name].std()
    print('{0:.3f}\t +/- {1:.3f}\t {2}'.format(mean, std, label))


sns.lineplot(data=df, x='Items', y='Accuracy', hue='Query')
plt.show()

