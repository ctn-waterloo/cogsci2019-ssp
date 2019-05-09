import numpy as np
import nengo.spa as spa


def item_match(sp, vocab_vectors, item, sim_threshold=0.5):
    """
    :param sp: semantic pointer in question
    :param vocab_vectors: all vectors in the vocabulary
    :param item: vector for the item to be matched
    :param sim_threshold: similarity (dot product) threshold to count as a match
    :return: 1 if the closest vector in 'vocab_vectors' to 'sp' is 'item', 0 otherwise
    """
    if sp.__class__.__name__ == 'SemanticPointer':
        sim = np.tensordot(sp.v, vocab_vectors, axes=([0], [1]))
    else:
        sim = np.tensordot(sp, vocab_vectors, axes=([0], [1]))

    ind = np.argmax(sim)

    if sim[ind] < sim_threshold:
        return 0

    if np.allclose(vocab_vectors[ind], item):
        return 1
    else:
        return 0


def item_match_neural(sp, vocab_vectors, item, sim_threshold=0.5):
    """
    special version of the 'item_match' function, where the 'item' input doesn't have to be perfect
    :param sp: semantic pointer in question
    :param vocab_vectors: all vectors in the vocabulary
    :param item: vector for the item to be matched
    :param sim_threshold: similarity (dot product) threshold to count as a match
    :return: 1 if the closest vector in 'vocab_vectors' to 'sp' is 'item', 0 otherwise
    """
    if sp.__class__.__name__ == 'SemanticPointer':
        sim = np.tensordot(sp.v, vocab_vectors, axes=([0], [1]))
    else:
        sim = np.tensordot(sp, vocab_vectors, axes=([0], [1]))

    sim_true = np.tensordot(item, vocab_vectors, axes=([0], [1]))

    ind = np.argmax(sim)

    ind_true = np.argmax(sim_true)

    if sim[ind] < sim_threshold:
        if ind == ind_true:
            print("Warning: closest match is correct, but returning 0 due to it being below threshold")
        return 0

    if ind == ind_true:
        return 1
    else:
        return 0


def loc_match(sp, heatmap_vectors, coord, xs, ys, distance_threshold=0.5, sim_threshold=0.5):
    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = ys[xy[1]]

    # Not similar enough to anything, so count as incorrect
    if vs[xy] < sim_threshold:
        return 0

    # If within threshold of the correct location, count as correct
    if (x-coord[0])**2 + (y-coord[1])**2 < distance_threshold**2:
        return 1
    else:
        return 0


def loc_match_duplicate(sp, heatmap_vectors, coord1, coord2, xs, ys,
                        distance_threshold=0.5, sim_threshold=0.5, sigma=5, zero_range=8,
                        ):
    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)
    # print("")
    # print("Coords", coord1, coord2)
    # print("First match:", xy, [xs[xy[0]], ys[xy[1]]], vs[xy])

    # Not similar enough to anything, so count as incorrect
    if vs[xy] < sim_threshold:
        return 0

    score = 0.

    x = xs[xy[0]]
    y = ys[xy[1]]

    # Check if it found the first coordinate
    if (x - coord1[0]) ** 2 + (y - coord1[1]) ** 2 < distance_threshold ** 2:
        # Check if both points at the same location, if so, count them both as correct
        if (coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 < distance_threshold ** 2:
            return 1

        score += 0.5

        # Explicitly zero-out around the peak
        x1 = max(0, xy[0] - zero_range)
        x2 = min(len(xs), xy[0] + zero_range + 1)
        y1 = max(0, xy[1] - zero_range)
        y2 = min(len(ys), xy[1] + zero_range + 1)
        vs[x1:x2, y1:y2] = 0

        xy = np.unravel_index(vs.argmax(), vs.shape)
        # print("Second match:", xy, [xs[xy[0]], ys[xy[1]]], vs[xy])

        if vs[xy] < sim_threshold:
            return score

        x = xs[xy[0]]
        y = ys[xy[1]]

        if (x - coord2[0]) ** 2 + (y - coord2[1]) ** 2 < distance_threshold ** 2:
            return 1

    elif (x - coord2[0]) ** 2 + (y - coord2[1]) ** 2 < distance_threshold ** 2:
        # Check if both points at the same location, if so, count them both as correct
        if (coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 < distance_threshold ** 2:
            score += 0.5
            # return 1

        score += 0.5

        # Explicitly zero-out around the peak
        x1 = max(0, xy[0] - zero_range)
        x2 = min(len(xs), xy[0] + zero_range + 1)
        y1 = max(0, xy[1] - zero_range)
        y2 = min(len(ys), xy[1] + zero_range + 1)
        vs[x1:x2, y1:y2] = 0

        xy = np.unravel_index(vs.argmax(), vs.shape)
        # print("Second match:", xy, [xs[xy[0]], ys[xy[1]]], vs[xy])

        if vs[xy] < sim_threshold:
            return score

        x = xs[xy[0]]
        y = ys[xy[1]]

        if (x - coord1[0]) ** 2 + (y - coord1[1]) ** 2 < distance_threshold ** 2:
            score += 0.5
            # return 1

    # Very helpful for debugging
    # plt.imshow(vs)
    # plt.show()

    return score


def region_item_match(sp, vocab_vectors, vocab_indices, sim_threshold=0.5):
    if sp.__class__.__name__ == 'SemanticPointer':
        sim = np.tensordot(sp.v, vocab_vectors, axes=([0], [1]))
    else:
        sim = np.tensordot(sp, vocab_vectors, axes=([0], [1]))

    n_matches = len(vocab_indices)

    # sorts from lowest to highest by default
    indices = np.argsort(sim)
    # reverse to have highest to lowest
    indices = indices[::-1]

    # If nothing should be inside the region
    if n_matches == 0:
        if sim[indices[0]] < sim_threshold:
            return 1
        else:
            return 0

    acc = 0

    # for i in range(n_matches):
    #     if indices[i] in vocab_indices:
    #         acc += 1
    # acc /= n_matches

    for i, ind in enumerate(indices):
        if i < n_matches:
            # Should be in the region and detected in region
            if ind in vocab_indices:
                acc += 1
        else:
            # Should be outside the region and detected outside region
            if ind not in vocab_indices:
                acc += 1

    acc /= vocab_vectors.shape[0]

    return acc


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)


def encode_point(x, y, x_axis_sp, y_axis_sp):

    return power(x_axis_sp, x) * power(y_axis_sp, y)


def make_good_unitary(D, eps=1e-3, rng=np.random):
    a = rng.rand((D - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(D, dtype='complex64')
    fv[0] = 1
    fv[1:(D + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:D // 2:-1] = np.conj(fv[1:(D + 1) // 2])
    if D % 2 == 0:
        fv[D // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return spa.SemanticPointer(v)


def circular_region(xs, ys, radius=1, x_offset=0, y_offset=0):
    region = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (x - x_offset)**2 + (y - y_offset)**2 < radius**2:
                region[j, i] = 1

    return region


def simplify_angle(theta):
    """
    Convert the given angle to be between -pi and pi
    """
    while theta > np.pi:
        theta -= 2*np.pi
    while theta < -np.pi:
        theta += 2*np.pi
    return theta


def arc_region(xs, ys, center, arc):
    region = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            angle = np.arctan2(x, y)
            diff = simplify_angle(angle - center)
            if diff < arc / 2. and diff > -arc / 2.:
                region[j, i] = 1

    return region


def get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
            )
            vectors[i, j, :] = p.v

    return vectors


def spatial_dot(vec, xs, ys, x_axis_sp, y_axis_sp, z_axis_sp=None):
    vs = np.zeros((len(ys), len(xs)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if z_axis_sp is None:
                p = encode_point(
                    x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
                )
            else:
                p = encode_hex_point(
                    x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, z_axis_sp=z_axis_sp,
                )
            # Allow support for both vectors and semantic pointers as input
            if vec.__class__.__name__ == 'SemanticPointer':
                vs[j, i] = np.dot(vec.v, p.v)
            else:
                vs[j, i] = np.dot(vec, p.v)
    return vs


def pytorch_spatial_dot(vec, xs, ys, x_axis_sp, y_axis_sp):
    """
    Perform a spatial dot in a way that pytorch can compute the gradient
    vec is a pytorch tensor
    """
    # vs = torch.Tensor(np.zeros((len(ys), len(xs))))
    # including batch dimension
    vs = torch.Tensor(np.zeros((vec.shape[0], len(ys), len(xs))))

    vec_view = vec.view(vec.shape[0], 1, vec.shape[1])

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp
            )

            # Convert to pytorch tensor with the appropriate dtype
            pt = torch.from_numpy(p.v.astype(np.float32))

            # Expand to the batch dimension (manual broadcast) and shape the view for a dot product as a matrix multiply
            pt = pt.expand(vec.shape[0], pt.shape[0]).view(vec.shape[0], pt.shape[0], 1)

            # Compute the batchwise matrix multiply (dot prodict)
            # Shape the view to be one dimensional to fit in the appropriate element
            vs[:, j, i] = torch.bmm(vec_view, pt).view(-1)
    return vs


def generate_item_memory(dim, n_items, limits, x_axis_sp, y_axis_sp, normalize_memory=True, encoding='pow'):
    """
    Create a semantic pointer that contains a number of items bound with respective coordinates
    Returns the memory, along with a list of the items and coordinates used
    The encoding parameter determines which method is used to store items at locations
    """

    assert encoding in ['pow', 'mag', 'sep_pow']

    # Start with an empty memory
    memory_sp = spa.SemanticPointer(data=np.zeros((dim,)))
    coord_list = []
    item_list = []

    for n in range(n_items):
        # Generate random point
        x = np.random.uniform(low=limits[0], high=limits[1])
        y = np.random.uniform(low=limits[2], high=limits[3])

        # Generate random item
        item = spa.SemanticPointer(dim)

        # Add the item to memory at the particular location
        # This is done differently depending on the encoding method
        if encoding == 'pow':
            # Circular convolution power representation
            pos = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

            # Add the item at the point to memory
            memory_sp += (pos * item)
        elif encoding == 'mag':
            # Magnitude scaling representation
            memory_sp += (x * (item * x_axis_sp) + y * (item * y_axis_sp))
        elif encoding == 'sep_pow':
            # Power representation, but X and Y are independent
            memory_sp += (item * power(x_axis_sp, x) + item * power(y_axis_sp, y))

        coord_list.append((x, y))
        item_list.append(item)

    if normalize_memory:
        memory_sp.normalize()

    return memory_sp, coord_list, item_list


def compute_metrics(predictions, coords):
    """
    measure how well the predictions match the true coordinates
    """
    mse = mean_squared_error(y_true=coords, y_pred=predictions)
    abs_diff = np.linalg.norm(coords - predictions, axis=1)
    max_diff = np.max(abs_diff)
    info = {
        'mse': mse,
        'max_diff': max_diff,
        'abs_diff': abs_diff
    }

    return info


def generate_memory_dataset(
                            n_samples,
                            dim,
                            n_items,
                            item_set=None,
                            allow_duplicate_items=False,
                            x_axis_sp=None,
                            y_axis_sp=None,
                            z_axis_sp=None,
                            hexagonal_coordinates=False,
                            limits=(-1, 1, -1, 1),
                            seed=13,
                            normalize_memory=True):
    """
    Create a dataset of memories that contain items bound to coordinates

    :param n_samples: number of memories to create
    :param dim: dimensionality of the memories
    :param n_items: number of items in each memory
    :param item_set: optional list of possible item vectors. If not supplied they will be generated randomly
    :param allow_duplicate_items: if an item set is given, this will allow the same item to be at multiple places
    :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param z_axis_sp: optional z_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param hexagonal_coordinates: if three axes of a hexagonal system are to be used
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :param normalize_memory: if true, call normalize() on the memory semantic pointer after construction
    :return: memory, items, coords, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    if x_axis_sp is None:
        x_axis_sp = spa.SemanticPointer(dim)
        x_axis_sp.make_unitary()
    if y_axis_sp is None:
        y_axis_sp = spa.SemanticPointer(dim)
        y_axis_sp.make_unitary()
    if z_axis_sp is None:
        z_axis_sp = spa.SemanticPointer(dim)
        z_axis_sp.make_unitary()

    # This dataset can be used in two ways, given an item and a memory, come of with the coordinate,
    # or given an coordinate and a memory, come up with the item

    # Memory containing n_items of items bound to coordinates
    memory = np.zeros((n_samples, dim))

    # SP for the item of interest
    items = np.zeros((n_samples, n_items, dim))

    # Coordinate for the item of interest
    coords = np.zeros((n_samples, n_items, 2))

    for i in range(n_samples):
        memory_sp = spa.SemanticPointer(data=np.zeros((dim,)))

        # If a set of items is given, choose a subset to use now
        if item_set is not None:
            items_used = np.random.choice(item_set, size=n_items, replace=allow_duplicate_items)
        else:
            items_used = None

        for j in range(n_items):

            x = np.random.uniform(low=limits[0], high=limits[1])
            y = np.random.uniform(low=limits[2], high=limits[3])

            if hexagonal_coordinates:
                pos = encode_hex_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, z_axis_sp=z_axis_sp)
            else:
                pos = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

            if items_used is None:
                item = spa.SemanticPointer(dim)
            else:
                item = spa.SemanticPointer(data=items_used[j])

            items[i, j, :] = item.v
            coords[i, j, 0] = x
            coords[i, j, 1] = y
            memory_sp += (pos * item)

        if normalize_memory:
            memory_sp.normalize()

        memory[i, :] = memory_sp.v

    return memory, items, coords, x_axis_sp, y_axis_sp, z_axis_sp


def generate_coord_dataset(
                           n_samples,
                           dim,
                           x_axis_sp=None,
                           y_axis_sp=None,
                           z_axis_sp=None,
                           hexagonal_coordinates=False,
                           limits=(-1, 1, -1, 1),
                           seed=13):
    """
    Create a dataset of semantic pointer coordinates and their corresponding real coordinates

    :param n_samples: number of coordinates to create
    :param dim: dimensionality of the semantic pointers
    :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param z_axis_sp: optional z_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param hexagonal_coordinates: if three axes of a hexagonal system are to be used
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :return: vectors, coords, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    if x_axis_sp is None:
        x_axis_sp = spa.SemanticPointer(dim)
        x_axis_sp.make_unitary()
    if y_axis_sp is None:
        y_axis_sp = spa.SemanticPointer(dim)
        y_axis_sp.make_unitary()
    if z_axis_sp is None:
        z_axis_sp = spa.SemanticPointer(dim)
        z_axis_sp.make_unitary()

    # Semantic pointer vectors
    vectors = np.zeros((n_samples, dim))

    # Actual coordinates
    coords = np.zeros((n_samples, 2))

    for i in range(n_samples):
        x = np.random.uniform(low=limits[0], high=limits[1])
        y = np.random.uniform(low=limits[2], high=limits[3])
        if hexagonal_coordinates:
            vectors[i, :] = encode_hex_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, z_axis_sp=z_axis_sp).v
        else:
            vectors[i, :] = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp).v
        coords[i, 0] = x
        coords[i, 1] = y

    return vectors, coords, x_axis_sp, y_axis_sp, z_axis_sp


def generate_directional_relation_dataset(
                           n_samples,
                           dim,
                           x_axis_sp=None,
                           y_axis_sp=None,
                           z_axis_sp=None,
                           hexagonal_coordinates=False,
                           limits=(-1, 1, -1, 1),
                           seed=13):
    """
    Create a dataset of pairs of semantic pointer coordinates, their corresponding real coordinates,
    and the direction between them, going from 'first_coord' to 'second_coord'

    :param n_samples: number of coordinate pairs to create
    :param dim: dimensionality of the semantic pointers
    :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param z_axis_sp: optional z_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param hexagonal_coordinates: if three axes of a hexagonal system are to be used
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :return: first_vectors, second_vectors, first_coords, second_coords, directions, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    if x_axis_sp is None:
        x_axis_sp = spa.SemanticPointer(dim)
        x_axis_sp.make_unitary()
    if y_axis_sp is None:
        y_axis_sp = spa.SemanticPointer(dim)
        y_axis_sp.make_unitary()
    if z_axis_sp is None:
        z_axis_sp = spa.SemanticPointer(dim)
        z_axis_sp.make_unitary()

    # Semantic pointer vectors
    first_vectors = np.zeros((n_samples, dim))
    second_vectors = np.zeros((n_samples, dim))

    # Actual coordinates
    first_coords = np.zeros((n_samples, 2))
    second_coords = np.zeros((n_samples, 2))

    directions = np.zeros((n_samples, 2))

    magnitudes = np.zeros((n_samples, 1))

    for i in range(n_samples):
        x1 = np.random.uniform(low=limits[0], high=limits[1])
        y1 = np.random.uniform(low=limits[2], high=limits[3])
        x2 = np.random.uniform(low=limits[0], high=limits[1])
        y2 = np.random.uniform(low=limits[2], high=limits[3])
        displacement = np.array((x2 - x1, y2 - y1))
        magnitudes[i] = np.linalg.norm(displacement)
        directions[i, :] = displacement / np.linalg.norm(displacement)

        if hexagonal_coordinates:
            first_vectors[i, :] = encode_hex_point(
                x1, y1, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, z_axis_sp=z_axis_sp
            ).v
            second_vectors[i, :] = encode_hex_point(
                x2, y2, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, z_axis_sp=z_axis_sp
            ).v
        else:
            first_vectors[i, :] = encode_point(
                x1, y1, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp
            ).v
            second_vectors[i, :] = encode_point(
                x2, y2, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp
            ).v

        first_coords[i, 0] = x1
        first_coords[i, 1] = y1
        second_coords[i, 0] = x2
        second_coords[i, 1] = y2

    return {
        "first_vectors": first_vectors,
        "second_vectors": second_vectors,
        "first_coords": first_coords,
        "second_coords": second_coords,
        "directions": directions,
        "magnitudes": magnitudes,
        "x_axis_sp": x_axis_sp,
        "y_axis_sp": y_axis_sp,
        "z_axis_sp": z_axis_sp
    }


class MemoryDataset(object):

    def __init__(self,
                 # n_samples,
                 dim,
                 n_items,
                 # item_set=None,
                 allow_duplicate_items=False,
                 x_axis_sp=None,
                 y_axis_sp=None,
                 limits=(-1, 1, -1, 1),
                 seed=13,
                 normalize_memory=True
                 ):

        np.random.seed(seed)

        self.dim = dim
        self.n_items = n_items
        self.allow_duplicate_items = allow_duplicate_items
        self.limits = limits
        self.x_axis_sp = x_axis_sp
        self.y_axis_sp = y_axis_sp
        self.normalize_memory = normalize_memory

    def sample_generator(self, return_coord_sp=False, item_set=None):

        if return_coord_sp:
            while True:
                memory, items, coord_sps, coords = self.generate_simple_memory_dataset(
                    return_coord_sp=return_coord_sp,
                    item_set=item_set,
                    n_items=self.n_items,
                    allow_duplicate_items=self.allow_duplicate_items,
                )

                for i in range(self.n_items):
                    yield memory, items[i, :], coord_sps[i, :], coords[i, :]
        else:
            while True:
                memory, items, coords = self.generate_simple_memory_dataset(
                    return_coord_sp=return_coord_sp,
                    item_set=item_set,
                    n_items=self.n_items,
                    allow_duplicate_items=self.allow_duplicate_items,
                )

                for i in range(self.n_items):

                    yield memory, items[i, :], coords[i, :]

    def variable_item_sample_generator(self, return_coord_sp=False, item_set=None, n_items_min=2, n_items_max=8):

        while True:
            n_items = np.random.randint(low=n_items_min, high=n_items_max + 1)
            memory, items, coords = self.generate_simple_memory_dataset(
                return_coord_sp=return_coord_sp,
                item_set=item_set,
                n_items=n_items,
                allow_duplicate_items=self.allow_duplicate_items,
            )

            for i in range(n_items):
                yield memory, items[i, :], coords[i, :], n_items

    def duplicates_sample_generator(self, return_coord_sp=False, item_set=None, n_items_min=2, n_items_max=8):

        while True:
            n_items = np.random.randint(low=n_items_min, high=n_items_max+1)
            memory, items, coords = self.generate_simple_memory_dataset(
                return_coord_sp=return_coord_sp,
                item_set=item_set,
                n_items=n_items,
                allow_duplicate_items=True,
            )

            # Return only the first two items and coords, which correspond to the duplicates
            assert(np.allclose(items[0, :], items[1, :]))
            yield memory, items[0, :], coords[0, :], coords[1, :]

            # for i in range(n_items):
            #
            #     yield memory, items[i, :], coords[i, :]

    def multi_return_sample_generator(self, return_coord_sp=False, item_set=None, n_items=3, allow_duplicate_items=False):

        while True:
            memory, items, coords = self.generate_simple_memory_dataset(
                return_coord_sp=return_coord_sp,
                item_set=item_set,
                n_items=n_items,
                allow_duplicate_items=allow_duplicate_items,
            )

            yield memory, items, coords

    def region_sample_generator(self, xs, ys, vocab_vectors, return_coord_sp=False, n_items_min=2, n_items_max=8, allow_duplicate_items=False,
                                rad_min=1, rad_max=3):

        while True:
            n_items = np.random.randint(low=n_items_min, high=n_items_max+1)
            memory, items, coord_sps, coords, region_vec, vocab_indices = self.generate_region_memory_dataset(
                return_coord_sp=return_coord_sp,
                vocab_vectors=vocab_vectors,
                n_items=n_items,
                allow_duplicate_items=allow_duplicate_items,
                rad_min=rad_min,
                rad_max=rad_max,
                xs=xs,
                ys=ys,
            )

            yield memory, items, coords, region_vec, vocab_indices

    def generate_simple_memory_dataset(self, n_items, allow_duplicate_items, item_set=None, return_coord_sp=False):
        """
        Create a dataset of memories that contain items bound to coordinates.
        """

        # This dataset can be used in two ways, given an item and a memory, come of with the coordinate,
        # or given an coordinate and a memory, come up with the item

        # SP for the item of interest
        items = np.zeros((n_items, self.dim))

        # Coordinate for the item of interest
        coords = np.zeros((n_items, 2))

        # Coord SP for the item of interest
        coord_sps = np.zeros((n_items, self.dim))

        memory_sp = spa.SemanticPointer(data=np.zeros((self.dim,)))

        # If a set of items is given, choose a subset to use now
        # if item_set is not None:
        #     items_used = np.random.choice(item_set, size=self.n_items, replace=self.allow_duplicate_items)
        # else:
        #     items_used = None
        if item_set is not None:
            # Note: shuffle does an in-place change. If you want the original array not to be modified, make a copy
            np.random.shuffle(item_set)

        for j in range(n_items):

            x = np.random.uniform(low=self.limits[0], high=self.limits[1])
            y = np.random.uniform(low=self.limits[2], high=self.limits[3])

            pos = encode_point(x, y, x_axis_sp=self.x_axis_sp, y_axis_sp=self.y_axis_sp)
            coord_sps[j, :] = pos.v

            # if items_used is None:
            #     item = spa.SemanticPointer(self.dim)
            # else:
            #     item = spa.SemanticPointer(data=items_used[j])
            if item_set is None:
                item = spa.SemanticPointer(self.dim)
            else:
                if allow_duplicate_items and j == 1:
                    # Simple hack to guarantee the first two items are the same
                    item = spa.SemanticPointer(data=item_set[0, :])
                else:
                    item = spa.SemanticPointer(data=item_set[j, :])

            items[j, :] = item.v
            coords[j, 0] = x
            coords[j, 1] = y
            memory_sp += (pos * item)

        if self.normalize_memory:
            memory_sp.normalize()

        if return_coord_sp:
            return memory_sp.v, items, coord_sps, coords
        else:
            return memory_sp.v, items, coords

    def generate_region_memory_dataset(self, vocab_vectors, n_items, xs, ys, rad_min, rad_max,
                                       allow_duplicate_items, return_coord_sp=False):
        """
        Create a dataset of memories that contain items bound to coordinates along with a region and the indices
        of which vocab items are within the region.
        """

        rad = np.random.uniform(low=rad_min, high=rad_max)
        x_offset = np.random.uniform(low=self.limits[0], high=self.limits[1])
        y_offset = np.random.uniform(low=self.limits[2], high=self.limits[3])
        desired = circular_region(xs, ys, radius=rad, x_offset=x_offset, y_offset=y_offset)
        region_sp = generate_region_vector(desired, xs, ys, self.x_axis_sp, self.y_axis_sp)

        # This dataset can be used in two ways, given an item and a memory, come of with the coordinate,
        # or given an coordinate and a memory, come up with the item

        # SP for the item of interest
        items = np.zeros((n_items, self.dim))

        # Coordinate for the item of interest
        coords = np.zeros((n_items, 2))

        # Coord SP for the item of interest
        coord_sps = np.zeros((n_items, self.dim))

        memory_sp = spa.SemanticPointer(data=np.zeros((self.dim,)))

        indices = np.arange(vocab_vectors.shape[0])

        np.random.shuffle(indices)

        # indices for items inside the region
        inside_indices = []

        for j in range(n_items):

            x = np.random.uniform(low=self.limits[0], high=self.limits[1])
            y = np.random.uniform(low=self.limits[2], high=self.limits[3])

            # Check to see if the object will be encoded within the region
            if (x-x_offset)**2 + (y-y_offset)**2 < rad**2:
                inside_indices.append(indices[j])

            pos = encode_point(x, y, x_axis_sp=self.x_axis_sp, y_axis_sp=self.y_axis_sp)
            coord_sps[j, :] = pos.v

            item = spa.SemanticPointer(data=vocab_vectors[indices[j], :])

            items[j, :] = item.v
            coords[j, 0] = x
            coords[j, 1] = y
            memory_sp += (pos * item)

        if self.normalize_memory:
            memory_sp.normalize()

        return memory_sp.v, items, coord_sps, coords, region_sp.v, inside_indices

def rotate_vector(start_vec, end_vec, theta):
    """
    Rotate a vector starting at 'start_vec' in the plane formed by 'start_vec' and 'end_vec'
    in a direction toward 'end_vec' with an angle of 'theta'
    Returns a new vector that is the result of the rotation
    """
    A_prime = start_vec / np.linalg.norm(start_vec)

    B_tilde = end_vec - np.dot(A_prime, end_vec) * A_prime

    # Orthogonal normalized vector
    B_prime = B_tilde / np.linalg.norm(B_tilde)

    C = np.linalg.norm(start_vec) * ((np.cos(theta) * A_prime + np.sin(theta) * B_prime))

    C_prime = C / np.linalg.norm(C)

    return C_prime


def generate_region_vector(desired, xs, ys, x_axis_sp, y_axis_sp):
    """
    :param desired: occupancy grid of what points should be in the region and which ones should not be
    :param xs: linspace in x
    :param ys: linspace in y
    :param x_axis_sp: x axis semantic pointer
    :param y_axis_sp: y axis semantic pointer
    :return: a normalized semantic pointer designed to be highly similar to the desired region
    """

    vector = np.zeros_like((x_axis_sp.v))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if desired[i, j] == 1:
                vector += encode_point(x, y, x_axis_sp, y_axis_sp).v

    sp = spa.SemanticPointer(data=vector)
    sp.normalize()

    return sp
