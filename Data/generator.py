# Test 1
import numpy as np

np.random.seed(42)


def train_test_split(x, y, test_size=0.2):
    # Split data
    num_samples = len(x)
    num_test = int(test_size * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    x_train = [x[i] for i in train_indices]
    x_test = [x[i] for i in test_indices]

    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return x_train, x_test, y_train, y_test


def generate_data(num_samples=100, dimension=1, test_size=0.2, m=7, b=3, low=0, high=10, categorical=False):
    # Generate x values
    x = np.random.uniform(low=low, high=high, size=(num_samples, dimension))

    # Compute y values with noise
    if dimension == 1:
        # noise = np.random.normal(loc=0, scale=1, size=num_samples)
        noise = np.random.normal(loc=0, scale=1, size=(num_samples, 1))
        y = (m * x) + b + noise

    else:
        noise = np.random.normal(loc=0, scale=1, size=num_samples)
        y = np.dot(x, np.array([m] * dimension)) + b + noise

    y = y.reshape(-1, 1)  # Make y a column vector

    if categorical:
        quartiles = np.percentile(y, [25, 50, 75])
        labels = np.zeros(num_samples, dtype=int)
        labels[y[:, 0] <= quartiles[0]] = 0
        labels[(y[:, 0] > quartiles[0]) & (y[:, 0] <= quartiles[1])] = 1
        labels[(y[:, 0] > quartiles[1]) & (y[:, 0] <= quartiles[2])] = 2
        labels[y[:, 0] > quartiles[2]] = 3
        y = labels
        y = y.reshape(-1, 1)

    # Split data into train/test
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size)

    return xtrain, xtest, ytrain, ytest


def generate_sine_data(num_samples=100, dimension=1, test_size=0.2, amplitude=1, frequency=1, phase=0, low=0, high=10,
                       categorical=False):
    x = np.random.uniform(low=low, high=high, size=(num_samples, dimension))

    if dimension == 1:
        y = amplitude * np.sin(frequency * x + phase)
    else:
        y = np.zeros((num_samples, 1))
        for d in range(dimension):
            y += amplitude * np.sin(frequency * x[:, d].reshape(-1, 1) + phase)

    # print(max(y))
    # print(min(y))

    if categorical:
        # Define bins for y values
        bins = [np.min(y), np.percentile(y, 33), np.percentile(y, 66), np.max(y)]

        # Convert y values to categorical labels
        y = np.digitize(y, bins) - 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


# Generate data
# xtrain, xtest, ytrain, ytest = generate_data()

def q1_a():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000, test_size=0.2)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q1_b():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000, dimension=4, test_size=0.2, low=0, high=10)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    # print(xtrain)
    # print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q2_a():
    xtrain, xtest, ytrain, ytest = generate_sine_data(num_samples=10000, dimension=1, test_size=0.2)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    print(xtrain)
    print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q2_b():
    xtrain, xtest, ytrain, ytest = generate_sine_data(num_samples=100000, dimension=5, test_size=0.2, low=0, high=10)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    print(xtrain)
    print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q3_a_l():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000, test_size=0.2, categorical=True)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    # print(xtrain)
    # print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q3_a_nl():
    xtrain, xtest, ytrain, ytest = generate_sine_data(num_samples=10000, dimension=1, test_size=0.2, categorical=True)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    # print(xtrain)
    # print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q3_b_l():
    xtrain, xtest, ytrain, ytest = generate_data(num_samples=10000, dimension=4, test_size=0.2, low=0, high=10,
                                                 categorical=True)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    # print(xtrain)
    # print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


def q3_b_nl():
    xtrain, xtest, ytrain, ytest = generate_sine_data(num_samples=100000, dimension=5, test_size=0.2, low=0, high=10,
                                                      categorical=True)
    xtrain, xtest, ytrain, ytest = np.array(xtrain, dtype=np.float32), np.array(xtest, dtype=np.float32), np.array(
        ytrain, dtype=np.float32), np.array(ytest, dtype=np.float32)
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
    # print(xtrain)
    # print(ytrain)
    return {
        "train": (xtrain, ytrain),
        "test": (xtest, ytest)
    }


q1_a()
q2_a()
q3_a_l()
q3_a_nl()
print("-----------------------------------------")
q1_b()
q2_b()
q3_b_l()
q3_b_nl()

