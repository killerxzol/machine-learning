import numpy as np
from scratch.linear_model import SGDClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


def batch(X, y, max_iter, random_state):
    model = SGDClassifier(
        penalty="l2",
        alpha=0.01,
        random_state=random_state,
        learning_rate=0.5,
        optimization=None
    )
    for _ in range(max_iter):
        model.partial_fit(X, y)
    return model


def mini_batch(X, y, batch_size, max_iter, random_state):
    model = SGDClassifier(
        penalty="l1",
        alpha=0.1,
        random_state=random_state,
        learning_rate=0.1,
        optimization=None
    )
    random_generator = np.random.default_rng(seed=random_state)

    indices = np.arange(X.shape[0])
    for _ in range(max_iter):
        idx = random_generator.choice(indices, size=batch_size, replace=False)
        model.partial_fit(X[idx, :], y[idx])
    return model


def stochastic(X, y, max_iter, random_state):
    model = SGDClassifier(
        penalty="l2",
        alpha=0.1,
        max_iter=max_iter,
        random_state=random_state,
        learning_rate=0.5,
        optimization="adagrad"
    ).fit(X, y)
    return model


def plot_losses(losses, technique):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(losses, linewidth=2)
    ax.grid(alpha=0.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(f"technique={technique}")
    plt.show()


def main(technique, random_state=None):

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    if technique == "batch":
        model = batch(X_train, y_train, max_iter=300, random_state=random_state)
    elif technique == "mini_batch":
        model = mini_batch(X_train, y_train, batch_size=10, max_iter=300, random_state=random_state)
    elif technique == "stochastic":
        model = stochastic(X_train, y_train, max_iter=300, random_state=random_state)
    else:
        raise ValueError("technique should be `batch`, `mini_batch` or `stochastic`")

    plot_losses(model.losses_, technique)

    y_pred = model.predict(X_test)

    print(f"accuracy_score: {accuracy_score(y_test, y_pred)}")
    print(f"weights: {model.weights_}")


if __name__ == "__main__":
    main(technique="stochastic", random_state=42)
