from nltk.corpus import treebank
from scratch.generative_model import HMM, SmoothHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def flatten(sequences):
    return [obs for sequence in sequences for obs in sequence]


def metrics(y_true, y_pred):
    print(f"f1-score: {f1_score(y_true, y_pred, average=None)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")


def main():

    data = treebank.tagged_sents(tagset="universal")

    X, y = zip(*(zip(*sequence) for sequence in data))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = SmoothHMM(order=2, n_jobs=1)
    model = HMM(order=2, n_jobs=1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics(flatten(y_test), flatten(y_pred))


if __name__ == "__main__":
    main()
