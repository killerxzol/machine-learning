from scratch.generative_model import HMM, SmoothHMM


observations = [
    [],
    ['man'],
    ['The', 'man', 'saw', 'the', 'dog', 'with', 'the', 'telescope', '.']
]

states = [
    [],
    ['p'],
    ['o', 'p', 'o', 'o', 'a', 'o', 'a', 'o', 'o']
]


if __name__ == '__main__':

    model = SmoothHMM(order=2, n_jobs=1)
    model.fit(observations, states)
    decoded = model.predict(observations)

    print(decoded)
    print(model.decode_proba_)

    print(model.A)
