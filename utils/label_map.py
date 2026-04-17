import string

NUM_CLASSES = 226

def generate_labels(n):
    labels = {}
    letters = list(string.ascii_uppercase)

    for i in range(n):
        letter = letters[i % 26]
        repeat = i // 26

        if repeat == 0:
            name = f"Gesture {letter}"
        else:
            name = f"Gesture {letter}{repeat}"

        labels[i] = name

    return labels

LABELS = generate_labels(NUM_CLASSES)