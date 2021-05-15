import pickle


def load_pickled(pickled_object_filepath: str) -> object:
    """Load pickled object"""
    with open(pickled_object_filepath, "rb") as fin:
        target_object = pickle.load(fin)
    return target_object
