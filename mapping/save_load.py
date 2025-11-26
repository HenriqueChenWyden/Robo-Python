import pickle

def save_map(grid, visits, fname="map.pickle"):
    with open(fname, "wb") as f:
        pickle.dump({"grid": grid, "vis": visits}, f)

def load_map(fname="map.pickle"):
    with open(fname, "rb") as f:
        d = pickle.load(f)
        return d["grid"], d["vis"]
            