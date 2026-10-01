import numpy as np

from signtrainer import dataset


def write_clip(root, person, session, name, n, handedness="Right"):
    d = root / person / session
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name) + n)
    np.savez(d / f"{name}.npz", points=rng.uniform(0, 500, (n, 21, 3)).astype(np.float32),
             handedness=np.full(n, handedness))


def test_load_own_reads_labels_and_people(tmp_path):
    write_clip(tmp_path, "p01", "s01", "A_1", 5)
    write_clip(tmp_path, "p01", "s01", "A_2", 3)
    write_clip(tmp_path, "p02", "s01", "B_1", 4, "Left")
    data = dataset.load_own(tmp_path)
    assert data["points"].shape == (12, 21, 3)
    assert list(data["labels"]).count("A") == 8
    assert sorted(np.unique(data["person"])) == ["p01", "p02"]


def test_load_own_empty(tmp_path):
    data = dataset.load_own(tmp_path)
    assert all(len(data[k]) == 0 for k in dataset.KEYS)


def test_load_kaggle_missing_file(tmp_path):
    assert len(dataset.load_kaggle(tmp_path / "nope.npz")["labels"]) == 0


def test_features_shape(tmp_path):
    write_clip(tmp_path, "p01", "s01", "C_1", 6)
    X = dataset.features(dataset.load_own(tmp_path))
    assert X.shape == (6, 63)


def test_select_and_concat(tmp_path):
    write_clip(tmp_path, "p01", "s01", "A_1", 2)
    write_clip(tmp_path, "p02", "s01", "B_1", 3)
    data = dataset.load_own(tmp_path)
    p2 = dataset.select(data, data["person"] == "p02")
    assert set(p2["labels"]) == {"B"}
    both = dataset.concat([p2, dataset.empty(), p2])
    assert len(both["labels"]) == 6
