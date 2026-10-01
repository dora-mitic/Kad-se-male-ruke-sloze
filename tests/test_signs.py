from signtrainer import config, server


def test_sign_images_by_label(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIGN_ASSETS_DIR", tmp_path)
    for name in ("A.png", "b.gif", "C.JPG", "notes.txt", "D.webp"):
        (tmp_path / name).write_bytes(b"x")
    assert server.sign_images() == {"A": "A.png", "B": "b.gif", "C": "C.JPG", "D": "D.webp"}


def test_sign_images_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIGN_ASSETS_DIR", tmp_path)
    assert server.sign_images() == {}
