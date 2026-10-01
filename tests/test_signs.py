from signtrainer import config, server


def test_sign_images_by_label(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIGN_ASSETS_DIR", tmp_path)
    for name in ("A.png", "b.gif", "C.JPG", "notes.txt", "D.webp"):
        (tmp_path / name).write_bytes(b"x")
    assert server.sign_images() == {"A": "A.png", "B": "b.gif", "C": "C.JPG", "D": "D.webp"}


def test_sign_images_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIGN_ASSETS_DIR", tmp_path)
    assert server.sign_images() == {}


def test_trim_to_content_crops_transparent_border():
    import cv2
    import numpy as np

    from signtrainer.images import trim_to_content

    img = np.zeros((200, 100, 4), np.uint8)  # fully transparent
    img[50:70, 20:60] = (0, 0, 0, 255)  # a black 40x20 drawing
    ok, png = cv2.imencode(".png", img)
    out = cv2.imdecode(np.frombuffer(trim_to_content(png.tobytes()), np.uint8), cv2.IMREAD_UNCHANGED)
    h, w = out.shape[:2]
    assert 40 <= w <= 46 and 20 <= h <= 26  # drawing plus a small margin


def test_trim_to_content_leaves_garbage_alone():
    from signtrainer.images import trim_to_content

    assert trim_to_content(b"not an image") == b"not an image"
