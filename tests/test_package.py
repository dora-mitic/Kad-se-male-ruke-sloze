from signtrainer import __version__, config


def test_version_is_set():
    assert __version__


def test_paths_point_inside_repo():
    for path in (config.DATA_DIR, config.MODELS_DIR, config.REPORTS_DIR, config.ASSETS_DIR):
        assert path.parent == config.ROOT


def test_default_language_is_asl():
    assert config.LANGUAGE == "asl"
