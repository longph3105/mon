"""Tests for ``mon.core.constants``."""

import mon.core.constants as c


def test_extension_sets_are_lowercase_dotted():
    for s in (c.IMAGE_EXTENSIONS, c.RAW_IMAGE_EXTENSIONS, c.VIDEO_EXTENSIONS, c.WEIGHTS_EXTENSIONS, c.CONFIG_EXTENSIONS):
        assert all(e.startswith(".") and e == e.lower() for e in s)


def test_defaults_belong_to_their_sets():
    assert c.DEFAULT_IMAGE_EXT in c.IMAGE_EXTENSIONS
    assert c.DEFAULT_WEIGHTS_EXT in c.WEIGHTS_EXTENSIONS
    assert c.DEFAULT_CKPT_EXT in c.WEIGHTS_EXTENSIONS
    assert c.DEFAULT_LABEL_EXT in c.CONFIG_EXTENSIONS


def test_all_exports_exist():
    assert all(hasattr(c, name) for name in c.__all__)
    assert c.__all__ == sorted(c.__all__)
