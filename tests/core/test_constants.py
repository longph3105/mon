"""Tests for ``mon.core.constants``."""

import pytest

import mon.core
import mon.core.constants as c

DEPRECATED = ("ANNS_DIR", "DEBUG_DIR", "DEPTH_DIR", "IMAGES_DIR", "LABELS_DIR",
              "DEFAULT_CKPT_EXT", "DEFAULT_IMAGE_EXT", "DEFAULT_LABEL_EXT", "DEFAULT_WEIGHTS_EXT")


def test_extension_sets_are_lowercase_dotted():
    groups = (c.IMAGE_EXTENSIONS, c.RAW_IMAGE_EXTENSIONS, c.VIDEO_EXTENSIONS, c.WEIGHTS_EXTENSIONS, c.CONFIG_EXTENSIONS)
    for s in groups:
        assert all(e.startswith(".") and e == e.lower() for e in s)


def test_text_files_are_not_config_files():
    assert ".txt" not in c.CONFIG_EXTENSIONS  # e.g. classes.txt, YOLO labels


@pytest.mark.parametrize("name", DEPRECATED)
def test_deprecated_names_warn_from_the_module_and_the_namespace(name: str):
    assert name not in c.__all__ and name not in dir(mon.core)
    with pytest.warns(DeprecationWarning, match=f"{name} is deprecated .* removed in 3.1"):
        value = getattr(c, name)
    with pytest.warns(DeprecationWarning, match=name):
        assert getattr(mon.core, name) == value
    with pytest.raises(AttributeError):
        _ = c.NOT_A_CONSTANT


def test_all_exports_exist():
    assert all(hasattr(c, name) for name in c.__all__)
    assert c.__all__ == sorted(c.__all__)
