"""
Test — the build_notebooks CLI (plan_2026-10-10 C3).

Every committed notebook embeds the code it was built from, which is the
provenance of its kernel's results; rebuilding one that already ran rewrites
that record. Before C3, main() rebuilt every notebook on any invocation,
--help included. These tests point NB_DIR at a temporary directory, so the
committed notebooks are never touched.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

import benchmarks.build_notebooks as B

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def nb_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "NB_DIR", tmp_path)
    return tmp_path


def _written(d):
    return sorted(p.name for p in d.iterdir())


def test_no_arguments_writes_nothing(nb_dir) -> None:
    assert B.main([]) == 0
    assert _written(nb_dir) == []


def test_help_writes_nothing(nb_dir) -> None:
    with pytest.raises(SystemExit) as e:
        B.main(["--help"])
    assert e.value.code == 0
    assert _written(nb_dir) == []


def test_list_writes_nothing_and_names_every_notebook(nb_dir, capsys) -> None:
    assert B.main(["--list"]) == 0
    assert _written(nb_dir) == []
    names = capsys.readouterr().out.split()
    assert "k64_headline120_seed0" in names and "cuda_diag" in names
    # every name is the stem of a committed notebook, so --only is predictable
    for n in names:
        assert (ROOT / "notebooks" / ("kaggle_%s.ipynb" % n)).exists(), n


def test_only_writes_exactly_the_named_notebooks(nb_dir) -> None:
    assert B.main(["--only", "cuda_diag", "width_profile"]) == 0
    written = [n for n in _written(nb_dir) if n.endswith(".ipynb")]
    assert written == ["kaggle_cuda_diag.ipynb", "kaggle_width_profile.ipynb"]


def test_unknown_name_writes_nothing(nb_dir) -> None:
    with pytest.raises(SystemExit) as e:
        B.main(["--only", "cuda_diag", "no_such_notebook"])
    assert e.value.code == 2
    assert _written(nb_dir) == []
