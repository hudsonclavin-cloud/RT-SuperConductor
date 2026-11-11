import json

import numpy as np

from rtsc2 import cli, data


def test_stage_is_reused_until_settings_change(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    calls = []

    def compute():
        calls.append(1)
        return {"value": len(calls)}

    first = cli.stage("demo", {"k": 1}, source, compute, tmp_path)
    again = cli.stage("demo", {"k": 1}, source, compute, tmp_path)
    assert first["value"] == again["value"] == 1 and len(calls) == 1
    assert cli.stage("demo", {"k": 2}, source, compute, tmp_path)["value"] == 2
    source.write_text("a\n2\n", encoding="utf-8")  # changed input invalidates the stage
    assert cli.stage("demo", {"k": 2}, source, compute, tmp_path)["value"] == 3
    assert cli.stage("demo", {"k": 2}, source, compute, tmp_path, force=True)["value"] == 4
    assert json.loads((tmp_path / "demo.json").read_text())["provenance"]["candidates_sha256"]


def test_eligibility_screen(toy):
    from dataclasses import replace

    n = len(toy.ids)
    frame = toy.frame.assign(chemical_family="Other", is_metal="True", is_magnetic="False", num_sites="4")
    frame.loc[0, "is_magnetic"] = "True"
    frame.loc[1, "num_sites"] = "40"
    frame.loc[2, "chemical_family"] = "Cuprates"
    index = toy.el_index.copy()
    index[3, 0] = 89  # actinium: no stable isotope
    mask = cli.eligible_mask(replace(toy, frame=frame, el_index=index))
    assert mask[:4].tolist() == [False] * 4 and mask[4:].all() and len(mask) == n
