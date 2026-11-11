import numpy as np
import pytest

from rtsc2 import agent


def test_policies_pick_distinct_rows_and_respect_rules():
    rng = np.random.default_rng(0)
    mean = np.linspace(1, 0, 20)
    members = np.vstack([mean, mean[::-1]])
    groups = np.asarray(["a"] * 10 + ["b"] * 5 + ["c"] * 5)
    for policy in agent.POLICIES:
        picks = agent.choose(policy, mean, members, groups, 6, rng)
        assert len(picks) == len(set(picks.tolist())) == 6
    assert agent.choose("greedy", mean, members, groups, 3, rng).tolist() == [0, 1, 2]
    diverse = agent.choose("diverse", mean, members, groups, 6, rng, per_system=2)
    assert max(np.unique(groups[diverse], return_counts=True)[1]) == 2
    # too few systems to fill the batch: fall back to the best remaining rows
    assert len(agent.choose("diverse", mean, members, np.asarray(["a"] * 20), 6, rng)) == 6
    with pytest.raises(ValueError):
        agent.choose("nope", mean, members, groups, 3, rng)


def test_ucb_prefers_disagreement_when_means_tie():
    mean = np.asarray([0.5, 0.5, 0.5])
    members = np.asarray([[0.5, 0.1, 0.4], [0.5, 0.9, 0.6]])
    assert agent.choose("ucb", mean, members, np.asarray(list("abc")), 1, np.random.default_rng(0))[0] == 1


def test_replay_guided_search_beats_random(toy):
    target = toy.label.astype(int)
    summary = agent.compare_policies(toy, target, policies=("random", "greedy"), seeds=3, n_jobs=1,
                                     initial=40, batch=10, rounds=5)
    assert summary["policies"]["greedy"]["hits_mean"] > summary["policies"]["random"]["hits_mean"]
    assert summary["policies"]["greedy"]["enrichment_over_random"] > 1.0
    run = agent.replay(toy, target, "greedy", seed=1, initial=40, batch=10, rounds=5)
    assert run["queries"] == 50 and run["hits"] == sum(run["hits_per_round"]) <= 50
    assert 0 <= run["new_system_hits"] <= run["hits"]


def test_replay_is_deterministic(toy):
    kwargs = dict(seed=2, initial=40, batch=10, rounds=3)
    assert agent.replay(toy, toy.label, "thompson", **kwargs) == agent.replay(toy, toy.label, "thompson", **kwargs)


def test_propose_only_returns_untested_eligible_rows(toy):
    from dataclasses import replace

    from rtsc2.models import TreeBag

    label = toy.label.copy()
    label[:100] = np.nan  # first 100 rows are "untested"
    mixed = replace(toy, label=label)
    known = mixed.subset(mixed.labelled)
    model = TreeBag(members=2, max_iter=20, early_stopping=False).fit(known.X, known.label, groups=known.groups)
    eligible = np.arange(len(label)) % 2 == 0
    table = agent.propose(mixed, model, batch=10, eligible=eligible)
    rows = [int(i.split("-")[1]) for i in table["mp_id"]]
    assert len(table) == 10 and all(r < 100 and r % 2 == 0 for r in rows)
    assert table["rank"].tolist() == list(range(1, 11))
