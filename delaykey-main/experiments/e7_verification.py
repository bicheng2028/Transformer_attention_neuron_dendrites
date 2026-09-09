"""E7: is somatic integration a weighted sum, or a conjunctive check?

The manuscript asserts throughout that the soma performs "cross-source
verification, not weighted averaging", but the somatic equation it writes down
is a weighted sum with a threshold, and nothing in the paper tests the claim
(Gap 6). In Experiment 1 the proximal pathway carries a fixed subthreshold
constant, so it carries no content at all and the distinction cannot even be
posed.

Comparing matched against mismatched content is *not* the discriminating test,
which is worth stating because it is the obvious thing to run. Matched content
delivers more proximal excitation than mismatched content, so a plain weighted
sum separates those two cells perfectly well; observing the separation tells you
only that the content pathway is connected.

The test that does discriminate is the ordering against a third condition, no
content at all. For any somatic rule that sums non-negative drives,

    y(mismatch) = w.V'_wrong + a.distal - I_inh  >=  w.0 + a.distal - I_inh
                = y(no content),

since a rectified agreement term cannot be negative. A weighted sum is therefore
*forbidden* from firing less often to wrong content than to absent content:
P(fire | mismatch) >= P(fire | absent) at every parameter setting. Verification
predicts the reverse, because a conflict is positive evidence against the
hypothesis rather than merely missing support for it.

The reported quantity is therefore the veto index,

    P(fire | content absent) - P(fire | content mismatched),

with the distal sequence present throughout. It is bounded above by zero for any
weighted sum and positive for a genuine veto, so its sign alone decides the
question.

This runs the 2x3 -- content {match, mismatch, absent} x distal
{present, absent} -- for two somatic rules that are identical except in how
I_inh is set:

*sum*         I_inh is a constant, so the soma is exactly the weighted sum of
              Eq. 8 as currently written. The constant is swept over its whole
              useful range, so the conclusion is not an artefact of one choice.
*conjunctive* I_inh is driven by the *disagreement* between the arriving content
              vector V' and the neuron's stored template, and is gated by the
              presence of proximal input -- no content, no conflict, hence no
              inhibition. Nothing is added to the architecture: Eq. 8 already
              carries an I_inh term, and this only says what sets it.

Content is a unit vector in R^d rather than a scalar; agreement is the cosine
similarity between the arriving V' and the neuron's stored template, rectified
so that an anticorrelated content is treated as absent rather than as negative
evidence. Rectification is what makes the bound above hold, and it is the
conservative choice: allowing negative drive would hand the weighted sum a veto
for free.

Usage::

    python experiments/e7_verification.py --quick
    python experiments/e7_verification.py --seeds 20
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.core import NeuronParams, simulate_population  # noqa: E402
from delaykey.stimuli import random_connectivity, two_event_stimulus  # noqa: E402

FIELDNAMES = [
    "rule", "inhibition", "content_dim", "dispersion", "theta", "seed",
    "n_detectors",
    "fire_match_distal", "fire_mismatch_distal", "fire_absent_distal",
    "fire_match_nodistal", "fire_mismatch_nodistal", "fire_absent_nodistal",
    "margin", "veto_index",
]


def content_agreement(v_prime: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Rectified cosine similarity between one content vector and each template.

    Returns a value in [0, 1] per neuron. Rectification means an anticorrelated
    content counts as absent rather than as evidence against, which keeps the
    weighted-sum control a sum of non-negative drives -- the case the argument
    is about.
    """
    v = v_prime / (np.linalg.norm(v_prime) + 1e-12)
    t = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-12)
    return np.maximum(t @ v, 0.0)


def run_cell(job: dict) -> list[dict]:
    """Run one (rule, inhibition, seed) cell of the 2x2."""
    rng = np.random.default_rng(job["seed"])
    params = NeuronParams(theta=job["theta"], alpha=job["alpha"],
                          plateau_steps=job["plateau_steps"],
                          refractory=job["refractory"], v_contrib=0.0)

    stim = two_event_stimulus(
        rng, n_features=job["n_features"], n_per_event=job["n_per_event"],
        onset1=job["onset1"], interval=job["interval"], spread=job["spread"],
        n_time=job["n_time"],
    )
    delays, weights = random_connectivity(
        rng, n_neurons=job["n_neurons"], n_features=job["n_features"],
        dispersion=job["dispersion"], n_branches=job["n_branches"],
        p_connect=job["p_connect"],
    )

    # Distal-present means the sequence the delays encode; distal-absent means
    # the same features and within-volley timing in the reversed order, so the
    # two conditions differ only in what the delay key is being asked to match.
    distal_present = stim.both
    distal_absent = stim.reversed_order().both

    # The neurons under test are those whose delays actually match the target:
    # asking about somatic integration is only meaningful where the distal vote
    # exists to be integrated.
    detectors = (simulate_population(distal_present, delays, weights, params) > 0)
    n_detectors = int(detectors.sum())
    if n_detectors == 0:
        return []
    delays_d = delays[detectors]
    weights_d = weights[detectors]

    d = job["content_dim"]
    templates = rng.normal(size=(n_detectors, d))
    matched = templates.copy()
    # A mismatched content is an independent draw, so its expected agreement
    # with the stored template is that of two random vectors in R^d, near zero.
    mismatched = rng.normal(size=(n_detectors, d))

    agree_match = np.array([content_agreement(matched[i], templates[i:i + 1])[0]
                            for i in range(n_detectors)])
    agree_mismatch = np.array([content_agreement(mismatched[i], templates[i:i + 1])[0]
                               for i in range(n_detectors)])

    w_v = job["w_v"]
    inhibition = job["inhibition"]
    zero = np.zeros(n_detectors)

    def somatic_terms(agreement, content_present: bool):
        """Proximal excitation and inhibition under the rule being tested.

        ``agreement`` is the rectified cosine in [0, 1]; ``content_present``
        distinguishes "the wrong content arrived" from "no content arrived",
        which is the contrast the whole experiment turns on.
        """
        drive = w_v * agreement if content_present else zero
        if job["rule"] == "sum":
            # Eq. 8 as written: I_inh is a constant, blind to content. Applied
            # in every condition, since a constant cannot know whether content
            # arrived.
            inhib = np.full(n_detectors, inhibition)
        elif content_present:
            # I_inh driven by disagreement: vanishes at full agreement, full
            # strength at zero agreement.
            inhib = inhibition * (1.0 - agreement)
        else:
            # No proximal input, so nothing drives the mismatch interneuron and
            # there is no conflict to signal.
            inhib = zero
        return drive, inhib

    out = {}
    for label, x, agreement, present in (
        ("match_distal", distal_present, agree_match, True),
        ("mismatch_distal", distal_present, agree_mismatch, True),
        ("absent_distal", distal_present, zero, False),
        ("match_nodistal", distal_absent, agree_match, True),
        ("mismatch_nodistal", distal_absent, agree_mismatch, True),
        ("absent_nodistal", distal_absent, zero, False),
    ):
        drive, inhib = somatic_terms(agreement, present)
        counts = simulate_population(
            x, delays_d, weights_d, params,
            proximal_drive=drive, proximal_inhibition=inhib)
        out[label] = float((counts > 0).mean())

    return [{
        "rule": job["rule"],
        "inhibition": inhibition,
        "content_dim": d,
        "dispersion": job["dispersion"],
        "theta": job["theta"],
        "seed": job["seed"],
        "n_detectors": n_detectors,
        "fire_match_distal": out["match_distal"],
        "fire_mismatch_distal": out["mismatch_distal"],
        "fire_absent_distal": out["absent_distal"],
        "fire_match_nodistal": out["match_nodistal"],
        "fire_mismatch_nodistal": out["mismatch_nodistal"],
        "fire_absent_nodistal": out["absent_nodistal"],
        "margin": out["match_distal"] - out["mismatch_distal"],
        # Positive only if wrong content silences more than absent content.
        # Bounded above by zero for any weighted sum of non-negative drives.
        "veto_index": out["absent_distal"] - out["mismatch_distal"],
    }]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("results/e7_verification.csv"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--n-neurons", type=int, default=2000)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--n-per-event", type=int, default=40)
    parser.add_argument("--content-dim", type=int, default=64)
    parser.add_argument("--w-v", type=float, default=0.6)
    parser.add_argument("--theta", type=float, default=2.5)
    parser.add_argument("--dispersion", type=int, default=20)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args(argv)

    seeds = range(args.seed_base, args.seed_base + (2 if args.quick else args.seeds))
    # Swept over the whole useful range: 0 is no inhibition, and by 1.2 the
    # conjunctive rule silences even matched content. If the weighted sum could
    # separate the two cells at any setting, it would show up here.
    inhibitions = ([0.0, 0.6] if args.quick
                   else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    rules = ["sum", "conjunctive"]

    jobs = [{
        "rule": rule, "inhibition": inh, "seed": seed,
        "content_dim": args.content_dim, "w_v": args.w_v,
        "theta": args.theta, "alpha": 2.5, "plateau_steps": 15, "refractory": 2,
        "n_neurons": args.n_neurons, "n_features": args.n_features,
        "n_per_event": args.n_per_event, "n_branches": 1, "p_connect": 0.1,
        "onset1": 5, "interval": args.interval, "spread": 3, "n_time": 60,
        "dispersion": args.dispersion,
    } for rule, inh, seed in itertools.product(rules, inhibitions, seeds)]

    n_jobs = args.jobs or max(1, min(mp.cpu_count() - 2, len(jobs)))
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(run_cell, jobs)
    else:
        results = [run_cell(j) for j in jobs]
    rows = [r for group in results for r in group]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}  ({len(rows)} rows)")

    print(f"\n{'rule':>12} {'I_inh':>6} {'match':>7} {'absent':>7} "
          f"{'mismatch':>9} {'veto':>7}")
    for rule in rules:
        for inh in inhibitions:
            sel = [r for r in rows if r["rule"] == rule and r["inhibition"] == inh]
            if not sel:
                continue
            m = np.mean([r["fire_match_distal"] for r in sel])
            ab = np.mean([r["fire_absent_distal"] for r in sel])
            mm = np.mean([r["fire_mismatch_distal"] for r in sel])
            print(f"{rule:>12} {inh:>6.1f} {m:>7.3f} {ab:>7.3f} "
                  f"{mm:>9.3f} {ab - mm:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
