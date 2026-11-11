"""
Build leakage-free hypergraphs for 5-fold DHGCDA evaluation.

For fold i, all association-derived views are rebuilt strictly from:

    Positive_Sample_Train{i}.csv

Rebuilt per fold:
    hy_mc       circRNA-side association hypergraph
    hy_cm       disease-side association hypergraph
    lg_m2       disease Jaccard similarity matrix
    hy_mm_top5  disease-similarity hypergraph

Unchanged from template:
    hy_cc_top5  circRNA sequence-similarity hypergraph
    lg_c2       circRNA sequence-similarity matrix
    hy_mi       external circRNA-miRNA hypergraph
    hy_do       external disease-ontology hypergraph
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_hyg import (  # noqa: E402
    build_disease_similarity,
    build_hy_mc_cm,
    build_hy_mm,
)


N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strictly train-only hypergraphs for 5-fold DHGCDA."
    )

    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing Positive_Sample_Train0.csv through "
            "Positive_Sample_Train4.csv."
        ),
    )

    parser.add_argument(
        "--template-hyg",
        type=Path,
        required=True,
        help=(
            "Template hyg_dict pickle. Only label-independent views are inherited."
        ),
    )

    parser.add_argument(
        "--disease-ids",
        type=Path,
        default=PROJECT_ROOT / "data" / "disease_ids_unique_313.txt",
        help="Disease ID list whose row order matches the model disease indices.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for hyg_dict0.pickle through hyg_dict4.pickle.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top-k value for the disease-similarity hypergraph.",
    )

    return parser.parse_args()


def read_disease_ids(path: Path) -> list[str]:
    """Read disease names/IDs in model index order."""
    disease_ids: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            disease_ids.append(line.split(",")[0].strip())

    if not disease_ids:
        raise ValueError(f"No disease IDs found in {path}")

    if len(disease_ids) != len(set(disease_ids)):
        raise ValueError(f"Duplicate disease IDs found in {path}")

    return disease_ids


def read_positive_train(
    path: Path,
    disease_to_index: dict[str, int],
    n_circrna: int,
) -> list[tuple[int, int]]:
    """
    Read positive training pairs.

    Expected CSV format:
        circRNA_index,disease_id

    circRNA_index in the CSV is 1-based and is converted to 0-based.
    Disease indices follow disease_ids_unique_313.txt.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training split not found: {path}")

    pairs: set[tuple[int, int]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)

        for line_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue

            circ_value = row[0].strip()
            disease_id = row[1].strip()

            try:
                circ_index = int(circ_value) - 1
            except ValueError as exc:
                raise ValueError(
                    f"Invalid circRNA index at {path}:{line_number}: "
                    f"{circ_value!r}"
                ) from exc

            if not 0 <= circ_index < n_circrna:
                raise ValueError(
                    f"circRNA index out of range at {path}:{line_number}: "
                    f"{circ_value}; expected 1..{n_circrna}"
                )

            if disease_id not in disease_to_index:
                raise KeyError(
                    f"Unknown disease ID at {path}:{line_number}: "
                    f"{disease_id!r}"
                )

            disease_index = disease_to_index[disease_id]
            pairs.add((circ_index, disease_index))

    if not pairs:
        raise ValueError(f"No positive training pairs found in {path}")

    return sorted(pairs)


def load_template(path: Path) -> dict:
    """Load and minimally validate the hypergraph template."""
    if not path.exists():
        raise FileNotFoundError(f"Template hypergraph not found: {path}")

    with path.open("rb") as handle:
        template = pickle.load(handle)

    if not isinstance(template, dict):
        raise TypeError("Template hypergraph must be a dictionary.")

    required_keys = {"circ_ids", "disease_ids"}
    missing = required_keys - set(template)

    if missing:
        raise KeyError(
            f"Template is missing required keys: {sorted(missing)}"
        )

    return template


def build_fold_hypergraph(
    template: dict,
    train_pairs: list[tuple[int, int]],
    fold_index: int,
    top_n: int,
) -> dict:
    """
    Rebuild every association-derived view from one fold's training positives.
    """
    n_circrna = len(template["circ_ids"])
    n_disease = len(template["disease_ids"])

    # circRNA-disease association hypergraphs.
    hy_mc, hy_cm = build_hy_mc_cm(
        n_circrna,
        n_disease,
        train_pairs,
    )

    # Disease similarity computed only from training positive associations.
    disease_similarity = build_disease_similarity(
        n_disease,
        train_pairs,
    ).astype(np.float32)

    # Disease-similarity hypergraph computed from train-only similarity.
    hy_mm = build_hy_mm(
        disease_similarity,
        topk=top_n,
    )

    # Shallow copy is sufficient because only the following entries are replaced.
    fold_hyg = dict(template)

    fold_hyg["hy_mc"] = hy_mc
    fold_hyg["hy_cm"] = hy_cm
    fold_hyg["lg_m2"] = disease_similarity
    fold_hyg["hy_mm_top5"] = hy_mm

    # Audit information.
    fold_hyg["cv_fold"] = fold_index + 1
    fold_hyg["association_source"] = (
        f"Positive_Sample_Train{fold_index}.csv only"
    )
    fold_hyg["leakage_free"] = True
    fold_hyg["train_positive_count"] = len(train_pairs)
    fold_hyg["disease_similarity_topk"] = top_n

    return fold_hyg


def main() -> None:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than zero.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    disease_ids = read_disease_ids(args.disease_ids)
    disease_to_index = {
        disease_id: index
        for index, disease_id in enumerate(disease_ids)
    }

    template = load_template(args.template_hyg)

    n_circrna = len(template["circ_ids"])
    n_disease = len(template["disease_ids"])

    if len(disease_ids) != n_disease:
        raise ValueError(
            "Disease count mismatch: "
            f"disease ID file has {len(disease_ids)}, "
            f"but template has {n_disease}."
        )

    print("Building leakage-free 5-fold hypergraphs")
    print(f"  circRNAs: {n_circrna}")
    print(f"  diseases: {n_disease}")
    print(f"  top-k:    {args.top_n}")

    for fold_index in range(N_FOLDS):
        train_path = (
            args.samples_dir
            / f"Positive_Sample_Train{fold_index}.csv"
        )

        train_pairs = read_positive_train(
            train_path,
            disease_to_index,
            n_circrna,
        )

        fold_hyg = build_fold_hypergraph(
            template=template,
            train_pairs=train_pairs,
            fold_index=fold_index,
            top_n=args.top_n,
        )

        output_path = (
            args.output_dir
            / f"hyg_dict{fold_index}.pickle"
        )

        with output_path.open("wb") as handle:
            pickle.dump(
                fold_hyg,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"Fold {fold_index + 1}: "
            f"{len(train_pairs)} training positives -> {output_path}"
        )

    print("Finished. Generated exactly 5 fold-specific hypergraphs.")


if __name__ == "__main__":
    main()
