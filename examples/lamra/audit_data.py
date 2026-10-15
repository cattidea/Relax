# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Audit every possible training image/positive before a complete M-BEIR FT."""

import argparse
import json
from pathlib import Path

from relax.models.lamra.dataset import resolve_media_path
from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import sha256_file


logger = get_logger(__name__)


def audit(query: Path, candidates: Path, instructions: Path, media_roots: list[Path]) -> dict:
    pool = {}
    with candidates.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["did"] in pool:
                raise ValueError("Duplicate M-BEIR candidate ID")
            pool[row["did"]] = row.get("img_path")
    images: set[str] = set()
    missing_positives = 0
    query_count = 0
    with query.open() as stream:
        for line in stream:
            row = json.loads(line)
            query_count += 1
            if row.get("query_img_path"):
                images.add(row["query_img_path"])
            positives = row.get("pos_cand_list", [])
            if not positives:
                missing_positives += 1
            for did in positives:
                if did not in pool:
                    missing_positives += 1
                elif pool[did]:
                    images.add(pool[did])
    missing = []
    for image in sorted(images):
        try:
            resolve_media_path(image, media_roots)
        except FileNotFoundError:
            missing.append(image)
    return {
        "status": "passed" if not missing and not missing_positives else "failed",
        "query_count": query_count,
        "candidate_count": len(pool),
        "referenced_images": len(images),
        "missing_images": len(missing),
        "missing_image_examples": missing[:20],
        "missing_positives": missing_positives,
        "input_sha256": {
            name: sha256_file(path)
            for name, path in {
                "query": query,
                "candidates": candidates,
                "instructions": instructions,
            }.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.query, args.candidates, args.instructions, args.media_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    logger.info(
        f"M-BEIR audit: {result['status']}, queries={result['query_count']}, missing={result['missing_images']}"
    )
    if result["status"] != "passed":
        raise SystemExit("M-BEIR audit failed; no missing samples may be filtered out")


if __name__ == "__main__":
    main()
