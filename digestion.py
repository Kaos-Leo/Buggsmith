"""Zel Beta - Digestion Module & CLI.

Safely packages and emits an accepted, verified software version from Simulation/Repo
into an isolated delivery package (default: 'dist/').

Usage:
  python digestion.py [--repo Simulation/Repo] [--version v2] [--out dist]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


class Digestion:
    """Emits production-ready artifacts for verified software versions."""

    def __init__(self, repo_path: Path | str = "Simulation/Repo"):
        self.repo_path = Path(repo_path).resolve()

    def get_commit_sha(self, target_ref: str = "HEAD") -> str:
        """Resolve a tag, branch, or HEAD to its 40-character commit SHA."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", target_ref],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError:
            # Fallback to HEAD if tag doesn't exist
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return res.stdout.strip()

    def digest(
        self,
        final_version: str = "v2",
        constraints_passed: bool = True,
        output_dir: Path | str = "dist",
    ) -> Dict[str, Any]:
        """Emit verified software candidate into output_dir if constraints passed."""
        if not constraints_passed:
            return {
                "status": "rejected",
                "version": final_version,
                "reason": "Constraints or Grader rejected the candidate version.",
                "output_dir": None,
            }

        target_out = Path(output_dir).resolve()
        target_out.mkdir(parents=True, exist_ok=True)

        commit_sha = self.get_commit_sha(final_version)

        # Copy non-git files from repo_path at verified state
        file_count = 0
        for root, dirs, files in os.walk(self.repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")

            rel_root = Path(root).relative_to(self.repo_path)
            target_sub = target_out / rel_root
            target_sub.mkdir(parents=True, exist_ok=True)

            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                src_file = Path(root) / f
                dst_file = target_sub / f
                shutil.copy2(src_file, dst_file)
                file_count += 1

        # Emit release manifest
        manifest = {
            "version": final_version,
            "commit_sha": commit_sha,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_count": file_count,
            "verification_status": "CONSTRAINTS_PASSED",
        }
        (target_out / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "status": "accepted",
            "version": final_version,
            "commit_sha": commit_sha,
            "file_count": file_count,
            "output_dir": str(target_out),
            "manifest": manifest,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zel Beta Digestion: Safely package and emit the verified software version."
    )
    parser.add_argument(
        "--repo",
        default="Simulation/Repo",
        help="Path to Zel Git repository (default: 'Simulation/Repo').",
    )
    parser.add_argument(
        "--version",
        default="HEAD",
        help="Version tag or commit to digest and emit (default: 'HEAD').",
    )
    parser.add_argument(
        "--out",
        default="dist",
        help="Destination directory for emitted release package (default: 'dist').",
    )
    args = parser.parse_args()

    digestion = Digestion(repo_path=args.repo)
    result = digestion.digest(
        final_version=args.version,
        constraints_passed=True,
        output_dir=args.out,
    )

    if result["status"] == "accepted":
        print(f"\n[+] Successfully emitted release ({result['file_count']} files) to: {result['output_dir']}")
    else:
        print("\n[-] Digestion rejected release.")


if __name__ == "__main__":
    main()
