"""Zel Beta - Ingestion Module

Ingestion takes software from multiple sources (GitHub via git, local directory, zip file),
places it into the Zel managed repository (`Repo`), and creates the initial Git commit
(the baseline version) from which iterative transformations and evaluations begin.
"""

import argparse
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from init import clear_directory_contents, init_repo

SourceType = Literal["git", "dir", "zip", "code"]


def _remove_readonly(
    func: Callable[..., object],
    path: str,
    _: BaseException,
) -> None:
    """Clear read-only bit (common on Windows) and retry removal."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_workspace(repo_path: Path) -> None:
    """Delete all files and directories in `repo_path` except the `.git` directory."""
    if not repo_path.exists():
        return
    for item in repo_path.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item, onexc=_remove_readonly)
        else:
            try:
                item.unlink()
            except PermissionError:
                os.chmod(item, stat.S_IWRITE)
                item.unlink()


def detect_source_type(source: str | Path) -> SourceType:
    """Automatically detect the source type based on URL patterns, paths, or file extension."""
    source_str = str(source).strip()

    # Git remote URL detection
    if (
        source_str.startswith(("http://", "https://", "git@", "ssh://"))
        or source_str.endswith(".git")
    ):
        return "git"

    path = Path(source_str).resolve()
    if path.is_file():
        if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
            return "zip"
        raise ValueError(
            f"File '{path}' is not a valid zip archive. Expected a .zip file."
        )

    if path.is_dir():
        return "dir"

    raise ValueError(
        f"Unable to determine source type for '{source_str}'. "
        "Must be a Git URL (https://...), a local directory path, or a .zip file."
    )


def _copy_contents_excluding_git(src_dir: Path, dest_dir: Path) -> int:
    """Copy all items from src_dir to dest_dir, skipping any .git folder.
    
    Returns the total number of files copied.
    """
    file_count = 0
    for root, dirs, files in os.walk(src_dir):
        # Prevent traversing into .git directories
        if ".git" in dirs:
            dirs.remove(".git")

        rel_root = Path(root).relative_to(src_dir)
        target_dir = dest_dir / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = Path(root) / file
            dest_file = target_dir / file
            shutil.copy2(src_file, dest_file)
            file_count += 1

    return file_count


def _ingest_from_git(
    url: str,
    target_repo: Path,
    branch: str | None = None,
) -> int:
    """Clone remote Git repository to a temporary directory and copy files into target_repo."""
    print(f"[*] Ingesting from Git repository: {url}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        clone_cmd = ["git", "clone", "--depth", "1"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([url, temp_dir])

        print(f"    Running: {' '.join(clone_cmd)}")
        subprocess.run(clone_cmd, check=True)

        file_count = _copy_contents_excluding_git(Path(temp_dir), target_repo)
        return file_count


def _ingest_from_dir(source_dir: Path, target_repo: Path) -> int:
    """Copy files from a local directory into target_repo."""
    resolved_dir = source_dir.resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {resolved_dir}")

    print(f"[*] Ingesting from local directory: {resolved_dir}")
    file_count = _copy_contents_excluding_git(resolved_dir, target_repo)
    return file_count


def _ingest_from_zip(zip_path: Path, target_repo: Path) -> int:
    """Extract a zip file to a temporary location and copy contents into target_repo."""
    resolved_zip = zip_path.resolve()
    if not resolved_zip.is_file():
        raise FileNotFoundError(f"Zip archive does not exist: {resolved_zip}")

    print(f"[*] Ingesting from Zip archive: {resolved_zip}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(resolved_zip, "r") as zf:
            zf.extractall(temp_path)

        # Check if the zip unpacks to a single root folder (common for GitHub downloads)
        extracted_items = list(temp_path.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            content_root = extracted_items[0]
        else:
            content_root = temp_path

        file_count = _copy_contents_excluding_git(content_root, target_repo)
        return file_count


def _ensure_git_user_config(repo_path: Path) -> None:
    """Ensure user.email and user.name are configured locally in repo."""
    try:
        subprocess.run(["git", "config", "user.email"], cwd=repo_path, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "config", "user.email", "zel-baseline@internal.sim"], cwd=repo_path, check=True)
    try:
        subprocess.run(["git", "config", "user.name"], cwd=repo_path, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "config", "user.name", "Zel Baseline Agent"], cwd=repo_path, check=True)


def commit_initial_version(
    repo_path: Path,
    message: str = "Initial version: Baseline software ingested",
    tag: str | None = "v0",
) -> str:
    """Stage all files in repo_path and make the initial Git commit and optional tag."""
    _ensure_git_user_config(repo_path)
    # Stage all files
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)

    # Check git status
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    if not status.stdout.strip():
        raise RuntimeError("No files were detected to commit. The source appears empty.")

    # Commit initial baseline version
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True)

    # Get the commit SHA
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_sha = rev.stdout.strip()

    # Create baseline tag if requested
    if tag:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", f"Baseline tag for initial commit {commit_sha[:7]}"],
            cwd=repo_path,
            check=True,
        )

    return commit_sha


def ingest(
    source: str | Path | dict[str, str],
    source_type: SourceType | None = None,
    repo_name: str = "Simulation/Repo",
    refresh_repo: bool = True,
    branch: str | None = None,
    commit_message: str | None = None,
    tag: str | None = "v0",
) -> dict[str, object]:
    """Ingest software from git, local directory, zip, or dict into the Zel repository."""
    base_dir = Path(__file__).resolve().parent
    repo_path = (base_dir / repo_name).resolve()

    if isinstance(source, dict):
        resolved_type = "code"
    else:
        resolved_type = source_type or detect_source_type(source)

    # Prepare repository
    if refresh_repo or not (repo_path / ".git").exists():
        print(f"[*] Initializing fresh repository at '{repo_path}' via init_repo()...")
        init_repo(relative_repo_path=repo_name)
    else:
        print(f"[*] Clearing previous workspace files in '{repo_path}' (preserving .git)...")
        clear_workspace(repo_path)

    # Ingest based on source type
    if resolved_type == "git":
        file_count = _ingest_from_git(str(source), repo_path, branch=branch)
    elif resolved_type == "dir":
        file_count = _ingest_from_dir(Path(source), repo_path)
    elif resolved_type == "zip":
        file_count = _ingest_from_zip(Path(source), repo_path)
    elif resolved_type == "code":
        file_count = 0
        if isinstance(source, dict):
            for rel_path, content in source.items():
                dest_file = repo_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                dest_file.write_text(content, encoding="utf-8")
                file_count += 1
    else:
        raise ValueError(f"Unsupported source type: {resolved_type}")

    print(f"[+] Successfully copied {file_count} files into '{repo_path}'.")

    # Create initial commit
    default_msg = f"Initial version: Ingested software baseline from {source} ({resolved_type})"
    msg = commit_message or default_msg
    commit_sha = commit_initial_version(repo_path, message=msg, tag=tag)

    print(f"[+] Initial version created!")
    print(f"    Commit: {commit_sha}")
    if tag:
        print(f"    Tag:    {tag}")

    return {
        "source": str(source),
        "source_type": resolved_type,
        "commit_sha": commit_sha,
        "tag": tag,
        "file_count": file_count,
        "repo_path": repo_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zel Ingestion: Ingest software from Git, local directory, or Zip archive into Repo."
    )
    parser.add_argument(
        "source",
        help="Path to local directory, .zip file, or Git clone URL.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="source_type",
        choices=["git", "dir", "zip"],
        default=None,
        help="Source type (optional; auto-detected if omitted).",
    )
    parser.add_argument(
        "-r",
        "--repo",
        dest="repo_name",
        default="Repo",
        help="Target repository folder name (default: 'Repo').",
    )
    parser.add_argument(
        "--no-refresh",
        dest="refresh_repo",
        action="store_false",
        default=True,
        help="Preserve existing Git repository structure instead of re-initializing via init.py.",
    )
    parser.add_argument(
        "-b",
        "--branch",
        default=None,
        help="Git branch to clone (only applies to git sources).",
    )
    parser.add_argument(
        "-m",
        "--message",
        dest="commit_message",
        default=None,
        help="Custom commit message for the baseline version.",
    )
    parser.add_argument(
        "--tag",
        default="v0",
        help="Git tag name for initial version (default: 'v0', pass '' to disable).",
    )

    args = parser.parse_args()
    tag = args.tag if args.tag != "" else None

    ingest(
        source=args.source,
        source_type=args.source_type,
        repo_name=args.repo_name,
        refresh_repo=args.refresh_repo,
        branch=args.branch,
        commit_message=args.commit_message,
        tag=tag,
    )


if __name__ == "__main__":
    main()
