import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Callable


def _remove_readonly(func: Callable[[str], None], path: str, _: object) -> None:
    """Clear read-only bit (common in .git/objects on Windows) and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_directory_contents(directory: Path) -> None:
    """Delete all files, symlinks, and subdirectories inside `directory`."""
    for item in directory.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item, onexc=_remove_readonly)
        else:
            try:
                item.unlink()
            except PermissionError:
                os.chmod(item, stat.S_IWRITE)
                item.unlink()


def init_repo(relative_repo_path: str = "Simulation/Repo") -> Path:
    # Resolves to Zel-beta/Simulation/Repo regardless of current working directory
    base_dir = Path(__file__).resolve().parent
    repo_path = base_dir / relative_repo_path

    if repo_path.exists():
        print(f"Found existing '{repo_path}'. Cleaning internal contents...")
        clear_directory_contents(repo_path)
    else:
        print(f"Creating '{repo_path}'...")
        repo_path.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Git repository in '{repo_path}'...")
    subprocess.run(["git", "init"], cwd=repo_path, check=True)

    return repo_path


if __name__ == "__main__":
    init_repo()