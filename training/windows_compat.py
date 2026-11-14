"""Let ``lerobot-train`` keep its ``checkpoints/last`` link on Windows without privileges.

After every save LeRobot points ``checkpoints/last`` at the newest checkpoint
with a symbolic link. Windows allows that only with Developer Mode enabled
through its own setting (not merely by policy) or in an elevated shell;
otherwise ``os.symlink`` raises WinError 1314 and the training run stops at its
first checkpoint. A directory junction needs no privilege and resolves the same
way for every path LeRobot opens through ``last``, so this falls back to one.

Behaviour is unchanged wherever the symlink succeeds. The one difference: a
junction stores an absolute target, so ``last`` breaks if the run folder is
moved; the numbered checkpoints themselves are unaffected.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_PRIVILEGE_NOT_HELD = 1314


def _remove_link(link: Path) -> None:
    """Remove a symlink or junction without touching what it points to."""
    if link.is_symlink():
        link.unlink()
    elif link.is_junction():
        link.rmdir()


def _with_junction_fallback(original: Callable[[Path], object]) -> Callable[[Path], object]:
    def update_last_checkpoint(checkpoint_dir: Path) -> object:
        last = checkpoint_dir.parent / "last"
        _remove_link(last)
        try:
            return original(checkpoint_dir)
        except OSError as exc:
            if getattr(exc, "winerror", None) != _PRIVILEGE_NOT_HELD:
                raise
        import _winapi

        _winapi.CreateJunction(str(checkpoint_dir.resolve()), str(last))
        return None

    return update_last_checkpoint


def install() -> None:
    """Patch LeRobot's ``update_last_checkpoint`` in every module that bound it."""
    if sys.platform != "win32":
        return
    from lerobot.common import train_utils
    from lerobot.scripts import lerobot_train

    patched = _with_junction_fallback(train_utils.update_last_checkpoint)
    train_utils.update_last_checkpoint = patched
    lerobot_train.update_last_checkpoint = patched
