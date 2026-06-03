from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from legged_obstacle_rl.teleop import TeleopState


@runtime_checkable
class TeleopCommand(Protocol):
    def inject_teleop(self, state: TeleopState) -> None: ...
