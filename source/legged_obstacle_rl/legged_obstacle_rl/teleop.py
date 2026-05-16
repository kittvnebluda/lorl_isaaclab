import threading
import time
from dataclasses import dataclass

import numpy as np
from evdev import InputDevice, categorize, ecodes, list_devices


@dataclass
class State:
    lin_x: float = 0.0
    lin_y: float = 0.0
    ang_z: float = 0.0
    height: float = 0.38
    stop: bool = False
    lock: bool = False


# Shared state instance
state = State()


def teleop_backend(state_obj):
    """Background worker that listens for keyboard events."""
    devices = [InputDevice(path) for path in list_devices()]
    dev = next((d for d in devices if "keyboard" in d.name.lower()), None)

    if not dev:
        print("Error: No keyboard found. Check permissions/group.")
        return

    key_map = {
        "KEY_I": ("lin_x", 0.1),
        "KEY_K": ("lin_x", -0.1),
        "KEY_J": ("lin_y", 0.1),
        "KEY_L": ("lin_y", -0.1),
        "KEY_U": ("ang_z", 0.1),
        "KEY_O": ("ang_z", -0.1),
        "KEY_Y": ("height", 0.01),
        "KEY_H": ("height", -0.01),
    }

    active_modifiers = set()

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            # evdev sometimes returns keycodes as a list; normalize to string
            key_name = key_event.keycode[0] if isinstance(key_event.keycode, list) else key_event.keycode
            keystate = key_event.keystate

            # Track modifier keys (Ctrl/Shift)
            if "CTRL" in key_name or "SHIFT" in key_name:
                if keystate in (1, 2):
                    active_modifiers.add(key_name)
                else:
                    active_modifiers.discard(key_name)

            # Quit condition
            if key_name in ("KEY_ESC", "KEY_8"):
                state_obj.stop = True
                break

            # Toggle Lock: Ctrl + L
            if key_name == "KEY_L" and any("CTRL" in m for m in active_modifiers) and keystate == 1:
                state_obj.lock = not state_obj.lock
                status = "LOCKED" if state_obj.lock else "UNLOCKED"
                # Overwrite line temporarily to show status clearly
                print(f"\r[TELEOP] {status}{' ' * 30}", end="", flush=True)
                continue  # Skip movement processing

            # Ignore movement/height commands if locked
            if state_obj.lock:
                continue

            # Process movement keys
            if keystate in (1, 2) and key_name in key_map:
                attr, val = key_map[key_name]
                new_val = getattr(state_obj, attr) + val
                setattr(state_obj, attr, new_val)

                # Clamp values
                state_obj.lin_x = np.clip(state_obj.lin_x, -1.5, 1.5)
                state_obj.lin_y = np.clip(state_obj.lin_y, -1.5, 1.5)
                state_obj.ang_z = np.clip(state_obj.ang_z, -1.5, 1.5)
                state_obj.height = np.clip(state_obj.height, 0.1, 0.5)


def start_teleop_thread():
    teleop_thread = threading.Thread(target=teleop_backend, args=(state,), daemon=True)
    teleop_thread.start()


def print_commands():
    lock_indicator = " [LOCKED]" if state.lock else ""
    print(
        f"\r[COMMANDS] VX: {state.lin_x: 1.2f} | VY: {state.lin_y: 1.2f} | WZ: {state.ang_z: 1.2f} | H: {state.height: 0.2f}{lock_indicator}   ",
        end="",
        flush=True,
    )


def main():
    start_teleop_thread()
    print("Teleop thread started.")
    print("  Move: I/K (Vx), J/L (Vy), U/O (Wz), Y/H (Height)")
    print("  Toggle Lock: Ctrl+L | Quit: ESC")

    try:
        while not state.stop:
            print_commands()
            time.sleep(0.02)
    except KeyboardInterrupt:
        state.stop = True

    print("\nRobot control stopped.")


if __name__ == "__main__":
    main()
