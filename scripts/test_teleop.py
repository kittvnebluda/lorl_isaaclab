import time
from contextlib import suppress

from legged_obstacle_rl import teleop


def main():
    teleop.start()
    print("Teleop thread started.")
    print("  Move: I/K (Vx), J/L (Vy), U/O (Wz), Y/H (Height)")
    print("  Toggle Lock: Ctrl+L | Quit: ESC")

    with suppress(KeyboardInterrupt):
        while not teleop.state.stop:
            teleop.print_commands()
            time.sleep(0.02)

    print("\nRobot control stopped.")


if __name__ == "__main__":
    main()
