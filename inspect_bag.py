#!/usr/bin/env python3
"""Simple inspector for a ROS1 bag file.

Usage:
    python inspect_bag.py /path/to/bagfile.bag

Prints topics, message types, message counts, and the first
timestamp per topic.
"""

import sys
from pathlib import Path

try:
    import rosbag  # type: ignore
except ImportError:
    print("rosbag Python module not available. Install ROS1 rosbag.")
    sys.exit(1)


def inspect_bag(bag_path: Path) -> None:
    if not bag_path.exists():
        print(f"Bag file not found: {bag_path}")
        sys.exit(1)

    try:
        bag = rosbag.Bag(str(bag_path), mode="r")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to open bag: {exc}")
        sys.exit(1)

    print(f"Bag: {bag_path}")
    print(f"Duration: {bag.duration:.3f} s")
    print(f"Start time: {bag.get_start_time():.3f}")
    print(f"End time:   {bag.get_end_time():.3f}")
    print()
    print("Topics:")

    info = bag.get_type_and_topic_info().topics
    for topic, topic_info in info.items():
        msg_type = topic_info.msg_type
        count = topic_info.message_count
        freq = topic_info.frequency
        first_time = None
        for _, _, t in bag.read_messages(topic):
            first_time = t.to_sec()
            break
        first_str = f"{first_time:.3f}" if first_time is not None else "n/a"
        freq_str = f"{freq:.3f}" if freq is not None else "n/a"
        print(
            f"  - {topic}\n"
            f"      type: {msg_type}\n"
            f"      count: {count}\n"
            f"      frequency: {freq_str} Hz\n"
            f"      first_stamp: {first_str}"
        )

    bag.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_bag.py path/to/file.bag")
        sys.exit(1)
    inspect_bag(Path(sys.argv[1]))
{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}