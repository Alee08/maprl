"""Configuration helpers for Maze Office experiments.

This module centralises the reward-machine transition patches that are
applied to each experiment so that :mod:`maze_office_main` can stay free
from global mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class TransitionPatch:
    """Describe a patch that must be merged into a reward machine."""

    transitions: Dict
    position: str = "before"
    prefix: str = "new"


def build_experiment_patches(coordinates_obj, goals) -> Mapping[str, Mapping[str, TransitionPatch]]:
    """Return the reward-machine patches for each supported experiment.

    Parameters
    ----------
    coordinates_obj:
        The coordinates parsed from the office layout.
    goals:
        The goal locations extracted from the office layout.
    """

    # Task 2
    a2_task2_patch = TransitionPatch(
        transitions={
            ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
            ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
            ("state2", ((goals["B"], True),)): ("state3", 0),
        }
    )

    a5_task2_patch = TransitionPatch(
        transitions={
            ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
            ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
            ("state2", ((goals["C"], True),)): ("state3", 0),
        }
    )

    # Task 3
    a1_task3_patch = TransitionPatch(
        transitions={
            ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
            ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
            ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
            ("state3", ((goals["O"], True),)): ("state4", 0),
            ("state4", ((goals["C"], True),)): ("state5", 0),
        }
    )

    a2_task3_patch = TransitionPatch(
        transitions={
            ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
            ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
            ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
            ("state3", ((goals["B"], True),)): ("state4", 0),
            ("state4", ((goals["O"], True),)): ("state5", 0),
        }
    )

    a3_task3_patch = TransitionPatch(
        transitions={
            ("state1", ((goals["C"], True),)): ("state2", 0),
            ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
            ("state3", ((coordinates_obj["coffee"][0], True),)): ("state4", 0),
            ("state3", ((coordinates_obj["coffee"][1], True),)): ("state4", 0),
            ("state3", ((goals["O"], True),)): ("state5", 0),
        }
    )

    a4_task3_patch = TransitionPatch(
        transitions={
            ("state1", ((goals["O"], True),)): ("state2", 0),
            ("state2", ((coordinates_obj["coffee"][0], True),)): ("state3", 0),
            ("state2", ((coordinates_obj["coffee"][1], True),)): ("state3", 0),
            ("state3", ((coordinates_obj["letter"][0], True),)): ("state4", 0),
            ("state4", ((goals["B"], True),)): ("state5", 0),
        }
    )

    a5_task3_patch = TransitionPatch(
        transitions={
            ("state1", ((goals["C"], True),)): ("state2", 0),
            ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
            ("state3", ((coordinates_obj["coffee"][0], True),)): ("state4", 0),
            ("state3", ((coordinates_obj["coffee"][1], True),)): ("state4", 0),
            ("state4", ((goals["O"], True),)): ("state5", 0),
        }
    )

    return {
        "exp1": {},
        "exp2": {
            "a2": a2_task2_patch,
            "a5": a5_task2_patch,
        },
        "exp3": {
            "a1": a1_task3_patch,
            "a2": a2_task3_patch,
            "a3": a3_task3_patch,
            "a4": a4_task3_patch,
            "a5": a5_task3_patch,
        },
    }


__all__ = ["TransitionPatch", "build_experiment_patches"]
