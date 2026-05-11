"""Project-wide numeric constants.

Centralised so the same magic numbers are not duplicated across modules.
Anything that varies per deployment lives in :mod:`playercount.config`
instead — this file holds invariants of the *taxonomy* (class IDs, team
IDs) and *presentation* (annotator colours).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Detector class IDs (match :data:`playercount.schemas.CLASS_NAMES`)
# ---------------------------------------------------------------------------

CLS_PLAYER = 0
CLS_GOALKEEPER = 1
CLS_REFEREE = 2
CLS_BALL = 3


# ---------------------------------------------------------------------------
# Team IDs
# ---------------------------------------------------------------------------

TEAM_A = 0
TEAM_B = 1


# ---------------------------------------------------------------------------
# Annotator colours — BGR triples (OpenCV convention)
# ---------------------------------------------------------------------------

COLOR_TEAM_A = (0, 0, 255)         # red
COLOR_TEAM_B = (255, 0, 0)         # blue
COLOR_REFEREE = (255, 255, 255)    # white
COLOR_GK_OUTLINE = (0, 255, 255)   # yellow outline for goalkeepers
COLOR_UNASSIGNED = (200, 200, 200)  # light grey for tracks without a team label
COLOR_HUD_BG = (0, 0, 0)            # black HUD strip
COLOR_HUD_TEXT = (255, 255, 255)    # white HUD text


__all__ = [
    "CLS_BALL",
    "CLS_GOALKEEPER",
    "CLS_PLAYER",
    "CLS_REFEREE",
    "COLOR_GK_OUTLINE",
    "COLOR_HUD_BG",
    "COLOR_HUD_TEXT",
    "COLOR_REFEREE",
    "COLOR_TEAM_A",
    "COLOR_TEAM_B",
    "COLOR_UNASSIGNED",
    "TEAM_A",
    "TEAM_B",
]
