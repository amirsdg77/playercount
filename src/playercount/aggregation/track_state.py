"""Sliding-window per-track team-vote bookkeeping.

The team classifier emits a per-frame guess for each track id. Single-frame
guesses are noisy (motion blur, occlusion, the player turning their back to the
camera), so we stabilize them with a sliding window of recent votes and report
the mode. That single change kills almost all visible flicker in the counts.

Design choices:

* **Owned by exactly one task.** :class:`TrackState` is created and consumed
  inside one ``aggregate_stage`` coroutine — there is no need for locking.
  If we ever scale to multiple aggregator workers (we won't, but the comment
  is here), this class would need a ``threading.Lock``.
* **Bounded memory per track.** A ``deque`` with ``maxlen=window`` caps the
  per-track storage, so a long match never grows the votes structure
  unboundedly.
* **Forget unseen tracks.** :meth:`forget` is called by the aggregator when
  the tracker drops an id; that prevents a slow leak across very long
  streams.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable


class TrackState:
    """Sliding-window team-vote storage with mode-based stabilization.

    ``vote(track_id, team_id)`` records one observation. ``majority(track_id)``
    returns the most-voted team in the window, with three tie-break rules
    (in order):

    1. ``None`` votes (referee / ball / pre-calibration) are excluded from the
       mode count but **do** still count toward the window length, so a track
       with all-``None`` history correctly returns ``None``.
    2. On ties between two team labels, the *latest* vote wins — this matches
       intuition that recent observations are more authoritative when the
       window is otherwise split.
    3. Empty history → ``None``.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._votes: dict[int, deque[int | None]] = {}

    # -- mutation ------------------------------------------------------------

    def vote(self, track_id: int, team_id: int | None) -> None:
        """Record one observation for ``track_id``."""
        bucket = self._votes.get(track_id)
        if bucket is None:
            bucket = deque(maxlen=self._window)
            self._votes[track_id] = bucket
        bucket.append(team_id)

    def vote_many(self, observations: Iterable[tuple[int, int | None]]) -> None:
        """Convenience: record many ``(track_id, team_id)`` tuples in one call."""
        for tid, team in observations:
            self.vote(tid, team)

    def forget(self, track_id: int) -> None:
        """Drop a track's history. Idempotent."""
        self._votes.pop(track_id, None)

    def forget_many(self, track_ids: Iterable[int]) -> None:
        """Drop a batch of track ids. Idempotent."""
        for tid in track_ids:
            self.forget(tid)

    # -- queries -------------------------------------------------------------

    def majority(self, track_id: int) -> int | None:
        """Return the stabilized team label for ``track_id``.

        Implements tie-breaks documented on the class.
        """
        bucket = self._votes.get(track_id)
        if not bucket:
            return None
        counts = Counter(v for v in bucket if v is not None)
        if not counts:
            return None
        top_count = max(counts.values())
        winners = [team for team, c in counts.items() if c == top_count]
        if len(winners) == 1:
            return winners[0]
        # Tie-break: latest occurrence wins.
        latest_index_by_team: dict[int, int] = {}
        for idx, team in enumerate(bucket):
            if team in winners:
                latest_index_by_team[team] = idx
        return max(latest_index_by_team, key=latest_index_by_team.__getitem__)

    def active_ids(self) -> set[int]:
        """All track ids currently held in state (whether labelled or not)."""
        return set(self._votes.keys())

    def __len__(self) -> int:  # number of distinct tracks tracked
        return len(self._votes)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._votes

    def window(self) -> int:
        return self._window


__all__ = ["TrackState"]
