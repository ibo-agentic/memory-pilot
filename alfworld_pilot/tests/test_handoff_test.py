"""Tests for handoff_test.py's fail-loudly precondition check -- the part
that can be verified without touching Kaggle (copy_in/copy_out_and_version
themselves inherit kaggle_session.py's own documented "untested outside a
real Kaggle session" limitation; see README's "Handoff test" section)."""

from __future__ import annotations

import pytest

from alfworld_pilot.handoff_test import check_session2_precondition


def test_session1_with_no_prior_jobs_is_fine():
    check_session2_precondition(set(), session=1)  # should not raise


def test_session1_with_some_prior_jobs_is_fine():
    # Unusual but not an error for session 1 -- e.g. a retried session 1.
    check_session2_precondition({"log_0"}, session=1)  # should not raise


def test_session2_with_no_prior_jobs_fails_loudly():
    with pytest.raises(SystemExit, match="FAIL"):
        check_session2_precondition(set(), session=2)


def test_session2_with_prior_jobs_is_fine():
    check_session2_precondition({"log_0", "log_1"}, session=2)  # should not raise
