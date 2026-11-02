"""Which clip stores the review window opens for each flag combination."""
from __future__ import annotations

from core.aeb.clip_store import contributed_clip_root, default_clip_root
from tools.aeb_review_remote import reviewer_clip_root, review_stores


def _roots(stores) -> list:
    return [s.root for s in stores]


def test_default_merges_local_and_contributed():
    stores = review_stores(root=None, contributed_only=False)
    assert _roots(stores) == [default_clip_root(), contributed_clip_root()]


def test_own_alone_is_local_only():
    stores = review_stores(root=None, contributed_only=False, own=True)
    assert _roots(stores) == [default_clip_root()]


def test_own_with_reviewer_is_local_only():
    stores = review_stores(root=None, contributed_only=False, reviewer=True, own=True)
    assert _roots(stores) == [default_clip_root()]


def test_reviewer_alone_is_batch_only():
    stores = review_stores(root=None, contributed_only=False, reviewer=True)
    assert _roots(stores) == [reviewer_clip_root()]
