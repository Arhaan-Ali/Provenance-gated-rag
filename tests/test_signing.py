"""The signed decisions trail: forged, edited, deleted or rolled-back files are refused."""
import json
import os
import stat

import pytest

from pgrag.core.registry import (
    DecisionsError,
    ReviewDecision,
    TrustTier,
    load_latest_decisions,
    save_decisions,
)
from pgrag.core.signing import KEY_DIR_ENV, KEY_NAME, key_dir, read_head, sign, write_head


def _decision(tier=TrustTier.B, excluded=False, name="wiki"):
    return ReviewDecision(name, tier, excluded, False, "r", files={"a.txt": "1" * 64})


def _read(path):
    return json.loads(path.read_text())


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2))


# --- the happy path -------------------------------------------------------

def test_a_saved_file_is_signed_and_loads(tmp_path):
    path = save_decisions([_decision()], tmp_path)

    assert len(_read(path)["signature"]) == 64
    assert load_latest_decisions(tmp_path)["wiki"].tier == TrustTier.B


def test_each_version_records_the_signature_of_the_one_before(tmp_path):
    v1 = save_decisions([_decision()], tmp_path)
    v2 = save_decisions([_decision(TrustTier.C)], tmp_path)

    assert _read(v1)["previous_signature"] is None
    assert _read(v2)["previous_signature"] == _read(v1)["signature"]
    assert load_latest_decisions(tmp_path)["wiki"].tier == TrustTier.C


def test_the_key_is_created_owner_only_and_outside_the_decisions_folder(tmp_path):
    save_decisions([_decision()], tmp_path)

    key = key_dir() / KEY_NAME
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    assert tmp_path not in key.parents


def test_the_newest_signed_version_is_recorded_next_to_the_key(tmp_path):
    save_decisions([_decision()], tmp_path)
    v2 = save_decisions([_decision()], tmp_path)

    assert read_head(tmp_path) == {"version": 2, "signature": _read(v2)["signature"]}


# --- forging --------------------------------------------------------------

def test_a_forged_unsigned_newer_version_is_refused(tmp_path):
    save_decisions([_decision()], tmp_path)
    forged = _read(tmp_path / "decisions_v1.json")
    forged.update(version=9999, decisions=[{**forged["decisions"][0], "tier": "A", "overridden": True}])
    del forged["signature"]
    _write(tmp_path / "decisions_v9999.json", forged)

    with pytest.raises(DecisionsError, match="not signed"):
        load_latest_decisions(tmp_path)


def test_a_forged_version_with_a_made_up_signature_is_refused(tmp_path):
    v1 = save_decisions([_decision()], tmp_path)
    forged = {**_read(v1), "version": 2, "previous_signature": _read(v1)["signature"],
              "signature": "0" * 64}
    _write(tmp_path / "decisions_v2.json", forged)

    with pytest.raises(DecisionsError, match="signature does not match"):
        load_latest_decisions(tmp_path)


def test_a_version_signed_with_someone_elses_key_is_refused(tmp_path):
    v1 = save_decisions([_decision()], tmp_path)
    forged = {**_read(v1), "version": 2, "previous_signature": _read(v1)["signature"]}
    forged["signature"] = sign(forged, b"attacker key".ljust(32, b"\0"))
    _write(tmp_path / "decisions_v2.json", forged)

    with pytest.raises(DecisionsError, match="signature does not match"):
        load_latest_decisions(tmp_path)


def test_editing_the_newest_file_is_refused(tmp_path):
    path = save_decisions([_decision(TrustTier.D)], tmp_path)
    payload = _read(path)
    payload["decisions"][0]["tier"] = "A"
    _write(path, payload)

    with pytest.raises(DecisionsError, match="signature does not match"):
        load_latest_decisions(tmp_path)


def test_editing_an_older_version_breaks_the_chain(tmp_path):
    v1 = save_decisions([_decision()], tmp_path)
    save_decisions([_decision()], tmp_path)
    payload = _read(v1)
    payload["decisions"][0]["excluded"] = True
    _write(v1, payload)

    with pytest.raises(DecisionsError, match="signature does not match"):
        load_latest_decisions(tmp_path)


def test_a_signed_file_with_a_renamed_version_is_refused(tmp_path):
    save_decisions([_decision()], tmp_path)
    (tmp_path / "decisions_v1.json").rename(tmp_path / "decisions_v7.json")

    with pytest.raises(DecisionsError, match="file name says 7"):
        load_latest_decisions(tmp_path)


# --- deleting and rolling back --------------------------------------------

def test_deleting_the_newest_file_to_undo_a_revocation_is_refused(tmp_path):
    save_decisions([_decision(TrustTier.A)], tmp_path)
    v2 = save_decisions([_decision(excluded=True)], tmp_path)
    v2.unlink()

    with pytest.raises(DecisionsError, match="older decision was put back"):
        load_latest_decisions(tmp_path)


def test_deleting_a_version_in_the_middle_is_refused(tmp_path):
    save_decisions([_decision()], tmp_path)
    save_decisions([_decision()], tmp_path)
    save_decisions([_decision()], tmp_path)
    (tmp_path / "decisions_v2.json").unlink()

    with pytest.raises(DecisionsError, match="decisions_v2.json is missing"):
        load_latest_decisions(tmp_path)


def test_deleting_every_file_is_refused(tmp_path):
    path = save_decisions([_decision()], tmp_path)
    path.unlink()

    with pytest.raises(DecisionsError, match="were deleted"):
        load_latest_decisions(tmp_path)


def test_saving_refuses_to_extend_a_tampered_trail(tmp_path):
    path = save_decisions([_decision()], tmp_path)
    payload = _read(path)
    payload["decisions"][0]["tier"] = "A"
    _write(path, payload)

    with pytest.raises(DecisionsError):
        save_decisions([_decision()], tmp_path)
    assert not (tmp_path / "decisions_v2.json").exists()


def test_a_crash_between_writing_a_file_and_the_head_still_loads(tmp_path):
    v1 = save_decisions([_decision()], tmp_path)
    save_decisions([_decision(TrustTier.C)], tmp_path)
    write_head(tmp_path, 1, _read(v1)["signature"])   # as if v2's head write never happened

    assert load_latest_decisions(tmp_path)["wiki"].tier == TrustTier.C
    assert read_head(tmp_path)["version"] == 2


# --- the key --------------------------------------------------------------

def test_signed_files_without_the_key_are_refused(tmp_path, monkeypatch, tmp_path_factory):
    save_decisions([_decision()], tmp_path)
    monkeypatch.setenv(KEY_DIR_ENV, str(tmp_path_factory.mktemp("elsewhere")))

    with pytest.raises(DecisionsError, match="no key"):
        load_latest_decisions(tmp_path)
    with pytest.raises(DecisionsError, match="no key"):
        save_decisions([_decision()], tmp_path)   # and no new key is made to cover it up
    assert not (key_dir() / KEY_NAME).exists()


# --- files from before signing --------------------------------------------

def _unsigned_v1(tmp_path):
    _write(tmp_path / "decisions_v1.json", {
        "version": 1,
        "decided_at": "2026-06-01T10:00:00+00:00",
        "decisions": [{"origin_name": "wiki", "tier": "A", "excluded": False,
                       "overridden": True, "reason": "r"}],
    })


def test_unsigned_files_from_before_signing_are_not_trusted(tmp_path):
    _unsigned_v1(tmp_path)

    assert load_latest_decisions(tmp_path) == {}   # every origin is reviewed again


def test_the_first_signed_save_after_unsigned_files_starts_a_new_chain(tmp_path):
    _unsigned_v1(tmp_path)

    v2 = save_decisions([_decision(TrustTier.C)], tmp_path)

    assert v2.name == "decisions_v2.json"
    assert _read(v2)["previous_signature"] is None
    assert load_latest_decisions(tmp_path)["wiki"].tier == TrustTier.C
