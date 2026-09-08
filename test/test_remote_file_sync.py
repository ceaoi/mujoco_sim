import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import mujoco_sim.configs as sync_configs
from mujoco_sim.configs import PROJECT_ROOT, RemoteFileSyncConfig
from mujoco_sim.utils import pull_file as remote_sync


def _scan_result(entries):
    payload = json.dumps(entries)
    return subprocess.CompletedProcess(
        args=["ssh"],
        returncode=0,
        stdout=f"banner\n{remote_sync._RESULT_PREFIX}{payload}\n",
        stderr="",
    )


def test_remote_file_sync_config_defaults():
    config = RemoteFileSyncConfig()

    assert config.remote_url == (
        "sftp://l40/data/chensj/wl_lab2/logs/rsl_rl/zb02w_teacher"
    )
    assert config.file_pattern == "model_*.pt"
    assert config.local_dir.is_absolute()
    assert config.local_dir.is_relative_to(PROJECT_ROOT)


def test_parse_sftp_url_supports_username_port_and_decoded_path():
    location = remote_sync.parse_sftp_url(
        "sftp://alice@example.test:2222/data/model%20logs"
    )

    assert location.host == "example.test"
    assert location.port == 2222
    assert location.username == "alice"
    assert location.ssh_target == "alice@example.test"
    assert location.path == "/data/model logs"


@pytest.mark.parametrize(
    "remote_url",
    (
        "https://host/path",
        "sftp:///path",
        "sftp://user:secret@host/path",
        "sftp://host/path?query=yes",
    ),
)
def test_parse_sftp_url_rejects_invalid_or_unsafe_urls(remote_url):
    with pytest.raises(remote_sync.RemoteFileSyncError):
        remote_sync.parse_sftp_url(remote_url)


def test_select_latest_uses_mtime_then_name():
    entries = [
        remote_sync.RemoteEntry("run_a", "/logs/run_a", 10),
        remote_sync.RemoteEntry("run_b", "/logs/run_b", 20),
        remote_sync.RemoteEntry("run_c", "/logs/run_c", 20),
    ]

    assert remote_sync._select_latest(entries) == entries[2]
    assert remote_sync._select_latest([]) is None


def test_file_pattern_filters_before_selecting_latest(tmp_path, monkeypatch):
    responses = [
        [remote_sync.RemoteEntry("run", "/logs/run", 10)],
        [
            remote_sync.RemoteEntry("events.log", "/logs/run/events.log", 100),
            remote_sync.RemoteEntry("model_20.pt", "/logs/run/model_20.pt", 20),
            remote_sync.RemoteEntry("model_40.pt", "/logs/run/model_40.pt", 40),
        ],
    ]
    monkeypatch.setattr(
        remote_sync,
        "_list_remote_entries",
        lambda *args, **kwargs: responses.pop(0),
    )
    selected_paths = []
    monkeypatch.setattr(
        remote_sync,
        "_download_without_overwrite",
        lambda location, remote_path, destination: selected_paths.append(remote_path)
        or True,
    )

    assert remote_sync.sync_latest_remote_file(
        RemoteFileSyncConfig(local_dir=tmp_path)
    )
    assert selected_paths == ["/logs/run/model_40.pt"]


def test_list_remote_entries_decodes_metadata_and_uses_ssh_config(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _scan_result(
            [{"name": "run", "path": "/logs/run", "mtime_ns": 123}]
        )

    monkeypatch.setattr(remote_sync.subprocess, "run", fake_run)
    location = remote_sync.SftpLocation("l40", "/logs", port=2222)

    entries = remote_sync._list_remote_entries(
        location,
        "/logs",
        entry_type="directory",
    )

    assert entries == [remote_sync.RemoteEntry("run", "/logs/run", 123)]
    assert calls[0][0][:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-p",
        "2222",
        "l40",
    ]


def test_remote_list_script_lists_only_requested_entry_type(tmp_path):
    model_path = tmp_path / "model_10.pt"
    model_path.write_bytes(b"checkpoint")
    (tmp_path / "run").mkdir()

    result = subprocess.run(
        [sys.executable, "-c", remote_sync._REMOTE_LIST_SCRIPT, "file", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    entries = remote_sync._decode_remote_entries(result.stdout)
    assert len(entries) == 1
    assert entries[0].name == model_path.name
    assert entries[0].path == str(model_path)


def test_remote_scan_failure_reports_remote_error(monkeypatch):
    monkeypatch.setattr(
        remote_sync.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["ssh"], returncode=255, stdout="", stderr="permission denied"
        ),
    )

    with pytest.raises(remote_sync.RemoteFileSyncError, match="permission denied"):
        remote_sync._list_remote_entries(
            remote_sync.SftpLocation("l40", "/logs"),
            "/logs",
            entry_type="directory",
        )


def test_sftp_failure_reports_download_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        remote_sync.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["sftp"], returncode=1, stdout="", stderr="remote open failed"
        ),
    )

    with pytest.raises(remote_sync.RemoteFileSyncError, match="remote open failed"):
        remote_sync._run_sftp_get(
            remote_sync.SftpLocation("l40", "/logs"),
            "/logs/run/model.pt",
            tmp_path / "model.pt",
        )


def test_missing_local_directory_stops_before_remote_access(tmp_path, monkeypatch):
    missing_directory = tmp_path / "missing"
    monkeypatch.setattr(
        remote_sync,
        "_list_remote_entries",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("remote access must not occur")
        ),
    )

    with pytest.raises(remote_sync.RemoteFileSyncError, match="does not exist"):
        remote_sync.sync_latest_remote_file(
            RemoteFileSyncConfig(local_dir=missing_directory)
        )
    assert not missing_directory.exists()


def test_no_remote_directory_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_sync, "_list_remote_entries", lambda *args, **kwargs: [])

    with pytest.raises(remote_sync.RemoteFileSyncError, match="No subdirectories"):
        remote_sync.sync_latest_remote_file(RemoteFileSyncConfig(local_dir=tmp_path))


def test_no_matching_file_is_reported(tmp_path, monkeypatch, capsys):
    responses = [
        [remote_sync.RemoteEntry("run", "/logs/run", 10)],
        [],
    ]
    monkeypatch.setattr(
        remote_sync,
        "_list_remote_entries",
        lambda *args, **kwargs: responses.pop(0),
    )

    with pytest.raises(remote_sync.RemoteFileSyncError, match="No files matching"):
        remote_sync.sync_latest_remote_file(RemoteFileSyncConfig(local_dir=tmp_path))
    assert "Latest remote directory: /logs/run" in capsys.readouterr().out


def test_existing_destination_is_skipped_without_download(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "model_20.pt"
    destination.write_bytes(b"existing")
    responses = [
        [remote_sync.RemoteEntry("run", "/logs/run", 10)],
        [remote_sync.RemoteEntry("model_20.pt", "/logs/run/model_20.pt", 20)],
    ]
    monkeypatch.setattr(
        remote_sync,
        "_list_remote_entries",
        lambda *args, **kwargs: responses.pop(0),
    )
    monkeypatch.setattr(
        remote_sync,
        "_download_without_overwrite",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download must not occur")
        ),
    )

    copied = remote_sync.sync_latest_remote_file(
        RemoteFileSyncConfig(local_dir=tmp_path)
    )

    assert not copied
    assert destination.read_bytes() == b"existing"
    assert "Skipped existing local file" in capsys.readouterr().out


def test_download_publishes_original_name_and_removes_temporary_file(
    tmp_path, monkeypatch
):
    location = remote_sync.SftpLocation("l40", "/logs")
    destination = tmp_path / "model_40.pt"

    def fake_sftp_get(actual_location, remote_path, local_path):
        assert actual_location == location
        assert remote_path == "/logs/run/model_40.pt"
        local_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(remote_sync, "_run_sftp_get", fake_sftp_get)

    assert remote_sync._download_without_overwrite(
        location,
        "/logs/run/model_40.pt",
        destination,
    )
    assert destination.read_bytes() == b"checkpoint"
    assert list(tmp_path.iterdir()) == [destination]


def test_download_does_not_overwrite_file_created_during_transfer(
    tmp_path, monkeypatch
):
    location = remote_sync.SftpLocation("l40", "/logs")
    destination = tmp_path / "model_60.pt"

    def fake_sftp_get(actual_location, remote_path, local_path):
        local_path.write_bytes(b"downloaded")
        destination.write_bytes(b"raced")

    monkeypatch.setattr(remote_sync, "_run_sftp_get", fake_sftp_get)

    assert not remote_sync._download_without_overwrite(
        location,
        "/logs/run/model_60.pt",
        destination,
    )
    assert destination.read_bytes() == b"raced"
    assert all(not path.name.endswith(".part") for path in tmp_path.iterdir())


def test_main_returns_nonzero_and_prints_error(monkeypatch, capsys):
    monkeypatch.setattr(
        remote_sync,
        "sync_latest_remote_file",
        lambda config: (_ for _ in ()).throw(remote_sync.RemoteFileSyncError("boom")),
    )

    assert remote_sync.main() == 1
    assert "[remote_file_sync] Error: boom" in capsys.readouterr().err


def test_direct_pull_file_entry_point_runs_without_import_errors(tmp_path, monkeypatch):
    entry_point = Path(remote_sync.__file__).with_name("pull_file.py")
    missing_directory = tmp_path / "missing"
    monkeypatch.setattr(
        sync_configs,
        "RemoteFileSyncConfig",
        lambda: RemoteFileSyncConfig(local_dir=missing_directory),
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(entry_point), run_name="__main__")

    assert exc_info.value.code == 1
    assert not missing_directory.exists()
