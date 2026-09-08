#!/usr/bin/env python3
"""Copy the newest matching file from the newest remote directory."""

from __future__ import annotations

import fnmatch
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_sim.configs import RemoteFileSyncConfig


_RESULT_PREFIX = "REMOTE_FILE_SYNC_RESULT="
_REMOTE_LIST_SCRIPT = f"""
import json
import os
import sys

result_prefix = {_RESULT_PREFIX!r}
entry_type, directory = sys.argv[1:]
entries = []
with os.scandir(directory) as iterator:
    for entry in iterator:
        if entry_type == "directory":
            include = entry.is_dir(follow_symlinks=False)
        elif entry_type == "file":
            include = entry.is_file(follow_symlinks=False)
        else:
            raise ValueError(f"unsupported entry type: {{entry_type}}")
        if include:
            stat_result = entry.stat(follow_symlinks=False)
            entries.append(
                {{"name": entry.name, "path": entry.path, "mtime_ns": stat_result.st_mtime_ns}}
            )
print(result_prefix + json.dumps(entries, ensure_ascii=False))
""".strip()


class RemoteFileSyncError(RuntimeError):
    """Raised when the remote file cannot be selected or downloaded."""


@dataclass(frozen=True)
class SftpLocation:
    """OpenSSH connection target parsed from an SFTP URL."""

    host: str
    path: str
    username: str | None = None
    port: int | None = None

    @property
    def ssh_target(self) -> str:
        if self.username is None:
            return self.host
        return f"{self.username}@{self.host}"


@dataclass(frozen=True)
class RemoteEntry:
    """Metadata needed to choose one remote directory or file."""

    name: str
    path: str
    mtime_ns: int


def parse_sftp_url(remote_url: str) -> SftpLocation:
    """Parse and validate an absolute SFTP URL."""
    try:
        parsed = urlsplit(remote_url)
        port = parsed.port
    except ValueError as exc:
        raise RemoteFileSyncError(f"Invalid SFTP URL {remote_url!r}: {exc}") from exc

    if parsed.scheme.lower() != "sftp":
        raise RemoteFileSyncError(
            f"Invalid remote URL scheme {parsed.scheme!r}; expected 'sftp'"
        )
    if parsed.hostname is None:
        raise RemoteFileSyncError("Invalid SFTP URL: host is missing")
    if parsed.password is not None:
        raise RemoteFileSyncError(
            "Passwords in SFTP URLs are not supported; use OpenSSH authentication"
        )
    if parsed.query or parsed.fragment:
        raise RemoteFileSyncError("SFTP URL must not contain a query or fragment")

    remote_path = unquote(parsed.path)
    if not remote_path.startswith("/"):
        raise RemoteFileSyncError("SFTP URL path must be absolute")
    if "\x00" in remote_path:
        raise RemoteFileSyncError("SFTP URL path must not contain a NUL byte")

    username = unquote(parsed.username) if parsed.username is not None else None
    return SftpLocation(
        host=parsed.hostname,
        path=remote_path,
        username=username,
        port=port,
    )


def _ssh_command(location: SftpLocation, remote_command: str) -> list[str]:
    command = ["ssh", "-o", "BatchMode=yes"]
    if location.port is not None:
        command.extend(("-p", str(location.port)))
    command.extend((location.ssh_target, remote_command))
    return command


def _decode_remote_entries(stdout: str) -> list[RemoteEntry]:
    payload_line = next(
        (line for line in reversed(stdout.splitlines()) if line.startswith(_RESULT_PREFIX)),
        None,
    )
    if payload_line is None:
        raise RemoteFileSyncError("Remote scan did not return a metadata result")

    try:
        payload: Any = json.loads(payload_line[len(_RESULT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise RemoteFileSyncError(f"Remote scan returned invalid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise RemoteFileSyncError("Remote scan metadata must be a list")

    entries: list[RemoteEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RemoteFileSyncError("Remote scan metadata contains a non-object entry")
        try:
            name = item["name"]
            path = item["path"]
            mtime_ns = item["mtime_ns"]
        except KeyError as exc:
            raise RemoteFileSyncError(
                f"Remote scan metadata is missing field {exc.args[0]!r}"
            ) from exc
        if not isinstance(name, str) or not isinstance(path, str):
            raise RemoteFileSyncError("Remote entry name and path must be strings")
        if isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int):
            raise RemoteFileSyncError("Remote entry mtime_ns must be an integer")
        entries.append(RemoteEntry(name=name, path=path, mtime_ns=mtime_ns))
    return entries


def _list_remote_entries(
    location: SftpLocation,
    directory: str,
    *,
    entry_type: str,
) -> list[RemoteEntry]:
    remote_command = shlex.join(
        ("python3", "-c", _REMOTE_LIST_SCRIPT, entry_type, directory)
    )
    try:
        result = subprocess.run(
            _ssh_command(location, remote_command),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RemoteFileSyncError(f"Failed to start SSH: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise RemoteFileSyncError(
            f"Remote {entry_type} scan failed with exit code {result.returncode}: {detail}"
        )
    return _decode_remote_entries(result.stdout)


def _select_latest(entries: list[RemoteEntry]) -> RemoteEntry | None:
    if not entries:
        return None
    return max(entries, key=lambda entry: (entry.mtime_ns, entry.name))


def _quote_sftp_path(path: str) -> str:
    if "\n" in path or "\r" in path or "\x00" in path:
        raise RemoteFileSyncError(
            "SFTP paths containing newline, carriage return, or NUL are not supported"
        )
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_sftp_get(
    location: SftpLocation,
    remote_path: str,
    local_path: Path,
) -> None:
    command = ["sftp", "-b", "-"]
    if location.port is not None:
        command.extend(("-P", str(location.port)))
    command.append(location.ssh_target)
    batch = f"get {_quote_sftp_path(remote_path)} {_quote_sftp_path(str(local_path))}\n"

    try:
        result = subprocess.run(
            command,
            input=batch,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RemoteFileSyncError(f"Failed to start SFTP: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise RemoteFileSyncError(
            f"SFTP download failed with exit code {result.returncode}: {detail}"
        )


def _download_without_overwrite(
    location: SftpLocation,
    remote_path: str,
    destination: Path,
) -> bool:
    """Download through a temporary file and publish only if absent."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        _run_sftp_get(location, remote_path, temporary_path)
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            return False
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def sync_latest_remote_file(config: RemoteFileSyncConfig) -> bool:
    """Synchronize one file and return whether a new local file was copied."""
    local_dir = Path(config.local_dir)
    if not local_dir.is_dir():
        raise RemoteFileSyncError(
            f"Local destination directory does not exist or is not a directory: {local_dir}"
        )

    location = parse_sftp_url(config.remote_url)
    directories = _list_remote_entries(
        location,
        location.path,
        entry_type="directory",
    )
    latest_directory = _select_latest(directories)
    if latest_directory is None:
        raise RemoteFileSyncError(
            f"No subdirectories found in remote path: {config.remote_url}"
        )
    print(f"[remote_file_sync] Latest remote directory: {latest_directory.path}")

    files = _list_remote_entries(
        location,
        latest_directory.path,
        entry_type="file",
    )
    matching_files = [
        entry for entry in files if fnmatch.fnmatchcase(entry.name, config.file_pattern)
    ]
    latest_file = _select_latest(matching_files)
    if latest_file is None:
        raise RemoteFileSyncError(
            f"No files matching {config.file_pattern!r} found in: {latest_directory.path}"
        )
    print(f"[remote_file_sync] Latest matching file: {latest_file.path}")

    destination = local_dir / latest_file.name
    if os.path.lexists(destination):
        print(f"[remote_file_sync] Skipped existing local file: {destination}")
        return False

    copied = _download_without_overwrite(location, latest_file.path, destination)
    if copied:
        print(f"[remote_file_sync] Copied to: {destination}")
    else:
        print(f"[remote_file_sync] Skipped existing local file: {destination}")
    return copied


def main() -> int:
    try:
        sync_latest_remote_file(RemoteFileSyncConfig())
    except RemoteFileSyncError as exc:
        print(f"[remote_file_sync] Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
