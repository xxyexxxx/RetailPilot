"""Content-addressed storage and explicitly allowlisted HTTPS imports."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import re
import socket
import ssl
import tempfile
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .asset_config import asset_settings as config


class AssetStore(Protocol):
    def put(self, data: bytes) -> str: ...
    def read(self, key: str) -> bytes: ...


class LocalAssetStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, key: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("invalid_storage_key")
        path = self.root / key[:2] / key
        if path.resolve() != path:
            raise ValueError("invalid_storage_path")
        return path

    def put(self, data: bytes) -> str:
        key = hashlib.sha256(data).hexdigest()
        path = self.path(key)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temp = Path(output.name)
                output.write(data)
            try:
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        return key

    def read(self, key: str) -> bytes:
        return self.path(key).read_bytes()


def store() -> LocalAssetStore:
    return LocalAssetStore(config.store)


def detect_mime(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported_file_type")


def download_image(url: str) -> tuple[bytes, str]:
    """Reject redirects and connect to the exact public IP checked by DNS."""
    hosts = {
        item.strip().lower() for item in config.remote_hosts.split(",") if item.strip()
    }
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or host not in hosts
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
    ):
        raise ValueError("remote_host_not_allowed")
    addresses = sorted(
        {r[4][0] for r in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    )
    ips = [ipaddress.ip_address(address) for address in addresses]
    if not ips or any(
        not ip.is_global or ip.is_multicast or ip.is_reserved for ip in ips
    ):
        raise ValueError("remote_address_not_allowed")
    deadline = time.monotonic() + 15

    class PinnedHTTPS(http.client.HTTPSConnection):
        def connect(self):
            conn = socket.create_connection((addresses[0], 443), timeout=5)
            try:
                self.sock = ssl.create_default_context().wrap_socket(
                    conn, server_hostname=host
                )
            except BaseException:
                conn.close()
                raise

    connection = PinnedHTTPS(host, timeout=5)
    try:
        connection.request(
            "GET",
            (parsed.path or "/") + ("?" + parsed.query if parsed.query else ""),
            headers={"Accept": "image/png, image/jpeg, image/webp"},
        )
        response = connection.getresponse()
        mime = response.headers.get_content_type()
        length = response.headers.get("Content-Length")
        if response.status != 200 or mime not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise ValueError("remote_image_rejected")
        if length and int(length) > config.max_bytes:
            raise ValueError("file_too_large")
        data = bytearray()
        while block := response.read1(min(65536, config.max_bytes + 1 - len(data))):
            data.extend(block)
            if len(data) > config.max_bytes or time.monotonic() > deadline:
                raise ValueError("remote_image_limit")
        if detect_mime(data) != mime:
            raise ValueError("mime_mismatch")
        return bytes(data), mime
    finally:
        connection.close()
