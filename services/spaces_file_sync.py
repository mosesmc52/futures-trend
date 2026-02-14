#!/usr/bin/env python3
"""
spaces_file_sync.py

General file sync utility for DigitalOcean Spaces:
  - download: Spaces -> local file
  - upload:   local file -> Spaces (replace object)
  - sync:     download then upload

Examples:
  # Download object to local path
  python spaces_file_sync.py download --key db/portfolio.sqlite --path ./data/portfolio.sqlite

  # Upload local file to a specific key (replace)
  python spaces_file_sync.py upload --key db/portfolio.sqlite --path ./data/portfolio.sqlite

  # Upload local file to a different key
  python spaces_file_sync.py upload --key db/portfolio.sqlite --path ./data/portfolio.sqlite --dest-key db/portfolio_v2.sqlite

  # Upload using a prefix + auto key based on filename
  python spaces_file_sync.py upload --path ./data/portfolio.sqlite --prefix db/

  # Round trip sync
  python spaces_file_sync.py sync --key db/portfolio.sqlite --path ./data/portfolio.sqlite

Environment (preferred):
  SPACES_KEY, SPACES_SECRET, SPACES_BUCKET
  SPACES_REGION (default nyc3)
  SPACES_ENDPOINT (optional)
  SPACES_CDN_BASE (optional)

Notes:
- Uses SpacesClient.client (boto3 S3 client) for get_object/put_object/copy_object.
- For very large files, consider multipart upload.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from botocore.exceptions import ClientError
from dotenv import find_dotenv, load_dotenv
from spaces import SpacesClient  # your class

load_dotenv(find_dotenv())


# -----------------------------
# Config
# -----------------------------
@dataclass
class SyncConfig:
    key: Optional[str]  # source key (download) OR target key (upload)
    path: Path  # local file path
    dest_key: Optional[str] = None  # optional override for upload target key
    prefix: Optional[str] = None  # optional upload prefix (if key not provided)
    acl: str = "private"  # "private" or "public-read"
    backup_prefix: Optional[str] = None  # e.g. "backups"
    overwrite_local: bool = True
    refuse_empty_upload: bool = True


# -----------------------------
# Helpers
# -----------------------------
def guess_content_type(p: Path) -> str:
    # mimetypes uses filename extensions; fallback to octet-stream
    ct, _ = mimetypes.guess_type(str(p))
    return ct or "application/octet-stream"


def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def resolve_upload_key(cfg: SyncConfig) -> str:
    """
    Decide which key to upload to.
    Priority:
      1) cfg.dest_key
      2) cfg.key
      3) cfg.prefix + basename(path)
    """
    if cfg.dest_key:
        return cfg.dest_key.lstrip("/")
    if cfg.key:
        return cfg.key.lstrip("/")
    if cfg.prefix:
        px = cfg.prefix.strip("/")
        return f"{px}/{cfg.path.name}"
    raise ValueError("Upload requires one of: --key, --dest-key, or --prefix")


def _maybe_backup_remote(
    spaces: SpacesClient,
    *,
    src_key: str,
    backup_prefix: str,
    acl: str,
    content_type: str,
) -> Optional[str]:
    """
    Copy existing object to backup prefix before replacing.
    Returns backup_key if created, else None.
    """
    backup_key = (
        f"{backup_prefix.rstrip('/')}/{src_key.lstrip('/')}.{int(time.time())}.bak"
    )

    try:
        spaces.client.copy_object(
            Bucket=spaces.bucket,
            CopySource={"Bucket": spaces.bucket, "Key": src_key},
            Key=backup_key,
            ACL=acl,
            ContentType=content_type,
            MetadataDirective="REPLACE",
        )
        return backup_key
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        # If original doesn't exist, backup fails — ignore
        if code in ("NoSuchKey", "404"):
            return None
        raise


# -----------------------------
# Operations
# -----------------------------
def download_file(spaces: SpacesClient, cfg: SyncConfig) -> Optional[Path]:
    if not cfg.key:
        print("ERROR: download requires --key")
        return None

    try:
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create directory for {cfg.path}")
        print(str(e))
        return None

    if cfg.path.exists() and not cfg.overwrite_local:
        print(f"ERROR: Local file exists and overwrite disabled: {cfg.path}")
        return None

    try:
        resp = spaces.client.get_object(Bucket=spaces.bucket, Key=cfg.key)
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            print(f"Remote file not found: s3://{spaces.bucket}/{cfg.key}")
            return None
        raise RuntimeError(
            f"Failed to download s3://{spaces.bucket}/{cfg.key}: {e}"
        ) from e

    body = resp["Body"].read()
    if body is None:
        raise RuntimeError(
            f"Downloaded object has no body: s3://{spaces.bucket}/{cfg.key}"
        )

    cfg.path.write_bytes(body)
    return cfg.path


def upload_file_replace(spaces: SpacesClient, cfg: SyncConfig) -> str:
    if not cfg.path.exists():
        raise FileNotFoundError(f"Local file not found: {cfg.path}")

    data = cfg.path.read_bytes()
    if cfg.refuse_empty_upload and not data:
        raise RuntimeError(f"Local file is empty; refusing to upload: {cfg.path}")

    key = resolve_upload_key(cfg)
    content_type = guess_content_type(cfg.path)

    # Optional backup
    if cfg.backup_prefix:
        try:
            backup_key = _maybe_backup_remote(
                spaces,
                src_key=key,
                backup_prefix=cfg.backup_prefix,
                acl=cfg.acl,
                content_type=content_type,
            )
            if backup_key:
                print(f"Backup created: s3://{spaces.bucket}/{backup_key}")
            else:
                print("No existing object to backup (or not accessible). Continuing...")
        except ClientError as e:
            raise RuntimeError(f"Failed to backup existing object: {e}") from e

    try:
        spaces.client.put_object(
            Bucket=spaces.bucket,
            Key=key,
            Body=data,
            ACL=cfg.acl,
            ContentType=content_type,
        )
    except ClientError as e:
        raise RuntimeError(
            f"Failed to upload (replace) s3://{spaces.bucket}/{key}: {e}"
        ) from e

    return spaces.public_url(key)


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="General download/upload/sync tool for DigitalOcean Spaces."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--path", required=True, help="Local filesystem path to read/write."
        )
        sp.add_argument(
            "--acl",
            default="private",
            help='ACL (default "private"; use "public-read" if needed).',
        )
        sp.add_argument(
            "--backup-prefix",
            default=None,
            help="If set, backups existing remote object before replace.",
        )

        sp.add_argument(
            "--no-refuse-empty-upload",
            action="store_true",
            help="Allow uploading empty files.",
        )

    # download
    sp_d = sub.add_parser("download", help="Download object from Spaces to local path.")
    add_common(sp_d)
    sp_d.add_argument(
        "--key", required=True, help="Spaces object key (e.g., db/my.sqlite)"
    )
    sp_d.add_argument(
        "--no-overwrite-local",
        action="store_true",
        help="Refuse overwrite if local exists.",
    )

    # upload
    sp_u = sub.add_parser("upload", help="Upload local file to Spaces (replace).")
    add_common(sp_u)
    sp_u.add_argument("--key", default=None, help="Target Spaces key.")
    sp_u.add_argument(
        "--dest-key", default=None, help="Override upload key (more explicit)."
    )
    sp_u.add_argument(
        "--prefix",
        default=None,
        help="Upload prefix; key becomes <prefix>/<basename(path)> if no key provided.",
    )

    # sync
    sp_s = sub.add_parser("sync", help="Download then upload (round-trip).")
    add_common(sp_s)
    sp_s.add_argument(
        "--key",
        required=True,
        help="Spaces object key (download source + upload target).",
    )
    sp_s.add_argument(
        "--no-overwrite-local",
        action="store_true",
        help="Refuse overwrite if local exists.",
    )

    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    cfg = SyncConfig(
        key=getattr(args, "key", None),
        path=Path(args.path).expanduser().resolve(),
        dest_key=getattr(args, "dest_key", None),
        prefix=getattr(args, "prefix", None),
        acl=args.acl,
        backup_prefix=args.backup_prefix,
        overwrite_local=not getattr(args, "no_overwrite_local", False),
        refuse_empty_upload=not getattr(args, "no_refuse_empty_upload", False),
    )

    spaces = SpacesClient()  # reads env vars by default

    if args.cmd == "download":
        out = download_file(spaces, cfg)
        if out is None:
            return 1
        print(f"Downloaded to: {out}")
        return 0

    if args.cmd == "upload":
        url = upload_file_replace(spaces, cfg)
        key = resolve_upload_key(cfg)
        print(f"Uploaded (replaced): s3://{spaces.bucket}/{key}")
        print(f"URL (may require auth if private): {url}")
        return 0

    if args.cmd == "sync":
        out = download_file(spaces, cfg)
        if out is None:
            return 1
        print(f"Downloaded to: {out}")
        url = upload_file_replace(spaces, cfg)
        print(f"Uploaded (replaced): s3://{spaces.bucket}/{cfg.key}")
        print(f"URL (may require auth if private): {url}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
