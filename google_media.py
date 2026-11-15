#!/usr/bin/env python3
"""Download Google Vids media and analyze its audio with AI."""

import argparse
import mimetypes
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path


CACHE = Path.home() / ".cache" / "english" / "google-media"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def file_id(url):
    m = re.search(r"/(?:videos(?:/d)?|document/d|file/d)/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"not a Google media URL: {url}")
    return m.group(1)


def drive():
    from google.auth import default
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build

    creds, _ = default(scopes=SCOPES)
    http = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=30))
    return build("drive", "v3", http=http, cache_discovery=False), creds


def extension(meta):
    if meta["mimeType"].startswith("application/vnd.google-apps."):
        return ".mp4"
    s = Path(meta["name"]).suffix.lower()
    if s in {".mp3", ".mp4", ".m4a", ".ogg", ".oga", ".wav", ".webm"}:
        return s
    return mimetypes.guess_extension(meta["mimeType"]) or ".bin"


def download(url):
    fid = file_id(url)
    svc, creds = drive()
    meta = svc.files().get(
        fileId=fid,
        fields="id,name,mimeType,modifiedTime",
    ).execute()
    ext = extension(meta)
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{fid}-{meta['modifiedTime']}{ext}")
    path = CACHE / name
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        if ext == ".mp4":
            op = svc.files().download(
                fileId=fid, mimeType="video/mp4").execute()
            while not op.get("done"):
                time.sleep(10)
                op = svc.operations().get(name=op["name"]).execute()
            if op.get("error"):
                raise RuntimeError(op["error"])
            req = urllib.request.Request(
                op["response"]["downloadUri"],
                headers={"Authorization": f"Bearer {creds.token}"},
            )
            with urllib.request.urlopen(req, timeout=60) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        else:
            from googleapiclient.http import MediaIoBaseDownload
            with open(tmp, "wb") as f:
                done = False
                downloader = MediaIoBaseDownload(
                    f, svc.files().get_media(fileId=fid))
                while not done:
                    _, done = downloader.next_chunk()
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def analyze(path, text=""):
    import pronounce

    pronounce.load_cfg()
    pronounce.load_data()
    raw = pronounce._load_audio_file(path)
    expected = f' Expected text: "{text}".' if text else ""
    prompt = (
        "Analyze this English speech recording. Transcribe it, then list "
        "pronunciation errors with fixes and grammar or syntax errors with "
        f"corrections.{expected} Be concise and objective."
    )
    print(pronounce._ask_ai(raw, prompt))


def self_test():
    assert file_id("https://docs.google.com/videos/ABC-123/edit") == "ABC-123"
    assert file_id("https://docs.google.com/videos/d/ABC-123/edit") == "ABC-123"
    assert file_id("https://drive.google.com/file/d/xyz_456/view") == "xyz_456"
    assert extension({"mimeType": "application/vnd.google-apps.vid", "name": "x"}) == ".mp4"
    assert extension({"mimeType": "video/webm", "name": "My Video.WEBM"}) == ".webm"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", nargs="?", help="Google Vids or Drive media URL")
    p.add_argument("--text", help="expected text, if known")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        self_test()
        return
    if not a.url:
        p.error("url is required")

    path = download(a.url)
    print(path)
    analyze(path, a.text)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as e:
        sys.exit(e)
