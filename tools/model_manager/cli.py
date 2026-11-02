"""Compatibility CLI for the canonical model downloader."""

from .downloader import SmartModelDownloader, download_all_models, main

__all__ = ["SmartModelDownloader", "download_all_models", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
