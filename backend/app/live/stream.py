"""Live camera stream abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.live.config import LiveCameraConfig


class LiveStream(ABC):
    """Abstract live video stream.

    Implementations provide a uniform frame source for ``LiveCameraManager``.
    The current release uses browser-managed capture. Future server-side
    implementations are documented below.
    """

    @abstractmethod
    def open(self, config: LiveCameraConfig) -> None:
        """Open the stream using the supplied configuration."""

    @abstractmethod
    def close(self) -> None:
        """Close the stream and release resources."""

    @abstractmethod
    def read_frame(self) -> Any | None:
        """Return the next frame if available."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Return whether the stream is open."""


class BrowserWebcamStream(LiveStream):
    """Browser-managed webcam stream.

    Frames are captured in the Web Console and submitted through REST or
    WebSocket transport. The server does not open a local camera device.
    """

    def __init__(self) -> None:
        self._config: LiveCameraConfig | None = None
        self._open = False

    def open(self, config: LiveCameraConfig) -> None:
        self._config = config
        self._open = True

    def close(self) -> None:
        self._open = False
        self._config = None

    def read_frame(self) -> Any | None:
        return None

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def config(self) -> LiveCameraConfig | None:
        return self._config


class OpenCVCameraStream(LiveStream):
    """Reserved server-side local camera stream using OpenCV."""

    def open(self, config: LiveCameraConfig) -> None:
        raise NotImplementedError("OpenCVCameraStream is reserved for a future release.")

    def close(self) -> None:
        raise NotImplementedError("OpenCVCameraStream is reserved for a future release.")

    def read_frame(self) -> Any | None:
        raise NotImplementedError("OpenCVCameraStream is reserved for a future release.")

    @property
    def is_open(self) -> bool:
        return False


class RTSPStream(LiveStream):
    """Reserved RTSP network camera stream."""

    def open(self, config: LiveCameraConfig) -> None:
        raise NotImplementedError("RTSPStream is reserved for a future release.")

    def close(self) -> None:
        raise NotImplementedError("RTSPStream is reserved for a future release.")

    def read_frame(self) -> Any | None:
        raise NotImplementedError("RTSPStream is reserved for a future release.")

    @property
    def is_open(self) -> bool:
        return False


class USBStream(LiveStream):
    """Reserved dedicated USB camera stream."""

    def open(self, config: LiveCameraConfig) -> None:
        raise NotImplementedError("USBStream is reserved for a future release.")

    def close(self) -> None:
        raise NotImplementedError("USBStream is reserved for a future release.")

    def read_frame(self) -> Any | None:
        raise NotImplementedError("USBStream is reserved for a future release.")

    @property
    def is_open(self) -> bool:
        return False


class VideoFileStream(LiveStream):
    """Reserved recorded video file stream."""

    def open(self, config: LiveCameraConfig) -> None:
        raise NotImplementedError("VideoFileStream is reserved for a future release.")

    def close(self) -> None:
        raise NotImplementedError("VideoFileStream is reserved for a future release.")

    def read_frame(self) -> Any | None:
        raise NotImplementedError("VideoFileStream is reserved for a future release.")

    @property
    def is_open(self) -> bool:
        return False
