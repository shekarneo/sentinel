"""
Pipeline profile definitions.

Built-in profiles describe which biometric stages run for each use case.
"""

from enum import Enum


class PipelineProfile(str, Enum):
    """Supported biometric pipeline use cases."""

    ENROLLMENT = "enrollment"
    ATTENDANCE = "attendance"
    SURVEILLANCE = "surveillance"
    KYC = "kyc"
    ACCESS_CONTROL = "access_control"
    SEARCH = "search"
    CUSTOM = "custom"
