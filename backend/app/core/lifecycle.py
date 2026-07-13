from backend.app.config.configuration import Configuration


def startup_app() -> None:
    """Initialize and display application startup information."""
    settings = Configuration().load()

    print("=" * 60)
    print(settings.app.name)
    print(f"Version      : {settings.app.version}")
    print(f"Environment  : {settings.app.environment}")
    print("=" * 60)
    print()
    print("System Status : READY")


def shutdown_app() -> None:
    """Release application resources during shutdown."""
    pass
