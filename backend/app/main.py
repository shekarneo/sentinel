# Application entry point
from backend.app.core.lifecycle import startup_app

def main():
    startup_app()

if __name__ == "__main__":
    main()