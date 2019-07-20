import sys

from app import Api

def main():
    """Main entry point for the script."""
    api = Api()
    api.app.run(
        host='0.0.0.0',
        debug=api.settings.debug,
    )

if __name__ == '__main__':
    sys.exit(main())
