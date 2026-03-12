"""
run.py — Flask application entry point.

Usage:
    python run.py

The app will be available at http://localhost:5000
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config.get("DEBUG", False),
        threaded=True,
        # Disable reloader to prevent the camera thread from starting twice
        use_reloader=False,
    )
