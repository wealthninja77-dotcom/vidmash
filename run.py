"""
VidMash - Video-to-Document Analysis Tool
Entry point for running the application server.
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    """Start the VidMash application server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
