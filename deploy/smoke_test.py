"""Check the built Linux container's workflow and explorer HTTP routes."""

import subprocess
import sys
import time
from html.parser import HTMLParser

import pandas as pd
import requests

from tools.dtale_explorer import create_dtale_session, get_dtale_url


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value and value.startswith("/explorer/"):
                self.paths.append(value)


def main():
    server = subprocess.Popen([sys.executable, "deploy/serve.py"])
    base = "http://127.0.0.1:8080"
    try:
        for _ in range(40):
            try:
                if requests.get(base + "/_stcore/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Streamlit health check failed")
        session = create_dtale_session(pd.DataFrame({"feature": [1, 2, 3]}))
        response = requests.get(base + get_dtale_url(session), timeout=15)
        response.raise_for_status()
        assets = Assets()
        assets.feed(response.text)
        if not assets.paths:
            raise RuntimeError("D-Tale did not generate prefixed asset links")
        for path in assets.paths[:5]:
            requests.get(base + path, timeout=15).raise_for_status()
        print("Container smoke checks passed: Streamlit health, D-Tale page and assets.")
    finally:
        server.terminate()
        server.wait(timeout=20)


if __name__ == "__main__":
    main()
