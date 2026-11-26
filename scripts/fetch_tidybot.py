"""Fetch the official TidyBot XML, meshes and licenses at a fixed revision."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "eea4c7431b63da8278b0ff31638e0b0e941cec0e"
REPO = "google-deepmind/mujoco_menagerie"


def read_url(url):
    with urlopen(Request(url, headers={"User-Agent": "Fishingbot-scene-v0"}), timeout=90) as response:
        return response.read()


def main():
    tree = json.loads(read_url(f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1"))
    paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"
             and (item["path"].startswith("stanford_tidybot/assets/")
                  or item["path"] in ("stanford_tidybot/tidybot.xml", "stanford_tidybot/README.md",
                                      "stanford_tidybot/LICENSE", "kinova_gen3/LICENSE", "robotiq_2f85/LICENSE"))]
    assert "stanford_tidybot/tidybot.xml" in paths

    def download(path):
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe upstream path: {path}")
        target = ROOT / "models" / relative
        data = read_url(f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(download, paths))
    (ROOT / "models" / "tidybot-source.json").write_text(json.dumps(
        {"repository": f"https://github.com/{REPO}", "revision": REVISION, "files": files}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Downloaded {len(files)} files at {REVISION}")


if __name__ == "__main__":
    main()
