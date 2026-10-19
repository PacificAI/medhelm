import os


class ProgressFile:
    """File-like object: each tqdm write replaces the file with the current bar."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, s):
        text = s.replace("\r", "").replace("\n", "").strip()
        if not text:
            return
        with open(self.path, "w", encoding="utf-8") as fp:
            fp.write(text + "\n")

    def flush(self):
        pass

    def isatty(self):
        return False

    def close(self):
        pass
