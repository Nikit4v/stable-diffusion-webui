from modules import shared, hashes, errors
import os
from modules import paths
from .checkpoint_loader import CheckpointLoader, LocalCheckpointLoader

checkpoints_list = {}
checkpoint_aliases = {}

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))


class CheckpointInfo:
    loader: CheckpointLoader

    def __init__(self, filename, loader: CheckpointLoader = None):
        if loader is None:
            self.loader = LocalCheckpointLoader(filename)
        self.filename = filename

        self.name = self.loader.name()
        self.name_for_extra = self.loader.name_for_extra()
        self.model_name = self.loader.model_name()
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{self.name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = self.name if self.shorthash is None else f'{self.name} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, self.name, f'{self.name} [{self.hash}]'] + (
            [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])

        self.metadata = {}

        _, ext = os.path.splitext(self.filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = read_metadata_from_safetensors(filename)
            except Exception as e:
                errors.display(e, f"reading checkpoint metadata: {filename}")

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        self.shorthash = self.sha256[0:10]

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']

        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()

        return self.shorthash


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len - 2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass

        return res
