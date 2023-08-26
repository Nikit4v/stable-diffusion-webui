import abc
import os
import warnings
from pathlib import Path
from typing import Union

from modules import paths, shared

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))


class CheckpointLoader(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def name_for_extra(self) -> str:
        pass

    @abc.abstractmethod
    def model_name(self) -> str:
        pass


class LocalCheckpointLoader(CheckpointLoader):
    path: Path

    def __init__(self, filename: Union[str, Path]):
        if isinstance(filename, str):
            filename = Path(filename)
        self.path = filename

    def name(self) -> str:
        """
        Get name of the checkpoint, aka path, relative to model_dir
        @return:
        """

        return self._check_name_compat(
            str(self.path.relative_to(model_path))
        )

    def _check_name_compat(self, name_to_check: str) -> str:
        abspath = str(self.path.absolute())

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = self.path.name

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        if name != name_to_check:
            warnings.warn(
                f"{name} != {name_to_check}! This only means that new behavior isn't compatible "
                f"with a new one. Currently we will use old-style name.")
            return name
        return name_to_check

    def name_for_extra(self) -> str:
        return self._check_name_for_extra_compat(self.path.stem)

    def _check_name_for_extra_compat(self, name_to_check: str):
        name = os.path.splitext(os.path.basename(self.path.name))[0]
        if name != name_to_check:
            warnings.warn(
                f"{name} != {name_to_check}! This only means that new behavior isn't compatible "
                f"with a new one. Currently we will use old-style name.")
            return name
        return name_to_check

    def model_name(self) -> str:
        return os.path.splitext(self.name().replace("/", "_").replace("\\", "_"))[0]
