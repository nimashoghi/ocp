import contextlib
from typing import Optional

import torch


class ActSaveProvider:
    @torch.jit.ignore()
    def __call__(self, *args, **kwargs):
        pass

    @contextlib.contextmanager
    def context(self, name: Optional[str] = None):
        yield


ActSave = ActSaveProvider()
