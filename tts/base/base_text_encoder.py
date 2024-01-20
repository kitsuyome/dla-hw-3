from torch import Tensor


class BaseTextEncoder:
    def encode(self, text) -> Tensor:
        raise NotImplementedError()
