"""What the a1 network pads its input with, and a switch to run it unpadded.

No import here touches `model` or sets an environment switch, so this module is safe to
import from the export scripts (xformers disabled) and from the ordinary pipeline alike.
"""

PAD_FACTOR = 112  # model/waft_a1.py: Padder(image1.shape, factor=112)
PADDING_MODES = ("reference", "none")


class ExactPadder:
    """Stand-in for utils.utils.Padder that pads nothing.

    The reference Padder pads to the next multiple of `factor` above size + 8, so it adds
    between 9 and 120 pixels per axis even to a size that already is a multiple: 448x336
    runs as 560x448. This one is only valid for sizes the network accepts unpadded.
    """

    def __init__(self, dims, mode="sintel", factor=32):
        ht, wd = dims[-2:]
        if ht % factor or wd % factor:
            raise ValueError(f"{wd}x{ht} is not a multiple of {factor}; it cannot run unpadded")

    def pad(self, x):
        return x

    def unpad(self, x):
        return x


def set_padding(mode: str) -> None:
    """Select what model/waft_a1.py pads with. "reference" is the pinned code's Padder."""
    import model.waft_a1 as waft_a1
    import utils.utils as waft_utils

    if mode not in PADDING_MODES:
        raise ValueError(f"unknown padding mode {mode!r}, expected one of {PADDING_MODES}")
    waft_a1.Padder = ExactPadder if mode == "none" else waft_utils.Padder


def padded_size(height: int, width: int, mode: str) -> tuple[int, int]:
    """(height, width) the network actually runs at for an input of this size."""
    if mode == "none":
        ExactPadder((height, width), factor=PAD_FACTOR)
        return height, width
    return (((height + 8) // PAD_FACTOR) + 1) * PAD_FACTOR, (((width + 8) // PAD_FACTOR) + 1) * PAD_FACTOR
