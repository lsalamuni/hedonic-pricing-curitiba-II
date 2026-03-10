"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()
BLD_DATA: Path = BLD.joinpath("data").resolve()
BLD_ANALYSIS: Path = BLD.joinpath("analysis").resolve()
BLD_IMAGES: Path = BLD.joinpath("images").resolve()
BLD_FINAL: Path = BLD.joinpath("final").resolve()

DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

MARKET_TIERS: tuple[str, ...] = ("low", "mid", "high")
