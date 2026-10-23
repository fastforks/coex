"""Zoo round 2 generic fp16 path (zoo_fastgen: copy of the IGEV-general 858fdb4 kernels + generic preparation)."""
import os

from zoo_fastgen.wrap import prepare_wrapped


def prepare(model):
    return prepare_wrapped(model, repo_root=os.path.dirname(os.path.abspath(__file__)))
