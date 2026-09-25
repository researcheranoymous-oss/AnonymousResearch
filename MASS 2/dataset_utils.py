"""Load the MASS training dataset, preferring a local curl-downloaded parquet copy.

Some environments cannot reach the HF hub/mirror API (``load_dataset`` fails with
``LocalEntryNotFoundError``). Download the parquet with curl into

    $MASS_DATASET_ROOT/Openthoughts_math_30k_opsd/     (default root: /root/autodl-tmp/datasets)

and this reads it via pandas — which also bypasses the hub API entirely. Falls back to the hub
(through ``HF_ENDPOINT`` if set) only when no local copy is present.
"""

import glob
import os

from datasets import Dataset, load_dataset

HUB_ID = "siyanzhao/Openthoughts_math_30k_opsd"
DATASET_LOCAL_ROOT = os.environ.get("MASS_DATASET_ROOT", "/root/autodl-tmp/datasets")


def load_openthoughts(split: str = "train") -> Dataset:
    """Return the Openthoughts math dataset split, preferring local parquet over the hub."""
    local_dir = os.path.join(DATASET_LOCAL_ROOT, "Openthoughts_math_30k_opsd")
    parquets = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    if parquets:
        import pandas as pd

        print(f"Loading Openthoughts from local parquet ({len(parquets)} file(s)) under {local_dir}")
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        return Dataset.from_pandas(df, preserve_index=False)
    print(f"Local Openthoughts not found under {local_dir}; loading {HUB_ID} from hub (needs network).")
    return load_dataset(HUB_ID, split=split)
