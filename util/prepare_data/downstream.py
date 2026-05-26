# downstream data  complete
import argparse
import os

from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_dataset

def process_downstream_datasets(
    dataset_names: list[str],
    cache_dir_base: str,
    skip_download: bool,
    dataset_script_path: str | None,
    data_dir_base: str,
    output_dir_base: str | None,
    n_workers: int,
    delete_archive: bool,
):
    """
    Loads and prepares BirdSet datasets.

    Args:
        dataset_names: A list of dataset names to process (e.g., ["PER", "NES"]).
        cache_dir_base: The base directory for caching datasets (e.g., "/scratch/birdset").
    """
    use_local_script = bool(dataset_script_path)
    effective_skip_download = skip_download or use_local_script
    if delete_archive:
        os.environ["BIRDSET_DELETE_ARCHIVE"] = "1"
    else:
        os.environ["BIRDSET_DELETE_ARCHIVE"] = "0"

    if not effective_skip_download:
        for name in dataset_names:
            print(f"Loading {name}", flush=True)
            cache_path = f"{cache_dir_base}/{name}"
            # Ensure cache directory exists or is created by load_dataset
            load_dataset("DBD-research-group/BirdSet", name, num_proc=5, cache_dir=cache_path)
            print(f"Loaded {name}", flush=True)
    else:
        for name in dataset_names:
            data_path = f"{data_dir_base}/{name}"
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Local data not found for {name}: {data_path}. "
                    "Either set --data-dir-base correctly or remove --skip-download."
                )
            print(f"Using local data for {name}: {data_path}", flush=True)

    for name in dataset_names:
        print(f"preparing {name}", flush=True)
        data_dir = f"{data_dir_base}/{name}"
        hf_path = dataset_script_path or "DBD-research-group/BirdSet"
        if dataset_script_path:
            script_dir = os.path.dirname(dataset_script_path)
            data_dir = os.path.relpath(data_dir, start=script_dir)

        dm = BirdSetDataModule(
            dataset= DatasetConfig(
                data_dir=data_dir,
                cache_dir=f"{cache_dir_base}/{name}",
                output_dir=(f"{output_dir_base}/{name}" if output_dir_base else None),
                hf_path=hf_path,
                hf_name=name,
                n_workers=n_workers,
                val_split=0.0001,
                task="multilabel",
                classlimit=500,
                eventlimit=5,
                sampling_rate=32_000,
                trust_remote_code=bool(dataset_script_path),
            ),
        )
        dm.prepare_data()
        print(f"Prepared data for {name} saved to: {dm.disk_save_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and prepare BirdSet datasets.")
    parser.add_argument(
        "--dataset-names",
        nargs='+',
        default=["PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE"],
        help="List of dataset names to process (e.g., PER NES XCL)."
    )
    parser.add_argument(
        "--cache-dir-base",
        type=str,
        default="/data/birdset",
        help="Base directory for datasets cache output."
    )
    parser.add_argument(
        "--data-dir-base",
        type=str,
        default=None,
        help="Base directory for local BirdSet data (e.g., /data/birdset)."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading from Hugging Face and use local cache only."
    )
    parser.add_argument(
        "--dataset-script-path",
        type=str,
        default=None,
        help="Local BirdSet script path (e.g., /path/to/BirdSet.py) for offline use."
    )
    parser.add_argument(
        "--output-dir-base",
        type=str,
        default=None,
        help="Base directory for processed dataset output (e.g., /data/birdset/processed)."
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=3,
        help="Number of workers for dataset map operations (e.g., one-hot encoding)."
    )
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="Delete tar.gz archives after extraction (default: keep)."
    )

    args = parser.parse_args()
    
    data_dir_base = args.data_dir_base or args.cache_dir_base
    process_downstream_datasets(
        args.dataset_names,
        args.cache_dir_base,
        args.skip_download,
        args.dataset_script_path,
        data_dir_base,
        args.output_dir_base,
        args.n_workers,
        args.delete_archive,
    )

