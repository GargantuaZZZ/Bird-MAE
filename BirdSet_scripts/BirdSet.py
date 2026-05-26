# Copyright 2025 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics"""

import os
import datasets
import pandas as pd
from tqdm.auto import tqdm
import tarfile
from pathlib import Path

from . import classes

from .classes import BIRD_NAMES_NIPS4BPLUS, BIRD_NAMES_AMAZON_BASIN, BIRD_NAMES_HAWAII, \
    BIRD_NAMES_HIGH_SIERRAS, BIRD_NAMES_SIERRA_NEVADA, BIRD_NAMES_POWDERMILL_NATURE, BIRD_NAMES_SAPSUCKER, \
    BIRD_NAMES_COLUMBIA_COSTA_RICA, BIRD_NAMES_XENOCANTO, BIRD_NAMES_XENOCANTO_M

from .descriptions import _NIPS4BPLUS_CITATION, _NIPS4BPLUS_DESCRIPTION, \
    _HIGH_SIERRAS_DESCRIPTION, _HIGH_SIERRAS_CITATION, _SIERRA_NEVADA_DESCRIPTION, _SIERRA_NEVADA_CITATION, \
    _POWDERMILL_NATURE_DESCRIPTION, _POWDERMILL_NATURE_CITATION, _AMAZON_BASIN_DESCRIPTION, _AMAZON_BASIN_CITATION, \
    _SAPSUCKER_WOODS_DESCRIPTION, _SAPSUCKER_WOODS_CITATION, _COLUMBIA_COSTA_RICA_CITATION, \
    _COLUMBIA_COSTA_RICA_DESCRIPTION, _HAWAIIAN_ISLANDS_CITATION, _HAWAIIAN_ISLANDS_DESCRIPTION


#############################################
_BIRDSET_CITATION = """\
    @misc{rauch2025birdsetlargescaledatasetaudio,
          title={BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics}, 
          author={Lukas Rauch and Raphael Schwinger and Moritz Wirth and Rene Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz},
          year={2025},
          eprint={2403.10380},
          archivePrefix={arXiv},
          primaryClass={cs.SD},
          url={https://arxiv.org/abs/2403.10380}, 
    }
"""
_BIRDSET_DESCRIPTION = """\
    Deep learning (DL) has greatly advanced audio classification, 
    yet the field is limited by the scarcity of large-scale benchmark datasets that have propelled progress in other domains. 
    While AudioSet is a pivotal step to bridge this gap as a universal-domain dataset, its restricted accessibility and 
    limited range of evaluation use cases challenge its role as the sole resource. Therefore, we introduce BirdSet, 
    a large-scale benchmark dataset for audio classification focusing on avian bioacoustics. 
    BirdSet surpasses AudioSet with over 6,800 recording hours (+17%) from nearly 10,000 classes (x18) for training and more 
    than 400 hours (x7) across eight strongly labeled evaluation datasets. It serves as a versatile resource for use 
    cases such as multi-label classification, covariate shift or self-supervised learning. We benchmark six well-known 
    DL models in multi-label classification across three distinct training scenarios and outline further evaluation use 
    cases in audio classification. We host our dataset on Hugging Face for easy accessibility and offer an extensive 
    codebase to reproduce our results. 
"""

base_url = "https://huggingface.co/datasets/DBD-research-group/BirdSet/resolve/data"
DELETE_ARCHIVE = os.getenv("BIRDSET_DELETE_ARCHIVE", "0").lower() in {"1", "true", "yes"}


def _extract_all_to_same_folder(tar_path, output_dir):
    """custom extraction for tar.gz files, that extracts all files to output_dir without subfolders"""
    # check if data already exists
    if not os.path.isfile(output_dir) and os.path.isdir(output_dir) and os.listdir(output_dir):
        return output_dir
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = os.path.basename(member.name)
                tar.extract(member, path=output_dir)

    return output_dir


def _extract_and_delete(dl_dir: dict, cache_dir: str = None, delete_archive: bool = False) -> dict:
    """extracts downloaded files, optionally deleting the archive file immediately, with progress bar."""
    audio_paths = {name: [] for name, data in dl_dir.items() if isinstance(data, list)}
    for name, data in dl_dir.items():
        if not isinstance(data, list):
            continue

        # extract archives, optionally delete after extraction
        for path in tqdm(data, f"Extracting {name} split"):
            directory, filename = os.path.split(path)
            output_dir = os.path.join(cache_dir or directory, "extracted", filename.split(".")[0])
            # audio_path = dl_manager.extract(path)  # if all archive files are without subfolders this works just fine
            audio_path = _extract_all_to_same_folder(path, output_dir)
            if delete_archive:
                os.remove(path)
            # datasets >3.0.0 handles cache differently
            os.remove(f"{path}.lock") if os.path.exists(f"{path}.lock") else None
            os.remove(f"{path}.json") if os.path.exists(f"{path}.json") else None
            audio_paths[name].append(audio_path)

    return audio_paths


class BirdSetConfig(datasets.BuilderConfig):
    def __init__(
            self,
            name,
            citation,
            class_list,
            genus_list,
            species_group_list,
            order_list,
            **kwargs):
        super().__init__(version=datasets.Version("0.0.4"), name=name, **kwargs)

        features = datasets.Features({
            "audio": datasets.Audio(sampling_rate=32_000, mono=True, decode=False),
            "filepath": datasets.Value("string"),
            "start_time": datasets.Value("float64"),
            "end_time": datasets.Value("float64"),
            "low_freq": datasets.Value("int64"),
            "high_freq": datasets.Value("int64"),
            "ebird_code": datasets.ClassLabel(names=class_list),
            "ebird_code_multilabel": datasets.Sequence(datasets.ClassLabel(names=class_list)),
            "ebird_code_secondary": datasets.Sequence(datasets.Value("string")),
            "call_type": datasets.Value("string"),
            "sex": datasets.Value("string"),
            "lat": datasets.Value("float64"),
            "long": datasets.Value("float64"),
            "length": datasets.Value("int64"),
            "microphone": datasets.Value("string"),
            "license": datasets.Value("string"),
            "source": datasets.Value("string"),
            "local_time": datasets.Value("string"),
            "detected_events": datasets.Sequence(datasets.Sequence(datasets.Value("float64"))),
            "event_cluster": datasets.Sequence(datasets.Value("int64")),
            "peaks": datasets.Sequence(datasets.Value("float64")),
            "quality": datasets.Value("string"),
            "recordist": datasets.Value("string"),
            "genus": datasets.ClassLabel(names=genus_list),
            "species_group": datasets.ClassLabel(names=species_group_list),
            "order": datasets.ClassLabel(names=order_list),
            "genus_multilabel": datasets.Sequence(datasets.ClassLabel(names=genus_list)),
            "species_group_multilabel": datasets.Sequence(datasets.ClassLabel(names=species_group_list)),
            "order_multilabel": datasets.Sequence(datasets.ClassLabel(names=order_list)),
        })

        self.features = features
        self.citation = citation


class BirdSet(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""
    # ram problems?
    DEFAULT_WRITER_BATCH_SIZE = 500

    BUILDER_CONFIGS = [
        BirdSetConfig(
            name="SSW",
            description=_SAPSUCKER_WOODS_DESCRIPTION,
            citation=_SAPSUCKER_WOODS_CITATION,
            data_dir=f"{base_url}/SSW",
            class_list=BIRD_NAMES_SAPSUCKER,
            genus_list=classes.GENUS_SSW,
            species_group_list=classes.SPECIES_GROUP_SSW,
            order_list=classes.ORDER_SSW,
        ),
        BirdSetConfig(
            name="SSW_xc",
            description=_SAPSUCKER_WOODS_DESCRIPTION,
            citation=_SAPSUCKER_WOODS_CITATION,
            data_dir=f"{base_url}/SSW",
            class_list=BIRD_NAMES_SAPSUCKER,
            genus_list=classes.GENUS_SSW,
            species_group_list=classes.SPECIES_GROUP_SSW,
            order_list=classes.ORDER_SSW,
        ),
        BirdSetConfig(
            name="SSW_scape",
            description=_SAPSUCKER_WOODS_DESCRIPTION,
            citation=_SAPSUCKER_WOODS_CITATION,
            data_dir=f"{base_url}/SSW",
            class_list=BIRD_NAMES_SAPSUCKER,
            genus_list=classes.GENUS_SSW,
            species_group_list=classes.SPECIES_GROUP_SSW,
            order_list=classes.ORDER_SSW,
        ),
        BirdSetConfig(
            name="PER",
            description=_AMAZON_BASIN_DESCRIPTION,
            citation=_AMAZON_BASIN_CITATION,
            data_dir=f"{base_url}/PER",
            class_list=BIRD_NAMES_AMAZON_BASIN,
            genus_list=classes.GENUS_PER,
            species_group_list=classes.SPECIES_GROUP_PER,
            order_list=classes.ORDER_PER,
        ),
        BirdSetConfig(
            name="PER_xc",
            description=_AMAZON_BASIN_DESCRIPTION,
            citation=_AMAZON_BASIN_CITATION,
            data_dir=f"{base_url}/PER",
            class_list=BIRD_NAMES_AMAZON_BASIN,
            genus_list=classes.GENUS_PER,
            species_group_list=classes.SPECIES_GROUP_PER,
            order_list=classes.ORDER_PER,
        ),
        BirdSetConfig(
            name="PER_scape",
            description=_AMAZON_BASIN_DESCRIPTION,
            citation=_AMAZON_BASIN_CITATION,
            data_dir=f"{base_url}/PER",
            class_list=BIRD_NAMES_AMAZON_BASIN,
            genus_list=classes.GENUS_PER,
            species_group_list=classes.SPECIES_GROUP_PER,
            order_list=classes.ORDER_PER,
        ),
        BirdSetConfig(
            name="UHH",
            description=_HAWAIIAN_ISLANDS_DESCRIPTION,
            citation=_HAWAIIAN_ISLANDS_CITATION,
            data_dir=f"{base_url}/UHH",
            class_list=BIRD_NAMES_HAWAII,
            genus_list=classes.GENUS_UHH,
            species_group_list=classes.SPECIES_GROUP_UHH,
            order_list=classes.ORDER_UHH,
        ),
        BirdSetConfig(
            name="UHH_xc",
            description=_HAWAIIAN_ISLANDS_DESCRIPTION,
            citation=_HAWAIIAN_ISLANDS_CITATION,
            data_dir=f"{base_url}/UHH",
            class_list=BIRD_NAMES_HAWAII,
            genus_list=classes.GENUS_UHH,
            species_group_list=classes.SPECIES_GROUP_UHH,
            order_list=classes.ORDER_UHH,
        ),
        BirdSetConfig(
            name="UHH_scape",
            description=_HAWAIIAN_ISLANDS_DESCRIPTION,
            citation=_HAWAIIAN_ISLANDS_CITATION,
            data_dir=f"{base_url}/UHH",
            class_list=BIRD_NAMES_HAWAII,
            genus_list=classes.GENUS_UHH,
            species_group_list=classes.SPECIES_GROUP_UHH,
            order_list=classes.ORDER_UHH,
        ),
        BirdSetConfig(
            name="SNE",
            description=_SIERRA_NEVADA_DESCRIPTION,
            citation=_SIERRA_NEVADA_CITATION,
            data_dir=f"{base_url}/SNE",
            class_list=BIRD_NAMES_SIERRA_NEVADA,
            genus_list=classes.GENUS_SNE,
            species_group_list=classes.SPECIES_GROUP_SNE,
            order_list=classes.ORDER_SNE,
        ),
        BirdSetConfig(
            name="SNE_xc",
            description=_SIERRA_NEVADA_DESCRIPTION,
            citation=_SIERRA_NEVADA_CITATION,
            data_dir=f"{base_url}/SNE",
            class_list=BIRD_NAMES_SIERRA_NEVADA,
            genus_list=classes.GENUS_SNE,
            species_group_list=classes.SPECIES_GROUP_SNE,
            order_list=classes.ORDER_SNE,
        ),
        BirdSetConfig(
            name="SNE_scape",
            description=_SIERRA_NEVADA_DESCRIPTION,
            citation=_SIERRA_NEVADA_CITATION,
            data_dir=f"{base_url}/SNE",
            class_list=BIRD_NAMES_SIERRA_NEVADA,
            genus_list=classes.GENUS_SNE,
            species_group_list=classes.SPECIES_GROUP_SNE,
            order_list=classes.ORDER_SNE,
        ),
        BirdSetConfig(
            name="POW",
            description=_POWDERMILL_NATURE_DESCRIPTION,
            citation=_POWDERMILL_NATURE_CITATION,
            data_dir=f"{base_url}/POW",
            class_list=BIRD_NAMES_POWDERMILL_NATURE,
            genus_list=classes.GENUS_POW,
            species_group_list=classes.SPECIES_GROUP_POW,
            order_list=classes.ORDER_POW,
        ),
        BirdSetConfig(
            name="POW_xc",
            description=_POWDERMILL_NATURE_DESCRIPTION,
            citation=_POWDERMILL_NATURE_CITATION,
            data_dir=f"{base_url}/POW",
            class_list=BIRD_NAMES_POWDERMILL_NATURE,
            genus_list=classes.GENUS_POW,
            species_group_list=classes.SPECIES_GROUP_POW,
            order_list=classes.ORDER_POW,
        ),
        BirdSetConfig(
            name="POW_scape",
            description=_POWDERMILL_NATURE_DESCRIPTION,
            citation=_POWDERMILL_NATURE_CITATION,
            data_dir=f"{base_url}/POW",
            class_list=BIRD_NAMES_POWDERMILL_NATURE,
            genus_list=classes.GENUS_POW,
            species_group_list=classes.SPECIES_GROUP_POW,
            order_list=classes.ORDER_POW,
        ),
        BirdSetConfig(
            name="HSN",
            description=_HIGH_SIERRAS_DESCRIPTION,
            citation=_HIGH_SIERRAS_CITATION,
            data_dir=f"{base_url}/HSN",
            class_list=BIRD_NAMES_HIGH_SIERRAS,
            genus_list=classes.GENUS_HSN,
            species_group_list=classes.SPECIES_GROUP_HSN,
            order_list=classes.ORDER_HSN,
        ),
        BirdSetConfig(
            name="HSN_xc",
            description=_HIGH_SIERRAS_DESCRIPTION,
            citation=_HIGH_SIERRAS_CITATION,
            data_dir=f"{base_url}/HSN",
            class_list=BIRD_NAMES_HIGH_SIERRAS,
            genus_list=classes.GENUS_HSN,
            species_group_list=classes.SPECIES_GROUP_HSN,
            order_list=classes.ORDER_HSN,
        ),
        BirdSetConfig(
            name="HSN_scape",
            description=_HIGH_SIERRAS_DESCRIPTION,
            citation=_HIGH_SIERRAS_CITATION,
            data_dir=f"{base_url}/HSN",
            class_list=BIRD_NAMES_HIGH_SIERRAS,
            genus_list=classes.GENUS_HSN,
            species_group_list=classes.SPECIES_GROUP_HSN,
            order_list=classes.ORDER_HSN,
        ),
        BirdSetConfig(
            name="NES",
            description=_COLUMBIA_COSTA_RICA_DESCRIPTION,
            citation=_COLUMBIA_COSTA_RICA_CITATION,
            data_dir=f"{base_url}/NES",
            class_list=BIRD_NAMES_COLUMBIA_COSTA_RICA,
            genus_list=classes.GENUS_NES,
            species_group_list=classes.SPECIES_GROUP_NES,
            order_list=classes.ORDER_NES,
        ),
        BirdSetConfig(
            name="NES_xc",
            description=_COLUMBIA_COSTA_RICA_DESCRIPTION,
            citation=_COLUMBIA_COSTA_RICA_CITATION,
            data_dir=f"{base_url}/NES",
            class_list=BIRD_NAMES_COLUMBIA_COSTA_RICA,
            genus_list=classes.GENUS_NES,
            species_group_list=classes.SPECIES_GROUP_NES,
            order_list=classes.ORDER_NES,
        ),
        BirdSetConfig(
            name="NES_scape",
            description=_COLUMBIA_COSTA_RICA_DESCRIPTION,
            citation=_COLUMBIA_COSTA_RICA_CITATION,
            data_dir=f"{base_url}/NES",
            class_list=BIRD_NAMES_COLUMBIA_COSTA_RICA,
            genus_list=classes.GENUS_NES,
            species_group_list=classes.SPECIES_GROUP_NES,
            order_list=classes.ORDER_NES,
        ),
        BirdSetConfig(
            name="NBP",
            description=_NIPS4BPLUS_DESCRIPTION,
            citation=_NIPS4BPLUS_CITATION,
            data_dir=f"{base_url}/NBP",
            class_list=BIRD_NAMES_NIPS4BPLUS,
            genus_list=classes.GENUS_NBP,
            species_group_list=classes.SPECIES_GROUP_NBP,
            order_list=classes.ORDER_NBP,
        ),
        BirdSetConfig(
            name="NBP_xc",
            description=_NIPS4BPLUS_DESCRIPTION,
            citation=_NIPS4BPLUS_CITATION,
            data_dir=f"{base_url}/NBP",
            class_list=BIRD_NAMES_NIPS4BPLUS,
            genus_list=classes.GENUS_NBP,
            species_group_list=classes.SPECIES_GROUP_NBP,
            order_list=classes.ORDER_NBP,
        ),
        BirdSetConfig(
            name="NBP_scape",
            description=_NIPS4BPLUS_DESCRIPTION,
            citation=_NIPS4BPLUS_CITATION,
            data_dir=f"{base_url}/NBP",
            class_list=BIRD_NAMES_NIPS4BPLUS,
            genus_list=classes.GENUS_NBP,
            species_group_list=classes.SPECIES_GROUP_NBP,
            order_list=classes.ORDER_NBP,
        ),
        BirdSetConfig(
            name="XCM",
            description="TODO",
            citation="TODO",
            data_dir=f"{base_url}/XCM",
            class_list=BIRD_NAMES_XENOCANTO_M,
            genus_list=classes.GENUS_XCM,
            species_group_list=classes.SPECIES_GROUP_XCM,
            order_list=classes.ORDER_XCM,
        ),
        BirdSetConfig(
            name="XCL",
            description="TODO",
            citation="TODO",
            data_dir=f"{base_url}/XCL",
            class_list=BIRD_NAMES_XENOCANTO,
            genus_list=classes.GENUS_XCL,
            species_group_list=classes.SPECIES_GROUP_XCL,
            order_list=classes.ORDER_XCL,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_BIRDSET_DESCRIPTION + self.config.description,
            features=self.config.features,
            citation=self.config.citation + "\n" + _BIRDSET_CITATION,
        )

    def _split_generators(self, dl_manager):
        ds_name = self.config.name
        # settings for how much archives (tar.gz) files are uploaded for a specific dataset
        train_files = {"PER": 11,
                       "NES": 13,
                       "UHH": 5,
                       "HSN": 7,
                       "NBP": 32,
                       "POW": 9,
                       "SSW": 29,
                       "SNE": 21,
                       "XCM": 182,
                       "XCL": 98}

        test_files = {"PER": 3,
                      "NES": 8,
                      "UHH": 7,
                      "HSN": 3,
                      "NBP": 1,
                      "POW": 3,
                      "SSW": 36,
                      "SNE": 5}

        test_5s_files = {"PER": 1,
                      "NES": 1,
                      "UHH": 1,
                      "HSN": 1,
                      "NBP": 1,
                      "POW": 1,
                      "SSW": 4,
                      "SNE": 1}

        # different configs, determine what needs to be downloaded
        if self.config.name.endswith("_xc"):
            ds_name = ds_name[:-3]
            dl_dir = dl_manager.download({
                "train": [os.path.join(self.config.data_dir, f"{ds_name}_train_shard_{n:04d}.tar.gz") for n in range(1, train_files[ds_name] + 1)],
                "meta_train": os.path.join(self.config.data_dir, f"{ds_name}_metadata_train.parquet"),
            })

        elif self.config.name.endswith("_scape"):
            ds_name = ds_name[:-6]
            dl_dir = dl_manager.download({
                "test": [os.path.join(self.config.data_dir, f"{ds_name}_test_shard_{n:04d}.tar.gz") for n in range(1, test_files[ds_name] + 1)],
                "test_5s": [os.path.join(self.config.data_dir, f"{ds_name}_test5s_shard_{n:04d}.tar.gz") for n in range(1, test_5s_files[ds_name] + 1)],
                "meta_test": os.path.join(self.config.data_dir, f"{ds_name}_metadata_test.parquet"),
                "meta_test_5s": os.path.join(self.config.data_dir, f"{ds_name}_metadata_test_5s.parquet"),
            })

        # use POW for XCM/XCL validation
        elif self.config.name.startswith("XC"):
            dl_dir = dl_manager.download({
                "train": [os.path.join(self.config.data_dir, f"{ds_name}_shard_{n:04d}.tar.gz") for n in range(1, train_files[ds_name] + 1)],
                "valid": [os.path.join(self.config.data_dir[:-3] + "POW", f"POW_test5s_shard_{n:04d}.tar.gz") for n in range(1, test_5s_files["POW"] + 1)],
                "meta_train": os.path.join(self.config.data_dir, f"{ds_name}_metadata.parquet"),
                "meta_valid": os.path.join(self.config.data_dir[:-3] + "POW", f"POW_metadata_test_5s.parquet"),
            })

        else:
            dl_dir = dl_manager.download({
                "train": [os.path.join(self.config.data_dir, f"{ds_name}_train_shard_{n:04d}.tar.gz") for n in range(1, train_files[ds_name] + 1)],
                "test": [os.path.join(self.config.data_dir, f"{ds_name}_test_shard_{n:04d}.tar.gz") for n in range(1, test_files[ds_name] + 1)],
                "test_5s": [os.path.join(self.config.data_dir, f"{ds_name}_test5s_shard_{n:04d}.tar.gz") for n in range(1, test_5s_files[ds_name] + 1)],
                "meta_train": os.path.join(self.config.data_dir, f"{ds_name}_metadata_train.parquet"),
                "meta_test": os.path.join(self.config.data_dir, f"{ds_name}_metadata_test.parquet"),
                "meta_test_5s": os.path.join(self.config.data_dir, f"{ds_name}_metadata_test_5s.parquet"),
            })

        # custom extraction that keeps archives by default
        audio_paths = _extract_and_delete(
            dl_dir,
            dl_manager.download_config.cache_dir,
            delete_archive=DELETE_ARCHIVE,
        ) if not dl_manager.is_streaming else None

        # construct split generators
        # assumes every key in dl_dir of NAME also has meta_NAME
        names = [name for name in dl_dir.keys() if not name.startswith("meta_")]
        is_streaming = dl_manager.is_streaming

        return [datasets.SplitGenerator(
            name=name,
            gen_kwargs={
                "audio_archive_iterators": (dl_manager.iter_archive(archive_path) for archive_path in dl_dir[name]) if is_streaming else (),
                "audio_extracted_paths": audio_paths[name] if not is_streaming else (),
                "meta_path": dl_dir[f"meta_{name}"],
                "split": name
            }
        ) for name in names]

    def _generate_examples(self, audio_archive_iterators, audio_extracted_paths, meta_path, split):
        metadata = pd.read_parquet(meta_path)
        if metadata.index.name != "filepath":
            metadata.index = metadata["filepath"].str.split("/").apply(lambda x: x[-1])

        idx = 0
        # in case of streaming
        for audio_archive_iterator in audio_archive_iterators:
            for audio_path_in_archive, audio_file in audio_archive_iterator:
                file_name = os.path.split(audio_path_in_archive)[-1]
                rows = metadata.loc[[file_name]]
                audio = audio_file.read()
                for _, row in rows.iterrows():
                    yield idx, self._metadata_from_row(row, split, audio_path=file_name, audio=audio)
                idx += 1

        # in case of not streaming
        for audio_extracted_path in audio_extracted_paths:
            audio_files = os.listdir(audio_extracted_path)
            current_metadata = metadata.loc[audio_files]
            for audio_file, row in current_metadata.iterrows():
                audio_path = os.path.join(audio_extracted_path, audio_file)
                yield idx, self._metadata_from_row(row, split, audio_path=audio_path)
                idx += 1

    @staticmethod
    def _metadata_from_row(row, split: str, audio_path=None, audio=None) -> dict:
        return {"audio": audio_path if not audio else {"path": None, "bytes": audio},
                "filepath": audio_path,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "low_freq": row["low_freq"],
                "high_freq": row["high_freq"],
                "ebird_code": row["ebird_code"] if split != "test_5s" else None,
                "ebird_code_multilabel": row.get("ebird_code_multilabel", None),
                "ebird_code_secondary": row.get("ebird_code_secondary", None),
                "call_type": row["call_type"],
                "sex": row["sex"],
                "lat": row["lat"],
                "long": row["long"],
                "length": row.get("length", None),
                "microphone": row["microphone"],
                "license": row.get("license", None),
                "source": row["source"],
                "local_time": row["local_time"],
                "detected_events": row.get("detected_events", None),
                "event_cluster": row.get("event_cluster", None),
                "peaks": row.get("peaks", None),
                "quality": row.get("quality", None),
                "recordist": row.get("recordist", None),
                "genus": row.get("genus", None) if split != "test_5s" else None,
                "species_group": row.get("species_group", None) if split != "test_5s" else None,
                "order": row.get("order", None) if split != "test_5s" else None,
                "genus_multilabel": row.get("genus_multilabel", [row.get("genus")]),
                "species_group_multilabel": row.get("species_group_multilabel", [row.get("species_group")]),
                "order_multilabel": row.get("order_multilabel", [row.get("order")]),
            }
