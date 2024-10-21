#!/usr/bin/python3
"""Recipe for training a classifier using the
AudioMNIST dataset.

To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/lfb.yaml

Author
    * Nicolas Aspert 2024
"""
import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
import speechbrain.nnet.CNN
from speechbrain.utils.distributed import run_on_main


class AudioBrain(sb.core.Brain):
    """Class for AudioMNIST training" """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + command classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if isinstance(
            self.modules.compute_features, speechbrain.lobes.features.Leaf
        ):
            # if leaf, first normalize the wavs before feeding them to leaf
            # no normalization is needed after LEAF
            feats = self.modules.mean_var_norm(wavs, lens)
            feats = self.modules.compute_features(feats)
        else:
            # Feature extraction and normalization
            feats = self.modules.compute_features(wavs)
            feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        # Ecapa model uses softmax outside of its classifier
        if "softmax" in self.modules.keys():
            outputs = self.modules.softmax(outputs)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using command-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        command, _ = batch.command_encoded

        # compute the cost function
        loss = self.hparams.compute_cost(predictions, command, lens)
        # loss = sb.nnet.losses.nll_loss(predictions, command, lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, command, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


DATASET_SPLITS = ["train", "valid", "test"]


def apply_sort(hparams, dataset):
    if hparams["sort"]:
        dataset = dataset.filtered_sorted(sort_key=hparams["sort"])
    if hparams["batch_shuffle"]:
        dataset = dataset.batch_shuffle(hparams["batch_size"])
    return dataset


def load_dataset(hparams):
    dataset_splits = {}
    data_folder = hparams["data_save_folder"]
    for split_id in DATASET_SPLITS:
        split_path = hparams[f"{split_id}_json"]
        dataset_split = sb.dataio.dataset.DynamicItemDataset.from_json(
            split_path, replacements={"data_root": data_folder}
        )
        dataset_split = apply_sort(hparams, dataset_split)
        dataset_splits[split_id] = dataset_split
    return dataset_splits


def dataio_prep(hparams):
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_name")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and output it"""
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("digit", "speaker_id")
    @sb.utils.data_pipeline.provides("digit_label", "speaker_label")
    def labels_pipeline(digit, speaker_id):
        yield int(digit)
        yield int(speaker_id) - 1

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.

    dataset_splits = load_dataset(hparams)
    dataset_splits_values = dataset_splits.values()

    output_keys = ["file_name", "sig", "digit_label", "speaker_label"]
    if "done_detector" in hparams:
        output_keys += ["file_name_random", "sig_random"]

    sb.dataio.dataset.set_output_keys(
        dataset_splits_values,
        output_keys,
    )
    sb.dataio.dataset.add_dynamic_item(dataset_splits_values, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(dataset_splits_values, labels_pipeline)

    train_split = dataset_splits["train"]

    if hparams["train_data_count"] is not None:
        data_count = hparams["train_data_count"]
        train_split.data_ids = train_split.data_ids[:data_count]
    dataset_splits["train"] = train_split

    return dataset_splits


if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing GSC and annotation into csv files)
    from audiomnist_prepare import prepare_audiomnist

    # Data preparation
    run_on_main(
        prepare_audiomnist,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_json": hparams["train_json"],
            "valid_json": hparams["valid_json"],
            "test_json": hparams["test_json"],
            "metadata_folder": hparams["metadata_folder"],
            "norm": hparams["data_prepare_norm"],
            "trim": hparams["data_prepare_trim"],
            "trim_threshold": hparams["data_prepare_trim_threshold"],
            "src_sample_rate": hparams["data_prepare_sample_rate_src"],
            "tgt_sample_rate": hparams["data_prepare_sample_rate_tgt"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    audio_datasets = dataio_prep(hparams)

    # Brain class initialization
    audio_brain = AudioBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # with torch.autograd.detect_anomaly():
    # Training
    audio_brain.fit(
        audio_brain.hparams.epoch_counter,
        train_set=audio_datasets["train"],
        valid_set=audio_datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = audio_brain.evaluate(
        test_set=audio_datasets["test"],
        min_key="ErrorRate",
        test_loader_kwargs=hparams["dataloader_options"],
    )