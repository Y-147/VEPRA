from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.modules.batchnorm import _BatchNorm


CSSC_POSITIVE_MODES = {
    "same_trial",
    "same_emotion_cross_trial",
}


def validate_cssc_config(
    positive_mode: str,
    trials_per_class: int,
    subjects_per_trial: int,
) -> None:
    if positive_mode not in CSSC_POSITIVE_MODES:
        choices = ", ".join(sorted(CSSC_POSITIVE_MODES))
        raise ValueError(
            "Invalid VEPRA configuration."
        )
    if subjects_per_trial < 2:
        raise ValueError("Invalid VEPRA configuration.")
    if positive_mode == "same_emotion_cross_trial" and trials_per_class < 2:
        raise ValueError(
            "Invalid VEPRA configuration."
        )


@contextmanager
def preserve_all_rng_states():
    
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()

    torch_cuda_states = None
    if torch.cuda.is_available():
        torch_cuda_states = torch.cuda.get_rng_state_all()

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_states is not None:
            torch.cuda.set_rng_state_all(torch_cuda_states)


class CSSCSourceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        subject_id: np.ndarray | torch.Tensor,
        trial_id: np.ndarray | torch.Tensor,
        expected_subjects: int = 14,
        expected_trials: int | None = 24,
        require_global_trial_labels: bool = True,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long).view(-1)
        self.subject_id = torch.as_tensor(subject_id, dtype=torch.long).view(-1)
        self.trial_id = torch.as_tensor(trial_id, dtype=torch.long).view(-1)
        self.expected_subjects = int(expected_subjects)
        self.expected_trials = None if expected_trials is None else int(expected_trials)
        self.require_global_trial_labels = bool(require_global_trial_labels)

        n = self.x.shape[0]
        if not (len(self.y) == n and len(self.subject_id) == n and len(self.trial_id) == n):
            raise ValueError("Invalid VEPRA configuration.")

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        unique_subjects = torch.unique(self.subject_id)
        unique_trials = torch.unique(self.trial_id)

        print(
            f"[CSSC metadata] samples={len(self.x)}, "
            f"subjects={len(unique_subjects)}, "
            f"trials={len(unique_trials)}"
        )

        if len(unique_subjects) != self.expected_subjects:
            raise ValueError(
                "Invalid VEPRA configuration."
                "Invalid VEPRA configuration."
            )
        if self.expected_trials is not None and len(unique_trials) != self.expected_trials:
            raise ValueError(
                "Invalid VEPRA configuration."
                "Invalid VEPRA configuration."
            )

        expected_trial_count = (
            len(unique_trials) if self.expected_trials is None else self.expected_trials
        )
        for subject in unique_subjects:
            subject_mask = self.subject_id == subject
            subject_trials = torch.unique(self.trial_id[subject_mask])
            if subject_trials.numel() != expected_trial_count:
                raise ValueError(
                    "Invalid VEPRA configuration."
                    "Invalid VEPRA configuration."
                )

        for trial in unique_trials:
            mask = self.trial_id == trial
            labels = torch.unique(self.y[mask])
            subjects = torch.unique(self.subject_id[mask])
            if self.require_global_trial_labels and labels.numel() != 1:
                raise ValueError(
                    "Invalid VEPRA configuration."
                )
            if subjects.numel() != unique_subjects.numel():
                raise ValueError(
                    "Invalid VEPRA configuration."
                    "Invalid VEPRA configuration."
                )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return {
            "x": self.x[index],
            "y": self.y[index],
            "subject_id": self.subject_id[index],
            "trial_id": self.trial_id[index],
        }


class StimulusBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        labels: Sequence[int] | np.ndarray | torch.Tensor,
        subject_ids: Sequence[int] | np.ndarray | torch.Tensor,
        trial_ids: Sequence[int] | np.ndarray | torch.Tensor,
        batches_per_epoch: int,
        num_classes: int = 4,
        trials_per_class: int = 1,
        subjects_per_trial: int = 4,
        windows_per_group: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(None)
        self.labels = _to_numpy_int64(labels)
        self.subject_ids = _to_numpy_int64(subject_ids)
        self.trial_ids = _to_numpy_int64(trial_ids)
        self.batches_per_epoch = int(batches_per_epoch)
        self.num_classes = int(num_classes)
        self.trials_per_class = int(trials_per_class)
        self.subjects_per_trial = int(subjects_per_trial)
        self.windows_per_group = int(windows_per_group)
        self.seed = int(seed)
        self.epoch = 0

        if not (len(self.labels) == len(self.subject_ids) == len(self.trial_ids)):
            raise ValueError("Invalid VEPRA configuration.")
        if self.trials_per_class <= 0:
            raise ValueError("Invalid VEPRA configuration.")
        if self.subjects_per_trial <= 0:
            raise ValueError("Invalid VEPRA configuration.")
        if self.windows_per_group <= 0:
            raise ValueError("Invalid VEPRA configuration.")

        self.pair_to_indices: Dict[Tuple[int, int], np.ndarray] = {}
        self.trial_to_subjects: Dict[int, np.ndarray] = {}
        self.trial_to_label: Dict[int, int] = {}
        self.class_to_trials: Dict[int, List[int]] = {}

        self._build_index()

    def _build_index(self) -> None:
        unique_trials = np.unique(self.trial_ids)
        for trial in unique_trials:
            trial_mask = self.trial_ids == trial
            trial_labels = np.unique(self.labels[trial_mask])
            if len(trial_labels) != 1:
                raise ValueError("Invalid VEPRA configuration.")
            label = int(trial_labels[0])
            subjects = np.unique(self.subject_ids[trial_mask])
            valid_subjects = []
            for subject in subjects:
                pair_mask = (self.trial_ids == trial) & (self.subject_ids == subject)
                indices = np.flatnonzero(pair_mask)
                if len(indices) > 0:
                    self.pair_to_indices[(int(subject), int(trial))] = indices
                    valid_subjects.append(int(subject))
            if len(valid_subjects) < self.subjects_per_trial:
                continue
            self.trial_to_subjects[int(trial)] = np.asarray(valid_subjects, dtype=np.int64)
            self.trial_to_label[int(trial)] = label
            self.class_to_trials.setdefault(label, []).append(int(trial))

        available_classes = sorted(self.class_to_trials.keys())
        if len(available_classes) != self.num_classes:
            raise ValueError("Invalid VEPRA configuration.")
        for label in available_classes:
            trial_count = len(self.class_to_trials[label])
            if trial_count < self.trials_per_class:
                raise ValueError(
                    "Invalid VEPRA configuration."
                    "Invalid VEPRA configuration."
                )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        classes = sorted(self.class_to_trials.keys())
        for _ in range(self.batches_per_epoch):
            batch_indices: List[int] = []
            for label in classes:
                candidate_trials = self.class_to_trials[label]
                if self.trials_per_class == 1:
                    
                    selected_trials = [int(rng.choice(candidate_trials))]
                else:
                    selected_trials = rng.choice(
                        candidate_trials,
                        size=self.trials_per_class,
                        replace=False,
                    ).astype(int).tolist()

                for trial in selected_trials:
                    available_subjects = self.trial_to_subjects[trial]
                    selected_subjects = rng.choice(
                        available_subjects,
                        size=self.subjects_per_trial,
                        replace=False,
                    )
                    for subject in selected_subjects:
                        pool = self.pair_to_indices[(int(subject), trial)]
                        selected_windows = rng.choice(
                            pool,
                            size=self.windows_per_group,
                            replace=len(pool) < self.windows_per_group,
                        )
                        batch_indices.extend(selected_windows.astype(int).tolist())
            yield batch_indices


class CrossSubjectEmotionBatchSampler(Sampler[List[int]]):
    

    def __init__(
        self,
        labels: Sequence[int] | np.ndarray | torch.Tensor,
        subject_ids: Sequence[int] | np.ndarray | torch.Tensor,
        trial_ids: Sequence[int] | np.ndarray | torch.Tensor,
        batches_per_epoch: int,
        num_classes: int,
        trials_per_class: int,
        subjects_per_trial: int,
        windows_per_group: int,
        seed: int,
    ) -> None:
        super().__init__(None)
        self.labels = _to_numpy_int64(labels)
        self.subject_ids = _to_numpy_int64(subject_ids)
        self.trial_ids = _to_numpy_int64(trial_ids)
        self.batches_per_epoch = int(batches_per_epoch)
        self.num_classes = int(num_classes)
        self.groups_per_class = int(trials_per_class) * int(subjects_per_trial)
        self.windows_per_group = int(windows_per_group)
        self.seed = int(seed)
        self.epoch = 0

        if self.groups_per_class < 2:
            raise ValueError("Invalid VEPRA configuration.")
        if self.windows_per_group <= 0:
            raise ValueError("Invalid VEPRA configuration.")
        if not (len(self.labels) == len(self.subject_ids) == len(self.trial_ids)):
            raise ValueError("Invalid VEPRA configuration.")

        self.pair_to_indices: Dict[Tuple[int, int], np.ndarray] = {}
        self.class_to_groups: Dict[int, List[Tuple[int, int]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        group_keys = np.unique(
            np.stack([self.subject_ids, self.trial_ids], axis=1),
            axis=0,
        )
        for subject, trial in group_keys:
            mask = (self.subject_ids == subject) & (self.trial_ids == trial)
            group_labels = np.unique(self.labels[mask])
            if len(group_labels) != 1:
                raise ValueError(
                    "Invalid VEPRA configuration."
                    f"{group_labels.tolist()}"
                )
            key = (int(subject), int(trial))
            self.pair_to_indices[key] = np.flatnonzero(mask)
            self.class_to_groups.setdefault(int(group_labels[0]), []).append(key)

        available_classes = sorted(self.class_to_groups)
        if available_classes != list(range(self.num_classes)):
            raise ValueError(
                "Invalid VEPRA configuration."
                "Invalid VEPRA configuration."
            )
        for label, groups in self.class_to_groups.items():
            subject_count = len({subject for subject, _ in groups})
            trial_count = len({trial for _, trial in groups})
            if subject_count < self.groups_per_class or trial_count < 2:
                raise ValueError(
                    "Invalid VEPRA configuration."
                    "Invalid VEPRA configuration."
                    "Invalid VEPRA configuration."
                )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _sample_groups(
        self,
        rng: np.random.Generator,
        label: int,
    ) -> List[Tuple[int, int]]:
        candidates = self.class_to_groups[label]
        for _ in range(100):
            order = rng.permutation(len(candidates))
            selected: List[Tuple[int, int]] = []
            used_subjects = set()
            for index in order:
                subject, trial = candidates[int(index)]
                if subject in used_subjects:
                    continue
                selected.append((subject, trial))
                used_subjects.add(subject)
                if len(selected) == self.groups_per_class:
                    if len({item[1] for item in selected}) >= 2:
                        return selected
                    break
        raise RuntimeError(
            "Invalid VEPRA configuration."
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches_per_epoch):
            batch_indices: List[int] = []
            for label in range(self.num_classes):
                for key in self._sample_groups(rng, label):
                    pool = self.pair_to_indices[key]
                    selected_windows = rng.choice(
                        pool,
                        size=self.windows_per_group,
                        replace=len(pool) < self.windows_per_group,
                    )
                    batch_indices.extend(selected_windows.astype(int).tolist())
            yield batch_indices


class CSSCProjectionHead(nn.Module):
    def __init__(self, feature_dim: int = 64, projection_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "Invalid VEPRA configuration."
            )
        expected_dim = self.net[0].in_features
        if x.shape[1] != expected_dim:
            raise ValueError(
                "Invalid VEPRA configuration."
                "Invalid VEPRA configuration."
            )
        return self.net(x)


def _to_numpy_int64(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.int64).reshape(-1)


def build_subject_trial_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    trial_ids: torch.Tensor,
):
    if features.ndim != 2:
        raise ValueError("Invalid VEPRA configuration.")
    if features.shape[0] != labels.numel():
        raise ValueError("Invalid VEPRA configuration.")

    labels = labels.view(-1).long()
    subject_ids = subject_ids.view(-1).long()
    trial_ids = trial_ids.view(-1).long()

    keys = torch.stack([subject_ids, trial_ids], dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    num_groups = unique_keys.shape[0]
    feature_dim = features.shape[1]

    prototypes = torch.zeros(num_groups, feature_dim, dtype=features.dtype, device=features.device)
    prototypes.index_add_(0, inverse, features)
    counts = torch.bincount(inverse, minlength=num_groups).to(features.dtype).unsqueeze(1)
    prototypes = prototypes / counts.clamp_min(1.0)

    group_labels = torch.empty(num_groups, dtype=torch.long, device=features.device)
    for g in range(num_groups):
        mask = inverse == g
        current_labels = torch.unique(labels[mask])
        if current_labels.numel() != 1:
            raise RuntimeError("Invalid VEPRA configuration.")
        group_labels[g] = current_labels[0]

    group_subjects = unique_keys[:, 0]
    group_trials = unique_keys[:, 1]
    return prototypes, group_labels, group_subjects, group_trials


def build_cssc_pair_masks(
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    trial_ids: torch.Tensor,
    positive_mode: str = "same_trial",
):
    if positive_mode not in CSSC_POSITIVE_MODES:
        choices = ", ".join(sorted(CSSC_POSITIVE_MODES))
        raise ValueError(
            "Invalid VEPRA configuration."
        )

    labels = labels.view(-1)
    subject_ids = subject_ids.view(-1)
    trial_ids = trial_ids.view(-1)
    if not (labels.numel() == subject_ids.numel() == trial_ids.numel()):
        raise ValueError("Invalid VEPRA configuration.")

    batch_size = labels.numel()
    device = labels.device
    eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    run_loso = subject_ids[:, None] != subject_ids[None, :]
    same_trial = trial_ids[:, None] == trial_ids[None, :]
    same_emotion = labels[:, None] == labels[None, :]

    if positive_mode == "same_trial":
        positive_mask = run_loso & same_trial & ~eye_mask
    else:
        positive_mask = (
            run_loso
            & ~same_trial
            & same_emotion
            & ~eye_mask
        )

    negative_mask = run_loso & ~same_emotion & ~eye_mask
    return positive_mask, negative_mask


def cssc_contrastive_loss(
    projected_prototypes: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    trial_ids: torch.Tensor,
    temperature: float = 0.1,
    positive_mode: str = "same_trial",
    strict: bool = True,
):
    if temperature <= 0:
        raise ValueError("Invalid VEPRA configuration.")
    if projected_prototypes.shape[0] != labels.numel():
        raise ValueError(
            "Invalid VEPRA configuration."
            "Invalid VEPRA configuration."
        )

    z = F.normalize(projected_prototypes.float(), p=2, dim=1)
    cosine_sim = torch.matmul(z, z.t())
    logits = cosine_sim / float(temperature)

    positive_mask, negative_mask = build_cssc_pair_masks(
        labels,
        subject_ids,
        trial_ids,
        positive_mode=positive_mode,
    )
    valid_pair_mask = positive_mask | negative_mask

    mask_value = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~valid_pair_mask, mask_value)

    log_denom = torch.logsumexp(masked_logits, dim=1)
    positive_count = positive_mask.sum(dim=1)
    negative_count = negative_mask.sum(dim=1)
    valid_anchor = (positive_count > 0) & (negative_count > 0)

    if not torch.any(valid_anchor):
        if strict:
            raise RuntimeError(
                "Invalid VEPRA configuration."
            )
        return projected_prototypes.sum() * 0.0, {
            "positive_similarity": float("nan"),
            "negative_similarity": float("nan"),
            "valid_anchors": 0,
            "min_positive_count": 0,
            "max_positive_count": 0,
        }

    log_prob = logits - log_denom.unsqueeze(1)
    pos_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_anchor = -pos_log_prob.sum(dim=1) / positive_count.clamp_min(1)
    loss = loss_per_anchor[valid_anchor].mean()

    with torch.no_grad():
        pos_sim = cosine_sim[positive_mask].mean()
        neg_sim = cosine_sim[negative_mask].mean()
        min_pos_count = int(positive_count[valid_anchor].min().detach().cpu())
        max_pos_count = int(positive_count[valid_anchor].max().detach().cpu())

    stats = {
        "positive_similarity": float(pos_sim.detach().cpu()),
        "negative_similarity": float(neg_sim.detach().cpu()),
        "valid_anchors": int(valid_anchor.sum().detach().cpu()),
        "min_positive_count": min_pos_count,
        "max_positive_count": max_pos_count,
    }
    return loss, stats


def build_cssc_source_loader(args, train_dataset):
    validate_cssc_config(
        args.cssc_positive_mode,
        args.cssc_trials_per_class,
        args.cssc_subjects_per_trial,
    )

    groups = np.asarray(train_dataset["groups"])
    labels = np.asarray(train_dataset["labels"])
    if groups.ndim != 2 or groups.shape[1] < 2:
        raise ValueError(
            "Invalid VEPRA configuration."
            "Invalid VEPRA configuration."
        )
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)

    expected_trials_by_dataset = {
        "seed3": 15,
        "seed4": 24,
        "deap": 40,
    }
    expected_trials = expected_trials_by_dataset.get(args.dataset_name)

    dataset = CSSCSourceDataset(
        x=train_dataset["data"],
        y=labels,
        subject_id=groups[:, 0],
        trial_id=groups[:, 1],
        expected_subjects=args.num_sources,
        expected_trials=expected_trials,
        require_global_trial_labels=args.cssc_positive_mode == "same_trial",
    )
    sampler_class = (
        StimulusBatchSampler if args.cssc_positive_mode == "same_trial" else CrossSubjectEmotionBatchSampler
    )
    sampler = sampler_class(
        labels=dataset.y,
        subject_ids=dataset.subject_id,
        trial_ids=dataset.trial_id,
        batches_per_epoch=args.max_iter,
        num_classes=args.num_classes,
        trials_per_class=args.cssc_trials_per_class,
        subjects_per_trial=args.cssc_subjects_per_trial,
        windows_per_group=args.cssc_windows_per_group,
        seed=args.seed + 54321,
    )
    cssc_generator = torch.Generator()
    cssc_generator.manual_seed(args.seed + 54321)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        generator=cssc_generator,
    )


@contextmanager
def freeze_batchnorm_running_stats(module: nn.Module):
    bn_modules = [m for m in module.modules() if isinstance(m, _BatchNorm)]
    original_states = [m.training for m in bn_modules]
    try:
        for m in bn_modules:
            m.eval()
        yield
    finally:
        for m, state in zip(bn_modules, original_states):
            m.train(state)
