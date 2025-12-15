from sklearn.model_selection import KFold
import random

def make_range(prefix, start, end):
    step = -1 if start >= end else 1
    return [f"{prefix}{i}.npy" for i in range(start, end + step, step)]


def generate_her2_splits():
    # All 36 sample IDs
    her2_all = make_range("SPA", 154, 119)

    # First 24 samples split into 4 groups of 6 (6x4)
    her2_6x_samples = [her2_all[i * 6:(i + 1) * 6] for i in range(4)]
    # Last 12 samples split into 4 groups of 3 (3x4)
    her2_3x_samples = [her2_all[24 + i * 3:24 + (i + 1) * 3] for i in range(4)]

    folds = []
    for i in range(4):
        # Validation set for this fold: all 6 + 3 samples from group i
        val = her2_6x_samples[i] + her2_3x_samples[i]
        # Training set = all samples minus validation set
        train = [sid for sid in her2_all if sid not in val]
        folds.append({'train': train, 'val': val})

        print({'train': train, 'val': val})
    return folds


def generate_bc_splits():
    bc_all = [s for s in make_range("SPA", 118, 51) if s not in ("SPA112.npy", "SPA111.npy")]
    bc_samples = [bc_all[i * 3:(i + 1) * 3] for i in range(len(bc_all) // 3)]
    random.seed(42)
    random.shuffle(bc_samples)

    folds = []
    kf = KFold(n_splits=4)
    for train_idx, val_idx in kf.split(bc_samples):
        train_samples = [bc_samples[i] for i in train_idx]
        val_samples = [bc_samples[i] for i in val_idx]
        val_first_slide = [sample[0] for sample in val_samples]
        val_slide = [sample[0] for sample in val_samples]
        train = sum(train_samples, []) + val_first_slide + val_slide
        val = sum(val_samples, [])
        folds.append({'train': train, 'val': val})
    return folds



def generate_kidney_splits():
    kidney_all = make_range("NCBI", 714, 692)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in kf.split(kidney_all):
        train = [kidney_all[i] for i in train_idx]
        val = [kidney_all[i] for i in val_idx]
        folds.append({'train': train, 'val': val})
    return folds


def get_folds(dataset_name):
    if dataset_name == "HER2":
        return generate_her2_splits()
    elif dataset_name == "BC":
        return generate_bc_splits()
    elif dataset_name == "Kidney":
        return generate_kidney_splits()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
