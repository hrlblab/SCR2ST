import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from tensorboardX import SummaryWriter

from eval_metric import calculate_pcc
from cross_fold import get_folds

# ====== Additional modules for Active RL ====== #
from dataset import STDataset
from rl_sampler import RLSampler
from reward import *  # if used internally

from model import DenseRegressor, retrieval_module


# =====================================================
# For active version: load records from npy file list
# Each npy: array(list[dict])
# =====================================================
def load_records_from_npy_list(paths):
    all_recs = []
    logging.info(f"[INFO] Loading ST npy files ...")
    for p in tqdm(paths, desc="Loading NPY", ncols=80):
        arr = np.load(p, allow_pickle=True)
        all_recs.extend(arr.tolist())
    logging.info(f"[INFO] Loaded {len(all_recs)} spots from {len(paths)} files.")
    return all_recs


def merge_epoch_metrics_to_excel(root_dir: str, save_dir: str):
    """
    Every 10 epochs, merge slidewise_metrics/epoch_*.csv from all folds
    into separate .xlsx files, named epoch_{N}.xlsx.
    """
    all_folds = sorted(glob(os.path.join(root_dir, "fold*")))
    epoch_dict = {}

    for fold_dir in all_folds:
        fold_idx = int(os.path.basename(fold_dir).replace("fold", ""))
        metrics_dir = os.path.join(fold_dir, "slidewise_metrics")
        if not os.path.isdir(metrics_dir):
            continue

        for csv_file in glob(os.path.join(metrics_dir, "epoch_*.csv")):
            epoch_name = os.path.splitext(os.path.basename(csv_file))[0]  # "epoch_10"
            if epoch_name not in epoch_dict:
                epoch_dict[epoch_name] = []

            df = pd.read_csv(csv_file)
            # Insert or overwrite fold column
            df["fold"] = fold_idx
            epoch_dict[epoch_name].append(df)

    os.makedirs(save_dir, exist_ok=True)

    for epoch_name, df_list in epoch_dict.items():
        df_epoch = pd.concat(df_list, ignore_index=True)
        epoch_num = epoch_name.split('_')[1]
        xlsx_path = os.path.join(save_dir, f"epoch_{epoch_num}.xlsx")
        df_epoch.to_excel(xlsx_path, index=False)
        logging.info(f"[✓] Merged metrics saved to {xlsx_path}")



def run_fold(args, fold_idx):
    logging.info(f"\n========== Fold {fold_idx} / {args.dataset} ==========\n")

    # ----- ST npy paths (our_format) -----
    label_root = os.path.join(args.root_path, args.dataset)
    folds = get_folds(args.dataset)
    split = folds[fold_idx]
    train_files = [os.path.join(label_root, f) for f in split["train"]]
    val_files = [os.path.join(label_root, f) for f in split["val"]]

    # ----- Load records (no few-shot) ----- #
    train_records = load_records_from_npy_list(train_files)
    val_records = load_records_from_npy_list(val_files)
    logging.info(f"[Use Full Data] Fold {fold_idx}: {len(train_records)} samples for training.")

    # ----- Transforms ----- #
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ----- Dataset & Val loaders (split by slide, consistent with original structure) ----- #
    train_dataset = STDataset(train_records, args.patch_root, img_transform)
    val_dataset = STDataset(val_records, args.patch_root, img_transform)

    val_loaders_by_slide = {}
    slide_ids = sorted(list(set([r["sample_id"] for r in val_records])))
    for sid in slide_ids:
        idxs = [i for i, r in enumerate(val_records) if r["sample_id"] == sid]
        sub = torch.utils.data.Subset(val_dataset, idxs)
        val_loader = torch.utils.data.DataLoader(
            sub,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        val_loaders_by_slide[sid] = val_loader

    # ----- Paths (keep consistent) ----- #
    base = f"./weights/weights_types_{args.num_type}_topk_{args.topk_retrieval}/{args.dataset}/{args.total_ratio}"
    save_path = os.path.join(base, f"fold{fold_idx}")
    save_weight_path = os.path.join(save_path, "checkpoint_epoch")
    save_gt_path = os.path.join(save_path, "gt")
    save_pred_path = os.path.join(save_path, "pred")
    save_metrics_path = os.path.join(save_path, "slidewise_metrics")
    for p in [save_path, save_weight_path, save_gt_path, save_pred_path, save_metrics_path]:
        os.makedirs(p, exist_ok=True)

    writer = SummaryWriter(os.path.join(save_path, "log"))


    model = DenseRegressor().cuda()
    retrieval_part = retrieval_module(topk=args.topk_retrieval, num_type=args.num_type).cuda()
    retrieval_part.train()


    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    optimizer_ret = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)

    loss_fn = nn.MSELoss()

    # ====== Active RL Sampler ====== #
    embed_dim = train_records[0]["sc_embed"].shape[0]
    sc_path = os.path.join(args.sc_root, args.dataset, "merged_data.npy")
    sc_arr = np.load(sc_path, allow_pickle=True).tolist()
    sc_embed = np.array([x["embedding"] for x in sc_arr])
    sc_type = np.array([x["cell_type"] for x in sc_arr])
    assert sc_embed.shape[1] == embed_dim, "SC/ST embedding dim mismatch!"

    embeds = np.stack([r["sc_embed"] for r in train_records], axis=0)
    coords = np.stack([np.array(r["position"], dtype=np.float32)
                       for r in train_records], axis=0)
    raw_expr = np.stack([r["expression"] for r in train_records], axis=0)
    raw_ct   = np.stack([r["sc_type"] for r in train_records], axis=0)


    # ====== Calculate fixed ratio per round ====== #
    ssampling_cutoff = args.sampling_cutoff  # Sampling stops after this epoch

    effective_epochs = max(0, ssampling_cutoff - args.warmup_epochs)

    # Actual number of sampling rounds (satisfying sample_interval)
    num_rounds = max(1, effective_epochs // args.sample_interval) + 1

    # Per-round sampling ratio = total_ratio / actual number of rounds
    fixed_round_ratio = args.total_ratio / num_rounds

    sampler = RLSampler(
        embed_dim=embed_dim,
        sc_embed=sc_embed,
        sc_type=sc_type,
        total_ratio=args.total_ratio,
        per_round_ratio=fixed_round_ratio,  # This ratio is not used here, only total_ratio matters
        w_sc=args.w_sc,
        w_type=args.w_type,
        w_spatial=args.w_spatial
    )

    logging.info(
        f"[Sampling Strategy] total_ratio={args.total_ratio} | warmup={args.warmup_epochs} | "
        f"interval={args.sample_interval} | rounds={num_rounds}"
    )
    logging.info(
        f"[Sampling Each Round] {fixed_round_ratio:.4f} "
        f"({fixed_round_ratio*100:.2f}% of all spots ≈ {int(len(train_dataset)*fixed_round_ratio)} spots)"
    )

    # ====== Initialize sampling ====== #
    selected_indices = []
    selected_set = set()

    # ====== Training ====== #
    iter_num = 0
    max_epoch = args.max_epochs
    log_intv = 10
    rng = np.random.default_rng(args.seed + fold_idx)
    start_time = time.time()

    for epoch in range(max_epoch):
        logging.info(f"[Fold {fold_idx}] ===== Epoch {epoch + 1}/{max_epoch} =====")

        # ==== Number of samples per round ====
        per_round_k = max(1, int(len(train_dataset) * fixed_round_ratio))
        sampler.per_round_ratio = fixed_round_ratio  # Update sampler's ratio

        # ==== Default state when not sampling ====
        idx, reward, parts = [], 0.0, {"r_sc": 0.0, "r_type": 0.0, "r_sp": 0.0}

        # ==== Case 1: Random sampling in first round ====
        if epoch == 0:
            total_allow = int(len(train_dataset) * args.total_ratio)
            idx = rng.choice(len(train_dataset), size=min(per_round_k, total_allow), replace=False).tolist()
            logging.info(f"[Init Sampling] ratio={fixed_round_ratio:.4f}, k={per_round_k}")

        # ==== Case 2: Warm-up, no sampling ====
        elif epoch < args.warmup_epochs:
            logging.info(f"[Warmup] No sampling at epoch {epoch+1}")

        # ==== Case 3: Exceeded sampling cutoff, no more sampling ====
        elif epoch + 1 > ssampling_cutoff:
            logging.info(f"[Stop Sampling] Epoch {epoch+1} > {ssampling_cutoff}. No more sampling.")

        # ==== Case 4: RL sampling ====
        elif (epoch - args.warmup_epochs) % args.sample_interval == 0:
            logging.info(f"[RL Sampling] ratio={fixed_round_ratio:.4f}, k={per_round_k}")
            idx, reward, parts = sampler.sample_and_update(embeds, coords)

        else:
            logging.info(f"[Skip] No sampling at epoch {epoch+1}")

        # ---- Update selected ---- #
        new_idx = [int(i) for i in idx if i not in selected_set]
        selected_indices += new_idx
        selected_set.update(new_idx)
        all_expr = raw_expr[selected_indices]
        all_ct   = raw_ct[selected_indices]
        all_expr_tensor = torch.tensor(all_expr, dtype=torch.float32).cuda()

        logging.info(
            f"[Sample] Epoch {epoch + 1}: selected={len(selected_indices)} (+{len(new_idx)}) | "
            f"Reward={reward:.4f} | SC={(parts['r_sc']*args.w_sc):.4f} | TYPE={(parts['r_type']*args.w_type):.4f} | SPATIAL={(parts['r_sp']*args.w_spatial):.4f}"
        )

        # ---- Loader based on selected ---- #
        train_subset = torch.utils.data.Subset(train_dataset, selected_indices)
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        # ---- Train: consistent with original ---- #
        model.train()
        train_bar = tqdm(train_loader,
                         desc=f"Fold {fold_idx} | Epoch {epoch + 1}",
                         ncols=100)
        for batch in train_bar:
            images, gene, _, _ = batch
            images, gene = images.cuda(), gene.cuda()

            for pg in optimizer.param_groups:
                pg["lr"] = args.base_lr * (1 - iter_num / args.max_iterations) ** args.lr_decay

            feats, pred = model(images)
            loss = loss_fn(pred, gene)

            batch_reg = {
                "image": images, "expr": gene,'feature': feats,
                "all_expr": all_expr_tensor,
                "all_cell_type": all_ct
            }

            mask, pred_ret = retrieval_part.forward_retrieval(batch_reg)
            loss_kd = 0.01 * (loss_fn(pred, pred_ret) * mask).mean()

            # 3) Combined backward & update
            loss_total = loss + loss_kd
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()


            for pg_ret in optimizer_ret.param_groups:
                pg_ret["lr"] = 0.1*args.base_lr * (1 - iter_num / args.max_iterations) ** args.lr_decay

            batch_reg_ret = {
                "reduced_expression": gene,'feat': feats.detach(),
                "all_expr": all_expr_tensor,
                "all_cell_type": all_ct
            }

            loss_con = retrieval_part.forward_contrastive(batch_reg_ret)
            optimizer_ret.zero_grad()
            loss_con.backward()
            optimizer_ret.step()

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iter_num)
            writer.add_scalar("loss", loss.item(), iter_num)
            if iter_num % log_intv == 0:
                train_bar.set_postfix({
                    "Iter":   iter_num,
                    "MSE":    f"{loss.item():.4f}",
                    "KD":     f"{loss_kd.item():.4f}",
                    "Total":  f"{loss_total.item():.4f}",
                    "Con":    f"{loss_con.item():.4f}",
                    "LR":     f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            iter_num += 1
            if iter_num >= args.max_iterations:
                break

        # ---- Eval: consistent with original ---- #
        if (epoch + 1) % 10 == 0 or epoch == max_epoch - 1:
            model.eval()
            results = []
            for slide, vloader in tqdm(val_loaders_by_slide.items(),
                                       desc=f"Eval Fold {fold_idx} | Epoch {epoch + 1}",
                                       ncols=100):
                preds, targets = [], []
                with torch.no_grad():
                    for imgs, labs, _, _ in vloader:
                        _, out = model(imgs.cuda())
                        preds.append(out.cpu())
                        targets.append(labs.cpu())
                preds = torch.cat(preds).numpy()
                targets = torch.cat(targets).numpy()
                mse = np.mean((preds - targets) ** 2)
                mae = np.mean(np.abs(preds - targets))
                pcc = calculate_pcc(torch.tensor(preds), torch.tensor(targets)).item()

                epdir = os.path.join(save_pred_path, f"epoch_{epoch + 1}")
                os.makedirs(epdir, exist_ok=True)
                np.save(os.path.join(epdir, f"{slide}_pred.npy"), preds)
                np.save(os.path.join(save_gt_path, f"{slide}_gt.npy"), targets)

                logging.info(
                    f"[Epoch {epoch + 1} | {slide}] MSE:{mse:.4f}, PCC:{pcc:.4f}, MAE:{mae:.4f}"
                )
                results.append({
                    "slide": slide,
                    "epoch": epoch + 1,
                    "mse": mse,
                    "pcc": pcc,
                    "mae": mae
                })

            df = pd.DataFrame(results)
            csvp = os.path.join(save_metrics_path, f"epoch_{epoch + 1}.csv")
            df.to_csv(csvp, index=False)
            logging.info(f"Saved metrics CSV to {csvp}")
            model.train()

        if iter_num >= args.max_iterations:
            break

    elapsed = (time.time() - start_time) / 60
    logging.info(f"Fold {fold_idx} done in {elapsed:.2f} min.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HER2", "BC", "Kidney"], default="HER2")
    parser.add_argument("--root_path",
                        default="/data/zhuj28/MIDL2026/data_format/our_format_final")
    parser.add_argument("--patch_root",
                        default="/data/zhuj28/summer_intern/data/processed_data/patches")
    parser.add_argument("--sc_root",
                        default="/data/zhuj28/MIDL2026/data_format/sc_embedding")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_iterations", type=int, default=10000000000)
    parser.add_argument("--base_lr", type=float, default=5e-5)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--sample_interval", type=int, default=2)
    parser.add_argument("--sampling_cutoff", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)

    # RL hyperparameters
    parser.add_argument("--total_ratio", type=float, choices=[0.15, 0.3, 0.5, 0.75, 1.0],  default=0.5)
    parser.add_argument("--w_sc", type=float, default=20)
    parser.add_argument("--w_type", type=float, default=5)
    parser.add_argument("--w_spatial", type=float, default=0.05)

    parser.add_argument("--topk_retrieval", type=int, default=100)
    parser.add_argument("--num_type", type=int, default=10)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    root_dir = f"./weights/weights_types_{args.num_type}_topk_{args.topk_retrieval}/{args.dataset}/{args.total_ratio}"
    # run 4 folds
    for fi in range(4):
        run_fold(args, fi)

    # merge results: structure unchanged, reuse directly
    merge_epoch_metrics_to_excel(
        root_dir=root_dir,
        save_dir=os.path.join(root_dir, "merged_metrics")
    )


if __name__ == "__main__":
    main()
