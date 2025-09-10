import os, ast, math, argparse
from typing import List
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# -------------------------
# Utils
# -------------------------

def safe_eval_dict(raw):
    if isinstance(raw, str) and raw.strip().startswith("{") and raw.strip().endswith("}"):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return {}

def concat_text_fields(row: pd.Series, text_cols: List[str], sep="; "):
    parts = []
    for c in text_cols:
        if c not in row:
            continue
        v = row[c]
        if isinstance(v, str) and v.strip().startswith("{"):
            d = safe_eval_dict(v)
            if d:
                parts.append("; ".join(f"{k}: {d[k]}" for k in d))
        elif pd.isna(v):
            continue
        else:
            s = str(v).strip()
            if s:
                parts.append(s)
    joined = sep.join([p for p in parts if p])
    return joined if joined else ""  # may be empty; we’ll handle fallback

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    # avoids NaNs and ensures unit vectors when possible
    return x / torch.clamp(x.norm(p=2, dim=dim, keepdim=True), min=eps)

# -------------------------
# Spatial Encoders
# -------------------------

class MLPSpatial(nn.Module):
    def __init__(self, output_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, lonlat):
        return self.net(lonlat.float())


class SinusoidalSpatial(nn.Module):
    """
    Encode lon/lat with periodic features to respect wrap-around.
    Uses multiple frequency bands for richer locality signals.
    """
    def __init__(self, output_dim=128, hidden_dim=128, bands=(1.0, 2.0, 4.0)):
        super().__init__()
        self.register_buffer("_bands", torch.tensor(bands, dtype=torch.float32))
        in_dim = 4 * len(bands)  # sin(lon*b), cos(lon*b), sin(lat*b), cos(lat*b) for each band
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, lonlat):
        # lonlat in degrees -> radians
        lonlat = lonlat.float()
        lon_rad = torch.deg2rad(lonlat[..., 0])
        lat_rad = torch.deg2rad(lonlat[..., 1])

        # shape handling: (B,) -> (B, 1)
        lon_rad = lon_rad.unsqueeze(-1)  # (B,1)
        lat_rad = lat_rad.unsqueeze(-1)  # (B,1)

        # (B, nbands)
        lon_b = lon_rad * self._bands
        lat_b = lat_rad * self._bands

        # periodic features
        feats = torch.cat([
            torch.sin(lon_b), torch.cos(lon_b),
            torch.sin(lat_b), torch.cos(lat_b)
        ], dim=-1)  # (B, 4*nbands)

        return self.net(feats)


class MercatorSpatial(nn.Module):
    """
    Map (lon,lat) to a Mercator-like (x,y) then MLP.
    x = lon_rad
    y = ln(tan(pi/4 + lat_rad/2))   with latitude clamped to avoid poles
    """
    def __init__(self, output_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, lonlat):
        lonlat = lonlat.float()
        lon_deg = lonlat[..., 0]
        lat_deg = lonlat[..., 1]

        # clamp latitude to avoid infinities near the poles
        # Web Mercator typically clamps around ~85.05113°
        lat_deg = torch.clamp(lat_deg, -85.05112878, 85.05112878)

        lon_rad = torch.deg2rad(lon_deg)
        lat_rad = torch.deg2rad(lat_deg)

        # y = ln(tan(pi/4 + lat/2))
        y = torch.log(torch.tan(math.pi / 4.0 + lat_rad / 2.0))
        x = lon_rad
        xy = torch.stack([x, y], dim=-1)  # (B,2)

        return self.net(xy)


def build_spatial_encoder(kind: str, output_dim: int, hidden: int) -> nn.Module:
    kind = kind.lower()
    if kind == "mlp":
        return MLPSpatial(output_dim=output_dim, hidden_dim=hidden)
    if kind == "sin":
        # You can tweak bands for your dataset scale
        return SinusoidalSpatial(output_dim=output_dim, hidden_dim=hidden, bands=(1.0, 2.0, 4.0, 8.0))
    if kind == "mercator":
        return MercatorSpatial(output_dim=output_dim, hidden_dim=hidden)
    raise ValueError(f"Unknown spatial-encoder: {kind}")


# -------------------------
# Contrastive Model
# -------------------------

class ContrastiveModel(nn.Module):
    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proj_dim: int = 128,
        spatial_encoder: str = "mlp",
        spatial_hidden: int = 128,
        freeze_text: bool = True,
        w_text: float = 1.0,
        w_spatial: float = 1.0,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.loc_encoder = build_spatial_encoder(spatial_encoder, proj_dim, spatial_hidden)

        # CLIP-style temperature; will learn, but clamp its range during use
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))
        self.w_text = float(w_text)
        self.w_spatial = float(w_spatial)
    @staticmethod
    def mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return (last_hidden * mask).sum(dim=1) / denom
    
    def _fuse(self, zt, zl):
        # zt, zl are already unit vectors; weight then renormalize
        z = self.w_text * zt + self.w_spatial * zl
        return F.normalize(z, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        z = self.text_proj(pooled)
        return safe_normalize(z)

    def encode_coords(self, lonlat):
        z = self.loc_encoder(lonlat.float())
        return safe_normalize(z)

    def forward(self, lonlat, input_ids, attention_mask):
        z_t = self.encode_text(input_ids, attention_mask)   # (B,D)
        z_l = self.encode_coords(lonlat)                    # (B,D)
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits = scale * (z_t @ z_l.T)
        return logits, logits.T


# -------------------------
# Dataset
# -------------------------

class POIDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        text_cols: List[str],
        lon_col: str = "lon",
        lat_col: str = "lat",
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_len: int = 64,
    ):
        df = pd.read_csv(csv_path)
        # basic hygiene
        df = df.dropna(subset=[lon_col, lat_col])
        df = df[(df[lon_col].astype(str) != "") & (df[lat_col].astype(str) != "")]
        # build text once
        df["__text"] = df.apply(lambda r: concat_text_fields(r, text_cols), axis=1)
        # fallback to a lightweight pseudo-text so the encoder NEVER sees empty input
        # (prevents zero vectors that kill gradients)
        fallback = "poi"
        df["__text"] = df["__text"].apply(lambda s: s if isinstance(s, str) and s.strip() else fallback)

        self.df = df.reset_index(drop=True)
        self.lon_col, self.lat_col = lon_col, lat_col
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["__text"]
        lon = float(row[self.lon_col]); lat = float(row[self.lat_col])
        enc = self.tok(
            text, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_len,
        )
        return {
            "lonlat": torch.tensor([lon, lat], dtype=torch.float32),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

def collate(batch):
    lonlat = torch.stack([b["lonlat"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return lonlat, input_ids, attention_mask

# -------------------------
# Train / Encode / Bench
# -------------------------

def symmetric_info_nce_loss(logits_t2l, logits_l2t):
    B = logits_t2l.size(0)
    labels = torch.arange(B, device=logits_t2l.device)
    ce = nn.CrossEntropyLoss()
    return 0.5 * (ce(logits_t2l, labels) + ce(logits_l2t, labels))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = POIDataset(args.csv, args.text_cols, args.lon_col, args.lat_col, args.text_encoder, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)

    model = ContrastiveModel(
        text_encoder_name=args.text_encoder,
        proj_dim=args.proj_dim,
        spatial_encoder=args.spatial_encoder,
        spatial_hidden=args.spatial_hidden,
        freeze_text=args.freeze_text,
        w_text=args.w_text,            # NEW
        w_spatial=args.w_spatial,      # NEW
    ).to(device)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for lonlat, input_ids, attention_mask in dl:
            lonlat, input_ids, attention_mask = lonlat.to(device), input_ids.to(device), attention_mask.to(device)
            optim.zero_grad(set_to_none=True)
            logits_t2l, logits_l2t = model(lonlat, input_ids, attention_mask)
            loss = symmetric_info_nce_loss(logits_t2l, logits_l2t)
            loss.backward()
            optim.step()
            running += loss.item()
        avg = running / max(1, len(dl))
        print(f"epoch {epoch:03d} | loss {avg:.4f} | temp={model.logit_scale.exp().item():.3f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"✅ saved → {args.checkpoint}")

def encode(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = POIDataset(args.csv, args.text_cols, args.lon_col, args.lat_col, args.text_encoder, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = ContrastiveModel(
        text_encoder_name=args.text_encoder,
        proj_dim=args.proj_dim,
        spatial_encoder=args.spatial_encoder,
        spatial_hidden=args.spatial_hidden,
        freeze_text=args.freeze_text,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_text, all_coord = [], []
    with torch.no_grad():
        for lonlat, input_ids, attention_mask in dl:
            lonlat, input_ids, attention_mask = lonlat.to(device), input_ids.to(device), attention_mask.to(device)
            zt = model.encode_text(input_ids, attention_mask)
            zl = model.encode_coords(lonlat)
            all_text.append(zt.cpu().numpy()); all_coord.append(zl.cpu().numpy())

    text_np = np.vstack(all_text); coord_np = np.vstack(all_coord)
    os.makedirs(os.path.dirname(args.text_out) or ".", exist_ok=True)
    np.save(args.text_out, text_np); np.save(args.coord_out, coord_np)
    print(f"💾 text → {args.text_out}  {text_np.shape}")
    print(f"💾 coord → {args.coord_out} {coord_np.shape}")

def bench(args):
    from sklearn.metrics.pairwise import cosine_similarity
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    df = pd.read_csv(args.csv)
    fused = np.load(args.fused)

    model = ContrastiveModel(
        text_encoder_name=args.text_encoder,
        proj_dim=args.proj_dim,
        spatial_encoder=args.spatial_encoder,
        spatial_hidden=args.spatial_hidden,
        freeze_text=args.freeze_text,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.text_encoder)

    print("=== Bench ===")
    with torch.no_grad():
        for q in args.queries:
            text, lat_s, lon_s = q.split("|")
            lat, lon = float(lat_s), float(lon_s)
            enc = tok(text if text.strip() else "poi", return_tensors="pt",
                      padding="max_length", truncation=True, max_length=args.max_len)
            input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            lonlat = torch.tensor([[lon, lat]], dtype=torch.float32, device=device)

            z = model._fuse(model.encode_text(input_ids, attention_mask), model.encode_coords(lonlat))

            sims = cosine_similarity(z.cpu().numpy(), fused)[0]
            top_k = np.argsort(sims)[-args.topk:][::-1]

            print(f"\n--- Query: '{text}' ({lat:.5f},{lon:.5f}) ---")
            for rank, idx in enumerate(top_k, 1):
                row = df.iloc[idx]
                print(f"#{rank:02d} score={sims[idx]:.4f} id={row.get('id', idx)} tags={row.get('tags','')}")

# -------------------------
# CLI
# -------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Contrastive Text↔Location")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp):
        sp.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
        sp.add_argument("--proj-dim", type=int, default=128)
        sp.add_argument("--spatial-encoder", choices=["mlp", "sin", "mercator"], default="mlp")
        sp.add_argument("--spatial-hidden", type=int, default=128)
        sp.add_argument("--freeze-text", action="store_true")
        sp.add_argument("--max-len", type=int, default=64)
        sp.add_argument("--lon-col", default="lon")
        sp.add_argument("--lat-col", default="lat")
        sp.add_argument("--text-cols", nargs="+", required=True)
        sp.add_argument("--w-text", type=float, default=1.0)
        sp.add_argument("--w-spatial", type=float, default=1.0)

    sp_tr = sub.add_parser("train")
    sp_tr.add_argument("csv"); add_shared(sp_tr)
    sp_tr.add_argument("--epochs", type=int, default=10)
    sp_tr.add_argument("--batch-size", type=int, default=64)
    sp_tr.add_argument("--lr", type=float, default=1e-4)
    sp_tr.add_argument("--wd", type=float, default=0.01)
    sp_tr.add_argument("--workers", type=int, default=4)
    sp_tr.add_argument("--checkpoint", default="contrastive/contrastive_model.pt")
    sp_tr.add_argument("--cpu", action="store_true")
    sp_tr.set_defaults(func=train)

    sp_en = sub.add_parser("encode")
    sp_en.add_argument("csv"); sp_en.add_argument("text_out"); sp_en.add_argument("coord_out")
    add_shared(sp_en)
    sp_en.add_argument("--batch-size", type=int, default=256)
    sp_en.add_argument("--workers", type=int, default=4)
    sp_en.add_argument("--checkpoint", required=True)
    sp_en.add_argument("--cpu", action="store_true")
    sp_en.set_defaults(func=encode)

    sp_be = sub.add_parser("bench")
    sp_be.add_argument("csv"); sp_be.add_argument("fused")
    add_shared(sp_be)
    sp_be.add_argument("--checkpoint", required=True)
    sp_be.add_argument("--queries", nargs="+", default=[
        "Where can I buy medicine?|-37.8000|144.9710",
        "Good restaurant|-37.8670|144.9780",
    ])
    sp_be.add_argument("--topk", type=int, default=10)
    sp_be.add_argument("--cpu", action="store_true")
    sp_be.set_defaults(func=bench)
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
