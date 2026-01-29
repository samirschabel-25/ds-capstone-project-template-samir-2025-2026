#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =========================================================
# NOTEBOOK 2 (FRONT+BACK Pflicht) — FULL CODE (ROBUST + PDF PLOTS ONLY)
# ✅ Zwei Modelle (Front/Back) + kNN pro Seite
# ✅ Combined kNN Cache (Front+Back Embedding) als Tie-Breaker bei Disagreement
# ✅ FINAL = Combined, aber: wenn Back != Front → kNN/Guardrails entscheiden
# ✅ Grades: Back, Front, Combined (je 1–10)
# ✅ PDF Report: 1 Seite (Badge + PLOTS ONLY) — keine Tabellen, keine Sources
# =========================================================

# =========================
# Imports & Settings
# =========================
import os, io, json, uuid, hashlib, socket, traceback, datetime
from collections import Counter
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image, ImageOps

import psycopg2
import psycopg2.extras
import gradio as gr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth


# ====== DB Table ======
TABLE = "pokemon_card_samples_fb"

# ====== Labels ======
LABELS = ["NM", "EX", "GD", "LP", "PL", "PO"]
LABEL2IDX = {l:i for i,l in enumerate(LABELS)}
IDX2LABEL = {i:l for l,i in LABEL2IDX.items()}
LABEL_IDX = LABEL2IDX

IMG_H = 352
IMG_W = 256

# ====== Multi-View ======
CROP_FRAC = 0.62
AUG_FULL = True
AUG_CORNERS = True

# Inferenz-stabiler: Jitter runter, Inference-AUG aus
USE_JITTER = True
N_JITTER = 2
JITTER_MAX_PX = 4

# ====== Embedding / kNN ======
TOPK = 10
EMB_DIM = 128
KNN_K = 10   # ✅ statt 7: wir nutzen wirklich die 10 nächsten

# ====== Embedding: nur Rand/Ecken (gegen Artwork-Ähnlichkeit) ======
EMB_RING_FRAC_BACK  = 0.18   # Anteil je Seite, der als "Rand" bleibt
EMB_RING_FRAC_FRONT = 0.20   # Front etwas dicker, weil Artwork/Holo stark stört
EMB_FILL_VALUE = 128         # Grau füllen (neutral)


# Per-Side Fusion (nur wenn CNN unsicher)
USE_KNN_FUSION = True
FUSION_CONF_TH = 0.50

# kNN Mischung in Side-Probs (für Combined-Reporting)
KNN_MIX_BASE = 0.12

# kNN Override nur bei wirklich guter Ähnlichkeit
KNN_FUSION_MIN_BEST_SIM = 0.78
KNN_FUSION_MIN_CONF = 0.45
KNN_MIX_MIN_BEST_SIM = 0.72

# ====== Ensemble ======
ENSEMBLE_RUNS = 10
DETERMINISTIC_PER_IMAGE = True
SEED_STEP = 10007

# ====== Training ======
BATCH_SIZE = 32
EPOCHS_HEAD = 6
EPOCHS_FINETUNE = 6
LR_HEAD = 1e-3
LR_FINETUNE = 2e-4

MODEL_PATH_BACK  = "card_back_condition_model.keras"
MODEL_PATH_FRONT = "card_front_condition_model.keras"

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ====== Preprocess ======
QUAD_EXPAND_BACK  = 1.04
QUAD_EXPAND_FRONT = 1.03
CORNER_RADIUS_PX = int(round(IMG_W * 0.05))
APPLY_ROUNDED_MASK_TO_IMAGE = False

# ====== Combined Decision ======
WEIGHT_BACK  = 0.50
WEIGHT_FRONT = 0.50

# ====== Local Retrieval (Top-10 Combined Neighbors) ======
USE_LOCAL_TOP10 = True

LOCAL_TOPK = 10                 # genau 10 Nachbarn
LOCAL_PRIOR_POWER = 3.0         # sim^power -> betont sehr ähnliche stärker
LOCAL_MIN_SIM_FOR_MIX = 0.75    # ab welcher best_sim wir überhaupt mischen
LOCAL_MAX_MIX = 0.35            # max Anteil Neighbor-Verteilung im Final


# ✅ Wenn Back != Front → Combined-Tiebreak durch kNN/Guardrails
COMBINED_TIEBREAK_BY_KNN = True

# Inference Augmentierung (bei Ausreißern: AUS)
INFER_AUG = False


# ====== Robustness / Guardrails ======
VIEW_TRIM_Q = 0.20                # 20% der schlimmsten/besten Views wegtrimmen
MAX_OUTLIER_JUMP = 2              # mehr als 2 Klassen Sprung -> nur bei starker Evidenz erlauben

EXTREME_ALLOW_CNN_CONF = 0.86     # wenn CNN sehr sicher ist, darf auch extrem sein
EXTREME_ALLOW_SIM = 0.90          # oder wenn kNN sehr ähnlich ist...
EXTREME_ALLOW_KNN_CONF = 0.60     # ...und kNN konsistent ist

DISAGREE_HARD_DIFF = 3            # NM vs PL/PO etc. => "harte" Diskrepanz
DISAGREE_REQUIRE_SIM = 0.86       # combined-kNN wird nur genutzt, wenn best_sim hoch
DISAGREE_REQUIRE_CONF = 0.55      # und combined-kNN conf ausreichend ist

REQUIRE_EXTRACTION = True         # wenn Warp/Quad nicht sauber → lieber abbrechen statt Mist klassifizieren


# =========================
# DB Verbindung + Schema
# =========================
def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5434")),
        dbname=os.getenv("PGDATABASE", "sam1988"),
        user=os.getenv("PGUSER", "sam1988"),
        password=os.getenv("PGPASSWORD", "")
    )

def ensure_schema():
    labels_sql = ",".join([f"'{l}'" for l in LABELS])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                label TEXT NOT NULL CHECK (label IN ({labels_sql})),
                note TEXT,

                -- BACK (Pflicht)
                back_raw_sha256 TEXT NOT NULL UNIQUE,
                back_raw_format TEXT NOT NULL,
                back_raw_w INT,
                back_raw_h INT,
                back_raw_bytes BYTEA NOT NULL,

                back_proc_format TEXT NOT NULL,
                back_proc_w INT NOT NULL,
                back_proc_h INT NOT NULL,
                back_proc_bytes BYTEA NOT NULL,

                back_proc_mask_format TEXT NOT NULL DEFAULT 'png',
                back_proc_mask_w INT NOT NULL DEFAULT {IMG_W},
                back_proc_mask_h INT NOT NULL DEFAULT {IMG_H},
                back_proc_mask_bytes BYTEA,

                back_proc_method TEXT,
                back_proc_quad_expand REAL,

                -- FRONT (Pflicht)
                front_raw_sha256 TEXT NOT NULL UNIQUE,
                front_raw_format TEXT NOT NULL,
                front_raw_w INT,
                front_raw_h INT,
                front_raw_bytes BYTEA NOT NULL,

                front_proc_format TEXT NOT NULL,
                front_proc_w INT NOT NULL,
                front_proc_h INT NOT NULL,
                front_proc_bytes BYTEA NOT NULL,

                front_proc_mask_format TEXT NOT NULL DEFAULT 'png',
                front_proc_mask_w INT NOT NULL DEFAULT {IMG_W},
                front_proc_mask_h INT NOT NULL DEFAULT {IMG_H},
                front_proc_mask_bytes BYTEA,

                front_proc_method TEXT,
                front_proc_quad_expand REAL
            );
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_label ON {TABLE}(label);")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_created_at ON {TABLE}(created_at DESC);")

def db_counts():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT label, COUNT(*)::int AS n
                FROM {TABLE}
                GROUP BY label
                ORDER BY label;
            """)
            rows = cur.fetchall()
    counts = {r["label"]: r["n"] for r in rows}
    for l in LABELS:
        counts.setdefault(l, 0)
    counts["TOTAL"] = sum(counts[l] for l in LABELS)
    return counts

def fmt_pg_error(e: Exception) -> str:
    if isinstance(e, psycopg2.Error):
        parts = [f"{type(e).__name__}: {e}"]
        if getattr(e, "pgcode", None): parts.append(f"pgcode: {e.pgcode}")
        if getattr(e, "pgerror", None): parts.append(f"pgerror: {e.pgerror}")
        diag = getattr(e, "diag", None)
        if diag is not None:
            for k in ["message_detail","message_hint","schema_name","table_name","column_name","constraint_name"]:
                v = getattr(diag, k, None)
                if v: parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return f"{type(e).__name__}: {e}"

ensure_schema()
print(f"✅ Schema OK ({TABLE}). Counts:", db_counts())


# =========================
# Utils (Images / Views) + PREPROCESS (Warp)
# =========================
def proc_png_bytes_to_np(proc_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(proc_bytes)).convert("RGB")
    arr = np.array(pil, dtype=np.uint8)
    if arr.shape[:2] != (IMG_H, IMG_W):
        arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return arr

def resize_to_base(img: np.ndarray) -> np.ndarray:
    if img.shape[:2] != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return img

def mask_center_keep_border(img_uint8: np.ndarray, ring_frac: float, fill: int = 128) -> np.ndarray:
    """
    Neutralisiert die Bildmitte, lässt nur einen Rand-Ring stehen.
    Dadurch wird kNN/Embedding condition-orientiert (Whitening/Kanten),
    statt artwork-orientiert.
    """
    img = img_uint8.copy()
    H, W = img.shape[:2]
    t = int(round(min(H, W) * float(ring_frac)))
    y0, y1 = t, H - t
    x0, x1 = t, W - t
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = int(fill)
    return img


def seed_from_image_uint8(arr_uint8: np.ndarray) -> int:
    h = hashlib.sha256(arr_uint8.tobytes()).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def jitter_perspective(img: np.ndarray, max_px: int = 6, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    H, W = img.shape[:2]
    src = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    j = rng.integers(-max_px, max_px+1, size=(4,2)).astype(np.float32)
    dst = np.clip(src + j, [0,0], [W-1,H-1]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out

def make_views(img_uint8: np.ndarray, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()

    img = resize_to_base(img_uint8)
    H, W = img.shape[:2]
    ch = int(round(H * CROP_FRAC))
    cw = int(round(W * CROP_FRAC))

    def crop(y0, x0, y1, x1):
        c = img[y0:y1, x0:x1]
        return resize_to_base(c)

    full = img
    tl = crop(0, 0, ch, cw)
    tr = crop(0, W-cw, ch, W)
    bl = crop(H-ch, 0, H, cw)
    br = crop(H-ch, W-cw, H, W)

    y_mid0 = (H - ch)//2
    x_mid0 = (W - cw)//2
    top = crop(0, x_mid0, ch, x_mid0+cw)
    bottom = crop(H-ch, x_mid0, H, x_mid0+cw)
    left = crop(y_mid0, 0, y_mid0+ch, cw)
    right = crop(y_mid0, W-cw, y_mid0+ch, W)

    base = [
        ("full", full),
        ("corner_tl", tl), ("corner_tr", tr), ("corner_bl", bl), ("corner_br", br),
        ("edge_top", top), ("edge_bottom", bottom), ("edge_left", left), ("edge_right", right),
    ]

    def aug(name, v):
        return [
            (name, v),
            (name + "_hflip", cv2.flip(v, 1)),
            (name + "_vflip", cv2.flip(v, 0)),
            (name + "_rot180", cv2.rotate(v, cv2.ROTATE_180)),
        ]

    views = []
    views.extend(base)
    if AUG_FULL:
        views.extend(aug("full", full)[1:])
    if AUG_CORNERS:
        for nm, v in [("corner_tl", tl), ("corner_tr", tr), ("corner_bl", bl), ("corner_br", br)]:
            views.extend(aug(nm, v)[1:])
    if USE_JITTER:
        for j in range(N_JITTER):
            views.append((f"full_jitter{j+1}", jitter_perspective(full, max_px=JITTER_MAX_PX, rng=rng)))

    imgs  = [im for _,im in views]
    names = [n for n,_ in views]
    return imgs, names

def infer_augment_np(x_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not INFER_AUG:
        return x_uint8.astype(np.float32)
    y = x_uint8.astype(np.float32)
    y += float(rng.uniform(-8, 8))  # brightness
    c = float(rng.uniform(0.95, 1.05))  # contrast
    y = (y - 128.0) * c + 128.0
    y += rng.normal(0, 2.0, size=y.shape).astype(np.float32)  # noise
    return np.clip(y, 0, 255).astype(np.float32)

# ---- Warp/Normalize ----
def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 92) -> bytes:
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def bgr_from_pil(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # tl
    rect[2] = pts[np.argmax(s)]      # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # tr
    rect[3] = pts[np.argmax(diff)]   # bl
    return rect

# ---- Quad validation (verhindert falsche Warps auf Hintergrund) ----
AR_MIN, AR_MAX = 0.55, 0.90             # Pokemon-Karte ~0.72 (min/max erlaubt etwas Spielraum)
MIN_AREA_RATIO_BACK  = 0.12
MIN_AREA_RATIO_FRONT = 0.18

def quad_is_reasonable(bgr: np.ndarray, quad: np.ndarray, min_area_ratio: float) -> bool:
    q = order_points(np.asarray(quad, dtype=np.float32))
    H, W = bgr.shape[:2]
    area = abs(cv2.contourArea(q))
    if area < (min_area_ratio * H * W):
        return False

    rect = cv2.minAreaRect(q)
    rw, rh = rect[1]
    if rw <= 1 or rh <= 1:
        return False

    ar = min(rw, rh) / max(rw, rh)  # 0..1
    if not (AR_MIN <= ar <= AR_MAX):
        return False

    return True

def expand_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    q = quad.astype(np.float32)
    c = q.mean(axis=0, keepdims=True)
    return (c + (q - c) * scale).astype(np.float32)

def warp_quad(bgr, quad, out_w=IMG_W, out_h=IMG_H):
    rect = order_points(quad)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        bgr, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

def overlay_quad_debug(bgr, quad):
    dbg = bgr.copy()
    q = order_points(quad).astype(int)
    cv2.polylines(dbg, [q], isClosed=True, color=(0, 255, 0), thickness=4)
    for (x, y) in q:
        cv2.circle(dbg, (x, y), 10, (0, 0, 255), -1)
    return dbg

def try_extract_by_blue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([80, 40, 40], dtype=np.uint8)
    upper = np.array([150, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.10 * (bgr.shape[0] * bgr.shape[1]):
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(-1, 2)

    rect = cv2.minAreaRect(cnt)
    return cv2.boxPoints(rect)

def try_extract_by_edges(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 0.10 * (bgr.shape[0] * bgr.shape[1]):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2)
    return None

def try_extract_by_adaptive_thresh(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 7
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.10 * (bgr.shape[0] * bgr.shape[1]):
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(-1, 2)

    rect = cv2.minAreaRect(cnt)
    return cv2.boxPoints(rect)

def normalize_to_target(bgr, side: str, out_w=IMG_W, out_h=IMG_H):
    if side == "back":
        candidates = [("blue_mask", try_extract_by_blue_mask), ("edges", try_extract_by_edges), ("adaptive", try_extract_by_adaptive_thresh)]
        quad_expand = QUAD_EXPAND_BACK
        min_area_ratio = MIN_AREA_RATIO_BACK
    else:
        candidates = [("edges", try_extract_by_edges), ("adaptive", try_extract_by_adaptive_thresh), ("blue_mask", try_extract_by_blue_mask)]
        quad_expand = QUAD_EXPAND_FRONT
        min_area_ratio = MIN_AREA_RATIO_FRONT

    quad = None
    method = None
    for nm, fn in candidates:
        q = fn(bgr)
        if q is None:
            continue
        if not quad_is_reasonable(bgr, q, min_area_ratio=min_area_ratio):
            continue
        quad = q
        method = nm
        break

    if quad is None:
        resized = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return resized, "fallback_resize", False, None, quad_expand

    quad = expand_quad(np.array(quad), scale=quad_expand)
    dbg = overlay_quad_debug(bgr, quad)
    warped = warp_quad(bgr, quad, out_w=out_w, out_h=out_h)

    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    warped = cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return warped, f"{method}_expand{quad_expand}", True, dbg, quad_expand

def encode_png_bytes_gray(gray: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("PNG encoding failed (mask)")
    return enc.tobytes()

def rounded_rect_mask(h: int, w: int, r: int) -> np.ndarray:
    r = int(max(0, r))
    r = min(r, min(h, w) // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    if r == 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r - 1, r), r, 255, -1)
    cv2.circle(mask, (r, h - r - 1), r, 255, -1)
    cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, -1)
    return mask

def preprocess_pil_to_proc_rgb(pil_img: Image.Image, side: str):
    pil_fixed = ImageOps.exif_transpose(pil_img).convert("RGB")
    bgr = bgr_from_pil(pil_fixed)
    proc_bgr, method, extracted, dbg_bgr, quad_expand = normalize_to_target(bgr, side=side, out_w=IMG_W, out_h=IMG_H)

    mask = rounded_rect_mask(IMG_H, IMG_W, CORNER_RADIUS_PX)
    if APPLY_ROUNDED_MASK_TO_IMAGE:
        proc_bgr = cv2.bitwise_and(proc_bgr, proc_bgr, mask=mask)

    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    dbg_rgb = cv2.cvtColor(dbg_bgr, cv2.COLOR_BGR2RGB) if dbg_bgr is not None else None

    meta = {"method": method, "extracted": extracted, "quad_expand": float(quad_expand)}
    return proc_rgb.astype(np.uint8), mask, dbg_rgb, meta


# =========================
# Training Data Fetch (Side)
# =========================
def fetch_training_samples(side: str):
    side_col = "back_proc_bytes" if side == "back" else "front_proc_bytes"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, label, {side_col} AS proc_bytes
                FROM {TABLE}
                WHERE {side_col} IS NOT NULL AND label IS NOT NULL
            """)
            rows = cur.fetchall()

    X, y, ids = [], [], []
    for r in rows:
        lbl = r["label"]
        if lbl not in LABELS:
            continue
        arr = proc_png_bytes_to_np(bytes(r["proc_bytes"]))
        X.append(arr)
        y.append(LABEL2IDX[lbl])
        ids.append(str(r["id"]))

    if len(X) == 0:
        raise RuntimeError(f"Keine Trainingsdaten gefunden für side={side} ({side_col}/label).")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    return X, y, ids

def make_splits(X, y, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(X)*val_ratio)))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return (X[tr_idx], y[tr_idx]), (X[val_idx], y[val_idx])

def augment_tf(img):
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.90, 1.10)
    return img

def make_ds(X, y, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 2000), reshuffle_each_iteration=True)
    ds = ds.map(lambda a,b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda a,b: (augment_tf(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# Model bauen/laden (+ optional trainieren) pro Side
# =========================
def build_model(name: str):
    inp = keras.Input(shape=(IMG_H, IMG_W, 3), name="img")
    x = keras.applications.mobilenet_v2.preprocess_input(inp)

    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    emb = layers.Dense(EMB_DIM, activation=None, name="embedding")(x)
    emb = layers.LayerNormalization()(emb)

    out = layers.Dense(len(LABELS), activation="softmax", name="class")(emb)
    return keras.Model(inp, out, name=name)

def find_mobilenet_submodel(m: keras.Model):
    for l in m.layers:
        if isinstance(l, keras.Model) and hasattr(l, "layers"):
            name = (l.name or "").lower()
            if "mobilenet" in name and len(l.layers) > 20:
                return l
    return None

def finetune_with_submodel(m: keras.Model, base: keras.Model, unfreeze_from_ratio=0.70):
    base.trainable = True
    n = len(base.layers)
    cut = int(n * unfreeze_from_ratio)
    for i, l in enumerate(base.layers):
        if isinstance(l, keras.layers.BatchNormalization):
            l.trainable = False
        else:
            l.trainable = (i >= cut)

    m.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

def load_or_train_model(side: str, path: str):
    if os.path.exists(path):
        m = keras.models.load_model(path)
        print(f"✅ Loaded {side} model:", path)
        return m

    try:
        X, y, ids = fetch_training_samples(side)
    except Exception as e:
        print(f"⚠️ Keine Trainingsdaten für {side}. Fallback später. Details:", e)
        return None

    print(f"Loaded training {side}:", X.shape, y.shape)
    (X_tr, y_tr), (X_va, y_va) = make_splits(X, y, val_ratio=0.15)
    train_ds = make_ds(X_tr, y_tr, training=True)
    val_ds   = make_ds(X_va, y_va, training=False)

    # ✅ Class Weights gegen Imbalance
    cnt = Counter(y_tr.tolist())
    present = [k for k,v in cnt.items() if v > 0]
    cw = {k: (len(y_tr) / (len(present) * cnt[k])) for k in present}
    print("Class weights:", {IDX2LABEL[k]: round(v,3) for k,v in cw.items()})

    m = build_model(name=f"card_{side}_condition")
    m.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")]
    m.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks, class_weight=cw)

    base = find_mobilenet_submodel(m)
    if base is not None:
        finetune_with_submodel(m, base, 0.70)
        m.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks, class_weight=cw)

    m.save(path)
    print(f"✅ Saved {side} model:", path)
    return m

model_back  = load_or_train_model("back",  MODEL_PATH_BACK)
model_front = load_or_train_model("front", MODEL_PATH_FRONT)

if model_back is None and model_front is None:
    raise RuntimeError("Weder BACK noch FRONT Modell verfügbar. Bitte Trainingsdaten speichern oder Model-Dateien bereitstellen.")
if model_back is None:
    print("⚠️ BACK Modell fehlt → nutze FRONT Modell als Fallback.")
    model_back = model_front
if model_front is None:
    print("⚠️ FRONT Modell fehlt → nutze BACK Modell als Fallback.")
    model_front = model_back

embedder_back  = keras.Model(inputs=model_back.input,  outputs=model_back.get_layer("embedding").output,  name="embedder_back")
embedder_front = keras.Model(inputs=model_front.input, outputs=model_front.get_layer("embedding").output, name="embedder_front")


# =========================
# Embedding Utils
# =========================
def l2_normalize(v: np.ndarray, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def embed_images_np(embedder: keras.Model, imgs_uint8: np.ndarray, batch=64) -> np.ndarray:
    embs = []
    for i in range(0, len(imgs_uint8), batch):
        x = imgs_uint8[i:i+batch].astype(np.float32)
        e = embedder.predict(x, verbose=0)
        embs.append(e)
    embs = np.concatenate(embs, axis=0)
    return l2_normalize(embs)


# =========================
# Reference Cache (kNN) - BACK / FRONT / COMBINED
# =========================
REF = {
    "back":     {"emb": None, "meta": None},
    "front":    {"emb": None, "meta": None},
    "combined": {"emb": None, "meta": None},
}

def fetch_reference_rows_side(side: str):
    col = "back_proc_bytes" if side == "back" else "front_proc_bytes"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, label, {col} AS proc_bytes, created_at
                FROM {TABLE}
                WHERE {col} IS NOT NULL AND label IS NOT NULL
                ORDER BY created_at ASC
            """)
            return cur.fetchall()

def fetch_reference_rows_combined():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, label, back_proc_bytes, front_proc_bytes, created_at
                FROM {TABLE}
                WHERE back_proc_bytes IS NOT NULL AND front_proc_bytes IS NOT NULL AND label IS NOT NULL
                ORDER BY created_at ASC
            """)
            return cur.fetchall()

def rebuild_reference_cache_side(side: str):
    rows = fetch_reference_rows_side(side)
    meta = []
    emb_list = []
    embder = embedder_back if side == "back" else embedder_front

    ring = EMB_RING_FRAC_BACK if side == "back" else EMB_RING_FRAC_FRONT

    for r in rows:
        lbl = r["label"]
        if lbl not in LABELS:
            continue

        img = proc_png_bytes_to_np(bytes(r["proc_bytes"]))
        seed = seed_from_image_uint8(img)
        rng = np.random.default_rng(seed)

        views, _ = make_views(img, rng=rng)

        # ✅ WICHTIG: Cache-Embeddings auch nur Rand/Ecken sehen lassen
        views_emb = [mask_center_keep_border(v, ring_frac=ring, fill=EMB_FILL_VALUE) for v in views]
        views_np = np.stack(views_emb, axis=0)

        e = embed_images_np(embder, views_np)
        e_mean = e.mean(axis=0)
        e_mean = e_mean / (np.linalg.norm(e_mean) + 1e-12)

        emb_list.append(e_mean.astype(np.float32))
        meta.append({
            "id": str(r["id"]),
            "label": lbl,
            "proc_bytes": bytes(r["proc_bytes"]),
            "created_at": r["created_at"]
        })

    if len(meta) == 0:
        REF[side]["emb"] = np.zeros((0, EMB_DIM), dtype=np.float32)
        REF[side]["meta"] = []
        return

    REF[side]["emb"] = np.stack(emb_list, axis=0).astype(np.float32)
    REF[side]["meta"] = meta


def rebuild_reference_cache_combined():
    rows = fetch_reference_rows_combined()
    meta = []
    emb_list = []

    wb, wf = WEIGHT_BACK, WEIGHT_FRONT
    s = wb + wf + 1e-12
    wb, wf = wb/s, wf/s

    for r in rows:
        lbl = r["label"]
        if lbl not in LABELS:
            continue

        back_img = proc_png_bytes_to_np(bytes(r["back_proc_bytes"]))
        front_img = proc_png_bytes_to_np(bytes(r["front_proc_bytes"]))

        seed = seed_from_image_uint8(back_img) ^ (seed_from_image_uint8(front_img) << 1)
        rng = np.random.default_rng(seed)

        # --- BACK views ---
        b_views, _ = make_views(back_img, rng=rng)
        b_views_emb = [mask_center_keep_border(v, ring_frac=EMB_RING_FRAC_BACK, fill=EMB_FILL_VALUE) for v in b_views]
        b_embs = embed_images_np(embedder_back, np.stack(b_views_emb, axis=0))
        b_mean = b_embs.mean(axis=0)
        b_mean = b_mean / (np.linalg.norm(b_mean) + 1e-12)

        # --- FRONT views ---
        f_views, _ = make_views(front_img, rng=rng)
        f_views_emb = [mask_center_keep_border(v, ring_frac=EMB_RING_FRAC_FRONT, fill=EMB_FILL_VALUE) for v in f_views]
        f_embs = embed_images_np(embedder_front, np.stack(f_views_emb, axis=0))
        f_mean = f_embs.mean(axis=0)
        f_mean = f_mean / (np.linalg.norm(f_mean) + 1e-12)

        # --- combined mean ---
        c_mean = wb*b_mean + wf*f_mean
        c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-12)

        emb_list.append(c_mean.astype(np.float32))
        meta.append({
            "id": str(r["id"]),
            "label": lbl,
            "back_proc_bytes": bytes(r["back_proc_bytes"]),
            "front_proc_bytes": bytes(r["front_proc_bytes"]),
            "created_at": r["created_at"]
        })

    if len(meta) == 0:
        REF["combined"]["emb"] = np.zeros((0, EMB_DIM), dtype=np.float32)
        REF["combined"]["meta"] = []
        return

    REF["combined"]["emb"] = np.stack(emb_list, axis=0).astype(np.float32)
    REF["combined"]["meta"] = meta


def rebuild_all_caches():
    rebuild_reference_cache_side("back")
    rebuild_reference_cache_side("front")
    rebuild_reference_cache_combined()

rebuild_all_caches()
print("✅ Ref cache BACK:", len(REF["back"]["meta"]), REF["back"]["emb"].shape)
print("✅ Ref cache FRONT:", len(REF["front"]["meta"]), REF["front"]["emb"].shape)
print("✅ Ref cache COMBINED:", len(REF["combined"]["meta"]), REF["combined"]["emb"].shape)

def topk_matches(cache_key: str, query_emb: np.ndarray, k=TOPK):
    emb = REF[cache_key]["emb"]
    meta = REF[cache_key]["meta"]
    if emb is None or len(emb) == 0:
        return [], []
    sims = emb @ query_emb
    idx = np.argsort(-sims)[:k]
    return idx.tolist(), sims[idx].tolist()

def knn_vote(cache_key: str, idxs, sims):
    meta = REF[cache_key]["meta"]
    scores = {l: 0.0 for l in LABELS}
    for i, s in list(zip(idxs, sims))[:KNN_K]:
        scores[meta[i]["label"]] += float(max(0.0, s))
    best_lbl, best_score = max(scores.items(), key=lambda x: x[1])
    total = sum(scores.values()) + 1e-12
    conf = best_score / total
    return best_lbl, float(conf), scores

def local_prior_probs_from_neighbors(cache_key: str, idxs: list, sims: list,
                                     power: float = 3.0) -> np.ndarray:
    """
    Baut aus den (idxs,sims) eine lokale Klassenverteilung p(label|neighbors).
    Weighting: w_i = max(0, sim_i)^power
    """
    p = np.zeros((len(LABELS),), dtype=np.float32)
    meta = REF[cache_key]["meta"] or []
    if not idxs or not sims or not meta:
        return p

    for i, s in zip(idxs, sims):
        if i < 0 or i >= len(meta):
            continue
        lbl = meta[i].get("label")
        if lbl not in LABEL2IDX:
            continue
        w = max(0.0, float(s)) ** float(power)
        p[LABEL2IDX[lbl]] += float(w)

    ssum = float(p.sum())
    if ssum <= 1e-12:
        return p
    return (p / ssum).astype(np.float32)



# =========================
# Analyse + Robust Aggregation + Combined
# =========================
def probs_from_knn_scores(scores: dict) -> np.ndarray:
    p = np.zeros((len(LABELS),), dtype=np.float32)
    if not scores:
        return p
    for l, v in scores.items():
        if l in LABEL_IDX:
            p[LABEL_IDX[l]] = max(0.0, float(v))
    s = float(p.sum())
    if s <= 1e-12:
        return p
    return (p / s).astype(np.float32)

def effective_side_probs(status_side: dict, probs_side: np.ndarray) -> np.ndarray:
    probs_side = np.asarray(probs_side, dtype=np.float32)
    probs_side = probs_side / (float(probs_side.sum()) + 1e-12)

    cls_conf = float((status_side.get("classifier") or {}).get("conf", 0.0))
    knn = status_side.get("knn") or {}
    knn_scores = knn.get("scores", {}) or {}
    best_sim = float(knn.get("best_sim", 0.0))

    if best_sim < KNN_MIX_MIN_BEST_SIM:
        return probs_side

    p_knn = probs_from_knn_scores(knn_scores)
    mix = KNN_MIX_BASE * max(0.0, (0.80 - cls_conf) / 0.80)
    mix = float(np.clip(mix, 0.0, 0.35))

    if p_knn.sum() <= 1e-12 or mix <= 1e-6:
        return probs_side

    p = (1.0 - mix) * probs_side + mix * p_knn
    p = p / (float(p.sum()) + 1e-12)
    return p.astype(np.float32)

def expected_index_from_probs(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float32)
    p = p / (float(p.sum()) + 1e-12)
    idx = np.arange(len(LABELS), dtype=np.float32)
    return float(np.sum(idx * p))

def label_from_expected_idx(idx: float) -> str:
    i = int(np.clip(int(round(idx)), 0, len(LABELS)-1))
    return IDX2LABEL[i]

def aggregate_probs_robust(probs_v: np.ndarray, trim_q: float = VIEW_TRIM_Q) -> Tuple[np.ndarray, dict]:
    pv = np.asarray(probs_v, dtype=np.float32)
    pv = pv / (pv.sum(axis=1, keepdims=True) + 1e-12)

    sev = np.sum(pv * np.arange(len(LABELS), dtype=np.float32)[None, :], axis=1)
    lo = float(np.quantile(sev, trim_q))
    hi = float(np.quantile(sev, 1.0 - trim_q))
    keep = (sev >= lo) & (sev <= hi)

    if int(keep.sum()) < max(3, int(0.5 * len(sev))):
        keep[:] = True

    p = pv[keep].mean(axis=0)
    p = p / (float(p.sum()) + 1e-12)
    meta = {"views_total": int(len(sev)), "views_kept": int(keep.sum()), "sev_median": float(np.median(sev))}
    return p.astype(np.float32), meta

def guardrail_label(candidate_label: str, probs: np.ndarray, cls_conf: float, best_sim: float, knn_conf: float) -> Tuple[str, str]:
    idx_exp = expected_index_from_probs(probs)
    lbl_exp = label_from_expected_idx(idx_exp)

    diff = abs(LABEL2IDX[candidate_label] - LABEL2IDX[lbl_exp])
    strong = (cls_conf >= EXTREME_ALLOW_CNN_CONF) or (best_sim >= EXTREME_ALLOW_SIM and knn_conf >= EXTREME_ALLOW_KNN_CONF)

    if diff > MAX_OUTLIER_JUMP and not strong:
        return lbl_exp, f"Guardrail(exp={lbl_exp}, idx={idx_exp:.2f}, diff={diff})"
    return candidate_label, ""

def analyze_once(side: str, proc_np_uint8: np.ndarray, rng: np.random.Generator):
    proc_np_uint8 = resize_to_base(proc_np_uint8)

    model = model_back if side == "back" else model_front
    embder = embedder_back if side == "back" else embedder_front

    views, _ = make_views(proc_np_uint8, rng=rng)
    views_np = np.stack([(infer_augment_np(v, rng) if INFER_AUG else v.astype(np.float32)) for v in views], axis=0)

    probs_v = model.predict(views_np, verbose=0)
    probs, agg_meta = aggregate_probs_robust(probs_v, trim_q=VIEW_TRIM_Q)

    cls_idx = int(np.argmax(probs))
    cls_label = IDX2LABEL[cls_idx]
    cls_conf = float(probs[cls_idx])

    ring = EMB_RING_FRAC_BACK if side == "back" else EMB_RING_FRAC_FRONT
    views_emb = [mask_center_keep_border(v, ring_frac=ring, fill=EMB_FILL_VALUE) for v in views]
    e = embed_images_np(embder, np.stack(views_emb, axis=0))

    q_emb = e.mean(axis=0)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    idxs, sims = topk_matches(side, q_emb, k=TOPK)
    best_sim = float(sims[0]) if sims else 0.0

    if idxs:
        knn_label, knn_conf, knn_scores = knn_vote(side, idxs, sims)
    else:
        knn_label, knn_conf, knn_scores = cls_label, 0.0, {}

    use_knn_override = (
        USE_KNN_FUSION and
        cls_conf < FUSION_CONF_TH and
        best_sim >= KNN_FUSION_MIN_BEST_SIM and
        float(knn_conf) >= KNN_FUSION_MIN_CONF
    )

    if use_knn_override:
        candidate = knn_label
        final_source = f"kNN override (conf<{FUSION_CONF_TH}, sim={best_sim:.3f})"
    else:
        candidate = cls_label
        final_source = "Classifier"

    final_label, gr_note = guardrail_label(candidate, probs, cls_conf, best_sim, float(knn_conf))
    if gr_note:
        final_source = final_source + " + " + gr_note

    return {
        "final_label": final_label,
        "final_source": final_source,
        "cls_label": cls_label,
        "cls_conf": cls_conf,
        "probs": probs,
        "q_emb": q_emb,
        "knn_label": knn_label,
        "knn_conf": knn_conf,
        "knn_scores": knn_scores,
        "best_sim": best_sim,
        "idxs": idxs,
        "sims": sims,
        "agg_meta": agg_meta
    }

# --- PSA Scale (inkl. 1.5) ---
PSA_SCALE = {
    10.0: ("GEM-MT", "nahezu perfekt"),
    9.0:  ("MINT",   "superb, nur minimale Mängel"),
    8.0:  ("NM-MT",  "fast mint, kleine Unsauberkeiten bei genauer Prüfung"),
    7.0:  ("NM",     "leichte Abnutzung sichtbar (bei genauer Prüfung)"),
    6.0:  ("EX-MT",  "sichtbare Abnutzung/kleiner Druckfehler möglich"),
    5.0:  ("EX",     "mehr sichtbare Wear, evtl. kleine Chips/leichte Kratzer"),
    4.0:  ("VG-EX",  "moderater Wear, ggf. leichter Knick"),
    3.0:  ("VG",     "deutlicher Wear, Knick(e) möglich"),
    2.0:  ("GOOD",   "starker Wear, mehrere Knicke möglich"),
    1.5:  ("FR",     "extrem abgenutzt, muss aber „intakt“ sein"),
    1.0:  ("PR",     "sehr starker Schaden, Eye-Appeal nahezu weg"),
}

def psa_round_grade(g: float) -> float:
    """
    PSA: grundsätzlich ganze Zahlen, außer 1.5.
    => <2.0 auf 0.5er Schritte runden, sonst auf ganze Zahl.
    """
    g = float(max(1.0, min(10.0, g)))
    if g < 2.0:
        return round(g * 2.0) / 2.0
    return float(int(round(g)))

def format_grade(g: float) -> str:
    g = float(g)
    if abs(g - int(g)) < 1e-9:
        return str(int(g))
    return f"{g:.1f}".rstrip("0").rstrip(".")

def psa_line(g: float) -> str:
    g = psa_round_grade(g)
    abbr, desc = PSA_SCALE.get(g, ("?", ""))
    return f"PSA {format_grade(g)} – {abbr}: {desc}"


# Deine Klassen -> PSA-Grade-Ranges (wie bisher, aber PO erlaubt jetzt 1.5)
GRADE_RANGES = {
    "NM": (9.0, 10.0),
    "EX": (7.0, 8.0),
    "GD": (5.0, 6.0),
    "LP": (3.0, 4.0),
    "PL": (2.0, 2.0),
    "PO": (1.0, 1.5),   # ✅ neu: 1 oder 1.5
}


def compute_grade_1to10(final_label: str, probs: np.ndarray, knn_conf: float, sims: list):
    lo, hi = GRADE_RANGES.get(final_label, (1.0, 10.0))

    p_sorted = np.sort(probs)[::-1]
    p1 = float(p_sorted[0]) if len(p_sorted) else 0.0
    p2 = float(p_sorted[1]) if len(p_sorted) > 1 else 0.0
    margin = max(0.0, p1 - p2)

    sim_strength = float(np.mean(sims[:min(5, len(sims))])) if sims else 0.0
    sim_strength = max(0.0, min(1.0, sim_strength))

    score = 0.55*p1 + 0.25*margin + 0.12*float(knn_conf) + 0.08*sim_strength
    score = max(0.0, min(1.0, score))

    # continuous grade within label-range
    if abs(hi - lo) < 1e-9:
        g_cont = lo
    else:
        g_cont = lo + (hi - lo) * score

    # PSA rounding rule
    g = psa_round_grade(g_cont)

    # clamp to range (wichtig für PO: 1..1.5)
    g = max(lo, min(hi, g))

    meta = {
        "score": round(score, 4),
        "p1": round(p1, 4),
        "margin": round(margin, 4),
        "knn_conf": round(float(knn_conf), 4),
        "sim_strength": round(sim_strength, 4),
        "grade_cont": round(float(g_cont), 4),
        "grade_psa": format_grade(g),
        "psa": psa_line(g),
    }
    return float(g), meta


def pregrading_text(label: str, grade_1to10: float):
    tips = {
        "NM": "Sehr guter Kandidat für Grading. Prüfe Whitening an Ecken/Kanten + feine Kratzer im Licht.",
        "EX": "Guter Kandidat. Kleine Mängel wahrscheinlich (Whitening/leichte Lines).",
        "GD": "Mittelzustand. Grading lohnt sich meist nur bei hochpreisigen Karten.",
        "LP": "Deutlich bespielt. Grading eher zur Dokumentation/bei Seltenheit.",
        "PL": "Stark bespielt. Grading selten wirtschaftlich.",
        "PO": "Beschädigt. Grading meist nicht wirtschaftlich (außer sehr selten)."
    }
    g = psa_round_grade(float(grade_1to10))
    return (
        f"Pregrading (TCG-Klasse): {label} | Grade: {format_grade(g)}/10\n"
        f"{psa_line(g)}\n"
        + tips.get(label, "")
    )


def label_from_grade_1to10(g: float) -> str:
    g = float(max(1.0, min(10.0, g)))
    if g >= 9.0:
        return "NM"
    if g >= 7.0:
        return "EX"
    if g >= 5.0:
        return "GD"
    if g >= 3.0:
        return "LP"
    if g >= 2.0:
        return "PL"
    return "PO"



def analyze_ensemble(side: str, proc_np_uint8: np.ndarray, n_runs: int = ENSEMBLE_RUNS):
    proc_np_uint8 = resize_to_base(proc_np_uint8)
    base_seed = seed_from_image_uint8(proc_np_uint8) if DETERMINISTIC_PER_IMAGE else int(np.random.default_rng().integers(0, 2**32-1))

    runs = []
    probs_list = []
    emb_list = []
    label_counts = {l: 0 for l in LABELS}
    grades = []

    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i * SEED_STEP)
        r = analyze_once(side, proc_np_uint8, rng=rng)
        runs.append(r)
        probs_list.append(r["probs"])
        emb_list.append(r["q_emb"])
        if r["final_label"] in label_counts:
            label_counts[r["final_label"]] += 1
        g, _ = compute_grade_1to10(r["final_label"], r["probs"], r["knn_conf"], r["sims"])
        grades.append(g)

    best_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
    best_count = label_counts[best_label]
    tied = [l for l,cnt in label_counts.items() if cnt == best_count]

    probs_avg = np.mean(np.stack(probs_list, axis=0), axis=0)
    probs_avg = probs_avg / (probs_avg.sum() + 1e-12)

    q_emb_avg = np.mean(np.stack(emb_list, axis=0), axis=0)
    q_emb_avg = q_emb_avg / (np.linalg.norm(q_emb_avg) + 1e-12)

    idxs, sims = topk_matches(side, q_emb_avg, k=TOPK)
    best_sim = float(sims[0]) if sims else 0.0

    if idxs:
        knn_label, knn_conf, knn_scores = knn_vote(side, idxs, sims)
    else:
        knn_label, knn_conf, knn_scores = IDX2LABEL[int(np.argmax(probs_avg))], 0.0, {}

    cls_idx = int(np.argmax(probs_avg))
    cls_label = IDX2LABEL[cls_idx]
    cls_conf = float(probs_avg[cls_idx])

    if USE_KNN_FUSION and cls_conf < FUSION_CONF_TH and best_sim >= KNN_FUSION_MIN_BEST_SIM and float(knn_conf) >= KNN_FUSION_MIN_CONF:
        agg_final_label = knn_label
        agg_source = f"kNN (cls_conf<{FUSION_CONF_TH}, sim={best_sim:.3f})"
    else:
        agg_final_label = cls_label
        agg_source = "Classifier"

    if len(tied) == 1:
        final_label = best_label
        final_source = f"Ensemble majority ({best_count}/{n_runs})"
    else:
        final_label = agg_final_label
        final_source = f"Ensemble tie-break → {agg_source} (ties={tied})"

    grades_for_label = [g for g, r in zip(grades, runs) if r["final_label"] == final_label]
    if grades_for_label:
        final_grade = psa_round_grade(float(np.median(np.array(grades_for_label))))
        grade_meta = {"method": "median_over_majority_runs", "n": len(grades_for_label)}
    else:
        final_grade, gm = compute_grade_1to10(final_label, probs_avg, knn_conf, sims)
        grade_meta = {"method": "aggregated_fallback", **gm}

    gal = []
    meta = REF[side]["meta"]
    for i, s in zip(idxs, sims):
        m = meta[i]
        img = Image.open(io.BytesIO(m["proc_bytes"])).convert("RGB")
        cap = f'{m["label"]} | sim={s:.3f} | {m["id"][:8]}'
        gal.append((img, cap))

    topk_list = [{"label": meta[i]["label"], "sim": float(s), "id": meta[i]["id"]} for i, s in zip(idxs, sims)]

    status = {
        "side": side,
        "final_label": final_label,
        "final_source": final_source,
        "grade": float(final_grade),
        "classifier": {
            "label": cls_label,
            "conf": round(cls_conf, 4),
            "probs": {IDX2LABEL[i]: float(probs_avg[i]) for i in range(len(LABELS))}
        },
        "knn": {
            "label": knn_label,
            "conf": round(float(knn_conf), 4),
            "best_sim": round(float(best_sim), 4),
            "scores": {k: round(float(v), 4) for k, v in (knn_scores or {}).items()}
        },
        "views_used": len(make_views(proc_np_uint8, rng=np.random.default_rng(base_seed))[0]),
        "reference_size": len(meta),
        "topk": topk_list,
        "ensemble": {
            "runs": n_runs,
            "deterministic_per_image": bool(DETERMINISTIC_PER_IMAGE),
            "seed": int(base_seed),
            "label_counts": label_counts,
            "grades": grades
        }
    }

    txt = (
        f"{side.upper()}: {final_label} | Grade {format_grade(final_grade)}/10 | {psa_line(final_grade)} | {final_source} | "
        f"CNN: {cls_label} ({cls_conf:.2f}) | kNN: {knn_label} ({knn_conf:.2f}, sim={best_sim:.2f}) | "
        f"Votes: {label_counts}"
    )

    state_payload = {"status": status, "q_emb": q_emb_avg}
    return status, gal, state_payload, txt, probs_avg, idxs, sims, final_grade, grade_meta

def compute_combined(
    status_b, probs_b, sims_b, st_b, grade_b, grade_meta_b,
    status_f, probs_f, sims_f, st_f, grade_f, grade_meta_f
):

    wb, wf = WEIGHT_BACK, WEIGHT_FRONT
    s = wb + wf + 1e-12
    wb, wf = wb/s, wf/s

    # --- Side probs (evtl. schon leicht mit Side-kNN gemischt) ---
    p_b = effective_side_probs(status_b, probs_b)
    p_f = effective_side_probs(status_f, probs_f)

    # --- Raw CNN Combined ---
    p_c_raw = wb*p_b + wf*p_f
    p_c_raw = p_c_raw / (float(p_c_raw.sum()) + 1e-12)

    # --- Combined embedding ---
    q_b = st_b.get("q_emb")
    q_f = st_f.get("q_emb")
    if q_b is not None and q_f is not None:
        q_c = wb*q_b + wf*q_f
        q_c = q_c / (np.linalg.norm(q_c) + 1e-12)
    else:
        q_c = q_b if q_b is not None else q_f

    # --- Combined retrieval (Top-10) + kNN vote ---
    idxs_c, sims_c = ([], [])
    knn_c_label, knn_c_conf, knn_c_scores, best_sim_c = None, 0.0, {}, 0.0

    if q_c is not None:
        idxs_c, sims_c = topk_matches("combined", q_c, k=TOPK)
        best_sim_c = float(sims_c[0]) if sims_c else 0.0

        if idxs_c:
            knn_c_label, knn_c_conf, knn_c_scores = knn_vote("combined", idxs_c, sims_c)
        else:
            knn_c_label, knn_c_conf, knn_c_scores = IDX2LABEL[int(np.argmax(p_c_raw))], 0.0, {}
    else:
        knn_c_label, knn_c_conf, knn_c_scores = IDX2LABEL[int(np.argmax(p_c_raw))], 0.0, {}

    # --- Local Neighbor Prior (Top-10) → fuse in combined probs ---
    p_c = p_c_raw.copy().astype(np.float32)
    local_mix = 0.0
    p_local = np.zeros_like(p_c, dtype=np.float32)

    if USE_LOCAL_TOP10 and q_c is not None and idxs_c and (best_sim_c >= LOCAL_MIN_SIM_FOR_MIX):
        p_local = local_prior_probs_from_neighbors(
            "combined",
            idxs_c[:LOCAL_TOPK],
            sims_c[:LOCAL_TOPK],
            power=LOCAL_PRIOR_POWER
        )

        t = (best_sim_c - LOCAL_MIN_SIM_FOR_MIX) / (1.0 - LOCAL_MIN_SIM_FOR_MIX + 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
        local_mix = float(LOCAL_MAX_MIX * t)

        if float(p_local.sum()) > 1e-12 and local_mix > 1e-6:
            p_c = (1.0 - local_mix) * p_c + local_mix * p_local
            p_c = p_c / (float(p_c.sum()) + 1e-12)

    # --- Labels (Back/Front) ---
    back_label  = status_b.get("final_label", "?")
    front_label = status_f.get("final_label", "?")
    diff_lbl = abs(LABEL2IDX.get(back_label, 0) - LABEL2IDX.get(front_label, 0))

    # --- Final decision logic ---
    if COMBINED_TIEBREAK_BY_KNN and back_label != front_label:
        if diff_lbl >= DISAGREE_HARD_DIFF:
            if (best_sim_c >= DISAGREE_REQUIRE_SIM) and (float(knn_c_conf) >= DISAGREE_REQUIRE_CONF) and (knn_c_label in (back_label, front_label)):
                final_label = knn_c_label
                final_source = f"DISAGREE hard(diff={diff_lbl}) → Combined kNN (lbl={knn_c_label}, conf={knn_c_conf:.2f}, sim={best_sim_c:.2f})"
            else:
                idx_pick = int(np.argmax(p_c))
                lo = min(LABEL2IDX[back_label], LABEL2IDX[front_label])
                hi = max(LABEL2IDX[back_label], LABEL2IDX[front_label])
                idx_pick = int(np.clip(idx_pick, lo, hi))
                final_label = IDX2LABEL[idx_pick]
                final_source = f"DISAGREE hard(diff={diff_lbl}) → Clamp fused combined probs to [{IDX2LABEL[lo]}..{IDX2LABEL[hi]}]"
        else:
            if knn_c_label in (back_label, front_label) and best_sim_c >= KNN_MIX_MIN_BEST_SIM:
                final_label = knn_c_label
                final_source = f"DISAGREE → Combined kNN tie-break (lbl={knn_c_label}, conf={knn_c_conf:.2f}, sim={best_sim_c:.2f})"
            else:
                cb = float((status_b.get("classifier") or {}).get("conf", 0.0))
                cf = float((status_f.get("classifier") or {}).get("conf", 0.0))
                sb = float((status_b.get("knn") or {}).get("best_sim", 0.0))
                sf = float((status_f.get("knn") or {}).get("best_sim", 0.0))
                score_b = 0.65*cb + 0.35*sb
                score_f = 0.65*cf + 0.35*sf
                if score_f > score_b:
                    final_label = front_label
                    final_source = f"DISAGREE → evidence pick FRONT (score_f={score_f:.2f} > score_b={score_b:.2f})"
                else:
                    final_label = back_label
                    final_source = f"DISAGREE → evidence pick BACK (score_b={score_b:.2f} >= score_f={score_f:.2f})"
    else:
        final_label = IDX2LABEL[int(np.argmax(p_c))]
        final_source = "AGREE/Combined fused probs"

    # --- Grade auf fused p_c (Model-Grade) ---
    sims_for_grade = sims_c[:5] if sims_c else []
    grade_c_model, grade_meta_c = compute_grade_1to10(final_label, p_c, float(knn_c_conf), sims_for_grade)

    # ✅ Disagree-Guardrail: Wenn Back != Front → Grade = Median(side grades) & Label passend zum Grade
    if back_label != front_label:
        grade_c = psa_round_grade(float(np.median([float(grade_b), float(grade_f)])))
        grade_c = max(1, min(10, grade_c))
        final_label = label_from_grade_1to10(grade_c)
        final_source = f"{final_source} + GradeGuard(median(back={grade_b}, front={grade_f}))"
        grade_meta_c = {
            "method": "median_side_grades_due_to_disagree",
            "grade_b": int(grade_b),
            "grade_f": int(grade_f),
            "grade_model": int(grade_c_model),
        }
    else:
        grade_c = int(grade_c_model)
        grade_meta_c.update({
            "w_back": round(float(wb), 3),
            "w_front": round(float(wf), 3),
            "tiebreak_by_knn": bool(COMBINED_TIEBREAK_BY_KNN),
            "combined_knn_best_sim": round(float(best_sim_c), 3),
            "combined_knn_conf": round(float(knn_c_conf), 3),
            "local_mix": round(float(local_mix), 4),
            "local_min_sim_for_mix": float(LOCAL_MIN_SIM_FOR_MIX),
            "local_topk": int(LOCAL_TOPK),
            "local_prior_power": float(LOCAL_PRIOR_POWER),
        })

    # --- TopK Meta ---
    meta_c = REF["combined"]["meta"] or []
    topk_c = [{"label": meta_c[i]["label"], "sim": float(s), "id": meta_c[i]["id"]} for i, s in zip(idxs_c, sims_c)]

    # --- Status ---
    status_c = {
        "side": "combined",
        "final_label": final_label,
        "final_source": final_source,
        "weights": {"back": round(float(wb), 4), "front": round(float(wf), 4)},
        "grade": int(grade_c),

        "cnn_raw": {
            "label": IDX2LABEL[int(np.argmax(p_c_raw))],
            "conf": round(float(np.max(p_c_raw)), 4),
            "probs": {IDX2LABEL[i]: float(p_c_raw[i]) for i in range(len(LABELS))}
        },

        "classifier": {
            "label": IDX2LABEL[int(np.argmax(p_c))],
            "conf": round(float(np.max(p_c)), 4),
            "probs": {IDX2LABEL[i]: float(p_c[i]) for i in range(len(LABELS))}
        },

        "knn": {
            "label": knn_c_label,
            "conf": round(float(knn_c_conf), 4),
            "best_sim": round(float(best_sim_c), 4),
            "scores": {k: round(float(v), 4) for k, v in (knn_c_scores or {}).items()}
        },

        "local_retrieval": {
            "enabled": bool(USE_LOCAL_TOP10),
            "local_mix": round(float(local_mix), 4),
            "best_sim": round(float(best_sim_c), 4),
            "min_sim_for_mix": float(LOCAL_MIN_SIM_FOR_MIX),
            "topk": int(LOCAL_TOPK),
        },

        "topk": topk_c,
        "back": {"label": back_label, "grade": int(grade_b)},
        "front": {"label": front_label, "grade": int(grade_f)},
    }

    state_c = {"status": status_c, "q_emb": q_c}
    return status_c, p_c, state_c, grade_c, grade_meta_c



# =========================
# DB Insert (FRONT+BACK Pflicht) + Approve Save
# =========================
def insert_sample_fb(
    label, note,
    back_raw_bytes, back_raw_format, back_raw_w, back_raw_h,
    back_proc_bytes, back_proc_format, back_proc_w, back_proc_h,
    back_mask_bytes, back_mask_format="png", back_mask_w=IMG_W, back_mask_h=IMG_H,
    back_proc_method=None, back_proc_quad_expand=None,
    front_raw_bytes=None, front_raw_format="jpeg", front_raw_w=None, front_raw_h=None,
    front_proc_bytes=None, front_proc_format="png", front_proc_w=IMG_W, front_proc_h=IMG_H,
    front_mask_bytes=None, front_mask_format="png", front_mask_w=IMG_W, front_mask_h=IMG_H,
    front_proc_method=None, front_proc_quad_expand=None,
):
    if front_raw_bytes is None or front_proc_bytes is None:
        raise ValueError("Front ist Pflicht (raw/proc).")

    back_sha  = hashlib.sha256(back_raw_bytes).hexdigest()
    front_sha = hashlib.sha256(front_raw_bytes).hexdigest()
    sample_id = uuid.uuid4()

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TABLE} (
                        id, label, note,

                        back_raw_sha256, back_raw_format, back_raw_w, back_raw_h, back_raw_bytes,
                        back_proc_format, back_proc_w, back_proc_h, back_proc_bytes,
                        back_proc_mask_format, back_proc_mask_w, back_proc_mask_h, back_proc_mask_bytes,
                        back_proc_method, back_proc_quad_expand,

                        front_raw_sha256, front_raw_format, front_raw_w, front_raw_h, front_raw_bytes,
                        front_proc_format, front_proc_w, front_proc_h, front_proc_bytes,
                        front_proc_mask_format, front_proc_mask_w, front_proc_mask_h, front_proc_mask_bytes,
                        front_proc_method, front_proc_quad_expand
                    )
                    VALUES (
                        %s,%s,%s,
                        %s,%s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s,
                        %s,%s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s
                    )
                """, (
                    str(sample_id), label, note,

                    back_sha, back_raw_format, back_raw_w, back_raw_h, psycopg2.Binary(back_raw_bytes),
                    back_proc_format, back_proc_w, back_proc_h, psycopg2.Binary(back_proc_bytes),
                    back_mask_format, back_mask_w, back_mask_h, psycopg2.Binary(back_mask_bytes) if back_mask_bytes else None,
                    back_proc_method, float(back_proc_quad_expand) if back_proc_quad_expand is not None else None,

                    front_sha, front_raw_format, front_raw_w, front_raw_h, psycopg2.Binary(front_raw_bytes),
                    front_proc_format, front_proc_w, front_proc_h, psycopg2.Binary(front_proc_bytes),
                    front_mask_format, front_mask_w, front_mask_h, psycopg2.Binary(front_mask_bytes) if front_mask_bytes else None,
                    front_proc_method, float(front_proc_quad_expand) if front_proc_quad_expand is not None else None,
                ))
        return True, str(sample_id), back_sha, front_sha, "✅ Gespeichert (Front+Back)."
    except Exception as e:
        return False, None, back_sha, front_sha, "❌ DB-Fehler:\n" + fmt_pg_error(e) + "\n\n" + traceback.format_exc()

def approve_and_save(state, user_note, label_override):
    if state is None:
        return "❌ Kein Analyse-State. Bitte erst analysieren.", db_counts()

    predicted = state["combined"]["status"]["final_label"]
    if predicted not in LABELS:
        return f"❌ Interner Fehler: predicted label ungültig: {predicted}", db_counts()

    label_used = label_override if label_override in LABELS else predicted
    was_corrected = (label_used != predicted)

    note_obj = {
        "user_note": (user_note or "").strip(),
        "predicted_combined": predicted,
        "predicted_back": state["back"]["status"]["final_label"],
        "predicted_front": state["front"]["status"]["final_label"],
        "label_used": label_used,
        "was_corrected": was_corrected,
        "combined": state["combined"]["status"],
        "back": state["back"]["status"],
        "front": state["front"]["status"],
        "grade_combined": state.get("_grade_combined"),
        "grade_meta_combined": state.get("_grade_meta_combined"),
    }
    note = json.dumps(note_obj, ensure_ascii=False)

    back_pil = state.get("_last_back_pil")
    front_pil = state.get("_last_front_pil")
    if back_pil is None or front_pil is None:
        return "❌ Interner Fehler: Upload-Images fehlen im State.", db_counts()

    back_fixed = ImageOps.exif_transpose(back_pil).convert("RGB")
    front_fixed = ImageOps.exif_transpose(front_pil).convert("RGB")
    back_w, back_h = back_fixed.size
    front_w, front_h = front_fixed.size

    back_raw_bytes = pil_to_jpeg_bytes(back_fixed, 92)
    front_raw_bytes = pil_to_jpeg_bytes(front_fixed, 92)

    back_proc_np = state.get("_proc_back_np")
    front_proc_np = state.get("_proc_front_np")
    if back_proc_np is None or front_proc_np is None:
        return "❌ Interner Fehler: proc fehlt im State.", db_counts()

    okb, encb = cv2.imencode(".png", cv2.cvtColor(back_proc_np, cv2.COLOR_RGB2BGR))
    okf, encf = cv2.imencode(".png", cv2.cvtColor(front_proc_np, cv2.COLOR_RGB2BGR))
    if not okb or not okf:
        return "❌ Konnte proc PNG nicht encodieren.", db_counts()
    back_proc_png = encb.tobytes()
    front_proc_png = encf.tobytes()

    back_mask = state.get("_mask_back_np")
    front_mask = state.get("_mask_front_np")
    back_mask_png = encode_png_bytes_gray(back_mask) if back_mask is not None else None
    front_mask_png = encode_png_bytes_gray(front_mask) if front_mask is not None else None

    back_meta = state.get("_meta_back") or {}
    front_meta = state.get("_meta_front") or {}

    ok, new_id, back_sha, front_sha, msg = insert_sample_fb(
        label=label_used,
        note=note,

        back_raw_bytes=back_raw_bytes, back_raw_format="jpeg", back_raw_w=back_w, back_raw_h=back_h,
        back_proc_bytes=back_proc_png, back_proc_format="png", back_proc_w=IMG_W, back_proc_h=IMG_H,
        back_mask_bytes=back_mask_png, back_proc_method=back_meta.get("method"), back_proc_quad_expand=back_meta.get("quad_expand"),

        front_raw_bytes=front_raw_bytes, front_raw_format="jpeg", front_raw_w=front_w, front_raw_h=front_h,
        front_proc_bytes=front_proc_png, front_proc_format="png", front_proc_w=IMG_W, front_proc_h=IMG_H,
        front_mask_bytes=front_mask_png, front_proc_method=front_meta.get("method"), front_proc_quad_expand=front_meta.get("quad_expand"),
    )

    if ok:
        try:
            rebuild_all_caches()
        except:
            pass

        extra = " (korrigiert)" if was_corrected else ""
        return f"✅ Gespeichert: {label_used}{extra}. id={new_id[:8]} back_sha={back_sha[:10]} front_sha={front_sha[:10]}", db_counts()

    return msg, db_counts()


# =========================
# Plot + PDF Report (1 Seite) — PLOTS ONLY
# =========================
def plot_compact_onepage(probs_c, probs_b, probs_f, out_path: str):
    fig = plt.figure(figsize=(8.2, 7.2))
    x = np.arange(len(LABELS))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.bar(x, probs_c)
    ax1.set_xticks(x, LABELS)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Combined probabilities (Front+Back)")
    ax1.set_ylabel("Prob")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.bar(x, probs_b)
    ax2.set_xticks(x, LABELS)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("BACK probabilities")
    ax2.set_ylabel("Prob")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.bar(x, probs_f)
    ax3.set_xticks(x, LABELS)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("FRONT probabilities")
    ax3.set_ylabel("Prob")

    fig.tight_layout(pad=2.0, h_pad=1.2)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    

def build_pdf_report_1page(
    out_pdf_path: str,
    plot_path: str,
    status_combined: dict,
    status_back: dict,
    status_front: dict,
):
    c = canvas.Canvas(out_pdf_path, pagesize=A4)
    W, H = A4

    M = 16 * mm
    G = 8 * mm
    header_h = 22 * mm

    CARD_PAD = 10
    TITLE_BAND = 20
    TITLE_BASELINE = 14

    def draw_header(title: str, subtitle: str):
        c.setFillColor(colors.HexColor("#0B1220"))
        c.rect(0, H - header_h, W, header_h, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(M, H - 14 * mm, title)
        c.setFont("Helvetica", 9.5)
        c.setFillColor(colors.HexColor("#C9D4F2"))
        c.drawString(M, H - 19 * mm, subtitle)

    def draw_card(x, y, w, h, title=None):
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor("#E6EAF2"))
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 10, stroke=1, fill=1)
        if title:
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont("Helvetica-Bold", 10.5)
            c.drawString(x + CARD_PAD, y + h - TITLE_BASELINE, title)
        reserved_top = TITLE_BAND if title else 0
        inner_x = x + CARD_PAD
        inner_y = y + CARD_PAD
        inner_w = w - 2 * CARD_PAD
        inner_h = h - 2 * CARD_PAD - reserved_top
        return inner_x, inner_y, inner_w, inner_h

    def draw_kv(x, y, key, val, key_w=78, font_size=9.8):
        c.setFont("Helvetica-Bold", font_size)
        c.setFillColor(colors.HexColor("#111827"))
        c.drawString(x, y, key)
        c.setFont("Helvetica", font_size)
        c.setFillColor(colors.HexColor("#374151"))
        c.drawString(x + key_w, y, val)

    def draw_grade_badge(x, y, w, h, grade: int, label: str, sublabel: str):
        if grade >= 9:
            bg = colors.HexColor("#0B3D2E")
        elif grade >= 7:
            bg = colors.HexColor("#123B6D")
        elif grade >= 5:
            bg = colors.HexColor("#5A3B00")
        elif grade >= 3:
            bg = colors.HexColor("#6D1A1A")
        else:
            bg = colors.HexColor("#2B2B2B")

        c.setFillColor(bg)
        c.setStrokeColor(colors.HexColor("#111827"))
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 18, stroke=1, fill=1)

        c.setStrokeColor(colors.HexColor("#FFFFFF"))
        c.setLineWidth(0.6)
        c.roundRect(x+3, y+3, w-6, h-6, 16, stroke=1, fill=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 46)
        g = format_grade(grade)
        tw = stringWidth(g, "Helvetica-Bold", 46)
        c.drawString(x + (w - tw)/2, y + h*0.60 - 24, g)

        c.setFont("Helvetica-Bold", 11)
        tw2 = stringWidth(label, "Helvetica-Bold", 11)
        c.drawString(x + (w - tw2)/2, y + 16, label)

        c.setFont("Helvetica", 8.8)
        c.setFillColor(colors.HexColor("#E5E7EB"))
        tw3 = stringWidth(sublabel, "Helvetica", 8.8)
        c.drawString(x + (w - tw3)/2, y + 6, sublabel)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    comb_label = status_combined.get("final_label", "?")
    comb_grade = float(status_combined.get("grade", 0))

    back_label = status_back.get("final_label", "?")
    back_grade = int(status_back.get("grade", 0))

    front_label = status_front.get("final_label", "?")
    front_grade = int(status_front.get("grade", 0))

    draw_header("Pokemon TCG Grading Report", f"Generated {now}  •  Final = Combined (Front+Back)")

    content_top = H - header_h - M

    badge_w = 62 * mm
    badge_h = 42 * mm
    badge_x = W - M - badge_w
    badge_y = content_top - badge_h

    draw_grade_badge(
        badge_x, badge_y, badge_w, badge_h,
        int(comb_grade),
        "CERTIFIED GRADE",
        f"Condition: {comb_label}"
    )

    sum_x = M
    sum_y = badge_y
    sum_w = badge_x - M - G
    sum_h = badge_h
    sx, sy, sw, sh = draw_card(sum_x, sum_y, sum_w, sum_h, title="Results (Combined + Sides)")

    draw_kv(sx, sy + sh - 16, "Combined:", f"{comb_label}  • Grade {comb_grade}/10")
    draw_kv(sx, sy + sh - 30, "Back:",     f"{back_label}  • Grade {back_grade}/10")
    draw_kv(sx, sy + sh - 44, "Front:",    f"{front_label} • Grade {front_grade}/10")

    # --- PLOTS (full width) ---
    plot_h = 150 * mm
    plot_y = sum_y - G - plot_h
    px, py, pw, ph = draw_card(M, plot_y, W - 2*M, plot_h, title="Plots")
    try:
        c.drawImage(ImageReader(plot_path), px, py, width=pw, height=ph, preserveAspectRatio=True, anchor='c')
    except:
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#B91C1C"))
        c.drawString(px, py + ph/2, "Plot could not be loaded.")

    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(M, 10*mm, "AI-assisted estimate. Inspect under proper lighting; official grading may differ.")

    c.showPage()
    c.save()


# =========================
# Gradio UI (BACK + FRONT Pflicht) — 1-Page PDF (Plots only)
# =========================
with gr.Blocks(title="Pokemon Condition (Combined Front+Back) + Approve + 1-page PDF (Plots)") as app2:
    gr.Markdown(
        "# 🧠 Notebook 2: Analyse (Front+Back Pflicht)\n"
    )

    state = gr.State(None)

    with gr.Row():
        back_in = gr.Image(label="Upload: Rückseite (Pflicht)", type="pil")
        front_in = gr.Image(label="Upload: Vorderseite (Pflicht)", type="pil")

    with gr.Row():
        back_proc_prev = gr.Image(label="BACK Proc Preview (Analyse Input)", type="pil")
        front_proc_prev = gr.Image(label="FRONT Proc Preview (Analyse Input)", type="pil")

    with gr.Row():
        btn_analyze = gr.Button("Analysieren (x10) + PDF", variant="primary")
        btn_approve = gr.Button("✅ Zustimmen & Speichern (Label=Combined)", variant="secondary")
        btn_recache = gr.Button("🔄 Cache neu laden", variant="secondary")

    final_txt = gr.Textbox(label="FINAL Ergebnis (Combined)", interactive=False)
    comb_txt  = gr.Textbox(label="Combined Details", interactive=False)
    back_txt  = gr.Textbox(label="BACK Ergebnis (Details)", interactive=False)
    front_txt = gr.Textbox(label="FRONT Ergebnis (Details)", interactive=False)

    pregrade_txt = gr.Textbox(label="Pregrading (1-10) [Combined]", interactive=False)
    pdf_out = gr.File(label="📄 PDF Report (1 Seite)")

    label_override = gr.Dropdown(
        choices=["AUTO (Model)"] + LABELS,
        value="AUTO (Model)",
        label="Falls falsch: richtiges Label auswählen (optional)"
    )
    note = gr.Textbox(label="Notiz (optional)", placeholder="z.B. 'Back NM, Front minimal Whitening'")

    status_json = gr.JSON(label="Details (Combined + Back + Front)")
    with gr.Row():
        gallery_back = gr.Gallery(label=f"Top-{TOPK} Referenzen (BACK)", columns=4, height="auto")
        gallery_front = gr.Gallery(label=f"Top-{TOPK} Referenzen (FRONT)", columns=4, height="auto")

    save_msg = gr.Textbox(label="Speicher-Status", interactive=False)
    counts = gr.JSON(label="DB Counts")

    def on_analyze(back_pil, front_pil):
        try:
            if back_pil is None or front_pil is None:
                return (
                    None, None, None, [], [], None,
                    "❌ Bitte Rückseite UND Vorderseite hochladen.",
                    "", "", "", "", "",
                    None, "AUTO (Model)", db_counts()
                )

            back_proc_np, back_mask, _back_dbg, back_meta = preprocess_pil_to_proc_rgb(back_pil, side="back")
            front_proc_np, front_mask, _front_dbg, front_meta = preprocess_pil_to_proc_rgb(front_pil, side="front")

            back_proc_prev_pil = Image.fromarray(back_proc_np)
            front_proc_prev_pil = Image.fromarray(front_proc_np)

            if REQUIRE_EXTRACTION and ((not back_meta.get("extracted")) or (not front_meta.get("extracted"))):
                msg = (
                    "❌ Karte konnte nicht sauber erkannt/gewrappt werden (Quad/Warp).\n"
                    "Tipps: Karte näher, gerade, ohne starken Schatten/Glanz; blauer Rand vollständig im Bild."
                )
                return (
                    None,
                    back_proc_prev_pil, front_proc_prev_pil,
                    [], [], None,
                    msg,
                    "", "", "", "", "",
                    None, "AUTO (Model)", db_counts()
                )

            status_b, gal_b, st_b, txt_b, probs_b, idxs_b, sims_b, grade_b, grade_meta_b = analyze_ensemble(
                "back", back_proc_np, n_runs=ENSEMBLE_RUNS
            )
            status_f, gal_f, st_f, txt_f, probs_f, idxs_f, sims_f, grade_f, grade_meta_f = analyze_ensemble(
                "front", front_proc_np, n_runs=ENSEMBLE_RUNS
            )

            status_c, probs_c, st_c, grade_c, grade_meta_c = compute_combined(
                status_b, probs_b, sims_b, st_b, grade_b, grade_meta_b,
                status_f, probs_f, sims_f, st_f, grade_f, grade_meta_f
            )

            run_id = uuid.uuid4().hex[:10]
            plot_path = os.path.join(REPORT_DIR, f"compact_plot_{run_id}.png")
            pdf_path  = os.path.join(REPORT_DIR, f"grading_report_{run_id}.pdf")

            plot_compact_onepage(probs_c, probs_b, probs_f, plot_path)
            build_pdf_report_1page(
                out_pdf_path=pdf_path,
                plot_path=plot_path,
                status_combined=status_c,
                status_back=status_b,
                status_front=status_f,
            )

            pre_txt = pregrading_text(status_c["final_label"], int(grade_c))

            votes_b = status_b.get("ensemble", {}).get("label_counts", {})
            votes_f = status_f.get("ensemble", {}).get("label_counts", {})

            votes_b_str = " | ".join([f"{k}:{v}" for k, v in votes_b.items() if v > 0]) or str(votes_b)
            votes_f_str = " | ".join([f"{k}:{v}" for k, v in votes_f.items() if v > 0]) or str(votes_f)

            final_line = f"FINAL (Combined): {status_c['final_label']} | Grade {int(grade_c)}/10"
            comb_line = f"COMBINED: {status_c['final_label']} | Grade {int(grade_c)}/10 | {status_c.get('final_source')}"
            back_line  = f"{txt_b}\nVotes: {votes_b_str}"
            front_line = f"{txt_f}\nVotes: {votes_f_str}"

            combined_state = {
                "combined": st_c,
                "back": st_b,
                "front": st_f,

                "_last_back_pil": back_pil,
                "_last_front_pil": front_pil,
                "_proc_back_np": back_proc_np,
                "_proc_front_np": front_proc_np,
                "_mask_back_np": back_mask,
                "_mask_front_np": front_mask,
                "_meta_back": back_meta,
                "_meta_front": front_meta,

                "_pdf_path": pdf_path,
                "_grade_combined": int(grade_c),
                "_grade_meta_combined": grade_meta_c,
            }

            status_all = {"combined": status_c, "back": status_b, "front": status_f}

            return (
                status_all,
                back_proc_prev_pil, front_proc_prev_pil,
                gal_b, gal_f,
                combined_state,
                "",
                final_line, comb_line, back_line, front_line,
                pre_txt,
                pdf_path,
                "AUTO (Model)",
                db_counts()
            )

        except Exception as e:
            err = f"❌ Fehler in on_analyze:\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            return (
                None, None, None, [], [], None,
                err,
                "", "", "", "", "",
                None, "AUTO (Model)", db_counts()
            )

    def on_approve(st, user_note, override_choice):
        override = override_choice if override_choice in LABELS else None
        return approve_and_save(st, user_note, override)

    def on_recache():
        rebuild_all_caches()
        return "✅ Cache neu geladen (Back/Front/Combined).", db_counts()

    btn_analyze.click(
        fn=on_analyze,
        inputs=[back_in, front_in],
        outputs=[
            status_json, back_proc_prev, front_proc_prev, gallery_back, gallery_front, state,
            save_msg, final_txt, comb_txt, back_txt, front_txt, pregrade_txt, pdf_out, label_override, counts
        ]
    )

    btn_approve.click(
        fn=on_approve,
        inputs=[state, note, label_override],
        outputs=[save_msg, counts]
    )

    btn_recache.click(
        fn=on_recache,
        inputs=[],
        outputs=[save_msg, counts]
    )

    counts.value = db_counts()


# =========================
# Server Start (freier Port)
# =========================
def guess_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

local_ip = guess_local_ip()
port = get_free_port()

print(f"👉 Öffne am Handy (gleiches WLAN): http://{local_ip}:{port}")
app2.launch(server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app2.launch(server_name="0.0.0.0", server_port=port, share=False)
