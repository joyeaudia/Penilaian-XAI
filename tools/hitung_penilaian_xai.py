"""Hitung rata-rata penilaian peta XAI dari CSV lima penilai.

Plan: docs/plans/2026-11-17-script-hitung-penilaian-xai.md

Cara pakai (dari folder J:/tempo-xai-web):
  python tools/hitung_penilaian_xai.py
  python tools/hitung_penilaian_xai.py --folder "D:/hasil_penilai"   # folder berisi ratings_R1.csv ... ratings_R5.csv

Yang dihitung:
  1. Rata-rata skor per penilai.
  2. Rata-rata per dimensi untuk PETA ASLI dan PETA KONTROL (pakai kunci blind), serta selisihnya.
     Peta XAI baru terbukti membantu bila skor peta asli lebih tinggi daripada kontrol.
  3. Rata-rata keseluruhan.
  4. ICC(2,1) dan ICC(2,k): seberapa konsisten para penilai.
Sebelum angka dicetak, isian diperiksa keabsahannya.
"""
from __future__ import annotations

import argparse
import itertools
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Jalan dari J:/tempo-xai-web (paket = subfolder) maupun dari repo paket itu sendiri.
PAKET = ROOT if (ROOT / "KOORDINATOR").is_dir() else ROOT / "PAKET_PENILAI_XAI_V2_20261117"
KUNCI = PAKET / "KOORDINATOR" / "KUNCI_BLIND.csv"
DIMENSI = {"visual_clarity": "Kejelasan peta",
           "visual_relevance": "Keterkaitan dengan isi sampul",
           "decision_understanding": "Membantu memahami bacaan model"}
BARIS_PER_PENILAI = 16 * len(DIMENSI)


def cari_file(folder: Path | None) -> dict[str, Path]:
    """ratings_R*.csv dari tombol unduh; bila tidak ada, template PENILAI_XAI_R*/ratings.csv."""
    if folder:
        files = sorted(folder.glob("ratings_R*.csv"))
        return {re.search(r"ratings_(R\d+)\.csv", f.name).group(1): f for f in files}
    files = sorted(PAKET.glob("ratings_R*.csv")) or sorted(PAKET.glob("PENILAI_XAI_R*/ratings.csv"))
    out = {}
    for f in files:
        m = re.search(r"ratings_(R\d+)\.csv", f.name) or re.search(r"PENILAI_XAI_(R\d+)", str(f))
        out[m.group(1)] = f
    return out


def periksa(data: dict[str, pd.DataFrame], files: dict[str, Path]) -> list[str]:
    """Kembalikan daftar masalah. Kosong berarti isian lolos pemeriksaan dasar."""
    masalah = []
    for r, d in data.items():
        if len(d) != BARIS_PER_PENILAI:
            masalah.append(f"{r}: {len(d)} baris (seharusnya {BARIS_PER_PENILAI}).")
        if d["score"].isna().any() or not d["score"].between(1, 5).all():
            masalah.append(f"{r}: ada skor kosong atau di luar 1-5.")
        if (d["alasan_singkat"].fillna("").str.strip().str.len() < 5).any():
            masalah.append(f"{r}: ada alasan kosong atau terlalu pendek.")
        ids = sorted(d["rater_id"].astype(str).unique())
        if ids != [r]:
            masalah.append(f"{r}: ID penilai di dalam file {ids}, tidak sama dengan nama file ({r}). "
                           "File hasil tombol unduh selalu memuat ID yang benar.")
    penilai = list(data)
    if len(penilai) >= 2:
        alasan = {r: data[r].drop_duplicates("item_id").set_index("item_id")["alasan_singkat"].fillna("") for r in penilai}
        item = alasan[penilai[0]].index
        kembar = sum(alasan[a][i].strip() == alasan[b][i].strip()
                     for a, b in itertools.combinations(penilai, 2) for i in item if i in alasan[b].index)
        if kembar:
            masalah.append(f"Ada {kembar} pasangan alasan yang identik antar-penilai.")
        mirip = np.mean([SequenceMatcher(None, alasan[a][i], alasan[b][i]).ratio()
                         for a, b in itertools.combinations(penilai, 2) for i in item if i in alasan[b].index])
        if mirip > 0.45:
            masalah.append(f"Alasan antar-penilai sangat mirip (rata-rata kemiripan teks {mirip:.2f}); "
                           "periksa apakah ditulis sendiri-sendiri.")
        skor = pd.DataFrame({r: data[r].sort_values(["item_id", "dimension"])["score"].to_numpy() for r in penilai})
        sama = float((skor.nunique(axis=1) == 1).mean())
        if sama > 0.5:
            masalah.append(f"{sama:.0%} baris diberi skor yang persis sama oleh semua penilai; "
                           "tidak lazim untuk penilai yang bekerja sendiri-sendiri.")
        waktu = sorted(f.stat().st_mtime for f in files.values())
        if len(waktu) >= 3 and waktu[-1] - waktu[0] < 15 * 60:
            masalah.append(f"Semua file disimpan dalam rentang {(waktu[-1] - waktu[0]) / 60:.1f} menit, "
                           "padahal satu penilai butuh 20-30 menit.")
    return masalah


def icc(matriks: np.ndarray) -> tuple[float, float]:
    """ICC(2,1) dan ICC(2,k) (Shrout & Fleiss): baris = item, kolom = penilai."""
    n, k = matriks.shape
    rata = matriks.mean()
    msr = k * ((matriks.mean(axis=1) - rata) ** 2).sum() / (n - 1)
    msc = n * ((matriks.mean(axis=0) - rata) ** 2).sum() / (k - 1)
    sse = ((matriks - matriks.mean(axis=1, keepdims=True) - matriks.mean(axis=0, keepdims=True) + rata) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc1 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    icck = (msr - mse) / (msr + (msc - mse) / n)
    return float(icc1), float(icck)


def main():
    ap = argparse.ArgumentParser(description="Hitung rata-rata penilaian peta XAI.")
    ap.add_argument("--folder", type=Path, help="folder berisi ratings_R1.csv ... ratings_R5.csv")
    a = ap.parse_args()

    files = cari_file(a.folder)
    if not files:
        raise SystemExit("Tidak ada file ratings ditemukan.")
    data = {r: pd.read_csv(f) for r, f in files.items()}
    kunci = pd.read_csv(KUNCI)[["item_id", "sample_id", "jenis"]]

    print("=" * 70)
    print("PEMERIKSAAN KEABSAHAN ISIAN")
    print("=" * 70)
    masalah = periksa(data, files)
    if masalah:
        for m in masalah:
            print("  [!] " + m)
        print("\n  STATUS: TIDAK LOLOS. Angka di bawah hanya untuk diperiksa,")
        print("  JANGAN dilaporkan sebagai hasil penilaian manusia.")
    else:
        print("  Lolos pemeriksaan dasar.")

    semua = pd.concat([d.assign(penilai=r) for r, d in data.items()], ignore_index=True)
    semua = semua.merge(kunci, on="item_id", how="left")
    semua["jenis"] = semua["jenis"].map({"asli_gradxact": "Peta asli", "kontrol_acak": "Kontrol"})

    print("\n" + "=" * 70)
    print("1. RATA-RATA PER PENILAI (semua item)")
    print("=" * 70)
    print(semua.groupby("penilai")["score"].mean().round(2).to_string())

    print("\n" + "=" * 70)
    print("2. RATA-RATA PER DIMENSI: PETA ASLI vs KONTROL")
    print("=" * 70)
    tabel = semua.pivot_table(index="dimension", columns="jenis", values="score", aggfunc="mean")
    tabel = tabel.reindex(list(DIMENSI))
    tabel.index = [DIMENSI[d] for d in tabel.index]
    tabel["Selisih (asli - kontrol)"] = tabel["Peta asli"] - tabel["Kontrol"]
    print(tabel.round(2).to_string())

    print("\n" + "=" * 70)
    print("3. RATA-RATA KESELURUHAN")
    print("=" * 70)
    for jenis, nilai in semua.groupby("jenis")["score"].mean().items():
        print(f"  {jenis:10s}: {nilai:.2f}")
    print(f"  {'Semua item':10s}: {semua['score'].mean():.2f}")

    print("\n" + "=" * 70)
    print("4. KONSISTENSI ANTAR-PENILAI (ICC)")
    print("=" * 70)
    for dim, nama in DIMENSI.items():
        m = semua[semua["dimension"] == dim].pivot_table(index="item_id", columns="penilai", values="score").to_numpy(float)
        if m.shape[1] >= 2:
            i1, ik = icc(m)
            print(f"  {nama:32s}: ICC(2,1) = {i1:.2f}   ICC(2,k) = {ik:.2f}")
    print("\n  ICC(2,1) = konsistensi satu penilai; ICC(2,k) = konsistensi rata-rata semua penilai.")
    print("  Pedoman umum: < 0,50 lemah · 0,50-0,75 sedang · 0,75-0,90 baik · > 0,90 sangat baik.")


if __name__ == "__main__":
    main()
