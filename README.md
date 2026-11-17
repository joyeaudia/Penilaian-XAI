# Paket Penilai XAI V2 (17 November 2026)

Panduan ini untuk **koordinator**. Penilai cukup menerima satu file HTML miliknya.

## Isi folder

| Folder | Untuk siapa | Isi |
|---|---|---|
| `PENILAI_XAI_R1/` … `PENILAI_XAI_R5/` | masing-masing penilai | `MULAI_DI_SINI.html` (gambar sudah tertanam, bisa dikirim sendirian) dan `ratings.csv` (template, untuk validasi koordinator) |
| `KOORDINATOR/` | **koordinator saja, jangan dibagikan** | `KUNCI_BLIND.csv` (item mana peta asli dan mana kontrol), `PROTOKOL.json`, dan `gambar_sumber/` resolusi penuh |

## Yang dinilai

- 8 sampul dari protokol R4 (seed 20261101). `TMP-20D70DD55DE3` diganti `TMP-B49A0E2496DA` karena teks datasetnya milik edisi lain (lihat `CATATAN_KUALITAS_GAMBAR.md`).
- Setiap sampul muncul dua kali, jadi totalnya 16 item, diacak dengan seed 20261117:
  - **peta asli**: |gradient × activation| E42 website (CLIP ViT-L/14 `resblocks[-2]`, view full dan crop dirata-ratakan, peta gabungan tiga head, checkpoint held-out fold);
  - **peta kontrol**: grid patch peta asli yang sama, digeser melingkar 5–11 patch (seed tetap). Bentuk dan sebaran nilainya identik, hanya lokasinya pindah.
- Kedua jenis peta digambar dengan fungsi website yang sama, `heatmap_overlay_linear(persentil=95)`, persis seperti peta di tab Bukti gambar.
- Penilai juga melihat **Bacaan model** (prediksi Task 1/2/3 E42), supaya dimensi "membantu memahami bacaan model" punya acuan. Paket R4 lama tidak menampilkan ini.
- Rubrik: kejelasan peta, keterkaitan dengan isi sampul, dan membantu memahami bacaan model, masing-masing skala 1–5 dengan deskripsi tiap angka.

## Cara menjalankan penilaian

1. Pilih **lima orang berbeda** yang tidak ikut anotasi dataset.
2. Kirim ke penilai R*n* hanya file `PENILAI_XAI_R`*n*`/MULAI_DI_SINI.html` miliknya. Jangan kirim folder `KOORDINATOR/`.
3. Minta mereka mengerjakan sendiri (sekitar 20–30 menit), lalu mengirim balik `ratings_R`*n*`.csv` hasil tombol unduh.
4. Sebelum analisis, cek tiap file: 48 baris, skor 1–5 lengkap, alasan tidak kosong, dan alasan antar-penilai **tidak identik**. Isian identik akan ditolak lagi seperti R34 lama.
5. Hitung rerata per dimensi untuk **peta asli dan kontrol secara terpisah** (pakai `KOORDINATOR/KUNCI_BLIND.csv`), lalu hitung ICC antar-penilai. Peta asli baru dianggap membantu bila skornya lebih tinggi daripada kontrol.

## Regenerasi

```bash
python tools/build_xai_rater_package_v2.py   (dijalankan dari J:/tempo-xai-web)
```

Script ini deterministik (seed tetap). Commit website yang dipakai tercatat di `KOORDINATOR/PROTOKOL.json`. Plan: `docs/plans/2026-11-17-paket-penilai-xai-v2.md` dan `docs/plans/2026-11-17-peta-tanpa-sambungan-dan-paket-penilai.md`. Status kualitas gambar dan daftar perbaikan ada di `CATATAN_KUALITAS_GAMBAR.md`; paket **siap dibagikan**.
