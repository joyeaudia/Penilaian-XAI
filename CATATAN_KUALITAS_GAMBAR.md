# Catatan Kualitas Gambar Paket Penilai XAI V2

**Diperbarui:** 2026-11-17
**Status:** **siap dibagikan** ke 5 penilai. Semua masalah gambar sebelumnya sudah diperbaiki, dan paket dibuat ulang dari website versi terbaru.

Plan: `docs/plans/2026-11-17-peta-tanpa-sambungan-dan-paket-penilai.md`.

## Ringkasan perbaikan

| # | Masalah sebelumnya | Perbaikan | Berlaku juga di website? |
|---|---|---|---|
| 1 | Peta kontrol berupa noise per piksel, mudah ditebak | Kontrol dibuat dari **grid patch peta asli yang sama, digeser melingkar** 5–11 patch (seed tetap), lalu digabung dan digambar lewat jalur yang sama. Bentuk gumpalan dan sebaran nilainya identik; hanya lokasinya pindah. | Tidak (khusus paket) |
| 2 | Garis sambungan di sekitar 20% dan 94% tinggi sampul | View full dan crop **dirata-ratakan**, bukan dijumlahkan (`tail_rich_engine.gabung_view`) | **Ya**, sudah diperbaiki |
| 3 | Warna menutupi hampir seluruh sampul (regangan peringkat) | Gaya **fokus**: skala linear dipotong persentil 95, area lemah dibiarkan bening (`heatmap_overlay_linear`) | **Ya**, sudah diperbaiki |
| 4 | Resolusi kasar 16×16 patch | Tidak bisa diubah; ini batas CLIP ViT-L/14. Peta diperbesar dengan interpolasi halus. | Sama |
| 5 | Token artefak ViT (bercak di logo atau latar) | Tidak diubah. Ini sifat model dan justru bagian yang dinilai penilai. | Sama |
| 6 | Kompresi gambar | Tetap JPEG q86 (maks. tinggi 1000 px) agar HTML ±4 MB dan mudah dikirim | — |
| 7 | Teks salah pasang `TMP-20D70DD55DE3` | Teks dataset sampul ini milik edisi lain ("Sat-Set Menghalau Angket" pada sampul "Skandal Guru Besar Abal-Abal"). Sampul **diganti** `TMP-B49A0E2496DA` ("Minyak Hilang Elpiji Tak Terbilang"): pilihan acak pertama (seed 20261117) yang teksnya diperiksa cocok dengan gambar. Judul dan sinopsis semua item kini diambil langsung dari dataset. | Dataset tidak diubah |

## Catatan untuk koordinator
- Sampul `TMP-20D70DD55DE3` di dataset memiliki teks yang tidak sesuai gambar. Model membaca teks itu saat pelatihan dan analisis. Ini hanya 1 dari 740 sampul.
- Folder simulasi skor (`SIMULASI_BUKAN_DATA_MANUSIA/`) dihapus dari paket. Paket ini hanya untuk mengumpulkan penilaian manusia sungguhan.
