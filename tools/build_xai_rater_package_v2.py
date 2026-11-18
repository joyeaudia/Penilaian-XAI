"""Build the blinded XAI rater package V2 (plan: docs/plans/2026-11-17-paket-penilai-xai-v2.md).

Real maps: website E42 gradient x activation (unified map, `cam_padu`, full/crop views
averaged without seam) for the eight R4-protocol samples. Controls: the same raw patch
grid circularly shifted by a seeded offset (5-11 patches per axis, per view), combined
and drawn through the identical path, so blob shapes and values match the real map. Both use the website overlay
`heatmap_overlay_linear(persentil=95)`. Headline/synopsis come from the dataset CSV.
Images are embedded as base64 so each HTML works on its own.
Plan: docs/plans/2026-11-17-peta-tanpa-sambungan-dan-paket-penilai.md

Usage (from the website repo):
  python tools/build_xai_rater_package_v2.py
"""
from __future__ import annotations

import argparse, base64, csv, hashlib, html, io, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # website repo (tempo-xai-web)
OUT = ROOT / "PAKET_PENILAI_XAI_V2_20261117"
# Eight samples: the R4 rater protocol (research repo, seed 20261101), except
# TMP-20D70DD55DE3, whose dataset headline/supporting text belong to another edition
# ("Sat-Set Menghalau Angket" on the "Skandal Guru Besar Abal-Abal" cover). It is
# replaced by the first seeded pick (seed 20261117) whose text matches its cover.
SAMPLES = ["TMP-0514AFE1ADC9", "TMP-B4159E66B480", "TMP-E8E65B336C7E", "TMP-5802D0009C82",
           "TMP-AE16180EE0D6", "TMP-B49A0E2496DA", "TMP-E9A5F94EF9F7", "TMP-3950DA9B1729"]
DATASET = ROOT / "03_data" / "DATASET_740_TASK2_DOMAIN.csv"
SEED = 20261117
RATERS = [f"R{i}" for i in range(1, 6)]
DIMS = [
    ("visual_clarity", "Kejelasan peta",
     "Seberapa mudah kamu melihat bagian mana yang disorot peta?",
     ["Tidak bisa dibaca sama sekali; warnanya kacau atau menutupi semuanya.",
      "Sulit dibaca; hanya sedikit yang bisa ditangkap.",
      "Sebagian area jelas, sebagian membingungkan.",
      "Umumnya jelas; area yang disorot mudah ditunjuk.",
      "Sangat jelas; area utama langsung terlihat tanpa perlu menebak."]),
    ("visual_relevance", "Keterkaitan dengan isi sampul",
     "Apakah area yang disorot berhubungan dengan isi penting sampul (tokoh, objek, ilustrasi, judul) sesuai topiknya?",
     ["Tidak terkait; yang disorot latar kosong atau pinggiran yang tidak penting.",
      "Sedikit terkait; sebagian besar sorotan di tempat yang tidak penting.",
      "Sebagian terkait; ada elemen penting yang disorot, tapi banyak juga yang tidak.",
      "Cukup terkait; sebagian besar sorotan ada pada elemen penting.",
      "Sangat terkait; sorotan jatuh tepat pada elemen yang paling penting untuk topik sampul."]),
    ("decision_understanding", "Membantu memahami bacaan model",
     "Setelah melihat peta dan 'Bacaan model', apakah kamu jadi lebih paham kenapa model membaca sampul seperti itu?",
     ["Tidak membantu sama sekali.",
      "Sangat sedikit membantu.",
      "Membantu sebagian; ada petunjuk tapi alasannya masih samar.",
      "Cukup membantu; alasan model bisa ditebak dengan masuk akal.",
      "Sangat membantu; jelas bagian gambar mana yang mendasari bacaan model."]),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_texts(samples):
    """Headline and supporting text straight from the dataset (fixes the R4 text mix-up)."""
    import pandas as pd
    d = pd.read_csv(DATASET).set_index("sample_id")
    out = {}
    for sid in samples:
        r = d.loc[sid]
        sup = r.get("supporting_text_verified")
        out[sid] = {"judul": str(r["headline_verified"]).strip(),
                    "sinopsis": "" if not isinstance(sup, str) else sup.strip()}
    return out


def embed(path: Path, max_h=1000) -> str:
    img = Image.open(path).convert("RGB")
    if img.height > max_h:
        img = img.resize((round(img.width * max_h / img.height), max_h), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_maps(web_root: Path, samples, work: Path):
    sys.path.insert(0, str(web_root / "02_web_demo"))
    os.environ.setdefault("TEMPO_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import app as A  # noqa: E402  (website app, backend e42-tail-grid)
    from xai_core import heatmap_overlay_linear
    from tail_rich_engine import gabung_view
    captured = {}
    original = A.ENG.gradcam_terpadu
    def capture(image_path, txt_feat):
        out = original(image_path, txt_feat)
        captured["padu"] = out[0]; captured["mentah"] = {t: m.copy() for t, m in A.ENG._peta_mentah.items()}
        return out
    A.ENG.gradcam_terpadu = capture
    norm = A.ENG._normalize
    client = A.app.test_client()
    models = json.loads(client.get("/api/models").data)
    rng = np.random.default_rng(SEED)
    rows = []
    for sid in samples:
        t = time.time()
        r = client.post("/api/analisis", data={"sample_id": sid, "task": "task2_domain_sumber", "metode_atribusi": "occlusion"})
        if r.status_code != 200:
            raise RuntimeError(f"{sid}: HTTP {r.status_code} {r.data[:200]!r}")
        d = json.loads(r.data); cam = np.asarray(captured.pop("padu"), dtype=np.float32); mentah = captured.pop("mentah")
        img_path = web_root / "05_gambar" / Path(d.get("gambar_url", "")).name
        if not img_path.exists():
            img_path = next((web_root / "05_gambar").glob(f"{sid}.*"))
        # Control: the same patch grid per task and view, circularly shifted by a seeded
        # offset of 5-11 patches on each axis (shared by the three tasks). Blob shapes and
        # values stay identical; only their location on the cover moves.
        side = int(round(next(iter(mentah.values())).shape[1] ** .5))
        shifts = [tuple(int(v) for v in rng.integers(5, side - 4, size=2)) for _ in range(2)]
        geser = lambda v, k: np.roll(v.reshape(side, side), shifts[k], axis=(0, 1)).reshape(-1)
        ctrl_cam = norm(np.mean([norm(gabung_view(np.stack([geser(m[0], 0), geser(m[1], 1)]))) for m in mentah.values()], axis=0))
        orig = work / f"{sid}_original.jpg"; Image.open(img_path).convert("RGB").save(orig, quality=95)
        real = work / f"{sid}_gradxact.png"; heatmap_overlay_linear(img_path, cam, real, persentil=95)
        ctrl = work / f"{sid}_kontrol_acak.png"; heatmap_overlay_linear(img_path, ctrl_cam, ctrl, persentil=95)
        pred = d["prediksi"]
        bacaan = {k: pred[k].get("prediksi_tampil") or pred[k]["prediksi"] for k in pred}
        rows.append(dict(sample_id=sid, cam_shape=list(cam.shape), original=orig, real=real, control=ctrl,
                         bacaan=bacaan, model_version=pred["task1_issue"].get("model_version"), seconds=round(time.time() - t, 1)))
        print("map", sid, rows[-1]["seconds"], "s", flush=True)
    return rows, models


CSS = """
:root{color-scheme:light;--bg:#f6f7f9;--card:#fff;--ink:#17212b;--muted:#5b6876;--line:#d9dee4;--accent:#1f5fae;--ok:#1d7a46;--warn:#9a5b00;--soft:#eef3fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 "Times New Roman",Times,serif}
main{max-width:1000px;margin:0 auto;padding:16px 16px 120px}h1{font-size:1.6rem;margin:.6em 0 .2em}h2{font-size:1.25rem;margin:1.4em 0 .5em}h3{margin:.2em 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
.soft{background:var(--soft)}.muted{color:var(--muted)}.pill{display:inline-block;padding:2px 10px;border-radius:99px;background:var(--soft);margin:2px 4px 2px 0;font-size:.9rem}
ol li,ul li{margin:.25em 0}table{border-collapse:collapse;width:100%;font-size:.93rem}th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
.tablewrap{overflow-x:auto}.legend{height:14px;border-radius:7px;background:linear-gradient(90deg,#0000ff00,#2a6cf0,#3cc06a,#f2d43c,#e8402a);border:1px solid var(--line);margin:6px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pair figure{margin:0}.pair img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#000}
figcaption{font-size:.85rem;color:var(--muted);text-align:center}.q{border-top:1px dashed var(--line);padding-top:10px;margin-top:10px}
.scale{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}.scale label{flex:1 1 52px;min-width:52px;border:1px solid var(--line);border-radius:8px;text-align:center;padding:6px 0;cursor:pointer;user-select:none}
.scale input{position:absolute;opacity:0}.scale label:has(input:checked){background:var(--accent);color:#fff;border-color:var(--accent)}
.scale label:has(input:focus-visible){outline:2px solid var(--accent);outline-offset:2px}.hint{font-size:.85rem;color:var(--muted);min-height:1.3em}
textarea{width:100%;min-height:64px;font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.status{font-size:.9rem;font-weight:600}.status.done{color:var(--ok)}.status.todo{color:var(--warn)}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;box-shadow:0 -4px 14px #0002}
.bar .in{max-width:1000px;margin:0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.prog{flex:1 1 200px;height:10px;background:var(--soft);border-radius:5px;overflow:hidden}
.prog i{display:block;height:100%;background:var(--ok);width:0}button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
button.ghost{background:transparent;color:var(--accent)}@media (max-width:640px){.pair{grid-template-columns:1fr}h1{font-size:1.35rem}}
@page{size:A4;margin:12mm 12mm}
@media print{body{background:#fff;font-size:11pt;line-height:1.35}main{max-width:none;padding:0}.bar{display:none}
.card{border:1px solid #bbb;padding:8px 12px;margin:8px 0;break-inside:auto}h1{margin-top:0}h2{break-after:avoid}
main>h2,main>section:last-of-type{display:none}article.card{break-before:page;page-break-before:always;break-inside:avoid;page-break-inside:avoid;border:none;padding:0;margin:0}
.pair{grid-template-columns:1fr 1fr;gap:8px}.pair img{max-height:92mm;width:auto;max-width:100%;margin:0 auto;display:block;background:#fff}
figcaption{font-size:9pt}.q{padding-top:5px;margin-top:5px}.q .muted{font-size:9.5pt}
.scale{margin:3px 0}.scale label{border:1px solid #555;padding:1px 0}.scale label:has(input:checked){background:#fff;color:#000}.hint,.status{display:none}
textarea{min-height:24mm;border:1px solid #555;resize:none}textarea::placeholder{color:transparent}
.soft{background:#f1f4f8;-webkit-print-color-adjust:exact;print-color-adjust:exact}.legend,.pill{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""

JS = r"""
const RATER=document.body.dataset.rater, KEY='xai-v2-draft-'+RATER;
const DIMS=JSON.parse(document.getElementById('dims').textContent);
const ITEMS=[...document.querySelectorAll('article[data-item]')].map(a=>a.dataset.item);
let draft={};try{draft=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){draft={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(draft))}catch(e){}}
function complete(id){const d=draft[id]||{};return DIMS.every(x=>d[x.id])&&(d.alasan||'').trim().length>=5}
function refresh(){let n=0;ITEMS.forEach(id=>{const ok=complete(id);if(ok)n++;const s=document.getElementById('st-'+id);
  s.textContent=ok?'Selesai':'Belum lengkap';s.className='status '+(ok?'done':'todo')});
  document.getElementById('progtext').textContent=n+' dari '+ITEMS.length+' item selesai';
  document.getElementById('progbar').style.width=(100*n/ITEMS.length)+'%'}
document.querySelectorAll('article[data-item]').forEach(a=>{const id=a.dataset.item;draft[id]=draft[id]||{};
  a.querySelectorAll('input[type=radio]').forEach(r=>{if(draft[id][r.name.split('|')[1]]===r.value)r.checked=true;
    r.addEventListener('change',()=>{draft[id][r.name.split('|')[1]]=r.value;save();refresh();showHint(r)})});
  const t=a.querySelector('textarea');t.value=draft[id].alasan||'';t.addEventListener('input',()=>{draft[id].alasan=t.value;save();refresh()})});
function showHint(r){const dim=DIMS.find(x=>x.id===r.name.split('|')[1]);document.getElementById('h-'+r.name.replace('|','-')).textContent=r.value+' = '+dim.skala[r.value-1]}
document.querySelectorAll('input[type=radio]:checked').forEach(showHint);
function q(s){s=String(s).replace(/\r?\n/g,' ').trim();return /[",]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
document.getElementById('unduh').addEventListener('click',()=>{const kurang=ITEMS.filter(id=>!complete(id));
  if(kurang.length&&!confirm('Masih ada '+kurang.length+' item belum lengkap ('+kurang.join(', ')+').\nFile tetap diunduh, tetapi koordinator hanya bisa memakai file yang lengkap.\nLanjut unduh?'))return;
  const rows=['item_id,rater_id,dimension,score,alasan_singkat'];
  ITEMS.forEach(id=>DIMS.forEach(x=>rows.push([id,RATER,x.id,(draft[id]||{})[x.id]||'',q((draft[id]||{}).alasan||'')].join(','))));
  const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='ratings_'+RATER+'.csv';document.body.append(a);a.click();a.remove()});
document.getElementById('reset').addEventListener('click',()=>{if(confirm('Hapus semua isian di halaman ini? Tindakan ini tidak bisa dibatalkan.')){draft={};save();location.reload()}});
refresh();
"""


def rater_html(rater, items):
    e = html.escape
    rub = "".join(
        f"<tr><th>{e(lbl)}<div class='muted' style='font-weight:400'>{e(qs)}</div></th>"
        + "".join(f"<td><b>{i+1}</b> — {e(s)}</td>" for i, s in enumerate(sk)) + "</tr>"
        for _, lbl, qs, sk in DIMS)
    arts = []
    for n, it in enumerate(items, 1):
        qhtml = ""
        for did, lbl, qs, sk in DIMS:
            opts = "".join(f"<label title='{e(sk[v-1])}'><input type='radio' name='{it['item_id']}|{did}' value='{v}'>{v}</label>" for v in range(1, 6))
            qhtml += (f"<div class='q'><b>{e(lbl)}</b><div class='muted'>{e(qs)}</div>"
                      f"<div class='scale' role='radiogroup' aria-label='{e(lbl)}'>{opts}</div>"
                      f"<div class='hint' id='h-{it['item_id']}-{did}'></div></div>")
        b = it["bacaan"]
        arts.append(f"""<article class="card" data-item="{it['item_id']}" id="{it['item_id']}">
<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap"><h3>Item {n} dari {len(items)} · {it['item_id']}</h3><span class="status todo" id="st-{it['item_id']}">Belum lengkap</span></div>
<p style="margin:.2em 0"><b>Judul sampul:</b> {e(it['judul'])}</p><p class="muted" style="margin:.2em 0 .6em">{e(it['sinopsis'])}</p>
<div class="pair"><figure><img src="{it['img_original']}" alt="Sampul asli {it['item_id']}"><figcaption>Sampul asli</figcaption></figure>
<figure><img src="{it['img_map']}" alt="Peta panas {it['item_id']}"><figcaption>Peta panas yang dinilai</figcaption></figure></div>
<p class="soft" style="padding:8px 10px;border-radius:8px;margin:10px 0 0"><b>Bacaan model:</b>
<span class="pill">Topik: {e(b['task1_issue'])}</span><span class="pill">Jenis kiasan: {e(b['task2_domain_sumber'])}</span><span class="pill">Pihak disorot: {e(b['task3_primary_entity_type'])}</span></p>
{qhtml}
<div class="q"><b>Alasan singkat (wajib)</b><div class="muted">1–2 kalimat dengan kata-katamu sendiri: apa yang kamu lihat pada peta dan kenapa memberi skor itu.</div>
<textarea aria-label="Alasan singkat {it['item_id']}" placeholder="Contoh: sorotan paling kuat di wajah tokoh dan judul, tapi latar kanan ikut merah sehingga agak membingungkan."></textarea></div>
</article>""")
    dims_json = json.dumps([{"id": d, "skala": sk} for d, _, _, sk in DIMS], ensure_ascii=False)
    return f"""<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Penilaian Peta XAI {rater}</title><style>{CSS}</style></head><body data-rater="{rater}"><main>
<h1>Penilaian Peta Penjelasan Model — Penilai {rater}</h1>
<p class="muted">Penelitian dekoding metafora sampul Majalah Tempo · 16 item · perkiraan waktu 20–30 menit</p>

<section class="card soft"><h2 style="margin-top:0">Ringkasnya, apa yang kamu kerjakan?</h2>
<p>Kamu melihat 16 pasang gambar: <b>sampul asli</b> dan <b>peta panas</b> di atas sampul yang sama. Untuk setiap item kamu memberi <b>tiga skor 1–5</b> dan menulis <b>satu alasan singkat</b>. Setelah semua selesai, klik <b>Unduh ratings_{rater}.csv</b> di bagian bawah, lalu kirim file itu ke koordinator.</p></section>

<section class="card"><h2 style="margin-top:0">1. Apa itu peta panas?</h2>
<p>Sebuah model AI membaca sampul Tempo (gambar + judul) lalu menebak topiknya, jenis kiasan/metaforanya, dan pihak yang disorot. Peta panas mencoba menunjukkan <b>bagian gambar yang paling berpengaruh</b> terhadap bacaan model itu.</p>
<div class="legend" aria-hidden="true"></div><p class="muted" style="margin-top:0">Bening/biru = pengaruh kecil · hijau/kuning = sedang · <b>merah = pengaruh paling besar</b>.</p>
<p>Peta tidak selalu bagus. Ada peta yang jelas dan masuk akal, ada yang kabur atau menyorot tempat tidak penting. <b>Tugasmu menilai seberapa berguna peta itu menurut pandanganmu sendiri</b>, bukan menilai apakah model benar.</p>
<p>Beberapa sampul muncul dua kali dengan peta berbeda. Nilai setiap item <b>apa adanya</b>; tidak perlu membandingkan atau menebak alasannya.</p></section>

<section class="card"><h2 style="margin-top:0">2. Aturan penting</h2><ul>
<li><b>Kerjakan sendiri.</b> Jangan berdiskusi, melihat, atau menyalin isian penilai lain, dan jangan membagikan isianmu.</li>
<li><b>Tidak ada jawaban benar atau salah.</b> Kami butuh penilaian jujurmu, termasuk skor rendah bila peta memang tidak membantu.</li>
<li><b>Pakai seluruh rentang 1–5.</b> Jangan memberi skor yang sama untuk semua item kalau kesanmu memang berbeda.</li>
<li><b>Alasan ditulis dengan kata-katamu sendiri</b>, minimal satu kalimat per item. Jangan menyalin alasan yang sama untuk semua item.</li>
<li>Kamu boleh berhenti di tengah jalan. Isian tersimpan otomatis di browser ini; buka lagi file yang sama di <b>perangkat dan browser yang sama</b> untuk melanjutkan.</li></ul></section>

<section class="card"><h2 style="margin-top:0">3. Langkah kerja</h2><ol>
<li>Baca rubrik di bawah sekali sampai paham.</li>
<li>Untuk setiap item: bandingkan sampul asli dengan peta panas, baca judul dan <b>Bacaan model</b>.</li>
<li>Pilih skor 1–5 untuk ketiga pertanyaan. Arti angka yang kamu pilih muncul di bawah tombol.</li>
<li>Tulis alasan singkat. Status item berubah menjadi <span class="status done">Selesai</span> bila tiga skor dan alasan sudah terisi.</li>
<li>Setelah 16 item selesai, klik <b>Unduh ratings_{rater}.csv</b> di bar bawah.</li>
<li>Kirim file <b>ratings_{rater}.csv</b> ke koordinator. Jangan mengubah isi atau nama file.</li></ol></section>

<section class="card"><h2 style="margin-top:0">4. Rubrik skor</h2><div class="tablewrap"><table>
<tr><th>Pertanyaan</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr>{rub}</table></div>
<p class="muted">Skor ini mengukur seberapa mudah dipahami dan relevan peta menurut manusia, bukan bukti bahwa model benar.</p></section>

<h2>5. Item yang dinilai</h2>
{''.join(arts)}
<section class="card soft"><h2 style="margin-top:0">Selesai?</h2><p>Pastikan bar bawah menunjukkan <b>16 dari 16 item selesai</b>, klik <b>Unduh ratings_{rater}.csv</b>, lalu kirim file itu ke koordinator. Terima kasih!</p></section>
</main>
<div class="bar"><div class="in"><span id="progtext" class="status">0 dari 16 item selesai</span><div class="prog"><i id="progbar"></i></div>
<button id="unduh" type="button">Unduh ratings_{rater}.csv</button><button id="reset" class="ghost" type="button">Hapus isian</button></div></div>
<script type="application/json" id="dims">{dims_json}</script><script>{JS}</script></body></html>"""


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    web = ROOT
    samples = SAMPLES
    info = dataset_texts(samples)
    coord = OUT / "KOORDINATOR"; srcdir = coord / "gambar_sumber"; srcdir.mkdir(parents=True, exist_ok=True)
    maps, models = generate_maps(web, samples, srcdir)
    entries = []
    for m in maps:
        for kind, path in (("asli_gradxact", m["real"]), ("kontrol_acak", m["control"])):
            entries.append(dict(m, kind=kind, map_path=path))
    order = np.random.default_rng(SEED).permutation(len(entries))
    items, key = [], []
    for n, idx in enumerate(order, 1):
        e = entries[idx]; iid = f"XAI-{n:03d}"
        items.append(dict(item_id=iid, judul=info[e["sample_id"]]["judul"], sinopsis=info[e["sample_id"]]["sinopsis"],
                          bacaan=e["bacaan"], img_original=embed(e["original"]), img_map=embed(e["map_path"])))
        key.append(dict(item_id=iid, sample_id=e["sample_id"], jenis=e["kind"], file=e["map_path"].name, sha256=sha(e["map_path"])))
    with open(coord / "KUNCI_BLIND.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    commit = subprocess.run(["git", "-C", str(web), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    protocol = dict(schema="xai-rater-package-v2", seed=SEED, sample_ids=samples, sample_source="R4 rater protocol, tempo-clip-indobert-xai delivery/HASIL_REVISI_JOY_20261108_R4 (seed 20261101)", text_source="03_data/DATASET_740_TASK2_DOMAIN.csv",
                    website_commit=commit, backend=models, model_version=maps[0]["model_version"],
                    real_map="E42 website |gradient x activation| CLIP ViT-L/14 resblocks[-2], full/crop views averaged (tail_rich_engine.gabung_view), unified map (cam_padu) over three task heads, held-out fold for dataset samples",
                    control_map="same raw patch grid as the real map, circularly shifted by a seeded offset of 5-11 patches per axis and view (np.roll), combined and normalised through the identical path; shapes and value distribution identical, location displaced",
                    rendering="website xai_core.heatmap_overlay_linear(persentil=95) for both real and control; embedded as JPEG q86 max height 1000 px",
                    cam_shapes=sorted({tuple(m["cam_shape"]) for m in maps}), items=len(items), raters=RATERS,
                    rater_shows_model_reading=True, map_seconds={m["sample_id"]: m["seconds"] for m in maps})
    (coord / "PROTOKOL.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in RATERS:
        d = OUT / f"PENILAI_XAI_{r}"; d.mkdir(parents=True, exist_ok=True)
        (d / "MULAI_DI_SINI.html").write_text(rater_html(r, items), encoding="utf-8")
        with open(d / "ratings.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["item_id", "rater_id", "dimension", "score", "alasan_singkat"])
            for it in items:
                for did, *_ in DIMS: w.writerow([it["item_id"], r, did, "", ""])
    files = {str(p.relative_to(OUT)).replace("\\", "/"): sha(p) for p in sorted(OUT.rglob("*"))
             if p.is_file() and p.name != "MANIFEST_SHA256.json"}
    (OUT / "MANIFEST_SHA256.json").write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")
    print("done", len(items), "items;", len(files), "files")


if __name__ == "__main__":
    main()
