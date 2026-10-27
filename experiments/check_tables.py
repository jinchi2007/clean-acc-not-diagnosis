import json, re
from pathlib import Path
ROOT = Path(".").resolve()

def tex_tables():
    txt = (ROOT/"paper"/"main.tex").read_text()
    out = {}
    for m in re.finditer(r"\\label\{(tab:[a-z0-9]+)\}(.*?)\\end\{tabular\}", txt, re.S):
        label, body = m.group(1), m.group(2)
        rows = []
        for line in body.split("\\\\"):
            line = re.sub(r"\\(?:mid|top|bottom)rule", "", line)
            line = re.sub(r"\\cmidrule(?:\(lr\))?\{[^}]*\}", "", line)
            if "&" in line:
                rows.append([c.strip() for c in line.split("&")])
        out[label] = rows
    return out

def mkey(cell):
    """Model name as used in the results JSON (ds-cnn -> ds_cnn)."""
    return clean(cell).lower().replace("-", "_").replace(" ", "")

def clean(cell):
    return cell.replace("\\best{", "").replace("}", "").strip()

def mean_of(cell):
    return clean(cell.split("$\\pm$")[0])

def fracs(cell, nd=3):
    pat = r"-?\d*\.\d{%d}" % nd
    return [float(x) for x in re.findall(pat, mean_of(cell))]

def lead(v):
    return v[0] if isinstance(v, list) else v

tabs = tex_tables()
errs = []
def cmp(tag, got, exp, tol=5e-4):
    if len(got) != len(exp):
        errs.append(f"{tag}: count {len(got)} {got} vs {exp}")
        return
    for g, e in zip(got, exp):
        if abs(g - round(e, 3)) > tol:
            errs.append(f"{tag}: tex {g} != results {e}")

# ---------- tab:sat ----------
sat = json.load(open(ROOT/"results"/"e0_sat.json"))["table"]
for cells in tabs["tab:sat"]:
    name = mkey(cells[0])
    if name not in sat: continue
    got = [x for c in cells[1:] for x in fracs(c)]
    cmp(f"tab:sat {name}", got, [lead(sat[name][c]) for c in ("leaky","safe","crossload","filedisjoint")])

# ---------- tab:paderborn ----------
p = json.load(open(ROOT/"results"/"e8_paderborn.json"))["table"]
last = None
for cells in tabs["tab:paderborn"]:
    model = mkey(cells[0]) or last
    if not model: continue
    last = model
    key = f"{model}:{clean(cells[1]).lower()}"
    if key not in p: continue
    got = [x for c in cells[2:] for x in fracs(c)]
    cmp(f"tab:paderborn {key}", got, [lead(p[key][k]) for k in ("clean","snr10","snr5","snr0")])

# ---------- tab:openset ----------
e3 = json.load(open(ROOT/"results"/"e3_openset.json"))["table"]
for cells in tabs["tab:openset"]:
    name = mkey(cells[0])
    if name not in e3: continue
    got = [x for c in cells[1:] for x in fracs(c)]
    a1, a2 = e3[name]["hold369"], e3[name]["hold147"]
    cmp(f"tab:openset {name}", got,
        [a1["auroc"], a2["auroc"], a1["tpr95"], a2["tpr95"],
         (a1["auroc"]+a2["auroc"])/2, (a1["tpr95"]+a2["tpr95"])/2])

# ---------- tab:noisefam ----------
e6 = json.load(open(ROOT/"results"/"e6_noise_types.json"))["tables"]
e2 = json.load(open(ROOT/"results"/"e2_noise_clean.json"))["table"]
for cells in tabs["tab:noisefam"]:
    name = mkey(cells[0])
    if name not in e6: continue
    got = [x for c in cells[1:] for x in fracs(c)]
    t = e6[name]
    cmp(f"tab:noisefam {name}", got,
        [lead(e2[name]["+10dB"]), lead(t["harmonic"]["10.0"]), lead(t["impulse"]["10.0"]), lead(t["drift"]["10.0"]),
         lead(e2[name]["+0dB"]), lead(t["harmonic"]["0.0"]), lead(t["impulse"]["0.0"]), lead(t["drift"]["0.0"])])

# ---------- tab:eff ----------
e7 = {r["model"]: r for r in json.load(open(ROOT/"results"/"e7_efficiency.json"))["rows"]}
def big(cell):
    out = []
    for x in re.findall(r"\d[\d,]*(?:\.\d+)?", cell.replace("{,}", "").replace("\best{", "").replace("}", "")):
        x = x.rstrip(".,")
        if x: out.append(float(x.replace(",", "")))
    return out
for cells in tabs["tab:eff"]:
    name = mkey(cells[0])
    if name not in e7: continue
    r_ = e7[name]
    got = [x for c in cells[1:] for x in big(c)]
    exp = [float(r_["n_params"]), r_["lat_b1_ms"], r_["lat_b256_ms"],
           r_["lat_b256_per_sample_ms"], float(r_["thr_b256"])]
    if len(got) != len(exp):
        errs.append(f"tab:eff {name}: count {len(got)} {got}")
        continue
    for g, e in zip(got, exp):
        tol = max(6e-3, abs(e) * 3e-3)
        if abs(g - e) > tol:
            errs.append(f"tab:eff {name}: tex {g} != results {e}")

# ---------- tab:stem ----------
st = json.load(open(ROOT/"results"/"stem_ablation.json"))["rows"]
base = {r["backbone"]: r for r in st if r["k"] == 0}
for cells in tabs["tab:stem"]:
    cfg = mkey(cells[0])
    m = re.match(r"([a-z0-9_]+?)(plain|k(\d+)c(\d+))$", cfg)
    if not m: continue
    backbone = m.group(1)
    k = 0 if m.group(2) == "plain" else int(m.group(3))
    c = 0 if m.group(2) == "plain" else int(m.group(4))
    row = next((r for r in st if r["backbone"] == backbone and r["k"] == k and r["c"] == c), None)
    if row is None:
        errs.append(f"tab:stem {cfg}: config not in results")
        continue
    got = [x for cl in cells[2:6] for x in fracs(cl, 4)]
    exp = [row["clean"], row["by_snr"]["10.0"], row["by_snr"]["5.0"], row["by_snr"]["0.0"]]
    if len(got) != 4:
        errs.append(f"tab:stem {cfg}: parsed {got}")
        continue
    for g, e in zip(got, exp):
        if abs(g - round(e, 4)) > 1e-4:
            errs.append(f"tab:stem {cfg}: tex {g} != results {e:.4f}")
    d = (row["by_snr"]["10.0"] - base[backbone]["by_snr"]["10.0"]) * 100
    dt = [float(x) for x in re.findall(r"[+-]?\d+\.\d", clean(cells[-1]))]
    if dt and abs(dt[-1] - round(d, 1)) > 0.051:
        errs.append(f"tab:stem {cfg}: delta10 tex {dt[-1]} != results {d:.1f}")

# ---------- tab:combo ----------
cb = json.load(open(ROOT/"results"/"e5_combo.json"))
for cells in tabs["tab:combo"]:
    model = mkey(cells[0])
    if model not in cb["table"]: continue
    txt = clean(cells[1]).replace("+", "").lower()
    if "stem" in txt and "aug" in txt: cond = "stem_aug"
    elif "stem" in txt: cond = "stem"
    elif "aug" in txt: cond = "aug"
    else: cond = "plain"
    v = cb["table"][model][cond]
    got = [x for cl in cells[2:6] for x in fracs(cl)]
    exp = [lead(v["clean"]), lead(v["10.0"]), lead(v["5.0"]), lead(v["0.0"])]
    if len(got) != 4:
        errs.append(f"tab:combo {model}/{cond}: parsed {got}")
    else:
        for g, e in zip(got, exp):
            if abs(g - round(e, 3)) > 5e-4:
                errs.append(f"tab:combo {model}/{cond}: tex {g} != results {e:.4f}")
    if cond != "plain":
        dd = cb["deltas_vs_plain"][model][cond]
        dt = [float(x) for x in re.findall(r"[-+]?\d+\.\d", clean(cells[-1]))]
        if len(dt) >= 2:
            if abs(dt[0] - round(dd["clean"][0], 1)) > 0.051:
                errs.append(f"tab:combo {model}/{cond}: dclean tex {dt[0]} != {dd['clean'][0]:.1f}")
            if abs(dt[1] - round(dd["10.0"][0], 1)) > 0.051:
                errs.append(f"tab:combo {model}/{cond}: d10 tex {dt[1]} != {dd['10.0'][0]:.1f}")


# ---------- tab:e2 / tab:e4 (4 decimals, ignore the parenthetical std) ----------
def mean4(cell):
    s = re.sub(r"\(.*?\)", "", clean(cell))
    return [float(x) for x in re.findall(r"-?\d*\.\d{4}", s)]

e2t = json.load(open(ROOT/"results"/"e2_noise_clean.json"))["table"]
e4t = json.load(open(ROOT/"results"/"e4_aug.json"))["table"]
COLS = ["clean", "+15dB", "+10dB", "+5dB", "+0dB", "-5dB"]
for label, tbl in (("tab:e2", e2t), ("tab:e4", e4t)):
    for cells in tabs[label]:
        name = mkey(cells[0])
        if name not in tbl: continue
        off = 2 if label == "tab:e2" else 1
        got = [x for c in cells[off:] for x in mean4(c)]
        exp = [lead(tbl[name][c]) for c in COLS]
        if len(got) != len(exp):
            errs.append(f"{label} {name}: count {len(got)} {got}")
            continue
        for g, e in zip(got, exp):
            if abs(g - round(e, 4)) > 6e-5:
                errs.append(f"{label} {name}: tex {g} != results {e:.4f}")

# ---------- tab:wilcoxon ----------
st = json.load(open(ROOT/"results"/"stats.json"))["tables"]["T2_aug_vs_clean"]
for cells in tabs["tab:wilcoxon"]:
    metric = clean(cells[0]).replace(" ", "").lower()
    key = {"clean": "clean", "10db": "10.0 dB", "5db": "5.0 dB", "0db": "0.0 dB"}.get(metric)
    if key is None:
        continue
    d = st[key]
    got = [float(x) for c in cells[1:4] for x in fracs(c, 4)]
    exp = [d["clean_trained"], d["aug"], d["delta"]]
    if len(got) != 3:
        errs.append(f"tab:wilcoxon {metric}: parsed {got}")
        continue
    for g, e in zip(got, exp):
        if abs(g - round(e, 4)) > 6e-5:
            errs.append(f"tab:wilcoxon {metric}: tex {g} != results {e:.4f}")


# ---------- tab:opensetaug ----------
e9 = json.load(open(ROOT/"results"/"e9_openset_aug.json"))["table"]
def e3_mean(name, key):
    return (e3[name]["hold369"][key] + e3[name]["hold147"][key]) / 2
rows_seen = []
for cells in tabs["tab:opensetaug"]:
    name = mkey(cells[0])
    if name == "mean":
        continue
    if name not in e9 or name not in e3:
        continue
    rows_seen.append(name)
    ca, ct = e3_mean(name, "auroc"), e3_mean(name, "tpr95")
    aa, at = e9[name]["mean"]["auroc"], e9[name]["mean"]["tpr95"]
    got = [x for c in cells[1:] for x in fracs(c)]
    exp = [ca, aa, aa - ca, ct, at, at - ct]
    if len(got) != 6:
        errs.append(f"tab:opensetaug {name}: parsed {got}")
        continue
    for g, e in zip(got, exp):
        if abs(g - round(e, 3)) > 6e-4:
            errs.append(f"tab:opensetaug {name}: tex {g} != results {e:.4f}")
pm = e9.get("paired_vs_e3", {}).get("metrics", {})
if pm:
    mean_cells = [c for cells in tabs["tab:opensetaug"] if clean(cells[0]).lower() == "mean" for c in cells]
    if mean_cells:
        got = [x for c in mean_cells[1:] for x in fracs(c)]
        exp = [pm["auroc"]["clean_trained"], pm["auroc"]["augmented"], pm["auroc"]["delta"],
               pm["tpr95"]["clean_trained"], pm["tpr95"]["augmented"], pm["tpr95"]["delta"]]
        if len(got) == 6:
            for g, e in zip(got, exp):
                if abs(g - round(e, 3)) > 6e-4:
                    errs.append(f"tab:opensetaug mean: tex {g} != results {e:.4f}")
        else:
            errs.append(f"tab:opensetaug mean: parsed {got}")

# ---------- tab:scores (E10 decision-layer scores) ----------
e10 = json.load(open(ROOT/"results"/"e10_scores.json"))["table"]
SC = ("softmax", "energy", "mahalanobis")
last_model, is_mean_row = None, False
for cells in tabs["tab:scores"]:
    model_cell = mkey(cells[0].replace("\\textit{", "").replace("}", ""))
    mode = clean(cells[1]).lower()
    if mode not in ("clean", "aug"):
        continue
    if model_cell == "mean":
        is_mean_row = True
    elif model_cell:
        is_mean_row, last_model = False, model_cell
    got = [x for c in cells[2:] for x in fracs(c)]
    if len(got) != 6:
        errs.append(f"tab:scores {model_cell}/{mode}: parsed {got}")
        continue
    if is_mean_row:
        exp = [sum(e10[n][mode][s]["auroc"] for n in e10) / len(e10) for s in SC] + \
              [sum(e10[n][mode][s]["tpr95"] for n in e10) / len(e10) for s in SC]
        tag = f"tab:scores mean/{mode}"
    elif last_model in e10:
        name = last_model
        exp = [e10[name][mode][s]["auroc"] for s in SC] + [e10[name][mode][s]["tpr95"] for s in SC]
        tag = f"tab:scores {name}/{mode}"
    else:
        continue
    for g, e in zip(got, exp):
        if abs(g - round(e, 3)) > 6e-4:
            errs.append(f"{tag}: tex {g} != results {e:.4f}")

print("\n".join(errs) if errs else "OK: 13 tables (sat, e2, e4, wilcoxon, stem, combo, openset, opensetaug, scores, noisefam, eff, paderborn) match results/*.json")
