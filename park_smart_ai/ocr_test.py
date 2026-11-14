"""
Test očitavanja tablica i brzine sistema na skupu za testiranje.

Pokretanje (iz foldera projekta, gdje je main.py):
    python ocr_test.py

Skripta:
  1. na svakoj test slici pokrene YOLO model,
  2. svaku pronađenu tablicu izreže i pošalje na OCR (ista funkcija kao na serveru),
  3. očitani tekst uporedi sa listom stvarnih tablica sa makete,
  4. ispiše rezultate i sačuva detalje.

Rezultat:
  - tabela u terminalu
  - ocr_test_rezultati.csv  (svaka tablica: šta je očitano i da li je tačno)
  - folder ocr_test_crops/  (izrezane tablice, u imenu je očitani tekst)
"""
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from ocr import ocr_iz_cropa

# Stvarne tablice na autićima na maketi
TACNE_TABLICE = {
    "M86-C-722", "J23-U-391", "M82-J-389", "T66-W-166", "K84-M-612", "F25-Z-397",
    "A77-N-711", "J52-H-693", "E59-X-895", "T19-K-454", "E88-N-947",
}

TEST_DIR = Path("dataset/images/test")
LABELS_DIR = Path("dataset/labels/test")
CROPS_DIR = Path("ocr_test_crops")
CROPS_DIR.mkdir(exist_ok=True)

model = YOLO("runs/detect/park_smart/weights/best.pt")
ID_TABLICE = [k for k, v in model.names.items() if v == "tablica"][0]


def levenshtein(a, b):
    """Broj izmjena (zamjena, brisanje, dodavanje znaka) da se od a dobije b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def kategorija(tekst):
    if tekst in TACNE_TABLICE:
        return "tacno"
    if not tekst:
        return "nije_ocitano"
    if len(tekst) <= 2:
        return "jedan_znak"
    if min(levenshtein(tekst, t) for t in TACNE_TABLICE) == 1:
        return "jedan_znak_pogresan"
    return "potpuno_pogresno"


# broj stvarnih tablica u test skupu (iz oznaka)
ukupno_tablica = 0
for lbl in LABELS_DIR.glob("*.txt"):
    with open(lbl) as f:
        ukupno_tablica += sum(1 for red in f if red.split() and int(red.split()[0]) == ID_TABLICE)

slike = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

# zagrijavanje (prvo pokretanje je uvijek sporije)
model.predict(cv2.imread(str(slike[0])), conf=0.5, verbose=False)

redovi = []
vrijeme_yolo, vrijeme_ocr = [], []
brojac = {"tacno": 0, "jedan_znak": 0, "jedan_znak_pogresan": 0, "potpuno_pogresno": 0, "nije_ocitano": 0}

for slika in slike:
    img = cv2.imread(str(slika))

    t0 = time.perf_counter()
    rez = model.predict(img, conf=0.5, verbose=False)
    vrijeme_yolo.append(time.perf_counter() - t0)

    for i, box in enumerate(rez[0].boxes):
        if int(box.cls[0]) != ID_TABLICE:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        t0 = time.perf_counter()
        tekst = ocr_iz_cropa(crop) or ""
        vrijeme_ocr.append(time.perf_counter() - t0)

        kat = kategorija(tekst)
        brojac[kat] += 1

        cv2.imwrite(str(CROPS_DIR / f"{slika.stem}_{i}_{tekst or 'NIJE_OCITANO'}.png"), crop)
        redovi.append([slika.name, f"{float(box.conf[0]):.2f}", tekst, kat])

with open("ocr_test_rezultati.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["slika", "pouzdanost_detekcije", "ocitano", "rezultat"])
    w.writerows(redovi)

pronadjeno = len(redovi)
pogresno = pronadjeno - brojac["tacno"]
pct = lambda x, od: f"{100 * x / od:.1f} %" if od else "-"
prosjek = lambda l: 1000 * sum(l) / len(l) if l else 0

print()
print("=" * 62)
print(" REZULTATI OČITAVANJA TABLICA NA SKUPU ZA TESTIRANJE")
print("=" * 62)
print(f" {'':40}{'Broj':>8}{'Udio':>12}")
print("-" * 62)
print(f" {'Tablice u skupu za testiranje':40}{ukupno_tablica:>8}{'100 %':>12}")
print(f" {'Tablice koje je pronašao YOLO model':40}{pronadjeno:>8}{pct(pronadjeno, ukupno_tablica):>12}")
print(f" {'Tačno očitane tablice':40}{brojac['tacno']:>8}{pct(brojac['tacno'], pronadjeno):>12}")
print(f" {'Pogrešno očitane tablice, od toga:':40}{pogresno:>8}{pct(pogresno, pronadjeno):>12}")
print(f" {'   očitan samo jedan znak':40}{brojac['jedan_znak']:>8}")
print(f" {'   jedan znak pogrešan ili nedostaje':40}{brojac['jedan_znak_pogresan']:>8}")
print(f" {'   potpuno pogrešno očitano':40}{brojac['potpuno_pogresno']:>8}")
if brojac["nije_ocitano"]:
    print(f" {'   nije očitano (ispod praga)':40}{brojac['nije_ocitano']:>8}")
print("-" * 62)
print(f" Tačno očitano od svih tablica: {pct(brojac['tacno'], ukupno_tablica)}")
print(f" Prosječno vrijeme YOLO po slici:    {prosjek(vrijeme_yolo):6.0f} ms")
print(f" Prosječno vrijeme OCR po tablici:   {prosjek(vrijeme_ocr):6.0f} ms")
print("=" * 62)
print(" Detalji: ocr_test_rezultati.csv, izrezane tablice: ocr_test_crops/")