"""
Pravi sliku za rad: detekcija tablica -> izrezane tablice -> očitani tekst.
"""
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch

SLIKA = sys.argv[1] if len(sys.argv) > 1 else "dataset/images/test/52199184-IMG_0961.png"

ZELENA = (46, 160, 67)          # boja okvira (RGB)
ZELENA_MPL = tuple(c / 255 for c in ZELENA)
CROP_W, CROP_H = 360, 110       # sve izrezane tablice se prikazuju u istoj veličini


def detektuj_i_ocitaj(putanja):
    from ultralytics import YOLO
    from ocr import ocr_iz_cropa

    model = YOLO("runs/detect/park_smart/weights/best.pt")
    img = cv2.imread(putanja)
    if img is None:
        sys.exit(f"Ne mogu otvoriti sliku: {putanja}")

    rez = model.predict(img, conf=0.5, verbose=False)
    tablice = []
    for box in rez[0].boxes:
        if model.names[int(box.cls[0])] != "tablica":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        tekst = ocr_iz_cropa(crop) or "—"
        tablice.append(((x1, y1, x2, y2), crop, tekst, float(box.conf[0])))

    if not tablice:
        sys.exit("Model nije pronašao nijednu tablicu na ovoj slici. Probaj drugu sliku.")

    # numeracija odozgo prema dolje, pa slijeva nadesno
    tablice.sort(key=lambda t: (t[0][1], t[0][0]))
    return img, tablice


def pripremi_crop(crop):
    """Okreće tablicu da bude položena i smješta je na bijelu podlogu iste veličine."""
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    h, w = crop.shape[:2]
    k = min((CROP_W - 20) / w, (CROP_H - 20) / h)
    crop = cv2.resize(crop, (int(w * k), int(h * k)), interpolation=cv2.INTER_CUBIC)
    platno = 255 * __import__("numpy").ones((CROP_H, CROP_W, 3), dtype="uint8")
    y0 = (CROP_H - crop.shape[0]) // 2
    x0 = (CROP_W - crop.shape[1]) // 2
    platno[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return platno


def nacrtaj(img, tablice, izlaz="ocr_primjer.png"):
    oznacena = img.copy()
    deb = max(3, img.shape[1] // 250)
    for i, ((x1, y1, x2, y2), _, _, _) in enumerate(tablice, 1):
        cv2.rectangle(oznacena, (x1, y1), (x2, y2), ZELENA[::-1], deb)
        # broj u zelenom kvadratu pored okvira
        s = max(40, img.shape[1] // 22)
        bx, by = max(0, x1 - s - deb), max(0, y1)
        cv2.rectangle(oznacena, (bx, by), (bx + s, by + s), ZELENA[::-1], -1)
        cv2.putText(oznacena, str(i), (bx + s // 4, by + int(s * 0.78)),
                    cv2.FONT_HERSHEY_SIMPLEX, s / 38, (255, 255, 255), max(2, deb - 1), cv2.LINE_AA)

    n = len(tablice)
    fig = plt.figure(figsize=(11, 5.6), dpi=200)
    outer = GridSpec(1, 2, width_ratios=[1, 1.35], wspace=0.08, figure=fig)

    # a) cijela slika
    ax = fig.add_subplot(outer[0])
    ax.imshow(cv2.cvtColor(oznacena, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    # b) i c) jedan red po tablici
    desno = GridSpecFromSubplotSpec(n, 3, subplot_spec=outer[1],
                                    width_ratios=[0.18, 1.25, 1], hspace=0.25, wspace=0.08)

    redovi = []
    for i, (_, crop, tekst, conf) in enumerate(tablice, 1):
        r = i - 1
        a_br = fig.add_subplot(desno[r, 0])
        a_br.add_patch(FancyBboxPatch((0.15, 0.3), 0.7, 0.4, boxstyle="round,pad=0.02",
                                      fc=ZELENA_MPL, ec="none", transform=a_br.transAxes))
        a_br.text(0.5, 0.5, str(i), ha="center", va="center", color="white",
                  fontsize=12, fontweight="bold", transform=a_br.transAxes)
        a_br.axis("off")

        a_cr = fig.add_subplot(desno[r, 1])
        a_cr.imshow(cv2.cvtColor(pripremi_crop(crop), cv2.COLOR_BGR2RGB))
        a_cr.set_xticks([]); a_cr.set_yticks([])
        for sp in a_cr.spines.values():
            sp.set_edgecolor("#cccccc")

        a_tx = fig.add_subplot(desno[r, 2])
        a_tx.add_patch(FancyBboxPatch((0.04, 0.18), 0.92, 0.64, boxstyle="round,pad=0.02",
                                      fc="white", ec="#333333", lw=1.2, transform=a_tx.transAxes))
        a_tx.text(0.5, 0.5, tekst, ha="center", va="center", fontsize=15,
                  fontweight="bold", family="monospace", transform=a_tx.transAxes)
        a_tx.axis("off")
        redovi.append((a_cr, a_tx))

    # naslovi sva tri dijela u istoj liniji
    fig.canvas.draw()
    y = max(ax.get_position().y1, redovi[0][0].get_position().y1) + 0.03
    for osa, naslov in ((ax, "a) Detekcija tablica"), (redovi[0][0], "b) Izrezana tablica"),
                        (redovi[0][1], "c) Očitani tekst")):
        p = osa.get_position()
        fig.text((p.x0 + p.x1) / 2, y, naslov, ha="center", va="bottom", fontsize=12)

    fig.savefig(izlaz, bbox_inches="tight", facecolor="white")
    print(f"Sačuvano: {izlaz}")
    for i, (_, _, tekst, conf) in enumerate(tablice, 1):
        print(f"  {i}. {tekst}   (pouzdanost detekcije {conf:.2f})")


if __name__ == "__main__":
    img, tablice = detektuj_i_ocitaj(SLIKA)
    nacrtaj(img, tablice)