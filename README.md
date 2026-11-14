# Opis rada

Praktični dio završnog rada **„Sistem za automatski nadzor parkinga i detekciju prekršaja primjenom računarskog vida, IoT uređaja i bespilotne letjelice“**, Fakultet informacijskih tehnologija, Univerzitet „Džemal Bijedić“ u Mostaru.

Sistem na slikama parkinga prepoznaje vozila, parking mjesta i prekršaje pomoću modela **YOLOv8n**, a registarske tablice očitava pomoću **EasyOCR**. Slike šalju ESP32-CAM kamera na ulazu u parking i kamera na dronu. Očitana tablica se prikazuje na OLED displeju, a rezultati u web aplikaciji. Sistem je testiran na maketi parkinga.

## Struktura repozitorija

```
park_smart_ai/
├── main.py                 # FastAPI server (detekcija + OCR)
├── dashboard.py            # web aplikacija (Streamlit)
├── ocr.py                  # očitavanje registarskih tablica (EasyOCR)
├── detect.py               # detekcija na pojedinačnoj slici
├── train.py                # treniranje YOLOv8n modela
├── split_dataset.py        # podjela dataseta na train/val/test
├── ocr_test.py             # testiranje tačnosti OCR-a
├── ocr_primjer.py          # slika postupka očitavanja tablica
├── data.yaml               # konfiguracija dataseta i klase
├── requirements.txt
└── runs/detect/park_smart/weights/best.pt   # istrenirani model
hardware/
├── esp32_cam/              # kamera na ulazu u parking
├── esp32_cam_drone/        # kamera na dronu
└── esp8266/                # ESP8266 sa OLED displejem
```

## Klase modela

| ID | Klasa | Opis |
|---|---|---|
| 0 | `auto` | vozilo |
| 1 | `invalid_oznaka` | oznaka vozila osobe s invaliditetom |
| 2 | `invalidsko_mjesto` | parking mjesto za osobe s invaliditetom |
| 3 | `parking_spot` | obično parking mjesto |
| 4 | `pravilno_parkirano` | vozilo pravilno parkirano unutar mjesta |
| 5 | `prekrsaj_invalidsko` | vozilo na mjestu za osobe s invaliditetom bez oznake |
| 6 | `prekrsaj_nije_parking` | vozilo parkirano na mjestu koje nije parking mjesto |
| 7 | `prekrsaj_van_okvira` | vozilo koje nije unutar okvira parking mjesta |
| 8 | `tablica` | registarska tablica |

## Pokretanje servera i web aplikacije

Potreban je Python 3.10 ili noviji.

```bash
cd park_smart_ai
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Server (FastAPI, port 8000):

```bash
python main.py
```

Web aplikacija (u drugom terminalu):

```bash
streamlit run dashboard.py
```

### Krajnje tačke servera

| Metoda | Putanja | Opis |
|---|---|---|
| POST | `/analyze` | prima sliku, vraća detekcije, očitane tablice i označenu sliku |
| POST | `/esp_feed` | prima sliku sa kamere na dronu (bez detekcije) |
| GET | `/stream/drone` | prenos slike sa drona uživo (MJPEG) |
| GET | `/latest` | posljednja slika i rezultati sa ulazne kamere |
| GET | `/latest_entry` | posljednja očitana tablica (za OLED displej) |

## ESP uređaji

Kod se otvara u Arduino IDE. Potrebne biblioteke: podrška za ESP32 i ESP8266 ploče, `U8g2` i `ArduinoJson`.

Podaci za Wi-Fi nisu u repozitoriju. U folderu svakog sketcha (`esp32_cam`, `esp32_cam_drone`, `esp8266`) treba napraviti fajl `secrets.h`:

```cpp
#pragma once

const char* ssid = "NAZIV_MREZE";
const char* password = "LOZINKA";
```

U `.ino` fajlovima treba upisati i IP adresu računara na kojem radi server (`serverUrl`).

Spajanje OLED displeja (SH1106) na ESP8266: VCC → 3V3, GND → GND, SCK → D5 (GPIO14), SDA → D7 (GPIO13).

## Dataset i rezultati

Dataset je napravljen samostalno na maketi parkinga (304 slike: 212 za treniranje, 61 za validaciju, 31 za testiranje) i zbog veličine nije uključen u repozitorij.

Rezultati na testnom skupu:

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0,960 | 0,925 | 0,951 | 0,628 |

Očitavanje tablica (EasyOCR): od 124 tablice u testnom skupu model je detektovao 105, a od njih je 68 (64,8 %) očitano potpuno tačno.

## Autor

Adelisa Mujarić, IB220199
Mentor: Prof. dr. sc. Elmir Babović
