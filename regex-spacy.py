import pandas as pd
import re
import json
from transformers import pipeline

# 📥 CSV'den veriyi oku
df = pd.read_csv("turkcell_sikayetleri.csv")
metinler = df["Şikayet"].dropna().tolist()

# 🤖 Transformers tabanlı Türkçe NER modeli
ner_pipeline = pipeline("ner", model="savasy/bert-base-turkish-ner-cased", grouped_entities=True)

# 📍 Türkiye'deki yaygın şehir ve ilçe isimleri
yerler = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Adana", "Antalya", "Konya", "Mersin", "Kayseri", "Diyarbakır",
    "Şanlıurfa", "Gaziantep", "Trabzon", "Eskişehir", "Samsun", "Malatya", "Manisa", "Van", "Sakarya", "Denizli",
    "Kadıköy", "Üsküdar", "Beşiktaş", "Çankaya", "Keçiören", "Yenimahalle", "Osmangazi", "Nilüfer", "Karatay", "Selçuklu"
]
yer_pattern = r"\b(" + "|".join(map(re.escape, yerler)) + r")(?:'?[dltDLAEİIÜ][aeıiuü]?[knm]?[yi]?|’?[dltDLAEİIÜ][aeıiuü]?[knm]?[yi]?)?\b"

# 🔍 Gelişmiş Regex kalıpları
regexler = {
    "telefon": r"(?:\+?90|0)?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}|\*{7,9}\d{2}",
    "tc_kimlik": r"\b\d{11}\b|\*{7,9}\d{2}",
    "tarih": r"\b\d{1,2}[\.\/\-\s]?(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık|ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)[\.\/\-\s]?\d{0,4}\b|\b\d{1,2}[\.\/\-]\d{1,2}([\.\/\-]\d{2,4})?\b|\b\d{1,2}\s?(Ocak|Şubat|Mart|...|aralık)\s?\d{0,4}\b",
    "para": r"\b\d{1,4}([.,]\d{1,2})?\s?(TL|₺|tl|lira)\b",
    "ad_soyad": r"\b[A-ZÇŞĞÜİÖ][a-zçşğüöı]{2,}\s[A-ZÇŞĞÜİÖ][a-zçşğüöı]{2,}\b",
    "adres": yer_pattern
}

etiketli_veriler = []

# 🔁 Her metni işle
for metin in metinler:
    entities = []

    # 🧠 Transformers NER sonucu
    ner_results = ner_pipeline(metin)
    for ent in ner_results:
        label_map = {
            "PER": "ad_soyad",
            "LOC": "adres",
            "ORG": "sirket",
            "DATE": "tarih",
            "MONEY": "para"
        }
        entity_type = ent["entity_group"]
        if entity_type in label_map:
            entities.append({
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "label": label_map[entity_type]
            })

    # 🔎 Regex ile ek etiketleri bul
    for label, pattern in regexler.items():
        for match in re.finditer(pattern, metin):
            entities.append({
                "start": match.start(),
                "end": match.end(),
                "label": label
            })

    etiketli_veriler.append({
        "text": metin,
        "entities": entities
    })

# 💾 JSONL formatında kaydet
with open("etiketli_veri.jsonl", "w", encoding="utf-8") as f:
    for item in etiketli_veriler:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ JSONL dosyası dict formatıyla başarıyla oluşturuldu.")
