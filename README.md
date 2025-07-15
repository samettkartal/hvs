#hvs
#vericekmeplus.py

import os
import csv
import time
import random
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

sirketler = [
    "trendyol", "hepsiburada", "getir", "n11", "ciceksepeti",
    "yurtici-kargo", "aras-kargo", "mng-kargo", "ptt-kargo",
    "thy", "pegasus", "anadolujet",
    "migros", "a101", "bim-market",
    "ziraat-bankasi", "garanti-bbva", "is-bankasi",
    "sahibinden-com", "arcelik", "vestel", "spotify"
]

csv_dosya = "toplu_sikayetler_playwright.csv"

if not os.path.exists(csv_dosya):
    print(f"'{csv_dosya}' dosyası bulunamadı, başlıklar eklenerek oluşturuluyor...")
    with open(csv_dosya, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Başlık", "Şikayet"])

def get_complaint_links(page, sirket_slug):
    try:
        page.wait_for_selector("a.complaint-layer", timeout=7000)
        links = page.locator("a.complaint-layer").all()
        hrefs = [link.get_attribute("href") for link in links]
        full_links = [
            f"https://www.sikayetvar.com{href}"
            for href in hrefs
            if href and href.startswith(f"/{sirket_slug}/")
        ]
        return full_links
    except PlaywrightTimeoutError:
        print("  - Bu sayfada şikayet linki bulunamadı veya sayfa yüklenemedi.")
        return []

def get_title_and_first_paragraph(page):
    try:
        page.wait_for_selector("h1", timeout=10000)
        title = page.locator("h1").inner_text().strip()
    except PlaywrightTimeoutError:
        title = "❌ Başlık bulunamadı"

    try:
        paragraph = page.locator("xpath=//h1/following::p[1]").inner_text().strip()
    except Exception:
        paragraph = "❌ Paragraf alınamadı"

    return title, paragraph

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    )
    
    page = context.new_page()

    for sirket_slug in sirketler:
        print("\n" + "="*60)
        print(f"🏢 İŞLEM BAŞLADI: '{sirket_slug.upper()}' ŞİRKETİ İÇİN VERİ ÇEKİLİYOR")
        print("="*60)
        
        for sayfa in range(1, 11):
            base_url = f"https://www.sikayetvar.com/{sirket_slug}?page={sayfa}"
            
            print(f"\n📄 Sayfa {sayfa} ({sirket_slug}) taranıyor...")
            try:
                page.goto(base_url, timeout=30000)
                
                time.sleep(random.uniform(2.5, 4.5))

                links = get_complaint_links(page, sirket_slug)
                print(f"🔗 {len(links)} geçerli şikayet bulundu.")

                for link in links:
                    try:
                        page.goto(link, timeout=30000)
                        time.sleep(random.uniform(1.5, 3.0))
                        
                        title, para = get_title_and_first_paragraph(page)
                        
                        if "❌" not in para:
                            print(f"  ✅ Başarıyla çekildi: {title[:70]}...")
                        else:
                            print(f"  ⚠️ Paragraf çekilemedi: {title[:70]}...")

                        with open(csv_dosya, mode="a", encoding="utf-8", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([title, para])
                    except Exception as e:
                        print(f"  ❌ HATA (Link işlenirken: {link}): {str(e)}")
            except PlaywrightTimeoutError:
                 print(f"  ❌ KRİTİK HATA: Sayfa yüklenirken zaman aşımına uğradı. Muhtemelen Cloudflare engeline takıldı.")
                 print("  ➡️ Bir sonraki şirkete geçiliyor...")
                 break
            except Exception as e:
                print(f"  ❌ KRİTİK HATA (Sayfa {sayfa} açılamadı): {str(e)}")
                break

    print("\n\n🎉 Tüm işlemler bitti. Tarayıcı kapatılıyor.")
    browser.close()
#regex-spacy.py

import pandas as pd
import re
import json
from transformers import pipeline

df = pd.read_csv("turkcell_sikayetleri.csv")
metinler = df["Şikayet"].dropna().tolist()

ner_pipeline = pipeline("ner", model="savasy/bert-base-turkish-ner-cased", grouped_entities=True)

yerler = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Adana", "Antalya", "Konya", "Mersin", "Kayseri", "Diyarbakır",
    "Şanlıurfa", "Gaziantep", "Trabzon", "Eskişehir", "Samsun", "Malatya", "Manisa", "Van", "Sakarya", "Denizli",
    "Kadıköy", "Üsküdar", "Beşiktaş", "Çankaya", "Keçiören", "Yenimahalle", "Osmangazi", "Nilüfer", "Karatay", "Selçuklu"
]
yer_pattern = r"\b(" + "|".join(map(re.escape, yerler)) + r")(?:'?[dltDLAEİIÜ][aeıiuü]?[knm]?[yi]?|’?[dltDLAEİIÜ][aeıiuü]?[knm]?[yi]?)?\b"

regexler = {
    "telefon": r"(?:\+?90|0)?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}|\*{7,9}\d{2}",
    "tc_kimlik": r"\b\d{11}\b|\*{7,9}\d{2}",
    "tarih": r"\b\d{1,2}[\.\/\-\s]?(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık|ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)[\.\/\-\s]?\d{0,4}\b|\b\d{1,2}[\.\/\-]\d{1,2}([\.\/\-]\d{2,4})?\b|\b\d{1,2}\s?(Ocak|Şubat|Mart|...|aralık)\s?\d{0,4}\b",
    "para": r"\b\d{1,4}([.,]\d{1,2})?\s?(TL|₺|tl|lira)\b",
    "ad_soyad": r"\b[A-ZÇŞĞÜİÖ][a-zçşğüöı]{2,}\s[A-ZÇŞĞÜİÖ][a-zçşğüöı]{2,}\b",
    "adres": yer_pattern
}

etiketli_veriler = []

for metin in metinler:
    entities = []

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

with open("etiketli_veri.jsonl", "w", encoding="utf-8") as f:
    for item in etiketli_veriler:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ JSONL dosyası dict formatıyla başarıyla oluşturuldu.")

#model.py

import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)

labels = ['O', 'B-sirket', 'I-sirket', 'B-tarih', 'I-tarih', 'B-ad_soyad', 'I-ad_soyad',
          'B-para', 'I-para', 'B-adres', 'I-adres', 'B-telefon', 'I-telefon',
          'B-tc_kimlik', 'I-tc_kimlik']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

with open("etiketli_veri.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        return_offsets_mapping=True
    )
    
    labels_out = ['O'] * len(tokenized["input_ids"])
    
    for entity in example["entities"]:
        start, end, label = int(entity["start"]), int(entity["end"]), entity["label"]
        
        is_first_token = True
        
        for i, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
            if token_start is None or token_end is None:
                continue
            
            if max(token_start, start) < min(token_end, end):
                if is_first_token:
                    labels_out[i] = 'B-' + label
                    is_first_token = False
                else:
                    labels_out[i] = 'I-' + label

    tokenized["labels"] = [label2id.get(l, label2id['O']) for l in labels_out]
    tokenized.pop("offset_mapping")
    return tokenized

dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(tokenize_and_align_labels)

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="./ner_model",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

trainer.train()

trainer.save_model("./ner_model")
tokenizer.save_pretrained("./ner_model")

#app.py

import gradio as gr
import tempfile
import uuid
import os
import fitz
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_path = "./ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

label_colors = {
    "sirket": (0.56, 0.93, 0.56),
    "tarih": (0.68, 0.85, 0.90),
    "ad_soyad": (1.0, 0.71, 0.76),
    "para": (0.94, 0.90, 0.55),
    "adres": (0.94, 0.50, 0.50),
    "telefon": (0.87, 0.63, 0.87),
    "tc_kimlik": (1.0, 0.65, 0.0)
}

def etiketle(text):
    entities = ner_pipeline(text)
    entities = sorted(entities, key=lambda x: x['start'])
    highlighted = ""
    last_idx = 0
    for ent in entities:
        start, end, label = ent['start'], ent['end'], ent['entity_group']
        color = label_colors.get(label, "lightgray")
        highlighted += text[last_idx:start]
        highlighted += f"<span style='background-color:{color}; padding:2px; border-radius:4px;' title='{label}'>"
        highlighted += "*" * (end - start)
        highlighted += "</span>"
        last_idx = end
    highlighted += text[last_idx:]
    return highlighted

def etiketli_pdf_uret(pdf_file):
    doc = fitz.open(pdf_file.name)

    for page in doc:
        words = page.get_text("words")
        text = " ".join(w[4] for w in words)
        ner_results = ner_pipeline(text)

        matched = []

        for ent in ner_results:
            ent_text = ent['word'].replace("##", "").strip()
            label = ent['entity_group']
            color = label_colors.get(label, (1, 1, 0))

            for w in words:
                kelime = w[4].strip()
                if kelime.lower() == ent_text.lower():
                    rect = fitz.Rect(w[0], w[1], w[2], w[3])

                    highlight = page.add_rect_annot(rect)
                    highlight.set_colors(stroke=color, fill=color)
                    highlight.set_opacity(0.4)
                    highlight.update()
                    
                    yildizli = "*" * len(kelime)

                    genis_rect = fitz.Rect(rect.x0 - 0.5, rect.y0 - 0.5, rect.x1 + 0.5, rect.y1 + 0.5)

                    page.draw_rect(genis_rect, color=(1, 1, 1), fill=(1, 1, 1))

                    page.insert_text(
                        point=(genis_rect.x0 + 0.5, genis_rect.y1 - 1),
                        text=yildizli,
                        fontsize=11,
                        fontname="helv",
                        fill=(0, 0, 0)
                    )

    output_path = os.path.join(tempfile.gettempdir(), f"etiketlenmis_{uuid.uuid4().hex}.pdf")
    doc.save(output_path)
    doc.close()
    return output_path

demo = gr.Interface(
    fn=etiketle,
    inputs=gr.Textbox(lines=6, placeholder="Şikayet metnini buraya yaz...", label="Metin Girişi"),
    outputs=gr.HTML(label="Etiketli Metin"),
    title="🔎 Türkçe Şikayet NER Etiketleyici",
    description="Metin içindeki özel bilgileri otomatik renklendirir ve maskeleyerek gösterir."
)

pdf_demo = gr.Interface(
    fn=etiketli_pdf_uret,
    inputs=gr.File(label="PDF Dosyası Yükle (.pdf)"),
    outputs=gr.File(label="Etiketlenmiş PDF Dosyası"),
    title="📄 PDF Üzerinde NER Etiketleme",
    description="Yüklediğiniz PDF içinde özel bilgiler renklendirilir ve yıldızlarla maskelenir."
)

app = gr.TabbedInterface([demo, pdf_demo], ["Metin Etiketleme", "PDF Etiketleme"])
app.launch()
