import gradio as gr
import tempfile
import uuid
import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Model yükleniyor
model_path = "./ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Etiket renkleri (RGB 0–1 aralığında)
label_colors = {
    "sirket": (0.56, 0.93, 0.56),
    "tarih": (0.68, 0.85, 0.90),
    "ad_soyad": (1.0, 0.71, 0.76),
    "para": (0.94, 0.90, 0.55),
    "adres": (0.94, 0.50, 0.50),
    "telefon": (0.87, 0.63, 0.87),
    "tc_kimlik": (1.0, 0.65, 0.0)
}

# HTML çıktı için metin etiketleme
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

# PDF etiketleme
def etiketli_pdf_uret(pdf_file):
    doc = fitz.open(pdf_file.name)

    for page in doc:
        words = page.get_text("words")
        text = " ".join(w[4] for w in words)
        ner_results = ner_pipeline(text)

        matched = []  # eşleşen kelimeleri burada topla

        for ent in ner_results:
            ent_text = ent['word'].replace("##", "").strip()
            label = ent['entity_group']
            color = label_colors.get(label, (1, 1, 0))  # yellow default

            for w in words:
                kelime = w[4].strip()
                if kelime.lower() == ent_text.lower():
                    rect = fitz.Rect(w[0], w[1], w[2], w[3])

                    highlight = page.add_rect_annot(rect)
                    highlight.set_colors(stroke=color, fill=color)
                    highlight.set_opacity(0.4)
                    highlight.update()

                    # Yıldız yerleştirme kısmını güncelle:
                    # Yıldız metni
                    yildizli = "*" * len(kelime)

# Yazı kutusunu biraz genişlet
                    genis_rect = fitz.Rect(rect.x0 - 0.5, rect.y0 - 0.5, rect.x1 + 0.5, rect.y1 + 0.5)

# Opak (beyaz) arka plan kutusu çizerek alttaki yazıyı kapat
                    page.draw_rect(genis_rect, color=(1, 1, 1), fill=(1, 1, 1))

# Yıldızları ekle (daha büyük ve koyu yazı)
                    page.insert_text(
                        point=(genis_rect.x0 + 0.5, genis_rect.y1 - 1),
                        text=yildizli,
                        fontsize=11,
                        fontname="helv",
                        fill=(0, 0, 0)
)

    # Benzersiz geçici dosya adı
    output_path = os.path.join(tempfile.gettempdir(), f"etiketlenmis_{uuid.uuid4().hex}.pdf")
    doc.save(output_path)
    doc.close()
    return output_path

# Gradio Arayüzleri
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
