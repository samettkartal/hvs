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

# HTML çıktı için metin etiketleme (Bu fonksiyonda değişiklik yok)
def etiketle(text):
    entities = ner_pipeline(text)
    entities = sorted(entities, key=lambda x: x['start'])
    highlighted = ""
    last_idx = 0
    for ent in entities:
        start, end, label = ent['start'], ent['end'], ent['entity_group']
        # Modelin tanımadığı etiketler için varsayılan bir renk ata
        color_rgb = label_colors.get(label)
        if color_rgb:
             # Gradio HTML'i RGB değerleri yerine hex kodlarını veya renk isimlerini tercih eder.
             # Ancak background-color için bu şekilde de çalışabilir. Daha güvenli olması için
             # renkleri 'rgb(255,0,0)' formatına çevirmek daha iyi olabilir.
             # Basitlik için şimdilik bu şekilde bırakıyoruz.
            color_str = f"rgba({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)}, 0.5)"
        else:
            color_str = "lightgray" # Tanınmayan etiketler için

        highlighted += text[last_idx:start]
        highlighted += f"<span style='background-color:{color_str}; padding:2px 4px; border-radius:4px; font-weight: bold;' title='{label}'>"
        highlighted += "*" * len(text[start:end])
        highlighted += "</span>"
        last_idx = end
    highlighted += text[last_idx:]
    return highlighted

# ✨ PDF ETİKETLEME FONKSİYONU GÜNCELLENDİ
def etiketli_pdf_uret(pdf_file, secilen_etiketler):
    # Eğer hiç etiket seçilmemişse, orijinal dosyayı döndür ve uyarı ver
    if not secilen_etiketler:
        gr.Warning("Hiçbir etiket türü seçilmedi! Orijinal PDF döndürülüyor.")
        return pdf_file.name

    doc = fitz.open(pdf_file.name)

    for page in doc:
        words = page.get_text("words")
        # Eğer sayfada kelime yoksa, bir sonraki sayfaya geç
        if not words:
            continue
            
        text = " ".join(w[4] for w in words)
        ner_results = ner_pipeline(text)

        for ent in ner_results:
            label = ent['entity_group']
            
            # ✨ YENİ EKlenen MANTIK: Eğer modelin bulduğu etiket, kullanıcının seçtikleri arasında değilse, bu adımı atla
            if label not in secilen_etiketler:
                continue

            ent_text = ent['word'].replace("##", "").strip()
            color = label_colors.get(label, (1, 1, 0))  # yellow default

            # Kelime eşleştirme mantığı, modelin bulduğu metinle sayfadaki kelimeleri karşılaştırır
            # Bu kısım bazen zorlayıcı olabilir, çünkü model birleşik kelimeler bulabilir.
            # Şimdilik basit bir eşleştirme ile devam ediyoruz.
            for w in words:
                kelime = w[4].strip()
                if kelime.lower() in ent_text.lower() or ent_text.lower() in kelime.lower():
                    rect = fitz.Rect(w[0], w[1], w[2], w[3])

                    # Orijinal metni beyaz bir kutu ile kapat
                    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

                    # Arkasına renkli bir vurgu ekle (daha estetik)
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=color)
                    highlight.update(opacity=0.4)
                    
                    # Üzerine yıldızları ekle
                    yildizli = "*" * len(kelime)
                    page.insert_text(
                        point=(rect.x0, rect.y1-1), # Metnin başladığı yere yıldızları koy
                        text=yildizli,
                        fontsize=10, # Yazıtipi boyutunu orijinal metne yakın ayarla
                        fontname="helv",
                        color=(0,0,0) # Siyah renk
                    )

    output_path = os.path.join(tempfile.gettempdir(), f"etiketlenmis_{uuid.uuid4().hex}.pdf")
    doc.save(output_path, garbage=4, deflate=True, clean=True)
    doc.close()
    return output_path

# Metin etiketleme arayüzü (değişiklik yok)
demo = gr.Interface(
    fn=etiketle,
    inputs=gr.Textbox(lines=8, placeholder="Şikayet metnini buraya yazın veya yapıştırın...", label="Metin Girişi"),
    outputs=gr.HTML(label="Etiketlenmiş Metin"),
    title="🔎 Metin İçindeki Özel Bilgileri Etiketleme",
    description="Metin içindeki Şirket, Tarih, Kişi, Para, Adres, Telefon ve T.C. Kimlik gibi özel bilgileri otomatik olarak bulur, renklendirir ve yıldızlarla maskeler."
)

# ✨ PDF ETİKETLEME ARAYÜZÜ GÜNCELLENDİ
# Checkbox'lar için seçenekleri ve varsayılanları tanımla
etiket_secenekleri = list(label_colors.keys())
varsayilan_secim = list(label_colors.keys()) # Hepsi varsayılan olarak seçili

pdf_demo = gr.Interface(
    fn=etiketli_pdf_uret,
    inputs=[
        gr.File(label="PDF Dosyası Yükle (.pdf)"),
        gr.CheckboxGroup(
            choices=etiket_secenekleri,
            value=varsayilan_secim,
            label="Maskelenecek Bilgi Türleri",
            info="Maskelenmesini istemediğiniz bilgi türünün işaretini kaldırın."
        )
    ],
    outputs=gr.File(label="İşlenmiş PDF Dosyası"),
    title="📄 PDF Üzerindeki Özel Bilgileri Etiketleme ve Maskeleme",
    description="Yüklediğiniz PDF dosyası içindeki özel bilgileri bulur, seçiminize göre renklendirir ve yıldızlarla maskeler."
)

# Sekmeli arayüzü oluştur
app = gr.TabbedInterface(
    [demo, pdf_demo], 
    ["Metin Etiketleme", "PDF Etiketleme"],
    title="Gelişmiş Bilgi Gizleme ve Etiketleme Aracı (NER)"
)
app.launch()
