import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)

# 1. Etiket listesi
labels = ['O', 'B-sirket', 'I-sirket', 'B-tarih', 'I-tarih', 'B-ad_soyad', 'I-ad_soyad',
          'B-para', 'I-para', 'B-adres', 'I-adres', 'B-telefon', 'I-telefon',
          'B-tc_kimlik', 'I-tc_kimlik']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# 2. JSONL veriyi oku
with open("etiketli_veri.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

# 3. Tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

# 4. Token + BIO etiket hizalama
# KODUNUZDAKİ BU FONKSİYONU AŞAĞIDAKİ İLE DEĞİŞTİRİN

def tokenize_and_align_labels(example):
    # Metni tokenizer ile parçalara ayır
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        return_offsets_mapping=True # Karakter indekslerini almak için
    )
    
    # Başlangıçta tüm etiketleri 'O' (Outside) olarak ayarla
    labels_out = ['O'] * len(tokenized["input_ids"])
    
    # JSON'daki her bir varlık (entity) için döngü başlat
    for entity in example["entities"]:
        start, end, label = int(entity["start"]), int(entity["end"]), entity["label"]
        
        # Bu varlığın ilk token'ını bulduk mu diye kontrol etmek için bir bayrak
        is_first_token = True
        
        # Tokenizer tarafından üretilen her bir token'ın başlangıç ve bitiş indeksleri üzerinde döngü başlat
        for i, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
            # [CLS], [SEP] gibi özel token'ları atla
            if token_start is None or token_end is None:
                continue
            
            # Eğer token'ın kapsadığı alan, bizim varlığımızın alanı ile kesişiyorsa
            if max(token_start, start) < min(token_end, end):
            
                # Eğer bu, varlığın bulduğumuz ilk token'ı ise 'B-' (Beginning) etiketini ata
                if is_first_token:
                    labels_out[i] = 'B-' + label
                    is_first_token = False # Bayrağı indir, sonraki token'lar 'I-' olacak
                # Eğer ilk token değilse, 'I-' (Inside) etiketini ata
                else:
                    labels_out[i] = 'I-' + label

    # Son olarak, string etiketleri ID'lere dönüştür
    tokenized["labels"] = [label2id.get(l, label2id['O']) for l in labels_out]
    # Artık offset_mapping'e ihtiyacımız yok
    tokenized.pop("offset_mapping")
    return tokenized

# 5. Dataset oluştur ve token-label hizala
dataset = Dataset.from_list(raw_data)
tokenized_dataset = dataset.map(tokenize_and_align_labels)

# 6. Modeli yükle
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# 7. Eğitim ayarları
args = TrainingArguments(
    output_dir="./ner_model",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    save_strategy="no",  # ✔️ Checkpoint kaydı yok
    logging_dir="./logs",
    logging_steps=10,
    seed=42
)


# 8. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# 9. Eğitimi başlat
trainer.train()

# 10. Eğitilen modeli kaydet
trainer.save_model("./ner_model")
tokenizer.save_pretrained("./ner_model")
