import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
import numpy as np
import evaluate

# 1. Etiket listesi (Değişiklik yok)
labels = ['O', 'B-sirket', 'I-sirket', 'B-tarih', 'I-tarih', 'B-ad_soyad', 'I-ad_soyad',
          'B-para', 'I-para', 'B-adres', 'I-adres', 'B-telefon', 'I-telefon',
          'B-tc_kimlik', 'I-tc_kimlik']
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# 2. JSONL veriyi oku (DAHA SAĞLAM HALE GETİRİLDİ)
raw_data = []
with open("etiketli_veri.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line: # Sadece boş olmayan satırları işle
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Hatalı satır atlandı: {line[:100]}...")

# 3. Tokenizer yükle (Değişiklik yok)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

# 4. Token + BIO etiket hizalama fonksiyonu (Değişiklik yok, zaten doğruydu)
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

# 5. Dataset oluştur, token-label hizala ve veri setini böl (GELİŞTİRİLDİ)
full_dataset = Dataset.from_list(raw_data)
tokenized_full_dataset = full_dataset.map(tokenize_and_align_labels, remove_columns=full_dataset.column_names)

# Veri setini %80 eğitim, %20 test olarak ayır. Bu, modelin gerçek performansını ölçer.
train_test_split = tokenized_full_dataset.train_test_split(test_size=0.2, seed=42)
final_dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# 6. Modeli yükle (Değişiklik yok)
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# 7. Metrikleri hesaplamak için fonksiyon (YENİ EKLENDİ)
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 8. Eğitim ayarları (GELİŞTİRİLDİ)
# SADECE güncelleme yapmak istemiyorsanız bu alternatifi kullanın
args = TrainingArguments(
    output_dir="./ner_model_checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    # ESKİ YÖNTEM: Değerlendirme ve kaydetme adımlarını sayısal olarak belirtin.
    # Örneğin, veri setinizde 1000 örnek varsa ve batch_size 8 ise,
    # bir epoch yaklaşık 125 adım sürer. Her epoch sonunda değerlendirme yapmak için
    # bu değeri kullanabilirsiniz.
    eval_steps=125, # Değerlendirme adımı
    save_steps=125, # Kaydetme adımı
    # evaluation_strategy ve save_strategy yerine bunlar kullanılır.
    # load_best_model_at_end=True, # Bu da yeni versiyon özelliği, kaldırılmalı.
    logging_dir="./logs",
    logging_steps=50,
    seed=42
)

# 9. Trainer (GELİŞTİRİLDİ)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["test"], # Test setini değerlendirme için kullan
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics # Metrik fonksiyonunu ekle
)

# 10. Eğitimi başlat
trainer.train()

# 11. En iyi modeli kalıcı olarak kaydet
trainer.save_model("./ner_model_final")
tokenizer.save_pretrained("./ner_model_final")

print("\nEğitim tamamlandı. En iyi model './ner_model_final' klasörüne kaydedildi.")
