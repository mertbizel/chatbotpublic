import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pickle


file_path = r"C:\Users\merta\OneDrive\Masaüstü\212503036_doğaldilişleme\atasozleri.csv"

# Türkçe karakter mapping
char_map = str.maketrans({
    'ð': 'ğ', 'þ': 'ş', 'ý': 'ı', 'Ý': 'İ',
    'Þ': 'Ş', 'Ð': 'Ğ', 'æ': 'ç', '¢': 'ö',
    '¤': 'ü', '¸': 'ş', '¥': 'ğ', '£': 'ö',
    'á': 'ç'
})

# tek tip bir veri
df = pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=2, names=['atasozu', 'kullanim_alani'])
df['atasozu'] = df['atasozu'].apply(lambda x: str(x).translate(char_map).lower())
df['kullanim_alani'] = df['kullanim_alani'].apply(lambda x: str(x).translate(char_map).lower())

# encode
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['kullanim_alani'])

#  Veriyi böl
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['atasozu'], df['label'], test_size=0.2, random_state=42
)

# Türkçeye özel tokenizer
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

#  Dataset sınıfı
class AtasozuDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=32)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

#  Dataset 
train_dataset = AtasozuDataset(train_texts, train_labels, tokenizer)
val_dataset = AtasozuDataset(val_texts, val_labels, tokenizer)

#  Model 
model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(label_encoder.classes_)
)

#  TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

#  Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Eğitimi başlat
trainer.train()

#  Tahmin fonksiyonu
def tahmin_et(atasozu):
    inputs = tokenizer(atasozu, return_tensors="pt", truncation=True, padding=True, max_length=32)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]
# Eğitilen modeli kaydet
model.save_pretrained("./model_kayit")
tokenizer.save_pretrained("./model_kayit")

with open("etiket_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


