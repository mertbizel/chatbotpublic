import tkinter as tk
from tkinter import messagebox
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

#  Model, Tokenizer ve LabelEncoder'ı dosyadan import
try:
    model = BertForSequenceClassification.from_pretrained("./model_kayit")
    tokenizer = BertTokenizer.from_pretrained("./model_kayit")
    with open("etiket_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Model veya tokenizer yüklenemedi: {e}")

#  Tahmin fonksiyonu
def tahmin_et_gui():
    atasozu = entry.get()
    if not atasozu.strip():
        messagebox.showwarning("Uyarı", "Lütfen bir atasözü girin.")
        return

    try:
        # bert ile girdiyi işleme
        inputs = tokenizer(atasozu, return_tensors="pt", truncation=True, padding=True, max_length=32)
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        sonuc = label_encoder.inverse_transform([pred])[0]
        result_label.config(text=f"📌 Tahmini Kullanım Alanı: {sonuc}")
    except Exception as e:
        messagebox.showerror("Hata", str(e))

# ✅ Tkinter form arayüzü
pencere = tk.Tk()
pencere.title("Atasözü Sınıflandırıcı Chatbot 🤖")
pencere.geometry("450x220")
pencere.configure(bg="#f0f0f0")

# Başlık
tk.Label(pencere, text="Atasözü giriniz:", bg="#f0f0f0", font=("Arial", 11)).pack(pady=10)

# Giriş kutusu
entry = tk.Entry(pencere, width=50, font=("Arial", 11))
entry.pack(pady=5)

# Tahmin butonu
tk.Button(pencere, text="Tahmin Et", command=tahmin_et_gui, font=("Arial", 10)).pack(pady=10)

# Sonuç etiketi
result_label = tk.Label(pencere, text="", font=("Arial", 12), fg="#333", bg="#f0f0f0")
result_label.pack(pady=10)

# Formu çalıştır
pencere.mainloop()
