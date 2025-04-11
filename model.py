import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import string
import random



# Define model class
class LanguageLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LanguageLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
    


# Load saved vocabulary and metadata
with open("model_config.json", "r") as f:
    config = json.load(f)

vocab = config["vocab"]
lang_to_idx = config["lang_to_idx"]
idx_to_lang = {int(k): v for k, v in config["idx_to_lang"].items()}
max_length = config["max_length"]



# Load model with correct vocabulary size
vocab_size = len(vocab) + 1  # Use exact vocab size from config
embed_dim = 64
hidden_dim = 128
output_dim = len(lang_to_idx)

model = LanguageLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("language_model.pth"))
model.eval()



# Preprocessing functions
def preprocess(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()

def encode_text(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in preprocess(text)]



# Prediction function
# def predict_language(text):
#     encoded = encode_text(text)
#     padded = encoded + [0] * (max_length - len(encoded))
#     tensor_input = torch.tensor([padded], dtype=torch.long)

#     with torch.no_grad():
#         output = model(tensor_input)
#         predicted_lang_idx = torch.argmax(output).item()

#     return idx_to_lang[predicted_lang_idx]

def predict_language(text):
    # Encoding the text and padding
    encoded = encode_text(text)
    padded = encoded + [0] * (max_length - len(encoded))
    tensor_input = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        # Getting the model output
        output = model(tensor_input)
        
        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()

    # Get predicted language
    predicted_lang_idx = torch.argmax(output).item()

    # Mapping of languages to index (assuming idx_to_lang is a dictionary)
    predicted_lang = idx_to_lang[predicted_lang_idx]
    
    # Get probability distribution of languages
    language_probs = {lang: prob * 100 for lang, prob in zip(idx_to_lang.values(), probabilities)}

    return predicted_lang, language_probs



def test_accuracy(dataset, log_test=True):
    """
    Expected dataset formate:
    ```python
    dataset = [
            ('72,496 बर्ग किलʼमितारसो गोसारना थानाय पाकिस्ताननि सा-ओनसोलफोरा बेनि खोला ओनसोलनि बाहिनी मुहिनि गोनां।', 'brx'), 
            ('A priest and nun walked past us earlier.', 'en'), ('अरे! मुझे विश्वास नहीं हो सकता कि मैंने उसे अपनी सूची से हटा दिया।', 'hi'), 
            ('Je ne sais pas si ceci est vrai.', 'fr'), ('No creo que vaya a necesitar nada más.', 'es'), 
            ('বিএনপির ডাকা অবরোধের চতুর্থ দিনে চট্টগ্রাম নগরীতে একটি বাস কাউন্টারের সামনে দুটি হাতবোমার বিস্ফোরণ ঘটিয়েছে দুর্বৃত্তরা।', 'bn'), 
            ('足久保茶', 'ja'), 
            ('Komm schon.', 'de')
        ]
    ```

    returns:  
        wrong: list[tuple]  
            a list of tuple, the 1st index is text, 2nd index is expected, 3rd index is prediction  
        accuracy: int  
            number of correct prediction
        test_count: int  
            number of text it was tested on (same as dataset length)
    """
    random.shuffle(dataset)
    wrong = []
    test_count = 0
    accuracy = 0
    for language in dataset:
        test_count += 1
        text = language[0]
        expect = language[1]

        prediction = predict_language(text)[0]

        if prediction == expect:
            accuracy += 1
        else:
            x = (text, expect, prediction)
            wrong.append(x)
        if log_test:
            print(accuracy/test_count * 100)
    if log_test:
        print(f"{accuracy} out of {test_count}")
    return wrong, accuracy, test_count



# # Example Predictions
print(predict_language("लामाफोरा आरो दंफांजों महर")[0])  # Expected 'brx'
print(predict_language("Good morning, have a nice day!")[0])  # Expected 'en'
print(predict_language("आप कर सकते हैं?")[0])  # Expected 'hi'
print(predict_language("I am a robot")[0])  # Expected 'en'
print(predict_language("Je vais à l'école chaque jour pour étudier, mais parfois je dois aider mes parents à la maison avant de pouvoir sortir avec mes amis le week-end.")[0]) #expected 'fr'
print(predict_language("Yo nunca había comido ninguna clase de comida thai, así que estaba")[0]) #expected 'es