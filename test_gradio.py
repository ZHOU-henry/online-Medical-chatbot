

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import accelerate
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from torch import nn
import random
import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from timeit import default_timer as timer
from typing import Tuple, Dict
'''
import nltk
import nltk_u
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
'''

#########################################################################################################################
'''
nltk.download('punkt')
nltk.download('stopwords')
stemmer= SnowballStemmer(language= 'english')

# Tokenize text, e.g "I am sick" = ['i', 'am', 'sick']
def tokenize(text):
  return [stemmer.stem(token) for token in word_tokenize(text)]

# Create stopwords to reduce noise
english_stopwords= stopwords.words('english')

# Create a vectosizer to learn all words in order to convert them into numbers
def vectorizer():
    vectorizer= TfidfVectorizer(tokenizer=tokenize, stop_words=english_stopwords)
    return vectorizer
'''
#########################################################################################################################

class GRU_model(nn.Module):
  def __init__(self):
    super().__init__()

    self.rnn= nn.GRU(input_size=384, hidden_size=240,num_layers=2, bias= True).to('cuda') ## nonlinearity= 'relu',
    self.output= nn.Linear(in_features=240, out_features=24).to('cuda')

  def forward(self, x):
    y, hidden= self.rnn(x)
    y = y.to('cuda')
    x= self.output(y).to('cuda')
    return(x)
#embedder = SentenceTransformer("bge-small-en-v1.5", device="cuda")
embedder = SentenceTransformer("C:/OneDrive/git/ARIN_7102/download/bge-small-en-v1.5", device="cuda")
#########################################################################################################################
if torch.cuda.is_available():
    device = "cuda"
    print(f'################################################################# device: {device}#################################################################')
else:
    device = "cpu"
'''
# Import data
df= pd.read_csv('Symptom2Disease_1.csv')

# Preprocess data
df.drop('Unnamed: 0', axis= 1, inplace= True)
df.drop_duplicates(inplace= True)
train_data, test_data= train_test_split(df, test_size=0.2, random_state=1)
'''
#########################################################################################################################
#vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=english_stopwords).fit(train_data.text)
#vectorizer= nltk_u.vectorizer()
#vectorizer.fit(train_data.text)
#from sklearn.feature_extraction.text import TfidfVectorizer
#from spacy.lang.de.stop_words import STOP_WORDS
#vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=list(STOP_WORDS)).fit(train_data.text)
#########################################################################################################################
# Setup class names
class_names= {0: 'Acne', 1: 'Arthritis', 2: 'Bronchial Asthma', 3: 'Cervical spondylosis', 4: 'Chicken pox', 5: 'Common Cold', 6: 'Dengue', 7: 'Dimorphic Hemorrhoids', 8: 'Fungal infection', 9: 'Hypertension',
              10: 'Impetigo', 11: 'Jaundice', 12: 'Malaria', 13: 'Migraine', 14: 'Pneumonia', 15: 'Psoriasis', 16: 'Typhoid', 17: 'Varicose Veins', 18: 'allergy', 19: 'diabetes', 20: 'drug reaction', 
              21: 'gastroesophageal reflux disease', 22: 'peptic ulcer disease', 23: 'urinary tract infection'}
# Disease Advice
disease_advice = {
    'Acne': "Maintain a proper skincare routine, avoid excessive touching of the affected areas, and consider using over-the-counter topical treatments. If severe, consult a dermatologist.",
    'Arthritis': "Stay active with gentle exercises, manage weight, and consider pain-relief strategies like hot/cold therapy. Consult a rheumatologist for tailored guidance.",
    'Bronchial Asthma': "Follow prescribed inhaler and medication regimen, avoid triggers like smoke and allergens, and have an asthma action plan. Regular check-ups with a pulmonologist are important.",
    'Cervical spondylosis': "Maintain good posture, do neck exercises, and use ergonomic support. Physical therapy and pain management techniques might be helpful.",
    'Chicken pox': "Rest, maintain hygiene, and avoid scratching. Consult a doctor for appropriate antiviral treatment.",
    'Common Cold': "Get plenty of rest, stay hydrated, and consider over-the-counter remedies for symptom relief. Seek medical attention if symptoms worsen or last long.",
    'Dengue': "Stay hydrated, rest, and manage fever with acetaminophen. Seek medical care promptly, as dengue can escalate quickly.",
    'Dimorphic Hemorrhoids': "Follow a high-fiber diet, maintain good hygiene, and consider stool softeners. Consult a doctor if symptoms persist.",
    'Fungal infection': "Keep the affected area clean and dry, use antifungal creams, and avoid sharing personal items. Consult a dermatologist if it persists.",
    'Hypertension': "Follow a balanced diet, exercise regularly, reduce salt intake, and take prescribed medications. Regular check-ups with a healthcare provider are important.",
    'Impetigo': "Keep the affected area clean, use prescribed antibiotics, and avoid close contact. Consult a doctor for proper treatment.",
    'Jaundice': "Get plenty of rest, maintain hydration, and follow a doctor's advice for diet and medications. Regular monitoring is important.",
    'Malaria': "Take prescribed antimalarial medications, rest, and manage fever. Seek medical attention for severe cases.",
    'Migraine': "Identify triggers, manage stress, and consider pain-relief medications. Consult a neurologist for personalized management.",
    'Pneumonia': "Follow prescribed antibiotics, rest, stay hydrated, and monitor symptoms. Seek immediate medical attention for severe cases.",
    'Psoriasis': "Moisturize, use prescribed creams, and avoid triggers. Consult a dermatologist for effective management.",
    'Typhoid': "Take prescribed antibiotics, rest, and stay hydrated. Dietary precautions are important. Consult a doctor for proper treatment.",
    'Varicose Veins': "Elevate legs, exercise regularly, and wear compression stockings. Consult a vascular specialist for evaluation and treatment options.",
    'allergy': "Identify triggers, manage exposure, and consider antihistamines. Consult an allergist for comprehensive management.",
    'diabetes': "Follow a balanced diet, exercise, monitor blood sugar levels, and take prescribed medications. Regular visits to an endocrinologist are essential.",
    'drug reaction': "Discontinue the suspected medication, seek medical attention if symptoms are severe, and inform healthcare providers about the reaction.",
    'gastroesophageal reflux disease': "Follow dietary changes, avoid large meals, and consider medications. Consult a doctor for personalized management.",
    'peptic ulcer disease': "Avoid spicy and acidic foods, take prescribed medications, and manage stress. Consult a gastroenterologist for guidance.",
    'urinary tract infection': "Stay hydrated, take prescribed antibiotics, and maintain good hygiene. Consult a doctor for appropriate treatment."
}

howto= """Welcome to the <b>Medical Chatbot</b>, powered by Gradio.
Currently, the chatbot can WELCOME YOU, PREDICT DISEASE based on your symptoms and SUGGEST POSSIBLE SOLUTIONS AND RECOMENDATIONS, and BID YOU FAREWELL.
<b>How to Start:</b> Simply type your messages in the textbox to chat with the Chatbot and press enter!<br><br>
The bot will respond based on the best possible answers to your messages."""


# Create the gradio demo
with gr.Blocks(css = """#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 520px; overflow: auto;}""") as demo:
  gr.HTML('<h1 align="center">Medical Chatbot: ARIN 7102 project')
  with gr.Accordion("Follow these Steps to use the Gradio WebUI", open=True):
      gr.HTML(howto)
  chatbot = gr.Chatbot()
  msg = gr.Textbox()
  clear = gr.ClearButton([msg, chatbot])
  def respond(message, chat_history, base_model = "gru_model", embedder = SentenceTransformer("C:/OneDrive/git/ARIN_7102/download/bge-small-en-v1.5", device="cuda"), device='cuda'): # "meta-llama/Meta-Llama-3-70B"
                                    #base_model =/home/henry/Desktop/ARIN7102/phi-2 # gru_model
        if base_model == "gru_model":
           # Model and transforms preparation
            model= GRU_model()
            # Load state dict
            model.load_state_dict(torch.load(f= 'pretrained_gru_model.pth', map_location= device))
            # Random greetings in list format
            greetings = ["hello!",'hello', 'hii !', 'hi', "hi there!",  "hi there!", "heyy", 'good morning', 'good afternoon', 'good evening', "hey", "how are you", "how are you?", "how is it going", "how is it going?", "what's up?",
                        "how are you?", "hey, how are you?", "what is popping", "good to see you!", "howdy!", "hi, nice to meet you.", "hiya!", "hi", "hi, what's new?", "hey, how's your day?", "hi, how have you been?", "greetings"]
            # Random Greetings responses
            greetings_responses = ["Thank you for using our medical chatbot. Please provide the symptoms you're experiencing, and I'll do my best to predict the possible disease.",
                "Hello! I'm here to help you with medical predictions based on your symptoms. Please describe your symptoms in as much detail as possible.",
                "Greetings! I am a specialized medical chatbot trained to predict potential diseases based on the symptoms you provide. Kindly list your symptoms explicitly.",
                "Welcome to the medical chatbot. To assist you accurately, please share your symptoms in explicit detail.",
                "Hi there! I'm a medical chatbot specialized in analyzing symptoms to suggest possible diseases. Please provide your symptoms explicitly.",
                "Hey! I'm your medical chatbot. Describe your symptoms with as much detail as you can, and I'll generate potential disease predictions.",
                "How can I assist you today? I'm a medical chatbot trained to predict diseases based on symptoms. Please be explicit while describing your symptoms.",
                "Hello! I'm a medical chatbot capable of predicting diseases based on the symptoms you provide. Your explicit symptom description will help me assist you better.",
                "Greetings! I'm here to help with medical predictions. Describe your symptoms explicitly, and I'll offer insights into potential diseases.",
                "Hi, I'm the medical chatbot. I've been trained to predict diseases from symptoms. The more explicit you are about your symptoms, the better I can assist you.",
                "Hi, I specialize in medical predictions based on symptoms. Kindly provide detailed symptoms for accurate disease predictions.",
                "Hello! I'm a medical chatbot with expertise in predicting diseases from symptoms. Please describe your symptoms explicitly to receive accurate insights."]
            # Random goodbyes
            goodbyes = ["farewell!",'bye', 'goodbye','good-bye', 'good bye', 'bye', 'thank you', 'later', "take care!",
                "see you later!", 'see you', 'see ya', 'see-you', 'thanks', 'thank', 'bye bye', 'byebye'
                "catch you on the flip side!", "adios!",
                "goodbye for now!", "till we meet again!",
                "so long!", "hasta la vista!",
                "bye-bye!", "keep in touch!",
                "toodles!", "ciao!",
                "later, gator!", "stay safe and goodbye!",
                "peace out!", "until next time!", "off I go!"]
            # Random Goodbyes responses
            goodbyes_replies = ["Take care of yourself! If you have more questions, don't hesitate to reach out.", "Stay well! Remember, I'm here if you need further medical advice.",
                "Goodbye for now! Don't hesitate to return if you need more information in the future.", "Wishing you good health ahead! Feel free to come back if you have more concerns.",
                "Farewell! If you have more symptoms or questions, don't hesitate to consult again.", "Take care and stay informed about your health. Feel free to chat anytime.",
                "Bye for now! Remember, your well-being is a priority. Don't hesitate to ask if needed.", "Have a great day ahead! If you need medical guidance later on, I'll be here.",
                "Stay well and take it easy! Reach out if you need more medical insights.", "Until next time! Prioritize your health and reach out if you need assistance.",
                "Goodbye! Your health matters. Feel free to return if you have more health-related queries.", "Stay healthy and stay curious about your health! If you need more info, just ask.",
                "Wishing you wellness on your journey! If you have more questions, I'm here to help.", "Stay well and stay proactive about your health! If you have more queries, feel free to ask.",
                "Take care and remember, your health is important. Don't hesitate to reach out if needed.", "Goodbye for now! Stay informed and feel free to consult if you require medical advice.",
                "Farewell! Remember, I'm here whenever you need reliable medical information.", "Bye for now! Stay vigilant about your health and don't hesitate to return if necessary.",
                "Take care and keep your well-being a priority! Reach out if you have more health questions.", "Wishing you good health ahead! Don't hesitate to chat if you need medical insights.",
                "Goodbye! Stay well and remember, I'm here to assist you with medical queries."]

            if message.lower() in greetings:
                bot_message= random.choice(greetings_responses)
            elif message.lower() in goodbyes:
                bot_message= random.choice(goodbyes_replies)
            else: 
                #transform_text= vectorizer.transform([message])

                embedder = SentenceTransformer("bge-small-en-v1.5", device="cuda")
                sentence_embeddings = embedder.encode(message)
                sentence_embeddings = torch.from_numpy(sentence_embeddings).float().to(device).unsqueeze(dim=0)

                #transform_text= torch.tensor(transform_text.toarray()).to(torch.float32)
                model.eval()
                
                with torch.inference_mode():
                    y_logits=model(sentence_embeddings.to(device))
                    pred_prob= torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
                test_pred= class_names[pred_prob.item()] 
                bot_message = f' Based on your symptoms, I believe you are having {test_pred} and I would advice you {disease_advice[test_pred]}'


        else:
                
            # define the model and tokenizer.
            # model = PhiForCausalLM.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            # feel free to change the prompt to your liking.
            #prompt = f"Patient: coercive spondylitis, pain in the lumbosacral area when turning over during sleep at night, no pain in any other part of the body.  
            #/n Doctor: It shouldn't be a problem, but it's better to upload the images. /n Patient: {message} /n Doctor:"
            output_termination = "\nOutput:"
            prompt = f"Instruct: Hi, i am patient, {message} what is wrong with my body? What drugs should i take, and what is the side-effect of this drug? What should i do?{output_termination}"
            print(prompt)
            # apply the tokenizer.
            tokens = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            #tokens = tokens.to(device)
            #eos_token_id = tokenizer.eos_token_id
            # use the model to generate new tokens.
            generated_output = model.generate(**tokens, use_cache=True, max_new_tokens=2000, eos_token_id=50256, pad_token_id=50256)

            # Find the position of "Output:" and extract the text after it
            generated_text = tokenizer.batch_decode(generated_output)[0]
            # Split the text at "Output:" and take the second part
            split_text = generated_text.split("Output:", 1)
            bot_message = split_text[1].strip() if len(split_text) > 1 else ""
            bot_message = bot_message.replace("<|endoftext|>", "").strip()
            #return bot_message
            #chat_history.append((message, bot_message))
            #time.sleep(2)
            #return "", chat_history
        chat_history.append((message, bot_message))
        time.sleep(2)
        #return bot_message
        return "", chat_history


  msg.submit(respond, [msg, chatbot], [msg, chatbot])
# Launch the demo
demo.launch(share=True)

  #msg.submit(respond, [msg, chatbot], [msg, chatbot])
# Launch the demo
#demo.launch()


#gr.ChatInterface(respond).launch()


#a = respond('hi', list())
#a = respond("Hi, good morning")
#a = respond("My skin has been peeling, especially on my knees, elbows, and scalp. This peeling is often accompanied by a burning or stinging sensation.")
#a = respond("I have blurry vision, and it seems to be getting worse. I'm continuously fatigued and worn out. I also occasionally have acute lightheadedness and vertigo, can you give me some advice?")
#print(a)
#gr.ChatInterface(respond).launch()
#demo = gr.ChatInterface(fn=random_response, examples=[{"text": "Hello", "files": []}], title="Echo Bot", multimodal=True)
#demo.launch()