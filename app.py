import locale
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
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
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.de.stop_words import STOP_WORDS
vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
'''
import nltk
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

    self.rnn= nn.GRU(input_size=1477, hidden_size=240,num_layers=1, bias= True).to(device) ## nonlinearity= 'relu',
    self.output= nn.Linear(in_features=240, out_features=24).to(device)

  def forward(self, x):
    y, hidden= self.rnn(x)
    y = y.to(device)
    x= self.output(y).to(device)
    return(x)

class RNN_model(nn.Module):
  def __init__(self):
    super().__init__()

    self.rnn= nn.RNN(input_size=1477, hidden_size=240,num_layers=1, nonlinearity= 'relu', bias= True).to(device)
    self.output= nn.Linear(in_features=240, out_features=24).to(device)

  def forward(self, x):
    y, hidden= self.rnn(x)
    y = y.to(device)
    x= self.output(y).to(device)
    return(x)

#embedder = SentenceTransformer("bge-small-en-v1.5", device=device)
#embedder = SentenceTransformer("bge-small-en-v1.5", device=device)

df= pd.read_csv('Symptom2Disease_1.csv')

target=['Psoriasis', 'Varicose Veins', 'Typhoid', 'Chicken pox',
       'Impetigo', 'Dengue', 'Fungal infection', 'Common Cold',
       'Pneumonia', 'Dimorphic Hemorrhoids', 'Arthritis', 'Acne',
       'Bronchial Asthma', 'Hypertension', 'Migraine',
       'Cervical spondylosis', 'Jaundice', 'Malaria',
       'urinary tract infection', 'allergy',
       'gastroesophageal reflux disease', 'drug reaction',
       'peptic ulcer disease', 'diabetes']
target_dict= {i:j for i,j in enumerate(sorted(target))}
df['label']= df['label'].replace({j:i for i,j in enumerate(sorted(target))})
df.drop('Unnamed: 0', axis= 1, inplace= True)
df.duplicated().sum()
df[df.duplicated]
df.drop_duplicates(inplace= True)
df['label'].value_counts()
train_data, test_data= train_test_split(df, test_size=0.15, random_state=42 )
train_data['label'].value_counts().sort_index()
test_data['label'].value_counts().sort_index()
vectorizer.fit(train_data.text)
vectorizer.get_feature_names_out()[: 100]
vectorizer= vectorizer
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
    'Acne': "Acne, commonly referred to as \"acne vulgaris\" or \"pimples,\" is a prevalent skin condition that primarily affects teenagers and young adults but can occur at any age. Acne typically appears on the face, back, and chest, particularly in areas with active sebaceous glands. It manifests as skin bumps, comedones, blackheads, whiteheads, and sometimes pustules. You can use topical medications such as benzoyl peroxide, salicylic acid, topical antibiotics, and retinoids (like tretinoin), or oral medications such as antibiotics (like tetracycline and erythromycin), oral contraceptives (which may help some women), and oral retinoids (such as isotretinoin) for improvement. When using these medications on your own, it is essential to follow the prescription and instructions of a doctor. At the same time, in your daily life, it is important to maintain a proper skincare routine, avoid excessive touching of the affected areas, minimize the intake of high-sugar and high-fat foods, and consume more fruits, vegetables, and whole grains.",
    'Arthritis': "Arthritis is a term that refers to a variety of joint diseases, involving pain and inflammation in the joints. Common types include osteoarthritis (OA) and rheumatoid arthritis (RA). Osteoarthritis is caused by the wear and tear of joint cartilage, while rheumatoid arthritis is an autoimmune disease. You can use nonsteroidal anti-inflammatory drugs (NSAIDs) and analgesics such as ibuprofen or naproxen, acetaminophen (paracetamol), etc., to reduce inflammation and relieve pain. Corticosteroid medications like prednisone can be used for rapid inflammation relief but have potential side effects with long-term use. For rheumatoid arthritis, disease-modifying antirheumatic drugs (DMARDs) like methotrexate can be used to slow disease progression. When using these medications, it is crucial to follow the prescription and instructions of a doctor. In daily life, it is important to maintain a healthy weight, engage in moderate exercise, follow a healthy diet, avoid joint injuries, ensure adequate rest, and avoid overusing the joints.",
    'Bronchial Asthma': "Bronchial Asthma is a chronic inflammatory respiratory disease characterized by recurrent episodes of wheezing, chest tightness, and difficulty breathing. To control and prevent asthma symptoms and attacks over the long term, you can use inhaled corticosteroids (such as fluticasone), long-acting beta2 agonists (such as salmeterol), leukotriene receptor antagonists (such as montelukast), and immunomodulators (such as omalizumab); for urgent situations requiring rapid relief of symptoms during an attack, you can use inhaled short-acting beta2 agonists (such as albuterol) or oral or intravenous corticosteroids (for severe attacks). When taking these medications, follow the prescription and guidance of a doctor, and do not change the dosage or discontinue on your own. In daily life, it is important to avoid known triggers, such as tobacco smoke and dust mites, pay attention to outdoor air quality forecasts, engage in regular appropriate exercise, maintain a healthy lifestyle, and manage stress effectively.",
    'Cervical spondylosis': "Cervical spondylosis is a common degenerative condition that increases with age and affects the vertebrae and intervertebral discs of the cervical spine. Treatment options include medication with nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen, as well as physical therapy where a therapist guides the patient through specific exercises. Alternative therapies such as acupuncture, massage, and spinal manipulation may also be considered. For prevention or improvement of cervical spondylosis, it is important to maintain good posture, take regular breaks and change positions frequently, engage in exercises to strengthen and improve the flexibility of neck muscles, maintain a healthy weight, ensure quality sleep, and use an appropriate pillow.",
    'Chicken pox': "Chickenpox is a contagious disease caused by the varicella-zoster virus (VZV). It typically affects children but can also occur in unvaccinated adults. The characteristic symptoms of chickenpox include an itchy, blister-like rash across the body, which may be accompanied by fever, fatigue, and loss of appetite. For most healthy children, chickenpox usually doesn't require special treatment as it tends to resolve on its own within about two weeks. However, for certain groups such as adults, pregnant women, and people with weakened immune systems, antiviral medications like acyclovir may be used, as well as antihistamines to reduce itching and acetaminophen to relieve pain and fever. It is important to always follow the doctor's instructions when taking medication. In terms of prevention, vaccination is the most effective measure against chickenpox. Additionally, since chickenpox is spread through droplets, avoiding contact with infected individuals can prevent transmission. Maintaining good personal hygiene and frequent handwashing are also important in preventing the spread of the virus.",
    'Common Cold': "The common cold, often simply referred to as a cold, is an extremely common minor respiratory infection. There is no cure for the common cold, and treatment primarily involves symptom relief. Over-the-counter medications such as acetaminophen (to reduce pain and fever), antihistamines (to alleviate sneezing, nasal congestion, and runny nose), and cough suppressants are commonly used. In daily life, personal hygiene is important, including frequent handwashing and avoiding close contact with carriers of the cold virus. A balanced diet, regular exercise, and sufficient sleep contribute to a healthy lifestyle that can help boost immunity.",
    'Dengue': "Dengue fever is a tropical infectious disease caused by the dengue virus, primarily transmitted through the bites of Aedes mosquitoes, such as Aedes aegypti or Aedes albopictus. Symptoms may include high fever, severe headache, pain in the muscles and joints, rash, nausea, and vomiting. Symptoms typically appear 5 to 7 days after the bite and can last about a week. There is no specific treatment for dengue fever. Treatment is mainly supportive and symptomatic, starting with ensuring adequate rest to aid the body's recovery. To prevent dehydration, it is important to consume plenty of fluids and electrolytes. Acetaminophen (Paracetamol) can be used to control fever and relieve pain, but aspirin or non-steroidal anti-inflammatory drugs (NSAIDs) should be avoided as they may increase the risk of bleeding. During treatment, it is crucial to follow medical advice strictly, and if symptoms worsen, such as subcutaneous bleeding, persistent vomiting, or severe abdominal pain, medical help should be sought immediately. In everyday life, to prevent dengue fever, one should avoid mosquito bites by using mosquito nets, wearing long-sleeved clothing, applying insect repellent, etc., and also eliminate standing water near living areas to prevent mosquito breeding.",
    'Dimorphic Hemorrhoids': "Hemorrhoids are the pathological dilation of vascular structures in the anal canal, which can sometimes lead to pain, bleeding, and swelling. Treatment for hemorrhoids typically includes lifestyle changes, medication such as topical ointments, creams, or suppositories, and in more severe cases, surgical interventions like rubber band ligation, sclerotherapy, laser treatment, or conventional surgical excision. When undergoing treatment, please follow the guidance of a professional doctor. Additionally, in daily life, it is advisable to eat a diet rich in fiber, ensure adequate fluid intake, avoid sitting for prolonged periods, and avoid straining during bowel movements.",
    'Fungal infection': "Fungal infections are caused by a variety of fungi and typically result in symptoms such as itching, redness, peeling, rashes, and sometimes pain. Topical antifungal medications, such as clotrimazole and terbinafine ointments or creams, can be used to treat skin fungal infections. For more severe infections or those affecting internal organs, oral medications like fluconazole or itraconazole may be required. For specific areas, such as vaginal yeast infections, vaginal suppositories or oral medications may be necessary. It is essential to consult a doctor or pharmacist before using any medication. In daily life, maintaining personal hygiene and keeping the body dry, wearing breathable clothing, avoiding direct contact, eating a balanced diet, and reducing sugar intake can help prevent and alleviate fungal infections.",
    'Hypertension': "Hypertension, commonly known as high blood pressure, is a prevalent chronic condition where the blood pressure in the arteries is persistently elevated above normal levels. Long-term hypertension increases the risk of health issues such as heart disease, stroke, and kidney disease. The following medications can be considered for treatment: diuretics, beta-blockers, calcium channel blockers, angiotensin-converting enzyme (ACE) inhibitors, and angiotensin II receptor blockers (ARBs). It is important to consult a doctor before taking any medication and to monitor blood pressure changes regularly during treatment. In daily life, it is recommended to reduce salt intake, limit alcohol consumption, increase the intake of vegetables and fruits, maintain a healthy weight to reduce the burden on the heart, control weight, and quit smoking.",
    'Impetigo': "Impetigo is a common superficial bacterial infection characterized by the appearance of blisters and pus-filled lesions on the skin, which then burst and form yellow crusts. Topical medications such as mupirocin ointment can be applied directly to the infected area. If the infection is severe or widespread, oral antibiotics like amoxicillin or other drugs targeting specific bacteria may be used. Before using any medication, it is essential to consult a doctor. In daily life, it is important to maintain personal hygiene, keep the skin dry and clean, promptly clean and protect minor wounds, and avoid contact with infected individuals.",
    'Jaundice': "Jaundice is a symptom caused by an elevated level of bilirubin in the blood, which typically results in a yellow discoloration of the skin and eyes. Treatment for jaundice requires addressing the underlying cause, and possible treatments include antiviral or antibiotic therapy, surgery or other procedures to remove stones, discontinuing the use of related drugs, and taking measures to protect the liver. During treatment, it is important to follow the doctor's guidance and medication prescriptions. Methods to prevent jaundice generally involve protecting the liver and maintaining a healthy lifestyle, which includes avoiding excessive alcohol consumption, taking preventive measures to avoid hepatitis virus infections, such as getting vaccinated and avoiding contact with the blood and bodily fluids of infected individuals, and maintaining a healthy diet.",
    'Malaria': "Malaria is an infectious disease caused by the malaria parasite and transmitted to humans through mosquito bites. Symptoms of malaria include fever, chills, headache, muscle and joint pain, fatigue, and anemia. The primary treatment for malaria involves antimalarial drugs, with commonly used medications including chloroquine, quinine, artemether, and amodiaquine. It is important to follow the doctor's advice and prescription when using antimalarial drugs. In daily life, you can take the following preventive measures to avoid malaria: avoiding mosquito bites, getting vaccinated against malaria if available, and using mosquito repellents.",
    'Migraine': "Migraine is a neurological disorakder characterized by recurrent episodes of headache. Treatment options for migraine include acute therapies, such as over-the-counter medications like nonsteroidal anti-inflammatory drugs (e.g., ibuprofen, aspirin), and preventive therapies, including tricyclic antidepressants, beta blockers, and antiepileptic drugs. When using medication for treatment, it is important to follow the doctor's advice and prescription. In daily life, you can take the following preventive measures to reduce the occurrence of migraines: maintaining a regular sleep schedule, identifying and avoiding personal triggers such as specific foods, emotional stress, and weather changes, maintaining a balanced diet, and avoiding excessive alcohol and caffeine consumption.",
    'Pneumonia': "Pneumonia is an infection of the lungs, typically caused by bacteria, viruses, or fungi. It can lead to difficulty breathing, coughing, fever, and sometimes chest pain. The treatment for pneumonia depends on the type of pathogen, as well as the patient's age, health condition, and the severity of symptoms. Bacterial pneumonia is usually treated with antibiotics such as penicillin, amoxicillin, azithromycin, or rifampin, while viral pneumonia may require antiviral medications like oseltamivir. Fungal pneumonia necessitates the use of antifungal medications, such as fluconazole or itraconazole. When taking any medication, it is important to follow the doctor's prescription. Here are some measures to prevent pneumonia, including regular vaccinations like the flu shot and pneumococcal vaccine, frequent hand washing, and maintaining healthy lifestyle habits.",
    'Psoriasis': "Psoriasis is a chronic autoimmune disease that causes excessive growth of skin cells, resulting in symptoms such as red patches, scales, and itchiness. This condition can affect any part of the body, but it is most commonly found on the scalp, elbows, knees, palms, and soles of the feet. Treatment options for Psoriasis include topical medications (such as corticosteroid creams, vitamin D analogues, keratolytics, and anti-inflammatory drugs), phototherapy, and systemic medications. It is important to follow the doctor's advice and prescription when using medication. In daily life, you can take the following preventive measures to reduce the occurrence and alleviate the symptoms of Psoriasis: maintaining a healthy lifestyle, avoiding stress, excessive tension, skin injuries, and infections, and keeping the skin clean and moisturized.",
    'Typhoid': "Typhoid is a serious bacterial infection caused by the bacterium Yersinia pestis. Treatment for plague typically requires the use of antibiotics such as streptomycin, doxycycline, and gentamicin sulfate. When using antibiotics to treat plague, it is important to strictly follow the doctor's instructions for medication. The key to preventing typhoid is to avoid contact with potentially infected fleas and animals.",
    'Varicose Veins': "Varicose Veins is a common vascular condition that primarily affects the veins in the lower limbs. Treatment options for Varicose Veins include compression therapy, medication (such as pain relievers and non-steroidal anti-inflammatory drugs (NSAIDs), and surgical interventions. It is important to follow the doctor's advice when using medication. In daily life, you can take the following preventive measures to reduce the occurrence of Varicose Veins, such as maintaining a healthy weight, exercising, and avoiding tight-fitting clothing.",
    'allergy': "Allergies occur when your immune system overreacts to typically harmless substances known as allergens, such as pollen, dust mites, pet dander, certain foods, or medications. This reaction can lead to a variety of symptoms, including sneezing, runny nose, itchy skin, redness of the eyes, and swelling. Treatment for allergies depends on the type and severity of the allergic reaction. Common medications include antihistamines, nasal sprays, and eye drops. When taking allergy medication, it is important to strictly follow the doctor's prescription instructions. The key to preventing allergies is to avoid contact with known allergens, understand and avoid the foods or substances that trigger your allergic reactions, regularly clean your home, and minimize outdoor activities during pollen season.",
    'diabetes': "Diabetes is a chronic condition characterized by abnormally high levels of blood sugar (glucose). There are primarily two types of diabetes: Type 1 Diabetes: Usually occurs in children or young adults. Type 2 Diabetes: More common and associated with age, body weight, and lifestyle. Medical Treatment includes patients with Type 1 diabetes require lifelong insulin injections to control blood sugar levels. Those with Type 2 diabetes may need oral medications, such as Metformin, and sometimes insulin or other injectable medications as well. When taking diabetes medications, it is important to follow the doctor's instructions and not change the dosage on your own.  While Type 1 diabetes cannot be prevented, the risk of developing Type 2 diabetes can be reduced by maintaining a healthy weight, eating a balanced diet. Regularly exercising and increasing physical activity, regularly checking blood sugar levels, especially if there is a family history of diabetes.",
    'drug reaction': "A drug reaction refers to an adverse response or side effects that occur after taking a specific medication. These reactions can range from mild to severe and may include symptoms such as rashes, hives, fever, joint pain, difficulty breathing, swelling, among others. In some cases, a drug reaction can lead to anaphylactic shock, an emergency situation that requires immediate medical attention. The treatment for a drug reaction depends on the severity of the reaction and the specific symptoms. Mild reactions, such as rashes or hives, may only require antihistamines, while severe reactions, like difficulty breathing or anaphylactic shock, require urgent medical intervention. When dealing with a drug reaction, it is important to immediately discontinue the medication that caused the reaction and contact your doctor. Measures to prevent drug reactions include: informing your doctor of all known drug allergies and past drug reactions before starting any new medication, carefully reading the medication's instructions to understand potential side effects, and following the doctor's prescription guidance without altering the dosage or discontinuing the medication on your own.",
    'gastroesophageal reflux disease': "Gastroesophageal reflux disease (GERD) is a common digestive disorder characterized by the backflow of stomach acids or contents into the esophagus, causing symptoms and/or complications. Common symptoms include a burning sensation behind the breastbone (heartburn), sour taste in the mouth, difficulty swallowing, sore throat, or cough. Treatment for GERD may involve lifestyle changes and medication, with primary drugs including proton pump inhibitors (PPIs), H2 receptor antagonists, antacids, and prokinetics. It is important to strictly follow the doctor's prescription and guidance when using medications. Measures to prevent gastroesophageal reflux disease include maintaining a healthy weight, avoiding overeating, reducing or avoiding consumption of irritant foods and beverages, quitting smoking, and avoiding tight-fitting clothing.",
    'peptic ulcer disease': "Peptic ulcer disease (PUD) refers to damage to the inner lining of the stomach or duodenum (the beginning part of the small intestine), resulting in ulcers. Common symptoms include upper abdominal pain, discomfort after eating, nausea, vomiting, or weight loss. Treatment for stomach ulcers typically involves a combination of medications, such as antacids, acid secretion inhibitors, antibiotics, and mucosal protectants. It is essential to consult with a healthcare professional for an accurate diagnosis and personalized treatment. To prevent peptic ulcer disease, measures include avoiding excessive use of nonsteroidal anti-inflammatory drugs (NSAIDs), regulating your diet to avoid excessive intake of spicy, fatty, or acidic foods, quitting smoking and limiting alcohol consumption, and maintaining good hygiene practices.",
    'urinary tract infection': "Urinary tract infections (UTIs) refer to infections in any part of the urinary system, which includes the kidneys, ureters, bladder, and urethra. UTIs are commonly caused by bacteria, particularly Escherichia coli (E. coli). Symptoms may include a strong urge to urinate, frequent urination, pain in the lower abdomen, cloudy or bloody urine, and a strong odor to the urine. The common treatment for a UTI is the use of antibiotics, such as amoxicillin, nitrofurantoin, and trimethoprim/sulfamethoxazole. To prevent UTIs, it is recommended to maintain adequate fluid intake, wear breathable cotton underwear, and practice other hygiene measures."
}

howto= """Welcome to the <b>Medical Chatbot</b>, powered by Gradio.
Currently, the chatbot can WELCOME YOU, PREDICT DISEASE based on your symptoms and SUGGEST POSSIBLE SOLUTIONS AND RECOMENDATIONS, and BID YOU FAREWELL.
<b>How to Start:</b> Simply type your messages in the textbox to chat with the Chatbot and press enter!<br><br>
The bot will respond based on the best possible answers to your messages."""


# Create the gradio demo
with gr.Blocks(css = """#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 520px; overflow: auto;}""") as demo:
  gr.HTML('<h1 align="center"> Medical Chatbot: ARIN 7102 project 🌟🏥🤖 (RNN/GRU + TfidfVectorizer in sklearn)') # (Phi-2 + bge-small-en-v1.5) # (RNN/GRU + TfidfVectorizer in sklearn)
  with gr.Accordion("Follow these Steps to use the Gradio WebUI", open=True):
      gr.HTML(howto)
  chatbot = gr.Chatbot()
  msg = gr.Textbox()
  clear = gr.ClearButton([msg, chatbot])
  def respond(message, chat_history, base_model = "gru_model", device='cpu'): #embedder = SentenceTransformer("bge-small-en-v1.5", device="cuda"), device='cuda'): # "meta-llama/Meta-Llama-3-70B"
                                    #base_model = /home/henry/Desktop/ARIN_7102/download/phi-2 # gru_model # embedder = SentenceTransformer("/home/henry/Desktop/ARIN_7102/download/bge-small-en-v1.5", device="cuda")
        if base_model == "gru_model":
           # Model and transforms preparation
            model= RNN_model().to(device)
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
                transformed_new= vectorizer.transform([message])
                transformed_new= torch.tensor(transformed_new.toarray()).to(torch.float32).to(device)
                #transform_text= vectorizer.transform([message])
                #sentence_embeddings = embedder.encode(message)
                #sentence_embeddings = torch.from_numpy(sentence_embeddings).float().to(device).unsqueeze(dim=0)
                #sentence_embeddings.shape
                #transform_text= torch.tensor(transform_text.toarray()).to(torch.float32)
                model.eval()
                with torch.inference_mode():
                    y_logits=model(transformed_new.to(device))
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
demo.launch()
#demo.launch(share=True)


#gr.ChatInterface(respond).launch()

#a = respond('hi', list())
#a = respond("Hi, good morning")
#a = respond("My skin has been peeling, especially on my knees, elbows, and scalp. This peeling is often accompanied by a burning or stinging sensation.")
#a = respond("I have blurry vision, and it seems to be getting worse. I'm continuously fatigued and worn out. I also occasionally have acute lightheadedness and vertigo, can you give me some advice?")
#print(a)
#gr.ChatInterface(respond).launch()
#demo = gr.ChatInterface(fn=random_response, examples=[{"text": "Hello", "files": []}], title="Echo Bot", multimodal=True)
#demo.launch()