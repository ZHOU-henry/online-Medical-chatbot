
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
#####################################################################################################################

def preprocess_data(label_X, target_y):
    preprocessed= TensorDataset(label_X, target_y)
    return preprocessed

def dataloader(dataset, batch_size, shuffle, num_workers):
    dataloader= DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle= shuffle,
                           num_workers=num_workers)
    return (dataloader)

class RNN_model(nn.Module):
  def __init__(self):
    super().__init__()

    self.rnn= nn.RNN(input_size=1477, hidden_size=240,num_layers=1, nonlinearity= 'relu', bias= True).to('cuda')
    self.output= nn.Linear(in_features=240, out_features=24).to('cuda')

  def forward(self, x):
    y, hidden= self.rnn(x)
    y = y.to('cuda')
    x= self.output(y).to('cuda')
    return(x)
#####################################################################################################################
# import data
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
#####################################################################################################################
train_data, test_data= train_test_split(df, test_size=0.15, random_state=42 )
train_data['label'].value_counts().sort_index()
test_data['label'].value_counts().sort_index()
#vectorizer= nltk_u.vectorizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.de.stop_words import STOP_WORDS
vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
vectorizer.fit(train_data.text)
vectorizer.get_feature_names_out()[: 100]
vectorizer= vectorizer
data_input= vectorizer.transform(train_data.text)
test_data_input= vectorizer.transform(test_data.text)
#####################################################################################################################
# Convert vectors to tensors
input_data_tensors= torch.tensor(data_input.toarray()).to(torch.float32)
test_data_tensors= torch.tensor(test_data_input.toarray()).to(torch.float32)
train_data_output= torch.tensor(train_data['label'].values)
test_data_output= torch.tensor(test_data['label'].values)
train_dataset= preprocess_data(input_data_tensors, train_data_output)
test_dataset= preprocess_data(test_data_tensors, test_data_output)
train_dataloader= dataloader(dataset=train_dataset,
                                             batch_size=32, shuffle= True, num_workers=2)
test_dataloader= dataloader(dataset=test_dataset,
                                             batch_size=32, shuffle= False, num_workers=2)
text, target= next(iter(train_dataloader))
#####################################################################################################################
if torch.cuda.is_available():
    device = "cuda"
    print(f'################################################################# device: {device}#################################################################')
else:
    device = "cpu"
#####################################################################################################################
model= RNN_model().to(device)
loss_fn= CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(), lr= 0.1, weight_decay=0)
#####################################################################################################################
## train model
epoch= 300

results= {
      "train_loss": [],
      "train_accuracy": [],
      "test_loss": [],
      "test_accuracy": []
      }

for i in range(epoch):
  train_loss=0
  train_acc=0
  for batch, (X, y) in enumerate(train_dataloader):
    X, y= X.to('cuda'), y.to('cuda')
    # Train the model
    model.train()
    optimizer.zero_grad()
    y_logits= model(X).to('cuda')
    # Calculate the loss
    loss= loss_fn(y_logits, y).to('cuda')
    train_loss += loss
    # ypreds
    y_preds= torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    accuracy = accuracy_score(y.cpu(), y_preds.cpu())
    train_acc += accuracy
    # zero grad
    optimizer.zero_grad()
    # Loss backward
    loss.backward()
    # Optimizer step
    optimizer.step()
  train_loss /= len(train_dataloader)
  train_acc /=len(train_dataloader)
  test_loss = 0
  test_acc=0
  model.eval()
  with torch.inference_mode():
    for X, y in test_dataloader:
      X, y= X.to('cuda'), y.to('cuda')
      y_logits= model(X).to('cuda')
      loss= loss_fn(y_logits, y).to('cuda')
      test_loss += loss
      test_preds= torch.argmax(torch.softmax(y_logits, dim=1), dim=1).to('cuda')
      accuracy = accuracy_score(y.cpu(), test_preds.cpu())
      test_acc += accuracy
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  results['train_loss'].append(train_loss)
  results['train_accuracy'].append(train_acc)
  results['test_loss'].append(test_loss)
  results['test_accuracy'].append(test_acc)
  if i % 50 == 0:
    print(f"\nTrain loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | Test loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} |")

#####################################################################################################################
'''
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(results['train_loss'], label= 'train')
plt.plot(results['test_loss'], label= 'test')
plt.title('loss curve for train and test')
plt.legend()
plt.subplot(1,2,2)
plt.plot(results['train_accuracy'], label= 'train')
plt.plot(results['test_accuracy'], label= 'test')
plt.title('accuracy score for train and test')
plt.legend()
'''
#####################################################################################################################
new_data= 'I have been having burning pain anytime i am peeing, what could be the issue?'
transformed_new= vectorizer.transform([new_data])
transformed_new= torch.tensor(transformed_new.toarray()).to(torch.float32).to('cuda')
model.eval()
with torch.inference_mode():
  y_logits=model(transformed_new).to('cuda')
  test_preds= torch.argmax(torch.softmax(y_logits, dim=1), dim=1).to('cuda')
  test_pred= target_dict[test_preds.item()]
  print(f'based on your symptoms, I believe you are having {test_pred}')

target_dir_path = Path('')
target_dir_path.mkdir(parents=True,
                      exist_ok=True)
model_path= target_dir_path / 'pretrained_gru_model.pth'
torch.save(obj=model.state_dict(),f= model_path)
print('########### model saved ###########')
#####################################################################################################################