import pandas as pd
import numpy as np
import torch
from balanced_loss import Loss
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, set_seed
set_seed(42)
from peft import LoraConfig
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

llama_token = '[insert-llama-access-token-here]'

df = pd.read_excel('novel_generate_v3.xlsx', keep_default_na=False)
idx_values = df.index.values.tolist()
text_values = df.text.values.tolist()
label_values = df.label.values.tolist()
# Define pretrained tokenizer and model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=llama_token)
tokenizer.pad_token = tokenizer.eos_token
BATCH_SIZE = 1
max_len = 512
LR = 5e-5
EPOCH = 1
loss_type = "cross_entropy"
class_balanced = True
random_state_list = [111,222,555,666,777]
excel_name = 'novel_generate_v3_onlyU_curri2_2.xlsx'
df_curri = pd.read_excel(excel_name, keep_default_na=False)

df_avg_name = f'df_llama_2-2_epoch{EPOCH}_4bit_{LR}_prompt.xlsx'
df_avg = pd.DataFrame()

quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

# ----- 1. Preprocess data -----#
# Preprocess data

for random_state in random_state_list:
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, quantization_config=quantization_config, device_map="auto", token=llama_token)
	model.config.pad_token_id = tokenizer.eos_token_id
	model.add_adapter(lora_config, adapter_name="adapter_1")

	idx_train_ori, idx_test, X_train_ori, X_test, y_train_ori, y_test = train_test_split(idx_values, text_values, label_values, shuffle=True, test_size=0.4, random_state=random_state)
	idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(idx_train_ori, X_train_ori, y_train_ori, shuffle=True, test_size=1/3, random_state=random_state)
	df_train = pd.DataFrame({'chosen':idx_train,'label':y_train})
	idx_train_onlyU = df_train[df_train.label==1].chosen.to_list()

	X_train = X_train + df_curri.loc[idx_train_onlyU].text.to_list() 
	y_train = y_train + df_curri.loc[idx_train_onlyU].label.to_list()

	X_train_tokenized = tokenizer(X_train, padding='max_length', truncation=True, max_length=max_len)
	X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True, max_length=max_len)
	
	# Create torch dataset
	class Dataset(torch.utils.data.Dataset):
		def __init__(self, encodings, labels=None):
			self.encodings = encodings
			self.labels = labels
		def __getitem__(self, idx):
			item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
			if self.labels:
				item["labels"] = torch.tensor(self.labels[idx])
			return item
		def __len__(self):
			return len(self.encodings["input_ids"])

	train_dataset = Dataset(X_train_tokenized, y_train)
	val_dataset = Dataset(X_test_tokenized, y_test)
	# ----- 2. Fine-tune pretrained model -----#
	# Define Trainer parameters
	def compute_metrics(p):
		pred, labels = p
		pred = np.argmax(pred, axis=-1)
		accuracy = accuracy_score(y_true=labels, y_pred=pred)
		return {"accuracy": accuracy}
	# Define Trainer
	class CustomTrainer(Trainer):
		def compute_loss(self, model, inputs, return_outputs=False):
			labels = inputs.get("labels")
			outputs = model(**inputs)   
			logits = outputs.get('logits')
			loss_fct = Loss(
				loss_type=loss_type, 
				samples_per_class=[6590,92],
				class_balanced=class_balanced
			)		
			# cross entropy loss for classifier
			loss = loss_fct(logits, labels)
			return (loss, outputs) if return_outputs else loss
	args = TrainingArguments(
		output_dir="output",
		evaluation_strategy="epoch",
		per_device_train_batch_size=BATCH_SIZE,
		per_device_eval_batch_size=BATCH_SIZE,
		learning_rate=LR,
		num_train_epochs=EPOCH,
		save_strategy='epoch',
		fp16=True,
		weight_decay=0.01,
	)
	trainer = CustomTrainer(
		model=model,
		args=args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		compute_metrics=compute_metrics,
	)
	# Train pre-trained model
	trainer.train()
	
	# ----- 3. Predict -----#
	# Create torch dataset
	test_dataset = Dataset(X_test_tokenized)

	# Make prediction
	raw_pred, _, _ = trainer.predict(test_dataset)

	# Preprocess raw predictions
	y_pred = np.argmax(raw_pred, axis=1)
	print('random_state =', random_state)
	print(classification_report(y_test, y_pred, target_names=["Common","Unique"], digits=4))
	print(confusion_matrix(y_test, y_pred))

	df_vote = pd.DataFrame(columns=['y_pred', 'label', 'X_test'])
	df_vote['y_pred'] = y_pred
	df_vote['label'] = y_test
	df_vote['X_test'] = X_test

	df_avg = pd.concat([df_avg, df_vote])

print('random state average')
print(classification_report(df_avg.label, df_avg.y_pred, target_names = ['Common','Unique'], digits=4))
print(confusion_matrix(df_avg.label, df_avg.y_pred))
df_avg.to_excel(df_avg_name, index=False)