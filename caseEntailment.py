# CSI 5386
# Final project
# Michel Custeau, 8658589
# Beril Borali, 300036112



import os
import pandas as pd
import numpy as np
import re
import random
import swifter
import spacy
from spacy.lang.en import English
import pickle
from tqdm import tqdm
import nltk
import string
import json
import zipfile
import torch
from torch.utils.data import DataLoader,TensorDataset
from rank_bm25 import BM25Okapi as BM25
from sentence_transformers import SentenceTransformer, util, InputExample,losses
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, AdamW, get_scheduler,AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer



nlp = English()

wn = nltk.WordNetLemmatizer()

class ForT5Dataset(torch.utils.data.Dataset):
	def __init__(self, inputs, targets):
		self.inputs = inputs
		self.targets = targets

	def __len__(self):
		return len(self.targets.input_ids)

	def __getitem__(self, index):
		input_ids = torch.tensor(self.inputs["input_ids"][index]).squeeze()
		target_ids = torch.tensor(self.targets["input_ids"][index]).squeeze()

		input_ids_am = torch.tensor(self.inputs["attention_mask"][index]).squeeze()
		target_ids_am = torch.tensor(self.targets["attention_mask"][index]).squeeze()

		return {"input_ids": input_ids, "input_attention_mask": input_ids_am, "labels": target_ids, "labels_attention_mask": target_ids_am}




class caseEntailment():

	def __init__(self, datasetName='COLIEE2021', stopwords_file='stopwords.txt', resultFile='Results.txt', python36=False):
					
		self.stopwords_file = stopwords_file
		self.stopwordLst = []
		self.resultFile = resultFile
		self.datasetName = datasetName

		if(datasetName=='COLIEE2021'):
			self.test_folder = datasetName+"/task_2/task2_2021_test_nolabels"
			self.train_folder = datasetName+"/task_2/task2_2021_train"
			self.test_labels = datasetName+"/task_2/task2_test_labels_2021.json"
			self.train_labels = datasetName+"/task_2/task2_train_labels_2021.json"
		elif(datasetName=='COLIEE2022'):
			self.test_folder = datasetName+"/task_2/task2_2022_test"
			self.train_folder = datasetName+"/task_2/task2_2022_train"
			self.test_labels = datasetName+"/task_2/task2_test_labels_2022.json"
			self.train_labels = datasetName+"/task_2/task2_train_labels_2022.json"
		else:
			raise Exception("Invalid dataset name")

		self.caseDataFrame = None
		self.caseDataFrame_train = None

		self.bm25 = None 
		self.model = None
		self.customStopwordLst = [" " , "\n", "'s", "...", "\n ", " \n", " \n ", "\xa0"]

		if(python36):
			self.pickleFilePath = datasetName+"/task_2/test_pickles/clean_query_36"
			self.pickleFilePath_train = datasetName+"/task_2/train_pickles/clean_query_36"

		else:
			self.pickleFilePath = datasetName+"/task_2/test_pickles/clean_query_coliee2021"
			self.pickleFilePath_train = datasetName+"/task_2/train_pickles/clean_query_coliee2021"


########## Clean Data

	# creates a list of stopwords
	@staticmethod
	def read_and_parse_stopwords_file(self):

		# Open file from string 
		file = open(self.stopwords_file, "r")

		# Read from provided stopwords_file file
		raw_data_stopwords = file.read()

		# Assign list of stopwords_file
		self.stopwordLst = raw_data_stopwords.replace('\t', '\n').split('\n')

		file.close()

		pass

	# tokenize and lemmatize words
	@staticmethod
	def cleanText(self,txt):

		my_doc = nlp(txt)
		tokens = [token.text for token in my_doc]

		lst = []
		for word in tokens:
			low = word.lower()
			lem = wn.lemmatize(low)
			if (lem not in self.stopwordLst) and (lem not in self.customStopwordLst) and (lem not in list(string.punctuation)):
				lst.append(lem)
			
		return lst


	########## Initialize Data

	# Creates a pandas DataFrame with columns: case_number, paragraphs, paragraph_names, base_case, entailed_fragment
	# it also contains a "clean" version of some of the columns: base_case_clean, entailed_fragment_clean, paragraphs_clean						
	@staticmethod
	def dataFrameCreation(self, folderCategory, dataframe, pickleFilePath):

		folder_names = []
		paragraph_per_folder = []
		paragraph_names_per_folder = []
		query_cases = []
		decisions = []

		# parse through each case
		for folder_name in os.listdir(folderCategory):

			if not folder_name.startswith('.'): 
				folder = os.path.join(folderCategory, folder_name)
				paragraphs = []
				paragraph_names = []

			# create list of paragraph names and paragraph content of the specific case
			for paragraph_name in os.listdir(folder+'/paragraphs'):
				
		
				f = open(os.path.join(folder+'/paragraphs', paragraph_name))
				paragraphs.append(f.read())
				paragraph_names.append(paragraph_name.replace(".txt",""))

			# extract the base case and the decision of the specific case
			base_case = open(folder+'/base_case.txt')
			decision = open(folder+'/entailed_fragment.txt')

			folder_names.append(folder_name)
			paragraph_per_folder.append(paragraphs)
			paragraph_names_per_folder.append(paragraph_names)
			query_cases.append(base_case.read())
			decisions.append(decision.read())

		dataframe = pd.DataFrame({'case_number': folder_names, 'paragraphs': paragraph_per_folder, 'paragraph_names': paragraph_names_per_folder, 'base_case': query_cases, 'entailed_fragment': decisions})								

		dataframe['base_case_clean'] = dataframe['base_case'].swifter.apply(lambda x: self.cleanText(self, x))

		dataframe['entailed_fragment_clean'] = dataframe['entailed_fragment'].swifter.apply(lambda x: self.cleanText(self, x))

		paragraphs_clean = []
		for paras in paragraph_per_folder:
			l=[]
			for p in paras:
				p_clean = self.cleanText(self, p)
				l.append(p_clean)
			paragraphs_clean.append(l)

		dataframe['paragraphs_clean'] = paragraphs_clean

		dataframe.to_pickle(pickleFilePath)

	# create test and train DataFrame
	def createDataFrames(self):

		self.read_and_parse_stopwords_file(self)
		self.dataFrameCreation(self, self.test_folder, self.caseDataFrame, self.pickleFilePath)
		self.dataFrameCreation(self, self.train_folder, self.caseDataFrame_train, self.pickleFilePath_train)

	# Once DataFrames have been created, they can be retreived using this function
	def preProcess(self):
		self.caseDataFrame = pd.read_pickle(self.pickleFilePath)
		self.caseDataFrame_train = pd.read_pickle(self.pickleFilePath_train)


	########## Evaluate Data

	@staticmethod

	def calculateRecall(self, results, labels):
		lf = open(labels, "r")
		rf = open(results, "r")

		results_lines = rf.readlines()
		labels_json = json.load(lf)
		rel_num = 0
		rel_cases = 0
		for label in labels_json.keys():
			relevant_docs = labels_json[label].split(", ")
			rel_cases += len(relevant_docs)
			for i in range(len(relevant_docs)):
				relevant_docs[i] = relevant_docs[i].replace(".txt","")
			retreived = []
			for line in results_lines:
				line_split = line.split('\t')
				if(line_split[0]==label.replace(".txt","")):
					retreived.append(line_split[2])


			for i in range(len(retreived)):
				if(retreived[i] in relevant_docs):
					rel_num+=1

		recall = rel_num/rel_cases
		print("recall :", recall)
		return recall


	@staticmethod
	def calculatePrecision(self, results, labels):
		lf = open(labels, "r")
		rf = open(results, "r")
		results_lines = rf.readlines()
		labels_json = json.load(lf)
		retreived_cases = 0
		rel_num = 0
		num_queries=0
		for label in labels_json.keys():
			num_queries +=1
			relevant_docs = labels_json[label].split(", ")
			for i in range(len(relevant_docs)):
				relevant_docs[i] = relevant_docs[i].replace(".txt","")
			retreived = []
			for line in results_lines:
				line_split = line.split('\t')
				if(line_split[0]==label.replace(".txt","")):
					retreived.append(line_split[2])
		
			for i in range(len(retreived)):
				retreived_cases +=1
				if(retreived[i] in relevant_docs):
					rel_num+=1

		# retreived_cases = num_queries*topn
		precision = rel_num/retreived_cases
		print("precision :", precision)
		return precision



	def calculateF1(self, results, labels):
		recall = self.calculateRecall(self, results, labels)
		precision = self.calculatePrecision(self, results, labels)
		f1 = (2*precision*recall)/(precision+recall)
		print("f1 :", f1)
		return f1


######### Preprocess for training


	def preProcessT5(self, train=True, undersample = False, oversample=False):

		if(train):
			case_df = self.caseDataFrame_train
			label_file = self.train_labels
		else:
			case_df = self.caseDataFrame
			label_file = self.test_labels

		lf = open(label_file, "r")
		labels_json = json.load(lf)

		if(undersample):

			sentences_positive = []
			labels_positive = []
			sentences_negative = []
			labels_negative = []

			totalCases = len(case_df)
			for caseNum in tqdm(range(totalCases)):
				case_number = case_df['case_number'][caseNum]
				relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number]]
				for i in range(len(case_df['paragraphs'][caseNum])):
					if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
						sentences_positive.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_positive.append("true")
					else:
						sentences_negative.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_negative.append("false")


				np.random.shuffle(sentences_negative)
				sentences_negative = sentences_negative[:len(sentences_positive)]
				labels_negative = labels_negative[:len(sentences_positive)]

				sentences = sentences_negative + sentences_positive
				labels = labels_positive + labels_negative

				with open("./"+self.datasetName+"_sentences_undersampled.pickle", "wb") as f:
					pickle.dump(sentences, f)
				with open("./"+self.datasetName+"_labels_undersamples.pickle", "wb") as f:
					pickle.dump(labels, f)

		elif(oversample):
			sentences_positive = []
			labels_positive = []
			sentences_artificial = []
			labels_artificial = []
			sentences_negative = []
			labels_negative = []

			totalCases = len(case_df)
			for caseNum in tqdm(range(totalCases)):
				case_number = case_df['case_number'][caseNum]
				#if(case_number=='016'):
				base_case = case_df['base_case'][caseNum]
				base_paragraphs = re.split('\[\d{1,2}\]', base_case)

				
				for z in range(len(base_paragraphs)):
					if(case_df["entailed_fragment"][caseNum] in base_paragraphs[z]):
						#print(base_paragraphs[z])
						frag_paragraph  = base_paragraphs[z].split(" ")
						break
				
				frag_paragraph = list(filter(lambda c: c!='', frag_paragraph))
				frag_paragraph = list(filter(lambda c: c!='\n', frag_paragraph))
				frag_paragraph = list(filter(lambda c: c!='\t', frag_paragraph))
				artificial_examples = []
				
				start2 = 0
				end2 = random.randint(25, 55)
				stride = int(end2/4)

				while(end2<len(frag_paragraph)):
					artificial_examples.append(' '.join(frag_paragraph[start2:end2]))
					start2+=stride
					end2+=stride

				relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number]]
				print(relevant_paragraphs)
				for i in range(len(case_df['paragraphs'][caseNum])):
					if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
						sentences_positive.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_positive.append("true")
						for example in artificial_examples:
							sentences_artificial.append([example, case_df['paragraphs'][caseNum][i]])
							labels_artificial.append("true")
		
					else:
						sentences_negative.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_negative.append("false")

				

			

			print(len(sentences_positive))
			print(len(sentences_artificial))
			print(len(sentences_negative))

			sentences = sentences_positive + sentences_artificial + sentences_negative
			labels = labels_positive + labels_artificial + labels_negative

			# with open("./"+self.datasetName+"_sentences_oversampled.pickle", "wb") as f:
			# 	pickle.dump(sentences, f)
			# with open("./"+self.datasetName+"_labels_oversampled.pickle", "wb") as f:
			# 	pickle.dump(labels, f)

		else:
			sentences = []
			labels = []

			ent = 0
			ex = 0

			totalCases = len(case_df)
			for caseNum in tqdm(range(totalCases)):
				case_number = case_df['case_number'][caseNum]
				relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number].split(", ")]
				print(relevant_paragraphs)
				for i in range(len(case_df['paragraphs'][caseNum])):
					sentences.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
					if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
						labels.append("true")
						ent+=1
						print("oiui")
					else:
						labels.append("false")
						ex+=1
		print(ent)
		print(ex)

		return sentences, labels
		



	def preProcessBERT(self, train=True, undersample = False, oversample=False):

		if(train):
			case_df = self.caseDataFrame_train
			label_file = self.train_labels
		else:
			case_df = self.caseDataFrame
			label_file = self.test_labels

		lf = open(label_file, "r")
		labels_json = json.load(lf)

		if(undersample):

			sentences_positive = []
			labels_positive = []
			sentences_negative = []
			labels_negative = []

			totalCases = len(case_df)
			for caseNum in tqdm(range(totalCases)):
				case_number = case_df['case_number'][caseNum]
				relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number]]
				for i in range(len(case_df['paragraphs'][caseNum])):
					if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
						sentences_positive.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_positive.append(1)
					else:
						sentences_negative.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_negative.append(0)


				np.random.shuffle(sentences_negative)
				sentences_negative = sentences_negative[:len(sentences_positive)]
				labels_negative = labels_negative[:len(sentences_positive)]

				sentences = sentences_negative + sentences_positive
				labels = labels_positive + labels_negative

				with open("./"+self.datasetName+"_sentences_undersampled.pickle", "wb") as f:
					pickle.dump(sentences, f)
				with open("./"+self.datasetName+"_labels_undersamples.pickle", "wb") as f:
					pickle.dump(labels, f)

		elif(oversample):
			sentences_positive = []
			labels_positive = []
			sentences_artificial = []
			labels_artificial = []
			sentences_negative = []
			labels_negative = []

			totalCases = len(case_df)
			for caseNum in tqdm(range(totalCases)):
				case_number = case_df['case_number'][caseNum]
				#if(case_number=='016'):
				base_case = case_df['base_case'][caseNum]
				base_paragraphs = re.split('\[\d{1,2}\]', base_case)

				
				for z in range(len(base_paragraphs)):
					if(case_df["entailed_fragment"][caseNum] in base_paragraphs[z]):
						#print(base_paragraphs[z])
						frag_paragraph  = base_paragraphs[z].split(" ")
						break
				
				frag_paragraph = list(filter(lambda c: c!='', frag_paragraph))
				frag_paragraph = list(filter(lambda c: c!='\n', frag_paragraph))
				frag_paragraph = list(filter(lambda c: c!='\t', frag_paragraph))
				artificial_examples = []
				
				start2 = 0
				end2 = random.randint(25, 55)
				stride = int(end2/9)

				while(end2<len(frag_paragraph)):
					artificial_examples.append(' '.join(frag_paragraph[start2:end2]))
					start2+=stride
					end2+=stride

				relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number]]
				for i in range(len(case_df['paragraphs'][caseNum])):
					if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
						sentences_positive.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_positive.append(1)
						for example in artificial_examples:
							sentences_artificial.append([example, case_df['paragraphs'][caseNum][i]])
							labels_artificial.append(1)
		
					else:
						sentences_negative.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
						labels_negative.append(0)

				



			print(len(sentences_positive))
			print(len(sentences_artificial))
			print(len(sentences_negative))

			sentences = sentences_positive + sentences_artificial + sentences_negative
			labels = labels_positive + labels_artificial + labels_negative

			with open("./"+self.datasetName+"_sentences_oversampled.pickle", "wb") as f:
				pickle.dump(sentences, f)
			with open("./"+self.datasetName+"_labels_oversampled.pickle", "wb") as f:
				pickle.dump(labels, f)

		else:
			sentences = []
			labels = []

			totalCases = len(case_df)
			for caseNum in tqdm(range(totalCases)):
				case_number = case_df['case_number'][caseNum]
				relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number]]
				for i in range(len(case_df['paragraphs'][caseNum])):
					sentences.append([case_df["entailed_fragment"][caseNum], case_df['paragraphs'][caseNum][i]])
					if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
						labels.append(1)
					else:
						labels.append(0)
			with open("./"+self.datasetName+"_sentences.pickle", "wb") as f:
					pickle.dump(sentences, f)
			with open("./"+self.datasetName+"_labels.pickle", "wb") as f:
					pickle.dump(labels, f)

		return sentences, labels





	def preProcessPairs(self, sentence_pairs, labels):
		#USE preProcessBERT() before calling this function
		data_dict={}
		pair={}

		for i in range(len(labels)):

			if labels[i]=="Entailment":
				pair={"label":1, "text":sentence_pairs[i]}
			else:
				pair={"label":0, "text":sentence_pairs[i]} 
			data_dict.append(pair)

		return data_dict





########## Training

## T5

	def trainT5(self,epochs=3):
		with open("./"+self.datasetName+"_sentences_oversampled.pickle", "rb") as f:
				paragraph_pairs=pickle.load(f)
		with open("./"+self.datasetName+"_labels_oversampled.pickle", "rb") as f:
				labels=pickle.load(f)

		tokenizer = T5Tokenizer.from_pretrained("t5-base")
		model = T5ForConditionalGeneration.from_pretrained("t5-base")
		#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

		tokenized_paragraphs = tokenizer(paragraph_pairs, padding=True, return_tensors="pt", truncation=True)
		tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt")

		train_data = ForT5Dataset(tokenized_paragraphs, tokenized_labels)

		loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

		for batch in loader:
			break
		print({k: v.shape for k, v in batch.items()})

		optim = AdamW(model.parameters(), lr=3e-4)

		num_epochs = epochs
		start_epoch = 0

		device = torch.device('cuda')
		model.to(device)

		# checkpoint = torch.load('./models/checkpoint5_am.pth.tar', map_location='cpu')
		# model.load_state_dict(checkpoint['model_state_dict'])
		# optim.load_state_dict(checkpoint['optimizer_state_dict'])
		# last_epoch = checkpoint['epoch']
		# loss = checkpoint['loss']


		print(next(model.parameters()).is_cuda) # returns a boolean


		model.train()

		start_epoch=0
		for epoch in range(start_epoch, num_epochs):
			loop = tqdm(loader)
			for batch in loop:

				paragraphs = batch['input_ids'].to(device)
				labels = batch['labels'].to(device)
				paragraphs_am = batch['input_attention_mask'].to(device)

				output = model(input_ids=paragraphs, labels=labels, attention_mask=paragraphs_am)


				loss = output.loss
				loss.backward()
				optim.step()
				optim.zero_grad()

			torch.save({

			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optim.state_dict(),
			'loss': loss,

			}, './models/checkpoint.pth.tar')

			print('saved checkpoint')
		model.save_pretrained('./models/t5_2021_5ep_oversampled')


## sBERT


	def TrainBERT(self, model_path, epochs=3,model_name="./models/bert_base_model_test",undersample = False, oversample=False):
		# model_name="bert-base-uncased"

		with open("./"+self.datasetName+"_sentences.pickle", "rb") as f:
				sentence_pairs=pickle.load(f)
		with open("./"+self.datasetName+"_labels.pickle", "rb") as f:
				labels=pickle.load(f)
		 
		if (undersample):
			with open("./"+self.datasetName+"_sentences_undersampled.pickle", "rb") as f:
				sentence_pairs=pickle.load(f)
			with open("./"+self.datasetName+"_labels_undersampled.pickle", "rb") as f:
				labels=pickle.load(f)
		elif (oversample):
			with open("./"+self.datasetName+"_sentences_oversampled.pickle", "rb") as f:
				sentence_pairs=pickle.load(f)
			with open("./"+self.datasetName+"_labels_oversampled.pickle", "rb") as f:
				labels=pickle.load(f) 

		self.model = self.transformer_preprocess(model_name)

		
		train_examples = []
		

		for i in range(len(sentence_pairs)):
			train_examples.append(InputExample(texts=[sentence_pairs[i][0], sentence_pairs[i][1]], label=float(labels[i])))

		train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
		train_loss = losses.CosineSimilarityLoss(self.model)

		device = torch.device('cuda')
		self.model.to(device)
		print(next(self.model.parameters()).is_cuda) # returns a boolean

		self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100, output_path=model_path)



## BERT for Classification


	def TrainBERTClassification(self, paragraph_pairs, labels, model_path, model_name,epochs=5, ):

		tokenizer = AutoTokenizer.from_pretrained(model_name)

		self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
		#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

		tokenized_paragraphs = tokenizer(paragraph_pairs, padding=True, return_tensors="pt", truncation=True)

		## convert lists to tensors

		train_seq = torch.tensor(tokenized_paragraphs['input_ids'])
		train_mask = torch.tensor(tokenized_paragraphs['attention_mask'])
		train_y = torch.tensor(labels)

		# wrap tensors
		train_data = TensorDataset(train_seq, train_mask, train_y)

		loader = DataLoader(train_data, batch_size=4, shuffle=True)

		# for batch in loader:
		# 	break
		# print({k: v.shape for k, v in batch.items()})

		optim = AdamW(self.model.parameters(), lr=3e-4)

		num_epochs = epochs
		start_epoch = 0

		device = torch.device('cuda')
		self.model.to(device)

		# checkpoint = torch.load('./models/checkpoint5_am.pth.tar', map_location='cpu')
		# model.load_state_dict(checkpoint['model_state_dict'])
		# optim.load_state_dict(checkpoint['optimizer_state_dict'])
		# last_epoch = checkpoint['epoch']
		# loss = checkpoint['loss']


		print(next(self.model.parameters()).is_cuda) # returns a boolean


		self.model.train()

		start_epoch=0
		for epoch in range(start_epoch, num_epochs):
			loop = tqdm(loader)
			for batch in loop:

				# push the batch to gpu
					batch = [r.to(device) for r in batch]
					# print(batch)

					paragraphs, mask, labels = batch

				# paragraphs = batch['input_ids'].to(device)
				# labels = batch['labels'].to(device)
				# paragraphs_am = batch['input_attention_mask'].to(device)

					output = self.model(input_ids=paragraphs, labels=labels, attention_mask=mask)

					loss = output.loss
					loss.backward()
					optim.step()
					optim.zero_grad()

			torch.save({
				'epoch': epoch,
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': optim.state_dict(),
				'loss': loss,
				}, './models/bert_class_checkpoint.pth.tar')
			print('saved checkpoint')
		self.model.save_pretrained(model_path)






########## Computing Results


	# Write output in result txt file

	@staticmethod
	def results(testQuerieNum, rankedDocs, resultFile, topn=100, entailment=False, thresh=None):

		if entailment: 
			topn=len(rankedDocs)
			sortedDocs = [(k, v) for k,v in rankedDocs if v >= thresh]
			if len(sortedDocs)==0: sortedDocs=[rankedDocs[0]]
			rankedDocs = sortedDocs

		for x in range(min(topn, len(rankedDocs))):
			rank = str(x + 1)
			docID, score = rankedDocs[x]
			resultFile.write(testQuerieNum + "\tQ0\t" + str(docID) +
				"\t" + rank + "\t" + str(score) + "\tmyRun\n")
			pass
		pass 






########### Evaluate


	#returns ranked by bm25 score dictionary with doc id as key and score as value
	@staticmethod
	def rankDocsBM25(self, testQuerie, documentLabels):
		# calculate scores
		doc_scores = self.bm25.get_scores(testQuerie)

		# assign scores to document label
		x=0
		listScores = []
		for sim in doc_scores:
			listScores.append((documentLabels[x], sim))
			x += 1

		# create list of tuples with first element as doc id and second as value
		rankedDocs = [(k, v) for k, v in sorted(listScores, key=lambda item: item[1], reverse=True)]
		return rankedDocs

	# Use BM25 score to measure entailement
	def bm25Entailment(self):
		results = open(self.resultFile, 'w+')
		totalCases = len(self.caseDataFrame)
		for caseNum in tqdm(range(totalCases)):
			# initialize bm25 with paragraphs
			self.bm25 = BM25(self.caseDataFrame['paragraphs_clean'][caseNum])
			# calculate scores
			rankedDocs = self.rankDocsBM25(self, self.caseDataFrame['entailed_fragment_clean'][caseNum], self.caseDataFrame['paragraph_names'][caseNum])

			# write results
			self.results(self.caseDataFrame['case_number'][caseNum], rankedDocs, results, topn=1)


		results.close()



	def EvaluateSimilarityT5(self, tokenizer_model, model_name):
		results = open(self.resultFile, 'w+')
		
		tokenizer = T5Tokenizer.from_pretrained(tokenizer_model)
		model = T5ForConditionalGeneration.from_pretrained(model_name)
		#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

		model.eval()

		device = torch.device('cuda')
		model.to(device)

		# checkpoint = torch.load(model_name, map_location='cpu')
		# model.load_state_dict(checkpoint['model_state_dict'])

		print(next(model.parameters()).is_cuda) # returns a boolean


		totalCases = len(self.caseDataFrame)

		case_completed = 0
		for caseNum in range(totalCases):
			similarity_scores = []
			print("Processing:", case_completed, "out of",totalCases,"cases")


			# parse through each sentence pair of dataset
			for i in tqdm(range(len(self.caseDataFrame['paragraphs'][caseNum]))):
				paragraph_pair = [[self.caseDataFrame["entailed_fragment"][caseNum], self.caseDataFrame['paragraphs'][caseNum][i]]]
				tokenized_paragraphs = tokenizer(paragraph_pair, padding=True, return_tensors="pt", truncation=True)

				input_ids = tokenized_paragraphs['input_ids'].to(device)
				mask = tokenized_paragraphs['attention_mask'].to(device)

				output = model.generate(input_ids, attention_mask=mask)

				prediction = tokenizer.decode(output[0])
				#print(prediction)

				entailment = "<pad> true</s>"
				exclusion = "<pad> false</s>"

				if(prediction==entailment):
					similarity_scores.append((self.caseDataFrame['paragraph_names'][caseNum][i], 1))
				elif(prediction==exclusion):
					similarity_scores.append((self.caseDataFrame['paragraph_names'][caseNum][i], 0))

			sortedDocs_unfiltered = [(k, v) for k, v in sorted(similarity_scores, key=lambda item: item[1], reverse=True)]
			self.results(self.caseDataFrame['case_number'][caseNum], sortedDocs_unfiltered, results, entailment=True, thresh=0.5)
			case_completed += 1
		results.close()
		 


	
	     

	def transformer_preprocess(self, model_name):

		#load model
		self.model = SentenceTransformer(model_name)

		# check if we can use GPU for training
		#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		#         torch.cuda.empty_cache()
		#         self.model.to(device)
		return self.model






	@staticmethod
	def FindSimilarityWithTransformer(self, entailed_fragment, paragraph):

		# create embeddings of our two sentences
		query_embedding = self.model.encode(entailed_fragment, normalize_embeddings=True)
		passage_embedding = self.model.encode(paragraph, normalize_embeddings=True)

		# find their cosine similarity by calculating dot score of their embeddings
		similarity = util.dot_score(query_embedding, passage_embedding)[0][0]

		return similarity.item()



	# evaluate similarity for sentences with given model
	def EvaluateSimilaritySBERT(self,thresh=0.7, model_name="./models/bert_base_model_test"):
		results = open(self.resultFile, 'w+')
		
		self.model = self.transformer_preprocess(model_name)

		# parse through our 5 datasets of sentence pairs
		totalCases = len(self.caseDataFrame)
		case_completed = 0
		for caseNum in tqdm(range(totalCases)):

			similarity_scores = []
			print("Processing:", case_completed, "out of",totalCases,"cases")

			# parse through each sentence pair of dataset
			for i in tqdm(range(len(self.caseDataFrame['paragraphs'][caseNum]))):
				# calculate their cosine similarity
				similarity_scores.append(( self.caseDataFrame['paragraph_names'][caseNum][i],
											self.FindSimilarityWithTransformer( self, self.caseDataFrame['entailed_fragment'][caseNum],
																				self.caseDataFrame['paragraphs'][caseNum][i] ) ))

			sortedDocs_unfiltered = [(k, v) for k, v in sorted(similarity_scores, key=lambda item: item[1], reverse=True)]

			self.results(self.caseDataFrame['case_number'][caseNum], sortedDocs_unfiltered, results, entailment=True,thresh=thresh)
			case_completed+=1
		results.close()
	

	# evaluate similarity for paragraphs with given model
	def EvaluateSimilarityBERTClassification(self,model_path, model_name,thresh=0.5):
		results = open(self.resultFile, 'w+')
		
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_path)


		self.model.eval()

		device = torch.device('cuda')
		self.model.to(device)

		# checkpoint = torch.load(model_name, map_location='cpu')
		# model.load_state_dict(checkpoint['model_state_dict'])

		print(next(self.model.parameters()).is_cuda) # returns a boolean


		totalCases = len(self.caseDataFrame)

		case_completed = 0
		for caseNum in range(totalCases):
			similarity_scores = []
			print("Processing:", case_completed, "out of",totalCases,"cases")


			# parse through each sentence pair of dataset
			for i in tqdm(range(len(self.caseDataFrame['paragraphs'][caseNum]))):
				paragraph_pair = [[self.caseDataFrame["entailed_fragment"][caseNum], self.caseDataFrame['paragraphs'][caseNum][i]]]
				
				tokenized_paragraphs = tokenizer(paragraph_pair, padding=True, return_tensors="pt", truncation=True)
				input_ids = tokenized_paragraphs['input_ids'].to(device)
				mask = tokenized_paragraphs['attention_mask'].to(device)

				# get predictions for test data
				# with torch.no_grad():
				# 	output = self.model(input_ids, attention_mask=mask)
				# 	output = output.detach().cpu().numpy()

				output = self.model(input_ids, attention_mask=mask)
		
		
				probs = torch.nn.functional.softmax(output[0], dim=-1)
				probs = probs.mean(dim=0)
				pred = torch.argmax(probs).item()
				# print("probs:",probs,"; preds:",pred)
 

				if(pred==1):
					similarity_scores.append((self.caseDataFrame['paragraph_names'][caseNum][i], 1))
				elif(pred==0):
					similarity_scores.append((self.caseDataFrame['paragraph_names'][caseNum][i], 0))

			sortedDocs_unfiltered = [(k, v) for k, v in sorted(similarity_scores, key=lambda item: item[1], reverse=True)]
			self.results(self.caseDataFrame['case_number'][caseNum], sortedDocs_unfiltered, results, entailment=True, thresh=0.5)
			case_completed += 1
		results.close()




entailment_model = caseEntailment('COLIEE2021')

# run only once!
# entailment_model.createDataFrames()

entailment_model.preProcess()


# run only once!

#### For sBert

## original unbalanced data
# sentences, labels = entailment_model.preProcessBERT(train=True)

## undersample the negative examples
# sentences, labels = entailment_model.preProcessBERT(train=True,undersample=True)

## oversample the positive examples
# sentences, labels = entailment_model.preProcessBERT(train=True,oversample=True)


#### For T5

## original unbalanced data
# sentences, labels = entailment_model.preProcessT5(train=True)

## undersample the negative examples
# sentences, labels = entailment_model.preProcessT5(train=True,undersample=True)

## oversample the positive examples
# sentences, labels = entailment_model.preProcessT5(train=True,oversample=True)



###### TRAIN

# tokenizer_model="t5-base"
# model_in="bert-base-uncased"
# model_out"./models/2021/bert_base_5epochs"


#### For sBert

## train with original unbalanced data
# entailment_model.TrainBERT(model_name=model_in, epochs=3, model_path=model_out)

## train with oversampled data
# entailment_model.TrainBERT(model_name=model_in, epochs=3, model_path=model_out,oversample=True)

## train with undersampled data
# entailment_model.TrainBERT(model_name=model_in, epochs=10, model_path=model_out,undersample=True)


#### For T5

# change the path for unbalanced, oversampled or undersampled data inside the function
# entailment_model.TrainT5(model_name=model_in, epochs=3)





######## EVALUATE

#### For sBert
# entailment_model.EvaluateSimilaritySBERT(model_name=model_out)

#### For T5
# entailment_model.EvaluateSimilarityT5(tokenizer_model, model_name=model_out)


# entailment_model.calculateF1(entailment_model.resultFile, entailment_model.test_labels)



