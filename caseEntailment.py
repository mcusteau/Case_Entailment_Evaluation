import os
import pandas as pd
import swifter
import spacy
from spacy.lang.en import English
import pickle
from tqdm import tqdm
import nltk
import string
import json
import torch

from rank_bm25 import BM25Okapi as BM25
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, AdamW, get_scheduler

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
			# in case we ever need to use a server that has a lower python version, not in use currently
		    pass
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
	def calculateRecall(self, results, labels, topn):
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

	        for i in range(min(topn, len(retreived))):
	            if(retreived[i] in relevant_docs):
	                rel_num+=1

	    recall = rel_num/rel_cases
	    print("recall at top "+str(topn)+" found is", recall)
	    return recall


	@staticmethod
	def calculatePrecision(self, results, labels, topn):
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
		
	        for i in range(min(topn, len(retreived))):
	            retreived_cases +=1
	            if(retreived[i] in relevant_docs):
	                rel_num+=1

	    # retreived_cases = num_queries*topn
	    precision = rel_num/retreived_cases
	    print("precision at top "+str(topn)+" found is", precision)
	    return precision



	def calculateF1(self, results, labels, topn):
	    recall = self.calculateRecall(self, results, labels, topn)
	    precision = self.calculatePrecision(self, results, labels, topn)
	    f1 = (2*precision*recall)/(precision+recall)
	    print("f1 at top "+str(topn)+" found is", f1)
	    return f1


########## Training

	def trainT5(self, paragraph_pairs, labels):

		tokenizer = T5Tokenizer.from_pretrained("t5-small")
		model = T5ForConditionalGeneration.from_pretrained("t5-small")
		#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

		tokenized_paragraphs = tokenizer(paragraph_pairs, padding=True, return_tensors="pt", truncation=True)
		tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt")

		train_data = ForT5Dataset(tokenized_paragraphs, tokenized_labels)

		loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

		for batch in loader:
			break
		print({k: v.shape for k, v in batch.items()})

		optim = AdamW(model.parameters(), lr=5e-5)

		num_epochs = 2
		start_epoch = 0

		num_training_steps = num_epochs * len(loader)
		lr_scheduler = get_scheduler(
		    "linear",
		    optimizer=optim,
		    num_warmup_steps=0,
		    num_training_steps=num_training_steps,
		)


		device = torch.device('cuda')
		model.to(device)

		print(next(model.parameters()).is_cuda) # returns a boolean


		model.train()

		start_epoch=0
		for epoch in range(start_epoch, num_epochs):
			loop = tqdm(loader)
			for batch in loop:

				paragraphs = batch['input_ids'].to(device)
				labels = batch['labels'].to(device)
				paragraphs_am = batch['input_attention_mask'].to(device)
				labels = batch['labels'].to(device)
				labels_am = batch['labels_attention_mask'].to(device)
				output = model(input_ids=paragraphs, labels=labels, attention_mask=paragraphs_am, decoder_attention_mask=labels_am)

				loss = output.loss
				loss.backward()
				optim.step()
				lr_scheduler.step()
				optim.zero_grad()
			torch.save({
		    'epoch': epoch,
		    'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optim.state_dict(),
		    'scheduler_state_dict': lr_scheduler.state_dict(),
		    'loss': loss,
		    }, './models/checkpoint'+str(epoch)+'_am.pth.tar')
			print('saved checkpoint')
		model.save_pretrained('./models/t5_am')


	def preProcessT5(self, train=True):

		if(train):
			case_df = self.caseDataFrame_train
			label_file = self.train_labels
		else:
			case_df = self.caseDataFrame
			label_file = self.test_labels

		lf = open(label_file, "r")
		labels_json = json.load(lf)

		sentences = []
		labels = []

		totalCases = len(case_df)
		for caseNum in tqdm(range(totalCases)):
			case_number = case_df['case_number'][caseNum]
			relevant_paragraphs = [i.replace(".txt", "") for i in labels_json[case_number]]
			for i in range(len(case_df['paragraphs'][caseNum])):
				sentences.append("[CLS]"+case_df["entailed_fragment"][caseNum]+"[SEP]"+case_df['paragraphs'][caseNum][i]+"[SEP]")
				if(case_df['paragraph_names'][caseNum][i] in relevant_paragraphs):
					labels.append("Entailment")
				else:
					labels.append("Exclusion")

		return sentences, labels


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

    # # Write output in result txt file
    # @staticmethod
    # def relevantResults(testQuerieNum, rankedDocs, resultFile, topn=100,relevant=True):

    #     for x in range( len(rankedDocs)):
    #         rank = str(x + 1)
    #         docID, score = rankedDocs[x]
    #         resultFile.write(testQuerieNum + "\tQ0\t" + str(docID) +
    #                 "\t" + rank + "\t" + str(score) + "\tmyRun\n")
    #         pass
    #     pass 


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
	        self.results(self.caseDataFrame['case_number'][caseNum], rankedDocs, results, topn=5)

	    results.close()


	def EvaluateSimilarityT5(self, paragraph_pairs, labels, model_name):

		
		tokenizer = T5Tokenizer.from_pretrained("t5-small")
		model = T5ForConditionalGeneration.from_pretrained(model_name)
		#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

		model.eval()

		device = torch.device('cuda')
		model.to(device)

		print(next(model.parameters()).is_cuda) # returns a boolean


		totalCases = len(self.caseDataFrame)
	    case_completed = 0
	    for caseNum in tqdm(range(totalCases)):
	    	similarity_scores = []
	        print("Processing:", case_completed, "out of",totalCases,"cases")

	        # parse through each sentence pair of dataset
	        for i in tqdm(range(len(self.caseDataFrame['paragraphs'][caseNum]))):
	        	paragraph_pair = "[CLS]"+self.caseDataFrame["entailed_fragment"][caseNum]+"[SEP]"+self.caseDataFrame['paragraphs'][caseNum][i]+"[SEP]"
	        	tokenized_paragraphs = tokenizer(paragraph_pair, padding=True, return_tensors="pt", truncation=True)
				
				output = model(tokenized_paragraphs['input_ids'], attention_mask=tokenized_paragraphs['attention_mask'])
				
				prediction = tokenizer.decode(output)

				
	     
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
	def EvaluateSimilaritySBERT(self,thresh=0.7):
	    results = open(self.resultFile, 'w+')
	    self.model = self.transformer_preprocess("usc-isi/sbert-roberta-large-anli-mnli-snli")

	    # parse through our 5 datasets of sentence pairs
	    totalCases = len(self.caseDataFrame)
	    case_completed = 0
	    for caseNum in tqdm(range(totalCases)):

	        similarity_scores = []
	        print("Processing:", case_completed, "out of",totalCases,"cases")

	        # parse through each sentence pair of dataset
	        for i in tqdm(range(len(self.caseDataFrame['paragraphs'][caseNum]))):
	            # calculate their cosine similarity
	            similarity_scores.append((self.caseDataFrame['paragraph_names'][caseNum][i],self.FindSimilarityWithTransformer(self.caseDataFrame['entailed_fragment'][caseNum], self.caseDataFrame['paragraphs'][caseNum][i])))

	        sortedDocs_unfiltered = [(k, v) for k, v in sorted(similarity_scores, key=lambda item: item[1], reverse=True)]
	        
	        self.results(self.caseDataFrame['case_number'][caseNum], sortedDocs_unfiltered, results, entailment=True,thresh=thresh)
	        case_completed+=1

	    results.close()
	    
	    





entailment_model = caseEntailment('COLIEE2021')
entailment_model.preProcess()
# # # entailment_model.bm25Entailment()
sentences, labels = entailment_model.preProcessT5(train=True)
entailment_model.trainT5(sentences, labels)
# sentences, labels = entailment_model.preProcessT5(train=False)
# entailment_model.EvaluateSimilarityT5(sentences, labels, "./models/t5")

        
   


entailment_model = caseEntailment('COLIEE2021')
entailment_model.preProcess()
# entailment_model.bm25Entailment()
entailment_model.EvaluateSimilaritySBERT()
entailment_model.calculateF1(entailment_model.resultFile, entailment_model.test_labels, 5)
