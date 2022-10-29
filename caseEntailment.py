import os
import pandas as pd
import swifter
import spacy
from spacy.lang.en import English
import pickle
from tqdm import tqdm
import nltk
import string

from rank_bm25 import BM25Okapi as BM25

nlp = English()

wn = nltk.WordNetLemmatizer()

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
			l = []
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
			relevant_docs = labels_json[label]
			if(type(relevant_docs) is str):
				relevant_docs = [relevant_docs]
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
		rel_num = 0
		num_queries = 0
		for label in labels_json.keys():
			num_queries +=1
			relevant_docs = labels_json[label]
			if(type(relevant_docs) is str):
				relevant_docs = [relevant_docs]
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

		retreived_cases = num_queries*topn
		precision = rel_num/retreived_cases
		print("precision at top "+str(topn)+" found is", precision)
		return precision



	def calculateF1(self, results, labels, topn):
		recall = self.calculateRecall(self, results, labels, topn)
		precision = self.calculatePrecision(self, results, labels, topn)
		f1 = (2*precision*recall)/(precision+recall)
		print("f1 at top "+str(topn)+" found is", f1)
		return f1


########## Computation


	# Write output in result txt file
	@staticmethod
	def results(testQuerieNum, rankedDocs, resultFile, topn=100):

		for x in range(min(topn, len(rankedDocs))):
			rank = str(x + 1)
			docID, score = rankedDocs[x]
			resultFile.write(testQuerieNum + "\tQ0\t" + str(docID) +
			        "\t" + rank + "\t" + str(score) + "\tmyRun\n")
			pass
		pass 




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

entailment_model = caseEntailment('COLIEE2022')
entailment_model.createDataFrames()
# ce.preProcess()
# ce.bm25Entailment()
# ce.calculateF1(ce.resultFile, ce.test_labels, 1)