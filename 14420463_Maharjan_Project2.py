import os
from pathlib import Path
import numpy as np
import math
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

project_dir = Path(__file__).resolve().parent
dataset_root = project_dir

class_name_list=['20_newsgroups/alt.atheism', '20_newsgroups/comp.graphics', '20_newsgroups/comp.os.ms-windows.misc', \
            '20_newsgroups/comp.sys.ibm.pc.hardware','20_newsgroups/comp.sys.mac.hardware',  '20_newsgroups/comp.windows.x', \
            '20_newsgroups/misc.forsale','20_newsgroups/rec.autos', '20_newsgroups/rec.motorcycles',\
             '20_newsgroups/rec.sport.baseball', '20_newsgroups/rec.sport.hockey', '20_newsgroups/sci.crypt', '20_newsgroups/sci.electronics',\
             '20_newsgroups/sci.med', '20_newsgroups/sci.space','20_newsgroups/soc.religion.christian', \
            '20_newsgroups/talk.politics.guns', '20_newsgroups/talk.politics.mideast', '20_newsgroups/talk.politics.misc', \
            '20_newsgroups/talk.religion.misc']
words_to_ignore = [
    # Standard English stopwords
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
    "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', 
    "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 
    'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 
    'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 
    'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 
    'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 
    'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 
    "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 
    'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", 
    "you've", 'your', 'yours', 'yourself', 'yourselves',
    'subject', 're', 'edu', 'use', 'com', 'writes', 'article', 'organization', 'lines', 'sender', 'message',
    'host', 'nntp', 'posting', 'reply', 'distribution', 'references', 'university', 'news', 'group', 
    'posting', 'path', 'gmt', 'date', 'email', 'address', 'contact', 'phone', 'fax', 'http', 'https', 'ftp', 
    'archive', 'info', 'mail', 'list', 'email', 'reply', 'thanks', 'regards', 'dear', 'mr', 'ms', 'dr',
    'would', 'could', 'should', 'also', 'etc', 'one', 'two', 'three', 'may', 'might', 'said', 'say', 'see', 
    'like', 'get', 'got', 'make', 'made', 'even', 'still', 'many', 'much', 'way', 'use', 'used', 'using', 
    'want', 'need', 'know', 'think', 'sure', 'well', 'good', 'bad', 'better', 'best', 'yes', 'no', 'ok',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '000', 'etc', 'vs', 'etc.', 'na', 'none',
    'writes:', 'posting:', 'organization:', 'subject:', 'article:', 'summary:', 'keywords:', 'header:', 
    'reply-to:', 'distribution:', 'message-id:', 'newsgroups:', 'version', 'server', 'system', 'file', 
    'software', 'windows', 'dos', 'mac', 'computer', 'data', 'program', 'code', 'run', 'output', 'input',
    'ago', 'almost', 'along', 'already', 'always', 'another', 'anyone', 'anything', 'anyway', 'anywhere',
    'around', 'ask', 'asking', 'away', 'back', 'based', 'become', 'becomes', 'began', 'behind', 'below',
    'beside', 'besides', 'beyond', 'bit', 'board', 'bottom', 'bring', 'brought', 'building', 'call', 'came',
    'case', 'certain', 'certainly', 'change', 'clear', 'close', 'coming', 'common', 'completely', 'consider',
    'control', 'course', 'current', 'data', 'day', 'days', 'deal', 'definitely', 'different', 'discuss',
    'doesn', 'done', 'drive', 'during', 'early', 'either', 'else', 'enough', 'especially', 'every', 'everyone',
    'everything', 'everywhere', 'example', 'fact', 'far', 'few', 'finally', 'follow', 'following', 'form',
    'found', 'free', 'front', 'full', 'general', 'getting', 'give', 'given', 'giving', 'goes', 'gone', 'great',
    'group', 'guess', 'half', 'hard', 'having', 'high', 'however', 'include', 'including', 'instead', 'keep',
    'kept', 'kind', 'kinds', 'knew', 'known', 'large', 'later', 'least', 'less', 'let', 'likely', 'long',
    'look', 'looked', 'looking', 'lot', 'low', 'main', 'major', 'makes', 'making', 'matter', 'maybe', 'mean',
    'means', 'member', 'mention', 'might', 'mind', 'minute', 'moment', 'move', 'much', 'must', 'near', 'nearly',
    'need', 'needed', 'never', 'next', 'none', 'normal', 'nothing', 'notice', 'often', 'old', 'once', 'open',
    'part', 'particular', 'perhaps', 'place', 'point', 'possible', 'power', 'pretty', 'probably', 'problem',
    'put', 'question', 'quite', 'rather', 'read', 'real', 'really', 'reason', 'receive', 'recent', 'recently',
    'related', 'remember', 'rest', 'result', 'run', 'said', 'seem', 'seemed', 'seems', 'sense', 'sent', 'set',
    'short', 'show', 'shown', 'side', 'since', 'small', 'sort', 'start', 'state', 'stop', 'system', 'taken',
    'taking', 'talk', 'tell', 'term', 'thing', 'things', 'think', 'though', 'thought', 'took', 'top', 'try',
    'trying', 'type', 'used', 'value', 'various', 'version', 'want', 'wanted', 'way', 'went', 'whole', 'without',
    'word', 'words', 'work', 'working', 'world', 'write', 'written', 'wrong', 'year', 'years', 'yes', 'yet',
    'zero', 'ftp', 'org', 'url', 'new', 'time', 'home', 'thanks', 'ok', 'okay', 'oh', 'yeah', 'hey', 'via',
    'able', 'access', 'according', 'across', 'add', 'added', 'addition', 'ago', 'agree', 'ahead', 'allow',
    'almost', 'along', 'already', 'amount', 'answer', 'application', 'area', 'asked', 'associated', 'available',
    'based', 'basic', 'became', 'begin', 'begins', 'behind', 'believe', 'below', 'beyond', 'bring', 'broken',
    'build', 'built', 'busy', 'buy', 'byte', 'california', 'called', 'calling', 'cannot', 'cause', 'center',
    'check', 'choice', 'city', 'class', 'clear', 'code', 'comment', 'complete', 'connection', 'considered',
    'contact', 'continue', 'copy', 'correct', 'cost', 'country', 'current', 'cut', 'data', 'deal', 'default',
    'design', 'details', 'difference', 'directory', 'disk', 'display', 'document', 'drive', 'early', 'easy',
    'edit', 'either', 'enough', 'enter', 'error', 'event', 'exactly', 'example', 'except', 'exists', 'expect',
    'experience', 'explain', 'express', 'fact', 'fairly', 'fast', 'file', 'find', 'finished', 'follow', 'format',
    'forward', 'found', 'function', 'future', 'getting', 'given', 'global', 'gone', 'guess', 'handle', 'happen',
    'hardware', 'held', 'help', 'history', 'hold', 'hope', 'idea', 'image', 'important', 'improve', 'included',
    'including', 'increase', 'information', 'inside', 'install', 'instead', 'interesting', 'interface', 'internet',
    'issue', 'item', 'john', 'join', 'keep', 'key', 'kind', 'known', 'lack', 'large', 'later', 'latest', 'leave',
    'left', 'level', 'life', 'light', 'line', 'list', 'local', 'location', 'longer', 'machine', 'mainly', 'maintain',
    'major', 'making', 'manage', 'manager', 'manual', 'mark', 'maybe', 'means', 'memory', 'mention', 'menu', 'method',
    'microsoft', 'middle', 'model', 'moment', 'month', 'mostly', 'move', 'multiple', 'natural', 'near', 'necessary',
    'network', 'normal', 'note', 'notice', 'number', 'object', 'office', 'online', 'open', 'operation', 'option',
    'order', 'original', 'output', 'outside', 'page', 'paper', 'parameter', 'particular', 'passed', 'past', 'path',
    'perform', 'performance', 'perhaps', 'person', 'physical', 'place', 'point', 'policy', 'possible', 'post',
    'powerful', 'practice', 'prepare', 'present', 'press', 'pretty', 'previous', 'probably', 'problem', 'procedure',
    'process', 'produce', 'product', 'programming', 'proper', 'provide', 'purpose', 'quality', 'question', 'quick',
    'quite', 'range', 'rate', 'rather', 'read', 'ready', 'reason', 'recent', 'record', 'reference', 'region',
    'regular', 'related', 'release', 'remain', 'remove', 'replace', 'report', 'request', 'require', 'required',
    'research', 'resource', 'response', 'result', 'return', 'right', 'role', 'run', 'save', 'screen', 'search',
    'section', 'security', 'select', 'send', 'sense', 'server', 'service', 'set', 'share', 'short', 'show',
    'signal', 'similar', 'simple', 'simply', 'single', 'site', 'situation', 'software', 'source', 'space',
    'special', 'specific', 'standard', 'start', 'state', 'statement', 'step', 'stop', 'storage', 'store',
    'strong', 'structure', 'stuff', 'subject', 'support', 'system', 'take', 'taken', 'task', 'technical',
    'technology', 'tell', 'term', 'test', 'text', 'thing', 'think', 'thread', 'time', 'told', 'tool', 'topic',
    'total', 'training', 'type', 'unit', 'university', 'update', 'useful', 'user', 'using', 'usually', 'value',
    'version', 'view', 'wait', 'wanted', 'way', 'window', 'within', 'without', 'word', 'work', 'write', 'written'
]
def count_words_in_file(file_path, 
                        words_to_ignore,
                        class_name,
                        word_count,
                        total_emails):
    total_words_in_class=0
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    half_count = math.ceil(len(files) * 0.5)
    total_emails+=half_count

    for filename in files[:half_count]:
        file_path_full = file_path / filename
        with open(file_path_full, 'r', errors='ignore') as file:
            text = file.read().lower()
            body = text.split('\n\n', 1)[-1].lower()
            words_in_email = body.split() 
            for word in words_in_email:
                if word not in words_to_ignore:
                    word_count[class_name][word] = word_count[class_name].get(word, 0) + 1
                    if word not in word_count['unique_words']:
                        word_count['unique_words'].append(word)
                    total_words_in_class+=1

        word_count[class_name+'word_total_in_class'] = total_words_in_class
        print(f"Processed file: {filename} in class: {class_name}")
    return word_count,total_emails

def likelihood_estimation(class_name, 
                          word_count,
                          alpha):
    likelihoods = {}
    unique_word_total=len(word_count['unique_words'])
    total_words=word_count[class_name+'word_total_in_class']

    for word in word_count['unique_words']:
        count = word_count[class_name].get(word, 0)
        likelihoods[word] = (count + alpha) / (total_words + unique_word_total*alpha)
    likelihoods["unseen"] = alpha / (total_words + unique_word_total*alpha) 
    return likelihoods

def prior_estimation(file_path,total_emails):
    num_emails_in_class = 0
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    num_emails_in_class = len(files)*0.5
    return num_emails_in_class / total_emails

def classify_email(email_text, likelihood_library, prior_library, class_name_list):
      prob_word_given_class = 1
      body = email_text.split('\n\n', 1)[-1].lower()
      email_words = body.split()
      class_probabilities = {}
      class_probabilities={class_name: 1 for class_name in class_name_list}
      for class_name in class_name_list:
            for word in email_words:
                  if word not in likelihood_library[class_name]:
                        prob_word_given_class = likelihood_library[class_name]["unseen"]
                  else:
                        prob_word_given_class = likelihood_library[class_name][word]
                  class_probabilities[class_name]=class_probabilities[class_name] * prob_word_given_class
            class_probabilities[class_name] *= prior_library[class_name]
      predicted_class = max(class_probabilities, key=class_probabilities.get)
      print(class_probabilities)
      return predicted_class

def classify_email_log_method(email_text, likelihood_library, prior_library, class_name_list):
      body = email_text.split('\n\n', 1)[-1].lower()
      email_words = body.split()
      class_probabilities = {}
      for class_name in class_name_list:
            log_prob = math.log(prior_library[class_name]) if prior_library[class_name] > 0 else float('-inf')
            unseen_prob = likelihood_library[class_name]["unseen"]
            
            for word in email_words:
                  prob_word_given_class = likelihood_library[class_name].get(word, unseen_prob)
                  log_prob += math.log(prob_word_given_class)
            class_probabilities[class_name] = log_prob
      predicted_class = max(class_probabilities, key=class_probabilities.get)
      return predicted_class

compiled_list={}
word_count={}
word_count = defaultdict(lambda: defaultdict(int))
word_count['unique_words']=[]
total_emails=0  

for class_name in class_name_list:
    total_words=0
    word_count,total_emails=count_words_in_file(project_dir / class_name, 
                                                            words_to_ignore,
                                                            class_name,
                                                            word_count,
                                                            total_emails)
    
likelihood_library={}
prior_library={}
for class_name in class_name_list:
    likelihood_library[class_name]=likelihood_estimation(class_name,
                                                         word_count,
                                                         1)
    prior_library[class_name]=prior_estimation(project_dir / class_name,total_emails)

with open(r'.\20_newsgroups\comp.graphics\37919', 
          'r', 
          errors='ignore') as file:
    email_text = file.read()
    predicted_class = classify_email(email_text, likelihood_library, prior_library, class_name_list)
    print(f"The predicted class for the email using the normal Naive Bayes is: {predicted_class}")
    predicted_class_log = classify_email_log_method(email_text, likelihood_library, prior_library, class_name_list)
    print(f"The predicted class for the email using the log method Naive Bayes is: {predicted_class_log}") 

df_results = pd.DataFrame(columns=['Filename', 'Actual_Class', 'Predicted_Class_Log_Method'])
results=[]
for class_name in class_name_list:
    file_path = project_dir / class_name
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    half_count = math.ceil(len(files) * 0.5)
    for filename in files[half_count:]:
        file_path_full = file_path / filename
        with open(file_path_full, 'r') as file:
            email_text = file.read()
            predicted_class_log = classify_email_log_method(email_text, likelihood_library, prior_library, class_name_list)
            results.append({'Filename': filename,
                            'Actual_Class': class_name,
                            'Predicted_Class_Log_Method': predicted_class_log})

df_results=pd.DataFrame(results)
report= classification_report(df_results['Actual_Class'], 
                               df_results['Predicted_Class_Log_Method'], 
                               output_dict=True)
accuracy =accuracy_score(df_results['Actual_Class'], df_results['Predicted_Class_Log_Method'])

print(f"Overall Accuracy: {accuracy*100:.2f}%")
print("Per-Class Performance:")
print("-" * 60)
for class_name in report:
    if class_name not in ['accuracy']:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        print(f"{class_name:35} | "
              f"Precision: {precision*100:6.2f}% | "
              f"Recall: {recall*100:6.2f}% | "
              f"F1-Score: {f1*100:6.2f}% | "
              f"Samples: {support}")