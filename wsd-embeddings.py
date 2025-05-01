"""
Performs WSD for the word "line" using GloVe word embeddings
as features. Implements SVM with scikit and a Neural Network with PyTorch.

Usage
    python wsd-embeddings.py <train_file> <test_file> <embedding_file> [SVM|NN] > my-line-answers.txt

Models used
    SVM and NN (default is SVM)
    SVM - Scikit-learn SVC with a linear kernel
    NN - PyTorch Neural Network with one hidden layer

Results
    SVM Accuracy: 90.48%
            phone product
    phone   63     9
    product 3      51

    NN Accuracy: 88.89%
            phone product
    phone   62     10
    product 4      50

    Both models performed better than the most frequent sense baseline accuracy of 80.16%, while the SVM model performed slightly better than the NN model.
    Both models also performed better than the Linear SVC (84.13%) and Decision Tree (74.60%) classifiers from the previous assignment, 
    however the Multinomial Naive Bayes classifier from the previous assignment performed better than both models (92.86%).

"""

import sys
import re
from lxml import etree
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_context_tokens(context_elem):
    """
    Extracts text from context element and tokenizes it, and convert to lowercase.
    Removes 'line and 'lines'.
    """
    context_text = " ".join(context_elem.itertext())

    tokens = re.findall(r'\w+', context_text.lower())
    #remove the target word 'line' and its plural form
    return [token for token in tokens if token not in ["line", "lines"]]


def parse_train(filename):
    """
    Returns:
    - list of instance identifiers.
    - list of lists of context tokens.
    - list of gold sense labels.
    """
    instance_ids = []
    context_token_lists = []
    labels = []
    try:
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        tree = etree.parse(filename, parser)
        root = tree.getroot()

        #find instance elements
        for instance in root.xpath('.//instance'):
            instance_id = instance.get('id')
            if not instance_id:
                continue

            #locate answer elements and extract its senseid
            answer_elem = instance.find('answer')
            #check if its an answer element or senseid attribute
            if answer_elem is None or answer_elem.get('senseid') is None:
                 continue #skip if theres none
            label = answer_elem.get('senseid')

            #find the context element
            context_elem = instance.find('context')
            if context_elem is None:
                 continue #skip if theres no context tag

            #get processed tokens from the context
            context_tokens = get_context_tokens(context_elem)
            context_token_lists.append(context_tokens)

            instance_ids.append(instance_id)
            labels.append(label)

    except Exception as e:
        print(f"An unexpected error occurred while parsing the training file: {e}", file=sys.stderr)
        sys.exit(1)

    return instance_ids, context_token_lists, labels


def parse_test(filename):
    """
    Parses the test XML file using lxml.
    Returns:
    -list of instance ids.
    -list of lists of context tokens.
    """
    instance_ids = []
    context_token_lists = []
    try:
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        tree = etree.parse(filename, parser)
        root = tree.getroot()

        for instance in root.xpath('.//instance'):
            instance_id = instance.get('id')
            if not instance_id:
                 continue

            context_elem = instance.find('context')
            if context_elem is None:
                 continue

            context_tokens = get_context_tokens(context_elem)
            context_token_lists.append(context_tokens)
            instance_ids.append(instance_id)

    except Exception as e:
        print(f"An unexpected error occurred while parsing the testfile: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(instance_ids)} test instances.", file=sys.stderr)
    return instance_ids, context_token_lists

def processEmbeddings(embedding_file):
    """
    Loads GloVe word embeddings from a file into a dictionary.
    """
    embeddings = {}
    count = 0
    skippedLines= 0
    embedding_dim = 0

    try:
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                count += 1
                values = line.strip().split()

                #each line should contain a word and at least one dimension
                if len(values) < 2:
                    skippedLines += 1
                    continue

                word = values[0] #sets the first value as the word
                try:
                    vector = np.array(values[1:], "float32")
                    if embedding_dim == 0:
                        #initialize embedding_dim with the first valid vector's length
                        embedding_dim = len(vector)
                    #skips words with wrong dimensions
                    elif embedding_dim != len(vector):
                        skippedLines += 1
                        continue

                    embeddings[word] = vector #sets the vectors as the value of the words

                except ValueError:
                    skippedLines += 1
                    continue

    except FileNotFoundError:
         print(f"File not found.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"Error reading file {embedding_file}", file=sys.stderr)
         sys.exit(1)


    count = len(embeddings)
    return embeddings, embedding_dim

def feature_extraction(context_token_lists, embeddings, dimension):
    """
    Creates the feature vectors by averaging word embeddings for each context.
    """
    feature_vectors = []
    for context_token_list in context_token_lists:
        #list of vectors for each instance
        instance_vectors = []
        for context_token in context_token_list:
            vector = embeddings.get(context_token)
            if vector is not None:
                instance_vectors.append(vector)

        if instance_vectors: #if the list is not empty
            #find the average of the vectors for the context tokens
            feature_vector = np.mean(instance_vectors, axis=0)
        else:
            #if no vectors were found, create a zero vector
            feature_vector = np.zeros(dimension, dtype=np.float32)

        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)

#neural network model with PyTorch
class WSDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WSDNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        return out

def train_NNmodel(train_loader, input_dim, hidden_dim, output_dim, num_epochs, learning_rate):
    model = WSDNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train() #training mode

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        processed_samples = 0 #calculate average loss 
        for i, (features, labels) in enumerate(train_loader):
            #forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)
            processed_samples += features.size(0)

        if processed_samples > 0:
             avg_epoch_loss = epoch_loss / processed_samples
             print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}', file=sys.stderr)
        else:
             print(f'Epoch [{epoch+1}/{num_epochs}], No samples processed.', file=sys.stderr)


    #print("Training completed", file=sys.stderr)
    return model

def main():

    #train file, test file, embedding file.
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    embedding_file = sys.argv[3]

    #model choice, the default is SVM
    model_choice = 'SVM'
    if len(sys.argv) > 4:
        model_arg = sys.argv[4].upper()
        if model_arg in ['SVM', 'NN']:
            model_choice = model_arg

    #processes embeddings, returns a dictionary of words and their vectors, and the dimension of the vectors.
    embeddings, dimension = processEmbeddings(embedding_file)

    #parse the training and test data.
    train_ids, train_context_tokens, train_labels = parse_train(train_file)
    test_ids, test_context_tokens = parse_test(test_file)

    #extract training and test feature vectors using the embeddings.
    x_train = feature_extraction(train_context_tokens, embeddings, dimension)
    x_test = feature_extraction(test_context_tokens, embeddings, dimension)

    #get training labels
    y_train = train_labels

    #model training process

    trainedmodel = None

    #label encoder
    label_encoder = LabelEncoder()

    if model_choice.lower() == "svm":
        #convert labels to numpy array for the SVM model
        svm_y_train = np.array(y_train)

        #train model with SVC
        trainedmodel = SVC(kernel='linear', probability=True, random_state=42)
        trainedmodel.fit(x_train, svm_y_train)

    elif model_choice.lower() == "nn": #
        #converts training and testing data into tensors for the NN model
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

        #encode labels using label encoder
        y_train_strings = train_labels
        y_train_encoded = label_encoder.fit_transform(y_train_strings)
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)


        batch_size = 32
        #create the dataloader for training data
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #define the parameters
        hidden_dim = 64
        output_dim = len(label_encoder.classes_)
        num_epochs = 15
        learning_rate = 0.001

        #train the model
        trainedmodel = train_NNmodel(train_loader, dimension, hidden_dim, output_dim, num_epochs, learning_rate)

    #predict senses for the test data using the trained model
    predictions = []
    if model_choice.upper() == "SVM":
        raw_predictions = trainedmodel.predict(x_test)
        predictions = list(raw_predictions) #convert predictions to a list

    elif model_choice.upper() == "NN":
        trainedmodel.eval() #evaluation mode
        with torch.no_grad():
            
            outputs = trainedmodel(x_test_tensor)
            predicted_indices = torch.argmax(outputs, dim=1)
            #convert predictions into string labels
            predictions = label_encoder.inverse_transform(predicted_indices.cpu().numpy())

    #print answers to STDOUT with redirect
    for instance_id, pred in zip(test_ids, predictions):
        print(f'<answer instance="{instance_id}" senseid="{pred}"/>')

if __name__ == '__main__':
    main()