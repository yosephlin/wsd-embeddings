Performs WSD for the word "line" using GloVe word embeddings
as features. Implements SVM with scikit and a Neural Network with PyTorch.

Usage
    python wsd-embeddings.py <train_file> <test_file> <embedding_file> [SVM|NN] > my-line-answers.txt

Models used
    SVM and NN (default is SVM)
    SVM - Scikit-learn SVC with a linear kernel
    NN - PyTorch Neural Network with one hidden layer
