from serverConnection.server_connection import server_Connection as SC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, f1_score
import pandas  as pd
import pyodbc  as pbo
import numpy   as np
import sklearn as a
import matplotlib.pyplot as plt
import seaborn as SB


# =================================================================================================

def read_data():
    conn = pbo.connect('Driver={ODBC Driver 17 for SQL Server};'
                                'Server=RD10-129\MASOUD_0;'
                                'Database=PFMData;'
                                'Trusted_Connection=yes;')

    query = ('select [Pivot].[taglog].*,GuildCode from [Pivot].[taglog] tablesample(100000 rows) ' +
             'left join [PFM].[TerminalGuilds]' +
             'on [PFM].[TerminalGuilds].terminalkey = [Pivot].[taglog].TerminalKey')
    frame = pd.read_sql_query(query,conn)
    frame.set_index('terminalkey')
    frame.fillna(0,inplace = True)
    pr = frame['GuildCode']
    products = pr.unique()
    p = products.tolist()
    p.remove(0)
    return frame , p

#===================================================================================================

def get_features_and_labels(frame , productNumber):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''
    # Replace missing values with 0.0 , or we can use
    # scikit-learn to calculate missing values (below)
    #frame[frame.isnull()] = 0.0
    terminalkeys = frame['terminalkey'].copy(deep=True)
    labels = frame['GuildCode'].copy(deep=True)
    labelsLoc = labels.index[labels == productNumber].tolist()
    L = np.zeros(np.shape(labels))
    L[labelsLoc] = 1
    del frame['terminalkey']
    frame['GuildCode'] = L
    try:
    # Convert values to floats
        arr = np.array(frame, dtype=np.float)
    except Exception as e :
        print(e)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
    
    # Use 80% of the data for training; test against the rest
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn.pipeline.make_pipeline could also be used to chain 
    # processing and classification into a black box, but here we do
    # them separately.
    
    # If values are missing we could impute them from the training data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test , terminalkeys

# ========================================================================================================

def evaluate_classifier(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different classifiers to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, precision, recall)
    for each learner.
    '''


    
    # Here we create classifiers with default parameters. These need
    # to be adjusted to obtain optimal performance on your data set.
    
    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    ## Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    ## Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    print(precision)
    print(recall)
    ## Include the score in the title
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall


    classifier = RandomForestClassifier()
    ## Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    ## Generate the P-R curve
    y_prob = classifier.predict_proba(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
    print(precision)
    print(recall)
    ## Include the score in the title
    yield 'RandomForrestClassifier (F1 score={:.3f})'.format(score), precision, recall

    classifier = DecisionTreeClassifier()
    ## Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    ## Generate the P-R curve
    y_prob = classifier.predict_proba(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
    print(precision)
    print(recall)
    ## Include the score in the title
    yield 'decisionTreeClassifier (F1 score={:.3f})'.format(score), precision, recall

    classifier = QuadraticDiscriminantAnalysis()
    ## Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    ## Generate the P-R curve
    y_prob = classifier.predict_proba(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
    print(precision)
    print(recall)
    ## Include the score in the title
    yield 'QDA Classifier (F1 score={:.3f})'.format(score), precision, recall

    # Test the Nu support vector classifier
    ##for i in range(1,100):
    ##    Nu = i/100
    ##    try :
    ##        classifier = NuSVC(kernel='rbf', nu=Nu, gamma=0.5)
    ##        # Fit the classifier
    ##        classifier.fit(X_train, y_train)
    ##        score = f1_score(y_test, classifier.predict(X_test))
    ##        # Generate the P-R curve
    ##        y_prob = classifier.decision_function(X_test)
    ##        precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ##        print('found feasible Nu : {}'.format(Nu))
    ##        # Include the score in the title
    ##    except Exception as e:
    ##        print(e)
    ##    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall

# ==================================================================================================================

def plot(results):
    '''
    Create a plot comparing multiple learners.
    `results` is a list of tuples containing:
        (title, precision, recall)
    All the elements in results will be plotted.
    '''
    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from Masoud')

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='upper right')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()

# ==========================================================================================

if __name__ == '__main__':
    # Download the data set from URL
    ##print("Downloading data from {}".format(URL))
    frame , p = read_data()
    product = p[0]
    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test , terminalkeys = get_features_and_labels(frame,product)

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    plot(results)