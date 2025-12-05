from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

plt.rc('font', size=14)  # 기본 폰트 크기
plt.rc('axes', labelsize=14)  # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=12)  # 범례 폰트 크기
plt.rc('figure', titlesize=18)

# ============================================================
# 1. Dataset selection +  prefix selection
# ============================================================
DATASET = 'breast'   # In case of breast cancer dataset, you can choose it.
# DATASET = 'wifi'   # You can change it when you want to analyze wifi localization dataset.

if DATASET == 'breast':
    prefix = '[breast]'
elif DATASET == 'wifi':
    prefix = '[wifi]'
else:
    raise ValueError("DATASET must be 'breast' or 'wifi'")

# ============================================================
# 2. Dataset loading (Breast cancer dataset or wifi localization):
# ============================================================
if DATASET == 'breast':
    data1 = pd.read_csv('data/wisc_bc_data.csv')
    data1n = data1.values.copy()
    num_row, num_col = np.shape(data1n)
    for i in range(num_row):
        if data1n[i, 0] == 'B':
            data1n[i, 0] = 1
        else:
            data1n[i, 0] = 0
    minmax = preprocessing.MinMaxScaler()
    x = data1n[:, 1:]
    x = minmax.fit_transform(x)
    x = x.astype(float)

    y = data1n[:, 0]
    y = y.astype(int)

elif DATASET == 'wifi':
    data1 = pd.read_csv('data/wifi_localization.csv')
    data1n = data1.values.copy()
    num_row, num_col = np.shape(data1n)

    x = data1n[:, :-1].astype(float)
    y = data1n[:, -1].astype(int)

    minmax = preprocessing.MinMaxScaler()
    x = minmax.fit_transform(x)

# =======================================================================
# 3. Comparison of Five Classification Models(DT, NN, SVM, Ada, and KNN)
# =======================================================================

#x,y = shuffle(x, y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size = 0.3, random_state=42, shuffle = True)
wall_time = []
accuracy = []

# 1. Decision Tree
def best_parameter_dt(Xtrain, Ytrain, cv):
    clf1 = tree.DecisionTreeClassifier(random_state=42)
    max_range = np.arange(1, 20, 1)
    train_score1, validation_score1 = validation_curve(clf1, Xtrain, Ytrain, param_name='max_depth', param_range=max_range, cv=cv, n_jobs=-1)
    best_max_depth = max_range[np.argmax(np.mean(validation_score1, axis=1))]
    clf2 = tree.DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    ccp_range = np.arange(0.00, 0.03, 0.001)
    train_score2, validation_score2 = validation_curve(clf2, Xtrain, Ytrain, param_name='ccp_alpha', param_range=ccp_range, cv=cv, n_jobs=-1)
    best_ccp = ccp_range[np.argmax(np.mean(validation_score2, axis=1))]
    best_classifier = tree.DecisionTreeClassifier(max_depth=best_max_depth, ccp_alpha=best_ccp, random_state=42)

    plt.figure(1)
    plt.title("Validation Curve of DT")
    plt.xlabel("Max depth")
    plt.ylabel("Score")
    plt.plot(np.arange(1, 20), np.mean(train_score1, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(np.arange(1, 20), np.mean(validation_score1, axis=1), label='valid score', marker='o', color='red', lw=1.5)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(prefix + 'DT-maxdepth.png')

    plt.figure(2)
    plt.title("Validation Curve of DT")
    plt.xlabel("ccp_alphas")
    plt.ylabel("Score")
    plt.plot(ccp_range,np.mean(train_score2, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(ccp_range,np.mean(validation_score2, axis=1), label='valid score', marker='o', color='red', lw=1.5)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(prefix + 'DT-ccp.png')
    #plt.show()
    return best_classifier

best_classifier = best_parameter_dt(Xtrain, Ytrain, 10)
start_time = time.time()
best_classifier.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time - start_time)
print('소요시간 DT:', end_time-start_time)
accuracy.append(accuracy_score(Ytest, best_classifier.predict(Xtest)))
print('ccp:', best_classifier.ccp_alpha)
print('Max depth:', best_classifier.max_depth)
print("best DT classifier accuracy: ", accuracy_score(Ytest, best_classifier.predict(Xtest)))

def plot_learning_curve_dt(Xtrain, Ytrain, best_classifier, cv):
    train_sizes, train_score3, validation_score3 = learning_curve(best_classifier, Xtrain, Ytrain, train_sizes= np.linspace(0.1,1.0,10), cv = cv, n_jobs=-1)
    plt.figure(3)
    plt.title("Learning Curve of DT")
    plt.xlabel("Percent of training examples")
    plt.ylabel("Score")
    plt.plot(np.mean(train_score3, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score3, axis=1), label='valid score', marker='o',color='red', lw=1.5)
    ind = np.arange(len(np.mean(validation_score3, axis=1)))
    labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
    plt.xticks(ind, labels)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(prefix + 'Learning-curve-DT.png')
    #plt.show()
plot_learning_curve_dt(Xtrain, Ytrain, best_classifier, 10)


# 2. Neural Network
def best_parameter_nn(Xtrain, Ytrain, cv):
    initial_range = np.logspace(-4, 1, 10)
    alpha_range = np.logspace(-4, 1, 10)
    hidden_range = np.arange(5, 30, 1)
    hidden_range2 = [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6), (8, 4), (8, 5)]
    clf1 = MLPClassifier(max_iter=2000, solver='adam', random_state=42)
    train_score1, validation_score1 = validation_curve(clf1, Xtrain, Ytrain, param_name='learning_rate_init', param_range=initial_range,cv=cv, n_jobs=-1)
    best_learning_rate = initial_range[np.argmax(np.mean(validation_score1, axis=1))]
    #best_learning_rate = 0.1
    clf2 = MLPClassifier(max_iter=2000, solver='adam', learning_rate_init=best_learning_rate, random_state=42)
    train_score2, validation_score2 = validation_curve(clf2, Xtrain, Ytrain, param_name='alpha', param_range=alpha_range, cv=cv, n_jobs=-1)
    best_alpha = alpha_range[np.argmax(np.mean(validation_score2, axis=1))]
    clf3 = MLPClassifier(max_iter=2000, solver='adam', learning_rate_init=best_learning_rate, alpha=best_alpha, random_state=42)
    train_score3, validation_score3 = validation_curve(clf3, Xtrain, Ytrain, param_name='hidden_layer_sizes',param_range=hidden_range, cv=cv, n_jobs=-1)
    train_score4, validation_score4 = validation_curve(clf3, Xtrain, Ytrain, param_name='hidden_layer_sizes',param_range=hidden_range2, cv=cv, n_jobs=-1)
    best_hidden_layer1 = hidden_range[np.argmax(np.mean(validation_score3, axis=1))]
    best_hidden_layer2 = hidden_range2[np.argmax(np.mean(validation_score4, axis=1))]
    if max(np.mean(validation_score4, axis=1)) > max(np.mean(validation_score3, axis=1)):
        best_hidden_layer = best_hidden_layer2
    else:
        best_hidden_layer = best_hidden_layer1
    plt.figure(4)
    plt.title("Validation Curve of NN")
    plt.xlabel("initial learning_rate")
    plt.ylabel("Score")
    plt.semilogx(initial_range, np.mean(train_score1, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.semilogx(initial_range, np.mean(validation_score1, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'NN-initial_rate.png')

    plt.figure(5)
    plt.title("Validation Curve of NN")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Score")
    plt.grid()
    plt.semilogx(alpha_range, np.mean(train_score2, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.semilogx(alpha_range, np.mean(validation_score2, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'NN-alpha.png')

    plt.figure(6)
    plt.title("Validation Curve of NN")
    plt.xlabel("Number of nodes (Single layer)")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(hidden_range, np.mean(train_score3, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.plot(hidden_range, np.mean(validation_score3, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'NN-hidden-layer.png')

    plt.figure(7)
    plt.title("Validation Curve of NN")
    plt.xlabel("Number of nodes (2 layers)")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(np.mean(train_score4, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score4, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    ind = np.arange(len(np.mean(validation_score4, axis=1)))
    labels = ['(4, 4)', '(4, 5)', '(4, 6)', '(5, 5)', '(5, 6)', '(6, 6)', '(8, 4)', '(8, 5)']
    plt.xticks(ind, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'NN-2-hidden-layer.png')
    #plt.show()
    best_classifier = MLPClassifier(max_iter=2000, solver='adam', learning_rate_init=best_learning_rate, alpha=best_alpha, hidden_layer_sizes=best_hidden_layer, random_state=42)
    return best_classifier

best_classifier = best_parameter_nn(Xtrain, Ytrain, 10)
print('init:', best_classifier.learning_rate_init)
print('alpha:', best_classifier.alpha)
print('layer:', best_classifier.hidden_layer_sizes)

start_time = time.time()
best_classifier.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time - start_time)
accuracy.append(accuracy_score(Ytest, best_classifier.predict(Xtest)))
print("best NN classifier accuracy: ", accuracy_score(Ytest, best_classifier.predict(Xtest)))

def plot_learning_curve_nn(Xtrain, Ytrain, best_classifier, cv):
    train_sizes, train_score4, validation_score4 = learning_curve(best_classifier, Xtrain, Ytrain, train_sizes= np.linspace(0.1,1.0,10), cv = cv, n_jobs=-1)
    plt.figure(8)
    plt.title("Learning Curve of NN")
    plt.xlabel("samples")
    plt.ylabel("Score")
    plt.plot(np.mean(train_score4, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score4, axis=1), label='valid score', marker='o',color='red', lw=1.5)
    plt.legend()
    plt.grid()
    ind = np.arange(len(np.mean(validation_score4, axis=1)))
    labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
    plt.xticks(ind, labels)
    plt.tight_layout()
    plt.savefig(prefix + 'NN-learning-curve.png')
    #plt.show()
plot_learning_curve_nn(Xtrain, Ytrain, best_classifier, 10)

def plot_loss(Xtrain, Ytrain, best_classifier):
    epoch = 1000
    clf = MLPClassifier(alpha=best_classifier.alpha, hidden_layer_sizes=best_classifier.hidden_layer_sizes, random_state=42,
                        max_iter=1, learning_rate_init= best_classifier.learning_rate_init, solver='adam', warm_start=True)
    loss_value = np.zeros(epoch)
    X_train, Xval, Y_train, Yval = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=42, shuffle=True)
    for i in range(epoch):
        clf.fit(X_train, Y_train)
        loss_value[i] = log_loss(Yval, clf.predict_proba(Xval))
    plt.figure(9)
    plt.plot(np.arange(epoch), loss_value, label='loss value for each epoch')
    plt.title("Loss function (Cross entropy)")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'NN-loss.png')
    #plt.show()
plot_loss(Xtrain, Ytrain, best_classifier)

# 3. Support Vector Machine
def best_parameter_svm(Xtrain, Ytrain, cv):
    kernel = ['rbf', 'linear', 'sigmoid']
    clf1 = SVC(random_state=42)
    train_score1, validation_score1 = validation_curve(clf1, Xtrain, Ytrain, param_name='kernel', param_range=kernel,cv=cv, n_jobs=-1)
    best_kernel = kernel[np.argmax(np.mean(validation_score1, axis=1))]
    print(best_kernel)
    C_range = np.logspace(-3, 2, 10)
    clf2 = SVC(kernel=best_kernel, random_state=42)
    train_score2, validation_score2 = validation_curve(clf2, Xtrain, Ytrain, param_name='C', param_range=C_range, cv=cv, n_jobs=-1)
    best_C = C_range[np.argmax(np.mean(validation_score2, axis=1))]
    gamma_range = np.logspace(-3, 2, 10)
    clf3 = SVC(kernel=best_kernel, C = best_C, random_state=42)
    train_score3, validation_score3 = validation_curve(clf3, Xtrain, Ytrain, param_name='gamma', param_range=gamma_range, cv=cv,n_jobs=-1)
    best_gamma = gamma_range[np.argmax(np.mean(validation_score3, axis=1))]
    best_classifier = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma, random_state=42)

    plt.figure(10)
    plt.title("SVM kernel selection")
    plt.xlabel("Kernel types")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(np.mean(train_score1, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score1, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    ind = np.arange(len(np.mean(validation_score1, axis=1)))
    labels = ['rbf', 'linear', 'sigmoid']
    plt.xticks(ind, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'SVM-kernel.png')

    plt.figure(11)
    plt.title("SVM C selection")
    plt.xlabel("C values")
    plt.ylabel("Score")
    plt.grid()
    plt.semilogx(C_range, np.mean(train_score2, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.semilogx(C_range, np.mean(validation_score2, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'SVM-C.png')

    plt.figure(12)
    plt.title("SVM gamma selection")
    plt.xlabel("gamma values")
    plt.ylabel("Score")
    plt.grid()
    plt.semilogx(C_range, np.mean(train_score3, axis=1), marker='o', label='train score', color='blue', lw=1.5)
    plt.semilogx(C_range, np.mean(validation_score3, axis=1), marker='o', label='valid score', color='red', lw=1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'SVM-gamma.png')
    #plt.show()
    return best_classifier

best_classifier = best_parameter_svm(Xtrain, Ytrain, 10)
print('kernel:', best_classifier.kernel)
print('C:', best_classifier.C)
print('gamma:', best_classifier.gamma)

start_time = time.time()
best_classifier.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time - start_time)
accuracy.append(accuracy_score(Ytest, best_classifier.predict(Xtest)))
print("best SVM classifier accuracy: ", accuracy_score(Ytest, best_classifier.predict(Xtest)))

def plot_learning_curve_svm(Xtrain, Ytrain, best_classifier, cv):
    train_sizes, train_score4, validation_score4 = learning_curve(best_classifier, Xtrain, Ytrain, train_sizes= np.linspace(0.1,1.0,10), cv = cv, n_jobs=-1)
    plt.figure(13)
    plt.title("Learning Curve of SVM")
    plt.xlabel("samples")
    plt.ylabel("Score")
    plt.plot(np.mean(train_score4, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score4, axis=1), label='valid score', marker='o',color='red', lw=1.5)
    plt.legend()
    plt.grid()
    ind = np.arange(len(np.mean(validation_score4, axis=1)))
    labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
    plt.xticks(ind, labels)
    plt.tight_layout()
    plt.savefig(prefix + 'SVM-learning-curve.png')
    #plt.show()
plot_learning_curve_svm(Xtrain, Ytrain, best_classifier, 10)

# 4. AdaBoost Classifier
def best_parameter_adaboost(Xtrain, Ytrain, cv):
    clf1 = AdaBoostClassifier(random_state=42)
    lr_rate_range = np.logspace(-3, 1, 10)
    train_score1, validation_score1 = validation_curve(clf1, Xtrain, Ytrain, param_name="learning_rate", param_range=lr_rate_range, cv = cv)
    best_lr_rate=lr_rate_range[np.argmax(np.mean(validation_score1, axis=1))]
    clf2 = AdaBoostClassifier(learning_rate=best_lr_rate, random_state=42)
    n_estimator_range = np.arange(5, 500, 20)
    train_score2, validation_score2 = validation_curve(clf2, Xtrain, Ytrain, param_name="n_estimators", param_range=n_estimator_range, cv = cv)
    best_n_estimators=n_estimator_range[np.argmax(np.mean(validation_score2, axis=1))]
    best_classifier = AdaBoostClassifier(learning_rate=best_lr_rate, n_estimators=best_n_estimators, random_state=42)

    plt.figure(14)
    plt.title("Validation Curve of Adaboost LR")
    plt.xlabel("learning_rates")
    plt.ylabel("Score")
    plt.plot(lr_rate_range, np.mean(train_score1, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.semilogx(lr_rate_range, np.mean(validation_score1, axis=1), label='valid score', marker='o', color='red', lw=1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'Adaboost-LR.png')

    plt.figure(15)
    plt.title("Validation Curve of Adaboost N estimates")
    plt.xlabel("n_estimates")
    plt.ylabel("Score")
    plt.plot(n_estimator_range, np.mean(train_score2, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(n_estimator_range, np.mean(validation_score2, axis=1), label='valid score', marker='o', color='red', lw=1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'Adaboost-n_estimate.png')
    #plt.show()
    return best_classifier

best_classifier = best_parameter_adaboost(Xtrain, Ytrain, 10)
start_time = time.time()
best_classifier.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time - start_time)
accuracy.append(accuracy_score(Ytest, best_classifier.predict(Xtest)))
print('lr_rate:', best_classifier.learning_rate)
print('n_estimator:', best_classifier.n_estimators)
print("best AdaBoost classifier accuracy: ", accuracy_score(Ytest, best_classifier.predict(Xtest)))

def plot_learning_curve_ada(Xtrain, Ytrain, best_classifier, cv):
    train_sizes, train_score4, validation_score4 = learning_curve(best_classifier, Xtrain, Ytrain, train_sizes= np.linspace(0.1,1.0,10), cv = cv, n_jobs=-1)
    plt.figure(16)
    plt.title("Learning Curve of AdaBoost")
    plt.xlabel("samples")
    plt.ylabel("Score")
    plt.plot(np.mean(train_score4, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score4, axis=1), label='valid score', marker='o',color='red', lw=1.5)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    ind = np.arange(len(np.mean(validation_score4, axis=1)))
    labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
    plt.xticks(ind, labels)
    plt.savefig(prefix + 'AdaBoost-learning-curve.png')
    #plt.show()
plot_learning_curve_ada(Xtrain, Ytrain, best_classifier, 10)


# 5. K-neighbors classifier
def best_parameter_knn(Xtrain, Ytrain, cv=5):
    p_range = np.arange(1,5)
    clf1 = KNeighborsClassifier()
    train_score1, validation_score1 = validation_curve(clf1, Xtrain, Ytrain, param_name="p",param_range=p_range, cv=cv, n_jobs=-1)
    best_p=p_range[np.argmax(np.mean(validation_score1, axis=1))]
    clf2 = KNeighborsClassifier(p=best_p)
    knn_range = np.arange(1, 50)
    train_score2, validation_score2 = validation_curve(clf2, Xtrain, Ytrain, param_name="n_neighbors",param_range=knn_range, cv=cv, n_jobs=-1)
    best_neighbor = knn_range[np.argmax(np.mean(validation_score2, axis=1))]
    best_classifier=KNeighborsClassifier(p=best_p, n_neighbors=best_neighbor)

    plt.figure(17)
    plt.title("Validation Curve of KNN")
    plt.xlabel("Minkowski metric")
    plt.ylabel("Score")
    plt.plot(p_range, np.mean(train_score1, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(p_range, np.mean(validation_score1, axis=1), label='valid score', marker='o', color='red', lw=1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'KNN-p.png')

    plt.figure(18)
    plt.title("Validation Curve of KNN")
    plt.xlabel("n_neighbors")
    plt.ylabel("Score")
    plt.plot(knn_range, np.mean(train_score2, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(knn_range, np.mean(validation_score2, axis=1), label='valid score', marker='o', color='red', lw=1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + 'KNN-n_neighbors.png')
    #plt.show()
    return best_classifier

best_classifier = best_parameter_knn(Xtrain, Ytrain, 10)

start_time = time.time()
best_classifier.fit(Xtrain, Ytrain)
end_time = time.time()
wall_time.append(end_time - start_time)
print('소요시간 KNN:', end_time-start_time)
accuracy.append(accuracy_score(Ytest, best_classifier.predict(Xtest)))
print('best_p:', best_classifier.p)
print('best_neighbor:', best_classifier.n_neighbors)
print("best KNN classifier accuracy: ", accuracy_score(Ytest, best_classifier.predict(Xtest)))

def plot_learning_curve_knn(Xtrain, Ytrain, best_classifier, cv):
    train_sizes, train_score4, validation_score4 = learning_curve(best_classifier, Xtrain, Ytrain, train_sizes= np.linspace(0.1,1.0,10), cv = cv, n_jobs=-1)
    plt.figure(19)
    plt.title("Learning Curve of KNN")
    plt.xlabel("samples")
    plt.ylabel("Score")
    plt.plot(np.mean(train_score4, axis=1), label='train score', marker='o', color='blue', lw=1.5)
    plt.plot(np.mean(validation_score4, axis=1), label='valid score', marker='o',color='red', lw=1.5)
    plt.legend()
    plt.grid()
    ind = np.arange(len(np.mean(validation_score4, axis=1)))
    labels = ['10%','20%', '30%','40%','50%','60%','70%','80%','90%','100%']
    plt.xticks(ind, labels)
    plt.tight_layout()
    plt.savefig(prefix + 'KNN-learning-curve.png')
    #plt.show()

plot_learning_curve_knn(Xtrain, Ytrain, best_classifier, 10)

clfs = ['DT', 'NeuralNet', 'SVM', 'AdaBoost', 'KNN']
x = np.arange(len(clfs))
plt.figure(20)
bar = plt.bar(x, wall_time)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.5f'%height, ha='center', va='bottom', size=12)
plt.title('Training Times for five Classifiers')
plt.xticks(x, clfs)
plt.ylabel('Training Time (seconds)')
plt.tight_layout()
plt.savefig(prefix + 'train_time.png')
#plt.show()

plt.figure(21)
bar = plt.bar(x, accuracy)
for i in bar:
    height = i.get_height()
    plt.text(i.get_x() + i.get_width()/2, height,'%.4f'%height, ha='center', va='bottom', size=12)
plt.title('Accuracy for five Classifiers')
plt.xticks(x, clfs)
plt.ylabel('Accuracy score')
plt.tight_layout()
plt.savefig(prefix + 'train_accuracy.png')

print('Train time for each Classifier :', wall_time)
print('Accuracy for each Classifier :', accuracy)