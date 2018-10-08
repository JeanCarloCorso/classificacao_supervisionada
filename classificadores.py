from sklearn import svm
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import tree

def pca_func(X, m):# faz o pca
	pca = decomposition.PCA(n_components=m)
	pca.fit(X)
	Y = pca.transform(X)

	return pca.explained_variance_ratio_.astype(np.float32), Y.astype(np.float32)

def variabilidade(explain,colunas):
    soma = 0
    t = u = 0.0
    vetor_variabilidade = [[0,0,0],[0,0,0]]
    for i in range(0,colunas):
        soma += explain[i]
        if soma >= 0.75 and t < 0.75:
            print("\nvariancia: ", soma, "\nquantia: ",i+1)
            t = 0.75
            vetor_variabilidade[0][0] = i
            vetor_variabilidade[1][0] = soma
        if soma >= 0.90 and u < 0.90:
            print("\n\nvariancia: ", soma, "\nquantia: ",i+1)
            u = 0.90
            vetor_variabilidade[0][1] = i
            vetor_variabilidade[1][1] = soma
        if soma >= 0.99:
            print("\n\nvariancia: ", soma, "\nquantia: ",i+1)
            vetor_variabilidade[0][2] = i
            vetor_variabilidade[1][2] = soma
            break
    return vetor_variabilidade


def mostra(y_pred, teste_labels, acuracia, confusao, i, variabilidade):
    print("--> Variabilidade: ", variabilidade[1][i])
    #print("Tamanho do vetor de predição: ",y_pred.shape, "\nTamanho do vetor de labels teste:", teste_labels.shape)
    #print("Vetor de acerto:\n",teste_labels.ravel() == y_pred) 
    print("Acuracia: ",acuracia)
    print("Matriz de confuzão:\n",confusao)


def normalization(X):
	#normalizacao
	for i in range(X.shape[1]):
		X[...,i] = (X[...,i] - np.min(X[...,i])) / (np.max(X[...,i])
		 - np.min(X[...,i]))
	
	return X

def particionar(arquivo, numeradaor, denominador, colunas):
    linhas, coluna_label = arquivo.shape
    coluna_label -= 1
    linhas = int(linhas/denominador * numeradaor)
    embaralhado = arquivo
    np.random.shuffle(embaralhado)

    treino = embaralhado[0:linhas,...]
    teste = embaralhado[linhas:,...]

    treino_dados = treino[...,0:colunas]
    treino_labels = treino[...,coluna_label:]    
    
    teste_dados = teste[...,0:colunas:]
    teste_labels = teste[...,coluna_label:]
	
    return treino_dados, treino_labels, teste_dados, teste_labels

def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)}) # Para imprimir em decimal
    X = np.loadtxt("cancer.data", delimiter=",") # pega o dataset
    label_bruto = open("cancer-label.data", 'r')

    label = np.zeros(569).reshape((569, 1))
    c = 0
    for l in label_bruto:
        #print(l)
        if(l == "M\n"):
            #print("entrou M")
            label[c][0] = 1
        elif(l == "B\n"):
            #print("entrou B")
            label[c][0] = 0
        c = c + 1
    integrada = np.concatenate((X,label), axis=1) #junta o X e o label
    integrada = normalization(integrada) #normalização

    explain, Y = pca_func(integrada, integrada.shape[1])
    vetor_variabilidade = variabilidade(explain, integrada.shape[1])

    for i in range(0,3):
        treino_dados, treino_labels, teste_dados, teste_labels = particionar(integrada,2,3,vetor_variabilidade[0][i])


        C = 1.0  # SVM regularization parameter
        #SVM linear
        models = (svm.SVC(kernel='linear', C=C))
        models = (models.fit(treino_dados, treino_labels.ravel()))

        #compara a capacidade de previsão
        y_pred = models.predict(teste_dados)
        #print(y_pred)
        acuraciaSVMlinear = np.sum(teste_labels.ravel() == y_pred)/teste_labels.ravel().shape[0] 
        confusaoSVMlinear = confusion_matrix(teste_labels.ravel(), y_pred)

        print("\n-----------------------SVM-LINEAR-----------------------\n")
        
        mostra(y_pred, teste_labels, acuraciaSVMlinear, confusaoSVMlinear,i,vetor_variabilidade)

        #SVM não linear
        rbf = (svm.SVC(kernel='rbf', gamma=0.7, C=C))
        rbf = (rbf.fit(treino_dados, treino_labels.ravel()))

        y_pred = rbf.predict(teste_dados)
        acuraciaSVM_Nao_linear = np.sum(teste_labels.ravel() == y_pred)/teste_labels.ravel().shape[0] 
        confusaoSVM_Nao_linear = confusion_matrix(teste_labels.ravel(), y_pred)
        
        print("\n-----------------------SVM-NAO-LINEAR-----------------------\n")

        mostra(y_pred, teste_labels, acuraciaSVM_Nao_linear, confusaoSVM_Nao_linear,i,vetor_variabilidade)


        gnb = GaussianNB() #Cria o modelo Naive Bayes
        gnb.fit(treino_dados, treino_labels) # Treina o modelo com base nos dados de X
        y_pred = gnb.predict(teste_dados) # Prediz os dados de X com base no modelo criado

        #Avalia quantos acertos e erros o modelo obteve
        acuraciaNB = np.sum(teste_labels.ravel() == y_pred)/teste_labels.ravel().shape[0] 
        confusaoNB = confusion_matrix(teste_labels.ravel(), y_pred)

        print("\n-----------------------Naive-Bayes-----------------------\n")

        mostra(y_pred, teste_labels, acuraciaNB, confusaoNB,i,vetor_variabilidade)

        cart = tree.DecisionTreeRegressor()
        cart = cart.fit(treino_dados, treino_labels)
        y_pred = cart.predict(teste_dados)

        acuraciaCART = np.sum(teste_labels.ravel() == y_pred)/teste_labels.ravel().shape[0] 
        confusaoCART = confusion_matrix(teste_labels.ravel(), y_pred)

        print("\n-----------------------CART-----------------------\n")

        mostra(y_pred, teste_labels, acuraciaCART, confusaoCART,i,vetor_variabilidade)

    print("|   Classificador   |   Variabilidade   |   Colunas   |   Acuracia   |   Matriz de confusão   |")
    print("|   ")
    print("")
    print("")
    print("")
    print("")

main()
