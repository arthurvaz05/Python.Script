
# coding: utf-8

# In[167]:


import pandas as pd
import numpy as np
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)


# In[168]:


receita = pd.read_csv('/Users/arthurlambletvaz/Downloads/base de receita.csv', sep =';')
receita_treino = pd.read_csv('/Users/arthurlambletvaz/Downloads/receias.treino.csv', sep =',')


# In[169]:


receita.corr()


# In[170]:


sns.lmplot('Pasta de Nuts', 'Oleo de coco', data=receita, hue='Nivel de doce',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})


# In[171]:


ingredientes = receita[['Pasta de Nuts','Oleo de coco']].values
receita_attr = receita[['Nivel de doce']]


# In[172]:


#Train SVMs with different kernels
svc = svm.SVC(kernel='linear').fit(ingredientes, receita['Nivel de doce'])
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7).fit(ingredientes, receita['Nivel de doce'])
poly_svc = svm.SVC(kernel='poly', degree=3).fit(ingredientes, receita['Nivel de doce'])

#Create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = ingredientes[:, 0].min() - 1, ingredientes[:, 0].max() + 1
y_min, y_max = ingredientes[:, 1].min() - 1, ingredientes[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#Define title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure(i)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(ingredientes[:, 0], ingredientes[:, 1], c=receita['Nivel de doce'], cmap=plt.cm.ocean)
    plt.xlabel('Pasta de Nuts')
    plt.ylabel('Oleo de coco')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()


# In[130]:


receita_treino = receita_treino.drop('Unnamed: 0', 1)


# In[161]:


receita_treino_list = rbf_svc.predict(receita_treino[['Pasta de Nuts', 'Oleo de coco']])


# In[162]:


receita_treino ['Nivel de doce'] = receita_treino_list


# In[163]:


sns.lmplot('Pasta de Nuts', 'Oleo de coco', data=receita_treino, hue='Nivel de doce',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})


# In[166]:


receita_treino = receita_treino.drop('Unnamed: 0', 1)
receita_treino_list = poly_svc.predict(receita_treino[['Pasta de Nuts', 'Oleo de coco']])
receita_treino ['Nivel de doce'] = receita_treino_list
sns.lmplot('Pasta de Nuts', 'Oleo de coco', data=receita_treino, hue='Nivel de doce',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})

