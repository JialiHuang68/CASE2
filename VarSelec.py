def Var_Select_auto(orgdata, alphaMax=1, alphastep=0.2):
    if orgdata.iloc[:,1].count()>1000:
        data = orgdata.sample(1000)
    else:
        data = orgdata
    
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA, SparsePCA
    from functools import reduce
        
    data = preprocessing.scale(data)
    #data = pd.DataFrame(data)

    pca=PCA(whiten=True)
    newData=pca.fit(data)
    variance_ratio_ = pca.explained_variance_ratio_
    covMat = np.cov(data, rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    #print(eigVals)
    #print(variance_ratio_)

    ratio_sum = 0
    n_components=0
    for i,j in list(zip(variance_ratio_,eigVals)):
        ratio_sum += i
        if ratio_sum < 0.95 and j > 0.5:
            n_components += 1
        else:
            break
    
    #print(n_components)
    
    pca_n = list()
    for i in np.arange(0.1, alphaMax, alphastep):
        pca_model = SparsePCA(n_components=n_components, alpha=i)
        newData = pca_model.fit(data)
        pca = pd.DataFrame(pca_model.components_).T
        n = data.shape[1] - reduce(lambda x, y: sum(pca.iloc[:, 0] != 0) + sum(pca.iloc[:, 1] != 0),
                                   np.arange(n_components))
        pca_n.append((i, n))
    result = pd.DataFrame(pca_n)
    for i, j in enumerate(result.iloc[:, 1]):
        if j == 0:
            global best_alpha
            best_alpha = result.iloc[i, 0]
            break
    pca_model = SparsePCA(n_components=n_components, alpha=best_alpha)
    newData = pca_model.fit(data)
    pca = pd.DataFrame(pca_model.components_).T
    data = pd.DataFrame(data)
    score = pd.DataFrame(np.dot(data, pca))
    r = []
    R_square = []
    for paj in range(n_components):  # paj主成分个数
        for xk in range(data.shape[1]):  # xk输入变量个数
            r.append(abs(np.corrcoef(data.iloc[:, xk], score.iloc[:, paj])[0, 1]))
            r_max1 = max(r)
            r.remove(r_max1)
            r.append(-2)
            r_max2 = max(r)
            R_square.append((1 - r_max1 ** 2) / (1 - r_max2 ** 2))

    R_square = abs(pd.DataFrame(np.array(R_square).reshape((data.shape[1], n_components))))
    var_list = []
    #print(R_square)
   
    for i in range(n_components):
        vmin = R_square[i].min()
        #print(R_square[i])
        #print(vmin)
        #print(R_square[R_square[i] == min][i])
        var_list.append(R_square[R_square[i] == vmin][i].index)

    data_vc = orgdata.iloc[:, np.array(var_list).reshape(len(var_list))]
    return data_vc
    
    
    
    
def Var_Select(orgdata, k, alphaMax=1, alphastep=0.2):
    if orgdata.iloc[:,1].count()>1000:
        data = orgdata.sample(1000)
    else:
        data = orgdata
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA, SparsePCA
    from functools import reduce
    data = preprocessing.scale(data)
    n_components = k
    pca_n = list()
    for i in np.arange(0.1, alphaMax, alphastep):
        pca_model = SparsePCA(n_components=n_components, alpha=i)
        newData = pca_model.fit(data)
        pca = pd.DataFrame(pca_model.components_).T
        n = data.shape[1] - reduce(lambda x, y: sum(pca.iloc[:, 0] != 0) + sum(pca.iloc[:, 1] != 0),
                                   np.arange(n_components))
        pca_n.append((i, n))
    result = pd.DataFrame(pca_n)
    best_alpha=0.66
    for i, j in enumerate(result.iloc[:, 1]):
        if j == 0:
            
            best_alpha = result.iloc[i, 0]
            break
    
    pca_model = SparsePCA(n_components=n_components, alpha=best_alpha)
    newData = pca_model.fit(data)
    pca = pd.DataFrame(pca_model.components_).T
    data = pd.DataFrame(data)
    score = pd.DataFrame(np.dot(data, pca))
    r = []
    R_square = []
    for paj in range(n_components):  # paj主成分个数
        for xk in range(data.shape[1]):  # xk输入变量个数
            r.append(abs(np.corrcoef(data.iloc[:, xk], score.iloc[:, paj])[0, 1]))
            r_max1 = max(r)
            r.remove(r_max1)
            r.append(-2)
            r_max2 = max(r)
            R_square.append((1 - r_max1 ** 2) / (1 - r_max2 ** 2))

    R_square = abs(pd.DataFrame(np.array(R_square).reshape((data.shape[1], n_components))))
    var_list = []
    #print(R_square)
   
    for i in range(n_components):
        vmin = R_square[i].min()
        #print(R_square[i])
        #print(vmin)
        #print(R_square[R_square[i] == min][i])
        var_list.append(R_square[R_square[i] == vmin][i].index)

    data_vc = orgdata.iloc[:, np.array(var_list).reshape(len(var_list))]
    return data_vc