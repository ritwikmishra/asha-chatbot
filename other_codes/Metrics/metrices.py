import pandas as pd 
import numpy as np 
import collections as cols

# convert the to binary matrix 
def get_data(acolumns,right_ans):
    repd = []
    for _,vec in acolumns.iterrows(): # vec = [2,3,1]
        rep = []
        for ind in vec.values:
            if ind in right_ans:
                rep.append(1)
            else:
                rep.append(0)
        repd.append(rep)
    return np.array(repd)

def precision_at_k(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    prec = []
    for vec in data:
        rev_doc = 0
        tot_doc = 0
        prec_vec = []
        for doc in vec:
            if doc == 1:
                rev_doc +=1
            tot_doc +=1
            prec_vec.append(rev_doc/tot_doc)
        prec.append(np.array(prec_vec))
    return np.array(prec)

def precision_at_3(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    prec_at_k = precision_at_k(data)
    k = 3-1
    prec_at_3 = [vec[k] for vec in prec_at_k]
    return np.array(prec_at_3)


def avg_precision_at_3(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    data = precision_at_k(data)
    aprec = []
    for vec in data:
        aprec.append(sum(vec)/3)
    return np.array(aprec)

def reciprocal_rank(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    rr = []
    for vec in data: #  vec = [0,1,1]
        i = 1
        for ind in vec:
            if ind == 1:
                break
            i+=1
        #print(i)
        if i !=4:
            rr.append(1/i)
        else:
            rr.append(0)
    return np.array(rr)

def success_ratio(data):
    """
    param : data
    2-D array representing right ans in each row
    """
    sr_mat = []
    for vec in data:
        sr_mat.append(sum(vec)/len(vec))
    
    return np.array(sr_mat)

def invert_rel(element):
    rmap = {1:2,2:1,3:0}
    return rmap[element] 

def ndcg(acolumn):
    """
    param : acolumn
    pandas series representing ans in each cell
    """
    idcg_vec = np.array([1/np.log2(2) , 1/np.log2(3) , 1/np.log2(4)])
    temp = pd.DataFrame()
    temp["final_1"] = acolumn["final_1"].apply(invert_rel)
    temp["final_2"] = acolumn["final_2"].apply(invert_rel)
    temp["final_3"] = acolumn["final_3"].apply(invert_rel)

    ndcg_vec = []

    for _,vec in temp.iterrows(): #  vec = [1,2,1] 
        vec = vec.values  
        idcg = sum(idcg_vec*sorted(vec,reverse=True))
        if idcg!=0:
            ndcg_vec.append(np.sum(vec*idcg_vec)/idcg)
        else:
            ndcg_vec.append(0)
    return np.array(ndcg_vec)

def grp_mean(grp_mat):
    return (np.mean(grp_mat[0]),np.mean(grp_mat[1]),np.mean(grp_mat[2]))

def calc_metrics(df1):
    """
    param : acolumn
    pandas series representing ans in each cell
    """
    acolumn = df1[["final_1","final_2","final_3"]] 
    data = get_data(acolumn,[1,2])
    m_mat = {}

    m_mat["map"] = np.mean(avg_precision_at_3(data))
    m_mat["mrr"] = np.mean(reciprocal_rank(data))
    m_mat["avg_sr"] = np.mean(success_ratio(data))
    m_mat["avg_ndcg"] = np.mean(ndcg(acolumn))
    m_mat["avg_p3"] = np.mean(precision_at_3(data))
    return m_mat

