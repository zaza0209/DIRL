import numpy as np
def IC(loss, changepoints, g_index, N, T, K, C):
    # """
    # Information criterion
    # Parameters
    # ----------
    # loss : TYPE
    #     DESCRIPTION.
    # C : h(\sum \tau_i) = C log(\sum \tau_i)

    # """
    Kl = K*np.log(np.sum(T-1 -changepoints))
    Ck, indicesList, occurCount = np.unique(g_index, return_index = True,return_counts=True)
    return loss - Kl+ occurCount.dot((T-1 -changepoints)[np.s_[indicesList]]) * C * np.log(np.sum(T-1 -changepoints))


