
"""This file contains different ensemble and fake video decision strategies"""



import numpy as np

def max(a,b):
    if(a>b):
        return a
    else:
        return b

def ensemble_strategy_1(a,b):
    return (a+b)/2

def ensemble_strategy_2(a,b):
    return max(a,b)



def ensemble_strategy_3(a,b):
    if(a>=0.7 and b>=0.7):
        return max(a,b)
    elif(a>=0.4 and a<= 0.6 and b>=0.4 and b<=0.6):
        return min(a,b)
    elif(a>=0.6 and a<=0.7 and b>=0.6 and b<=0.7):
        return max(a,b)
    else:
        return min(a,b)

def ensemble_strategy_4(a,b,w1=0.4,w2=0.6):
    return (a*w1+b*w2)


def confident_strategy_4(preds):
    preds.sort()
    average=0
    start=int(len(preds)/3)
    end = len(preds)
    for i in range(start,end):
        average+=preds[i]
    return average/(end-start)


def ensemble_strategy_5(a,b):
    if(a>=0.7 and b>=0.7):
        return max(a,b)
    if(a<=0.25 and b<=0.25):
        return min(a,b)
    else:
        return (a+b)/2

confident = lambda p: np.mean(np.abs(p-0.5)*2) >= 0.7
label_spread = lambda x: x-np.log10(x) if x >= 0.8 else x


def confident_strategy_1(pred):
    return np.mean(pred)

def confident_strategy_2(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)




def confident_strategy_3(preds):
	#
	# If there is a fake id and we're confident,
	# return spreaded fake score, otherwise return
	# the original fake score.
	# If everyone is real and we're confident return
	# the minimum real score, otherwise return the
	# mean of all predictions.
	#
	preds = np.array(preds)
	p_max = np.max(preds)
	if p_max >= 0.8:
		if confident(preds):
			return label_spread(p_max)
		return p_max
	if confident(preds):
		return np.min(preds)
	return np.mean(preds)
