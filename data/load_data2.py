
import pickle


f = open('bags.pckl')
bags= pickle.load(f)
f.close()

f = open('labels.pckl')
labels= pickle.load(f)
f.close()