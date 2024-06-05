import numpy as np
import pickle
#LOADING THE SAVED MODEL
loaded_model=pickle.load(open('C:/Users\prath\OneDrive\Desktop\machine_learning/trained_model.sav','rb'))


input_data=[5,166,72,19,175,25.8,0.587,51]

#CHANGING THE INPUT DATA TO NUMPY ARRAY
ipdata_as_nparray=np.asarray(input_data)

#RESHAPE THE ARRAY AS WE ARE PREDICTING FOR ONE INSTANCE
ipdata_reshape=ipdata_as_nparray.reshape(1,-1)

pred=loaded_model.predict(ipdata_reshape)
print(pred)

if(pred[0]==0):
  print("The person is not diabetic")
else:
  print("The person is diabetic")