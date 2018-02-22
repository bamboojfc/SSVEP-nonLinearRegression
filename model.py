
# coding: utf-8

# In[38]:


import scipy.io
import numpy as np
import itertools
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from random import randint
import sklearn.linear_model as linear_model
import sklearn.preprocessing as prep
import math


# In[21]:


## import dataset
mat = scipy.io.loadmat('SSVEPDataset.mat')
data = mat['subject'][0]
number_of_subjects = len(data)
number_of_conditions = len(data[0])
number_of_samplings = len(data[0][0])
print "Data includes", number_of_subjects, "subjects :"
print "(", number_of_conditions, "conditions per subject )"
print "(", number_of_samplings, "samplings per condition )"
#print data


# In[22]:


## select one condition data and filter out first 10-second data
#set params
condition_id =3 # ( 1 to 5 )
number_of_samplings_per_sec = 250
filtered_secs = 10
all_secs = number_of_samplings/number_of_samplings_per_sec
used_secs = all_secs - filtered_secs
number_of_filter_out_samplings = number_of_samplings_per_sec * filtered_secs

#get data
data_selected = np.zeros((number_of_subjects, number_of_samplings-number_of_filter_out_samplings))
for i, d in enumerate(data):
    join_list = list(itertools.chain.from_iterable(d[condition_id-1]))
    
    #bandpass filter
    nyq = 0.5 * number_of_samplings_per_sec
    low = 7 / nyq
    high = 8 / nyq
    order = 2
    b, a = butter(order, [low, high], btype='band')
    f = lfilter(b, a, join_list)
    
    #filter out first-ten second
    data_selected[i] = f[number_of_filter_out_samplings:]

print "Select data from condition #", condition_id
print "Size of data is", len(data_selected), "subjects with", len(data_selected[0]), "samplings per subject."


# In[23]:


## perform Fast Fourier Transform (FFT)
#set params
window_size = 5 #seconds
number_of_slide_windows = used_secs-window_size+1
fft_out_max_list = np.zeros((number_of_subjects, number_of_slide_windows))
print "Each subjects contains", number_of_slide_windows, "windows."

#FFT
for i, d in enumerate(data_selected):
    print "==== FFT with subjects #", i, "===="
    for index in range(0, number_of_slide_windows):
        #print "From second #", index, "to", index+window_size-1,"( sampling no.", index*number_of_samplings_per_sec, "to", (index + window_size) * number_of_samplings_per_sec - 1, ")"
        
        #FFT with one window
        fft_out = fft(d[index*number_of_samplings_per_sec : (index + window_size) * number_of_samplings_per_sec])
        freqs = fftfreq(len(fft_out)) * number_of_samplings_per_sec
        
        #Get value from maximum freq
        fft_out_max_list[i][index] = np.abs(fft_out)[np.where(freqs==7.6)]

        if index == number_of_slide_windows - 1:
            #plot FFT of some specific window
            fig, ax = plt.subplots()
            ax.plot(freqs, np.abs(fft_out))
            ax.set_xlabel('Frequency in Hertz [Hz]')
            ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
            ax.set_xlim(1, 30)
            ax.set_ylim(1, 8000)
            plt.grid()
            plt.show() 
            
    plt.plot(fft_out_max_list[i], 'ro')
    plt.xlabel('Window no.')
    plt.ylabel('Max Spectrum Magnitude')
    plt.grid()
    plt.show()
                


# In[24]:


print fft_out_max_list.shape


# In[25]:


def train_test_split(x, y, test_size = 1):
    r = []
    for i in range(test_size):
        rand = 0
        while True:
            rand = randint(0, number_of_subjects-1)
            if rand not in r:
                break
            else:
                print rand, r
        r.append(rand)
    print "Random result: subject #", r, "is test set."
    
    y_test = [y[i] for i in r]
    y_train = [y[i] for i in range(number_of_subjects) if i not in r]
    x_tr = np.zeros(shape = ((number_of_subjects-test_size)*number_of_slide_windows, 1))
    y_tr = np.zeros(shape = ((number_of_subjects-test_size)*number_of_slide_windows, 1))
    x_te = np.zeros(shape = (number_of_slide_windows, 1))
    y_te = np.zeros(shape = (number_of_slide_windows, 1))
    
    for i, xx in enumerate(itertools.chain.from_iterable(x[1:number_of_subjects-test_size+1])):
        x_tr[i] = [xx]
        
    for i, yy in enumerate(itertools.chain.from_iterable(y_train)):
        y_tr[i] = [yy]
        
    for j in range(test_size):
        for i in range(0, number_of_slide_windows):
            x_te[test_size*j + i] = [x[0][i]]
            
    for i, yy in enumerate(y_test[0]):
        y_te[i] = [yy]
        
    return x_tr,             x_te,             y_tr,             y_te


# In[26]:


## Non-Linear Regression
# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.0001
lasso_nalpha=20
lasso_iter=5000

# set params
degree = 3
test_set_fraction = 1.0/12

# Test/train split
light_intensity = np.arange(105, 241, 3)
light_intensity = np.tile(light_intensity,(12,1))
# X_train, X_test, y_train, y_test = train_test_split(light_intensity, fft_out_max_list, test_size=test_set_fraction)
X_train, X_test, y_train, y_test = train_test_split(x = light_intensity, y = fft_out_max_list, test_size = 1)
print "X_train :", len(X_train), ", X_test :", len(X_test), ", y_train :", len(y_train), ", y_test :", len(y_test)
print

# # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
# model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,
# normalize=True,cv=5))
# model.fit(X_train,y_train)
# test_pred = np.array(model.predict(X_test))
# RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
# test_score = model.score(X_test,y_test)


# In[42]:


## try linear regression (not be used)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#calculate Mean Square Error
mse = np.sum(abs(y_pred-y_test)) / len(y_test)
mse


# In[43]:


## plot linear regression result (not be used)
x = list(itertools.islice(itertools.count(), 105, 241, 3))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, y_pred, s=10, c='b', marker="s", label='pred')
ax1.scatter(x, y_test, s=10, c='r', marker="o", label='actual')
plt.legend(loc='upper left');
plt.show()


# In[44]:


## non-linear regression
model = prep.PolynomialFeatures(degree=3)
X_tr = model.fit_transform(X_train)
X_te = model.fit_transform(X_test)

clf = linear_model.LinearRegression()
clf.fit(X_tr, y_train)
y_pred = clf.predict(X_te)

# calculate Root Mean Square Error
rmse = math.sqrt(np.sum((y_pred-y_test)**2) / len(y_test))
rmse


# In[45]:


## plot non-linear regression result
x = list(itertools.islice(itertools.count(), 105, 241, 3))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, y_pred, s=10, c='b', marker="s", label='pred')
ax1.scatter(x, y_test, s=10, c='r', marker="o", label='actual')
plt.legend(loc='upper left');
plt.show()

