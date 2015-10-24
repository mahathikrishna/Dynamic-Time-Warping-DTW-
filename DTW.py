import numpy as np
import math
import decimal
import os
import matplotlib.pyplot as plt

from decimal import *
from numpy import *

def dytiw(x,y):
    m = size(x)
    n = size(y)
    s = (m+1,n+1)
    f = (m,n);
    Dg = np.zeros(s)

    cost = np.zeros(s);
    for i in range(m):
        for j in range(n):
            a = x[0,i] - y[0,j];
            b = math.fabs(a);
            cost[i,j] = b;
            cost[0,0] = 0;


    for i in range(m):
        Dg[i:,0] = i;
    for j in range(n):
        Dg[0,j:] = j;

    for i in range(m):
        for j in range(n):
            Dg[i+1,j+1] = cost[i,j] + min(Dg[i,j+1],Dg[i+1,j],Dg[i,j]);
          


    Distgrid = np.zeros(f)
    for i in range(m):
        for j in range(n):
            Distgrid[i,j] = Dg[i+1,j+1];



    rw ,c = Distgrid.shape
    ##print rw
    ##print c

    k = (rw,c);
    l = (rw+1,c+1);
    D = np.zeros(l)
    D[0, 1:] = inf
    D[1:, 0] = inf
    D[0,0] = 0;

    for i in range(rw):
        for j in range(c):
             D[i+1, j+1] = Distgrid[i,j];


    ### TRACEBACK

    i, j = array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        tb = argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))
        if (tb == 0):
            i = i - 1
            j = j - 1
        elif (tb == 1):
            i = i - 1
        elif (tb == 2):
            j = j - 1        
        p.insert(0, i)
        q.insert(0, j)

    Parr = array(p);
    Qarr = array(q);


    P = Parr.reshape(1,len(Parr));
    Q = Qarr.reshape(1,len(Qarr));


    ##print P
    ##print Q


    diag = [];
    diag = Distgrid.diagonal();
    ##print diag;

    min_cost = Distgrid[rw-1,c-1];
    ##print min_cost

    count = 0;
    plen = size(P);
    qlen = size(Q);

    for i in range(plen):
        if (P[0,i] == Q[0,i]):
            count = count+1;

    ##print count;

##    dissimilarity = np.array(8)
    dissimilarity = (count * min_cost)/100 ;
##    dis_vect = dissimilarity;
    print dissimilarity
    
    return dissimilarity



norm1 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV1");
print size(norm1)
n=0
params1 = 0
X1 = np.zeros((316,));
for filenames in norm1:
    params1 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV1\\"+params1,float,-1," ");
    X1 = np.add(X1,X)
m = size(X1)
TR1 = X1/15;
TR1 = TR1.reshape(1,len(TR1))

norm2 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV2");
n=0
params2 = 0
X2 = np.zeros((316,));
for filenames in norm2:
    params2 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV2\\"+params2,float,-1," ");
    X2 = np.add(X1,X)
m = size(X2)
TR2 = X2/15;
TR2 = TR2.reshape(1,len(TR2))

norm3 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV3");
n=0
params3 = 0
X3 = np.zeros((316,));
for filenames in norm3:
    params3 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV3\\"+params3,float,-1," ");
    X3 = np.add(X3,X)
m = size(X3)
TR3 = X3/15;
TR3 = TR3.reshape(1,len(TR3))

norm4 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV4");
n=0
params4 = 0
X4 = np.zeros((316,));
for filenames in norm4:
    params4 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV4\\"+params4,float,-1," ");
    X4 = np.add(X4,X)
m = size(X4)
TR4 = X4/15;
TR4 = TR4.reshape(1,len(TR4))


norm5 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV5");
n=0
params5 = 0
X5 = np.zeros((316,));
for filenames in norm5:
    params5 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV5\\"+params5,float,-1," ");
    X5 = np.add(X5,X)
m = size(X5)
TR5 = X5/15;
TR5 = TR5.reshape(1,len(TR5))

norm6 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV6");
n=0
params6 = 0
X6 = np.zeros((316,));
for filenames in norm6:
    params6 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV6\\"+params6,float,-1," ");
    X6 = np.add(X6,X)
m = size(X6)
TR6 = X6/15;
TR6 = TR6.reshape(1,len(TR6))

norm7 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV7");
n=0
params7 = 0
X7 = np.zeros((316,));
for filenames in norm7:
    params7 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV7\\"+params7,float,-1," ");
    X7 = np.add(X7,X)
m = size(X7)
TR7 = X7/15;
TR7 = TR7.reshape(1,len(TR7))

norm8 = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\FeatureVectors_Train\FV8");
n=0
params8 = 0
X8 = np.zeros((316,));
for filenames in norm8:
    params8 = filenames;
    X= np.fromfile("FeatureVectors_Train\FV8\\"+params8,float,-1," ");
    X8 = np.add(X8,X)
m = size(X8)
TR8 = X8/15;
TR8 = TR8.reshape(1,len(TR8))



test = os.listdir("C:\Users\Mahathi\Desktop\SEM 3\BIOMETRICS\TERM PROJECT\PROJECT\Features_test");
dis_vect = [];

for filenames in test:
        files = "Features_test\\"+filenames;
        f = [line.rstrip('\n') for line in open(files)];
        fv = array(f);
        FV = fv.reshape(1,len(fv))
        testvect = FV.astype(np.float)
        x1 = dytiw(TR1,testvect)
        dis_vect.append(x1);
        x2 = dytiw(TR2,testvect)
        dis_vect.append(x2);
        x3 = dytiw(TR3,testvect)
        dis_vect.append(x3);
        x4 = dytiw(TR4,testvect)
        dis_vect.append(x4);
        x5 = dytiw(TR5,testvect)
        dis_vect.append(x5);
        x6 = dytiw(TR6,testvect)
        dis_vect.append(x6);
        x7 = dytiw(TR7,testvect)
        dis_vect.append(x7);
        x8 = dytiw(TR8,testvect)
        dis_vect.append(x8);
##print dis_vect
####print dis_vect.shape
##print len(dis_vect)

diss_arr = array(dis_vect)
diarray = diss_arr.reshape((79,8))
##print diarray
##print size(diarray)
##print diarray.shape
length = size((diarray[:,1]))
##print length
##print diarray[0,:]

ind = np.zeros((79,1),dtype = int32);
array_min = np.zeros((79,1),dtype = int32);
for i in range(79):
    array_min[i,0] = min(diarray[i,:])
    ind[i,0] = int(argmin((diarray[i,:])))+1;
##print array_min
print ind


train_label11 = np.array([1,1,1,1,1,1,1,1,1]);
train_label1 = train_label11.reshape(len(train_label11),1);
t1 = size(train_label1);
print t1
print train_label1[8,0]
train_label22 = np.array([2,2,2,2,2,2,2,2,2]);
train_label2 = train_label22.reshape(len(train_label22),1);
t2 = size(train_label2);
train_label33 = np.array([3,3,3,3,3,3,3,3,3]);
train_label3 = train_label33.reshape(len(train_label33),1);
t3 = size(train_label3);
train_label44 = np.array([4,4,4,4,4,4,4,4,4]);
train_label4 = train_label44.reshape(len(train_label44),1);
t4 = size(train_label4);
train_label55 = np.array([5,5,5,5,5,5,5,5,5,5]);
train_label5 = train_label55.reshape(len(train_label55),1);
t5 = size(train_label5);
train_label66 = np.array([6,6,6,6,6,6,6,6,6,6,6]);
train_label6 = train_label66.reshape(len(train_label66),1);
t6 = size(train_label6);
train_label77 = np.array([7,7,7,7,7,7,7,7,7,7,7]);
train_label7 = train_label77.reshape(len(train_label77),1);
t7 = size(train_label7);
train_label88 = np.array([8,8,8,8,8,8,8,8,8,8,8,8]);
train_label8 = train_label88.reshape(len(train_label88),1);
t8 = size(train_label8);

##train_label = np.vstack((train_label1,train_label2,train_label3,train_label4,train_label5,train_label6,train_label7,train_label8));
##
##print train_label
##print train_label[0,0]

##v = size(train_label)
##b = size(ind)
##print b


ind1 = np.zeros((9,1), dtype = int32);
for i in range(8):
    ind1[i,0] = ind[i,0];
print ind1
i1 = size(ind1)
print i1

ind2 = np.zeros((9,1),dtype = int32);
for i in range(9):
    ind2[i,0] = ind[i+10,0];
print ind2
i2 = size(ind2)

ind3 = np.zeros((9,1),dtype = int32);
for i in range(9):
    ind3[i,0] = ind[i+19,0];
print ind3
i3 = size(ind3)

ind4 = np.zeros((9,1),dtype = int32);
for i in range(9):
    ind4[i,0] = ind[i+28,0];
print ind4
i4 = size(ind4)

ind5 = np.zeros((10,1),dtype = int32);
for i in range(10):
    ind5[i,0] = ind[i+37,0];
print ind5
i5 = size(ind5)

ind6 = np.zeros((10,1),dtype = int32);
for i in range(10):
    ind6[i,0] = ind[i+46,0];
print ind6
i6 = size(ind6)

ind7 = np.zeros((11,1),dtype = int32);
for i in range(11):
    ind7[i,0] = ind[i+56,0];
print ind7
i7 = size(ind7)

ind8 = np.zeros((12,1),dtype = int32);
for i in range(12):
    ind8[i,0] = ind[i+67,0];
print ind8
i8 = size(ind8)



count1 = 0;
for i in range(i1):
        if (ind1[i,0] == train_label1[i,0]):
            count1 = count1+1;
c1 = count1
print c1 

count2 = 0;
for i in range(i2):
        if (ind2[i,0] == train_label2[i,0]):
            count2 = count2+1;
c2 = count2
print c2

count3 = 0;
for i in range(i3):
        if (ind3[i,0] == train_label3[i,0]):
            count3 = count3+1;
c3 = count3
print c3

count4 = 0;
for i in range(i4):
        if (ind4[i,0] == train_label4[i,0]):
            count4 = count4+1;
c4 = count4
print c4

count5 = 0;
for i in range(i5):
        if (ind5[i,0] == train_label5[i,0]):
            count5 = count5+1;
c5 = count5
print c5

count6 = 0;
for i in range(i6):
        if (ind6[i,0] == train_label6[i,0]):
            count6 = count6+1;
c6 = count6
print c6

count7 = 0;
for i in range(i7):
        if (ind7[i,0] == train_label7[i,0]):
            count7 = count7+1;
c7 = count7
print c7

count8 = 0;
for i in range(i8):
        if (ind8[i,0] == train_label8[i,0]):
            count8 = count8+1;
c8 = count8
print c8

##from __future__ import division
misclass1 = (float(c1)/9)*100;
print "misclass1 = %f" % misclass1
accuracy1 = 100-misclass1
print "accuracy = %f" % accuracy1
misclass2 = (float(c2)/9)*100;
print "misclass2 = %f" % misclass2
accuracy2 = 100-misclass2
print "accuracy = %f" % accuracy2
misclass3 = (float(c3)/9)*100;
print "misclass3 = %f" % misclass3
accuracy3 = 100-misclass3
print "accuracy = %f" % accuracy3
misclass4 = (float(c4)/9)*100;
print "misclass4 = %f" % misclass4
accuracy4 = 100-misclass4
print "accuracy = %f" % accuracy4
misclass5 = (float(c5)/10)*100;
print "misclass5 = %f" % misclass5
accuracy5 = 100-misclass5
print "accuracy = %f" % accuracy5
misclass6 = (float(c6)/10)*100;
print "misclass6 = %f" % misclass6
accuracy6 = 100-misclass6
print "accuracy = %f" % accuracy6
misclass7 = (float(c7)/11)*100;
print "misclass7 = %f" % misclass7
accuracy7 = 100-misclass7
print "accuracy = %f" % accuracy7
misclass8 = (float(c8)/12)*100;
print "misclass8 = %f" % misclass8
accuracy8 = 100-misclass8
print "accuracy = %f" % accuracy8

accuracy = ([accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7,accuracy8])
class_label = ([1,2,3,4,5,6,7,8])

plt.plot(class_label,accuracy)
plt.xlabel('CLASS LABELS')
plt.ylabel('ACCURACY')
plt.show()
