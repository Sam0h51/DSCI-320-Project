import numpy.random as rn
import numpy as np
import math
import csv

wrtfile = open('DesiredColumns.csv', 'w')
trainfile = open('TrainingData.csv', 'w')
testfile = open('TestingData.csv', 'w')

with open('breastcancerdat.csv', 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        if(row[1] == 'M'):
            outstr = str(row[2]) + ',' + str(row[3]) + ',' + str(row[4]) + ',' + str(row[5]) + ',' + str(row[6]) + ',' + str(row[7]) + ',' + str(row[8]) + ',' + str(row[9]) + ',' + str(row[10]) + ',' + str(row[11]) + ',' + str(1) +',\n'
            print(outstr)
            wrtfile.write(outstr)
            s = rn.rand()
            if(s > 0.2):
                trainfile.write(outstr)
            else:
                testfile.write(outstr)
        else:
            outstr = str(row[2]) + ',' + str(row[3]) + ',' + str(row[4]) + ',' + str(row[5]) + ',' + str(row[6]) + ',' + str(row[7]) + ',' + str(row[8]) + ',' + str(row[9]) + ',' + str(row[10]) + ',' + str(row[11]) + ',' + str(0) +',\n'
            print(outstr)
            wrtfile.write(outstr)
            s = rn.rand()
            if(s > 0.2):
                trainfile.write(outstr)
            else:
                testfile.write(outstr)

testfile.close()
trainfile.close()
wrtfile.close()























