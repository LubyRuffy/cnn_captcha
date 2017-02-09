#!/usr/local/env python
#-*- coding: UTF-8 -*-

import os
import sys
import random
import time


def list_2_fullpath(path):
	newpath = path[0:-4] + '_fullpath' + path[-4:]
	newfile = open(newpath, 'w')
	for line in open(path):
		line = line.strip()
		#os.system('pwd ')
		cur_path = sys.path[0]
		newline = cur_path + '/' + line
		newfile.write(newline + '\n')




def split_list_0(path):
	lst = []
	ftrain = open(path+'-8.txt', 'w')
	ftest = open(path+'-2.txt', 'w')

	for line in open(path):
		line = line.strip()
		lst.append(line)
	lstlen = len(lst)
	print 'lstlen=',lstlen
	loop = random.randint(10,1000)
	print 'loop=',loop
	for i in range(0, loop):
		random.shuffle(lst)
	train_num = int(0.8*lstlen)
	train_lst = lst[0:train_num]
	print 'train_lst=',len(train_lst)
	for e in train_lst:
		ftrain.write(e + '\n')

	test_lst = lst[train_num:]
	print 'test_lst=',len(test_lst)
	for e in test_lst:
		ftest.write(e + '\n')
	print 'end'


def split_list(path):
	lst = []
        ftrain = open(path[0:-4]+'-8.txt', 'w')
        ftest = open(path[0:-4]+'-2.txt', 'w')

	for line in open(path):
		line = line.strip()
		lst.append(line)

        lstlen = len(lst)
	print 'lstlen=',lstlen

        random.seed(time.time())
        random.shuffle(lst)

        train_num = int(0.8*lstlen)
	train_lst = lst[0:train_num]

        print 'train_lst=',len(train_lst)
	for e in train_lst:
		ftrain.write(e + '\n')

	test_lst = lst[train_num:]
	print 'test_lst=',len(test_lst)
	for e in test_lst:
		ftest.write(e + '\n')



def split_list_fromtxt(pathall, pathtrain, pathtest):
	lstall = []
	lsttrain = []
	lsttest = []


        for line in open(pathall):
            line = line.strip()
            lstall.append(line)

        for line in open(pathtrain):
            line = line.strip()
            lsttrain.append(line)

        for line in open(pathtest):
            line = line.strip()
            lsttest.append(line)


        print 'len of all',len(lstall)
        print 'len of train',len(lsttrain)
        print 'len of test',len(lsttest)

        #x = raw_input('pause\n')

	ftrain = open(pathall+'-8.txt', 'w')
	ftest = open(pathall+'-2.txt', 'w')


        cntall = 0
        cnttrain = 0
        cnttest = 0

        for e in lstall:

            ee = e[0:-8]+'.png'
            ee1 = e[0:-8]+'.jpg'

            print ee
            print 'cntall=',cntall

            if ee in lsttrain or ee1 in lsttrain:
                ftrain.write(e+'\n')
                cnttrain = cnttrain + 1

            if ee in lsttest or ee1 in lsttest:
                ftest.write(e+'\n')
                cnttest = cnttest + 1

            if ee not in lsttrain and ee not in lsttest:
                print e
                print ee
            cntall = cntall + 1

        print 'cntall:',cntall
        print 'cnttrain:',cnttrain
        print 'cnttest:',cnttest


def random_list(path):
    if path[-4:] != '.txt':
        return None

    if not os.path.exists(path):
        return None

    lst = []
    for line in open(path):
        lst.append(line.strip())

    print 'lst len =',len(lst)

    random.seed(time.time())

    random.shuffle(lst)

    newfile = path[0:-4] + '-random' + path[-4:]

    fnewfile = open(newfile, 'w')

    for e in lst:
        fnewfile.write(e+'\n')

    fnewfile.close()

    print 'random over\n'



if __name__ == '__main__':
	print 'main'
	arg1 = sys.argv[1]
	print 'arg1=',arg1
	#list_2_fullpath(arg1)
	split_list(arg1)
        #random_list(arg1)


        #split_list_fromtxt('./txt_list.txt', './train_list-org.txt', './test_list-org.txt')
        pass
