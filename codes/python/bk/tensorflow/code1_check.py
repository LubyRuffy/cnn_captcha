#!/usr/bin/python
#coding=utf-8

import os,sys
import urllib
import string
import cv2

def check():
    lst = []
    for line in open('code1_check.txt', 'r'):
        lst.append(line.strip())

    sample_cnt = len(lst)

    for sam_idx in range(len(lst)):
        line = lst[sam_idx]
        v = string.split(line, ' ')
        v_len = len(v)
        img0 = cv2.imread(v[0])
        if img0 != None:
            img00 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            new_name = './out1/' + v[1]+v[2]+v[3]+v[4]+'.jpg'
            cv2.imwrite(new_name, img00)
            print new_name
            

if __name__ == '__main__':
    print "Main\n"
    path = sys.path[0]
    os.chdir(path)
    check()