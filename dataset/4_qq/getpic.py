#!/usr/bin/python
#coding=utf-8

import os,sys
import urllib
import string




def get_file_cnt(path):
    lst = os.listdir(path)
    return len(lst)               #输出结果
    
    
def get_pics(path):  
    #url = "https://www.ed3688.com/sb2/me/generate_validation_code.jsp"
    url = "http://captcha.qq.com/getimage?0.6939826908284301"
    url = "https://passport.baidu.com/cgi-bin/genimage?njGc106e28b8592de95022a1477de016713670aa6064e010018&v=1478741328830"
    idx = get_file_cnt(path) - 2 + 1
    filename = ''
    while(True or idx > 60000):
        web = urllib.urlopen(url)
        jpg = web.read()
        try:
            filename=path + "/{0:06d}.jpg".format(idx)
            print(filename)
            File = open(filename, "wb")
            File.write(jpg)
            File.close()
            idx = idx+1
        except IOError:
            print("error\n")
            
            
def get_label(path):
    lst = []
    for e in open(path, 'r'):
        lst.append(e.strip())
    print(len(lst))
    
    newpath = path[0:-4] + '_label.txt'
    newf = open(newpath, 'w')
    for e in lst:
        v = e.split(' ')
        newline = v[0]
        newline = newline + ' ' + v[1][-5:-1]
        print(newline)
        newline1 = v[0]+' ' + str(ord(v[1][-5])-97) + ' ' + str(ord(v[1][-4])-97) + ' ' + str(ord(v[1][-3])-97) + ' ' + str(ord(v[1][-2])-97) + '\n'
        newf.write(newline1)
    
    newf.close()
    
    

if __name__ == '__main__':
    print("Main\n")
    print(os.getcwd())
    path = sys.path[0]
    #os.chdir(path)
    #get_pics('./yzm')
    get_label('./4_qq_res-all.txt')
    pass


