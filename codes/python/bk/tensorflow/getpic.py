#!/usr/bin/python
#coding=utf-8

import os,sys
import urllib  

def main():
    try:
        f = open("111.txt","w")
    except IOError,msg:
        print "file not exists\n"
    finally:
        pass
    '''    
    try:
        f.close()
    except IOError,msg:
        print "file not exists\n"
    finally:
        pass
    '''

'''
if __name__ == '__main__':
    main()
    pass
'''

print "Main\n"
path = sys.path[0]
os.chdir(path)



url = "https://www.ed3688.com/sb2/me/generate_validation_code.jsp"
idx=2000
filename = ''
while(idx<=40000):
    web = urllib.urlopen(url)
    jpg = web.read()
    try:
        filename="./code1/{0:06d}.jpg".format(idx)
        print filename
        File = open(filename,"wb" )
        File.write( jpg)
        File.close()
        idx = idx+1
    except IOError:
        print("error\n")

print('Pic Saved!')   