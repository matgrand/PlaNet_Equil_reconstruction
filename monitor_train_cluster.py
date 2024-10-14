import os
from time import sleep

for _ in range(23*60*2):
    dir = "jobs"
    #read all the files in the directory
    files = os.listdir(dir)
    #sort the files by the time they were created
    files.sort(key=lambda x: os.path.getmtime(dir + "/" + x))
    #get the last file
    last_file = files[-1]
    #print the last file
    print(last_file)
    #print everything in the last file
    with open(dir + "/" + last_file) as f:
        print(f.read())