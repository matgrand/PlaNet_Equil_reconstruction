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
    #wait for 30 seconds
    sleep(30)
