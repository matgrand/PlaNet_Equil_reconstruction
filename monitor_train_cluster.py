import os
from time import sleep
for _ in range(23*60*2): #run for 23 hours
    dir = "jobs"
    files = os.listdir(dir) #list all the files in the directory
    files.sort(key=lambda x: os.path.getmtime(dir + "/" + x)) #sort the files by the time they were last modified
    with open(dir + "/" + files[-1]) as f: print(f.read()) #print everything in the most recently modified file
    print(files[-1].split(".")[0]) #print the name of the most recently modified file
    for i in range(30): 
        print(f" [{'='*i}{' '*(29-i)}]", end="\r") #print a progress bar
        sleep(1)