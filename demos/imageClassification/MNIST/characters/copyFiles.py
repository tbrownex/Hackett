from google.cloud import storage
import subprocess
import sys
import os

def formatCmd(file, config):
    cmd = "gsutil cp "+"gs://"+config["bucketName"]
    cmd += "/"+file
    cmd += " /tmp/"
    return cmd

def checkFile(file):
    ''' "file" has extra folder name on the cloud storage that needs to be removed '''
    loc = file.find("/")
    file = file[loc+1:]
    file = "/tmp/"+file
    exists = os.path.isfile(file)
    return exists

def copyFiles(config):
    ''' We store the Enhanced MNIST data on GCloud Storage. Copy it to /tmp to be processed '''
    client = storage.Client()
    bucket = client.get_bucket(config["bucketName"])
    for file in config["fileNames"]:
        exists = checkFile(file)
        if exists:
            pass
        else:
            cmd = formatCmd(file, config)
            result = subprocess.call(cmd, shell=True)
            if result != 0:
                print("CMD failed...aborting")
                print(cmd)
                sys.exit()