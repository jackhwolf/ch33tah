import boto3
from dotenv import load_dotenv
import time 
import os 
import joblib 

load_dotenv()


class ResultsManager:

    def __init__(self):
        self.region = 'us-west-1'

    @property
    def cli(self):
        return boto3.client('s3', aws_access_key_id=os.environ.get('awsaccess'),
                            aws_secret_access_key=os.environ.get('awssecret'),)

    def create_bucket(self, bucket_name):
        ''' create new s3 bucket with (formatted) given name '''
        bucket_name = bucket_name.rstrip('/') + "-" + str(int(time.time()*1000))[-10:]
        self.cli.create_bucket(
               Bucket=bucket_name
            )
        return bucket_name

    def upload_run(self, bucket_name, run_dirname):
        ''' push the last run to s3 '''
        files = os.listdir(run_dirname)
        for f in files:
            filename = run_dirname + '/' + f
            self.cli.upload_file(
                Filename=filename,
                Bucket=bucket_name,
                Key=f)
        return 1

    def get_as_obj(self, bucket, file):
        ''' download a file as an object, without writing to disk '''
        s3_response_object = self.cli.get_object(Bucket=bucket, Key=file)
        return s3_response_object['Body'].read()

    def iterate_bucket(self, bucket):
        ''' loop through bucket with given bucket_name & yield all keys '''
        paginator = self.cli.get_paginator('list_objects')
        page_iter = paginator.paginate(Bucket=bucket)
        for page in page_iter:
            try:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key[-1] != '/':
                        yield key
            except:
                yield

    def reload_models(self, bucket):
        ''' download and restore the models '''
        tmp = '/tmp/1-ch33tah.sav'
        models = self.iterate_bucket(bucket)
        m = {}
        for model in models:
            with open(tmp, 'wb') as fp:
                fp.write(self.get_as_obj(bucket, model))
            m[model.split('.')[0]] = joblib.load(tmp)
        os.remove(tmp)
        return m
