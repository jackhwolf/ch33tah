# ch33tah
The fastest way to cheat at machine learning

## inspiration
For HackIllinois' 2020 open source challenge I wanted to make an open source project of my own that would make 
machine learning much easier for those who don't code that much. Someone like a chemist or lab 
scientist, who has data, tech skill, but maybe not ML or even much programming skill. 

## how it works
`Ch33tah` is just one big grid-search cross-validation algorithm to determine a well-suited model and
set of hyperparameters. All that the user has to provide are the features, labels, whether it is a classification 
or regression problem, and a name for this test. I've used `dask` and `Distributed` to make this grid search CV 
embarassingly parallel so that (best case, if the client's machine can handle it) every CV fold of every hyperparameter
set for every model runs in parallel. This can be scaled up and down by adding `dask-worker`s. 
See [this link](https://stsievert.com/blog/2016/09/09/dask-cluster/) for more notes on `dask`. Once this finishes, the user
will get back a report detailing the best performing hyperparameters for each model tested and their performances. This report 
will also be uploaded to S3 so that the user can use these models for inference in the future. This is where `Ch33taRetest` 
comes in. Users provide a link to the s3 bucket containing the model weight files, and then can give the module new data to predict


## License: MIT

Jack Wolf