# ch33tah
The fastest way to cheat at machine learning

## inspiration
For HackIllinois' 2020 open source challenge I wanted to make an open source project of my own that would make 
machine learning much easier for those who don't code. The idea in my head was someone like a chemist or lab 
scientist, who has data, tech skill, but maybe not ML or even much programming skill. With a clone of `ch33tah` and
a valid AWS account, they can have multiple ML solutions at their fingertips fast.

## How it works
This is, at the core, just one big grid-search cross-validation algorithm to determine a well-suited model and
set of hyperparameters. All that the user has to provide are the features, labels, whether it is a classification 
or regression problem, and a name for this test. 
But, heres the catch for them: I have *fully* distributed this algorithm over all possible cores on the host server. 
That means that not only is every set of hyperparameters being tested at the same time, but so is each CV fold for 
each set. This leads to a *fast* system that flies through the model and hyperparameter space to give the user the 
best ML experience they can get. 
After all models have been trained and evaluated, the top performing models of each category are selected. From there, 
those models are uploaded to an s3 bucket of the same name as the test. This is where the `Ch33taRetest` class comes in. 
Users can give this class a path to an s3 bucket containing models from a previous run, and `ch33tah` will load the models 
in and allow you to easily make and store predictions.

## License: MIT

Jack Wolf