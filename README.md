# Intro

Simple convolution model implementation in Tensorflow (v1) for
classification of two dimensional images of sign-language signs for
numbers 0-5. The model and dataset is taken from the course
["Convolutional Neural Networks"](https://www.coursera.org/learn/convolutional-neural-networks).
The dataset is not provided here, but may be downloaded by navigating
to the notebook of the assignment on Coursera. Create a directory
"datasets" in project root and put the two h5-files in this. 
Note that the code has been completely
re-written to fit within the generic project template
<https://github.com/MrGemy95/Tensorflow-Project-Template>.

# Usage

Ensure that consistent versions of the various packages required for
the code are used by generating a virtual environment and installing
pip packages (python3) listed in
[requirements.txt](https://github.com/spietz/Tensorflow-Project-Signs/blob/master/requirements.txt):
```
pip3 install -r requirements.txt
```

Begin the training by running the main script:
```
python3 train_test_signs.py -c configs/signs.json
```

Setup live monitoring of the training history output in the summary using tensorboard:
```
tensorboard --logdir experiments/signs/summary
```
In a browser then open "localhost:6006" to see the graphs.


# TODO

* print accuracy at given intervals
* ~~add tests to summary~~
* ~~show the total iteration number when continuing from a checkpoint~~
* use minibatch when testing