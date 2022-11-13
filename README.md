# lt2326-assignment_1
> Part of the Master in Language Technology at the University of Gothenburg
>
> **Course:** Machine learning for statistical NLP: introduction (LT2326)
>
> **Assignment:** Assignment 1

## Introduction
Image captioning is an important problem, as it allows both for people with visual impairments and for people with low internet bandwidth to be able to fully access the information available online. We might want to be able to verify the accuracy of these captions.

For this assignment we won’t be working directly on the captioning itself and will focus on the verification part. The idea is that, given an image and a piece of text, you will build a system that rates whether the text is a description of the image or not.
 
## Preparation
Install pycocotools into your mltgpu account and wherever else you may need it: `pip3 install --user pycocotools`

You will create a new Jupyter notebook.  You can also borrow code from any of the demo notebooks.  You will also create a Python module that will be invoked from the notebook.

Write any notes in a markdown file. 

## Part 1: collecting training data (8 points)

Every image in the COCO 2017 dataset has five captions.  In the Jupyter notebook, collect a reasonable random sample of training items from the COCO data structure using the pycocotools Python module.  You may have to do some exploration of the structure of the dataset in order to identify the captions.  Split the data you select into training, validation, and testing datasets.  You may wish to save your splits so that you should have to go through the process every single time you test your code.  The actual numbers are up to you, and you will likely change them during development.   Justify in your notes why you chose the sample size that you did. 

You will use the training captions from the dataset corresponding to each image, meaning that every image you select will produce five image/sample pairs.

Explain your process of exploring the data and any observations about it you might have in your notes.

## Part 2: modeling (15 points)
You should implement a model in PyTorch that takes your representations of the texts and of the images and outputs whether the text describes or not the image as a probability (meaning in the interval from greater than 0 to exactly 1). Don’t forget to save your model(s) for easy reproducibility/testing and to provide a way to save and load your models.  Make sure that your code can be run on GPU or CPU on mltgpu.

For this section you will have to describe your model, as well as any design choices you made in the markdown file. Remember that we will grade based on a “reasonable effort” basis.

Hint – You are free to determine the architecture of your model. Having said that, you will most likely have to add separate layers to process the images (which will likely involve one or more convolutional layers) and the text. In this case, the output of these layers should then be merged via some merging function (concatenation, multiplication, or anything else you can think of) and fed to the rest of your network.

Make sure you also write code that allows a trained model to be tested without an actual training loop.

The model code and the training loop should appear in a separate module file that is invoked from the notebook.  You should also describe in the comments of the module file how to invoke your module, the ideal being properly documenting your classes and functions (unlike what we have been doing in class ;) ).  We may invoke the code from another notebook.

## Part 3: training and testing (3 points)

Write code in the notebook that actually trains your models using the code you wrote in part 2.  You may wish to save the current model every few epochs ("checkpointing") so that you can resume training in case you must interrupt or the system crashes.  Print the loss on the training and validation sets every few epochs.

Finally, run the code on your testing set and compute any accuracy statistics you might choose.  (There is more than one way of doing this.)

You will submit your best model.  (Make sure you clean up after yourself in deleting old models and checkpoints.)

## Part 4: evaluation and error analysis (9 points)
How did your model fare? What kind of mistakes does it tend to make? How could you improve it?

Here we want to know the performance of your model and to check which errors it makes while classifying. This “error analysis” can be qualitative and/or quantitative (ideally, it would have a bit of both).

Remember, we are grading on a “reasonable effort” basis. This means that even if your model did not reach a high performance, we still want to see that you were able to realize an error analysis and that you have ideas on how the model could be improved.

Any explorations you share with us can be part of the notebook, and any observations can go in the markdown file.

## Part Bonus: negative sampling (15 points)
Find a way to implement negative sampling in training on the dataset -- that is, sometimes also giving the model incorrect captions, and then rewarding/punishing the model accordingly.  This can be done a number of ways, depending on how you define "incorrect" and how you calculate the loss.  Incorporate this process into the above code in a manner that makes it possible to turn negative sampling on or off. 

In the markdown file, write up how you augmented your code to include negative sampling and any observations on what effect it had, if any.

You can get partial credit for this bonus for a good-faith attempt at it as long as you document what you tried, even if you did not finish it.
