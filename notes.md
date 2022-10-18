# 1. Collecting data
The exploring of the data was actually a relatively quick part. For part 1 to 4, we only needed a small part of the whole COCO dataset, namely the annotations and the images. Therefore, also the loading of the data is only a few lines of code. I wanted to prepare the data as much as possible before using it in the models to save training time. So, I already converted the images to tensors, split and encoded the captions in an own and glove vocabulary and also converted them to tensors.

For the negative sampling, I used a very simple aproach, by just associating the five captions of image A with image A+1. With this, I very quickly got double amount of samples. Here, I also introduced a variable that controls, how many negative samples should be produced. The lower it was, the better the model performed. This was very liklely due to the fact that this variable also influenced the amount of negative samples in the testing set. Seperating the variables for training and testing would be an interesting task to observe in the future.

During the training of the models, I changed the sample size a lot. In the end, I used a sample size of 10.000 images, which produces 50.000 positive pairs and around 48.000 negative pairs, in total around 98.000 samples. With my 80% split, these were around 78.000 samples for training. I used this number, since the more samples I used, the better the performance got. With 78.000 samples, the training time was with around 30 minutes still reasonable.

# 2. Modeling
I split up the model in two parts. One part encoded the image, the other one encoded the caption. 

For the **captions**, I used a bidirectional LSTM. This LSTM takes embeddings as an input. First, I used randomly initilaized embeddings, which actually performed already not to bad (since I changed a lot of parameters all the time and optimized the models after changing to different initilizations of embeddings, it is hard to compare these changes. Hence *'not to bad'*). Then, I thought of initializing the embeddings with th GloVe vectors. Here, I chose 300 dimensions and 6 billion tokens. After checking, that only around 1500 words of the 48.000 sentences were labeled *unknown*, I decided that I didn't need to use GloVe with more tokens. Here, I added the tokens for padding and unknown words manually (initialized with zeros). Since, GloVe vectors are already pretrained and the rest of the model is random in the beginning, I froze their parameters in the beginning, and started to fine-tune them just after training half of the epochs. This seemed to improve the performance slightly.

For the LSTM, I only used one layer. Using more layers was having a high negative impact on the performance. In the future, it could be interesting, to only use the hidden state of specific layers instead of all, as I did.

Since I was using around 25 epochs (the loss was still going down significantly), I introduced a dropout after embedding and LSTM layer, to reduce overfitting to the training data. This helped the performance as well.

For the **images**, I used one of the CNNs of the demo sessions and adjusted some parameters. As seen in the results later, this worked actually quite well.

Both encodings were then flattened and concatenated. This was then run through multiple Linear layers and a sigmoid for the classifier. For the hidden layers, I used 2000 and 1000 dimensions. Higher and lower dimensions sizes decreased the performance.

# 3. Training/Testing
Outputs:
```
25 EPOCHS - 2438 BATCHES PER EPOCH
epoch 0, batch 2438: 0.6978
epoch 1, batch 2438: 0.6544
epoch 2, batch 2438: 0.5843
epoch 3, batch 2438: 0.5037
epoch 4, batch 2438: 0.4139
epoch 5, batch 2438: 0.3187
epoch 6, batch 2438: 0.2413
epoch 7, batch 2438: 0.1869
epoch 8, batch 2438: 0.2719
epoch 9, batch 2438: 0.1734
epoch 10, batch 2438: 0.1328
epoch 11, batch 2438: 0.1098
epoch 12, batch 2438: 0.0995
epoch 13, batch 2438: 0.0796
epoch 14, batch 2438: 0.0701
epoch 15, batch 2438: 0.0587
epoch 16, batch 2438: 0.0571
epoch 17, batch 2438: 0.0507
epoch 18, batch 2438: 0.0455
epoch 19, batch 2438: 0.0439
epoch 20, batch 2438: 0.0383
epoch 21, batch 2438: 0.0363
epoch 22, batch 2438: 0.0354
epoch 23, batch 2438: 0.0329
epoch 24, batch 2247: 0.0311
```
Testing out different thresholds for the classifier split:
```
--- THRESHOLD: 0.1 ---
Accuracy: 0.9477101845522898
Precision: 0.9799918886034878
Recall: 0.9215611492499365
F1-Score: 0.9498787918495709
--- THRESHOLD: 0.2 ---
Accuracy: 0.9535201640464799
Precision: 0.9726916317425983
Recall: 0.937703636126678
F1-Score: 0.9548772395487723
--- THRESHOLD: 0.3 ---
Accuracy: 0.9559125085440875
Precision: 0.9676896038934704
Recall: 0.9463246959280803
F1-Score: 0.9568879085622619
--- THRESHOLD: 0.4 ---
Accuracy: 0.956390977443609
Precision: 0.962146816276869
Recall: 0.9521070234113712
F1-Score: 0.9571005917159764
--- THRESHOLD: 0.5 ---
Accuracy: 0.9565276828434723
Precision: 0.957144788427741
Recall: 0.9568860656845519
F1-Score: 0.9570154095701541
--- THRESHOLD: 0.6 ---
Accuracy: 0.9566643882433357
Precision: 0.9521427605786129
Recall: 0.9617643042468933
F1-Score: 0.9569293478260869
--- THRESHOLD: 0.7 ---
Accuracy: 0.9553656869446343
Precision: 0.9449776936595917
Recall: 0.9660033167495854
F1-Score: 0.9553748376956195
--- THRESHOLD: 0.8 ---
Accuracy: 0.9535201640464799
Precision: 0.9357847776125456
Recall: 0.9712361442402133
F1-Score: 0.9531809418892867
--- THRESHOLD: 0.9 ---
Accuracy: 0.9479835953520164
Precision: 0.918886034879005
Recall: 0.9768611670020121
F1-Score: 0.9469871125043539
```

# 4. Evaluation
My model performed actually really well. A threshold somewhere between 0.4 and 0.5 preduced the best performance. The precision lies around 96% while the recall is around 95.5%. This shows that the model is really good in finding all correct captions, but also in assigning *only* the correct captions.

One explanation for this in my eyes surprsingly good performance may be the used data and especially the negative sampling. I used a rather naive approach to create negative samples. In most of the cases, the 'negative' caption will describe somthing completely different than the image and makes it therefore easier for the model to differentiate between correct and these wrong captions. It would be more interesting to have closer negative samples like e.g. 'A dog is playing' instead of 'A cat is playing'. This will probably challenge the model a lot more.

Since, the model already makes few errors, it is hard for me, to draw specific conclusions, how I can improve it. Still, there are some general ways to to try out and compare with the current approach. First of all, we could train more epochs. Currently, the loss is still going down after 25 epochs. So, it would still be possible, to add a few more epochs. Here, we of course need to be careful, to not overfit the model to the training data. Adding more dropout would be for example an option.

Furthermore, it would also be possible to use more layers in the LSTM. As already mentioned above, just concatenating all of these layer worsened the performance significantly. But it would be worth to have a closer look and for example use only specific layers or different ways to combine them (like averaging, weighted averages, sums, ...)

Interesting would also be to test different methods to combine the image and caption representation. Currently, I am using a concatenation, but there may be other better options. One may even using the image representation as input for the LSTM of the caption representation.