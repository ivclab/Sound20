# Sound20 

<!-- Maybe add some segments of audio sound HERE --> 

This is a dataset including 20 animal and instrument sounds. This dataset is constructed using [Animal Sound Data](http://alumni.cs.ucr.edu/~yhao/animalsoundfingerprint.html) and [Instrument Data](https://drive.google.com/file/d/0B8x0IeJAaBccUk1xMDBNTzNFb0E/view). Each audio is split into multiple samples, and we make sure that samples in Train, Validation, Test sets are disjoint and separated.  

## Data 

You can find spectrograms of samples in `spectrogram_data`.

### Statistics 

Split name | Train set | Validation set | Test set 
---------- | --------- | -------------- | -------- 
Number of samples | 16,636 | 3,249 | 3,727 

### Labels 

Each sample is assigned to one of the 20 labels, which include sounds of drums, guitars, frogs, and insects. 

Label | Description 
----- | -----------
0 | Drum_FloorTom
1 | Drum_HiHat 
2 | Drum_Kick 
3 | Drum_MidTom
4 | Drum_Ride 
5 | Drum_Rim
6 | Drum_SmallTom
7 | Drum_Snare
8 | Guitar_3rd_Fret
9 | Guitar_9th_Fret 
10 | Guitar_Chord1
11 | Guitar_Chord2
12 | Guitar_7th_Fret 
13 | Bufo_Alvarius (a type of toads)
14 | Bufo_Canorus (a type of toads)
15 | Pseudacris_Crucifer (a type of frogs)
16 | Allonemobius_Allardi (a type of crickets)
17 | Anaxipha_Exigua (a type of crickets)
18 | Amblycorypha_Carinata (a type of katydid)
19 | Belocephalus_Sabalis (a type of katydid)

## Usage 

### Loading data 

You can load this dataset using Python with Numpy. 

```python
import numpy as np 
x_train = np.load('spectrogram_data/train_X.npy')
y_train = np.load('spectrogram_data/train_Y.npy')
```

### Evaluate using CNNs 

We conduct two experiments on this dataset using **LeNet** and **VGG\_F** network structures. To run the experiments, please using the following commands. 

For **LeNet**, use 
```
sh scripts/run_LeNet.sh 
```

For **VGG_F**, use 
```
sh scripts/run_VFF_F.sh 
```
Note that each script will run training and testing procedures and store the Train, Val, Test accuracy. 


### Experimental results 

The experimental results using **LeNet** and **VGG\_F** network structure. 

Network structure | Testing Accuracy 
----------------- | ----------------
LeNet | 78.07%
VGG_F | 79.15%

### References

[Recognizing Sounds (A Deep Learning Case Study)](https://medium.com/@awjuliani/recognizing-sounds-a-deep-learning-case-study-1bc37444d44d)

[Animal Sound Data](http://alumni.cs.ucr.edu/~yhao/animalsoundfingerprint.html)
