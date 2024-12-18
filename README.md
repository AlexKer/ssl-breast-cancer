# Organization 

The code repository for our experimentation is divided into the SSL methods we attempted and the data processing.

There are five folders-- one for data, one for the combined MoCo and Shuffle & Learn model, another for just Shuffle & Learn, SparK, and SWaV code. 


## Dataset 

To access the ISPY1 MRI Dataset, visit this webpage https://www.cancerimagingarchive.net/collection/ispy1/. You may need to download a NBIA Data Retriever to download the full dataset. The data is around 80GB, and around 20GB after preprocessing. 


## Usage 

### Preprocessing 

Run the preprocessing scripts (Mac or Windows) in the data/preprocessing directory. We also included a script for loading a dataset from Duke that can also be used for experimentation. When running the preprocessing script, you should get an output of .npy files according to the patient data and a .json file that describes the transformed data. Both are critical for further training.

### MoCo and Shuffle 

First, run `trainMedical.ipynb` or `main.py` to pre-train the model on MoCo. This will pre-train using a combined MoCo + Shuffle & Learn method, if one wants to pre-train on just MoCo then set the shuffle_prob argument to 0 in the `main` function. Then, run `downstreamMoCo.ipynb` or `downstreamMoCoShuffle.ipynb` notebooks to fine-tune the model (according to your pre-train method) on downstream tasks. 

### SWaV

First, pre-train the model by running `swav_final.py` or `swav_light.py`--the latter has been slightly modified to run faster but this may compromise on performance. Afterward, finetune on downstream tasks by running `finetune_dwnstm.py`.

### SparK

Similar to other workflows, running `main.py` should run the full pre-training and fine-tuning procedure. 
