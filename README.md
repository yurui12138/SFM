## SFM

#### Sentence Pair Modeling Based on Semantic Feature Map for Human Interaction with IoT Devices
Rui Yu, Wenpeng Lu, Huimin Lu, Shoujin Wang, Fangfang Li, Xu Zhang, Jiguo Yu  
The paper has been accepted by International Journal of Machine Learning and Cybernetics.

#### Prerequisites
python 3.6  
numpy==1.16.4  
pandas==0.22.0  
tensorboard==1.12.0  
tensorflow-gpu==1.12.0  
keras==2.2.4  
gensim==3.0.0  

#### Example to run the codes
Run SFM.py  
`python3 SFM.py`  

#### Dataset
We used two datasets: BQ & LCQMC.  
1. "The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification", https://www.aclweb.org/anthology/D18-1536/.  
2. "LCQMC: A Large-scale Chinese Question Matching Corpus", https://www.aclweb.org/anthology/C18-1166/.

### Note
Due to the differences between the two data sets, some parameters adopted by SFM are different. Therefore, we provide two versions of the code for the two data sets.
