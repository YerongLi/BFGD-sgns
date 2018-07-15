## Usage

- Clone **BFGD-sgns** repostirtory:

git clone https://github.com/YerongLi/BFGD-sgns.git
cd BFGD-sngs

- Download [enwik9](http://mattmahoney.net/dc/enwik9.zip) dataset and preprocess raw data with Perl script [main_.pl](main_.pl). 

```
wget http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
mkdir data
perl main_.pl enwik9 > data/enwik9.txti
cd data
sed -i 's/\./\n/g' enwik9.txt
jupyter notebook enwik_experiments.ipynb  # for training and illustration
jupyter notebook evaluation.ipynb         # for evaluation
```

## Note

[data/x1](data/x1) is the first small batch from enwik9.txt
