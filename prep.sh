#!/bin/bash
sudo apt-get install libsndfile1
pip3 install -U librosa

x=$1

wget http://www.openslr.org/resources/60/$x.tar.gz
mv $x.tar.gz ..
tar -xzf ../$x.tar.gz -C ..
rm ../$x.tar.gz
echo "Done unzip libritts $x"

python3 data/preprocess.py -d ../$x
echo "Done preprocess ljspeech"

y="LJSpeech-1.1"

wget https://data.keithito.com/data/speech/$y.tar.bz2
mv $y.tar.bz2 ..
tar -xf ../$y.tar.bz2 -C ..
rm ../$y.tar.bz2
echo "Done unzip ljspeech"

python3 data/preprocess.py -d ../$y
echo "Done preprocess ljspeech"
