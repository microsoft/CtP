## Data Preparation

#### UCF-101

1. Download the compressed file of raw videos from the official website. 

   ```bash
   mkdir -p data/ucf101/
   wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O data/ucf101/UCF101.rar --no-check-certificate 
   ```

2. Unzip the compressed file.

   ```bash
   # install unrar if necessary 
   # sudo apt-get install unrar
   unrar e data/ucf101/UCF101.rar data/ucf101/UCF101_raw/
   ```

3. Download train/test split file

   ```bash
   wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip -O data/ucf101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
   unzip data/ucf101/UCF101TrainTestSplits-RecognitionTask.zip -d data/ucf101/.
   ```

4. Run the preprocessing script (it takes about 1hours to extract raw frames).

   ```bash
   python scripts/process_ucf101.py --raw_dir data/ucf101/UCF101_raw/ --ann_dir data/ucf101/ucfTrainTestlist/ --out_dir data/ucf101/
   ```

5. (Optional) The generated annotation file is in format of .txt. One can convert it into the json format by `scripts/cvt_txt_to_json.py`.

6. (Optional) delete raw videos to save disk space.

   ```bash
   rm data/ucf101/UCF101.rar
   rm -r data/ucf101/UCF101_raw/
   ```

   

#### HMDB-51

1. Download the compressed file of raw videos from the official website.

   ```bash
   mkdir -p data/hmdb51/
   wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar -O data/hmdb51/HMDB51.rar --no-check-certificate 
   ```

2. Unzip the compressed file.

   ```bash
   unrar e data/hmdb51/HMDB51.rar data/hmdb51/HMDB51_raw/
   for file in data/hmdb51/HMDB51_raw/*.rar; do unrar e ${file} ${file%".rar"}/; done
   ```

3. Download train/test split file.

   ```bash
   wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar -O data/hmdb51/test_train_splits.rar --no-check-certificate
   unrar e data/hmdb51/test_train_splits.rar data/hmdb51/test_train_splits/
   ```

4. Run the preprocessing script (it takes about 20 mins to extract raw frames).

   ```bash
   python scripts/process_ucf101.py --raw_dir data/hmdb51/HMDB51_raw/ --ann_dir data/hmdb51/test_train_splits/ --out_dir data/hmdb51/
   ```

5. (Optional) delete raw videos to save disk space.

   ```bash
   rm data/hmdb51/HMDB51.rar
   rm -r data/hmdb51/HMDB51_raw/
   ```



#### Kinetics

Since Kinetics-400/600 dataset is relatively large. We do not provide the download script and the preprocessing script here.

You can easily follow the building instruction just like UCF-101. The raw video frames are extracted from the video file and they are further saved in a compressed .zip file.

