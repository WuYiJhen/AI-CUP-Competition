# **教育部全國大專校院人工智慧競賽(AI CUP)-機器閱讀紀錄-課程挑戰賽**

[AI CUP Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/12)

Team Name : 結果尚未公布\
Team Member : 吳亦振 (Wu, Yi-Chen)\
Final Rank : 1st/148\
Public Score : 0.743 (1st/148)\
Private Score : 0.736 (1st/148)

## `Run Scripts`

#### Data preprocess

Please upload trainset.csv and testset.csv to `./src/data/` folder first.
```
$ cd src
$ python preprocess.py
```

#### Eval valid data and test data

Please download the model from [AICUP](https://drive.google.com/drive/folders/1sEeuPFRMG4OgYamTXImoDVhQ3Kr1m3Mf?usp=sharing) and upload to `./src/model/` folder.
```
$ python eval_valid.py (optional)
$ python eval_test.py
```