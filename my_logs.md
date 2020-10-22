[TOC]

## Project Structure
```
_tensorflow-yolov3/
    |
    |-> core
        |-> dataset.py
```

## TODO
- [x] Prepare dataset and use absolute directories.
- [x] Prepare `voc_train.txt` and `voc_test.txt` with my own image path.
- [ ] 



## dataset.py
在用 `python main.py` 进行测试的时候，要注意 `core/config.py` 中 `__C.TRAIN.ANNOT_PATH = "./data/dataset/voc_train.txt"`
而不应该是 `__C.TRAIN.ANNOT_PATH = "../data/dataset/voc_train.txt"`

```python
__C.TRAIN.ANNOT_PATH = "../data/dataset/voc_train.txt"  # Wrong
__C.TRAIN.ANNOT_PATH = "./data/dataset/voc_train.txt"
```
其中 `./data/dataset/voc_train.txt` 是想对于 `_tensorflow-yolov3/` 的路径。       
但是此时是用 YunYang 的 `voc_train.txt` 和 `voc_test.txt`，其中的 `image_path` 并不是我的
image path，所以接下来需要生成我本机的 `voc_train.txt` and `voc_test.txt` 文件。





## GitHub Setup
![](.images/GitHub_Quick_Setup.png)
