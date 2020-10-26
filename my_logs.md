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
- [x] Parse annot and draw bounding boxes on original image.
    - [x] Cannot draw bounding boxes on flipped image. Why?
    - [x] [Python OpenCV drawing errors after manipulating array with numpy](https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy)
- [ ] `dataset.py/preprocess_true_boxes()`


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

### Snippet Analyze
#### [dataset.parse_annotation()](https://github.com/YunYang1994/tensorflow-yolov3/blob/add5920130cd8fd9474da6e4d8dd33b24a56524f/core/dataset.py#L154)
```python
bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
```
* `line[1:]` 是一个 sublist，每个 element 代表了一个 truth box 的信息，也就是 `xmin,ymin,xmax,ymax,class_id`.
* `lambda x: int(float(x))`，此 lambda function，先将一个 string argument `x` 映射成 float 类型，再将 float
类型映射成 int 类型。
* `map()` 返回的结果的是一个 `map` object，需要 `list(map object)` 将 `map` object 转换成 list。 

```python
anno = '/home/ylqi007/work/DATA/VOC2007/train/JPEGImages/000017.jpg 185,62,279,199,14 90,78,403,336,12'     # a string
line = anno.split()     # ['/home/ylqi007/work/DATA/VOC2007/train/JPEGImages/000017.jpg', '185,62,279,199,14', '90,78,403,336,12']
img_path = line[0]      # '/home/ylqi007/work/DATA/VOC2007/train/JPEGImages/000017.jpg'
box = line[1]           # '185,62,279,199,14'
box.split(',')          # ['185', '62', '279', '199', '14'], a list of string
res = list(map(lambda x: float(x), box.split(',')))     # [185.0, 62.0, 279.0, 199.0, 14.0], a list of float
res = list(map(lambda x: int(x), list(map(lambda x: float(x), box.split(',')))))    # [185, 62, 279, 199, 14], a list of int
```

#### [random_horizontal_flip()](https://github.com/YunYang1994/tensorflow-yolov3/blob/add5920130cd8fd9474da6e4d8dd33b24a56524f/core/dataset.py#L100)
![](.images/original_with_bbox.png)
![](.images/flipped_img_with_bbox.png)
* Random Flipped Image

#### [random_crop()](https://github.com/YunYang1994/tensorflow-yolov3/blob/add5920130cd8fd9474da6e4d8dd33b24a56524f/core/dataset.py#L109)
```python
max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
```
* `bboxes[:, 0:2]` represents all rows of column 0 and 1;
* `bboxes[:, [0, 2]]` represents all rows of column 0 and 2, without column 1;
* `np.min(bboxes[:, 0:2], axis=0)` represents the min value of `xmin` and min value of `ymin`;
* `np.max(bboxes[:, 2:4], axis=0)` represents the max value of `xmax` and max value of `ymax`;
- [ ] Question here
* `np.concatenate([min_coord, max_coord], axis=-1)`, concatenates the last dim???

```python
image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]     # row/col, i.e. y/x
```
* the first dimension is height, i.e. y or rows
* the second dimension is width, i.e. x or columns

![](.images/original_cropped.png) 
![](.images/flipped_cropped.png)
* Randrom Cropped Image

#### [random_translate()](https://github.com/YunYang1994/tensorflow-yolov3/blob/add5920130cd8fd9474da6e4d8dd33b24a56524f/core/dataset.py#L132)
```python
tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

M = np.array([[1, 0, tx], [0, 1, ty]])
image = cv2.warpAffine(image, M, (w, h))
```
* 将 image 上下左右随机移动，可以移动的范围是 `[tx, ty]`
* Why `(max_l_trans-1)`? 为什么要有 `-1`.

![](.images/original_translate.png)
![](.images/flipped_translate.png)        
* Random translate

#### [image_preporcess](https://github.com/YunYang1994/tensorflow-yolov3/blob/add5920130cd8fd9474da6e4d8dd33b24a56524f/core/utils.py#L38)
![](.images/original_uitls.png)
![](.images/flipped_utils.png)
* `cv2.imread()` 返回的是 BGR format 的 ndarray，如果用`cv2` 显示的时候，显示正常，没有问题。此时用 `plt.show()` 则会显示不正常。
* `image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)`, 此时经过 format，image 是 RGB format 的image，则此时用 `cv2.imshow()` 会不正常，而 `plt.show()` 正常。

#### [preprocess_true_boxes](https://github.com/YunYang1994/tensorflow-yolov3/blob/add5920130cd8fd9474da6e4d8dd33b24a56524f/core/dataset.py#L193)
* If the `self.train_input_size = 512`, and `self.strides = np.array([8, 16, 32])`, 
then the `self.train_output_sizes = [64, 32, 16]`, i.e. the original input size (512)
will be divided into `64 x 64` cells.
* `self.anchor_per_scale = 3` means that at each cell there has 3 anchors with different sizes.
* Then the `64 x 64` layer will have `64 x 64 x 3` anchors,
the `32 x 32` layer will have `32 x 32 x 3` anchors,
the `16 x 16` layer will have `16 x 16 x 3` anchors,
all of these three layers have anchors much more than 150.
But only `self.max_bbox_per_scale` in each layer.

##### A bbox sample
```
bbox = [349, 128, 515, 466]         # [xmin, ymin, xmax, ymax], why 515 > 512?
bbox_xywh = [432, 297, 166, 338]    # [x_center, y_center, width, height]
strides = [[8], [16], [32]]
bbox_scaled = [[54.      37.125   20.75    42.25   ]
                [27.      18.5625  10.375   21.125  ]
                [13.5      9.28125  5.1875  10.5625 ]]
anchors_xywh = [[54.5   37.5    1.25   1.625]       # top left corner is (54, 37) in 64x64
                 [27.5   18.5    2.     3.75 ]      # top left corner is (27, 18) in 32x32
                 [13.5    9.5    4.125  2.875]]     # top left corner is (13, 9) in 16x16
                [[54.5    37.5     1.875   3.8125]
                 [27.5    18.5     3.875   2.8125]
                 [13.5     9.5     3.6875  7.4375]]
                [[54.5     37.5      3.625    2.8125 ]
                 [27.5     18.5      4.875    6.1875 ]
                 [13.5      9.5     11.65625 10.1875 ]]
```

```python
label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5 + self.num_classes)) for i in range(3)]
bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
bbox_count = np.zeros((3,))
```
* Supporse input_size = 512, stides = [8, 16, 32] ==> output_size = [64, 32, 16]
* `label`: [Shape(64, 64, 3, 25), Shape(32, 32, 3, 25), Shape(16, 16, 3, 25)]
    * `64x64`, i.e. 将 512x512 的 image 缩小8倍得到的大小，也就是 64x64 的 cells；
    * `3`，是指每个 cell 位置上有 3 个 anchor，没别对应不同的大小和长宽比例；
    * `25`，[:4] 对应这个 object truth box 的 position `(xc, yc, w, h)`, [4] 对应是否有 object，[5:] 对应每个 class 的 initial possibility。
    * 在 `3` 个 anchors 中，只有与 bbox 的 IoU 大于 0.3 的时候，才考虑下个维度的数，也就是 `[25]` 这个维度。 
* `bboxes_xywh`: [Shape(3, 4), Shape(3, 4), Shape(3, 4)]
* `bbox_count`: np.zeros((3,)), 统计每个 scale 上 bbox？

* `if np.any(iou_mask):`    说明 `bbox_xywh_scaled[i]` 与当前的 `anchor` 有交叠。


### cv2
#### cv2 coordinates
[OpenCV Point(x,y) represent (column,row) or (row,column)](https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column)

`image = image[:, ::-1, :]`

![](.images/flip_image.png)

#### cv2, image display
* [CV01-OpenCV窗口手动关闭后堵塞程序运行的问题](https://jameslei.com/cv01-opencv-cjxbqdb52000b9ys1kjj31yn0)

#### [CV2: Geometric Transformations of Images](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)
* `cv2.warpAffine` and `cv2.warpPerspective`
* **Scaling** Scaling is just resizing of the image. OpecCV comes with a function `cv2.resize()` for this purpose.
    * Scaling factor
    * Interpolation methods: `cv2.INTER_AREA` for shrinking, `cv2.INTER_CUBIC` and `cv2.INTER_LINEAR` for zooming.
* **Translation** Translation is the shifting of object's location. Let the shift in (x, y) direction be (tx, ty),
you can create the transformation matrix **M**.
    * `cv2.warpAfine()` 
* **Rotation** Transformation matrix. OpenCV provides scaled rotation with adjustable center of rotation so that
you can rotate at any location you perfer.
    * `cv2.getRorationMatrix2D`
* **Affine Transformation** In affine transformation, all parallel lines in the original image will still be
parallel in the output image.
    * `cv2.getAffineTransform`
    * `cv2.warpAffine`
* **Perspective Transformation**
    * `cv2.getPerspectiveTransform`
    * `cv2.warpPerspective`
* **Additional Resources:** [“Computer Vision: Algorithms and Applications”, Richard Szeliski]()


---
## GitHub Setup
![](.images/GitHub_Quick_Setup.png)
