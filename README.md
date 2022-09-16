#### mvHOTA: A multi-view higher order tracking accuracy metric to measure temporal and spatial associations in multi-point tracking
[Lalith Sharan](linik-zentrum/herzchirurgie/forschung/ag-artificial-intelligence-in-cardiovascular-medicine#layer=/personen/lalith-sharan-msc-8155), 
Halvar Kelm, Gabriele Romano, Matthias Karck, Raffaele De Simone, [Sandy Engelhardt](https://www.klinikum.uni-heidelberg.de/chirurgische-klinik-zentrum/herzchirurgie/forschung/ag-artificial-intelligence-in-cardiovascular-medicine#layer=/personen/jun-prof-dr-sandy-engelhardt-7654)

AE-CAI Workshop 2022

### Overview

Multi-point tracking is a challenging task that involves detecting points in the scene and tracking them across a 
sequence of frames. Computing detection-based measures like the F-measure on a frame-by-frame basis is not sufficient 
to assess the overall performance, as it does not interpret performance in the temporal domain.
The main evaluation metric available comes from Multi-object tracking (MOT) methods to benchmark performance on datasets 
such as KITTI with the recently proposed _higher order tracking accuracy (HOTA)_ metric, which is capable of providing a 
better description of the performance over metrics such as MOTA, DetA, and IDF1.
While the HOTA metric takes into account temporal associations, it does not provide a tailored means to analyse the 
spatial associations of a dataset in a multi-camera setup. 
Moreover, there are differences in evaluating the detection task for points when compared to objects 
(point distances vs. bounding box overlap). 
Therefore in this work, we propose a multi-view higher order tracking metric _mvHOTA_ to determine the accuracy of
multi-point (multi-instance and multi-class) tracking methods, while taking into account temporal and spatial 
associations.
mvHOTA can be interpreted as the geometric mean of detection, temporal, and spatial associations, thereby providing 
equal weighting to each of the factors. 
We demonstrate the use of this metric to evaluate the tracking performance on an endoscopic point detection dataset 
from a previously organised surgical data science challenge.
Furthermore, we compare with other adjusted MOT metrics for this use-case, discuss the properties of mvHOTA, 
and show how the proposed _multi-view Association_ and the _Occlusion index (OI)_ facilitate analysis of methods with 
respect to handling of occlusions.

### Usage

The set of ground-truth detections and predicted trackers for a sequence need to be provided in the following format:
```gt_dets_seq_view = [ {'Point_ID_1': (123, 21), 'Point_ID_2': (213, 12)...}, {}, {}....]```
where each dict represents one frame of the sequence. The mvHOTA value can then be averaged for each class and each
sequence.

### References

The code of this repo is inspired by and is an extension of the _HOTA_ metric that was proposed by 
```
Luiten, J., A. Osep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixe, and B. Leibe (2021, February). HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. International
Journal of Computer Vision 129 (2), 548–578. arXiv: 2009.07736
```

#### For any suggestions and queries please contact [Lalith Sharan](mailto:lalithnag.sharangururaj@med.uni-heidelberg.de)

