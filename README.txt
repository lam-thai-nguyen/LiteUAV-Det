Implementation of LiteUAV-Det: A lightweight network for robust small-object detection in complex aerial scenes (Neurocomputing 2026)
Paper: https://www.sciencedirect.com/science/article/pii/S0925231226011793

***

Summary:
- LiteUAV_Det-n: 3.41M
- LiteUAV_Det-s: 12.71M
- LiteUAV_Det-m: 23.96M
- LiteUAV_Det-l: 34.96M
- LiteUAV_Det-x: 54.60M

***

Some discrepancies in the LiteUAV-Det paper:

- Depthwise separable convolution (DSConv) is shown in Fig 3 as DWConv with the following steps: Depthwise -> BN -> ReLU -> Pointwise -> BN -> ReLU. However, in Fig 10, it is shown as Pointwise -> Depthwise.
--> I choose the one shown in Fig 3, following the steps of MobileNet's depthwise separable convolution.

- Partial convolution (PConv) number of divisions is not stated.
--> I set n_div = 4, following the original paper of PConv.

- SelFusion block is stated to have PAConv (i.e. PConv) with (k,s,p)=(3,1,0). However, such values can't produce an output of the same shape as the output, thereby not allowing the following concatenation.
--> I set (k,s,p)=(3,1,1), following the original paper of PConv.

- EPPF module's Maxpool2d kernel size is not clearly given in section 3.4. It says the first kernel size equals 1, which makes Maxpool2d an Identity operation, which is very unlikely. 
--> Since EPPF is an alternative to YOLOv5's SPPF, I choose 3 kernel size of Maxpool2d to be (5,5).

***

Other modifications: 
Since I couldn't get 6.38M params like in the paper due to some unclarity, I made some modifications to make the architecture more lightweight.

- Issue 1: The 2 3x3 Conv layers in the DSSF module introduce too many params
--> I use a hidden dimension c_ in DSSF.
--> Results: LiteUAV_Det-n from 10.34M to 5.69M

- Issue 2: The 2 3x3 Conv layers in the EPPF module introduce too many params
--> I use a hidden dimension c_ in EPPF, just like Ultralytics' SPPF.
--> Results: LiteUAV_Det-n from 10.34M to 8.06M

- Combining these 2 changes, LiteUAV_Det-n from 10.34M to 3.41M.

***

Notes:

- I have not implemented Adaptive Brightness Enhancement Module and DBSnet Module during the Dataloading phase.
- I have not implemented WiseIoU v3 Loss during the training phase.
- This project's implementation focuses solely on the model architecture.

*** 

Further read:

- PConv paper: https://arxiv.org/abs/2303.03667
- PConv code: https://github.com/JierunChen/FasterNet/tree/master
- LiteUAV-Det code: https://github.com/dhuvisionlab/LiteUAV-Det
(as of today 20/05/2026, LiteUAV-Det code for model architecture is not publicly released)