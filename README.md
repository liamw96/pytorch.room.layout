# Brief
This code implements the room layout estimation network described in

[**Physics Inspired Optimization on Semantic Transfer Features: An Alternative Method for Room Layout Estimation**](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Physics_Inspired_Optimization_CVPR_2017_paper.html)

# Data
Download pre-processed SUNRGBD dataset at [sunrgbd.zip](https://drive.google.com/open?id=1oP0-n0AHW5mlfNrORLmQAAXqv0ByjIRg)

Download pre-processed LSUN dataset at [lsun.zip](https://drive.google.com/open?id=1e40AC_9CwgWPQL9eh18y2k9u4O0X3rl4)

Execute `mkdir datasets` in the root folder and unzip the datasets therein.

# Commands - SUNRGBD
`python segment_st.py train -d datasets/sunrgbd/ -c 37 -s 480 --arch drn_d_105 --batch-size 32 --random-scale 1.75 --random-rotate 15 --epochs 100 --lr 0.01 --momentum 0.9 --lr-mode poly`

can be used to train a semantic segmentation network on sunrgbd.

`python segment_st.py test -d datasets/sunrgbd/ -c 37 --arch drn_d_105 -s 480 --resume sunrgbd.pth.tar --phase val --batch-size 1 --ms`

can be used to evaluate the sunrgbd network.

pre-trained [sunrgbd.pth.tar](https://drive.google.com/open?id=1-O45ENLICItubbah8osWkhe--BS-_of0)

It should report 41.64 mIOU on SUNRGBD-val, with per-category IOU as

76.207 88.351 44.797 65.945 69.527 51.554 52.003 42.822 48.869 36.718 49.844 28.896 32.876 18.304 5.300 57.734 43.211 27.770 36.247 0.000 29.458 66.611 36.991 35.774 55.149 21.252 27.205 0.000 27.276 52.067 54.973 5.531 75.905 59.562 41.282 55.303 19.429

# Commands - LSUN

`python segment_rl.py train -d datasets/lsun/ -c 4 -s 480 --arch drn_d_105 --batch-size 32 --random-scale 1.75 --random-rotate 15 --epochs 100 --lr 0.01 --momentum 0.9 --lr-mode poly --pretrained sunrgbd.pth.tar`

can be used to train a room layout network by transfering representations trained on sunrgbd.

For evaluation,

exectute `mkdir features` in the root folder to store network outputs before softmax.

pre-trained [lsun.pth.tar](https://drive.google.com/open?id=1cyw3cfV4qPH2yS_XfeKDnJYbPERHA3tU)

Then use this command for evaluation:

`python segment_rl.py test -d datasets/lsun/ -c 4 --arch drn_d_105 -s 480 --resume lsun.pth.tar --phase val --batch-size 1 --ms`

The mIOU style performance should be reported as: 

75.661 70.330 67.400 76.282

