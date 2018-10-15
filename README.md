# Brief
This code tries to implement the room layout estimation network described in

Physics Inspired Optimization on Semantic Transfer Features: An Alternative Method for Room Layout Estimation [CVPR 2017]

# Data
Download pre-processed SUNRGBD dataset at [sunrgbd.zip](https://drive.google.com/open?id=1oP0-n0AHW5mlfNrORLmQAAXqv0ByjIRg)

Execute 'mkdir datasets' in the root folder and unzip the dataset therein.

# Commands
python segment_st.py train -d datasets/sunrgbd/ -c 37 -s 480 --arch drn_d_105 --batch-size 32 --random-scale 1.75 --random-rotate 15 --epochs 100 --lr 0.01 --momentum 0.9 --lr-mode poly

can be used to train a semantic segmentation network on sunrgbd.

python segment_st.py test -d datasets/sunrgbd/ -c 37 --arch drn_d_105 -s 480 --resume model_best.pth.tar --phase val --batch-size 1 --ms

can be used to evaluate the sunrgbd network.
