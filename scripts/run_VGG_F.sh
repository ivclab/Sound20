#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9
do
  python exec_model.py --network VGG_F --res_dir VGG_F
done
