import splitfolders
import os
path = "./data"
splitfolders.ratio(path,seed=1337, output="data-splitted", ratio=(0.7, 0.15, 0.15))

