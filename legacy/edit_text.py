#
# This code is part of the Blossom project.
#
# Copyright (c) Jimmy Kang - All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Jimmy Kang <jimmykang1016@gmail.com>, March 2019
#

file_name = input("Point to file:")


origin_file = "../datasets/review_data1/features/"
origin_file = origin_file + file_name

print("origin file:", origin_file)

target_file = "../datasets/review_data/features/"
target_file = target_file + file_name

print("target file:", target_file)


origin = open(origin_file, 'r')
target = open(target_file, 'w+')

filedata = origin.read()
filedata = filedata.replace("\t", ",")
origin.close()

target.write(filedata)

print("SUCCESS!")
