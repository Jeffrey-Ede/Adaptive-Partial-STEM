import os

SINK = "//Desktop-sa1evjv/h/sink/recurrent_conv-1/"

dirs = [SINK + str(i) + "/" for i in range(98, 144)]

for dir in dirs:
    for subdir in [dir+d for d in os.listdir(dir)]:
        try:
            os.rmdir(subdir)
        except:
            continue