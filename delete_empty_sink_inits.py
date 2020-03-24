import os

SINK = "//Desktop-sa1evjv/h/sink/recurrent_conv-1/"

files = [SINK + str(i) + "/dnc/__init__.py" for i in range(98, 144)]

for f in files:
    try:
        os.remove(f)
    except:
        continue