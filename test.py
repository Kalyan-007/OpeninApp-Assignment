import os
for root, dirs, files in os.walk("D:/VCTK", topdown=False):
   for name in files:
      print(os.path.join(root, name))