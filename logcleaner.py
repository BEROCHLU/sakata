import re

strRead = ""
with open("./result/pbatch.log", "r") as f:
    strRead = f.read()

with open("./result/pbatch.log", "w") as f:
    s = re.sub("[\[\]',]", "", strRead)
    f.write(s)
