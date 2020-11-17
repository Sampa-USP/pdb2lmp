#!/usr/bin/env python3
"""
Receives a .lmp and a bond number.
Erease this bond from file (also removing the angles due to it) and redo the IDs correctly.

Author: Henrique Musseli Cezar
Date: JUL/2020
"""

import argparse
import os
import re

# from https://stackoverflow.com/a/11541495
def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def get_bonds_angles_delete(lmpfile, bnum):
  bdel = []
  adel = []
  with open(lmpfile, "r") as f:
    line = f.readline()

    while "Bond Coeffs" not in line:
      line = f.readline()

    line = f.readline()
    while not line.strip():
      line = f.readline()

    while line.split()[0] != str(bnum):
      line = f.readline()

    btype = line.split("#")[1].strip()
    btype2 = btype.split("-")[1]+" - "+btype.split("-")[0].strip()

    while "Angle Coeffs" not in line:
      line = f.readline()

    while (btype not in line) and (btype2 not in line) and ("Bonds" not in line):
      line = f.readline()

    angnum = line.split()[0]
    atype = [line.split("#")[1].strip()]

    line = f.readline()
    while "Bonds" not in line:
      if (btype in line) or (btype2 in line):
        atype.append(line.split("#")[1].strip())
      line = f.readline()

    while "Angles" not in line:
      if not line:
        line = f.readline()
        continue

      if btype in line:
        bdel.append(line.split()[0])

      line = f.readline()

    while line:
      if not line:
        line = f.readline()
        continue

      for at in atype: 
        if at in line:
          adel.append(line.split()[0])

      line = f.readline()

  return bdel, adel, btype, atype


def print_clean_lmp(lmpfile, bnum, bdel, adel, btype, atype):
  with open(lmpfile, "r") as f:
    line = f.readline()

    while "bonds" not in line:
      print(line.rstrip())
      line = f.readline()

    nbonds = int(re.findall(r'^\D*(\d+)', line)[0])
    nline = re.sub(r'^\D*(\d+)', "\t"+str(nbonds-len(bdel)), line)
    print(nline.rstrip())

    line = f.readline()
    while "angles"not in line:
      print(line.rstrip())
      line = f.readline()

    nangles = int(re.findall(r'^\D*(\d+)', line)[0])
    nline = re.sub(r'^\D*(\d+)', "\t"+str(nangles-len(adel)), line)
    print(nline.rstrip())

    line = f.readline()
    while "bond"not in line:
      print(line.rstrip())
      line = f.readline()

    nbonds = int(re.findall(r'^\D*(\d+)', line)[0])
    nline = re.sub(r'^\D*(\d+)', "\t"+str(nbonds-1), line)
    print(nline.rstrip())

    line = f.readline()
    while "angle"not in line:
      print(line.rstrip())
      line = f.readline()

    nangles = int(re.findall(r'^\D*(\d+)', line)[0])
    nline = re.sub(r'^\D*(\d+)', "\t"+str(nangles-len(atype)), line)
    print(nline.rstrip())

    while "Bond Coeffs" not in line:
      line = f.readline()
      print(line.rstrip())

    bcount = 0
    line = f.readline()
    bindex = {}
    while (btype not in line):
      if not line.strip():
        print("")
        line = f.readline()
        continue

      print(line.rstrip())
      bcount += 1

      try: 
        bindex[line.split("#")[1].strip()] = bcount
      except:
        pass

      line = f.readline()

    i = 0
    line = f.readline()
    while "Angle Coeffs" not in line:
      if not line.strip():
        print("")
        line = f.readline()
        continue

      i += 1
      bindex[line.split("#")[1].strip()] = bcount+i
      nline = re.sub(r'^\D*(\d+)', "\t"+str(bcount+i), line)
      print(nline.rstrip())

      line = f.readline()

    print(line.rstrip())
    line = f.readline()
    acount = 1
    aindex = {}
    while "Atoms" not in line:
      if not line.strip():
        print("")
        line = f.readline()
        continue

      flag = False
      for at in atype:
        if at in line:
          flag = True
          line = f.readline()
          continue
      if flag:
        continue

      aindex[line.split("#")[1].strip()] = acount
      nline = re.sub(r'^\D*(\d+)', "\t"+str(acount), line)
      print(nline.rstrip())

      acount += 1
      line = f.readline()

    print(line.rstrip())
    while "Bonds" not in line:
      line = f.readline()
      print(line.rstrip())

    line = f.readline()
    bcount = 1
    while "Angles" not in line:
      if not line.strip():
        print("")
        line = f.readline()
        continue

      if btype in line:
        line = f.readline()
        continue

      nline = re.sub(r'^\D*(\d+)\D*(\d+)', "\t"+str(bcount)+"\t"+str(bindex[line.split("#")[1].strip()]), line)
      print(nline.rstrip())
      bcount += 1
      line = f.readline()

    print(line.rstrip())
    line = f.readline()
    acount = 1
    while line:
      if not line.strip():
        print("")
        line = f.readline()
        continue

      flag = False
      for at in atype:
        if at in line:
          line = f.readline()
          flag = True
          continue  
      if flag: 
        continue

      nline = re.sub(r'^\D*(\d+)\D*(\d+)', "\t"+str(acount)+"\t"+str(aindex[line.split("#")[1].strip()]), line)
      print(nline.rstrip())
      acount += 1
      line = f.readline()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Receives a .lmp and a bond number to erease this bond from topology.")
  parser.add_argument("lmpfile", type=extant_file, help="path to the .lmp file")
  parser.add_argument("bond", type=int, help="number of bond (from .lmp) to be deleted")

  args = parser.parse_args()

  # first pass to detect bond type (from comment) and bonds and angles to be removed
  bdel, adel, btype, atype = get_bonds_angles_delete(args.lmpfile, args.bond)

  # print new file deleting the bonds and angles
  print_clean_lmp(args.lmpfile, args.bond, bdel, adel, btype, atype)
