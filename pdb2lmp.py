#!/usr/bin/env python3
"""
Receives a .pdb and returns the lammps data file for
a system with only atoms, bonds and angles.

Author: Henrique Musseli Cezar
Date: MAY/2020
"""

import sys
import argparse
import os
import openbabel
import pybel
import numpy as np
import math
from ase import Atom, Atoms
from ase.io import write
from scipy.spatial import cKDTree

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


def parse_mol_info(fname, fcharges, axis, buffa, buffo, pbcbonds, printdih):
  iaxis = {"x": 0, "y": 1, "z": 2}
  if axis in iaxis:
    repaxis = iaxis[axis]
  else:
    print("Error: invalid axis")
    sys.exit(0)

  if fcharges:
    chargesLabel = {}
    with open(fcharges, "r") as f:
      for line in f:
        chargesLabel[line.split()[0]] = float(line.split()[1])

  # set openbabel file format
  obConversion = openbabel.OBConversion()
  obConversion.SetInAndOutFormats("pdb","xyz")
  # trick to disable ring perception and make the ReadFile waaaay faster
  # Source: https://sourceforge.net/p/openbabel/mailman/openbabel-discuss/thread/56e1812d-396a-db7c-096d-d378a077853f%40ipcms.unistra.fr/#msg36225392
  obConversion.AddOption("b", openbabel.OBConversion.INOPTIONS) 

  # read molecule to OBMol object
  mol = openbabel.OBMol()
  obConversion.ReadFile(mol, fname)
  mol.ConnectTheDots() # necessary because of the 'b' INOPTION

  # split the molecules
  molecules = mol.Separate()

  # detect the molecules types
  mTypes = {}
  mapmTypes = {}
  atomIdToMol = {}
  nty = 0
  for i, submol in enumerate(molecules, start=1):
    atomiter = openbabel.OBMolAtomIter(submol)
    atlist = []
    for at in atomiter:
      atlist.append(at.GetAtomicNum())
      atomIdToMol[at.GetId()] = i
    foundType = None

    for ty in mTypes:
      # check if there's already a molecule of this type
      if atlist == mTypes[ty]:
        foundType = ty

    # if not, create a new type
    if not foundType:
      nty += 1
      foundType = nty
      mTypes[nty] = atlist

    mapmTypes[i] = foundType

  # get atomic labels from pdb
  idToAtomicLabel = {}
  for res in openbabel.OBResidueIter(mol):
    for atom in openbabel.OBResidueAtomIter(res):
      if atomIdToMol[atom.GetId()] > 1:
        idToAtomicLabel[atom.GetId()] = res.GetAtomID(atom).strip()+str(mapmTypes[atomIdToMol[atom.GetId()]])
      else:
        idToAtomicLabel[atom.GetId()] = res.GetAtomID(atom).strip()

  # print(idToAtomicLabel)

  # identify atom types and get masses
  outMasses = "Masses\n\n"

  massTypes = {}
  mapTypes = {}
  nmassTypes = 0
  atomIterator = openbabel.OBMolAtomIter(mol)
  for atom in atomIterator:
    i = atom.GetId()
    if idToAtomicLabel[i] not in massTypes:
      nmassTypes += 1
      mapTypes[nmassTypes] = idToAtomicLabel[i]
      massTypes[idToAtomicLabel[i]] = nmassTypes
      outMasses += "\t%d\t%.3f\t# %s\n" % (nmassTypes, atom.GetAtomicMass(), idToAtomicLabel[i])

  # create atoms list
  outAtoms = "Atoms # full\n\n"

  xmin = float("inf")
  xmax = float("-inf")
  ymin = float("inf")
  ymax = float("-inf")
  zmin = float("inf")
  zmax = float("-inf")
  natoms = 0
  acoords = []
  for mnum, imol in enumerate(molecules, start=1):
    atomIterator = openbabel.OBMolAtomIter(imol)
    for atom in sorted(atomIterator, key=lambda x: x.GetId()):
      natoms += 1
      i = atom.GetId()
      apos = (atom.GetX(), atom.GetY(), atom.GetZ())
      acoords.append(Atom(atom.GetAtomicNum(), apos))

      # look for the maximum and minimum x for the box (improve later with numpy and all coordinates)
      if apos[0] > xmax:
        xmax = apos[0]
      if apos[0] < xmin:
        xmin = apos[0]
      if apos[1] > ymax:
        ymax = apos[1]
      if apos[1] < ymin:
        ymin = apos[1]
      if apos[2] > zmax:
        zmax = apos[2]
      if apos[2] < zmin:
        zmin = apos[2]

      if fcharges:
        outAtoms += "\t%d\t%d\t%d\t%.6f\t%.4f\t%.4f\t%.4f\t# %s\n" % (i+1, mnum, massTypes[idToAtomicLabel[i]], chargesLabel[idToAtomicLabel[i]], atom.GetX(), atom.GetY(), atom.GetZ(), idToAtomicLabel[i])
      else:
        outAtoms += "\t%d\t%d\t%d\tX.XXXXXX\t%.4f\t%.4f\t%.4f\t# %s\n" % (i+1, mnum, massTypes[idToAtomicLabel[i]], atom.GetX(), atom.GetY(), atom.GetZ(), idToAtomicLabel[i])

  # define box shape and size
  try:
    fromBounds = False
    rcell = mol.GetData(12)
    cell = openbabel.toUnitCell(rcell)
    v1 = [cell.GetCellVectors()[0].GetX(), cell.GetCellVectors()[0].GetY(), cell.GetCellVectors()[0].GetZ()]
    v2 = [cell.GetCellVectors()[1].GetX(), cell.GetCellVectors()[1].GetY(), cell.GetCellVectors()[1].GetZ()]
    v3 = [cell.GetCellVectors()[2].GetX(), cell.GetCellVectors()[2].GetY(), cell.GetCellVectors()[2].GetZ()]
    boxinfo = [v1,v2,v3]
    orthogonal = True
    for i, array in enumerate(boxinfo):
      for j in range(3):
        if i == j:
          continue
        if not math.isclose(0., array[j], abs_tol=1e-6):
          orthogonal = False
  except:
    fromBounds = True
    v1 = [xmax - xmin, 0., 0.]
    v2 = [0., ymax - ymin, 0.]
    v3 = [0., 0., zmax - zmin]
    orthogonal = True

  # add buffer
  if orthogonal:
    buf = []
    boxinfo = [v1,v2,v3]
    for i, val in enumerate(boxinfo[repaxis]):
      if i == repaxis:
        buf.append(val+buffa)
      else:
        buf.append(val)
    boxinfo[repaxis] = buf
    for i in range(3):
      if i == repaxis:
        continue
      buf = []
      for j, val in enumerate(boxinfo[i]):
        if j == i:
          buf.append(val+buffo)
        else:
          buf.append(val)
      boxinfo[i] = buf

  # print(boxinfo)

  # Duplicate to get the bonds in the PBC. Taken from (method _crd2bond):
  # https://github.com/tongzhugroup/mddatasetbuilder/blob/66eb0f15e972be0f5534dcda27af253cd8891ff2/mddatasetbuilder/detect.py#L213
  if pbcbonds:
    acoords = Atoms(acoords, cell=boxinfo, pbc=True)
    repatoms = acoords.repeat(2)[natoms:] # repeat the unit cell in each direction (len(repatoms) = 7*natoms)
    tree = cKDTree(acoords.get_positions())
    d = tree.query(repatoms.get_positions(), k=1)[0]
    nearest = d < 8.
    ghost_atoms = repatoms[nearest]
    realnumber = np.where(nearest)[0] % natoms
    acoords += ghost_atoms

    write("replicated.xyz", acoords) # write the structure with the replicated atoms

    # # write new mol with new bonds
    nmol = openbabel.OBMol()
    nmol.BeginModify()
    for idx, (num, position) in enumerate(zip(acoords.get_atomic_numbers(), acoords.positions)):
        a = nmol.NewAtom(idx)
        a.SetAtomicNum(int(num))
        a.SetVector(*position)
    nmol.ConnectTheDots()
    # nmol.PerceiveBondOrders() # super slow becauses it looks for rings
    nmol.EndModify()
  else:
    nmol = mol

  # identify bond types and create bond list
  outBonds = "Bonds # harmonic\n\n"

  bondTypes = {}
  mapbTypes = {}
  nbondTypes = 0
  nbonds = 0
  bondsToDelete = []
  bondIterator = openbabel.OBMolBondIter(nmol)
  for i, bond in enumerate(bondIterator, 1):    
    b1 = bond.GetBeginAtom().GetId()    
    b2 = bond.GetEndAtom().GetId()

    # check if its a bond of the replica only
    if (b1 >= natoms) and (b2 >= natoms):
      bondsToDelete.append(bond)
      continue
    # remap to a real atom if needed
    if b1 >= natoms:
      b1 = realnumber[b1-natoms]
    if b2 >= natoms:
      b2 = realnumber[b2-natoms]

    # identify bond type
    btype1 = "%s - %s" % (idToAtomicLabel[b1],idToAtomicLabel[b2])
    btype2 = "%s - %s" % (idToAtomicLabel[b2],idToAtomicLabel[b1])

    if btype1 in bondTypes:
      bondid = bondTypes[btype1]
      bstring = btype1
    elif btype2 in bondTypes:
      bondid = bondTypes[btype2]
      bstring = btype2
    else:
      nbondTypes += 1
      mapbTypes[nbondTypes] = btype1
      bondid = nbondTypes
      bondTypes[btype1] = nbondTypes
      bstring = btype1

    nbonds += 1
    outBonds += "\t%d\t%d\t%d\t%d\t# %s\n" % (nbonds, bondid, b1+1, b2+1, bstring)


  # delete the bonds of atoms from other replicas
  for bond in bondsToDelete:
    nmol.DeleteBond(bond)

  # identify angle types and create angle list
  nmol.FindAngles()
  outAngles = "Angles # harmonic\n\n"

  angleTypes = {}
  mapaTypes = {}
  nangleTypes = 0
  nangles = 0
  angleIterator = openbabel.OBMolAngleIter(nmol)
  for i, angle in enumerate(angleIterator, 1):
    a1 = angle[1]
    a2 = angle[0]
    a3 = angle[2]

    # remap to a real atom if needed
    if a1 >= natoms:
      a1 = realnumber[a1-natoms]
    if a2 >= natoms:
      a2 = realnumber[a2-natoms]
    if a3 >= natoms:
      a3 = realnumber[a3-natoms]

    atype1 = "%s - %s - %s" % (idToAtomicLabel[a1],idToAtomicLabel[a2],idToAtomicLabel[a3])
    atype2 = "%s - %s - %s" % (idToAtomicLabel[a3],idToAtomicLabel[a2],idToAtomicLabel[a1])

    if atype1 in angleTypes:
      angleid = angleTypes[atype1]
      astring = atype1
    elif atype2 in angleTypes:
      angleid = angleTypes[atype2]
      astring = atype2
    else:
      nangleTypes += 1
      mapaTypes[nangleTypes] = atype1
      angleid = nangleTypes
      angleTypes[atype1] = nangleTypes
      astring = atype1

    nangles += 1
    outAngles += "\t%d\t%d\t%d\t%d\t%d\t# %s\n" % (nangles, angleid, a1+1, a2+1, a3+1, astring)

  # identify dihedral types and create dihedral list
  nmol.FindTorsions()
  if printdih:
    outDihedrals = "Dihedrals # charmmfsw\n\n"

    dihedralTypes = {}
    mapdTypes = {}
    ndihedralTypes = 0
    ndihedrals = 0
    dihedralIterator = openbabel.OBMolTorsionIter(nmol)
    for i, dihedral in enumerate(dihedralIterator, 1):
      a1 = dihedral[0]
      a2 = dihedral[1]
      a3 = dihedral[2]
      a4 = dihedral[3]

      # remap to a real atom if needed
      if a1 >= natoms:
        a1 = realnumber[a1-natoms]
      if a2 >= natoms:
        a2 = realnumber[a2-natoms]
      if a3 >= natoms:
        a3 = realnumber[a3-natoms]
      if a4 >= natoms:
        a4 = realnumber[a4-natoms]

      dtype1 = "%s - %s - %s - %s" % (idToAtomicLabel[a1],idToAtomicLabel[a2],idToAtomicLabel[a3],idToAtomicLabel[a4])
      dtype2 = "%s - %s - %s - %s" % (idToAtomicLabel[a4],idToAtomicLabel[a3],idToAtomicLabel[a2],idToAtomicLabel[a1])

      if dtype1 in dihedralTypes:
        dihedralid = dihedralTypes[dtype1]
        dstring = dtype1
      elif dtype2 in dihedralTypes:
        dihedralid = dihedralTypes[dtype2]
        dstring = dtype2
      else:
        ndihedralTypes += 1
        mapaTypes[ndihedralTypes] = dtype1
        dihedralid = ndihedralTypes
        dihedralTypes[dtype1] = ndihedralTypes
        dstring = dtype1

      ndihedrals += 1
      outDihedrals += "\t%d\t%d\t%d\t%d\t%d\t# %s\n" % (ndihedrals, dihedralid, a1+1, a2+1, a3+1, a4+1, dstring)
  else:
    outDihedrals = ""

  # print header
  if printdih:
    header = "LAMMPS topology created from %s using pdb2lmp.py - By Henrique Musseli Cezar, 2020\n\n\t%d atoms\n\t%d bonds\n\t%d angles\n\t%d dihedrals\n\n\t%d atom types\n\t%d bond types\n\t%d angle types\n\t%d dihedral types\n\n" % (fname, natoms, nbonds, nangles, ndihedrals, nmassTypes, nbondTypes, nangleTypes, ndihedralTypes)
  else:
    header = "LAMMPS topology created from %s using pdb2lmp.py - By Henrique Musseli Cezar, 2020\n\n\t%d atoms\n\t%d bonds\n\t%d angles\n\n\t%d atom types\n\t%d bond types\n\t%d angle types\n\n" % (fname, natoms, nbonds, nangles, nmassTypes, nbondTypes, nangleTypes)

  # add box info
  if fromBounds:
    boxsize = [(xmin,xmax),(ymin,ymax),(zmin,zmax)]
    boxsize[repaxis] = (boxsize[repaxis][0]-buffa/2., boxsize[repaxis][1]+buffa/2.)
    for i in range(3):
      if i == repaxis:
        continue
      boxsize[i] = (boxsize[i][0]-buffo/2., boxsize[i][1]+buffo/2.)
    header += "\t%.8f\t%.8f\t xlo xhi\n\t%.8f\t%.8f\t ylo yhi\n\t%.8f\t%.8f\t zlo zhi\n" % (boxsize[0][0], boxsize[0][1], boxsize[1][0], boxsize[1][1], boxsize[2][0], boxsize[2][1])
  else:
    if orthogonal:
      header += "\t%.8f\t%.8f\t xlo xhi\n\t%.8f\t%.8f\t ylo yhi\n\t%.8f\t%.8f\t zlo zhi\n" % (0., boxinfo[0][0], 0., boxinfo[1][1], 0., boxinfo[2][2])
    else:
      header += "\t%.8f\t%.8f\t xlo xhi\n\t%.8f\t%.8f\t ylo yhi\n\t%.8f\t%.8f\t zlo zhi\n\t%.8f\t%.8f\t%.8f\t xy xz yz\n" % (0., boxinfo[0][0], 0., boxinfo[1][1], 0., boxinfo[2][2], boxinfo[1][0], boxinfo[2][0], boxinfo[2][1])

  # print Coeffs
  outCoeffs = "Pair Coeffs\n\n"

  for i in range(1,nmassTypes+1):
    outCoeffs += "\t%d\teps\tsig\t# %s\n" % (i, mapTypes[i])

  outCoeffs += "\nBond Coeffs\n\n"

  for i in range(1,nbondTypes+1):
    outCoeffs += "\t%d\tK\tr_0\t# %s\n" % (i, mapbTypes[i])

  outCoeffs += "\nAngle Coeffs\n\n"

  for i in range(1,nangleTypes+1):
    outCoeffs += "\t%d\tK\ttetha_0 (deg)\t# %s\n" % (i, mapaTypes[i])

  return header+"\n"+outMasses+"\n"+outCoeffs+"\n"+outAtoms+"\n"+outBonds+"\n"+outAngles


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Receives a .pdb and generate a LAMMPS data file with atoms, bonds and angles.")
  parser.add_argument("pdbfile", type=extant_file, help="path to the .pdb file")
  parser.add_argument("--charges", type=extant_file, help="path to a file associating PDB label to atomic charge (one pair per line)")
  parser.add_argument("--axis", default="z", help="axis to replicate and check for bonds and angles in the PBC (default: z)")
  parser.add_argument("--buffer-length-axis", type=float, help="length of the extra space in the replicated axis for PBC (default: 1.0 - NOT considered for non orthogonal cell)", default=1.)
  parser.add_argument("--buffer-length-orthogonal", type=float, help="length of size orthogonal to the axis with PBC (default: 30.0 - NOT considered for non orthogonal cell)", default=30.)
  parser.add_argument("--pbc-bonds", action="store_true", help="look for bonds and angles in the pbc images?")
  parser.add_argument("--ignore-dihedrals", action="store_true", help="does not print info about dihedrdals in the topology")

  args = parser.parse_args()

  # get basename and file extension
  base, ext = os.path.splitext(args.pdbfile)

  if ext[1:] != "pdb":
    print("Error: only .pdb files are accepted.")
    sys.exit(0)

  if args.ignore_dihedrals:
    printdih = False
  else:
    printdih = True

  outlmp = parse_mol_info(args.pdbfile, args.charges, args.axis, args.buffer_length_axis, args.buffer_length_orthogonal, args.pbc_bonds, printdih)

  print(outlmp)

