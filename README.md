# pdb2lmp
Python 3 script to create a LAMMPS topology from a PDB file, or from [any other file format supported by OpenBabel](https://open-babel.readthedocs.io/en/latest/FileFormats/Overview.html). 

The highlight of this implementation in comparison with others is that it detects bonds across periodic boundary conditions.
This may be important in models for solids, surfaces or nanostructures that include bonded terms and are periodic in one or more directions.

You are free to use whatever you want to generate the initial structure (PDB file). For example, [packmol](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml) is a very powerful tool.

## Installing dependencies

The dependencies can be easy installed using [Anaconda](https://docs.conda.io/en/latest/):

```bash
conda install -c conda-forge ase
conda install -c conda-forge openbabel
```

## Usage

You can check the command line options with:
```bash
python /path/to/pdb2lmp.py -h
```

The basic usage (with default parameters) is just:
```bash
python /path/to/pdb2lmp.py structure.pdb > my_topology.lmp
```

The simulation box size is determined by the CRYST1 line in the PDB file, or the `--box-size` option, or by automatic detection. You can use the `--buffer-length-axis` and `--buffer-length-orthogonal` options to set an extra buffer to be added in the axial and perpendicular directions, respectively.

The `--axis` option can be used when the structure has some kind of axial symmetry, such as in nanotubes.
You can run the script with the `-h` option to see a help with all the available options.

A simple example of the usage for a box full of water molecules can be seen [here](https://github.com/Sampa-USP/useful-hacks/tree/master/TopoLiquid).

## Contributing

Feel free to create issues reporting bugs and/or suggesting enhancements, I'll address them as soon as possible. Pull requests are also encouraged.