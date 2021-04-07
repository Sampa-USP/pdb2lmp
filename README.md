# pdb2lmp
Python 3 script to create a LAMMPS topology from a PDB file. 

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

The simulation box size is determined by either the CRYST1 line in the PDB file or by automatic detection. By default, a 1 angstrom buffer is added in each direction, use the `--buffer-length-axis` and `--buffer-length-orthogonal` options to set it to 0.0 if necessary.

The `--axis` option can be used when the structure has some kind of axial symmetry, such as in nanotubes.

## Contributing

Feel free to create issues reporting bugs and/or suggesting enhancements, I'll address them as soon as possible. Pull requests are also encouraged.