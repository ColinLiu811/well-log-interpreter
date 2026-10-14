# wellog

Well log interpreter for LAS files.

Work in progress. Currently supports:

- Loading LAS 1.2 / 2.0 files with [lasio](https://github.com/kinverarity1/lasio),
  with vendor mnemonics (ILD, RHOZ, TNPH, ...) mapped to canonical curve names
- Generating synthetic, physically consistent example wells (`python -m wellog.synthetic`)
