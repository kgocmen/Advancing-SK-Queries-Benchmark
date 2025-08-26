# Research

STEPS:

1. Download your .osm.pbf extended data and put it into _raw_data_tools folder.

2. Run ./_raw_data_tools/pbftocsv.sh <basename>.osm.pbf <basename> [minlon minlat maxlon maxlat]

! YOUR COMPUTER MAY CRASH IF THE .osm.pbf is too large or the boundaries given are too big. !

This command will create data/<basename>.csv file.


OPTIONAL: Clean your csv.
OPTIONAL: Put some queries into data/queries/ folder. It should be in the format of .txt or .csv and contain <text>, lat, lon.

3. Run ./docker.sh

4. Run ./run.sh

Note: Comment out the experiments you WANT TO run. As default nothing is commented out, it will run nothing.
Everything will be created automatically.