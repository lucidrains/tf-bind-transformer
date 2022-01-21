#/bin/bash

filename=$1
peaks_path=$2
num_lines_split="${3:=1000000}"

if [ ! -f $filename ]; then
    echo "non-peaks file not found"
    exit 1
fi

if [ ! -f $peaks_path ]; then
    echo "peaks file not found"
    exit 1
fi

split -l $num_lines_split $peaks_path chunked_remap

cp $filename "$filename.filtered"

for i in ./chunked_remap*; do
  echo "filtering $filename.filtered with $i to $i.filtered";
  bedtools intersect -v -a "$filename.filtered" -b "$i" > "$i.filtered"
  rm "$filename.filtered"
  filename=$i
done

echo "success"
