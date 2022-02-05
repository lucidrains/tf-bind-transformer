#/bin/bash

remap_path=$1

if [ ! -f $remap_path ]; then
    echo "remap path with id and parsed exp-target-cell-type not found"
    exit 1
fi

numrows=$(wc -l $remap_path | cut -d " " -f 1)

n=0
for i in ./*.bed; do
  bedtools intersect -v -a $remap_path -b $i > $i.negative
  echo "created $i.negative"
  python to_npy.py $i.negative $numrows
  echo "processed $i"
  rm $i.negative

  n=$((n+1))
  echo $n
done
