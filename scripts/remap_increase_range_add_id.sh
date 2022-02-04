#/bin/bash
# usage - sh ./scripts/remap_increase_range_add_id.sh remap.bed 4096 remap_out.bed

filename=$1
extend_len=$2
output_filename=$3
pad_len=$(($extend_len / 2))

if [ ! -f $filename ]; then
    echo "remap file not found"
    exit 1
fi

awk -v l="$pad_len" -F '\t' '{X=l; mid=(int($2)+int($3))/2;printf("%s\t%d\t%d\t%s\t%d\n",$1,(mid-X<0?0:mid-X),mid+X,$4,NR);}' $filename > $output_filename

echo 'success'
