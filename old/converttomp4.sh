#!/bin/bash
# $1 = ordner


fold="$1"
odir=`pwd`

cd $fold

list=`ls | grep ".avi"`
nlist=`ls | grep ".avi"|wc -l`

i=1
while [ $i -le $nlist ];do

filen="$(echo "$list" | sed -n $i'p')"
outfile="$(echo "$filen" | sed 's/avi$/mp4/')"

ffmpeg -fflags +genpts -i "$filen" -c:v copy -c:a copy -y "$outfile"

rm "$filen"

let i++
done

