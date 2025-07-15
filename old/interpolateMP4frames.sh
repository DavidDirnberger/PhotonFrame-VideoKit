#!/bin/bash
# $1 = input file
# $2 = fps (optional)
# $3 = outputfile (optional)

oinput="$1"

if [ -z "$oinput" ];then
	while true; do
	   read -p "Please enter the input file name:" oinput
	   if [ ! -z $oinput ];then break;fi
        done
  
else

ofps="$2"


if [[ "$ofps" =~ ^[0-9]+$ ]];then
fps=$ofps
else
fps=60
fi

ooutput="$3"

output="$(echo "$ooutput"|sed 's/\.mp4//')"

fi


input="$(echo "$oinput"|sed 's/\.mp4//')"
if [ -z "$output" ];then output="$input.$fps";fi

ffmpeg -i "$input.mp4" -vf "minterpolate=fps=$fps:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" "$output.mp4"
