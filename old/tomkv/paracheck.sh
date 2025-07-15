#!/bin/bash
# $1 = directory of default parameter input file

checkplotdefault() {
dpfile="$1"
dpword="$2"

if [ -f "$dpfile" ];then
  lword=`grep -ic "$dpword" $dpfile`
else
  lword=0
fi
if [ $lword -eq 0 ];then
  takefile="$INPAR"
  ldefaultvalue=true
else

 firstdigit=`grep -i "$dpword" "$dpfile" | sed 's/^[ \t]*//' | grep -Eo "^.{1}"`
 if [ "$firstdigit" == '#' ] || [ "$firstdigit" == '!' ];then
  takefile="$INPAR"
  ldefaultvalue=true
 else
  takefile="$dpfile"
  ldefaultvalue=false
 fi
fi

    firstdigit=`grep -i "$dpword" "$takefile" | tail -n1 | sed 's/^[ \t]*//' | grep -Eo "^.{1}"`
    if [ "$firstdigit" == '#' ] || [ "$firstdigit" == '!' ];then
      defaultvalue='#'
    else
      defaultvalue=`grep -i "$dpword" "$takefile"| tail -n1 |cut -d'=' -f2-|sed 's/^[ \t]*//;s/[ \t]*$//'`
    fi
}


checkplotdefaults() {
dpfile="$1"
nextjump=1
for ik in "$@";do
if [ $nextjump -eq 0 ];then
  checkplotdefault "$dpfile" "$ik"
  eval $ik=\$defaultvalue
  eval 'IO'$ik=\$ldefaultvalue
else
  nextjump=0
fi
done
}

PARANAME="$1" #name of the parameter input file
INPAR="$2" #directory of default parameter input file
tempINPAR="$3"

parameter=(CODEC FFRAMERATE CRF VSYNC PRESET TUNE DELODAT)
outpar="OUTPARA"
userinpar="INPAR.user.temp"
if [ -f "$PARANAME" ];then
sed 's/^[ \t]*//' "$PARANAME"|sed '/^#/d;/^$/d'|cut -d'#' -f1 > $userinpar
fi
checkplotdefaults "$userinpar" "${parameter[@]}"
rm -f $userinpar
if [ -f $tempINPAR ]; then rm -f $tempINPAR;fi
echo "$outpar::A list of all read TOMKV parameters" > $outpar
echo "--------------------------------------------" >> $outpar

n=0
nall=$((${#parameter[*]} - 1))
#write all in temp$INPAR
while [[ $n -le $nall ]];do
   eval par='$'${parameter[$n]}

   echo "$par" >> "$tempINPAR"
   echo "${parameter[$n]} = $par" >> $outpar
   ((n=$n+1))
done
echo "A list of all read parameters was written in $outpar"
