#!/bin/bash
# Die inam oname variante muss noch implimentiert werden
iname="$1"
oname="$2"

tempfile="tomkv_tempfile.temp"
paraname="TOMKVPARA"
defaultpara="/home/dave/syscripts/tomkv/$paraname"
outpara="temp$paraname"
formats=('mp4' 'mov' 'mpg' 'avi' 'webm')

  odir=`pwd`
  adir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
  #packagedir="$( cd "$adir" && cd .. && pwd )"
  #bin="$packagedir/bin"



if [ `echo "$iname"|grep -Eic '^[-]{1,}info$|^[-]{1,}help$'` -eq 1 ];then
  cat "$adir/infofile"
  exit
fi



  if [ `echo "$iname"|grep -Eic '^[-]{1,}par$'` -eq 1 ];then
  	cp "$defaultpara" .
  	ls
  	exit
  fi
  if [ `echo $PATH|grep -c "$adir"` -eq 0 ];then PATH=$PATH:"$adir";fi

  paracheck.sh "$paraname" "$defaultpara" "$outpara"


 codec=`sed -n 1p "$outpara"`
 ffr=`sed -n 2p "$outpara"`
 crf=`sed -n 3p "$outpara"`
 vsync=`sed -n 4p "$outpara"`
 preset=`sed -n 5p "$outpara"`
 tune=`sed -n 6p "$outpara"`
 delodat=`sed -n 7p "$outpara"`
 
 lcrf=0
 ffmidstring=''

 if [ `echo "$ffr"|grep -Ec '^[0-9]+([.][0-9]+)?$'` -eq 1 ];then ffstartstring="-r $ffr" ;fi

 if [ `echo "$crf"|grep -Ec '^[0-9]+([.][0-9]+)?$'` -eq 1 ];then  lcrf=1; fi
 if [ `echo "$vsync"|grep -Ec '^[0-9]+([.][0-9]+)?$'` -eq 1 ];then ffmidstring="-vsync $vsync";fi
 if [ `echo "$tune"|grep -ic 'off'` -eq 0 ];then ffmidstring="-tune $tune $ffmidstring" ;fi
 if [ `echo "$preset"|grep -ic 'off'` -eq 0 ];then ffmidstring="-preset $preset $ffmidstring";fi
 if [ `echo "$delodat"|grep -ic 'yes'` -eq 1 ];then
	 ldel=2
 elif [ `echo "$delodat"|grep -ic 'ask'` -eq 1 ];then
        ldel=1
 else 
	ldel=0
 fi

 

 if [ `echo "$codec"|grep -ic 'h264'` -gt 0 ];then
     ffmidstring='-c:v libx264 -c:a copy '"$ffmidstring"
     if [ $lcrf -eq 1 ];then ffmidstring="-crf $crf $ffmidstring";fi

 elif [ `echo "$codec"|grep -ic 'h265'` -gt 0 ];then
     if [ $lcrf -eq 0 -o $crf -eq 0 ];then
       ffmidstring='-c:v libx265 -x265-params lossless=1 -c:a copy '"$ffmidstring"
     else
       ffmidstring='-c:v libx265 -c:a copy '"$ffmidstring"
       ffmidstring="-crf $crf $ffmidstring"
     fi
 else
     ffmidstring='-c:v copy -c:a copy '"$ffmidstring"
     if [ $lcrf -eq 1 ];then ffmidstring="-crf $crf $ffmidstring";fi
 fi

 ffmidstring="$(echo $ffmidstring|tr -s ' ' ' ')"
 lmerge=0

  for fm in "${formats[@]}";do

	count=`ls | grep -E "*\.$fm"|wc -l`
	if [ $count -gt 0 ];then

	    if [ $ldel -eq 1 ];then
		read -p "Löschen der Orginaldateien ($fm)? [y|n=default]:" ans
		if [ -n "$ans" ];then
	  		terase=0
          		if [ $(echo "$ans"|grep -ciE '^y') -ge 1 ];then terase=1;fi
		else
	   		terase=0
		fi
            elif [ $ldel -eq 2 ];then
               terase=1
	    else
               terase=0
	    fi


		ls | grep -E "*\.$fm" > $tempfile


		if [ $count -eq 1 ];then
			iname="$(cat $tempfile)"
			oname="$(cat $tempfile|tr -s ' ' ' '|sed "s/\.$fm/\.mkv/g")"
			rm $tempfile
			ffmpeg -i "$iname" $ffmidstring "$oname"
			echo "ffmpeg -i '$iname' $ffmidstring '$oname'"
    
  			if [ $terase -eq 1 ];then rm "$iname";fi
  			srtcount=`ls | grep -E "*\.srt"|wc -l`
  			if [ $srtcount -eq 1 ];then
	 			sname="$(ls | grep -E "*\.srt")"
         			mkvmerge -o "t$oname" "$oname" "$sname"
	 			mv "t$oname" "$oname"
				lmerge=1
  			fi
		else

			for fil in `cat $tempfile`;do

  				oname="$(echo $fil|tr -s ' ' ' '|sed "s/\.$fm/\.mkv/g")"
				ffmpeg -i "$fil" $ffmidstring "$oname"
 				echo "ffmpeg -i '$fil' $ffmidstring '$oname'"
  				if [ $terase -eq 1 ];then rm "$fil";fi
			done
			rm $tempfile
		fi

	fi
  done

  if [ $lmerge -eq 0 ];then
       mkvcount=`ls | grep -E "*\.mkv"|wc -l` 
       srtcount=`ls | grep -E "*\.srt"|wc -l`
       if [ $srtcount -eq 1 ]&&[ $mkvcount -eq 1 ];then
	       oname="$(ls | grep -E '*.mkv')"
               sname="$(ls | grep -E "*\.srt")"
               mkvmerge -o "t$oname" "$oname" "$sname"
               mv "t$oname" "$oname"
       fi
  fi


rm "$outpara"
