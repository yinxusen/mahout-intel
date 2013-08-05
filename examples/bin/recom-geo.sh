#!/bin/bash
# Author: 
# Func: 

set -e -x

curdir=`dirname $0`
# In production, "logutil.sh" file should be in the same working dir as current running script is.
source $curdir/logutil.sh


MAHOUT=/home/spark/yinxusen/mahout-distribution-0.7/bin/mahout

queue_name=${QUEUE_NAME:-'temp'}

function usage
{
    echo $(basename $0) -i input -o output
}

input=''
output=''

while getopts i:o:t: opt
do
    case "$opt" in
    i) input="$OPTARG";;
    o) output="$OPTARG";;
    ?) usage
        exit 1;;
    esac
done

if [ "$input" == "" ] || [ "$output" == "" ] ; then
    echo "ERROR! input and output cannot be empty"
    usage
    exit 1
fi
shift $((OPTIND-1))

exist=$(is_hfile_exist ${output})
if [ "$exist" == "1" ]; then
    echo "$output exists, I will generate new work dir for you :)";
    output=${output}-$(seconds_of_date)
    echo "new path: ${output}"
fi

echo $output

# from run_xxx.abc to xxx
now_level_name=$(level_name $0)
temp=${now_level_name}-temp-$(seconds_of_date)

$MAHOUT recommenditembased --input $input --output $output --tempDir $temp --similarityClassname SIMILARITY_LOGLIKELIHOOD

if [ "$?" == "0" ]; then
    echo "OK in $now_level_name"
else
    echo "Error in $now_level_name"
    exit 1
fi

exit 0

