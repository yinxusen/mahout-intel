#!/bin/bash
# Author: 
# Func: 

set -e -x

curdir=`dirname $0`
# In production, "logutil.sh" file should be in the same working dir as current running script is.
source $curdir/logutil.sh

queue_name=${QUEUE_NAME:-'temp'}

function usage
{
    echo $(basename $0) -l local_work -h hdfs_work
}

local_work=''
hdfs_work=''
MAHOUT=/home/spark/yinxusen/mahout-distribution-0.8/bin/mahout

while getopts l:h:o: opt
do
    case "$opt" in
    l) local_work="$OPTARG";;
    h) hdfs_work="$OPTARG";;
    ?) usage
        exit 1;;
    esac
done

if [ "$local_work" == "" ] || [ "$hdfs_work" == "" ] ; then
    echo "ERROR! local_work and hdfs_work cannot be empty"
    usage
    exit 1
fi
shift $((OPTIND-1))

exist=$(is_hfile_exist ${hdfs_work})
if [ "$exist" == "1" ]; then
    echo "$hdfs_work exists, I will generate new work dir for you :)";
    #hdfs_work=${hdfs_work}-$(seconds_of_date)
    echo "new path: ${hdfs_work}"
fi

# from run_xxx.abc to xxx
now_level_name=$(level_name $0)

#MAHOUT_LOCAL=true $MAHOUT seqdirectory -i ${local_work} -o ${local_work}/weibo-out-seqdir -c UTF-8 -chunk 5
MAHOUT_LOCAL=""
HADOOP_HOME="test"

# we know weibo-out-seqdir exists on a local disk at
# this point, if we're running in clustered mode,
# copy it up to hdfs
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
    HADOOP="/home/spark/local/opt/hadoop/bin/hadoop"
    if [ ! -e $HADOOP ]; then
        echo "Can't find hadoop in $HADOOP, exiting"
        exit 1
    fi
    $HADOOP dfs -put ${local_work}/weibo-out-seqdir ${hdfs_work}/weibo-out-seqdir
fi

$MAHOUT seq2sparse \
    -i ${hdfs_work}/weibo-out-seqdir/ \
    -o ${hdfs_work}/weibo-out-seqdir-sparse  --maxDFPercent 85 --namedVector \
    -a org.apache.lucene.analysis.cn.smart.SmartChineseAnalyzer \
&& \
$MAHOUT rowid \
    -i ${hdfs_work}/weibo-out-seqdir-sparse/tf-vectors \
    -o ${hdfs_work}/weibo-out-matrix \
&& \
$MAHOUT cvb \
    -i ${hdfs_work}/weibo-out-matrix/matrix \
    -o ${hdfs_work}/weibo-lda -k 20 -ow -x 2 -a 2 \
    -dict ${hdfs_work}/weibo-out-seqdir-sparse/dictionary.file-0 \
    -dt ${hdfs_work}/weibo-dt \
    -mt ${hdfs_work}/weibo-mt

if [ "$?" == "0" ]; then
    echo "OK in $now_level_name"
else
    echo "Error in $now_level_name"
    exit 1
fi

exit 0

