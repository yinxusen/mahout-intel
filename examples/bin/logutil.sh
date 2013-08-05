#!/bin/bash

# check where the pid specified in a log file is running
function is_run
{
    PID_FILE=$1
    flag=0
    if [ -e $PID_FILE ]
    then
        EPID=`cat $PID_FILE`
        if [ "`ps aux | awk '$2=='$EPID'{print "OK"}'`" == "OK" ]
        then
            flag=1
        fi
    fi
    echo $flag
}

function level_name
{
    script_name=$1
    now_level_name=`basename $script_name`
    now_level_name=${now_level_name%.*}
    now_level_name=`echo $now_level_name | sed -e's/-/_/g'` 
    now_level_name=${now_level_name##run_}
    echo $now_level_name
}

# check if the hdfs file exists?
function is_hfile_exist
{
    file=$1
    exist=0
    non_exist=`/home/spark/local/opt/hadoop/bin/hadoop fs -ls $file 1>/dev/null 2>&1; echo $?`
    if [ "$non_exist" == "0" ]
    then
        exist=1
    else
        exist=0
    fi
    echo $exist
}

seconds_of_date()  
{  
    if [ "$1" ]; then  
        date -d "$1 $2" +%s  
    else  
        date +%s  
    fi  
}  
