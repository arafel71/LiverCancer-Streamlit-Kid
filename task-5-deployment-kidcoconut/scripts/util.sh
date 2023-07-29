#!/bin/bash

#--- Note:  this file acts as a bash function library


<<blockComment 
    Name:       
    Usage:      
    PreReqs:    
blockComment



#--- declarations
<<blockComment
function utl_trace_config (aryConfigVars) {
    for each strConfig in aryConfigVars
    trace_var("kstr_defDkrHubId")
    #echo "TRACE:  kstr_defDkrHubId = ${kstr_defDkrHubId}"
    echo "TRACE:  kstr_defDkrImageName = ${kstr_defDkrImageName}"
    echo "TRACE:  kstr_defDkrTagVersion = ${kstr_defDkrTagVersion}"
    echo "TRACE:  kstr_defDkrTagStage = ${kstr_defDkrTagStage}"
    echo "TRACE:  kstr_dkrImg = ${kstr_dkrImg}"
    echo "TRACE:  kstr_dkrCtr = ${kstr_dkrCtr}"
}
blockComment


function utl_strRepeat {
    #--- Usage1:  utl_strRepeat <repeatVal> <repeatCount> <returnVar> 
    local strRptVal="${1:-null}"       #--- value to repeat
    local intRptCount="${2:-1}"        #--- repeat count; num times to repeat text
    local varReturn="${3:-output}"     #--- output var
    local strTemp                      #--- temp variable
    printf -v strTemp '%*s' "$intRptCount"
    printf -v "$varReturn" '%s' "${strTemp// /$strRptVal}"
}


function utl_valIsEmpty {
    #--- Usage1:    utl_valIsEmpty <val> | "<val>"
    #--- Usage2:    returnVal=$(utl_valIsEmpty <val> | "<val>")
    local strTestVal="${1:-null}"      #--- test value
    local __blnReturn=NULL             #--- output var
    
    #echo -e "TRACE: \t $1\n"
#    if [ ! -z "$strTestVal" -a "$strTestVal" != " " -a "$strTestVal" != "" ]; then      #--- check for empty string
    if [ ! -z "$strTestVal" ]; then
        __blnReturn=false
        if [ "$strTestVal" == " " ] || [ "$strTestVal" == "" ]; then
            __blnReturn=true            
        else  
            if [ "$strTestVal" == null ] || [ "$strTestVal" == NULL ] || [ "$strTestVal" == Null ]; then   
                    __blnReturn=true
            fi
        fi
    else
        __blnReturn=true
    fi
    echo "$__blnReturn"
}


function utl_varIsEmpty {
    local strTestVar="${1:-null}"      #--- test variable
    local __blnReturn=false            #--- output var
    
    if [ -z "$strTestVar" ]; then      #--- check for empty string
        __blnReturn=true
    fi
    echo "$__blnReturn"
}


function utl_getScriptLoc {
    local __blnReturn=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    echo "$__blnReturn"
}


function utl_logMsg {
    #--- format: <date/time> \ t <logType> <logLevel> \t <preMsg> \t <Msg> \t <postMsg> \t <varName> \t <varValue>
    local blnLogOn="${1:-true}"        #--- true: log, false:  do not log 
    local strLogType="${2:-LOG}"       #--- INFO, ERR, WARN, TRACE, TRMOD, TRFXN, EXCPT, DEBUG
    local strLogLevel="${3:-0}"        #--- Depth of log (tab indent) 
    local strMsgPrefix="${4:-null}"    #--- 
    local strMsg="${5:-null}"          #--- 
    local strMsgPostfix="${6:-null}"   #---     
    local strVarName="${7:-null}"      #--- 
    local strVarVal="${8:-null}"       #--- 

    #local blnIsEmpty=$(utl_valIsEmpty $strMsg)
    #echo -e "TRACE: blnIsEmpty=$blnIsEmpty \t $strMsg \t $(blnEmpty==false) \t $(blnEmpty=='false')"
    if $(utl_valIsEmpty $strMsgPrefix); then strMsgPrefix=""; else strMsgPrefix="$strMsgPrefix \t"; fi
    if $(utl_valIsEmpty $strMsg); then strMsg=""; else strMsg="$strMsg \t"; fi
    if $(utl_valIsEmpty $strMsgPostfix); then strMsgPostfix=""; else strMsgPostfix="$strMsgPostfix \t"; fi
    if $(utl_valIsEmpty $strVarName); then strVarName=""; else strVarName="$strVarName = "; fi
    if $(utl_valIsEmpty $strVarVal); then strVarVal=""; else strVarVal="$strVarVal"; fi

    local intTabLevel="$strLogLevel"
    #echo "TRACE (utl_logMsg):  $strLogLevel"
    utl_strRepeat "\t" $intTabLevel strTabLevel
    #echo "TRACE (utl_logMsg):  $strTabLevel"

    if $(utl_valIsEmpty $strLogLevel); then strLogLevel="\b"; strTabLevel=""; fi
    if [ "$strLogLevel" -eq "0" ]; then strLogLevel="\b"; strTabLevel=""; fi

    #if $($strLogLevel==0); then strLogLevel=""; strTabLevel=""; fi

    if [ "$blnLogOn" ]; then 
        echo -e "$strTabLevel $strLogType $strLogLevel: \t $strMsgPrefix $strMsg $strMsgPostfix $strVarName $strVarVal"
    fi
}


function utl_logTrace {
    #--- format: <date/time> \ t <logType> <logLevel> \t <preMsg> \t <Msg> \t <postMsg> \t <varName> \t <varValue>
    local blnLogOn="${1:-true}"        #--- true: log, false:  do not log 
    local strLogType="TRACE"           #--- INFO, ERR, WARN, TRACE, TRMOD, TRFXN, EXCPT, DEBUG
    local strLogLevel="${2:-0}"        #--- Depth of log (tab indent) 
    local strMsg="${3:-null}"          #--- 

    utl_logMsg $blnLogOn "$strLogType" "$strLogLevel" null "$strMsg" null null null
}

function utl_logInfo {
    #--- format: <date/time> \ t <logType> <logLevel> \t <preMsg> \t <Msg> \t <postMsg> \t <varName> \t <varValue>
    local blnLogOn="${1:-true}"        #--- true: log, false:  do not log 
    local strLogType="INFO"           #--- INFO, ERR, WARN, TRACE, TRMOD, TRFXN, EXCPT, DEBUG
    local strLogLevel="${2:-0}"        #--- Depth of log (tab indent) 
    local strMsg="${3:-null}"          #--- 

    utl_logMsg $blnLogOn "$strLogType" "$strLogLevel" null "$strMsg" null null null
}


function utl_trace_var {
    local strVarName="${1:-null}"      #--- 
    local strVarVal="${2:-null}"       #--- 
    #echo "\t(util.utl_trace_var) TRACE: \t strVarName = ${strVarName}"

    #kstr_tracePtn="TRACE:  <var> = \${<var>}"
    #str_tracePtn="${kstr_tracePtn//<var>/"$strVarName"}"          #--- bash replace all occurrences
    #echo ${str_tracePtn}
    #echo "TRACE (utl_trace_var):  $strVarName = $strVarVal"
    #utl_logMsg true "TRACE" 0 "msgPrefix" "msg" "msgPostfix" $strVarName $strVarVal
    utl_logMsg true "TRACE" 0 null null null $strVarName $strVarVal
}
