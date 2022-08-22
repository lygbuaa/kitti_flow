 #!/bin/bash -e

 if [ $(whoami) != root ]; then
     echo Error: Run this script as a root user
     exit 1
 fi
  
 clkfile=/tmp/defclocks.conf
 pwrfile=/tmp/defpower.conf
  
 # Xavier
 if [ -e /sys/devices/platform/13e10000.host1x/15340000.vic ]; then
     vicctrl=/sys/devices/platform/13e10000.host1x/15340000.vic
     vicfreqctrl=$vicctrl/devfreq/15340000.vic
 # Orin?
 elif [ -e /sys/devices/platform/13e40000.host1x/15340000.vic ]; then
     vicctrl=/sys/devices/platform/13e40000.host1x/15340000.vic
     vicfreqctrl=$vicctrl/devfreq/15340000.vic
 fi

 maxclocks()
 {
     if [ ! -e $clkfile ]; then
         jetson_clocks --store $clkfile
         if [ -n "$vicctrl" ]; then
             echo "vicfreqctrl/governor:`cat $vicfreqctrl/governor`">>$clkfile
             echo "vicfreqctrl/maxfreq:`cat $vicfreqctrl/maxfreq`">>$clkfile
             echo "vicctrl/power/control:`cat $vicctrl/power/control`">>$clkfile
         fi
     fi
  
     if [ ! -e $pwrfile ]; then
         echo `nvpmodel -q|tail -n1` > $pwrfile
     fi
  
     nvpmodel -m 0
  
     jetson_clocks --fan
     jetson_clocks
  
     if [ -n "$vicctrl" ]; then
         echo on > $vicctrl/power/control
         echo userspace > $vicfreqctrl/governor
         sleep 1
         maxfreq=`cat $vicfreqctrl/available_frequencies | rev | cut -f1 -d' ' | rev`
         echo $maxfreq>$vicfreqctrl/max_freq
         echo $maxfreq>$vicfreqctrl/userspace/set_freq
     fi
 }
  
 restore()
 {
     if [ -e $clkfile ]; then
         jetson_clocks --restore $clkfile > /dev/null 2>&1
     fi
  
     if [ -e $pwrfile ]; then
         nvpmodel -m `cat pwrfile`
     fi
 }
  
 action="$1"
  
 case "$action" in
     --restore)
         restore
         ;;
     --max)
         maxclocks
         ;;
     *)
         echo "Unknown option '$action'."
         echo "Usage: (basename0) <--max|--restore>"
         exit 1
         ;;
 esac