IP=`docker inspect test | grep "IPAddress" | tail -n 1`
#IP=${IP##*:}
IP=`echo $IP | cut -d : -f 2 | xargs basename`
IP=${IP%*,}
echo $IP>server/IPAddress.txt
