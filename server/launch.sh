#!/bin/bash
docker stop test
docker rm test
clear
docker pull urmilparikh/quillpad-server
echo 'pulled'
docker run -it --name test urmilparikh/quillpad-server
echo 'ran'
docker stop test
echo 'stop'
docker rm test
echo 'removed'