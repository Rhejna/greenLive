#!/bin/bash

docker build -t greenlive .
docker run -p 5010:38856 greenlive

ssh-keygen -t ed25519 -C "vannrheez@gmail.com"

#cd /Users/vianneymfetie/Movies/greenlive_AI/otherThings
#ssh -i "greenliveKeys.cer" ec2-user@ec2-13-38-218-98.eu-west-3.compute.amazonaws.com
#scp /Users/vianneymfetie/Movies/greenlive_AI/greenlive.zip ec2-user@ec2-13-38-7-86.eu-west-3.compute.amazonaws.com
#scp /Users/vianneymfetie/Movies/greenlive_AI/greenlive.zip -i "greenliveKeys.cer" ec2-user@ec2-13-38-7-86.eu-west-3.compute.amazonaws.com:/home/ec2-user
# scp /Users/vianneymfetie/Movies/greenlive_AI/greenlive_venv.zip -i "greenliveKeys.cer" ec2-user@ec2-13-38-218-98.eu-west-3.compute.amazonaws.com:/home/ec2-user

#cd /Users/vianneymfetie/Movies/greenlive_AI/otherThings
#ssh -i "greenlive_theend.cer" ec2-user@ec2-13-37-216-19.eu-west-3.compute.amazonaws.com
#scp /Users/vianneymfetie/Movies/greenlive_AI/greenlive.zip ec2-user@ip-172-31-32-123.eu-west-3.compute.amazonaws.com:/home/ec2-user
#scp /Users/vianneymfetie/Movies/greenlive_AI/greenlive.zip -i "greenlive_theend.cer" ec2-user@ec2-13-37-216-19.eu-west-3.compute.amazonaws.com:/home/ec2-user