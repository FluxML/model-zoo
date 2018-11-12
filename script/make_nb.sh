scp trial.sh <user>@ip:~/
ssh <user>@ip "chmod +x ~/driver.sh"
ssh -o ConnectTimeout=10 <user>@ip "~/driver.sh"
