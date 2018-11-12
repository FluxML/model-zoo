scp trial.sh dhairyagandhi96@gpublr.juliacomputing.io:~/
ssh dhairyagandhi96@gpublr.juliacomputing.io "chmod +x ~/driver.sh"
ssh -o ConnectTimeout=10 dhairyagandhi96@gpublr.juliacomputing.io "~/driver.sh"
